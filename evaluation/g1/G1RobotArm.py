# -*- coding: utf-8 -*-
"""Read-only G1 RobotArm adapter for HumanEgo dry-run inference.

This file intentionally refuses to move the robot. It is the bridge we need
before control: read current G1 TCP pose, compute camera-frame pose with the
validated head-camera transform, and expose the HumanEgo RobotArm interface.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from G1Geometry import fixed_T_tcp_in_link7
except ImportError:  # pragma: no cover
    from inference.G1Geometry import fixed_T_tcp_in_link7


G1_ROOT = Path(__file__).resolve().parent
LINGBOT_ROOT = G1_ROOT.parents[1]
PROJECT_ROOT = LINGBOT_ROOT
DEFAULT_PARAMETER_PY = G1_ROOT / "G1" / "parameter.py"
HEAD_YAW_RAD_ABS_LIMIT = 1.5708
HEAD_PITCH_RAD_ABS_LIMIT = 0.5233
HEAD_UNIT_LIMIT_MARGIN_RAD = 0.05


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_xyzw_to_R(q: Any) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def xyzquat_xyzw_to_T(values: Any) -> np.ndarray:
    vals = [float(v) for v in values]
    if len(vals) != 7:
        raise ValueError(f"expected xyzquat length 7, got {len(vals)}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_R(vals[3:])
    T[:3, 3] = vals[:3]
    return T


def parse_motion_pose(frame: dict) -> np.ndarray:
    p = frame["position"]
    q = frame["orientation"]["quaternion"]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_R([q["x"], q["y"], q["z"], q["w"]])
    T[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return T


def coerce_state_list(value: Any) -> list[float]:
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if value is None:
        raise ValueError("state is None")
    values = list(value)
    if not values:
        raise ValueError("state is empty")
    if any(v is None for v in values):
        raise ValueError(f"state contains None values: {values!r}")
    return [float(v) for v in values]


def normalize_angle_maybe_degrees(value: float) -> float:
    value = float(value)
    if abs(value) > 2.0 * math.pi:
        return math.radians(value)
    return value


def normalize_head_joint_states_rad(head_states: Any) -> list[float]:
    """Return G1 head yaw/pitch in radians.

    RobotDds can expose head states in degrees. The head pair must be converted
    together: a yaw like 4 deg is below 2*pi, but if pitch is 10 deg the whole
    pair is degree-valued.
    """
    values = coerce_state_list(head_states)
    if len(values) < 2:
        return values
    yaw, pitch = values[:2]
    looks_like_degrees = (
        abs(yaw) > HEAD_YAW_RAD_ABS_LIMIT + HEAD_UNIT_LIMIT_MARGIN_RAD
        or abs(pitch) > HEAD_PITCH_RAD_ABS_LIMIT + HEAD_UNIT_LIMIT_MARGIN_RAD
    )
    if looks_like_degrees:
        converted = [math.radians(v) for v in values]
        return converted
    return values


def compute_corobot_head_pitch_in_base(head_states: Any, waist_states: Any, urdf_path: str | None = None) -> Dict[str, Any]:
    from corobot.utils.kinematics import Kinematics

    resolved_urdf = urdf_path
    if not resolved_urdf:
        from corobot.utils.fk_solver import _find_urdf_solver_dir

        resolved_urdf = str((_find_urdf_solver_dir() / "A2D_viz.urdf").resolve())

    raw_head = coerce_state_list(head_states)
    head = normalize_head_joint_states_rad(raw_head)
    waist = coerce_state_list(waist_states)
    head_yaw = float(head[0])
    head_pitch = float(head[1])
    waist_pitch = normalize_angle_maybe_degrees(waist[0])
    waist_height = float(waist[1])

    with redirect_stdout(StringIO()):
        kinematics = Kinematics(str(resolved_urdf))
    xyzquat = kinematics.compute_head_fk(head_yaw, head_pitch, waist_pitch, waist_height)
    return {
        "urdf_path": str(resolved_urdf),
        "raw_head_states": raw_head,
        "raw_waist_states": waist,
        "used": {
            "head_yaw_rad": head_yaw,
            "head_pitch_rad": head_pitch,
            "waist_pitch_rad": waist_pitch,
            "waist_height_m": waist_height,
        },
        "xyzquat_xyzw": [float(v) for v in xyzquat],
        "T_head_pitch_in_base": xyzquat_xyzw_to_T(xyzquat),
    }


def wait_motion_status(controller: Any, tries: int = 30, sleep_s: float = 0.1):
    last_status = None
    for _ in range(tries):
        status = controller.get_motion_status()
        last_status = status
        if isinstance(status, dict) and status.get("frames"):
            return status
        time.sleep(sleep_s)
    return last_status


def wait_state(getter: Any, name: str, min_len: int, tries: int = 50, sleep_s: float = 0.1) -> list[float]:
    last_value = None
    last_error = None
    for _ in range(max(1, tries)):
        try:
            last_value = getter()
            values = coerce_state_list(last_value)
            if len(values) >= min_len:
                return values
            last_error = f"expected at least {min_len} values, got {len(values)}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(sleep_s)
    raise RuntimeError(f"{name} not ready after {tries} tries, last={last_value!r}, last_error={last_error}")


class G1RobotArmReadOnly:
    """G1 right/left arm state adapter. Control methods are deliberately blocked."""

    def __init__(
        self,
        side: str = "right",
        parameter_py: str | Path = DEFAULT_PARAMETER_PY,
        urdf_path: str | None = None,
        motion_tries: int = 30,
        motion_sleep_s: float = 0.1,
        state_tries: int = 50,
        state_sleep_s: float = 0.1,
    ):
        side = side.lower()
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.link7_frame_name = f"arm_{side}_link7"
        self.urdf_path = urdf_path
        self.motion_tries = int(motion_tries)
        self.motion_sleep_s = float(motion_sleep_s)
        self.state_tries = int(state_tries)
        self.state_sleep_s = float(state_sleep_s)

        from a2d_sdk.robot import RobotController, RobotDds

        self.robot = RobotDds()
        self.controller = RobotController()

        parameter_path = Path(parameter_py).expanduser()
        if not parameter_path.exists() and (PROJECT_ROOT / "scripts" / "parameters.py").exists():
            parameter_path = PROJECT_ROOT / "scripts" / "parameters.py"
        self.parameter_module = load_module(parameter_path, "g1_parameter_runtime")
        params = self.parameter_module.load_all_parameters("head")
        self.K = np.asarray(params["intrinsics"]["K"], dtype=np.float64).reshape(3, 3)
        # Verified 2026-06-22: this is T_head_pitch_camera, not final base-camera.
        self.T_head_pitch_camera = np.asarray(params["extrinsics"]["T"], dtype=np.float64).reshape(4, 4)
        self.parameter_source = {
            "intrinsics": params["intrinsics"].get("source"),
            "extrinsics": params["extrinsics"].get("source"),
        }
        self.T_tcp_in_link7 = fixed_T_tcp_in_link7(side)
        self.T_base_in_cam = np.eye(4, dtype=np.float64)
        self._last_fk: Optional[Dict[str, Any]] = None

    def get_T_base_camera(self) -> np.ndarray:
        head_states = wait_state(
            self.robot.head_joint_states,
            "head_joint_states",
            min_len=2,
            tries=self.state_tries,
            sleep_s=self.state_sleep_s,
        )
        waist_states = wait_state(
            self.robot.waist_joint_states,
            "waist_joint_states",
            min_len=2,
            tries=self.state_tries,
            sleep_s=self.state_sleep_s,
        )
        fk = compute_corobot_head_pitch_in_base(head_states, waist_states, self.urdf_path)
        self._last_fk = fk
        return fk["T_head_pitch_in_base"] @ self.T_head_pitch_camera

    def get_T_base_in_cam(self) -> np.ndarray:
        T_base_camera = self.get_T_base_camera()
        self.T_base_in_cam = np.linalg.inv(T_base_camera)
        return self.T_base_in_cam

    def get_T_link7_in_base(self) -> np.ndarray:
        motion_status = wait_motion_status(self.controller, self.motion_tries, self.motion_sleep_s)
        if not isinstance(motion_status, dict):
            raise RuntimeError(f"get_motion_status did not return a dict: {motion_status!r}")
        frames = motion_status.get("frames") or {}
        if self.link7_frame_name not in frames:
            raise RuntimeError(f"{self.link7_frame_name} not in motion_status frames: {sorted(frames.keys())}")
        return parse_motion_pose(frames[self.link7_frame_name])

    def get_T_tcp_in_base(self) -> np.ndarray:
        return self.get_T_link7_in_base() @ self.T_tcp_in_link7

    def get_T_ee_in_cam(self) -> np.ndarray:
        """Return current G1 TCP pose in camera frame."""
        T_base_in_cam = self.get_T_base_in_cam()
        return T_base_in_cam @ self.get_T_tcp_in_base()

    def get_debug_state(self) -> dict[str, Any]:
        T_base_camera = self.get_T_base_camera()
        T_base_in_cam = np.linalg.inv(T_base_camera)
        T_link7_in_base = self.get_T_link7_in_base()
        T_tcp_in_base = T_link7_in_base @ self.T_tcp_in_link7
        T_tcp_in_cam = T_base_in_cam @ T_tcp_in_base
        gripper_state = self.get_gripper_state()
        return {
            "side": self.side,
            "link7_frame_name": self.link7_frame_name,
            "parameter_source": self.parameter_source,
            "K": self.K,
            "T_head_pitch_camera": self.T_head_pitch_camera,
            "corobot_fk": self._last_fk,
            "T_base_camera": T_base_camera,
            "T_base_in_cam": T_base_in_cam,
            "T_link7_in_base": T_link7_in_base,
            "T_tcp_in_link7": self.T_tcp_in_link7,
            "T_tcp_in_base": T_tcp_in_base,
            "T_tcp_in_cam": T_tcp_in_cam,
            "gripper_state": gripper_state,
            "gripper": gripper_state["normalized_proxy"],
        }

    def get_gripper_state(self) -> dict[str, Any]:
        timestamp = None
        value = wait_state(
            self.robot.gripper_states,
            "gripper_states",
            min_len=1,
            tries=self.state_tries,
            sleep_s=self.state_sleep_s,
        )
        vals = coerce_state_list(value)
        idx = 0 if self.side == "left" else min(1, len(vals) - 1)
        raw = float(vals[idx])
        if 0.0 <= raw <= 1.0:
            normalized = raw
        else:
            normalized = float(np.clip(raw / 120.0, 0.0, 1.0))
        return {
            "raw": vals,
            "timestamp": timestamp,
            "selected_side": self.side,
            "selected_index": idx,
            "selected_raw": raw,
            "normalized_proxy": normalized,
        }

    def get_gripper(self) -> float:
        """Best-effort normalized gripper state, 0=open, 1=closed.

        Observed G1 values are around 120 in the open-ish state. Until the exact
        range is verified, return a clipped normalized proxy and include raw state.
        """
        return float(self.get_gripper_state()["normalized_proxy"])

    def move_ee_in_cam(self, T_ee_in_cam: np.ndarray, duration: float, blocking: bool = False) -> bool:
        raise RuntimeError("G1RobotArmReadOnly refuses to move. Use a control-enabled adapter after validation.")

    def set_gripper(self, value: float, blocking: bool = False) -> None:
        raise RuntimeError("G1RobotArmReadOnly refuses to move gripper. Use a control-enabled adapter after validation.")

    def go_home(self, blocking: bool = True) -> None:
        raise RuntimeError("G1RobotArmReadOnly refuses to home. Use a control-enabled adapter after validation.")

    def close(self) -> None:
        if hasattr(self, "robot") and hasattr(self.robot, "shutdown"):
            self.robot.shutdown()
        if hasattr(self, "controller"):
            for method_name in ("shutdown", "close"):
                if hasattr(self.controller, method_name):
                    getattr(self.controller, method_name)()
                    break
