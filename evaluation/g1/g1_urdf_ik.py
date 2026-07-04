#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight G1 URDF FK/IK utilities for left/right 7-DoF arms."""

from __future__ import annotations

import math
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


G1_ROOT = Path(__file__).resolve().parent
LINGBOT_ROOT = G1_ROOT.parents[1]
PROJECT_ROOT = LINGBOT_ROOT
DEFAULT_G1_ZIP = G1_ROOT / "G1" / "G1_URDF_Omnipicker.zip"
DEFAULT_URDF_IN_ZIP = "G1_URDF_Omnipicker/urdf/G1/G1_omnipicker_omnipicker.urdf"
EPS = 1e-12


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


@dataclass
class IkResult:
    side: str
    success: bool
    q_init: np.ndarray
    q_solution: np.ndarray
    position_error_m: float
    rotation_error_deg: float
    dq_norm: float
    min_limit_margin_rad: float | None
    cost: float
    message: str
    num_function_evals: int

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "success": bool(self.success),
            "q_init": self.q_init.tolist(),
            "q_solution": self.q_solution.tolist(),
            "dq": (self.q_solution - self.q_init).tolist(),
            "dq_norm": float(self.dq_norm),
            "position_error_m": float(self.position_error_m),
            "rotation_error_deg": float(self.rotation_error_deg),
            "min_limit_margin_rad": self.min_limit_margin_rad,
            "cost": float(self.cost),
            "message": self.message,
            "num_function_evals": int(self.num_function_evals),
        }


def parse_vec(text: str | None, default: list[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def T_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_from_rpy_xyz(rpy: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return T_from_R_t(rpy_to_R(float(rpy[0]), float(rpy[1]), float(rpy[2])), xyz)


def axis_angle_T(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= EPS:
        return np.eye(4, dtype=np.float64)
    R = Rotation.from_rotvec(axis / norm * float(angle)).as_matrix()
    return T_from_R_t(R, np.zeros(3, dtype=np.float64))


def axis_translation_T(axis: np.ndarray, distance: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    T = np.eye(4, dtype=np.float64)
    if norm > EPS:
        T[:3, 3] = axis / norm * float(distance)
    return T


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    R_delta = np.asarray(R_delta, dtype=np.float64).reshape(3, 3)
    value = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> dict[str, Any]:
    T_current = np.asarray(T_current, dtype=np.float64).reshape(4, 4)
    T_target = np.asarray(T_target, dtype=np.float64).reshape(4, 4)
    position_error = T_current[:3, 3] - T_target[:3, 3]
    R_error = T_current[:3, :3].T @ T_target[:3, :3]
    return {
        "position_error_m": float(np.linalg.norm(position_error)),
        "position_error_vector_m": position_error.tolist(),
        "rotation_error_deg": rotation_angle_deg(R_error),
        "rotation_error_rotvec": Rotation.from_matrix(R_error).as_rotvec().tolist(),
    }


def normalize_angle_maybe_degrees(value: float) -> float:
    value = float(value)
    if abs(value) > 2.0 * math.pi:
        return math.radians(value)
    return value


def normalize_waist_states(waist_states: list[float] | tuple[float, ...] | None) -> dict[str, float]:
    """Return URDF body joint values from SDK waist states.

    Existing G1 code treats waist states as [waist_pitch, waist_height].
    URDF order is body_joint1=height, body_joint2=pitch.
    """
    if waist_states is None or len(waist_states) < 2:
        return {"idx01_body_joint1": 0.0, "idx02_body_joint2": 0.0}
    waist_pitch = normalize_angle_maybe_degrees(float(waist_states[0]))
    waist_height = float(waist_states[1])
    return {"idx01_body_joint1": waist_height, "idx02_body_joint2": waist_pitch}


class G1UrdfKinematics:
    def __init__(self, urdf_zip: str | Path = DEFAULT_G1_ZIP, urdf_in_zip: str = DEFAULT_URDF_IN_ZIP):
        self.urdf_zip = Path(urdf_zip).expanduser().resolve()
        self.urdf_in_zip = urdf_in_zip
        self.robot_name, self.joints_by_child, self.joints_by_name = self._load_urdf()
        self.arm_joint_names = {
            "left": [f"idx{idx}_arm_l_joint{j}" for idx, j in zip(range(21, 28), range(1, 8))],
            "right": [f"idx{idx}_arm_r_joint{j}" for idx, j in zip(range(61, 68), range(1, 8))],
        }
        self.target_links = {
            "left": "arm_l_end_link",
            "right": "arm_r_end_link",
        }
        self.sdk_frame_names = {
            "left": "arm_left_link7",
            "right": "arm_right_link7",
        }

    def _load_urdf(self) -> tuple[str, dict[str, JointInfo], dict[str, JointInfo]]:
        with zipfile.ZipFile(self.urdf_zip) as zf:
            text = zf.read(self.urdf_in_zip).decode("utf-8")
        root = ET.fromstring(text)
        by_child: dict[str, JointInfo] = {}
        by_name: dict[str, JointInfo] = {}
        for joint_el in root.findall("joint"):
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            if parent_el is None or child_el is None:
                continue
            origin_el = joint_el.find("origin")
            axis_el = joint_el.find("axis")
            limit_el = joint_el.find("limit")
            info = JointInfo(
                name=joint_el.attrib["name"],
                joint_type=joint_el.attrib.get("type", "fixed"),
                parent=parent_el.attrib["link"],
                child=child_el.attrib["link"],
                origin_xyz=parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, [0.0, 0.0, 0.0]),
                origin_rpy=parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, [0.0, 0.0, 0.0]),
                axis=parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, [0.0, 0.0, 1.0]),
                lower=float(limit_el.attrib["lower"]) if limit_el is not None and "lower" in limit_el.attrib else None,
                upper=float(limit_el.attrib["upper"]) if limit_el is not None and "upper" in limit_el.attrib else None,
            )
            by_child[info.child] = info
            by_name[info.name] = info
        return root.attrib.get("name", ""), by_child, by_name

    def chain_to(self, target_link: str, root_link: str = "base_link") -> list[JointInfo]:
        chain: list[JointInfo] = []
        current = target_link
        while current != root_link:
            joint = self.joints_by_child.get(current)
            if joint is None:
                raise KeyError(f"no parent joint for link {current!r}; cannot reach {root_link!r}")
            chain.append(joint)
            current = joint.parent
        return list(reversed(chain))

    def side_from_name(self, side: str) -> str:
        side = side.lower()
        if side in {"l", "left"}:
            return "left"
        if side in {"r", "right"}:
            return "right"
        raise ValueError(f"side must be left/right, got {side!r}")

    def joint_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        side = self.side_from_name(side)
        lows = []
        highs = []
        for name in self.arm_joint_names[side]:
            joint = self.joints_by_name[name]
            lows.append(-math.pi if joint.lower is None else joint.lower)
            highs.append(math.pi if joint.upper is None else joint.upper)
        return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)

    def home_q(self, side: str) -> np.ndarray:
        lows, highs = self.joint_limits(side)
        # A mild bent-elbow seed, clipped per side because joint2 limits differ.
        seed = np.asarray([0.0, 0.25, 0.0, -0.65, 0.0, 0.45, 0.0], dtype=np.float64)
        if self.side_from_name(side) == "left":
            seed[1] = -0.25
        return np.clip(seed, lows, highs)

    def side_joint_values(self, side: str, q: np.ndarray) -> dict[str, float]:
        side = self.side_from_name(side)
        q = np.asarray(q, dtype=np.float64).reshape(7)
        return {name: float(q[idx]) for idx, name in enumerate(self.arm_joint_names[side])}

    def fk(
        self,
        side: str,
        q: np.ndarray,
        waist_states: list[float] | tuple[float, ...] | None = None,
        target_link: str | None = None,
        extra_joint_values: dict[str, float] | None = None,
    ) -> np.ndarray:
        side = self.side_from_name(side)
        target_link = target_link or self.target_links[side]
        values = normalize_waist_states(waist_states)
        values.update(self.side_joint_values(side, q))
        if extra_joint_values:
            values.update({str(k): float(v) for k, v in extra_joint_values.items()})
        T = np.eye(4, dtype=np.float64)
        for joint in self.chain_to(target_link):
            T = T @ T_from_rpy_xyz(joint.origin_rpy, joint.origin_xyz)
            value = float(values.get(joint.name, 0.0))
            if joint.joint_type in {"revolute", "continuous"}:
                T = T @ axis_angle_T(joint.axis, value)
            elif joint.joint_type == "prismatic":
                T = T @ axis_translation_T(joint.axis, value)
            elif joint.joint_type == "fixed":
                pass
            else:
                raise ValueError(f"unsupported joint type {joint.joint_type!r} for {joint.name}")
        return T

    def link7_fk(self, side: str, q: np.ndarray, waist_states: list[float] | tuple[float, ...] | None = None) -> np.ndarray:
        side = self.side_from_name(side)
        return self.fk(side, q, waist_states=waist_states, target_link=self.target_links[side])

    def tcp_fk(self, side: str, q: np.ndarray, waist_states: list[float] | tuple[float, ...] | None = None) -> np.ndarray:
        side = self.side_from_name(side)
        target = "gripper_l_center_link" if side == "left" else "gripper_r_center_link"
        return self.fk(side, q, waist_states=waist_states, target_link=target)

    def solve_link7_ik(
        self,
        side: str,
        target_T_link7_in_base: np.ndarray,
        q_init: np.ndarray | None = None,
        waist_states: list[float] | tuple[float, ...] | None = None,
        max_nfev: int = 250,
        position_weight: float = 20.0,
        rotation_weight: float = 1.0,
        smooth_weight: float = 0.035,
        home_weight: float = 0.005,
    ) -> IkResult:
        side = self.side_from_name(side)
        target_T = np.asarray(target_T_link7_in_base, dtype=np.float64).reshape(4, 4)
        lows, highs = self.joint_limits(side)
        if q_init is None:
            q_init = self.home_q(side)
        q_init = np.clip(np.asarray(q_init, dtype=np.float64).reshape(7), lows, highs)
        home = self.home_q(side)

        def residual(q: np.ndarray) -> np.ndarray:
            T_cur = self.link7_fk(side, q, waist_states=waist_states)
            pos_err = T_cur[:3, 3] - target_T[:3, 3]
            rot_err = Rotation.from_matrix(T_cur[:3, :3].T @ target_T[:3, :3]).as_rotvec()
            smooth = q - q_init
            home_err = q - home
            return np.concatenate(
                [
                    float(position_weight) * pos_err,
                    float(rotation_weight) * rot_err,
                    float(smooth_weight) * smooth,
                    float(home_weight) * home_err,
                ]
            )

        result = least_squares(
            residual,
            q_init,
            bounds=(lows, highs),
            max_nfev=int(max_nfev),
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
        )
        q_sol = np.asarray(result.x, dtype=np.float64)
        err = pose_error(self.link7_fk(side, q_sol, waist_states=waist_states), target_T)
        margins = np.minimum(q_sol - lows, highs - q_sol)
        min_margin = float(np.min(margins)) if len(margins) else None
        success = bool(result.success and err["position_error_m"] <= 0.005 and err["rotation_error_deg"] <= 2.0)
        return IkResult(
            side=side,
            success=success,
            q_init=q_init,
            q_solution=q_sol,
            position_error_m=float(err["position_error_m"]),
            rotation_error_deg=float(err["rotation_error_deg"]),
            dq_norm=float(np.linalg.norm(q_sol - q_init)),
            min_limit_margin_rad=min_margin,
            cost=float(result.cost),
            message=str(result.message),
            num_function_evals=int(result.nfev),
        )

    def describe_side(self, side: str) -> dict[str, Any]:
        side = self.side_from_name(side)
        lows, highs = self.joint_limits(side)
        return {
            "side": side,
            "sdk_frame_name": self.sdk_frame_names[side],
            "target_link": self.target_links[side],
            "arm_joint_names": self.arm_joint_names[side],
            "joint_lower": lows.tolist(),
            "joint_upper": highs.tolist(),
            "chain": [
                {
                    "name": joint.name,
                    "type": joint.joint_type,
                    "parent": joint.parent,
                    "child": joint.child,
                    "origin_xyz": joint.origin_xyz.tolist(),
                    "origin_rpy": joint.origin_rpy.tolist(),
                    "axis": joint.axis.tolist(),
                    "lower": joint.lower,
                    "upper": joint.upper,
                }
                for joint in self.chain_to(self.target_links[side])
            ],
        }


def extract_urdf_to_temp(urdf_zip: str | Path = DEFAULT_G1_ZIP) -> Path:
    root = Path(tempfile.mkdtemp(prefix="g1_urdf_"))
    with zipfile.ZipFile(Path(urdf_zip).expanduser().resolve()) as zf:
        zf.extractall(root)
    return root
