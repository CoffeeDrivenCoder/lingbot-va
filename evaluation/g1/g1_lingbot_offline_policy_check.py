#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run LingBot-VA on one real G1 observation and validate the EEF action chunk.

This is read-only on the robot side. It captures the current head RGB frame and
arm state, calls a LingBot-VA websocket policy server, converts predicted
camera-frame hand poses into G1 base/link7 targets, and writes a JSON report.
No robot control command is sent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


G1_ROOT = Path(__file__).resolve().parent
LINGBOT_ROOT = G1_ROOT.parents[1]
DEFAULT_CFG = G1_ROOT / "cfg" / "g1_serve_bread_right.yaml"
DEFAULT_NORM_STATS = G1_ROOT / "meta" / "lingbot_action_norm_stats.json"

for path in (
    G1_ROOT,
    LINGBOT_ROOT / "evaluation" / "robotwin",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from msgpack_numpy import Packer, unpackb  # noqa: E402


EPS = 1e-9


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(f"[g1_lingbot_offline] {message}", flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), G1_ROOT, LINGBOT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (G1_ROOT / path).resolve()


def letterbox_rgb(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert BGR image to RGB and letterbox to the LingBot training size."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image HxWx3, got {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    src_h, src_w = rgb.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= EPS:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def T_from_xyzquat_xyzw(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_R(values[3:7])
    T[:3, 3] = values[:3]
    return T


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    value = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> dict[str, Any]:
    T_current = np.asarray(T_current, dtype=np.float64).reshape(4, 4)
    T_target = np.asarray(T_target, dtype=np.float64).reshape(4, 4)
    delta = T_target[:3, 3] - T_current[:3, 3]
    R_error = T_current[:3, :3].T @ T_target[:3, :3]
    return {
        "translation_delta_m": delta.tolist(),
        "translation_norm_m": float(np.linalg.norm(delta)),
        "rotation_error_deg": rotation_angle_deg(R_error),
    }


def side_q_from_arm_state(arm_values: list[float], side: str, mapping: str) -> tuple[np.ndarray, list[int]]:
    values = [float(v) for v in arm_values]
    if len(values) < 14:
        raise ValueError(f"arm_joint_states must contain at least 14 values, got {len(values)}")
    side = side.lower()
    mapping = mapping.lower()
    if mapping == "left_first":
        indices = list(range(0, 7)) if side == "left" else list(range(7, 14))
    elif mapping == "right_first":
        indices = list(range(7, 14)) if side == "left" else list(range(0, 7))
    else:
        raise ValueError(f"unknown arm state mapping {mapping!r}")
    return np.asarray([values[i] for i in indices], dtype=np.float64), indices


def connect_policy(host: str, port: int):
    import websockets.sync.client

    uri = f"ws://{host}:{port}"
    log(f"connecting LingBot websocket server: {uri}")
    ws = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        ping_interval=None,
        close_timeout=10,
    )
    metadata = unpackb(ws.recv())
    return ws, metadata


def policy_infer(ws, payload: dict[str, Any]) -> dict[str, Any]:
    packer = Packer()
    ws.send(packer.pack(payload))
    response = ws.recv()
    if isinstance(response, str):
        raise RuntimeError(f"policy server returned error string:\n{response}")
    return unpackb(response)


def load_real_g1_input(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    from G1Camera import G1HeadRGBDCamera
    from G1RobotArm import G1RobotArmReadOnly, wait_state

    cam = None
    arm = None
    try:
        log("initializing G1 camera")
        cam = G1HeadRGBDCamera(resolve_project_path(cfg["camera"]["cfg_path"]))
        log("initializing read-only G1 arm state adapter")
        arm = G1RobotArmReadOnly(side=args.side)
        log("reading one RGB-D frame")
        frame = cam.get_frame()
        log("reading current G1 state")
        state = arm.get_debug_state()
        arm_values = wait_state(arm.robot.arm_joint_states, "arm_joint_states", min_len=14)
        waist_values = wait_state(arm.robot.waist_joint_states, "waist_joint_states", min_len=2)
        state["joint_states_for_validation"] = {
            "arm": arm_values,
            "waist": waist_values,
        }
        return frame.rgb, state, {"rgb_shape": list(frame.rgb.shape), "depth_shape": list(frame.depth_m.shape)}
    finally:
        if cam is not None and args.close_camera:
            cam.close()
        if arm is not None and args.close_arm:
            arm.close()


def load_fixture_input(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read --image-path: {args.image_path}")
    if args.state_json is None:
        state = {
            "T_base_camera": np.eye(4, dtype=np.float64),
            "T_base_in_cam": np.eye(4, dtype=np.float64),
            "T_link7_in_base": np.eye(4, dtype=np.float64),
            "T_tcp_in_link7": np.eye(4, dtype=np.float64),
            "T_tcp_in_base": np.eye(4, dtype=np.float64),
            "T_tcp_in_cam": np.eye(4, dtype=np.float64),
            "gripper": 0.0,
            "joint_states_for_validation": None,
        }
    else:
        state = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    return image, state, {"fixture": True, "rgb_shape": list(image.shape)}


def flatten_action_for_execution(action: np.ndarray, skip_first_latent: bool) -> tuple[np.ndarray, list[dict[str, int]]]:
    action = np.asarray(action, dtype=np.float64)
    if action.shape[0] != 8 or action.ndim != 3:
        raise ValueError(f"expected action shape (8,F,H), got {action.shape}")
    sequence: list[np.ndarray] = []
    indices: list[dict[str, int]] = []
    start_f = 1 if skip_first_latent else 0
    for f in range(start_f, action.shape[1]):
        for h in range(action.shape[2]):
            sequence.append(action[:, f, h])
            indices.append({"latent_frame": f, "action_index": h})
    return np.stack(sequence, axis=0), indices


def validate_sequence(
    sequence: np.ndarray,
    indices: list[dict[str, int]],
    state: dict[str, Any],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from G1RobotArm import wait_state
    from g1_urdf_ik import G1UrdfKinematics

    T_base_camera = np.asarray(state["T_base_camera"], dtype=np.float64).reshape(4, 4)
    T_tcp_in_link7 = np.asarray(state["T_tcp_in_link7"], dtype=np.float64).reshape(4, 4)
    T_current_link7 = np.asarray(state["T_link7_in_base"], dtype=np.float64).reshape(4, 4)
    T_current_tcp = np.asarray(state["T_tcp_in_base"], dtype=np.float64).reshape(4, 4)
    T_align = np.asarray(cfg["robot"]["T_align"], dtype=np.float64).reshape(4, 4)
    T_align_inv = np.linalg.inv(T_align)

    norm_stats = json.loads(Path(args.norm_stats).read_text(encoding="utf-8"))
    q01 = np.asarray(norm_stats["compact_q01"], dtype=np.float64)
    q99 = np.asarray(norm_stats["compact_q99"], dtype=np.float64)

    joint_state = state.get("joint_states_for_validation")
    q_seed = None
    waist_values = None
    ik_available = False
    if joint_state and joint_state.get("arm") is not None and joint_state.get("waist") is not None:
        q_seed, arm_indices = side_q_from_arm_state(
            joint_state["arm"],
            args.side,
            args.arm_state_mapping,
        )
        waist_values = [float(v) for v in joint_state["waist"]]
        ik_available = True
    else:
        arm_indices = None

    kin = G1UrdfKinematics(args.urdf_zip) if ik_available else None
    prev_tcp = T_current_tcp
    prev_q = q_seed.copy() if q_seed is not None else None

    step_reports = []
    for step_id, (vec, idx) in enumerate(zip(sequence, indices, strict=True)):
        quat_norm = float(np.linalg.norm(vec[3:7]))
        gripper = float(vec[7])
        T_hand_cam = T_from_xyzquat_xyzw(vec[:7])
        T_tcp_base = T_base_camera @ T_hand_cam @ T_align_inv
        T_link7_base = T_tcp_base @ np.linalg.inv(T_tcp_in_link7)
        current_err = pose_error(T_current_tcp, T_tcp_base)
        prev_err = pose_error(prev_tcp, T_tcp_base)

        within_train = bool(np.all(vec >= (q01 - args.train_range_margin)) and np.all(vec <= (q99 + args.train_range_margin)))
        position_in_base = T_tcp_base[:3, 3]
        workspace_ok = bool(
            args.base_x_min <= position_in_base[0] <= args.base_x_max
            and args.base_y_min <= position_in_base[1] <= args.base_y_max
            and args.base_z_min <= position_in_base[2] <= args.base_z_max
        )
        continuity_ok = bool(prev_err["translation_norm_m"] <= args.max_step_delta_m)
        first_delta_ok = bool(current_err["translation_norm_m"] <= args.max_first_delta_m) if step_id == 0 else True
        quat_ok = bool(args.min_quat_norm <= quat_norm <= args.max_quat_norm)
        gripper_ok = bool(-args.gripper_margin <= gripper <= 1.0 + args.gripper_margin)

        ik_report = None
        ik_ok = None
        if kin is not None and prev_q is not None and waist_values is not None:
            ik = kin.solve_link7_ik(
                args.side,
                T_link7_base,
                q_init=prev_q,
                waist_states=waist_values,
                max_nfev=args.ik_max_nfev,
            )
            q_delta = ik.q_solution - prev_q
            q_delta_abs_max = float(np.max(np.abs(q_delta)))
            ik_relaxed_ok = bool(
                ik.position_error_m <= args.ik_position_tol_m
                and ik.rotation_error_deg <= args.ik_rotation_tol_deg
            )
            joint_delta_ok = bool(q_delta_abs_max <= args.max_joint_delta_rad)
            ik_ok = bool(ik_relaxed_ok and joint_delta_ok)
            ik_report = {
                **ik.to_json(),
                "ik_relaxed_ok": ik_relaxed_ok,
                "joint_delta_ok": joint_delta_ok,
                "q_delta_abs_max_rad": q_delta_abs_max,
                "q_delta_from_previous": q_delta.tolist(),
            }
            if ik_ok or args.update_ik_seed_on_failure:
                prev_q = ik.q_solution

        step_ok = bool(
            quat_ok
            and gripper_ok
            and workspace_ok
            and continuity_ok
            and first_delta_ok
            and (ik_ok is not False)
        )
        step_reports.append(
            {
                "step": step_id,
                **idx,
                "action_camera_hand": vec.tolist(),
                "T_hand_in_camera": T_hand_cam.tolist(),
                "T_tcp_in_base": T_tcp_base.tolist(),
                "T_link7_in_base": T_link7_base.tolist(),
                "tcp_position_in_base": position_in_base.tolist(),
                "quat_norm": quat_norm,
                "gripper": gripper,
                "within_training_quantile_margin": within_train,
                "workspace_ok": workspace_ok,
                "continuity_from_previous": prev_err,
                "current_to_target": current_err,
                "continuity_ok": continuity_ok,
                "first_delta_ok": first_delta_ok,
                "quat_ok": quat_ok,
                "gripper_ok": gripper_ok,
                "ik": ik_report,
                "step_ok": step_ok,
            }
        )
        prev_tcp = T_tcp_base

    hard_failures = [
        item
        for item in step_reports
        if not (
            item["quat_ok"]
            and item["gripper_ok"]
            and item["workspace_ok"]
            and item["continuity_ok"]
            and item["first_delta_ok"]
            and (item["ik"] is None or item["ik"]["ik_relaxed_ok"])
        )
    ]
    return {
        "num_candidate_steps": len(step_reports),
        "ik_available": ik_available,
        "arm_state_indices": arm_indices,
        "current": {
            "T_link7_in_base": T_current_link7.tolist(),
            "T_tcp_in_base": T_current_tcp.tolist(),
            "T_base_camera": T_base_camera.tolist(),
        },
        "thresholds": {
            "max_first_delta_m": args.max_first_delta_m,
            "max_step_delta_m": args.max_step_delta_m,
            "base_x_range": [args.base_x_min, args.base_x_max],
            "base_y_range": [args.base_y_min, args.base_y_max],
            "base_z_range": [args.base_z_min, args.base_z_max],
            "ik_position_tol_m": args.ik_position_tol_m,
            "ik_rotation_tol_deg": args.ik_rotation_tol_deg,
            "max_joint_delta_rad": args.max_joint_delta_rad,
        },
        "summary": {
            "hard_failure_count": len(hard_failures),
            "executable_candidate": len(hard_failures) == 0,
            "max_step_delta_m": max((s["continuity_from_previous"]["translation_norm_m"] for s in step_reports), default=0.0),
            "max_current_delta_m": max((s["current_to_target"]["translation_norm_m"] for s in step_reports), default=0.0),
            "max_ik_position_error_m": max(
                (s["ik"]["position_error_m"] for s in step_reports if s["ik"] is not None),
                default=None,
            ),
            "max_ik_rotation_error_deg": max(
                (s["ik"]["rotation_error_deg"] for s in step_reports if s["ik"] is not None),
                default=None,
            ),
            "steps_outside_training_range": sum(not s["within_training_quantile_margin"] for s in step_reports),
        },
        "hard_failures": hard_failures[: args.max_failures_in_report],
        "steps": step_reports,
    }


def g1_arm_joint_names(side: str) -> list[str]:
    if side == "left":
        return [f"idx{idx}_arm_l_joint{j}" for idx, j in zip(range(21, 28), range(1, 8), strict=True)]
    return [f"idx{idx}_arm_r_joint{j}" for idx, j in zip(range(61, 68), range(1, 8), strict=True)]


def write_ik_joint_outputs(run_dir: Path, validation: dict[str, Any], side: str) -> dict[str, Any]:
    steps = validation.get("steps") or []
    ik_steps = [step for step in steps if step.get("ik") is not None]
    if not ik_steps:
        return {
            "available": False,
            "reason": "IK was not available; check real arm/waist joint state reads.",
        }

    joint_names = g1_arm_joint_names(side)
    q_rows = []
    csv_path = run_dir / "ik_joint_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = (
            ["step", "latent_frame", "action_index"]
            + joint_names
            + [f"dq_{name}" for name in joint_names]
            + [
                "q_delta_abs_max_rad",
                "position_error_m",
                "rotation_error_deg",
                "min_limit_margin_rad",
                "ik_relaxed_ok",
                "joint_delta_ok",
                "step_ok",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in ik_steps:
            ik = step["ik"]
            q = np.asarray(ik["q_solution"], dtype=np.float64)
            dq = np.asarray(ik["q_delta_from_previous"], dtype=np.float64)
            q_rows.append(q)
            row = {
                "step": step["step"],
                "latent_frame": step["latent_frame"],
                "action_index": step["action_index"],
                "q_delta_abs_max_rad": ik["q_delta_abs_max_rad"],
                "position_error_m": ik["position_error_m"],
                "rotation_error_deg": ik["rotation_error_deg"],
                "min_limit_margin_rad": ik["min_limit_margin_rad"],
                "ik_relaxed_ok": ik["ik_relaxed_ok"],
                "joint_delta_ok": ik["joint_delta_ok"],
                "step_ok": step["step_ok"],
            }
            row.update({name: float(q[idx]) for idx, name in enumerate(joint_names)})
            row.update({f"dq_{name}": float(dq[idx]) for idx, name in enumerate(joint_names)})
            writer.writerow(row)

    q_arr = np.stack(q_rows, axis=0)
    max_abs_step_delta = max(float(step["ik"]["q_delta_abs_max_rad"]) for step in ik_steps)
    max_pos_err = max(float(step["ik"]["position_error_m"]) for step in ik_steps)
    max_rot_err = max(float(step["ik"]["rotation_error_deg"]) for step in ik_steps)
    min_limit_margin_values = [
        float(step["ik"]["min_limit_margin_rad"])
        for step in ik_steps
        if step["ik"].get("min_limit_margin_rad") is not None
    ]
    summary = {
        "available": True,
        "csv_path": str(csv_path),
        "side": side,
        "joint_names": joint_names,
        "num_ik_steps": len(ik_steps),
        "q_first_rad": q_arr[0].tolist(),
        "q_last_rad": q_arr[-1].tolist(),
        "q_min_rad": q_arr.min(axis=0).tolist(),
        "q_max_rad": q_arr.max(axis=0).tolist(),
        "max_abs_step_delta_rad": max_abs_step_delta,
        "max_position_error_m": max_pos_err,
        "max_rotation_error_deg": max_rot_err,
        "min_limit_margin_rad": min(min_limit_margin_values) if min_limit_margin_values else None,
        "all_ik_relaxed_ok": all(bool(step["ik"]["ik_relaxed_ok"]) for step in ik_steps),
        "all_joint_delta_ok": all(bool(step["ik"]["joint_delta_ok"]) for step in ik_steps),
        "all_step_ok": all(bool(step["step_ok"]) for step in ik_steps),
    }
    (run_dir / "joint_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=29536)
    parser.add_argument("--prompt", default="serve bread")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--out-dir", type=Path, default=G1_ROOT / "artifacts" / "lingbot_offline_policy_check")
    parser.add_argument("--tag", default="lingbot_g1_offline")
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-path", type=Path, default=None, help="Optional local image fixture instead of G1 camera.")
    parser.add_argument("--state-json", type=Path, default=None, help="Optional state fixture used with --image-path.")
    parser.add_argument("--skip-first-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-camera", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--close-arm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--train-range-margin", type=float, default=0.10)
    parser.add_argument("--arm-state-mapping", choices=["left_first", "right_first"], default="left_first")
    parser.add_argument("--urdf-zip", default=str(G1_ROOT / "G1" / "G1_URDF_Omnipicker.zip"))
    parser.add_argument("--max-first-delta-m", type=float, default=0.30)
    parser.add_argument("--max-step-delta-m", type=float, default=0.08)
    parser.add_argument("--base-x-min", type=float, default=-0.20)
    parser.add_argument("--base-x-max", type=float, default=0.90)
    parser.add_argument("--base-y-min", type=float, default=-0.80)
    parser.add_argument("--base-y-max", type=float, default=0.30)
    parser.add_argument("--base-z-min", type=float, default=-0.30)
    parser.add_argument("--base-z-max", type=float, default=1.30)
    parser.add_argument("--min-quat-norm", type=float, default=0.70)
    parser.add_argument("--max-quat-norm", type=float, default=1.30)
    parser.add_argument("--gripper-margin", type=float, default=0.05)
    parser.add_argument("--ik-max-nfev", type=int, default=160)
    parser.add_argument("--ik-position-tol-m", type=float, default=0.015)
    parser.add_argument("--ik-rotation-tol-deg", type=float, default=8.0)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.45)
    parser.add_argument("--update-ik-seed-on-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-failures-in-report", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    run_dir = args.out_dir / f"{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path is None:
        image_bgr, state, input_info = load_real_g1_input(args, cfg)
    else:
        image_bgr, state, input_info = load_fixture_input(args)

    ego_rgb = letterbox_rgb(image_bgr, args.image_width, args.image_height)
    cv2.imwrite(str(run_dir / "g1_rgb_raw_bgr.jpg"), image_bgr)
    cv2.imwrite(str(run_dir / "lingbot_ego_rgb_320x256.png"), cv2.cvtColor(ego_rgb, cv2.COLOR_RGB2BGR))
    (run_dir / "input_state.json").write_text(
        json.dumps(json_safe(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ws, metadata = connect_policy(args.server_host, args.server_port)
    try:
        reset_started = time.time()
        reset_ret = policy_infer(ws, {"reset": True, "prompt": args.prompt})
        infer_started = time.time()
        response = policy_infer(
            ws,
            {
                "obs": {
                    "observation.images.ego_rgb": ego_rgb,
                },
            },
        )
    finally:
        ws.close()

    if "action" not in response:
        raise RuntimeError(f"LingBot response has no action: {response.keys()}")
    action = np.asarray(response["action"], dtype=np.float64)
    np.save(run_dir / "raw_action.npy", action)

    sequence, indices = flatten_action_for_execution(action, args.skip_first_latent)
    validation = validate_sequence(sequence, indices, state, cfg, args)
    joint_summary = write_ik_joint_outputs(run_dir, validation, args.side)
    report = {
        "ok": bool(validation["summary"]["executable_candidate"]),
        "run_dir": str(run_dir),
        "prompt": args.prompt,
        "input": input_info,
        "server": {
            "host": args.server_host,
            "port": args.server_port,
            "metadata": json_safe(metadata),
            "reset_response": json_safe(reset_ret),
            "reset_sec": round(infer_started - reset_started, 3),
            "infer_sec": round(time.time() - infer_started, 3),
            "server_timing": json_safe(response.get("server_timing")),
        },
        "action_shape": list(action.shape),
        "skip_first_latent": bool(args.skip_first_latent),
        "joint_summary": joint_summary,
        "validation": validation,
    }
    (run_dir / "offline_policy_check_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": report["ok"],
                "action_shape": report["action_shape"],
                "summary": validation["summary"],
                "joint_summary": joint_summary,
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"wrote report: {run_dir / 'offline_policy_check_report.json'}")
    log(f"summary: {json.dumps(json_safe(validation['summary']), ensure_ascii=False)}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
