# -*- coding: utf-8 -*-
"""Fixed G1 gripper geometry and HumanEgo hand-frame alignment.

The matrices here are intentionally FK-free. They only describe fixed transforms
from the G1 Omnipicker URDF and the released HumanEgo/Aria right-hand convention.
Runtime base/camera transforms still need head/waist FK.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


G1_TCP_OFFSET_M = 0.14308

# HumanEgo released checkpoints use the Aria midpoint hand frame:
#   +X: thumb/index spread direction
#   +Y: wrist/base -> fingertips
#   +Z: X cross Y
#
# This is the same fixed alignment shipped in cfg/inference/example_dualarm.yaml.
# It is used as T_hand_in_tcp in run_inference:
#   T_hand_in_cam = T_tcp_in_cam @ T_HUMANEGO_RIGHT_HAND_IN_G1_TCP
T_HUMANEGO_RIGHT_HAND_IN_G1_TCP = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def T_from_R_t(R: np.ndarray, t: Any) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def fixed_T_tcp_in_link7(side: str = "right") -> np.ndarray:
    """Return fixed URDF transform from arm_*_link7/end_link to gripper_center.

    The SDK motion frame names are `arm_right_link7` / `arm_left_link7`; the URDF
    child link is `arm_r_end_link` / `arm_l_end_link`.
    """
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    ee_yaw = math.pi / 2.0 if side == "left" else -math.pi / 2.0
    T_gripper_base_in_link7 = T_from_R_t(rpy_to_R(0.0, 0.0, ee_yaw), [0.0, 0.0, 0.0])
    T_tcp_in_gripper_base = T_from_R_t(
        rpy_to_R(0.0, 0.0, -math.pi / 2.0),
        [0.0, 0.0, G1_TCP_OFFSET_M],
    )
    return T_gripper_base_in_link7 @ T_tcp_in_gripper_base


def fixed_T_hand_in_tcp(side: str = "right", convention: str = "humanego_aria_right") -> np.ndarray:
    """Return T_align, the fixed transform from G1 TCP frame to HumanEgo hand frame.

    This is T_hand_in_tcp, used as:
        T_hand_in_cam = T_tcp_in_cam @ T_hand_in_tcp
        T_tcp_target_in_cam = T_hand_target_in_cam @ inv(T_hand_in_tcp)

    `serve_bread` is a single right-hand Aria/Ego checkpoint, so the only
    production-ready convention here is `humanego_aria_right`.
    """
    side = side.lower()
    convention = convention.lower()
    if convention in {"identity", "tcp"}:
        return np.eye(4, dtype=np.float64)
    if convention != "humanego_aria_right":
        raise ValueError(f"unsupported T_align convention: {convention!r}")
    if side != "right":
        raise ValueError("humanego_aria_right is only validated for the right hand")
    return T_HUMANEGO_RIGHT_HAND_IN_G1_TCP.copy()


def axes_summary(T_child_in_parent: np.ndarray) -> Dict[str, list[float]]:
    R = np.asarray(T_child_in_parent, dtype=np.float64).reshape(4, 4)[:3, :3]
    return {
        "+x": R[:, 0].round(9).tolist(),
        "+y": R[:, 1].round(9).tolist(),
        "+z": R[:, 2].round(9).tolist(),
    }


def geometry_summary(side: str = "right") -> Dict[str, Any]:
    T_tcp_in_link7 = fixed_T_tcp_in_link7(side)
    T_hand_in_tcp = fixed_T_hand_in_tcp(side)
    return {
        "side": side,
        "urdf_source": "G1/G1_URDF_Omnipicker.zip: G1_omnipicker_omnipicker.urdf",
        "sdk_link7_frame": f"arm_{side}_link7",
        "urdf_link7_frame": "arm_r_end_link" if side == "right" else "arm_l_end_link",
        "tcp_frame": "gripper_r_center_link" if side == "right" else "gripper_l_center_link",
        "T_tcp_in_link7": T_tcp_in_link7.tolist(),
        "tcp_axes_in_link7": axes_summary(T_tcp_in_link7),
        "T_hand_in_tcp": T_hand_in_tcp.tolist(),
        "hand_axes_in_tcp": axes_summary(T_hand_in_tcp),
        "notes": [
            "G1 TCP origin is the URDF gripper center, 0.14308 m along gripper base +Z.",
            "G1 TCP +Z is the gripper approach/tip direction.",
            "HumanEgo/Aria hand +X is spread, +Y is wrist/base-to-fingertips.",
            "Translation in T_hand_in_tcp is zero for the first version because TCP is already gripper center.",
        ],
    }
