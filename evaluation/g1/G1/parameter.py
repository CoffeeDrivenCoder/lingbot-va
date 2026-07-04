from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover
    Rotation = None


BASE_DIR = Path(__file__).resolve().parent
ROBOT_PARAMETERS_DIR = Path(os.environ.get("G1_PARAMETERS_DIR", "/data/parameters"))
PARAMETERS_DIR = BASE_DIR / "parameters"
PARAMETERS_ZIP = BASE_DIR / "parameters.zip"
PARAMETERS_HTTP_BASE = os.environ.get(
    "G1_PARAMETERS_HTTP_BASE",
    "http://10.42.0.101:8849/camera_parameters",
).rstrip("/")
_LAST_PARAMETER_SOURCES: dict[str, str] = {}


def load_params(filename: str, timeout: float = 2.0) -> dict[str, Any]:
    """Load a camera parameter JSON from the G1 parameter HTTP service."""
    url = f"{PARAMETERS_HTTP_BASE}/{filename}"
    try:
        with urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            _LAST_PARAMETER_SOURCES[filename] = url
            return data
    except URLError as exc:
        raise FileNotFoundError(f"cannot fetch {url}: {exc}") from exc


def _read_json_text(name: str) -> dict[str, Any]:
    for root in (ROBOT_PARAMETERS_DIR, PARAMETERS_DIR):
        path = root / name
        if path.exists():
            _LAST_PARAMETER_SOURCES[name] = str(path)
            return json.loads(path.read_text(encoding="utf-8"))

        nested_path = root / "parameters" / name
        if nested_path.exists():
            _LAST_PARAMETER_SOURCES[name] = str(nested_path)
            return json.loads(nested_path.read_text(encoding="utf-8"))

    try:
        return load_params(name)
    except FileNotFoundError:
        pass

    if PARAMETERS_ZIP.exists():
        with zipfile.ZipFile(PARAMETERS_ZIP) as zf:
            try:
                _LAST_PARAMETER_SOURCES[name] = f"{PARAMETERS_ZIP}!parameters/{name}"
                return json.loads(zf.read(f"parameters/{name}").decode("utf-8"))
            except KeyError as exc:  # pragma: no cover
                raise FileNotFoundError(f"{name} not found in {PARAMETERS_ZIP}") from exc

    raise FileNotFoundError(
        f"cannot find {name} in {ROBOT_PARAMETERS_DIR}, {PARAMETERS_DIR}, "
        f"{PARAMETERS_HTTP_BASE}, or {PARAMETERS_ZIP}"
    )


def _camera_intrinsic_matrix(camera_intrinsic: dict[str, Any]) -> np.ndarray:
    intrinsic = camera_intrinsic.get("intrinsic") or camera_intrinsic
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic.get("cx", intrinsic.get("ppx")))
    cy = float(intrinsic.get("cy", intrinsic.get("ppy")))
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_matrix_from_extrinsic(extrinsic: dict[str, Any]) -> np.ndarray:
    data = extrinsic.get("extrinsic") or extrinsic
    if "rotation_matrix" in data:
        return np.asarray(data["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    if "rotation" in data:
        if Rotation is None:
            raise ImportError("scipy is required to convert quaternion extrinsics")
        return Rotation.from_quat(data["rotation"]).as_matrix()
    raise KeyError("extrinsic must contain rotation_matrix or rotation")


def _translation_from_extrinsic(extrinsic: dict[str, Any]) -> np.ndarray:
    data = extrinsic.get("extrinsic") or extrinsic
    if "translation_vector" in data:
        return np.asarray(data["translation_vector"], dtype=np.float64).reshape(3)
    if "translation" in data:
        return np.asarray(data["translation"], dtype=np.float64).reshape(3)
    raise KeyError("extrinsic must contain translation_vector or translation")


def load_camera_intrinsics(name: str = "head") -> dict[str, Any]:
    filename = f"{name}_intrinsic_params.json"
    data = _read_json_text(filename)
    intrinsic = data.get("intrinsic") or data
    K = _camera_intrinsic_matrix(data)
    return {
        "camera_name": name,
        "source": _LAST_PARAMETER_SOURCES.get(filename),
        "raw": data,
        "intrinsic": intrinsic,
        "K": K,
        "fx": float(intrinsic["fx"]),
        "fy": float(intrinsic["fy"]),
        "cx": float(intrinsic.get("cx", intrinsic.get("ppx"))),
        "cy": float(intrinsic.get("cy", intrinsic.get("ppy"))),
        "distortion_model": intrinsic.get("distortion_model"),
        "distortion_coeffs": [intrinsic.get(k) for k in ("k1", "k2", "p1", "p2", "k3") if intrinsic.get(k) is not None],
    }


def load_camera_extrinsic(name: str = "head") -> dict[str, Any]:
    filename = f"{name}_extrinsic_params.json"
    data = _read_json_text(filename)
    extrinsic = data.get("extrinsic") or data
    R = _rotation_matrix_from_extrinsic(data)
    t = _translation_from_extrinsic(data)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return {
        "camera_name": name,
        "source": _LAST_PARAMETER_SOURCES.get(filename),
        "raw": data,
        "extrinsic": extrinsic,
        "T": T,
        "rotation_matrix": R,
        "translation_vector": t,
    }


def load_all_parameters(camera_name: str = "head") -> dict[str, Any]:
    return {
        "intrinsics": load_camera_intrinsics(camera_name),
        "extrinsics": load_camera_extrinsic(camera_name),
    }


def pixel_to_camera(u: float, v: float, depth_mm: float, K: np.ndarray) -> np.ndarray:
    z = float(depth_mm) / 1000.0
    x = (float(u) - K[0, 2]) * z / K[0, 0]
    y = (float(v) - K[1, 2]) * z / K[1, 1]
    return np.asarray([x, y, z], dtype=np.float64)


def pixel_to_base(
    u: float,
    v: float,
    depth_mm: float,
    K: np.ndarray,
    T_camera_to_base: np.ndarray,
) -> np.ndarray:
    point_camera = pixel_to_camera(u, v, depth_mm, K)
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point_camera
    return (np.asarray(T_camera_to_base, dtype=np.float64).reshape(4, 4) @ homogeneous)[:3]


head_params: dict[str, Any] | None = None
head_K: np.ndarray | None = None
head_T: np.ndarray | None = None


def load_head_defaults() -> dict[str, Any]:
    global head_params, head_K, head_T
    head_params = load_all_parameters("head")
    head_K = head_params["intrinsics"]["K"]
    head_T = head_params["extrinsics"]["T"]
    return head_params


def head_pixel_to_base(u: float, v: float, depth_mm: float, T_camera_to_base: np.ndarray | None = None) -> np.ndarray:
    global head_K, head_T
    if head_K is None or head_T is None:
        load_head_defaults()
    if T_camera_to_base is None:
        T_camera_to_base = head_T
    return pixel_to_base(u, v, depth_mm, head_K, T_camera_to_base)


def right_hand_pixel_to_base(
    u: float,
    v: float,
    depth_mm: float,
    K_hand: np.ndarray,
    T_camera_to_base: np.ndarray,
) -> np.ndarray:
    return pixel_to_base(u, v, depth_mm, K_hand, T_camera_to_base)


if __name__ == "__main__":
    params = load_head_defaults()
    print(json.dumps({
        "camera_name": "head",
        "intrinsics_source": params["intrinsics"].get("source"),
        "extrinsics_source": params["extrinsics"].get("source"),
        "K": params["intrinsics"]["K"].tolist(),
        "T": params["extrinsics"]["T"].tolist(),
        "intrinsics_raw": params["intrinsics"]["raw"],
        "extrinsics_raw": params["extrinsics"]["raw"],
    }, ensure_ascii=False, indent=2))
