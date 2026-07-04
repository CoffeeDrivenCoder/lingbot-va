# -*- coding: utf-8 -*-
"""G1 CosineCamera RGB-D adapter for HumanEgo inference.

This adapter assumes the head and waist are fixed during inference. Camera
parameters are loaded once in __init__(); get_frame() only fetches the current
RGB-D images and reuses the cached intrinsics.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

try:
    from interfaces import Camera, Frame
except ImportError:  # pragma: no cover
    from .interfaces import Camera, Frame


G1_ROOT = Path(__file__).resolve().parent
LINGBOT_ROOT = G1_ROOT.parents[1]


class G1HeadRGBDCamera(Camera):
    """Adapter over G1 SDK's a2d_sdk.robot.CosineCamera.

    Output contract:
        Frame.rgb     -> H x W x 3 uint8 BGR
        Frame.depth_m -> H x W float32 meters
        Frame.K       -> 3 x 3 float32 intrinsics matching Frame.rgb
    """

    def __init__(self, cam_cfg_path: str | os.PathLike[str] | Mapping[str, Any]):
        self.cfg = _load_cfg(cam_cfg_path)

        cameras_cfg = self.cfg.get("cameras", {})
        self.rgb_name = cameras_cfg.get("rgb_name", "head")
        self.depth_name = cameras_cfg.get("depth_name", "head_depth")
        self.parameter_camera_name = cameras_cfg.get("parameter_camera_name", self.rgb_name)

        sync_cfg = self.cfg.get("sync", {})
        self.sync_method = sync_cfg.get("method", "get_image_nearest")
        self.fallback_method = sync_cfg.get("fallback_method", "get_latest_image")
        self.max_delta_ns = sync_cfg.get("max_delta_ns")

        format_cfg = self.cfg.get("format", {})
        self.sdk_color_order = format_cfg.get("sdk_color_order", "RGB").upper()
        self.sdk_depth_dtype = np.dtype(format_cfg.get("sdk_depth_dtype", "uint16"))
        self.sdk_depth_unit = format_cfg.get("sdk_depth_unit", "mm").lower()

        resize_cfg = self.cfg.get("resize", {})
        self.resize_enabled = bool(resize_cfg.get("enabled", False))
        self.target_width = resize_cfg.get("target_width")
        self.target_height = resize_cfg.get("target_height")

        runtime_cfg = self.cfg.get("runtime", {})
        self.warmup_s = float(runtime_cfg.get("warmup_s", 2.0))
        self.frame_tries = int(runtime_cfg.get("frame_tries", 30))
        self.frame_sleep_s = float(runtime_cfg.get("frame_sleep_s", 0.1))

        self._warned_depth_resize = False
        self._warned_depth_sync = False

        self.camera_group = self._build_camera_group()
        if self.warmup_s > 0:
            time.sleep(self.warmup_s)
        self.camera_parameters = self._load_camera_parameters_once()
        self.K = self.camera_parameters["K"]
        self.T_parameter_extrinsic = self.camera_parameters.get("T")

    def _build_camera_group(self):
        driver_path = self.cfg.get("sdk_driver", "a2d_sdk.robot.CosineCamera")
        driver_cls = _import_symbol(driver_path)
        return driver_cls([self.rgb_name, self.depth_name])

    def _load_camera_parameters_once(self) -> dict[str, np.ndarray]:
        """Load intrinsics/extrinsics once.

        With a fixed head and waist, these values are constant during an
        episode. If either joint moves, the robot-side camera/base transform
        must be recomputed from joint states instead of relying only on this.
        """
        parameters_cfg = self.cfg.get("parameters", {})
        loader_path = parameters_cfg.get("loader_module")
        if loader_path is None:
            loader_path = G1_ROOT / "G1" / "parameter.py"
        else:
            loader_path = Path(loader_path)
            if not loader_path.is_absolute():
                for base in (Path.cwd(), G1_ROOT, LINGBOT_ROOT):
                    candidate = (base / loader_path).resolve()
                    if candidate.exists():
                        loader_path = candidate
                        break
                else:
                    loader_path = (G1_ROOT / loader_path).resolve()
        if not loader_path.exists():
            fallback_loader = G1_ROOT / "G1" / "parameter.py"
            if fallback_loader.exists():
                loader_path = fallback_loader

        parameter_module = _load_module_from_path(loader_path)
        if hasattr(parameter_module, "load_all_parameters"):
            params = parameter_module.load_all_parameters(self.parameter_camera_name)
            intr = params["intrinsics"]
            extr = params.get("extrinsics", {})
        else:
            intr = parameter_module.load_camera_intrinsics(self.parameter_camera_name)
            extr = {}

        out = {
            "K": np.asarray(intr["K"], dtype=np.float32).reshape(3, 3),
        }
        if "T" in extr:
            out["T"] = np.asarray(extr["T"], dtype=np.float32).reshape(4, 4)
        return out

    def get_frame(self) -> Frame:
        rgb, rgb_ts = self._wait_latest_image(self.rgb_name)
        if rgb is None:
            raise RuntimeError(
                f"G1 RGB frame is None after {self.frame_tries} tries: {self.rgb_name}. "
                "Check camera stream/topic and SDK camera name."
            )

        depth_raw, depth_ts = self._get_depth_near_rgb(rgb_ts)
        if depth_raw is None:
            raise RuntimeError(
                f"G1 depth frame is None after {self.frame_tries} tries: {self.depth_name}. "
                "Check depth_enable/head_depth stream."
            )

        rgb_bgr = self._to_bgr(rgb)
        depth_m = self._decode_depth_m(depth_raw, rgb_bgr.shape[:2])
        K = self.K.copy()

        if self.max_delta_ns is not None and rgb_ts is not None and depth_ts is not None:
            delta_ns = abs(int(depth_ts) - int(rgb_ts))
            if delta_ns > int(self.max_delta_ns) and not self._warned_depth_sync:
                print(
                    f"[G1HeadRGBDCamera] warning: RGB/depth timestamp delta is "
                    f"{delta_ns} ns, threshold is {self.max_delta_ns} ns"
                )
                self._warned_depth_sync = True

        if self.resize_enabled:
            rgb_bgr, depth_m, K = self._resize_rgbd_and_K(rgb_bgr, depth_m, K)

        return Frame(rgb=rgb_bgr, depth_m=depth_m, K=K)

    def _wait_latest_image(self, camera_name: str):
        last_img = None
        last_ts = None
        for _ in range(max(1, self.frame_tries)):
            img, ts = self.camera_group.get_latest_image(camera_name)
            last_img, last_ts = img, ts
            if img is not None:
                return img, ts
            time.sleep(self.frame_sleep_s)
        return last_img, last_ts

    def _get_depth_near_rgb(self, rgb_ts):
        if (
            rgb_ts is not None
            and self.sync_method == "get_image_nearest"
            and hasattr(self.camera_group, "get_image_nearest")
        ):
            try:
                depth_raw, depth_ts = self.camera_group.get_image_nearest(self.depth_name, int(rgb_ts))
                if depth_raw is not None:
                    return depth_raw, depth_ts
            except Exception as exc:
                print(f"[G1HeadRGBDCamera] warning: get_image_nearest failed: {exc}")

        if self.fallback_method == "get_latest_image":
            return self._wait_latest_image(self.depth_name)

        fallback = getattr(self.camera_group, self.fallback_method, None)
        if fallback is not None:
            return fallback(self.depth_name)
        return self._wait_latest_image(self.depth_name)

    def _to_bgr(self, image: Any) -> np.ndarray:
        if isinstance(image, (bytes, bytearray, memoryview)):
            decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"failed to decode encoded RGB image: {self.rgb_name}")
            return decoded

        arr = np.asarray(image)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise RuntimeError(f"unexpected RGB image shape from {self.rgb_name}: {arr.shape}")

        if self.sdk_color_order == "RGB":
            code = cv2.COLOR_RGB2BGR if arr.shape[2] == 3 else cv2.COLOR_RGBA2BGR
            return cv2.cvtColor(arr, code)
        if self.sdk_color_order == "BGR":
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            return np.ascontiguousarray(arr)
        raise ValueError(f"unsupported sdk_color_order: {self.sdk_color_order}")

    def _decode_depth_m(self, depth_raw: Any, rgb_hw: tuple[int, int]) -> np.ndarray:
        depth = self._depth_to_array(depth_raw, rgb_hw)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise RuntimeError(f"unexpected depth image shape from {self.depth_name}: {depth.shape}")

        if depth.shape != rgb_hw:
            if not self._warned_depth_resize:
                print(
                    f"[G1HeadRGBDCamera] warning: depth shape {depth.shape} does not "
                    f"match RGB shape {rgb_hw}; resizing depth with nearest neighbor. "
                    "Verify on the real robot that RGB-D is aligned."
                )
                self._warned_depth_resize = True
            depth = cv2.resize(
                depth,
                (rgb_hw[1], rgb_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        depth = depth.astype(np.float32, copy=False)
        if self.sdk_depth_unit == "mm":
            return depth / 1000.0
        if self.sdk_depth_unit == "m":
            return depth
        raise ValueError(f"unsupported sdk_depth_unit: {self.sdk_depth_unit}")

    def _depth_to_array(self, depth_raw: Any, rgb_hw: tuple[int, int]) -> np.ndarray:
        if isinstance(depth_raw, (bytes, bytearray, memoryview)):
            depth_vec = np.frombuffer(depth_raw, dtype=self.sdk_depth_dtype)
            return self._reshape_depth_vector(depth_vec, rgb_hw)

        depth = np.asarray(depth_raw)
        if depth.ndim == 1:
            depth = depth.astype(self.sdk_depth_dtype, copy=False)
            return self._reshape_depth_vector(depth, rgb_hw)
        return depth

    def _reshape_depth_vector(self, depth_vec: np.ndarray, rgb_hw: tuple[int, int]) -> np.ndarray:
        candidates: list[tuple[int, int]] = []
        shape = self._get_sdk_image_shape(self.depth_name)
        if shape is not None:
            width, height = shape
            candidates.extend([(height, width), (width, height)])
        candidates.append(rgb_hw)

        seen = set()
        for h, w in candidates:
            h, w = int(h), int(w)
            if (h, w) in seen:
                continue
            seen.add((h, w))
            if depth_vec.size == h * w:
                return depth_vec.reshape(h, w)

        raise RuntimeError(
            f"unexpected depth buffer size from {self.depth_name}: {depth_vec.size}; "
            f"tried shapes {candidates}"
        )

    def _get_sdk_image_shape(self, camera_name: str) -> tuple[int, int] | None:
        get_shape = getattr(self.camera_group, "get_image_shape", None)
        if get_shape is None:
            return None
        try:
            shape = get_shape(camera_name)
        except Exception as exc:
            print(f"[G1HeadRGBDCamera] warning: get_image_shape({camera_name}) failed: {exc}")
            return None
        if shape is None or len(shape) < 2:
            return None
        return int(shape[0]), int(shape[1])

    def _resize_rgbd_and_K(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.target_width is None or self.target_height is None:
            raise ValueError("resize.enabled=true requires target_width and target_height")

        src_h, src_w = rgb_bgr.shape[:2]
        dst_w, dst_h = int(self.target_width), int(self.target_height)
        if (src_w, src_h) == (dst_w, dst_h):
            return rgb_bgr, depth_m, K

        scale_x = dst_w / float(src_w)
        scale_y = dst_h / float(src_h)
        K = K.copy()
        K[0, 0] *= scale_x
        K[0, 2] *= scale_x
        K[1, 1] *= scale_y
        K[1, 2] *= scale_y

        rgb_bgr = cv2.resize(rgb_bgr, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        depth_m = cv2.resize(depth_m, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
        return rgb_bgr, depth_m, K

    def close(self) -> None:
        for method_name in ("close", "shutdown", "stop"):
            method = getattr(self.camera_group, method_name, None)
            if callable(method):
                method()
                return


def _load_cfg(cam_cfg_path: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cam_cfg_path, Mapping):
        return dict(cam_cfg_path)
    with open(cam_cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _import_symbol(dotted_path: str):
    module_name, sep, symbol_name = dotted_path.rpartition(".")
    if not sep:
        raise ValueError(f"expected a dotted import path, got: {dotted_path}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"G1HeadRGBDCamera requires `{dotted_path}`. Install or run inside "
            "the G1/a2d SDK environment before using this camera adapter."
        ) from exc
    return getattr(module, symbol_name)


def _load_module_from_path(path: str | os.PathLike[str]) -> ModuleType:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("g1_parameter_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load G1 parameter module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
