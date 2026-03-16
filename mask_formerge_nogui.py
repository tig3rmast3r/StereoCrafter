#!/usr/bin/env python3
"""
Build processed merge masks from replace-mask videos only (headless, monothread).

This isolates the mask pipeline used in merging:
1) grayscale conversion
2) optional binarize
3) optional dilate
4) optional blur
5) optional shadow expansion with temporal state across chunks

Outputs are written in a dedicated folder (default: ./work/mask_formerge) as
lossless grayscale FFV1 MKV files.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu  # type: ignore

LOG = logging.getLogger("mask_formerge")

# Shadow-motion tuning defaults (mask_for_merge only).
# Kept here on purpose for quick manual tuning before GUI integration.
SHADOW_MOTION_CHAIN_ENABLED_DEFAULT = True
SHADOW_ALPHA_UP_DEFAULT = 0.45
SHADOW_ALPHA_DOWN_DEFAULT = 0.45
SHADOW_MAX_DELTA_UP_FRAC_DEFAULT = 0.35
SHADOW_MAX_DELTA_DOWN_FRAC_DEFAULT = 0.20
SHADOW_MAX_LEN_CAP_MULT_DEFAULT = 4.0
SHADOW_MOTION_MAX_PX_DEFAULT = 40.0
SHADOW_AREA_MIN_PX_DEFAULT = 0.0
SHADOW_AREA_MAX_PX_DEFAULT = 0.0
SHADOW_AREA_RESET_RATIO_DEFAULT = 1.8
SHADOW_AREA_RESET_ABS_PX_DEFAULT = 0.0
SHADOW_COMPONENT_MERGE_Y_TOL_PX_DEFAULT = 0
SHADOW_MOTION_GAIN_DEFAULT = 1.0
SHADOW_MOTION_DEADZONE_PX_DEFAULT = 4.0
MOTION_DEFAULTS_FILENAME = "config_mask_formerge_nogui_motion_defaults.json"
MOTION_DEFAULTS_ENV = "MASK_FORMERGE_MOTION_DEFAULTS_JSON"


def _limit_native_threads() -> None:
    # Keep each worker process single-threaded at the native level; the shell
    # launcher already runs multiple worker processes in parallel.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


_limit_native_threads()


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(fallback)


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(fallback)


def _load_motion_defaults_from_json() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "shadow_motion_gain": SHADOW_MOTION_GAIN_DEFAULT,
        "shadow_motion_deadzone_px": SHADOW_MOTION_DEADZONE_PX_DEFAULT,
        "shadow_motion_max_px": SHADOW_MOTION_MAX_PX_DEFAULT,
        "shadow_area_min_px": SHADOW_AREA_MIN_PX_DEFAULT,
        "shadow_area_max_px": SHADOW_AREA_MAX_PX_DEFAULT,
        "shadow_area_reset_ratio": SHADOW_AREA_RESET_RATIO_DEFAULT,
        "shadow_area_reset_abs_px": SHADOW_AREA_RESET_ABS_PX_DEFAULT,
        "shadow_component_merge_y_tol_px": SHADOW_COMPONENT_MERGE_Y_TOL_PX_DEFAULT,
        "shadow_alpha_down": SHADOW_ALPHA_DOWN_DEFAULT,
        "shadow_motion_chain_enabled": SHADOW_MOTION_CHAIN_ENABLED_DEFAULT,
    }
    cfg_raw = str(os.environ.get(MOTION_DEFAULTS_ENV, MOTION_DEFAULTS_FILENAME)).strip()
    if os.path.isabs(cfg_raw):
        cfg_path = cfg_raw
    else:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_raw)
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.isfile(cfg_path):
        return defaults
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return defaults
    if not isinstance(data, dict):
        return defaults

    defaults["shadow_motion_gain"] = max(
        0.0,
        _coerce_float(data.get("shadow_motion_gain"), defaults["shadow_motion_gain"]),
    )
    defaults["shadow_motion_deadzone_px"] = max(
        0.0,
        _coerce_float(
            data.get("shadow_motion_deadzone_px"),
            defaults["shadow_motion_deadzone_px"],
        ),
    )
    defaults["shadow_motion_max_px"] = max(
        defaults["shadow_motion_deadzone_px"],
        _coerce_float(data.get("shadow_motion_max_px"), defaults["shadow_motion_max_px"]),
    )
    defaults["shadow_area_min_px"] = max(
        0.0,
        _coerce_float(data.get("shadow_area_min_px"), defaults["shadow_area_min_px"]),
    )
    defaults["shadow_area_max_px"] = max(
        defaults["shadow_area_min_px"],
        _coerce_float(data.get("shadow_area_max_px"), defaults["shadow_area_max_px"]),
    )
    if "shadow_area_reset_ratio" in data:
        defaults["shadow_area_reset_ratio"] = max(
            1.0,
            _coerce_float(
                data.get("shadow_area_reset_ratio"),
                defaults["shadow_area_reset_ratio"],
            ),
        )
    elif "shadow_area_reset_pct" in data:
        pct = max(0.0, _coerce_float(data.get("shadow_area_reset_pct"), 0.0))
        defaults["shadow_area_reset_ratio"] = 1.0 + (pct / 100.0)
    defaults["shadow_area_reset_abs_px"] = max(
        0.0,
        _coerce_float(
            data.get("shadow_area_reset_abs_px"),
            defaults["shadow_area_reset_abs_px"],
        ),
    )
    defaults["shadow_component_merge_y_tol_px"] = max(
        0,
        _coerce_int(
            data.get("shadow_component_merge_y_tol_px"),
            defaults["shadow_component_merge_y_tol_px"],
        ),
    )
    defaults["shadow_alpha_down"] = min(
        1.0,
        max(
            0.0,
            _coerce_float(data.get("shadow_alpha_down"), defaults["shadow_alpha_down"]),
        ),
    )
    defaults["shadow_motion_chain_enabled"] = _coerce_bool(
        data.get("shadow_motion_chain_enabled"),
        bool(defaults["shadow_motion_chain_enabled"]),
    )
    return defaults


def _shadow_curve_opacity(u: float, curve: float) -> float:
    u = float(max(0.0, min(1.0, u)))
    c = float(max(-1.0, min(1.0, curve)))
    linear = 1.0 - u
    if abs(c) <= 1e-6:
        return linear
    if c > 0.0:
        bulged = (1.0 - u) ** 0.5
        return (1.0 - c) * linear + c * bulged
    t = -c
    front_drop = (1.0 - u) ** 2.0
    return (1.0 - t) * linear + t * front_drop


def _bbox_intersection_area(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0
    return int((ix2 - ix1 + 1) * (iy2 - iy1 + 1))


def _ramp01(value: float, lo: float, hi: float) -> float:
    v = float(value)
    a = float(lo)
    b = float(hi)
    if b <= a + 1e-6:
        return 1.0 if v >= b else 0.0
    return float(max(0.0, min(1.0, (v - a) / (b - a))))


def apply_shadow_blur(
    mask: torch.Tensor,
    base_length_px: int,
    curve: float,
    motion_gain: float,
    motion_deadzone_px: float = 4.0,
    motion_max_px: float = SHADOW_MOTION_MAX_PX_DEFAULT,
    motion_chain_enabled: bool = SHADOW_MOTION_CHAIN_ENABLED_DEFAULT,
    area_min_px: float = SHADOW_AREA_MIN_PX_DEFAULT,
    area_max_px: float = SHADOW_AREA_MAX_PX_DEFAULT,
    area_reset_ratio: float = SHADOW_AREA_RESET_RATIO_DEFAULT,
    area_reset_abs_px: float = SHADOW_AREA_RESET_ABS_PX_DEFAULT,
    component_merge_y_tol_px: int = SHADOW_COMPONENT_MERGE_Y_TOL_PX_DEFAULT,
    alpha_down: float = SHADOW_ALPHA_DOWN_DEFAULT,
    width_adaptive: bool = True,
    use_gpu: bool = True,
    state: Optional[Dict[str, Any]] = None,
    border_tolerance_px: int = 2,
    width_ref_px: float = 20.0,
    width_power: float = 1.0,
) -> torch.Tensor:
    if int(base_length_px) <= 0:
        return mask

    component_thresh = 0.05
    base_len = float(max(0.0, base_length_px))
    curve = float(max(-1.0, min(1.0, curve)))
    motion_gain = float(max(0.0, motion_gain))
    width_adaptive = bool(width_adaptive)
    width_ref_px = float(max(1.0, width_ref_px))
    width_power = float(max(0.1, width_power))
    border_tol = max(1, int(border_tolerance_px))
    motion_chain_enabled = bool(motion_chain_enabled)
    motion_deadzone_px = float(max(0.0, motion_deadzone_px))
    motion_max_px = float(max(motion_deadzone_px, motion_max_px))
    area_min_px = float(max(0.0, area_min_px))
    area_max_px = float(max(area_min_px, area_max_px))
    area_reset_ratio = float(max(1.0, area_reset_ratio))
    area_reset_abs_px = float(max(0.0, area_reset_abs_px))
    component_merge_y_tol_px = int(max(0, component_merge_y_tol_px))

    alpha_up = SHADOW_ALPHA_UP_DEFAULT
    alpha_down = float(max(0.0, min(1.0, alpha_down)))
    max_delta_up = max(1.0, SHADOW_MAX_DELTA_UP_FRAC_DEFAULT * base_len)
    max_delta_down = max(1.0, SHADOW_MAX_DELTA_DOWN_FRAC_DEFAULT * base_len)
    max_len_cap = int(max(100, SHADOW_MAX_LEN_CAP_MULT_DEFAULT * base_len))

    mask_cpu = mask.detach().to(device="cpu", dtype=torch.float32).numpy()
    t_count, _c, height, width = mask_cpu.shape
    right_touch_start = width - border_tol

    prev_components: List[Dict[str, Any]] = []
    if motion_chain_enabled and state is not None:
        prev_components = list(state.get("prev_components", []) or [])

    out_np = np.empty_like(mask_cpu, dtype=np.float32)

    for t in range(t_count):
        frame = np.asarray(mask_cpu[t, 0], dtype=np.float32)
        frame_bin = (frame > component_thresh).astype(np.uint8)
        canvas = frame.copy()

        if frame_bin.any():
            frame_cc = frame_bin
            if component_merge_y_tol_px > 0:
                # Merge small vertical gaps before CC to avoid noisy split into tiny blobs.
                k_h = int(2 * component_merge_y_tol_px + 1)
                k_h = max(1, k_h)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
                frame_cc = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                frame_cc, connectivity=8
            )

            curr_components: List[Dict[str, Any]] = []
            comp_len_by_label: Dict[int, float] = {}
            curr_candidates: List[Dict[str, Any]] = []

            for lab in range(1, int(n_labels)):
                x = int(stats[lab, cv2.CC_STAT_LEFT])
                y = int(stats[lab, cv2.CC_STAT_TOP])
                w = int(stats[lab, cv2.CC_STAT_WIDTH])
                h = int(stats[lab, cv2.CC_STAT_HEIGHT])
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area <= 0 or w <= 0 or h <= 0:
                    continue
                bbox = (x, y, x + w - 1, y + h - 1)
                cx = float(centroids[lab][0])
                cy = float(centroids[lab][1])
                curr_candidates.append(
                    {
                        "label": int(lab),
                        "bbox": bbox,
                        "centroid": (cx, cy),
                        "area": int(area),
                    }
                )

            if motion_chain_enabled:
                n_curr = len(curr_candidates)
                assigned_prev_idx: List[Optional[int]] = [None] * n_curr
                assigned_mode: List[str] = ["none"] * n_curr
                assigned_overlap: List[float] = [0.0] * n_curr

                if prev_components and n_curr > 0:
                    # Primary match policy: one previous component per current component.
                    # This avoids multi-match averaging that can inflate motion on split/merge boundaries.
                    for ci, curr in enumerate(curr_candidates):
                        best_pi = -1
                        best_inter = 0
                        for pi, prev in enumerate(prev_components):
                            inter = _bbox_intersection_area(curr["bbox"], prev["bbox"])
                            if inter > best_inter:
                                best_inter = inter
                                best_pi = pi
                        if best_pi >= 0 and best_inter > 0:
                            assigned_prev_idx[ci] = int(best_pi)
                            assigned_mode[ci] = "overlap"
                            assigned_overlap[ci] = float(best_inter)

                    for ci, curr in enumerate(curr_candidates):
                        if assigned_prev_idx[ci] is not None:
                            continue
                        best_pi = -1
                        best_d = 1e9
                        max_dist = max(8.0, 0.5 * np.sqrt(float(curr["area"])))
                        cx = float(curr["centroid"][0])
                        cy = float(curr["centroid"][1])
                        for pi, prev in enumerate(prev_components):
                            dx = float(cx - prev["centroid"][0])
                            dy = float(cy - prev["centroid"][1])
                            d = float(np.hypot(dx, dy))
                            if d < best_d and d <= max_dist:
                                best_d = d
                                best_pi = pi
                        if best_pi >= 0:
                            assigned_prev_idx[ci] = int(best_pi)
                            assigned_mode[ci] = "nearest"

                children_by_prev: Dict[int, List[int]] = {}
                for ci, pi in enumerate(assigned_prev_idx):
                    if pi is None or assigned_mode[ci] != "overlap":
                        continue
                    children_by_prev.setdefault(int(pi), []).append(int(ci))

                split_children: set[int] = set()
                for pi, child_idxs in children_by_prev.items():
                    if len(child_idxs) <= 1:
                        continue
                    # Split detected: same previous component contributes to multiple current components.
                    # Keep inherited len_smooth continuity for all children.
                    # Reset X-motion only on detached children (non-primary child).
                    keep_ci = max(
                        child_idxs,
                        key=lambda ci: (assigned_overlap[ci], float(curr_candidates[ci]["area"])),
                    )
                    for ci in child_idxs:
                        if ci == keep_ci:
                            continue
                        split_children.add(int(ci))

                for ci, curr in enumerate(curr_candidates):
                    pi = assigned_prev_idx[ci]
                    if pi is not None:
                        prev = prev_components[int(pi)]
                        prev_len = float(prev.get("len_smooth", base_len))
                        motion_px = float(
                            abs(float(curr["centroid"][0]) - float(prev["centroid"][0]))
                        )
                        prev_area = float(prev.get("area", curr["area"]))
                    else:
                        prev_len = float(base_len)
                        motion_px = 0.0
                        prev_area = float(curr["area"])

                    motion_norm = _ramp01(
                        motion_px,
                        motion_deadzone_px,
                        motion_max_px,
                    )
                    area_norm = _ramp01(float(curr["area"]), area_min_px, area_max_px)

                    # Reset X-motion contribution when area changes sharply (merge/split/noisy holes).
                    area_changed = False
                    if pi is not None:
                        ratio_num = max(float(curr["area"]), prev_area)
                        ratio_den = max(1.0, min(float(curr["area"]), prev_area))
                        area_ratio = ratio_num / ratio_den
                        if area_reset_ratio > 1.0 and area_ratio >= area_reset_ratio:
                            area_changed = True
                        if (
                            area_reset_abs_px > 0.0
                            and abs(float(curr["area"]) - prev_area) >= area_reset_abs_px
                        ):
                            area_changed = True
                    if area_changed:
                        motion_norm = 0.0

                    if ci in split_children:
                        # On split frame, keep continuity from parent but reset X deviation.
                        motion_norm = 0.0

                    motion_mult = 1.0 + motion_gain * motion_norm * area_norm
                    target_len = float(base_len) * motion_mult
                    target_len = float(max(0.0, min(float(max_len_cap), target_len)))

                    if target_len >= prev_len:
                        alpha = alpha_up
                        max_delta = max_delta_up
                    else:
                        alpha = alpha_down
                        max_delta = max_delta_down
                    smoothed = prev_len + alpha * (target_len - prev_len)
                    delta = float(smoothed - prev_len)
                    delta = float(max(-max_delta, min(max_delta, delta)))
                    len_smooth = float(max(0.0, min(float(max_len_cap), prev_len + delta)))

                    curr_components.append(
                        {
                            "label": int(curr["label"]),
                            "bbox": curr["bbox"],
                            "centroid": curr["centroid"],
                            "area": int(curr["area"]),
                            "len_smooth": len_smooth,
                        }
                    )
                    comp_len_by_label[int(curr["label"])] = len_smooth
            else:
                for curr in curr_candidates:
                    len_smooth = float(base_len)
                    curr_components.append(
                        {
                            "label": int(curr["label"]),
                            "bbox": curr["bbox"],
                            "centroid": curr["centroid"],
                            "area": int(curr["area"]),
                            "len_smooth": len_smooth,
                        }
                    )
                    comp_len_by_label[int(curr["label"])] = len_smooth

            for y in range(height):
                row_bin = frame_bin[y]
                x = 0
                while x < width:
                    if row_bin[x] == 0:
                        x += 1
                        continue
                    run_start = x
                    while x + 1 < width and row_bin[x + 1] != 0:
                        x += 1
                    run_end = x
                    x += 1

                    run_vals = frame[y, run_start : run_end + 1]
                    if run_vals.size <= 0:
                        continue

                    run_labels = labels[y, run_start : run_end + 1]
                    lab_vals = run_labels[run_labels > 0]
                    if lab_vals.size > 0:
                        lab = int(np.bincount(lab_vals).argmax())
                    else:
                        lab = 0

                    comp_len = float(comp_len_by_label.get(lab, base_len))
                    run_w = int(run_end - run_start + 1)
                    width_mult = (
                        float((max(1.0, float(run_w)) / width_ref_px) ** width_power)
                        if width_adaptive
                        else 1.0
                    )
                    run_len = int(round(comp_len * width_mult))
                    run_len = int(max(0, min(max_len_cap, run_len)))
                    if run_len <= 0:
                        continue

                    touches_left = run_start < border_tol
                    touches_right = run_end >= right_touch_start
                    dir_left = bool(touches_right and not touches_left)

                    for s in range(1, run_len + 1):
                        opacity = float(_shadow_curve_opacity(float(s) / float(run_len), curve))
                        if opacity <= 0.0:
                            continue
                        if dir_left:
                            dst_a = int(run_start - s)
                            dst_b = int(run_end - s)
                        else:
                            dst_a = int(run_start + s)
                            dst_b = int(run_end + s)

                        clip_a = max(0, dst_a)
                        clip_b = min(width - 1, dst_b)
                        if clip_b < clip_a:
                            continue
                        src_start = int(clip_a - dst_a)
                        src_end = int(src_start + (clip_b - clip_a + 1))
                        src_slice = run_vals[src_start:src_end]
                        if src_slice.size <= 0:
                            continue
                        dst_slice = canvas[y, clip_a : clip_b + 1]
                        src_vals = src_slice * opacity
                        mask_gt = src_vals > dst_slice
                        dst_slice[mask_gt] = src_vals[mask_gt]

            if motion_chain_enabled:
                prev_components = curr_components

        out_np[t, 0] = canvas

    if state is not None:
        state["prev_components"] = prev_components if motion_chain_enabled else []

    return torch.from_numpy(out_np).to(device=mask.device, dtype=mask.dtype)


def apply_mask_dilation(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    if kernel_size <= 0:
        return mask
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if use_gpu:
        padding = k // 2
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=padding)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    out = []
    for t in range(mask.shape[0]):
        frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        dil = cv2.dilate(frame_np, kernel, iterations=1)
        out.append(torch.from_numpy(dil).float().div(255.0).unsqueeze(0))
    return torch.stack(out).to(mask.device)


def apply_gaussian_blur(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    if kernel_size <= 0:
        return mask
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if use_gpu:
        sigma = k / 6.0
        ax = torch.arange(-k // 2 + 1.0, k // 2 + 1.0, device=mask.device)
        gauss = torch.exp(-(ax**2) / (2 * sigma**2))
        kernel_1d = (gauss / gauss.sum()).view(1, 1, 1, k)
        blurred = F.conv2d(mask, kernel_1d, padding=(0, k // 2), groups=mask.shape[1])
        blurred = F.conv2d(
            blurred,
            kernel_1d.permute(0, 1, 3, 2),
            padding=(k // 2, 0),
            groups=mask.shape[1],
        )
        return torch.clamp(blurred, 0.0, 1.0)
    out = []
    for t in range(mask.shape[0]):
        frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        bl = cv2.GaussianBlur(frame_np, (k, k), 0)
        out.append(torch.from_numpy(bl).float().div(255.0).unsqueeze(0))
    return torch.stack(out).to(mask.device)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _to_gray_batch_tensor(batch_np: np.ndarray) -> torch.Tensor:
    if batch_np.ndim == 4:
        if batch_np.shape[3] >= 3:
            gray = batch_np[..., :3].mean(axis=3)
        else:
            gray = batch_np[..., 0]
    elif batch_np.ndim == 3:
        gray = batch_np
    else:
        raise RuntimeError(f"Unsupported replace-mask batch shape: {batch_np.shape}")

    gray = gray.astype(np.float32)
    if gray.size > 0 and float(np.nanmax(gray)) > 1.5:
        gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)
    return torch.from_numpy(gray).float().unsqueeze(1)


def _fps_from_reader(reader: VideoReader) -> float:
    try:
        fps = float(reader.get_avg_fps())
    except Exception:
        fps = 0.0
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 24.0
    return fps


def _fmt_fps(fps: float) -> str:
    txt = f"{fps:.6f}".rstrip("0").rstrip(".")
    return txt if txt else "24"


class GrayFFmpegWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float) -> None:
        self.output_path = output_path
        self.proc: Optional[subprocess.Popen] = None
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s:v",
            f"{width}x{height}",
            "-r",
            _fmt_fps(fps),
            "-i",
            "-",
            "-an",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-slices",
            "16",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "gray",
            output_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin pipe.")

    def write_gray_u8(self, frames_u8: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("FFmpeg writer is closed.")
        if frames_u8.dtype != np.uint8:
            raise RuntimeError(f"Expected uint8 frames, got {frames_u8.dtype}")
        if not frames_u8.flags["C_CONTIGUOUS"]:
            frames_u8 = np.ascontiguousarray(frames_u8)
        self.proc.stdin.write(frames_u8.tobytes())

    def close(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        if proc.stdin is not None:
            try:
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.stdin.close()
            except Exception:
                pass
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc} for {self.output_path}")


def _iter_inputs(input_dir: str, glob_expr: str) -> List[str]:
    patterns = [p.strip() for p in str(glob_expr).split(",") if p.strip()]
    if not patterns:
        patterns = ["*_replace_mask.*"]
    found: List[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(input_dir, pattern)))
    uniq = sorted({os.path.abspath(p) for p in found if os.path.isfile(p)})
    return uniq


def _make_output_path(output_dir: str, input_path: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{stem}.mkv")


def process_one_file(
    path: str, args: argparse.Namespace, device: torch.device, run_stats: Dict[str, float]
) -> Tuple[bool, int]:
    name = os.path.basename(path)
    output_path = _make_output_path(args.output_dir, path)
    if args.skip_existing and os.path.isfile(output_path):
        LOG.info(f"[SKIP] {name} -> {os.path.basename(output_path)} exists")
        return True, 0

    os.makedirs(args.output_dir, exist_ok=True)

    reader: Optional[VideoReader] = None
    writer: Optional[GrayFFmpegWriter] = None
    frames_written = 0
    try:
        reader = VideoReader(path, ctx=cpu(0))
        num_frames = int(len(reader))
        if num_frames <= 0:
            raise RuntimeError("video has zero frames")

        first = reader[0].asnumpy()
        if first.ndim == 3:
            height, width = int(first.shape[0]), int(first.shape[1])
        elif first.ndim == 2:
            height, width = int(first.shape[0]), int(first.shape[1])
        else:
            raise RuntimeError(f"unsupported frame shape: {first.shape}")

        fps = _fps_from_reader(reader)
        writer = GrayFFmpegWriter(output_path, width=width, height=height, fps=fps)

        shadow_state: Dict[str, Any] = {"prev_components": []}
        chunk_size = max(1, int(args.chunk_size))
        for frame_start in range(0, num_frames, chunk_size):
            frame_end = min(frame_start + chunk_size, num_frames)
            frame_indices = list(range(frame_start, frame_end))
            batch_np = reader.get_batch(frame_indices).asnumpy()
            mask = _to_gray_batch_tensor(batch_np).to(device=device)

            if float(args.mask_binarize_threshold) >= 0.0:
                processed = (mask > float(args.mask_binarize_threshold)).float()
            else:
                processed = (mask > 0.5).float()

            if int(args.mask_dilate_kernel_size) > 0:
                processed = apply_mask_dilation(
                    processed,
                    int(args.mask_dilate_kernel_size),
                    use_gpu=bool(args.use_gpu_mask_ops),
                )
            if int(args.mask_blur_kernel_size) > 0:
                processed = apply_gaussian_blur(
                    processed,
                    int(args.mask_blur_kernel_size),
                    use_gpu=bool(args.use_gpu_mask_ops),
                )
            if int(args.shadow_length_px) > 0:
                processed = apply_shadow_blur(
                    processed,
                    base_length_px=int(args.shadow_length_px),
                    curve=float(args.shadow_curve),
                    motion_gain=float(args.shadow_motion_gain),
                    motion_deadzone_px=float(args.shadow_motion_deadzone_px),
                    motion_max_px=float(args.shadow_motion_max_px),
                    motion_chain_enabled=bool(args.shadow_motion_chain_enabled),
                    area_min_px=float(args.shadow_area_min_px),
                    area_max_px=float(args.shadow_area_max_px),
                    area_reset_ratio=float(args.shadow_area_reset_ratio),
                    area_reset_abs_px=float(args.shadow_area_reset_abs_px),
                    component_merge_y_tol_px=int(args.shadow_component_merge_y_tol_px),
                    alpha_down=float(args.shadow_alpha_down),
                    width_adaptive=bool(args.shadow_width_adaptive),
                    use_gpu=bool(args.use_gpu_mask_ops),
                    state=shadow_state,
                    border_tolerance_px=2,
                    width_ref_px=20.0,
                    width_power=1.0,
                )

            out_u8 = (
                torch.clamp(processed, 0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()[:, 0, :, :]
            )
            writer.write_gray_u8(out_u8)
            chunk_frames = int(frame_end - frame_start)
            frames_written += chunk_frames
            run_stats["frames_done"] = float(run_stats.get("frames_done", 0.0) + chunk_frames)

            if frame_end == num_frames or ((frame_start // chunk_size) % 5 == 0):
                now_ts = time.perf_counter()
                total_elapsed_s = max(1e-6, now_ts - float(run_stats["start_ts"]))
                total_frames = int(run_stats.get("frames_done", 0.0))
                fpm_global = (total_frames * 60.0) / total_elapsed_s
                fps_global = total_frames / total_elapsed_s
                pct = (100.0 * frame_end) / float(num_frames)
                LOG.info(
                    f"[RUN ] {name}: {frame_end}/{num_frames} ({pct:.1f}%) | "
                    f"GLOBAL FPM(avg)={fpm_global:.1f} | GLOBAL FPS(avg)={fps_global:.2f}"
                )

        writer.close()
        writer = None
        now_ts = time.perf_counter()
        total_elapsed_s = max(1e-6, now_ts - float(run_stats["start_ts"]))
        total_frames = int(run_stats.get("frames_done", 0.0))
        fpm_global = (total_frames * 60.0) / total_elapsed_s
        fps_global = total_frames / total_elapsed_s
        LOG.info(
            f"[DONE] {name} -> {output_path} | frames={frames_written} | "
            f"GLOBAL FPM(avg)={fpm_global:.1f} | GLOBAL FPS(avg)={fps_global:.2f}"
        )
        return True, frames_written
    except Exception as exc:
        LOG.error(f"[FAIL] {name}: {exc}", exc_info=args.verbose)
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            if os.path.isfile(output_path):
                os.remove(output_path)
        except Exception:
            pass
        return False, frames_written
    finally:
        del reader
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    motion_defaults = _load_motion_defaults_from_json()
    parser = argparse.ArgumentParser(description="Mask-only preprocessing for merging.")
    parser.add_argument(
        "--input-dir",
        default="./work/mask/fixed/",
        help="Folder containing replace-mask videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="./work/mask_formerge/",
        help="Folder where processed masks are written.",
    )
    parser.add_argument(
        "--glob",
        default="*_replace_mask.*",
        help="Input glob pattern(s), comma-separated.",
    )
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--mask-binarize-threshold", type=float, default=0.5)
    parser.add_argument("--mask-dilate-kernel-size", type=int, default=2)
    parser.add_argument("--mask-blur-kernel-size", type=int, default=4)
    parser.add_argument("--shadow-length-px", type=int, default=30)
    parser.add_argument("--shadow-curve", type=float, default=1.0)
    parser.add_argument(
        "--shadow-motion-gain",
        type=float,
        default=float(motion_defaults["shadow_motion_gain"]),
    )
    parser.add_argument(
        "--shadow-motion-deadzone-px",
        type=float,
        default=float(motion_defaults["shadow_motion_deadzone_px"]),
    )
    parser.add_argument(
        "--shadow-motion-max-px",
        type=float,
        default=float(motion_defaults["shadow_motion_max_px"]),
        help="Motion (abs dx) value that maps to 100%% motion gain.",
    )
    parser.add_argument(
        "--shadow-area-min-px",
        type=float,
        default=float(motion_defaults["shadow_area_min_px"]),
        help="Mask area deadzone for motion gain modulation.",
    )
    parser.add_argument(
        "--shadow-area-max-px",
        type=float,
        default=float(motion_defaults["shadow_area_max_px"]),
        help="Mask area that maps to 100%% motion gain modulation.",
    )
    parser.add_argument(
        "--shadow-area-reset-ratio",
        type=float,
        default=float(motion_defaults["shadow_area_reset_ratio"]),
        help="Reset motion offset when area ratio vs previous component exceeds this value.",
    )
    parser.add_argument(
        "--shadow-area-reset-abs-px",
        type=float,
        default=float(motion_defaults["shadow_area_reset_abs_px"]),
        help="Reset motion offset when absolute area delta exceeds this value (0 disables).",
    )
    parser.add_argument(
        "--shadow-component-merge-y-tol-px",
        type=int,
        default=int(motion_defaults["shadow_component_merge_y_tol_px"]),
        help="Vertical tolerance (px) used to merge fragmented components before CC.",
    )
    parser.add_argument(
        "--shadow-alpha-down",
        type=float,
        default=float(motion_defaults["shadow_alpha_down"]),
        help="IIR down alpha for shadow length release.",
    )
    parser.add_argument(
        "--shadow-motion-chain-enabled",
        action="store_true",
        default=bool(motion_defaults["shadow_motion_chain_enabled"]),
        help="Enable full motion-detection chain for shadow expansion.",
    )
    parser.add_argument(
        "--no-shadow-motion-chain-enabled",
        dest="shadow_motion_chain_enabled",
        action="store_false",
        help="Disable full motion-detection chain and use static shadow length.",
    )
    parser.add_argument("--shadow-width-adaptive", action="store_true", default=True)
    parser.add_argument("--no-shadow-width-adaptive", dest="shadow_width_adaptive", action="store_false")
    parser.add_argument("--use-gpu-mask-ops", action="store_true", default=True)
    parser.add_argument("--no-use-gpu-mask-ops", dest="use_gpu_mask_ops", action="store_false")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.verbose)

    if not os.path.isdir(args.input_dir):
        LOG.error(f"Input folder not found: {args.input_dir}")
        return 2

    files = _iter_inputs(args.input_dir, args.glob)
    if not files:
        LOG.warning(f"No input files found in {args.input_dir} with glob={args.glob!r}")
        return 0

    use_gpu = bool(args.use_gpu_mask_ops and torch.cuda.is_available())
    if args.use_gpu_mask_ops and not use_gpu:
        LOG.warning("CUDA not available. Falling back to CPU mask ops.")
    device = torch.device("cuda" if use_gpu else "cpu")
    LOG.info(
        f"Starting mask-only run: files={len(files)} | chunk={args.chunk_size} | "
        f"gpu_mask_ops={use_gpu} | shadow_width_adaptive={args.shadow_width_adaptive}"
    )

    ok = 0
    fail = 0
    run_stats: Dict[str, float] = {"start_ts": time.perf_counter(), "frames_done": 0.0}
    for idx, path in enumerate(files, start=1):
        LOG.info(f"[FILE] {idx}/{len(files)} {os.path.basename(path)}")
        success, _frames_done = process_one_file(path, args, device=device, run_stats=run_stats)
        if success:
            ok += 1
        else:
            fail += 1

    run_elapsed_s = max(1e-6, time.perf_counter() - float(run_stats["start_ts"]))
    total_frames = int(run_stats.get("frames_done", 0.0))
    total_fpm_avg = (total_frames * 60.0) / run_elapsed_s
    total_fps_avg = total_frames / run_elapsed_s
    LOG.info(
        f"Completed. success={ok} failed={fail} output_dir={os.path.abspath(args.output_dir)} | "
        f"frames={total_frames} | GLOBAL FPM(avg)={total_fpm_avg:.1f} | GLOBAL FPS(avg)={total_fps_avg:.2f}"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
