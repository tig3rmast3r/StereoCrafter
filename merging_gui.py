import os
import glob
import json
import csv
import shutil
import threading
import gc
import tkinter as tk  # Used for PanedWindow
from tkinter import filedialog, messagebox, ttk
from ttkthemes import ThemedTk
from typing import Optional, Tuple, Callable, Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageTk
from decord import VideoReader, cpu
import logging
import time
import queue
import faulthandler
import signal
from concurrent.futures import ThreadPoolExecutor
from dependency.stereocrafter_util import (
    Tooltip,
    logger,
    get_video_stream_info,
    draw_progress_bar,
    release_cuda_memory,
    set_util_logger_level,
    encode_frames_to_mp4,
    read_video_frames_decord,
    start_ffmpeg_pipe_process,
    apply_color_transfer,
    create_single_slider_with_label_updater,
    apply_dubois_anaglyph,
    apply_optimized_anaglyph,
    SidecarConfigManager,
    find_video_by_core_name,
    find_sidecar_file,
    read_clip_sidecar,
    apply_borders_to_frames,
)
from dependency.video_previewer import VideoPreviewer

GUI_VERSION = "26-02-08.3"
_FAULTHANDLER_LOG = None


def _enable_debug_faulthandler() -> None:
    """Enable crash stack dumps when debug mode is enabled."""
    global _FAULTHANDLER_LOG
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "merging_gui_faulthandler.log")
        _FAULTHANDLER_LOG = open(log_path, "a", buffering=1)
        _FAULTHANDLER_LOG.write(
            f"\n=== debug session {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===\n"
        )
        _FAULTHANDLER_LOG.flush()
        faulthandler.enable(file=_FAULTHANDLER_LOG, all_threads=True)
        # Optional on-demand dump while process is alive:
        # kill -USR1 <pid>  -> writes full traceback of all threads to log.
        try:
            faulthandler.register(signal.SIGUSR1, file=_FAULTHANDLER_LOG, all_threads=True)
        except Exception:
            # Keep crash dumping even if signal registration is not available.
            pass
        logger.info(f"Debug faulthandler active: {log_path}")
    except Exception as e:
        logger.warning(f"Failed to enable debug faulthandler: {e}")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


# --- MASK PROCESSING FUNCTIONS (from test.py) ---
def apply_mask_dilation(
    mask: torch.Tensor, kernel_size: int, use_gpu: bool = True
) -> torch.Tensor:
    if kernel_size <= 0:
        return mask
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if use_gpu:
        padding = kernel_val // 2
        return F.max_pool2d(mask, kernel_size=kernel_val, stride=1, padding=padding)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_val, kernel_val))
        processed_frames = []
        for t in range(mask.shape[0]):
            frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            dilated_np = cv2.dilate(frame_np, kernel, iterations=1)
            dilated_tensor = torch.from_numpy(dilated_np).float() / 255.0
            processed_frames.append(dilated_tensor.unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)


def apply_gaussian_blur(
    mask: torch.Tensor, kernel_size: int, use_gpu: bool = True
) -> torch.Tensor:
    if kernel_size <= 0:
        return mask
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if use_gpu:
        sigma = kernel_val / 6.0
        ax = torch.arange(
            -kernel_val // 2 + 1.0, kernel_val // 2 + 1.0, device=mask.device
        )
        gauss = torch.exp(-(ax**2) / (2 * sigma**2))
        kernel_1d = (gauss / gauss.sum()).view(1, 1, 1, kernel_val)
        blurred_mask = F.conv2d(
            mask, kernel_1d, padding=(0, kernel_val // 2), groups=mask.shape[1]
        )
        blurred_mask = F.conv2d(
            blurred_mask,
            kernel_1d.permute(0, 1, 3, 2),
            padding=(kernel_val // 2, 0),
            groups=mask.shape[1],
        )
        return torch.clamp(blurred_mask, 0.0, 1.0)
    else:
        processed_frames = []
        for t in range(mask.shape[0]):
            frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            blurred_np = cv2.GaussianBlur(frame_np, (kernel_val, kernel_val), 0)
            blurred_tensor = torch.from_numpy(blurred_np).float() / 255.0
            processed_frames.append(blurred_tensor.unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)


def _shadow_curve_opacity(u: float, curve: float) -> float:
    u = float(max(0.0, min(1.0, u)))
    c = float(max(-1.0, min(1.0, curve)))
    linear = 1.0 - u
    if abs(c) <= 1e-6:
        return linear
    if c > 0.0:
        # Positive: fuller at start, steeper tail near the end.
        bulged = (1.0 - u) ** 0.5
        return (1.0 - c) * linear + c * bulged
    # Negative: faster drop at the beginning, softer tail.
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
    motion_max_px: float = 40.0,
    motion_chain_enabled: bool = True,
    area_min_px: float = 0.0,
    area_max_px: float = 0.0,
    area_reset_ratio: float = 1.8,
    area_reset_abs_px: float = 0.0,
    component_merge_y_tol_px: int = 0,
    alpha_down: float = 0.45,
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

    alpha_up = 0.45
    alpha_down = float(max(0.0, min(1.0, alpha_down)))
    max_delta_up = max(1.0, 0.35 * base_len)
    max_delta_down = max(1.0, 0.20 * base_len)
    max_len_cap = int(max(100, 4 * base_len))

    mask_cpu = mask.detach().to(device="cpu", dtype=torch.float32).numpy()  # T,1,H,W
    t_count, _c, height, width = mask_cpu.shape
    right_touch_start = width - border_tol

    prev_components: List[Dict[str, Any]] = []
    if motion_chain_enabled and state is not None:
        prev_components = list(state.get("prev_components", []) or [])

    # Keep shadow processing fully on NumPy buffers and convert back to torch once.
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
                for _pi, child_idxs in children_by_prev.items():
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

            # Row/run-aware dynamic shadow length + right-border inversion.
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
                        # Avoid np.maximum in-place on overlapping views: observed to
                        # trigger sporadic native crashes (SIGSEGV/SIGILL) on long runs.
                        mask_gt = src_vals > dst_slice
                        dst_slice[mask_gt] = src_vals[mask_gt]

            if motion_chain_enabled:
                prev_components = curr_components

        out_np[t, 0] = canvas

    if state is not None:
        state["prev_components"] = prev_components if motion_chain_enabled else []

    return torch.from_numpy(out_np).to(device=mask.device, dtype=mask.dtype)



# --- COLOR TRANSFER (SAFE) HELPERS ---

def _telea_inpaint_rgb_uint8(frame_rgb_u8: np.ndarray, mask_u8: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    OpenCV inpaint helper (TELEA). frame_rgb_u8: HxWx3 RGB uint8, mask_u8: HxW uint8 0/255.
    Returns RGB uint8.
    """
    try:
        # cv2.inpaint expects 1-channel mask, non-zero indicates inpaint region
        out_bgr = cv2.inpaint(cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2BGR), mask_u8, radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Telea inpaint failed: {e}", exc_info=True)
        return frame_rgb_u8


def _normalize_warped_fill_mode(fill_mode: str) -> str:
    mode = str(fill_mode).strip().lower().replace("-", "_")
    if mode not in ("telea", "directional", "hybrid"):
        return "hybrid"
    return mode


def _directional_fill_rgb_uint8(
    frame_rgb_u8: np.ndarray,
    mask_u8: np.ndarray,
    border_tol_px: int = 2,
) -> np.ndarray:
    """
    Directional fill for warped_filled reference:
    - default runs copy color from RIGHT side of each masked run
    - runs touching RIGHT border (with tolerance) copy from LEFT side

    Direction is decided per row/run (not per connected component), so the side
    can switch inside the same connected shape.
    """
    try:
        if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
            return frame_rgb_u8
        if mask_u8.ndim != 2:
            return frame_rgb_u8
        h, w = mask_u8.shape
        if h != frame_rgb_u8.shape[0] or w != frame_rgb_u8.shape[1]:
            return frame_rgb_u8

        mask_bin = (mask_u8 > 0).astype(np.uint8)
        if not np.any(mask_bin):
            return frame_rgb_u8.copy()

        out = frame_rgb_u8.copy()
        border_cols = max(1, min(int(border_tol_px), w))
        right_touch_start = w - border_cols

        def _find_valid_x(y: int, start_x: int, step: int) -> int:
            x = int(start_x)
            while 0 <= x < w:
                if mask_bin[y, x] == 0:
                    return x
                x += step
            return -1

        for y in range(h):
            xs = np.flatnonzero(mask_bin[y])
            if xs.size == 0:
                continue

            run_start = int(xs[0])
            prev = int(xs[0])

            def _paint_run(a: int, b: int):
                touches_left = a < border_cols
                touches_right = b >= right_touch_start
                prefer_left = touches_right and not touches_left
                if prefer_left:
                    src_x = _find_valid_x(y, a - 1, -1)
                    if src_x < 0:
                        src_x = _find_valid_x(y, b + 1, 1)
                else:
                    src_x = _find_valid_x(y, b + 1, 1)
                    if src_x < 0:
                        src_x = _find_valid_x(y, a - 1, -1)
                if src_x >= 0:
                    out[y, a : b + 1, :] = frame_rgb_u8[y, src_x : src_x + 1, :]

            for cur in xs[1:]:
                cur = int(cur)
                if cur != prev + 1:
                    _paint_run(run_start, prev)
                    run_start = cur
                prev = cur
            _paint_run(run_start, prev)

        return out
    except Exception as e:
        logger.error(f"Directional fill failed: {e}", exc_info=True)
        return frame_rgb_u8


def _build_warped_filled_reference(
    frame_rgb_u8: np.ndarray,
    mask_u8: np.ndarray,
    fill_mode: str = "hybrid",
    border_tol_px: int = 2,
    telea_radius: int = 3,
) -> np.ndarray:
    mode = _normalize_warped_fill_mode(fill_mode)
    if mode == "telea":
        return _telea_inpaint_rgb_uint8(frame_rgb_u8, mask_u8, radius=telea_radius)

    directional = _directional_fill_rgb_uint8(
        frame_rgb_u8, mask_u8, border_tol_px=border_tol_px
    )
    if mode == "directional":
        return directional

    # Hybrid: keep TELEA as base and replace only runs that touch the right border
    # (with tolerance) with directional fill. This avoids wrong-side leakage at
    # the right edge without changing interior regions.
    try:
        mask_bin = (mask_u8 > 0).astype(np.uint8)
        if not np.any(mask_bin):
            return frame_rgb_u8.copy()

        h, w = mask_bin.shape
        border_cols = max(1, min(int(border_tol_px), w))
        edge_mask = np.zeros((h, w), dtype=bool)
        right_touch_start = w - border_cols
        for y in range(h):
            xs = np.flatnonzero(mask_bin[y])
            if xs.size == 0:
                continue
            run_start = int(xs[0])
            prev = int(xs[0])
            for cur in xs[1:]:
                cur = int(cur)
                if cur != prev + 1:
                    a, b = run_start, prev
                    touches_left = a < border_cols
                    touches_right = b >= right_touch_start
                    if touches_right and not touches_left:
                        edge_mask[y, a : b + 1] = True
                    run_start = cur
                prev = cur
            a, b = run_start, prev
            touches_left = a < border_cols
            touches_right = b >= right_touch_start
            if touches_right and not touches_left:
                edge_mask[y, a : b + 1] = True

        if not np.any(edge_mask):
            return _telea_inpaint_rgb_uint8(frame_rgb_u8, mask_u8, radius=telea_radius)

        telea = _telea_inpaint_rgb_uint8(frame_rgb_u8, mask_u8, radius=telea_radius)
        telea[edge_mask] = directional[edge_mask]
        return telea
    except Exception as e:
        logger.error(f"Hybrid warped fill failed: {e}", exc_info=True)
        return _telea_inpaint_rgb_uint8(frame_rgb_u8, mask_u8, radius=telea_radius)


def _make_stats_mask(
    mask_1hw: torch.Tensor,
    stats_region: str,
    ring_width: int,
    use_gpu: bool = False,
) -> torch.Tensor:
    """
    Returns [H,W] float mask in {0,1} to be used as VALID region for stats.
    stats_region: global|nonmask|ring
    mask_1hw: [1,H,W] or [H,W] (values 0..1 where 1 indicates inpaint region)
    """
    m = mask_1hw
    if m.dim() == 3 and m.shape[0] == 1:
        m = m[0]
    if m.dim() != 2:
        raise ValueError("mask must be [H,W] or [1,H,W]")

    if stats_region == "global":
        return torch.ones_like(m)

    inv = (1.0 - (m > 0.5).float())

    if stats_region == "nonmask":
        return inv

    # ring
    if ring_width <= 0:
        return inv

    # Directional ring by row/run:
    # - default runs: collect stats on RIGHT side of mask
    # - runs touching RIGHT border (2px tol): collect on LEFT side
    # Ring width adapts to ROI/run size: effective width = min(ring_width, run_len).
    base = (m > 0.5).float()
    base_np = (base.detach().cpu().numpy() > 0.5).astype(np.uint8)
    h, w = base_np.shape
    if h <= 0 or w <= 0:
        return inv

    border_tol_px = 2
    border_cols = max(1, min(border_tol_px, w))
    rw = int(ring_width)
    if rw <= 0:
        return inv
    right_touch_start = w - border_cols
    ring_right = np.zeros_like(base_np, dtype=np.uint8)
    ring_left = np.zeros_like(base_np, dtype=np.uint8)

    for y in range(h):
        xs = np.flatnonzero(base_np[y])
        if xs.size == 0:
            continue
        run_start = int(xs[0])
        prev = int(xs[0])
        for cur in xs[1:]:
            cur = int(cur)
            if cur != prev + 1:
                a, b = run_start, prev
                run_len = int(b - a + 1)
                width = max(1, min(rw, run_len))
                touches_left = a < border_cols
                touches_right = b >= right_touch_start
                if touches_right and not touches_left:
                    x1 = a
                    x0 = max(0, x1 - width)
                    if x0 < x1:
                        ring_left[y, x0:x1] = 1
                    else:
                        xr0 = b + 1
                        xr1 = min(w, xr0 + width)
                        if xr0 < xr1:
                            ring_right[y, xr0:xr1] = 1
                else:
                    x0 = b + 1
                    x1 = min(w, x0 + width)
                    if x0 < x1:
                        ring_right[y, x0:x1] = 1
                    else:
                        xl1 = a
                        xl0 = max(0, xl1 - width)
                        if xl0 < xl1:
                            ring_left[y, xl0:xl1] = 1
                run_start = cur
            prev = cur

        a, b = run_start, prev
        run_len = int(b - a + 1)
        width = max(1, min(rw, run_len))
        touches_left = a < border_cols
        touches_right = b >= right_touch_start
        if touches_right and not touches_left:
            x1 = a
            x0 = max(0, x1 - width)
            if x0 < x1:
                ring_left[y, x0:x1] = 1
            else:
                xr0 = b + 1
                xr1 = min(w, xr0 + width)
                if xr0 < xr1:
                    ring_right[y, xr0:xr1] = 1
        else:
            x0 = b + 1
            x1 = min(w, x0 + width)
            if x0 < x1:
                ring_right[y, x0:x1] = 1
            else:
                xl1 = a
                xl0 = max(0, xl1 - width)
                if xl0 < xl1:
                    ring_left[y, xl0:xl1] = 1

    ring_bool = ((ring_right > 0) | (ring_left > 0)) & (base_np == 0)
    if np.any(ring_bool):
        ring = torch.from_numpy(ring_bool.astype(np.float32)).to(device=m.device, dtype=m.dtype)
        return ring

    # Fallback: isotropic outer ring, then nonmask.
    mm = base.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    k = int(ring_width) * 2 + 1
    pad = k // 2
    dil = F.max_pool2d(mm, kernel_size=k, stride=1, padding=pad)
    ring = (dil[0, 0] - mm[0, 0]).clamp(0, 1)
    if ring.sum().item() < 1.0:
        return inv
    return ring


def apply_color_transfer_safe(
    source_frame: torch.Tensor,
    target_frame: torch.Tensor,
    *,
    black_thresh: float = 8.0,
    min_valid_ratio: float = 0.01,
    min_valid: int = 300,
    strength: float = 1.0,
    clamp_scale_L: Tuple[float, float] = (0.7, 1.3),
    clamp_scale_ab: Tuple[float, float] = (0.6, 1.4),
    exclude_black_in_target: bool = False,
    source_valid_mask: Optional[torch.Tensor] = None,
    target_valid_mask: Optional[torch.Tensor] = None,
    target_stats_frame: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reinhard-like color transfer in LAB (float32, clamped), adapted from inpainting_gui.

    - Expects [C,H,W] float [0,1] tensors.
    - Stats are computed on valid masks (optional) and can be computed from a separate target_stats_frame.
    - Scales are clamped to prevent extreme shifts on small crops.
    """
    try:
        src_t = source_frame.detach().cpu().float()
        tgt_t = target_frame.detach().cpu().float()

        # Accept [1,3,H,W] by squeezing batch dim
        if src_t.dim() == 4 and src_t.shape[0] == 1:
            src_t = src_t[0]
        if tgt_t.dim() == 4 and tgt_t.shape[0] == 1:
            tgt_t = tgt_t[0]

        if src_t.dim() != 3 or tgt_t.dim() != 3 or src_t.shape[0] != 3 or tgt_t.shape[0] != 3:
            return target_frame

        Hs, Ws = int(src_t.shape[1]), int(src_t.shape[2])
        Ht, Wt = int(tgt_t.shape[1]), int(tgt_t.shape[2])
        if Hs != Ht or Ws != Wt:
            return target_frame

        if target_stats_frame is None:
            tstats_t = tgt_t
        else:
            tstats_t = target_stats_frame.detach().cpu().float()
            if tstats_t.dim() == 4 and tstats_t.shape[0] == 1:
                tstats_t = tstats_t[0]
            if tstats_t.shape != tgt_t.shape:
                return target_frame

        src_np = torch.clamp(src_t, 0.0, 1.0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)
        tgt_np = torch.clamp(tgt_t, 0.0, 1.0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)
        tstats_np = torch.clamp(tstats_t, 0.0, 1.0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)

        thr = float(black_thresh) / 255.0
        src_valid = (src_np.max(axis=2) > thr)

        if exclude_black_in_target:
            tgt_valid = (tstats_np.max(axis=2) > thr)
        else:
            tgt_valid = np.ones((Ht, Wt), dtype=bool)

        def _merge_mask(valid: np.ndarray, m: torch.Tensor) -> np.ndarray:
            mm = m.detach().cpu()
            if mm.dim() == 3 and mm.shape[0] == 1:
                mm = mm[0]
            if mm.dim() == 2 and mm.shape[0] == Ht and mm.shape[1] == Wt:
                return valid & (mm.numpy() > 0.5)
            return valid

        if source_valid_mask is not None:
            src_valid = _merge_mask(src_valid, source_valid_mask)

        if target_valid_mask is not None:
            tgt_valid = _merge_mask(tgt_valid, target_valid_mask)

        n_valid = int(src_valid.sum())
        min_valid_eff = max(int(min_valid), int(min_valid_ratio * Hs * Ws))
        if n_valid < min_valid_eff:
            return target_frame

        src_lab = cv2.cvtColor(src_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        tstats_lab = cv2.cvtColor(tstats_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_vals = src_lab[src_valid].reshape(-1, 3)
        tgt_vals = tstats_lab[tgt_valid].reshape(-1, 3)
        if tgt_vals.shape[0] == 0:
            tgt_vals = tstats_lab.reshape(-1, 3)

        src_mean = src_vals.mean(axis=0)
        src_std = src_vals.std(axis=0)
        tgt_mean = tgt_vals.mean(axis=0)
        tgt_std = tgt_vals.std(axis=0)

        src_std = np.clip(src_std, 1e-6, None)
        tgt_std = np.clip(tgt_std, 1e-6, None)

        scale = src_std / tgt_std
        scale[0] = float(np.clip(scale[0], clamp_scale_L[0], clamp_scale_L[1]))
        scale[1] = float(np.clip(scale[1], clamp_scale_ab[0], clamp_scale_ab[1]))
        scale[2] = float(np.clip(scale[2], clamp_scale_ab[0], clamp_scale_ab[1]))

        out_lab = (tgt_lab - tgt_mean) * scale + src_mean
        out_rgb = cv2.cvtColor(out_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32)
        out_t = torch.from_numpy(out_rgb).permute(2, 0, 1).contiguous()

        if strength >= 1.0:
            return out_t
        if strength <= 0.0:
            return target_frame
        return target_frame * (1.0 - strength) + out_t * strength

    except Exception as e:
        logger.error(f"Error during SAFE color transfer: {e}. Returning original target frame.", exc_info=True)
        return target_frame

# --- END COLOR TRANSFER (SAFE) HELPERS ---

# --- CT PRESETS (ranked by effectiveness from analyzer) ---
CT_PRESETS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "label": "1) safe sr=ring ts=inpainted ref=warped",
        "mode": "safe",
        "stats_region": "ring",
        "target_stats_source": "inpainted",
        "reference_source": "warped_filled",
        "warped_fill_mode": "telea",
    },
    {
        "id": 2,
        "label": "2) safe sr=ring ts=inpainted ref=left",
        "mode": "safe",
        "stats_region": "ring",
        "target_stats_source": "inpainted",
        "reference_source": "left",
        "warped_fill_mode": "telea",
    },
    {
        "id": 3,
        "label": "3) safe sr=global ts=inpainted ref=warped_filled fill=directional",
        "mode": "safe",
        "stats_region": "global",
        "target_stats_source": "inpainted",
        "reference_source": "warped_filled",
        "warped_fill_mode": "directional",
    },
    {
        "id": 4,
        "label": "4) safe sr=nonmask ts=inpainted ref=left",
        "mode": "safe",
        "stats_region": "nonmask",
        "target_stats_source": "inpainted",
        "reference_source": "left",
        "warped_fill_mode": "telea",
    },
    {
        "id": 5,
        "label": "5) safe sr=nonmask ts=inpainted ref=warped",
        "mode": "safe",
        "stats_region": "nonmask",
        "target_stats_source": "inpainted",
        "reference_source": "warped_filled",
        "warped_fill_mode": "telea",
    },
    {
        "id": 6,
        "label": "6) safe sr=global ts=inpainted ref=left",
        "mode": "safe",
        "stats_region": "global",
        "target_stats_source": "inpainted",
        "reference_source": "left",
        "warped_fill_mode": "telea",
    },
    {
        "id": 7,
        "label": "7) legacy",
        "mode": "legacy",
        "stats_region": "ring",
        "target_stats_source": "inpainted",
        "reference_source": "left",
        "warped_fill_mode": "telea",
    },
    {
        "id": 8,
        "label": "8) safe sr=ring ts=warped ref=left",
        "mode": "safe",
        "stats_region": "ring",
        "target_stats_source": "warped",
        "reference_source": "left",
        "warped_fill_mode": "telea",
    },
]

CT_PRESET_LABELS = [p["label"] for p in CT_PRESETS]
CT_PRESET_BY_LABEL = {p["label"]: p for p in CT_PRESETS}
CT_PRESET_BY_ID = {int(p["id"]): p for p in CT_PRESETS}
CT_PRESET_DEFAULT_LABEL = CT_PRESET_LABELS[0]
# Auto-eval order optimized to reuse caches (stats/ref) across adjacent presets.
CT_PRESET_AUTO_EVAL_ORDER = [1, 2, 8, 4, 5, 6, 3, 7]
CT_AUTO_EVAL_MAX_WORKERS = 3
CT_AUTO_MODE_OFF = "Off"
CT_AUTO_MODE_ON = "On"
CT_AUTO_MODE_CSV_BLEND = "CSV Blend"
CT_AUTO_MODE_OPTIONS = [CT_AUTO_MODE_OFF, CT_AUTO_MODE_ON, CT_AUTO_MODE_CSV_BLEND]
CT_CSV_BLEND_TRANSITION_ALPHAS = [0.24, 0.40, 0.60, 0.80]
CT_CSV_BLEND_OSC_ALPHA = 0.80
CT_CSV_BLEND_OSC_WINDOW = 6
CT_CSV_BLEND_MAX_ACTIVE_PRESETS = 4
CT_CSV_BLEND_PRUNE_EPS = 1e-3


def _resolve_ct_auto_mode_label(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in CT_AUTO_MODE_OPTIONS:
        return raw
    low = raw.lower()
    if low in {"off", "false", "0", "disabled", "manual"}:
        return CT_AUTO_MODE_OFF
    if low in {"on", "true", "1", "auto", "auto_eval"}:
        return CT_AUTO_MODE_ON
    if low in {"csv", "csv_blend", "csv-blend", "csv blend"}:
        return CT_AUTO_MODE_CSV_BLEND
    return CT_AUTO_MODE_ON


def _resolve_ct_auto_mode_from_settings(settings: Dict[str, Any]) -> str:
    if "ct_auto_mode" in settings:
        return _resolve_ct_auto_mode_label(settings.get("ct_auto_mode"))
    return CT_AUTO_MODE_ON if bool(settings.get("auto_ct_eval", False)) else CT_AUTO_MODE_OFF


def _parse_inpainted_basename(
    base_name: str,
) -> Tuple[Optional[str], Optional[str], bool]:
    inpaint_suffix = "_inpainted_right_eye.mp4"
    sbs_suffix = "_inpainted_sbs.mp4"
    if str(base_name).endswith(inpaint_suffix):
        core_with_width = str(base_name)[: -len(inpaint_suffix)]
        is_sbs_input = False
    elif str(base_name).endswith(sbs_suffix):
        core_with_width = str(base_name)[: -len(sbs_suffix)]
        is_sbs_input = True
    else:
        return None, None, False

    last_underscore_idx = core_with_width.rfind("_")
    if last_underscore_idx <= 0:
        return core_with_width, None, is_sbs_input
    core_name = core_with_width[:last_underscore_idx]
    return core_with_width, core_name, is_sbs_input


def _infer_splatted_layout_from_path(splatted_path: str) -> str:
    """Return one of: single (_splatted1), dual (_splatted2), quad (_splatted4/default)."""
    base = os.path.basename(str(splatted_path or ""))
    if "_splatted1" in base:
        return "single"
    if "_splatted2" in base:
        return "dual"
    return "quad"


def _normalize_csv_blend_lookup_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    base = os.path.basename(text)
    stem, _ext = os.path.splitext(base)
    return stem if stem else base


def _load_csv_blend_preset_map(csv_path: str) -> Dict[str, Dict[int, int]]:
    preset_by_key: Dict[str, Dict[int, int]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            frame_raw = row.get("frame") or row.get("frame_idx") or row.get("frame_index") or ""
            preset_raw = row.get("best_preset") or row.get("winner_preset") or row.get("preset") or ""
            try:
                frame_idx = int(float(str(frame_raw).strip()))
                preset_id = int(float(str(preset_raw).strip()))
            except Exception:
                continue
            if frame_idx < 0:
                continue
            if preset_id not in CT_PRESET_BY_ID:
                continue

            status = str(row.get("status", "") or "").strip().lower()
            if status and status not in {"ok", "done", "complete"}:
                # Keep deterministic fallback behavior for non-CT rows (e.g. low_mask).
                continue

            keys: List[str] = []
            for col in (
                "video",
                "inpainted",
                "filename",
                "clip",
                "core_with_width",
                "core_name",
            ):
                v = str(row.get(col, "") or "").strip()
                if v:
                    keys.append(v)

            for key in keys:
                normalized = _normalize_csv_blend_lookup_key(key)
                if normalized:
                    preset_by_key.setdefault(normalized, {})[frame_idx] = preset_id
                parsed_core_with_width, parsed_core_name, _ = _parse_inpainted_basename(
                    os.path.basename(key)
                )
                for alias in (parsed_core_with_width, parsed_core_name):
                    alias_norm = _normalize_csv_blend_lookup_key(alias)
                    if alias_norm:
                        preset_by_key.setdefault(alias_norm, {})[frame_idx] = preset_id
    return preset_by_key


def _lookup_csv_blend_preset_rows(
    preset_by_key: Dict[str, Dict[int, int]],
    inpainted_path: str,
    core_with_width: Optional[str],
    core_name: Optional[str],
) -> Tuple[Dict[int, int], str]:
    candidates: List[str] = []
    base_name = os.path.basename(str(inpainted_path or ""))
    if base_name:
        candidates.append(base_name)
        candidates.append(os.path.splitext(base_name)[0])
        parsed_core_with_width, parsed_core_name, _ = _parse_inpainted_basename(base_name)
        if parsed_core_with_width:
            candidates.append(parsed_core_with_width)
        if parsed_core_name:
            candidates.append(parsed_core_name)
    if core_with_width:
        candidates.append(core_with_width)
    if core_name:
        candidates.append(core_name)

    best_map: Dict[int, int] = {}
    best_key = ""
    for key in candidates:
        normalized = _normalize_csv_blend_lookup_key(key)
        if not normalized:
            continue
        rows = preset_by_key.get(normalized)
        if rows and len(rows) > len(best_map):
            best_map = rows
            best_key = normalized
    return dict(best_map), best_key


def _compute_preset_oscillator_flags(
    seq: List[int], min_len: int = 4
) -> List[bool]:
    flags = [False for _ in seq]
    n = len(seq)
    if n < max(4, int(min_len)):
        return flags

    i = 0
    while i + 3 < n:
        a = int(seq[i])
        b = int(seq[i + 1])
        if a == b:
            i += 1
            continue
        if int(seq[i + 2]) != a or int(seq[i + 3]) != b:
            i += 1
            continue

        j = i
        while j < n:
            expected = a if ((j - i) % 2 == 0) else b
            if int(seq[j]) != expected:
                break
            j += 1
        if (j - i) >= int(min_len):
            for k in range(i, j):
                flags[k] = True
            i = j
            continue
        i += 1

    return flags


def _build_csv_blend_weights_by_frame(
    target_preset_ids: List[int],
    transition_alphas: Optional[List[float]] = None,
    osc_alpha: float = CT_CSV_BLEND_OSC_ALPHA,
    max_active_presets: int = CT_CSV_BLEND_MAX_ACTIVE_PRESETS,
    prune_eps: float = CT_CSV_BLEND_PRUNE_EPS,
) -> Tuple[List[Dict[int, float]], List[bool]]:
    if transition_alphas is None or not transition_alphas:
        transition_alphas = list(CT_CSV_BLEND_TRANSITION_ALPHAS)

    if not target_preset_ids:
        return [], []

    out_weights: List[Dict[int, float]] = []
    weights: Dict[int, float] = {}
    run_pid: Optional[int] = None
    run_len = 0

    osc_flags = _compute_preset_oscillator_flags(target_preset_ids)

    for idx, pid_raw in enumerate(target_preset_ids):
        pid = int(pid_raw)
        if pid not in CT_PRESET_BY_ID:
            pid = int(CT_PRESET_AUTO_EVAL_ORDER[0])

        if run_pid == pid:
            run_len += 1
        else:
            run_pid = pid
            run_len = 1

        is_osc = bool(osc_flags[idx]) if idx < len(osc_flags) else False

        if is_osc:
            alpha = float(osc_alpha)
        elif run_len <= len(transition_alphas):
            alpha = float(transition_alphas[run_len - 1])
        else:
            alpha = 1.0
        alpha = float(max(0.0, min(1.0, alpha)))

        if not weights:
            weights = {pid: 1.0}
        elif alpha >= 0.999999:
            weights = {pid: 1.0}
        else:
            for k in list(weights.keys()):
                weights[k] = float(weights[k]) * (1.0 - alpha)
            weights[pid] = float(weights.get(pid, 0.0)) + alpha

        # Prune tiny tails.
        weights = {int(k): float(v) for k, v in weights.items() if float(v) > float(prune_eps)}
        if not weights:
            weights = {pid: 1.0}

        # Hard cap on active presets in one frame.
        if len(weights) > int(max_active_presets):
            top_items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[
                : int(max_active_presets)
            ]
            weights = {int(k): float(v) for k, v in top_items}

        s = float(sum(weights.values()))
        if s <= 0.0:
            weights = {pid: 1.0}
            s = 1.0
        out_weights.append({int(k): float(v) / s for k, v in weights.items()})

    return out_weights, osc_flags


def _build_auto_ct_eval_groups(
    preset_order: List[int],
) -> Tuple[List[List[int]], List[int]]:
    """
    Build fixed groups for auto-eval:
    - parallel: ring, nonmask, global (max 3 workers)
    - serial: legacy/other
    """
    buckets: Dict[str, List[int]] = {"ring": [], "nonmask": [], "global": []}
    serial_ids: List[int] = []
    for pid in preset_order:
        preset = CT_PRESET_BY_ID[int(pid)]
        mode = str(preset.get("mode", "safe"))
        if mode == "legacy":
            serial_ids.append(int(pid))
            continue
        sr = str(preset.get("stats_region", "ring"))
        if sr in buckets:
            buckets[sr].append(int(pid))
        else:
            serial_ids.append(int(pid))

    parallel_groups: List[List[int]] = []
    for key in ("ring", "nonmask", "global"):
        if buckets[key]:
            parallel_groups.append(buckets[key])
    return parallel_groups, serial_ids


CT_AUTO_EVAL_PARALLEL_GROUPS, CT_AUTO_EVAL_SERIAL_IDS = _build_auto_ct_eval_groups(
    CT_PRESET_AUTO_EVAL_ORDER
)


def _is_better_auto_ct_candidate(
    score: float,
    preset_id: int,
    best_score: float,
    best_preset_id: int,
    order_index: Dict[int, int],
) -> bool:
    if score > best_score:
        return True
    if score < best_score:
        return False
    # Stable tie-break: preserve canonical auto-eval order.
    return order_index.get(int(preset_id), 10**9) < order_index.get(
        int(best_preset_id), 10**9
    )


def _eval_auto_ct_subset(
    preset_ids: List[int],
    inpainted_3: torch.Tensor,
    original_left_3: torch.Tensor,
    warped_3: torch.Tensor,
    mask_bin_1hw: torch.Tensor,
    settings: Dict[str, Any],
    eval_ref_lab: Optional[np.ndarray],
    mask_bool: np.ndarray,
    order_index: Dict[int, int],
) -> Tuple[float, torch.Tensor, int]:
    stats_valid_cache: Dict[str, torch.Tensor] = {}
    warped_ref_cache: Dict[str, torch.Tensor] = {}

    best_score = -1.0
    best_frame = inpainted_3
    best_preset_id = int(preset_ids[0]) if preset_ids else int(CT_PRESET_AUTO_EVAL_ORDER[0])

    for pid in preset_ids:
        preset = CT_PRESET_BY_ID[int(pid)]
        adjusted_3 = _apply_ct_preset_frame(
            preset=preset,
            inpainted_3=inpainted_3,
            original_left_3=original_left_3,
            warped_3=warped_3,
            mask_bin_1hw=mask_bin_1hw,
            settings=settings,
            stats_valid_cache=stats_valid_cache,
            warped_ref_cache=warped_ref_cache,
        )
        if eval_ref_lab is not None:
            score = _masked_delta_e_score(
                pred_rgb01=_tensor_chw_to_rgb_np01(adjusted_3),
                ref_lab32=eval_ref_lab,
                mask_bool=mask_bool,
            )
        else:
            score = 0.0
        if _is_better_auto_ct_candidate(
            score,
            int(pid),
            best_score,
            best_preset_id,
            order_index,
        ):
            best_score = score
            best_frame = adjusted_3
            best_preset_id = int(pid)

    return best_score, best_frame, best_preset_id


def _select_best_auto_ct_preset_frame(
    inpainted_3: torch.Tensor,
    original_left_3: torch.Tensor,
    warped_3: torch.Tensor,
    mask_bin_1hw: torch.Tensor,
    settings: Dict[str, Any],
    fallback_preset_id: int,
    candidate_preset_ids: Optional[List[int]] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[torch.Tensor, int]:
    mask_bool = mask_bin_1hw.squeeze(0).cpu().numpy() > 0.5
    if int(mask_bool.sum()) > 0:
        mm_u8 = mask_bool.astype(np.uint8) * 255
        warped_u8 = (np.clip(_tensor_chw_to_rgb_np01(warped_3), 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
        eval_ref_u8 = _build_ring_shift_reference(warped_u8, mm_u8, border_tol_px=2)
        eval_ref_lab = _rgb01_to_lab32(eval_ref_u8.astype(np.float32) / 255.0)
    else:
        eval_ref_lab = None

    if candidate_preset_ids:
        allowed = {int(pid) for pid in candidate_preset_ids if int(pid) in CT_PRESET_BY_ID}
        eval_order = [int(pid) for pid in CT_PRESET_AUTO_EVAL_ORDER if int(pid) in allowed]
    else:
        eval_order = [int(pid) for pid in CT_PRESET_AUTO_EVAL_ORDER]
    if not eval_order:
        eval_order = [int(CT_PRESET_AUTO_EVAL_ORDER[0])]

    order_index = {int(pid): i for i, pid in enumerate(eval_order)}
    parallel_groups, serial_ids = _build_auto_ct_eval_groups(eval_order)
    best_score = -1.0
    best_frame = inpainted_3
    best_preset_id = int(fallback_preset_id)
    if best_preset_id not in order_index:
        best_preset_id = int(eval_order[0])

    def _consider(candidate: Tuple[float, torch.Tensor, int]) -> None:
        nonlocal best_score, best_frame, best_preset_id
        score, frame, pid = candidate
        if _is_better_auto_ct_candidate(
            score, int(pid), best_score, best_preset_id, order_index
        ):
            best_score = score
            best_frame = frame
            best_preset_id = int(pid)

    if executor is not None and parallel_groups:
        futures = [
            executor.submit(
                _eval_auto_ct_subset,
                preset_ids=group,
                inpainted_3=inpainted_3,
                original_left_3=original_left_3,
                warped_3=warped_3,
                mask_bin_1hw=mask_bin_1hw,
                settings=settings,
                eval_ref_lab=eval_ref_lab,
                mask_bool=mask_bool,
                order_index=order_index,
            )
            for group in parallel_groups
        ]
        for fut in futures:
            _consider(fut.result())
    else:
        for group in parallel_groups:
            _consider(
                _eval_auto_ct_subset(
                    preset_ids=group,
                    inpainted_3=inpainted_3,
                    original_left_3=original_left_3,
                    warped_3=warped_3,
                    mask_bin_1hw=mask_bin_1hw,
                    settings=settings,
                    eval_ref_lab=eval_ref_lab,
                    mask_bool=mask_bool,
                    order_index=order_index,
                )
            )

    if serial_ids:
        _consider(
            _eval_auto_ct_subset(
                preset_ids=serial_ids,
                inpainted_3=inpainted_3,
                original_left_3=original_left_3,
                warped_3=warped_3,
                mask_bin_1hw=mask_bin_1hw,
                settings=settings,
                eval_ref_lab=eval_ref_lab,
                mask_bool=mask_bool,
                order_index=order_index,
            )
        )

    return best_frame, best_preset_id


def _resolve_ct_preset_label(value: str) -> str:
    v = str(value or "").strip()
    if v in CT_PRESET_BY_LABEL:
        return v
    return CT_PRESET_DEFAULT_LABEL


def _parse_ct_preset_arg(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return CT_PRESET_DEFAULT_LABEL
    if v.isdigit():
        pid = int(v)
        if pid in CT_PRESET_BY_ID:
            return CT_PRESET_BY_ID[pid]["label"]
    return _resolve_ct_preset_label(v)


def _tensor_chw_to_rgb_np01(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float()
    if x.dim() == 4 and x.shape[0] == 1:
        x = x[0]
    return np.clip(x.permute(1, 2, 0).numpy().astype(np.float32), 0.0, 1.0)


def _rgb01_to_lab32(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb01.astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float32)


def _masked_delta_e_score(
    pred_rgb01: np.ndarray, ref_lab32: np.ndarray, mask_bool: np.ndarray
) -> float:
    n = int(mask_bool.sum())
    if n <= 0:
        return 0.0
    pred_lab = _rgb01_to_lab32(pred_rgb01)
    diff = pred_lab - ref_lab32
    de = np.sqrt(np.sum(diff * diff, axis=2))
    de_mean = float(de[mask_bool].mean())
    return 1.0 / (1.0 + max(0.0, de_mean))


def _build_ring_shift_reference(
    frame_rgb_u8: np.ndarray, mask_u8: np.ndarray, border_tol_px: int = 2
) -> np.ndarray:
    """
    Build reference by row/run ring shift:
    - default: copy adjacent block from right side into mask run
    - runs touching right border (with tolerance): copy from left side
    Fallbacks keep behavior robust near edges.
    """
    if frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
        return frame_rgb_u8
    if mask_u8.ndim != 2:
        return frame_rgb_u8
    h, w = mask_u8.shape
    if h != frame_rgb_u8.shape[0] or w != frame_rgb_u8.shape[1]:
        return frame_rgb_u8

    frame_t = torch.as_tensor(frame_rgb_u8, dtype=torch.uint8, device="cpu")
    mask_t = torch.as_tensor(mask_u8, device="cpu")
    mask_bin = mask_t > 0
    if not bool(mask_bin.any().item()):
        return frame_rgb_u8.copy()

    out = frame_t.clone()
    border_cols = max(1, min(int(border_tol_px), w))
    right_touch_start = w - border_cols

    def _nearest_nonmask_x(y: int, start_x: int, step: int) -> int:
        x = int(start_x)
        while 0 <= x < w:
            if not bool(mask_bin[y, x].item()):
                return x
            x += step
        return -1

    def _copy_block_if_valid(y: int, dst_a: int, dst_b: int, src_a: int, src_b: int) -> bool:
        if src_a < 0 or src_b > w or src_a >= src_b:
            return False
        if bool(mask_bin[y, src_a:src_b].any().item()):
            return False
        out[y, dst_a : dst_b + 1, :] = frame_t[y, src_a:src_b, :]
        return True

    for y in range(h):
        xs = torch.where(mask_bin[y])[0]
        if int(xs.numel()) == 0:
            continue
        run_start = int(xs[0].item())
        prev = run_start

        def _paint_run(a: int, b: int) -> None:
            run_len = int(b - a + 1)
            touches_left = a < border_cols
            touches_right = b >= right_touch_start
            prefer_left = touches_right and not touches_left

            copied = False
            if prefer_left:
                copied = _copy_block_if_valid(y, a, b, a - run_len, a)
                if not copied:
                    copied = _copy_block_if_valid(y, a, b, b + 1, b + 1 + run_len)
            else:
                copied = _copy_block_if_valid(y, a, b, b + 1, b + 1 + run_len)
                if not copied:
                    copied = _copy_block_if_valid(y, a, b, a - run_len, a)
            if copied:
                return

            if prefer_left:
                src_x = _nearest_nonmask_x(y, a - 1, -1)
                if src_x < 0:
                    src_x = _nearest_nonmask_x(y, b + 1, 1)
            else:
                src_x = _nearest_nonmask_x(y, b + 1, 1)
                if src_x < 0:
                    src_x = _nearest_nonmask_x(y, a - 1, -1)
            if src_x >= 0:
                out[y, a : b + 1, :] = frame_t[y, src_x : src_x + 1, :]

        for idx in range(1, int(xs.numel())):
            cur = int(xs[idx].item())
            if cur != prev + 1:
                _paint_run(run_start, prev)
                run_start = cur
            prev = cur
        _paint_run(run_start, prev)

    return out.numpy()


def _apply_ct_preset_frame(
    preset: Dict[str, Any],
    inpainted_3: torch.Tensor,
    original_left_3: torch.Tensor,
    warped_3: torch.Tensor,
    mask_bin_1hw: torch.Tensor,
    settings: Dict[str, Any],
    stats_valid_cache: Dict[str, torch.Tensor],
    warped_ref_cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    mode = str(preset.get("mode", "safe"))
    if mode == "legacy":
        return apply_color_transfer(original_left_3.cpu(), inpainted_3.cpu())

    stats_region = str(preset.get("stats_region", "ring"))
    if stats_region not in stats_valid_cache:
        stats_valid_cache[stats_region] = _make_stats_mask(
            mask_bin_1hw,
            stats_region=stats_region,
            ring_width=int(settings.get("ct_ring_width", 40)),
            use_gpu=False,
        )
    stats_valid = stats_valid_cache[stats_region]

    if str(preset.get("target_stats_source", "inpainted")) == "warped":
        tgt_stats = warped_3.cpu()
    else:
        tgt_stats = inpainted_3.cpu()

    ref_src = str(preset.get("reference_source", "left"))
    if ref_src == "warped_filled":
        if stats_region != "global":
            ref = warped_3.cpu()
        else:
            fill_mode = str(preset.get("warped_fill_mode", "directional"))
            if fill_mode not in warped_ref_cache:
                wf = warped_3.cpu()
                wf_u8 = (
                    torch.clamp(wf, 0, 1).permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                mm = (mask_bin_1hw.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                ref_u8 = _build_warped_filled_reference(
                    wf_u8,
                    mm,
                    fill_mode=fill_mode,
                    border_tol_px=2,
                    telea_radius=3,
                )
                warped_ref_cache[fill_mode] = (
                    torch.from_numpy(ref_u8).permute(2, 0, 1).float() / 255.0
                )
            ref = warped_ref_cache[fill_mode]
    else:
        ref = original_left_3.cpu()

    return apply_color_transfer_safe(
        ref,
        inpainted_3.cpu(),
        black_thresh=float(settings.get("ct_black_thresh", 0.0)),
        min_valid_ratio=float(settings.get("ct_min_valid_ratio", 0.0)),
        min_valid=int(settings.get("ct_min_valid", 0)),
        strength=float(settings.get("ct_strength", 1.0)),
        clamp_scale_L=(
            float(settings.get("ct_clamp_L_min", 0.1)),
            float(settings.get("ct_clamp_L_max", 2.0)),
        ),
        clamp_scale_ab=(
            float(settings.get("ct_clamp_ab_min", 0.1)),
            float(settings.get("ct_clamp_ab_max", 3.0)),
        ),
        exclude_black_in_target=bool(settings.get("ct_exclude_black_in_target", True)),
        source_valid_mask=stats_valid,
        target_valid_mask=stats_valid,
        target_stats_frame=tgt_stats,
    )


class MergingGUI(ThemedTk):
    MOTION_DEFAULTS_CONFIG_PATH = "config_merging_gui_motion_defaults.json"
    PREVIEW_SHADOW_WARMUP_MAX_FRAMES = 20
    PREVIEW_SHADOW_WARMUP_RESIDUAL = 0.05
    OUTPUT_FORMAT_CHOICES = [
        "Full SBS (Left-Right)",
        "Double SBS",
        "Half SBS (Left-Right)",
        "Full SBS Cross-eye (Right-Left)",
        "Anaglyph (Red/Cyan)",
        "Anaglyph Half-Color",
    ]
    MOTION_DEFAULTS_FALLBACK = {
        "shadow_motion_gain": 1.0,
        "shadow_motion_enabled": True,
        "shadow_motion_deadzone_px": 20.0,
        "shadow_motion_max_px": 40.0,
        "shadow_area_min_px": 1000.0,
        "shadow_area_max_px": 2000.0,
        "shadow_area_reset_ratio": 1.65,
        "shadow_area_reset_abs_px": 0.0,
        "shadow_component_merge_y_tol_px": 4,
        "shadow_alpha_down": 0.45,
    }

    # --- Centralized Default Settings ---
    APP_DEFAULTS = {
        "inpainted_folder": "./completed_output",
        "original_folder": "./input_source_clips",
        "mask_folder": "./output_splatted/hires",
        "replace_mask_folder": "",  # optional; if empty uses splatted folder
        "mask_formerge_folder": "./work/mask_for_merge",
        "output_folder": "./final_videos",
        "use_replace_mask": True,
        "use_mask_formerge": False,
        "mask_binarize_threshold": -0.01,
        "mask_dilate_kernel_size": 2,
        "mask_blur_kernel_size": 4,
        "shadow_length_px": 30,
        "shadow_width_adaptive": True,
        "shadow_curve": 0.0,
        "shadow_motion_gain": 1.0,  # hidden in GUI, managed via motion defaults
        "shadow_motion_enabled": True,
        "shadow_motion_deadzone_px": 20.0,
        "shadow_motion_max_px": 40.0,
        "shadow_area_min_px": 1000.0,
        "shadow_area_max_px": 2000.0,
        "shadow_area_reset_ratio": 1.65,
        "shadow_area_reset_abs_px": 0.0,
        "shadow_component_merge_y_tol_px": 4,
        "shadow_alpha_down": 0.45,
        "preview_shadow_temporal": False,
        "use_gpu": True,
        "output_format": "Full SBS (Left-Right)",
        "pad_to_16_9": False,
        "add_borders": False,
        "enable_color_transfer": True,
        "ct_preset": "1) safe sr=ring ts=inpainted ref=warped",
        "auto_ct_eval": True,
        "ct_auto_mode": CT_AUTO_MODE_ON,
        "ct_csv_blend_path": "",
        "show_blend_in_preview": True,
        "ct_advanced": False,
        "ct_strength": 1.0,
        "ct_black_thresh": 0.0,
        "ct_min_valid_ratio": 0,
        "ct_min_valid": 0,
        "ct_clamp_L_min": 0.1,
        "ct_clamp_L_max": 2,
        "ct_clamp_ab_min": 0.1,
        "ct_clamp_ab_max": 3,
        "ct_exclude_black_in_target": True,
        "ct_ring_width": 20,

        "batch_chunk_size": "20",
        "preview_size": "100%",
        "debug_logging_enabled": True,
    }

    def __init__(self):
        super().__init__(theme="clam")
        self.title(f"Stereocrafter Merging GUI {GUI_VERSION}")
        self.app_config = self._load_config()
        self.motion_defaults = self._load_motion_defaults()
        self.help_data = self._load_help_texts()

        # --- Sidecar Config Manager ---
        self.sidecar_manager = SidecarConfigManager()

        # --- Window Geometry ---
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get(
            "window_width", 700
        )  # A reasonable default
        self.window_height = self.app_config.get(
            "window_height", 800
        )  # A reasonable default

        # --- Core App State ---
        self.stop_event = threading.Event()
        self.is_processing = False
        self.cleanup_queue = queue.Queue()

        self._is_startup = True  # Flag to prevent resizing during initialization
        self.preview_original_left_tensor = None
        self.preview_blended_right_tensor = None
        self._preview_shadow_temporal_cache: Optional[Dict[str, Any]] = None
        # --- GUI Variables ---
        self.pil_image_for_preview = None
        self.inpainted_folder_var = tk.StringVar(
            value=self.app_config.get(
                "inpainted_folder", self.APP_DEFAULTS["inpainted_folder"]
            )
        )
        self.inpainted_folder_var.trace_add("write", self._on_folder_changed)
        self.original_folder_var = tk.StringVar(
            value=self.app_config.get(
                "original_folder", self.APP_DEFAULTS["original_folder"]
            )
        )
        self.original_folder_var.trace_add("write", self._on_folder_changed)
        self.mask_folder_var = tk.StringVar(
            value=self.app_config.get("mask_folder", self.APP_DEFAULTS["mask_folder"])
        )
        self.mask_folder_var.trace_add("write", self._on_folder_changed)

        self.replace_mask_folder_var = tk.StringVar(
            value=str(
                self.app_config.get(
                    "replace_mask_folder",
                    self.APP_DEFAULTS.get("replace_mask_folder", ""),
                )
            )
        )
        self.replace_mask_folder_var.trace_add("write", self._on_folder_changed)
        self.mask_formerge_folder_var = tk.StringVar(
            value=str(
                self.app_config.get(
                    "mask_formerge_folder",
                    self.APP_DEFAULTS.get("mask_formerge_folder", ""),
                )
            )
        )
        self.mask_formerge_folder_var.trace_add("write", self._on_folder_changed)


        # --- Optional: Use external replace-mask video instead of embedded splat mask ---
        self.use_replace_mask_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "use_replace_mask", self.APP_DEFAULTS.get("use_replace_mask", False)
                )
            )
        )
        self.use_mask_formerge_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "use_mask_formerge",
                    self.APP_DEFAULTS.get("use_mask_formerge", False),
                )
            )
        )
        self.output_folder_var = tk.StringVar(
            value=self.app_config.get(
                "output_folder", self.APP_DEFAULTS["output_folder"]
            )
        )

        # --- Mask Processing Parameters ---
        self.mask_binarize_threshold_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "mask_binarize_threshold",
                    self.APP_DEFAULTS["mask_binarize_threshold"],
                )
            )
        )
        self.mask_dilate_kernel_size_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "mask_dilate_kernel_size",
                    self.APP_DEFAULTS["mask_dilate_kernel_size"],
                )
            )
        )
        self.mask_blur_kernel_size_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "mask_blur_kernel_size", self.APP_DEFAULTS["mask_blur_kernel_size"]
                )
            )
        )
        # Backward compatibility: infer approximate length from legacy decay if needed.
        legacy_length_fallback = float(self.APP_DEFAULTS["shadow_length_px"])
        if "shadow_length_px" not in self.app_config:
            try:
                legacy_decay = float(self.app_config.get("shadow_opacity_decay", 0.0))
                if legacy_decay > 1e-6:
                    legacy_length_fallback = max(
                        0.0, min(200.0, 1.0 / float(legacy_decay))
                    )
            except Exception:
                pass
        self.shadow_length_px_var = tk.DoubleVar(
            value=float(
                self.app_config.get("shadow_length_px", legacy_length_fallback)
            )
        )
        self.shadow_width_adaptive_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "shadow_width_adaptive",
                    self.APP_DEFAULTS["shadow_width_adaptive"],
                )
            )
        )
        self.shadow_curve_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_curve",
                    self.APP_DEFAULTS["shadow_curve"],
                )
            )
        )
        motion_gain_default = float(
            self.motion_defaults.get(
                "shadow_motion_gain",
                self.APP_DEFAULTS.get("shadow_motion_gain", 1.0),
            )
        )
        self.shadow_motion_gain_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_motion_gain",
                    motion_gain_default,
                )
            )
        )
        motion_enabled_default = bool(
            self.motion_defaults.get(
                "shadow_motion_enabled",
                self.APP_DEFAULTS.get("shadow_motion_enabled", True),
            )
        )
        self.shadow_motion_enabled_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "shadow_motion_enabled",
                    motion_enabled_default,
                )
            )
        )
        motion_deadzone_default = float(
            self.motion_defaults.get(
                "shadow_motion_deadzone_px",
                self.APP_DEFAULTS.get("shadow_motion_deadzone_px", 20.0),
            )
        )
        self.shadow_motion_deadzone_px_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_motion_deadzone_px",
                    motion_deadzone_default,
                )
            )
        )
        motion_max_default = float(
            self.motion_defaults.get(
                "shadow_motion_max_px",
                self.APP_DEFAULTS.get("shadow_motion_max_px", 40.0),
            )
        )
        self.shadow_motion_max_px_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_motion_max_px",
                    motion_max_default,
                )
            )
        )
        area_min_default = float(
            self.motion_defaults.get(
                "shadow_area_min_px",
                self.APP_DEFAULTS.get("shadow_area_min_px", 1000.0),
            )
        )
        self.shadow_area_min_px_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_area_min_px",
                    area_min_default,
                )
            )
        )
        area_max_default = float(
            self.motion_defaults.get(
                "shadow_area_max_px",
                self.APP_DEFAULTS.get("shadow_area_max_px", 2000.0),
            )
        )
        self.shadow_area_max_px_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_area_max_px",
                    area_max_default,
                )
            )
        )
        area_reset_ratio_default = float(
            self.motion_defaults.get(
                "shadow_area_reset_ratio",
                self.APP_DEFAULTS.get("shadow_area_reset_ratio", 1.65),
            )
        )
        self.shadow_area_reset_ratio_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_area_reset_ratio",
                    area_reset_ratio_default,
                )
            )
        )
        area_reset_abs_default = float(
            self.motion_defaults.get(
                "shadow_area_reset_abs_px",
                self.APP_DEFAULTS.get("shadow_area_reset_abs_px", 0.0),
            )
        )
        self.shadow_area_reset_abs_px_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_area_reset_abs_px",
                    area_reset_abs_default,
                )
            )
        )
        component_y_tol_default = int(
            self.motion_defaults.get(
                "shadow_component_merge_y_tol_px",
                self.APP_DEFAULTS.get("shadow_component_merge_y_tol_px", 4),
            )
        )
        self.shadow_component_merge_y_tol_px_var = tk.IntVar(
            value=int(
                self.app_config.get(
                    "shadow_component_merge_y_tol_px",
                    component_y_tol_default,
                )
            )
        )
        alpha_down_default = float(
            self.motion_defaults.get(
                "shadow_alpha_down",
                self.APP_DEFAULTS.get("shadow_alpha_down", 0.45),
            )
        )
        self.shadow_alpha_down_var = tk.DoubleVar(
            value=float(
                self.app_config.get(
                    "shadow_alpha_down",
                    alpha_down_default,
                )
            )
        )
        self.preview_shadow_temporal_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "preview_shadow_temporal",
                    self.APP_DEFAULTS["preview_shadow_temporal"],
                )
            )
        )

        self.use_gpu_var = tk.BooleanVar(
            value=self.app_config.get("use_gpu", self.APP_DEFAULTS["use_gpu"])
        )
        self.output_format_var = tk.StringVar(
            value=self.app_config.get(
                "output_format", self.APP_DEFAULTS["output_format"]
            )
        )
        if self.output_format_var.get() not in self.OUTPUT_FORMAT_CHOICES:
            self.output_format_var.set(self.APP_DEFAULTS["output_format"])
        self.pad_to_16_9_var = tk.BooleanVar(
            value=self.app_config.get("pad_to_16_9", self.APP_DEFAULTS["pad_to_16_9"])
        )
        self.enable_color_transfer_var = tk.BooleanVar(
            value=self.app_config.get(
                "enable_color_transfer", self.APP_DEFAULTS["enable_color_transfer"]
            )
                )

        # --- Color Transfer (Preset + Safe controls) ---
        self.ct_preset_var = tk.StringVar(
            value=_resolve_ct_preset_label(
                self.app_config.get(
                    "ct_preset",
                    self.APP_DEFAULTS["ct_preset"],
                )
            )
        )
        legacy_auto_ct_eval = bool(
            self.app_config.get(
                "auto_ct_eval", self.APP_DEFAULTS.get("auto_ct_eval", True)
            )
        )
        ct_auto_mode_raw = self.app_config.get(
            "ct_auto_mode",
            self.APP_DEFAULTS.get(
                "ct_auto_mode",
                CT_AUTO_MODE_ON if legacy_auto_ct_eval else CT_AUTO_MODE_OFF,
            ),
        )
        self.ct_auto_mode_var = tk.StringVar(
            value=_resolve_ct_auto_mode_label(ct_auto_mode_raw)
        )
        # Kept for backward compatibility with saved settings.
        self.auto_ct_eval_var = tk.BooleanVar(
            value=(self.ct_auto_mode_var.get() == CT_AUTO_MODE_ON)
        )
        self.ct_csv_blend_path_var = tk.StringVar(
            value=str(
                self.app_config.get(
                    "ct_csv_blend_path",
                    self.APP_DEFAULTS.get("ct_csv_blend_path", ""),
                )
            )
        )
        self.show_blend_in_preview_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "show_blend_in_preview",
                    self.APP_DEFAULTS.get("show_blend_in_preview", True),
                )
            )
        )
        self.ct_advanced_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "ct_advanced",
                    self.APP_DEFAULTS.get("ct_advanced", False),
                )
            )
        )
        self.auto_ct_best_var = tk.StringVar(
            value=(
                "Auto CT best: pending..."
                if self.ct_auto_mode_var.get() == CT_AUTO_MODE_ON
                else (
                    "Auto CT CSV Blend: pending..."
                    if self.ct_auto_mode_var.get() == CT_AUTO_MODE_CSV_BLEND
                    else "Auto CT best: (disabled)"
                )
            )
        )
        self._ct_csv_blend_cache_path = ""
        self._ct_csv_blend_cache_mtime = -1.0
        self._ct_csv_blend_cache: Dict[str, Dict[int, int]] = {}
        self.ct_strength_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_strength", self.APP_DEFAULTS["ct_strength"]))
        )
        self.ct_black_thresh_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_black_thresh", self.APP_DEFAULTS["ct_black_thresh"]))
        )
        self.ct_min_valid_ratio_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_min_valid_ratio", self.APP_DEFAULTS["ct_min_valid_ratio"]))
        )
        self.ct_min_valid_var = tk.IntVar(
            value=int(self.app_config.get("ct_min_valid", self.APP_DEFAULTS["ct_min_valid"]))
        )
        self.ct_clamp_L_min_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_clamp_L_min", self.APP_DEFAULTS["ct_clamp_L_min"]))
        )
        self.ct_clamp_L_max_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_clamp_L_max", self.APP_DEFAULTS["ct_clamp_L_max"]))
        )
        self.ct_clamp_ab_min_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_clamp_ab_min", self.APP_DEFAULTS["ct_clamp_ab_min"]))
        )
        self.ct_clamp_ab_max_var = tk.DoubleVar(
            value=float(self.app_config.get("ct_clamp_ab_max", self.APP_DEFAULTS["ct_clamp_ab_max"]))
        )
        self.ct_exclude_black_in_target_var = tk.BooleanVar(
            value=bool(self.app_config.get("ct_exclude_black_in_target", self.APP_DEFAULTS["ct_exclude_black_in_target"]))
        )
        self.ct_ring_width_var = tk.IntVar(
            value=int(self.app_config.get("ct_ring_width", self.APP_DEFAULTS["ct_ring_width"]))
        )
        # Ensure preset is normalized early.
        self._apply_ct_preset_to_controls(self.ct_preset_var.get())
        # --- END Color Transfer (Preset + Safe controls) ---
        # Debug build: start verbose logging by default every run.
        self.debug_logging_var = tk.BooleanVar(value=True)
        self.dark_mode_var = tk.BooleanVar(
            value=self.app_config.get("dark_mode_enabled", False)
        )
        self.batch_chunk_size_var = tk.StringVar(
            value=str(
                self.app_config.get(
                    "batch_chunk_size", self.APP_DEFAULTS["batch_chunk_size"]
                )
            )
        )
        self.preview_source_var = tk.StringVar(
            value=self.app_config.get("preview_source", "Blended Image")
        )
        self.preview_size_var = tk.StringVar(
            value=str(self.app_config.get("preview_size", "100%"))
        )

        # --- GUI Status Variables ---
        self.slider_label_updaters = []
        # --- END FIX ---
        self.progress_var = tk.DoubleVar(value=0)
        self.widgets_to_disable = []
        self._is_refreshing_mode_constraints = False
        self._last_mode_constraints_video_index = -999999

        self.create_widgets()

        # Define a custom style for the loading button
        self.style = ttk.Style(self)
        self.style.configure("Loading.TButton", foreground="red")

        self._apply_theme()
        self._configure_logging()  # Set initial logging level
        self.after(
            0, lambda: setattr(self, "_is_startup", False)
        )  # Set startup flag to false after GUI is built
        self.after(0, self._set_saved_geometry)  # Restore window position
        self.protocol("WM_DELETE_WINDOW", self.exit_application)

        # Call all the label updaters to set the initial text from the loaded config
        for updater in self.slider_label_updaters:
            updater()
        self._refresh_mode_constraints(trigger_preview=False)
        self.update_status_label("Ready.")

        # --- FIX: Initialize the previewer AFTER the main GUI is fully built ---
        # This ensures the previewer gets the correct initial slider values.
        # No longer needed, previewer will call get_current_settings() itself.
        pass

    def _set_saved_geometry(self):
        """
        Applies the saved window width, height, and position.
        """
        logger.debug("--- Setting Saved Geometry (Startup) ---")
        self.update_idletasks()

        # 1. Use the saved/default width and height, with fallbacks
        current_width = self.window_width
        current_height = self.window_height
        logger.debug(
            f"  - Using saved/default width: {current_width}, height: {current_height}"
        )

        if current_width < 500:  # Minimum sensible width
            current_width = 700
            logger.debug(f"  - Width was < 500, using fallback: {current_width}")
        if current_height < 400:  # Minimum sensible height
            current_height = 800
            logger.debug(f"  - Height was < 400, using fallback: {current_height}")

        # 2. Construct the geometry string
        geometry_string = f"{current_width}x{current_height}"
        if self.window_x is not None and self.window_y is not None:
            geometry_string += f"+{self.window_x}+{self.window_y}"
            logger.debug(f"  - Using saved position: +{self.window_x}+{self.window_y}")

        # 3. Apply the geometry
        self.geometry(geometry_string)
        logger.debug(f"  - Applied geometry string: '{geometry_string}'")
        logger.debug("--- End Setting Saved Geometry ---")

    def create_menubar(self):
        """Creates the main menu bar for the application."""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # --- File Menu ---
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(
            label="Load Settings...", command=self.load_settings_dialog
        )
        self.file_menu.add_command(
            label="Save Settings...", command=self.save_settings_dialog
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Save Preview Frame...",
            command=lambda: self.previewer.save_preview_frame(),
        )
        self.file_menu.add_command(
            label="Save Preview as SBS...", command=self._save_preview_sbs_frame
        )  # Keep this one here as it needs access to both eyes
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Reset to Default", command=self.reset_to_defaults
        )
        self.file_menu.add_command(
            label="Restore Finished Files", command=self.restore_finished_files
        )
        self.file_menu.add_separator()
        self.file_menu.add_checkbutton(
            label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_application)

        # --- Help Menu ---
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_checkbutton(
            label="Enable Debug Logging",
            variable=self.debug_logging_var,
            command=self._toggle_debug_logging,
        )
        self.help_menu.add_separator()
        self.help_menu.add_command(label="User Guide", command=self.show_user_guide)
        self.help_menu.add_command(label="About", command=self.show_about_dialog)

    def _create_hover_tooltip(self, widget, help_key):
        """Creates a mouse-over tooltip for the given widget."""
        if help_key in self.help_data:
            Tooltip(widget, self.help_data[help_key])

    def _apply_theme(self):
        """Applies the selected theme (dark or light) to the GUI."""
        if self.dark_mode_var.get():
            bg_color, fg_color, entry_bg = "#2b2b2b", "white", "#3c3c3c"
            self.style.theme_use("black")
        else:
            bg_color, fg_color, entry_bg = "#d9d9d9", "black", "#ffffff"
            self.style.theme_use("clam")

        self.configure(bg=bg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        self.style.configure(
            "TLabelframe.Label", background=bg_color, foreground=fg_color
        )
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        self.style.map(
            "TCheckbutton",
            foreground=[("active", fg_color)],
            background=[("active", bg_color)],
        )
        self.style.configure(
            "TEntry",
            fieldbackground=entry_bg,
            foreground=fg_color,
            insertcolor=fg_color,
        )
        # --- NEW: Add Combobox styling ---
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", entry_bg)],
            foreground=[("readonly", fg_color)],
            selectbackground=[("readonly", entry_bg)],
            selectforeground=[("readonly", fg_color)],
        )
        # Manually set the background for the previewer's canvas widget
        if hasattr(self, "previewer") and hasattr(self.previewer, "preview_canvas"):
            self.previewer.preview_canvas.config(bg=bg_color, highlightthickness=0)

        # --- FIX: Re-apply the custom loading button style after the theme changes ---
        # This ensures the red text color is not overridden by the theme's default button style.
        self.style.configure("Loading.TButton", foreground="red")

        # Adjust window height for new theme if not starting up
        if not self._is_startup:
            self._adjust_window_height_for_content()

    def show_about_dialog(self):
        """Displays an 'About' dialog for the application."""
        about_text = (
            f"Stereocrafter Merging GUI\n"
            f"Version: {GUI_VERSION}\n\n"
            "This tool blends inpainted right-eye videos with their corresponding "
            "high-resolution source files to create final stereoscopic videos.\n\n"
            "It provides interactive controls for mask processing and color matching."
        )
        messagebox.showinfo("About Merging GUI", about_text)

    def show_user_guide(self):
        """Reads and displays the user guide from a markdown file in a new window."""
        guide_path = os.path.join("assets", "merger_gui_guide.md")
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                guide_content = f.read()
        except FileNotFoundError:
            messagebox.showerror(
                "File Not Found",
                f"The user guide file could not be found at:\n{os.path.abspath(guide_path)}",
            )
            return
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred while reading the user guide:\n{e}"
            )
            return

        # Determine colors based on current theme
        if self.dark_mode_var.get():
            bg_color, fg_color = "#2b2b2b", "white"
        else:
            # Use a standard light bg for text that's slightly different from the main window
            bg_color, fg_color = "#fdfdfd", "black"

        # Create a new Toplevel window
        guide_window = tk.Toplevel(self)
        guide_window.title("Merging GUI - User Guide")
        guide_window.geometry("600x700")
        guide_window.transient(self)  # Keep it on top of the main window
        guide_window.grab_set()  # Modal behavior
        guide_window.configure(bg=bg_color)

        text_frame = ttk.Frame(guide_window, padding="10")
        text_frame.configure(style="TFrame")  # Ensure it follows the theme
        text_frame.pack(expand=True, fill="both")

        # Apply theme colors to the Text widget
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            relief="flat",
            borderwidth=0,
            padx=5,
            pady=5,
            font=("Segoe UI", 9),
            bg=bg_color,
            fg=fg_color,
            insertbackground=fg_color,
        )
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED)  # Make it read-only

        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=text_widget.yview
        )
        text_widget["yscrollcommand"] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, expand=True, fill="both")

        button_frame = ttk.Frame(guide_window, padding=(0, 0, 0, 10))
        button_frame.pack()
        ok_button = ttk.Button(button_frame, text="Close", command=guide_window.destroy)
        ok_button.pack(pady=10)

    def reset_to_defaults(self):
        """Resets all GUI parameters to their default values using the _apply_settings method."""
        if not messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?",
        ):
            return  # User cancelled

        self._apply_settings(self.APP_DEFAULTS)
        self.save_config()
        # messagebox.showinfo("Settings Reset", "All settings have been reset to their default values.")
        logger.info("GUI settings reset to defaults.")

    def _apply_settings(self, settings_dict: dict):
        """
        A centralized function to apply a dictionary of settings to the GUI's tk.Variables.
        This is used by both Load Settings and Reset to Defaults.
        """
        logger.debug(
            f"Applying settings dictionary:\n{json.dumps(settings_dict, indent=2)}"
        )
        for key, value in settings_dict.items():
            var_name = key + "_var"
            if hasattr(self, var_name):
                tk_var = getattr(self, var_name)
                try:
                    tk_var.set(value)
                except (ValueError, tk.TclError) as e:
                    logger.error(
                        f"Could not apply setting for '{key}' with value '{value}': {e}"
                    )

        if "ct_auto_mode" not in settings_dict and "auto_ct_eval" in settings_dict:
            self.ct_auto_mode_var.set(
                CT_AUTO_MODE_ON
                if bool(settings_dict.get("auto_ct_eval", False))
                else CT_AUTO_MODE_OFF
            )

        # Keep derived CT controls coherent after loading/applying settings.
        self._apply_ct_preset_to_controls(self.ct_preset_var.get())
        self._on_auto_ct_eval_toggle()
        self._toggle_ct_advanced_controls()

        # After setting all variables, manually update the slider labels to match.
        for updater in self.slider_label_updaters:
            updater()
        self._last_mode_constraints_video_index = -999999
        self._refresh_mode_constraints(trigger_preview=False)
        logger.info("Applied settings to GUI and updated labels.")

    def _toggle_ct_advanced_controls(self) -> None:
        """Show/hide advanced CT sliders based on the Advanced checkbox."""
        row = getattr(self, "_ct_sliders_row", None)
        if row is None:
            return
        try:
            # Keep the toplevel geometry stable when toggling advanced controls.
            current_w = int(self.winfo_width())
            current_h = int(self.winfo_height())
            if bool(self.ct_advanced_var.get()):
                row.grid()
            else:
                row.grid_remove()
            if current_w > 1 and current_h > 1:
                self.update_idletasks()
                self.geometry(f"{current_w}x{current_h}")
        except Exception:
            pass

    def _configure_logging(self):
        """Sets the logging level based on the debug_logging_var."""
        if self.debug_logging_var.get():
            level = logging.DEBUG
        else:
            level = logging.INFO

        set_util_logger_level(level)
        logging.getLogger().setLevel(level)
        logger.info(f"Logging level set to {logging.getLevelName(level)}.")

    def _adjust_window_height_for_content(self):
        """Adjusts the window height to fit the current content, preserving user-set width."""
        if self._is_startup:  # Don't adjust during initial setup
            return

        current_actual_width = self.winfo_width()
        if current_actual_width <= 1:  # Fallback for very first call
            current_actual_width = self.window_width

        # --- NEW: More accurate height calculation ---
        # --- FIX: Calculate base_height by summing widgets *other* than the previewer ---
        # This is more stable than subtracting a potentially out-of-sync canvas height.
        base_height = 0
        for widget in self.winfo_children():
            if widget is not self.previewer:
                # --- FIX: Correctly handle tuple and int for pady ---
                try:
                    pady_value = widget.pack_info().get("pady", 0)
                    total_pady = 0
                    if isinstance(pady_value, int):
                        total_pady = pady_value * 2
                    elif isinstance(pady_value, (tuple, list)):
                        total_pady = sum(pady_value)
                    base_height += widget.winfo_reqheight() + total_pady
                except tk.TclError:
                    # This widget (e.g., the menubar) is not packed, so it has no pady.
                    base_height += widget.winfo_reqheight()
        # --- END FIX ---

        # Get the actual height of the displayed preview image, if it exists
        preview_image_height = 0
        if (
            hasattr(self.previewer, "preview_image_tk")
            and self.previewer.preview_image_tk
        ):
            preview_image_height = self.previewer.preview_image_tk.height()

        # Add a small buffer for padding/borders
        padding = 10

        # The new total height is the base UI height + the actual image height + padding
        new_height = base_height + preview_image_height + padding
        # --- END NEW ---

        self.geometry(f"{current_actual_width}x{new_height}")
        logger.debug(
            f"Content resize applied geometry: {current_actual_width}x{new_height}"
        )

        # Update stored width and height for the next time save_config is called.
        self.window_width = current_actual_width

    def _toggle_debug_logging(self):
        """Callback for the debug logging checkbox."""
        self._configure_logging()
        self.save_config()

    def _load_help_texts(self):
        """Loads help texts from the dedicated JSON file."""
        try:
            with open(os.path.join("dependency", "merge_help.json"), "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            return int(default)

    @classmethod
    def _compute_preview_shadow_warmup_frames(cls, alpha_down: float) -> int:
        """Estimate temporal warmup frames from IIR decay (residual-based)."""
        a = float(alpha_down)
        if not np.isfinite(a):
            return 1
        a = max(0.0, min(0.999, a))
        if a <= 1e-6:
            return 1
        if a >= 0.999:
            return int(cls.PREVIEW_SHADOW_WARMUP_MAX_FRAMES)
        residual = float(max(1e-4, min(0.5, cls.PREVIEW_SHADOW_WARMUP_RESIDUAL)))
        try:
            frames = int(np.ceil(np.log(residual) / np.log(a)))
        except Exception:
            frames = 1
        return int(max(1, min(int(cls.PREVIEW_SHADOW_WARMUP_MAX_FRAMES), frames)))

    def _load_motion_defaults(self) -> Dict[str, Any]:
        defaults = dict(self.MOTION_DEFAULTS_FALLBACK)
        path = self.MOTION_DEFAULTS_CONFIG_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return defaults

            parsed = dict(defaults)
            # Backward compatibility for percent-based payloads.
            if (
                "shadow_area_reset_ratio" not in payload
                and "shadow_area_reset_pct" in payload
            ):
                try:
                    pct = float(payload.get("shadow_area_reset_pct", 0.0))
                    payload["shadow_area_reset_ratio"] = 1.0 + (pct / 100.0)
                except Exception:
                    pass

            parsed["shadow_motion_gain"] = self._coerce_float(
                payload.get("shadow_motion_gain", parsed["shadow_motion_gain"]),
                parsed["shadow_motion_gain"],
            )
            parsed["shadow_motion_enabled"] = bool(
                payload.get("shadow_motion_enabled", parsed["shadow_motion_enabled"])
            )
            parsed["shadow_motion_deadzone_px"] = self._coerce_float(
                payload.get(
                    "shadow_motion_deadzone_px", parsed["shadow_motion_deadzone_px"]
                ),
                parsed["shadow_motion_deadzone_px"],
            )
            parsed["shadow_motion_max_px"] = self._coerce_float(
                payload.get("shadow_motion_max_px", parsed["shadow_motion_max_px"]),
                parsed["shadow_motion_max_px"],
            )
            parsed["shadow_area_min_px"] = self._coerce_float(
                payload.get("shadow_area_min_px", parsed["shadow_area_min_px"]),
                parsed["shadow_area_min_px"],
            )
            parsed["shadow_area_max_px"] = self._coerce_float(
                payload.get("shadow_area_max_px", parsed["shadow_area_max_px"]),
                parsed["shadow_area_max_px"],
            )
            parsed["shadow_area_reset_ratio"] = self._coerce_float(
                payload.get(
                    "shadow_area_reset_ratio", parsed["shadow_area_reset_ratio"]
                ),
                parsed["shadow_area_reset_ratio"],
            )
            parsed["shadow_area_reset_abs_px"] = self._coerce_float(
                payload.get(
                    "shadow_area_reset_abs_px", parsed["shadow_area_reset_abs_px"]
                ),
                parsed["shadow_area_reset_abs_px"],
            )
            parsed["shadow_component_merge_y_tol_px"] = self._coerce_int(
                payload.get(
                    "shadow_component_merge_y_tol_px",
                    parsed["shadow_component_merge_y_tol_px"],
                ),
                parsed["shadow_component_merge_y_tol_px"],
            )
            parsed["shadow_alpha_down"] = self._coerce_float(
                payload.get("shadow_alpha_down", parsed["shadow_alpha_down"]),
                parsed["shadow_alpha_down"],
            )
            logger.info(f"Loaded motion defaults from {path}")
            return parsed
        except FileNotFoundError:
            return defaults
        except Exception as e:
            logger.warning(f"Failed loading motion defaults from '{path}': {e}")
            return defaults

    @staticmethod
    def _path_exists(path: Any) -> bool:
        return isinstance(path, str) and bool(path) and os.path.exists(path)

    def _get_current_source_metadata(self) -> Optional[Dict[str, Any]]:
        previewer = getattr(self, "previewer", None)
        if previewer is None:
            return None
        video_list = getattr(previewer, "video_list", []) or []
        idx = int(getattr(previewer, "current_video_index", -1))
        if 0 <= idx < len(video_list):
            meta = video_list[idx]
            if isinstance(meta, dict):
                return meta
        return None

    def _set_widget_enabled(self, widget: Any, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        try:
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="readonly" if enabled else "disabled")
            else:
                widget.configure(state=state)
        except Exception:
            pass

    def _set_children_enabled(
        self,
        parent: Any,
        enabled: bool,
        skip_widgets: Optional[List[Any]] = None,
    ) -> None:
        if parent is None:
            return
        skip = set(skip_widgets or [])
        for child in parent.winfo_children():
            if child in skip:
                continue
            self._set_widget_enabled(child, enabled)
            self._set_children_enabled(child, enabled, skip_widgets=skip_widgets)

    def _set_mask_slider_row_enabled(self, row: int, enabled: bool) -> None:
        frame = getattr(self, "_mask_sliders_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            try:
                grid_row = int(child.grid_info().get("row", -1))
            except Exception:
                grid_row = -1
            if grid_row != int(row):
                continue
            self._set_widget_enabled(child, enabled)

    def _refresh_mode_constraints(self, trigger_preview: bool = False) -> None:
        if self._is_refreshing_mode_constraints:
            return
        if not hasattr(self, "_mask_sliders_frame") or not hasattr(self, "_replace_mask_check"):
            return
        self._is_refreshing_mode_constraints = True
        try:
            meta = self._get_current_source_metadata()
            previewer = getattr(self, "previewer", None)
            video_list = getattr(previewer, "video_list", []) if previewer is not None else []
            source_readers = (
                getattr(previewer, "source_readers", {}) if previewer is not None else {}
            )
            if meta is None and video_list:
                first = video_list[0]
                meta = first if isinstance(first, dict) else None

            has_single = bool(meta and (meta.get("input_layout") == "single" or meta.get("is_single_input", False)))
            has_replace_mask = bool(meta and self._path_exists(meta.get("replace_mask")))
            has_mask_formerge = bool(meta and self._path_exists(meta.get("mask_formerge")))
            if isinstance(source_readers, dict):
                has_replace_mask = bool(
                    has_replace_mask or (source_readers.get("replace_mask") is not None)
                )
                has_mask_formerge = bool(
                    has_mask_formerge or (source_readers.get("mask_formerge") is not None)
                )

            use_replace_mask_effective = bool(self.use_replace_mask_var.get()) or has_single
            use_mask_formerge_effective = bool(self.use_mask_formerge_var.get()) and has_mask_formerge

            # Single-warp always requires replace-mask and keeps checkbox locked.
            if has_single and not bool(self.use_replace_mask_var.get()):
                self.use_replace_mask_var.set(True)
                use_replace_mask_effective = True
            self._set_widget_enabled(self._replace_mask_check, not has_single)

            # If mask-for-merge is active for current clip, skip all mask-preprocess controls.
            self._set_children_enabled(
                self._mask_sliders_frame,
                not use_mask_formerge_effective,
            )
            self._set_widget_enabled(
                self._temporal_shadow_preview_check,
                not use_mask_formerge_effective,
            )
            self._set_widget_enabled(
                self._shadow_width_adaptive_check,
                not use_mask_formerge_effective,
            )
            self._set_widget_enabled(
                self._motion_chain_check,
                not use_mask_formerge_effective,
            )

            # With replace-mask ON, initial binarization is bypassed (fixed binary mask).
            if not use_mask_formerge_effective:
                self._set_mask_slider_row_enabled(
                    row=0,
                    enabled=not (use_replace_mask_effective and has_replace_mask),
                )

            # CT availability depends on replace-mask presence.
            ct_available = bool(has_replace_mask)
            if not ct_available and bool(self.enable_color_transfer_var.get()):
                self.enable_color_transfer_var.set(False)
            self._set_widget_enabled(self._color_transfer_check, ct_available)
            self._set_children_enabled(self._ct_frame, ct_available)

            if bool(meta and meta.get("input_layout") == "single") and not has_replace_mask:
                logger.error(
                    "Single-warp clip selected but replace-mask is missing."
                )

            if trigger_preview and previewer is not None and getattr(previewer, "video_list", []):
                previewer.update_preview()
        finally:
            self._is_refreshing_mode_constraints = False

    def create_widgets(self):
        self.create_menubar()
        # The main window will now be a simple vertical layout.
        # We will pack frames directly into `self`.        # --- FOLDER FRAME ---
        folder_frame = ttk.LabelFrame(self, text="Folders", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5)

        # Two-column layout to reduce vertical space
        folder_frame.grid_columnconfigure(0, weight=1)
        folder_frame.grid_columnconfigure(1, weight=1)

        left_paths = ttk.Frame(folder_frame)
        right_paths = ttk.Frame(folder_frame)
        left_paths.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_paths.grid(row=0, column=1, sticky="nsew")

        left_paths.grid_columnconfigure(1, weight=1)
        right_paths.grid_columnconfigure(1, weight=1)

        # --- Left column (3 paths) ---
        # Inpainted Video Folder
        ttk.Label(left_paths, text="Inpainted Video Folder:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        entry_inpaint = ttk.Entry(left_paths, textvariable=self.inpainted_folder_var)
        entry_inpaint.grid(row=0, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_inpaint, "inpainted_folder")
        btn_inpaint = ttk.Button(left_paths, text="Browse", command=lambda: self._browse_folder(self.inpainted_folder_var))
        btn_inpaint.grid(row=0, column=2, padx=5)
        self.widgets_to_disable.append(entry_inpaint)
        self.widgets_to_disable.append(btn_inpaint)

        # Original Video Folder (for Left Eye)
        ttk.Label(left_paths, text="Original Video Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        entry_orig = ttk.Entry(left_paths, textvariable=self.original_folder_var)
        entry_orig.grid(row=1, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_orig, "original_folder")
        btn_orig = ttk.Button(left_paths, text="Browse", command=lambda: self._browse_folder(self.original_folder_var))
        btn_orig.grid(row=1, column=2, padx=5)
        self.widgets_to_disable.append(entry_orig)
        self.widgets_to_disable.append(btn_orig)

        # Splat Folder
        ttk.Label(left_paths, text="Splat Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        entry_mask = ttk.Entry(left_paths, textvariable=self.mask_folder_var)
        entry_mask.grid(row=2, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_mask, "mask_folder")
        btn_mask = ttk.Button(left_paths, text="Browse", command=lambda: self._browse_folder(self.mask_folder_var))
        btn_mask.grid(row=2, column=2, padx=5)
        self.widgets_to_disable.append(entry_mask)
        self.widgets_to_disable.append(btn_mask)

        # --- Right column (4 paths) ---
        # Replace Mask Folder (optional)
        ttk.Label(right_paths, text="Replace Mask Folder (optional):").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        entry_rmask = ttk.Entry(right_paths, textvariable=self.replace_mask_folder_var)
        entry_rmask.grid(row=0, column=1, padx=5, sticky="ew")
        btn_rmask = ttk.Button(right_paths, text="Browse", command=lambda: self._browse_folder(self.replace_mask_folder_var))
        btn_rmask.grid(row=0, column=2, padx=5)
        self.widgets_to_disable.append(entry_rmask)
        self.widgets_to_disable.append(btn_rmask)

        # Mask-for-merge Folder (optional preprocessed mask for final blend)
        ttk.Label(right_paths, text="Mask-for-merge Folder (optional):").grid(
            row=1, column=0, sticky="e", padx=5, pady=2
        )
        entry_mform = ttk.Entry(right_paths, textvariable=self.mask_formerge_folder_var)
        entry_mform.grid(row=1, column=1, padx=5, sticky="ew")
        btn_mform = ttk.Button(
            right_paths, text="Browse", command=lambda: self._browse_folder(self.mask_formerge_folder_var)
        )
        btn_mform.grid(row=1, column=2, padx=5)
        self.widgets_to_disable.append(entry_mform)
        self.widgets_to_disable.append(btn_mform)

        # Output Folder
        ttk.Label(right_paths, text="Output Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        entry_out = ttk.Entry(right_paths, textvariable=self.output_folder_var)
        entry_out.grid(row=2, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_out, "output_folder")
        btn_out = ttk.Button(right_paths, text="Browse", command=lambda: self._browse_folder(self.output_folder_var))
        btn_out.grid(row=2, column=2, padx=5)
        self.widgets_to_disable.append(entry_out)
        self.widgets_to_disable.append(btn_out)

        # CT CSV Blend Path (per-frame best preset map)
        ttk.Label(right_paths, text="CT CSV Blend Path:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        entry_ctcsv = ttk.Entry(right_paths, textvariable=self.ct_csv_blend_path_var)
        entry_ctcsv.grid(row=3, column=1, padx=5, sticky="ew")
        btn_ctcsv = ttk.Button(right_paths, text="Browse", command=self._browse_ct_csv_blend_map)
        btn_ctcsv.grid(row=3, column=2, padx=5)
        self.widgets_to_disable.append(entry_ctcsv)
        self.widgets_to_disable.append(btn_ctcsv)

        # --- PREVIEW FRAME (using the new module) ---
        # Moved back to its original position after the folder frame.
        self.previewer = VideoPreviewer(
            self,
            processing_callback=self._preview_processing_callback,
            find_sources_callback=self._find_preview_sources_callback,
            get_params_callback=self.get_current_settings,  # Pass the settings getter
            preview_size_var=self.preview_size_var,  # Pass the preview size variable
            resize_callback=self._adjust_window_height_for_content,  # Pass the resize callback
            help_data=self.help_data,
        )
        self.previewer.preview_source_combo.configure(
            textvariable=self.preview_source_var
        )

        # --- FIX: Add previewer's buttons to the list of widgets to disable ---
        self.widgets_to_disable.append(self.previewer.load_preview_button)
        self.widgets_to_disable.append(self.previewer.prev_video_button)
        self.widgets_to_disable.append(self.previewer.next_video_button)
        self.widgets_to_disable.append(self.previewer.video_jump_entry)
        # Pack the previewer right after the folder frame
        self.previewer.pack(fill="both", expand=True, padx=10, pady=5)

        # --- MASK PROCESSING PARAMETERS ---
        # Place Mask Processing and Color Transfer side-by-side to save vertical space
        params_ct_row = ttk.Frame(self)
        params_ct_row.pack(fill="x", padx=10, pady=5)
        params_ct_row.grid_columnconfigure(0, weight=1)
        params_ct_row.grid_columnconfigure(1, weight=1)

        param_frame = ttk.LabelFrame(
            params_ct_row, text="Mask Processing Parameters", padding=10
        )
        param_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        param_frame.grid_columnconfigure(1, weight=1)
        self._mask_param_frame = param_frame
        # Keep sliders and checkboxes in separate sub-frames for stable vertical layout.
        mask_sliders_frame = ttk.Frame(param_frame)
        mask_sliders_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        mask_sliders_frame.grid_columnconfigure(1, weight=1)
        self._mask_sliders_frame = mask_sliders_frame
        # Vertical spacer that pushes checkbox row to the bottom of the panel.
        param_frame.grid_rowconfigure(1, weight=1)

        # def create_slider_with_label_updater(parent, text, var, from_, to, row, decimals=0) -> None:
        #     """Creates a slider, its value label, and all necessary event bindings."""
        #     label = ttk.Label(parent, text=text)
        #     label.grid(row=row, column=0, sticky="e", padx=5, pady=2)
        #     slider = ttk.Scale(parent, from_=from_, to=to, variable=var, orient="horizontal")
        #     slider.grid(row=row, column=1, sticky="ew", padx=5)
        #     value_label = ttk.Label(parent, text="", width=5) # Start with empty text
        #     value_label.grid(row=row, column=2, sticky="w", padx=5)

        #     def update_label_and_preview(value_str: str) -> None:
        #         """Updates the text label. Called by user interaction."""
        #         value_label.config(text=f"{float(value_str):.{decimals}f}")

        #     def set_value_and_update_label(new_value: float) -> None:
        #         """Programmatically sets the slider's value and updates its label."""
        #         var.set(new_value)
        #         value_label.config(text=f"{new_value:.{decimals}f}")
        #         logger.debug(f"new_value {new_value:.{decimals}f}")

        #     slider.configure(command=update_label_and_preview)
        #     slider.bind("<ButtonRelease-1>", self.on_slider_release)
        #     self._create_hover_tooltip(label, text.lower().replace(":", "").replace(" ", "_").replace(".", ""))
        #     self.slider_label_updaters.append(lambda: set_value_and_update_label(var.get())) # Add updater to list
        #     self.widgets_to_disable.append(slider)

        #     def on_trough_click(event):
        #         """Handles clicks on the slider's trough for precise positioning."""
        #         # Check if the click is on the trough to avoid interfering with handle drags
        #         if 'trough' in slider.identify(event.x, event.y):
        #             # --- FIX: Force the widget to update its size info before calculating ---
        #             # This ensures winfo_width() is accurate, which is critical for fractional sliders.
        #             slider.update_idletasks()
        #             new_value = from_ + (to - from_) * (event.x / slider.winfo_width())
        #             var.set(new_value) # Set the tk.Variable, which triggers the command and updates the UI
        #             # --- FIX: Manually update the label's text after setting the variable ---
        #             value_label.config(text=f"{new_value:.{decimals}f}")
        #             self.on_slider_release(event) # Manually trigger preview update
        #             return "break" # IMPORTANT: Prevents the default slider click behavior

        #     slider.bind("<Button-1>", on_trough_click)

        create_single_slider_with_label_updater(
            self,
            mask_sliders_frame,
            "Binarize Thresh (<0=Off):",
            self.mask_binarize_threshold_var,
            -0.01,
            1.0,
            0,
            decimals=2,
            step_size=0.01,
        )
        create_single_slider_with_label_updater(
            self,
            mask_sliders_frame,
            "Dilate Kernel:",
            self.mask_dilate_kernel_size_var,
            0,
            101,
            1,
        )
        create_single_slider_with_label_updater(
            self, mask_sliders_frame, "Blur Kernel:", self.mask_blur_kernel_size_var, 0, 101, 2
        )
        create_single_slider_with_label_updater(
            self,
            mask_sliders_frame,
            "Shadow Length (px):",
            self.shadow_length_px_var,
            0,
            100,
            3,
            decimals=0,
            step_size=1.0,
        )
        create_single_slider_with_label_updater(
            self,
            mask_sliders_frame,
            "Shadow Curve (-1..1):",
            self.shadow_curve_var,
            -1.0,
            1.0,
            4,
            decimals=2,
            step_size=0.01,
        )

        mask_checks_row = ttk.Frame(param_frame)
        mask_checks_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=(8, 2))
        self._mask_checks_row = mask_checks_row
        temporal_shadow_preview_check = ttk.Checkbutton(
            mask_checks_row,
            text="Temporal Shadow Preview (dynamic warmup)",
            variable=self.preview_shadow_temporal_var,
            command=lambda: self.on_slider_release(None),
        )
        shadow_width_adaptive_check = ttk.Checkbutton(
            mask_checks_row,
            text="Dynamic shadow by mask width",
            variable=self.shadow_width_adaptive_var,
            command=lambda: self.on_slider_release(None),
        )
        replace_mask_check = ttk.Checkbutton(
            mask_checks_row,
            text="Use Replace Mask",
            variable=self.use_replace_mask_var,
            command=self._on_use_replace_mask_changed,
        )
        mask_formerge_check = ttk.Checkbutton(
            mask_checks_row,
            text="Use Mask-for-merge (preprocessed)",
            variable=self.use_mask_formerge_var,
            command=self._on_use_mask_formerge_changed,
        )
        motion_chain_check = ttk.Checkbutton(
            mask_checks_row,
            text="Motion Chain Enabled",
            variable=self.shadow_motion_enabled_var,
            command=lambda: self.on_slider_release(None),
        )
        temporal_shadow_preview_check.grid(row=0, column=0, sticky="w", padx=(0, 12), pady=2)
        shadow_width_adaptive_check.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=2)
        replace_mask_check.grid(row=0, column=2, sticky="w", padx=(0, 12), pady=2)
        mask_formerge_check.grid(row=1, column=0, sticky="w", padx=(0, 12), pady=2)
        motion_chain_check.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=2)
        self.widgets_to_disable.append(temporal_shadow_preview_check)
        self.widgets_to_disable.append(shadow_width_adaptive_check)
        self.widgets_to_disable.append(replace_mask_check)
        self.widgets_to_disable.append(mask_formerge_check)
        self.widgets_to_disable.append(motion_chain_check)
        self._temporal_shadow_preview_check = temporal_shadow_preview_check
        self._shadow_width_adaptive_check = shadow_width_adaptive_check
        self._replace_mask_check = replace_mask_check
        self._mask_formerge_check = mask_formerge_check
        self._motion_chain_check = motion_chain_check

        # --- COLOR TRANSFER (PRESET-ONLY) PARAMETERS ---
        ct_frame = ttk.LabelFrame(params_ct_row, text="Color Transfer (Safe)", padding=10)
        ct_frame.grid(row=0, column=1, sticky="nsew")
        self._ct_frame = ct_frame
        for _c in range(4):
            ct_frame.grid_columnconfigure(_c, weight=1 if _c in (1, 3) else 0)

        ct_preset_row = ttk.Frame(ct_frame)
        ct_preset_row.grid(
            row=0, column=0, columnspan=4, sticky="ew", padx=0, pady=(0, 4)
        )
        ct_preset_row.grid_columnconfigure(1, weight=1)
        ct_preset_row.grid_columnconfigure(2, weight=0)

        ttk.Label(ct_preset_row, text="Preset:").grid(
            row=0, column=0, sticky="e", padx=5, pady=2
        )
        ct_preset_combo = ttk.Combobox(
            ct_preset_row,
            textvariable=self.ct_preset_var,
            values=CT_PRESET_LABELS,
            state="readonly",
            width=60,
        )
        ct_preset_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.widgets_to_disable.append(ct_preset_combo)

        auto_ct_best_label = ttk.Label(
            ct_preset_row, textvariable=self.auto_ct_best_var, anchor="w"
        )
        auto_ct_best_label.grid(row=0, column=2, sticky="w", padx=5, pady=2)

        ct_auto_row = ttk.Frame(ct_frame)
        ct_auto_row.grid(
            row=1, column=0, columnspan=4, sticky="ew", padx=0, pady=(0, 4)
        )
        ct_auto_row.grid_columnconfigure(1, weight=0)
        ct_auto_row.grid_columnconfigure(2, weight=0)
        ct_auto_row.grid_columnconfigure(3, weight=0)
        ct_auto_row.grid_columnconfigure(4, weight=1)

        ttk.Label(ct_auto_row, text="Auto CT:").grid(
            row=0, column=0, sticky="e", padx=5, pady=2
        )
        ct_auto_mode_combo = ttk.Combobox(
            ct_auto_row,
            textvariable=self.ct_auto_mode_var,
            values=CT_AUTO_MODE_OPTIONS,
            state="readonly",
            width=14,
        )
        ct_auto_mode_combo.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.widgets_to_disable.append(ct_auto_mode_combo)

        ct_excl = ttk.Checkbutton(
            ct_auto_row,
            text="Exclude near-black in target stats",
            variable=self.ct_exclude_black_in_target_var,
        )
        ct_excl.grid(row=0, column=2, sticky="w", padx=(12, 8), pady=2)
        self._create_hover_tooltip(ct_excl, "ct_exclude_black_in_target")
        self.widgets_to_disable.append(ct_excl)

        show_blend_preview_check = ttk.Checkbutton(
            ct_auto_row,
            text="Show blend in preview",
            variable=self.show_blend_in_preview_var,
            command=lambda: self.on_slider_release(None),
        )
        show_blend_preview_check.grid(row=0, column=3, sticky="w", padx=(0, 5), pady=2)
        self.widgets_to_disable.append(show_blend_preview_check)

        ct_advanced_check = ttk.Checkbutton(
            ct_auto_row,
            text="Advanced",
            variable=self.ct_advanced_var,
            command=self._toggle_ct_advanced_controls,
        )
        ct_advanced_check.grid(row=0, column=4, sticky="w", padx=(8, 5), pady=2)
        self.widgets_to_disable.append(ct_advanced_check)

        # Sliders (two columns) — keep preview updates on release
        ct_sliders_row = ttk.Frame(ct_frame)
        ct_sliders_row.grid(row=2, column=0, columnspan=4, sticky="ew", padx=0, pady=(6, 0))
        ct_sliders_row.grid_columnconfigure(0, weight=1)
        ct_sliders_row.grid_columnconfigure(1, weight=1)
        self._ct_sliders_row = ct_sliders_row

        ct_sliders_left = ttk.Frame(ct_sliders_row)
        ct_sliders_right = ttk.Frame(ct_sliders_row)
        ct_sliders_left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ct_sliders_right.grid(row=0, column=1, sticky="nsew")

        ct_sliders_left.grid_columnconfigure(1, weight=1)
        ct_sliders_right.grid_columnconfigure(1, weight=1)

        # Left column
        create_single_slider_with_label_updater(
            self, ct_sliders_left, "CT Strength:", self.ct_strength_var,
            0.0, 1.0, 0, decimals=2, step_size=0.01
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_left, "Black Thresh (0..32):", self.ct_black_thresh_var,
            0.0, 32.0, 1, decimals=1, step_size=1.0
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_left, "Min Valid Ratio:", self.ct_min_valid_ratio_var,
            0.0, 0.10, 2, decimals=3, step_size=0.001
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_left, "Min Valid Pixels:", self.ct_min_valid_var,
            0, 50000, 3, decimals=0, step_size=100
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_left, "Ring Width (px):", self.ct_ring_width_var,
            0, 200, 4, decimals=0, step_size=1
        )

        # Right column
        create_single_slider_with_label_updater(
            self, ct_sliders_right, "Clamp L Min:", self.ct_clamp_L_min_var,
            0.1, 2.0, 0, decimals=2, step_size=0.01
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_right, "Clamp L Max:", self.ct_clamp_L_max_var,
            0.1, 2.0, 1, decimals=2, step_size=0.01
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_right, "Clamp ab Min:", self.ct_clamp_ab_min_var,
            0.1, 3.0, 2, decimals=2, step_size=0.01
        )
        create_single_slider_with_label_updater(
            self, ct_sliders_right, "Clamp ab Max:", self.ct_clamp_ab_max_var,
            0.1, 3.0, 3, decimals=2, step_size=0.01
        )

        # Make comboboxes trigger preview refresh immediately on change
        for _v in (
            self.ct_preset_var,
            self.ct_auto_mode_var,
            self.ct_csv_blend_path_var,
            self.ct_exclude_black_in_target_var,
            self.show_blend_in_preview_var,
        ):
            try:
                if _v is self.ct_preset_var:
                    _v.trace_add("write", self._on_ct_preset_changed)
                elif _v is self.ct_auto_mode_var:
                    _v.trace_add("write", self._on_auto_ct_eval_toggle)
                else:
                    _v.trace_add("write", lambda *args: self.on_slider_release(None))
            except Exception:
                pass
        self._toggle_ct_advanced_controls()
        # --- END COLOR TRANSFER (PRESET-ONLY) PARAMETERS ---
        # --- OPTIONS FRAME ---
        # Dock Options beside the preview controls row (Preview Source / Prev/Next / Jump / Scale).
        # The most reliable anchor is the parent frame that owns preview_source_combo.
        options_parent = getattr(self.previewer, "preview_source_combo", None)
        if options_parent is not None:
            options_parent = options_parent.master
        else:
            options_parent = self  # fallback: keep the old vertical placement

        options_frame = ttk.LabelFrame(options_parent, text="Options", padding=10)

        def _parent_uses_grid(parent) -> bool:
            try:
                for w in parent.winfo_children():
                    if w.winfo_manager() == "grid":
                        return True
            except Exception:
                pass
            return False

        if options_parent is self:
            options_frame.pack(fill="x", padx=10, pady=5)
        else:
            if _parent_uses_grid(options_parent):
                # Place at the far right of the top controls row
                try:
                    cols, rows = options_parent.grid_size()
                except Exception:
                    cols, rows = (0, 0)
                col = int(cols) if cols is not None else 0
                options_parent.grid_columnconfigure(col, weight=0)
                options_frame.grid(row=0, column=col, sticky="ne", padx=(10, 0), pady=0)
            else:
                options_frame.pack(side="right", padx=(10, 0), pady=0, anchor="ne")
        gpu_check = ttk.Checkbutton(options_frame, text="Use GPU", variable=self.use_gpu_var)
        gpu_check.pack(side="left", padx=5)
        self._create_hover_tooltip(gpu_check, "use_gpu")
        self.widgets_to_disable.append(gpu_check)

        # Output format dropdown
        ttk.Label(options_frame, text="Output:").pack(side="left", padx=(10, 5))
        output_format_combo = ttk.Combobox(
            options_frame,
            textvariable=self.output_format_var,
            values=self.OUTPUT_FORMAT_CHOICES,
            state="readonly",
            width=22,
        )
        output_format_combo.pack(side="left", padx=5)
        self._create_hover_tooltip(output_format_combo, "output_format")
        self.widgets_to_disable.append(output_format_combo)

        color_check = ttk.Checkbutton(
            options_frame,
            text="Color Transfer",
            variable=self.enable_color_transfer_var,
        )
        color_check.pack(side="left", padx=5)
        self._create_hover_tooltip(color_check, "enable_color_transfer")
        self.widgets_to_disable.append(color_check)
        self._color_transfer_check = color_check

        pad_check = ttk.Checkbutton(
            options_frame, text="Pad 16:9", variable=self.pad_to_16_9_var
        )
        pad_check.pack(side="left", padx=(10, 5))
        self._create_hover_tooltip(pad_check, "pad_to_16_9")
        self.widgets_to_disable.append(pad_check)

        # Add Borders
        self.add_borders_var = tk.BooleanVar(
            value=bool(
                self.app_config.get(
                    "add_borders",
                    self.APP_DEFAULTS.get("add_borders", False),
                )
            )
        )
        self.add_borders_var.trace_add("write", self._on_add_borders_changed)
        borders_check = ttk.Checkbutton(
            options_frame, text="Borders", variable=self.add_borders_var
        )
        borders_check.pack(side="left", padx=(10, 5))
        self._create_hover_tooltip(borders_check, "add_borders")
        self.widgets_to_disable.append(borders_check)

        # Resume
        self.resume_var = tk.BooleanVar(value=self.app_config.get("resume", False))
        self.resume_var.trace_add("write", self._on_resume_changed)
        resume_check = ttk.Checkbutton(
            options_frame, text="Resume", variable=self.resume_var
        )
        resume_check.pack(side="left", padx=(10, 5))
        self._create_hover_tooltip(resume_check, "resume")
        self.widgets_to_disable.append(resume_check)

        # Batch chunk size
        ttk.Label(options_frame, text="Chunk:").pack(side="left", padx=(12, 5))
        entry_chunk = ttk.Entry(options_frame, textvariable=self.batch_chunk_size_var, width=6)
        entry_chunk.pack(side="left")
        self._create_hover_tooltip(entry_chunk, "batch_chunk_size")
        self.widgets_to_disable.append(entry_chunk)

        # --- PROGRESS & BUTTONS ---

        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)

        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_columnconfigure(1, weight=0)

        # Row 0: progress bar (left) + buttons (right)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, length=400, mode="determinate"
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=(0, 5))

        buttons_frame = ttk.Frame(progress_frame)
        buttons_frame.grid(row=0, column=1, sticky="e", pady=(0, 5))

        self.start_button = ttk.Button(
            buttons_frame, text="Start Blending", command=self.start_processing
        )
        self.start_button.grid(row=0, column=0, padx=5)
        self._create_hover_tooltip(self.start_button, "start_blending")
        self.widgets_to_disable.append(self.start_button)  # disable during processing

        self.stop_button = ttk.Button(
            buttons_frame, text="Stop", command=self.stop_processing, state="disabled"
        )
        # Stop button is handled separately in _set_ui_processing_state
        self.stop_button.grid(row=0, column=1, padx=5)
        self._create_hover_tooltip(self.stop_button, "stop_blending")

        # --- NEW: Process Current Clip button ---
        self.process_current_button = ttk.Button(
            buttons_frame,
            text="Process Current Clip",
            command=self.process_current_clip,
        )
        self.process_current_button.grid(row=0, column=2, padx=5)
        self._create_hover_tooltip(self.process_current_button, "process_current_clip")
        self.widgets_to_disable.append(self.process_current_button)
        # --- END NEW ---

        # Row 1+: status text
        self.status_label_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_label_var)
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # --- Border Info ---
        self.border_info_var = tk.StringVar(value="Borders: N/A")
        self.border_info_label = ttk.Label(progress_frame, textvariable=self.border_info_var)
        self.border_info_label.grid(row=2, column=0, columnspan=2, sticky="w")
    def _browse_folder(self, var: tk.StringVar):
        folder = filedialog.askdirectory(initialdir=var.get())
        if folder:
            var.set(folder)

    def _browse_ct_csv_blend_map(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.ct_csv_blend_path_var.get() or "."),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select CSV Blend Auto CT Map",
        )
        if path:
            self.ct_csv_blend_path_var.set(path)

    def _find_video_by_core_name(self, folder: str, core_name: str) -> Optional[str]:
        """Scans a folder for a file matching the core_name with any common video extension."""
        return find_video_by_core_name(folder, core_name)

    def _find_replace_mask_for_splatted(
        self, splatted_path: str, replace_mask_folder: str = ""
    ) -> Optional[str]:
        """Return external replace-mask video path if present.

        Naming: <splatted_basename_without_ext> + '_replace_mask.mkv' (or .mp4).
        Folder: replace_mask_folder if provided, else same folder as splatted_path.
        """
        try:
            base = os.path.splitext(os.path.basename(splatted_path))[0]
            folder = (replace_mask_folder or "").strip()
            if not folder:
                folder = os.path.dirname(splatted_path)

            for ext in [".mkv", ".mp4"]:
                candidate = os.path.join(folder, f"{base}_replace_mask{ext}")
                if os.path.exists(candidate):
                    return candidate
            return None
        except Exception:
            return None

    def _find_mask_formerge_for_splatted(
        self, splatted_path: str, mask_formerge_folder: str = ""
    ) -> Optional[str]:
        """Return preprocessed mask-for-merge path for the given splatted clip."""
        try:
            base = os.path.splitext(os.path.basename(splatted_path))[0]
            folder = (mask_formerge_folder or "").strip()
            if not folder:
                return None
            for ext in [".mkv", ".mp4"]:
                candidate = os.path.join(folder, f"{base}_replace_mask{ext}")
                if os.path.exists(candidate):
                    return candidate
            return None
        except Exception:
            return None

    def _find_sidecar_file(self, base_path: str) -> Optional[str]:
        """Looks for a sidecar JSON file next to the video file."""
        return find_sidecar_file(base_path)

    def _read_clip_sidecar(self, video_path: str, core_name: str) -> dict:
        """
        Reads the sidecar file for a clip if it exists.
        Returns a dictionary of sidecar data merged with defaults.
        """
        search_folders = []
        if self.inpainted_folder_var.get():
            search_folders.append(self.inpainted_folder_var.get())
        if self.original_folder_var.get():
            search_folders.append(self.original_folder_var.get())
        return read_clip_sidecar(
            self.sidecar_manager, video_path, core_name, search_folders
        )

    def _update_border_info(self, left_border: float, right_border: float):
        """Updates the border info display in the GUI."""
        if left_border > 0 or right_border > 0:
            self.border_info_var.set(
                f"Borders: L={left_border:.3f}%, R={right_border:.3f}%"
            )
        else:
            self.border_info_var.set("Borders: None")

    def _clear_border_info(self):
        """Clears the border info display."""
        self.border_info_var.set("Borders: N/A")

    def _apply_ct_preset_to_controls(self, preset_label: str) -> None:
        """Normalize preset label to one of the known CT preset options."""
        label = _resolve_ct_preset_label(preset_label)
        if self.ct_preset_var.get() != label:
            self.ct_preset_var.set(label)

    def _on_ct_preset_changed(self, *args):
        """Sync dependent CT controls when preset changes."""
        self._apply_ct_preset_to_controls(self.ct_preset_var.get())
        try:
            self.on_slider_release(None)
        except Exception:
            pass

    def _get_ct_csv_blend_preset_map_cached(
        self, csv_path: str
    ) -> Dict[str, Dict[int, int]]:
        path = os.path.abspath(os.path.expanduser(str(csv_path or "").strip()))
        if not path or not os.path.exists(path):
            self._ct_csv_blend_cache_path = ""
            self._ct_csv_blend_cache_mtime = -1.0
            self._ct_csv_blend_cache = {}
            return {}
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            return {}

        if (
            path != self._ct_csv_blend_cache_path
            or mtime != self._ct_csv_blend_cache_mtime
        ):
            try:
                self._ct_csv_blend_cache = _load_csv_blend_preset_map(path)
                self._ct_csv_blend_cache_path = path
                self._ct_csv_blend_cache_mtime = mtime
                logger.info(
                    f"Loaded CSV Blend Auto CT map: keys={len(self._ct_csv_blend_cache)} from {path}"
                )
            except Exception as e:
                logger.error(f"Failed to load CSV Blend Auto CT map '{path}': {e}")
                self._ct_csv_blend_cache = {}
        return dict(self._ct_csv_blend_cache)

    def _on_auto_ct_eval_toggle(self, *args):
        mode = _resolve_ct_auto_mode_label(self.ct_auto_mode_var.get())
        if self.ct_auto_mode_var.get() != mode:
            self.ct_auto_mode_var.set(mode)
        self.auto_ct_eval_var.set(mode == CT_AUTO_MODE_ON)
        if mode == CT_AUTO_MODE_ON:
            self.auto_ct_best_var.set("Auto CT best: pending...")
        elif mode == CT_AUTO_MODE_CSV_BLEND:
            self.auto_ct_best_var.set("Auto CT CSV Blend: pending...")
        else:
            self.auto_ct_best_var.set("Auto CT best: (disabled)")
        try:
            self.on_slider_release(None)
        except Exception:
            pass

    def on_slider_release(self, event):
        """Called when a slider is released. Updates the preview."""
        # This now just collects parameters and sends them to the previewer module.
        params = self.get_current_settings()
        if params:
            self.previewer.set_parameters(params)

    def _on_add_borders_changed(self, *args):
        """Called when the Add Borders checkbox is toggled. Updates the preview."""
        if hasattr(self, "previewer") and self.previewer.video_list:
            self.previewer.update_preview()

    def _on_use_replace_mask_changed(self, *args):
        """Called when replace-mask mode changes."""
        self._refresh_mode_constraints(trigger_preview=True)

    def _on_use_mask_formerge_changed(self, *args):
        """Called when preprocessed mask-for-merge mode changes."""
        self._refresh_mode_constraints(trigger_preview=True)

    def _on_folder_changed(self, *args):
        """Called when a folder path changes. Resets the video list scan flag."""
        if hasattr(self, "previewer"):
            self.previewer.reset_video_list_scan()
        self._last_mode_constraints_video_index = -999999
        self._refresh_mode_constraints(trigger_preview=False)

    def _on_resume_changed(self, *args):
        """Called when the Resume checkbox is changed. Clears preview to apply new setting."""
        if hasattr(self, "previewer") and self.previewer.video_list:
            # Update preview to reflect the new setting
            self.previewer.update_preview()

    def _set_ui_processing_state(self, is_processing: bool):
        """Disables or enables all interactive widgets during processing."""
        # --- FIX: Explicitly handle start/stop button states ---
        try:
            self.start_button.config(state="disabled" if is_processing else "normal")
            self.stop_button.config(state="normal" if is_processing else "disabled")
        except tk.TclError:
            pass  # Ignore if widgets don't exist yet
        # --- END FIX ---
        state = "disabled" if is_processing else "normal"
        for widget in self.widgets_to_disable:
            try:
                # Special handling for combobox which uses 'readonly' instead of 'normal'
                if isinstance(widget, ttk.Combobox):
                    widget.config(state="disabled" if is_processing else "readonly")
                else:
                    widget.config(state=state)
            except tk.TclError:
                # Widget might have been destroyed, ignore
                pass
        if not is_processing:
            self._refresh_mode_constraints(trigger_preview=False)

    def update_status_label(self, message):
        self.status_label_var.set(message)
        self.update_idletasks()

    def _clear_preview_resources(self):
        """Closes all preview-related video readers and clears the preview display."""
        self.previewer.cleanup()

    def _cleanup_worker(self):
        """
        A worker thread that processes a queue of files to be moved.
        It will retry moving a file until it succeeds.
        """
        stop_signal_received = False
        while not stop_signal_received or not self.cleanup_queue.empty():
            try:
                # Wait for an item, but with a timeout so the loop can check the stop condition
                item = self.cleanup_queue.get(timeout=1)

                if item is None:
                    logger.debug(
                        "Cleanup worker received stop signal. Will exit when queue is empty."
                    )
                    stop_signal_received = True
                    continue  # Continue loop to check if queue is empty

                src_path, dest_folder = item

                try:
                    if not os.path.exists(src_path):
                        logger.debug(
                            f"Cleanup: Source file '{os.path.basename(src_path)}' no longer exists. Skipping move."
                        )
                        continue

                    finished_dir = os.path.join(dest_folder, "finished")
                    os.makedirs(finished_dir, exist_ok=True)
                    dest_path = os.path.join(finished_dir, os.path.basename(src_path))

                    if os.path.exists(dest_path):
                        logger.debug(
                            f"Cleanup: Destination '{os.path.basename(dest_path)}' exists. Deleting source."
                        )
                        os.remove(src_path)
                    else:
                        shutil.move(src_path, finished_dir)
                    logger.info(
                        f"Cleanup: Successfully moved '{os.path.basename(src_path)}'."
                    )
                except (PermissionError, OSError):
                    logger.debug(
                        f"Cleanup: File '{os.path.basename(src_path)}' is locked. Retrying in 3 seconds..."
                    )
                    time.sleep(3)
                    self.cleanup_queue.put(item)  # Put it back on the queue to retry
                except Exception as e:
                    logger.error(
                        f"Cleanup worker encountered an unexpected error for {os.path.basename(src_path)}: {e}",
                        exc_info=True,
                    )

            except queue.Empty:
                # This is expected when waiting for items. The loop condition will handle exit.
                continue
        logger.debug("Cleanup worker has finished its queue and is now exiting.")

    def _retry_failed_moves(self):
        """Attempts to move any files that previously failed to move."""
        if not self.failed_moves:
            return

        logger.info(
            f"Retrying {len(self.failed_moves)} previously failed file moves..."
        )

        # Use a copy of the list to iterate over, so we can safely remove from the original
        remaining_failures = []
        for src_path, dest_folder in self.failed_moves:
            try:
                # --- FIX: Check for source existence FIRST ---
                if not os.path.exists(src_path):
                    logger.debug(
                        f"Retry: Source file '{os.path.basename(src_path)}' no longer exists. Assuming it was moved successfully."
                    )
                    continue  # This item is resolved, do not add to remaining_failures

                finished_dir = os.path.join(dest_folder, "finished")
                dest_path = os.path.join(finished_dir, os.path.basename(src_path))

                if os.path.exists(dest_path):
                    # Destination exists, so the move likely succeeded. We just need to delete the source.
                    logger.info(
                        f"Retry: Destination '{os.path.basename(dest_path)}' exists. Deleting source '{os.path.basename(src_path)}'."
                    )
                    try:
                        os.remove(src_path)
                    except Exception as e_del:
                        logger.error(
                            f"Retry: Failed to delete source '{os.path.basename(src_path)}' even though destination exists: {e_del}"
                        )
                        remaining_failures.append(
                            (src_path, dest_folder)
                        )  # Keep it for the next final retry
                else:
                    # Destination does not exist, but we know the source does. This is a true move retry.
                    shutil.move(src_path, finished_dir)
                    logger.debug(
                        f"Successfully moved previously failed file: {os.path.basename(src_path)}"
                    )

            except (PermissionError, OSError) as e:
                logger.warning(
                    f"Retry failed for {os.path.basename(src_path)}: {e}. Will try again later."
                )
                remaining_failures.append(
                    (src_path, dest_folder)
                )  # Add back to the list for the next attempt
            except Exception as e:
                logger.error(
                    f"Unexpected error during retry for {os.path.basename(src_path)}: {e}",
                    exc_info=True,
                )

        self.failed_moves = remaining_failures

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        self.is_processing = True
        self.stop_event.clear()
        self._set_ui_processing_state(True)  # Disable UI

        # --- NEW: Start the cleanup worker thread ---
        self.cleanup_queue = queue.Queue()  # Clear the queue from any previous run
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("File cleanup worker thread started.")
        # --- END NEW ---

        # --- NEW: Clear preview resources before starting batch processing ---
        self._clear_preview_resources()

        self.update_status_label("Starting...")

        # Collect settings
        settings = self.get_current_settings()

        # Run in a separate thread
        self.processing_thread = threading.Thread(
            target=self.run_batch_process, args=(settings, None), daemon=True
        )
        self.processing_thread.start()

    def stop_processing(self):
        if self.is_processing:
            self.stop_event.set()
            self.update_status_label("Stopping...")

    def process_current_clip(self):
        """Process the currently selected clip only."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # Get current video from previewer
        if not hasattr(self, "previewer") or not self.previewer.video_list:
            messagebox.showwarning("No Video", "No video loaded in previewer.")
            return

        current_index = getattr(self.previewer, "current_video_index", 0)
        if current_index < 0 or current_index >= len(self.previewer.video_list):
            messagebox.showwarning("Invalid Index", "No video selected.")
            return

        source_dict = self.previewer.video_list[current_index]
        inpainted_path = source_dict.get("inpainted")

        if not inpainted_path or not os.path.exists(inpainted_path):
            messagebox.showwarning("Invalid Path", "Inpainted video path not found.")
            return

        # Get current settings
        settings = self.get_current_settings()
        if not settings:
            return

        # Temporarily set inpainted_folder to just this file's directory
        settings["inpainted_folder"] = os.path.dirname(inpainted_path)

        self.is_processing = True
        self.stop_event.clear()
        self._set_ui_processing_state(True)

        # --- Start the cleanup worker thread (needed for Resume moves) ---
        self.cleanup_queue = queue.Queue()  # Clear the queue from any previous run
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("File cleanup worker thread started.")
        self._clear_preview_resources()
        base_name = os.path.basename(inpainted_path)
        self.update_status_label(f"Processing single clip: {base_name}")

        # Run in a separate thread using the existing batch processor
        # Pass the specific video path to process only this one
        self.processing_thread = threading.Thread(
            target=self.run_batch_process, args=(settings, inpainted_path), daemon=True
        )
        self.processing_thread.start()

    def processing_done(self, stopped=False):
        self.is_processing = False
        self._set_ui_processing_state(False)  # Re-enable UI
        message = "Processing stopped." if stopped else "Processing completed."
        self.update_status_label(message)
        self.progress_var.set(0)
        self._clear_border_info()

        # --- NEW: Schedule VRAM release after a short delay to ensure stability ---
        delay_ms = 2000  # 2 seconds
        logger.info(f"Scheduling VRAM release in {delay_ms / 1000} seconds...")
        self.after(delay_ms, release_cuda_memory)
        # --- END NEW ---

    def get_current_settings(self):
        """Collects all GUI settings into a dictionary, performing type conversion."""
        try:
            selected_preset_label = _resolve_ct_preset_label(self.ct_preset_var.get())
            selected_auto_mode = _resolve_ct_auto_mode_label(self.ct_auto_mode_var.get())
            alpha_down = float(self.shadow_alpha_down_var.get())
            dynamic_warmup = self._compute_preview_shadow_warmup_frames(alpha_down)
            settings = {
                "inpainted_folder": self.inpainted_folder_var.get(),
                "original_folder": self.original_folder_var.get(),
                "mask_folder": self.mask_folder_var.get(),
                "replace_mask_folder": self.replace_mask_folder_var.get(),
                "mask_formerge_folder": self.mask_formerge_folder_var.get(),
                "output_folder": self.output_folder_var.get(),
                "use_gpu": self.use_gpu_var.get(),
                "pad_to_16_9": self.pad_to_16_9_var.get(),
                "add_borders": self.add_borders_var.get(),
                "resume": self.resume_var.get(),
                "output_format": self.output_format_var.get(),
                "batch_chunk_size": int(self.batch_chunk_size_var.get()),
                "enable_color_transfer": self.enable_color_transfer_var.get(),
                "ct_preset": selected_preset_label,
                "ct_auto_mode": selected_auto_mode,
                "ct_csv_blend_path": str(self.ct_csv_blend_path_var.get() or "").strip(),
                "show_blend_in_preview": bool(self.show_blend_in_preview_var.get()),
                "ct_advanced": bool(self.ct_advanced_var.get()),
                # Legacy key kept for backward compatibility with old configs/scripts.
                "auto_ct_eval": bool(selected_auto_mode == CT_AUTO_MODE_ON),
                "ct_strength": float(self.ct_strength_var.get()),
                "ct_black_thresh": float(self.ct_black_thresh_var.get()),
                "ct_min_valid_ratio": float(self.ct_min_valid_ratio_var.get()),
                "ct_min_valid": int(self.ct_min_valid_var.get()),
                "ct_clamp_L_min": float(self.ct_clamp_L_min_var.get()),
                "ct_clamp_L_max": float(self.ct_clamp_L_max_var.get()),
                "ct_clamp_ab_min": float(self.ct_clamp_ab_min_var.get()),
                "ct_clamp_ab_max": float(self.ct_clamp_ab_max_var.get()),
                "ct_exclude_black_in_target": bool(self.ct_exclude_black_in_target_var.get()),
                "ct_ring_width": int(self.ct_ring_width_var.get()),

                "preview_size": self.preview_size_var.get(),
                "preview_source": self.preview_source_var.get(),
                "use_replace_mask": bool(self.use_replace_mask_var.get()),
                "use_mask_formerge": bool(self.use_mask_formerge_var.get()),
                # Mask params
                "mask_binarize_threshold": float(
                    self.mask_binarize_threshold_var.get()
                ),
                "mask_dilate_kernel_size": int(self.mask_dilate_kernel_size_var.get()),
                "mask_blur_kernel_size": int(self.mask_blur_kernel_size_var.get()),
                "shadow_length_px": int(self.shadow_length_px_var.get()),
                "shadow_width_adaptive": bool(self.shadow_width_adaptive_var.get()),
                "shadow_curve": float(self.shadow_curve_var.get()),
                "shadow_motion_gain": float(self.shadow_motion_gain_var.get()),
                "shadow_motion_enabled": bool(self.shadow_motion_enabled_var.get()),
                "shadow_motion_deadzone_px": float(self.shadow_motion_deadzone_px_var.get()),
                "shadow_motion_max_px": float(self.shadow_motion_max_px_var.get()),
                "shadow_area_min_px": float(self.shadow_area_min_px_var.get()),
                "shadow_area_max_px": float(self.shadow_area_max_px_var.get()),
                "shadow_area_reset_ratio": float(self.shadow_area_reset_ratio_var.get()),
                "shadow_area_reset_abs_px": float(self.shadow_area_reset_abs_px_var.get()),
                "shadow_component_merge_y_tol_px": int(
                    self.shadow_component_merge_y_tol_px_var.get()
                ),
                "shadow_alpha_down": alpha_down,
                "preview_shadow_temporal": bool(self.preview_shadow_temporal_var.get()),
                "preview_shadow_warmup_frames": int(dynamic_warmup),
            }
            return settings
        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Invalid Settings",
                f"Please check your parameter values. They must be valid numbers.\n\nError: {e}",
            )
            return None

    def _read_ffmpeg_output(self, pipe, log_level):
        """Helper method to read FFmpeg's output without blocking."""
        try:
            # Use iter to read line by line
            for line in iter(
                pipe.readline, b""
            ):  # Read bytes until an empty byte string
                if line:
                    # Decode bytes to string for logging, ignoring potential decoding errors
                    logger.log(
                        log_level,
                        f"FFmpeg: {line.decode('utf-8', errors='ignore').strip()}",
                    )
        except Exception as e:
            logger.error(f"Error reading FFmpeg pipe: {e}")
        finally:
            if pipe:
                pipe.close()

    def run_batch_process(self, settings, single_video_path=None):
        """
        This is the main logic that will run in a background thread.
        If single_video_path is provided, only process that one video.
        """

        # Safety init for cleanup variables (must exist for any try/finally path)
        inpainted_reader = None
        splatted_reader = None
        replace_mask_reader = None
        mask_formerge_reader = None
        original_reader = None
        ffmpeg_process = None
        if settings is None:
            self.after(0, self.processing_done, True)
            return

        ct_auto_mode_global = _resolve_ct_auto_mode_from_settings(settings)
        ct_csv_blend_preset_map: Dict[str, Dict[int, int]] = {}
        if (
            bool(settings.get("enable_color_transfer", False))
            and ct_auto_mode_global == CT_AUTO_MODE_CSV_BLEND
        ):
            csv_blend_path = str(settings.get("ct_csv_blend_path", "") or "").strip()
            if not csv_blend_path:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "CSV Blend Auto CT",
                        "CT mode is 'CSV Blend' but no CSV path was provided.",
                    ),
                )
                self.after(0, self.processing_done, True)
                return
            if not os.path.exists(csv_blend_path):
                self.after(
                    0,
                    lambda p=csv_blend_path: messagebox.showerror(
                        "CSV Blend Auto CT",
                        f"CSV Blend map not found:\n{p}",
                    ),
                )
                self.after(0, self.processing_done, True)
                return
            ct_csv_blend_preset_map = self._get_ct_csv_blend_preset_map_cached(
                csv_blend_path
            )
            if not ct_csv_blend_preset_map:
                logger.warning(
                    f"CSV Blend Auto CT map loaded but lookup map is empty: {csv_blend_path}"
                )

        # Single video mode
        if single_video_path and os.path.exists(single_video_path):
            inpainted_videos = [single_video_path]
            single_mode = True
        else:
            inpainted_videos = sorted(
                glob.glob(os.path.join(settings["inpainted_folder"], "*.mp4"))
            )
            single_mode = False

        if not inpainted_videos:
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Info", "No .mp4 files found in the inpainted video folder."
                ),
            )
            self.after(0, self.processing_done)
            return

        # --- NEW: Skip already finished files when Resume is enabled ---
        resume_enabled = settings.get("resume", False)
        if resume_enabled and not single_mode:
            finished_dir = os.path.join(settings["inpainted_folder"], "finished")
            if os.path.isdir(finished_dir):
                finished_files = set(os.listdir(finished_dir))
                original_count = len(inpainted_videos)
                inpainted_videos = [
                    v
                    for v in inpainted_videos
                    if os.path.basename(v) not in finished_files
                ]
                skipped_count = original_count - len(inpainted_videos)
                if skipped_count > 0:
                    logger.info(
                        f"Resume: Skipped {skipped_count} already processed files."
                    )
            else:
                logger.info("Resume: No 'finished' folder found. Processing all files.")

        if not inpainted_videos:
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Info", "All videos have already been processed (Resume mode)."
                ),
            )
            self.after(0, self.processing_done)
            return
        # --- END NEW ---

        # --- NEW: Clear any failed moves from a previous run ---
        self.failed_moves = []

        total_videos = len(inpainted_videos)
        self.progress_bar.config(maximum=total_videos)

        for i, inpainted_video_path in enumerate(inpainted_videos):
            if self.stop_event.is_set():
                logger.info("Processing stopped by user.")
                break

            # In single mode, stop after processing the first video
            if single_mode and i > 0:
                break

            base_name = os.path.basename(inpainted_video_path)
            self.after(
                0,
                self.update_status_label,
                f"Processing {i + 1}/{total_videos}: {base_name}",
            )

            # Initialize readers to None for robust cleanup
            inpainted_reader, splatted_reader, replace_mask_reader, mask_formerge_reader, original_reader = None, None, None, None, None
            original_video_path_to_move = None  # To track which original file to move
            try:
                # --- 1. Find corresponding files (same logic as preview) ---
                core_name_with_width, core_name, is_sbs_input = _parse_inpainted_basename(
                    base_name
                )
                if not core_name_with_width or not core_name:
                    logger.error(
                        f"Could not parse core name from '{base_name}'. Skipping video."
                    )
                    self.after(
                        0, self.progress_var.set, i + 1
                    )  # Still advance progress bar
                    continue

                # --- NEW: Read sidecar file for this clip ---
                clip_sidecar_data = self._read_clip_sidecar(
                    inpainted_video_path, core_name
                )
                logger.info(
                    f"Sidecar for '{core_name}': left_border={clip_sidecar_data.get('left_border')}, right_border={clip_sidecar_data.get('right_border')}"
                )
                left_border = clip_sidecar_data.get("left_border", 0.0)
                right_border = clip_sidecar_data.get("right_border", 0.0)
                self._update_border_info(left_border, right_border)
                # --- END NEW ---

                mask_folder = settings["mask_folder"]
                splatted1_pattern = os.path.join(
                    mask_folder, f"{core_name}_*_splatted1.mp4"
                )
                splatted4_pattern = os.path.join(
                    mask_folder, f"{core_name}_*_splatted4.mp4"
                )
                splatted2_pattern = os.path.join(
                    mask_folder, f"{core_name}_*_splatted2.mp4"
                )
                splatted1_matches = glob.glob(splatted1_pattern)
                splatted4_matches = glob.glob(splatted4_pattern)
                splatted2_matches = glob.glob(splatted2_pattern)

                splatted_file_path = None
                input_layout = "quad"
                if splatted1_matches:
                    splatted_file_path = splatted1_matches[0]
                    input_layout = "single"
                    is_dual_input = False
                elif splatted4_matches:
                    splatted_file_path = splatted4_matches[0]
                    input_layout = "quad"
                    is_dual_input = False
                elif splatted2_matches:
                    splatted_file_path = splatted2_matches[0]
                    input_layout = "dual"
                    is_dual_input = True
                is_quad_input = input_layout == "quad"
                is_single_input = input_layout == "single"

                # 2. Open readers, don't load all frames
                # --- FIX: Validate all file paths before attempting to open them ---
                if not splatted_file_path or not os.path.exists(splatted_file_path):
                    logger.error(
                        f"Missing required splatted file for '{core_name}'. "
                        f"Searched for '{splatted1_pattern}', '{splatted4_pattern}' and '{splatted2_pattern}'. Skipping video."
                    )
                    self.after(0, self.progress_var.set, i + 1)
                    continue

                inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
                splatted_reader = VideoReader(splatted_file_path, ctx=cpu(0))

                # Optional external replace-mask video (binary mkv/mp4)
                replace_mask_reader = None
                replace_mask_path = None
                mask_formerge_reader = None
                mask_formerge_path = None
                use_replace_mask_setting = bool(settings.get("use_replace_mask", False))
                use_mask_formerge_setting = bool(settings.get("use_mask_formerge", False))
                use_replace_mask = bool(use_replace_mask_setting)
                if is_single_input and not use_replace_mask:
                    logger.warning(
                        f"{base_name} uses single-warp input: forcing replace-mask ON."
                    )
                    use_replace_mask = True
                replace_mask_path = self._find_replace_mask_for_splatted(
                    splatted_file_path, settings.get("replace_mask_folder", "")
                )
                has_replace_mask = bool(replace_mask_path and os.path.exists(replace_mask_path))
                if has_replace_mask:
                    try:
                        replace_mask_reader = VideoReader(
                            replace_mask_path, ctx=cpu(0)
                        )
                        if use_replace_mask:
                            logger.info(
                                f"Using external replace mask: {os.path.basename(replace_mask_path)}"
                            )
                        else:
                            logger.info(
                                f"Replace mask available for CT reference: {os.path.basename(replace_mask_path)}"
                            )
                    except Exception as e_rm:
                        logger.warning(
                            f"Failed to open replace mask '{replace_mask_path}': {e_rm}"
                        )
                        replace_mask_reader = None
                        has_replace_mask = False
                elif use_replace_mask and not has_replace_mask and not is_single_input:
                    logger.warning(
                        f"Replace-mask toggle ON but no replace-mask found for '{base_name}'. Falling back to legacy mask for blend."
                    )
                if is_single_input and not has_replace_mask:
                    raise RuntimeError(
                        f"Single-warp input requires replace mask, but none was found/opened for '{base_name}'."
                    )

                # Optional preprocessed mask-for-merge (final blend mask source).
                if use_mask_formerge_setting:
                    mask_formerge_path = self._find_mask_formerge_for_splatted(
                        splatted_file_path, settings.get("mask_formerge_folder", "")
                    )
                    if mask_formerge_path and os.path.exists(mask_formerge_path):
                        try:
                            mask_formerge_reader = VideoReader(mask_formerge_path, ctx=cpu(0))
                            logger.info(
                                f"Using preprocessed mask-for-merge: {os.path.basename(mask_formerge_path)}"
                            )
                        except Exception as e_mf:
                            logger.warning(
                                f"Failed to open mask-for-merge '{mask_formerge_path}': {e_mf}"
                            )
                            mask_formerge_reader = None
                    else:
                        logger.warning(
                            f"Mask-for-merge toggle ON but no preprocessed mask found for '{base_name}'. Falling back to in-GUI mask chain."
                        )

                # Keep explicit flags for later per-chunk behavior.
                has_replace_mask = bool(replace_mask_reader is not None)
                use_replace_mask_effective = bool(has_replace_mask and use_replace_mask)
                use_mask_formerge_effective = bool(mask_formerge_reader is not None)
                ct_available = bool(has_replace_mask)
                ct_enabled_effective = bool(settings.get("enable_color_transfer", False) and ct_available)
                if bool(settings.get("enable_color_transfer", False)) and not ct_available:
                    logger.warning(
                        f"Color Transfer disabled for '{base_name}': replace-mask not available."
                    )

                # --- FIX: Determine original_reader based on input type ---
                original_reader = None  # Assume None initially
                if is_dual_input or is_single_input:  # splatted2 / splatted1
                    # --- MODIFIED: Use helper to find original video with any extension ---
                    original_video_path = self._find_video_by_core_name(
                        settings["original_folder"], core_name
                    )
                    original_video_path_to_move = (
                        original_video_path  # Track for moving later
                    )

                    if original_video_path and os.path.exists(original_video_path):
                        logger.info(
                            f"Found matching original video for {input_layout}-input: {os.path.basename(original_video_path)}"
                        )
                        original_reader = VideoReader(original_video_path, ctx=cpu(0))
                    else:
                        logger.warning(
                            f"Original video not found for {input_layout}-input mode: '{core_name}.*'."
                        )
                        raise RuntimeError(
                            f"Missing original video for {input_layout}-input mode: '{core_name}.*'. "
                            "Output requires original left-eye content."
                        )
                else:  # splatted4 (quad)
                    # For quad-splatted files, the splatted file itself is the source for the left eye.
                    # We can use the splatted_reader as a placeholder to indicate a valid left-eye source exists.
                    original_reader = splatted_reader
                # --- END FIX ---

                # 3. Setup encoder pipe
                num_frames = len(inpainted_reader)
                fps = inpainted_reader.get_avg_fps()
                video_stream_info = get_video_stream_info(inpainted_video_path)

                # Determine output dimensions from a sample frame
                sample_splatted_np = splatted_reader.get_batch([0]).asnumpy()
                _, H_splat, W_splat, _ = sample_splatted_np.shape
                if is_single_input:
                    hires_H, hires_W = H_splat, W_splat
                elif is_dual_input:
                    hires_H, hires_W = H_splat, W_splat // 2
                else:
                    hires_H, hires_W = H_splat // 2, W_splat // 2

                output_format = settings["output_format"]
                if output_format not in self.OUTPUT_FORMAT_CHOICES:
                    raise RuntimeError(
                        f"Unsupported output format '{output_format}'. "
                        f"Supported formats: {', '.join(self.OUTPUT_FORMAT_CHOICES)}"
                    )
                if original_reader is None:
                    raise RuntimeError(
                        f"Original video is missing for '{base_name}'. Cannot render output format '{output_format}'."
                    )

                # --- NEW: Determine output dimensions, perceived width for filename, and suffix ---
                perceived_width_for_filename = hires_W  # Default to single-eye width
                output_height = hires_H

                if output_format == "Full SBS Cross-eye (Right-Left)":
                    output_width = hires_W * 2
                    output_suffix = "_merged_full_sbsx.mp4"
                    # Perceived width is single eye
                elif output_format == "Full SBS (Left-Right)":
                    output_width = hires_W * 2
                    output_suffix = "_merged_full_sbs.mp4"
                    # Perceived width is single eye
                elif output_format == "Double SBS":
                    output_width = hires_W * 2
                    output_height = hires_H * 2
                    output_suffix = "_merged_half_sbs.mp4"
                    perceived_width_for_filename = (
                        hires_W * 2
                    )  # Use the full file width for the filename
                elif output_format == "Half SBS (Left-Right)":
                    output_width = hires_W
                    output_suffix = "_merged_half_sbs.mp4"
                    # Perceived width is single eye, as player will stretch it.
                elif output_format in ["Anaglyph (Red/Cyan)", "Anaglyph Half-Color"]:
                    output_width = hires_W
                    output_suffix = "_merged_anaglyph.mp4"
                    # Perceived width is the full output width
                else:
                    raise RuntimeError(f"Unsupported output format: {output_format}")

                # Construct the final filename using the core name and the new perceived width
                output_filename = (
                    f"{core_name}_{perceived_width_for_filename}{output_suffix}"
                )
                output_path = os.path.join(settings["output_folder"], output_filename)
                # --- END NEW ---

                # --- NEW: Pass padding setting to FFmpeg ---
                ffmpeg_process = start_ffmpeg_pipe_process(
                    content_width=output_width,
                    content_height=output_height,
                    final_output_mp4_path=output_path,
                    fps=fps,
                    video_stream_info=video_stream_info,
                    pad_to_16_9=settings["pad_to_16_9"],
                    output_format_str=output_format,
                )  # Pass the format string

                if ffmpeg_process is None:
                    raise RuntimeError("Failed to start FFmpeg pipe process.")

                # --- NEW: Start threads to read stdout and stderr to prevent deadlock ---
                stdout_thread = threading.Thread(
                    target=self._read_ffmpeg_output,
                    args=(ffmpeg_process.stdout, logging.DEBUG),
                    daemon=True,
                )
                stderr_thread = threading.Thread(
                    target=self._read_ffmpeg_output,
                    args=(ffmpeg_process.stderr, logging.DEBUG),
                    daemon=True,
                )
                stdout_thread.start()
                stderr_thread.start()

                # 4. Loop through chunks
                chunk_size = settings.get("batch_chunk_size", 32)
                ct_usage_counts = {int(p["id"]): 0.0 for p in CT_PRESETS}
                selected_label = _resolve_ct_preset_label(
                    settings.get("ct_preset", CT_PRESET_DEFAULT_LABEL)
                )
                selected_preset = CT_PRESET_BY_LABEL[selected_label]
                ct_auto_mode = _resolve_ct_auto_mode_from_settings(settings)
                csv_blend_weights_by_frame: List[Dict[int, float]] = []
                csv_blend_lookup_key = ""
                if ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
                    csv_rows_by_frame, csv_blend_lookup_key = _lookup_csv_blend_preset_rows(
                        ct_csv_blend_preset_map,
                        inpainted_video_path,
                        core_name_with_width,
                        core_name,
                    )
                    fallback_selected_id = int(selected_preset["id"])
                    csv_target_ids = [fallback_selected_id for _ in range(num_frames)]
                    applied_rows = 0
                    for frame_i, preset_i in csv_rows_by_frame.items():
                        fi = int(frame_i)
                        pid = int(preset_i)
                        if 0 <= fi < num_frames and pid in CT_PRESET_BY_ID:
                            csv_target_ids[fi] = pid
                            applied_rows += 1
                    csv_blend_weights_by_frame, _csv_osc_flags = _build_csv_blend_weights_by_frame(
                        csv_target_ids
                    )
                    if applied_rows <= 0:
                        logger.warning(
                            f"CSV Blend Auto CT: no per-frame presets found for {base_name}; "
                            f"falling back to preset #{fallback_selected_id}."
                        )
                    else:
                        logger.info(
                            f"CSV Blend Auto CT: {base_name} loaded {applied_rows} frame presets "
                            f"(lookup='{csv_blend_lookup_key}')."
                        )
                shadow_state: Dict[str, Any] = {"prev_components": []}
                for frame_start in range(0, num_frames, chunk_size):
                    if self.stop_event.is_set():
                        break

                    frame_end = min(frame_start + chunk_size, num_frames)
                    frame_indices = list(range(frame_start, frame_end))
                    if not frame_indices:
                        break

                    self.after(
                        0,
                        self.update_status_label,
                        f"Processing frames {frame_start + 1}-{frame_end}/{num_frames}...",
                    )

                    # Load current chunk
                    inpainted_np = inpainted_reader.get_batch(frame_indices).asnumpy()
                    splatted_np = splatted_reader.get_batch(frame_indices).asnumpy()

                    replace_mask_np = None
                    if replace_mask_reader is not None:
                        try:
                            replace_mask_np = (
                                replace_mask_reader.get_batch(frame_indices).asnumpy()
                            )
                        except Exception as e_rmread:
                            logger.warning(
                                f"Replace mask read failed for {base_name} frames {frame_start}-{frame_end}: {e_rmread}"
                            )
                            replace_mask_np = None
                    mask_formerge_np = None
                    if mask_formerge_reader is not None:
                        try:
                            mask_formerge_np = (
                                mask_formerge_reader.get_batch(frame_indices).asnumpy()
                            )
                        except Exception as e_mfread:
                            logger.warning(
                                f"Mask-for-merge read failed for {base_name} frames {frame_start}-{frame_end}: {e_mfread}"
                            )
                            mask_formerge_np = None
                    if is_single_input and replace_mask_np is None:
                        raise RuntimeError(
                            f"Single-warp input requires replace mask on every chunk; read failed at frames {frame_start}-{frame_end}."
                        )


                    # Convert to tensors and extract parts (same logic as preview)
                    # ... (this logic is identical to update_preview's frame loading part)
                    inpainted_tensor_full = (
                        torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float()
                        / 255.0
                    )
                    splatted_tensor = (
                        torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float()
                        / 255.0
                    )
                    inpainted = (
                        inpainted_tensor_full[
                            :, :, :, inpainted_tensor_full.shape[3] // 2 :
                        ]
                        if is_sbs_input
                        else inpainted_tensor_full
                    )
                    _, _, H, W = splatted_tensor.shape

                    if is_single_input:
                        if original_reader is None:
                            original_left = torch.zeros_like(inpainted)
                        else:
                            original_np = original_reader.get_batch(
                                frame_indices
                            ).asnumpy()
                            original_left = (
                                torch.from_numpy(original_np)
                                .permute(0, 3, 1, 2)
                                .float()
                                / 255.0
                            )
                        mask_raw = torch.zeros_like(splatted_tensor)
                        warped_original = splatted_tensor
                    elif is_dual_input:
                        # --- NEW: Handle missing original_reader for dual input ---
                        if original_reader is None:
                            # Create a black tensor as a placeholder for the left eye
                            original_left = torch.zeros_like(
                                inpainted
                            )  # Match inpainted shape
                        else:
                            original_np = original_reader.get_batch(
                                frame_indices
                            ).asnumpy()
                            original_left = (
                                torch.from_numpy(original_np)
                                .permute(0, 3, 1, 2)
                                .float()
                                / 255.0
                            )
                        # --- END NEW ---
                        mask_raw = splatted_tensor[:, :, :, : W // 2]
                        warped_original = splatted_tensor[:, :, :, W // 2 :]
                    else:
                        original_left = splatted_tensor[:, :, : H // 2, : W // 2]
                        mask_raw = splatted_tensor[:, :, H // 2 :, : W // 2]
                        warped_original = splatted_tensor[:, :, H // 2 :, W // 2 :]

                    def _gray_mask_batch_from_numpy(arr: np.ndarray) -> torch.Tensor:
                        if arr.ndim == 4 and arr.shape[3] >= 1:
                            gray = arr[..., :3].mean(axis=3)
                        elif arr.ndim == 3:
                            gray = arr
                        else:
                            gray = np.squeeze(arr)
                        gray = gray.astype("float32")
                        if gray.size > 0 and float(np.nanmax(gray)) > 1.5:
                            gray = gray / 255.0
                        gray = np.clip(gray, 0.0, 1.0)
                        return torch.from_numpy(gray).float().unsqueeze(1)

                    legacy_mask_np = mask_raw.permute(0, 2, 3, 1).cpu().numpy()
                    legacy_mask = _gray_mask_batch_from_numpy(legacy_mask_np)
                    replace_mask_batch = (
                        _gray_mask_batch_from_numpy(replace_mask_np)
                        if replace_mask_np is not None
                        else None
                    )
                    mask_formerge_batch = (
                        _gray_mask_batch_from_numpy(mask_formerge_np)
                        if mask_formerge_np is not None
                        else None
                    )

                    use_replace_mask_chunk = bool(
                        use_replace_mask_effective and replace_mask_batch is not None
                    ) or bool(is_single_input)
                    use_mask_formerge_chunk = bool(
                        use_mask_formerge_effective and mask_formerge_batch is not None
                    )
                    if use_mask_formerge_chunk:
                        mask = mask_formerge_batch
                    elif use_replace_mask_chunk and replace_mask_batch is not None:
                        mask = replace_mask_batch
                    else:
                        mask = legacy_mask

                    # CT always uses replace-mask when available.
                    ct_mask = (
                        replace_mask_batch
                        if replace_mask_batch is not None
                        else legacy_mask
                    )
                    ct_enabled_chunk = bool(
                        ct_enabled_effective and replace_mask_batch is not None
                    )

                    # Process chunk
                    use_gpu = settings["use_gpu"] and torch.cuda.is_available()
                    device = "cuda" if use_gpu else "cpu"
                    mask, ct_mask, inpainted, original_left, warped_original = (
                        mask.to(device),
                        ct_mask.to(device),
                        inpainted.to(device),
                        original_left.to(device),
                        warped_original.to(device),
                    )

                    if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                        inpainted = F.interpolate(
                            inpainted,
                            size=(hires_H, hires_W),
                            mode="bicubic",
                            align_corners=False,
                        )
                        mask = F.interpolate(
                            mask,
                            size=(hires_H, hires_W),
                            mode="bilinear",
                            align_corners=False,
                        )
                        ct_mask = F.interpolate(
                            ct_mask,
                            size=(hires_H, hires_W),
                            mode="bilinear",
                            align_corners=False,
                        )

                    # GUI-aligned CT mask: clean binary mask before post-process blend pipeline.
                    if ct_enabled_chunk:
                        mask_bin = (ct_mask > 0.5).float()
                    elif settings["mask_binarize_threshold"] >= 0.0:
                        mask_bin = (ct_mask > settings["mask_binarize_threshold"]).float()
                    else:
                        mask_bin = (ct_mask > 0.5).float()

                    if ct_enabled_chunk:
                        adjusted_frames = []
                        eval_candidate_ids: Optional[List[int]] = None
                        if ct_auto_mode == CT_AUTO_MODE_ON:
                            eval_candidate_ids = list(CT_PRESET_AUTO_EVAL_ORDER)
                        ct_eval_executor = (
                            ThreadPoolExecutor(max_workers=CT_AUTO_EVAL_MAX_WORKERS)
                            if eval_candidate_ids is not None
                            else None
                        )
                        try:
                            for frame_idx in range(inpainted.shape[0]):
                                inpainted_3 = inpainted[frame_idx].cpu()
                                original_left_3 = original_left[frame_idx].cpu()
                                warped_3 = warped_original[frame_idx].cpu()
                                mask_bin_1hw = mask_bin[frame_idx].cpu()

                                if eval_candidate_ids is not None:
                                    best_frame, best_preset_id = _select_best_auto_ct_preset_frame(
                                        inpainted_3=inpainted_3,
                                        original_left_3=original_left_3,
                                        warped_3=warped_3,
                                        mask_bin_1hw=mask_bin_1hw,
                                        settings=settings,
                                        fallback_preset_id=int(selected_preset["id"]),
                                        candidate_preset_ids=eval_candidate_ids,
                                        executor=ct_eval_executor,
                                    )
                                    ct_usage_counts[best_preset_id] += 1.0
                                    adjusted_frames.append(best_frame.to(device))
                                else:
                                    if ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
                                        global_frame_idx = int(frame_indices[frame_idx])
                                        blend_weights = (
                                            csv_blend_weights_by_frame[global_frame_idx]
                                            if 0 <= global_frame_idx < len(csv_blend_weights_by_frame)
                                            else {}
                                        )
                                        if not blend_weights:
                                            fallback_selected_id = int(selected_preset["id"])
                                            blend_weights = {fallback_selected_id: 1.0}
                                        stats_valid_cache: Dict[str, torch.Tensor] = {}
                                        warped_ref_cache: Dict[str, torch.Tensor] = {}
                                        blended_3: Optional[torch.Tensor] = None
                                        for pid_i, weight_i in sorted(
                                            blend_weights.items(),
                                            key=lambda kv: kv[1],
                                            reverse=True,
                                        ):
                                            pid = int(pid_i)
                                            w = float(max(0.0, min(1.0, float(weight_i))))
                                            if w <= 0.0:
                                                continue
                                            preset_i = CT_PRESET_BY_ID.get(pid, selected_preset)
                                            adjusted_3 = _apply_ct_preset_frame(
                                                preset=preset_i,
                                                inpainted_3=inpainted_3,
                                                original_left_3=original_left_3,
                                                warped_3=warped_3,
                                                mask_bin_1hw=mask_bin_1hw,
                                                settings=settings,
                                                stats_valid_cache=stats_valid_cache,
                                                warped_ref_cache=warped_ref_cache,
                                            )
                                            if blended_3 is None:
                                                blended_3 = adjusted_3 * w
                                            else:
                                                blended_3 = blended_3 + (adjusted_3 * w)
                                            ct_usage_counts[pid] += w
                                        if blended_3 is None:
                                            fallback_selected_id = int(selected_preset["id"])
                                            fallback_preset = CT_PRESET_BY_ID[fallback_selected_id]
                                            stats_valid_cache = {}
                                            warped_ref_cache = {}
                                            blended_3 = _apply_ct_preset_frame(
                                                preset=fallback_preset,
                                                inpainted_3=inpainted_3,
                                                original_left_3=original_left_3,
                                                warped_3=warped_3,
                                                mask_bin_1hw=mask_bin_1hw,
                                                settings=settings,
                                                stats_valid_cache=stats_valid_cache,
                                                warped_ref_cache=warped_ref_cache,
                                            )
                                            ct_usage_counts[fallback_selected_id] += 1.0
                                        adjusted_frames.append(
                                            torch.clamp(blended_3, 0.0, 1.0).to(device)
                                        )
                                    else:
                                        selected_id_for_frame = int(selected_preset["id"])
                                        selected_preset_for_frame = selected_preset
                                        stats_valid_cache: Dict[str, torch.Tensor] = {}
                                        warped_ref_cache: Dict[str, torch.Tensor] = {}
                                        adjusted_3 = _apply_ct_preset_frame(
                                            preset=selected_preset_for_frame,
                                            inpainted_3=inpainted_3,
                                            original_left_3=original_left_3,
                                            warped_3=warped_3,
                                            mask_bin_1hw=mask_bin_1hw,
                                            settings=settings,
                                            stats_valid_cache=stats_valid_cache,
                                            warped_ref_cache=warped_ref_cache,
                                        )
                                        ct_usage_counts[selected_id_for_frame] += 1.0
                                        adjusted_frames.append(adjusted_3.to(device))
                        finally:
                            if ct_eval_executor is not None:
                                ct_eval_executor.shutdown(wait=True)

                        inpainted = torch.stack(adjusted_frames)

                    processed_mask = mask.clone()
                    if not use_mask_formerge_chunk:
                        # Skip first binarization when blending from replace-mask.
                        if (
                            (not use_replace_mask_chunk)
                            and settings["mask_binarize_threshold"] >= 0.0
                        ):
                            processed_mask = (
                                mask > settings["mask_binarize_threshold"]
                            ).float()

                        if settings["mask_dilate_kernel_size"] > 0:
                            processed_mask = apply_mask_dilation(
                                processed_mask, settings["mask_dilate_kernel_size"], use_gpu
                            )
                        if settings["mask_blur_kernel_size"] > 0:
                            processed_mask = apply_gaussian_blur(
                                processed_mask, settings["mask_blur_kernel_size"], use_gpu
                            )

                        if int(settings.get("shadow_length_px", 0)) > 0:
                            processed_mask = apply_shadow_blur(
                                processed_mask,
                                base_length_px=int(settings.get("shadow_length_px", 0)),
                                curve=float(settings.get("shadow_curve", 0.0)),
                                motion_gain=float(settings.get("shadow_motion_gain", 0.0)),
                                motion_deadzone_px=float(
                                    settings.get("shadow_motion_deadzone_px", 4.0)
                                ),
                                motion_max_px=float(
                                    settings.get("shadow_motion_max_px", 40.0)
                                ),
                                motion_chain_enabled=bool(
                                    settings.get("shadow_motion_enabled", True)
                                ),
                                area_min_px=float(
                                    settings.get("shadow_area_min_px", 0.0)
                                ),
                                area_max_px=float(
                                    settings.get("shadow_area_max_px", 0.0)
                                ),
                                area_reset_ratio=float(
                                    settings.get("shadow_area_reset_ratio", 1.8)
                                ),
                                area_reset_abs_px=float(
                                    settings.get("shadow_area_reset_abs_px", 0.0)
                                ),
                                component_merge_y_tol_px=int(
                                    settings.get("shadow_component_merge_y_tol_px", 0)
                                ),
                                alpha_down=float(settings.get("shadow_alpha_down", 0.45)),
                                width_adaptive=bool(
                                    settings.get("shadow_width_adaptive", True)
                                ),
                                use_gpu=use_gpu,
                                state=shadow_state,
                                border_tolerance_px=2,
                                width_ref_px=20.0,
                                width_power=1.0,
                            )

                    blended_right_eye = (
                        warped_original * (1 - processed_mask)
                        + inpainted * processed_mask
                    )

                    # --- NEW: Apply borders from sidecar ---
                    left_border = clip_sidecar_data.get("left_border", 0.0)
                    right_border = clip_sidecar_data.get("right_border", 0.0)
                    logger.debug(f"Borders: left={left_border}%, right={right_border}%")
                    if settings.get("add_borders", True) and (
                        left_border > 0 or right_border > 0
                    ):
                        logger.debug(
                            f"Before border: original_left shape={original_left.shape}, blended_right_eye shape={blended_right_eye.shape}"
                        )
                        original_left, blended_right_eye = apply_borders_to_frames(
                            left_border, right_border, original_left, blended_right_eye
                        )
                        logger.debug(
                            f"After border: original_left shape={original_left.shape}, blended_right_eye shape={blended_right_eye.shape}"
                        )
                    # --- END NEW ---

                    # --- NEW: Assemble final frame based on output format ---
                    if output_format == "Full SBS (Left-Right)":
                        final_chunk = torch.cat(
                            [original_left, blended_right_eye], dim=3
                        )
                    elif output_format == "Full SBS Cross-eye (Right-Left)":
                        final_chunk = torch.cat(
                            [blended_right_eye, original_left], dim=3
                        )
                    elif output_format == "Half SBS (Left-Right)":
                        resized_left = F.interpolate(
                            original_left,
                            size=(hires_H, hires_W // 2),
                            mode="bilinear",
                            align_corners=False,
                        )
                        resized_right = F.interpolate(
                            blended_right_eye,
                            size=(hires_H, hires_W // 2),
                            mode="bilinear",
                            align_corners=False,
                        )
                        final_chunk = torch.cat([resized_left, resized_right], dim=3)
                    elif output_format == "Double SBS":
                        sbs_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                        final_chunk = F.interpolate(
                            sbs_chunk,
                            size=(hires_H * 2, hires_W * 2),
                            mode="bilinear",
                            align_corners=False,
                        )
                    elif output_format == "Anaglyph (Red/Cyan)":
                        # Red from Left, Green/Blue from Right
                        final_chunk = torch.cat(
                            [
                                original_left[:, 0:1, :, :],  # R channel from left
                                blended_right_eye[
                                    :, 1:3, :, :
                                ],  # G, B channels from right
                            ],
                            dim=1,
                        )
                    elif output_format == "Anaglyph Half-Color":
                        # Convert left to grayscale for the red channel
                        left_gray = (
                            original_left[:, 0, :, :] * 0.299
                            + original_left[:, 1, :, :] * 0.587
                            + original_left[:, 2, :, :] * 0.114
                        )
                        left_gray = left_gray.unsqueeze(1)  # Add channel dimension back
                        final_chunk = torch.cat(
                            [
                                left_gray,  # R channel from grayscale left
                                blended_right_eye[
                                    :, 1:3, :, :
                                ],  # G, B channels from right
                            ],
                            dim=1,
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported output format during assembly: {output_format}"
                        )
                    # --- END NEW ---

                    cpu_chunk = final_chunk.cpu()
                    for frame_tensor in cpu_chunk:
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()
                        frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(
                            np.uint16
                        )
                        frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)
                        ffmpeg_process.stdin.write(frame_bgr.tobytes())

                    # --- NEW: Draw console progress bar for the current video's chunks ---
                    draw_progress_bar(
                        frame_end, num_frames, prefix=f"  Encoding {base_name}:"
                    )

                # 5. Finalize FFmpeg process
                if ct_enabled_effective:
                    total_ct = float(sum(ct_usage_counts.values()))
                    if total_ct > 0:
                        ct_line = " ".join(
                            [
                                f"{pid}:{(100.0 * ct_usage_counts.get(pid, 0) / total_ct):.1f}%"
                                for pid in range(1, 9)
                            ]
                        )
                        logger.info(f"CT usage [{base_name}] {ct_line}")
                        if ct_auto_mode == CT_AUTO_MODE_ON:
                            best_pid = max(
                                range(1, 9), key=lambda pid: ct_usage_counts.get(pid, 0)
                            )
                            best_pct = (
                                100.0 * ct_usage_counts.get(best_pid, 0) / total_ct
                            )
                            self.after(
                                0,
                                lambda t=f"Auto CT best: #{best_pid} ({best_pct:.1f}%)": self.auto_ct_best_var.set(
                                    t
                                ),
                            )
                        elif ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
                            best_pid = max(
                                range(1, 9), key=lambda pid: ct_usage_counts.get(pid, 0.0)
                            )
                            best_pct = (
                                100.0 * ct_usage_counts.get(best_pid, 0.0) / total_ct
                            )
                            self.after(
                                0,
                                lambda t=f"Auto CT CSV Blend: #{best_pid} ({best_pct:.1f}%)": self.auto_ct_best_var.set(
                                    t
                                ),
                            )
                        else:
                            selected_id = int(selected_preset["id"])
                            self.after(
                                0,
                                lambda t=f"Auto CT best: #{selected_id} (manual)": self.auto_ct_best_var.set(
                                    t
                                ),
                            )

                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.close()

                # --- FIX: Wait for the process to finish FIRST, then join threads ---
                ffmpeg_process.wait(timeout=120)  # Wait for ffmpeg to exit
                stdout_thread.join(timeout=5)  # Wait for stdout reader to finish
                stderr_thread.join(timeout=5)  # Wait for stderr reader to finish
                # --- END FIX ---

                if ffmpeg_process.returncode != 0:
                    logger.error(
                        f"FFmpeg encoding failed for {base_name}. Check console for details."
                    )
                elif self.stop_event.is_set():
                    logger.warning(
                        f"Processing was stopped for {base_name}. Source files will not be moved."
                    )
                    # Do not queue files for moving if the job was stopped.
                else:
                    logger.debug(
                        "FFmpeg process and threads terminated, proceeding to move files."
                    )
                    logger.info(f"Successfully encoded video to {output_path}")

                    # Explicitly close video readers BEFORE attempting to move their files
                    del ffmpeg_process
                    if inpainted_reader:
                        inpainted_reader = None
                    if splatted_reader:
                        splatted_reader = None
                    if replace_mask_reader:
                        replace_mask_reader = None
                    if mask_formerge_reader:
                        mask_formerge_reader = None
                    if original_reader:
                        original_reader = None
                    inpainted_reader, splatted_reader, replace_mask_reader, mask_formerge_reader, original_reader = (
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    time.sleep(0.1)  # Give OS a moment to release file handles
                    logger.debug("Source video file handles released.")

                    # --- NEW: Move files to finished folder if Resume is enabled ---
                    if settings.get("resume", False):
                        self.cleanup_queue.put(
                            (inpainted_video_path, settings["inpainted_folder"])
                        )
                        self.cleanup_queue.put(
                            (splatted_file_path, settings["mask_folder"])
                        )
                        if replace_mask_path and os.path.exists(replace_mask_path):
                            self.cleanup_queue.put(
                                (replace_mask_path, os.path.dirname(replace_mask_path))
                            )
                        if original_video_path_to_move:
                            self.cleanup_queue.put(
                                (
                                    original_video_path_to_move,
                                    settings["original_folder"],
                                )
                            )
                            # Also move sidecar for original video
                            original_base = os.path.splitext(
                                original_video_path_to_move
                            )[0]
                            for ext in [".fssidecar", ".json"]:
                                sidecar_path = original_base + ext
                                if os.path.exists(sidecar_path):
                                    self.cleanup_queue.put(
                                        (sidecar_path, settings["original_folder"])
                                    )
                        # Also move sidecar if it exists
                        inpainted_base = os.path.splitext(inpainted_video_path)[0]
                        for ext in [".fssidecar", ".json"]:
                            sidecar_path = inpainted_base + ext
                            if os.path.exists(sidecar_path):
                                self.cleanup_queue.put(
                                    (sidecar_path, settings["inpainted_folder"])
                                )
                    # --- END NEW ---
            except Exception as e:
                # --- FIX: Ensure readers are closed on exception before the finally block ---
                if splatted_reader:
                    splatted_reader = None
                if replace_mask_reader:
                    replace_mask_reader = None
                if mask_formerge_reader:
                    mask_formerge_reader = None
                if original_reader:
                    original_reader = None
                inpainted_reader, splatted_reader, replace_mask_reader, mask_formerge_reader, original_reader = None, None, None, None, None
                # --- END FIX ---
                logger.error(f"Failed to process {base_name}: {e}", exc_info=True)
                self.after(
                    0,
                    lambda base_name=base_name, e=e: messagebox.showerror(
                        "Processing Error",
                        f"An error occurred while processing {base_name}:\n\n{e}",
                    ),
                )
                # --- NEW: Stop the entire batch if one video fails critically ---
                self.stop_event.set()
                # --- END NEW ---
            finally:
                # Ensure readers are always cleaned up, even on error
                # This is now a secondary safety net; the primary cleanup happens before file moves.
                if inpainted_reader:
                    inpainted_reader = None
                if splatted_reader:
                    splatted_reader = None
                if replace_mask_reader:
                    replace_mask_reader = None
                if mask_formerge_reader:
                    mask_formerge_reader = None
                if original_reader:
                    original_reader = None
                # --- END: CHUNK-BASED PROCESSING ---

            self.after(0, self.progress_var.set, i + 1)

        # --- NEW: Signal the cleanup worker to stop after it finishes its queue ---
        self.cleanup_queue.put(None)
        logger.info(
            "Main processing loop finished. Stop signal sent to cleanup worker."
        )

        self.after(0, self.processing_done, self.stop_event.is_set())

    def restore_finished_files(self):
        """Moves all files from 'finished' subfolders back to their parent directories."""
        if not messagebox.askyesno(
            "Restore Finished Files",
            "Are you sure you want to move all processed videos from the 'finished' folders back to their respective input directories?",
        ):
            return

        folders_to_check = {
            "Inpainted": self.inpainted_folder_var.get(),
            "Original": self.original_folder_var.get(),
            "Splat": self.mask_folder_var.get(),
        }

        restored_count = 0
        error_count = 0

        for folder_name, base_folder in folders_to_check.items():
            if not base_folder or not os.path.isdir(base_folder):
                logger.warning(
                    f"Skipping restore for '{folder_name}' folder: Path is not a valid directory ('{base_folder}')."
                )
                continue

            finished_dir = os.path.join(base_folder, "finished")
            if os.path.isdir(finished_dir):
                logger.info(f"Checking for files to restore in: {finished_dir}")
                for filename in os.listdir(finished_dir):
                    src_path = os.path.join(finished_dir, filename)
                    dest_path = os.path.join(base_folder, filename)
                    try:
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                        logger.debug(f"Restored '{filename}' to '{base_folder}'")
                    except Exception as e:
                        error_count += 1
                        logger.error(
                            f"Error restoring file '{filename}': {e}", exc_info=True
                        )
            else:
                logger.info(
                    f"No 'finished' subfolder found in '{base_folder}'. Nothing to restore."
                )

        messagebox.showinfo(
            "Restore Complete",
            f"Restore operation finished.\n\nFiles Restored: {restored_count}\nErrors: {error_count}",
        )
        self.update_status_label(
            f"Restore complete. Moved {restored_count} files with {error_count} errors."
        )

        # --- NEW: Reset video list scan flag and refresh preview ---
        if hasattr(self, "previewer"):
            self.previewer.reset_video_list_scan()
            # Trigger a full refresh scan
            if self.previewer.find_sources_callback:
                self.previewer.load_video_list(
                    find_sources_callback=self.previewer.find_sources_callback
                )
        # --- END NEW ---

    def _find_preview_sources_callback(self) -> list:
        """
        A callback function for the VideoPreviewer.
        It scans the folders and returns a list of dictionaries,
        where each dictionary contains the paths to the source files for one video.
        """
        inpainted_folder = self.inpainted_folder_var.get()
        if not os.path.isdir(inpainted_folder):
            messagebox.showerror(
                "Error", "Inpainted Video Folder is not a valid directory."
            )
            return []

        all_mp4s = sorted(glob.glob(os.path.join(inpainted_folder, "*.mp4")))
        valid_inpainted_videos = [
            f
            for f in all_mp4s
            if f.endswith("_inpainted_right_eye.mp4")
            or f.endswith("_inpainted_sbs.mp4")
        ]

        video_source_list = []
        single_missing_replace: List[str] = []
        self._clear_border_info()  # Clear border info before scanning

        for inpainted_path in valid_inpainted_videos:
            base_name = os.path.basename(inpainted_path)
            inpaint_suffix = "_inpainted_right_eye.mp4"
            logger.debug(f"Preview Scan: Checking '{base_name}'...")
            sbs_suffix = "_inpainted_sbs.mp4"

            is_sbs_input = False  # Assume single-eye unless proven otherwise

            if base_name.endswith(inpaint_suffix):
                core_name_with_width = base_name[: -len(inpaint_suffix)]
            elif base_name.endswith(sbs_suffix):
                core_name_with_width = base_name[: -len(sbs_suffix)]
                is_sbs_input = True  # Set flag for double-wide inpainted video
            else:
                continue

            last_underscore_idx = core_name_with_width.rfind("_")
            if last_underscore_idx == -1:
                logger.warning(
                    f"Preview Scan: Skipping '{base_name}'. Could not determine core name (expected format '..._width_suffix.mp4')."
                )
                continue
            core_name = core_name_with_width[:last_underscore_idx]

            # --- NEW: Read sidecar file for this clip ---
            clip_sidecar_data = self._read_clip_sidecar(inpainted_path, core_name)
            logger.debug(
                f"Preview Scan: Sidecar for '{core_name}': convergence_plane={clip_sidecar_data.get('convergence_plane')}, max_disparity={clip_sidecar_data.get('max_disparity')}, left_border={clip_sidecar_data.get('left_border')}, right_border={clip_sidecar_data.get('right_border')}"
            )
            left_border = clip_sidecar_data.get("left_border", 0.0)
            right_border = clip_sidecar_data.get("right_border", 0.0)
            self._update_border_info(left_border, right_border)
            # --- END NEW ---

            mask_folder = self.mask_folder_var.get()
            splatted1_pattern = os.path.join(
                mask_folder, f"{core_name}_*_splatted1.mp4"
            )
            splatted4_pattern = os.path.join(
                mask_folder, f"{core_name}_*_splatted4.mp4"
            )
            splatted2_pattern = os.path.join(
                mask_folder, f"{core_name}_*_splatted2.mp4"
            )
            logger.debug(
                f"  - Searching for splatted file with patterns: '{splatted1_pattern}', '{splatted4_pattern}' and '{splatted2_pattern}'"
            )
            splatted1_matches = glob.glob(splatted1_pattern)
            splatted4_matches = glob.glob(splatted4_pattern)
            splatted2_matches = glob.glob(splatted2_pattern)

            source_dict = {
                "inpainted": inpainted_path,
                "splatted": None,
                "original": None,
                "replace_mask": None,
                "mask_formerge": None,
                "is_sbs_input": is_sbs_input,
                "is_quad_input": False,
                "is_single_input": False,
                "input_layout": "quad",
                "sidecar": clip_sidecar_data,  # Store sidecar data for borders
            }

            if splatted1_matches:
                splatted_path = splatted1_matches[0]
                logger.debug(
                    f"  - Found single-warp match: {os.path.basename(splatted_path)}"
                )
                source_dict["splatted"] = splatted_path
                source_dict["is_single_input"] = True
                source_dict["input_layout"] = "single"
                source_dict["replace_mask"] = self._find_replace_mask_for_splatted(
                    splatted_path, self.replace_mask_folder_var.get()
                )
                source_dict["mask_formerge"] = self._find_mask_formerge_for_splatted(
                    splatted_path, self.mask_formerge_folder_var.get()
                )
                original_path = self._find_video_by_core_name(
                    self.original_folder_var.get(), core_name
                )
                if original_path:
                    source_dict["original"] = original_path
                else:
                    logger.warning(
                        f"  - For single-warp input '{base_name}', the original video '{core_name}.*' was not found."
                    )
                if not self._path_exists(source_dict.get("replace_mask")):
                    single_missing_replace.append(base_name)
                    logger.error(
                        f"Single-warp input '{base_name}' missing replace-mask. Skipping preview entry."
                    )
                    continue
            elif splatted4_matches:
                splatted_path = splatted4_matches[0]
                logger.debug(
                    f"  - Found quad-splatted match: {os.path.basename(splatted_path)}"
                )
                source_dict["splatted"] = splatted_path
                source_dict["is_quad_input"] = True  # Set flag for quad-splatted input
                source_dict["input_layout"] = "quad"
                source_dict["replace_mask"] = self._find_replace_mask_for_splatted(splatted_path, self.replace_mask_folder_var.get())
                source_dict["mask_formerge"] = self._find_mask_formerge_for_splatted(
                    splatted_path, self.mask_formerge_folder_var.get()
                )
                # 'original' remains None, which is the necessary structural fix for the crash
            elif splatted2_matches:
                splatted_path = splatted2_matches[0]
                logger.debug(
                    f"  - Found dual-splatted match: {os.path.basename(splatted_path)}"
                )
                source_dict["splatted"] = splatted_path
                source_dict["input_layout"] = "dual"
                source_dict["replace_mask"] = self._find_replace_mask_for_splatted(splatted_path, self.replace_mask_folder_var.get())
                source_dict["mask_formerge"] = self._find_mask_formerge_for_splatted(
                    splatted_path, self.mask_formerge_folder_var.get()
                )
                original_path = self._find_video_by_core_name(
                    self.original_folder_var.get(), core_name
                )

                if original_path:
                    logger.debug(
                        f"  - Found matching original video: {os.path.basename(original_path)}"
                    )
                    source_dict["original"] = original_path
                else:
                    logger.warning(
                        f"  - For dual-splatted input '{base_name}', the original video '{core_name}.*' was not found. It will be treated as optional."
                    )
            else:
                logger.warning(
                    f"Preview Scan: Skipping '{base_name}'. No matching splatted file found in '{mask_folder}'."
                )
                continue  # Skip to the next video if no splatted file is found

            video_source_list.append(source_dict)
        if any(bool(v.get("is_single_input", False)) for v in video_source_list):
            if not self.use_replace_mask_var.get():
                logger.info(
                    "Single-warp input detected in preview list: forcing 'Use Replace Mask' ON."
                )
                self.use_replace_mask_var.set(True)
        if single_missing_replace:
            names = "\n".join(single_missing_replace[:8])
            extra = "" if len(single_missing_replace) <= 8 else f"\n... (+{len(single_missing_replace)-8} more)"
            messagebox.showerror(
                "Missing Replace Mask",
                "Single-warp clips require replace-mask and were excluded from preview:\n\n"
                f"{names}{extra}",
            )
        self._last_mode_constraints_video_index = -999999
        self.after_idle(lambda: self._refresh_mode_constraints(trigger_preview=False))
        return video_source_list

    def _preview_processing_callback(
        self, source_frames: dict, params: dict
    ) -> Optional[Image.Image]:
        """
        This function contains the actual blending logic for the preview.
        It's called by the VideoPreviewer module.
        """
        try:
            # --- FIX: Always get the latest parameters when the preview is updated ---
            # This ensures that changing the preview source uses the current slider values.
            params = self.get_current_settings()
            if not params:
                return None  # Exit if settings are invalid
            # --- END FIX ---
            # 1. Extract tensors from the source_frames dict
            inpainted_tensor_full = source_frames.get("inpainted")
            splatted_tensor = source_frames.get("splatted")
            original_tensor = source_frames.get(
                "original"
            )  # Will be None for quad input

            if inpainted_tensor_full is None or splatted_tensor is None:
                raise ValueError(
                    "Missing 'inpainted' or 'splatted' source for preview."
                )

            # --- FIX: Determine input type based on metadata from the video list ---
            current_source_metadata = self.previewer.video_list[
                self.previewer.current_video_index
            ]
            current_video_index = int(self.previewer.current_video_index)
            if current_video_index != self._last_mode_constraints_video_index:
                self._last_mode_constraints_video_index = current_video_index
                self.after_idle(lambda: self._refresh_mode_constraints(trigger_preview=False))
            is_sbs_input = current_source_metadata.get("is_sbs_input", False)
            is_quad_input = current_source_metadata.get(
                "is_quad_input", False
            )  # <--- GET NEW FLAG
            is_single_input = current_source_metadata.get("is_single_input", False)
            input_layout = current_source_metadata.get("input_layout", "")
            if input_layout not in {"single", "dual", "quad"}:
                if is_quad_input:
                    input_layout = "quad"
                elif is_single_input:
                    input_layout = "single"
                else:
                    input_layout = "dual"
            # --- END FIX ---

            # 2. Determine input types and extract frame parts
            # Use the correct is_sbs_input flag to extract the right eye if the input is SBS
            inpainted = (
                inpainted_tensor_full[:, :, :, inpainted_tensor_full.shape[3] // 2 :]
                if is_sbs_input
                else inpainted_tensor_full
            )

            # Extract parts from the splatted frame
            _, _, H, W = splatted_tensor.shape

            # --- FIX: Use is_quad_input for reliable tensor extraction ---
            if input_layout == "quad":  # Splatted4 (Original Left and Mask/Warped are all inside the splatted file)
                half_h, half_w = H // 2, W // 2
                original_left = splatted_tensor[:, :, :half_h, :half_w]
                depth_map_vis = splatted_tensor[:, :, :half_h, half_w:]
                mask_raw = splatted_tensor[:, :, half_h:, :half_w]
                right_eye_original = splatted_tensor[:, :, half_h:, half_w:]
                is_dual_input = False  # For clarity
            elif input_layout == "single":  # Splatted1 (only warped frame, mask must come from replace-mask)
                original_left = original_tensor
                depth_map_vis = None
                right_eye_original = splatted_tensor
                mask_raw = torch.zeros_like(splatted_tensor)
                is_dual_input = False
            else:  # Splatted2 (Original Left is a separate file provided by original_tensor)
                half_w = W // 2
                mask_raw = splatted_tensor[:, :, :, :half_w]
                right_eye_original = splatted_tensor[:, :, :, half_w:]
                original_left = original_tensor
                depth_map_vis = None
                is_dual_input = True  # For clarity

            if original_left is None:
                original_left = torch.zeros_like(inpainted)

            # Configure preview source dropdown based on input type
            preview_options = [
                "Blended Image",
                "Original (Left Eye)",
                "Warped (Right BG)",
                "Inpainted Right Eye",  # <--- ADDED INPAINTED
                "Processed Mask",
                "Anaglyph 3D",
                "Dubois Anaglyph",  # <--- ADDED ANAGLYPH
                "Optimized Anaglyph",  # <--- ADDED ANAGLYPH
                "Wigglegram",
            ]
            if input_layout == "quad":  # Depth map is only in quad-splatted files
                preview_options.append("Depth Map")
            self.previewer.set_preview_source_options(preview_options)

            def _gray_mask_from_tensor(frame_tensor: torch.Tensor) -> torch.Tensor:
                frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                gray_np = np.mean(frame_np[..., :3], axis=2) if frame_np.ndim == 3 else frame_np
                if gray_np.size > 0 and float(np.nanmax(gray_np)) > 1.5:
                    gray_np = gray_np / 255.0
                gray_np = np.clip(gray_np, 0.0, 1.0)
                return torch.from_numpy(gray_np).float().unsqueeze(0).unsqueeze(0)

            def _mask_from_splatted_tensor(sp_tensor: torch.Tensor) -> torch.Tensor:
                _h = int(sp_tensor.shape[2])
                _w = int(sp_tensor.shape[3])
                if input_layout == "quad":
                    _half_h, _half_w = _h // 2, _w // 2
                    _mask_raw = sp_tensor[:, :, _half_h:, :_half_w]
                elif input_layout == "single":
                    _mask_raw = torch.zeros_like(sp_tensor[:, :, :, :1]).repeat(1, 1, 1, _w)
                else:
                    _half_w = _w // 2
                    _mask_raw = sp_tensor[:, :, :, :_half_w]
                return _gray_mask_from_tensor(_mask_raw)

            def _load_mask_for_preview_frame(frame_idx: int) -> Optional[torch.Tensor]:
                readers = getattr(self.previewer, "source_readers", {}) or {}
                effective_use_replace_mask = bool(
                    params.get("use_replace_mask", False) or input_layout == "single"
                )
                if effective_use_replace_mask:
                    rm_reader = readers.get("replace_mask")
                    if rm_reader is not None:
                        try:
                            rm_np = rm_reader.get_batch([int(frame_idx)]).asnumpy()
                            rm_tensor = (
                                torch.from_numpy(rm_np).permute(0, 3, 1, 2).float() / 255.0
                            )
                            return _gray_mask_from_tensor(rm_tensor)
                        except Exception as e_rm:
                            logger.debug(
                                f"Preview temporal shadow: replace_mask read failed at frame {frame_idx}: {e_rm}"
                            )
                sp_reader = readers.get("splatted")
                if sp_reader is None:
                    return None
                try:
                    sp_np = sp_reader.get_batch([int(frame_idx)]).asnumpy()
                    sp_tensor = torch.from_numpy(sp_np).permute(0, 3, 1, 2).float() / 255.0
                    return _mask_from_splatted_tensor(sp_tensor)
                except Exception as e_sp:
                    logger.debug(
                        f"Preview temporal shadow: splatted read failed at frame {frame_idx}: {e_sp}"
                    )
                    return None

            # Convert mask to grayscale (optionally using an external replace-mask video)
            replace_mask_tensor = source_frames.get("replace_mask")
            mask_formerge_tensor = source_frames.get("mask_formerge")
            has_replace_mask = replace_mask_tensor is not None
            has_mask_formerge = mask_formerge_tensor is not None

            if input_layout == "single" and not has_replace_mask:
                raise RuntimeError(
                    "Single-warp input requires replace-mask, but it is missing for preview."
                )

            effective_use_replace_mask = bool(
                (params.get("use_replace_mask", False) and has_replace_mask)
                or input_layout == "single"
            )
            use_mask_formerge_effective = bool(
                params.get("use_mask_formerge", False) and has_mask_formerge
            )

            if use_mask_formerge_effective:
                mask = _gray_mask_from_tensor(mask_formerge_tensor)
            elif effective_use_replace_mask and has_replace_mask:
                mask = _gray_mask_from_tensor(replace_mask_tensor)
            else:
                mask = _gray_mask_from_tensor(mask_raw)

            # CT always consumes replace-mask when available (even if blend uses legacy mask).
            ct_mask_source = (
                _gray_mask_from_tensor(replace_mask_tensor)
                if has_replace_mask
                else _gray_mask_from_tensor(mask_raw)
            )
            ct_available = bool(has_replace_mask)

            # 3. Process the frames
            # Define the processing device based on the 'use_gpu' parameter
            use_gpu = params.get("use_gpu", False) and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"

            # Move tensors to the processing device
            mask = mask.to(device)
            inpainted = inpainted.to(device)
            original_left = original_left.to(device)
            right_eye_original = right_eye_original.to(device)

            hires_H, hires_W = right_eye_original.shape[2], right_eye_original.shape[3]
            if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                logger.debug(
                    f"Upscaling preview frames from {inpainted.shape[3]}x{inpainted.shape[2]} to {hires_W}x{hires_H}"
                )
                inpainted = F.interpolate(
                    inpainted,
                    size=(hires_H, hires_W),
                    mode="bicubic",
                    align_corners=False,
                )
                mask = F.interpolate(
                    mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False
                )
            if (
                ct_mask_source.shape[2] != hires_H
                or ct_mask_source.shape[3] != hires_W
            ):
                ct_mask_source = F.interpolate(
                    ct_mask_source,
                    size=(hires_H, hires_W),
                    mode="bilinear",
                    align_corners=False,
                )

            def _preprocess_mask_for_shadow(mask_in: torch.Tensor) -> torch.Tensor:
                proc = mask_in.clone()
                if use_mask_formerge_effective:
                    return proc
                # Skip the initial binarize step when blending from replace-mask.
                if (
                    not effective_use_replace_mask
                    and params.get("mask_binarize_threshold", -1.0) >= 0.0
                ):
                    proc = (proc > params["mask_binarize_threshold"]).float()
                if params.get("mask_dilate_kernel_size", 0) > 0:
                    proc = apply_mask_dilation(
                        proc, int(params["mask_dilate_kernel_size"]), use_gpu
                    )
                if params.get("mask_blur_kernel_size", 0) > 0:
                    proc = apply_gaussian_blur(
                        proc, int(params["mask_blur_kernel_size"]), use_gpu
                    )
                return proc

            processed_mask = _preprocess_mask_for_shadow(mask)
            current_pre_shadow_mask = processed_mask
            shadow_len_px = int(params.get("shadow_length_px", 0))
            use_temporal_shadow_preview = bool(
                params.get("preview_shadow_temporal", False)
            )
            warmup_frames = int(params.get("preview_shadow_warmup_frames", 20))
            shadow_motion_gain = float(params.get("shadow_motion_gain", 0.0))
            shadow_motion_enabled = bool(params.get("shadow_motion_enabled", True))
            shadow_motion_deadzone_px = float(
                params.get("shadow_motion_deadzone_px", 4.0)
            )

            if shadow_len_px > 0 and not use_mask_formerge_effective:
                if (
                    use_temporal_shadow_preview
                    and shadow_motion_gain > 0.0
                    and shadow_motion_enabled
                    and warmup_frames > 0
                ):
                    preview_frame_idx = int(self.previewer.frame_scrubber_var.get())
                    source_key = str(current_source_metadata.get("inpainted", "") or "")
                    shadow_cache_key = (
                        source_key,
                        bool(use_mask_formerge_effective),
                        bool(effective_use_replace_mask),
                        float(params.get("mask_binarize_threshold", -1.0)),
                        int(params.get("mask_dilate_kernel_size", 0)),
                        int(params.get("mask_blur_kernel_size", 0)),
                        int(shadow_len_px),
                        float(params.get("shadow_curve", 0.0)),
                        float(shadow_motion_gain),
                        bool(shadow_motion_enabled),
                        float(shadow_motion_deadzone_px),
                        float(params.get("shadow_motion_max_px", 40.0)),
                        float(params.get("shadow_area_min_px", 0.0)),
                        float(params.get("shadow_area_max_px", 0.0)),
                        float(params.get("shadow_area_reset_ratio", 1.8)),
                        float(params.get("shadow_area_reset_abs_px", 0.0)),
                        int(params.get("shadow_component_merge_y_tol_px", 0)),
                        float(params.get("shadow_alpha_down", 0.45)),
                        bool(params.get("shadow_width_adaptive", True)),
                        int(warmup_frames),
                        int(hires_H),
                        int(hires_W),
                    )

                    cache = self._preview_shadow_temporal_cache
                    can_step_forward = (
                        cache is not None
                        and cache.get("key") == shadow_cache_key
                        and int(cache.get("frame_idx", -999999)) + 1 == int(preview_frame_idx)
                        and isinstance(cache.get("state"), dict)
                    )

                    if can_step_forward:
                        # Fast path for sequential frame navigation: re-use previous temporal state.
                        shadow_state = cache["state"]
                        processed_mask = apply_shadow_blur(
                            current_pre_shadow_mask,
                            base_length_px=shadow_len_px,
                            curve=float(params.get("shadow_curve", 0.0)),
                            motion_gain=shadow_motion_gain,
                            motion_deadzone_px=shadow_motion_deadzone_px,
                            motion_max_px=float(params.get("shadow_motion_max_px", 40.0)),
                            motion_chain_enabled=shadow_motion_enabled,
                            area_min_px=float(params.get("shadow_area_min_px", 0.0)),
                            area_max_px=float(params.get("shadow_area_max_px", 0.0)),
                            area_reset_ratio=float(params.get("shadow_area_reset_ratio", 1.8)),
                            area_reset_abs_px=float(params.get("shadow_area_reset_abs_px", 0.0)),
                            component_merge_y_tol_px=int(params.get("shadow_component_merge_y_tol_px", 0)),
                            alpha_down=float(params.get("shadow_alpha_down", 0.45)),
                            width_adaptive=bool(
                                params.get("shadow_width_adaptive", True)
                            ),
                            use_gpu=use_gpu,
                            state=shadow_state,
                            border_tolerance_px=2,
                            width_ref_px=20.0,
                            width_power=1.0,
                        )
                    else:
                        # Fallback: rebuild from a short warmup window so preview remains close to batch.
                        warmup_start_idx = max(0, preview_frame_idx - warmup_frames)
                        shadow_state = {"prev_components": []}
                        for warmup_idx in range(warmup_start_idx, preview_frame_idx + 1):
                            if warmup_idx == preview_frame_idx:
                                warmup_mask = current_pre_shadow_mask
                            else:
                                warmup_raw_mask = _load_mask_for_preview_frame(warmup_idx)
                                if warmup_raw_mask is None:
                                    continue
                                warmup_raw_mask = warmup_raw_mask.to(device)
                                if (
                                    warmup_raw_mask.shape[2] != hires_H
                                    or warmup_raw_mask.shape[3] != hires_W
                                ):
                                    warmup_raw_mask = F.interpolate(
                                        warmup_raw_mask,
                                        size=(hires_H, hires_W),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                warmup_mask = _preprocess_mask_for_shadow(warmup_raw_mask)
                            processed_mask = apply_shadow_blur(
                                warmup_mask,
                                base_length_px=shadow_len_px,
                                curve=float(params.get("shadow_curve", 0.0)),
                                motion_gain=shadow_motion_gain,
                                motion_deadzone_px=shadow_motion_deadzone_px,
                                motion_max_px=float(params.get("shadow_motion_max_px", 40.0)),
                                motion_chain_enabled=shadow_motion_enabled,
                                area_min_px=float(params.get("shadow_area_min_px", 0.0)),
                                area_max_px=float(params.get("shadow_area_max_px", 0.0)),
                                area_reset_ratio=float(params.get("shadow_area_reset_ratio", 1.8)),
                                area_reset_abs_px=float(params.get("shadow_area_reset_abs_px", 0.0)),
                                component_merge_y_tol_px=int(params.get("shadow_component_merge_y_tol_px", 0)),
                                alpha_down=float(params.get("shadow_alpha_down", 0.45)),
                                width_adaptive=bool(
                                    params.get("shadow_width_adaptive", True)
                                ),
                                use_gpu=use_gpu,
                                state=shadow_state,
                                border_tolerance_px=2,
                                width_ref_px=20.0,
                                width_power=1.0,
                            )

                    self._preview_shadow_temporal_cache = {
                        "key": shadow_cache_key,
                        "frame_idx": int(preview_frame_idx),
                        "state": shadow_state,
                    }
                else:
                    self._preview_shadow_temporal_cache = None
                    processed_mask = apply_shadow_blur(
                        processed_mask,
                        base_length_px=shadow_len_px,
                        curve=float(params.get("shadow_curve", 0.0)),
                        motion_gain=shadow_motion_gain,
                        motion_deadzone_px=shadow_motion_deadzone_px,
                        motion_max_px=float(params.get("shadow_motion_max_px", 40.0)),
                        motion_chain_enabled=shadow_motion_enabled,
                        area_min_px=float(params.get("shadow_area_min_px", 0.0)),
                        area_max_px=float(params.get("shadow_area_max_px", 0.0)),
                        area_reset_ratio=float(params.get("shadow_area_reset_ratio", 1.8)),
                        area_reset_abs_px=float(params.get("shadow_area_reset_abs_px", 0.0)),
                        component_merge_y_tol_px=int(params.get("shadow_component_merge_y_tol_px", 0)),
                        alpha_down=float(params.get("shadow_alpha_down", 0.45)),
                        width_adaptive=bool(
                            params.get("shadow_width_adaptive", True)
                        ),
                        use_gpu=use_gpu,
                        state=None,  # Fast stateless preview path.
                        border_tolerance_px=2,
                        width_ref_px=20.0,
                            width_power=1.0,
                    )
            else:
                self._preview_shadow_temporal_cache = None
            processed_mask = processed_mask.squeeze(0)  # Remove batch dim

            if (
                params.get("enable_color_transfer", False)
                and ct_available
                and original_left is not None
            ):
                selected_label = _resolve_ct_preset_label(
                    params.get("ct_preset", CT_PRESET_DEFAULT_LABEL)
                )
                selected_preset = CT_PRESET_BY_LABEL[selected_label]
                ct_auto_mode = _resolve_ct_auto_mode_from_settings(params)
                csv_blend_weights_for_frame: Dict[int, float] = {}
                csv_detected_preset_for_frame = int(selected_preset["id"])
                if ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
                    csv_blend_path = str(params.get("ct_csv_blend_path", "") or "").strip()
                    csv_blend_map = (
                        self._get_ct_csv_blend_preset_map_cached(csv_blend_path)
                        if csv_blend_path
                        else {}
                    )
                    preview_inpainted_path = str(
                        current_source_metadata.get("inpainted", "") or ""
                    )
                    preview_base_name = os.path.basename(preview_inpainted_path)
                    preview_core_with_width, preview_core_name, _ = _parse_inpainted_basename(
                        preview_base_name
                    )
                    csv_rows_by_frame, _csv_lookup_key = _lookup_csv_blend_preset_rows(
                        csv_blend_map,
                        preview_inpainted_path,
                        preview_core_with_width,
                        preview_core_name,
                    )
                    fallback_selected_id = int(selected_preset["id"])
                    preview_frame_idx = int(self.previewer.frame_scrubber_var.get())
                    seq_len = max(
                        preview_frame_idx + 1,
                        int(current_source_metadata.get("total_frames", 0) or 0),
                        (max(csv_rows_by_frame.keys()) + 1) if csv_rows_by_frame else 0,
                    )
                    csv_target_ids = [fallback_selected_id for _ in range(seq_len)]
                    for frame_i, preset_i in csv_rows_by_frame.items():
                        fi = int(frame_i)
                        pid = int(preset_i)
                        if 0 <= fi < seq_len and pid in CT_PRESET_BY_ID:
                            csv_target_ids[fi] = pid
                    if 0 <= preview_frame_idx < len(csv_target_ids):
                        csv_detected_preset_for_frame = int(
                            csv_target_ids[preview_frame_idx]
                        )
                    csv_blend_weights_by_frame, _csv_osc_flags = _build_csv_blend_weights_by_frame(
                        csv_target_ids
                    )
                    if 0 <= preview_frame_idx < len(csv_blend_weights_by_frame):
                        csv_blend_weights_for_frame = dict(
                            csv_blend_weights_by_frame[preview_frame_idx]
                        )
                    if not csv_blend_weights_for_frame:
                        csv_blend_weights_for_frame = {fallback_selected_id: 1.0}

                # Preview uses the same clean binary CT mask strategy as batch.
                if has_replace_mask:
                    mask_bin = (ct_mask_source > 0.5).float()
                else:
                    if params.get("mask_binarize_threshold", -1.0) >= 0.0:
                        mask_bin = (
                            ct_mask_source > params["mask_binarize_threshold"]
                        ).float()
                    else:
                        mask_bin = (ct_mask_source > 0.5).float()

                inpainted_3 = inpainted[0].cpu() if inpainted.dim() == 4 else inpainted.cpu()
                original_left_3 = (
                    original_left[0].cpu() if original_left.dim() == 4 else original_left.cpu()
                )
                warped_3 = (
                    right_eye_original[0].cpu()
                    if right_eye_original.dim() == 4
                    else right_eye_original.cpu()
                )
                mask_bin_1hw = mask_bin[0].cpu() if mask_bin.dim() == 4 else mask_bin.cpu()

                if ct_auto_mode == CT_AUTO_MODE_ON:
                    candidate_ids = list(CT_PRESET_AUTO_EVAL_ORDER)
                    with ThreadPoolExecutor(max_workers=CT_AUTO_EVAL_MAX_WORKERS) as ct_eval_executor:
                        best_frame, best_preset_id = _select_best_auto_ct_preset_frame(
                            inpainted_3=inpainted_3,
                            original_left_3=original_left_3,
                            warped_3=warped_3,
                            mask_bin_1hw=mask_bin_1hw,
                            settings=params,
                            fallback_preset_id=int(selected_preset["id"]),
                            candidate_preset_ids=candidate_ids,
                            executor=ct_eval_executor,
                        )
                    out_3 = best_frame.to(device)
                    self.auto_ct_best_var.set(f"Auto CT best: #{best_preset_id} (preview)")
                else:
                    if ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
                        show_blend_preview = bool(
                            params.get("show_blend_in_preview", True)
                        )
                        weights_sorted = sorted(
                            csv_blend_weights_for_frame.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )
                        if weights_sorted:
                            main_pid = int(weights_sorted[0][0])
                            main_pct = float(weights_sorted[0][1]) * 100.0
                        else:
                            main_pid = int(selected_preset["id"])
                            main_pct = 100.0
                        apply_weights = (
                            dict(csv_blend_weights_for_frame)
                            if show_blend_preview
                            else {int(csv_detected_preset_for_frame): 1.0}
                        )
                        stats_valid_cache: Dict[str, torch.Tensor] = {}
                        warped_ref_cache: Dict[str, torch.Tensor] = {}
                        out_mix: Optional[torch.Tensor] = None
                        for pid_i, weight_i in sorted(
                            apply_weights.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        ):
                            pid = int(pid_i)
                            w = float(max(0.0, min(1.0, float(weight_i))))
                            if w <= 0.0:
                                continue
                            preset_i = CT_PRESET_BY_ID.get(pid, selected_preset)
                            adjusted_3 = _apply_ct_preset_frame(
                                preset=preset_i,
                                inpainted_3=inpainted_3,
                                original_left_3=original_left_3,
                                warped_3=warped_3,
                                mask_bin_1hw=mask_bin_1hw,
                                settings=params,
                                stats_valid_cache=stats_valid_cache,
                                warped_ref_cache=warped_ref_cache,
                            )
                            if out_mix is None:
                                out_mix = adjusted_3 * w
                            else:
                                out_mix = out_mix + (adjusted_3 * w)
                        if out_mix is None:
                            fallback_selected_id = int(selected_preset["id"])
                            stats_valid_cache = {}
                            warped_ref_cache = {}
                            out_mix = _apply_ct_preset_frame(
                                preset=CT_PRESET_BY_ID[fallback_selected_id],
                                inpainted_3=inpainted_3,
                                original_left_3=original_left_3,
                                warped_3=warped_3,
                                mask_bin_1hw=mask_bin_1hw,
                                settings=params,
                                stats_valid_cache=stats_valid_cache,
                                warped_ref_cache=warped_ref_cache,
                            )
                        out_3 = torch.clamp(out_mix, 0.0, 1.0).to(device)
                        blend_txt = ", ".join(
                            [
                                f"#{int(pid)}:{(100.0 * float(w)):.0f}%"
                                for pid, w in weights_sorted[:4]
                            ]
                        ) or f"#{main_pid}:100%"
                        if show_blend_preview:
                            self.auto_ct_best_var.set(
                                f"Auto CT CSV preview BLEND | main=#{main_pid} ({main_pct:.0f}%) | detected=#{int(csv_detected_preset_for_frame)} | {blend_txt}"
                            )
                        else:
                            self.auto_ct_best_var.set(
                                f"Auto CT CSV preview DETECTED | #{int(csv_detected_preset_for_frame)} (blend off) | planned {blend_txt}"
                            )
                    else:
                        selected_id_for_frame = int(selected_preset["id"])
                        selected_preset_for_frame = selected_preset
                        stats_valid_cache = {}
                        warped_ref_cache = {}
                        out_3 = _apply_ct_preset_frame(
                            preset=selected_preset_for_frame,
                            inpainted_3=inpainted_3,
                            original_left_3=original_left_3,
                            warped_3=warped_3,
                            mask_bin_1hw=mask_bin_1hw,
                            settings=params,
                            stats_valid_cache=stats_valid_cache,
                            warped_ref_cache=warped_ref_cache,
                        ).to(device)
                        self.auto_ct_best_var.set(
                            f"Auto CT best: #{selected_id_for_frame} (manual)"
                        )

                if inpainted.dim() == 4:
                    inpainted = out_3.unsqueeze(0)
                else:
                    inpainted = out_3

            blended_frame = (
                right_eye_original * (1 - processed_mask) + inpainted * processed_mask
            )

            # --- NEW: Apply borders from sidecar ---
            current_source_metadata = self.previewer.video_list[
                self.previewer.current_video_index
            ]
            clip_sidecar = current_source_metadata.get("sidecar", {})
            left_border = clip_sidecar.get("left_border", 0.0)
            right_border = clip_sidecar.get("right_border", 0.0)
            logger.debug(f"Preview Borders: left={left_border}%, right={right_border}%")
            if clip_sidecar:
                self._update_border_info(left_border, right_border)
            else:
                self._clear_border_info()

            if self.add_borders_var.get() and (left_border > 0 or right_border > 0):
                logger.debug(
                    f"Preview: Before border - original_left shape={original_left.shape}, blended_frame shape={blended_frame.shape}"
                )
                original_left, blended_frame = apply_borders_to_frames(
                    left_border, right_border, original_left, blended_frame
                )
                logger.debug(
                    f"Preview: After border - original_left shape={original_left.shape}, blended_frame shape={blended_frame.shape}"
                )
            # --- END NEW ---

            # 4. Select the final frame to display based on the dropdown
            preview_source = self.preview_source_var.get()
            logger.debug(f"Preview source selected: '{preview_source}'")
            final_frame_4d = None  # Initialize to None

            if preview_source == "Blended Image":
                logger.debug("  -> Displaying Blended Image.")
                final_frame_4d = blended_frame
            elif preview_source == "Inpainted Right Eye":  # <--- ADDED INPAINTED
                logger.debug("  -> Displaying Inpainted Right Eye.")
                final_frame_4d = inpainted
            elif preview_source == "Original (Left Eye)":
                logger.debug("  -> Displaying Original (Left Eye).")
                # --- FIX: Handle missing original_tensor for quad input ---
                if original_left is not None:
                    final_frame_4d = original_left
                else:
                    # This case should not be reachable if logic is correct, but as a fallback:
                    logger.warning(
                        "Preview: 'Original (Left Eye)' selected, but no source is available."
                    )
                    final_frame_4d = torch.zeros_like(
                        blended_frame
                    )  # Show a black screen
                # --- END FIX ---
            elif preview_source == "Warped (Right BG)":
                logger.debug("  -> Displaying Warped (Right BG).")
                final_frame_4d = right_eye_original
            elif preview_source == "Processed Mask":
                logger.debug("  -> Displaying Processed Mask.")
                final_frame_4d = processed_mask.repeat(
                    1, 3, 1, 1
                )  # Convert grayscale mask to 3-channel for display
            elif preview_source == "Anaglyph 3D":
                logger.debug(" -> Displaying Anaglyph 3D.")
                left_np = (
                    original_left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                right_np = (
                    blended_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                left_gray_np = cv2.cvtColor(
                    left_np, cv2.COLOR_RGB2GRAY
                )  # Use standard for old red/cyan
                anaglyph_np = right_np.copy()
                anaglyph_np[:, :, 0] = (
                    left_gray_np  # Red channel from grayscale left eye
                )
                final_frame_4d = (
                    torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0
                ).unsqueeze(0)
            elif preview_source == "Dubois Anaglyph":
                logger.debug(" -> Displaying Dubois Anaglyph.")
                left_np = (
                    original_left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                right_np = (
                    blended_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                anaglyph_np = apply_dubois_anaglyph(
                    left_np, right_np
                )  # Use imported utility
                final_frame_4d = (
                    torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0
                ).unsqueeze(0)
            elif preview_source == "Optimized Anaglyph":
                logger.debug(" -> Displaying Optimized Anaglyph.")
                left_np = (
                    original_left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                right_np = (
                    blended_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                anaglyph_np = apply_optimized_anaglyph(
                    left_np, right_np
                )  # Use imported utility
                final_frame_4d = (
                    torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0
                ).unsqueeze(0)
            elif preview_source == "Wigglegram":
                logger.debug(" -> Starting Wigglegram animation.")
                self.previewer._start_wigglegram_animation(original_left, blended_frame)
                return None  # Wigglegram handles its own display
            elif preview_source == "Depth Map" and depth_map_vis is not None:
                logger.debug("  -> Displaying Depth Map.")
                final_frame_4d = depth_map_vis.to(device)
            else:
                logger.debug(
                    f"  -> Fallback: Displaying Blended Image for unknown source '{preview_source}'."
                )
                final_frame_4d = blended_frame

            # Fallback in case final_frame wasn't set
            if final_frame_4d is None:
                final_frame_4d = blended_frame

            # Store for saving SBS
            self.preview_original_left_tensor = original_left.squeeze(0).cpu()
            self.preview_blended_right_tensor = blended_frame.squeeze(0).cpu()

            # 5. Convert to PIL Image for returning
            final_frame_cpu = final_frame_4d.cpu()
            pil_img = Image.fromarray(
                (final_frame_cpu.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8
                )
            )
            return pil_img
        except Exception as e:
            logger.error(f"Error in preview processing callback: {e}", exc_info=True)
            return None

    def save_config(self):
        """Gathers current settings and saves them to the config file."""
        config = self.get_current_settings()
        if config:
            # Add window geometry and other non-processing settings to the config dictionary
            config["window_x"] = self.winfo_x()
            config["window_y"] = self.winfo_y()
            config["window_width"] = self.winfo_width()
            config["window_height"] = self.winfo_height()
            config["debug_logging_enabled"] = self.debug_logging_var.get()
            config["dark_mode_enabled"] = self.dark_mode_var.get()
            # The following settings are already gathered by get_current_settings(),
            # so these stray lines are removed.

            try:
                with open("config_merging.mergecfg", "w") as f:
                    json.dump(config, f, indent=4)
                logger.info("Merging GUI configuration saved.")
            except Exception as e:
                logger.error(f"Failed to save merging GUI config: {e}")

    def _load_config(self):
        """Loads configuration from a JSON file."""
        try:
            with open("config_merging.mergecfg", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Failed to load merging GUI config: {e}")
            return {}

    def load_settings_dialog(self):
        """Loads settings from a user-selected JSON file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".mergecfg",
            filetypes=[("Merge Config Files", "*.mergecfg"), ("All files", "*.*")],
            title="Load Settings from File",
        )
        if not filepath:
            return
        try:
            with open(filepath, "r") as f:
                settings_to_load = json.load(f)

            self._apply_settings(settings_to_load)
            self._apply_theme()
            logger.info(f"Settings loaded from {filepath}")
        except Exception as e:
            messagebox.showerror(
                "Load Error", f"Failed to load settings from {filepath}:\n{e}"
            )

    def save_settings_dialog(self):
        """Saves current GUI settings to a user-selected JSON file."""
        config_to_save = self.get_current_settings()
        if not config_to_save:
            return  # get_current_settings failed validation

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mergecfg",
            filetypes=[("Merge Config Files", "*.mergecfg"), ("All files", "*.*")],
            title="Save Settings to File",
        )
        if not filepath:
            return
        try:
            with open(filepath, "w") as f:
                json.dump(config_to_save, f, indent=4)
            logger.info(f"Settings saved to {filepath}")
        except Exception as e:
            messagebox.showerror(
                "Save Error", f"Failed to save settings to {filepath}:\n{e}"
            )

    def _save_preview_sbs_frame(self):
        """Saves the current preview as a full side-by-side image."""
        if (
            self.preview_original_left_tensor is None
            or self.preview_blended_right_tensor is None
        ):
            messagebox.showwarning(
                "No Preview Data",
                "There is no preview data to save. Please load and preview a video first.",
            )
            return

        try:
            # Convert tensors to PIL Images
            left_np = (
                self.preview_original_left_tensor.permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            right_np = (
                self.preview_blended_right_tensor.permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)

            left_pil = Image.fromarray(left_np)
            right_pil = Image.fromarray(right_np)

            # Check if dimensions match
            if left_pil.size != right_pil.size:
                messagebox.showerror(
                    "Dimension Mismatch",
                    "The left and right eye images have different dimensions. Cannot create SBS image.",
                )
                return

            # Create SBS image
            width, height = left_pil.size
            sbs_image = Image.new("RGB", (width * 2, height))
            sbs_image.paste(left_pil, (0, 0))
            sbs_image.paste(right_pil, (width, 0))

            # Suggest a default filename
            default_filename = "preview_sbs_frame.png"
            if self.previewer.current_video_index != -1:
                source_paths = self.previewer.video_list[
                    self.previewer.current_video_index
                ]
                base_name = os.path.splitext(
                    os.path.basename(next(iter(source_paths.values())))
                )[0]
                frame_num = int(self.previewer.frame_scrubber_var.get())
                default_filename = f"{base_name}_frame_{frame_num:05d}_SBS.png"

            filepath = filedialog.asksaveasfilename(
                title="Save SBS Preview Frame As...",
                initialfile=default_filename,
                defaultextension=".png",
                filetypes=[
                    ("PNG Image", "*.png"),
                    ("JPEG Image", "*.jpg"),
                    ("All Files", "*.*"),
                ],
            )

            if filepath:
                sbs_image.save(filepath)
                logger.info(f"SBS preview frame saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save SBS preview frame: {e}", exc_info=True)
            messagebox.showerror(
                "Save Error",
                f"An error occurred while creating or saving the SBS image:\n{e}",
            )

    def exit_application(self):
        """Handles application exit gracefully."""
        if self.is_processing:
            if messagebox.askyesno(
                "Confirm Exit",
                "Processing is in progress. Are you sure you want to stop and exit?",
            ):
                self.stop_processing()
                self.previewer.cleanup()
                self.save_config()
                self.destroy()
        else:
            self.save_config()
            self.previewer.cleanup()
            self.destroy()


if __name__ == "__main__":
    debug_enabled = _env_flag("MERGE_DEBUG", True)
    # By default, faulthandler follows debug mode. You can still override explicitly:
    # MERGE_FAULTHANDLER=1 to force ON, MERGE_FAULTHANDLER=0 to force OFF.
    faulthandler_enabled = _env_flag("MERGE_FAULTHANDLER", debug_enabled)
    # Basic logging setup
    logging.basicConfig(
        level=logging.DEBUG if debug_enabled else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if faulthandler_enabled:
        _enable_debug_faulthandler()
        logger.info("Faulthandler enabled (set MERGE_FAULTHANDLER=0 to disable).")
    if debug_enabled:
        logger.info("Debug mode enabled (MERGE_DEBUG=1).")
    app = MergingGUI()
    app.mainloop()
