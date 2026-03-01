#!/usr/bin/env python3
"""
Headless batch runner for StereoCrafter merging step (no Tk GUI).

- Streams frame chunks from decord VideoReader (no full-video RAM load).
- Pipes frames to ffmpeg via start_ffmpeg_pipe_process (same as merging_gui).
- Skip-if-exists (default ON)
- Retry (default 1 => at most 2 attempts total)
- Optional move inputs to finished/failed (default OFF)
- Cleans up partial outputs on failure

Designed to be driven by an outer .sh that sets directories and adds extra crash-handling.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import logging
import os
import re
import shutil
import threading
import time
import faulthandler
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu  # type: ignore

# These utilities are already used by merging_gui.py
from dependency.stereocrafter_util import (  # type: ignore
    apply_borders_to_frames,
    draw_progress_bar,
    get_video_stream_info,
    read_clip_sidecar,
    start_ffmpeg_pipe_process,
    apply_color_transfer,
)

LOG = logging.getLogger("merge_runner")
_FAULTHANDLER_LOG = None


def _enable_debug_faulthandler() -> None:
    """Enable crash stack dumps for nogui runs when debug mode is enabled."""
    global _FAULTHANDLER_LOG
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join(
            "logs", "merging_nogui_batch_faulthandler.log"
        )
        _FAULTHANDLER_LOG = open(log_path, "a", buffering=1)
        _FAULTHANDLER_LOG.write(
            f"\n=== debug session {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===\n"
        )
        _FAULTHANDLER_LOG.flush()
        faulthandler.enable(file=_FAULTHANDLER_LOG, all_threads=True)
        LOG.info(f"Debug faulthandler active: {log_path}")
    except Exception as e:
        LOG.warning(f"Failed to enable debug faulthandler: {e}")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _default_stop_marker_path(output_folder: str) -> str:
    return os.path.join(os.path.abspath(output_folder), ".stop_after_current")


def _stop_marker_exists(path: str) -> bool:
    if not path:
        return False
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _clear_stop_marker(path: str) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# =========================
# Editable defaults (hardcoded)
# =========================

DEFAULTS: Dict[str, object] = {
    # Performance
    "device": "cuda",                 # "cuda" or "cpu" (for the torch ops only)
    "use_gpu_mask_ops": True,         # applies to dilate/blur/shadow (cuda if available)

    # Batch/stream
    "batch_chunk_size": 20,           # decord batch size

    # Output
    "output_format": "Full SBS (Left-Right)",  # see OUTPUT_FORMAT_CHOICES
    "pad_to_16_9": False,
    "add_borders": False,               # sidecar-based borders (no-op if sidecar missing / 0%)
    "skip_existing": True,


    # Color transfer
    "enable_color_transfer": True,
    "ct_preset": "1) safe sr=ring ts=inpainted ref=warped",
    "ct_auto_mode": "On",
    "ct_csv_blend_path": "",
    "ct_strength": 1.0,
    "ct_black_thresh": 0.0,
    "ct_min_valid_ratio": 0.0,
    "ct_min_valid": 0,
    "ct_clamp_L_min": 0.1,
    "ct_clamp_L_max": 2,
    "ct_clamp_ab_min": 0.1,
    "ct_clamp_ab_max": 3,
    "ct_exclude_black_in_target": True,
    "ct_ring_width": 20,
    "mask_binarize_threshold": -0.01,   # used for stats-mask and optional binarize step if you add it later

    # Replace mask
    "use_replace_mask": True,

    # Mask post-processing
    "mask_dilate_kernel_size": 2,
    "mask_blur_kernel_size": 4,

    # Shadow (soft edge) post-processing
    "shadow_length_px": 30,
    "shadow_width_adaptive": True,
    "shadow_curve": 0.0,
    "shadow_motion_gain": 1.0,

    # Robustness / workflow
    "retries": 1,
    "move_finished": False,
    "move_failed": False,
    "cleanup_partial_outputs": True,
}

OUTPUT_FORMAT_CHOICES = [
    "Full SBS (Left-Right)",
    "Full SBS Cross-eye (Right-Left)",
    "Half SBS (Left-Right)",
    "Double SBS",
    "Right-Eye Only",
    "Anaglyph (Red/Cyan)",
    "Anaglyph Half-Color",
]



# =========================
# Color transfer (SAFE) helpers (copied/adapted from merging_gui.py)
# =========================

def _telea_inpaint_rgb_uint8(frame_rgb_u8: np.ndarray, mask_u8: np.ndarray, radius: int = 3) -> np.ndarray:
    """OpenCV inpaint helper (TELEA). frame_rgb_u8: HxWx3 RGB uint8, mask_u8: HxW uint8 0/255."""
    try:
        out_bgr = cv2.inpaint(cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2BGR), mask_u8, radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        LOG.error(f"Telea inpaint failed: {e!r}", exc_info=True)
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
        LOG.error(f"Directional fill failed: {e!r}", exc_info=True)
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
    # (with tolerance) with directional fill.
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
        LOG.error(f"Hybrid warped fill failed: {e!r}", exc_info=True)
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
    return CT_AUTO_MODE_ON


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
            if frame_idx < 0 or preset_id not in CT_PRESET_BY_ID:
                continue
            status = str(row.get("status", "") or "").strip().lower()
            if status and status not in {"ok", "done", "complete"}:
                continue

            keys: List[str] = []
            for col in ("video", "inpainted", "filename", "clip", "core_with_width", "core_name"):
                v = str(row.get(col, "") or "").strip()
                if v:
                    keys.append(v)

            for key in keys:
                norm = _normalize_csv_blend_lookup_key(key)
                if norm:
                    preset_by_key.setdefault(norm, {})[frame_idx] = preset_id
                base = os.path.basename(key)
                stem, _ext = os.path.splitext(base)
                for alias in (base, stem):
                    anorm = _normalize_csv_blend_lookup_key(alias)
                    if anorm:
                        preset_by_key.setdefault(anorm, {})[frame_idx] = preset_id
                if base.endswith("_inpainted_right_eye.mp4") or base.endswith("_inpainted_sbs.mp4"):
                    core_with_width, _is_sbs = parse_inpainted_name(base)
                    core_name, _w = parse_core_and_width(core_with_width)
                    for alias in (core_with_width, core_name):
                        anorm = _normalize_csv_blend_lookup_key(alias)
                        if anorm:
                            preset_by_key.setdefault(anorm, {})[frame_idx] = preset_id
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
        if base_name.endswith("_inpainted_right_eye.mp4") or base_name.endswith("_inpainted_sbs.mp4"):
            parsed_core_with_width, _ = parse_inpainted_name(base_name)
            parsed_core_name, _w = parse_core_and_width(parsed_core_with_width)
            candidates.extend([parsed_core_with_width, parsed_core_name])
    if core_with_width:
        candidates.append(core_with_width)
    if core_name:
        candidates.append(core_name)

    best_map: Dict[int, int] = {}
    best_key = ""
    for key in candidates:
        norm = _normalize_csv_blend_lookup_key(key)
        if not norm:
            continue
        rows = preset_by_key.get(norm)
        if rows and len(rows) > len(best_map):
            best_map = rows
            best_key = norm
    return dict(best_map), best_key


def _prepare_ct_csv_blend_map_once(
    settings: Dict[str, object]
) -> Dict[str, Dict[int, int]]:
    if not bool(settings.get("enable_color_transfer", False)):
        return {}
    if _resolve_ct_auto_mode_from_settings(settings) != CT_AUTO_MODE_CSV_BLEND:
        return {}

    csv_blend_path = str(settings.get("ct_csv_blend_path", "") or "").strip()
    if not csv_blend_path:
        raise RuntimeError("CT mode is 'CSV Blend' but no CSV path was provided.")
    if not os.path.exists(csv_blend_path):
        raise RuntimeError(f"CSV Blend Auto CT map not found: {csv_blend_path}")

    ct_csv_blend_preset_map = _load_csv_blend_preset_map(csv_blend_path)
    if not ct_csv_blend_preset_map:
        LOG.warning(
            f"CSV Blend Auto CT map loaded but lookup map is empty: {csv_blend_path}"
        )
    else:
        LOG.info(
            f"CSV Blend Auto CT map preloaded once: keys={len(ct_csv_blend_preset_map)}"
        )
    return ct_csv_blend_preset_map


def _compute_preset_oscillator_flags(seq: List[int], min_len: int = 4) -> List[bool]:
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

        weights = {int(k): float(v) for k, v in weights.items() if float(v) > float(prune_eps)}
        if not weights:
            weights = {pid: 1.0}

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
    grouped: Dict[str, List[int]] = {"ring": [], "nonmask": [], "global": []}
    serial_ids: List[int] = []
    for pid in preset_order:
        preset = CT_PRESET_BY_ID[int(pid)]
        mode = str(preset.get("mode", "safe"))
        if mode == "legacy":
            serial_ids.append(int(pid))
            continue
        sr = str(preset.get("stats_region", "ring"))
        if sr in grouped:
            grouped[sr].append(int(pid))
        else:
            serial_ids.append(int(pid))

    parallel_groups: List[List[int]] = []
    for key in ("ring", "nonmask", "global"):
        if grouped[key]:
            parallel_groups.append(grouped[key])
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
    best_preset_id = (
        int(preset_ids[0]) if preset_ids else int(CT_PRESET_AUTO_EVAL_ORDER[0])
    )

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

    order_index = {int(pid): i for i, pid in enumerate(CT_PRESET_AUTO_EVAL_ORDER)}
    best_score = -1.0
    best_frame = inpainted_3
    best_preset_id = int(fallback_preset_id)

    def _consider(candidate: Tuple[float, torch.Tensor, int]) -> None:
        nonlocal best_score, best_frame, best_preset_id
        score, frame, pid = candidate
        if _is_better_auto_ct_candidate(
            score, int(pid), best_score, best_preset_id, order_index
        ):
            best_score = score
            best_frame = frame
            best_preset_id = int(pid)

    if executor is not None and CT_AUTO_EVAL_PARALLEL_GROUPS:
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
            for group in CT_AUTO_EVAL_PARALLEL_GROUPS
        ]
        for fut in futures:
            _consider(fut.result())
    else:
        for group in CT_AUTO_EVAL_PARALLEL_GROUPS:
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

    if CT_AUTO_EVAL_SERIAL_IDS:
        _consider(
            _eval_auto_ct_subset(
                preset_ids=CT_AUTO_EVAL_SERIAL_IDS,
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


def _masked_delta_e_score(pred_rgb01: np.ndarray, ref_lab32: np.ndarray, mask_bool: np.ndarray) -> float:
    n = int(mask_bool.sum())
    if n <= 0:
        return 0.0
    pred_lab = _rgb01_to_lab32(pred_rgb01)
    diff = pred_lab - ref_lab32
    de = np.sqrt(np.sum(diff * diff, axis=2))
    de_mean = float(de[mask_bool].mean())
    return 1.0 / (1.0 + max(0.0, de_mean))


def _build_ring_shift_reference(frame_rgb_u8: np.ndarray, mask_u8: np.ndarray, border_tol_px: int = 2) -> np.ndarray:
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
        # Optimization: for non-global stats, filled mask area does not affect stats.
        if stats_region != "global":
            ref = warped_3.cpu()
        else:
            fill_mode = str(preset.get("warped_fill_mode", "directional"))
            if fill_mode not in warped_ref_cache:
                wf = warped_3.cpu()
                wf_u8 = (torch.clamp(wf, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                mm = (mask_bin_1hw.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                ref_u8 = _build_warped_filled_reference(
                    wf_u8,
                    mm,
                    fill_mode=fill_mode,
                    border_tol_px=2,
                    telea_radius=3,
                )
                warped_ref_cache[fill_mode] = torch.from_numpy(ref_u8).permute(2, 0, 1).float() / 255.0
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


def apply_shadow_blur(
    mask: torch.Tensor,
    base_length_px: int,
    curve: float,
    motion_gain: float,
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
    motion_deadzone_px = 3.0  # Ignore micro-jitter up to this displacement.
    motion_ref_px = 6.0  # Additional px/frame (beyond deadzone) for full motion gain.

    alpha_up = 0.45
    alpha_down = 0.20
    max_delta_up = max(1.0, 0.35 * base_len)
    max_delta_down = max(1.0, 0.20 * base_len)
    max_len_cap = int(max(100, 4 * base_len))

    mask_cpu = mask.detach().to(device="cpu", dtype=torch.float32).numpy()  # T,1,H,W
    t_count, _c, height, width = mask_cpu.shape
    right_touch_start = width - border_tol

    prev_components: List[Dict[str, Any]] = []
    if state is not None:
        prev_components = list(state.get("prev_components", []) or [])

    # Keep shadow processing fully on NumPy buffers and convert back to torch once.
    out_np = np.empty_like(mask_cpu, dtype=np.float32)

    for t in range(t_count):
        frame = np.asarray(mask_cpu[t, 0], dtype=np.float32)
        frame_bin = (frame > component_thresh).astype(np.uint8)
        canvas = frame.copy()

        if frame_bin.any():
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                frame_bin, connectivity=8
            )

            curr_components: List[Dict[str, Any]] = []
            comp_len_by_label: Dict[int, float] = {}

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

                matches: List[Tuple[float, Dict[str, Any]]] = []
                for prev in prev_components:
                    inter = _bbox_intersection_area(bbox, prev["bbox"])
                    if inter > 0:
                        matches.append((float(inter), prev))

                if not matches and prev_components:
                    # Fallback nearest match for mild displacements without overlap.
                    best_prev = None
                    best_d = 1e9
                    max_dist = max(8.0, 0.5 * np.sqrt(float(area)))
                    for prev in prev_components:
                        dx = float(cx - prev["centroid"][0])
                        dy = float(cy - prev["centroid"][1])
                        d = float(np.hypot(dx, dy))
                        if d < best_d and d <= max_dist:
                            best_d = d
                            best_prev = prev
                    if best_prev is not None:
                        matches.append((1.0, best_prev))

                if matches:
                    wsum = float(sum(mw for mw, _ in matches))
                    prev_len = float(
                        sum(mw * float(m["len_smooth"]) for mw, m in matches)
                        / max(1e-6, wsum)
                    )
                    motion_px = float(
                        sum(
                            mw
                            * float(
                                np.hypot(
                                    cx - float(m["centroid"][0]),
                                    cy - float(m["centroid"][1]),
                                )
                            )
                            for mw, m in matches
                        )
                        / max(1e-6, wsum)
                    )
                else:
                    prev_len = float(base_len)
                    motion_px = 0.0

                motion_eff_px = max(0.0, motion_px - motion_deadzone_px)
                motion_mult = 1.0 + motion_gain * min(1.0, motion_eff_px / motion_ref_px)
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
                        "label": int(lab),
                        "bbox": bbox,
                        "centroid": (cx, cy),
                        "area": area,
                        "len_smooth": len_smooth,
                    }
                )
                comp_len_by_label[int(lab)] = len_smooth

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

            prev_components = curr_components

        out_np[t, 0] = canvas

    if state is not None:
        state["prev_components"] = prev_components

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
        blurred = F.conv2d(blurred, kernel_1d.permute(0, 1, 3, 2), padding=(k // 2, 0), groups=mask.shape[1])
        return torch.clamp(blurred, 0.0, 1.0)
    out = []
    for t in range(mask.shape[0]):
        frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        bl = cv2.GaussianBlur(frame_np, (k, k), 0)
        out.append(torch.from_numpy(bl).float().div(255.0).unsqueeze(0))
    return torch.stack(out).to(mask.device)
    
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
    Reinhard-like color transfer in LAB (float32, clamped).
    Expects [C,H,W] float [0,1] tensors.
    """
    try:
        src_t = source_frame.detach().cpu().float()
        tgt_t = target_frame.detach().cpu().float()

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
        min_valid_eff = max(int(min_valid), int(float(min_valid_ratio) * Hs * Ws))
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
        LOG.error(f"SAFE color transfer failed: {e!r}", exc_info=True)
        return target_frame

# =========================
# Helpers
# =========================

def setup_logging(verbosity: int) -> None:
    level = logging.DEBUG
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _read_ffmpeg_output(pipe, level=logging.DEBUG) -> None:
    try:
        for line in iter(pipe.readline, b""):
            if not line:
                break
            try:
                msg = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                msg = repr(line)
            LOG.log(level, f"[ffmpeg] {msg}")
    except Exception as e:
        LOG.debug(f"ffmpeg pipe reader ended: {e!r}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def safe_makedirs(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def move_file(src: Optional[str], dst_dir: str) -> None:
    if not src:
        return
    if not os.path.exists(src):
        return
    safe_makedirs(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    # If already exists at destination, don't overwrite; keep source in place.
    if os.path.exists(dst):
        LOG.warning(f"Destination already exists, not moving: {dst}")
        return
    shutil.move(src, dst)


def delete_if_exists(p: Optional[str]) -> None:
    if not p:
        return
    try:
        if os.path.exists(p):
            os.remove(p)
    except Exception as e:
        LOG.warning(f"Failed to remove '{p}': {e}")


def partial_output_path(output_path: str) -> str:
    """
    Temp output path that preserves a valid container extension for ffmpeg.
    Example: clip.mp4 -> clip.part.mp4
    """
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".mp4"
    return f"{root}.part{ext}"


def legacy_partial_output_path(output_path: str) -> str:
    """Previous temp naming kept for backward-compatible cleanup."""
    return f"{output_path}.part"


def cleanup_partial_output_files(output_path: Optional[str]) -> None:
    if not output_path:
        return
    new_partial = partial_output_path(output_path)
    old_partial = legacy_partial_output_path(output_path)
    delete_if_exists(new_partial)
    if old_partial != new_partial:
        delete_if_exists(old_partial)


def find_video_by_core_name(folder: str, core_name: str) -> Optional[str]:
    """
    Finds 'core_name.*' in folder, preferring common video extensions.
    """
    if not folder:
        return None
    patterns = [
        os.path.join(folder, f"{core_name}.mp4"),
        os.path.join(folder, f"{core_name}.mkv"),
        os.path.join(folder, f"{core_name}.mov"),
        os.path.join(folder, f"{core_name}.webm"),
        os.path.join(folder, f"{core_name}.*"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[0]
    return None


def parse_inpainted_name(filename: str) -> Tuple[str, bool]:
    """
    Extract core_name from inpainted filename.
    Handles:
      - <core>_inpainted_right_eye.mp4
      - <core>_inpainted_sbs.mp4
    Returns (core_name, is_sbs_input)
    """
    base = os.path.basename(filename)
    is_sbs_input = base.endswith("_inpainted_sbs.mp4")
    if is_sbs_input:
        core = base.replace("_inpainted_sbs.mp4", "")
    else:
        core = base.replace("_inpainted_right_eye.mp4", "")
    return core, is_sbs_input


def parse_core_and_width(core_with_width: str) -> Tuple[str, Optional[int]]:
    """
    If core has suffix _<width> (e.g. source-Scene-0003_1920), returns (core_name, width).
    Otherwise width None.
    """
    m = re.match(r"^(.*)_(\d+)$", core_with_width)
    if not m:
        return core_with_width, None
    return m.group(1), int(m.group(2))


def find_replace_mask_for_splatted(splatted_path: str, replace_mask_folder: str) -> Optional[str]:
    """
    Tries to locate a binary replace-mask video matching splatted path.
    Common patterns:
      - source-Scene-xxxx_1920_splatted2.mp4  -> source-Scene-xxxx_1920_splatted2_replace_mask.mkv/mp4
      - source-Scene-xxxx_1920_splatted4.mp4  -> source-Scene-xxxx_1920_splatted4_replace_mask.mkv/mp4
    """
    folder = replace_mask_folder.strip() or os.path.dirname(splatted_path)
    base = os.path.basename(splatted_path)
    stem, _ext = os.path.splitext(base)

    candidates = [
        f"{stem}_replace_mask.mkv",
        f"{stem}_replace_mask.mp4",
        f"{stem}_replace_mask.webm",
        f"{stem}_replace_mask.avi",
        f"{stem}_replace_mask.*",
    ]
    for c in candidates:
        p = os.path.join(folder, c)
        hits = glob.glob(p)
        if hits:
            hits.sort()
            return hits[0]
    return None


@dataclass
class JobPaths:
    inpainted_video_path: str
    splatted_video_path: str
    original_video_path: Optional[str]
    replace_mask_path: Optional[str]
    output_path: str
    inpainted_base: str
    core_name: str
    is_sbs_input: bool


def build_output_path(
    output_folder: str,
    core_name: str,
    hires_w: int,
    hires_h: int,
    output_format: str,
) -> Tuple[str, int, int]:
    """
    Mirrors the naming logic in merging_gui run_batch_process.
    Returns (output_path, output_width, output_height).
    """
    perceived_width_for_filename = hires_w
    output_height = hires_h

    if output_format == "Full SBS Cross-eye (Right-Left)":
        output_width = hires_w * 2
        suffix = "_merged_full_sbsx.mp4"
    elif output_format == "Full SBS (Left-Right)":
        output_width = hires_w * 2
        suffix = "_merged_full_sbs.mp4"
    elif output_format == "Double SBS":
        output_width = hires_w * 2
        output_height = hires_h * 2
        suffix = "_merged_half_sbs.mp4"
        perceived_width_for_filename = hires_w * 2
    elif output_format == "Half SBS (Left-Right)":
        output_width = hires_w
        suffix = "_merged_half_sbs.mp4"
    elif output_format in ["Anaglyph (Red/Cyan)", "Anaglyph Half-Color"]:
        output_width = hires_w
        suffix = "_merged_anaglyph.mp4"
    else:  # Right-Eye Only
        output_width = hires_w
        suffix = "_merged_right_eye.mp4"

    output_filename = f"{core_name}_{perceived_width_for_filename}{suffix}"
    return os.path.join(output_folder, output_filename), output_width, output_height


def output_suffix_and_width_factor(output_format: str) -> Tuple[str, int]:
    """
    Returns (output_suffix, filename_width_factor).
    filename_width_factor is applied to the perceived width in the output filename.
    """
    if output_format == "Full SBS Cross-eye (Right-Left)":
        return "_merged_full_sbsx.mp4", 1
    if output_format == "Full SBS (Left-Right)":
        return "_merged_full_sbs.mp4", 1
    if output_format == "Double SBS":
        return "_merged_half_sbs.mp4", 2
    if output_format == "Half SBS (Left-Right)":
        return "_merged_half_sbs.mp4", 1
    if output_format in ["Anaglyph (Red/Cyan)", "Anaglyph Half-Color"]:
        return "_merged_anaglyph.mp4", 1
    return "_merged_right_eye.mp4", 1


def assemble_output_chunk(
    output_format: str,
    hires_h: int,
    hires_w: int,
    original_left: torch.Tensor,
    blended_right_eye: torch.Tensor,
) -> torch.Tensor:
    """
    Mirrors the output assembly logic in merging_gui.
    Expects tensors [T,C,H,W] in 0..1 float.
    """
    if output_format == "Full SBS (Left-Right)":
        return torch.cat([original_left, blended_right_eye], dim=3)
    if output_format == "Full SBS Cross-eye (Right-Left)":
        return torch.cat([blended_right_eye, original_left], dim=3)
    if output_format == "Half SBS (Left-Right)":
        resized_left = F.interpolate(original_left, size=(hires_h, hires_w // 2), mode="bilinear", align_corners=False)
        resized_right = F.interpolate(blended_right_eye, size=(hires_h, hires_w // 2), mode="bilinear", align_corners=False)
        return torch.cat([resized_left, resized_right], dim=3)
    if output_format == "Double SBS":
        sbs_chunk = torch.cat([original_left, blended_right_eye], dim=3)
        return F.interpolate(sbs_chunk, size=(hires_h * 2, hires_w * 2), mode="bilinear", align_corners=False)
    if output_format == "Anaglyph (Red/Cyan)":
        return torch.cat([original_left[:, 0:1, :, :], blended_right_eye[:, 1:3, :, :]], dim=1)
    if output_format == "Anaglyph Half-Color":
        left_gray = (
            original_left[:, 0, :, :] * 0.299
            + original_left[:, 1, :, :] * 0.587
            + original_left[:, 2, :, :] * 0.114
        ).unsqueeze(1)
        return torch.cat([left_gray, blended_right_eye[:, 1:3, :, :]], dim=1)
    # Right eye only
    return blended_right_eye


def write_chunk_to_ffmpeg(ffmpeg_process, chunk: torch.Tensor) -> None:
    """
    Mirrors merging_gui: convert to uint16 RGB->BGR and write raw bytes.
    Expects chunk [T,C,H,W] float 0..1 on CPU.
    """
    cpu_chunk = chunk.cpu()
    for frame_tensor in cpu_chunk:
        frame_np = frame_tensor.permute(1, 2, 0).numpy()
        frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
        frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)
        ffmpeg_process.stdin.write(frame_bgr.tobytes())


def should_skip_output(
    output_path: str,
    skip_existing: bool,
    strict_validate: bool = False,
) -> bool:
    if not skip_existing:
        return False
    if not os.path.exists(output_path):
        return False
    if os.path.getsize(output_path) <= 0:
        return False

    # Keep skip checks fast by default; strict validation is optional.
    if strict_validate:
        stream_info = get_video_stream_info(output_path)
        if stream_info is None:
            LOG.warning(
                f"Existing output appears invalid/corrupted, will re-encode: {os.path.basename(output_path)}"
            )
            return False
    return True


def infer_output_path_from_inpainted(
    inpainted_video_path: str,
    output_folder: str,
    output_format: str,
) -> Optional[str]:
    """
    Best-effort fast output path inference from inpainted filename.
    Returns None when width token is not available in filename.
    """
    core_with_width, _ = parse_inpainted_name(os.path.basename(inpainted_video_path))
    core_name, width_from_name = parse_core_and_width(core_with_width)
    if width_from_name is None:
        return None
    suffix, width_factor = output_suffix_and_width_factor(output_format)
    perceived_width_for_filename = int(width_from_name) * int(width_factor)
    return os.path.join(output_folder, f"{core_name}_{perceived_width_for_filename}{suffix}")


def collect_jobs(
    inpainted_folder: str,
    splatted_folder: str,
    original_folder: str,
    output_folder: str,
    only: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Collect inpainted files and find matching splatted.
    Returns list of (inpainted_path, splatted_path).
    """
    patterns = [
        os.path.join(inpainted_folder, "*_inpainted_right_eye.mp4"),
        os.path.join(inpainted_folder, "*_inpainted_sbs.mp4"),
    ]
    inpainted_files: List[str] = []
    for pat in patterns:
        inpainted_files.extend(glob.glob(pat))
    inpainted_files = sorted(set(inpainted_files))

    if only:
        inpainted_files = [p for p in inpainted_files if os.path.basename(p) == only or os.path.basename(p).startswith(only)]

    pairs: List[Tuple[str, str]] = []
    for inpainted_path in inpainted_files:
        base = os.path.basename(inpainted_path)
        core_with_width, _is_sbs = parse_inpainted_name(base)
        # Try to locate splatted in splatted_folder: match by core_with_width prefix
        # Prefer splatted2, then splatted4
        candidates = [
            os.path.join(splatted_folder, f"{core_with_width}_splatted2.mp4"),
            os.path.join(splatted_folder, f"{core_with_width}_splatted4.mp4"),
            os.path.join(splatted_folder, f"{core_with_width}_splatted2.mkv"),
            os.path.join(splatted_folder, f"{core_with_width}_splatted4.mkv"),
            os.path.join(splatted_folder, f"{core_with_width}_splatted*.mp4"),
            os.path.join(splatted_folder, f"{core_with_width}_splatted*.mkv"),
        ]
        splatted_path = None
        for c in candidates:
            hits = glob.glob(c)
            if hits:
                hits.sort()
                splatted_path = hits[0]
                break
        if not splatted_path:
            LOG.warning(f"Missing splatted for {base}: looked for {core_with_width}_splatted*. Skipping.")
            continue
        pairs.append((inpainted_path, splatted_path))
    return pairs


def process_one_job(
    inpainted_video_path: str,
    splatted_video_path: str,
    original_folder: str,
    output_folder: str,
    settings: Dict[str, object],
    stop_marker_path: str = "",
    preloaded_ct_csv_blend_preset_map: Optional[Dict[str, Dict[int, int]]] = None,
) -> JobPaths:
    """
    Open readers and run the streaming merge pipeline for one video.
    Returns JobPaths for optional moving.
    """
    inpainted_base_name = os.path.basename(inpainted_video_path).rsplit(".", 1)[0]
    core_with_width, is_sbs_input = parse_inpainted_name(os.path.basename(inpainted_video_path))
    core_name, width_from_name = parse_core_and_width(core_with_width)

    # Determine input type from splatted filename and probe original availability early.
    is_dual_input = "_splatted2" in os.path.basename(splatted_video_path)
    original_video_path_to_move: Optional[str] = None
    original_missing_for_dual = False
    if is_dual_input:
        original_video_path_to_move = find_video_by_core_name(original_folder, core_name)
        if not (original_video_path_to_move and os.path.exists(original_video_path_to_move)):
            original_missing_for_dual = True

    # Decide effective output format early (needed for fast skip path).
    output_format = str(settings["output_format"])
    skip_existing = bool(settings.get("skip_existing", True))
    cleanup_partials = bool(settings.get("cleanup_partial_outputs", True))
    strict_existing_validate = bool(settings.get("strict_existing_validate", False))
    if original_missing_for_dual and output_format != "Right-Eye Only":
        LOG.warning(f"Original video is missing for '{inpainted_base_name}'. Forcing output format to 'Right-Eye Only'.")
        output_format = "Right-Eye Only"

    # Fast path: if width is encoded in filename, compute output path without opening readers.
    precomputed_output_path: Optional[str] = None
    if width_from_name is not None:
        suffix, width_factor = output_suffix_and_width_factor(output_format)
        perceived_width_for_filename = int(width_from_name) * int(width_factor)
        precomputed_output_path = os.path.join(
            output_folder,
            f"{core_name}_{perceived_width_for_filename}{suffix}",
        )
        if should_skip_output(precomputed_output_path, skip_existing, strict_validate=strict_existing_validate):
            if cleanup_partials:
                cleanup_partial_output_files(precomputed_output_path)
            LOG.info(f"SKIP (exists-fast): {os.path.basename(precomputed_output_path)}")
            return JobPaths(
                inpainted_video_path,
                splatted_video_path,
                original_video_path_to_move,
                None,
                precomputed_output_path,
                os.path.splitext(inpainted_video_path)[0],
                core_name,
                is_sbs_input,
            )
        if cleanup_partials and skip_existing and os.path.exists(precomputed_output_path):
            LOG.warning(
                f"Removing invalid existing output before re-encode: "
                f"{os.path.basename(precomputed_output_path)}"
            )
            delete_if_exists(precomputed_output_path)
        if cleanup_partials:
            cleanup_partial_output_files(precomputed_output_path)

    # sidecar (may be empty dict)
    inpainted_base = os.path.splitext(inpainted_video_path)[0]
    try:
        clip_sidecar_data = read_clip_sidecar(inpainted_base) or {}
    except Exception as e:
        LOG.debug(f"Sidecar read failed for {inpainted_base}: {e}")
        clip_sidecar_data = {}

    # 1) Open readers
    inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
    splatted_reader = VideoReader(splatted_video_path, ctx=cpu(0))

    # Optional replace-mask
    replace_mask_reader = None
    replace_mask_path: Optional[str] = None
    if bool(settings.get("use_replace_mask", False)):
        replace_mask_path = find_replace_mask_for_splatted(
            splatted_video_path, str(settings.get("replace_mask_folder", "") or "")
        )
        if replace_mask_path and os.path.exists(replace_mask_path):
            try:
                replace_mask_reader = VideoReader(replace_mask_path, ctx=cpu(0))
                LOG.info(f"Using external replace mask: {os.path.basename(replace_mask_path)}")
            except Exception as e_rm:
                LOG.warning(f"Failed to open replace mask '{replace_mask_path}': {e_rm}")
                replace_mask_reader = None
                replace_mask_path = None

    # Original reader:
    original_reader = None
    original_video_path: Optional[str] = original_video_path_to_move
    if is_dual_input:
        if original_video_path and os.path.exists(original_video_path):
            LOG.info(f"Found matching original video for dual-input: {os.path.basename(original_video_path)}")
            original_reader = VideoReader(original_video_path, ctx=cpu(0))
        else:
            if not original_missing_for_dual:
                LOG.warning(f"Original video not found for dual-input mode: '{core_name}.*'.")
                LOG.warning("Will proceed, but only 'Right-Eye Only' output will be possible for this video.")
            original_reader = None
    else:
        # quad: splatted itself contains left eye
        original_reader = splatted_reader

    # 2) Determine dims
    num_frames = len(inpainted_reader)
    fps = inpainted_reader.get_avg_fps()
    video_stream_info = get_video_stream_info(inpainted_video_path)

    sample_splatted_np = splatted_reader.get_batch([0]).asnumpy()
    _, H_splat, W_splat, _ = sample_splatted_np.shape
    if is_dual_input:
        hires_H, hires_W = H_splat, W_splat // 2
    else:
        hires_H, hires_W = H_splat // 2, W_splat // 2

    # 3) Output format constraints (double-check with actual reader state)
    if original_reader is None and output_format != "Right-Eye Only":
        LOG.warning(f"Original video is missing for '{inpainted_base_name}'. Forcing output format to 'Right-Eye Only'.")
        output_format = "Right-Eye Only"

    output_path, output_width, output_height = build_output_path(
        output_folder=output_folder,
        core_name=core_name,
        hires_w=hires_W,
        hires_h=hires_H,
        output_format=output_format,
    )
    if precomputed_output_path and os.path.normpath(precomputed_output_path) != os.path.normpath(output_path):
        LOG.debug(
            f"Precomputed output path differs from probed path: "
            f"{os.path.basename(precomputed_output_path)} vs {os.path.basename(output_path)}"
        )

    output_partial_path = partial_output_path(output_path)
    if should_skip_output(output_path, skip_existing, strict_validate=strict_existing_validate):
        if cleanup_partials:
            cleanup_partial_output_files(output_path)
        LOG.info(f"SKIP (exists): {os.path.basename(output_path)}")
        return JobPaths(inpainted_video_path, splatted_video_path, original_video_path_to_move, replace_mask_path, output_path, inpainted_base, core_name, is_sbs_input)
    if cleanup_partials and skip_existing and os.path.exists(output_path):
        LOG.warning(
            f"Removing invalid existing output before re-encode: {os.path.basename(output_path)}"
        )
        delete_if_exists(output_path)
    if cleanup_partials:
        cleanup_partial_output_files(output_path)

    safe_makedirs(output_folder)

    # 4) Start ffmpeg pipe
    ffmpeg_process = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None

    # 5) Chunk loop + ffmpeg finalize
    chunk_size = int(settings.get("batch_chunk_size", 32))
    device = torch.device(str(settings.get("device", "cuda")) if torch.cuda.is_available() else "cpu")
    use_gpu_mask_ops = bool(settings.get("use_gpu_mask_ops", True)) and torch.cuda.is_available()
    ct_usage_counts = {int(p["id"]): 0.0 for p in CT_PRESETS}
    selected_ct_label = _resolve_ct_preset_label(
        str(settings.get("ct_preset", CT_PRESET_DEFAULT_LABEL))
    )
    selected_ct_preset = CT_PRESET_BY_LABEL[selected_ct_label]
    ct_auto_mode = _resolve_ct_auto_mode_from_settings(settings)
    ct_csv_blend_preset_map: Dict[str, Dict[int, int]] = (
        preloaded_ct_csv_blend_preset_map or {}
    )
    if (
        bool(settings.get("enable_color_transfer", False))
        and ct_auto_mode == CT_AUTO_MODE_CSV_BLEND
        and not ct_csv_blend_preset_map
    ):
        # Fallback for direct callers that do not pass the preloaded map.
        ct_csv_blend_preset_map = _prepare_ct_csv_blend_map_once(settings)
    encode_ok = False

    try:
        ffmpeg_process = start_ffmpeg_pipe_process(
            content_width=output_width,
            content_height=output_height,
            final_output_mp4_path=output_partial_path,
            fps=fps,
            video_stream_info=video_stream_info,
            pad_to_16_9=bool(settings.get("pad_to_16_9", False)),
            output_format_str=output_format,
        )
        if ffmpeg_process is None:
            raise RuntimeError("Failed to start FFmpeg pipe process.")

        stdout_thread = threading.Thread(
            target=_read_ffmpeg_output,
            args=(ffmpeg_process.stdout, logging.DEBUG),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_read_ffmpeg_output,
            args=(ffmpeg_process.stderr, logging.DEBUG),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        csv_blend_weights_by_frame: List[Dict[int, float]] = []
        csv_blend_lookup_key = ""
        if ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
            csv_rows_by_frame, csv_blend_lookup_key = _lookup_csv_blend_preset_rows(
                ct_csv_blend_preset_map,
                inpainted_video_path,
                core_with_width,
                core_name,
            )
            fallback_selected_id = int(selected_ct_preset["id"])
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
                LOG.warning(
                    f"CSV Blend Auto CT: no per-frame presets found for {os.path.basename(inpainted_video_path)}; "
                    f"falling back to preset #{fallback_selected_id}."
                )
            else:
                LOG.info(
                    f"CSV Blend Auto CT: {os.path.basename(inpainted_video_path)} loaded {applied_rows} frame presets "
                    f"(lookup='{csv_blend_lookup_key}')."
                )

        shadow_state: Dict[str, Any] = {"prev_components": []}
        for frame_start in range(0, num_frames, chunk_size):
            frame_end = min(frame_start + chunk_size, num_frames)
            frame_indices = list(range(frame_start, frame_end))
            if not frame_indices:
                break

            LOG.debug(f"Processing frames {frame_start + 1}-{frame_end}/{num_frames}...")

            inpainted_np = inpainted_reader.get_batch(frame_indices).asnumpy()
            splatted_np = splatted_reader.get_batch(frame_indices).asnumpy()

            replace_mask_np = None
            if replace_mask_reader is not None:
                try:
                    replace_mask_np = replace_mask_reader.get_batch(frame_indices).asnumpy()
                except Exception as e_rmread:
                    LOG.warning(f"Replace mask read failed for {inpainted_base_name} frames {frame_start}-{frame_end}: {e_rmread}")
                    replace_mask_np = None

            # tensors
            inpainted_tensor_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
            splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0

            inpainted = (
                inpainted_tensor_full[:, :, :, inpainted_tensor_full.shape[3] // 2 :]
                if is_sbs_input
                else inpainted_tensor_full
            )

            _, _, H, W = splatted_tensor.shape

            if is_dual_input:
                if original_reader is None:
                    original_left = torch.zeros_like(inpainted)
                else:
                    original_np = original_reader.get_batch(frame_indices).asnumpy()
                    original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0

                mask_raw = splatted_tensor[:, :, :, : W // 2]
                warped_original = splatted_tensor[:, :, :, W // 2 :]
            else:
                # quad: top-left is left eye, bottom-left is mask, bottom-right is warped
                original_left = splatted_tensor[:, :, : H // 2, : W // 2]
                mask_raw = splatted_tensor[:, :, H // 2 :, : W // 2]
                warped_original = splatted_tensor[:, :, H // 2 :, W // 2 :]

            # Use external replace mask if enabled
            if replace_mask_np is not None and bool(settings.get("use_replace_mask", False)):
                # GUI-aligned replace-mask conversion: use grayscale from RGB (not channel-0 only).
                if replace_mask_np.ndim == 4 and replace_mask_np.shape[3] >= 1:
                    rm_gray = replace_mask_np[..., :3].mean(axis=3)  # T,H,W
                elif replace_mask_np.ndim == 3:
                    rm_gray = replace_mask_np  # T,H,W
                else:
                    rm_gray = np.squeeze(replace_mask_np)
                rm_gray = rm_gray.astype(np.float32)
                if rm_gray.size > 0 and float(np.nanmax(rm_gray)) > 1.5:
                    rm_gray = rm_gray / 255.0
                rm = torch.from_numpy(rm_gray).float().unsqueeze(1)  # T,1,H,W
                # Ensure same H,W as mask_raw (resize if needed)
                if rm.shape[2:] != mask_raw.shape[2:]:
                    rm = F.interpolate(rm, size=mask_raw.shape[2:], mode="nearest")
                mask_raw = rm.repeat(1, 3, 1, 1)

            # Build clean binary mask for CT/ring/warped_filled (pre post-processing, GUI-aligned).
            mask_clean = mask_raw[:, 0:1, :, :].to(device)
            bin_thr = float(settings.get("mask_binarize_threshold", -1.0))
            if bin_thr >= 0.0:
                mask_bin_clean = (mask_clean > bin_thr).float()
            else:
                mask_bin_clean = (mask_clean > 0.5).float()

            # Processed mask is used only for final blending.
            processed_mask = mask_bin_clean.clone()
            # Post-process mask
            if int(settings.get("mask_dilate_kernel_size", 0)) > 0:
                processed_mask = apply_mask_dilation(processed_mask, int(settings["mask_dilate_kernel_size"]), use_gpu_mask_ops)
            if int(settings.get("mask_blur_kernel_size", 0)) > 0:
                processed_mask = apply_gaussian_blur(processed_mask, int(settings["mask_blur_kernel_size"]), use_gpu_mask_ops)

            if int(settings.get("shadow_length_px", 0)) > 0:
                processed_mask = apply_shadow_blur(
                    processed_mask,
                    base_length_px=int(settings.get("shadow_length_px", 0)),
                    curve=float(settings.get("shadow_curve", 0.0)),
                    motion_gain=float(settings.get("shadow_motion_gain", 0.0)),
                    width_adaptive=bool(settings.get("shadow_width_adaptive", True)),
                    use_gpu=use_gpu_mask_ops,
                    state=shadow_state,
                    border_tolerance_px=2,
                    width_ref_px=20.0,
                    width_power=1.0,
                )

            warped_original = warped_original.to(device)
            inpainted = inpainted.to(device)
            original_left = original_left.to(device)

            # --- Color Transfer ---
            if bool(settings.get("enable_color_transfer", True)):
                mask_bin = mask_bin_clean
                adjusted_frames: List[torch.Tensor] = []
                ct_eval_executor = (
                    ThreadPoolExecutor(max_workers=CT_AUTO_EVAL_MAX_WORKERS)
                    if ct_auto_mode == CT_AUTO_MODE_ON
                    else None
                )
                try:
                    for fi in range(inpainted.shape[0]):
                        inpainted_3 = inpainted[fi].cpu()
                        original_left_3 = original_left[fi].cpu()
                        warped_3 = warped_original[fi].cpu()
                        mask_bin_1hw = mask_bin[fi].cpu()

                        if ct_auto_mode == CT_AUTO_MODE_ON:
                            best_frame, best_preset_id = _select_best_auto_ct_preset_frame(
                                inpainted_3=inpainted_3,
                                original_left_3=original_left_3,
                                warped_3=warped_3,
                                mask_bin_1hw=mask_bin_1hw,
                                settings=settings,
                                fallback_preset_id=int(selected_ct_preset["id"]),
                                executor=ct_eval_executor,
                            )
                            ct_usage_counts[best_preset_id] += 1.0
                            adjusted_frames.append(best_frame.to(device))
                        else:
                            if ct_auto_mode == CT_AUTO_MODE_CSV_BLEND:
                                global_frame_idx = int(frame_indices[fi])
                                blend_weights = (
                                    csv_blend_weights_by_frame[global_frame_idx]
                                    if 0 <= global_frame_idx < len(csv_blend_weights_by_frame)
                                    else {}
                                )
                                if not blend_weights:
                                    fallback_selected_id = int(selected_ct_preset["id"])
                                    blend_weights = {fallback_selected_id: 1.0}
                                stats_valid_cache: Dict[str, torch.Tensor] = {}
                                warped_ref_cache: Dict[str, torch.Tensor] = {}
                                blended_3: Optional[torch.Tensor] = None
                                for pid_i, weight_i in sorted(
                                    blend_weights.items(), key=lambda kv: kv[1], reverse=True
                                ):
                                    pid = int(pid_i)
                                    w = float(max(0.0, min(1.0, float(weight_i))))
                                    if w <= 0.0:
                                        continue
                                    preset_i = CT_PRESET_BY_ID.get(pid, selected_ct_preset)
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
                                    fallback_selected_id = int(selected_ct_preset["id"])
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
                                adjusted_frames.append(torch.clamp(blended_3, 0.0, 1.0).to(device))
                            else:
                                stats_valid_cache = {}
                                warped_ref_cache = {}
                                adjusted_3 = _apply_ct_preset_frame(
                                    preset=selected_ct_preset,
                                    inpainted_3=inpainted_3,
                                    original_left_3=original_left_3,
                                    warped_3=warped_3,
                                    mask_bin_1hw=mask_bin_1hw,
                                    settings=settings,
                                    stats_valid_cache=stats_valid_cache,
                                    warped_ref_cache=warped_ref_cache,
                                )
                                ct_usage_counts[int(selected_ct_preset["id"])] += 1.0
                                adjusted_frames.append(adjusted_3.to(device))
                finally:
                    if ct_eval_executor is not None:
                        ct_eval_executor.shutdown(wait=True)

                inpainted = torch.stack(adjusted_frames, dim=0)

            blended_right_eye = warped_original * (1 - processed_mask) + inpainted * processed_mask

            # Borders from sidecar
            left_border = float(clip_sidecar_data.get("left_border", 0.0))
            right_border = float(clip_sidecar_data.get("right_border", 0.0))
            if bool(settings.get("add_borders", True)) and (left_border > 0 or right_border > 0):
                original_left, blended_right_eye = apply_borders_to_frames(
                    left_border, right_border, original_left, blended_right_eye
                )

            # Assemble output chunk
            final_chunk = assemble_output_chunk(output_format, hires_H, hires_W, original_left, blended_right_eye)

            # Write frames
            write_chunk_to_ffmpeg(ffmpeg_process, final_chunk.detach().cpu())

            draw_progress_bar(frame_end, num_frames, prefix=f"  Encoding {inpainted_base_name}:")

            # free per-chunk tensors
            del inpainted_tensor_full, splatted_tensor, inpainted, mask_raw, warped_original, processed_mask, blended_right_eye, final_chunk
            if replace_mask_np is not None:
                del replace_mask_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if bool(settings.get("enable_color_transfer", False)):
            total_ct = float(sum(ct_usage_counts.values()))
            if total_ct > 0.0:
                ct_line = " ".join(
                    [
                        f"{pid}:{(100.0 * ct_usage_counts.get(pid, 0) / total_ct):.1f}%"
                        for pid in range(1, 9)
                    ]
                )
                LOG.info(f"CT usage [{inpainted_base_name}] {ct_line}")

        # 6) Finalize ffmpeg
        try:
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
        except Exception:
            pass

        ffmpeg_process.wait(timeout=120)
        rc = getattr(ffmpeg_process, "returncode", 0)
        stop_requested_now = bool(stop_marker_path and _stop_marker_exists(stop_marker_path))

        if rc not in (0, None):
            # During graceful stop, ffmpeg may return non-zero late in finalize even when
            # a decodable .part was produced. Keep that output and stop cleanly.
            if stop_requested_now and should_skip_output(
                output_partial_path, True, strict_validate=True
            ):
                LOG.warning(
                    f"FFmpeg returned rc={rc} during stop request for {inpainted_base_name}; "
                    f"promoting readable partial output."
                )
                os.replace(output_partial_path, output_path)
                encode_ok = True
            else:
                raise RuntimeError(f"ffmpeg failed with returncode={rc} for {inpainted_base_name}")

        if not encode_ok:
            # Promote completed temp file atomically.
            os.replace(output_partial_path, output_path)
            encode_ok = True
    finally:
        # Best-effort process/thread cleanup for error paths and abrupt ffmpeg exits.
        try:
            if ffmpeg_process is not None and ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
        except Exception:
            pass

        try:
            if ffmpeg_process is not None and ffmpeg_process.poll() is None:
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=10)
        except Exception:
            try:
                if ffmpeg_process is not None:
                    ffmpeg_process.kill()
            except Exception:
                pass

        if stdout_thread is not None:
            stdout_thread.join(timeout=5)
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)

        if not encode_ok and cleanup_partials:
            cleanup_partial_output_files(output_path)

    # Cleanup reader refs
    try:
        del inpainted_reader, splatted_reader, original_reader, replace_mask_reader
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOG.info(f"DONE: {os.path.basename(output_path)}")
    return JobPaths(inpainted_video_path, splatted_video_path, original_video_path_to_move, replace_mask_path, output_path, inpainted_base, core_name, is_sbs_input)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless single-worker batch runner for merging_gui pipeline (streaming)."
    )
    ap.add_argument("--inpainted-folder", required=True, help="Folder containing *_inpainted_right_eye.mp4 or *_inpainted_sbs.mp4")
    ap.add_argument("--splatted-folder", required=True, help="Folder containing *_splatted2.mp4 / *_splatted4.mp4")
    ap.add_argument("--original-folder", required=True, help="Folder containing original left-eye videos (dual-input case)")
    ap.add_argument("--output-folder", required=True, help="Output folder for merged files")
    ap.add_argument("--stop-marker", default="", help="Path to a marker file used for graceful stop-after-current-file behavior.")
    ap.add_argument("--only", default=None, help="Process only one file (basename or prefix match)")
    ap.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (or set MERGE_DEBUG=1).")
    ap.add_argument("--verbosity", type=int, default=None, help="0=warnings,1=info,2=debug")

    # Overrides for the most relevant knobs (everything else stays in DEFAULTS above)
    ap.add_argument("--output-format", choices=OUTPUT_FORMAT_CHOICES, default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--retries", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=None)
    ap.add_argument("--no-skip-existing", action="store_true", default=False)
    ap.add_argument("--pad-to-16-9", action="store_true", default=None)
    ap.add_argument("--no-pad-to-16-9", action="store_true", default=False)
    ap.add_argument("--add-borders", action="store_true", default=None)
    ap.add_argument("--no-add-borders", action="store_true", default=False)
    # Color transfer
    ap.add_argument("--no-color-transfer", action="store_true", default=False, help="Disable color transfer entirely")
    ap.add_argument("--ct-preset", default=None, help="CT preset label or id (1..8)")
    ap.add_argument(
        "--ct-auto-mode",
        choices=CT_AUTO_MODE_OPTIONS,
        default=None,
        help="Auto CT mode: Off | On | CSV Blend",
    )
    ap.add_argument(
        "--ct-csv-blend-path",
        default=None,
        help="CSV path for per-frame preset map used by 'CSV Blend' mode.",
    )
    ap.add_argument("--ct-strength", type=float, default=None)
    ap.add_argument("--ct-black-thresh", type=float, default=None)
    ap.add_argument("--ct-min-valid-ratio", type=float, default=None)
    ap.add_argument("--ct-min-valid", type=int, default=None)
    ap.add_argument("--ct-clamp-L-min", type=float, default=None)
    ap.add_argument("--ct-clamp-L-max", type=float, default=None)
    ap.add_argument("--ct-clamp-ab-min", type=float, default=None)
    ap.add_argument("--ct-clamp-ab-max", type=float, default=None)
    ap.add_argument("--ct-exclude-black-in-target", action="store_true", default=None)
    ap.add_argument("--no-ct-exclude-black-in-target", action="store_true", default=False)
    ap.add_argument("--ct-ring-width", type=int, default=None)
    ap.add_argument("--mask-binarize-threshold", type=float, default=None, help="Threshold for building binary stats mask; -1 disables")
    ap.add_argument("--use-replace-mask", action="store_true", default=None)
    ap.add_argument("--replace-mask-folder", default=None)
    ap.add_argument("--move-finished", action="store_true", default=None)
    ap.add_argument("--move-failed", action="store_true", default=None)
    ap.add_argument("--no-cleanup-partials", action="store_true", default=False)

    args = ap.parse_args()
    debug_enabled = bool(args.debug or _env_flag("MERGE_DEBUG", False))
    verbosity = int(args.verbosity) if args.verbosity is not None else (2 if debug_enabled else 1)
    setup_logging(verbosity)
    if debug_enabled:
        _enable_debug_faulthandler()
        LOG.info("Debug mode enabled (MERGE_DEBUG/--debug).")

    # Build settings
    settings: Dict[str, object] = dict(DEFAULTS)

    if args.output_format is not None:
        settings["output_format"] = args.output_format
    if args.chunk_size is not None:
        settings["batch_chunk_size"] = args.chunk_size
    if args.retries is not None:
        settings["retries"] = max(0, args.retries)

    if args.skip_existing is True:
        settings["skip_existing"] = True
    if args.no_skip_existing is True:
        settings["skip_existing"] = False

    if args.pad_to_16_9 is True:
        settings["pad_to_16_9"] = True
    if args.no_pad_to_16_9 is True:
        settings["pad_to_16_9"] = False

    if args.add_borders is True:
        settings["add_borders"] = True
    if args.no_add_borders is True:
        settings["add_borders"] = False

    # Color transfer settings
    if args.no_color_transfer is True:
        settings["enable_color_transfer"] = False
    if args.ct_preset is not None:
        settings["ct_preset"] = _parse_ct_preset_arg(args.ct_preset)
    if args.ct_auto_mode is not None:
        settings["ct_auto_mode"] = _resolve_ct_auto_mode_label(args.ct_auto_mode)
    if args.ct_csv_blend_path is not None:
        settings["ct_csv_blend_path"] = str(args.ct_csv_blend_path).strip()
    if args.ct_strength is not None:
        settings["ct_strength"] = float(args.ct_strength)
    if args.ct_black_thresh is not None:
        settings["ct_black_thresh"] = float(args.ct_black_thresh)
    if args.ct_min_valid_ratio is not None:
        settings["ct_min_valid_ratio"] = float(args.ct_min_valid_ratio)
    if args.ct_min_valid is not None:
        settings["ct_min_valid"] = int(args.ct_min_valid)
    if args.ct_clamp_L_min is not None:
        settings["ct_clamp_L_min"] = float(args.ct_clamp_L_min)
    if args.ct_clamp_L_max is not None:
        settings["ct_clamp_L_max"] = float(args.ct_clamp_L_max)
    if args.ct_clamp_ab_min is not None:
        settings["ct_clamp_ab_min"] = float(args.ct_clamp_ab_min)
    if args.ct_clamp_ab_max is not None:
        settings["ct_clamp_ab_max"] = float(args.ct_clamp_ab_max)
    if args.ct_exclude_black_in_target is True:
        settings["ct_exclude_black_in_target"] = True
    if args.no_ct_exclude_black_in_target is True:
        settings["ct_exclude_black_in_target"] = False
    if args.ct_ring_width is not None:
        settings["ct_ring_width"] = int(args.ct_ring_width)
    if args.mask_binarize_threshold is not None:
        settings["mask_binarize_threshold"] = float(args.mask_binarize_threshold)

    settings["ct_preset"] = _parse_ct_preset_arg(str(settings.get("ct_preset", CT_PRESET_DEFAULT_LABEL)))
    settings["ct_auto_mode"] = _resolve_ct_auto_mode_from_settings(settings)

    if args.use_replace_mask is True:
        settings["use_replace_mask"] = True
    if args.replace_mask_folder is not None:
        settings["replace_mask_folder"] = args.replace_mask_folder

    if args.move_finished is True:
        settings["move_finished"] = True
    if args.move_failed is True:
        settings["move_failed"] = True

    if args.no_cleanup_partials is True:
        settings["cleanup_partial_outputs"] = False
    stop_marker_path = (
        os.path.abspath(args.stop_marker)
        if args.stop_marker
        else _default_stop_marker_path(args.output_folder)
    )

    # Collect jobs
    pairs = collect_jobs(
        inpainted_folder=args.inpainted_folder,
        splatted_folder=args.splatted_folder,
        original_folder=args.original_folder,
        output_folder=args.output_folder,
        only=args.only,
    )
    if not pairs:
        LOG.warning("No matching jobs found.")
        return 0

    try:
        preloaded_ct_csv_blend_preset_map = _prepare_ct_csv_blend_map_once(settings)
    except Exception as e:
        LOG.error(str(e))
        return 2

    LOG.info(f"Jobs: {len(pairs)}")
    finished_root = os.path.join(args.inpainted_folder, "finished")
    failed_root = os.path.join(args.inpainted_folder, "failed")
    splat_finished_root = os.path.join(args.splatted_folder, "finished")
    splat_failed_root = os.path.join(args.splatted_folder, "failed")
    orig_finished_root = os.path.join(args.original_folder, "finished")
    orig_failed_root = os.path.join(args.original_folder, "failed")
    rm_finished_root = os.path.join(str(settings.get("replace_mask_folder") or args.splatted_folder), "finished")
    rm_failed_root = os.path.join(str(settings.get("replace_mask_folder") or args.splatted_folder), "failed")

    for (inpainted_path, splatted_path) in pairs:
        if _stop_marker_exists(stop_marker_path):
            LOG.info(f"[STOP] marker detected, stopping before next file: {stop_marker_path}")
            return 0

        base = os.path.basename(inpainted_path)
        attempts_total = 1 + int(settings.get("retries", 0))
        ok = False
        last_err: Optional[Exception] = None
        job_paths: Optional[JobPaths] = None

        for attempt in range(1, attempts_total + 1):
            if _stop_marker_exists(stop_marker_path):
                LOG.info(
                    f"[STOP] marker detected before attempt {attempt}/{attempts_total} for {base}; "
                    "stopping without retry."
                )
                return 0
            try:
                LOG.info(f"[{attempt}/{attempts_total}] {base}")
                job_paths = process_one_job(
                    inpainted_video_path=inpainted_path,
                    splatted_video_path=splatted_path,
                    original_folder=args.original_folder,
                    output_folder=args.output_folder,
                    settings=settings,
                    stop_marker_path=stop_marker_path,
                    preloaded_ct_csv_blend_preset_map=preloaded_ct_csv_blend_preset_map,
                )
                ok = True
                break
            except Exception as e:
                last_err = e
                LOG.exception(f"FAILED attempt {attempt}/{attempts_total} for {base}: {e}")

                if _stop_marker_exists(stop_marker_path):
                    LOG.info(
                        f"[STOP] marker detected after failed attempt for {base}; "
                        "skipping remaining retries and exiting."
                    )
                    if bool(settings.get("cleanup_partial_outputs", True)):
                        inferred_output_path = infer_output_path_from_inpainted(
                            inpainted_video_path=inpainted_path,
                            output_folder=args.output_folder,
                            output_format=str(settings.get("output_format", "Right-Eye Only")),
                        )
                        if inferred_output_path:
                            cleanup_partial_output_files(inferred_output_path)
                    return 0

                # Cleanup partial output if requested
                if bool(settings.get("cleanup_partial_outputs", True)):
                    if job_paths is not None:
                        delete_if_exists(job_paths.output_path)
                        cleanup_partial_output_files(job_paths.output_path)
                    else:
                        inferred_output_path = infer_output_path_from_inpainted(
                            inpainted_video_path=inpainted_path,
                            output_folder=args.output_folder,
                            output_format=str(settings.get("output_format", "Right-Eye Only")),
                        )
                        if inferred_output_path:
                            # Keep valid complete outputs, remove only stale temp/invalid leftovers.
                            cleanup_partial_output_files(inferred_output_path)
                            if os.path.exists(inferred_output_path) and not should_skip_output(
                                inferred_output_path, True, strict_validate=True
                            ):
                                delete_if_exists(inferred_output_path)

                # Give some breathing room
                time.sleep(2)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if ok:
            if bool(settings.get("move_finished", False)) and job_paths is not None:
                move_file(job_paths.inpainted_video_path, finished_root)
                # move sidecar json if exists (same base)
                sidecar_candidates = glob.glob(job_paths.inpainted_base + ".*json") + glob.glob(job_paths.inpainted_base + ".json")
                for sc in sidecar_candidates:
                    move_file(sc, finished_root)

                move_file(job_paths.splatted_video_path, splat_finished_root)
                move_file(job_paths.original_video_path, orig_finished_root)
                move_file(job_paths.replace_mask_path, rm_finished_root)
        else:
            LOG.error(f"GIVING UP: {base} -> {last_err}")
            # Final best-effort cleanup on terminal failure.
            if bool(settings.get("cleanup_partial_outputs", True)):
                if job_paths is not None:
                    cleanup_partial_output_files(job_paths.output_path)
                else:
                    inferred_output_path = infer_output_path_from_inpainted(
                        inpainted_video_path=inpainted_path,
                        output_folder=args.output_folder,
                        output_format=str(settings.get("output_format", "Right-Eye Only")),
                    )
                    if inferred_output_path:
                        cleanup_partial_output_files(inferred_output_path)

            if bool(settings.get("move_failed", False)):
                move_file(inpainted_path, failed_root)
                core_with_width, _ = parse_inpainted_name(base)
                # Move matching splatted too if present
                move_file(splatted_path, splat_failed_root)

    if _stop_marker_exists(stop_marker_path):
        LOG.info(f"[STOP] marker still present at end of batch: {stop_marker_path}")

    LOG.info("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
