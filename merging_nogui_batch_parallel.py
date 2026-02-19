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
import gc
import glob
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    "add_borders": False,              # sidecar-based borders (no-op if sidecar missing / 0%)
    "skip_existing": True,


    # Color transfer
    "enable_color_transfer": True,
    "color_transfer_mode": "safe",    # safe | legacy
    "ct_strength": 1.0,
    "ct_black_thresh": 0.0,
    "ct_min_valid_ratio": 0.0,
    "ct_min_valid": 0,
    "ct_clamp_L_min": 0.1,
    "ct_clamp_L_max": 2,
    "ct_clamp_ab_min": 0.1,
    "ct_clamp_ab_max": 3,
    "ct_exclude_black_in_target": True,
    "ct_stats_region": "ring",     # global | nonmask | ring
    "ct_ring_width": 40,
    "ct_target_stats_source": "warped",       # warped | inpainted
    "ct_reference_source": "warped_filled",            # left | warped_filled
    "mask_binarize_threshold": -0.01,   # used for stats-mask and optional binarize step if you add it later

    # Replace mask
    "use_replace_mask": True,

    # Mask post-processing
    "mask_dilate_kernel_size": 2,
    "mask_blur_kernel_size": 4,

    # Shadow (soft edge) post-processing
    "shadow_shift": 1,
    "shadow_start_opacity": 1.0,
    "shadow_opacity_decay": 0.06,
    "shadow_min_opacity": 0.0,
    "shadow_decay_gamma": 1.0,

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

    mm = (m > 0.5).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    k = int(ring_width) * 2 + 1
    pad = k // 2
    dil = F.max_pool2d(mm, kernel_size=k, stride=1, padding=pad)
    ring = (dil[0, 0] - mm[0, 0]).clamp(0, 1)
    if ring.sum().item() < 1.0:
        return inv
    return ring
    
def apply_shadow_blur(
    mask: torch.Tensor,
    shift_per_step: int,
    start_opacity: float,
    opacity_decay_per_step: float,
    min_opacity: float,
    decay_gamma: float = 1.0,
    use_gpu: bool = True,
) -> torch.Tensor:
    if shift_per_step <= 0:
        return mask
    if opacity_decay_per_step <= 1e-6:
        return mask
    num_steps = int((start_opacity - min_opacity) / opacity_decay_per_step) + 1
    if num_steps <= 0:
        return mask

    if use_gpu:
        canvas = mask.clone()
        stamp = mask.clone()
        for i in range(num_steps):
            t = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
            curved = t ** decay_gamma
            cur_op = min_opacity + (start_opacity - min_opacity) * curved
            total_shift = (i + 1) * shift_per_step
            padded = F.pad(stamp, (total_shift, 0), "constant", 0)
            shifted = padded[:, :, :, :-total_shift]
            canvas = torch.max(canvas, shifted * cur_op)
        return canvas

    out = []
    for t in range(mask.shape[0]):
        canvas_np = mask[t].squeeze(0).cpu().numpy()
        stamp_np = canvas_np.copy()
        for i in range(num_steps):
            time_step = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
            curved_t = time_step ** decay_gamma
            cur_op = min_opacity + (start_opacity - min_opacity) * curved_t
            total_shift = (i + 1) * shift_per_step
            shifted = np.roll(stamp_np, total_shift, axis=1)
            canvas_np = np.maximum(canvas_np, shifted * cur_op)
        out.append(torch.from_numpy(canvas_np).unsqueeze(0))
    return torch.stack(out).to(mask.device)

    
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
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity <= 0:
        level = logging.WARNING
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


def should_skip_output(output_path: str, skip_existing: bool) -> bool:
    return skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0


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
) -> JobPaths:
    """
    Open readers and run the streaming merge pipeline for one video.
    Returns JobPaths for optional moving.
    """
    inpainted_base_name = os.path.basename(inpainted_video_path).rsplit(".", 1)[0]
    core_with_width, is_sbs_input = parse_inpainted_name(os.path.basename(inpainted_video_path))
    core_name, _w = parse_core_and_width(core_with_width)

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

    # Determine input type from splatted filename
    is_dual_input = "_splatted2" in os.path.basename(splatted_video_path)

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
    original_video_path: Optional[str] = None
    original_video_path_to_move: Optional[str] = None
    if is_dual_input:
        original_video_path = find_video_by_core_name(original_folder, core_name)
        original_video_path_to_move = original_video_path
        if original_video_path and os.path.exists(original_video_path):
            LOG.info(f"Found matching original video for dual-input: {os.path.basename(original_video_path)}")
            original_reader = VideoReader(original_video_path, ctx=cpu(0))
        else:
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

    # 3) Output format constraints
    output_format = str(settings["output_format"])
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

    if should_skip_output(output_path, bool(settings.get("skip_existing", True))):
        LOG.info(f"SKIP (exists): {os.path.basename(output_path)}")
        return JobPaths(inpainted_video_path, splatted_video_path, original_video_path_to_move, replace_mask_path, output_path, inpainted_base, core_name, is_sbs_input)

    safe_makedirs(output_folder)

    # 4) Start ffmpeg pipe
    ffmpeg_process = start_ffmpeg_pipe_process(
        content_width=output_width,
        content_height=output_height,
        final_output_mp4_path=output_path,
        fps=fps,
        video_stream_info=video_stream_info,
        pad_to_16_9=bool(settings.get("pad_to_16_9", False)),
        output_format_str=output_format,
    )
    if ffmpeg_process is None:
        raise RuntimeError("Failed to start FFmpeg pipe process.")

    stdout_thread = threading.Thread(target=_read_ffmpeg_output, args=(ffmpeg_process.stdout, logging.DEBUG), daemon=True)
    stderr_thread = threading.Thread(target=_read_ffmpeg_output, args=(ffmpeg_process.stderr, logging.DEBUG), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # 5) Chunk loop
    chunk_size = int(settings.get("batch_chunk_size", 32))
    device = torch.device(str(settings.get("device", "cuda")) if torch.cuda.is_available() else "cpu")
    use_gpu_mask_ops = bool(settings.get("use_gpu_mask_ops", True)) and torch.cuda.is_available()

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
            # Expect replace_mask_np either single-channel or 3-channel; take first channel.
            rm_t = torch.from_numpy(replace_mask_np).permute(0, 3, 1, 2).float() / 255.0
            rm = rm_t[:, 0:1, :, :]
            # Ensure same H,W as mask_raw (resize if needed)
            if rm.shape[2:] != mask_raw.shape[2:]:
                rm = F.interpolate(rm, size=mask_raw.shape[2:], mode="nearest")
            mask_raw = rm.repeat(1, 3, 1, 1)

        # processed mask (grayscale 0..1)
        processed_mask = mask_raw[:, 0:1, :, :].to(device)


        # Match GUI: optional binarization as FIRST step of the mask chain (affects blending mask)
        bin_thr = float(settings.get("mask_binarize_threshold", -1.0))
        if bin_thr >= 0.0:
            processed_mask = (processed_mask > bin_thr).float()
        # Post-process mask
        if int(settings.get("mask_dilate_kernel_size", 0)) > 0:
            processed_mask = apply_mask_dilation(processed_mask, int(settings["mask_dilate_kernel_size"]), use_gpu_mask_ops)
        if int(settings.get("mask_blur_kernel_size", 0)) > 0:
            processed_mask = apply_gaussian_blur(processed_mask, int(settings["mask_blur_kernel_size"]), use_gpu_mask_ops)

        if int(settings.get("shadow_shift", 0)) > 0:
            processed_mask = apply_shadow_blur(
                processed_mask,
                int(settings["shadow_shift"]),
                float(settings.get("shadow_start_opacity", 0.0)),
                float(settings.get("shadow_opacity_decay", 0.0)),
                float(settings.get("shadow_min_opacity", 0.0)),
                float(settings.get("shadow_decay_gamma", 1.0)),
                use_gpu_mask_ops,
            )

        warped_original = warped_original.to(device)
        inpainted = inpainted.to(device)
        original_left = original_left.to(device)

        # --- Color Transfer ---
        if bool(settings.get("enable_color_transfer", True)):
            mode = str(settings.get("color_transfer_mode", "safe")).lower().strip()
            if mode == "legacy":
                # legacy util: per-frame
                adjusted = []
                for fi in range(inpainted.shape[0]):
                    adj = apply_color_transfer(original_left[fi].cpu(), inpainted[fi].cpu()).to(device)
                    adjusted.append(adj)
                inpainted = torch.stack(adjusted, dim=0)
            else:
                # SAFE mode: stats on stable region (global/nonmask/ring) and clamped scales
                # Match GUI: stats binarization is fixed at 0.5 on the (already post-processed) mask.
                mask_bin = (processed_mask > 0.5).float()

                adjusted = []
                for fi in range(inpainted.shape[0]):
                    stats_valid = _make_stats_mask(
                        mask_bin[fi],  # [1,H,W]
                        stats_region=str(settings.get("ct_stats_region", "nonmask")),
                        ring_width=int(settings.get("ct_ring_width", 20)),
                        use_gpu=False,
                    )

                    # choose target stats frame
                    if str(settings.get("ct_target_stats_source", "warped")) == "warped":
                        tgt_stats = warped_original[fi].cpu()
                    else:
                        tgt_stats = inpainted[fi].cpu()

                    # choose reference frame
                    ref_src = str(settings.get("ct_reference_source", "left")).strip().lower().replace("-", "_")
                    if ref_src == "warped_filled":
                        wf = warped_original[fi].cpu()
                        wf_u8 = (torch.clamp(wf, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        mm = (mask_bin[fi].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                        ref_u8 = _telea_inpaint_rgb_uint8(wf_u8, mm, radius=3)
                        ref = torch.from_numpy(ref_u8).permute(2, 0, 1).float() / 255.0
                    else:
                        ref = original_left[fi].cpu()

                    adj = apply_color_transfer_safe(
                        ref,
                        inpainted[fi].cpu(),
                        black_thresh=float(settings.get("ct_black_thresh", 8.0)),
                        min_valid_ratio=float(settings.get("ct_min_valid_ratio", 0.01)),
                        min_valid=int(settings.get("ct_min_valid", 300)),
                        strength=float(settings.get("ct_strength", 1.0)),
                        clamp_scale_L=(
                            float(settings.get("ct_clamp_L_min", 0.7)),
                            float(settings.get("ct_clamp_L_max", 1.3)),
                        ),
                        clamp_scale_ab=(
                            float(settings.get("ct_clamp_ab_min", 0.6)),
                            float(settings.get("ct_clamp_ab_max", 1.4)),
                        ),
                        exclude_black_in_target=bool(settings.get("ct_exclude_black_in_target", False)),
                        source_valid_mask=stats_valid,
                        target_valid_mask=stats_valid,
                        target_stats_frame=tgt_stats,
                    ).to(device)
                    adjusted.append(adj)
                inpainted = torch.stack(adjusted, dim=0)

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

    # 6) Finalize ffmpeg
    try:
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
    except Exception:
        pass

    # Wait then join threads
    ffmpeg_process.wait(timeout=120)
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    rc = getattr(ffmpeg_process, "returncode", 0)
    if rc not in (0, None):
        raise RuntimeError(f"ffmpeg failed with returncode={rc} for {inpainted_base_name}")

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
    ap = argparse.ArgumentParser(description="Headless batch runner for merging_gui pipeline (streaming).")
    ap.add_argument("--inpainted-folder", required=True, help="Folder containing *_inpainted_right_eye.mp4 or *_inpainted_sbs.mp4")
    ap.add_argument("--splatted-folder", required=True, help="Folder containing *_splatted2.mp4 / *_splatted4.mp4")
    ap.add_argument("--original-folder", required=True, help="Folder containing original left-eye videos (dual-input case)")
    ap.add_argument("--output-folder", required=True, help="Output folder for merged files")
    ap.add_argument("--only", default=None, help="Process only one file (basename or prefix match)")
    ap.add_argument("--verbosity", type=int, default=1, help="0=warnings,1=info,2=debug")

    # Parallel sharding (run N processes in parallel; each takes a deterministic slice)
    ap.add_argument("--num-workers", type=int, default=1, help="Total parallel workers (processes).")
    ap.add_argument("--worker-id", type=int, default=0, help="This worker index in [0, num-workers-1].")

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
    ap.add_argument("--color-transfer-mode", choices=["safe", "legacy"], default=None)
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
    ap.add_argument("--ct-stats-region", choices=["global", "nonmask", "ring"], default=None)
    ap.add_argument("--ct-ring-width", type=int, default=None)
    ap.add_argument("--ct-target-stats-source", choices=["warped", "inpainted"], default=None)
    ap.add_argument("--ct-reference-source", choices=["left", "warped_filled"], default=None)
    ap.add_argument("--mask-binarize-threshold", type=float, default=None, help="Threshold for building binary stats mask; -1 disables")
    ap.add_argument("--use-replace-mask", action="store_true", default=None)
    ap.add_argument("--replace-mask-folder", default=None)
    ap.add_argument("--move-finished", action="store_true", default=None)
    ap.add_argument("--move-failed", action="store_true", default=None)
    ap.add_argument("--no-cleanup-partials", action="store_true", default=False)

    args = ap.parse_args()
    setup_logging(args.verbosity)

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
    if args.color_transfer_mode is not None:
        settings["color_transfer_mode"] = args.color_transfer_mode
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
    if args.ct_stats_region is not None:
        settings["ct_stats_region"] = args.ct_stats_region
    if args.ct_ring_width is not None:
        settings["ct_ring_width"] = int(args.ct_ring_width)
    if args.ct_target_stats_source is not None:
        settings["ct_target_stats_source"] = args.ct_target_stats_source
    if args.ct_reference_source is not None:
        settings["ct_reference_source"] = args.ct_reference_source
    if args.mask_binarize_threshold is not None:
        settings["mask_binarize_threshold"] = float(args.mask_binarize_threshold)

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

    # Collect jobs
    pairs = collect_jobs(
        inpainted_folder=args.inpainted_folder,
        splatted_folder=args.splatted_folder,
        original_folder=args.original_folder,
        output_folder=args.output_folder,
        only=args.only,
    )
    # Shard jobs across multiple parallel processes (deterministic by sorted order)
    nw = max(1, int(args.num_workers))
    wid = int(args.worker_id)
    if wid < 0 or wid >= nw:
        LOG.error(f"Invalid --worker-id {wid} for --num-workers {nw}")
        return 2
    if nw > 1:
        pairs = [p for i, p in enumerate(pairs) if (i % nw) == wid]
        LOG.info(f"[SHARD] worker {wid}/{nw} will process {len(pairs)} jobs")
    if not pairs:
        LOG.warning("No matching jobs found.")
        return 0

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
        base = os.path.basename(inpainted_path)
        attempts_total = 1 + int(settings.get("retries", 0))
        ok = False
        last_err: Optional[Exception] = None
        job_paths: Optional[JobPaths] = None

        for attempt in range(1, attempts_total + 1):
            try:
                LOG.info(f"[{attempt}/{attempts_total}] {base}")
                job_paths = process_one_job(
                    inpainted_video_path=inpainted_path,
                    splatted_video_path=splatted_path,
                    original_folder=args.original_folder,
                    output_folder=args.output_folder,
                    settings=settings,
                )
                ok = True
                break
            except Exception as e:
                last_err = e
                LOG.exception(f"FAILED attempt {attempt}/{attempts_total} for {base}: {e}")

                # Cleanup partial output if requested
                if bool(settings.get("cleanup_partial_outputs", True)) and job_paths is not None:
                    delete_if_exists(job_paths.output_path)

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
            # Cleanup partial output even if job_paths wasn't built
            if bool(settings.get("cleanup_partial_outputs", True)):
                # best-effort: infer output path by opening splatted frame 0 is too expensive; skip
                pass

            if bool(settings.get("move_failed", False)):
                move_file(inpainted_path, failed_root)
                core_with_width, _ = parse_inpainted_name(base)
                # Move matching splatted too if present
                move_file(splatted_path, splat_failed_root)

    LOG.info("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
