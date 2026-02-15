#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless batch runner for StereoCrafter merging stage.

- Scans inpainted folder for *_inpainted_right_eye.mp4 or *_inpainted_sbs.mp4
- Finds matching splatted2/splatted4 in splat folder
- Optional external replace-mask video in a separate folder
- Handles skip, validation, cleanup of corrupted outputs, and per-file retry

Designed to be launched by a wrapper .sh that sets paths and restarts on crashes.
"""

import os
import sys
import glob
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from decord import VideoReader, cpu

from dependency.stereocrafter_util import (
    logger,
    get_video_stream_info,
    draw_progress_bar,
    release_cuda_memory,
    set_util_logger_level,
    start_ffmpeg_pipe_process,
    apply_color_transfer,
    find_video_by_core_name,
    find_sidecar_file,
    read_clip_sidecar,
    apply_borders_to_frames,
)

# -------------------------
# Hardcoded knobs (edit here)
# -------------------------
USE_GPU = True
PAD_TO_16_9 = False
ADD_BORDERS = False
ENABLE_COLOR_TRANSFER = True
OUTPUT_FORMAT = "Full SBS (Left-Right)"  # or "Right-Eye Only", "Half SBS (Left-Right)", etc.

# Mask processing
USE_REPLACE_MASK = True
MASK_BINARIZE_THRESHOLD = -0.01
MASK_DILATE_KERNEL_SIZE = 4
MASK_BLUR_KERNEL_SIZE = 6
SHADOW_SHIFT = 1
SHADOW_START_OPACITY = 1
SHADOW_OPACITY_DECAY = 0.03
SHADOW_MIN_OPACITY = 0
SHADOW_DECAY_GAMMA = 1

# Batch mechanics
CHUNK_SIZE = 20
SKIP_EXISTING = True
VALIDATE_EXISTING = True  # if False, any existing output triggers skip
RETRY_ON_FAIL = 1         # retries per clip (1 = one retry)
CLEANUP_ON_FAIL = True    # delete partial output file on failure
RESUME_MOVE_TO_FINISHED = False  # like GUI "Resume": move inputs to "finished" after success

# Logging
LOG_LEVEL = logging.INFO


# -------------------------
# Mask ops (copied from GUI)
# -------------------------

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


# -------------------------
# Helpers
# -------------------------

def safe_unlink(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.warning(f"[CLEANUP] Deleted residual file: {path}")
    except Exception as e:
        logger.warning(f"[CLEANUP] Failed to delete '{path}': {e}")


def is_valid_video(path: str) -> bool:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if not VALIDATE_EXISTING:
        return True
    try:
        _ = get_video_stream_info(path)
        return True
    except Exception:
        return False


def find_replace_mask_for_splatted(splatted_path: str, replace_mask_folder: str = "") -> Optional[str]:
    """Matches GUI logic: basename of splatted + '_replace_mask.(mkv|mp4)'.

    If replace_mask_folder is empty, uses splatted's folder.
    """
    base = os.path.splitext(os.path.basename(splatted_path))[0]
    folder = (replace_mask_folder or "").strip() or os.path.dirname(splatted_path)
    for ext in (".mkv", ".mp4"):
        cand = os.path.join(folder, base + "_replace_mask" + ext)
        if os.path.exists(cand):
            return cand
    return None


def move_to_finished(src_path: str, parent_folder: str):
    if not src_path or not parent_folder:
        return
    try:
        finished_dir = os.path.join(parent_folder, "finished")
        os.makedirs(finished_dir, exist_ok=True)
        dst = os.path.join(finished_dir, os.path.basename(src_path))
        if os.path.exists(dst):
            # avoid overwrite; keep newest
            safe_unlink(dst)
        os.rename(src_path, dst)
    except Exception as e:
        logger.warning(f"[RESUME] Move failed for '{src_path}': {e}")


def parse_core_name(inpainted_filename: str) -> Tuple[Optional[str], bool]:
    """Return (core_name, is_sbs_input) from an inpainted filename."""
    inpaint_suffix = "_inpainted_right_eye.mp4"
    sbs_suffix = "_inpainted_sbs.mp4"

    is_sbs = False
    if inpainted_filename.endswith(inpaint_suffix):
        core_with_w = inpainted_filename[: -len(inpaint_suffix)]
    elif inpainted_filename.endswith(sbs_suffix):
        core_with_w = inpainted_filename[: -len(sbs_suffix)]
        is_sbs = True
    else:
        return None, False

    last_us = core_with_w.rfind("_")
    if last_us == -1:
        return None, is_sbs
    core = core_with_w[:last_us]
    return core, is_sbs


def determine_output_params(hires_w: int, hires_h: int, output_format: str) -> Tuple[int, int, int, str]:
    """Return (out_w, out_h, perceived_width_for_filename, suffix)."""
    perceived_w = hires_w
    out_h = hires_h

    if output_format == "Full SBS Cross-eye (Right-Left)":
        out_w = hires_w * 2
        suffix = "_merged_full_sbsx.mp4"
    elif output_format == "Full SBS (Left-Right)":
        out_w = hires_w * 2
        suffix = "_merged_full_sbs.mp4"
    elif output_format == "Double SBS":
        out_w = hires_w * 2
        out_h = hires_h * 2
        suffix = "_merged_half_sbs.mp4"
        perceived_w = hires_w * 2
    elif output_format == "Half SBS (Left-Right)":
        out_w = hires_w
        suffix = "_merged_half_sbs.mp4"
    elif output_format in ("Anaglyph (Red/Cyan)", "Anaglyph Half-Color"):
        out_w = hires_w
        suffix = "_merged_anaglyph.mp4"
    else:
        out_w = hires_w
        suffix = "_merged_right_eye.mp4"
    return out_w, out_h, perceived_w, suffix


@dataclass
class Settings:
    inpainted_folder: str
    splat_folder: str
    original_folder: str
    output_folder: str
    replace_mask_folder: str = ""


def process_one_clip(inpainted_path: str, s: Settings) -> bool:
    """Returns True on success."""

    base_name = os.path.basename(inpainted_path)
    core_name, is_sbs_input = parse_core_name(base_name)
    if not core_name:
        logger.error(f"Could not parse core name from '{base_name}', skipping.")
        return False

    # Sidecar (optional)
    try:
        sidecar_data = read_clip_sidecar(inpainted_path, core_name) or {}
    except Exception:
        sidecar_data = {}

    left_border = float(sidecar_data.get("left_border", 0.0) or 0.0)
    right_border = float(sidecar_data.get("right_border", 0.0) or 0.0)

    # Find splatted
    splatted4 = glob.glob(os.path.join(s.splat_folder, f"{core_name}_*_splatted4.mp4"))
    splatted2 = glob.glob(os.path.join(s.splat_folder, f"{core_name}_*_splatted2.mp4"))

    if splatted4:
        splatted_path = splatted4[0]
        is_dual_input = False
    elif splatted2:
        splatted_path = splatted2[0]
        is_dual_input = True
    else:
        logger.error(f"Missing required splatted file for '{core_name}' in '{s.splat_folder}'.")
        return False

    # Original (only for dual input)
    original_path = None
    if is_dual_input:
        original_path = find_video_by_core_name(s.original_folder, core_name)

    # Output format constraints
    output_format = OUTPUT_FORMAT
    if is_dual_input and not original_path and output_format != "Right-Eye Only":
        logger.warning(f"Original missing for '{base_name}', forcing output to Right-Eye Only.")
        output_format = "Right-Eye Only"

    # Open readers
    inpainted_reader = VideoReader(inpainted_path, ctx=cpu(0))
    splatted_reader = VideoReader(splatted_path, ctx=cpu(0))

    replace_mask_reader = None
    replace_mask_path = None
    if USE_REPLACE_MASK:
        replace_mask_path = find_replace_mask_for_splatted(splatted_path, s.replace_mask_folder)
        if replace_mask_path:
            try:
                replace_mask_reader = VideoReader(replace_mask_path, ctx=cpu(0))
                logger.info(f"Using external replace mask: {os.path.basename(replace_mask_path)}")
            except Exception as e:
                logger.warning(f"Failed to open replace mask '{replace_mask_path}': {e}")
                replace_mask_reader = None
                replace_mask_path = None

    original_reader = None
    if is_dual_input:
        if original_path and os.path.exists(original_path):
            original_reader = VideoReader(original_path, ctx=cpu(0))
        else:
            original_reader = None
    else:
        # quad: left eye is inside splatted itself
        original_reader = splatted_reader

    # Determine output dimensions
    sample = splatted_reader.get_batch([0]).asnumpy()
    _, Hs, Ws, _ = sample.shape
    if is_dual_input:
        hires_h, hires_w = Hs, Ws // 2
    else:
        hires_h, hires_w = Hs // 2, Ws // 2

    out_w, out_h, perceived_w, suffix = determine_output_params(hires_w, hires_h, output_format)
    output_filename = f"{core_name}_{perceived_w}{suffix}"
    output_path = os.path.join(s.output_folder, output_filename)

    # Skip logic
    if SKIP_EXISTING and is_valid_video(output_path):
        logger.info(f"[SKIP] Output exists: {output_filename}")
        return True
    elif SKIP_EXISTING and os.path.exists(output_path) and not is_valid_video(output_path):
        logger.warning(f"[RETRY] Output exists but invalid, deleting: {output_filename}")
        safe_unlink(output_path)

    # Encoder pipe
    os.makedirs(s.output_folder, exist_ok=True)
    num_frames = len(inpainted_reader)
    fps = inpainted_reader.get_avg_fps()
    video_stream_info = get_video_stream_info(inpainted_path)

    ff = start_ffmpeg_pipe_process(
        content_width=out_w,
        content_height=out_h,
        final_output_mp4_path=output_path,
        fps=fps,
        video_stream_info=video_stream_info,
        pad_to_16_9=PAD_TO_16_9,
        output_format_str=output_format,
    )
    if ff is None:
        raise RuntimeError("Failed to start FFmpeg pipe process.")

    # Chunk loop
    use_gpu = bool(USE_GPU and torch.cuda.is_available())
    device = "cuda" if use_gpu else "cpu"

    for frame_start in range(0, num_frames, CHUNK_SIZE):
        frame_end = min(frame_start + CHUNK_SIZE, num_frames)
        idxs = list(range(frame_start, frame_end))
        if not idxs:
            break

        inpainted_np = inpainted_reader.get_batch(idxs).asnumpy()
        splatted_np = splatted_reader.get_batch(idxs).asnumpy()

        replace_mask_np = None
        if replace_mask_reader is not None:
            try:
                replace_mask_np = replace_mask_reader.get_batch(idxs).asnumpy()
            except Exception as e_rm:
                logger.warning(f"Replace mask read failed {base_name} {frame_start}-{frame_end}: {e_rm}")
                replace_mask_np = None

        inpainted_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float().div(255.0)
        splatted = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float().div(255.0)

        inpainted = (
            inpainted_full[:, :, :, inpainted_full.shape[3] // 2 :]
            if is_sbs_input
            else inpainted_full
        )

        _, _, H, W = splatted.shape
        if is_dual_input:
            if original_reader is None:
                original_left = torch.zeros_like(inpainted)
            else:
                orig_np = original_reader.get_batch(idxs).asnumpy()
                original_left = torch.from_numpy(orig_np).permute(0, 3, 1, 2).float().div(255.0)
            mask_raw = splatted[:, :, :, : W // 2]
            warped_original = splatted[:, :, :, W // 2 :]
        else:
            original_left = splatted[:, :, : H // 2, : W // 2]
            mask_raw = splatted[:, :, H // 2 :, : W // 2]
            warped_original = splatted[:, :, H // 2 :, W // 2 :]

        # Mask prefer replace-mask if available
        if replace_mask_np is not None:
            if replace_mask_np.ndim == 4 and replace_mask_np.shape[3] >= 1:
                rm_gray = replace_mask_np[..., :3].mean(axis=3)
            elif replace_mask_np.ndim == 3:
                rm_gray = replace_mask_np
            else:
                rm_gray = replace_mask_np.squeeze()
            rm_gray = rm_gray.astype("float32")
            if rm_gray.max() > 1.5:
                rm_gray = rm_gray / 255.0
            mask = torch.from_numpy(rm_gray).float().unsqueeze(1)
        else:
            mask_np = mask_raw.permute(0, 2, 3, 1).cpu().numpy()
            mask_gray = np.mean(mask_np, axis=3)
            mask = torch.from_numpy(mask_gray).float().unsqueeze(1)

        mask = mask.to(device)
        inpainted = inpainted.to(device)
        original_left = original_left.to(device)
        warped_original = warped_original.to(device)

        if inpainted.shape[2] != hires_h or inpainted.shape[3] != hires_w:
            inpainted = F.interpolate(inpainted, size=(hires_h, hires_w), mode="bicubic", align_corners=False)
            mask = F.interpolate(mask, size=(hires_h, hires_w), mode="bilinear", align_corners=False)

        if ENABLE_COLOR_TRANSFER:
            adjusted = []
            for fi in range(inpainted.shape[0]):
                adj = apply_color_transfer(original_left[fi].cpu(), inpainted[fi].cpu())
                adjusted.append(adj.to(device))
            inpainted = torch.stack(adjusted)

        processed_mask = mask
        if MASK_BINARIZE_THRESHOLD >= 0.0:
            processed_mask = (mask > MASK_BINARIZE_THRESHOLD).float()

        if MASK_DILATE_KERNEL_SIZE > 0:
            processed_mask = apply_mask_dilation(processed_mask, int(MASK_DILATE_KERNEL_SIZE), use_gpu)
        if MASK_BLUR_KERNEL_SIZE > 0:
            processed_mask = apply_gaussian_blur(processed_mask, int(MASK_BLUR_KERNEL_SIZE), use_gpu)
        if SHADOW_SHIFT > 0:
            processed_mask = apply_shadow_blur(
                processed_mask,
                int(SHADOW_SHIFT),
                float(SHADOW_START_OPACITY),
                float(SHADOW_OPACITY_DECAY),
                float(SHADOW_MIN_OPACITY),
                float(SHADOW_DECAY_GAMMA),
                use_gpu,
            )

        blended_right = warped_original * (1 - processed_mask) + inpainted * processed_mask

        # Borders
        if ADD_BORDERS and (left_border > 0 or right_border > 0):
            original_left, blended_right = apply_borders_to_frames(left_border, right_border, original_left, blended_right)

        # Assemble
        if output_format == "Full SBS (Left-Right)":
            final_chunk = torch.cat([original_left, blended_right], dim=3)
        elif output_format == "Full SBS Cross-eye (Right-Left)":
            final_chunk = torch.cat([blended_right, original_left], dim=3)
        elif output_format == "Half SBS (Left-Right)":
            resized_left = F.interpolate(original_left, size=(hires_h, hires_w // 2), mode="bilinear", align_corners=False)
            resized_right = F.interpolate(blended_right, size=(hires_h, hires_w // 2), mode="bilinear", align_corners=False)
            final_chunk = torch.cat([resized_left, resized_right], dim=3)
        elif output_format == "Double SBS":
            sbs_chunk = torch.cat([original_left, blended_right], dim=3)
            final_chunk = F.interpolate(sbs_chunk, size=(hires_h * 2, hires_w * 2), mode="bilinear", align_corners=False)
        elif output_format == "Anaglyph (Red/Cyan)":
            final_chunk = torch.cat([original_left[:, 0:1, :, :], blended_right[:, 1:3, :, :]], dim=1)
        elif output_format == "Anaglyph Half-Color":
            left_gray = (original_left[:, 0, :, :] * 0.299 + original_left[:, 1, :, :] * 0.587 + original_left[:, 2, :, :] * 0.114).unsqueeze(1)
            final_chunk = torch.cat([left_gray, blended_right[:, 1:3, :, :]], dim=1)
        else:
            final_chunk = blended_right

        cpu_chunk = final_chunk.cpu()
        for frame_tensor in cpu_chunk:
            frame_np = frame_tensor.permute(1, 2, 0).numpy()
            frame_u16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
            frame_bgr = cv2.cvtColor(frame_u16, cv2.COLOR_RGB2BGR)
            ff.stdin.write(frame_bgr.tobytes())

        draw_progress_bar(frame_end, num_frames, prefix=f"  Encoding {base_name}:")

    # finalize
    if ff.stdin:
        ff.stdin.close()
    ff.wait(timeout=120)

    if ff.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with returncode={ff.returncode}")

    # success: optional resume move
    if RESUME_MOVE_TO_FINISHED:
        move_to_finished(inpainted_path, s.inpainted_folder)
        move_to_finished(splatted_path, s.splat_folder)
        if replace_mask_path and os.path.exists(replace_mask_path):
            move_to_finished(replace_mask_path, os.path.dirname(replace_mask_path))
        if original_path and os.path.exists(original_path):
            move_to_finished(original_path, s.original_folder)

        # sidecars for inpainted/original if present
        for p in (inpainted_path, original_path):
            if not p:
                continue
            base = os.path.splitext(p)[0]
            for ext in (".fssidecar", ".json"):
                sp = base + ext
                if os.path.exists(sp):
                    move_to_finished(sp, os.path.dirname(p))

    # release
    del inpainted_reader
    del splatted_reader
    if replace_mask_reader is not None:
        del replace_mask_reader
    if original_reader is not None and original_reader is not splatted_reader:
        del original_reader

    release_cuda_memory()
    return True


def main():
    parser = argparse.ArgumentParser(description="Headless merging runner")
    parser.add_argument("--inpainted", required=True, help="Folder with *_inpainted_*.mp4")
    parser.add_argument("--splat", required=True, help="Splat folder (hires) with *_splatted2/_splatted4.mp4")
    parser.add_argument("--original", required=True, help="Original folder (used for splatted2 mode)")
    parser.add_argument("--out", required=True, help="Output folder for merged videos")
    parser.add_argument("--replace-mask", default="", help="Replace-mask folder (optional)")
    parser.add_argument("--single", default="", help="Process only this inpainted file path")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override CHUNK_SIZE (frames per chunk)")
    parser.add_argument("--retry", type=int, default=None, help="Override RETRY_ON_FAIL (retries per clip)")
    parser.add_argument("--use-replace-mask", action="store_true", help="Force USE_REPLACE_MASK=True (use external replace-mask if available)")
    args = parser.parse_args()

    # CLI overrides for a few commonly tuned knobs (kept minimal to avoid big diffs)
    global CHUNK_SIZE, RETRY_ON_FAIL, USE_REPLACE_MASK
    if args.chunk_size is not None:
        CHUNK_SIZE = max(1, int(args.chunk_size))
    if args.retry is not None:
        RETRY_ON_FAIL = max(0, int(args.retry))
    if args.use_replace_mask:
        USE_REPLACE_MASK = True

    set_util_logger_level(LOG_LEVEL)

    s = Settings(
        inpainted_folder=os.path.abspath(args.inpainted),
        splat_folder=os.path.abspath(args.splat),
        original_folder=os.path.abspath(args.original),
        output_folder=os.path.abspath(args.out),
        replace_mask_folder=os.path.abspath(args.replace_mask) if args.replace_mask else "",
    )

    if args.single:
        vids = [os.path.abspath(args.single)]
    else:
        all_mp4 = sorted(glob.glob(os.path.join(s.inpainted_folder, "*.mp4")))
        vids = [
            p for p in all_mp4
            if p.endswith("_inpainted_right_eye.mp4") or p.endswith("_inpainted_sbs.mp4")
        ]

    if not vids:
        logger.error("No matching inpainted .mp4 files found.")
        return 2

    os.makedirs(s.output_folder, exist_ok=True)

    failed = []
    for idx, inpainted_path in enumerate(vids, start=1):
        base = os.path.basename(inpainted_path)
        logger.info(f"[{idx}/{len(vids)}] {base}")

        ok = False
        last_err = None
        for attempt in range(RETRY_ON_FAIL + 1):
            try:
                if attempt > 0:
                    logger.warning(f"[RETRY] Attempt {attempt}/{RETRY_ON_FAIL} for {base}")
                ok = process_one_clip(inpainted_path, s)
                if ok:
                    break
            except Exception as e:
                last_err = e
                logger.error(f"[FAIL] {base}: {e}")
                # try to cleanup output (residual/corrupt)
                if CLEANUP_ON_FAIL:
                    # compute output path deterministically (same logic as process_one_clip)
                    core_name, _ = parse_core_name(base)
                    if core_name:
                        # best-effort: delete any matching merged outputs for this core
                        for pat in (
                            os.path.join(s.output_folder, f"{core_name}_*_merged_*.mp4"),
                        ):
                            for fp in glob.glob(pat):
                                safe_unlink(fp)
                release_cuda_memory()
                time.sleep(0.5)

        if not ok:
            failed.append((inpainted_path, str(last_err) if last_err else "unknown"))

    if failed:
        logger.error("\nFailed clips:")
        for p, err in failed:
            logger.error(f"- {os.path.basename(p)} :: {err}")
        return 1

    logger.info("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
