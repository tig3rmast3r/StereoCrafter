#!/usr/bin/env python3
"""Apply the fixed post-inpaint sharpen preset used by Pipeline Master GUI."""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from apply_inpaint_detail_match import (
    apply_gain_to_frame,
    component_union_mask,
    finalize_ffmpeg,
    find_mask_for_video,
    format_fps,
    grayscale,
    load_sharpness_index,
    lookup_sharpness,
    probe_video_meta,
)


DEFAULT_GLOB = "*.mp4"
DEFAULT_MIN_SHARPNESS_LEVEL = 8
DEFAULT_GAIN = 2.0
DEFAULT_MASK_THRESHOLD = 1
DEFAULT_MIN_COMPONENT_WIDTH = 5
DEFAULT_MAX_COMPONENTS = 0
DEFAULT_MIN_VALID_PIXELS = 1
DEFAULT_MIN_RATIO = 1.0
DEFAULT_LEFT_ERODE_PX = 4
DEFAULT_GROW_X_RIGHT = 8
DEFAULT_GROW_Y = 0
DEFAULT_FEATHER_PX = 8.0
DEFAULT_SHARP_SIGMA = 0.10
DEFAULT_DETAIL_SIGMA_SMALL = 0.50
DEFAULT_DETAIL_SIGMA_LARGE = 3.00
DEFAULT_CODEC = "libx264"
DEFAULT_PIX_FMT = "yuv444p"
DEFAULT_PRESET = "slow"
DEFAULT_CRF = "0"


def partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")


def cleanup_partial_output(output_path: Path) -> None:
    try:
        partial_output_path(output_path).unlink()
    except FileNotFoundError:
        pass


def probe_video_timing(path: Path) -> Tuple[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,time_base",
        "-of",
        "json",
        str(path),
    ]
    try:
        import json

        out = subprocess.check_output(cmd, text=True)
        doc = json.loads(out or "{}")
    except Exception:
        return "", ""
    streams = doc.get("streams") or []
    if not streams:
        return "", ""
    stream = streams[0] or {}
    return str(stream.get("r_frame_rate") or ""), str(stream.get("time_base") or "")


@dataclass
class PresetResult:
    file: str
    output_file: str
    status: str
    reason: str
    sharpness_raw: float
    sharpness_level: int
    mask_path: str
    frames_written: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Apply the fixed detail-match/components sharpen preset to inpainted clips. "
            "Only clips with sharpness level > 7 are eligible."
        )
    )
    ap.add_argument("input_dir", help="Folder with inpainted clips.")
    ap.add_argument("mask_dir", help="Folder with replace-mask clips.")
    ap.add_argument("output_dir", help="Destination folder.")
    ap.add_argument(
        "--sharpness-csv",
        required=True,
        help="sharpness.csv used to decide eligibility (level > 7).",
    )
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="Input glob.")
    ap.add_argument(
        "--input-video",
        default="",
        help="Optional single video path. Overrides input_dir/glob when set.",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Optional basename/prefix filter when scanning input_dir.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    ap.add_argument("--codec", default=DEFAULT_CODEC)
    ap.add_argument("--pix-fmt", default=DEFAULT_PIX_FMT)
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--crf", default=DEFAULT_CRF)
    ap.add_argument("--output-extra-args", default="")
    return ap.parse_args()


def _build_processing_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        scope="components",
        mode="detail_match",
        mask_threshold=DEFAULT_MASK_THRESHOLD,
        min_component_width=DEFAULT_MIN_COMPONENT_WIDTH,
        max_components=DEFAULT_MAX_COMPONENTS,
        min_valid_pixels=DEFAULT_MIN_VALID_PIXELS,
        min_ratio=DEFAULT_MIN_RATIO,
        component_erode_left=DEFAULT_LEFT_ERODE_PX,
        component_grow_x=DEFAULT_GROW_X_RIGHT,
        component_grow_y=DEFAULT_GROW_Y,
        feather_px=DEFAULT_FEATHER_PX,
        sharp_sigma=DEFAULT_SHARP_SIGMA,
        detail_sigma_small=DEFAULT_DETAIL_SIGMA_SMALL,
        detail_sigma_large=DEFAULT_DETAIL_SIGMA_LARGE,
        codec=str(cli_args.codec),
        pix_fmt=str(cli_args.pix_fmt),
        preset=str(cli_args.preset),
        crf=str(cli_args.crf),
        output_extra_args=str(getattr(cli_args, "output_extra_args", "") or "").strip(),
    )


def open_ffmpeg_writer(
    output_path: Path,
    width: int,
    height: int,
    fps_arg: str,
    args: SimpleNamespace,
) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        str(fps_arg or "24000/1001"),
        "-i",
        "-",
        "-an",
        "-c:v",
        str(args.codec),
        "-preset",
        str(args.preset),
        "-pix_fmt",
        str(args.pix_fmt),
        "-crf",
        str(args.crf),
    ]
    extra = str(getattr(args, "output_extra_args", "") or "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    cmd.append(str(output_path))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _iter_inputs(args: argparse.Namespace) -> List[Path]:
    if str(args.input_video).strip():
        path = Path(str(args.input_video)).expanduser().resolve()
        return [path] if path.is_file() else []
    input_dir = Path(args.input_dir).expanduser().resolve()
    only = str(args.only or "").strip()
    paths = sorted(Path(p).resolve() for p in glob.glob(str(input_dir / str(args.glob))))
    if only:
        paths = [
            p for p in paths if p.name == only or p.name.startswith(only)
        ]
    return [p for p in paths if p.is_file()]


def is_eligible(
    sharpness_index: Dict[str, Tuple[float, int]],
    file_name: str,
    *,
    min_level: int = DEFAULT_MIN_SHARPNESS_LEVEL,
) -> Tuple[bool, float, int]:
    sharpness_raw, sharpness_level = lookup_sharpness(sharpness_index, file_name)
    return int(sharpness_level) >= int(min_level), float(sharpness_raw), int(sharpness_level)


def apply_sharpen_clip(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    processing_args: SimpleNamespace,
    *,
    gain: float = DEFAULT_GAIN,
) -> int:
    width, height, _frames, fps = probe_video_meta(input_path)
    input_r_frame_rate, _input_time_base = probe_video_timing(input_path)
    fps_arg = input_r_frame_rate or format_fps(float(fps))
    partial_path = partial_output_path(output_path)
    cleanup_partial_output(output_path)
    cap = cv2.VideoCapture(str(input_path))
    mcap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")
    if not mcap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open mask video: {mask_path}")

    writer = open_ffmpeg_writer(partial_path, width, height, fps_arg, processing_args)
    frames_written = 0
    try:
        while True:
            ok_in, frame = cap.read()
            ok_mask, mask_frame = mcap.read()
            if not ok_in or frame is None or not ok_mask or mask_frame is None:
                break

            selected_mask, _widths = component_union_mask(
                mask_gray=grayscale(mask_frame),
                threshold=int(processing_args.mask_threshold),
                min_component_width=int(processing_args.min_component_width),
                max_components=int(processing_args.max_components),
            )
            out = apply_gain_to_frame(
                frame_bgr=frame,
                gain=float(gain),
                mode="detail_match",
                scope="components",
                component_mask_u8=selected_mask,
                args=processing_args,
            )
            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin is not available")
            writer.stdin.write(np.ascontiguousarray(out).tobytes())
            frames_written += 1
    except Exception:
        try:
            if writer.stdin is not None:
                writer.stdin.close()
        except Exception:
            pass
        writer.kill()
        cleanup_partial_output(output_path)
        raise
    finally:
        cap.release()
        mcap.release()

    finalize_ffmpeg(writer, partial_path)
    partial_path.replace(output_path)
    return frames_written


def existing_output_is_valid(input_path: Path, output_path: Path) -> bool:
    try:
        in_w, in_h, in_frames, _in_fps = probe_video_meta(input_path)
        out_w, out_h, out_frames, _out_fps = probe_video_meta(output_path)
        in_r_frame_rate, in_time_base = probe_video_timing(input_path)
        out_r_frame_rate, out_time_base = probe_video_timing(output_path)
    except Exception:
        return False
    if in_w <= 0 or in_h <= 0 or out_w <= 0 or out_h <= 0:
        return False
    if out_w != in_w or out_h != in_h:
        return False
    if in_frames > 0 and out_frames > 0 and out_frames != in_frames:
        return False
    if in_r_frame_rate and out_r_frame_rate and out_r_frame_rate != in_r_frame_rate:
        return False
    if in_time_base and out_time_base and out_time_base != in_time_base:
        return False
    return bool(out_frames > 0 or in_frames <= 0)


def process_video(
    input_path: Path,
    mask_dir: Path,
    output_dir: Path,
    sharpness_index: Dict[str, Tuple[float, int]],
    *,
    skip_existing: bool,
    processing_args: SimpleNamespace,
) -> PresetResult:
    eligible, sharpness_raw, sharpness_level = is_eligible(sharpness_index, input_path.name)
    if not eligible:
        return PresetResult(
            file=input_path.name,
            output_file="",
            status="skipped",
            reason="sharpness_below_threshold",
            sharpness_raw=float(sharpness_raw),
            sharpness_level=int(sharpness_level),
            mask_path="",
            frames_written=0,
        )

    mask_path = find_mask_for_video(mask_dir, input_path.name)
    if mask_path is None:
        return PresetResult(
            file=input_path.name,
            output_file="",
            status="error",
            reason="missing_mask",
            sharpness_raw=float(sharpness_raw),
            sharpness_level=int(sharpness_level),
            mask_path="",
            frames_written=0,
        )

    output_path = output_dir / input_path.name
    cleanup_partial_output(output_path)
    if bool(skip_existing) and output_path.is_file():
        if existing_output_is_valid(input_path, output_path):
            return PresetResult(
                file=input_path.name,
                output_file=output_path.name,
                status="skipped",
                reason="existing_output",
                sharpness_raw=float(sharpness_raw),
                sharpness_level=int(sharpness_level),
                mask_path=str(mask_path),
                frames_written=0,
            )
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = apply_sharpen_clip(
        input_path=input_path,
        mask_path=mask_path,
        output_path=output_path,
        processing_args=processing_args,
        gain=DEFAULT_GAIN,
    )
    return PresetResult(
        file=input_path.name,
        output_file=output_path.name,
        status="applied",
        reason="applied",
        sharpness_raw=float(sharpness_raw),
        sharpness_level=int(sharpness_level),
        mask_path=str(mask_path),
        frames_written=int(frames_written),
    )


def _print_result(row: PresetResult) -> None:
    if row.status == "applied":
        print(
            f"[APPLY] {row.file} gain={DEFAULT_GAIN:.2f} "
            f"level={row.sharpness_level} frames={row.frames_written}"
        )
    elif row.status == "skipped":
        print(f"[SKIP]  {row.file} reason={row.reason}")
    else:
        print(f"[ERR]   {row.file} reason={row.reason}")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {input_dir}")
    if not mask_dir.is_dir():
        raise SystemExit(f"mask_dir not found: {mask_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_inputs(args)
    if not paths:
        raise SystemExit("no input files found")

    sharpness_index = load_sharpness_index(str(args.sharpness_csv))
    processing_args = _build_processing_args(args)
    had_error = False
    for input_path in paths:
        try:
            row = process_video(
                input_path=input_path,
                mask_dir=mask_dir,
                output_dir=output_dir,
                sharpness_index=sharpness_index,
                skip_existing=bool(args.skip_existing),
                processing_args=processing_args,
            )
        except Exception as exc:
            row = PresetResult(
                file=input_path.name,
                output_file="",
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                sharpness_raw=0.0,
                sharpness_level=0,
                mask_path="",
                frames_written=0,
            )
        _print_result(row)
        had_error = had_error or row.status == "error"
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
