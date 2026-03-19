#!/usr/bin/env python3
"""Apply reference-guided sharpness/detail compensation to inpainted clips."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_inpaint_sharpness import make_right_band_roi


DEFAULT_GLOB = "*.mp4"
DEFAULT_MODE = "auto"
DEFAULT_SCOPE = "frame"
DEFAULT_MIN_SHARPNESS_LEVEL = 7
DEFAULT_MIN_COMPONENT_WIDTH = 5
DEFAULT_MAX_COMPONENTS = 0
DEFAULT_ANALYSIS_STRIDE = 1
DEFAULT_MASK_THRESHOLD = 1
DEFAULT_MIN_VALID_PIXELS = 64
DEFAULT_MIN_RATIO = 1.15
DEFAULT_MIN_REF_ENERGY = 0.0
DEFAULT_GAIN_CAP = 5.0
DEFAULT_CALIBRATION_GAINS = "1.00,1.15,1.30,1.50,1.75,2.00,2.50,3.00,4.00,5.00"
DEFAULT_LEFT_ERODE_PX = 0
DEFAULT_GROW_X = 10
DEFAULT_GROW_Y = 6
DEFAULT_FEATHER_PX = 3.0
DEFAULT_SHARP_SIGMA = 1.0
DEFAULT_DETAIL_SIGMA_SMALL = 0.85
DEFAULT_DETAIL_SIGMA_LARGE = 1.6
DEFAULT_CODEC = "libx264"
DEFAULT_PIX_FMT = "yuv444p"
DEFAULT_PRESET = "slow"
DEFAULT_CRF = "0"


@dataclass
class AnalysisResult:
    status: str
    reason: str
    frames_seen: int
    frames_sampled: int
    frames_with_components: int
    valid_metric_frames: int
    selected_component_hits: int
    top_width_1: int
    top_width_2: int
    ref_energy: float
    dst_energy: float
    mismatch_ratio: float
    correction_gain: float
    requested_mode: str
    chosen_mode: str
    calibration_frame: int
    calibration_baseline_score: float
    calibration_best_score: float
    calibration_improvement: float
    sharp_best_gain: float
    sharp_best_score: float
    detail_best_gain: float
    detail_best_score: float


@dataclass
class ReportRow:
    file: str
    output_file: str
    status: str
    reason: str
    copied_original: int
    sharpness_raw: float
    sharpness_level: int
    frames_seen: int
    frames_sampled: int
    frames_with_components: int
    valid_metric_frames: int
    selected_component_hits: int
    top_width_1: int
    top_width_2: int
    ref_energy: float
    dst_energy: float
    mismatch_ratio: float
    correction_gain: float
    requested_mode: str
    chosen_mode: str
    calibration_frame: int
    calibration_baseline_score: float
    calibration_best_score: float
    calibration_improvement: float
    sharp_best_gain: float
    sharp_best_score: float
    detail_best_gain: float
    detail_best_score: float
    scope: str
    mask_path: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Apply reference-guided sharpness/detail compensation to inpainted clips. "
            "The reference area is derived from the same right-band mask logic used by "
            "Utilities/analyze_inpaint_sharpness.py."
        )
    )
    ap.add_argument("input_dir", help="Folder with inpainted clips to process.")
    ap.add_argument("mask_dir", help="Folder with matching replace-mask videos.")
    ap.add_argument("output_dir", help="Destination folder for processed outputs.")
    ap.add_argument(
        "--sharpness-csv",
        default="",
        help="Optional sharpness CSV. If present, clips below --min-sharpness-level are skipped.",
    )
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="Input glob inside input_dir.")
    ap.add_argument(
        "--mode",
        choices=["auto", "sharp_match", "detail_match"],
        default=DEFAULT_MODE,
        help="Correction mode. 'auto' compares both methods on the first valid frame and picks the better one.",
    )
    ap.add_argument(
        "--scope",
        choices=["frame", "components"],
        default=DEFAULT_SCOPE,
        help="Apply correction to the whole frame or only around selected components.",
    )
    ap.add_argument(
        "--min-sharpness-level",
        type=int,
        default=DEFAULT_MIN_SHARPNESS_LEVEL,
        help="Skip clips whose sharpness level is lower than this.",
    )
    ap.add_argument(
        "--min-component-width",
        type=int,
        default=DEFAULT_MIN_COMPONENT_WIDTH,
        help="Minimum max per-row component width required to consider a mask component.",
    )
    ap.add_argument(
        "--max-components",
        type=int,
        default=DEFAULT_MAX_COMPONENTS,
        help="How many widest components per sampled frame to keep. Use 0 or a negative value to keep all valid components.",
    )
    ap.add_argument(
        "--analysis-stride",
        type=int,
        default=DEFAULT_ANALYSIS_STRIDE,
        help="Sample every Nth frame during the analysis pass.",
    )
    ap.add_argument(
        "--mask-threshold",
        type=int,
        default=DEFAULT_MASK_THRESHOLD,
        help="Mask threshold on grayscale values.",
    )
    ap.add_argument(
        "--min-valid-pixels",
        type=int,
        default=DEFAULT_MIN_VALID_PIXELS,
        help="Minimum pixels required in both target and reference masks to accept a sampled frame.",
    )
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help="Skip clips whose reference/detail mismatch ratio is below this.",
    )
    ap.add_argument(
        "--min-ref-energy",
        type=float,
        default=DEFAULT_MIN_REF_ENERGY,
        help="Optional absolute minimum reference detail energy required to apply any correction.",
    )
    ap.add_argument(
        "--gain-cap",
        type=float,
        default=DEFAULT_GAIN_CAP,
        help="Hard cap on tested/applied correction gains.",
    )
    ap.add_argument(
        "--calibration-gains",
        default=DEFAULT_CALIBRATION_GAINS,
        help="Comma-separated list of direct gains tested during first-frame calibration.",
    )
    ap.add_argument(
        "--component-erode-left",
        type=int,
        default=DEFAULT_LEFT_ERODE_PX,
        help="Pixels trimmed from the left side of each selected mask run before any expansion.",
    )
    ap.add_argument(
        "--component-grow-x",
        type=int,
        default=DEFAULT_GROW_X,
        help="Right-only horizontal expansion used only for scope=components.",
    )
    ap.add_argument(
        "--component-grow-y",
        type=int,
        default=DEFAULT_GROW_Y,
        help="Symmetric vertical expansion used only for scope=components.",
    )
    ap.add_argument(
        "--feather-px",
        "--feather-sigma",
        dest="feather_px",
        type=float,
        default=DEFAULT_FEATHER_PX,
        help="Right-side feather width in pixels outside the grown component mask for scope=components.",
    )
    ap.add_argument(
        "--sharp-sigma",
        type=float,
        default=DEFAULT_SHARP_SIGMA,
        help="Gaussian sigma used by sharp_match.",
    )
    ap.add_argument(
        "--detail-sigma-small",
        type=float,
        default=DEFAULT_DETAIL_SIGMA_SMALL,
        help="Small-scale sigma used by detail_match.",
    )
    ap.add_argument(
        "--detail-sigma-large",
        type=float,
        default=DEFAULT_DETAIL_SIGMA_LARGE,
        help="Large-scale sigma used by detail_match.",
    )
    ap.add_argument(
        "--report-csv",
        default="",
        help="Optional report CSV path. Default: <output_dir>/detail_match_report.csv",
    )
    ap.add_argument(
        "--summary-txt",
        default="",
        help="Optional text summary path. Default: <output_dir>/detail_match_summary.txt",
    )
    ap.add_argument(
        "--copy-skipped",
        action="store_true",
        default=True,
        help="Copy untouched originals for skipped clips (default: enabled).",
    )
    ap.add_argument(
        "--no-copy-skipped",
        dest="copy_skipped",
        action="store_false",
        help="Do not copy untouched originals for skipped clips.",
    )
    ap.add_argument("--codec", default=DEFAULT_CODEC, help="ffmpeg video codec for applied outputs.")
    ap.add_argument("--pix-fmt", default=DEFAULT_PIX_FMT, help="ffmpeg output pixel format for applied outputs.")
    ap.add_argument("--preset", default=DEFAULT_PRESET, help="ffmpeg preset for applied outputs.")
    ap.add_argument("--crf", default=DEFAULT_CRF, help="ffmpeg CRF for applied outputs.")
    return ap.parse_args()


def normalize_key(name: str) -> str:
    stem = Path(name).stem
    for suffix in (
        "_replace_mask",
        "_inpainted_right_eye",
        "_splatted1",
        "_splatted2",
        "_splatted4",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def sharpness_level_from_raw(raw: float) -> int:
    try:
        value = float(raw)
    except Exception:
        value = 0.0
    return max(5, min(11, int(math.trunc(value / 1100.0)) + 4))


def load_sharpness_index(path: str) -> Dict[str, Tuple[float, int]]:
    if not path:
        return {}
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"sharpness CSV not found: {csv_path}")

    out: Dict[str, Tuple[float, int]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str((row or {}).get("file", "")).strip()
            if not name:
                continue
            raw = float((row or {}).get("sharpness_raw", "0") or 0.0)
            level = sharpness_level_from_raw(raw)
            out[Path(name).name] = (raw, level)
            out[normalize_key(name)] = (raw, level)
    return out


def parse_gain_list(spec: str, gain_cap: float) -> List[float]:
    raw_items = [part.strip() for part in str(spec).split(",")]
    out: List[float] = []
    for item in raw_items:
        if not item:
            continue
        try:
            val = float(item)
        except Exception:
            continue
        if val < 1.0:
            continue
        val = min(float(gain_cap), float(val))
        if not any(abs(val - prev) <= 1e-6 for prev in out):
            out.append(val)
    if not out:
        out = [1.0, min(float(gain_cap), 1.5), min(float(gain_cap), 2.0)]
    out.sort()
    if out[0] > 1.0:
        out.insert(0, 1.0)
    return out


def find_mask_for_video(mask_dir: Path, video_name: str) -> Optional[Path]:
    stem = Path(video_name).stem
    core = normalize_key(video_name)

    candidates: List[str] = []
    for cand in (stem, core):
        if cand and cand not in candidates:
            candidates.append(cand)
        if cand and not any(
            cand.endswith(suffix) for suffix in ("_splatted1", "_splatted2", "_splatted4")
        ):
            for suffix in ("_splatted1", "_splatted2", "_splatted4"):
                alt = cand + suffix
                if alt not in candidates:
                    candidates.append(alt)

    for cand in candidates:
        matches = sorted(mask_dir.glob(cand + "_replace_mask.*"))
        if matches:
            return matches[0]

    if core:
        loose = sorted(mask_dir.glob(core + "*_replace_mask.*"))
        if loose:
            return loose[0]
    return None


def grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def component_union_mask(
    mask_gray: np.ndarray,
    threshold: int,
    min_component_width: int,
    max_components: int,
) -> Tuple[np.ndarray, List[int]]:
    mask = (mask_gray >= int(threshold)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: List[Tuple[int, int, int]] = []
    for lab in range(1, int(n_labels)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        comp_mask = labels[y : y + h, x : x + w] == lab
        row_widths = comp_mask.sum(axis=1)
        max_total_width = int(row_widths.max()) if row_widths.size else 0
        if max_total_width < int(min_component_width):
            continue
        comps.append((lab, max_total_width, area))

    if not comps:
        return np.zeros(mask.shape, dtype=np.uint8), []

    comps.sort(key=lambda item: (-item[1], -item[2], item[0]))
    if int(max_components) <= 0:
        keep = comps
    else:
        keep = comps[: max(1, int(max_components))]
    union = np.zeros(mask.shape, dtype=np.uint8)
    widths: List[int] = []
    for lab, width, _area in keep:
        union[labels == lab] = 255
        widths.append(int(width))
    return union, widths


def detail_energy(gray: np.ndarray, mask_u8: np.ndarray) -> float:
    if gray.size == 0 or mask_u8.size == 0:
        return 0.0
    sel = mask_u8 > 0
    if int(np.count_nonzero(sel)) <= 0:
        return 0.0
    luma = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(luma, (0, 0), 1.0)
    high = luma - blur
    return float(np.mean(np.abs(high[sel])))


def laplacian_energy(gray: np.ndarray, mask_u8: np.ndarray) -> float:
    if gray.size == 0 or mask_u8.size == 0:
        return 0.0
    sel = mask_u8 > 0
    if int(np.count_nonzero(sel)) <= 0:
        return 0.0
    luma = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(luma, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap[sel])))


def contrast_energy(gray: np.ndarray, mask_u8: np.ndarray) -> float:
    if gray.size == 0 or mask_u8.size == 0:
        return 0.0
    sel = mask_u8 > 0
    if int(np.count_nonzero(sel)) <= 0:
        return 0.0
    vals = gray.astype(np.float32)[sel]
    if vals.size <= 0:
        return 0.0
    return float(np.std(vals) / 255.0)


def metric_bundle(gray: np.ndarray, mask_u8: np.ndarray) -> Tuple[float, float, float]:
    return (
        detail_energy(gray, mask_u8),
        laplacian_energy(gray, mask_u8),
        contrast_energy(gray, mask_u8),
    )


def metric_score(
    ref_metrics: Tuple[float, float, float],
    dst_metrics: Tuple[float, float, float],
) -> float:
    rd, rl, rc = ref_metrics
    dd, dl, dc = dst_metrics
    eps = 1e-9
    err_detail = abs(rd - dd) / max(rd, eps)
    err_lap = abs(rl - dl) / max(rl, eps)
    err_contrast = abs(rc - dc) / max(rc, eps)

    overshoot = 0.0
    if dd > rd * 1.10:
        overshoot += (dd - rd * 1.10) / max(rd, eps)
    if dl > rl * 1.12:
        overshoot += 0.65 * ((dl - rl * 1.12) / max(rl, eps))
    if dc > rc * 1.15:
        overshoot += 0.35 * ((dc - rc * 1.15) / max(rc, eps))

    return float(err_detail * 0.50 + err_lap * 0.35 + err_contrast * 0.15 + overshoot)


def _iter_row_runs(row_mask: np.ndarray) -> Sequence[Tuple[int, int]]:
    xs = np.where(row_mask > 0)[0]
    if xs.size <= 0:
        return []
    runs: List[Tuple[int, int]] = []
    start = int(xs[0])
    prev = start
    for cur in xs[1:]:
        cur_i = int(cur)
        if cur_i != prev + 1:
            runs.append((start, prev))
            start = cur_i
        prev = cur_i
    runs.append((start, prev))
    return runs


def soft_component_mask(
    mask_u8: np.ndarray,
    erode_left_px: int,
    grow_x: int,
    grow_y: int,
    feather_px: float,
) -> np.ndarray:
    base = (mask_u8 > 0).astype(np.uint8)
    h, w = base.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros_like(base, dtype=np.float32)

    work = base.copy()
    erode_left_px = max(0, int(erode_left_px))
    if erode_left_px > 0:
        eroded = np.zeros_like(work)
        for y in range(h):
            for start, end in _iter_row_runs(work[y]):
                new_start = min(end + 1, start + erode_left_px)
                if new_start <= end:
                    eroded[y, new_start : end + 1] = 1
        work = eroded

    grow_x = max(0, int(grow_x))
    if grow_x > 0:
        grown = work.copy()
        for y in range(h):
            for start, end in _iter_row_runs(work[y]):
                grown[y, start : min(w, end + 1 + grow_x)] = 1
        work = grown

    grow_y = max(0, int(grow_y))
    if grow_y > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grow_y * 2 + 1))
        work = cv2.dilate(work, kernel, iterations=1)

    alpha = work.astype(np.float32)
    feather_px = float(max(0.0, feather_px))
    if feather_px > 0.0 and int(np.count_nonzero(work)) > 0:
        feather_n = max(1, int(np.ceil(feather_px)))
        denom = feather_px + 1.0
        for y in range(h):
            for _start, end in _iter_row_runs(work[y]):
                for step in range(1, feather_n + 1):
                    x = end + step
                    if x >= w:
                        break
                    value = max(0.0, 1.0 - (step / denom))
                    if value <= 0.0:
                        break
                    if value > alpha[y, x]:
                        alpha[y, x] = value
    return np.clip(alpha, 0.0, 1.0)


def apply_gain_to_frame(
    frame_bgr: np.ndarray,
    gain: float,
    mode: str,
    scope: str,
    component_mask_u8: Optional[np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    frame = np.ascontiguousarray(frame_bgr)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0] / 255.0

    gain = float(max(1.0, gain))
    if mode == "detail_match":
        small = y - cv2.GaussianBlur(y, (0, 0), float(args.detail_sigma_small))
        large = y - cv2.GaussianBlur(y, (0, 0), float(args.detail_sigma_large))
        amount_small = min(1.40, max(0.0, (gain - 1.0) * 1.45))
        amount_large = min(0.85, max(0.0, (gain - 1.0) * 0.80))
        y_adj = y + amount_small * small + amount_large * large
    else:
        high = y - cv2.GaussianBlur(y, (0, 0), float(args.sharp_sigma))
        amount = min(1.20, max(0.0, (gain - 1.0) * 1.15))
        y_adj = y + amount * high

    y_adj = np.clip(y_adj, 0.0, 1.0)

    if scope == "components":
        if component_mask_u8 is None or int(np.count_nonzero(component_mask_u8)) <= 0:
            return frame
        alpha = soft_component_mask(
            component_mask_u8,
            erode_left_px=int(getattr(args, "component_erode_left", 0)),
            grow_x=int(args.component_grow_x),
            grow_y=int(args.component_grow_y),
            feather_px=float(getattr(args, "feather_px", 0.0)),
        )
        y_out = y * (1.0 - alpha) + y_adj * alpha
    else:
        y_out = y_adj

    ycrcb[:, :, 0] = np.clip(np.round(y_out * 255.0), 0, 255)
    out = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out


def calibrate_mode_on_sample(
    sample_frame_bgr: np.ndarray,
    sample_selected_mask: np.ndarray,
    ref_metrics: Tuple[float, float, float],
    gains: Sequence[float],
    mode: str,
    scope: str,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    best_gain = 1.0
    base_gray = grayscale(sample_frame_bgr)
    base_metrics = metric_bundle(base_gray, sample_selected_mask)
    best_score = metric_score(ref_metrics, base_metrics)
    for gain in gains:
        corrected = apply_gain_to_frame(
            frame_bgr=sample_frame_bgr,
            gain=float(gain),
            mode=str(mode),
            scope=str(scope),
            component_mask_u8=sample_selected_mask if scope == "components" else None,
            args=args,
        )
        corrected_gray = grayscale(corrected)
        score = metric_score(ref_metrics, metric_bundle(corrected_gray, sample_selected_mask))
        if (score + 1e-9) < best_score or (
            abs(score - best_score) <= 1e-9 and float(gain) < float(best_gain)
        ):
            best_score = float(score)
            best_gain = float(gain)
    return float(best_gain), float(best_score)


def format_fps(fps: float) -> str:
    if fps <= 0.0:
        return "24000/1001"
    return f"{fps:.6f}"


def open_ffmpeg_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
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
        format_fps(float(fps)),
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
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def finalize_ffmpeg(proc: subprocess.Popen[bytes], output_path: Path) -> None:
    stderr = b""
    if proc.stdin is not None:
        proc.stdin.close()
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()
    rc = proc.wait()
    if rc != 0:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        msg = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg encode failed ({rc}): {msg}")


def probe_video_meta(path: Path) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return width, height, frames, fps


def analyze_clip(input_path: Path, mask_path: Path, args: argparse.Namespace) -> AnalysisResult:
    in_w, in_h, in_frames, _fps = probe_video_meta(input_path)
    mask_w, mask_h, mask_frames, _mask_fps = probe_video_meta(mask_path)
    if in_w <= 0 or in_h <= 0:
        return AnalysisResult("skipped", "invalid_input_dimensions", 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0)
    if mask_w != in_w or mask_h != in_h:
        return AnalysisResult("skipped", "mask_size_mismatch", 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0)
    if in_frames > 0 and mask_frames > 0 and in_frames != mask_frames:
        return AnalysisResult("skipped", "frame_count_mismatch", 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0)

    cap = cv2.VideoCapture(str(input_path))
    mcap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")
    if not mcap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open mask video: {mask_path}")

    stride = max(1, int(args.analysis_stride))
    frames_seen = 0
    frames_sampled = 0
    frames_with_components = 0
    valid_metric_frames = 0
    selected_component_hits = 0
    top_width_1 = 0
    top_width_2 = 0
    ref_vals: List[float] = []
    dst_vals: List[float] = []
    ratios: List[float] = []
    calibration_frame_idx = -1
    calibration_frame_bgr: Optional[np.ndarray] = None
    calibration_selected_mask: Optional[np.ndarray] = None
    calibration_ref_mask: Optional[np.ndarray] = None

    try:
        while True:
            ok_in, frame = cap.read()
            ok_mask, mask_frame = mcap.read()
            if not ok_in or frame is None or not ok_mask or mask_frame is None:
                break
            frame_idx = frames_seen
            frames_seen += 1
            if frame_idx % stride != 0:
                continue
            frames_sampled += 1

            gray = grayscale(frame)
            mask_gray = grayscale(mask_frame)
            selected_mask, widths = component_union_mask(
                mask_gray=mask_gray,
                threshold=int(args.mask_threshold),
                min_component_width=int(args.min_component_width),
                max_components=int(args.max_components),
            )
            if not widths:
                continue

            frames_with_components += 1
            selected_component_hits += int(len(widths))
            widths_sorted = sorted(int(w) for w in widths)[::-1]
            if widths_sorted:
                top_width_1 = max(top_width_1, int(widths_sorted[0]))
            if len(widths_sorted) > 1:
                top_width_2 = max(top_width_2, int(widths_sorted[1]))

            ref_mask = make_right_band_roi(selected_mask, int(args.mask_threshold))
            dst_pixels = int(np.count_nonzero(selected_mask))
            ref_pixels = int(np.count_nonzero(ref_mask))
            if dst_pixels < int(args.min_valid_pixels) or ref_pixels < int(args.min_valid_pixels):
                continue

            dst_energy = detail_energy(gray, selected_mask)
            ref_energy = detail_energy(gray, ref_mask)
            if dst_energy <= 1e-9 or ref_energy <= 1e-9:
                continue

            valid_metric_frames += 1
            ref_vals.append(float(ref_energy))
            dst_vals.append(float(dst_energy))
            ratios.append(float(ref_energy / max(dst_energy, 1e-9)))
            if calibration_frame_bgr is None:
                calibration_frame_idx = int(frame_idx)
                calibration_frame_bgr = frame.copy()
                calibration_selected_mask = selected_mask.copy()
                calibration_ref_mask = ref_mask.copy()
    finally:
        cap.release()
        mcap.release()

    if frames_with_components <= 0:
        return AnalysisResult(
            "skipped",
            "no_wide_components",
            frames_seen,
            frames_sampled,
            frames_with_components,
            valid_metric_frames,
            selected_component_hits,
            top_width_1,
            top_width_2,
            0.0,
            0.0,
            0.0,
            1.0,
            str(args.mode),
            "",
            -1,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        )

    if valid_metric_frames <= 0:
        return AnalysisResult(
            "skipped",
            "no_valid_reference_frames",
            frames_seen,
            frames_sampled,
            frames_with_components,
            valid_metric_frames,
            selected_component_hits,
            top_width_1,
            top_width_2,
            0.0,
            0.0,
            0.0,
            1.0,
            str(args.mode),
            "",
            -1,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        )

    ref_energy = float(np.median(np.asarray(ref_vals, dtype=np.float32)))
    dst_energy = float(np.median(np.asarray(dst_vals, dtype=np.float32)))
    ratio = float(np.median(np.asarray(ratios, dtype=np.float32)))
    gain = 1.0

    if ref_energy < float(args.min_ref_energy):
        return AnalysisResult(
            "skipped",
            "reference_too_soft",
            frames_seen,
            frames_sampled,
            frames_with_components,
            valid_metric_frames,
            selected_component_hits,
            top_width_1,
            top_width_2,
            ref_energy,
            dst_energy,
            ratio,
            gain,
            str(args.mode),
            "",
            int(calibration_frame_idx),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        )

    if calibration_frame_bgr is None or calibration_selected_mask is None or calibration_ref_mask is None:
        return AnalysisResult(
            "skipped",
            "no_valid_reference_frames",
            frames_seen,
            frames_sampled,
            frames_with_components,
            valid_metric_frames,
            selected_component_hits,
            top_width_1,
            top_width_2,
            ref_energy,
            dst_energy,
            ratio,
            gain,
            str(args.mode),
            "",
            int(calibration_frame_idx),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        )

    calibration_gains = parse_gain_list(str(args.calibration_gains), float(args.gain_cap))
    ref_metrics = metric_bundle(grayscale(calibration_frame_bgr), calibration_ref_mask)
    base_metrics = metric_bundle(grayscale(calibration_frame_bgr), calibration_selected_mask)
    baseline_score = metric_score(ref_metrics, base_metrics)

    sharp_best_gain = 1.0
    sharp_best_score = baseline_score
    detail_best_gain = 1.0
    detail_best_score = baseline_score

    if str(args.mode) in ("auto", "sharp_match"):
        sharp_best_gain, sharp_best_score = calibrate_mode_on_sample(
            sample_frame_bgr=calibration_frame_bgr,
            sample_selected_mask=calibration_selected_mask,
            ref_metrics=ref_metrics,
            gains=calibration_gains,
            mode="sharp_match",
            scope=str(args.scope),
            args=args,
        )
    if str(args.mode) in ("auto", "detail_match"):
        detail_best_gain, detail_best_score = calibrate_mode_on_sample(
            sample_frame_bgr=calibration_frame_bgr,
            sample_selected_mask=calibration_selected_mask,
            ref_metrics=ref_metrics,
            gains=calibration_gains,
            mode="detail_match",
            scope=str(args.scope),
            args=args,
        )

    chosen_mode = str(args.mode)
    chosen_gain = 1.0
    chosen_score = baseline_score
    if str(args.mode) == "sharp_match":
        chosen_mode = "sharp_match"
        chosen_gain = float(sharp_best_gain)
        chosen_score = float(sharp_best_score)
    elif str(args.mode) == "detail_match":
        chosen_mode = "detail_match"
        chosen_gain = float(detail_best_gain)
        chosen_score = float(detail_best_score)
    else:
        chosen_mode = "sharp_match"
        chosen_gain = float(sharp_best_gain)
        chosen_score = float(sharp_best_score)
        if float(detail_best_score) + 1e-9 < float(chosen_score) or (
            abs(float(detail_best_score) - float(chosen_score)) <= 1e-9
            and float(detail_best_gain) < float(chosen_gain)
        ):
            chosen_mode = "detail_match"
            chosen_gain = float(detail_best_gain)
            chosen_score = float(detail_best_score)

    gain = float(chosen_gain)
    improvement = float(baseline_score - chosen_score)
    if gain <= 1.0 + 1e-6:
        reason = "low_mismatch" if ratio < float(args.min_ratio) else "no_calibration_improvement"
        return AnalysisResult(
            "skipped",
            reason,
            frames_seen,
            frames_sampled,
            frames_with_components,
            valid_metric_frames,
            selected_component_hits,
            top_width_1,
            top_width_2,
            ref_energy,
            dst_energy,
            ratio,
            gain,
            str(args.mode),
            chosen_mode,
            int(calibration_frame_idx),
            float(baseline_score),
            float(chosen_score),
            float(improvement),
            float(sharp_best_gain),
            float(sharp_best_score),
            float(detail_best_gain),
            float(detail_best_score),
        )

    return AnalysisResult(
        "apply",
        "apply",
        frames_seen,
        frames_sampled,
        frames_with_components,
        valid_metric_frames,
        selected_component_hits,
        top_width_1,
        top_width_2,
        ref_energy,
        dst_energy,
        ratio,
        gain,
        str(args.mode),
        chosen_mode,
        int(calibration_frame_idx),
        float(baseline_score),
        float(chosen_score),
        float(improvement),
        float(sharp_best_gain),
        float(sharp_best_score),
        float(detail_best_gain),
        float(detail_best_score),
    )


def apply_clip(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    gain: float,
    mode: str,
    args: argparse.Namespace,
) -> int:
    width, height, _frames, fps = probe_video_meta(input_path)
    cap = cv2.VideoCapture(str(input_path))
    mcap = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open input video: {input_path}")
    if not mcap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open mask video: {mask_path}")

    writer = open_ffmpeg_writer(output_path, width, height, fps, args)
    frames_written = 0
    try:
        while True:
            ok_in, frame = cap.read()
            ok_mask, mask_frame = mcap.read()
            if not ok_in or frame is None or not ok_mask or mask_frame is None:
                break

            component_mask = None
            if args.scope == "components":
                selected_mask, _widths = component_union_mask(
                    mask_gray=grayscale(mask_frame),
                    threshold=int(args.mask_threshold),
                    min_component_width=int(args.min_component_width),
                    max_components=int(args.max_components),
                )
                component_mask = selected_mask

            out = apply_gain_to_frame(
                frame_bgr=frame,
                gain=float(gain),
                mode=str(mode),
                scope=str(args.scope),
                component_mask_u8=component_mask,
                args=args,
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
        raise
    finally:
        cap.release()
        mcap.release()

    finalize_ffmpeg(writer, output_path)
    return frames_written


def resolve_report_path(output_dir: Path, value: str, default_name: str) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return output_dir / default_name


def lookup_sharpness(index: Dict[str, Tuple[float, int]], name: str) -> Tuple[float, int]:
    if not index:
        return 0.0, 99
    if name in index:
        return index[name]
    key = normalize_key(name)
    if key in index:
        return index[key]
    return 0.0, 0


def process_file(
    input_path: Path,
    mask_dir: Path,
    output_dir: Path,
    sharpness_index: Dict[str, Tuple[float, int]],
    args: argparse.Namespace,
) -> ReportRow:
    output_path = output_dir / input_path.name
    sharpness_raw, sharpness_level = lookup_sharpness(sharpness_index, input_path.name)
    copied_original = 0

    if sharpness_index and int(sharpness_level) < int(args.min_sharpness_level):
        if args.copy_skipped:
            shutil.copy2(input_path, output_path)
            copied_original = 1
        return ReportRow(
            file=input_path.name,
            output_file=output_path.name if copied_original else "",
            status="skipped",
            reason="sharpness_below_threshold",
            copied_original=copied_original,
            sharpness_raw=float(sharpness_raw),
            sharpness_level=int(sharpness_level),
            frames_seen=0,
            frames_sampled=0,
            frames_with_components=0,
            valid_metric_frames=0,
            selected_component_hits=0,
            top_width_1=0,
            top_width_2=0,
            ref_energy=0.0,
            dst_energy=0.0,
            mismatch_ratio=0.0,
            correction_gain=1.0,
            requested_mode=str(args.mode),
            chosen_mode="",
            calibration_frame=-1,
            calibration_baseline_score=0.0,
            calibration_best_score=0.0,
            calibration_improvement=0.0,
            sharp_best_gain=1.0,
            sharp_best_score=0.0,
            detail_best_gain=1.0,
            detail_best_score=0.0,
            scope=str(args.scope),
            mask_path="",
        )

    mask_path = find_mask_for_video(mask_dir, input_path.name)
    if mask_path is None:
        if args.copy_skipped:
            shutil.copy2(input_path, output_path)
            copied_original = 1
        return ReportRow(
            file=input_path.name,
            output_file=output_path.name if copied_original else "",
            status="skipped",
            reason="missing_mask",
            copied_original=copied_original,
            sharpness_raw=float(sharpness_raw),
            sharpness_level=int(sharpness_level),
            frames_seen=0,
            frames_sampled=0,
            frames_with_components=0,
            valid_metric_frames=0,
            selected_component_hits=0,
            top_width_1=0,
            top_width_2=0,
            ref_energy=0.0,
            dst_energy=0.0,
            mismatch_ratio=0.0,
            correction_gain=1.0,
            requested_mode=str(args.mode),
            chosen_mode="",
            calibration_frame=-1,
            calibration_baseline_score=0.0,
            calibration_best_score=0.0,
            calibration_improvement=0.0,
            sharp_best_gain=1.0,
            sharp_best_score=0.0,
            detail_best_gain=1.0,
            detail_best_score=0.0,
            scope=str(args.scope),
            mask_path="",
        )

    analysis = analyze_clip(input_path, mask_path, args)
    if analysis.status != "apply":
        if args.copy_skipped:
            shutil.copy2(input_path, output_path)
            copied_original = 1
        return ReportRow(
            file=input_path.name,
            output_file=output_path.name if copied_original else "",
            status="skipped",
            reason=analysis.reason,
            copied_original=copied_original,
            sharpness_raw=float(sharpness_raw),
            sharpness_level=int(sharpness_level),
            frames_seen=analysis.frames_seen,
            frames_sampled=analysis.frames_sampled,
            frames_with_components=analysis.frames_with_components,
            valid_metric_frames=analysis.valid_metric_frames,
            selected_component_hits=analysis.selected_component_hits,
            top_width_1=analysis.top_width_1,
            top_width_2=analysis.top_width_2,
            ref_energy=analysis.ref_energy,
            dst_energy=analysis.dst_energy,
            mismatch_ratio=analysis.mismatch_ratio,
            correction_gain=analysis.correction_gain,
            requested_mode=analysis.requested_mode,
            chosen_mode=analysis.chosen_mode,
            calibration_frame=analysis.calibration_frame,
            calibration_baseline_score=analysis.calibration_baseline_score,
            calibration_best_score=analysis.calibration_best_score,
            calibration_improvement=analysis.calibration_improvement,
            sharp_best_gain=analysis.sharp_best_gain,
            sharp_best_score=analysis.sharp_best_score,
            detail_best_gain=analysis.detail_best_gain,
            detail_best_score=analysis.detail_best_score,
            scope=str(args.scope),
            mask_path=str(mask_path),
        )

    frames_written = apply_clip(
        input_path,
        mask_path,
        output_path,
        analysis.correction_gain,
        analysis.chosen_mode,
        args,
    )
    return ReportRow(
        file=input_path.name,
        output_file=output_path.name,
        status="applied",
        reason="applied",
        copied_original=0,
        sharpness_raw=float(sharpness_raw),
        sharpness_level=int(sharpness_level),
        frames_seen=max(analysis.frames_seen, frames_written),
        frames_sampled=analysis.frames_sampled,
        frames_with_components=analysis.frames_with_components,
        valid_metric_frames=analysis.valid_metric_frames,
        selected_component_hits=analysis.selected_component_hits,
        top_width_1=analysis.top_width_1,
        top_width_2=analysis.top_width_2,
        ref_energy=analysis.ref_energy,
        dst_energy=analysis.dst_energy,
        mismatch_ratio=analysis.mismatch_ratio,
        correction_gain=analysis.correction_gain,
        requested_mode=analysis.requested_mode,
        chosen_mode=analysis.chosen_mode,
        calibration_frame=analysis.calibration_frame,
        calibration_baseline_score=analysis.calibration_baseline_score,
        calibration_best_score=analysis.calibration_best_score,
        calibration_improvement=analysis.calibration_improvement,
        sharp_best_gain=analysis.sharp_best_gain,
        sharp_best_score=analysis.sharp_best_score,
        detail_best_gain=analysis.detail_best_gain,
        detail_best_score=analysis.detail_best_score,
        scope=str(args.scope),
        mask_path=str(mask_path),
    )


def write_report(report_path: Path, rows: Sequence[ReportRow]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else list(ReportRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(summary_path: Path, rows: Sequence[ReportRow]) -> None:
    counter = Counter()
    reason_counter = Counter()
    for row in rows:
        counter[row.status] += 1
        reason_counter[row.reason] += 1

    lines = [
        f"total_files={len(rows)}",
        f"applied={counter.get('applied', 0)}",
        f"skipped={counter.get('skipped', 0)}",
        f"errors={counter.get('error', 0)}",
        "",
        "reasons:",
    ]
    for reason, count in sorted(reason_counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"{reason}={count}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_row(row: ReportRow) -> None:
    if row.status == "applied":
        print(
            f"[APPLY] {row.file} mode={row.chosen_mode} gain={row.correction_gain:.3f} "
            f"ratio={row.mismatch_ratio:.3f} calib_frame={row.calibration_frame} "
            f"scores={row.calibration_baseline_score:.4f}->{row.calibration_best_score:.4f}"
        )
    else:
        extra = f" copied={row.copied_original}" if row.copied_original else ""
        print(f"[SKIP]  {row.file} reason={row.reason}{extra}")


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

    paths = sorted(Path(p).resolve() for p in glob.glob(str(input_dir / str(args.glob))))
    if not paths:
        raise SystemExit(f"no files found: {input_dir / str(args.glob)}")

    sharpness_index = load_sharpness_index(str(args.sharpness_csv)) if str(args.sharpness_csv).strip() else {}
    report_rows: List[ReportRow] = []

    for input_path in paths:
        try:
            row = process_file(input_path, mask_dir, output_dir, sharpness_index, args)
        except Exception as exc:
            row = ReportRow(
                file=input_path.name,
                output_file="",
                status="error",
                reason=f"{type(exc).__name__}",
                copied_original=0,
                sharpness_raw=float(lookup_sharpness(sharpness_index, input_path.name)[0]),
                sharpness_level=int(lookup_sharpness(sharpness_index, input_path.name)[1]),
                frames_seen=0,
                frames_sampled=0,
                frames_with_components=0,
                valid_metric_frames=0,
                selected_component_hits=0,
                top_width_1=0,
                top_width_2=0,
                ref_energy=0.0,
                dst_energy=0.0,
                mismatch_ratio=0.0,
                correction_gain=1.0,
                requested_mode=str(args.mode),
                chosen_mode="",
                calibration_frame=-1,
                calibration_baseline_score=0.0,
                calibration_best_score=0.0,
                calibration_improvement=0.0,
                sharp_best_gain=1.0,
                sharp_best_score=0.0,
                detail_best_gain=1.0,
                detail_best_score=0.0,
                scope=str(args.scope),
                mask_path="",
            )
            print(f"[ERR]   {input_path.name} {type(exc).__name__}: {exc}")
        else:
            print_row(row)
        report_rows.append(row)

    report_path = resolve_report_path(output_dir, str(args.report_csv), "detail_match_report.csv")
    summary_path = resolve_report_path(output_dir, str(args.summary_txt), "detail_match_summary.txt")
    write_report(report_path, report_rows)
    write_summary(summary_path, report_rows)
    print(f"[REPORT] {report_path}")
    print(f"[SUMMARY] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
