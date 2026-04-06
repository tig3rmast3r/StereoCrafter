#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless batch runner for StereoCrafter splatting.

- Uses the modular `core.splatting` batch/render pipeline directly.
- Avoids instantiating `SplatterGUI`, so no Tk/display is required.
- Keeps the existing skip/retry/progress behavior used by Pipeline Master.

Usage:
  python3 batch_splatting_runner.py
"""

import os
import sys
import time
import traceback
import argparse
import csv
import logging
import queue
import threading
from types import MethodType
import numpy as np
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pathlib import Path

from dependency.stereocrafter_util import set_util_logger_level
from core.splatting.batch_processing import BatchProcessor, ProcessingSettings


def _normalize_output_root(p: Path) -> Path:
    # splatting_gui.py will create subfolders 'hires' and 'lowres' under output_splatted.
    # If output_splatted is accidentally set to .../hires or .../lowres, normalize to the parent.
    if p.name.lower() in ("hires", "lowres"):
        return p.parent
    return p



# -------------------------
# HARD-CODED SETTINGS
# -------------------------
SPLAT_GUI_PY = "./splatting_gui.py"  # legacy compatibility arg, ignored by the headless runner

INPUT_SOURCE_CLIPS = "./work/seg/"
INPUT_DEPTH_MAPS   = "./work/depthmap/"
OUTPUT_SPLATTED    = "./work/splat/"
MASK_OUTPUT        = "./work/mask/"   # empty => same folder as main output

# Core splat params
MAX_DISP = 20.0
ZERO_DISPARITY_ANCHOR = 0.20
PROCESS_LENGTH = -1  # -1 = full clip

# Task selection
ENABLE_FULL_RES = True
FULL_RES_BATCH_SIZE = 50

ENABLE_LOW_RES = False
LOW_RES_W = 1024
LOW_RES_H = 512
LOW_RES_BATCH_SIZE = 15

OUTPUT_LAYOUT = "dual"  # "quad" => _splatted4, "dual" => _splatted2, "single_warp" => _splatted1
DUAL_OUTPUT = True      # legacy fallback if OUTPUT_LAYOUT is invalid
ENABLE_GLOBAL_NORM = False
MATCH_DEPTH_RES = True

# Output encode quality values kept only for the underlying splatting code paths.
OUTPUT_CRF_FULL = 1
OUTPUT_CRF_LOW  = 23
FFMPEG_CODEC = ""
ENCODER_MODE = ""
FFMPEG_EXTRA_ARGS = ""

# Depth pre-processing
DEPTH_GAMMA = 1.0
DEPTH_DILATE_X = 1.0
DEPTH_DILATE_Y = 1.0
DEPTH_BLUR_X   = 0.0
DEPTH_BLUR_Y   = 0.0
DEPTH_DILATE_LEFT = 2.0
DEPTH_BLUR_LEFT   = 0.0

# Auto convergence mode: "Off" | "Manual" | "Average" | "Peak" | "Hybrid" | "MinBorders"
AUTO_CONVERGENCE_MODE = "MinBorders"

# Sidecar policy (runner safety): "keep" | "warn" | "prompt-delete" | "delete-all"
SIDECAR_POLICY = "delete-all"

# Optional convergence CSV (e.g. from analyze_auto_convergence_borders.py)
AUTO_CONVERGENCE_CSV = "auto_convergence_void_scan.csv"
# CSV policy: "fill-missing" (do not override sidecar anchors) | "override-all"
AUTO_CONVERGENCE_CSV_POLICY = "override-all"

# Sidecar control toggles
ENABLE_SIDECAR_GAMMA = False
ENABLE_SIDECAR_BLUR_DILATE = False

# File-moving policy
MOVE_TO_FINISHED = False  # IMPORTANT: disable finished/failed moving in GUI logic

# Skip policy
SKIP_IF_OUTPUT_EXISTS = True  # skip each task if its final mp4 exists and size>0

# Retry / cleanup policy (handles occasional ffmpeg encode failures)
RETRY_ON_FAIL = 1          # number of retries for the same clip when encoding fails (0 disables)
CLEANUP_ON_FAIL = True     # delete leftover corrupted output(s) before retry / before moving on


# Hires skip target width (matches your naming: <name>_1920_splatted2.mp4) (matches your naming: <name>_1920_splatted2.mp4)
HIRES_SKIP_WIDTH = 1920

# -------------------------
# Blur / Stair smoothing (module-level knobs in splatting_gui)
# -------------------------
# These are NOT part of the GUI settings dict; the splatting script reads them as globals.
SPLAT_STAIR_SMOOTH_ENABLED = True   # enable/disable staircase smoothing blur band
SPLAT_BLUR_KERNEL = 3               # selectable: 3/5/7/9 (box blur)
SPLAT_STAIR_EDGE_X_OFFSET = 2       # +1 shifts mask 1px to the right (inside), -1 left
SPLAT_STAIR_STRIP_PX = 3            # width (px) to the LEFT of the warped edge
SPLAT_STAIR_STRENGTH = 1.0          # 0..1

# -------------------------
# Replace-mask export (module-level knobs in splatting_gui)
# -------------------------
REPLACE_MASK_ENABLED = True          # export replace-mask (edge→hole-run)
REPLACE_MASK_SCALE = 1.0
REPLACE_MASK_MIN_PX = 1
REPLACE_MASK_MAX_PX = 32
REPLACE_MASK_GAP_TOL = 0            # not needed anymore
REPLACE_MASK_DRAW_EDGE = True       # adds one extra-left edge line (stability stays active even when False)
REPLACE_MASK_CODEC = "ffv1"
STOP_MARKER = ""


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return bool(value)
    txt = str(value or "").strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _normalize_auto_convergence_mode(value: str) -> str:
    raw = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if raw in {"off", "0", "false", "none"}:
        return "Off"
    if raw in {"manual"}:
        return "Manual"
    if raw in {"average", "avg"}:
        return "Average"
    if raw in {"peak"}:
        return "Peak"
    if raw in {"hybrid"}:
        return "Hybrid"
    if raw in {"minborders", "min borders", "minborder"}:
        return "MinBorders"
    return "MinBorders"


def _parse_args():
    p = argparse.ArgumentParser(description="Headless batch runner for the core splatting pipeline.")
    p.add_argument(
        "--gui_script",
        default=SPLAT_GUI_PY,
        help="Deprecated compatibility arg. Ignored by the headless runner.",
    )
    p.add_argument("--input_source_clips", default=INPUT_SOURCE_CLIPS, help="Folder with source clip segments.")
    p.add_argument("--input_depth_maps", default=INPUT_DEPTH_MAPS, help="Folder with depth map videos.")
    p.add_argument("--output_splatted", default=OUTPUT_SPLATTED, help="Output folder root for splatted videos.")
    p.add_argument("--mask_output", default=MASK_OUTPUT, help="Output folder for exported clean masks.")
    p.add_argument("--full_res_batch_size", type=int, default=FULL_RES_BATCH_SIZE, help="Batch size for full-res processing.")
    p.add_argument("--disparity", type=float, default=MAX_DISP, help="Max disparity percent.")
    p.add_argument("--process_length", type=int, default=PROCESS_LENGTH, help="Frames to process (-1 = full).")
    p.add_argument(
        "--enable_full_res",
        type=_parse_bool_arg,
        default=ENABLE_FULL_RES,
        help="Enable full-res task.",
    )
    p.add_argument(
        "--enable_low_res",
        type=_parse_bool_arg,
        default=ENABLE_LOW_RES,
        help="Enable low-res task.",
    )
    p.add_argument(
        "--output_layout",
        default=OUTPUT_LAYOUT,
        choices=["quad", "dual", "single_warp"],
        help="Output layout.",
    )
    p.add_argument(
        "--convergence",
        type=float,
        default=ZERO_DISPARITY_ANCHOR,
        help="Manual convergence anchor used when auto convergence is Off.",
    )
    p.add_argument("--dilate_x", type=float, default=DEPTH_DILATE_X)
    p.add_argument("--dilate_y", type=float, default=DEPTH_DILATE_Y)
    p.add_argument("--blur_x", type=float, default=DEPTH_BLUR_X)
    p.add_argument("--blur_y", type=float, default=DEPTH_BLUR_Y)
    p.add_argument("--dilate_left", type=float, default=DEPTH_DILATE_LEFT)
    p.add_argument("--blur_balance", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=DEPTH_GAMMA)
    p.add_argument(
        "--stair_smooth",
        type=_parse_bool_arg,
        default=SPLAT_STAIR_SMOOTH_ENABLED,
        help="Enable stair smoothing.",
    )
    p.add_argument("--stair_smooth_kernel", type=int, default=SPLAT_BLUR_KERNEL)
    p.add_argument("--stair_smooth_x_off", type=int, default=SPLAT_STAIR_EDGE_X_OFFSET)
    p.add_argument("--stair_smooth_strip", type=int, default=SPLAT_STAIR_STRIP_PX)
    p.add_argument("--stair_smooth_strength", type=float, default=SPLAT_STAIR_STRENGTH)
    p.add_argument(
        "--use_replace_mask",
        type=_parse_bool_arg,
        default=REPLACE_MASK_ENABLED,
        help="Enable replace-mask export.",
    )
    p.add_argument("--replace_mask_scale", type=float, default=REPLACE_MASK_SCALE)
    p.add_argument("--replace_mask_min", type=int, default=REPLACE_MASK_MIN_PX)
    p.add_argument("--replace_mask_max", type=int, default=REPLACE_MASK_MAX_PX)
    p.add_argument("--replace_mask_gap", type=int, default=REPLACE_MASK_GAP_TOL)
    p.add_argument(
        "--replace_mask_edge",
        type=_parse_bool_arg,
        default=REPLACE_MASK_DRAW_EDGE,
        help="Draw extra left edge line on replace-mask.",
    )
    p.add_argument("--replace_mask_codec", default=REPLACE_MASK_CODEC, help="Replace-mask codec.")
    p.add_argument("--ffmpeg_codec", default=FFMPEG_CODEC, help="Force output codec (optional).")
    p.add_argument("--encoder_mode", default=ENCODER_MODE, help="Shared encoder mode (lossless, crf/qp 0, crf/qp 1).")
    p.add_argument("--ffmpeg_extra_args", default=FFMPEG_EXTRA_ARGS, help="Append-only extra ffmpeg args.")
    p.add_argument("--stop_marker", default=STOP_MARKER, help="Graceful stop marker file.")
    p.add_argument(
        "--log-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose DEBUG logging for runner and splatting modules.",
    )
    p.add_argument(
        "--auto_convergence_mode",
        default=AUTO_CONVERGENCE_MODE,
        help="Auto-convergence mode to pass into processing settings (off/min_borders/... accepted).",
    )
    p.add_argument(
        "--sidecar-policy",
        default="keep",
        choices=["keep", "warn", "prompt-delete", "delete-all"],
        help="Deprecated compatibility arg. Ignored by the headless runner.",
    )
    p.add_argument(
        "--auto-convergence-csv",
        default=AUTO_CONVERGENCE_CSV,
        help="Optional CSV with per-clip convergence overrides (expects best_convergence).",
    )
    p.add_argument(
        "--auto-convergence-csv-policy",
        default=AUTO_CONVERGENCE_CSV_POLICY,
        choices=["fill-missing", "override-all"],
        help="How CSV convergence should interact with sidecar anchors.",
    )
    return p.parse_args()

def _normalize_output_layout(value, fallback_dual: bool = False) -> str:
    raw = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if raw in {"dual", "dual output"}:
        return "dual"
    if raw in {"single warp", "single", "warp", "splatted1"}:
        return "single_warp"
    if raw in {"quad", "grid", "splatted4"}:
        return "quad"
    return "dual" if bool(fallback_dual) else "quad"


def _output_layout_suffix(output_layout: str) -> str:
    layout = _normalize_output_layout(output_layout)
    if layout == "single_warp":
        return "_splatted1"
    if layout == "dual":
        return "_splatted2"
    return "_splatted4"


def _compute_task_final_out(output_video_path_base: str, target_output_width: int, output_layout: str) -> str:
    suffix = _output_layout_suffix(output_layout)
    base_no_ext = os.path.splitext(str(output_video_path_base))[0]
    return f"{base_no_ext}_{int(target_output_width)}{suffix}.mp4"


def _compute_replace_mask_out(final_out: str, replace_mask_enabled: bool, replace_mask_dir: str) -> str | None:
    if not bool(replace_mask_enabled):
        return None
    out_dir = str(replace_mask_dir or "").strip()
    if not out_dir:
        out_dir = os.path.dirname(final_out)
    base_no_ext = os.path.splitext(os.path.basename(final_out))[0]
    return os.path.join(out_dir, f"{base_no_ext}_replace_mask.mkv")


def _safe_remove(path: str | None, tag: str):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"[CLEAN] removed {tag}: {path}")
    except Exception as ex:
        print(f"[WARN] failed to remove {tag} '{path}': {ex}")


def _get_call_arg(args_list: list, kwargs: dict, index: int, name: str):
    if name in kwargs:
        return kwargs.get(name)
    if index < len(args_list):
        return args_list[index]
    return None


def _set_call_arg(args_list: list, kwargs: dict, index: int, name: str, value):
    if name in kwargs:
        kwargs[name] = value
        return
    if index < len(args_list):
        args_list[index] = value
        return
    kwargs[name] = value


def _strip_depth_suffix(stem: str) -> str:
    low = stem.lower()
    if low.endswith("_depth"):
        return stem[: -len("_depth")]
    return stem


def _strip_splatted_suffixes(stem: str) -> str:
    """Remove common post-splat suffixes to get back to source clip core name."""
    out = str(stem)
    # Handles names like:
    # - clip_1920_splatted2
    # - clip_1920_splatted2_replace_mask
    # - clip_splatted4
    # - clip_replace_mask
    out = re.sub(r"_(\d+_)?splatted[124](?:_replace_mask)?$", "", out, flags=re.IGNORECASE)
    out = re.sub(r"_replace_mask$", "", out, flags=re.IGNORECASE)
    return out


def _normalize_clip_key(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    p = Path(s)
    stem = p.stem if p.suffix else s
    stem = _strip_splatted_suffixes(stem)
    stem = _strip_depth_suffix(stem)
    return stem.strip().lower()


def _parse_float(value) -> float | None:
    try:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        return float(txt)
    except Exception:
        return None


def _read_convergence_overrides(csv_path: str) -> dict[str, float]:
    """Read clip->convergence map from CSV exported by auto-convergence analysis."""
    if not csv_path:
        return {}
    p = Path(csv_path)
    if not p.exists():
        print(f"[WARN] convergence CSV not found: {csv_path}")
        return {}

    mapping: dict[str, float] = {}
    conv_cols = ("best_convergence", "convergence", "convergence_plane")
    key_cols = ("clip_name", "video_name", "core_name", "name", "clip")
    path_cols = ("source_path", "depth_path", "source_video", "video_path", "depth_map_path")

    try:
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_val = None
                for col in conv_cols:
                    if col in row:
                        conv_val = _parse_float(row.get(col))
                    if conv_val is not None:
                        break
                if conv_val is None:
                    continue

                keys = set()
                for col in key_cols:
                    if col in row:
                        k = _normalize_clip_key(row.get(col, ""))
                        if k:
                            keys.add(k)
                for col in path_cols:
                    if col in row:
                        k = _normalize_clip_key(row.get(col, ""))
                        if k:
                            keys.add(k)
                if not keys:
                    continue

                for k in keys:
                    mapping[k] = float(conv_val)
    except Exception as ex:
        print(f"[WARN] failed parsing convergence CSV '{csv_path}': {ex}")
        return {}

    print(f"[INFO] Loaded {len(mapping)} convergence override key(s) from CSV: {csv_path}")
    return mapping


def _lookup_convergence_override(
    mapping: dict[str, float],
    video_path: str | None = None,
    depth_path: str | None = None,
) -> float | None:
    if not mapping:
        return None
    candidates = []
    if video_path:
        candidates.append(_normalize_clip_key(video_path))
    if depth_path:
        candidates.append(_normalize_clip_key(depth_path))
    for key in candidates:
        if key and key in mapping:
            return float(mapping[key])
    return None


def _list_sidecars(sidecar_folder: str, sidecar_ext: str) -> list[str]:
    try:
        folder = Path(sidecar_folder)
        if not folder.exists() or not folder.is_dir():
            return []
        return sorted(str(p) for p in folder.glob(f"*{sidecar_ext}") if p.is_file())
    except Exception:
        return []


def _handle_sidecar_policy(sidecar_folder: str, sidecar_ext: str, policy: str) -> None:
    sidecars = _list_sidecars(sidecar_folder, sidecar_ext)
    if not sidecars:
        print(f"[INFO] No sidecar files found in: {sidecar_folder}")
        return

    print(f"[WARN] Found {len(sidecars)} sidecar file(s) in: {sidecar_folder}")
    for p in sidecars[:5]:
        print(f"  - {p}")
    if len(sidecars) > 5:
        print(f"  ... and {len(sidecars) - 5} more")

    pol = str(policy or "warn").strip().lower()
    if pol in ("keep", "warn"):
        return

    do_delete = pol == "delete-all"
    if pol == "prompt-delete":
        if not sys.stdin.isatty():
            print("[WARN] prompt-delete requested but stdin is not interactive. Keeping sidecars.")
            return
        ans = input("Delete ALL sidecar files listed above? [y/N]: ").strip().lower()
        do_delete = ans in ("y", "yes")

    if not do_delete:
        print("[INFO] Keeping existing sidecar files.")
        return

    deleted = 0
    for p in sidecars:
        try:
            os.remove(p)
            deleted += 1
        except Exception as ex:
            print(f"[WARN] failed deleting sidecar '{p}': {ex}")
    print(f"[INFO] Deleted {deleted}/{len(sidecars)} sidecar file(s).")


def _default_sidecar_folder(input_depth_maps: str) -> str:
    path = Path(input_depth_maps).resolve()
    if path.is_dir():
        return str(path)
    return str(path.parent)


def main():
    args = _parse_args()
    log_level = logging.DEBUG if bool(args.log_verbose) else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for h in root_logger.handlers:
        h.setLevel(log_level)
    print(f"[INFO] runner logging level: {logging.getLevelName(log_level)}")

    # Override from CLI
    global SPLAT_GUI_PY, INPUT_SOURCE_CLIPS, INPUT_DEPTH_MAPS, OUTPUT_SPLATTED, MASK_OUTPUT, FULL_RES_BATCH_SIZE
    global MAX_DISP, PROCESS_LENGTH, ENABLE_FULL_RES, ENABLE_LOW_RES, OUTPUT_LAYOUT, DUAL_OUTPUT
    global ZERO_DISPARITY_ANCHOR, AUTO_CONVERGENCE_MODE
    global DEPTH_DILATE_X, DEPTH_DILATE_Y, DEPTH_BLUR_X, DEPTH_BLUR_Y, DEPTH_DILATE_LEFT, DEPTH_GAMMA
    global SPLAT_STAIR_SMOOTH_ENABLED, SPLAT_BLUR_KERNEL, SPLAT_STAIR_EDGE_X_OFFSET, SPLAT_STAIR_STRIP_PX, SPLAT_STAIR_STRENGTH
    global REPLACE_MASK_ENABLED, REPLACE_MASK_SCALE, REPLACE_MASK_MIN_PX, REPLACE_MASK_MAX_PX, REPLACE_MASK_GAP_TOL, REPLACE_MASK_DRAW_EDGE, REPLACE_MASK_CODEC
    global FFMPEG_CODEC, ENCODER_MODE, FFMPEG_EXTRA_ARGS, STOP_MARKER
    INPUT_SOURCE_CLIPS = args.input_source_clips
    INPUT_DEPTH_MAPS = args.input_depth_maps
    OUTPUT_SPLATTED = args.output_splatted
    MASK_OUTPUT = args.mask_output
    FULL_RES_BATCH_SIZE = int(args.full_res_batch_size)
    MAX_DISP = float(args.disparity)
    PROCESS_LENGTH = int(args.process_length)
    ENABLE_FULL_RES = bool(args.enable_full_res)
    ENABLE_LOW_RES = bool(args.enable_low_res)
    OUTPUT_LAYOUT = _normalize_output_layout(args.output_layout, fallback_dual=bool(DUAL_OUTPUT))
    DUAL_OUTPUT = bool(OUTPUT_LAYOUT == "dual")
    conv_raw = float(args.convergence)
    if conv_raw > 1.0:
        conv_raw = conv_raw / 100.0
    ZERO_DISPARITY_ANCHOR = max(0.0, min(1.0, conv_raw))
    AUTO_CONVERGENCE_MODE = _normalize_auto_convergence_mode(args.auto_convergence_mode)
    DEPTH_DILATE_X = float(args.dilate_x)
    DEPTH_DILATE_Y = float(args.dilate_y)
    DEPTH_BLUR_X = float(args.blur_x)
    DEPTH_BLUR_Y = float(args.blur_y)
    DEPTH_DILATE_LEFT = float(args.dilate_left)
    DEPTH_GAMMA = float(args.gamma)
    SPLAT_STAIR_SMOOTH_ENABLED = bool(args.stair_smooth)
    SPLAT_BLUR_KERNEL = int(args.stair_smooth_kernel)
    SPLAT_STAIR_EDGE_X_OFFSET = int(args.stair_smooth_x_off)
    SPLAT_STAIR_STRIP_PX = int(args.stair_smooth_strip)
    SPLAT_STAIR_STRENGTH = float(args.stair_smooth_strength)
    REPLACE_MASK_ENABLED = bool(args.use_replace_mask)
    REPLACE_MASK_SCALE = float(args.replace_mask_scale)
    REPLACE_MASK_MIN_PX = int(args.replace_mask_min)
    REPLACE_MASK_MAX_PX = int(args.replace_mask_max)
    REPLACE_MASK_GAP_TOL = int(args.replace_mask_gap)
    REPLACE_MASK_DRAW_EDGE = bool(args.replace_mask_edge)
    REPLACE_MASK_CODEC = str(args.replace_mask_codec).strip() or "ffv1"
    FFMPEG_CODEC = str(args.ffmpeg_codec).strip().lower()
    ENCODER_MODE = str(args.encoder_mode).strip()
    FFMPEG_EXTRA_ARGS = str(args.ffmpeg_extra_args).strip()
    STOP_MARKER = str(args.stop_marker).strip()

    if FFMPEG_CODEC or ENCODER_MODE or FFMPEG_EXTRA_ARGS:
        print(
            "[INFO] ffmpeg output policy: "
            f"codec={FFMPEG_CODEC or 'default'} "
            f"mode={ENCODER_MODE or 'default'} "
            f"extra={'set' if FFMPEG_EXTRA_ARGS else 'none'}"
        )
    try:
        set_util_logger_level(log_level)
    except Exception as ex:
        print(f"[WARN] failed applying util logger level: {ex}")

    # Optional ffmpeg output-override hook (used by pipeline master overrides).
    import core.splatting.render_processor as _render_mod
    _orig_start_ffmpeg = _render_mod.start_ffmpeg_pipe_process

    def _start_ffmpeg_pipe_process_wrapper(*ff_args, **ff_kwargs):
        if FFMPEG_CODEC and not ff_kwargs.get("force_output_codec"):
            ff_kwargs["force_output_codec"] = FFMPEG_CODEC
        if ENCODER_MODE and not ff_kwargs.get("encoding_mode"):
            ff_kwargs["encoding_mode"] = ENCODER_MODE
        if FFMPEG_EXTRA_ARGS and not ff_kwargs.get("ffmpeg_extra_output_args"):
            ff_kwargs["ffmpeg_extra_output_args"] = FFMPEG_EXTRA_ARGS
        return _orig_start_ffmpeg(*ff_args, **ff_kwargs)

    _render_mod.start_ffmpeg_pipe_process = _start_ffmpeg_pipe_process_wrapper  # type: ignore

    # Monkey-patch RenderProcessor.render_video for:
    # - per-task skip-if-output-exists
    # - retry on failure
    # - cleanup failed outputs
    from core.splatting.render_processor import RenderProcessor

    orig_render_video = RenderProcessor.render_video
    progress_tracker = {"done": 0, "total": 0}
    progress_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    batch_processor = BatchProcessor(
        progress_queue=progress_queue,
        stop_event=stop_event,
        sidecar_manager=None,
    )

    def _emit_task_progress(delta: int = 1) -> None:
        progress_tracker["done"] = max(0, int(progress_tracker["done"]) + int(delta))
        total = max(0, int(progress_tracker.get("total", 0)))
        if total > 0:
            print(f"[RUN ] {progress_tracker['done']}/{total}")
        else:
            print(f"[RUN ] {progress_tracker['done']}")

    def render_video_wrapper(self, *args, **kwargs):
        args_list = list(args)

        output_video_path_base = _get_call_arg(args_list, kwargs, 4, "output_video_path_base")
        target_output_width = _get_call_arg(args_list, kwargs, 6, "target_output_width")
        dual_output_raw = _get_call_arg(args_list, kwargs, 9, "dual_output")
        dual_output = bool(DUAL_OUTPUT if dual_output_raw is None else dual_output_raw)
        output_layout_raw = _get_call_arg(args_list, kwargs, 10, "output_layout")
        output_layout = _normalize_output_layout(output_layout_raw, fallback_dual=dual_output)
        replace_mask_enabled_raw = _get_call_arg(
            args_list, kwargs, 38, "replace_mask_enabled"
        )
        if replace_mask_enabled_raw is None:
            replace_mask_enabled_raw = _get_call_arg(args_list, kwargs, 37, "replace_mask_enabled")
        replace_mask_enabled = bool(
            REPLACE_MASK_ENABLED
            if replace_mask_enabled_raw is None
            else replace_mask_enabled_raw
        )
        replace_mask_dir_raw = _get_call_arg(args_list, kwargs, 39, "replace_mask_dir")
        if replace_mask_dir_raw is None:
            replace_mask_dir_raw = _get_call_arg(args_list, kwargs, 38, "replace_mask_dir")
        replace_mask_dir = str(MASK_OUTPUT if replace_mask_dir_raw is None else replace_mask_dir_raw)

        final_out = None
        replace_out = None
        try:
            if output_video_path_base is not None and target_output_width is not None:
                final_out = _compute_task_final_out(
                    str(output_video_path_base),
                    int(target_output_width),
                    output_layout,
                )
                replace_out = _compute_replace_mask_out(
                    final_out=final_out,
                    replace_mask_enabled=replace_mask_enabled,
                    replace_mask_dir=replace_mask_dir,
                )

                if SKIP_IF_OUTPUT_EXISTS and os.path.exists(final_out) and os.path.getsize(final_out) > 0:
                    print(f"[SKIP] task output exists: {final_out}")
                    _emit_task_progress(1)
                    return True
        except Exception as ex:
            print(f"[WARN] task skip-check failed, continuing: {ex}")

        max_attempts = 1 + max(0, int(RETRY_ON_FAIL))
        last_exc = None

        for attempt in range(1, max_attempts + 1):
            ok = False
            try:
                ok = bool(orig_render_video(self, *args_list, **kwargs))
            except Exception as ex:
                last_exc = ex
                ok = False
                print(f"[ERR ] render_video raised: {ex}")

            if ok:
                _emit_task_progress(1)
                return True

            if getattr(self, "stop_event", None) is not None and self.stop_event.is_set():
                return False

            if bool(CLEANUP_ON_FAIL):
                _safe_remove(final_out, "splat out")
                _safe_remove(replace_out, "replace mask")

            if attempt < max_attempts:
                print(f"[RETRY] render failed, retrying ({attempt}/{max_attempts - 1})...")
                continue

        if last_exc is not None:
            print(f"[ERR ] giving up after {max_attempts} attempt(s). Last exception: {last_exc}")
        else:
            print(f"[ERR ] giving up after {max_attempts} attempt(s).")
        _emit_task_progress(1)
        return False

    RenderProcessor.render_video = render_video_wrapper  # type: ignore

    # Monkey-patch BatchProcessor orchestration for early skip of whole video.
    if SKIP_IF_OUTPUT_EXISTS:
        orig_orchestration = batch_processor._process_single_video_orchestration

        def _process_single_video_orchestration_skip_wrapper(self, *args, **kwargs):
            try:
                video_path = kwargs.get("video_path", None)
                settings = kwargs.get("settings", None)
                if video_path is None and len(args) >= 1:
                    video_path = args[0]
                if settings is None and len(args) >= 2:
                    settings = args[1]

                if video_path is not None and settings is not None and getattr(settings, "enable_full_resolution", False):
                    if STOP_MARKER and os.path.exists(STOP_MARKER):
                        stop_event.set()
                        print("[STOP] stop marker detected before next clip. Exiting batch loop.")
                        return len(self.get_defined_tasks(settings))
                    out_root = Path(getattr(settings, "output_splatted", OUTPUT_SPLATTED)).resolve()
                    base_name = Path(str(video_path)).stem
                    suffix = _output_layout_suffix(
                        _normalize_output_layout(
                            getattr(settings, "output_layout", None),
                            fallback_dual=bool(getattr(settings, "dual_output", DUAL_OUTPUT)),
                        )
                    )
                    final_out = out_root / "hires" / f"{base_name}_{int(HIRES_SKIP_WIDTH)}{suffix}.mp4"
                    if final_out.exists() and final_out.stat().st_size > 0:
                        print(f"[SKIP] whole video (hires exists): {final_out}")
                        skipped = len(self.get_defined_tasks(settings))
                        _emit_task_progress(skipped)
                        return skipped
            except Exception as ex:
                print(f"[WARN] early skip-check failed, continuing: {ex}")

            result = orig_orchestration(*args, **kwargs)
            if STOP_MARKER and os.path.exists(STOP_MARKER):
                stop_event.set()
                print("[STOP] stop marker detected after current clip. Stopping before next clip.")
            return result

        batch_processor._process_single_video_orchestration = MethodType(  # type: ignore[method-assign]
            _process_single_video_orchestration_skip_wrapper,
            batch_processor,
        )

    sidecar_folder = _default_sidecar_folder(INPUT_DEPTH_MAPS)
    sidecar_ext = ".fssidecar"

    conv_overrides = _read_convergence_overrides(str(args.auto_convergence_csv).strip())
    if conv_overrides:
        orig_get_video_specific_settings = batch_processor._get_video_specific_settings
        csv_policy = str(args.auto_convergence_csv_policy)

        def _get_video_specific_settings_csv_wrapper(self, *g_args, **g_kwargs):
            res = orig_get_video_specific_settings(*g_args, **g_kwargs)
            if not isinstance(res, dict) or res.get("error"):
                return res

            video_path = g_kwargs.get("video_path")
            if video_path is None and len(g_args) >= 1:
                video_path = g_args[0]
            depth_path = res.get("actual_depth_map_path")

            csv_conv = _lookup_convergence_override(
                conv_overrides,
                video_path=str(video_path) if video_path is not None else None,
                depth_path=str(depth_path) if depth_path else None,
            )
            if csv_conv is None:
                return res

            anchor_source = str(res.get("anchor_source", "GUI"))
            if csv_policy == "fill-missing" and anchor_source == "Sidecar":
                return res

            res["convergence_plane"] = float(csv_conv)
            res["anchor_source"] = "CSV"
            clip_name = Path(str(video_path)).stem if video_path else "unknown"
            print(f"[INFO] CSV convergence override: {clip_name} -> {float(csv_conv):.4f}")
            return res

        batch_processor._get_video_specific_settings = MethodType(  # type: ignore[method-assign]
            _get_video_specific_settings_csv_wrapper,
            batch_processor,
        )

    normalized_output_layout = _normalize_output_layout(OUTPUT_LAYOUT, fallback_dual=bool(DUAL_OUTPUT))

    settings = ProcessingSettings(
        input_source_clips=str(Path(INPUT_SOURCE_CLIPS).resolve()),
        input_depth_maps=str(Path(INPUT_DEPTH_MAPS).resolve()),
        output_splatted=str(_normalize_output_root(Path(OUTPUT_SPLATTED).resolve())),
        max_disp=float(MAX_DISP),
        process_length=int(PROCESS_LENGTH),
        enable_full_resolution=bool(ENABLE_FULL_RES),
        full_res_batch_size=int(FULL_RES_BATCH_SIZE),
        enable_low_resolution=bool(ENABLE_LOW_RES),
        low_res_width=int(LOW_RES_W),
        low_res_height=int(LOW_RES_H),
        low_res_batch_size=int(LOW_RES_BATCH_SIZE),
        dual_output=bool(normalized_output_layout == "dual"),
        output_layout=str(normalized_output_layout),
        zero_disparity_anchor=float(ZERO_DISPARITY_ANCHOR),
        enable_global_norm=bool(ENABLE_GLOBAL_NORM),
        match_depth_res=bool(MATCH_DEPTH_RES),
        move_to_finished=bool(MOVE_TO_FINISHED),
        output_crf=int(OUTPUT_CRF_FULL),
        output_crf_full=int(OUTPUT_CRF_FULL),
        output_crf_low=int(OUTPUT_CRF_LOW),
        depth_gamma=float(DEPTH_GAMMA),
        depth_dilate_size_x=float(DEPTH_DILATE_X),
        depth_dilate_size_y=float(DEPTH_DILATE_Y),
        depth_blur_size_x=float(DEPTH_BLUR_X),
        depth_blur_size_y=float(DEPTH_BLUR_Y),
        depth_dilate_left=float(DEPTH_DILATE_LEFT),
        depth_blur_left=float(DEPTH_BLUR_LEFT),
        depth_blur_left_mix=float(args.blur_balance),
        auto_convergence_mode=str(AUTO_CONVERGENCE_MODE),
        enable_sidecar_gamma=False,
        enable_sidecar_blur_dilate=False,
        multi_map=False,
        selected_depth_map="",
        sidecar_ext=sidecar_ext,
        sidecar_folder=sidecar_folder,
        stair_smooth_enabled=bool(SPLAT_STAIR_SMOOTH_ENABLED),
        stair_blur_kernel=int(SPLAT_BLUR_KERNEL),
        stair_edge_x_offset=int(SPLAT_STAIR_EDGE_X_OFFSET),
        stair_strip_px=int(SPLAT_STAIR_STRIP_PX),
        stair_strength=float(SPLAT_STAIR_STRENGTH),
        replace_mask_enabled=bool(REPLACE_MASK_ENABLED),
        replace_mask_dir=str(MASK_OUTPUT),
        replace_mask_scale=float(REPLACE_MASK_SCALE),
        replace_mask_min_px=int(REPLACE_MASK_MIN_PX),
        replace_mask_max_px=int(REPLACE_MASK_MAX_PX),
        replace_mask_gap_tol=int(REPLACE_MASK_GAP_TOL),
        replace_mask_codec=str(REPLACE_MASK_CODEC),
        replace_mask_draw_edge=bool(REPLACE_MASK_DRAW_EDGE),
    )

    try:
        setup_preview = batch_processor.setup_batch_processing(settings)
        task_defs = batch_processor.get_defined_tasks(settings)
        if not setup_preview.error:
            progress_tracker["total"] = max(0, len(setup_preview.input_videos) * len(task_defs))
            if progress_tracker["total"] > 0:
                print(f"[TOTAL] {progress_tracker['total']}")
    except Exception as ex:
        print(f"[WARN] failed to precompute task total: {ex}")

    print("[INFO] Starting splatting batch with settings:")
    for k in sorted(settings.__dict__.keys()):
        print(f"  - {k} = {getattr(settings, k)}")

    # Lightweight queue monitor to expose runner progress in CLI for pipeline GUI parsing.
    monitor_stop = threading.Event()

    def _progress_monitor() -> None:
        while not monitor_stop.is_set():
            try:
                msg = progress_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg == "finished":
                break
            if not isinstance(msg, tuple) or not msg:
                continue
            kind = msg[0]
            if kind == "total":
                try:
                    total = int(msg[1])
                    if total > 0:
                        progress_tracker["total"] = max(progress_tracker["total"], total)
                except Exception:
                    pass
            elif kind == "status":
                try:
                    status_line = str(msg[1]).strip()
                    if status_line:
                        print(f"[STAT] {status_line}")
                except Exception:
                    pass
            elif kind == "update_info":
                try:
                    info = msg[1] if len(msg) > 1 else {}
                    if isinstance(info, dict):
                        clip = str(info.get("filename", "")).strip()
                        if clip:
                            print(f"[CLIP] {clip}")
                except Exception:
                    pass

    monitor_thread = threading.Thread(target=_progress_monitor, daemon=True)
    monitor_thread.start()

    t0 = time.time()
    try:
        batch_processor.run_batch_process(settings)
    except Exception as e:
        print(f"[ERR ] batch failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
    finally:
        monitor_stop.set()
        try:
            monitor_thread.join(timeout=1.5)
        except Exception:
            pass
        dt = time.time() - t0
        print(f"[DONE] elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
