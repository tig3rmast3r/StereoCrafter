#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless-ish batch runner for Stereocrafter Splatting GUI (splatting_gui.py).

- Keeps SplatterGUI logic (sidecars, task loop, encoding) but runs without mainloop.
- Disables move-to-finished by default (can be changed in HARD-CODED SETTINGS below).
- Adds SKIP/retry behavior by monkey-patching BatchProcessor/RenderProcessor.

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
import numpy as np
import re
from pathlib import Path


def _normalize_output_root(p: Path) -> Path:
    # splatting_gui.py will create subfolders 'hires' and 'lowres' under output_splatted.
    # If output_splatted is accidentally set to .../hires or .../lowres, normalize to the parent.
    if p.name.lower() in ("hires", "lowres"):
        return p.parent
    return p



# -------------------------
# HARD-CODED SETTINGS
# -------------------------
SPLAT_GUI_PY = "./splatting_gui.py"  # path to your splatting GUI script

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

DUAL_OUTPUT = True  # False => _splatted4, True => _splatted2
ENABLE_GLOBAL_NORM = False
MATCH_DEPTH_RES = True

# Output encode CRF (separate hi/lo)
OUTPUT_CRF_FULL = 1
OUTPUT_CRF_LOW  = 23

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

# Optional in-memory pre-crop on source + depth before splatting.
# Useful to remove tiny top/bottom bars without creating new files.
PRE_CROP_TOP = 16
PRE_CROP_BOTTOM = 16
PRE_CROP_MULTIPLE_OF = 8



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
REPLACE_MASK_DRAW_EDGE = True       # must be True (removes ondulations)


def _parse_args():
    p = argparse.ArgumentParser(description="Batch runner for splatting_gui.py (Tk-based; use xvfb-run if headless).")
    p.add_argument("--gui_script", default=SPLAT_GUI_PY, help="Path to GUI script (splatting_gui.py).")
    p.add_argument("--input_source_clips", default=INPUT_SOURCE_CLIPS, help="Folder with source clip segments.")
    p.add_argument("--input_depth_maps", default=INPUT_DEPTH_MAPS, help="Folder with depth map videos.")
    p.add_argument("--output_splatted", default=OUTPUT_SPLATTED, help="Output folder root for splatted videos.")
    p.add_argument("--mask_output", default=MASK_OUTPUT, help="Output folder for exported clean masks.")
    p.add_argument("--full_res_batch_size", type=int, default=FULL_RES_BATCH_SIZE, help="Batch size for full-res processing.")
    p.add_argument(
        "--log-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose DEBUG logging for runner and splatting modules.",
    )
    p.add_argument(
        "--auto_convergence_mode",
        default=AUTO_CONVERGENCE_MODE,
        choices=["Off", "Manual", "Average", "Peak", "Hybrid", "MinBorders"],
        help="Auto-convergence mode to pass into processing settings.",
    )
    p.add_argument(
        "--sidecar-policy",
        default=SIDECAR_POLICY,
        choices=["keep", "warn", "prompt-delete", "delete-all"],
        help="How to handle existing sidecar files before processing.",
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
    p.add_argument(
        "--pre-crop-top",
        type=int,
        default=PRE_CROP_TOP,
        help="Rows to crop from top before splatting (applies to source+depth readers).",
    )
    p.add_argument(
        "--pre-crop-bottom",
        type=int,
        default=PRE_CROP_BOTTOM,
        help="Rows to crop from bottom before splatting (applies to source+depth readers).",
    )
    p.add_argument(
        "--pre-crop-multiple-of",
        type=int,
        default=PRE_CROP_MULTIPLE_OF,
        help="If >1, enforce cropped height to be divisible by this value by trimming extra rows.",
    )
    p.add_argument(
        "--disable-pre-crop",
        action="store_true",
        help="Disable pre-crop even if hard-coded/CLI crop rows are set.",
    )
    return p.parse_args()

def _import_module_from_path(py_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _compute_task_final_out(output_video_path_base: str, target_output_width: int, dual_output: bool) -> str:
    suffix = "_splatted2" if dual_output else "_splatted4"
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


class _NumpyBatch:
    """Minimal wrapper to keep Decord-like .asnumpy() API."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


class _VerticalCropReader:
    """Reader wrapper that crops top/bottom rows on get_batch()."""

    def __init__(self, inner, crop_top: int, crop_bottom: int):
        self._inner = inner
        self._crop_top = max(0, int(crop_top))
        self._crop_bottom = max(0, int(crop_bottom))

    def __len__(self) -> int:
        return len(self._inner)

    def seek(self, *args, **kwargs):
        if hasattr(self._inner, "seek"):
            return self._inner.seek(*args, **kwargs)
        return None

    def close(self):
        if hasattr(self._inner, "close"):
            return self._inner.close()
        return None

    def get_batch(self, indices):
        arr = self._inner.get_batch(indices).asnumpy()
        if arr.ndim < 3:
            return _NumpyBatch(arr)
        if self._crop_top <= 0 and self._crop_bottom <= 0:
            return _NumpyBatch(arr)

        h = int(arr.shape[1])
        top = min(self._crop_top, max(0, h - 1))
        max_bottom = max(0, h - top - 1)
        bottom = min(self._crop_bottom, max_bottom)
        end = h - bottom
        if end <= top:
            raise ValueError(
                f"Invalid crop window for height={h}: top={top}, bottom={bottom}"
            )
        return _NumpyBatch(arr[:, top:end, ...])

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _compute_vertical_crop(
    input_height: int,
    crop_top: int,
    crop_bottom: int,
    enforce_multiple_of: int,
) -> tuple[int, int, int]:
    """Compute effective top/bottom crop and resulting output height."""
    h = int(input_height)
    top = max(0, int(crop_top))
    bottom = max(0, int(crop_bottom))
    mul = int(enforce_multiple_of)

    if top + bottom >= h:
        raise ValueError(
            f"Pre-crop too aggressive for height={h}: top={top}, bottom={bottom}"
        )

    out_h = h - top - bottom
    if mul > 1 and out_h > 0:
        rem = out_h % mul
        if rem:
            # Keep framing centered by splitting extra trim between top/bottom.
            extra_top = rem // 2
            extra_bottom = rem - extra_top
            top += extra_top
            bottom += extra_bottom
            out_h = h - top - bottom
            if out_h <= 0:
                raise ValueError(
                    f"Pre-crop+alignment invalid for height={h} (mul={mul})."
                )
            if out_h % mul != 0:
                raise ValueError(
                    f"Failed to enforce multiple-of-{mul} (result={out_h})."
                )

    return top, bottom, out_h


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
    out = re.sub(r"_(\d+_)?splatted[24](?:_replace_mask)?$", "", out, flags=re.IGNORECASE)
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
    global PRE_CROP_TOP, PRE_CROP_BOTTOM, PRE_CROP_MULTIPLE_OF
    SPLAT_GUI_PY = args.gui_script
    INPUT_SOURCE_CLIPS = args.input_source_clips
    INPUT_DEPTH_MAPS = args.input_depth_maps
    OUTPUT_SPLATTED = args.output_splatted
    MASK_OUTPUT = args.mask_output
    FULL_RES_BATCH_SIZE = int(args.full_res_batch_size)
    PRE_CROP_TOP = max(0, int(args.pre_crop_top))
    PRE_CROP_BOTTOM = max(0, int(args.pre_crop_bottom))
    PRE_CROP_MULTIPLE_OF = max(0, int(args.pre_crop_multiple_of))
    if bool(args.disable_pre_crop):
        PRE_CROP_TOP = 0
        PRE_CROP_BOTTOM = 0
    pre_crop_enabled = bool(PRE_CROP_TOP > 0 or PRE_CROP_BOTTOM > 0)
    if pre_crop_enabled:
        print(
            f"[INFO] Pre-crop enabled: top={PRE_CROP_TOP}, bottom={PRE_CROP_BOTTOM}, "
            f"multiple_of={PRE_CROP_MULTIPLE_OF}"
        )
    else:
        print("[INFO] Pre-crop disabled.")

    gui_path = Path(SPLAT_GUI_PY).resolve()
    if not gui_path.exists():
        raise FileNotFoundError(f"Cannot find splatting_gui.py at: {gui_path}")

    mod = _import_module_from_path(gui_path)
    try:
        if hasattr(mod, "set_util_logger_level"):
            mod.set_util_logger_level(log_level)
    except Exception as ex:
        print(f"[WARN] failed applying util logger level: {ex}")

    # -------------------------
    # Apply module-level knobs (blur + replace mask)
    # -------------------------
    # These are read as globals inside the splatting script during processing.
    for k, v in {
        # Blur/Stair smoothing
        "SPLAT_STAIR_SMOOTH_ENABLED": bool(SPLAT_STAIR_SMOOTH_ENABLED),
        "SPLAT_BLUR_KERNEL": int(SPLAT_BLUR_KERNEL),
        "SPLAT_STAIR_EDGE_X_OFFSET": int(SPLAT_STAIR_EDGE_X_OFFSET),
        "SPLAT_STAIR_STRIP_PX": int(SPLAT_STAIR_STRIP_PX),
        "SPLAT_STAIR_STRENGTH": float(SPLAT_STAIR_STRENGTH),

        # Replace mask export
        "REPLACE_MASK_ENABLED": bool(REPLACE_MASK_ENABLED),
        "MASK_OUTPUT": str(MASK_OUTPUT),
        "REPLACE_MASK_SCALE": float(REPLACE_MASK_SCALE),
        "REPLACE_MASK_MIN_PX": int(REPLACE_MASK_MIN_PX),
        "REPLACE_MASK_MAX_PX": int(REPLACE_MASK_MAX_PX),
        "REPLACE_MASK_GAP_TOL": int(REPLACE_MASK_GAP_TOL),
        "REPLACE_MASK_DRAW_EDGE": bool(REPLACE_MASK_DRAW_EDGE),
    }.items():
        if hasattr(mod, k):
            setattr(mod, k, v)
        else:
            print(f"[WARN] splatting module does not define '{k}' (ignored)")
    if not hasattr(mod, "SplatterGUI"):
        raise AttributeError("splatting_gui.py must expose class SplatterGUI")

    # Instantiate GUI app but don't mainloop.
    # Withdraw to avoid popping a window (still requires a display).
    app = mod.SplatterGUI()
    if bool(args.log_verbose):
        try:
            app.debug_mode_var.set(True)
            app._configure_logging()
        except Exception as ex:
            print(f"[WARN] failed enabling GUI debug logging mode: {ex}")
    try:
        app.withdraw()
    except Exception:
        pass

    # Force move-to-finished OFF (belt & suspenders)
    try:
        app.move_to_finished_var.set(bool(MOVE_TO_FINISHED))
    except Exception:
        pass

    # Monkey-patch RenderProcessor.render_video for:
    # - per-task skip-if-output-exists
    # - retry on failure
    # - cleanup failed outputs
    from core.splatting.render_processor import RenderProcessor

    orig_render_video = RenderProcessor.render_video

    def render_video_wrapper(self, *args, **kwargs):
        args_list = list(args)

        output_video_path_base = _get_call_arg(args_list, kwargs, 4, "output_video_path_base")
        target_output_width = _get_call_arg(args_list, kwargs, 6, "target_output_width")
        dual_output_raw = _get_call_arg(args_list, kwargs, 9, "dual_output")
        dual_output = bool(DUAL_OUTPUT if dual_output_raw is None else dual_output_raw)
        replace_mask_enabled_raw = _get_call_arg(
            args_list, kwargs, 37, "replace_mask_enabled"
        )
        replace_mask_enabled = bool(
            REPLACE_MASK_ENABLED
            if replace_mask_enabled_raw is None
            else replace_mask_enabled_raw
        )
        replace_mask_dir_raw = _get_call_arg(args_list, kwargs, 38, "replace_mask_dir")
        replace_mask_dir = str(MASK_OUTPUT if replace_mask_dir_raw is None else replace_mask_dir_raw)

        if pre_crop_enabled:
            target_output_height = _get_call_arg(args_list, kwargs, 5, "target_output_height")
            src_reader = _get_call_arg(args_list, kwargs, 0, "input_video_reader")
            depth_reader = _get_call_arg(args_list, kwargs, 1, "depth_map_reader")

            if target_output_height is not None and src_reader is not None and depth_reader is not None:
                try:
                    eff_top, eff_bottom, cropped_h = _compute_vertical_crop(
                        input_height=int(target_output_height),
                        crop_top=int(PRE_CROP_TOP),
                        crop_bottom=int(PRE_CROP_BOTTOM),
                        enforce_multiple_of=int(PRE_CROP_MULTIPLE_OF),
                    )
                    if eff_top > 0 or eff_bottom > 0:
                        _set_call_arg(
                            args_list,
                            kwargs,
                            0,
                            "input_video_reader",
                            _VerticalCropReader(src_reader, eff_top, eff_bottom),
                        )
                        _set_call_arg(
                            args_list,
                            kwargs,
                            1,
                            "depth_map_reader",
                            _VerticalCropReader(depth_reader, eff_top, eff_bottom),
                        )
                        _set_call_arg(args_list, kwargs, 5, "target_output_height", int(cropped_h))
                        base_label = (
                            os.path.basename(str(output_video_path_base))
                            if output_video_path_base is not None
                            else "unknown"
                        )
                        print(
                            f"[INFO] Pre-crop {base_label}: "
                            f"H {int(target_output_height)} -> {int(cropped_h)} "
                            f"(top={eff_top}, bottom={eff_bottom})"
                        )
                except Exception as ex:
                    print(f"[WARN] pre-crop skipped (invalid params): {ex}")

        final_out = None
        replace_out = None
        try:
            if output_video_path_base is not None and target_output_width is not None:
                final_out = _compute_task_final_out(str(output_video_path_base), int(target_output_width), dual_output)
                replace_out = _compute_replace_mask_out(
                    final_out=final_out,
                    replace_mask_enabled=replace_mask_enabled,
                    replace_mask_dir=replace_mask_dir,
                )

                if SKIP_IF_OUTPUT_EXISTS and os.path.exists(final_out) and os.path.getsize(final_out) > 0:
                    print(f"[SKIP] task output exists: {final_out}")
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
        return False

    RenderProcessor.render_video = render_video_wrapper  # type: ignore

    # Monkey-patch BatchProcessor orchestration for early skip of whole video.
    if SKIP_IF_OUTPUT_EXISTS and hasattr(app, "batch_processor"):
        orig_orchestration = app.batch_processor._process_single_video_orchestration

        def _process_single_video_orchestration_skip_wrapper(*args, **kwargs):
            try:
                video_path = kwargs.get("video_path", None)
                settings = kwargs.get("settings", None)
                if video_path is None and len(args) >= 1:
                    video_path = args[0]
                if settings is None and len(args) >= 2:
                    settings = args[1]

                if video_path is not None and settings is not None and getattr(settings, "enable_full_resolution", False):
                    out_root = Path(getattr(settings, "output_splatted", OUTPUT_SPLATTED)).resolve()
                    base_name = Path(str(video_path)).stem
                    suffix = "_splatted2" if bool(getattr(settings, "dual_output", DUAL_OUTPUT)) else "_splatted4"
                    final_out = out_root / "hires" / f"{base_name}_{int(HIRES_SKIP_WIDTH)}{suffix}.mp4"
                    if final_out.exists() and final_out.stat().st_size > 0:
                        print(f"[SKIP] whole video (hires exists): {final_out}")
                        return len(app.batch_processor.get_defined_tasks(settings))
            except Exception as ex:
                print(f"[WARN] early skip-check failed, continuing: {ex}")

            return orig_orchestration(*args, **kwargs)

        app.batch_processor._process_single_video_orchestration = _process_single_video_orchestration_skip_wrapper  # type: ignore

    sidecar_folder = ""
    try:
        sidecar_folder = app._get_sidecar_base_folder()
    except Exception:
        sidecar_folder = str(Path(INPUT_DEPTH_MAPS).resolve())
    sidecar_ext = getattr(app, "APP_CONFIG_DEFAULTS", {}).get("SIDECAR_EXT", ".fssidecar")

    _handle_sidecar_policy(sidecar_folder=sidecar_folder, sidecar_ext=sidecar_ext, policy=str(args.sidecar_policy))

    conv_overrides = _read_convergence_overrides(str(args.auto_convergence_csv).strip())
    if conv_overrides and hasattr(app, "batch_processor"):
        orig_get_video_specific_settings = app.batch_processor._get_video_specific_settings
        csv_policy = str(args.auto_convergence_csv_policy)

        def _get_video_specific_settings_csv_wrapper(*g_args, **g_kwargs):
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

        app.batch_processor._get_video_specific_settings = _get_video_specific_settings_csv_wrapper  # type: ignore

    settings = mod.ProcessingSettings(
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
        dual_output=bool(DUAL_OUTPUT),
        zero_disparity_anchor=float(ZERO_DISPARITY_ANCHOR),
        enable_global_norm=bool(ENABLE_GLOBAL_NORM),
        match_depth_res=bool(MATCH_DEPTH_RES),
        move_to_finished=bool(MOVE_TO_FINISHED),
        output_crf=int(OUTPUT_CRF_FULL),  # legacy
        output_crf_full=int(OUTPUT_CRF_FULL),
        output_crf_low=int(OUTPUT_CRF_LOW),
        depth_gamma=float(DEPTH_GAMMA),
        depth_dilate_size_x=float(DEPTH_DILATE_X),
        depth_dilate_size_y=float(DEPTH_DILATE_Y),
        depth_blur_size_x=float(DEPTH_BLUR_X),
        depth_blur_size_y=float(DEPTH_BLUR_Y),
        depth_dilate_left=float(DEPTH_DILATE_LEFT),
        depth_blur_left=float(DEPTH_BLUR_LEFT),
        auto_convergence_mode=str(args.auto_convergence_mode),
        enable_sidecar_gamma=bool(ENABLE_SIDECAR_GAMMA),
        enable_sidecar_blur_dilate=bool(ENABLE_SIDECAR_BLUR_DILATE),
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
        replace_mask_draw_edge=bool(REPLACE_MASK_DRAW_EDGE),
    )

    print("[INFO] Starting splatting batch with settings:")
    for k in sorted(settings.__dict__.keys()):
        print(f"  - {k} = {getattr(settings, k)}")

    t0 = time.time()
    try:
        app._run_batch_process(settings)
    except Exception as e:
        print(f"[ERR ] batch failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
    finally:
        dt = time.time() - t0
        print(f"[DONE] elapsed={dt:.1f}s")
        try:
            app.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
