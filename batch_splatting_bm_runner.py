#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless-ish batch runner for Stereocrafter Splatting GUI (splatting_gui.py).

- Keeps SplatterGUI logic (sidecars, task loop, encoding) but runs without mainloop.
- Disables move-to-finished by default (can be changed in HARD-CODED SETTINGS below).
- Adds SKIP-if-output-exists behavior by monkey-patching depthSplatting().

Usage:
  python3 batch_splatting_bm_runner.py
"""

import os
import sys
import time
import traceback
import argparse
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
SPLAT_GUI_PY = "./splatting_bm_gui.py"  # path to your splatting GUI script

INPUT_SOURCE_CLIPS = "./work/seg/"
INPUT_DEPTH_MAPS   = "./work/depthmap/"
OUTPUT_SPLATTED    = "./work/splat/"
MASK_OUTPUT        = "./work/mask/"   # empty => same folder as main output

# Core splat params
MAX_DISP = 20.0
ZERO_DISPARITY_ANCHOR = 0.50
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

# Auto convergence mode: "Off" | "Average" | "Peak" (as in GUI)
AUTO_CONVERGENCE_MODE = "Off"

# Sidecar control toggles
ENABLE_SIDECAR_GAMMA = True
ENABLE_SIDECAR_BLUR_DILATE = True

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
REPLACE_MASK_DRAW_EDGE = True       # must be True (removes ondulations)


def _parse_args():
    p = argparse.ArgumentParser(description="Batch runner for splatting_bm_gui.py (Tk-based; use xvfb-run if headless).")
    p.add_argument("--gui_script", default=SPLAT_GUI_PY, help="Path to GUI script (splatting_bm_gui.py).")
    p.add_argument("--input_source_clips", default=INPUT_SOURCE_CLIPS, help="Folder with source clip segments.")
    p.add_argument("--input_depth_maps", default=INPUT_DEPTH_MAPS, help="Folder with depth map videos.")
    p.add_argument("--output_splatted", default=OUTPUT_SPLATTED, help="Output folder root for splatted videos.")
    p.add_argument("--mask_output", default=MASK_OUTPUT, help="Output folder for exported clean masks.")
    p.add_argument("--full_res_batch_size", type=int, default=FULL_RES_BATCH_SIZE, help="Batch size for full-res processing.")
    return p.parse_args()

def _import_module_from_path(py_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main():
    args = _parse_args()

    # Override from CLI
    global SPLAT_GUI_PY, INPUT_SOURCE_CLIPS, INPUT_DEPTH_MAPS, OUTPUT_SPLATTED, MASK_OUTPUT, FULL_RES_BATCH_SIZE
    SPLAT_GUI_PY = args.gui_script
    INPUT_SOURCE_CLIPS = args.input_source_clips
    INPUT_DEPTH_MAPS = args.input_depth_maps
    OUTPUT_SPLATTED = args.output_splatted
    MASK_OUTPUT = args.mask_output
    FULL_RES_BATCH_SIZE = int(args.full_res_batch_size)

    gui_path = Path(SPLAT_GUI_PY).resolve()
    if not gui_path.exists():
        raise FileNotFoundError(f"Cannot find splatting_gui.py at: {gui_path}")

    mod = _import_module_from_path(gui_path)

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
    try:
        app.withdraw()
    except Exception:
        pass

    # Force move-to-finished OFF (belt & suspenders)
    try:
        app.move_to_finished_var.set(bool(MOVE_TO_FINISHED))
    except Exception:
        pass

    # Monkey-patch depthSplatting to implement:
    # - skip-if-output-exists (optional)
    # - cleanup of corrupted leftover outputs on failure
    # - single retry on failure (configurable)
    orig_depthSplatting = app.depthSplatting

    def _compute_hires_final_out(output_video_path_base, target_output_width):
        base_dir = os.path.dirname(str(output_video_path_base))
        base_name = os.path.splitext(os.path.basename(str(output_video_path_base)))[0]

        # Ensure we are checking inside .../hires
        if os.path.basename(base_dir).lower() != "hires":
            hires_dir = os.path.join(base_dir, "hires")
        else:
            hires_dir = base_dir

        final_out = os.path.join(
            hires_dir,
            f"{base_name}_{int(target_output_width)}_splatted2.mp4"
        )
        return final_out

    def _compute_replace_mask_out(final_out: str):
        # Must mirror splatting_bm_gui.py naming:
        #   <basename_without_ext>_replace_mask.mkv
        if not bool(REPLACE_MASK_ENABLED):
            return None
        out_dir = str(MASK_OUTPUT).strip()
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

    def depthSplatting_wrapper(*args, **kwargs):
        # We only care about hires + splatted2 output (user constraint).
        output_video_path_base = kwargs.get("output_video_path_base", None)
        target_output_width = kwargs.get("target_output_width", None)

        if output_video_path_base is None and len(args) >= 6:
            output_video_path_base = args[5]
        if target_output_width is None and len(args) >= 8:
            target_output_width = args[7]

        final_out = None
        replace_out = None

        try:
            final_out = _compute_hires_final_out(output_video_path_base, target_output_width)
            replace_out = _compute_replace_mask_out(final_out)
            if SKIP_IF_OUTPUT_EXISTS and os.path.exists(final_out) and os.path.getsize(final_out) > 0:
                print(f"[SKIP] hires splatted2 exists: {final_out}")
                return True
        except Exception as ex:
            # If anything goes wrong, fall back to normal behavior.
            print(f"[WARN] skip/compute paths failed, continuing: {ex}")

        max_attempts = 1 + max(0, int(RETRY_ON_FAIL))
        last_exc = None

        for attempt in range(1, max_attempts + 1):
            ok = False
            try:
                ok = bool(orig_depthSplatting(*args, **kwargs))
            except Exception as ex:
                last_exc = ex
                ok = False
                print(f"[ERR ] depthSplatting raised: {ex}")

            if ok:
                return True

            # Failure path: cleanup corrupted leftovers
            if bool(CLEANUP_ON_FAIL):
                _safe_remove(final_out, "hires out")
                _safe_remove(replace_out, "replace mask")

            if attempt < max_attempts:
                print(f"[RETRY] encoding failed, retrying ({attempt}/{max_attempts - 1})...")
                continue

        if last_exc is not None:
            print(f"[ERR ] giving up after {max_attempts} attempt(s). Last exception: {last_exc}")
        else:
            print(f"[ERR ] giving up after {max_attempts} attempt(s).")

        return False

    app.depthSplatting = depthSplatting_wrapper  # type: ignore


    # Monkey-patch _process_single_video_tasks to skip the WHOLE chain early
    # (avoids global depth stats pre-pass when output already exists).
    if SKIP_IF_OUTPUT_EXISTS and hasattr(app, "_process_single_video_tasks"):
        orig_process = app._process_single_video_tasks

        def _process_single_video_tasks_skip_wrapper(*args, **kwargs):
            try:
                settings = kwargs.get("settings", None)
                if settings is None and len(args) >= 2:
                    settings = args[1]
                if not isinstance(settings, dict):
                    settings = {}
                out_root = Path(settings.get("output_splatted", OUTPUT_SPLATTED)).resolve()
                hires_dir = out_root / "hires"
                video_path = kwargs.get("video_path", None)
                if video_path is None and len(args) >= 1:
                    video_path = args[0]
                if video_path is None:
                    # Cannot determine current video path; just run original.
                    return orig_process(*args, **kwargs)
                base_name = Path(str(video_path)).stem
                final_out = hires_dir / f"{base_name}_{int(HIRES_SKIP_WIDTH)}_splatted2.mp4"
                if final_out.exists() and final_out.stat().st_size > 0:
                    print(f"[SKIP] whole video (hires splatted2 exists): {final_out}")
                    counter = kwargs.get("initial_overall_task_counter", None)
                    if counter is None and len(args) >= 3:
                        counter = args[2]
                    if counter is None:
                        # Fallback: return 0 to keep batch running
                        counter = 0
                    return (counter, None)
            except Exception as ex:
                print(f"[WARN] early skip-check failed, continuing: {ex}")

            return orig_process(*args, **kwargs)

        app._process_single_video_tasks = _process_single_video_tasks_skip_wrapper  # type: ignore


    settings = {
        "input_source_clips": str(Path(INPUT_SOURCE_CLIPS).resolve()),
        "input_depth_maps": str(Path(INPUT_DEPTH_MAPS).resolve()),
        "output_splatted": str(_normalize_output_root(Path(OUTPUT_SPLATTED).resolve())),
        "max_disp": float(MAX_DISP),
        "process_length": int(PROCESS_LENGTH),
        "enable_full_resolution": bool(ENABLE_FULL_RES),
        "full_res_batch_size": int(FULL_RES_BATCH_SIZE),
        "enable_low_resolution": bool(ENABLE_LOW_RES),
        "low_res_width": int(LOW_RES_W),
        "low_res_height": int(LOW_RES_H),
        "low_res_batch_size": int(LOW_RES_BATCH_SIZE),
        "dual_output": bool(DUAL_OUTPUT),
        "zero_disparity_anchor": float(ZERO_DISPARITY_ANCHOR),
        "enable_global_norm": bool(ENABLE_GLOBAL_NORM),
        "match_depth_res": bool(MATCH_DEPTH_RES),
        "move_to_finished": bool(MOVE_TO_FINISHED),
        "output_crf": int(OUTPUT_CRF_FULL),  # legacy
        "output_crf_full": int(OUTPUT_CRF_FULL),
        "output_crf_low": int(OUTPUT_CRF_LOW),
        "depth_gamma": float(DEPTH_GAMMA),
        "depth_dilate_size_x": float(DEPTH_DILATE_X),
        "depth_dilate_size_y": float(DEPTH_DILATE_Y),
        "depth_blur_size_x": float(DEPTH_BLUR_X),
        "depth_blur_size_y": float(DEPTH_BLUR_Y),
        "depth_dilate_left": float(DEPTH_DILATE_LEFT),
        "depth_blur_left": int(round(float(DEPTH_BLUR_LEFT))),
        "auto_convergence_mode": str(AUTO_CONVERGENCE_MODE),
        "enable_sidecar_gamma": bool(ENABLE_SIDECAR_GAMMA),
        "enable_sidecar_blur_dilate": bool(ENABLE_SIDECAR_BLUR_DILATE),
    }

    print("[INFO] Starting splatting batch with settings:")
    for k in sorted(settings.keys()):
        print(f"  - {k} = {settings[k]}")

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
