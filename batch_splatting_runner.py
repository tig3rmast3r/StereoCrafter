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
    p = argparse.ArgumentParser(description="Batch runner for splatting_gui.py (Tk-based; use xvfb-run if headless).")
    p.add_argument("--gui_script", default=SPLAT_GUI_PY, help="Path to GUI script (splatting_gui.py).")
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

    # Monkey-patch RenderProcessor.render_video for:
    # - per-task skip-if-output-exists
    # - retry on failure
    # - cleanup failed outputs
    from core.splatting.render_processor import RenderProcessor

    orig_render_video = RenderProcessor.render_video

    def render_video_wrapper(self, *args, **kwargs):
        output_video_path_base = kwargs.get("output_video_path_base", None)
        target_output_width = kwargs.get("target_output_width", None)
        dual_output = bool(kwargs.get("dual_output", DUAL_OUTPUT))
        replace_mask_enabled = bool(kwargs.get("replace_mask_enabled", REPLACE_MASK_ENABLED))
        replace_mask_dir = str(kwargs.get("replace_mask_dir", MASK_OUTPUT))

        if output_video_path_base is None and len(args) >= 5:
            output_video_path_base = args[4]
        if target_output_width is None and len(args) >= 7:
            target_output_width = args[6]

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
                ok = bool(orig_render_video(self, *args, **kwargs))
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
        auto_convergence_mode=str(AUTO_CONVERGENCE_MODE),
        enable_sidecar_gamma=bool(ENABLE_SIDECAR_GAMMA),
        enable_sidecar_blur_dilate=bool(ENABLE_SIDECAR_BLUR_DILATE),
        sidecar_ext=getattr(app, "APP_CONFIG_DEFAULTS", {}).get("SIDECAR_EXT", ".fssidecar"),
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
