#!/usr/bin/env python3
"""Headless batch runner for StereoCrafter inpainting (no GUI / no display).

It reuses the GUI implementation's processing code (chunking, streaming encode, mask ops, tiling)
by instantiating a minimal subclass of `InpaintingGUI` **without** initializing Tk.

Important: run it from the StereoCrafter repo root (so ./weights/... resolves).
"""

import os
import sys
import glob
import shutil
import argparse
import csv
import threading
import gc
import subprocess
import json
import math

import torch

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# The GUI module contains the full inpainting implementation we want to reuse.
# Importing it is fine headless; we just must not create a real Tk window.

import inpainting_gui as igs

RESTART_EVERY = 15           # or from env/arg
PLANNED_RESTART_CODE = 99
class _Var:
    """Tiny stand-in for tkinter's StringVar/BooleanVar/IntVar."""

    def __init__(self, value):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class HeadlessInpainting(igs.InpaintingGUI):
    """Subclass InpaintingGUI but avoid initializing Tk / ThemedTk."""

    def __init__(
        self,
        output_folder: str,
        input_folder: str = "",
        hires_blend_folder: str = "",
        debug_mode: bool = False,
        enable_color_transfer: bool = True,
        enable_post_inpainting_blend: bool = False,
        mask_initial_threshold: float = 0.3,
        mask_morph_kernel_size: float = 0.0,
        mask_dilate_kernel_size: int = 5,
        mask_blur_kernel_size: int = 10,
    ):
        # DO NOT call super().__init__() (it would create a Tk window)
        self.output_folder_var = _Var(output_folder)
        self.input_folder_var = _Var(input_folder)
        self.hires_blend_folder_var = _Var(hires_blend_folder)

        self.debug_mode_var = _Var(bool(debug_mode))
        self.enable_color_transfer = _Var(bool(enable_color_transfer))
        self.enable_post_inpainting_blend = _Var(bool(enable_post_inpainting_blend))

        # GUI stores these as StringVar; processing code casts to float/int.
        self.mask_initial_threshold_var = _Var(str(mask_initial_threshold))
        self.mask_morph_kernel_size_var = _Var(str(mask_morph_kernel_size))
        self.mask_dilate_kernel_size_var = _Var(str(mask_dilate_kernel_size))
        self.mask_blur_kernel_size_var = _Var(str(mask_blur_kernel_size))

        # Some methods expect these exist
        self.stop_event = threading.Event()
        self.pipeline = None

    # ---- Tk compatibility shims ----
    def after(self, _ms, func=None, *args, **kwargs):
        """Tk schedules callbacks asynchronously; do the same to avoid recursion."""
        if func is None:
            return None
        import threading
        t = threading.Timer(max(0, _ms) / 1000.0, func, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        return t

    def update_status_label(self, _message: str):
        # GUI only; ignore
        return None

    def __getattr__(self, name):
        """Headless mode: never delegate to Tk's interpreter.

        - Auto-stub Tk-style variables used by the GUI (StringVar/BooleanVar/IntVar)
        - Provide no-op stubs for common Tk methods some codepaths may call
        """
        if name.endswith("_var"):
            v = _Var(None)
            setattr(self, name, v)
            return v

        if name in ("update", "update_idletasks", "winfo_exists", "winfo_width", "winfo_height", "quit", "destroy"):
            return lambda *a, **k: None

        raise AttributeError(name)

def _safe_release_cuda():
    """Try hard to free VRAM between files."""
    try:
        igs.release_cuda_memory()
    except Exception:
        pass
    gc.collect()


def _run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _ffprobe_nb_frames(path: str) -> int:
    """
    Return an accurate decoded frame count for the first video stream.
    This is used to detect truncated outputs (files that look valid but are incomplete).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nw=1:nk=1",
        path,
    ]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed rc={rc}: {err}")
    try:
        return int(out.splitlines()[0].strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe returned invalid nb_read_frames: {out!r}") from e


def _cleanup_outputs(out_path: str) -> None:
    if not out_path:
        return
    for p in (out_path, out_path + ".tmp", out_path + ".part", out_path + ".temp"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass



def _resume_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".resume_state.json")


def _load_resume_state(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_resume_state(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _clear_resume_state(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def _is_output_complete(input_path: str, output_path: str, process_length: int, tol_frames: int = 1) -> bool:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False
    try:
        in_frames = _ffprobe_nb_frames(input_path)
        out_frames = _ffprobe_nb_frames(output_path)
        expected = in_frames
        if process_length is not None:
            try:
                pl = int(process_length)
            except Exception:
                pl = -1
            if pl and pl > 0:
                expected = min(in_frames, pl)
        return out_frames >= max(0, expected - int(tol_frames))
    except Exception:
        return False


def _move_to_subfolder(path: str, subfolder_name: str) -> str:
    folder = os.path.join(os.path.dirname(path), subfolder_name)
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, os.path.basename(path))
    try:
        shutil.move(path, dst)
    except Exception:
        # If move fails, keep original
        return path
    return dst

def _load_sharpness_csv(csv_path: str):
    """Return mapping {basename -> sharpness_raw}. If csv missing, returns empty dict."""
    if not csv_path:
        return {}
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return {}
        out = {}
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                name = (row.get("file") or "").strip()
                if not name:
                    continue
                raw_s = (row.get("sharpness_raw") or "").strip()
                pct_s = (row.get("sharpness_pct") or "").strip()
                try:
                    raw = float(raw_s) if raw_s != "" else float(pct_s)
                except Exception:
                    continue
                out[name] = raw
        return out
    except Exception:
        return {}

def _load_chunk_csv(csv_path: str):
    """Return mapping {basename -> frames_chunk}. If csv missing or column missing, returns empty dict.

    Looks for any of these columns (first found wins per row):
      - frames_chunk
      - frame_chunk
      - chunk
      - chunk_size
    """
    if not csv_path:
        return {}
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return {}
        out = {}
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                name = (row.get("file") or "").strip()
                if not name:
                    continue
                # try multiple possible column names
                val = None
                for key in ("frames_chunk", "frame_chunk", "chunk", "chunk_size"):
                    s = (row.get(key) or "").strip()
                    if s != "":
                        val = s
                        break
                if val is None:
                    continue
                try:
                    c = int(float(val))
                except Exception:
                    continue
                if c > 0:
                    out[name] = c
        return out
    except Exception:
        return {}


def _get_video_wh(path: str):
    """Fast width/height probe using OpenCV. Returns (w,h) or (None,None)."""
    try:
        if cv2 is None:
            return (None, None)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return (None, None)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w > 0 and h > 0:
            return (w, h)
        return (None, None)
    except Exception:
        return (None, None)

DEFAULT_CHUNK_K = 3840 * 832 * 16  # reference: 1920x832 -> 16 frames_chunk


def _steps_from_sharpness(val: float) -> int:
    """
    Rule:
    raw <= 500 -> 5
    +1 step every additional 500
    max 12
    """
    try:
        v = float(val)
    except Exception:
        return 5

    if v <= 500:
        return 5

    steps = 5 + int(v // 500)
    if steps > 12:
        steps = 12
    return steps



def run_batch(args):
    os.makedirs(args.output_dir, exist_ok=True)

    processed_this_run = 0

    runner = HeadlessInpainting(
        output_folder=args.output_dir,
        input_folder=(args.input_dir or (os.path.dirname(args.input_video) if args.input_video else "")),
        hires_blend_folder=args.hires_blend_folder or "",
        debug_mode=args.debug,
        enable_color_transfer=not args.disable_color_transfer,
        enable_post_inpainting_blend=args.enable_post_inpainting_blend,
        mask_initial_threshold=args.mask_initial_threshold,
        mask_morph_kernel_size=args.mask_morph_kernel_size,
        mask_dilate_kernel_size=args.mask_dilate_kernel_size,
        mask_blur_kernel_size=args.mask_blur_kernel_size,
    )

    # Load pipeline once (same as GUI)
    pipeline = igs.load_inpainting_pipeline(
        pre_trained_path=r"./weights/stable-video-diffusion-img2vid-xt-1-1",
        unet_path=r"./weights/StereoCrafter",
        device="cuda",
        dtype=torch.float16,
        offload_type=args.offload_type,
    )
    runner.pipeline = pipeline

    # Build file list
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))

    if not videos:
        print("[ERR] no input videos found")
        return 2

    resume_path = _resume_state_path(args.output_dir)
    resume = _load_resume_state(resume_path)
    fast_resume_start = None
    if resume and resume.get("mode") == "planned_restart":
        last_ok_idx = resume.get("last_ok_idx")
        if isinstance(last_ok_idx, int) and 0 <= last_ok_idx < len(videos):
            fast_resume_start = max(0, last_ok_idx - 1)
            print(f"[RESUME] Fast resume enabled. Rechecking index {fast_resume_start + 1} then continuing.")
        else:
            _clear_resume_state(resume_path)
    elif resume:
        _clear_resume_state(resume_path)


    stop_event = threading.Event()

    # Optional: load sharpness.csv once (mapping basename -> sharpness_raw).
    sharpness_map = {}
    sharp_csv = ""
    if not args.no_sharpness_csv:
        sharp_base = args.sharpness_base
        if not sharp_base:
            if args.input_dir:
                sharp_base = args.input_dir
            elif args.input_video:
                sharp_base = os.path.dirname(args.input_video)
            else:
                sharp_base = os.getcwd()
        sharp_csv = os.path.join(os.path.abspath(sharp_base), "sharpness.csv")
        sharpness_map = _load_sharpness_csv(sharp_csv)
        print(f"[INFO] sharpness.csv: {sharp_csv} (rows={len(sharpness_map)})")
        chunk_map = _load_chunk_csv(sharp_csv)
        if chunk_map:
            print(f"[INFO] per-file frames_chunk overrides: {len(chunk_map)}")
        else:
            print("[INFO] no per-file frames_chunk overrides found in sharpness.csv")
    else:
        print(f"[INFO] sharpness.csv disabled; using fixed steps={args.fixed_steps}")
        chunk_map = {}

    for idx, video_path in enumerate(videos, 1):
        i = idx - 1
        if fast_resume_start is not None and i < fast_resume_start:
            continue
        base = os.path.basename(video_path)
        print(f"\n[{idx}/{len(videos)}] {base}")

        out_path = ""
        hi_res_input_path = None

        try:
            # Ensure GUI vars are consistent for hi-res matching safety checks
            runner.input_folder_var.set(args.input_dir or os.path.dirname(video_path))
            
            # If hires blending folder is empty, force-disable hi-res matching by setting it equal to input folder.
            # This avoids accidental globbing in CWD and keeps output naming stable (also makes --skip_existing reliable).
            if not args.hires_blend_folder:
                runner.hires_blend_folder_var.set(runner.input_folder_var.get())
            else:
                runner.hires_blend_folder_var.set(args.hires_blend_folder)
            

            # Determine expected output name like GUI would, to support skip.
            name_wo_ext = os.path.splitext(base)[0]
            is_dual_input = name_wo_ext.endswith("_splatted2")
            out_path, _hires = runner._setup_video_info_and_hires(video_path, args.output_dir, is_dual_input)

            if args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                # Fast skip (legacy behavior). Only the resume-boundary file is strictly validated.
                if fast_resume_start is not None and i == fast_resume_start:
                    if _is_output_complete(video_path, out_path, args.process_length):
                        print(f"[SKIP] exists: {out_path}")
                        fast_resume_start = None
                        _clear_resume_state(resume_path)
                        continue
                    print(f"[WARN] existing output looks incomplete, deleting: {out_path}")
                    _cleanup_outputs(out_path)
                else:
                    print(f"[SKIP] exists: {out_path}")
                    continue

                        # Determine inference steps:
            # - If sharpness.csv is enabled and has a row for this basename, derive steps from it.
            # - Otherwise, use --fixed_steps.
            sharp_val = sharpness_map.get(base) if sharpness_map else None
            if sharp_val is None:
                num_steps = int(args.fixed_steps)
                print(f"[INFO] steps={num_steps} (fixed)")
            else:
                num_steps = _steps_from_sharpness(sharp_val)
                print(f"[INFO] steps={num_steps} (sharp_raw={sharp_val:.2f})")

            # frames_chunk selection:
            # - If --no_dynamic_chunk is NOT set, compute frames_chunk from frame area using chunk_k:
            #     frames_chunk ~= chunk_k / (W*H)
            #   Reference default: 1920x832 -> 24 frames_chunk  (chunk_k = 1920*832*24)
            # - If sharpness.csv provides a per-file override column, it wins.
            frames_chunk = int(args.frames_chunk)

            if not args.no_dynamic_chunk:
                vw, vh = _get_video_wh(video_path)
                if vw and vh:
                    dyn = int(round(float(args.chunk_k) / float(vw * vh)))
                    if dyn < 1:
                        dyn = 1
                    # clamp
                    dyn = max(int(args.chunk_min), min(int(args.chunk_max), dyn))
                    frames_chunk = dyn
                    print(f"[INFO] frames_chunk={frames_chunk} (dynamic from {vw}x{vh}, chunk_k={int(args.chunk_k)})")
                else:
                    print("[WARN] dynamic frames_chunk enabled but failed to probe video size; using fixed frames_chunk")

            # Per-file override (from sharpness.csv columns, if present).
            if chunk_map and base in chunk_map:
                frames_chunk = int(chunk_map[base])
                # clamp even on override
                frames_chunk = max(int(args.chunk_min), min(int(args.chunk_max), frames_chunk))
                print(f"[INFO] frames_chunk={frames_chunk} (per-file override)")

            # Keep overlap valid: must be < frames_chunk (otherwise chunking can't progress).
            overlap = int(args.overlap)
            if frames_chunk <= 0:
                frames_chunk = int(args.frames_chunk)
            if overlap < 0:
                overlap = 0
            if overlap >= frames_chunk:
                new_overlap = max(0, frames_chunk - 1)
                print(f"[WARN] overlap={overlap} >= frames_chunk={frames_chunk}; clamping overlap -> {new_overlap}")
                overlap = new_overlap

            completed, hi_res_input_path = runner.process_single_video(
                pipeline=pipeline,
                input_video_path=video_path,
                save_dir=args.output_dir,
                frames_chunk=frames_chunk,
                overlap=overlap,
                tile_num=args.tile_num,
                vf=None,
                num_inference_steps=num_steps,
                stop_event=stop_event,
                update_info_callback=None,
                original_input_blend_strength=args.original_input_blend_strength,
                output_crf=args.output_crf,
                process_length=args.process_length,
            )


            if completed and _is_output_complete(video_path, out_path, args.process_length):
                print(f"[OK] wrote: {out_path}")
                processed_this_run += 1
                if fast_resume_start is not None and i >= fast_resume_start:
                    fast_resume_start = None
                    _clear_resume_state(resume_path)
                if RESTART_EVERY > 0 and processed_this_run >= RESTART_EVERY:
                    print(f"[PLANNED RESTART] processed_this_run={processed_this_run}, exiting {PLANNED_RESTART_CODE}")
                    _save_resume_state(resume_path, {
                        "mode": "planned_restart",
                        "last_ok_idx": i,
                        "last_ok_input": video_path,
                        "last_ok_output": out_path,
                    })
                    sys.exit(PLANNED_RESTART_CODE)
                if args.move_finished:
                    _move_to_subfolder(video_path, args.finished_subdir)
                    if hi_res_input_path and os.path.exists(hi_res_input_path):
                        _move_to_subfolder(hi_res_input_path, args.finished_subdir)
            else:
                if completed:
                    print(f"[FAIL] output incomplete, deleting: {out_path}")
                else:
                    print("[FAIL] processing returned incomplete")
                _cleanup_outputs(out_path)
                if args.move_failed:
                    _move_to_subfolder(video_path, args.failed_subdir)

        except torch.OutOfMemoryError as e:
            print(f"[OOM] {e}")
            _cleanup_outputs(out_path)
            if args.move_failed:
                _move_to_subfolder(video_path, args.failed_subdir)
            _safe_release_cuda()
            continue
        except Exception as e:
            print(f"[ERR] {type(e).__name__}: {e}")
            _cleanup_outputs(out_path)
            if args.move_failed:
                _move_to_subfolder(video_path, args.failed_subdir)
            _safe_release_cuda()
            continue
        finally:
            # keep VRAM stable between files
            _safe_release_cuda()

    return 0


def main():
    p = argparse.ArgumentParser(description="StereoCrafter inpainting headless batch runner")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", type=str, help="Folder containing input videos")
    src.add_argument("--input_video", type=str, help="Single input video path")

    p.add_argument("--output_dir", type=str, required=True, help="Output folder")
    p.add_argument("--glob", type=str, default="*.mp4", help="Glob pattern when using --input_dir")
    p.add_argument("--sharpness_base", type=str, default="", help="Base folder containing sharpness.csv (defaults to input folder)")
    p.add_argument("--no_sharpness_csv", action="store_true",
                   help="Ignore sharpness.csv and use --fixed_steps for all files")
    p.add_argument("--fixed_steps", type=int, default=8,
                   help="Fallback steps when sharpness.csv is missing or ignored")
    p.add_argument("--tile_num", type=int, default=2)
    p.add_argument("--frames_chunk", type=int, default=50)
    p.add_argument("--no_dynamic_chunk", action="store_true",
                   help="Disable dynamic frames_chunk computation; always use --frames_chunk (unless CSV override exists)")
    p.add_argument("--chunk_k", type=float, default=float(DEFAULT_CHUNK_K),
                   help="Constant for dynamic frames_chunk: frames_chunk ~= chunk_k/(W*H). Default based on 1920x832->24.")
    p.add_argument("--chunk_min", type=int, default=20, help="Minimum frames_chunk when dynamic/override is used")
    p.add_argument("--chunk_max", type=int, default=500, help="Maximum frames_chunk when dynamic/override is used")
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--original_input_blend_strength", type=float, default=0.0)
    p.add_argument("--output_crf", type=int, default=1)
    p.add_argument("--process_length", type=int, default=-1)

    p.add_argument("--offload_type", type=str, default="model", choices=["none", "model", "sequential"],
                   help="Matches GUI offload_type")

    p.add_argument("--hires_blend_folder", type=str, default="", help="Optional hires folder (same as GUI)")

    p.add_argument("--mask_initial_threshold", type=float, default=0.3)
    p.add_argument("--mask_morph_kernel_size", type=float, default=0.0)
    p.add_argument("--mask_dilate_kernel_size", type=int, default=5)
    p.add_argument("--mask_blur_kernel_size", type=int, default=10)

    p.add_argument("--enable_post_inpainting_blend", action="store_true")
    p.add_argument("--disable_color_transfer", action="store_true")

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--move_failed", action="store_true")
    p.add_argument("--move_finished", action="store_true")
    p.add_argument("--failed_subdir", type=str, default="failed")
    p.add_argument("--finished_subdir", type=str, default="finished")

    p.add_argument("--debug", action="store_true", help="Enable debug image saving (uses GUI code)")

    args = p.parse_args()

    # Normalize paths
    if args.input_video:
        args.input_video = os.path.abspath(args.input_video)
    if args.input_dir:
        args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
