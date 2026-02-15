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

import torch

# The GUI module contains the full inpainting implementation we want to reuse.
# Importing it is fine headless; we just must not create a real Tk window.

import inpainting_gui as igs



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
    else:
        print(f"[INFO] sharpness.csv disabled; using fixed steps={args.fixed_steps}")

    for idx, video_path in enumerate(videos, 1):
        base = os.path.basename(video_path)
        print(f"\n[{idx}/{len(videos)}] {base}")

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

            completed, hi_res_input_path = runner.process_single_video(
                pipeline=pipeline,
                input_video_path=video_path,
                save_dir=args.output_dir,
                frames_chunk=args.frames_chunk,
                overlap=args.overlap,
                tile_num=args.tile_num,
                vf=None,
                num_inference_steps=num_steps,
                stop_event=stop_event,
                update_info_callback=None,
                original_input_blend_strength=args.original_input_blend_strength,
                output_crf=args.output_crf,
                process_length=args.process_length,
            )

            if completed:
                print(f"[OK] wrote: {out_path}")
                if args.move_finished:
                    _move_to_subfolder(video_path, args.finished_subdir)
                    if hi_res_input_path and os.path.exists(hi_res_input_path):
                        _move_to_subfolder(hi_res_input_path, args.finished_subdir)
            else:
                print("[FAIL] processing returned incomplete")
                if args.move_failed:
                    _move_to_subfolder(video_path, args.failed_subdir)

        except torch.OutOfMemoryError as e:
            print(f"[OOM] {e}")
            if args.move_failed:
                _move_to_subfolder(video_path, args.failed_subdir)
            _safe_release_cuda()
            continue
        except Exception as e:
            print(f"[ERR] {type(e).__name__}: {e}")
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
