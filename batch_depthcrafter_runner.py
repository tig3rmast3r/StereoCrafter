#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner (warm pipeline) + preprocess/postprocess for DepthCrafter depth-only.

Per input video:
  1) Preprocess with ffmpeg:
       - scale content to half of the input (WxH/2)
       - pad up to multiples of 64 ONLY when needed
     Output: temp_pre.mp4 (yuv420p bt709)

  2) Run DepthCrafter on temp_pre.mp4 at fixed 1024x576.

  3) Postprocess depth video with ffmpeg:
       - crop center 960x540 (remove pad)
       - OPTIONAL upscale 2x to original WxH with nearest (pixel-perfect 2x)
     Output: final depth mp4 in output_dir (yuv420p bt709)

Features:
  - Warm pipeline (keeps model loaded)
  - Skip if final output exists
  - Optional OOM retries (can be disabled)
  - Temp files cleaned (unless --keep_temps)
  - IMPORTANT: After ANY failed file, we HARD-RESET the pipeline (OOM can poison CUDA state)
               to avoid immediate cascade failures on the next file.

Requires ffmpeg and ffprobe in PATH.
Tries NVENC first; falls back to libx264 if NVENC fails.
"""
import gc
import importlib.util
import shlex
import subprocess
import time
import traceback
from pathlib import Path


def _load_worker_module(worker_script: str):
    worker_path = Path(worker_script).resolve()
    if not worker_path.exists():
        raise FileNotFoundError(f"worker_script not found: {worker_path}")
    spec = importlib.util.spec_from_file_location("depth_worker", str(worker_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {worker_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg)


def _cuda_cleanup():
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _default_out_name(in_file: Path, out_ext: str = ".mp4", suffix: str = "_depth") -> str:
    return f"{in_file.stem}{suffix}{out_ext}"


def _run(cmd, log_prefix="[FFMPEG]", check=True):
    print(f"{log_prefix} " + " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def _ffprobe_wh_fps(path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"ffprobe unexpected output: {p.stdout}")
    w = int(lines[0]); h = int(lines[1]); fps_str = lines[2]
    return w, h, fps_str


def _try_nvenc_encode(nvenc_cmd, x264_cmd):
    try:
        p = _run(nvenc_cmd, log_prefix="[FFMPEG-NVENC]", check=True)
        return True, p
    except subprocess.CalledProcessError as e:
        print("[WARN] NVENC encode failed, fallback to x264.")
        if e.stderr:
            tail = e.stderr.splitlines()[-6:]
            print("[WARN] NVENC stderr tail:")
            for line in tail:
                print("   ", line)
        p2 = _run(x264_cmd, log_prefix="[FFMPEG-X264]", check=True)
        return False, p2


def _set_arg_value(cmd: list[str], key: str, value: str) -> None:
    try:
        idx = cmd.index(key)
        if idx + 1 < len(cmd):
            cmd[idx + 1] = str(value)
    except ValueError:
        pass


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _preprocess_video(
    src: Path,
    dst: Path,
    content_w: int,
    content_h: int,
    pad_w: int,
    pad_h: int,
    pad_x: int,
    pad_y: int,
    src_crop_top: int = 0,
    src_crop_bottom: int = 0,
    ffmpeg_codec: str = "",
    ffmpeg_crf: int = -1,
    ffmpeg_preset: str = "",
    ffmpeg_pix_fmt: str = "",
    ffmpeg_extra_args: str = "",
):
    # Scale always to content_w x content_h, then pad ONLY if needed.
    # Important: pad placement is explicit (pad_x/pad_y) so caller can anchor content.
    pix_fmt = (ffmpeg_pix_fmt or "yuv420p").strip()
    crop_top = max(0, int(src_crop_top))
    crop_bottom = max(0, int(src_crop_bottom))
    vf_parts = []
    if crop_top > 0 or crop_bottom > 0:
        vf_parts.append(f"crop=iw:ih-{crop_top + crop_bottom}:0:{crop_top}")
    vf_parts.append(f"scale={content_w}:{content_h}:flags=lanczos")
    if pad_w != content_w or pad_h != content_h:
        vf_parts.append(f"pad={pad_w}:{pad_h}:{int(pad_x)}:{int(pad_y)}:black")
    vf_parts.append(f"format={pix_fmt}")
    vf = ",".join(vf_parts)

    nvenc = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-an",
        "-vf", vf,
        "-c:v", "h264_nvenc",
        "-preset", "medium",
        "-rc", "constqp",
        "-qp", "0",                # lossless-ish preprocess
        "-profile:v", "main",
        "-pix_fmt", pix_fmt,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+write_colr",
        str(dst),
    ]

    x264 = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-an",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "0",
        "-pix_fmt", pix_fmt,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+write_colr",
        str(dst),
    ]

    preset = (ffmpeg_preset or "").strip()
    if preset:
        _set_arg_value(nvenc, "-preset", preset)
        _set_arg_value(x264, "-preset", preset)
    if int(ffmpeg_crf) >= 0:
        _set_arg_value(nvenc, "-qp", str(int(ffmpeg_crf)))
        _set_arg_value(x264, "-crf", str(int(ffmpeg_crf)))

    extra = (ffmpeg_extra_args or "").strip()
    if extra:
        extra_tokens = shlex.split(extra)
        nvenc = nvenc[:-1] + extra_tokens + [nvenc[-1]]
        x264 = x264[:-1] + extra_tokens + [x264[-1]]

    codec = (ffmpeg_codec or "").strip().lower()
    if codec:
        if "nvenc" in codec:
            _set_arg_value(nvenc, "-c:v", ffmpeg_codec.strip())
            _run(nvenc, log_prefix="[FFMPEG-NVENC]", check=True)
            return
        if codec.startswith("libx"):
            _set_arg_value(x264, "-c:v", ffmpeg_codec.strip())
            _run(x264, log_prefix="[FFMPEG-X264]", check=True)
            return
        print(f"[WARN] unsupported ffmpeg_codec override for preprocess: {ffmpeg_codec} (using auto fallback)")

    _try_nvenc_encode(nvenc, x264)


def _postprocess_depth(
    src_depth: Path,
    dst_final: Path,
    crop_w: int,
    crop_h: int,
    crop_x: int,
    crop_y: int,
    out_w: int,
    out_h: int,
    recenter_w: int = 0,
    recenter_h: int = 0,
    recenter_pad_top: int = 0,
    final_upscale: bool = True,
    padded: bool = True,
    ffmpeg_codec: str = "",
    ffmpeg_crf: int = -1,
    ffmpeg_preset: str = "",
    ffmpeg_pix_fmt: str = "",
    ffmpeg_extra_args: str = "",
):
    # If padded is True, remove pad using explicit crop offsets before optional upscale.
    pix_fmt = (ffmpeg_pix_fmt or "yuv420p").strip()
    vf_parts = []
    if padded:
        vf_parts.append(f"crop={crop_w}:{crop_h}:{int(crop_x)}:{int(crop_y)}")
    if final_upscale:
        vf_parts.append(f"scale={out_w}:{out_h}:flags=neighbor")
    if int(recenter_w) > 0 and int(recenter_h) > 0:
        pad_top = max(0, int(recenter_pad_top))
        vf_parts.append(f"pad={int(recenter_w)}:{int(recenter_h)}:0:{pad_top}:black")
    vf_parts.append(f"format={pix_fmt}")
    vf = ",".join(vf_parts)

    nvenc = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_depth),
        "-an",
        "-vf", vf,
        "-c:v", "h264_nvenc",
        "-preset", "medium",
        "-rc", "constqp",
        "-qp", "0",                # depthmap: preserve max precision
        "-profile:v", "main",
        "-pix_fmt", pix_fmt,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+write_colr",
        str(dst_final),
    ]

    x264 = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_depth),
        "-an",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "0",
        "-pix_fmt", pix_fmt,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+write_colr",
        str(dst_final),
    ]

    preset = (ffmpeg_preset or "").strip()
    if preset:
        _set_arg_value(nvenc, "-preset", preset)
        _set_arg_value(x264, "-preset", preset)
    if int(ffmpeg_crf) >= 0:
        _set_arg_value(nvenc, "-qp", str(int(ffmpeg_crf)))
        _set_arg_value(x264, "-crf", str(int(ffmpeg_crf)))

    extra = (ffmpeg_extra_args or "").strip()
    if extra:
        extra_tokens = shlex.split(extra)
        nvenc = nvenc[:-1] + extra_tokens + [nvenc[-1]]
        x264 = x264[:-1] + extra_tokens + [x264[-1]]

    codec = (ffmpeg_codec or "").strip().lower()
    if codec:
        if "nvenc" in codec:
            _set_arg_value(nvenc, "-c:v", ffmpeg_codec.strip())
            _run(nvenc, log_prefix="[FFMPEG-NVENC]", check=True)
            return
        if codec.startswith("libx"):
            _set_arg_value(x264, "-c:v", ffmpeg_codec.strip())
            _run(x264, log_prefix="[FFMPEG-X264]", check=True)
            return
        print(f"[WARN] unsupported ffmpeg_codec override for postprocess: {ffmpeg_codec} (using auto fallback)")

    _try_nvenc_encode(nvenc, x264)


def run(
    worker_script: str = "./depthcrafter_nogui_batch.py",
    input_dir: str = ".",
    output_dir: str = "./out",
    glob: str = "*.mp4",
    out_ext: str = ".mp4",
    suffix: str = "_depth",
    # DepthCrafter knobs
    guidance_scale: float = 1.0,
    inference_steps: int = 5,
    window_size: int = 64,
    overlap: int = 16,
    seed: int = 42,
    cpu_offload_mode: str = "model",
    decode_chunk_size: int = 8,
    process_length: int = -1,
    target_fps: float = -1.0,
    max_res: int = 1920,
    far_black: bool = True,
    crf: int = 0,
    preset: str = "medium",
    debug_mem: bool = False,
    # Pre/post sizes (used only if auto_sizes=False)
    auto_sizes: bool = True,
    content_w: int = 960,
    content_h: int = 540,
    pad_w: int = 1024,
    pad_h: int = 576,
    # Policy
    retry_sequential: bool = True,
    retry_decode_chunk_size_1: bool = True,
    remove_partial_output: bool = True,
    restart_every: int = 0,
    keep_temps: bool = False,
    # NEW: Final upscale policy (default keeps current behavior)
    final_upscale: bool = True,
    # Pad anchor policy
    pad_align_bottom: bool = True,
    # Strip initial scene pad (top/bottom) before DepthCrafter, then re-center at output.
    scene_strip_pad_top: int = 0,
    scene_strip_pad_bottom: int = 0,
    # Optional ffmpeg overrides applied to preprocess + worker output + postprocess.
    # Keep empty / -1 to preserve existing behavior.
    ffmpeg_codec: str = "",
    ffmpeg_crf: int = -1,
    ffmpeg_preset: str = "",
    ffmpeg_pix_fmt: str = "",
    ffmpeg_extra_args: str = "",
):
    input_dir_p = Path(input_dir).resolve()
    output_dir_p = Path(output_dir).resolve()
    _ensure_dir(output_dir_p)

    temp_dir = output_dir_p / ".tmp_depthcrafter"
    _ensure_dir(temp_dir)

    mod = _load_worker_module(worker_script)
    if not hasattr(mod, "DepthCrafterDepthOnly"):
        raise AttributeError("worker_script must expose class DepthCrafterDepthOnly")

    # weights relative to StereoCrafter root
    unet_path = "./weights/DepthCrafter"
    pre_trained_path = "./weights/stable-video-diffusion-img2vid-xt-1-1"

    def make_runner(mode: str):
        r = mod.DepthCrafterDepthOnly(
            unet_path=unet_path,
            pre_trained_path=pre_trained_path,
            cpu_offload_mode=mode,
        )
        setattr(r, "_batch_mode", mode)
        return r

    runner = make_runner(cpu_offload_mode)

    def hard_reset_runner(mode: str):
        """Hard reset pipeline after a failure (OOM can poison CUDA state)."""
        nonlocal runner
        try:
            del runner
        except Exception:
            pass
        _cuda_cleanup()
        time.sleep(1.0)
        runner = make_runner(mode)

    files = sorted(input_dir_p.glob(glob))
    if not files:
        print(f"[INFO] No inputs matched: {input_dir_p}/{glob}")
        return

    total = len(files)
    ok = skipped = failed = 0
    processed = 0

    print(f"[INFO] Inputs: {total} | input_dir={input_dir_p} | output_dir={output_dir_p}")
    print(f"[INFO] Worker: {Path(worker_script).resolve()}")
    print(f"[INFO] Sizes: auto_sizes={auto_sizes} | fixed_pad={pad_w}x{pad_h} fixed_content={content_w}x{content_h}")
    print(f"[INFO] Params: offload={cpu_offload_mode} decode_chunk_size={decode_chunk_size} window={window_size} overlap={overlap} steps={inference_steps} gs={guidance_scale}")
    print(f"[INFO] final_upscale={final_upscale}")
    print(
        f"[INFO] scene_strip_pad: top={max(0, int(scene_strip_pad_top))} "
        f"bottom={max(0, int(scene_strip_pad_bottom))}"
    )
    print(
        "[INFO] ffmpeg overrides: "
        f"codec={ffmpeg_codec or 'default'} "
        f"crf={ffmpeg_crf if int(ffmpeg_crf) >= 0 else 'default'} "
        f"preset={ffmpeg_preset or 'default'} "
        f"pix_fmt={ffmpeg_pix_fmt or 'default'} "
        f"extra={'set' if str(ffmpeg_extra_args).strip() else 'none'}"
    )

    for i, in_path in enumerate(files, 1):
        processed += 1

        if restart_every and processed > 1 and (processed - 1) % int(restart_every) == 0:
            print(f"[INFO] Restarting pipeline (restart_every={restart_every}) ...")
            hard_reset_runner(cpu_offload_mode)

        out_name = _default_out_name(in_path, out_ext=out_ext, suffix=suffix)
        out_final = output_dir_p / out_name

        if out_final.exists() and out_final.stat().st_size > 0:
            skipped += 1
            print(f"[SKIP] {i}/{total} {in_path.name} -> {out_final.name} (exists)")
            continue

        # Probe original size
        try:
            orig_w, orig_h, _fps_str = _ffprobe_wh_fps(in_path)
        except Exception as pe:
            print(f"[ERR ] ffprobe failed for {in_path}: {pe}")
            failed += 1
            hard_reset_runner(cpu_offload_mode)
            continue
        # Scene pad stripping: remove initial top/bottom pad before DepthCrafter,
        # then re-center after postprocess to preserve original geometry.
        strip_top = max(0, int(scene_strip_pad_top))
        strip_bottom = max(0, int(scene_strip_pad_bottom))
        max_strip = max(0, int(orig_h) - 2)
        strip_total = strip_top + strip_bottom
        if strip_total > max_strip:
            strip_top, strip_bottom = _split_by_ratio(max_strip, strip_top, strip_bottom)
            strip_total = strip_top + strip_bottom
        core_h = int(orig_h) - strip_total
        if core_h < 2:
            strip_top = 0
            strip_bottom = 0
            strip_total = 0
            core_h = int(orig_h)

        # Compute per-file working sizes on stripped content.
        if auto_sizes:
            cw = int(orig_w) // 2
            ch = max(1, int(core_h) // 2)
            pw = _round_up(cw, 64)
            ph = _round_up(ch, 64)
        else:
            cw, ch, pw, ph = int(content_w), int(content_h), int(pad_w), int(pad_h)

        padded = (pw != cw) or (ph != ch)
        pad_x = max(0, (pw - cw) // 2)
        if bool(pad_align_bottom):
            # Keep content aligned to the bottom of padded canvas to avoid synthetic bottom bar artifacts.
            pad_y = max(0, ph - ch)
        else:
            pad_y = max(0, (ph - ch) // 2)

        if bool(final_upscale):
            post_scale_w = int(orig_w)
            post_scale_h = int(core_h)
            recenter_w = int(orig_w)
            recenter_h = int(orig_h)
        else:
            post_scale_w = int(cw)
            post_scale_h = int(ch)
            recenter_w = max(1, int(orig_w) // 2)
            recenter_h = max(1, int(orig_h) // 2)

        recenter_pad_total = max(0, int(recenter_h) - int(post_scale_h))
        recenter_pad_top, _recenter_pad_bottom = _split_by_ratio(
            recenter_pad_total, strip_top, strip_bottom
        )
        recenter_enabled = recenter_pad_total > 0

        stem = in_path.stem
        tmp_pre = temp_dir / f"{stem}__pre_{pw}x{ph}.mp4"
        tmp_depth = temp_dir / f"{stem}__depth_{pw}x{ph}.mp4"

        for p in (tmp_pre, tmp_depth):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        print(
            f"[RUN ] {i}/{total} {in_path.name} -> {out_final.name} "
            f"(orig {orig_w}x{orig_h} | half {cw}x{ch} | pad {pw}x{ph} "
            f"| strip_tb={strip_top},{strip_bottom} | pad_xy={pad_x},{pad_y} "
            f"| recenter={'yes' if recenter_enabled else 'no'} | padded={padded})"
        )
        t0 = time.perf_counter()

        def attempt(mode: str, dcs: int):
            nonlocal runner
            if mode != getattr(runner, "_batch_mode", mode):
                hard_reset_runner(mode)

            _preprocess_video(
                in_path,
                tmp_pre,
                cw,
                ch,
                pw,
                ph,
                pad_x,
                pad_y,
                src_crop_top=strip_top,
                src_crop_bottom=strip_bottom,
                ffmpeg_codec=ffmpeg_codec,
                ffmpeg_crf=int(ffmpeg_crf),
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_pix_fmt=ffmpeg_pix_fmt,
                ffmpeg_extra_args=ffmpeg_extra_args,
            )

            runner.infer_to_gray_video(
                input_video_path=str(tmp_pre),
                output_video_path=str(tmp_depth),
                guidance_scale=float(guidance_scale),
                inference_steps=int(inference_steps),
                target_width=int(pw),
                target_height=int(ph),
                window_size=int(window_size),
                overlap=int(overlap),
                seed=int(seed),
                cpu_offload_mode=str(mode),
                process_length=int(process_length),
                target_fps=float(target_fps),
                max_res=int(max_res),
                far_black=bool(far_black),
                crf=int(ffmpeg_crf) if int(ffmpeg_crf) >= 0 else int(crf),
                preset=(str(ffmpeg_preset).strip() or str(preset)),
                debug_mem=bool(debug_mem),
                decode_chunk_size=int(dcs),
                codec=(str(ffmpeg_codec).strip() or "h264_nvenc"),
                pix_fmt=(str(ffmpeg_pix_fmt).strip() or "yuv420p"),
                ffmpeg_extra_args=str(ffmpeg_extra_args or ""),
            )

            _postprocess_depth(
                tmp_depth,
                out_final,
                cw,
                ch,
                pad_x,
                pad_y,
                post_scale_w,
                post_scale_h,
                recenter_w=(recenter_w if recenter_enabled else 0),
                recenter_h=(recenter_h if recenter_enabled else 0),
                recenter_pad_top=(recenter_pad_top if recenter_enabled else 0),
                final_upscale=bool(final_upscale),
                padded=bool(padded),
                ffmpeg_codec=ffmpeg_codec,
                ffmpeg_crf=int(ffmpeg_crf),
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_pix_fmt=ffmpeg_pix_fmt,
                ffmpeg_extra_args=ffmpeg_extra_args,
            )

        try:
            attempt(cpu_offload_mode, decode_chunk_size)
            dt = time.perf_counter() - t0
            ok += 1
            print(f"[OK  ] {i}/{total} done in {dt:.1f}s")

        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"[ERR ] {i}/{total} failed in {dt:.1f}s: {type(e).__name__}: {e}")
            _cuda_cleanup()

            retried = False
            if retry_sequential and _is_oom(e) and cpu_offload_mode.lower() != "sequential":
                try:
                    print("[RETRY] OOM -> cpu_offload_mode=sequential")
                    dcs2 = 1 if retry_decode_chunk_size_1 else decode_chunk_size
                    attempt("sequential", dcs2)
                    ok += 1
                    retried = True
                    print("[OK  ] retry sequential done")
                except Exception as e2:
                    print(f"[FAIL] retry sequential: {type(e2).__name__}: {e2}")
                    _cuda_cleanup()

            if (not retried) and retry_decode_chunk_size_1 and _is_oom(e) and int(decode_chunk_size) != 1:
                try:
                    print("[RETRY] OOM -> decode_chunk_size=1 (same offload)")
                    attempt(cpu_offload_mode, 1)
                    ok += 1
                    retried = True
                    print("[OK  ] retry dcs=1 done")
                except Exception as e2:
                    print(f"[FAIL] retry dcs=1: {type(e2).__name__}: {e2}")
                    _cuda_cleanup()

            if retried:
                # even after a retry success, do a soft cleanup
                if not keep_temps:
                    for p in (tmp_pre, tmp_depth):
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass
                _cuda_cleanup()
                continue

            failed += 1

            if remove_partial_output and out_final.exists():
                try:
                    out_final.unlink()
                except Exception:
                    pass

            traceback.print_exc()

            # CRITICAL: hard reset after a failed file to avoid cascade failures
            hard_reset_runner(cpu_offload_mode)

        finally:
            if not keep_temps:
                for p in (tmp_pre, tmp_depth):
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass
            _cuda_cleanup()

    print("-----")
    print(f"[DONE] ok={ok} skipped={skipped} failed={failed} total={total}")
    print(f"[DIR ] output_dir={output_dir_p}")
    print(f"[DIR ] temp_dir={temp_dir} (kept={keep_temps})")


def _bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y", "on")


def _split_by_ratio(total: int, top_ref: int, bottom_ref: int) -> tuple[int, int]:
    total_i = max(0, int(total))
    if total_i <= 0:
        return 0, 0
    top_i = max(0, int(top_ref))
    bot_i = max(0, int(bottom_ref))
    den = top_i + bot_i
    if den <= 0:
        top = total_i // 2
    else:
        top = int(round((total_i * float(top_i)) / float(den)))
    top = max(0, min(total_i, top))
    return top, total_i - top


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_script", default="./depthcrafter_nogui_batch.py")
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--output_dir", default="./out")
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--out_ext", default=".mp4")
    ap.add_argument("--suffix", default="_depth")

    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--inference_steps", type=int, default=5)
    ap.add_argument("--window_size", type=int, default=64)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu_offload_mode", default="model")
    ap.add_argument("--decode_chunk_size", type=int, default=8)
    ap.add_argument("--process_length", type=int, default=-1)
    ap.add_argument("--target_fps", type=float, default=-1.0)
    ap.add_argument("--max_res", type=int, default=1920)
    ap.add_argument("--far_black", type=_bool, default=True)
    ap.add_argument("--crf", type=int, default=0)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--debug_mem", type=_bool, default=False)

    ap.add_argument("--auto_sizes", type=_bool, default=True)

    ap.add_argument("--content_w", type=int, default=960)
    ap.add_argument("--content_h", type=int, default=540)
    ap.add_argument("--pad_w", type=int, default=1024)
    ap.add_argument("--pad_h", type=int, default=576)

    ap.add_argument("--retry_sequential", type=_bool, default=False)
    ap.add_argument("--retry_decode_chunk_size_1", type=_bool, default=False)
    ap.add_argument("--remove_partial_output", type=_bool, default=True)
    ap.add_argument("--restart_every", type=int, default=0)
    ap.add_argument("--keep_temps", type=_bool, default=False)

    # NEW: final upscale on/off (default True keeps current behavior)
    ap.add_argument("--final_upscale", type=_bool, default=True)
    ap.add_argument("--pad_align_bottom", type=_bool, default=True)
    ap.add_argument("--scene_strip_pad_top", type=int, default=0)
    ap.add_argument("--scene_strip_pad_bottom", type=int, default=0)

    # Optional ffmpeg overrides (empty/-1 keeps legacy behavior).
    ap.add_argument("--ffmpeg_codec", default="")
    ap.add_argument("--ffmpeg_crf", type=int, default=-1)
    ap.add_argument("--ffmpeg_preset", default="")
    ap.add_argument("--ffmpeg_pix_fmt", default="")
    ap.add_argument("--ffmpeg_extra_args", default="")

    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
