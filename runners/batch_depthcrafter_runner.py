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
"""
import gc
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dependency.ffmpeg_encoding_profiles import (
    normalize_codec_strict,
    resolve_depth_final_grayscale_profile,
    resolve_depth_preprocess_profile,
)


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


def _resolve_worker_class(mod):
    for attr_name in ("DepthCrafterDepthOnly", "DepthCrafterDepthOnlyStream"):
        cls = getattr(mod, attr_name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        "worker_script must expose class DepthCrafterDepthOnly or DepthCrafterDepthOnlyStream"
    )


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


def _cuda_allocated_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
    except Exception:
        pass
    return 0.0


def _load_retry_resume_state(path: Path) -> dict[str, object] | None:
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return None


def _save_retry_resume_state(path: Path, input_name: str, next_attempt: int, total_attempts: int) -> None:
    payload = {
        "input_name": str(input_name),
        "next_attempt": int(next_attempt),
        "total_attempts": int(total_attempts),
        "updated_at": int(time.time()),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] failed writing retry resume state: {e}")


def _clear_retry_resume_state(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _load_retry_skip_manifest(path: Path) -> set[str]:
    try:
        if not path.exists():
            return set()
        out: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            name = str(line).strip()
            if name:
                out.add(name)
        return out
    except Exception:
        return set()


def _save_retry_skip_manifest(path: Path, names: set[str]) -> None:
    try:
        ordered = sorted({str(x).strip() for x in names if str(x).strip()})
        if not ordered:
            if path.exists():
                path.unlink()
            return
        path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[WARN] failed writing retry skip manifest: {e}")


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


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _ensure_even_min2(n: int) -> int:
    v = max(2, int(n))
    if v % 2:
        v -= 1
    return max(2, v)


def _scaled_even_min2(n: int, factor: float) -> int:
    return _ensure_even_min2(int(int(n) * float(factor)))


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
):
    # Scale always to content_w x content_h, then pad ONLY if needed.
    # Important: pad placement is explicit (pad_x/pad_y) so caller can anchor content.
    crop_top = max(0, int(src_crop_top))
    crop_bottom = max(0, int(src_crop_bottom))
    vf_parts = []
    if crop_top > 0 or crop_bottom > 0:
        vf_parts.append(f"crop=iw:ih-{crop_top + crop_bottom}:0:{crop_top}")
    vf_parts.append(f"scale={content_w}:{content_h}:flags=lanczos")
    if pad_w != content_w or pad_h != content_h:
        vf_parts.append(f"pad={pad_w}:{pad_h}:{int(pad_x)}:{int(pad_y)}:black")
    vf_parts.append("format=yuv444p")
    vf = ",".join(vf_parts)
    selected_codec = str(ffmpeg_codec or "").strip() or "libx264"
    normalize_codec_strict(selected_codec)
    profile = resolve_depth_preprocess_profile(selected_codec)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-an",
        "-vf", vf,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+write_colr",
    ]
    cmd.extend(profile.generated_args)
    cmd.append(str(dst))
    _run(cmd, log_prefix="[FFMPEG-PREPROCESS]", check=True)


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
):
    # If padded is True, remove pad using explicit crop offsets before optional upscale.
    vf_parts = []
    if padded:
        vf_parts.append(f"crop={crop_w}:{crop_h}:{int(crop_x)}:{int(crop_y)}")
    if final_upscale:
        vf_parts.append(f"scale={out_w}:{out_h}:flags=neighbor")
    if int(recenter_w) > 0 and int(recenter_h) > 0:
        pad_top = max(0, int(recenter_pad_top))
        vf_parts.append(f"pad={int(recenter_w)}:{int(recenter_h)}:0:{pad_top}:black")
    vf_parts.append("format=yuv444p")
    vf = ",".join(vf_parts)
    profile = resolve_depth_final_grayscale_profile()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_depth),
        "-an",
        "-vf", vf,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv",
        "-movflags", "+write_colr",
    ]
    cmd.extend(profile.generated_args)
    cmd.append(str(dst_final))
    _run(cmd, log_prefix="[FFMPEG-POSTPROCESS]", check=True)


def run(
    worker_script: str = "./runners/depthcrafter_nogui_batch.py",
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
    debug_mem: bool = False,
    # Pre/post sizes (used only if auto_sizes=False)
    auto_sizes: bool = True,
    content_w: int = 960,
    content_h: int = 540,
    pad_w: int = 1024,
    pad_h: int = 576,
    scale_factor: float = 0.5,
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
    # Depth preprocess codec only. Temporary/final grayscale outputs are fixed.
    ffmpeg_codec: str = "",
    retry_policy_json: str = "",
    retry_process_restart_alloc_mb: int = 1024,
):
    input_dir_p = Path(input_dir).resolve()
    output_dir_p = Path(output_dir).resolve()
    _ensure_dir(output_dir_p)

    temp_dir = output_dir_p / ".tmp_depthcrafter"
    _ensure_dir(temp_dir)

    # weights relative to StereoCrafter root
    unet_path = "./weights/DepthCrafter"
    pre_trained_path = "./weights/stable-video-diffusion-img2vid-xt-1-1"

    def make_runner(mode: str, worker_script_path: str):
        mod = _load_worker_module(worker_script_path)
        worker_cls = _resolve_worker_class(mod)
        r = worker_cls(
            unet_path=unet_path,
            pre_trained_path=pre_trained_path,
            cpu_offload_mode=mode,
        )
        setattr(r, "_batch_mode", mode)
        setattr(r, "_batch_worker_script", str(Path(worker_script_path).resolve()))
        return r

    worker_script_base = str(worker_script)
    runner = make_runner(cpu_offload_mode, worker_script_base)

    def hard_reset_runner(mode: str, worker_script_path: str | None = None):
        """Hard reset pipeline after a failure (OOM can poison CUDA state)."""
        nonlocal runner
        try:
            del runner
        except Exception:
            pass
        _cuda_cleanup()
        time.sleep(1.0)
        runner = make_runner(mode, worker_script_path or worker_script_base)

    files = sorted(input_dir_p.glob(glob))
    if not files:
        print(f"[INFO] No inputs matched: {input_dir_p}/{glob}")
        return

    total = len(files)
    ok = skipped = failed = 0
    stream_success_files: list[str] = []
    processed_non_skip = 0
    try:
        scale_factor_f = float(scale_factor)
    except Exception:
        scale_factor_f = 0.5
    scale_factor_f = max(0.5, scale_factor_f)

    print(f"[INFO] Inputs: {total} | input_dir={input_dir_p} | output_dir={output_dir_p}")
    print(f"[INFO] Worker: {Path(worker_script_base).resolve()}")
    print(
        f"[INFO] Sizes: auto_sizes={auto_sizes} scale_factor={scale_factor_f:.2f} "
        f"| fixed_pad={pad_w}x{pad_h} fixed_content={content_w}x{content_h}"
    )
    print(f"[INFO] Params: offload={cpu_offload_mode} decode_chunk_size={decode_chunk_size} window={window_size} overlap={overlap} steps={inference_steps} gs={guidance_scale}")
    print(f"[INFO] final_upscale={final_upscale}")
    print(
        f"[INFO] scene_strip_pad: top={max(0, int(scene_strip_pad_top))} "
        f"bottom={max(0, int(scene_strip_pad_bottom))}"
    )
    print(
        "[INFO] ffmpeg policy: "
        f"depth_preprocess_codec={(str(ffmpeg_codec).strip() or 'libx264')} "
        "depth_preprocess_mode=lossless "
        "depth_temp_gray=libx264 qp0 yuv444p "
        "depth_final=libx264 qp0 yuv444p"
    )
    retry_profiles, policy_was_explicit = _parse_retry_policy_profiles(
        retry_policy_json,
        cpu_offload_mode,
        worker_script_base,
        window_size,
        overlap,
    )
    retry_resume_state_file = output_dir_p / ".depth_retry_resume_state.json"
    retry_skip_manifest_file = output_dir_p / ".depth_retry_skipped.txt"
    retry_resume_state = _load_retry_resume_state(retry_resume_state_file)
    retry_skip_persisted = _load_retry_skip_manifest(retry_skip_manifest_file)
    if retry_resume_state and str(retry_resume_state.get("input_name") or "").strip():
        print(
            "[INFO] retry resume state found: "
            f"input={retry_resume_state.get('input_name')} "
            f"next_attempt={retry_resume_state.get('next_attempt')}"
        )
    if retry_skip_persisted:
        print(f"[INFO] persisted retry-skip files loaded: {len(retry_skip_persisted)}")
    if policy_was_explicit:
        print("[INFO] retry policy source=gui/env")
    else:
        print("[INFO] retry policy source=default")
    for prof in retry_profiles:
        alloc = _allocator_conf_from_profile(prof) or "default"
        print(
            "[INFO] retry profile "
            f"{prof['name']}: offload={prof['cpu_offload_mode']} alloc={alloc} "
            f"script={Path(str(prof['worker_script'])).name} "
            f"window={int(prof['window_size'])} overlap={int(prof['overlap'])}"
        )
    retry_skipped: list[str] = []

    for i, in_path in enumerate(files, 1):
        out_name = _default_out_name(in_path, out_ext=out_ext, suffix=suffix)
        out_final = output_dir_p / out_name

        if in_path.name in retry_skip_persisted:
            skipped += 1
            print(f"[SKIP] {i}/{total} {in_path.name} (retry-skip persisted)")
            continue

        if out_final.exists() and out_final.stat().st_size > 0:
            skipped += 1
            print(f"[SKIP] {i}/{total} {in_path.name} -> {out_final.name} (exists)")
            if retry_resume_state and str(retry_resume_state.get("input_name") or "") == in_path.name:
                _clear_retry_resume_state(retry_resume_state_file)
                retry_resume_state = None
            continue

        processed_non_skip += 1
        if restart_every and processed_non_skip > 1 and (processed_non_skip - 1) % int(restart_every) == 0:
            print(
                "[INFO] Restarting pipeline "
                f"(restart_every={restart_every}, non_skip={processed_non_skip}) ..."
            )
            hard_reset_runner(cpu_offload_mode)

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
            cw = _scaled_even_min2(int(orig_w), scale_factor_f)
            ch = _scaled_even_min2(int(core_h), scale_factor_f)
            pw = _round_up(cw, 64)
            ph = _round_up(ch, 64)
        else:
            cw, ch, pw, ph = int(content_w), int(content_h), int(pad_w), int(pad_h)
            cw = _ensure_even_min2(cw)
            ch = _ensure_even_min2(ch)
            pw = _round_up(max(cw, pw), 64)
            ph = _round_up(max(ch, ph), 64)

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
            recenter_w = _scaled_even_min2(int(orig_w), scale_factor_f)
            recenter_h = _scaled_even_min2(int(orig_h), scale_factor_f)

        recenter_pad_total = max(0, int(recenter_h) - int(post_scale_h))
        # Keep output anchored to bottom: never center on re-pad.
        recenter_pad_top = int(recenter_pad_total)
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
            f"(orig {orig_w}x{orig_h} | scale{scale_factor_f:.2f} {cw}x{ch} | pad {pw}x{ph} "
            f"| strip_tb={strip_top},{strip_bottom} | pad_xy={pad_x},{pad_y} "
            f"| recenter={'yes' if recenter_enabled else 'no'} | padded={padded})"
        )
        t0 = time.perf_counter()

        def attempt(mode: str, dcs: int, worker_script_path: str, window_size_value: int, overlap_value: int):
            nonlocal runner
            target_script = str(Path(worker_script_path).resolve())
            if (
                mode != getattr(runner, "_batch_mode", mode)
                or target_script != getattr(runner, "_batch_worker_script", target_script)
            ):
                hard_reset_runner(mode, worker_script_path)

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
            )
            worker_kwargs = dict(
                input_video_path=str(tmp_pre),
                output_video_path=str(tmp_depth),
                guidance_scale=float(guidance_scale),
                inference_steps=int(inference_steps),
                target_width=int(pw),
                target_height=int(ph),
                window_size=int(window_size_value),
                overlap=int(overlap_value),
                seed=int(seed),
                cpu_offload_mode=str(mode),
                process_length=int(process_length),
                target_fps=float(target_fps),
                max_res=int(max_res),
                far_black=bool(far_black),
                debug_mem=bool(debug_mem),
                decode_chunk_size=int(dcs),
            )
            sig = inspect.signature(runner.infer_to_gray_video)
            worker_kwargs = {k: v for k, v in worker_kwargs.items() if k in sig.parameters}
            runner.infer_to_gray_video(**worker_kwargs)

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
            )

        def _cleanup_local_temps():
            if not keep_temps:
                for p in (tmp_pre, tmp_depth):
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

        if policy_was_explicit:
            success = False
            last_exc: Exception | None = None
            start_attempt_idx = 1
            if retry_resume_state and str(retry_resume_state.get("input_name") or "") == in_path.name:
                try:
                    start_attempt_idx = int(retry_resume_state.get("next_attempt") or 1)
                except Exception:
                    start_attempt_idx = 1
                start_attempt_idx = max(1, min(len(retry_profiles), start_attempt_idx))
                if start_attempt_idx > 1:
                    print(
                        f"[RETRY] resuming {i}/{total} from attempt "
                        f"{start_attempt_idx}/{len(retry_profiles)} after process restart"
                    )
            try:
                for attempt_idx in range(start_attempt_idx, len(retry_profiles) + 1):
                    prof = retry_profiles[attempt_idx - 1]
                    alloc_conf = _allocator_conf_from_profile(prof)
                    offload_mode = str(prof["cpu_offload_mode"])
                    profile_worker_script = str(prof["worker_script"])
                    profile_window_size = int(prof["window_size"])
                    profile_overlap = int(prof["overlap"])
                    _save_retry_resume_state(
                        retry_resume_state_file,
                        in_path.name,
                        attempt_idx,
                        len(retry_profiles),
                    )
                    print(
                        f"[RETRY] {i}/{total} attempt {attempt_idx}/{len(retry_profiles)} "
                        f"profile={prof['name']} offload={offload_mode} "
                        f"alloc={alloc_conf or 'default'} "
                        f"script={Path(profile_worker_script).name} "
                        f"window={profile_window_size} overlap={profile_overlap}"
                    )
                    _apply_allocator_conf(alloc_conf)
                    _cuda_cleanup()
                    t_attempt = time.perf_counter()
                    try:
                        attempt(
                            offload_mode,
                            decode_chunk_size,
                            profile_worker_script,
                            profile_window_size,
                            profile_overlap,
                        )
                        success = True
                        ok += 1
                        dt = time.perf_counter() - t0
                        dta = time.perf_counter() - t_attempt
                        if "stream" in Path(profile_worker_script).name.lower():
                            stream_success_files.append(in_path.name)
                        print(
                            f"[OK  ] {i}/{total} done in {dt:.1f}s "
                            f"(attempt={attempt_idx} profile={prof['name']} attempt_time={dta:.1f}s)"
                        )
                        if in_path.name in retry_skip_persisted:
                            retry_skip_persisted.discard(in_path.name)
                            _save_retry_skip_manifest(retry_skip_manifest_file, retry_skip_persisted)
                        _clear_retry_resume_state(retry_resume_state_file)
                        retry_resume_state = None
                        break
                    except Exception as e:
                        last_exc = e
                        dta = time.perf_counter() - t_attempt
                        print(
                            f"[ERR ] {i}/{total} attempt {attempt_idx}/{len(retry_profiles)} "
                            f"failed in {dta:.1f}s: {type(e).__name__}: {e}"
                        )
                        _cuda_cleanup()
                        alloc_after_fail_mb = _cuda_allocated_mb()
                        print(
                            f"[RETRY] post-fail cuda_alloc={alloc_after_fail_mb:.0f}MB "
                            f"(threshold={int(retry_process_restart_alloc_mb)}MB)"
                        )
                        if remove_partial_output and out_final.exists():
                            try:
                                out_final.unlink()
                            except Exception:
                                pass
                        if (
                            attempt_idx < len(retry_profiles)
                            and float(alloc_after_fail_mb) >= float(retry_process_restart_alloc_mb)
                        ):
                            next_attempt = attempt_idx + 1
                            _save_retry_resume_state(
                                retry_resume_state_file,
                                in_path.name,
                                next_attempt,
                                len(retry_profiles),
                            )
                            print(
                                f"[RETRY] requesting process restart before attempt "
                                f"{next_attempt}/{len(retry_profiles)}: residual CUDA memory high"
                            )
                            try:
                                sys.stdout.flush()
                                sys.stderr.flush()
                            except Exception:
                                pass
                            raise SystemExit(124)
                        # Soft reset even when no process restart is needed.
                        hard_reset_runner(offload_mode, profile_worker_script)

                if not success:
                    failed += 1
                    retry_skipped.append(in_path.name)
                    retry_skip_persisted.add(in_path.name)
                    _save_retry_skip_manifest(retry_skip_manifest_file, retry_skip_persisted)
                    dt = time.perf_counter() - t0
                    print(
                        f"[SKIP] {i}/{total} {in_path.name} skipped after "
                        f"{len(retry_profiles)} retry profiles ({dt:.1f}s)"
                    )
                    if last_exc is not None:
                        traceback.print_exception(
                            type(last_exc),
                            last_exc,
                            last_exc.__traceback__,
                        )
                    _clear_retry_resume_state(retry_resume_state_file)
                    retry_resume_state = None
                    hard_reset_runner(cpu_offload_mode, worker_script_base)
            finally:
                _cleanup_local_temps()
                _cuda_cleanup()
            continue

        try:
            attempt(
                cpu_offload_mode,
                decode_chunk_size,
                worker_script_base,
                int(window_size),
                int(overlap),
            )
            dt = time.perf_counter() - t0
            ok += 1
            if "stream" in Path(worker_script_base).name.lower():
                stream_success_files.append(in_path.name)
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
                    attempt("sequential", dcs2, worker_script_base, int(window_size), int(overlap))
                    ok += 1
                    if "stream" in Path(worker_script_base).name.lower():
                        stream_success_files.append(in_path.name)
                    retried = True
                    print("[OK  ] retry sequential done")
                except Exception as e2:
                    print(f"[FAIL] retry sequential: {type(e2).__name__}: {e2}")
                    _cuda_cleanup()

            if (not retried) and retry_decode_chunk_size_1 and _is_oom(e) and int(decode_chunk_size) != 1:
                try:
                    print("[RETRY] OOM -> decode_chunk_size=1 (same offload)")
                    attempt(cpu_offload_mode, 1, worker_script_base, int(window_size), int(overlap))
                    ok += 1
                    if "stream" in Path(worker_script_base).name.lower():
                        stream_success_files.append(in_path.name)
                    retried = True
                    print("[OK  ] retry dcs=1 done")
                except Exception as e2:
                    print(f"[FAIL] retry dcs=1: {type(e2).__name__}: {e2}")
                    _cuda_cleanup()

            if retried:
                _cleanup_local_temps()
                _cuda_cleanup()
                continue

            failed += 1

            if remove_partial_output and out_final.exists():
                try:
                    out_final.unlink()
                except Exception:
                    pass

            traceback.print_exc()
            hard_reset_runner(cpu_offload_mode)

        finally:
            _cleanup_local_temps()
            _cuda_cleanup()

    print("-----")
    print(f"[DONE] ok={ok} skipped={skipped} failed={failed} total={total}")
    if retry_skipped:
        preview = ", ".join(retry_skipped[:10])
        more = "" if len(retry_skipped) <= 10 else f", ... (+{len(retry_skipped) - 10} more)"
        print(f"[DONE] retry-skip files ({len(retry_skipped)}): {preview}{more}")
    if stream_success_files:
        stream_success_unique = list(dict.fromkeys(stream_success_files))
        print(f"[DONE] stream output files ({len(stream_success_unique)}):")
        for name in stream_success_unique:
            print(f"  - {name}")
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


RETRY_PROFILE_ORDER = ("run", "retry1", "retry2", "retry3")
RETRY_OFFLOAD_CHOICES = {"none", "model", "sequential"}
RETRY_MAX_SPLIT_CHOICES = {64, 128, 256, 512}


def _norm_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _norm_offload_mode(value: object, fallback: str = "model") -> str:
    mode = str(value or "").strip().lower()
    if mode in RETRY_OFFLOAD_CHOICES:
        return mode
    fb = str(fallback or "model").strip().lower()
    return fb if fb in RETRY_OFFLOAD_CHOICES else "model"


def _norm_max_split(value: object) -> int | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"", "none", "off", "0", "false"}:
        return None
    try:
        parsed = int(float(s))
    except Exception:
        return None
    if parsed in RETRY_MAX_SPLIT_CHOICES:
        return parsed
    return None


def _norm_retry_worker_script(value: object, fallback: str) -> str:
    txt = str(value or "").strip()
    return txt if txt else str(fallback)


def _norm_retry_window_size(value: object, fallback: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(fallback)
    return max(1, parsed)


def _norm_retry_overlap(value: object, fallback: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(fallback)
    return max(0, parsed)


def _default_retry_profiles(
    base_offload: str,
    base_worker_script: str,
    base_window_size: int,
    base_overlap: int,
) -> list[dict[str, object]]:
    inherited = _norm_offload_mode(base_offload, "model")
    return [
        {
            "name": "run",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": None,
            "cpu_offload_mode": inherited,
            "worker_script": str(base_worker_script),
            "window_size": int(base_window_size),
            "overlap": int(base_overlap),
        },
        {
            "name": "retry1",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 512,
            "cpu_offload_mode": inherited,
            "worker_script": str(base_worker_script),
            "window_size": int(base_window_size),
            "overlap": int(base_overlap),
        },
        {
            "name": "retry2",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 64,
            "cpu_offload_mode": inherited,
            "worker_script": str(base_worker_script),
            "window_size": int(base_window_size),
            "overlap": int(base_overlap),
        },
        {
            "name": "retry3",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 64,
            "cpu_offload_mode": "sequential",
            "worker_script": str(base_worker_script),
            "window_size": int(base_window_size),
            "overlap": int(base_overlap),
        },
    ]


def _parse_retry_policy_profiles(
    policy_json: str,
    base_offload: str,
    base_worker_script: str,
    base_window_size: int,
    base_overlap: int,
) -> tuple[list[dict[str, object]], bool]:
    defaults = _default_retry_profiles(
        base_offload,
        base_worker_script,
        base_window_size,
        base_overlap,
    )
    txt = str(policy_json or "").strip()
    if not txt:
        return defaults, False
    try:
        raw = json.loads(txt)
    except Exception as e:
        print(f"[WARN] retry_policy_json parse failed: {e}. Using defaults.")
        return defaults, True
    if not isinstance(raw, dict):
        print("[WARN] retry_policy_json is not an object. Using defaults.")
        return defaults, True

    out: list[dict[str, object]] = []
    for idx, name in enumerate(RETRY_PROFILE_ORDER):
        base = defaults[idx]
        node = raw.get(name, {})
        if not isinstance(node, dict):
            node = {}
        out.append(
            {
                "name": name,
                "garbage_collection_threshold": _norm_bool(
                    node.get(
                        "garbage_collection_threshold",
                        base["garbage_collection_threshold"],
                    ),
                    bool(base["garbage_collection_threshold"]),
                ),
                "expandable_segments": _norm_bool(
                    node.get("expandable_segments", base["expandable_segments"]),
                    bool(base["expandable_segments"]),
                ),
                "max_split_size_mb": _norm_max_split(
                    node.get("max_split_size_mb", base["max_split_size_mb"])
                ),
                "cpu_offload_mode": _norm_offload_mode(
                    node.get("cpu_offload_mode", base["cpu_offload_mode"]),
                    str(base["cpu_offload_mode"]),
                ),
                "worker_script": _norm_retry_worker_script(
                    node.get("worker_script", base["worker_script"]),
                    str(base["worker_script"]),
                ),
                "window_size": _norm_retry_window_size(
                    node.get("window_size", base["window_size"]),
                    int(base["window_size"]),
                ),
                "overlap": _norm_retry_overlap(
                    node.get("overlap", base["overlap"]),
                    int(base["overlap"]),
                ),
            }
        )
    return out, True


def _allocator_conf_from_profile(profile: dict[str, object]) -> str:
    parts: list[str] = []
    if _norm_bool(profile.get("garbage_collection_threshold"), True):
        parts.append("garbage_collection_threshold:0.8")
    if _norm_bool(profile.get("expandable_segments"), True):
        parts.append("expandable_segments:True")
    max_split = _norm_max_split(profile.get("max_split_size_mb"))
    if max_split is not None:
        parts.append(f"max_split_size_mb:{max_split}")
    return ",".join(parts)


def _apply_allocator_conf(conf: str) -> None:
    conf_s = str(conf or "").strip()
    if conf_s:
        os.environ["PYTORCH_ALLOC_CONF"] = conf_s
    else:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    try:
        import torch

        alt = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if callable(alt):
            alt(conf_s)
            return
        setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
        if callable(setter):
            setter(conf_s)
    except Exception as e:
        print(f"[WARN] failed to apply allocator settings '{conf_s or 'default'}': {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker_script", default="./runners/depthcrafter_nogui_batch.py")
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
    ap.add_argument("--debug_mem", type=_bool, default=False)

    ap.add_argument("--auto_sizes", type=_bool, default=True)

    ap.add_argument("--content_w", type=int, default=960)
    ap.add_argument("--content_h", type=int, default=540)
    ap.add_argument("--pad_w", type=int, default=1024)
    ap.add_argument("--pad_h", type=int, default=576)
    ap.add_argument("--scale_factor", type=float, default=0.5)

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

    # Depth preprocess codec only; grayscale outputs are fixed.
    ap.add_argument("--ffmpeg_codec", default="")
    ap.add_argument("--retry_policy_json", default="")
    ap.add_argument("--retry_process_restart_alloc_mb", type=int, default=1024)

    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
