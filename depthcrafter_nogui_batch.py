#!/usr/bin/env python3
"""
DepthCrafter -> depth-map video ONLY (grayscale), no splatting.

Based on the streaming fork script (lossless ffmpeg writer, pipeline window/overlap)
but stripped down to just DepthCrafter inference + grayscale export.

CLI exposes (as requested): guidance scale, inference steps, target width/height,
seed, cpu offload mode (none/model/sequential).
Plus: window_size and overlap (streaming chunking).
"""

import os
import inspect
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

def _mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def _cuda_mem(tag: str):
    if not torch.cuda.is_available():
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        print(
            f"[MEM] {tag} | free={_mb(free_b):.0f}MB total={_mb(total_b):.0f}MB | "
            f"alloc={_mb(alloc):.0f}MB reserved={_mb(reserved):.0f}MB peak={_mb(peak):.0f}MB"
        )
    except Exception as e:
        print(f"[MEM] {tag} | (failed to query CUDA mem: {e})")


from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter


# ----------------------------
# ffprobe / ffmpeg helpers
# ----------------------------

def _probe_fps_str(path: str) -> str:
    """Return avg_frame_rate as string, e.g. '24000/1001'."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "csv=p=0",
                path,
            ],
            text=True,
        ).strip()
        return out if out else ""
    except Exception:
        return ""


def _fps_str_to_float(s: str) -> float:
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return 0.0


def _pick_timescale(fps_f: float) -> int:
    # Coerente con MP4, evita time_base strani.
    if 23.95 <= fps_f <= 24.05:
        return 24000
    if 29.90 <= fps_f <= 30.10:
        return 30000
    if 24.90 <= fps_f <= 25.10:
        return 25000
    return int(max(1000, round(fps_f * 1000)))


def _start_ffmpeg_gray_writer(
    path: str,
    w: int,
    h: int,
    fps_str: str,
    crf: int = 0,
    preset: str = "medium",
    debug_mem: bool = False,
    decode_chunk_size: int = 4,
    pix_out: str = "yuv420p",
    loglevel: str = "error",
    vf: str = None,
):
    """
    Stream raw GRAY8 frames into ffmpeg -> libx264.
    crf=0 is lossless for libx264.
    """
    timescale = _pick_timescale(_fps_str_to_float(fps_str) or 0.0)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", loglevel, "-y",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", fps_str, "-i", "-",
        "-vsync", "cfr", "-r", fps_str,
        "-video_track_timescale", str(timescale),
        "-color_primaries", "bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv","-movflags","+write_colr",
    ]
    if vf:
        cmd.extend(["-vf", vf])
    cmd.extend([
        "-c:v", "h264_nvenc", "-preset", preset, "-qp", str(crf),
        "-pix_fmt", pix_out,
        path,
    ])
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ----------------------------
# Video decode helper
# ----------------------------

def _round64(x: int) -> int:
    return max(64, int(round(x / 64.0) * 64))


def read_video_frames(
    video_path: str,
    process_length: int,
    target_fps: float,
    max_res: int,
    dataset: str = "open",
    target_width: int = 0,
    target_height: int = 0,
):
    """
    Decode frames with decord.
    - If target_width/height provided (>0), uses them (rounded to multiples of 64).
    - Otherwise keeps the original "max_res" logic.
    Returns: frames float32 [T,H,W,3] in [0,1], fps_str, original_h, original_w
    """
    if dataset != "open":
        raise ValueError("Only dataset='open' is supported in this stripped script.")

    vid0 = VideoReader(video_path, ctx=cpu(0))
    f0 = vid0.get_batch([0]).asnumpy()[0]
    original_height, original_width = f0.shape[0], f0.shape[1]

    if target_width > 0 and target_height > 0:
        width = _round64(target_width)
        height = _round64(target_height)
    else:
        # Original heuristic: scale down if larger than max_res, keep multiple of 64
        height = _round64(original_height)
        width = _round64(original_width)
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = _round64(int(original_height * scale))
            width = _round64(int(original_width * scale))

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    src_fps = float(vid.get_avg_fps())
    # Use exact ffprobe avg_frame_rate if possible (keeps CFR sane in output).
    fps_str = _probe_fps_str(video_path) or str(src_fps)

    if target_fps is None or float(target_fps) <= 0:
        # Use source fps
        stride = 1
    else:
        stride = round(src_fps / float(target_fps))
        stride = max(stride, 1)
        fps_str = str(float(target_fps))  # force output fps

    frames_idx = list(range(0, len(vid), stride))
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]

    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, fps_str, original_height, original_width


# ----------------------------
# DepthCrafter runner
# ----------------------------

class DepthCrafterDepthOnly:
    def __init__(
        self,
    unet_path: str = None,
    pre_trained_path: str = None,
cpu_offload_mode: str = "model",  # none|model|sequential
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # CPU offload modes
        mode = (cpu_offload_mode or "model").lower()
        if mode == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif mode == "model":
            self.pipe.enable_model_cpu_offload()
        elif mode == "none":
            self.pipe.to("cuda")
        else:
            raise ValueError("cpu_offload_mode must be one of: none, model, sequential")

        # Attention mem savers (best-effort)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def infer_to_gray_video(
        self,
        input_video_path: str,
        output_video_path: str,
        guidance_scale: float = 1.0,
        inference_steps: int = 5,
        target_width: int = 640,
        target_height: int = 384,
        window_size: int = 40,
        overlap: int = 8,
        seed: int = 42,
        cpu_offload_mode: str = "model",  # kept for CLI symmetry; init already did it
        # extra knobs (keep them, but they're not the "main" GUI ones)
        process_length: int = -1,
        target_fps: float = -1.0,
        max_res: int = 1024,
        far_black: bool = False,  # match GUI default: far=black
        crf: int = 0,
        preset: str = "medium",
        debug_mem: bool = True,
        decode_chunk_size: int = 8,
    ):
        if window_size <= overlap:
            raise ValueError("window_size must be > overlap")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        set_seed(int(seed))
        if debug_mem and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _cuda_mem('start infer_to_gray_video')


        frames, fps_str, original_h, original_w = read_video_frames(
            input_video_path,
            process_length=process_length,
            target_fps=target_fps,
            max_res=max_res,
            dataset="open",
            target_width=int(target_width),
            target_height=int(target_height),
        )
        if debug_mem:
            print(f"[DBG] decoded frames: {frames.shape} dtype={frames.dtype} fps={fps_str} orig={original_w}x{original_h}")
            _cuda_mem('after read_video_frames')

        # DepthCrafter inference
        call_kwargs = dict(
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            num_inference_steps=int(inference_steps),
            window_size=int(window_size),
            overlap=int(overlap),
            decode_chunk_size=int(decode_chunk_size),
            track_time=bool(debug_mem),
        )

        # compat: guidance_scale vs min/max_guidance_scale
        sig = inspect.signature(self.pipe.__call__).parameters
        if "guidance_scale" in sig:
            call_kwargs["guidance_scale"] = float(guidance_scale)
        elif "min_guidance_scale" in sig and "max_guidance_scale" in sig:
            call_kwargs["min_guidance_scale"] = float(guidance_scale)
            call_kwargs["max_guidance_scale"] = float(guidance_scale)

        # keep only supported kwargs
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig}

        if debug_mem:
            os.environ["SC_TRACE_MEM"] = "1"
            print(f"[DBG] pipe kwargs: {call_kwargs}")
        res = self.pipe(frames, **call_kwargs).frames[0]  # [T,H,W,3] float
        if debug_mem:
            _cuda_mem('after pipe __call__ returns')
        res = res.sum(-1) / res.shape[-1]  # [T,H,W]
        # Normalize over the whole video
        dmin = float(res.min())
        dmax = float(res.max())
        if dmax > dmin:
            res = (res - dmin) / (dmax - dmin)
        else:
            res = np.zeros_like(res, dtype=np.float32)

        #if far_black:
            # If higher depth means farther, invert so far -> 0 (black)
            #res = 1.0 - res

        # Stream grayscale video to ffmpeg (no giant RGB visualization array)

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".", exist_ok=True)
        p = _start_ffmpeg_gray_writer(
            output_video_path,
            w=res.shape[2],
            h=res.shape[1],
            fps_str=fps_str,
            crf=int(crf),
            preset=str(preset),
            pix_out="yuv420p",
            vf=f"scale={original_w}:{original_h}:flags=bilinear",
        )
        try:
            for i in range(res.shape[0]):
                frame_u8 = np.clip(res[i] * 255.0, 0, 255).astype(np.uint8)
                p.stdin.write(frame_u8.tobytes())
        finally:
            try:
                p.stdin.close()
            except Exception:
                pass
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited with code {rc} while writing: {output_video_path}")

        return output_video_path


def main(
    input_video_path: str,
    output_video_path: str,
    unet_path: str = None,
    pre_trained_path: str = None,
    guidance_scale: float = 1.0,
    inference_steps: int = 5,
    target_width: int = 1408,
    target_height: int = 768,
    window_size: int = 40,
    overlap: int = 8,
    seed: int = 42,
    cpu_offload_mode: str = "model",  # none|model|sequential
    # optional knobs (keep)
    process_length: int = -1,
    target_fps: float = -1.0,
    max_res: int = 1920,
    far_black: bool = True,
    crf: int = 0,
    preset: str = "medium",
    debug_mem: bool = False,
    decode_chunk_size: int = 8,
):
    # Resolve model locations. If None, use local StereoCrafter/weights defaults.
    if unet_path is None:
        unet_path = "./weights/DepthCrafter"
    if pre_trained_path is None:
        pre_trained_path = "./weights/stable-video-diffusion-img2vid-xt-1-1"

    runner = DepthCrafterDepthOnly(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path,
        cpu_offload_mode=cpu_offload_mode,
    )
    runner.infer_to_gray_video(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        target_width=target_width,
        target_height=target_height,
        seed=seed,
        cpu_offload_mode=cpu_offload_mode,
        process_length=process_length,
        target_fps=target_fps,
        window_size=window_size,
        overlap=overlap,
        max_res=max_res,
        far_black=far_black,
        crf=crf,
        preset=preset,
        debug_mem=debug_mem,
        decode_chunk_size=decode_chunk_size,
    )


if __name__ == "__main__":
    Fire(main)
