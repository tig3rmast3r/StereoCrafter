#!/usr/bin/env python3
"""
Experimental streaming DepthCrafter worker with carry-only latent overlap.

This variant is the next step after the previous streaming experiments:

- keeps decoded RGB frames on CPU
- encodes and denoises only one external chunk at a time
- injects only the previous chunk's final latent overlap at the first denoise
  step of the next chunk
- never builds a full-video `latents_all` tensor in memory
- decodes only finalized latent slices, stores raw grayscale slices temporarily,
  then performs one final global normalization + ffmpeg write pass

This stays closer to the original pipeline behavior than output-level blending,
while still avoiding the large full-video CUDA load at startup.
"""

import gc
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from decord import VideoReader, cpu
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    retrieve_timesteps,
)
from diffusers.training_utils import set_seed
from diffusers.utils.torch_utils import randn_tensor
from fire import Fire

from dependency.ffmpeg_encoding_profiles import resolve_depth_final_grayscale_profile
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter


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


def _cuda_cleanup():
    if not torch.cuda.is_available():
        gc.collect()
        return
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
    gc.collect()


def _probe_fps_str(path: str) -> str:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "csv=p=0",
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
    loglevel: str = "error",
    vf: str | None = None,
):
    timescale = _pick_timescale(_fps_str_to_float(fps_str) or 0.0)
    profile = resolve_depth_final_grayscale_profile()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{w}x{h}",
        "-r",
        fps_str,
        "-i",
        "-",
        "-vsync",
        "cfr",
        "-r",
        fps_str,
        "-video_track_timescale",
        str(timescale),
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+write_colr",
    ]
    if vf:
        cmd.extend(["-vf", vf])
    cmd.extend(profile.generated_args)
    cmd.append(path)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


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
    if dataset != "open":
        raise ValueError("Only dataset='open' is supported in this stripped script.")

    vid0 = VideoReader(video_path, ctx=cpu(0))
    f0 = vid0.get_batch([0]).asnumpy()[0]
    original_height, original_width = f0.shape[0], f0.shape[1]

    if target_width > 0 and target_height > 0:
        width = _round64(target_width)
        height = _round64(target_height)
    else:
        height = _round64(original_height)
        width = _round64(original_width)
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = _round64(int(original_height * scale))
            width = _round64(int(original_width * scale))

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    src_fps = float(vid.get_avg_fps())
    fps_str = _probe_fps_str(video_path) or str(src_fps)

    if target_fps is None or float(target_fps) <= 0:
        stride = 1
    else:
        stride = max(round(src_fps / float(target_fps)), 1)
        fps_str = str(float(target_fps))

    frames_idx = list(range(0, len(vid), stride))
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]

    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, fps_str, original_height, original_width


def _compute_chunks(num_frames: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if num_frames <= 0:
        return []
    if window_size <= overlap:
        raise ValueError("window_size must be > overlap")
    stride = window_size - overlap
    out: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        out.append((start, end))
        if end >= num_frames:
            break
        start += stride
    return out


def _normalize_depth_chunk(chunk: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    if global_max > global_min:
        return np.clip((chunk - global_min) / (global_max - global_min), 0.0, 1.0).astype(np.float32)
    return np.zeros_like(chunk, dtype=np.float32)


def _latent_overlap_weights(overlap: int) -> torch.Tensor:
    if overlap <= 0:
        return torch.zeros((1, 0, 1, 1, 1), dtype=torch.float32)
    return torch.linspace(0.0, 1.0, overlap, dtype=torch.float32).view(1, overlap, 1, 1, 1)


def _segment_name(idx: int, kind: str) -> str:
    return f"segment_{idx:05d}_{kind}.npy"


class DepthCrafterDepthOnlyStream:
    def __init__(
        self,
        unet_path: str | None = None,
        pre_trained_path: str | None = None,
        cpu_offload_mode: str = "model",
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

        mode = (cpu_offload_mode or "model").lower()
        if mode == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif mode == "model":
            self.pipe.enable_model_cpu_offload()
        elif mode == "none":
            self.pipe.to("cuda")
        else:
            raise ValueError("cpu_offload_mode must be one of: none, model, sequential")

        try:
            self.pipe.disable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def _infer_chunk_latents_with_carry(
        self,
        chunk_frames: np.ndarray,
        guidance_scale: float,
        inference_steps: int,
        decode_chunk_size: int,
        debug_mem: bool,
        prev_overlap_latents: torch.Tensor | None,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
    ) -> torch.Tensor:
        device = self.pipe._execution_device
        chunk_len = int(chunk_frames.shape[0])
        height = int(chunk_frames.shape[1])
        width = int(chunk_frames.shape[2])

        if debug_mem and torch.cuda.is_available():
            _cuda_mem(f"chunk {chunk_index} before carry-pipe [{chunk_start}:{chunk_end})")

        print(
            f"[STREAM-CARRY] chunk {chunk_index}: frames[{chunk_start}:{chunk_end}) "
            f"len={chunk_len} carry={0 if prev_overlap_latents is None else int(prev_overlap_latents.shape[1])}"
        )

        self.pipe.check_inputs(chunk_frames, height, width)
        self.pipe._guidance_scale = float(guidance_scale)

        video = torch.from_numpy(chunk_frames.transpose(0, 3, 1, 2)).to(device=device, dtype=self.pipe.dtype)
        video = video * 2.0 - 1.0

        video_embeddings = self.pipe.encode_video(video, chunk_size=int(decode_chunk_size)).unsqueeze(0)

        noise = randn_tensor(video.shape, generator=None, device=device, dtype=video.dtype)
        video = video + 0.02 * noise

        needs_upcasting = (
            self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
        )
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float32)

        video_latents = self.pipe.encode_vae_video(
            video.to(self.pipe.vae.dtype),
            chunk_size=int(decode_chunk_size),
        ).unsqueeze(0)

        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)

        added_time_ids = self.pipe._get_add_time_ids(
            7,
            127,
            0.02,
            video_embeddings.dtype,
            1,
            1,
            False,
        ).to(device)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, int(inference_steps), device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order
        self.pipe._num_timesteps = len(timesteps)

        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            1,
            chunk_len,
            num_channels_latents,
            height,
            width,
            video_embeddings.dtype,
            device,
            None,
            None,
        )

        carry_count = 0
        if prev_overlap_latents is not None:
            carry_count = min(int(prev_overlap_latents.shape[1]), int(latents.shape[1]))

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if carry_count > 0 and i == 0:
                    prev_gpu = prev_overlap_latents[:, -carry_count:].to(device=device, dtype=latents.dtype)
                    latents[:, :carry_count] = (
                        prev_gpu
                        + latents[:, :carry_count]
                        / self.pipe.scheduler.init_noise_sigma
                        * self.pipe.scheduler.sigmas[i]
                    )

                latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat([latent_model_input, video_latents], dim=2)
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=video_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                if self.pipe.do_classifier_free_guidance:
                    latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, torch.zeros_like(latent_model_input)],
                        dim=2,
                    )
                    noise_pred_uncond = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=torch.zeros_like(video_embeddings),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )

                latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0
                ):
                    progress_bar.update()

        latents_cpu = latents.detach().to("cpu")
        del video
        del video_embeddings
        del video_latents
        del noise
        del latents
        if carry_count > 0:
            del prev_gpu
        if debug_mem and torch.cuda.is_available():
            _cuda_mem(f"chunk {chunk_index} after carry-pipe [{chunk_start}:{chunk_end})")
        _cuda_cleanup()
        return latents_cpu

    @torch.inference_mode()
    def _decode_latent_slice_raw(
        self,
        latents_cpu: torch.Tensor,
        latent_decode_chunk_size: int,
        debug_mem: bool,
        tag: str,
    ) -> np.ndarray:
        if int(latents_cpu.shape[1]) <= 0:
            return np.zeros((0, 0, 0), dtype=np.float32)
        device = getattr(self.pipe, "_execution_device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae_dtype = getattr(self.pipe.vae, "dtype", torch.float16)
        if debug_mem and torch.cuda.is_available():
            _cuda_mem(f"{tag} before decode")
        latents_gpu = latents_cpu.to(device=device, dtype=vae_dtype)
        decoded = self.pipe.decode_latents(
            latents_gpu,
            num_frames=int(latents_gpu.shape[1]),
            decode_chunk_size=int(latent_decode_chunk_size),
        )
        raw_gray = decoded[0].mean(0).detach().to("cpu").numpy().astype(np.float32)
        del latents_gpu
        del decoded
        if debug_mem and torch.cuda.is_available():
            _cuda_mem(f"{tag} after decode")
        _cuda_cleanup()
        return raw_gray

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
        cpu_offload_mode: str = "model",
        process_length: int = -1,
        target_fps: float = -1.0,
        max_res: int = 1024,
        debug_mem: bool = True,
        decode_chunk_size: int = 8,
        temp_dir: str = "",
        keep_temps: bool = False,
        latent_decode_chunk_size: int = 5,
    ):
        del cpu_offload_mode
        if window_size <= overlap:
            raise ValueError("window_size must be > overlap")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        set_seed(int(seed))
        if debug_mem and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _cuda_mem("start infer_to_gray_video_stream_carry")

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
            print(
                f"[DBG] decoded frames: {frames.shape} dtype={frames.dtype} "
                f"fps={fps_str} orig={original_w}x{original_h}"
            )
            _cuda_mem("after read_video_frames (CPU only)")

        num_frames = int(frames.shape[0])
        chunks = _compute_chunks(num_frames, int(window_size), int(overlap))
        if not chunks:
            raise RuntimeError(f"No frames decoded from: {input_video_path}")

        temp_root = Path(temp_dir) if str(temp_dir).strip() else Path(
            tempfile.mkdtemp(prefix="depthcrafter_stream_carry_")
        )
        latent_dir = temp_root / "latent_chunks"
        raw_dir = temp_root / "decoded_segments"
        latent_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"[STREAM-CARRY] temp_root={temp_root}")
        print(f"[STREAM-CARRY] chunks={len(chunks)} window_size={window_size} overlap={overlap}")

        chunk_infos: list[dict[str, object]] = []
        segment_infos: list[dict[str, object]] = []
        global_min = float("inf")
        global_max = float("-inf")
        segment_idx = 0

        def decode_and_store(latent_slice: torch.Tensor, kind: str) -> None:
            nonlocal global_min, global_max, segment_idx
            if int(latent_slice.shape[1]) <= 0:
                return
            raw_gray = self._decode_latent_slice_raw(
                latent_slice,
                latent_decode_chunk_size=int(latent_decode_chunk_size),
                debug_mem=bool(debug_mem),
                tag=f"segment {segment_idx} {kind}",
            )
            seg_min = float(raw_gray.min())
            seg_max = float(raw_gray.max())
            global_min = min(global_min, seg_min)
            global_max = max(global_max, seg_max)
            seg_path = raw_dir / _segment_name(segment_idx, kind)
            np.save(seg_path, raw_gray)
            segment_infos.append(
                {
                    "index": segment_idx,
                    "kind": kind,
                    "path": str(seg_path),
                    "num_frames": int(raw_gray.shape[0]),
                    "raw_min": seg_min,
                    "raw_max": seg_max,
                }
            )
            print(
                f"[STREAM-CARRY] decoded segment {segment_idx} kind={kind} "
                f"frames={raw_gray.shape[0]} raw_min={seg_min:.6f} raw_max={seg_max:.6f}"
            )
            segment_idx += 1

        try:
            pending: torch.Tensor | None = None
            pending_head_trim = 0
            for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
                prev_carry = None
                if pending is not None and int(overlap) > 0 and int(pending.shape[1]) > 0:
                    carry_count = min(int(overlap), int(pending.shape[1]))
                    prev_carry = pending[:, -carry_count:].clone()
                chunk_frames = frames[chunk_start:chunk_end]
                current = self._infer_chunk_latents_with_carry(
                    chunk_frames=chunk_frames,
                    guidance_scale=guidance_scale,
                    inference_steps=inference_steps,
                    decode_chunk_size=decode_chunk_size,
                    debug_mem=debug_mem,
                    prev_overlap_latents=prev_carry,
                    chunk_index=chunk_index,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                chunk_path = latent_dir / f"chunk_{chunk_index:04d}.pt"
                torch.save(current, chunk_path)
                chunk_infos.append(
                    {
                        "index": chunk_index,
                        "start": chunk_start,
                        "end": chunk_end,
                        "path": str(chunk_path),
                        "shape": [int(x) for x in current.shape],
                        "dtype": str(current.dtype),
                    }
                )
                print(
                    f"[STREAM-CARRY] saved chunk {chunk_index} frames[{chunk_start}:{chunk_end}) "
                    f"latent_shape={tuple(current.shape)}"
                )

                if pending is None:
                    pending = current
                    pending_head_trim = 0
                    continue

                current_overlap = min(int(overlap), int(pending.shape[1]), int(current.shape[1]))
                body_end = int(pending.shape[1]) - current_overlap
                if body_end > pending_head_trim:
                    decode_and_store(pending[:, pending_head_trim:body_end], "body")
                if current_overlap > 0:
                    weights = _latent_overlap_weights(current_overlap)
                    blended = pending[:, -current_overlap:].float() * (1.0 - weights) + current[:, :current_overlap].float() * weights
                    decode_and_store(blended, "overlap")
                pending = current
                pending_head_trim = current_overlap

            if pending is None:
                raise RuntimeError("No chunks were processed")
            if int(pending.shape[1]) > pending_head_trim:
                decode_and_store(pending[:, pending_head_trim:], "tail")

            manifest = {
                "input_video_path": str(input_video_path),
                "output_video_path": str(output_video_path),
                "num_frames": num_frames,
                "window_size": int(window_size),
                "overlap": int(overlap),
                "decode_chunk_size": int(decode_chunk_size),
                "latent_decode_chunk_size": int(latent_decode_chunk_size),
                "fps_str": str(fps_str),
                "decoded_shape": [int(x) for x in frames.shape],
                "original_size": [int(original_w), int(original_h)],
                "global_raw_min": float(global_min),
                "global_raw_max": float(global_max),
                "chunks": chunk_infos,
                "segments": segment_infos,
            }
            (temp_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"[STREAM-CARRY] global raw range: min={global_min:.6f} max={global_max:.6f}"
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".", exist_ok=True)
            writer = _start_ffmpeg_gray_writer(
                output_video_path,
                w=int(frames.shape[2]),
                h=int(frames.shape[1]),
                fps_str=fps_str,
                vf=f"scale={original_w}:{original_h}:flags=bilinear",
            )
            if writer.stdin is None:
                raise RuntimeError("ffmpeg writer stdin is not available")

            try:
                for seg in segment_infos:
                    raw = np.load(seg["path"])
                    norm = _normalize_depth_chunk(raw, global_min, global_max)
                    for frame in norm:
                        writer.stdin.write(np.clip(frame * 255.0, 0, 255).astype(np.uint8).tobytes())
            finally:
                try:
                    writer.stdin.close()
                except Exception:
                    pass
                rc = writer.wait()
                if rc != 0:
                    raise RuntimeError(f"ffmpeg exited with code {rc} while writing: {output_video_path}")
        finally:
            if not keep_temps:
                shutil.rmtree(temp_root, ignore_errors=True)

        return output_video_path


def main(
    input_video_path: str,
    output_video_path: str,
    unet_path: str | None = None,
    pre_trained_path: str | None = None,
    guidance_scale: float = 1.0,
    inference_steps: int = 5,
    target_width: int = 1408,
    target_height: int = 768,
    window_size: int = 40,
    overlap: int = 8,
    seed: int = 42,
    cpu_offload_mode: str = "model",
    process_length: int = -1,
    target_fps: float = -1.0,
    max_res: int = 1920,
    debug_mem: bool = False,
    decode_chunk_size: int = 8,
    temp_dir: str = "",
    keep_temps: bool = False,
    latent_decode_chunk_size: int = 5,
):
    if unet_path is None:
        unet_path = "./weights/DepthCrafter"
    if pre_trained_path is None:
        pre_trained_path = "./weights/stable-video-diffusion-img2vid-xt-1-1"

    runner = DepthCrafterDepthOnlyStream(
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
        window_size=window_size,
        overlap=overlap,
        seed=seed,
        cpu_offload_mode=cpu_offload_mode,
        process_length=process_length,
        target_fps=target_fps,
        max_res=max_res,
        debug_mem=debug_mem,
        decode_chunk_size=decode_chunk_size,
        temp_dir=temp_dir,
        keep_temps=keep_temps,
        latent_decode_chunk_size=latent_decode_chunk_size,
    )


if __name__ == "__main__":
    Fire(main)
