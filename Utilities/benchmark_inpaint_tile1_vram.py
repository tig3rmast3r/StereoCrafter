#!/usr/bin/env python3
"""Benchmark StereoCrafter inpaint VRAM by window size and chunk size.

This is intentionally standalone and synthetic: it exercises the same
inpainting pipeline call used by the runtime, but it does not read/write video.
The output CSV is meant as a calibration baseline for deciding when tile1 can
replace tile2 for smaller segment windows.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from transformers import CLIPVisionModelWithProjection

from inpainting_gui import InpaintingGUI, spatial_tiled_process
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    configure_attention_processors,
    load_inpainting_pipeline,
)


DEFAULT_PRETRAINED = REPO_ROOT / "weights" / "stable-video-diffusion-img2vid-xt-1-1"
DEFAULT_UNET = REPO_ROOT / "weights" / "StereoCrafter"


@dataclass(frozen=True)
class Case:
    scale_pct: int
    width: int
    height: int
    chunk: int


@dataclass(frozen=True)
class SampleClip:
    frames: torch.Tensor
    mask: torch.Tensor | None


RETRY_PROFILE_ORDER = ("run", "retry1", "retry2", "retry3")
RETRY_MAX_SPLIT_CHOICES = {64, 128, 256, 512}


def _norm_bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(fallback)


def _norm_max_split(value: object) -> int | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s == "off":
        return None
    try:
        parsed = int(float(s))
    except Exception:
        return None
    return parsed if parsed in RETRY_MAX_SPLIT_CHOICES else None


def _default_retry_profiles(base_offload: str) -> list[dict[str, object]]:
    return [
        {
            "name": "run",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": None,
            "cpu_offload_mode": base_offload,
        },
        {
            "name": "retry1",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 256,
            "cpu_offload_mode": base_offload,
        },
        {
            "name": "retry2",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 64,
            "cpu_offload_mode": base_offload,
        },
        {
            "name": "retry3",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 64,
            "cpu_offload_mode": "sequential",
        },
    ]


def parse_retry_profiles(policy_json: str, base_offload: str) -> list[dict[str, object]]:
    defaults = _default_retry_profiles(base_offload)
    txt = str(policy_json or "").strip()
    if not txt:
        return defaults
    try:
        raw = json.loads(txt)
    except Exception as exc:
        print(f"[WARN] retry policy parse failed: {exc}. Using benchmark defaults.", flush=True)
        return defaults
    if not isinstance(raw, dict):
        print("[WARN] retry policy is not an object. Using benchmark defaults.", flush=True)
        return defaults
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
                    node.get("garbage_collection_threshold", base["garbage_collection_threshold"]),
                    bool(base["garbage_collection_threshold"]),
                ),
                "expandable_segments": _norm_bool(
                    node.get("expandable_segments", base["expandable_segments"]),
                    bool(base["expandable_segments"]),
                ),
                "max_split_size_mb": _norm_max_split(
                    node.get("max_split_size_mb", base["max_split_size_mb"])
                ),
                "cpu_offload_mode": str(node.get("cpu_offload_mode", base["cpu_offload_mode"]) or base_offload),
            }
        )
    return out


def allocator_conf_from_profile(profile: dict[str, object]) -> str:
    parts: list[str] = []
    if _norm_bool(profile.get("garbage_collection_threshold"), True):
        parts.append("garbage_collection_threshold:0.8")
    if _norm_bool(profile.get("expandable_segments"), True):
        parts.append("expandable_segments:True")
    max_split = _norm_max_split(profile.get("max_split_size_mb"))
    if max_split is not None:
        parts.append(f"max_split_size_mb:{max_split}")
    return ",".join(parts)


def apply_allocator_conf(conf: str) -> None:
    conf_s = str(conf or "").strip()
    if conf_s:
        os.environ["PYTORCH_ALLOC_CONF"] = conf_s
    else:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    try:
        alt = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if callable(alt):
            alt(conf_s)
            return
        setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
        if callable(setter):
            setter(conf_s)
    except Exception as exc:
        print(f"[WARN] failed to apply allocator settings '{conf_s or 'default'}': {exc}", flush=True)


class NvidiaSmiMonitor:
    def __init__(self, gpu_index: int, interval_sec: float = 0.05) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_sec = max(0.01, float(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []

    def __enter__(self) -> "NvidiaSmiMonitor":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> int:
        return max(self.samples) if self.samples else -1

    def _poll(self) -> None:
        while not self._stop.is_set():
            used = query_gpu_used_mb(self.gpu_index)
            if used >= 0:
                self.samples.append(used)
            time.sleep(self.interval_sec)


def query_gpu_used_mb(gpu_index: int) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return -1
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        return int(first)
    except ValueError:
        return -1


def query_gpu_total_mb(gpu_index: int) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return -1
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        return int(first)
    except ValueError:
        return -1


def align_down(value: float, step: int = 8, minimum: int = 8) -> int:
    aligned = int(value) // step * step
    return max(minimum, aligned)


def parse_chunks(value: str) -> list[int]:
    chunks: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            if len(bits) not in (2, 3):
                raise argparse.ArgumentTypeError(f"invalid chunk range: {part}")
            start, stop = bits[0], bits[1]
            step = bits[2] if len(bits) == 3 else 1
            if step <= 0:
                raise argparse.ArgumentTypeError(f"invalid chunk step: {part}")
            chunks.update(range(start, stop + 1, step))
        else:
            chunks.add(int(part))
    if not chunks:
        raise argparse.ArgumentTypeError("at least one chunk is required")
    return sorted(chunks)


def iter_scales(max_pct: int, min_pct: int, step_pct: int) -> Iterable[int]:
    if step_pct <= 0:
        raise ValueError("step_pct must be > 0")
    current = int(max_pct)
    while current >= int(min_pct):
        yield current
        current -= int(step_pct)


def build_cases(
    baseline_width: int,
    baseline_height: int,
    scale_max_pct: int,
    scale_min_pct: int,
    scale_step_pct: int,
    chunks: list[int],
    low_scale_min_chunk_pct: int,
    low_scale_min_chunk: int,
) -> list[Case]:
    cases: list[Case] = []
    for scale_pct in iter_scales(scale_max_pct, scale_min_pct, scale_step_pct):
        scale = scale_pct / 100.0
        width = align_down(baseline_width * scale)
        height = align_down(baseline_height * scale)
        for chunk in chunks:
            if low_scale_min_chunk_pct > 0 and scale_pct <= low_scale_min_chunk_pct and chunk < low_scale_min_chunk:
                continue
            cases.append(Case(scale_pct=scale_pct, width=width, height=height, chunk=chunk))
    return cases


def build_dynamic_case(
    baseline_width: int,
    baseline_height: int,
    scale_pct: int,
    chunk: int,
) -> Case:
    width, height = InpaintingGUI._dynamic_resolution_target_size(
        baseline_width,
        baseline_height,
        float(scale_pct) / 100.0,
        1,
    )
    return Case(scale_pct=int(scale_pct), width=width, height=height, chunk=int(chunk))


def find_replace_mask_for_input(input_path: str, replace_mask_folder: str) -> str:
    if not input_path:
        return ""
    stem = Path(input_path).stem
    mask_dir = Path(replace_mask_folder).resolve() if replace_mask_folder else Path(input_path).resolve().parent
    hits = sorted(glob.glob(str(mask_dir / f"{stem}_replace_mask.*")))
    return hits[0] if hits else ""


def _read_video_frames(path: str, frame_count: int) -> np.ndarray:
    vr = VideoReader(str(path), ctx=cpu(0))
    if len(vr) <= 0:
        raise RuntimeError(f"sample video has no frames: {path}")
    idxs = [i % len(vr) for i in range(max(1, int(frame_count)))]
    return vr.get_batch(idxs).asnumpy()


def load_sample_clip(input_path: str, mask_path: str, frame_count: int) -> SampleClip:
    frame_np = _read_video_frames(input_path, frame_count)
    if frame_np.ndim != 4:
        raise RuntimeError(f"sample input has invalid shape {frame_np.shape}: {input_path}")
    if frame_np.shape[-1] >= 3:
        frame_np = frame_np[..., :3]
    elif frame_np.shape[-1] == 1:
        frame_np = np.repeat(frame_np, 3, axis=-1)
    else:
        raise RuntimeError(f"sample input has invalid channel count {frame_np.shape[-1]}: {input_path}")
    frames = torch.from_numpy(np.ascontiguousarray(frame_np)).permute(0, 3, 1, 2).float().div_(255.0)

    mask = None
    if mask_path:
        mask_np = _read_video_frames(mask_path, frame_count)
        if mask_np.ndim == 4:
            if mask_np.shape[-1] >= 3:
                mask_gray = mask_np[..., :3].mean(axis=-1)
            elif mask_np.shape[-1] == 1:
                mask_gray = mask_np[..., 0]
            else:
                raise RuntimeError(f"sample mask has invalid channel count {mask_np.shape[-1]}: {mask_path}")
        elif mask_np.ndim == 3:
            mask_gray = mask_np
        else:
            raise RuntimeError(f"sample mask has invalid shape {mask_np.shape}: {mask_path}")
        mask = torch.from_numpy(np.ascontiguousarray(mask_gray)).unsqueeze(1).float().div_(255.0).clamp_(0.0, 1.0)
    return SampleClip(frames=frames, mask=mask)


def sample_tensors_for_case(sample: SampleClip, frame_count: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    frames = sample.frames[:frame_count]
    if frames.shape[0] < frame_count:
        repeats = int(np.ceil(frame_count / max(1, frames.shape[0])))
        frames = frames.repeat((repeats, 1, 1, 1))[:frame_count]
    frames = F.interpolate(frames, size=(height, width), mode="area")

    if sample.mask is None:
        mask = torch.ones((frame_count, 1, height, width), dtype=torch.float32)
    else:
        mask = sample.mask[:frame_count]
        if mask.shape[0] < frame_count:
            repeats = int(np.ceil(frame_count / max(1, mask.shape[0])))
            mask = mask.repeat((repeats, 1, 1, 1))[:frame_count]
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False).clamp_(0.0, 1.0)
    return frames.contiguous(), mask.contiguous()


def load_pipeline_flexible(
    pretrained_path: Path,
    unet_path: Path,
    device: str,
    dtype: torch.dtype,
    offload_type: str,
) -> StableVideoDiffusionInpaintingPipeline:
    """Load current runtime pipeline, accepting both historical UNet layouts."""
    if (unet_path / "unet_diffusers").exists():
        return load_inpainting_pipeline(
            pre_trained_path=str(pretrained_path),
            unet_path=str(unet_path),
            device=device,
            dtype=dtype,
            offload_type=offload_type,
        )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(pretrained_path),
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=dtype,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        str(pretrained_path),
        subfolder="vae",
        variant="fp16",
        torch_dtype=dtype,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        str(unet_path),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    pipe = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        str(pretrained_path),
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=dtype,
    ).to(device)
    configure_attention_processors(pipe)
    if offload_type == "model":
        pipe.enable_model_cpu_offload()
    elif offload_type == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload_type == "none":
        pass
    else:
        raise ValueError(f"invalid offload_type: {offload_type}")
    return pipe


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_case(
    pipeline: StableVideoDiffusionInpaintingPipeline,
    case: Case,
    tile_num: int,
    steps: int,
    decode_chunk_size: int,
    gpu_index: int,
    monitor_interval_sec: float,
    seed: int,
    include_decode: bool,
    tail_pad: int = 0,
    overlap: int = 2,
    two_chunk_runtime: bool = True,
    sample_clip: SampleClip | None = None,
    synthetic_mask_ratio: float = 0.20,
) -> dict[str, object]:
    release_cuda()
    torch.cuda.reset_peak_memory_stats()
    model_frames = int(case.chunk) + max(0, int(tail_pad))
    first_model_frames = max(1, int(case.chunk) - max(0, int(overlap)) + max(0, int(tail_pad)))
    chunk_lengths = [first_model_frames, model_frames] if two_chunk_runtime else [model_frames]
    torch.manual_seed(int(seed) + case.width * 3 + case.height * 5 + model_frames * 7)

    base_used_mb = query_gpu_used_mb(gpu_index)
    if sample_clip is not None:
        frames, mask = sample_tensors_for_case(sample_clip, max(chunk_lengths), case.height, case.width)
    else:
        frames = torch.rand((max(chunk_lengths), 3, case.height, case.width), dtype=torch.float32)
        mask = torch.zeros((max(chunk_lengths), 1, case.height, case.width), dtype=torch.float32)
        active_w = max(1, min(case.width, int(round(case.width * max(0.0, min(1.0, synthetic_mask_ratio))))))
        start_x = max(0, (case.width - active_w) // 2)
        mask[:, :, :, start_x : start_x + active_w] = 1.0

    started = time.time()
    status = "ok"
    error = ""
    torch_peak_allocated_mb = -1
    torch_peak_reserved_mb = -1
    peak_gpu_used_mb = -1
    try:
        with NvidiaSmiMonitor(gpu_index=gpu_index, interval_sec=monitor_interval_sec) as mon:
            with torch.inference_mode():
                for chunk_len in chunk_lengths:
                    latents = spatial_tiled_process(
                        cond_frames=frames[:chunk_len],
                        mask_frames=mask[:chunk_len],
                        process_func=pipeline,
                        tile_num=tile_num,
                        spatial_n_compress=8,
                        min_guidance_scale=1.01,
                        max_guidance_scale=1.01,
                        decode_chunk_size=decode_chunk_size,
                        fps=7,
                        motion_bucket_id=127,
                        noise_aug_strength=0.0,
                        num_inference_steps=steps,
                    )
                    if include_decode:
                        latents = latents.unsqueeze(0)
                        pipeline.vae.to(dtype=torch.float16)
                        decoded = pipeline.decode_latents(
                            latents,
                            num_frames=latents.shape[1],
                            decode_chunk_size=decode_chunk_size,
                        )
                        del decoded
                    del latents
                    release_cuda()
            peak_gpu_used_mb = mon.peak_mb
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc).replace("\n", " ")[:500]
        peak_gpu_used_mb = query_gpu_used_mb(gpu_index)
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        peak_gpu_used_mb = query_gpu_used_mb(gpu_index)
    finally:
        if torch.cuda.is_available():
            torch_peak_allocated_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024)
            torch_peak_reserved_mb = int(torch.cuda.max_memory_reserved() / 1024 / 1024)
        del frames, mask
        release_cuda()

    elapsed = time.time() - started
    return {
        "scale_pct": case.scale_pct,
        "width": case.width,
        "height": case.height,
        "pixels": case.width * case.height,
        "pixel_pct_of_baseline": "",
        "chunk": case.chunk,
        "model_frames": model_frames,
        "first_model_frames": first_model_frames,
        "bench_chunks": len(chunk_lengths),
        "tile_num": tile_num,
        "steps": steps,
        "decode_chunk_size": decode_chunk_size,
        "include_decode": int(include_decode),
        "sample_source": int(sample_clip is not None),
        "synthetic_mask_ratio": f"{synthetic_mask_ratio:.6f}" if sample_clip is None else "",
        "retry_profile": "",
        "allocator_conf": "",
        "status": status,
        "elapsed_sec": f"{elapsed:.3f}",
        "gpu_base_used_mb": base_used_mb,
        "peak_gpu_used_mb": peak_gpu_used_mb,
        "peak_gpu_delta_mb": peak_gpu_used_mb - base_used_mb if peak_gpu_used_mb >= 0 and base_used_mb >= 0 else -1,
        "torch_peak_allocated_mb": torch_peak_allocated_mb,
        "torch_peak_reserved_mb": torch_peak_reserved_mb,
        "error": error,
    }


def write_summary(raw_csv: Path, summary_csv: Path, vram_limit_mb: int) -> None:
    with raw_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["scale_pct"], []).append(row)

    fieldnames = [
        "scale_pct",
        "width",
        "height",
        "pixels",
        "safe_max_chunk",
        "first_over_limit_chunk",
        "first_error_chunk",
        "best_safe_peak_gpu_used_mb",
        "vram_limit_mb",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scale_pct in sorted(grouped, key=lambda x: int(x), reverse=True):
            items = sorted(grouped[scale_pct], key=lambda x: int(x["chunk"]))
            safe = [
                r for r in items
                if r["status"] == "ok"
                and int(float(r["peak_gpu_used_mb"])) >= 0
                and int(float(r["peak_gpu_used_mb"])) <= vram_limit_mb
            ]
            over = [
                r for r in items
                if r["status"] == "ok"
                and int(float(r["peak_gpu_used_mb"])) > vram_limit_mb
            ]
            errors = [r for r in items if r["status"] != "ok"]
            best = safe[-1] if safe else None
            first = items[0]
            writer.writerow(
                {
                    "scale_pct": scale_pct,
                    "width": first["width"],
                    "height": first["height"],
                    "pixels": first["pixels"],
                    "safe_max_chunk": best["chunk"] if best else "",
                    "first_over_limit_chunk": over[0]["chunk"] if over else "",
                    "first_error_chunk": errors[0]["chunk"] if errors else "",
                    "best_safe_peak_gpu_used_mb": best["peak_gpu_used_mb"] if best else "",
                    "vram_limit_mb": vram_limit_mb,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark inpaint tile1 VRAM over scaled window sizes and chunk sizes."
    )
    parser.add_argument("--out-csv", default="/data/workdp/tile1_vram_benchmark.csv")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--json-out", default=str(REPO_ROOT / "inpaint_tile1_chunk_benchmark.json"))
    parser.add_argument("--baseline-width", type=int, default=1920)
    parser.add_argument("--baseline-height", type=int, default=800)
    parser.add_argument("--scale-max-pct", type=int, default=100)
    parser.add_argument("--scale-min-pct", type=int, default=25)
    parser.add_argument("--scale-step-pct", type=int, default=5)
    parser.add_argument("--chunks", type=parse_chunks, default=parse_chunks("22,26,30,34,38,42,46,50,55"))
    parser.add_argument("--adaptive-oom", action="store_true")
    parser.add_argument("--adaptive-scales", default="100,90,80,70,60,50")
    parser.add_argument("--adaptive-tiles", default="1", help="Comma-separated tile modes to benchmark in adaptive mode.")
    parser.add_argument("--adaptive-start-chunk", type=int, default=22)
    parser.add_argument("--adaptive-start-chunk-tile1", type=int, default=20)
    parser.add_argument("--adaptive-start-chunk-tile2", type=int, default=50)
    parser.add_argument("--adaptive-step", type=int, default=5)
    parser.add_argument("--adaptive-max-chunk", type=int, default=120)
    parser.add_argument(
        "--low-scale-min-chunk-pct",
        type=int,
        default=0,
        help="If >0, scales <= this percent skip chunks below --low-scale-min-chunk.",
    )
    parser.add_argument(
        "--low-scale-min-chunk",
        type=int,
        default=50,
        help="Minimum chunk to test for scales covered by --low-scale-min-chunk-pct.",
    )
    parser.add_argument("--tile-num", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--decode-chunk-size", type=int, default=2)
    parser.add_argument("--offload-type", choices=["none", "model", "sequential"], default="model")
    parser.add_argument("--pretrained-path", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--unet-path", default=str(DEFAULT_UNET))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--vram-limit-mb", type=int, default=23000)
    parser.add_argument("--monitor-interval-sec", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-decode", action="store_true")
    parser.add_argument("--tail-pad", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--sample-input", default="", help="Optional real splatted/input clip used instead of synthetic random frames.")
    parser.add_argument("--sample-mask", default="", help="Optional real replace mask clip used with --sample-input.")
    parser.add_argument(
        "--sample-mask-folder",
        default="",
        help="Optional folder to auto-resolve <sample_input_stem>_replace_mask.* when --sample-mask is empty.",
    )
    parser.add_argument(
        "--synthetic-mask-ratio",
        type=float,
        default=0.20,
        help="Fallback synthetic mask width ratio when no real sample mask is used.",
    )
    parser.add_argument(
        "--retry-policy-json",
        default="",
        help="Pipeline Master retry policy JSON; benchmark uses --allocator-profile from it.",
    )
    parser.add_argument(
        "--allocator-profile",
        default="run",
        choices=RETRY_PROFILE_ORDER,
        help="Allocator profile to apply for benchmark probes. Default matches normal Pipeline Master run.",
    )
    parser.add_argument(
        "--single-chunk",
        action="store_true",
        help="Benchmark one max-size chunk only. Default runs first+second runtime-like chunks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write planned cases only; do not load the model.")
    parser.add_argument(
        "--stop-scale-after-limit",
        action="store_true",
        help="For each scale, stop trying larger chunks after VRAM limit, OOM, or error.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    out_csv = Path(args.out_csv).resolve()
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else out_csv.with_name(out_csv.stem + "_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    cases = build_cases(
        baseline_width=args.baseline_width,
        baseline_height=args.baseline_height,
        scale_max_pct=args.scale_max_pct,
        scale_min_pct=args.scale_min_pct,
        scale_step_pct=args.scale_step_pct,
        chunks=args.chunks,
        low_scale_min_chunk_pct=args.low_scale_min_chunk_pct,
        low_scale_min_chunk=args.low_scale_min_chunk,
    )
    baseline_pixels = args.baseline_width * args.baseline_height
    gpu_total_mb = query_gpu_total_mb(args.gpu_index)

    fieldnames = [
        "scale_pct",
        "width",
        "height",
        "pixels",
        "pixel_pct_of_baseline",
        "chunk",
        "model_frames",
        "first_model_frames",
        "bench_chunks",
        "tile_num",
        "steps",
        "decode_chunk_size",
        "include_decode",
        "sample_source",
        "synthetic_mask_ratio",
        "retry_profile",
        "allocator_conf",
        "status",
        "elapsed_sec",
        "gpu_total_mb",
        "gpu_base_used_mb",
        "peak_gpu_used_mb",
        "peak_gpu_delta_mb",
        "torch_peak_allocated_mb",
        "torch_peak_reserved_mb",
        "vram_limit_mb",
        "offload_type",
        "error",
    ]

    pipeline = None
    retry_profiles = parse_retry_profiles(args.retry_policy_json, args.offload_type)
    selected_profile = next(
        (p for p in retry_profiles if str(p.get("name")) == str(args.allocator_profile)),
        retry_profiles[0],
    )
    allocator_conf = allocator_conf_from_profile(selected_profile)
    apply_allocator_conf(allocator_conf)
    print(
        f"[ALLOC] profile={selected_profile.get('name')} "
        f"offload={selected_profile.get('cpu_offload_mode')} "
        f"conf={allocator_conf or 'default'}",
        flush=True,
    )
    sample_clip = None
    sample_input = str(args.sample_input or "").strip()
    sample_mask = str(args.sample_mask or "").strip()
    if sample_input and not sample_mask:
        sample_mask = find_replace_mask_for_input(sample_input, str(args.sample_mask_folder or "").strip())
    if sample_input:
        sample_frames = max(int(args.adaptive_max_chunk), max(int(c.chunk) for c in cases)) + max(0, int(args.tail_pad))
        sample_clip = load_sample_clip(sample_input, sample_mask, sample_frames)
        print(
            f"[SAMPLE] input={sample_input} mask={sample_mask or '(none)'} "
            f"frames={sample_clip.frames.shape[0]}",
            flush=True,
        )
    if not args.dry_run:
        pipeline = load_pipeline_flexible(
            pretrained_path=Path(args.pretrained_path).resolve(),
            unet_path=Path(args.unet_path).resolve(),
            device="cuda",
            dtype=torch.float16,
            offload_type=args.offload_type,
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if args.adaptive_oom:
            scale_values = [int(x.strip()) for x in str(args.adaptive_scales).split(",") if x.strip()]
            tile_values = [int(x.strip()) for x in str(args.adaptive_tiles).split(",") if x.strip()]
            if not tile_values:
                tile_values = [int(args.tile_num)]
            benchmark_chunks_by_tile: dict[int, list[int]] = {}
            bench_chunks = 1 if args.single_chunk else 2

            def _planned_row(case: Case, tile_num: int, model_frames: int, first_model_frames: int) -> dict[str, object]:
                return {
                    "scale_pct": case.scale_pct,
                    "width": case.width,
                    "height": case.height,
                    "pixels": case.width * case.height,
                    "pixel_pct_of_baseline": f"{(case.width * case.height / baseline_pixels) * 100.0:.6f}",
                    "chunk": case.chunk,
                    "model_frames": model_frames,
                    "first_model_frames": first_model_frames,
                    "bench_chunks": bench_chunks,
                    "tile_num": tile_num,
                    "steps": args.steps,
                    "decode_chunk_size": args.decode_chunk_size,
                    "include_decode": int(args.include_decode),
                    "sample_source": int(sample_clip is not None),
                    "synthetic_mask_ratio": f"{float(args.synthetic_mask_ratio):.6f}" if sample_clip is None else "",
                    "retry_profile": selected_profile.get("name", ""),
                    "allocator_conf": allocator_conf,
                    "status": "planned",
                    "elapsed_sec": "",
                    "gpu_total_mb": gpu_total_mb,
                    "gpu_base_used_mb": "",
                    "peak_gpu_used_mb": "",
                    "peak_gpu_delta_mb": "",
                    "torch_peak_allocated_mb": "",
                    "torch_peak_reserved_mb": "",
                    "vram_limit_mb": args.vram_limit_mb,
                    "offload_type": args.offload_type,
                    "error": "",
                }

            def _run_adaptive_case(case: Case, tile_num: int) -> dict[str, object]:
                model_frames = case.chunk + int(args.tail_pad)
                first_model_frames = max(1, case.chunk - int(args.overlap) + int(args.tail_pad))
                if args.dry_run:
                    return _planned_row(case, tile_num, model_frames, first_model_frames)
                assert pipeline is not None
                row = run_case(
                    pipeline=pipeline,
                    case=case,
                    tile_num=tile_num,
                    steps=args.steps,
                    decode_chunk_size=args.decode_chunk_size,
                    gpu_index=args.gpu_index,
                    monitor_interval_sec=args.monitor_interval_sec,
                    seed=args.seed,
                    include_decode=args.include_decode,
                    tail_pad=args.tail_pad,
                    overlap=args.overlap,
                    two_chunk_runtime=not args.single_chunk,
                    sample_clip=sample_clip,
                    synthetic_mask_ratio=args.synthetic_mask_ratio,
                )
                row["pixel_pct_of_baseline"] = f"{(case.width * case.height / baseline_pixels) * 100.0:.6f}"
                row["gpu_total_mb"] = gpu_total_mb
                row["vram_limit_mb"] = args.vram_limit_mb
                row["offload_type"] = args.offload_type
                row["retry_profile"] = selected_profile.get("name", "")
                row["allocator_conf"] = allocator_conf
                return row

            for tile_index, tile_num in enumerate(tile_values, start=1):
                measured_safe_chunks: list[int] = []
                default_chunks: list[int] = []
                if int(tile_num) == 1:
                    tile_start_chunk = int(args.adaptive_start_chunk_tile1)
                elif int(tile_num) == 2:
                    tile_start_chunk = int(args.adaptive_start_chunk_tile2)
                else:
                    tile_start_chunk = int(args.adaptive_start_chunk)
                current_start = max(1, tile_start_chunk)
                for scale_index, scale_pct in enumerate(scale_values, start=1):
                    last_ok = 0
                    first_bad = None
                    chunk = current_start
                    while chunk <= int(args.adaptive_max_chunk):
                        case = build_dynamic_case(args.baseline_width, args.baseline_height, scale_pct, chunk)
                        model_frames = case.chunk + int(args.tail_pad)
                        first_model_frames = max(1, case.chunk - int(args.overlap) + int(args.tail_pad))
                        print(
                            f"[ADAPT tile={tile_num} {tile_index}/{len(tile_values)} "
                            f"scale={scale_pct}% {scale_index}/{len(scale_values)}] "
                            f"size={case.width}x{case.height} chunk={case.chunk} "
                            f"first_model={first_model_frames} model={model_frames} "
                            f"bench_chunks={bench_chunks}",
                            flush=True,
                        )
                        row = _run_adaptive_case(case, tile_num)
                        if not args.dry_run:
                            print(
                                f"    status={row['status']} peak={row['peak_gpu_used_mb']}MB "
                                f"delta={row['peak_gpu_delta_mb']}MB elapsed={row['elapsed_sec']}s",
                                flush=True,
                            )
                        writer.writerow(row)
                        f.flush()
                        if row["status"] == "ok" or args.dry_run:
                            last_ok = chunk
                            chunk += max(1, int(args.adaptive_step))
                        else:
                            first_bad = chunk
                            break
                    if first_bad is not None and last_ok == 0:
                        for down_chunk in range(first_bad - 1, 0, -max(1, int(args.adaptive_step))):
                            case = build_dynamic_case(args.baseline_width, args.baseline_height, scale_pct, down_chunk)
                            model_frames = case.chunk + int(args.tail_pad)
                            first_model_frames = max(1, case.chunk - int(args.overlap) + int(args.tail_pad))
                            print(
                                f"[DESCEND tile={tile_num}] scale={scale_pct}% size={case.width}x{case.height} "
                                f"chunk={case.chunk} first_model={first_model_frames} "
                                f"model={model_frames} bench_chunks={bench_chunks}",
                                flush=True,
                            )
                            row = _run_adaptive_case(case, tile_num)
                            if not args.dry_run:
                                print(
                                    f"    status={row['status']} peak={row['peak_gpu_used_mb']}MB "
                                    f"delta={row['peak_gpu_delta_mb']}MB elapsed={row['elapsed_sec']}s",
                                    flush=True,
                                )
                            writer.writerow(row)
                            f.flush()
                            if row["status"] == "ok" or args.dry_run:
                                last_ok = down_chunk
                                break
                    if first_bad is not None and last_ok > 0:
                        for refined in range(last_ok + 1, first_bad):
                            case = build_dynamic_case(args.baseline_width, args.baseline_height, scale_pct, refined)
                            model_frames = case.chunk + int(args.tail_pad)
                            first_model_frames = max(1, case.chunk - int(args.overlap) + int(args.tail_pad))
                            print(
                                f"[REFINE tile={tile_num}] scale={scale_pct}% size={case.width}x{case.height} "
                                f"chunk={case.chunk} first_model={first_model_frames} "
                                f"model={model_frames} bench_chunks={bench_chunks}",
                                flush=True,
                            )
                            row = _run_adaptive_case(case, tile_num)
                            if not args.dry_run:
                                print(
                                    f"    status={row['status']} peak={row['peak_gpu_used_mb']}MB "
                                    f"delta={row['peak_gpu_delta_mb']}MB elapsed={row['elapsed_sec']}s",
                                    flush=True,
                                )
                            writer.writerow(row)
                            f.flush()
                            if row["status"] == "ok" or args.dry_run:
                                last_ok = refined
                            else:
                                break
                    measured_safe = last_ok if last_ok > 0 else 0
                    default_chunk = max(1, measured_safe - 2) if measured_safe > 0 else 1
                    measured_safe_chunks.append(measured_safe)
                    default_chunks.append(default_chunk)
                    current_start = max(1, measured_safe)
                    print(
                        f"[ADAPT] tile={tile_num} scale={scale_pct}% "
                        f"measured_safe_chunk={measured_safe} default_chunk={default_chunk}",
                        flush=True,
                    )
                    if measured_safe <= 0:
                        print(
                            f"[ADAPT][WARN] tile={tile_num} scale={scale_pct}% no safe chunk found; "
                            f"using conservative default_chunk={default_chunk}",
                            flush=True,
                        )
                benchmark_chunks_by_tile[tile_num] = default_chunks
            json_out = Path(args.json_out).resolve()
            payload = {
                "schema": "stereocrafter.inpaint_tile_chunk_benchmark.v3",
                "default_chunk_offset": -2,
                "scale_pcts": scale_values,
                "tail_pad": int(args.tail_pad),
                "overlap": int(args.overlap),
                "bench_chunks": bench_chunks,
                "sample_input": sample_input,
                "sample_mask": sample_mask,
                "synthetic_mask_ratio": float(args.synthetic_mask_ratio),
                "retry_profile": selected_profile.get("name", ""),
                "allocator_conf": allocator_conf,
                "steps": int(args.steps),
                "include_decode": bool(args.include_decode),
                "offload_type": args.offload_type,
                "baseline_width": int(args.baseline_width),
                "baseline_height": int(args.baseline_height),
                "updated_at": int(time.time()),
            }
            for tile_num, chunks in benchmark_chunks_by_tile.items():
                payload[f"tile{tile_num}_max_chunks"] = chunks
            json_out.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[DONE] raw={out_csv}")
            print(f"[DONE] benchmark={json_out}")
            return 0
        current_scale = None
        skip_rest_of_scale = False
        for idx, case in enumerate(cases, start=1):
            if current_scale != case.scale_pct:
                current_scale = case.scale_pct
                skip_rest_of_scale = False
            if skip_rest_of_scale:
                continue

            print(
                f"[{idx}/{len(cases)}] scale={case.scale_pct}% "
                f"size={case.width}x{case.height} chunk={case.chunk} "
                f"first_model={max(1, case.chunk - int(args.overlap) + int(args.tail_pad))} "
                f"model={case.chunk + int(args.tail_pad)} "
                f"bench_chunks={1 if args.single_chunk else 2} tile={args.tile_num}",
                flush=True,
            )
            if args.dry_run:
                model_frames = case.chunk + int(args.tail_pad)
                first_model_frames = max(1, case.chunk - int(args.overlap) + int(args.tail_pad))
                row = {
                    "scale_pct": case.scale_pct,
                    "width": case.width,
                    "height": case.height,
                    "pixels": case.width * case.height,
                    "pixel_pct_of_baseline": f"{(case.width * case.height / baseline_pixels) * 100.0:.6f}",
                    "chunk": case.chunk,
                    "model_frames": model_frames,
                    "first_model_frames": first_model_frames,
                    "bench_chunks": 1 if args.single_chunk else 2,
                    "tile_num": args.tile_num,
                    "steps": args.steps,
                    "decode_chunk_size": args.decode_chunk_size,
                    "include_decode": int(args.include_decode),
                    "sample_source": int(sample_clip is not None),
                    "synthetic_mask_ratio": f"{float(args.synthetic_mask_ratio):.6f}" if sample_clip is None else "",
                    "retry_profile": selected_profile.get("name", ""),
                    "allocator_conf": allocator_conf,
                    "status": "planned",
                    "elapsed_sec": "",
                    "gpu_total_mb": gpu_total_mb,
                    "gpu_base_used_mb": "",
                    "peak_gpu_used_mb": "",
                    "peak_gpu_delta_mb": "",
                    "torch_peak_allocated_mb": "",
                    "torch_peak_reserved_mb": "",
                    "vram_limit_mb": args.vram_limit_mb,
                    "offload_type": args.offload_type,
                    "error": "",
                }
            else:
                assert pipeline is not None
                row = run_case(
                    pipeline=pipeline,
                    case=case,
                    tile_num=args.tile_num,
                    steps=args.steps,
                    decode_chunk_size=args.decode_chunk_size,
                    gpu_index=args.gpu_index,
                    monitor_interval_sec=args.monitor_interval_sec,
                    seed=args.seed,
                    include_decode=args.include_decode,
                    tail_pad=args.tail_pad,
                    overlap=args.overlap,
                    two_chunk_runtime=not args.single_chunk,
                    sample_clip=sample_clip,
                    synthetic_mask_ratio=args.synthetic_mask_ratio,
                )
                row["pixel_pct_of_baseline"] = f"{(case.width * case.height / baseline_pixels) * 100.0:.6f}"
                row["gpu_total_mb"] = gpu_total_mb
                row["vram_limit_mb"] = args.vram_limit_mb
                row["offload_type"] = args.offload_type
                row["retry_profile"] = selected_profile.get("name", "")
                row["allocator_conf"] = allocator_conf

                peak = int(row["peak_gpu_used_mb"])
                print(
                    f"    status={row['status']} peak={peak}MB "
                    f"delta={row['peak_gpu_delta_mb']}MB elapsed={row['elapsed_sec']}s",
                    flush=True,
                )
                if args.stop_scale_after_limit and (
                    row["status"] != "ok" or (peak >= 0 and peak > args.vram_limit_mb)
                ):
                    skip_rest_of_scale = True

            writer.writerow(row)
            f.flush()

    if not args.dry_run:
        write_summary(out_csv, summary_csv, args.vram_limit_mb)
        print(f"[DONE] raw={out_csv}")
        print(f"[DONE] summary={summary_csv}")
    else:
        print(f"[DONE] dry-run plan={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
