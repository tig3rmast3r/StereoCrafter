from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache


FFMPEG_CODEC_CHOICES = ("libx264", "libx265", "h264_nvenc", "hevc_nvenc")
GLOBAL_ENCODER_MODE_CHOICES = ("lossless", "crf/qp 0", "crf/qp 1")


def _codec_aliases() -> dict[str, str]:
    return {
        "x264": "libx264",
        "x265": "libx265",
        "h265": "libx265",
    }


@dataclass(frozen=True)
class FFmpegEncodingProfile:
    name: str
    codec: str
    generated_args: tuple[str, ...]
    summary: str
    synthetic_source: str


def normalize_codec(codec: str, fallback: str = "libx264") -> str:
    raw = str(codec or "").strip().lower()
    aliases = _codec_aliases()
    raw = aliases.get(raw, raw)
    if raw in FFMPEG_CODEC_CHOICES:
        return raw
    fb = str(fallback or "libx264").strip().lower()
    fb = aliases.get(fb, fb)
    if fb in FFMPEG_CODEC_CHOICES:
        return fb
    return "libx264"


def normalize_codec_strict(codec: str) -> str:
    raw = str(codec or "").strip().lower()
    aliases = _codec_aliases()
    raw = aliases.get(raw, raw)
    if raw in FFMPEG_CODEC_CHOICES:
        return raw
    raise RuntimeError(f"Unsupported ffmpeg codec: {codec}")


def normalize_global_encoder_mode(mode: str, fallback: str = "lossless") -> str:
    raw = str(mode or "").strip().lower()
    if raw in {"lossless", "crf/qp 0", "crf/qp 1"}:
        return raw
    if raw in {"0", "crf0", "qp0", "crf/qp0"}:
        return "crf/qp 0"
    if raw in {"1", "crf1", "qp1", "crf/qp1"}:
        return "crf/qp 1"
    fb = str(fallback or "lossless").strip().lower()
    if fb in GLOBAL_ENCODER_MODE_CHOICES:
        return fb
    return "lossless"


@lru_cache(maxsize=1)
def ffmpeg_available_encoders() -> frozenset[str]:
    cmd = ["ffmpeg", "-hide_banner", "-encoders"]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg -encoders failed")
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.rstrip()
        if not line.startswith(" "):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out.add(parts[1].strip())
    return frozenset(out)


def ensure_ffmpeg_encoder_available(codec: str) -> None:
    codec_name = normalize_codec_strict(codec)
    if codec_name not in ffmpeg_available_encoders():
        raise RuntimeError(f"ffmpeg encoder not available: {codec_name}")


def _base_color_profile(codec: str, mode: str) -> tuple[tuple[str, ...], str]:
    codec_name = normalize_codec_strict(codec)
    mode_name = normalize_global_encoder_mode(mode)

    if mode_name == "lossless":
        if codec_name == "libx264":
            return (
                ("-c:v", codec_name, "-preset", "fast", "-qp", "0", "-pix_fmt", "yuv444p"),
                f"{codec_name} lossless yuv444p",
            )
        if codec_name == "libx265":
            return (
                (
                    "-c:v",
                    codec_name,
                    "-preset",
                    "fast",
                    "-pix_fmt",
                    "yuv444p",
                    "-x265-params",
                    "lossless=1",
                ),
                f"{codec_name} lossless yuv444p",
            )
        return (
            (
                "-c:v",
                codec_name,
                "-preset",
                "lossless",
                "-tune",
                "lossless",
                "-rc",
                "constqp",
                "-qp",
                "0",
                "-pix_fmt",
                "yuv444p",
            ),
            f"{codec_name} lossless yuv444p",
        )

    if "nvenc" in codec_name:
        qp_value = "0" if mode_name == "crf/qp 0" else "1"
        return (
            (
                "-c:v",
                codec_name,
                "-preset",
                "p7",
                "-rc",
                "constqp",
                "-qp",
                qp_value,
                "-pix_fmt",
                "yuv444p",
            ),
            f"{codec_name} {mode_name} yuv444p",
        )

    crf_value = "0" if mode_name == "crf/qp 0" else "1"
    return (
        ("-c:v", codec_name, "-preset", "fast", "-crf", crf_value, "-pix_fmt", "yuv444p"),
        f"{codec_name} {mode_name} yuv444p",
    )


def append_ffmpeg_extra_args(cmd: list[str], extra_args: str) -> list[str]:
    extra = str(extra_args or "").strip()
    if not extra:
        return list(cmd)
    return list(cmd) + shlex.split(extra)


def resolve_color_encoding_profile(codec: str, mode: str) -> FFmpegEncodingProfile:
    codec_name = normalize_codec_strict(codec)
    ensure_ffmpeg_encoder_available(codec_name)
    generated_args, summary = _base_color_profile(codec_name, mode)
    return FFmpegEncodingProfile(
        name=f"color:{codec_name}:{normalize_global_encoder_mode(mode)}",
        codec=codec_name,
        generated_args=generated_args,
        summary=summary,
        synthetic_source="color",
    )


def resolve_depth_preprocess_profile(codec: str) -> FFmpegEncodingProfile:
    return resolve_color_encoding_profile(codec, "lossless")


def resolve_fixed_grayscale_x264_profile(name: str) -> FFmpegEncodingProfile:
    ensure_ffmpeg_encoder_available("libx264")
    return FFmpegEncodingProfile(
        name=name,
        codec="libx264",
        generated_args=(
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv444p",
        ),
        summary="libx264 grayscale-safe qp0 yuv444p",
        synthetic_source="gray",
    )


def resolve_depth_final_grayscale_profile() -> FFmpegEncodingProfile:
    return resolve_fixed_grayscale_x264_profile("depth_final_gray")


def resolve_mask_for_merge_grayscale_profile() -> FFmpegEncodingProfile:
    return resolve_fixed_grayscale_x264_profile("mask_for_merge_gray")


def resolve_replace_mask_binary_profile() -> FFmpegEncodingProfile:
    return FFmpegEncodingProfile(
        name="replace_mask_binary",
        codec="ffv1",
        generated_args=(
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-slices",
            "16",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "gray",
        ),
        summary="ffv1 binary gray",
        synthetic_source="gray",
    )


def build_validation_command(
    profile: FFmpegEncodingProfile,
    output_path: str,
    *,
    extra_args: str = "",
    loglevel: str = "error",
) -> list[str]:
    color_testsrc = "testsrc2=size=64x64:rate=1"
    gray_testsrc = "nullsrc=s=64x64:r=1,format=gray"
    if profile.codec in {"h264_nvenc", "hevc_nvenc"}:
        color_testsrc = "testsrc2=size=256x144:rate=1"
        gray_testsrc = "nullsrc=s=256x144:r=1,format=gray"
    if profile.synthetic_source == "gray":
        input_args = [
            "-f",
            "lavfi",
            "-i",
            gray_testsrc,
            "-frames:v",
            "1",
        ]
    else:
        input_args = [
            "-f",
            "lavfi",
            "-i",
            color_testsrc,
            "-frames:v",
            "1",
        ]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", loglevel, "-y"] + input_args
    cmd.extend(profile.generated_args)
    cmd = append_ffmpeg_extra_args(cmd, extra_args)
    cmd.append(output_path)
    return cmd


def profile_preview_line(profile: FFmpegEncodingProfile, extra_args: str = "") -> str:
    parts = [profile.codec]
    parts.extend(profile.generated_args[2:] if profile.generated_args[:2] == ("-c:v", profile.codec) else profile.generated_args)
    line = " ".join(str(x) for x in parts)
    extra = str(extra_args or "").strip()
    if extra:
        line = f"{line} | + {extra}"
    return line
