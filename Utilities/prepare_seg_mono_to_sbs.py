#!/usr/bin/env python3
"""Create SBS clips from seg-mono clips before final join.

For each input clip in seg-mono, this utility generates one SBS clip in the
target sbs folder, unless a matching output already exists.
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare seg-mono clips into SBS clips.")
    p.add_argument("--seg-mono-dir", required=True, help="Input folder with mono scene clips.")
    p.add_argument("--sbs-dir", required=True, help="Output SBS folder (work/sbs).")
    p.add_argument(
        "--layout",
        choices=("full_sbs", "half_sbs"),
        default="full_sbs",
        help="Target layout for generated mono scenes.",
    )
    p.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg binary.")
    p.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe binary.")
    p.add_argument("--codec", default="libx264", help="Encoder codec for generated clips.")
    p.add_argument("--quality-flag", default="crf", help="Quality flag without dash (crf/cq/qp).")
    p.add_argument("--quality", default="1", help="Quality value.")
    p.add_argument("--preset", default="fast", help="Encoder preset.")
    p.add_argument("--pix-fmt", default="yuv420p", help="Pixel format.")
    p.add_argument("--extra-ffmpeg-args", default="", help="Extra ffmpeg args.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite already existing outputs.")
    return p.parse_args()


def _probe_video_info(ffprobe_bin: str, video_path: Path) -> dict[str, int | str | None]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_packets",
        "-show_entries",
        (
            "stream=codec_type,width,height,nb_read_packets,nb_read_frames,"
            "r_frame_rate,time_base,color_range,color_space,color_transfer,color_primaries"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or f"ffprobe rc={p.returncode}").strip())
    try:
        import json

        doc = json.loads(p.stdout or "{}")
    except Exception as exc:
        raise RuntimeError(f"invalid ffprobe json: {exc}") from exc
    streams = doc.get("streams") or []
    if not streams:
        raise RuntimeError("missing video stream in ffprobe output")
    video_stream = None
    extra_streams = 0
    for stream in streams:
        st = stream or {}
        if str(st.get("codec_type") or "").strip().lower() == "video" and video_stream is None:
            video_stream = st
        elif str(st.get("codec_type") or "").strip().lower() != "video":
            extra_streams += 1
    if not video_stream:
        raise RuntimeError("missing video stream in ffprobe output")
    st = video_stream
    width = st.get("width")
    height = st.get("height")
    if width in (None, "", "N/A"):
        raise RuntimeError("missing width in ffprobe output")
    if height in (None, "", "N/A"):
        raise RuntimeError("missing height in ffprobe output")
    packets = st.get("nb_read_packets")
    if packets in (None, "", "N/A"):
        packets = st.get("nb_read_frames")
    packet_count = None
    if packets not in (None, "", "N/A"):
        try:
            packet_count = int(float(packets))
        except Exception:
            packet_count = None
    return {
        "width": int(float(width)),
        "height": int(float(height)),
        "packets": packet_count,
        "r_frame_rate": str(st.get("r_frame_rate") or ""),
        "time_base": str(st.get("time_base") or ""),
        "color_range": str(st.get("color_range") or ""),
        "color_space": str(st.get("color_space") or ""),
        "color_transfer": str(st.get("color_transfer") or ""),
        "color_primaries": str(st.get("color_primaries") or ""),
        "extra_streams": int(extra_streams),
    }


def _collect_input_clips(seg_mono_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in VIDEO_EXTS:
        files.extend(p for p in seg_mono_dir.glob(f"*{ext}") if p.is_file())
        files.extend(p for p in seg_mono_dir.glob(f"*{ext.upper()}") if p.is_file())
    uniq = {str(p.resolve()): p.resolve() for p in files}
    return sorted(uniq.values())


def _find_matching_outputs(sbs_dir: Path, stem: str) -> list[Path]:
    patterns = [
        f"{stem}_*_merged_*.mp4",
        f"{stem}_*_merged_*.mkv",
        f"{stem}_*_merged_*.mov",
        f"{stem}_*_merged_*.avi",
        f"{stem}_*_merged_*.webm",
    ]
    hits: dict[str, Path] = {}
    for pat in patterns:
        for hit in glob.glob(str(sbs_dir / pat)):
            p = Path(hit).resolve()
            if p.is_file():
                hits[str(p)] = p
    return sorted(hits.values())


def _build_filter_graph(layout: str) -> str:
    if layout == "half_sbs":
        return (
            "[0:v]split=2[l][r];"
            "[l]scale=iw/2:ih[left];"
            "[r]scale=iw/2:ih[right];"
            "[left][right]hstack=inputs=2[v]"
        )
    return "[0:v]split=2[l][r];[l][r]hstack=inputs=2[v]"


def _build_output_name(stem: str, width: int, layout: str) -> str:
    suffix = "_merged_half_sbs.mp4" if layout == "half_sbs" else "_merged_full_sbs.mp4"
    return f"{stem}_{width}{suffix}"


def _is_existing_output_healthy(
    *,
    ffprobe_bin: str,
    src_info: dict[str, int | str | None],
    dst: Path,
    layout: str,
) -> tuple[bool, str]:
    try:
        out_info = _probe_video_info(ffprobe_bin, dst)
    except Exception as exc:
        return False, str(exc)
    expected_width = int(src_info["width"]) if layout == "half_sbs" else int(src_info["width"]) * 2
    expected_height = int(src_info["height"])
    out_width = int(out_info["width"])
    out_height = int(out_info["height"])
    if out_width != expected_width:
        return False, f"width={out_width} expected={expected_width}"
    if out_height != expected_height:
        return False, f"height={out_height} expected={expected_height}"
    src_packets = src_info.get("packets")
    out_packets = out_info.get("packets")
    if src_packets is None or out_packets is None:
        return False, "packet count unavailable"
    if abs(int(src_packets) - int(out_packets)) > 1:
        return False, f"packets={out_packets} expected={src_packets}"
    src_r_frame_rate = str(src_info.get("r_frame_rate") or "").strip()
    out_r_frame_rate = str(out_info.get("r_frame_rate") or "").strip()
    if src_r_frame_rate and out_r_frame_rate and out_r_frame_rate != src_r_frame_rate:
        return False, f"r_frame_rate={out_r_frame_rate} expected={src_r_frame_rate}"
    src_time_base = str(src_info.get("time_base") or "").strip()
    out_time_base = str(out_info.get("time_base") or "").strip()
    if src_time_base and out_time_base and out_time_base != src_time_base:
        return False, f"time_base={out_time_base} expected={src_time_base}"
    if int(out_info.get("extra_streams") or 0) != 0:
        return False, f"extra_streams={int(out_info.get('extra_streams') or 0)}"
    for field in ("color_space", "color_transfer", "color_primaries"):
        value = str(out_info.get(field) or "").strip().lower()
        if value != "bt709":
            return False, f"{field}={value or '(missing)'} expected=bt709"
    return True, "ok"


def _remove_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _run_ffmpeg(
    *,
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    filter_graph: str,
    codec: str,
    quality_flag: str,
    quality: str,
    preset: str,
    pix_fmt: str,
    extra_ffmpeg_args: str,
    overwrite: bool,
) -> tuple[int, str]:
    cmd: list[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-write_tmcd",
        "0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        codec,
    ]
    if preset:
        cmd.extend(["-preset", preset])
    if quality_flag and quality:
        cmd.extend([f"-{quality_flag}", quality])
    if pix_fmt:
        cmd.extend(["-pix_fmt", pix_fmt])
    cmd.extend(
        [
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-movflags",
            "+write_colr",
        ]
    )
    if extra_ffmpeg_args.strip():
        cmd.extend(shlex.split(extra_ffmpeg_args.strip()))
    cmd.append(str(dst))

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout or ""


def main() -> int:
    args = _parse_args()
    seg_mono_dir = Path(args.seg_mono_dir).expanduser().resolve()
    sbs_dir = Path(args.sbs_dir).expanduser().resolve()

    if not shutil_which(args.ffmpeg_bin):
        print(f"[ERR] ffmpeg not found in PATH: {args.ffmpeg_bin}")
        return 2
    if not shutil_which(args.ffprobe_bin):
        print(f"[ERR] ffprobe not found in PATH: {args.ffprobe_bin}")
        return 2

    if not seg_mono_dir.is_dir():
        print(f"[INFO] seg-mono folder not found, nothing to prepare: {seg_mono_dir}")
        return 0

    sbs_dir.mkdir(parents=True, exist_ok=True)
    mono_files = _collect_input_clips(seg_mono_dir)
    if not mono_files:
        print(f"[INFO] No mono scene clips found in: {seg_mono_dir}")
        return 0

    filter_graph = _build_filter_graph(args.layout)
    created = 0
    skipped = 0
    failed = 0
    total = len(mono_files)

    for idx, src in enumerate(mono_files, start=1):
        stem = src.stem
        try:
            src_info = _probe_video_info(args.ffprobe_bin, src)
        except Exception as exc:
            print(f"[MONO][ERR  {idx}/{total}] probe failed for {src.name}: {exc}")
            failed += 1
            continue

        width = int(src_info["width"])
        out_name = _build_output_name(stem, width, args.layout)
        dst = sbs_dir / out_name
        stale_outputs = [p for p in _find_matching_outputs(sbs_dir, stem) if p != dst]
        for stale in stale_outputs:
            print(f"[MONO][CLEAN {idx}/{total}] removing stale output: {stale.name}")
            _remove_file(stale)

        if dst.exists() and dst.stat().st_size > 0 and not args.overwrite:
            healthy, reason = _is_existing_output_healthy(
                ffprobe_bin=args.ffprobe_bin,
                src_info=src_info,
                dst=dst,
                layout=args.layout,
            )
            if healthy:
                print(f"[MONO][SKIP {idx}/{total}] healthy output exists: {dst.name}")
                skipped += 1
                continue
            print(f"[MONO][REGEN {idx}/{total}] existing output invalid, recreating {dst.name}: {reason}")
            _remove_file(dst)

        print(f"[MONO][RUN  {idx}/{total}] {src.name} -> {dst.name}")
        rc, output = _run_ffmpeg(
            ffmpeg_bin=args.ffmpeg_bin,
            src=src,
            dst=dst,
            filter_graph=filter_graph,
            codec=args.codec,
            quality_flag=(args.quality_flag or "").strip().lower(),
            quality=args.quality.strip(),
            preset=args.preset.strip(),
            pix_fmt=args.pix_fmt.strip(),
            extra_ffmpeg_args=args.extra_ffmpeg_args,
            overwrite=args.overwrite,
        )
        if rc == 0 and dst.exists() and dst.stat().st_size > 0:
            healthy, reason = _is_existing_output_healthy(
                ffprobe_bin=args.ffprobe_bin,
                src_info=src_info,
                dst=dst,
                layout=args.layout,
            )
            if not healthy:
                _remove_file(dst)
                failed += 1
                print(f"[MONO][ERR  {idx}/{total}] output failed post-check for {src.name}: {reason}")
                if output.strip():
                    print(output.rstrip())
                continue
            created += 1
            print(f"[MONO][OK   {idx}/{total}] {dst.name}")
        else:
            _remove_file(dst)
            failed += 1
            print(f"[MONO][ERR  {idx}/{total}] ffmpeg failed for {src.name} (rc={rc})")
            if output.strip():
                print(output.rstrip())

    print(
        f"[MONO][DONE] total={total} created={created} skipped={skipped} failed={failed} "
        f"layout={args.layout} out={sbs_dir}"
    )
    return 1 if failed else 0


def shutil_which(binary: str) -> str | None:
    # Local import keeps script startup minimal and dependency-free.
    import shutil

    return shutil.which(binary)


if __name__ == "__main__":
    raise SystemExit(main())
