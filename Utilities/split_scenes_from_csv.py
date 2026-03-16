#!/usr/bin/env python3
"""Split scenes from a SceneDetect CSV using parallel ffmpeg workers."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class SceneRow:
    scene_num: int
    start_frame: int | None
    frame_count: int | None
    start_sec: float | None
    end_sec: float | None


def parse_ratio(value: str) -> Fraction | None:
    txt = str(value or "").strip()
    if not txt or txt.upper() == "N/A":
        return None
    if "/" in txt:
        num_txt, den_txt = txt.split("/", 1)
        try:
            num = int(num_txt.strip())
            den = int(den_txt.strip())
        except Exception:
            return None
        if num <= 0 or den <= 0:
            return None
        return Fraction(num, den)
    try:
        num = float(txt)
    except Exception:
        return None
    if num <= 0:
        return None
    return Fraction(str(num))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel scene splitting from SceneDetect CSV.",
    )
    p.add_argument("--input-video", required=True, help="Input source video path.")
    p.add_argument("--scene-csv", required=True, help="SceneDetect CSV path.")
    p.add_argument("--output-dir", required=True, help="Output clip folder.")
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Parallel ffmpeg workers (default: 8).",
    )
    p.add_argument(
        "--ffmpeg-args",
        required=True,
        help="ffmpeg output args string (quoted).",
    )
    p.add_argument(
        "--skip-existing",
        choices=["yes", "no"],
        default="yes",
        help="Skip existing non-empty outputs.",
    )
    p.add_argument(
        "--delete-failed",
        choices=["yes", "no"],
        default="yes",
        help="Delete output file if ffmpeg fails.",
    )
    return p.parse_args()


def parse_seconds_or_timecode(value: str) -> float | None:
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        pass
    parts = txt.split(":")
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
    except Exception:
        return None
    return float(hh * 3600 + mm * 60) + float(ss)


def parse_intish(value: str) -> int | None:
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        return int(txt)
    except Exception:
        pass
    try:
        return int(float(txt))
    except Exception:
        return None


def output_name_from_scene(source_path: str, scene_number: int) -> str:
    stem = Path(source_path).stem or "source"
    return f"{stem}-Scene-{int(scene_number):03d}.mp4"


def probe_source_fps(source_path: str) -> Fraction | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(source_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream0 = streams[0] or {}
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = parse_ratio(stream0.get(key) or "")
        if fps is not None and fps > 0:
            return fps
    return None


def load_scene_rows(csv_path: Path) -> list[SceneRow]:
    out: list[SceneRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if not isinstance(row, dict):
                continue
            norm = {str(k or "").strip().lower(): str(v or "").strip() for k, v in row.items()}
            scene_raw = norm.get("scene number") or norm.get("scene") or norm.get("scene #") or ""
            try:
                scene_num = int(float(scene_raw)) if scene_raw else idx
            except Exception:
                scene_num = idx
            start_frame = parse_intish(norm.get("start frame") or "")
            frame_count = parse_intish(
                norm.get("length (frames)")
                or norm.get("length frames")
                or ""
            )
            end_frame = parse_intish(norm.get("end frame") or "")
            if (start_frame is not None) and (start_frame <= 0):
                start_frame = None
            if (frame_count is not None) and (frame_count <= 0):
                frame_count = None
            if (start_frame is not None) and (frame_count is None) and (end_frame is not None):
                if end_frame >= start_frame:
                    frame_count = (end_frame - start_frame) + 1
            start_raw = (
                norm.get("start time (seconds)")
                or norm.get("start seconds")
                or norm.get("start timecode")
                or ""
            )
            end_raw = (
                norm.get("end time (seconds)")
                or norm.get("end seconds")
                or norm.get("end timecode")
                or ""
            )
            start_sec = parse_seconds_or_timecode(start_raw)
            end_sec = parse_seconds_or_timecode(end_raw)
            if (start_frame is not None) and (frame_count is not None):
                out.append(
                    SceneRow(
                        scene_num=int(scene_num),
                        start_frame=int(start_frame),
                        frame_count=int(frame_count),
                        start_sec=float(start_sec) if start_sec is not None else None,
                        end_sec=float(end_sec) if end_sec is not None else None,
                    )
                )
                continue
            if start_sec is None or end_sec is None:
                continue
            if end_sec <= start_sec:
                continue
            out.append(
                SceneRow(
                    scene_num=int(scene_num),
                    start_frame=None,
                    frame_count=None,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                )
            )
    return out


def apply_frame_output_guard(ffmpeg_tokens: list[str], frame_count: int) -> list[str]:
    frame_count = max(1, int(frame_count))
    tokens = list(ffmpeg_tokens)
    if not any(tok == "-frames:v" or tok.startswith("-frames:v") for tok in tokens):
        tokens.extend(["-frames:v", str(frame_count)])
    return tokens


def frame_to_seconds(frame_number: int, fps: Fraction) -> float:
    fps_fraction = Fraction(fps)
    return float(Fraction(int(frame_number), 1) / fps_fraction)


def format_seconds(value: float) -> str:
    txt = f"{float(value):.15f}".rstrip("0").rstrip(".")
    return txt if txt else "0"


def run_one_job(
    source_path: str,
    output_path: Path,
    scene_row: SceneRow,
    ffmpeg_tokens: list[str],
    source_fps: Fraction | None,
    skip_existing: bool,
    delete_failed: bool,
) -> tuple[str, str]:
    if skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
        return "skipped", str(output_path)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if (
        (scene_row.start_frame is not None)
        and (scene_row.frame_count is not None)
        and (source_fps is not None)
    ):
        # SceneDetect CSV stores 1-based frame numbers; derive exact seek/duration from frames
        # like PySceneDetect does internally, but keep a frame cap as a guard-rail.
        start_frame_zero = max(0, int(scene_row.start_frame) - 1)
        frame_count = max(1, int(scene_row.frame_count))
        start_sec = frame_to_seconds(start_frame_zero, source_fps)
        duration_sec = frame_to_seconds(frame_count, source_fps)
        cmd.extend(
            [
                "-ss",
                format_seconds(start_sec),
                "-i",
                source_path,
                "-t",
                format_seconds(duration_sec),
            ]
        )
        cmd.extend(apply_frame_output_guard(ffmpeg_tokens, frame_count=frame_count))
    else:
        start_sec = float(scene_row.start_sec or 0.0)
        end_sec = float(scene_row.end_sec or 0.0)
        duration = max(0.001, end_sec - start_sec)
        cmd.extend(
            [
                "-ss",
                f"{start_sec:.6f}",
                "-i",
                source_path,
                "-t",
                f"{duration:.6f}",
            ]
        )
        cmd.extend(ffmpeg_tokens)
    cmd.append(str(output_path))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
        return "done", str(output_path)

    if delete_failed:
        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass
    err = (proc.stderr or proc.stdout or "").strip()
    return "failed", f"{output_path} :: {err or 'ffmpeg failed'}"


def main() -> int:
    args = parse_args()
    source_path = str(Path(args.input_video).expanduser().resolve())
    scene_csv = Path(args.scene_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not Path(source_path).is_file():
        print(f"[ERROR] source video not found: {source_path}", flush=True)
        return 2
    if not scene_csv.is_file():
        print(f"[ERROR] scene csv not found: {scene_csv}", flush=True)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_tokens = shlex.split(args.ffmpeg_args)
    if not ffmpeg_tokens:
        print("[ERROR] empty ffmpeg args.", flush=True)
        return 2

    scene_rows = load_scene_rows(scene_csv)
    if not scene_rows:
        print(f"[ERROR] no valid scene rows in CSV: {scene_csv}", flush=True)
        return 2

    source_fps = probe_source_fps(source_path)
    if source_fps is not None:
        print(
            (
                f"[SPLIT] source_fps={source_fps.numerator}/{source_fps.denominator} "
                f"({float(source_fps):.6f})"
            ),
            flush=True,
        )
    else:
        print(
            "[SPLIT][WARN] failed to probe source fps, falling back to CSV time-based split.",
            flush=True,
        )

    jobs: list[tuple[Path, SceneRow]] = []
    for scene_row in scene_rows:
        out_name = output_name_from_scene(source_path, scene_row.scene_num)
        jobs.append((output_dir / out_name, scene_row))

    workers = max(1, int(args.threads))
    print(
        f"[SPLIT] jobs={len(jobs)} workers={workers} "
        f"skip_existing={args.skip_existing} delete_failed={args.delete_failed}",
        flush=True,
    )

    done = 0
    skipped = 0
    failed = 0
    total = len(jobs)
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                run_one_job,
                source_path,
                out_path,
                scene_row,
                ffmpeg_tokens,
                source_fps,
                args.skip_existing == "yes",
                args.delete_failed == "yes",
            )
            for out_path, scene_row in jobs
        ]
        completed = 0
        for fut in as_completed(futures):
            status, payload = fut.result()
            completed += 1
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(payload)
            pct = (float(completed) / float(total)) * 100.0 if total > 0 else 100.0
            print(f"[SPLIT] progress {completed}/{total} ({pct:.1f}%)", flush=True)

    print(
        f"[SPLIT] summary: done={done} skipped={skipped} failed={failed} total={total}",
        flush=True,
    )
    if failures:
        for item in failures[:50]:
            print(f"[SPLIT][FAILED] {item}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
