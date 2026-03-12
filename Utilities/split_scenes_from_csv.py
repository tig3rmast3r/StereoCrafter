#!/usr/bin/env python3
"""Split scenes from a SceneDetect CSV using parallel ffmpeg workers."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


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


def output_name_from_scene(source_path: str, scene_number: int) -> str:
    stem = Path(source_path).stem or "source"
    return f"{stem}-Scene-{int(scene_number):03d}.mp4"


def load_scene_rows(csv_path: Path) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
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
            if start_sec is None or end_sec is None:
                continue
            if end_sec <= start_sec:
                continue
            out.append((int(scene_num), float(start_sec), float(end_sec)))
    return out


def run_one_job(
    source_path: str,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    ffmpeg_tokens: list[str],
    skip_existing: bool,
    delete_failed: bool,
) -> tuple[str, str]:
    if skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
        return "skipped", str(output_path)

    duration = max(0.001, float(end_sec) - float(start_sec))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.6f}",
    ]
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

    jobs: list[tuple[Path, float, float]] = []
    for scene_num, start_sec, end_sec in scene_rows:
        out_name = output_name_from_scene(source_path, scene_num)
        jobs.append((output_dir / out_name, start_sec, end_sec))

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
                start_sec,
                end_sec,
                ffmpeg_tokens,
                args.skip_existing == "yes",
                args.delete_failed == "yes",
            )
            for out_path, start_sec, end_sec in jobs
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
