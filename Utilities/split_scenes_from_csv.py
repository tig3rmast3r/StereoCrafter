#!/usr/bin/env python3
"""Split scenes from a SceneDetect CSV using parallel ffmpeg workers."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneRow:
    scene_num: int
    start_frame: int | None
    frame_count: int | None
    start_sec: float | None
    end_sec: float | None


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


def apply_frame_trim(ffmpeg_tokens: list[str], start_frame: int, frame_count: int) -> list[str]:
    start_frame_zero = max(0, int(start_frame) - 1)
    frame_count = max(1, int(frame_count))
    end_frame_zero = start_frame_zero + frame_count - 1
    frame_filter = (
        f"select='between(n,{start_frame_zero},{end_frame_zero})',"
        "setpts=N/FRAME_RATE/TB"
    )
    tokens = list(ffmpeg_tokens)
    for idx, tok in enumerate(tokens[:-1]):
        if tok == "-vf" or tok.startswith("-filter:v"):
            existing = str(tokens[idx + 1] or "").strip()
            tokens[idx + 1] = f"{frame_filter},{existing}" if existing else frame_filter
            break
    else:
        tokens.extend(["-vf", frame_filter])
    if ("-fps_mode" not in tokens) and ("-vsync" not in tokens):
        tokens.extend(["-fps_mode", "passthrough"])
    if not any(tok == "-frames:v" or tok.startswith("-frames:v") for tok in tokens):
        tokens.extend(["-frames:v", str(frame_count)])
    return tokens


def run_one_job(
    source_path: str,
    output_path: Path,
    scene_row: SceneRow,
    ffmpeg_tokens: list[str],
    skip_existing: bool,
    delete_failed: bool,
) -> tuple[str, str]:
    if skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
        return "skipped", str(output_path)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if (scene_row.start_frame is not None) and (scene_row.frame_count is not None):
        cmd.extend(["-i", source_path])
        cmd.extend(
            apply_frame_trim(
                ffmpeg_tokens,
                start_frame=scene_row.start_frame,
                frame_count=scene_row.frame_count,
            )
        )
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
