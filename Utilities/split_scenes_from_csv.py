#!/usr/bin/env python3
"""Split scenes from a SceneDetect CSV using parallel ffmpeg workers."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

_STOP_REQUESTED = False


@dataclass(frozen=True)
class SceneRow:
    scene_num: int
    start_frame: int | None
    frame_count: int | None
    start_sec: float | None
    end_sec: float | None


RETRYABLE_ENCODER_MARKERS = (
    "OpenEncodeSessionEx failed",
    "No capable devices found",
    "Error while opening encoder",
    "Could not open encoder before EOF",
    "Nothing was written into output file",
    "incompatible client key",
)

MAX_RETRY_ROUNDS = 4


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
    p.add_argument(
        "--stop-marker",
        default="",
        help="Optional graceful-stop marker file. Default: output-dir/.stop_after_current.",
    )
    return p.parse_args()


def _set_stop_requested() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _handle_signal(_signum, _frame) -> None:
    _set_stop_requested()


def _stop_requested(stop_marker: str) -> bool:
    if _STOP_REQUESTED:
        return True
    marker = str(stop_marker or "").strip()
    return bool(marker) and Path(marker).is_file()


def _clear_stop_marker(stop_marker: str) -> None:
    marker = str(stop_marker or "").strip()
    if not marker:
        return
    try:
        p = Path(marker)
        if p.is_file():
            p.unlink()
    except Exception:
        pass


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
    return f"{stem}-Scene-{int(scene_number):04d}.mp4"


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


def uses_nvenc(ffmpeg_tokens: list[str]) -> bool:
    joined = " ".join(str(tok or "").strip().lower() for tok in ffmpeg_tokens)
    return "nvenc" in joined


def is_retryable_encoder_failure(err_text: str) -> bool:
    err_lower = str(err_text or "").strip().lower()
    if not err_lower:
        return False
    return any(marker.lower() in err_lower for marker in RETRYABLE_ENCODER_MARKERS)


def retry_workers_for_round(initial_workers: int, round_idx: int, nvenc_active: bool) -> int:
    workers = max(1, int(initial_workers))
    if not nvenc_active:
        return workers
    if round_idx <= 1:
        return workers
    if round_idx == 2:
        return max(1, min(workers, 4))
    if round_idx == 3:
        return max(1, min(workers, 2))
    return 1


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
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
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
    stop_marker = (
        str(Path(args.stop_marker).expanduser().resolve())
        if str(args.stop_marker).strip()
        else str((output_dir / ".stop_after_current").resolve())
    )
    if Path(stop_marker).is_file():
        print(f"[INFO] removing stale stop marker: {stop_marker}", flush=True)
        _clear_stop_marker(stop_marker)
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
    nvenc_active = uses_nvenc(ffmpeg_tokens)
    print(
        f"[SPLIT] jobs={len(jobs)} workers={workers} "
        f"skip_existing={args.skip_existing} delete_failed={args.delete_failed}",
        flush=True,
    )
    if nvenc_active:
        print("[SPLIT] encoder path detected: nvenc", flush=True)

    done = 0
    skipped = 0
    failed = 0
    total = len(jobs)
    failures: list[str] = []
    stop_logged = False

    def _note_stop_requested() -> bool:
        nonlocal stop_logged
        if not _stop_requested(stop_marker):
            return False
        if not stop_logged:
            print("[STOP] graceful stop requested. Waiting current split job(s) to finish.", flush=True)
            stop_logged = True
        return True

    pending_jobs = list(jobs)
    round_idx = 0
    resolved = 0
    while pending_jobs:
        if _note_stop_requested():
            break
        round_idx += 1
        round_workers = retry_workers_for_round(workers, round_idx, nvenc_active)
        round_total = len(pending_jobs)
        round_done = 0
        round_skipped = 0
        round_failed_retryable: list[tuple[Path, SceneRow, str]] = []
        round_failed_hard: list[str] = []
        print(
            f"[SPLIT] round {round_idx}/{MAX_RETRY_ROUNDS} jobs={round_total} workers={round_workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=round_workers) as ex:
            round_pending = list(pending_jobs)
            future_to_job = {}
            completed_in_round = 0
            retry_waiting = 0
            while round_pending or future_to_job:
                while (
                    round_pending
                    and len(future_to_job) < round_workers
                    and not _stop_requested(stop_marker)
                ):
                    out_path, scene_row = round_pending.pop(0)
                    future_to_job[
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
                    ] = (out_path, scene_row)
                _note_stop_requested()
                if not future_to_job:
                    break
                done_set, _pending = wait(
                    tuple(future_to_job.keys()),
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                if not done_set:
                    continue
                for fut in done_set:
                    out_path, scene_row = future_to_job.pop(fut)
                    status, payload = fut.result()
                    completed_in_round += 1
                    if status == "done":
                        done += 1
                        round_done += 1
                        resolved += 1
                    elif status == "skipped":
                        skipped += 1
                        round_skipped += 1
                        resolved += 1
                    else:
                        if (
                            round_idx < MAX_RETRY_ROUNDS
                            and is_retryable_encoder_failure(payload)
                            and not _stop_requested(stop_marker)
                        ):
                            round_failed_retryable.append((out_path, scene_row, payload))
                            retry_waiting += 1
                        else:
                            failed += 1
                            round_failed_hard.append(payload)
                            failures.append(payload)
                            resolved += 1
                    remaining_current = len(round_pending) + len(future_to_job)
                    pending_total = remaining_current + retry_waiting
                    progress_done = total - pending_total
                    pct = (float(progress_done) / float(total)) * 100.0 if total > 0 else 100.0
                    print(f"[SPLIT] progress {progress_done}/{total} ({pct:.1f}%)", flush=True)

        print(
            (
                f"[SPLIT] round {round_idx} summary: "
                f"done={round_done} skipped={round_skipped} "
                f"retry={len(round_failed_retryable)} failed={len(round_failed_hard)}"
            ),
            flush=True,
        )
        if round_failed_retryable and not _stop_requested(stop_marker):
            pending_jobs = [(out_path, scene_row) for out_path, scene_row, _err in round_failed_retryable]
            sleep_sec = min(6, 2 * round_idx)
            print(
                (
                    f"[SPLIT][RETRY] round {round_idx}: retrying {len(pending_jobs)} "
                    f"encoder-open failure(s) after {sleep_sec}s"
                ),
                flush=True,
            )
            for _out_path, _scene_row, err in round_failed_retryable[:20]:
                print(f"[SPLIT][RETRYABLE] {err}", flush=True)
            time.sleep(float(sleep_sec))
        else:
            pending_jobs = []

    print(
        f"[SPLIT] summary: done={done} skipped={skipped} failed={failed} total={total}",
        flush=True,
    )
    if _stop_requested(stop_marker):
        _clear_stop_marker(stop_marker)
        remaining = max(0, total - (done + skipped + failed))
        print(
            f"[STOP] graceful stop completed: remaining_jobs={remaining} marker={stop_marker}",
            flush=True,
        )
    if failures:
        for item in failures[:50]:
            print(f"[SPLIT][FAILED] {item}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
