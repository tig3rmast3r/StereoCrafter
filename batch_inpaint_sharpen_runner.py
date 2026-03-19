#!/usr/bin/env python3
"""Batch runner for the fixed post-inpaint sharpen preset."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
UTILS_DIR = REPO_ROOT / "Utilities"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from apply_inpaint_sharpen_preset import (
    DEFAULT_MIN_SHARPNESS_LEVEL,
    _build_processing_args,
    is_eligible,
    load_sharpness_index,
    process_video,
)


DEFAULT_GLOB = "*.mp4"
DEFAULT_WORKERS = 8
DEFAULT_MAX_ATTEMPTS = 3

STOP_REQUESTED = False
STOP_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the fixed post-inpaint sharpen preset on eligible scenes."
    )
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sharpness_csv_path", required=True)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--stop_marker", default="")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--pix_fmt", default="yuv444p")
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--crf", default="0")
    ap.add_argument("--output_extra_args", default="")
    return ap.parse_args()


def _set_stop_requested() -> None:
    global STOP_REQUESTED
    with STOP_LOCK:
        STOP_REQUESTED = True


def _stop_requested(stop_marker: str) -> bool:
    global STOP_REQUESTED
    if STOP_REQUESTED:
        return True
    if stop_marker and Path(stop_marker).is_file():
        with STOP_LOCK:
            STOP_REQUESTED = True
        return True
    return False


def _handle_signal(_signum, _frame) -> None:
    _set_stop_requested()


def _iter_inputs(input_dir: Path, glob_pat: str, only: str) -> List[Path]:
    paths = sorted(Path(p).resolve() for p in glob.glob(str(input_dir / str(glob_pat))))
    if only:
        paths = [p for p in paths if p.name == only or p.name.startswith(only)]
    return [p for p in paths if p.is_file()]


def _eligible_inputs(
    paths: List[Path],
    sharpness_index: Dict[str, Tuple[float, int]],
) -> List[Path]:
    out: List[Path] = []
    for path in paths:
        eligible, _raw, _level = is_eligible(
            sharpness_index,
            path.name,
            min_level=DEFAULT_MIN_SHARPNESS_LEVEL,
        )
        if eligible:
            out.append(path)
    return out


def _run_one(
    input_path: Path,
    args: argparse.Namespace,
    processing_args,
    sharpness_index: Dict[str, Tuple[float, int]],
) -> dict:
    attempts = 0
    last_error = ""
    while attempts < DEFAULT_MAX_ATTEMPTS:
        attempts += 1
        try:
            row = process_video(
                input_path=input_path,
                mask_dir=Path(args.mask_dir).expanduser().resolve(),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                sharpness_index=sharpness_index,
                skip_existing=bool(args.skip_existing),
                processing_args=processing_args,
            )
            return {
                "file": input_path.name,
                "status": row.status,
                "reason": row.reason,
                "attempts": attempts,
                "frames_written": row.frames_written,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts < DEFAULT_MAX_ATTEMPTS:
                print(
                    f"[RETRY] {input_path.name} attempt {attempts + 1}/{DEFAULT_MAX_ATTEMPTS} "
                    f"after {last_error}",
                    flush=True,
                )
                time.sleep(0.5)
    return {
        "file": input_path.name,
        "status": "error",
        "reason": last_error or "unknown_error",
        "attempts": attempts,
        "frames_written": 0,
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {input_dir}")
    if not mask_dir.is_dir():
        raise SystemExit(f"mask_dir not found: {mask_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    sharpness_index = load_sharpness_index(str(args.sharpness_csv_path))
    all_inputs = _iter_inputs(input_dir, str(args.glob), str(args.only or "").strip())
    if not all_inputs:
        print("[ERR] no input files found", flush=True)
        return 2

    eligible = _eligible_inputs(all_inputs, sharpness_index)
    total = len(eligible)
    if total <= 0:
        print("[DONE] 0/0 eligible scenes (sharpness <= 7 for all inputs).", flush=True)
        return 0

    workers = max(1, int(args.workers))
    processing_args = _build_processing_args(args)
    completed = 0
    had_error = False
    pending = list(eligible)
    futures: dict[concurrent.futures.Future, Path] = {}

    print(
        f"[RUN] eligible={total} workers={workers} output={output_dir} sharpness_csv={args.sharpness_csv_path}",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while pending or futures:
            while pending and len(futures) < workers and not _stop_requested(str(args.stop_marker or "")):
                cur = pending.pop(0)
                fut = executor.submit(_run_one, cur, args, processing_args, sharpness_index)
                futures[fut] = cur

            if not futures:
                break

            done, _not_done = concurrent.futures.wait(
                list(futures.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                input_path = futures.pop(fut)
                completed += 1
                prefix = f"[{completed}/{total}] {input_path.name}"
                try:
                    result = fut.result()
                except Exception as exc:
                    had_error = True
                    print(f"{prefix}", flush=True)
                    print(f"[ERR] {completed}/{total} {input_path.name} {type(exc).__name__}: {exc}", flush=True)
                    continue

                print(prefix, flush=True)
                status = str(result.get("status", "error")).strip().lower()
                reason = str(result.get("reason", "")).strip()
                if status == "applied":
                    print(
                        f"[OK] {completed}/{total} {input_path.name} "
                        f"frames={int(result.get('frames_written', 0))}",
                        flush=True,
                    )
                elif status == "skipped":
                    print(
                        f"[SKIP] {completed}/{total} {input_path.name} reason={reason}",
                        flush=True,
                    )
                else:
                    had_error = True
                    print(
                        f"[ERR] {completed}/{total} {input_path.name} reason={reason}",
                        flush=True,
                    )

            if _stop_requested(str(args.stop_marker or "")):
                print("[STOP] graceful stop requested; waiting for running sharpen jobs to finish current file.", flush=True)
                pending.clear()

    if _stop_requested(str(args.stop_marker or "")):
        print(f"[DONE] stopped after {completed}/{total} eligible scene(s).", flush=True)
        return 130

    summary = {
        "eligible": total,
        "processed": completed,
        "errors": bool(had_error),
    }
    print(f"[DONE] {json.dumps(summary, ensure_ascii=True)}", flush=True)
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
