#!/usr/bin/env python3
"""
Pre-analyze Auto CT winners and write per-frame CSV guidance.

Workflow:
- analyze every frame (frame-by-frame, no stride)
- for each frame run the same Auto CT selector used in merging
- write one CSV row per frame:
    video, frame, best_preset
- optional extra debug columns help diagnostics/resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu  # type: ignore

from merging_nogui_batch import (
    _select_best_auto_ct_preset_frame,
    collect_jobs,
    find_replace_mask_for_splatted,
    find_video_by_core_name,
    parse_core_and_width,
    parse_inpainted_name,
)

DEFAULT_CT_SETTINGS: Dict[str, object] = {
    "ct_strength": 1.0,
    "ct_black_thresh": 0.0,
    "ct_min_valid_ratio": 0.0,
    "ct_min_valid": 0,
    "ct_clamp_L_min": 0.1,
    "ct_clamp_L_max": 2.0,
    "ct_clamp_ab_min": 0.1,
    "ct_clamp_ab_max": 3.0,
    "ct_exclude_black_in_target": True,
    "ct_ring_width": 20,
}


def _safe_uint(v: int) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _dbg(args: argparse.Namespace, msg: str) -> None:
    if bool(getattr(args, "debug", False)):
        print(f"[DBG] {msg}", flush=True)


class ProgressTracker:
    def __init__(self, every_sec: float, every_frames: int) -> None:
        self.every_sec = max(0.0, float(every_sec))
        self.every_frames = max(0, int(every_frames))
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._t_last = self._t0
        self._frames_total = 0
        self._frames_last = 0
        self._frames_complete_total = 0
        self._frames_complete_last = 0

    def bump(self, n_total: int = 1, n_complete: int = 0) -> None:
        if self.every_sec <= 0.0 and self.every_frames <= 0:
            return
        now = time.monotonic()
        to_print: Optional[str] = None
        with self._lock:
            self._frames_total += int(n_total)
            self._frames_complete_total += int(n_complete)
            due_time = self.every_sec > 0.0 and (now - self._t_last) >= self.every_sec
            due_frames = (
                self.every_frames > 0
                and (self._frames_complete_total - self._frames_complete_last)
                >= self.every_frames
            )
            if due_time or due_frames:
                elapsed_total = max(1e-6, now - self._t0)
                elapsed_window = max(1e-6, now - self._t_last)
                delta_complete = self._frames_complete_total - self._frames_complete_last
                fpm_avg = (self._frames_complete_total / elapsed_total) * 60.0
                fpm_window = (delta_complete / elapsed_window) * 60.0
                to_print = (
                    f"[PROG] frames_complete={self._frames_complete_total} | "
                    f"frames_seen={self._frames_total} | "
                    f"fpm_complete_window={fpm_window:.1f} | "
                    f"fpm_complete_avg={fpm_avg:.1f}"
                )
                self._t_last = now
                self._frames_last = self._frames_total
                self._frames_complete_last = self._frames_complete_total
        if to_print is not None:
            print(to_print, flush=True)

    def final_report(self) -> None:
        now = time.monotonic()
        with self._lock:
            elapsed_total = max(1e-6, now - self._t0)
            fpm_avg = (self._frames_complete_total / elapsed_total) * 60.0
            print(
                f"[PROG-FINAL] frames_complete={self._frames_complete_total} | "
                f"frames_seen={self._frames_total} | "
                f"fpm_complete_avg={fpm_avg:.1f}",
                flush=True,
            )


def _open_video_reader(path: str, args: argparse.Namespace) -> VideoReader:
    return VideoReader(path, ctx=cpu(0))


def _load_existing_frame_keys(csv_path: str) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "video" not in fieldnames or "frame" not in fieldnames:
                print(
                    f"[WARN] existing CSV has no video/frame columns, "
                    f"row-level resume disabled: {csv_path}"
                )
                return done
            for row in reader:
                name = str((row or {}).get("video", "")).strip()
                frame_raw = str((row or {}).get("frame", "")).strip()
                if not name:
                    continue
                try:
                    frame_idx = int(frame_raw)
                except Exception:
                    continue
                done.add((name, frame_idx))
    except Exception as e:
        print(f"[WARN] failed to read existing CSV for resume ({csv_path}): {e}")
    return done


def _build_existing_frames_by_video(
    keys: Set[Tuple[str, int]]
) -> Dict[str, Set[int]]:
    by_video: Dict[str, Set[int]] = {}
    for video, frame_idx in keys:
        by_video.setdefault(video, set()).add(int(frame_idx))
    return by_video


def _process_video_job(
    inpainted_path: str,
    splatted_path: str,
    done_frames: Optional[Set[int]],
    progress: Optional[ProgressTracker],
    progress_queue: Optional[Any],
    args: argparse.Namespace,
) -> Optional[List[Dict[str, object]]]:
    inpainted_name = os.path.basename(inpainted_path)
    core_with_width, is_sbs_input = parse_inpainted_name(inpainted_name)
    core_name, _width = parse_core_and_width(core_with_width)

    original_path = find_video_by_core_name(args.original_folder, core_name)
    if not original_path or not os.path.exists(original_path):
        print(f"[WARN] missing original for {inpainted_name}; skip")
        return None

    replace_mask_path: Optional[str] = None
    if args.use_replace_mask:
        candidate = find_replace_mask_for_splatted(
            splatted_path, args.replace_mask_folder or ""
        )
        if candidate and os.path.exists(candidate):
            replace_mask_path = candidate

    def _open_all_readers() -> Tuple[VideoReader, VideoReader, VideoReader, Optional[VideoReader]]:
        local_replace_mask_reader: Optional[VideoReader] = None
        if replace_mask_path is not None:
            try:
                local_replace_mask_reader = _open_video_reader(replace_mask_path, args)
            except Exception as e:
                print(f"[WARN] replace mask open failed for {inpainted_name}: {e}")
        inpainted_reader_local = _open_video_reader(inpainted_path, args)
        splatted_reader_local = _open_video_reader(splatted_path, args)
        original_reader_local = _open_video_reader(original_path, args)
        return (
            inpainted_reader_local,
            splatted_reader_local,
            original_reader_local,
            local_replace_mask_reader,
        )

    try:
        (
            inpainted_reader,
            splatted_reader,
            original_reader,
            replace_mask_reader,
        ) = _open_all_readers()
    except Exception as e:
        print(f"[WARN] reader open failed for {inpainted_name}: {e}")
        return None

    n_frames = min(len(inpainted_reader), len(splatted_reader), len(original_reader))
    if replace_mask_reader is not None:
        n_frames = min(n_frames, len(replace_mask_reader))
    if n_frames <= 0:
        print(f"[WARN] zero frames for {inpainted_name}; skip")
        return None

    done_frames_set = set(done_frames or set())
    frame_idx_all = [fi for fi in range(0, n_frames) if fi not in done_frames_set]
    if not frame_idx_all:
        _dbg(args, f"{inpainted_name}: skip, all {n_frames} frames already in CSV")
        return []

    is_dual_input = "_splatted2" in os.path.basename(splatted_path)

    rows: List[Dict[str, object]] = []
    total_frames = len(frame_idx_all)
    min_mask_pixels = max(1, int(args.min_mask_pixels))
    sample_chunk_size = max(1, int(args.sample_chunk_size))
    reload_every_chunks = max(0, int(args.reload_readers_every_chunks))
    chunks_since_reload = 0
    _dbg(
        args,
        (
            f"{inpainted_name}: n_frames={n_frames}, pending={total_frames}, "
            f"chunk={sample_chunk_size}, dual={is_dual_input}, "
            f"reload_every_chunks={reload_every_chunks}, "
            f"use_replace_mask={replace_mask_path is not None}, "
            f"already_done={len(done_frames_set)}"
        ),
    )

    pending_progress_total = 0
    pending_progress_complete = 0

    for chunk_start in range(0, total_frames, sample_chunk_size):
        if (
            reload_every_chunks > 0
            and chunk_start > 0
            and chunks_since_reload >= reload_every_chunks
        ):
            _dbg(
                args,
                f"{inpainted_name}: reloading readers after {chunks_since_reload} chunks",
            )
            try:
                del inpainted_reader, splatted_reader, original_reader
                if replace_mask_reader is not None:
                    del replace_mask_reader
                gc.collect()
                (
                    inpainted_reader,
                    splatted_reader,
                    original_reader,
                    replace_mask_reader,
                ) = _open_all_readers()
                chunks_since_reload = 0
            except Exception as e:
                print(f"[WARN] reader reload failed for {inpainted_name}: {e}")
                break

        chunk_idx = frame_idx_all[chunk_start : chunk_start + sample_chunk_size]
        if not chunk_idx:
            continue
        chunk_tag = (
            f"{inpainted_name} chunk {chunk_start // sample_chunk_size + 1}/"
            f"{(total_frames + sample_chunk_size - 1) // sample_chunk_size} "
            f"(frames {chunk_idx[0]}..{chunk_idx[-1]}, n={len(chunk_idx)})"
        )
        _dbg(args, f"read {chunk_tag}")

        try:
            inpainted_np = inpainted_reader.get_batch(chunk_idx).asnumpy()
            splatted_np = splatted_reader.get_batch(chunk_idx).asnumpy()
            original_np = original_reader.get_batch(chunk_idx).asnumpy()
            replace_mask_np = None
            if replace_mask_reader is not None:
                replace_mask_np = replace_mask_reader.get_batch(chunk_idx).asnumpy()
        except Exception as e:
            print(f"[WARN] read failed for {chunk_tag}: {e}")
            _dbg(args, f"skip {chunk_tag} due to read failure")
            continue

        inpainted_t = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
        splatted_t = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0
        original_left_t = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0

        if is_sbs_input:
            inpainted_t = inpainted_t[:, :, :, inpainted_t.shape[3] // 2 :]

        _, _, h_splat, w_splat = splatted_t.shape
        if is_dual_input:
            mask_raw = splatted_t[:, :, :, : w_splat // 2]
            warped_t = splatted_t[:, :, :, w_splat // 2 :]
        else:
            mask_raw = splatted_t[:, :, h_splat // 2 :, : w_splat // 2]
            warped_t = splatted_t[:, :, h_splat // 2 :, w_splat // 2 :]

        if replace_mask_np is not None:
            if replace_mask_np.ndim == 4 and replace_mask_np.shape[3] >= 1:
                rm_gray = replace_mask_np[..., :3].mean(axis=3)
            elif replace_mask_np.ndim == 3:
                rm_gray = replace_mask_np
            else:
                rm_gray = np.squeeze(replace_mask_np)
            rm_gray = rm_gray.astype(np.float32)
            if rm_gray.size > 0 and float(np.nanmax(rm_gray)) > 1.5:
                rm_gray = rm_gray / 255.0
            rm_t = torch.from_numpy(rm_gray).float().unsqueeze(1)
            if rm_t.shape[2:] != mask_raw.shape[2:]:
                rm_t = torch.nn.functional.interpolate(
                    rm_t, size=mask_raw.shape[2:], mode="nearest"
                )
            mask_raw = rm_t.repeat(1, 3, 1, 1)

        mask_clean = mask_raw[:, 0:1, :, :].float()
        bin_thr = float(args.mask_binarize_threshold)
        if bin_thr >= 0.0:
            mask_bin_clean = (mask_clean > bin_thr).float()
        else:
            mask_bin_clean = (mask_clean > 0.5).float()
        mask_bin = (mask_bin_clean > float(args.mask_threshold)).float()

        for fi in range(inpainted_t.shape[0]):
            frame_no = int(chunk_idx[fi])
            inpainted_3 = inpainted_t[fi].cpu()
            original_3 = original_left_t[fi].cpu()
            warped_3 = warped_t[fi].cpu()
            mask_1hw = mask_bin[fi].cpu()

            mask_pixels = int((mask_1hw.squeeze(0).numpy() > 0.5).sum())
            valid_mask = int(mask_pixels >= min_mask_pixels)
            best_preset_id = 1
            status = "ok"

            if valid_mask:
                try:
                    _best_frame, best_preset_id = _select_best_auto_ct_preset_frame(
                        inpainted_3=inpainted_3,
                        original_left_3=original_3,
                        warped_3=warped_3,
                        mask_bin_1hw=mask_1hw,
                        settings=DEFAULT_CT_SETTINGS,
                        fallback_preset_id=1,
                        executor=None,
                    )
                    best_preset_id = int(best_preset_id)
                except Exception as e:
                    print(
                        f"[WARN] auto-ct selection failed for {inpainted_name} "
                        f"frame#{frame_no}: {e}"
                    )
                    _dbg(
                        args,
                        f"{inpainted_name} frame#{frame_no} failed (mask_px={mask_pixels})",
                    )
                    status = "selector_error"
                    best_preset_id = 1
            else:
                status = "low_mask"

            rows.append(
                {
                    "video": inpainted_name,
                    "frame": _safe_uint(frame_no),
                    "best_preset": _safe_uint(best_preset_id),
                    "valid_mask": _safe_uint(valid_mask),
                    "mask_pixels": _safe_uint(mask_pixels),
                    "status": status,
                }
            )
            is_complete = int(status == "ok")
            if progress_queue is not None:
                pending_progress_total += 1
                pending_progress_complete += is_complete
                if pending_progress_total >= 32:
                    try:
                        progress_queue.put_nowait(
                            (pending_progress_total, pending_progress_complete)
                        )
                    except Exception:
                        pass
                    pending_progress_total = 0
                    pending_progress_complete = 0
            elif progress is not None:
                progress.bump(n_total=1, n_complete=is_complete)

        del inpainted_np, splatted_np, original_np
        del inpainted_t, splatted_t, original_left_t
        del mask_raw, mask_clean, mask_bin_clean, mask_bin, warped_t
        if replace_mask_reader is not None:
            del replace_mask_np
        _dbg(args, f"done {chunk_tag}")
        chunks_since_reload += 1

    if pending_progress_total > 0:
        if progress_queue is not None:
            try:
                progress_queue.put_nowait(
                    (pending_progress_total, pending_progress_complete)
                )
            except Exception:
                pass
        elif progress is not None:
            progress.bump(
                n_total=pending_progress_total,
                n_complete=pending_progress_complete,
            )

    try:
        del inpainted_reader, splatted_reader, original_reader
        if replace_mask_reader is not None:
            del replace_mask_reader
        gc.collect()
    except Exception:
        pass
    return rows


def _process_video_job_mp(
    payload: Tuple[str, str, Set[int], argparse.Namespace, Optional[Any]]
) -> Tuple[str, Optional[List[Dict[str, object]]], Optional[str]]:
    inpainted_path, splatted_path, done_frames, args, progress_queue = payload
    name = os.path.basename(inpainted_path)
    try:
        rows = _process_video_job(
            inpainted_path=inpainted_path,
            splatted_path=splatted_path,
            done_frames=done_frames,
            progress=None,
            progress_queue=progress_queue,
            args=args,
        )
        return name, rows, None
    except Exception as e:
        return name, None, str(e)


def _progress_queue_consumer(
    progress: ProgressTracker,
    q: Any,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            n = q.get(timeout=0.5)
        except queue_mod.Empty:
            continue
        except Exception:
            continue
        try:
            if isinstance(n, tuple) and len(n) == 2:
                progress.bump(n_total=int(n[0]), n_complete=int(n[1]))
            else:
                progress.bump(n_total=int(n), n_complete=0)
        except Exception:
            pass
    while True:
        try:
            n = q.get_nowait()
        except Exception:
            break
        try:
            if isinstance(n, tuple) and len(n) == 2:
                progress.bump(n_total=int(n[0]), n_complete=int(n[1]))
            else:
                progress.bump(n_total=int(n), n_complete=0)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Precompute Auto CT CSV guidance per frame (frame-by-frame)."
    )
    ap.add_argument("--inpainted-folder", required=True)
    ap.add_argument("--splatted-folder", required=True)
    ap.add_argument("--original-folder", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--process-per-file",
        dest="process_per_file",
        action="store_true",
        default=True,
        help="Run each file in a fresh process (maxtasksperchild=1).",
    )
    ap.add_argument(
        "--no-process-per-file",
        dest="process_per_file",
        action="store_false",
        help="Use in-process threaded workers (legacy mode).",
    )
    ap.add_argument("--only", default=None, help="Filename or prefix filter")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--use-replace-mask", action="store_true")
    ap.add_argument("--replace-mask-folder", default="")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--mask-binarize-threshold", type=float, default=-0.01)
    ap.add_argument("--min-mask-pixels", type=int, default=64)
    ap.add_argument(
        "--sample-chunk-size",
        type=int,
        default=1,
        help="Frames processed per read chunk (lower => less RAM).",
    )
    ap.add_argument(
        "--reload-readers-every-chunks",
        type=int,
        default=0,
        help="Reopen decord readers every N chunks (0 disables).",
    )
    ap.add_argument(
        "--progress-every-sec",
        type=float,
        default=30.0,
        help="Print throughput every N seconds (0 disables time trigger).",
    )
    ap.add_argument(
        "--progress-every-frames",
        type=int,
        default=0,
        help="Print throughput every N processed frames (0 disables frame trigger).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output and faulthandler.",
    )
    ap.add_argument(
        "--debug-log",
        default="",
        help="Optional log file for faulthandler traceback output.",
    )
    args = ap.parse_args()
    progress = ProgressTracker(args.progress_every_sec, args.progress_every_frames)

    debug_log_handle = None
    if args.debug:
        try:
            import faulthandler

            if str(args.debug_log).strip():
                dbg_path = os.path.abspath(str(args.debug_log).strip())
                dbg_dir = os.path.dirname(dbg_path)
                if dbg_dir:
                    os.makedirs(dbg_dir, exist_ok=True)
                debug_log_handle = open(dbg_path, "a", encoding="utf-8")
                debug_log_handle.write("\n=== analyze_auto_ct_csv debug session ===\n")
                debug_log_handle.flush()
                faulthandler.enable(file=debug_log_handle, all_threads=True)
                print(f"[DBG] faulthandler enabled: {dbg_path}")
            else:
                faulthandler.enable(all_threads=True)
                print("[DBG] faulthandler enabled on stderr")
        except Exception as e:
            print(f"[WARN] failed to enable faulthandler: {e}")

    columns = [
        "video",
        "frame",
        "best_preset",
        "valid_mask",
        "mask_pixels",
        "status",
    ]

    out_csv = os.path.abspath(args.output_csv)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    jobs = collect_jobs(
        inpainted_folder=args.inpainted_folder,
        splatted_folder=args.splatted_folder,
        original_folder=args.original_folder,
        output_folder="",
        only=args.only,
    )
    if args.max_videos > 0:
        jobs = jobs[: int(args.max_videos)]
    if not jobs:
        print("[ERR] no matching jobs found")
        return 2

    total_jobs = len(jobs)
    existing_frame_keys = _load_existing_frame_keys(out_csv)
    existing_frames_by_video = _build_existing_frames_by_video(existing_frame_keys)
    if existing_frame_keys:
        print(
            f"[RESUME] existing frame rows: {len(existing_frame_keys)} | "
            f"jobs to scan: {total_jobs}"
        )

    csv_exists = os.path.exists(out_csv)
    write_header = (not csv_exists) or (os.path.getsize(out_csv) == 0)

    written_rows: Set[Tuple[str, int]] = set(existing_frame_keys)
    written_new = 0
    skipped_dup = 0

    out_handle = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_handle, fieldnames=columns)
    if write_header:
        writer.writeheader()
        out_handle.flush()
        os.fsync(out_handle.fileno())

    def _append_rows(rows: List[Dict[str, object]]) -> None:
        nonlocal written_new, skipped_dup
        for row in rows:
            video = str(row.get("video", "")).strip()
            frame_raw = row.get("frame", None)
            if not video:
                continue
            try:
                frame_idx = int(frame_raw)
            except Exception:
                continue
            key = (video, frame_idx)
            if key in written_rows:
                skipped_dup += 1
                continue
            writer.writerow(row)
            written_rows.add(key)
            written_new += 1
        out_handle.flush()
        os.fsync(out_handle.fileno())

    workers = max(1, int(args.workers))
    progress_queue: Optional[Any] = None
    progress_stop_event: Optional[threading.Event] = None
    progress_thread: Optional[threading.Thread] = None
    progress_manager: Optional[Any] = None

    if args.process_per_file and workers > 1:
        progress_stop_event = threading.Event()
        progress_manager = mp.Manager()
        progress_queue = progress_manager.Queue(max(2048, workers * 256))
        progress_thread = threading.Thread(
            target=_progress_queue_consumer,
            args=(progress, progress_queue, progress_stop_event),
            daemon=True,
        )
        progress_thread.start()

    try:
        if workers == 1:
            for idx, (inpainted_path, splatted_path) in enumerate(jobs, 1):
                name = os.path.basename(inpainted_path)
                print(f"[{idx}/{len(jobs)}] {name}")
                done_frames = existing_frames_by_video.get(name, set())
                rows = _process_video_job(
                    inpainted_path,
                    splatted_path,
                    done_frames,
                    progress,
                    None,
                    args,
                )
                if rows is not None:
                    _append_rows(rows)
        elif args.process_per_file:
            # Hard reset of the worker chain after each file.
            # maxtasksperchild=1 guarantees a fresh process per scene.
            ctx = mp.get_context("fork")
            payloads = [
                (
                    inpainted_path,
                    splatted_path,
                    set(existing_frames_by_video.get(os.path.basename(inpainted_path), set())),
                    args,
                    progress_queue,
                )
                for (inpainted_path, splatted_path) in jobs
            ]
            with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
                done_count = 0
                for name, rows, err in pool.imap_unordered(_process_video_job_mp, payloads):
                    done_count += 1
                    if err:
                        print(f"[WARN] worker failed for {name}: {err}")
                    if rows is not None:
                        _append_rows(rows)
                    print(f"[DONE] {done_count}/{len(jobs)} {name}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_name: Dict[concurrent.futures.Future, str] = {}
                for inpainted_path, splatted_path in jobs:
                    name = os.path.basename(inpainted_path)
                    done_frames = existing_frames_by_video.get(name, set())
                    fut = ex.submit(
                        _process_video_job,
                        inpainted_path,
                        splatted_path,
                        done_frames,
                        progress,
                        None,
                        args,
                    )
                    fut_to_name[fut] = name
                done_count = 0
                for fut in concurrent.futures.as_completed(fut_to_name):
                    done_count += 1
                    name = fut_to_name[fut]
                    try:
                        rows = fut.result()
                    except Exception as e:
                        print(f"[WARN] worker failed for {name}: {e}")
                        rows = None
                    if rows is not None:
                        _append_rows(rows)
                    print(f"[DONE] {done_count}/{len(jobs)} {name}")
    finally:
        out_handle.close()
        if progress_stop_event is not None:
            progress_stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=2.0)
        if progress_manager is not None:
            try:
                progress_manager.shutdown()
            except Exception:
                pass
        if debug_log_handle is not None:
            debug_log_handle.flush()
            debug_log_handle.close()
    progress.final_report()

    print(f"[OK] updated CSV: {out_csv}")
    print(f"[OK] existing frame rows before run: {len(existing_frame_keys)}")
    print(f"[OK] newly written frame rows this run: {written_new}")
    print(f"[OK] duplicate frame rows skipped this run: {skipped_dup}")
    print(f"[OK] total unique frame rows in CSV: {len(written_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
