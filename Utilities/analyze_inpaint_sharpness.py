#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import glob
import multiprocessing as mp
import os
import signal
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import torch

try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:
    BrokenProcessPool = RuntimeError


# -------------------------
# RECOMMENDED DEFAULTS
# -------------------------
DEFAULT_GLOB = "*.mp4"
DEFAULT_OUT_CSV = "sharpness.csv"
DEFAULT_FRAME_STRIDE = 6
DEFAULT_MAX_SAMPLES = 0  # 0 = no cap, scan full video at stride
DEFAULT_AGG_MODE = "upper_trimmed"  # upper_trimmed | median | mean
DEFAULT_AGG_LOW_PCT = 20.0
DEFAULT_AGG_HIGH_PCT = 98.0
DEFAULT_AGG_TOP_RATIO = 0.40
DEFAULT_THR = 100
DEFAULT_MASK_DILATE_K = 0
DEFAULT_MASK_DILATE_ITER = 0
DEFAULT_RIGHT_BORDER_TOL_PX = 2
DEFAULT_MIN_ROI_PIXELS = 250
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))
_STOP_REQUESTED = False


def _set_stop_requested() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _stop_requested(stop_marker: str) -> bool:
    if _STOP_REQUESTED:
        return True
    marker = str(stop_marker or "").strip()
    return bool(marker) and os.path.isfile(marker)


def _handle_signal(_signum, _frame) -> None:
    _set_stop_requested()


def _init_worker_ignore_signals() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _clear_stop_marker(stop_marker: str) -> None:
    marker = str(stop_marker or "").strip()
    if not marker:
        return
    try:
        if os.path.isfile(marker):
            os.remove(marker)
    except Exception:
        pass


def find_mask_for_video(mask_dir: str, video_basename: str) -> Optional[str]:
    """
    Find external replace-mask video for a given input video basename.

    Convention: <stem>_replace_mask.<any extension>
    Example: foo_splatted2.mp4 -> foo_splatted2_replace_mask.mkv
    """
    if not mask_dir:
        return None
    stem, _ext = os.path.splitext(video_basename)
    patt = os.path.join(mask_dir, stem + "_replace_mask.*")
    matches = sorted(glob.glob(patt))
    return matches[0] if matches else None


def tenengrad(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def make_right_band_roi(
    mask_gray: np.ndarray,
    thr: int,
) -> np.ndarray:
    """
    Build analysis ROI from the same row/run ring-shift logic used by
    merging_gui reference-mask generation.

    ROI marks the source pixels selected for copy (not the destination run),
    so sharpness is measured on real warped content.
    """
    if mask_gray.ndim != 2:
        return np.zeros((0, 0), dtype=np.uint8)

    _, m = cv2.threshold(mask_gray, int(thr), 255, cv2.THRESH_BINARY)
    mask_bin = torch.as_tensor(m > 0, dtype=torch.bool, device="cpu")
    h, w = mask_bin.shape
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    if not bool(mask_bin.any().item()):
        return np.zeros((h, w), dtype=np.uint8)

    roi = torch.zeros((h, w), dtype=torch.bool, device="cpu")
    border_cols = max(1, min(int(DEFAULT_RIGHT_BORDER_TOL_PX), w))
    right_touch_start = w - border_cols

    def _nearest_nonmask_x(y: int, start_x: int, step: int) -> int:
        x = int(start_x)
        while 0 <= x < w:
            if not bool(mask_bin[y, x].item()):
                return x
            x += step
        return -1

    def _mark_block_if_valid(y: int, src_a: int, src_b: int) -> bool:
        # [src_a, src_b)
        if src_a < 0 or src_b > w or src_a >= src_b:
            return False
        if bool(mask_bin[y, src_a:src_b].any().item()):
            return False
        roi[y, src_a:src_b] = True
        return True

    for y in range(h):
        xs = torch.where(mask_bin[y])[0]
        if int(xs.numel()) == 0:
            continue

        run_start = int(xs[0].item())
        prev = run_start

        def _mark_run(a: int, b: int) -> None:
            run_len = int(b - a + 1)
            touches_left = a < border_cols
            touches_right = b >= right_touch_start
            prefer_left = touches_right and not touches_left

            marked = False
            if prefer_left:
                marked = _mark_block_if_valid(y, a - run_len, a)
                if not marked:
                    marked = _mark_block_if_valid(y, b + 1, b + 1 + run_len)
            else:
                marked = _mark_block_if_valid(y, b + 1, b + 1 + run_len)
                if not marked:
                    marked = _mark_block_if_valid(y, a - run_len, a)
            if marked:
                return

            if prefer_left:
                src_x = _nearest_nonmask_x(y, a - 1, -1)
                if src_x < 0:
                    src_x = _nearest_nonmask_x(y, b + 1, 1)
            else:
                src_x = _nearest_nonmask_x(y, b + 1, 1)
                if src_x < 0:
                    src_x = _nearest_nonmask_x(y, a - 1, -1)
            if src_x >= 0:
                roi[y, src_x] = True

        for idx in range(1, int(xs.numel())):
            cur = int(xs[idx].item())
            if cur != prev + 1:
                _mark_run(run_start, prev)
                run_start = cur
            prev = cur
        _mark_run(run_start, prev)

    roi = torch.logical_and(roi, ~mask_bin)
    return (roi.to(torch.uint8).numpy() * 255).astype(np.uint8)


def compute_file_sharpness(
    path: str,
    mask_path: Optional[str],
    frame_stride: int,
    max_samples: int,
    agg_mode: str,
    agg_low_pct: float,
    agg_high_pct: float,
    agg_top_ratio: float,
    thr: int,
    mask_dilate_k: int,
    mask_dilate_iter: int,
    min_roi_pixels: int,
) -> Tuple[float, int, float]:
    container = av.open(path)

    mask_container = av.open(mask_path) if mask_path else None

    stride = max(1, int(frame_stride))

    sharp_vals: List[float] = []
    cov_vals: List[float] = []

    idx = 0
    picked = 0

    mask_iter = mask_container.decode(video=0) if mask_container else None
    for frame in container.decode(video=0):
        idx += 1
        mframe = None
        if mask_iter is not None:
            try:
                mframe = next(mask_iter)
            except StopIteration:
                break
        if (idx - 1) % stride != 0:
            continue

        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        if w < 2:
            continue

        half = w // 2
        mask_embedded = img[:, :half, :]
        warped = img[:, half : half + half, :]
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        if mask_iter is not None:
            # External replace-mask.
            try:
                m = mframe.to_ndarray(format="gray")
                mask_gray = m if m.ndim == 2 else m[:, :, 0]
            except Exception:
                mbgr = mframe.to_ndarray(format="bgr24")
                mask_gray = cv2.cvtColor(mbgr, cv2.COLOR_BGR2GRAY)

            # Ensure mask matches warped size (H x W_right).
            mh, mw = mask_gray.shape[:2]
            wh, ww = warped_gray.shape[:2]
            if (mh, mw) != (wh, ww):
                if mh == wh and mw > ww:
                    mask_gray = mask_gray[:, -ww:]
                    mh, mw = mask_gray.shape[:2]
                if (mh, mw) != (wh, ww):
                    continue
        else:
            mask_gray = cv2.cvtColor(mask_embedded, cv2.COLOR_BGR2GRAY)

        # Optional dilation on mask before building right-band ROI.
        if int(mask_dilate_k) > 0 and int(mask_dilate_iter) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(mask_dilate_k), int(mask_dilate_k)))
            mask_gray = cv2.dilate(mask_gray, k, iterations=int(mask_dilate_iter))

        roi = make_right_band_roi(
            mask_gray=mask_gray,
            thr=thr,
        )
        roi_pixels = int(np.count_nonzero(roi))
        if roi_pixels < int(min_roi_pixels):
            continue

        E = tenengrad(warped_gray)
        m = roi.astype(bool)
        val = float(np.mean(E[m]))
        sharp_vals.append(val)

        cov = 100.0 * roi_pixels / float(roi.size)
        cov_vals.append(float(cov))

        picked += 1
        if int(max_samples) > 0 and picked >= int(max_samples):
            break

    container.close()
    if mask_container is not None:
        mask_container.close()

    if not sharp_vals:
        return 0.0, 0, 0.0

    sharp_raw = aggregate_sharpness(
        sharp_vals,
        mode=agg_mode,
        low_pct=agg_low_pct,
        high_pct=agg_high_pct,
        top_ratio=agg_top_ratio,
    )
    cov_med = float(np.median(cov_vals)) if cov_vals else 0.0
    return sharp_raw, len(sharp_vals), cov_med


def aggregate_sharpness(
    values: List[float],
    mode: str,
    low_pct: float,
    high_pct: float,
    top_ratio: float,
) -> float:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0

    mode = str(mode).strip().lower()
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))

    # upper_trimmed: drop extremes, then average top fraction to favor sharp regions.
    lp = float(np.clip(low_pct, 0.0, 100.0))
    hp = float(np.clip(high_pct, 0.0, 100.0))
    if hp < lp:
        lp, hp = hp, lp

    lo = float(np.percentile(arr, lp))
    hi = float(np.percentile(arr, hp))
    core = arr[(arr >= lo) & (arr <= hi)]
    if core.size == 0:
        core = arr

    ratio = float(np.clip(top_ratio, 1e-6, 1.0))
    k = max(1, int(np.ceil(core.size * ratio)))
    kth = max(0, int(core.size - k))
    top = np.partition(core, kth)[kth:]
    return float(np.mean(top))


def robust_percent(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float32)
    if len(arr) == 0:
        return []
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    if p95 <= p5 + 1e-9:
        return [50.0 for _ in values]
    pct = (arr - p5) * 100.0 / (p95 - p5)
    pct = np.clip(pct, 0.0, 100.0)
    return [float(x) for x in pct]


def load_existing_csv(path: str) -> Dict[str, Tuple[float, int, float]]:
    """
    Returns: { basename -> (sharp_raw, samples_used, roi_coverage_pct) }
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}

    cache: Dict[str, Tuple[float, int, float]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                name = row.get("file") or ""
                if not name:
                    continue
                raw = float(row.get("sharpness_raw", "0") or 0)
                samples = int(float(row.get("samples_used", "0") or 0))
                cov = float(row.get("roi_coverage_pct", "0") or 0)
                cache[name] = (raw, samples, cov)
            except Exception:
                continue
    return cache


def _worker_compute(job):
    (
        p,
        mask_dir,
        frame_stride,
        max_samples,
        agg_mode,
        agg_low_pct,
        agg_high_pct,
        agg_top_ratio,
        thr,
        mask_dilate_k,
        mask_dilate_iter,
        min_roi_pixels,
    ) = job
    bn = os.path.basename(p)
    try:
        mask_path = find_mask_for_video(mask_dir, bn) if mask_dir else None
        if mask_dir and not mask_path:
            return (bn, 0.0, 0, 0.0, "MISS_MASK")
        sharp_raw, n, cov = compute_file_sharpness(
            path=p,
            mask_path=mask_path,
            frame_stride=frame_stride,
            max_samples=max_samples,
            agg_mode=agg_mode,
            agg_low_pct=agg_low_pct,
            agg_high_pct=agg_high_pct,
            agg_top_ratio=agg_top_ratio,
            thr=thr,
            mask_dilate_k=mask_dilate_k,
            mask_dilate_iter=mask_dilate_iter,
            min_roi_pixels=min_roi_pixels,
        )
        return (bn, float(sharp_raw), int(n), float(cov), "OK")
    except Exception as e:
        return (bn, 0.0, 0, 0.0, f"ERR:{type(e).__name__}")


def _print_status(bn: str, raw: float, n: int, cov: float, status: str, mask_dir: Optional[str]) -> None:
    if status == "MISS_MASK":
        print(f"[MISS MASK] {bn}  (looked for: {mask_dir}/{os.path.splitext(bn)[0]}_replace_mask.*)")
    elif status.startswith("ERR:"):
        print(f"[ERR]  {bn}  {status}")
    else:
        print(f"[OK]   {bn}  raw={raw:.2f}  samples={n}  roi_cov={cov:.2f}%")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Analyze sharpness on a ring-shift source ROI aligned with "
            "merging_gui reference-mask logic (row/run-aware)."
        )
    )
    ap.add_argument("in_dir")
    ap.add_argument("mask_dir", nargs="?", default=None)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--frame_stride", type=int, default=DEFAULT_FRAME_STRIDE)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Optional cap on sampled frames. 0 = no cap (full video at stride)",
    )
    ap.add_argument(
        "--agg_mode",
        type=str,
        default=DEFAULT_AGG_MODE,
        choices=["upper_trimmed", "median", "mean"],
        help="How to aggregate per-frame sharpness into file sharpness",
    )
    ap.add_argument(
        "--agg_low_pct",
        type=float,
        default=DEFAULT_AGG_LOW_PCT,
        help="Lower percentile for upper_trimmed aggregation",
    )
    ap.add_argument(
        "--agg_high_pct",
        type=float,
        default=DEFAULT_AGG_HIGH_PCT,
        help="Upper percentile for upper_trimmed aggregation",
    )
    ap.add_argument(
        "--agg_top_ratio",
        type=float,
        default=DEFAULT_AGG_TOP_RATIO,
        help="Top fraction kept after percentile trim for upper_trimmed aggregation",
    )
    ap.add_argument("--thr", type=int, default=DEFAULT_THR)
    ap.add_argument("--mask_dilate_k", type=int, default=DEFAULT_MASK_DILATE_K)
    ap.add_argument("--mask_dilate_iter", type=int, default=DEFAULT_MASK_DILATE_ITER)
    ap.add_argument("--min_roi_pixels", type=int, default=DEFAULT_MIN_ROI_PIXELS)
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers (processes). 1 = sequential",
    )
    ap.add_argument(
        "--stop-marker",
        default="",
        help="Optional graceful-stop marker file. Default: alongside out_csv.",
    )
    args = ap.parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    in_dir = os.path.abspath(args.in_dir)
    paths = sorted(glob.glob(os.path.join(in_dir, args.glob)))
    if not paths:
        raise SystemExit(f"No files found: {in_dir}/{args.glob}")

    out_csv = os.path.abspath(args.out_csv)
    stop_marker = (
        os.path.abspath(str(args.stop_marker).strip())
        if str(args.stop_marker).strip()
        else os.path.join(os.path.dirname(out_csv) or os.getcwd(), ".stop_after_current")
    )
    if os.path.isfile(stop_marker):
        print(f"[INFO] removing stale stop marker: {stop_marker}")
        _clear_stop_marker(stop_marker)
    existing = load_existing_csv(out_csv)

    reused = 0
    computed = 0
    results: Dict[str, Tuple[float, int, float]] = {}
    stop_logged = False

    def _note_stop_requested() -> bool:
        nonlocal stop_logged
        if not _stop_requested(stop_marker):
            return False
        if not stop_logged:
            print("[STOP] graceful stop requested. Waiting current file(s) to finish.")
            stop_logged = True
        return True

    # Reuse from existing CSV
    for p in paths:
        bn = os.path.basename(p)
        if bn in existing:
            raw, n, cov = existing[bn]
            reused += 1
            results[bn] = (raw, n, cov)
            print(f"[SKIP] {bn}  raw={raw:.2f}  samples={n}  roi_cov={cov:.2f}%  (from CSV)")

    mask_dir = os.path.abspath(args.mask_dir) if args.mask_dir else None
    jobs = []
    for p in paths:
        bn = os.path.basename(p)
        if bn in results:
            continue
        jobs.append(
            (
                p,
                mask_dir,
                args.frame_stride,
                args.max_samples,
                args.agg_mode,
                args.agg_low_pct,
                args.agg_high_pct,
                args.agg_top_ratio,
                args.thr,
                args.mask_dilate_k,
                args.mask_dilate_iter,
                args.min_roi_pixels,
            )
        )

    # Compute (sequential or parallel)
    if args.workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            if _note_stop_requested():
                break
            bn, raw, n, cov, status = _worker_compute(job)
            _print_status(bn, raw, n, cov, status, mask_dir)
            results[bn] = (raw, n, cov)
            computed += 1
    else:
        pool_broken = False
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker_ignore_signals,
            ) as ex:
                pending_jobs = list(jobs)
                fut_to_job: Dict[concurrent.futures.Future, tuple] = {}
                while pending_jobs or fut_to_job:
                    while (
                        pending_jobs
                        and len(fut_to_job) < max(1, int(args.workers))
                        and not _stop_requested(stop_marker)
                    ):
                        job = pending_jobs.pop(0)
                        fut_to_job[ex.submit(_worker_compute, job)] = job
                    _note_stop_requested()
                    if not fut_to_job:
                        break
                    done, _pending = concurrent.futures.wait(
                        tuple(fut_to_job.keys()),
                        timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        continue
                    for fut in done:
                        job = fut_to_job.pop(fut)
                        try:
                            bn, raw, n, cov, status = fut.result()
                        except Exception as e:
                            # If the pool dies abruptly, recover in sequential mode below.
                            if isinstance(e, BrokenProcessPool):
                                pool_broken = True
                                print(f"[WARN] process pool broken, fallback to sequential: {e}")
                                ex.shutdown(wait=False, cancel_futures=True)
                                fut_to_job.clear()
                                break
                            bn = os.path.basename(job[0])
                            raw, n, cov, status = 0.0, 0, 0.0, f"ERR:{type(e).__name__}"
                        _print_status(bn, raw, n, cov, status, mask_dir)
                        results[bn] = (raw, n, cov)
                        computed += 1
                    if pool_broken:
                        break
        except Exception as e:
            if isinstance(e, BrokenProcessPool):
                pool_broken = True
                print(f"[WARN] process pool setup/teardown broken, fallback to sequential: {e}")
            else:
                raise

        if pool_broken and not _stop_requested(stop_marker):
            remaining_jobs = [job for job in jobs if os.path.basename(job[0]) not in results]
            print(f"[INFO] recovering remaining jobs sequentially: {len(remaining_jobs)}")
            for job in remaining_jobs:
                if _note_stop_requested():
                    break
                bn, raw, n, cov, status = _worker_compute(job)
                _print_status(bn, raw, n, cov, status, mask_dir)
                results[bn] = (raw, n, cov)
                computed += 1

    # Stable order
    tmp: List[Tuple[str, float, int, float]] = []
    for p in paths:
        bn = os.path.basename(p)
        if bn not in results:
            continue
        raw, n, cov = results[bn]
        tmp.append((bn, raw, n, cov))

    raws = [x[1] for x in tmp]
    pcts = robust_percent(raws)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "sharpness_raw", "sharpness_pct", "samples_used", "roi_coverage_pct"])
        for (bn, raw, n, cov), pct in zip(tmp, pcts):
            w.writerow([bn, f"{raw:.6f}", f"{pct:.2f}", n, f"{cov:.3f}"])

    if _stop_requested(stop_marker):
        _clear_stop_marker(stop_marker)

    print(
        f"\nDone: {out_csv}  "
        f"(reused={reused}, computed={computed}, total={len(tmp)})  "
        f"stride={args.frame_stride} max_samples={args.max_samples} "
        f"agg={args.agg_mode} "
        "roi_mode=ring_shift_source"
    )
    if _stop_requested(stop_marker):
        remaining = max(0, len(paths) - len(tmp))
        print(
            f"[STOP] graceful stop completed: remaining_files={remaining} "
            f"marker={stop_marker}"
        )


if __name__ == "__main__":
    main()
