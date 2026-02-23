#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import glob
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np

try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:
    BrokenProcessPool = RuntimeError


# -------------------------
# RECOMMENDED DEFAULTS
# -------------------------
DEFAULT_GLOB = "*.mp4"
DEFAULT_OUT_CSV = "sharpness.csv"
DEFAULT_SAMPLE_FRAMES = 48
DEFAULT_THR = 100
DEFAULT_MASK_DILATE_K = 0
DEFAULT_MASK_DILATE_ITER = 0
DEFAULT_BAND_MODE = "match_run"  # match_run | fixed
DEFAULT_BAND_PX = 10
DEFAULT_BAND_GAP_PX = 0
DEFAULT_RIGHT_BORDER_TOL_PX = 2
DEFAULT_MIN_ROI_PIXELS = 250
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))


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
    band_mode: str,
    band_px: int,
    band_gap_px: int,
) -> np.ndarray:
    """
    Build a dynamic ROI immediately to the right of each white run in the mask.

    Default behavior builds a right-side band per mask run.
    For runs that touch the RIGHT border (with tolerance), build the band on
    the LEFT side instead.

    Direction is decided per row/run (not per connected component), so the
    side can switch inside the same shape when rows stop touching the border.
    """
    _, m = cv2.threshold(mask_gray, int(thr), 255, cv2.THRESH_BINARY)
    base = m.astype(np.uint8)

    h, w = base.shape
    roi = np.zeros((h, w), dtype=np.uint8)

    if band_mode not in ("match_run", "fixed"):
        band_mode = "match_run"
    if band_mode == "fixed" and band_px <= 0:
        return roi

    border_cols = max(1, min(int(DEFAULT_RIGHT_BORDER_TOL_PX), w))
    right_touch_start = w - border_cols

    row_has = np.any(base > 0, axis=1)
    for y in np.where(row_has)[0]:
        row = base[y] > 0

        # Run starts/ends.
        starts = np.where(row & np.r_[True, ~row[:-1]])[0]
        ends = np.where(row & np.r_[~row[1:], True])[0]
        if starts.size == 0 or ends.size == 0:
            continue

        for xs, xe in zip(starts, ends):
            run_len = int(xe - xs + 1)
            width = run_len if band_mode == "match_run" else int(band_px)
            if width <= 0:
                continue
            touches_left = int(xs) < border_cols
            touches_right = int(xe) >= right_touch_start
            prefer_left = touches_right and not touches_left

            if prefer_left:
                x1 = int(xs) - int(band_gap_px)
                x0 = x1 - width
                x0c = max(0, x0)
                x1c = min(w, x1)
                if x0c < x1c:
                    roi[y, x0c:x1c] = 255
                    continue
                # Fallback to right side if left side band is out of bounds.

            x0 = int(xe) + 1 + int(band_gap_px)
            x1 = min(w, x0 + width)
            if x0 < x1:
                roi[y, x0:x1] = 255

    # Ensure ROI excludes original mask pixels.
    roi = cv2.bitwise_and(roi, cv2.bitwise_not(base))
    return roi


def compute_file_sharpness(
    path: str,
    mask_path: Optional[str],
    sample_frames: int,
    thr: int,
    mask_dilate_k: int,
    mask_dilate_iter: int,
    band_mode: str,
    band_px: int,
    band_gap_px: int,
    min_roi_pixels: int,
) -> Tuple[float, int, float]:
    container = av.open(path)
    stream = container.streams.video[0]

    mask_container = av.open(mask_path) if mask_path else None
    mask_stream = mask_container.streams.video[0] if mask_container else None

    fps = float(stream.average_rate) if stream.average_rate else None
    total_est = None
    if stream.duration is not None and fps is not None:
        secs = float(stream.duration * stream.time_base)
        if secs > 0:
            total_est = int(secs * fps)

    if total_est and total_est > 0:
        stride = max(1, total_est // max(1, sample_frames))
    else:
        stride = 10

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
            band_mode=band_mode,
            band_px=band_px,
            band_gap_px=band_gap_px,
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
        if picked >= sample_frames:
            break

    container.close()
    if mask_container is not None:
        mask_container.close()

    if not sharp_vals:
        return 0.0, 0, 0.0

    sharp_raw = float(np.median(sharp_vals))
    cov_med = float(np.median(cov_vals)) if cov_vals else 0.0
    return sharp_raw, len(sharp_vals), cov_med


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
        sample_frames,
        thr,
        mask_dilate_k,
        mask_dilate_iter,
        band_mode,
        band_px,
        band_gap_px,
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
            sample_frames=sample_frames,
            thr=thr,
            mask_dilate_k=mask_dilate_k,
            mask_dilate_iter=mask_dilate_iter,
            band_mode=band_mode,
            band_px=band_px,
            band_gap_px=band_gap_px,
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
            "Analyze sharpness on a dynamic right-side band next to each mask run "
            "(per-row, no fixed global shift)."
        )
    )
    ap.add_argument("in_dir")
    ap.add_argument("mask_dir", nargs="?", default=None)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--sample_frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    ap.add_argument("--thr", type=int, default=DEFAULT_THR)
    ap.add_argument("--mask_dilate_k", type=int, default=DEFAULT_MASK_DILATE_K)
    ap.add_argument("--mask_dilate_iter", type=int, default=DEFAULT_MASK_DILATE_ITER)
    ap.add_argument(
        "--band_mode",
        type=str,
        default=DEFAULT_BAND_MODE,
        choices=["match_run", "fixed"],
        help="Right band width policy: match each mask-run width, or fixed width",
    )
    ap.add_argument("--band_px", type=int, default=DEFAULT_BAND_PX, help="Right-side analysis band width in pixels")
    ap.add_argument("--band_gap_px", type=int, default=DEFAULT_BAND_GAP_PX, help="Gap from mask edge before band")
    ap.add_argument("--min_roi_pixels", type=int, default=DEFAULT_MIN_ROI_PIXELS)
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers (processes). 1 = sequential",
    )
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    paths = sorted(glob.glob(os.path.join(in_dir, args.glob)))
    if not paths:
        raise SystemExit(f"No files found: {in_dir}/{args.glob}")

    out_csv = os.path.abspath(args.out_csv)
    existing = load_existing_csv(out_csv)

    reused = 0
    computed = 0
    results: Dict[str, Tuple[float, int, float]] = {}

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
                args.sample_frames,
                args.thr,
                args.mask_dilate_k,
                args.mask_dilate_iter,
                args.band_mode,
                args.band_px,
                args.band_gap_px,
                args.min_roi_pixels,
            )
        )

    # Compute (sequential or parallel)
    if args.workers <= 1 or len(jobs) <= 1:
        for job in jobs:
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
            ) as ex:
                fut_to_job = {ex.submit(_worker_compute, job): job for job in jobs}
                for fut in concurrent.futures.as_completed(fut_to_job):
                    try:
                        bn, raw, n, cov, status = fut.result()
                    except Exception as e:
                        # If the pool dies abruptly, recover in sequential mode below.
                        if isinstance(e, BrokenProcessPool):
                            pool_broken = True
                            print(f"[WARN] process pool broken, fallback to sequential: {e}")
                            ex.shutdown(wait=False, cancel_futures=True)
                            break
                        job = fut_to_job[fut]
                        bn = os.path.basename(job[0])
                        raw, n, cov, status = 0.0, 0, 0.0, f"ERR:{type(e).__name__}"
                    _print_status(bn, raw, n, cov, status, mask_dir)
                    results[bn] = (raw, n, cov)
                    computed += 1
        except Exception as e:
            if isinstance(e, BrokenProcessPool):
                pool_broken = True
                print(f"[WARN] process pool setup/teardown broken, fallback to sequential: {e}")
            else:
                raise

        if pool_broken:
            remaining_jobs = [job for job in jobs if os.path.basename(job[0]) not in results]
            print(f"[INFO] recovering remaining jobs sequentially: {len(remaining_jobs)}")
            for job in remaining_jobs:
                bn, raw, n, cov, status = _worker_compute(job)
                _print_status(bn, raw, n, cov, status, mask_dir)
                results[bn] = (raw, n, cov)
                computed += 1

    # Stable order
    tmp: List[Tuple[str, float, int, float]] = []
    for p in paths:
        bn = os.path.basename(p)
        raw, n, cov = results.get(bn, (0.0, 0, 0.0))
        tmp.append((bn, raw, n, cov))

    raws = [x[1] for x in tmp]
    pcts = robust_percent(raws)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "sharpness_raw", "sharpness_pct", "samples_used", "roi_coverage_pct"])
        for (bn, raw, n, cov), pct in zip(tmp, pcts):
            w.writerow([bn, f"{raw:.6f}", f"{pct:.2f}", n, f"{cov:.3f}"])

    print(
        f"\nDone: {out_csv}  "
        f"(reused={reused}, computed={computed}, total={len(tmp)})  "
        f"band_mode={args.band_mode} band_px={args.band_px} gap={args.band_gap_px}"
    )


if __name__ == "__main__":
    main()
