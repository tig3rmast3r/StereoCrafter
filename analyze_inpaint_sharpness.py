#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import concurrent.futures
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import av
import numpy as np
import cv2


@dataclass
class Result:
    path: str
    sharp_raw: float
    sharp_pct: float
    samples_used: int
    roi_cov_pct: float



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


def make_roi(mask_gray: np.ndarray, thr: int, dilate_k: int, dilate_iter: int, shift_x: int) -> Tuple[np.ndarray, np.ndarray]:
    _, m = cv2.threshold(mask_gray, thr, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    base = cv2.dilate(m, k, iterations=dilate_iter)

    h, w = base.shape
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    shifted = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    roi = cv2.bitwise_and(shifted, cv2.bitwise_not(base))
    return base, roi


def compute_file_sharpness(
    path: str,
    mask_path: Optional[str],
    sample_frames: int,
    thr: int,
    dilate_k: int,
    dilate_iter: int,
    shift_x: int,
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
        mask = img[:, :half, :]
        warped = img[:, half:half + half, :]

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        if mask_iter is not None:
            # External replace-mask is already binary; use it as BASE mask,
            # then build ROI by shifting it (so we do not analyze "holes").
            try:
                m = mframe.to_ndarray(format="gray")
                mask_gray_ext = m if m.ndim == 2 else m[:, :, 0]
            except Exception:
                mbgr = mframe.to_ndarray(format="bgr24")
                mask_gray_ext = cv2.cvtColor(mbgr, cv2.COLOR_BGR2GRAY)

            # Ensure BASE mask matches warped size (H x W_right).
            mh, mw = mask_gray_ext.shape[:2]
            wh, ww = warped_gray.shape[:2]
            if (mh, mw) != (wh, ww):
                # If mask is wider (e.g. full width), keep the right-most ww pixels.
                if mh == wh and mw > ww:
                    mask_gray_ext = mask_gray_ext[:, -ww:]
                    mh, mw = mask_gray_ext.shape[:2]
                if (mh, mw) != (wh, ww):
                    continue

            base = (mask_gray_ext > 0).astype(np.uint8) * 255

            h2, w2 = base.shape
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            shifted = cv2.warpAffine(base, M, (w2, h2), flags=cv2.INTER_NEAREST, borderValue=0)
            roi = cv2.bitwise_and(shifted, cv2.bitwise_not(base))
        else:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, roi = make_roi(mask_gray, thr, dilate_k, dilate_iter, shift_x)

        roi_pixels = int(np.count_nonzero(roi))
        if roi_pixels < min_roi_pixels:
            if mask_iter is not None:
                continue
            _, base = cv2.threshold(mask_gray, thr, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            dil = cv2.dilate(base, k, iterations=dilate_iter)
            ero = cv2.erode(base, k, iterations=max(1, dilate_iter))
            roi = cv2.bitwise_and(dil, cv2.bitwise_not(ero))
            roi_pixels = int(np.count_nonzero(roi))
            if roi_pixels < min_roi_pixels:
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
        # expected headers: file, sharpness_raw, sharpness_pct, samples_used, roi_coverage_pct
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
    """Worker job for parallel execution."""
    (p, mask_dir, sample_frames, thr, dilate_k, dilate_iter, shift_x, min_roi_pixels) = job
    bn = os.path.basename(p)
    try:
        mask_path = find_mask_for_video(mask_dir, bn) if mask_dir else None
        if mask_dir and not mask_path:
            return (bn, 0.0, 0, 0.0, "MISS_MASK")
        sharp_raw, n, cov = compute_file_sharpness(
            p,
            mask_path,
            sample_frames=sample_frames,
            thr=thr,
            dilate_k=dilate_k,
            dilate_iter=dilate_iter,
            shift_x=shift_x,
            min_roi_pixels=min_roi_pixels,
        )
        return (bn, float(sharp_raw), int(n), float(cov), "OK")
    except Exception as e:
        return (bn, 0.0, 0, 0.0, f"ERR:{type(e).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("mask_dir", nargs="?", default=None)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--out_csv", default="sharpness.csv")
    ap.add_argument("--sample_frames", type=int, default=30)
    ap.add_argument("--thr", type=int, default=100)
    ap.add_argument("--dilate_k", type=int, default=9)
    ap.add_argument("--dilate_iter", type=int, default=2)
    ap.add_argument("--shift_x", type=int, default=20)
    ap.add_argument("--min_roi_pixels", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (processes). 1 = sequential")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    paths = sorted(glob.glob(os.path.join(in_dir, args.glob)))
    if not paths:
        raise SystemExit(f"No files found: {in_dir}/{args.glob}")

    out_csv = os.path.abspath(args.out_csv)
    existing = load_existing_csv(out_csv)

    tmp: List[Tuple[str, float, int, float]] = []  # (basename, raw, samples, cov)

    # Build results in a stable order (same order as `paths`)
    reused = 0
    computed = 0
    results: Dict[str, Tuple[float, int, float]] = {}  # bn -> (raw, n, cov)

    # First reuse from existing CSV
    for p in paths:
        bn = os.path.basename(p)
        if bn in existing:
            raw, n, cov = existing[bn]
            reused += 1
            results[bn] = (raw, n, cov)
            print(f"[SKIP] {bn}  raw={raw:.2f}  samples={n}  roi_cov={cov:.2f}%  (from CSV)")

    # Jobs to compute
    mask_dir = os.path.abspath(args.mask_dir) if args.mask_dir else None
    jobs = []
    for p in paths:
        bn = os.path.basename(p)
        if bn in results:
            continue
        jobs.append((p, mask_dir, args.sample_frames, args.thr, args.dilate_k, args.dilate_iter, args.shift_x, args.min_roi_pixels))

    # Compute (sequential or parallel)
    if args.workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            bn, raw, n, cov, status = _worker_compute(job)
            if status == "MISS_MASK":
                print(f"[MISS MASK] {bn}  (looked for: {mask_dir}/{os.path.splitext(bn)[0]}_replace_mask.*)")
            elif status.startswith("ERR:"):
                print(f"[ERR]  {bn}  {status}")
            else:
                print(f"[OK]   {bn}  raw={raw:.2f}  samples={n}  roi_cov={cov:.2f}%")
            results[bn] = (raw, n, cov)
            computed += 1
    else:
        # Processes are safer than threads here (PyAV/OpenCV + GIL).
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker_compute, job) for job in jobs]
            for fut in concurrent.futures.as_completed(futs):
                bn, raw, n, cov, status = fut.result()
                if status == "MISS_MASK":
                    print(f"[MISS MASK] {bn}  (looked for: {mask_dir}/{os.path.splitext(bn)[0]}_replace_mask.*)")
                elif status.startswith("ERR:"):
                    print(f"[ERR]  {bn}  {status}")
                else:
                    print(f"[OK]   {bn}  raw={raw:.2f}  samples={n}  roi_cov={cov:.2f}%")
                results[bn] = (raw, n, cov)
                computed += 1

    # Rebuild tmp list in original stable order
    tmp = []
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

    print(f"\nDone: {out_csv}  (reused={reused}, computed={computed}, total={len(tmp)})")


if __name__ == "__main__":
    main()

