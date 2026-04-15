#!/usr/bin/env python3
"""
Analyze replace-mask videos and report the maximum "static hold" duration per file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from dependency.mask_static_hold import (
    DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
    DEFAULT_BORDER_TOLERANCE_PX,
    DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_THRESHOLD_U8,
    analyze_mask_video,
)


DEFAULT_GLOB_PATTERNS = ("*.mkv", "*.mp4", "*.mov", "*.avi", "*.m4v", "*.webm")
DEFAULT_OUT_CSV = "mask_static_hold_report.csv"


def _iter_video_files(
    input_path: str,
    patterns: Iterable[str],
    recursive: bool,
) -> List[str]:
    src = os.path.abspath(input_path)
    if os.path.isfile(src):
        return [src]
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Input path not found: {src}")

    out: List[str] = []
    for patt in patterns:
        if recursive:
            out.extend(glob.glob(os.path.join(src, "**", patt), recursive=True))
        else:
            out.extend(glob.glob(os.path.join(src, patt)))
    return sorted({os.path.abspath(p) for p in out if os.path.isfile(p)})


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze replace-mask clips and report, per file, the maximum duration "
            "for which a significant non-border component still preserves enough "
            "of its own initial anchor position."
        )
    )
    ap.add_argument("input", help="Input video file or folder of mask videos")
    ap.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=[],
        help=(
            "Glob pattern(s) when input is a folder. Can be repeated. "
            f"Default: {', '.join(DEFAULT_GLOB_PATTERNS)}"
        ),
    )
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV path")
    ap.add_argument(
        "--threshold-u8",
        type=int,
        default=DEFAULT_THRESHOLD_U8,
        help=f"Mask threshold on 8-bit grayscale. Default: {DEFAULT_THRESHOLD_U8}",
    )
    ap.add_argument(
        "--min-area-px",
        type=int,
        default=DEFAULT_MIN_AREA_PX,
        help=f"Ignore components smaller than this area. Default: {DEFAULT_MIN_AREA_PX}",
    )
    ap.add_argument(
        "--border-tolerance-px",
        type=int,
        default=DEFAULT_BORDER_TOLERANCE_PX,
        help=f"Ignore components touching left/right border within this tolerance. Default: {DEFAULT_BORDER_TOLERANCE_PX}",
    )
    ap.add_argument(
        "--component-merge-y-tol-px",
        type=int,
        default=DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
        help=(
            "Merge small vertical gaps before connected-components, mirroring "
            f"mask_for_merge. Default: {DEFAULT_COMPONENT_MERGE_Y_TOL_PX}"
        ),
    )
    ap.add_argument(
        "--anchor-overlap-min-ratio",
        type=float,
        default=DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
        help=(
            "Minimum overlap ratio versus the anchor area required to keep the same "
            f"hold interval alive. Default: {DEFAULT_ANCHOR_OVERLAP_MIN_RATIO}"
        ),
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    patterns = tuple(args.globs) if args.globs else DEFAULT_GLOB_PATTERNS
    files = _iter_video_files(args.input, patterns, recursive=bool(args.recursive))
    if not files:
        print("No input files found.")
        return 1

    results: List[Dict[str, object]] = []
    for idx, path in enumerate(files, start=1):
        res = analyze_mask_video(
            path=path,
            threshold_u8=int(args.threshold_u8),
            min_area_px=int(args.min_area_px),
            border_tolerance_px=int(args.border_tolerance_px),
            component_merge_y_tol_px=int(args.component_merge_y_tol_px),
            anchor_overlap_min_ratio=float(args.anchor_overlap_min_ratio),
        )
        results.append(res)
        print(
            f"[{idx}/{len(files)}] {os.path.basename(path)}  "
            f"max_hold={int(res['max_hold_frames'])}f  "
            f"({float(res['max_hold_seconds']):.3f}s)"
        )

    results.sort(key=lambda row: (-int(row["max_hold_frames"]), str(row["file"])))

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "frames",
                "fps",
                "max_hold_frames",
                "max_hold_seconds",
                "max_hold_start_frame",
                "max_hold_end_frame",
                "max_hold_area_px",
            ]
        )
        for row in results:
            w.writerow(
                [
                    row["file"],
                    row["frames"],
                    f"{float(row['fps']):.6f}",
                    row["max_hold_frames"],
                    f"{float(row['max_hold_seconds']):.6f}",
                    row["max_hold_start_frame"],
                    row["max_hold_end_frame"],
                    row["max_hold_area_px"],
                ]
            )

    best = results[0]
    print("")
    print(f"Wrote: {out_csv}")
    print(
        "Top hold: "
        f"{os.path.basename(str(best['file']))}  "
        f"{int(best['max_hold_frames'])}f  "
        f"({float(best['max_hold_seconds']):.3f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
