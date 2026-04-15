from __future__ import annotations

import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_THRESHOLD_U8 = 13
DEFAULT_MIN_AREA_PX = 1000
DEFAULT_BORDER_TOLERANCE_PX = 2
DEFAULT_COMPONENT_MERGE_Y_TOL_PX = 4
DEFAULT_ANCHOR_OVERLAP_MIN_RATIO = 0.1

ComponentBBox = Tuple[int, int, int, int]
FrameBinProvider = Callable[[int], Optional[np.ndarray]]


def _bbox_intersection_area(a: ComponentBBox, b: ComponentBBox) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0
    return int((ix2 - ix1 + 1) * (iy2 - iy1 + 1))


def _mask_overlap_area(
    bbox_a: ComponentBBox,
    mask_a: np.ndarray,
    bbox_b: ComponentBBox,
    mask_b: np.ndarray,
) -> int:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0

    a_crop = mask_a[iy1 - ay1 : iy2 - ay1 + 1, ix1 - ax1 : ix2 - ax1 + 1]
    b_crop = mask_b[iy1 - by1 : iy2 - by1 + 1, ix1 - bx1 : ix2 - bx1 + 1]
    if a_crop.size == 0 or b_crop.size == 0:
        return 0
    return int(np.count_nonzero(np.logical_and(a_crop, b_crop)))


def _extract_components(
    frame_bin: np.ndarray,
    min_area_px: int,
    border_tolerance_px: int,
    component_merge_y_tol_px: int,
) -> List[Dict[str, object]]:
    height, width = frame_bin.shape
    border_tol = max(1, min(int(border_tolerance_px), max(1, width)))
    right_touch_start = width - border_tol

    frame_cc = frame_bin
    if component_merge_y_tol_px > 0:
        k_h = max(1, int(2 * component_merge_y_tol_px + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        frame_cc = cv2.morphologyEx(frame_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        frame_cc, connectivity=8
    )

    components: List[Dict[str, object]] = []
    for lab in range(1, int(n_labels)):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_area_px) or w <= 0 or h <= 0:
            continue

        x2 = x + w - 1
        y2 = y + h - 1
        if x < border_tol or x2 >= right_touch_start:
            continue

        comp_mask = labels[y : y + h, x : x + w] == lab
        if not bool(comp_mask.any()):
            continue

        components.append(
            {
                "label": int(lab),
                "bbox": (x, y, x2, y2),
                "centroid": (float(centroids[lab][0]), float(centroids[lab][1])),
                "area": int(area),
                "mask": comp_mask,
            }
        )

    return components


def _match_components(
    prev_components: List[Dict[str, object]],
    curr_components: List[Dict[str, object]],
) -> List[Optional[int]]:
    assigned_prev_idx: List[Optional[int]] = [None] * len(curr_components)
    if not prev_components or not curr_components:
        return assigned_prev_idx

    for ci, curr in enumerate(curr_components):
        best_pi = -1
        best_inter = 0
        curr_bbox = curr["bbox"]
        assert isinstance(curr_bbox, tuple)
        for pi, prev in enumerate(prev_components):
            prev_bbox = prev["bbox"]
            assert isinstance(prev_bbox, tuple)
            inter = _bbox_intersection_area(curr_bbox, prev_bbox)
            if inter > best_inter:
                best_inter = inter
                best_pi = pi
        if best_pi >= 0 and best_inter > 0:
            assigned_prev_idx[ci] = int(best_pi)

    for ci, curr in enumerate(curr_components):
        if assigned_prev_idx[ci] is not None:
            continue
        best_pi = -1
        best_d = 1e9
        curr_area = float(curr["area"])
        max_dist = max(8.0, 0.5 * math.sqrt(curr_area))
        curr_centroid = curr["centroid"]
        assert isinstance(curr_centroid, tuple)
        for pi, prev in enumerate(prev_components):
            prev_centroid = prev["centroid"]
            assert isinstance(prev_centroid, tuple)
            dx = float(curr_centroid[0] - prev_centroid[0])
            dy = float(curr_centroid[1] - prev_centroid[1])
            d = float(math.hypot(dx, dy))
            if d < best_d and d <= max_dist:
                best_d = d
                best_pi = pi
        if best_pi >= 0:
            assigned_prev_idx[ci] = int(best_pi)

    return assigned_prev_idx


def analyze_mask_frame_bins(
    frame_provider: FrameBinProvider,
    *,
    fps: float = 0.0,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    border_tolerance_px: int = DEFAULT_BORDER_TOLERANCE_PX,
    component_merge_y_tol_px: int = DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
    anchor_overlap_min_ratio: float = DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
    file_label: str = "",
) -> Dict[str, object]:
    anchor_overlap_min_ratio = float(max(0.0, min(1.0, anchor_overlap_min_ratio)))
    prev_components: List[Dict[str, object]] = []
    frame_idx_1b = 0

    max_hold_frames = 0
    max_hold_start_frame = 0
    max_hold_end_frame = 0
    max_hold_area_px = 0

    while True:
        frame_bin = frame_provider(frame_idx_1b)
        if frame_bin is None:
            break
        frame_idx_1b += 1

        curr_raw = _extract_components(
            frame_bin=frame_bin.astype(np.uint8, copy=False),
            min_area_px=int(min_area_px),
            border_tolerance_px=int(border_tolerance_px),
            component_merge_y_tol_px=int(component_merge_y_tol_px),
        )
        assigned_prev_idx = _match_components(prev_components, curr_raw)

        curr_components: List[Dict[str, object]] = []
        for ci, curr in enumerate(curr_raw):
            prev_idx = assigned_prev_idx[ci]
            if prev_idx is not None:
                prev = prev_components[int(prev_idx)]
                anchor_bbox = prev["anchor_bbox"]
                anchor_mask = prev["anchor_mask"]
                curr_bbox = curr["bbox"]
                curr_mask = curr["mask"]
                assert isinstance(anchor_bbox, tuple)
                assert isinstance(anchor_mask, np.ndarray)
                assert isinstance(curr_bbox, tuple)
                assert isinstance(curr_mask, np.ndarray)
                overlap_px = _mask_overlap_area(
                    curr_bbox,
                    curr_mask,
                    anchor_bbox,
                    anchor_mask,
                )
                anchor_area_px = max(1, int(np.count_nonzero(anchor_mask)))
                overlap_ratio = float(overlap_px) / float(anchor_area_px)
                if overlap_ratio >= anchor_overlap_min_ratio:
                    hold_start_frame = int(prev["hold_start_frame"])
                    hold_frames = int(prev["hold_frames"]) + 1
                else:
                    anchor_bbox = curr_bbox
                    anchor_mask = curr_mask
                    hold_start_frame = frame_idx_1b
                    hold_frames = 1
            else:
                anchor_bbox = curr["bbox"]
                anchor_mask = curr["mask"]
                hold_start_frame = frame_idx_1b
                hold_frames = 1

            hold_end_frame = frame_idx_1b
            area_px = int(curr["area"])
            if hold_frames > max_hold_frames:
                max_hold_frames = int(hold_frames)
                max_hold_start_frame = int(hold_start_frame)
                max_hold_end_frame = int(hold_end_frame)
                max_hold_area_px = int(area_px)

            curr_components.append(
                {
                    "bbox": curr["bbox"],
                    "centroid": curr["centroid"],
                    "area": area_px,
                    "mask": curr["mask"],
                    "anchor_bbox": anchor_bbox,
                    "anchor_mask": anchor_mask,
                    "hold_start_frame": int(hold_start_frame),
                    "hold_frames": int(hold_frames),
                }
            )

        prev_components = curr_components

    max_hold_seconds = float(max_hold_frames / fps) if fps > 0.0 else 0.0
    return {
        "file": os.path.abspath(file_label) if file_label else "",
        "frames": int(frame_idx_1b),
        "fps": float(fps),
        "max_hold_frames": int(max_hold_frames),
        "max_hold_seconds": float(max_hold_seconds),
        "max_hold_start_frame": int(max_hold_start_frame),
        "max_hold_end_frame": int(max_hold_end_frame),
        "max_hold_area_px": int(max_hold_area_px),
    }


def analyze_mask_video(
    path: str,
    threshold_u8: int = DEFAULT_THRESHOLD_U8,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    border_tolerance_px: int = DEFAULT_BORDER_TOLERANCE_PX,
    component_merge_y_tol_px: int = DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
    anchor_overlap_min_ratio: float = DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
) -> Dict[str, object]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def _provider(_idx: int) -> Optional[np.ndarray]:
        ok, frame = cap.read()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return (gray > int(threshold_u8)).astype(np.uint8)

    try:
        return analyze_mask_frame_bins(
            _provider,
            fps=fps,
            min_area_px=min_area_px,
            border_tolerance_px=border_tolerance_px,
            component_merge_y_tol_px=component_merge_y_tol_px,
            anchor_overlap_min_ratio=anchor_overlap_min_ratio,
            file_label=path,
        )
    finally:
        cap.release()


def analyze_embedded_mask_video(
    path: str,
    *,
    input_layout: str,
    threshold_u8: int = DEFAULT_THRESHOLD_U8,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    border_tolerance_px: int = DEFAULT_BORDER_TOLERANCE_PX,
    component_merge_y_tol_px: int = DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
    anchor_overlap_min_ratio: float = DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
    process_length: int = -1,
) -> Dict[str, object]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames_seen = 0
    layout = str(input_layout or "").strip().lower()
    if layout not in {"dual", "quad"}:
        raise ValueError(f"Unsupported embedded mask layout: {input_layout}")

    def _provider(_idx: int) -> Optional[np.ndarray]:
        nonlocal frames_seen
        if process_length > 0 and frames_seen >= process_length:
            return None
        ok, frame = cap.read()
        if not ok:
            return None
        frames_seen += 1
        if frame.ndim != 3:
            return None
        height, width = frame.shape[:2]
        if layout == "dual":
            mask_view = frame[:, : width // 2]
        else:
            half_h = height // 2
            half_w = width // 2
            mask_view = frame[half_h:, :half_w]
        gray = cv2.cvtColor(mask_view, cv2.COLOR_BGR2GRAY)
        return (gray > int(threshold_u8)).astype(np.uint8)

    try:
        return analyze_mask_frame_bins(
            _provider,
            fps=fps,
            min_area_px=min_area_px,
            border_tolerance_px=border_tolerance_px,
            component_merge_y_tol_px=component_merge_y_tol_px,
            anchor_overlap_min_ratio=anchor_overlap_min_ratio,
            file_label=path,
        )
    finally:
        cap.release()
