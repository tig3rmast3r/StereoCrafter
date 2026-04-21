from __future__ import annotations

import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_THRESHOLD_U8 = 13
DEFAULT_MIN_AREA_PX = 2500
DEFAULT_BORDER_TOLERANCE_PX = 2
DEFAULT_COMPONENT_MERGE_Y_TOL_PX = 4
DEFAULT_ANCHOR_OVERLAP_MIN_RATIO = 0.1
DEFAULT_CONTENT_GRAY_DELTA_THRESHOLD = 6.0
DEFAULT_CONTENT_EDGE_DELTA_THRESHOLD = 25.0
DEFAULT_ROI_MASK_DILATE_K = 0
DEFAULT_ROI_MASK_DILATE_ITER = 0
DEFAULT_MIN_CONTENT_ROI_PIXELS = 64

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


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _infer_warped_layout_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("_splatted1"):
        return "single"
    if stem.endswith("_splatted2"):
        return "dual"
    if stem.endswith("_splatted4"):
        return "quad"
    return "single"


def _extract_warped_gray_from_frame(frame: np.ndarray, layout: str) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    else:
        if layout == "single":
            warped = frame
        elif layout == "dual":
            warped = frame[:, frame.shape[1] // 2 :]
        else:
            half_h = frame.shape[0] // 2
            half_w = frame.shape[1] // 2
            warped = frame[half_h:, half_w:]
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8, copy=False)


def _dilate_component_mask(
    comp_mask: np.ndarray,
    *,
    kernel_size: int,
    iterations: int,
) -> np.ndarray:
    if int(kernel_size) <= 0 or int(iterations) <= 0:
        return comp_mask.astype(bool, copy=False)
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(
        comp_mask.astype(np.uint8, copy=False),
        kernel,
        iterations=max(1, int(iterations)),
    )
    return dil.astype(bool, copy=False)


def _build_component_source_roi(
    *,
    bbox: ComponentBBox,
    comp_mask: np.ndarray,
    frame_width: int,
    warped_gray: np.ndarray,
    warped_edge: np.ndarray,
    border_tolerance_px: int,
    roi_mask_dilate_k: int,
    roi_mask_dilate_iter: int,
) -> Optional[Dict[str, object]]:
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        return None
    comp_mask = _dilate_component_mask(
        comp_mask,
        kernel_size=roi_mask_dilate_k,
        iterations=roi_mask_dilate_iter,
    )
    h, w = comp_mask.shape[:2]
    if h <= 0 or w <= 0 or not bool(comp_mask.any()):
        return None

    border_cols = max(1, min(int(border_tolerance_px), max(1, frame_width)))
    right_touch_start = frame_width - border_cols
    roi_rows = np.zeros((h, frame_width), dtype=np.uint8)

    def _is_mask(local_y: int, global_x: int) -> bool:
        if global_x < x1 or global_x > x2:
            return False
        return bool(comp_mask[local_y, global_x - x1])

    def _mark_block_if_valid(local_y: int, src_a: int, src_b: int) -> bool:
        if src_a < 0 or src_b > frame_width or src_a >= src_b:
            return False
        for gx in range(src_a, src_b):
            if _is_mask(local_y, gx):
                return False
        roi_rows[local_y, src_a:src_b] = 1
        return True

    def _nearest_nonmask_x(local_y: int, start_x: int, step: int) -> int:
        gx = int(start_x)
        while 0 <= gx < frame_width:
            if not _is_mask(local_y, gx):
                return gx
            gx += step
        return -1

    for local_y in range(h):
        xs = np.flatnonzero(comp_mask[local_y])
        if xs.size == 0:
            continue
        run_start = int(xs[0])
        prev = int(xs[0])

        def _mark_run(local_a: int, local_b: int) -> None:
            global_a = x1 + int(local_a)
            global_b = x1 + int(local_b)
            run_len = int(global_b - global_a + 1)
            touches_left = global_a < border_cols
            touches_right = global_b >= right_touch_start
            prefer_left = touches_right and not touches_left

            marked = False
            if prefer_left:
                marked = _mark_block_if_valid(local_y, global_a - run_len, global_a)
                if not marked:
                    marked = _mark_block_if_valid(local_y, global_b + 1, global_b + 1 + run_len)
            else:
                marked = _mark_block_if_valid(local_y, global_b + 1, global_b + 1 + run_len)
                if not marked:
                    marked = _mark_block_if_valid(local_y, global_a - run_len, global_a)
            if marked:
                return

            if prefer_left:
                src_x = _nearest_nonmask_x(local_y, global_a - 1, -1)
                if src_x < 0:
                    src_x = _nearest_nonmask_x(local_y, global_b + 1, 1)
            else:
                src_x = _nearest_nonmask_x(local_y, global_b + 1, 1)
                if src_x < 0:
                    src_x = _nearest_nonmask_x(local_y, global_a - 1, -1)
            if src_x >= 0:
                roi_rows[local_y, src_x] = 1

        for cur in xs[1:]:
            cur = int(cur)
            if cur != prev + 1:
                _mark_run(run_start, prev)
                run_start = cur
            prev = cur
        _mark_run(run_start, prev)

    ys, xs = np.where(roi_rows > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    roi_y1 = y1 + int(ys.min())
    roi_y2 = y1 + int(ys.max())
    roi_x1 = int(xs.min())
    roi_x2 = int(xs.max())
    roi_mask = roi_rows[int(ys.min()) : int(ys.max()) + 1, roi_x1 : roi_x2 + 1].astype(bool, copy=False)
    if not bool(roi_mask.any()):
        return None

    gray_crop = warped_gray[roi_y1 : roi_y2 + 1, roi_x1 : roi_x2 + 1].astype(np.float32, copy=False)
    edge_crop = warped_edge[roi_y1 : roi_y2 + 1, roi_x1 : roi_x2 + 1].astype(np.float32, copy=False)
    return {
        "bbox": (roi_x1, roi_y1, roi_x2, roi_y2),
        "mask": roi_mask,
        "gray": gray_crop,
        "edge": edge_crop,
        "pixels": int(np.count_nonzero(roi_mask)),
    }


def _compare_component_content(
    anchor_roi: Optional[Dict[str, object]],
    curr_roi: Optional[Dict[str, object]],
    *,
    min_roi_pixels: int,
    gray_delta_threshold: float,
    edge_delta_threshold: float,
) -> Dict[str, object]:
    out = {
        "ok": False,
        "roi_pixels": 0,
        "roi_overlap_ratio": 0.0,
        "gray_delta": float("inf"),
        "edge_delta": float("inf"),
    }
    if not anchor_roi or not curr_roi:
        return out

    anchor_bbox = anchor_roi["bbox"]
    curr_bbox = curr_roi["bbox"]
    assert isinstance(anchor_bbox, tuple)
    assert isinstance(curr_bbox, tuple)
    ix1 = max(anchor_bbox[0], curr_bbox[0])
    iy1 = max(anchor_bbox[1], curr_bbox[1])
    ix2 = min(anchor_bbox[2], curr_bbox[2])
    iy2 = min(anchor_bbox[3], curr_bbox[3])
    if ix2 < ix1 or iy2 < iy1:
        return out

    anchor_mask = anchor_roi["mask"]
    curr_mask = curr_roi["mask"]
    anchor_gray = anchor_roi["gray"]
    curr_gray = curr_roi["gray"]
    anchor_edge = anchor_roi["edge"]
    curr_edge = curr_roi["edge"]
    assert isinstance(anchor_mask, np.ndarray)
    assert isinstance(curr_mask, np.ndarray)
    assert isinstance(anchor_gray, np.ndarray)
    assert isinstance(curr_gray, np.ndarray)
    assert isinstance(anchor_edge, np.ndarray)
    assert isinstance(curr_edge, np.ndarray)

    a_mask = anchor_mask[
        iy1 - anchor_bbox[1] : iy2 - anchor_bbox[1] + 1,
        ix1 - anchor_bbox[0] : ix2 - anchor_bbox[0] + 1,
    ]
    c_mask = curr_mask[
        iy1 - curr_bbox[1] : iy2 - curr_bbox[1] + 1,
        ix1 - curr_bbox[0] : ix2 - curr_bbox[0] + 1,
    ]
    valid = np.logical_and(a_mask, c_mask)
    roi_pixels = int(np.count_nonzero(valid))
    anchor_pixels = max(1, int(anchor_roi.get("pixels") or 0))
    curr_pixels = max(1, int(curr_roi.get("pixels") or 0))
    overlap_ratio = float(roi_pixels) / float(max(1, min(anchor_pixels, curr_pixels)))
    out["roi_pixels"] = int(roi_pixels)
    out["roi_overlap_ratio"] = float(overlap_ratio)
    if roi_pixels <= 0:
        return out

    a_gray = anchor_gray[
        iy1 - anchor_bbox[1] : iy2 - anchor_bbox[1] + 1,
        ix1 - anchor_bbox[0] : ix2 - anchor_bbox[0] + 1,
    ]
    c_gray = curr_gray[
        iy1 - curr_bbox[1] : iy2 - curr_bbox[1] + 1,
        ix1 - curr_bbox[0] : ix2 - curr_bbox[0] + 1,
    ]
    a_edge = anchor_edge[
        iy1 - anchor_bbox[1] : iy2 - anchor_bbox[1] + 1,
        ix1 - anchor_bbox[0] : ix2 - anchor_bbox[0] + 1,
    ]
    c_edge = curr_edge[
        iy1 - curr_bbox[1] : iy2 - curr_bbox[1] + 1,
        ix1 - curr_bbox[0] : ix2 - curr_bbox[0] + 1,
    ]
    gray_delta = float(np.mean(np.abs(a_gray[valid] - c_gray[valid])))
    edge_delta = float(np.mean(np.abs(a_edge[valid] - c_edge[valid])))
    gray_ok = gray_delta_threshold <= 0.0 or gray_delta <= gray_delta_threshold
    edge_ok = edge_delta_threshold <= 0.0 or edge_delta <= edge_delta_threshold
    out["gray_delta"] = float(gray_delta)
    out["edge_delta"] = float(edge_delta)
    out["ok"] = bool(roi_pixels >= int(min_roi_pixels) and gray_ok and edge_ok)
    return out


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


def analyze_mask_video_with_warped_content(
    *,
    mask_path: str,
    warped_path: str,
    threshold_u8: int = DEFAULT_THRESHOLD_U8,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    border_tolerance_px: int = DEFAULT_BORDER_TOLERANCE_PX,
    component_merge_y_tol_px: int = DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
    anchor_overlap_min_ratio: float = DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
    content_gray_delta_threshold: float = DEFAULT_CONTENT_GRAY_DELTA_THRESHOLD,
    content_edge_delta_threshold: float = DEFAULT_CONTENT_EDGE_DELTA_THRESHOLD,
    roi_mask_dilate_k: int = DEFAULT_ROI_MASK_DILATE_K,
    roi_mask_dilate_iter: int = DEFAULT_ROI_MASK_DILATE_ITER,
    min_content_roi_pixels: int = DEFAULT_MIN_CONTENT_ROI_PIXELS,
) -> Dict[str, object]:
    cap_mask = cv2.VideoCapture(mask_path)
    if not cap_mask.isOpened():
        raise RuntimeError(f"Could not open mask video: {mask_path}")
    cap_warped = cv2.VideoCapture(warped_path)
    if not cap_warped.isOpened():
        cap_mask.release()
        raise RuntimeError(f"Could not open warped video: {warped_path}")

    fps = float(cap_mask.get(cv2.CAP_PROP_FPS) or cap_warped.get(cv2.CAP_PROP_FPS) or 0.0)
    warped_layout = _infer_warped_layout_from_path(warped_path)
    anchor_overlap_min_ratio = float(max(0.0, min(1.0, anchor_overlap_min_ratio)))
    prev_components: List[Dict[str, object]] = []
    frame_idx_1b = 0

    max_hold_frames = 0
    max_hold_start_frame = 0
    max_hold_end_frame = 0
    max_hold_area_px = 0
    max_hold_roi_pixels = 0
    max_hold_gray_delta_last = 0.0
    max_hold_edge_delta_last = 0.0
    max_hold_gray_delta_mean = 0.0
    max_hold_edge_delta_mean = 0.0
    max_hold_gray_delta_max = 0.0
    max_hold_edge_delta_max = 0.0
    max_hold_roi_overlap_ratio = 0.0

    try:
        while True:
            ok_mask, mask_frame = cap_mask.read()
            ok_warped, warped_frame = cap_warped.read()
            if not ok_mask or not ok_warped:
                break
            frame_idx_1b += 1

            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if mask_frame.ndim == 3 else mask_frame
            frame_bin = (mask_gray > int(threshold_u8)).astype(np.uint8)
            warped_gray = _extract_warped_gray_from_frame(warped_frame, warped_layout)
            if warped_gray.shape != frame_bin.shape:
                if (
                    warped_gray.shape[0] == frame_bin.shape[0]
                    and warped_gray.shape[1] > frame_bin.shape[1]
                ):
                    warped_gray = warped_gray[:, -frame_bin.shape[1] :]
                if warped_gray.shape != frame_bin.shape:
                    prev_components = []
                    continue

            warped_edge = _edge_magnitude(warped_gray)
            curr_raw = _extract_components(
                frame_bin=frame_bin,
                min_area_px=int(min_area_px),
                border_tolerance_px=int(border_tolerance_px),
                component_merge_y_tol_px=int(component_merge_y_tol_px),
            )
            assigned_prev_idx = _match_components(prev_components, curr_raw)

            curr_components: List[Dict[str, object]] = []
            for ci, curr in enumerate(curr_raw):
                curr_bbox = curr["bbox"]
                curr_mask = curr["mask"]
                area_px = int(curr["area"])
                assert isinstance(curr_bbox, tuple)
                assert isinstance(curr_mask, np.ndarray)
                curr_roi = _build_component_source_roi(
                    bbox=curr_bbox,
                    comp_mask=curr_mask,
                    frame_width=frame_bin.shape[1],
                    warped_gray=warped_gray,
                    warped_edge=warped_edge,
                    border_tolerance_px=int(border_tolerance_px),
                    roi_mask_dilate_k=int(roi_mask_dilate_k),
                    roi_mask_dilate_iter=int(roi_mask_dilate_iter),
                )

                prev_idx = assigned_prev_idx[ci]
                if prev_idx is not None:
                    prev = prev_components[int(prev_idx)]
                    anchor_bbox = prev["anchor_bbox"]
                    anchor_mask = prev["anchor_mask"]
                    anchor_roi = prev["anchor_roi"]
                    assert isinstance(anchor_bbox, tuple)
                    assert isinstance(anchor_mask, np.ndarray)
                    overlap_px = _mask_overlap_area(
                        curr_bbox,
                        curr_mask,
                        anchor_bbox,
                        anchor_mask,
                    )
                    anchor_area_px = max(1, int(np.count_nonzero(anchor_mask)))
                    overlap_ratio = float(overlap_px) / float(anchor_area_px)
                    content_cmp = _compare_component_content(
                        anchor_roi,
                        curr_roi,
                        min_roi_pixels=int(min_content_roi_pixels),
                        gray_delta_threshold=float(content_gray_delta_threshold),
                        edge_delta_threshold=float(content_edge_delta_threshold),
                    )
                    if overlap_ratio >= anchor_overlap_min_ratio and bool(content_cmp["ok"]):
                        hold_start_frame = int(prev["hold_start_frame"])
                        hold_frames = int(prev["hold_frames"]) + 1
                        gray_delta_sum = float(prev["gray_delta_sum"]) + float(content_cmp["gray_delta"])
                        edge_delta_sum = float(prev["edge_delta_sum"]) + float(content_cmp["edge_delta"])
                        delta_count = int(prev["delta_count"]) + 1
                        gray_delta_max = max(float(prev["gray_delta_max"]), float(content_cmp["gray_delta"]))
                        edge_delta_max = max(float(prev["edge_delta_max"]), float(content_cmp["edge_delta"]))
                    else:
                        anchor_bbox = curr_bbox
                        anchor_mask = curr_mask
                        anchor_roi = curr_roi
                        hold_start_frame = frame_idx_1b
                        hold_frames = 1
                        content_cmp = {
                            "gray_delta": 0.0,
                            "edge_delta": 0.0,
                            "roi_overlap_ratio": 0.0,
                        }
                        gray_delta_sum = 0.0
                        edge_delta_sum = 0.0
                        delta_count = 0
                        gray_delta_max = 0.0
                        edge_delta_max = 0.0
                else:
                    anchor_bbox = curr_bbox
                    anchor_mask = curr_mask
                    anchor_roi = curr_roi
                    hold_start_frame = frame_idx_1b
                    hold_frames = 1
                    content_cmp = {
                        "gray_delta": 0.0,
                        "edge_delta": 0.0,
                        "roi_overlap_ratio": 0.0,
                    }
                    gray_delta_sum = 0.0
                    edge_delta_sum = 0.0
                    delta_count = 0
                    gray_delta_max = 0.0
                    edge_delta_max = 0.0

                if hold_frames > max_hold_frames:
                    max_hold_frames = int(hold_frames)
                    max_hold_start_frame = int(hold_start_frame)
                    max_hold_end_frame = int(frame_idx_1b)
                    max_hold_area_px = int(area_px)
                    max_hold_roi_pixels = int((curr_roi or {}).get("pixels") or 0)
                    max_hold_gray_delta_last = float(content_cmp["gray_delta"])
                    max_hold_edge_delta_last = float(content_cmp["edge_delta"])
                    max_hold_gray_delta_mean = (
                        float(gray_delta_sum) / float(delta_count) if delta_count > 0 else 0.0
                    )
                    max_hold_edge_delta_mean = (
                        float(edge_delta_sum) / float(delta_count) if delta_count > 0 else 0.0
                    )
                    max_hold_gray_delta_max = float(gray_delta_max)
                    max_hold_edge_delta_max = float(edge_delta_max)
                    max_hold_roi_overlap_ratio = float(content_cmp["roi_overlap_ratio"])

                curr_components.append(
                    {
                        "bbox": curr_bbox,
                        "centroid": curr["centroid"],
                        "area": area_px,
                        "mask": curr_mask,
                        "anchor_bbox": anchor_bbox,
                        "anchor_mask": anchor_mask,
                        "anchor_roi": anchor_roi,
                        "hold_start_frame": int(hold_start_frame),
                        "hold_frames": int(hold_frames),
                        "gray_delta_sum": float(gray_delta_sum),
                        "edge_delta_sum": float(edge_delta_sum),
                        "delta_count": int(delta_count),
                        "gray_delta_max": float(gray_delta_max),
                        "edge_delta_max": float(edge_delta_max),
                    }
                )

            prev_components = curr_components
    finally:
        cap_mask.release()
        cap_warped.release()

    max_hold_seconds = float(max_hold_frames / fps) if fps > 0.0 else 0.0
    return {
        "file": os.path.abspath(mask_path),
        "warped_file": os.path.abspath(warped_path),
        "warped_layout": str(warped_layout),
        "frames": int(frame_idx_1b),
        "fps": float(fps),
        "max_hold_frames": int(max_hold_frames),
        "max_hold_seconds": float(max_hold_seconds),
        "max_hold_start_frame": int(max_hold_start_frame),
        "max_hold_end_frame": int(max_hold_end_frame),
        "max_hold_area_px": int(max_hold_area_px),
        "max_hold_roi_pixels": int(max_hold_roi_pixels),
        "max_hold_gray_delta_last": float(max_hold_gray_delta_last),
        "max_hold_edge_delta_last": float(max_hold_edge_delta_last),
        "max_hold_gray_delta_mean": float(max_hold_gray_delta_mean),
        "max_hold_edge_delta_mean": float(max_hold_edge_delta_mean),
        "max_hold_gray_delta_max": float(max_hold_gray_delta_max),
        "max_hold_edge_delta_max": float(max_hold_edge_delta_max),
        "max_hold_roi_overlap_ratio": float(max_hold_roi_overlap_ratio),
    }
