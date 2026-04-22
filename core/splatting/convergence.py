"""Auto-convergence estimation module using U2NETP neural network.

Provides convergence plane estimation using visual saliency analysis
of RGB and depth map pairs.
"""

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from dependency.stereocrafter_util import get_video_stream_info
from .depth_processing import (
    DEPTH_VIS_TV10_BLACK_NORM,
    DEPTH_VIS_TV10_WHITE_NORM,
    _infer_depth_bit_depth,
)

logger = logging.getLogger(__name__)


class ConvergenceEstimatorWrapper:
    """Handles automatic convergence plane detection using U2NETP.

    Estimates the optimal zero-parallax plane based on visual saliency
    analysis of RGB and depth map pairs. Wraps the dependency.convergence_estimator
    neural network model.

    Args:
        model_path: Optional path to U2NETP model weights
        device: Optional torch device ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_model: bool = True,
    ):
        """Initialize convergence estimation model.

        Args:
            model_path: Path to model weights file
            device: Torch device for inference
            load_model: If False, defer neural model loading until needed
        """
        self.logger = logging.getLogger(__name__)
        self._model_path = model_path
        self._device = device
        self._estimator = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        """Load the U2NETP model for convergence estimation."""
        try:
            # Lazy import to avoid circular dependency issues
            from dependency.convergence_estimator import (
                ConvergenceEstimator as NeuralEstimator,
            )

            self._estimator = NeuralEstimator(
                model_path=self._model_path, device=self._device
            )
            if self._estimator.model is None:
                self.logger.error(
                    "ConvergenceEstimator model failed to load."
                )
        except ImportError as e:
            self.logger.error(f"Could not import ConvergenceEstimator: {e}")
            self._estimator = None

    def is_model_loaded(self) -> bool:
        """Check if the neural network model is loaded and ready.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._estimator is not None and self._estimator.model is not None

    def ensure_model_loaded(self) -> bool:
        """Load the neural model on demand when AI convergence modes are used."""
        if self.is_model_loaded():
            return True
        self._load_model()
        return self.is_model_loaded()

    @staticmethod
    def _to_gray_depth(frame_raw: np.ndarray) -> np.ndarray:
        """Convert a decoded depth frame to a single-channel array."""
        if frame_raw.ndim == 2:
            return frame_raw
        if frame_raw.ndim == 3:
            return frame_raw.mean(axis=2)
        raise ValueError(f"Unsupported depth frame shape: {frame_raw.shape}")

    @staticmethod
    def _normalize_depth_to_0_1(depth: np.ndarray) -> np.ndarray:
        """Normalize depth arrays from common code ranges to 0..1."""
        d = depth.astype(np.float32, copy=False)
        if d.size == 0:
            return d
        maxv = float(np.max(d))
        if maxv > 1.5:
            if maxv <= 256.0:
                d = d / 255.0
            elif maxv <= 1024.0:
                d = d / 1023.0
            elif maxv <= 4096.0:
                d = d / 4095.0
            elif maxv <= 65536.0:
                d = d / 65535.0
            else:
                d = d / maxv
        return np.clip(d, 0.0, 1.0)

    @staticmethod
    def _build_convergence_grid(conv_min: float, conv_max: float, conv_step: float) -> np.ndarray:
        """Build a clipped, stable convergence search grid in [0,1]."""
        mn = max(0.0, min(1.0, float(conv_min)))
        mx = max(0.0, min(1.0, float(conv_max)))
        if mx < mn:
            mn, mx = mx, mn
        st = max(1e-6, float(conv_step))
        vals = np.arange(mn, mx + 0.5 * st, st, dtype=np.float32)
        vals = np.clip(vals, 0.0, 1.0)
        vals = np.unique(np.round(vals, 6))
        return vals

    @staticmethod
    def _edge_runs_from_dest(dest_x: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute left/right uncovered run lengths from destination x map."""
        x0 = np.floor(dest_x).astype(np.int32)
        x1 = x0 + 1

        valid0 = (x0 >= 0) & (x0 < width)
        valid1 = (x1 >= 0) & (x1 < width)

        first0 = np.min(np.where(valid0, x0, width), axis=1)
        first1 = np.min(np.where(valid1, x1, width), axis=1)
        first = np.minimum(first0, first1)

        last0 = np.max(np.where(valid0, x0, -1), axis=1)
        last1 = np.max(np.where(valid1, x1, -1), axis=1)
        last = np.maximum(last0, last1)

        has_occ = (first < width) & (last >= 0)
        left_run = np.where(has_occ, first, width).astype(np.int32)
        right_run = np.where(has_occ, (width - 1 - last), width).astype(np.int32)
        return left_run, right_run

    @staticmethod
    def _get_tv_compensation(depth_path: str, mode: str = "auto") -> float:
        """Return disparity compensation for 10-bit TV-range depth maps."""
        mode_l = str(mode).strip().lower()
        if mode_l == "off":
            return 1.0
        tv_factor = 1.0 / (DEPTH_VIS_TV10_WHITE_NORM - DEPTH_VIS_TV10_BLACK_NORM)
        if mode_l == "on":
            return tv_factor
        if mode_l != "auto":
            return 1.0
        try:
            info = get_video_stream_info(depth_path)
            if _infer_depth_bit_depth(info) > 8:
                color_range = str((info or {}).get("color_range", "unknown")).lower()
                if color_range == "tv":
                    return tv_factor
        except Exception:
            pass
        return 1.0

    def estimate_convergence(
        self,
        rgb_path: str,
        depth_path: str,
        process_length: int = -1,
        sample_stride: int = 6,
        gamma: float = 1.0,
        fallback_value: float = 0.5,
        stop_event: Optional[threading.Event] = None,
        scan_borders: bool = False,
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Estimate optimal convergence plane and optionally scan borders.

        Samples frames uniformly from the video, analyzes them using the
        U2NETP model to detect salient objects, and returns both average
        and peak convergence values. If scan_borders is True, also returns
        the maximum depth values found at the left and right 5px edges.

        Args:
            rgb_path: Path to RGB source video
            depth_path: Path to depth map video
            process_length: Number of frames to process (-1 for all)
            sample_stride: Stride between sampled frames (default: 6)
            gamma: Gamma correction for depth (default: 1.0)
            fallback_value: Value to return on failure (default: 0.5)
            stop_event: Optional threading.Event for cancellation
            scan_borders: Whether to also scan left/right edges for max depth

        Returns:
            Tuple of (average_convergence, peak_convergence, max_edge_l, max_edge_r)
        """
        if not self.is_model_loaded():
            if not self.ensure_model_loaded():
                self.logger.warning("Model not loaded, returning fallback values")
                return fallback_value, fallback_value, None, None

        try:
            # Initialize Readers
            vr_rgb = VideoReader(rgb_path, ctx=cpu(0))
            vr_depth = VideoReader(depth_path, ctx=cpu(0))

            len_rgb = len(vr_rgb)
            len_depth = len(vr_depth)

            # Sanity check
            if len_rgb == 0 or len_depth == 0:
                self.logger.warning("Empty video or depth map found.")
                return fallback_value, fallback_value, None, None

            total_frames = min(len_rgb, len_depth)

            # Respect process_length if set > 0
            if process_length > 0:
                total_frames = min(total_frames, process_length)

            # Sample frames
            indices = list(range(0, total_frames, sample_stride))

            # Ensure at least one frame is sampled
            if not indices:
                indices = [0]

            estimates = []
            max_edge_l = 0.0 if scan_borders else None
            max_edge_r = 0.0 if scan_borders else None

            self.logger.info(
                f"Auto-Converge{' + Border Scan' if scan_borders else ''}: Sampling {len(indices)} frames from {os.path.basename(rgb_path)}..."
            )

            for idx in indices:
                if stop_event and stop_event.is_set():
                    self.logger.info("Auto-Converge scan cancelled.")
                    break

                # Read RGB
                rgb_frame = vr_rgb[idx].asnumpy()  # H, W, 3 (uint8)
                # Read Depth
                depth_frame = vr_depth[idx].asnumpy()  # H, W, C or H, W

                # Depth: Handle various formats (Gray8, Gray16, RGB-encoding)
                if depth_frame.ndim == 3:
                    depth_mono = depth_frame.mean(axis=2)
                else:
                    depth_mono = depth_frame

                # --- Optional Border Scan (Lightweight) ---
                if scan_borders:
                    # Sample 5px wide at each edge
                    # We use numpy here as it's already a numpy array from decord
                    L_sample = depth_mono[:, :5]
                    R_sample = depth_mono[:, -5:]

                    # 99th percentile to ignore noise
                    d_L = float(np.percentile(L_sample, 99))
                    d_R = float(np.percentile(R_sample, 99))

                    # Normalize if uint8
                    if depth_mono.dtype == np.uint8 or d_L > 1.0 or d_R > 1.0:
                        d_L /= 255.0
                        d_R /= 255.0

                    max_edge_l = max(max_edge_l, d_L)
                    max_edge_r = max(max_edge_r, d_R)

                # Preprocess for Torch (NN inference)
                rgb_t = (
                    torch.from_numpy(rgb_frame).float().permute(2, 0, 1) / 255.0
                )

                depth_t = torch.from_numpy(depth_mono).float()
                # Normalize if not 0-1
                if depth_t.max() > 1.0:
                    depth_t = depth_t / 255.0

                # Clamp and apply gamma
                depth_t = torch.clamp(depth_t, 0.0, 1.0)
                gamma_f = float(gamma) if gamma else 1.0
                if gamma_f != 1.0:
                    depth_t = 1.0 - torch.pow((1.0 - depth_t), gamma_f)

                # Format: 1, C, H, W
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                rgb_b = rgb_t.unsqueeze(0)

                # Predict
                res = self._estimator.predict(rgb_b, depth_t)
                estimates.extend(res)

            if not estimates:
                return fallback_value, fallback_value, max_edge_l, max_edge_r

            avg_val = sum(estimates) / len(estimates)
            # Using Max as 'Peak' estimate
            peak_val = max(estimates)

            self.logger.info(
                f"Auto-Converge Result: Avg={avg_val:.3f}, Peak={peak_val:.3f}"
            )
            if scan_borders:
                self.logger.info(f"Edge Depth Result: L={max_edge_l:.3f}, R={max_edge_r:.3f}")

            return avg_val, peak_val, max_edge_l, max_edge_r

        except Exception as e:
            self.logger.error(
                f"Auto convergence determination failed: {e}", exc_info=True
            )
            return fallback_value, fallback_value, None, None

    def estimate_min_borders_convergence(
        self,
        depth_path: str,
        process_length: int = -1,
        sample_stride: int = 6,
        max_disp: float = 20.0,
        gamma: float = 1.0,
        fallback_value: float = 0.5,
        stop_event: Optional[threading.Event] = None,
        conv_min: float = 0.2,
        conv_max: float = 0.8,
        conv_step: float = 0.02,
        tv_comp_mode: str = "auto",
    ) -> float:
        """Estimate convergence by minimizing mean border void area.

        Uses only the depth video and the same disparity mapping used by render:
            disp_px = (depth - conv) * 2 * actual_max_disp_pixels

        The historical MinBorders policy intentionally constrains the search
        range to 0.2..0.8 so the solver cannot collapse to extreme values like
        0.0 that produce overly aggressive pop-out on typical content.
        """
        if not depth_path or not os.path.exists(depth_path):
            self.logger.warning("MinBorders: depth path not found, using fallback.")
            return float(fallback_value)

        conv_values = self._build_convergence_grid(conv_min, conv_max, conv_step)
        if conv_values.size == 0:
            self.logger.warning("MinBorders: empty convergence grid, using fallback.")
            return float(fallback_value)

        try:
            vr_depth = VideoReader(depth_path, ctx=cpu(0))
            total_frames = len(vr_depth)
            if total_frames <= 0:
                self.logger.warning("MinBorders: empty depth video, using fallback.")
                return float(fallback_value)

            if process_length > 0:
                total_frames = min(total_frames, int(process_length))
            step = max(1, int(sample_stride))
            indices = list(range(0, total_frames, step))
            if not indices:
                indices = [0]

            first = vr_depth[0].asnumpy()
            h, w = first.shape[:2]
            width = int(w)
            height = int(h)
            if width <= 0 or height <= 0:
                return float(fallback_value)

            tv_comp = self._get_tv_compensation(depth_path, mode=tv_comp_mode)
            actual_max_disp_pixels = (float(max_disp) / 20.0 / 100.0) * float(width) * float(tv_comp)
            shift_scale = 2.0 * float(actual_max_disp_pixels)
            x_coords = np.arange(width, dtype=np.float32)[None, :]
            frame_area = float(width * height)
            gamma_f = float(gamma) if gamma else 1.0

            sums_left = np.zeros(len(conv_values), dtype=np.float64)
            sums_right = np.zeros(len(conv_values), dtype=np.float64)
            sampled = 0

            self.logger.info(
                f"MinBorders: Sampling {len(indices)} frames from {os.path.basename(depth_path)}..."
            )

            for idx in indices:
                if stop_event and stop_event.is_set():
                    self.logger.info("MinBorders scan cancelled.")
                    break

                raw = vr_depth[int(idx)].asnumpy()
                depth = self._to_gray_depth(raw)
                depth = self._normalize_depth_to_0_1(depth)
                if gamma_f != 1.0:
                    depth = 1.0 - np.power((1.0 - depth).clip(0.0, 1.0), gamma_f)
                    depth = np.clip(depth, 0.0, 1.0)

                dest_base = x_coords + (depth * shift_scale)
                for k, conv in enumerate(conv_values):
                    dest_x = dest_base - (float(conv) * shift_scale)
                    left_run, right_run = self._edge_runs_from_dest(dest_x, width=width)
                    sums_left[k] += float(np.sum(left_run))
                    sums_right[k] += float(np.sum(right_run))

                sampled += 1

            if sampled <= 0:
                return float(fallback_value)

            mean_total = (sums_left + sums_right) / float(sampled)
            best_idx = int(np.argmin(mean_total))
            best_conv = float(conv_values[best_idx])
            best_void_pct = float((mean_total[best_idx] / frame_area) * 100.0) if frame_area > 0.0 else 0.0
            self.logger.info(
                f"MinBorders result: conv={best_conv:.3f}, mean_void={best_void_pct:.3f}%"
            )
            return best_conv

        except Exception as e:
            self.logger.error(f"MinBorders convergence failed: {e}", exc_info=True)
            return float(fallback_value)

    def calculate_hybrid_value(
        self, avg_value: float, peak_value: float
    ) -> float:
        """Calculate hybrid convergence value from average and peak.

        Args:
            avg_value: Average convergence value
            peak_value: Peak convergence value

        Returns:
            Hybrid value (average of avg and peak)
        """
        return (avg_value + peak_value) / 2.0

    def get_cached_value(
        self,
        mode: str,
        cache: Dict[str, float],
        fallback: float = 0.5,
    ) -> float:
        """Get convergence value from cache based on mode.

        Args:
            mode: Mode - 'Average', 'Peak', or 'Hybrid'
            cache: Dictionary with cached values
            fallback: Fallback value if mode not in cache

        Returns:
            Convergence value for the specified mode
        """
        if mode == "Average":
            return cache.get("Average", fallback)
        elif mode == "Peak":
            return cache.get("Peak", fallback)
        elif mode == "MinBorders":
            return cache.get("MinBorders", fallback)
        elif mode == "Hybrid":
            avg = cache.get("Average", fallback)
            peak = cache.get("Peak", fallback)
            return self.calculate_hybrid_value(avg, peak)
        else:
            return fallback
