"""Main splatting render processor.

Implements the core video splatting algorithm, handling the processing loop,
GPU computation (forward warping), and FFmpeg encoding.
"""

import math
import os
import time
import logging
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader

from dependency.stereocrafter_util import (
    start_ffmpeg_pipe_process,
    release_cuda_memory,
    draw_progress_bar,
)
from .forward_warp import ForwardWarpStereo
from .depth_processing import process_depth_batch
from .post_warp import (
    apply_staircase_smooth_bgside,
    build_replace_mask_edge_hole_run,
    left_black_run_mask_from_rgb,
)

logger = logging.getLogger(__name__)


class RenderProcessor:
    """Handles the core splatting render loop for a single video task."""

    def __init__(
        self,
        stop_event: threading.Event,
        progress_queue: queue.Queue,
    ):
        """Initialize render processor.

        Args:
            stop_event: Event to signal stop/cancellation
            progress_queue: Queue for sending progress updates to GUI
        """
        self.stop_event = stop_event
        self.progress_queue = progress_queue
        self._color_encode_flags = {}

    def render_video(
        self,
        input_video_reader: VideoReader,
        depth_map_reader: VideoReader,
        total_frames_to_process: int,
        processed_fps: float,
        output_video_path_base: str,
        target_output_height: int,
        target_output_width: int,
        max_disp: float,
        batch_size: int,
        dual_output: bool,
        zero_disparity_anchor_val: float,
        output_layout: str,
        video_stream_info: Optional[dict],
        input_bias: float,
        assume_raw_input: bool,
        global_depth_min: float,
        global_depth_max: float,
        depth_stream_info: Optional[dict],
        user_output_crf: Optional[int] = None,
        is_low_res_task: bool = False,
        depth_gamma: float = 1.0,
        depth_dilate_size_x: float = 0.0,
        depth_dilate_size_y: float = 0.0,
        depth_blur_size_x: float = 0.0,
        depth_blur_size_y: float = 0.0,
        depth_dilate_left: float = 0.0,
        depth_blur_left: float = 0.0,
        depth_blur_left_mix: float = 0.5,
        skip_lowres_preproc: bool = False,
        color_tags_mode: str = "Auto",
        is_test_mode: bool = False,
        test_target_frame_idx: Optional[int] = None,
        stair_smooth_enabled: bool = False,
        stair_blur_kernel: int = 3,
        stair_edge_x_offset: int = 2,
        stair_strip_px: int = 3,
        stair_strength: float = 1.0,
        stair_debug_mask: bool = False,
        replace_mask_enabled: bool = False,
        replace_mask_dir: str = "./work/mask/",
        replace_mask_scale: float = 1.0,
        replace_mask_min_px: int = 1,
        replace_mask_max_px: int = 32,
        replace_mask_gap_tol: int = 0,
        replace_mask_codec: str = "ffv1",
        replace_mask_draw_edge: bool = True,
        gpu_microbatch_size: int = 2,
    ) -> bool:
        """Core splatting render loop.

        Args:
            input_video_reader: Reader for source video
            depth_map_reader: Reader for depth map
            total_frames_to_process: Number of frames to process
            processed_fps: Output video FPS
            output_video_path_base: Base path for output video
            target_output_height: Target height
            target_output_width: Target width
            max_disp: Max disparity percentage
            batch_size: Frames per batch
            dual_output: Legacy bool flag (True=dual, False=quad)
            output_layout: Output layout ("quad" | "dual" | "single_warp")
            zero_disparity_anchor_val: Convergence anchor (0-1)
            video_stream_info: Metadata for source video
            input_bias: Depth input bias
            assume_raw_input: Whether to skip normalization
            global_depth_min: Global min depth used for normalization
            global_max_depth: Global max depth used for normalization
            depth_stream_info: Metadata for depth map
            user_output_crf: FFmpeg CRF value
            is_low_res_task: Whether this is a low-res pass
            depth_gamma: Gamma correction for depth
            depth_dilate_size_x: X dilation for depth
            depth_dilate_size_y: Y dilation for depth
            depth_blur_size_x: X blur for depth
            depth_blur_size_y: Y blur for depth
            depth_dilate_left: Left-eye dilation
            depth_blur_left: Left-eye blur
            depth_blur_left_mix: Mix factor for left-eye blur
            skip_lowres_preproc: Whether to skip preprocessing for low-res
            color_tags_mode: FFmpeg color tagging mode
            is_test_mode: Whether in diagnostic test mode
            test_target_frame_idx: Specific frame for diagnostic test

        Returns:
            True if completed successfully, False otherwise
        """
        logger.debug("==> Initializing ForwardWarpStereo module")
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        height, width = target_output_height, target_output_width
        os.makedirs(os.path.dirname(output_video_path_base), exist_ok=True)

        # Determine output layout and final path
        layout = str(output_layout or "").strip().lower()
        if layout not in {"quad", "dual", "single_warp"}:
            layout = "dual" if bool(dual_output) else "quad"

        if layout == "single_warp":
            grid_height, grid_width = height, width
            suffix = "_splatted1"
        elif layout == "dual":
            grid_height, grid_width = height, width * 2
            suffix = "_splatted2"
        else:
            grid_height, grid_width = height * 2, width * 2
            suffix = "_splatted4"
        res_suffix = f"_{width}"
        final_output_video_path = f"{os.path.splitext(output_video_path_base)[0]}{res_suffix}{suffix}.mp4"

        task_name = "LowRes" if is_low_res_task else "HiRes"
        self._log_color_metadata(video_stream_info, task_name)

        ffmpeg_process = None
        if not is_test_mode:
            encode_stream_info = self._get_encode_stream_info(video_stream_info, color_tags_mode)
            ffmpeg_process = start_ffmpeg_pipe_process(
                content_width=grid_width,
                content_height=grid_height,
                final_output_mp4_path=final_output_video_path,
                fps=processed_fps,
                video_stream_info=encode_stream_info,
                user_output_crf=user_output_crf,
                output_format_str="splatted_single_warp" if layout == "single_warp" else "splatted_grid",
                debug_label=task_name,
            )
            if ffmpeg_process is None:
                logger.error("Failed to start FFmpeg pipe. Aborting splatting task.")
                return False
            
            self._compare_encoding_flags(ffmpeg_process, task_name)

        replace_mask_proc = None
        replace_mask_path = None
        if replace_mask_enabled and not is_test_mode:
            out_dir = str(replace_mask_dir or "").strip()
            if not out_dir:
                out_dir = os.path.dirname(final_output_video_path)
            os.makedirs(out_dir, exist_ok=True)
            base_no_ext = os.path.splitext(os.path.basename(final_output_video_path))[0]
            replace_mask_path = os.path.join(out_dir, f"{base_no_ext}_replace_mask.mkv")
            logger.info(f"==> Replace-mask export enabled: {replace_mask_path}")

        max_expected_raw_value = self._get_max_expected_raw_depth(depth_stream_info)
        logger.debug(f"[DEPTH] Max expected raw value: {max_expected_raw_value}, assume_raw_input: {assume_raw_input}, global_depth_min: {global_depth_min:.2f}, global_depth_max: {global_depth_max:.2f}")
        
        frame_count = 0
        encoding_successful = True
        try:
            gpu_microbatch_size_i = max(1, int(gpu_microbatch_size))
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid gpu_microbatch_size='{gpu_microbatch_size}', using 2."
            )
            gpu_microbatch_size_i = 2

        try:
            frame_index_iter = (
                [test_target_frame_idx]
                if test_target_frame_idx is not None
                else range(0, total_frames_to_process, batch_size)
            )

            for i in frame_index_iter:
                if self.stop_event.is_set() or (ffmpeg_process is not None and ffmpeg_process.poll() is not None):
                    break

                batch_indices = list(range(i, min(i + batch_size, total_frames_to_process)))
                if not batch_indices:
                    break

                # 1. Fetch frames
                batch_video_numpy = input_video_reader.get_batch(batch_indices).asnumpy()
                batch_depth_numpy_raw = depth_map_reader.get_batch(batch_indices).asnumpy()

                # 2. Normalize and apply gamma (BEFORE dilation/blur)
                # Convert to grayscale if needed
                if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
                    batch_depth_gray = batch_depth_numpy_raw.mean(axis=-1)
                elif batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1:
                    batch_depth_gray = batch_depth_numpy_raw.squeeze(-1)
                else:
                    batch_depth_gray = batch_depth_numpy_raw
                
                batch_depth_float = batch_depth_gray.astype(np.float32)
                
                # Debug: Log raw depth range
                logger.debug(f"[DEPTH] Raw depth range: min={batch_depth_float.min():.2f}, max={batch_depth_float.max():.2f}, shape={batch_depth_float.shape}")
                
                # Normalize to 0-1 range (matches original depthSplatting logic)
                if assume_raw_input:
                    # Raw input mode:
                    # If global_depth_max is > 1.0 (e.g. 255 or 1023 passed from content scan), use it.
                    # Otherwise fallback to max_expected_raw_value from metadata.
                    if global_depth_max > 1.0:
                         batch_depth_normalized = batch_depth_float / global_depth_max
                    else:
                         batch_depth_normalized = batch_depth_float / max(max_expected_raw_value, 1.0)
                else:
                    # Global normalization mode
                    depth_range = global_depth_max - global_depth_min
                    if depth_range > 1e-5:
                        batch_depth_normalized = (batch_depth_float - global_depth_min) / depth_range
                    else:
                        # Collapsed range - fill with convergence anchor value
                        batch_depth_normalized = np.full_like(
                            batch_depth_float,
                            fill_value=zero_disparity_anchor_val,
                            dtype=np.float32,
                        )
                        logger.warning(
                            f"Normalization collapsed to zero range ({global_depth_min:.4f} - {global_depth_max:.4f})."
                        )
                
                batch_depth_normalized = np.clip(batch_depth_normalized, 0.0, 1.0)
                logger.debug(f"[DEPTH] After normalization: min={batch_depth_normalized.min():.4f}, max={batch_depth_normalized.max():.4f}")
                
                # Apply gamma correction (inverted gamma formula)
                if round(float(depth_gamma), 2) != 1.0:
                    batch_depth_normalized = 1.0 - np.power(1.0 - batch_depth_normalized, depth_gamma)
                    batch_depth_normalized = np.clip(batch_depth_normalized, 0.0, 1.0)
                    logger.debug(f"[DEPTH] After gamma {depth_gamma}: min={batch_depth_normalized.min():.4f}, max={batch_depth_normalized.max():.4f}")
                
                # Convert back to "raw" format for process_depth_batch (which expects raw-like values)
                # Scale back to max_raw_value range so dilation/blur work correctly
                batch_depth_for_processing = batch_depth_normalized * max_expected_raw_value
                
                # Add channel dimension for process_depth_batch
                if batch_depth_for_processing.ndim == 3:
                    batch_depth_for_processing = batch_depth_for_processing[..., None]

                # 3. Process depth batch (dilation/blur)
                batch_depth_processed = process_depth_batch(
                    batch_depth_numpy_raw=batch_depth_for_processing,
                    depth_gamma=1.0,  # Already applied above
                    depth_dilate_size_x=depth_dilate_size_x,
                    depth_dilate_size_y=depth_dilate_size_y,
                    depth_blur_size_x=depth_blur_size_x,
                    depth_blur_size_y=depth_blur_size_y,
                    max_raw_value=max_expected_raw_value,
                    depth_dilate_left=depth_dilate_left,
                    depth_blur_left=depth_blur_left,
                    depth_blur_left_mix=depth_blur_left_mix,
                    skip_preprocessing=skip_lowres_preproc and is_low_res_task,
                )
                
                # Normalize back to 0-1 after processing
                batch_depth_numpy_float = batch_depth_processed / max(max_expected_raw_value, 1.0)
                batch_depth_numpy_float = np.clip(batch_depth_numpy_float, 0.0, 1.0)

                # 4. GPU Splatting
                batch_processed_frames = self._process_gpu_splatting(
                    stereo_projector=stereo_projector,
                    batch_video_numpy=batch_video_numpy,
                    batch_depth_numpy_float=batch_depth_numpy_float,
                    target_width=width,
                    target_height=height,
                    max_disp=max_disp,
                    zero_disparity_anchor_val=zero_disparity_anchor_val,
                    input_bias=input_bias,
                    stair_smooth_enabled=stair_smooth_enabled,
                    stair_blur_kernel=stair_blur_kernel,
                    stair_edge_x_offset=stair_edge_x_offset,
                    stair_strip_px=stair_strip_px,
                    stair_strength=stair_strength,
                    stair_debug_mask=stair_debug_mask,
                    replace_mask_enabled=replace_mask_enabled,
                    replace_mask_scale=replace_mask_scale,
                    replace_mask_min_px=replace_mask_min_px,
                    replace_mask_max_px=replace_mask_max_px,
                    replace_mask_gap_tol=replace_mask_gap_tol,
                    replace_mask_draw_edge=replace_mask_draw_edge,
                    gpu_microbatch_size=gpu_microbatch_size_i,
                )

                if replace_mask_enabled and replace_mask_path:
                    if replace_mask_proc is None:
                        codec = str(replace_mask_codec or "ffv1").strip().lower()
                        allowed_codecs = {"ffv1", "huffyuv", "utvideo", "png"}
                        if codec not in allowed_codecs:
                            codec = "ffv1"
                        first_mask = None
                        for _res in batch_processed_frames:
                            if "replace_mask" in _res:
                                first_mask = _res["replace_mask"]
                                break
                        if first_mask is not None:
                            mask_h, mask_w = first_mask.shape[:2]
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                "-f",
                                "rawvideo",
                                "-pix_fmt",
                                "gray",
                                "-s",
                                f"{mask_w}x{mask_h}",
                                "-r",
                                str(processed_fps),
                                "-i",
                                "-",
                                "-an",
                                "-c:v",
                                codec,
                                "-pix_fmt",
                                "gray",
                                replace_mask_path,
                            ]
                            replace_mask_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                            logger.info(f"==> Replace-mask ffmpeg started: {replace_mask_path}")
                    if replace_mask_proc is not None and replace_mask_proc.stdin is not None:
                        for _res in batch_processed_frames:
                            mask_u8 = _res.get("replace_mask")
                            if mask_u8 is None:
                                continue
                            if mask_u8.ndim == 3:
                                mask_u8 = mask_u8[..., 0]
                            if mask_u8.dtype != np.uint8:
                                mask_u8 = mask_u8.astype(np.uint8)
                            replace_mask_proc.stdin.write(mask_u8.tobytes())

                # 5. Handle results (diag tests or FFmpeg write)
                if is_test_mode and test_target_frame_idx is not None:
                    self._handle_diagnostic_capture(batch_processed_frames, layout)
                elif ffmpeg_process:
                    self._write_to_ffmpeg(ffmpeg_process, batch_processed_frames, layout)

                frame_count += len(batch_indices)
                self.progress_queue.put(("processed", frame_count))
                if not is_test_mode:
                    draw_progress_bar(frame_count, total_frames_to_process, suffix=f"{task_name} Batch {i//batch_size}")
                
                # Cleanup batch
                del batch_video_numpy, batch_depth_numpy_raw, batch_depth_numpy_float, batch_processed_frames
                release_cuda_memory()

        except Exception as e:
            logger.error(f"Render error: {e}", exc_info=True)
            encoding_successful = False
        finally:
            if ffmpeg_process:
                try:
                    if ffmpeg_process.stdin:
                        ffmpeg_process.stdin.close()
                    ffmpeg_process.wait(timeout=30)
                except Exception as e:
                    logger.warning(f"Error closing FFmpeg: {e}")

            if replace_mask_proc:
                try:
                    if replace_mask_proc.stdin:
                        replace_mask_proc.stdin.close()
                    replace_mask_proc.wait(timeout=30)
                    if replace_mask_proc.returncode not in (0, None):
                        logger.warning(
                            f"Replace-mask ffmpeg exited with code {replace_mask_proc.returncode}: {replace_mask_path}"
                        )
                except Exception as e:
                    logger.warning(f"Error closing replace-mask FFmpeg: {e}")
            
            del stereo_projector
            release_cuda_memory()

        return encoding_successful

    def _log_color_metadata(self, info: Optional[dict], task_name: str):
        if not info: return
        try:
            logger.debug(
                f"[COLOR_META][{task_name}] input ffprobe: "
                f"pix_fmt={info.get('pix_fmt')}, range={info.get('color_range')}, "
                f"primaries={info.get('color_primaries')}, trc={info.get('transfer_characteristics')}, "
                f"matrix={info.get('color_space')}"
            )
        except Exception: pass

    def _get_encode_stream_info(self, source_info: Optional[dict], mode: str) -> dict:
        info = dict(source_info) if source_info else {}
        defaults = {
            "color_primaries": "bt709",
            "transfer_characteristics": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
        }
        
        if mode == "Auto":
            for k, v in defaults.items(): info.setdefault(k, v)
        elif mode == "BT.709 L":
            info.update(defaults)
        elif mode == "BT.709 F":
            info.update(defaults)
            info["color_range"] = "pc"
        elif mode == "BT.2020 PQ":
            info.update({"color_primaries": "bt2020", "transfer_characteristics": "smpte2084", "color_space": "bt2020nc", "color_range": "tv"})
        elif mode == "BT.2020 HLG":
            info.update({"color_primaries": "bt2020", "transfer_characteristics": "arib-std-b67", "color_space": "bt2020nc", "color_range": "tv"})
        else:
            for k, v in defaults.items(): info.setdefault(k, v)
        return info

    def _compare_encoding_flags(self, process: Any, task_name: str):
        try:
            flags = getattr(process, "sc_encode_flags", None)
            if not flags: return
            subset_keys = ["enc_codec", "enc_pix_fmt", "enc_profile", "enc_color_primaries", "enc_color_trc", "enc_colorspace", "quality_mode", "quality_value"]
            subset = {k: flags.get(k) for k in subset_keys}
            self._color_encode_flags[task_name] = subset
            
            other_name = "HiRes" if task_name == "LowRes" else "LowRes"
            if other_name in self._color_encode_flags:
                other = self._color_encode_flags[other_name]
                diffs = {k: (other.get(k), subset.get(k)) for k in subset_keys if other.get(k) != subset.get(k)}
                if diffs:
                    logger.warning(f"[COLOR_META] Encoding flags differ ({other_name} vs {task_name}): {diffs}")
                else:
                    logger.debug(f"[COLOR_META] Encoding flags match between {other_name} and {task_name}.")
        except Exception: pass

    def _get_max_expected_raw_depth(self, info: Optional[dict]) -> float:
        pix_fmt = info.get("pix_fmt") if info else None
        profile = info.get("profile") if info else None
        logger.debug(f"[DEPTH] Detecting bit depth from depth_stream_info: pix_fmt={pix_fmt}, profile={profile}, full_info={info}")
        if pix_fmt:
            if "10" in pix_fmt or "gray10" in pix_fmt or "12" in pix_fmt or (profile and "main10" in profile):
                logger.debug(f"[DEPTH] Detected 10-bit depth (pix_fmt={pix_fmt})")
                return 1023.0
            if "8" in pix_fmt or pix_fmt in ["yuv420p", "yuv422p", "yuv444p"]:
                logger.debug(f"[DEPTH] Detected 8-bit depth (pix_fmt={pix_fmt})")
                return 255.0
            if "float" in pix_fmt:
                logger.debug(f"[DEPTH] Detected float depth (pix_fmt={pix_fmt})")
                return 1.0
        logger.warning(f"[DEPTH] Could not detect bit depth, defaulting to 1.0 (pix_fmt={pix_fmt})")
        return 1.0

    def _process_gpu_splatting(
        self,
        stereo_projector: ForwardWarpStereo,
        batch_video_numpy: np.ndarray,
        batch_depth_numpy_float: np.ndarray,
        target_width: int,
        target_height: int,
        max_disp: float,
        zero_disparity_anchor_val: float,
        input_bias: float,
        stair_smooth_enabled: bool = False,
        stair_blur_kernel: int = 3,
        stair_edge_x_offset: int = 2,
        stair_strip_px: int = 3,
        stair_strength: float = 1.0,
        stair_debug_mask: bool = False,
        replace_mask_enabled: bool = False,
        replace_mask_scale: float = 1.0,
        replace_mask_min_px: int = 1,
        replace_mask_max_px: int = 32,
        replace_mask_gap_tol: int = 0,
        replace_mask_draw_edge: bool = True,
        gpu_microbatch_size: int = 2,
    ) -> List[np.ndarray]:
        """Process GPU splatting on normalized depth maps.
        
        Args:
            batch_depth_numpy_float: Pre-normalized depth in range [0, 1]
        """
        # CRITICAL: Ensure depth matches video resolution before GPU processing
        # batch_video_numpy: [B, H, W, 3]
        # batch_depth_numpy_float: [B, H', W'] - already normalized to [0, 1]
        
        video_h, video_w = batch_video_numpy.shape[1], batch_video_numpy.shape[2]
        
        # Handle depth shape - ensure it's [B, H, W]
        if batch_depth_numpy_float.ndim == 4:
            if batch_depth_numpy_float.shape[-1] == 1:
                batch_depth_numpy_float = batch_depth_numpy_float.squeeze(-1)
            elif batch_depth_numpy_float.shape[-1] == 3:
                batch_depth_numpy_float = batch_depth_numpy_float[..., 0]
        
        depth_h, depth_w = batch_depth_numpy_float.shape[1], batch_depth_numpy_float.shape[2]
        
        # Resize depth if dimensions don't match
        if depth_h != video_h or depth_w != video_w:
            logger.debug(f"Resizing depth from {depth_w}x{depth_h} to match video {video_w}x{video_h}")

            
            interp = cv2.INTER_AREA if (video_w < depth_w and video_h < depth_h) else cv2.INTER_LINEAR
            resized_depth = np.empty((batch_depth_numpy_float.shape[0], video_h, video_w), dtype=batch_depth_numpy_float.dtype)
            
            for idx in range(batch_depth_numpy_float.shape[0]):
                resized_depth[idx] = cv2.resize(
                    batch_depth_numpy_float[idx],
                    (video_w, video_h),
                    interpolation=interp
                )
            batch_depth_numpy_float = resized_depth
        
        total_frames = int(batch_video_numpy.shape[0])
        if total_frames <= 0:
            return []

        try:
            microbatch = max(1, int(gpu_microbatch_size))
        except (TypeError, ValueError):
            microbatch = 2
        microbatch = min(microbatch, total_frames)

        # Keep disparity scaling identical to full-batch path.
        actual_max_disp_pixels = (max_disp / 20.0 / 100.0) * target_width
        results: List[dict] = []

        for mb_start in range(0, total_frames, microbatch):
            mb_end = min(mb_start + microbatch, total_frames)
            batch_video_mb = batch_video_numpy[mb_start:mb_end]
            batch_depth_mb = batch_depth_numpy_float[mb_start:mb_end]

            # Move only a micro-batch to GPU to cap peak VRAM.
            source_tensor = (
                torch.from_numpy(batch_video_mb).permute(0, 3, 1, 2).float().cuda() / 255.0
            )
            depth_tensor = torch.from_numpy(batch_depth_mb).unsqueeze(1).float().cuda()

            depth_tensor = torch.clip(depth_tensor, 0.0, 1.0)
            if input_bias != 0:
                depth_tensor = torch.clip(depth_tensor + input_bias, 0.0, 1.0)

            disp_map = (depth_tensor - zero_disparity_anchor_val) * 2.0
            disp_map = disp_map * actual_max_disp_pixels

            with torch.no_grad():
                right_eye_raw, occlusion_mask = stereo_projector(source_tensor, disp_map)

                disp_out_winner = None
                if stair_smooth_enabled or stair_debug_mask or replace_mask_enabled:
                    maxd = (
                        float(actual_max_disp_pixels)
                        if float(actual_max_disp_pixels) != 0.0
                        else 1.0
                    )
                    disp_norm_rgb = (
                        (disp_map / (2.0 * maxd) + 0.5)
                        .clamp(0.0, 1.0)
                        .repeat(1, 3, 1, 1)
                    )
                    right_disp_norm_rgb, _ = stereo_projector(disp_norm_rgb, disp_map)
                    disp_out_winner = (right_disp_norm_rgb[:, 0:1] - 0.5) * (2.0 * maxd)

                right_eye_processed = right_eye_raw
                if stair_smooth_enabled or stair_debug_mask:
                    right_eye_processed = apply_staircase_smooth_bgside(
                        right_eye_processed,
                        occlusion_mask,
                        disp_out_winner if disp_out_winner is not None else disp_map,
                        max_disp=float(actual_max_disp_pixels),
                        edge_mode="pos",
                        grad_thr_px=1.0,
                        strip_px=int(stair_strip_px),
                        strength=float(stair_strength),
                        right_margin_extra=0,
                        debug_mask=bool(stair_debug_mask),
                        exclude_near_holes=True,
                        hole_dilate=8,
                        edge_x_offset=int(stair_edge_x_offset),
                        blur_kernel=int(stair_blur_kernel),
                    )

                replace_mask_u8 = None
                if replace_mask_enabled and disp_out_winner is not None:
                    hole_bool = occlusion_mask > 0.5
                    rep = build_replace_mask_edge_hole_run(
                        disp_out_winner,
                        hole_bool,
                        grad_thr_px=1.0,
                        min_px=int(replace_mask_min_px),
                        max_px=int(replace_mask_max_px),
                        scale=float(replace_mask_scale),
                        gap_tol=int(replace_mask_gap_tol),
                        draw_edge=bool(replace_mask_draw_edge),
                    )
                    try:
                        eff_max = max(
                            1,
                            int(
                                round(
                                    int(replace_mask_max_px) * float(replace_mask_scale)
                                )
                            ),
                        )
                        left_run = left_black_run_mask_from_rgb(
                            right_eye_processed, tol=0.0, max_px=eff_max
                        )
                        rep = rep | left_run
                    except Exception:
                        pass
                    replace_mask_u8 = (
                        (rep[:, 0].to(torch.uint8) * 255).contiguous().cpu().numpy()
                    )

            left_cpu = source_tensor.cpu().numpy()
            right_cpu = right_eye_processed.cpu().numpy()
            occl_cpu = occlusion_mask.cpu().numpy()
            depth_cpu = depth_tensor.cpu().numpy()

            for j in range(len(batch_video_mb)):
                item = {
                    "left": (
                        np.clip(left_cpu[j].transpose(1, 2, 0), 0, 1) * 255
                    ).astype(np.uint8),
                    "right": (
                        np.clip(right_cpu[j].transpose(1, 2, 0), 0, 1) * 255
                    ).astype(np.uint8),
                    "occlusion": (
                        np.clip(occl_cpu[j].transpose(1, 2, 0), 0, 1) * 255
                    ).astype(np.uint8),
                    "depth": (
                        np.clip(depth_cpu[j].transpose(1, 2, 0), 0, 1) * 255
                    ).astype(np.uint8),
                }
                if replace_mask_u8 is not None:
                    item["replace_mask"] = replace_mask_u8[j]
                results.append(item)

        return results


    def _handle_diagnostic_capture(self, batch_results: List[dict], output_layout: str):
        # In test mode, we usually only have one frame
        res = batch_results[0]
        # This is a bit tricky since we don't have direct access to GUI previewer here.
        # We'll put it in the queue for the GUI to handle.
        grid = self._construct_grid(res, output_layout)
        self.progress_queue.put(("diagnostic_capture", grid))

    def _write_to_ffmpeg(self, process: Any, batch_results: List[dict], output_layout: str):
        for res in batch_results:
            grid = self._construct_grid(res, output_layout)
            # Convert to 16-bit and BGR for FFmpeg
            grid_uint16 = (np.clip(grid, 0.0, 1.0) * 65535.0).astype(np.uint16)
            grid_bgr = cv2.cvtColor(grid_uint16, cv2.COLOR_RGB2BGR)
            process.stdin.write(grid_bgr.tobytes())

    def _construct_grid(self, res: dict, output_layout: str) -> np.ndarray:
        """Construct output grid for encoding.
        
        output_layout='dual': [occlusion_mask | right_eye] (2-panel)
        output_layout='quad': [left_eye | depth_vis] / [occlusion_mask | right_eye] (4-panel)
        output_layout='single_warp': [right_eye] only (1-panel)
        
        Returns float32 array in range [0, 1]
        """
        # Convert uint8 back to float for grid assembly
        left = res["left"].astype(np.float32) / 255.0
        right = res["right"].astype(np.float32) / 255.0
        occlusion = res["occlusion"].astype(np.float32) / 255.0
        depth = res["depth"].astype(np.float32) / 255.0
        
        # Ensure all are 3-channel
        if occlusion.ndim == 2 or (occlusion.ndim == 3 and occlusion.shape[-1] == 1):
            occlusion = np.stack([occlusion.squeeze()] * 3, axis=-1)
        if depth.ndim == 2 or (depth.ndim == 3 and depth.shape[-1] == 1):
            depth = np.stack([depth.squeeze()] * 3, axis=-1)
        
        layout = str(output_layout or "").strip().lower()
        if layout == "single_warp":
            # 1-panel: warped right eye only
            return right
        if layout == "dual":
            # 2-panel: occlusion on left, warped right eye on right
            return np.concatenate([occlusion, right], axis=1)
        # 4-panel: top row (left, depth), bottom row (occlusion, right)
        top_row = np.concatenate([left, depth], axis=1)
        bot_row = np.concatenate([occlusion, right], axis=1)
        return np.concatenate([top_row, bot_row], axis=0)
