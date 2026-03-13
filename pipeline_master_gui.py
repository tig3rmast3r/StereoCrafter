import json
import os
import queue
import re
import csv
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import concurrent.futures
import glob
import tkinter as tk
from datetime import datetime
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from ttkthemes import ThemedTk
except Exception:
    ThemedTk = None

GUI_VERSION = "2026-03-13"


class PipelineMasterGUI:
    CONFIG_FILENAME = "config_pipeline_master_gui.json"
    DEFAULT_SCENE_BACKEND = "OpenCV"
    DEFAULT_SCENE_CODEC = "libx264"
    DEFAULT_WINDOW_GEOMETRY = "1400x1050"
    FFMPEG_CODEC_CHOICES = ("libx264", "libx265", "h264_nvenc", "hevc_nvenc")
    FFMPEG_CODEC_ALIASES = {
        "x264": "libx264",
        "x265": "libx265",
        "h265": "libx265",
    }
    DEFAULT_DEPTH_SCALE_FACTOR = 0.5
    MIN_DEPTH_SCALE_FACTOR = 0.5
    MAX_DEPTH_SCALE_FACTOR = 0.8
    DEFAULT_DEPTH_REALESRGAN_WORKERS = 4
    DEFAULT_SPLIT_SCENES_WORKERS = 8
    DEFAULT_PIPELINE_TEST_RUN_FILES = 5
    RETRY_POLICY_PROFILES = ("run", "retry1", "retry2", "retry3")
    RETRY_POLICY_MAX_SPLIT_CHOICES = ("off", "64", "128", "256", "512")
    RETRY_POLICY_OFFLOAD_CHOICES = ("none", "model", "sequential")
    RETRY_POLICY_DEFAULT = {
        "run": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "off",
            "cpu_offload_inherited": True,
            "cpu_offload_mode": "model",
        },
        "retry1": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "512",
            "cpu_offload_inherited": True,
            "cpu_offload_mode": "model",
        },
        "retry2": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "64",
            "cpu_offload_inherited": True,
            "cpu_offload_mode": "model",
        },
        "retry3": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "64",
            "cpu_offload_inherited": False,
            "cpu_offload_mode": "sequential",
        },
    }
    VERIFY_QUICK_FFPROBE_TIMEOUT_SEC = 3600.0
    VERIFY_QUICK_FFPROBE_TIMEOUT_RETRIES = 1
    VERIFY_DEEP_FFPROBE_TIMEOUT_SEC = 180.0
    VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES = 1
    CONTENT_THRESHOLD_NOTICE = (
        "Content detector usually requires a much higher threshold.\n"
        "Suggested start: around 27 (instead of ~2 for Adaptive)."
    )
    HDR_FORCE_NOTICE = (
        "HDR source detected.\n\n"
        "StereoCrafter pipeline does not preserve full 10-bit HDR end-to-end.\n"
        "Input will be converted immediately to SDR BT.709 8-bit before splitting.\n\n"
        "Tonemap:\n"
        "- Mobius: more HDR-like rolloff.\n"
        "- Hable: brighter SDR-style look."
    )
    DEPTH_AUTO_INFO = (
        "Auto mode: source scenes are downscaled with a selectable factor (0.50..0.80),\n"
        "processed with DepthCrafter, then restored with RealESRGAN.\n"
        "RealESRGAN runtime can be selected in Options -> DepthCrafter:\n"
        "Bundled (Utilities/realesrgan) or Local (system/custom).\n"
        "If local runtime has issues, switch to Bundled or rebuild locally."
    )
    DEPTH_MANUAL_INFO = (
        "Manual mode: choose parameters freely. For compatibility with the pipeline,\n"
        "output scenes will still be restored automatically to original resolution.\n"
        "Segmenting is not supported in this script.\n"
        "If you need segmenting, use depthcrafter_gui_seg.py manually."
    )
    DEPTH_OVERRIDE_WARNING = (
        "Depth encoding override enabled.\n\n"
        "These manual changes are NOT monitored by the pipeline.\n"
        "Only proceed if you know exactly what you are writing."
    )
    DEPTH_SEG_MONO_NOTICE = (
        "Before running DepthCrafter, review scene clips and move any parts you do not want "
        "to convert (for example intros/endings) into `seg-mono`.\n"
        "`seg-mono` is created automatically right after SceneDetect completes.\n"
        "Files moved there are skipped in DepthCrafter and recovered in the final Rejoin step."
    )
    SPLAT_AUTO_INFO = (
        "Auto mode applies recommended splatting settings."
    )
    SPLAT_MANUAL_INFO = (
        "Manual mode unlocks all splatting controls.\n"
        "Use only if you know exactly how these parameters affect output."
    )
    SPLAT_DISCLAIMER = (
        "Borders/sidecar handling and lowres encoding are not supported in this script.\n"
        "If needed, use splatting_gui.py."
    )
    SPLAT_OVERRIDE_WARNING = (
        "Splat encoding override enabled.\n\n"
        "These manual changes are NOT monitored by the pipeline.\n"
        "Only proceed if you know exactly what you are writing."
    )
    INPAINT_OVERRIDE_WARNING = (
        "Inpainting encoding override enabled.\n\n"
        "These manual changes are NOT monitored by the pipeline.\n"
        "Only proceed if you know exactly what you are writing."
    )
    INPAINT_AUTO_INFO = (
        "Auto mode applies recommended inpainting settings."
    )
    INPAINT_MANUAL_INFO = (
        "Manual mode unlocks all inpainting controls.\n"
        "Use only if you know exactly how these parameters affect output."
    )
    INPAINT_DISCLAIMER = (
        "Post-processing blend and color transfer are not supported in this script.\n"
        "If needed, use inpainting_gui.py."
    )
    MERGE_AUTO_INFO = (
        "Auto mode applies recommended merge defaults."
    )
    MERGE_MANUAL_INFO = (
        "Manual mode unlocks merge tuning controls."
    )
    MERGE_DISCLAIMER = (
        "Borders and sidecar handling are not supported in this script.\n"
        "If needed, use merging_gui.py."
    )
    SPLAT_REPLACE_MASK_WARNING = (
        "Replace mask export is disabled.\n\n"
        "Some downstream features (Sharpness CSV and AutoCT CSV) will be unavailable."
    )
    MERGE_OVERRIDE_WARNING = (
        "Merging encoding override enabled.\n\n"
        "These manual changes are NOT monitored by the pipeline.\n"
        "Only proceed if you know exactly what you are writing."
    )
    JOIN_AUTO_INFO = (
        "Auto mode uses Rejoin defaults with NVENC-style settings.\n"
        "Only quality value is editable here."
    )
    JOIN_MANUAL_INFO = (
        "Manual mode unlocks encoder and advanced ffmpeg args for join.\n"
        "Use only if you know exactly what you are doing."
    )
    JOIN_MANUAL_WARNING = (
        "Manual Join mode enabled.\n\n"
        "Advanced ffmpeg args are not validated by the pipeline.\n"
        "Default args are tuned for NVENC; if you change encoder/args, verify output carefully."
    )
    JOIN_DEFAULT_ARGS = (
        "-rc vbr -b:v 0 -multipass fullres -spatial_aq 1 -temporal_aq 1 "
        "-aq-strength 12 -rc-lookahead 32 -bf 3"
    )
    PIPELINE_STEPS = [
        ("scenedetect", "SceneDetect"),
        ("split_scenes", "Split Scenes"),
        ("depthcrafter", "DepthCrafter"),
        ("depth_upscale", "Depth Upscale"),
        ("splatting", "Splatting"),
        ("sharpness_csv", "Sharpness CSV"),
        ("inpaint", "Inpaint"),
        ("mask_for_merge", "Mask-for-merge"),
        ("autoct_csv", "AutoCT CSV"),
        ("merging", "Merging"),
        ("mono_to_sbs", "Mono->SBS"),
        ("join", "Join"),
        ("remux", "Remux"),
    ]
    PIPELINE_STEPS_WITH_VERIFY = {
        "split_scenes",
        "depthcrafter",
        "depth_upscale",
        "splatting",
        "inpaint",
        "mask_for_merge",
        "merging",
        "mono_to_sbs",
        "join",
    }
    PIPELINE_CSV_STEPS = {"sharpness_csv", "autoct_csv"}
    PIPELINE_OPTIONAL_STEPS = {"sharpness_csv", "autoct_csv"}
    PIPELINE_VERIFY_CHOICES = ["Disabled", "Quick", "Deep"]
    PIPELINE_STATE_FILENAME = "pipeline_state.json"
    VERIFY_VIDEO_PATTERNS = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"]
    VERIFY_REPLACE_MASK_PATTERNS = [
        "*_replace_mask.mp4",
        "*_replace_mask.mkv",
        "*_replace_mask.mov",
        "*_replace_mask.avi",
        "*_replace_mask.webm",
    ]
    VERIFY_ALL_VIDEO_EXTENSIONS = ".mp4,.mkv,.mov,.avi,.webm"

    STANDARD_SUBDIRS = {
        "scenes": "seg",
        "depth": "depthmap",
        "splat": "splat",
        "mask": "mask",
        "mask_for_merge": "mask_for_merge",
        "inpaint": "output",
        "merge": "sbs",
        "join": "final",
    }

    TONEMAP_PRESET_TO_FFMPEG = {
        "Mobius (HDR style, available only for 10-bit input source)": "mobius",
        "Hable (brighter SDR style, available only for 10-bit input source)": "hable",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"StereoCrafter Pipeline GUI {GUI_VERSION}")
        self._config = self._load_config()
        saved_geometry = str(self._config.get("window_geometry", "")).strip()
        if saved_geometry:
            try:
                self.root.geometry(saved_geometry)
            except Exception:
                self.root.geometry(self.DEFAULT_WINDOW_GEOMETRY)
        else:
            self.root.geometry(self.DEFAULT_WINDOW_GEOMETRY)
        self.root.minsize(1180, 760)
        self._log_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

        self._scene_thread: threading.Thread | None = None
        self._scene_process: subprocess.Popen | None = None
        self._scene_stop_requested = False
        self._scene_active_step = "scenedetect"

        self._verify_thread: threading.Thread | None = None
        self._verify_running = False
        self._verify_mode: str = ""
        self._scene_verify_result_applied = False

        self._analysis_thread: threading.Thread | None = None
        self._analysis_running = False
        self._analysis_stop_requested = False
        self._analysis_process: subprocess.Popen | None = None

        self._depth_thread: threading.Thread | None = None
        self._depth_process: subprocess.Popen | None = None
        self._depth_stop_requested = False
        self._depth_stop_clicks = 0
        self._splat_thread: threading.Thread | None = None
        self._splat_process: subprocess.Popen | None = None
        self._splat_stop_requested = False
        self._splat_stop_clicks = 0
        self._inpaint_thread: threading.Thread | None = None
        self._inpaint_process: subprocess.Popen | None = None
        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self._inpaint_resume_after_sharpness = False
        self._merge_thread: threading.Thread | None = None
        self._merge_process: subprocess.Popen | None = None
        self._merge_stop_requested = False
        self._merge_stop_clicks = 0
        self._join_thread: threading.Thread | None = None
        self._join_process: subprocess.Popen | None = None
        self._join_stop_requested = False
        self._join_stop_clicks = 0
        self._join_manual_notice_shown = False
        self._join_expected_duration_sec: float | None = None
        self._join_active_output_path: str = ""
        self._pipeline_step_state = self._default_pipeline_step_state()
        self._pipeline_step_widgets: dict[str, dict[str, tk.Widget]] = {}
        self._pipeline_autorun = False
        self._pipeline_pending_action: tuple[str, str, str] | None = None
        self._pipeline_check_files_done = False
        self._pipeline_file_scan: dict[str, object] = {}
        self._pipeline_test_active = False
        self._pipeline_pause_after_split_scenes = False
        self._pipeline_test_manifest: list[str] = []
        self._pipeline_test_scene_stems: list[str] = []
        self._pipeline_test_source_dir: str = ""
        self._pipeline_test_dir: str = ""
        self._pipeline_test_prev_paths: dict[str, str] = {}
        self._pipeline_test_step_state = self._default_pipeline_step_state()
        self._pipeline_ui_noninteractive = False
        self._pipeline_popup_log_buffer: list[str] = []
        self.pipeline_popup_log_text: tk.Text | None = None
        self._messagebox_originals = {
            "showinfo": messagebox.showinfo,
            "showwarning": messagebox.showwarning,
            "showerror": messagebox.showerror,
            "askyesno": messagebox.askyesno,
        }
        self._splat_override_notice_shown = False
        self._inpaint_override_notice_shown = False
        self._merge_override_notice_shown = False

        self._source_video_info: dict = {}
        self._source_capabilities: dict = {}
        self._recommended_crop_filters: dict = {}
        self._crop_recommendation_profile: dict = {}
        self._depth_input_resolution_cache_key: tuple[str, str, int] | None = None
        self._depth_input_resolution_cache_value: tuple[int | None, int | None] = (None, None)
        self._scene_crop_target_syncing = False
        self._crop_notice_shown = False
        self._hdr_notice_shown = False
        self._content_notice_shown = False
        self._depth_override_notice_shown = False
        self._install_messagebox_wrappers()

        self._init_vars()
        self._build_ui()
        self._refresh_standard_paths()
        self._load_pipeline_state()
        self._apply_option_states()
        self._preview_scene_command()
        self._refresh_pipeline_status_panel()
        self._poll_log_queue()
        self.root.after(200, self._start_source_analysis_on_startup)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_vars(self) -> None:
        self.work_folder_var = tk.StringVar(
            value=self._config.get("work_folder", os.path.normpath("./work"))
        )
        self.scene_input_var = tk.StringVar(
            value=self._config.get("scene_input", os.path.normpath("./work/source.mkv"))
        )
        self.scene_output_var = tk.StringVar(value="")

        self.scene_detector_var = tk.StringVar(
            value=self._config.get("scene_detector", "Adaptive")
        )
        self.scene_threshold_var = tk.StringVar(
            value=str(self._config.get("scene_threshold", "2.0"))
        )
        self.scene_backend_var = tk.StringVar(
            value=self._config.get("scene_backend", self.DEFAULT_SCENE_BACKEND)
        )

        crop_mode = str(self._config.get("scene_crop_mode", "auto")).strip().lower()
        if crop_mode in {"min", "clean"}:
            crop_mode = "auto"
        if crop_mode in {"none", "custom"}:
            crop_mode = "manual"
        if crop_mode not in {"auto", "manual"}:
            crop_mode = "auto"
        self.scene_crop_mode_var = tk.StringVar(value=crop_mode)
        self.scene_crop_custom_var = tk.StringVar(
            value=self._config.get("scene_crop_custom", "")
        )
        self.scene_crop_target_h_var = tk.StringVar(
            value=str(self._config.get("scene_crop_target_h", "")).strip()
        )
        self.scene_crop_auto_desc_var = tk.StringVar(value="n.d.")
        self.scene_crop_tile_compat_var = tk.StringVar(value="n.d.")

        self.scene_layout_var = tk.StringVar(
            value=self._config.get("scene_layout", "Full-SBS (quality)")
        )
        self.scene_tonemap_var = tk.StringVar(
            value=self._config.get(
                "scene_tonemap",
                "Mobius (HDR style, available only for 10-bit input source)",
            )
        )
        self.scene_chroma_var = tk.StringVar(
            value=self._config.get("scene_chroma", "420")
        )

        self.scene_codec_var = tk.StringVar(
            value=self._config.get("scene_codec", self.DEFAULT_SCENE_CODEC)
        )
        self.scene_crf_var = tk.StringVar(value=str(self._config.get("scene_crf", "1")))
        self.scene_encoder_preset_var = tk.StringVar(
            value=self._config.get("scene_encoder_preset", "fast")
        )
        self.scene_pix_fmt_var = tk.StringVar(
            value=self._config.get("scene_pix_fmt", "yuv420p")
        )
        self.scene_extra_ffmpeg_args_var = tk.StringVar(
            value=self._config.get("scene_extra_ffmpeg_args", "")
        )
        self.scene_split_threads_var = tk.StringVar(
            value=str(
                self._config.get(
                    "scene_split_threads",
                    str(self.DEFAULT_SPLIT_SCENES_WORKERS),
                )
            )
        )

        self.scene_cmd_preview_var = tk.StringVar(value="")
        self.scene_status_var = tk.StringVar(value="Ready")
        self.scene_progress_var = tk.DoubleVar(value=0.0)
        self.scene_analysis_status_var = tk.StringVar(value="Ready")
        self.scene_option_hint_var = tk.StringVar(
            value="Analyze source video to unlock source-driven options."
        )

        # Structured source analysis fields.
        self.analysis_source_path_var = tk.StringVar(value="n.d.")
        self.analysis_resolution_var = tk.StringVar(value="n.d.")
        self.analysis_bars_var = tk.StringVar(value="n.d.")
        self.analysis_color_var = tk.StringVar(value="n.d.")
        self.analysis_pixfmt_var = tk.StringVar(value="n.d.")
        self.analysis_length_var = tk.StringVar(value="n.d.")
        self.analysis_fps_var = tk.StringVar(value="n.d.")
        self.analysis_bitrate_var = tk.StringVar(value="n.d.")

        self.hdr_policy_var = tk.StringVar(
            value="HDR -> SDR 8-bit BT.709: auto (requires source analysis)"
        )

        # DepthCrafter tab (GUI only for now, no runner integration yet).
        self.depth_input_var = tk.StringVar(value="")
        self.depth_output_var = tk.StringVar(value="")
        self.depth_upscaled_var = tk.StringVar(value="")
        self.depth_mode_var = tk.StringVar(
            value=self._config.get("depth_mode", "Auto (recommended)")
        )
        self.depth_info_text_var = tk.StringVar(value=self.DEPTH_AUTO_INFO)
        self.depth_chunk_size_var = tk.StringVar(
            value=str(self._config.get("depth_chunk_size", "70"))
        )
        self.depth_overlap_var = tk.StringVar(
            value=str(self._config.get("depth_overlap", "20"))
        )
        self.depth_inference_steps_var = tk.StringVar(
            value=str(self._config.get("depth_inference_steps", "5"))
        )
        self.depth_cpu_offload_var = tk.StringVar(
            value=self._config.get("depth_cpu_offload", "model")
        )
        self.depth_seed_var = tk.StringVar(
            value=str(self._config.get("depth_seed", "42"))
        )
        self.depth_guidance_scale_var = tk.StringVar(
            value=str(self._config.get("depth_guidance_scale", "1.0"))
        )
        self.depth_decode_chunk_size_var = tk.StringVar(
            value=str(self._config.get("depth_decode_chunk_size", "2"))
        )
        self.depth_restart_every_var = tk.StringVar(
            value=str(self._config.get("depth_restart_every", "100"))
        )
        self.depth_debug_mem_var = tk.BooleanVar(
            value=bool(self._config.get("depth_debug_mem", True))
        )
        self.depth_glob_var = tk.StringVar(
            value=self._config.get("depth_glob", "*.mp4")
        )
        self.depth_worker_script_var = tk.StringVar(
            value=self._config.get("depth_worker_script", "./depthcrafter_nogui_batch.py")
        )
        depth_scale_factor_raw = self._config.get(
            "depth_scale_factor", self.DEFAULT_DEPTH_SCALE_FACTOR
        )
        try:
            depth_scale_factor_cfg = float(depth_scale_factor_raw)
        except Exception:
            depth_scale_factor_cfg = float(self.DEFAULT_DEPTH_SCALE_FACTOR)
        self.depth_scale_factor_var = tk.DoubleVar(value=depth_scale_factor_cfg)
        self.depth_scale_factor_text_var = tk.StringVar(
            value=f"{depth_scale_factor_cfg:.2f}x"
        )
        self.depth_pad_target_var = tk.StringVar(value="n.d.")
        self.depth_res_x_var = tk.StringVar(
            value=str(self._config.get("depth_res_x", "640"))
        )
        self.depth_res_y_var = tk.StringVar(
            value=str(self._config.get("depth_res_y", "384"))
        )

        self.depth_encode_override_var = tk.BooleanVar(
            value=bool(self._config.get("depth_encode_override", False))
        )
        self.depth_codec_var = tk.StringVar(
            value=self._config.get("depth_codec", self.scene_codec_var.get())
        )
        self.depth_crf_var = tk.StringVar(
            value=str(self._config.get("depth_crf", self.scene_crf_var.get()))
        )
        self.depth_preset_var = tk.StringVar(
            value=self._config.get("depth_preset", self.scene_encoder_preset_var.get())
        )
        self.depth_pix_fmt_var = tk.StringVar(
            value=self._config.get("depth_pix_fmt", self.scene_pix_fmt_var.get())
        )
        self.depth_extra_ffmpeg_args_var = tk.StringVar(
            value=self._config.get("depth_extra_ffmpeg_args", "")
        )
        self.depth_realesrgan_source_var = tk.StringVar(
            value=self._config.get(
                "depth_realesrgan_source", "Bundled (Utilities/realesrgan)"
            )
        )
        self.depth_realesrgan_workers_var = tk.StringVar(
            value=str(
                self._config.get(
                    "depth_realesrgan_workers",
                    str(self.DEFAULT_DEPTH_REALESRGAN_WORKERS),
                )
            )
        )
        self.depth_cmd_preview_var = tk.StringVar(value="")
        self.depth_status_var = tk.StringVar(value="Ready")
        self.depth_progress_var = tk.DoubleVar(value=0.0)

        # Splatting tab (GUI layout + menu controls).
        self.splat_input_clips_var = tk.StringVar(value="")
        self.splat_input_depth_var = tk.StringVar(value="")
        self.splat_output_var = tk.StringVar(value="")
        self.splat_mask_output_var = tk.StringVar(value="")
        self.splat_mode_var = tk.StringVar(
            value=self._config.get("splat_mode", "Auto (recommended)")
        )
        self.splat_info_text_var = tk.StringVar(value=self.SPLAT_AUTO_INFO)

        self.splat_batch_size_var = tk.StringVar(
            value=str(self._config.get("splat_batch_size", "50"))
        )
        self.splat_workers_var = tk.StringVar(
            value=str(self._config.get("splat_workers", "2"))
        )
        self.splat_disparity_var = tk.StringVar(
            value=str(self._config.get("splat_disparity", "20"))
        )
        self.splat_layout_var = tk.StringVar(
            value=self._config.get("splat_layout", "Single Warp")
        )
        self.splat_auto_convergence_var = tk.StringVar(
            value=self._config.get("splat_auto_convergence", "Min Borders")
        )
        self.splat_dilate_x_var = tk.StringVar(
            value=str(self._config.get("splat_dilate_x", "3"))
        )
        self.splat_dilate_y_var = tk.StringVar(
            value=str(self._config.get("splat_dilate_y", "3"))
        )
        self.splat_blur_x_var = tk.StringVar(
            value=str(self._config.get("splat_blur_x", "0"))
        )
        self.splat_blur_y_var = tk.StringVar(
            value=str(self._config.get("splat_blur_y", "0"))
        )
        self.splat_dilate_left_var = tk.StringVar(
            value=str(self._config.get("splat_dilate_left", "2"))
        )
        self.splat_blur_balance_var = tk.StringVar(
            value=str(self._config.get("splat_blur_balance", "0.5"))
        )
        self.splat_gamma_var = tk.StringVar(
            value=str(self._config.get("splat_gamma", "1"))
        )
        self.splat_convergence_var = tk.StringVar(
            value=str(self._config.get("splat_convergence", "50"))
        )
        self.splat_stair_smooth_var = tk.BooleanVar(
            value=bool(self._config.get("splat_stair_smooth", True))
        )
        self.splat_stair_kernel_var = tk.StringVar(
            value=str(self._config.get("splat_stair_kernel", "3"))
        )
        self.splat_stair_x_off_var = tk.StringVar(
            value=str(self._config.get("splat_stair_x_off", "2"))
        )
        self.splat_stair_strip_var = tk.StringVar(
            value=str(self._config.get("splat_stair_strip", "4"))
        )
        self.splat_stair_strength_var = tk.StringVar(
            value=str(self._config.get("splat_stair_strength", "1"))
        )
        self.splat_replace_mask_var = tk.BooleanVar(
            value=bool(self._config.get("splat_replace_mask", True))
        )
        self.splat_replace_mask_scale_var = tk.StringVar(
            value=str(self._config.get("splat_replace_mask_scale", "1"))
        )
        self.splat_replace_mask_min_var = tk.StringVar(
            value=str(self._config.get("splat_replace_mask_min", "1"))
        )
        self.splat_replace_mask_max_var = tk.StringVar(
            value=str(self._config.get("splat_replace_mask_max", "32"))
        )
        self.splat_replace_mask_gap_var = tk.StringVar(
            value=str(self._config.get("splat_replace_mask_gap", "0"))
        )
        self.splat_replace_mask_edge_var = tk.BooleanVar(
            value=bool(self._config.get("splat_replace_mask_edge", False))
        )

        self.splat_encode_override_var = tk.BooleanVar(
            value=bool(self._config.get("splat_encode_override", False))
        )
        self.splat_codec_var = tk.StringVar(
            value=self._config.get("splat_codec", self.scene_codec_var.get())
        )
        self.splat_crf_var = tk.StringVar(
            value=str(self._config.get("splat_crf", self.scene_crf_var.get()))
        )
        self.splat_preset_var = tk.StringVar(
            value=self._config.get("splat_preset", self.scene_encoder_preset_var.get())
        )
        self.splat_pix_fmt_var = tk.StringVar(
            value=self._config.get("splat_pix_fmt", self.scene_pix_fmt_var.get())
        )
        self.splat_extra_ffmpeg_args_var = tk.StringVar(
            value=self._config.get("splat_extra_ffmpeg_args", "")
        )
        self.splat_cmd_preview_var = tk.StringVar(value="")
        self.splat_status_var = tk.StringVar(value="Ready")
        self.splat_progress_var = tk.DoubleVar(value=0.0)

        # Inpainting tab.
        self.inpaint_input_var = tk.StringVar(value="")
        self.inpaint_mask_var = tk.StringVar(value="")
        self.inpaint_output_var = tk.StringVar(value="")
        self.inpaint_mode_var = tk.StringVar(
            value=self._config.get("inpaint_mode", "Auto (recommended)")
        )
        self.inpaint_info_text_var = tk.StringVar(value=self.INPAINT_AUTO_INFO)
        self.inpaint_frames_chunk_var = tk.StringVar(
            value=str(self._config.get("inpaint_frames_chunk", "50"))
        )
        self.inpaint_cpu_offload_var = tk.StringVar(
            value=self._config.get("inpaint_cpu_offload", "model")
        )
        self.inpaint_tile_num_var = tk.StringVar(
            value=str(self._config.get("inpaint_tile_num", "2"))
        )
        self.inpaint_input_bias_var = tk.StringVar(
            value=str(self._config.get("inpaint_input_bias", "0"))
        )
        self.inpaint_overlap_var = tk.StringVar(
            value=str(self._config.get("inpaint_overlap", "3"))
        )
        self.inpaint_tail_pad_var = tk.StringVar(
            value=str(self._config.get("inpaint_tail_pad", "2"))
        )
        self.inpaint_use_sharpness_csv_var = tk.BooleanVar(
            value=bool(self._config.get("inpaint_use_sharpness_csv", True))
        )
        default_sharp_workers = "19"
        self.inpaint_sharpness_workers_var = tk.StringVar(
            value=str(self._config.get("inpaint_sharpness_workers", default_sharp_workers))
        )
        self.inpaint_inference_steps_var = tk.StringVar(
            value=str(self._config.get("inpaint_inference_steps", "8"))
        )
        self.inpaint_encode_override_var = tk.BooleanVar(
            value=bool(self._config.get("inpaint_encode_override", False))
        )
        self.inpaint_codec_var = tk.StringVar(
            value=self._config.get("inpaint_codec", self.scene_codec_var.get())
        )
        self.inpaint_crf_var = tk.StringVar(
            value=str(self._config.get("inpaint_crf", self.scene_crf_var.get()))
        )
        self.inpaint_preset_var = tk.StringVar(
            value=self._config.get("inpaint_preset", self.scene_encoder_preset_var.get())
        )
        self.inpaint_pix_fmt_var = tk.StringVar(
            value=self._config.get("inpaint_pix_fmt", self.scene_pix_fmt_var.get())
        )
        self.inpaint_extra_ffmpeg_args_var = tk.StringVar(
            value=self._config.get("inpaint_extra_ffmpeg_args", "")
        )
        self.inpaint_sharpness_csv_var = tk.StringVar(value="")
        self.inpaint_cmd_preview_var = tk.StringVar(value="")
        self.inpaint_status_var = tk.StringVar(value="Ready")
        self.inpaint_progress_var = tk.DoubleVar(value=0.0)

        # Merging tab.
        self.merge_inpainted_var = tk.StringVar(value="")
        self.merge_splatted_var = tk.StringVar(value="")
        self.merge_original_var = tk.StringVar(value="")
        self.merge_replace_mask_var = tk.StringVar(value="")
        self.merge_mask_formerge_var = tk.StringVar(value="")
        self.merge_output_var = tk.StringVar(value="")
        self.merge_autoct_csv_var = tk.StringVar(value="")
        self.merge_mode_var = tk.StringVar(
            value=self._config.get("merge_mode", "Auto (recommended)")
        )
        self.merge_info_text_var = tk.StringVar(value=self.MERGE_AUTO_INFO)
        self.merge_autoct_workers_var = tk.StringVar(
            value=str(self._config.get("merge_autoct_workers", "8"))
        )
        self.merge_mask_formerge_workers_var = tk.StringVar(
            value=str(
                self._config.get(
                    "merge_mask_formerge_workers",
                    self._config.get("merge_autoct_workers", "8"),
                )
            )
        )
        self.merge_parallel_var = tk.BooleanVar(
            value=bool(self._config.get("merge_parallel", True))
        )
        self.merge_parallel_workers_var = tk.StringVar(
            value=str(self._config.get("merge_parallel_workers", "2"))
        )
        self.merge_use_gpu_var = tk.BooleanVar(
            value=bool(self._config.get("merge_use_gpu", False))
        )
        self.merge_output_format_var = tk.StringVar(
            value=self._config.get("merge_output_format", "Full SBS")
        )
        self.merge_chunks_var = tk.StringVar(
            value=str(self._config.get("merge_chunks", "20"))
        )
        self.merge_mask_binarize_var = tk.StringVar(
            value=str(self._config.get("merge_mask_binarize", "0.5"))
        )
        self.merge_mask_dilate_var = tk.StringVar(
            value=str(self._config.get("merge_mask_dilate", "2"))
        )
        self.merge_mask_blur_var = tk.StringVar(
            value=str(self._config.get("merge_mask_blur", "4"))
        )
        self.merge_shadow_length_var = tk.StringVar(
            value=str(self._config.get("merge_shadow_length", "25"))
        )
        self.merge_shadow_curve_var = tk.StringVar(
            value=str(self._config.get("merge_shadow_curve", "0"))
        )
        _legacy_merge_shadow_motion_gain = str(
            self._config.get("merge_shadow_motion_gain", "1")
        ).strip()
        try:
            _legacy_merge_shadow_motion_enabled = float(_legacy_merge_shadow_motion_gain) > 0.0
        except Exception:
            _legacy_merge_shadow_motion_enabled = (
                _legacy_merge_shadow_motion_gain.lower() not in {"0", "false", "no", "off"}
            )
        _merge_shadow_motion_enabled_cfg = self._config.get(
            "merge_shadow_motion_enabled",
            _legacy_merge_shadow_motion_enabled,
        )
        if isinstance(_merge_shadow_motion_enabled_cfg, str):
            _merge_shadow_motion_enabled = (
                _merge_shadow_motion_enabled_cfg.strip().lower()
                in {"1", "true", "yes", "on"}
            )
        else:
            _merge_shadow_motion_enabled = bool(_merge_shadow_motion_enabled_cfg)
        self.merge_shadow_motion_enabled_var = tk.BooleanVar(
            value=_merge_shadow_motion_enabled
        )
        self.merge_dynamic_shadow_width_var = tk.BooleanVar(
            value=bool(self._config.get("merge_dynamic_shadow_width", True))
        )
        self.merge_use_replace_mask_var = tk.BooleanVar(
            value=bool(self._config.get("merge_use_replace_mask", True))
        )
        self.merge_ct_preset_var = tk.StringVar(
            value=str(self._config.get("merge_ct_preset", "1"))
        )
        self.merge_ct_auto_mode_var = tk.StringVar(
            value=self._config.get("merge_ct_auto_mode", "CSV Blend")
        )
        self.merge_ct_exclude_black_var = tk.BooleanVar(
            value=bool(self._config.get("merge_ct_exclude_black", True))
        )
        self.merge_encode_override_var = tk.BooleanVar(
            value=bool(self._config.get("merge_encode_override", False))
        )
        self.merge_codec_var = tk.StringVar(
            value=self._config.get("merge_codec", self.scene_codec_var.get())
        )
        self.merge_crf_var = tk.StringVar(
            value=str(self._config.get("merge_crf", self.scene_crf_var.get()))
        )
        self.merge_preset_var = tk.StringVar(
            value=self._config.get("merge_preset", self.scene_encoder_preset_var.get())
        )
        self.merge_pix_fmt_var = tk.StringVar(
            value=self._config.get("merge_pix_fmt", self.scene_pix_fmt_var.get())
        )
        self.merge_extra_ffmpeg_args_var = tk.StringVar(
            value=self._config.get("merge_extra_ffmpeg_args", "")
        )
        self.merge_cmd_preview_var = tk.StringVar(value="")
        self.merge_status_var = tk.StringVar(value="Ready")
        self.merge_progress_var = tk.DoubleVar(value=0.0)

        # Joining tab.
        self.join_input_var = tk.StringVar(value="")
        self.join_seg_mono_var = tk.StringVar(value="")
        self.join_output_var = tk.StringVar(value="")
        self.join_mode_var = tk.StringVar(
            value=self._config.get("join_mode", "Auto (recommended)")
        )
        self.join_info_text_var = tk.StringVar(value=self.JOIN_AUTO_INFO)
        self.join_encoder_var = tk.StringVar(
            value=self._config.get("join_encoder", "hevc_nvenc")
        )
        self.join_crf_var = tk.StringVar(
            value=str(self._config.get("join_crf", "16"))
        )
        self.join_preset_var = tk.StringVar(
            value=self._config.get("join_preset", "p7")
        )
        self.join_pix_fmt_override_var = tk.BooleanVar(
            value=bool(self._config.get("join_pix_fmt_override", False))
        )
        self.join_pix_fmt_var = tk.StringVar(
            value=self._config.get("join_pix_fmt", self.scene_pix_fmt_var.get())
        )
        self.join_extra_args_var = tk.StringVar(
            value=self._config.get("join_extra_args", self.JOIN_DEFAULT_ARGS)
        )
        self.join_cmd_preview_var = tk.StringVar(value="")
        self.join_status_var = tk.StringVar(value="Ready")
        self.join_progress_var = tk.DoubleVar(value=0.0)

        # Options and Run.
        default_verify_workers = "19"
        self.verify_scenes_workers_var = tk.StringVar(
            value=str(self._config.get("verify_scenes_workers", default_verify_workers))
        )
        self.pipeline_verify_after_var = tk.StringVar(
            value=self._config.get("pipeline_verify_after", "Quick")
        )
        self.pipeline_test_run_files_var = tk.StringVar(
            value=str(
                self._config.get(
                    "pipeline_test_run_files",
                    str(self.DEFAULT_PIPELINE_TEST_RUN_FILES),
                )
            )
        )
        depth_retry_cfg = self._retry_policy_from_config_key("depth_retry_policy")
        inpaint_retry_cfg = self._retry_policy_from_config_key("inpaint_retry_policy")
        self.depth_retry_policy_vars: dict[str, dict[str, tk.Variable]] = {}
        self.inpaint_retry_policy_vars: dict[str, dict[str, tk.Variable]] = {}
        for profile in self.RETRY_POLICY_PROFILES:
            dcfg = depth_retry_cfg.get(profile, self.RETRY_POLICY_DEFAULT[profile])
            icfg = inpaint_retry_cfg.get(profile, self.RETRY_POLICY_DEFAULT[profile])
            self.depth_retry_policy_vars[profile] = {
                "garbage_collection_threshold": tk.BooleanVar(
                    value=bool(dcfg.get("garbage_collection_threshold", True))
                ),
                "expandable_segments": tk.BooleanVar(
                    value=bool(dcfg.get("expandable_segments", True))
                ),
                "max_split_size_mb": tk.StringVar(
                    value=self._normalize_retry_max_split(dcfg.get("max_split_size_mb", "off"))
                ),
                "cpu_offload_inherited": tk.BooleanVar(
                    value=bool(dcfg.get("cpu_offload_inherited", True))
                ),
                "cpu_offload_mode": tk.StringVar(
                    value=self._normalize_retry_offload_mode(dcfg.get("cpu_offload_mode", "model"))
                ),
            }
            self.inpaint_retry_policy_vars[profile] = {
                "garbage_collection_threshold": tk.BooleanVar(
                    value=bool(icfg.get("garbage_collection_threshold", True))
                ),
                "expandable_segments": tk.BooleanVar(
                    value=bool(icfg.get("expandable_segments", True))
                ),
                "max_split_size_mb": tk.StringVar(
                    value=self._normalize_retry_max_split(icfg.get("max_split_size_mb", "off"))
                ),
                "cpu_offload_inherited": tk.BooleanVar(
                    value=bool(icfg.get("cpu_offload_inherited", True))
                ),
                "cpu_offload_mode": tk.StringVar(
                    value=self._normalize_retry_offload_mode(icfg.get("cpu_offload_mode", "model"))
                ),
            }
        self._depth_retry_offload_widgets: dict[str, ttk.Combobox] = {}
        self._inpaint_retry_offload_widgets: dict[str, ttk.Combobox] = {}
        self.pipeline_run_status_var = tk.StringVar(value="Idle")
        self.pipeline_run_progress_var = tk.DoubleVar(value=0.0)
        self.pipeline_checked_files_var = tk.StringVar(value="Check Files: not run")

        self.resume_enabled_var = tk.BooleanVar(
            value=bool(self._config.get("resume_enabled", True))
        )
        self.stop_on_error_var = tk.BooleanVar(
            value=bool(self._config.get("stop_on_error", True))
        )
        self.auto_advance_var = tk.BooleanVar(
            value=bool(self._config.get("auto_advance", False))
        )

        # Normalize legacy config values to current defaults.
        if self.scene_layout_var.get().strip() == "Half-SBS final-only":
            self.scene_layout_var.set("Full-SBS (quality)")
        if self.scene_backend_var.get().strip() not in {"MoviePy", "OpenCV"}:
            self.scene_backend_var.set(self.DEFAULT_SCENE_BACKEND)
        if self.scene_codec_var.get().strip() == "":
            self.scene_codec_var.set(self.DEFAULT_SCENE_CODEC)
        self.scene_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.scene_codec_var.get(),
                self.DEFAULT_SCENE_CODEC,
            )
        )
        if self.scene_crf_var.get().strip() in {"", "0"}:
            self.scene_crf_var.set("1")
        if self.scene_encoder_preset_var.get().strip().lower() in {"", "veryfast"}:
            self.scene_encoder_preset_var.set("fast")
        if self.depth_chunk_size_var.get().strip() in {"", "110"}:
            self.depth_chunk_size_var.set("70")
        if self.depth_overlap_var.get().strip() in {"", "25"}:
            self.depth_overlap_var.set("20")
        if self.depth_inference_steps_var.get().strip() == "":
            self.depth_inference_steps_var.set("5")
        self.depth_scale_factor_var.set(
            self._normalize_depth_scale_factor(self.depth_scale_factor_var.get())
        )
        self.depth_scale_factor_text_var.set(
            f"{float(self.depth_scale_factor_var.get()):.2f}x"
        )
        if self.depth_realesrgan_source_var.get().strip() not in {
            "Bundled (Utilities/realesrgan)",
            "Local (system/custom path)",
        }:
            self.depth_realesrgan_source_var.set("Bundled (Utilities/realesrgan)")
        try:
            if int(self.depth_realesrgan_workers_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.depth_realesrgan_workers_var.set(str(self.DEFAULT_DEPTH_REALESRGAN_WORKERS))
        if self.splat_mode_var.get().strip() not in {"Auto (recommended)", "Manual"}:
            self.splat_mode_var.set("Auto (recommended)")
        try:
            if int(self.splat_workers_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.splat_workers_var.set("2")
        if self.splat_layout_var.get().strip() not in {"Single Warp", "Dual", "Quad"}:
            self.splat_layout_var.set("Single Warp")
        if self.splat_auto_convergence_var.get().strip() not in {"Min Borders", "Off"}:
            self.splat_auto_convergence_var.set("Min Borders")
        if self.inpaint_mode_var.get().strip() not in {"Auto (recommended)", "Manual"}:
            self.inpaint_mode_var.set("Auto (recommended)")
        if self.inpaint_frames_chunk_var.get().strip() == "":
            self.inpaint_frames_chunk_var.set("50")
        if self.inpaint_cpu_offload_var.get().strip() == "":
            self.inpaint_cpu_offload_var.set("model")
        if self.inpaint_inference_steps_var.get().strip() == "":
            self.inpaint_inference_steps_var.set("8")
        self.depth_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.depth_codec_var.get(),
                self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
            )
        )
        self.splat_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.splat_codec_var.get(),
                self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
            )
        )
        if self.inpaint_crf_var.get().strip() == "":
            self.inpaint_crf_var.set(self.scene_crf_var.get().strip() or "1")
        if self.inpaint_codec_var.get().strip() == "":
            self.inpaint_codec_var.set(
                self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC
            )
        self.inpaint_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.inpaint_codec_var.get(),
                self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
            )
        )
        if self.inpaint_preset_var.get().strip() == "":
            self.inpaint_preset_var.set(self.scene_encoder_preset_var.get().strip() or "fast")
        if self.inpaint_pix_fmt_var.get().strip() == "":
            self.inpaint_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip() or "yuv420p")
        try:
            if int(self.inpaint_sharpness_workers_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.inpaint_sharpness_workers_var.set("19")
        if self.merge_mode_var.get().strip() not in {"Auto (recommended)", "Manual"}:
            self.merge_mode_var.set("Auto (recommended)")
        if self.merge_output_format_var.get().strip() not in {"Full SBS", "Half SBS"}:
            self.merge_output_format_var.set("Full SBS")
        if self.merge_ct_auto_mode_var.get().strip() not in {"CSV Blend", "On", "Off"}:
            self.merge_ct_auto_mode_var.set("CSV Blend")
        if self.merge_ct_preset_var.get().strip() == "":
            self.merge_ct_preset_var.set("1")
        if self.merge_autoct_workers_var.get().strip() == "":
            self.merge_autoct_workers_var.set("8")
        if self.merge_mask_formerge_workers_var.get().strip() == "":
            self.merge_mask_formerge_workers_var.set(
                self.merge_autoct_workers_var.get().strip() or "8"
            )
        if self.merge_parallel_workers_var.get().strip() == "":
            self.merge_parallel_workers_var.set("2")
        if self.merge_chunks_var.get().strip() == "":
            self.merge_chunks_var.set("20")
        if self.merge_mask_binarize_var.get().strip() == "":
            self.merge_mask_binarize_var.set("0.5")
        if self.merge_mask_dilate_var.get().strip() == "":
            self.merge_mask_dilate_var.set("2")
        if self.merge_mask_blur_var.get().strip() == "":
            self.merge_mask_blur_var.set("4")
        if self.merge_shadow_length_var.get().strip() == "":
            self.merge_shadow_length_var.set("25")
        if self.merge_shadow_curve_var.get().strip() == "":
            self.merge_shadow_curve_var.set("0")
        if not bool(self.splat_replace_mask_var.get()):
            self.splat_replace_mask_var.set(True)
        if not bool(self.merge_use_replace_mask_var.get()):
            self.merge_use_replace_mask_var.set(True)
        if self.merge_crf_var.get().strip() == "":
            self.merge_crf_var.set(self.scene_crf_var.get().strip() or "1")
        if self.merge_codec_var.get().strip() == "":
            self.merge_codec_var.set(
                self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC
            )
        self.merge_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.merge_codec_var.get(),
                self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
            )
        )
        if self.merge_preset_var.get().strip() == "":
            self.merge_preset_var.set(self.scene_encoder_preset_var.get().strip() or "fast")
        if self.merge_pix_fmt_var.get().strip() == "":
            self.merge_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip() or "yuv420p")
        if self.join_mode_var.get().strip() not in {"Auto (recommended)", "Manual"}:
            self.join_mode_var.set("Auto (recommended)")
        if self.join_encoder_var.get().strip() == "":
            self.join_encoder_var.set("hevc_nvenc")
        self.join_encoder_var.set(
            self._normalize_ffmpeg_codec(
                self.join_encoder_var.get(),
                "hevc_nvenc",
            )
        )
        if self.join_crf_var.get().strip() == "":
            self.join_crf_var.set("16")
        if self.join_preset_var.get().strip() == "":
            self.join_preset_var.set("p7")
        if self.join_pix_fmt_var.get().strip() == "":
            self.join_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip() or "yuv420p")
        if self.join_extra_args_var.get().strip() == "":
            self.join_extra_args_var.set(self.JOIN_DEFAULT_ARGS)
        try:
            if int(self.verify_scenes_workers_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.verify_scenes_workers_var.set("19")
        try:
            if int(self.scene_split_threads_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.scene_split_threads_var.set(str(self.DEFAULT_SPLIT_SCENES_WORKERS))
        try:
            if int(self.pipeline_test_run_files_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.pipeline_test_run_files_var.set(str(self.DEFAULT_PIPELINE_TEST_RUN_FILES))
        if self.pipeline_verify_after_var.get().strip() not in self.PIPELINE_VERIFY_CHOICES:
            self.pipeline_verify_after_var.set("Quick")

        # Keep pix_fmt aligned to chroma at startup.
        self.scene_pix_fmt_var.set(self._chroma_to_pixfmt(self.scene_chroma_var.get().strip()))
        if not self.depth_encode_override_var.get():
            self._sync_depth_encoding_from_scene()
        if not self.splat_encode_override_var.get():
            self._sync_splat_encoding_from_scene()
        if not self.inpaint_encode_override_var.get():
            self._sync_inpaint_encoding_from_scene()
        if not self.merge_encode_override_var.get():
            self._sync_merge_encoding_from_scene()
        if not self.join_pix_fmt_override_var.get():
            self._sync_join_encoding_from_scene()

        # Live inherit scene encoding values into depth (when override is off).
        self.scene_codec_var.trace_add("write", self._on_scene_encode_var_changed)
        self.scene_crf_var.trace_add("write", self._on_scene_encode_var_changed)
        self.scene_encoder_preset_var.trace_add("write", self._on_scene_encode_var_changed)
        self.scene_pix_fmt_var.trace_add("write", self._on_scene_encode_var_changed)
        self.scene_crop_target_h_var.trace_add("write", self._on_scene_crop_target_changed)
        self.depth_scale_factor_var.trace_add("write", self._on_depth_scale_factor_changed)

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.scene_tab = ttk.Frame(notebook, padding=10)
        self.depth_tab = ttk.Frame(notebook, padding=10)
        self.splat_tab = ttk.Frame(notebook, padding=10)
        self.inpaint_tab = ttk.Frame(notebook, padding=10)
        self.merge_tab = ttk.Frame(notebook, padding=10)
        self.join_tab = ttk.Frame(notebook, padding=10)
        self.options_tab = ttk.Frame(notebook, padding=10)

        notebook.add(self.scene_tab, text="SceneDetect")
        notebook.add(self.depth_tab, text="DepthCrafter")
        notebook.add(self.splat_tab, text="Splatting")
        notebook.add(self.inpaint_tab, text="Inpainting")
        notebook.add(self.merge_tab, text="Merging")
        notebook.add(self.join_tab, text="Joining")
        notebook.add(self.options_tab, text="Options and Run")

        self._build_scene_tab(self.scene_tab)
        self._build_depth_tab(self.depth_tab)
        self._build_splat_tab(self.splat_tab)
        self._build_inpaint_tab(self.inpaint_tab)
        self._build_merge_tab(self.merge_tab)
        self._build_join_tab(self.join_tab)
        self._build_options_tab(self.options_tab)

    def _build_scene_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(10, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="Work folder:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.work_folder_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Browse...", command=self._browse_work_folder).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(parent, text="Source video:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.scene_input_var).grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Browse...", command=self._browse_scene_input).grid(
            row=1, column=2, padx=4
        )

        ttk.Label(parent, text="Scene output (auto):").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.scene_output_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_scene_output_folder).grid(
            row=2, column=2, padx=4
        )

        options_frame = ttk.LabelFrame(parent, text="SceneDetect Options", padding=8)
        options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
        options_frame.grid_columnconfigure(7, weight=1)

        ttk.Label(options_frame, text="Detector:").grid(row=0, column=0, sticky="w")
        self.scene_detector_combo = ttk.Combobox(
            options_frame,
            textvariable=self.scene_detector_var,
            values=["Adaptive", "Content"],
            width=12,
            state="readonly",
        )
        self.scene_detector_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.scene_detector_combo.bind("<<ComboboxSelected>>", self._on_detector_changed)

        ttk.Label(options_frame, text="Threshold:").grid(row=0, column=2, sticky="w")
        ttk.Entry(options_frame, textvariable=self.scene_threshold_var, width=8).grid(
            row=0, column=3, sticky="w", padx=(6, 12)
        )

        ttk.Label(options_frame, text="Read backend:").grid(row=0, column=4, sticky="w")
        self.scene_backend_combo = ttk.Combobox(
            options_frame,
            textvariable=self.scene_backend_var,
            values=["MoviePy", "OpenCV"],
            width=12,
            state="readonly",
        )
        self.scene_backend_combo.grid(row=0, column=5, sticky="w", padx=(6, 12))
        self.scene_backend_combo.bind("<<ComboboxSelected>>", self._on_backend_changed)

        ttk.Label(options_frame, text="Split threads:").grid(row=0, column=6, sticky="w")
        ttk.Entry(options_frame, textvariable=self.scene_split_threads_var, width=6).grid(
            row=0, column=7, sticky="w", padx=(6, 0)
        )

        ttk.Label(options_frame, text="Auto crop:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.crop_mode_auto_toggle = ttk.Checkbutton(
            options_frame,
            text="Enabled (recommended)",
            variable=self.scene_crop_mode_var,
            onvalue="auto",
            offvalue="manual",
            command=self._on_crop_mode_changed,
        )
        self.crop_mode_auto_toggle.grid(row=1, column=1, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(options_frame, textvariable=self.scene_crop_auto_desc_var).grid(
            row=1, column=4, columnspan=4, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(options_frame, text="Auto target H (final, step 8):").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.scene_crop_target_spin = ttk.Spinbox(
            options_frame,
            from_=8,
            to=4320,
            increment=8,
            width=8,
            textvariable=self.scene_crop_target_h_var,
            command=self._on_scene_crop_target_spin,
        )
        self.scene_crop_target_spin.grid(row=2, column=1, sticky="w", padx=(6, 12), pady=(6, 0))
        ttk.Label(options_frame, textvariable=self.scene_crop_tile_compat_var).grid(
            row=2, column=2, columnspan=6, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        policy_frame = ttk.LabelFrame(parent, text="Source-Driven Menu", padding=8)
        policy_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        policy_frame.grid_columnconfigure(5, weight=1)

        ttk.Label(policy_frame, text="Pipeline layout:").grid(row=0, column=0, sticky="w")
        self.scene_layout_combo = ttk.Combobox(
            policy_frame,
            textvariable=self.scene_layout_var,
            values=["Full-SBS (quality)", "Half-SBS early (fast)"],
            width=22,
            state="readonly",
        )
        self.scene_layout_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.scene_layout_combo.bind("<<ComboboxSelected>>", self._on_layout_changed)

        ttk.Label(policy_frame, text="Tonemap:").grid(row=0, column=2, sticky="w")
        self.scene_tonemap_combo = ttk.Combobox(
            policy_frame,
            textvariable=self.scene_tonemap_var,
            values=list(self.TONEMAP_PRESET_TO_FFMPEG.keys()),
            width=62,
            state="disabled",
        )
        self.scene_tonemap_combo.grid(row=0, column=3, columnspan=3, sticky="w", padx=(6, 0))
        self.scene_tonemap_combo.bind("<<ComboboxSelected>>", self._on_tonemap_changed)

        ttk.Label(policy_frame, textvariable=self.hdr_policy_var).grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )

        ttk.Label(policy_frame, text="Chroma target:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        chroma_row = ttk.Frame(policy_frame)
        chroma_row.grid(row=2, column=1, columnspan=4, sticky="w", pady=(8, 0))
        self.chroma_444_rb = ttk.Radiobutton(
            chroma_row,
            text="yuv444p",
            variable=self.scene_chroma_var,
            value="444",
            command=self._on_chroma_changed,
        )
        self.chroma_444_rb.grid(row=0, column=0, sticky="w", padx=(0, 18))
        self.chroma_422_rb = ttk.Radiobutton(
            chroma_row,
            text="yuv422p",
            variable=self.scene_chroma_var,
            value="422",
            command=self._on_chroma_changed,
        )
        self.chroma_422_rb.grid(row=0, column=1, sticky="w", padx=(0, 18))
        self.chroma_420_rb = ttk.Radiobutton(
            chroma_row,
            text="yuv420p",
            variable=self.scene_chroma_var,
            value="420",
            command=self._on_chroma_changed,
        )
        self.chroma_420_rb.grid(row=0, column=2, sticky="w")

        ttk.Label(policy_frame, textvariable=self.scene_option_hint_var).grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )

        ffmpeg_frame = ttk.LabelFrame(parent, text="Split Encoding Args", padding=8)
        ffmpeg_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        ffmpeg_frame.grid_columnconfigure(9, weight=1)

        ttk.Label(ffmpeg_frame, text="Codec:").grid(row=0, column=0, sticky="w")
        self.scene_codec_combo = ttk.Combobox(
            ffmpeg_frame,
            textvariable=self.scene_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.scene_codec_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.scene_codec_combo.bind("<<ComboboxSelected>>", lambda _e: self._preview_scene_command())
        ttk.Label(ffmpeg_frame, text="Quality (CRF/QP):").grid(row=0, column=2, sticky="w")
        ttk.Entry(ffmpeg_frame, textvariable=self.scene_crf_var, width=6).grid(
            row=0, column=3, sticky="w", padx=(6, 12)
        )
        ttk.Label(ffmpeg_frame, text="Preset:").grid(row=0, column=4, sticky="w")
        ttk.Entry(ffmpeg_frame, textvariable=self.scene_encoder_preset_var, width=10).grid(
            row=0, column=5, sticky="w", padx=(6, 12)
        )
        ttk.Label(ffmpeg_frame, text="PixFmt (menu-driven):").grid(row=0, column=6, sticky="w")
        ttk.Entry(ffmpeg_frame, textvariable=self.scene_pix_fmt_var, width=10, state="readonly").grid(
            row=0, column=7, sticky="w", padx=(6, 12)
        )

        ttk.Label(ffmpeg_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(ffmpeg_frame, textvariable=self.scene_extra_ffmpeg_args_var).grid(
            row=1, column=1, columnspan=9, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(
            cmd_frame, textvariable=self.scene_cmd_preview_var, state="readonly"
        ).grid(row=0, column=0, sticky="ew")

        buttons = ttk.Frame(parent)
        buttons.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.scene_analyze_btn = ttk.Button(
            buttons, text="Analyze Source Video", command=self._start_source_analysis
        )
        self.scene_analyze_btn.grid(row=0, column=0, padx=(0, 6))
        self.scene_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_scene_command
        )
        self.scene_preview_btn.grid(row=0, column=1, padx=(0, 6))
        self.scene_run_btn = ttk.Button(
            buttons, text="Run SceneDetect", command=self._start_scene_detect
        )
        self.scene_run_btn.grid(row=0, column=2, padx=6)
        self.scene_split_btn = ttk.Button(
            buttons, text="Split Scenes", command=self._start_split_scenes
        )
        self.scene_split_btn.grid(row=0, column=3, padx=6)
        self.scene_verify_quick_btn = ttk.Button(
            buttons, text="Verify Scenes (Quick)", command=self._start_verify_quick
        )
        self.scene_verify_quick_btn.grid(row=0, column=4, padx=6)
        self.scene_verify_deep_btn = ttk.Button(
            buttons, text="Verify Scenes (Deep)", command=self._start_verify_deep
        )
        self.scene_verify_deep_btn.grid(row=0, column=5, padx=6)
        self.scene_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_scene_detect, state=tk.DISABLED
        )
        self.scene_stop_btn.grid(row=0, column=6, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_scene_log).grid(
            row=0, column=7, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=8, column=0, columnspan=3, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(3, weight=1)
        status_frame.grid_columnconfigure(4, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.scene_status_var).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        ttk.Label(status_frame, text="Analyze:").grid(row=0, column=2, sticky="w")
        ttk.Label(status_frame, textvariable=self.scene_analysis_status_var).grid(
            row=0, column=3, sticky="w", padx=(6, 12)
        )
        self.scene_progress = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.scene_progress_var,
            maximum=100.0,
        )
        self.scene_progress.grid(row=0, column=4, sticky="ew", padx=4)

        analysis_frame = ttk.LabelFrame(parent, text="Source Analysis", padding=8)
        analysis_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(4, 6))
        for i in range(8):
            analysis_frame.grid_columnconfigure(i, weight=1 if i % 2 == 1 else 0)

        fields_row1 = [
            ("Total resolution:", self.analysis_resolution_var),
            ("Top/bottom bars:", self.analysis_bars_var),
            ("Color depth/range:", self.analysis_color_var),
            ("Pixel format/chroma:", self.analysis_pixfmt_var),
        ]
        fields_row2 = [
            ("Length:", self.analysis_length_var),
            ("FPS:", self.analysis_fps_var),
            ("Video bitrate:", self.analysis_bitrate_var),
        ]

        for idx, (label, var) in enumerate(fields_row1):
            c = idx * 2
            ttk.Label(analysis_frame, text=label).grid(row=0, column=c, sticky="w", pady=2)
            ttk.Entry(analysis_frame, textvariable=var, state="readonly").grid(
                row=0, column=c + 1, sticky="ew", padx=(6, 12), pady=2
            )

        for idx, (label, var) in enumerate(fields_row2):
            c = idx * 2
            ttk.Label(analysis_frame, text=label).grid(row=1, column=c, sticky="w", pady=2)
            ttk.Entry(analysis_frame, textvariable=var, state="readonly").grid(
                row=1, column=c + 1, sticky="ew", padx=(6, 12), pady=2
            )

        log_frame = ttk.LabelFrame(parent, text="SceneDetect Log", padding=6)
        log_frame.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.scene_log_text = tk.Text(log_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.scene_log_text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.scene_log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.scene_log_text.configure(yscrollcommand=yscroll.set)

    def _build_depth_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(11, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(
            parent,
            text=self.DEPTH_SEG_MONO_NOTICE,
            justify="left",
            wraplength=1100,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(parent, text="Depth input scenes (auto):").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.depth_input_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_depth_input_folder).grid(
            row=1, column=2, padx=4
        )

        ttk.Label(parent, text="Depth output folder (auto):").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.depth_output_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_depth_output_folder).grid(
            row=2, column=2, padx=4
        )

        ttk.Label(parent, text="Depth upscaled folder (auto):").grid(
            row=3, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.depth_upscaled_var, state="readonly").grid(
            row=3, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_depth_upscaled_folder).grid(
            row=3, column=2, padx=4
        )

        mode_frame = ttk.LabelFrame(parent, text="Depth Mode", padding=8)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        mode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(mode_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.depth_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.depth_mode_var,
            values=["Auto (recommended)", "Manual"],
            width=18,
            state="readonly",
        )
        self.depth_mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.depth_mode_combo.bind("<<ComboboxSelected>>", self._on_depth_mode_changed)

        ttk.Label(
            mode_frame,
            textvariable=self.depth_info_text_var,
            justify="left",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params_frame = ttk.LabelFrame(parent, text="Depth Parameters", padding=8)
        params_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        for col in range(8):
            params_frame.grid_columnconfigure(col, weight=0)
        params_frame.grid_columnconfigure(7, weight=1)

        ttk.Label(params_frame, text="Chunk size:").grid(row=0, column=0, sticky="w")
        self.depth_chunk_size_entry = ttk.Entry(
            params_frame, textvariable=self.depth_chunk_size_var, width=8
        )
        self.depth_chunk_size_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Overlap:").grid(row=0, column=2, sticky="w")
        self.depth_overlap_entry = ttk.Entry(
            params_frame, textvariable=self.depth_overlap_var, width=8
        )
        self.depth_overlap_entry.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Inference steps:").grid(row=0, column=4, sticky="w")
        self.depth_inference_steps_entry = ttk.Entry(
            params_frame, textvariable=self.depth_inference_steps_var, width=8
        )
        self.depth_inference_steps_entry.grid(row=0, column=5, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="CPU offload:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.depth_cpu_offload_combo = ttk.Combobox(
            params_frame,
            textvariable=self.depth_cpu_offload_var,
            values=["none", "model", "sequential"],
            width=12,
            state="readonly",
        )
        self.depth_cpu_offload_combo.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Seed:").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.depth_seed_entry = ttk.Entry(params_frame, textvariable=self.depth_seed_var, width=8)
        self.depth_seed_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Resolution X:").grid(row=1, column=4, sticky="w", pady=(8, 0))
        self.depth_res_x_entry = ttk.Entry(params_frame, textvariable=self.depth_res_x_var, width=8)
        self.depth_res_x_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Resolution Y:").grid(row=1, column=6, sticky="w", pady=(8, 0))
        self.depth_res_y_entry = ttk.Entry(params_frame, textvariable=self.depth_res_y_var, width=8)
        self.depth_res_y_entry.grid(row=1, column=7, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(params_frame, text="Scale factor:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.depth_scale_factor_scale = ttk.Scale(
            params_frame,
            from_=self.MIN_DEPTH_SCALE_FACTOR,
            to=self.MAX_DEPTH_SCALE_FACTOR,
            variable=self.depth_scale_factor_var,
            orient=tk.HORIZONTAL,
            command=self._on_depth_scale_slider_moved,
            length=220,
        )
        self.depth_scale_factor_scale.grid(
            row=2, column=1, columnspan=3, sticky="w", padx=(6, 12), pady=(8, 0)
        )
        ttk.Label(params_frame, textvariable=self.depth_scale_factor_text_var, width=7).grid(
            row=2, column=4, sticky="w", pady=(8, 0)
        )
        ttk.Label(params_frame, text="Pad target:").grid(row=2, column=5, sticky="w", pady=(8, 0))
        ttk.Label(params_frame, textvariable=self.depth_pad_target_var).grid(
            row=2, column=6, columnspan=2, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(params_frame, text="ESRGAN workers:").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self.depth_realesrgan_workers_entry = ttk.Entry(
            params_frame, textvariable=self.depth_realesrgan_workers_var, width=8
        )
        self.depth_realesrgan_workers_entry.grid(
            row=3, column=1, sticky="w", padx=(6, 12), pady=(8, 0)
        )
        ttk.Label(
            params_frame,
            text="Used by Run ESRGAN (editable in Auto mode).",
        ).grid(row=3, column=2, columnspan=6, sticky="w", pady=(8, 0))

        encode_frame = ttk.LabelFrame(parent, text="Encoding Args (inherited)", padding=8)
        encode_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(9, weight=1)

        self.depth_override_check = ttk.Checkbutton(
            encode_frame,
            text="Override",
            variable=self.depth_encode_override_var,
            command=self._on_depth_override_toggle,
        )
        self.depth_override_check.grid(row=0, column=0, sticky="w")

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.depth_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.depth_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.depth_codec_entry.grid(row=0, column=2, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Quality (CRF/QP):").grid(row=0, column=3, sticky="w")
        self.depth_crf_entry = ttk.Entry(encode_frame, textvariable=self.depth_crf_var, width=6)
        self.depth_crf_entry.grid(row=0, column=4, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Preset:").grid(row=0, column=5, sticky="w")
        self.depth_preset_entry = ttk.Entry(encode_frame, textvariable=self.depth_preset_var, width=10)
        self.depth_preset_entry.grid(row=0, column=6, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="PixFmt:").grid(row=0, column=7, sticky="w")
        self.depth_pixfmt_entry = ttk.Entry(encode_frame, textvariable=self.depth_pix_fmt_var, width=10)
        self.depth_pixfmt_entry.grid(row=0, column=8, sticky="w", padx=(6, 0))

        ttk.Label(encode_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.depth_extra_ffmpeg_entry = ttk.Entry(
            encode_frame, textvariable=self.depth_extra_ffmpeg_args_var
        )
        self.depth_extra_ffmpeg_entry.grid(
            row=1, column=1, columnspan=9, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.depth_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.depth_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_depth_command
        )
        self.depth_preview_btn.grid(row=0, column=0, padx=(0, 6))
        self.depth_run_btn = ttk.Button(
            buttons, text="Run DepthCrafter", command=self._run_depth_placeholder
        )
        self.depth_run_btn.grid(row=0, column=1, padx=6)
        self.depth_upscale_btn = ttk.Button(
            buttons, text="Run ESRGAN", command=self._run_depth_upscale_placeholder
        )
        self.depth_upscale_btn.grid(row=0, column=2, padx=6)
        self.depth_verify_quick_btn = ttk.Button(
            buttons, text="Verify Depth (Quick)", command=self._start_depth_verify_quick
        )
        self.depth_verify_quick_btn.grid(row=0, column=3, padx=6)
        self.depth_verify_deep_btn = ttk.Button(
            buttons, text="Verify Depth (Deep)", command=self._start_depth_verify_deep
        )
        self.depth_verify_deep_btn.grid(row=0, column=4, padx=6)
        self.depth_upscaled_verify_quick_btn = ttk.Button(
            buttons, text="Verify Upscale (Quick)", command=self._start_depth_upscaled_verify_quick
        )
        self.depth_upscaled_verify_quick_btn.grid(row=0, column=5, padx=6)
        self.depth_upscaled_verify_deep_btn = ttk.Button(
            buttons, text="Verify Upscale (Deep)", command=self._start_depth_upscaled_verify_deep
        )
        self.depth_upscaled_verify_deep_btn.grid(row=0, column=6, padx=6)
        self.depth_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_depth_placeholder
        )
        self.depth_stop_btn.grid(row=0, column=7, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_depth_log).grid(
            row=0, column=8, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=9, column=0, columnspan=3, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(2, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.depth_status_var).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        self.depth_progress = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.depth_progress_var,
            maximum=100.0,
        )
        self.depth_progress.grid(row=0, column=2, sticky="ew", padx=4)

        log_frame = ttk.LabelFrame(parent, text="DepthCrafter Log", padding=6)
        log_frame.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.depth_log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.depth_log_text.grid(row=0, column=0, sticky="nsew")
        dscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.depth_log_text.yview)
        dscroll.grid(row=0, column=1, sticky="ns")
        self.depth_log_text.configure(yscrollcommand=dscroll.set)

        self._on_depth_mode_changed()
        self._on_depth_override_toggle(initial=True)
        self._preview_depth_command()
        self._set_depth_running(False)

    def _build_splat_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(10, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="Splat input scenes (auto):").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.splat_input_clips_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_splat_input_clips_folder).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(parent, text="Splat input depth (auto):").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.splat_input_depth_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_splat_input_depth_folder).grid(
            row=1, column=2, padx=4
        )

        ttk.Label(parent, text="Splat output folder (auto):").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.splat_output_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_splat_output_folder).grid(
            row=2, column=2, padx=4
        )

        ttk.Label(parent, text="Mask output folder (auto):").grid(
            row=3, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.splat_mask_output_var, state="readonly").grid(
            row=3, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_splat_mask_output_folder).grid(
            row=3, column=2, padx=4
        )

        mode_frame = ttk.LabelFrame(parent, text="Splatting Mode", padding=8)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        mode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(mode_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.splat_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.splat_mode_var,
            values=["Auto (recommended)", "Manual"],
            width=18,
            state="readonly",
        )
        self.splat_mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.splat_mode_combo.bind("<<ComboboxSelected>>", self._on_splat_mode_changed)

        ttk.Label(
            mode_frame,
            text=self.SPLAT_DISCLAIMER,
            justify="left",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        ttk.Label(
            mode_frame,
            textvariable=self.splat_info_text_var,
            justify="left",
            wraplength=1000,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params_frame = ttk.LabelFrame(parent, text="Splatting Parameters", padding=8)
        params_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        for col in range(12):
            params_frame.grid_columnconfigure(col, weight=0)
        params_frame.grid_columnconfigure(11, weight=1)

        ttk.Label(params_frame, text="Batch size:").grid(row=0, column=0, sticky="w")
        self.splat_batch_size_entry = ttk.Entry(
            params_frame, textvariable=self.splat_batch_size_var, width=8
        )
        self.splat_batch_size_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Disparity:").grid(row=0, column=2, sticky="w")
        self.splat_disparity_entry = ttk.Entry(
            params_frame, textvariable=self.splat_disparity_var, width=8
        )
        self.splat_disparity_entry.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Workers:").grid(row=0, column=4, sticky="w")
        self.splat_workers_entry = ttk.Entry(
            params_frame, textvariable=self.splat_workers_var, width=8
        )
        self.splat_workers_entry.grid(row=0, column=5, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Layout:").grid(row=0, column=6, sticky="w")
        self.splat_layout_combo = ttk.Combobox(
            params_frame,
            textvariable=self.splat_layout_var,
            values=["Single Warp", "Dual", "Quad"],
            width=14,
            state="readonly",
        )
        self.splat_layout_combo.grid(row=0, column=7, sticky="w", padx=(6, 12))
        self.splat_layout_combo.bind("<<ComboboxSelected>>", self._on_splat_layout_changed)

        ttk.Label(params_frame, text="Auto-Convergence:").grid(row=0, column=8, sticky="w")
        self.splat_auto_convergence_combo = ttk.Combobox(
            params_frame,
            textvariable=self.splat_auto_convergence_var,
            values=["Min Borders", "Off"],
            width=14,
            state="readonly",
        )
        self.splat_auto_convergence_combo.grid(row=0, column=9, sticky="w", padx=(6, 12))
        self.splat_auto_convergence_combo.bind(
            "<<ComboboxSelected>>", self._on_splat_auto_convergence_changed
        )

        ttk.Label(params_frame, text="Dilate X:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.splat_dilate_x_entry = ttk.Entry(
            params_frame, textvariable=self.splat_dilate_x_var, width=8
        )
        self.splat_dilate_x_entry.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Dilate Y:").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.splat_dilate_y_entry = ttk.Entry(
            params_frame, textvariable=self.splat_dilate_y_var, width=8
        )
        self.splat_dilate_y_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Blur X:").grid(row=1, column=4, sticky="w", pady=(8, 0))
        self.splat_blur_x_entry = ttk.Entry(
            params_frame, textvariable=self.splat_blur_x_var, width=8
        )
        self.splat_blur_x_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Blur Y:").grid(row=1, column=6, sticky="w", pady=(8, 0))
        self.splat_blur_y_entry = ttk.Entry(
            params_frame, textvariable=self.splat_blur_y_var, width=8
        )
        self.splat_blur_y_entry.grid(row=1, column=7, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Dilate left:").grid(row=1, column=8, sticky="w", pady=(8, 0))
        self.splat_dilate_left_entry = ttk.Entry(
            params_frame, textvariable=self.splat_dilate_left_var, width=8
        )
        self.splat_dilate_left_entry.grid(row=1, column=9, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Blur balance:").grid(row=1, column=10, sticky="w", pady=(8, 0))
        self.splat_blur_balance_entry = ttk.Entry(
            params_frame, textvariable=self.splat_blur_balance_var, width=8
        )
        self.splat_blur_balance_entry.grid(row=1, column=11, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(params_frame, text="Gamma:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.splat_gamma_entry = ttk.Entry(
            params_frame, textvariable=self.splat_gamma_var, width=8
        )
        self.splat_gamma_entry.grid(row=2, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Convergence:").grid(row=2, column=2, sticky="w", pady=(8, 0))
        self.splat_convergence_entry = ttk.Entry(
            params_frame, textvariable=self.splat_convergence_var, width=8
        )
        self.splat_convergence_entry.grid(row=2, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        self.splat_stair_smooth_check = ttk.Checkbutton(
            params_frame,
            text="Stair smooth",
            variable=self.splat_stair_smooth_var,
        )
        self.splat_stair_smooth_check.grid(row=2, column=4, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(params_frame, text="Stair kernel:").grid(row=2, column=6, sticky="w", pady=(8, 0))
        self.splat_stair_kernel_entry = ttk.Entry(
            params_frame, textvariable=self.splat_stair_kernel_var, width=8
        )
        self.splat_stair_kernel_entry.grid(row=2, column=7, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Stair X off:").grid(row=2, column=8, sticky="w", pady=(8, 0))
        self.splat_stair_x_off_entry = ttk.Entry(
            params_frame, textvariable=self.splat_stair_x_off_var, width=8
        )
        self.splat_stair_x_off_entry.grid(row=2, column=9, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Stair strip:").grid(row=2, column=10, sticky="w", pady=(8, 0))
        self.splat_stair_strip_entry = ttk.Entry(
            params_frame, textvariable=self.splat_stair_strip_var, width=8
        )
        self.splat_stair_strip_entry.grid(row=2, column=11, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(params_frame, text="Stair strength:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.splat_stair_strength_entry = ttk.Entry(
            params_frame, textvariable=self.splat_stair_strength_var, width=8
        )
        self.splat_stair_strength_entry.grid(row=3, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        self.splat_replace_mask_check = ttk.Checkbutton(
            params_frame,
            text="Replace mask",
            variable=self.splat_replace_mask_var,
            command=self._on_splat_replace_mask_toggled,
        )
        self.splat_replace_mask_check.grid(row=3, column=2, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(params_frame, text="Replace scale:").grid(row=3, column=4, sticky="w", pady=(8, 0))
        self.splat_replace_scale_entry = ttk.Entry(
            params_frame, textvariable=self.splat_replace_mask_scale_var, width=8
        )
        self.splat_replace_scale_entry.grid(row=3, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Replace min:").grid(row=3, column=6, sticky="w", pady=(8, 0))
        self.splat_replace_min_entry = ttk.Entry(
            params_frame, textvariable=self.splat_replace_mask_min_var, width=8
        )
        self.splat_replace_min_entry.grid(row=3, column=7, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Replace max:").grid(row=3, column=8, sticky="w", pady=(8, 0))
        self.splat_replace_max_entry = ttk.Entry(
            params_frame, textvariable=self.splat_replace_mask_max_var, width=8
        )
        self.splat_replace_max_entry.grid(row=3, column=9, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Replace gap:").grid(row=3, column=10, sticky="w", pady=(8, 0))
        self.splat_replace_gap_entry = ttk.Entry(
            params_frame, textvariable=self.splat_replace_mask_gap_var, width=8
        )
        self.splat_replace_gap_entry.grid(row=3, column=11, sticky="w", padx=(6, 0), pady=(8, 0))

        self.splat_replace_mask_edge_check = ttk.Checkbutton(
            params_frame,
            text="Replace mask edge",
            variable=self.splat_replace_mask_edge_var,
        )
        self.splat_replace_mask_edge_check.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(
            params_frame,
            text=(
                "Replace mask is mandatory for downstream steps "
                "(Sharpness CSV / AutoCT CSV / Merge noGUI)."
            ),
            justify="left",
        ).grid(row=5, column=0, columnspan=12, sticky="w", pady=(6, 0))

        encode_frame = ttk.LabelFrame(parent, text="Encoding Args (inherited)", padding=8)
        encode_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(9, weight=1)

        self.splat_override_check = ttk.Checkbutton(
            encode_frame,
            text="Override",
            variable=self.splat_encode_override_var,
            command=self._on_splat_override_toggle,
        )
        self.splat_override_check.grid(row=0, column=0, sticky="w")

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.splat_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.splat_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.splat_codec_entry.grid(row=0, column=2, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Quality (CRF/QP):").grid(row=0, column=3, sticky="w")
        self.splat_crf_entry = ttk.Entry(encode_frame, textvariable=self.splat_crf_var, width=6)
        self.splat_crf_entry.grid(row=0, column=4, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Preset:").grid(row=0, column=5, sticky="w")
        self.splat_preset_entry = ttk.Entry(encode_frame, textvariable=self.splat_preset_var, width=10)
        self.splat_preset_entry.grid(row=0, column=6, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="PixFmt:").grid(row=0, column=7, sticky="w")
        self.splat_pixfmt_entry = ttk.Entry(encode_frame, textvariable=self.splat_pix_fmt_var, width=10)
        self.splat_pixfmt_entry.grid(row=0, column=8, sticky="w", padx=(6, 0))

        ttk.Label(encode_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.splat_extra_ffmpeg_entry = ttk.Entry(
            encode_frame, textvariable=self.splat_extra_ffmpeg_args_var
        )
        self.splat_extra_ffmpeg_entry.grid(
            row=1, column=1, columnspan=9, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.splat_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.splat_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_splat_command
        )
        self.splat_preview_btn.grid(row=0, column=0, padx=(0, 6))
        self.splat_run_btn = ttk.Button(
            buttons, text="Run Splatting", command=self._run_splat_placeholder
        )
        self.splat_run_btn.grid(row=0, column=1, padx=6)
        self.splat_verify_quick_btn = ttk.Button(
            buttons, text="Verify Scenes (Quick)", command=self._start_splat_verify_quick
        )
        self.splat_verify_quick_btn.grid(row=0, column=2, padx=6)
        self.splat_verify_deep_btn = ttk.Button(
            buttons, text="Verify Scenes (Deep)", command=self._start_splat_verify_deep
        )
        self.splat_verify_deep_btn.grid(row=0, column=3, padx=6)
        self.splat_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_splat_placeholder, state=tk.DISABLED
        )
        self.splat_stop_btn.grid(row=0, column=4, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_splat_log).grid(
            row=0, column=5, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=9, column=0, columnspan=3, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(2, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.splat_status_var).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        self.splat_progress = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.splat_progress_var,
            maximum=100.0,
        )
        self.splat_progress.grid(row=0, column=2, sticky="ew", padx=4)

        log_frame = ttk.LabelFrame(parent, text="Splatting Log", padding=6)
        log_frame.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.splat_log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.splat_log_text.grid(row=0, column=0, sticky="nsew")
        sscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.splat_log_text.yview)
        sscroll.grid(row=0, column=1, sticky="ns")
        self.splat_log_text.configure(yscrollcommand=sscroll.set)

        self._splat_manual_combo_widgets = [
            self.splat_layout_combo,
            self.splat_auto_convergence_combo,
        ]
        self._splat_manual_entry_widgets = [
            self.splat_dilate_x_entry,
            self.splat_dilate_y_entry,
            self.splat_blur_x_entry,
            self.splat_blur_y_entry,
            self.splat_dilate_left_entry,
            self.splat_blur_balance_entry,
            self.splat_gamma_entry,
            self.splat_convergence_entry,
            self.splat_stair_kernel_entry,
            self.splat_stair_x_off_entry,
            self.splat_stair_strip_entry,
            self.splat_stair_strength_entry,
            self.splat_replace_scale_entry,
            self.splat_replace_min_entry,
            self.splat_replace_max_entry,
            self.splat_replace_gap_entry,
        ]
        self._splat_manual_check_widgets = [
            self.splat_stair_smooth_check,
            self.splat_replace_mask_check,
            self.splat_replace_mask_edge_check,
        ]

        self._on_splat_mode_changed()
        self._on_splat_override_toggle(initial=True)
        self._preview_splat_command()
        self._set_splat_running(False)

    def _build_inpaint_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(11, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="Inpaint input clips (auto):").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.inpaint_input_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_inpaint_input_folder).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(parent, text="Inpaint mask folder (auto):").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.inpaint_mask_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_inpaint_mask_folder).grid(
            row=1, column=2, padx=4
        )

        ttk.Label(parent, text="Inpaint output folder (auto):").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.inpaint_output_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_inpaint_output_folder).grid(
            row=2, column=2, padx=4
        )

        mode_frame = ttk.LabelFrame(parent, text="Inpainting Mode", padding=8)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
        mode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(mode_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.inpaint_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.inpaint_mode_var,
            values=["Auto (recommended)", "Manual"],
            width=18,
            state="readonly",
        )
        self.inpaint_mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.inpaint_mode_combo.bind("<<ComboboxSelected>>", self._on_inpaint_mode_changed)

        ttk.Label(
            mode_frame,
            text=self.INPAINT_DISCLAIMER,
            justify="left",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        ttk.Label(
            mode_frame,
            textvariable=self.inpaint_info_text_var,
            justify="left",
            wraplength=1000,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params_frame = ttk.LabelFrame(parent, text="Inpainting Parameters", padding=8)
        params_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        for col in range(10):
            params_frame.grid_columnconfigure(col, weight=0)
        params_frame.grid_columnconfigure(9, weight=1)

        ttk.Label(params_frame, text="Frames Chunk:").grid(row=0, column=0, sticky="w")
        self.inpaint_frames_chunk_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_frames_chunk_var, width=8
        )
        self.inpaint_frames_chunk_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="CPU offload:").grid(row=0, column=2, sticky="w")
        self.inpaint_cpu_offload_combo = ttk.Combobox(
            params_frame,
            textvariable=self.inpaint_cpu_offload_var,
            values=["none", "model", "sequential"],
            width=12,
            state="readonly",
        )
        self.inpaint_cpu_offload_combo.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Tile Number:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.inpaint_tile_num_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_tile_num_var, width=8
        )
        self.inpaint_tile_num_entry.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Input Bias:").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.inpaint_input_bias_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_input_bias_var, width=8
        )
        self.inpaint_input_bias_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Overlap:").grid(row=1, column=4, sticky="w", pady=(8, 0))
        self.inpaint_overlap_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_overlap_var, width=8
        )
        self.inpaint_overlap_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="TailPad:").grid(row=1, column=6, sticky="w", pady=(8, 0))
        self.inpaint_tail_pad_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_tail_pad_var, width=8
        )
        self.inpaint_tail_pad_entry.grid(row=1, column=7, sticky="w", padx=(6, 12), pady=(8, 0))

        self.inpaint_use_sharpness_check = ttk.Checkbutton(
            params_frame,
            text="Use sharpness CSV (auto steps)",
            variable=self.inpaint_use_sharpness_csv_var,
            command=self._on_inpaint_auto_steps_toggle,
        )
        self.inpaint_use_sharpness_check.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        ttk.Label(params_frame, text="Inference steps:").grid(row=2, column=4, sticky="w", pady=(8, 0))
        self.inpaint_inference_steps_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_inference_steps_var, width=8
        )
        self.inpaint_inference_steps_entry.grid(row=2, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Sharpness CSV workers:").grid(
            row=2, column=6, sticky="w", pady=(8, 0)
        )
        self.inpaint_sharpness_workers_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_sharpness_workers_var, width=8
        )
        self.inpaint_sharpness_workers_entry.grid(
            row=2, column=7, sticky="w", padx=(6, 12), pady=(8, 0)
        )

        encode_frame = ttk.LabelFrame(parent, text="Encoding Args (inherited)", padding=8)
        encode_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(9, weight=1)

        self.inpaint_override_check = ttk.Checkbutton(
            encode_frame,
            text="Override",
            variable=self.inpaint_encode_override_var,
            command=self._on_inpaint_override_toggle,
        )
        self.inpaint_override_check.grid(row=0, column=0, sticky="w")

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.inpaint_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.inpaint_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.inpaint_codec_entry.grid(row=0, column=2, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Quality (CRF/QP):").grid(row=0, column=3, sticky="w")
        self.inpaint_crf_entry = ttk.Entry(encode_frame, textvariable=self.inpaint_crf_var, width=6)
        self.inpaint_crf_entry.grid(row=0, column=4, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Preset:").grid(row=0, column=5, sticky="w")
        self.inpaint_preset_entry = ttk.Entry(encode_frame, textvariable=self.inpaint_preset_var, width=10)
        self.inpaint_preset_entry.grid(row=0, column=6, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="PixFmt:").grid(row=0, column=7, sticky="w")
        self.inpaint_pixfmt_entry = ttk.Entry(encode_frame, textvariable=self.inpaint_pix_fmt_var, width=10)
        self.inpaint_pixfmt_entry.grid(row=0, column=8, sticky="w", padx=(6, 0))

        ttk.Label(encode_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.inpaint_extra_ffmpeg_entry = ttk.Entry(
            encode_frame, textvariable=self.inpaint_extra_ffmpeg_args_var
        )
        self.inpaint_extra_ffmpeg_entry.grid(
            row=1, column=1, columnspan=9, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.inpaint_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.inpaint_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_inpaint_command
        )
        self.inpaint_preview_btn.grid(row=0, column=0, padx=(0, 6))
        self.inpaint_sharp_btn = ttk.Button(
            buttons, text="Create Sharpness CSV", command=self._start_inpaint_sharpness_csv
        )
        self.inpaint_sharp_btn.grid(row=0, column=1, padx=6)
        self.inpaint_run_btn = ttk.Button(
            buttons, text="Run Inpainting", command=self._run_inpaint_placeholder
        )
        self.inpaint_run_btn.grid(row=0, column=2, padx=6)
        self.inpaint_verify_quick_btn = ttk.Button(
            buttons, text="Verify Scenes (Quick)", command=self._start_inpaint_verify_quick
        )
        self.inpaint_verify_quick_btn.grid(row=0, column=3, padx=6)
        self.inpaint_verify_deep_btn = ttk.Button(
            buttons, text="Verify Scenes (Deep)", command=self._start_inpaint_verify_deep
        )
        self.inpaint_verify_deep_btn.grid(row=0, column=4, padx=6)
        self.inpaint_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_inpaint_placeholder, state=tk.DISABLED
        )
        self.inpaint_stop_btn.grid(row=0, column=5, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_inpaint_log).grid(
            row=0, column=6, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=8, column=0, columnspan=3, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(2, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.inpaint_status_var).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        self.inpaint_progress = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.inpaint_progress_var,
            maximum=100.0,
        )
        self.inpaint_progress.grid(row=0, column=2, sticky="ew", padx=4)

        log_frame = ttk.LabelFrame(parent, text="Inpainting Log", padding=6)
        log_frame.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.inpaint_log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.inpaint_log_text.grid(row=0, column=0, sticky="nsew")
        iscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.inpaint_log_text.yview)
        iscroll.grid(row=0, column=1, sticky="ns")
        self.inpaint_log_text.configure(yscrollcommand=iscroll.set)

        self._inpaint_manual_widgets = [
            self.inpaint_tile_num_entry,
            self.inpaint_input_bias_entry,
            self.inpaint_overlap_entry,
            self.inpaint_tail_pad_entry,
            self.inpaint_use_sharpness_check,
        ]

        self._on_inpaint_mode_changed()
        self._on_inpaint_override_toggle(initial=True)
        self._preview_inpaint_command()
        self._set_inpaint_running(False)

    def _open_inpaint_input_folder(self) -> None:
        folder = self.inpaint_input_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_inpaint_log(f"Inpaint input folder ready: {folder}")

    def _open_inpaint_mask_folder(self) -> None:
        folder = self.inpaint_mask_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_inpaint_log(f"Inpaint mask folder ready: {folder}")

    def _open_inpaint_output_folder(self) -> None:
        folder = self.inpaint_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_inpaint_log(f"Inpaint output folder ready: {folder}")

    def _on_inpaint_mode_changed(self, _event=None) -> None:
        mode = self.inpaint_mode_var.get().strip()
        if mode == "Manual":
            self.inpaint_info_text_var.set(self.INPAINT_MANUAL_INFO)
        else:
            self.inpaint_mode_var.set("Auto (recommended)")
            self.inpaint_info_text_var.set(self.INPAINT_AUTO_INFO)
            self._reset_inpaint_auto_locked_defaults()
        self._apply_inpaint_control_states()

    def _reset_inpaint_auto_locked_defaults(self) -> None:
        # Fields disabled in Auto mode.
        self.inpaint_tile_num_var.set("2")
        self.inpaint_input_bias_var.set("0")
        self.inpaint_overlap_var.set("3")
        self.inpaint_tail_pad_var.set("2")
        self.inpaint_use_sharpness_csv_var.set(True)
        self.inpaint_inference_steps_var.set("8")

    def _on_inpaint_auto_steps_toggle(self) -> None:
        self._apply_inpaint_control_states()

    def _apply_inpaint_control_states(self) -> None:
        mode_manual = self.inpaint_mode_var.get().strip() == "Manual"

        self.inpaint_frames_chunk_entry.configure(state=tk.NORMAL)
        self.inpaint_cpu_offload_combo.configure(state="readonly")

        manual_state = tk.NORMAL if mode_manual else tk.DISABLED
        for widget in getattr(self, "_inpaint_manual_widgets", []):
            widget.configure(state=manual_state)

        use_auto_steps = bool(self.inpaint_use_sharpness_csv_var.get())
        if mode_manual and not use_auto_steps:
            self.inpaint_inference_steps_entry.configure(state=tk.NORMAL)
        else:
            self.inpaint_inference_steps_entry.configure(state=tk.DISABLED)

        self._preview_inpaint_command()
        self._refresh_pipeline_status_panel()

    def _sync_inpaint_encoding_from_scene(self) -> None:
        self.inpaint_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.scene_codec_var.get(),
                self.DEFAULT_SCENE_CODEC,
            )
        )
        self.inpaint_crf_var.set(self.scene_crf_var.get().strip())
        self.inpaint_preset_var.set(self.scene_encoder_preset_var.get().strip())
        self.inpaint_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip())

    def _on_inpaint_override_toggle(self, initial: bool = False) -> None:
        enabled = bool(self.inpaint_encode_override_var.get())
        if not enabled:
            self._sync_inpaint_encoding_from_scene()
        elif not initial and not self._inpaint_override_notice_shown:
            self._inpaint_override_notice_shown = True
            messagebox.showwarning("Inpainting Encode Override", self.INPAINT_OVERRIDE_WARNING)

        state = tk.NORMAL if enabled else tk.DISABLED
        self._set_codec_widget_override_state(self.inpaint_codec_entry, enabled)
        for widget in (
            self.inpaint_crf_entry,
            self.inpaint_preset_entry,
            self.inpaint_pixfmt_entry,
            self.inpaint_extra_ffmpeg_entry,
        ):
            widget.configure(state=state)
        self._preview_inpaint_command()

    def _build_inpaint_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        input_dir = self.inpaint_input_var.get().strip()
        output_dir = self.inpaint_output_var.get().strip()
        mask_dir = self.inpaint_mask_var.get().strip()
        work_dir = self.work_folder_var.get().strip() or "./work"
        sharp_base = os.path.normpath(work_dir)
        sharp_csv_path = self.inpaint_sharpness_csv_var.get().strip()
        use_sharpness_csv = bool(self.inpaint_use_sharpness_csv_var.get())
        codec_value = self._normalize_ffmpeg_codec(
            self.inpaint_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        self.inpaint_codec_var.set(codec_value)

        env_updates: dict[str, str] = {
            "PYTHON": sys.executable,
            "RUNNER": "batch_inpainting_runner.py",
            "INPUT_DIR": input_dir,
            "OUTPUT_DIR": output_dir,
            "GLOB": "*.mp4",
            "REPLACE_MASK_FOLDER": mask_dir,
            "USE_REPLACE_MASK": "1",
            "OFFLOAD_TYPE": self.inpaint_cpu_offload_var.get().strip() or "model",
            "FRAMES_CHUNK": self.inpaint_frames_chunk_var.get().strip() or "50",
            "TILE_NUM": self.inpaint_tile_num_var.get().strip() or "2",
            "OVERLAP": self.inpaint_overlap_var.get().strip() or "3",
            "TAIL_PAD": self.inpaint_tail_pad_var.get().strip() or "2",
            "ORIGINAL_INPUT_BLEND_STRENGTH": self.inpaint_input_bias_var.get().strip() or "0",
            "OUTPUT_CRF": self.inpaint_crf_var.get().strip() or "1",
            "OUTPUT_CODEC": codec_value,
            "OUTPUT_PRESET": self.inpaint_preset_var.get().strip(),
            "OUTPUT_PIX_FMT": self.inpaint_pix_fmt_var.get().strip(),
            "OUTPUT_EXTRA_ARGS": self.inpaint_extra_ffmpeg_args_var.get().strip(),
            "NO_SHARPNESS_CSV": "0" if use_sharpness_csv else "1",
            "SHARPNESS_BASE": sharp_base,
            "SHARPNESS_CSV_PATH": sharp_csv_path,
            "FIXED_STEPS": self.inpaint_inference_steps_var.get().strip() or "8",
            # Hardcoded unsupported features in this tab.
            "ENABLE_POST_INPAINTING_BLEND": "0",
            "DISABLE_COLOR_TRANSFER": "1",
            "DISABLE_DYNAMIC_CHUNK": "1",
            "STOP_MARKER": os.path.join(
                output_dir or os.path.join(work_dir, self.STANDARD_SUBDIRS["inpaint"]),
                ".stop_after_current",
            ),
            "RETRY_POLICY_JSON": self._build_retry_policy_json(
                self.inpaint_retry_policy_vars,
                self.inpaint_cpu_offload_var.get().strip() or "model",
            ),
        }

        cmd = ["bash", "run_inpainting_runner.sh"]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview

    def _preview_inpaint_command(self) -> None:
        try:
            _cmd, _env, preview = self._build_inpaint_runner_payload()
            self.inpaint_cmd_preview_var.set(preview)
        except Exception as e:
            self.inpaint_cmd_preview_var.set(f"Invalid options: {e}")

    def _append_inpaint_log(self, line: str) -> None:
        self.inpaint_log_text.configure(state=tk.NORMAL)
        self.inpaint_log_text.insert(tk.END, line + "\n")
        self.inpaint_log_text.see(tk.END)
        self.inpaint_log_text.configure(state=tk.DISABLED)

    def _clear_inpaint_log(self) -> None:
        self.inpaint_log_text.configure(state=tk.NORMAL)
        self.inpaint_log_text.delete("1.0", tk.END)
        self.inpaint_log_text.configure(state=tk.DISABLED)

    def _run_inpaint_placeholder(self) -> None:
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Inpainting", "Inpainting is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Inpainting", "Stop verification before starting Inpainting.")
            return
        try:
            cmd, env_updates, _preview = self._build_inpaint_runner_payload()
        except Exception as exc:
            messagebox.showerror("Inpainting", f"Invalid inpainting options:\n{exc}")
            return

        launcher_script = Path("run_inpainting_runner.sh").resolve()
        if not launcher_script.is_file():
            messagebox.showerror("Inpainting", f"Launcher not found:\n{launcher_script}")
            return

        runner_script = Path(env_updates.get("RUNNER", "batch_inpainting_runner.py")).resolve()
        if not runner_script.is_file():
            messagebox.showerror("Inpainting", f"Runner not found:\n{runner_script}")
            return

        input_dir = self.inpaint_input_var.get().strip()
        mask_dir = self.inpaint_mask_var.get().strip()
        output_dir = self.inpaint_output_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Inpainting", f"Input folder not found:\n{input_dir or '(empty)'}")
            return
        if not mask_dir or not os.path.isdir(mask_dir):
            messagebox.showerror("Inpainting", f"Mask folder not found:\n{mask_dir or '(empty)'}")
            return
        if not output_dir:
            messagebox.showerror("Inpainting", "Output folder is required.")
            return
        os.makedirs(output_dir, exist_ok=True)

        requires_sharpness_csv = (
            self.inpaint_mode_var.get().strip() == "Auto (recommended)"
            or bool(self.inpaint_use_sharpness_csv_var.get())
        )
        if requires_sharpness_csv:
            mask_supported = self._is_splat_replace_mask_active()
            has_masks = self._has_any_replace_masks(mask_dir)
            if not mask_supported or not has_masks:
                messagebox.showerror(
                    "Inpainting",
                    (
                        "Sharpness CSV requires replace masks exported in Splatting.\n"
                        "Enable Replace mask in Splatting and ensure mask files exist in work/mask."
                    ),
                )
                return
            sharp_csv_path = self.inpaint_sharpness_csv_var.get().strip()
            if not sharp_csv_path:
                sharp_csv_path = str(
                    Path(self.work_folder_var.get().strip() or "./work").resolve()
                    / "sharpness.csv"
                )
                self.inpaint_sharpness_csv_var.set(sharp_csv_path)

            sharp_ok, sharp_msg, sharp_missing = self._verify_sharpness_csv_coverage(
                input_dir, sharp_csv_path
            )
            if not sharp_ok:
                if not self._pipeline_test_active:
                    self._pipeline_set_completed("sharpness_csv", False)
                    self._pipeline_set_verified("sharpness_csv", "none")
                    self._refresh_pipeline_status_panel()
                    self._save_pipeline_state()
                self._append_inpaint_log(f"[SHARP][VERIFY] {sharp_msg}")
                if sharp_missing:
                    preview = "\n".join(sharp_missing[:12])
                    more = (
                        ""
                        if len(sharp_missing) <= 12
                        else f"\n... and {len(sharp_missing) - 12} more"
                    )
                    self._append_inpaint_log(
                        f"[SHARP][VERIFY] Missing rows:\n{preview}{more}"
                    )
                messagebox.showwarning(
                    "Inpainting",
                    (
                        f"{sharp_msg}\n\n"
                        "Sharpness CSV will be regenerated now before Inpainting."
                    ),
                )
                self.inpaint_status_var.set("Sharpness CSV incomplete, rebuilding...")
                self._start_inpaint_sharpness_csv(resume_inpaint_after=True)
                return
            if not self._pipeline_test_active:
                self._pipeline_set_completed("sharpness_csv", True)
                self._pipeline_set_verified("sharpness_csv", "none")
                self._refresh_pipeline_status_panel()
                self._save_pipeline_state()

        self._inpaint_resume_after_sharpness = False
        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self.inpaint_status_var.set("Starting...")
        self.inpaint_progress_var.set(0.0)
        self._set_inpaint_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("inpaint")
        self._append_inpaint_log("=== Inpainting started ===")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_inpaint_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._inpaint_thread = threading.Thread(
            target=self._run_inpaint_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._inpaint_thread.start()

    def _run_inpaint_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        step_success = False
        try:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._inpaint_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("inpaint_line", line))
                    self._try_parse_inpaint_progress(line)
                if self._inpaint_stop_requested:
                    break
            rc = proc.wait()
            if self._inpaint_stop_requested:
                self._log_queue.put(("inpaint_status", "Stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("inpaint_status", "Completed"))
                self._log_queue.put(("inpaint_progress", "100"))
            else:
                self._log_queue.put(("inpaint_status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("inpaint_line", f"[INPAINT][ERROR] {exc}"))
            self._log_queue.put(("inpaint_status", "Failed"))
        finally:
            self._inpaint_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("inpaint_done", {"step": "inpaint", "success": step_success}))

    def _start_inpaint_sharpness_csv(self, resume_inpaint_after: bool = False) -> None:
        # Enable auto-resume only when Sharpness CSV is launched as Inpaint preflight.
        self._inpaint_resume_after_sharpness = False
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Inpainting", "Another inpainting task is running.")
            return
        if self._verify_running:
            messagebox.showinfo("Inpainting", "Stop verification before creating sharpness CSV.")
            return

        script_path = Path("Utilities/analyze_inpaint_sharpness.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Inpainting", f"Script not found:\n{script_path}")
            return

        input_dir = self.inpaint_input_var.get().strip()
        mask_dir = self.inpaint_mask_var.get().strip()
        work_dir = self.work_folder_var.get().strip() or "./work"
        if self._pipeline_test_active:
            cur_test_csv = self.inpaint_sharpness_csv_var.get().strip()
            if cur_test_csv:
                out_csv = Path(cur_test_csv).resolve()
            elif self._pipeline_test_dir:
                out_csv = Path(self._pipeline_test_dir).resolve() / "sharpness_test.csv"
            else:
                out_csv = Path(work_dir).resolve() / "sharpness_test.csv"
        else:
            out_csv = Path(work_dir).resolve() / "sharpness.csv"
        os.makedirs(str(out_csv.parent), exist_ok=True)
        self.inpaint_sharpness_csv_var.set(str(out_csv))

        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Inpainting", f"Input folder not found:\n{input_dir or '(empty)'}")
            return
        if not mask_dir or not os.path.isdir(mask_dir):
            messagebox.showerror("Inpainting", f"Mask folder not found:\n{mask_dir or '(empty)'}")
            return
        if not self._has_any_replace_masks(mask_dir):
            messagebox.showerror(
                "Inpainting",
                (
                    "No replace-mask files found.\n"
                    "Sharpness CSV requires replace masks from Splatting."
                ),
            )
            return
        missing = self._find_missing_replace_masks(input_dir, mask_dir)
        if missing:
            show = "\n".join(missing[:8])
            more = "" if len(missing) <= 8 else f"\n... and {len(missing) - 8} more"
            messagebox.showwarning(
                "Inpainting Replace Mask",
                (
                    "Some replace masks are missing.\n"
                    "Sharpness CSV will continue for available scenes and report missing masks.\n\n"
                    f"Missing examples:\n{show}{more}"
                ),
            )

        default_workers = 19
        try:
            workers = max(1, int(self.inpaint_sharpness_workers_var.get().strip() or str(default_workers)))
        except Exception:
            workers = default_workers
        if self.inpaint_sharpness_workers_var.get().strip() != str(workers):
            self.inpaint_sharpness_workers_var.set(str(workers))
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(input_dir).resolve()),
            str(Path(mask_dir).resolve()),
            "--out_csv",
            str(out_csv),
            "--workers",
            str(workers),
        ]

        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self.inpaint_status_var.set("Creating sharpness CSV...")
        self.inpaint_progress_var.set(0.0)
        self._set_inpaint_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("sharpness_csv")
        self._append_inpaint_log("=== Sharpness CSV creation started ===")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._inpaint_resume_after_sharpness = bool(resume_inpaint_after)
        self._inpaint_thread = threading.Thread(
            target=self._run_inpaint_sharpness_worker,
            args=(cmd, str(out_csv)),
            daemon=True,
        )
        self._inpaint_thread.start()

    def _run_inpaint_sharpness_worker(self, cmd: list[str], out_csv: str) -> None:
        proc = None
        step_success = False
        try:
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=preexec,
            )
            self._inpaint_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("inpaint_line", f"[SHARP] {line}"))
            rc = proc.wait()
            if self._inpaint_stop_requested:
                self._log_queue.put(("inpaint_status", "Sharpness CSV stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("inpaint_status", "Sharpness CSV created"))
                self._log_queue.put(("inpaint_progress", "100"))
                self._log_queue.put(("inpaint_line", f"[SHARP] output: {out_csv}"))
            else:
                self._log_queue.put(("inpaint_status", f"Sharpness CSV failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("inpaint_line", f"[SHARP][ERROR] {exc}"))
            self._log_queue.put(("inpaint_status", "Sharpness CSV failed"))
        finally:
            self._inpaint_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("inpaint_done", {"step": "sharpness_csv", "success": step_success}))

    def _stop_inpaint_placeholder(self, prompt_user: bool = True) -> None:
        running = bool(self._inpaint_thread and self._inpaint_thread.is_alive())
        if not running:
            return
        if self._inpaint_stop_clicks == 0 and prompt_user:
            messagebox.showwarning(
                "Stop Inpainting",
                "Graceful stop requested.\n\n"
                "Current process will be interrupted like Ctrl+C.\n"
                "Click Stop again to force kill immediately.",
            )
        self._inpaint_stop_requested = True
        self._inpaint_stop_clicks += 1

        if self._inpaint_stop_clicks == 1:
            self.inpaint_status_var.set("Graceful stop requested...")
            self._append_inpaint_log(
                "[STOP] graceful stop requested (click Stop again for immediate force stop)."
            )
            self.inpaint_stop_btn.configure(text="Force Stop")
        else:
            self.inpaint_status_var.set("Force stop requested...")
            self._append_inpaint_log("[STOP] force stop requested.")

        self._send_inpaint_signal(signal.SIGINT)
        if self._inpaint_stop_clicks >= 2:
            self.root.after(1000, self._force_kill_inpaint)

    def _send_inpaint_signal(self, sig: int) -> None:
        proc = self._inpaint_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception as exc:
            self._append_inpaint_log(f"Signal send failed: {exc}")

    def _force_kill_inpaint(self) -> None:
        proc = self._inpaint_process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                self._append_inpaint_log("Inpainting process force-killed.")
            except Exception as exc:
                self._append_inpaint_log(f"Inpainting kill failed: {exc}")

    def _set_inpaint_running(self, is_running: bool) -> None:
        self.inpaint_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.inpaint_sharp_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.inpaint_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.inpaint_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        verify_state = tk.DISABLED if (is_running or self._verify_running) else tk.NORMAL
        self.inpaint_verify_quick_btn.configure(state=verify_state)
        self.inpaint_verify_deep_btn.configure(state=verify_state)
        if is_running:
            self.inpaint_stop_btn.configure(text="Stop")
        else:
            self.inpaint_stop_btn.configure(text="Stop")
            self._inpaint_stop_clicks = 0
            self._inpaint_stop_requested = False
        self._update_replace_mask_dependent_controls()

    def _try_parse_inpaint_progress(self, line: str) -> None:
        m = re.search(r"^\[(\d+)\s*/\s*(\d+)\]", line)
        if m:
            try:
                idx = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    prog = max(0.0, min(100.0, (idx / total) * 100.0))
                    self._log_queue.put(("inpaint_progress", str(prog)))
            except Exception:
                pass
            return
        m2 = re.search(r"^\[(?:RUN|OK|SKIP|ERR)\s*\]\s*(\d+)\s*/\s*(\d+)", line)
        if m2:
            try:
                idx = int(m2.group(1))
                total = int(m2.group(2))
                if total > 0:
                    prog = max(0.0, min(100.0, (idx / total) * 100.0))
                    self._log_queue.put(("inpaint_progress", str(prog)))
            except Exception:
                pass
            return
        if line.startswith("[DONE]"):
            self._log_queue.put(("inpaint_progress", "100"))

    def _validate_inpaint_verify_inputs(self) -> tuple[bool, str, str, list[str]]:
        out_dir = self.inpaint_output_var.get().strip()
        if not out_dir:
            messagebox.showerror("Verify Inpainting", "Inpainting output folder is required.")
            return False, "", "", []
        if not os.path.isdir(out_dir):
            messagebox.showerror("Verify Inpainting", f"Inpainting output folder not found:\n{out_dir}")
            return False, "", "", []
        ok_ref, ref_dir, ref_patterns, ref_kind = self._resolve_verify_reference(
            "inpaint", "Verify Inpainting"
        )
        if not ok_ref:
            return False, "", "", []
        out_dir = self._pipeline_prepare_verify_subset_dir(
            out_dir, "inpaint_target", ["*.mp4"]
        )
        self._append_inpaint_log(f"[VERIFY] reference source: {ref_kind} ({ref_dir})")
        return True, out_dir, ref_dir, ref_patterns

    def _start_inpaint_verify_quick(self) -> None:
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Verify Inpainting", "Stop Inpainting before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Inpainting", "Another verification is already running.")
            return
        ok, out_dir, ref_dir, ref_patterns = self._validate_inpaint_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Inpainting", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="inpaint_quick")
        self.inpaint_status_var.set("Verify (Quick) running...")
        self._append_inpaint_log("=== Verify Scenes (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_inpaint_verify_quick_worker,
            args=(out_dir, ref_dir, ref_patterns),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_inpaint_verify_quick_worker(
        self, out_dir: str, ref_dir: str, ref_patterns: list[str]
    ) -> None:
        try:
            out_files = sorted([str(p) for p in Path(out_dir).glob("*.mp4") if p.is_file()])
            ref_files = self._collect_files_for_patterns(ref_dir, ref_patterns)
            if not out_files:
                self._log_queue.put(("inpaint_verify_quick_result", {
                    "ok": False,
                    "message": "No .mp4 files found in inpainting output folder.",
                    "broken_output": [],
                    "broken_reference": [],
                }))
                return
            if not ref_files:
                self._log_queue.put(("inpaint_verify_quick_result", {
                    "ok": False,
                    "message": "No reference video files found in selected reference folder.",
                    "broken_output": [],
                    "broken_reference": [],
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            self._log_queue.put(
                ("inpaint_line", f"[QUICK] checking output files={len(out_files)} and reference files={len(ref_files)} with {max_workers} workers")
            )

            def _probe_group(file_list: list[str], label: str) -> dict:
                broken: list[str] = []
                total_duration = 0.0
                duration_available = True
                total_frames = 0
                frames_available = True

                def _probe_one(fp: str) -> tuple[str, dict]:
                    return fp, self._probe_video_basic(fp)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_probe_one, fp) for fp in file_list]
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        fp, meta = fut.result()
                        done += 1
                        if not meta.get("ok"):
                            broken.append(fp)
                            self._log_queue.put(("inpaint_line", f"[QUICK][{label.upper()}][BROKEN] {fp} :: {meta.get('error')}"))
                        else:
                            dur = meta.get("duration")
                            frm = meta.get("frames")
                            if dur is None:
                                duration_available = False
                            else:
                                total_duration += float(dur)
                            if frm is None:
                                frames_available = False
                            else:
                                total_frames += int(frm)
                        if done % 25 == 0 or done == len(file_list):
                            self._log_queue.put(("inpaint_line", f"[QUICK][{label.upper()}] progress {done}/{len(file_list)}"))
                return {
                    "broken": broken,
                    "total_duration": total_duration,
                    "duration_available": duration_available,
                    "total_frames": total_frames,
                    "frames_available": frames_available,
                }

            out_stats = _probe_group(out_files, "output")
            ref_stats = _probe_group(ref_files, "reference")

            count_ok = len(out_files) == len(ref_files)
            count_msg = f"output={len(out_files)} vs reference={len(ref_files)}"

            duration_ok = False
            duration_msg = "n.d."
            if out_stats["duration_available"] and ref_stats["duration_available"]:
                dd = abs(float(out_stats["total_duration"]) - float(ref_stats["total_duration"]))
                duration_ok = dd <= 0.35
                duration_msg = (
                    f"output={float(out_stats['total_duration']):.3f}s vs "
                    f"reference={float(ref_stats['total_duration']):.3f}s (delta={dd:.3f}s)"
                )

            frames_ok = False
            frames_msg = "n.d."
            if out_stats["frames_available"] and ref_stats["frames_available"]:
                df = abs(int(out_stats["total_frames"]) - int(ref_stats["total_frames"]))
                frames_ok = df <= 1
                frames_msg = (
                    f"output={int(out_stats['total_frames'])} vs "
                    f"reference={int(ref_stats['total_frames'])} (delta={df})"
                )

            self._log_queue.put(("inpaint_line", f"[QUICK] file count check: {count_msg}"))
            self._log_queue.put(("inpaint_line", f"[QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("inpaint_line", f"[QUICK] packet check: {frames_msg}"))

            ok_final = (
                not out_stats["broken"]
                and not ref_stats["broken"]
                and count_ok
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Inpainting quick verify completed.\n"
                f"Broken output files: {len(out_stats['broken'])}\n"
                f"Broken reference files: {len(ref_stats['broken'])}\n"
                f"File count match: {'YES' if count_ok else 'NO'} ({count_msg})\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(
                (
                    "inpaint_verify_quick_result",
                    {
                        "ok": ok_final,
                        "message": message,
                        "broken_output": out_stats["broken"],
                        "broken_reference": ref_stats["broken"],
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(("inpaint_verify_quick_result", {
                "ok": False,
                "message": f"Inpainting quick verify failed: {type(e).__name__}: {e}",
                "broken_output": [],
                "broken_reference": [],
            }))
        finally:
            self._log_queue.put(("verify_done", "inpaint_quick"))

    def _start_inpaint_verify_deep(self) -> None:
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Verify Inpainting", "Stop Inpainting before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Inpainting", "Another verification is already running.")
            return
        ok, out_dir, ref_dir, _ref_patterns = self._validate_inpaint_verify_inputs()
        if not ok:
            return

        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Inpainting", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(out_dir).resolve()),
            str(Path(ref_dir).resolve()),
            "--extensions",
            self.VERIFY_ALL_VIDEO_EXTENSIONS,
            "--workers",
            str(workers),
            "--probe-timeout-sec",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
            "--probe-timeout-retries",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
            "--delete",
            "yes",
            "--no-single-line-progress",
        ]

        self._set_verify_running(True, mode="inpaint_deep")
        self.inpaint_status_var.set("Verify (Deep) running...")
        self._append_inpaint_log("=== Verify Scenes (Deep) started ===")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))

        self._verify_thread = threading.Thread(
            target=self._run_inpaint_verify_deep_worker,
            args=(cmd, str(Path(out_dir).resolve())),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_inpaint_verify_deep_worker(self, cmd: list[str], out_dir: str) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=(os.setsid if hasattr(os, "setsid") else None),
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self._log_queue.put(("inpaint_line", f"[DEEP] {line}"))
                    bad_path = self._resolve_verifyscenes_bad_path(line, out_dir)
                    if bad_path and bad_path not in seen_bad:
                        seen_bad.add(bad_path)
                        bad_files.append(bad_path)
            rc = proc.wait()
        except Exception as e:
            self._log_queue.put(("inpaint_line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "inpaint_verify_deep_result",
                    {"rc": rc, "out_dir": out_dir, "bad_files": bad_files},
                )
            )
            self._log_queue.put(("verify_done", "inpaint_deep"))

    def _build_merge_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(13, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="Inpainted folder (auto):").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_inpainted_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_inpainted_folder).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(parent, text="Splatted folder (auto):").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_splatted_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_splatted_folder).grid(
            row=1, column=2, padx=4
        )

        ttk.Label(parent, text="Original folder (auto):").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_original_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_original_folder).grid(
            row=2, column=2, padx=4
        )

        ttk.Label(parent, text="Replace mask folder (auto):").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_replace_mask_var, state="readonly").grid(
            row=3, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_replace_mask_folder).grid(
            row=3, column=2, padx=4
        )

        ttk.Label(parent, text="Mask-for-merge folder (auto):").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_mask_formerge_var, state="readonly").grid(
            row=4, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_mask_formerge_folder).grid(
            row=4, column=2, padx=4
        )

        ttk.Label(parent, text="Merge output folder (auto):").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_output_var, state="readonly").grid(
            row=5, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_output_folder).grid(
            row=5, column=2, padx=4
        )

        ttk.Label(parent, text="AutoCT CSV (auto):").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.merge_autoct_csv_var, state="readonly").grid(
            row=6, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_merge_autoct_csv_folder).grid(
            row=6, column=2, padx=4
        )

        mode_frame = ttk.LabelFrame(parent, text="Merging Mode", padding=8)
        mode_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)
        mode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(mode_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.merge_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.merge_mode_var,
            values=["Auto (recommended)", "Manual"],
            width=18,
            state="readonly",
        )
        self.merge_mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.merge_mode_combo.bind("<<ComboboxSelected>>", self._on_merge_mode_changed)

        ttk.Label(
            mode_frame,
            textvariable=self.merge_info_text_var,
            justify="left",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(
            mode_frame,
            text=self.MERGE_DISCLAIMER,
            justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params_frame = ttk.LabelFrame(parent, text="Merge Parameters", padding=8)
        params_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=6)

        ttk.Label(params_frame, text="AutoCT CSV workers:").grid(row=0, column=0, sticky="w")
        self.merge_autoct_workers_entry = ttk.Entry(
            params_frame, textvariable=self.merge_autoct_workers_var, width=7
        )
        self.merge_autoct_workers_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Merge workers:").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        self.merge_parallel_workers_entry = ttk.Entry(
            params_frame, textvariable=self.merge_parallel_workers_var, width=7
        )
        self.merge_parallel_workers_entry.grid(row=0, column=3, sticky="w", padx=(6, 12))
        self.merge_parallel_workers_entry.bind("<KeyRelease>", self._on_merge_workers_changed)
        self.merge_parallel_workers_entry.bind("<FocusOut>", self._on_merge_workers_changed)

        ttk.Label(params_frame, text="Mask-for-merge workers:").grid(
            row=0, column=5, sticky="w", padx=(12, 0)
        )
        self.merge_mask_formerge_workers_entry = ttk.Entry(
            params_frame, textvariable=self.merge_mask_formerge_workers_var, width=7
        )
        self.merge_mask_formerge_workers_entry.grid(row=0, column=6, sticky="w", padx=(6, 12))

        self.merge_use_gpu_check = ttk.Checkbutton(
            params_frame,
            text="Use GPU",
            variable=self.merge_use_gpu_var,
            command=self._preview_merge_command,
        )
        self.merge_use_gpu_check.grid(row=0, column=7, sticky="w")

        ttk.Label(params_frame, text="Output:").grid(row=0, column=8, sticky="w", padx=(12, 0))
        self.merge_output_format_combo = ttk.Combobox(
            params_frame,
            textvariable=self.merge_output_format_var,
            values=["Full SBS", "Half SBS"],
            state="readonly",
            width=10,
        )
        self.merge_output_format_combo.grid(row=0, column=9, sticky="w", padx=(6, 12))
        self.merge_output_format_combo.bind("<<ComboboxSelected>>", self._preview_merge_command)

        ttk.Label(params_frame, text="Chunks:").grid(row=0, column=10, sticky="w")
        self.merge_chunks_entry = ttk.Entry(params_frame, textvariable=self.merge_chunks_var, width=7)
        self.merge_chunks_entry.grid(row=0, column=11, sticky="w", padx=(6, 0))

        ttk.Label(params_frame, text="Binarize thresh:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.merge_mask_binarize_entry = ttk.Entry(
            params_frame, textvariable=self.merge_mask_binarize_var, width=7
        )
        self.merge_mask_binarize_entry.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Dilate kernel:").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.merge_mask_dilate_entry = ttk.Entry(
            params_frame, textvariable=self.merge_mask_dilate_var, width=7
        )
        self.merge_mask_dilate_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Blur kernel:").grid(row=1, column=4, sticky="w", pady=(8, 0))
        self.merge_mask_blur_entry = ttk.Entry(
            params_frame, textvariable=self.merge_mask_blur_var, width=7
        )
        self.merge_mask_blur_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Shadow length:").grid(row=1, column=6, sticky="w", pady=(8, 0))
        self.merge_shadow_length_entry = ttk.Entry(
            params_frame, textvariable=self.merge_shadow_length_var, width=7
        )
        self.merge_shadow_length_entry.grid(row=1, column=7, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Shadow curve:").grid(row=1, column=8, sticky="w", pady=(8, 0))
        self.merge_shadow_curve_entry = ttk.Entry(
            params_frame, textvariable=self.merge_shadow_curve_var, width=7
        )
        self.merge_shadow_curve_entry.grid(row=1, column=9, sticky="w", padx=(6, 0), pady=(8, 0))

        self.merge_shadow_motion_enabled_check = ttk.Checkbutton(
            params_frame,
            text="Motion chain enabled",
            variable=self.merge_shadow_motion_enabled_var,
            command=self._preview_merge_command,
        )
        self.merge_shadow_motion_enabled_check.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        self.merge_dynamic_shadow_check = ttk.Checkbutton(
            params_frame,
            text="Dynamic shadow by mask width",
            variable=self.merge_dynamic_shadow_width_var,
            command=self._preview_merge_command,
        )
        self.merge_dynamic_shadow_check.grid(row=2, column=2, columnspan=2, sticky="w", pady=(8, 0))

        self.merge_use_replace_mask_check = ttk.Checkbutton(
            params_frame,
            text="Use replace mask",
            variable=self.merge_use_replace_mask_var,
            command=self._preview_merge_command,
        )
        self.merge_use_replace_mask_check.grid(row=2, column=4, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(params_frame, text="CT preset:").grid(row=2, column=6, sticky="w", pady=(8, 0))
        self.merge_ct_preset_combo = ttk.Combobox(
            params_frame,
            textvariable=self.merge_ct_preset_var,
            values=["1", "2", "3", "4", "5", "6", "7", "8"],
            width=4,
            state="readonly",
        )
        self.merge_ct_preset_combo.grid(row=2, column=7, sticky="w", padx=(6, 12), pady=(8, 0))
        self.merge_ct_preset_combo.bind("<<ComboboxSelected>>", self._preview_merge_command)

        ttk.Label(params_frame, text="AutoCT:").grid(row=2, column=8, sticky="w", pady=(8, 0))
        self.merge_ct_auto_mode_combo = ttk.Combobox(
            params_frame,
            textvariable=self.merge_ct_auto_mode_var,
            values=["CSV Blend", "On", "Off"],
            state="readonly",
            width=10,
        )
        self.merge_ct_auto_mode_combo.grid(row=2, column=9, sticky="w", padx=(6, 0), pady=(8, 0))
        self.merge_ct_auto_mode_combo.bind("<<ComboboxSelected>>", self._on_merge_ct_auto_mode_changed)

        self.merge_ct_exclude_black_check = ttk.Checkbutton(
            params_frame,
            text="Exclude near-black",
            variable=self.merge_ct_exclude_black_var,
            command=self._preview_merge_command,
        )
        self.merge_ct_exclude_black_check.grid(row=2, column=10, sticky="w", padx=(12, 0), pady=(8, 0))

        encode_frame = ttk.LabelFrame(parent, text="Encoding Args (inherited)", padding=8)
        encode_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(9, weight=1)

        self.merge_override_check = ttk.Checkbutton(
            encode_frame,
            text="Override",
            variable=self.merge_encode_override_var,
            command=self._on_merge_override_toggle,
        )
        self.merge_override_check.grid(row=0, column=0, sticky="w")

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.merge_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.merge_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.merge_codec_entry.grid(row=0, column=2, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Quality (CRF/QP):").grid(row=0, column=3, sticky="w")
        self.merge_crf_entry = ttk.Entry(encode_frame, textvariable=self.merge_crf_var, width=6)
        self.merge_crf_entry.grid(row=0, column=4, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="Preset:").grid(row=0, column=5, sticky="w")
        self.merge_preset_entry = ttk.Entry(encode_frame, textvariable=self.merge_preset_var, width=10)
        self.merge_preset_entry.grid(row=0, column=6, sticky="w", padx=(6, 12))

        ttk.Label(encode_frame, text="PixFmt:").grid(row=0, column=7, sticky="w")
        self.merge_pixfmt_entry = ttk.Entry(encode_frame, textvariable=self.merge_pix_fmt_var, width=10)
        self.merge_pixfmt_entry.grid(row=0, column=8, sticky="w", padx=(6, 0))

        ttk.Label(encode_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.merge_extra_ffmpeg_entry = ttk.Entry(
            encode_frame, textvariable=self.merge_extra_ffmpeg_args_var
        )
        self.merge_extra_ffmpeg_entry.grid(
            row=1, column=1, columnspan=9, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.merge_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=11, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.merge_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_merge_command
        )
        self.merge_preview_btn.grid(row=0, column=0, padx=(0, 6))
        self.merge_mask_run_btn = ttk.Button(
            buttons, text="Run Mask", command=self._run_merge_mask_placeholder
        )
        self.merge_mask_run_btn.grid(row=0, column=1, padx=6)
        self.merge_csv_btn = ttk.Button(
            buttons, text="Create AutoCT CSV", command=self._start_merge_autoct_csv
        )
        self.merge_csv_btn.grid(row=0, column=2, padx=6)
        self.merge_run_btn = ttk.Button(
            buttons, text="Run Merging", command=self._run_merge_placeholder
        )
        self.merge_run_btn.grid(row=0, column=3, padx=6)
        self.merge_mask_verify_quick_btn = ttk.Button(
            buttons, text="Verify Mask (Quick)", command=self._start_merge_mask_verify_quick
        )
        self.merge_mask_verify_quick_btn.grid(row=0, column=4, padx=6)
        self.merge_mask_verify_deep_btn = ttk.Button(
            buttons, text="Verify Mask (Deep)", command=self._start_merge_mask_verify_deep
        )
        self.merge_mask_verify_deep_btn.grid(row=0, column=5, padx=6)
        self.merge_verify_quick_btn = ttk.Button(
            buttons, text="Verify Merge (Quick)", command=self._start_merge_verify_quick
        )
        self.merge_verify_quick_btn.grid(row=0, column=6, padx=6)
        self.merge_verify_deep_btn = ttk.Button(
            buttons, text="Verify Merge (Deep)", command=self._start_merge_verify_deep
        )
        self.merge_verify_deep_btn.grid(row=0, column=7, padx=6)
        self.merge_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_merge_placeholder, state=tk.DISABLED
        )
        self.merge_stop_btn.grid(row=0, column=8, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_merge_log).grid(
            row=0, column=9, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=12, column=0, columnspan=3, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(2, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.merge_status_var).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        self.merge_progress = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.merge_progress_var,
            maximum=100.0,
        )
        self.merge_progress.grid(row=0, column=2, sticky="ew", padx=4)

        log_frame = ttk.LabelFrame(parent, text="Merging Log", padding=6)
        log_frame.grid(row=13, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.merge_log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.merge_log_text.grid(row=0, column=0, sticky="nsew")
        mscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.merge_log_text.yview)
        mscroll.grid(row=0, column=1, sticky="ns")
        self.merge_log_text.configure(yscrollcommand=mscroll.set)

        self._merge_auto_widgets = [
            self.merge_autoct_workers_entry,
            self.merge_parallel_workers_entry,
            self.merge_mask_formerge_workers_entry,
            self.merge_use_gpu_check,
            self.merge_output_format_combo,
            self.merge_chunks_entry,
        ]
        self._merge_manual_entry_widgets = [
            self.merge_mask_binarize_entry,
            self.merge_mask_dilate_entry,
            self.merge_mask_blur_entry,
            self.merge_shadow_length_entry,
            self.merge_shadow_curve_entry,
        ]
        self._merge_manual_check_widgets = [
            self.merge_shadow_motion_enabled_check,
            self.merge_dynamic_shadow_check,
            self.merge_use_replace_mask_check,
            self.merge_ct_exclude_black_check,
        ]
        self._merge_manual_combo_widgets = [
            self.merge_ct_preset_combo,
            self.merge_ct_auto_mode_combo,
        ]

        self._on_merge_mode_changed()
        self._on_merge_workers_changed()
        self._on_merge_override_toggle(initial=True)
        self._update_replace_mask_dependent_controls()
        self._preview_merge_command()
        self._set_merge_running(False)

    def _open_merge_inpainted_folder(self) -> None:
        folder = self.merge_inpainted_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"Merge inpainted folder ready: {folder}")

    def _open_merge_splatted_folder(self) -> None:
        folder = self.merge_splatted_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"Merge splatted folder ready: {folder}")

    def _open_merge_original_folder(self) -> None:
        folder = self.merge_original_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"Merge original folder ready: {folder}")

    def _open_merge_replace_mask_folder(self) -> None:
        folder = self.merge_replace_mask_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"Merge replace mask folder ready: {folder}")

    def _open_merge_mask_formerge_folder(self) -> None:
        folder = self.merge_mask_formerge_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"Merge mask-for-merge folder ready: {folder}")

    def _open_merge_output_folder(self) -> None:
        folder = self.merge_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"Merge output folder ready: {folder}")

    def _open_merge_autoct_csv_folder(self) -> None:
        csv_path = self.merge_autoct_csv_var.get().strip()
        if not csv_path:
            return
        folder = str(Path(csv_path).resolve().parent)
        os.makedirs(folder, exist_ok=True)
        self._append_merge_log(f"AutoCT CSV folder ready: {folder}")

    def _on_merge_mode_changed(self, _event=None) -> None:
        mode = self.merge_mode_var.get().strip()
        if mode == "Manual":
            self.merge_info_text_var.set(self.MERGE_MANUAL_INFO)
        else:
            self.merge_mode_var.set("Auto (recommended)")
            self.merge_info_text_var.set(self.MERGE_AUTO_INFO)
            self.merge_autoct_workers_var.set("8")
            self.merge_mask_formerge_workers_var.set(
                self.merge_autoct_workers_var.get().strip() or "8"
            )
            self.merge_parallel_workers_var.set("2")
            self.merge_use_gpu_var.set(False)
            self.merge_output_format_var.set("Full SBS")
            self.merge_chunks_var.set("20")
            self.merge_mask_binarize_var.set("0.5")
            self.merge_mask_dilate_var.set("2")
            self.merge_mask_blur_var.set("4")
            self.merge_shadow_length_var.set("25")
            self.merge_shadow_curve_var.set("0")
            self.merge_shadow_motion_enabled_var.set(True)
            self.merge_dynamic_shadow_width_var.set(True)
            self.merge_use_replace_mask_var.set(True)
            self.merge_ct_preset_var.set("1")
            self.merge_ct_auto_mode_var.set("CSV Blend")
            self.merge_ct_exclude_black_var.set(True)
        self._apply_merge_control_states()
        self._on_merge_workers_changed()
        self._refresh_pipeline_status_panel()

    def _on_merge_ct_auto_mode_changed(self, _event=None) -> None:
        self._preview_merge_command()
        self._refresh_pipeline_status_panel()

    def _get_merge_worker_count(self) -> int:
        workers_raw = self.merge_parallel_workers_var.get().strip()
        try:
            workers = max(1, int(workers_raw))
        except Exception:
            workers = 2
        if str(workers) != workers_raw:
            self.merge_parallel_workers_var.set(str(workers))
        self.merge_parallel_var.set(workers >= 2)
        return workers

    def _on_merge_workers_changed(self, _event=None) -> None:
        self._get_merge_worker_count()
        self.merge_parallel_workers_entry.configure(state=tk.NORMAL)
        self._preview_merge_command()

    def _apply_merge_control_states(self) -> None:
        mode_manual = self.merge_mode_var.get().strip() == "Manual"
        for widget in getattr(self, "_merge_auto_widgets", []):
            widget.configure(state=tk.NORMAL if not isinstance(widget, ttk.Combobox) else "readonly")

        entry_state = tk.NORMAL if mode_manual else tk.DISABLED
        combo_state = "readonly" if mode_manual else tk.DISABLED
        check_state = tk.NORMAL if mode_manual else tk.DISABLED
        for widget in getattr(self, "_merge_manual_entry_widgets", []):
            widget.configure(state=entry_state)
        for widget in getattr(self, "_merge_manual_combo_widgets", []):
            widget.configure(state=combo_state)
        for widget in getattr(self, "_merge_manual_check_widgets", []):
            widget.configure(state=check_state)

        # Replace-mask is mandatory in strict noGUI merge flow.
        self.merge_use_replace_mask_var.set(True)
        self.merge_use_replace_mask_check.configure(state=tk.DISABLED)

        self._on_merge_workers_changed()
        self._preview_merge_command()
        self._refresh_pipeline_status_panel()

    def _sync_merge_encoding_from_scene(self) -> None:
        self.merge_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.scene_codec_var.get(),
                self.DEFAULT_SCENE_CODEC,
            )
        )
        self.merge_crf_var.set(self.scene_crf_var.get().strip())
        self.merge_preset_var.set(self.scene_encoder_preset_var.get().strip())
        self.merge_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip())

    def _on_merge_override_toggle(self, initial: bool = False) -> None:
        enabled = bool(self.merge_encode_override_var.get())
        if not enabled:
            self._sync_merge_encoding_from_scene()
        elif not initial and not self._merge_override_notice_shown:
            self._merge_override_notice_shown = True
            messagebox.showwarning("Merging Encode Override", self.MERGE_OVERRIDE_WARNING)

        state = tk.NORMAL if enabled else tk.DISABLED
        self._set_codec_widget_override_state(self.merge_codec_entry, enabled)
        for widget in (
            self.merge_crf_entry,
            self.merge_preset_entry,
            self.merge_pixfmt_entry,
            self.merge_extra_ffmpeg_entry,
        ):
            widget.configure(state=state)
        self._preview_merge_command()

    def _build_merge_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        output_dir = self.merge_output_var.get().strip()
        workers = self._get_merge_worker_count()
        use_parallel = workers >= 2
        ct_mode = self.merge_ct_auto_mode_var.get().strip() or "CSV Blend"
        output_format_ui = self.merge_output_format_var.get().strip()
        output_format_runner = (
            "Half SBS (Left-Right)"
            if output_format_ui == "Half SBS"
            else "Full SBS (Left-Right)"
        )
        codec_value = self._normalize_ffmpeg_codec(
            self.merge_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        self.merge_codec_var.set(codec_value)
        env_updates: dict[str, str] = {
            "PYTHON": sys.executable,
            "RUNNER": "merging_nogui_batch_parallel.py" if use_parallel else "merging_nogui_batch.py",
            "INPAINTED_FOLDER": self.merge_inpainted_var.get().strip(),
            "SPLATTED_FOLDER": self.merge_splatted_var.get().strip(),
            "ORIGINAL_FOLDER": self.merge_original_var.get().strip(),
            "OUTPUT_FOLDER": output_dir,
            "REPLACE_MASK_FOLDER": self.merge_replace_mask_var.get().strip(),
            "PREPROCESSED_MASK_FOLDER": self.merge_mask_formerge_var.get().strip(),
            "CT_PRESET": self.merge_ct_preset_var.get().strip() or "1",
            "CT_AUTO_MODE": ct_mode,
            "CT_CSV_BLEND_PATH": self.merge_autoct_csv_var.get().strip(),
            "OUTPUT_FORMAT": output_format_runner,
            "CHUNK_SIZE": self.merge_chunks_var.get().strip() or "20",
            "USE_GPU": "1" if self.merge_use_gpu_var.get() else "0",
            "CT_EXCLUDE_BLACK_IN_TARGET": "1" if self.merge_ct_exclude_black_var.get() else "0",
            "CT_STRENGTH": "1",
            "CT_BLACK_THRESH": "0",
            "CT_MIN_VALID_RATIO": "0",
            "CT_MIN_VALID": "0",
            "CT_RING_WIDTH": "20",
            "CT_CLAMP_L_MIN": "0.1",
            "CT_CLAMP_L_MAX": "2",
            "CT_CLAMP_AB_MIN": "0.1",
            "CT_CLAMP_AB_MAX": "3",
            "PAD_TO_16_9": "0",
            "ADD_BORDERS": "0",
            "STOP_MARKER": os.path.join(output_dir or "./work/sbs", ".stop_after_current"),
            "MERGE_DEBUG": "0",
            "FFMPEG_CODEC": codec_value,
            "FFMPEG_CRF": self.merge_crf_var.get().strip() or "1",
            "FFMPEG_PRESET": self.merge_preset_var.get().strip(),
            "FFMPEG_PIX_FMT": self.merge_pix_fmt_var.get().strip(),
            "FFMPEG_EXTRA_ARGS": self.merge_extra_ffmpeg_args_var.get().strip(),
        }
        if use_parallel:
            env_updates["WORKERS"] = str(workers)
        cmd = ["bash", "run_merging_nogui_batch_parallel.sh" if use_parallel else "run_merging_nogui_batch.sh"]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview

    def _preview_merge_command(self) -> None:
        try:
            _cmd, _env, preview = self._build_merge_runner_payload()
            self.merge_cmd_preview_var.set(preview)
        except Exception as e:
            self.merge_cmd_preview_var.set(f"Invalid options: {e}")

    def _append_merge_log(self, line: str) -> None:
        self.merge_log_text.configure(state=tk.NORMAL)
        self.merge_log_text.insert(tk.END, line + "\n")
        self.merge_log_text.see(tk.END)
        self.merge_log_text.configure(state=tk.DISABLED)

    def _clear_merge_log(self) -> None:
        self.merge_log_text.configure(state=tk.NORMAL)
        self.merge_log_text.delete("1.0", tk.END)
        self.merge_log_text.configure(state=tk.DISABLED)

    def _build_mask_formerge_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        motion_enabled = bool(self.merge_shadow_motion_enabled_var.get())
        mask_workers_raw = self.merge_mask_formerge_workers_var.get().strip()
        try:
            mask_workers = max(1, int(mask_workers_raw))
        except Exception:
            autoct_raw = self.merge_autoct_workers_var.get().strip()
            try:
                mask_workers = max(1, int(autoct_raw))
            except Exception:
                mask_workers = 8
        if str(mask_workers) != mask_workers_raw:
            self.merge_mask_formerge_workers_var.set(str(mask_workers))
        env_updates: dict[str, str] = {
            "PYTHON": sys.executable,
            "RUNNER": "mask_formerge_nogui.py",
            "REPLACE_MASK_FOLDER": self.merge_replace_mask_var.get().strip(),
            "OUTPUT_FOLDER": self.merge_mask_formerge_var.get().strip(),
            "INPUT_GLOB": "*_replace_mask.*",
            "WORKERS": str(mask_workers),
            "CHUNK_SIZE": self.merge_chunks_var.get().strip() or "20",
            "USE_GPU": "1" if bool(self.merge_use_gpu_var.get()) else "0",
            "MASK_BINARIZE_THRESHOLD": self.merge_mask_binarize_var.get().strip() or "0.5",
            "MASK_DILATE_KERNEL_SIZE": self.merge_mask_dilate_var.get().strip() or "2",
            "MASK_BLUR_KERNEL_SIZE": self.merge_mask_blur_var.get().strip() or "4",
            "SHADOW_LENGTH_PX": self.merge_shadow_length_var.get().strip() or "25",
            "SHADOW_CURVE": self.merge_shadow_curve_var.get().strip() or "0",
            "SHADOW_MOTION_GAIN": "1" if motion_enabled else "0",
            "SHADOW_MOTION_DEADZONE_PX": "20",
            "SHADOW_MOTION_CHAIN_ENABLED": "1" if motion_enabled else "0",
            "SHADOW_WIDTH_ADAPTIVE": "1" if bool(self.merge_dynamic_shadow_width_var.get()) else "0",
            "SKIP_EXISTING": "1",
        }
        cmd = ["/usr/bin/env", "bash", "run_mask_formerge_nogui.sh"]
        preview = (
            " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
            + " "
            + " ".join(shlex.quote(x) for x in cmd)
        )
        return cmd, env_updates, preview

    def _run_merge_mask_placeholder(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Merging", "Another merging task is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Merging", "Stop verification before running Mask-for-merge.")
            return
        try:
            cmd, env_updates, _preview = self._build_mask_formerge_runner_payload()
        except Exception as exc:
            messagebox.showerror("Merging", f"Invalid mask-for-merge options:\n{exc}")
            return

        launcher_script = Path("run_mask_formerge_nogui.sh").resolve()
        if not launcher_script.is_file():
            messagebox.showerror("Merging", f"Launcher not found:\n{launcher_script}")
            return

        replace_mask_dir = env_updates.get("REPLACE_MASK_FOLDER", "").strip()
        if not replace_mask_dir or not os.path.isdir(replace_mask_dir):
            messagebox.showerror(
                "Merging",
                f"Replace-mask folder not found:\n{replace_mask_dir or '(empty)'}",
            )
            return

        output_dir = env_updates.get("OUTPUT_FOLDER", "").strip()
        if not output_dir:
            messagebox.showerror("Merging", "Mask-for-merge output folder is required.")
            return
        os.makedirs(output_dir, exist_ok=True)

        self._merge_stop_requested = False
        self._merge_stop_clicks = 0
        self.merge_status_var.set("Running Mask-for-merge...")
        self.merge_progress_var.set(0.0)
        self._set_merge_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("mask_for_merge")
        self._append_merge_log("=== Mask-for-merge started ===")
        self._append_merge_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_merge_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._merge_thread = threading.Thread(
            target=self._run_merge_mask_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._merge_thread.start()

    def _run_merge_mask_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        step_success = False
        try:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._merge_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", f"[MASK] {line}"))
                    self._try_parse_merge_progress(line)
                if self._merge_stop_requested:
                    break
            rc = proc.wait()
            if self._merge_stop_requested:
                self._log_queue.put(("merge_status", "Mask-for-merge stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("merge_status", "Mask-for-merge completed"))
                self._log_queue.put(("merge_progress", "100"))
            else:
                self._log_queue.put(("merge_status", f"Mask-for-merge failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("merge_line", f"[MASK][ERROR] {exc}"))
            self._log_queue.put(("merge_status", "Mask-for-merge failed"))
        finally:
            self._merge_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("merge_done", {"step": "mask_for_merge", "success": step_success}))

    def _run_merge_placeholder(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Merging", "Merging is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Merging", "Stop verification before starting Merging.")
            return
        try:
            cmd, env_updates, _preview = self._build_merge_runner_payload()
        except Exception as exc:
            messagebox.showerror("Merging", f"Invalid merging options:\n{exc}")
            return

        launcher_script = Path(cmd[1]).resolve() if len(cmd) > 1 else Path("run_merging_nogui_batch.sh").resolve()
        if not launcher_script.is_file():
            messagebox.showerror("Merging", f"Launcher not found:\n{launcher_script}")
            return

        for key, label in (
            ("INPAINTED_FOLDER", "Inpainted"),
            ("SPLATTED_FOLDER", "Splatted"),
            ("ORIGINAL_FOLDER", "Original"),
        ):
            path = env_updates.get(key, "").strip()
            if not path or not os.path.isdir(path):
                messagebox.showerror("Merging", f"{label} folder not found:\n{path or '(empty)'}")
                return
        output_dir = env_updates.get("OUTPUT_FOLDER", "").strip()
        if not output_dir:
            messagebox.showerror("Merging", "Output folder is required.")
            return
        os.makedirs(output_dir, exist_ok=True)
        mask_dir = env_updates.get("REPLACE_MASK_FOLDER", "").strip()
        if not mask_dir or not os.path.isdir(mask_dir):
            messagebox.showerror("Merging", f"Replace-mask folder not found:\n{mask_dir or '(empty)'}")
            return
        missing = self._find_missing_replace_masks(
            self.merge_splatted_var.get().strip(),
            mask_dir,
        )
        if missing:
            show = "\n".join(missing[:12])
            more = "" if len(missing) <= 12 else f"\n... and {len(missing) - 12} more"
            messagebox.showerror(
                "Merging Replace Mask",
                (
                    "Missing replace-mask files detected.\n"
                    "Strict noGUI merge mode requires full replace-mask coverage.\n\n"
                    f"Missing examples:\n{show}{more}"
                ),
            )
            return

        mask_formerge_dir = env_updates.get("PREPROCESSED_MASK_FOLDER", "").strip()
        if not mask_formerge_dir or not os.path.isdir(mask_formerge_dir):
            messagebox.showerror(
                "Merging",
                (
                    "Mask-for-merge folder not found.\n"
                    "This pipeline requires preprocessed masks.\n\n"
                    f"{mask_formerge_dir or '(empty)'}"
                ),
            )
            return
        missing_formerge = self._find_missing_replace_masks(
            self.merge_splatted_var.get().strip(),
            mask_formerge_dir,
        )
        if missing_formerge:
            show = "\n".join(missing_formerge[:12])
            more = "" if len(missing_formerge) <= 12 else f"\n... and {len(missing_formerge) - 12} more"
            messagebox.showerror(
                "Merging Mask-for-Merge",
                (
                    "Missing preprocessed mask_for_merge files detected.\n"
                    "Strict noGUI merge mode requires full mask_for_merge coverage.\n\n"
                    f"Missing examples:\n{show}{more}"
                ),
            )
            return

        if env_updates.get("CT_AUTO_MODE", "CSV Blend") == "CSV Blend":
            csv_path = self.merge_autoct_csv_var.get().strip()
            if not csv_path or not Path(csv_path).is_file():
                if not self._pipeline_test_active:
                    self._pipeline_set_completed("autoct_csv", False)
                    self._pipeline_set_verified("autoct_csv", "none")
                    self._refresh_pipeline_status_panel()
                    self._save_pipeline_state()
                messagebox.showwarning(
                    "Merging",
                    (
                        f"autoct.csv not found:\n{csv_path or '(empty)'}\n\n"
                        "CSV Blend mode requires this file.\n"
                        "AutoCT CSV will be regenerated now before Merging."
                    ),
                )
                self.merge_status_var.set("AutoCT CSV missing, rebuilding...")
                self._start_merge_autoct_csv()
                return
            try:
                autoct_ok, autoct_msg, incomplete_scenes = self._verify_autoct_csv_packet_coverage(
                    inpainted_dir=env_updates.get("INPAINTED_FOLDER", "").strip(),
                    splatted_dir=env_updates.get("SPLATTED_FOLDER", "").strip(),
                    replace_mask_dir=mask_dir,
                    csv_path=csv_path,
                    cleanup_incomplete=True,
                )
            except Exception as exc:
                messagebox.showerror(
                    "Merging",
                    f"AutoCT CSV packet verification failed:\n{exc}",
                )
                return
            if not autoct_ok:
                if not self._pipeline_test_active:
                    self._pipeline_set_completed("autoct_csv", False)
                    self._pipeline_set_verified("autoct_csv", "none")
                    self._refresh_pipeline_status_panel()
                    self._save_pipeline_state()
                self._append_merge_log(f"[AUTOCT][VERIFY] {autoct_msg}")
                if incomplete_scenes:
                    show = "\n".join(incomplete_scenes[:12])
                    more = (
                        ""
                        if len(incomplete_scenes) <= 12
                        else f"\n... and {len(incomplete_scenes) - 12} more"
                    )
                    self._append_merge_log(
                        f"[AUTOCT][VERIFY] Incomplete scenes:\n{show}{more}"
                    )
                messagebox.showwarning(
                    "Merging",
                    (
                        f"{autoct_msg}\n\n"
                        "AutoCT CSV will be regenerated now before Merging."
                    ),
                )
                self.merge_status_var.set("AutoCT CSV incomplete, rebuilding...")
                self._start_merge_autoct_csv()
                return
            if not self._pipeline_test_active:
                self._pipeline_set_completed("autoct_csv", True)
                self._pipeline_set_verified("autoct_csv", "none")
                self._refresh_pipeline_status_panel()
                self._save_pipeline_state()

        self._merge_stop_requested = False
        self._merge_stop_clicks = 0
        self.merge_status_var.set("Starting...")
        self.merge_progress_var.set(0.0)
        self._set_merge_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("merging")
        self._append_merge_log("=== Merging started ===")
        self._append_merge_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_merge_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._merge_thread = threading.Thread(
            target=self._run_merge_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._merge_thread.start()

    def _run_merge_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        step_success = False
        try:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._merge_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", line))
                    self._try_parse_merge_progress(line)
                if self._merge_stop_requested:
                    break
            rc = proc.wait()
            if self._merge_stop_requested:
                self._log_queue.put(("merge_status", "Stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("merge_status", "Completed"))
                self._log_queue.put(("merge_progress", "100"))
            else:
                self._log_queue.put(("merge_status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("merge_line", f"[MERGE][ERROR] {exc}"))
            self._log_queue.put(("merge_status", "Failed"))
        finally:
            self._merge_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("merge_done", {"step": "merging", "success": step_success}))

    def _start_merge_autoct_csv(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Merging", "Another merging task is running.")
            return
        if self._verify_running:
            messagebox.showinfo("Merging", "Stop verification before creating autoct.csv.")
            return

        script_path = Path("Utilities/analyze_auto_ct_csv.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Merging", f"Script not found:\n{script_path}")
            return

        inpainted = self.merge_inpainted_var.get().strip()
        splatted = self.merge_splatted_var.get().strip()
        original = self.merge_original_var.get().strip()
        out_csv = self.merge_autoct_csv_var.get().strip()
        mask_folder = self.merge_replace_mask_var.get().strip()
        if not inpainted or not os.path.isdir(inpainted):
            messagebox.showerror("Merging", f"Inpainted folder not found:\n{inpainted or '(empty)'}")
            return
        if not splatted or not os.path.isdir(splatted):
            messagebox.showerror("Merging", f"Splatted folder not found:\n{splatted or '(empty)'}")
            return
        if not original or not os.path.isdir(original):
            messagebox.showerror("Merging", f"Original folder not found:\n{original or '(empty)'}")
            return
        if not out_csv:
            messagebox.showerror("Merging", "AutoCT CSV output path is required.")
            return
        if not mask_folder or not os.path.isdir(mask_folder):
            messagebox.showerror(
                "Merging",
                (
                    "Replace-mask folder not found.\n"
                    "AutoCT CSV requires replace masks from Splatting.\n\n"
                    f"{mask_folder or '(empty)'}"
                ),
            )
            return
        if not self._has_any_replace_masks(mask_folder):
            messagebox.showerror(
                "Merging",
                (
                    "No replace-mask files found.\n"
                    "AutoCT CSV cannot run without replace masks."
                ),
            )
            return
        missing = self._find_missing_replace_masks(splatted, mask_folder)
        if missing:
            show = "\n".join(missing[:8])
            more = "" if len(missing) <= 8 else f"\n... and {len(missing) - 8} more"
            messagebox.showerror(
                "Merging Replace Mask",
                (
                    "Missing replace-mask files detected.\n"
                    "AutoCT CSV requires complete replace-mask coverage.\n\n"
                    f"Missing examples:\n{show}{more}"
                ),
            )
            return
        out_csv_path = Path(out_csv).resolve()
        os.makedirs(str(out_csv_path.parent), exist_ok=True)
        self.merge_autoct_csv_var.set(str(out_csv_path))

        try:
            workers = max(1, int(self.merge_autoct_workers_var.get().strip() or "8"))
        except Exception:
            workers = 8
        cmd = [
            sys.executable,
            str(script_path),
            "--inpainted-folder",
            str(Path(inpainted).resolve()),
            "--splatted-folder",
            str(Path(splatted).resolve()),
            "--original-folder",
            str(Path(original).resolve()),
            "--output-csv",
            str(out_csv_path),
            "--workers",
            str(workers),
            "--use-replace-mask",
            "--replace-mask-folder",
            str(Path(mask_folder).resolve()),
            "--resume-validate-packets",
        ]

        self._merge_stop_requested = False
        self._merge_stop_clicks = 0
        self.merge_status_var.set("Creating autoct.csv...")
        self.merge_progress_var.set(0.0)
        self._set_merge_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("autoct_csv")
        self._append_merge_log("=== AutoCT CSV creation started ===")
        self._append_merge_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._merge_thread = threading.Thread(
            target=self._run_merge_autoct_worker,
            args=(cmd, str(out_csv_path)),
            daemon=True,
        )
        self._merge_thread.start()

    def _run_merge_autoct_worker(self, cmd: list[str], out_csv: str) -> None:
        proc = None
        step_success = False
        try:
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=preexec,
            )
            self._merge_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", f"[AUTOCT] {line}"))
            rc = proc.wait()
            if self._merge_stop_requested:
                self._log_queue.put(("merge_status", "AutoCT CSV stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("merge_status", "AutoCT CSV created"))
                self._log_queue.put(("merge_progress", "100"))
                self._log_queue.put(("merge_line", f"[AUTOCT] output: {out_csv}"))
            else:
                self._log_queue.put(("merge_status", f"AutoCT CSV failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("merge_line", f"[AUTOCT][ERROR] {exc}"))
            self._log_queue.put(("merge_status", "AutoCT CSV failed"))
        finally:
            self._merge_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("merge_done", {"step": "autoct_csv", "success": step_success}))

    def _stop_merge_placeholder(self, prompt_user: bool = True) -> None:
        running = bool(self._merge_thread and self._merge_thread.is_alive())
        if not running:
            return
        if self._merge_stop_clicks == 0 and prompt_user:
            messagebox.showwarning(
                "Stop Merging",
                "Graceful stop requested.\n\n"
                "Current process will be interrupted like Ctrl+C.\n"
                "Click Stop again to force kill immediately.",
            )
        self._merge_stop_requested = True
        self._merge_stop_clicks += 1

        if self._merge_stop_clicks == 1:
            self.merge_status_var.set("Graceful stop requested...")
            self._append_merge_log(
                "[STOP] graceful stop requested (click Stop again for immediate force stop)."
            )
            self.merge_stop_btn.configure(text="Force Stop")
        else:
            self.merge_status_var.set("Force stop requested...")
            self._append_merge_log("[STOP] force stop requested.")

        self._send_merge_signal(signal.SIGINT)
        if self._merge_stop_clicks >= 2:
            self.root.after(1000, self._force_kill_merge)

    def _send_merge_signal(self, sig: int) -> None:
        proc = self._merge_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception as exc:
            self._append_merge_log(f"Signal send failed: {exc}")

    def _force_kill_merge(self) -> None:
        proc = self._merge_process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                self._append_merge_log("Merging process force-killed.")
            except Exception as exc:
                self._append_merge_log(f"Merging kill failed: {exc}")

    def _set_merge_running(self, is_running: bool) -> None:
        self.merge_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.merge_csv_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.merge_mask_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.merge_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.merge_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        verify_state = tk.DISABLED if (is_running or self._verify_running) else tk.NORMAL
        self.merge_mask_verify_quick_btn.configure(state=verify_state)
        self.merge_mask_verify_deep_btn.configure(state=verify_state)
        self.merge_verify_quick_btn.configure(state=verify_state)
        self.merge_verify_deep_btn.configure(state=verify_state)
        if is_running:
            self.merge_stop_btn.configure(text="Stop")
        else:
            self.merge_stop_btn.configure(text="Stop")
            self._merge_stop_clicks = 0
            self._merge_stop_requested = False
        self._update_replace_mask_dependent_controls()

    def _try_parse_merge_progress(self, line: str) -> None:
        m = re.search(r"^\[(?:RUN|OK|SKIP|ERR)\s*\]\s*(\d+)\s*/\s*(\d+)", line)
        if m:
            try:
                idx = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    prog = max(0.0, min(100.0, (idx / total) * 100.0))
                    self._log_queue.put(("merge_progress", str(prog)))
            except Exception:
                pass
            return
        if line.startswith("[DONE]"):
            self._log_queue.put(("merge_progress", "100"))

    def _validate_merge_mask_verify_inputs(self) -> tuple[bool, str, str, list[str]]:
        mask_dir = self.merge_mask_formerge_var.get().strip()
        if not mask_dir:
            messagebox.showerror("Verify Mask", "Mask-for-merge folder is required.")
            return False, "", "", []
        if not os.path.isdir(mask_dir):
            messagebox.showerror("Verify Mask", f"Mask-for-merge folder not found:\n{mask_dir}")
            return False, "", "", []
        ok_ref, ref_dir, ref_patterns, ref_kind = self._resolve_verify_reference(
            "merge_mask", "Verify Mask"
        )
        if not ok_ref:
            return False, "", "", []
        mask_dir = self._pipeline_prepare_verify_subset_dir(
            mask_dir, "merge_mask_target", list(self.VERIFY_VIDEO_PATTERNS)
        )
        self._append_merge_log(f"[VERIFY-MASK] reference source: {ref_kind} ({ref_dir})")
        return True, mask_dir, ref_dir, ref_patterns

    def _start_merge_mask_verify_quick(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Verify Mask", "Stop Merging before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Mask", "Another verification is already running.")
            return
        ok, mask_dir, ref_dir, ref_patterns = self._validate_merge_mask_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Mask", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="merge_mask_quick")
        self.merge_status_var.set("Verify Mask (Quick) running...")
        self._append_merge_log("=== Verify Mask (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_merge_mask_verify_quick_worker,
            args=(mask_dir, ref_dir, ref_patterns),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_merge_mask_verify_quick_worker(
        self, mask_dir: str, ref_dir: str, ref_patterns: list[str]
    ) -> None:
        try:
            out_files: list[str] = []
            for ext in ("*.mkv", "*.mp4", "*.mov", "*.avi", "*.webm"):
                out_files.extend([str(p) for p in Path(mask_dir).glob(ext) if p.is_file()])
            out_files = sorted(set(out_files))
            ref_files = self._collect_files_for_patterns(ref_dir, ref_patterns)
            if not out_files:
                self._log_queue.put(("merge_mask_verify_quick_result", {
                    "ok": False,
                    "message": "No video files found in mask-for-merge folder.",
                    "broken_output": [],
                    "broken_reference": [],
                }))
                return
            if not ref_files:
                self._log_queue.put(("merge_mask_verify_quick_result", {
                    "ok": False,
                    "message": "No reference video files found in selected reference folder.",
                    "broken_output": [],
                    "broken_reference": [],
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            self._log_queue.put(
                ("merge_line", f"[MASK-QUICK] checking mask files={len(out_files)} and reference files={len(ref_files)} with {max_workers} workers")
            )

            def _probe_group(file_list: list[str], label: str) -> dict:
                broken: list[str] = []
                total_duration = 0.0
                duration_available = True
                total_frames = 0
                frames_available = True

                def _probe_one(fp: str) -> tuple[str, dict]:
                    return fp, self._probe_video_basic(fp)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_probe_one, fp) for fp in file_list]
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        fp, meta = fut.result()
                        done += 1
                        if not meta.get("ok"):
                            broken.append(fp)
                            self._log_queue.put(("merge_line", f"[MASK-QUICK][{label.upper()}][BROKEN] {fp} :: {meta.get('error')}"))
                        else:
                            dur = meta.get("duration")
                            frm = meta.get("frames")
                            if dur is None:
                                duration_available = False
                            else:
                                total_duration += float(dur)
                            if frm is None:
                                frames_available = False
                            else:
                                total_frames += int(frm)
                        if done % 25 == 0 or done == len(file_list):
                            self._log_queue.put(("merge_line", f"[MASK-QUICK][{label.upper()}] progress {done}/{len(file_list)}"))
                return {
                    "broken": broken,
                    "total_duration": total_duration,
                    "duration_available": duration_available,
                    "total_frames": total_frames,
                    "frames_available": frames_available,
                }

            out_stats = _probe_group(out_files, "mask")
            ref_stats = _probe_group(ref_files, "reference")

            count_ok = len(out_files) == len(ref_files)
            count_msg = f"mask={len(out_files)} vs reference={len(ref_files)}"

            duration_ok = False
            duration_msg = "n.d."
            if out_stats["duration_available"] and ref_stats["duration_available"]:
                dd = abs(float(out_stats["total_duration"]) - float(ref_stats["total_duration"]))
                duration_ok = dd <= 0.35
                duration_msg = (
                    f"mask={float(out_stats['total_duration']):.3f}s vs "
                    f"reference={float(ref_stats['total_duration']):.3f}s (delta={dd:.3f}s)"
                )

            frames_ok = False
            frames_msg = "n.d."
            if out_stats["frames_available"] and ref_stats["frames_available"]:
                df = abs(int(out_stats["total_frames"]) - int(ref_stats["total_frames"]))
                frames_ok = df <= 1
                frames_msg = (
                    f"mask={int(out_stats['total_frames'])} vs "
                    f"reference={int(ref_stats['total_frames'])} (delta={df})"
                )

            self._log_queue.put(("merge_line", f"[MASK-QUICK] file count check: {count_msg}"))
            self._log_queue.put(("merge_line", f"[MASK-QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("merge_line", f"[MASK-QUICK] packet check: {frames_msg}"))

            ok_final = (
                not out_stats["broken"]
                and not ref_stats["broken"]
                and count_ok
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Mask quick verify completed.\n"
                f"Broken mask files: {len(out_stats['broken'])}\n"
                f"Broken reference files: {len(ref_stats['broken'])}\n"
                f"File count match: {'YES' if count_ok else 'NO'} ({count_msg})\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(
                (
                    "merge_mask_verify_quick_result",
                    {
                        "ok": ok_final,
                        "message": message,
                        "broken_output": out_stats["broken"],
                        "broken_reference": ref_stats["broken"],
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(("merge_mask_verify_quick_result", {
                "ok": False,
                "message": f"Mask quick verify failed: {type(e).__name__}: {e}",
                "broken_output": [],
                "broken_reference": [],
            }))
        finally:
            self._log_queue.put(("verify_done", "merge_mask_quick"))

    def _start_merge_mask_verify_deep(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Verify Mask", "Stop Merging before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Mask", "Another verification is already running.")
            return
        ok, mask_dir, ref_dir, _ref_patterns = self._validate_merge_mask_verify_inputs()
        if not ok:
            return

        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Mask", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(mask_dir).resolve()),
            str(Path(ref_dir).resolve()),
            "--extensions",
            self.VERIFY_ALL_VIDEO_EXTENSIONS,
            "--workers",
            str(workers),
            "--probe-timeout-sec",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
            "--probe-timeout-retries",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
            "--delete",
            "yes",
            "--no-single-line-progress",
        ]

        self._set_verify_running(True, mode="merge_mask_deep")
        self.merge_status_var.set("Verify Mask (Deep) running...")
        self._append_merge_log("=== Verify Mask (Deep) started ===")
        self._append_merge_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))

        self._verify_thread = threading.Thread(
            target=self._run_merge_mask_verify_deep_worker,
            args=(cmd, str(Path(mask_dir).resolve())),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_merge_mask_verify_deep_worker(self, cmd: list[str], mask_dir: str) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=(os.setsid if hasattr(os, "setsid") else None),
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", f"[MASK-DEEP] {line}"))
                    bad_path = self._resolve_verifyscenes_bad_path(line, mask_dir)
                    if bad_path and bad_path not in seen_bad:
                        seen_bad.add(bad_path)
                        bad_files.append(bad_path)
            rc = proc.wait()
        except Exception as e:
            self._log_queue.put(("merge_line", f"[MASK-DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "merge_mask_verify_deep_result",
                    {"rc": rc, "mask_dir": mask_dir, "bad_files": bad_files},
                )
            )
            self._log_queue.put(("verify_done", "merge_mask_deep"))

    def _validate_merge_verify_inputs(self) -> tuple[bool, str, str, list[str]]:
        merged_dir = self.merge_output_var.get().strip()
        if not merged_dir:
            messagebox.showerror("Verify Merging", "Merging output folder is required.")
            return False, "", "", []
        if not os.path.isdir(merged_dir):
            messagebox.showerror("Verify Merging", f"Merging output folder not found:\n{merged_dir}")
            return False, "", "", []
        ok_ref, ref_dir, ref_patterns, ref_kind = self._resolve_verify_reference(
            "merge", "Verify Merging"
        )
        if not ok_ref:
            return False, "", "", []
        merged_dir = self._pipeline_prepare_verify_subset_dir(
            merged_dir, "merge_target", ["*.mp4"]
        )
        self._append_merge_log(f"[VERIFY-MERGE] reference source: {ref_kind} ({ref_dir})")
        return True, merged_dir, ref_dir, ref_patterns

    def _start_merge_verify_quick(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Verify Merging", "Stop Merging before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Merging", "Another verification is already running.")
            return
        ok, merged_dir, ref_dir, ref_patterns = self._validate_merge_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Merging", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="merge_quick")
        self.merge_status_var.set("Verify Merge (Quick) running...")
        self._append_merge_log("=== Verify Merge (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_merge_verify_quick_worker,
            args=(merged_dir, ref_dir, ref_patterns),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_merge_verify_quick_worker(
        self, merged_dir: str, ref_dir: str, ref_patterns: list[str]
    ) -> None:
        try:
            out_files = sorted([str(p) for p in Path(merged_dir).glob("*.mp4") if p.is_file()])
            ref_files = self._collect_files_for_patterns(ref_dir, ref_patterns)
            if not out_files:
                self._log_queue.put(("merge_verify_quick_result", {
                    "ok": False,
                    "message": "No .mp4 files found in merging output folder.",
                    "broken_output": [],
                    "broken_reference": [],
                }))
                return
            if not ref_files:
                self._log_queue.put(("merge_verify_quick_result", {
                    "ok": False,
                    "message": "No reference video files found in selected reference folder.",
                    "broken_output": [],
                    "broken_reference": [],
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            self._log_queue.put(
                ("merge_line", f"[QUICK] checking merged files={len(out_files)} and reference files={len(ref_files)} with {max_workers} workers")
            )

            def _probe_group(file_list: list[str], label: str) -> dict:
                broken: list[str] = []
                total_duration = 0.0
                duration_available = True
                total_frames = 0
                frames_available = True

                def _probe_one(fp: str) -> tuple[str, dict]:
                    return fp, self._probe_video_basic(fp)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_probe_one, fp) for fp in file_list]
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        fp, meta = fut.result()
                        done += 1
                        if not meta.get("ok"):
                            broken.append(fp)
                            self._log_queue.put(("merge_line", f"[QUICK][{label.upper()}][BROKEN] {fp} :: {meta.get('error')}"))
                        else:
                            dur = meta.get("duration")
                            frm = meta.get("frames")
                            if dur is None:
                                duration_available = False
                            else:
                                total_duration += float(dur)
                            if frm is None:
                                frames_available = False
                            else:
                                total_frames += int(frm)
                        if done % 25 == 0 or done == len(file_list):
                            self._log_queue.put(("merge_line", f"[QUICK][{label.upper()}] progress {done}/{len(file_list)}"))
                return {
                    "broken": broken,
                    "total_duration": total_duration,
                    "duration_available": duration_available,
                    "total_frames": total_frames,
                    "frames_available": frames_available,
                }

            out_stats = _probe_group(out_files, "merged")
            ref_stats = _probe_group(ref_files, "reference")

            count_ok = len(out_files) == len(ref_files)
            count_msg = f"merged={len(out_files)} vs reference={len(ref_files)}"

            duration_ok = False
            duration_msg = "n.d."
            if out_stats["duration_available"] and ref_stats["duration_available"]:
                dd = abs(float(out_stats["total_duration"]) - float(ref_stats["total_duration"]))
                duration_ok = dd <= 0.35
                duration_msg = (
                    f"merged={float(out_stats['total_duration']):.3f}s vs "
                    f"reference={float(ref_stats['total_duration']):.3f}s (delta={dd:.3f}s)"
                )

            frames_ok = False
            frames_msg = "n.d."
            if out_stats["frames_available"] and ref_stats["frames_available"]:
                df = abs(int(out_stats["total_frames"]) - int(ref_stats["total_frames"]))
                frames_ok = df <= 1
                frames_msg = (
                    f"merged={int(out_stats['total_frames'])} vs "
                    f"reference={int(ref_stats['total_frames'])} (delta={df})"
                )

            self._log_queue.put(("merge_line", f"[QUICK] file count check: {count_msg}"))
            self._log_queue.put(("merge_line", f"[QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("merge_line", f"[QUICK] packet check: {frames_msg}"))

            ok_final = (
                not out_stats["broken"]
                and not ref_stats["broken"]
                and count_ok
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Merging quick verify completed.\n"
                f"Broken output files: {len(out_stats['broken'])}\n"
                f"Broken reference files: {len(ref_stats['broken'])}\n"
                f"File count match: {'YES' if count_ok else 'NO'} ({count_msg})\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(
                (
                    "merge_verify_quick_result",
                    {
                        "ok": ok_final,
                        "message": message,
                        "broken_output": out_stats["broken"],
                        "broken_reference": ref_stats["broken"],
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(("merge_verify_quick_result", {
                "ok": False,
                "message": f"Merging quick verify failed: {type(e).__name__}: {e}",
                "broken_output": [],
                "broken_reference": [],
            }))
        finally:
            self._log_queue.put(("verify_done", "merge_quick"))

    def _start_merge_verify_deep(self) -> None:
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Verify Merging", "Stop Merging before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Merging", "Another verification is already running.")
            return
        ok, merged_dir, ref_dir, _ref_patterns = self._validate_merge_verify_inputs()
        if not ok:
            return

        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Merging", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(merged_dir).resolve()),
            str(Path(ref_dir).resolve()),
            "--extensions",
            self.VERIFY_ALL_VIDEO_EXTENSIONS,
            "--workers",
            str(workers),
            "--probe-timeout-sec",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
            "--probe-timeout-retries",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
            "--delete",
            "yes",
            "--no-single-line-progress",
        ]

        self._set_verify_running(True, mode="merge_deep")
        self.merge_status_var.set("Verify Merge (Deep) running...")
        self._append_merge_log("=== Verify Merge (Deep) started ===")
        self._append_merge_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))

        self._verify_thread = threading.Thread(
            target=self._run_merge_verify_deep_worker,
            args=(cmd, str(Path(merged_dir).resolve())),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_merge_verify_deep_worker(self, cmd: list[str], merged_dir: str) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", f"[DEEP] {line}"))
                    bad_path = self._resolve_verifyscenes_bad_path(line, merged_dir)
                    if bad_path and bad_path not in seen_bad:
                        seen_bad.add(bad_path)
                        bad_files.append(bad_path)
            rc = proc.wait()
        except Exception as e:
            self._log_queue.put(("merge_line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "merge_verify_deep_result",
                    {"rc": rc, "merged_dir": merged_dir, "bad_files": bad_files},
                )
            )
            self._log_queue.put(("verify_done", "merge_deep"))

    def _build_join_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(9, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="SBS scenes folder (auto):").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.join_input_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_join_input_folder).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(parent, text="seg-mono folder (auto):").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.join_seg_mono_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_join_seg_mono_folder).grid(
            row=1, column=2, padx=4
        )
        ttk.Label(
            parent,
            text="Mono->SBS uses Merge output format and Merge encoding settings.",
            justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(parent, text="Joined output file:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.join_output_var).grid(
            row=3, column=1, sticky="ew", padx=6
        )
        out_btns = ttk.Frame(parent)
        out_btns.grid(row=3, column=2, sticky="w")
        ttk.Button(out_btns, text="Browse...", command=self._browse_join_output_file).grid(
            row=0, column=0, padx=(0, 4)
        )
        ttk.Button(out_btns, text="Open", command=self._open_join_output_folder).grid(
            row=0, column=1
        )

        mode_frame = ttk.LabelFrame(parent, text="Joining Mode", padding=8)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        mode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(mode_frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self.join_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.join_mode_var,
            values=["Auto (recommended)", "Manual"],
            width=18,
            state="readonly",
        )
        self.join_mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.join_mode_combo.bind("<<ComboboxSelected>>", self._on_join_mode_changed)

        ttk.Label(
            mode_frame,
            textvariable=self.join_info_text_var,
            justify="left",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params_frame = ttk.LabelFrame(parent, text="Join Parameters", padding=8)
        params_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        params_frame.grid_columnconfigure(9, weight=1)

        ttk.Label(params_frame, text="Encoder:").grid(row=0, column=0, sticky="w")
        self.join_encoder_entry = ttk.Combobox(
            params_frame,
            textvariable=self.join_encoder_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=14,
            state="readonly",
        )
        self.join_encoder_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Quality (CRF/CQ):").grid(row=0, column=2, sticky="w")
        self.join_crf_entry = ttk.Entry(params_frame, textvariable=self.join_crf_var, width=8)
        self.join_crf_entry.grid(row=0, column=3, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Preset:").grid(row=0, column=4, sticky="w")
        self.join_preset_entry = ttk.Entry(params_frame, textvariable=self.join_preset_var, width=10)
        self.join_preset_entry.grid(row=0, column=5, sticky="w", padx=(6, 12))

        self.join_pixfmt_override_check = ttk.Checkbutton(
            params_frame,
            text="Override pix_fmt",
            variable=self.join_pix_fmt_override_var,
            command=self._on_join_pixfmt_override_toggle,
        )
        self.join_pixfmt_override_check.grid(row=0, column=6, sticky="w")

        ttk.Label(params_frame, text="PixFmt (inherited):").grid(
            row=0, column=7, sticky="w", padx=(12, 0)
        )
        self.join_pixfmt_entry = ttk.Entry(params_frame, textvariable=self.join_pix_fmt_var, width=10)
        self.join_pixfmt_entry.grid(row=0, column=8, sticky="w", padx=(6, 0))

        ttk.Label(params_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.join_extra_args_entry = ttk.Entry(params_frame, textvariable=self.join_extra_args_var)
        self.join_extra_args_entry.grid(
            row=1, column=1, columnspan=8, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.join_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.join_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_join_command
        )
        self.join_preview_btn.grid(row=0, column=0, padx=(0, 6))
        self.join_mono_run_btn = ttk.Button(
            buttons, text="Run Mono->SBS", command=self._run_join_prepare_mono
        )
        self.join_mono_run_btn.grid(row=0, column=1, padx=6)
        self.join_mono_verify_btn = ttk.Button(
            buttons, text="Verify Mono->SBS", command=self._start_join_mono_verify
        )
        self.join_mono_verify_btn.grid(row=0, column=2, padx=6)
        self.join_run_btn = ttk.Button(
            buttons, text="Join Scenes", command=self._run_join_scenes
        )
        self.join_run_btn.grid(row=0, column=3, padx=6)
        self.join_verify_btn = ttk.Button(
            buttons, text="Verify Join", command=self._start_join_verify
        )
        self.join_verify_btn.grid(row=0, column=4, padx=6)
        self.join_remux_btn = ttk.Button(
            buttons, text="Remux", command=self._start_join_remux
        )
        self.join_remux_btn.grid(row=0, column=5, padx=6)
        self.join_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_join, state=tk.DISABLED
        )
        self.join_stop_btn.grid(row=0, column=6, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_join_log).grid(
            row=0, column=7, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=8, column=0, columnspan=3, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(2, weight=1)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.join_status_var).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        self.join_progress = ttk.Progressbar(
            status_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.join_progress_var,
            maximum=100.0,
        )
        self.join_progress.grid(row=0, column=2, sticky="ew", padx=4)

        log_frame = ttk.LabelFrame(parent, text="Joining Log", padding=6)
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.join_log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.join_log_text.grid(row=0, column=0, sticky="nsew")
        jscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.join_log_text.yview)
        jscroll.grid(row=0, column=1, sticky="ns")
        self.join_log_text.configure(yscrollcommand=jscroll.set)

        self._on_join_mode_changed()
        self._on_join_pixfmt_override_toggle()
        self._preview_join_command()
        self._set_join_running(False)

    def _open_join_input_folder(self) -> None:
        folder = self.join_input_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_join_log(f"SBS input folder ready: {folder}")

    def _open_join_seg_mono_folder(self) -> None:
        folder = self.join_seg_mono_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_join_log(f"seg-mono folder ready: {folder}")

    def _browse_join_output_file(self) -> None:
        current = self.join_output_var.get().strip()
        start_dir = str(Path(current).resolve().parent) if current else "."
        selected = filedialog.asksaveasfilename(
            title="Select joined output file",
            initialdir=start_dir,
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )
        if selected:
            self.join_output_var.set(selected)
            self._preview_join_command()

    def _open_join_output_folder(self) -> None:
        out_path = self.join_output_var.get().strip()
        if not out_path:
            return
        folder = str(Path(out_path).resolve().parent)
        os.makedirs(folder, exist_ok=True)
        self._append_join_log(f"Join output folder ready: {folder}")

    def _on_join_mode_changed(self, _event=None) -> None:
        mode = self.join_mode_var.get().strip()
        if mode == "Manual":
            self.join_info_text_var.set(self.JOIN_MANUAL_INFO)
            if not self._join_manual_notice_shown:
                self._join_manual_notice_shown = True
                messagebox.showwarning("Joining Manual Mode", self.JOIN_MANUAL_WARNING)
        else:
            self.join_mode_var.set("Auto (recommended)")
            self.join_info_text_var.set(self.JOIN_AUTO_INFO)
            self.join_encoder_var.set("hevc_nvenc")
            self.join_preset_var.set("p7")
            self.join_extra_args_var.set(self.JOIN_DEFAULT_ARGS)
        self.join_encoder_var.set(
            self._normalize_ffmpeg_codec(
                self.join_encoder_var.get(),
                "hevc_nvenc",
            )
        )
        self._apply_join_control_states()

    def _sync_join_encoding_from_scene(self) -> None:
        self.join_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip() or "yuv420p")

    def _on_join_pixfmt_override_toggle(self) -> None:
        if not self.join_pix_fmt_override_var.get():
            self._sync_join_encoding_from_scene()
            self.join_pixfmt_entry.configure(state=tk.DISABLED)
        else:
            self.join_pixfmt_entry.configure(state=tk.NORMAL)
        self._preview_join_command()

    def _apply_join_control_states(self) -> None:
        manual = self.join_mode_var.get().strip() == "Manual"
        self.join_encoder_entry.configure(state="readonly" if manual else tk.DISABLED)
        self.join_crf_entry.configure(state=tk.NORMAL)
        self.join_preset_entry.configure(state=tk.NORMAL if manual else tk.DISABLED)
        self.join_extra_args_entry.configure(state=tk.NORMAL if manual else tk.DISABLED)
        self._on_join_pixfmt_override_toggle()
        self._preview_join_command()

    def _quality_flag_for_codec(self, codec: str, fallback: str, nvenc_flag: str) -> str:
        codec_value = self._normalize_ffmpeg_codec(codec, fallback)
        return nvenc_flag if "nvenc" in codec_value else "crf"

    def _join_quality_flag(self) -> str:
        encoder = self._normalize_ffmpeg_codec(self.join_encoder_var.get(), "hevc_nvenc")
        return self._quality_flag_for_codec(encoder, "hevc_nvenc", "cq")

    def _merge_quality_flag(self) -> str:
        codec = self._normalize_ffmpeg_codec(
            self.merge_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        self.merge_codec_var.set(codec)
        return self._quality_flag_for_codec(
            codec,
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
            "qp",
        )

    def _join_layout_for_seg_mono(self) -> str:
        return "half_sbs" if self.merge_output_format_var.get().strip() == "Half SBS" else "full_sbs"

    def _build_join_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        out_path = self.join_output_var.get().strip()
        quality_value = self.join_crf_var.get().strip() or "16"
        encoder = self._normalize_ffmpeg_codec(self.join_encoder_var.get(), "hevc_nvenc")
        self.join_encoder_var.set(encoder)
        preset = self.join_preset_var.get().strip() or "p7"
        pix_fmt = self.join_pix_fmt_var.get().strip() or "yuv420p"
        extra_args = self.join_extra_args_var.get().strip()
        quality_flag = self._join_quality_flag()

        env_updates: dict[str, str] = {
            "DIR_SBS": self.join_input_var.get().strip(),
            "PATTERN": "*_sbs.mp4",
            "OUT": out_path,
            "FFMPEG_BIN": "ffmpeg",
            "ENCODER": encoder,
            "PRESET": preset,
            "QUALITY_FLAG": quality_flag,
            "QUALITY_VALUE": quality_value,
            "CQ": quality_value,
            "CRF": quality_value,
            "PIX_FMT": pix_fmt,
            "EXTRA_ARGS": extra_args,
            # Force final height to 1080 while preserving center framing:
            # if shorter -> pad, if taller (e.g. temporary crop pad) -> crop.
            "VF": "pad=iw:max(ih\\,1080):0:(max(ih\\,1080)-ih)/2:black,crop=iw:1080:0:(ih-1080)/2",
        }
        cmd = ["bash", "Utilities/Rejoin_HEVC_NVENC.sh"]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview

    def _default_remux_output_path(self) -> str:
        source_path = self.scene_input_var.get().strip()
        join_path = self.join_output_var.get().strip()
        if join_path:
            out_dir = Path(join_path).resolve().parent
        else:
            work_dir = Path(self.work_folder_var.get().strip() or "./work").resolve()
            out_dir = work_dir / self.STANDARD_SUBDIRS["join"]
        src_stem = Path(source_path).stem if source_path else "source"
        return str((out_dir / f"{src_stem}_3D_remux.mkv").resolve())

    def _build_join_remux_payload(self) -> tuple[list[str], dict[str, str], str, str]:
        source_path = self.scene_input_var.get().strip()
        video_3d_path = self.join_output_var.get().strip()
        out_path = self._default_remux_output_path()
        env_updates: dict[str, str] = {
            "SOURCE_FILE": source_path,
            "VIDEO_3D_FILE": video_3d_path,
            "OUT_FILE": out_path,
            "MKVMERGE_BIN": "mkvmerge",
            "OVERWRITE": "1",
        }
        cmd = ["bash", "Utilities/remux_replace_video_mkvtoolnix.sh"]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview, out_path

    def _build_join_prepare_mono_cmd(self) -> list[str]:
        merge_codec = self._normalize_ffmpeg_codec(
            self.merge_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        self.merge_codec_var.set(merge_codec)
        return [
            sys.executable,
            str(Path("Utilities/prepare_seg_mono_to_sbs.py").resolve()),
            "--seg-mono-dir",
            str(Path(self.join_seg_mono_var.get().strip()).resolve()),
            "--sbs-dir",
            str(Path(self.join_input_var.get().strip()).resolve()),
            "--layout",
            self._join_layout_for_seg_mono(),
            "--ffmpeg-bin",
            "ffmpeg",
            "--ffprobe-bin",
            "ffprobe",
            "--codec",
            merge_codec,
            "--quality-flag",
            self._merge_quality_flag(),
            "--quality",
            self.merge_crf_var.get().strip() or "1",
            "--preset",
            self.merge_preset_var.get().strip(),
            "--pix-fmt",
            self.merge_pix_fmt_var.get().strip(),
            "--extra-ffmpeg-args",
            self.merge_extra_ffmpeg_args_var.get().strip(),
        ]

    def _preview_join_command(self) -> None:
        try:
            _cmd, _env, preview = self._build_join_runner_payload()
            self.join_cmd_preview_var.set(preview)
        except Exception as e:
            self.join_cmd_preview_var.set(f"Invalid options: {e}")

    def _append_join_log(self, line: str) -> None:
        self.join_log_text.configure(state=tk.NORMAL)
        self.join_log_text.insert(tk.END, line + "\n")
        self.join_log_text.see(tk.END)
        self.join_log_text.configure(state=tk.DISABLED)

    def _clear_join_log(self) -> None:
        self.join_log_text.configure(state=tk.NORMAL)
        self.join_log_text.delete("1.0", tk.END)
        self.join_log_text.configure(state=tk.DISABLED)

    def _run_join_scenes(self) -> None:
        if self._join_thread and self._join_thread.is_alive():
            messagebox.showinfo("Joining", "Joining is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Joining", "Stop verification before starting Joining.")
            return

        try:
            cmd, env_updates, _preview = self._build_join_runner_payload()
        except Exception as exc:
            messagebox.showerror("Joining", f"Invalid joining options:\n{exc}")
            return

        join_script = Path("Utilities/Rejoin_HEVC_NVENC.sh").resolve()
        if not join_script.is_file():
            messagebox.showerror("Joining", f"Join script not found:\n{join_script}")
            return

        sbs_dir = self.join_input_var.get().strip()
        if not sbs_dir:
            messagebox.showerror("Joining", "SBS scenes folder is required.")
            return
        if not os.path.isdir(sbs_dir):
            messagebox.showerror("Joining", f"SBS scenes folder not found:\n{sbs_dir}")
            return
        out_path = self.join_output_var.get().strip()
        if not out_path:
            messagebox.showerror("Joining", "Joined output path is required.")
            return
        os.makedirs(str(Path(out_path).resolve().parent), exist_ok=True)
        if shutil.which("ffmpeg") is None:
            messagebox.showerror("Joining", "ffmpeg command not found in PATH.")
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Joining", "ffprobe command not found in PATH.")
            return
        mono_ok, mono_msg, _broken_targets, broken_reference = self._verify_join_mono_outputs_coverage(
            cleanup_incomplete=False
        )
        if not mono_ok:
            if not self._pipeline_test_active:
                self._pipeline_set_completed("mono_to_sbs", False)
                self._pipeline_set_verified("mono_to_sbs", "none")
                self._refresh_pipeline_status_panel()
                self._save_pipeline_state()
            msg = mono_msg
            if broken_reference:
                msg += self._format_corrupted_files_block(
                    broken_reference,
                    "Corrupted seg-mono source files",
                )
            messagebox.showwarning(
                "Joining",
                f"{msg}\n\nRun Mono->SBS and verify it before Join.",
            )
            return

        self._join_stop_requested = False
        self._join_stop_clicks = 0
        self._join_expected_duration_sec = None
        self._join_active_output_path = out_path
        self.join_status_var.set("Starting...")
        self.join_progress_var.set(0.0)
        self._set_join_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("join")
        self._append_join_log("=== Joining started ===")
        self._append_join_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_join_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._join_thread = threading.Thread(
            target=self._run_join_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._join_thread.start()

    def _run_join_prepare_mono(self) -> None:
        if self._join_thread and self._join_thread.is_alive():
            messagebox.showinfo("Mono->SBS", "Another join task is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Mono->SBS", "Stop verification before running Mono->SBS.")
            return

        try:
            prep_cmd = self._build_join_prepare_mono_cmd()
        except Exception as exc:
            messagebox.showerror("Mono->SBS", f"Invalid Mono->SBS options:\n{exc}")
            return

        prep_script = Path("Utilities/prepare_seg_mono_to_sbs.py").resolve()
        if not prep_script.is_file():
            messagebox.showerror("Mono->SBS", f"Script not found:\n{prep_script}")
            return

        seg_mono_dir = self.join_seg_mono_var.get().strip()
        if not seg_mono_dir:
            messagebox.showerror("Mono->SBS", "seg-mono folder is required.")
            return
        if not os.path.isdir(seg_mono_dir):
            messagebox.showerror("Mono->SBS", f"seg-mono folder not found:\n{seg_mono_dir}")
            return
        sbs_dir = self.join_input_var.get().strip()
        if not sbs_dir:
            messagebox.showerror("Mono->SBS", "SBS scenes folder is required.")
            return
        os.makedirs(sbs_dir, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            messagebox.showerror("Mono->SBS", "ffmpeg command not found in PATH.")
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Mono->SBS", "ffprobe command not found in PATH.")
            return

        self._join_stop_requested = False
        self._join_stop_clicks = 0
        self._join_expected_duration_sec = None
        self._join_active_output_path = ""
        self.join_status_var.set("Mono->SBS starting...")
        self.join_progress_var.set(0.0)
        self._set_join_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("mono_to_sbs")
        self._append_join_log("=== Mono->SBS started ===")
        self._append_join_log("CMD: " + " ".join(shlex.quote(x) for x in prep_cmd))
        self._join_thread = threading.Thread(
            target=self._run_join_prepare_mono_worker,
            args=(prep_cmd,),
            daemon=True,
        )
        self._join_thread.start()

    def _run_join_prepare_mono_worker(self, prep_cmd: list[str]) -> None:
        proc = None
        step_success = False
        try:
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                prep_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=preexec,
            )
            self._join_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("join_line", line))
                if self._join_stop_requested:
                    break
            rc = proc.wait()
            if self._join_stop_requested:
                self._log_queue.put(("join_status", "Stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("join_status", "Mono->SBS completed"))
                self._log_queue.put(("join_progress", "100"))
            else:
                self._log_queue.put(("join_status", f"Mono->SBS failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("join_line", f"[MONO][ERROR] {exc}"))
            self._log_queue.put(("join_status", "Mono->SBS failed"))
        finally:
            self._join_process = None
            self._join_expected_duration_sec = None
            self._join_active_output_path = ""
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("join_done", {"step": "mono_to_sbs", "success": step_success}))

    def _run_join_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        step_success = False
        try:
            preexec = os.setsid if hasattr(os, "setsid") else None
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._join_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("join_line", line))
                    self._try_parse_join_progress(line)
                if self._join_stop_requested:
                    break
            rc = proc.wait()
            if self._join_stop_requested:
                self._log_queue.put(("join_status", "Stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("join_status", "Completed"))
                self._log_queue.put(("join_progress", "100"))
            else:
                self._log_queue.put(("join_status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("join_line", f"[JOIN][ERROR] {exc}"))
            self._log_queue.put(("join_status", "Failed"))
        finally:
            self._join_process = None
            self._join_expected_duration_sec = None
            self._join_active_output_path = ""
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("join_done", {"step": "join", "success": step_success}))

    def _start_join_remux(self) -> None:
        if self._join_thread and self._join_thread.is_alive():
            messagebox.showinfo("Remux", "Join/Remux process is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Remux", "Stop verification before starting remux.")
            return
        try:
            cmd, env_updates, preview, out_path = self._build_join_remux_payload()
        except Exception as exc:
            messagebox.showerror("Remux", f"Invalid remux options:\n{exc}")
            return

        remux_script = Path("Utilities/remux_replace_video_mkvtoolnix.sh").resolve()
        if not remux_script.is_file():
            messagebox.showerror("Remux", f"Remux script not found:\n{remux_script}")
            return
        source_path = self.scene_input_var.get().strip()
        if not source_path or not os.path.isfile(source_path):
            messagebox.showerror("Remux", f"Source file not found:\n{source_path or '(empty)'}")
            return
        video_3d_path = self.join_output_var.get().strip()
        if not video_3d_path or not os.path.isfile(video_3d_path):
            messagebox.showerror("Remux", f"3D joined video not found:\n{video_3d_path or '(empty)'}")
            return
        if shutil.which("mkvmerge") is None:
            messagebox.showerror("Remux", "mkvmerge command not found in PATH.")
            return

        out_file = Path(out_path)
        os.makedirs(str(out_file.parent), exist_ok=True)
        if out_file.exists():
            overwrite = messagebox.askyesno(
                "Remux",
                f"Output already exists:\n{out_file}\n\nOverwrite it?",
            )
            if not overwrite:
                return

        self._join_stop_requested = False
        self._join_stop_clicks = 0
        self._join_expected_duration_sec = None
        self._join_active_output_path = str(out_file)
        self.join_status_var.set("Remux starting...")
        self.join_progress_var.set(0.0)
        self._set_join_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("remux")
        self._append_join_log("=== Remux started ===")
        self._append_join_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_join_log("ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()))
        self._append_join_log(f"[REMUX] output: {out_file}")

        self._join_thread = threading.Thread(
            target=self._run_join_remux_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._join_thread.start()

    def _run_join_remux_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        try:
            preexec = os.setsid if hasattr(os, "setsid") else None
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._join_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("join_line", line))
                if self._join_stop_requested:
                    break
            rc = proc.wait()
            if self._join_stop_requested:
                self._log_queue.put(("join_status", "Stopped by user"))
            elif rc == 0:
                self._log_queue.put(("join_status", "Completed"))
                self._log_queue.put(("join_progress", "100"))
            else:
                self._log_queue.put(("join_status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("join_line", f"[REMUX][ERROR] {exc}"))
            self._log_queue.put(("join_status", "Failed"))
        finally:
            self._join_process = None
            self._join_expected_duration_sec = None
            self._join_active_output_path = ""
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("join_done", {"step": "remux"}))

    def _stop_join(self, prompt_user: bool = True) -> None:
        running = bool(self._join_thread and self._join_thread.is_alive())
        if not running:
            return
        if prompt_user:
            messagebox.showwarning(
                "Stop Joining",
                "Immediate stop requested.\n\n"
                "Current join process will be force-killed and partial output will be deleted.",
            )
        self._join_stop_requested = True
        self.join_status_var.set("Force stop requested...")
        self._append_join_log("[STOP] immediate force stop requested.")
        self._force_kill_join()
        self._cleanup_join_output_after_stop()

    def _send_join_signal(self, sig: int) -> None:
        proc = self._join_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception as exc:
            self._append_join_log(f"Signal send failed: {exc}")

    def _force_kill_join(self) -> None:
        proc = self._join_process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                self._append_join_log("Joining process force-killed.")
            except Exception as exc:
                self._append_join_log(f"Joining kill failed: {exc}")
        self._cleanup_join_output_after_stop()

    def _cleanup_join_output_after_stop(self) -> None:
        out_path = self._join_active_output_path.strip() or self.join_output_var.get().strip()
        if not out_path:
            return
        out_file = Path(out_path)
        if out_file.is_file():
            try:
                out_file.unlink()
                self._append_join_log(f"[CLEANUP] removed partial output: {out_file}")
            except Exception as exc:
                self._append_join_log(f"[CLEANUP] could not remove partial output: {exc}")

    def _set_join_running(self, is_running: bool) -> None:
        self.join_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.join_mono_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.join_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.join_remux_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.join_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        verify_state = tk.DISABLED if (is_running or self._verify_running) else tk.NORMAL
        self.join_mono_verify_btn.configure(state=verify_state)
        self.join_verify_btn.configure(state=verify_state)
        if is_running:
            self.join_stop_btn.configure(text="Stop")
        else:
            self.join_stop_btn.configure(text="Stop")
            self._join_stop_requested = False

    def _try_parse_join_progress(self, line: str) -> None:
        t = self._parse_ffmpeg_time_seconds(line)
        if t is not None and self._join_expected_duration_sec and self._join_expected_duration_sec > 0:
            prog = max(0.0, min(100.0, (t / self._join_expected_duration_sec) * 100.0))
            self._log_queue.put(("join_progress", str(prog)))
            return
        if "Lsize=" in line or line.startswith("video:"):
            self._log_queue.put(("join_progress", "100"))

    @staticmethod
    def _parse_ffmpeg_time_seconds(line: str) -> float | None:
        m = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
        if not m:
            return None
        try:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ss = float(m.group(3))
            return float(hh * 3600 + mm * 60) + ss
        except Exception:
            return None

    def _join_manual_verify_mode(self) -> str:
        return "deep" if self.pipeline_verify_after_var.get().strip().lower() == "deep" else "quick"

    def _start_join_mono_verify(self) -> None:
        if self._join_thread and self._join_thread.is_alive():
            messagebox.showinfo("Verify Mono->SBS", "Stop Joining before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Mono->SBS", "Another verification is already running.")
            return

        mode = self._join_manual_verify_mode()
        seg_mono_dir = self.join_seg_mono_var.get().strip()
        if not seg_mono_dir:
            messagebox.showerror("Verify Mono->SBS", "seg-mono folder is required.")
            return
        if not os.path.isdir(seg_mono_dir):
            messagebox.showerror("Verify Mono->SBS", f"seg-mono folder not found:\n{seg_mono_dir}")
            return
        sbs_dir = self.join_input_var.get().strip()
        if not sbs_dir:
            messagebox.showerror("Verify Mono->SBS", "SBS scenes folder is required.")
            return
        if not os.path.isdir(sbs_dir):
            messagebox.showerror("Verify Mono->SBS", f"SBS scenes folder not found:\n{sbs_dir}")
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Mono->SBS", "ffprobe not found in PATH.")
            return
        script_path = ""
        if mode == "deep":
            if shutil.which("ffmpeg") is None:
                messagebox.showerror("Verify Mono->SBS", "ffmpeg not found in PATH.")
                return
            script = Path("Utilities/verifyscenes.py").resolve()
            if not script.is_file():
                messagebox.showerror("Verify Mono->SBS", f"Script not found:\n{script}")
                return
            script_path = str(script)

        self._set_verify_running(True, mode=f"join_mono_{mode}")
        self.join_status_var.set(f"Mono->SBS Verify ({mode}) running...")
        self._append_join_log(f"=== Verify Mono->SBS ({mode}) started ===")
        self._verify_thread = threading.Thread(
            target=(
                self._run_join_mono_verify_deep_worker
                if mode == "deep"
                else self._run_join_mono_verify_quick_worker
            ),
            args=((script_path,) if mode == "deep" else ()),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_join_mono_verify_quick_worker(self) -> None:
        try:
            ok, msg, broken_output, broken_reference = self._verify_join_mono_outputs_coverage(
                cleanup_incomplete=False
            )
            self._log_queue.put(
                (
                    "join_mono_verify_result",
                    {
                        "ok": ok,
                        "message": msg,
                        "mode": "quick",
                        "broken_output": broken_output,
                        "broken_reference": broken_reference,
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(
                (
                    "join_mono_verify_result",
                    {
                        "ok": False,
                        "message": f"Mono->SBS quick verify failed: {type(e).__name__}: {e}",
                        "mode": "quick",
                        "broken_output": [],
                        "broken_reference": [],
                    },
                )
            )
        finally:
            self._log_queue.put(("verify_done", "join_mono_quick"))

    def _run_join_mono_verify_deep_worker(self, script_path: str) -> None:
        bad_targets: list[str] = []
        broken_reference: list[str] = []
        overall_ok = False
        message = "Mono->SBS deep verify failed."
        try:
            quick_ok, quick_msg, broken_output, broken_reference = self._verify_join_mono_outputs_coverage(
                cleanup_incomplete=False
            )
            if not quick_ok:
                bad_targets = list(broken_output)
                overall_ok = False
                message = quick_msg
                return

            expected_pairs = self._collect_join_mono_expected_pairs()
            if not expected_pairs:
                overall_ok = True
                message = "Mono->SBS deep verify: no seg-mono clips found."
            else:
                workers = self._get_verify_scenes_workers()
                seg_mono_dir = str(Path(self.join_seg_mono_var.get().strip()).resolve())
                with tempfile.TemporaryDirectory(prefix="verify_join_mono_") as tmpdir:
                    tmp_root = Path(tmpdir)
                    target_dir = tmp_root / "targets"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    link_map: dict[str, str] = {}
                    for pair in expected_pairs:
                        target_path = pair["expected_output"]
                        link_path = target_dir / target_path.name
                        os.symlink(str(target_path), str(link_path))
                        link_map[str(link_path.resolve())] = str(target_path)

                    cmd = [
                        sys.executable,
                        script_path,
                        str(target_dir),
                        seg_mono_dir,
                        "--extensions",
                        self.VERIFY_ALL_VIDEO_EXTENSIONS,
                        "--workers",
                        str(workers),
                        "--probe-timeout-sec",
                        str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
                        "--probe-timeout-retries",
                        str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
                        "--delete",
                        "no",
                        "--no-single-line-progress",
                    ]
                    self._log_queue.put(
                        (
                            "join_line",
                            "[MONO][DEEP] cmd: " + " ".join(shlex.quote(x) for x in cmd),
                        )
                    )
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )
                    assert proc.stdout is not None
                    for raw in proc.stdout:
                        line = raw.rstrip("\n")
                        if line:
                            self._log_queue.put(("join_line", f"[MONO][DEEP] {line}"))
                            bad_path = self._resolve_verifyscenes_bad_path(line, str(target_dir))
                            if bad_path:
                                real_bad = link_map.get(bad_path)
                                if real_bad and real_bad not in bad_targets:
                                    bad_targets.append(real_bad)
                    rc = int(proc.wait() or 0)
                    overall_ok = rc == 0
                    message = (
                        "Mono->SBS deep verify completed successfully."
                        if overall_ok
                        else "Mono->SBS deep verify failed."
                    )
        except Exception as e:
            overall_ok = False
            message = f"Mono->SBS deep verify failed: {type(e).__name__}: {e}"
        finally:
            self._log_queue.put(
                (
                    "join_mono_verify_result",
                    {
                        "ok": overall_ok,
                        "message": message,
                        "mode": "deep",
                        "broken_output": bad_targets,
                        "broken_reference": broken_reference,
                    },
                )
            )
            self._log_queue.put(("verify_done", "join_mono_deep"))

    def _validate_join_verify_inputs(self) -> tuple[bool, str, str]:
        source_path = self.scene_input_var.get().strip()
        joined_path = self.join_output_var.get().strip()
        if not source_path:
            messagebox.showerror("Verify Join", "Source video path is required.")
            return False, "", ""
        if not os.path.isfile(source_path):
            messagebox.showerror("Verify Join", f"Source video not found:\n{source_path}")
            return False, "", ""
        if not joined_path:
            messagebox.showerror("Verify Join", "Joined output file is required.")
            return False, "", ""
        if not os.path.isfile(joined_path):
            messagebox.showerror("Verify Join", f"Joined output file not found:\n{joined_path}")
            return False, "", ""
        return True, source_path, joined_path

    def _start_join_verify(self) -> None:
        if self._join_thread and self._join_thread.is_alive():
            messagebox.showinfo("Verify Join", "Stop Joining before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Join", "Another verification is already running.")
            return
        ok, source_path, joined_path = self._validate_join_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Join", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="join_quick")
        self.join_status_var.set("Verify running...")
        self._append_join_log("=== Verify Join started ===")
        self._verify_thread = threading.Thread(
            target=self._run_join_verify_worker,
            args=(source_path, joined_path),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_join_verify_worker(self, source_path: str, joined_path: str) -> None:
        try:
            src = self._probe_video_basic(source_path)
            out = self._probe_video_basic(joined_path)
            if not src.get("ok"):
                self._log_queue.put(
                    (
                        "join_verify_result",
                        {
                            "ok": False,
                            "message": f"Source probe failed: {src.get('error')}",
                        },
                    )
                )
                return
            if not out.get("ok"):
                self._log_queue.put(
                    (
                        "join_verify_result",
                        {
                            "ok": False,
                            "message": f"Output probe failed: {out.get('error')}",
                        },
                    )
                )
                return

            duration_ok = False
            duration_msg = "n.d."
            if src.get("duration") is not None and out.get("duration") is not None:
                delta = abs(float(src["duration"]) - float(out["duration"]))
                duration_ok = delta <= 0.35
                duration_msg = (
                    f"source={float(src['duration']):.3f}s vs output={float(out['duration']):.3f}s "
                    f"(delta={delta:.3f}s)"
                )

            frames_ok = False
            frames_msg = "n.d."
            if src.get("frames") is not None and out.get("frames") is not None:
                delta_f = abs(int(src["frames"]) - int(out["frames"]))
                frames_ok = delta_f <= 1
                frames_msg = (
                    f"source={int(src['frames'])} vs output={int(out['frames'])} (delta={delta_f})"
                )

            ok_final = (frames_ok or frames_msg == "n.d.")
            msg = (
                "Join verify completed.\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(("join_verify_result", {"ok": ok_final, "message": msg}))
        except Exception as e:
            self._log_queue.put(
                (
                    "join_verify_result",
                    {"ok": False, "message": f"Join verify failed: {type(e).__name__}: {e}"},
                )
            )
        finally:
            self._log_queue.put(("verify_done", "join_quick"))

    def _build_placeholder_tab(self, parent: ttk.Frame, text: str) -> None:
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text=text,
            anchor="center",
            justify="center",
            font=("TkDefaultFont", 11),
        ).grid(row=0, column=0, sticky="nsew")

    def _retry_policy_default(self) -> dict[str, dict[str, object]]:
        return {
            key: {
                "garbage_collection_threshold": bool(cfg.get("garbage_collection_threshold", True)),
                "expandable_segments": bool(cfg.get("expandable_segments", True)),
                "max_split_size_mb": self._normalize_retry_max_split(cfg.get("max_split_size_mb", "off")),
                "cpu_offload_inherited": bool(cfg.get("cpu_offload_inherited", True)),
                "cpu_offload_mode": self._normalize_retry_offload_mode(cfg.get("cpu_offload_mode", "model")),
            }
            for key, cfg in self.RETRY_POLICY_DEFAULT.items()
        }

    def _normalize_retry_max_split(self, value: object) -> str:
        s = str(value or "").strip().lower()
        if s in {"", "none", "0", "off", "false"}:
            return "off"
        if s in self.RETRY_POLICY_MAX_SPLIT_CHOICES:
            return s
        try:
            parsed = int(float(s))
            sval = str(parsed)
            if sval in self.RETRY_POLICY_MAX_SPLIT_CHOICES:
                return sval
        except Exception:
            pass
        return "off"

    def _normalize_retry_offload_mode(self, value: object) -> str:
        s = str(value or "").strip().lower()
        if s in self.RETRY_POLICY_OFFLOAD_CHOICES:
            return s
        return "model"

    def _normalize_retry_policy_config(
        self, data: object
    ) -> dict[str, dict[str, object]]:
        out = self._retry_policy_default()
        if not isinstance(data, dict):
            return out
        for profile in self.RETRY_POLICY_PROFILES:
            raw = data.get(profile)
            if not isinstance(raw, dict):
                continue
            out[profile] = {
                "garbage_collection_threshold": bool(
                    raw.get(
                        "garbage_collection_threshold",
                        out[profile]["garbage_collection_threshold"],
                    )
                ),
                "expandable_segments": bool(
                    raw.get("expandable_segments", out[profile]["expandable_segments"])
                ),
                "max_split_size_mb": self._normalize_retry_max_split(
                    raw.get("max_split_size_mb", out[profile]["max_split_size_mb"])
                ),
                "cpu_offload_inherited": bool(
                    raw.get(
                        "cpu_offload_inherited",
                        out[profile]["cpu_offload_inherited"],
                    )
                ),
                "cpu_offload_mode": self._normalize_retry_offload_mode(
                    raw.get("cpu_offload_mode", out[profile]["cpu_offload_mode"])
                ),
            }
        return out

    def _retry_policy_from_config_key(self, key: str) -> dict[str, dict[str, object]]:
        return self._normalize_retry_policy_config(self._config.get(key))

    def _collect_retry_policy_config_from_vars(
        self, vars_map: dict[str, dict[str, tk.Variable]]
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for profile in self.RETRY_POLICY_PROFILES:
            row = vars_map.get(profile, {})
            out[profile] = {
                "garbage_collection_threshold": bool(
                    row["garbage_collection_threshold"].get()
                ),
                "expandable_segments": bool(row["expandable_segments"].get()),
                "max_split_size_mb": self._normalize_retry_max_split(
                    row["max_split_size_mb"].get()
                ),
                "cpu_offload_inherited": bool(row["cpu_offload_inherited"].get()),
                "cpu_offload_mode": self._normalize_retry_offload_mode(
                    row["cpu_offload_mode"].get()
                ),
            }
        return out

    def _build_retry_policy_runtime_payload(
        self,
        vars_map: dict[str, dict[str, tk.Variable]],
        inherited_offload: str,
    ) -> dict[str, dict[str, object]]:
        fallback_offload = self._normalize_retry_offload_mode(inherited_offload or "model")
        out: dict[str, dict[str, object]] = {}
        for profile in self.RETRY_POLICY_PROFILES:
            row = vars_map[profile]
            max_split_s = self._normalize_retry_max_split(row["max_split_size_mb"].get())
            max_split: int | None = None if max_split_s == "off" else int(max_split_s)
            inherited = bool(row["cpu_offload_inherited"].get())
            if inherited:
                offload_mode = fallback_offload
            else:
                offload_mode = self._normalize_retry_offload_mode(
                    row["cpu_offload_mode"].get()
                )
            out[profile] = {
                "garbage_collection_threshold": bool(
                    row["garbage_collection_threshold"].get()
                ),
                "expandable_segments": bool(row["expandable_segments"].get()),
                "max_split_size_mb": max_split,
                "cpu_offload_mode": offload_mode,
            }
        return out

    def _build_retry_policy_json(
        self,
        vars_map: dict[str, dict[str, tk.Variable]],
        inherited_offload: str,
    ) -> str:
        payload = self._build_retry_policy_runtime_payload(vars_map, inherited_offload)
        return json.dumps(payload, separators=(",", ":"))

    def _set_retry_policy_vars_to_defaults(self) -> None:
        defaults = self._retry_policy_default()
        for profile in self.RETRY_POLICY_PROFILES:
            drow = defaults[profile]
            for vars_map in (self.depth_retry_policy_vars, self.inpaint_retry_policy_vars):
                row = vars_map[profile]
                row["garbage_collection_threshold"].set(
                    bool(drow["garbage_collection_threshold"])
                )
                row["expandable_segments"].set(bool(drow["expandable_segments"]))
                row["max_split_size_mb"].set(str(drow["max_split_size_mb"]))
                row["cpu_offload_inherited"].set(bool(drow["cpu_offload_inherited"]))
                row["cpu_offload_mode"].set(str(drow["cpu_offload_mode"]))

    def _set_retry_policy_offload_widget_state(
        self,
        vars_map: dict[str, dict[str, tk.Variable]],
        widget_map: dict[str, ttk.Combobox],
    ) -> None:
        for profile in self.RETRY_POLICY_PROFILES:
            combo = widget_map.get(profile)
            row = vars_map.get(profile)
            if combo is None or row is None:
                continue
            inherited = bool(row["cpu_offload_inherited"].get())
            combo.configure(state=tk.DISABLED if inherited else "readonly")

    def _on_depth_retry_policy_changed(self) -> None:
        self._set_retry_policy_offload_widget_state(
            self.depth_retry_policy_vars,
            self._depth_retry_offload_widgets,
        )
        self._preview_depth_command()

    def _on_inpaint_retry_policy_changed(self) -> None:
        self._set_retry_policy_offload_widget_state(
            self.inpaint_retry_policy_vars,
            self._inpaint_retry_offload_widgets,
        )
        self._preview_inpaint_command()

    def _build_retry_policy_table(
        self,
        parent: ttk.LabelFrame,
        vars_map: dict[str, dict[str, tk.Variable]],
        widget_map: dict[str, ttk.Combobox],
        change_cb,
    ) -> None:
        parent.grid_columnconfigure(6, weight=1)
        ttk.Label(parent, text="Profile").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Label(parent, text="Garbage 0.8").grid(row=0, column=1, sticky="w", padx=(0, 6))
        ttk.Label(parent, text="Expandable").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Label(parent, text="Max split").grid(row=0, column=3, sticky="w", padx=(0, 6))
        ttk.Label(parent, text="CPU mode").grid(row=0, column=4, sticky="w", padx=(0, 6))
        ttk.Label(parent, text="Inherited").grid(row=0, column=5, sticky="w")

        for ridx, profile in enumerate(self.RETRY_POLICY_PROFILES, start=1):
            row = vars_map[profile]
            ttk.Label(parent, text=profile).grid(row=ridx, column=0, sticky="w", pady=2)
            ttk.Checkbutton(
                parent,
                variable=row["garbage_collection_threshold"],
                command=change_cb,
            ).grid(row=ridx, column=1, sticky="w", pady=2)
            ttk.Checkbutton(
                parent,
                variable=row["expandable_segments"],
                command=change_cb,
            ).grid(row=ridx, column=2, sticky="w", pady=2)
            split_combo = ttk.Combobox(
                parent,
                textvariable=row["max_split_size_mb"],
                values=self.RETRY_POLICY_MAX_SPLIT_CHOICES,
                state="readonly",
                width=6,
            )
            split_combo.grid(row=ridx, column=3, sticky="w", pady=2)
            split_combo.bind("<<ComboboxSelected>>", lambda _e: change_cb())
            offload_combo = ttk.Combobox(
                parent,
                textvariable=row["cpu_offload_mode"],
                values=self.RETRY_POLICY_OFFLOAD_CHOICES,
                state="readonly",
                width=10,
            )
            offload_combo.grid(row=ridx, column=4, sticky="w", pady=2)
            offload_combo.bind("<<ComboboxSelected>>", lambda _e: change_cb())
            ttk.Checkbutton(
                parent,
                variable=row["cpu_offload_inherited"],
                command=change_cb,
            ).grid(row=ridx, column=5, sticky="w", pady=2)
            widget_map[profile] = offload_combo

    def _build_options_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(4, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        verify_opts = ttk.LabelFrame(parent, text="VerifyScene (Global)", padding=8)
        verify_opts.grid(row=0, column=0, columnspan=2, sticky="ew", pady=4)
        verify_opts.grid_columnconfigure(3, weight=1)

        ttk.Label(verify_opts, text="Workers (all verify quick/deep):").grid(
            row=0, column=0, sticky="w"
        )
        self.verify_scenes_workers_entry = ttk.Entry(
            verify_opts, textvariable=self.verify_scenes_workers_var, width=7
        )
        self.verify_scenes_workers_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(
            verify_opts,
            text="Used by all Verify Scenes actions in every tab.",
        ).grid(row=0, column=2, columnspan=2, sticky="w")

        depth_opts = ttk.LabelFrame(parent, text="DepthCrafter", padding=8)
        depth_opts.grid(row=1, column=0, sticky="nsew", pady=4, padx=(0, 4))
        depth_opts.grid_columnconfigure(2, weight=1)

        ttk.Label(depth_opts, text="RealESRGAN runtime:").grid(row=0, column=0, sticky="w")
        self.depth_realesrgan_source_combo = ttk.Combobox(
            depth_opts,
            textvariable=self.depth_realesrgan_source_var,
            values=[
                "Bundled (Utilities/realesrgan)",
                "Local (system/custom path)",
            ],
            state="readonly",
            width=34,
        )
        self.depth_realesrgan_source_combo.grid(row=0, column=1, sticky="w", padx=(6, 10))

        ttk.Label(
            depth_opts,
            text=(
                "Bundled uses Utilities/realesrgan/realesrgan-ncnn-vulkan with anime 2x model.\n"
                "Local uses your PATH/custom runtime configuration."
            ),
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        retry_opts = ttk.LabelFrame(parent, text="GPU Retry Policies", padding=8)
        retry_opts.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        retry_opts.grid_columnconfigure(0, weight=1)
        retry_opts.grid_columnconfigure(1, weight=1)

        depth_retry_frame = ttk.LabelFrame(retry_opts, text="DepthCrafter Retry Policy", padding=6)
        depth_retry_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._build_retry_policy_table(
            depth_retry_frame,
            self.depth_retry_policy_vars,
            self._depth_retry_offload_widgets,
            self._on_depth_retry_policy_changed,
        )

        inpaint_retry_frame = ttk.LabelFrame(retry_opts, text="Inpainting Retry Policy", padding=6)
        inpaint_retry_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._build_retry_policy_table(
            inpaint_retry_frame,
            self.inpaint_retry_policy_vars,
            self._inpaint_retry_offload_widgets,
            self._on_inpaint_retry_policy_changed,
        )

        run_opts = ttk.LabelFrame(parent, text="Pipeline Run Controls", padding=8)
        run_opts.grid(row=1, column=1, sticky="nsew", pady=4, padx=(4, 0))
        run_opts.grid_columnconfigure(5, weight=1)

        ttk.Label(run_opts, text="Verify after each step:").grid(row=0, column=0, sticky="w")
        self.pipeline_verify_after_combo = ttk.Combobox(
            run_opts,
            textvariable=self.pipeline_verify_after_var,
            values=self.PIPELINE_VERIFY_CHOICES,
            state="readonly",
            width=12,
        )
        self.pipeline_verify_after_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.pipeline_verify_after_combo.bind(
            "<<ComboboxSelected>>", self._on_pipeline_verify_after_changed
        )
        ttk.Label(
            run_opts,
            textvariable=self.pipeline_checked_files_var,
        ).grid(row=0, column=2, columnspan=4, sticky="w", padx=(8, 0))
        ttk.Label(run_opts, text="Test run files:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(
            run_opts,
            textvariable=self.pipeline_test_run_files_var,
            width=6,
        ).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(6, 0))
        ttk.Label(run_opts, text="(max files from incomplete list)").grid(
            row=1, column=2, columnspan=4, sticky="w", pady=(6, 0)
        )

        step_frame = ttk.LabelFrame(parent, text="Pipeline Step State", padding=8)
        step_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)
        step_frame.grid_columnconfigure(0, weight=1)
        step_frame.grid_columnconfigure(1, weight=0)
        step_frame.grid_columnconfigure(2, weight=0)
        ttk.Label(step_frame, text="Step").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(step_frame, text="Done").grid(row=0, column=1, sticky="w", padx=(12, 20), pady=(0, 4))
        ttk.Label(step_frame, text="Verify").grid(row=0, column=2, sticky="w", pady=(0, 4))

        for idx, (step_key, step_label) in enumerate(self.PIPELINE_STEPS, start=1):
            ttk.Label(step_frame, text=step_label).grid(row=idx, column=0, sticky="w", pady=1)
            done_lbl = tk.Label(step_frame, text="-", width=4, anchor="w", fg="#888888")
            done_lbl.grid(row=idx, column=1, sticky="w", padx=(12, 20), pady=1)
            verify_lbl = tk.Label(step_frame, text="-", width=16, anchor="w", fg="#888888")
            verify_lbl.grid(row=idx, column=2, sticky="w", pady=1)
            self._pipeline_step_widgets[step_key] = {
                "done": done_lbl,
                "verify": verify_lbl,
            }

        run_frame = ttk.LabelFrame(parent, text="Run & Progress", padding=8)
        run_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=6)
        run_frame.grid_columnconfigure(0, weight=1)
        run_frame.grid_rowconfigure(3, weight=1)

        btn_row = ttk.Frame(run_frame)
        btn_row.grid(row=0, column=0, sticky="w")
        self.pipeline_reset_settings_btn = ttk.Button(
            btn_row, text="Reset Settings", command=self._reset_settings_to_defaults
        )
        self.pipeline_reset_settings_btn.grid(row=0, column=0, padx=(0, 8))
        self.pipeline_check_files_btn = ttk.Button(
            btn_row, text="Check Files", command=self._pipeline_check_files
        )
        self.pipeline_check_files_btn.grid(row=0, column=1, padx=8)
        self.pipeline_test_run_btn = ttk.Button(
            btn_row, text="Test Run", command=self._pipeline_test_run
        )
        self.pipeline_test_run_btn.grid(row=0, column=2, padx=8)
        self.pipeline_start_resume_btn = ttk.Button(
            btn_row, text="Start/Resume", command=self._pipeline_start_resume
        )
        self.pipeline_start_resume_btn.grid(row=0, column=3, padx=8)
        self.pipeline_clear_run_btn = ttk.Button(
            btn_row, text="Clear Run", command=self._pipeline_clear_run_flags
        )
        self.pipeline_clear_run_btn.grid(row=0, column=4, padx=8)

        ttk.Label(run_frame, textvariable=self.pipeline_run_status_var).grid(
            row=1, column=0, sticky="w", pady=(8, 4)
        )
        self.pipeline_global_progress = ttk.Progressbar(
            run_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.pipeline_run_progress_var,
            maximum=100.0,
        )
        self.pipeline_global_progress.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        popup_log_frame = ttk.LabelFrame(
            run_frame,
            text="Suppressed Popup Log (Run/Resume/Test)",
            padding=6,
        )
        popup_log_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        popup_log_frame.grid_columnconfigure(0, weight=1)
        popup_log_frame.grid_rowconfigure(0, weight=1)
        self.pipeline_popup_log_text = tk.Text(
            popup_log_frame,
            height=8,
            wrap="word",
            state=tk.DISABLED,
        )
        self.pipeline_popup_log_text.grid(row=0, column=0, sticky="nsew")
        popup_scroll = ttk.Scrollbar(
            popup_log_frame,
            orient=tk.VERTICAL,
            command=self.pipeline_popup_log_text.yview,
        )
        popup_scroll.grid(row=0, column=1, sticky="ns")
        self.pipeline_popup_log_text.configure(yscrollcommand=popup_scroll.set)
        self._flush_pipeline_popup_log_buffer()
        self._on_depth_retry_policy_changed()
        self._on_inpaint_retry_policy_changed()

    def _build_progress_tab(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text="Pipeline orchestration controls (resume/stop/auto-advance) will be wired in next steps.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        checks = ttk.Frame(parent)
        checks.grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(checks, text="Resume enabled", variable=self.resume_enabled_var).grid(
            row=0, column=0, padx=(0, 18)
        )
        ttk.Checkbutton(checks, text="Stop on error", variable=self.stop_on_error_var).grid(
            row=0, column=1, padx=(0, 18)
        )
        ttk.Checkbutton(checks, text="Auto advance", variable=self.auto_advance_var).grid(
            row=0, column=2
        )

    def _install_messagebox_wrappers(self) -> None:
        messagebox.showinfo = self._messagebox_showinfo_wrapper
        messagebox.showwarning = self._messagebox_showwarning_wrapper
        messagebox.showerror = self._messagebox_showerror_wrapper
        messagebox.askyesno = self._messagebox_askyesno_wrapper

    def _restore_messagebox_wrappers(self) -> None:
        for name, fn in self._messagebox_originals.items():
            try:
                setattr(messagebox, name, fn)
            except Exception:
                pass

    def _is_pipeline_popup_suppressed(self) -> bool:
        return bool(
            self._pipeline_ui_noninteractive or self._pipeline_autorun or self._pipeline_test_active
        )

    def _append_pipeline_popup_log(self, level: str, title: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        lvl = str(level).strip().upper() or "INFO"
        ttl = str(title or "Popup").strip() or "Popup"
        msg = str(message or "").rstrip()
        if not msg:
            msg = "(empty)"
        line = f"[{ts}] [{lvl}] {ttl}: {msg}\n"
        if self.pipeline_popup_log_text is None:
            self._pipeline_popup_log_buffer.append(line)
            return
        self.pipeline_popup_log_text.configure(state=tk.NORMAL)
        self.pipeline_popup_log_text.insert(tk.END, line)
        self.pipeline_popup_log_text.see(tk.END)
        self.pipeline_popup_log_text.configure(state=tk.DISABLED)

    def _flush_pipeline_popup_log_buffer(self) -> None:
        if self.pipeline_popup_log_text is None or not self._pipeline_popup_log_buffer:
            return
        self.pipeline_popup_log_text.configure(state=tk.NORMAL)
        for line in self._pipeline_popup_log_buffer:
            self.pipeline_popup_log_text.insert(tk.END, line)
        self.pipeline_popup_log_text.see(tk.END)
        self.pipeline_popup_log_text.configure(state=tk.DISABLED)
        self._pipeline_popup_log_buffer.clear()

    def _pipeline_sync_noninteractive_mode(self) -> None:
        if not self._pipeline_autorun and not self._pipeline_test_active:
            self._pipeline_ui_noninteractive = False

    def _messagebox_proxy(self, kind: str, title: object = "", message: object = "", **kwargs):
        title_txt = str(title or "")
        msg_txt = str(message or "")
        if not self._is_pipeline_popup_suppressed():
            fn = self._messagebox_originals.get(kind)
            if fn is None:
                return None
            return fn(title, message, **kwargs)

        level = "INFO"
        if kind == "showwarning":
            level = "WARN"
        elif kind == "showerror":
            level = "ERROR"
        elif kind == "askyesno":
            level = "ASK"

        if kind == "askyesno":
            self._append_pipeline_popup_log(
                level,
                title_txt,
                f"{msg_txt}\n[AUTO] answer=YES (non-interactive pipeline mode)",
            )
            return True

        self._append_pipeline_popup_log(level, title_txt, msg_txt)
        return "ok"

    def _messagebox_showinfo_wrapper(self, title=None, message=None, **kwargs):
        return self._messagebox_proxy("showinfo", title, message, **kwargs)

    def _messagebox_showwarning_wrapper(self, title=None, message=None, **kwargs):
        return self._messagebox_proxy("showwarning", title, message, **kwargs)

    def _messagebox_showerror_wrapper(self, title=None, message=None, **kwargs):
        return self._messagebox_proxy("showerror", title, message, **kwargs)

    def _messagebox_askyesno_wrapper(self, title=None, message=None, **kwargs):
        return bool(self._messagebox_proxy("askyesno", title, message, **kwargs))

    @staticmethod
    def _pipeline_scene_stem_from_name(scene_name: str) -> str:
        return Path(str(scene_name or "")).stem

    @staticmethod
    def _pipeline_name_matches_scene_stem(file_name: str, scene_stem: str) -> bool:
        name = str(file_name or "")
        stem = str(scene_stem or "")
        if not name or not stem:
            return False
        if name == stem:
            return True
        return name.startswith(stem + "_") or name.startswith(stem + ".")

    @staticmethod
    def _pipeline_link_or_copy_file(src: Path, dst: Path) -> bool:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.symlink(str(src), str(dst))
                return True
            except Exception:
                pass
            try:
                os.link(str(src), str(dst))
                return True
            except Exception:
                pass
            shutil.copy2(str(src), str(dst))
            return True
        except Exception:
            return False

    def _pipeline_collect_scene_matched_files(
        self,
        folder: str,
        patterns: list[str],
        scene_stems: list[str],
        *,
        must_contain: str = "",
    ) -> list[Path]:
        root = Path(folder)
        if not root.is_dir():
            return []
        out: dict[str, Path] = {}
        needle = str(must_contain or "").strip().lower()
        for pat in patterns:
            for p in root.glob(pat):
                if not p.is_file():
                    continue
                name = p.name
                if needle and needle not in name.lower():
                    continue
                if not any(self._pipeline_name_matches_scene_stem(name, s) for s in scene_stems):
                    continue
                out[str(p.resolve())] = p.resolve()
        return [out[k] for k in sorted(out.keys())]

    def _pipeline_link_scene_files(
        self,
        src_dir: str,
        dst_dir: str,
        patterns: list[str],
        scene_stems: list[str],
        *,
        must_contain: str = "",
    ) -> int:
        linked = 0
        src_files = self._pipeline_collect_scene_matched_files(
            src_dir,
            patterns,
            scene_stems,
            must_contain=must_contain,
        )
        dst_root = Path(dst_dir)
        for src in src_files:
            dst = dst_root / src.name
            if self._pipeline_link_or_copy_file(src, dst):
                linked += 1
        return linked

    def _pipeline_seed_test_sharpness_csv(
        self,
        scene_names: list[str],
        src_csv: str,
        dst_csv: str,
    ) -> None:
        if not src_csv or not os.path.isfile(src_csv):
            return
        selected = {str(Path(x).name) for x in scene_names}
        if not selected:
            return
        try:
            with open(src_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                if "file" not in set(fieldnames):
                    return
                rows = []
                for row in reader:
                    file_name = str(Path(str((row or {}).get("file", "")).strip()).name)
                    if file_name in selected:
                        rows.append(dict(row or {}))
            if not fieldnames:
                return
            dst_path = Path(dst_csv)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except Exception:
            return

    def _pipeline_seed_test_autoct_csv(
        self,
        scene_stems: list[str],
        src_csv: str,
        dst_csv: str,
    ) -> None:
        if not src_csv or not os.path.isfile(src_csv):
            return
        stems = [str(s).strip() for s in scene_stems if str(s).strip()]
        if not stems:
            return
        try:
            with open(src_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                if "video" not in set(fieldnames):
                    return
                rows = []
                for row in reader:
                    video_name = str(Path(str((row or {}).get("video", "")).strip()).name)
                    if any(self._pipeline_name_matches_scene_stem(video_name, s) for s in stems):
                        rows.append(dict(row or {}))
            if not fieldnames:
                return
            dst_path = Path(dst_csv)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except Exception:
            return

    def _pipeline_test_has_scene_outputs(
        self,
        folder: str,
        scene_stems: list[str],
        patterns: list[str],
        *,
        must_contain: str = "",
    ) -> bool:
        if not scene_stems:
            return False
        root = Path(folder)
        if not root.is_dir():
            return False
        files = self._pipeline_collect_scene_matched_files(
            folder, patterns, scene_stems, must_contain=must_contain
        )
        names = [p.name for p in files]
        for stem in scene_stems:
            if not any(self._pipeline_name_matches_scene_stem(n, stem) for n in names):
                return False
        return True

    def _pipeline_recompute_test_step_state(self) -> None:
        if not self._pipeline_test_active:
            return
        prev = dict(self._pipeline_test_step_state or {})
        state = self._default_pipeline_step_state()
        scene_names = [str(x) for x in (self._pipeline_test_manifest or [])]
        scene_stems = [self._pipeline_scene_stem_from_name(x) for x in scene_names]
        scene_stems = [s for s in scene_stems if s]

        scene_dir = self.scene_output_var.get().strip()
        scene_files_ready = bool(
            scene_names
            and all((Path(scene_dir) / name).is_file() for name in scene_names)
        )
        state["scenedetect"]["completed"] = scene_files_ready
        state["split_scenes"]["completed"] = scene_files_ready
        state["depthcrafter"]["completed"] = self._pipeline_test_has_scene_outputs(
            self.depth_output_var.get().strip(),
            scene_stems,
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
        )
        state["depth_upscale"]["completed"] = self._pipeline_test_has_scene_outputs(
            self.depth_upscaled_var.get().strip(),
            scene_stems,
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
        )
        state["splatting"]["completed"] = self._pipeline_test_has_scene_outputs(
            self._resolve_splat_hires_dir(),
            scene_stems,
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            must_contain="_splatted",
        )
        state["inpaint"]["completed"] = self._pipeline_test_has_scene_outputs(
            self.inpaint_output_var.get().strip(),
            scene_stems,
            ["*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"],
            must_contain="_inpainted_",
        )
        state["mask_for_merge"]["completed"] = self._pipeline_test_has_scene_outputs(
            self.merge_mask_formerge_var.get().strip(),
            scene_stems,
            ["*_replace_mask.*"],
            must_contain="_replace_mask",
        )
        state["merging"]["completed"] = self._pipeline_test_has_scene_outputs(
            self.merge_output_var.get().strip(),
            scene_stems,
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            must_contain="_merged_",
        )
        state["join"]["completed"] = Path(self.join_output_var.get().strip()).is_file()
        state["remux"]["completed"] = Path(self._default_remux_output_path()).is_file()

        if self._is_pipeline_step_required("sharpness_csv"):
            ok, _msg, _missing = self._verify_sharpness_csv_coverage(
                self.inpaint_input_var.get().strip(),
                self.inpaint_sharpness_csv_var.get().strip(),
            )
            state["sharpness_csv"]["completed"] = bool(ok)
        else:
            state["sharpness_csv"]["completed"] = False

        if self._is_pipeline_step_required("autoct_csv"):
            try:
                ok, _msg, _missing = self._verify_autoct_csv_packet_coverage(
                    inpainted_dir=self.merge_inpainted_var.get().strip(),
                    splatted_dir=self.merge_splatted_var.get().strip(),
                    replace_mask_dir=self.merge_replace_mask_var.get().strip(),
                    csv_path=self.merge_autoct_csv_var.get().strip(),
                    cleanup_incomplete=False,
                )
                state["autoct_csv"]["completed"] = bool(ok)
            except Exception:
                state["autoct_csv"]["completed"] = False
        else:
            state["autoct_csv"]["completed"] = False

        for step, _ in self.PIPELINE_STEPS:
            prev_entry = prev.get(step) if isinstance(prev, dict) else None
            prev_done = bool((prev_entry or {}).get("completed", False)) if isinstance(prev_entry, dict) else False
            prev_ver = str((prev_entry or {}).get("verified", "none")) if isinstance(prev_entry, dict) else "none"
            if bool(state[step]["completed"]) and prev_done:
                state[step]["verified"] = prev_ver if prev_ver in {"quick", "deep"} else "none"
            else:
                state[step]["verified"] = "none"

        self._pipeline_test_step_state = state

    @staticmethod
    def _pipeline_test_path_var_names() -> list[str]:
        return [
            "scene_output_var",
            "depth_input_var",
            "depth_output_var",
            "depth_upscaled_var",
            "splat_input_clips_var",
            "splat_input_depth_var",
            "splat_output_var",
            "splat_mask_output_var",
            "inpaint_input_var",
            "inpaint_mask_var",
            "inpaint_output_var",
            "inpaint_sharpness_csv_var",
            "merge_inpainted_var",
            "merge_splatted_var",
            "merge_original_var",
            "merge_replace_mask_var",
            "merge_mask_formerge_var",
            "merge_output_var",
            "merge_autoct_csv_var",
            "join_input_var",
            "join_output_var",
        ]

    def _default_pipeline_step_state(self) -> dict[str, dict[str, object]]:
        return {
            key: {"completed": False, "verified": "none"}
            for key, _ in self.PIPELINE_STEPS
        }

    def _pipeline_state_path(self) -> Path:
        work_dir = self.work_folder_var.get().strip() or "./work"
        return Path(work_dir).resolve() / self.PIPELINE_STATE_FILENAME

    def _load_pipeline_state(self) -> None:
        state_path = self._pipeline_state_path()
        self._pipeline_step_state = self._default_pipeline_step_state()
        if state_path.is_file():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    steps = data.get("steps")
                    if isinstance(steps, dict):
                        for key, _label in self.PIPELINE_STEPS:
                            entry = steps.get(key)
                            if isinstance(entry, dict):
                                self._pipeline_step_state[key]["completed"] = bool(entry.get("completed", False))
                                ver = str(entry.get("verified", "none")).strip().lower()
                                self._pipeline_step_state[key]["verified"] = ver if ver in {"none", "quick", "deep"} else "none"
                    verify_after = str(data.get("verify_after", "")).strip()
                    if verify_after in self.PIPELINE_VERIFY_CHOICES:
                        self.pipeline_verify_after_var.set(verify_after)
            except Exception:
                pass
        self._sync_pipeline_csv_done_flags()
        self._refresh_pipeline_status_panel()

    def _save_pipeline_state(self) -> None:
        state_path = self._pipeline_state_path()
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "verify_after": self.pipeline_verify_after_var.get().strip(),
                "steps": self._pipeline_step_state,
            }
            state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _is_pipeline_step_required(self, step: str) -> bool:
        if self._pipeline_test_active and step in {"mono_to_sbs", "join", "remux"}:
            return False
        if step == "sharpness_csv":
            return (
                self.inpaint_mode_var.get().strip() == "Auto (recommended)"
                or bool(self.inpaint_use_sharpness_csv_var.get())
            )
        if step == "autoct_csv":
            return self.merge_ct_auto_mode_var.get().strip() == "CSV Blend"
        return True

    def _refresh_pipeline_status_panel(self) -> None:
        if self._pipeline_test_active:
            self._pipeline_recompute_test_step_state()
            step_state = self._pipeline_test_step_state
        else:
            step_state = self._pipeline_step_state
        done_count = 0
        total_count = len(self.PIPELINE_STEPS)
        for step, _label in self.PIPELINE_STEPS:
            required = self._is_pipeline_step_required(step)
            st = step_state.get(step, {"completed": False, "verified": "none"})
            done = bool(st.get("completed", False))
            verified = str(st.get("verified", "none"))
            widgets = self._pipeline_step_widgets.get(step, {})
            done_w = widgets.get("done")
            verify_w = widgets.get("verify")
            if not required:
                if done_w is not None:
                    done_w.configure(text="-", fg="#999999")
                if verify_w is not None:
                    verify_w.configure(text="N/A", fg="#999999")
                done_count += 1
                continue
            if done:
                done_count += 1
            if done_w is not None:
                if done:
                    done_w.configure(text="V", fg="#1a8f35")
                else:
                    done_w.configure(text="-", fg="#888888")
            if verify_w is not None:
                if verified == "quick":
                    verify_w.configure(text="V (quick)", fg="#1a8f35")
                elif verified == "deep":
                    verify_w.configure(text="V (deep)", fg="#1a8f35")
                else:
                    verify_w.configure(text="-", fg="#888888")

        progress = 0.0 if total_count <= 0 else (float(done_count) / float(total_count)) * 100.0
        self.pipeline_run_progress_var.set(max(0.0, min(100.0, progress)))

    def _on_pipeline_verify_after_changed(self, _event=None) -> None:
        self._save_pipeline_state()
        self._refresh_pipeline_status_panel()

    def _pipeline_clear_run_flags(self) -> None:
        if self._any_pipeline_activity():
            messagebox.showwarning(
                "Clear Run",
                "Stop all running tasks before clearing run flags.",
            )
            return
        if not messagebox.askyesno(
            "Clear Run",
            (
                "Clear all pipeline DONE/VERIFIED flags?\n\n"
                "Settings and files will not be changed."
            ),
        ):
            return

        self._pipeline_autorun = False
        self._pipeline_pending_action = None
        self._pipeline_pause_after_split_scenes = False
        self._pipeline_step_state = self._default_pipeline_step_state()
        self._pipeline_test_step_state = self._default_pipeline_step_state()
        self.pipeline_checked_files_var.set("Check Files: not run")
        self.pipeline_run_status_var.set("Run flags cleared.")
        self._refresh_pipeline_status_panel()
        self._save_pipeline_state()

    def _reset_settings_to_defaults(self) -> None:
        if self._any_pipeline_activity():
            messagebox.showwarning(
                "Reset Settings",
                "Stop all running tasks before resetting settings.",
            )
            return
        if not messagebox.askyesno(
            "Reset Settings",
            (
                "Reset all GUI settings to defaults?\n\n"
                "Pipeline completion flags (DONE/VERIFIED) will be kept."
            ),
        ):
            return

        # SceneDetect defaults.
        self.scene_detector_var.set("Adaptive")
        self.scene_threshold_var.set("2.0")
        self.scene_backend_var.set(self.DEFAULT_SCENE_BACKEND)
        self.scene_crop_mode_var.set("auto")
        self.scene_crop_custom_var.set("")
        self.scene_crop_target_h_var.set("")
        self.scene_layout_var.set("Full-SBS (quality)")
        self.scene_tonemap_var.set(
            "Mobius (HDR style, available only for 10-bit input source)"
        )
        self.scene_chroma_var.set("420")
        self.scene_codec_var.set(self.DEFAULT_SCENE_CODEC)
        self.scene_crf_var.set("1")
        self.scene_encoder_preset_var.set("fast")
        self.scene_pix_fmt_var.set(self._chroma_to_pixfmt(self.scene_chroma_var.get().strip()))
        self.scene_extra_ffmpeg_args_var.set("")

        # Depth defaults.
        self.depth_mode_var.set("Auto (recommended)")
        self.depth_chunk_size_var.set("70")
        self.depth_overlap_var.set("20")
        self.depth_inference_steps_var.set("5")
        self.depth_cpu_offload_var.set("model")
        self.depth_seed_var.set("42")
        self.depth_guidance_scale_var.set("1.0")
        self.depth_decode_chunk_size_var.set("2")
        self.depth_restart_every_var.set("100")
        self.depth_debug_mem_var.set(True)
        self.depth_scale_factor_var.set(self.DEFAULT_DEPTH_SCALE_FACTOR)
        self.depth_glob_var.set("*.mp4")
        self.depth_worker_script_var.set("./depthcrafter_nogui_batch.py")
        self.depth_encode_override_var.set(False)
        self.depth_extra_ffmpeg_args_var.set("")
        self.depth_realesrgan_source_var.set("Bundled (Utilities/realesrgan)")
        self.depth_realesrgan_workers_var.set(str(self.DEFAULT_DEPTH_REALESRGAN_WORKERS))
        self._on_depth_mode_changed()
        self._on_depth_override_toggle(initial=True)

        # Splat defaults.
        self.splat_mode_var.set("Auto (recommended)")
        self.splat_batch_size_var.set("50")
        self.splat_workers_var.set("2")
        self.splat_disparity_var.set("20")
        self.splat_encode_override_var.set(False)
        self.splat_extra_ffmpeg_args_var.set("")
        self._on_splat_mode_changed()
        self._on_splat_override_toggle(initial=True)

        # Inpaint defaults.
        self.inpaint_mode_var.set("Auto (recommended)")
        self.inpaint_frames_chunk_var.set("50")
        self.inpaint_cpu_offload_var.set("model")
        self.inpaint_sharpness_workers_var.set("19")
        self.inpaint_encode_override_var.set(False)
        self.inpaint_extra_ffmpeg_args_var.set("")
        self._on_inpaint_mode_changed()
        self._on_inpaint_override_toggle(initial=True)

        # Merge defaults.
        self.merge_mode_var.set("Auto (recommended)")
        self.merge_encode_override_var.set(False)
        self.merge_extra_ffmpeg_args_var.set("")
        self._on_merge_mode_changed()
        self._on_merge_override_toggle(initial=True)

        # Join defaults.
        self.join_mode_var.set("Auto (recommended)")
        self.join_crf_var.set("16")
        self.join_pix_fmt_override_var.set(False)
        self._on_join_mode_changed()
        self._on_join_pixfmt_override_toggle()

        # Options defaults.
        self.scene_split_threads_var.set(str(self.DEFAULT_SPLIT_SCENES_WORKERS))
        self.verify_scenes_workers_var.set("19")
        self.pipeline_verify_after_var.set("Quick")
        self.pipeline_test_run_files_var.set(str(self.DEFAULT_PIPELINE_TEST_RUN_FILES))
        self._set_retry_policy_vars_to_defaults()
        self._on_depth_retry_policy_changed()
        self._on_inpaint_retry_policy_changed()
        self.resume_enabled_var.set(True)
        self.stop_on_error_var.set(True)
        self.auto_advance_var.set(False)

        self._refresh_standard_paths()
        self._apply_option_states()
        self._preview_scene_command()
        self._refresh_pipeline_status_panel()
        self.pipeline_run_status_var.set("Settings reset to defaults.")
        self._save_config()
        self._save_pipeline_state()

    @staticmethod
    def _pipeline_set_completed_in_state(
        state: dict[str, dict[str, object]],
        step: str,
        value: bool,
    ) -> None:
        if step not in state:
            return
        state[step]["completed"] = bool(value)
        if not value:
            state[step]["verified"] = "none"

    @staticmethod
    def _pipeline_set_verified_in_state(
        state: dict[str, dict[str, object]],
        step: str,
        mode: str,
    ) -> None:
        if step not in state:
            return
        mode_low = str(mode).strip().lower()
        state[step]["verified"] = mode_low if mode_low in {"quick", "deep"} else "none"

    def _pipeline_set_completed(self, step: str, value: bool) -> None:
        self._pipeline_set_completed_in_state(self._pipeline_step_state, step, value)

    def _pipeline_set_verified(self, step: str, mode: str) -> None:
        self._pipeline_set_verified_in_state(self._pipeline_step_state, step, mode)

    @staticmethod
    def _pipeline_verified_rank(mode: str) -> int:
        mode_low = str(mode).strip().lower()
        if mode_low == "deep":
            return 2
        if mode_low == "quick":
            return 1
        return 0

    def _pipeline_set_verified_best_in_state(
        self,
        state: dict[str, dict[str, object]],
        step: str,
        mode: str,
    ) -> None:
        if step not in state:
            return
        mode_low = "deep" if str(mode).strip().lower() == "deep" else "quick"
        current = str(state[step].get("verified", "none"))
        if self._pipeline_verified_rank(mode_low) >= self._pipeline_verified_rank(current):
            self._pipeline_set_verified_in_state(state, step, mode_low)

    def _pipeline_set_verified_best(self, step: str, mode: str) -> None:
        self._pipeline_set_verified_best_in_state(self._pipeline_step_state, step, mode)

    def _sync_pipeline_csv_done_flags_in_state(
        self,
        state: dict[str, dict[str, object]],
    ) -> None:
        sharp_done = Path(self.inpaint_sharpness_csv_var.get().strip()).is_file()
        autoct_done = Path(self.merge_autoct_csv_var.get().strip()).is_file()
        self._pipeline_set_completed_in_state(state, "sharpness_csv", sharp_done)
        self._pipeline_set_completed_in_state(state, "autoct_csv", autoct_done)
        self._pipeline_set_verified_in_state(state, "sharpness_csv", "none")
        self._pipeline_set_verified_in_state(state, "autoct_csv", "none")

    def _sync_pipeline_csv_done_flags(self) -> None:
        self._sync_pipeline_csv_done_flags_in_state(self._pipeline_step_state)

    def _pipeline_mark_previous_steps_done_verified_in_state(
        self,
        state: dict[str, dict[str, object]],
        step: str,
        mode: str,
    ) -> None:
        step_keys = [k for k, _ in self.PIPELINE_STEPS]
        if step not in step_keys:
            return
        target_mode = "deep" if str(mode).strip().lower() == "deep" else "quick"
        upto_idx = step_keys.index(step)
        for k in step_keys[: upto_idx + 1]:
            if k in self.PIPELINE_CSV_STEPS:
                continue
            self._pipeline_set_completed_in_state(state, k, True)
            if k in self.PIPELINE_STEPS_WITH_VERIFY:
                self._pipeline_set_verified_best_in_state(state, k, target_mode)

    def _pipeline_mark_previous_steps_done_verified(self, step: str, mode: str) -> None:
        self._pipeline_mark_previous_steps_done_verified_in_state(
            self._pipeline_step_state, step, mode
        )

    def _pipeline_invalidate_from_in_state(
        self,
        state: dict[str, dict[str, object]],
        step: str,
        include_current: bool = True,
    ) -> None:
        step_keys = [k for k, _ in self.PIPELINE_STEPS]
        if step not in step_keys:
            return
        start = step_keys.index(step)
        if not include_current:
            start += 1
        for k in step_keys[start:]:
            self._pipeline_set_completed_in_state(state, k, False)

    def _pipeline_invalidate_from(self, step: str, include_current: bool = True) -> None:
        self._pipeline_invalidate_from_in_state(
            self._pipeline_step_state,
            step,
            include_current=include_current,
        )
        self._refresh_pipeline_status_panel()
        self._save_pipeline_state()

    def _pipeline_invalidate_active_from(
        self, step: str, include_current: bool = True
    ) -> None:
        state = (
            self._pipeline_test_step_state
            if self._pipeline_test_active
            else self._pipeline_step_state
        )
        self._pipeline_invalidate_from_in_state(
            state, step, include_current=include_current
        )
        self._refresh_pipeline_status_panel()
        if not self._pipeline_test_active:
            self._save_pipeline_state()

    @staticmethod
    def _default_verify_scenes_workers() -> int:
        return 19

    def _get_verify_scenes_workers(self) -> int:
        default_workers = self._default_verify_scenes_workers()
        raw = self.verify_scenes_workers_var.get().strip()
        try:
            workers = int(raw)
        except Exception:
            workers = default_workers
        if workers < 1:
            workers = default_workers
        if str(workers) != raw:
            self.verify_scenes_workers_var.set(str(workers))
        return workers

    def _scene_csv_path(self) -> str:
        work_dir = self.work_folder_var.get().strip() or "./work"
        return os.path.normpath(os.path.join(work_dir, "scenedetect.csv"))

    def _get_scene_split_workers(self) -> int:
        raw = self.scene_split_threads_var.get().strip()
        try:
            workers = int(raw)
        except Exception:
            workers = self.DEFAULT_SPLIT_SCENES_WORKERS
        if workers < 1:
            workers = self.DEFAULT_SPLIT_SCENES_WORKERS
        if str(workers) != raw:
            self.scene_split_threads_var.set(str(workers))
        return workers

    def _get_pipeline_test_run_limit(self) -> int:
        raw = self.pipeline_test_run_files_var.get().strip()
        try:
            count = int(raw)
        except Exception:
            count = self.DEFAULT_PIPELINE_TEST_RUN_FILES
        if count < 1:
            count = self.DEFAULT_PIPELINE_TEST_RUN_FILES
        if str(count) != raw:
            self.pipeline_test_run_files_var.set(str(count))
        return count

    @staticmethod
    def _parse_scene_seconds_or_timecode(value: str) -> float | None:
        txt = str(value or "").strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            pass
        m = re.match(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$", txt)
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3))
        frac = m.group(4) or "0"
        frac_sec = float(f"0.{frac}")
        return float(hh * 3600 + mm * 60 + ss) + frac_sec

    @staticmethod
    def _scene_output_filename(source_path: str, scene_number: int) -> str:
        stem = Path(str(source_path or "")).stem or "source"
        return f"{stem}-Scene-{int(scene_number):03d}.mp4"

    def _load_scene_csv_entries(
        self,
        scene_csv_path: str | None = None,
    ) -> tuple[list[dict[str, float | int]], str]:
        csv_path = str(scene_csv_path or self._scene_csv_path()).strip()
        if not csv_path or not os.path.isfile(csv_path):
            return [], f"Scene CSV not found: {csv_path or '(empty)'}"
        rows: list[dict[str, float | int]] = []
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    if not isinstance(row, dict):
                        continue
                    norm = {
                        str(k or "").strip().lower(): str(v or "").strip()
                        for k, v in row.items()
                    }
                    scene_raw = (
                        norm.get("scene number")
                        or norm.get("scene")
                        or norm.get("scene #")
                        or ""
                    )
                    try:
                        scene_num = int(float(scene_raw)) if scene_raw else idx
                    except Exception:
                        scene_num = idx
                    start_raw = (
                        norm.get("start time (seconds)")
                        or norm.get("start seconds")
                        or norm.get("start timecode")
                        or ""
                    )
                    end_raw = (
                        norm.get("end time (seconds)")
                        or norm.get("end seconds")
                        or norm.get("end timecode")
                        or ""
                    )
                    start_sec = self._parse_scene_seconds_or_timecode(start_raw)
                    end_sec = self._parse_scene_seconds_or_timecode(end_raw)
                    if start_sec is None or end_sec is None:
                        continue
                    if float(end_sec) <= float(start_sec):
                        continue
                    rows.append(
                        {
                            "scene_number": int(scene_num),
                            "start_sec": float(start_sec),
                            "end_sec": float(end_sec),
                        }
                    )
        except Exception as exc:
            return [], f"Failed reading Scene CSV: {type(exc).__name__}: {exc}"
        if not rows:
            return [], f"No valid scene rows found in CSV: {csv_path}"
        return rows, ""

    def _collect_expected_split_scene_outputs(
        self,
        seg_dir: str,
        scene_csv_path: str | None = None,
    ) -> tuple[list[str], list[str], str]:
        entries, err = self._load_scene_csv_entries(scene_csv_path)
        if err:
            return [], [], err
        input_path = self.scene_input_var.get().strip()
        seg_root = Path(seg_dir).resolve()
        seg_mono_root: Path | None = None
        if not self._pipeline_test_active:
            try:
                seg_mono_candidate = Path(
                    self.work_folder_var.get().strip() or "./work"
                ).resolve() / "seg-mono"
                if seg_mono_candidate.is_dir() and seg_mono_candidate.resolve() != seg_root:
                    seg_mono_root = seg_mono_candidate.resolve()
            except Exception:
                seg_mono_root = None
        expected: list[str] = []
        missing: list[str] = []
        for idx, entry in enumerate(entries, start=1):
            scene_num = int(entry.get("scene_number", idx))
            out_name = self._scene_output_filename(input_path, scene_num)
            expected_path = str((seg_root / out_name).resolve())
            expected.append(expected_path)
            in_seg = Path(expected_path).is_file()
            in_seg_mono = bool(seg_mono_root and (seg_mono_root / out_name).is_file())
            if not (in_seg or in_seg_mono):
                missing.append(expected_path)
        return expected, missing, ""

    @staticmethod
    def _count_video_files(folder: str) -> int:
        if not folder or not os.path.isdir(folder):
            return 0
        exts = ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm")
        count = 0
        root = Path(folder)
        for ext in exts:
            count += len([p for p in root.glob(ext) if p.is_file()])
        return count

    def _pipeline_check_files(self, show_popup: bool = True) -> bool:
        seg_dir = self.scene_output_var.get().strip()
        out_dir = self.merge_output_var.get().strip()
        seg_files = sorted([p for p in Path(seg_dir).glob("*.mp4") if p.is_file()]) if os.path.isdir(seg_dir) else []
        scene_csv_path = self._scene_csv_path()
        scene_entries, scene_csv_err = self._load_scene_csv_entries(scene_csv_path)
        scene_csv_ok = bool(scene_entries) and not scene_csv_err

        expected_outputs, missing_split_outputs, split_cov_err = self._collect_expected_split_scene_outputs(
            seg_dir
        )
        split_expected_count = len(expected_outputs)
        split_missing_count = len(missing_split_outputs)

        if (not seg_files) and (not scene_csv_ok):
            self.pipeline_checked_files_var.set("Check Files: missing scene CSV and split files")
            if show_popup:
                messagebox.showwarning(
                    "Check Files",
                    (
                        "No scene CSV and no split scene files found.\n\n"
                        "Run SceneDetect first."
                    ),
                )
            return False

        incomplete: list[str] = []
        completed: list[str] = []
        if os.path.isdir(out_dir):
            for p in seg_files:
                stem = p.stem
                patt = str(Path(out_dir) / f"{stem}_*_merged_*.*")
                if glob.glob(patt):
                    completed.append(p.name)
                else:
                    incomplete.append(p.name)
        else:
            incomplete = [p.name for p in seg_files]

        seg_count = len(seg_files)
        depth_count = self._count_video_files(self.depth_output_var.get().strip())
        depth_upscaled_count = self._count_video_files(self.depth_upscaled_var.get().strip())
        splat_count = self._count_video_files(self._resolve_splat_hires_dir())
        inpaint_count = self._count_video_files(self.inpaint_output_var.get().strip())
        mask_formerge_count = self._count_video_files(self.merge_mask_formerge_var.get().strip())
        merge_count = self._count_video_files(self.merge_output_var.get().strip())
        mono_to_sbs_ok, _mono_msg, _mono_broken_output, _mono_broken_reference = (
            self._verify_join_mono_outputs_coverage(cleanup_incomplete=False)
        )
        join_done = Path(self.join_output_var.get().strip()).is_file()
        remux_done = Path(self._default_remux_output_path()).is_file()
        sharp_done = Path(self.inpaint_sharpness_csv_var.get().strip()).is_file()
        autoct_done = Path(self.merge_autoct_csv_var.get().strip()).is_file()
        split_ok = bool(scene_csv_ok and split_expected_count > 0 and split_missing_count == 0 and not split_cov_err)
        split_ref_count = split_expected_count if split_expected_count > 0 else seg_count

        self._pipeline_set_completed("scenedetect", bool(scene_csv_ok))
        self._pipeline_set_completed("split_scenes", split_ok)
        self._pipeline_set_completed(
            "depthcrafter", split_ok and split_ref_count > 0 and depth_count >= split_ref_count
        )
        self._pipeline_set_completed(
            "depth_upscale", split_ok and split_ref_count > 0 and depth_upscaled_count >= split_ref_count
        )
        self._pipeline_set_completed(
            "splatting", split_ok and split_ref_count > 0 and splat_count >= split_ref_count
        )
        self._pipeline_set_completed("sharpness_csv", sharp_done)
        self._pipeline_set_completed(
            "inpaint", split_ok and split_ref_count > 0 and inpaint_count >= split_ref_count
        )
        self._pipeline_set_completed("autoct_csv", autoct_done)
        self._pipeline_set_completed(
            "mask_for_merge", split_ok and split_ref_count > 0 and mask_formerge_count >= split_ref_count
        )
        self._pipeline_set_completed(
            "merging", split_ok and split_ref_count > 0 and merge_count >= split_ref_count
        )
        self._pipeline_set_completed("mono_to_sbs", bool(mono_to_sbs_ok))
        self._pipeline_set_completed("join", bool(join_done))
        self._pipeline_set_completed("remux", bool(remux_done))

        self._pipeline_file_scan = {
            "seg_total": seg_count,
            "seg_expected": split_ref_count,
            "split_missing": [str(Path(p).name) for p in missing_split_outputs],
            "completed_final": completed,
            "incomplete_final": incomplete,
        }
        self._pipeline_check_files_done = True
        self.pipeline_checked_files_var.set(
            (
                f"Check Files: csv={'ok' if scene_csv_ok else 'missing'}, "
                f"split={seg_count}/{split_ref_count}, "
                f"final done={len(completed)}, incomplete={len(incomplete)}"
            )
        )
        self._refresh_pipeline_status_panel()
        self._save_pipeline_state()

        if show_popup:
            csv_details = ""
            if scene_csv_err:
                csv_details = f"\nScene CSV details: {scene_csv_err}"
            elif split_cov_err:
                csv_details = f"\nSplit CSV details: {split_cov_err}"
            messagebox.showinfo(
                "Check Files",
                (
                    f"Scan completed.\n\n"
                    f"Scene CSV: {'OK' if scene_csv_ok else 'MISSING'}\n"
                    f"Split files: {seg_count}/{split_ref_count}\n"
                    f"Missing split files: {split_missing_count}\n"
                    f"Final completed: {len(completed)}\n"
                    f"Incomplete: {len(incomplete)}"
                    f"{csv_details}"
                ),
            )
        return True

    def _pipeline_test_run(self) -> None:
        self._pipeline_pause_after_split_scenes = False
        if self._any_pipeline_activity():
            messagebox.showinfo("Test Run", "Stop running tasks before starting a test run.")
            self._pipeline_sync_noninteractive_mode()
            return
        if not self._pipeline_check_files_done:
            if not self._pipeline_check_files(show_popup=False):
                self._pipeline_sync_noninteractive_mode()
                return
        split_state = self._pipeline_step_state.get(
            "split_scenes", {"completed": False, "verified": "none"}
        )
        if not bool(split_state.get("completed", False)):
            messagebox.showwarning(
                "Test Run",
                "Complete Split Scenes first. Test Run requires scene clips.",
            )
            self._pipeline_sync_noninteractive_mode()
            return
        split_verified = str(split_state.get("verified", "none")).strip().lower()
        if split_verified not in {"quick", "deep"}:
            messagebox.showwarning(
                "Test Run",
                (
                    "Run Verify Scenes (Quick or Deep) first.\n\n"
                    "Test Run can start only after scene verification."
                ),
            )
            self._pipeline_sync_noninteractive_mode()
            return

        incomplete = list(self._pipeline_file_scan.get("incomplete_final") or [])
        if not incomplete:
            messagebox.showinfo("Test Run", "No incomplete files found. Nothing to test.")
            self._pipeline_sync_noninteractive_mode()
            return
        test_run_limit = self._get_pipeline_test_run_limit()
        selected = incomplete[:test_run_limit]
        preview = "\n".join(selected)
        if not messagebox.askyesno(
            "Test Run",
            (
                f"Test run will process these files (max {test_run_limit}):\n\n"
                f"{preview}\n\nProceed?"
            ),
        ):
            self._pipeline_sync_noninteractive_mode()
            return

        if not self._prepare_test_scene_subset(selected):
            self._pipeline_sync_noninteractive_mode()
            return
        self._pipeline_ui_noninteractive = True
        self._append_pipeline_popup_log(
            "INFO",
            "Test Run",
            "Non-interactive mode enabled: popups are suppressed and auto-accepted.",
        )
        self.pipeline_run_status_var.set("Test run active (isolated subset)")
        self._pipeline_autorun = True
        self._pipeline_pending_action = None
        self._pipeline_trigger_next_action()

    def _prepare_test_scene_subset(self, scene_names: list[str]) -> bool:
        work_root = Path(self.work_folder_var.get().strip() or "./work").resolve()
        source_seg = Path(self.scene_output_var.get().strip()).resolve()
        if not source_seg.is_dir():
            source_seg = work_root / self.STANDARD_SUBDIRS["scenes"]
        if not source_seg.is_dir():
            messagebox.showerror("Test Run", f"seg folder not found:\n{source_seg}")
            return False

        selected: list[str] = []
        for name in scene_names:
            n = str(name or "").strip()
            if not n:
                continue
            if (source_seg / n).is_file():
                selected.append(n)
        if not selected:
            messagebox.showerror("Test Run", "No valid scene files found for test subset.")
            return False

        if self._pipeline_test_active:
            self._restore_test_scene_subset()

        test_root = work_root / ".pipeline_test_subset"
        try:
            if test_root.exists():
                shutil.rmtree(test_root, ignore_errors=True)
            test_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Test Run", f"Failed to prepare test subset root:\n{exc}")
            return False

        prev_paths: dict[str, str] = {}
        try:
            for var_name in self._pipeline_test_path_var_names():
                var_obj = getattr(self, var_name, None)
                if var_obj is not None:
                    prev_paths[var_name] = str(var_obj.get()).strip()
        except Exception as exc:
            messagebox.showerror("Test Run", f"Failed to capture current paths:\n{exc}")
            return False

        scene_stems = [self._pipeline_scene_stem_from_name(x) for x in selected]
        scene_stems = [s for s in scene_stems if s]

        test_seg = test_root / "seg"
        test_depth = test_root / "depthmap"
        test_depth_up = test_depth / "upscaled"
        test_splat_root = test_root / "splat"
        test_splat_hires = test_splat_root / "hires"
        test_mask = test_root / "mask"
        test_output = test_root / "output"
        test_mask_formerge = test_root / "mask_for_merge"
        test_sbs = test_root / "sbs"
        test_final = test_root / "final"
        for d in (
            test_seg,
            test_depth,
            test_depth_up,
            test_splat_root,
            test_splat_hires,
            test_mask,
            test_output,
            test_mask_formerge,
            test_sbs,
            test_final,
        ):
            d.mkdir(parents=True, exist_ok=True)

        for name in selected:
            src = source_seg / name
            dst = test_seg / name
            self._pipeline_link_or_copy_file(src, dst)

        depth_src = prev_paths.get("depth_output_var", "")
        depth_up_src = prev_paths.get("depth_upscaled_var", "")
        splat_src_root = prev_paths.get("splat_output_var", "")
        if splat_src_root:
            sroot = Path(splat_src_root)
            splat_hires_src = str((sroot / "hires").resolve()) if (sroot / "hires").is_dir() else str(sroot.resolve())
        else:
            splat_hires_src = ""
        mask_src = prev_paths.get("splat_mask_output_var", "")
        inpaint_src = prev_paths.get("inpaint_output_var", "")
        mask_formerge_src = prev_paths.get("merge_mask_formerge_var", "")
        merge_src = prev_paths.get("merge_output_var", "")

        self._pipeline_link_scene_files(
            depth_src,
            str(test_depth),
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            scene_stems,
        )
        self._pipeline_link_scene_files(
            depth_up_src,
            str(test_depth_up),
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            scene_stems,
        )
        self._pipeline_link_scene_files(
            splat_hires_src,
            str(test_splat_hires),
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            scene_stems,
            must_contain="_splatted",
        )
        self._pipeline_link_scene_files(
            mask_src,
            str(test_mask),
            ["*_replace_mask.*"],
            scene_stems,
            must_contain="_replace_mask",
        )
        self._pipeline_link_scene_files(
            inpaint_src,
            str(test_output),
            ["*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"],
            scene_stems,
            must_contain="_inpainted_",
        )
        self._pipeline_link_scene_files(
            mask_formerge_src,
            str(test_mask_formerge),
            ["*_replace_mask.*"],
            scene_stems,
            must_contain="_replace_mask",
        )
        self._pipeline_link_scene_files(
            merge_src,
            str(test_sbs),
            ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            scene_stems,
            must_contain="_merged_",
        )

        sharp_csv_test = test_root / "sharpness_test.csv"
        autoct_csv_test = test_root / "autoct_test.csv"
        self._pipeline_seed_test_sharpness_csv(
            selected,
            prev_paths.get("inpaint_sharpness_csv_var", ""),
            str(sharp_csv_test),
        )
        self._pipeline_seed_test_autoct_csv(
            scene_stems,
            prev_paths.get("merge_autoct_csv_var", ""),
            str(autoct_csv_test),
        )

        self.scene_output_var.set(str(test_seg))
        self.depth_input_var.set(str(test_seg))
        self.depth_output_var.set(str(test_depth))
        self.depth_upscaled_var.set(str(test_depth_up))
        self.splat_input_clips_var.set(str(test_seg))
        self.splat_input_depth_var.set(str(test_depth_up))
        self.splat_output_var.set(str(test_splat_root))
        self.splat_mask_output_var.set(str(test_mask))
        self.inpaint_input_var.set(str(test_splat_hires))
        self.inpaint_mask_var.set(str(test_mask))
        self.inpaint_output_var.set(str(test_output))
        self.inpaint_sharpness_csv_var.set(str(sharp_csv_test))
        self.merge_inpainted_var.set(str(test_output))
        self.merge_splatted_var.set(str(test_splat_hires))
        self.merge_original_var.set(str(test_seg))
        self.merge_replace_mask_var.set(str(test_mask))
        self.merge_mask_formerge_var.set(str(test_mask_formerge))
        self.merge_output_var.set(str(test_sbs))
        self.merge_autoct_csv_var.set(str(autoct_csv_test))
        self.join_input_var.set(str(test_sbs))
        self.join_output_var.set(str(test_final / "final_sbs_1080_hevc_nvenc.mp4"))

        self._pipeline_test_active = True
        self._pipeline_test_manifest = selected
        self._pipeline_test_scene_stems = scene_stems
        self._pipeline_test_source_dir = str(source_seg)
        self._pipeline_test_dir = str(test_root)
        self._pipeline_test_prev_paths = dict(prev_paths)
        self._pipeline_test_step_state = self._default_pipeline_step_state()
        self._pipeline_recompute_test_step_state()

        self._preview_depth_command()
        self._preview_splat_command()
        self._preview_inpaint_command()
        self._preview_merge_command()
        self._preview_join_command()
        self._refresh_pipeline_status_panel()
        return True

    @staticmethod
    def _copy_or_replace_file(src: Path, dst: Path) -> bool:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                try:
                    if os.path.samefile(str(src), str(dst)):
                        return True
                except Exception:
                    pass
                try:
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                except Exception:
                    return False
            shutil.copy2(str(src), str(dst), follow_symlinks=True)
            return True
        except Exception:
            return False

    def _pipeline_sync_test_scene_files(
        self,
        src_dir: Path,
        dst_dir: str,
        patterns: list[str],
        scene_stems: list[str],
        *,
        must_contain: str = "",
    ) -> tuple[int, int]:
        if not scene_stems:
            return 0, 0
        if not str(dst_dir or "").strip():
            return 0, 0
        if not src_dir.is_dir():
            return 0, 0
        dst_root = Path(dst_dir).expanduser().resolve()
        copied = 0
        errors = 0
        src_files = self._pipeline_collect_scene_matched_files(
            str(src_dir),
            patterns,
            scene_stems,
            must_contain=must_contain,
        )
        for src in src_files:
            dst = dst_root / src.name
            if self._copy_or_replace_file(src, dst):
                copied += 1
            else:
                errors += 1
        return copied, errors

    def _pipeline_merge_test_csv_rows(
        self,
        test_csv: Path,
        dst_csv: str,
        key_fields: str | list[str] | tuple[str, ...],
    ) -> tuple[int, int]:
        if not test_csv.is_file():
            return 0, 0
        dst_txt = str(dst_csv or "").strip()
        if not dst_txt:
            return 0, 0
        if isinstance(key_fields, str):
            key_names = [str(key_fields).strip()]
        else:
            key_names = [str(k).strip() for k in (key_fields or []) if str(k).strip()]
        key_names = [k for k in key_names if k]
        if not key_names:
            return 0, 1

        def _row_key(row: dict[str, str]) -> tuple[str, ...] | None:
            parts: list[str] = []
            for field in key_names:
                value = str((row or {}).get(field, "")).strip()
                if not value:
                    return None
                parts.append(value)
            return tuple(parts)

        try:
            with open(test_csv, "r", newline="", encoding="utf-8") as f:
                src_reader = csv.DictReader(f)
                src_fieldnames = [str(x) for x in (src_reader.fieldnames or []) if str(x).strip()]
                if any(field not in src_fieldnames for field in key_names):
                    return 0, 1
                src_rows = [dict(r or {}) for r in src_reader]
        except Exception:
            return 0, 1

        src_index: dict[tuple[str, ...], dict[str, str]] = {}
        src_order: list[tuple[str, ...]] = []
        for row in src_rows:
            key = _row_key(dict(row or {}))
            if key is None:
                continue
            if key not in src_index:
                src_order.append(key)
            src_index[key] = dict(row or {})
        if not src_index:
            return 0, 0

        dst_path = Path(dst_txt).expanduser().resolve()
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return 0, 1

        dst_rows_by_key: dict[tuple[str, ...], dict[str, str]] = {}
        write_order: list[tuple[str, ...]] = []
        dst_fieldnames: list[str] = []
        if dst_path.is_file():
            try:
                with open(dst_path, "r", newline="", encoding="utf-8") as f:
                    dst_reader = csv.DictReader(f)
                    dst_fieldnames = [str(x) for x in (dst_reader.fieldnames or []) if str(x).strip()]
                    for field in key_names:
                        if field not in dst_fieldnames:
                            dst_fieldnames.append(field)
                    for row in dst_reader:
                        row_dict = dict(row or {})
                        key = _row_key(row_dict)
                        if key is None:
                            continue
                        if key not in dst_rows_by_key:
                            write_order.append(key)
                        dst_rows_by_key[key] = row_dict
            except Exception:
                return 0, 1

        if not dst_fieldnames:
            dst_fieldnames = list(src_fieldnames)
        for field in src_fieldnames:
            if field not in dst_fieldnames:
                dst_fieldnames.append(field)
        for field in reversed(key_names):
            if field in dst_fieldnames:
                dst_fieldnames = [f for f in dst_fieldnames if f != field]
            dst_fieldnames.insert(0, field)

        merged_count = 0
        for key in src_order:
            row = dict(src_index[key])
            if key not in dst_rows_by_key:
                write_order.append(key)
            dst_rows_by_key[key] = row
            merged_count += 1

        try:
            with open(dst_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=dst_fieldnames)
                writer.writeheader()
                for key in write_order:
                    row = dict(dst_rows_by_key.get(key, {}))
                    for field in dst_fieldnames:
                        row.setdefault(field, "")
                    writer.writerow(row)
        except Exception:
            return 0, 1

        return merged_count, 0

    def _sync_test_scene_subset_to_real(
        self,
        prev_paths: dict[str, str],
        test_root: Path | None,
    ) -> tuple[int, int, int]:
        if test_root is None or not test_root.is_dir():
            return 0, 0, 0

        scene_stems = [str(s).strip() for s in (self._pipeline_test_scene_stems or []) if str(s).strip()]
        if not scene_stems:
            scene_stems = [
                self._pipeline_scene_stem_from_name(x)
                for x in (self._pipeline_test_manifest or [])
                if str(x).strip()
            ]
            scene_stems = [str(s).strip() for s in scene_stems if str(s).strip()]

        copied_total = 0
        error_total = 0
        csv_merged_total = 0

        def _sync_dir(src_dir: Path, dst_var: str, patterns: list[str], *, must_contain: str = "") -> None:
            nonlocal copied_total, error_total
            dst_dir = str(prev_paths.get(dst_var, "")).strip()
            copied, errs = self._pipeline_sync_test_scene_files(
                src_dir,
                dst_dir,
                patterns,
                scene_stems,
                must_contain=must_contain,
            )
            copied_total += copied
            error_total += errs

        video_patterns = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"]
        _sync_dir(test_root / "depthmap", "depth_output_var", video_patterns)
        _sync_dir(test_root / "depthmap" / "upscaled", "depth_upscaled_var", video_patterns)
        _sync_dir(test_root / "mask", "splat_mask_output_var", ["*_replace_mask.*"], must_contain="_replace_mask")
        _sync_dir(
            test_root / "output",
            "inpaint_output_var",
            ["*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"],
            must_contain="_inpainted_",
        )
        _sync_dir(
            test_root / "mask_for_merge",
            "merge_mask_formerge_var",
            ["*_replace_mask.*"],
            must_contain="_replace_mask",
        )
        _sync_dir(test_root / "sbs", "merge_output_var", video_patterns, must_contain="_merged_")

        splat_root = str(prev_paths.get("splat_output_var", "")).strip()
        if splat_root:
            splat_root_path = Path(splat_root).expanduser().resolve()
            splat_hires_dst = splat_root_path if splat_root_path.name.lower() == "hires" else splat_root_path / "hires"
            copied, errs = self._pipeline_sync_test_scene_files(
                test_root / "splat" / "hires",
                str(splat_hires_dst),
                video_patterns,
                scene_stems,
                must_contain="_splatted",
            )
            copied_total += copied
            error_total += errs

        join_src = Path(self.join_output_var.get().strip())
        join_dst = str(prev_paths.get("join_output_var", "")).strip()
        if join_src.is_file() and join_dst:
            if self._copy_or_replace_file(join_src, Path(join_dst).expanduser().resolve()):
                copied_total += 1
            else:
                error_total += 1

        merged_rows, csv_err = self._pipeline_merge_test_csv_rows(
            test_root / "sharpness_test.csv",
            prev_paths.get("inpaint_sharpness_csv_var", ""),
            ("file",),
        )
        csv_merged_total += merged_rows
        error_total += csv_err
        merged_rows, csv_err = self._pipeline_merge_test_csv_rows(
            test_root / "autoct_test.csv",
            prev_paths.get("merge_autoct_csv_var", ""),
            ("video", "frame"),
        )
        csv_merged_total += merged_rows
        error_total += csv_err

        return copied_total, csv_merged_total, error_total

    def _restore_test_scene_subset(self) -> None:
        if not self._pipeline_test_active:
            return
        prev_paths = dict(self._pipeline_test_prev_paths)
        test_root = Path(self._pipeline_test_dir) if self._pipeline_test_dir else None
        copied_count, csv_rows_count, sync_errors = self._sync_test_scene_subset_to_real(
            prev_paths,
            test_root,
        )

        for var_name in self._pipeline_test_path_var_names():
            var_obj = getattr(self, var_name, None)
            if var_obj is None:
                continue
            if var_name in prev_paths:
                var_obj.set(str(prev_paths[var_name]))

        cleaned = False
        if test_root is not None and test_root.exists() and sync_errors == 0:
            try:
                shutil.rmtree(test_root, ignore_errors=True)
                cleaned = True
            except Exception:
                cleaned = False

        self._pipeline_test_active = False
        self._pipeline_test_manifest = []
        self._pipeline_test_scene_stems = []
        self._pipeline_test_source_dir = ""
        self._pipeline_test_dir = ""
        self._pipeline_test_prev_paths = {}
        self._pipeline_test_step_state = self._default_pipeline_step_state()

        self._preview_depth_command()
        self._preview_splat_command()
        self._preview_inpaint_command()
        self._preview_merge_command()
        self._preview_join_command()
        self._refresh_pipeline_status_panel()
        if sync_errors > 0:
            self.pipeline_run_status_var.set(
                (
                    f"Test run synced {copied_count} file(s), {csv_rows_count} CSV row(s), "
                    f"errors={sync_errors}; subset kept."
                )
            )
        else:
            self.pipeline_run_status_var.set(
                (
                    f"Test run synced {copied_count} file(s), {csv_rows_count} CSV row(s); "
                    f"{'subset cleaned' if cleaned else 'subset closed'}."
                )
            )
        self._pipeline_sync_noninteractive_mode()

    def _pipeline_start_resume(self) -> None:
        if self._any_pipeline_activity():
            messagebox.showinfo("Start/Resume", "Another task is currently running.")
            self._pipeline_sync_noninteractive_mode()
            return
        self._pipeline_pause_after_split_scenes = False
        if not self._pipeline_test_active and self._pipeline_split_scenes_gate_pending():
            gate_msg = (
                "Pipeline will pause after Split Scenes verification.\n\n"
                "When Split Scenes verify is done, move clips you do NOT want to convert "
                "into the seg-mono folder.\n\n"
                "Then press Start/Resume again to continue."
            )
            self._append_pipeline_popup_log("INFO", "Start/Resume", gate_msg)
            self._show_pipeline_force_info("Start/Resume", gate_msg)
            self._pipeline_pause_after_split_scenes = True
        self._pipeline_ui_noninteractive = True
        self._append_pipeline_popup_log(
            "INFO",
            "Start/Resume",
            "Non-interactive mode enabled: popups are suppressed and routed to this log.",
        )
        self._pipeline_autorun = True
        self._pipeline_pending_action = None
        self.pipeline_run_status_var.set("Start/Resume running...")
        self._pipeline_trigger_next_action()

    def _pipeline_split_scenes_gate_pending(self) -> bool:
        st = self._pipeline_step_state.get("split_scenes", {"completed": False, "verified": "none"})
        if not bool(st.get("completed", False)):
            return True
        verify_mode = self.pipeline_verify_after_var.get().strip().lower()
        verified = str(st.get("verified", "none")).strip().lower()
        if verify_mode == "quick":
            return verified not in {"quick", "deep"}
        if verify_mode == "deep":
            return verified != "deep"
        return False

    def _show_pipeline_force_info(self, title: str, message: str) -> None:
        fn = self._messagebox_originals.get("showinfo")
        if callable(fn):
            try:
                fn(title, message)
                return
            except Exception:
                pass
        try:
            messagebox.showinfo(title, message)
        except Exception:
            pass

    def _pipeline_trigger_next_action(self) -> None:
        if not self._pipeline_autorun:
            self._pipeline_sync_noninteractive_mode()
            return
        if self._any_pipeline_activity():
            return
        action = self._pipeline_next_action()
        if action is None:
            self._pipeline_autorun = False
            self._pipeline_pending_action = None
            if self._pipeline_test_active:
                self._restore_test_scene_subset()
            self.pipeline_run_status_var.set("Pipeline: all required steps completed")
            self._pipeline_sync_noninteractive_mode()
            return

        step, act, mode = action
        self._pipeline_pending_action = (step, act, mode)
        started = False
        if act == "run":
            started = self._pipeline_dispatch_run(step)
        else:
            started = self._pipeline_dispatch_verify(step, mode)
        if not started:
            self._pipeline_autorun = False
            self._pipeline_pending_action = None
            if self._pipeline_test_active:
                self._restore_test_scene_subset()
            self.pipeline_run_status_var.set(f"Pipeline stopped: could not start {step} {act}")
            self._pipeline_sync_noninteractive_mode()
            return

    def _pipeline_next_action(self) -> tuple[str, str, str] | None:
        verify_mode = self.pipeline_verify_after_var.get().strip().lower()
        if self._pipeline_test_active:
            self._pipeline_recompute_test_step_state()
        step_state = (
            self._pipeline_test_step_state
            if self._pipeline_test_active
            else self._pipeline_step_state
        )
        for step, _label in self.PIPELINE_STEPS:
            if not self._is_pipeline_step_required(step):
                continue
            st = step_state.get(step, {"completed": False, "verified": "none"})
            if not bool(st.get("completed", False)):
                return step, "run", "none"
            if self._pipeline_test_active and step in {"scenedetect", "split_scenes"}:
                continue
            if step in self.PIPELINE_STEPS_WITH_VERIFY and verify_mode in {"quick", "deep"}:
                current = str(st.get("verified", "none"))
                if verify_mode == "quick":
                    if current not in {"quick", "deep"}:
                        return step, "verify", "quick"
                else:
                    if current != "deep":
                        return step, "verify", "deep"
        return None

    def _pipeline_dispatch_run(self, step: str) -> bool:
        self.pipeline_run_status_var.set(f"Running step: {step}")
        if step == "scenedetect":
            before = bool(self._scene_thread and self._scene_thread.is_alive())
            self._start_scene_detect()
            return bool(self._scene_thread and self._scene_thread.is_alive()) and not before
        if step == "split_scenes":
            before = bool(self._scene_thread and self._scene_thread.is_alive())
            self._start_split_scenes()
            return bool(self._scene_thread and self._scene_thread.is_alive()) and not before
        if step == "depthcrafter":
            before = bool(self._depth_thread and self._depth_thread.is_alive())
            self._run_depth_placeholder()
            return bool(self._depth_thread and self._depth_thread.is_alive()) and not before
        if step == "depth_upscale":
            before = bool(self._depth_thread and self._depth_thread.is_alive())
            self._run_depth_upscale_placeholder()
            return bool(self._depth_thread and self._depth_thread.is_alive()) and not before
        if step == "splatting":
            before = bool(self._splat_thread and self._splat_thread.is_alive())
            self._run_splat_placeholder()
            return bool(self._splat_thread and self._splat_thread.is_alive()) and not before
        if step == "sharpness_csv":
            before = bool(self._inpaint_thread and self._inpaint_thread.is_alive())
            self._start_inpaint_sharpness_csv()
            return bool(self._inpaint_thread and self._inpaint_thread.is_alive()) and not before
        if step == "inpaint":
            before = bool(self._inpaint_thread and self._inpaint_thread.is_alive())
            self._run_inpaint_placeholder()
            return bool(self._inpaint_thread and self._inpaint_thread.is_alive()) and not before
        if step == "autoct_csv":
            before = bool(self._merge_thread and self._merge_thread.is_alive())
            self._start_merge_autoct_csv()
            return bool(self._merge_thread and self._merge_thread.is_alive()) and not before
        if step == "mask_for_merge":
            before = bool(self._merge_thread and self._merge_thread.is_alive())
            self._run_merge_mask_placeholder()
            return bool(self._merge_thread and self._merge_thread.is_alive()) and not before
        if step == "merging":
            before = bool(self._merge_thread and self._merge_thread.is_alive())
            self._run_merge_placeholder()
            return bool(self._merge_thread and self._merge_thread.is_alive()) and not before
        if step == "mono_to_sbs":
            before = bool(self._join_thread and self._join_thread.is_alive())
            self._run_join_prepare_mono()
            return bool(self._join_thread and self._join_thread.is_alive()) and not before
        if step == "join":
            before = bool(self._join_thread and self._join_thread.is_alive())
            self._run_join_scenes()
            return bool(self._join_thread and self._join_thread.is_alive()) and not before
        if step == "remux":
            before = bool(self._join_thread and self._join_thread.is_alive())
            self._start_join_remux()
            return bool(self._join_thread and self._join_thread.is_alive()) and not before
        return False

    def _pipeline_dispatch_verify(self, step: str, mode: str) -> bool:
        self.pipeline_run_status_var.set(f"Verifying {step} ({mode})")
        before = self._verify_running
        if step == "split_scenes":
            self._start_verify_deep() if mode == "deep" else self._start_verify_quick()
        elif step == "depthcrafter":
            self._start_depth_verify_deep() if mode == "deep" else self._start_depth_verify_quick()
        elif step == "depth_upscale":
            self._start_depth_upscaled_verify_deep() if mode == "deep" else self._start_depth_upscaled_verify_quick()
        elif step == "splatting":
            self._start_splat_verify_deep() if mode == "deep" else self._start_splat_verify_quick()
        elif step == "inpaint":
            self._start_inpaint_verify_deep() if mode == "deep" else self._start_inpaint_verify_quick()
        elif step == "mask_for_merge":
            self._start_merge_mask_verify_deep() if mode == "deep" else self._start_merge_mask_verify_quick()
        elif step == "merging":
            self._start_merge_verify_deep() if mode == "deep" else self._start_merge_verify_quick()
        elif step == "mono_to_sbs":
            self._start_join_mono_verify()
        elif step == "join":
            # Join has only one verify mode; use it for quick/deep policy.
            self._start_join_verify()
        else:
            return False
        return self._verify_running and not before

    def _any_pipeline_activity(self) -> bool:
        return any(
            [
                bool(self._scene_thread and self._scene_thread.is_alive()),
                bool(self._analysis_thread and self._analysis_thread.is_alive()),
                bool(self._depth_thread and self._depth_thread.is_alive()),
                bool(self._splat_thread and self._splat_thread.is_alive()),
                bool(self._inpaint_thread and self._inpaint_thread.is_alive()),
                bool(self._merge_thread and self._merge_thread.is_alive()),
                bool(self._join_thread and self._join_thread.is_alive()),
                bool(self._verify_running),
            ]
        )

    def _pipeline_on_run_finished(self, step: str, success: bool) -> None:
        pending = self._pipeline_pending_action
        state = (
            self._pipeline_test_step_state
            if self._pipeline_test_active
            else self._pipeline_step_state
        )
        if success:
            self._pipeline_set_completed_in_state(state, step, True)
            self._pipeline_set_verified_in_state(state, step, "none")
            if step in self.PIPELINE_CSV_STEPS:
                self._sync_pipeline_csv_done_flags_in_state(state)
            self._refresh_pipeline_status_panel()
            if not self._pipeline_test_active:
                self._save_pipeline_state()
        if pending and pending[0] == step and pending[1] == "run":
            self._pipeline_pending_action = None
            if not success:
                self._pipeline_autorun = False
                if self._pipeline_test_active:
                    self._restore_test_scene_subset()
                self.pipeline_run_status_var.set(f"Pipeline stopped: step failed ({step})")
                self._pipeline_sync_noninteractive_mode()
                return
            if step == "split_scenes" and self._pipeline_pause_after_split_scenes and not self._pipeline_test_active:
                verify_mode = self.pipeline_verify_after_var.get().strip().lower()
                if verify_mode not in {"quick", "deep"}:
                    self._pipeline_pause_after_split_scenes = False
                    self._pipeline_autorun = False
                    pause_msg = (
                        "Split Scenes completed.\n\n"
                        "Please move clips you do NOT want to convert into seg-mono,\n"
                        "then press Start/Resume again to continue."
                    )
                    self.pipeline_run_status_var.set(
                        "Paused after Split Scenes. Move files to seg-mono, then Start/Resume."
                    )
                    self._append_pipeline_popup_log("INFO", "Split Scenes Pause", pause_msg)
                    self._show_pipeline_force_info("Split Scenes Pause", pause_msg)
                    self._pipeline_sync_noninteractive_mode()
                    return
            self._pipeline_trigger_next_action()

    def _pipeline_on_verify_finished(self, step: str, success: bool, mode: str) -> None:
        pending = self._pipeline_pending_action
        state = (
            self._pipeline_test_step_state
            if self._pipeline_test_active
            else self._pipeline_step_state
        )
        if success:
            self._pipeline_mark_previous_steps_done_verified_in_state(state, step, mode)
            self._sync_pipeline_csv_done_flags_in_state(state)
            self._refresh_pipeline_status_panel()
            if not self._pipeline_test_active:
                self._save_pipeline_state()
        else:
            self._pipeline_invalidate_active_from(step)
        if pending and pending[0] == step and pending[1] == "verify":
            self._pipeline_pending_action = None
            if not success:
                if self._pipeline_autorun:
                    self.pipeline_run_status_var.set(
                        f"Verify failed on {step}: re-running previous step output."
                    )
                    self._pipeline_trigger_next_action()
                    return
                self._pipeline_autorun = False
                if self._pipeline_test_active:
                    self._restore_test_scene_subset()
                self.pipeline_run_status_var.set(f"Pipeline stopped: verify failed ({step})")
                self._pipeline_sync_noninteractive_mode()
                return
            if step == "split_scenes" and self._pipeline_pause_after_split_scenes and not self._pipeline_test_active:
                self._pipeline_pause_after_split_scenes = False
                self._pipeline_autorun = False
                pause_msg = (
                    "Split Scenes verify completed.\n\n"
                    "Please move clips you do NOT want to convert into seg-mono,\n"
                    "then press Start/Resume again to continue."
                )
                self.pipeline_run_status_var.set(
                    "Paused after Split Scenes verify. Move files to seg-mono, then Start/Resume."
                )
                self._append_pipeline_popup_log("INFO", "Split Scenes Verify Pause", pause_msg)
                self._show_pipeline_force_info("Split Scenes Verify Pause", pause_msg)
                self._pipeline_sync_noninteractive_mode()
                return
            self._pipeline_trigger_next_action()
        elif not success:
            self.pipeline_run_status_var.set(
                f"Verify failed on {step}: cleared this step and downstream flags."
            )

    def _open_depth_input_folder(self) -> None:
        folder = self.depth_input_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_depth_log(f"Depth input folder ready: {folder}")

    def _open_depth_output_folder(self) -> None:
        folder = self.depth_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_depth_log(f"Depth output folder ready: {folder}")

    def _open_depth_upscaled_folder(self) -> None:
        folder = self.depth_upscaled_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_depth_log(f"Depth upscaled folder ready: {folder}")

    def _on_depth_mode_changed(self, _event=None) -> None:
        mode = self.depth_mode_var.get().strip()
        if mode == "Manual":
            self.depth_info_text_var.set(self.DEPTH_MANUAL_INFO)
            state_auto = tk.NORMAL
            state_res = tk.NORMAL
        else:
            self.depth_mode_var.set("Auto (recommended)")
            self.depth_info_text_var.set(self.DEPTH_AUTO_INFO)
            self._reset_depth_auto_locked_defaults()
            state_auto = tk.NORMAL
            state_res = tk.DISABLED

        self.depth_chunk_size_entry.configure(state=state_auto)
        self.depth_overlap_entry.configure(state=state_auto)
        self.depth_inference_steps_entry.configure(state=state_auto)
        self.depth_seed_entry.configure(state=state_auto)
        self.depth_scale_factor_scale.configure(state=state_auto)
        self.depth_cpu_offload_combo.configure(state="readonly" if state_auto == tk.NORMAL else tk.DISABLED)
        self.depth_res_x_entry.configure(state=state_res)
        self.depth_res_y_entry.configure(state=state_res)
        self._update_depth_resolution_preview()
        self._preview_depth_command()

    def _reset_depth_auto_locked_defaults(self) -> None:
        # Fields disabled in Auto mode are informational and update dynamically.
        self._update_depth_resolution_preview()

    def _normalize_depth_scale_factor(self, value) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(self.DEFAULT_DEPTH_SCALE_FACTOR)
        parsed = max(float(self.MIN_DEPTH_SCALE_FACTOR), min(float(self.MAX_DEPTH_SCALE_FACTOR), parsed))
        return round(parsed, 2)

    @staticmethod
    def _round_up_multiple(n: int, m: int) -> int:
        return ((int(n) + int(m) - 1) // int(m)) * int(m)

    @staticmethod
    def _ensure_even_min2(n: int) -> int:
        v = max(2, int(n))
        if v % 2:
            v -= 1
        return max(2, v)

    def _depth_scene_strip_pad(self) -> tuple[int, int]:
        strip_top = 0
        strip_bottom = 0
        try:
            rec = self._recommended_crop_filters.get("auto") or {}
            strip_top = max(0, int(rec.get("pad_top_src") or 0))
            strip_bottom = max(0, int(rec.get("pad_bottom_src") or 0))
        except Exception:
            strip_top = 0
            strip_bottom = 0
        return strip_top, strip_bottom

    def _get_depth_scale_factor(self) -> float:
        factor = self._normalize_depth_scale_factor(self.depth_scale_factor_var.get())
        try:
            current = float(self.depth_scale_factor_var.get())
        except Exception:
            current = factor
        if abs(current - factor) > 1e-9:
            self.depth_scale_factor_var.set(factor)
        self.depth_scale_factor_text_var.set(f"{factor:.2f}x")
        return factor

    def _compute_depth_working_sizes(
        self,
    ) -> tuple[int, int, int, int, int, int, int, int, int] | None:
        src_w, src_h = self._get_depth_input_reference_resolution()
        if not isinstance(src_w, int) or not isinstance(src_h, int):
            return None
        if src_w <= 0 or src_h <= 0:
            return None

        strip_top, strip_bottom = self._depth_scene_strip_pad()
        strip_total = max(0, strip_top + strip_bottom)
        max_strip = max(0, int(src_h) - 2)
        if strip_total > max_strip:
            if strip_total > 0:
                ratio_top = float(strip_top) / float(strip_total)
            else:
                ratio_top = 0.5
            strip_top = int(round(float(max_strip) * ratio_top))
            strip_top = max(0, min(max_strip, strip_top))
            strip_bottom = max_strip - strip_top
            strip_total = strip_top + strip_bottom

        core_h = int(src_h) - strip_total
        if core_h < 2:
            strip_top = 0
            strip_bottom = 0
            core_h = int(src_h)

        factor = self._get_depth_scale_factor()
        content_w = self._ensure_even_min2(int(int(src_w) * factor))
        content_h = self._ensure_even_min2(int(int(core_h) * factor))
        pad_w = self._round_up_multiple(content_w, 64)
        pad_h = self._round_up_multiple(content_h, 64)

        return (
            content_w,
            content_h,
            pad_w,
            pad_h,
            int(src_w),
            int(src_h),
            strip_top,
            strip_bottom,
        )

    def _update_depth_resolution_preview(self) -> None:
        sizes = self._compute_depth_working_sizes()
        if sizes is None:
            self.depth_pad_target_var.set("n.d.")
            if self.depth_mode_var.get().strip() != "Manual":
                self.depth_res_x_var.set("")
                self.depth_res_y_var.set("")
            return

        content_w, content_h, pad_w, pad_h, _src_w, _src_h, _strip_top, _strip_bottom = sizes
        self.depth_res_x_var.set(str(content_w))
        self.depth_res_y_var.set(str(content_h))
        self.depth_pad_target_var.set(f"{pad_w}x{pad_h} (bottom)")

    def _on_depth_scale_slider_moved(self, _value=None) -> None:
        factor = self._normalize_depth_scale_factor(self.depth_scale_factor_var.get())
        try:
            current = float(self.depth_scale_factor_var.get())
        except Exception:
            current = factor
        if abs(current - factor) > 1e-9:
            self.depth_scale_factor_var.set(factor)

    def _on_depth_scale_factor_changed(self, *_args) -> None:
        self._get_depth_scale_factor()
        self._update_depth_resolution_preview()
        self._preview_depth_command()

    def _normalize_ffmpeg_codec(self, value: str, fallback: str = "") -> str:
        raw = str(value or "").strip().lower()
        raw = self.FFMPEG_CODEC_ALIASES.get(raw, raw)
        if raw in self.FFMPEG_CODEC_CHOICES:
            return raw
        fb = str(fallback or "").strip().lower()
        fb = self.FFMPEG_CODEC_ALIASES.get(fb, fb)
        if fb in self.FFMPEG_CODEC_CHOICES:
            return fb
        return self.DEFAULT_SCENE_CODEC

    def _set_codec_widget_override_state(self, widget, enabled: bool) -> None:
        widget.configure(state="readonly" if enabled else tk.DISABLED)

    def _on_scene_encode_var_changed(self, *_args) -> None:
        if not self.depth_encode_override_var.get():
            self._sync_depth_encoding_from_scene()
            self._preview_depth_command()
        if not self.splat_encode_override_var.get():
            self._sync_splat_encoding_from_scene()
            self._preview_splat_command()
        if not self.inpaint_encode_override_var.get():
            self._sync_inpaint_encoding_from_scene()
            self._preview_inpaint_command()
        if not self.merge_encode_override_var.get():
            self._sync_merge_encoding_from_scene()
            self._preview_merge_command()
        if not self.join_pix_fmt_override_var.get():
            self._sync_join_encoding_from_scene()
            self._preview_join_command()

    def _sync_depth_encoding_from_scene(self) -> None:
        self.depth_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.scene_codec_var.get(),
                self.DEFAULT_SCENE_CODEC,
            )
        )
        self.depth_crf_var.set(self.scene_crf_var.get().strip())
        self.depth_preset_var.set(self.scene_encoder_preset_var.get().strip())
        self.depth_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip())

    def _on_depth_override_toggle(self, initial: bool = False) -> None:
        enabled = bool(self.depth_encode_override_var.get())
        if not enabled:
            self._sync_depth_encoding_from_scene()
        elif not initial and not self._depth_override_notice_shown:
            self._depth_override_notice_shown = True
            messagebox.showwarning("Depth Encode Override", self.DEPTH_OVERRIDE_WARNING)

        state = tk.NORMAL if enabled else tk.DISABLED
        self._set_codec_widget_override_state(self.depth_codec_entry, enabled)
        for widget in (
            self.depth_crf_entry,
            self.depth_preset_entry,
            self.depth_pixfmt_entry,
            self.depth_extra_ffmpeg_entry,
        ):
            widget.configure(state=state)
        self._preview_depth_command()

    def _build_depth_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        final_upscale = "False"
        depth_codec = self._normalize_ffmpeg_codec(
            self.depth_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        self.depth_codec_var.set(depth_codec)
        scene_strip_pad_top, scene_strip_pad_bottom = self._depth_scene_strip_pad()
        scale_factor = self._get_depth_scale_factor()

        env_updates: dict[str, str] = {
            "PYTHON": sys.executable,
            "WORKER_SCRIPT": self.depth_worker_script_var.get().strip() or "./depthcrafter_nogui_batch.py",
            "INPUT_DIR": self.depth_input_var.get().strip(),
            "OUTPUT_DIR": self.depth_output_var.get().strip(),
            "GLOB": self.depth_glob_var.get().strip() or "*.mp4",
            "WINDOW_SIZE": self.depth_chunk_size_var.get().strip(),
            "OVERLAP": self.depth_overlap_var.get().strip(),
            "INFERENCE_STEPS": self.depth_inference_steps_var.get().strip(),
            "GUIDANCE_SCALE": self.depth_guidance_scale_var.get().strip() or "1.0",
            "SEED": self.depth_seed_var.get().strip(),
            "CPU_OFFLOAD_MODE": self.depth_cpu_offload_var.get().strip(),
            "DECODE_CHUNK_SIZE": self.depth_decode_chunk_size_var.get().strip() or "2",
            "DEBUG_MEM": "True" if self.depth_debug_mem_var.get() else "False",
            "FINAL_UPSCALE": final_upscale,
            "SCALE_FACTOR": f"{scale_factor:.2f}",
            "RESTART_EVERY": self.depth_restart_every_var.get().strip() or "100",
            "PAD_ALIGN_BOTTOM": "True",
            "USE_REALESRGAN_UPSCALE": "False",
            "SCENE_STRIP_PAD_TOP": str(scene_strip_pad_top),
            "SCENE_STRIP_PAD_BOTTOM": str(scene_strip_pad_bottom),
            "FFMPEG_CODEC": depth_codec,
            "FFMPEG_CRF": self.depth_crf_var.get().strip() or "1",
            "FFMPEG_PRESET": self.depth_preset_var.get().strip(),
            "FFMPEG_PIX_FMT": self.depth_pix_fmt_var.get().strip(),
            "FFMPEG_EXTRA_ARGS": self.depth_extra_ffmpeg_args_var.get().strip(),
            "RETRY_POLICY_JSON": self._build_retry_policy_json(
                self.depth_retry_policy_vars,
                self.depth_cpu_offload_var.get().strip() or "model",
            ),
        }

        cmd = ["bash", "run_depthcrafter_nogui_batch.sh"]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview

    def _preview_depth_command(self) -> None:
        try:
            _cmd, _env, preview = self._build_depth_runner_payload()
            self.depth_cmd_preview_var.set(preview)
        except Exception as e:
            self.depth_cmd_preview_var.set(f"Invalid options: {e}")

    def _append_depth_log(self, line: str) -> None:
        self.depth_log_text.configure(state=tk.NORMAL)
        self.depth_log_text.insert(tk.END, line + "\n")
        self.depth_log_text.see(tk.END)
        self.depth_log_text.configure(state=tk.DISABLED)

    def _clear_depth_log(self) -> None:
        self.depth_log_text.configure(state=tk.NORMAL)
        self.depth_log_text.delete("1.0", tk.END)
        self.depth_log_text.configure(state=tk.DISABLED)

    def _run_depth_placeholder(self) -> None:
        if self._depth_thread and self._depth_thread.is_alive():
            messagebox.showinfo("DepthCrafter", "DepthCrafter is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("DepthCrafter", "Stop verification before starting DepthCrafter.")
            return
        try:
            cmd, env_updates, _preview = self._build_depth_runner_payload()
        except Exception as exc:
            messagebox.showerror("DepthCrafter", f"Invalid depth options:\n{exc}")
            return

        launcher_script = Path("run_depthcrafter_nogui_batch.sh").resolve()
        if not launcher_script.is_file():
            messagebox.showerror("DepthCrafter", f"Launcher not found:\n{launcher_script}")
            return

        worker_script = Path(self.depth_worker_script_var.get().strip() or "./depthcrafter_nogui_batch.py")
        worker_abs = worker_script.resolve()
        if not worker_abs.is_file():
            messagebox.showerror("DepthCrafter", f"Worker script not found:\n{worker_abs}")
            return

        input_dir = self.depth_input_var.get().strip()
        output_dir = self.depth_output_var.get().strip()
        if not input_dir:
            messagebox.showerror("DepthCrafter", "Depth input folder is required.")
            return
        if not os.path.isdir(input_dir):
            messagebox.showerror("DepthCrafter", f"Depth input folder not found:\n{input_dir}")
            return
        if not output_dir:
            messagebox.showerror("DepthCrafter", "Depth output folder is required.")
            return
        os.makedirs(output_dir, exist_ok=True)

        if env_updates.get("USE_REALESRGAN_UPSCALE", "False").lower() == "true":
            upscale_script = Path(env_updates.get("REALESRGAN_UPSCALE_SCRIPT", "")).resolve()
            if not upscale_script.is_file():
                messagebox.showerror("DepthCrafter", f"RealESRGAN script not found:\n{upscale_script}")
                return

            runtime_mode = env_updates.get("REALESRGAN_RUNTIME", "local").strip().lower()
            runtime_bin = env_updates.get("REALESRGAN_BIN", "").strip()
            if runtime_mode == "bundled":
                if not runtime_bin or not Path(runtime_bin).is_file():
                    messagebox.showerror(
                        "DepthCrafter",
                        f"Bundled RealESRGAN binary not found:\n{runtime_bin or '(empty)'}",
                    )
                    return
                model_dir = env_updates.get("REALESRGAN_MODEL_DIR", "").strip()
                if not model_dir or not Path(model_dir).is_dir():
                    messagebox.showerror(
                        "DepthCrafter",
                        f"Bundled RealESRGAN model dir not found:\n{model_dir or '(empty)'}",
                    )
                    return
            else:
                if runtime_bin:
                    if not Path(runtime_bin).is_file():
                        messagebox.showerror(
                            "DepthCrafter",
                            f"Local RealESRGAN binary not found:\n{runtime_bin}",
                        )
                        return
                else:
                    resolved_local = self._resolve_local_realesrgan_bin()
                    if not resolved_local:
                        messagebox.showerror(
                            "DepthCrafter",
                            (
                                "Local RealESRGAN runtime not found.\n\n"
                                "Searched PATH and current venv bin.\n"
                                "Install `realesrgan-ncnn-vulkan` or switch to Bundled runtime."
                            ),
                        )
                        return
                    env_updates["REALESRGAN_BIN"] = resolved_local
                    local_models = self._resolve_realesrgan_model_dir(resolved_local)
                    if local_models:
                        env_updates["REALESRGAN_MODEL_DIR"] = local_models

        self._depth_stop_requested = False
        self._depth_stop_clicks = 0
        self.depth_status_var.set("Starting...")
        self.depth_progress_var.set(0.0)
        self._set_depth_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("depthcrafter")
        self._append_depth_log("=== DepthCrafter started ===")
        self._append_depth_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_depth_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._depth_thread = threading.Thread(
            target=self._run_depth_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._depth_thread.start()

    def _run_depth_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        step_success = False
        try:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._depth_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("depth_line", line))
                    self._try_parse_depth_progress(line)
                if self._depth_stop_requested:
                    break
            rc = proc.wait()
            if self._depth_stop_requested:
                self._log_queue.put(("depth_status", "Stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("depth_status", "Completed"))
                self._log_queue.put(("depth_progress", "100"))
            else:
                self._log_queue.put(("depth_status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("depth_line", f"[DEPTH][ERROR] {exc}"))
            self._log_queue.put(("depth_status", "Failed"))
        finally:
            self._depth_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("depth_done", {"step": "depthcrafter", "success": step_success}))

    def _resolve_local_realesrgan_bin(self) -> str | None:
        candidates: list[str] = []
        env_bin = str(os.environ.get("REALESRGAN_BIN", "")).strip()
        if env_bin:
            candidates.append(env_bin)
        which_bin = shutil.which("realesrgan-ncnn-vulkan")
        if which_bin:
            candidates.append(which_bin)
        try:
            candidates.append(str((Path(sys.executable).resolve().parent / "realesrgan-ncnn-vulkan")))
        except Exception:
            pass

        seen: set[str] = set()
        for raw in candidates:
            if not raw:
                continue
            if raw in seen:
                continue
            seen.add(raw)
            p = Path(raw).expanduser()
            try:
                if p.is_file():
                    return str(p.resolve())
            except Exception:
                continue
        return None

    @staticmethod
    def _resolve_realesrgan_model_dir(bin_path: str) -> str:
        if not bin_path:
            return ""
        p = Path(bin_path).expanduser()
        try:
            model_dir = p.resolve().parent / "models"
        except Exception:
            return ""
        return str(model_dir) if model_dir.is_dir() else ""

    @staticmethod
    def _probe_video_resolution_fast(path: str) -> tuple[int | None, int | None]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20.0)
        except Exception:
            return None, None
        if cp.returncode != 0:
            return None, None
        try:
            doc = json.loads(cp.stdout or "{}")
            streams = doc.get("streams") or []
            if not streams:
                return None, None
            st = streams[0] or {}
            w = int(st.get("width")) if st.get("width") not in (None, "", "N/A") else None
            h = int(st.get("height")) if st.get("height") not in (None, "", "N/A") else None
            return w, h
        except Exception:
            return None, None

    def _get_depth_input_reference_resolution(self) -> tuple[int | None, int | None]:
        input_dir = self.depth_input_var.get().strip()
        if input_dir and os.path.isdir(input_dir):
            root = Path(input_dir)
            sample: Path | None = None
            patterns = self.VERIFY_VIDEO_PATTERNS or ["*.mp4"]
            for patt in patterns:
                for p in sorted(root.glob(patt)):
                    if p.is_file():
                        sample = p
                        break
                if sample is not None:
                    break

            if sample is not None:
                try:
                    cache_key = (
                        str(root.resolve()),
                        str(sample.resolve()),
                        int(sample.stat().st_mtime_ns),
                    )
                except Exception:
                    cache_key = (str(root), str(sample), 0)

                if cache_key == self._depth_input_resolution_cache_key:
                    w_cached, h_cached = self._depth_input_resolution_cache_value
                    if (
                        isinstance(w_cached, int)
                        and isinstance(h_cached, int)
                        and w_cached > 0
                        and h_cached > 0
                    ):
                        return w_cached, h_cached

                w_probe, h_probe = self._probe_video_resolution_fast(str(sample))
                self._depth_input_resolution_cache_key = cache_key
                self._depth_input_resolution_cache_value = (w_probe, h_probe)
                if (
                    isinstance(w_probe, int)
                    and isinstance(h_probe, int)
                    and w_probe > 0
                    and h_probe > 0
                ):
                    return w_probe, h_probe

        # Fallback for pre-split/pre-analysis states.
        src_w = self._source_video_info.get("width")
        src_h = self._source_video_info.get("height")
        if (
            isinstance(src_w, int)
            and isinstance(src_h, int)
            and src_w > 0
            and src_h > 0
        ):
            if self._needs_downscale_to_1080():
                scale = 1920.0 / float(src_w)
                est_w = self._ensure_even_min2(1920)
                est_h = self._ensure_even_min2(int(round(float(src_h) * scale)))
                return est_w, est_h
            return int(src_w), int(src_h)
        return None, None

    def _resolve_depth_upscale_dest(self, scale_factor: float) -> str:
        if abs(float(scale_factor) - 0.5) <= 1e-9:
            return ""
        ref_dir = self.depth_input_var.get().strip()
        if ref_dir and os.path.isdir(ref_dir):
            for p in sorted(Path(ref_dir).glob("*.mp4")):
                if not p.is_file():
                    continue
                w, h = self._probe_video_resolution_fast(str(p))
                if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                    return f"{w}x{h}"
        src_w = self._source_video_info.get("width")
        src_h = self._source_video_info.get("height")
        if (
            isinstance(src_w, int)
            and isinstance(src_h, int)
            and src_w > 0
            and src_h > 0
            and not self._needs_downscale_to_1080()
        ):
            return f"{src_w}x{src_h}"
        return ""

    def _build_depth_upscale_payload(self) -> tuple[list[str], dict[str, str], str]:
        depth_codec = (self.depth_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC).lower()
        uses_nvenc = "nvenc" in depth_codec
        script = "Utilities/upscale_esrgan_nvenc.sh" if uses_nvenc else "Utilities/upscale_esrgan_x264.sh"
        env_updates: dict[str, str] = {"REALESRGAN_RUNTIME": "local"}
        if self.depth_realesrgan_source_var.get().strip().startswith("Bundled"):
            env_updates["REALESRGAN_RUNTIME"] = "bundled"
            env_updates["REALESRGAN_BIN"] = str(
                Path("Utilities/realesrgan/realesrgan-ncnn-vulkan").resolve()
            )
            env_updates["REALESRGAN_MODEL_DIR"] = str(
                Path("Utilities/realesrgan/models").resolve()
            )
        else:
            local_bin = self._resolve_local_realesrgan_bin()
            if local_bin:
                env_updates["REALESRGAN_BIN"] = local_bin
                local_models = self._resolve_realesrgan_model_dir(local_bin)
                if local_models:
                    env_updates["REALESRGAN_MODEL_DIR"] = local_models

        in_dir = self.depth_output_var.get().strip()
        out_dir = self.depth_upscaled_var.get().strip()
        scale = "2"
        model = "realesr-animevideov3-x2"
        tile = "auto"
        scale_factor = self._get_depth_scale_factor()
        dest = self._resolve_depth_upscale_dest(scale_factor)
        try:
            jobs_num = max(
                1,
                int(
                    self.depth_realesrgan_workers_var.get().strip()
                    or str(self.DEFAULT_DEPTH_REALESRGAN_WORKERS)
                ),
            )
        except Exception:
            jobs_num = self.DEFAULT_DEPTH_REALESRGAN_WORKERS
        if self.depth_realesrgan_workers_var.get().strip() != str(jobs_num):
            self.depth_realesrgan_workers_var.set(str(jobs_num))
        jobs = str(jobs_num)
        retries = "3"

        cmd = [
            "bash",
            script,
            in_dir,
            out_dir,
            scale,
            model,
            tile,
            dest,
            jobs,
            retries,
        ]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview

    def _run_depth_upscale_placeholder(self) -> None:
        if self._depth_thread and self._depth_thread.is_alive():
            messagebox.showinfo("Depth Upscale", "Another depth task is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Depth Upscale", "Stop verification before running ESRGAN.")
            return
        try:
            cmd, env_updates, _preview = self._build_depth_upscale_payload()
        except Exception as exc:
            messagebox.showerror("Depth Upscale", f"Invalid ESRGAN options:\n{exc}")
            return

        script_path = Path(cmd[1]).resolve() if len(cmd) > 1 else Path("").resolve()
        if not script_path.is_file():
            messagebox.showerror("Depth Upscale", f"RealESRGAN script not found:\n{script_path}")
            return

        input_dir = self.depth_output_var.get().strip()
        output_dir = self.depth_upscaled_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Depth Upscale", f"Depth output folder not found:\n{input_dir or '(empty)'}")
            return
        if not output_dir:
            messagebox.showerror("Depth Upscale", "Upscaled output folder is required.")
            return
        os.makedirs(output_dir, exist_ok=True)

        runtime_mode = env_updates.get("REALESRGAN_RUNTIME", "local").strip().lower()
        runtime_bin = env_updates.get("REALESRGAN_BIN", "").strip()
        if runtime_mode == "bundled":
            if not runtime_bin or not Path(runtime_bin).is_file():
                messagebox.showerror(
                    "Depth Upscale",
                    f"Bundled RealESRGAN binary not found:\n{runtime_bin or '(empty)'}",
                )
                return
            model_dir = env_updates.get("REALESRGAN_MODEL_DIR", "").strip()
            if not model_dir or not Path(model_dir).is_dir():
                messagebox.showerror(
                    "Depth Upscale",
                    f"Bundled RealESRGAN model dir not found:\n{model_dir or '(empty)'}",
                )
                return
        else:
            if runtime_bin:
                if not Path(runtime_bin).is_file():
                    messagebox.showerror(
                        "Depth Upscale",
                        f"Local RealESRGAN binary not found:\n{runtime_bin}",
                    )
                    return
            else:
                resolved_local = self._resolve_local_realesrgan_bin()
                if not resolved_local:
                    messagebox.showerror(
                        "Depth Upscale",
                        (
                            "Local RealESRGAN runtime not found.\n\n"
                            "Searched PATH and current venv bin.\n"
                            "Install `realesrgan-ncnn-vulkan` or switch to Bundled runtime."
                        ),
                    )
                    return
                env_updates["REALESRGAN_BIN"] = resolved_local
                local_models = self._resolve_realesrgan_model_dir(resolved_local)
                if local_models:
                    env_updates["REALESRGAN_MODEL_DIR"] = local_models

        self._depth_stop_requested = False
        self._depth_stop_clicks = 0
        self.depth_status_var.set("Running ESRGAN upscale...")
        self.depth_progress_var.set(0.0)
        self._set_depth_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("depth_upscale")
        self._append_depth_log("=== ESRGAN upscale started ===")
        self._append_depth_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        if env_updates:
            self._append_depth_log(
                "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
            )
        self._depth_thread = threading.Thread(
            target=self._run_depth_upscale_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._depth_thread.start()

    def _run_depth_upscale_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        step_success = False
        try:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._depth_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("depth_line", f"[ESRGAN] {line}"))
                    if line.startswith("[DONE]"):
                        self._log_queue.put(("depth_progress", "100"))
                if self._depth_stop_requested:
                    break
            rc = proc.wait()
            if self._depth_stop_requested:
                self._log_queue.put(("depth_status", "ESRGAN stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("depth_status", "Upscale completed"))
                self._log_queue.put(("depth_progress", "100"))
            else:
                self._log_queue.put(("depth_status", f"Upscale failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("depth_line", f"[ESRGAN][ERROR] {exc}"))
            self._log_queue.put(("depth_status", "Upscale failed"))
        finally:
            self._depth_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("depth_done", {"step": "depth_upscale", "success": step_success}))

    def _stop_depth_placeholder(self, prompt_user: bool = True) -> None:
        running = bool(self._depth_thread and self._depth_thread.is_alive())
        if not running:
            return
        if self._depth_stop_clicks == 0 and prompt_user:
            messagebox.showwarning(
                "Stop DepthCrafter",
                "Graceful stop requested.\n\n"
                "Current process will be interrupted like Ctrl+C.\n"
                "Click Stop again to force kill immediately.",
            )
        self._depth_stop_requested = True
        self._depth_stop_clicks += 1

        if self._depth_stop_clicks == 1:
            self.depth_status_var.set("Graceful stop requested...")
            self._append_depth_log(
                "[STOP] graceful stop requested (click Stop again for immediate force stop)."
            )
            self.depth_stop_btn.configure(text="Force Stop")
        else:
            self.depth_status_var.set("Force stop requested...")
            self._append_depth_log("[STOP] force stop requested.")

        self._send_depth_signal(signal.SIGINT)
        if self._depth_stop_clicks >= 2:
            self.root.after(1000, self._force_kill_depth)

    def _send_depth_signal(self, sig: int) -> None:
        proc = self._depth_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception as exc:
            self._append_depth_log(f"Signal send failed: {exc}")

    def _force_kill_depth(self) -> None:
        proc = self._depth_process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                self._append_depth_log("Depth process force-killed.")
            except Exception as exc:
                self._append_depth_log(f"Depth kill failed: {exc}")
        self._cleanup_depth_residual_outputs()

    def _cleanup_depth_residual_outputs(self) -> None:
        out_dir = self.depth_output_var.get().strip()
        if not out_dir:
            return
        out_path = Path(out_dir)
        if not out_path.is_dir():
            return
        try:
            latest = sorted(
                [p for p in out_path.glob("*.mp4") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            latest = []
        if latest:
            candidate = latest[0]
            try:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,width,height,avg_frame_rate,nb_frames",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=nw=1:nk=1",
                        str(candidate),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                bad = probe.returncode != 0
            except Exception:
                bad = True
            if bad:
                try:
                    candidate.unlink()
                    self._append_depth_log(f"[CLEANUP] removed unreadable output: {candidate}")
                except Exception as exc:
                    self._append_depth_log(f"[CLEANUP] could not remove unreadable output: {exc}")

        tmp_dir = out_path / ".tmp_depthcrafter"
        if tmp_dir.is_dir():
            try:
                shutil.rmtree(tmp_dir)
                self._append_depth_log(f"[CLEANUP] removed temp dir: {tmp_dir}")
            except Exception as exc:
                self._append_depth_log(f"[CLEANUP] could not remove temp dir: {exc}")

    def _set_depth_running(self, is_running: bool) -> None:
        self.depth_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.depth_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.depth_upscale_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.depth_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        verify_state = tk.DISABLED if (is_running or self._verify_running) else tk.NORMAL
        self.depth_verify_quick_btn.configure(state=verify_state)
        self.depth_verify_deep_btn.configure(state=verify_state)
        self.depth_upscaled_verify_quick_btn.configure(state=verify_state)
        self.depth_upscaled_verify_deep_btn.configure(state=verify_state)
        if is_running:
            self.depth_stop_btn.configure(text="Stop")
        else:
            self.depth_stop_btn.configure(text="Stop")
            self._depth_stop_clicks = 0
            self._depth_stop_requested = False

    def _open_splat_input_clips_folder(self) -> None:
        folder = self.splat_input_clips_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_splat_log(f"Splat input clips folder ready: {folder}")

    def _open_splat_input_depth_folder(self) -> None:
        folder = self.splat_input_depth_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_splat_log(f"Splat input depth folder ready: {folder}")

    def _open_splat_output_folder(self) -> None:
        folder = self.splat_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_splat_log(f"Splat output folder ready: {folder}")

    def _open_splat_mask_output_folder(self) -> None:
        folder = self.splat_mask_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_splat_log(f"Splat mask output folder ready: {folder}")

    def _on_splat_mode_changed(self, _event=None) -> None:
        mode = self.splat_mode_var.get().strip()
        if mode == "Manual":
            self.splat_info_text_var.set(self.SPLAT_MANUAL_INFO)
        else:
            self.splat_mode_var.set("Auto (recommended)")
            self.splat_info_text_var.set(self.SPLAT_AUTO_INFO)
            self._reset_splat_auto_locked_defaults()
        self._apply_splat_control_states()

    def _reset_splat_auto_locked_defaults(self) -> None:
        # Fields disabled in Auto mode.
        self.splat_layout_var.set("Single Warp")
        self.splat_auto_convergence_var.set("Min Borders")
        self.splat_dilate_x_var.set("3")
        self.splat_dilate_y_var.set("3")
        self.splat_blur_x_var.set("0")
        self.splat_blur_y_var.set("0")
        self.splat_dilate_left_var.set("2")
        self.splat_blur_balance_var.set("0.5")
        self.splat_gamma_var.set("1")
        self.splat_convergence_var.set("50")
        self.splat_stair_smooth_var.set(True)
        self.splat_stair_kernel_var.set("3")
        self.splat_stair_x_off_var.set("2")
        self.splat_stair_strip_var.set("4")
        self.splat_stair_strength_var.set("1")
        self.splat_replace_mask_var.set(True)
        self.splat_replace_mask_scale_var.set("1")
        self.splat_replace_mask_min_var.set("1")
        self.splat_replace_mask_max_var.set("32")
        self.splat_replace_mask_gap_var.set("0")
        self.splat_replace_mask_edge_var.set(False)

    def _on_splat_layout_changed(self, _event=None) -> None:
        self._apply_splat_control_states()

    def _on_splat_auto_convergence_changed(self, _event=None) -> None:
        self._apply_splat_control_states()

    def _on_splat_replace_mask_toggled(self) -> None:
        if not self.splat_replace_mask_var.get():
            self.splat_replace_mask_var.set(True)
        self._update_replace_mask_dependent_controls()
        self._preview_splat_command()

    def _apply_splat_control_states(self) -> None:
        mode_manual = self.splat_mode_var.get().strip() == "Manual"

        # Always editable in both modes.
        self.splat_batch_size_entry.configure(state=tk.NORMAL)
        self.splat_disparity_entry.configure(state=tk.NORMAL)
        self.splat_workers_entry.configure(state=tk.NORMAL)

        # Manual-only widgets.
        for widget in getattr(self, "_splat_manual_entry_widgets", []):
            widget.configure(state=tk.NORMAL if mode_manual else tk.DISABLED)
        for widget in getattr(self, "_splat_manual_combo_widgets", []):
            widget.configure(state="readonly" if mode_manual else tk.DISABLED)
        for widget in getattr(self, "_splat_manual_check_widgets", []):
            widget.configure(state=tk.NORMAL if mode_manual else tk.DISABLED)

        # Convergence is editable only when manual and auto-convergence is Off.
        auto_conv_off = self.splat_auto_convergence_var.get().strip() == "Off"
        self.splat_convergence_entry.configure(
            state=tk.NORMAL if (mode_manual and auto_conv_off) else tk.DISABLED
        )

        # Replace-mask is mandatory for the whole merge/CT pipeline.
        self.splat_replace_mask_var.set(True)
        self.splat_replace_mask_check.configure(state=tk.DISABLED)

        self._update_replace_mask_dependent_controls()
        self._preview_splat_command()

    def _sync_splat_encoding_from_scene(self) -> None:
        self.splat_codec_var.set(
            self._normalize_ffmpeg_codec(
                self.scene_codec_var.get(),
                self.DEFAULT_SCENE_CODEC,
            )
        )
        self.splat_crf_var.set(self.scene_crf_var.get().strip())
        self.splat_preset_var.set(self.scene_encoder_preset_var.get().strip())
        self.splat_pix_fmt_var.set(self.scene_pix_fmt_var.get().strip())

    def _on_splat_override_toggle(self, initial: bool = False) -> None:
        enabled = bool(self.splat_encode_override_var.get())
        if not enabled:
            self._sync_splat_encoding_from_scene()
        elif not initial and not self._splat_override_notice_shown:
            self._splat_override_notice_shown = True
            messagebox.showwarning("Splat Encode Override", self.SPLAT_OVERRIDE_WARNING)

        state = tk.NORMAL if enabled else tk.DISABLED
        self._set_codec_widget_override_state(self.splat_codec_entry, enabled)
        for widget in (
            self.splat_crf_entry,
            self.splat_preset_entry,
            self.splat_pixfmt_entry,
            self.splat_extra_ffmpeg_entry,
        ):
            widget.configure(state=state)
        self._preview_splat_command()

    def _build_splat_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        layout_ui = self.splat_layout_var.get().strip()
        layout_cli = {
            "Single Warp": "single_warp",
            "Dual": "dual",
            "Quad": "quad",
        }.get(layout_ui, "single_warp")
        codec_value = self._normalize_ffmpeg_codec(
            self.splat_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        self.splat_codec_var.set(codec_value)
        workers_raw = self.splat_workers_var.get().strip()
        try:
            workers = max(1, int(workers_raw))
        except Exception:
            workers = 4
            self.splat_workers_var.set(str(workers))
        auto_conv_ui = self.splat_auto_convergence_var.get().strip()
        auto_conv_cli = "Off" if auto_conv_ui == "Off" else "MinBorders"
        replace_mask_enabled = True

        env_updates: dict[str, str] = {
            "PYTHON": sys.executable,
            "RUNNER": "batch_splatting_runner.py",
            "GUI_SCRIPT": "splatting_gui.py",
            "INPUT_SOURCE_CLIPS": self.splat_input_clips_var.get().strip(),
            "INPUT_DEPTH_MAPS": self.splat_input_depth_var.get().strip(),
            "OUTPUT_SPLATTED": self.splat_output_var.get().strip(),
            "MASK_OUTPUT": self.splat_mask_output_var.get().strip(),
            "FULL_RES_BATCH_SIZE": self.splat_batch_size_var.get().strip() or "50",
            "WORKERS": str(workers),
            "DISPARITY": self.splat_disparity_var.get().strip() or "20",
            "OUTPUT_LAYOUT": layout_cli,
            "AUTO_CONVERGENCE_MODE": auto_conv_cli,
            "DILATE_X": self.splat_dilate_x_var.get().strip() or "3",
            "DILATE_Y": self.splat_dilate_y_var.get().strip() or "3",
            "BLUR_X": self.splat_blur_x_var.get().strip() or "0",
            "BLUR_Y": self.splat_blur_y_var.get().strip() or "0",
            "DILATE_LEFT": self.splat_dilate_left_var.get().strip() or "2",
            "BLUR_BALANCE": self.splat_blur_balance_var.get().strip() or "0.5",
            "GAMMA": self.splat_gamma_var.get().strip() or "1",
            "CONVERGENCE": self.splat_convergence_var.get().strip() or "50",
            "STAIR_SMOOTH": "1" if self.splat_stair_smooth_var.get() else "0",
            "STAIR_SMOOTH_KERNEL": self.splat_stair_kernel_var.get().strip() or "3",
            "STAIR_SMOOTH_X_OFF": self.splat_stair_x_off_var.get().strip() or "2",
            "STAIR_SMOOTH_STRIP": self.splat_stair_strip_var.get().strip() or "4",
            "STAIR_SMOOTH_STRENGTH": self.splat_stair_strength_var.get().strip() or "1",
            "USE_REPLACE_MASK": "1" if replace_mask_enabled else "0",
            "REPLACE_MASK_SCALE": self.splat_replace_mask_scale_var.get().strip() or "1",
            "REPLACE_MASK_MIN": self.splat_replace_mask_min_var.get().strip() or "1",
            "REPLACE_MASK_MAX": self.splat_replace_mask_max_var.get().strip() or "32",
            "REPLACE_MASK_GAP": self.splat_replace_mask_gap_var.get().strip() or "0",
            "REPLACE_MASK_EDGE": "1" if self.splat_replace_mask_edge_var.get() else "0",
            # Fixed hardcoded settings (not exposed in GUI).
            "ENABLE_FULL_RES": "True",
            "ENABLE_LOW_RES": "False",
            "PROCESS_LENGTH": "-1",
            "ADD_BORDERS": "False",
            "REPLACE_MASK_CODEC": "ffv1",
            "FFMPEG_CODEC": codec_value,
            "FFMPEG_CRF": self.splat_crf_var.get().strip() or "1",
            "FFMPEG_PRESET": self.splat_preset_var.get().strip() or "fast",
            "FFMPEG_PIX_FMT": self.splat_pix_fmt_var.get().strip() or "yuv420p",
            "FFMPEG_EXTRA_ARGS": self.splat_extra_ffmpeg_args_var.get().strip(),
            "STOP_MARKER": os.path.join(
                self.splat_output_var.get().strip() or "./work/splat",
                ".stop_after_current",
            ),
        }

        launcher_name = (
            "run_splatting_runner_parallel.sh"
            if workers >= 2
            else "run_splatting_runner.sh"
        )
        cmd = ["bash", launcher_name]
        preview = " ".join(
            [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
            + [shlex.quote(x) for x in cmd]
        )
        return cmd, env_updates, preview

    def _preview_splat_command(self) -> None:
        try:
            _cmd, _env, preview = self._build_splat_runner_payload()
            self.splat_cmd_preview_var.set(preview)
        except Exception as e:
            self.splat_cmd_preview_var.set(f"Invalid options: {e}")

    def _append_splat_log(self, line: str) -> None:
        self.splat_log_text.configure(state=tk.NORMAL)
        self.splat_log_text.insert(tk.END, line + "\n")
        self.splat_log_text.see(tk.END)
        self.splat_log_text.configure(state=tk.DISABLED)

    def _clear_splat_log(self) -> None:
        self.splat_log_text.configure(state=tk.NORMAL)
        self.splat_log_text.delete("1.0", tk.END)
        self.splat_log_text.configure(state=tk.DISABLED)

    def _run_splat_placeholder(self) -> None:
        if self._splat_thread and self._splat_thread.is_alive():
            messagebox.showinfo("Splatting", "Splatting is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Splatting", "Stop verification before starting Splatting.")
            return
        try:
            cmd, env_updates, _preview = self._build_splat_runner_payload()
        except Exception as exc:
            messagebox.showerror("Splatting", f"Invalid splatting options:\n{exc}")
            return

        launcher_script = Path(cmd[1]).resolve() if len(cmd) > 1 else Path("run_splatting_runner.sh").resolve()
        if not launcher_script.is_file():
            messagebox.showerror("Splatting", f"Launcher not found:\n{launcher_script}")
            return

        runner_script = Path(env_updates.get("RUNNER", "batch_splatting_runner.py")).resolve()
        if not runner_script.is_file():
            messagebox.showerror("Splatting", f"Runner not found:\n{runner_script}")
            return

        gui_script = Path(env_updates.get("GUI_SCRIPT", "splatting_gui.py")).resolve()
        if not gui_script.is_file():
            messagebox.showerror("Splatting", f"GUI script not found:\n{gui_script}")
            return

        input_clips = self.splat_input_clips_var.get().strip()
        input_depth = self.splat_input_depth_var.get().strip()
        out_splat = self.splat_output_var.get().strip()
        out_mask = self.splat_mask_output_var.get().strip()
        if not input_clips or not os.path.isdir(input_clips):
            messagebox.showerror("Splatting", f"Input clips folder not found:\n{input_clips or '(empty)'}")
            return
        if not input_depth or not os.path.isdir(input_depth):
            messagebox.showerror("Splatting", f"Input depth folder not found:\n{input_depth or '(empty)'}")
            return
        if not out_splat:
            messagebox.showerror("Splatting", "Splat output folder is required.")
            return
        os.makedirs(out_splat, exist_ok=True)
        if out_mask:
            os.makedirs(out_mask, exist_ok=True)

        self._splat_stop_requested = False
        self._splat_stop_clicks = 0
        self.splat_status_var.set("Starting...")
        self.splat_progress_var.set(0.0)
        self._set_splat_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("splatting")
        self._append_splat_log("=== Splatting started ===")
        self._append_splat_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_splat_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._splat_thread = threading.Thread(
            target=self._run_splat_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._splat_thread.start()

    def _run_splat_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
        proc = None
        try:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_updates.items()})
            preexec = os.setsid if hasattr(os, "setsid") else None
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=preexec,
            )
            self._splat_process = proc
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("splat_line", line))
                    self._try_parse_splat_progress(line)
                if self._splat_stop_requested:
                    break
            rc = proc.wait()
            if self._splat_stop_requested:
                self._log_queue.put(("splat_status", "Stopped by user"))
            elif rc == 0:
                self._log_queue.put(("splat_status", "Completed"))
                self._log_queue.put(("splat_progress", "100"))
            else:
                self._log_queue.put(("splat_status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("splat_line", f"[SPLAT][ERROR] {exc}"))
            self._log_queue.put(("splat_status", "Failed"))
        finally:
            self._splat_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("splat_done", "1"))

    def _is_splat_replace_mask_active(self) -> bool:
        return True

    @staticmethod
    def _has_any_replace_masks(mask_dir: str) -> bool:
        if not mask_dir or not os.path.isdir(mask_dir):
            return False
        for pat in ("*_replace_mask.mkv", "*_replace_mask.mp4", "*_replace_mask.webm", "*_replace_mask.avi"):
            if glob.glob(os.path.join(mask_dir, pat)):
                return True
        if glob.glob(os.path.join(mask_dir, "*_replace_mask.*")):
            return True
        return False

    @staticmethod
    def _find_missing_replace_masks(video_dir: str, mask_dir: str) -> list[str]:
        if not video_dir or not os.path.isdir(video_dir):
            return []
        if not mask_dir or not os.path.isdir(mask_dir):
            video_files = sorted(
                [
                    p
                    for ext in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm")
                    for p in Path(video_dir).glob(ext)
                    if p.is_file()
                ]
            )
            return [p.name for p in video_files]
        missing: list[str] = []
        video_files = sorted(
            [
                p
                for ext in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm")
                for p in Path(video_dir).glob(ext)
                if p.is_file()
            ]
        )
        for p in video_files:
            stem = p.stem
            patt = os.path.join(mask_dir, f"{stem}_replace_mask.*")
            if not glob.glob(patt):
                missing.append(p.name)
        return missing

    @staticmethod
    def _parse_inpainted_core_with_width(base_name: str) -> str:
        base = str(base_name or "")
        for suffix in ("_inpainted_right_eye.mp4", "_inpainted_sbs.mp4"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return ""

    @staticmethod
    def _collect_inpainted_scene_files(inpainted_dir: str) -> list[Path]:
        if not inpainted_dir or not os.path.isdir(inpainted_dir):
            return []
        files: dict[str, Path] = {}
        for pat in ("*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"):
            for p in Path(inpainted_dir).glob(pat):
                if p.is_file():
                    files[p.name] = p
        return [files[k] for k in sorted(files.keys())]

    @staticmethod
    def _collect_video_files_for_patterns(folder: str, patterns: list[str] | tuple[str, ...]) -> list[Path]:
        if not folder or not os.path.isdir(folder):
            return []
        files: dict[str, Path] = {}
        root = Path(folder)
        for pat in patterns:
            for p in root.glob(pat):
                if p.is_file():
                    files[p.name] = p
        return [files[k] for k in sorted(files.keys())]

    @staticmethod
    def _find_join_mono_output_candidates(sbs_dir: str, stem: str) -> list[Path]:
        if not sbs_dir or not os.path.isdir(sbs_dir):
            return []
        root = Path(sbs_dir)
        hits: dict[str, Path] = {}
        patterns = (
            f"{stem}_*_merged_*.mp4",
            f"{stem}_*_merged_*.mkv",
            f"{stem}_*_merged_*.mov",
            f"{stem}_*_merged_*.avi",
            f"{stem}_*_merged_*.webm",
        )
        for pat in patterns:
            for p in root.glob(pat):
                if p.is_file():
                    hits[p.name] = p
        return [hits[k] for k in sorted(hits.keys())]

    def _expected_mono_codec_family(self) -> str:
        codec = self._normalize_ffmpeg_codec(
            self.merge_codec_var.get(),
            self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
        )
        return "hevc" if "265" in codec or codec.startswith("hevc") else "h264"

    def _build_join_mono_expected_output_name(self, stem: str, width: int) -> str:
        suffix = "_merged_half_sbs.mp4" if self._join_layout_for_seg_mono() == "half_sbs" else "_merged_full_sbs.mp4"
        return f"{stem}_{int(width)}{suffix}"

    def _collect_join_mono_expected_pairs(self) -> list[dict[str, object]]:
        seg_mono_dir = self.join_seg_mono_var.get().strip()
        sbs_dir = self.join_input_var.get().strip()
        source_files = self._collect_video_files_for_patterns(
            seg_mono_dir,
            self.VERIFY_VIDEO_PATTERNS,
        )
        pairs: list[dict[str, object]] = []
        expected_codec_family = self._expected_mono_codec_family()
        expected_pix_fmt = self.merge_pix_fmt_var.get().strip().lower()
        for src in source_files:
            src_meta = self._probe_video_basic(str(src))
            src_width = src_meta.get("width")
            src_height = src_meta.get("height")
            expected_name = ""
            expected_output = Path(sbs_dir).resolve() / "__missing__"
            if src_width not in (None, "", "N/A"):
                try:
                    expected_name = self._build_join_mono_expected_output_name(
                        src.stem,
                        int(src_width),
                    )
                    expected_output = Path(sbs_dir).resolve() / expected_name
                except Exception:
                    expected_name = ""
            pairs.append(
                {
                    "source_path": src,
                    "source_meta": src_meta,
                    "expected_name": expected_name,
                    "expected_output": expected_output,
                    "matches": self._find_join_mono_output_candidates(sbs_dir, src.stem),
                    "expected_width": (
                        None
                        if src_width in (None, "", "N/A")
                        else (int(src_width) if self._join_layout_for_seg_mono() == "half_sbs" else int(src_width) * 2)
                    ),
                    "expected_height": None if src_height in (None, "", "N/A") else int(src_height),
                    "expected_codec_family": expected_codec_family,
                    "expected_pix_fmt": expected_pix_fmt,
                }
            )
        return pairs

    def _verify_join_mono_outputs_coverage(
        self,
        *,
        cleanup_incomplete: bool,
    ) -> tuple[bool, str, list[str], list[str]]:
        pairs = self._collect_join_mono_expected_pairs()
        if not pairs:
            return True, "Mono->SBS verify: no seg-mono clips found.", [], []

        missing: list[str] = []
        broken_output: list[str] = []
        broken_reference: list[str] = []
        details: list[str] = []
        seen_output: set[str] = set()
        seen_reference: set[str] = set()

        for pair in pairs:
            source_path = Path(pair["source_path"])
            source_meta = dict(pair["source_meta"])
            expected_output = Path(pair["expected_output"])
            expected_name = str(pair["expected_name"] or "")
            matches = [Path(p) for p in pair.get("matches") or []]
            extras = [p for p in matches if p.resolve() != expected_output.resolve()]
            for extra in extras:
                extra_s = str(extra)
                if extra_s not in seen_output:
                    seen_output.add(extra_s)
                    broken_output.append(extra_s)
                details.append(f"{source_path.name}: stale alternative output present -> {extra.name}")

            if not bool(source_meta.get("ok", False)):
                src_s = str(source_path)
                if src_s not in seen_reference:
                    seen_reference.add(src_s)
                    broken_reference.append(src_s)
                details.append(
                    f"{source_path.name}: source probe failed ({source_meta.get('error', 'unknown error')})"
                )
                continue

            src_frames = source_meta.get("frames")
            src_width = pair.get("expected_width")
            src_height = pair.get("expected_height")
            if expected_name == "":
                src_s = str(source_path)
                if src_s not in seen_reference:
                    seen_reference.add(src_s)
                    broken_reference.append(src_s)
                details.append(f"{source_path.name}: could not derive expected output name from source probe")
                continue

            if not expected_output.is_file():
                missing.append(expected_name)
                details.append(f"{source_path.name}: missing expected output {expected_name}")
                continue

            target_meta = self._probe_video_basic(str(expected_output))
            if not bool(target_meta.get("ok", False)):
                out_s = str(expected_output)
                if out_s not in seen_output:
                    seen_output.add(out_s)
                    broken_output.append(out_s)
                details.append(
                    f"{expected_output.name}: target probe failed ({target_meta.get('error', 'unknown error')})"
                )
                continue

            if src_frames is None or target_meta.get("frames") is None:
                out_s = str(expected_output)
                if out_s not in seen_output:
                    seen_output.add(out_s)
                    broken_output.append(out_s)
                details.append(f"{expected_output.name}: packet count unavailable")
                continue

            frame_delta = abs(int(src_frames) - int(target_meta["frames"]))
            if frame_delta > 1:
                out_s = str(expected_output)
                if out_s not in seen_output:
                    seen_output.add(out_s)
                    broken_output.append(out_s)
                details.append(
                    f"{expected_output.name}: packet mismatch source={int(src_frames)} target={int(target_meta['frames'])}"
                )

            if src_width is not None and target_meta.get("width") not in (None, "", "N/A"):
                if int(target_meta["width"]) != int(src_width):
                    out_s = str(expected_output)
                    if out_s not in seen_output:
                        seen_output.add(out_s)
                        broken_output.append(out_s)
                    details.append(
                        f"{expected_output.name}: width mismatch target={int(target_meta['width'])} expected={int(src_width)}"
                    )
            if src_height is not None and target_meta.get("height") not in (None, "", "N/A"):
                if int(target_meta["height"]) != int(src_height):
                    out_s = str(expected_output)
                    if out_s not in seen_output:
                        seen_output.add(out_s)
                        broken_output.append(out_s)
                    details.append(
                        f"{expected_output.name}: height mismatch target={int(target_meta['height'])} expected={int(src_height)}"
                    )

            expected_codec_family = str(pair.get("expected_codec_family") or "").strip().lower()
            target_codec = str(target_meta.get("codec_name") or "").strip().lower()
            if expected_codec_family and target_codec and target_codec != expected_codec_family:
                out_s = str(expected_output)
                if out_s not in seen_output:
                    seen_output.add(out_s)
                    broken_output.append(out_s)
                details.append(
                    f"{expected_output.name}: codec mismatch target={target_codec} expected={expected_codec_family}"
                )

            expected_pix_fmt = str(pair.get("expected_pix_fmt") or "").strip().lower()
            target_pix_fmt = str(target_meta.get("pix_fmt") or "").strip().lower()
            if expected_pix_fmt and target_pix_fmt and target_pix_fmt != expected_pix_fmt:
                out_s = str(expected_output)
                if out_s not in seen_output:
                    seen_output.add(out_s)
                    broken_output.append(out_s)
                details.append(
                    f"{expected_output.name}: pix_fmt mismatch target={target_pix_fmt} expected={expected_pix_fmt}"
                )

        if cleanup_incomplete and broken_output:
            self._delete_file_paths(broken_output)

        if not missing and not broken_output and not broken_reference:
            return True, f"Mono->SBS verified: {len(pairs)}/{len(pairs)} clip(s) ready.", [], []

        detail_txt = " | ".join(details[:6])
        if len(details) > 6:
            detail_txt += f" | ... +{len(details) - 6} more"
        msg = (
            f"Mono->SBS incomplete/broken on {len(missing) + len(broken_output)} output item(s)"
            f" across {len(pairs)} seg-mono clip(s)."
        )
        if broken_reference:
            msg += f" Source issues={len(broken_reference)}."
        if detail_txt:
            msg += f" Details: {detail_txt}"
        return False, msg, broken_output, broken_reference

    @staticmethod
    def _find_splatted_for_core_with_width(splatted_dir: str, core_with_width: str) -> str:
        if not splatted_dir or not os.path.isdir(splatted_dir):
            return ""
        core = str(core_with_width or "").strip()
        if not core:
            return ""
        candidates = [
            os.path.join(splatted_dir, f"{core}_splatted1.mp4"),
            os.path.join(splatted_dir, f"{core}_splatted2.mp4"),
            os.path.join(splatted_dir, f"{core}_splatted4.mp4"),
            os.path.join(splatted_dir, f"{core}_splatted1.mkv"),
            os.path.join(splatted_dir, f"{core}_splatted2.mkv"),
            os.path.join(splatted_dir, f"{core}_splatted4.mkv"),
            os.path.join(splatted_dir, f"{core}_splatted*.mp4"),
            os.path.join(splatted_dir, f"{core}_splatted*.mkv"),
        ]
        for pat in candidates:
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[0]
        return ""

    @staticmethod
    def _find_replace_mask_for_splatted(splatted_path: str, mask_dir: str) -> str:
        if not splatted_path:
            return ""
        target_dir = str(mask_dir or "").strip()
        if not target_dir:
            target_dir = str(Path(splatted_path).resolve().parent)
        stem = Path(splatted_path).stem
        hits = sorted(glob.glob(os.path.join(target_dir, f"{stem}_replace_mask.*")))
        return hits[0] if hits else ""

    @staticmethod
    def _load_sharpness_csv_entries(csv_path: str) -> set[str]:
        rows: set[str] = set()
        if not csv_path or not os.path.isfile(csv_path):
            return rows
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = set(reader.fieldnames or [])
                if "file" not in fieldnames:
                    return rows
                for row in reader:
                    name = os.path.basename(str((row or {}).get("file", "")).strip())
                    if name:
                        rows.add(name)
        except Exception:
            return set()
        return rows

    def _verify_sharpness_csv_coverage(
        self,
        inpaint_input_dir: str,
        sharpness_csv_path: str,
    ) -> tuple[bool, str, list[str]]:
        if not inpaint_input_dir or not os.path.isdir(inpaint_input_dir):
            return False, f"Inpaint input folder not found: {inpaint_input_dir or '(empty)'}", []
        if not sharpness_csv_path or not os.path.isfile(sharpness_csv_path):
            return False, f"sharpness.csv not found: {sharpness_csv_path or '(empty)'}", []

        expected = [p.name for p in sorted(Path(inpaint_input_dir).glob("*.mp4")) if p.is_file()]
        present = self._load_sharpness_csv_entries(sharpness_csv_path)
        missing = [name for name in expected if name not in present]
        if missing:
            return (
                False,
                f"Sharpness CSV incomplete: found {len(present)}/{len(expected)} scene rows.",
                missing,
            )
        return (
            True,
            f"Sharpness CSV verified: {len(expected)}/{len(expected)} scene rows.",
            [],
        )

    @staticmethod
    def _load_autoct_csv_rows(
        csv_path: str,
    ) -> tuple[list[str], list[dict[str, str]], dict[str, set[int]], dict[str, Counter]]:
        fieldnames: list[str] = []
        rows: list[dict[str, str]] = []
        frames_by_video: dict[str, set[int]] = {}
        status_by_video: dict[str, Counter] = {}
        if not csv_path or not os.path.isfile(csv_path):
            return fieldnames, rows, frames_by_video, status_by_video

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row_dict: dict[str, str] = {
                    str(k): str(v) if v is not None else ""
                    for k, v in dict(row or {}).items()
                }
                rows.append(row_dict)
                video = str(row_dict.get("video", "")).strip()
                if not video:
                    continue
                status = str(row_dict.get("status", "")).strip().lower() or "unknown"
                status_by_video.setdefault(video, Counter())[status] += 1
                frame_raw = str(row_dict.get("frame", "")).strip()
                try:
                    frame_idx = int(float(frame_raw))
                except Exception:
                    continue
                frames_by_video.setdefault(video, set()).add(frame_idx)
        return fieldnames, rows, frames_by_video, status_by_video

    def _verify_autoct_csv_packet_coverage(
        self,
        inpainted_dir: str,
        splatted_dir: str,
        replace_mask_dir: str,
        csv_path: str,
        *,
        cleanup_incomplete: bool,
    ) -> tuple[bool, str, list[str]]:
        if not inpainted_dir or not os.path.isdir(inpainted_dir):
            raise RuntimeError(f"Inpainted folder not found: {inpainted_dir or '(empty)'}")
        if not splatted_dir or not os.path.isdir(splatted_dir):
            raise RuntimeError(f"Splatted folder not found: {splatted_dir or '(empty)'}")
        if not replace_mask_dir or not os.path.isdir(replace_mask_dir):
            raise RuntimeError(f"Replace-mask folder not found: {replace_mask_dir or '(empty)'}")
        if not csv_path or not os.path.isfile(csv_path):
            return False, f"autoct.csv not found: {csv_path or '(empty)'}", []

        expected_files = self._collect_inpainted_scene_files(inpainted_dir)
        if not expected_files:
            return True, "AutoCT CSV verify: no inpainted scenes found.", []

        fieldnames, rows, frames_by_video, status_by_video = self._load_autoct_csv_rows(
            csv_path
        )

        incomplete: list[str] = []
        incomplete_details: list[str] = []
        for inpainted_path in expected_files:
            video_name = inpainted_path.name
            core_with_width = self._parse_inpainted_core_with_width(video_name)
            if not core_with_width:
                incomplete.append(video_name)
                incomplete_details.append(
                    f"{video_name}: unsupported filename pattern"
                )
                continue
            splatted_path = self._find_splatted_for_core_with_width(
                splatted_dir, core_with_width
            )
            if not splatted_path:
                raise RuntimeError(
                    f"Missing splatted clip for autoct verify: {video_name} ({core_with_width}_splatted*)"
                )
            replace_mask_path = self._find_replace_mask_for_splatted(
                splatted_path, replace_mask_dir
            )
            if not replace_mask_path:
                raise RuntimeError(
                    f"Missing replace-mask for autoct verify: {os.path.basename(splatted_path)}"
                )
            probe = self._probe_video_basic(replace_mask_path)
            if not bool(probe.get("ok", False)):
                raise RuntimeError(
                    f"ffprobe failed for replace-mask '{replace_mask_path}': {probe.get('error', 'unknown error')}"
                )
            expected_packets_raw = probe.get("frames", None)
            if expected_packets_raw is None:
                raise RuntimeError(
                    f"No packet count reported for replace-mask '{replace_mask_path}'."
                )
            expected_packets = max(0, int(expected_packets_raw))
            frames_set = frames_by_video.get(video_name, set())
            in_range = {fi for fi in frames_set if 0 <= int(fi) < expected_packets}
            complete = (
                expected_packets > 0
                and len(frames_set) == expected_packets
                and len(in_range) == expected_packets
            )
            if not complete:
                incomplete.append(video_name)
                stat = status_by_video.get(video_name, Counter())
                incomplete_details.append(
                    (
                        f"{video_name}: rows={len(frames_set)}/{expected_packets} "
                        f"(ok={int(stat.get('ok', 0))}, "
                        f"low_mask={int(stat.get('low_mask', 0))}, "
                        f"selector_error={int(stat.get('selector_error', 0))})"
                    )
                )

        if not incomplete:
            return True, "AutoCT CSV verified by packet counts for all scenes.", []

        if cleanup_incomplete:
            drop = set(incomplete)
            kept_rows = [
                row
                for row in rows
                if str(row.get("video", "")).strip() not in drop
            ]
            out_fieldnames = (
                fieldnames
                if fieldnames
                else ["video", "frame", "best_preset", "valid_mask", "mask_pixels", "status"]
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=out_fieldnames)
                writer.writeheader()
                for row in kept_rows:
                    writer.writerow({k: row.get(k, "") for k in out_fieldnames})

        details = " | ".join(incomplete_details[:6])
        if len(incomplete_details) > 6:
            details += f" | ... +{len(incomplete_details) - 6} more"
        msg = (
            f"AutoCT CSV incomplete on {len(incomplete)} scene(s)."
            + (f" Details: {details}" if details else "")
        )
        if cleanup_incomplete:
            msg += " Incomplete scene rows were purged and will be regenerated."
        return False, msg, incomplete

    def _update_replace_mask_dependent_controls(self) -> None:
        mask_expected = self._is_splat_replace_mask_active()
        mask_dir = self.splat_mask_output_var.get().strip() or self.inpaint_mask_var.get().strip()
        has_masks = self._has_any_replace_masks(mask_dir)
        allow_csv_features = bool(mask_expected and has_masks)

        inpaint_running = bool(self._inpaint_thread and self._inpaint_thread.is_alive())
        merge_running = bool(self._merge_thread and self._merge_thread.is_alive())

        if hasattr(self, "inpaint_sharp_btn"):
            self.inpaint_sharp_btn.configure(
                state=tk.NORMAL if (allow_csv_features and not inpaint_running) else tk.DISABLED
            )
        if hasattr(self, "merge_csv_btn"):
            self.merge_csv_btn.configure(
                state=tk.NORMAL if (allow_csv_features and not merge_running) else tk.DISABLED
            )

    @staticmethod
    def _collect_files_for_patterns(folder: str, patterns: list[str]) -> list[str]:
        root = Path(folder)
        found: dict[str, str] = {}
        for pat in patterns:
            for p in root.glob(pat):
                if p.is_file():
                    found[str(p.resolve())] = str(p.resolve())
        return sorted(found.values())

    def _pipeline_filter_files_to_test_subset(self, files: list[str]) -> list[str]:
        normalized = sorted({str(x) for x in (files or []) if str(x).strip()})
        if not self._pipeline_test_active:
            return normalized
        scene_stems = [str(s).strip() for s in (self._pipeline_test_scene_stems or []) if str(s).strip()]
        if not scene_stems:
            scene_stems = [
                self._pipeline_scene_stem_from_name(x)
                for x in (self._pipeline_test_manifest or [])
            ]
            scene_stems = [str(s).strip() for s in scene_stems if str(s).strip()]
        if not scene_stems:
            return normalized
        filtered: list[str] = []
        for fp in normalized:
            name = Path(fp).name
            if any(self._pipeline_name_matches_scene_stem(name, stem) for stem in scene_stems):
                filtered.append(fp)
        return filtered

    def _pipeline_prepare_verify_subset_dir(
        self, folder: str, tag: str, patterns: list[str]
    ) -> str:
        try:
            resolved = str(Path(folder).resolve())
        except Exception:
            resolved = str(folder)
        if not self._pipeline_test_active:
            return resolved
        if not resolved or not os.path.isdir(resolved):
            return resolved
        test_root_raw = str(self._pipeline_test_dir or "").strip()
        if not test_root_raw:
            return resolved
        test_root = Path(test_root_raw)
        if not test_root.is_dir():
            return resolved
        all_files = self._collect_files_for_patterns(resolved, patterns)
        selected_files = self._pipeline_filter_files_to_test_subset(all_files)
        if not selected_files:
            return resolved
        safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(tag).strip() or "verify")
        subset_dir = test_root / "_verify_subset" / safe_tag
        try:
            if subset_dir.exists():
                shutil.rmtree(subset_dir, ignore_errors=True)
            subset_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return resolved
        linked = 0
        for fp in selected_files:
            src = Path(fp)
            if self._pipeline_link_or_copy_file(src, subset_dir / src.name):
                linked += 1
        if linked <= 0:
            return resolved
        try:
            return str(subset_dir.resolve())
        except Exception:
            return str(subset_dir)

    def _resolve_splat_hires_dir(self) -> str:
        root = self.splat_output_var.get().strip()
        if not root:
            return ""
        hires = os.path.join(root, "hires")
        if os.path.isdir(hires):
            return hires
        return root

    def _resolve_verify_reference(
        self, stage: str, dialog_title: str
    ) -> tuple[bool, str, list[str], str]:
        stage_key = str(stage).strip().lower()
        depth_ref_dir = self.depth_output_var.get().strip()

        if stage_key == "splat":
            if not depth_ref_dir:
                messagebox.showerror(dialog_title, "Reference depth folder is required.")
                return False, "", [], ""
            if not os.path.isdir(depth_ref_dir):
                messagebox.showerror(
                    dialog_title,
                    f"Reference depth folder not found:\n{depth_ref_dir}",
                )
                return False, "", [], ""
            depth_ref_dir = self._pipeline_prepare_verify_subset_dir(
                depth_ref_dir,
                "ref_splat_depth",
                list(self.VERIFY_VIDEO_PATTERNS),
            )
            return True, depth_ref_dir, list(self.VERIFY_VIDEO_PATTERNS), "depthmap"

        if stage_key in {"inpaint", "merge", "merge_mask"}:
            mask_candidates = [
                self.merge_replace_mask_var.get().strip(),
                self.inpaint_mask_var.get().strip(),
                self.splat_mask_output_var.get().strip(),
            ]
            seen_dirs: set[str] = set()
            for cand in mask_candidates:
                if not cand:
                    continue
                norm = os.path.normpath(cand)
                if norm in seen_dirs:
                    continue
                seen_dirs.add(norm)
                if self._has_any_replace_masks(cand):
                    ref_dir = self._pipeline_prepare_verify_subset_dir(
                        cand,
                        f"ref_{stage_key}_replace_mask",
                        list(self.VERIFY_REPLACE_MASK_PATTERNS),
                    )
                    return (
                        True,
                        ref_dir,
                        list(self.VERIFY_REPLACE_MASK_PATTERNS),
                        "replace-mask",
                    )
            messagebox.showerror(
                dialog_title,
                (
                    "Replace-mask reference is required for this verify stage.\n"
                    "No fallback to depthmap is allowed in strict mode."
                ),
            )
            return False, "", [], ""

        messagebox.showerror(dialog_title, f"Unsupported verify reference stage: {stage}")
        return False, "", [], ""

    def _validate_splat_verify_inputs(self) -> tuple[bool, str, str, str, bool, list[str]]:
        splat_dir = self._resolve_splat_hires_dir()
        mask_dir = self.splat_mask_output_var.get().strip()
        check_mask = self._is_splat_replace_mask_active()
        if not splat_dir:
            messagebox.showerror("Verify Splatting", "Splat output folder is required.")
            return False, "", "", "", False, []
        if not os.path.isdir(splat_dir):
            messagebox.showerror("Verify Splatting", f"Splat output folder not found:\n{splat_dir}")
            return False, "", "", "", False, []
        ok_ref, ref_dir, ref_patterns, ref_kind = self._resolve_verify_reference(
            "splat", "Verify Splatting"
        )
        if not ok_ref:
            return False, "", "", "", False, []
        splat_dir = self._pipeline_prepare_verify_subset_dir(
            splat_dir, "splat_target", list(self.VERIFY_VIDEO_PATTERNS)
        )
        if check_mask and mask_dir:
            mask_dir = self._pipeline_prepare_verify_subset_dir(
                mask_dir, "splat_mask_target", list(self.VERIFY_REPLACE_MASK_PATTERNS)
            )
        self._append_splat_log(f"[VERIFY] reference source: {ref_kind} ({ref_dir})")
        if check_mask:
            if not mask_dir:
                messagebox.showerror(
                    "Verify Splatting",
                    "Replace-mask verification is enabled but mask folder is empty.",
                )
                return False, "", "", "", False, []
            if not os.path.isdir(mask_dir):
                messagebox.showerror(
                    "Verify Splatting",
                    f"Replace-mask folder not found:\n{mask_dir}",
                )
                return False, "", "", "", False, []
        return True, splat_dir, mask_dir, ref_dir, check_mask, ref_patterns

    def _start_splat_verify_quick(self) -> None:
        if self._splat_thread and self._splat_thread.is_alive():
            messagebox.showinfo("Verify Splatting", "Stop Splatting before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Splatting", "Another verification is already running.")
            return
        ok, splat_dir, mask_dir, ref_dir, check_mask, ref_patterns = self._validate_splat_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Splatting", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="splat_quick")
        self.splat_status_var.set("Verify (Quick) running...")
        self._append_splat_log("=== Verify Scenes (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_splat_verify_quick_worker,
            args=(splat_dir, mask_dir, ref_dir, check_mask, ref_patterns),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_splat_verify_quick_worker(
        self, splat_dir: str, mask_dir: str, ref_dir: str, check_mask: bool, ref_patterns: list[str]
    ) -> None:
        try:
            max_workers = self._get_verify_scenes_workers()
            ref_files = self._collect_files_for_patterns(ref_dir, ref_patterns)
            if not ref_files:
                self._log_queue.put(
                    (
                        "splat_verify_quick_result",
                        {"ok": False, "message": "No reference video files found in selected reference folder."},
                    )
                )
                return

            def _probe_group(file_list: list[str], label: str) -> dict:
                broken: list[str] = []
                total_duration = 0.0
                duration_available = True
                total_frames = 0
                frames_available = True

                def _probe_one(fp: str) -> tuple[str, dict]:
                    return fp, self._probe_video_basic(fp)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_probe_one, fp) for fp in file_list]
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        fp, meta = fut.result()
                        done += 1
                        if not meta.get("ok"):
                            broken.append(fp)
                            self._log_queue.put(
                                (
                                    "splat_line",
                                    f"[QUICK][{label.upper()}][BROKEN] {fp} :: {meta.get('error')}",
                                )
                            )
                        else:
                            dur = meta.get("duration")
                            frm = meta.get("frames")
                            if dur is None:
                                duration_available = False
                            else:
                                total_duration += float(dur)
                            if frm is None:
                                frames_available = False
                            else:
                                total_frames += int(frm)
                        if done % 25 == 0 or done == len(file_list):
                            self._log_queue.put(
                                (
                                    "splat_line",
                                    f"[QUICK][{label.upper()}] progress {done}/{len(file_list)}",
                                )
                            )
                return {
                    "broken": broken,
                    "total_duration": total_duration,
                    "duration_available": duration_available,
                    "total_frames": total_frames,
                    "frames_available": frames_available,
                }

            def _run_quick_check(
                target_dir: str, target_patterns: list[str], target_label: str
            ) -> tuple[bool, str, list[str]]:
                target_files = self._collect_files_for_patterns(target_dir, target_patterns)
                if not target_files:
                    return False, f"[{target_label}] no target files found in {target_dir}", []
                self._log_queue.put(
                    (
                        "splat_line",
                        (
                            f"[QUICK] checking {target_label} files={len(target_files)} and "
                            f"reference files={len(ref_files)} with {max_workers} workers"
                        ),
                    )
                )

                target_stats = _probe_group(target_files, target_label)
                ref_stats = _probe_group(ref_files, "reference")

                count_ok = len(target_files) == len(ref_files)
                count_msg = f"{target_label}={len(target_files)} vs reference={len(ref_files)}"

                duration_ok = False
                duration_msg = "n.d."
                if target_stats["duration_available"] and ref_stats["duration_available"]:
                    dd = abs(
                        float(target_stats["total_duration"]) - float(ref_stats["total_duration"])
                    )
                    duration_ok = dd <= 0.35
                    duration_msg = (
                        f"{target_label}={float(target_stats['total_duration']):.3f}s vs "
                        f"reference={float(ref_stats['total_duration']):.3f}s (delta={dd:.3f}s)"
                    )

                frames_ok = False
                frames_msg = "n.d."
                if target_stats["frames_available"] and ref_stats["frames_available"]:
                    df = abs(int(target_stats["total_frames"]) - int(ref_stats["total_frames"]))
                    frames_ok = df <= 1
                    frames_msg = (
                        f"{target_label}={int(target_stats['total_frames'])} vs "
                        f"reference={int(ref_stats['total_frames'])} (delta={df})"
                    )

                self._log_queue.put(
                    ("splat_line", f"[QUICK] {target_label} file count check: {count_msg}")
                )
                self._log_queue.put(
                    ("splat_line", f"[QUICK] {target_label} duration check: {duration_msg}")
                )
                self._log_queue.put(
                    ("splat_line", f"[QUICK] {target_label} packet check: {frames_msg}")
                )

                ok_final = (
                    not target_stats["broken"]
                    and not ref_stats["broken"]
                    and count_ok
                    and (frames_ok or frames_msg == "n.d.")
                )
                msg = (
                    f"[{target_label}] Broken target files: {len(target_stats['broken'])}; "
                    f"Broken reference files: {len(ref_stats['broken'])}; "
                    f"File count: {'YES' if count_ok else 'NO'} ({count_msg}); "
                    f"Duration (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')} ({duration_msg}); "
                    f"Frames: {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')} ({frames_msg})"
                )
                return ok_final, msg, target_stats["broken"]

            messages: list[str] = []
            broken_targets: list[str] = []
            ok1, msg1, broken1 = _run_quick_check(splat_dir, ["*.mp4"], "splat-hires")
            messages.append(msg1)
            broken_targets.extend(broken1)
            ok_final = ok1

            if check_mask:
                ok2, msg2, broken2 = _run_quick_check(mask_dir, ["*.mkv", "*.mp4"], "replace-mask")
                messages.append(msg2)
                broken_targets.extend(broken2)
                ok_final = ok_final and ok2

            self._log_queue.put(
                (
                    "splat_verify_quick_result",
                    {
                        "ok": ok_final,
                        "message": "Splat quick verify completed.\n" + "\n".join(messages),
                        "broken_targets": broken_targets,
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(
                (
                    "splat_verify_quick_result",
                    {
                        "ok": False,
                        "message": f"Splat quick verify failed: {type(e).__name__}: {e}",
                        "broken_targets": [],
                    },
                )
            )
        finally:
            self._log_queue.put(("verify_done", "splat_quick"))

    def _start_splat_verify_deep(self) -> None:
        if self._splat_thread and self._splat_thread.is_alive():
            messagebox.showinfo("Verify Splatting", "Stop Splatting before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Splatting", "Another verification is already running.")
            return
        ok, splat_dir, mask_dir, ref_dir, check_mask, ref_patterns = self._validate_splat_verify_inputs()
        if not ok:
            return

        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Splatting", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        steps: list[tuple[str, str, str, list[str]]] = [
            (
                "splat-hires",
                str(Path(splat_dir).resolve()),
                str(Path(ref_dir).resolve()),
                [".mp4"] if "*.mp4" in ref_patterns else [".mp4", ".mkv", ".mov", ".avi", ".webm"],
            )
        ]
        if check_mask:
            steps.append(
                (
                    "replace-mask",
                    str(Path(mask_dir).resolve()),
                    str(Path(ref_dir).resolve()),
                    [".mkv", ".mp4"],
                )
            )

        self._set_verify_running(True, mode="splat_deep")
        self.splat_status_var.set("Verify (Deep) running...")
        self._append_splat_log("=== Verify Scenes (Deep) started ===")
        self._append_splat_log(f"Deep verify steps: {len(steps)}")
        self._verify_thread = threading.Thread(
            target=self._run_splat_verify_deep_worker,
            args=(str(script_path), steps, workers),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_splat_verify_deep_worker(
        self,
        script_path: str,
        steps: list[tuple[str, str, str, list[str]]],
        workers: int,
    ) -> None:
        overall_rc = 0
        failed_dirs: list[str] = []
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            for label, target_dir, ref_dir, exts in steps:
                cmd = [
                    sys.executable,
                    script_path,
                    target_dir,
                    ref_dir,
                    "--extensions",
                    ",".join(exts),
                    "--workers",
                    str(workers),
                    "--probe-timeout-sec",
                    str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
                    "--probe-timeout-retries",
                    str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
                    "--delete",
                    "yes",
                    "--no-single-line-progress",
                ]
                self._log_queue.put(
                    ("splat_line", f"[DEEP] step={label} cmd: {' '.join(shlex.quote(x) for x in cmd)}")
                )
                rc = 1
                proc = None
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )
                    assert proc.stdout is not None
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip("\n")
                        if line:
                            self._log_queue.put(("splat_line", f"[DEEP][{label}] {line}"))
                            bad_path = self._resolve_verifyscenes_bad_path(line, target_dir)
                            if bad_path and bad_path not in seen_bad:
                                seen_bad.add(bad_path)
                                bad_files.append(bad_path)
                    rc = int(proc.wait() or 0)
                except Exception as e:
                    self._log_queue.put(("splat_line", f"[DEEP][{label}][ERROR] {e}"))
                    rc = 1
                finally:
                    if proc and proc.stdout:
                        try:
                            proc.stdout.close()
                        except Exception:
                            pass

                if rc != 0:
                    overall_rc = rc if overall_rc == 0 else overall_rc
                    failed_dirs.append(target_dir)
                    self._log_queue.put(("splat_line", f"[DEEP][{label}] failed with rc={rc}"))
                else:
                    self._log_queue.put(("splat_line", f"[DEEP][{label}] completed successfully"))
        finally:
            self._log_queue.put(
                (
                    "splat_verify_deep_result",
                    {"rc": overall_rc, "failed_dirs": failed_dirs, "bad_files": bad_files},
                )
            )
            self._log_queue.put(("verify_done", "splat_deep"))

    def _stop_splat_placeholder(self, prompt_user: bool = True) -> None:
        running = bool(self._splat_thread and self._splat_thread.is_alive())
        if not running:
            return
        if self._splat_stop_clicks == 0 and prompt_user:
            messagebox.showwarning(
                "Stop Splatting",
                "Graceful stop requested.\n\n"
                "Current process will be interrupted like Ctrl+C.\n"
                "Click Stop again to force kill immediately.",
            )
        self._splat_stop_requested = True
        self._splat_stop_clicks += 1

        if self._splat_stop_clicks == 1:
            self.splat_status_var.set("Graceful stop requested...")
            self._append_splat_log(
                "[STOP] graceful stop requested (click Stop again for immediate force stop)."
            )
            self.splat_stop_btn.configure(text="Force Stop")
        else:
            self.splat_status_var.set("Force stop requested...")
            self._append_splat_log("[STOP] force stop requested.")

        self._send_splat_signal(signal.SIGINT)
        if self._splat_stop_clicks >= 2:
            self.root.after(1000, self._force_kill_splat)

    def _send_splat_signal(self, sig: int) -> None:
        proc = self._splat_process
        if proc is None or proc.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except Exception as exc:
            self._append_splat_log(f"Signal send failed: {exc}")

    def _force_kill_splat(self) -> None:
        proc = self._splat_process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                self._append_splat_log("Splatting process force-killed.")
            except Exception as exc:
                self._append_splat_log(f"Splatting kill failed: {exc}")

    def _set_splat_running(self, is_running: bool) -> None:
        self.splat_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.splat_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.splat_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        verify_state = tk.DISABLED if (is_running or self._verify_running) else tk.NORMAL
        self.splat_verify_quick_btn.configure(state=verify_state)
        self.splat_verify_deep_btn.configure(state=verify_state)
        if is_running:
            self.splat_stop_btn.configure(text="Stop")
        else:
            self.splat_stop_btn.configure(text="Stop")
            self._splat_stop_clicks = 0
            self._splat_stop_requested = False
        self._update_replace_mask_dependent_controls()

    def _try_parse_splat_progress(self, line: str) -> None:
        m = re.search(r"^\[(?:RUN|OK|SKIP|ERR)\s*\]\s*(\d+)\s*/\s*(\d+)", line)
        if m:
            try:
                idx = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    prog = max(0.0, min(100.0, (idx / total) * 100.0))
                    self._log_queue.put(("splat_progress", str(prog)))
            except Exception:
                pass
            return
        if line.startswith("[DONE]"):
            self._log_queue.put(("splat_progress", "100"))

    def _try_parse_depth_progress(self, line: str) -> None:
        m = re.search(r"^\[(?:RUN|OK|SKIP|ERR)\s*\]\s*(\d+)\s*/\s*(\d+)", line)
        if m:
            try:
                idx = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    prog = max(0.0, min(100.0, (idx / total) * 100.0))
                    self._log_queue.put(("depth_progress", str(prog)))
            except Exception:
                pass
            return
        if line.startswith("[DONE]"):
            self._log_queue.put(("depth_progress", "100"))

    def _validate_depth_verify_inputs(self) -> tuple[bool, str, str]:
        depth_dir = self.depth_output_var.get().strip()
        ref_dir = self.depth_input_var.get().strip()
        if not depth_dir:
            messagebox.showerror("Verify Depth", "Depth output folder is required.")
            return False, "", ""
        if not os.path.isdir(depth_dir):
            messagebox.showerror("Verify Depth", f"Depth output folder not found:\n{depth_dir}")
            return False, "", ""
        if not ref_dir:
            messagebox.showerror("Verify Depth", "Reference scenes folder is required.")
            return False, "", ""
        if not os.path.isdir(ref_dir):
            messagebox.showerror("Verify Depth", f"Reference scenes folder not found:\n{ref_dir}")
            return False, "", ""
        depth_dir = self._pipeline_prepare_verify_subset_dir(
            depth_dir, "depth_target", ["*.mp4"]
        )
        ref_dir = self._pipeline_prepare_verify_subset_dir(
            ref_dir, "depth_reference", ["*.mp4"]
        )
        return True, depth_dir, ref_dir

    def _start_depth_verify_quick(self) -> None:
        if self._depth_thread and self._depth_thread.is_alive():
            messagebox.showinfo("Verify Depth", "Stop DepthCrafter before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Depth", "Another verification is already running.")
            return
        ok, depth_dir, ref_dir = self._validate_depth_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Depth", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="depth_quick")
        self.depth_status_var.set("Verify (Quick) running...")
        self._append_depth_log("=== Verify Scenes (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_depth_verify_quick_worker,
            args=(depth_dir, ref_dir),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_depth_verify_quick_worker(self, depth_dir: str, ref_dir: str) -> None:
        try:
            depth_files = sorted([str(p) for p in Path(depth_dir).glob("*.mp4") if p.is_file()])
            ref_files = sorted([str(p) for p in Path(ref_dir).glob("*.mp4") if p.is_file()])
            if not depth_files:
                self._log_queue.put(("depth_verify_quick_result", {
                    "ok": False,
                    "message": "No .mp4 files found in depth output folder.",
                    "broken_depth": [],
                    "broken_reference": [],
                }))
                return
            if not ref_files:
                self._log_queue.put(("depth_verify_quick_result", {
                    "ok": False,
                    "message": "No .mp4 files found in reference seg folder.",
                    "broken_depth": [],
                    "broken_reference": [],
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            self._log_queue.put(
                ("depth_line", f"[QUICK] checking depth files={len(depth_files)} and reference files={len(ref_files)} with {max_workers} workers")
            )

            def _probe_group(file_list: list[str], label: str) -> dict:
                broken: list[str] = []
                total_duration = 0.0
                duration_available = True
                total_frames = 0
                frames_available = True

                def _probe_one(fp: str) -> tuple[str, dict]:
                    return fp, self._probe_video_basic(fp)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_probe_one, fp) for fp in file_list]
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        fp, meta = fut.result()
                        done += 1
                        if not meta.get("ok"):
                            broken.append(fp)
                            self._log_queue.put(("depth_line", f"[QUICK][{label.upper()}][BROKEN] {fp} :: {meta.get('error')}"))
                        else:
                            dur = meta.get("duration")
                            frm = meta.get("frames")
                            if dur is None:
                                duration_available = False
                            else:
                                total_duration += float(dur)
                            if frm is None:
                                frames_available = False
                            else:
                                total_frames += int(frm)
                        if done % 25 == 0 or done == len(file_list):
                            self._log_queue.put(("depth_line", f"[QUICK][{label.upper()}] progress {done}/{len(file_list)}"))
                return {
                    "broken": broken,
                    "total_duration": total_duration,
                    "duration_available": duration_available,
                    "total_frames": total_frames,
                    "frames_available": frames_available,
                }

            depth_stats = _probe_group(depth_files, "depth")
            ref_stats = _probe_group(ref_files, "reference")

            count_ok = len(depth_files) == len(ref_files)
            count_msg = f"depth={len(depth_files)} vs reference={len(ref_files)}"

            duration_ok = False
            duration_msg = "n.d."
            if depth_stats["duration_available"] and ref_stats["duration_available"]:
                dd = abs(float(depth_stats["total_duration"]) - float(ref_stats["total_duration"]))
                duration_ok = dd <= 0.35
                duration_msg = (
                    f"depth={float(depth_stats['total_duration']):.3f}s vs "
                    f"reference={float(ref_stats['total_duration']):.3f}s (delta={dd:.3f}s)"
                )

            frames_ok = False
            frames_msg = "n.d."
            if depth_stats["frames_available"] and ref_stats["frames_available"]:
                df = abs(int(depth_stats["total_frames"]) - int(ref_stats["total_frames"]))
                frames_ok = df <= 1
                frames_msg = (
                    f"depth={int(depth_stats['total_frames'])} vs "
                    f"reference={int(ref_stats['total_frames'])} (delta={df})"
                )

            self._log_queue.put(("depth_line", f"[QUICK] file count check: {count_msg}"))
            self._log_queue.put(("depth_line", f"[QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("depth_line", f"[QUICK] packet check: {frames_msg}"))

            ok_final = (
                not depth_stats["broken"]
                and not ref_stats["broken"]
                and count_ok
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Depth quick verify completed.\n"
                f"Broken depth files: {len(depth_stats['broken'])}\n"
                f"Broken reference files: {len(ref_stats['broken'])}\n"
                f"File count match: {'YES' if count_ok else 'NO'} ({count_msg})\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(
                (
                    "depth_verify_quick_result",
                    {
                        "ok": ok_final,
                        "message": message,
                        "broken_depth": depth_stats["broken"],
                        "broken_reference": ref_stats["broken"],
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(("depth_verify_quick_result", {
                "ok": False,
                "message": f"Depth quick verify failed: {type(e).__name__}: {e}",
                "broken_depth": [],
                "broken_reference": [],
            }))
        finally:
            self._log_queue.put(("verify_done", "depth_quick"))

    def _start_depth_verify_deep(self) -> None:
        if self._depth_thread and self._depth_thread.is_alive():
            messagebox.showinfo("Verify Depth", "Stop DepthCrafter before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Depth", "Another verification is already running.")
            return
        ok, depth_dir, ref_dir = self._validate_depth_verify_inputs()
        if not ok:
            return

        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Depth", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(depth_dir).resolve()),
            str(Path(ref_dir).resolve()),
            "--extensions",
            ".mp4",
            "--workers",
            str(workers),
            "--probe-timeout-sec",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
            "--probe-timeout-retries",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
            "--delete",
            "yes",
            "--no-single-line-progress",
        ]

        self._set_verify_running(True, mode="depth_deep")
        self.depth_status_var.set("Verify (Deep) running...")
        self._append_depth_log("=== Verify Scenes (Deep) started ===")
        self._append_depth_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))

        self._verify_thread = threading.Thread(
            target=self._run_depth_verify_deep_worker,
            args=(cmd, str(Path(depth_dir).resolve())),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_depth_verify_deep_worker(self, cmd: list[str], depth_dir: str) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self._log_queue.put(("depth_line", f"[DEEP] {line}"))
                    bad_path = self._resolve_verifyscenes_bad_path(line, depth_dir)
                    if bad_path and bad_path not in seen_bad:
                        seen_bad.add(bad_path)
                        bad_files.append(bad_path)
            rc = proc.wait()
        except Exception as e:
            self._log_queue.put(("depth_line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "depth_verify_deep_result",
                    {"rc": rc, "depth_dir": depth_dir, "bad_files": bad_files},
                )
            )
            self._log_queue.put(("verify_done", "depth_deep"))

    def _validate_depth_upscaled_verify_inputs(self) -> tuple[bool, str, str]:
        upscaled_dir = self.depth_upscaled_var.get().strip()
        ref_dir = self.depth_input_var.get().strip()
        if not upscaled_dir:
            messagebox.showerror("Verify Upscale", "Depth upscaled folder is required.")
            return False, "", ""
        if not os.path.isdir(upscaled_dir):
            messagebox.showerror("Verify Upscale", f"Depth upscaled folder not found:\n{upscaled_dir}")
            return False, "", ""
        if not ref_dir:
            messagebox.showerror("Verify Upscale", "Reference scenes folder is required.")
            return False, "", ""
        if not os.path.isdir(ref_dir):
            messagebox.showerror("Verify Upscale", f"Reference scenes folder not found:\n{ref_dir}")
            return False, "", ""
        upscaled_dir = self._pipeline_prepare_verify_subset_dir(
            upscaled_dir, "depth_upscaled_target", ["*.mp4"]
        )
        ref_dir = self._pipeline_prepare_verify_subset_dir(
            ref_dir, "depth_upscaled_reference", ["*.mp4"]
        )
        return True, upscaled_dir, ref_dir

    def _start_depth_upscaled_verify_quick(self) -> None:
        if self._depth_thread and self._depth_thread.is_alive():
            messagebox.showinfo("Verify Upscale", "Stop depth tasks before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Upscale", "Another verification is already running.")
            return
        ok, upscaled_dir, ref_dir = self._validate_depth_upscaled_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Upscale", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="depth_upscaled_quick")
        self.depth_status_var.set("Verify Upscale (Quick) running...")
        self._append_depth_log("=== Verify Upscale (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_depth_upscaled_verify_quick_worker,
            args=(upscaled_dir, ref_dir),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_depth_upscaled_verify_quick_worker(self, upscaled_dir: str, ref_dir: str) -> None:
        try:
            upscaled_files = sorted([str(p) for p in Path(upscaled_dir).glob("*.mp4") if p.is_file()])
            ref_files = sorted([str(p) for p in Path(ref_dir).glob("*.mp4") if p.is_file()])
            if not upscaled_files:
                self._log_queue.put(("depth_upscaled_verify_quick_result", {
                    "ok": False,
                    "message": "No .mp4 files found in depth upscaled folder.",
                    "broken_upscaled": [],
                    "broken_reference": [],
                }))
                return
            if not ref_files:
                self._log_queue.put(("depth_upscaled_verify_quick_result", {
                    "ok": False,
                    "message": "No .mp4 files found in reference seg folder.",
                    "broken_upscaled": [],
                    "broken_reference": [],
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            self._log_queue.put(
                ("depth_line", f"[UPSCALE-QUICK] checking upscaled files={len(upscaled_files)} and reference files={len(ref_files)} with {max_workers} workers")
            )

            def _probe_group(file_list: list[str], label: str) -> dict:
                broken: list[str] = []
                total_duration = 0.0
                duration_available = True
                total_frames = 0
                frames_available = True

                def _probe_one(fp: str) -> tuple[str, dict]:
                    return fp, self._probe_video_basic(fp)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_probe_one, fp) for fp in file_list]
                    done = 0
                    for fut in concurrent.futures.as_completed(futures):
                        fp, meta = fut.result()
                        done += 1
                        if not meta.get("ok"):
                            broken.append(fp)
                            self._log_queue.put(("depth_line", f"[UPSCALE-QUICK][{label.upper()}][BROKEN] {fp} :: {meta.get('error')}"))
                        else:
                            dur = meta.get("duration")
                            frm = meta.get("frames")
                            if dur is None:
                                duration_available = False
                            else:
                                total_duration += float(dur)
                            if frm is None:
                                frames_available = False
                            else:
                                total_frames += int(frm)
                        if done % 25 == 0 or done == len(file_list):
                            self._log_queue.put(("depth_line", f"[UPSCALE-QUICK][{label.upper()}] progress {done}/{len(file_list)}"))
                return {
                    "broken": broken,
                    "total_duration": total_duration,
                    "duration_available": duration_available,
                    "total_frames": total_frames,
                    "frames_available": frames_available,
                }

            upscaled_stats = _probe_group(upscaled_files, "upscaled")
            ref_stats = _probe_group(ref_files, "reference")

            count_ok = len(upscaled_files) == len(ref_files)
            count_msg = f"upscaled={len(upscaled_files)} vs reference={len(ref_files)}"

            duration_ok = False
            duration_msg = "n.d."
            if upscaled_stats["duration_available"] and ref_stats["duration_available"]:
                dd = abs(float(upscaled_stats["total_duration"]) - float(ref_stats["total_duration"]))
                duration_ok = dd <= 0.35
                duration_msg = (
                    f"upscaled={float(upscaled_stats['total_duration']):.3f}s vs "
                    f"reference={float(ref_stats['total_duration']):.3f}s (delta={dd:.3f}s)"
                )

            frames_ok = False
            frames_msg = "n.d."
            if upscaled_stats["frames_available"] and ref_stats["frames_available"]:
                df = abs(int(upscaled_stats["total_frames"]) - int(ref_stats["total_frames"]))
                frames_ok = df <= 1
                frames_msg = (
                    f"upscaled={int(upscaled_stats['total_frames'])} vs "
                    f"reference={int(ref_stats['total_frames'])} (delta={df})"
                )

            self._log_queue.put(("depth_line", f"[UPSCALE-QUICK] file count check: {count_msg}"))
            self._log_queue.put(("depth_line", f"[UPSCALE-QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("depth_line", f"[UPSCALE-QUICK] packet check: {frames_msg}"))

            ok_final = (
                not upscaled_stats["broken"]
                and not ref_stats["broken"]
                and count_ok
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Upscale quick verify completed.\n"
                f"Broken upscaled files: {len(upscaled_stats['broken'])}\n"
                f"Broken reference files: {len(ref_stats['broken'])}\n"
                f"File count match: {'YES' if count_ok else 'NO'} ({count_msg})\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(
                (
                    "depth_upscaled_verify_quick_result",
                    {
                        "ok": ok_final,
                        "message": message,
                        "broken_upscaled": upscaled_stats["broken"],
                        "broken_reference": ref_stats["broken"],
                    },
                )
            )
        except Exception as e:
            self._log_queue.put(("depth_upscaled_verify_quick_result", {
                "ok": False,
                "message": f"Upscale quick verify failed: {type(e).__name__}: {e}",
                "broken_upscaled": [],
                "broken_reference": [],
            }))
        finally:
            self._log_queue.put(("verify_done", "depth_upscaled_quick"))

    def _start_depth_upscaled_verify_deep(self) -> None:
        if self._depth_thread and self._depth_thread.is_alive():
            messagebox.showinfo("Verify Upscale", "Stop depth tasks before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Upscale", "Another verification is already running.")
            return
        ok, upscaled_dir, ref_dir = self._validate_depth_upscaled_verify_inputs()
        if not ok:
            return

        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Upscale", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(upscaled_dir).resolve()),
            str(Path(ref_dir).resolve()),
            "--extensions",
            ".mp4",
            "--workers",
            str(workers),
            "--probe-timeout-sec",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
            "--probe-timeout-retries",
            str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
            "--delete",
            "yes",
            "--no-single-line-progress",
        ]

        self._set_verify_running(True, mode="depth_upscaled_deep")
        self.depth_status_var.set("Verify Upscale (Deep) running...")
        self._append_depth_log("=== Verify Upscale (Deep) started ===")
        self._append_depth_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))

        self._verify_thread = threading.Thread(
            target=self._run_depth_upscaled_verify_deep_worker,
            args=(cmd, str(Path(upscaled_dir).resolve())),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_depth_upscaled_verify_deep_worker(self, cmd: list[str], upscaled_dir: str) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self._log_queue.put(("depth_line", f"[UPSCALE-DEEP] {line}"))
                    bad_path = self._resolve_verifyscenes_bad_path(line, upscaled_dir)
                    if bad_path and bad_path not in seen_bad:
                        seen_bad.add(bad_path)
                        bad_files.append(bad_path)
            rc = proc.wait()
        except Exception as e:
            self._log_queue.put(("depth_line", f"[UPSCALE-DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "depth_upscaled_verify_deep_result",
                    {"rc": rc, "upscaled_dir": upscaled_dir, "bad_files": bad_files},
                )
            )
            self._log_queue.put(("verify_done", "depth_upscaled_deep"))

    def _browse_scene_input(self) -> None:
        start_dir = os.path.dirname(self.scene_input_var.get().strip()) or "."
        selected = filedialog.askopenfilename(
            title="Select input video",
            initialdir=start_dir,
            filetypes=[("Video files", "*.mkv *.mp4 *.mov *.avi *.webm"), ("All files", "*.*")],
        )
        if selected:
            self.scene_input_var.set(selected)
            self.scene_analysis_status_var.set("Ready")
            self._clear_source_analysis_data()
            self._preview_scene_command()
            self._start_source_analysis(silent=True)

    def _browse_work_folder(self) -> None:
        start_dir = self.work_folder_var.get().strip() or "."
        selected = filedialog.askdirectory(title="Select work folder", initialdir=start_dir)
        if selected:
            self.work_folder_var.set(selected)
            self._refresh_standard_paths()
            self._load_pipeline_state()
            self._preview_scene_command()

    def _open_scene_output_folder(self) -> None:
        folder = self.scene_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_scene_log(f"Scene output folder ready: {folder}")

    def _refresh_standard_paths(self) -> None:
        work_dir = self.work_folder_var.get().strip() or "./work"
        scene_out = os.path.join(work_dir, self.STANDARD_SUBDIRS["scenes"])
        self.scene_output_var.set(os.path.normpath(scene_out))
        depth_in = os.path.join(work_dir, self.STANDARD_SUBDIRS["scenes"])
        depth_out = os.path.join(work_dir, self.STANDARD_SUBDIRS["depth"])
        depth_upscaled = os.path.join(depth_out, "upscaled")
        self.depth_input_var.set(os.path.normpath(depth_in))
        self.depth_output_var.set(os.path.normpath(depth_out))
        self.depth_upscaled_var.set(os.path.normpath(depth_upscaled))
        self.splat_input_clips_var.set(os.path.normpath(scene_out))
        self.splat_input_depth_var.set(os.path.normpath(depth_upscaled))
        self.splat_output_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["splat"]))
        )
        self.splat_mask_output_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["mask"]))
        )
        self.inpaint_input_var.set(
            os.path.normpath(
                os.path.join(work_dir, self.STANDARD_SUBDIRS["splat"], "hires")
            )
        )
        self.inpaint_mask_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["mask"]))
        )
        self.inpaint_output_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["inpaint"]))
        )
        self.inpaint_sharpness_csv_var.set(
            os.path.normpath(os.path.join(work_dir, "sharpness.csv"))
        )
        self.merge_inpainted_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["inpaint"]))
        )
        self.merge_splatted_var.set(
            os.path.normpath(
                os.path.join(work_dir, self.STANDARD_SUBDIRS["splat"], "hires")
            )
        )
        self.merge_original_var.set(os.path.normpath(scene_out))
        self.merge_replace_mask_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["mask"]))
        )
        self.merge_mask_formerge_var.set(
            os.path.normpath(
                os.path.join(work_dir, self.STANDARD_SUBDIRS["mask_for_merge"])
            )
        )
        self.merge_output_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["merge"]))
        )
        self.merge_autoct_csv_var.set(
            os.path.normpath(os.path.join(work_dir, "autoct.csv"))
        )
        self.join_input_var.set(
            os.path.normpath(os.path.join(work_dir, self.STANDARD_SUBDIRS["merge"]))
        )
        self.join_seg_mono_var.set(
            os.path.normpath(os.path.join(work_dir, "seg-mono"))
        )
        self.join_output_var.set(
            os.path.normpath(
                os.path.join(
                    work_dir,
                    self.STANDARD_SUBDIRS["join"],
                    "final_sbs_1080_hevc_nvenc.mp4",
                )
            )
        )
        self._preview_depth_command()
        self._preview_splat_command()
        self._preview_inpaint_command()
        self._preview_merge_command()
        self._preview_join_command()
        self._update_replace_mask_dependent_controls()
        self._refresh_pipeline_status_panel()

    def _clear_source_analysis_data(self) -> None:
        self._source_video_info = {}
        self._source_capabilities = {}
        self._recommended_crop_filters = {}
        self._crop_recommendation_profile = {}
        self._scene_crop_target_syncing = True
        try:
            self.scene_crop_target_h_var.set("")
        finally:
            self._scene_crop_target_syncing = False
        self.scene_crop_auto_desc_var.set("n.d.")
        self.scene_crop_tile_compat_var.set("n.d.")
        self.analysis_source_path_var.set("n.d.")
        self.analysis_resolution_var.set("n.d.")
        self.analysis_bars_var.set("n.d.")
        self.analysis_color_var.set("n.d.")
        self.analysis_pixfmt_var.set("n.d.")
        self.analysis_length_var.set("n.d.")
        self.analysis_fps_var.set("n.d.")
        self.analysis_bitrate_var.set("n.d.")
        self._update_depth_resolution_preview()
        self._apply_option_states()

    def _on_detector_changed(self, _event=None) -> None:
        if self.scene_detector_var.get().strip().lower() == "content" and not self._content_notice_shown:
            self._content_notice_shown = True
            messagebox.showinfo("Detector Hint", self.CONTENT_THRESHOLD_NOTICE)
        self._preview_scene_command()

    def _on_backend_changed(self, _event=None) -> None:
        self._preview_scene_command()

    def _on_scene_crop_target_spin(self) -> None:
        if self.scene_crop_mode_var.get().strip().lower() == "auto":
            self.scene_crop_mode_var.set("manual")
        self._sync_auto_crop_from_target()
        self._refresh_crop_controls_state()

    def _on_scene_crop_target_changed(self, *_args) -> None:
        if self._scene_crop_target_syncing:
            return
        if self.scene_crop_mode_var.get().strip().lower() == "auto":
            self.scene_crop_mode_var.set("manual")
        self._sync_auto_crop_from_target()
        self._refresh_crop_controls_state()

    def _on_crop_mode_changed(self) -> None:
        mode = self.scene_crop_mode_var.get().strip().lower()
        if mode == "auto":
            if "auto" not in self._recommended_crop_filters:
                self.scene_crop_mode_var.set("manual")
            else:
                self._show_crop_bar_popup_once()
                prof = self._crop_recommendation_profile or {}
                default_target = prof.get("default_target_eff_h")
                if default_target:
                    self._scene_crop_target_syncing = True
                    try:
                        self.scene_crop_target_h_var.set(str(int(default_target)))
                    finally:
                        self._scene_crop_target_syncing = False
                self._sync_auto_crop_from_target(preview=False)
        elif mode != "manual":
            self.scene_crop_mode_var.set("manual")
        self._refresh_crop_controls_state()
        self._preview_scene_command()

    def _on_layout_changed(self, _event=None) -> None:
        # Fast layout can favor 422 when available.
        if "Half-SBS early" in self.scene_layout_var.get().strip():
            allowed = set(
                self._source_capabilities.get(
                    "allowed_chroma",
                    {"444", "422", "420"},
                )
            )
            if "422" in allowed:
                self.scene_chroma_var.set("422")
                self.scene_pix_fmt_var.set(self._chroma_to_pixfmt("422"))
        self._preview_scene_command()

    def _on_tonemap_changed(self, _event=None) -> None:
        self._preview_scene_command()

    def _on_chroma_changed(self) -> None:
        self.scene_pix_fmt_var.set(self._chroma_to_pixfmt(self.scene_chroma_var.get().strip()))
        self._preview_scene_command()

    def _refresh_crop_controls_state(self) -> None:
        has_profile = bool(self._crop_recommendation_profile)
        self.scene_crop_target_spin.configure(state=tk.NORMAL if has_profile else tk.DISABLED)

    def _show_crop_bar_popup_once(self) -> None:
        if self._crop_notice_shown:
            return
        self._crop_notice_shown = True
        messagebox.showinfo(
            "Crop Recommendation",
            (
                "Recommendation: better to lose a few rows than keep black bars.\n\n"
                "A bar-free frame increases the chance of cleaner side-border inpainting."
            ),
        )

    @staticmethod
    def _backend_to_cli(backend_ui: str) -> str:
        low = (backend_ui or "").strip().lower()
        if low == "moviepy":
            return "moviepy"
        if low == "opencv":
            return "opencv"
        return "opencv"

    @staticmethod
    def _chroma_to_pixfmt(chroma: str) -> str:
        if chroma == "444":
            return "yuv444p"
        if chroma == "422":
            return "yuv422p"
        return "yuv420p"

    @staticmethod
    def _pixfmt_to_chroma(pix_fmt: str) -> str:
        low = (pix_fmt or "").lower()
        if "444" in low:
            return "444"
        if "422" in low:
            return "422"
        return "420"

    def _compute_allowed_chroma_set(self, info: dict) -> set[str]:
        width = info.get("width")
        height = info.get("height")
        source_chroma = str(info.get("chroma") or "").strip()
        is_4k_or_more = bool(
            (isinstance(width, int) and width >= 3840)
            or (isinstance(height, int) and height >= 2160)
        )
        if is_4k_or_more:
            return {"444", "422", "420"}
        if source_chroma == "444":
            return {"444", "422", "420"}
        if source_chroma == "422":
            return {"422", "420"}
        return {"420"}

    def _apply_option_states(self) -> None:
        caps = self._source_capabilities or {}
        has_analysis = bool(caps)
        is_hdr = bool(caps.get("is_hdr", False))
        allowed = set(caps.get("allowed_chroma", {"444", "422", "420"}))

        # Tonemap only for HDR source.
        if has_analysis and is_hdr:
            self.scene_tonemap_combo.configure(state="readonly")
            self.hdr_policy_var.set("HDR -> SDR 8-bit BT.709: forced for this source")
        elif has_analysis:
            self.scene_tonemap_combo.configure(state=tk.DISABLED)
            self.hdr_policy_var.set("HDR -> SDR 8-bit BT.709: disabled (source is SDR/non-HDR)")
        else:
            self.scene_tonemap_combo.configure(state=tk.DISABLED)
            self.hdr_policy_var.set("HDR -> SDR 8-bit BT.709: waiting for source analysis")

        # Chroma options always visible, disabled by capability rules.
        self.chroma_444_rb.configure(state=tk.NORMAL if "444" in allowed else tk.DISABLED)
        self.chroma_422_rb.configure(state=tk.NORMAL if "422" in allowed else tk.DISABLED)
        self.chroma_420_rb.configure(state=tk.NORMAL if "420" in allowed else tk.DISABLED)
        current = self.scene_chroma_var.get().strip()
        if current not in allowed:
            if "420" in allowed:
                current = "420"
            elif "422" in allowed:
                current = "422"
            else:
                current = "444"
            self.scene_chroma_var.set(current)
        if has_analysis:
            self.scene_pix_fmt_var.set(self._chroma_to_pixfmt(self.scene_chroma_var.get().strip()))
        elif not self.scene_pix_fmt_var.get().strip():
            self.scene_pix_fmt_var.set(self._chroma_to_pixfmt(self.scene_chroma_var.get().strip()))

        # Crop mode availability based on analysis.
        if has_analysis:
            has_auto = "auto" in self._recommended_crop_filters
            self.crop_mode_auto_toggle.configure(state=tk.NORMAL if has_auto else tk.DISABLED)
            if self.scene_crop_mode_var.get().strip().lower() == "auto" and not has_auto:
                self.scene_crop_mode_var.set("manual")
        else:
            # Keep saved crop choices available until source analysis is refreshed.
            self.crop_mode_auto_toggle.configure(state=tk.NORMAL)
        self._refresh_crop_controls_state()

        # User guidance line.
        source_tag = caps.get("source_tag", "n.d.")
        hint = [f"Source class: {source_tag}"]
        if not has_analysis:
            hint.append("Analyze Source Video to apply source-driven crop/chroma limits.")
        elif is_hdr:
            hint.append("HDR input: 8-bit BT.709 conversion will be applied automatically.")
        else:
            hint.append("SDR input: no forced tonemap chain.")
        if allowed == {"420"}:
            hint.append("Chroma capped at 420 for this source class.")
        elif allowed == {"422", "420"}:
            hint.append("Chroma capped at 422/420 for this source class.")
        else:
            hint.append("Chroma 444/422/420 available.")
        self.scene_option_hint_var.set(" | ".join(hint))

    def _update_source_capabilities(self, info: dict) -> None:
        dynamic_range = str(info.get("dynamic_range") or "").upper()
        is_hdr = "HDR" in dynamic_range
        allowed_chroma = self._compute_allowed_chroma_set(info)
        width = info.get("width")
        height = info.get("height")
        source_tag = "n.d."
        if isinstance(width, int) and isinstance(height, int):
            if width >= 3840 or height >= 2160:
                source_tag = "4K+"
            elif width >= 1920 or height >= 1080:
                source_tag = "FHD-class"
            else:
                source_tag = "sub-FHD"
        self._source_capabilities = {
            "is_hdr": is_hdr,
            "allowed_chroma": allowed_chroma,
            "source_tag": source_tag,
        }
        if is_hdr and not self._hdr_notice_shown:
            self._hdr_notice_shown = True
            messagebox.showinfo("HDR Source Policy", self.HDR_FORCE_NOTICE)
        self._apply_option_states()

    @staticmethod
    def _is_tile_height_compatible(height: int, tile_num: int = 2, tile_overlap: int = 128) -> bool:
        if height <= 0:
            return False
        if tile_num <= 1:
            return (height % 8) == 0
        tile_h = (height + tile_overlap * (tile_num - 1)) // tile_num
        return (tile_h % 8) == 0

    @staticmethod
    def _floor_even(value: float) -> int:
        iv = int(value)
        if iv % 2:
            iv -= 1
        return max(2, iv)

    @staticmethod
    def _align_down_step8(value: int) -> int:
        iv = int(value)
        if iv < 0:
            return 0
        return iv - (iv % 8)

    def _compatible_tiles_for_effective_height(self, effective_height: int, max_tile: int = 4) -> list[int]:
        out: list[int] = []
        if effective_height <= 0:
            return out
        for tile in range(1, max(1, int(max_tile)) + 1):
            if self._is_tile_height_compatible(effective_height, tile_num=tile, tile_overlap=128):
                out.append(tile)
        return out

    def _format_tile_compatibility_label(self, effective_height: int) -> str:
        compatible = self._compatible_tiles_for_effective_height(effective_height, max_tile=4)
        suffix = (
            " | extra pad will be removed in the final stage."
            if int(effective_height) > 1080
            else ""
        )
        if compatible:
            return f"Compatible tiles: {', '.join(str(x) for x in compatible)}{suffix}"
        return f"Compatible tiles: none (1..4){suffix}"

    def _normalize_scene_crop_target_effective(
        self,
        raw_value: str,
        profile: dict,
        fallback: int | None = None,
    ) -> int | None:
        max_target = int(profile.get("max_target_eff_h") or 0)
        min_target = int(profile.get("min_target_eff_h") or 8)
        if max_target < min_target:
            return None
        fallback_val = int(
            fallback
            if fallback is not None
            else (profile.get("default_target_eff_h") or max_target)
        )
        try:
            parsed = int(str(raw_value).strip())
        except Exception:
            parsed = fallback_val
        parsed = max(min_target, min(max_target, parsed))
        parsed = self._align_down_step8(parsed)
        if parsed < min_target:
            parsed = min_target
        if parsed > max_target:
            parsed = max_target
        if parsed <= 0:
            return None
        return parsed

    def _build_auto_crop_recommendation_from_profile(
        self, profile: dict, target_eff_h: int
    ) -> dict | None:
        source_height = int(profile.get("source_height") or 0)
        effective_source_h = int(profile.get("effective_source_h") or 0)
        clean_h = int(profile.get("clean_h") or 0)
        bars_top_eff = int(profile.get("bars_top_eff") or 0)
        bars_bottom_eff = int(profile.get("bars_bottom_eff") or 0)
        scale_ratio = float(profile.get("scale_ratio") or 1.0)
        if source_height <= 0 or effective_source_h <= 0 or clean_h <= 0 or scale_ratio <= 0.0:
            return None

        min_target = int(profile.get("min_target_eff_h") or 8)
        max_target = int(profile.get("max_target_eff_h") or 0)
        target_eff_h = max(min_target, min(max_target, int(target_eff_h)))
        target_eff_h = self._align_down_step8(target_eff_h)
        if target_eff_h <= 0:
            return None

        # Always remove detected bars first (clean_h), then crop/pad around clean content.
        extra_crop_eff = max(0, clean_h - target_eff_h)
        pad_eff = max(0, target_eff_h - clean_h)

        extra_crop_top_eff = (extra_crop_eff + 1) // 2
        extra_crop_bottom_eff = extra_crop_eff // 2

        crop_top_eff = bars_top_eff + extra_crop_top_eff
        crop_bottom_eff = bars_bottom_eff + extra_crop_bottom_eff
        crop_h_eff = max(1, effective_source_h - crop_top_eff - crop_bottom_eff)

        crop_top = int(round(crop_top_eff / scale_ratio))
        crop_bottom = int(round(crop_bottom_eff / scale_ratio))
        crop_h = source_height - crop_top - crop_bottom
        if crop_h <= 0:
            return None

        crop_top = max(0, min(crop_top, source_height - 1))
        crop_bottom = max(0, min(crop_bottom, source_height - crop_top - 1))
        crop_h = max(1, source_height - crop_top - crop_bottom)
        if crop_h % 2:
            crop_h -= 1
        if crop_h <= 0:
            return None

        pad_top_eff = (pad_eff + 1) // 2
        pad_bottom_eff = pad_eff // 2
        pad_top = max(0, int(round(pad_top_eff / scale_ratio)))
        pad_bottom = max(0, int(round(pad_bottom_eff / scale_ratio)))

        parts: list[str] = []
        if crop_top > 0 or crop_h != source_height:
            parts.append(f"crop=iw:{crop_h}:0:{crop_top}")
        if pad_top > 0 or pad_bottom > 0:
            pad_out_h = crop_h + pad_top + pad_bottom
            parts.append(f"pad=iw:{pad_out_h}:0:{pad_top}:black")

        filter_expr = ",".join(parts)
        # Keep auto mode valid even if no crop/pad is required for this target.
        # Empty filter means "no-op" for SceneDetect ffmpeg -vf chain.
        if not filter_expr:
            filter_expr = ""

        desc = (
            f"final {target_eff_h}px (clean={clean_h}px, "
            f"content crop={extra_crop_eff}px, pad={pad_eff}px)"
        )
        return {
            "filter": filter_expr,
            "desc": desc,
            "target_eff_h": target_eff_h,
            "pad_top_src": pad_top,
            "pad_bottom_src": pad_bottom,
        }

    def _sync_auto_crop_from_target(self, preview: bool = True) -> None:
        profile = self._crop_recommendation_profile or {}
        if not profile:
            self._recommended_crop_filters.pop("auto", None)
            self.scene_crop_auto_desc_var.set("n.d.")
            self.scene_crop_tile_compat_var.set("n.d.")
            if preview:
                self._preview_scene_command()
            return

        raw = self.scene_crop_target_h_var.get().strip()
        target_eff_h = self._normalize_scene_crop_target_effective(raw, profile)
        if target_eff_h is None:
            self._recommended_crop_filters.pop("auto", None)
            self.scene_crop_auto_desc_var.set("n.d.")
            self.scene_crop_tile_compat_var.set("n.d.")
            if preview:
                self._preview_scene_command()
            return

        target_txt = str(target_eff_h)
        if raw != target_txt:
            self._scene_crop_target_syncing = True
            try:
                self.scene_crop_target_h_var.set(target_txt)
            finally:
                self._scene_crop_target_syncing = False

        auto_rec = self._build_auto_crop_recommendation_from_profile(profile, target_eff_h)
        if auto_rec:
            self._recommended_crop_filters["auto"] = auto_rec
            self.scene_crop_auto_desc_var.set(auto_rec["desc"])
            self.scene_crop_tile_compat_var.set(
                self._format_tile_compatibility_label(target_eff_h)
            )
        else:
            self._recommended_crop_filters.pop("auto", None)
            self.scene_crop_auto_desc_var.set("n.d.")
            self.scene_crop_tile_compat_var.set("n.d.")

        self._refresh_crop_controls_state()
        self._update_depth_resolution_preview()
        if preview:
            self._preview_scene_command()

    def _build_crop_recommendations(
        self,
        source_width: int | None,
        source_height: int | None,
        bars_top: int | None,
        bars_bottom: int | None,
    ) -> dict:
        self._crop_recommendation_profile = {}
        out: dict = {}
        if not isinstance(source_width, int):
            return out
        if not isinstance(source_height, int):
            return out
        if not isinstance(bars_top, int) or not isinstance(bars_bottom, int):
            return out
        if source_width <= 0 or source_height <= 0 or bars_top < 0 or bars_bottom < 0:
            return out

        # Build recommendation on effective post-scale height (4K -> 1080 logic),
        # then map back to source-space crop values used by ffmpeg crop filter.
        downscale_to_1080 = source_width > 1920 or source_height > 1080
        scale_ratio = (1920.0 / float(source_width)) if downscale_to_1080 else 1.0
        scale_ratio = max(0.001, min(1.0, scale_ratio))

        effective_source_h = self._floor_even(source_height * scale_ratio)
        bars_top_eff = int(round(bars_top * scale_ratio))
        bars_bottom_eff = int(round(bars_bottom * scale_ratio))
        bars_top_eff = max(0, min(bars_top_eff, effective_source_h))
        bars_bottom_eff = max(0, min(bars_bottom_eff, effective_source_h - bars_top_eff))

        base_h = effective_source_h - bars_top_eff - bars_bottom_eff
        if base_h <= 0:
            return out

        max_target_eff_h = self._align_down_step8(base_h + 256)
        if max_target_eff_h < 8:
            return out

        start_h = min(base_h, max_target_eff_h)
        start_h = self._align_down_step8(start_h)
        target_eff_h = None
        for h in range(start_h, 7, -8):
            if self._is_tile_height_compatible(h, tile_num=2, tile_overlap=128):
                target_eff_h = h
                break
        if target_eff_h is None:
            target_eff_h = max_target_eff_h

        self._crop_recommendation_profile = {
            "source_height": source_height,
            "effective_source_h": effective_source_h,
            "clean_h": base_h,
            "bars_top_eff": bars_top_eff,
            "bars_bottom_eff": bars_bottom_eff,
            "scale_ratio": scale_ratio,
            "min_target_eff_h": 8,
            "max_target_eff_h": max_target_eff_h,
            "default_target_eff_h": target_eff_h,
        }

        auto_rec = self._build_auto_crop_recommendation_from_profile(
            self._crop_recommendation_profile, target_eff_h
        )
        if auto_rec:
            out["auto"] = auto_rec
        return out

    def _update_crop_recommendations(self, info: dict) -> None:
        self._recommended_crop_filters = self._build_crop_recommendations(
            source_width=info.get("width"),
            source_height=info.get("height"),
            bars_top=info.get("bars_top"),
            bars_bottom=info.get("bars_bottom"),
        )
        profile = self._crop_recommendation_profile or {}
        default_target = profile.get("default_target_eff_h")
        if default_target:
            current_raw = self.scene_crop_target_h_var.get().strip()
            mode = self.scene_crop_mode_var.get().strip().lower()
            should_apply_auto = (mode == "auto") or (current_raw == "")
            if should_apply_auto:
                self._scene_crop_target_syncing = True
                try:
                    self.scene_crop_target_h_var.set(str(int(default_target)))
                finally:
                    self._scene_crop_target_syncing = False
            self._sync_auto_crop_from_target(preview=False)
        else:
            self._scene_crop_target_syncing = True
            try:
                self.scene_crop_target_h_var.set("")
            finally:
                self._scene_crop_target_syncing = False
            self.scene_crop_auto_desc_var.set("n.d.")
            self.scene_crop_tile_compat_var.set("n.d.")
        self._apply_option_states()

    def _get_crop_filter(self) -> str:
        mode = self.scene_crop_mode_var.get().strip().lower()
        if mode in {"auto", "manual"}:
            rec = self._recommended_crop_filters.get("auto")
            return rec["filter"] if rec else ""
        return ""

    def _needs_downscale_to_1080(self) -> bool:
        width = self._source_video_info.get("width")
        height = self._source_video_info.get("height")
        return bool(
            isinstance(width, int)
            and isinstance(height, int)
            and (width > 1920 or height > 1080)
        )

    def _build_ffmpeg_vf_filters(self) -> list[str]:
        filters: list[str] = []
        crop_filter = self._get_crop_filter()
        if crop_filter:
            filters.append(crop_filter)

        is_hdr = bool(self._source_capabilities.get("is_hdr", False))
        pix_fmt = self.scene_pix_fmt_var.get().strip() or "yuv420p"
        tonemap_display = self.scene_tonemap_var.get().strip()
        tonemap = self.TONEMAP_PRESET_TO_FFMPEG.get(tonemap_display, "mobius")

        if is_hdr:
            filters.extend(
                [
                    "zscale=transferin=smpte2084:primariesin=bt2020:matrixin=bt2020nc:rangein=tv",
                    "zscale=transfer=linear:npl=100",
                    f"tonemap={tonemap}:desat=0",
                    "zscale=transfer=bt709:primaries=bt709:matrix=bt709:range=tv",
                ]
            )

        if self._needs_downscale_to_1080():
            filters.append("scale=1920:-2:flags=lanczos+accurate_rnd+full_chroma_int")

        if is_hdr:
            filters.append(f"format={pix_fmt}")
        return filters

    def _build_scene_split_ffmpeg_tokens(self) -> list[str]:
        scene_codec = self._normalize_ffmpeg_codec(
            self.scene_codec_var.get(),
            self.DEFAULT_SCENE_CODEC,
        )
        self.scene_codec_var.set(scene_codec)
        ffmpeg_tokens = [
            "-map",
            "0:v:0",
            "-an",
            "-dn",
            "-sn",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
        ]

        vf_filters = self._build_ffmpeg_vf_filters()
        if vf_filters:
            ffmpeg_tokens += ["-vf", ",".join(vf_filters)]

        ffmpeg_tokens += [
            "-c:v",
            scene_codec,
            "-" + self._quality_flag_for_codec(scene_codec, self.DEFAULT_SCENE_CODEC, "qp"),
            self.scene_crf_var.get().strip() or "1",
            "-preset",
            self.scene_encoder_preset_var.get().strip() or "fast",
            "-pix_fmt",
            self.scene_pix_fmt_var.get().strip() or "yuv420p",
            "-b:v",
            "0",
        ]

        extra_ffmpeg = self.scene_extra_ffmpeg_args_var.get().strip()
        if extra_ffmpeg:
            ffmpeg_tokens.extend(shlex.split(extra_ffmpeg))
        return ffmpeg_tokens

    def _build_scenedetect_command(self) -> list[str]:
        input_path = self.scene_input_var.get().strip()
        detector_ui = self.scene_detector_var.get().strip().lower()
        threshold = self.scene_threshold_var.get().strip()
        backend = self._backend_to_cli(self.scene_backend_var.get().strip())
        scene_csv_path = Path(self._scene_csv_path()).resolve()
        detect_cmd = "detect-content" if detector_ui == "content" else "detect-adaptive"
        return [
            "scenedetect",
            "-i",
            input_path,
            "-b",
            backend,
            detect_cmd,
            "-t",
            threshold,
            "list-scenes",
            "-o",
            str(scene_csv_path.parent),
            "-f",
            scene_csv_path.name,
            "--skip-cuts",
        ]

    def _build_split_scenes_command(self) -> list[str]:
        input_path = self.scene_input_var.get().strip()
        output_path = self.scene_output_var.get().strip()
        scene_csv_path = str(Path(self._scene_csv_path()).resolve())
        workers = self._get_scene_split_workers()
        ffmpeg_tokens = self._build_scene_split_ffmpeg_tokens()
        script_path = Path("Utilities/split_scenes_from_csv.py").resolve()
        return [
            sys.executable,
            str(script_path),
            "--input-video",
            input_path,
            "--scene-csv",
            scene_csv_path,
            "--output-dir",
            output_path,
            "--threads",
            str(workers),
            "--ffmpeg-args",
            " ".join(ffmpeg_tokens),
            "--skip-existing",
            "yes",
            "--delete-failed",
            "yes",
        ]

    def _preview_scene_command(self) -> None:
        try:
            cmd = self._build_scenedetect_command()
            self.scene_cmd_preview_var.set(" ".join(shlex.quote(x) for x in cmd))
        except Exception as exc:
            self.scene_cmd_preview_var.set(f"Invalid options: {exc}")

    def _validate_scene_form(self, require_scenedetect: bool = True) -> bool:
        input_path = self.scene_input_var.get().strip()
        output_path = self.scene_output_var.get().strip()
        if not input_path:
            messagebox.showerror("SceneDetect", "Source video path is required.")
            return False
        if not os.path.isfile(input_path):
            messagebox.showerror("SceneDetect", f"Source video not found:\n{input_path}")
            return False
        if not output_path:
            messagebox.showerror("SceneDetect", "Output folder is required.")
            return False
        try:
            _ = float(self.scene_threshold_var.get().strip())
        except Exception:
            messagebox.showerror("SceneDetect", "Threshold must be a valid number.")
            return False

        if require_scenedetect and shutil.which("scenedetect") is None:
            messagebox.showerror("SceneDetect", "scenedetect command not found in PATH.")
            return False
        return True

    def _set_scene_running(self, is_running: bool) -> None:
        self.scene_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.scene_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.scene_split_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        stop_enabled = bool(is_running or self._analysis_running)
        self.scene_stop_btn.configure(state=tk.NORMAL if stop_enabled else tk.DISABLED)
        if is_running:
            self.scene_verify_quick_btn.configure(state=tk.DISABLED)
            self.scene_verify_deep_btn.configure(state=tk.DISABLED)
        elif not self._verify_running and not self._analysis_running:
            self.scene_verify_quick_btn.configure(state=tk.NORMAL)
            self.scene_verify_deep_btn.configure(state=tk.NORMAL)

    def _set_analysis_running(self, is_running: bool) -> None:
        self._analysis_running = is_running
        self.scene_analyze_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        scene_is_running = bool(self._scene_thread and self._scene_thread.is_alive())
        self.scene_stop_btn.configure(state=tk.NORMAL if (is_running or scene_is_running) else tk.DISABLED)
        if is_running:
            self.scene_verify_quick_btn.configure(state=tk.DISABLED)
            self.scene_verify_deep_btn.configure(state=tk.DISABLED)
        elif (not scene_is_running) and (not self._verify_running):
            self.scene_verify_quick_btn.configure(state=tk.NORMAL)
            self.scene_verify_deep_btn.configure(state=tk.NORMAL)

    def _set_verify_running(self, is_running: bool, mode: str = "") -> None:
        self._verify_running = is_running
        self._verify_mode = mode if is_running else ""
        if is_running:
            self.scene_verify_quick_btn.configure(state=tk.DISABLED)
            self.scene_verify_deep_btn.configure(state=tk.DISABLED)
            self.depth_verify_quick_btn.configure(state=tk.DISABLED)
            self.depth_verify_deep_btn.configure(state=tk.DISABLED)
            self.depth_upscaled_verify_quick_btn.configure(state=tk.DISABLED)
            self.depth_upscaled_verify_deep_btn.configure(state=tk.DISABLED)
            self.splat_verify_quick_btn.configure(state=tk.DISABLED)
            self.splat_verify_deep_btn.configure(state=tk.DISABLED)
            self.inpaint_verify_quick_btn.configure(state=tk.DISABLED)
            self.inpaint_verify_deep_btn.configure(state=tk.DISABLED)
            self.merge_mask_verify_quick_btn.configure(state=tk.DISABLED)
            self.merge_mask_verify_deep_btn.configure(state=tk.DISABLED)
            self.merge_verify_quick_btn.configure(state=tk.DISABLED)
            self.merge_verify_deep_btn.configure(state=tk.DISABLED)
            self.join_mono_verify_btn.configure(state=tk.DISABLED)
            self.join_verify_btn.configure(state=tk.DISABLED)
        else:
            scene_is_running = bool(self._scene_thread and self._scene_thread.is_alive())
            if scene_is_running:
                self.scene_verify_quick_btn.configure(state=tk.DISABLED)
                self.scene_verify_deep_btn.configure(state=tk.DISABLED)
            else:
                if self._analysis_running:
                    self.scene_verify_quick_btn.configure(state=tk.DISABLED)
                    self.scene_verify_deep_btn.configure(state=tk.DISABLED)
                else:
                    self.scene_verify_quick_btn.configure(state=tk.NORMAL)
                    self.scene_verify_deep_btn.configure(state=tk.NORMAL)
            depth_is_running = bool(self._depth_thread and self._depth_thread.is_alive())
            if depth_is_running:
                self.depth_verify_quick_btn.configure(state=tk.DISABLED)
                self.depth_verify_deep_btn.configure(state=tk.DISABLED)
                self.depth_upscaled_verify_quick_btn.configure(state=tk.DISABLED)
                self.depth_upscaled_verify_deep_btn.configure(state=tk.DISABLED)
            else:
                self.depth_verify_quick_btn.configure(state=tk.NORMAL)
                self.depth_verify_deep_btn.configure(state=tk.NORMAL)
                self.depth_upscaled_verify_quick_btn.configure(state=tk.NORMAL)
                self.depth_upscaled_verify_deep_btn.configure(state=tk.NORMAL)
            splat_is_running = bool(self._splat_thread and self._splat_thread.is_alive())
            if splat_is_running:
                self.splat_verify_quick_btn.configure(state=tk.DISABLED)
                self.splat_verify_deep_btn.configure(state=tk.DISABLED)
            else:
                self.splat_verify_quick_btn.configure(state=tk.NORMAL)
                self.splat_verify_deep_btn.configure(state=tk.NORMAL)
            inpaint_is_running = bool(self._inpaint_thread and self._inpaint_thread.is_alive())
            if inpaint_is_running:
                self.inpaint_verify_quick_btn.configure(state=tk.DISABLED)
                self.inpaint_verify_deep_btn.configure(state=tk.DISABLED)
            else:
                self.inpaint_verify_quick_btn.configure(state=tk.NORMAL)
                self.inpaint_verify_deep_btn.configure(state=tk.NORMAL)
            merge_is_running = bool(self._merge_thread and self._merge_thread.is_alive())
            if merge_is_running:
                self.merge_mask_verify_quick_btn.configure(state=tk.DISABLED)
                self.merge_mask_verify_deep_btn.configure(state=tk.DISABLED)
                self.merge_verify_quick_btn.configure(state=tk.DISABLED)
                self.merge_verify_deep_btn.configure(state=tk.DISABLED)
            else:
                self.merge_mask_verify_quick_btn.configure(state=tk.NORMAL)
                self.merge_mask_verify_deep_btn.configure(state=tk.NORMAL)
                self.merge_verify_quick_btn.configure(state=tk.NORMAL)
                self.merge_verify_deep_btn.configure(state=tk.NORMAL)
            join_is_running = bool(self._join_thread and self._join_thread.is_alive())
            if join_is_running:
                self.join_mono_verify_btn.configure(state=tk.DISABLED)
                self.join_verify_btn.configure(state=tk.DISABLED)
            else:
                self.join_mono_verify_btn.configure(state=tk.NORMAL)
                self.join_verify_btn.configure(state=tk.NORMAL)

    def _validate_verify_inputs(self) -> tuple[bool, str, str]:
        source_path = self.scene_input_var.get().strip()
        seg_dir = self.scene_output_var.get().strip()
        if not source_path:
            messagebox.showerror("Verify Scenes", "Source video path is required.")
            return False, "", ""
        if not os.path.isfile(source_path):
            messagebox.showerror("Verify Scenes", f"Source video not found:\n{source_path}")
            return False, "", ""
        if not seg_dir:
            messagebox.showerror("Verify Scenes", "Scene output folder is required.")
            return False, "", ""
        if not os.path.isdir(seg_dir):
            messagebox.showerror("Verify Scenes", f"Scene output folder not found:\n{seg_dir}")
            return False, "", ""
        return True, source_path, seg_dir

    def _scene_verify_target_dirs(self, seg_dir: str) -> list[str]:
        targets: list[str] = []
        try:
            seg_resolved = str(Path(seg_dir).resolve())
        except Exception:
            seg_resolved = str(seg_dir)
        if seg_resolved:
            targets.append(seg_resolved)
        if self._pipeline_test_active:
            return targets

        # seg-mono is always managed under the active work folder.
        try:
            seg_mono_resolved = str(
                (Path(self.work_folder_var.get().strip() or "./work").resolve() / "seg-mono")
            )
        except Exception:
            seg_mono_resolved = ""
        if (
            seg_mono_resolved
            and os.path.isdir(seg_mono_resolved)
            and os.path.normpath(seg_mono_resolved) != os.path.normpath(seg_resolved)
        ):
            mono_files = self._collect_files_for_patterns(
                seg_mono_resolved, self._scene_quick_verify_patterns()
            )
            if mono_files:
                targets.append(seg_mono_resolved)
        return targets

    def _scene_quick_verify_patterns(self) -> list[str]:
        patterns: list[str] = []
        for ext_raw in str(self.VERIFY_ALL_VIDEO_EXTENSIONS).split(","):
            ext = str(ext_raw or "").strip()
            if not ext:
                continue
            low_pat = f"*{ext.lower()}"
            up_pat = f"*{ext.upper()}"
            if low_pat not in patterns:
                patterns.append(low_pat)
            if up_pat not in patterns:
                patterns.append(up_pat)
        if not patterns:
            return ["*.mp4", "*.MP4"]
        return patterns

    @staticmethod
    def _run_ffprobe_watchdog(cmd: list[str], timeout_sec: float) -> tuple[int, str, str, bool]:
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                out_txt, err_txt = proc.communicate(
                    timeout=(float(timeout_sec) if timeout_sec and float(timeout_sec) > 0 else None)
                )
                rc = int(proc.returncode or 0)
                return rc, (out_txt or "").strip(), (err_txt or "").strip(), False
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    out_txt, err_txt = proc.communicate(timeout=2)
                except Exception:
                    out_txt, err_txt = "", ""
                return 124, (out_txt or "").strip(), (err_txt or "").strip(), True
        except FileNotFoundError as e:
            return 127, "", str(e), False
        except Exception as e:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            return 126, "", str(e), False

    @staticmethod
    def _probe_video_basic(path: str) -> dict:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_packets,nb_read_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            watchdog_sec = float(
                os.environ.get(
                    "VERIFY_FFPROBE_WATCHDOG_SEC",
                    str(PipelineMasterGUI.VERIFY_QUICK_FFPROBE_TIMEOUT_SEC),
                )
            )
        except Exception:
            watchdog_sec = float(PipelineMasterGUI.VERIFY_QUICK_FFPROBE_TIMEOUT_SEC)
        if watchdog_sec < 0:
            watchdog_sec = 0.0

        try:
            timeout_retries = int(
                os.environ.get(
                    "VERIFY_FFPROBE_WATCHDOG_RETRIES",
                    str(PipelineMasterGUI.VERIFY_QUICK_FFPROBE_TIMEOUT_RETRIES),
                )
            )
        except Exception:
            timeout_retries = int(PipelineMasterGUI.VERIFY_QUICK_FFPROBE_TIMEOUT_RETRIES)
        timeout_retries = max(0, timeout_retries)
        total_attempts = 1 + timeout_retries

        last_rc = 126
        last_out = ""
        last_err = ""
        timeout_hits = 0
        for attempt in range(1, total_attempts + 1):
            rc, out_txt, err_txt, timed_out = PipelineMasterGUI._run_ffprobe_watchdog(cmd, watchdog_sec)
            last_rc, last_out, last_err = rc, out_txt, err_txt
            if timed_out:
                timeout_hits += 1
                if attempt < total_attempts:
                    continue
                return {
                    "ok": False,
                    "error": (
                        f"ffprobe watchdog timeout after {watchdog_sec:.1f}s "
                        f"(attempt {attempt}/{total_attempts}); "
                        f"file flagged as corrupted: {path}"
                    ),
                    "duration": None,
                    "frames": None,
                    "width": None,
                    "height": None,
                    "codec_name": "",
                    "pix_fmt": "",
                }
            if rc != 0:
                err_msg = (err_txt or out_txt or f"ffprobe rc={rc}").strip()
                if timeout_hits > 0:
                    err_msg = (
                        f"{err_msg} (after {timeout_hits} watchdog timeout "
                        f"{'retry' if timeout_hits == 1 else 'retries'})"
                    )
                return {
                    "ok": False,
                    "error": err_msg,
                    "duration": None,
                    "frames": None,
                    "width": None,
                    "height": None,
                    "codec_name": "",
                    "pix_fmt": "",
                }
            break
        else:
            return {
                "ok": False,
                "error": (last_err or last_out or f"ffprobe rc={last_rc}").strip(),
                "duration": None,
                "frames": None,
                "width": None,
                "height": None,
                "codec_name": "",
                "pix_fmt": "",
            }

        try:
            doc = json.loads(last_out or "{}")
            streams = doc.get("streams") or []
            if not streams:
                return {
                    "ok": False,
                    "error": "no video stream",
                    "duration": None,
                    "frames": None,
                    "width": None,
                    "height": None,
                    "codec_name": "",
                    "pix_fmt": "",
                }
            st = streams[0] or {}
            dur = None
            if st.get("duration") not in (None, "", "N/A"):
                try:
                    dur = float(st.get("duration"))
                except Exception:
                    dur = None
            if dur is None:
                fmt = (doc.get("format") or {}).get("duration")
                if fmt not in (None, "", "N/A"):
                    try:
                        dur = float(fmt)
                    except Exception:
                        dur = None
            frames = None
            # Quick verify uses packet counting for speed. Keep the return key name
            # as "frames" so existing quick-verify comparisons remain unchanged.
            nbf = st.get("nb_read_packets")
            if nbf in (None, "", "N/A"):
                nbf = st.get("nb_read_frames")
            if nbf not in (None, "", "N/A"):
                try:
                    frames = int(float(nbf))
                except Exception:
                    frames = None
            width = None
            height = None
            try:
                if st.get("width") not in (None, "", "N/A"):
                    width = int(float(st.get("width")))
            except Exception:
                width = None
            try:
                if st.get("height") not in (None, "", "N/A"):
                    height = int(float(st.get("height")))
            except Exception:
                height = None
            return {
                "ok": True,
                "error": "",
                "duration": dur,
                "frames": frames,
                "width": width,
                "height": height,
                "codec_name": str(st.get("codec_name") or ""),
                "pix_fmt": str(st.get("pix_fmt") or ""),
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"invalid ffprobe json: {e}",
                "duration": None,
                "frames": None,
                "width": None,
                "height": None,
                "codec_name": "",
                "pix_fmt": "",
            }

    def _start_verify_quick(self) -> None:
        if self._scene_thread and self._scene_thread.is_alive():
            messagebox.showinfo(
                "Verify Scenes",
                "Stop SceneDetect/Split Scenes before running verification.",
            )
            return
        if self._verify_running:
            messagebox.showinfo("Verify Scenes", "Another verification is already running.")
            return
        ok, source_path, seg_dir = self._validate_verify_inputs()
        if not ok:
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Scenes", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="quick")
        self._scene_verify_result_applied = False
        self.scene_status_var.set("Verify (Quick) running...")
        self._append_scene_log("=== Verify Scenes (Quick) started ===")
        target_dirs = self._scene_verify_target_dirs(seg_dir)
        self._append_scene_log(
            "Quick verify targets: " + ", ".join(target_dirs if target_dirs else [seg_dir])
        )
        self._verify_thread = threading.Thread(
            target=self._run_verify_quick_worker,
            args=(source_path, target_dirs),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_verify_quick_worker(self, source_path: str, target_dirs: list[str]) -> None:
        try:
            files: list[str] = []
            seg_count = 0
            seg_mono_count = 0
            expected_split_outputs: list[str] = []
            missing_split_outputs: list[str] = []
            split_cov_err = ""
            if target_dirs:
                expected_split_outputs, missing_split_outputs, split_cov_err = (
                    self._collect_expected_split_scene_outputs(target_dirs[0])
                )
            verify_patterns = self._scene_quick_verify_patterns()
            for idx, d in enumerate(target_dirs):
                cur = self._collect_files_for_patterns(d, verify_patterns)
                files.extend(cur)
                if idx == 0:
                    seg_count = len(cur)
                else:
                    seg_mono_count += len(cur)
            files = sorted(set(files))
            if not files:
                self._log_queue.put(("verify_quick_result", {
                    "ok": False,
                    "message": "No scene video files found in seg/seg-mono folders.",
                    "broken": [],
                    "missing_split": missing_split_outputs,
                    "split_cov_err": split_cov_err,
                    "duration_ok": False,
                    "frames_ok": False,
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            if len(target_dirs) > 1:
                self._log_queue.put(
                    (
                        "line",
                        (
                            f"[QUICK] checking {len(files)} scene files "
                            f"(seg={seg_count}, seg-mono={seg_mono_count}) "
                            f"with {max_workers} workers"
                        ),
                    )
                )
            else:
                self._log_queue.put(("line", f"[QUICK] checking {len(files)} scene files with {max_workers} workers"))

            broken: list[str] = []
            total_duration = 0.0
            duration_available = True
            total_frames = 0
            frames_available = True

            def _probe_one(fp: str) -> tuple[str, dict]:
                return fp, self._probe_video_basic(fp)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_probe_one, fp) for fp in files]
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    fp, meta = fut.result()
                    done += 1
                    if not meta.get("ok"):
                        broken.append(fp)
                        self._log_queue.put(("line", f"[QUICK][BROKEN] {fp} :: {meta.get('error')}"))
                    else:
                        dur = meta.get("duration")
                        frm = meta.get("frames")
                        if dur is None:
                            duration_available = False
                        else:
                            total_duration += float(dur)
                        if frm is None:
                            frames_available = False
                        else:
                            total_frames += int(frm)
                    if done % 25 == 0 or done == len(files):
                        self._log_queue.put(("line", f"[QUICK] progress {done}/{len(files)}"))

            src_meta = self._probe_video_basic(source_path)
            if not src_meta.get("ok"):
                self._log_queue.put(("verify_quick_result", {
                    "ok": False,
                    "message": f"Source probe failed: {src_meta.get('error')}",
                    "broken": broken,
                    "duration_ok": False,
                    "frames_ok": False,
                }))
                return

            src_duration = src_meta.get("duration")
            src_frames = src_meta.get("frames")
            duration_ok = False
            duration_msg = "n.d."
            if duration_available and src_duration is not None:
                dd = abs(float(total_duration) - float(src_duration))
                duration_ok = dd <= 0.35
                duration_msg = (
                    f"segments={total_duration:.3f}s vs source={float(src_duration):.3f}s "
                    f"(delta={dd:.3f}s)"
                )
            frames_ok = False
            frames_msg = "n.d."
            if frames_available and src_frames is not None:
                df = abs(int(total_frames) - int(src_frames))
                frames_ok = df <= 1
                frames_msg = (
                    f"segments={int(total_frames)} vs source={int(src_frames)} "
                    f"(delta={df})"
                )

            self._log_queue.put(("line", f"[QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("line", f"[QUICK] packet check: {frames_msg}"))
            if split_cov_err:
                self._log_queue.put(("line", f"[QUICK] split CSV check: ERROR: {split_cov_err}"))
            else:
                self._log_queue.put(
                    (
                        "line",
                        (
                            f"[QUICK] split CSV check: expected={len(expected_split_outputs)}, "
                            f"missing={len(missing_split_outputs)}"
                        ),
                    )
                )
            ok_final = (
                (len(broken) == 0)
                and (frames_ok or frames_msg == "n.d.")
                and (not split_cov_err)
                and (len(missing_split_outputs) == 0)
            )
            message = (
                f"Quick verify completed.\n"
                f"Broken files: {len(broken)}\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"Packet match (quick): {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')}\n"
                f"Packet details: {frames_msg}\n"
                f"Split CSV expected: {len(expected_split_outputs)}\n"
                f"Split CSV missing: {len(missing_split_outputs)}"
            )
            self._log_queue.put(("verify_quick_result", {
                "ok": ok_final,
                "message": message,
                "broken": broken,
                "missing_split": missing_split_outputs,
                "split_cov_err": split_cov_err,
                "duration_ok": duration_ok,
                "frames_ok": frames_ok,
            }))
        except Exception as e:
            self._log_queue.put(("verify_quick_result", {
                "ok": False,
                "message": f"Quick verify failed: {type(e).__name__}: {e}",
                "broken": [],
                "missing_split": [],
                "split_cov_err": "",
                "duration_ok": False,
                "frames_ok": False,
            }))
        finally:
            self._log_queue.put(("verify_done", "quick"))

    def _start_verify_deep(self) -> None:
        if self._scene_thread and self._scene_thread.is_alive():
            messagebox.showinfo(
                "Verify Scenes",
                "Stop SceneDetect/Split Scenes before running verification.",
            )
            return
        if self._verify_running:
            messagebox.showinfo("Verify Scenes", "Another verification is already running.")
            return
        ok, source_path, seg_dir = self._validate_verify_inputs()
        if not ok:
            return
        script_path = Path("Utilities/verifyscenes.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Verify Scenes", f"Script not found:\n{script_path}")
            return

        workers = self._get_verify_scenes_workers()
        source_resolved = str(Path(source_path).resolve())
        target_dirs = self._scene_verify_target_dirs(seg_dir)
        targets: list[tuple[str, str]] = []
        if target_dirs:
            targets.append(("seg", target_dirs[0]))
        for extra_dir in target_dirs[1:]:
            targets.append(("seg-mono", extra_dir))

        self._set_verify_running(True, mode="deep")
        self._scene_verify_result_applied = False
        self.scene_status_var.set("Verify (Deep) running...")
        self._append_scene_log("=== Verify Scenes (Deep) started ===")
        for label, target_dir in targets:
            cmd_preview = [
                sys.executable,
                str(script_path),
                target_dir,
                source_resolved,
                "--extensions",
                ".mp4",
                "--workers",
                str(workers),
                "--probe-timeout-sec",
                str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
                "--probe-timeout-retries",
                str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
                "--delete",
                "yes",
                "--no-single-line-progress",
            ]
            self._append_scene_log(
                f"CMD[{label}]: " + " ".join(shlex.quote(x) for x in cmd_preview)
            )

        self._verify_thread = threading.Thread(
            target=self._run_verify_deep_worker,
            args=(str(script_path), source_resolved, targets, workers),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_verify_deep_worker(
        self,
        script_path: str,
        source_path: str,
        targets: list[tuple[str, str]],
        workers: int,
    ) -> None:
        overall_rc = 0
        primary_target = targets[0][1] if targets else ""
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        missing_split_outputs: list[str] = []
        split_cov_err = ""
        if primary_target:
            _expected, missing_split_outputs, split_cov_err = self._collect_expected_split_scene_outputs(
                primary_target
            )
        try:
            for label, target_dir in targets:
                cmd = [
                    sys.executable,
                    script_path,
                    target_dir,
                    source_path,
                    "--extensions",
                    ".mp4",
                    "--workers",
                    str(workers),
                    "--probe-timeout-sec",
                    str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_SEC),
                    "--probe-timeout-retries",
                    str(self.VERIFY_DEEP_FFPROBE_TIMEOUT_RETRIES),
                    "--delete",
                    "yes",
                    "--no-single-line-progress",
                ]
                self._log_queue.put(
                    ("line", f"[DEEP][{label}] cmd: {' '.join(shlex.quote(x) for x in cmd)}")
                )
                rc = 1
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip("\n")
                    if line:
                        self._log_queue.put(("line", f"[DEEP][{label}] {line}"))
                        bad_path = self._resolve_verifyscenes_bad_path(line, target_dir)
                        if bad_path and bad_path not in seen_bad:
                            seen_bad.add(bad_path)
                            bad_files.append(bad_path)
                rc = int(proc.wait() or 0)
                if rc != 0:
                    overall_rc = rc if overall_rc == 0 else overall_rc
            if split_cov_err:
                self._log_queue.put(("line", f"[DEEP][split-csv] ERROR: {split_cov_err}"))
                overall_rc = overall_rc or 1
            elif missing_split_outputs:
                self._log_queue.put(
                    (
                        "line",
                        (
                            f"[DEEP][split-csv] missing split files: "
                            f"{len(missing_split_outputs)}"
                        ),
                    )
                )
                for miss in missing_split_outputs[:20]:
                    self._log_queue.put(("line", f"[DEEP][split-csv][MISSING] {miss}"))
                overall_rc = overall_rc or 1
        except Exception as e:
            self._log_queue.put(("line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            overall_rc = 1
        finally:
            self._log_queue.put(
                (
                    "verify_deep_result",
                    {
                        "rc": overall_rc,
                        "seg_dir": primary_target,
                        "bad_files": bad_files,
                        "missing_split": missing_split_outputs,
                        "split_cov_err": split_cov_err,
                    },
                )
            )
            self._log_queue.put(("verify_done", "deep"))

    @staticmethod
    def _delete_folder_contents(path: str) -> tuple[int, int, list[str]]:
        deleted_files = 0
        deleted_dirs = 0
        errors: list[str] = []
        root = Path(path)
        if not root.is_dir():
            return deleted_files, deleted_dirs, [f"folder not found: {path}"]
        for item in root.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    deleted_dirs += 1
                else:
                    item.unlink()
                    deleted_files += 1
            except Exception as e:
                errors.append(f"{item}: {e}")
        return deleted_files, deleted_dirs, errors

    @staticmethod
    def _delete_file_paths(paths: list[str]) -> tuple[int, list[str]]:
        deleted = 0
        errors: list[str] = []
        seen: set[str] = set()
        for raw in paths or []:
            path_txt = str(raw or "").strip()
            if not path_txt or path_txt in seen:
                continue
            seen.add(path_txt)
            fp = Path(path_txt)
            if not fp.exists():
                continue
            if not (fp.is_file() or fp.is_symlink()):
                continue
            try:
                fp.unlink()
                deleted += 1
            except Exception as e:
                errors.append(f"{fp}: {e}")
        return deleted, errors

    def _auto_cleanup_broken_files(self, paths: list[str], logger, label: str) -> tuple[int, int]:
        deleted, errors = self._delete_file_paths(paths)
        if deleted or errors:
            logger(f"[VERIFY][AUTO-CLEANUP] {label}: deleted={deleted}, errors={len(errors)}")
            for err in errors[:10]:
                logger(f"[VERIFY][AUTO-CLEANUP][ERR] {err}")
        return deleted, len(errors)

    @staticmethod
    def _extract_verifyscenes_bad_relpath(line: str) -> str | None:
        # Expected line format from Utilities/verifyscenes.py:
        # [BAD] 0001/0123 relative/path.mp4 :: reason text
        m = re.match(r"^\[BAD\]\s+\d+/\d+\s+(.+?)\s+::\s+", str(line or ""))
        if not m:
            return None
        rel = m.group(1).strip()
        return rel or None

    @staticmethod
    def _resolve_verifyscenes_bad_path(line: str, target_dir: str) -> str | None:
        rel = PipelineMasterGUI._extract_verifyscenes_bad_relpath(line)
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute():
            p = (Path(target_dir).resolve() / rel).resolve()
        return str(p)

    @staticmethod
    def _format_corrupted_files_block(
        paths: list[str],
        title: str = "Corrupted files",
        max_items: int = 20,
    ) -> str:
        uniq: list[str] = []
        seen: set[str] = set()
        for p in paths:
            s = str(p).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        if not uniq:
            return ""

        shown = uniq[: max(1, int(max_items))]
        more = len(uniq) - len(shown)
        out = f"\n\n{title} ({len(uniq)}):\n" + "\n".join(shown)
        if more > 0:
            out += f"\n... and {more} more"
        return out

    def _start_source_analysis(self, silent: bool = False) -> bool:
        if self._analysis_running:
            if not silent:
                messagebox.showinfo("Source Analysis", "Analysis already in progress.")
            return False
        source_path = self.scene_input_var.get().strip()
        if not source_path:
            if silent:
                self.scene_analysis_status_var.set("Ready (source not set)")
            else:
                messagebox.showerror("Source Analysis", "Source video path is required.")
            return False
        if not os.path.isfile(source_path):
            if silent:
                self.scene_analysis_status_var.set("Ready (source not found)")
                self._append_scene_log(f"[ANALYSIS] source video not found: {source_path}")
            else:
                messagebox.showerror("Source Analysis", f"Source video not found:\n{source_path}")
            return False
        if shutil.which("ffprobe") is None:
            if silent:
                self.scene_analysis_status_var.set("Ready (ffprobe missing)")
                self._append_scene_log("[ANALYSIS] ffprobe not found in PATH.")
            else:
                messagebox.showerror("Source Analysis", "ffprobe not found in PATH.")
            return False

        self.scene_analysis_status_var.set("Analyzing...")
        self._analysis_stop_requested = False
        self._set_analysis_running(True)
        self._analysis_thread = threading.Thread(
            target=self._run_source_analysis_worker, args=(source_path,), daemon=True
        )
        self._analysis_thread.start()
        return True

    def _start_source_analysis_on_startup(self) -> None:
        self._start_source_analysis(silent=True)

    def _run_source_analysis_worker(self, source_path: str) -> None:
        try:
            info = self._probe_source_video(source_path)
            if self._analysis_stop_requested:
                self._log_queue.put(("analysis_status", "Stopped by user"))
                return
            self._log_queue.put(("analysis_info", info))
            self._log_queue.put(("analysis_status", "Completed"))
        except Exception as exc:
            if self._analysis_stop_requested:
                self._log_queue.put(("analysis_status", "Stopped by user"))
            else:
                self._log_queue.put(("analysis_error", f"Analysis failed: {exc}"))
                self._log_queue.put(("analysis_status", "Failed"))
        finally:
            self._log_queue.put(("analysis_done", "1"))

    def _run_analysis_cmd(
        self, cmd: list[str], timeout_sec: float | None = None
    ) -> tuple[int, str, str]:
        if self._analysis_stop_requested:
            return 130, "", "analysis stopped by user"

        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._analysis_process = proc
            out, err = proc.communicate(timeout=timeout_sec)
            rc = int(proc.returncode or 0)
            return rc, out or "", err or ""
        except subprocess.TimeoutExpired:
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            return 124, "", f"timeout after {timeout_sec}s"
        finally:
            self._analysis_process = None

    def _run_subprocess_json(self, cmd: list[str], timeout_sec: float = 45.0) -> dict:
        rc, out, err = self._run_analysis_cmd(cmd, timeout_sec=timeout_sec)
        if self._analysis_stop_requested:
            raise RuntimeError("analysis stopped by user")
        if rc != 0:
            msg = (err or out).strip()
            raise RuntimeError(f"Command failed ({rc}): {msg}")
        return json.loads(out or "{}")

    def _probe_source_video(self, source_path: str) -> dict:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,pix_fmt,bits_per_raw_sample,color_transfer,"
                "color_space,color_primaries,avg_frame_rate,bit_rate,duration"
            ),
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            source_path,
        ]
        data = self._run_subprocess_json(ffprobe_cmd)
        streams = data.get("streams") or []
        if not streams:
            raise RuntimeError("No video stream found.")
        stream = streams[0]
        fmt = data.get("format") or {}

        width = self._parse_int(stream.get("width"))
        height = self._parse_int(stream.get("height"))
        pix_fmt = self._as_text(stream.get("pix_fmt"))
        bit_depth = self._extract_bit_depth(stream, pix_fmt)
        dynamic_range = self._infer_dynamic_range(stream, bit_depth)
        chroma = self._parse_chroma_subsampling(pix_fmt)
        fps = self._parse_fps(stream.get("avg_frame_rate"))
        duration_sec = self._parse_float(stream.get("duration"))
        if duration_sec is None:
            duration_sec = self._parse_float(fmt.get("duration"))
        bitrate = self._parse_int(stream.get("bit_rate"))

        bars_top = None
        bars_bottom = None
        if width and height:
            bars = self._detect_black_bars_cropdetect(
                source_path=source_path,
                source_height=height,
                duration_sec=duration_sec,
            )
            if bars is not None:
                bars_top, bars_bottom = bars

        return {
            "source_path": source_path,
            "width": width,
            "height": height,
            "bars_top": bars_top,
            "bars_bottom": bars_bottom,
            "dynamic_range": dynamic_range,
            "bit_depth": bit_depth,
            "pix_fmt": pix_fmt,
            "chroma": chroma,
            "duration_sec": duration_sec,
            "fps": fps,
            "bitrate": bitrate,
        }

    def _detect_black_bars_cropdetect(
        self, source_path: str, source_height: int, duration_sec: float | None
    ) -> tuple[int, int] | None:
        if self._analysis_stop_requested:
            return None
        if shutil.which("ffmpeg") is None:
            return None
        ss = 0.0
        if duration_sec and duration_sec > 0:
            ss = max(0.0, min(duration_sec * 0.15, max(0.0, duration_sec - 8.0)))
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-ss",
            f"{ss:.3f}",
            "-i",
            source_path,
            "-t",
            "8",
            "-vf",
            "cropdetect=limit=24:round=2:reset=0",
            "-an",
            "-f",
            "null",
            "-",
        ]
        rc, out, err = self._run_analysis_cmd(cmd, timeout_sec=60.0)
        if self._analysis_stop_requested:
            return None
        if rc != 0 and rc != 1:
            return None
        text = f"{out}\n{err}"
        matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", text)
        if not matches:
            return None
        normalized = [(int(w), int(h), int(x), int(y)) for (w, h, x, y) in matches]
        best, _count = Counter(normalized).most_common(1)[0]
        _w, crop_h, _x, crop_y = best
        top = max(0, int(crop_y))
        bottom = max(0, int(source_height - (crop_y + crop_h)))
        if top + bottom >= int(source_height * 0.45):
            return None
        return top, bottom

    def _update_analysis_fields(self, info: dict) -> None:
        self.analysis_source_path_var.set(info.get("source_path") or "n.d.")
        self.analysis_resolution_var.set(self._fmt_resolution(info.get("width"), info.get("height")))
        self.analysis_bars_var.set(self._fmt_bars(info.get("bars_top"), info.get("bars_bottom")))

        bit_depth = info.get("bit_depth")
        dynamic_range = info.get("dynamic_range") or "n.d."
        bit_label = f"{bit_depth}-bit" if bit_depth is not None else "n.d."
        self.analysis_color_var.set(f"{bit_label} ({dynamic_range})")

        pix_fmt = info.get("pix_fmt") or "n.d."
        chroma = info.get("chroma") or "n.d."
        self.analysis_pixfmt_var.set(f"{pix_fmt} ({chroma})")
        self.analysis_length_var.set(self._fmt_duration(info.get("duration_sec")))
        self.analysis_fps_var.set(self._fmt_number(info.get("fps"), nd="n.d.", suffix=" fps", decimals=3))
        self.analysis_bitrate_var.set(self._fmt_bitrate(info.get("bitrate")))
        self._update_depth_resolution_preview()

    @staticmethod
    def _as_text(value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _parse_int(value) -> int | None:
        if value in (None, "", "N/A"):
            return None
        try:
            return int(str(value))
        except Exception:
            return None

    @staticmethod
    def _parse_float(value) -> float | None:
        if value in (None, "", "N/A"):
            return None
        try:
            return float(str(value))
        except Exception:
            return None

    @staticmethod
    def _parse_fps(raw_value) -> float | None:
        if raw_value in (None, "", "0/0", "N/A"):
            return None
        text = str(raw_value).strip()
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            try:
                num = float(num_s)
                den = float(den_s)
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None
        try:
            return float(text)
        except Exception:
            return None

    @staticmethod
    def _parse_chroma_subsampling(pix_fmt: str) -> str | None:
        if not pix_fmt:
            return None
        pix = pix_fmt.lower()
        match = re.search(r"(?:yuv|yuva|nv)(420|422|444)", pix)
        if match:
            return match.group(1)
        if "gray" in pix:
            return "400"
        return None

    def _extract_bit_depth(self, stream: dict, pix_fmt: str) -> int | None:
        bits = self._parse_int(stream.get("bits_per_raw_sample"))
        if bits:
            return bits
        if pix_fmt:
            match = re.search(r"p(\d{2})(?:le|be)?$", pix_fmt.lower())
            if match:
                parsed = self._parse_int(match.group(1))
                if parsed:
                    return parsed
        return None

    def _infer_dynamic_range(self, stream: dict, bit_depth: int | None) -> str:
        transfer = self._as_text(stream.get("color_transfer")).lower()
        primaries = self._as_text(stream.get("color_primaries")).lower()
        if transfer in {"smpte2084", "arib-std-b67"}:
            return "HDR"
        if "bt2020" in primaries and bit_depth and bit_depth >= 10:
            return "HDR (likely)"
        if bit_depth and bit_depth >= 10:
            return "10-bit SDR"
        if bit_depth and bit_depth <= 8:
            return "8-bit SDR"
        return "n.d."

    @staticmethod
    def _fmt_resolution(width: int | None, height: int | None) -> str:
        if width is None or height is None:
            return "n.d."
        return f"{width}x{height}"

    @staticmethod
    def _fmt_bars(top: int | None, bottom: int | None) -> str:
        if top is None or bottom is None:
            return "n.d."
        if top == 0 and bottom == 0:
            return "none detected"
        return f"top={top}px, bottom={bottom}px"

    @staticmethod
    def _fmt_duration(duration_sec: float | None) -> str:
        if duration_sec is None:
            return "n.d."
        total_ms = int(round(duration_sec * 1000.0))
        hours = total_ms // 3_600_000
        rem = total_ms % 3_600_000
        minutes = rem // 60_000
        rem %= 60_000
        seconds = rem // 1000
        ms = rem % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

    @staticmethod
    def _fmt_number(
        value: float | None, nd: str = "n.d.", suffix: str = "", decimals: int = 2
    ) -> str:
        if value is None:
            return nd
        return f"{value:.{decimals}f}{suffix}"

    @staticmethod
    def _fmt_bitrate(value: int | None) -> str:
        if value is None:
            return "n.d."
        mbps = value / 1_000_000.0
        return f"{mbps:.3f} Mbps ({value} bps)"

    def _start_scene_detect(self) -> None:
        if self._scene_thread and self._scene_thread.is_alive():
            messagebox.showinfo("SceneDetect", "A SceneDetect task is already running.")
            return
        if not self._validate_scene_form():
            return

        self._preview_scene_command()
        self.scene_progress_var.set(0.0)
        self.scene_status_var.set("Starting...")
        self._set_scene_running(True)
        self._scene_stop_requested = False
        self._scene_active_step = "scenedetect"
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("scenedetect")

        cmd = self._build_scenedetect_command()
        self._scene_thread = threading.Thread(
            target=self._run_scene_detect_worker, args=(cmd,), daemon=True
        )
        self._scene_thread.start()

    def _run_scene_detect_worker(self, cmd: list[str]) -> None:
        proc = None
        step_name = "scenedetect"
        try:
            output_dir = self.scene_output_var.get().strip()
            csv_path = self._scene_csv_path()
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(str(Path(csv_path).resolve().parent), exist_ok=True)

            self._log_queue.put(("line", "=== SceneDetect started (CSV only) ==="))
            self._log_queue.put(("line", "CMD: " + " ".join(shlex.quote(p) for p in cmd)))

            popen_kwargs: dict[str, object] = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            elif os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                **popen_kwargs,
            )
            self._scene_process = proc

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                self._log_queue.put(("line", line))
                self._try_parse_progress(line)
                if self._scene_stop_requested:
                    break

            rc = proc.wait()
            if self._scene_stop_requested:
                self._log_queue.put(("status", "Stopped by user"))
            elif rc == 0:
                mono_dir = os.path.join(os.path.dirname(os.path.normpath(output_dir)), "seg-mono")
                try:
                    os.makedirs(mono_dir, exist_ok=True)
                    self._log_queue.put(("line", f"[INFO] seg-mono folder ready: {mono_dir}"))
                except Exception as exc:
                    self._log_queue.put(("line", f"[WARN] could not create seg-mono folder: {exc}"))
                self._log_queue.put(("line", f"[INFO] scene CSV: {csv_path}"))
                self._log_queue.put(("status", "Completed"))
                self._log_queue.put(("progress", "100"))
            else:
                self._log_queue.put(("status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("line", f"[ERROR] {exc}"))
            self._log_queue.put(("status", "Failed"))
        finally:
            self._scene_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(
                (
                    "done",
                    {
                        "step": step_name,
                        "success": (not self._scene_stop_requested and proc is not None and proc.returncode == 0),
                    },
                )
            )

    def _start_split_scenes(self) -> None:
        if self._scene_thread and self._scene_thread.is_alive():
            messagebox.showinfo("Split Scenes", "A SceneDetect task is already running.")
            return
        if self._verify_running:
            messagebox.showinfo("Split Scenes", "Stop verification before splitting scenes.")
            return
        if not self._validate_scene_form(require_scenedetect=False):
            return
        scene_csv_path = self._scene_csv_path()
        entries, csv_err = self._load_scene_csv_entries(scene_csv_path)
        if csv_err or not entries:
            messagebox.showerror(
                "Split Scenes",
                (
                    f"Scene CSV missing or invalid:\n{scene_csv_path}\n\n"
                    "Run SceneDetect first."
                ),
            )
            return
        script_path = Path("Utilities/split_scenes_from_csv.py").resolve()
        if not script_path.is_file():
            messagebox.showerror("Split Scenes", f"Script not found:\n{script_path}")
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror("Split Scenes", "ffmpeg not found in PATH.")
            return

        self.scene_progress_var.set(0.0)
        self.scene_status_var.set("Starting...")
        self._set_scene_running(True)
        self._scene_stop_requested = False
        self._scene_active_step = "split_scenes"
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("split_scenes")

        cmd = self._build_split_scenes_command()
        self._scene_thread = threading.Thread(
            target=self._run_split_scenes_worker,
            args=(cmd,),
            daemon=True,
        )
        self._scene_thread.start()

    def _run_split_scenes_worker(self, cmd: list[str]) -> None:
        proc = None
        step_name = "split_scenes"
        try:
            output_dir = self.scene_output_var.get().strip()
            os.makedirs(output_dir, exist_ok=True)
            self._log_queue.put(("line", "=== Split Scenes started ==="))
            self._log_queue.put(("line", "CMD: " + " ".join(shlex.quote(p) for p in cmd)))

            popen_kwargs: dict[str, object] = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            elif os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                **popen_kwargs,
            )
            self._scene_process = proc

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                self._log_queue.put(("line", line))
                self._try_parse_progress(line)
                if self._scene_stop_requested:
                    break

            rc = proc.wait()
            if self._scene_stop_requested:
                self._log_queue.put(("status", "Stopped by user"))
            elif rc == 0:
                self._log_queue.put(("status", "Completed"))
                self._log_queue.put(("progress", "100"))
            else:
                self._log_queue.put(("status", f"Failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("line", f"[ERROR] {exc}"))
            self._log_queue.put(("status", "Failed"))
        finally:
            self._scene_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(
                (
                    "done",
                    {
                        "step": step_name,
                        "success": (not self._scene_stop_requested and proc is not None and proc.returncode == 0),
                    },
                )
            )

    def _stop_scene_detect(self, prompt_user: bool = True) -> None:
        scene_running = bool(self._scene_thread and self._scene_thread.is_alive())
        analysis_running = bool(self._analysis_running)
        if not scene_running and not analysis_running:
            return
        active_label = "Split Scenes" if self._scene_active_step == "split_scenes" else "SceneDetect"

        if analysis_running:
            self._analysis_stop_requested = True
            self.scene_analysis_status_var.set("Stopping...")
            self._append_scene_log("Stopping source analysis...")
            proc_a = self._analysis_process
            if proc_a is not None and proc_a.poll() is None:
                try:
                    proc_a.terminate()
                except Exception as exc:
                    self._append_scene_log(f"Analysis terminate failed: {exc}")

        if scene_running:
            self._scene_stop_requested = True
            self.scene_status_var.set("Stopping...")
            self._append_scene_log(f"Stopping {active_label}...")
            proc_s = self._scene_process
            if proc_s is not None and proc_s.poll() is None:
                stop_sig = signal.SIGINT if self._scene_active_step == "split_scenes" else signal.SIGTERM
                sent = False
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        os.killpg(os.getpgid(proc_s.pid), stop_sig)
                        sent = True
                        self._append_scene_log(
                            f"[STOP] sent {stop_sig.name} to process group (pid={proc_s.pid})."
                        )
                    except Exception:
                        sent = False
                if not sent:
                    try:
                        proc_s.send_signal(stop_sig)
                        self._append_scene_log(
                            f"[STOP] sent {stop_sig.name} to process (no process-group signal)."
                        )
                    except Exception as send_exc:
                        try:
                            proc_s.terminate()
                            self._append_scene_log("[STOP] sent terminate() to process.")
                        except Exception as term_exc:
                            self._append_scene_log(
                                f"Terminate failed after send_signal error ({send_exc}): {term_exc}"
                            )
            # Give ffmpeg/python splitter time to flush and close current output.
            self.root.after(6000, self._force_kill_scene_detect)

    def _force_kill_scene_detect(self) -> None:
        proc = self._scene_process
        if proc is None:
            return
        if proc.poll() is None:
            killed = False
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    killed = True
                    self._append_scene_log("Process group killed after timeout.")
                except Exception:
                    killed = False
            if not killed:
                try:
                    proc.kill()
                    self._append_scene_log("Process killed after timeout.")
                except Exception as exc:
                    self._append_scene_log(f"Kill failed: {exc}")

    def _try_parse_progress(self, line: str) -> None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
        if not match:
            return
        self._log_queue.put(("progress", match.group(1)))

    def _clear_scene_log(self) -> None:
        self.scene_log_text.configure(state=tk.NORMAL)
        self.scene_log_text.delete("1.0", tk.END)
        self.scene_log_text.configure(state=tk.DISABLED)

    def _append_scene_log(self, line: str) -> None:
        self.scene_log_text.configure(state=tk.NORMAL)
        self.scene_log_text.insert(tk.END, line + "\n")
        self.scene_log_text.see(tk.END)
        self.scene_log_text.configure(state=tk.DISABLED)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "line":
                    self._append_scene_log(str(payload))
                elif kind == "depth_line":
                    self._append_depth_log(str(payload))
                elif kind == "splat_line":
                    self._append_splat_log(str(payload))
                elif kind == "inpaint_line":
                    self._append_inpaint_log(str(payload))
                elif kind == "merge_line":
                    self._append_merge_log(str(payload))
                elif kind == "join_line":
                    self._append_join_log(str(payload))
                elif kind == "status":
                    self.scene_status_var.set(str(payload))
                elif kind == "depth_status":
                    self.depth_status_var.set(str(payload))
                elif kind == "splat_status":
                    self.splat_status_var.set(str(payload))
                elif kind == "inpaint_status":
                    self.inpaint_status_var.set(str(payload))
                elif kind == "merge_status":
                    self.merge_status_var.set(str(payload))
                elif kind == "join_status":
                    self.join_status_var.set(str(payload))
                elif kind == "progress":
                    try:
                        self.scene_progress_var.set(max(0.0, min(100.0, float(payload))))
                    except Exception:
                        pass
                elif kind == "depth_progress":
                    try:
                        self.depth_progress_var.set(max(0.0, min(100.0, float(payload))))
                    except Exception:
                        pass
                elif kind == "splat_progress":
                    try:
                        self.splat_progress_var.set(max(0.0, min(100.0, float(payload))))
                    except Exception:
                        pass
                elif kind == "inpaint_progress":
                    try:
                        self.inpaint_progress_var.set(max(0.0, min(100.0, float(payload))))
                    except Exception:
                        pass
                elif kind == "merge_progress":
                    try:
                        self.merge_progress_var.set(max(0.0, min(100.0, float(payload))))
                    except Exception:
                        pass
                elif kind == "join_progress":
                    try:
                        self.join_progress_var.set(max(0.0, min(100.0, float(payload))))
                    except Exception:
                        pass
                elif kind == "done":
                    stop_requested = bool(self._scene_stop_requested)
                    self._set_scene_running(False)
                    status_txt = self.scene_status_var.get().strip().lower()
                    step_name = self._scene_active_step or "scenedetect"
                    success = status_txt == "completed"
                    if isinstance(payload, dict):
                        step_name = str(payload.get("step", step_name)).strip().lower() or step_name
                        if "success" in payload:
                            success = bool(payload.get("success", False))
                            if (not success) and ("completed" in status_txt):
                                success = True
                    if step_name not in {"scenedetect", "split_scenes"}:
                        step_name = "scenedetect"
                    self._scene_active_step = step_name
                    self._pipeline_on_run_finished(step_name, success)
                    if stop_requested:
                        stop_label = "Split Scenes" if step_name == "split_scenes" else "SceneDetect"
                        self._append_scene_log(f"[STOP] {stop_label} stopped.")
                    self._scene_stop_requested = False
                elif kind == "depth_done":
                    stop_requested = bool(self._depth_stop_requested)
                    status_txt = self.depth_status_var.get().strip().lower()
                    step_name = ""
                    success = False
                    if isinstance(payload, dict):
                        step_name = str(payload.get("step", "")).strip().lower()
                        success = bool(payload.get("success", False))
                    else:
                        step_name = "depthcrafter"
                        success = "completed" in status_txt
                    self._set_depth_running(False)
                    if step_name in {"depthcrafter", "depth_upscale"}:
                        self._pipeline_on_run_finished(step_name, success)
                    else:
                        self._pipeline_on_run_finished("depthcrafter", "completed" in status_txt)
                        step_name = "depthcrafter"
                    if stop_requested:
                        stop_label = "Depth Upscale" if step_name == "depth_upscale" else "DepthCrafter"
                        self._append_depth_log(f"[STOP] {stop_label} stopped.")
                elif kind == "splat_done":
                    stop_requested = bool(self._splat_stop_requested)
                    self._set_splat_running(False)
                    status_txt = self.splat_status_var.get().strip().lower()
                    self._pipeline_on_run_finished("splatting", "completed" in status_txt)
                    if stop_requested:
                        self._append_splat_log("[STOP] Splatting stopped.")
                elif kind == "inpaint_done":
                    stop_requested = bool(self._inpaint_stop_requested)
                    step_name = ""
                    success = False
                    pending_before = self._pipeline_pending_action
                    pending_inpaint_run = (
                        isinstance(pending_before, tuple)
                        and len(pending_before) >= 2
                        and str(pending_before[0]).strip().lower() == "inpaint"
                        and str(pending_before[1]).strip().lower() == "run"
                    )
                    should_resume_inpaint = False
                    if isinstance(payload, dict):
                        step_name = str(payload.get("step", "")).strip().lower()
                        success = bool(payload.get("success", False))
                        if (
                            step_name == "sharpness_csv"
                            and bool(self._inpaint_resume_after_sharpness)
                            and success
                            and not stop_requested
                        ):
                            should_resume_inpaint = True
                    self._set_inpaint_running(False)
                    if step_name in {"sharpness_csv", "inpaint"}:
                        self._pipeline_on_run_finished(step_name, success)
                    else:
                        status_txt = self.inpaint_status_var.get().strip().lower()
                        if "sharpness csv created" in status_txt:
                            self._pipeline_on_run_finished("sharpness_csv", True)
                            step_name = "sharpness_csv"
                        elif "completed" in status_txt:
                            self._pipeline_on_run_finished("inpaint", True)
                            step_name = "inpaint"
                        else:
                            pending = self._pipeline_pending_action
                            if pending and pending[1] == "run" and pending[0] in {"sharpness_csv", "inpaint"}:
                                self._pipeline_on_run_finished(pending[0], False)
                                step_name = pending[0]
                    if step_name == "sharpness_csv" and bool(self._inpaint_resume_after_sharpness):
                        self._inpaint_resume_after_sharpness = False
                        if should_resume_inpaint:
                            self._append_inpaint_log(
                                "[SHARP] CSV rebuilt. Resuming Inpainting automatically..."
                            )
                            self.root.after(10, self._run_inpaint_placeholder)
                        elif pending_inpaint_run and not success:
                            # Sharpness preflight failed while Inpaint run was pending.
                            self._pipeline_on_run_finished("inpaint", False)
                    if stop_requested:
                        stop_label = "Sharpness CSV" if step_name == "sharpness_csv" else "Inpainting"
                        self._append_inpaint_log(f"[STOP] {stop_label} stopped.")
                elif kind == "merge_done":
                    stop_requested = bool(self._merge_stop_requested)
                    step_name = ""
                    success = False
                    if isinstance(payload, dict):
                        step_name = str(payload.get("step", "")).strip().lower()
                        success = bool(payload.get("success", False))
                    self._set_merge_running(False)
                    if step_name in {"autoct_csv", "mask_for_merge", "merging"}:
                        self._pipeline_on_run_finished(step_name, success)
                    else:
                        status_txt = self.merge_status_var.get().strip().lower()
                        if "autoct csv created" in status_txt:
                            self._pipeline_on_run_finished("autoct_csv", True)
                            step_name = "autoct_csv"
                        elif "mask-for-merge completed" in status_txt:
                            self._pipeline_on_run_finished("mask_for_merge", True)
                            step_name = "mask_for_merge"
                        elif "completed" in status_txt:
                            self._pipeline_on_run_finished("merging", True)
                            step_name = "merging"
                        else:
                            pending = self._pipeline_pending_action
                            if pending and pending[1] == "run" and pending[0] in {"autoct_csv", "mask_for_merge", "merging"}:
                                self._pipeline_on_run_finished(pending[0], False)
                                step_name = pending[0]
                    if stop_requested:
                        label_map = {
                            "autoct_csv": "AutoCT CSV",
                            "mask_for_merge": "Mask-for-merge",
                            "merging": "Merging",
                        }
                        self._append_merge_log(f"[STOP] {label_map.get(step_name, 'Merging')} stopped.")
                elif kind == "join_done":
                    stop_requested = bool(self._join_stop_requested)
                    self._set_join_running(False)
                    status_txt = self.join_status_var.get().strip().lower()
                    step_name = "join"
                    if isinstance(payload, dict):
                        step_name = str(payload.get("step", "join")).strip().lower() or "join"
                    success = (
                        bool(payload.get("success", False))
                        if isinstance(payload, dict) and "success" in payload
                        else ("completed" in status_txt)
                    )
                    if step_name == "remux":
                        self._pipeline_on_run_finished("remux", success)
                    elif step_name == "mono_to_sbs":
                        self._pipeline_on_run_finished("mono_to_sbs", success)
                    else:
                        self._pipeline_on_run_finished("join", success)
                    if stop_requested:
                        label_map = {
                            "mono_to_sbs": "Mono->SBS",
                            "remux": "Remux",
                            "join": "Join",
                        }
                        self._append_join_log(f"[STOP] {label_map.get(step_name, 'Join')} stopped.")
                elif kind == "analysis_info" and isinstance(payload, dict):
                    self._source_video_info = payload
                    self._update_source_capabilities(payload)
                    self._update_crop_recommendations(payload)
                    self._update_analysis_fields(payload)
                    self._preview_scene_command()
                elif kind == "analysis_error":
                    self._append_scene_log(str(payload))
                elif kind == "analysis_status":
                    self.scene_analysis_status_var.set(str(payload))
                elif kind == "analysis_done":
                    analysis_stopped = bool(self._analysis_stop_requested)
                    self._set_analysis_running(False)
                    if analysis_stopped:
                        self._append_scene_log("[STOP] Source analysis stopped.")
                    self._analysis_stop_requested = False
                elif kind == "verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Quick verification finished."))
                    broken_files = [str(p) for p in (payload.get("broken") or []) if str(p).strip()]
                    missing_split = [str(p) for p in (payload.get("missing_split") or []) if str(p).strip()]
                    split_cov_err = str(payload.get("split_cov_err", "")).strip()
                    if ok:
                        self.scene_status_var.set("Verify (Quick) completed")
                        messagebox.showinfo("Verify Scenes (Quick)", msg)
                    else:
                        self.scene_status_var.set("Verify (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_files:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_files, self._append_scene_log, "seg/seg-mono"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_files:
                            msg += self._format_corrupted_files_block(
                                broken_files,
                                "Corrupted scene files",
                            )
                        if split_cov_err:
                            msg += f"\n\nSplit CSV check error:\n{split_cov_err}"
                        if missing_split:
                            msg += self._format_corrupted_files_block(
                                missing_split,
                                "Missing split files (from CSV)",
                            )
                        messagebox.showwarning("Verify Scenes (Quick)", msg)
                    self._scene_verify_result_applied = True
                    self._pipeline_on_verify_finished("split_scenes", ok, "quick")
                elif kind == "verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    missing_split = [str(p) for p in (payload.get("missing_split") or []) if str(p).strip()]
                    split_cov_err = str(payload.get("split_cov_err", "")).strip()
                    if rc == 0:
                        self.scene_status_var.set("Verify (Deep) completed")
                        messagebox.showinfo(
                            "Verify Scenes (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.scene_status_var.set(f"Verify (Deep) failed (exit {rc})")
                        warn_msg = (
                            "Deep verification failed.\n\n"
                            "Broken files were auto-deleted by verifier where possible."
                        )
                        if split_cov_err:
                            warn_msg += f"\n\nSplit CSV check error:\n{split_cov_err}"
                        warn_msg += self._format_corrupted_files_block(
                            bad_files,
                            "Corrupted scene files",
                        )
                        warn_msg += self._format_corrupted_files_block(
                            missing_split,
                            "Missing split files (from CSV)",
                        )
                        messagebox.showwarning(
                            "Verify Scenes (Deep)",
                            warn_msg,
                        )
                    self._scene_verify_result_applied = True
                    self._pipeline_on_verify_finished("split_scenes", rc == 0, "deep")
                elif kind == "depth_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Depth quick verification finished."))
                    broken_depth = [
                        str(p) for p in (payload.get("broken_depth") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    if ok:
                        self.depth_status_var.set("Verify (Quick) completed")
                        messagebox.showinfo("Verify Depth (Quick)", msg)
                    else:
                        self.depth_status_var.set("Verify (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_depth:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_depth, self._append_depth_log, "depthmap"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_depth:
                            msg += self._format_corrupted_files_block(
                                broken_depth,
                                "Corrupted depth files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted reference files",
                            )
                        messagebox.showwarning("Verify Depth (Quick)", msg)
                    self._pipeline_on_verify_finished("depthcrafter", ok, "quick")
                elif kind == "depth_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.depth_status_var.set("Verify (Deep) completed")
                        messagebox.showinfo(
                            "Verify Depth (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.depth_status_var.set(f"Verify (Deep) failed (exit {rc})")
                        messagebox.showwarning(
                            "Verify Depth (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were auto-deleted by verifier where possible."
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted depth files",
                            ),
                        )
                    self._pipeline_on_verify_finished("depthcrafter", rc == 0, "deep")
                elif kind == "depth_upscaled_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Upscale quick verification finished."))
                    broken_upscaled = [
                        str(p) for p in (payload.get("broken_upscaled") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    if ok:
                        self.depth_status_var.set("Verify Upscale (Quick) completed")
                        messagebox.showinfo("Verify Upscale (Quick)", msg)
                    else:
                        self.depth_status_var.set("Verify Upscale (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_upscaled:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_upscaled, self._append_depth_log, "depth upscaled"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_upscaled:
                            msg += self._format_corrupted_files_block(
                                broken_upscaled,
                                "Corrupted upscaled files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted reference files",
                            )
                        messagebox.showwarning("Verify Upscale (Quick)", msg)
                    self._pipeline_on_verify_finished("depth_upscale", ok, "quick")
                elif kind == "depth_upscaled_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.depth_status_var.set("Verify Upscale (Deep) completed")
                        messagebox.showinfo(
                            "Verify Upscale (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.depth_status_var.set(f"Verify Upscale (Deep) failed (exit {rc})")
                        messagebox.showwarning(
                            "Verify Upscale (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were auto-deleted by verifier where possible."
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted upscaled files",
                            ),
                        )
                    self._pipeline_on_verify_finished("depth_upscale", rc == 0, "deep")
                elif kind == "splat_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Splat quick verification finished."))
                    broken_targets = [
                        str(p) for p in (payload.get("broken_targets") or []) if str(p).strip()
                    ]
                    if ok:
                        self.splat_status_var.set("Verify (Quick) completed")
                        messagebox.showinfo("Verify Splatting (Quick)", msg)
                    else:
                        self.splat_status_var.set("Verify (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_targets:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_targets, self._append_splat_log, "splat targets"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_targets:
                            msg += self._format_corrupted_files_block(
                                broken_targets,
                                "Corrupted splat target files",
                            )
                        messagebox.showwarning("Verify Splatting (Quick)", msg)
                    self._pipeline_on_verify_finished("splatting", ok, "quick")
                elif kind == "splat_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    failed_dirs = payload.get("failed_dirs") or []
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.splat_status_var.set("Verify (Deep) completed")
                        messagebox.showinfo(
                            "Verify Splatting (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.splat_status_var.set(f"Verify (Deep) failed (exit {rc})")
                        if failed_dirs:
                            failed_txt = "\n".join(str(d) for d in failed_dirs)
                        else:
                            failed_txt = "n.d."
                        messagebox.showwarning(
                            "Verify Splatting (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were auto-deleted by verifier where possible.\n\n"
                                f"Failed target folders:\n{failed_txt}"
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted splat target files",
                            ),
                        )
                    self._pipeline_on_verify_finished("splatting", rc == 0, "deep")
                elif kind == "inpaint_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Inpainting quick verification finished."))
                    broken_output = [
                        str(p) for p in (payload.get("broken_output") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    if ok:
                        self.inpaint_status_var.set("Verify (Quick) completed")
                        messagebox.showinfo("Verify Inpainting (Quick)", msg)
                    else:
                        self.inpaint_status_var.set("Verify (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_output:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_output, self._append_inpaint_log, "inpaint output"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_output:
                            msg += self._format_corrupted_files_block(
                                broken_output,
                                "Corrupted inpaint output files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted reference files",
                            )
                        messagebox.showwarning("Verify Inpainting (Quick)", msg)
                    self._pipeline_on_verify_finished("inpaint", ok, "quick")
                elif kind == "inpaint_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.inpaint_status_var.set("Verify (Deep) completed")
                        messagebox.showinfo(
                            "Verify Inpainting (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.inpaint_status_var.set(f"Verify (Deep) failed (exit {rc})")
                        messagebox.showwarning(
                            "Verify Inpainting (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were auto-deleted by verifier where possible."
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted inpaint output files",
                            ),
                        )
                    self._pipeline_on_verify_finished("inpaint", rc == 0, "deep")
                elif kind == "merge_mask_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Mask quick verification finished."))
                    broken_output = [
                        str(p) for p in (payload.get("broken_output") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    if ok:
                        self.merge_status_var.set("Verify Mask (Quick) completed")
                        messagebox.showinfo("Verify Mask (Quick)", msg)
                    else:
                        self.merge_status_var.set("Verify Mask (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_output:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_output, self._append_merge_log, "mask_for_merge output"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_output:
                            msg += self._format_corrupted_files_block(
                                broken_output,
                                "Corrupted mask-for-merge files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted reference files",
                            )
                        messagebox.showwarning("Verify Mask (Quick)", msg)
                    self._pipeline_on_verify_finished("mask_for_merge", ok, "quick")
                elif kind == "merge_mask_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.merge_status_var.set("Verify Mask (Deep) completed")
                        messagebox.showinfo(
                            "Verify Mask (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.merge_status_var.set(f"Verify Mask (Deep) failed (exit {rc})")
                        messagebox.showwarning(
                            "Verify Mask (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were auto-deleted by verifier where possible."
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted mask-for-merge files",
                            ),
                        )
                    self._pipeline_on_verify_finished("mask_for_merge", rc == 0, "deep")
                elif kind == "merge_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Merging quick verification finished."))
                    broken_output = [
                        str(p) for p in (payload.get("broken_output") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    if ok:
                        self.merge_status_var.set("Verify (Quick) completed")
                        messagebox.showinfo("Verify Merging (Quick)", msg)
                    else:
                        self.merge_status_var.set("Verify (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        if broken_output:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_output, self._append_merge_log, "merged output"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_output:
                            msg += self._format_corrupted_files_block(
                                broken_output,
                                "Corrupted merged output files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted reference files",
                            )
                        messagebox.showwarning("Verify Merging (Quick)", msg)
                    self._pipeline_on_verify_finished("merging", ok, "quick")
                elif kind == "merge_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.merge_status_var.set("Verify (Deep) completed")
                        messagebox.showinfo(
                            "Verify Merging (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.merge_status_var.set(f"Verify (Deep) failed (exit {rc})")
                        messagebox.showwarning(
                            "Verify Merging (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were auto-deleted by verifier where possible."
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted merged output files",
                            ),
                        )
                    self._pipeline_on_verify_finished("merging", rc == 0, "deep")
                elif kind == "join_verify_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Join verification finished."))
                    if ok:
                        self.join_status_var.set("Verify completed")
                        messagebox.showinfo("Verify Join", msg)
                    else:
                        self.join_status_var.set("Verify failed")
                        messagebox.showwarning("Verify Join", msg)
                    pending = self._pipeline_pending_action
                    mode = "quick"
                    if pending and pending[0] == "join" and pending[1] == "verify":
                        mode = "deep" if pending[2] == "deep" else "quick"
                    self._pipeline_on_verify_finished("join", ok, mode)
                elif kind == "join_mono_verify_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Mono->SBS verification finished."))
                    mode = "deep" if str(payload.get("mode", "")).strip().lower() == "deep" else "quick"
                    broken_output = [
                        str(p) for p in (payload.get("broken_output") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    if ok:
                        self.join_status_var.set(
                            f"Mono->SBS Verify ({'Deep' if mode == 'deep' else 'Quick'}) completed"
                        )
                        messagebox.showinfo("Verify Mono->SBS", msg)
                    else:
                        self.join_status_var.set(
                            f"Mono->SBS Verify ({'Deep' if mode == 'deep' else 'Quick'}) failed"
                        )
                        deleted = 0
                        cleanup_err = 0
                        if broken_output:
                            deleted, cleanup_err = self._auto_cleanup_broken_files(
                                broken_output, self._append_join_log, "mono_to_sbs output"
                            )
                        if deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_output:
                            msg += self._format_corrupted_files_block(
                                broken_output,
                                "Corrupted Mono->SBS output files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted seg-mono source files",
                            )
                        messagebox.showwarning("Verify Mono->SBS", msg)
                    self._pipeline_on_verify_finished("mono_to_sbs", ok, mode)
                elif kind == "verify_done":
                    mode_txt = str(payload or "").strip().lower()
                    if mode_txt in {"quick", "deep"} and not self._scene_verify_result_applied:
                        status_txt = self.scene_status_var.get().strip().lower()
                        verify_ok = ("completed" in status_txt) and ("failed" not in status_txt)
                        self._pipeline_on_verify_finished("split_scenes", verify_ok, mode_txt)
                    self._scene_verify_result_applied = False
                    self._set_verify_running(False)
                    if self._pipeline_autorun:
                        self._pipeline_trigger_next_action()
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log_queue)

    def _collect_config(self) -> dict:
        return {
            "window_geometry": self._current_window_geometry(),
            "work_folder": self.work_folder_var.get().strip(),
            "scene_input": self.scene_input_var.get().strip(),
            "scene_detector": self.scene_detector_var.get().strip(),
            "scene_threshold": self.scene_threshold_var.get().strip(),
            "scene_backend": self.scene_backend_var.get().strip(),
            "scene_crop_mode": self.scene_crop_mode_var.get().strip(),
            "scene_crop_custom": self.scene_crop_custom_var.get().strip(),
            "scene_crop_target_h": self.scene_crop_target_h_var.get().strip(),
            "scene_layout": self.scene_layout_var.get().strip(),
            "scene_tonemap": self.scene_tonemap_var.get().strip(),
            "scene_chroma": self.scene_chroma_var.get().strip(),
            "scene_codec": self.scene_codec_var.get().strip(),
            "scene_crf": self.scene_crf_var.get().strip(),
            "scene_encoder_preset": self.scene_encoder_preset_var.get().strip(),
            "scene_pix_fmt": self.scene_pix_fmt_var.get().strip(),
            "scene_extra_ffmpeg_args": self.scene_extra_ffmpeg_args_var.get().strip(),
            "depth_mode": self.depth_mode_var.get().strip(),
            "depth_chunk_size": self.depth_chunk_size_var.get().strip(),
            "depth_overlap": self.depth_overlap_var.get().strip(),
            "depth_inference_steps": self.depth_inference_steps_var.get().strip(),
            "depth_cpu_offload": self.depth_cpu_offload_var.get().strip(),
            "depth_seed": self.depth_seed_var.get().strip(),
            "depth_guidance_scale": self.depth_guidance_scale_var.get().strip(),
            "depth_decode_chunk_size": self.depth_decode_chunk_size_var.get().strip(),
            "depth_restart_every": self.depth_restart_every_var.get().strip(),
            "depth_debug_mem": bool(self.depth_debug_mem_var.get()),
            "depth_glob": self.depth_glob_var.get().strip(),
            "depth_worker_script": self.depth_worker_script_var.get().strip(),
            "depth_scale_factor": f"{self._get_depth_scale_factor():.2f}",
            "depth_res_x": self.depth_res_x_var.get().strip(),
            "depth_res_y": self.depth_res_y_var.get().strip(),
            "depth_encode_override": bool(self.depth_encode_override_var.get()),
            "depth_codec": self.depth_codec_var.get().strip(),
            "depth_crf": self.depth_crf_var.get().strip(),
            "depth_preset": self.depth_preset_var.get().strip(),
            "depth_pix_fmt": self.depth_pix_fmt_var.get().strip(),
            "depth_extra_ffmpeg_args": self.depth_extra_ffmpeg_args_var.get().strip(),
            "depth_realesrgan_source": self.depth_realesrgan_source_var.get().strip(),
            "depth_realesrgan_workers": self.depth_realesrgan_workers_var.get().strip(),
            "splat_mode": self.splat_mode_var.get().strip(),
            "splat_batch_size": self.splat_batch_size_var.get().strip(),
            "splat_workers": self.splat_workers_var.get().strip(),
            "splat_disparity": self.splat_disparity_var.get().strip(),
            "splat_layout": self.splat_layout_var.get().strip(),
            "splat_auto_convergence": self.splat_auto_convergence_var.get().strip(),
            "splat_dilate_x": self.splat_dilate_x_var.get().strip(),
            "splat_dilate_y": self.splat_dilate_y_var.get().strip(),
            "splat_blur_x": self.splat_blur_x_var.get().strip(),
            "splat_blur_y": self.splat_blur_y_var.get().strip(),
            "splat_dilate_left": self.splat_dilate_left_var.get().strip(),
            "splat_blur_balance": self.splat_blur_balance_var.get().strip(),
            "splat_gamma": self.splat_gamma_var.get().strip(),
            "splat_convergence": self.splat_convergence_var.get().strip(),
            "splat_stair_smooth": bool(self.splat_stair_smooth_var.get()),
            "splat_stair_kernel": self.splat_stair_kernel_var.get().strip(),
            "splat_stair_x_off": self.splat_stair_x_off_var.get().strip(),
            "splat_stair_strip": self.splat_stair_strip_var.get().strip(),
            "splat_stair_strength": self.splat_stair_strength_var.get().strip(),
            "splat_replace_mask": bool(self.splat_replace_mask_var.get()),
            "splat_replace_mask_scale": self.splat_replace_mask_scale_var.get().strip(),
            "splat_replace_mask_min": self.splat_replace_mask_min_var.get().strip(),
            "splat_replace_mask_max": self.splat_replace_mask_max_var.get().strip(),
            "splat_replace_mask_gap": self.splat_replace_mask_gap_var.get().strip(),
            "splat_replace_mask_edge": bool(self.splat_replace_mask_edge_var.get()),
            "splat_encode_override": bool(self.splat_encode_override_var.get()),
            "splat_codec": self.splat_codec_var.get().strip(),
            "splat_crf": self.splat_crf_var.get().strip(),
            "splat_preset": self.splat_preset_var.get().strip(),
            "splat_pix_fmt": self.splat_pix_fmt_var.get().strip(),
            "splat_extra_ffmpeg_args": self.splat_extra_ffmpeg_args_var.get().strip(),
            "inpaint_mode": self.inpaint_mode_var.get().strip(),
            "inpaint_frames_chunk": self.inpaint_frames_chunk_var.get().strip(),
            "inpaint_cpu_offload": self.inpaint_cpu_offload_var.get().strip(),
            "inpaint_tile_num": self.inpaint_tile_num_var.get().strip(),
            "inpaint_input_bias": self.inpaint_input_bias_var.get().strip(),
            "inpaint_overlap": self.inpaint_overlap_var.get().strip(),
            "inpaint_tail_pad": self.inpaint_tail_pad_var.get().strip(),
            "inpaint_use_sharpness_csv": bool(self.inpaint_use_sharpness_csv_var.get()),
            "inpaint_sharpness_workers": self.inpaint_sharpness_workers_var.get().strip(),
            "inpaint_inference_steps": self.inpaint_inference_steps_var.get().strip(),
            "inpaint_encode_override": bool(self.inpaint_encode_override_var.get()),
            "inpaint_codec": self.inpaint_codec_var.get().strip(),
            "inpaint_crf": self.inpaint_crf_var.get().strip(),
            "inpaint_preset": self.inpaint_preset_var.get().strip(),
            "inpaint_pix_fmt": self.inpaint_pix_fmt_var.get().strip(),
            "inpaint_extra_ffmpeg_args": self.inpaint_extra_ffmpeg_args_var.get().strip(),
            "merge_mode": self.merge_mode_var.get().strip(),
            "merge_autoct_workers": self.merge_autoct_workers_var.get().strip(),
            "merge_mask_formerge_workers": self.merge_mask_formerge_workers_var.get().strip(),
            "merge_parallel": bool(self._get_merge_worker_count() >= 2),
            "merge_parallel_workers": self.merge_parallel_workers_var.get().strip(),
            "merge_use_gpu": bool(self.merge_use_gpu_var.get()),
            "merge_output_format": self.merge_output_format_var.get().strip(),
            "merge_chunks": self.merge_chunks_var.get().strip(),
            "merge_mask_binarize": self.merge_mask_binarize_var.get().strip(),
            "merge_mask_dilate": self.merge_mask_dilate_var.get().strip(),
            "merge_mask_blur": self.merge_mask_blur_var.get().strip(),
            "merge_shadow_length": self.merge_shadow_length_var.get().strip(),
            "merge_shadow_curve": self.merge_shadow_curve_var.get().strip(),
            "merge_shadow_motion_enabled": bool(self.merge_shadow_motion_enabled_var.get()),
            "merge_dynamic_shadow_width": bool(self.merge_dynamic_shadow_width_var.get()),
            "merge_use_replace_mask": bool(self.merge_use_replace_mask_var.get()),
            "merge_ct_preset": self.merge_ct_preset_var.get().strip(),
            "merge_ct_auto_mode": self.merge_ct_auto_mode_var.get().strip(),
            "merge_ct_exclude_black": bool(self.merge_ct_exclude_black_var.get()),
            "merge_encode_override": bool(self.merge_encode_override_var.get()),
            "merge_codec": self.merge_codec_var.get().strip(),
            "merge_crf": self.merge_crf_var.get().strip(),
            "merge_preset": self.merge_preset_var.get().strip(),
            "merge_pix_fmt": self.merge_pix_fmt_var.get().strip(),
            "merge_extra_ffmpeg_args": self.merge_extra_ffmpeg_args_var.get().strip(),
            "join_mode": self.join_mode_var.get().strip(),
            "join_encoder": self.join_encoder_var.get().strip(),
            "join_crf": self.join_crf_var.get().strip(),
            "join_preset": self.join_preset_var.get().strip(),
            "join_pix_fmt_override": bool(self.join_pix_fmt_override_var.get()),
            "join_pix_fmt": self.join_pix_fmt_var.get().strip(),
            "join_extra_args": self.join_extra_args_var.get().strip(),
            "scene_split_threads": self.scene_split_threads_var.get().strip(),
            "verify_scenes_workers": self.verify_scenes_workers_var.get().strip(),
            "pipeline_verify_after": self.pipeline_verify_after_var.get().strip(),
            "pipeline_test_run_files": self.pipeline_test_run_files_var.get().strip(),
            "depth_retry_policy": self._collect_retry_policy_config_from_vars(
                self.depth_retry_policy_vars
            ),
            "inpaint_retry_policy": self._collect_retry_policy_config_from_vars(
                self.inpaint_retry_policy_vars
            ),
            "resume_enabled": bool(self.resume_enabled_var.get()),
            "stop_on_error": bool(self.stop_on_error_var.get()),
            "auto_advance": bool(self.auto_advance_var.get()),
        }

    def _current_window_geometry(self) -> str:
        try:
            self.root.update_idletasks()
            return str(self.root.geometry())
        except Exception:
            return self.DEFAULT_WINDOW_GEOMETRY

    def _load_config(self) -> dict:
        if not os.path.isfile(self.CONFIG_FILENAME):
            return {}
        try:
            with open(self.CONFIG_FILENAME, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _save_config(self) -> None:
        data = self._collect_config()
        try:
            with open(self.CONFIG_FILENAME, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._restore_messagebox_wrappers()
        self._save_config()
        self._save_pipeline_state()
        self._stop_scene_detect(prompt_user=False)
        if self._depth_thread and self._depth_thread.is_alive():
            # Request graceful stop first, then immediate force stop to avoid orphan processes on GUI exit.
            self._stop_depth_placeholder(prompt_user=False)
            self._stop_depth_placeholder(prompt_user=False)
        if self._splat_thread and self._splat_thread.is_alive():
            self._stop_splat_placeholder(prompt_user=False)
            self._stop_splat_placeholder(prompt_user=False)
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            self._stop_inpaint_placeholder(prompt_user=False)
            self._stop_inpaint_placeholder(prompt_user=False)
        if self._merge_thread and self._merge_thread.is_alive():
            self._stop_merge_placeholder(prompt_user=False)
            self._stop_merge_placeholder(prompt_user=False)
        if self._join_thread and self._join_thread.is_alive():
            self._stop_join(prompt_user=False)
        if self._pipeline_test_active:
            self._restore_test_scene_subset()
        self.root.destroy()


def create_root() -> tk.Tk:
    if ThemedTk is not None:
        try:
            return ThemedTk(theme="clam")
        except Exception:
            pass
    return tk.Tk()


def main() -> None:
    root = create_root()
    PipelineMasterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
