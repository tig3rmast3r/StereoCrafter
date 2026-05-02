import argparse
import json
import math
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
from typing import Optional, Sequence
from datetime import datetime
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from dependency.ffmpeg_encoding_profiles import (
    FFMPEG_CODEC_CHOICES as SHARED_FFMPEG_CODEC_CHOICES,
    GLOBAL_ENCODER_MODE_CHOICES,
    build_validation_command,
    normalize_global_encoder_mode,
    profile_preview_line,
    resolve_color_encoding_profile,
    resolve_depth_final_grayscale_profile,
    resolve_depth_preprocess_profile,
    resolve_mask_for_merge_grayscale_profile,
    resolve_replace_mask_binary_profile,
)
from dependency.repo_paths import (
    config_path,
    repo_root,
    resolve_repo_path,
    runner_path,
    utilities_path,
)
from core.pipeline_master import builders as pm_builders
from core.pipeline_master import config as pm_config
from core.pipeline_master import orchestrator as pm_orchestrator
from core.pipeline_master import state as pm_state

try:
    from ttkthemes import ThemedTk
except Exception:
    ThemedTk = None

GUI_VERSION = "2026-05-01"
REPO_ROOT = repo_root()
DEFAULT_PIPELINE_MASTER_CONFIG_PATH = config_path("config_pipeline_master_gui.json")


class VerifyStopRequested(Exception):
    """Raised when an active verification job is explicitly stopped."""


class PipelineMasterGUI:
    CONFIG_FILENAME = str(DEFAULT_PIPELINE_MASTER_CONFIG_PATH)
    DEFAULT_SCENE_BACKEND = "OpenCV"
    DEFAULT_SCENE_CODEC = "libx264"
    DEFAULT_WINDOW_GEOMETRY = "1400x1050"
    FFMPEG_CODEC_CHOICES = SHARED_FFMPEG_CODEC_CHOICES
    FFMPEG_CODEC_ALIASES = {
        "x264": "libx264",
        "x265": "libx265",
        "h265": "libx265",
    }
    GLOBAL_ENCODER_MODE_CHOICES = GLOBAL_ENCODER_MODE_CHOICES
    DEFAULT_DEPTH_SCALE_FACTOR = 0.80
    MIN_DEPTH_SCALE_FACTOR = 0.5
    MAX_DEPTH_SCALE_FACTOR = 1.0
    DEFAULT_SPLIT_SCENES_WORKERS = 8
    DEFAULT_PIPELINE_TEST_RUN_FILES = 5
    INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS5_DEFAULT = "38"
    INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS6_DEFAULT = "26"
    INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS7_DEFAULT = "18"
    INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS_DEFAULT = "14"
    INPAINT_DYNAMIC_STATIC_MASK_DIVISOR_DEFAULT = "2.0"
    DEPTH_RUNTIME_MODE_CHOICES = ("original", "stream")
    DEPTH_RUNTIME_MODE_TO_SCRIPT = {
        "original": "./runners/depthcrafter_nogui_batch.py",
        "stream": "./runners/depthcrafter_nogui_stream_carry.py",
    }
    RETRY_POLICY_PROFILES = ("run", "retry1", "retry2", "retry3")
    RETRY_POLICY_MAX_SPLIT_CHOICES = ("off", "64", "128", "256", "512")
    RETRY_POLICY_OFFLOAD_CHOICES = ("none", "model", "sequential")
    DEPTH_RETRY_OFFSET_CHOICES = ("+20", "+15", "+10", "+5", "0", "-5", "-10", "-15", "-20")
    DEPTH_RETRY_POLICY_DEFAULT = {
        "run": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "off",
            "cpu_offload_inherited": True,
            "cpu_offload_mode": "model",
            "worker_mode": "original",
            "window_offset": "0",
            "overlap_offset": "0",
        },
        "retry1": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "off",
            "cpu_offload_inherited": False,
            "cpu_offload_mode": "model",
            "worker_mode": "original",
            "window_offset": "-10",
            "overlap_offset": "0",
        },
        "retry2": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "off",
            "cpu_offload_inherited": False,
            "cpu_offload_mode": "sequential",
            "worker_mode": "original",
            "window_offset": "-10",
            "overlap_offset": "0",
        },
        "retry3": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "off",
            "cpu_offload_inherited": False,
            "cpu_offload_mode": "model",
            "worker_mode": "stream",
            "window_offset": "+10",
            "overlap_offset": "+10",
        },
    }
    INPAINT_RETRY_POLICY_DEFAULT = {
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
            "max_split_size_mb": "256",
            "cpu_offload_inherited": False,
            "cpu_offload_mode": "model",
        },
        "retry2": {
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": "64",
            "cpu_offload_inherited": False,
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
    DEPTH_AUTO_INFO = ""
    DEPTH_MANUAL_INFO = ""
    DEPTH_STREAM_WARNING = (
        "Stream mode uses chunked streaming inference and is much less sensitive to total clip length,\n"
        "so it can often start and finish on files that Original mode cannot open at the same resolution.\n\n"
        "The output is not identical to Original mode:\n"
        "- chunk continuity is handled differently\n"
        "- the overall grayscale range is usually a bit narrower\n"
        "- the result can drift slightly from standard DepthCrafter output\n\n"
        "Use Stream when memory limits block Original mode, not when you need the closest possible match."
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
        "-tune hq -rc vbr -b:v 0 -multipass fullres -spatial_aq 1 -temporal_aq 1 "
        "-aq-strength 12 -rc-lookahead 32 -bf 3"
    )
    PIPELINE_STEPS = [
        ("scenedetect", "SceneDetect"),
        ("split_scenes", "Split Scenes"),
        ("depthcrafter", "DepthCrafter"),
        ("splatting", "Splatting"),
        ("sharpness_csv", "Sharpness CSV"),
        ("inpaint", "Inpaint"),
        ("sharpen", "Sharpen"),
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
        "splatting",
        "inpaint",
        "sharpen",
        "mask_for_merge",
        "merging",
        "mono_to_sbs",
        "join",
    }
    PIPELINE_CSV_STEPS = {"sharpness_csv", "autoct_csv"}
    PIPELINE_OPTIONAL_STEPS = {"sharpness_csv", "autoct_csv"}
    PIPELINE_VERIFY_CHOICES = ["Quick"]
    PIPELINE_STATE_FILENAME = "pipeline_state.json"
    PIPELINE_TEST_STATE_FILENAME = "pipeline_test_state.json"
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
        "inpaint_sharpen": "output-sharpen",
        "merge": "sbs",
        "join": "final",
    }

    TONEMAP_PRESET_TO_FFMPEG = {
        "Mobius (HDR style, available only for 10-bit input source)": "mobius",
        "Hable (brighter SDR style, available only for 10-bit input source)": "hable",
    }

    def __init__(
        self,
        root: tk.Tk,
        *,
        config_file: Optional[str] = None,
        work_dir_override: Optional[str] = None,
    ):
        self.root = root
        self.root.title(f"StereoCrafter Pipeline GUI {GUI_VERSION}")
        config_target = str(config_file or self.CONFIG_FILENAME).strip() or self.CONFIG_FILENAME
        self._config_file = str(Path(config_target).expanduser().resolve())
        self._work_dir_override = (
            str(Path(work_dir_override).expanduser().resolve())
            if work_dir_override
            else ""
        )
        self._config_save_ready = False
        self._config = self._load_config()
        if self._work_dir_override and not self._config.get("work_folder"):
            self._config["work_folder"] = self._work_dir_override
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
        self._scene_stop_marker_path: str = ""
        self._scene_stop_requested = False
        self._scene_stop_clicks = 0
        self._scene_active_step = "scenedetect"

        self._verify_thread: threading.Thread | None = None
        self._verify_running = False
        self._verify_mode: str = ""
        self._verify_stop_requested = False
        self._verify_stop_clicks = 0
        self._verify_processes: set[subprocess.Popen] = set()
        self._verify_processes_lock = threading.Lock()
        self._scene_verify_result_applied = False
        self._codec_validation_thread: threading.Thread | None = None
        self._codec_validation_running = False

        self._analysis_thread: threading.Thread | None = None
        self._analysis_running = False
        self._analysis_stop_requested = False
        self._analysis_process: subprocess.Popen | None = None

        self._depth_thread: threading.Thread | None = None
        self._depth_process: subprocess.Popen | None = None
        self._depth_stop_marker_path: str = ""
        self._depth_stop_requested = False
        self._depth_stop_clicks = 0
        self._splat_thread: threading.Thread | None = None
        self._splat_process: subprocess.Popen | None = None
        self._splat_stop_marker_path: str = ""
        self._splat_stop_requested = False
        self._splat_stop_clicks = 0
        self._inpaint_thread: threading.Thread | None = None
        self._inpaint_process: subprocess.Popen | None = None
        self._inpaint_stop_marker_path: str = ""
        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self._inpaint_resume_after_sharpness = False
        self._inpaint_resume_after_sharpen = False
        self._merge_thread: threading.Thread | None = None
        self._merge_process: subprocess.Popen | None = None
        self._merge_process_group_id: int | None = None
        self._merge_stop_marker_path: str = ""
        self._merge_resume_after_autoct = False
        self._merge_stop_requested = False
        self._merge_stop_clicks = 0
        self._join_thread: threading.Thread | None = None
        self._join_process: subprocess.Popen | None = None
        self._join_stop_requested = False
        self._join_stop_clicks = 0
        self._join_manual_notice_shown = False
        self._join_expected_duration_sec: float | None = None
        self._join_active_output_path: str = ""
        self._join_mark_completed = True
        self._pipeline_step_state = self._default_pipeline_step_state()
        self._pipeline_step_widgets: dict[str, dict[str, tk.Widget]] = {}
        self._pipeline_autorun = False
        self._pipeline_stop_requested = False
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
        self._pipeline_test_recovery_attempted = False
        self._pipeline_test_restore_scheduled = False
        self._pipeline_skip_notice_steps: set[str] = set()
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
        self._config_save_ready = True
        self.root.after(200, self._run_startup_tasks)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_vars(self) -> None:
        self.work_folder_var = tk.StringVar(
            value=self._config.get("work_folder", str(REPO_ROOT / "work"))
        )
        self.scene_input_var = tk.StringVar(
            value=self._config.get("scene_input", str(REPO_ROOT / "work" / "source.mkv"))
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
        self.scene_codec_var = tk.StringVar(
            value=self._config.get("scene_codec", self.DEFAULT_SCENE_CODEC)
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
        self.depth_mode_var = tk.StringVar(
            value=self._config.get("depth_mode", "Auto (recommended)")
        )
        self.depth_info_text_var = tk.StringVar(value=self.DEPTH_AUTO_INFO)
        self.depth_chunk_size_var = tk.StringVar(
            value=str(self._config.get("depth_chunk_size", "65"))
        )
        self.depth_overlap_var = tk.StringVar(
            value=str(self._config.get("depth_overlap", "15"))
        )
        self.depth_inference_steps_var = tk.StringVar(
            value=str(self._config.get("depth_inference_steps", "4"))
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
        self.depth_runtime_mode_var = tk.StringVar(
            value=self._migrate_depth_runtime_mode_from_legacy()
        )
        self.depth_worker_script_var = tk.StringVar(
            value=self._resolve_depth_worker_script(self.depth_runtime_mode_var.get())
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

        self.depth_codec_var = tk.StringVar(
            value=self._config.get("depth_codec", self.scene_codec_var.get())
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
            value=str(self._config.get("splat_workers", "8"))
        )
        self.splat_disparity_var = tk.StringVar(
            value=str(self._config.get("splat_disparity", "20"))
        )
        self.splat_layout_var = tk.StringVar(
            value=self._config.get("splat_layout", "Single Warp")
        )
        splat_auto_conv_cfg = str(
            self._config.get("splat_auto_convergence", "Min Borders")
        ).strip()
        if splat_auto_conv_cfg == "MinBorders":
            splat_auto_conv_cfg = "Min Borders"
        self.splat_auto_convergence_var = tk.StringVar(
            value=splat_auto_conv_cfg or "Min Borders"
        )
        self.splat_dilate_x_var = tk.StringVar(
            value=str(self._config.get("splat_dilate_x", "3"))
        )
        self.splat_dilate_y_var = tk.StringVar(
            value=str(self._config.get("splat_dilate_y", "1.5"))
        )
        self.splat_blur_x_var = tk.StringVar(
            value=str(self._config.get("splat_blur_x", "0"))
        )
        self.splat_blur_y_var = tk.StringVar(
            value=str(self._config.get("splat_blur_y", "0"))
        )
        self.splat_dilate_left_var = tk.StringVar(
            value=str(self._config.get("splat_dilate_left", "1"))
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

        self.splat_codec_var = tk.StringVar(
            value=self._config.get("splat_codec", self.scene_codec_var.get())
        )
        self.splat_cmd_preview_var = tk.StringVar(value="")
        self.splat_status_var = tk.StringVar(value="Ready")
        self.splat_progress_var = tk.DoubleVar(value=0.0)

        # Inpainting tab.
        self.inpaint_input_var = tk.StringVar(value="")
        self.inpaint_mask_var = tk.StringVar(value="")
        self.inpaint_output_var = tk.StringVar(value="")
        self.inpaint_sharpen_output_var = tk.StringVar(value="")
        self.inpaint_mode_var = tk.StringVar(
            value=self._config.get("inpaint_mode", "Auto (recommended)")
        )
        self.inpaint_info_text_var = tk.StringVar(value=self.INPAINT_AUTO_INFO)
        self.inpaint_frames_chunk_var = tk.StringVar(
            value=str(self._config.get("inpaint_frames_chunk", "22"))
        )
        self._inpaint_chunk_manual_cache = self.inpaint_frames_chunk_var.get().strip() or "22"
        self.inpaint_dynamic_chunk_var = tk.BooleanVar(
            value=bool(self._config.get("inpaint_dynamic_chunk", True))
        )
        self.inpaint_cpu_offload_var = tk.StringVar(
            value=self._config.get("inpaint_cpu_offload", "none")
        )
        self.inpaint_tile_mode_var = tk.StringVar(
            value=str(
                self._config.get(
                    "inpaint_tile_mode",
                    str(self._config.get("inpaint_tile_num", "1 and 2")),
                )
            )
        )
        self.inpaint_tile1_max_size_var = tk.StringVar(
            value=str(self._config.get("inpaint_tile1_max_size", "3,25,32,43,60,88"))
        )
        self.inpaint_tile2_max_size_var = tk.StringVar(
            value=str(self._config.get("inpaint_tile2_max_size", "71,86,107,117,117,117"))
        )
        self.inpaint_dynamic_visible_chunk_steps5_var = tk.StringVar(
            value=str(
                self._config.get(
                    "inpaint_dynamic_visible_chunk_steps5",
                    self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS5_DEFAULT,
                )
            )
        )
        self.inpaint_dynamic_visible_chunk_steps6_var = tk.StringVar(
            value=str(
                self._config.get(
                    "inpaint_dynamic_visible_chunk_steps6",
                    self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS6_DEFAULT,
                )
            )
        )
        self.inpaint_dynamic_visible_chunk_steps7_var = tk.StringVar(
            value=str(
                self._config.get(
                    "inpaint_dynamic_visible_chunk_steps7",
                    self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS7_DEFAULT,
                )
            )
        )
        self.inpaint_dynamic_visible_chunk_steps8_plus_var = tk.StringVar(
            value=str(
                self._config.get(
                    "inpaint_dynamic_visible_chunk_steps8_plus",
                    self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS_DEFAULT,
                )
            )
        )
        self.inpaint_dynamic_hold_divisor_var = tk.StringVar(
            value=str(
                self._config.get(
                    "inpaint_dynamic_hold_divisor",
                    self.INPAINT_DYNAMIC_STATIC_MASK_DIVISOR_DEFAULT,
                )
            )
        )
        self.inpaint_input_bias_var = tk.StringVar(
            value=str(self._config.get("inpaint_input_bias", "0"))
        )
        self.inpaint_overlap_var = tk.StringVar(
            value=str(self._config.get("inpaint_overlap", "2"))
        )
        self.inpaint_tail_pad_var = tk.StringVar(
            value=str(self._config.get("inpaint_tail_pad", "1"))
        )
        self.inpaint_use_sharpness_csv_var = tk.BooleanVar(
            value=bool(self._config.get("inpaint_use_sharpness_csv", True))
        )
        self.inpaint_dynamic_resolution_var = tk.BooleanVar(
            value=bool(self._config.get("inpaint_dynamic_resolution", True))
        )
        self.inpaint_resolution_limit_var = tk.StringVar(
            value=str(self._config.get("inpaint_resolution_limit", "90%"))
        )
        self.inpaint_use_sharpen_var = tk.BooleanVar(
            value=bool(self._config.get("inpaint_use_sharpen", True))
        )
        default_sharp_workers = "19"
        self.inpaint_sharpness_workers_var = tk.StringVar(
            value=str(self._config.get("inpaint_sharpness_workers", default_sharp_workers))
        )
        self.inpaint_sharpen_workers_var = tk.StringVar(
            value=str(self._config.get("inpaint_sharpen_workers", "19"))
        )
        self.inpaint_inference_steps_var = tk.StringVar(
            value=str(self._config.get("inpaint_inference_steps", "8"))
        )
        self.inpaint_codec_var = tk.StringVar(
            value=self._config.get("inpaint_codec", self.scene_codec_var.get())
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
                    self._config.get("merge_autoct_workers", "19"),
                )
            )
        )
        self.merge_parallel_var = tk.BooleanVar(
            value=bool(self._config.get("merge_parallel", True))
        )
        self.merge_parallel_workers_var = tk.StringVar(
            value=str(self._config.get("merge_parallel_workers", "4"))
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
            value=str(self._config.get("merge_mask_blur", "2"))
        )
        self.merge_shadow_length_var = tk.StringVar(
            value=str(self._config.get("merge_shadow_length", "15"))
        )
        self.merge_shadow_curve_var = tk.StringVar(
            value=str(self._config.get("merge_shadow_curve", "0"))
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
        self.merge_codec_var = tk.StringVar(
            value=self._config.get("merge_codec", self.scene_codec_var.get())
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
            value=str(self._config.get("join_crf", "12"))
        )
        self.join_preset_var = tk.StringVar(
            value=self._config.get("join_preset", "p7")
        )
        self.join_pix_fmt_var = tk.StringVar(
            value=self._config.get("join_pix_fmt", "yuv420p")
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
        self.global_encoder_mode_var = tk.StringVar(
            value=self._migrate_global_encoder_mode_from_legacy()
        )
        self.global_ffmpeg_extra_args_var = tk.StringVar(
            value=str(self._config.get("global_ffmpeg_extra_args", "")).strip()
        )
        self.global_encoder_preview_var = tk.StringVar(value="")
        depth_retry_cfg = self._retry_policy_from_config_key(
            "depth_retry_policy",
            self.DEPTH_RETRY_POLICY_DEFAULT,
        )
        inpaint_retry_cfg = self._retry_policy_from_config_key(
            "inpaint_retry_policy",
            self.INPAINT_RETRY_POLICY_DEFAULT,
        )
        self.depth_retry_policy_vars: dict[str, dict[str, tk.Variable]] = {}
        self.inpaint_retry_policy_vars: dict[str, dict[str, tk.Variable]] = {}
        for profile in self.RETRY_POLICY_PROFILES:
            dcfg = depth_retry_cfg.get(profile, self.DEPTH_RETRY_POLICY_DEFAULT[profile])
            icfg = inpaint_retry_cfg.get(profile, self.INPAINT_RETRY_POLICY_DEFAULT[profile])
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
                "worker_mode": tk.StringVar(
                    value=self._normalize_depth_runtime_mode(
                        dcfg.get("worker_mode", self.DEPTH_RETRY_POLICY_DEFAULT[profile].get("worker_mode", "original"))
                    )
                ),
                "window_offset": tk.StringVar(
                    value=self._normalize_depth_retry_offset(
                        dcfg.get("window_offset", self.DEPTH_RETRY_POLICY_DEFAULT[profile].get("window_offset", "0"))
                    )
                ),
                "overlap_offset": tk.StringVar(
                    value=self._normalize_depth_retry_offset(
                        dcfg.get("overlap_offset", self.DEPTH_RETRY_POLICY_DEFAULT[profile].get("overlap_offset", "0"))
                    )
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
        self._depth_retry_worker_widgets: dict[str, ttk.Combobox] = {}
        self._depth_retry_window_widgets: dict[str, tk.Widget] = {}
        self._depth_retry_overlap_widgets: dict[str, tk.Widget] = {}
        self._depth_retry_inherited_widgets: dict[str, tk.Widget] = {}
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
        if self.depth_chunk_size_var.get().strip() == "":
            self.depth_chunk_size_var.set("65")
        if self.depth_overlap_var.get().strip() == "":
            self.depth_overlap_var.set("15")
        if self.depth_inference_steps_var.get().strip() == "":
            self.depth_inference_steps_var.set("4")
        self.depth_scale_factor_var.set(
            self._normalize_depth_scale_factor(self.depth_scale_factor_var.get())
        )
        self.depth_scale_factor_text_var.set(
            f"{float(self.depth_scale_factor_var.get()):.2f}x"
        )
        if self.splat_mode_var.get().strip() not in {"Auto (recommended)", "Manual"}:
            self.splat_mode_var.set("Auto (recommended)")
        try:
            if int(self.splat_workers_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.splat_workers_var.set("8")
        if self.splat_layout_var.get().strip() not in {"Single Warp", "Dual", "Quad"}:
            self.splat_layout_var.set("Single Warp")
        auto_conv_ui = self.splat_auto_convergence_var.get().strip()
        if auto_conv_ui == "MinBorders":
            auto_conv_ui = "Min Borders"
        if auto_conv_ui not in {"Min Borders", "Average", "Peak", "Hybrid", "Off"}:
            auto_conv_ui = "Min Borders"
        self.splat_auto_convergence_var.set(auto_conv_ui)
        if self.inpaint_mode_var.get().strip() not in {"Auto (recommended)", "Manual"}:
            self.inpaint_mode_var.set("Auto (recommended)")
        if self.inpaint_mode_var.get().strip() == "Auto (recommended)":
            self.inpaint_use_sharpen_var.set(True)
        if self.inpaint_frames_chunk_var.get().strip() == "":
            self.inpaint_frames_chunk_var.set("22")
        self._inpaint_chunk_manual_cache = self.inpaint_frames_chunk_var.get().strip() or "22"
        if self.inpaint_cpu_offload_var.get().strip() == "":
            self.inpaint_cpu_offload_var.set("none")
        if self.inpaint_overlap_var.get().strip() == "":
            self.inpaint_overlap_var.set("2")
        if self.inpaint_tail_pad_var.get().strip() == "":
            self.inpaint_tail_pad_var.set("1")
        if self.inpaint_tile_mode_var.get().strip() not in {"1", "2", "1 and 2"}:
            self.inpaint_tile_mode_var.set("1 and 2")
        if self.inpaint_tile1_max_size_var.get().strip() == "":
            self.inpaint_tile1_max_size_var.set("3,25,32,43,60,88")
        if self.inpaint_tile2_max_size_var.get().strip() == "":
            self.inpaint_tile2_max_size_var.set("71,86,107,117,117,117")
        if self.inpaint_resolution_limit_var.get().strip() not in {"100%", "90%", "80%", "70%", "60%", "50%"}:
            self.inpaint_resolution_limit_var.set("90%")
        if self.inpaint_dynamic_visible_chunk_steps5_var.get().strip() == "":
            self.inpaint_dynamic_visible_chunk_steps5_var.set(
                self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS5_DEFAULT
            )
        if self.inpaint_dynamic_visible_chunk_steps6_var.get().strip() == "":
            self.inpaint_dynamic_visible_chunk_steps6_var.set(
                self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS6_DEFAULT
            )
        if self.inpaint_dynamic_visible_chunk_steps7_var.get().strip() == "":
            self.inpaint_dynamic_visible_chunk_steps7_var.set(
                self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS7_DEFAULT
            )
        if self.inpaint_dynamic_visible_chunk_steps8_plus_var.get().strip() == "":
            self.inpaint_dynamic_visible_chunk_steps8_plus_var.set(
                self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS_DEFAULT
            )
        if self.inpaint_dynamic_hold_divisor_var.get().strip() == "":
            self.inpaint_dynamic_hold_divisor_var.set(
                self.INPAINT_DYNAMIC_STATIC_MASK_DIVISOR_DEFAULT
            )
        if self.inpaint_inference_steps_var.get().strip() == "":
            self.inpaint_inference_steps_var.set("8")
        try:
            if int(self.inpaint_sharpen_workers_var.get().strip()) < 1:
                raise ValueError
        except Exception:
            self.inpaint_sharpen_workers_var.set("19")
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
            self.merge_mask_formerge_workers_var.set("19")
        if self.merge_parallel_workers_var.get().strip() == "":
            self.merge_parallel_workers_var.set("4")
        if self.merge_chunks_var.get().strip() == "":
            self.merge_chunks_var.set("20")
        if self.merge_mask_binarize_var.get().strip() == "":
            self.merge_mask_binarize_var.set("0.5")
        if self.merge_mask_dilate_var.get().strip() == "":
            self.merge_mask_dilate_var.set("2")
        if self.merge_mask_blur_var.get().strip() == "":
            self.merge_mask_blur_var.set("2")
        if self.merge_shadow_length_var.get().strip() == "":
            self.merge_shadow_length_var.set("15")
        if self.merge_shadow_curve_var.get().strip() == "":
            self.merge_shadow_curve_var.set("0")
        if not bool(self.splat_replace_mask_var.get()):
            self.splat_replace_mask_var.set(True)
        if not bool(self.merge_use_replace_mask_var.get()):
            self.merge_use_replace_mask_var.set(True)
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
            self.join_crf_var.set("12")
        if self.join_preset_var.get().strip() == "":
            self.join_preset_var.set("p7")
        if self.join_pix_fmt_var.get().strip() == "":
            self.join_pix_fmt_var.set("yuv420p")
        if self.join_extra_args_var.get().strip() == "":
            self.join_extra_args_var.set(self.JOIN_DEFAULT_ARGS)
        self.global_encoder_mode_var.set(
            normalize_global_encoder_mode(self.global_encoder_mode_var.get())
        )
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

        if self.join_mode_var.get().strip() != "Manual":
            self.join_pix_fmt_var.set("yuv420p")

        self.global_encoder_mode_var.trace_add("write", self._on_global_encoder_settings_changed)
        self.global_ffmpeg_extra_args_var.trace_add("write", self._on_global_encoder_settings_changed)
        self.scene_crop_target_h_var.trace_add("write", self._on_scene_crop_target_changed)
        self.depth_scale_factor_var.trace_add("write", self._on_depth_scale_factor_changed)
        self.depth_cpu_offload_var.trace_add("write", self._on_depth_retry_inherited_source_changed)
        self.depth_runtime_mode_var.trace_add("write", self._on_depth_retry_inherited_source_changed)
        self.depth_chunk_size_var.trace_add("write", self._on_depth_retry_inherited_source_changed)
        self.depth_overlap_var.trace_add("write", self._on_depth_retry_inherited_source_changed)

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

        ttk.Label(policy_frame, textvariable=self.scene_option_hint_var).grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )

        ffmpeg_frame = ttk.LabelFrame(parent, text="Split Encoding Args", padding=8)
        ffmpeg_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        ffmpeg_frame.grid_columnconfigure(3, weight=1)

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
        ttk.Label(
            ffmpeg_frame,
            text="Quality, pix_fmt and extra ffmpeg args are driven globally from Options and Run.",
        ).grid(row=0, column=2, columnspan=2, sticky="w")

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
            buttons, text="Verify Scenes", command=self._start_verify_quick
        )
        self.scene_verify_quick_btn.grid(row=0, column=4, padx=6)
        self.scene_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_scene_detect, state=tk.DISABLED
        )
        self.scene_stop_btn.grid(row=0, column=5, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_scene_log).grid(
            row=0, column=6, padx=6
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

        mode_frame = ttk.LabelFrame(parent, text="Depth Mode", padding=8)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
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

        ttk.Label(mode_frame, text="Runtime:").grid(row=0, column=2, sticky="w")
        self.depth_runtime_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.depth_runtime_mode_var,
            values=self.DEPTH_RUNTIME_MODE_CHOICES,
            width=12,
            state="readonly",
        )
        self.depth_runtime_mode_combo.grid(row=0, column=3, sticky="w", padx=(6, 0))
        self.depth_runtime_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_depth_runtime_mode_selected
        )

        ttk.Label(
            mode_frame,
            textvariable=self.depth_info_text_var,
            justify="left",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        params_frame = ttk.LabelFrame(parent, text="Depth Parameters", padding=8)
        params_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
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

        encode_frame = ttk.LabelFrame(parent, text="Encoding", padding=8)
        encode_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=0, sticky="w")
        self.depth_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.depth_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.depth_codec_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.depth_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 6))
        self.depth_preview_btn = ttk.Button(
            buttons, text="Preview Command", command=self._preview_depth_command
        )
        self.depth_preview_btn.grid(row=0, column=0, padx=(0, 6))
        self.depth_run_btn = ttk.Button(
            buttons, text="Run DepthCrafter", command=self._run_depth_placeholder
        )
        self.depth_run_btn.grid(row=0, column=1, padx=6)
        self.depth_verify_quick_btn = ttk.Button(
            buttons, text="Verify Depth", command=self._start_depth_verify_quick
        )
        self.depth_verify_quick_btn.grid(row=0, column=2, padx=6)
        self.depth_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_depth_placeholder
        )
        self.depth_stop_btn.grid(row=0, column=3, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_depth_log).grid(
            row=0, column=4, padx=6
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
            values=["Min Borders", "Average", "Peak", "Hybrid", "Off"],
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

        encode_frame = ttk.LabelFrame(parent, text="Encoding", padding=8)
        encode_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=0, sticky="w")
        self.splat_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.splat_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.splat_codec_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(
            encode_frame,
            text="Quality and ffmpeg args are driven globally from Options and Run.",
        ).grid(row=0, column=2, columnspan=2, sticky="w")

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
            buttons, text="Verify Scenes", command=self._start_splat_verify_quick
        )
        self.splat_verify_quick_btn.grid(row=0, column=2, padx=6)
        self.splat_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_splat_placeholder, state=tk.DISABLED
        )
        self.splat_stop_btn.grid(row=0, column=3, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_splat_log).grid(
            row=0, column=4, padx=6
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
        self._preview_splat_command()
        self._set_splat_running(False)

    def _build_inpaint_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(12, weight=1)
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

        ttk.Label(parent, text="Sharpen Output:").grid(
            row=3, column=0, sticky="w", pady=3
        )
        ttk.Entry(parent, textvariable=self.inpaint_sharpen_output_var, state="readonly").grid(
            row=3, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="Open", command=self._open_inpaint_sharpen_output_folder).grid(
            row=3, column=2, padx=4
        )

        mode_frame = ttk.LabelFrame(parent, text="Inpainting Mode", padding=8)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
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

        params_container = ttk.Frame(parent)
        params_container.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        params_container.grid_columnconfigure(0, weight=3)
        params_container.grid_columnconfigure(1, weight=2)

        params_frame = ttk.LabelFrame(params_container, text="Inpainting Parameters", padding=8)
        params_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        for col in range(10):
            params_frame.grid_columnconfigure(col, weight=0)
        params_frame.grid_columnconfigure(9, weight=1)

        ttk.Label(params_frame, text="Chunk Size:").grid(row=0, column=0, sticky="w")
        self.inpaint_frames_chunk_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_frames_chunk_var, width=8
        )
        self.inpaint_frames_chunk_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))

        self.inpaint_dynamic_chunk_check = ttk.Checkbutton(
            params_frame,
            text="Enable Dynamic Chunk",
            variable=self.inpaint_dynamic_chunk_var,
            command=self._on_inpaint_dynamic_chunk_toggle,
        )
        self.inpaint_dynamic_chunk_check.grid(row=0, column=2, columnspan=2, sticky="w")

        ttk.Label(params_frame, text="CPU offload:").grid(row=0, column=4, sticky="w")
        self.inpaint_cpu_offload_combo = ttk.Combobox(
            params_frame,
            textvariable=self.inpaint_cpu_offload_var,
            values=["none", "model", "sequential"],
            width=12,
            state="readonly",
        )
        self.inpaint_cpu_offload_combo.grid(row=0, column=5, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Res / Max Res:").grid(row=0, column=6, sticky="w")
        self.inpaint_resolution_limit_combo = ttk.Combobox(
            params_frame,
            textvariable=self.inpaint_resolution_limit_var,
            values=["100%", "90%", "80%", "70%", "60%", "50%"],
            width=8,
            state="readonly",
        )
        self.inpaint_resolution_limit_combo.grid(row=0, column=7, sticky="w", padx=(6, 12))

        ttk.Label(params_frame, text="Tile:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.inpaint_tile_mode_combo = ttk.Combobox(
            params_frame,
            textvariable=self.inpaint_tile_mode_var,
            values=["1", "2", "1 and 2"],
            width=10,
            state="readonly",
        )
        self.inpaint_tile_mode_combo.grid(row=1, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Tile 1 Max Size:").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.inpaint_tile1_max_size_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_tile1_max_size_var, width=20
        )
        self.inpaint_tile1_max_size_entry.grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Tile 2 Max Size:").grid(row=1, column=4, sticky="w", pady=(8, 0))
        self.inpaint_tile2_max_size_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_tile2_max_size_var, width=20
        )
        self.inpaint_tile2_max_size_entry.grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Overlap:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.inpaint_overlap_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_overlap_var, width=8
        )
        self.inpaint_overlap_entry.grid(row=2, column=1, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="TailPad:").grid(row=2, column=2, sticky="w", pady=(8, 0))
        self.inpaint_tail_pad_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_tail_pad_var, width=8
        )
        self.inpaint_tail_pad_entry.grid(row=2, column=3, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Input Bias:").grid(row=2, column=4, sticky="w", pady=(8, 0))
        self.inpaint_input_bias_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_input_bias_var, width=8
        )
        self.inpaint_input_bias_entry.grid(row=2, column=5, sticky="w", padx=(6, 12), pady=(8, 0))

        ttk.Label(params_frame, text="Inference steps:").grid(row=2, column=6, sticky="w", pady=(8, 0))
        self.inpaint_inference_steps_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_inference_steps_var, width=8
        )
        self.inpaint_inference_steps_entry.grid(row=2, column=7, sticky="w", padx=(6, 12), pady=(8, 0))

        self.inpaint_use_sharpness_check = ttk.Checkbutton(
            params_frame,
            text="Use sharpness CSV (auto steps)",
            variable=self.inpaint_use_sharpness_csv_var,
            command=self._on_inpaint_auto_steps_toggle,
        )
        self.inpaint_use_sharpness_check.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.inpaint_dynamic_resolution_check = ttk.Checkbutton(
            params_frame,
            text="Dynamic resolution",
            variable=self.inpaint_dynamic_resolution_var,
            command=self._on_inpaint_dynamic_resolution_toggle,
        )
        self.inpaint_dynamic_resolution_check.grid(row=3, column=4, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(params_frame, text="Sharpness CSV workers:").grid(
            row=3, column=6, sticky="w", pady=(8, 0)
        )
        self.inpaint_sharpness_workers_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_sharpness_workers_var, width=8
        )
        self.inpaint_sharpness_workers_entry.grid(
            row=3, column=7, sticky="w", padx=(6, 12), pady=(8, 0)
        )

        self.inpaint_sharpen_check = ttk.Checkbutton(
            params_frame,
            text="Sharpen",
            variable=self.inpaint_use_sharpen_var,
            command=self._on_inpaint_sharpen_toggle,
        )
        self.inpaint_sharpen_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(params_frame, text="Sharpen workers:").grid(
            row=4, column=2, sticky="w", pady=(8, 0)
        )
        self.inpaint_sharpen_workers_entry = ttk.Entry(
            params_frame, textvariable=self.inpaint_sharpen_workers_var, width=8
        )
        self.inpaint_sharpen_workers_entry.grid(
            row=4, column=3, sticky="w", padx=(6, 12), pady=(8, 0)
        )

        dynamic_frame = ttk.LabelFrame(params_container, text="Dynamic Chunk Settings", padding=8)
        dynamic_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        dynamic_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(dynamic_frame, text="Chunk @ 3.0 steps:").grid(row=0, column=0, sticky="w")
        self.inpaint_dynamic_visible_chunk_steps5_entry = ttk.Entry(
            dynamic_frame,
            textvariable=self.inpaint_dynamic_visible_chunk_steps5_var,
            width=8,
        )
        self.inpaint_dynamic_visible_chunk_steps5_entry.grid(
            row=0, column=1, sticky="w", padx=(6, 0)
        )

        ttk.Label(dynamic_frame, text="Chunk @ 4.0 steps:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.inpaint_dynamic_visible_chunk_steps6_entry = ttk.Entry(
            dynamic_frame,
            textvariable=self.inpaint_dynamic_visible_chunk_steps6_var,
            width=8,
        )
        self.inpaint_dynamic_visible_chunk_steps6_entry.grid(
            row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(dynamic_frame, text="Chunk @ 5.0 steps:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.inpaint_dynamic_visible_chunk_steps7_entry = ttk.Entry(
            dynamic_frame,
            textvariable=self.inpaint_dynamic_visible_chunk_steps7_var,
            width=8,
        )
        self.inpaint_dynamic_visible_chunk_steps7_entry.grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(dynamic_frame, text="Chunk @ 6.0+ steps:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.inpaint_dynamic_visible_chunk_steps8_plus_entry = ttk.Entry(
            dynamic_frame,
            textvariable=self.inpaint_dynamic_visible_chunk_steps8_plus_var,
            width=8,
        )
        self.inpaint_dynamic_visible_chunk_steps8_plus_entry.grid(
            row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(dynamic_frame, text="Static Mask Divisor:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.inpaint_dynamic_hold_divisor_entry = ttk.Entry(
            dynamic_frame,
            textvariable=self.inpaint_dynamic_hold_divisor_var,
            width=8,
        )
        self.inpaint_dynamic_hold_divisor_entry.grid(
            row=4, column=1, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        encode_frame = ttk.LabelFrame(parent, text="Encoding", padding=8)
        encode_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=0, sticky="w")
        self.inpaint_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.inpaint_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.inpaint_codec_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(
            encode_frame,
            text="Quality and ffmpeg args are driven globally from Options and Run.",
        ).grid(row=0, column=2, columnspan=2, sticky="w")

        cmd_frame = ttk.LabelFrame(parent, text="Command Preview", padding=8)
        cmd_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=6)
        cmd_frame.grid_columnconfigure(0, weight=1)
        ttk.Entry(cmd_frame, textvariable=self.inpaint_cmd_preview_var, state="readonly").grid(
            row=0, column=0, sticky="ew"
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 6))
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
        self.inpaint_benchmark_btn = ttk.Button(
            buttons, text="Benchmark", command=self._start_inpaint_tile_benchmark
        )
        self.inpaint_benchmark_btn.grid(row=0, column=3, padx=6)
        self.inpaint_sharpen_run_btn = ttk.Button(
            buttons, text="Run Sharpen", command=self._start_inpaint_sharpen
        )
        self.inpaint_sharpen_run_btn.grid(row=0, column=4, padx=6)
        self.inpaint_verify_quick_btn = ttk.Button(
            buttons, text="Verify Scenes", command=self._start_inpaint_verify_quick
        )
        self.inpaint_verify_quick_btn.grid(row=0, column=5, padx=6)
        self.inpaint_sharpen_verify_quick_btn = ttk.Button(
            buttons, text="Verify Sharpen", command=self._start_inpaint_sharpen_verify_quick
        )
        self.inpaint_sharpen_verify_quick_btn.grid(row=0, column=6, padx=6)
        self.inpaint_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_inpaint_placeholder, state=tk.DISABLED
        )
        self.inpaint_stop_btn.grid(row=0, column=7, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_inpaint_log).grid(
            row=0, column=8, padx=6
        )

        status_frame = ttk.Frame(parent)
        status_frame.grid(row=9, column=0, columnspan=3, sticky="ew")
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
        log_frame.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.inpaint_log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self.inpaint_log_text.grid(row=0, column=0, sticky="nsew")
        iscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.inpaint_log_text.yview)
        iscroll.grid(row=0, column=1, sticky="ns")
        self.inpaint_log_text.configure(yscrollcommand=iscroll.set)
        self.inpaint_resolution_limit_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._preview_inpaint_command(),
        )

        self._inpaint_manual_widgets = [
            self.inpaint_input_bias_entry,
            self.inpaint_overlap_entry,
            self.inpaint_tail_pad_entry,
            self.inpaint_use_sharpness_check,
            self.inpaint_dynamic_resolution_check,
        ]

        self._on_inpaint_mode_changed()
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

    def _open_inpaint_sharpen_output_folder(self) -> None:
        folder = self.inpaint_sharpen_output_var.get().strip()
        if not folder:
            return
        os.makedirs(folder, exist_ok=True)
        self._append_inpaint_log(f"Sharpen output folder ready: {folder}")

    def _get_inpaint_chunk_manual_value(self) -> str:
        raw = self.inpaint_frames_chunk_var.get().strip()
        if raw and raw.upper() != "N/A":
            self._inpaint_chunk_manual_cache = raw
            return raw
        return str(self._inpaint_chunk_manual_cache or "22")

    def _set_inpaint_chunk_entry_na(self) -> None:
        raw = self.inpaint_frames_chunk_var.get().strip()
        if raw and raw.upper() != "N/A":
            self._inpaint_chunk_manual_cache = raw
        self.inpaint_frames_chunk_var.set("N/A")

    def _restore_inpaint_chunk_entry_value(self) -> None:
        if self.inpaint_frames_chunk_var.get().strip().upper() == "N/A":
            self.inpaint_frames_chunk_var.set(self._get_inpaint_chunk_manual_value())

    @staticmethod
    def _parse_inpaint_positive_int(raw: str, label: str) -> int:
        value = int(str(raw).strip())
        if value < 1:
            raise ValueError(f"{label} must be >= 1.")
        return value

    @staticmethod
    def _parse_inpaint_nonnegative_int(raw: str, label: str) -> int:
        value = int(str(raw).strip())
        if value < 0:
            raise ValueError(f"{label} must be >= 0.")
        return value

    @staticmethod
    def _parse_inpaint_positive_float(raw: str, label: str) -> float:
        value = float(str(raw).strip())
        if value <= 0.0:
            raise ValueError(f"{label} must be > 0.")
        return value

    @staticmethod
    def _parse_inpaint_bounded_float(
        raw: str,
        label: str,
        *,
        min_value: float,
        max_value: float,
    ) -> float:
        value = float(str(raw).strip())
        if value < min_value or value > max_value:
            raise ValueError(f"{label} must be between {min_value} and {max_value}.")
        return value

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
        self.inpaint_dynamic_chunk_var.set(True)
        self.inpaint_tile_mode_var.set("1 and 2")
        self.inpaint_input_bias_var.set("0")
        self.inpaint_overlap_var.set("2")
        self.inpaint_tail_pad_var.set("1")
        self.inpaint_use_sharpness_csv_var.set(True)
        self.inpaint_dynamic_resolution_var.set(True)
        self.inpaint_resolution_limit_var.set("90%")
        self.inpaint_use_sharpen_var.set(True)
        self.inpaint_inference_steps_var.set("8")
        self.inpaint_dynamic_visible_chunk_steps5_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS5_DEFAULT
        )
        self.inpaint_dynamic_visible_chunk_steps6_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS6_DEFAULT
        )
        self.inpaint_dynamic_visible_chunk_steps7_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS7_DEFAULT
        )
        self.inpaint_dynamic_visible_chunk_steps8_plus_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS_DEFAULT
        )
        self.inpaint_dynamic_hold_divisor_var.set(
            self.INPAINT_DYNAMIC_STATIC_MASK_DIVISOR_DEFAULT
        )

    def _on_inpaint_auto_steps_toggle(self) -> None:
        self._apply_inpaint_control_states()

    def _on_inpaint_dynamic_chunk_toggle(self) -> None:
        self._apply_inpaint_control_states()

    def _on_inpaint_dynamic_resolution_toggle(self) -> None:
        self._apply_inpaint_control_states()

    def _on_inpaint_sharpen_toggle(self) -> None:
        self._apply_inpaint_control_states()

    def _sharpen_step_enabled_in_current_mode(self) -> bool:
        return bool(self.inpaint_use_sharpen_var.get())

    def _apply_inpaint_control_states(self) -> None:
        mode_manual = self.inpaint_mode_var.get().strip() == "Manual"
        dynamic_chunk = bool(self.inpaint_dynamic_chunk_var.get())

        self.inpaint_cpu_offload_combo.configure(state="readonly")

        manual_state = tk.NORMAL if mode_manual else tk.DISABLED
        for widget in getattr(self, "_inpaint_manual_widgets", []):
            widget.configure(state=manual_state)

        if not mode_manual:
            self.inpaint_dynamic_chunk_var.set(True)
            self.inpaint_dynamic_resolution_var.set(True)
            self.inpaint_resolution_limit_var.set("90%")
            dynamic_chunk = True

        if dynamic_chunk:
            self._set_inpaint_chunk_entry_na()
            self.inpaint_frames_chunk_entry.configure(state=tk.DISABLED)
        else:
            self._restore_inpaint_chunk_entry_value()
            self.inpaint_frames_chunk_entry.configure(state=tk.NORMAL if mode_manual else tk.DISABLED)

        self.inpaint_dynamic_chunk_check.configure(state=tk.DISABLED if not mode_manual else tk.NORMAL)
        self.inpaint_resolution_limit_combo.configure(state=tk.DISABLED if not mode_manual else "readonly")
        self.inpaint_tile_mode_combo.configure(state=tk.DISABLED if not mode_manual else "readonly")
        self.inpaint_tile1_max_size_entry.configure(state=tk.NORMAL)
        self.inpaint_tile2_max_size_entry.configure(state=tk.NORMAL)
        dynamic_settings_state = tk.NORMAL if (mode_manual and dynamic_chunk) else tk.DISABLED
        for widget in (
            getattr(self, "inpaint_dynamic_visible_chunk_steps5_entry", None),
            getattr(self, "inpaint_dynamic_visible_chunk_steps6_entry", None),
            getattr(self, "inpaint_dynamic_visible_chunk_steps7_entry", None),
            getattr(self, "inpaint_dynamic_visible_chunk_steps8_plus_entry", None),
            getattr(self, "inpaint_dynamic_hold_divisor_entry", None),
        ):
            if widget is not None:
                widget.configure(state=dynamic_settings_state)

        use_auto_steps = bool(self.inpaint_use_sharpness_csv_var.get())
        if mode_manual and not use_auto_steps:
            self.inpaint_inference_steps_entry.configure(state=tk.NORMAL)
        else:
            self.inpaint_inference_steps_entry.configure(state=tk.DISABLED)

        sharpen_manual_state = tk.NORMAL if mode_manual else tk.DISABLED
        self.inpaint_sharpen_check.configure(state=sharpen_manual_state)
        sharpen_workers_state = tk.NORMAL if self._sharpen_step_enabled_in_current_mode() else tk.DISABLED
        self.inpaint_sharpen_workers_entry.configure(state=sharpen_workers_state)
        sharpen_buttons_state = tk.NORMAL if self._sharpen_step_enabled_in_current_mode() else tk.DISABLED
        self.inpaint_sharpen_run_btn.configure(state=sharpen_buttons_state)
        if not (self._inpaint_thread and self._inpaint_thread.is_alive()) and not self._verify_running:
            self.inpaint_sharpen_verify_quick_btn.configure(state=sharpen_buttons_state)

        self._update_replace_mask_dependent_controls()
        self._preview_inpaint_command()
        self._refresh_pipeline_status_panel()

    def _build_inpaint_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        return pm_builders.build_inpaint_runner_payload(self)

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

        launcher_script = runner_path("run_inpainting_runner.sh")
        if not launcher_script.is_file():
            messagebox.showerror("Inpainting", f"Launcher not found:\n{launcher_script}")
            return

        runner_script = resolve_repo_path(
            env_updates.get("RUNNER", str(runner_path("batch_inpainting_runner.py")))
        )
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
        self._inpaint_stop_marker_path = env_updates.get("STOP_MARKER", "").strip()
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

    def _start_inpaint_tile_benchmark(self) -> None:
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Benchmark", "Another inpainting task is running.")
            return
        if self._verify_running:
            messagebox.showinfo("Benchmark", "Stop verification before benchmarking.")
            return
        script_path = utilities_path("benchmark_inpaint_tile1_vram.py")
        if not script_path.is_file():
            messagebox.showerror("Benchmark", f"Script not found:\n{script_path}")
            return
        input_dir = self.inpaint_input_var.get().strip()
        mask_dir = self.inpaint_mask_var.get().strip()
        sample_input = ""
        sample_mask = ""
        if input_dir and os.path.isdir(input_dir):
            sample_files = self._collect_video_files_for_patterns(input_dir, self.VERIFY_VIDEO_PATTERNS)
            for sample in sample_files:
                mask_hits = sorted(glob.glob(os.path.join(mask_dir, f"{sample.stem}_replace_mask.*"))) if mask_dir else []
                if mask_hits:
                    sample_input = str(sample)
                    sample_mask = mask_hits[0]
                    break
            if not sample_input and sample_files:
                sample_input = str(sample_files[0])
        bench_cfg_dir = repo_root() / "configs"
        bench_cfg_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = bench_cfg_dir / "inpaint_tile1_chunk_benchmark_raw.csv"
        json_out = bench_cfg_dir / "inpaint_tile1_chunk_benchmark.json"
        if json_out.is_file():
            rerun = messagebox.askyesno(
                "Inpaint Tile Benchmark",
                (
                    "A tile benchmark already exists.\n\n"
                    "Yes: rerun the benchmark.\n"
                    "No: apply the safe Tile 1/Tile 2 values from the last benchmark."
                ),
            )
            if not rerun:
                try:
                    self._apply_inpaint_tile_benchmark_results(json_out)
                    self.inpaint_status_var.set("Benchmark values restored")
                except Exception as exc:
                    messagebox.showerror(
                        "Inpaint Tile Benchmark",
                        f"Could not apply the previous benchmark values:\n{exc}",
                    )
                return
        warning = (
            "This benchmark can take a long time, but it normally only needs to be "
            "run once per machine/GPU.\n\n"
            "Before starting, close other GPU-heavy apps and make sure VRAM is as "
            "free as possible. The benchmark intentionally keeps increasing chunk "
            "size until it finds the real OOM boundary."
        )
        if not messagebox.askyesno("Inpaint Tile Benchmark", warning + "\n\nStart benchmark now?"):
            return
        cmd = [
            sys.executable,
            str(script_path),
            "--out-csv",
            str(raw_csv),
            "--json-out",
            str(json_out),
            "--adaptive-oom",
            "--adaptive-scales",
            "100,90,80,70,60,50",
            "--adaptive-tiles",
            "1,2",
            "--adaptive-start-chunk-tile1",
            "20",
            "--adaptive-start-chunk-tile2",
            "50",
            "--adaptive-step",
            "5",
            "--tile-num",
            "1",
            "--steps",
            "1",
            "--tail-pad",
            self.inpaint_tail_pad_var.get().strip() or "1",
            "--overlap",
            self.inpaint_overlap_var.get().strip() or "2",
            "--include-decode",
            "--synthetic-mask-ratio",
            "0.20",
            "--offload-type",
            self.inpaint_cpu_offload_var.get().strip() or "none",
            "--retry-policy-json",
            self._build_retry_policy_json(
                self.inpaint_retry_policy_vars,
                self.inpaint_cpu_offload_var.get().strip() or "none",
            ),
            "--allocator-profile",
            "run",
        ]
        if sample_input:
            cmd.extend(["--sample-input", sample_input])
        if sample_mask:
            cmd.extend(["--sample-mask", sample_mask])
        elif mask_dir:
            cmd.extend(["--sample-mask-folder", mask_dir])
        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self.inpaint_status_var.set("Running inpaint tile benchmark...")
        self.inpaint_progress_var.set(0.0)
        self._set_inpaint_running(True)
        self._append_inpaint_log("=== Inpaint tile benchmark started ===")
        if sample_input:
            self._append_inpaint_log(f"[BENCH] sample input: {sample_input}")
            self._append_inpaint_log(f"[BENCH] sample mask: {sample_mask or '(auto/none)'}")
        else:
            self._append_inpaint_log("[BENCH] no sample input found; using synthetic 20% mask benchmark")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._inpaint_stop_marker_path = ""
        self._inpaint_thread = threading.Thread(
            target=self._run_inpaint_benchmark_worker,
            args=(cmd, str(json_out)),
            daemon=True,
        )
        self._inpaint_thread.start()

    def _run_inpaint_benchmark_worker(self, cmd: list[str], json_out: str) -> None:
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
                    self._log_queue.put(("inpaint_line", f"[BENCH] {line}"))
            rc = proc.wait()
            if self._inpaint_stop_requested:
                self._log_queue.put(("inpaint_status", "Benchmark stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("inpaint_status", "Benchmark completed"))
                self._log_queue.put(("inpaint_progress", "100"))
                self._log_queue.put(("inpaint_line", f"[BENCH] output: {json_out}"))
            else:
                self._log_queue.put(("inpaint_status", f"Benchmark failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("inpaint_line", f"[BENCH][ERROR] {exc}"))
            self._log_queue.put(("inpaint_status", "Benchmark failed"))
        finally:
            self._inpaint_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("inpaint_done", {"step": "benchmark", "success": step_success, "json_out": json_out}))

    @staticmethod
    def _format_benchmark_chunk_values(raw_values: object) -> str:
        if not isinstance(raw_values, list):
            raise ValueError("Expected a list of six chunk values.")
        values: list[str] = []
        for raw in raw_values[:6]:
            value = int(float(raw))
            if value < 1:
                raise ValueError("Chunk values must be positive.")
            values.append(str(value))
        if not values:
            raise ValueError("No chunk values found.")
        while len(values) < 6:
            values.append(values[-1])
        return ",".join(values)

    def _apply_inpaint_tile_benchmark_results(self, json_path: object) -> None:
        path = Path(str(json_path or "")).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Benchmark JSON is not an object.")
        tile1_values = self._format_benchmark_chunk_values(payload.get("tile1_max_chunks"))
        tile2_values = self._format_benchmark_chunk_values(payload.get("tile2_max_chunks"))
        self.inpaint_tile1_max_size_var.set(tile1_values)
        self.inpaint_tile2_max_size_var.set(tile2_values)
        self._save_config()
        self._preview_inpaint_command()
        self._append_inpaint_log(
            f"[BENCH] Applied safe defaults from benchmark: tile1={tile1_values} tile2={tile2_values}"
        )
        self._append_inpaint_log("[BENCH] Runtime uses the visible GUI values; manual edits override the benchmark.")

    def _start_inpaint_sharpness_csv(self, resume_inpaint_after: bool = False) -> None:
        # Enable auto-resume only when Sharpness CSV is launched as Inpaint preflight.
        self._inpaint_resume_after_sharpness = False
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Inpainting", "Another inpainting task is running.")
            return
        if self._verify_running:
            messagebox.showinfo("Inpainting", "Stop verification before creating sharpness CSV.")
            return

        script_path = utilities_path("analyze_inpaint_sharpness.py")
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
        stop_marker = str(out_csv.parent / ".stop_after_current")
        cmd.extend(["--stop-marker", stop_marker])

        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self.inpaint_status_var.set("Creating sharpness CSV...")
        self.inpaint_progress_var.set(0.0)
        self._set_inpaint_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("sharpness_csv")
        self._append_inpaint_log("=== Sharpness CSV creation started ===")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._inpaint_stop_marker_path = stop_marker
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

    def _build_inpaint_sharpen_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        return pm_builders.build_inpaint_sharpen_runner_payload(self)

    def _start_inpaint_sharpen(self, resume_after_sharpness: bool = False) -> None:
        self._inpaint_resume_after_sharpen = False
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Sharpen", "Another inpainting/sharpen task is running.")
            return
        if self._verify_running:
            messagebox.showinfo("Sharpen", "Stop verification before running Sharpen.")
            return
        if not self._sharpen_step_enabled_in_current_mode():
            messagebox.showinfo("Sharpen", "Sharpen is disabled for the current Inpainting settings.")
            return
        try:
            cmd, env_updates, _preview = self._build_inpaint_sharpen_runner_payload()
        except Exception as exc:
            messagebox.showerror("Sharpen", f"Invalid sharpen options:\n{exc}")
            return

        launcher_script = runner_path("run_inpaint_sharpen_runner.sh")
        if not launcher_script.is_file():
            messagebox.showerror("Sharpen", f"Launcher not found:\n{launcher_script}")
            return
        runner_script = resolve_repo_path(
            env_updates.get("RUNNER", str(runner_path("batch_inpaint_sharpen_runner.py")))
        )
        if not runner_script.is_file():
            messagebox.showerror("Sharpen", f"Runner not found:\n{runner_script}")
            return

        input_dir = self.inpaint_output_var.get().strip()
        mask_dir = self.inpaint_mask_var.get().strip()
        output_dir = self.inpaint_sharpen_output_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Sharpen", f"Inpaint output folder not found:\n{input_dir or '(empty)'}")
            return
        if not mask_dir or not os.path.isdir(mask_dir):
            messagebox.showerror("Sharpen", f"Mask folder not found:\n{mask_dir or '(empty)'}")
            return
        if not output_dir:
            messagebox.showerror("Sharpen", "Sharpen output folder is required.")
            return
        os.makedirs(output_dir, exist_ok=True)

        mask_supported = self._is_splat_replace_mask_active()
        has_masks = self._has_any_replace_masks(mask_dir)
        if not mask_supported or not has_masks:
            messagebox.showerror(
                "Sharpen",
                (
                    "Sharpen requires replace masks exported in Splatting.\n"
                    "Enable Replace mask in Splatting and ensure mask files exist in work/mask."
                ),
            )
            return

        sharp_csv_path = self.inpaint_sharpness_csv_var.get().strip()
        if not sharp_csv_path:
            sharp_csv_path = str(
                Path(self.work_folder_var.get().strip() or "./work").resolve() / "sharpness.csv"
            )
            self.inpaint_sharpness_csv_var.set(sharp_csv_path)

        sharp_ok, sharp_msg, sharp_missing = self._verify_sharpness_csv_coverage(
            self.inpaint_input_var.get().strip(),
            sharp_csv_path,
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
                more = "" if len(sharp_missing) <= 12 else f"\n... and {len(sharp_missing) - 12} more"
                self._append_inpaint_log(f"[SHARP][VERIFY] Missing rows:\n{preview}{more}")
            messagebox.showwarning(
                "Sharpen",
                (
                    f"{sharp_msg}\n\n"
                    "Sharpness CSV will be regenerated now before Sharpen."
                ),
            )
            self.inpaint_status_var.set("Sharpness CSV incomplete, rebuilding...")
            self._inpaint_resume_after_sharpen = True
            self._start_inpaint_sharpness_csv(resume_inpaint_after=False)
            return

        expected_sharpen, sharpen_err = self._collect_expected_sharpen_outputs(
            self.inpaint_input_var.get().strip(),
            sharp_csv_path,
        )
        if sharpen_err:
            messagebox.showerror("Sharpen", sharpen_err)
            return
        if not expected_sharpen:
            self._append_inpaint_log("[SHARPEN] 0 eligible scenes (sharpness <= 7 for all inputs).")
            self.inpaint_status_var.set("Sharpen completed (0 eligible scenes)")
            self.inpaint_progress_var.set(100.0)
            self._pipeline_on_run_finished("sharpen", True)
            return

        self._inpaint_resume_after_sharpen = False
        self._inpaint_stop_requested = False
        self._inpaint_stop_clicks = 0
        self.inpaint_status_var.set("Running Sharpen...")
        self.inpaint_progress_var.set(0.0)
        self._set_inpaint_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("sharpen")
        self._append_inpaint_log("=== Sharpen started ===")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._append_inpaint_log(
            "ENV: " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        )
        self._inpaint_stop_marker_path = env_updates.get("STOP_MARKER", "").strip()
        self._inpaint_thread = threading.Thread(
            target=self._run_inpaint_sharpen_worker,
            args=(cmd, env_updates),
            daemon=True,
        )
        self._inpaint_thread.start()

    def _run_inpaint_sharpen_worker(self, cmd: list[str], env_updates: dict[str, str]) -> None:
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
                    self._log_queue.put(("inpaint_line", f"[SHARPEN] {line}"))
                    self._try_parse_inpaint_progress(line)
                if self._inpaint_stop_requested:
                    break
            rc = proc.wait()
            if self._inpaint_stop_requested:
                self._log_queue.put(("inpaint_status", "Sharpen stopped by user"))
            elif rc == 0:
                step_success = True
                self._log_queue.put(("inpaint_status", "Sharpen completed"))
                self._log_queue.put(("inpaint_progress", "100"))
            else:
                self._log_queue.put(("inpaint_status", f"Sharpen failed (exit {rc})"))
        except Exception as exc:
            self._log_queue.put(("inpaint_line", f"[SHARPEN][ERROR] {exc}"))
            self._log_queue.put(("inpaint_status", "Sharpen failed"))
        finally:
            self._inpaint_process = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            self._log_queue.put(("inpaint_done", {"step": "sharpen", "success": step_success}))

    def _preferred_inpainted_dir_for_consumers(self) -> str:
        if not self._sharpen_step_enabled_in_current_mode():
            return ""
        return self.inpaint_sharpen_output_var.get().strip()

    def _prepare_named_verify_subset_dir(
        self,
        folder: str,
        tag: str,
        names: Sequence[str],
    ) -> str:
        resolved = str(Path(folder).resolve()) if folder else str(folder)
        if not resolved or not os.path.isdir(resolved):
            return resolved
        wanted = sorted({str(x).strip() for x in (names or []) if str(x).strip()})
        if not wanted:
            return resolved
        if self._pipeline_test_active:
            root_dir = Path(self._pipeline_test_dir or "")
            if not root_dir.is_dir():
                return resolved
            subset_root = root_dir / "_verify_subset_named"
        else:
            subset_root = Path(self.work_folder_var.get().strip() or "./work").resolve() / ".verify_subset_named"
        safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(tag).strip() or "verify")
        subset_dir = subset_root / safe_tag
        try:
            if subset_dir.exists():
                shutil.rmtree(subset_dir, ignore_errors=True)
            subset_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return resolved
        linked = 0
        for name in wanted:
            src = Path(resolved) / name
            if src.is_file() and self._pipeline_link_or_copy_file(src, subset_dir / src.name):
                linked += 1
        if linked <= 0:
            return resolved
        return str(subset_dir.resolve())

    def _match_reference_files_for_targets(
        self,
        target_names: Sequence[str],
        ref_dir: str,
        ref_patterns: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        ref_files = self._collect_files_for_patterns(ref_dir, list(ref_patterns))
        exact_idx, norm_idx = self._quick_verify_build_name_indexes(ref_files)
        matched_refs: list[str] = []
        missing_refs: list[str] = []
        for name in target_names:
            target_path = str(Path(name))
            ref_path, _how = self._quick_verify_match_reference_path(
                target_path,
                exact_idx,
                norm_idx,
            )
            if ref_path:
                matched_refs.append(ref_path)
            else:
                missing_refs.append(str(name))
        matched_refs = sorted({str(p) for p in matched_refs if str(p).strip()})
        return matched_refs, missing_refs

    def _validate_inpaint_sharpen_verify_inputs(
        self,
    ) -> tuple[bool, str, str, list[str], list[str], list[str]]:
        sharp_csv_path = self.inpaint_sharpness_csv_var.get().strip()
        expected_names, err = self._collect_expected_sharpen_outputs(
            self.inpaint_input_var.get().strip(),
            sharp_csv_path,
        )
        if err:
            messagebox.showerror("Verify Sharpen", err)
            return False, "", "", [], [], []
        if not expected_names:
            return True, "", "", [], [], []
        out_dir = self.inpaint_sharpen_output_var.get().strip()
        if not out_dir:
            messagebox.showerror("Verify Sharpen", "Sharpen output folder is required.")
            return False, "", "", [], [], []
        if not os.path.isdir(out_dir):
            messagebox.showerror("Verify Sharpen", f"Sharpen output folder not found:\n{out_dir}")
            return False, "", "", [], [], []
        ok_ref, ref_dir, ref_patterns, ref_kind = self._resolve_verify_reference(
            "sharpen", "Verify Sharpen"
        )
        if not ok_ref:
            return False, "", "", [], [], []
        out_dir = self._pipeline_prepare_verify_subset_dir(
            out_dir, "sharpen_target", ["*.mp4"]
        )
        self._append_inpaint_log(f"[VERIFY] reference source: {ref_kind} ({ref_dir})")
        return True, out_dir, ref_dir, list(ref_patterns), expected_names, []

    def _start_inpaint_sharpen_verify_quick(self) -> None:
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Verify Sharpen", "Stop Sharpen before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Sharpen", "Another verification is already running.")
            return
        ok, out_dir, ref_dir, ref_patterns, expected_names, _unused = self._validate_inpaint_sharpen_verify_inputs()
        if not ok:
            return
        if not expected_names:
            self.inpaint_status_var.set("Verify Sharpen (Quick) completed")
            messagebox.showinfo("Verify Sharpen (Quick)", "0 eligible scenes.")
            self._pipeline_on_verify_finished("sharpen", True, "quick")
            return
        if shutil.which("ffprobe") is None:
            messagebox.showerror("Verify Sharpen", "ffprobe not found in PATH.")
            return

        self._set_verify_running(True, mode="sharpen_quick")
        self.inpaint_status_var.set("Verify Sharpen (Quick) running...")
        self._append_inpaint_log("=== Verify Sharpen (Quick) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_inpaint_sharpen_verify_quick_worker,
            args=(out_dir, ref_dir, ref_patterns, expected_names),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_inpaint_sharpen_verify_quick_worker(
        self,
        out_dir: str,
        ref_dir: str,
        ref_patterns: list[str],
        expected_names: list[str],
    ) -> None:
        try:
            root = Path(out_dir)
            target_files = [str(root / name) for name in expected_names if (root / name).is_file()]
            missing_output = [name for name in expected_names if not (root / name).is_file()]
            matched_refs, missing_refs = self._match_reference_files_for_targets(
                expected_names,
                ref_dir,
                ref_patterns,
            )
            if not target_files and not missing_output:
                self._log_queue.put(("sharpen_verify_quick_result", {
                    "ok": True,
                    "message": "0 eligible scenes.",
                    "broken_output": [],
                    "broken_reference": [],
                    "missing_output": [],
                }))
                return

            ref_files = list(matched_refs)
            if not ref_files:
                self._log_queue.put(("sharpen_verify_quick_result", {
                    "ok": False,
                    "message": "No reference replace-mask files found.",
                    "broken_output": [],
                    "broken_reference": [],
                    "missing_output": missing_output + list(missing_refs),
                }))
                return

            max_workers = self._get_verify_scenes_workers()
            self._log_queue.put(
                ("inpaint_line", f"[SHARPEN][QUICK] checking eligible output files={len(target_files)} and reference files={len(ref_files)} with {max_workers} workers")
            )
            out_stats = self._quick_verify_probe_group(
                target_files,
                max_workers,
                "inpaint_line",
                "output",
                "[SHARPEN][QUICK]",
            )
            ref_stats = self._quick_verify_probe_group(
                ref_files,
                max_workers,
                "inpaint_line",
                "reference",
                "[SHARPEN][QUICK]",
            )
            pair_stats = self._quick_verify_collect_packet_mismatch_targets(
                target_files,
                ref_files,
                out_stats.get("meta_by_path", {}),
                ref_stats.get("meta_by_path", {}),
                frame_tol=1,
            )
            packet_mismatch_output = pair_stats.get("mismatch_targets") or []
            unmatched_output = pair_stats.get("unmatched_targets") or []
            missing_reference = sorted(
                set((pair_stats.get("missing_reference") or []) + list(missing_refs))
            )
            broken_output = sorted(set((out_stats.get("broken") or []) + packet_mismatch_output))

            frames_ok = False
            frames_msg = "n.d."
            if out_stats["frames_available"] and ref_stats["frames_available"]:
                df = abs(int(out_stats["total_frames"]) - int(ref_stats["total_frames"]))
                frames_ok = df <= 1
                frames_msg = (
                    f"output={int(out_stats['total_frames'])} vs "
                    f"reference={int(ref_stats['total_frames'])} (delta={df})"
                )
            ok_final = (
                not missing_output
                and not broken_output
                and not ref_stats["broken"]
                and not unmatched_output
                and not missing_reference
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Sharpen quick verify completed.\n"
                f"Eligible scenes: {len(expected_names)}\n"
                f"Present output files: {len(target_files)}\n"
                f"Missing output files: {len(missing_output)}\n"
                f"Broken output files: {len(out_stats['broken'])}\n"
                f"Packet mismatch output files: {len(packet_mismatch_output)}\n"
                f"Unmatched output files: {len(unmatched_output)}\n"
                f"Missing reference files: {len(missing_reference)}\n"
                f"Broken reference files: {len(ref_stats['broken'])}\n"
                f"Packet details: {frames_msg}"
            )
            self._log_queue.put(("sharpen_verify_quick_result", {
                "ok": ok_final,
                "message": message,
                "broken_output": broken_output,
                "broken_reference": ref_stats["broken"],
                "missing_output": missing_output,
            }))
        except Exception as e:
            self._log_queue.put(("sharpen_verify_quick_result", {
                "ok": False,
                "message": f"Sharpen quick verify failed: {type(e).__name__}: {e}",
                "broken_output": [],
                "broken_reference": [],
                "missing_output": [],
            }))
        finally:
            self._log_queue.put(("verify_done", "sharpen_quick"))

    def _start_inpaint_sharpen_verify_deep(self) -> None:
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            messagebox.showinfo("Verify Sharpen", "Stop Sharpen before running verification.")
            return
        if self._verify_running:
            messagebox.showinfo("Verify Sharpen", "Another verification is already running.")
            return
        ok, out_dir, ref_dir, ref_patterns, expected_names, _unused = self._validate_inpaint_sharpen_verify_inputs()
        if not ok:
            return
        if not expected_names:
            self.inpaint_status_var.set("Verify Sharpen (Deep) completed")
            messagebox.showinfo("Verify Sharpen (Deep)", "0 eligible scenes.")
            self._pipeline_on_verify_finished("sharpen", True, "deep")
            return

        missing_output = [
            name for name in expected_names if not (Path(out_dir) / name).is_file()
        ]
        if missing_output:
            messagebox.showwarning(
                "Verify Sharpen (Deep)",
                "Missing eligible sharpen outputs:\n\n" + "\n".join(missing_output[:20]),
            )
            self._pipeline_on_verify_finished("sharpen", False, "deep")
            return

        script_path = utilities_path("verifyscenes.py")
        if not script_path.is_file():
            messagebox.showerror("Verify Sharpen", f"Script not found:\n{script_path}")
            return

        stage_out_dir = self._prepare_named_verify_subset_dir(
            out_dir,
            "sharpen_target",
            expected_names,
        )
        matched_refs, missing_refs = self._match_reference_files_for_targets(
            expected_names,
            ref_dir,
            ref_patterns,
        )
        if missing_refs:
            messagebox.showwarning(
                "Verify Sharpen (Deep)",
                "Missing replace-mask references for eligible sharpen outputs:\n\n"
                + "\n".join(missing_refs[:20]),
            )
            self._pipeline_on_verify_finished("sharpen", False, "deep")
            return
        stage_ref_dir = self._prepare_named_verify_subset_dir(
            ref_dir,
            "sharpen_ref",
            [Path(p).name for p in matched_refs],
        )

        workers = self._get_verify_scenes_workers()
        cmd = [
            sys.executable,
            str(script_path),
            str(Path(stage_out_dir).resolve()),
            str(Path(stage_ref_dir).resolve()),
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

        self._set_verify_running(True, mode="sharpen_deep")
        self.inpaint_status_var.set("Verify Sharpen (Deep) running...")
        self._append_inpaint_log("=== Verify Sharpen (Deep) started ===")
        self._append_inpaint_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._verify_thread = threading.Thread(
            target=self._run_inpaint_sharpen_verify_deep_worker,
            args=(cmd, str(Path(out_dir).resolve()), str(Path(stage_out_dir).resolve())),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_inpaint_sharpen_verify_deep_worker(
        self,
        cmd: list[str],
        original_out_dir: str,
        staged_out_dir: str,
    ) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        try:
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            proc = self._verify_popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            try:
                for raw in proc.stdout:
                    if self._verify_stop_requested:
                        raise VerifyStopRequested()
                    line = raw.rstrip("\n")
                    if line:
                        self._log_queue.put(("inpaint_line", f"[SHARPEN][DEEP] {line}"))
                        bad_path = self._resolve_verifyscenes_bad_path(line, staged_out_dir)
                        if bad_path:
                            orig = str(Path(original_out_dir) / Path(bad_path).name)
                            if orig not in seen_bad:
                                seen_bad.add(orig)
                                bad_files.append(orig)
                rc = proc.wait()
            finally:
                self._unregister_verify_process(proc)
        except VerifyStopRequested:
            rc = 1
        except Exception as e:
            self._log_queue.put(("inpaint_line", f"[SHARPEN][DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "sharpen_verify_deep_result",
                    {
                        "rc": rc,
                        "stopped": bool(self._verify_stop_requested),
                        "bad_files": bad_files,
                    },
                )
            )
            self._log_queue.put(("verify_done", "sharpen_deep"))

    def _touch_stop_marker_file(
        self,
        marker_path: str,
        logger: object | None = None,
    ) -> str:
        marker = str(marker_path or "").strip()
        if not marker:
            return ""
        try:
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            Path(marker).touch()
        except Exception as exc:
            if callable(logger):
                logger(f"[STOP] failed to create stop marker {marker}: {exc}")
            return ""
        return marker

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
            marker_path = self._touch_stop_marker_file(
                str(self._inpaint_stop_marker_path or "").strip()
                or os.path.join(
                    self.inpaint_output_var.get().strip() or "./work/output",
                    ".stop_after_current",
                ),
                self._append_inpaint_log,
            )
            if marker_path:
                self._inpaint_stop_marker_path = marker_path
        else:
            self.inpaint_status_var.set("Force stop requested...")
            self._append_inpaint_log("[STOP] force stop requested.")
            self._send_inpaint_signal(signal.SIGINT)
            self.root.after(1000, self._force_kill_inpaint)
        self._refresh_pipeline_run_button()

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
        self.inpaint_benchmark_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        sharpen_run_state = tk.DISABLED if (is_running or not self._sharpen_step_enabled_in_current_mode()) else tk.NORMAL
        self.inpaint_sharpen_run_btn.configure(state=sharpen_run_state)
        self.inpaint_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        verify_state = tk.DISABLED if (is_running or self._verify_running) else tk.NORMAL
        self.inpaint_verify_quick_btn.configure(state=verify_state)
        sharpen_verify_state = (
            tk.DISABLED
            if (is_running or self._verify_running or not self._sharpen_step_enabled_in_current_mode())
            else tk.NORMAL
        )
        self.inpaint_sharpen_verify_quick_btn.configure(state=sharpen_verify_state)
        if is_running:
            self.inpaint_stop_btn.configure(text="Stop")
        else:
            self.inpaint_stop_btn.configure(text="Stop")
            self._inpaint_stop_clicks = 0
            self._inpaint_stop_requested = False
            self._inpaint_stop_marker_path = ""
        self._update_replace_mask_dependent_controls()
        self._refresh_pipeline_run_button()

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
            out_stats = self._quick_verify_probe_group(
                out_files,
                max_workers,
                "inpaint_line",
                "output",
                "[QUICK]",
            )
            ref_stats = self._quick_verify_probe_group(
                ref_files,
                max_workers,
                "inpaint_line",
                "reference",
                "[QUICK]",
            )

            pair_stats = self._quick_verify_collect_packet_mismatch_targets(
                out_files,
                ref_files,
                out_stats.get("meta_by_path", {}),
                ref_stats.get("meta_by_path", {}),
                frame_tol=1,
            )
            packet_mismatch_output = pair_stats.get("mismatch_targets") or []
            unmatched_output = pair_stats.get("unmatched_targets") or []
            missing_reference = pair_stats.get("missing_reference") or []
            broken_output = sorted(set((out_stats.get("broken") or []) + packet_mismatch_output))

            self._log_queue.put(
                (
                    "inpaint_line",
                    (
                        "[QUICK] packet pair check: "
                        f"compared={int(pair_stats.get('pairs_compared', 0))}, "
                        f"n.d.={int(pair_stats.get('pairs_packet_nd', 0))}, "
                        f"mismatch={len(packet_mismatch_output)}, "
                        f"unmatched_output={len(unmatched_output)}, "
                        f"missing_reference={len(missing_reference)}"
                    ),
                )
            )

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
                not broken_output
                and not ref_stats["broken"]
                and count_ok
                and not unmatched_output
                and not missing_reference
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Inpainting quick verify completed.\n"
                f"Broken output files: {len(out_stats['broken'])}\n"
                f"Packet mismatch output files: {len(packet_mismatch_output)}\n"
                f"Unmatched output files: {len(unmatched_output)}\n"
                f"Missing reference files: {len(missing_reference)}\n"
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
                        "broken_output": broken_output,
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

        script_path = utilities_path("verifyscenes.py")
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
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            proc = self._verify_popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            try:
                for raw in proc.stdout:
                    if self._verify_stop_requested:
                        raise VerifyStopRequested()
                    line = raw.rstrip("\n")
                    if line:
                        self._log_queue.put(("inpaint_line", f"[DEEP] {line}"))
                        bad_path = self._resolve_verifyscenes_bad_path(line, out_dir)
                        if bad_path and bad_path not in seen_bad:
                            seen_bad.add(bad_path)
                            bad_files.append(bad_path)
                rc = proc.wait()
            finally:
                self._unregister_verify_process(proc)
        except VerifyStopRequested:
            rc = 1
        except Exception as e:
            self._log_queue.put(("inpaint_line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "inpaint_verify_deep_result",
                    {
                        "rc": rc,
                        "stopped": bool(self._verify_stop_requested),
                        "out_dir": out_dir,
                        "bad_files": bad_files,
                    },
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

        encode_frame = ttk.LabelFrame(parent, text="Encoding", padding=8)
        encode_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=6)
        encode_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(encode_frame, text="Codec:").grid(row=0, column=0, sticky="w")
        self.merge_codec_entry = ttk.Combobox(
            encode_frame,
            textvariable=self.merge_codec_var,
            values=self.FFMPEG_CODEC_CHOICES,
            width=12,
            state="readonly",
        )
        self.merge_codec_entry.grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(
            encode_frame,
            text="Mask for merge output is fixed grayscale-safe x264. This codec affects only the merged color output.",
            justify="left",
        ).grid(row=0, column=2, columnspan=2, sticky="w")

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
            buttons, text="Verify Mask", command=self._start_merge_mask_verify_quick
        )
        self.merge_mask_verify_quick_btn.grid(row=0, column=4, padx=6)
        self.merge_verify_quick_btn = ttk.Button(
            buttons, text="Verify Merge", command=self._start_merge_verify_quick
        )
        self.merge_verify_quick_btn.grid(row=0, column=5, padx=6)
        self.merge_stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop_merge_placeholder, state=tk.DISABLED
        )
        self.merge_stop_btn.grid(row=0, column=6, padx=6)
        ttk.Button(buttons, text="Clear Log", command=self._clear_merge_log).grid(
            row=0, column=7, padx=6
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
            self._reset_merge_auto_locked_defaults()
        self._apply_merge_control_states()
        self._on_merge_workers_changed()
        self._refresh_pipeline_status_panel()

    def _reset_merge_auto_locked_defaults(self) -> None:
        # Fields disabled in Auto mode.
        self.merge_mask_binarize_var.set("0.5")
        self.merge_mask_dilate_var.set("2")
        self.merge_mask_blur_var.set("2")
        self.merge_shadow_length_var.set("15")
        self.merge_shadow_curve_var.set("0")
        self.merge_dynamic_shadow_width_var.set(True)
        self.merge_use_replace_mask_var.set(True)
        self.merge_ct_preset_var.set("1")
        self.merge_ct_auto_mode_var.set("CSV Blend")
        self.merge_ct_exclude_black_var.set(True)

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

    def _build_merge_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        return pm_builders.build_merge_runner_payload(self)

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
        return pm_builders.build_mask_formerge_runner_payload(self)

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

        launcher_script = runner_path("run_mask_formerge_nogui.sh")
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
        self._merge_process_group_id = None
        self._merge_stop_marker_path = env_updates.get("STOP_MARKER", "").strip()
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
            self._merge_process_group_id = os.getpgid(proc.pid) if hasattr(os, "getpgid") else None
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", f"[MASK] {line}"))
                    self._try_parse_merge_progress(line)
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

        launcher_script = (
            resolve_repo_path(cmd[1]) if len(cmd) > 1 else runner_path("run_merging_nogui_batch.sh")
        )
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
                self._start_merge_autoct_csv(resume_merge_after=True)
                return
            try:
                autoct_ok, autoct_msg, incomplete_scenes = self._verify_autoct_csv_packet_coverage(
                    inpainted_dir=env_updates.get("INPAINTED_FOLDER", "").strip(),
                    preferred_inpainted_dir=env_updates.get("PREFERRED_INPAINTED_FOLDER", "").strip(),
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
                self._start_merge_autoct_csv(resume_merge_after=True)
                return
            if not self._pipeline_test_active:
                self._pipeline_set_completed("autoct_csv", True)
                self._pipeline_set_verified("autoct_csv", "none")
                self._refresh_pipeline_status_panel()
                self._save_pipeline_state()

        self._merge_resume_after_autoct = False
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
        self._merge_process_group_id = None
        self._merge_stop_marker_path = env_updates.get("STOP_MARKER", "").strip()
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
            self._merge_process_group_id = os.getpgid(proc.pid) if hasattr(os, "getpgid") else None
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._log_queue.put(("merge_line", line))
                    self._try_parse_merge_progress(line)
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

    def _start_merge_autoct_csv(self, resume_merge_after: bool = False) -> None:
        # Enable auto-resume only when AutoCT CSV is launched as Merging preflight.
        self._merge_resume_after_autoct = False
        if self._merge_thread and self._merge_thread.is_alive():
            messagebox.showinfo("Merging", "Another merging task is running.")
            return
        if self._verify_running:
            messagebox.showinfo("Merging", "Stop verification before creating autoct.csv.")
            return

        script_path = utilities_path("analyze_auto_ct_csv.py")
        if not script_path.is_file():
            messagebox.showerror("Merging", f"Script not found:\n{script_path}")
            return

        inpainted = self.merge_inpainted_var.get().strip()
        preferred_inpainted = self._preferred_inpainted_dir_for_consumers()
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
        if preferred_inpainted and os.path.isdir(preferred_inpainted):
            cmd.extend(
                [
                    "--preferred-inpainted-folder",
                    str(Path(preferred_inpainted).resolve()),
                ]
            )
        stop_marker = str(out_csv_path.parent / ".stop_after_current")
        cmd.extend(["--stop-marker", stop_marker])

        self._merge_stop_requested = False
        self._merge_stop_clicks = 0
        self.merge_status_var.set("Creating autoct.csv...")
        self.merge_progress_var.set(0.0)
        self._set_merge_running(True)
        if not self._pipeline_test_active:
            self._pipeline_invalidate_from("autoct_csv")
        self._append_merge_log("=== AutoCT CSV creation started ===")
        self._append_merge_log("CMD: " + " ".join(shlex.quote(x) for x in cmd))
        self._merge_process_group_id = None
        self._merge_stop_marker_path = stop_marker
        self._merge_resume_after_autoct = bool(resume_merge_after)
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
            self._merge_process_group_id = os.getpgid(proc.pid) if hasattr(os, "getpgid") else None
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
        running = bool(
            (self._merge_thread and self._merge_thread.is_alive())
            or (self._merge_process and self._merge_process.poll() is None)
            or self._merge_group_alive()
        )
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
            marker_path = self._ensure_merge_stop_marker()
            if marker_path:
                self._append_merge_log(f"[STOP] stop marker created: {marker_path}")
        else:
            self.merge_status_var.set("Force stop requested...")
            self._append_merge_log("[STOP] force stop requested.")
            self._send_merge_signal(signal.SIGINT)
            self.root.after(1000, self._force_kill_merge)
        self._refresh_pipeline_run_button()

    def _merge_group_alive(self) -> bool:
        pgid = self._merge_process_group_id
        if not pgid or not hasattr(os, "killpg"):
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
        return True

    def _ensure_merge_stop_marker(self) -> str:
        marker_path = str(self._merge_stop_marker_path or "").strip()
        if not marker_path:
            output_dir = self.merge_output_var.get().strip() or "./work/sbs"
            marker_path = os.path.join(output_dir, ".stop_after_current")
            self._merge_stop_marker_path = marker_path
        return self._touch_stop_marker_file(marker_path, self._append_merge_log)

    def _send_merge_signal(self, sig: int) -> None:
        proc = self._merge_process
        sent = False
        try:
            if proc is not None and proc.poll() is None:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    self._merge_process_group_id = pgid
                    os.killpg(pgid, sig)
                else:
                    proc.send_signal(sig)
                sent = True
            elif self._merge_group_alive():
                os.killpg(self._merge_process_group_id, sig)
                sent = True
        except Exception as exc:
            self._append_merge_log(f"Signal send failed: {exc}")
        if not sent:
            self._append_merge_log("[STOP] no active merge parent process found; relying on stop marker.")

    def _force_kill_merge(self) -> None:
        proc = self._merge_process
        try:
            if proc is not None and proc.poll() is None:
                if hasattr(os, "killpg"):
                    pgid = os.getpgid(proc.pid)
                    self._merge_process_group_id = pgid
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                self._append_merge_log("Merging process force-killed.")
            elif self._merge_group_alive():
                os.killpg(self._merge_process_group_id, signal.SIGKILL)
                self._append_merge_log("Merging worker process group force-killed.")
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
        self.merge_verify_quick_btn.configure(state=verify_state)
        if is_running:
            self.merge_stop_btn.configure(text="Stop")
        else:
            self.merge_stop_btn.configure(text="Stop")
            self._merge_stop_clicks = 0
            self._merge_stop_requested = False
            self._merge_stop_marker_path = ""
        self._update_replace_mask_dependent_controls()
        self._refresh_pipeline_run_button()

    def _handle_merge_done_event(self, payload: object) -> None:
        # AutoCT, mask-for-merge, and merge reuse the same thread slot.
        # Wait until the worker thread is actually dead before autorun starts the next step.
        if self._merge_thread and self._merge_thread.is_alive():
            self.root.after(50, lambda payload=payload: self._handle_merge_done_event(payload))
            return
        if self._merge_group_alive():
            self.root.after(100, lambda payload=payload: self._handle_merge_done_event(payload))
            return

        self._merge_thread = None
        self._merge_process = None
        self._merge_process_group_id = None
        self._merge_stop_marker_path = ""
        stop_requested = bool(self._merge_stop_requested)
        step_name = ""
        success = False
        pending_before = self._pipeline_pending_action
        pending_merge_run = (
            isinstance(pending_before, tuple)
            and len(pending_before) >= 2
            and str(pending_before[0]).strip().lower() == "merging"
            and str(pending_before[1]).strip().lower() == "run"
        )
        should_resume_merge = False
        if isinstance(payload, dict):
            step_name = str(payload.get("step", "")).strip().lower()
            success = bool(payload.get("success", False))
            if (
                step_name == "autoct_csv"
                and bool(self._merge_resume_after_autoct)
                and success
                and not stop_requested
            ):
                should_resume_merge = True
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
        if step_name == "autoct_csv" and bool(self._merge_resume_after_autoct):
            self._merge_resume_after_autoct = False
            if should_resume_merge:
                self._append_merge_log(
                    "[AUTOCT] CSV rebuilt. Resuming Merging automatically..."
                )
                self.root.after(10, self._run_merge_placeholder)
            elif pending_merge_run and not success:
                # AutoCT preflight failed while Merging run was pending.
                self._pipeline_on_run_finished("merging", False)
        if stop_requested:
            label_map = {
                "autoct_csv": "AutoCT CSV",
                "mask_for_merge": "Mask-for-merge",
                "merging": "Merging",
            }
            stop_label = label_map.get(step_name, "Merging")
            self._append_merge_log(f"[STOP] {stop_label} stopped.")
            self._finalize_pipeline_stop(stop_label)

    def _try_parse_merge_progress(self, line: str) -> None:
        m = re.search(r"^\[(?:RUN|OK|SKIP|ERR|DONE)\s*\]\s*(\d+)\s*/\s*(\d+)", line)
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
            out_stats = self._quick_verify_probe_group(
                out_files,
                max_workers,
                "merge_line",
                "mask",
                "[MASK-QUICK]",
            )
            ref_stats = self._quick_verify_probe_group(
                ref_files,
                max_workers,
                "merge_line",
                "reference",
                "[MASK-QUICK]",
            )

            pair_stats = self._quick_verify_collect_packet_mismatch_targets(
                out_files,
                ref_files,
                out_stats.get("meta_by_path", {}),
                ref_stats.get("meta_by_path", {}),
                frame_tol=1,
            )
            packet_mismatch_output = pair_stats.get("mismatch_targets") or []
            unmatched_output = pair_stats.get("unmatched_targets") or []
            missing_reference = pair_stats.get("missing_reference") or []
            broken_output = sorted(set((out_stats.get("broken") or []) + packet_mismatch_output))

            self._log_queue.put(
                (
                    "merge_line",
                    (
                        "[MASK-QUICK] packet pair check: "
                        f"compared={int(pair_stats.get('pairs_compared', 0))}, "
                        f"n.d.={int(pair_stats.get('pairs_packet_nd', 0))}, "
                        f"mismatch={len(packet_mismatch_output)}, "
                        f"unmatched_mask={len(unmatched_output)}, "
                        f"missing_reference={len(missing_reference)}"
                    ),
                )
            )

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
                not broken_output
                and not ref_stats["broken"]
                and count_ok
                and not unmatched_output
                and not missing_reference
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Mask quick verify completed.\n"
                f"Broken mask files: {len(out_stats['broken'])}\n"
                f"Packet mismatch mask files: {len(packet_mismatch_output)}\n"
                f"Unmatched mask files: {len(unmatched_output)}\n"
                f"Missing reference files: {len(missing_reference)}\n"
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
                        "broken_output": broken_output,
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

        script_path = utilities_path("verifyscenes.py")
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
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            proc = self._verify_popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            try:
                for raw in proc.stdout:
                    if self._verify_stop_requested:
                        raise VerifyStopRequested()
                    line = raw.rstrip("\n")
                    if line:
                        self._log_queue.put(("merge_line", f"[MASK-DEEP] {line}"))
                        bad_path = self._resolve_verifyscenes_bad_path(line, mask_dir)
                        if bad_path and bad_path not in seen_bad:
                            seen_bad.add(bad_path)
                            bad_files.append(bad_path)
                rc = proc.wait()
            finally:
                self._unregister_verify_process(proc)
        except VerifyStopRequested:
            rc = 1
        except Exception as e:
            self._log_queue.put(("merge_line", f"[MASK-DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "merge_mask_verify_deep_result",
                    {
                        "rc": rc,
                        "stopped": bool(self._verify_stop_requested),
                        "mask_dir": mask_dir,
                        "bad_files": bad_files,
                    },
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

    def _collect_merge_verify_seg_mono_stems(self) -> set[str]:
        stems: set[str] = set()
        candidate_dirs = [
            self.join_seg_mono_var.get().strip(),
            os.path.join(self.work_folder_var.get().strip(), "seg-mono"),
        ]
        seen_dirs: set[str] = set()
        for raw_dir in candidate_dirs:
            dir_txt = str(raw_dir or "").strip()
            if not dir_txt:
                continue
            try:
                resolved = str(Path(dir_txt).resolve())
            except Exception:
                resolved = dir_txt
            if not resolved or resolved in seen_dirs or not os.path.isdir(resolved):
                continue
            seen_dirs.add(resolved)
            root = Path(resolved)
            for pat in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"):
                for path in root.glob(pat):
                    if path.is_file():
                        stems.add(self._quick_verify_normalize_stem(path.stem))
        return stems

    def _collect_merge_verify_outputs(
        self, merged_dir: str, ref_files: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        all_outputs = sorted([str(p) for p in Path(merged_dir).glob("*.mp4") if p.is_file()])
        if not ref_files:
            return all_outputs, [], []
        exact_idx, norm_idx = self._quick_verify_build_name_indexes(ref_files)
        seg_mono_stems = self._collect_merge_verify_seg_mono_stems()
        matched_outputs: list[str] = []
        ignored_seg_mono: list[str] = []
        unmatched_outputs: list[str] = []
        for out_path in all_outputs:
            ref_path, _match_info = self._quick_verify_match_reference_path(
                out_path, exact_idx, norm_idx
            )
            if ref_path:
                matched_outputs.append(out_path)
                continue
            norm_key = self._quick_verify_normalize_stem(Path(out_path).stem)
            if norm_key in seg_mono_stems:
                ignored_seg_mono.append(out_path)
            else:
                unmatched_outputs.append(out_path)
        return matched_outputs, ignored_seg_mono, unmatched_outputs

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
            ref_files = self._collect_files_for_patterns(ref_dir, ref_patterns)
            matched_outputs, ignored_seg_mono, unmatched_output = self._collect_merge_verify_outputs(
                merged_dir, ref_files
            )
            if not matched_outputs:
                self._log_queue.put(("merge_verify_quick_result", {
                    "ok": False,
                    "message": "No merge output files matched the selected reference set.",
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
                (
                    "merge_line",
                    (
                        f"[QUICK] checking merged files={len(matched_outputs)} "
                        f"(ignored seg-mono SBS={len(ignored_seg_mono)}, extra-unmatched={len(unmatched_output)}) "
                        f"and reference files={len(ref_files)} with {max_workers} workers"
                    ),
                )
            )
            out_stats = self._quick_verify_probe_group(
                matched_outputs,
                max_workers,
                "merge_line",
                "merged",
                "[QUICK]",
            )
            ref_stats = self._quick_verify_probe_group(
                ref_files,
                max_workers,
                "merge_line",
                "reference",
                "[QUICK]",
            )

            pair_stats = self._quick_verify_collect_packet_mismatch_targets(
                matched_outputs,
                ref_files,
                out_stats.get("meta_by_path", {}),
                ref_stats.get("meta_by_path", {}),
                frame_tol=1,
            )
            packet_mismatch_output = pair_stats.get("mismatch_targets") or []
            missing_reference = pair_stats.get("missing_reference") or []
            broken_output = sorted(set((out_stats.get("broken") or []) + packet_mismatch_output))

            self._log_queue.put(
                (
                    "merge_line",
                    (
                        "[QUICK] packet pair check: "
                        f"compared={int(pair_stats.get('pairs_compared', 0))}, "
                        f"n.d.={int(pair_stats.get('pairs_packet_nd', 0))}, "
                        f"mismatch={len(packet_mismatch_output)}, "
                        f"unmatched_output={len(unmatched_output)}, "
                        f"missing_reference={len(missing_reference)}"
                    ),
                )
            )

            if ignored_seg_mono:
                self._log_queue.put(
                    (
                        "merge_line",
                        f"[QUICK] ignored {len(ignored_seg_mono)} Mono->SBS file(s) from seg-mono during merge verify",
                    )
                )

            count_ok = len(matched_outputs) == len(ref_files)
            count_msg = f"merged={len(matched_outputs)} vs reference={len(ref_files)}"

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
                not broken_output
                and not ref_stats["broken"]
                and count_ok
                and not unmatched_output
                and not missing_reference
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Merging quick verify completed.\n"
                f"Broken output files: {len(out_stats['broken'])}\n"
                f"Packet mismatch output files: {len(packet_mismatch_output)}\n"
                f"Ignored seg-mono SBS files: {len(ignored_seg_mono)}\n"
                f"Unmatched output files: {len(unmatched_output)}\n"
                f"Missing reference files: {len(missing_reference)}\n"
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
                        "broken_output": broken_output,
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
        ok, merged_dir, ref_dir, ref_patterns = self._validate_merge_verify_inputs()
        if not ok:
            return

        script_path = utilities_path("verifyscenes.py")
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
        self._append_merge_log(
            "CMD: merge deep verify will stage only merge-matched outputs into a temp subset, "
            "ignore Mono->SBS files coming from seg-mono, and run verifyscenes with manual cleanup on real outputs."
        )

        self._verify_thread = threading.Thread(
            target=self._run_merge_verify_deep_worker,
            args=(
                str(script_path),
                str(Path(merged_dir).resolve()),
                str(Path(ref_dir).resolve()),
                list(ref_patterns),
                workers,
            ),
            daemon=True,
        )
        self._verify_thread.start()

    def _run_merge_verify_deep_worker(
        self,
        script_path: str,
        merged_dir: str,
        ref_dir: str,
        ref_patterns: list[str],
        workers: int,
    ) -> None:
        rc = 1
        bad_files: list[str] = []
        seen_bad: set[str] = set()
        ignored_seg_mono: list[str] = []
        unmatched_output: list[str] = []
        deleted = 0
        cleanup_errors: list[str] = []
        try:
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            ref_files = self._collect_files_for_patterns(ref_dir, ref_patterns)
            if not ref_files:
                self._log_queue.put(("merge_line", "[DEEP][ERROR] no reference files found"))
                rc = 1
            else:
                matched_outputs, ignored_seg_mono, unmatched_output = (
                    self._collect_merge_verify_outputs(merged_dir, ref_files)
                )
                if ignored_seg_mono:
                    self._log_queue.put(
                        (
                            "merge_line",
                            (
                                f"[DEEP][IGNORE] excluded {len(ignored_seg_mono)} Mono->SBS "
                                "file(s) from seg-mono during merge verify"
                            ),
                        )
                    )
                for extra_path in unmatched_output[:20]:
                    self._log_queue.put(
                        (
                            "merge_line",
                            f"[DEEP][UNMATCHED] unexpected merged output not in reference set: {Path(extra_path).name}",
                        )
                    )
                if not matched_outputs:
                    self._log_queue.put(
                        ("merge_line", "[DEEP][ERROR] no merge output files matched the reference set")
                    )
                    rc = 1
                else:
                    with tempfile.TemporaryDirectory(prefix="verify_merge_") as tmpdir:
                        tmp_root = Path(tmpdir)
                        target_dir = tmp_root / "targets"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        link_map: dict[str, str] = {}
                        for out_path in matched_outputs:
                            src = Path(out_path)
                            tmp_target = target_dir / src.name
                            if self._pipeline_link_or_copy_file(src, tmp_target):
                                link_map[str(tmp_target.resolve())] = str(src)
                        cmd = [
                            sys.executable,
                            script_path,
                            str(target_dir),
                            str(Path(ref_dir).resolve()),
                            "--extensions",
                            self.VERIFY_ALL_VIDEO_EXTENSIONS,
                            "--workers",
                            str(max(1, int(workers))),
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
                                "merge_line",
                                "[DEEP] cmd: " + " ".join(shlex.quote(x) for x in cmd),
                            )
                        )
                        proc = self._verify_popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )
                        assert proc.stdout is not None
                        try:
                            for raw in proc.stdout:
                                if self._verify_stop_requested:
                                    raise VerifyStopRequested()
                                line = raw.rstrip("\n")
                                if line:
                                    self._log_queue.put(("merge_line", f"[DEEP] {line}"))
                                    bad_path = self._resolve_verifyscenes_bad_path(line, str(target_dir))
                                    if bad_path:
                                        real_bad = link_map.get(bad_path, "")
                                        if real_bad and real_bad not in seen_bad:
                                            seen_bad.add(real_bad)
                                            bad_files.append(real_bad)
                            rc = int(proc.wait() or 0)
                        finally:
                            self._unregister_verify_process(proc)

                    if bad_files:
                        deleted, cleanup_errors = self._delete_file_paths(bad_files)
                    if unmatched_output:
                        rc = rc or 1
        except VerifyStopRequested:
            rc = 1
        except Exception as e:
            self._log_queue.put(("merge_line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "merge_verify_deep_result",
                    {
                        "rc": rc,
                        "stopped": bool(self._verify_stop_requested),
                        "merged_dir": merged_dir,
                        "bad_files": bad_files,
                        "deleted": deleted,
                        "cleanup_errors": cleanup_errors,
                        "ignored_seg_mono": ignored_seg_mono,
                        "unmatched_output": unmatched_output,
                    },
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

        ttk.Label(params_frame, text="PixFmt:").grid(
            row=0, column=6, sticky="w", padx=(12, 0)
        )
        self.join_pixfmt_entry = ttk.Entry(params_frame, textvariable=self.join_pix_fmt_var, width=10)
        self.join_pixfmt_entry.grid(row=0, column=7, sticky="w", padx=(6, 0))

        ttk.Label(params_frame, text="Extra ffmpeg args:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.join_extra_args_entry = ttk.Entry(params_frame, textvariable=self.join_extra_args_var)
        self.join_extra_args_entry.grid(
            row=1, column=1, columnspan=7, sticky="ew", padx=(6, 0), pady=(8, 0)
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

    def _apply_join_control_states(self) -> None:
        manual = self.join_mode_var.get().strip() == "Manual"
        self.join_encoder_entry.configure(state="readonly" if manual else tk.DISABLED)
        self.join_crf_entry.configure(state=tk.NORMAL)
        self.join_preset_entry.configure(state=tk.NORMAL if manual else tk.DISABLED)
        if not manual:
            self.join_pix_fmt_var.set("yuv420p")
        self.join_pixfmt_entry.configure(state=tk.NORMAL if manual else tk.DISABLED)
        self.join_extra_args_entry.configure(state=tk.NORMAL if manual else tk.DISABLED)
        self._preview_join_command()

    def _quality_flag_for_codec(self, codec: str, fallback: str, nvenc_flag: str) -> str:
        codec_value = self._normalize_ffmpeg_codec(codec, fallback)
        return nvenc_flag if "nvenc" in codec_value else "crf"

    def _join_quality_flag(self) -> str:
        encoder = self._normalize_ffmpeg_codec(self.join_encoder_var.get(), "hevc_nvenc")
        return self._quality_flag_for_codec(encoder, "hevc_nvenc", "cq")

    def _join_layout_for_seg_mono(self) -> str:
        return pm_builders.join_layout_for_seg_mono(self)

    def _build_join_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        return pm_builders.build_join_runner_payload(self)

    def _default_remux_output_path(self) -> str:
        return pm_builders.default_remux_output_path(self)

    def _build_join_remux_payload(self) -> tuple[list[str], dict[str, str], str, str]:
        return pm_builders.build_join_remux_payload(self)

    def _build_join_prepare_mono_cmd(self) -> list[str]:
        return pm_builders.build_join_prepare_mono_cmd(self)

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

        join_script = utilities_path("Rejoin_HEVC_NVENC.sh")
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

        join_counts = self._collect_join_scene_count_stats()
        join_counts_match = bool(join_counts.get("counts_match", False))
        if not join_counts_match:
            prompt = (
                "Do you want to run an incomplete merge?\n\n"
                f"SBS found: {int(join_counts.get('sbs_count', 0))}\n"
                f"Expected from seg + seg-mono: {int(join_counts.get('expected_total', 0))}\n"
                f"(seg={int(join_counts.get('seg_count', 0))}, "
                f"seg-mono={int(join_counts.get('seg_mono_count', 0))})"
            )
            if not messagebox.askyesno("Joining", prompt):
                return

        self._join_stop_requested = False
        self._join_stop_clicks = 0
        self._join_expected_duration_sec = None
        self._join_active_output_path = out_path
        self._join_mark_completed = bool(join_counts_match)
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
        if not join_counts_match:
            self._append_join_log(
                "[JOIN] Incomplete mode accepted: "
                f"sbs={int(join_counts.get('sbs_count', 0))} "
                f"expected={int(join_counts.get('expected_total', 0))} "
                f"(seg={int(join_counts.get('seg_count', 0))}, "
                f"seg-mono={int(join_counts.get('seg_mono_count', 0))})"
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

        prep_script = utilities_path("prepare_seg_mono_to_sbs.py")
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

        remux_script = utilities_path("remux_replace_video_mkvtoolnix.sh")
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
        self._refresh_pipeline_run_button()

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
        self._refresh_pipeline_run_button()

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
        return "quick"

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
        self._set_verify_running(True, mode=f"join_mono_{mode}")
        self.join_status_var.set(f"Mono->SBS Verify ({mode}) running...")
        self._append_join_log(f"=== Verify Mono->SBS ({mode}) started ===")
        self._verify_thread = threading.Thread(
            target=self._run_join_mono_verify_quick_worker,
            args=(),
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
        except VerifyStopRequested:
            self._log_queue.put(
                (
                    "join_mono_verify_result",
                    {
                        "ok": False,
                        "stopped": True,
                        "message": "Mono->SBS quick verify stopped.",
                        "mode": "quick",
                        "broken_output": [],
                        "broken_reference": [],
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
                    proc = self._verify_popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )
                    assert proc.stdout is not None
                    try:
                        for raw in proc.stdout:
                            if self._verify_stop_requested:
                                raise VerifyStopRequested()
                            line = raw.rstrip("\n")
                            if line:
                                self._log_queue.put(("join_line", f"[MONO][DEEP] {line}"))
                                bad_path = self._resolve_verifyscenes_bad_path(line, str(target_dir))
                                if bad_path:
                                    real_bad = link_map.get(bad_path)
                                    if real_bad and real_bad not in bad_targets:
                                        bad_targets.append(real_bad)
                        rc = int(proc.wait() or 0)
                    finally:
                        self._unregister_verify_process(proc)
                    overall_ok = rc == 0
                    message = (
                        "Mono->SBS deep verify completed successfully."
                        if overall_ok
                        else "Mono->SBS deep verify failed."
                    )
        except VerifyStopRequested:
            overall_ok = False
            message = "Mono->SBS deep verify stopped."
        except Exception as e:
            overall_ok = False
            message = f"Mono->SBS deep verify failed: {type(e).__name__}: {e}"
        finally:
            self._log_queue.put(
                (
                    "join_mono_verify_result",
                    {
                        "ok": overall_ok,
                        "stopped": bool(self._verify_stop_requested),
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
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            src = self._probe_video_basic(source_path)
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            out = self._probe_video_basic(joined_path)
            if self._verify_stop_requested:
                raise VerifyStopRequested()
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
        except VerifyStopRequested:
            self._log_queue.put(
                (
                    "join_verify_result",
                    {"ok": False, "stopped": True, "message": "Join verify stopped."},
                )
            )
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

    def _retry_policy_default(
        self,
        default_map: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for key, cfg in default_map.items():
            row = {
                "garbage_collection_threshold": bool(cfg.get("garbage_collection_threshold", True)),
                "expandable_segments": bool(cfg.get("expandable_segments", True)),
                "max_split_size_mb": self._normalize_retry_max_split(cfg.get("max_split_size_mb", "off")),
                "cpu_offload_inherited": bool(cfg.get("cpu_offload_inherited", True)),
                "cpu_offload_mode": self._normalize_retry_offload_mode(cfg.get("cpu_offload_mode", "model")),
            }
            if "worker_mode" in cfg:
                row["worker_mode"] = self._normalize_depth_runtime_mode(cfg.get("worker_mode", "original"))
            if "window_size" in cfg:
                row["window_size"] = self._normalize_depth_retry_window_size(cfg.get("window_size", "65"))
            if "overlap" in cfg:
                row["overlap"] = self._normalize_depth_retry_overlap(cfg.get("overlap", "15"))
            out[key] = row
        return out

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
        self,
        data: object,
        default_map: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        out = self._retry_policy_default(default_map)
        if not isinstance(data, dict):
            return out
        for profile in self.RETRY_POLICY_PROFILES:
            raw = data.get(profile)
            if not isinstance(raw, dict):
                continue
            out_row = dict(out[profile])
            out_row["garbage_collection_threshold"] = bool(
                raw.get(
                    "garbage_collection_threshold",
                    out_row["garbage_collection_threshold"],
                )
            )
            out_row["expandable_segments"] = bool(
                raw.get("expandable_segments", out_row["expandable_segments"])
            )
            out_row["max_split_size_mb"] = self._normalize_retry_max_split(
                raw.get("max_split_size_mb", out_row["max_split_size_mb"])
            )
            out_row["cpu_offload_inherited"] = bool(
                raw.get(
                    "cpu_offload_inherited",
                    out_row["cpu_offload_inherited"],
                )
            )
            out_row["cpu_offload_mode"] = self._normalize_retry_offload_mode(
                raw.get("cpu_offload_mode", out_row["cpu_offload_mode"])
            )
            if "worker_mode" in out_row:
                out_row["worker_mode"] = self._normalize_depth_runtime_mode(
                    raw.get("worker_mode", out_row["worker_mode"])
                )
            if "window_offset" in out_row:
                out_row["window_offset"] = self._normalize_depth_retry_offset(
                    raw.get("window_offset", out_row["window_offset"])
                )
            if "overlap_offset" in out_row:
                out_row["overlap_offset"] = self._normalize_depth_retry_offset(
                    raw.get("overlap_offset", out_row["overlap_offset"])
                )
            out[profile] = out_row
        return out

    def _retry_policy_from_config_key(
        self,
        key: str,
        default_map: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        return self._normalize_retry_policy_config(self._config.get(key), default_map)

    def _collect_retry_policy_config_from_vars(
        self, vars_map: dict[str, dict[str, tk.Variable]]
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for profile in self.RETRY_POLICY_PROFILES:
            row = vars_map.get(profile, {})
            is_depth_run_row = vars_map is self.depth_retry_policy_vars and profile == "run"
            is_inpaint_run_row = vars_map is self.inpaint_retry_policy_vars and profile == "run"
            payload = {
                "garbage_collection_threshold": bool(
                    row["garbage_collection_threshold"].get()
                ),
                "expandable_segments": bool(row["expandable_segments"].get()),
                "max_split_size_mb": self._normalize_retry_max_split(
                    row["max_split_size_mb"].get()
                ),
            }
            if is_depth_run_row or is_inpaint_run_row:
                payload["cpu_offload_inherited"] = True
                out[profile] = payload
                continue
            payload["cpu_offload_inherited"] = bool(row["cpu_offload_inherited"].get())
            payload["cpu_offload_mode"] = self._normalize_retry_offload_mode(
                row["cpu_offload_mode"].get()
            )
            if "worker_mode" in row:
                payload["worker_mode"] = self._normalize_depth_runtime_mode(
                    row["worker_mode"].get()
                )
            if "window_offset" in row:
                payload["window_offset"] = self._normalize_depth_retry_offset(
                    row["window_offset"].get()
                )
            if "overlap_offset" in row:
                payload["overlap_offset"] = self._normalize_depth_retry_offset(
                    row["overlap_offset"].get()
                )
            out[profile] = payload
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
            is_depth_run_row = vars_map is self.depth_retry_policy_vars and profile == "run"
            is_inpaint_run_row = vars_map is self.inpaint_retry_policy_vars and profile == "run"
            inherited = bool(row["cpu_offload_inherited"].get()) or is_depth_run_row or is_inpaint_run_row
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
            if "worker_mode" in row:
                worker_mode = self._normalize_depth_runtime_mode(
                    self.depth_runtime_mode_var.get()
                    if is_depth_run_row
                    else row["worker_mode"].get()
                )
                out[profile]["worker_mode"] = worker_mode
                out[profile]["worker_script"] = self._resolve_depth_worker_script(worker_mode)
            if "window_offset" in row:
                base_window = int(
                    self._normalize_depth_retry_window_size(
                        self.depth_chunk_size_var.get(),
                        "65",
                    )
                )
                window_offset = int(
                    self._normalize_depth_retry_offset(
                        "0" if is_depth_run_row else row["window_offset"].get()
                    )
                )
                out[profile]["window_size"] = max(1, base_window + window_offset)
            if "overlap_offset" in row:
                base_overlap = int(
                    self._normalize_depth_retry_overlap(
                        self.depth_overlap_var.get(),
                        "15",
                    )
                )
                overlap_offset = int(
                    self._normalize_depth_retry_offset(
                        "0" if is_depth_run_row else row["overlap_offset"].get()
                    )
                )
                out[profile]["overlap"] = max(0, base_overlap + overlap_offset)
        return out

    def _build_retry_policy_json(
        self,
        vars_map: dict[str, dict[str, tk.Variable]],
        inherited_offload: str,
    ) -> str:
        payload = self._build_retry_policy_runtime_payload(vars_map, inherited_offload)
        return json.dumps(payload, separators=(",", ":"))

    def _set_retry_policy_vars_to_defaults(self) -> None:
        defaults_map = (
            (self.depth_retry_policy_vars, self._retry_policy_default(self.DEPTH_RETRY_POLICY_DEFAULT)),
            (self.inpaint_retry_policy_vars, self._retry_policy_default(self.INPAINT_RETRY_POLICY_DEFAULT)),
        )
        for profile in self.RETRY_POLICY_PROFILES:
            for vars_map, defaults in defaults_map:
                drow = defaults[profile]
                row = vars_map[profile]
                row["garbage_collection_threshold"].set(
                    bool(drow["garbage_collection_threshold"])
                )
                row["expandable_segments"].set(bool(drow["expandable_segments"]))
                row["max_split_size_mb"].set(str(drow["max_split_size_mb"]))
                row["cpu_offload_inherited"].set(bool(drow["cpu_offload_inherited"]))
                row["cpu_offload_mode"].set(str(drow["cpu_offload_mode"]))
                if "worker_mode" in row and "worker_mode" in drow:
                    row["worker_mode"].set(str(drow["worker_mode"]))
                if "window_offset" in row and "window_offset" in drow:
                    row["window_offset"].set(str(drow["window_offset"]))
                if "overlap_offset" in row and "overlap_offset" in drow:
                    row["overlap_offset"].set(str(drow["overlap_offset"]))

    def _save_config_if_ready(self) -> None:
        if not getattr(self, "_config_save_ready", False):
            return
        self._save_config()

    def _sync_depth_retry_inherited_values(self) -> None:
        inherited_offload = self._normalize_retry_offload_mode(
            self.depth_cpu_offload_var.get().strip() or "model"
        )
        inherited_worker_mode = self._normalize_depth_runtime_mode(
            self.depth_runtime_mode_var.get()
        )
        for profile in self.RETRY_POLICY_PROFILES:
            row = self.depth_retry_policy_vars.get(profile)
            if row is None:
                continue
            effective_inherited = bool(row["cpu_offload_inherited"].get()) or profile == "run"
            if profile == "run" and not bool(row["cpu_offload_inherited"].get()):
                row["cpu_offload_inherited"].set(True)
            if not effective_inherited:
                continue
            row["cpu_offload_mode"].set(inherited_offload)
            if "worker_mode" in row:
                row["worker_mode"].set(inherited_worker_mode)

    def _set_depth_retry_widget_states(self) -> None:
        for profile in self.RETRY_POLICY_PROFILES:
            row = self.depth_retry_policy_vars.get(profile)
            if row is None:
                continue
            inherited = bool(row["cpu_offload_inherited"].get()) or profile == "run"
            offload = self._depth_retry_offload_widgets.get(profile)
            worker = self._depth_retry_worker_widgets.get(profile)
            window = self._depth_retry_window_widgets.get(profile)
            overlap = self._depth_retry_overlap_widgets.get(profile)
            inherited_toggle = self._depth_retry_inherited_widgets.get(profile)
            combo_state = tk.DISABLED if inherited else "readonly"
            entry_state = tk.DISABLED if inherited else "readonly"
            if offload is not None:
                offload.configure(state=combo_state)
            if worker is not None:
                worker.configure(state=combo_state)
            if window is not None:
                window.configure(state=entry_state)
            if overlap is not None:
                overlap.configure(state=entry_state)
            if inherited_toggle is not None:
                inherited_toggle.configure(state=tk.DISABLED if profile == "run" else tk.NORMAL)

    def _on_depth_retry_inherited_source_changed(self, *_args) -> None:
        self._sync_depth_retry_inherited_values()
        self._preview_depth_command()

    def _on_depth_retry_policy_changed(self) -> None:
        self._sync_depth_retry_inherited_values()
        self._set_depth_retry_widget_states()
        self._preview_depth_command()
        self._save_config_if_ready()

    def _on_inpaint_retry_policy_changed(self) -> None:
        for profile in self.RETRY_POLICY_PROFILES:
            row = self.inpaint_retry_policy_vars.get(profile)
            combo = self._inpaint_retry_offload_widgets.get(profile)
            if row is not None and "cpu_offload_inherited" in row:
                row["cpu_offload_inherited"].set(profile == "run")
            if combo is not None:
                combo.configure(state="readonly")
        self._preview_inpaint_command()
        self._save_config_if_ready()

    def _build_retry_policy_table(
        self,
        parent: ttk.LabelFrame,
        vars_map: dict[str, dict[str, tk.Variable]],
        widget_map: dict[str, ttk.Combobox],
        change_cb,
        *,
        show_inherited: bool = True,
        include_depth_controls: bool = False,
    ) -> None:
        is_depth_policy = vars_map is self.depth_retry_policy_vars
        is_inpaint_policy = vars_map is self.inpaint_retry_policy_vars
        header: list[tuple[str, str]] = [
            ("profile", "Profile"),
            ("gc", "Garbage 0.8"),
            ("expand", "Expandable"),
            ("split", "Max split"),
            ("cpu", "CPU mode"),
        ]
        if include_depth_controls:
            header.extend(
                [
                    ("script", "Script"),
                    ("window", "Chunk Δ"),
                    ("overlap", "Overlap Δ"),
                ]
            )
        if show_inherited:
            header.append(("inherit", "Inherited"))
        for idx in range(len(header) + 1):
            parent.grid_columnconfigure(idx, weight=0)
        parent.grid_columnconfigure(len(header), weight=1)
        for cidx, (_key, text) in enumerate(header):
            ttk.Label(parent, text=text).grid(
                row=0, column=cidx, sticky="w", padx=(0, 6)
            )

        for ridx, profile in enumerate(self.RETRY_POLICY_PROFILES, start=1):
            row = vars_map[profile]
            col = 0
            ttk.Label(parent, text=profile).grid(row=ridx, column=col, sticky="w", pady=2)
            col += 1
            ttk.Checkbutton(
                parent,
                variable=row["garbage_collection_threshold"],
                command=change_cb,
            ).grid(row=ridx, column=col, sticky="w", pady=2)
            col += 1
            ttk.Checkbutton(
                parent,
                variable=row["expandable_segments"],
                command=change_cb,
            ).grid(row=ridx, column=col, sticky="w", pady=2)
            col += 1
            split_combo = ttk.Combobox(
                parent,
                textvariable=row["max_split_size_mb"],
                values=self.RETRY_POLICY_MAX_SPLIT_CHOICES,
                state="readonly",
                width=6,
            )
            split_combo.grid(row=ridx, column=col, sticky="w", pady=2)
            split_combo.bind("<<ComboboxSelected>>", lambda _e: change_cb())
            col += 1
            if profile == "run" and (is_depth_policy or is_inpaint_policy):
                continue
            offload_combo = ttk.Combobox(
                parent,
                textvariable=row["cpu_offload_mode"],
                values=self.RETRY_POLICY_OFFLOAD_CHOICES,
                state="readonly",
                width=10,
            )
            offload_combo.grid(row=ridx, column=col, sticky="w", pady=2)
            offload_combo.bind("<<ComboboxSelected>>", lambda _e: change_cb())
            col += 1
            if include_depth_controls:
                worker_combo = ttk.Combobox(
                    parent,
                    textvariable=row["worker_mode"],
                    values=self.DEPTH_RUNTIME_MODE_CHOICES,
                    state="readonly",
                    width=9,
                )
                worker_combo.grid(row=ridx, column=col, sticky="w", pady=2)
                worker_combo.bind("<<ComboboxSelected>>", lambda _e: change_cb())
                col += 1
                window_entry = ttk.Combobox(
                    parent,
                    textvariable=row["window_offset"],
                    values=self.DEPTH_RETRY_OFFSET_CHOICES,
                    state="readonly",
                    width=6,
                )
                window_entry.grid(row=ridx, column=col, sticky="w", pady=2)
                window_entry.bind("<<ComboboxSelected>>", lambda _e: change_cb())
                col += 1
                overlap_entry = ttk.Combobox(
                    parent,
                    textvariable=row["overlap_offset"],
                    values=self.DEPTH_RETRY_OFFSET_CHOICES,
                    state="readonly",
                    width=6,
                )
                overlap_entry.grid(row=ridx, column=col, sticky="w", pady=2)
                overlap_entry.bind("<<ComboboxSelected>>", lambda _e: change_cb())
                self._depth_retry_worker_widgets[profile] = worker_combo
                self._depth_retry_window_widgets[profile] = window_entry
                self._depth_retry_overlap_widgets[profile] = overlap_entry
                col += 1
            if show_inherited:
                inherited_btn = ttk.Checkbutton(
                    parent,
                    variable=row["cpu_offload_inherited"],
                    command=change_cb,
                )
                inherited_btn.grid(row=ridx, column=col, sticky="w", pady=2)
                if is_depth_policy:
                    self._depth_retry_inherited_widgets[profile] = inherited_btn
            widget_map[profile] = offload_combo

    def _build_options_tab(self, parent: ttk.Frame) -> None:
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        global_encode = ttk.LabelFrame(parent, text="Global Encoder Setting", padding=8)
        global_encode.grid(row=0, column=0, columnspan=2, sticky="ew", pady=4)
        global_encode.grid_columnconfigure(3, weight=1)

        ttk.Label(global_encode, text="Global encoder mode:").grid(row=0, column=0, sticky="w")
        self.global_encoder_mode_combo = ttk.Combobox(
            global_encode,
            textvariable=self.global_encoder_mode_var,
            values=self.GLOBAL_ENCODER_MODE_CHOICES,
            state="readonly",
            width=12,
        )
        self.global_encoder_mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 12))
        self.global_encoder_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_global_encoder_mode_selected
        )
        ttk.Label(global_encode, text="Generated args preview:").grid(
            row=0, column=2, sticky="w", padx=(0, 8)
        )
        ttk.Entry(
            global_encode,
            textvariable=self.global_encoder_preview_var,
            state="readonly",
        ).grid(row=0, column=3, sticky="ew")

        ttk.Label(global_encode, text="Global extra ffmpeg args (append):").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(global_encode, textvariable=self.global_ffmpeg_extra_args_var).grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        middle_frame = ttk.Frame(parent)
        middle_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=6)
        middle_frame.grid_columnconfigure(0, weight=3)
        middle_frame.grid_columnconfigure(1, weight=2)
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_rowconfigure(1, weight=1)

        step_frame = ttk.LabelFrame(middle_frame, text="Pipeline Step State", padding=8)
        step_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 4))
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

        depth_retry_frame = ttk.LabelFrame(middle_frame, text="DepthCrafter Retry Policy", padding=6)
        depth_retry_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        self._build_retry_policy_table(
            depth_retry_frame,
            self.depth_retry_policy_vars,
            self._depth_retry_offload_widgets,
            self._on_depth_retry_policy_changed,
            show_inherited=True,
            include_depth_controls=True,
        )

        inpaint_retry_frame = ttk.LabelFrame(middle_frame, text="Inpainting Retry Policy", padding=6)
        inpaint_retry_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))
        self._build_retry_policy_table(
            inpaint_retry_frame,
            self.inpaint_retry_policy_vars,
            self._inpaint_retry_offload_widgets,
            self._on_inpaint_retry_policy_changed,
            show_inherited=False,
            include_depth_controls=False,
        )

        run_frame = ttk.LabelFrame(parent, text="Run & Progress", padding=8)
        run_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=6)
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
        self.codec_validation_btn = ttk.Button(
            btn_row,
            text="Codec Validation",
            command=self._start_codec_validation,
        )
        self.codec_validation_btn.grid(row=0, column=5, padx=(18, 8))
        ttk.Label(btn_row, text="Verify workers:").grid(row=0, column=6, padx=(12, 6))
        self.verify_scenes_workers_entry = ttk.Entry(
            btn_row, textvariable=self.verify_scenes_workers_var, width=7
        )
        self.verify_scenes_workers_entry.grid(row=0, column=7, sticky="w")
        ttk.Label(btn_row, text="Test run files:").grid(row=0, column=8, padx=(12, 6))
        ttk.Entry(
            btn_row,
            textvariable=self.pipeline_test_run_files_var,
            width=6,
        ).grid(row=0, column=9, sticky="w")

        status_row = ttk.Frame(run_frame)
        status_row.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        status_row.grid_columnconfigure(0, weight=1)
        ttk.Label(status_row, textvariable=self.pipeline_run_status_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_row, textvariable=self.pipeline_checked_files_var).grid(
            row=0, column=1, sticky="e", padx=(12, 0)
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
        self._refresh_global_encoder_preview()
        self._on_depth_retry_policy_changed()
        self._on_inpaint_retry_policy_changed()

    def _build_join_validation_command(self, output_path: str) -> tuple[list[str], str]:
        encoder = self._normalize_ffmpeg_codec(self.join_encoder_var.get(), "hevc_nvenc")
        quality_flag = self._join_quality_flag()
        quality_value = self.join_crf_var.get().strip() or "12"
        preset = self.join_preset_var.get().strip() or "p7"
        pix_fmt = self.join_pix_fmt_var.get().strip() or "yuv420p"
        extra_args = self.join_extra_args_var.get().strip()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x144:rate=1",
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            encoder,
            "-preset",
            preset,
            f"-{quality_flag}",
            quality_value,
            "-pix_fmt",
            pix_fmt,
        ]
        if extra_args:
            cmd.extend(shlex.split(extra_args))
        cmd.append(output_path)
        summary = f"{encoder} {quality_flag}={quality_value} {pix_fmt}"
        if extra_args:
            summary = f"{summary} | + {extra_args}"
        return cmd, summary

    def _build_codec_validation_specs(self) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        extra_args = self._current_global_ffmpeg_extra_args()
        color_steps = [
            ("Scene split", self.scene_codec_var.get()),
            ("Splat output", self.splat_codec_var.get()),
            ("Inpaint output", self.inpaint_codec_var.get()),
            ("Merge output", self.merge_codec_var.get()),
        ]
        for label, codec in color_steps:
            profile = self._resolve_color_profile_for_codec(codec)
            specs.append(
                {
                    "label": label,
                    "summary": profile_preview_line(profile, extra_args),
                    "builder": lambda out_path, p=profile, extra=extra_args: build_validation_command(
                        p, out_path, extra_args=extra
                    ),
                    "ext": ".mp4",
                }
            )

        depth_pre = resolve_depth_preprocess_profile(self.depth_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC)
        specs.append(
            {
                "label": "Depth preprocess",
                "summary": profile_preview_line(depth_pre),
                "builder": lambda out_path, p=depth_pre: build_validation_command(p, out_path),
                "ext": ".mp4",
            }
        )

        depth_final = resolve_depth_final_grayscale_profile()
        specs.append(
            {
                "label": "Depth final grayscale",
                "summary": profile_preview_line(depth_final),
                "builder": lambda out_path, p=depth_final: build_validation_command(p, out_path),
                "ext": ".mp4",
            }
        )

        replace_mask = resolve_replace_mask_binary_profile()
        specs.append(
            {
                "label": "Replace mask binary",
                "summary": profile_preview_line(replace_mask),
                "builder": lambda out_path, p=replace_mask: build_validation_command(p, out_path),
                "ext": ".mkv",
            }
        )

        mask_gray = resolve_mask_for_merge_grayscale_profile()
        specs.append(
            {
                "label": "Mask for merge grayscale",
                "summary": profile_preview_line(mask_gray),
                "builder": lambda out_path, p=mask_gray: build_validation_command(p, out_path),
                "ext": ".mp4",
            }
        )

        specs.append(
            {
                "label": "Join output",
                "summary": "join runtime settings",
                "builder": self._build_join_validation_command,
                "ext": ".mp4",
            }
        )
        return specs

    def _start_codec_validation(self) -> None:
        if self._codec_validation_running:
            messagebox.showinfo("Codec Validation", "Codec validation is already running.")
            return
        self._codec_validation_running = True
        self.pipeline_run_status_var.set("Codec validation running...")
        if hasattr(self, "codec_validation_btn"):
            self.codec_validation_btn.configure(state=tk.DISABLED)
        self._codec_validation_thread = threading.Thread(
            target=self._run_codec_validation_worker,
            daemon=True,
        )
        self._codec_validation_thread.start()

    def _run_codec_validation_worker(self) -> None:
        results: list[dict[str, str]] = []
        error_message = ""
        try:
            specs = self._build_codec_validation_specs()
            with tempfile.TemporaryDirectory(prefix="codec_validation_") as tmp_dir:
                tmp_root = Path(tmp_dir)
                for idx, spec in enumerate(specs, start=1):
                    label = str(spec["label"])
                    ext = str(spec.get("ext", ".mp4"))
                    out_path = str(tmp_root / f"{idx:02d}_{label.lower().replace(' ', '_')}{ext}")
                    builder = spec["builder"]
                    try:
                        built = builder(out_path)
                        if isinstance(built, tuple):
                            cmd, summary = built
                        else:
                            cmd = built
                            summary = str(spec.get("summary", "")).strip() or label
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                        )
                        stderr_tail = " | ".join(
                            line.strip()
                            for line in (proc.stderr or "").splitlines()[-3:]
                            if line.strip()
                        )
                        ok = proc.returncode == 0
                        results.append(
                            {
                                "label": label,
                                "summary": str(summary),
                                "status": "OK" if ok else "FAILED",
                                "reason": stderr_tail or ("" if ok else f"exit {proc.returncode}"),
                            }
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "label": label,
                                "summary": str(spec.get("summary", "")).strip() or label,
                                "status": "FAILED",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
        self.root.after(0, lambda: self._finish_codec_validation(results, error_message))

    def _finish_codec_validation(
        self,
        results: list[dict[str, str]],
        error_message: str = "",
    ) -> None:
        self._codec_validation_running = False
        if hasattr(self, "codec_validation_btn"):
            self.codec_validation_btn.configure(state=tk.NORMAL)
        if error_message:
            self.pipeline_run_status_var.set("Codec validation failed.")
            self._append_pipeline_popup_log("ERROR", "Codec Validation", error_message)
            messagebox.showerror("Codec Validation", error_message)
            return

        lines: list[str] = []
        has_failed = False
        for row in results:
            line = f"{row['label']} | {row['summary']} | {row['status']}"
            reason = str(row.get("reason", "")).strip()
            if reason:
                line = f"{line} | {reason}"
            lines.append(line)
            if row["status"] != "OK":
                has_failed = True
        body = "\n".join(lines) if lines else "No codec validation rows generated."
        self._append_pipeline_popup_log("INFO", "Codec Validation", body)
        self.pipeline_run_status_var.set(
            "Codec validation completed with failures." if has_failed else "Codec validation completed."
        )
        if has_failed:
            messagebox.showwarning("Codec Validation", body)
        else:
            messagebox.showinfo("Codec Validation", body)

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

    def _pipeline_collect_seg_scene_names_and_stems(
        self,
        seg_dir: str,
    ) -> tuple[list[str], list[str]]:
        root = Path(seg_dir)
        if not root.is_dir():
            return [], []
        names: list[str] = []
        stems: list[str] = []
        seen_stems: set[str] = set()
        for path in self._collect_video_files_for_patterns(seg_dir, self.VERIFY_VIDEO_PATTERNS):
            if not path.is_file():
                continue
            names.append(path.name)
            stem = self._pipeline_scene_stem_from_name(path.name)
            if stem and stem not in seen_stems:
                seen_stems.add(stem)
                stems.append(stem)
        return names, stems

    def _pipeline_collect_scene_coverage_stems(
        self,
        folder: str,
        scene_stems: list[str],
        patterns: list[str],
        *,
        must_contain: str = "",
    ) -> set[str]:
        covered: set[str] = set()
        if not scene_stems:
            return covered
        files = self._pipeline_collect_scene_matched_files(
            folder,
            patterns,
            scene_stems,
            must_contain=must_contain,
        )
        for path in files:
            name = path.name
            for stem in scene_stems:
                if self._pipeline_name_matches_scene_stem(name, stem):
                    covered.add(stem)
                    break
        return covered

    def _pipeline_count_scene_output_coverage(
        self,
        folder: str,
        scene_stems: list[str],
        patterns: list[str],
        *,
        must_contain: str = "",
    ) -> int:
        return len(
            self._pipeline_collect_scene_coverage_stems(
                folder,
                scene_stems,
                patterns,
                must_contain=must_contain,
            )
        )

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
        selected_names = {str(Path(x).name) for x in scene_names if str(x).strip()}
        selected_stems = {
            self._pipeline_scene_stem_from_name(x)
            for x in scene_names
            if self._pipeline_scene_stem_from_name(x)
        }
        if not selected_names and not selected_stems:
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
                    if file_name in selected_names or any(
                        self._pipeline_name_matches_scene_stem(file_name, stem)
                        for stem in selected_stems
                    ):
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
        normal_scene_state = self._pipeline_step_state.get("scenedetect", {"completed": False})
        normal_split_state = self._pipeline_step_state.get(
            "split_scenes",
            {"completed": False, "verified": "none"},
        )
        state["scenedetect"]["completed"] = bool(
            scene_files_ready or bool(normal_scene_state.get("completed", False))
        )
        state["split_scenes"]["completed"] = bool(
            scene_files_ready or bool(normal_split_state.get("completed", False))
        )
        state["depthcrafter"]["completed"] = self._pipeline_test_has_scene_outputs(
            self.depth_output_var.get().strip(),
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
        sharpen_expected, sharpen_err = self._collect_expected_sharpen_outputs(
            self.inpaint_input_var.get().strip(),
            self.inpaint_sharpness_csv_var.get().strip(),
        )
        state["sharpen"]["completed"] = bool(
            not sharpen_err
            and (
                len(sharpen_expected) == 0
                or self._count_named_existing_files(
                    self.inpaint_sharpen_output_var.get().strip(),
                    sharpen_expected,
                )
                >= len(sharpen_expected)
            )
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
                    preferred_inpainted_dir=self._preferred_inpainted_dir_for_consumers(),
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
                state[step]["verified"] = "quick" if prev_ver in {"quick", "deep"} else "none"
            else:
                state[step]["verified"] = "none"

        self._pipeline_test_step_state = state

    @staticmethod
    def _pipeline_test_path_var_names() -> list[str]:
        return [
            "scene_output_var",
            "depth_input_var",
            "depth_output_var",
            "splat_input_clips_var",
            "splat_input_depth_var",
            "splat_output_var",
            "splat_mask_output_var",
            "inpaint_input_var",
            "inpaint_mask_var",
            "inpaint_output_var",
            "inpaint_sharpen_output_var",
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

    @classmethod
    def _pipeline_test_state_file(cls, test_root: Path | str) -> Path:
        return Path(test_root).expanduser().resolve() / cls.PIPELINE_TEST_STATE_FILENAME

    def _capture_pipeline_test_prev_paths(self) -> dict[str, str]:
        prev_paths: dict[str, str] = {}
        for var_name in self._pipeline_test_path_var_names():
            var_obj = getattr(self, var_name, None)
            if var_obj is not None:
                prev_paths[var_name] = str(var_obj.get()).strip()
        return prev_paths

    def _pipeline_test_restore_should_wait(self) -> bool:
        if not self._pipeline_test_active:
            return False
        if self._any_pipeline_activity():
            return True
        pending = self._pipeline_pending_action
        if isinstance(pending, tuple) and len(pending) >= 2:
            step_name = str(pending[0]).strip().lower()
            action = str(pending[1]).strip().lower()
            if step_name and action in {"run", "verify"}:
                return True
        return False

    def _schedule_pipeline_test_restore(self, delay_ms: int = 150) -> None:
        if not self._pipeline_test_active or self._pipeline_test_restore_scheduled:
            return
        self._pipeline_test_restore_scheduled = True

        def _deferred_restore() -> None:
            self._pipeline_test_restore_scheduled = False
            self._restore_test_scene_subset()

        try:
            self.root.after(max(0, int(delay_ms)), _deferred_restore)
        except Exception:
            self._pipeline_test_restore_scheduled = False

    def _save_pipeline_test_resume_state(self, test_root: Path | str) -> None:
        test_root_path = Path(test_root).expanduser().resolve()
        payload = {
            "manifest": [str(x).strip() for x in (self._pipeline_test_manifest or []) if str(x).strip()],
            "scene_stems": [str(x).strip() for x in (self._pipeline_test_scene_stems or []) if str(x).strip()],
            "source_dir": str(self._pipeline_test_source_dir or "").strip(),
            "prev_paths": {
                str(k): str(v).strip()
                for k, v in dict(self._pipeline_test_prev_paths or {}).items()
                if str(k).strip()
            },
        }
        try:
            test_root_path.mkdir(parents=True, exist_ok=True)
            self._pipeline_test_state_file(test_root_path).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_pipeline_test_resume_state(self, test_root: Path | str) -> dict[str, object]:
        path = self._pipeline_test_state_file(test_root)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _run_startup_tasks(self) -> None:
        if self._recover_pipeline_test_subset_on_startup():
            return
        self._start_source_analysis_on_startup()

    def _recover_pipeline_test_subset_on_startup(self) -> bool:
        if self._pipeline_test_recovery_attempted or self._pipeline_test_active:
            return False
        self._pipeline_test_recovery_attempted = True

        work_root = Path(self.work_folder_var.get().strip() or "./work").resolve()
        test_root = work_root / ".pipeline_test_subset"
        if not test_root.is_dir():
            return False

        test_seg = test_root / "seg"
        if not test_seg.is_dir():
            return False

        meta = self._load_pipeline_test_resume_state(test_root)
        seg_files = self._collect_video_files_for_patterns(
            str(test_seg),
            self.VERIFY_VIDEO_PATTERNS,
        )
        seg_names = [p.name for p in seg_files if p.is_file()]
        if not seg_names:
            return False

        meta_manifest = [str(x).strip() for x in (meta.get("manifest") or []) if str(x).strip()]
        selected = [name for name in meta_manifest if (test_seg / name).is_file()]
        if not selected:
            selected = seg_names

        scene_stems = [self._pipeline_scene_stem_from_name(x) for x in selected]
        scene_stems = [s for s in scene_stems if s]
        if not scene_stems:
            return False

        current_prev_paths = self._capture_pipeline_test_prev_paths()
        meta_prev_paths_raw = meta.get("prev_paths") or {}
        meta_prev_paths = (
            {
                str(k): str(v).strip()
                for k, v in dict(meta_prev_paths_raw).items()
                if str(k).strip()
            }
            if isinstance(meta_prev_paths_raw, dict)
            else {}
        )
        prev_paths = {
            var_name: str(meta_prev_paths.get(var_name) or current_prev_paths.get(var_name) or "").strip()
            for var_name in self._pipeline_test_path_var_names()
        }

        test_depth = test_root / "depthmap"
        test_splat_root = test_root / "splat"
        test_splat_hires = test_splat_root / "hires"
        test_mask = test_root / "mask"
        test_output = test_root / "output"
        test_output_sharpen = test_root / "output-sharpen"
        test_mask_formerge = test_root / "mask_for_merge"
        test_sbs = test_root / "sbs"
        test_final = test_root / "final"
        sharp_csv_test = test_root / "sharpness_test.csv"
        autoct_csv_test = test_root / "autoct_test.csv"

        self.scene_output_var.set(str(test_seg))
        self.depth_input_var.set(str(test_seg))
        self.depth_output_var.set(str(test_depth))
        self.splat_input_clips_var.set(str(test_seg))
        self._sync_depth_to_splat_input_path()
        self.splat_output_var.set(str(test_splat_root))
        self.splat_mask_output_var.set(str(test_mask))
        self.inpaint_input_var.set(str(test_splat_hires))
        self.inpaint_mask_var.set(str(test_mask))
        self.inpaint_output_var.set(str(test_output))
        self.inpaint_sharpen_output_var.set(str(test_output_sharpen))
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
        self._pipeline_test_manifest = list(selected)
        self._pipeline_test_scene_stems = list(scene_stems)
        self._pipeline_test_source_dir = str(meta.get("source_dir") or current_prev_paths.get("scene_output_var") or "").strip()
        self._pipeline_test_dir = str(test_root)
        self._pipeline_test_prev_paths = dict(prev_paths)
        self._pipeline_test_restore_scheduled = False
        self._pipeline_test_step_state = {
            key: {
                "completed": bool((self._pipeline_step_state.get(key) or {}).get("completed", False)),
                "verified": (
                    str((self._pipeline_step_state.get(key) or {}).get("verified", "none")).strip().lower()
                    if str((self._pipeline_step_state.get(key) or {}).get("verified", "none")).strip().lower()
                    in {"none", "quick", "deep"}
                    else "none"
                ),
            }
            for key, _label in self.PIPELINE_STEPS
        }
        self._pipeline_recompute_test_step_state()
        self._preview_depth_command()
        self._preview_splat_command()
        self._preview_inpaint_command()
        self._preview_merge_command()
        self._preview_join_command()
        self._refresh_pipeline_status_panel()

        recover_msg = (
            f"Recovered Test Run subset from disk ({len(selected)} scene clip(s)). "
            "Resuming automatically from the first incomplete step."
        )
        self.pipeline_run_status_var.set(recover_msg)
        self._append_pipeline_popup_log("INFO", "Startup Test Run Recovery", recover_msg)
        self._pipeline_reset_skip_notices()
        self.root.after(150, self._resume_recovered_test_subset_when_ready)
        return True

    def _resume_recovered_test_subset_when_ready(self) -> None:
        if not self._pipeline_test_active:
            return
        if self._any_pipeline_activity():
            self.root.after(250, self._resume_recovered_test_subset_when_ready)
            return
        self._pipeline_start_resume()

    def _default_pipeline_step_state(self) -> dict[str, dict[str, object]]:
        return pm_state.default_pipeline_step_state(self)

    def _pipeline_state_path(self) -> Path:
        return pm_state.pipeline_state_path(self)

    def _load_pipeline_state(self) -> None:
        pm_state.load_pipeline_state(self)

    def _save_pipeline_state(self) -> None:
        pm_state.save_pipeline_state(self)

    def _is_pipeline_step_required(self, step: str) -> bool:
        return pm_state.is_pipeline_step_required(self, step)

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
                    verify_w.configure(
                        text="Disabled" if step == "sharpen" else "N/A",
                        fg="#999999",
                    )
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
        self._pipeline_stop_requested = False
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
        self.scene_codec_var.set(self.DEFAULT_SCENE_CODEC)

        # Depth defaults.
        self.depth_mode_var.set("Auto (recommended)")
        self.depth_chunk_size_var.set("65")
        self.depth_overlap_var.set("15")
        self.depth_inference_steps_var.set("4")
        self.depth_cpu_offload_var.set("model")
        self.depth_seed_var.set("42")
        self.depth_guidance_scale_var.set("1.0")
        self.depth_decode_chunk_size_var.set("2")
        self.depth_restart_every_var.set("100")
        self.depth_debug_mem_var.set(True)
        self.depth_scale_factor_var.set(self.DEFAULT_DEPTH_SCALE_FACTOR)
        self.depth_glob_var.set("*.mp4")
        self.depth_runtime_mode_var.set("original")
        self.depth_worker_script_var.set("./runners/depthcrafter_nogui_batch.py")
        self.depth_codec_var.set(self.DEFAULT_SCENE_CODEC)
        self._on_depth_mode_changed()

        # Splat defaults.
        self.splat_mode_var.set("Auto (recommended)")
        self.splat_batch_size_var.set("50")
        self.splat_workers_var.set("8")
        self.splat_disparity_var.set("20")
        self.splat_codec_var.set(self.DEFAULT_SCENE_CODEC)
        self._on_splat_mode_changed()

        # Inpaint defaults.
        self.inpaint_mode_var.set("Auto (recommended)")
        self.inpaint_frames_chunk_var.set("22")
        self._inpaint_chunk_manual_cache = "22"
        self.inpaint_dynamic_chunk_var.set(True)
        self.inpaint_cpu_offload_var.set("none")
        self.inpaint_tile_mode_var.set("1 and 2")
        self.inpaint_tile1_max_size_var.set("3,25,32,43,60,88")
        self.inpaint_tile2_max_size_var.set("71,86,107,117,117,117")
        self.inpaint_dynamic_resolution_var.set(True)
        self.inpaint_resolution_limit_var.set("90%")
        self.inpaint_dynamic_visible_chunk_steps5_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS5_DEFAULT
        )
        self.inpaint_dynamic_visible_chunk_steps6_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS6_DEFAULT
        )
        self.inpaint_dynamic_visible_chunk_steps7_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS7_DEFAULT
        )
        self.inpaint_dynamic_visible_chunk_steps8_plus_var.set(
            self.INPAINT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS_DEFAULT
        )
        self.inpaint_dynamic_hold_divisor_var.set(
            self.INPAINT_DYNAMIC_STATIC_MASK_DIVISOR_DEFAULT
        )
        self.inpaint_sharpness_workers_var.set("19")
        self.inpaint_use_sharpen_var.set(True)
        self.inpaint_sharpen_workers_var.set("19")
        self.inpaint_codec_var.set(self.DEFAULT_SCENE_CODEC)
        self._on_inpaint_mode_changed()

        # Merge defaults.
        self.merge_mode_var.set("Auto (recommended)")
        self.merge_parallel_workers_var.set("4")
        self.merge_codec_var.set(self.DEFAULT_SCENE_CODEC)
        self._on_merge_mode_changed()

        # Join defaults.
        self.join_mode_var.set("Auto (recommended)")
        self.join_crf_var.set("12")
        self._on_join_mode_changed()

        # Options defaults.
        self.scene_split_threads_var.set(str(self.DEFAULT_SPLIT_SCENES_WORKERS))
        self.verify_scenes_workers_var.set("19")
        self.pipeline_verify_after_var.set("Quick")
        self.pipeline_test_run_files_var.set(str(self.DEFAULT_PIPELINE_TEST_RUN_FILES))
        self.global_encoder_mode_var.set("lossless")
        self.global_ffmpeg_extra_args_var.set("")
        self._set_retry_policy_vars_to_defaults()
        self._on_depth_retry_policy_changed()
        self._on_inpaint_retry_policy_changed()
        self.resume_enabled_var.set(True)
        self.stop_on_error_var.set(True)
        self.auto_advance_var.set(False)

        self._refresh_standard_paths()
        self._apply_option_states()
        self._refresh_global_encoder_preview()
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
        pm_state.pipeline_set_completed_in_state(state, step, value)

    @staticmethod
    def _pipeline_set_verified_in_state(
        state: dict[str, dict[str, object]],
        step: str,
        mode: str,
    ) -> None:
        pm_state.pipeline_set_verified_in_state(state, step, mode)

    def _pipeline_set_completed(self, step: str, value: bool) -> None:
        self._pipeline_set_completed_in_state(self._pipeline_step_state, step, value)

    def _pipeline_set_verified(self, step: str, mode: str) -> None:
        self._pipeline_set_verified_in_state(self._pipeline_step_state, step, mode)

    @staticmethod
    def _pipeline_verified_rank(mode: str) -> int:
        return pm_state.pipeline_verified_rank(mode)

    def _pipeline_set_verified_best_in_state(
        self,
        state: dict[str, dict[str, object]],
        step: str,
        mode: str,
    ) -> None:
        pm_state.pipeline_set_verified_best_in_state(state, step, mode)

    def _pipeline_set_verified_best(self, step: str, mode: str) -> None:
        self._pipeline_set_verified_best_in_state(self._pipeline_step_state, step, mode)

    def _sync_pipeline_csv_done_flags_in_state(
        self,
        state: dict[str, dict[str, object]],
    ) -> None:
        pm_state.sync_pipeline_csv_done_flags_in_state(self, state)

    def _sync_pipeline_csv_done_flags(self) -> None:
        pm_state.sync_pipeline_csv_done_flags(self)

    def _pipeline_mark_previous_steps_done_verified_in_state(
        self,
        state: dict[str, dict[str, object]],
        step: str,
        mode: str,
    ) -> None:
        pm_state.pipeline_mark_previous_steps_done_verified_in_state(
            self, state, step, mode
        )

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
        pm_state.pipeline_invalidate_from_in_state(
            self, state, step, include_current=include_current
        )

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
        return f"{stem}-Scene-{int(scene_number):04d}.mp4"

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
                    start_frame_raw = norm.get("start frame") or ""
                    end_frame_raw = norm.get("end frame") or ""
                    length_frames_raw = (
                        norm.get("length (frames)")
                        or norm.get("length frames")
                        or ""
                    )
                    start_frame = self._parse_scene_intish(start_frame_raw)
                    end_frame = self._parse_scene_intish(end_frame_raw)
                    frame_count = self._parse_scene_intish(length_frames_raw)
                    if start_frame is not None and start_frame <= 0:
                        start_frame = None
                    if frame_count is not None and frame_count <= 0:
                        frame_count = None
                    if (
                        start_frame is not None
                        and frame_count is None
                        and end_frame is not None
                        and end_frame >= start_frame
                    ):
                        frame_count = (end_frame - start_frame) + 1
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
                            "start_frame": int(start_frame) if start_frame is not None else 0,
                            "end_frame": int(end_frame) if end_frame is not None else 0,
                            "frame_count": int(frame_count) if frame_count is not None else 0,
                            "start_sec": float(start_sec),
                            "end_sec": float(end_sec),
                        }
                    )
        except Exception as exc:
            return [], f"Failed reading Scene CSV: {type(exc).__name__}: {exc}"
        if not rows:
            return [], f"No valid scene rows found in CSV: {csv_path}"
        return rows, ""

    @staticmethod
    def _parse_scene_intish(value) -> int | None:
        try:
            txt = str(value or "").strip()
        except Exception:
            txt = ""
        if not txt:
            return None
        try:
            return int(txt)
        except Exception:
            pass
        try:
            return int(float(txt))
        except Exception:
            return None

    def _scene_csv_expected_by_name(
        self,
        source_path: str,
        scene_csv_path: str | None = None,
    ) -> tuple[dict[str, dict[str, int]], str]:
        entries, err = self._load_scene_csv_entries(scene_csv_path)
        if err:
            return {}, err
        out: dict[str, dict[str, int]] = {}
        for idx, entry in enumerate(entries, start=1):
            scene_num = int(entry.get("scene_number", idx))
            out_name = self._scene_output_filename(source_path, scene_num)
            out[out_name] = {
                "scene_number": scene_num,
                "frame_count": int(entry.get("frame_count", 0) or 0),
            }
        return out, ""

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

    def _collect_join_scene_count_stats(self) -> dict[str, int | bool]:
        seg_dir = self.scene_output_var.get().strip()
        seg_mono_dir = self.join_seg_mono_var.get().strip()
        sbs_dir = self.join_input_var.get().strip()
        seg_count = self._count_video_files(seg_dir)
        seg_mono_count = 0
        if seg_mono_dir and os.path.isdir(seg_mono_dir):
            same_as_seg = False
            try:
                same_as_seg = Path(seg_mono_dir).resolve() == Path(seg_dir).resolve()
            except Exception:
                same_as_seg = False
            if not same_as_seg:
                seg_mono_count = self._count_video_files(seg_mono_dir)
        sbs_count = self._count_video_files(sbs_dir)
        expected_total = int(seg_count) + int(seg_mono_count)
        return {
            "seg_count": int(seg_count),
            "seg_mono_count": int(seg_mono_count),
            "sbs_count": int(sbs_count),
            "expected_total": int(expected_total),
            "counts_match": bool(int(sbs_count) == int(expected_total)),
        }

    def _join_incomplete_flag_path(self) -> Path:
        work_dir = self.work_folder_var.get().strip() or "./work"
        return Path(work_dir).resolve() / ".join_incomplete.flag"

    def _join_incomplete_flag_exists(self) -> bool:
        try:
            return self._join_incomplete_flag_path().is_file()
        except Exception:
            return False

    def _set_join_incomplete_flag(self, incomplete: bool) -> None:
        flag_path = self._join_incomplete_flag_path()
        try:
            if incomplete:
                flag_path.parent.mkdir(parents=True, exist_ok=True)
                flag_path.write_text(
                    "Join Scenes was last completed in incomplete mode.\n",
                    encoding="utf-8",
                )
            elif flag_path.exists():
                flag_path.unlink()
        except Exception:
            pass

    @staticmethod
    def _scene_csv_exists(scene_csv_path: str) -> bool:
        csv_path = str(scene_csv_path or "").strip()
        if not csv_path:
            return False
        try:
            return Path(csv_path).is_file() and Path(csv_path).stat().st_size > 0
        except Exception:
            return False

    def _pipeline_check_files(self, show_popup: bool = True) -> bool:
        seg_dir = self.scene_output_var.get().strip()
        out_dir = self.merge_output_var.get().strip()
        scene_names, scene_stems = self._pipeline_collect_seg_scene_names_and_stems(seg_dir)
        scene_csv_path = self._scene_csv_path()
        scene_csv_present = self._scene_csv_exists(scene_csv_path)
        scene_entries, scene_csv_err = self._load_scene_csv_entries(scene_csv_path)
        split_csv_ok = bool(scene_entries) and not scene_csv_err

        expected_outputs, missing_split_outputs, split_cov_err = self._collect_expected_split_scene_outputs(
            seg_dir
        )
        split_expected_count = len(expected_outputs)
        split_missing_count = len(missing_split_outputs)

        if (not scene_names) and (not scene_csv_present):
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
        merged_covered_stems = self._pipeline_collect_scene_coverage_stems(
            out_dir,
            scene_stems,
            self.VERIFY_VIDEO_PATTERNS,
            must_contain="_merged_",
        )
        for name in scene_names:
            stem = self._pipeline_scene_stem_from_name(name)
            if stem in merged_covered_stems:
                completed.append(name)
            else:
                incomplete.append(name)

        seg_count = len(scene_names)
        seg_ref_count = len(scene_stems)
        depth_count = self._pipeline_count_scene_output_coverage(
            self.depth_output_var.get().strip(),
            scene_stems,
            self.VERIFY_VIDEO_PATTERNS,
        )
        splat_count = self._pipeline_count_scene_output_coverage(
            self._resolve_splat_hires_dir(),
            scene_stems,
            self.VERIFY_VIDEO_PATTERNS,
            must_contain="_splatted",
        )
        inpaint_count = self._pipeline_count_scene_output_coverage(
            self.inpaint_output_var.get().strip(),
            scene_stems,
            ["*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"],
            must_contain="_inpainted_",
        )
        sharpen_expected, sharpen_err = self._collect_expected_sharpen_outputs(
            self.inpaint_input_var.get().strip(),
            self.inpaint_sharpness_csv_var.get().strip(),
        )
        sharpen_count = self._count_named_existing_files(
            self.inpaint_sharpen_output_var.get().strip(),
            sharpen_expected,
        )
        mask_formerge_count = self._pipeline_count_scene_output_coverage(
            self.merge_mask_formerge_var.get().strip(),
            scene_stems,
            self.VERIFY_REPLACE_MASK_PATTERNS,
            must_contain="_replace_mask",
        )
        merge_count = len(merged_covered_stems)
        mono_to_sbs_ok, _mono_msg, _mono_broken_output, _mono_broken_reference = (
            self._verify_join_mono_outputs_coverage(cleanup_incomplete=False)
        )
        join_counts = self._collect_join_scene_count_stats()
        join_done = Path(self.join_output_var.get().strip()).is_file()
        remux_done = Path(self._default_remux_output_path()).is_file()
        sharp_done = Path(self.inpaint_sharpness_csv_var.get().strip()).is_file()
        autoct_done = Path(self.merge_autoct_csv_var.get().strip()).is_file()
        split_ok = bool(split_csv_ok and split_expected_count > 0 and split_missing_count == 0 and not split_cov_err)
        split_ref_count = split_expected_count if split_expected_count > 0 else seg_ref_count
        prev_state = self._pipeline_step_state
        prev_scene_done = bool((prev_state.get("scenedetect") or {}).get("completed", False))
        prev_split_done = bool((prev_state.get("split_scenes") or {}).get("completed", False))
        scene_ref_ready = seg_ref_count > 0
        completed_map: dict[str, bool] = {
            "scenedetect": bool(prev_scene_done or scene_csv_present),
            "split_scenes": bool(prev_split_done or split_ok),
            "depthcrafter": bool(scene_ref_ready and depth_count >= seg_ref_count),
            "splatting": bool(scene_ref_ready and splat_count >= seg_ref_count),
            "inpaint": bool(scene_ref_ready and inpaint_count >= seg_ref_count),
            "sharpen": (
                bool(
                    not sharpen_err
                    and (
                        len(sharpen_expected) == 0
                        or sharpen_count >= len(sharpen_expected)
                    )
                )
                if self._is_pipeline_step_required("sharpen")
                else False
            ),
            "mask_for_merge": bool(scene_ref_ready and mask_formerge_count >= seg_ref_count),
            "merging": bool(scene_ref_ready and merge_count >= seg_ref_count),
            "mono_to_sbs": bool(mono_to_sbs_ok),
            "join": bool(
                join_done
                and bool(join_counts.get("counts_match", False))
                and not self._join_incomplete_flag_exists()
            ),
            "remux": bool(remux_done),
        }

        sharpness_csv_idx = next(
            idx for idx, (step_name, _label) in enumerate(self.PIPELINE_STEPS) if step_name == "sharpness_csv"
        )
        autoct_csv_idx = next(
            idx for idx, (step_name, _label) in enumerate(self.PIPELINE_STEPS) if step_name == "autoct_csv"
        )

        def _upstream_steps_completed(limit_idx: int) -> bool:
            for step_name, _label in self.PIPELINE_STEPS[:limit_idx]:
                if not self._is_pipeline_step_required(step_name):
                    continue
                if not completed_map.get(step_name, False):
                    return False
            return True

        prev_sharp_done = bool((prev_state.get("sharpness_csv") or {}).get("completed", False))
        prev_autoct_done = bool((prev_state.get("autoct_csv") or {}).get("completed", False))
        completed_map["sharpness_csv"] = (
            bool(prev_sharp_done or sharp_done)
            if self._is_pipeline_step_required("sharpness_csv") and _upstream_steps_completed(sharpness_csv_idx)
            else False
        )
        completed_map["autoct_csv"] = (
            bool(prev_autoct_done or autoct_done)
            if self._is_pipeline_step_required("autoct_csv") and _upstream_steps_completed(autoct_csv_idx)
            else False
        )

        for step_name, _label in self.PIPELINE_STEPS:
            self._pipeline_set_completed(step_name, completed_map.get(step_name, False))

        self._pipeline_file_scan = {
            "seg_total": seg_count,
            "seg_expected": split_ref_count,
            "seg_scene_stems": list(scene_stems),
            "split_ok_actual": bool(split_ok),
            "split_missing": [str(Path(p).name) for p in missing_split_outputs],
            "join_expected_total": int(join_counts.get("expected_total", 0)),
            "join_sbs_total": int(join_counts.get("sbs_count", 0)),
            "join_incomplete_flag": bool(self._join_incomplete_flag_exists()),
            "completed_final": completed,
            "incomplete_final": incomplete,
        }
        self._pipeline_check_files_done = True
        self.pipeline_checked_files_var.set(
            (
                f"Check Files: csv={'present' if scene_csv_present else 'missing'}, "
                f"split={seg_count}/{split_ref_count}, "
                f"final done={len(completed)}, incomplete={len(incomplete)}"
            )
        )
        self._refresh_pipeline_status_panel()
        self._save_pipeline_state()

        if show_popup:
            csv_details = ""
            if scene_csv_present and scene_csv_err:
                csv_details = (
                    "\nSplit reference details "
                    "(SceneDetect CSV present, but unusable for split verification): "
                    f"{scene_csv_err}"
                )
            elif split_cov_err:
                csv_details = f"\nSplit CSV details: {split_cov_err}"
            messagebox.showinfo(
                "Check Files",
                (
                    f"Scan completed.\n\n"
                    f"Scene CSV: {'PRESENT' if scene_csv_present else 'MISSING'}\n"
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
        if not self._pipeline_check_files(show_popup=False):
            self._pipeline_sync_noninteractive_mode()
            return
        split_state = self._pipeline_step_state.get("split_scenes", {"verified": "none"})
        if int(self._pipeline_file_scan.get("seg_total") or 0) < 1:
            messagebox.showwarning(
                "Test Run",
                "No scene clips found in seg. Test Run requires split scene files.",
            )
            self._pipeline_sync_noninteractive_mode()
            return
        split_verified = str(split_state.get("verified", "none")).strip().lower()
        if split_verified != "quick":
            messagebox.showwarning(
                "Test Run",
                (
                    "Run Verify Scenes (Quick) first.\n\n"
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
        self._pipeline_reset_skip_notices()
        self._pipeline_autorun = True
        self._pipeline_stop_requested = False
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
        test_splat_root = test_root / "splat"
        test_splat_hires = test_splat_root / "hires"
        test_mask = test_root / "mask"
        test_output = test_root / "output"
        test_output_sharpen = test_root / "output-sharpen"
        test_mask_formerge = test_root / "mask_for_merge"
        test_sbs = test_root / "sbs"
        test_final = test_root / "final"
        for d in (
            test_seg,
            test_depth,
            test_splat_root,
            test_splat_hires,
            test_mask,
            test_output,
            test_output_sharpen,
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
        splat_src_root = prev_paths.get("splat_output_var", "")
        if splat_src_root:
            sroot = Path(splat_src_root)
            splat_hires_src = str((sroot / "hires").resolve()) if (sroot / "hires").is_dir() else str(sroot.resolve())
        else:
            splat_hires_src = ""
        mask_src = prev_paths.get("splat_mask_output_var", "")
        inpaint_src = prev_paths.get("inpaint_output_var", "")
        inpaint_sharpen_src = prev_paths.get("inpaint_sharpen_output_var", "")
        mask_formerge_src = prev_paths.get("merge_mask_formerge_var", "")
        merge_src = prev_paths.get("merge_output_var", "")

        self._pipeline_link_scene_files(
            depth_src,
            str(test_depth),
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
            inpaint_sharpen_src,
            str(test_output_sharpen),
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
        self.splat_input_clips_var.set(str(test_seg))
        self._sync_depth_to_splat_input_path()
        self.splat_output_var.set(str(test_splat_root))
        self.splat_mask_output_var.set(str(test_mask))
        self.inpaint_input_var.set(str(test_splat_hires))
        self.inpaint_mask_var.set(str(test_mask))
        self.inpaint_output_var.set(str(test_output))
        self.inpaint_sharpen_output_var.set(str(test_output_sharpen))
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
        self._pipeline_test_restore_scheduled = False
        self._pipeline_test_step_state = {
            key: {
                "completed": bool((self._pipeline_step_state.get(key) or {}).get("completed", False)),
                "verified": (
                    str((self._pipeline_step_state.get(key) or {}).get("verified", "none")).strip().lower()
                    if str((self._pipeline_step_state.get(key) or {}).get("verified", "none")).strip().lower()
                    in {"none", "quick", "deep"}
                    else "none"
                ),
            }
            for key, _label in self.PIPELINE_STEPS
        }
        self._save_pipeline_test_resume_state(test_root)
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
        _sync_dir(test_root / "mask", "splat_mask_output_var", ["*_replace_mask.*"], must_contain="_replace_mask")
        _sync_dir(
            test_root / "output",
            "inpaint_output_var",
            ["*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"],
            must_contain="_inpainted_",
        )
        _sync_dir(
            test_root / "output-sharpen",
            "inpaint_sharpen_output_var",
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

    def _restore_test_scene_subset(self, *, force: bool = False) -> bool:
        if not self._pipeline_test_active:
            self._pipeline_test_restore_scheduled = False
            return False
        if not force and self._pipeline_test_restore_should_wait():
            self._schedule_pipeline_test_restore()
            return False
        self._pipeline_test_restore_scheduled = False
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
        self._sync_depth_to_splat_input_path()

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
        return True

    def _pipeline_start_resume(self) -> None:
        pm_orchestrator.pipeline_start_resume(self)

    def _pipeline_split_scenes_gate_pending(self) -> bool:
        return pm_orchestrator.pipeline_split_scenes_gate_pending(self)

    def _pipeline_reset_skip_notices(self) -> None:
        pm_orchestrator.pipeline_reset_skip_notices(self)

    def _pipeline_maybe_log_completed_autoct_skip(self, next_step: str) -> None:
        pm_orchestrator.pipeline_maybe_log_completed_autoct_skip(self, next_step)

    def _show_pipeline_force_info(self, title: str, message: str) -> None:
        pm_orchestrator.show_pipeline_force_info(self, title, message)

    def _pipeline_trigger_next_action(self) -> None:
        pm_orchestrator.pipeline_trigger_next_action(self)

    def _pipeline_next_action(self) -> tuple[str, str, str] | None:
        return pm_orchestrator.pipeline_next_action(self)

    def _pipeline_dispatch_run(self, step: str) -> bool:
        return pm_orchestrator.pipeline_dispatch_run(self, step)

    def _pipeline_dispatch_verify(self, step: str, mode: str) -> bool:
        return pm_orchestrator.pipeline_dispatch_verify(self, step, mode)

    def _any_pipeline_activity(self) -> bool:
        return pm_orchestrator.any_pipeline_activity(self)

    def _pipeline_on_run_finished(
        self,
        step: str,
        success: bool,
        *,
        mark_completed: bool = True,
    ) -> None:
        pm_orchestrator.pipeline_on_run_finished(
            self,
            step,
            success,
            mark_completed=mark_completed,
        )

    def _pipeline_on_verify_finished(
        self,
        step: str,
        success: bool,
        mode: str,
        retry_on_failure: bool = True,
    ) -> None:
        pm_orchestrator.pipeline_on_verify_finished(
            self,
            step,
            success,
            mode,
            retry_on_failure=retry_on_failure,
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

    def _on_depth_mode_changed(self, _event=None) -> None:
        mode = self.depth_mode_var.get().strip()
        if mode == "Manual":
            self.depth_info_text_var.set(self.DEPTH_MANUAL_INFO)
        else:
            self.depth_mode_var.set("Auto (recommended)")
            self.depth_info_text_var.set(self.DEPTH_AUTO_INFO)
            self._reset_depth_auto_locked_defaults()
        self._apply_depth_control_states()

    def _reset_depth_auto_locked_defaults(self) -> None:
        # Fields disabled in Auto mode are informational and update dynamically.
        self.depth_runtime_mode_var.set("original")
        self.depth_overlap_var.set("15")
        self.depth_inference_steps_var.set("4")
        self.depth_seed_var.set("42")
        self.depth_scale_factor_var.set(self.DEFAULT_DEPTH_SCALE_FACTOR)
        self.depth_scale_factor_text_var.set(f"{float(self.depth_scale_factor_var.get()):.2f}x")
        self.depth_worker_script_var.set(self._resolve_depth_worker_script("original"))
        self._update_depth_resolution_preview()

    def _on_depth_runtime_mode_selected(self, _event=None) -> None:
        mode = self._normalize_depth_runtime_mode(self.depth_runtime_mode_var.get())
        self.depth_runtime_mode_var.set(mode)
        self.depth_worker_script_var.set(self._resolve_depth_worker_script(mode))
        if mode == "stream":
            messagebox.showwarning("DepthCrafter", self.DEPTH_STREAM_WARNING)
        self._preview_depth_command()

    def _sync_depth_to_splat_input_path(self) -> None:
        preferred = self.depth_output_var.get().strip()
        self.splat_input_depth_var.set(os.path.normpath(preferred) if preferred else "")

    def _refresh_depth_action_buttons(self, is_running: bool | None = None) -> None:
        if is_running is None:
            is_running = bool(self._depth_thread and self._depth_thread.is_alive())
        verify_active = bool(self._verify_running)
        self.depth_preview_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.depth_run_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        verify_state = tk.DISABLED if (is_running or verify_active) else tk.NORMAL
        self.depth_verify_quick_btn.configure(state=verify_state)

    def _apply_depth_control_states(self) -> None:
        mode_manual = self.depth_mode_var.get().strip() == "Manual"

        self.depth_chunk_size_entry.configure(state=tk.NORMAL)
        if mode_manual:
            self.depth_overlap_entry.configure(state=tk.NORMAL)
            self.depth_inference_steps_entry.configure(state=tk.NORMAL)
            self.depth_seed_entry.configure(state=tk.NORMAL)
            self.depth_scale_factor_scale.configure(state=tk.NORMAL)
        else:
            self.depth_overlap_entry.configure(state=tk.DISABLED)
            self.depth_inference_steps_entry.configure(state=tk.DISABLED)
            self.depth_seed_entry.configure(state=tk.DISABLED)
            self.depth_scale_factor_scale.configure(state=tk.DISABLED)
        self.depth_cpu_offload_combo.configure(state="readonly")
        if mode_manual:
            self.depth_runtime_mode_combo.configure(state="readonly")
        else:
            self.depth_runtime_mode_var.set("original")
            self.depth_runtime_mode_combo.configure(state=tk.DISABLED)
        self.depth_worker_script_var.set(
            self._resolve_depth_worker_script(self.depth_runtime_mode_var.get())
        )
        self.depth_res_x_entry.configure(state=tk.DISABLED)
        self.depth_res_y_entry.configure(state=tk.DISABLED)

        self._sync_depth_to_splat_input_path()
        self._update_depth_resolution_preview()
        self._preview_depth_command()
        self._preview_splat_command()
        self._refresh_depth_action_buttons()
        self._refresh_pipeline_status_panel()

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

    def _migrate_global_encoder_mode_from_legacy(self) -> str:
        existing = str(self._config.get("global_encoder_mode", "")).strip()
        if existing:
            return normalize_global_encoder_mode(existing)
        legacy_quality = str(self._config.get("scene_crf", "")).strip()
        if legacy_quality == "0":
            return "crf/qp 0"
        if legacy_quality == "1":
            return "crf/qp 1"
        return "lossless"

    def _normalize_depth_runtime_mode(self, value: object) -> str:
        mode = str(value or "").strip().lower()
        if mode in self.DEPTH_RUNTIME_MODE_CHOICES:
            return mode
        return "original"

    def _resolve_depth_worker_script(self, mode: object) -> str:
        resolved_mode = self._normalize_depth_runtime_mode(mode)
        return str(
            self.DEPTH_RUNTIME_MODE_TO_SCRIPT.get(
                resolved_mode,
                self.DEPTH_RUNTIME_MODE_TO_SCRIPT["original"],
            )
        )

    def _migrate_depth_runtime_mode_from_legacy(self) -> str:
        existing = str(self._config.get("depth_runtime_mode", "")).strip()
        if existing:
            return self._normalize_depth_runtime_mode(existing)
        legacy_worker = str(self._config.get("depth_worker_script", "")).strip().lower()
        if "stream" in legacy_worker:
            return "stream"
        return "original"

    def _normalize_depth_retry_window_size(self, value: object, fallback: str = "65") -> str:
        sval = str(value or "").strip()
        try:
            parsed = int(float(sval))
        except Exception:
            parsed = int(float(fallback))
        return str(max(1, parsed))

    def _normalize_depth_retry_overlap(self, value: object, fallback: str = "15") -> str:
        sval = str(value or "").strip()
        try:
            parsed = int(float(sval))
        except Exception:
            parsed = int(float(fallback))
        return str(max(0, parsed))

    def _normalize_depth_retry_offset(self, value: object, fallback: str = "0") -> str:
        sval = str(value or "").strip()
        if sval.startswith("+"):
            sval = sval[1:]
        try:
            parsed = int(float(sval))
        except Exception:
            fb = str(fallback).strip()
            if fb.startswith("+"):
                fb = fb[1:]
            try:
                parsed = int(float(fb))
            except Exception:
                parsed = 0
        parsed = max(-20, min(20, parsed))
        parsed = int(round(parsed / 5.0) * 5)
        return f"+{parsed}" if parsed > 0 else str(parsed)

    def _current_global_encoder_mode(self) -> str:
        mode = normalize_global_encoder_mode(self.global_encoder_mode_var.get())
        if self.global_encoder_mode_var.get().strip() != mode:
            self.global_encoder_mode_var.set(mode)
        return mode

    def _current_global_ffmpeg_extra_args(self) -> str:
        return str(self.global_ffmpeg_extra_args_var.get() or "").strip()

    def _resolve_color_profile_for_codec(self, codec: str):
        codec_value = self._normalize_ffmpeg_codec(codec, self.DEFAULT_SCENE_CODEC)
        return resolve_color_encoding_profile(codec_value, self._current_global_encoder_mode())

    @staticmethod
    def _append_extra_args_to_tokens(tokens: list[str], extra_args: str) -> list[str]:
        extra = str(extra_args or "").strip()
        if not extra:
            return list(tokens)
        return list(tokens) + shlex.split(extra)

    def _refresh_global_encoder_preview(self) -> None:
        extra_args = self._current_global_ffmpeg_extra_args()
        lines: list[str] = []
        for codec in self.FFMPEG_CODEC_CHOICES:
            try:
                profile = resolve_color_encoding_profile(codec, self._current_global_encoder_mode())
                lines.append(profile_preview_line(profile, extra_args))
            except Exception as exc:
                lines.append(f"{codec} FAILED: {exc}")
        self.global_encoder_preview_var.set(" | ".join(lines))

    def _on_global_encoder_mode_selected(self, _event=None) -> None:
        self._on_global_encoder_settings_changed()

    def _on_global_encoder_settings_changed(self, *_args) -> None:
        self._refresh_global_encoder_preview()
        self._preview_scene_command()
        self._preview_splat_command()
        self._preview_inpaint_command()
        self._preview_merge_command()

    def _build_depth_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        return pm_builders.build_depth_runner_payload(self)

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

        launcher_script = runner_path("run_depthcrafter_nogui_batch.sh")
        if not launcher_script.is_file():
            messagebox.showerror("DepthCrafter", f"Launcher not found:\n{launcher_script}")
            return

        worker_abs = resolve_repo_path(
            self.depth_worker_script_var.get().strip() or "./runners/depthcrafter_nogui_batch.py"
        )
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
        self._depth_stop_marker_path = env_updates.get("STOP_MARKER", "").strip()
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
            marker_path = self._touch_stop_marker_file(
                str(self._depth_stop_marker_path or "").strip()
                or os.path.join(
                    self.depth_output_var.get().strip() or "./work/depthmap",
                    ".stop_after_current",
                ),
                self._append_depth_log,
            )
            if marker_path:
                self._depth_stop_marker_path = marker_path
        else:
            self.depth_status_var.set("Force stop requested...")
            self._append_depth_log("[STOP] force stop requested.")
            self._send_depth_signal(signal.SIGINT)
            self.root.after(1000, self._force_kill_depth)
        self._refresh_pipeline_run_button()

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
        self.depth_stop_btn.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        self._refresh_depth_action_buttons(is_running)
        if is_running:
            self.depth_stop_btn.configure(text="Stop")
        else:
            self.depth_stop_btn.configure(text="Stop")
            self._depth_stop_clicks = 0
            self._depth_stop_requested = False
            self._depth_stop_marker_path = ""
        self._refresh_pipeline_run_button()

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
        self.splat_dilate_y_var.set("1.5")
        self.splat_blur_x_var.set("0")
        self.splat_blur_y_var.set("0")
        self.splat_dilate_left_var.set("1")
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

    def _build_splat_runner_payload(self) -> tuple[list[str], dict[str, str], str]:
        return pm_builders.build_splat_runner_payload(self)

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

        launcher_script = (
            resolve_repo_path(cmd[1]) if len(cmd) > 1 else runner_path("run_splatting_runner.sh")
        )
        if not launcher_script.is_file():
            messagebox.showerror("Splatting", f"Launcher not found:\n{launcher_script}")
            return

        runner_script = resolve_repo_path(
            env_updates.get("RUNNER", str(runner_path("batch_splatting_runner.py")))
        )
        if not runner_script.is_file():
            messagebox.showerror("Splatting", f"Runner not found:\n{runner_script}")
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
        self._splat_stop_marker_path = env_updates.get("STOP_MARKER", "").strip()
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
    def _collect_inpainted_scene_files(
        inpainted_dir: str,
        preferred_inpainted_dir: str = "",
    ) -> list[Path]:
        files: dict[str, Path] = {}
        for folder in (preferred_inpainted_dir, inpainted_dir):
            if not folder or not os.path.isdir(folder):
                continue
            root = Path(folder)
            for pat in ("*_inpainted_right_eye.mp4", "*_inpainted_sbs.mp4"):
                for p in root.glob(pat):
                    if p.is_file() and p.name not in files:
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
        expected_pix_fmt = "yuv444p"
        try:
            mono_profile = resolve_color_encoding_profile(
                self._normalize_ffmpeg_codec(
                    self.merge_codec_var.get(),
                    self.scene_codec_var.get().strip() or self.DEFAULT_SCENE_CODEC,
                ),
                self._current_global_encoder_mode(),
            )
            expected_pix_fmt = str(mono_profile.pix_fmt or "yuv444p").strip().lower()
        except Exception:
            pass
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
            if self._verify_stop_requested:
                raise VerifyStopRequested()
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
            if self._verify_stop_requested:
                raise VerifyStopRequested()
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

    @staticmethod
    def _normalize_sharpness_scene_key(name: str) -> str:
        stem = Path(str(name or "")).stem
        for suffix in (
            "_replace_mask",
            "_inpainted_right_eye",
            "_inpainted_sbs",
            "_splatted1",
            "_splatted2",
            "_splatted4",
        ):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        return stem

    @classmethod
    def _expected_inpaint_output_name_from_source_name(cls, source_name: str) -> str:
        base = Path(str(source_name or "")).name
        stem = Path(base).stem
        is_quad = stem.endswith("_splatted4")
        core = cls._normalize_sharpness_scene_key(base)
        suffix = "_inpainted_sbs" if is_quad else "_inpainted_right_eye"
        return f"{core}{suffix}.mp4"

    @classmethod
    def _load_sharpness_csv_level_map(
        cls,
        csv_path: str,
    ) -> dict[str, tuple[float, int, str]]:
        if not csv_path or not os.path.isfile(csv_path):
            return {}
        out: dict[str, tuple[float, int, str]] = {}
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = str((row or {}).get("file", "")).strip()
                    if not name:
                        continue
                    try:
                        raw = float((row or {}).get("sharpness_raw", "0") or 0.0)
                    except Exception:
                        raw = 0.0
                    level = max(5, min(11, int(math.trunc(raw / 1100.0)) + 4))
                    out[Path(name).name] = (raw, level, name)
                    out[cls._normalize_sharpness_scene_key(name)] = (raw, level, name)
        except Exception:
            return {}
        return out

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

    def _collect_expected_sharpen_outputs(
        self,
        inpaint_input_dir: str,
        sharpness_csv_path: str,
        *,
        min_level: int = 8,
    ) -> tuple[list[str], str]:
        ok, msg, _missing = self._verify_sharpness_csv_coverage(
            inpaint_input_dir,
            sharpness_csv_path,
        )
        if not ok:
            return [], msg
        if not inpaint_input_dir or not os.path.isdir(inpaint_input_dir):
            return [], f"Inpaint input folder not found: {inpaint_input_dir or '(empty)'}"

        level_map = self._load_sharpness_csv_level_map(sharpness_csv_path)
        if not level_map:
            return [], f"sharpness.csv unreadable or empty: {sharpness_csv_path or '(empty)'}"

        expected: list[str] = []
        seen: set[str] = set()
        for p in self._collect_video_files_for_patterns(
            inpaint_input_dir,
            self.VERIFY_VIDEO_PATTERNS,
        ):
            key = self._normalize_sharpness_scene_key(p.name)
            info = level_map.get(p.name) or level_map.get(key)
            if not info:
                continue
            _raw, level, source_name = info
            if int(level) < int(min_level):
                continue
            out_name = self._expected_inpaint_output_name_from_source_name(source_name)
            if out_name not in seen:
                seen.add(out_name)
                expected.append(out_name)
        expected.sort()
        return expected, ""

    @staticmethod
    def _count_named_existing_files(folder: str, expected_names: Sequence[str]) -> int:
        if not folder or not os.path.isdir(folder):
            return 0
        root = Path(folder)
        count = 0
        for name in expected_names:
            if (root / str(name)).is_file():
                count += 1
        return count

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
        preferred_inpainted_dir: str = "",
    ) -> tuple[bool, str, list[str]]:
        if not inpainted_dir or not os.path.isdir(inpainted_dir):
            raise RuntimeError(f"Inpainted folder not found: {inpainted_dir or '(empty)'}")
        if not splatted_dir or not os.path.isdir(splatted_dir):
            raise RuntimeError(f"Splatted folder not found: {splatted_dir or '(empty)'}")
        if not replace_mask_dir or not os.path.isdir(replace_mask_dir):
            raise RuntimeError(f"Replace-mask folder not found: {replace_mask_dir or '(empty)'}")
        if not csv_path or not os.path.isfile(csv_path):
            return False, f"autoct.csv not found: {csv_path or '(empty)'}", []

        expected_files = self._collect_inpainted_scene_files(
            inpainted_dir,
            preferred_inpainted_dir=preferred_inpainted_dir,
        )
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
        sharpen_enabled = self._sharpen_step_enabled_in_current_mode()
        if hasattr(self, "inpaint_sharpen_run_btn"):
            self.inpaint_sharpen_run_btn.configure(
                state=tk.NORMAL
                if (allow_csv_features and sharpen_enabled and not inpaint_running)
                else tk.DISABLED
            )
        if hasattr(self, "inpaint_sharpen_verify_quick_btn"):
            verify_state = tk.NORMAL if (allow_csv_features and sharpen_enabled and not inpaint_running and not self._verify_running) else tk.DISABLED
            self.inpaint_sharpen_verify_quick_btn.configure(state=verify_state)
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

    @staticmethod
    def _quick_verify_normalize_stem(stem: str) -> str:
        s = str(stem or "").strip().lower()
        if not s:
            return s
        patterns = [
            r"_(\d+_)?splatted[124](?:_replace_mask)?$",
            r"_(\d+_)?merged_full_sbs$",
            r"_(\d+_)?inpainted_(right_eye|sbs)$",
            r"_replace_mask$",
            r"_depth$",
        ]
        changed = True
        while changed and s:
            changed = False
            for pat in patterns:
                ns = re.sub(pat, "", s, flags=re.IGNORECASE)
                if ns != s:
                    s = ns
                    changed = True
        return s

    @staticmethod
    def _quick_verify_build_name_indexes(
        files: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        exact_idx: dict[str, list[str]] = {}
        norm_idx: dict[str, list[str]] = {}
        for fp in files:
            p = Path(str(fp))
            exact_key = p.stem.lower()
            norm_key = PipelineMasterGUI._quick_verify_normalize_stem(p.stem)
            exact_idx.setdefault(exact_key, []).append(str(fp))
            norm_idx.setdefault(norm_key, []).append(str(fp))
        return exact_idx, norm_idx

    @staticmethod
    def _quick_verify_pick_single_candidate(
        candidates: list[str], target_suffix: str
    ) -> tuple[str | None, str]:
        if not candidates:
            return None, "no candidates"
        if len(candidates) == 1:
            return str(candidates[0]), ""
        same_ext = [
            str(p)
            for p in candidates
            if Path(str(p)).suffix.lower() == str(target_suffix or "").lower()
        ]
        if len(same_ext) == 1:
            return same_ext[0], ""
        return None, "ambiguous"

    def _quick_verify_match_reference_path(
        self,
        target_path: str,
        exact_idx: dict[str, list[str]],
        norm_idx: dict[str, list[str]],
    ) -> tuple[str | None, str]:
        target = Path(str(target_path))
        exact_key = target.stem.lower()
        ref_path, err = self._quick_verify_pick_single_candidate(
            exact_idx.get(exact_key, []), target.suffix
        )
        if ref_path:
            return ref_path, "exact"
        if err == "ambiguous":
            return None, "ambiguous exact match"

        norm_key = self._quick_verify_normalize_stem(target.stem)
        ref_path, err = self._quick_verify_pick_single_candidate(
            norm_idx.get(norm_key, []), target.suffix
        )
        if ref_path:
            return ref_path, f"normalized:{norm_key}"
        if err == "ambiguous":
            return None, "ambiguous normalized match"
        return None, "reference not found"

    def _quick_verify_collect_packet_mismatch_targets(
        self,
        target_files: list[str],
        ref_files: list[str],
        target_meta_by_path: dict[str, dict],
        ref_meta_by_path: dict[str, dict],
        frame_tol: int = 1,
    ) -> dict:
        target_norm = sorted({str(x) for x in (target_files or []) if str(x).strip()})
        ref_norm = sorted({str(x) for x in (ref_files or []) if str(x).strip()})
        exact_idx, norm_idx = self._quick_verify_build_name_indexes(ref_norm)

        mismatch_targets: list[str] = []
        unmatched_targets: list[str] = []
        matched_refs: set[str] = set()
        pairs_compared = 0
        pairs_packet_nd = 0
        tol = max(0, int(frame_tol))

        for target_path in target_norm:
            ref_path, _match_info = self._quick_verify_match_reference_path(
                target_path, exact_idx, norm_idx
            )
            if not ref_path:
                unmatched_targets.append(target_path)
                continue
            matched_refs.add(ref_path)

            t_meta = target_meta_by_path.get(target_path) or {}
            r_meta = ref_meta_by_path.get(ref_path) or {}
            if not bool(t_meta.get("ok")) or not bool(r_meta.get("ok")):
                continue

            t_frames = t_meta.get("frames")
            r_frames = r_meta.get("frames")
            if t_frames is None or r_frames is None:
                pairs_packet_nd += 1
                continue
            try:
                delta = abs(int(t_frames) - int(r_frames))
            except Exception:
                pairs_packet_nd += 1
                continue
            pairs_compared += 1
            if delta > tol:
                mismatch_targets.append(target_path)

        missing_reference = [ref for ref in ref_norm if ref not in matched_refs]
        return {
            "mismatch_targets": sorted(set(mismatch_targets)),
            "unmatched_targets": sorted(set(unmatched_targets)),
            "missing_reference": missing_reference,
            "pairs_compared": pairs_compared,
            "pairs_packet_nd": pairs_packet_nd,
        }

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

        if stage_key in {"inpaint", "sharpen", "merge", "merge_mask"}:
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

                target_stats = self._quick_verify_probe_group(
                    target_files,
                    max_workers,
                    "splat_line",
                    target_label,
                    "[QUICK]",
                )
                ref_stats = self._quick_verify_probe_group(
                    ref_files,
                    max_workers,
                    "splat_line",
                    "reference",
                    "[QUICK]",
                )

                pair_stats = self._quick_verify_collect_packet_mismatch_targets(
                    target_files,
                    ref_files,
                    target_stats.get("meta_by_path", {}),
                    ref_stats.get("meta_by_path", {}),
                    frame_tol=1,
                )
                packet_mismatch_targets = pair_stats.get("mismatch_targets") or []
                unmatched_targets = pair_stats.get("unmatched_targets") or []
                missing_reference = pair_stats.get("missing_reference") or []
                broken_targets = sorted(
                    set((target_stats.get("broken") or []) + packet_mismatch_targets)
                )

                self._log_queue.put(
                    (
                        "splat_line",
                        (
                            f"[QUICK] {target_label} packet pair check: "
                            f"compared={int(pair_stats.get('pairs_compared', 0))}, "
                            f"n.d.={int(pair_stats.get('pairs_packet_nd', 0))}, "
                            f"mismatch={len(packet_mismatch_targets)}, "
                            f"unmatched_target={len(unmatched_targets)}, "
                            f"missing_reference={len(missing_reference)}"
                        ),
                    )
                )

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
                    not broken_targets
                    and not ref_stats["broken"]
                    and count_ok
                    and not unmatched_targets
                    and not missing_reference
                    and (frames_ok or frames_msg == "n.d.")
                )
                msg = (
                    f"[{target_label}] Broken target files: {len(target_stats['broken'])}; "
                    f"Packet mismatch target files: {len(packet_mismatch_targets)}; "
                    f"Unmatched target files: {len(unmatched_targets)}; "
                    f"Missing reference files: {len(missing_reference)}; "
                    f"Broken reference files: {len(ref_stats['broken'])}; "
                    f"File count: {'YES' if count_ok else 'NO'} ({count_msg}); "
                    f"Duration (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')} ({duration_msg}); "
                    f"Frames: {'YES' if frames_ok else ('N.D.' if frames_msg == 'n.d.' else 'NO')} ({frames_msg})"
                )
                return ok_final, msg, broken_targets

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

        script_path = utilities_path("verifyscenes.py")
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
                if self._verify_stop_requested:
                    raise VerifyStopRequested()
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
                    proc = self._verify_popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )
                    assert proc.stdout is not None
                    try:
                        for raw_line in proc.stdout:
                            if self._verify_stop_requested:
                                raise VerifyStopRequested()
                            line = raw_line.rstrip("\n")
                            if line:
                                self._log_queue.put(("splat_line", f"[DEEP][{label}] {line}"))
                                bad_path = self._resolve_verifyscenes_bad_path(line, target_dir)
                                if bad_path and bad_path not in seen_bad:
                                    seen_bad.add(bad_path)
                                    bad_files.append(bad_path)
                        rc = int(proc.wait() or 0)
                    finally:
                        self._unregister_verify_process(proc)
                except VerifyStopRequested:
                    raise
                except Exception as e:
                    self._log_queue.put(("splat_line", f"[DEEP][{label}][ERROR] {e}"))
                    rc = 1

                if rc != 0:
                    overall_rc = rc if overall_rc == 0 else overall_rc
                    failed_dirs.append(target_dir)
                    self._log_queue.put(("splat_line", f"[DEEP][{label}] failed with rc={rc}"))
                else:
                    self._log_queue.put(("splat_line", f"[DEEP][{label}] completed successfully"))
        except VerifyStopRequested:
            overall_rc = overall_rc or 1
        finally:
            self._log_queue.put(
                (
                    "splat_verify_deep_result",
                    {
                        "rc": overall_rc,
                        "stopped": bool(self._verify_stop_requested),
                        "failed_dirs": failed_dirs,
                        "bad_files": bad_files,
                    },
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
            marker_path = self._touch_stop_marker_file(
                str(self._splat_stop_marker_path or "").strip()
                or os.path.join(
                    self.splat_output_var.get().strip() or "./work/splat",
                    ".stop_after_current",
                ),
                self._append_splat_log,
            )
            if marker_path:
                self._splat_stop_marker_path = marker_path
        else:
            self.splat_status_var.set("Force stop requested...")
            self._append_splat_log("[STOP] force stop requested.")
            self._send_splat_signal(signal.SIGINT)
            self.root.after(1000, self._force_kill_splat)
        self._refresh_pipeline_run_button()

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
        if is_running:
            self.splat_stop_btn.configure(text="Stop")
        else:
            self.splat_stop_btn.configure(text="Stop")
            self._splat_stop_clicks = 0
            self._splat_stop_requested = False
            self._splat_stop_marker_path = ""
        self._update_replace_mask_dependent_controls()
        self._refresh_pipeline_run_button()

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
            depth_stats = self._quick_verify_probe_group(
                depth_files,
                max_workers,
                "depth_line",
                "depth",
                "[QUICK]",
            )
            ref_stats = self._quick_verify_probe_group(
                ref_files,
                max_workers,
                "depth_line",
                "reference",
                "[QUICK]",
            )

            pair_stats = self._quick_verify_collect_packet_mismatch_targets(
                depth_files,
                ref_files,
                depth_stats.get("meta_by_path", {}),
                ref_stats.get("meta_by_path", {}),
                frame_tol=1,
            )
            packet_mismatch_depth = pair_stats.get("mismatch_targets") or []
            unmatched_depth = pair_stats.get("unmatched_targets") or []
            missing_reference = pair_stats.get("missing_reference") or []
            broken_depth = sorted(set((depth_stats.get("broken") or []) + packet_mismatch_depth))

            self._log_queue.put(
                (
                    "depth_line",
                    (
                        "[QUICK] packet pair check: "
                        f"compared={int(pair_stats.get('pairs_compared', 0))}, "
                        f"n.d.={int(pair_stats.get('pairs_packet_nd', 0))}, "
                        f"mismatch={len(packet_mismatch_depth)}, "
                        f"unmatched_depth={len(unmatched_depth)}, "
                        f"missing_reference={len(missing_reference)}"
                    ),
                )
            )

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
                not broken_depth
                and not ref_stats["broken"]
                and count_ok
                and not unmatched_depth
                and not missing_reference
                and (frames_ok or frames_msg == "n.d.")
            )
            message = (
                f"Depth quick verify completed.\n"
                f"Broken depth files: {len(depth_stats['broken'])}\n"
                f"Packet mismatch depth files: {len(packet_mismatch_depth)}\n"
                f"Unmatched depth files: {len(unmatched_depth)}\n"
                f"Missing reference files: {len(missing_reference)}\n"
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
                        "broken_depth": broken_depth,
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

        script_path = utilities_path("verifyscenes.py")
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
            if self._verify_stop_requested:
                raise VerifyStopRequested()
            proc = self._verify_popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            try:
                for raw in proc.stdout:
                    if self._verify_stop_requested:
                        raise VerifyStopRequested()
                    line = raw.rstrip("\n")
                    if line:
                        self._log_queue.put(("depth_line", f"[DEEP] {line}"))
                        bad_path = self._resolve_verifyscenes_bad_path(line, depth_dir)
                        if bad_path and bad_path not in seen_bad:
                            seen_bad.add(bad_path)
                            bad_files.append(bad_path)
                rc = proc.wait()
            finally:
                self._unregister_verify_process(proc)
        except VerifyStopRequested:
            rc = 1
        except Exception as e:
            self._log_queue.put(("depth_line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            rc = 1
        finally:
            self._log_queue.put(
                (
                    "depth_verify_deep_result",
                    {
                        "rc": rc,
                        "stopped": bool(self._verify_stop_requested),
                        "depth_dir": depth_dir,
                        "bad_files": bad_files,
                    },
                )
            )
            self._log_queue.put(("verify_done", "depth_deep"))

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
        self.depth_input_var.set(os.path.normpath(depth_in))
        self.depth_output_var.set(os.path.normpath(depth_out))
        self.splat_input_clips_var.set(os.path.normpath(scene_out))
        self._sync_depth_to_splat_input_path()
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
        self.inpaint_sharpen_output_var.set(
            os.path.normpath(
                os.path.join(work_dir, self.STANDARD_SUBDIRS["inpaint_sharpen"])
            )
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

    def _scene_crop_target_matches_auto_default(self, raw_value: str | None = None) -> bool:
        profile = self._crop_recommendation_profile or {}
        default_target = profile.get("default_target_eff_h")
        if not default_target:
            return False
        try:
            default_target_int = int(default_target)
        except Exception:
            return False
        current_raw = self.scene_crop_target_h_var.get().strip() if raw_value is None else str(raw_value).strip()
        if current_raw == "":
            return False
        normalized = self._normalize_scene_crop_target_effective(
            current_raw,
            profile,
            fallback=default_target_int,
        )
        if normalized is None:
            return False
        return int(normalized) == default_target_int

    def _on_scene_crop_target_spin(self) -> None:
        if (
            self.scene_crop_mode_var.get().strip().lower() == "auto"
            and not self._scene_crop_target_matches_auto_default()
        ):
            self.scene_crop_mode_var.set("manual")
        self._sync_auto_crop_from_target()
        self._refresh_crop_controls_state()

    def _on_scene_crop_target_changed(self, *_args) -> None:
        if self._scene_crop_target_syncing:
            return
        if (
            self.scene_crop_mode_var.get().strip().lower() == "auto"
            and not self._scene_crop_target_matches_auto_default()
        ):
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
        self._preview_scene_command()

    def _on_tonemap_changed(self, _event=None) -> None:
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

    def _apply_option_states(self) -> None:
        caps = self._source_capabilities or {}
        has_analysis = bool(caps)
        is_hdr = bool(caps.get("is_hdr", False))

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
            hint.append("Analyze Source Video to apply source-driven crop recommendations.")
        elif is_hdr:
            hint.append("HDR input: 8-bit BT.709 conversion will be applied automatically.")
        else:
            hint.append("SDR input: no forced tonemap chain.")
        hint.append("Intermediate color steps now use yuv444p via the global encoder policy.")
        self.scene_option_hint_var.set(" | ".join(hint))

    def _update_source_capabilities(self, info: dict) -> None:
        dynamic_range = str(info.get("dynamic_range") or "").upper()
        is_hdr = "HDR" in dynamic_range
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
            if mode != "auto" and self._scene_crop_target_matches_auto_default(current_raw):
                self.scene_crop_mode_var.set("auto")
                mode = "auto"
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
            filters.append("format=yuv444p")
        return filters

    def _build_scene_split_ffmpeg_tokens(self) -> list[str]:
        scene_codec = self._normalize_ffmpeg_codec(
            self.scene_codec_var.get(),
            self.DEFAULT_SCENE_CODEC,
        )
        self.scene_codec_var.set(scene_codec)
        profile = self._resolve_color_profile_for_codec(scene_codec)
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

        ffmpeg_tokens += list(profile.generated_args)

        extra_ffmpeg = self._current_global_ffmpeg_extra_args()
        if extra_ffmpeg:
            ffmpeg_tokens = self._append_extra_args_to_tokens(ffmpeg_tokens, extra_ffmpeg)
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
        script_path = utilities_path("split_scenes_from_csv.py")
        stop_marker = os.path.join(output_path or "./work/seg", ".stop_after_current")
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
            "--stop-marker",
            stop_marker,
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
        if not is_running and not self._analysis_running:
            self.scene_stop_btn.configure(text="Stop")
            self._scene_stop_clicks = 0
            self._scene_stop_marker_path = ""
        if is_running:
            self.scene_verify_quick_btn.configure(state=tk.DISABLED)
        elif not self._verify_running and not self._analysis_running:
            self.scene_verify_quick_btn.configure(state=tk.NORMAL)
        self._refresh_pipeline_run_button()

    def _set_analysis_running(self, is_running: bool) -> None:
        self._analysis_running = is_running
        self.scene_analyze_btn.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        scene_is_running = bool(self._scene_thread and self._scene_thread.is_alive())
        self.scene_stop_btn.configure(state=tk.NORMAL if (is_running or scene_is_running) else tk.DISABLED)
        if is_running:
            self.scene_verify_quick_btn.configure(state=tk.DISABLED)
        elif (not scene_is_running) and (not self._verify_running):
            self.scene_verify_quick_btn.configure(state=tk.NORMAL)
        self._refresh_pipeline_run_button()

    def _verify_button_specs(self) -> list[tuple[str, str, object]]:
        return [
            ("scene_verify_quick_btn", "Verify Scenes", self._start_verify_quick),
            ("depth_verify_quick_btn", "Verify Depth", self._start_depth_verify_quick),
            ("splat_verify_quick_btn", "Verify Scenes", self._start_splat_verify_quick),
            ("inpaint_verify_quick_btn", "Verify Scenes", self._start_inpaint_verify_quick),
            ("inpaint_sharpen_verify_quick_btn", "Verify Sharpen", self._start_inpaint_sharpen_verify_quick),
            (
                "merge_mask_verify_quick_btn",
                "Verify Mask",
                self._start_merge_mask_verify_quick,
            ),
            ("merge_verify_quick_btn", "Verify Merge", self._start_merge_verify_quick),
            ("join_mono_verify_btn", "Verify Mono->SBS", self._start_join_mono_verify),
            ("join_verify_btn", "Verify Join", self._start_join_verify),
        ]

    def _active_verify_button_attr(self) -> str:
        mode = str(self._verify_mode or "").strip().lower()
        return {
            "quick": "scene_verify_quick_btn",
            "depth_quick": "depth_verify_quick_btn",
            "splat_quick": "splat_verify_quick_btn",
            "inpaint_quick": "inpaint_verify_quick_btn",
            "sharpen_quick": "inpaint_sharpen_verify_quick_btn",
            "merge_mask_quick": "merge_mask_verify_quick_btn",
            "merge_quick": "merge_verify_quick_btn",
            "join_mono_quick": "join_mono_verify_btn",
            "join_quick": "join_verify_btn",
        }.get(mode, "")

    def _refresh_verify_buttons(self) -> None:
        for attr_name, text, command in self._verify_button_specs():
            btn = getattr(self, attr_name, None)
            if btn is None:
                continue
            btn.configure(text=text, command=command)
        if not self._verify_running:
            return
        active_attr = self._active_verify_button_attr()
        if not active_attr:
            return
        active_btn = getattr(self, active_attr, None)
        if active_btn is None:
            return
        active_btn.configure(
            text="Force Stop" if self._verify_stop_clicks > 0 else "Stop",
            command=self._stop_active_verify,
            state=tk.NORMAL,
        )

    def _current_pipeline_stop_button_text(self) -> str:
        if self._verify_running:
            return "Force Stop" if self._verify_stop_clicks > 0 else "Stop"
        if (self._merge_thread and self._merge_thread.is_alive()) or self._merge_group_alive():
            return "Force Stop" if self._merge_stop_clicks > 0 else "Stop"
        if self._inpaint_thread and self._inpaint_thread.is_alive():
            return "Force Stop" if self._inpaint_stop_clicks > 0 else "Stop"
        if self._depth_thread and self._depth_thread.is_alive():
            return "Force Stop" if self._depth_stop_clicks > 0 else "Stop"
        if self._splat_thread and self._splat_thread.is_alive():
            return "Force Stop" if self._splat_stop_clicks > 0 else "Stop"
        if self._join_thread and self._join_thread.is_alive():
            return "Force Stop" if self._join_stop_requested else "Stop"
        if (self._scene_thread and self._scene_thread.is_alive()) or self._analysis_running:
            return "Stop"
        return "Start/Resume"

    def _refresh_pipeline_run_button(self) -> None:
        btn = getattr(self, "pipeline_start_resume_btn", None)
        if btn is None:
            return
        if self._any_pipeline_activity():
            btn.configure(
                text=self._current_pipeline_stop_button_text(),
                command=self._pipeline_stop_active,
            )
        else:
            btn.configure(text="Start/Resume", command=self._pipeline_start_resume)

    def _pipeline_stop_active(self) -> None:
        if not self._any_pipeline_activity():
            self._refresh_pipeline_run_button()
            return
        pipeline_owned = bool(
            self._pipeline_autorun or self._pipeline_pending_action is not None or self._pipeline_test_active
        )
        if pipeline_owned:
            self._pipeline_autorun = False
            self._pipeline_stop_requested = True
        if self._verify_running:
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: active verification...")
            self._stop_active_verify(prompt_user=False)
            self._refresh_pipeline_run_button()
            return
        if (self._scene_thread and self._scene_thread.is_alive()) or self._analysis_running:
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: scene step...")
            self._stop_scene_detect(prompt_user=False)
        elif self._depth_thread and self._depth_thread.is_alive():
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: depth step...")
            self._stop_depth_placeholder(prompt_user=False)
        elif self._splat_thread and self._splat_thread.is_alive():
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: splatting step...")
            self._stop_splat_placeholder(prompt_user=False)
        elif self._inpaint_thread and self._inpaint_thread.is_alive():
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: inpaint step...")
            self._stop_inpaint_placeholder(prompt_user=False)
        elif (self._merge_thread and self._merge_thread.is_alive()) or self._merge_group_alive():
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: merge step...")
            self._stop_merge_placeholder(prompt_user=False)
        elif self._join_thread and self._join_thread.is_alive():
            if pipeline_owned:
                self.pipeline_run_status_var.set("Pipeline stopping: join step...")
            self._stop_join(prompt_user=False)
        self._refresh_pipeline_run_button()

    def _finalize_pipeline_stop(self, label: str) -> None:
        if not self._pipeline_stop_requested:
            return
        self._pipeline_autorun = False
        self._pipeline_pending_action = None
        if self._pipeline_test_active:
            self._restore_test_scene_subset()
        self.pipeline_run_status_var.set(f"Pipeline stopped during {label}.")
        self._pipeline_stop_requested = False
        self._pipeline_sync_noninteractive_mode()

    def _verify_mode_ui_bundle(
        self,
        mode: str | None = None,
    ) -> tuple[tk.StringVar | None, object | None, str]:
        cur = str(mode or self._verify_mode or "").strip().lower()
        if cur == "quick":
            return self.scene_status_var, self._append_scene_log, "Verify Scenes"
        if cur.startswith("depth_"):
            return self.depth_status_var, self._append_depth_log, "Verify Depth"
        if cur.startswith("splat_"):
            return self.splat_status_var, self._append_splat_log, "Verify Splatting"
        if cur.startswith("sharpen_"):
            return self.inpaint_status_var, self._append_inpaint_log, "Verify Sharpen"
        if cur.startswith("inpaint_"):
            return self.inpaint_status_var, self._append_inpaint_log, "Verify Inpainting"
        if cur.startswith("merge_mask_"):
            return self.merge_status_var, self._append_merge_log, "Verify Mask"
        if cur.startswith("merge_"):
            return self.merge_status_var, self._append_merge_log, "Verify Merging"
        if cur.startswith("join_mono_"):
            return self.join_status_var, self._append_join_log, "Verify Mono->SBS"
        if cur.startswith("join_"):
            return self.join_status_var, self._append_join_log, "Verify Join"
        return None, None, "Verify"

    @staticmethod
    def _is_verify_result_kind(kind: str) -> bool:
        return str(kind or "").strip() in {
            "verify_quick_result",
            "verify_deep_result",
            "depth_verify_quick_result",
            "depth_verify_deep_result",
            "splat_verify_quick_result",
            "splat_verify_deep_result",
            "inpaint_verify_quick_result",
            "inpaint_verify_deep_result",
            "sharpen_verify_quick_result",
            "sharpen_verify_deep_result",
            "merge_mask_verify_quick_result",
            "merge_mask_verify_deep_result",
            "merge_verify_quick_result",
            "merge_verify_deep_result",
            "join_verify_result",
            "join_mono_verify_result",
        }

    def _handle_stopped_verify_result(self, kind: str, payload: dict) -> None:
        kind_txt = str(kind or "").strip()
        status_map: dict[str, tuple[tk.StringVar, str]] = {
            "verify_quick_result": (self.scene_status_var, "Verify Scenes (Quick) stopped"),
            "verify_deep_result": (self.scene_status_var, "Verify Scenes (Deep) stopped"),
            "depth_verify_quick_result": (self.depth_status_var, "Verify Depth (Quick) stopped"),
            "depth_verify_deep_result": (self.depth_status_var, "Verify Depth (Deep) stopped"),
            "splat_verify_quick_result": (
                self.splat_status_var,
                "Verify Splatting (Quick) stopped",
            ),
            "splat_verify_deep_result": (
                self.splat_status_var,
                "Verify Splatting (Deep) stopped",
            ),
            "inpaint_verify_quick_result": (
                self.inpaint_status_var,
                "Verify Inpainting (Quick) stopped",
            ),
            "inpaint_verify_deep_result": (
                self.inpaint_status_var,
                "Verify Inpainting (Deep) stopped",
            ),
            "sharpen_verify_quick_result": (
                self.inpaint_status_var,
                "Verify Sharpen (Quick) stopped",
            ),
            "sharpen_verify_deep_result": (
                self.inpaint_status_var,
                "Verify Sharpen (Deep) stopped",
            ),
            "merge_mask_verify_quick_result": (
                self.merge_status_var,
                "Verify Mask (Quick) stopped",
            ),
            "merge_mask_verify_deep_result": (
                self.merge_status_var,
                "Verify Mask (Deep) stopped",
            ),
            "merge_verify_quick_result": (
                self.merge_status_var,
                "Verify Merging (Quick) stopped",
            ),
            "merge_verify_deep_result": (
                self.merge_status_var,
                "Verify Merging (Deep) stopped",
            ),
            "join_verify_result": (self.join_status_var, "Verify Join stopped"),
            "join_mono_verify_result": (
                self.join_status_var,
                (
                    "Mono->SBS Verify (Deep) stopped"
                    if str(payload.get("mode", "")).strip().lower() == "deep"
                    else "Mono->SBS Verify (Quick) stopped"
                ),
            ),
        }
        status_entry = status_map.get(kind_txt)
        if status_entry:
            status_var, text = status_entry
            status_var.set(text)
        if kind_txt in {"verify_quick_result", "verify_deep_result"}:
            self._scene_verify_result_applied = True
        if self._pipeline_stop_requested:
            self.pipeline_run_status_var.set("Pipeline stopped during verification.")

    def _stop_active_verify(self, prompt_user: bool = True) -> None:
        if not self._verify_running:
            return
        status_var, logger, title = self._verify_mode_ui_bundle()
        pipeline_owned = bool(
            self._pipeline_autorun or self._pipeline_pending_action is not None or self._pipeline_test_active
        )
        if pipeline_owned:
            self._pipeline_autorun = False
            self._pipeline_stop_requested = True
            self.pipeline_run_status_var.set("Pipeline stopping: active verification...")
        if self._verify_stop_clicks == 0 and prompt_user:
            messagebox.showwarning(
                title,
                (
                    "Stop requested.\n\n"
                    "Verification workers and ffprobe subprocesses will be interrupted.\n"
                    "Click Stop again to force-kill any remaining verify process immediately."
                ),
            )
        self._verify_stop_requested = True
        self._verify_stop_clicks += 1
        if self._verify_mode == "quick":
            self._scene_verify_result_applied = True
        if status_var is not None:
            status_var.set("Force stop requested..." if self._verify_stop_clicks > 1 else "Stop requested...")
        if callable(logger):
            if self._verify_stop_clicks > 1:
                logger("[STOP] verification force stop requested.")
            else:
                logger("[STOP] verification stop requested (click Stop again for immediate force stop).")
        self._signal_verify_processes(signal.SIGINT)
        if self._verify_stop_clicks >= 2:
            self.root.after(50, self._force_kill_verify)
        else:
            self.root.after(1000, self._force_kill_verify)
        self._refresh_verify_buttons()
        self._refresh_pipeline_run_button()

    def _set_verify_running(self, is_running: bool, mode: str = "") -> None:
        self._verify_running = is_running
        self._verify_mode = mode if is_running else ""
        if is_running:
            self._verify_stop_requested = False
            self._verify_stop_clicks = 0
        if is_running:
            self.scene_verify_quick_btn.configure(state=tk.DISABLED)
            self.splat_verify_quick_btn.configure(state=tk.DISABLED)
            self.inpaint_verify_quick_btn.configure(state=tk.DISABLED)
            self.merge_mask_verify_quick_btn.configure(state=tk.DISABLED)
            self.merge_verify_quick_btn.configure(state=tk.DISABLED)
            self.join_mono_verify_btn.configure(state=tk.DISABLED)
            self.join_verify_btn.configure(state=tk.DISABLED)
        else:
            scene_is_running = bool(self._scene_thread and self._scene_thread.is_alive())
            if scene_is_running:
                self.scene_verify_quick_btn.configure(state=tk.DISABLED)
            else:
                if self._analysis_running:
                    self.scene_verify_quick_btn.configure(state=tk.DISABLED)
                else:
                    self.scene_verify_quick_btn.configure(state=tk.NORMAL)
            depth_is_running = bool(self._depth_thread and self._depth_thread.is_alive())
            splat_is_running = bool(self._splat_thread and self._splat_thread.is_alive())
            if splat_is_running:
                self.splat_verify_quick_btn.configure(state=tk.DISABLED)
            else:
                self.splat_verify_quick_btn.configure(state=tk.NORMAL)
            inpaint_is_running = bool(self._inpaint_thread and self._inpaint_thread.is_alive())
            if inpaint_is_running:
                self.inpaint_verify_quick_btn.configure(state=tk.DISABLED)
            else:
                self.inpaint_verify_quick_btn.configure(state=tk.NORMAL)
            merge_is_running = bool(self._merge_thread and self._merge_thread.is_alive())
            if merge_is_running:
                self.merge_mask_verify_quick_btn.configure(state=tk.DISABLED)
                self.merge_verify_quick_btn.configure(state=tk.DISABLED)
            else:
                self.merge_mask_verify_quick_btn.configure(state=tk.NORMAL)
                self.merge_verify_quick_btn.configure(state=tk.NORMAL)
            join_is_running = bool(self._join_thread and self._join_thread.is_alive())
            if join_is_running:
                self.join_mono_verify_btn.configure(state=tk.DISABLED)
                self.join_verify_btn.configure(state=tk.DISABLED)
            else:
                self.join_mono_verify_btn.configure(state=tk.NORMAL)
                self.join_verify_btn.configure(state=tk.NORMAL)
            self._verify_stop_requested = False
            self._verify_stop_clicks = 0
        self._refresh_depth_action_buttons()
        self._refresh_verify_buttons()
        self._refresh_pipeline_run_button()

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
    def _verify_popen_kwargs() -> dict[str, object]:
        if os.name == "nt":
            flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) or 0)
            return {"creationflags": flags} if flags else {}
        return {"start_new_session": True}

    def _register_verify_process(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        with self._verify_processes_lock:
            self._verify_processes.add(proc)

    def _unregister_verify_process(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        with self._verify_processes_lock:
            self._verify_processes.discard(proc)

    def _signal_verify_process(self, proc: subprocess.Popen | None, sig: int) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name != "nt" and hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), sig)
            elif sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except Exception:
            try:
                if sig == signal.SIGKILL:
                    proc.kill()
                else:
                    proc.terminate()
            except Exception:
                pass

    def _signal_verify_processes(self, sig: int) -> None:
        with self._verify_processes_lock:
            processes = list(self._verify_processes)
        for proc in processes:
            self._signal_verify_process(proc, sig)

    def _verify_popen(self, cmd: list[str], **kwargs) -> subprocess.Popen:
        popen_kwargs = dict(kwargs)
        for key, value in self._verify_popen_kwargs().items():
            popen_kwargs.setdefault(key, value)
        proc = subprocess.Popen(cmd, **popen_kwargs)
        self._register_verify_process(proc)
        if self._verify_stop_requested:
            self._signal_verify_process(proc, signal.SIGINT)
        return proc

    def _force_kill_verify(self) -> None:
        if not self._verify_running:
            return
        self._signal_verify_processes(signal.SIGKILL)

    def _run_ffprobe_watchdog(self, cmd: list[str], timeout_sec: float) -> tuple[int, str, str, bool]:
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **self._verify_popen_kwargs(),
            )
            self._register_verify_process(proc)
            if self._verify_stop_requested:
                self._signal_verify_process(proc, signal.SIGINT)
            try:
                out_txt, err_txt = proc.communicate(
                    timeout=(float(timeout_sec) if timeout_sec and float(timeout_sec) > 0 else None)
                )
                rc = int(proc.returncode or 0)
                return rc, (out_txt or "").strip(), (err_txt or "").strip(), False
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    try:
                        self._signal_verify_process(proc, signal.SIGKILL)
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
                    self._signal_verify_process(proc, signal.SIGKILL)
                except Exception:
                    pass
            return 126, "", str(e), False
        finally:
            self._unregister_verify_process(proc)

    def _probe_video_basic(self, path: str, count_mode: str = "packets") -> dict:
        use_frames = str(count_mode or "").strip().lower() == "frames"
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames" if use_frames else "-count_packets",
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
            if self._verify_stop_requested:
                return {
                    "ok": False,
                    "error": "verify stop requested",
                    "duration": None,
                    "frames": None,
                    "width": None,
                    "height": None,
                    "codec_name": "",
                    "pix_fmt": "",
                }
            rc, out_txt, err_txt, timed_out = self._run_ffprobe_watchdog(cmd, watchdog_sec)
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
            nbf = st.get("nb_read_frames" if use_frames else "nb_read_packets")
            if nbf in (None, "", "N/A"):
                nbf = st.get("nb_read_packets" if use_frames else "nb_read_frames")
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

    def _quick_verify_probe_group(
        self,
        file_list: list[str],
        max_workers: int,
        log_kind: str,
        label: str,
        prefix: str = "[QUICK]",
        count_mode: str = "packets",
    ) -> dict:
        files = sorted({str(x) for x in (file_list or []) if str(x).strip()})
        broken: list[str] = []
        meta_by_path: dict[str, dict] = {}
        total_duration = 0.0
        duration_available = True
        total_frames = 0
        frames_available = True

        def _probe_one(fp: str) -> tuple[str, dict]:
            return fp, self._probe_video_basic(fp, count_mode=count_mode)

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        futures: list[concurrent.futures.Future] = []
        try:
            futures = [ex.submit(_probe_one, fp) for fp in files]
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                if self._verify_stop_requested:
                    raise VerifyStopRequested()
                fp, meta = fut.result()
                meta_by_path[fp] = dict(meta or {})
                done += 1
                if not meta.get("ok"):
                    broken.append(fp)
                    self._log_queue.put(
                        (
                            log_kind,
                            f"{prefix}[{label.upper()}][BROKEN] {fp} :: {meta.get('error')}",
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
                if done % 25 == 0 or done == len(files):
                    self._log_queue.put(
                        (
                            log_kind,
                            f"{prefix}[{label.upper()}] progress {done}/{len(files)}",
                        )
                    )
        except VerifyStopRequested:
            for fut in futures:
                fut.cancel()
            raise
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return {
            "broken": sorted(set(broken)),
            "meta_by_path": meta_by_path,
            "total_duration": total_duration,
            "duration_available": duration_available,
            "total_frames": total_frames,
            "frames_available": frames_available,
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
            verify_patterns = self._scene_quick_verify_patterns()
            for idx, d in enumerate(target_dirs):
                cur = self._collect_files_for_patterns(d, verify_patterns)
                files.extend(cur)
                if idx == 0:
                    seg_count = len(cur)
                else:
                    seg_mono_count += len(cur)
            files = sorted(set(files))

            expected_by_name, csv_ref_err = self._scene_csv_expected_by_name(source_path)
            files_by_name: dict[str, list[str]] = {}
            for fp in files:
                files_by_name.setdefault(Path(fp).name, []).append(fp)
            missing_split_outputs: list[str] = []
            if target_dirs and expected_by_name:
                seg_root = Path(target_dirs[0]).resolve()
                for out_name in expected_by_name:
                    if not files_by_name.get(out_name):
                        missing_split_outputs.append(str((seg_root / out_name).resolve()))
            duplicate_files: list[str] = []
            duplicate_names: list[str] = []
            for out_name, paths in sorted(files_by_name.items()):
                if out_name in expected_by_name and len(paths) > 1:
                    duplicate_names.append(out_name)
                    duplicate_files.extend(sorted({str(p) for p in paths if str(p).strip()}))
                    self._log_queue.put(
                        (
                            "line",
                            (
                                f"[QUICK][DUPLICATE] {out_name} :: "
                                + " | ".join(sorted(paths))
                            ),
                        )
                    )

            if not files:
                self._log_queue.put(("verify_quick_result", {
                    "ok": False,
                    "message": "No scene video files found in seg/seg-mono folders.",
                    "broken": [],
                    "missing_split": missing_split_outputs,
                    "duplicate_split": duplicate_files,
                    "csv_ref_err": csv_ref_err,
                    "duration_ok": False,
                    "frames_ok": False,
                    "retryable_failure": (not csv_ref_err) and (len(duplicate_files) == 0),
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

            probe = self._quick_verify_probe_group(
                files,
                max_workers,
                "line",
                "scene",
                prefix="[QUICK]",
                count_mode="frames",
            )
            broken = [str(p) for p in (probe.get("broken") or []) if str(p).strip()]
            total_duration = float(probe.get("total_duration") or 0.0)
            duration_available = bool(probe.get("duration_available", False))
            meta_by_path = {
                str(k): dict(v or {})
                for k, v in (probe.get("meta_by_path") or {}).items()
                if str(k).strip()
            }

            frame_mismatch_files: list[str] = []
            matched_scene_count = 0
            for fp, meta in sorted(meta_by_path.items()):
                if not meta.get("ok"):
                    continue
                out_name = Path(fp).name
                expected_info = expected_by_name.get(out_name)
                if not expected_info:
                    continue
                expected_frames = int(expected_info.get("frame_count", 0) or 0)
                actual_frames = meta.get("frames")
                if expected_frames <= 0:
                    matched_scene_count += 1
                    continue
                if actual_frames is None:
                    frame_mismatch_files.append(fp)
                    self._log_queue.put(
                        (
                            "line",
                            f"[QUICK][FRAME-MISMATCH] {fp} :: expected={expected_frames}, actual=n.d.",
                        )
                    )
                    continue
                actual_frames_int = int(actual_frames)
                if actual_frames_int != expected_frames:
                    frame_mismatch_files.append(fp)
                    self._log_queue.put(
                        (
                            "line",
                            (
                                f"[QUICK][FRAME-MISMATCH] {fp} :: "
                                f"expected={expected_frames}, actual={actual_frames_int}"
                            ),
                        )
                    )
                    continue
                matched_scene_count += 1

            if self._verify_stop_requested:
                raise VerifyStopRequested()
            src_meta = self._probe_video_basic(source_path)
            src_duration = src_meta.get("duration")
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
            frames_msg = (
                f"expected={len(expected_by_name)}, "
                f"matched={matched_scene_count}, "
                f"mismatched={len(frame_mismatch_files)}, "
                f"missing={len(missing_split_outputs)}, "
                f"duplicates={len(duplicate_names)}"
            )
            frames_ok = (
                (len(broken) == 0)
                and (len(frame_mismatch_files) == 0)
                and (len(missing_split_outputs) == 0)
                and (len(duplicate_names) == 0)
                and (not csv_ref_err)
            )

            self._log_queue.put(("line", f"[QUICK] duration check: {duration_msg}"))
            self._log_queue.put(("line", f"[QUICK] csv frame check: {frames_msg}"))
            if csv_ref_err:
                self._log_queue.put(("line", f"[QUICK] csv reference error: {csv_ref_err}"))
            ok_final = frames_ok
            bad_files = sorted(set(broken + frame_mismatch_files))
            message = (
                f"Quick verify completed.\n"
                f"Broken files: {len(broken)}\n"
                f"Wrong-length files: {len(frame_mismatch_files)}\n"
                f"Duplicate scene files: {len(duplicate_names)}\n"
                f"Duration match (informational only): {'YES' if duration_ok else ('N.D.' if duration_msg == 'n.d.' else 'NO')}\n"
                f"Duration details: {duration_msg}\n"
                f"CSV frame match: {'YES' if frames_ok else 'NO'}\n"
                f"CSV frame details: {frames_msg}\n"
                f"Missing split files: {len(missing_split_outputs)}"
            )
            self._log_queue.put(("verify_quick_result", {
                "ok": ok_final,
                "message": message,
                "broken": bad_files,
                "frame_mismatch": frame_mismatch_files,
                "missing_split": missing_split_outputs,
                "duplicate_split": duplicate_files,
                "csv_ref_err": csv_ref_err,
                "duration_ok": duration_ok,
                "frames_ok": frames_ok,
                "retryable_failure": (not csv_ref_err) and (len(duplicate_files) == 0),
            }))
        except VerifyStopRequested:
            self._log_queue.put(("verify_quick_result", {
                "ok": False,
                "stopped": True,
                "message": "Quick verify stopped.",
                "broken": [],
                "frame_mismatch": [],
                "missing_split": [],
                "duplicate_split": [],
                "csv_ref_err": "",
                "duration_ok": False,
                "frames_ok": False,
                "retryable_failure": True,
            }))
        except Exception as e:
            self._log_queue.put(("verify_quick_result", {
                "ok": False,
                "message": f"Quick verify failed: {type(e).__name__}: {e}",
                "broken": [],
                "frame_mismatch": [],
                "missing_split": [],
                "duplicate_split": [],
                "csv_ref_err": "",
                "duration_ok": False,
                "frames_ok": False,
                "retryable_failure": False,
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
        script_path = utilities_path("verifyscenes.py")
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
        csv_ref_err = ""
        duplicate_files: list[str] = []
        frame_mismatch_files: list[str] = []
        try:
            for label, target_dir in targets:
                if self._verify_stop_requested:
                    raise VerifyStopRequested()
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
                proc = self._verify_popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                assert proc.stdout is not None
                try:
                    for raw in proc.stdout:
                        if self._verify_stop_requested:
                            raise VerifyStopRequested()
                        line = raw.rstrip("\n")
                        if line:
                            self._log_queue.put(("line", f"[DEEP][{label}] {line}"))
                            bad_path = self._resolve_verifyscenes_bad_path(line, target_dir)
                            if bad_path and bad_path not in seen_bad:
                                seen_bad.add(bad_path)
                                bad_files.append(bad_path)
                    rc = int(proc.wait() or 0)
                finally:
                    self._unregister_verify_process(proc)
                if rc != 0:
                    overall_rc = rc if overall_rc == 0 else overall_rc

            all_files: list[str] = []
            verify_patterns = self._scene_quick_verify_patterns()
            for _label, target_dir in targets:
                all_files.extend(self._collect_files_for_patterns(target_dir, verify_patterns))
            all_files = sorted(set(all_files))

            expected_by_name, csv_ref_err = self._scene_csv_expected_by_name(source_path)
            files_by_name: dict[str, list[str]] = {}
            for fp in all_files:
                files_by_name.setdefault(Path(fp).name, []).append(fp)

            if primary_target and expected_by_name:
                seg_root = Path(primary_target).resolve()
                for out_name in expected_by_name:
                    if not files_by_name.get(out_name):
                        missing_split_outputs.append(str((seg_root / out_name).resolve()))

            for out_name, paths in sorted(files_by_name.items()):
                if out_name in expected_by_name and len(paths) > 1:
                    duplicate_files.extend(sorted({str(p) for p in paths if str(p).strip()}))
                    self._log_queue.put(
                        (
                            "line",
                            (
                                f"[DEEP][DUPLICATE] {out_name} :: "
                                + " | ".join(sorted(paths))
                            ),
                        )
                    )

            if all_files:
                probe = self._quick_verify_probe_group(
                    all_files,
                    workers,
                    "line",
                    "scene",
                    prefix="[DEEP-FRAME]",
                    count_mode="frames",
                )
                for fp, meta in sorted((probe.get("meta_by_path") or {}).items()):
                    fp_txt = str(fp).strip()
                    if not fp_txt or not meta.get("ok"):
                        continue
                    out_name = Path(fp_txt).name
                    expected_info = expected_by_name.get(out_name)
                    if not expected_info:
                        continue
                    expected_frames = int(expected_info.get("frame_count", 0) or 0)
                    actual_frames = meta.get("frames")
                    if expected_frames <= 0:
                        continue
                    if actual_frames is None or int(actual_frames) != expected_frames:
                        frame_mismatch_files.append(fp_txt)
                        self._log_queue.put(
                            (
                                "line",
                                (
                                    f"[DEEP][FRAME-MISMATCH] {fp_txt} :: "
                                    f"expected={expected_frames}, actual="
                                    f"{'n.d.' if actual_frames is None else int(actual_frames)}"
                                ),
                            )
                        )

            if frame_mismatch_files:
                deleted, errors = self._auto_cleanup_broken_files(
                    frame_mismatch_files,
                    lambda line: self._log_queue.put(("line", line)),
                    "seg/seg-mono frame-mismatch",
                )
                self._log_queue.put(
                    (
                        "line",
                        (
                            f"[DEEP][FRAME-MISMATCH] auto-cleanup deleted={deleted}, "
                            f"errors={errors}"
                        ),
                    )
                )

            for fp in sorted(set(frame_mismatch_files)):
                if fp not in seen_bad:
                    seen_bad.add(fp)
                    bad_files.append(fp)

            if csv_ref_err:
                self._log_queue.put(("line", f"[DEEP][csv-ref] ERROR: {csv_ref_err}"))
                overall_rc = overall_rc or 1
            if duplicate_files:
                overall_rc = overall_rc or 1
            if frame_mismatch_files:
                overall_rc = overall_rc or 1
            if missing_split_outputs:
                self._log_queue.put(
                    (
                        "line",
                        (
                            f"[DEEP][split-files] missing split files: "
                            f"{len(missing_split_outputs)}"
                        ),
                    )
                )
                for miss in missing_split_outputs[:20]:
                    self._log_queue.put(("line", f"[DEEP][split-files][MISSING] {miss}"))
                overall_rc = overall_rc or 1
        except VerifyStopRequested:
            overall_rc = 1
        except Exception as e:
            self._log_queue.put(("line", f"[DEEP][ERROR] {type(e).__name__}: {e}"))
            overall_rc = 1
        finally:
            self._log_queue.put(
                (
                    "verify_deep_result",
                    {
                        "rc": overall_rc,
                        "stopped": bool(self._verify_stop_requested),
                        "seg_dir": primary_target,
                        "bad_files": bad_files,
                        "frame_mismatch": frame_mismatch_files,
                        "missing_split": missing_split_outputs,
                        "duplicate_split": duplicate_files,
                        "csv_ref_err": csv_ref_err,
                        "retryable_failure": (not csv_ref_err) and (len(duplicate_files) == 0),
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
        deleted_real_targets: set[str] = set()
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
            counted_deleted = False
            if fp.is_symlink():
                try:
                    real_target = fp.resolve(strict=True)
                except Exception:
                    real_target = None
                if real_target is not None and real_target != fp and real_target.is_file():
                    real_key = str(real_target)
                    if real_key not in deleted_real_targets:
                        try:
                            real_target.unlink()
                            deleted_real_targets.add(real_key)
                            counted_deleted = True
                        except Exception as e:
                            errors.append(f"{real_target}: {e}")
            try:
                fp.unlink()
                if not counted_deleted:
                    deleted += 1
            except Exception as e:
                errors.append(f"{fp}: {e}")
                continue
            if counted_deleted:
                deleted += 1
        return deleted, errors

    def _auto_cleanup_broken_files(self, paths: list[str], logger, label: str) -> tuple[int, int]:
        deleted, errors = self._delete_file_paths(paths)
        if deleted or errors:
            logger(f"[VERIFY][AUTO-CLEANUP] {label}: deleted={deleted}, errors={len(errors)}")
            for err in errors[:10]:
                logger(f"[VERIFY][AUTO-CLEANUP][ERR] {err}")
        return deleted, len(errors)

    def _cleanup_broken_files_with_confirmation(
        self,
        paths: list[str],
        logger,
        label: str,
        dialog_title: str,
    ) -> tuple[int, int, bool, int]:
        uniq = sorted({str(p) for p in (paths or []) if str(p).strip()})
        if not uniq:
            return 0, 0, False, 0

        flagged = len(uniq)
        if not self._pipeline_autorun:
            confirm_msg = (
                f"Verify found {flagged} corrupted/incomplete file(s) in {label}.\n\n"
                "Delete them now so the next run can regenerate them?\n"
                "Check the folder selection before confirming."
            )
            confirm_msg += self._format_corrupted_files_block(
                uniq,
                "Files marked for deletion",
                max_items=20,
            )
            if not messagebox.askyesno(dialog_title, confirm_msg):
                logger(
                    f"[VERIFY][AUTO-CLEANUP] {label}: skipped by user, "
                    f"flagged={flagged}"
                )
                return 0, 0, True, flagged

        deleted, errors = self._auto_cleanup_broken_files(uniq, logger, label)
        return deleted, errors, False, flagged

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
        script_path = utilities_path("split_scenes_from_csv.py")
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
        self._scene_stop_clicks = 0
        self._scene_active_step = "split_scenes"
        self._scene_stop_marker_path = os.path.join(
            self.scene_output_var.get().strip() or "./work/seg",
            ".stop_after_current",
        )
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
            proc_s = self._scene_process
            if self._scene_active_step == "split_scenes":
                self._scene_stop_clicks += 1
                if self._scene_stop_clicks == 1:
                    self.scene_status_var.set("Stop requested...")
                    self._append_scene_log(
                        "[STOP] graceful stop requested (click Stop again for immediate force stop)."
                    )
                    self.scene_stop_btn.configure(text="Force Stop")
                    marker_path = self._touch_stop_marker_file(
                        str(self._scene_stop_marker_path or "").strip()
                        or os.path.join(
                            self.scene_output_var.get().strip() or "./work/seg",
                            ".stop_after_current",
                        ),
                        self._append_scene_log,
                    )
                    if marker_path:
                        self._scene_stop_marker_path = marker_path
                else:
                    self.scene_status_var.set("Force stop requested...")
                    self._append_scene_log("[STOP] force stop requested.")
            else:
                self.scene_status_var.set("Stopping...")
                self._append_scene_log(f"Stopping {active_label}...")
            if proc_s is not None and proc_s.poll() is None and (
                self._scene_active_step != "split_scenes" or self._scene_stop_clicks >= 2
            ):
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
            if self._scene_active_step == "split_scenes":
                if self._scene_stop_clicks >= 2:
                    self.root.after(1000, self._force_kill_scene_detect)
            else:
                # Give ffmpeg/python splitter time to flush and close current output.
                self.root.after(6000, self._force_kill_scene_detect)
        self._refresh_pipeline_run_button()

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
                        self._finalize_pipeline_stop(stop_label)
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
                    if step_name == "depthcrafter":
                        self._pipeline_on_run_finished(step_name, success)
                    else:
                        self._pipeline_on_run_finished("depthcrafter", "completed" in status_txt)
                        step_name = "depthcrafter"
                    if stop_requested:
                        self._append_depth_log("[STOP] DepthCrafter stopped.")
                        self._finalize_pipeline_stop("DepthCrafter")
                elif kind == "splat_done":
                    stop_requested = bool(self._splat_stop_requested)
                    self._set_splat_running(False)
                    status_txt = self.splat_status_var.get().strip().lower()
                    self._pipeline_on_run_finished("splatting", "completed" in status_txt)
                    if stop_requested:
                        self._append_splat_log("[STOP] Splatting stopped.")
                        self._finalize_pipeline_stop("Splatting")
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
                    pending_sharpen_run = (
                        isinstance(pending_before, tuple)
                        and len(pending_before) >= 2
                        and str(pending_before[0]).strip().lower() == "sharpen"
                        and str(pending_before[1]).strip().lower() == "run"
                    )
                    should_resume_inpaint = False
                    should_resume_sharpen = False
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
                        if (
                            step_name == "sharpness_csv"
                            and bool(self._inpaint_resume_after_sharpen)
                            and success
                            and not stop_requested
                        ):
                            should_resume_sharpen = True
                    self._set_inpaint_running(False)
                    if step_name == "benchmark":
                        if success:
                            try:
                                self._apply_inpaint_tile_benchmark_results(
                                    payload.get("json_out") if isinstance(payload, dict) else ""
                                )
                            except Exception as exc:
                                success = False
                                self._append_inpaint_log(
                                    f"[BENCH][ERROR] Could not apply benchmark results: {exc}"
                                )
                                self.inpaint_status_var.set("Benchmark completed, apply failed")
                    elif step_name in {"sharpness_csv", "inpaint", "sharpen"}:
                        self._pipeline_on_run_finished(step_name, success)
                    else:
                        status_txt = self.inpaint_status_var.get().strip().lower()
                        if "sharpness csv created" in status_txt:
                            self._pipeline_on_run_finished("sharpness_csv", True)
                            step_name = "sharpness_csv"
                        elif "sharpen completed" in status_txt:
                            self._pipeline_on_run_finished("sharpen", True)
                            step_name = "sharpen"
                        elif "completed" in status_txt:
                            self._pipeline_on_run_finished("inpaint", True)
                            step_name = "inpaint"
                        else:
                            pending = self._pipeline_pending_action
                            if pending and pending[1] == "run" and pending[0] in {"sharpness_csv", "inpaint", "sharpen"}:
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
                    if step_name == "sharpness_csv" and bool(self._inpaint_resume_after_sharpen):
                        self._inpaint_resume_after_sharpen = False
                        if should_resume_sharpen:
                            self._append_inpaint_log(
                                "[SHARPEN] Sharpness CSV rebuilt. Resuming Sharpen automatically..."
                            )
                            self.root.after(10, self._start_inpaint_sharpen)
                        elif pending_sharpen_run and not success:
                            self._pipeline_on_run_finished("sharpen", False)
                    if stop_requested:
                        stop_label = (
                            "Sharpness CSV"
                            if step_name == "sharpness_csv"
                            else (
                                "Sharpen"
                                if step_name == "sharpen"
                                else ("Benchmark" if step_name == "benchmark" else "Inpainting")
                            )
                        )
                        self._append_inpaint_log(f"[STOP] {stop_label} stopped.")
                        self._finalize_pipeline_stop(stop_label)
                elif kind == "merge_done":
                    self._handle_merge_done_event(payload)
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
                    mark_completed = True
                    if step_name == "join":
                        mark_completed = bool(self._join_mark_completed)
                        if success:
                            self._set_join_incomplete_flag(not mark_completed)
                            if not mark_completed:
                                self.join_status_var.set("Completed (incomplete merge)")
                                self._append_join_log(
                                    "[JOIN] Completed in incomplete mode; step left not completed."
                                )
                        self._join_mark_completed = True
                    if step_name == "remux":
                        self._pipeline_on_run_finished("remux", success)
                    elif step_name == "mono_to_sbs":
                        self._pipeline_on_run_finished("mono_to_sbs", success)
                    else:
                        self._pipeline_on_run_finished(
                            "join",
                            success,
                            mark_completed=mark_completed,
                        )
                    if stop_requested:
                        label_map = {
                            "mono_to_sbs": "Mono->SBS",
                            "remux": "Remux",
                            "join": "Join",
                        }
                        stop_label = label_map.get(step_name, "Join")
                        self._append_join_log(f"[STOP] {stop_label} stopped.")
                        self._finalize_pipeline_stop(stop_label)
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
                elif (
                    self._is_verify_result_kind(kind)
                    and isinstance(payload, dict)
                    and (bool(payload.get("stopped")) or self._verify_stop_requested)
                ):
                    self._handle_stopped_verify_result(kind, payload)
                elif kind == "verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Quick verification finished."))
                    broken_files = [str(p) for p in (payload.get("broken") or []) if str(p).strip()]
                    frame_mismatch = [str(p) for p in (payload.get("frame_mismatch") or []) if str(p).strip()]
                    missing_split = [str(p) for p in (payload.get("missing_split") or []) if str(p).strip()]
                    duplicate_split = [str(p) for p in (payload.get("duplicate_split") or []) if str(p).strip()]
                    csv_ref_err = str(payload.get("csv_ref_err", "")).strip()
                    retryable_failure = bool(payload.get("retryable_failure", True))
                    if ok:
                        self.scene_status_var.set("Verify (Quick) completed")
                        messagebox.showinfo("Verify Scenes (Quick)", msg)
                    else:
                        self.scene_status_var.set("Verify (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_files:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_files,
                                    self._append_scene_log,
                                    "seg/seg-mono",
                                    "Verify Scenes (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_files:
                            msg += self._format_corrupted_files_block(
                                broken_files,
                                "Broken or wrong-length scene files",
                            )
                        if duplicate_split:
                            msg += self._format_corrupted_files_block(
                                duplicate_split,
                                "Duplicate split scene files",
                            )
                        if csv_ref_err:
                            msg += f"\n\nCSV reference error:\n{csv_ref_err}"
                        if missing_split:
                            msg += self._format_corrupted_files_block(
                                missing_split,
                                "Missing split files",
                            )
                        messagebox.showwarning("Verify Scenes (Quick)", msg)
                    self._scene_verify_result_applied = True
                    self._pipeline_on_verify_finished(
                        "split_scenes",
                        ok,
                        "quick",
                        retry_on_failure=retryable_failure,
                    )
                elif kind == "verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    frame_mismatch = [str(p) for p in (payload.get("frame_mismatch") or []) if str(p).strip()]
                    missing_split = [str(p) for p in (payload.get("missing_split") or []) if str(p).strip()]
                    duplicate_split = [str(p) for p in (payload.get("duplicate_split") or []) if str(p).strip()]
                    csv_ref_err = str(payload.get("csv_ref_err", "")).strip()
                    retryable_failure = bool(payload.get("retryable_failure", True))
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
                        if csv_ref_err:
                            warn_msg += f"\n\nCSV reference error:\n{csv_ref_err}"
                        warn_msg += self._format_corrupted_files_block(
                            bad_files,
                            "Broken or wrong-length scene files",
                        )
                        warn_msg += self._format_corrupted_files_block(
                            duplicate_split,
                            "Duplicate split scene files",
                        )
                        warn_msg += self._format_corrupted_files_block(
                            missing_split,
                            "Missing split files",
                        )
                        messagebox.showwarning(
                            "Verify Scenes (Deep)",
                            warn_msg,
                        )
                    self._scene_verify_result_applied = True
                    self._pipeline_on_verify_finished(
                        "split_scenes",
                        rc == 0,
                        "deep",
                        retry_on_failure=retryable_failure,
                    )
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
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_depth:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_depth,
                                    self._append_depth_log,
                                    "depthmap",
                                    "Verify Depth (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
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
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_targets:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_targets,
                                    self._append_splat_log,
                                    "splat targets",
                                    "Verify Splatting (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
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
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_output:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_output,
                                    self._append_inpaint_log,
                                    "inpaint output",
                                    "Verify Inpainting (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
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
                elif kind == "sharpen_verify_quick_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Sharpen quick verification finished."))
                    broken_output = [
                        str(p) for p in (payload.get("broken_output") or []) if str(p).strip()
                    ]
                    broken_reference = [
                        str(p) for p in (payload.get("broken_reference") or []) if str(p).strip()
                    ]
                    missing_output = [
                        str(p) for p in (payload.get("missing_output") or []) if str(p).strip()
                    ]
                    if ok:
                        self.inpaint_status_var.set("Verify Sharpen (Quick) completed")
                        messagebox.showinfo("Verify Sharpen (Quick)", msg)
                    else:
                        self.inpaint_status_var.set("Verify Sharpen (Quick) failed")
                        deleted = 0
                        cleanup_err = 0
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_output:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_output,
                                    self._append_inpaint_log,
                                    "sharpen output",
                                    "Verify Sharpen (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
                            msg = (
                                f"{msg}\n\n"
                                f"Auto-cleanup: deleted {deleted} broken file(s), "
                                f"errors={cleanup_err}."
                            )
                        if broken_output:
                            msg += self._format_corrupted_files_block(
                                broken_output,
                                "Corrupted sharpen output files",
                            )
                        if missing_output:
                            msg += self._format_corrupted_files_block(
                                missing_output,
                                "Missing eligible sharpen output files",
                            )
                        if broken_reference:
                            msg += self._format_corrupted_files_block(
                                broken_reference,
                                "Corrupted reference files",
                            )
                        messagebox.showwarning("Verify Sharpen (Quick)", msg)
                    self._pipeline_on_verify_finished("sharpen", ok, "quick")
                elif kind == "sharpen_verify_deep_result" and isinstance(payload, dict):
                    rc = int(payload.get("rc", 1))
                    bad_files = [str(p) for p in (payload.get("bad_files") or []) if str(p).strip()]
                    if rc == 0:
                        self.inpaint_status_var.set("Verify Sharpen (Deep) completed")
                        messagebox.showinfo(
                            "Verify Sharpen (Deep)",
                            "Deep verification completed successfully.",
                        )
                    else:
                        self.inpaint_status_var.set(f"Verify Sharpen (Deep) failed (exit {rc})")
                        messagebox.showwarning(
                            "Verify Sharpen (Deep)",
                            (
                                "Deep verification failed.\n\n"
                                "Broken files were flagged from the eligible sharpen subset."
                            )
                            + self._format_corrupted_files_block(
                                bad_files,
                                "Corrupted sharpen output files",
                            ),
                        )
                    self._pipeline_on_verify_finished("sharpen", rc == 0, "deep")
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
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_output:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_output,
                                    self._append_merge_log,
                                    "mask_for_merge output",
                                    "Verify Mask (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
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
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_output:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_output,
                                    self._append_merge_log,
                                    "merged output",
                                    "Verify Merging (Quick)",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
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
                    deleted = int(payload.get("deleted", 0) or 0)
                    cleanup_errors = [str(x) for x in (payload.get("cleanup_errors") or []) if str(x).strip()]
                    ignored_seg_mono = [
                        str(p) for p in (payload.get("ignored_seg_mono") or []) if str(p).strip()
                    ]
                    unmatched_output = [
                        str(p) for p in (payload.get("unmatched_output") or []) if str(p).strip()
                    ]
                    if rc == 0:
                        self.merge_status_var.set("Verify (Deep) completed")
                        messagebox.showinfo(
                            "Verify Merging (Deep)",
                            (
                                "Deep verification completed successfully."
                                + (
                                    f"\n\nIgnored seg-mono SBS files: {len(ignored_seg_mono)}"
                                    if ignored_seg_mono
                                    else ""
                                )
                            ),
                        )
                    else:
                        self.merge_status_var.set(f"Verify (Deep) failed (exit {rc})")
                        warn_msg = "Deep verification failed."
                        if deleted or cleanup_errors:
                            warn_msg += (
                                "\n\n"
                                f"Cleanup after deep verify: deleted {deleted} broken merged file(s), "
                                f"errors={len(cleanup_errors)}."
                            )
                        else:
                            warn_msg += (
                                "\n\nBroken files were auto-deleted by verifier where possible."
                            )
                        if ignored_seg_mono:
                            warn_msg += f"\n\nIgnored seg-mono SBS files: {len(ignored_seg_mono)}."
                        warn_msg += self._format_corrupted_files_block(
                            unmatched_output,
                            "Unexpected extra merged outputs",
                        )
                        warn_msg += self._format_corrupted_files_block(
                            bad_files,
                            "Corrupted merged output files",
                        )
                        messagebox.showwarning(
                            "Verify Merging (Deep)",
                            warn_msg,
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
                    mode = "quick"
                    self._pipeline_on_verify_finished("join", ok, mode)
                elif kind == "join_mono_verify_result" and isinstance(payload, dict):
                    ok = bool(payload.get("ok", False))
                    msg = str(payload.get("message", "Mono->SBS verification finished."))
                    mode = "quick"
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
                        cleanup_skipped = False
                        cleanup_total = 0
                        if broken_output:
                            deleted, cleanup_err, cleanup_skipped, cleanup_total = (
                                self._cleanup_broken_files_with_confirmation(
                                    broken_output,
                                    self._append_join_log,
                                    "mono_to_sbs output",
                                    "Verify Mono->SBS",
                                )
                            )
                        if cleanup_skipped:
                            msg = (
                                f"{msg}\n\n"
                                f"Cleanup skipped by user: {cleanup_total} file(s) flagged for deletion."
                            )
                        elif deleted or cleanup_err:
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
                    verify_stopped = bool(self._verify_stop_requested)
                    pipeline_stop_requested = bool(self._pipeline_stop_requested)
                    if (
                        mode_txt == "quick"
                        and not self._scene_verify_result_applied
                        and not verify_stopped
                    ):
                        status_txt = self.scene_status_var.get().strip().lower()
                        verify_ok = ("completed" in status_txt) and ("failed" not in status_txt)
                        self._pipeline_on_verify_finished("split_scenes", verify_ok, mode_txt)
                    self._scene_verify_result_applied = False
                    self._verify_thread = None
                    self._set_verify_running(False)
                    if verify_stopped and pipeline_stop_requested:
                        self._pipeline_autorun = False
                        self._pipeline_pending_action = None
                        if self._pipeline_test_active:
                            self._restore_test_scene_subset()
                        self.pipeline_run_status_var.set("Pipeline stopped during verification.")
                        self._pipeline_stop_requested = False
                        self._pipeline_sync_noninteractive_mode()
                    elif self._pipeline_autorun:
                        self._pipeline_trigger_next_action()
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log_queue)

    def _collect_config(self) -> dict:
        return pm_config.collect_config(self)

    def _current_window_geometry(self) -> str:
        try:
            self.root.update_idletasks()
            return str(self.root.geometry())
        except Exception:
            return self.DEFAULT_WINDOW_GEOMETRY

    def _load_config(self) -> dict:
        if not os.path.isfile(self._config_file):
            return {}
        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _save_config(self) -> None:
        data = self._collect_config()
        try:
            os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
            with open(self._config_file, "w", encoding="utf-8") as f:
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
        if (self._merge_thread and self._merge_thread.is_alive()) or self._merge_group_alive():
            self._stop_merge_placeholder(prompt_user=False)
            self._stop_merge_placeholder(prompt_user=False)
        if self._join_thread and self._join_thread.is_alive():
            self._stop_join(prompt_user=False)
        if self._pipeline_test_active:
            self._restore_test_scene_subset(force=True)
        self.root.destroy()


def create_root() -> tk.Tk:
    if ThemedTk is not None:
        try:
            return ThemedTk(theme="clam")
        except Exception:
            pass
    return tk.Tk()


def main() -> None:
    parser = argparse.ArgumentParser(description="StereoCrafter Pipeline GUI")
    parser.add_argument(
        "--work_dir",
        default="",
        help="Optional work directory. When provided, config is loaded/saved as <work_dir>/config_pipeline_master_gui.json.",
    )
    args = parser.parse_args()
    work_dir = str(args.work_dir or "").strip()
    config_file = (
        str(Path(work_dir).expanduser().resolve() / "config_pipeline_master_gui.json")
        if work_dir
        else str(DEFAULT_PIPELINE_MASTER_CONFIG_PATH)
    )
    root = create_root()
    PipelineMasterGUI(root, config_file=config_file, work_dir_override=work_dir or None)
    root.mainloop()


if __name__ == "__main__":
    main()
