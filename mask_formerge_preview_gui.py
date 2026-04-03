#!/usr/bin/env python3
"""
Mask-for-merge preview-only GUI.

Purpose:
- Tune mask preprocessing/shadow parameters visually on existing replace-mask clips.
- Show only two preview sources: original mask and processed mask.
- No batch processing and no output writing.
"""

from __future__ import annotations

import glob
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu  # type: ignore
from PIL import Image, ImageTk

from mask_formerge_nogui import (
    apply_gaussian_blur,
    apply_mask_dilation,
    apply_shadow_blur,
    load_motion_defaults,
)


def _parse_percent(text: str, fallback: float = 100.0) -> float:
    raw = str(text or "").strip().replace("%", "")
    try:
        v = float(raw)
    except Exception:
        return float(fallback)
    return float(max(10.0, min(400.0, v)))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _ratio_to_area_reset_pct(ratio: float) -> float:
    r = float(max(1.0, ratio))
    return float(max(0.0, min(100.0, (r - 1.0) * 100.0)))


def _area_reset_pct_to_ratio(pct: float) -> float:
    p = float(max(0.0, min(100.0, pct)))
    return float(1.0 + p / 100.0)


def _load_canonical_motion_defaults() -> Dict[str, Any]:
    defaults = dict(load_motion_defaults())
    defaults["shadow_area_reset_pct"] = _ratio_to_area_reset_pct(
        float(defaults.get("shadow_area_reset_ratio", 1.0))
    )
    return defaults


class MaskForMergePreviewGUI(tk.Tk):
    PREVIEW_SOURCES = ["Mask Original", "Mask Processed"]
    PREVIEW_SIZES = ["50%", "75%", "100%", "125%", "150%", "200%"]
    DEFAULT_WARMUP_FRAMES = 8
    WARMUP_MAX_FRAMES = 120
    PLAY_INTERVAL_MS = 40
    SETTINGS_FILENAME = "config_mask_formerge_preview_gui.json"
    MOTION_DEFAULTS_FILENAME = "config_mask_formerge_nogui_motion_defaults.json"

    def __init__(self) -> None:
        super().__init__()
        self.title("Mask-for-merge Preview GUI")
        self.geometry("1460x960")
        self.motion_defaults = _load_canonical_motion_defaults()

        self.input_folder_var = tk.StringVar(value="./work/mask/")
        self.input_glob_var = tk.StringVar(value="*_replace_mask.*")
        self.preview_source_var = tk.StringVar(value=self.PREVIEW_SOURCES[1])
        self.preview_size_var = tk.StringVar(value="100%")
        self.current_file_var = tk.StringVar(value="")
        self.frame_idx_var = tk.DoubleVar(value=0.0)
        self.frame_label_var = tk.StringVar(value="Frame: 0 / 0")
        self.status_var = tk.StringVar(value="Ready.")
        self.scene_jump_var = tk.StringVar(value="1")
        self.play_button_text_var = tk.StringVar(value="Play")

        self.mask_binarize_threshold_var = tk.DoubleVar(value=0.5)
        self.mask_dilate_kernel_size_var = tk.IntVar(value=2)
        self.mask_blur_kernel_size_var = tk.IntVar(value=4)
        self.shadow_length_px_var = tk.IntVar(value=25)
        self.shadow_curve_var = tk.DoubleVar(value=0.0)
        self.shadow_motion_gain_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_motion_gain"])
        )
        self.shadow_motion_deadzone_px_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_motion_deadzone_px"])
        )
        self.shadow_motion_max_px_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_motion_max_px"])
        )
        self.shadow_area_min_px_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_area_min_px"])
        )
        self.shadow_area_max_px_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_area_max_px"])
        )
        self.shadow_area_reset_pct_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_area_reset_pct"])
        )
        self.shadow_area_reset_abs_px_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_area_reset_abs_px"])
        )
        self.shadow_component_merge_y_tol_px_var = tk.IntVar(
            value=int(round(float(self.motion_defaults["shadow_component_merge_y_tol_px"])))
        )
        self.shadow_alpha_down_var = tk.DoubleVar(
            value=float(self.motion_defaults["shadow_alpha_down"])
        )
        self.shadow_width_adaptive_var = tk.BooleanVar(value=True)
        self.shadow_motion_chain_enabled_var = tk.BooleanVar(
            value=bool(self.motion_defaults["shadow_motion_chain_enabled"])
        )
        self.use_gpu_mask_ops_var = tk.BooleanVar(value=False)
        self.warmup_frames_var = tk.IntVar(value=self.DEFAULT_WARMUP_FRAMES)

        self._files: List[str] = []
        self._reader: Optional[VideoReader] = None
        self._num_frames: int = 0
        self._tk_preview_img: Optional[ImageTk.PhotoImage] = None
        self._refresh_after_id: Optional[str] = None
        self._shadow_temporal_cache: Optional[Dict[str, Any]] = None
        self._play_after_id: Optional[str] = None
        self._is_playing: bool = False
        self._play_button: Optional[ttk.Button] = None
        self._settings_path = os.path.abspath(self.SETTINGS_FILENAME)
        self._motion_defaults_path = os.path.abspath(self.MOTION_DEFAULTS_FILENAME)
        self._startup_restore_clip_name: Optional[str] = None
        self._startup_restore_frame_idx: Optional[int] = None

        self._load_settings()
        self._build_ui()
        self._scan_files()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="Input Mask Folder:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(top, textvariable=self.input_folder_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self._on_browse_input).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="Glob:").grid(row=0, column=3, sticky="e", padx=(12, 6))
        ttk.Entry(top, textvariable=self.input_glob_var, width=22).grid(row=0, column=4, sticky="w")
        ttk.Button(top, text="Refresh", command=self._scan_files).grid(row=0, column=5, padx=(6, 0))

        nav = ttk.Frame(self, padding=(8, 0, 8, 8))
        nav.pack(fill=tk.X)
        nav.columnconfigure(1, weight=1)

        ttk.Label(nav, text="Clip:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.file_combo = ttk.Combobox(
            nav,
            textvariable=self.current_file_var,
            state="readonly",
            values=[],
        )
        self.file_combo.grid(row=0, column=1, sticky="ew")
        self.file_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_select_clip())

        ttk.Button(nav, text="Prev Clip", command=lambda: self._step_clip(-1)).grid(
            row=0, column=2, padx=(6, 4)
        )
        ttk.Button(nav, text="Next Clip", command=lambda: self._step_clip(1)).grid(
            row=0, column=3, padx=(0, 8)
        )

        ttk.Label(nav, text="Source:").grid(row=0, column=4, sticky="e", padx=(0, 6))
        src_combo = ttk.Combobox(
            nav,
            textvariable=self.preview_source_var,
            state="readonly",
            values=list(self.PREVIEW_SOURCES),
            width=18,
        )
        src_combo.grid(row=0, column=5, sticky="w")
        src_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_preview())

        ttk.Label(nav, text="Preview Size:").grid(row=0, column=6, sticky="e", padx=(12, 6))
        size_combo = ttk.Combobox(
            nav,
            textvariable=self.preview_size_var,
            state="readonly",
            values=list(self.PREVIEW_SIZES),
            width=8,
        )
        size_combo.grid(row=0, column=7, sticky="w")
        size_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_preview())

        ttk.Button(nav, text="Prev Frame", command=lambda: self._step_frame(-1)).grid(
            row=0, column=8, padx=(12, 4)
        )
        ttk.Button(nav, text="Next Frame", command=lambda: self._step_frame(1)).grid(
            row=0, column=9, padx=(0, 0)
        )
        ttk.Label(nav, text="Scene #:").grid(row=0, column=10, sticky="e", padx=(12, 6))
        scene_entry = ttk.Entry(nav, textvariable=self.scene_jump_var, width=7)
        scene_entry.grid(row=0, column=11, sticky="w")
        scene_entry.bind("<Return>", lambda _e: self._jump_to_scene())
        ttk.Button(nav, text="Go", command=self._jump_to_scene, width=5).grid(
            row=0, column=12, padx=(4, 8)
        )
        self._play_button = ttk.Button(
            nav,
            textvariable=self.play_button_text_var,
            command=self._toggle_play,
            width=8,
        )
        self._play_button.grid(row=0, column=13, sticky="w")

        frame_nav = ttk.Frame(self, padding=(8, 0, 8, 8))
        frame_nav.pack(fill=tk.X)
        frame_nav.columnconfigure(0, weight=1)
        self.frame_scale = tk.Scale(
            frame_nav,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_idx_var,
            showvalue=False,
            command=lambda _v: self._schedule_preview(),
            resolution=1,
        )
        self.frame_scale.grid(row=0, column=0, sticky="ew")
        self.frame_scale.bind("<ButtonRelease-1>", lambda _e: self._schedule_preview(10))
        ttk.Label(frame_nav, textvariable=self.frame_label_var).grid(row=0, column=1, padx=(10, 0))

        body = ttk.Frame(self, padding=8)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        params = ttk.LabelFrame(body, text="Mask Params", padding=8)
        params.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self._add_sliders(params)

        preview_box = ttk.LabelFrame(body, text="Preview", padding=8)
        preview_box.grid(row=0, column=1, sticky="nsew")
        preview_box.rowconfigure(0, weight=1)
        preview_box.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            preview_box,
            bg="black",
            highlightthickness=0,
            width=960,
            height=600,
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(8, 0, 8, 8))
        status.pack(fill=tk.X)

    def _add_sliders(self, parent: ttk.LabelFrame) -> None:
        parent.columnconfigure(0, weight=1)

        base_frame = ttk.LabelFrame(parent, text="Base / Preprocess", padding=6)
        base_frame.grid(row=0, column=0, sticky="ew")
        base_frame.columnconfigure(0, weight=1)

        motion_frame = ttk.LabelFrame(
            parent,
            text="Motion Mask (Hardcoded / non esposta in Pipeline-Merging)",
            padding=6,
        )
        motion_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        motion_frame.columnconfigure(0, weight=1)

        row = 0
        self._slider(base_frame, "Mask Binarize Thr", self.mask_binarize_threshold_var, -1.0, 1.0, row, 0.01)
        row += 1
        self._slider(base_frame, "Mask Dilate Kernel", self.mask_dilate_kernel_size_var, 0, 101, row, 1)
        row += 1
        self._slider(base_frame, "Mask Blur Kernel", self.mask_blur_kernel_size_var, 0, 101, row, 1)
        row += 1
        self._slider(base_frame, "Shadow Length (px)", self.shadow_length_px_var, 0, 120, row, 1)
        row += 1
        self._slider(base_frame, "Shadow Curve", self.shadow_curve_var, -1.0, 1.0, row, 0.01)
        row += 1
        self._slider(
            base_frame,
            "Warmup Frames (Preview)",
            self.warmup_frames_var,
            0,
            self.WARMUP_MAX_FRAMES,
            row,
            1,
        )

        row = 0
        self._slider(motion_frame, "Shadow Motion Gain", self.shadow_motion_gain_var, 0.0, 4.0, row, 0.01)
        row += 1
        self._slider(motion_frame, "Motion Deadzone (px)", self.shadow_motion_deadzone_px_var, 0.0, 80.0, row, 0.5)
        row += 1
        self._slider(motion_frame, "Motion Max (px)", self.shadow_motion_max_px_var, 0.0, 160.0, row, 0.5)
        row += 1
        self._slider(motion_frame, "Area Min (px)", self.shadow_area_min_px_var, 0.0, 15000.0, row, 10.0)
        row += 1
        self._slider(motion_frame, "Area Max (px)", self.shadow_area_max_px_var, 0.0, 15000.0, row, 10.0)
        row += 1
        self._slider(motion_frame, "Area Reset (%)", self.shadow_area_reset_pct_var, 0.0, 100.0, row, 1.0)
        row += 1
        self._slider(motion_frame, "Area Reset Abs", self.shadow_area_reset_abs_px_var, 0.0, 200000.0, row, 10.0)
        row += 1
        self._slider(
            motion_frame,
            "Comp Merge Y Tol (px)",
            self.shadow_component_merge_y_tol_px_var,
            0,
            40,
            row,
            1,
        )
        row += 1
        self._slider(motion_frame, "Shadow Alpha Down", self.shadow_alpha_down_var, 0.0, 1.0, row, 0.01)
        row += 1

        motion_actions = ttk.Frame(motion_frame)
        motion_actions.grid(row=row, column=0, sticky="ew", pady=(8, 2))
        ttk.Button(
            motion_actions,
            text="Reset Motion Defaults",
            command=self._reset_motion_defaults,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            motion_actions,
            text="Push Motion Settings to GUI + noGUI",
            command=self._push_motion_settings_to_targets,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        checks = ttk.Frame(parent)
        checks.grid(row=2, column=0, sticky="ew", pady=(8, 2))
        ttk.Checkbutton(
            checks,
            text="Motion Chain Enabled",
            variable=self.shadow_motion_chain_enabled_var,
            command=self._schedule_preview,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            checks,
            text="Shadow Width Adaptive",
            variable=self.shadow_width_adaptive_var,
            command=self._schedule_preview,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            checks,
            text="Use GPU Mask Ops",
            variable=self.use_gpu_mask_ops_var,
            command=self._schedule_preview,
        ).grid(row=2, column=0, sticky="w")

    def _slider(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.Variable,
        lo: float,
        hi: float,
        row: int,
        resolution: float,
    ) -> None:
        line = ttk.Frame(parent)
        line.grid(row=row, column=0, sticky="ew", pady=1)
        line.columnconfigure(1, weight=1)
        ttk.Label(line, text=label, width=22).grid(row=0, column=0, sticky="w")
        scale = tk.Scale(
            line,
            from_=lo,
            to=hi,
            orient=tk.HORIZONTAL,
            variable=var,
            resolution=resolution,
            showvalue=False,
            command=lambda _v: self._schedule_preview(),
            length=260,
        )
        scale.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        value_lbl = ttk.Label(line, width=10, anchor="e")
        value_lbl.grid(row=0, column=2, sticky="e")

        def _update_label(*_args: object) -> None:
            try:
                v = float(var.get())  # type: ignore[arg-type]
                if resolution >= 1:
                    value_lbl.configure(text=f"{int(round(v))}")
                else:
                    value_lbl.configure(text=f"{v:.2f}")
            except Exception:
                value_lbl.configure(text=str(var.get()))

        var.trace_add("write", _update_label)
        _update_label()

    def _on_browse_input(self) -> None:
        path = filedialog.askdirectory(
            title="Select replace-mask folder",
            initialdir=self.input_folder_var.get().strip() or ".",
        )
        if not path:
            return
        self.input_folder_var.set(path)
        self._scan_files()

    def _scan_files(self) -> None:
        self._stop_playback(silent=True)
        folder = self.input_folder_var.get().strip()
        glob_expr = self.input_glob_var.get().strip() or "*_replace_mask.*"
        if not folder or not os.path.isdir(folder):
            self.status_var.set(f"Invalid input folder: {folder or '(empty)'}")
            self._set_files([])
            return

        patterns = [p.strip() for p in glob_expr.split(",") if p.strip()]
        if not patterns:
            patterns = ["*_replace_mask.*"]
        found: List[str] = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(folder, pat)))
        files = sorted({os.path.abspath(p) for p in found if os.path.isfile(p)})
        self._set_files(files)

    def _set_files(self, files: List[str]) -> None:
        self._files = files
        names = [os.path.basename(p) for p in files]
        self.file_combo["values"] = names
        if not files:
            self.scene_jump_var.set("1")
            self.current_file_var.set("")
            self._close_reader()
            self._shadow_temporal_cache = None
            self.preview_canvas.delete("all")
            self.frame_label_var.set("Frame: 0 / 0")
            self.status_var.set("No files found.")
            return
        target_idx = 0
        if self._startup_restore_clip_name:
            for i, p in enumerate(files):
                if os.path.basename(p) == self._startup_restore_clip_name:
                    target_idx = i
                    break
        self.current_file_var.set(names[target_idx])
        self._open_reader_by_index(target_idx)
        if self._startup_restore_frame_idx is not None and self._num_frames > 0:
            frame = max(0, min(self._num_frames - 1, int(self._startup_restore_frame_idx)))
            self.frame_idx_var.set(float(frame))
            self._schedule_preview(10)
        self._startup_restore_clip_name = None
        self._startup_restore_frame_idx = None

    def _close_reader(self) -> None:
        try:
            if self._reader is not None:
                del self._reader
        except Exception:
            pass
        self._reader = None
        self._num_frames = 0

    def _open_reader_by_index(self, idx: int) -> None:
        if not self._files:
            return
        self._stop_playback(silent=True)
        idx = max(0, min(len(self._files) - 1, idx))
        path = self._files[idx]
        try:
            self._close_reader()
            self._reader = VideoReader(path, ctx=cpu(0))
            self._num_frames = int(len(self._reader))
            if self._num_frames <= 0:
                raise RuntimeError("video has zero frames")
            self.current_file_var.set(os.path.basename(path))
            self.frame_scale.configure(to=max(0, self._num_frames - 1))
            self.frame_idx_var.set(0.0)
            self._shadow_temporal_cache = None
            self._sync_scene_jump()
            self.status_var.set(
                f"Loaded: {os.path.basename(path)} ({self._num_frames} frames) | "
                f"scene={idx + 1}/{len(self._files)}"
            )
            self._schedule_preview(10)
        except Exception as e:
            self._close_reader()
            messagebox.showerror("Preview", f"Failed to open video:\n{path}\n\n{e}")

    def _current_file_index(self) -> int:
        name = self.current_file_var.get().strip()
        for i, p in enumerate(self._files):
            if os.path.basename(p) == name:
                return i
        return 0

    def _on_select_clip(self) -> None:
        self._stop_playback(silent=True)
        self._open_reader_by_index(self._current_file_index())

    def _step_clip(self, step: int) -> None:
        if not self._files:
            return
        self._stop_playback(silent=True)
        cur = self._current_file_index()
        nxt = max(0, min(len(self._files) - 1, cur + int(step)))
        self._open_reader_by_index(nxt)

    def _step_frame(self, step: int) -> None:
        if self._reader is None or self._num_frames <= 0:
            return
        self._stop_playback(silent=True)
        cur = int(self.frame_idx_var.get())
        nxt = max(0, min(self._num_frames - 1, cur + int(step)))
        self.frame_idx_var.set(float(nxt))
        self._schedule_preview(10)

    def _sync_scene_jump(self) -> None:
        if not self._files:
            self.scene_jump_var.set("1")
            return
        idx = self._current_file_index()
        self.scene_jump_var.set(str(max(1, idx + 1)))

    def _jump_to_scene(self) -> None:
        if not self._files:
            return
        try:
            scene_idx_1b = int(str(self.scene_jump_var.get()).strip())
        except Exception:
            self.status_var.set("Invalid scene number.")
            self._sync_scene_jump()
            return
        scene_idx_1b = max(1, min(len(self._files), scene_idx_1b))
        self.scene_jump_var.set(str(scene_idx_1b))
        self._open_reader_by_index(scene_idx_1b - 1)

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._stop_playback(silent=True)
            self.status_var.set("Playback stopped.")
            return
        self._start_playback()

    def _start_playback(self) -> None:
        if self._reader is None or self._num_frames <= 0:
            self.status_var.set("No clip loaded.")
            return
        if int(self.frame_idx_var.get()) >= (self._num_frames - 1):
            self.status_var.set("Already at last frame.")
            return
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
            self._refresh_after_id = None
        self._is_playing = True
        self.play_button_text_var.set("Stop")
        self.status_var.set("Playback running...")
        self._play_tick()

    def _stop_playback(self, silent: bool = False) -> None:
        if self._play_after_id is not None:
            try:
                self.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None
        was_playing = self._is_playing
        self._is_playing = False
        self.play_button_text_var.set("Play")
        if was_playing and not silent:
            self.status_var.set("Playback stopped.")

    def _play_tick(self) -> None:
        if not self._is_playing:
            return
        if self._reader is None or self._num_frames <= 0:
            self._stop_playback(silent=True)
            return
        cur = int(self.frame_idx_var.get())
        if cur >= self._num_frames - 1:
            self._stop_playback(silent=True)
            self.status_var.set("Playback completed (end of clip).")
            return
        self.frame_idx_var.set(float(cur + 1))
        self._update_preview()
        self._play_after_id = self.after(self.PLAY_INTERVAL_MS, self._play_tick)

    def _schedule_preview(self, delay_ms: int = 80) -> None:
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
            self._refresh_after_id = None
        self._refresh_after_id = self.after(delay_ms, self._update_preview)

    def _get_mask_tensor(self, frame_idx: int, device: torch.device) -> torch.Tensor:
        if self._reader is None:
            raise RuntimeError("No reader loaded.")
        frame_np = self._reader[frame_idx].asnumpy()
        if frame_np.ndim == 3:
            gray = frame_np[..., :3].mean(axis=2)
        elif frame_np.ndim == 2:
            gray = frame_np
        else:
            raise RuntimeError(f"Unsupported frame shape: {frame_np.shape}")
        gray = gray.astype(np.float32)
        if gray.size > 0 and float(np.nanmax(gray)) > 1.5:
            gray = gray / 255.0
        gray = np.clip(gray, 0.0, 1.0)
        t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        return t.to(device=device)

    def _params_key(self, video_path: str, use_gpu_ops: bool) -> Tuple[Any, ...]:
        return (
            video_path,
            float(self.mask_binarize_threshold_var.get()),
            int(self.mask_dilate_kernel_size_var.get()),
            int(self.mask_blur_kernel_size_var.get()),
            int(self.shadow_length_px_var.get()),
            float(self.shadow_curve_var.get()),
            float(self.shadow_motion_gain_var.get()),
            float(self.shadow_motion_deadzone_px_var.get()),
            float(self.shadow_motion_max_px_var.get()),
            float(self.shadow_area_min_px_var.get()),
            float(self.shadow_area_max_px_var.get()),
            float(self.shadow_area_reset_pct_var.get()),
            float(self.shadow_area_reset_abs_px_var.get()),
            int(self.shadow_component_merge_y_tol_px_var.get()),
            float(self.shadow_alpha_down_var.get()),
            bool(self.shadow_width_adaptive_var.get()),
            bool(self.shadow_motion_chain_enabled_var.get()),
            bool(use_gpu_ops),
            int(self.warmup_frames_var.get()),
        )

    def _motion_settings_payload(self) -> Dict[str, Any]:
        area_reset_pct = float(self.shadow_area_reset_pct_var.get())
        return {
            "shadow_motion_gain": float(self.shadow_motion_gain_var.get()),
            "shadow_motion_deadzone_px": float(self.shadow_motion_deadzone_px_var.get()),
            "shadow_motion_max_px": float(self.shadow_motion_max_px_var.get()),
            "shadow_area_min_px": float(self.shadow_area_min_px_var.get()),
            "shadow_area_max_px": float(self.shadow_area_max_px_var.get()),
            "shadow_area_reset_pct": area_reset_pct,
            "shadow_area_reset_ratio": _area_reset_pct_to_ratio(area_reset_pct),
            "shadow_area_reset_abs_px": float(self.shadow_area_reset_abs_px_var.get()),
            "shadow_component_merge_y_tol_px": int(self.shadow_component_merge_y_tol_px_var.get()),
            "shadow_alpha_down": float(self.shadow_alpha_down_var.get()),
            "shadow_motion_chain_enabled": bool(self.shadow_motion_chain_enabled_var.get()),
        }

    def _reset_motion_defaults(self) -> None:
        self.motion_defaults = _load_canonical_motion_defaults()
        self.shadow_motion_gain_var.set(float(self.motion_defaults["shadow_motion_gain"]))
        self.shadow_motion_deadzone_px_var.set(
            float(self.motion_defaults["shadow_motion_deadzone_px"])
        )
        self.shadow_motion_max_px_var.set(float(self.motion_defaults["shadow_motion_max_px"]))
        self.shadow_area_min_px_var.set(float(self.motion_defaults["shadow_area_min_px"]))
        self.shadow_area_max_px_var.set(float(self.motion_defaults["shadow_area_max_px"]))
        self.shadow_area_reset_pct_var.set(float(self.motion_defaults["shadow_area_reset_pct"]))
        self.shadow_area_reset_abs_px_var.set(
            float(self.motion_defaults["shadow_area_reset_abs_px"])
        )
        self.shadow_component_merge_y_tol_px_var.set(
            int(round(float(self.motion_defaults["shadow_component_merge_y_tol_px"])))
        )
        self.shadow_alpha_down_var.set(float(self.motion_defaults["shadow_alpha_down"]))
        self.shadow_motion_chain_enabled_var.set(
            bool(self.motion_defaults["shadow_motion_chain_enabled"])
        )
        self.status_var.set("Motion defaults restored.")
        self._schedule_preview(10)

    def _write_json_atomic(self, target_path: str, payload: Dict[str, Any]) -> None:
        tmp_path = f"{target_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True)
        os.replace(tmp_path, target_path)

    def _push_motion_settings_to_targets(self) -> None:
        payload = self._motion_settings_payload()
        payload["updated_from"] = "mask_formerge_preview_gui"
        written: List[str] = []
        failures: List[str] = []
        try:
            self._write_json_atomic(self._motion_defaults_path, payload)
            self.motion_defaults = _load_canonical_motion_defaults()
            written.append(os.path.basename(self._motion_defaults_path))
        except Exception as e:
            failures.append(f"canonical defaults: {e}")
            try:
                tmp_path = f"{self._motion_defaults_path}.tmp"
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        if failures:
            if written:
                self.status_var.set(
                    "Motion push partial: wrote "
                    + ", ".join(written)
                    + " | failed: "
                    + " ; ".join(failures)
                )
            else:
                self.status_var.set("Motion push failed: " + " ; ".join(failures))
        else:
            self.status_var.set("Motion settings pushed: " + ", ".join(written))
        self._schedule_preview(10)

    def _preprocess_mask(
        self,
        mask: torch.Tensor,
        use_gpu_ops: bool,
        shadow_state: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        proc = mask
        thr = float(self.mask_binarize_threshold_var.get())
        if thr >= 0.0:
            proc = (proc > thr).float()
        else:
            proc = (proc > 0.5).float()

        dil_k = int(self.mask_dilate_kernel_size_var.get())
        if dil_k > 0:
            proc = apply_mask_dilation(proc, dil_k, use_gpu=use_gpu_ops)

        blur_k = int(self.mask_blur_kernel_size_var.get())
        if blur_k > 0:
            proc = apply_gaussian_blur(proc, blur_k, use_gpu=use_gpu_ops)

        shadow_len = int(self.shadow_length_px_var.get())
        if shadow_len > 0:
            proc = apply_shadow_blur(
                proc,
                base_length_px=shadow_len,
                curve=float(self.shadow_curve_var.get()),
                motion_gain=float(self.shadow_motion_gain_var.get()),
                motion_deadzone_px=float(self.shadow_motion_deadzone_px_var.get()),
                motion_max_px=float(self.shadow_motion_max_px_var.get()),
                motion_chain_enabled=bool(self.shadow_motion_chain_enabled_var.get()),
                area_min_px=float(self.shadow_area_min_px_var.get()),
                area_max_px=float(self.shadow_area_max_px_var.get()),
                area_reset_ratio=_area_reset_pct_to_ratio(float(self.shadow_area_reset_pct_var.get())),
                area_reset_abs_px=float(self.shadow_area_reset_abs_px_var.get()),
                component_merge_y_tol_px=int(self.shadow_component_merge_y_tol_px_var.get()),
                alpha_down=float(self.shadow_alpha_down_var.get()),
                width_adaptive=bool(self.shadow_width_adaptive_var.get()),
                use_gpu=use_gpu_ops,
                state=shadow_state,
                border_tolerance_px=2,
                width_ref_px=20.0,
                width_power=1.0,
            )
        return torch.clamp(proc, 0.0, 1.0)

    def _render_mask_tensor(self, mask: torch.Tensor) -> None:
        img = mask.detach().cpu().squeeze(0).squeeze(0).numpy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([u8, u8, u8], axis=2)
        pil = Image.fromarray(rgb, mode="RGB")

        scale = _parse_percent(self.preview_size_var.get(), fallback=100.0)
        if abs(scale - 100.0) > 1e-6:
            w = max(1, int(round(pil.width * scale / 100.0)))
            h = max(1, int(round(pil.height * scale / 100.0)))
            pil = pil.resize((w, h), Image.NEAREST)

        self._tk_preview_img = ImageTk.PhotoImage(pil)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self._tk_preview_img, anchor="nw")
        self.preview_canvas.configure(scrollregion=(0, 0, pil.width, pil.height))

    def _update_preview(self) -> None:
        self._refresh_after_id = None
        if self._reader is None or self._num_frames <= 0:
            return

        frame_idx = max(0, min(self._num_frames - 1, int(self.frame_idx_var.get())))
        self.frame_idx_var.set(float(frame_idx))
        self.frame_label_var.set(f"Frame: {frame_idx + 1} / {self._num_frames}")

        use_gpu_ops = bool(self.use_gpu_mask_ops_var.get() and torch.cuda.is_available())
        if bool(self.use_gpu_mask_ops_var.get()) and not use_gpu_ops:
            self.status_var.set("GPU mask ops requested but CUDA unavailable. Using CPU.")

        device = torch.device("cuda" if use_gpu_ops else "cpu")
        video_path = self._files[self._current_file_index()] if self._files else ""
        source = self.preview_source_var.get().strip()
        warmup = max(0, int(self.warmup_frames_var.get()))
        temporal_active = False

        try:
            if source == "Mask Original":
                self._shadow_temporal_cache = None
                mask = self._get_mask_tensor(frame_idx, device=device)
            else:
                key = self._params_key(video_path, use_gpu_ops)
                shadow_len = int(self.shadow_length_px_var.get())
                use_temporal = (
                    warmup > 0
                    and shadow_len > 0
                    and bool(self.shadow_motion_chain_enabled_var.get())
                )
                temporal_active = bool(use_temporal)

                can_step_forward = (
                    use_temporal
                    and self._shadow_temporal_cache is not None
                    and self._shadow_temporal_cache.get("key") == key
                    and int(self._shadow_temporal_cache.get("frame_idx", -10**9)) + 1 == frame_idx
                    and isinstance(self._shadow_temporal_cache.get("state"), dict)
                )

                if can_step_forward:
                    state = self._shadow_temporal_cache["state"]
                    current = self._get_mask_tensor(frame_idx, device=device)
                    mask = self._preprocess_mask(current, use_gpu_ops, state)
                elif use_temporal:
                    start = max(0, frame_idx - warmup)
                    state = {"prev_components": []}
                    mask = None
                    for wi in range(start, frame_idx + 1):
                        src = self._get_mask_tensor(wi, device=device)
                        mask = self._preprocess_mask(src, use_gpu_ops, state)
                    if mask is None:
                        raise RuntimeError("Warmup rebuild failed.")
                else:
                    state = None
                    current = self._get_mask_tensor(frame_idx, device=device)
                    mask = self._preprocess_mask(current, use_gpu_ops, None)

                if use_temporal and state is not None:
                    self._shadow_temporal_cache = {
                        "key": key,
                        "frame_idx": int(frame_idx),
                        "state": state,
                    }
                else:
                    self._shadow_temporal_cache = None

            self._render_mask_tensor(mask)
            if source == "Mask Processed":
                self.status_var.set(
                    f"Processed preview | frame={frame_idx + 1}/{self._num_frames} | "
                    f"gpu_mask_ops={'on' if use_gpu_ops else 'off'} | "
                    f"temporal={'on' if temporal_active else 'off'} | warmup={warmup}"
                )
            else:
                self.status_var.set(
                    f"Original preview | frame={frame_idx + 1}/{self._num_frames}"
                )
        except Exception as e:
            self.status_var.set(f"Preview error: {e}")

    def _set_var_from_settings(
        self,
        data: Dict[str, Any],
        key: str,
        var: tk.Variable,
        cast: Optional[Any] = None,
    ) -> None:
        if key not in data:
            return
        try:
            raw = data[key]
            value = cast(raw) if cast is not None else raw
            var.set(value)
        except Exception:
            pass

    def _load_settings(self) -> None:
        if not os.path.isfile(self._settings_path):
            return
        try:
            with open(self._settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
        except Exception:
            return

        self._set_var_from_settings(data, "input_folder", self.input_folder_var, str)
        self._set_var_from_settings(data, "input_glob", self.input_glob_var, str)
        src = str(data.get("preview_source", "")).strip()
        if src in self.PREVIEW_SOURCES:
            self.preview_source_var.set(src)
        size = str(data.get("preview_size", "")).strip()
        if size in self.PREVIEW_SIZES:
            self.preview_size_var.set(size)

        self._set_var_from_settings(data, "mask_binarize_threshold", self.mask_binarize_threshold_var, float)
        self._set_var_from_settings(data, "mask_dilate_kernel_size", self.mask_dilate_kernel_size_var, int)
        self._set_var_from_settings(data, "mask_blur_kernel_size", self.mask_blur_kernel_size_var, int)
        self._set_var_from_settings(data, "shadow_length_px", self.shadow_length_px_var, int)
        self._set_var_from_settings(data, "shadow_curve", self.shadow_curve_var, float)
        self._set_var_from_settings(data, "shadow_width_adaptive", self.shadow_width_adaptive_var, _parse_bool)
        self._set_var_from_settings(data, "use_gpu_mask_ops", self.use_gpu_mask_ops_var, _parse_bool)
        self._set_var_from_settings(data, "warmup_frames", self.warmup_frames_var, int)

        clip_name = str(data.get("current_clip_basename", "")).strip()
        if clip_name:
            self._startup_restore_clip_name = clip_name
        try:
            if "current_frame_idx" in data:
                self._startup_restore_frame_idx = int(data["current_frame_idx"])
        except Exception:
            self._startup_restore_frame_idx = None

    def _build_settings_dict(self) -> Dict[str, Any]:
        clip_name = self.current_file_var.get().strip()
        if not clip_name and self._files:
            clip_name = os.path.basename(self._files[self._current_file_index()])
        return {
            "input_folder": self.input_folder_var.get().strip(),
            "input_glob": self.input_glob_var.get().strip(),
            "preview_source": self.preview_source_var.get().strip(),
            "preview_size": self.preview_size_var.get().strip(),
            "mask_binarize_threshold": float(self.mask_binarize_threshold_var.get()),
            "mask_dilate_kernel_size": int(self.mask_dilate_kernel_size_var.get()),
            "mask_blur_kernel_size": int(self.mask_blur_kernel_size_var.get()),
            "shadow_length_px": int(self.shadow_length_px_var.get()),
            "shadow_curve": float(self.shadow_curve_var.get()),
            "shadow_width_adaptive": bool(self.shadow_width_adaptive_var.get()),
            "use_gpu_mask_ops": bool(self.use_gpu_mask_ops_var.get()),
            "warmup_frames": int(self.warmup_frames_var.get()),
            "current_clip_basename": clip_name,
            "current_frame_idx": int(self.frame_idx_var.get()),
        }

    def _save_settings(self) -> None:
        data = self._build_settings_dict()
        tmp_path = f"{self._settings_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, sort_keys=True)
        os.replace(tmp_path, self._settings_path)

    def _on_close(self) -> None:
        self._stop_playback(silent=True)
        try:
            self._save_settings()
        except Exception as e:
            # Keep close robust even if settings save fails.
            self.status_var.set(f"Settings save error: {e}")
        self.destroy()


def main() -> int:
    app = MaskForMergePreviewGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
