from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from dependency.repo_paths import runner_path, utilities_path


def _parse_chunk_limit_list(gui: Any, raw: str, label: str) -> str:
    values: list[str] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(str(gui._parse_inpaint_positive_int(part, label)))
    if not values:
        raise ValueError(f"{label} requires at least one positive integer.")
    return ",".join(values)


def _parse_resolution_scale(raw: str) -> str:
    text = str(raw or "100%").strip().replace("%", "")
    try:
        value = float(text)
    except Exception as exc:
        raise ValueError("Res / Max Res must be one of 100%, 90%, 80%, 70%, 60%, 50%.") from exc
    if value > 1.0:
        value = value / 100.0
    allowed = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    nearest = min(allowed, key=lambda item: abs(item - value))
    if abs(nearest - value) > 0.001:
        raise ValueError("Res / Max Res must be one of 100%, 90%, 80%, 70%, 60%, 50%.")
    return f"{nearest:.2f}"


def build_depth_runner_payload(gui: Any) -> tuple[list[str], dict[str, str], str]:
    final_upscale = "False"
    depth_codec = gui._normalize_ffmpeg_codec(
        gui.depth_codec_var.get(),
        gui.DEFAULT_SCENE_CODEC,
    )
    gui.depth_codec_var.set(depth_codec)
    depth_runtime_mode = gui._normalize_depth_runtime_mode(
        gui.depth_runtime_mode_var.get()
    )
    gui.depth_runtime_mode_var.set(depth_runtime_mode)
    depth_worker_script = gui._resolve_depth_worker_script(depth_runtime_mode)
    gui.depth_worker_script_var.set(depth_worker_script)
    scene_strip_pad_top, scene_strip_pad_bottom = gui._depth_scene_strip_pad()
    scale_factor = gui._get_depth_scale_factor()

    env_updates: dict[str, str] = {
        "PYTHON": sys.executable,
        "WORKER_SCRIPT": depth_worker_script,
        "INPUT_DIR": gui.depth_input_var.get().strip(),
        "OUTPUT_DIR": gui.depth_output_var.get().strip(),
        "GLOB": gui.depth_glob_var.get().strip() or "*.mp4",
        "WINDOW_SIZE": gui.depth_chunk_size_var.get().strip(),
        "OVERLAP": gui.depth_overlap_var.get().strip(),
        "INFERENCE_STEPS": gui.depth_inference_steps_var.get().strip(),
        "GUIDANCE_SCALE": gui.depth_guidance_scale_var.get().strip() or "1.0",
        "SEED": gui.depth_seed_var.get().strip(),
        "CPU_OFFLOAD_MODE": gui.depth_cpu_offload_var.get().strip(),
        "DECODE_CHUNK_SIZE": gui.depth_decode_chunk_size_var.get().strip() or "2",
        "DEBUG_MEM": "True" if gui.depth_debug_mem_var.get() else "False",
        "FINAL_UPSCALE": final_upscale,
        "SCALE_FACTOR": f"{scale_factor:.2f}",
        "RESTART_EVERY": gui.depth_restart_every_var.get().strip() or "100",
        "PAD_ALIGN_BOTTOM": "True",
        "SCENE_STRIP_PAD_TOP": str(scene_strip_pad_top),
        "SCENE_STRIP_PAD_BOTTOM": str(scene_strip_pad_bottom),
        "FFMPEG_CODEC": depth_codec,
        "STOP_MARKER": os.path.join(
            gui.depth_output_var.get().strip() or "./work/depthmap",
            ".stop_after_current",
        ),
        "RETRY_POLICY_JSON": gui._build_retry_policy_json(
            gui.depth_retry_policy_vars,
            gui.depth_cpu_offload_var.get().strip() or "model",
        ),
    }

    cmd = ["bash", str(runner_path("run_depthcrafter_nogui_batch.sh"))]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview


def build_splat_runner_payload(gui: Any) -> tuple[list[str], dict[str, str], str]:
    layout_ui = gui.splat_layout_var.get().strip()
    layout_cli = {
        "Single Warp": "single_warp",
        "Dual": "dual",
        "Quad": "quad",
    }.get(layout_ui, "single_warp")
    codec_value = gui._normalize_ffmpeg_codec(
        gui.splat_codec_var.get(),
        gui.DEFAULT_SCENE_CODEC,
    )
    gui.splat_codec_var.set(codec_value)
    workers_raw = gui.splat_workers_var.get().strip()
    try:
        workers = max(1, int(workers_raw))
    except Exception:
        workers = 8
        gui.splat_workers_var.set(str(workers))
    auto_conv_ui = gui.splat_auto_convergence_var.get().strip()
    auto_conv_cli = {
        "Off": "Off",
        "Min Borders": "MinBorders",
        "Average": "Average",
        "Peak": "Peak",
        "Hybrid": "Hybrid",
    }.get(auto_conv_ui, "MinBorders")
    replace_mask_enabled = True

    env_updates: dict[str, str] = {
        "PYTHON": sys.executable,
        "RUNNER": str(runner_path("batch_splatting_runner.py")),
        "INPUT_SOURCE_CLIPS": gui.splat_input_clips_var.get().strip(),
        "INPUT_DEPTH_MAPS": gui.splat_input_depth_var.get().strip(),
        "OUTPUT_SPLATTED": gui.splat_output_var.get().strip(),
        "MASK_OUTPUT": gui.splat_mask_output_var.get().strip(),
        "FULL_RES_BATCH_SIZE": gui.splat_batch_size_var.get().strip() or "50",
        "WORKERS": str(workers),
        "DISPARITY": gui.splat_disparity_var.get().strip() or "20",
        "OUTPUT_LAYOUT": layout_cli,
        "AUTO_CONVERGENCE_MODE": auto_conv_cli,
        "DILATE_X": gui.splat_dilate_x_var.get().strip() or "3",
        "DILATE_Y": gui.splat_dilate_y_var.get().strip() or "1.5",
        "BLUR_X": gui.splat_blur_x_var.get().strip() or "0",
        "BLUR_Y": gui.splat_blur_y_var.get().strip() or "0",
        "DILATE_LEFT": gui.splat_dilate_left_var.get().strip() or "1",
        "BLUR_BALANCE": gui.splat_blur_balance_var.get().strip() or "0.5",
        "GAMMA": gui.splat_gamma_var.get().strip() or "1",
        "CONVERGENCE": gui.splat_convergence_var.get().strip() or "50",
        "STAIR_SMOOTH": "1" if gui.splat_stair_smooth_var.get() else "0",
        "STAIR_SMOOTH_KERNEL": gui.splat_stair_kernel_var.get().strip() or "3",
        "STAIR_SMOOTH_X_OFF": gui.splat_stair_x_off_var.get().strip() or "2",
        "STAIR_SMOOTH_STRIP": gui.splat_stair_strip_var.get().strip() or "4",
        "STAIR_SMOOTH_STRENGTH": gui.splat_stair_strength_var.get().strip() or "1",
        "USE_REPLACE_MASK": "1" if replace_mask_enabled else "0",
        "REPLACE_MASK_SCALE": gui.splat_replace_mask_scale_var.get().strip() or "1",
        "REPLACE_MASK_MIN": gui.splat_replace_mask_min_var.get().strip() or "1",
        "REPLACE_MASK_MAX": gui.splat_replace_mask_max_var.get().strip() or "32",
        "REPLACE_MASK_GAP": gui.splat_replace_mask_gap_var.get().strip() or "0",
        "REPLACE_MASK_EDGE": "1" if gui.splat_replace_mask_edge_var.get() else "0",
        "ENABLE_FULL_RES": "True",
        "ENABLE_LOW_RES": "False",
        "PROCESS_LENGTH": "-1",
        "ADD_BORDERS": "False",
        "REPLACE_MASK_CODEC": "ffv1",
        "FFMPEG_CODEC": codec_value,
        "ENCODER_MODE": gui._current_global_encoder_mode(),
        "FFMPEG_EXTRA_ARGS": gui._current_global_ffmpeg_extra_args(),
        "STOP_MARKER": os.path.join(
            gui.splat_output_var.get().strip() or "./work/splat",
            ".stop_after_current",
        ),
    }

    launcher_name = (
        str(runner_path("run_splatting_runner_parallel.sh"))
        if workers >= 2
        else str(runner_path("run_splatting_runner.sh"))
    )
    cmd = ["bash", launcher_name]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview


def build_inpaint_runner_payload(gui: Any) -> tuple[list[str], dict[str, str], str]:
    input_dir = gui.inpaint_input_var.get().strip()
    output_dir = gui.inpaint_output_var.get().strip()
    mask_dir = gui.inpaint_mask_var.get().strip()
    work_dir = gui.work_folder_var.get().strip() or "./work"
    sharp_base = os.path.normpath(work_dir)
    sharp_csv_path = gui.inpaint_sharpness_csv_var.get().strip()
    use_sharpness_csv = bool(gui.inpaint_use_sharpness_csv_var.get())
    codec_value = gui._normalize_ffmpeg_codec(
        gui.inpaint_codec_var.get(),
        gui.DEFAULT_SCENE_CODEC,
    )
    gui.inpaint_codec_var.set(codec_value)
    chunk_size = gui._parse_inpaint_positive_int(
        gui._get_inpaint_chunk_manual_value(),
        "Chunk Size",
    )
    tile1_max_size = _parse_chunk_limit_list(
        gui,
        gui.inpaint_tile1_max_size_var.get(),
        "Tile 1 Max Size",
    )
    tile2_max_size = _parse_chunk_limit_list(
        gui,
        gui.inpaint_tile2_max_size_var.get(),
        "Tile 2 Max Size",
    )
    overlap = gui._parse_inpaint_nonnegative_int(
        gui.inpaint_overlap_var.get(),
        "Overlap",
    )
    tail_pad = gui._parse_inpaint_nonnegative_int(
        gui.inpaint_tail_pad_var.get(),
        "TailPad",
    )
    input_bias = gui._parse_inpaint_bounded_float(
        gui.inpaint_input_bias_var.get(),
        "Input Bias",
        min_value=0.0,
        max_value=1.0,
    )
    fixed_steps = gui._parse_inpaint_positive_float(
        gui.inpaint_inference_steps_var.get(),
        "Inference steps",
    )
    resolution_scale = _parse_resolution_scale(
        gui.inpaint_resolution_limit_var.get()
    )
    dynamic_steps5 = gui._parse_inpaint_positive_int(
        gui.inpaint_dynamic_visible_chunk_steps5_var.get(),
        "Chunk @ 3.0 steps",
    )
    dynamic_steps6 = gui._parse_inpaint_positive_int(
        gui.inpaint_dynamic_visible_chunk_steps6_var.get(),
        "Chunk @ 4.0 steps",
    )
    dynamic_steps7 = gui._parse_inpaint_positive_int(
        gui.inpaint_dynamic_visible_chunk_steps7_var.get(),
        "Chunk @ 5.0 steps",
    )
    dynamic_steps8_plus = gui._parse_inpaint_positive_int(
        gui.inpaint_dynamic_visible_chunk_steps8_plus_var.get(),
        "Chunk @ 6.0+ steps",
    )
    dynamic_hold_divisor = gui._parse_inpaint_positive_float(
        gui.inpaint_dynamic_hold_divisor_var.get(),
        "Static Mask Divisor",
    )
    tile_mode = gui.inpaint_tile_mode_var.get().strip() or "1 and 2"
    if tile_mode not in {"1", "2", "1 and 2"}:
        raise ValueError("Tile must be one of: 1, 2, 1 and 2.")

    env_updates: dict[str, str] = {
        "PYTHON": sys.executable,
        "RUNNER": str(runner_path("batch_inpainting_runner.py")),
        "INPUT_DIR": input_dir,
        "OUTPUT_DIR": output_dir,
        "GLOB": "*.mp4",
        "REPLACE_MASK_FOLDER": mask_dir,
        "USE_REPLACE_MASK": "1",
        "OFFLOAD_TYPE": gui.inpaint_cpu_offload_var.get().strip() or "model",
        "CHUNK_SIZE": str(chunk_size),
        "ENABLE_DYNAMIC_CHUNK": "1" if bool(gui.inpaint_dynamic_chunk_var.get()) else "0",
        "TILE_MODE": tile_mode,
        "TILE1_MAX_SIZE": str(tile1_max_size),
        "TILE2_MAX_SIZE": str(tile2_max_size),
        "DYNAMIC_VISIBLE_CHUNK_STEPS5": str(dynamic_steps5),
        "DYNAMIC_VISIBLE_CHUNK_STEPS6": str(dynamic_steps6),
        "DYNAMIC_VISIBLE_CHUNK_STEPS7": str(dynamic_steps7),
        "DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS": str(dynamic_steps8_plus),
        "DYNAMIC_HOLD_DIVISOR": str(dynamic_hold_divisor),
        "OVERLAP": str(overlap),
        "TAIL_PAD": str(tail_pad),
        "ORIGINAL_INPUT_BLEND_STRENGTH": str(input_bias),
        "OUTPUT_CODEC": codec_value,
        "OUTPUT_ENCODING_MODE": gui._current_global_encoder_mode(),
        "OUTPUT_EXTRA_ARGS": gui._current_global_ffmpeg_extra_args(),
        "NO_SHARPNESS_CSV": "0" if use_sharpness_csv else "1",
        "DYNAMIC_RESOLUTION": "1" if bool(gui.inpaint_dynamic_resolution_var.get()) else "0",
        "RESOLUTION_SCALE": resolution_scale,
        "SHARPNESS_BASE": sharp_base,
        "SHARPNESS_CSV_PATH": sharp_csv_path,
        "FIXED_STEPS": str(fixed_steps),
        "ENABLE_POST_INPAINTING_BLEND": "0",
        "DISABLE_COLOR_TRANSFER": "1",
        "STOP_MARKER": os.path.join(
            output_dir or os.path.join(work_dir, gui.STANDARD_SUBDIRS["inpaint"]),
            ".stop_after_current",
        ),
        "RETRY_POLICY_JSON": gui._build_retry_policy_json(
            gui.inpaint_retry_policy_vars,
            gui.inpaint_cpu_offload_var.get().strip() or "model",
        ),
    }

    cmd = ["bash", str(runner_path("run_inpainting_runner.sh"))]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview


def build_inpaint_sharpen_runner_payload(
    gui: Any,
) -> tuple[list[str], dict[str, str], str]:
    output_dir = gui.inpaint_sharpen_output_var.get().strip()
    env_updates: dict[str, str] = {
        "PYTHON": sys.executable,
        "RUNNER": str(runner_path("batch_inpaint_sharpen_runner.py")),
        "INPUT_DIR": gui.inpaint_output_var.get().strip(),
        "MASK_DIR": gui.inpaint_mask_var.get().strip(),
        "OUTPUT_DIR": output_dir,
        "SHARPNESS_CSV_PATH": gui.inpaint_sharpness_csv_var.get().strip(),
        "GLOB": "*.mp4",
        "WORKERS": gui.inpaint_sharpen_workers_var.get().strip() or "19",
        "STOP_MARKER": os.path.join(
            output_dir
            or os.path.join(
                gui.work_folder_var.get().strip() or "./work",
                gui.STANDARD_SUBDIRS["inpaint_sharpen"],
            ),
            ".stop_after_current",
        ),
        "SKIP_EXISTING": "1",
        "OUTPUT_CODEC": "libx264",
        "OUTPUT_PRESET": "fast",
        "OUTPUT_PIX_FMT": "yuv444p",
        "OUTPUT_CRF": "0",
        "OUTPUT_EXTRA_ARGS": "",
    }
    cmd = ["bash", str(runner_path("run_inpaint_sharpen_runner.sh"))]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview


def build_merge_runner_payload(gui: Any) -> tuple[list[str], dict[str, str], str]:
    output_dir = gui.merge_output_var.get().strip()
    workers = gui._get_merge_worker_count()
    use_parallel = workers >= 2
    ct_mode = gui.merge_ct_auto_mode_var.get().strip() or "CSV Blend"
    preferred_inpainted_dir = gui._preferred_inpainted_dir_for_consumers()
    output_format_ui = gui.merge_output_format_var.get().strip()
    output_format_runner = (
        "Half SBS (Left-Right)"
        if output_format_ui == "Half SBS"
        else "Full SBS (Left-Right)"
    )
    codec_value = gui._normalize_ffmpeg_codec(
        gui.merge_codec_var.get(),
        gui.DEFAULT_SCENE_CODEC,
    )
    gui.merge_codec_var.set(codec_value)
    env_updates: dict[str, str] = {
        "PYTHON": sys.executable,
        "RUNNER": str(
            runner_path(
                "merging_nogui_batch_parallel.py"
                if use_parallel
                else "merging_nogui_batch.py"
            )
        ),
        "INPAINTED_FOLDER": gui.merge_inpainted_var.get().strip(),
        "PREFERRED_INPAINTED_FOLDER": preferred_inpainted_dir,
        "SPLATTED_FOLDER": gui.merge_splatted_var.get().strip(),
        "ORIGINAL_FOLDER": gui.merge_original_var.get().strip(),
        "OUTPUT_FOLDER": output_dir,
        "REPLACE_MASK_FOLDER": gui.merge_replace_mask_var.get().strip(),
        "PREPROCESSED_MASK_FOLDER": gui.merge_mask_formerge_var.get().strip(),
        "CT_PRESET": gui.merge_ct_preset_var.get().strip() or "1",
        "CT_AUTO_MODE": ct_mode,
        "CT_CSV_BLEND_PATH": gui.merge_autoct_csv_var.get().strip(),
        "OUTPUT_FORMAT": output_format_runner,
        "CHUNK_SIZE": gui.merge_chunks_var.get().strip() or "20",
        "USE_GPU": "1" if gui.merge_use_gpu_var.get() else "0",
        "CT_EXCLUDE_BLACK_IN_TARGET": "1"
        if gui.merge_ct_exclude_black_var.get()
        else "0",
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
        "ENCODING_MODE": gui._current_global_encoder_mode(),
        "FFMPEG_EXTRA_ARGS": gui._current_global_ffmpeg_extra_args(),
        "RESTART_EVERY": "1",
    }
    if use_parallel:
        env_updates["WORKERS"] = str(workers)
    cmd = [
        "bash",
        str(
            runner_path(
                "run_merging_nogui_batch_parallel.sh"
                if use_parallel
                else "run_merging_nogui_batch.sh"
            )
        ),
    ]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview


def build_mask_formerge_runner_payload(
    gui: Any,
) -> tuple[list[str], dict[str, str], str]:
    mask_workers_raw = gui.merge_mask_formerge_workers_var.get().strip()
    try:
        mask_workers = max(1, int(mask_workers_raw))
    except Exception:
        autoct_raw = gui.merge_autoct_workers_var.get().strip()
        try:
            mask_workers = max(1, int(autoct_raw))
        except Exception:
            mask_workers = 8
    if str(mask_workers) != mask_workers_raw:
        gui.merge_mask_formerge_workers_var.set(str(mask_workers))
    env_updates: dict[str, str] = {
        "PYTHON": sys.executable,
        "RUNNER": str(runner_path("mask_formerge_nogui.py")),
        "REPLACE_MASK_FOLDER": gui.merge_replace_mask_var.get().strip(),
        "OUTPUT_FOLDER": gui.merge_mask_formerge_var.get().strip(),
        "INPUT_GLOB": "*_replace_mask.*",
        "WORKERS": str(mask_workers),
        "CHUNK_SIZE": gui.merge_chunks_var.get().strip() or "20",
        "USE_GPU": "1" if bool(gui.merge_use_gpu_var.get()) else "0",
        "MASK_BINARIZE_THRESHOLD": gui.merge_mask_binarize_var.get().strip() or "0.5",
        "MASK_DILATE_KERNEL_SIZE": gui.merge_mask_dilate_var.get().strip() or "2",
        "MASK_BLUR_KERNEL_SIZE": gui.merge_mask_blur_var.get().strip() or "4",
        "SHADOW_LENGTH_PX": gui.merge_shadow_length_var.get().strip() or "25",
        "SHADOW_CURVE": gui.merge_shadow_curve_var.get().strip() or "0",
        "SHADOW_WIDTH_ADAPTIVE": "1"
        if bool(gui.merge_dynamic_shadow_width_var.get())
        else "0",
        "SKIP_EXISTING": "1",
        "STOP_MARKER": os.path.join(
            gui.merge_mask_formerge_var.get().strip() or "./work/mask_for_merge",
            ".stop_after_current",
        ),
    }
    cmd = ["/usr/bin/env", "bash", str(runner_path("run_mask_formerge_nogui.sh"))]
    preview = (
        " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items())
        + " "
        + " ".join(shlex.quote(x) for x in cmd)
    )
    return cmd, env_updates, preview


def join_layout_for_seg_mono(gui: Any) -> str:
    return (
        "half_sbs"
        if gui.merge_output_format_var.get().strip() == "Half SBS"
        else "full_sbs"
    )


def build_join_runner_payload(gui: Any) -> tuple[list[str], dict[str, str], str]:
    out_path = gui.join_output_var.get().strip()
    quality_value = gui.join_crf_var.get().strip() or "12"
    encoder = gui._normalize_ffmpeg_codec(gui.join_encoder_var.get(), "hevc_nvenc")
    gui.join_encoder_var.set(encoder)
    preset = gui.join_preset_var.get().strip() or "p7"
    pix_fmt = gui.join_pix_fmt_var.get().strip() or "yuv420p"
    extra_args = gui.join_extra_args_var.get().strip()
    quality_flag = gui._join_quality_flag()

    env_updates: dict[str, str] = {
        "DIR_SBS": gui.join_input_var.get().strip(),
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
        "VF": "pad=iw:max(ih\\,1080):0:(max(ih\\,1080)-ih)/2:black,crop=iw:1080:0:(ih-1080)/2",
    }
    cmd = ["bash", str(utilities_path("Rejoin_HEVC_NVENC.sh"))]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview


def default_remux_output_path(gui: Any) -> str:
    source_path = gui.scene_input_var.get().strip()
    join_path = gui.join_output_var.get().strip()
    if join_path:
        out_dir = Path(join_path).resolve().parent
    else:
        work_dir = Path(gui.work_folder_var.get().strip() or "./work").resolve()
        out_dir = work_dir / gui.STANDARD_SUBDIRS["join"]
    src_stem = Path(source_path).stem if source_path else "source"
    return str((out_dir / f"{src_stem}_3D_remux.mkv").resolve())


def build_join_remux_payload(
    gui: Any,
) -> tuple[list[str], dict[str, str], str, str]:
    source_path = gui.scene_input_var.get().strip()
    video_3d_path = gui.join_output_var.get().strip()
    out_path = default_remux_output_path(gui)
    env_updates: dict[str, str] = {
        "SOURCE_FILE": source_path,
        "VIDEO_3D_FILE": video_3d_path,
        "OUT_FILE": out_path,
        "MKVMERGE_BIN": "mkvmerge",
        "OVERWRITE": "1",
    }
    cmd = ["bash", str(utilities_path("remux_replace_video_mkvtoolnix.sh"))]
    preview = " ".join(
        [f"{k}={shlex.quote(str(v))}" for k, v in env_updates.items()]
        + [shlex.quote(x) for x in cmd]
    )
    return cmd, env_updates, preview, out_path


def build_join_prepare_mono_cmd(gui: Any) -> list[str]:
    merge_codec = gui._normalize_ffmpeg_codec(
        gui.merge_codec_var.get(),
        gui.scene_codec_var.get().strip() or gui.DEFAULT_SCENE_CODEC,
    )
    gui.merge_codec_var.set(merge_codec)
    return [
        sys.executable,
        str(utilities_path("prepare_seg_mono_to_sbs.py")),
        "--seg-mono-dir",
        str(Path(gui.join_seg_mono_var.get().strip()).resolve()),
        "--sbs-dir",
        str(Path(gui.join_input_var.get().strip()).resolve()),
        "--layout",
        join_layout_for_seg_mono(gui),
        "--ffmpeg-bin",
        "ffmpeg",
        "--ffprobe-bin",
        "ffprobe",
        "--codec",
        merge_codec,
        "--encoder-mode",
        gui._current_global_encoder_mode(),
        "--extra-ffmpeg-args",
        gui._current_global_ffmpeg_extra_args(),
    ]
