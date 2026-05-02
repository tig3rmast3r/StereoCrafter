#!/usr/bin/env python3
"""
Analyze replace-mask videos and report the maximum "static hold" duration per file.

When a Pipeline Master work folder is available, the report can also resolve the
effective inpaint chunk plan per scene by combining:
- the content-aware static-hold result from this utility
- the work folder's sharpness.csv
- the work folder's config_pipeline_master_gui.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from dependency.mask_static_hold import (
    DEFAULT_ANCHOR_OVERLAP_MIN_RATIO,
    DEFAULT_BORDER_TOLERANCE_PX,
    DEFAULT_COMPONENT_MERGE_Y_TOL_PX,
    DEFAULT_CONTENT_EDGE_DELTA_THRESHOLD,
    DEFAULT_CONTENT_GRAY_DELTA_THRESHOLD,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_MIN_CONTENT_ROI_PIXELS,
    DEFAULT_ROI_MASK_DILATE_ITER,
    DEFAULT_ROI_MASK_DILATE_K,
    DEFAULT_THRESHOLD_U8,
    analyze_mask_video,
    analyze_mask_video_with_warped_content,
)


DEFAULT_GLOB_PATTERNS = ("*.mkv", "*.mp4", "*.mov", "*.avi", "*.m4v", "*.webm")
DEFAULT_OUT_CSV = "mask_static_hold_report.csv"
DEFAULT_WORK_CONFIG_NAME = "config_pipeline_master_gui.json"
DEFAULT_SHARPNESS_CSV_NAME = "sharpness.csv"
DEFAULT_WARPED_RELATIVE_DIRS = (
    os.path.join("splat", "hires"),
    "splat",
    "hires",
)

# Keep these defaults aligned with the live inpaint planner in inpainting_gui.py.
DEFAULT_INPAINT_CHUNK_SIZE = 22
DEFAULT_INPAINT_TILE_MODE = "1 and 2"
DEFAULT_INPAINT_TILE1_MAX_SIZE = 22
DEFAULT_INPAINT_TILE2_MAX_SIZE = 55
DEFAULT_INPAINT_FIXED_STEPS = 8.0
DEFAULT_INPAINT_OVERLAP = 2
DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS5 = 38
DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS6 = 26
DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS7 = 18
DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS = 14
DEFAULT_DYNAMIC_STATIC_MASK_DIVISOR = 2.0

PLANNER_FIELDNAMES = [
    "planner_status",
    "planner_warning",
    "planner_mode",
    "planner_dynamic_chunk_enabled",
    "planner_use_sharpness_csv",
    "planner_steps_source",
    "planner_sharpness_key",
    "planner_sharpness_raw",
    "planner_chunk_size_manual",
    "planner_overlap",
    "planner_tile_mode",
    "planner_tile1_max_size",
    "planner_tile2_max_size",
    "planner_dynamic_visible_chunk_steps5",
    "planner_dynamic_visible_chunk_steps6",
    "planner_dynamic_visible_chunk_steps7",
    "planner_dynamic_visible_chunk_steps8_plus",
    "planner_static_mask_divisor",
    "planner_hold_source",
    "planner_max_hold_frames_used",
    "planner_effective_steps",
    "planner_model_steps",
    "planner_visible_chunk_from_steps",
    "planner_hold_floor_visible",
    "planner_visible_chunk_after_hold",
    "planner_visible_chunk_final",
    "planner_processed_chunk_final",
    "planner_selected_tile",
    "planner_clamp_reason",
]


def _blank_planner_row(status: str = "", warning: str = "") -> Dict[str, object]:
    row: Dict[str, object] = {key: "" for key in PLANNER_FIELDNAMES}
    if status:
        row["planner_status"] = status
    if warning:
        row["planner_warning"] = warning
    return row


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


def _mask_core_name(path: str) -> str:
    stem = Path(path).stem
    suffix = "_replace_mask"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _find_warped_for_mask(
    *,
    mask_path: str,
    warped_dir: str,
    recursive: bool,
) -> Optional[str]:
    base_dir = os.path.abspath(warped_dir)
    if not os.path.isdir(base_dir):
        return None
    core = _mask_core_name(mask_path)
    patt = (
        os.path.join(base_dir, "**", core + ".*")
        if recursive
        else os.path.join(base_dir, core + ".*")
    )
    matches = sorted(
        p for p in glob.glob(patt, recursive=recursive) if os.path.isfile(p)
    )
    return os.path.abspath(matches[0]) if matches else None


def _looks_like_work_dir(path: str) -> bool:
    root = os.path.abspath(path)
    return (
        os.path.isdir(os.path.join(root, "mask"))
        or os.path.isfile(os.path.join(root, DEFAULT_WORK_CONFIG_NAME))
        or os.path.isfile(os.path.join(root, DEFAULT_SHARPNESS_CSV_NAME))
        or os.path.isdir(os.path.join(root, "splat"))
    )


def _resolve_input_root_and_work_dir(
    input_path: str,
    explicit_work_dir: str,
) -> Tuple[str, str]:
    src = os.path.abspath(input_path)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Input path not found: {src}")

    auto_work_dir = ""
    resolved_input_root = src
    if os.path.isfile(src):
        parent = os.path.dirname(src)
        if os.path.basename(parent).lower() == "mask":
            auto_work_dir = os.path.dirname(parent)
        elif _looks_like_work_dir(parent):
            auto_work_dir = parent
    elif os.path.isdir(src):
        mask_subdir = os.path.join(src, "mask")
        if os.path.isdir(mask_subdir):
            resolved_input_root = mask_subdir
            auto_work_dir = src
        elif os.path.basename(src).lower() == "mask":
            auto_work_dir = os.path.dirname(src)
        elif _looks_like_work_dir(src):
            auto_work_dir = src

    if explicit_work_dir:
        return resolved_input_root, os.path.abspath(explicit_work_dir)
    return resolved_input_root, auto_work_dir


def _resolve_auto_warped_dir(work_dir: str) -> str:
    if not work_dir:
        return ""
    root = os.path.abspath(work_dir)
    for rel in DEFAULT_WARPED_RELATIVE_DIRS:
        cand = os.path.join(root, rel)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return ""


def _norm_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _parse_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return int(default)


def _parse_nonnegative_int(value: object, default: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
        if parsed >= 0:
            return parsed
    except Exception:
        pass
    return int(default)


def _parse_positive_float(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
        if math.isfinite(parsed) and parsed > 0.0:
            return parsed
    except Exception:
        pass
    return float(default)


def _normalize_tile_mode(tile_mode: Optional[str], tile_num: int = 1) -> str:
    raw = str(tile_mode or "").strip().lower()
    if raw in {"1", "tile 1"}:
        return "1"
    if raw in {"2", "tile 2"}:
        return "2"
    if raw in {"1 and 2", "1&2", "1+2", "auto"}:
        return "1 and 2"
    legacy = max(1, int(tile_num))
    return "1" if legacy <= 1 else "2"


def _visible_chunk_from_steps(
    effective_inference_steps: float,
    *,
    visible_chunk_steps5: int,
    visible_chunk_steps6: int,
    visible_chunk_steps7: int,
    visible_chunk_steps8_plus: int,
) -> float:
    steps = max(1.0, float(effective_inference_steps))
    chunk_5 = float(max(1, int(visible_chunk_steps5)))
    chunk_6 = float(max(1, int(visible_chunk_steps6)))
    chunk_7 = float(max(1, int(visible_chunk_steps7)))
    chunk_8 = float(max(1, int(visible_chunk_steps8_plus)))
    if steps <= 5.0:
        return chunk_5
    if steps < 6.0:
        return chunk_5 + (steps - 5.0) * (chunk_6 - chunk_5)
    if steps < 7.0:
        return chunk_6 + (steps - 6.0) * (chunk_7 - chunk_6)
    if steps < 8.0:
        return chunk_7 + (steps - 7.0) * (chunk_8 - chunk_7)
    return chunk_8


def _steps_from_sharpness(val: float) -> float:
    try:
        v = float(val)
    except Exception:
        return 5.0
    if v <= 1100.0:
        return 5.0
    steps = 5.0 + ((v - 1100.0) / 1100.0)
    return max(5.0, min(8.0, steps))


def _load_json_dict(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        return None
    return None


def _load_sharpness_csv_maps(csv_path: str) -> Dict[str, Dict[str, float]]:
    by_name: Dict[str, float] = {}
    by_stem: Dict[str, float] = {}
    if not csv_path:
        return {"by_name": by_name, "by_stem": by_stem}
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return {"by_name": by_name, "by_stem": by_stem}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                name = str(row.get("file") or "").strip()
                if not name:
                    continue
                raw_s = str(row.get("sharpness_raw") or "").strip()
                pct_s = str(row.get("sharpness_pct") or "").strip()
                try:
                    raw = float(raw_s) if raw_s != "" else float(pct_s)
                except Exception:
                    continue
                by_name[name] = raw
                by_stem[Path(name).stem] = raw
    except Exception:
        return {"by_name": {}, "by_stem": {}}
    return {"by_name": by_name, "by_stem": by_stem}


def _lookup_sharpness_value(
    sharpness_maps: Dict[str, Dict[str, float]],
    *,
    mask_path: str,
    warped_path: str,
) -> Tuple[str, Optional[float]]:
    by_name = sharpness_maps.get("by_name", {})
    by_stem = sharpness_maps.get("by_stem", {})
    candidates: List[str] = []
    warped_base = os.path.basename(warped_path) if warped_path else ""
    if warped_base:
        candidates.append(warped_base)
    candidates.append(_mask_core_name(mask_path))
    candidates.append(os.path.basename(mask_path))

    seen: set[str] = set()
    for cand in candidates:
        if not cand:
            continue
        base = os.path.basename(cand)
        if base and base not in seen:
            seen.add(base)
            if base in by_name:
                return base, by_name[base]
        stem = Path(base).stem
        if stem and stem not in seen:
            seen.add(stem)
            if stem in by_stem:
                return stem, by_stem[stem]
    return "", None


def _planner_settings_from_config(config: Dict[str, object]) -> Dict[str, object]:
    mode = str(config.get("inpaint_mode", "Auto (recommended)") or "").strip()
    manual_mode = mode == "Manual"
    tile_mode = _normalize_tile_mode(
        config.get("inpaint_tile_mode", DEFAULT_INPAINT_TILE_MODE),
        tile_num=1,
    )
    settings = {
        "mode": "Manual" if manual_mode else "Auto (recommended)",
        "chunk_size": _parse_positive_int(
            config.get("inpaint_frames_chunk", DEFAULT_INPAINT_CHUNK_SIZE),
            DEFAULT_INPAINT_CHUNK_SIZE,
        ),
        "tile_mode": tile_mode,
        "tile1_max_size": _parse_positive_int(
            config.get("inpaint_tile1_max_size", DEFAULT_INPAINT_TILE1_MAX_SIZE),
            DEFAULT_INPAINT_TILE1_MAX_SIZE,
        ),
        "tile2_max_size": _parse_positive_int(
            config.get("inpaint_tile2_max_size", DEFAULT_INPAINT_TILE2_MAX_SIZE),
            DEFAULT_INPAINT_TILE2_MAX_SIZE,
        ),
        "overlap": _parse_nonnegative_int(
            config.get("inpaint_overlap", DEFAULT_INPAINT_OVERLAP),
            DEFAULT_INPAINT_OVERLAP,
        ),
    }
    if manual_mode:
        settings.update(
            {
                "enable_dynamic_chunk": _norm_bool(
                    config.get("inpaint_dynamic_chunk", True), True
                ),
                "use_sharpness_csv": _norm_bool(
                    config.get("inpaint_use_sharpness_csv", True), True
                ),
                "fixed_steps": _parse_positive_float(
                    config.get("inpaint_inference_steps", DEFAULT_INPAINT_FIXED_STEPS),
                    DEFAULT_INPAINT_FIXED_STEPS,
                ),
                "dynamic_visible_chunk_steps5": _parse_positive_int(
                    config.get(
                        "inpaint_dynamic_visible_chunk_steps5",
                        DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS5,
                    ),
                    DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS5,
                ),
                "dynamic_visible_chunk_steps6": _parse_positive_int(
                    config.get(
                        "inpaint_dynamic_visible_chunk_steps6",
                        DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS6,
                    ),
                    DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS6,
                ),
                "dynamic_visible_chunk_steps7": _parse_positive_int(
                    config.get(
                        "inpaint_dynamic_visible_chunk_steps7",
                        DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS7,
                    ),
                    DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS7,
                ),
                "dynamic_visible_chunk_steps8_plus": _parse_positive_int(
                    config.get(
                        "inpaint_dynamic_visible_chunk_steps8_plus",
                        DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS,
                    ),
                    DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS,
                ),
                "static_mask_divisor": _parse_positive_float(
                    config.get(
                        "inpaint_dynamic_hold_divisor",
                        DEFAULT_DYNAMIC_STATIC_MASK_DIVISOR,
                    ),
                    DEFAULT_DYNAMIC_STATIC_MASK_DIVISOR,
                ),
            }
        )
    else:
        settings.update(
            {
                "enable_dynamic_chunk": True,
                "use_sharpness_csv": True,
                "fixed_steps": DEFAULT_INPAINT_FIXED_STEPS,
                "dynamic_visible_chunk_steps5": DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS5,
                "dynamic_visible_chunk_steps6": DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS6,
                "dynamic_visible_chunk_steps7": DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS7,
                "dynamic_visible_chunk_steps8_plus": DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS,
                "static_mask_divisor": DEFAULT_DYNAMIC_STATIC_MASK_DIVISOR,
                "tile_mode": DEFAULT_INPAINT_TILE_MODE,
                "overlap": DEFAULT_INPAINT_OVERLAP,
            }
        )
    return settings


def _resolve_scene_chunk_plan(
    *,
    effective_inference_steps: float,
    max_hold_frames: int,
    chunk_size: int,
    overlap: int,
    enable_dynamic_chunk: bool,
    tile_mode: str,
    tile1_max_size: int,
    tile2_max_size: int,
    dynamic_visible_chunk_steps5: int,
    dynamic_visible_chunk_steps6: int,
    dynamic_visible_chunk_steps7: int,
    dynamic_visible_chunk_steps8_plus: int,
    static_mask_divisor: float,
) -> Dict[str, object]:
    current_overlap = max(0, int(overlap))
    min_processed_chunk = max(1, 2 * current_overlap + 1)
    effective_steps = max(1.0, float(effective_inference_steps))
    model_steps = max(1, _round_half_up(effective_steps))
    hold_divisor = _parse_positive_float(
        static_mask_divisor, DEFAULT_DYNAMIC_STATIC_MASK_DIVISOR
    )
    visible_chunk_from_steps = _visible_chunk_from_steps(
        effective_steps,
        visible_chunk_steps5=dynamic_visible_chunk_steps5,
        visible_chunk_steps6=dynamic_visible_chunk_steps6,
        visible_chunk_steps7=dynamic_visible_chunk_steps7,
        visible_chunk_steps8_plus=dynamic_visible_chunk_steps8_plus,
    )

    tile1_limit = max(min_processed_chunk, int(tile1_max_size))
    tile2_limit = max(min_processed_chunk, int(tile2_max_size))
    max_hold = max(0, int(max_hold_frames))
    hold_floor_visible = 0
    if enable_dynamic_chunk:
        if max_hold > 0:
            hold_floor_visible = int(math.ceil(float(max_hold) / hold_divisor))
        visible_chunk_after_hold = max(
            visible_chunk_from_steps, float(hold_floor_visible)
        )
        processed_chunk_target = max(
            float(min_processed_chunk),
            visible_chunk_after_hold + float(current_overlap),
        )
    else:
        visible_chunk_after_hold = max(
            1.0, float(max(1, int(chunk_size))) - float(current_overlap)
        )
        processed_chunk_target = max(
            float(min_processed_chunk), float(max(1, int(chunk_size)))
        )

    normalized_tile_mode = _normalize_tile_mode(tile_mode, tile_num=1)
    clamp_reason = ""
    if not enable_dynamic_chunk and normalized_tile_mode == "1 and 2":
        selected_tile = 2
        processed_chunk_selected = processed_chunk_target
    elif normalized_tile_mode == "1":
        selected_tile = 1
        processed_chunk_selected = min(processed_chunk_target, float(tile1_limit))
        if processed_chunk_selected < processed_chunk_target:
            clamp_reason = "tile1_max"
    elif normalized_tile_mode == "2":
        selected_tile = 2
        processed_chunk_selected = min(processed_chunk_target, float(tile2_limit))
        if processed_chunk_selected < processed_chunk_target:
            clamp_reason = "tile2_max"
    else:
        if processed_chunk_target <= float(tile1_limit):
            selected_tile = 1
            processed_chunk_selected = processed_chunk_target
        else:
            selected_tile = 2
            processed_chunk_selected = min(processed_chunk_target, float(tile2_limit))
            if processed_chunk_selected < processed_chunk_target:
                clamp_reason = "tile2_max"

    processed_chunk_final = max(
        min_processed_chunk, _round_half_up(processed_chunk_selected)
    )
    visible_chunk_final = max(1, processed_chunk_final - current_overlap)
    return {
        "effective_steps": float(effective_steps),
        "model_steps": int(model_steps),
        "visible_chunk_from_steps": float(visible_chunk_from_steps),
        "max_hold_frames": int(max_hold),
        "hold_floor_visible": int(hold_floor_visible),
        "static_mask_divisor": float(hold_divisor),
        "visible_chunk_after_hold": float(visible_chunk_after_hold),
        "selected_tile": int(selected_tile),
        "processed_chunk_final": int(processed_chunk_final),
        "visible_chunk_final": int(visible_chunk_final),
        "clamp_reason": clamp_reason,
    }


def _discover_planner_context(work_dir: str) -> Dict[str, object]:
    ctx: Dict[str, object] = {
        "work_dir": "",
        "config_path": "",
        "sharpness_csv_path": "",
        "settings": None,
        "sharpness_maps": None,
        "warnings": [],
    }
    if not work_dir:
        return ctx
    root = os.path.abspath(work_dir)
    ctx["work_dir"] = root
    if not os.path.isdir(root):
        ctx["warnings"].append(f"planner work_dir not found: {root}")
        return ctx

    config_path = os.path.join(root, DEFAULT_WORK_CONFIG_NAME)
    ctx["config_path"] = config_path
    config = _load_json_dict(config_path)
    if config is None:
        ctx["warnings"].append(
            f"planner config not found or unreadable: {config_path}"
        )
        return ctx

    settings = _planner_settings_from_config(config)
    ctx["settings"] = settings

    sharpness_csv_path = os.path.join(root, DEFAULT_SHARPNESS_CSV_NAME)
    ctx["sharpness_csv_path"] = sharpness_csv_path
    if settings["use_sharpness_csv"]:
        if not os.path.isfile(sharpness_csv_path):
            ctx["warnings"].append(
                f"planner sharpness.csv not found: {sharpness_csv_path}"
            )
        else:
            ctx["sharpness_maps"] = _load_sharpness_csv_maps(sharpness_csv_path)
            row_count = len((ctx["sharpness_maps"] or {}).get("by_name", {}))
            if row_count <= 0:
                ctx["warnings"].append(
                    f"planner sharpness.csv unreadable or empty: {sharpness_csv_path}"
                )
    return ctx


def _build_planner_row(
    *,
    planner_ctx: Dict[str, object],
    mask_path: str,
    warped_path: str,
    content_row: Optional[Dict[str, object]],
) -> Dict[str, object]:
    settings = planner_ctx.get("settings")
    if not settings:
        return _blank_planner_row("missing_config", "planner_config_unavailable")

    typed_settings = dict(settings)
    row = _blank_planner_row()
    row.update(
        {
            "planner_mode": typed_settings["mode"],
            "planner_dynamic_chunk_enabled": (
                "1" if bool(typed_settings["enable_dynamic_chunk"]) else "0"
            ),
            "planner_use_sharpness_csv": (
                "1" if bool(typed_settings["use_sharpness_csv"]) else "0"
            ),
            "planner_chunk_size_manual": int(typed_settings["chunk_size"]),
            "planner_overlap": int(typed_settings["overlap"]),
            "planner_tile_mode": str(typed_settings["tile_mode"]),
            "planner_tile1_max_size": int(typed_settings["tile1_max_size"]),
            "planner_tile2_max_size": int(typed_settings["tile2_max_size"]),
            "planner_dynamic_visible_chunk_steps5": int(
                typed_settings["dynamic_visible_chunk_steps5"]
            ),
            "planner_dynamic_visible_chunk_steps6": int(
                typed_settings["dynamic_visible_chunk_steps6"]
            ),
            "planner_dynamic_visible_chunk_steps7": int(
                typed_settings["dynamic_visible_chunk_steps7"]
            ),
            "planner_dynamic_visible_chunk_steps8_plus": int(
                typed_settings["dynamic_visible_chunk_steps8_plus"]
            ),
            "planner_static_mask_divisor": float(
                typed_settings["static_mask_divisor"]
            ),
        }
    )

    effective_steps = float(typed_settings["fixed_steps"])
    sharpness_key = ""
    sharpness_raw: Optional[float] = None
    if bool(typed_settings["use_sharpness_csv"]):
        sharpness_maps = planner_ctx.get("sharpness_maps")
        if not sharpness_maps or not (sharpness_maps.get("by_name") or sharpness_maps.get("by_stem")):
            row["planner_status"] = "missing_sharpness_csv"
            row["planner_warning"] = "planner_sharpness_csv_unavailable"
            return row
        sharpness_key, sharpness_raw = _lookup_sharpness_value(
            sharpness_maps,
            mask_path=mask_path,
            warped_path=warped_path,
        )
        if sharpness_raw is None:
            row["planner_status"] = "missing_sharpness_row"
            row["planner_warning"] = "planner_sharpness_row_missing"
            return row
        effective_steps = _steps_from_sharpness(sharpness_raw)
        row["planner_steps_source"] = "sharpness_csv"
        row["planner_sharpness_key"] = sharpness_key
        row["planner_sharpness_raw"] = float(sharpness_raw)
    else:
        row["planner_steps_source"] = "fixed_steps"

    hold_source = ""
    max_hold_frames = 0
    if bool(typed_settings["enable_dynamic_chunk"]):
        if content_row is None:
            row["planner_status"] = "missing_warped_hold"
            row["planner_warning"] = "planner_combined_hold_unavailable"
            return row
        max_hold_frames = int(content_row.get("max_hold_frames") or 0)
        hold_source = "combined_hold"
    else:
        hold_source = "fixed_chunk_manual"

    plan = _resolve_scene_chunk_plan(
        effective_inference_steps=effective_steps,
        max_hold_frames=max_hold_frames,
        chunk_size=int(typed_settings["chunk_size"]),
        overlap=int(typed_settings["overlap"]),
        enable_dynamic_chunk=bool(typed_settings["enable_dynamic_chunk"]),
        tile_mode=str(typed_settings["tile_mode"]),
        tile1_max_size=int(typed_settings["tile1_max_size"]),
        tile2_max_size=int(typed_settings["tile2_max_size"]),
        dynamic_visible_chunk_steps5=int(
            typed_settings["dynamic_visible_chunk_steps5"]
        ),
        dynamic_visible_chunk_steps6=int(
            typed_settings["dynamic_visible_chunk_steps6"]
        ),
        dynamic_visible_chunk_steps7=int(
            typed_settings["dynamic_visible_chunk_steps7"]
        ),
        dynamic_visible_chunk_steps8_plus=int(
            typed_settings["dynamic_visible_chunk_steps8_plus"]
        ),
        static_mask_divisor=float(typed_settings["static_mask_divisor"]),
    )
    row.update(
        {
            "planner_status": "ok",
            "planner_warning": "",
            "planner_hold_source": hold_source,
            "planner_max_hold_frames_used": int(plan["max_hold_frames"]),
            "planner_effective_steps": float(plan["effective_steps"]),
            "planner_model_steps": int(plan["model_steps"]),
            "planner_visible_chunk_from_steps": float(
                plan["visible_chunk_from_steps"]
            ),
            "planner_hold_floor_visible": int(plan["hold_floor_visible"]),
            "planner_visible_chunk_after_hold": float(
                plan["visible_chunk_after_hold"]
            ),
            "planner_visible_chunk_final": int(plan["visible_chunk_final"]),
            "planner_processed_chunk_final": int(plan["processed_chunk_final"]),
            "planner_selected_tile": int(plan["selected_tile"]),
            "planner_clamp_reason": str(plan["clamp_reason"]),
        }
    )
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze replace-mask clips and report, per file, the maximum duration "
            "for which a significant non-border component still preserves enough "
            "of its own initial anchor position. When a Pipeline Master work "
            "folder is available, also emit the effective inpaint chunk plan "
            "resolved from sharpness.csv + config_pipeline_master_gui.json."
        )
    )
    ap.add_argument("input", help="Input work folder, mask folder, or mask video file")
    ap.add_argument(
        "--work-dir",
        default="",
        help=(
            "Optional Pipeline Master work folder. If omitted, the tool tries to "
            "infer it from the input path (for example <work>/mask or <work>)."
        ),
    )
    ap.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=[],
        help=(
            "Glob pattern(s) when input resolves to a folder. Can be repeated. "
            f"Default: {', '.join(DEFAULT_GLOB_PATTERNS)}"
        ),
    )
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV path")
    ap.add_argument(
        "--warped-dir",
        default="",
        help=(
            "Optional folder containing warped source clips matching the replace-mask "
            "basenames. If omitted and a work folder is available, the tool tries "
            "to auto-detect <work>/splat/hires."
        ),
    )
    ap.add_argument(
        "--warped-file",
        default="",
        help="Optional explicit warped clip path when analyzing a single mask file.",
    )
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
        help=(
            "Ignore components touching left/right border within this tolerance. "
            f"Default: {DEFAULT_BORDER_TOLERANCE_PX}"
        ),
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
    ap.add_argument(
        "--content-gray-delta-threshold",
        type=float,
        default=DEFAULT_CONTENT_GRAY_DELTA_THRESHOLD,
        help=(
            "Max allowed mean absolute grayscale delta on the warped ROI versus the "
            f"anchor ROI. <=0 disables the grayscale gate. Default: {DEFAULT_CONTENT_GRAY_DELTA_THRESHOLD}"
        ),
    )
    ap.add_argument(
        "--content-edge-delta-threshold",
        type=float,
        default=DEFAULT_CONTENT_EDGE_DELTA_THRESHOLD,
        help=(
            "Max allowed mean absolute edge-magnitude delta on the warped ROI versus "
            f"the anchor ROI. <=0 disables the edge gate. Default: {DEFAULT_CONTENT_EDGE_DELTA_THRESHOLD}"
        ),
    )
    ap.add_argument(
        "--roi-mask-dilate-k",
        type=int,
        default=DEFAULT_ROI_MASK_DILATE_K,
        help=(
            "Optional dilation kernel size applied to each component before building "
            f"the warped ROI band. Default: {DEFAULT_ROI_MASK_DILATE_K}"
        ),
    )
    ap.add_argument(
        "--roi-mask-dilate-iter",
        type=int,
        default=DEFAULT_ROI_MASK_DILATE_ITER,
        help=(
            "Optional dilation iterations applied before building the warped ROI band. "
            f"Default: {DEFAULT_ROI_MASK_DILATE_ITER}"
        ),
    )
    ap.add_argument(
        "--min-content-roi-pixels",
        type=int,
        default=DEFAULT_MIN_CONTENT_ROI_PIXELS,
        help=(
            "Minimum overlapping warped ROI pixels required to accept a content match. "
            f"Default: {DEFAULT_MIN_CONTENT_ROI_PIXELS}"
        ),
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    patterns = tuple(args.globs) if args.globs else DEFAULT_GLOB_PATTERNS
    input_root, work_dir = _resolve_input_root_and_work_dir(args.input, args.work_dir)
    files = _iter_video_files(input_root, patterns, recursive=bool(args.recursive))
    if not files:
        print("No input files found.")
        return 1
    if args.warped_file and len(files) != 1:
        raise SystemExit(
            "--warped-file can only be used when the input resolves to a single mask file."
        )

    resolved_warped_dir = os.path.abspath(str(args.warped_dir).strip()) if str(args.warped_dir).strip() else ""
    if not resolved_warped_dir and not str(args.warped_file).strip():
        resolved_warped_dir = _resolve_auto_warped_dir(work_dir)

    planner_ctx = _discover_planner_context(work_dir)
    planner_warning_counts: Dict[str, int] = {}

    if os.path.abspath(input_root) != os.path.abspath(args.input):
        print(f"[INFO] input resolved to mask folder: {input_root}")
    if work_dir:
        print(f"[INFO] work_dir: {work_dir}")
    if resolved_warped_dir:
        print(f"[INFO] warped_dir: {resolved_warped_dir}")
    if planner_ctx.get("config_path"):
        print(f"[INFO] planner config: {planner_ctx['config_path']}")
    if planner_ctx.get("settings"):
        settings = planner_ctx["settings"]
        print(
            "[INFO] planner settings: "
            f"mode={settings['mode']} "
            f"dynamic={bool(settings['enable_dynamic_chunk'])} "
            f"sharpness_csv={bool(settings['use_sharpness_csv'])} "
            f"tile_mode={settings['tile_mode']} "
            f"overlap={int(settings['overlap'])} "
            f"divisor={float(settings['static_mask_divisor']):.2f}"
        )
    if planner_ctx.get("sharpness_csv_path") and planner_ctx.get("settings"):
        if planner_ctx["settings"]["use_sharpness_csv"]:
            print(f"[INFO] planner sharpness.csv: {planner_ctx['sharpness_csv_path']}")
    for msg in planner_ctx.get("warnings", []):
        print(f"[WARN] {msg}")

    results: List[Dict[str, object]] = []
    content_requested = bool(resolved_warped_dir or str(args.warped_file).strip())
    content_enabled_count = 0
    for idx, path in enumerate(files, start=1):
        mask_only = analyze_mask_video(
            path=path,
            threshold_u8=int(args.threshold_u8),
            min_area_px=int(args.min_area_px),
            border_tolerance_px=int(args.border_tolerance_px),
            component_merge_y_tol_px=int(args.component_merge_y_tol_px),
            anchor_overlap_min_ratio=float(args.anchor_overlap_min_ratio),
        )
        warped_path = ""
        if str(args.warped_file).strip():
            warped_path = os.path.abspath(str(args.warped_file).strip())
        elif resolved_warped_dir:
            found = _find_warped_for_mask(
                mask_path=path,
                warped_dir=resolved_warped_dir,
                recursive=bool(args.recursive),
            )
            warped_path = found or ""

        content_row: Optional[Dict[str, object]] = None
        if warped_path:
            content_row = analyze_mask_video_with_warped_content(
                mask_path=path,
                warped_path=warped_path,
                threshold_u8=int(args.threshold_u8),
                min_area_px=int(args.min_area_px),
                border_tolerance_px=int(args.border_tolerance_px),
                component_merge_y_tol_px=int(args.component_merge_y_tol_px),
                anchor_overlap_min_ratio=float(args.anchor_overlap_min_ratio),
                content_gray_delta_threshold=float(args.content_gray_delta_threshold),
                content_edge_delta_threshold=float(args.content_edge_delta_threshold),
                roi_mask_dilate_k=int(args.roi_mask_dilate_k),
                roi_mask_dilate_iter=int(args.roi_mask_dilate_iter),
                min_content_roi_pixels=int(args.min_content_roi_pixels),
            )
            content_enabled_count += 1

        planner_row = _build_planner_row(
            planner_ctx=planner_ctx,
            mask_path=path,
            warped_path=warped_path,
            content_row=content_row,
        )
        planner_status = str(planner_row.get("planner_status") or "")
        if planner_status and planner_status != "ok":
            planner_warning_counts[planner_status] = (
                planner_warning_counts.get(planner_status, 0) + 1
            )

        row = {
            "file": mask_only["file"],
            "warped_file": os.path.abspath(warped_path) if warped_path else "",
            "frames": mask_only["frames"],
            "fps": mask_only["fps"],
            "max_hold_frames_mask_only": mask_only["max_hold_frames"],
            "max_hold_seconds_mask_only": mask_only["max_hold_seconds"],
            "max_hold_start_frame_mask_only": mask_only["max_hold_start_frame"],
            "max_hold_end_frame_mask_only": mask_only["max_hold_end_frame"],
            "max_hold_area_px_mask_only": mask_only["max_hold_area_px"],
            "max_hold_frames_combined": (
                content_row["max_hold_frames"] if content_row is not None else ""
            ),
            "max_hold_seconds_combined": (
                content_row["max_hold_seconds"] if content_row is not None else ""
            ),
            "max_hold_start_frame_combined": (
                content_row["max_hold_start_frame"] if content_row is not None else ""
            ),
            "max_hold_end_frame_combined": (
                content_row["max_hold_end_frame"] if content_row is not None else ""
            ),
            "max_hold_area_px_combined": (
                content_row["max_hold_area_px"] if content_row is not None else ""
            ),
            "warped_layout": (
                content_row["warped_layout"] if content_row is not None else ""
            ),
            "max_hold_roi_pixels_combined": (
                content_row["max_hold_roi_pixels"] if content_row is not None else ""
            ),
            "max_hold_gray_delta_last": (
                content_row["max_hold_gray_delta_last"] if content_row is not None else ""
            ),
            "max_hold_edge_delta_last": (
                content_row["max_hold_edge_delta_last"] if content_row is not None else ""
            ),
            "max_hold_gray_delta_mean": (
                content_row["max_hold_gray_delta_mean"] if content_row is not None else ""
            ),
            "max_hold_edge_delta_mean": (
                content_row["max_hold_edge_delta_mean"] if content_row is not None else ""
            ),
            "max_hold_gray_delta_max": (
                content_row["max_hold_gray_delta_max"] if content_row is not None else ""
            ),
            "max_hold_edge_delta_max": (
                content_row["max_hold_edge_delta_max"] if content_row is not None else ""
            ),
            "max_hold_roi_overlap_ratio": (
                content_row["max_hold_roi_overlap_ratio"] if content_row is not None else ""
            ),
        }
        row.update(planner_row)
        results.append(row)

        if content_row is not None:
            msg = (
                f"[{idx}/{len(files)}] {os.path.basename(path)}  "
                f"mask_hold={int(mask_only['max_hold_frames'])}f  "
                f"combined_hold={int(content_row['max_hold_frames'])}f  "
                f"gray_mean={float(content_row['max_hold_gray_delta_mean']):.3f}  "
                f"edge_mean={float(content_row['max_hold_edge_delta_mean']):.3f}  "
                f"roi={int(content_row['max_hold_roi_pixels'])}"
            )
        else:
            missing_note = "  warped=missing" if content_requested else ""
            msg = (
                f"[{idx}/{len(files)}] {os.path.basename(path)}  "
                f"mask_hold={int(mask_only['max_hold_frames'])}f  "
                f"({float(mask_only['max_hold_seconds']):.3f}s){missing_note}"
            )
        if planner_status == "ok":
            msg += (
                f"  chunk={int(planner_row['planner_processed_chunk_final'])} "
                f"tile={int(planner_row['planner_selected_tile'])} "
                f"steps={float(planner_row['planner_effective_steps']):.2f}"
            )
        elif planner_status:
            msg += f"  planner={planner_status}"
        print(msg)

    results.sort(
        key=lambda row: (
            -int(row["max_hold_frames_combined"] or row["max_hold_frames_mask_only"]),
            str(row["file"]),
        )
    )

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "warped_file",
                "warped_layout",
                "frames",
                "fps",
                "max_hold_frames_mask_only",
                "max_hold_seconds_mask_only",
                "max_hold_start_frame_mask_only",
                "max_hold_end_frame_mask_only",
                "max_hold_area_px_mask_only",
                "max_hold_frames_combined",
                "max_hold_seconds_combined",
                "max_hold_start_frame_combined",
                "max_hold_end_frame_combined",
                "max_hold_area_px_combined",
                "max_hold_roi_pixels_combined",
                "max_hold_gray_delta_last",
                "max_hold_edge_delta_last",
                "max_hold_gray_delta_mean",
                "max_hold_edge_delta_mean",
                "max_hold_gray_delta_max",
                "max_hold_edge_delta_max",
                "max_hold_roi_overlap_ratio",
                *PLANNER_FIELDNAMES,
            ]
        )
        for row in results:
            w.writerow(
                [
                    row["file"],
                    row["warped_file"],
                    row["warped_layout"],
                    row["frames"],
                    f"{float(row['fps']):.6f}",
                    row["max_hold_frames_mask_only"],
                    f"{float(row['max_hold_seconds_mask_only']):.6f}",
                    row["max_hold_start_frame_mask_only"],
                    row["max_hold_end_frame_mask_only"],
                    row["max_hold_area_px_mask_only"],
                    row["max_hold_frames_combined"],
                    (
                        f"{float(row['max_hold_seconds_combined']):.6f}"
                        if row["max_hold_seconds_combined"] != ""
                        else ""
                    ),
                    row["max_hold_start_frame_combined"],
                    row["max_hold_end_frame_combined"],
                    row["max_hold_area_px_combined"],
                    row["max_hold_roi_pixels_combined"],
                    (
                        f"{float(row['max_hold_gray_delta_last']):.6f}"
                        if row["max_hold_gray_delta_last"] != ""
                        else ""
                    ),
                    (
                        f"{float(row['max_hold_edge_delta_last']):.6f}"
                        if row["max_hold_edge_delta_last"] != ""
                        else ""
                    ),
                    (
                        f"{float(row['max_hold_gray_delta_mean']):.6f}"
                        if row["max_hold_gray_delta_mean"] != ""
                        else ""
                    ),
                    (
                        f"{float(row['max_hold_edge_delta_mean']):.6f}"
                        if row["max_hold_edge_delta_mean"] != ""
                        else ""
                    ),
                    (
                        f"{float(row['max_hold_gray_delta_max']):.6f}"
                        if row["max_hold_gray_delta_max"] != ""
                        else ""
                    ),
                    (
                        f"{float(row['max_hold_edge_delta_max']):.6f}"
                        if row["max_hold_edge_delta_max"] != ""
                        else ""
                    ),
                    (
                        f"{float(row['max_hold_roi_overlap_ratio']):.6f}"
                        if row["max_hold_roi_overlap_ratio"] != ""
                        else ""
                    ),
                    row["planner_status"],
                    row["planner_warning"],
                    row["planner_mode"],
                    row["planner_dynamic_chunk_enabled"],
                    row["planner_use_sharpness_csv"],
                    row["planner_steps_source"],
                    row["planner_sharpness_key"],
                    (
                        f"{float(row['planner_sharpness_raw']):.6f}"
                        if row["planner_sharpness_raw"] != ""
                        else ""
                    ),
                    row["planner_chunk_size_manual"],
                    row["planner_overlap"],
                    row["planner_tile_mode"],
                    row["planner_tile1_max_size"],
                    row["planner_tile2_max_size"],
                    row["planner_dynamic_visible_chunk_steps5"],
                    row["planner_dynamic_visible_chunk_steps6"],
                    row["planner_dynamic_visible_chunk_steps7"],
                    row["planner_dynamic_visible_chunk_steps8_plus"],
                    (
                        f"{float(row['planner_static_mask_divisor']):.6f}"
                        if row["planner_static_mask_divisor"] != ""
                        else ""
                    ),
                    row["planner_hold_source"],
                    row["planner_max_hold_frames_used"],
                    (
                        f"{float(row['planner_effective_steps']):.6f}"
                        if row["planner_effective_steps"] != ""
                        else ""
                    ),
                    row["planner_model_steps"],
                    (
                        f"{float(row['planner_visible_chunk_from_steps']):.6f}"
                        if row["planner_visible_chunk_from_steps"] != ""
                        else ""
                    ),
                    row["planner_hold_floor_visible"],
                    (
                        f"{float(row['planner_visible_chunk_after_hold']):.6f}"
                        if row["planner_visible_chunk_after_hold"] != ""
                        else ""
                    ),
                    row["planner_visible_chunk_final"],
                    row["planner_processed_chunk_final"],
                    row["planner_selected_tile"],
                    row["planner_clamp_reason"],
                ]
            )

    best = results[0]
    print("")
    print(f"Wrote: {out_csv}")
    if content_enabled_count > 0:
        print(
            "Top combined hold: "
            f"{os.path.basename(str(best['file']))}  "
            f"{int(best['max_hold_frames_combined'] or 0)}f  "
            f"(mask_only={int(best['max_hold_frames_mask_only'])}f)"
        )
    else:
        print(
            "Top hold: "
            f"{os.path.basename(str(best['file']))}  "
            f"{int(best['max_hold_frames_mask_only'])}f  "
            f"({float(best['max_hold_seconds_mask_only']):.3f}s)"
        )
    if planner_warning_counts:
        print("Planner blanks by reason:")
        for key in sorted(planner_warning_counts):
            print(f"  - {key}: {planner_warning_counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
