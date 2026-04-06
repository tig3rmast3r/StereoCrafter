#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable
from dependency.repo_paths import config_path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
# The source-video stem is variable, so scene ids are keyed only by the
# trailing "-Scene-####" structure, not by a hardcoded "source" prefix.
SCENE_ID_PATTERN = r"[^/\\\\]+?-Scene-\d+(?:-Scene-\d+)*"

STEP_OPTIONS = [
    ("depthcrafter", "depthcrafter"),
    ("splatting", "splatting"),
    ("inpainting", "inpainting"),
    ("sharpen", "sharpen"),
    ("mask_for_merge", "mask for merge"),
    ("merging", "merging"),
]
STEP_ID_TO_LABEL = {step_id: label for step_id, label in STEP_OPTIONS}
STEP_LABEL_TO_ID = {label: step_id for step_id, label in STEP_OPTIONS}
STEP_ORDER = [step_id for step_id, _label in STEP_OPTIONS]
STEP_INDEX = {step_id: idx for idx, step_id in enumerate(STEP_ORDER)}
SUBSET_MODE_OPTIONS = [
    ("none", "Do Not create subset"),
    ("copy_all", "Create subset (copy all)"),
    ("copy_prev", "Create subset (copy only until previous step)"),
]
SUBSET_MODE_ID_TO_LABEL = {mode_id: label for mode_id, label in SUBSET_MODE_OPTIONS}
SUBSET_MODE_LABEL_TO_ID = {label.lower(): mode_id for mode_id, label in SUBSET_MODE_OPTIONS}

STEP_TO_DIRS = {
    # "mask" is the replace-mask folder produced by splatting (work/mask).
    # It is distinct from "mask_for_merge", which is generated later by the
    # mask preprocessing step used by merge.
    "depthcrafter": ["depthmap", "splat/hires", "mask", "output", "output-sharpen", "mask_for_merge", "sbs"],
    "splatting": ["splat/hires", "mask", "output", "output-sharpen", "mask_for_merge", "sbs"],
    "inpainting": ["output", "output-sharpen", "mask_for_merge", "sbs"],
    "sharpen": ["output-sharpen", "sbs"],
    "mask_for_merge": ["mask_for_merge", "sbs"],
    "merging": ["sbs"],
}
DIR_ORDER = ["depthmap", "splat/hires", "mask", "output", "output-sharpen", "mask_for_merge", "sbs"]
SUBSET_DIR_ORDER = ["seg", *DIR_ORDER]
DIR_LABELS = {
    "seg": "seg",
    "depthmap": "depthmap",
    "splat/hires": "splat/hires",
    "mask": "mask",
    "output": "output",
    "output-sharpen": "output-sharpen",
    "mask_for_merge": "mask_for_merge",
    "sbs": "sbs",
}
STEP_TO_SUBSET_PREV_DIRS = {
    "depthcrafter": ["seg"],
    "splatting": ["seg", "depthmap"],
    "inpainting": ["seg", "splat/hires", "mask"],
    "sharpen": ["seg", "splat/hires", "mask", "output"],
    "mask_for_merge": ["seg", "splat/hires", "mask", "output", "output-sharpen"],
    "merging": ["seg", "splat/hires", "mask", "output", "output-sharpen", "mask_for_merge"],
}

CSV_BASENAME_FIELDS = ("file_name", "filename", "file", "video", "file_path", "path")
OPTIONAL_MISSING_MATCH_DIRS = {"output-sharpen"}
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MASTER_CONFIG_PATH = config_path("config_pipeline_master_gui.json")
JOIN_DEFAULT_PRESET = "p7"
JOIN_DEFAULT_QUALITY = "16"
JOIN_DEFAULT_PIX_FMT = "yuv420p"
JOIN_DEFAULT_EXTRA_ARGS = (
    "-rc vbr -b:v 0 -multipass fullres -spatial_aq 1 "
    "-temporal_aq 1 -aq-strength 12 -rc-lookahead 32 -bf 3"
)
JOIN_DEFAULT_VF = "pad=iw:max(ih\\,1080):0:(max(ih\\,1080)-ih)/2:black,crop=iw:1080:0:(ih-1080)/2"
FFMPEG_CODEC_ALIASES = {
    "x264": "libx264",
    "x265": "libx265",
    "h265": "libx265",
}

SBS_FILENAME_RE = re.compile(
    rf"^(?P<core>(?P<scene>{SCENE_ID_PATTERN})_(?P<width>\d+))_merged_full_sbs\.(?P<ext>[^.]+)$",
    re.IGNORECASE,
)
SEG_FILENAME_RE = re.compile(
    rf"^(?P<scene>{SCENE_ID_PATTERN})\.[^.]+$",
    re.IGNORECASE,
)
DEPTH_FILENAME_RE = re.compile(
    rf"^(?P<scene>{SCENE_ID_PATTERN})_depth\.[^.]+$",
    re.IGNORECASE,
)
SPLAT_FILENAME_RE = re.compile(
    rf"^(?P<core>(?P<scene>{SCENE_ID_PATTERN})_\d+)_splatted[^.]*\.[^.]+$",
    re.IGNORECASE,
)
INPAINT_FILENAME_RE = re.compile(
    rf"^(?P<core>(?P<scene>{SCENE_ID_PATTERN})_\d+)_inpainted_right_eye\.[^.]+$",
    re.IGNORECASE,
)
MASK_FILENAME_RE = re.compile(
    rf"^(?P<core>(?P<scene>{SCENE_ID_PATTERN})_\d+)(?:_splatted[^.]*)?_replace_mask\.[^.]+$",
    re.IGNORECASE,
)
MERGED_FILENAME_RE = re.compile(
    rf"^(?P<core>(?P<scene>{SCENE_ID_PATTERN})_\d+)_merged_full_sbs\.[^.]+$",
    re.IGNORECASE,
)

LogFn = Callable[[str], None]


class RequeueError(RuntimeError):
    pass


@dataclass
class SbsSceneClip:
    scene_id: str
    scene_num: int
    core_with_width: str
    path: Path
    packets: int
    cumulative_packets: int = 0


@dataclass
class SceneRequest:
    scene_id: str
    step_id: str
    sources: list[str] = field(default_factory=list)
    core_hints: set[str] = field(default_factory=set)


@dataclass
class CsvRewritePlan:
    label: str
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]
    kept_rows: list[dict[str, str]]
    removed_rows: list[dict[str, str]]
    match_field: str
    match_names: list[str]

    @property
    def removed_count(self) -> int:
        return len(self.removed_rows)


@dataclass
class PlanConfig:
    work_dir: Path
    annotation_csv: Path | None
    csv_from_step: str
    delete_from_main: bool
    subset_mode: str
    textbox_names: list[str]
    remove_sharpness_rows: bool
    remove_autoct_rows: bool


@dataclass
class SubsetPlan:
    mode: str
    subset_dir: Path
    copy_targets: dict[str, list[Path]]
    purge_targets: dict[str, list[str]]
    manifest: dict[str, object]


@dataclass
class ExecutionPlan:
    config: PlanConfig
    csv_click_count: int
    textbox_line_count: int
    unique_scene_count: int
    scene_requests: dict[str, SceneRequest]
    file_targets: dict[str, list[Path]]
    sharpness_plan: CsvRewritePlan | None
    sharpness_delete_plan: CsvRewritePlan | None
    autoct_plan: CsvRewritePlan | None
    subset_plan: SubsetPlan | None
    warnings: list[str] = field(default_factory=list)

    def preview_text(self) -> str:
        scene_items = sorted(
            self.scene_requests.values(),
            key=lambda item: (_scene_sort_key(item.scene_id), item.scene_id),
        )
        folder_lines = []
        for rel_dir in DIR_ORDER:
            count = len(self.file_targets.get(rel_dir, []))
            if count:
                folder_lines.append(f"- {DIR_LABELS[rel_dir]}: {count}")
        if not folder_lines:
            folder_lines.append("- no matching files")

        scene_preview = ", ".join(
            f"{item.scene_id} ({STEP_ID_TO_LABEL[item.step_id]})"
            for item in scene_items[:12]
        )
        if len(scene_items) > 12:
            scene_preview += f", ... +{len(scene_items) - 12} more"

        sharp_rows = self.sharpness_delete_plan.removed_count if self.sharpness_delete_plan else 0
        auto_rows = self.autoct_plan.removed_count if self.autoct_plan else 0
        subset_sharp_rows = self.sharpness_plan.removed_count if self.sharpness_plan else 0
        subset_auto_rows = self.autoct_plan.removed_count if self.autoct_plan else 0
        remove_sharp_rows = (
            sharp_rows
            if self.config.delete_from_main and self.config.remove_sharpness_rows
            else 0
        )
        remove_auto_rows = (
            auto_rows
            if self.config.delete_from_main and self.config.remove_autoct_rows
            else 0
        )

        subset_lines = []
        subset_csv_lines = []
        if self.subset_plan is not None:
            for rel_dir in SUBSET_DIR_ORDER:
                count = len(self.subset_plan.copy_targets.get(rel_dir, []))
                if count:
                    subset_lines.append(f"- {DIR_LABELS[rel_dir]}: {count}")
            if not subset_lines:
                subset_lines.append("- no matching files")
            subset_csv_lines = [
                f"- sharpness.csv rows seeded: {subset_sharp_rows}",
                f"- autoct.csv rows seeded: {subset_auto_rows}",
            ]

        lines = [
            f"Delete from main workdir: {'yes' if self.config.delete_from_main else 'no'}",
            f"Subset mode: {SUBSET_MODE_ID_TO_LABEL[self.config.subset_mode]}",
            f"Work folder: {self.config.work_dir}",
            f"CSV clicks: {self.csv_click_count}",
            f"Textbox basenames: {self.textbox_line_count}",
            f"Unique scenes: {self.unique_scene_count}",
            f"Scenes: {scene_preview or '(none)'}",
            "",
            "Files by folder:",
            *folder_lines,
            "",
            "CSV rows to remove from main:",
            f"- sharpness.csv: {remove_sharp_rows}",
            f"- autoct.csv: {remove_auto_rows}",
        ]
        if self.subset_plan is not None:
            lines.extend(
                [
                    "",
                    f"Subset folder: {self.subset_plan.subset_dir}",
                    "Subset files to copy:",
                    *subset_lines,
                    "",
                    "Subset CSV rows to seed:",
                    *subset_csv_lines,
                ]
            )
        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings[:8]:
                lines.append(f"- {warning}")
            if len(self.warnings) > 8:
                lines.append(f"- ... +{len(self.warnings) - 8} more")
        lines.extend(["", "Proceed with these changes?"])
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    subset_dir: str = ""
    subset_file_count: int = 0
    file_action_count: int = 0
    file_error_count: int = 0
    sharpness_removed: int = 0
    autoct_removed: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConfirmRequest:
    title: str
    message: str
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


@dataclass
class RestorePlan:
    work_dir: Path
    subset_dir: Path
    purge_targets: dict[str, list[Path]]
    copy_targets: dict[str, list[Path]]
    sharpness_csv: Path | None
    autoct_csv: Path | None
    sharpness_scope_names: list[str]
    autoct_scope_names: list[str]
    warnings: list[str] = field(default_factory=list)

    def preview_text(self) -> str:
        purge_lines = []
        copy_lines = []
        for rel_dir in SUBSET_DIR_ORDER:
            purge_count = len(self.purge_targets.get(rel_dir, []))
            if purge_count:
                purge_lines.append(f"- {DIR_LABELS[rel_dir]}: {purge_count}")
            copy_count = len(self.copy_targets.get(rel_dir, []))
            if copy_count:
                copy_lines.append(f"- {DIR_LABELS[rel_dir]}: {copy_count}")
        if not purge_lines:
            purge_lines.append("- no existing managed files")
        if not copy_lines:
            copy_lines.append("- no subset files found")

        lines = [
            f"Target work folder: {self.work_dir}",
            f"Subset folder: {self.subset_dir}",
            "",
            "Existing main files to purge:",
            *purge_lines,
            "",
            "Subset files to restore:",
            *copy_lines,
            "",
            "CSV scope replacement:",
            f"- sharpness.csv scope names: {len(self.sharpness_scope_names)}",
            f"- autoct.csv scope videos: {len(self.autoct_scope_names)}",
            "",
            "Proceed with restore?",
        ]
        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings[:8]:
                lines.append(f"- {warning}")
            if len(self.warnings) > 8:
                lines.append(f"- ... +{len(self.warnings) - 8} more")
        return "\n".join(lines)


@dataclass
class RestoreResult:
    purged_file_count: int = 0
    copied_file_count: int = 0
    file_error_count: int = 0
    sharpness_replaced: int = 0
    autoct_replaced: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class JoinSettings:
    config_path: Path
    encoder: str
    preset: str
    quality_flag: str
    quality_value: str
    pix_fmt: str
    extra_args: str


def _scene_sort_key(scene_id: str) -> int:
    try:
        return int(scene_id.rsplit("-", 1)[-1])
    except Exception:
        return 10**9


def _normalize_join_encoder(value: str) -> str:
    raw = str(value or "").strip().lower()
    raw = FFMPEG_CODEC_ALIASES.get(raw, raw)
    return raw or "hevc_nvenc"


def _join_quality_flag_for_encoder(encoder: str) -> str:
    return "cq" if "nvenc" in str(encoder or "").lower() else "crf"


def _load_join_settings(config_path: Path = PIPELINE_MASTER_CONFIG_PATH) -> JoinSettings:
    if not config_path.is_file():
        raise RequeueError(f"Pipeline config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RequeueError(f"Could not read pipeline config {config_path}: {exc}") from exc

    encoder = _normalize_join_encoder(payload.get("join_encoder", "hevc_nvenc"))
    preset = str(payload.get("join_preset", JOIN_DEFAULT_PRESET) or JOIN_DEFAULT_PRESET).strip()
    quality_value = str(payload.get("join_crf", JOIN_DEFAULT_QUALITY) or JOIN_DEFAULT_QUALITY).strip()
    pix_fmt = str(payload.get("join_pix_fmt", JOIN_DEFAULT_PIX_FMT) or JOIN_DEFAULT_PIX_FMT).strip()
    extra_args = str(payload.get("join_extra_args", JOIN_DEFAULT_EXTRA_ARGS) or "").strip()
    if not extra_args:
        extra_args = JOIN_DEFAULT_EXTRA_ARGS

    return JoinSettings(
        config_path=config_path.resolve(),
        encoder=encoder,
        preset=preset,
        quality_flag=_join_quality_flag_for_encoder(encoder),
        quality_value=quality_value or JOIN_DEFAULT_QUALITY,
        pix_fmt=pix_fmt or JOIN_DEFAULT_PIX_FMT,
        extra_args=extra_args,
    )


def _collect_sbs_join_map(folder: Path) -> dict[str, Path]:
    results: dict[str, Path] = {}
    if not folder.is_dir():
        return results
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "_sbs" not in path.stem.lower():
            continue
        key = path.stem.lower()
        if key not in results:
            results[key] = path
    return results


def _default_join_output_basename(encoder: str) -> str:
    return f"final_sbs_1080_{_normalize_join_encoder(encoder)}.mp4"


def _build_rejoin_inplace_sequence(
    work_dir: Path,
    subset_dir: Path,
    log: LogFn,
) -> tuple[list[tuple[str, Path]], int, int]:
    work_map = _collect_sbs_join_map(work_dir / "sbs")
    if not work_map:
        raise RequeueError(f"No '*_sbs.*' scene files found in {work_dir / 'sbs'}")
    subset_map = _collect_sbs_join_map(subset_dir / "sbs")

    selected: list[tuple[str, Path]] = []
    subset_hits = 0
    work_hits = 0
    for key in sorted(set(work_map) | set(subset_map)):
        chosen = subset_map.get(key) or work_map.get(key)
        if chosen is None:
            continue
        selected.append((key, chosen))
        if key in subset_map:
            subset_hits += 1
        else:
            work_hits += 1

    if not selected:
        raise RequeueError("No joinable scene clips were found for rejoin.")

    log(
        f"[SCAN] Rejoin in-place inputs: total={len(selected)} subset_overrides={subset_hits} "
        f"work_fallbacks={work_hits}"
    )
    return selected, subset_hits, work_hits


def _build_compare_sequence(
    work_dir: Path,
    subset_dir: Path,
    log: LogFn,
) -> tuple[list[tuple[str, Path]], int]:
    work_map = _collect_sbs_join_map(work_dir / "sbs")
    subset_map = _collect_sbs_join_map(subset_dir / "sbs")
    shared_keys = sorted(set(work_map) & set(subset_map))
    if not shared_keys:
        raise RequeueError(
            f"No matching '*_sbs.*' scene pairs found between {work_dir / 'sbs'} and {subset_dir / 'sbs'}"
        )

    selected: list[tuple[str, Path]] = []
    for key in shared_keys:
        selected.append((f"{key}__orig", work_map[key]))
        selected.append((f"{key}__subset", subset_map[key]))

    log(f"[SCAN] Compare join pairs: matched={len(shared_keys)} concat_items={len(selected)}")
    return selected, len(shared_keys)


def _run_join_script_on_sequence(
    sequence: list[tuple[str, Path]],
    output_path: Path,
    settings: JoinSettings,
    log: LogFn,
) -> None:
    if not sequence:
        raise RequeueError("Empty join sequence.")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = (REPO_ROOT / "Utilities" / "Rejoin_HEVC_NVENC.sh").resolve()
    if not script_path.is_file():
        raise RequeueError(f"Join script not found: {script_path}")

    with tempfile.TemporaryDirectory(prefix=".requeue_join_stage_", dir=str(output_path.parent.parent if output_path.parent.parent.is_dir() else output_path.parent)) as tmpdir:
        stage_dir = Path(tmpdir)
        for idx, (_key, src) in enumerate(sequence, start=1):
            dst_name = f"{idx:06d}__{src.name}"
            dst = stage_dir / dst_name
            dst.symlink_to(src.resolve())

        env = os.environ.copy()
        env.update(
            {
                "DIR_SBS": str(stage_dir),
                "PATTERN": "*",
                "OUT": str(output_path),
                "FFMPEG_BIN": "ffmpeg",
                "ENCODER": settings.encoder,
                "PRESET": settings.preset,
                "QUALITY_FLAG": settings.quality_flag,
                "QUALITY_VALUE": settings.quality_value,
                "CQ": settings.quality_value,
                "CRF": settings.quality_value,
                "PIX_FMT": settings.pix_fmt,
                "EXTRA_ARGS": settings.extra_args,
                "VF": JOIN_DEFAULT_VF,
            }
        )
        cmd = ["bash", str(script_path)]
        log(
            "[INFO] Join settings from config: "
            f"encoder={settings.encoder} {settings.quality_flag}={settings.quality_value} "
            f"preset={settings.preset} pix_fmt={settings.pix_fmt}"
        )
        log(f"[INFO] Join config source: {settings.config_path}")
        log(
            "CMD: "
            + " ".join([f"{k}={shlex.quote(str(v))}" for k, v in env.items() if k in {
                'DIR_SBS', 'PATTERN', 'OUT', 'ENCODER', 'PRESET', 'QUALITY_FLAG',
                'QUALITY_VALUE', 'PIX_FMT', 'EXTRA_ARGS', 'VF'
            }] + [shlex.quote(x) for x in cmd])
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = str(line or "").rstrip()
            if clean:
                log(clean)
        rc = proc.wait()
        if rc != 0:
            raise RequeueError(f"Join command failed with rc={rc}")


def _canonical_step(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in STEP_ID_TO_LABEL:
        return raw
    if raw in STEP_LABEL_TO_ID:
        return STEP_LABEL_TO_ID[raw]
    raise RequeueError(f"Unsupported step: {value!r}")


def _canonical_subset_mode(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in SUBSET_MODE_ID_TO_LABEL:
        return raw
    if raw in SUBSET_MODE_LABEL_TO_ID:
        return SUBSET_MODE_LABEL_TO_ID[raw]
    raise RequeueError(f"Unsupported subset mode: {value!r}")


def _earliest_step(step_a: str, step_b: str) -> str:
    return step_a if STEP_INDEX[step_a] <= STEP_INDEX[step_b] else step_b


def _video_glob_matches(folder: Path, pattern: str) -> list[Path]:
    matches = []
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            matches.append(path)
    return matches


def _add_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _basename_from_csv_value(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = raw.replace("\\", "/")
    return raw.rsplit("/", 1)[-1].strip()


def _extract_csv_click_basename(row: dict[str, str]) -> str:
    for field in CSV_BASENAME_FIELDS:
        basename = _basename_from_csv_value((row or {}).get(field, ""))
        if basename:
            return basename
    return ""


def _annotation_csv_resolution_mode(csv_path: Path) -> str:
    unique_basenames: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basename = _extract_csv_click_basename(dict(row or {}))
            if not basename:
                continue
            unique_basenames.add(basename)
            if len(unique_basenames) > 1:
                return "basename"
    return "frame"


def _parse_csv_scene_basename(name: str) -> tuple[str, str | None]:
    basename = _basename_from_csv_value(name)
    for pattern in (
        SEG_FILENAME_RE,
        DEPTH_FILENAME_RE,
        SPLAT_FILENAME_RE,
        INPAINT_FILENAME_RE,
        MASK_FILENAME_RE,
        MERGED_FILENAME_RE,
    ):
        match = pattern.match(basename)
        if not match:
            continue
        scene_id = str(match.group("scene"))
        core = match.groupdict().get("core")
        return scene_id, core

    raise RequeueError(
        "Unsupported annotation CSV basename: "
        f"{basename}. Supported basenames: scene clips, *_depth.*, *_splatted*, "
        "*_inpainted_right_eye.*, *_replace_mask.*, *_merged_full_sbs.*"
    )


def _parse_textbox_basename(name: str) -> tuple[str, str, str | None]:
    if "/" in name or "\\" in name:
        raise RequeueError(
            f"Textbox input must contain basenames only, not paths: {name}"
        )

    for pattern, step_id in (
        (DEPTH_FILENAME_RE, "depthcrafter"),
        (SPLAT_FILENAME_RE, "splatting"),
        (INPAINT_FILENAME_RE, "inpainting"),
        (MASK_FILENAME_RE, "mask_for_merge"),
        (MERGED_FILENAME_RE, "merging"),
    ):
        match = pattern.match(name)
        if not match:
            continue
        scene_id = str(match.group("scene"))
        core = match.groupdict().get("core")
        return scene_id, step_id, core

    raise RequeueError(
        "Unsupported textbox filename pattern: "
        f"{name}. Supported basenames: *_depth.*, *_splatted*, "
        "*_inpainted_right_eye.*, *_replace_mask.*, *_merged_full_sbs.*"
    )


def _ffprobe_packet_count(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or f"ffprobe rc={proc.returncode}").strip()
        raise RequeueError(f"ffprobe failed for {path.name}: {msg}")
    out = (proc.stdout or "").strip().splitlines()
    raw = out[0].strip() if out else ""
    try:
        return int(raw)
    except Exception as exc:
        raise RequeueError(
            f"Invalid ffprobe packet count for {path.name}: {raw!r}"
        ) from exc


def _build_sbs_index(work_dir: Path, log: LogFn, warnings: list[str]) -> list[SbsSceneClip]:
    sbs_dir = work_dir / "sbs"
    if not sbs_dir.is_dir():
        raise RequeueError(f"SBS folder not found: {sbs_dir}")

    parsed: list[tuple[int, str, str, Path]] = []
    ignored = 0
    scene_ids_seen: set[str] = set()
    duplicates: set[str] = set()

    for path in sorted(sbs_dir.iterdir()):
        if not path.is_file():
            continue
        match = SBS_FILENAME_RE.match(path.name)
        if not match:
            ignored += 1
            continue
        scene_id = str(match.group("scene"))
        if scene_id in scene_ids_seen:
            duplicates.add(scene_id)
            continue
        scene_ids_seen.add(scene_id)
        parsed.append(
            (
                _scene_sort_key(scene_id),
                scene_id,
                str(match.group("core")),
                path,
            )
        )

    if duplicates:
        dup_txt = ", ".join(sorted(duplicates))
        raise RequeueError(f"Duplicate SBS merged clips found for scenes: {dup_txt}")
    if not parsed:
        raise RequeueError(f"No '*_merged_full_sbs.*' files found in {sbs_dir}")
    if ignored:
        _add_warning(warnings, f"Ignored {ignored} non-SBS file(s) in {sbs_dir}.")

    parsed.sort(key=lambda item: (item[0], item[3].name))
    total = len(parsed)
    clips: list[SbsSceneClip] = []
    cumulative = 0
    for idx, (_scene_num, scene_id, core_with_width, path) in enumerate(parsed, start=1):
        if idx == 1 or idx == total or idx % 100 == 0:
            log(f"[SCAN] ffprobe packets {idx}/{total}: {path.name}")
        packets = _ffprobe_packet_count(path)
        cumulative += packets
        clips.append(
            SbsSceneClip(
                scene_id=scene_id,
                scene_num=_scene_sort_key(scene_id),
                core_with_width=core_with_width,
                path=path,
                packets=packets,
                cumulative_packets=cumulative,
            )
        )
    log(f"[SCAN] Indexed {len(clips)} SBS clip(s), total_packets={cumulative}")
    return clips


def _add_scene_request(
    requests: dict[str, SceneRequest],
    scene_id: str,
    step_id: str,
    source: str,
    core_hint: str | None,
) -> None:
    if scene_id in requests:
        existing = requests[scene_id]
        existing.step_id = _earliest_step(existing.step_id, step_id)
        existing.sources.append(source)
        if core_hint:
            existing.core_hints.add(core_hint)
        return
    requests[scene_id] = SceneRequest(
        scene_id=scene_id,
        step_id=step_id,
        sources=[source],
        core_hints={core_hint} if core_hint else set(),
    )


def _parse_annotation_csv(
    csv_path: Path,
    from_step: str,
    resolution_mode: str,
    sbs_index: list[SbsSceneClip] | None,
    requests: dict[str, SceneRequest],
    log: LogFn,
) -> int:
    if not csv_path.is_file():
        raise RequeueError(f"Annotation CSV not found: {csv_path}")

    if resolution_mode == "basename":
        click_count = 0
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader, start=2):
                basename = _extract_csv_click_basename(dict(row or {}))
                if not basename:
                    raise RequeueError(
                        f"Missing file basename in {csv_path} line {row_idx}; "
                        f"expected one of columns: {', '.join(CSV_BASENAME_FIELDS)}"
                    )
                scene_id, core_hint = _parse_csv_scene_basename(basename)
                _add_scene_request(
                    requests,
                    scene_id=scene_id,
                    step_id=from_step,
                    source=f"csv basename {basename}",
                    core_hint=core_hint,
                )
                click_count += 1
        if click_count <= 0:
            raise RequeueError(f"No annotation rows found in {csv_path}")
        log(
            f"[SCAN] Parsed {click_count} annotation click(s) by basename from {csv_path.name}"
        )
        return click_count

    if sbs_index is None:
        raise RequeueError("SBS index is required for frame-based annotation CSV parsing.")

    cumulative = [clip.cumulative_packets for clip in sbs_index]
    click_count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "frame" not in fieldnames:
            raise RequeueError(f"CSV column 'frame' not found in {csv_path}")
        for row_idx, row in enumerate(reader, start=2):
            raw = str((row or {}).get("frame", "")).strip()
            if raw == "":
                raise RequeueError(f"Missing frame value in {csv_path} line {row_idx}")
            try:
                frame_idx = int(float(raw))
            except Exception as exc:
                raise RequeueError(
                    f"Invalid frame value in {csv_path} line {row_idx}: {raw!r}"
                ) from exc
            if frame_idx < 0:
                raise RequeueError(
                    f"Negative frame value in {csv_path} line {row_idx}: {frame_idx}"
                )
            clip_idx = bisect.bisect_right(cumulative, frame_idx)
            if clip_idx >= len(sbs_index):
                raise RequeueError(
                    f"Frame {frame_idx} in {csv_path} line {row_idx} is outside SBS total packets"
                )
            clip = sbs_index[clip_idx]
            _add_scene_request(
                requests,
                scene_id=clip.scene_id,
                step_id=from_step,
                source=f"csv frame {frame_idx} -> {clip.path.name}",
                core_hint=clip.core_with_width,
            )
            click_count += 1
    if click_count <= 0:
        raise RequeueError(f"No annotation rows found in {csv_path}")
    log(f"[SCAN] Parsed {click_count} annotation click(s) from {csv_path.name}")
    return click_count


def _parse_textbox_requests(
    textbox_names: list[str],
    requests: dict[str, SceneRequest],
    log: LogFn,
) -> int:
    valid_count = 0
    for line_idx, raw in enumerate(textbox_names, start=1):
        name = str(raw or "").strip()
        if not name:
            continue
        basename = Path(name).name
        if basename != name:
            raise RequeueError(
                f"Textbox line {line_idx} must be a basename only, got: {name}"
            )
        scene_id, step_id, core_hint = _parse_textbox_basename(basename)
        _add_scene_request(
            requests,
            scene_id=scene_id,
            step_id=step_id,
            source=f"textbox:{basename}",
            core_hint=core_hint,
        )
        valid_count += 1
    if valid_count:
        log(f"[SCAN] Parsed {valid_count} textbox basename(s)")
    return valid_count


def _step_needs_sharpness_subset_rows(step_id: str) -> bool:
    return step_id in {"depthcrafter", "splatting", "inpainting", "sharpen"}


def _step_invalidates_sharpness_rows(step_id: str) -> bool:
    return step_id in {"depthcrafter", "splatting", "inpainting"}


def _step_needs_autoct_rows(step_id: str) -> bool:
    return step_id in {"depthcrafter", "splatting", "inpainting", "sharpen"}


def _suppress_missing_match_warning(rel_dir: str) -> bool:
    return rel_dir in OPTIONAL_MISSING_MATCH_DIRS


def _find_scene_matches(folder: Path, rel_dir: str, scene_id: str) -> list[Path]:
    pattern_map = {
        "seg": [f"{scene_id}.*"],
        "depthmap": [f"{scene_id}_depth.*"],
        "splat/hires": [f"{scene_id}_*_splatted*.*"],
        "mask": [f"{scene_id}_*_replace_mask.*"],
        "output": [f"{scene_id}_*_inpainted_right_eye.*"],
        "output-sharpen": [f"{scene_id}_*_inpainted_right_eye.*"],
        "mask_for_merge": [f"{scene_id}_*_replace_mask.*"],
        "sbs": [f"{scene_id}_*_merged_full_sbs.*"],
    }
    matches: dict[str, Path] = {}
    for pattern in pattern_map[rel_dir]:
        for path in _video_glob_matches(folder, pattern):
            matches[path.name] = path
    return [matches[name] for name in sorted(matches)]


def _copy_or_replace_file(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
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


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _relative_to_work_dir(work_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(work_dir.resolve()))
    except Exception as exc:
        raise RequeueError(f"Path is outside work folder: {path}") from exc


def _create_subset_dir(work_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    for idx in range(1000):
        suffix = "" if idx == 0 else f"_{idx:02d}"
        candidate = work_dir / f"_requeue_subset_{stamp}{suffix}"
        if not candidate.exists():
            return candidate
    raise RequeueError(f"Could not find a free subset folder name under {work_dir}")


def _collect_file_targets(
    work_dir: Path,
    requests: dict[str, SceneRequest],
    warnings: list[str],
) -> dict[str, list[Path]]:
    targets: dict[str, list[Path]] = {rel_dir: [] for rel_dir in DIR_ORDER}
    seen: dict[str, set[Path]] = {rel_dir: set() for rel_dir in DIR_ORDER}

    for request in sorted(requests.values(), key=lambda item: (_scene_sort_key(item.scene_id), item.scene_id)):
        for rel_dir in STEP_TO_DIRS[request.step_id]:
            folder = work_dir / rel_dir
            if not folder.is_dir():
                if not _suppress_missing_match_warning(rel_dir):
                    _add_warning(warnings, f"Folder missing: {folder}")
                continue
            matches = _find_scene_matches(folder, rel_dir, request.scene_id)
            if not matches:
                if not _suppress_missing_match_warning(rel_dir):
                    _add_warning(
                        warnings,
                        f"No {DIR_LABELS[rel_dir]} files found for {request.scene_id}",
                    )
                continue
            for path in matches:
                if path not in seen[rel_dir]:
                    seen[rel_dir].add(path)
                    targets[rel_dir].append(path)

    for rel_dir in DIR_ORDER:
        targets[rel_dir].sort(key=lambda path: path.name)
    return targets


def _load_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {str(key): "" if value is None else str(value) for key, value in dict(row or {}).items()}
            for row in reader
        ]
    return fieldnames, rows


def _scene_matches_prefixed_basename(name: str, request: SceneRequest, suffix: str) -> bool:
    basename = Path(name).name
    for core in sorted(request.core_hints):
        if basename.startswith(core + suffix):
            return True
    scene_prefix = request.scene_id + "_"
    return basename.startswith(scene_prefix) and suffix in basename


def _build_sharpness_rewrite_plan(
    work_dir: Path,
    requests: dict[str, SceneRequest],
    *,
    step_filter: Callable[[str], bool] | None = None,
) -> CsvRewritePlan:
    csv_path = work_dir / "sharpness.csv"
    if not csv_path.is_file():
        raise RequeueError(f"Requested sharpness.csv cleanup but file is missing: {csv_path}")
    fieldnames, rows = _load_csv_rows(csv_path)
    if "file" not in set(fieldnames):
        raise RequeueError(f"Column 'file' not found in {csv_path}")

    impacted = [
        request
        for request in requests.values()
        if (
            step_filter(request.step_id)
            if step_filter is not None
            else "splat/hires" in STEP_TO_DIRS[request.step_id]
        )
    ]
    kept_rows = []
    removed_rows = []
    impacted_files: set[str] = set()
    for row in rows:
        basename = str(row.get("file", "")).strip()
        if basename and any(
            _scene_matches_prefixed_basename(basename, request, "_splatted")
            for request in impacted
        ):
            removed_rows.append(row)
            impacted_files.add(Path(basename).name)
            continue
        kept_rows.append(row)
    return CsvRewritePlan(
        label="sharpness.csv",
        path=csv_path,
        fieldnames=fieldnames,
        rows=rows,
        kept_rows=kept_rows,
        removed_rows=removed_rows,
        match_field="file",
        match_names=sorted(impacted_files),
    )


def _build_autoct_rewrite_plan(
    work_dir: Path,
    requests: dict[str, SceneRequest],
    *,
    step_filter: Callable[[str], bool] | None = None,
) -> CsvRewritePlan:
    csv_path = work_dir / "autoct.csv"
    if not csv_path.is_file():
        raise RequeueError(f"Requested autoct.csv cleanup but file is missing: {csv_path}")
    fieldnames, rows = _load_csv_rows(csv_path)
    if "video" not in set(fieldnames):
        raise RequeueError(f"Column 'video' not found in {csv_path}")

    impacted = [
        request
        for request in requests.values()
        if (
            step_filter(request.step_id)
            if step_filter is not None
            else "output" in STEP_TO_DIRS[request.step_id]
        )
    ]
    kept_rows = []
    removed_rows = []
    impacted_videos: set[str] = set()
    for row in rows:
        basename = str(row.get("video", "")).strip()
        if basename and any(
            _scene_matches_prefixed_basename(basename, request, "_inpainted_right_eye")
            for request in impacted
        ):
            removed_rows.append(row)
            impacted_videos.add(Path(basename).name)
            continue
        kept_rows.append(row)
    return CsvRewritePlan(
        label="autoct.csv",
        path=csv_path,
        fieldnames=fieldnames,
        rows=rows,
        kept_rows=kept_rows,
        removed_rows=removed_rows,
        match_field="video",
        match_names=sorted(impacted_videos),
    )


def _maybe_build_csv_rewrite_plan(
    builder: Callable[[Path, dict[str, SceneRequest]], CsvRewritePlan],
    work_dir: Path,
    requests: dict[str, SceneRequest],
    *,
    required: bool,
    warnings: list[str],
) -> CsvRewritePlan | None:
    try:
        return builder(work_dir, requests)
    except RequeueError as exc:
        if required:
            raise
        _add_warning(warnings, str(exc))
        return None


def _subset_dirs_for_step(step_id: str, subset_mode: str) -> list[str]:
    if subset_mode == "copy_all":
        return list(SUBSET_DIR_ORDER)
    if subset_mode == "copy_prev":
        return list(STEP_TO_SUBSET_PREV_DIRS.get(step_id, ["seg"]))
    return []


def _collect_subset_copy_targets(
    work_dir: Path,
    requests: dict[str, SceneRequest],
    subset_mode: str,
    warnings: list[str],
) -> dict[str, list[Path]]:
    targets: dict[str, list[Path]] = {rel_dir: [] for rel_dir in SUBSET_DIR_ORDER}
    seen: dict[str, set[Path]] = {rel_dir: set() for rel_dir in SUBSET_DIR_ORDER}
    for request in sorted(requests.values(), key=lambda item: (_scene_sort_key(item.scene_id), item.scene_id)):
        for rel_dir in _subset_dirs_for_step(request.step_id, subset_mode):
            folder = work_dir / rel_dir
            if not folder.is_dir():
                if not _suppress_missing_match_warning(rel_dir):
                    _add_warning(warnings, f"Subset source folder missing: {folder}")
                continue
            matches = _find_scene_matches(folder, rel_dir, request.scene_id)
            if not matches:
                if not _suppress_missing_match_warning(rel_dir):
                    _add_warning(
                        warnings,
                        f"No subset {DIR_LABELS[rel_dir]} files found for {request.scene_id}",
                    )
                continue
            for path in matches:
                if path not in seen[rel_dir]:
                    seen[rel_dir].add(path)
                    targets[rel_dir].append(path)
    for rel_dir in SUBSET_DIR_ORDER:
        targets[rel_dir].sort(key=lambda path: path.name)
    return targets


def _serialize_scene_requests(requests: dict[str, SceneRequest]) -> list[dict[str, object]]:
    items = []
    for request in sorted(requests.values(), key=lambda item: (_scene_sort_key(item.scene_id), item.scene_id)):
        items.append(
            {
                "scene_id": request.scene_id,
                "step_id": request.step_id,
                "sources": list(request.sources),
                "core_hints": sorted(request.core_hints),
            }
        )
    return items


def _build_subset_plan(
    work_dir: Path,
    config: PlanConfig,
    requests: dict[str, SceneRequest],
    file_targets: dict[str, list[Path]],
    sharpness_plan: CsvRewritePlan | None,
    autoct_plan: CsvRewritePlan | None,
    warnings: list[str],
) -> SubsetPlan | None:
    if config.subset_mode == "none":
        return None

    subset_dir = _create_subset_dir(work_dir)
    copy_targets = _collect_subset_copy_targets(work_dir, requests, config.subset_mode, warnings)
    purge_targets: dict[str, list[str]] = {rel_dir: [] for rel_dir in SUBSET_DIR_ORDER}
    purge_seen: dict[str, set[str]] = {rel_dir: set() for rel_dir in SUBSET_DIR_ORDER}

    for rel_dir in SUBSET_DIR_ORDER:
        combined = list(copy_targets.get(rel_dir, []))
        if rel_dir in file_targets:
            combined.extend(file_targets.get(rel_dir, []))
        relpaths: list[str] = []
        for path in combined:
            relpath = _relative_to_work_dir(work_dir, path)
            if relpath not in purge_seen[rel_dir]:
                purge_seen[rel_dir].add(relpath)
                relpaths.append(relpath)
        purge_targets[rel_dir] = sorted(relpaths)

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_work_dir": str(work_dir),
        "subset_mode": config.subset_mode,
        "delete_from_main": bool(config.delete_from_main),
        "requests": _serialize_scene_requests(requests),
        "copied_relpaths": {
            rel_dir: [
                _relative_to_work_dir(work_dir, path)
                for path in copy_targets.get(rel_dir, [])
            ]
            for rel_dir in SUBSET_DIR_ORDER
            if copy_targets.get(rel_dir)
        },
        "purge_relpaths": {
            rel_dir: list(paths)
            for rel_dir, paths in purge_targets.items()
            if paths
        },
        "sharpness_scope_names": list(sharpness_plan.match_names) if sharpness_plan else [],
        "autoct_scope_videos": list(autoct_plan.match_names) if autoct_plan else [],
    }
    return SubsetPlan(
        mode=config.subset_mode,
        subset_dir=subset_dir,
        copy_targets=copy_targets,
        purge_targets=purge_targets,
        manifest=manifest,
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise RequeueError(f"Failed to read JSON file {path}: {exc}") from exc


def _normalize_scope_names(names: list[str]) -> set[str]:
    out = set()
    for name in names:
        txt = str(name or "").strip()
        if txt:
            out.add(Path(txt).name)
    return out


def _row_key(row: dict[str, str], key_fields: tuple[str, ...]) -> tuple[str, ...] | None:
    parts: list[str] = []
    for field in key_fields:
        value = str((row or {}).get(field, "")).strip()
        if not value:
            return None
        parts.append(value)
    return tuple(parts)


def _replace_csv_rows_by_scope(
    dst_csv: Path,
    src_csv: Path,
    key_fields: tuple[str, ...],
    scope_field: str,
    scope_names: list[str],
) -> int:
    if not src_csv.is_file():
        return 0

    src_fieldnames, src_rows = _load_csv_rows(src_csv)
    if scope_field not in set(src_fieldnames):
        raise RequeueError(f"Column '{scope_field}' not found in {src_csv}")
    if any(field not in set(src_fieldnames) for field in key_fields):
        missing = ", ".join(field for field in key_fields if field not in set(src_fieldnames))
        raise RequeueError(f"Column(s) {missing} not found in {src_csv}")

    remove_names = _normalize_scope_names(scope_names)
    for row in src_rows:
        basename = Path(str(row.get(scope_field, "")).strip()).name
        if basename:
            remove_names.add(basename)

    dst_fieldnames: list[str] = []
    dst_rows: list[dict[str, str]] = []
    if dst_csv.is_file():
        dst_fieldnames, dst_rows = _load_csv_rows(dst_csv)

    out_fieldnames = list(dst_fieldnames or src_fieldnames)
    for field in src_fieldnames:
        if field not in out_fieldnames:
            out_fieldnames.append(field)
    for field in reversed(key_fields):
        if field in out_fieldnames:
            out_fieldnames = [item for item in out_fieldnames if item != field]
        out_fieldnames.insert(0, field)

    kept_unkeyed: list[dict[str, str]] = []
    merged_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    write_order: list[tuple[str, ...]] = []
    for row in dst_rows:
        scope_basename = Path(str(row.get(scope_field, "")).strip()).name
        if scope_basename and scope_basename in remove_names:
            continue
        key = _row_key(row, key_fields)
        if key is None:
            kept_unkeyed.append(row)
            continue
        if key not in merged_by_key:
            write_order.append(key)
        merged_by_key[key] = row

    src_unkeyed: list[dict[str, str]] = []
    for row in src_rows:
        key = _row_key(row, key_fields)
        if key is None:
            src_unkeyed.append(row)
            continue
        if key not in merged_by_key:
            write_order.append(key)
        merged_by_key[key] = row

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with dst_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in kept_unkeyed:
            writer.writerow({field: row.get(field, "") for field in out_fieldnames})
        for key in write_order:
            row = dict(merged_by_key.get(key, {}))
            writer.writerow({field: row.get(field, "") for field in out_fieldnames})
        for row in src_unkeyed:
            writer.writerow({field: row.get(field, "") for field in out_fieldnames})
    return len(src_rows)


def build_restore_plan(work_dir: Path, subset_dir: Path, log: LogFn) -> RestorePlan:
    work_root = work_dir.resolve()
    subset_root = subset_dir.resolve()
    if not work_root.is_dir():
        raise RequeueError(f"Work folder not found: {work_root}")
    if not subset_root.is_dir():
        raise RequeueError(f"Subset folder not found: {subset_root}")

    manifest_path = subset_root / ".requeue_subset_manifest.json"
    if not manifest_path.is_file():
        raise RequeueError(f"Subset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)

    source_work_dir = Path(str(manifest.get("source_work_dir", "")).strip() or ".").resolve()
    if source_work_dir != work_root:
        raise RequeueError(
            f"Subset was created from {source_work_dir}, but current work folder is {work_root}"
        )

    purge_targets: dict[str, list[Path]] = {rel_dir: [] for rel_dir in SUBSET_DIR_ORDER}
    raw_purge = manifest.get("purge_relpaths", {})
    if isinstance(raw_purge, dict):
        for rel_dir, relpaths in raw_purge.items():
            if rel_dir not in purge_targets or not isinstance(relpaths, list):
                continue
            resolved: list[Path] = []
            for relpath in relpaths:
                txt = str(relpath or "").strip()
                if not txt:
                    continue
                resolved.append((work_root / txt).resolve())
            purge_targets[rel_dir] = resolved

    copy_targets: dict[str, list[Path]] = {rel_dir: [] for rel_dir in SUBSET_DIR_ORDER}
    for rel_dir in SUBSET_DIR_ORDER:
        folder = subset_root / rel_dir
        if not folder.is_dir():
            continue
        files = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        copy_targets[rel_dir] = files
        if files:
            log(f"[SCAN] Restore subset {rel_dir}: {len(files)} file(s)")

    return RestorePlan(
        work_dir=work_root,
        subset_dir=subset_root,
        purge_targets=purge_targets,
        copy_targets=copy_targets,
        sharpness_csv=(subset_root / "sharpness.csv") if (subset_root / "sharpness.csv").is_file() else None,
        autoct_csv=(subset_root / "autoct.csv") if (subset_root / "autoct.csv").is_file() else None,
        sharpness_scope_names=[
            str(x)
            for x in (manifest.get("sharpness_scope_names") or [])
            if str(x).strip()
        ],
        autoct_scope_names=[
            str(x)
            for x in (manifest.get("autoct_scope_videos") or [])
            if str(x).strip()
        ],
    )


def build_execution_plan(config: PlanConfig, log: LogFn) -> ExecutionPlan:
    work_dir = config.work_dir.resolve()
    if not work_dir.is_dir():
        raise RequeueError(f"Work folder not found: {work_dir}")

    from_step = _canonical_step(config.csv_from_step)
    subset_mode = _canonical_subset_mode(config.subset_mode)

    requests: dict[str, SceneRequest] = {}
    warnings: list[str] = []
    csv_click_count = 0
    textbox_line_count = 0

    nonempty_textbox = [line for line in config.textbox_names if str(line or "").strip()]
    if config.annotation_csv is None and not nonempty_textbox:
        raise RequeueError("Please provide an annotation CSV, textbox basenames, or both.")

    csv_resolution_mode = "frame"
    if config.annotation_csv is not None:
        csv_path = config.annotation_csv.resolve()
        csv_resolution_mode = _annotation_csv_resolution_mode(csv_path)
        sbs_index: list[SbsSceneClip] | None = None
        if csv_resolution_mode == "frame":
            if shutil.which("ffprobe") is None:
                raise RequeueError("ffprobe not found in PATH.")
            if not (work_dir / "sbs").is_dir():
                raise RequeueError(f"SBS folder not found: {work_dir / 'sbs'}")
            sbs_index = _build_sbs_index(work_dir, log, warnings)
        else:
            log(
                f"[SCAN] Annotation CSV contains per-scene basenames; using direct scene-name resolution from {csv_path.name}"
            )
        csv_click_count = _parse_annotation_csv(
            csv_path,
            from_step,
            csv_resolution_mode,
            sbs_index,
            requests,
            log,
        )

    textbox_line_count = _parse_textbox_requests(nonempty_textbox, requests, log)

    if not requests:
        raise RequeueError("No valid scene requests were resolved from the provided inputs.")

    normalized_config = PlanConfig(
        work_dir=work_dir,
        annotation_csv=config.annotation_csv.resolve() if config.annotation_csv else None,
        csv_from_step=from_step,
        delete_from_main=bool(config.delete_from_main),
        subset_mode=subset_mode,
        textbox_names=nonempty_textbox,
        remove_sharpness_rows=bool(config.remove_sharpness_rows),
        remove_autoct_rows=bool(config.remove_autoct_rows),
    )

    file_targets = _collect_file_targets(work_dir, requests, warnings)
    need_sharpness_subset = any(
        _step_needs_sharpness_subset_rows(request.step_id) for request in requests.values()
    )
    need_sharpness_delete = any(
        _step_invalidates_sharpness_rows(request.step_id) for request in requests.values()
    )
    need_autoct = any(_step_needs_autoct_rows(request.step_id) for request in requests.values())

    sharpness_plan = (
        _maybe_build_csv_rewrite_plan(
            lambda wd, reqs: _build_sharpness_rewrite_plan(
                wd,
                reqs,
                step_filter=_step_needs_sharpness_subset_rows,
            ),
            work_dir,
            requests,
            required=False,
            warnings=warnings,
        )
        if need_sharpness_subset and normalized_config.subset_mode != "none"
        else None
    )
    sharpness_delete_plan = (
        _maybe_build_csv_rewrite_plan(
            lambda wd, reqs: _build_sharpness_rewrite_plan(
                wd,
                reqs,
                step_filter=_step_invalidates_sharpness_rows,
            ),
            work_dir,
            requests,
            required=bool(normalized_config.delete_from_main and normalized_config.remove_sharpness_rows),
            warnings=warnings,
        )
        if need_sharpness_delete and normalized_config.remove_sharpness_rows
        else None
    )
    autoct_plan = (
        _maybe_build_csv_rewrite_plan(
            lambda wd, reqs: _build_autoct_rewrite_plan(
                wd,
                reqs,
                step_filter=_step_needs_autoct_rows,
            ),
            work_dir,
            requests,
            required=bool(
                normalized_config.delete_from_main
                and normalized_config.remove_autoct_rows
                and need_autoct
            ),
            warnings=warnings,
        )
        if need_autoct and (normalized_config.remove_autoct_rows or normalized_config.subset_mode != "none")
        else None
    )
    subset_plan = _build_subset_plan(
        work_dir,
        normalized_config,
        requests,
        file_targets,
        sharpness_plan,
        autoct_plan,
        warnings,
    )

    delete_file_count = (
        sum(len(paths) for paths in file_targets.values())
        if normalized_config.delete_from_main
        else 0
    )
    delete_csv_rows = 0
    if (
        normalized_config.delete_from_main
        and normalized_config.remove_sharpness_rows
        and sharpness_delete_plan is not None
    ):
        delete_csv_rows += sharpness_delete_plan.removed_count
    if normalized_config.delete_from_main and normalized_config.remove_autoct_rows and autoct_plan is not None:
        delete_csv_rows += autoct_plan.removed_count

    subset_file_count = 0
    subset_csv_rows = 0
    if subset_plan is not None:
        subset_file_count = sum(len(paths) for paths in subset_plan.copy_targets.values())
        if sharpness_plan is not None:
            subset_csv_rows += sharpness_plan.removed_count
        if autoct_plan is not None:
            subset_csv_rows += autoct_plan.removed_count

    if delete_file_count <= 0 and delete_csv_rows <= 0 and subset_file_count <= 0 and subset_csv_rows <= 0:
        raise RequeueError("Nothing to do: no matching files, subset content, or CSV rows were found.")

    return ExecutionPlan(
        config=normalized_config,
        csv_click_count=csv_click_count,
        textbox_line_count=textbox_line_count,
        unique_scene_count=len(requests),
        scene_requests=requests,
        file_targets=file_targets,
        sharpness_plan=sharpness_plan,
        sharpness_delete_plan=sharpness_delete_plan,
        autoct_plan=autoct_plan,
        subset_plan=subset_plan,
        warnings=warnings,
    )

def _rewrite_csv(plan: CsvRewritePlan) -> int:
    if plan.removed_count <= 0:
        return 0
    tmp_path = plan.path.with_name(plan.path.name + ".tmp")
    _write_csv_rows(tmp_path, plan.fieldnames, plan.kept_rows)
    tmp_path.replace(plan.path)
    return plan.removed_count


def execute_plan(plan: ExecutionPlan, log: LogFn) -> ExecutionResult:
    result = ExecutionResult()
    work_dir = plan.config.work_dir

    if plan.subset_plan is not None:
        subset_root = plan.subset_plan.subset_dir
        if subset_root.exists():
            raise RequeueError(f"Subset folder already exists: {subset_root}")
        subset_root.mkdir(parents=True, exist_ok=False)
        result.subset_dir = str(subset_root)
        for rel_dir in SUBSET_DIR_ORDER:
            paths = plan.subset_plan.copy_targets.get(rel_dir, [])
            if not paths:
                continue
            dst_root = subset_root / rel_dir
            ok_count = 0
            err_count = 0
            log(f"[RUN ] subset copy {len(paths)} file(s) from {DIR_LABELS[rel_dir]}")
            for path in paths:
                dst = dst_root / path.name
                if _copy_or_replace_file(path, dst):
                    ok_count += 1
                    result.subset_file_count += 1
                else:
                    err_count += 1
                    result.file_error_count += 1
                    log(f"[ERR ] subset copy failed: {path} -> {dst}")
            log(f"[DONE] subset {DIR_LABELS[rel_dir]} ok={ok_count} errors={err_count}")
        if plan.sharpness_plan is not None and plan.sharpness_plan.removed_rows:
            _write_csv_rows(
                subset_root / "sharpness.csv",
                plan.sharpness_plan.fieldnames,
                plan.sharpness_plan.removed_rows,
            )
            log(f"[DONE] subset sharpness.csv rows={plan.sharpness_plan.removed_count}")
        if plan.autoct_plan is not None and plan.autoct_plan.removed_rows:
            _write_csv_rows(
                subset_root / "autoct.csv",
                plan.autoct_plan.fieldnames,
                plan.autoct_plan.removed_rows,
            )
            log(f"[DONE] subset autoct.csv rows={plan.autoct_plan.removed_count}")
        manifest_path = subset_root / ".requeue_subset_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(plan.subset_plan.manifest, handle, indent=2, sort_keys=True)
        log(f"[DONE] subset manifest: {manifest_path}")

    if plan.config.delete_from_main:
        for rel_dir in DIR_ORDER:
            paths = plan.file_targets.get(rel_dir, [])
            if not paths:
                continue
            ok_count = 0
            err_count = 0
            log(f"[RUN ] delete {len(paths)} file(s) in {DIR_LABELS[rel_dir]}")
            for path in paths:
                try:
                    if not path.exists():
                        warning = f"Skipping missing file: {path}"
                        result.warnings.append(warning)
                        log(f"[WARN] {warning}")
                        continue
                    path.unlink()
                    ok_count += 1
                    result.file_action_count += 1
                except Exception as exc:
                    err_count += 1
                    result.file_error_count += 1
                    log(f"[ERR ] {path}: {exc}")
            log(f"[DONE] {DIR_LABELS[rel_dir]} ok={ok_count} errors={err_count}")

    if (
        plan.config.delete_from_main
        and plan.config.remove_sharpness_rows
        and plan.sharpness_delete_plan is not None
    ):
        removed = _rewrite_csv(plan.sharpness_delete_plan)
        result.sharpness_removed = removed
        log(f"[DONE] sharpness.csv rows removed={removed}")
    if plan.config.delete_from_main and plan.config.remove_autoct_rows and plan.autoct_plan is not None:
        removed = _rewrite_csv(plan.autoct_plan)
        result.autoct_removed = removed
        log(f"[DONE] autoct.csv rows removed={removed}")

    return result


def _build_result_summary(plan: ExecutionPlan, result: ExecutionResult) -> str:
    lines = [
        f"Delete from main: {'yes' if plan.config.delete_from_main else 'no'}",
        f"Subset mode: {SUBSET_MODE_ID_TO_LABEL[plan.config.subset_mode]}",
        f"Subset files copied: {result.subset_file_count}",
        f"Main files deleted: {result.file_action_count}",
        f"File errors: {result.file_error_count}",
        f"sharpness.csv rows removed: {result.sharpness_removed}",
        f"autoct.csv rows removed: {result.autoct_removed}",
    ]
    if result.subset_dir:
        lines.append(f"Subset folder: {result.subset_dir}")
    if result.warnings:
        lines.append(f"Warnings: {len(result.warnings)}")
    return "\n".join(lines)


def execute_restore_plan(plan: RestorePlan, log: LogFn) -> RestoreResult:
    result = RestoreResult()
    for rel_dir in SUBSET_DIR_ORDER:
        paths = plan.purge_targets.get(rel_dir, [])
        if not paths:
            continue
        ok_count = 0
        err_count = 0
        log(f"[RUN ] purge {len(paths)} existing file(s) from {DIR_LABELS[rel_dir]}")
        for path in paths:
            try:
                if not path.exists():
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                ok_count += 1
                result.purged_file_count += 1
            except Exception as exc:
                err_count += 1
                result.file_error_count += 1
                log(f"[ERR ] purge failed: {path}: {exc}")
        log(f"[DONE] purge {DIR_LABELS[rel_dir]} ok={ok_count} errors={err_count}")

    for rel_dir in SUBSET_DIR_ORDER:
        paths = plan.copy_targets.get(rel_dir, [])
        if not paths:
            continue
        dst_root = plan.work_dir / rel_dir
        ok_count = 0
        err_count = 0
        log(f"[RUN ] restore copy {len(paths)} file(s) into {DIR_LABELS[rel_dir]}")
        for path in paths:
            dst = dst_root / path.name
            if _copy_or_replace_file(path, dst):
                ok_count += 1
                result.copied_file_count += 1
            else:
                err_count += 1
                result.file_error_count += 1
                log(f"[ERR ] restore copy failed: {path} -> {dst}")
        log(f"[DONE] restore {DIR_LABELS[rel_dir]} ok={ok_count} errors={err_count}")

    if plan.sharpness_csv is not None:
        replaced = _replace_csv_rows_by_scope(
            plan.work_dir / "sharpness.csv",
            plan.sharpness_csv,
            ("file",),
            "file",
            plan.sharpness_scope_names,
        )
        result.sharpness_replaced = replaced
        log(f"[DONE] sharpness.csv rows replaced={replaced}")
    if plan.autoct_csv is not None:
        replaced = _replace_csv_rows_by_scope(
            plan.work_dir / "autoct.csv",
            plan.autoct_csv,
            ("video", "frame"),
            "video",
            plan.autoct_scope_names,
        )
        result.autoct_replaced = replaced
        log(f"[DONE] autoct.csv rows replaced={replaced}")

    return result


def _build_restore_result_summary(plan: RestorePlan, result: RestoreResult) -> str:
    lines = [
        f"Restored subset: {plan.subset_dir}",
        f"Purged main files: {result.purged_file_count}",
        f"Copied back files: {result.copied_file_count}",
        f"File errors: {result.file_error_count}",
        f"sharpness.csv rows replaced: {result.sharpness_replaced}",
        f"autoct.csv rows replaced: {result.autoct_replaced}",
    ]
    if result.warnings:
        lines.append(f"Warnings: {len(result.warnings)}")
    return "\n".join(lines)


class RequeueAnnotatedScenesGUI(tk.Tk):
    POLL_INTERVAL_MS = 100

    def __init__(self) -> None:
        super().__init__()
        self.title("Requeue Annotated Scenes")
        self.geometry("1080x860")
        self.minsize(920, 720)

        self.work_dir_var = tk.StringVar(value="./work")
        self.annotation_csv_var = tk.StringVar(value="")
        self.subset_dir_var = tk.StringVar(value="")
        self.from_step_var = tk.StringVar(value="splatting")
        self.delete_main_var = tk.BooleanVar(value=True)
        self.subset_mode_var = tk.StringVar(value="Do Not create subset")
        self.remove_sharpness_var = tk.BooleanVar(value=True)
        self.remove_autoct_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")

        self._message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False

        self._build_ui()
        self.after(self.POLL_INTERVAL_MS, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(5, weight=1)
        root.rowconfigure(7, weight=1)

        ttk.Label(root, text="Work folder:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Entry(root, textvariable=self.work_dir_var).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(root, text="Browse", command=self._browse_work_dir).grid(row=0, column=2, pady=(0, 6))

        ttk.Label(root, text="Annotation CSV:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Entry(root, textvariable=self.annotation_csv_var).grid(row=1, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(root, text="Browse", command=self._browse_annotation_csv).grid(row=1, column=2, pady=(0, 6))

        ttk.Label(root, text="Subset folder:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Entry(root, textvariable=self.subset_dir_var).grid(row=2, column=1, sticky="ew", pady=(0, 6))
        self.subset_browse_button = ttk.Button(root, text="Browse", command=self._browse_subset_dir)
        self.subset_browse_button.grid(row=2, column=2, pady=(0, 6))

        options = ttk.Frame(root)
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Subset mode:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.subset_mode_combo = ttk.Combobox(
            options,
            textvariable=self.subset_mode_var,
            state="readonly",
            values=[label for _mode_id, label in SUBSET_MODE_OPTIONS],
            width=34,
        )
        self.subset_mode_combo.grid(row=0, column=1, sticky="w", padx=(0, 18))

        ttk.Label(options, text="From step (CSV only):").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.step_combo = ttk.Combobox(
            options,
            textvariable=self.from_step_var,
            state="readonly",
            values=[label for _step_id, label in STEP_OPTIONS],
            width=18,
        )
        self.step_combo.grid(row=0, column=3, sticky="w")

        checks = ttk.Frame(root)
        checks.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self.delete_main_check = ttk.Checkbutton(
            checks,
            text="Delete from main workdir",
            variable=self.delete_main_var,
        )
        self.delete_main_check.pack(side=tk.LEFT, padx=(0, 16))
        self.remove_sharpness_check = ttk.Checkbutton(
            checks,
            text="Remove sharpness rows",
            variable=self.remove_sharpness_var,
        )
        self.remove_sharpness_check.pack(side=tk.LEFT, padx=(0, 16))
        self.remove_autoct_check = ttk.Checkbutton(
            checks,
            text="Remove autoct rows",
            variable=self.remove_autoct_var,
        )
        self.remove_autoct_check.pack(side=tk.LEFT)

        text_frame = ttk.LabelFrame(root, text="Basenames (one per line)")
        text_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        ttk.Label(
            text_frame,
            text=(
                "Optional input. Valid basenames: *_depth.*, *_splatted*, "
                "*_inpainted_right_eye.*, *_replace_mask.*, *_merged_full_sbs.* "
                "Step is inferred from the typed basename."
            ),
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 4))
        self.names_text = ScrolledText(text_frame, wrap=tk.NONE, height=12)
        self.names_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        buttons = ttk.Frame(root)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        buttons.columnconfigure(5, weight=1)
        self.run_button = ttk.Button(buttons, text="Run", command=self._on_run)
        self.run_button.grid(row=0, column=0, padx=(0, 8))
        self.restore_button = ttk.Button(buttons, text="Restore subset (overwrite)", command=self._on_restore)
        self.restore_button.grid(row=0, column=1, padx=(0, 8))
        self.rejoin_button = ttk.Button(buttons, text="Rejoin in-place", command=self._on_rejoin_in_place)
        self.rejoin_button.grid(row=0, column=2, padx=(0, 8))
        self.compare_join_button = ttk.Button(buttons, text="Compare join", command=self._on_compare_join)
        self.compare_join_button.grid(row=0, column=3, padx=(0, 8))
        self.clear_log_button = ttk.Button(buttons, text="Clear log", command=self._clear_log)
        self.clear_log_button.grid(row=0, column=4, padx=(0, 8))
        ttk.Label(buttons, textvariable=self.status_var).grid(row=0, column=5, sticky="e")

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=14, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _browse_work_dir(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.work_dir_var.get().strip() or ".")
        if folder:
            self.work_dir_var.set(folder)

    def _browse_annotation_csv(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(Path(self.annotation_csv_var.get().strip() or ".").parent),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.annotation_csv_var.set(path)

    def _browse_subset_dir(self) -> None:
        folder = filedialog.askdirectory(
            initialdir=self.subset_dir_var.get().strip() or self.work_dir_var.get().strip() or "."
        )
        if folder:
            self.subset_dir_var.set(folder)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self.status_var.set(status)
        widgets: list[tk.Widget] = [
            self.run_button,
            self.restore_button,
            self.rejoin_button,
            self.compare_join_button,
            self.clear_log_button,
            self.subset_browse_button,
            self.subset_mode_combo,
            self.step_combo,
            self.delete_main_check,
            self.remove_sharpness_check,
            self.remove_autoct_check,
        ]
        state = tk.DISABLED if busy else tk.NORMAL
        combo_state = "disabled" if busy else "readonly"
        for widget in widgets:
            try:
                widget.configure(state=state)
            except Exception:
                pass
        self.subset_mode_combo.configure(state=combo_state)
        self.step_combo.configure(state=combo_state)

    def _build_config(self) -> PlanConfig:
        work_dir = Path(self.work_dir_var.get().strip() or "./work")
        annotation_csv_raw = self.annotation_csv_var.get().strip()
        annotation_csv = Path(annotation_csv_raw) if annotation_csv_raw else None
        textbox_names = self.names_text.get("1.0", tk.END).splitlines()
        return PlanConfig(
            work_dir=work_dir,
            annotation_csv=annotation_csv,
            csv_from_step=self.from_step_var.get().strip() or "splatting",
            delete_from_main=bool(self.delete_main_var.get()),
            subset_mode=self.subset_mode_var.get().strip() or "Do Not create subset",
            textbox_names=textbox_names,
            remove_sharpness_rows=bool(self.remove_sharpness_var.get()),
            remove_autoct_rows=bool(self.remove_autoct_var.get()),
        )

    def _on_run(self) -> None:
        if self._busy:
            messagebox.showwarning("Busy", "An operation is already running.")
            return
        config = self._build_config()
        has_csv = config.annotation_csv is not None
        has_text = any(str(line or "").strip() for line in config.textbox_names)
        if not has_csv and not has_text:
            messagebox.showerror(
                "Missing Input",
                "Provide an annotation CSV, one or more textbox basenames, or both.",
            )
            return

        self._append_log("[INFO] Building execution plan...")
        self._set_busy(True, "Preparing preview...")
        worker = threading.Thread(target=self._worker_run, args=(config,), daemon=True)
        worker.start()

    def _resolve_subset_dir_from_ui(self, title: str) -> Path | None:
        subset_raw = self.subset_dir_var.get().strip()
        if subset_raw:
            subset_dir = Path(subset_raw)
            if subset_dir.is_dir():
                return subset_dir
            raise RequeueError(f"Subset folder not found: {subset_dir}")
        initial_dir = self.work_dir_var.get().strip() or "."
        chosen = filedialog.askdirectory(title=title, initialdir=initial_dir)
        if not chosen:
            return None
        self.subset_dir_var.set(chosen)
        return Path(chosen)

    def _on_restore(self) -> None:
        if self._busy:
            messagebox.showwarning("Busy", "An operation is already running.")
            return
        try:
            subset_dir = self._resolve_subset_dir_from_ui("Select subset folder to restore")
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return
        if subset_dir is None:
            return
        work_dir = Path(self.work_dir_var.get().strip() or "./work")
        self._append_log("[INFO] Building restore plan...")
        self._set_busy(True, "Preparing restore preview...")
        worker = threading.Thread(
            target=self._worker_restore,
            args=(work_dir, subset_dir),
            daemon=True,
        )
        worker.start()

    def _on_rejoin_in_place(self) -> None:
        if self._busy:
            messagebox.showwarning("Busy", "An operation is already running.")
            return
        try:
            subset_dir = self._resolve_subset_dir_from_ui("Select subset folder for rejoin")
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return
        if subset_dir is None:
            return
        work_dir = Path(self.work_dir_var.get().strip() or "./work")
        self._append_log("[INFO] Starting rejoin in-place...")
        self._set_busy(True, "Rejoining...")
        worker = threading.Thread(
            target=self._worker_join_action,
            args=("rejoin_in_place", work_dir, subset_dir),
            daemon=True,
        )
        worker.start()

    def _on_compare_join(self) -> None:
        if self._busy:
            messagebox.showwarning("Busy", "An operation is already running.")
            return
        try:
            subset_dir = self._resolve_subset_dir_from_ui("Select subset folder for compare join")
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)
            return
        if subset_dir is None:
            return
        work_dir = Path(self.work_dir_var.get().strip() or "./work")
        self._append_log("[INFO] Starting compare join...")
        self._set_busy(True, "Compare joining...")
        worker = threading.Thread(
            target=self._worker_join_action,
            args=("compare_join", work_dir, subset_dir),
            daemon=True,
        )
        worker.start()

    def _worker_run(self, config: PlanConfig) -> None:
        def log(message: str) -> None:
            self._message_queue.put(("log", message))

        try:
            plan = build_execution_plan(config, log)
            confirm = ConfirmRequest(
                title="Confirm Requeue",
                message=plan.preview_text(),
            )
            self._message_queue.put(("confirm", confirm))
            confirm.event.wait()
            if not confirm.approved:
                self._message_queue.put(("done", (False, "Operation cancelled.")))
                return
            log("[INFO] Preview approved. Applying changes...")
            result = execute_plan(plan, log)
            summary = _build_result_summary(plan, result)
            ok = result.file_error_count == 0
            self._message_queue.put(("done", (ok, summary, result.subset_dir)))
        except Exception as exc:
            self._message_queue.put(("error", str(exc)))

    def _worker_restore(self, work_dir: Path, subset_dir: Path) -> None:
        def log(message: str) -> None:
            self._message_queue.put(("log", message))

        try:
            plan = build_restore_plan(work_dir, subset_dir, log)
            confirm = ConfirmRequest(
                title="Confirm Restore",
                message=plan.preview_text(),
            )
            self._message_queue.put(("confirm", confirm))
            confirm.event.wait()
            if not confirm.approved:
                self._message_queue.put(("restore_done", (False, "Restore cancelled.")))
                return
            log("[INFO] Restore preview approved. Applying restore...")
            result = execute_restore_plan(plan, log)
            summary = _build_restore_result_summary(plan, result)
            ok = result.file_error_count == 0
            self._message_queue.put(("restore_done", (ok, summary)))
        except Exception as exc:
            self._message_queue.put(("error", str(exc)))

    def _worker_join_action(self, action: str, work_dir: Path, subset_dir: Path) -> None:
        def log(message: str) -> None:
            self._message_queue.put(("log", message))

        title = "Rejoin in-place" if action == "rejoin_in_place" else "Compare join"
        try:
            settings = _load_join_settings()
            final_dir = subset_dir / "final"
            if action == "rejoin_in_place":
                sequence, subset_hits, work_hits = _build_rejoin_inplace_sequence(work_dir, subset_dir, log)
                out_path = (final_dir / _default_join_output_basename(settings.encoder)).resolve()
                log(
                    f"[RUN ] {title}: total_inputs={len(sequence)} subset_overrides={subset_hits} "
                    f"work_fallbacks={work_hits}"
                )
                _run_join_script_on_sequence(sequence, out_path, settings, log)
                summary = "\n".join(
                    [
                        f"Completed: {title}",
                        f"Output: {out_path}",
                        f"Inputs joined: {len(sequence)}",
                        f"Subset overrides used: {subset_hits}",
                        f"Original work clips used: {work_hits}",
                    ]
                )
            else:
                sequence, pair_count = _build_compare_sequence(work_dir, subset_dir, log)
                out_path = (final_dir / f"compare_{_default_join_output_basename(settings.encoder)}").resolve()
                log(f"[RUN ] {title}: matched_pairs={pair_count} concat_items={len(sequence)}")
                _run_join_script_on_sequence(sequence, out_path, settings, log)
                summary = "\n".join(
                    [
                        f"Completed: {title}",
                        f"Output: {out_path}",
                        f"Matched pairs: {pair_count}",
                        f"Concat items joined: {len(sequence)}",
                    ]
                )
            self._message_queue.put(("join_done", (True, title, summary, str(subset_dir))))
        except Exception as exc:
            self._message_queue.put(("join_done", (False, title, str(exc), str(subset_dir))))

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self._message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
                continue

            if kind == "confirm":
                confirm = payload
                assert isinstance(confirm, ConfirmRequest)
                self.status_var.set("Waiting for confirmation...")
                approved = messagebox.askokcancel(
                    confirm.title,
                    confirm.message,
                    parent=self,
                )
                confirm.approved = bool(approved)
                confirm.event.set()
                if approved:
                    self._append_log("[INFO] Preview confirmed by user.")
                    self.status_var.set("Running...")
                else:
                    self._append_log("[INFO] Operation cancelled at preview step.")
                continue

            if kind == "done":
                ok, summary, subset_dir = payload
                self._set_busy(False, "Ready")
                if subset_dir:
                    self.subset_dir_var.set(str(subset_dir))
                self._append_log("[INFO] " + str(summary).replace("\n", " | "))
                if ok:
                    messagebox.showinfo("Complete", str(summary), parent=self)
                else:
                    messagebox.showwarning("Stopped", str(summary), parent=self)
                continue

            if kind == "restore_done":
                ok, summary = payload
                self._set_busy(False, "Ready")
                self._append_log("[INFO] " + str(summary).replace("\n", " | "))
                if ok:
                    messagebox.showinfo("Restore Complete", str(summary), parent=self)
                else:
                    messagebox.showwarning("Restore Stopped", str(summary), parent=self)
                continue

            if kind == "join_done":
                ok, title, summary, subset_dir = payload
                self._set_busy(False, "Ready")
                if subset_dir:
                    self.subset_dir_var.set(str(subset_dir))
                self._append_log("[INFO] " + str(summary).replace("\n", " | "))
                if ok:
                    messagebox.showinfo(title, str(summary), parent=self)
                else:
                    messagebox.showwarning(f"{title} failed", str(summary), parent=self)
                continue

            if kind == "error":
                self._set_busy(False, "Ready")
                self._append_log(f"[ERR ] {payload}")
                messagebox.showerror("Error", str(payload), parent=self)
                continue

        self.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _on_close(self) -> None:
        if self._busy:
            messagebox.showwarning(
                "Busy",
                "An operation is still running. Wait for it to finish before closing the window.",
                parent=self,
            )
            return
        self.destroy()


def main() -> None:
    app = RequeueAnnotatedScenesGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
