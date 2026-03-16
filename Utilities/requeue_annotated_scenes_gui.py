#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import queue
import re
import shutil
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

STEP_OPTIONS = [
    ("depthcrafter", "depthcrafter"),
    ("realesrgan", "realesrgan"),
    ("splatting", "splatting"),
    ("inpainting", "inpainting"),
    ("mask_for_merge", "mask for merge"),
    ("merging", "merging"),
]
STEP_ID_TO_LABEL = {step_id: label for step_id, label in STEP_OPTIONS}
STEP_LABEL_TO_ID = {label: step_id for step_id, label in STEP_OPTIONS}
STEP_ORDER = [step_id for step_id, _label in STEP_OPTIONS]
STEP_INDEX = {step_id: idx for idx, step_id in enumerate(STEP_ORDER)}

STEP_TO_DIRS = {
    # "mask" is the replace-mask folder produced by splatting (work/mask).
    # It is distinct from "mask_for_merge", which is generated later by the
    # mask preprocessing step used by merge.
    "depthcrafter": ["depthmap", "depthmap/upscaled", "splat/hires", "mask", "output", "mask_for_merge", "sbs"],
    "realesrgan": ["depthmap/upscaled", "splat/hires", "mask", "output", "mask_for_merge", "sbs"],
    "splatting": ["splat/hires", "mask", "output", "mask_for_merge", "sbs"],
    "inpainting": ["output", "mask_for_merge", "sbs"],
    "mask_for_merge": ["mask_for_merge", "sbs"],
    "merging": ["sbs"],
}
DIR_ORDER = ["depthmap", "depthmap/upscaled", "splat/hires", "mask", "output", "mask_for_merge", "sbs"]
OLD_DIR_MAP = {
    "depthmap": "depthmap_old",
    "depthmap/upscaled": "depthmap/upscaled_old",
    "splat/hires": "splat/hires_old",
    "mask": "mask_old",
    "output": "output_old",
    "mask_for_merge": "mask_for_merge_old",
    "sbs": "sbs_old",
}
DIR_LABELS = {
    "depthmap": "depthmap",
    "depthmap/upscaled": "depthmap/upscaled",
    "splat/hires": "splat/hires",
    "mask": "mask",
    "output": "output",
    "mask_for_merge": "mask_for_merge",
    "sbs": "sbs",
}

SBS_FILENAME_RE = re.compile(
    r"^(?P<core>(?P<scene>source-Scene-\d+)_(?P<width>\d+))_merged_full_sbs\.(?P<ext>[^.]+)$",
    re.IGNORECASE,
)
DEPTH_FILENAME_RE = re.compile(
    r"^(?P<scene>source-Scene-\d+)_depth\.[^.]+$",
    re.IGNORECASE,
)
SPLAT_FILENAME_RE = re.compile(
    r"^(?P<core>(?P<scene>source-Scene-\d+)_\d+)_splatted[^.]*\.[^.]+$",
    re.IGNORECASE,
)
INPAINT_FILENAME_RE = re.compile(
    r"^(?P<core>(?P<scene>source-Scene-\d+)_\d+)_inpainted_right_eye\.[^.]+$",
    re.IGNORECASE,
)
MASK_FILENAME_RE = re.compile(
    r"^(?P<core>(?P<scene>source-Scene-\d+)_\d+)(?:_splatted[^.]*)?_replace_mask\.[^.]+$",
    re.IGNORECASE,
)
MERGED_FILENAME_RE = re.compile(
    r"^(?P<core>(?P<scene>source-Scene-\d+)_\d+)_merged_full_sbs\.[^.]+$",
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
    removed_count: int


@dataclass
class PlanConfig:
    work_dir: Path
    annotation_csv: Path | None
    csv_from_step: str
    action: str
    textbox_names: list[str]
    remove_sharpness_rows: bool
    remove_autoct_rows: bool


@dataclass
class ExecutionPlan:
    config: PlanConfig
    csv_click_count: int
    textbox_line_count: int
    unique_scene_count: int
    scene_requests: dict[str, SceneRequest]
    file_targets: dict[str, list[Path]]
    sharpness_plan: CsvRewritePlan | None
    autoct_plan: CsvRewritePlan | None
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

        sharp_rows = self.sharpness_plan.removed_count if self.sharpness_plan else 0
        auto_rows = self.autoct_plan.removed_count if self.autoct_plan else 0

        lines = [
            f"Action: {self.config.action}",
            f"Work folder: {self.config.work_dir}",
            f"CSV clicks: {self.csv_click_count}",
            f"Textbox basenames: {self.textbox_line_count}",
            f"Unique scenes: {self.unique_scene_count}",
            f"Scenes: {scene_preview or '(none)'}",
            "",
            "Files by folder:",
            *folder_lines,
            "",
            "CSV rows to remove:",
            f"- sharpness.csv: {sharp_rows}",
            f"- autoct.csv: {auto_rows}",
        ]
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
    file_action_count: int = 0
    file_error_count: int = 0
    sharpness_removed: int = 0
    autoct_removed: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConfirmRequest:
    plan: ExecutionPlan
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


def _scene_sort_key(scene_id: str) -> int:
    try:
        return int(scene_id.rsplit("-", 1)[-1])
    except Exception:
        return 10**9


def _canonical_step(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in STEP_ID_TO_LABEL:
        return raw
    if raw in STEP_LABEL_TO_ID:
        return STEP_LABEL_TO_ID[raw]
    raise RequeueError(f"Unsupported step: {value!r}")


def _canonical_action(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw not in {"move to old", "delete"}:
        raise RequeueError(f"Unsupported action: {value!r}")
    return raw


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
    sbs_index: list[SbsSceneClip],
    requests: dict[str, SceneRequest],
    log: LogFn,
) -> int:
    if not csv_path.is_file():
        raise RequeueError(f"Annotation CSV not found: {csv_path}")

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


def _find_scene_matches(folder: Path, rel_dir: str, scene_id: str) -> list[Path]:
    pattern_map = {
        "depthmap": [f"{scene_id}_depth.*"],
        "depthmap/upscaled": [f"{scene_id}_depth.*"],
        "splat/hires": [f"{scene_id}_*_splatted*.*"],
        "mask": [f"{scene_id}_*_replace_mask.*"],
        "output": [f"{scene_id}_*_inpainted_right_eye.*"],
        "mask_for_merge": [f"{scene_id}_*_replace_mask.*"],
        "sbs": [f"{scene_id}_*_merged_full_sbs.*"],
    }
    matches: dict[str, Path] = {}
    for pattern in pattern_map[rel_dir]:
        for path in _video_glob_matches(folder, pattern):
            matches[path.name] = path
    return [matches[name] for name in sorted(matches)]


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
                _add_warning(warnings, f"Folder missing: {folder}")
                continue
            matches = _find_scene_matches(folder, rel_dir, request.scene_id)
            if not matches:
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
        if "splat/hires" in STEP_TO_DIRS[request.step_id]
    ]
    kept_rows = []
    removed_count = 0
    for row in rows:
        basename = str(row.get("file", "")).strip()
        if basename and any(
            _scene_matches_prefixed_basename(basename, request, "_splatted")
            for request in impacted
        ):
            removed_count += 1
            continue
        kept_rows.append(row)
    return CsvRewritePlan(
        label="sharpness.csv",
        path=csv_path,
        fieldnames=fieldnames,
        rows=rows,
        kept_rows=kept_rows,
        removed_count=removed_count,
    )


def _build_autoct_rewrite_plan(
    work_dir: Path,
    requests: dict[str, SceneRequest],
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
        if "output" in STEP_TO_DIRS[request.step_id]
    ]
    kept_rows = []
    removed_count = 0
    for row in rows:
        basename = str(row.get("video", "")).strip()
        if basename and any(
            _scene_matches_prefixed_basename(basename, request, "_inpainted_right_eye")
            for request in impacted
        ):
            removed_count += 1
            continue
        kept_rows.append(row)
    return CsvRewritePlan(
        label="autoct.csv",
        path=csv_path,
        fieldnames=fieldnames,
        rows=rows,
        kept_rows=kept_rows,
        removed_count=removed_count,
    )


def build_execution_plan(config: PlanConfig, log: LogFn) -> ExecutionPlan:
    work_dir = config.work_dir.resolve()
    if not work_dir.is_dir():
        raise RequeueError(f"Work folder not found: {work_dir}")
    if shutil.which("ffprobe") is None:
        raise RequeueError("ffprobe not found in PATH.")
    if not (work_dir / "sbs").is_dir():
        raise RequeueError(f"SBS folder not found: {work_dir / 'sbs'}")

    action = _canonical_action(config.action)
    from_step = _canonical_step(config.csv_from_step)

    requests: dict[str, SceneRequest] = {}
    warnings: list[str] = []
    csv_click_count = 0
    textbox_line_count = 0

    nonempty_textbox = [line for line in config.textbox_names if str(line or "").strip()]
    if config.annotation_csv is None and not nonempty_textbox:
        raise RequeueError("Please provide an annotation CSV, textbox basenames, or both.")

    if config.annotation_csv is not None:
        sbs_index = _build_sbs_index(work_dir, log, warnings)
        csv_click_count = _parse_annotation_csv(
            config.annotation_csv.resolve(),
            from_step,
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
        action=action,
        textbox_names=nonempty_textbox,
        remove_sharpness_rows=bool(config.remove_sharpness_rows),
        remove_autoct_rows=bool(config.remove_autoct_rows),
    )

    file_targets = _collect_file_targets(work_dir, requests, warnings)
    sharpness_plan = (
        _build_sharpness_rewrite_plan(work_dir, requests)
        if normalized_config.remove_sharpness_rows
        else None
    )
    autoct_plan = (
        _build_autoct_rewrite_plan(work_dir, requests)
        if normalized_config.remove_autoct_rows
        else None
    )

    total_files = sum(len(paths) for paths in file_targets.values())
    total_csv_rows = 0
    if sharpness_plan is not None:
        total_csv_rows += sharpness_plan.removed_count
    if autoct_plan is not None:
        total_csv_rows += autoct_plan.removed_count
    if total_files <= 0 and total_csv_rows <= 0:
        raise RequeueError("Nothing to do: no matching files or CSV rows were found.")

    return ExecutionPlan(
        config=normalized_config,
        csv_click_count=csv_click_count,
        textbox_line_count=textbox_line_count,
        unique_scene_count=len(requests),
        scene_requests=requests,
        file_targets=file_targets,
        sharpness_plan=sharpness_plan,
        autoct_plan=autoct_plan,
        warnings=warnings,
    )


def _unique_old_destination(dst_dir: Path, src_name: str) -> Path:
    candidate = dst_dir / src_name
    if not candidate.exists():
        return candidate

    suffix = "".join(Path(src_name).suffixes)
    stem = src_name[: -len(suffix)] if suffix else src_name
    for idx in range(1, 10000):
        candidate = dst_dir / f"{stem}__old_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RequeueError(f"Could not find free destination name in {dst_dir} for {src_name}")


def _rewrite_csv(plan: CsvRewritePlan) -> int:
    if plan.removed_count <= 0:
        return 0
    tmp_path = plan.path.with_name(plan.path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=plan.fieldnames)
        writer.writeheader()
        for row in plan.kept_rows:
            writer.writerow({field: row.get(field, "") for field in plan.fieldnames})
    tmp_path.replace(plan.path)
    return plan.removed_count


def execute_plan(plan: ExecutionPlan, log: LogFn) -> ExecutionResult:
    result = ExecutionResult()
    action = plan.config.action
    work_dir = plan.config.work_dir

    for rel_dir in DIR_ORDER:
        paths = plan.file_targets.get(rel_dir, [])
        if not paths:
            continue
        ok_count = 0
        err_count = 0
        log(f"[RUN ] {action} {len(paths)} file(s) in {DIR_LABELS[rel_dir]}")
        dst_dir = None
        if action == "move to old":
            dst_dir = work_dir / OLD_DIR_MAP[rel_dir]
            dst_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            try:
                if not path.exists():
                    warning = f"Skipping missing file: {path}"
                    result.warnings.append(warning)
                    log(f"[WARN] {warning}")
                    continue
                if action == "move to old":
                    assert dst_dir is not None
                    dst_path = _unique_old_destination(dst_dir, path.name)
                    shutil.move(str(path), str(dst_path))
                else:
                    path.unlink()
                ok_count += 1
                result.file_action_count += 1
            except Exception as exc:
                err_count += 1
                result.file_error_count += 1
                log(f"[ERR ] {path}: {exc}")
        log(f"[DONE] {DIR_LABELS[rel_dir]} ok={ok_count} errors={err_count}")

    if plan.sharpness_plan is not None:
        removed = _rewrite_csv(plan.sharpness_plan)
        result.sharpness_removed = removed
        log(f"[DONE] sharpness.csv rows removed={removed}")
    if plan.autoct_plan is not None:
        removed = _rewrite_csv(plan.autoct_plan)
        result.autoct_removed = removed
        log(f"[DONE] autoct.csv rows removed={removed}")

    return result


def _build_result_summary(plan: ExecutionPlan, result: ExecutionResult) -> str:
    lines = [
        f"Completed action: {plan.config.action}",
        f"Files changed: {result.file_action_count}",
        f"File errors: {result.file_error_count}",
        f"sharpness.csv rows removed: {result.sharpness_removed}",
        f"autoct.csv rows removed: {result.autoct_removed}",
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
        self.action_var = tk.StringVar(value="move to old")
        self.from_step_var = tk.StringVar(value="splatting")
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
        root.rowconfigure(4, weight=1)
        root.rowconfigure(6, weight=1)

        ttk.Label(root, text="Work folder:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Entry(root, textvariable=self.work_dir_var).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(root, text="Browse", command=self._browse_work_dir).grid(row=0, column=2, pady=(0, 6))

        ttk.Label(root, text="Annotation CSV:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Entry(root, textvariable=self.annotation_csv_var).grid(row=1, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(root, text="Browse", command=self._browse_annotation_csv).grid(row=1, column=2, pady=(0, 6))

        options = ttk.Frame(root)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Action:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.action_combo = ttk.Combobox(
            options,
            textvariable=self.action_var,
            state="readonly",
            values=["move to old", "delete"],
            width=18,
        )
        self.action_combo.grid(row=0, column=1, sticky="w", padx=(0, 18))

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
        checks.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Checkbutton(
            checks,
            text="Remove sharpness rows",
            variable=self.remove_sharpness_var,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            checks,
            text="Remove autoct rows",
            variable=self.remove_autoct_var,
        ).pack(side=tk.LEFT)

        text_frame = ttk.LabelFrame(root, text="Basenames (one per line)")
        text_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        ttk.Label(
            text_frame,
            text=(
                "Optional input. Valid basenames: *_depth.*, *_splatted*, "
                "*_inpainted_right_eye.*, *_replace_mask.*, *_merged_full_sbs.*"
            ),
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 4))
        self.names_text = ScrolledText(text_frame, wrap=tk.NONE, height=12)
        self.names_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        buttons = ttk.Frame(root)
        buttons.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        buttons.columnconfigure(2, weight=1)
        self.run_button = ttk.Button(buttons, text="Run", command=self._on_run)
        self.run_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Clear log", command=self._clear_log).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(buttons, textvariable=self.status_var).grid(row=0, column=2, sticky="e")

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
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
            self.action_combo,
            self.step_combo,
        ]
        state = tk.DISABLED if busy else tk.NORMAL
        combo_state = "disabled" if busy else "readonly"
        self.run_button.configure(state=state)
        self.action_combo.configure(state=combo_state)
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
            action=self.action_var.get().strip() or "move to old",
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

    def _worker_run(self, config: PlanConfig) -> None:
        def log(message: str) -> None:
            self._message_queue.put(("log", message))

        try:
            plan = build_execution_plan(config, log)
            confirm = ConfirmRequest(plan=plan)
            self._message_queue.put(("confirm", confirm))
            confirm.event.wait()
            if not confirm.approved:
                self._message_queue.put(("done", (False, "Operation cancelled.")))
                return
            log("[INFO] Preview approved. Applying changes...")
            result = execute_plan(plan, log)
            summary = _build_result_summary(plan, result)
            ok = result.file_error_count == 0
            self._message_queue.put(("done", (ok, summary)))
        except Exception as exc:
            self._message_queue.put(("error", str(exc)))

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
                    "Confirm Requeue",
                    confirm.plan.preview_text(),
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
                ok, summary = payload
                self._set_busy(False, "Ready")
                self._append_log("[INFO] " + str(summary).replace("\n", " | "))
                if ok:
                    messagebox.showinfo("Complete", str(summary), parent=self)
                else:
                    messagebox.showwarning("Stopped", str(summary), parent=self)
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
