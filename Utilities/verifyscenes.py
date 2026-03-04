#!/usr/bin/env python3
"""Verify scene files against a reference folder or a single source file.

Checks each target video for:
1) file size sanity
2) ffprobe readability
3) full decode without ffmpeg errors
4) parity checks:
   - folder mode: frame parity with matching reference clip
   - single-file mode: final total frame parity vs source file
   - duration is collected/reported only as informational metadata

Usage:
  python verifyscenes.py /path/to/targets /path/to/reference_folder
  python verifyscenes.py /path/to/targets /path/to/source_file.mkv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ProbeInfo:
    path: Path
    size_bytes: int
    nb_frames: Optional[int]
    duration_sec: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    error: Optional[str]


@dataclass
class TargetTask:
    idx: int
    target: Path
    ref_path: Optional[Path]
    match_info: str
    pre_reasons: List[str]


@dataclass
class TargetResult:
    idx: int
    target: Path
    ref_path: Optional[Path]
    match_info: str
    probe: ProbeInfo
    reasons: List[str]


def _run_cmd(cmd: Sequence[str], timeout_sec: Optional[float] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=(float(timeout_sec) if timeout_sec and float(timeout_sec) > 0 else None),
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        cmd0 = cmd[0] if cmd else "cmd"
        return 124, (e.stdout or "").strip() if isinstance(e.stdout, str) else "", f"{cmd0} timeout after {timeout_sec}s"
    except FileNotFoundError as e:
        return 127, "", str(e)
    except Exception as e:
        return 126, "", str(e)


def _parse_int(v) -> Optional[int]:
    try:
        if v in (None, "", "N/A"):
            return None
        return int(float(v))
    except Exception:
        return None


def _parse_float(v) -> Optional[float]:
    try:
        if v in (None, "", "N/A"):
            return None
        return float(v)
    except Exception:
        return None


def _parse_rate(v) -> Optional[float]:
    if v in (None, "", "N/A"):
        return None
    try:
        return float(Fraction(str(v)))
    except Exception:
        return _parse_float(v)


def probe_video(
    path: Path,
    probe_timeout_sec: float = 20.0,
    probe_timeout_retries: int = 1,
    count_frames: bool = True,
) -> ProbeInfo:
    if not path.exists() or not path.is_file():
        return ProbeInfo(path, 0, None, None, None, None, None, "file not found")

    try:
        size_bytes = int(path.stat().st_size)
    except Exception as e:
        return ProbeInfo(path, 0, None, None, None, None, None, f"stat failed: {e}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if bool(count_frames):
        cmd.append("-count_frames")
    cmd += [
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate,width,height,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    max_retries = max(0, int(probe_timeout_retries))
    total_attempts = 1 + max_retries
    timeout_hits = 0
    last_rc = 126
    last_out = ""
    last_err = ""

    for attempt in range(1, total_attempts + 1):
        rc, out, err = _run_cmd(cmd, timeout_sec=probe_timeout_sec)
        last_rc, last_out, last_err = rc, out, err
        if rc == 124 and float(probe_timeout_sec) > 0:
            timeout_hits += 1
            if attempt < total_attempts:
                continue
            return ProbeInfo(
                path,
                size_bytes,
                None,
                None,
                None,
                None,
                None,
                (
                    f"ffprobe timeout after {probe_timeout_sec:.1f}s "
                    f"(attempt {attempt}/{total_attempts}); "
                    f"file flagged as corrupted: {path}"
                ),
            )
        if rc != 0:
            msg = (err or out or f"ffprobe rc={rc}").strip()
            if timeout_hits > 0:
                msg = (
                    f"{msg} (after {timeout_hits} watchdog timeout "
                    f"{'retry' if timeout_hits == 1 else 'retries'})"
                )
            return ProbeInfo(
                path,
                size_bytes,
                None,
                None,
                None,
                None,
                None,
                f"ffprobe failed rc={rc} on {path}: {msg}",
            )
        break
    else:
        return ProbeInfo(
            path,
            size_bytes,
            None,
            None,
            None,
            None,
            None,
            f"ffprobe failed rc={last_rc} on {path}: {last_err or last_out}",
        )

    try:
        doc = json.loads(last_out) if last_out else {}
    except Exception as e:
        return ProbeInfo(path, size_bytes, None, None, None, None, None, f"invalid ffprobe json: {e}")

    streams = doc.get("streams") or []
    if not streams:
        return ProbeInfo(path, size_bytes, None, None, None, None, None, "no video stream")

    st = streams[0] or {}
    nb_frames = _parse_int(st.get("nb_read_frames"))
    if nb_frames is None:
        nb_frames = _parse_int(st.get("nb_frames"))
    width = _parse_int(st.get("width"))
    height = _parse_int(st.get("height"))
    fps = _parse_rate(st.get("avg_frame_rate"))
    dur_stream = _parse_float(st.get("duration"))
    dur_fmt = _parse_float((doc.get("format") or {}).get("duration"))
    duration = dur_stream if dur_stream is not None else dur_fmt

    return ProbeInfo(path, size_bytes, nb_frames, duration, fps, width, height, None)


def decode_check_video(path: Path, decode_timeout_sec: float = 0.0) -> Tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    rc, _out, err = _run_cmd(cmd, timeout_sec=decode_timeout_sec)
    if rc == 0:
        return True, ""
    msg = (err or "").splitlines()[0].strip() if err else f"ffmpeg rc={rc}"
    if not msg:
        msg = f"ffmpeg rc={rc}"
    return False, msg


def normalize_stem(stem: str) -> str:
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


def collect_video_files(root: Path, recursive: bool, exts: set[str]) -> List[Path]:
    if recursive:
        it = root.rglob("*")
    else:
        it = root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    out.sort()
    return out


def _pick_single_candidate(cands: List[Path], target_suffix: str) -> Tuple[Optional[Path], Optional[str]]:
    if not cands:
        return None, "no candidates"
    if len(cands) == 1:
        return cands[0], None
    same_ext = [p for p in cands if p.suffix.lower() == target_suffix.lower()]
    if len(same_ext) == 1:
        return same_ext[0], None
    names = ", ".join(p.name for p in cands[:5])
    more = "" if len(cands) <= 5 else f" (+{len(cands)-5} more)"
    return None, f"ambiguous reference matches: {names}{more}"


def build_ref_indexes(ref_files: List[Path]) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    exact_idx: Dict[str, List[Path]] = {}
    norm_idx: Dict[str, List[Path]] = {}
    for p in ref_files:
        ek = p.stem.lower()
        nk = normalize_stem(p.stem)
        exact_idx.setdefault(ek, []).append(p)
        norm_idx.setdefault(nk, []).append(p)
    return exact_idx, norm_idx


def find_missing_in_target(
    ref_files: List[Path],
    target_exact_idx: Dict[str, List[Path]],
    target_norm_idx: Dict[str, List[Path]],
) -> List[Path]:
    missing: List[Path] = []
    for ref in ref_files:
        exact_key = ref.stem.lower()
        norm_key = normalize_stem(ref.stem)
        if target_exact_idx.get(exact_key):
            continue
        if target_norm_idx.get(norm_key):
            continue
        missing.append(ref)
    return missing


def match_reference(
    target: Path,
    exact_idx: Dict[str, List[Path]],
    norm_idx: Dict[str, List[Path]],
) -> Tuple[Optional[Path], str]:
    exact_key = target.stem.lower()
    cands = exact_idx.get(exact_key, [])
    ref, err = _pick_single_candidate(cands, target.suffix)
    if ref is not None:
        return ref, "exact"
    if err and "ambiguous" in err:
        return None, err

    norm_key = normalize_stem(target.stem)
    cands = norm_idx.get(norm_key, [])
    ref, err = _pick_single_candidate(cands, target.suffix)
    if ref is not None:
        return ref, f"normalized:{norm_key}"
    if err and "ambiguous" in err:
        return None, err

    return None, f"reference not found (exact='{exact_key}', normalized='{norm_key}')"


def compare_parity(
    target: ProbeInfo,
    ref: ProbeInfo,
    frame_tol: int,
) -> List[str]:
    reasons: List[str] = []

    if target.nb_frames is not None and ref.nb_frames is not None:
        dframes = abs(int(target.nb_frames) - int(ref.nb_frames))
        if dframes > int(frame_tol):
            reasons.append(
                f"frame mismatch: target={target.nb_frames}, ref={ref.nb_frames}, delta={dframes}, tol={frame_tol}"
            )
    else:
        if target.nb_frames is None:
            reasons.append("target frame count unavailable")
        if ref.nb_frames is None:
            reasons.append("reference frame count unavailable")

    return reasons


def verify_target_task(
    task: TargetTask,
    ref_probe_cache: Dict[Path, ProbeInfo],
    min_bytes: int,
    skip_decode: bool,
    skip_parity_compare: bool,
    frame_tol: int,
    probe_timeout_sec: float,
    probe_timeout_retries: int,
    decode_timeout_sec: float,
) -> TargetResult:
    reasons = list(task.pre_reasons)
    t_info = probe_video(
        task.target,
        probe_timeout_sec=probe_timeout_sec,
        probe_timeout_retries=probe_timeout_retries,
    )

    if t_info.error:
        reasons.append(f"probe: {t_info.error}")

    if t_info.size_bytes < int(min_bytes):
        reasons.append(f"file too small: {t_info.size_bytes} bytes < min {int(min_bytes)}")

    if not bool(skip_decode):
        ok_decode, decode_msg = decode_check_video(task.target, decode_timeout_sec=decode_timeout_sec)
        if not ok_decode:
            reasons.append(f"decode: {decode_msg}")

    # Keep parity behavior strict: only compare when target file passed checks.
    if not reasons and task.ref_path is not None and not bool(skip_parity_compare):
        r_info = ref_probe_cache.get(task.ref_path)
        if r_info is None:
            reasons.append(f"reference probe: missing cached probe for {task.ref_path}")
        elif r_info.error:
            reasons.append(f"reference probe: {r_info.error}")
        else:
            reasons.extend(
                compare_parity(
                    target=t_info,
                    ref=r_info,
                    frame_tol=int(frame_tol),
                )
            )

    return TargetResult(
        idx=task.idx,
        target=task.target,
        ref_path=task.ref_path,
        match_info=task.match_info,
        probe=t_info,
        reasons=reasons,
    )


def _delete_files(paths: List[Path]) -> Tuple[int, List[Tuple[Path, str]]]:
    deleted = 0
    failed: List[Tuple[Path, str]] = []
    for p in paths:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            failed.append((p, str(e)))
    return deleted, failed


def _to_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify scene files against reference clips (decode + frame parity), "
            "or against a single source file (final total frame parity). "
            "Duration is reported for diagnostics only."
        )
    )
    ap.add_argument("target_dir", help="Folder to verify.")
    ap.add_argument(
        "reference_path",
        help="Reference folder OR single source video file.",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders recursively in target/reference.",
    )
    ap.add_argument(
        "--extensions",
        default=".mp4,.mkv,.mov,.avi,.m4v,.webm",
        help="Comma-separated extensions to include.",
    )
    ap.add_argument("--frame-tol", type=int, default=0, help="Allowed frame count delta.")
    ap.add_argument(
        "--duration-tol-sec",
        type=float,
        default=0.15,
        help="Informational only (duration never affects pass/fail).",
    )
    ap.add_argument(
        "--min-bytes",
        type=int,
        default=1024,
        help="Minimum target file size in bytes.",
    )
    ap.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip full ffmpeg decode check (faster, less strict).",
    )
    ap.add_argument(
        "--delete",
        choices=("ask", "yes", "no"),
        default="ask",
        help="Delete failed target files: ask (default), yes, no.",
    )
    ap.add_argument(
        "--report-csv",
        default="",
        help="Optional CSV report path.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=19,
        help="Parallel workers for target verification (default: 8).",
    )
    ap.add_argument(
        "--probe-timeout-sec",
        type=float,
        default=0.0,
        help="Timeout for each ffprobe call in seconds (0 = disabled).",
    )
    ap.add_argument(
        "--probe-timeout-retries",
        type=int,
        default=1,
        help="Retries after ffprobe timeout on the same file (default: 1 => 2 attempts total).",
    )
    ap.add_argument(
        "--decode-timeout-sec",
        type=float,
        default=0.0,
        help="Timeout for each ffmpeg decode check (0 = disabled).",
    )
    ap.add_argument(
        "--progress-interval-sec",
        type=float,
        default=2.0,
        help="Heartbeat interval in seconds while waiting for workers.",
    )
    ap.add_argument(
        "--single-line-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use single-line carriage-return progress on TTY (default). "
            "Pass --no-single-line-progress to print progress on separate lines."
        ),
    )
    ap.add_argument("--verbose", action="store_true", help="Print OK entries too.")
    args = ap.parse_args()

    target_root = Path(args.target_dir).expanduser().resolve()
    ref_input = Path(args.reference_path).expanduser().resolve()
    single_reference_mode = False

    if not target_root.is_dir():
        print(f"[ERR] target_dir is not a folder: {target_root}")
        return 1
    if ref_input.is_dir():
        ref_root = ref_input
    elif ref_input.is_file():
        ref_root = ref_input.parent
        single_reference_mode = True
    else:
        print(f"[ERR] reference_path is not a folder or file: {ref_input}")
        return 1

    exts = {e.strip().lower() for e in str(args.extensions).split(",") if e.strip()}
    if not exts:
        print("[ERR] no valid extensions configured")
        return 1
    exts = {e if e.startswith(".") else f".{e}" for e in exts}

    targets = collect_video_files(target_root, recursive=bool(args.recursive), exts=exts)
    if single_reference_mode:
        refs = [ref_input]
    else:
        refs = collect_video_files(ref_root, recursive=bool(args.recursive), exts=exts)

    if not targets:
        print(f"[ERR] no target files found in {target_root} for extensions {sorted(exts)}")
        return 1
    if not refs:
        if single_reference_mode:
            print(f"[ERR] reference file not found: {ref_input}")
        else:
            print(f"[ERR] no reference files found in {ref_root} for extensions {sorted(exts)}")
        return 1

    print(f"[INFO] target files: {len(targets)}")
    if single_reference_mode:
        print(f"[INFO] mode: single reference file")
        print(f"[INFO] reference file: {refs[0]}")
    else:
        print(f"[INFO] reference files: {len(refs)}")

    exact_idx: Dict[str, List[Path]] = {}
    norm_idx: Dict[str, List[Path]] = {}
    if not single_reference_mode:
        exact_idx, norm_idx = build_ref_indexes(refs)
        target_exact_idx, target_norm_idx = build_ref_indexes(targets)
        missing_from_target = find_missing_in_target(
            ref_files=refs,
            target_exact_idx=target_exact_idx,
            target_norm_idx=target_norm_idx,
        )
        print(f"[INFO] missing in target vs reference: {len(missing_from_target)}")
        for miss in missing_from_target:
            print(f"[MISS] {_to_rel(miss, ref_root)}")
    else:
        print("[INFO] missing-in-target scan skipped in single reference mode")

    workers = max(1, int(args.workers))
    progress_interval = max(0.2, float(args.progress_interval_sec))
    probe_timeout_sec = max(0.0, float(args.probe_timeout_sec))
    probe_timeout_retries = max(0, int(args.probe_timeout_retries))
    decode_timeout_sec = max(0.0, float(args.decode_timeout_sec))
    stdout_is_tty = bool(sys.stdout.isatty())
    single_line_progress = bool(args.single_line_progress and stdout_is_tty)
    probe_timeout_label = "disabled" if probe_timeout_sec <= 0 else f"{probe_timeout_sec:.1f}s"
    decode_timeout_label = "disabled" if decode_timeout_sec <= 0 else f"{decode_timeout_sec:.1f}s"

    print(f"[INFO] workers: {workers}")
    print(
        f"[INFO] probe timeout: {probe_timeout_label} "
        f"(retries={probe_timeout_retries}) | decode timeout: {decode_timeout_label}"
    )

    tasks: List[TargetTask] = []
    if single_reference_mode:
        single_ref = refs[0]
        for idx, target in enumerate(targets):
            tasks.append(
                TargetTask(
                    idx=idx,
                    target=target,
                    ref_path=single_ref,
                    match_info="single-reference-total-only",
                    pre_reasons=[],
                )
            )
    else:
        for idx, target in enumerate(targets):
            ref_path, match_info = match_reference(target, exact_idx, norm_idx)
            pre_reasons: List[str] = []
            if ref_path is None:
                pre_reasons.append(f"match: {match_info}")
            tasks.append(
                TargetTask(
                    idx=idx,
                    target=target,
                    ref_path=ref_path,
                    match_info=match_info,
                    pre_reasons=pre_reasons,
                )
            )

    unique_refs = sorted({t.ref_path for t in tasks if t.ref_path is not None}, key=lambda p: str(p))
    ref_probe_cache: Dict[Path, ProbeInfo] = {}
    if unique_refs:
        print(f"[INFO] stage: probing {len(unique_refs)} reference file(s)")
    else:
        print("[INFO] stage: no reference files to probe")

    ref_count_frames = not bool(single_reference_mode)
    if single_reference_mode:
        print("[INFO] single-reference probe mode: metadata-only (no full frame scan on source)")

    if unique_refs:
        ref_total = len(unique_refs)
        ref_done = 0

        def _print_ref_progress(force: bool = False, note: str = ""):
            msg = f"[PROG-REF] {ref_done:04d}/{ref_total:04d}"
            if note:
                msg += f" {note}"
            if single_line_progress:
                print(f"\r{msg}", end="", flush=True)
            else:
                if force:
                    print(msg, flush=True)

        _print_ref_progress(force=True)
        ref_workers = min(workers, len(unique_refs))
        if ref_workers <= 1:
            for rp in unique_refs:
                ref_probe_cache[rp] = probe_video(
                    rp,
                    probe_timeout_sec=probe_timeout_sec,
                    probe_timeout_retries=probe_timeout_retries,
                    count_frames=ref_count_frames,
                )
                ref_done += 1
                _print_ref_progress(force=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=ref_workers) as ex:
                fut_to_ref = {
                    ex.submit(
                        probe_video,
                        rp,
                        probe_timeout_sec=probe_timeout_sec,
                        probe_timeout_retries=probe_timeout_retries,
                        count_frames=ref_count_frames,
                    ): rp
                    for rp in unique_refs
                }
                pending = set(fut_to_ref.keys())
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=progress_interval,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        _print_ref_progress(force=True, note=f"waiting={len(pending)}")
                        continue

                    for fut in done:
                        rp = fut_to_ref[fut]
                        try:
                            ref_probe_cache[rp] = fut.result()
                        except Exception as e:
                            ref_probe_cache[rp] = ProbeInfo(
                                path=rp,
                                size_bytes=0,
                                nb_frames=None,
                                duration_sec=None,
                                fps=None,
                                width=None,
                                height=None,
                                error=f"probe exception: {e}",
                            )
                        ref_done += 1
                        _print_ref_progress(force=True)
        if single_line_progress:
            print("")

    bad_files: List[Path] = []
    report_rows: List[Optional[Dict[str, str]]] = [None] * len(targets)
    results_cache: List[Optional[TargetResult]] = [None] * len(targets)
    ok_count = 0
    bad_count = 0
    done_count = 0

    # Initial progress line (helps when first file takes long).
    init_progress = (
        f"[PROG] {0:04d}/{len(targets):04d} "
        f"ok={0:04d} bad={0:04d}"
    )
    if single_line_progress:
        print(f"\r{init_progress}", end="", flush=True)
    else:
        print(init_progress, flush=True)

    def _handle_result(res: TargetResult):
        nonlocal done_count, ok_count, bad_count
        done_count += 1

        is_ok = not res.reasons
        if is_ok:
            ok_count += 1
        else:
            bad_count += 1
            bad_files.append(res.target)

        progress_line = (
            f"[PROG] {done_count:04d}/{len(targets):04d} "
            f"ok={ok_count:04d} bad={bad_count:04d} "
            f"file={_to_rel(res.target, target_root)}"
        )
        if single_line_progress:
            print(f"\r{progress_line}", end="", flush=True)
        else:
            print(progress_line, flush=True)

        pos = res.idx + 1
        if is_ok:
            if args.verbose:
                if single_line_progress:
                    print("")
                print(f"[OK ] {pos:04d}/{len(targets):04d} {_to_rel(res.target, target_root)}  ({res.match_info})")
        else:
            if single_line_progress:
                print("")
            print(
                f"[BAD] {pos:04d}/{len(targets):04d} {_to_rel(res.target, target_root)} :: "
                + " | ".join(res.reasons)
            )

        report_rows[res.idx] = {
            "target_file": str(res.target),
            "reference_file": str(res.ref_path) if res.ref_path else "",
            "status": "OK" if is_ok else "BAD",
            "reasons": " | ".join(res.reasons),
            "target_size_bytes": str(res.probe.size_bytes),
            "target_frames": "" if res.probe.nb_frames is None else str(res.probe.nb_frames),
            "target_duration_sec": "" if res.probe.duration_sec is None else f"{res.probe.duration_sec:.6f}",
            "match_info": res.match_info,
        }
        results_cache[res.idx] = res

    verify_workers = min(workers, len(tasks))
    if verify_workers <= 1:
        for task in tasks:
            try:
                res = verify_target_task(
                    task=task,
                    ref_probe_cache=ref_probe_cache,
                    min_bytes=int(args.min_bytes),
                    skip_decode=bool(args.skip_decode),
                    skip_parity_compare=bool(single_reference_mode),
                    frame_tol=int(args.frame_tol),
                    probe_timeout_sec=probe_timeout_sec,
                    probe_timeout_retries=probe_timeout_retries,
                    decode_timeout_sec=decode_timeout_sec,
                )
            except Exception as e:
                res = TargetResult(
                    idx=task.idx,
                    target=task.target,
                    ref_path=task.ref_path,
                    match_info=task.match_info,
                    probe=ProbeInfo(
                        path=task.target,
                        size_bytes=0,
                        nb_frames=None,
                        duration_sec=None,
                        fps=None,
                        width=None,
                        height=None,
                        error=f"worker exception: {e}",
                    ),
                    reasons=[f"internal error: {type(e).__name__}: {e}"],
                )
            _handle_result(res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=verify_workers) as ex:
            fut_to_task = {
                ex.submit(
                    verify_target_task,
                    task,
                    ref_probe_cache,
                    int(args.min_bytes),
                    bool(args.skip_decode),
                    bool(single_reference_mode),
                    int(args.frame_tol),
                    probe_timeout_sec,
                    probe_timeout_retries,
                    decode_timeout_sec,
                ): task
                for task in tasks
            }
            pending = set(fut_to_task.keys())
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=progress_interval,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    wait_msg = (
                        f"[WAIT] done={done_count:04d}/{len(tasks):04d} "
                        f"running={len(pending):04d}"
                    )
                    if single_line_progress:
                        print(f"\r{wait_msg}", end="", flush=True)
                    else:
                        print(wait_msg, flush=True)
                    continue

                for fut in done:
                    task = fut_to_task[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = TargetResult(
                            idx=task.idx,
                            target=task.target,
                            ref_path=task.ref_path,
                            match_info=task.match_info,
                            probe=ProbeInfo(
                                path=task.target,
                                size_bytes=0,
                                nb_frames=None,
                                duration_sec=None,
                                fps=None,
                                width=None,
                                height=None,
                                error=f"worker exception: {e}",
                            ),
                            reasons=[f"internal error: {type(e).__name__}: {e}"],
                        )
                    _handle_result(res)

    if single_line_progress:
        print("")

    total_parity_reasons: List[str] = []
    if single_reference_mode:
        ref_file = refs[0]
        ref_info = ref_probe_cache.get(ref_file)
        ok_results = [r for r in results_cache if r is not None and not r.reasons]
        if ref_info is None:
            total_parity_reasons.append("single-reference probe missing from cache")
        elif ref_info.error:
            total_parity_reasons.append(f"single-reference probe failed: {ref_info.error}")
        elif not ok_results:
            total_parity_reasons.append("no valid target files available for total parity check")
        else:
            total_target_frames: Optional[int] = 0
            total_target_duration: Optional[float] = 0.0
            for rr in ok_results:
                assert rr is not None
                if rr.probe.nb_frames is None:
                    total_target_frames = None
                elif total_target_frames is not None:
                    total_target_frames += int(rr.probe.nb_frames)

                if rr.probe.duration_sec is None:
                    total_target_duration = None
                elif total_target_duration is not None:
                    total_target_duration += float(rr.probe.duration_sec)

            if total_target_frames is not None and ref_info.nb_frames is not None:
                dframes = abs(int(total_target_frames) - int(ref_info.nb_frames))
                print(
                    f"[TOTAL] frames target_sum={total_target_frames} ref={ref_info.nb_frames} "
                    f"delta={dframes} tol={int(args.frame_tol)}"
                )
                if dframes > int(args.frame_tol):
                    total_parity_reasons.append(
                        f"total frame mismatch: target_sum={total_target_frames}, "
                        f"ref={ref_info.nb_frames}, delta={dframes}, tol={int(args.frame_tol)}"
                    )
            else:
                print("[TOTAL] frames target_sum/ref unavailable (metadata-only source probe)")

            if total_target_duration is not None and ref_info.duration_sec is not None:
                ddur = abs(float(total_target_duration) - float(ref_info.duration_sec))
                print(
                    f"[TOTAL-INFO] duration target_sum={total_target_duration:.3f}s "
                    f"ref={ref_info.duration_sec:.3f}s delta={ddur:.3f}s "
                    f"tol={float(args.duration_tol_sec):.3f}s (informational only)"
                )
            else:
                print("[TOTAL-INFO] duration target_sum/ref unavailable (metadata-only source probe)")

    if total_parity_reasons:
        for reason in total_parity_reasons:
            print(f"[BAD-TOTAL] {reason}")

    print("")
    print(f"[SUMMARY] checked={len(targets)}  ok={ok_count}  bad={bad_count}")
    if single_reference_mode:
        print(
            f"[SUMMARY-TOTAL] single-reference total parity: "
            f"{'OK' if not total_parity_reasons else 'BAD'}"
        )

    if args.report_csv:
        csv_path = Path(args.report_csv).expanduser().resolve()
        try:
            rows_to_write = [r for r in report_rows if r is not None]
            if len(rows_to_write) != len(report_rows):
                print(
                    f"[WARN] report has {len(report_rows) - len(rows_to_write)} missing row(s); writing partial CSV."
                )
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "target_file",
                        "reference_file",
                        "status",
                        "reasons",
                        "target_size_bytes",
                        "target_frames",
                        "target_duration_sec",
                        "match_info",
                    ],
                )
                w.writeheader()
                w.writerows(rows_to_write)
            print(f"[INFO] report written: {csv_path}")
        except Exception as e:
            print(f"[WARN] failed to write report csv '{csv_path}': {e}")

    if bad_files:
        do_delete = False
        mode = str(args.delete).strip().lower()
        if mode == "yes":
            do_delete = True
        elif mode == "ask":
            if sys.stdin.isatty():
                ans = input(f"Delete {len(bad_files)} bad file(s) from target_dir? [y/N]: ").strip().lower()
                do_delete = ans in ("y", "yes")
            else:
                print("[INFO] non-interactive stdin: skipping delete prompt.")

        if do_delete:
            deleted, failed = _delete_files(bad_files)
            print(f"[DELETE] deleted={deleted}/{len(bad_files)}")
            for p, err in failed[:10]:
                print(f"[WARN] delete failed: {p} :: {err}")
            if len(failed) > 10:
                print(f"[WARN] ... and {len(failed)-10} more delete failures")

    overall_failed = bool(bad_files) or bool(total_parity_reasons)
    return 0 if not overall_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
