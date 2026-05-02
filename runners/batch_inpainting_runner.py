#!/usr/bin/env python3
"""Headless batch runner for StereoCrafter inpainting (no GUI / no display).

It reuses the GUI implementation's processing code (chunking, streaming encode, mask ops, tiling)
by instantiating a minimal subclass of `InpaintingGUI` **without** initializing Tk.

Important: run it from the StereoCrafter repo root (so ./weights/... resolves).
"""

import os
import sys
import glob
import shutil
import argparse
import csv
import threading
import gc
import subprocess
import json
import math
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

# The GUI module contains the full inpainting implementation we want to reuse.
# Importing it is fine headless; we just must not create a real Tk window.

import inpainting_gui as igs

RESTART_EVERY = 0           # or from env/arg
PLANNED_RESTART_CODE = 99
class _Var:
    """Tiny stand-in for tkinter's StringVar/BooleanVar/IntVar."""

    def __init__(self, value):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class HeadlessInpainting(igs.InpaintingGUI):
    """Subclass InpaintingGUI but avoid initializing Tk / ThemedTk."""

    def __init__(
        self,
        output_folder: str,
        input_folder: str = "",
        hires_blend_folder: str = "",
        replace_mask_folder: str = "",
        use_replace_mask: bool = False,
        debug_mode: bool = False,
        enable_color_transfer: bool = True,
        enable_post_inpainting_blend: bool = False,
        mask_initial_threshold: float = 0.3,
        mask_morph_kernel_size: float = 0.0,
        mask_dilate_kernel_size: int = 5,
        mask_blur_kernel_size: int = 10,
    ):
        # DO NOT call super().__init__() (it would create a Tk window)
        self.output_folder_var = _Var(output_folder)
        self.input_folder_var = _Var(input_folder)
        self.hires_blend_folder_var = _Var(hires_blend_folder)
        self.replace_mask_folder_var = _Var(replace_mask_folder)
        self.use_replace_mask_var = _Var(bool(use_replace_mask))

        self.debug_mode_var = _Var(bool(debug_mode))
        self.enable_color_transfer = _Var(bool(enable_color_transfer))
        self.enable_post_inpainting_blend = _Var(bool(enable_post_inpainting_blend))

        # GUI stores these as StringVar; processing code casts to float/int.
        self.mask_initial_threshold_var = _Var(str(mask_initial_threshold))
        self.mask_morph_kernel_size_var = _Var(str(mask_morph_kernel_size))
        self.mask_dilate_kernel_size_var = _Var(str(mask_dilate_kernel_size))
        self.mask_blur_kernel_size_var = _Var(str(mask_blur_kernel_size))

        # Some methods expect these exist
        self.stop_event = threading.Event()
        self.pipeline = None

    # ---- Tk compatibility shims ----
    def after(self, _ms, func=None, *args, **kwargs):
        """Tk schedules callbacks asynchronously; do the same to avoid recursion."""
        if func is None:
            return None
        import threading
        t = threading.Timer(max(0, _ms) / 1000.0, func, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        return t

    def update_status_label(self, _message: str):
        # GUI only; ignore
        return None

    def __getattr__(self, name):
        """Headless mode: never delegate to Tk's interpreter.

        - Auto-stub Tk-style variables used by the GUI (StringVar/BooleanVar/IntVar)
        - Provide no-op stubs for common Tk methods some codepaths may call
        """
        if name.endswith("_var"):
            v = _Var(None)
            setattr(self, name, v)
            return v

        if name in ("update", "update_idletasks", "winfo_exists", "winfo_width", "winfo_height", "quit", "destroy"):
            return lambda *a, **k: None

        raise AttributeError(name)

def _safe_release_cuda():
    """Try hard to free VRAM between files."""
    try:
        igs.release_cuda_memory()
    except Exception:
        pass
    gc.collect()


def _load_retry_resume_state(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return None


def _save_retry_resume_state(path: str, input_name: str, next_attempt: int, total_attempts: int) -> None:
    payload = {
        "input_name": str(input_name),
        "next_attempt": int(next_attempt),
        "total_attempts": int(total_attempts),
        "updated_at": int(time.time()),
    }
    try:
        _save_resume_state(path, payload)
    except Exception as e:
        print(f"[WARN] failed writing retry resume state: {e}")


def _clear_retry_resume_state(path: str) -> None:
    _clear_resume_state(path)


def _load_retry_skip_manifest(path: str) -> set[str]:
    try:
        if not os.path.exists(path):
            return set()
        out: set[str] = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = str(line).strip()
                if name:
                    out.add(name)
        return out
    except Exception:
        return set()


def _save_retry_skip_manifest(path: str, names: set[str]) -> None:
    try:
        ordered = sorted({str(x).strip() for x in names if str(x).strip()})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for name in ordered:
                f.write(name + "\n")
        os.replace(tmp, path)
    except Exception as e:
        print(f"[WARN] failed writing retry skip manifest: {e}")


def _run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _ffprobe_nb_packets(path: str) -> int:
    """
    Return packet count for the first video stream.
    We use count_packets (faster than full decode count_frames) to validate skip/restart outputs.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=nw=1:nk=1",
        path,
    ]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed rc={rc}: {err}")

    for line in (out or "").splitlines():
        s = line.strip()
        if not s or s.upper() == "N/A":
            continue
        try:
            return int(float(s))
        except Exception:
            continue
    raise RuntimeError(f"ffprobe returned invalid nb_read_packets: {out!r}")


def _find_replace_mask_for_input(input_path: str, replace_mask_folder: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    mask_dir = os.path.abspath(replace_mask_folder) if replace_mask_folder else os.path.dirname(os.path.abspath(input_path))
    hits = sorted(glob.glob(os.path.join(mask_dir, f"{stem}_replace_mask.*")))
    return hits[0] if hits else ""


def _resolve_validation_reference(input_path: str, replace_mask_folder: str) -> tuple[str, str]:
    """
    Prefer replace-mask as reference (fast count_packets); fallback to input video.
    Returns (reference_path, reference_kind).
    """
    mask_path = _find_replace_mask_for_input(input_path, replace_mask_folder)
    if mask_path and os.path.exists(mask_path):
        return mask_path, "replace_mask"
    return input_path, "input"


def _cleanup_outputs(out_path: str) -> None:
    if not out_path:
        return
    for p in (out_path, out_path + ".tmp", out_path + ".part", out_path + ".temp"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass



def _resume_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".resume_state.json")


def _current_job_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".current_job.json")


def _retry_resume_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".inpaint_retry_resume_state.json")


def _retry_skip_manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".inpaint_retry_skipped.txt")


def _load_resume_state(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_resume_state(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _clear_resume_state(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _load_current_job_state(path: str):
    return _load_resume_state(path)


def _save_current_job_state(path: str, data: dict):
    _save_resume_state(path, data)


def _clear_current_job_state(path: str):
    _clear_resume_state(path)


def _recover_interrupted_current_job(current_job_path: str):
    state = _load_current_job_state(current_job_path)
    if not isinstance(state, dict):
        return

    input_path = str(state.get("input_path") or "")
    output_path = str(state.get("output_path") or "")
    process_length = state.get("process_length", -1)

    try:
        if not output_path:
            print("[RECOVER] current-job marker found without output_path. Clearing marker.")
            return

        if not os.path.exists(output_path):
            print(f"[RECOVER] current-job output not found (already absent): {output_path}")
            return

        if input_path and os.path.exists(input_path):
            if _is_output_complete(input_path, output_path, process_length):
                print(f"[RECOVER] interrupted job output already complete, keeping: {output_path}")
            else:
                print(f"[RECOVER] interrupted job output incomplete, deleting: {output_path}")
                _cleanup_outputs(output_path)
            return

        # Input missing: fallback to basic readability probe.
        try:
            _ffprobe_nb_packets(output_path)
            print(f"[RECOVER] input missing for interrupted job; output decodes, keeping: {output_path}")
        except Exception:
            print(f"[RECOVER] interrupted job output unreadable, deleting: {output_path}")
            _cleanup_outputs(output_path)
    finally:
        _clear_current_job_state(current_job_path)


def _default_stop_marker_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".stop_after_current")


def _stop_marker_exists(path: str) -> bool:
    if not path:
        return False
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _clear_stop_marker(path: str) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _is_output_complete(
    input_path: str,
    output_path: str,
    process_length: int,
    replace_mask_path: str = "",
    tol_packets: int = 1,
) -> bool:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False
    try:
        ref_path = replace_mask_path if (replace_mask_path and os.path.exists(replace_mask_path)) else input_path
        ref_packets = _ffprobe_nb_packets(ref_path)
        out_packets = _ffprobe_nb_packets(output_path)
        expected = ref_packets
        if process_length is not None:
            try:
                pl = int(process_length)
            except Exception:
                pl = -1
            if pl and pl > 0:
                expected = min(ref_packets, pl)
        return out_packets >= max(0, expected - int(tol_packets))
    except Exception:
        return False


def _move_to_subfolder(path: str, subfolder_name: str) -> str:
    folder = os.path.join(os.path.dirname(path), subfolder_name)
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, os.path.basename(path))
    try:
        shutil.move(path, dst)
    except Exception:
        # If move fails, keep original
        return path
    return dst

def _load_sharpness_csv(csv_path: str):
    """Return mapping {basename -> sharpness_raw}. If csv missing, returns empty dict."""
    if not csv_path:
        return {}
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return {}
        out = {}
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                name = (row.get("file") or "").strip()
                if not name:
                    continue
                raw_s = (row.get("sharpness_raw") or "").strip()
                pct_s = (row.get("sharpness_pct") or "").strip()
                try:
                    raw = float(raw_s) if raw_s != "" else float(pct_s)
                except Exception:
                    continue
                out[name] = raw
        return out
    except Exception:
        return {}

RETRY_PROFILE_ORDER = ("run", "retry1", "retry2", "retry3")
RETRY_OFFLOAD_CHOICES = {"none", "model", "sequential"}
RETRY_MAX_SPLIT_CHOICES = {64, 128, 256, 512}


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


def _norm_offload_mode(value: object, fallback: str = "model") -> str:
    mode = str(value or "").strip().lower()
    if mode in RETRY_OFFLOAD_CHOICES:
        return mode
    fb = str(fallback or "model").strip().lower()
    return fb if fb in RETRY_OFFLOAD_CHOICES else "model"


def _norm_max_split(value: object) -> int | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"", "none", "off", "0", "false"}:
        return None
    try:
        parsed = int(float(s))
    except Exception:
        return None
    if parsed in RETRY_MAX_SPLIT_CHOICES:
        return parsed
    return None


def _default_retry_profiles(base_offload: str) -> list[dict[str, object]]:
    inherited = _norm_offload_mode(base_offload, "model")
    return [
        {
            "name": "run",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": None,
            "cpu_offload_mode": inherited,
        },
        {
            "name": "retry1",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 512,
            "cpu_offload_mode": inherited,
        },
        {
            "name": "retry2",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 64,
            "cpu_offload_mode": inherited,
        },
        {
            "name": "retry3",
            "garbage_collection_threshold": True,
            "expandable_segments": True,
            "max_split_size_mb": 64,
            "cpu_offload_mode": "sequential",
        },
    ]


def _parse_retry_policy_profiles(
    policy_json: str,
    base_offload: str,
) -> tuple[list[dict[str, object]], bool]:
    defaults = _default_retry_profiles(base_offload)
    txt = str(policy_json or "").strip()
    if not txt:
        return defaults, False
    try:
        raw = json.loads(txt)
    except Exception as e:
        print(f"[WARN] retry_policy_json parse failed: {e}. Using defaults.")
        return defaults, True
    if not isinstance(raw, dict):
        print("[WARN] retry_policy_json is not an object. Using defaults.")
        return defaults, True

    out: list[dict[str, object]] = []
    for idx, name in enumerate(RETRY_PROFILE_ORDER):
        base = defaults[idx]
        node = raw.get(name, {})
        if not isinstance(node, dict):
            node = {}
        out.append(
            {
                "name": name,
                "garbage_collection_threshold": _norm_bool(
                    node.get(
                        "garbage_collection_threshold",
                        base["garbage_collection_threshold"],
                    ),
                    bool(base["garbage_collection_threshold"]),
                ),
                "expandable_segments": _norm_bool(
                    node.get("expandable_segments", base["expandable_segments"]),
                    bool(base["expandable_segments"]),
                ),
                "max_split_size_mb": _norm_max_split(
                    node.get("max_split_size_mb", base["max_split_size_mb"])
                ),
                "cpu_offload_mode": _norm_offload_mode(
                    node.get("cpu_offload_mode", base["cpu_offload_mode"]),
                    str(base["cpu_offload_mode"]),
                ),
            }
        )
    return out, True


def _allocator_conf_from_profile(profile: dict[str, object]) -> str:
    parts: list[str] = []
    if _norm_bool(profile.get("garbage_collection_threshold"), True):
        parts.append("garbage_collection_threshold:0.8")
    if _norm_bool(profile.get("expandable_segments"), True):
        parts.append("expandable_segments:True")
    max_split = _norm_max_split(profile.get("max_split_size_mb"))
    if max_split is not None:
        parts.append(f"max_split_size_mb:{max_split}")
    return ",".join(parts)


def _apply_allocator_conf(conf: str) -> None:
    conf_s = str(conf or "").strip()
    if conf_s:
        os.environ["PYTORCH_ALLOC_CONF"] = conf_s
    else:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    try:
        alt = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if callable(alt):
            alt(conf_s)
            return
        setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
        if callable(setter):
            setter(conf_s)
    except Exception as e:
        print(f"[WARN] failed to apply allocator settings '{conf_s or 'default'}': {e}")


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _steps_from_sharpness(val: float) -> float:
    """
    Continuous rule:
    raw <= 1100 -> 3.0
    every additional 1100 sharpness points adds +1.0 step
    clamp to 8.0

    The dynamic chunk planner can still see this through
    _chunk_planner_steps_from_model_steps(), preserving the old 5..8 chunk
    behavior for the new 3..6 model-step range.
    """
    try:
        v = float(val)
    except Exception:
        return 3.0

    if v <= 1100.0:
        return 3.0

    steps = 3.0 + ((v - 1100.0) / 1100.0)
    return max(3.0, min(8.0, steps))


def _chunk_planner_steps_from_model_steps(model_effective_steps: float) -> float:
    return max(5.0, min(8.0, float(model_effective_steps) + 2.0))


def _dynamic_resolution_scale_from_steps(model_effective_steps: float) -> float:
    rounded = max(3, min(8, _round_half_up(model_effective_steps)))
    return 0.50 + (float(rounded) - 3.0) * 0.10


def _clamp_resolution_scale(value: float) -> float:
    return max(0.50, min(1.0, float(value)))


def _parse_chunk_limit_list(value) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return [22]
    parsed: list[int] = []
    for part in parts:
        parsed.append(max(1, int(float(part))))
    return parsed


def _chunk_limit_for_scale(value, dynamic_scale: float | None) -> int:
    limits = _parse_chunk_limit_list(value)
    if len(limits) == 1 or dynamic_scale is None:
        return limits[0]
    if dynamic_scale >= 0.995:
        idx = 0
    elif dynamic_scale >= 0.895:
        idx = 1
    elif dynamic_scale >= 0.795:
        idx = 2
    elif dynamic_scale >= 0.695:
        idx = 3
    elif dynamic_scale >= 0.595:
        idx = 4
    else:
        idx = 5
    return limits[min(idx, len(limits) - 1)]


def _load_repo_benchmark_tile1_limits() -> str | None:
    for path in (
        REPO_ROOT / "inpaint_tile1_chunk_benchmark.json",
        REPO_ROOT / "inpaint_tile1_chunk_benchmark.csv",
    ):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".json":
                raw = json.loads(path.read_text(encoding="utf-8"))
                values = raw.get("tile1_max_chunks") if isinstance(raw, dict) else None
                if isinstance(values, list) and values:
                    return ",".join(str(max(1, int(v))) for v in values[:6])
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                raw_values = row.get("tile1_max_chunks") or row.get("max_chunks") or ""
                if raw_values.strip():
                    return raw_values.strip()
        except Exception as exc:
            print(f"[WARN] failed reading benchmark limits from {path}: {exc}")
    return None



def run_batch(args):
    os.makedirs(args.output_dir, exist_ok=True)

    processed_this_run = 0

    runner = HeadlessInpainting(
        output_folder=args.output_dir,
        input_folder=(args.input_dir or (os.path.dirname(args.input_video) if args.input_video else "")),
        hires_blend_folder=args.hires_blend_folder or "",
        replace_mask_folder=args.replace_mask_folder or "",
        use_replace_mask=args.use_replace_mask,
        debug_mode=args.debug,
        enable_color_transfer=not args.disable_color_transfer,
        enable_post_inpainting_blend=args.enable_post_inpainting_blend,
        mask_initial_threshold=args.mask_initial_threshold,
        mask_morph_kernel_size=args.mask_morph_kernel_size,
        mask_dilate_kernel_size=args.mask_dilate_kernel_size,
        mask_blur_kernel_size=args.mask_blur_kernel_size,
    )

    retry_profiles, policy_was_explicit = _parse_retry_policy_profiles(
        getattr(args, "retry_policy_json", ""),
        args.offload_type,
    )
    if policy_was_explicit:
        print("[INFO] retry policy source=gui/env")
    else:
        print("[INFO] retry policy source=default")
    for prof in retry_profiles:
        alloc = _allocator_conf_from_profile(prof) or "default"
        print(
            "[INFO] retry profile "
            f"{prof['name']}: offload={prof['cpu_offload_mode']} alloc={alloc}"
        )
    retry_skipped: list[str] = []

    pipeline = None
    pipeline_mode = ""

    def _drop_pipeline():
        nonlocal pipeline, pipeline_mode
        if pipeline is not None:
            try:
                del pipeline
            except Exception:
                pass
        pipeline = None
        pipeline_mode = ""
        runner.pipeline = None
        _safe_release_cuda()

    def _ensure_pipeline(offload_type: str):
        nonlocal pipeline, pipeline_mode
        mode = _norm_offload_mode(offload_type, "none")
        if pipeline is not None and pipeline_mode == mode:
            return
        _drop_pipeline()
        pipeline = igs.load_inpainting_pipeline(
            pre_trained_path=r"./weights/stable-video-diffusion-img2vid-xt-1-1",
            unet_path=r"./weights/StereoCrafter",
            device="cuda",
            dtype=torch.float16,
            offload_type=mode,
        )
        runner.pipeline = pipeline
        pipeline_mode = mode

    # Build file list
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))

    if not videos:
        print("[ERR] no input videos found")
        return 2

    resume_path = _resume_state_path(args.output_dir)
    current_job_path = _current_job_state_path(args.output_dir)
    retry_resume_state_file = _retry_resume_state_path(args.output_dir)
    retry_skip_manifest_file = _retry_skip_manifest_path(args.output_dir)
    stop_marker_path = (
        os.path.abspath(args.stop_marker)
        if args.stop_marker
        else _default_stop_marker_path(args.output_dir)
    )
    resume = _load_resume_state(resume_path)
    fast_resume_start = None
    if resume and resume.get("mode") == "planned_restart":
        last_ok_idx = resume.get("last_ok_idx")
        if isinstance(last_ok_idx, int) and 0 <= last_ok_idx < len(videos):
            fast_resume_start = max(0, last_ok_idx - 1)
            print(f"[RESUME] Fast resume enabled. Rechecking index {fast_resume_start + 1} then continuing.")
        else:
            _clear_resume_state(resume_path)
    elif resume:
        _clear_resume_state(resume_path)

    retry_resume_state = _load_retry_resume_state(retry_resume_state_file)
    retry_skip_persisted = _load_retry_skip_manifest(retry_skip_manifest_file)
    if retry_resume_state and str(retry_resume_state.get("input_name") or "").strip():
        print(
            "[INFO] retry resume state found: "
            f"input={retry_resume_state.get('input_name')} "
            f"next_attempt={retry_resume_state.get('next_attempt')}"
        )
    if retry_skip_persisted:
        print(f"[INFO] persisted retry-skip files loaded: {len(retry_skip_persisted)}")

    # Recover interrupted per-file work from a previous crash/restart.
    _recover_interrupted_current_job(current_job_path)


    stop_event = threading.Event()

    # Optional: load sharpness CSV once (mapping basename -> sharpness_raw).
    sharpness_map = {}
    sharp_csv = ""
    if not args.no_sharpness_csv:
        sharp_csv_override = str(args.sharpness_csv_path or "").strip()
        if sharp_csv_override:
            sharp_csv = os.path.abspath(sharp_csv_override)
        else:
            sharp_base = args.sharpness_base
            if not sharp_base:
                if args.input_dir:
                    sharp_base = args.input_dir
                elif args.input_video:
                    sharp_base = os.path.dirname(args.input_video)
                else:
                    sharp_base = os.getcwd()
            sharp_csv = os.path.join(os.path.abspath(sharp_base), "sharpness.csv")
        sharpness_map = _load_sharpness_csv(sharp_csv)
        print(f"[INFO] sharpness_csv: {sharp_csv} (rows={len(sharpness_map)})")
    else:
        print(f"[INFO] sharpness CSV disabled; using fixed steps={args.fixed_steps}")

    for idx, video_path in enumerate(videos, 1):
        if _stop_marker_exists(stop_marker_path):
            print(f"[STOP] marker detected, stopping before next file: {stop_marker_path}")
            _clear_stop_marker(stop_marker_path)
            return 0

        i = idx - 1
        if fast_resume_start is not None and i < fast_resume_start:
            continue
        base = os.path.basename(video_path)
        print(f"\n[{idx}/{len(videos)}] {base}")

        if base in retry_skip_persisted:
            print(f"[SKIP] {idx}/{len(videos)} {base} (retry-skip persisted)")
            continue

        out_path = ""
        hi_res_input_path = None
        current_job_marked = False
        validation_ref_path = video_path
        validation_ref_kind = "input"

        try:
            # Ensure GUI vars are consistent for hi-res matching safety checks
            runner.input_folder_var.set(args.input_dir or os.path.dirname(video_path))
            
            # If hires blending folder is empty, force-disable hi-res matching by setting it equal to input folder.
            # This avoids accidental globbing in CWD and keeps output naming stable (also makes --skip_existing reliable).
            if not args.hires_blend_folder:
                runner.hires_blend_folder_var.set(runner.input_folder_var.get())
            else:
                runner.hires_blend_folder_var.set(args.hires_blend_folder)
            

            # Determine expected output name like GUI would, to support skip.
            name_wo_ext = os.path.splitext(base)[0]
            input_layout = runner._infer_input_layout_from_stem(name_wo_ext)
            out_path, _hires = runner._setup_video_info_and_hires(video_path, args.output_dir, input_layout)
            validation_ref_path, validation_ref_kind = _resolve_validation_reference(
                video_path,
                args.replace_mask_folder,
            )

            if args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                # Strict skip: validate readability + packet length parity before skipping.
                if _is_output_complete(
                    video_path,
                    out_path,
                    args.process_length,
                    replace_mask_path=validation_ref_path if validation_ref_kind == "replace_mask" else "",
                ):
                    print(f"[SKIP] exists+valid ({validation_ref_kind}): {out_path}")
                    if retry_resume_state and str(retry_resume_state.get("input_name") or "") == base:
                        _clear_retry_resume_state(retry_resume_state_file)
                        retry_resume_state = None
                    if fast_resume_start is not None and i >= fast_resume_start:
                        fast_resume_start = None
                        _clear_resume_state(resume_path)
                    continue
                print(f"[WARN] existing output invalid vs {validation_ref_kind}; deleting: {out_path}")
                _cleanup_outputs(out_path)

                        # Determine inference steps:
            # - If sharpness.csv is enabled and has a row for this basename, derive steps from it.
            # - Otherwise, use --fixed_steps.
            sharp_val = sharpness_map.get(base) if sharpness_map else None
            if sharp_val is None:
                effective_steps = max(1.0, float(args.fixed_steps))
                model_steps = _round_half_up(effective_steps)
                planner_steps = effective_steps
                print(
                    f"[INFO] effective_steps={effective_steps:.2f} "
                    f"planner_steps={planner_steps:.2f} "
                    f"model_steps={model_steps} (fixed)"
                )
            else:
                effective_steps = _steps_from_sharpness(sharp_val)
                model_steps = _round_half_up(effective_steps)
                planner_steps = _chunk_planner_steps_from_model_steps(effective_steps)
                print(
                    f"[INFO] effective_steps={effective_steps:.2f} "
                    f"planner_steps={planner_steps:.2f} "
                    f"model_steps={model_steps} (sharp_raw={sharp_val:.2f})"
                )
            resolution_scale = _clamp_resolution_scale(float(args.resolution_scale))
            if bool(args.enable_dynamic_resolution):
                dynamic_resolution_scale = min(
                    _dynamic_resolution_scale_from_steps(effective_steps),
                    resolution_scale,
                )
            else:
                dynamic_resolution_scale = resolution_scale if resolution_scale < 0.999 else None
            if dynamic_resolution_scale is not None:
                print(f"[INFO] dynamic_resolution scale={dynamic_resolution_scale:.2f}")
            tile1_max_size = _chunk_limit_for_scale(args.tile1_max_size, dynamic_resolution_scale)
            tile2_max_size = _chunk_limit_for_scale(args.tile2_max_size, dynamic_resolution_scale)

            frames_chunk = max(1, int(args.chunk_size))
            print(
                f"[INFO] chunk_setup dynamic={bool(args.enable_dynamic_chunk)} "
                f"chunk_size={frames_chunk} tile_mode={args.tile_mode} "
                f"tile1_max={int(tile1_max_size)} tile2_max={int(tile2_max_size)} "
                f"overlap={int(args.overlap)} tail_pad={int(args.tail_pad)} "
                f"steps5={int(args.dynamic_visible_chunk_steps5)} "
                f"steps6={int(args.dynamic_visible_chunk_steps6)} "
                f"steps7={int(args.dynamic_visible_chunk_steps7)} "
                f"steps8plus={int(args.dynamic_visible_chunk_steps8_plus)} "
                f"hold_divisor={float(args.dynamic_hold_divisor):.2f}"
            )

            # Keep overlap valid: must be < frames_chunk (otherwise chunking can't progress).
            overlap = int(args.overlap)
            if frames_chunk <= 0:
                frames_chunk = max(1, int(args.chunk_size))
            if overlap < 0:
                overlap = 0
            if overlap >= frames_chunk:
                new_overlap = max(0, frames_chunk - 1)
                print(f"[WARN] overlap={overlap} >= frames_chunk={frames_chunk}; clamping overlap -> {new_overlap}")
                overlap = new_overlap
            tail_pad = int(args.tail_pad)
            if tail_pad < 0:
                print(f"[WARN] tail_pad={tail_pad} < 0; clamping tail_pad -> 0")
                tail_pad = 0
            if tail_pad >= frames_chunk:
                new_tail_pad = max(0, frames_chunk - 1)
                print(f"[WARN] tail_pad={tail_pad} >= frames_chunk={frames_chunk}; clamping tail_pad -> {new_tail_pad}")
                tail_pad = new_tail_pad

            _save_current_job_state(current_job_path, {
                "mode": "active_job",
                "idx": idx,
                "total": len(videos),
                "input_path": video_path,
                "output_path": out_path,
                "chunk_size": frames_chunk,
                "dynamic_chunk": bool(args.enable_dynamic_chunk),
                "tile_mode": args.tile_mode,
                "overlap": overlap,
                "tail_pad": tail_pad,
                "process_length": int(args.process_length),
            })
            current_job_marked = True

            run_ok = False
            start_attempt_idx = 1
            if retry_resume_state and str(retry_resume_state.get("input_name") or "") == base:
                try:
                    start_attempt_idx = int(retry_resume_state.get("next_attempt") or 1)
                except Exception:
                    start_attempt_idx = 1
                start_attempt_idx = max(1, min(len(retry_profiles), start_attempt_idx))
                if start_attempt_idx > 1:
                    print(
                        f"[RETRY] resuming {idx}/{len(videos)} from attempt "
                        f"{start_attempt_idx}/{len(retry_profiles)} after process restart"
                    )

            for attempt_idx in range(start_attempt_idx, len(retry_profiles) + 1):
                prof = retry_profiles[attempt_idx - 1]
                alloc_conf = _allocator_conf_from_profile(prof)
                offload_mode = str(prof["cpu_offload_mode"])
                _save_retry_resume_state(
                    retry_resume_state_file,
                    base,
                    attempt_idx,
                    len(retry_profiles),
                )
                print(
                    f"[RETRY] {idx}/{len(videos)} attempt {attempt_idx}/{len(retry_profiles)} "
                    f"profile={prof['name']} offload={offload_mode} "
                    f"alloc={alloc_conf or 'default'}"
                )
                _apply_allocator_conf(alloc_conf)
                _ensure_pipeline(offload_mode)

                try:
                    completed, hi_res_input_path = runner.process_single_video(
                        pipeline=pipeline,
                        input_video_path=video_path,
                        save_dir=args.output_dir,
                        frames_chunk=frames_chunk,
                        overlap=overlap,
                        tail_pad=tail_pad,
                        enable_dynamic_chunk=bool(args.enable_dynamic_chunk),
                        tile_mode=args.tile_mode,
                        tile1_max_size=tile1_max_size,
                        tile2_max_size=tile2_max_size,
                        vf=None,
                        effective_inference_steps=effective_steps,
                        chunk_planner_effective_steps=planner_steps,
                        dynamic_resolution_scale=dynamic_resolution_scale,
                        stop_event=stop_event,
                        update_info_callback=None,
                        original_input_blend_strength=args.original_input_blend_strength,
                        output_codec=args.output_codec,
                        output_encoding_mode=args.output_encoding_mode,
                        output_extra_args=args.output_extra_args,
                        process_length=args.process_length,
                        dynamic_visible_chunk_steps5=args.dynamic_visible_chunk_steps5,
                        dynamic_visible_chunk_steps6=args.dynamic_visible_chunk_steps6,
                        dynamic_visible_chunk_steps7=args.dynamic_visible_chunk_steps7,
                        dynamic_visible_chunk_steps8_plus=args.dynamic_visible_chunk_steps8_plus,
                        static_mask_divisor=args.dynamic_hold_divisor,
                    )
                except Exception as e:
                    completed = False
                    print(
                        f"[ERR] attempt {attempt_idx}/{len(retry_profiles)} "
                        f"failed: {type(e).__name__}: {e}"
                    )
                    _cleanup_outputs(out_path)
                    _drop_pipeline()
                    continue

                if completed and _is_output_complete(
                    video_path,
                    out_path,
                    args.process_length,
                    replace_mask_path=validation_ref_path if validation_ref_kind == "replace_mask" else "",
                ):
                    run_ok = True
                    if base in retry_skip_persisted:
                        retry_skip_persisted.discard(base)
                        _save_retry_skip_manifest(retry_skip_manifest_file, retry_skip_persisted)
                    _clear_retry_resume_state(retry_resume_state_file)
                    retry_resume_state = None
                    break

                if completed:
                    print(
                        f"[FAIL] attempt {attempt_idx}/{len(retry_profiles)} "
                        f"output incomplete, deleting: {out_path}"
                    )
                else:
                    print(
                        f"[FAIL] attempt {attempt_idx}/{len(retry_profiles)} "
                        "processing returned incomplete"
                    )
                _cleanup_outputs(out_path)
                _drop_pipeline()

            if run_ok:
                if current_job_marked:
                    _clear_current_job_state(current_job_path)
                    current_job_marked = False
                print(f"[OK] wrote: {out_path}")
                processed_this_run += 1
                if fast_resume_start is not None and i >= fast_resume_start:
                    fast_resume_start = None
                    _clear_resume_state(resume_path)
                if RESTART_EVERY > 0 and processed_this_run >= RESTART_EVERY:
                    print(f"[PLANNED RESTART] processed_this_run={processed_this_run}, exiting {PLANNED_RESTART_CODE}")
                    _save_resume_state(resume_path, {
                        "mode": "planned_restart",
                        "last_ok_idx": i,
                        "last_ok_input": video_path,
                        "last_ok_output": out_path,
                    })
                    sys.exit(PLANNED_RESTART_CODE)
                if args.move_finished:
                    _move_to_subfolder(video_path, args.finished_subdir)
                    if hi_res_input_path and os.path.exists(hi_res_input_path):
                        _move_to_subfolder(hi_res_input_path, args.finished_subdir)
            else:
                print(
                    f"[SKIP] {idx}/{len(videos)} {base} skipped after "
                    f"{len(retry_profiles)} retry profiles"
                )
                retry_skipped.append(base)
                retry_skip_persisted.add(base)
                _save_retry_skip_manifest(retry_skip_manifest_file, retry_skip_persisted)
                _cleanup_outputs(out_path)
                _clear_retry_resume_state(retry_resume_state_file)
                retry_resume_state = None
                if current_job_marked:
                    _clear_current_job_state(current_job_path)
                    current_job_marked = False

        except Exception as e:
            print(f"[ERR] {type(e).__name__}: {e}")
            _cleanup_outputs(out_path)
            if current_job_marked:
                _clear_current_job_state(current_job_path)
                current_job_marked = False
            _drop_pipeline()
            continue
        finally:
            try:
                igs._cleanup_ffmpeg_writers()
            except Exception:
                pass
            # keep VRAM stable between files
            _safe_release_cuda()

    if _stop_marker_exists(stop_marker_path):
        print(f"[STOP] marker detected at end of batch, clearing: {stop_marker_path}")
        _clear_stop_marker(stop_marker_path)

    if retry_skipped:
        preview = ", ".join(retry_skipped[:10])
        more = "" if len(retry_skipped) <= 10 else f", ... (+{len(retry_skipped) - 10} more)"
        print(f"[DONE] retry-skip files ({len(retry_skipped)}): {preview}{more}")

    _drop_pipeline()

    return 0


def main():
    p = argparse.ArgumentParser(description="StereoCrafter inpainting headless batch runner")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", type=str, help="Folder containing input videos")
    src.add_argument("--input_video", type=str, help="Single input video path")

    p.add_argument("--output_dir", type=str, required=True, help="Output folder")
    p.add_argument("--glob", type=str, default="*.mp4", help="Glob pattern when using --input_dir")
    p.add_argument("--sharpness_base", type=str, default="", help="Base folder containing sharpness.csv (defaults to input folder)")
    p.add_argument("--sharpness_csv_path", type=str, default="",
                   help="Explicit sharpness CSV path (overrides --sharpness_base).")
    p.add_argument("--no_sharpness_csv", action="store_true",
                   help="Ignore sharpness.csv and use --fixed_steps for all files")
    p.add_argument("--fixed_steps", type=float, default=8.0,
                   help="Fallback effective steps when sharpness.csv is missing or ignored")
    p.add_argument("--resolution_scale", type=float, default=0.90,
                   help="Dynamic resolution max scale when enabled, fixed model scale when disabled.")
    p.add_argument("--enable_dynamic_resolution", dest="enable_dynamic_resolution", action="store_true",
                   help="Resize inpaint inputs dynamically from model steps, then upscale back to source size.")
    p.add_argument("--disable_dynamic_resolution", dest="enable_dynamic_resolution", action="store_false",
                   help="Keep inpaint inputs at source resolution.")
    p.set_defaults(enable_dynamic_resolution=True)
    p.add_argument("--chunk_size", type=int, default=22,
                   help="Processed chunk size (visible + overlap, no tail) used when dynamic chunk is disabled.")
    p.add_argument("--enable_dynamic_chunk", dest="enable_dynamic_chunk", action="store_true",
                   help="Resolve chunk size per scene from steps + mask hold.")
    p.add_argument("--disable_dynamic_chunk", dest="enable_dynamic_chunk", action="store_false",
                   help="Use fixed --chunk_size instead of dynamic chunk selection.")
    p.set_defaults(enable_dynamic_chunk=True)
    p.add_argument("--tile_mode", type=str, default="1 and 2", choices=["1", "2", "1 and 2"])
    p.add_argument("--tile1_max_size", type=str, default=igs.DEFAULT_TILE1_CHUNK_BUCKETS_4090)
    p.add_argument("--tile2_max_size", type=str, default=igs.DEFAULT_TILE2_CHUNK_BUCKETS_4090)
    p.add_argument("--dynamic_visible_chunk_steps5", type=int, default=igs.DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS5)
    p.add_argument("--dynamic_visible_chunk_steps6", type=int, default=igs.DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS6)
    p.add_argument("--dynamic_visible_chunk_steps7", type=int, default=igs.DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS7)
    p.add_argument("--dynamic_visible_chunk_steps8_plus", type=int, default=igs.DEFAULT_DYNAMIC_VISIBLE_CHUNK_STEPS8_PLUS)
    p.add_argument("--dynamic_hold_divisor", type=float, default=igs.DEFAULT_DYNAMIC_STATIC_MASK_DIVISOR)
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument("--tail_pad", type=int, default=1,
                   help="Guard frames used for both non-last chunk handoff and last-chunk duplication.")
    p.add_argument("--original_input_blend_strength", type=float, default=0.0)
    p.add_argument("--process_length", type=int, default=-1)

    p.add_argument("--offload_type", type=str, default="none", choices=["none", "model", "sequential"],
                   help="Matches GUI offload_type")
    p.add_argument("--retry_policy_json", type=str, default="",
                   help="Optional JSON policy for per-file retries (run/retry1/retry2/retry3).")
    p.add_argument("--output_codec", type=str, default="", help="Optional ffmpeg output codec override")
    p.add_argument("--output_encoding_mode", type=str, default="", help="Shared encoder mode (lossless, crf/qp 0, crf/qp 1).")
    p.add_argument("--output_extra_args", type=str, default="", help="Optional extra ffmpeg args")

    p.add_argument("--hires_blend_folder", type=str, default="", help="Optional hires folder (same as GUI)")
    p.add_argument("--use_replace_mask", action="store_true",
                   help="Use external replace-mask files (<splatted_stem>_replace_mask.*). Fast-fail if missing/invalid.")
    p.add_argument("--replace_mask_folder", type=str, default="",
                   help="Folder containing external replace-mask files. Empty => input video folder.")

    p.add_argument("--mask_initial_threshold", type=float, default=0.3)
    p.add_argument("--mask_morph_kernel_size", type=float, default=0.0)
    p.add_argument("--mask_dilate_kernel_size", type=int, default=5)
    p.add_argument("--mask_blur_kernel_size", type=int, default=10)

    p.add_argument("--enable_post_inpainting_blend", action="store_true")
    p.add_argument("--disable_color_transfer", action="store_true")

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--move_finished", action="store_true")
    p.add_argument("--finished_subdir", type=str, default="finished")

    p.add_argument("--debug", action="store_true", help="Enable debug image saving (uses GUI code)")
    p.add_argument(
        "--stop_marker",
        type=str,
        default="",
        help="Path to a marker file used for graceful stop-after-current-file behavior",
    )

    args = p.parse_args()

    # Normalize paths
    if args.input_video:
        args.input_video = os.path.abspath(args.input_video)
    if args.input_dir:
        args.input_dir = os.path.abspath(args.input_dir)
    if args.replace_mask_folder:
        args.replace_mask_folder = os.path.abspath(args.replace_mask_folder)
    args.output_dir = os.path.abspath(args.output_dir)

    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
