from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_pipeline_step_state(gui: Any) -> dict[str, dict[str, object]]:
    return {key: {"completed": False, "verified": "none"} for key, _ in gui.PIPELINE_STEPS}


def pipeline_state_path(gui: Any) -> Path:
    work_dir = gui.work_folder_var.get().strip() or "./work"
    return Path(work_dir).resolve() / gui.PIPELINE_STATE_FILENAME


def load_pipeline_state(gui: Any) -> None:
    state_path = pipeline_state_path(gui)
    gui._pipeline_step_state = default_pipeline_step_state(gui)
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                steps = data.get("steps")
                if isinstance(steps, dict):
                    for key, _label in gui.PIPELINE_STEPS:
                        entry = steps.get(key)
                        if isinstance(entry, dict):
                            gui._pipeline_step_state[key]["completed"] = bool(
                                entry.get("completed", False)
                            )
                            ver = str(entry.get("verified", "none")).strip().lower()
                            if ver in {"quick", "deep"}:
                                gui._pipeline_step_state[key]["verified"] = "quick"
                            else:
                                gui._pipeline_step_state[key]["verified"] = "none"
                verify_after = str(data.get("verify_after", "")).strip()
                if verify_after in gui.PIPELINE_VERIFY_CHOICES:
                    gui.pipeline_verify_after_var.set(verify_after)
                elif verify_after.lower() == "deep":
                    gui.pipeline_verify_after_var.set("Quick")
        except Exception:
            pass
    gui._sync_pipeline_csv_done_flags()
    gui._refresh_pipeline_status_panel()


def save_pipeline_state(gui: Any) -> None:
    state_path = pipeline_state_path(gui)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "verify_after": gui.pipeline_verify_after_var.get().strip(),
            "steps": gui._pipeline_step_state,
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def is_pipeline_step_required(gui: Any, step: str) -> bool:
    if gui._pipeline_test_active and step in {"mono_to_sbs", "join", "remux"}:
        return False
    if step == "sharpness_csv":
        return (
            gui.inpaint_mode_var.get().strip() == "Auto (recommended)"
            or bool(gui.inpaint_use_sharpness_csv_var.get())
        )
    if step == "sharpen":
        return gui._sharpen_step_enabled_in_current_mode()
    if step == "autoct_csv":
        return gui.merge_ct_auto_mode_var.get().strip() == "CSV Blend"
    return True


def pipeline_set_completed_in_state(
    state: dict[str, dict[str, object]],
    step: str,
    value: bool,
) -> None:
    if step not in state:
        return
    state[step]["completed"] = bool(value)
    if not value:
        state[step]["verified"] = "none"


def pipeline_set_verified_in_state(
    state: dict[str, dict[str, object]],
    step: str,
    mode: str,
) -> None:
    if step not in state:
        return
    mode_low = str(mode).strip().lower()
    state[step]["verified"] = "quick" if mode_low in {"quick", "deep"} else "none"


def pipeline_verified_rank(mode: str) -> int:
    mode_low = str(mode).strip().lower()
    if mode_low == "quick":
        return 1
    return 0


def pipeline_set_verified_best_in_state(
    state: dict[str, dict[str, object]],
    step: str,
    mode: str,
) -> None:
    if step not in state:
        return
    mode_low = "quick" if str(mode).strip().lower() in {"quick", "deep"} else "none"
    current = str(state[step].get("verified", "none"))
    if pipeline_verified_rank(mode_low) >= pipeline_verified_rank(current):
        pipeline_set_verified_in_state(state, step, mode_low)


def sync_pipeline_csv_done_flags_in_state(
    gui: Any,
    state: dict[str, dict[str, object]],
) -> None:
    sharp_done = Path(gui.inpaint_sharpness_csv_var.get().strip()).is_file()
    autoct_done = Path(gui.merge_autoct_csv_var.get().strip()).is_file()
    pipeline_set_completed_in_state(state, "sharpness_csv", sharp_done)
    pipeline_set_completed_in_state(state, "autoct_csv", autoct_done)
    pipeline_set_verified_in_state(state, "sharpness_csv", "none")
    pipeline_set_verified_in_state(state, "autoct_csv", "none")


def sync_pipeline_csv_done_flags(gui: Any) -> None:
    sync_pipeline_csv_done_flags_in_state(gui, gui._pipeline_step_state)


def pipeline_mark_previous_steps_done_verified_in_state(
    gui: Any,
    state: dict[str, dict[str, object]],
    step: str,
    mode: str,
) -> None:
    step_keys = [k for k, _ in gui.PIPELINE_STEPS]
    if step not in step_keys:
        return
    target_mode = "quick" if str(mode).strip().lower() in {"quick", "deep"} else "none"
    upto_idx = step_keys.index(step)
    for key in step_keys[: upto_idx + 1]:
        if key in gui.PIPELINE_CSV_STEPS:
            continue
        pipeline_set_completed_in_state(state, key, True)
        if key in gui.PIPELINE_STEPS_WITH_VERIFY:
            pipeline_set_verified_best_in_state(state, key, target_mode)


def pipeline_invalidate_from_in_state(
    gui: Any,
    state: dict[str, dict[str, object]],
    step: str,
    include_current: bool = True,
) -> None:
    step_keys = [k for k, _ in gui.PIPELINE_STEPS]
    if step not in step_keys:
        return
    start = step_keys.index(step)
    if not include_current:
        start += 1
    for key in step_keys[start:]:
        pipeline_set_completed_in_state(state, key, False)

