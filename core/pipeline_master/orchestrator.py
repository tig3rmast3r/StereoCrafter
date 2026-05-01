from __future__ import annotations

from tkinter import messagebox
from typing import Any


def show_pipeline_force_info(gui: Any, title: str, message: str) -> None:
    fn = gui._messagebox_originals.get("showinfo")
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


def pipeline_start_resume(gui: Any) -> None:
    if gui._any_pipeline_activity():
        messagebox.showinfo("Start/Resume", "Another task is currently running.")
        gui._pipeline_sync_noninteractive_mode()
        return
    gui._pipeline_pause_after_split_scenes = False
    if not gui._pipeline_test_active and gui._pipeline_split_scenes_gate_pending():
        gate_msg = (
            "Pipeline will pause after Split Scenes verification.\n\n"
            "When Split Scenes verify is done, move clips you do NOT want to convert "
            "into the seg-mono folder.\n\n"
            "Then press Start/Resume again to continue."
        )
        gui._append_pipeline_popup_log("INFO", "Start/Resume", gate_msg)
        gui._show_pipeline_force_info("Start/Resume", gate_msg)
        gui._pipeline_pause_after_split_scenes = True
    gui._pipeline_ui_noninteractive = True
    gui._append_pipeline_popup_log(
        "INFO",
        "Start/Resume",
        "Non-interactive mode enabled: popups are suppressed and routed to this log.",
    )
    gui._pipeline_reset_skip_notices()
    gui._pipeline_autorun = True
    gui._pipeline_pending_action = None
    gui.pipeline_run_status_var.set("Start/Resume running...")
    gui._pipeline_trigger_next_action()


def pipeline_split_scenes_gate_pending(gui: Any) -> bool:
    st = gui._pipeline_step_state.get("split_scenes", {"completed": False, "verified": "none"})
    if not bool(st.get("completed", False)):
        return True
    verify_mode = gui.pipeline_verify_after_var.get().strip().lower()
    verified = str(st.get("verified", "none")).strip().lower()
    if verify_mode == "quick":
        return verified != "quick"
    return False


def pipeline_reset_skip_notices(gui: Any) -> None:
    gui._pipeline_skip_notice_steps.clear()


def pipeline_maybe_log_completed_autoct_skip(gui: Any, next_step: str) -> None:
    if "autoct_csv" in gui._pipeline_skip_notice_steps:
        return
    if not gui._is_pipeline_step_required("autoct_csv"):
        return
    step_state = (
        gui._pipeline_test_step_state
        if gui._pipeline_test_active
        else gui._pipeline_step_state
    )
    if not bool((step_state.get("autoct_csv") or {}).get("completed", False)):
        return
    step_order = [name for name, _label in gui.PIPELINE_STEPS]
    try:
        next_idx = step_order.index(str(next_step).strip())
        autoct_idx = step_order.index("autoct_csv")
    except ValueError:
        return
    if next_idx <= autoct_idx:
        return
    msg = (
        "[AUTOCT] Existing CSV already valid for current test subset. "
        "Skipping AutoCT CSV step."
        if gui._pipeline_test_active
        else "[AUTOCT] Existing CSV already marked complete. Skipping AutoCT CSV step."
    )
    gui._append_merge_log(msg)
    gui._pipeline_skip_notice_steps.add("autoct_csv")


def pipeline_trigger_next_action(gui: Any) -> None:
    if not gui._pipeline_autorun:
        gui._pipeline_sync_noninteractive_mode()
        return
    if gui._any_pipeline_activity():
        return
    action = gui._pipeline_next_action()
    if action is None:
        gui._pipeline_autorun = False
        gui._pipeline_pending_action = None
        if gui._pipeline_test_active:
            gui._restore_test_scene_subset()
        gui.pipeline_run_status_var.set("Pipeline: all required steps completed")
        gui._pipeline_sync_noninteractive_mode()
        return

    step, act, mode = action
    gui._pipeline_maybe_log_completed_autoct_skip(step)
    gui._pipeline_pending_action = (step, act, mode)
    started = False
    if act == "run":
        started = gui._pipeline_dispatch_run(step)
    else:
        started = gui._pipeline_dispatch_verify(step, mode)
    if not started:
        gui._pipeline_autorun = False
        gui._pipeline_pending_action = None
        if gui._pipeline_test_active:
            gui._restore_test_scene_subset()
        gui.pipeline_run_status_var.set(f"Pipeline stopped: could not start {step} {act}")
        gui._pipeline_sync_noninteractive_mode()


def pipeline_next_action(gui: Any) -> tuple[str, str, str] | None:
    verify_mode = gui.pipeline_verify_after_var.get().strip().lower()
    if gui._pipeline_test_active:
        gui._pipeline_recompute_test_step_state()
    step_state = (
        gui._pipeline_test_step_state
        if gui._pipeline_test_active
        else gui._pipeline_step_state
    )
    for step, _label in gui.PIPELINE_STEPS:
        if not gui._is_pipeline_step_required(step):
            continue
        st = step_state.get(step, {"completed": False, "verified": "none"})
        if not bool(st.get("completed", False)):
            return step, "run", "none"
        if gui._pipeline_test_active and step in {"scenedetect", "split_scenes"}:
            continue
        if step in gui.PIPELINE_STEPS_WITH_VERIFY and verify_mode == "quick":
            current = str(st.get("verified", "none"))
            if current != "quick":
                return step, "verify", "quick"
    return None


def pipeline_dispatch_run(gui: Any, step: str) -> bool:
    gui.pipeline_run_status_var.set(f"Running step: {step}")
    if step == "scenedetect":
        before = bool(gui._scene_thread and gui._scene_thread.is_alive())
        gui._start_scene_detect()
        return bool(gui._scene_thread and gui._scene_thread.is_alive()) and not before
    if step == "split_scenes":
        before = bool(gui._scene_thread and gui._scene_thread.is_alive())
        gui._start_split_scenes()
        return bool(gui._scene_thread and gui._scene_thread.is_alive()) and not before
    if step == "depthcrafter":
        before = bool(gui._depth_thread and gui._depth_thread.is_alive())
        gui._run_depth_placeholder()
        return bool(gui._depth_thread and gui._depth_thread.is_alive()) and not before
    if step == "splatting":
        before = bool(gui._splat_thread and gui._splat_thread.is_alive())
        gui._run_splat_placeholder()
        return bool(gui._splat_thread and gui._splat_thread.is_alive()) and not before
    if step == "sharpness_csv":
        before = bool(gui._inpaint_thread and gui._inpaint_thread.is_alive())
        gui._start_inpaint_sharpness_csv()
        return bool(gui._inpaint_thread and gui._inpaint_thread.is_alive()) and not before
    if step == "inpaint":
        before = bool(gui._inpaint_thread and gui._inpaint_thread.is_alive())
        gui._run_inpaint_placeholder()
        return bool(gui._inpaint_thread and gui._inpaint_thread.is_alive()) and not before
    if step == "sharpen":
        before = bool(gui._inpaint_thread and gui._inpaint_thread.is_alive())
        gui._start_inpaint_sharpen()
        return bool(gui._inpaint_thread and gui._inpaint_thread.is_alive()) and not before
    if step == "autoct_csv":
        before = bool(gui._merge_thread and gui._merge_thread.is_alive())
        gui._start_merge_autoct_csv()
        return bool(gui._merge_thread and gui._merge_thread.is_alive()) and not before
    if step == "mask_for_merge":
        before = bool(gui._merge_thread and gui._merge_thread.is_alive())
        gui._run_merge_mask_placeholder()
        return bool(gui._merge_thread and gui._merge_thread.is_alive()) and not before
    if step == "merging":
        before = bool(gui._merge_thread and gui._merge_thread.is_alive())
        gui._run_merge_placeholder()
        return bool(gui._merge_thread and gui._merge_thread.is_alive()) and not before
    if step == "mono_to_sbs":
        before = bool(gui._join_thread and gui._join_thread.is_alive())
        gui._run_join_prepare_mono()
        return bool(gui._join_thread and gui._join_thread.is_alive()) and not before
    if step == "join":
        before = bool(gui._join_thread and gui._join_thread.is_alive())
        gui._run_join_scenes()
        return bool(gui._join_thread and gui._join_thread.is_alive()) and not before
    if step == "remux":
        before = bool(gui._join_thread and gui._join_thread.is_alive())
        gui._start_join_remux()
        return bool(gui._join_thread and gui._join_thread.is_alive()) and not before
    return False


def pipeline_dispatch_verify(gui: Any, step: str, mode: str) -> bool:
    gui.pipeline_run_status_var.set(f"Verifying {step} ({mode})")
    before = gui._verify_running
    if step == "split_scenes":
        gui._start_verify_quick()
    elif step == "depthcrafter":
        gui._start_depth_verify_quick()
    elif step == "splatting":
        gui._start_splat_verify_quick()
    elif step == "inpaint":
        gui._start_inpaint_verify_quick()
    elif step == "sharpen":
        gui._start_inpaint_sharpen_verify_quick()
    elif step == "mask_for_merge":
        gui._start_merge_mask_verify_quick()
    elif step == "merging":
        gui._start_merge_verify_quick()
    elif step == "mono_to_sbs":
        gui._start_join_mono_verify()
    elif step == "join":
        gui._start_join_verify()
    else:
        return False
    return gui._verify_running and not before


def any_pipeline_activity(gui: Any) -> bool:
    return any(
        [
            bool(gui._scene_thread and gui._scene_thread.is_alive()),
            bool(gui._analysis_thread and gui._analysis_thread.is_alive()),
            bool(gui._depth_thread and gui._depth_thread.is_alive()),
            bool(gui._splat_thread and gui._splat_thread.is_alive()),
            bool(gui._inpaint_thread and gui._inpaint_thread.is_alive()),
            bool(gui._merge_thread and gui._merge_thread.is_alive())
            or gui._merge_group_alive(),
            bool(gui._join_thread and gui._join_thread.is_alive()),
            bool(gui._verify_running),
        ]
    )


def pipeline_on_run_finished(
    gui: Any,
    step: str,
    success: bool,
    *,
    mark_completed: bool = True,
) -> None:
    pending = gui._pipeline_pending_action
    state = gui._pipeline_test_step_state if gui._pipeline_test_active else gui._pipeline_step_state
    if success:
        gui._pipeline_set_completed_in_state(state, step, bool(mark_completed))
        gui._pipeline_set_verified_in_state(state, step, "none")
        if step in gui.PIPELINE_CSV_STEPS:
            gui._sync_pipeline_csv_done_flags_in_state(state)
        gui._refresh_pipeline_status_panel()
        if not gui._pipeline_test_active:
            gui._save_pipeline_state()
    if pending and pending[0] == step and pending[1] == "run":
        gui._pipeline_pending_action = None
        if not success:
            gui._pipeline_autorun = False
            if gui._pipeline_test_active:
                gui._restore_test_scene_subset()
            gui.pipeline_run_status_var.set(f"Pipeline stopped: step failed ({step})")
            gui._pipeline_sync_noninteractive_mode()
            return
        if not mark_completed:
            gui._pipeline_autorun = False
            gui.pipeline_run_status_var.set(
                f"Pipeline paused: {step} ran in incomplete mode and was not marked complete."
            )
            gui._pipeline_sync_noninteractive_mode()
            return
        if (
            step == "split_scenes"
            and gui._pipeline_pause_after_split_scenes
            and not gui._pipeline_test_active
        ):
            verify_mode = gui.pipeline_verify_after_var.get().strip().lower()
            if verify_mode != "quick":
                gui._pipeline_pause_after_split_scenes = False
                gui._pipeline_autorun = False
                pause_msg = (
                    "Split Scenes completed.\n\n"
                    "Please move clips you do NOT want to convert into seg-mono,\n"
                    "then press Start/Resume again to continue."
                )
                gui.pipeline_run_status_var.set(
                    "Paused after Split Scenes. Move files to seg-mono, then Start/Resume."
                )
                gui._append_pipeline_popup_log("INFO", "Split Scenes Pause", pause_msg)
                gui._show_pipeline_force_info("Split Scenes Pause", pause_msg)
                gui._pipeline_sync_noninteractive_mode()
                return
        gui._pipeline_trigger_next_action()


def pipeline_on_verify_finished(
    gui: Any,
    step: str,
    success: bool,
    mode: str,
    retry_on_failure: bool = True,
) -> None:
    pending = gui._pipeline_pending_action
    state = gui._pipeline_test_step_state if gui._pipeline_test_active else gui._pipeline_step_state
    if success:
        gui._pipeline_mark_previous_steps_done_verified_in_state(state, step, mode)
        gui._sync_pipeline_csv_done_flags_in_state(state)
        gui._refresh_pipeline_status_panel()
        if not gui._pipeline_test_active:
            gui._save_pipeline_state()
    else:
        gui._pipeline_invalidate_active_from(step)
    if pending and pending[0] == step and pending[1] == "verify":
        gui._pipeline_pending_action = None
        if not success:
            if gui._pipeline_autorun:
                if not retry_on_failure:
                    gui._pipeline_autorun = False
                    if gui._pipeline_test_active:
                        gui._restore_test_scene_subset()
                    gui.pipeline_run_status_var.set(
                        f"Pipeline stopped: verify failed ({step}), manual fix required."
                    )
                    gui._pipeline_sync_noninteractive_mode()
                    return
                gui.pipeline_run_status_var.set(
                    f"Verify failed on {step}: re-running previous step output."
                )
                gui._pipeline_trigger_next_action()
                return
            gui._pipeline_autorun = False
            if gui._pipeline_test_active:
                gui._restore_test_scene_subset()
            gui.pipeline_run_status_var.set(f"Pipeline stopped: verify failed ({step})")
            gui._pipeline_sync_noninteractive_mode()
            return
        if (
            step == "split_scenes"
            and gui._pipeline_pause_after_split_scenes
            and not gui._pipeline_test_active
        ):
            gui._pipeline_pause_after_split_scenes = False
            gui._pipeline_autorun = False
            pause_msg = (
                "Split Scenes verify completed.\n\n"
                "Please move clips you do NOT want to convert into seg-mono,\n"
                "then press Start/Resume again to continue."
            )
            gui.pipeline_run_status_var.set(
                "Paused after Split Scenes verify. Move files to seg-mono, then Start/Resume."
            )
            gui._append_pipeline_popup_log("INFO", "Split Scenes Verify Pause", pause_msg)
            gui._show_pipeline_force_info("Split Scenes Verify Pause", pause_msg)
            gui._pipeline_sync_noninteractive_mode()
            return
        gui._pipeline_trigger_next_action()
    elif not success:
        gui.pipeline_run_status_var.set(
            f"Verify failed on {step}: cleared this step and downstream flags."
        )
