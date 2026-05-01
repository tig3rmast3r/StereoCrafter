from __future__ import annotations

import os
import queue
import re
import select
import signal
import shlex
import shutil
import sys
import termios
import threading
import time
import tkinter as tk
import tty
from pathlib import Path
from tkinter import messagebox
from typing import Any

from dependency.repo_paths import utilities_path
from pipeline_master_gui import DEFAULT_PIPELINE_MASTER_CONFIG_PATH, PipelineMasterGUI


class HeadlessTkRoot:
    def __init__(self, interp: tk.Misc) -> None:
        self._interp = interp
        self._geometry = PipelineMasterGUI.DEFAULT_WINDOW_GEOMETRY

    def title(self, _value: str) -> None:
        return None

    def geometry(self, value: str | None = None) -> str:
        if value is None:
            return self._geometry
        self._geometry = str(value)
        return self._geometry

    def minsize(self, _width: int, _height: int) -> None:
        return None

    def after(self, _delay_ms: int, _callback=None):
        if _callback is None:
            return None
        delay = max(0.0, float(_delay_ms) / 1000.0)
        timer = threading.Timer(delay, _callback)
        timer.daemon = True
        timer.start()
        return timer

    def protocol(self, _name: str, _callback) -> None:
        return None

    def update_idletasks(self) -> None:
        return None

    def destroy(self) -> None:
        return None


class HeadlessWidgetStub:
    def configure(self, **_kwargs) -> None:
        return None


class HeadlessPipelineMaster(PipelineMasterGUI):
    _HEADLESS_STEP_DOMAIN_MAP = {
        "scenedetect": "scene",
        "split_scenes": "scene",
        "depthcrafter": "depth",
        "splatting": "splat",
        "sharpness_csv": "inpaint",
        "inpaint": "inpaint",
        "sharpen": "inpaint",
        "mask_for_merge": "merge",
        "autoct_csv": "merge",
        "merging": "merge",
        "mono_to_sbs": "join",
        "join": "join",
        "remux": "join",
    }
    _HEADLESS_LINE_KIND_DOMAIN_MAP = {
        "line": "scene",
        "depth_line": "depth",
        "splat_line": "splat",
        "inpaint_line": "inpaint",
        "merge_line": "merge",
        "join_line": "join",
    }
    _HEADLESS_STATUS_VAR_MAP = {
        "scene": "scene_status_var",
        "depth": "depth_status_var",
        "splat": "splat_status_var",
        "inpaint": "inpaint_status_var",
        "merge": "merge_status_var",
        "join": "join_status_var",
    }
    _HEADLESS_PROGRESS_VAR_MAP = {
        "scene": "scene_progress_var",
        "depth": "depth_progress_var",
        "splat": "splat_progress_var",
        "inpaint": "inpaint_progress_var",
        "merge": "merge_progress_var",
        "join": "join_progress_var",
    }
    _HEADLESS_RATIO_RE = re.compile(r"(?<!\d)(\d{1,9})\s*/\s*(\d{1,9})(?!\d)")
    _HEADLESS_HEADER_HEIGHT = 6

    def __init__(
        self,
        *,
        work_dir: str = "",
        config_file: str = "",
        verify_after: str = "config",
    ) -> None:
        self._headless_interp = tk.Tcl()
        tk._default_root = self._headless_interp
        root = HeadlessTkRoot(self._headless_interp)

        resolved_work_dir = (
            str(Path(work_dir).expanduser().resolve()) if str(work_dir or "").strip() else ""
        )
        config_target = self._resolve_config_target(
            work_dir=resolved_work_dir,
            config_file=config_file,
        )
        super().__init__(
            root,
            config_file=config_target,
            work_dir_override=resolved_work_dir or None,
        )
        self._pipeline_ui_noninteractive = True
        self.scene_stop_btn = HeadlessWidgetStub()
        self.depth_stop_btn = HeadlessWidgetStub()
        self.splat_stop_btn = HeadlessWidgetStub()
        self.inpaint_stop_btn = HeadlessWidgetStub()
        self.merge_stop_btn = HeadlessWidgetStub()
        if resolved_work_dir:
            self.work_folder_var.set(resolved_work_dir)
            self._config["work_folder"] = resolved_work_dir
            self._refresh_standard_paths()
            self._load_pipeline_state()
        verify_after_value = str(verify_after or "config").strip().lower()
        if verify_after_value == "quick":
            self.pipeline_verify_after_var.set("Quick")
        elif verify_after_value == "none":
            self.pipeline_verify_after_var.set("")
        self._save_pipeline_state()
        self._headless_sticky_header_supported = self._supports_sticky_header()
        self._headless_sticky_header_active = False
        self._headless_header_top = 1
        self._headless_log_start = self._HEADLESS_HEADER_HEIGHT + 1
        self._headless_active_steps: list[str] = []
        self._headless_current_step = ""
        self._headless_current_action = ""
        self._headless_current_mode = ""
        self._headless_verify_retry_budget = 0
        self._headless_verify_attempts: dict[str, int] = {}
        self._headless_ratio_by_domain: dict[str, tuple[int, int]] = {}
        self._headless_signal_handlers_installed = False
        self._headless_previous_signal_handlers: dict[int, object] = {}
        self._headless_user_stop_exit_code: int | None = None

    @staticmethod
    def _resolve_config_target(*, work_dir: str, config_file: str) -> str:
        explicit = str(config_file or "").strip()
        if explicit:
            return str(Path(explicit).expanduser().resolve())
        if work_dir:
            return str(Path(work_dir).resolve() / "config_pipeline_master_gui.json")
        return str(Path(DEFAULT_PIPELINE_MASTER_CONFIG_PATH).resolve())

    @staticmethod
    def _supports_sticky_header() -> bool:
        term = str(os.environ.get("TERM", "")).strip().lower()
        return bool(sys.stdout.isatty() and sys.stdin.isatty() and term and term != "dumb")

    @staticmethod
    def _signal_exit_code(signum: int) -> int:
        try:
            return 128 + int(signum)
        except Exception:
            return 130

    @staticmethod
    def _terminal_write(text: str) -> None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except BrokenPipeError:
            return None

    @staticmethod
    def _sanitize_header_text(value: object) -> str:
        return str(value or "").replace("\r", " ").replace("\n", " ").strip()

    @classmethod
    def _truncate_header_line(cls, value: object, width: int) -> str:
        text = cls._sanitize_header_text(value)
        if width <= 0:
            return text
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    @staticmethod
    def _query_cursor_position() -> tuple[int, int] | None:
        if not sys.stdin.isatty():
            return None
        fd = sys.stdin.fileno()
        try:
            old_attrs = termios.tcgetattr(fd)
        except Exception:
            return None

        response = bytearray()
        try:
            tty.setcbreak(fd)
            HeadlessPipelineMaster._terminal_write("\x1b[6n")
            deadline = time.time() + 0.25
            while time.time() < deadline:
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                chunk = os.read(fd, 32)
                if not chunk:
                    break
                response.extend(chunk)
                if b"R" in chunk:
                    break
        except Exception:
            return None
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

        match = re.search(r"\x1b\[(\d+);(\d+)R", response.decode(errors="ignore"))
        if not match:
            return None
        try:
            return int(match.group(1)), int(match.group(2))
        except Exception:
            return None

    def _step_domain(self, step: str) -> str:
        return str(self._HEADLESS_STEP_DOMAIN_MAP.get(str(step or "").strip(), "")).strip()

    def _scope_summary(self) -> str:
        if not self._headless_active_steps:
            return "-"
        if len(self._headless_active_steps) == 1:
            return self._headless_active_steps[0]
        return (
            f"{self._headless_active_steps[0]} -> "
            f"{self._headless_active_steps[-1]} ({len(self._headless_active_steps)} steps)"
        )

    def _required_active_steps(self) -> list[str]:
        return [step for step in self._headless_active_steps if self._is_pipeline_step_required(step)]

    def _current_step_position(self) -> tuple[int, int]:
        ordered = self._required_active_steps()
        if not ordered:
            return 0, 0
        if self._headless_current_step in ordered:
            return ordered.index(self._headless_current_step) + 1, len(ordered)
        return 0, len(ordered)

    def _current_domain_status_text(self) -> str:
        domain = self._step_domain(self._headless_current_step)
        if domain:
            attr_name = self._HEADLESS_STATUS_VAR_MAP.get(domain, "")
            if attr_name:
                value = getattr(self, attr_name).get().strip()
                if value:
                    return value
        return self.pipeline_run_status_var.get().strip() or "Idle"

    def _current_progress_text(self) -> str:
        if self._headless_current_action != "run":
            return "-"

        domain = self._step_domain(self._headless_current_step)
        if not domain:
            return "-"

        ratio = self._headless_ratio_by_domain.get(domain)
        if ratio is not None:
            current, total = ratio
            percent = int(round((current / total) * 100.0)) if total else 0
            return f"{current}/{total} ({percent}%)"

        attr_name = self._HEADLESS_PROGRESS_VAR_MAP.get(domain, "")
        if attr_name:
            try:
                value = float(getattr(self, attr_name).get())
            except Exception:
                value = 0.0
            bucket = max(0, min(100, int(round(value))))
            return f"{bucket}%"
        return "0%"

    def _pipeline_progress_text(self) -> str:
        active_steps = self._required_active_steps()
        if not active_steps:
            return "0/0 steps complete"

        completed_steps = 0
        verify_total = 0
        verify_done = 0
        quick_verify_enabled = self.pipeline_verify_after_var.get().strip().lower() == "quick"
        for step in active_steps:
            st = self._pipeline_step_state.get(step, {"completed": False, "verified": "none"})
            if bool(st.get("completed", False)):
                completed_steps += 1
            if quick_verify_enabled and step in self.PIPELINE_STEPS_WITH_VERIFY:
                verify_total += 1
                if str(st.get("verified", "none")).strip().lower() == "quick":
                    verify_done += 1

        line = f"{completed_steps}/{len(active_steps)} steps complete"
        if verify_total:
            line += f" | quick verify {verify_done}/{verify_total}"
        return line

    def _build_footer_line(self) -> str:
        step_index, step_total = self._current_step_position()
        current_step = self._headless_current_step or "-"
        current_action = self._headless_current_action or "idle"
        if self._headless_current_mode:
            current_action = f"{current_action}:{self._headless_current_mode}"

        progress_text = self._current_progress_text()
        status_text = self._current_domain_status_text()
        pipeline_text = self._pipeline_progress_text()
        segments = [f"step=[{step_index}/{step_total}] {current_step}"]
        if current_action and current_action != "idle":
            segments.append(f"action={current_action}")
        if progress_text != "-":
            segments.append(f"progress={progress_text}")
        segments.append(f"status={status_text}")
        segments.append(f"pipeline={pipeline_text}")
        return " | ".join(segments)

    def _render_sticky_header(self) -> None:
        if not self._headless_sticky_header_active:
            return
        terminal_size = shutil.get_terminal_size(fallback=(120, 30))
        width = max(20, terminal_size.columns)
        line = self._truncate_header_line(self._build_footer_line(), width)
        self._terminal_write("\r\x1b[2K")
        self._terminal_write(line)

    def _start_sticky_header(self) -> None:
        if self._headless_sticky_header_active or not self._headless_sticky_header_supported:
            return
        self._headless_sticky_header_active = True
        self._render_sticky_header()

    def _stop_sticky_header(self) -> None:
        if not self._headless_sticky_header_active:
            return
        self._terminal_write("\r\x1b[2K")
        self._headless_sticky_header_active = False
        self._terminal_write("\n")

    def _emit_console_line(self, line: object) -> None:
        text = str(line)
        if not self._headless_sticky_header_active:
            print(text, flush=True)
            return
        self._terminal_write("\r\x1b[2K")
        print(text, flush=True)
        self._render_sticky_header()

    def _install_signal_handlers(self) -> None:
        if self._headless_signal_handlers_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._headless_previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_headless_signal)
            except Exception:
                continue
        self._headless_signal_handlers_installed = True

    def _restore_signal_handlers(self) -> None:
        if not self._headless_signal_handlers_installed:
            return
        for signum, handler in self._headless_previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception:
                continue
        self._headless_previous_signal_handlers = {}
        self._headless_signal_handlers_installed = False

    def _handle_headless_signal(self, signum: int, _frame) -> None:
        self._headless_user_stop_exit_code = self._signal_exit_code(signum)
        try:
            had_activity = bool(self._any_pipeline_activity())
        except Exception:
            had_activity = False

        if had_activity:
            try:
                self._pipeline_stop_active()
                self._render_sticky_header()
            except Exception as exc:
                self._emit_console_line(f"[STOP][ERROR] failed to request stop: {exc}")
            return

        try:
            label = signal.Signals(signum).name
        except Exception:
            label = str(signum)
        self._emit_console_line(f"[STOP] signal {label} received.")

    def _headless_stop_requested_for_action(self, *, step: str, action: str) -> bool:
        if bool(self._pipeline_stop_requested):
            return True
        action_key = str(action or "").strip().lower()
        if action_key == "verify":
            return bool(self._verify_stop_requested)

        domain = self._step_domain(step)
        if domain == "scene":
            return bool(self._scene_stop_requested)
        if domain == "depth":
            return bool(self._depth_stop_requested)
        if domain == "splat":
            return bool(self._splat_stop_requested)
        if domain == "inpaint":
            return bool(self._inpaint_stop_requested)
        if domain == "merge":
            return bool(self._merge_stop_requested)
        if domain == "join":
            return bool(self._join_stop_requested)
        return False

    def _headless_stop_label_for_action(self, *, step: str, action: str) -> str:
        action_key = str(action or "").strip().lower()
        if action_key == "verify":
            return "verification"
        return self.step_label_map().get(str(step or "").strip(), str(step or "").strip() or "pipeline")

    def _finalize_headless_user_stop(self, *, step: str, action: str) -> int:
        self._finalize_pipeline_stop(self._headless_stop_label_for_action(step=step, action=action))
        self._render_sticky_header()
        self._save_pipeline_state()
        return int(self._headless_user_stop_exit_code or 130)

    def _update_ratio_from_line(self, kind: str, line: str) -> None:
        domain = self._HEADLESS_LINE_KIND_DOMAIN_MAP.get(kind, "")
        if not domain:
            return
        matches = self._HEADLESS_RATIO_RE.findall(str(line or ""))
        if not matches:
            return
        current_text, total_text = matches[-1]
        try:
            current = int(current_text)
            total = int(total_text)
        except Exception:
            return
        if total <= 0 or current < 0 or current > total:
            return
        self._headless_ratio_by_domain[domain] = (current, total)
        self._render_sticky_header()

    def _set_current_action(
        self,
        *,
        active_steps: list[str] | None = None,
        step: str = "",
        action: str = "",
        mode: str = "",
        verify_retry_budget: int | None = None,
        verify_attempts: dict[str, int] | None = None,
    ) -> None:
        if active_steps is not None:
            self._headless_active_steps = [str(item).strip() for item in active_steps]
        if verify_retry_budget is not None:
            self._headless_verify_retry_budget = int(verify_retry_budget)
        if verify_attempts is not None:
            self._headless_verify_attempts = dict(verify_attempts)

        self._headless_current_step = str(step or "").strip()
        self._headless_current_action = str(action or "").strip()
        self._headless_current_mode = str(mode or "").strip()

        domain = self._step_domain(self._headless_current_step)
        if domain and self._headless_current_action == "run":
            self._headless_ratio_by_domain.pop(domain, None)
            attr_name = self._HEADLESS_PROGRESS_VAR_MAP.get(domain, "")
            if attr_name:
                try:
                    getattr(self, attr_name).set(0.0)
                except Exception:
                    pass
        self._render_sticky_header()

    def _build_ui(self) -> None:
        return None

    def _apply_option_states(self) -> None:
        return None

    def _poll_log_queue(self) -> None:
        return None

    def _run_startup_tasks(self) -> None:
        return None

    def _refresh_crop_controls_state(self) -> None:
        return None

    def _set_scene_running(self, is_running: bool) -> None:
        if not is_running:
            self._scene_stop_clicks = 0
            self._scene_stop_marker_path = ""

    def _set_depth_running(self, _is_running: bool) -> None:
        return None

    def _set_splat_running(self, _is_running: bool) -> None:
        return None

    def _set_inpaint_running(self, is_running: bool) -> None:
        if not is_running:
            self._inpaint_stop_clicks = 0
            self._inpaint_stop_requested = False
            self._inpaint_stop_marker_path = ""

    def _set_merge_running(self, is_running: bool) -> None:
        if not is_running:
            self._merge_stop_clicks = 0
            self._merge_stop_requested = False
            self._merge_stop_marker_path = ""

    def _set_join_running(self, is_running: bool) -> None:
        if not is_running:
            self._join_stop_requested = False

    def _set_verify_running(self, is_running: bool, mode: str = "") -> None:
        self._verify_running = bool(is_running)
        self._verify_mode = mode if is_running else ""
        if is_running:
            self._verify_stop_requested = False
            self._verify_stop_clicks = 0
        else:
            self._verify_stop_requested = False
            self._verify_stop_clicks = 0

    def _refresh_depth_action_buttons(self, is_running: bool | None = None) -> None:
        return None

    def _refresh_verify_buttons(self) -> None:
        return None

    def _refresh_pipeline_run_button(self) -> None:
        return None

    def _current_window_geometry(self) -> str:
        return self.DEFAULT_WINDOW_GEOMETRY

    def _is_pipeline_popup_suppressed(self) -> bool:
        return True

    def _append_pipeline_popup_log(self, level: str, title: str, message: str) -> None:
        lvl = str(level or "INFO").strip().upper()
        ttl = str(title or "Popup").strip() or "Popup"
        msg = str(message or "").strip() or "(empty)"
        self._emit_console_line(f"[POPUP][{lvl}] {ttl}: {msg}")

    def _append_scene_log(self, line: str) -> None:
        self._emit_console_line(str(line))

    def _append_depth_log(self, line: str) -> None:
        self._emit_console_line(str(line))

    def _append_splat_log(self, line: str) -> None:
        self._emit_console_line(str(line))

    def _append_inpaint_log(self, line: str) -> None:
        self._emit_console_line(str(line))

    def _append_merge_log(self, line: str) -> None:
        self._emit_console_line(str(line))

    def _append_join_log(self, line: str) -> None:
        self._emit_console_line(str(line))

    def _clear_scene_log(self) -> None:
        return None

    def _clear_depth_log(self) -> None:
        return None

    def _clear_splat_log(self) -> None:
        return None

    def _clear_inpaint_log(self) -> None:
        return None

    def _clear_merge_log(self) -> None:
        return None

    def _clear_join_log(self) -> None:
        return None

    def close(self) -> None:
        self._stop_sticky_header()
        self._restore_signal_handlers()
        self._restore_messagebox_wrappers()
        try:
            self._headless_interp.destroy()
        except Exception:
            pass

    @classmethod
    def ordered_step_keys(cls) -> list[str]:
        return [str(step).strip() for step, _label in cls.PIPELINE_STEPS]

    @classmethod
    def step_label_map(cls) -> dict[str, str]:
        return {str(step).strip(): str(label).strip() for step, label in cls.PIPELINE_STEPS}

    @staticmethod
    def _normalize_step_token(value: object) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        while "__" in token:
            token = token.replace("__", "_")
        return token

    @classmethod
    def resolve_step_name(cls, value: object) -> str:
        token = cls._normalize_step_token(value)
        if not token:
            raise ValueError("Step name cannot be empty.")

        alias_map: dict[str, str] = {}
        for step, label in cls.PIPELINE_STEPS:
            step_key = str(step).strip()
            label_text = str(label).strip()
            alias_map[cls._normalize_step_token(step_key)] = step_key
            alias_map[cls._normalize_step_token(label_text)] = step_key
        alias_map["depth"] = "depthcrafter"
        alias_map["splat"] = "splatting"
        alias_map["merge"] = "merging"
        alias_map["mono"] = "mono_to_sbs"
        alias_map["mono_to_sbs"] = "mono_to_sbs"
        alias_map["mono2sbs"] = "mono_to_sbs"

        resolved = alias_map.get(token, "")
        if resolved:
            return resolved
        available = ", ".join(cls.ordered_step_keys())
        raise ValueError(f"Unknown step '{value}'. Available steps: {available}")

    @classmethod
    def format_step_listing(cls) -> str:
        verify_steps = set(cls._verify_step_specs().keys())
        lines: list[str] = []
        for step, label in cls.PIPELINE_STEPS:
            verify_tag = " | quick-verify" if step in verify_steps else ""
            lines.append(f"{step:<14} {label}{verify_tag}")
        return "\n".join(lines)

    def build_step_scope(
        self,
        *,
        from_step: str = "",
        to_step: str = "",
        only_step: str = "",
    ) -> list[str]:
        if str(only_step or "").strip() and (
            str(from_step or "").strip() or str(to_step or "").strip()
        ):
            raise ValueError("--only-step cannot be combined with --from-step/--to-step.")

        ordered = self.ordered_step_keys()
        if str(only_step or "").strip():
            only_resolved = self.resolve_step_name(only_step)
            return [only_resolved]

        start_idx = 0
        end_idx = len(ordered) - 1
        if str(from_step or "").strip():
            start_idx = ordered.index(self.resolve_step_name(from_step))
        if str(to_step or "").strip():
            end_idx = ordered.index(self.resolve_step_name(to_step))
        if start_idx > end_idx:
            raise ValueError("--from-step cannot come after --to-step.")
        return ordered[start_idx : end_idx + 1]

    def invalidate_from_step(self, step: str) -> None:
        self._pipeline_invalidate_from(step)
        self._render_sticky_header()
        self._emit_console_line(f"[PIPELINE] invalidated run state from step={step}")

    def run_pipeline(
        self,
        *,
        max_verify_retries: int = 1,
        from_step: str = "",
        to_step: str = "",
        only_step: str = "",
    ) -> int:
        try:
            self._install_signal_handlers()
            self._headless_user_stop_exit_code = None
            self._pipeline_autorun = True
            self._pipeline_stop_requested = False
            self._pipeline_pending_action = None
            active_steps = self.build_step_scope(
                from_step=from_step,
                to_step=to_step,
                only_step=only_step,
            )
            self._set_current_action(
                active_steps=active_steps,
                verify_retry_budget=max_verify_retries,
                verify_attempts={},
            )
            self._start_sticky_header()
            if (
                str(from_step or "").strip()
                or str(to_step or "").strip()
                or str(only_step or "").strip()
            ):
                self.invalidate_from_step(active_steps[0])
            self._emit_console_line(f"[PIPELINE] active scope: {', '.join(active_steps)}")

            retry_budget = max(0, int(max_verify_retries))
            verify_retries: dict[str, int] = {}
            while True:
                action = self._pipeline_next_action_for_scope(active_steps)
                if action is None:
                    self._pipeline_pending_action = None
                    self.pipeline_run_status_var.set("Pipeline completed.")
                    self._set_current_action(
                        active_steps=active_steps,
                        action="done",
                        verify_retry_budget=retry_budget,
                        verify_attempts=verify_retries,
                    )
                    self._emit_console_line("Pipeline headless completed.")
                    return 0
                step, act, mode = action
                self._pipeline_pending_action = (step, act, mode)
                self._set_current_action(
                    active_steps=active_steps,
                    step=step,
                    action=act,
                    mode=mode,
                    verify_retry_budget=retry_budget,
                    verify_attempts=verify_retries,
                )
                self._emit_console_line(f"[PIPELINE] next action: step={step} action={act} mode={mode}")
                if act == "run":
                    verify_retries.pop(step, None)
                    success, mark_completed = self._run_step_headless(step)
                    stop_requested = self._headless_stop_requested_for_action(step=step, action=act)
                    if stop_requested:
                        if success:
                            self._pipeline_set_completed(step, bool(mark_completed))
                            self._pipeline_set_verified(step, "none")
                            if step in self.PIPELINE_CSV_STEPS:
                                self._sync_pipeline_csv_done_flags()
                            self._refresh_pipeline_status_panel()
                            self._render_sticky_header()
                        return self._finalize_headless_user_stop(step=step, action=act)
                    if not success:
                        self.pipeline_run_status_var.set(f"Pipeline stopped: step failed ({step})")
                        self._render_sticky_header()
                        self._save_pipeline_state()
                        return 1
                    self._pipeline_set_completed(step, bool(mark_completed))
                    self._pipeline_set_verified(step, "none")
                    if step in self.PIPELINE_CSV_STEPS:
                        self._sync_pipeline_csv_done_flags()
                    self._refresh_pipeline_status_panel()
                    self._render_sticky_header()
                    self._save_pipeline_state()
                    if not mark_completed:
                        self.pipeline_run_status_var.set(
                            f"Pipeline paused: {step} completed in incomplete mode."
                        )
                        self._render_sticky_header()
                        return 2
                    continue

                success, retryable = self._verify_step_headless(step, mode)
                stop_requested = self._headless_stop_requested_for_action(step=step, action=act)
                if stop_requested:
                    return self._finalize_headless_user_stop(step=step, action=act)
                if success:
                    self._pipeline_mark_previous_steps_done_verified_in_state(
                        self._pipeline_step_state,
                        step,
                        mode,
                    )
                    self._sync_pipeline_csv_done_flags()
                    self._refresh_pipeline_status_panel()
                    self._render_sticky_header()
                    self._save_pipeline_state()
                    verify_retries.pop(step, None)
                    continue

                self._pipeline_invalidate_from(step)
                self._render_sticky_header()
                attempts = verify_retries.get(step, 0)
                if retryable and attempts < retry_budget:
                    verify_retries[step] = attempts + 1
                    self._set_current_action(
                        active_steps=active_steps,
                        step=step,
                        action="retrying-run",
                        verify_retry_budget=retry_budget,
                        verify_attempts=verify_retries,
                    )
                    self._emit_console_line(
                        (
                            f"[PIPELINE] verify failed on {step}; "
                            f"retrying run ({verify_retries[step]}/{retry_budget})."
                        )
                    )
                    continue

                self.pipeline_run_status_var.set(
                    f"Pipeline stopped: verify failed ({step})"
                )
                self._render_sticky_header()
                self._save_pipeline_state()
                return 1
        finally:
            self._pipeline_pending_action = None
            self._pipeline_autorun = False
            self.close()

    def _pipeline_next_action_for_scope(
        self,
        active_steps: list[str],
    ) -> tuple[str, str, str] | None:
        verify_mode = self.pipeline_verify_after_var.get().strip().lower()
        active_set = set(active_steps)
        for step, _label in self.PIPELINE_STEPS:
            if step not in active_set:
                continue
            if not self._is_pipeline_step_required(step):
                continue
            st = self._pipeline_step_state.get(step, {"completed": False, "verified": "none"})
            if not bool(st.get("completed", False)):
                return step, "run", "none"
            if step in self.PIPELINE_STEPS_WITH_VERIFY and verify_mode == "quick":
                current = str(st.get("verified", "none")).strip().lower()
                if current != "quick":
                    return step, "verify", "quick"
        return None

    def _run_step_headless(self, step: str) -> tuple[bool, bool]:
        spec = self._run_step_specs().get(step)
        if spec is None:
            self._emit_console_line(f"[PIPELINE] unsupported run step: {step}")
            return False, True

        getattr(self, spec["start"])()
        thread = getattr(self, spec["thread"], None)
        if thread is None or not thread.is_alive():
            self._emit_console_line(f"[PIPELINE] {step} did not start.")
            return False, True

        events = self._wait_for_thread(spec["thread"])
        success = self._extract_run_success(step, spec, events)
        mark_completed = True
        if step == "join":
            mark_completed = bool(self._join_mark_completed)
            self._join_mark_completed = True
        return success, mark_completed

    def _verify_step_headless(self, step: str, mode: str) -> tuple[bool, bool]:
        mode_low = str(mode or "").strip().lower()
        if mode_low != "quick":
            self._emit_console_line(f"[PIPELINE] unsupported verify mode for {step}: {mode}")
            return False, False

        spec = self._verify_step_specs().get(step)
        if spec is None:
            self._emit_console_line(f"[PIPELINE] unsupported verify step: {step}")
            return False, False

        getattr(self, spec["start"])()
        thread = getattr(self, spec["thread"], None)
        if thread is None or not thread.is_alive():
            self._emit_console_line(f"[PIPELINE] verify for {step} did not start.")
            return False, False

        events = self._wait_for_thread(spec["thread"])
        payload = self._extract_last_event(events, spec["result_kind"])
        if not isinstance(payload, dict):
            return False, False
        ok = bool(payload.get("ok", False))
        retryable = bool(payload.get("retryable_failure", True))
        return ok, retryable

    @classmethod
    def _run_step_specs(cls) -> dict[str, dict[str, str]]:
        return {
            "scenedetect": {"start": "_start_scene_detect", "thread": "_scene_thread", "done_kind": "done"},
            "split_scenes": {"start": "_start_split_scenes", "thread": "_scene_thread", "done_kind": "done"},
            "depthcrafter": {"start": "_run_depth_placeholder", "thread": "_depth_thread", "done_kind": "depth_done"},
            "splatting": {"start": "_run_splat_placeholder", "thread": "_splat_thread", "done_kind": "splat_done"},
            "sharpness_csv": {"start": "_start_inpaint_sharpness_csv", "thread": "_inpaint_thread", "done_kind": "inpaint_done"},
            "inpaint": {"start": "_run_inpaint_placeholder", "thread": "_inpaint_thread", "done_kind": "inpaint_done"},
            "sharpen": {"start": "_start_inpaint_sharpen", "thread": "_inpaint_thread", "done_kind": "inpaint_done"},
            "mask_for_merge": {"start": "_run_merge_mask_placeholder", "thread": "_merge_thread", "done_kind": "merge_done"},
            "autoct_csv": {"start": "_start_merge_autoct_csv", "thread": "_merge_thread", "done_kind": "merge_done"},
            "merging": {"start": "_run_merge_placeholder", "thread": "_merge_thread", "done_kind": "merge_done"},
            "mono_to_sbs": {"start": "_run_join_prepare_mono", "thread": "_join_thread", "done_kind": "join_done"},
            "join": {"start": "_run_join_scenes", "thread": "_join_thread", "done_kind": "join_done"},
            "remux": {"start": "_start_join_remux", "thread": "_join_thread", "done_kind": "join_done"},
        }

    @classmethod
    def _verify_step_specs(cls) -> dict[str, dict[str, str]]:
        return {
            "split_scenes": {"start": "_start_verify_quick", "thread": "_verify_thread", "result_kind": "verify_quick_result"},
            "depthcrafter": {"start": "_start_depth_verify_quick", "thread": "_verify_thread", "result_kind": "depth_verify_quick_result"},
            "splatting": {"start": "_start_splat_verify_quick", "thread": "_verify_thread", "result_kind": "splat_verify_quick_result"},
            "inpaint": {"start": "_start_inpaint_verify_quick", "thread": "_verify_thread", "result_kind": "inpaint_verify_quick_result"},
            "sharpen": {"start": "_start_inpaint_sharpen_verify_quick", "thread": "_verify_thread", "result_kind": "sharpen_verify_quick_result"},
            "mask_for_merge": {"start": "_start_merge_mask_verify_quick", "thread": "_verify_thread", "result_kind": "merge_mask_verify_quick_result"},
            "merging": {"start": "_start_merge_verify_quick", "thread": "_verify_thread", "result_kind": "merge_verify_quick_result"},
            "mono_to_sbs": {"start": "_start_join_mono_verify", "thread": "_verify_thread", "result_kind": "join_mono_verify_result"},
            "join": {"start": "_start_join_verify", "thread": "_verify_thread", "result_kind": "join_verify_result"},
        }

    def _wait_for_thread(self, attr_name: str) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        while True:
            try:
                events.extend(self._drain_log_queue())
                thread = getattr(self, attr_name, None)
                if thread is None or not thread.is_alive():
                    break
                time.sleep(0.2)
            except KeyboardInterrupt:
                self._handle_headless_signal(signal.SIGINT, None)
                continue
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            self._handle_headless_signal(signal.SIGINT, None)
        events.extend(self._drain_log_queue())
        setattr(self, attr_name, None)
        return events

    def _drain_log_queue(self) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        while True:
            try:
                kind, payload = self._log_queue.get_nowait()
            except queue.Empty:
                break
            events.append((kind, payload))
            self._handle_headless_event(kind, payload)
        return events

    def _handle_headless_event(self, kind: str, payload: object) -> None:
        line_kinds = {
            "line",
            "depth_line",
            "splat_line",
            "inpaint_line",
            "merge_line",
            "join_line",
        }
        if kind in line_kinds:
            self._update_ratio_from_line(kind, str(payload))
            self._emit_console_line(str(payload))
            return

        status_vars = {
            "status": self.scene_status_var,
            "depth_status": self.depth_status_var,
            "splat_status": self.splat_status_var,
            "inpaint_status": self.inpaint_status_var,
            "merge_status": self.merge_status_var,
            "join_status": self.join_status_var,
        }
        if kind in status_vars:
            text = str(payload)
            status_vars[kind].set(text)
            self._render_sticky_header()
            self._emit_console_line(f"[STATUS] {text}")
            return

        progress_vars = {
            "progress": ("scene", self.scene_progress_var),
            "depth_progress": ("depth", self.depth_progress_var),
            "splat_progress": ("splat", self.splat_progress_var),
            "inpaint_progress": ("inpaint", self.inpaint_progress_var),
            "merge_progress": ("merge", self.merge_progress_var),
            "join_progress": ("join", self.join_progress_var),
        }
        if kind in progress_vars:
            label, var = progress_vars[kind]
            try:
                value = max(0.0, min(100.0, float(payload)))
            except Exception:
                return
            var.set(value)
            self._render_sticky_header()
            bucket = int(value)
            cache_name = f"_headless_progress_{label}"
            last_bucket = getattr(self, cache_name, -1)
            if bucket != last_bucket and (bucket == 100 or bucket % 5 == 0):
                setattr(self, cache_name, bucket)
                self._emit_console_line(f"[PROGRESS][{label}] {bucket}%")
            return

        if kind == "verify_done":
            self._set_verify_running(False)
            self._render_sticky_header()
            return

    @staticmethod
    def _extract_last_event(
        events: list[tuple[str, object]], kind: str
    ) -> object | None:
        for event_kind, payload in reversed(events):
            if event_kind == kind:
                return payload
        return None

    def _extract_run_success(
        self,
        step: str,
        spec: dict[str, str],
        events: list[tuple[str, object]],
    ) -> bool:
        payload = self._extract_last_event(events, spec["done_kind"])
        if spec["done_kind"] == "done" and isinstance(payload, dict):
            return (
                str(payload.get("step", "")).strip().lower() == step
                and bool(payload.get("success", False))
            )
        if spec["done_kind"] in {"depth_done", "inpaint_done", "merge_done", "join_done"} and isinstance(payload, dict):
            payload_step = str(payload.get("step", "")).strip().lower()
            if payload_step and payload_step != step:
                return False
            if "success" in payload:
                return bool(payload.get("success", False))

        if step == "splatting":
            return "completed" in self.splat_status_var.get().strip().lower()
        if step == "remux":
            remux_path = Path(self._default_remux_output_path())
            return remux_path.is_file() and "completed" in self.join_status_var.get().strip().lower()
        if step in {"scenedetect", "split_scenes"}:
            return "completed" in self.scene_status_var.get().strip().lower()
        if step == "depthcrafter":
            return "completed" in self.depth_status_var.get().strip().lower()
        if step in {"sharpness_csv", "inpaint", "sharpen"}:
            return "created" in self.inpaint_status_var.get().strip().lower() or "completed" in self.inpaint_status_var.get().strip().lower()
        if step in {"mask_for_merge", "autoct_csv", "merging"}:
            return "created" in self.merge_status_var.get().strip().lower() or "completed" in self.merge_status_var.get().strip().lower()
        if step in {"mono_to_sbs", "join"}:
            return "completed" in self.join_status_var.get().strip().lower()
        return False

    def print_active_paths(self) -> None:
        self._emit_console_line(f"[CONFIG] file: {self._config_file}")
        self._emit_console_line(f"[WORK] folder: {self.work_folder_var.get().strip()}")
        self._emit_console_line(f"[STATE] file: {self._pipeline_state_path()}")
        self._emit_console_line(
            f"[VERIFY] mode: {self.pipeline_verify_after_var.get().strip() or 'Disabled'}"
        )

    def build_sharpness_csv_command(self) -> list[str]:
        script_path = utilities_path("analyze_inpaint_sharpness.py")
        input_dir = self.inpaint_input_var.get().strip()
        mask_dir = self.inpaint_mask_var.get().strip()
        work_dir = self.work_folder_var.get().strip() or "./work"
        out_csv = Path(work_dir).resolve() / "sharpness.csv"
        self.inpaint_sharpness_csv_var.set(str(out_csv))
        workers = max(1, int(self.inpaint_sharpness_workers_var.get().strip() or "19"))
        stop_marker = str(out_csv.parent / ".stop_after_current")
        return [
            sys.executable,
            str(script_path),
            str(Path(input_dir).resolve()),
            str(Path(mask_dir).resolve()),
            "--out_csv",
            str(out_csv),
            "--workers",
            str(workers),
            "--stop-marker",
            stop_marker,
        ]

    def build_autoct_csv_command(self) -> list[str]:
        script_path = utilities_path("analyze_auto_ct_csv.py")
        inpainted = self.merge_inpainted_var.get().strip()
        preferred_inpainted = self._preferred_inpainted_dir_for_consumers()
        splatted = self.merge_splatted_var.get().strip()
        original = self.merge_original_var.get().strip()
        out_csv_path = Path(self.merge_autoct_csv_var.get().strip()).resolve()
        mask_folder = self.merge_replace_mask_var.get().strip()
        workers = max(1, int(self.merge_autoct_workers_var.get().strip() or "8"))
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
        return cmd
