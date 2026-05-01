#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP_HELP_LINES = [
    "scenedetect    SceneDetect",
    "split_scenes   Split Scenes | quick-verify",
    "depthcrafter   DepthCrafter | quick-verify",
    "splatting      Splatting | quick-verify",
    "sharpness_csv  Sharpness CSV",
    "inpaint        Inpaint | quick-verify",
    "sharpen        Sharpen | quick-verify",
    "mask_for_merge Mask-for-merge | quick-verify",
    "autoct_csv     AutoCT CSV",
    "merging        Merging | quick-verify",
    "mono_to_sbs    Mono->SBS | quick-verify",
    "join           Join | quick-verify",
    "remux          Remux",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline Master headless runner. "
            "With --work_dir, it uses <work_dir>/config_pipeline_master_gui.json "
            "for settings when present and <work_dir>/pipeline_state.json for resume state."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python runners/pipeline_master_headless.py --work_dir /data/work\n"
            "  python runners/pipeline_master_headless.py --work_dir /data/work --from-step depthcrafter --to-step merging\n"
            "  python runners/pipeline_master_headless.py --work_dir /data/work --only-step join\n"
            "  python runners/pipeline_master_headless.py --list-steps\n"
        ),
    )
    parser.add_argument(
        "--work_dir",
        default="",
        help=(
            "Single work directory used as the local source of truth.\n"
            "If set, config is auto-loaded from <work_dir>/config_pipeline_master_gui.json when present,\n"
            "and resume state is always read/written as <work_dir>/pipeline_state.json."
        ),
    )
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Optional explicit config JSON path.\n"
            "Overrides the automatic config path selection."
        ),
    )
    parser.add_argument(
        "--verify-after",
        choices=["config", "quick", "none"],
        default="config",
        help=(
            "Verification mode override.\n"
            "'config' keeps the saved Pipeline Master setting."
        ),
    )
    parser.add_argument(
        "--max-verify-retries",
        type=int,
        default=1,
        help="How many auto reruns to allow after a quick-verify failure.",
    )
    parser.add_argument(
        "--from-step",
        default="",
        help=(
            "Start the headless run from this step onward.\n"
            "This invalidates run state from the selected step before starting."
        ),
    )
    parser.add_argument(
        "--to-step",
        default="",
        help=(
            "Stop the headless run after this step (and its quick verify, if enabled).\n"
            "Can be combined with --from-step."
        ),
    )
    parser.add_argument(
        "--only-step",
        default="",
        help=(
            "Run just one step (and its quick verify, if enabled).\n"
            "Cannot be combined with --from-step/--to-step."
        ),
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print the canonical step names accepted by --from-step/--to-step/--only-step and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_steps:
        print("\n".join(STEP_HELP_LINES))
        raise SystemExit(0)

    from core.pipeline_master.headless import HeadlessPipelineMaster

    runner = HeadlessPipelineMaster(
        work_dir=args.work_dir,
        config_file=args.config,
        verify_after=args.verify_after,
    )
    runner.print_active_paths()
    try:
        exit_code = runner.run_pipeline(
            max_verify_retries=args.max_verify_retries,
            from_step=args.from_step,
            to_step=args.to_step,
            only_step=args.only_step,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        try:
            runner.close()
        except Exception:
            pass
        raise SystemExit(130)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
