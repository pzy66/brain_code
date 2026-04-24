from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
ASYNC_DIR = THIS_DIR / "2026-04_async_fbcca_idle_decoder"
TARGET_SCRIPT = ASYNC_DIR / "ssvep_dataset_collection_ui.py"
DEFAULT_STIM_REFRESH_RATE_HZ = 60.0
DEFAULT_STIM_MEAN = 0.5
DEFAULT_STIM_AMP = 0.5
DEFAULT_STIM_PHI = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct entry: SSVEP dataset collection UI.")
    parser.add_argument("--serial-port", type=str, default="auto", help="serial port, e.g. COM4, or auto")
    parser.add_argument("--board-id", type=int, default=0)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--dataset-dir", type=str, default=str(ASYNC_DIR / "profiles" / "datasets"))
    parser.add_argument("--subject-id", type=str, default="subject001")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--session-index", type=int, default=1)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="forward remaining args to ssvep_dataset_collection_ui.py",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script = TARGET_SCRIPT
    if not script.exists():
        raise FileNotFoundError(f"missing script: {script}")
    if str(ASYNC_DIR) not in sys.path:
        sys.path.insert(0, str(ASYNC_DIR))

    import async_fbcca_validation_ui as stim_ui  # type: ignore[import-not-found]

    init_sig = inspect.signature(stim_ui.FourArrowStimWidget.__init__)
    needs_compat_patch = (
        "freqs" in init_sig.parameters
        and init_sig.parameters["freqs"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    if needs_compat_patch:
        base_cls = stim_ui.FourArrowStimWidget

        class _CompatFourArrowStimWidget(base_cls):
            def __init__(self, *cargs, **ckwargs):
                arg_list = list(cargs)
                if arg_list and "freqs" not in ckwargs:
                    ckwargs["freqs"] = arg_list.pop(0)
                if arg_list and "refresh_rate_hz" not in ckwargs:
                    ckwargs["refresh_rate_hz"] = float(arg_list.pop(0))
                if arg_list and "mean" not in ckwargs:
                    ckwargs["mean"] = float(arg_list.pop(0))
                if arg_list and "amp" not in ckwargs:
                    ckwargs["amp"] = float(arg_list.pop(0))
                if arg_list and "phi" not in ckwargs:
                    ckwargs["phi"] = float(arg_list.pop(0))
                if arg_list and "parent" not in ckwargs:
                    ckwargs["parent"] = arg_list.pop(0)
                ckwargs.setdefault("refresh_rate_hz", float(DEFAULT_STIM_REFRESH_RATE_HZ))
                ckwargs.setdefault("mean", float(DEFAULT_STIM_MEAN))
                ckwargs.setdefault("amp", float(DEFAULT_STIM_AMP))
                ckwargs.setdefault("phi", float(DEFAULT_STIM_PHI))
                if arg_list:
                    raise TypeError(
                        "FourArrowStimWidget compat patch received unexpected positional args: "
                        f"{arg_list}"
                    )
                super().__init__(**ckwargs)

        stim_ui.FourArrowStimWidget = _CompatFourArrowStimWidget

    import ssvep_dataset_collection_ui as collection_ui  # type: ignore[import-not-found]

    forward_argv = [
        "--serial-port",
        str(args.serial_port),
        "--board-id",
        str(int(args.board_id)),
        "--freqs",
        str(args.freqs),
        "--dataset-dir",
        str(args.dataset_dir),
        "--subject-id",
        str(args.subject_id),
        "--session-index",
        str(int(args.session_index)),
    ]
    if str(args.session_id).strip():
        forward_argv.extend(["--session-id", str(args.session_id).strip()])
    if args.extra_args:
        forward_argv.extend(list(args.extra_args))

    print(f"[launcher] {sys.executable} {TARGET_SCRIPT} {' '.join(forward_argv)}", flush=True)
    return int(collection_ui.main(forward_argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
