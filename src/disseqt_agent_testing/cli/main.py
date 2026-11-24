"""The ``disseqt`` console script: a small argparse front end with a subcommand registry."""

import sys
import threading
from collections.abc import Callable, Sequence

Subcommand = Callable[[list[str]], int]
_LOCK = threading.RLock()
_COMMANDS: dict[str, Subcommand] = {}


def register_subcommand(name: str, fn: Subcommand, *, override: bool = False) -> None:
    """``fn(argv) -> exit_code``. ``test`` and ``judges`` are built in."""
    with _LOCK:
        if name in _COMMANDS and not override:
            raise ValueError(f"subcommand {name!r} is already registered")
        _COMMANDS[name] = fn


def unregister_subcommand(name: str) -> None:
    with _LOCK:
        _COMMANDS.pop(name, None)


def subcommands() -> tuple[str, ...]:
    _builtin()
    with _LOCK:
        return tuple(sorted(_COMMANDS))


def _builtin() -> None:
    if "test" not in _COMMANDS:
        from .test_cmd import run_test_command

        register_subcommand("test", run_test_command, override=True)
    if "judges" not in _COMMANDS:
        from ..judges.cli import main as judges_main

        register_subcommand("judges", judges_main, override=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _builtin()
    if not args or args[0] in ("-h", "--help"):
        names = ", ".join(subcommands())
        print(
            f"usage: disseqt <command> [options]\n\ncommands: {names}\n\nrun 'disseqt test --help' for the test runner"
        )
        return 0 if args else 3  # usage error: 2 means incomplete
    with _LOCK:
        fn = _COMMANDS.get(args[0])
    if fn is None:
        print(
            f"disseqt: unknown command '{args[0]}' (known: {', '.join(subcommands())})",
            file=sys.stderr,
        )
        return 3
    try:
        return fn(args[1:])
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
