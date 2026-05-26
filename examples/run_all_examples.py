#!/usr/bin/env python3
"""Run all Disseqt SDK examples with structured logging.

Usage:
    python examples/run_all_examples.py
    python examples/run_all_examples.py --log-level DEBUG
    python examples/run_all_examples.py --log-file logs/sdk_examples.log
    python examples/run_all_examples.py --examples input output rag
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure local src/ is used over any installed package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("disseqt.examples")


# ---------------------------------------------------------------------------
# Example registry
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent

EXAMPLES: dict[str, dict] = {
    "input": {
        "label": "Input Validation",
        "file": "example_input_validation.py",
        "description": "15 input validators — Safety, Bias, Security, Child Safety",
    },
    "output": {
        "label": "Output Validation",
        "file": "example_output_validation.py",
        "description": "32 output validators — Quality, Safety, Bias, Security, Scoring",
    },
    "rag": {
        "label": "RAG Grounding",
        "file": "example_rag_grounding.py",
        "description": "7 RAG validators — Context relevance, faithfulness, recall, precision",
    },
    "agentic": {
        "label": "Agentic Behavior",
        "file": "example_agentic_behavior.py",
        "description": "8 agentic validators — Goal accuracy, tool usage, planning, intent",
    },
    "mcp": {
        "label": "MCP / Security",
        "file": "example_mcp_security.py",
        "description": "3 MCP security validators — Prompt injection, data leakage, insecure output",
    },
    "composite": {
        "label": "Composite Score",
        "file": "example_composite_score.py",
        "description": "Composite Score API — 18 metrics across 3 categories",
    },
    "prompt_packs": {
        "label": "Prompt Packs API",
        "file": "example_prompt_packs.py",
        "description": "Full Prompt Packs lifecycle — generate, run, validate, export CSV",
    },
    "general": {
        "label": "General / Mixed",
        "file": "example.py",
        "description": "Mixed examples covering multiple validators",
    },
}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class ExampleResult:
    key: str
    label: str
    file: str
    status: str = "pending"  # pending | ok | error | skipped
    elapsed_s: float = 0.0
    error: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def load_and_run(path: Path, log: logging.Logger) -> tuple[bool, str]:
    """Import and execute `main()` from the given example file."""
    spec = importlib.util.spec_from_file_location("_example_mod", path)
    if spec is None or spec.loader is None:
        return False, f"Cannot load spec for {path}"

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # module-level code / imports
    except Exception as e:
        return False, f"Import error: {e}"

    if not hasattr(mod, "main"):
        return False, "No main() function found"

    try:
        mod.main()
        return True, ""
    except Exception as e:
        return False, f"Runtime error: {e}"


def run_example(key: str, meta: dict, log: logging.Logger) -> ExampleResult:
    result = ExampleResult(key=key, label=meta["label"], file=meta["file"])
    path = EXAMPLES_DIR / meta["file"]

    log.info("=" * 60)
    log.info("START  %-20s  %s", meta["label"], meta["description"])
    log.info("File:  %s", path)
    log.info("=" * 60)

    if not path.exists():
        result.status = "skipped"
        result.notes.append(f"File not found: {path}")
        log.warning("SKIP   %s — file not found", meta["label"])
        return result

    t0 = time.perf_counter()
    ok, err = load_and_run(path, log)
    result.elapsed_s = time.perf_counter() - t0

    if ok:
        result.status = "ok"
        log.info("DONE   %-20s  (%.2fs)", meta["label"], result.elapsed_s)
    else:
        result.status = "error"
        result.error = err
        log.error("FAIL   %-20s  (%.2fs)  —  %s", meta["label"], result.elapsed_s, err)

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[ExampleResult], log: logging.Logger, total_s: float) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY  —  %d examples  (%.2fs total)", len(results), total_s)
    log.info("=" * 60)

    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    skipped = [r for r in results if r.status == "skipped"]

    for r in results:
        icon = {"ok": "✓", "error": "✗", "skipped": "⊘"}.get(r.status, "?")
        msg = f"  {icon}  {r.label:<28} {r.elapsed_s:>6.2f}s"
        if r.error:
            msg += f"  →  {r.error}"
        if r.status == "ok":
            log.info(msg)
        elif r.status == "error":
            log.error(msg)
        else:
            log.warning(msg)

    log.info("")
    log.info("  Passed : %d", len(ok))
    if errors:
        log.error("  Failed : %d", len(errors))
    if skipped:
        log.warning("  Skipped: %d", len(skipped))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all (or selected) Disseqt SDK examples with structured logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([f"  {k:<15} {v['description']}" for k, v in EXAMPLES.items()]),
    )
    parser.add_argument(
        "--examples",
        "-e",
        nargs="*",
        choices=list(EXAMPLES.keys()),
        metavar="NAME",
        help="Examples to run (default: all). Choices: " + ", ".join(EXAMPLES.keys()),
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        "-f",
        default=None,
        metavar="PATH",
        help="Also write logs to this file (e.g. logs/run.log)",
    )
    parser.add_argument(
        "--stop-on-error",
        "-x",
        action="store_true",
        help="Stop immediately on first failure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logging(level=args.log_level, log_file=args.log_file)

    keys_to_run = args.examples if args.examples else list(EXAMPLES.keys())

    log.info("Disseqt SDK — Example Runner")
    log.info("Running %d example(s): %s", len(keys_to_run), ", ".join(keys_to_run))
    if args.log_file:
        log.info("Log file: %s", args.log_file)
    log.info("")

    results: list[ExampleResult] = []
    wall_start = time.perf_counter()

    for key in keys_to_run:
        meta = EXAMPLES[key]
        result = run_example(key, meta, log)
        results.append(result)
        log.info("")

        if args.stop_on_error and result.status == "error":
            log.error("Stopping early due to --stop-on-error flag.")
            break

    total_s = time.perf_counter() - wall_start
    print_summary(results, log, total_s)

    failed = sum(1 for r in results if r.status == "error")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
