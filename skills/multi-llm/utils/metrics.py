"""Metrics utility for recording and reporting Task tool resource usage.

Records per-subagent metrics (tokens, tool uses, duration) into state JSON
and generates markdown reports for individual phases or cross-phase summaries.
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


def _read_state(state_path: Path) -> dict:
    """Read and parse the state JSON file."""
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_state(state_path: Path, state: dict) -> None:
    """Write state JSON atomically (temp file + os.replace)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=str(state_path.parent),
        suffix=".tmp",
        prefix=".metrics_",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        temp_fd = None  # fdopen closes the fd
        os.replace(temp_path, str(state_path))
        temp_path = None  # file has been moved
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def format_duration(ms: Optional[int]) -> str:
    """Format milliseconds as human-readable duration.

    Examples:
        83000  -> "1m 23s"
        52000  -> "0m 52s"
        500    -> "<1s"
        0      -> "0s"
        None   -> "-"
    """
    if ms is None:
        return "-"
    if ms == 0:
        return "0s"
    if ms < 1000:
        return "<1s"
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}m {seconds}s"


def format_number(n: Optional[int]) -> str:
    """Format an integer with comma separators.

    Examples:
        12450 -> "12,450"
        0     -> "0"
        None  -> "-"
    """
    if n is None:
        return "-"
    return f"{n:,}"


def _escape_pipe(text: str) -> str:
    """Escape pipe characters for markdown table cells."""
    return text.replace("|", "\\|")


def _should_colorize_stderr() -> bool:
    """True when ANSI color codes should be emitted on stderr.

    Suppressed when ``NO_COLOR`` is set (https://no-color.org/) or when stderr
    is not a TTY (e.g. piped to a file or captured by pytest).
    """
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def _colorize(text: str, code: str, enabled: bool) -> str:
    """Wrap ``text`` in an ANSI color escape when ``enabled``."""
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def start_phase(state_path: Path, phase: str, total_batches: int) -> None:
    """Record the start time and total batch count for a phase.

    Writes ``metrics_progress.{phase}`` = {"started_at": ISO timestamp,
    "total_batches": int}. Used by ``record_metric`` to compute wall-clock ETA.
    Idempotent: a second call for the same phase replaces the prior entry
    (necessary so ``--resume`` runs reset the progress baseline).
    """
    state = _read_state(state_path)
    if "metrics_progress" not in state:
        state["metrics_progress"] = {}
    state["metrics_progress"][phase] = {
        "started_at": datetime.now().isoformat(),
        "total_batches": int(total_batches),
    }
    _write_state(state_path, state)


def finish_phase(state_path: Path, phase: str) -> Optional[int]:
    """Remove ``metrics_progress.{phase}`` and return total elapsed ms.

    Returns None if the phase was never started (graceful no-op).
    """
    state = _read_state(state_path)
    progress = state.get("metrics_progress", {})
    entry = progress.pop(phase, None)
    if entry is None:
        return None

    elapsed_ms: Optional[int] = None
    started_at_raw = entry.get("started_at")
    if started_at_raw:
        try:
            started_at = datetime.fromisoformat(started_at_raw)
            elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
        except (ValueError, TypeError):
            elapsed_ms = None

    # Clean up empty progress dict
    if not progress and "metrics_progress" in state:
        del state["metrics_progress"]
    else:
        state["metrics_progress"] = progress

    _write_state(state_path, state)

    color_on = _should_colorize_stderr()
    complete_segment = _colorize(
        f"phase complete in {format_duration(elapsed_ms)}", "1;32", color_on,
    )
    print(f"[ETA] {phase}: {complete_segment}", file=sys.stderr)
    return elapsed_ms


def _compute_eta(
    entries: list,
    progress: Optional[dict],
    phase: str,
    batch_index: int,
    total_batches: int,
    this_duration_ms: Optional[int],
    colorize: bool = False,
) -> dict:
    """Compute ETA from per-item rate and wall-clock rate.

    Args:
        entries: List of metric entries for this phase (including the just-
            recorded one).
        progress: ``metrics_progress.{phase}`` dict (or None if absent).
        phase: Phase name.
        batch_index: 1-based index of the just-completed batch.
        total_batches: Total batch count for the phase.
        this_duration_ms: Duration of the just-completed batch (for the
            `done in X` display).

    Returns:
        Dict with keys: ``per_item_ms``, ``wall_clock_ms``, ``eta_ms``,
        ``elapsed_ms``, ``done``, ``total``, ``remaining``, ``line``.
    """
    done = max(int(batch_index), 0)
    total = max(int(total_batches), 0)
    remaining = max(total - done, 0) if total > 0 else 0

    durations = [
        e["duration_ms"] for e in entries
        if isinstance(e, dict) and e.get("duration_ms") is not None
    ]
    per_item_ms: Optional[int] = None
    if durations:
        per_item_ms = (sum(durations) // len(durations)) * remaining

    wall_clock_ms: Optional[int] = None
    elapsed_ms: Optional[int] = None
    if progress and progress.get("started_at"):
        try:
            started_at = datetime.fromisoformat(progress["started_at"])
            elapsed_ms = int((datetime.now() - started_at).total_seconds() * 1000)
            if done > 0:
                wall_clock_ms = (elapsed_ms // done) * remaining
        except (ValueError, TypeError):
            elapsed_ms = None
            wall_clock_ms = None

    if per_item_ms is not None and wall_clock_ms is not None:
        eta_ms: Optional[int] = min(per_item_ms, wall_clock_ms)
    elif per_item_ms is not None:
        eta_ms = per_item_ms
    elif wall_clock_ms is not None:
        eta_ms = wall_clock_ms
    else:
        eta_ms = None

    parts = [f"[ETA] {phase}: batch {done}/{total} done in {format_duration(this_duration_ms)}"]
    if per_item_ms is not None:
        parts.append(f"per-item ~{format_duration(per_item_ms)}")
    if wall_clock_ms is not None:
        parts.append(f"wall-clock ~{format_duration(wall_clock_ms)}")
    if remaining == 0:
        parts.append(_colorize("ALL BATCHES COMPLETE", "1;32", colorize))
    elif eta_ms is not None:
        parts.append(_colorize(f"~{format_duration(eta_ms)} remaining", "1;36", colorize))

    if done > total:
        # Defensive: caller bug — emit a warning but still return a usable line.
        parts.append(_colorize("WARNING: batch_index > total_batches", "1;33", colorize))

    line = " · ".join(parts)

    return {
        "per_item_ms": per_item_ms,
        "wall_clock_ms": wall_clock_ms,
        "eta_ms": eta_ms,
        "elapsed_ms": elapsed_ms,
        "done": done,
        "total": total,
        "remaining": remaining,
        "line": line,
    }


def record_metric(
    state_path: Path,
    phase: str,
    label: str,
    subagent_type: Optional[str] = None,
    token_count: Optional[int] = None,
    tool_uses: Optional[int] = None,
    duration_ms: Optional[int] = None,
    total_batches: Optional[int] = None,
    batch_index: Optional[int] = None,
    print_eta: bool = True,
) -> None:
    """Record a single metric entry into the state file.

    Appends to the ``metrics.{phase}`` array in the state JSON.
    Auto-generates a timestamp for the entry.

    When both ``total_batches`` and ``batch_index`` are provided and
    ``print_eta`` is True, prints an ``[ETA]`` line to stderr after writing
    the metric.

    Args:
        state_path: Path to the state JSON file (must exist).
        phase: Workflow phase name (e.g. "implement", "apply-suggestions").
        label: Human-readable label for this metric entry.
        subagent_type: Optional subagent type (e.g. "general-purpose").
        token_count: Optional token count consumed.
        tool_uses: Optional number of tool invocations.
        duration_ms: Optional wall-clock duration in milliseconds.
        total_batches: Optional total batch count (for ETA).
        batch_index: Optional 1-based batch index (for ETA).
        print_eta: When False, suppresses ETA stderr output.
    """
    state = _read_state(state_path)

    if "metrics" not in state:
        state["metrics"] = {}
    if phase not in state["metrics"]:
        state["metrics"][phase] = []

    entry = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
    }
    if subagent_type is not None:
        entry["subagent_type"] = subagent_type
    if token_count is not None:
        entry["token_count"] = token_count
    if tool_uses is not None:
        entry["tool_uses"] = tool_uses
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms

    state["metrics"][phase].append(entry)
    _write_state(state_path, state)

    if print_eta and total_batches is not None and batch_index is not None:
        entries = state["metrics"][phase]
        progress = state.get("metrics_progress", {}).get(phase)
        result = _compute_eta(
            entries=entries,
            progress=progress,
            phase=phase,
            batch_index=batch_index,
            total_batches=total_batches,
            this_duration_ms=duration_ms,
            colorize=_should_colorize_stderr(),
        )
        print(result["line"], file=sys.stderr)


def generate_phase_report(state_path: Path, phase: str) -> str:
    """Generate a markdown table report for a single phase.

    Returns an empty string if no metrics exist for the phase.

    Args:
        state_path: Path to the state JSON file (must exist).
        phase: Phase name to report on.

    Returns:
        Markdown string with a resource-usage table, or "".
    """
    state = _read_state(state_path)
    entries = state.get("metrics", {}).get(phase, [])
    if not entries:
        return ""

    lines = [
        "## Resource Usage",
        "",
        "| Task | Subagent Type | Tokens | Tool Uses | Duration |",
        "|------|---------------|--------|-----------|----------|",
    ]

    total_tokens: Optional[int] = None
    total_tool_uses: Optional[int] = None
    total_duration_ms: Optional[int] = None

    for entry in entries:
        label = _escape_pipe(entry.get("label", ""))
        sa_type = entry.get("subagent_type", "-")
        tokens = entry.get("token_count")
        tools = entry.get("tool_uses")
        dur = entry.get("duration_ms")

        lines.append(
            f"| {label} | {sa_type} | {format_number(tokens)} "
            f"| {format_number(tools)} | {format_duration(dur)} |"
        )

        if tokens is not None:
            total_tokens = (total_tokens or 0) + tokens
        if tools is not None:
            total_tool_uses = (total_tool_uses or 0) + tools
        if dur is not None:
            total_duration_ms = (total_duration_ms or 0) + dur

    # Bold total row
    t_tok = f"**{format_number(total_tokens)}**"
    t_tool = f"**{format_number(total_tool_uses)}**"
    t_dur = f"**{format_duration(total_duration_ms)}**"
    lines.append(f"| **Total** | | {t_tok} | {t_tool} | {t_dur} |")

    lines.append("")
    lines.append("*Duration is cumulative subagent time.*")

    return "\n".join(lines)


def generate_all_phases_report(state_path: Path) -> str:
    """Generate an aggregated markdown report across all phases.

    Phases are sorted alphabetically. Returns an empty string if no
    metrics exist in the state file.

    Args:
        state_path: Path to the state JSON file (must exist).

    Returns:
        Markdown string with a cross-phase summary table, or "".
    """
    state = _read_state(state_path)
    all_metrics = state.get("metrics", {})
    if not all_metrics:
        return ""

    # Filter out phases with empty entry lists
    phases = sorted(p for p, entries in all_metrics.items() if entries)
    if not phases:
        return ""

    lines = [
        "### Resource Usage",
        "",
        "| Phase | Subagent Calls | Tokens | Tool Uses | Duration |",
        "|-------|---------------|--------|-----------|----------|",
    ]

    grand_calls = 0
    grand_tokens: Optional[int] = None
    grand_tool_uses: Optional[int] = None
    grand_duration_ms: Optional[int] = None

    for phase in phases:
        entries = all_metrics[phase]
        count = len(entries)
        grand_calls += count

        phase_tokens: Optional[int] = None
        phase_tools: Optional[int] = None
        phase_dur: Optional[int] = None

        for entry in entries:
            t = entry.get("token_count")
            u = entry.get("tool_uses")
            d = entry.get("duration_ms")
            if t is not None:
                phase_tokens = (phase_tokens or 0) + t
            if u is not None:
                phase_tools = (phase_tools or 0) + u
            if d is not None:
                phase_dur = (phase_dur or 0) + d

        lines.append(
            f"| {_escape_pipe(phase)} | {count} "
            f"| {format_number(phase_tokens)} | {format_number(phase_tools)} "
            f"| {format_duration(phase_dur)} |"
        )

        if phase_tokens is not None:
            grand_tokens = (grand_tokens or 0) + phase_tokens
        if phase_tools is not None:
            grand_tool_uses = (grand_tool_uses or 0) + phase_tools
        if phase_dur is not None:
            grand_duration_ms = (grand_duration_ms or 0) + phase_dur

    t_calls = f"**{grand_calls}**"
    t_tok = f"**{format_number(grand_tokens)}**"
    t_tool = f"**{format_number(grand_tool_uses)}**"
    t_dur = f"**{format_duration(grand_duration_ms)}**"
    lines.append(f"| **Total** | {t_calls} | {t_tok} | {t_tool} | {t_dur} |")

    lines.append("")
    lines.append("*Aggregated from per-phase metrics. Duration is cumulative subagent time.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m utils.metrics",
        description="Record and report task resource-usage metrics.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- record --
    rec = subparsers.add_parser("record", help="Record a metric entry")
    rec.add_argument("--state-file", required=True, help="Path to state JSON file")
    rec.add_argument("--phase", required=True, help="Workflow phase name")
    rec.add_argument("--label", required=True, help="Human-readable label")
    rec.add_argument("--subagent-type", default=None, help="Subagent type")
    rec.add_argument("--tokens", type=int, default=None, help="Token count")
    rec.add_argument("--tool-uses", type=int, default=None, help="Tool invocation count")
    rec.add_argument("--duration-ms", type=int, default=None, help="Duration in ms")
    rec.add_argument("--total-batches", type=int, default=None,
                     help="Total batch count for ETA (use with --batch-index)")
    rec.add_argument("--batch-index", type=int, default=None,
                     help="1-based index of this batch (use with --total-batches)")
    rec.add_argument("--no-eta", action="store_true",
                     help="Suppress ETA stderr line even when --total-batches/--batch-index are set")

    # -- start --
    start = subparsers.add_parser("start", help="Mark phase start for ETA tracking")
    start.add_argument("--state-file", required=True, help="Path to state JSON file")
    start.add_argument("--phase", required=True, help="Workflow phase name")
    start.add_argument("--total-batches", type=int, required=True, help="Total batches in this phase")

    # -- finish --
    finish = subparsers.add_parser("finish", help="Mark phase finish, clear progress")
    finish.add_argument("--state-file", required=True, help="Path to state JSON file")
    finish.add_argument("--phase", required=True, help="Workflow phase name")

    # -- report --
    rep = subparsers.add_parser("report", help="Generate a metrics report")
    rep.add_argument("--state-file", required=True, help="Path to state JSON file")
    group = rep.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", default=None, help="Report for a single phase")
    group.add_argument(
        "--all-phases", action="store_true", default=False,
        help="Report aggregated across all phases",
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    state_file = Path(args.state_file)

    if not state_file.exists():
        print(f"Error: state file not found: {state_file}", file=sys.stderr)
        sys.exit(1)

    if args.command == "record":
        record_metric(
            state_path=state_file,
            phase=args.phase,
            label=args.label,
            subagent_type=args.subagent_type,
            token_count=args.tokens,
            tool_uses=args.tool_uses,
            duration_ms=args.duration_ms,
            total_batches=args.total_batches,
            batch_index=args.batch_index,
            print_eta=not args.no_eta,
        )

    elif args.command == "start":
        start_phase(
            state_path=state_file,
            phase=args.phase,
            total_batches=args.total_batches,
        )

    elif args.command == "finish":
        finish_phase(state_path=state_file, phase=args.phase)

    elif args.command == "report":
        if args.all_phases:
            output = generate_all_phases_report(state_file)
        else:
            output = generate_phase_report(state_file, args.phase)
        if output:
            print(output)
