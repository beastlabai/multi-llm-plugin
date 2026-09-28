"""Metrics utility for recording and reporting Task tool resource usage.

Records per-subagent metrics (tokens, tool uses, duration) into state JSON
and generates markdown reports for individual phases or cross-phase summaries.
"""

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .stream_bootstrap import bootstrap_streams
except ImportError:
    # Direct script invocation (`python utils/metrics.py`): sys.path[0]
    # is the utils/ directory, so import the sibling module directly.
    from stream_bootstrap import bootstrap_streams


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


def describe_usage(usage: Optional[Dict[str, Any]]) -> str:
    """One clause naming what a run cost, or ``""``.

    ``$25.05 est., 137,348 output tokens`` — the dollar figure first because
    it is the number a human acts on, the output count second because it is
    the one that says how far the run got. Input and cache counts are not
    rendered: they are on ``details["usage"]`` for anything that wants them,
    and a seven-figure cache-read count beside a failure reads as noise. Cost
    is rendered to the cent and dropped when it rounds to nothing, so a
    duration-only or free run says nothing rather than "$0.00".

    **Why "est." and not a bare figure.** ``cost_usd`` is whatever the CLI
    computed from its own price catalog: an estimate the runner relays, never
    a vendor invoice and never a billed amount, and a report that printed it
    unqualified would be read as one. One renderer serves every surface that
    names the number (the failed-slot error line, the ``**Models:**`` line,
    the HTML model badge and the report header), so the qualifier is attached
    once, here, rather than trusted to four call sites to remember.

    **Why this module.** It formats a usage metric, beside
    :func:`format_number` and :func:`format_duration`, and — unlike
    ``llm_client``, where it was first written for the failed-slot error line
    — it imports nothing but the standard library. That is what lets
    ``html_report_generator`` (which must keep working as
    ``python utils/html_report_generator.py``, where a relative-importing
    module cannot be loaded at all) render the same clause the fan-out logs,
    instead of a second copy of these rules drifting in a template.
    ``llm_client.describe_usage`` re-exports it for the callers that have
    always reached it there.
    """
    if not isinstance(usage, dict) or not usage:
        return ""
    parts = []
    cost = usage.get("cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0.005:
        parts.append(f"${cost:.2f} est.")
    output = usage.get("output_tokens")
    if isinstance(output, (int, float)) and not isinstance(output, bool) and output > 0:
        parts.append(f"{int(output):,} output tokens")
    return ", ".join(parts)


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


def start_phase_in_state(state: dict, phase: str, total_batches: int) -> None:
    """``start_phase``'s mutation, applied to an in-memory state dict.

    Split out so a holder of a live ``StateManager`` dict can record progress
    without becoming a *second* writer of ``state.json``: the two race, and the
    ``StateManager``'s own ``save()`` — which dumps its dict wholesale and never
    merges — always wins. Measured 2026-09-16: three independent runs finished
    with `metrics: null` after every batch had recorded one (F4).

    Each start also stamps a fresh ``run_id``, which every entry recorded under
    it carries. It is what tells a second record of the *same* event (collapsed
    by ``record_metric_in_state``) from a rerun's or a ``--resume``'s real
    measurement of the same label: ``metrics`` outlives runs in ``state.json``,
    and a rerun or resume starts the phase again, so it gets a new id.
    """
    progress = state.get("metrics_progress")
    if not isinstance(progress, dict):
        progress = {}
        state["metrics_progress"] = progress
    progress[phase] = {
        "started_at": datetime.now().isoformat(),
        "total_batches": int(total_batches),
        "run_id": uuid.uuid4().hex,
    }


def start_phase(state_path: Path, phase: str, total_batches: int) -> None:
    """Record the start time and total batch count for a phase.

    Writes ``metrics_progress.{phase}`` = {"started_at": ISO timestamp,
    "total_batches": int, "run_id": uuid4 hex}. Used by ``record_metric`` to
    compute wall-clock ETA and to recognise this run's own entries.
    Idempotent: a second call for the same phase replaces the prior entry
    (necessary so ``--resume`` runs reset the progress baseline, and start a new
    ``run_id``).
    """
    state = _read_state(state_path)
    start_phase_in_state(state, phase, total_batches)
    _write_state(state_path, state)


def finish_phase_in_state(
    state: dict, phase: str
) -> Optional[Tuple[Optional[int], str]]:
    """``finish_phase``'s mutation, applied to an in-memory state dict.

    Returns None — leaving ``state`` untouched — when the phase was never
    started; otherwise pops the progress entry and returns
    ``(elapsed_ms, line)``, where ``line`` is the closing ``[ETA] … phase
    complete`` line. It is **returned, not printed**, mirroring
    ``record_metric_in_state``: the caller decides where it goes — stderr for
    the CLI, the phase log for the scheduler, which is where every other
    ``[ETA]`` line of that run already went (F7). See ``start_phase_in_state``
    for why the in-memory form exists.
    """
    progress = state.get("metrics_progress") or {}
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

    color_on = _should_colorize_stderr()
    complete_segment = _colorize(
        f"phase complete in {format_duration(elapsed_ms)}", "1;32", color_on,
    )
    return elapsed_ms, f"[ETA] {phase}: {complete_segment}"


def finish_phase(state_path: Path, phase: str) -> Optional[int]:
    """Remove ``metrics_progress.{phase}`` and return total elapsed ms.

    Returns None if the phase was never started (graceful no-op).
    """
    state = _read_state(state_path)
    if phase not in (state.get("metrics_progress") or {}):
        # Never started: nothing to mutate, so nothing to write back.
        return None
    finished = finish_phase_in_state(state, phase)
    if finished is None:  # pragma: no cover - guarded above
        return None
    elapsed_ms, line = finished
    # The CLI leg is the one that owns stderr; the in-memory twin only returns
    # the line (F7).
    print(line, file=sys.stderr)
    _write_state(state_path, state)
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
    """Compute ETA from two independent remaining-time estimates.

    ``per_item_ms`` is ``mean(batch duration) * batches_remaining`` and
    ``wall_clock_ms`` is ``(elapsed / done) * batches_remaining`` — both are
    whole-phase remaining-time estimates from different signals, not a
    per-item duration, however the first one used to be labelled: printing
    both used to read as three quantities (a per-item time, a wall-clock
    time, and a remaining time barely different from either), when a batch
    ≠ item phase made "per-item" actively misleading (F13, 2026-09-16). The
    rendered line now reports only the number a reader acts on — ``eta_ms``,
    the min of the two — and keeps the intermediates in the returned dict
    for callers/tests that want them.

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


def normalize_recorded_executor(value: str) -> str:
    """Map a legacy ``subagent_type`` value onto the recorded executor value.

    ``general-purpose`` is Claude Code vocabulary that exists on no other host
    and carries no information, so it records as ``agent``; ``human`` keeps its
    meaning (no invoker ever ran it). Anything else — notably the resolved
    ``provider:model`` runner spec the Python executor records — passes through
    verbatim, which is what makes the two invokers read differently in
    ``--status`` output instead of both printing ``general-purpose``.
    """
    if value == "general-purpose":
        return "agent"
    return value


#: The measured fields of a metric entry. A record carrying none of them, or
#: only zeros in them, is *empty*: nothing was measured (the host's subagent
#: tool reports no usage) or a zero was fabricated for a value nobody had.
_MEASURED_FIELDS = ("token_count", "tool_uses", "duration_ms")

#: Every field two records of one event are compared on.
_PAYLOAD_FIELDS = ("subagent_type",) + _MEASURED_FIELDS

_EMPTY_NONE = "no numbers"
_EMPTY_ZERO = "only zeros"


def _current_run_id(state: dict, phase: str) -> Optional[str]:
    """The ``run_id`` ``start_phase_in_state`` stamped for ``phase``, or None.

    None when the phase was never started (or finished), or was started before
    run ids existed: such a record appends unchecked, exactly as before.
    """
    progress = state.get("metrics_progress")
    if not isinstance(progress, dict):
        return None
    entry = progress.get(phase)
    if not isinstance(entry, dict):
        return None
    run_id = entry.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _emptiness(entry: dict) -> Optional[str]:
    """Why ``entry`` measured nothing, or None when it carries a real number."""
    measured = [entry[k] for k in _MEASURED_FIELDS if entry.get(k) is not None]
    if not measured:
        return _EMPTY_NONE
    if all(value == 0 for value in measured):
        return _EMPTY_ZERO
    return None


def _payload(entry: dict) -> Dict[str, Any]:
    return {k: entry[k] for k in _PAYLOAD_FIELDS if entry.get(k) is not None}


def _describe_event(phase: str, label: str, batch_index: Optional[int]) -> str:
    where = f" (batch {batch_index})" if batch_index is not None else ""
    return f'{phase}: "{label}"{where}'


def _replacement(existing: dict, new: dict, keep_measured: bool) -> dict:
    """``new``, keeping any field of ``existing`` it does not carry itself.

    The executor role always survives. A measured field survives only from a
    non-empty entry (``keep_measured``): carrying an empty entry's zeros into
    the replacement would re-fabricate exactly what the replacement removes.
    """
    merged = dict(new)
    for key in _PAYLOAD_FIELDS:
        if key in merged or existing.get(key) is None:
            continue
        if key == "subagent_type" or keep_measured:
            merged[key] = existing[key]
    return merged


def _collapse_duplicate(
    entries: list, index: int, new: dict, event: str
) -> bool:
    """Settle a second record of the event ``entries[index]`` already holds.

    Returns True when ``entries[index]`` now holds the new record's numbers
    (so the caller's ``[ETA]`` line reflects them), False when the new record
    was dropped. Either way nothing is appended: two entries for one batch
    would count it twice in every per-phase total and average.
    """
    existing = entries[index]
    old_empty = _emptiness(existing)
    new_empty = _emptiness(new)

    if new_empty is not None:
        # An empty record never displaces a measured one. Of two empty ones,
        # the one without numbers is kept over the all-zero one (whose zeros
        # were made up and would average in), so the incident's pair
        # (one all zeros, one with none, 45 ms apart) collapses to the same
        # entry whichever write lands first.
        if old_empty == _EMPTY_ZERO and new_empty == _EMPTY_NONE:
            entries[index] = _replacement(existing, new, keep_measured=False)
            print(
                f"[METRICS_DUPLICATE] {event} recorded twice in this run; kept "
                f"the record with no numbers over the one with only zeros",
                file=sys.stderr,
            )
            return True
        print(
            f"[METRICS_DUPLICATE] {event} already recorded in this run; skipped "
            f"the second record (it carried {new_empty})",
            file=sys.stderr,
        )
        return False

    if old_empty is not None:
        # An empty row that landed first must not cost the real numbers.
        entries[index] = _replacement(existing, new, keep_measured=False)
        print(
            f"[METRICS_DUPLICATE] {event} recorded twice in this run; the second "
            f"record replaced the first, which carried {old_empty}",
            file=sys.stderr,
        )
        return True

    old_payload = _payload(existing)
    new_payload = _payload(new)
    differing = sorted(
        k for k in old_payload.keys() & new_payload.keys()
        if old_payload[k] != new_payload[k]
    )
    if differing:
        print(
            f"[METRICS_CONFLICT] {event} recorded twice in this run with "
            f"different {', '.join(differing)}: kept {json.dumps(old_payload, sort_keys=True)}, "
            f"dropped {json.dumps(new_payload, sort_keys=True)}",
            file=sys.stderr,
        )
        return False
    if new_payload.keys() <= old_payload.keys():
        print(
            f"[METRICS_DUPLICATE] {event} already recorded in this run; skipped "
            f"the second record (it repeats the recorded values)",
            file=sys.stderr,
        )
        return False
    entries[index] = _replacement(existing, new, keep_measured=True)
    added = sorted(new_payload.keys() - old_payload.keys())
    print(
        f"[METRICS_DUPLICATE] {event} recorded twice in this run; the second "
        f"record agrees and adds {', '.join(added)}, so it replaced the first",
        file=sys.stderr,
    )
    return True


def record_metric_in_state(
    state: dict,
    phase: str,
    label: str,
    subagent_type: Optional[str] = None,
    token_count: Optional[int] = None,
    tool_uses: Optional[int] = None,
    duration_ms: Optional[int] = None,
    total_batches: Optional[int] = None,
    batch_index: Optional[int] = None,
) -> Optional[str]:
    """``record_metric``'s mutation, applied to an in-memory state dict.

    Returns the rendered ``[ETA]`` line when ``total_batches`` and
    ``batch_index`` are both given, so the caller decides *where* it goes —
    stderr for the CLI, the phase log for the scheduler — and nothing is
    printed for a caller that wants none. See ``start_phase_in_state`` for why
    the in-memory form exists.

    **A second record of the same event collapses; nothing else does.** An
    event is ``(phase, run_id, label, batch_index)``, where ``run_id`` is the
    one ``start_phase_in_state`` stamped for this run. On the native leg the
    orchestrating agent works out each record across separate shell calls, and
    one audited run recorded ``Batch 4 (step_4)`` twice, 45 ms apart, one copy
    all zeros and the other with no numbers. The second record of an event:

    - replaces the first when the first measured nothing and it did;
    - is skipped when it measured nothing (of two empty records, the one with
      no numbers is kept over the all-zero one), or repeats the first, or is a
      subset agreeing on every field it carries;
    - replaces the first when it agrees on every shared field and adds more;
    - is dropped with ``[METRICS_CONFLICT]`` when a shared field differs,
      because appending would count one batch twice in the phase averages.

    Every one of those prints a line on stderr (``[METRICS_DUPLICATE]`` or
    ``[METRICS_CONFLICT]``): it reports a caller's mistake, the way
    ``prior_changes append`` reports a duplicate entry. A skipped or conflicting
    record returns no ``[ETA]`` line, since it recorded no progress.

    A record with no started phase has no ``run_id`` and appends unchecked, as
    before, and an entry without one (every entry recorded before run ids) is
    never matched: keying on ``(phase, label)`` alone would drop a rerun's real
    measurements, since metrics outlive runs in ``state.json``.

    **``batch_index`` is checked against this run's own progress** (only on a
    record that appends): the expected index is the highest one this run has
    recorded, plus one, and anything else prints
    ``[METRICS_INDEX_MISMATCH]``. A warning, never a rejection — a skipped
    batch is legitimate — and the high-water mark means one gap warns once.
    ``len(entries) + 1`` would warn on every record after a rerun or a resume
    (``entries`` spans every run) and after any skipped record. A run's first
    record is not checked: a resumed scheduler run's first index legitimately
    starts past 1, because it counts the batches the resume skipped.
    """
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        # Tolerate the `metrics: null` some older state files carry: `"metrics"
        # not in state` was True for a dict and False for a null, so a null
        # would have raised here rather than being replaced.
        metrics = {}
        state["metrics"] = metrics
    entries = metrics.get(phase)
    if not isinstance(entries, list):
        entries = []
        metrics[phase] = entries

    entry = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
    }
    if subagent_type is not None:
        # D2h: the flag name and the JSON key stay put (this is a CLI surface
        # with recorded history in existing metrics.json files); what goes *in*
        # becomes the host-neutral executor role. `general-purpose` carried no
        # information — it is pure Claude Code vocabulary — so it is normalized
        # to `agent`. A resolved `provider:model` runner spec (the executor path)
        # or any other value passes through verbatim.
        entry["subagent_type"] = normalize_recorded_executor(subagent_type)
    if token_count is not None:
        entry["token_count"] = token_count
    if tool_uses is not None:
        entry["tool_uses"] = tool_uses
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms
    if batch_index is not None:
        entry["batch_index"] = batch_index

    run_id = _current_run_id(state, phase)
    if run_id is not None:
        entry["run_id"] = run_id
        event = _describe_event(phase, label, batch_index)
        this_run = [
            (i, e) for i, e in enumerate(entries)
            if isinstance(e, dict) and e.get("run_id") == run_id
        ]
        for i, recorded in this_run:
            if recorded.get("label") == label and recorded.get("batch_index") == batch_index:
                if not _collapse_duplicate(entries, i, entry, event):
                    return None
                break
        else:
            marks = [e["batch_index"] for _, e in this_run if _is_index(e.get("batch_index"))]
            if _is_index(batch_index) and marks:
                expected = max(marks) + 1
                if batch_index != expected:
                    print(
                        f"[METRICS_INDEX_MISMATCH] {event} was recorded as batch "
                        f"{batch_index}; this run's next index is {expected} "
                        f"(highest recorded so far: {max(marks)}). Recorded as given.",
                        file=sys.stderr,
                    )
            entries.append(entry)
    else:
        entries.append(entry)

    if total_batches is None or batch_index is None:
        return None
    progress = (state.get("metrics_progress") or {}).get(phase)
    result = _compute_eta(
        entries=entries,
        progress=progress,
        phase=phase,
        batch_index=batch_index,
        total_batches=total_batches,
        this_duration_ms=duration_ms,
        colorize=_should_colorize_stderr(),
    )
    return result["line"]


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

    Appends to the ``metrics.{phase}`` array in the state JSON, except that a
    second record of an event this run already recorded collapses into the
    first (see ``record_metric_in_state``). Auto-generates a timestamp for the
    entry.

    When both ``total_batches`` and ``batch_index`` are provided and
    ``print_eta`` is True, prints an ``[ETA]`` line to stderr after writing
    the metric.

    Args:
        state_path: Path to the state JSON file (must exist).
        phase: Workflow phase name (e.g. "implement", "apply-suggestions").
        label: Human-readable label for this metric entry.
        subagent_type: Optional executor value ("agent", "human", or the
            resolved provider:model runner spec). The flag and JSON key keep
            their historical name; see normalize_recorded_executor.
        token_count: Optional token count consumed.
        tool_uses: Optional number of tool invocations.
        duration_ms: Optional wall-clock duration in milliseconds.
        total_batches: Optional total batch count (for ETA).
        batch_index: Optional 1-based batch index (for ETA).
        print_eta: When False, suppresses ETA stderr output.
    """
    state = _read_state(state_path)
    eta_line = record_metric_in_state(
        state,
        phase,
        label,
        subagent_type=subagent_type,
        token_count=token_count,
        tool_uses=tool_uses,
        duration_ms=duration_ms,
        total_batches=total_batches,
        batch_index=batch_index,
    )
    _write_state(state_path, state)

    if print_eta and eta_line:
        print(eta_line, file=sys.stderr)


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
        "| Task | Executor | Tokens | Tool Uses | Duration |",
        "|------|----------|--------|-----------|----------|",
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


def main(argv=None) -> int:
    bootstrap_streams()
    parser = _build_parser()
    args = parser.parse_args(argv)

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

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
