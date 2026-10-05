#!/usr/bin/env python3
"""The Phase-3 instruction-refactor regression guard: capture, normalize, diff.

The instruction core is ~5k lines of markdown a driving agent reads. Nothing in
Python parses it, so no unit test can prove a rewrite of it left the *workflow*
alone. What can be proved is the complement: that the orchestrator-produced
artifacts of one named, fully mocked workflow run are byte-identical before and
after the rewrite. That is this module.

Three pieces, deliberately in one file:

* :func:`capture` drives the guard fixture — ``tests/fixtures/e2e/plans/auth-feature.md``
  through ``tests/fixtures/e2e/scenarios/full_workflow.yaml``, against the mock
  provider harness — and returns the bundle.
* :func:`normalize_run` erases everything that legitimately varies between two
  runs of identical instructions (timestamps, durations, token counts, run ids,
  pids, absolute paths, dict ordering). Those fields are asserted for *presence
  and type*, never for value, which is why the guard has no timing or token
  threshold to tune.
* :func:`diff_bundles` reports every remaining difference as a normalized
  JSON-pointer path, which is the vocabulary ``tests/golden/expected_changes.yaml``
  is written in.

**Why one self-contained file.** The baseline is captured on the
``phase3-pre-refactor`` tag, where no guard test exists. The capture therefore
runs by copying *this file* into a worktree checked out at that tag and executing
it there against that tree's own harness — so both sides of the comparison are
produced by identical capture and normalization code, and a baseline captured by
the same commit it polices (which would prove nothing) is impossible by
construction.

Usage (capture)::

    python tests/golden/instruction_refactor_bundle.py --skill-dir <skill root> \
        --out <bundle.json> [--commit <sha>]

The phase sequence is the one that exists on *both* sides of the refactor:
review-plan → apply-suggestions → implement → review-code → apply-code-fixes,
each driven exactly as the pre-refactor instruction core drove it. ``generate-tasks``
and ``full`` are deliberately outside the bundle: their orchestrators did not
exist at the baseline commit (they arrived with the Phase-3 sequencing commit,
not with the instruction refactor), so including them would compare two different
implementations and call the difference a regression.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The named guard fixture. Changing either is a deliberate re-capture with its
#: own commit, never an unexplained diff in the guard.
GUARD_PLAN = "auth-feature"
GUARD_SCENARIO = "full_workflow"

#: Bundle format version. Bumped only when the *shape* of the bundle changes, so
#: a baseline that predates a shape change fails loudly instead of diffing
#: nonsensically.
BUNDLE_VERSION = 1

#: Keys whose value is time-, machine- or identity-dependent. Replaced by a type
#: token so presence and type are still asserted and the value never is.
_VOLATILE_KEYS = frozenset({
    "timestamp", "timestamps", "created_at", "updated_at", "started_at",
    "finished_at", "completed_at", "capture_date", "date", "generated",
    "generated_at", "last_updated", "heartbeat", "heartbeat_at",
    "duration", "duration_ms", "duration_s", "duration_seconds", "elapsed",
    "elapsed_ms", "eta", "eta_s", "wall_clock_ms", "per_item_ms",
    "tokens", "token_count", "total_tokens", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_creation_tokens", "usage", "cost", "cost_usd",
    "run_id", "pid", "process_start_time", "ppid", "session_id",
    "plan_hash", "head_at_start", "head", "commit", "sha",
    # The checkout's own git state: the baseline is captured in a detached
    # worktree at the tag, so a branch name is a property of the capture, never of
    # the run.
    "branch", "branch_name", "git_branch",
    "model_version", "python_version", "hostname", "user",
})

#: Keys whose value is a filesystem path (absolute in the raw artifact). Kept,
#: but rewritten relative to the run root.
_PATH_KEY_RE = re.compile(
    r"(^|_)(path|paths|file|files|dir|dirs|directory|root|log|logs)$"
)

_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _type_token(value: Any) -> str:
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, int):
        return "<int>"
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, str):
        return "<str>"
    if value is None:
        return "<null>"
    if isinstance(value, list):
        return "<list>"
    if isinstance(value, dict):
        return "<dict>"
    return "<other>"


def _relativize(text: str, root: str) -> str:
    """Rewrite absolute paths under *root* (and the skill dir) to relative form."""
    out = text
    if root:
        for candidate in (root, root.replace("\\", "/"), root.replace("/", "\\")):
            if candidate:
                out = out.replace(candidate, "<run-root>")
    out = _UUID_RE.sub("<uuid>", out)
    out = _ISO_RE.sub("<timestamp>", out)
    return out


def normalize_run(value: Any, root: str = "", *, _key: str = "") -> Any:
    """Erase everything that legitimately varies between two identical runs.

    Timestamps, durations, token/usage numbers, run ids, pids, absolute paths and
    dict ordering. A volatile scalar becomes its *type token*, so the guard still
    fails when a field disappears or changes type — it simply never compares a
    clock reading to another clock reading.

    One shared helper rather than per-test ad hoc rules, so the pre- and
    post-refactor bundles cannot be normalized by two subtly different sets of
    rules.
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key in sorted(value):
            out[key] = normalize_run(value[key], root, _key=key)
        return out
    if isinstance(value, list):
        return [normalize_run(item, root, _key=_key) for item in value]
    if _key in _VOLATILE_KEYS:
        return _type_token(value)
    if isinstance(value, str):
        # Content hashes (group/suggestion ids) are *derived* from the fixture
        # text, so they are stable and deliberately left comparable — a changed
        # grouping is exactly what this guard is for. Only the genuinely
        # nondeterministic identifiers are folded, by key or by shape.
        return _relativize(value, root)
    return value


def _structure(value: Any) -> Any:
    """Keys and value *types* only — what "the structure of metrics.json" means."""
    if isinstance(value, dict):
        return {key: _structure(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        kinds = sorted({_type_token(item) for item in value})
        return {"<list>": len(value), "<item-types>": kinds,
                "<items>": [_structure(item) for item in value]}
    return _type_token(value)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

#: Artifacts compared in full (normalized). Orchestrator-produced JSON only —
#: never raw model text, and never a rendered report whose prose is the model's.
_COMPARED_JSON = (
    "grouped.json",
    "issues.json",
    "validation.json",
    "orchestrator_output.json",
    "validation_tasks.json",
    "revalidation_tasks.json",
    "consolidation_tasks.json",
    "human_decision_tasks.json",
    "consolidated.json",
    "subagent_batch_report.json",
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - reported in-bundle
        return {"<unreadable>": str(exc)}


def _relative_paths(base: Path) -> List[str]:
    """Every path under *base*, relative and POSIX-spelled, sorted.

    Volatile filenames are folded to a stable token: a log file named after a run
    id, and the per-attempt diagnostics under ``failed/``, would otherwise make
    the created-path set differ on every run.
    """
    if not base.exists():
        return []
    out = set()
    for path in base.rglob("*"):
        rel = path.relative_to(base).as_posix()
        rel = _UUID_RE.sub("<uuid>", rel)
        rel = re.sub(r"orchestrator-run\.[^.]+\.log$", "orchestrator-run.<run>.log", rel)
        out.add(rel + ("/" if path.is_dir() else ""))
    return sorted(out)


_MARKER_RE = re.compile(r"\[[A-Z][A-Z0-9_]{3,}[^\]\n]*\]")


def _markers(text: str) -> List[str]:
    """The orchestrator-emitted ``[MARKER]`` sequence, in order.

    Markers are orchestrator output, not model text, and they are the one part of
    stdout the pre-refactor instruction core branched on — so a refactor that
    changed which markers a phase emits has changed the contract even when every
    artifact matched.
    """
    return [
        _UUID_RE.sub("<uuid>", match.group(0))
        for match in _MARKER_RE.finditer(text)
    ]


def _phase_bundle(phase_dir: Path, root: str) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {"exists": phase_dir.exists()}
    if not phase_dir.exists():
        return bundle
    bundle["created_paths"] = _relative_paths(phase_dir)
    artifacts: Dict[str, Any] = {}
    for name in _COMPARED_JSON:
        candidate = phase_dir / name
        if candidate.exists():
            artifacts[name] = normalize_run(_load_json(candidate), root)
    bundle["artifacts"] = artifacts
    status = phase_dir / ".status.json"
    if status.exists():
        raw = _load_json(status)
        bundle["status_steps"] = _status_steps(raw)
        bundle["status_shape"] = _structure(raw)
    return bundle


def _status_steps(raw: Any) -> List[str]:
    """The ``.status.json`` phase/step sequence, in order, values only."""
    if not isinstance(raw, dict):
        return []
    steps: List[str] = []
    for key in ("phase", "state"):
        if key in raw:
            steps.append(f"{key}={raw[key]}")
    run = raw.get("run")
    if isinstance(run, dict):
        for key in ("state", "exit_code"):
            if key in run:
                steps.append(f"run.{key}={run[key]}")
        for entry in run.get("resolutions") or []:
            if isinstance(entry, dict):
                steps.append(f"run.resolution={entry.get('site')}")
    progress = raw.get("progress")
    if isinstance(progress, dict):
        steps.extend(f"progress.{key}" for key in sorted(progress))
    return steps


def _derived(grouped: Any, validation: Any, state: Any) -> Dict[str, Any]:
    """The derived scalars the plan names, computed from the artifacts."""
    derived: Dict[str, Any] = {}
    groups = []
    if isinstance(grouped, dict):
        groups = grouped.get("groups") or grouped.get("suggestion_groups") or []
    elif isinstance(grouped, list):
        groups = grouped
    derived["group_count"] = len(groups)
    suggestions = 0
    histogram: Dict[str, int] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        items = group.get("suggestions") or group.get("issues") or []
        suggestions += len(items)
        for item in items:
            importance = str(
                (item.get("importance") if isinstance(item, dict) else None)
                or group.get("importance")
                or "UNKNOWN"
            )
            histogram[importance] = histogram.get(importance, 0) + 1
    derived["suggestion_count"] = suggestions
    derived["importance_histogram"] = dict(sorted(histogram.items()))

    partition: Dict[str, int] = {}
    records: Sequence[Any] = ()
    if isinstance(validation, dict):
        records = validation.get("groups") or validation.get("results") or ()
    elif isinstance(validation, list):
        records = validation
    for record in records:
        if isinstance(record, dict):
            status = str(record.get("status") or "unknown")
            partition[status] = partition.get(status, 0) + 1
    derived["validation_partition"] = dict(sorted(partition.items()))

    if isinstance(state, dict):
        derived["phases_completed"] = sorted(state.get("phases_completed") or {})
        derived["tracked_file_count"] = len(state.get("tracked_files") or [])
    return derived


#: The dispatcher's terminal stdout line. Parsed out of every phase so the
#: machine-readable contract a host reads is part of the compared surface.
_RESULT_MARKER = "MULTI_LLM_RESULT:"


def _result_line(text: str, root: str) -> Any:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(_RESULT_MARKER):
            try:
                payload = json.loads(stripped[len(_RESULT_MARKER):].strip())
            except ValueError:  # pragma: no cover - reported in-bundle
                return {"<unparseable>": stripped[:200]}
            return normalize_run(payload, root)
    return None


def _seed_completed(state: Dict[str, Any], phase: str) -> None:
    """Stand in for the host's closing step: a fixed completion stamp.

    The write phases' native emit records the phase *started* and leaves
    completion to the host's closing ``--mark-completed`` call, which pops
    that started record in the same write (``mark_phase_completed_in_state``).
    A seed that stamped the completion alone left the emit's fresh start
    beside a 2025 completion, which reads — correctly — as a re-run in
    progress (fix round 6, 2026-10-05), so the seed pops the start too.
    """
    state.setdefault("phases_completed", {})[phase] = "2025-01-01T00:00:00"
    started = state.get("phases_started")
    if isinstance(started, dict):
        started.pop(phase, None)


def capture(skill_dir: Path, workdir: Path) -> Dict[str, Any]:
    """Run the guard fixture and return the normalized bundle.

    Every provider call is served by the mock harness from the pinned scenario, so
    two runs of identical instructions must produce a byte-identical bundle: the
    acceptable diff threshold is zero, and any diff is a change to explain.
    """
    skill_dir = Path(skill_dir)
    sys.path.insert(0, str(skill_dir / "tests"))
    sys.path.insert(0, str(skill_dir))
    from harness import FixtureManager, SkillRunner  # type: ignore

    fixtures = FixtureManager(workdir)
    runner = SkillRunner(workdir, skill_dir=skill_dir)
    scenario = fixtures.load_scenario(GUARD_SCENARIO)
    runner.set_scenario(scenario)
    runner.extra_env["MOCK_LLM_SCENARIO"] = str(scenario)

    plan = fixtures.load_plan(GUARD_PLAN)
    root = str(workdir)
    phases: Dict[str, Any] = {}
    exits: Dict[str, int] = {}
    markers: Dict[str, List[str]] = {}
    results: Dict[str, Any] = {}

    import subprocess

    def _dispatch(mode: str, *args: str) -> None:
        """Run one phase the way a shim does: through ``multi_llm.py <mode>``.

        Driving the dispatcher rather than the orchestrator script is what makes
        the guard cover the surface a host actually invokes — the mode word, the
        positional normalization, the cross-cutting flags and the terminal
        ``MULTI_LLM_RESULT`` line — instead of only the orchestrator underneath it.
        """
        command = [
            sys.executable, str(skill_dir / "multi_llm.py"), mode,
            str(plan.plan_path), *args,
        ]
        proc = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=180, env=runner._build_env(),
            cwd=str(skill_dir),
        )
        exits[mode] = proc.returncode
        markers[mode] = _markers(proc.stdout) + _markers(proc.stderr)
        results[mode] = _result_line(proc.stdout, root)

    # -- review-plan ----------------------------------------------------
    _dispatch("review-plan", "--models", "cursor-agent", "--skip-validation")
    review_dir = plan.output_dir / "review-plan"

    # -- apply-suggestions ----------------------------------------------
    # The apply orchestrators emit batch JSON and never edit the plan themselves,
    # so this runs for real (no --dry-run) and the plan file is asserted unchanged
    # at the end.
    validation = fixtures.load_response("validation", "all_valid", validate=False)
    (review_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    _dispatch("apply-suggestions", "--approve-all", "--yes",
              "--output-format", "json", "--no-confirm")

    # -- implement ------------------------------------------------------
    state_path = plan.output_dir / "state.json"
    state = _load_json(state_path) if state_path.exists() else {}
    if isinstance(state, dict):
        _seed_completed(state, "apply-suggestions")
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _dispatch("implement", "--dry-run")

    # -- review-code ----------------------------------------------------
    state = _load_json(state_path) if state_path.exists() else {}
    if isinstance(state, dict):
        _seed_completed(state, "implement")
        state["head_at_start"] = "0" * 40
        state["tracked_files"] = [
            {"path": "src/services/auth_service.py", "task_id": "T001",
             "action": "created"},
            {"path": "src/api/auth_routes.py", "task_id": "T002",
             "action": "created"},
        ]
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _dispatch("review-code", "--models", "cursor-agent", "--skip-validation")

    # -- apply-code-fixes -----------------------------------------------
    code_review_dir = plan.output_dir / "code-review"
    if code_review_dir.exists():
        (code_review_dir / "validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
    _dispatch("apply-code-fixes", "--approve-all", "--yes",
              "--output-format", "json", "--no-confirm")

    # Phase-dir names, not mode words: `apply-code-fixes` writes under
    # `apply-fixes/`, and `generate-tasks`/`review-tasks` are listed so their
    # *absence* is pinned too (a refactor that started creating them silently
    # would be a diff, not a shrug).
    for phase in ("review-plan", "apply-suggestions", "generate-tasks",
                  "review-tasks", "apply-task-suggestions", "implement",
                  "code-review", "apply-fixes"):
        phases[phase] = _phase_bundle(plan.output_dir / phase, root)

    state = _load_json(state_path) if state_path.exists() else {}
    grouped = _load_json(review_dir / "grouped.json") if (
        review_dir / "grouped.json").exists() else {}
    validation_out = _load_json(review_dir / "validation.json") if (
        review_dir / "validation.json").exists() else {}

    return {
        "bundle_version": BUNDLE_VERSION,
        "fixture": {"plan": GUARD_PLAN, "scenario": GUARD_SCENARIO},
        "exit_codes": exits,
        "markers": markers,
        "result_lines": results,
        "phases": phases,
        "state": normalize_run(state, root),
        "state_shape": _structure(state),
        "metrics_shape": _structure(
            state.get("metrics") if isinstance(state, dict) else None
        ),
        "derived": _derived(grouped, validation_out, state),
        "plan_unchanged": plan.plan_path.read_text(encoding="utf-8") == plan.content,
        "output_root_paths": _relative_paths(plan.output_dir),
    }


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            token = str(key).replace("~", "~0").replace("/", "~1")
            out.update(_flatten(value[key], f"{prefix}/{token}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}/{index}"))
    else:
        out[prefix or "/"] = value
    return out


def diff_bundles(baseline: Any, fresh: Any) -> List[Tuple[str, Any, Any]]:
    """Every difference as ``(json-pointer, baseline, fresh)``, pointer-sorted."""
    left, right = _flatten(baseline), _flatten(fresh)
    diffs: List[Tuple[str, Any, Any]] = []
    for pointer in sorted(set(left) | set(right)):
        before = left.get(pointer, "<absent>")
        after = right.get(pointer, "<absent>")
        if before != after:
            diffs.append((pointer, before, after))
    return diffs


# ---------------------------------------------------------------------------
# CLI (what runs inside the worktree at the baseline tag)
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--commit", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--workdir", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    import tempfile

    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="guard-capture-")
    )
    workdir.mkdir(parents=True, exist_ok=True)
    bundle = capture(Path(args.skill_dir), workdir)
    bundle["manifest"] = {
        "commit": args.commit,
        "label": args.label,
        "fixture_plan": f"tests/fixtures/e2e/plans/{GUARD_PLAN}.md",
        "fixture_scenario": f"tests/fixtures/e2e/scenarios/{GUARD_SCENARIO}.yaml",
        "bundle_version": BUNDLE_VERSION,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stderr.write(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
