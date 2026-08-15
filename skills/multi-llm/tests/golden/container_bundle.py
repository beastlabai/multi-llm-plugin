#!/usr/bin/env python3
"""The container lane's normalized run bundle: one per CLI, from an exported row (C8).

The instruction-refactor guard captures its bundle by *running* the workflow in
process (`instruction_refactor_bundle.capture`). The container lane cannot: its
run happens inside a container that is gone by the time anything grades it, and
the only thing left is what the entrypoint's `finally` export copied to
`/work/out`. So this module captures from **evidence on disk** — a row bundle
under `tests/container/runs/<utc>/<row-id>/` — and produces a bundle in the same
shape, folded by the same shared rules (`bundle_compare`), so "the same run" means
the same thing on both guards.

**Why per CLI.** Every chain row drives the identical fixture chain through the
identical product code; what differs is the binary in the middle and the model
spec it was pinned to. A single lane-wide baseline would therefore have to erase
exactly the things that distinguish one CLI's parsed output from another's — which
is the drift this gate exists to catch. Hence
`tests/golden/container/<cli>/bundle.json`, and hence an allowlist entry that must
name the CLI it excuses.

**What is compared** — orchestrator-produced artifacts only: the phase JSONs, the
`.status.json` phase/step sequences and shapes, `state.json` plus its structure
and the structure of the metrics it carries, the set of relative paths the run
created under the plan's output dir, the derived scalars (group/suggestion counts,
importance histogram, validation partition, phases completed), the `[MARKER]`
sequence and the `MULTI_LLM_RESULT` line of every dispatcher invocation, each
command's exit code, and which workspace files outside the output dir moved.

**What is folded**, beyond `normalize_run`'s own volatile keys:

* **long text** (> ``TEXT_DIGEST_THRESHOLD`` chars) and every ``prompt`` become a
  ``<text:sha256:…:len>`` token. C8's rule is "never raw model text": a change is
  still a diff, because the digest moves, but the review reads a token instead of
  two kilobytes of prose, and a prompt body stays pinned where it belongs —
  `tests/golden/prompts/`, in plain CI.
* **`runner_versions`** values become type tokens. The installed CLI version is
  C9's subject (`cli-versions.lock`, the drift report, the run record); a moved
  pin must show up there and must not turn every container baseline red at the
  same time.

Nothing else is erased. In particular the content hashes (group ids, suggestion
ids, `prior_changes_id`) stay compared, because a changed grouping is precisely
what a silently-changed parse produces.

Capture is deliberately **separate from asserting**, like `expected_argv.yaml`:
`tests/container/seed_container_golden.py` writes baselines on the host from a
green run, and `tests/test_container_full_chain.py` only ever reads them. A guard
that can rewrite its own baseline is not a guard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# The shared machinery — normalize/diff/allowlist plus the capture primitives the
# instruction guard folds with. Imported both ways for the same reason
# `bundle_compare` is: the container lane reaches this as a package member, the
# host-side seeder puts `tests/golden` on `sys.path` and imports it flat.
try:  # pragma: no cover - exercised by whichever import style ran
    from .bundle_compare import (  # noqa: F401
        COMPARED_JSON,
        derived_scalars,
        diff_bundles,
        load_json,
        markers,
        normalize_run,
        relative_paths,
        status_steps,
        structure,
    )
except ImportError:  # flat import, `tests/golden` on sys.path
    from bundle_compare import (  # noqa: F401
        COMPARED_JSON,
        derived_scalars,
        diff_bundles,
        load_json,
        markers,
        normalize_run,
        relative_paths,
        status_steps,
        structure,
    )

__all__ = [
    "CONTAINER_BUNDLE_VERSION",
    "CHAIN_PHASE_DIRS",
    "CONTAINER_COMPARED_JSON",
    "WORKSPACE_ROOT",
    "capture_bundle",
    "baseline_path",
    "load_baseline",
    "write_baseline",
    "row_id_for",
]

#: Bundle *shape* version. Bumped when the layout below changes, so a baseline
#: captured against an older shape fails loudly instead of diffing nonsensically.
CONTAINER_BUNDLE_VERSION = 1

#: The in-container workspace root. A constant rather than a discovered path, and
#: that is the point: every absolute path in a container artifact is a path inside
#: an image, identical on every machine that runs the row, so relativizing against
#: it makes the bundle machine-independent without erasing anything.
WORKSPACE_ROOT = "/work/sample-project"

#: Phase **directory** names, in chain order. Note the two places a directory name
#: is not the mode word: `apply-code-fixes` writes under `apply-fixes/`, and mode
#: `review-code` is phase `code-review`. `ask` and `full` are covered by
#: `output_root_paths` and the status map rather than listed here — `ask`'s status
#: file lives one level deeper, under its question slug.
CHAIN_PHASE_DIRS = (
    "review-plan",
    "apply-suggestions",
    "generate-tasks",
    "review-tasks",
    "apply-task-suggestions",
    "implement",
    "code-review",
    "apply-fixes",
)

#: The shared compared set plus the four documents only a real executor run
#: produces. `write_batch_report.json` is the important addition: it carries the
#: D3g outcome axis (`applied` / `clean-failure` / …), the observed `files_touched`
#: and `finalized`, which is where a scheduler regression shows up first.
CONTAINER_COMPARED_JSON = tuple(COMPARED_JSON) + (
    "write_batch_report.json",
    "write_tasks.json",
    "task_generation_result.json",
    "task_generation_tasks.json",
)

#: Strings longer than this are compared as a digest rather than verbatim.
TEXT_DIGEST_THRESHOLD = 200

_RESULT_MARKER = "MULTI_LLM_RESULT:"


# ---------------------------------------------------------------------------
# The container lane's own folds, on top of the shared normalization
# ---------------------------------------------------------------------------

def _digest(text: str) -> str:
    """A stable token that changes when the text does and carries its length."""
    return (f"<text:sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
            f":{len(text)}>")


def fold(value: Any, _key: str = "") -> Any:
    """Apply the container lane's extra folds to an already-normalized value."""
    if isinstance(value, dict):
        if _key == "runner_versions":
            # Keys kept, values erased: a *new* runner still shows as a diff, a
            # moved CLI pin does not.
            return {
                key: ("<str>" if isinstance(item, str) else fold(item, key))
                for key, item in sorted(value.items())
            }
        return {key: fold(value[key], key) for key in sorted(value)}
    if isinstance(value, list):
        return [fold(item, _key) for item in value]
    if isinstance(value, str) and (_key == "prompt"
                                   or len(value) > TEXT_DIGEST_THRESHOLD):
        return _digest(value)
    return value


def _normalize(value: Any) -> Any:
    return fold(normalize_run(value, WORKSPACE_ROOT))


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def output_root(out_dir: Path) -> Path | None:
    """The exported plan output dir: ``<bundle>/artifacts/<plan-stem>/``.

    Discovered rather than named, so a row driven from a differently-named plan
    fixture captures rather than silently returning an empty bundle.
    """
    artifacts = Path(out_dir) / "artifacts"
    if not artifacts.is_dir():
        return None
    children = sorted(p for p in artifacts.iterdir() if p.is_dir())
    return children[0] if len(children) == 1 else None


def _phase_bundle(phase_dir: Path) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {"exists": phase_dir.is_dir()}
    if not phase_dir.is_dir():
        return bundle
    artifacts: Dict[str, Any] = {}
    for name in CONTAINER_COMPARED_JSON:
        candidate = phase_dir / name
        if candidate.exists():
            artifacts[name] = _normalize(load_json(candidate))
    bundle["artifacts"] = artifacts
    return bundle


def _status_map(root: Path) -> Dict[str, Any]:
    """Every ``.status.json`` under the output root, keyed by relative path.

    Keyed by path rather than by phase because two of them are not directly under
    a phase dir — `ask` writes one under its question slug, and `full` owns a run
    dir of its own — and both are part of the sequence a chain must reproduce.
    """
    out: Dict[str, Any] = {}
    for path in sorted(root.rglob(".status.json")):
        raw = load_json(path)
        out[path.relative_to(root).as_posix()] = {
            "steps": status_steps(raw),
            "shape": structure(raw),
        }
    return out


def _commands(out_dir: Path) -> List[Dict[str, Any]]:
    """``commands.json``: one entry per dispatcher invocation, in order."""
    path = Path(out_dir) / "commands.json"
    if not path.is_file():
        return []
    records = load_json(path)
    if not isinstance(records, list):
        return []
    return [
        {
            "command": _normalize(record.get("command")),
            "exit": record.get("exit"),
            "env": sorted(record.get("env") or []),
        }
        for record in records
        if isinstance(record, dict)
    ]


def _text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _result_lines(text: str) -> List[Any]:
    """Every ``MULTI_LLM_RESULT`` line, in order — one per dispatcher invocation.

    A list rather than a map keyed by mode: the order the chain emits them in is
    part of what a row reproduces, and a row that dispatched the same mode twice
    would otherwise lose one.
    """
    out: List[Any] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_RESULT_MARKER):
            continue
        payload = stripped[len(_RESULT_MARKER):].strip()
        try:
            out.append(_normalize(json.loads(payload)))
        except ValueError:  # pragma: no cover - reported in-bundle
            out.append({"<unparseable>": payload[:200]})
    return out


def _workspace_delta(out_dir: Path, root_rel: str) -> List[str]:
    """Which workspace files moved, by name, **outside** the plan's output dir.

    Names, never hashes: the plan file's `after` hash is a function of the marker
    the run inserted into it, and the output dir's own churn is already pinned by
    `output_root_paths`. What is left is the question this line answers — did a
    write phase move bytes in the sample project, and which ones.
    """
    before = load_json(Path(out_dir) / "workspace-census-before.json")
    after = load_json(Path(out_dir) / "workspace-census-after.json")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    moved = [
        name for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
        and not name.startswith(f"{root_rel}/")
    ]
    return moved


def capture_bundle(out_dir: Path) -> Dict[str, Any]:
    """Build the normalized bundle for one exported row bundle.

    *out_dir* is a `tests/container/runs/<utc>/<row-id>/` directory — what
    `harness.RowResult.out_dir` points at, and what the driver has after the
    container is gone.
    """
    out_dir = Path(out_dir)
    root = output_root(out_dir)
    stdout = _text(out_dir / "container-stdout.txt")
    stderr = _text(out_dir / "container-stderr.txt")

    state: Any = {}
    grouped: Any = {}
    validation: Any = {}
    phases: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    paths: List[str] = []
    delta: List[str] = []

    if root is not None:
        state = load_json(root / "state.json") if (root / "state.json").is_file() else {}
        review = root / "review-plan"
        if (review / "grouped.json").is_file():
            grouped = load_json(review / "grouped.json")
        if (review / "validation.json").is_file():
            validation = load_json(review / "validation.json")
        phases = {name: _phase_bundle(root / name) for name in CHAIN_PHASE_DIRS}
        status = _status_map(root)
        paths = relative_paths(root)
        # `plans/auth-feature` — the output dir as the workspace census names it.
        delta = _workspace_delta(out_dir, _census_prefix(out_dir, root))

    return {
        "bundle_version": CONTAINER_BUNDLE_VERSION,
        "commands": _commands(out_dir),
        "markers": markers(stdout) + markers(stderr),
        "result_lines": _result_lines(stdout),
        "phases": phases,
        "status": status,
        "output_root_paths": paths,
        "state": _normalize(state),
        "state_shape": structure(state),
        "metrics_shape": structure(
            state.get("metrics") if isinstance(state, dict) else None),
        "derived": derived_scalars(grouped, validation, state),
        "workspace_delta": delta,
    }


def _census_prefix(out_dir: Path, root: Path) -> str:
    """Where the exported output dir sat inside the workspace census.

    The census is keyed relative to `/work/sample-project`, and the output dir is
    `<plan-parent>/<plan-stem>` under it. Recovered from the census itself rather
    than assumed, so a row whose plan lives somewhere else still resolves.
    """
    before = load_json(Path(out_dir) / "workspace-census-before.json")
    if isinstance(before, dict):
        for name in before:
            # `plans/auth-feature.md` -> `plans/auth-feature`
            if name.endswith(f"/{root.name}.md") or name == f"{root.name}.md":
                return name[: -len(".md")]
    return root.name


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

#: Where the per-CLI baselines live. A sibling of the allowlist that governs them.
BASELINE_DIR = Path(__file__).resolve().parent / "container"
BASELINE_NAME = "bundle.json"
ALLOWLIST_PATH = BASELINE_DIR / "expected_changes.yaml"


def baseline_path(cli: str) -> Path:
    return BASELINE_DIR / cli / BASELINE_NAME


def row_id_for(cli: str) -> str:
    """The chain row whose evidence a CLI's baseline is captured from."""
    return f"chain-{cli}"


def load_baseline(cli: str) -> Dict[str, Any]:
    return json.loads(baseline_path(cli).read_text(encoding="utf-8"))


def write_baseline(cli: str, bundle: Dict[str, Any],
                   manifest: Dict[str, Any]) -> Path:
    """Write ``tests/golden/container/<cli>/bundle.json`` with its manifest.

    The manifest records where the bundle came from and is **excluded from the
    diff** (`compare_payload`), like the instruction guard's: it describes the
    capture, not the run.
    """
    path = baseline_path(cli)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(bundle)
    document["manifest"] = manifest
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def compare_payload(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """A baseline with its manifest removed — the half that is diffed."""
    payload = dict(baseline)
    payload.pop("manifest", None)
    return payload


def entries_for(cli: str, entries: Iterable[dict]) -> List[dict]:
    """The allowlist entries that apply to *cli*.

    Every container entry must name a `cli`. Without that, one CLI's declared
    exception would excuse the same pointer on the other ten — and the stale-entry
    rule could not run per row at all, because an entry that legitimately explains
    only `codex`'s diff would be reported as stale on every other row.
    """
    return [entry for entry in entries if str(entry.get("cli") or "") == cli]


def validate_entries(entries: Sequence[dict], path: Path) -> None:
    """The container lane's extra allowlist rule: every entry names a CLI."""
    for entry in entries:
        if not entry.get("cli"):
            raise AssertionError(
                f"{path.name}: entry {entry.get('pointer')!r} names no `cli`. A "
                f"container allowlist entry is per CLI — an unscoped one would "
                f"excuse the same pointer on every other row and could never be "
                f"checked for staleness."
            )
