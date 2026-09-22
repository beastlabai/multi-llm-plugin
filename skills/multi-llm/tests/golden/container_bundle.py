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
* **`tasks_sha256`, `sha256_before` and `sha256_after`** values become type
  tokens — keys kept, values erased, exactly the `runner_versions` treatment.
  These are sha256s **of files on disk**, and one of the files they hash carries
  a wall-clock reading: `update_plan_tasks.generate_tasks_markdown` renders
  `*Generated: <YYYY-MM-DD HH:MM:SS>*` as the third line of every `tasks.md`.
  `normalize_run` folds a timestamp out of a JSON *value* (that same line, where
  it reaches a prompt, is already `<timestamp>`), but it cannot fold one out of a
  **hash** of a file that contains one — so every `tasks.md` digest is effectively
  a fresh random number per run. Measured 2026-09-22 on `chain-omp`,
  `chain-writer-opencode` and `chain-writer-claude-code`: two consecutive captures
  of the same row differed in exactly 23 pointers, all of them that one file's
  digest (11 `tasks_sha256`, 12 `sha256_before`/`sha256_after` entries keyed by
  `plans/<plan>/tasks/tasks.md`), payload byte-length identical, zero diffs
  elsewhere. A value that cannot repeat grades nothing; it only prevents any row
  that runs the task phases from ever having a baseline.

  **Erased by key rather than by path**, which costs something and is the honest
  trade: the other 40 hash pointers in a chain bundle (`plans/<plan>.md`,
  `src/*.py`, `tests/*.py`) *are* stable, and their values stop being compared
  too. What those uniquely attested is the byte identity of each written file and
  whether `before != after` — i.e. whether a write phase moved bytes. Both remain
  graded one level up, by the same artifacts in the same bundle: `write_outcome`
  (`applied` / `no-op` / the failure words), `files_touched`, `expected_ids` /
  `returned_ids` and `workspace_delta` — which is where the D3g outcome axis is
  decided in the product too, the hashes being the evidence a *reader* checks it
  against rather than the verdict. The alternative — keying the fold on the tasks
  file's path — would encode "this product stamps a clock into that file" in the
  rig, and would stop applying, silently, the day a row runs a differently-named
  plan fixture. The *set* of hashed paths keeps grading either way, which is what
  catches a write that touched a file it should not have.

Nothing else is erased. In particular the content hashes that are derived from
the fixture text rather than read off a clock-stamped file (group ids, suggestion
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
import platform
import subprocess
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
    "ERASED_VALUE_KEYS",
    "WORKSPACE_ROOT",
    "AXES",
    "AXIS_REVIEWER",
    "AXIS_WRITER",
    "capture_bundle",
    "baseline_path",
    "baseline_dir_name",
    "host_provenance",
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

#: Keys whose value is erased while the key itself keeps grading. Two families,
#: one rule: the value varies between two identical runs for a reason that is
#: not this guard's subject, and the *presence* of the key is what a regression
#: would remove.
#:
#: * ``runner_versions`` — the installed CLI pin, C9's subject.
#: * ``tasks_sha256`` / ``sha256_before`` / ``sha256_after`` — sha256s of files
#:   on disk, one of which (``tasks.md``) carries a rendered wall clock, so the
#:   digest is unreproducible by construction. See the module docstring for the
#:   measurement and for what the by-key scope costs.
#:
#: A **scalar** at one of these keys becomes its type token; a **mapping** keeps
#: its keys and tokenizes each value, so the set of hashed paths is still
#: compared exactly.
ERASED_VALUE_KEYS = frozenset({
    "runner_versions",
    "tasks_sha256",
    "sha256_before",
    "sha256_after",
})


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
        if _key in ERASED_VALUE_KEYS:
            # Keys kept, values erased: a *new* runner — or a write that touched
            # a file no baseline run touched — still shows as a diff; a moved CLI
            # pin, or a digest of a file the product stamps with a clock, does
            # not. `None` is left as itself rather than tokenized: on a
            # `sha256_before` map it means "this file did not exist before the
            # batch", which is a fact about the run and not a volatile value.
            return {
                key: ("<str>" if isinstance(item, str) else fold(item, key))
                for key, item in sorted(value.items())
            }
        return {key: fold(value[key], key) for key in sorted(value)}
    if isinstance(value, list):
        return [fold(item, _key) for item in value]
    if _key in ERASED_VALUE_KEYS:
        # The scalar spelling of the same rule — `tasks_sha256` is one string,
        # not a map. Type token rather than a bare drop, so a key that stops
        # being written, or starts being written as something other than a
        # string, is still a diff.
        return "<str>" if isinstance(value, str) else value
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

#: The two questions a chain row can be graded on, and the two baseline
#: namespaces that keep them apart.
#:
#: **reviewer** — the historical axis and still the default, so every caller
#: written before the split keeps its exact behaviour: one provider CLI answers
#: the eight-phase fan-out, one shared helper does the writing, and the baseline
#: is `tests/golden/container/<cli>/bundle.json`.
#:
#: **writer** — the axis added by `chain_writer_axis:` in `redirect.yaml`: the
#: reviewer is *pinned* (today `pi`) and the **write runner** is what varies, so
#: the baseline is `tests/golden/container/writer-<cli>/bundle.json`. The
#: separate namespace is not tidiness. `chain-codex` and `chain-writer-codex`
#: are two different runs asking two different questions, and with one path
#: function they would land on the same file and the second capture would
#: silently overwrite the first.
AXIS_REVIEWER = "reviewer"
AXIS_WRITER = "writer"
AXES = (AXIS_REVIEWER, AXIS_WRITER)


def _check_axis(axis: str) -> str:
    """Refuse an unknown axis loudly rather than inventing a directory for it.

    A typo silently resolving to `tests/golden/container/writter-codex/` would
    be captured, never graded, and then reported as an orphan directory by a
    test that cannot say why.
    """
    if axis not in AXES:
        raise ValueError(
            f"unknown baseline axis {axis!r}; expected one of {list(AXES)}")
    return axis


def baseline_dir_name(cli: str, axis: str = AXIS_REVIEWER) -> str:
    """The directory name under `tests/golden/container/` for one baseline.

    Split out from :func:`baseline_path` because the orphan check compares
    *names* against the two declared axes, and it must derive them the same way
    a capture does rather than re-spelling the `writer-` prefix.
    """
    return cli if _check_axis(axis) == AXIS_REVIEWER else f"{AXIS_WRITER}-{cli}"


def baseline_path(cli: str, axis: str = AXIS_REVIEWER) -> Path:
    return BASELINE_DIR / baseline_dir_name(cli, axis) / BASELINE_NAME


def row_id_for(cli: str, axis: str = AXIS_REVIEWER) -> str:
    """The chain row whose evidence a CLI's baseline is captured from.

    Axis-aware for the same reason the path is: the writer axis's rows are
    `chain-writer-<cli>`, and a `row_id` that ignored the axis would point a
    writer manifest at the reviewer row's evidence — a provenance claim that
    reads fine and is false.
    """
    return (f"chain-{cli}" if _check_axis(axis) == AXIS_REVIEWER
            else f"chain-{AXIS_WRITER}-{cli}")


def load_baseline(cli: str, axis: str = AXIS_REVIEWER) -> Dict[str, Any]:
    return json.loads(baseline_path(cli, axis).read_text(encoding="utf-8"))


def write_baseline(cli: str, bundle: Dict[str, Any],
                   manifest: Dict[str, Any],
                   axis: str = AXIS_REVIEWER) -> Path:
    """Write ``tests/golden/container/<dir>/bundle.json`` with its manifest.

    The manifest records where the bundle came from and is **excluded from the
    diff** (`compare_payload`), like the instruction guard's: it describes the
    capture, not the run. That exclusion is what lets the three fields stamped
    here be host- and axis-specific without destabilising a single comparison.

    Three fields are stamped rather than trusted to the caller:

    * **`axis`** — which question this bundle answers. A writer bundle and the
      reviewer bundle of the same CLI have the same *shape*, so nothing in the
      payload distinguishes them; without this field a file moved between
      directories by hand would be graded as whatever its new parent implies.
    * **`reviewer`** — the **reviewer pin**: which CLI answered the fan-out.
      The writer axis holds the reviewer fixed (`chain_writer_axis:`'s
      `reviewer` key, today `pi`) precisely so the three writer baselines
      differ from one another *only* in the write half. A writer bundle
      captured against a different reviewer is therefore not comparable to its
      siblings at all, and the pin is recorded beside the axis so that a
      re-seed under a changed pin is a visible manifest diff rather than a
      silently incomparable payload. Stamped on reviewer bundles too, where it
      is the CLI itself, so a reader never has to know the axis to answer "who
      reviewed".
    * **`host`** — :func:`host_provenance`, back-filled when the caller did not
      supply it. This exists so a *frozen* baseline records the host it can no
      longer be reproduced on, rather than leaving that in a reviewer's memory.
      The live case: `tests/golden/container/codex/bundle.json` was captured
      `2026-08-27T14:10:33Z` from `run_bundle: 20260827T131354Z`, three weeks
      before the D6 sandbox probe landed on 2026-09-17, and it records a green
      codex write path **this host can no longer reproduce** — `chain-codex`'s
      write batches are now refused `reason_code: sandbox_unavailable` before
      any child starts. Nothing in that file says so today.
    """
    _check_axis(axis)
    path = baseline_path(cli, axis)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(bundle)
    stamped = dict(manifest)
    declared = stamped.get("axis")
    if declared is not None and declared != axis:
        # Not a merge: a manifest that says `reviewer` being written to a
        # `writer-` directory means one of the two is a mistake, and guessing
        # which is how the collision this namespace exists to prevent comes
        # back.
        raise ValueError(
            f"manifest declares axis {declared!r} but the baseline is being "
            f"written on the {axis!r} axis ({path})")
    stamped["axis"] = axis
    stamped.setdefault("reviewer", cli if axis == AXIS_REVIEWER else None)
    if stamped.get("host") is None:
        stamped["host"] = host_provenance()
    document["manifest"] = stamped
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Host provenance
# ---------------------------------------------------------------------------

#: The two sysctls that decide, at kernel level, whether an unprivileged user
#: namespace may be created. Read rather than shelled out to: both are plain
#: files, both are free, and neither needs a binary on `PATH`.
_USERNS_SYSCTLS = (
    ("kernel.unprivileged_userns_clone",
     "/proc/sys/kernel/unprivileged_userns_clone"),
    ("kernel.apparmor_restrict_unprivileged_userns",
     "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"),
)

#: Seconds to wait on the `unshare` probe. It either answers instantly or the
#: machine is in no state to be capturing baselines.
_USERNS_PROBE_TIMEOUT_S = 5.0


def _read_sysctl(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _userns_probe() -> Dict[str, Any]:
    """Whether this host let an unprivileged user namespace be created.

    **What it proves.** `unshare -Un true` performs the actual
    `clone(CLONE_NEWUSER)` as this user, so `permitted` means the syscall
    really succeeded here at capture time, not that a sysctl was set to a
    permissive value.

    **What it does NOT prove, measured on this host and worth the words.** On
    `6.17.0-35-generic` the probe exits 0, `kernel.unprivileged_userns_clone`
    reads `1` and `kernel.apparmor_restrict_unprivileged_userns` also reads
    `1` — and codex's bubblewrap still cannot start
    (`bwrap: No permissions to create a new namespace`), because the AppArmor
    restriction bites per *profile* and `bwrap` is confined where `unshare` is
    not. So a `permitted` verdict here is emphatically **not** a claim that any
    runner's sandbox can start. All three readings are recorded together for
    exactly that reason: they are the cheap, reproducible facts about the
    machine, and the only thing that answers "can this runner sandbox itself"
    is the product's own `sandbox_verdict(provider, 'agent-write')` probe, run
    inside the image. This field deliberately does not re-derive it — a second
    copy of the detection drifts from the first the moment the probe changes.
    """
    sysctl = {name: _read_sysctl(path) for name, path in _USERNS_SYSCTLS}
    verdict, how = "unknown", "no probe answered"
    try:
        completed = subprocess.run(
            ["unshare", "-Un", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_USERNS_PROBE_TIMEOUT_S, check=False)
        verdict = "permitted" if completed.returncode == 0 else "denied"
        how = "unshare -Un true"
    except (OSError, subprocess.SubprocessError):
        # No `unshare` on PATH, or it could not run. Fall back to the sysctl,
        # which is weaker evidence — it is a permission, not an attempt — and
        # say so in `probe` rather than passing it off as the same reading.
        clone = sysctl.get("kernel.unprivileged_userns_clone")
        if clone in ("0", "1"):
            verdict = "permitted" if clone == "1" else "denied"
            how = "kernel.unprivileged_userns_clone (no unshare on PATH)"
    return {"verdict": verdict, "probe": how, "sysctl": sysctl}


def host_provenance() -> Dict[str, Any]:
    """Which machine and kernel produced a capture (§5.1, open question 5).

    Recorded in the manifest, which `compare_payload` strips, so this can vary
    per host without turning a single baseline red.

    The hostname is stored as a **digest**, not in the clear: these files are
    committed, and the question a reviewer actually asks of this field is "was
    this captured on the same machine as that one, and can I reproduce it
    here?" — which a stable token answers and a machine name only decorates.
    The kernel release and the userns verdict are the substantive halves: they
    are what says a frozen baseline belongs to a host configuration that no
    longer exists.
    """
    node = platform.node() or ""
    return {
        "kernel_release": platform.release(),
        "system": platform.system(),
        "machine": platform.machine(),
        # Salted with nothing and truncated: a correlation token, not a secret.
        "host_id": ("sha256:"
                    + hashlib.sha256(node.encode("utf-8")).hexdigest()[:12]),
        "unprivileged_userns": _userns_probe(),
    }


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
