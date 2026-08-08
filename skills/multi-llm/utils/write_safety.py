#!/usr/bin/env python3
"""Everything a headless *write* dispatch must be safe before it is useful.

``utils/subagent_executor.py`` hardens what the executor writes and where the
runner is rooted for the read/analyze half. The write half's exposure is one level
out: ``claude -p --permission-mode acceptEdits``, ``codex exec --sandbox
workspace-write`` and opencode's ``build`` agent **pre-approve every write the
model chooses to make**, anywhere in the repository, and under ``--detach`` or in
CI there is no human between the model and the working tree at all. "Sandbox flags
per adapter" is a boundary, not a security model. The five controls here make it
one, all on by default:

* **A pinned repo root** (D3h). ``git rev-parse --show-toplevel`` of the target
  workspace, ``realpath``-resolved, and every path the executor snapshots,
  expects a batch to touch, or observes it having touched must resolve beneath it.
  Both sides of every containment comparison are canonicalized — asymmetry is the
  bug (D1c), not the shortcut.
* **A declared write policy, enforced twice** (D3l). ``driver.write_paths``
  ``allow``/``deny`` globs resolved against that root. **Pre-dispatch**, a batch
  whose declared file set is not wholly inside the policy is refused before the
  runner starts. **Post-run**, the observed touched set is recomputed and anything
  outside the policy is a containment violation: the batch fails, the offending
  files are restored from their pre-batch snapshot, the loop stops. Stated
  honestly: post-run enforcement is *detective*. The runner's own sandbox stays
  the preventive control; this is the second layer that makes the violation loud.
* **A mandatory pre-batch snapshot** (D3h/D3g). Under ``{phase}/pre_batch/{task_id}/``
  — never a shared workspace temp path — so concurrent or resumed batches cannot
  clobber each other's rollback point, and a partial application leaves
  recoverable bytes. The snapshot is also the *verdict*: the model's claim is
  never trusted, byte comparison decides.
* **Dispatch-time disjointness admission** (D3h). Parallel write dispatch is
  admitted only when the in-flight tasks' declared file sets are pairwise
  disjoint; a task declaring no file set runs alone. Checked executor-side rather
  than trusted from the batcher, so a hand-edited task file cannot produce two
  agents editing one file.
* **An append-only audit record** (D3l). One fsynced line per dispatch and one
  per classification in ``{phase}/subagent_audit.jsonl``, with the prompt reduced
  to its sha256 and byte length. The prompt body, the runner's environment and any
  auth value are **never** logged: prompts are large and carry untrusted text, and
  an audit log that leaks credentials is a new vulnerability rather than a control.

Scope, so the claim is not oversold: this is workspace-write discipline, not a
sandbox-escape defense. A runner that ignores its own sandbox flags, or a host
that grants the executor more than the repo, is outside what this can enforce.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from .provider_registry import get_write_paths
    from .subagent_tasks import declared_file_set
except ImportError:  # pragma: no cover - direct-script import shim
    from utils.provider_registry import get_write_paths
    from utils.subagent_tasks import declared_file_set


__all__ = [
    "AUDIT_FILENAME",
    "DRY_RUN_FILENAME",
    "PRE_BATCH_DIRNAME",
    "WriteSafetyError",
    "PathEscape",
    "PolicyViolation",
    "WritePolicy",
    "AuditLog",
    "TaskSnapshot",
    "SnapshotEntry",
    "resolve_repo_root",
    "contain_under_root",
    "relative_to_root",
    "admit_disjoint_groups",
    "file_sha256",
]


#: The append-only write log (D3l). A ``.jsonl`` name, so the reaggregation glob
#: (``{phase_dir}/*.json``) never sees it.
AUDIT_FILENAME = "subagent_audit.jsonl"

#: ``--dry-run-subagents``'s preflight report (D3l).
DRY_RUN_FILENAME = "subagent_dry_run.json"

#: Where per-task snapshots live, under the *phase* dir (D3h) rather than the
#: shared workspace temp path the markdown uses, so concurrent or resumed batches
#: cannot clobber each other's rollback point.
PRE_BATCH_DIRNAME = "pre_batch"

#: The per-snapshot manifest inside ``{phase}/pre_batch/{task_id}/``.
MANIFEST_NAME = "manifest.json"

#: Snapshot payloads live in a sub-directory so the manifest name can never
#: collide with a snapshotted file's own flattened name.
PAYLOAD_DIRNAME = "files"

#: Refuse to snapshot (and therefore to dispatch) a file larger than this. A
#: snapshot that silently did not happen is worse than a refused batch: the
#: classification would read "unchanged" for a file with no rollback point.
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024


class WriteSafetyError(Exception):
    """A write dispatch the executor refuses. Exit 3 ("needs a human", D6)."""

    exit_code = 3

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PathEscape(WriteSafetyError):
    """A path from a task file resolves outside the pinned repo root (D3f/D3h)."""


class PolicyViolation(WriteSafetyError):
    """A path is outside ``driver.write_paths`` (D3l). Carries the offenders."""

    def __init__(self, message: str, paths: Sequence[str], rule: str = "") -> None:
        super().__init__(message)
        self.paths = list(paths)
        self.rule = rule


# ---------------------------------------------------------------------------
# The pinned repo root (D3h) and containment (D3f)
# ---------------------------------------------------------------------------

def resolve_repo_root(anchor: Optional[str]) -> str:
    """The write path's one root: ``git rev-parse --show-toplevel``, canonicalized.

    ``realpath``, not ``abspath``: this value is one side of every containment
    comparison below and the other side is canonicalized too, so a symlinked
    checkout compares equal instead of every path reading as an escape. (The
    opposite rule applies to the *skill* dir, which must keep its symlinks — see
    ``utils/cli_common``. The distinction is exactly D1c: canonicalize for
    containment, preserve for invocation.)

    Falls back to the anchor's own directory when the anchor is not in a git repo,
    because a plan in a bare directory is a legitimate target — but the root is
    still pinned to *something* explicit, never left to the process cwd.
    """
    start = Path(anchor).resolve() if anchor else Path.cwd().resolve()
    probe_dir = str(start if start.is_dir() else start.parent)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=probe_dir, timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return str(Path(probe_dir).resolve())
    if completed.returncode == 0 and completed.stdout.strip():
        return str(Path(completed.stdout.strip()).resolve())
    return str(Path(probe_dir).resolve())


def contain_under_root(candidate: str, root: str, *, what: str = "path") -> Path:
    """Resolve ``candidate`` and require it beneath ``root``, or raise (D3f/D3h).

    Resolution happens **after** joining, so a symlinked path component cannot
    redirect the write, and *both* sides are resolved — an asymmetric comparison
    (one resolved, one not) is the classic way this check passes while the write
    lands outside.
    """
    resolved_root = Path(root).resolve()
    raw = Path(candidate)
    joined = raw if raw.is_absolute() else (resolved_root / raw)
    if _looks_like_a_windows_escape(str(candidate)):
        raise PathEscape(
            f"Refusing the {what} {candidate!r}: a drive-relative, UNC or "
            f"alternate-data-stream path is never resolved against the repo root. "
            f"Repo root: {resolved_root}. Nothing was dispatched."
        )
    resolved = joined.resolve()
    try:
        contained = resolved.is_relative_to(resolved_root)
    except AttributeError:  # pragma: no cover - Python < 3.9
        contained = os.path.commonpath(
            [str(resolved), str(resolved_root)]
        ) == str(resolved_root)
    if not contained:
        raise PathEscape(
            f"Refusing the {what} {candidate!r}: it resolves to {resolved}, outside "
            f"the pinned repo root {resolved_root}. A batch cannot be steered into "
            f"$HOME or a sibling checkout through an injected path. Nothing was "
            f"dispatched."
        )
    return resolved


def _looks_like_a_windows_escape(text: str) -> bool:
    """Drive-relative (``C:foo``), UNC (``\\\\host\\share``) or an ADS (``a:b``)."""
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    if re.match(r"^[A-Za-z]:[^\\/]", text):
        return True
    # An NTFS alternate data stream: a colon that is not a drive separator.
    return bool(re.search(r"(?<![A-Za-z]):(?=[^\\/])", text[2:] if len(text) > 2 else ""))


def relative_to_root(path: str, root: str) -> str:
    """``path`` as a POSIX repo-relative string, for policy matching and logs."""
    resolved_root = Path(root).resolve()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = resolved_root / resolved
    try:
        return resolved.resolve().relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved.resolve().as_posix()


def file_sha256(path: str) -> Optional[str]:
    """The file's sha256, or ``None`` when it does not exist.

    ``None`` is a real state, not an error: the artifact an apply batch edits is
    frequently untracked and may not exist yet, and "absent then present" is the
    creation case the classification has to distinguish from "unchanged".
    """
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except (OSError, IsADirectoryError):
        return None


# ---------------------------------------------------------------------------
# D3l — the declared write policy
# ---------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    """Translate a ``write_paths`` glob into an anchored regex over POSIX paths.

    ``**/`` spans zero or more directories (so ``**/.env`` matches both ``.env``
    and ``a/b/.env``), ``**`` spans anything, ``*`` and ``?`` stop at a separator.
    Hand-rolled rather than ``fnmatch`` because ``fnmatch``'s ``*`` crosses ``/``,
    which would make ``**/node_modules/**`` and ``*.pem`` mean the same thing.
    """
    text = pattern.replace("\\", "/")
    # Strip a leading "./" PREFIX only — `str.lstrip("./")` strips leading
    # characters, which silently turned `.git/**` into `git/**` and let every
    # `.git` write through the deny list.
    while text.startswith("./"):
        text = text[2:]
    out: List[str] = ["^"]
    index = 0
    while index < len(text):
        char = text[index]
        if text.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
            continue
        if text.startswith("**", index):
            out.append(".*")
            index += 2
            continue
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(char))
        index += 1
    out.append("$")
    return re.compile("".join(out))


@dataclass
class WritePolicy:
    """``driver.write_paths``, resolved against the pinned repo root (D3l)."""

    root: str
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: str, *, anchor: Optional[str] = None) -> "WritePolicy":
        """Read the merged ``driver.write_paths`` through the typed accessor."""
        configured = get_write_paths(anchor=anchor)
        return cls(root=str(Path(root).resolve()),
                   allow=list(configured["allow"]),
                   deny=list(configured["deny"]))

    def __post_init__(self) -> None:
        self._allow = [_glob_to_regex(p) for p in self.allow]
        self._deny = [_glob_to_regex(p) for p in self.deny]

    def violation(self, path: str) -> Optional[str]:
        """The rule ``path`` violates, or ``None`` when the policy permits it.

        Deny wins: a path matched by both lists is denied, because the deny list is
        the statement of what a code-editing batch has no business touching and an
        ``allow: ["**"]`` default would otherwise neutralize it entirely.
        """
        relative = relative_to_root(path, self.root)
        if Path(relative).is_absolute() or relative.startswith(".."):
            return f"outside the pinned repo root {self.root}"
        for pattern, matcher in zip(self.deny, self._deny):
            if matcher.match(relative):
                return f"driver.write_paths.deny pattern {pattern!r}"
        if not self._allow:
            return "driver.write_paths.allow is empty, so nothing may be written"
        for matcher in self._allow:
            if matcher.match(relative):
                return None
        return (
            f"no driver.write_paths.allow pattern matches it "
            f"(allow: {', '.join(self.allow)})"
        )

    def check(self, paths: Iterable[str], *, when: str) -> None:
        """Raise :class:`PolicyViolation` naming every offender and its rule."""
        offenders: List[Tuple[str, str]] = []
        for path in paths:
            rule = self.violation(path)
            if rule is not None:
                offenders.append((relative_to_root(path, self.root), rule))
        if not offenders:
            return
        detail = "; ".join(f"{path} ({rule})" for path, rule in offenders)
        raise PolicyViolation(
            f"Write policy violation ({when}): {detail}. Repo root: {self.root}. "
            f"driver.write_paths is enforced both before dispatch and after the "
            f"run; widen `allow`/narrow `deny` in .multi-llm/providers.yaml if this "
            f"path is genuinely in scope.",
            paths=[path for path, _ in offenders],
            rule=offenders[0][1],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"root": self.root, "allow": list(self.allow), "deny": list(self.deny)}


# ---------------------------------------------------------------------------
# D3h/D3g — the pre-batch snapshot, which is also the verdict
# ---------------------------------------------------------------------------

@dataclass
class SnapshotEntry:
    """One snapshotted file: where it was, what it hashed to, where the copy is."""

    relative: str
    absolute: str
    existed: bool
    sha256: Optional[str]
    size: int
    payload: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.relative,
            "absolute": self.absolute,
            "existed": self.existed,
            "sha256_before": self.sha256,
            "size_before": self.size,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotEntry":
        return cls(
            relative=str(data.get("path", "")),
            absolute=str(data.get("absolute", "")),
            existed=bool(data.get("existed")),
            sha256=data.get("sha256_before"),
            size=int(data.get("size_before") or 0),
            payload=data.get("payload"),
        )


@dataclass
class TaskSnapshot:
    """The mandatory pre-batch snapshot for one write task (D3h).

    Lives at ``{phase_dir}/pre_batch/{task_id}/`` with a ``manifest.json`` and a
    ``files/`` payload dir. Deliberately **not** ``git``-based: the artifact an
    apply batch edits is frequently untracked, which every ``git diff`` variant
    silently reports as unchanged — the exact failure mode that would make a clean
    failure read as success (D3g).
    """

    task_id: str
    directory: str
    root: str
    entries: List[SnapshotEntry] = field(default_factory=list)

    # -- creation -------------------------------------------------------

    @classmethod
    def take(
        cls,
        *,
        phase_dir: str,
        task_id: str,
        files: Sequence[str],
        root: str,
    ) -> "TaskSnapshot":
        """Snapshot ``files`` before dispatch. Every path is contained first.

        A snapshot is taken even for a file that does not exist yet: "absent
        before, present after" is the creation case, and recording the absence is
        what lets ``util restore-batch`` delete a file the batch created.
        """
        directory = Path(phase_dir) / PRE_BATCH_DIRNAME / _safe_component(task_id)
        payload_dir = directory / PAYLOAD_DIRNAME
        payload_dir.mkdir(parents=True, exist_ok=True)
        entries: List[SnapshotEntry] = []
        for index, candidate in enumerate(files):
            resolved = contain_under_root(candidate, root, what="snapshot target")
            relative = relative_to_root(str(resolved), root)
            existed = resolved.is_file()
            size = resolved.stat().st_size if existed else 0
            if existed and size > MAX_SNAPSHOT_BYTES:
                raise WriteSafetyError(
                    f"Refusing to dispatch task {task_id}: {relative} is {size} "
                    f"bytes, over the {MAX_SNAPSHOT_BYTES}-byte snapshot cap. A "
                    f"batch with no rollback point is not dispatched — every "
                    f"declared file must be snapshottable (D3h)."
                )
            payload_name = f"{index:03d}-{_safe_component(relative)}"
            payload = None
            if existed:
                shutil.copy2(str(resolved), str(payload_dir / payload_name))
                payload = payload_name
            entries.append(SnapshotEntry(
                relative=relative,
                absolute=str(resolved),
                existed=existed,
                sha256=file_sha256(str(resolved)) if existed else None,
                size=size,
                payload=payload,
            ))
        snapshot = cls(task_id=task_id, directory=str(directory), root=str(root),
                       entries=entries)
        snapshot.save()
        return snapshot

    def save(self) -> str:
        path = Path(self.directory) / MANIFEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(str(path), json.dumps({
            "snapshot_version": 1,
            "task_id": self.task_id,
            "root": self.root,
            "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "files": [entry.to_dict() for entry in self.entries],
        }, indent=2))
        return str(path)

    @classmethod
    def load(cls, phase_dir: str, task_id: str) -> Optional["TaskSnapshot"]:
        """Re-read a snapshot from disk (resume, and ``util restore-batch``)."""
        directory = Path(phase_dir) / PRE_BATCH_DIRNAME / _safe_component(task_id)
        manifest = directory / MANIFEST_NAME
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls(
            task_id=str(data.get("task_id", task_id)),
            directory=str(directory),
            root=str(data.get("root", "")),
            entries=[SnapshotEntry.from_dict(row) for row in data.get("files", [])],
        )

    # -- the verdict ----------------------------------------------------

    def changed(self) -> List[str]:
        """Repo-relative paths whose bytes moved since the snapshot (D3g).

        Byte comparison, in Python, on the artifacts the batch declared — the
        classification the markdown performs with ``diff -u``, moved into code so
        both invokers reach the same verdict.
        """
        moved: List[str] = []
        for entry in self.entries:
            now = file_sha256(entry.absolute)
            before = entry.sha256 if entry.existed else None
            if now != before:
                moved.append(entry.relative)
        return moved

    def observed(self) -> Dict[str, Dict[str, Any]]:
        """Per-file before/after hashes, for the audit record (D3l)."""
        out: Dict[str, Dict[str, Any]] = {}
        for entry in self.entries:
            out[entry.relative] = {
                "sha256_before": entry.sha256,
                "sha256_after": file_sha256(entry.absolute),
                "existed_before": entry.existed,
                "exists_after": Path(entry.absolute).is_file(),
            }
        return out

    # -- rollback -------------------------------------------------------

    def entry_for(self, relative: str) -> Optional[SnapshotEntry]:
        for entry in self.entries:
            if entry.relative == relative:
                return entry
        return None

    def restore(
        self,
        paths: Optional[Sequence[str]] = None,
        *,
        expected: Optional[Dict[str, Optional[str]]] = None,
        force: bool = False,
    ) -> Dict[str, str]:
        """Restore snapshot bytes for ``paths`` (default: everything snapshotted).

        Args:
            expected: ``relative -> sha256`` the audit recorded at completion. A
                file whose current hash no longer matches is **refused** rather
                than restored: someone edited it since the run, and silently
                reverting their edit is a second data-loss bug. ``--force``
                overrides, deliberately explicitly.

        Returns:
            ``relative -> "restored" | "deleted" | "skipped: <why>"``.
        """
        wanted = list(paths) if paths is not None else [e.relative for e in self.entries]
        report: Dict[str, str] = {}
        for relative in wanted:
            entry = self.entry_for(relative)
            if entry is None:
                report[relative] = "skipped: not in this task's snapshot"
                continue
            if expected is not None and not force:
                recorded = expected.get(relative, "__missing__")
                if recorded != "__missing__":
                    current = file_sha256(entry.absolute)
                    if current != recorded:
                        report[relative] = (
                            "skipped: changed after the run (recorded "
                            f"{_short(recorded)}, now {_short(current)}) — pass "
                            "--force to overwrite anyway"
                        )
                        continue
            if entry.existed and entry.payload:
                source = Path(self.directory) / PAYLOAD_DIRNAME / entry.payload
                if not source.is_file():
                    report[relative] = "skipped: snapshot payload is missing"
                    continue
                Path(entry.absolute).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), entry.absolute)
                report[relative] = "restored"
            else:
                # Absent before the batch: the correct rollback is deletion.
                try:
                    Path(entry.absolute).unlink()
                    report[relative] = "deleted"
                except FileNotFoundError:
                    report[relative] = "deleted (already absent)"
                except OSError as exc:
                    report[relative] = f"skipped: {exc}"
        return report


def _short(digest: Optional[str]) -> str:
    return (digest or "absent")[:12]


def _safe_component(text: str) -> str:
    """A filesystem-safe single path component derived from ``text``.

    Never a passthrough: ``task_id`` and repo-relative paths both reach this, and
    either could carry a separator, so the sanitization is the containment for the
    snapshot tree itself.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", text or "")
    cleaned = cleaned.strip("._") or "task"
    return cleaned[:160]


def _atomic_write(path: str, payload: str) -> None:
    """Same-directory temp file, 0600, ``os.replace`` — as D3f requires."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    handle = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    os.replace(str(tmp), str(target))


# ---------------------------------------------------------------------------
# D3h — dispatch-time disjointness admission
# ---------------------------------------------------------------------------

def admit_disjoint_groups(
    tasks: Sequence[Dict[str, Any]],
    *,
    ceiling: int,
    root: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """Partition one batch's tasks into co-schedulable groups (D3h).

    A group may run concurrently only when every pair of its tasks declares a
    disjoint file set, every task in it declares ``can_parallelize``, and the group
    is no larger than ``ceiling``. A task declaring **no** file set runs alone —
    "undeclared" is not "touches nothing", and treating it as the latter is how a
    hand-edited task file would end up with two agents editing one file.

    Checked here rather than trusted from the batcher, which is the point: the
    batchers' partitioning invariant is real, but a regenerated or hand-edited task
    file is not bound by it.
    """
    ceiling = max(1, int(ceiling))
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_files: Set[str] = set()

    def flush() -> None:
        nonlocal current, current_files
        if current:
            groups.append(current)
        current = []
        current_files = set()

    for task in tasks:
        declared = declared_file_set(task)
        if root is not None:
            declared = [relative_to_root(path, root) for path in declared]
        parallel_ok = bool(task.get("can_parallelize")) and bool(declared)
        if ceiling == 1 or not parallel_ok:
            flush()
            groups.append([task])
            continue
        overlap = current_files & set(declared)
        if not current:
            current = [task]
            current_files = set(declared)
            continue
        if overlap or len(current) >= ceiling:
            flush()
            current = [task]
            current_files = set(declared)
            continue
        current.append(task)
        current_files |= set(declared)
    flush()
    return groups


def overlapping_pairs(
    tasks: Sequence[Dict[str, Any]], *, root: Optional[str] = None
) -> List[Tuple[str, str, List[str]]]:
    """Every pair of tasks whose declared file sets intersect, for a refusal message."""
    resolved: List[Tuple[str, Set[str]]] = []
    for task in tasks:
        declared = declared_file_set(task)
        if root is not None:
            declared = [relative_to_root(path, root) for path in declared]
        resolved.append((str(task.get("task_id", "")), set(declared)))
    clashes: List[Tuple[str, str, List[str]]] = []
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            shared = resolved[i][1] & resolved[j][1]
            if shared:
                clashes.append((resolved[i][0], resolved[j][0], sorted(shared)))
    return clashes


# ---------------------------------------------------------------------------
# D3l — the append-only audit record
# ---------------------------------------------------------------------------

class AuditLog:
    """``{phase}/subagent_audit.jsonl``: one fsynced line per event, never a rewrite.

    What makes an unattended write run reviewable *after* the fact instead of
    merely trusted before it. Deliberately **not** a record of the prompt: the
    prompt is reduced to its sha256 and byte length, and the runner's environment
    is never touched, because prompts embed untrusted plan/diff text and an audit
    log that leaks a credential is a new vulnerability rather than a control.
    """

    #: argv elements whose *value* is redacted before logging, matched
    #: case-insensitively on the flag that precedes them. Belt-and-braces: no
    #: adapter puts a secret on argv, and if one ever does this catches it.
    SECRET_FLAG_HINTS = ("key", "token", "secret", "password", "credential", "auth")

    def __init__(self, phase_dir: str) -> None:
        self.path = str(Path(phase_dir) / AUDIT_FILENAME)

    # -- writing --------------------------------------------------------

    def _append(self, record: Dict[str, Any]) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        handle = os.open(self.path, flags, 0o600)
        try:
            with os.fdopen(handle, "a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:  # pragma: no cover - append must not mask a dispatch
            raise

    def dispatched(
        self,
        *,
        task_id: str,
        kind: str,
        attempt: int,
        runner: str,
        profile: str,
        cwd: Optional[str],
        sandbox_root: str,
        argv: Sequence[str],
        prompt: str,
        declared_files: Sequence[str],
        snapshot_dir: Optional[str] = None,
        snapshot_before: Optional[Dict[str, Any]] = None,
    ) -> None:
        """The pre-dispatch line. Written **before** the child starts."""
        self._append({
            "event": "dispatched",
            "timestamp": _now(),
            "task_id": task_id,
            "kind": kind,
            "attempt": attempt,
            "runner": runner,
            "profile": profile,
            "cwd": cwd,
            "sandbox_root": sandbox_root,
            "argv": redact_argv(argv, prompt),
            "prompt_sha256": _sha256_text(prompt),
            "prompt_bytes": len((prompt or "").encode("utf-8")),
            "declared_files": list(declared_files),
            "snapshot_dir": snapshot_dir,
            "snapshot_before": snapshot_before or {},
        })

    def completed(
        self,
        *,
        task_id: str,
        kind: str,
        attempt: int,
        runner: str,
        profile: str,
        status: str,
        write_outcome: Optional[str],
        exit_code: Optional[int],
        files_touched: Sequence[str],
        observed: Dict[str, Any],
        duration_seconds: float,
        reason: Optional[str] = None,
    ) -> None:
        """The post-classification line. The evidence behind D3g's verdict."""
        self._append({
            "event": "completed",
            "timestamp": _now(),
            "task_id": task_id,
            "kind": kind,
            "attempt": attempt,
            "runner": runner,
            "profile": profile,
            "status": status,
            "write_outcome": write_outcome,
            "exit_code": exit_code,
            "files_touched": list(files_touched),
            "observed": observed,
            "duration_seconds": round(duration_seconds, 3),
            "reason": reason,
        })

    def restored(self, *, task_id: str, report: Dict[str, str], force: bool) -> None:
        """``util restore-batch``'s own line — a rollback is an auditable event."""
        self._append({
            "event": "restored",
            "timestamp": _now(),
            "task_id": task_id,
            "force": bool(force),
            "result": dict(report),
        })

    # -- reading --------------------------------------------------------

    def records(self) -> List[Dict[str, Any]]:
        """Every line, in order. Corrupted lines are skipped, never guessed at."""
        try:
            text = Path(self.path).read_text(encoding="utf-8")
        except OSError:
            return []
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def completion_for(self, task_id: str) -> Optional[Dict[str, Any]]:
        """The last ``completed`` record for ``task_id``, if any."""
        found = None
        for record in self.records():
            if record.get("event") == "completed" and record.get("task_id") == task_id:
                found = record
        return found

    def touched_with_hashes(self, task_id: str) -> Dict[str, Optional[str]]:
        """``relative -> sha256_after`` from the completion record (restore input)."""
        record = self.completion_for(task_id)
        if not record:
            return {}
        observed = record.get("observed") or {}
        return {
            str(path): (info or {}).get("sha256_after")
            for path, info in observed.items()
            if str(path) in set(record.get("files_touched") or [])
        }

    def failed_task_ids(self) -> List[str]:
        """Task ids whose last completion was not a clean apply."""
        latest: Dict[str, Dict[str, Any]] = {}
        for record in self.records():
            if record.get("event") == "completed":
                latest[str(record.get("task_id"))] = record
        return sorted(
            task_id for task_id, record in latest.items()
            if record.get("status") != "ok"
            or record.get("write_outcome") in ("partial-application", "policy-violation")
        )


def redact_argv(argv: Sequence[str], prompt: str) -> List[str]:
    """The rendered argv with the prompt elided and any secret-shaped value cut.

    The prompt becomes ``<prompt sha256:… bytes:…>`` so the audit record proves
    *which* prompt ran without carrying its bytes.
    """
    digest = _sha256_text(prompt)
    length = len((prompt or "").encode("utf-8"))
    out: List[str] = []
    redact_next = False
    for arg in argv:
        if prompt and arg == prompt:
            out.append(f"<prompt sha256:{digest} bytes:{length}>")
            redact_next = False
            continue
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        lowered = arg.lower()
        if arg.startswith("-") and any(
            hint in lowered for hint in AuditLog.SECRET_FLAG_HINTS
        ):
            if "=" in arg:
                out.append(arg.split("=", 1)[0] + "=<redacted>")
                continue
            out.append(arg)
            redact_next = True
            continue
        out.append(arg)
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
