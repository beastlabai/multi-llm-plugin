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
  never trusted, byte comparison decides. It covers the declared set at
  ``take`` time and grows by :meth:`TaskSnapshot.adopt` for the paths the
  scheduler's post-run sweep observed *outside* that set — the only paths the
  post-run policy check can actually fail on, and therefore the ones a rollback
  point has to exist for.
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
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
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
    "redact_argv",
    "redact_text",
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
    """Drive-relative (``C:foo``), UNC (``\\\\host\\share``) or an ADS (``a.json:b``)."""
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    if re.match(r"^[A-Za-z]:[^\\/]", text):
        return True
    # An NTFS alternate data stream: *any* colon left once the drive prefix has
    # been sliced off, mirroring ``subagent_executor._is_unsafe_windows_path``.
    # The standard shape is ``name.ext:stream``, whose colon is preceded by a
    # letter — a rule that exempted those would flag nothing that matters (D3f).
    remainder = text[2:] if re.match(r"^[A-Za-z]:", text) else text
    return ":" in remainder


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
    """One snapshotted file: where it was, what it hashed to, where the copy is.

    ``declared`` separates the two ways an entry gets here: ``take`` records the
    batch's declared set, :meth:`TaskSnapshot.adopt` records a write the post-run
    sweep observed outside it. Both are restorable; only the first was ever
    promised, so the classification's prose must not confuse them.
    """

    relative: str
    absolute: str
    existed: bool
    sha256: Optional[str]
    size: int
    payload: Optional[str]
    declared: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.relative,
            "absolute": self.absolute,
            "existed": self.existed,
            "sha256_before": self.sha256,
            "size_before": self.size,
            "payload": self.payload,
            "declared": self.declared,
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
            # Absent in a v1 manifest, where every entry was a declared file.
            declared=bool(data.get("declared", True)),
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

    def adopt(
        self,
        relative: str,
        *,
        existed: bool,
        sha256: Optional[str] = None,
        size: int = 0,
        payload_source: Optional[str] = None,
        save: bool = True,
    ) -> Optional[SnapshotEntry]:
        """Record a rollback point for a path the batch touched but never declared.

        ``take`` covers the declared set, and that set is the one thing the
        post-run policy check can never fail on — it passed the byte-identical
        pre-dispatch check. The paths that *can* fail it are the ones a runner
        wrote outside what it declared, and a violation the executor can name but
        not undo is half a control. The scheduler's containment sweep retains
        those files' pre-run bytes in a content-addressed store; adopting one
        here puts it in the same manifest :meth:`changed`, :meth:`restore` and
        ``util restore-batch`` already read, so an undeclared write rolls back
        exactly like a declared one.

        Args:
            payload_source: The retained pre-run bytes. Required when
                ``existed`` — an entry claiming a rollback point it has no copy
                for is worse than an honest ``None``, which is what this returns
                instead.
            save: Off when a caller adopts a batch of paths and saves once; the
                manifest is rewritten whole, so per-path saves are quadratic.

        Returns:
            The entry, or ``None`` when no rollback point could be recorded.
        """
        resolved = contain_under_root(relative, self.root, what="observed write")
        key = relative_to_root(str(resolved), self.root)
        existing = self.entry_for(key)
        if existing is not None:
            return existing
        # `existed=False` needs no bytes at all: the rollback for a file the
        # runner created is deletion, which `restore` already does for an entry
        # with no payload.
        payload: Optional[str] = None
        if existed:
            if not payload_source or not Path(payload_source).is_file():
                return None
            payload = f"{len(self.entries):03d}-{_safe_component(key)}"
            payload_dir = Path(self.directory) / PAYLOAD_DIRNAME
            payload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(payload_source), str(payload_dir / payload))
        entry = SnapshotEntry(
            relative=key,
            absolute=str(resolved),
            existed=existed,
            sha256=sha256,
            size=size,
            payload=payload,
            declared=False,
        )
        self.entries.append(entry)
        if save:
            self.save()
        return entry

    def declared_entries(self) -> List[SnapshotEntry]:
        """Only the entries ``take`` recorded — what the batch said it would edit."""
        return [entry for entry in self.entries if entry.declared]

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
                "declared": entry.declared,
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

        Every destination is re-derived from the pinned root and revalidated
        component by component before anything is written (see
        :func:`_revalidated_target`). The manifest's ``absolute`` is a claim about
        the tree as it was *before* a write-capable runner had it, so following it
        now — which ``shutil.copy2`` onto the destination would do — is how a
        rollback ends up writing the saved bytes through a symlink the runner left
        behind, into a file outside the repository. The path *entry* is replaced or
        unlinked instead; neither operation follows a link.

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
            restoring = bool(entry.existed and entry.payload)
            try:
                target = _revalidated_target(
                    self.root, entry.relative, create_parents=restoring
                )
            except WriteSafetyError as exc:
                report[relative] = f"skipped: {exc.message}"
                continue
            except OSError as exc:
                report[relative] = f"skipped: {exc}"
                continue
            if target is None:
                # The delete leg, with a parent directory already gone.
                report[relative] = "deleted (already absent)"
                continue
            symlinked = _is_symlink(target)
            if expected is not None and not force:
                recorded = expected.get(relative, "__missing__")
                if recorded != "__missing__":
                    if symlinked:
                        report[relative] = (
                            "skipped: the path is a symlink now, not the file "
                            "that was snapshotted — pass --force to replace the "
                            "link itself (its target is never written through)"
                        )
                        continue
                    current = file_sha256(str(target))
                    if current != recorded:
                        report[relative] = (
                            "skipped: changed after the run (recorded "
                            f"{_short(recorded)}, now {_short(current)}) — pass "
                            "--force to overwrite anyway"
                        )
                        continue
            if restoring:
                source = Path(self.directory) / PAYLOAD_DIRNAME / entry.payload
                if not source.is_file():
                    report[relative] = "skipped: snapshot payload is missing"
                    continue
                try:
                    _replace_with_payload(source, target)
                except OSError as exc:
                    report[relative] = f"skipped: {exc}"
                    continue
                report[relative] = "restored"
            else:
                # Absent before the batch: the correct rollback is deletion.
                # `os.unlink` removes the entry itself, so a symlink left in the
                # file's place is removed rather than followed.
                try:
                    os.unlink(str(target))
                    report[relative] = "deleted"
                except FileNotFoundError:
                    report[relative] = "deleted (already absent)"
                except OSError as exc:
                    report[relative] = f"skipped: {exc}"
        return report


def _short(digest: Optional[str]) -> str:
    return (digest or "absent")[:12]


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(str(path)).st_mode)
    except OSError:
        return False


def _revalidated_target(
    root: str,
    relative: str,
    *,
    create_parents: bool,
) -> Optional[Path]:
    """Re-derive ``relative`` under ``root``, checking every component no-follow.

    Rollback runs *after* a write-capable runner had the working tree, so the
    snapshot's absolute path is a claim about the past: the file it names can have
    become a symlink into ``$HOME`` since, and so can any directory above it. Each
    component below the pinned root is therefore re-checked with ``os.lstat``
    (never ``stat``, which answers about a link's target, and never
    ``Path.resolve``, which follows one silently) and a symlinked or non-directory
    parent is refused. The path ``take`` recorded was canonicalized at snapshot
    time, so a symlink on this walk is always something that appeared afterwards.

    Args:
        create_parents: On for the restore leg, which may have to recreate a
            directory the batch removed — each level with ``os.mkdir``, which
            fails rather than silently accepting an existing symlink the way
            ``mkdir(parents=True, exist_ok=True)`` would. Off for the delete leg,
            which returns ``None`` for a missing parent: nothing left to delete.

    Raises:
        PathEscape: any component is a symlink, or the path is not a plain
            repo-relative one.
    """
    if not root:
        raise PathEscape(
            f"the snapshot manifest records no repo root, so {relative} cannot be "
            f"revalidated before writing"
        )
    resolved_root = Path(root).resolve()
    if not resolved_root.is_dir():
        raise PathEscape(
            f"the pinned repo root {resolved_root} is gone, so {relative} cannot "
            f"be revalidated before writing"
        )
    text = str(relative).replace("\\", "/")
    parts = [part for part in PurePosixPath(text).parts if part != "."]
    if (
        Path(relative).is_absolute()
        or PurePosixPath(text).is_absolute()
        or _looks_like_a_windows_escape(str(relative))
        or not parts
        or any(part == ".." for part in parts)
    ):
        raise PathEscape(
            f"{relative!r} is not a plain path under the pinned repo root "
            f"{resolved_root}"
        )
    current = resolved_root
    for part in parts[:-1]:
        current = current / part
        try:
            info = os.lstat(str(current))
        except FileNotFoundError:
            if not create_parents:
                return None
            os.mkdir(str(current))
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PathEscape(
                f"{current} is a symlink now, so rolling {relative} back through "
                f"it would leave the pinned repo root {resolved_root}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise PathEscape(
                f"{current} is not a directory, so {relative} cannot be restored "
                f"under the pinned repo root {resolved_root}"
            )
    return current / parts[-1]


def _replace_with_payload(source: Path, target: Path) -> None:
    """Write ``source``'s bytes as a new file and ``os.replace`` it over ``target``.

    Never ``shutil.copy2`` onto the destination: it opens the destination for
    writing and follows it, so a snapshotted regular file that has since become a
    symlink would have the saved bytes written *through* the link. ``os.replace``
    swaps the directory entry itself — atomic, and no-follow — and the temp file is
    created ``O_EXCL`` in the same revalidated directory, so it cannot land on an
    existing symlink and the rename cannot cross a filesystem.
    """
    tmp = target.with_name(f".{target.name}.restore.{os.getpid()}.tmp")
    handle = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(handle, "wb") as stream:
            with open(str(source), "rb") as payload:
                shutil.copyfileobj(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # What `copy2` preserved, minus the destination-following write.
        shutil.copystat(str(source), str(tmp))
        os.replace(str(tmp), str(target))
    except BaseException:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise


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


# ---------------------------------------------------------------------------
# Redaction over rendered text (§4.5)
# ---------------------------------------------------------------------------
#
# One boundary, four sinks: the success log (``llm_client._save_log``), the
# per-attempt failure captures (``failed/*.attemptN.json``), the batch report
# (``subagent_batch_report.json``) and the exit-3 stdout block that becomes
# ``orchestrator-run.log`` under ``--detach``. Provider stdout/stderr reached all
# four unfiltered; each growing its own filter is the shape this replaces.
#
# **This runs BEFORE any cap.** Both the 64 KiB/line rule and the last-20-lines
# rule cut by construction, and a cut applied first slices a secret into a
# remnant no pattern matches — a byte tail keeps a value whose key name is gone,
# and a line tail keeps a PEM body whose ``-----BEGIN`` header is gone. Nothing
# downstream can recover either.
#
# **Why not just reuse `.gitleaks.toml`.** gitleaks is a Go binary over a file
# tree at commit time; this runs in-process on every rendered line of every
# fan-out slot. The blast radii also run opposite ways — a commit-time false
# positive is a red PR, a runtime false negative is a live credential in
# ``log_*.txt`` — so this set is deliberately a strict *superset*, carrying PII
# and low-entropy delimiter shapes gitleaks must never flag. The one-way drift
# gate lives in ``tests/test_redaction.py``; this table is never transcribed into
# the TOML, which would make the scanner red against this repo's committed
# decoys on day one.

#: The key half of a delimited key/value secret. Underscore/dot/dash-joined
#: compounds around the keyword are deliberate: ``aws_secret_access_key`` is a
#: real shape that a word-anchored ``\bsecret\b`` never matches.
#: ``auth``/``authorization`` are excluded **on purpose** — P2 covers the header
#: form, and a bare ``auth:`` key is far too common in this repo's auth-plan
#: fixtures. That is the one entry on which this and ``SECRET_FLAG_HINTS``
#: differ, because ``auth`` is a reliable signal in a *flag name*.
_SECRET_KEY = (
    r"[A-Za-z0-9_.\-]*"
    r"(?:secret|passwd|password|token|api[_-]?key|apikey|credential)"
    r"[A-Za-z0-9_.\-]*"
)

#: An explicit ``:`` or ``=`` is mandatory, and the optional quotes are what make
#: ``"api_key": "…"`` match. **Never relax this to whitespace**: the measured
#: 114 false positives — "token generation", "password hashing", "Token refresh"
#: — all came from a whitespace-separated branch, and no real secret is emitted
#: without a delimiter.
_DELIM = r"[\"']?\s*[:=]\s*[\"']?"

#: The candidate value, qualified further by :func:`_looks_like_credential`.
_CAND = r"[^\s\"',;]{8,}"

#: Punctuation that says "expression", not "credential". Model output routinely
#: contains code, and code contains ``JWT_SECRET = os.environ.get(…)``.
_CODEY = set("(){}[]<>$\\!`|&*")

#: Values that are the keyword itself (``password=password``) or a placeholder.
_PLACEHOLDER_VALUES = {
    "password", "secret", "token", "apikey", "api_key", "credential", "passwd",
    "changeme", "example", "redacted", "required", "optional",
    "true", "false", "none", "null",
}


def _looks_like_credential(value: str) -> bool:
    """Whether a delimited value is plausibly a credential rather than code.

    Delimiter-anchoring alone is not sufficient — it still fires on
    ``password=password``, ``JWT_SECRET = os.environ.get(…)`` and
    ``password_hash=bcrypt.hashpw(…)`` in the e2e response fixtures. This is the
    other half of the precision story, and it is not optional.
    """
    if len(value) < 8 or (_CODEY & set(value)):
        return False
    if value.strip("'\"").lower() in _PLACEHOLDER_VALUES:
        return False
    return (
        any(c.isdigit() for c in value)
        or (any(c.isupper() for c in value) and any(c.islower() for c in value))
        or any(c in "-_./+=" for c in value)
    )


#: ``(name, compiled pattern, group to replace, predicate)``. The *value* is
#: replaced, never the whole line, so surrounding text stays diffable and a
#: redacted log is still a usable failure artifact.
#:
#: Order is load-bearing in one direction only: the multi-line and prefix shapes
#: run before the generic key/value branch, so a value they already replaced
#: reaches P1 as ``<redacted:…>`` — which ``_looks_like_credential`` rejects on
#: its angle brackets. That is also what makes :func:`redact_text` idempotent.
_REDACTIONS: List[Tuple[str, Any, int, Optional[Any]]] = [
    # P12 — multi-line, so it runs first: nothing else may chew its body.
    (
        "pem-private-key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
        ),
        0,
        None,
    ),
    # P13 — identity/PII (§5.2). The **quoted-JSON key form** rather than a
    # generic delimiter, because bare `shell = …` and `email = …` appear in
    # ordinary prose and source. A bare unkeyed email pattern is deliberately
    # not shipped: it would redact commit-message prose in container_turns/.
    (
        "identity-json",
        re.compile(
            r"[\"'](?:userEmail|email|terminalProgram|shell)[\"']\s*:\s*[\"']([^\"']{3,})[\"']"
        ),
        1,
        None,
    ),
    # P3–P10 — vendor prefixes, every one qualified by length and charset so a
    # short decoy is a non-match by construction rather than by exception.
    ("sk-family", re.compile(r"\bsk-(?:ant-|proj-|or-v1-|live-)?[A-Za-z0-9_\-]{16,}\b"), 0, None),
    ("slack", re.compile(r"\bxox[baprse]-[A-Za-z0-9-]{8,}\b"), 0, None),
    ("github", re.compile(r"\bgh[opusr]_[A-Za-z0-9]{16,}\b"), 0, None),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35,}"), 0, None),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0, None),
    ("gitlab-pat", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), 0, None),
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), 0, None),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
        0,
        None,
    ),
    # P11 — the password half only; the username survives so the URL stays
    # recognisable in a failure artifact.
    ("url-basic-auth", re.compile(r"\b[a-z][a-z0-9+.\-]*://([^/\s:@]+):([^/\s@]+)@"), 2, None),
    # P2 — the header form, and the one place `bearer`/`basic` are keywords.
    (
        "auth-scheme",
        re.compile(r"(?i)\b(?:authorization\s*:\s*)?(?:bearer|basic)\s+([A-Za-z0-9._\-+/=]{8,})"),
        1,
        None,
    ),
    # P1 — the generic delimited key/value branch, last and predicate-gated.
    (
        "secret-kv",
        re.compile(r"(?i)\b" + _SECRET_KEY + r"(?:" + _DELIM + r")(" + _CAND + r")"),
        1,
        _looks_like_credential,
    ),
]


def _replace_group(match: "re.Match", group: int, name: str) -> str:
    """The matched text with ``group`` swapped for ``<redacted:{name}>``."""
    start, end = match.span(group)
    if start < 0:  # an optional group that did not participate
        return match.group(0)
    base = match.start()
    return match.group(0)[: start - base] + f"<redacted:{name}>" + match.group(0)[end - base:]


def redact_text(text: str) -> str:
    """Credential- and identity-shaped values in ``text``, replaced in place.

    Applied to the **fully rendered** line — and always *before* any truncation
    cap, which is why L3 and L4 land together. The matched value is replaced with
    ``<redacted:{name}>``; the surrounding text is untouched.

    Out of scope, stated rather than implied: a high-entropy value with no
    keyword, no prefix and no delimiter (a bare 40-char blob in prose) cannot be
    caught by regex. That is accepted — keeping the credential out of the child's
    environment in the first place is the primary control, and this is the
    backstop.
    """
    if not text:
        return text or ""
    out = text
    for name, pattern, group, predicate in _REDACTIONS:
        def _sub(match, _name=name, _group=group, _predicate=predicate):
            if _predicate is not None and not _predicate(match.group(_group)):
                return match.group(0)
            return _replace_group(match, _group, _name)

        out = pattern.sub(_sub, out)
    return out


def redact_argv(argv: Sequence[str], prompt: str) -> List[str]:
    """The rendered argv with the prompt elided and any secret-shaped value cut.

    The prompt becomes ``<prompt sha256:… bytes:…>`` so the audit record proves
    *which* prompt ran without carrying its bytes.

    Two passes, and the flag-hint one is **kept, not replaced** (§4.5.5): knowing
    a value's *position* from ``--api-key`` is strictly stronger than any pattern
    and catches the bare high-entropy blob no regex can. Every argument that
    survives that pass then goes through :func:`redact_text`, which closes the
    inverse hole — a secret in an argument that carries no hinted flag.
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
        out.append(redact_text(arg))
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
