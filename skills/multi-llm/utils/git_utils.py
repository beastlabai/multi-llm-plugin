"""Git utilities for the multi-llm skill."""

import functools
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Union,
)

class GitError(Exception):
    """Raised when a git operation fails."""
    pass


#: The ``-c`` overrides every git command the plugin spawns carries (root
#: grant, 2026-10-05). A repo-wide batch on the plugin's Landlock jail can
#: write ``.git/config`` and ``.git/hooks/`` — the user decided against
#: backing the metadata up around each batch — so the plugin's own git calls
#: that run after it (the sweep's ``status``, ``ls-files`` and
#: ``hash-object``, intent-to-add's ``add -N``, review-code's diffs) must not
#: execute anything the batch planted: ``core.fsmonitor`` is a hook ``status``
#: and ``diff`` run on every call, and ``core.hooksPath`` is where ``add -N``
#: finds ``post-index-change``. Cheap and detective: the user's own git still
#: honours the planted config, which :func:`git_metadata_fingerprint` and the
#: scheduler's ``[GIT_CONFIG_CHANGED]`` line exist to say.
GIT_HARDENING_ARGS: Tuple[str, ...] = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    # `log.showSignature` makes `git log` / `git show <commit>` run
    # `gpg.program` on a signed commit (review fixes 2026-10-05).
    "-c", "log.showSignature=false",
    # Review fixes 2026-10-05: a planted `diff.noprefix` / `diff.mnemonicPrefix`
    # moves the `a/`, `b/` prefixes `_parse_unified_diff` keys hunks by, so
    # `capture_diff_hunks` returned no hunks and review-code reviewed nothing;
    # `diff.relative` from a subdirectory cwd prints nothing at all for a path
    # named from the root (measured, git 2.43). `diff.srcPrefix`/`dstPrefix`
    # are git 2.45+ and silently ignored before that.
    "-c", "diff.noprefix=false",
    "-c", "diff.mnemonicPrefix=false",
    "-c", "diff.srcPrefix=a/",
    "-c", "diff.dstPrefix=b/",
    "-c", "diff.relative=false",
)
#: The diff options for the same reason: ``diff.external`` and a driver's
#: ``diff.<x>.command`` / ``diff.<x>.textconv`` (bound by a plantable
#: ``.gitattributes``) are executables ``git diff`` runs by config.
GIT_DIFF_HARDENING_ARGS: Tuple[str, ...] = ("--no-ext-diff", "--no-textconv")

#: The git subcommands that take :data:`GIT_DIFF_HARDENING_ARGS`.
_DIFF_SUBCOMMANDS = frozenset({"diff", "diff-index", "diff-tree", "diff-files", "log", "show"})

#: The git global options that take a value, skipped by :func:`git_argv`'s
#: walk to the subcommand.
_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree"})

#: The config keys a batch can plant that make git **execute** a program on
#: the subcommands the plugin runs, beyond the static set above: a clean /
#: smudge / process filter bound to a path by any attributes file (the
#: worktree's ``.gitattributes``, ``.git/info/attributes``, or a
#: ``core.attributesFile`` the same config names). Measured on git 2.43
#: (review fixes 2026-10-05): ``status --porcelain``, ``diff``, ``diff
#: --numstat``, ``add`` and ``hash-object`` (without ``--no-filters``) all run
#: a clean filter on a modified matching file; ``--attr-source=<empty tree>``
#: and ``required=false`` do not stop it, an empty ``-c filter.<n>.clean=``
#: does. Only a driver the **repository's own** config defines (scope
#: ``local`` or ``worktree``, includes resolved) is neutralized: that is what a
#: batch can write under the root grant, while a global ``filter.lfs.*`` is
#: the user's own tool, bound by the same attributes and left in force so the
#: plugin's view of an LFS-tracked file stays the user's.
_PLANTABLE_DRIVER_KEYS = r"^(filter\..+\.(clean|smudge|process|required)|core\.attributesfile)$"
_DRIVER_SCOPES_A_BATCH_CAN_WRITE = frozenset({"local", "worktree"})


def _hardened_argv(*args: str) -> List[str]:
    """``git`` plus the static :data:`GIT_HARDENING_ARGS` plus ``args``: the
    one place a ``["git", …]`` list is spelled. :func:`git_argv` adds the
    per-repository driver overrides on top; the probe that reads them uses
    this builder directly, which is what keeps the two from recursing."""
    return ["git", *GIT_HARDENING_ARGS, *args]


def _parse_scoped_config(stdout: str) -> List[Tuple[str, str, str]]:
    """``(scope, key, value)`` triples from ``git config -z --show-scope
    --get-regexp`` output: ``scope NUL key LF value NUL`` per match, a
    valueless key carrying no LF."""
    tokens = stdout.split("\0")
    out: List[Tuple[str, str, str]] = []
    for index in range(0, len(tokens) - 1, 2):
        scope, rest = tokens[index], tokens[index + 1]
        key, _, value = rest.partition("\n")
        if key:
            out.append((scope, key, value))
    return out


def _cwd_from_global_options(args: Sequence[str]) -> Optional[str]:
    """The ``-C <dir>`` a caller put ahead of the subcommand, if any."""
    index = 0
    while index < len(args) and args[index].startswith("-"):
        if args[index] == "-C" and index + 1 < len(args):
            return args[index + 1]
        index += 2 if args[index] in _GIT_GLOBAL_OPTIONS_WITH_VALUE else 1
    return None


def planted_driver_overrides(cwd: Optional[str] = None) -> List[str]:
    """The ``-c`` overrides that neutralize every filter driver, and any
    ``core.attributesFile``, the repository at ``cwd`` defines in its own
    config (see :data:`_PLANTABLE_DRIVER_KEYS`). Read without executing
    anything — one ``git config`` probe, itself hardened and never a
    :func:`git_argv` call — and empty outside a repository, when git is
    absent, or when the probe fails: a probe may not fail a git call. Read
    on every call rather than cached, because the batch that plants a driver
    runs between the plugin's git calls (~1 ms measured)."""
    try:
        proc = subprocess.Popen(
            _hardened_argv(
                "config", "-z", "--show-scope", "--get-regexp", _PLANTABLE_DRIVER_KEYS,
            ),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            **({"cwd": cwd} if cwd else {}),
        )
        try:
            stdout, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if not isinstance(stdout, bytes) or proc.returncode not in (0, 1):
        return []
    names: Set[str] = set()
    attributes_file = False
    for scope, key, _value in _parse_scoped_config(
        stdout.decode("utf-8", errors="replace")
    ):
        if scope not in _DRIVER_SCOPES_A_BATCH_CAN_WRITE:
            continue
        if key == "core.attributesfile":
            attributes_file = True
        elif key.startswith("filter."):
            name = key[len("filter."):].rsplit(".", 1)[0]
            if name:
                names.add(name)
    overrides: List[str] = []
    for name in sorted(names):
        overrides += [
            "-c", f"filter.{name}.clean=",
            "-c", f"filter.{name}.smudge=",
            "-c", f"filter.{name}.process=",
            # `required=true` beside an emptied driver is `fatal: clean filter
            # failed` (measured): the planted key must not break the call.
            "-c", f"filter.{name}.required=false",
        ]
    if attributes_file:
        overrides += ["-c", f"core.attributesFile={os.devnull}"]
    return overrides


def git_argv(*args: str, cwd: Optional[str] = None) -> List[str]:
    """``git`` plus :data:`GIT_HARDENING_ARGS` plus the repository's
    :func:`planted_driver_overrides` plus ``args`` — the one way the plugin
    spells a git command (``tests/test_git_utils.py`` fails any runtime git
    list literal outside :func:`_hardened_argv`). ``cwd`` is where the
    command will run — pass the same value the spawn gets, since the driver
    overrides are read from that repository; a ``-C <dir>`` global option
    counts, else the process's cwd. A diff-producing subcommand also gets
    :data:`GIT_DIFF_HARDENING_ARGS` right after its name; a ``git show
    <ref>:<path>`` blob read accepts them harmlessly."""
    rest = list(args)
    argv = _hardened_argv(*planted_driver_overrides(cwd or _cwd_from_global_options(rest)))
    # Global options (`--literal-pathspecs`, `-C <dir>`) precede the subcommand.
    index = 0
    while index < len(rest) and rest[index].startswith("-"):
        index += 2 if rest[index] in _GIT_GLOBAL_OPTIONS_WITH_VALUE else 1
    if index < len(rest) and rest[index] in _DIFF_SUBCOMMANDS:
        rest[index + 1:index + 1] = list(GIT_DIFF_HARDENING_ARGS)
    return argv + rest


def _run_git(
    *args: str, check: bool = True, cwd: Optional[str] = None
) -> Tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode).

    ``cwd`` is where git runs; ``None`` is the process's cwd (and passes no
    ``cwd`` at all, so every caller that names none runs exactly as before).
    """
    # Decode git output as UTF-8 with errors="replace": lossy for non-UTF-8
    # bytes but guarantees a total decode, the standard choice for tool/LLM
    # text output. Rejected alternatives: surrogateescape (lone surrogates
    # raise UnicodeEncodeError on downstream UTF-8 re-encode) and
    # backslashreplace (noisier in model-visible output).
    where: Dict[str, str] = {"cwd": cwd} if cwd is not None else {}
    result = subprocess.run(
        git_argv(*args, cwd=cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **where,
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout, result.stderr, result.returncode

def get_project_root(from_path: str) -> Optional[str]:
    """Detect project root via git rev-parse --show-toplevel.

    Args:
        from_path: A file path within the project (used to determine
                   which directory to run git from).

    Returns:
        Absolute path to the project root, or None if not in a git repo.
    """
    probe_dir = os.path.dirname(from_path) or "."
    result = subprocess.run(
        git_argv("rev-parse", "--show-toplevel", cwd=probe_dir),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=probe_dir,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


#: Environment variables that point git at another repository or working
#: tree than the one the plan is in. :func:`pin_repo_root` refuses while any
#: is set: the plugin's git calls and a repo-wide jail would then disagree
#: about which tree they act on.
GIT_ENV_REDIRECTS: Tuple[str, ...] = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR")


def find_git_entry_root(start: str) -> Optional[str]:
    """The nearest directory at or above ``start`` holding a ``.git`` entry
    (a directory, or the gitfile of a linked worktree or submodule),
    ``realpath``'d; ``None`` when there is none. Pure path walking — git
    itself is not asked, so no configuration can move the answer."""
    current = os.path.realpath(start)
    while True:
        if os.path.lexists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def pin_repo_root(start_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """``(root, problem)``: the repository root pinned independently of git's
    configuration, and why it cannot be trusted when it cannot.

    The root is :func:`find_git_entry_root`'s walk, never ``git rev-parse
    --show-toplevel`` alone: a repo-wide batch on the root grant can write
    ``[core] worktree = <dir>`` into ``.git/config`` (or replace ``.git`` with
    a gitfile naming a git dir whose config says so), after which every
    later phase and ``--resume`` re-pinned the root from git's answer and the
    next repo-wide jail's one rule landed on the planted directory (review
    fixes 2026-10-05). git is still asked, and its answer must **equal** the
    walked root; a ``core.worktree`` that agrees (a submodule sets one
    legitimately) is fine, one that disagrees is named as the cause. While a
    :data:`GIT_ENV_REDIRECTS` variable is set nothing is pinned at all.
    ``(None, None)`` outside any repository.
    """
    # Spelled out rather than iterated: the codex package's environment
    # census resolves literal reads and module constants only.
    redirected = [
        name for name, value in (
            ("GIT_DIR", os.environ.get("GIT_DIR")),
            ("GIT_WORK_TREE", os.environ.get("GIT_WORK_TREE")),
            ("GIT_COMMON_DIR", os.environ.get("GIT_COMMON_DIR")),
        ) if value
    ]
    if redirected:
        return None, (
            f"{', '.join(redirected)} is set in the environment, so git would act "
            f"on another repository or working tree than the one the plan is in; "
            f"unset it for this run"
        )
    root = find_git_entry_root(start_dir)
    if root is None:
        return None, None
    worktree_out, _, worktree_rc = _run_git(
        "config", "--get", "core.worktree", check=False, cwd=root,
    )
    configured = worktree_out.strip() if worktree_rc == 0 else ""
    top_out, _, top_rc = _run_git("rev-parse", "--show-toplevel", check=False, cwd=root)
    answered = os.path.realpath(top_out.strip()) if top_rc == 0 and top_out.strip() else None
    if answered == root or (answered is None and not configured):
        # Agreement — or nothing to compare: git cannot answer (a `.git`
        # that is not a repository, dubious ownership) and no `core.worktree`
        # is configured, so the walked root stands; a redirect needs the
        # configuration git would have honoured, and none is there.
        return root, None
    cause = (
        f"core.worktree = {configured!r} in its configuration redirects the "
        f"working tree — a repo-wide batch can plant that key in .git/config, so "
        f"inspect .git/config and remove it (git config --unset core.worktree)"
        if configured else
        f"the .git entry or the configuration it points at redirects the "
        f"working tree (a gitfile naming another git directory, say); inspect "
        f"{os.path.join(root, '.git')}"
    )
    where = (
        f"git answers {answered} as the top level" if answered
        else "git cannot confirm the top level"
    )
    return root, (
        f"{where} of the tree whose .git entry is at {root}: {cause}"
    )


def get_project_root_from_dir(directory: Optional[str] = None) -> Optional[str]:
    """Detect the git project root by running git from a *directory* directly.

    Unlike :func:`get_project_root`, which takes a *file* path and uses its
    ``dirname`` to choose where to run git, this takes a directory and runs
    ``git rev-parse --show-toplevel`` there with no path manipulation. Pass a
    directory (e.g. CWD or a target-repo dir) directly without the brittle
    synthetic-path trick (``get_project_root(os.path.join(cwd, "_"))``) that
    relies on ``dirname`` stripping a fake segment.

    Args:
        directory: Directory to run git discovery from. Defaults to CWD.

    Returns:
        Absolute path to the project root, or None if not in a git repo.
    """
    result = subprocess.run(
        git_argv("rev-parse", "--show-toplevel", cwd=directory or "."),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=directory or "."
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _porcelain_destination(path_field: str) -> str:
    """Return the on-disk path of a ``git status --porcelain`` path field.

    Rename and copy entries render as ``<old> -> <new>``; the path that exists
    in the working tree is the destination, so the source half is dropped. Git
    quotes either side when it contains a space, which is what makes the
    separator unambiguous: an unquoted destination contains no space of its own,
    and a quoted one is the trailing ``"..."``.
    """
    if path_field.endswith('"'):
        index = path_field.rfind(' -> "')
        if index != -1:
            return path_field[index + len(" -> "):]
        return path_field
    index = path_field.rfind(" -> ")
    if index != -1:
        return path_field[index + len(" -> "):]
    return path_field


_C_ESCAPES = {
    "a": "\a", "b": "\b", "f": "\f", "n": "\n",
    "r": "\r", "t": "\t", "v": "\v", "\\": "\\", '"': '"',
}


def _unescape_c_quoted(body: str) -> str:
    """Undo git's C-style escaping inside a quoted path field.

    Git escapes the control characters and the quote/backslash by name and
    every other byte it does not consider printable as a three-digit octal
    ``\\303`` — one per UTF-8 *byte*, so the octals are collected as bytes and
    decoded together at the end (``errors="replace"``, matching `_run_git`).
    """
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\" or i + 1 >= len(body):
            out += ch.encode("utf-8")
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in _C_ESCAPES:
            out += _C_ESCAPES[nxt].encode("utf-8")
            i += 2
        elif nxt.isdigit() and len(body) >= i + 4:
            try:
                out.append(int(body[i + 1:i + 4], 8))
                i += 4
            except ValueError:
                out += ch.encode("utf-8")
                i += 1
        else:
            # Not an escape git produces; keep the backslash verbatim rather
            # than inventing a path that does not exist.
            out += ch.encode("utf-8")
            i += 1
    return out.decode("utf-8", errors="replace")


def _unquote_porcelain_path(path: str) -> str:
    """Return the real on-disk path behind a quoted ``git status`` path field.

    Git quotes a path that contains a space, a quote or a non-printable byte
    and C-escapes what is inside. Both halves have to come off: a census entry
    left as ``"sp ace.txt"`` (quotes included) matches no baseline entry, lands
    in ``tracked_files`` verbatim, and `review-code` then diffs a path that
    does not exist (F6, 2026-09-16). An unquoted path is returned untouched —
    a literal backslash in a bare porcelain path is not an escape.
    """
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        return _unescape_c_quoted(path[1:-1])
    return path


def subtract_baseline(paths: Iterable[str], baseline: Iterable[str]) -> List[str]:
    """Return the ``paths`` a dirty-tree ``baseline`` does *not* already cover.

    A baseline entry ending in ``/`` is a *directory*, and it covers everything
    under it. That is not a convenience: ``git status --porcelain`` reports an
    untracked directory as a single ``?? some_dir/`` line, so every baseline
    ever recorded names directories alongside files, and a plain set difference
    subtracts none of them — no file path equals ``"plans/"``. Measured
    2026-09-16: a run whose baseline was the four directories ``.claude/``,
    ``.multi-llm/``, ``plans/``, ``saved/`` subtracted nothing and tracked all
    1 511 untracked files in the repo as its own changes.

    Order is preserved so a caller's list stays in the order it discovered
    things in.
    """
    exact = set()
    prefixes = []
    for entry in baseline:
        if not entry:
            continue
        exact.add(entry)
        bare = _unquote_porcelain_path(entry)
        if bare.endswith("/"):
            # The directory itself, with and without the separator, plus
            # everything beneath it.
            exact.add(bare)
            exact.add(bare.rstrip("/"))
            prefixes.append(bare)
        else:
            exact.add(bare)
    kept = []
    for path in paths:
        if path in exact:
            continue
        bare_path = _unquote_porcelain_path(path)
        if bare_path in exact:
            continue
        if any(bare_path.startswith(prefix) for prefix in prefixes):
            continue
        kept.append(path)
    return kept


def baseline_covers(path: str, baseline: Iterable[str]) -> bool:
    """True when ``baseline`` already accounts for ``path`` (see above)."""
    return not subtract_baseline([path], baseline)


def get_modified_files() -> List[str]:
    """Get list of modified and untracked files.

    Deliberately plain ``--porcelain``, so an untracked *directory* stays one
    ``?? some_dir/`` line instead of expanding to every file beneath it. A
    file-granular census cannot tell an edit inside an untracked tree from the
    rest of that tree anyway — an untracked file has no "modified" state in git
    — so the expansion bought nothing and cost a great deal: it made the census
    disagree with the compact baselines already stored in ``state.json`` (see
    ``subtract_baseline``, which is what reconciles the two), and in a repo with
    1 511 untracked files it wrote all of them into ``tracked_files``. The
    authoritative signal for what a run changed is the write executor's verified
    ``files_touched``, threaded to ``finalize_tracking`` as ``--touched``.
    """
    stdout, _, _ = _run_git("status", "--porcelain")
    files = []
    # NEVER strip leading whitespace off this output: a worktree-only change is
    # reported as `" M path"`, so `stdout.strip()` would eat the first line's
    # status column and `line[3:]` would then cut a character off the filename.
    # Only the trailing newline is dropped, by filtering short/blank lines.
    for line in stdout.split("\n"):
        # A real entry is `XY <path>`: two status columns, a space, a path.
        if len(line) < 4:
            continue
        status = line[:2]
        if not status.strip():
            continue
        # Unquote here and not only in `subtract_baseline`: the census is what
        # gets STORED (in `tracked_files`, and handed to every consumer that
        # opens the file), so a path with a space has to leave this function as
        # the path it really is (F6, 2026-09-16).
        filename = _unquote_porcelain_path(_porcelain_destination(line[3:]))
        if filename:
            files.append(filename)
    return files


def list_untracked_files(directory: str) -> List[str]:
    """List the untracked files git reports under ``directory``.

    The counterpart of the compact census above: `get_modified_files` collapses
    an untracked directory to one ``dir/`` entry, which is what lets a stored
    baseline subtract a whole pre-existing tree cheaply — but a directory the
    run itself CREATED survives that subtraction, and a directory path is not
    something a diff or a file read can act on. Expanding just that entry
    recovers the files without re-censusing the whole repo.

    ``-z`` so the paths come back raw, with no C-quoting to undo. Best-effort:
    a git failure (or a directory that has since gone) returns nothing and the
    caller keeps whatever it had.
    """
    stdout, _, rc = _run_git(
        "ls-files", "--others", "--exclude-standard", "-z", "--", directory,
        check=False,
    )
    if rc != 0:
        return []
    return [path for path in stdout.split("\0") if path]

def paths_in_commit(
    ref: str, paths: Iterable[str], *, cwd: Optional[str] = None,
) -> Optional[Set[str]]:
    """The subset of repo-root-relative *paths* present in *ref*'s tree.

    ``None`` when *ref* does not resolve to a commit (an unborn ``HEAD``, a
    pruned ``head_before_implement``), so the caller can tell "absent from
    the tree" from "no tree to ask". One ``git cat-file --batch-check`` over
    ``<sha>:<path>`` lines: exact paths, never pathspec patterns (a ``*`` in a
    filename is a filename). A path holding a newline cannot be asked and is
    reported absent.
    """
    candidates = [path for path in paths if path and "\n" not in path]
    stdout, _, rc = _run_git(
        "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}",
        check=False, cwd=cwd,
    )
    sha = stdout.strip()
    if rc != 0 or not sha:
        return None
    if not candidates:
        return set()
    where: Dict[str, str] = {"cwd": cwd} if cwd is not None else {}
    result = subprocess.run(
        git_argv("cat-file", "--batch-check=%(objecttype)", cwd=cwd),
        input="".join(f"{sha}:{path}\n" for path in candidates),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        **where,
    )
    if result.returncode != 0:
        return None
    answers = result.stdout.splitlines()
    if len(answers) != len(candidates):
        return None
    return {
        path for path, answer in zip(candidates, answers)
        if not answer.endswith(" missing")
    }


def ignored_paths(paths: Iterable[str], *, cwd: Optional[str] = None) -> Set[str]:
    """The subset of ``paths`` git ignores, as the strings given.

    ``git check-ignore`` without ``--no-index``, so a *tracked* file that a
    pattern happens to match is not reported: only untracked-and-ignored paths
    (transcripts, build output, caches) come back. Best-effort: outside a
    repository, or when git fails, nothing is ignored, because dropping a path
    on a guess is worse than keeping it. ``-z`` both ways so no path is C-quoted.

    A path outside the repository is never asked about: ``check-ignore`` exits
    128 ("fatal: … is outside repository") for the whole batch on one such
    path, which used to answer "nothing is ignored" for every path beside it
    (adversarial review, 2026-10-02; ``finalize_tracking --touched`` can carry
    one). Such a path is not ignored — it is not the repository's at all.
    """
    candidates = [path for path in paths if path]
    if not candidates:
        return set()
    where: Dict[str, str] = {"cwd": cwd} if cwd is not None else {}
    stdout, _, rc = _run_git("rev-parse", "--show-toplevel", check=False, cwd=cwd)
    if rc != 0:
        return set()
    toplevel = os.path.realpath(stdout.strip())
    base = os.path.realpath(cwd) if cwd is not None else os.path.realpath(os.getcwd())
    inside = [
        path for path in candidates
        if _under_toplevel(os.path.join(base, path), toplevel)
    ]
    if not inside:
        return set()
    try:
        result = subprocess.run(
            git_argv("check-ignore", "-z", "--stdin", cwd=cwd),
            input="\0".join(inside) + "\0",
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            **where,
        )
    except OSError:
        return set()
    if result.returncode not in (0, 1):  # 128: not a repository, bad path
        return set()
    return {path for path in result.stdout.split("\0") if path}


def _under_toplevel(path: str, toplevel: str) -> bool:
    """Whether ``path`` (absolute or cwd-joined) lies at or under ``toplevel``.

    Lexical after ``normpath`` on the path and ``realpath`` on the root: the
    path need not exist (a deleted file is still a candidate) and a symlink
    inside the repo pointing out is ``check-ignore``'s own question to answer.
    """
    resolved = os.path.normpath(path)
    return resolved == toplevel or resolved.startswith(toplevel.rstrip(os.sep) + os.sep)


def get_staged_files() -> List[str]:
    """Get list of currently staged files."""
    stdout, _, _ = _run_git("diff", "--cached", "--name-only")
    # Like porcelain, --name-only C-quotes non-ASCII and control characters.
    # Keep the same decoded paths as get_modified_files so their union cannot
    # contain a second, nonexistent quoted spelling of a staged file.
    return [_unquote_porcelain_path(f) for f in stdout.split("\n") if f]

def stage_files(files: List[str]) -> None:
    """Stage specific files."""
    if files:
        _run_git("add", "--", *files)

def unstage_files(files: List[str]) -> None:
    """Unstage specific files."""
    if files:
        _run_git("reset", "HEAD", "--", *files)


@contextmanager
def intent_to_add_untracked(
    file_paths: List[str], *, cwd: Optional[str] = None
) -> Iterator[List[str]]:
    """Make untracked files visible to `git diff` without touching the real index.

    `git diff` (all variants -- `git diff`, `git diff HEAD`, `git diff A..B`) silently
    ignores fully-untracked files. For code review flows that rely on git diff to
    surface implementation changes, this means new files never show up. Marking them
    with `git add -N` creates an empty index entry so diffs treat them as full-file
    additions without staging any content.

    The mark is made in a **private copy of the index**, exported as
    ``GIT_INDEX_FILE`` for the duration of the block. Every git call in this
    process, and every reviewer subprocess that inherits the environment, reads
    the copy and sees the files as additions; the user's own ``.git/index`` is
    never written, so there is nothing to undo, an interrupted run cannot leave
    ``intent-to-add`` entries behind, and a ``git status`` the user runs in
    another terminal meanwhile shows a clean tree (2026-10-02: the old
    ``git add -N`` / ``git reset HEAD`` pair mutated the real index for the whole
    review, flagged as a side effect of ``review-code``). On exit the variable is
    restored (or removed) and the copy deleted. Files that were already in the
    index (staged, or already intent-to-add by the user) are never touched.

    Args:
        file_paths: Candidate files to consider. Files not currently untracked are
            skipped. A file listed here but missing from the working tree is also
            skipped.
        cwd: Where git runs, and so what ``file_paths`` are relative to. The
            review scope is repository-root-relative, so a caller holding one
            passes the repository root: from a subdirectory, the process's cwd
            would read every path relative to that subdirectory and mark
            nothing.

    Yields:
        The list of files this context manager actually marked (may be empty).
    """
    touched: List[str] = []
    private_index: Optional[str] = None
    had_index_var = "GIT_INDEX_FILE" in os.environ
    previous_index_var = os.environ.get("GIT_INDEX_FILE")
    if file_paths:
        try:
            stdout, _, rc = _run_git(
                "ls-files", "--others", "--exclude-standard", "--", *file_paths,
                check=False, cwd=cwd,
            )
            if rc == 0:
                touched = [line for line in stdout.splitlines() if line.strip()]
        except Exception as exc:
            print(
                f"Warning: could not list untracked files for intent-to-add: {exc}",
                file=sys.stderr,
            )
            touched = []

        if touched:
            try:
                private_index = _private_index_copy(cwd)
                os.environ["GIT_INDEX_FILE"] = private_index
                _run_git("add", "-N", "--", *touched, cwd=cwd)
                print(
                    f"Marked {len(touched)} untracked file(s) as intent-to-add "
                    f"in a private index copy so review can see their contents"
                )
            except (GitError, OSError) as exc:
                _restore_index_var(had_index_var, previous_index_var)
                _discard_private_index(private_index)
                private_index = None
                print(
                    f"Warning: could not mark untracked files intent-to-add "
                    f"({exc}); untracked files may be invisible to reviewers",
                    file=sys.stderr,
                )
                touched = []
    try:
        yield touched
    finally:
        if private_index is not None:
            _restore_index_var(had_index_var, previous_index_var)
            _discard_private_index(private_index)


def _private_index_copy(cwd: Optional[str]) -> str:
    """Copy the index git would use into a file of its own; return its path.

    Source: ``$GIT_INDEX_FILE`` if the caller already redirected git, else the
    worktree's own index (``git rev-parse --git-path index``, which is right for
    linked worktrees). The copy sits beside it inside the git directory, so a
    reviewer sandbox that can read the repository can read it; a git directory
    that refuses the write falls back to the system temp dir. A repository with
    no index yet (nothing ever staged) copies nothing: git creates the file.
    """
    source = os.environ.get("GIT_INDEX_FILE")
    if not source:
        stdout, _, _ = _run_git("rev-parse", "--git-path", "index", cwd=cwd)
        source = stdout.strip()
    # Either value can be relative, and both are relative to where git runs
    # (`cwd`), not to this process's own directory: a git hook exports
    # `GIT_INDEX_FILE=.git/index`, and resolving that against the process cwd
    # pointed the copy at a directory that did not exist, so `git add -N`
    # failed on its lock file and no untracked file reached the reviewers
    # (adversarial review, 2026-10-02).
    if not os.path.isabs(source):
        source = os.path.join(cwd or os.getcwd(), source)
    source = os.path.abspath(source)
    _sweep_dead_index_copies(os.path.dirname(source))
    name = f"multi-llm-index.{os.getpid()}.{time.monotonic_ns()}"
    target = os.path.join(os.path.dirname(source), name)
    try:
        if os.path.exists(source):
            _copy_index_keeping_its_time(source, target)
    except OSError:
        import tempfile

        descriptor, target = tempfile.mkstemp(prefix="multi-llm-index.")
        os.close(descriptor)
        if os.path.exists(source):
            _copy_index_keeping_its_time(source, target)
        else:
            os.unlink(target)
    return target


def _copy_index_keeping_its_time(source: str, target: str) -> None:
    """Copy the index at *source* to *target*, with *source*'s timestamps.

    git trusts an entry's cached stat data (size, mtime, inode) to say a
    tracked file is unchanged, except when the entry is not older than the
    index file itself: such a "racily clean" entry may have been edited in the
    same second it was cached, so git compares its contents instead. A copy
    stamped with the time it was made is newer than every entry, which turns
    that check off: an edit of the same size made within the cached entry's
    second (review-code's fixture rewrites ``return 1`` as ``return 2`` at
    once) matched its stale stat data and dropped out of every diff the review
    ran on the copy (commit review of 5025463..fb0634f, 2026-10-08). The copy
    is read from one open file, so the time it carries is that file's even
    if git replaces the index meanwhile. A copy that fails halfway is removed.
    """
    try:
        with open(source, "rb") as reader:
            stamp = os.fstat(reader.fileno())
            with open(target, "wb") as writer:
                shutil.copyfileobj(reader, writer)
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    except OSError:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise


#: The private index copies, as :func:`_private_index_copy` names them:
#: ``multi-llm-index.<pid>.<monotonic ns>`` plus git's own ``.lock`` sibling.
_INDEX_COPY_NAME = re.compile(r"^multi-llm-index\.(\d+)\.\d+(\.lock)?$")


def _sweep_dead_index_copies(git_dir: str) -> None:
    """Remove the index copies a killed run left behind in *git_dir*.

    The ``finally`` in :func:`intent_to_add_untracked` cannot run under
    SIGKILL, so a hard-killed review leaves ``multi-llm-index.<pid>.<ns>`` (and
    its ``.lock``) in the git directory forever: harmless to git, which never
    reads them, but they accumulate (adversarial review, 2026-10-02). A copy
    is swept only when the pid in its name is not alive; this process's own
    copies (a nested block) and another live run's are left alone. Best
    effort: a directory that cannot be listed, or a file that cannot be
    removed, changes nothing about the run.
    """
    try:
        from .process_control import process_alive
    except ImportError:  # pragma: no cover - direct-script import shim
        from utils.process_control import process_alive
    try:
        names = os.listdir(git_dir)
    except OSError:
        return
    for name in names:
        match = _INDEX_COPY_NAME.match(name)
        if not match:
            continue
        pid = int(match.group(1))
        if pid == os.getpid() or process_alive(pid):
            continue
        try:
            os.unlink(os.path.join(git_dir, name))
        except OSError:
            pass


def _restore_index_var(had: bool, previous: Optional[str]) -> None:
    if had and previous is not None:
        os.environ["GIT_INDEX_FILE"] = previous
    else:
        os.environ.pop("GIT_INDEX_FILE", None)


def _discard_private_index(path: Optional[str]) -> None:
    if not path:
        return
    for leftover in (path, f"{path}.lock"):
        try:
            os.unlink(leftover)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(
                f"Warning: could not remove the private index copy {leftover}: {exc}",
                file=sys.stderr,
            )


def get_staged_diff() -> str:
    """Get diff of all staged changes."""
    stdout, _, _ = _run_git("diff", "--cached")
    return stdout

def get_file_diff(file_path: str, base_ref: Optional[str] = None) -> str:
    """Get diff for a specific file."""
    if base_ref:
        stdout, _, _ = _run_git("diff", base_ref, "--", file_path)
    else:
        stdout, _, _ = _run_git("diff", "--", file_path)
    return stdout

def get_current_head() -> str:
    """Get current HEAD commit hash."""
    stdout, _, _ = _run_git("rev-parse", "HEAD")
    return stdout.strip()

def get_branch_name() -> Optional[str]:
    """Get current branch name, or None if detached HEAD."""
    stdout, _, code = _run_git("symbolic-ref", "--short", "HEAD", check=False)
    if code == 0:
        return stdout.strip()
    return None

def is_clean_working_tree() -> bool:
    """Check if working tree is clean (no uncommitted changes)."""
    _, _, code = _run_git("diff", "--quiet", check=False)
    if code != 0:
        return False
    _, _, code = _run_git("diff", "--cached", "--quiet", check=False)
    return code == 0

def get_diff_since_ref(base_ref: str) -> str:
    """Get diff of all changes since a reference commit."""
    stdout, _, _ = _run_git("diff", base_ref)
    return stdout

def get_files_changed_since_ref(base_ref: str) -> List[str]:
    """Get list of files changed since a reference commit."""
    stdout, _, _ = _run_git("diff", "--name-only", base_ref)
    # Same rule as get_staged_files: one unquoted path per line, so the only
    # thing safe to drop is the line separator.
    return [f for f in stdout.split("\n") if f]


def validate_git_ref(ref: str) -> str:
    """Validate a git ref using git rev-parse --verify.

    If the ref is invalid, falls back to HEAD~1. If HEAD~1 is also invalid
    (or the original ref was already HEAD~1), returns an empty string.

    Args:
        ref: The git ref to validate (e.g., a commit hash, branch name, HEAD~1).

    Returns:
        The validated ref if valid, 'HEAD~1' as fallback, or empty string
        if both the ref and HEAD~1 are invalid.
    """
    if not ref:
        return ""

    try:
        subprocess.run(
            git_argv("rev-parse", "--verify", ref),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return ref
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"Warning: git ref '{ref}' is invalid, falling back to HEAD~1",
            file=sys.stderr,
        )

    # If the ref was already HEAD~1, don't redundantly retry it
    if ref == "HEAD~1":
        print(
            "Warning: HEAD~1 is not a valid git ref (shallow clone or initial commit?), "
            "using empty base_ref",
            file=sys.stderr,
        )
        return ""

    # Fallback to HEAD~1
    try:
        subprocess.run(
            git_argv("rev-parse", "--verify", "HEAD~1"),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return "HEAD~1"
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Warning: HEAD~1 is also not a valid git ref (shallow clone or initial commit?), "
            "using empty base_ref",
            file=sys.stderr,
        )
        return ""


# ---------------------------------------------------------------------------
# The repository metadata a repo-wide batch can now write (root grant,
# 2026-10-05): a fingerprint, never a copy
# ---------------------------------------------------------------------------

#: The name a changed-metadata line carries.
GIT_CONFIG_CHANGED_MARKER = "[GIT_CONFIG_CHANGED]"


def git_common_dir(repo_root: str) -> Optional[str]:
    """The repository's common git directory (``.git``, or the main
    repository's for a linked worktree), absolute; ``None`` when git cannot
    say. ``config`` and ``hooks/`` live there."""
    stdout, _, rc = _run_git("rev-parse", "--git-common-dir", check=False, cwd=repo_root)
    if rc != 0 or not stdout.strip():
        return None
    path = stdout.strip()
    if not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    return os.path.realpath(path)


def _file_sha256_or_none(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def git_config_include_targets(repo_root: str, common_dir: str) -> List[str]:
    """The files the repository's own config pulls in through ``include.path``
    / ``includeIf.<cond>.path`` (scope ``local`` or ``worktree``), absolute, in
    config order; empty when git cannot say. A relative target is resolved
    the way git resolves it, against the including file's directory — the
    git directory for ``config`` — and ``~`` is expanded."""
    try:
        proc = subprocess.run(
            git_argv("config", "-z", "--show-scope", "--get-regexp", r"^include(if\..*)?\.path$",
                     cwd=repo_root),
            capture_output=True, cwd=repo_root, timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if proc.returncode not in (0, 1) or not isinstance(proc.stdout, bytes):
        return []
    targets: List[str] = []
    for scope, _key, value in _parse_scoped_config(proc.stdout.decode("utf-8", errors="replace")):
        if scope not in _DRIVER_SCOPES_A_BATCH_CAN_WRITE or not value:
            continue
        path = os.path.expanduser(value)
        if not os.path.isabs(path):
            path = os.path.join(common_dir, path)
        targets.append(os.path.normpath(path))
    return targets


def git_metadata_fingerprint(repo_root: str) -> Optional[Dict[str, Any]]:
    """A fingerprint of what a batch could plant for the user's next git
    command to run: the sha256 of ``<git-dir>/config``, of
    ``<git-dir>/config.worktree``, of ``<git-dir>/info/attributes`` (binds a
    filter or diff driver to paths) and ``<git-dir>/info/exclude`` (hides
    paths from ``status``), of every file the config includes
    (:func:`git_config_include_targets`), and a listing of
    ``<git-dir>/hooks/`` (name → size, mode, mtime_ns; a symlink's
    destination). No bytes are kept; ``None`` when there is no git directory.
    Taken before a repo-wide dispatch and compared after it by
    :func:`describe_git_metadata_change` — detective, never a rollback."""
    common = git_common_dir(repo_root)
    if common is None:
        return None
    config_sha = _file_sha256_or_none(os.path.join(common, "config"))
    files: Dict[str, Optional[str]] = {
        ".git/config.worktree": _file_sha256_or_none(os.path.join(common, "config.worktree")),
        ".git/info/attributes": _file_sha256_or_none(os.path.join(common, "info", "attributes")),
        ".git/info/exclude": _file_sha256_or_none(os.path.join(common, "info", "exclude")),
    }
    includes: Dict[str, Optional[str]] = {}
    for target in git_config_include_targets(repo_root, common):
        includes[target] = _file_sha256_or_none(target)
    hooks: Dict[str, Tuple[Any, ...]] = {}
    hooks_dir = os.path.join(common, "hooks")
    try:
        names = sorted(os.listdir(hooks_dir))
    except OSError:
        names = []
    for name in names:
        path = os.path.join(hooks_dir, name)
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if os.path.islink(path):
            try:
                hooks[name] = ("link", os.readlink(path))
            except OSError:
                hooks[name] = ("link", "unreadable")
        else:
            hooks[name] = ("file", info.st_size, info.st_mode, info.st_mtime_ns)
    return {
        "git_dir": common, "config_sha256": config_sha, "hooks": hooks,
        "files": files, "includes": includes,
    }


def _describe_hash_moves(
    old: Mapping[str, Any], new: Mapping[str, Any], label: Callable[[str], str],
) -> List[str]:
    parts: List[str] = []
    for name in sorted(set(old) | set(new)):
        was, now = old.get(name), new.get(name)
        if was == now:
            continue
        if was is None:
            parts.append(f"{label(name)} added")
        elif now is None:
            parts.append(f"{label(name)} removed")
        else:
            parts.append(f"{label(name)} changed")
    return parts


def describe_git_metadata_change(
    before: Optional[Mapping[str, Any]], after: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """What moved between two fingerprints, as ``.git``-relative names
    (``.git/config changed; .git/hooks/pre-commit added; .git/info/attributes
    added; include.path target /x/evil.inc changed``), or ``None`` when
    nothing did or either side is missing."""
    if not before or not after:
        return None
    parts: List[str] = []
    if before.get("config_sha256") != after.get("config_sha256"):
        parts.append(".git/config changed")
    parts += _describe_hash_moves(
        before.get("files") or {}, after.get("files") or {}, lambda name: name,
    )
    parts += _describe_hash_moves(
        before.get("includes") or {}, after.get("includes") or {},
        lambda path: f"include.path target {path}",
    )
    old = before.get("hooks") or {}
    new = after.get("hooks") or {}
    for name in sorted(set(old) | set(new)):
        if name not in old:
            parts.append(f".git/hooks/{name} added")
        elif name not in new:
            parts.append(f".git/hooks/{name} removed")
        elif old[name] != new[name]:
            parts.append(f".git/hooks/{name} changed")
    return "; ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Diff hunk capture and file content capture utilities
# ---------------------------------------------------------------------------

# Budget constants for embedded context data
_MAX_LINES_PER_FILE = 200
_MAX_TOTAL_BYTES = 500 * 1024  # 500KB
_GIT_TIMEOUT_SECONDS = 10

# Rough amortized width of one diff line (prefix char + content + newline),
# used only to convert a byte budget into a line-count cap — see
# _per_file_line_cap.
_ASSUMED_BYTES_PER_DIFF_LINE = 80


def _per_file_line_cap(file_count: int) -> int:
    """Per-file line budget for ``_parse_unified_diff``, scaled by how many
    files share ``_MAX_TOTAL_BYTES``.

    ``_MAX_LINES_PER_FILE`` used to be a flat cap applied to every file
    regardless of how many were being reviewed, so a single ~1,400-line file
    (the common case for a whole-file review) got the same 200-line cap as
    one file among fifty — 86% of it silently dropped — while the byte
    budget that actually bounds total output sat almost unused. Scale the
    cap with the remaining budget split across the files being diffed, so a
    lone file is allowed most of the budget and a large fan-out still gets a
    floor of ``_MAX_LINES_PER_FILE``.
    """
    if file_count <= 0:
        return _MAX_LINES_PER_FILE
    per_file_bytes = _MAX_TOTAL_BYTES / file_count
    return max(_MAX_LINES_PER_FILE, int(per_file_bytes / _ASSUMED_BYTES_PER_DIFF_LINE))


def _is_binary_file(path: str) -> bool:
    """Heuristic check for binary file content.

    Reads the first 8KB and looks for null bytes.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except (OSError, IOError):
        return False


def _parse_unified_diff(diff_text: str, max_lines: int = _MAX_LINES_PER_FILE) -> Dict[str, Any]:
    """Parse unified diff text for a single file into structured hunk data.

    Args:
        diff_text: unified diff for one file.
        max_lines: cap on captured add/remove/context lines (see
            ``_per_file_line_cap`` for how callers scale this).

    Returns a dict with keys:
      - hunks: list of hunk objects (header + lines)
      - old_path / new_path: for renamed files
      - binary: True if the diff indicates a binary file
      - deleted: True if the file was deleted
      - truncated_at: int if lines were capped
      - total_lines: int, present only alongside truncated_at — the full
        count of add/remove/context lines the diff actually has, so a
        reader can say "showing first N of <total_lines>" instead of "?"

    Each hunk contains:
      - header: the @@ line
      - lines: list of {type, old_line, new_line, content}
    """
    lines = diff_text.split("\n")
    result: Dict[str, Any] = {
        "hunks": [],
        "old_path": None,
        "new_path": None,
        "binary": False,
        "deleted": False,
    }

    total_content_lines = 0
    total_lines = 0
    truncated = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect binary file marker
        if line.startswith("Binary files") or line.startswith("GIT binary patch"):
            result["binary"] = True
            return result

        # Detect rename
        if line.startswith("rename from "):
            result["old_path"] = line[len("rename from "):]
        elif line.startswith("rename to "):
            result["new_path"] = line[len("rename to "):]

        # Detect old/new paths from --- / +++ headers
        if line.startswith("--- a/"):
            result["old_path"] = result["old_path"] or line[6:]
        elif line.startswith("--- /dev/null"):
            # New file – no old path
            pass
        elif line.startswith("+++ b/"):
            result["new_path"] = result["new_path"] or line[6:]
        elif line.startswith("+++ /dev/null"):
            result["deleted"] = True

        # Parse hunk header
        hunk_match = re.match(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line
        )
        if hunk_match:
            old_start = int(hunk_match.group(1))
            new_start = int(hunk_match.group(3))
            hunk_header_text = line

            hunk: Dict[str, Any] = {
                "header": hunk_header_text,
                "lines": [],
            }

            old_line = old_start
            new_line = new_start

            i += 1
            while i < len(lines):
                hline = lines[i]

                # Stop at next hunk, next file header, or end
                if (
                    hline.startswith("@@ ")
                    or hline.startswith("diff --git")
                    or hline.startswith("--- a/")
                    or hline.startswith("--- /dev/null")
                ):
                    break

                # Count every real content line up front (even once
                # truncated) so a truncated result can still report how
                # many lines it dropped.
                if hline.startswith("+") or hline.startswith("-") or hline.startswith(" "):
                    total_lines += 1

                if truncated:
                    i += 1
                    continue

                if total_content_lines >= max_lines:
                    truncated = True
                    result["truncated_at"] = max_lines
                    i += 1
                    continue

                if hline.startswith("+"):
                    hunk["lines"].append({
                        "type": "add",
                        "old_line": None,
                        "new_line": new_line,
                        "content": hline[1:],
                    })
                    new_line += 1
                    total_content_lines += 1
                elif hline.startswith("-"):
                    hunk["lines"].append({
                        "type": "remove",
                        "old_line": old_line,
                        "new_line": None,
                        "content": hline[1:],
                    })
                    old_line += 1
                    total_content_lines += 1
                elif hline.startswith(" "):
                    hunk["lines"].append({
                        "type": "context",
                        "old_line": old_line,
                        "new_line": new_line,
                        "content": hline[1:],
                    })
                    old_line += 1
                    new_line += 1
                    total_content_lines += 1
                elif hline.startswith("\\ No newline at end of file"):
                    # Marker line – skip, don't count toward budget
                    pass
                else:
                    # Unknown line in hunk (e.g., empty trailing line) – skip
                    pass

                i += 1

            result["hunks"].append(hunk)
            continue  # Don't increment i again – already positioned

        i += 1

    if truncated:
        result["total_lines"] = total_lines

    return result


def _split_diff_by_file(raw_diff: str) -> Dict[str, str]:
    """Split a multi-file unified diff into per-file chunks.

    Returns a dict mapping file paths to their diff text.
    The key is the new-side path (b/ path) from the diff header.
    """
    file_diffs: Dict[str, str] = {}
    current_path: Optional[str] = None
    current_lines: List[str] = []

    for line in raw_diff.split("\n"):
        if line.startswith("diff --git"):
            # Save previous file's diff
            if current_path is not None:
                file_diffs[current_path] = "\n".join(current_lines)

            # Extract path from "diff --git a/path b/path"
            parts = line.split(" b/", 1)
            current_path = parts[1] if len(parts) > 1 else None
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last file
    if current_path is not None:
        file_diffs[current_path] = "\n".join(current_lines)

    return file_diffs


def capture_diff_hunks(
    base_ref: str,
    file_paths: List[str],
) -> Dict[str, Any]:
    """Capture structured diff hunks for specified files against a base ref.

    Runs ``git diff base_ref -- <files>`` and parses the unified diff output
    into structured per-file data consumed by the PR-style report template.

    Args:
        base_ref: Git ref to diff against (commit SHA, branch name, HEAD~N).
        file_paths: List of file paths to capture diffs for.  Only files in
            this list are processed (scoped to suggestion-referenced files).

    Returns:
        A dict with:
          - Per file-path key: structured hunk data (see ``_parse_unified_diff``).
          - ``_notices``: list of notice strings (e.g., uncommitted changes).
          - ``_error``: error string when git is unavailable.

    Edge cases handled:
      - Binary files → ``{binary: true}``
      - Renamed files → ``old_path`` and ``new_path`` included
      - Deleted files → ``deleted: true``, content retrieved via ``git show``
      - Invalid base_ref → empty dict with warning
      - Git not available → empty dict with notice
      - 10-second timeout enforced
      - Per-file line cap, scaled by file count (floor 200 lines; see
        ``_per_file_line_cap``) so a single-file review is not held to the
        same cap as a large fan-out
      - 500KB total budget
    """
    if not file_paths:
        return {}

    result: Dict[str, Any] = {}
    notices: List[str] = []

    # --- Check for uncommitted changes ---
    try:
        status_result = subprocess.run(
            git_argv("status", "--porcelain"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        if status_result.returncode == 0 and status_result.stdout.strip():
            notices.append("Diff includes uncommitted changes")
    except FileNotFoundError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (not installed or not in PATH)",
            file=sys.stderr,
        )
        return result
    except subprocess.TimeoutExpired:
        notices.append("git status timed out")
    except OSError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (OS error)",
            file=sys.stderr,
        )
        return result

    # --- Run git diff ---
    try:
        diff_result = subprocess.run(
            git_argv("diff", base_ref, "--", *file_paths),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (not installed or not in PATH)",
            file=sys.stderr,
        )
        return result
    except subprocess.TimeoutExpired:
        notices.append("diff capture timed out")
        result["_notices"] = notices
        return result
    except OSError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (OS error)",
            file=sys.stderr,
        )
        return result

    if diff_result.returncode != 0:
        # Invalid base_ref or other git error
        print(
            f"Warning: git diff failed for base_ref '{base_ref}': "
            f"{diff_result.stderr.strip()}",
            file=sys.stderr,
        )
        result["_notices"] = notices
        return result

    raw_diff = diff_result.stdout

    # --- Split and parse per-file diffs ---
    per_file_diffs = _split_diff_by_file(raw_diff)
    total_bytes = 0
    # Scaled once per call: a review of one file should not be held to the
    # same flat 200-line cap as a review of fifty (see _per_file_line_cap).
    line_cap = _per_file_line_cap(len(file_paths))

    start_time = time.monotonic()

    for fpath in file_paths:
        # Timeout check
        if time.monotonic() - start_time > _GIT_TIMEOUT_SECONDS:
            notices.append("diff capture timed out")
            break

        file_diff_text = per_file_diffs.get(fpath)
        if file_diff_text is None:
            # File may not have changes in the diff – skip silently
            continue

        # Budget check
        diff_bytes = len(file_diff_text.encode("utf-8", errors="replace"))
        if total_bytes + diff_bytes > _MAX_TOTAL_BYTES:
            result[fpath] = {"truncated": True}
            notices.append(
                f"Size budget exceeded at {fpath} – remaining files omitted"
            )
            break

        total_bytes += diff_bytes

        parsed = _parse_unified_diff(file_diff_text, max_lines=line_cap)
        result[fpath] = parsed

    # --- Handle deleted files: retrieve last-known content ---
    for fpath in file_paths:
        if fpath in result and isinstance(result[fpath], dict) and result[fpath].get("deleted"):
            try:
                show_result = subprocess.run(
                    git_argv("show", f"{base_ref}:{fpath}"),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=_GIT_TIMEOUT_SECONDS,
                )
                if show_result.returncode == 0:
                    result[fpath]["last_known_content"] = show_result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass  # Best effort – content unavailable

    if notices:
        result["_notices"] = notices

    return result


def capture_file_snapshots(
    file_paths: List[str],
) -> Dict[str, List[str]]:
    """Capture file content for multiple files at a point in time.

    Call this at the START of a review phase (before model invocations begin)
    to snapshot file content.  The returned dict can be passed as
    ``file_snapshots`` to ``generate_html_report()`` and ultimately to
    ``capture_file_context()``, ensuring the report reflects file state at
    review start rather than at report-generation time.

    Args:
        file_paths: List of file paths (absolute or relative to cwd) to
            snapshot.

    Returns:
        Dict mapping each readable, non-binary file path to a list of its
        lines (with trailing newlines stripped).  Files that do not exist,
        are binary, or cannot be read are silently omitted.
    """
    snapshots: Dict[str, List[str]] = {}
    for fpath in file_paths:
        if not fpath or not os.path.isfile(fpath):
            continue
        if _is_binary_file(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                snapshots[fpath] = [
                    ln.rstrip("\n").rstrip("\r") for ln in f.readlines()
                ]
        except (OSError, IOError):
            continue
    return snapshots


def capture_file_context(
    file_path: str,
    line_range: List[int],
    context_lines: int = 5,
    file_snapshots: Optional[Dict[str, List[str]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Read lines around a line range from a file.

    Returns an array of ``{line_number: int, content: str}`` objects covering
    ``context_lines`` before and after the specified ``line_range``.

    When ``file_snapshots`` is provided and contains ``file_path``, the
    pre-captured snapshot is used instead of reading the file from disk.
    This allows callers to capture file content at the START of a review
    phase (via ``capture_file_snapshots()``) and avoid stale data if files
    are modified between review start and report generation.

    When no snapshot is available, falls back to reading the file from disk
    at call time.  For accurate context when files may have been modified
    after review start, prefer providing ``file_snapshots`` or using
    ``capture_diff_hunks`` with a ``base_ref`` (which uses git history).

    Args:
        file_path: Path to the file on disk (absolute or relative to cwd).
        line_range: Two-element list ``[start_line, end_line]`` (1-indexed,
            inclusive).
        context_lines: Number of extra lines to include before ``start_line``
            and after ``end_line``.  Defaults to 5.
        file_snapshots: Optional dict mapping file paths to pre-captured
            line lists (as returned by ``capture_file_snapshots()``).  When
            the file is found in this dict, its snapshot is used instead of
            reading from disk.

    Returns:
        List of ``{line_number, content}`` dicts, or ``None`` if the file
        does not exist or is binary.
    """
    if not file_path or not line_range:
        return None

    # Try snapshot first
    if file_snapshots and file_path in file_snapshots:
        all_lines = file_snapshots[file_path]
    else:
        # Fall back to reading from disk
        if not os.path.isfile(file_path):
            return None

        # Check for binary content
        if _is_binary_file(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = [
                    ln.rstrip("\n").rstrip("\r") for ln in f.readlines()
                ]
        except (OSError, IOError):
            return None

    if not all_lines:
        return None

    start_line = line_range[0]
    end_line = line_range[1] if len(line_range) > 1 else start_line

    # Clamp to valid range
    context_start = max(1, start_line - context_lines)
    context_end = end_line + context_lines

    # Clamp end to file length
    context_end = min(context_end, len(all_lines))

    result = []
    for i in range(context_start - 1, context_end):
        if i < 0 or i >= len(all_lines):
            continue
        result.append({
            "line_number": i + 1,
            "content": all_lines[i],
        })

    return result if result else None


# ---------------------------------------------------------------------------
# Review scope: the one classifier every consumer of the scope goes through
# ---------------------------------------------------------------------------
#
# A code review's scope is grouped and ordered in more than one place: the file
# list each reviewer's prompt carries, and the scope manifest plus per-file
# diffs written beside it (plan: review-run-scope-salvage-and-failure-visibility
# §2, D2.1/D2.2). Every one of them asks `classify_review_paths`, and nothing
# else holds a copy of the rules, so the prompt and the manifest can never put
# one path in two different groups.

REVIEW_GROUP_SOURCE = "source"
REVIEW_GROUP_TESTS = "tests"
REVIEW_GROUP_FIXTURES = "fixtures"
REVIEW_GROUP_GENERATED = "generated"

#: The groups in review order. The prompt's file list is cut at a character
#: budget, so whatever sorts last is what a cut drops, and hand-written source
#: is what a reviewer needs first. In plain path order `…/tests/…` sorts ahead
#: of `…/utils/…`: measured on a 1 040-path scope (880 of them
#: `tests/fixtures/opencode/**`), a populated list cut at 5 000 characters would
#: have kept the fixtures and dropped every `utils/` source file.
REVIEW_GROUP_ORDER: Tuple[str, ...] = (
    REVIEW_GROUP_SOURCE,
    REVIEW_GROUP_TESTS,
    REVIEW_GROUP_FIXTURES,
    REVIEW_GROUP_GENERATED,
)

#: The two configurable rules, as the ``code_review.collapse_paths`` block of
#: ``providers.yaml`` spells them. The shipped defaults below reproduce the
#: built-in fixture rule exactly (a ``tests/fixtures/`` or ``tests/golden/``
#: segment at any depth), so an unconfigured repository classifies as it always
#: did. This is the one copy of the defaults: the config accessor
#: (``provider_registry.get_collapse_paths``) falls back to it, and a test pins
#: the shipped ``providers.yaml`` block to it. Treat it as read-only.
COLLAPSE_PATH_KEYS: Tuple[str, ...] = (REVIEW_GROUP_GENERATED, REVIEW_GROUP_FIXTURES)
DEFAULT_COLLAPSE_PATHS: Mapping[str, Tuple[str, ...]] = {
    REVIEW_GROUP_GENERATED: (),
    REVIEW_GROUP_FIXTURES: ("**/tests/fixtures/**", "**/tests/golden/**"),
}

#: Directory components that make a path a test. Whole components only, so
#: ``contests/x.py`` and ``latest/x.py`` stay source.
_TEST_DIR_SEGMENTS = frozenset({"tests", "test"})
#: File-name markers that make a path a test, looked for in the file name only
#: (never in a directory name): ``test_x.py``, ``x_test.go``, ``x.test.ts``,
#: ``x.spec.js``.
_TEST_NAME_PREFIX = "test_"
_TEST_NAME_MARKERS: Tuple[str, ...] = ("_test.", ".test.", ".spec.")


@functools.lru_cache(maxsize=512)
def compile_path_glob(pattern: str) -> Pattern[str]:
    """Compile one ``collapse_paths`` glob into a regex over a repo-relative path.

    ``fnmatch`` does not fit: its ``*`` crosses ``/`` and it has no ``**``, so
    the shipped ``**/tests/fixtures/**`` would miss a top-level
    ``tests/fixtures/x.json`` (the leading ``**/`` would demand a separator
    before ``tests``). The rules here are the gitignore ones a user writing
    such a pattern already expects:

    * ``**/`` matches zero or more whole directories, a trailing ``/**``
      everything beneath a directory, and any other ``**`` any run of
      characters;
    * ``*`` and ``?`` never cross ``/``; ``[...]`` is a character class and
      ``[!...]`` its negation;
    * a pattern with no ``/`` in it matches the file name at any depth, so
      ``*.lock`` collapses ``poetry.lock`` and ``web/yarn.lock`` alike.

    Raises ``re.error`` for a class the regex engine rejects (``[z-a]``); the
    config loader compiles every configured pattern at load time so that
    surfaces as a config error, never mid-review.
    """
    pat = pattern.strip().replace("\\", "/")
    while pat.startswith("./"):
        pat = pat[2:]
    pat = pat.lstrip("/")
    if "/" not in pat:
        pat = "**/" + pat
    out: List[str] = []
    index, length = 0, len(pat)
    while index < length:
        if pat.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
            continue
        if pat.startswith("/**", index) and index + 3 == length:
            out.append("/.*")
            index += 3
            continue
        if pat.startswith("**", index):
            out.append(".*")
            index += 2
            continue
        char = pat[index]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            close = pat.find("]", index + 2)
            if close == -1:
                out.append(re.escape(char))
            else:
                body = pat[index + 1:close]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                index = close + 1
                continue
        else:
            out.append(re.escape(char))
        index += 1
    return re.compile("".join(out), re.DOTALL)


def _normalized_review_path(path: str) -> str:
    """A path as the group rules read it: POSIX separators, no leading ``./``."""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _matches_any_glob(path: str, patterns: Sequence[str]) -> bool:
    return any(compile_path_glob(pattern).fullmatch(path) for pattern in patterns)


def _is_test_path(path: str) -> bool:
    """D2.1 rule 3: a ``tests/``/``test/`` directory component, or a test file name."""
    parts = [part for part in path.split("/") if part]
    if not parts:
        return False
    directories, name = parts[:-1], parts[-1]
    if any(part in _TEST_DIR_SEGMENTS for part in directories):
        return True
    if name.startswith(_TEST_NAME_PREFIX):
        return True
    return any(marker in name for marker in _TEST_NAME_MARKERS)


def _resolved_collapse_globs(
    collapse_paths: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, Tuple[str, ...]]:
    """Both glob lists, each falling back to its shipped default when absent.

    Per key, like ``get_write_paths``: a caller that passes only ``generated``
    keeps the default fixture rule rather than silently losing it. An explicit
    empty list is honoured — that is how a repository switches a rule off.
    """
    configured: Mapping[str, Sequence[str]] = collapse_paths or {}
    resolved: Dict[str, Tuple[str, ...]] = {}
    for key in COLLAPSE_PATH_KEYS:
        patterns = configured.get(key)
        if patterns is None:
            resolved[key] = tuple(DEFAULT_COLLAPSE_PATHS[key])
        else:
            resolved[key] = tuple(str(p) for p in patterns if str(p).strip())
    return resolved


def _linguist_generated(paths: Sequence[str], cwd: Optional[str] = None) -> Set[str]:
    """The paths carrying the ``linguist-generated`` attribute, in one git call.

    ``--stdin -z`` rather than one argument per path: a large scope (1 040 paths
    on the run that motivated this) must not hit the argv limit, and ``-z``
    hands every path back byte-for-byte as given, with no quoting to undo.
    ``linguist-generated`` and ``linguist-generated=true`` both count (git
    reports them as ``set`` and ``true``); ``-linguist-generated`` and ``=false``
    do not.

    A failed call — outside a repository, or no ``git`` at all — marks nothing
    generated and prints one warning. It never raises: the attribute is one of
    four rules, and the other three still classify every path.
    """
    if not paths:
        return set()
    try:
        result = subprocess.run(
            git_argv("check-attr", "--stdin", "-z", "linguist-generated", cwd=cwd),
            input="\0".join(paths) + "\0",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
    except (OSError, ValueError) as exc:
        print(
            f"Warning: git check-attr linguist-generated could not run ({exc}); "
            f"no review path is classified generated by attribute — the other "
            f"rules still apply",
            file=sys.stderr,
        )
        return set()
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        print(
            f"Warning: git check-attr linguist-generated failed "
            f"({detail[0] if detail else f'exit {result.returncode}'}); no review "
            f"path is classified generated by attribute — the other rules still "
            f"apply",
            file=sys.stderr,
        )
        return set()
    fields = result.stdout.split("\0")
    generated: Set[str] = set()
    for index in range(0, len(fields) - 2, 3):
        path, _attribute, info = fields[index:index + 3]
        if info.strip().lower() in ("set", "true"):
            generated.add(path)
    return generated


def classify_review_paths(
    paths: Iterable[str],
    collapse_paths: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    cwd: Optional[str] = None,
) -> Dict[str, str]:
    """Put every review path in exactly one group; the first matching rule wins.

    1. *generated* — the path has the ``linguist-generated`` attribute (one
       batched ``git check-attr`` call, see :func:`_linguist_generated`), or it
       matches a ``collapse_paths["generated"]`` glob;
    2. *fixtures* — it matches a ``collapse_paths["fixtures"]`` glob (shipped:
       any ``tests/fixtures/`` or ``tests/golden/`` segment);
    3. *tests* — a ``tests/`` or ``test/`` directory component at any depth, or
       a file name starting ``test_`` or containing ``_test.``, ``.test.`` or
       ``.spec.``;
    4. *source* — everything else.

    So generated beats fixtures beats tests beats source: the codex mirror's
    ``utils/*.py`` copies are generated, not source, and a ``tests/fixtures/``
    file is a fixture, never a test.

    ``collapse_paths`` is the loaded ``code_review.collapse_paths`` block
    (``provider_registry.get_collapse_paths``) passed in by the caller, which
    keeps this free of config reads; ``None`` — or a key left out — means the
    shipped default for that rule. ``cwd`` is where git runs; the paths are
    the root-relative ones ``git diff --name-only`` prints, so a caller passes
    the repository root. The default, the process's cwd, is right only when
    that cwd IS the root: from a subdirectory git reads each path relative to
    it, and ``check-attr`` answers for files that do not exist.

    Returns ``{path: group}`` keyed by each distinct, non-empty input path as
    given; the order is the input's.
    """
    unique = [path for path in dict.fromkeys(paths) if path]
    globs = _resolved_collapse_globs(collapse_paths)
    by_attribute = _linguist_generated(unique, cwd=cwd)
    groups: Dict[str, str] = {}
    for path in unique:
        normalized = _normalized_review_path(path)
        if path in by_attribute or _matches_any_glob(
            normalized, globs[REVIEW_GROUP_GENERATED]
        ):
            groups[path] = REVIEW_GROUP_GENERATED
        elif _matches_any_glob(normalized, globs[REVIEW_GROUP_FIXTURES]):
            groups[path] = REVIEW_GROUP_FIXTURES
        elif _is_test_path(normalized):
            groups[path] = REVIEW_GROUP_TESTS
        else:
            groups[path] = REVIEW_GROUP_SOURCE
    return groups


def order_review_paths(
    paths: Iterable[str],
    collapse_paths: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    cwd: Optional[str] = None,
) -> List[str]:
    """The review order: source, tests, fixtures, generated; by path within each.

    The order is decided here, after resolution, and never taken from the
    resolver: its tracked-files branch de-duplicates through a ``set`` (hash
    order, different on every run) and its git fallback returns path order.
    Sorting within each group makes the result a function of the *set* of
    paths, so every permutation of the same input — and so every run over the
    same scope — renders the same list. Duplicates and empty strings are
    dropped.
    """
    return _review_order(classify_review_paths(paths, collapse_paths, cwd=cwd))


def _review_order(groups: Mapping[str, str]) -> List[str]:
    """The paths of a :func:`classify_review_paths` result, in review order.

    Split out so :func:`write_review_diff` can order and group from ONE
    classification (one ``git check-attr``) instead of classifying twice.
    """
    rank = {group: index for index, group in enumerate(REVIEW_GROUP_ORDER)}
    return sorted(groups, key=lambda path: (rank[groups[path]], path))


def count_review_groups(groups: Mapping[str, str]) -> Dict[str, int]:
    """``{group: count}`` in :data:`REVIEW_GROUP_ORDER`, every group present."""
    counts = {group: 0 for group in REVIEW_GROUP_ORDER}
    for group in groups.values():
        counts[group] = counts.get(group, 0) + 1
    return counts


def worktree_file_stats(
    repo_root: str, *, exclude_prefixes: Iterable[str] = ()
) -> Optional[Dict[str, Tuple[int, int]]]:
    """``path -> (size, mtime_ns)`` for the files git sees in the worktree:
    tracked plus untracked-not-ignored, repo-relative with ``/``.

    The map :func:`worktree_fingerprint` digests, and what the native leg's
    check guard compares across one check's dispatch (``util write-prompt``
    records it, ``util batch-done`` reads it back; fix round 13), so both
    see the same files the same way. *exclude_prefixes* are repo-relative
    files or directories whose contents are left out. ``None`` when git
    cannot answer.
    """
    try:
        stdout, _, returncode = _run_git(
            "ls-files", "--cached", "--others", "--exclude-standard", "-z",
            "--full-name", "--", ":/",
            check=False, cwd=repo_root,
        )
    except OSError:
        return None
    if returncode != 0:
        return None
    excluded_paths = {
        prefix.strip("/").replace(os.sep, "/")
        for prefix in exclude_prefixes if prefix and prefix.strip("/")
    }
    excluded = tuple(path + "/" for path in excluded_paths)
    stats: Dict[str, Tuple[int, int]] = {}
    for path in sorted(p for p in stdout.split("\0") if p):
        if path in excluded_paths or path.startswith(excluded) or path in stats:
            continue
        try:
            info = os.lstat(os.path.join(repo_root, path))
        except OSError:
            continue
        stats[path] = (info.st_size, info.st_mtime_ns)
    return stats


def worktree_fingerprint(
    repo_root: str, *, exclude_prefixes: Iterable[str] = ()
) -> Optional[str]:
    """One digest over the files git sees in the worktree: tracked plus
    untracked-not-ignored, each as ``path``, size and ``mtime_ns``
    (:func:`worktree_file_stats`).

    What a resumed ``implement`` asks before re-running a verification-only
    task that failed: has anything the check could have read changed since?
    (racing re-run 3, 2026-10-05: ``--resume`` re-ran a read-only
    ``html-validate`` check on an unchanged page, 148 s for the same
    failure). Cheap on purpose — one ``git ls-files`` plus an ``lstat`` per
    path, no content hashing — and a ``mtime`` move counts as a change, so
    the answer errs towards re-running. Gitignored paths are left out, since
    a host's transcript or a build log growing beside the code is not a code
    change. *exclude_prefixes* are repo-relative directories (the plugin's
    own output trees) whose files are left out too. ``None`` when git cannot
    answer, so a caller re-runs rather than guesses.
    """
    stats = worktree_file_stats(repo_root, exclude_prefixes=exclude_prefixes)
    if stats is None:
        return None
    digest = hashlib.sha256()
    for path in sorted(stats):
        size, mtime_ns = stats[path]
        digest.update(f"{path}\0{size}\0{mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def list_worktree_untracked_files() -> Optional[List[str]]:
    """Every untracked, not-ignored file in the whole worktree, root-relative.

    What ``git diff <ref>`` leaves out of a fallback scope: git diff never shows
    a fully untracked file. The ``:/`` pathspec and ``--full-name`` make the
    answer the same from any directory inside the repository, and ``-z`` hands
    each path back raw. ``None`` when git fails, so a caller can say the count
    is unknown rather than report zero.
    """
    try:
        stdout, _, returncode = _run_git(
            "ls-files", "--others", "--exclude-standard", "-z", "--full-name",
            "--", ":/",
            check=False,
        )
    except OSError:
        return None
    if returncode != 0:
        return None
    return [path for path in stdout.split("\0") if path]


# ---------------------------------------------------------------------------
# The scope manifest and per-file diffs a code review reads
# ---------------------------------------------------------------------------
#
# A code reviewer used to be told only to run `git diff`, and a lane with no
# shell (cursor-agent in `--mode ask`, claude-code under `acceptEdits`) never saw
# a diff at all: on the run behind this, kimi returned nothing and grok spent
# 89% of its budget and made one wrong high-severity claim. The diff cannot be
# put in the prompt either — an argv lane's whole prompt is one argument (Linux
# `MAX_ARG_STRLEN`, 128 KB) and that change was 288 KB for its non-test source
# alone. So the scope and one whole diff per source or test file are written as
# files beside the report, and every lane that can read files reads them by
# path; its harness pages through a large file itself (plan:
# review-run-scope-salvage-and-failure-visibility §2, D2.2, decision 11.4).

REVIEW_SCOPE_FILENAME = "review-scope.md"
REVIEW_DIFF_DIRNAME = "diff"
DIFF_COVERAGE_COMPLETE = "complete"
DIFF_COVERAGE_PARTIAL = "partial"
DIFF_STATUS_FULL = "full"
#: The one reason a source or test path has no diff file although it changed.
#: There is no size-based omission: every diff is written whole.
DIFF_OMITTED_REASON = "git diff failed"
DIFF_STATUS_OMITTED = "omitted: " + DIFF_OMITTED_REASON
WRITTEN_BY_FANOUT = "fan-out"
WRITTEN_BY_REAGGREGATE = "reaggregate (regenerated from the current tree)"

#: A diff file name is ``<n>-<slug>.diff``; the slug is capped so a deep path
#: cannot produce a name the filesystem refuses. ``<n>`` keeps names unique.
_DIFF_SLUG_MAX_CHARS = 80
#: Generous on purpose: a timeout omits the path (coverage goes partial), so it
#: must only ever fire on a git that is genuinely stuck, never on a large file.
_DIFF_GIT_TIMEOUT_S = 300
#: Paths per ``git diff --numstat`` call, far under any argv limit even for
#: long paths (a 1 040-path scope is three calls).
_NUMSTAT_CHUNK = 500
_SCOPE_ID_CHARS = 12
_DIFF_FAILED_PRINT_LIMIT = 5
_DIFF_PARTIAL_SAMPLE = 5
#: `*` ignores the directory's contents and the file itself: a plans directory
#: that is committed does not commit a diff snapshot with every run, and
#: `git ls-files --others --exclude-standard` (the fallback's untracked count)
#: never counts an earlier run's diff files.
_DIFF_GITIGNORE = "*\n"
#: Diff flags that make the bytes a function of the tree, not of the user's
#: config: no colour codes from `color.ui=always`, no external diff program
#: (which may be interactive), and no rename pairing — each path's diff stands
#: on its own, so a deleted file shows every removed line.
_DIFF_FLAGS = ("--no-color", "--no-ext-diff", "--no-renames")


class ReviewDiffContainmentError(RuntimeError):
    """A diff file whose resolved target would not lie directly inside ``diff/``.

    Raised and never caught: the slug rule makes it impossible, so reaching it
    means that rule is broken, and a write outside the phase dir must not be
    skipped quietly. It is not a git failure and is not reported as one.
    """


class ReviewDiffSet(NamedTuple):
    """What :func:`write_review_diff` wrote (or found already on disk).

    ``coverage`` is :data:`DIFF_COVERAGE_COMPLETE` or
    :data:`DIFF_COVERAGE_PARTIAL`; ``omitted`` names every source or test path
    that has no diff file although it may have changed, with its reason.
    ``written`` is ``False`` when the manifest on disk already described this
    exact set and nothing was touched; ``written_by`` is then the header on
    disk, which may be an earlier reaggregate's.
    """

    coverage: str
    omitted: List[Tuple[str, str]]
    scope_id: str
    scope_file: str
    diff_dir: str
    base_ref: str
    base_commit: Optional[str]
    written_by: str
    written: bool
    failure: Optional[str]


class ReviewScopeHeader(NamedTuple):
    """The header and coverage block of a ``review-scope.md``, read back."""

    scope_id: str
    base_ref: str
    base_commit: Optional[str]
    written_by: str
    coverage: str
    omitted: List[Tuple[str, str]]
    #: Every diff file the manifest names, relative to the manifest
    #: (``diff/<n>-<slug>.diff``).
    diff_files: List[str]
    #: The byte size the manifest records for each of ``diff_files``.
    diff_sizes: Dict[str, int]


def review_diff_slug(path: str) -> str:
    """``path`` as a file-name fragment that can never leave ``diff/``.

    Every character outside ``[A-Za-z0-9._-]`` — ``/`` included — becomes
    ``-``, leading dots are stripped and the length is capped, so a slug holds
    no separator and is never ``.`` or ``..``.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", path).lstrip(".")
    return slug[:_DIFF_SLUG_MAX_CHARS] or "file"


def _code_span(text: str) -> str:
    """``text`` as a Markdown code span that survives a backtick inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _git_diff_output(
    args: Sequence[str], cwd: Optional[str]
) -> Tuple[Optional[bytes], Optional[str]]:
    """One git call for the diff set: ``(stdout bytes, None)`` or ``(None, why)``.

    Bytes, not text: a diff file must be the bytes git printed, and
    ``_run_git``'s lossy decode would change a non-UTF-8 line. Every way git can
    fail — a non-zero exit, a binary that will not start, a hang — comes back as
    the first line of its reason and never raises, because a git failure
    degrades the artifacts, never the phase. ``--literal-pathspecs`` so a path
    holding ``*`` or ``:(`` means that file and nothing else.
    """
    try:
        proc = subprocess.run(
            git_argv("--literal-pathspecs", *args, cwd=cwd),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            timeout=_DIFF_GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {_DIFF_GIT_TIMEOUT_S} s"
    except (OSError, ValueError) as exc:
        return None, f"git could not run ({exc})"
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return None, detail[0] if detail else f"exit {proc.returncode}"
    return proc.stdout, None


def _resolve_base_commit(base_ref: str, cwd: Optional[str]) -> Optional[str]:
    out, error = _git_diff_output(
        ["rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}"], cwd
    )
    if error is not None or not out:
        return None
    return out.decode("utf-8", errors="replace").strip() or None


_Counts = Tuple[Optional[int], Optional[int]]


def _scope_bytes(text: str) -> bytes:
    """``text`` as UTF-8, each ``surrogateescape`` stand-in as the byte it replaced.

    A file name that is not valid UTF-8 reaches this module the way Python
    decodes argv and file names — a lone surrogate per undecodable byte — and
    :func:`_numstat_counts` decodes git's ``-z`` output the same way, so the
    scope's paths can carry them; strict UTF-8 raises on one. The scope id and
    the manifest both encode through this, so each holds the bytes the file is
    really named by. Any other lone surrogate names no file git could have been
    handed, and falls back to ``surrogatepass`` so this never raises.
    """
    try:
        return text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="surrogatepass")


def _numstat_counts(
    base_ref: str, paths: Sequence[str], cwd: Optional[str]
) -> Tuple[Dict[str, _Counts], Optional[str]]:
    """``{path: (added, deleted)}`` for the scope, or the first failure.

    ``-z`` hands each path back byte-for-byte, and it is decoded with
    ``surrogateescape`` — the inverse of how ``subprocess`` encoded the path it
    was given — so a path that is not valid UTF-8 keys its row under the very
    string the scope holds rather than a lossy stand-in :func:`_entry_counts`
    would never find. A binary (or ``-diff``) path reports ``-`` for both
    counts, kept as ``None``.
    """
    counts: Dict[str, _Counts] = {}
    for start in range(0, len(paths), _NUMSTAT_CHUNK):
        chunk = list(paths[start:start + _NUMSTAT_CHUNK])
        out, error = _git_diff_output(
            ["diff", "--numstat", "-z", *_DIFF_FLAGS, base_ref, "--", *chunk], cwd
        )
        if error is not None:
            return counts, error
        for record in (out or b"").decode("utf-8", errors="surrogateescape").split("\0"):
            fields = record.split("\t", 2)
            if len(fields) != 3 or not fields[2]:
                continue
            added, deleted, path = fields
            counts[path] = (
                int(added) if added.isdigit() else None,
                int(deleted) if deleted.isdigit() else None,
            )
    return counts, None


def _entry_counts(path: str, counts: Mapping[str, _Counts]) -> _Counts:
    """A scope entry's counts: its own row, or the sum of the rows beneath it.

    A path absent from the numstat changed nothing (``(0, 0)``). A directory
    entry (an old ``tracked_files`` can hold one) sums what lies under it.
    """
    if path in counts:
        return counts[path]
    prefix = path.rstrip("/") + "/"
    rows = [value for key, value in counts.items() if key.startswith(prefix)]
    if not rows:
        return (0, 0)
    if any(a is None or d is None for a, d in rows):
        return (None, None)
    return (sum(a for a, _ in rows), sum(d for _, d in rows))  # type: ignore[misc]


def _counts_text(counts: Optional[_Counts]) -> str:
    if counts is None:
        return ""
    added, deleted = counts
    if added is None or deleted is None:
        return " (binary)"
    return f" (+{added} -{deleted})"


def _render_review_scope(
    *,
    scope_id: str,
    base_ref: str,
    base_commit: Optional[str],
    scope_source: Optional[str],
    written_by: str,
    failure: Optional[str],
    coverage: str,
    omitted: Sequence[Tuple[str, str]],
    reviewed_total: int,
    entries: Sequence[Dict[str, Any]],
    collapsed: Sequence[Dict[str, Any]],
) -> str:
    if not base_ref:
        base_line = "- Base ref: (none: neither the requested ref nor HEAD~1 resolves)"
    elif base_commit:
        base_line = f"- Base ref: `{base_ref}` (commit `{base_commit}`)"
    else:
        base_line = f"- Base ref: `{base_ref}` (does not resolve to a commit)"
    lines = [
        "# Review scope",
        "",
        f"- Scope id: `{scope_id}`",
        base_line,
        f"- Scope source: {scope_source or 'not recorded'}",
        f"- Written: {datetime.now().isoformat(timespec='seconds')}",
        f"- Written by: {written_by}",
        "",
    ]
    if failure is not None:
        lines.append(f"Diff failed: {failure}")
    if coverage == DIFF_COVERAGE_COMPLETE:
        lines.append(f"Diff coverage: {DIFF_COVERAGE_COMPLETE}")
    else:
        lines.append(
            f"Diff coverage: {DIFF_COVERAGE_PARTIAL} — {len(omitted)} of "
            f"{reviewed_total} source and test files omitted"
        )
        for path, reason in omitted:
            lines.append(f"- {_code_span(path)}: {reason}")
    lines.extend([
        "",
        "Every path under review is listed below, grouped as source, tests, "
        "fixtures and generated. Each source and test entry names its diff file "
        "(relative to this file) with its size, or says on its own line why it "
        "has none. The diff files are the authoritative record of the change: "
        "each is one file's whole unified diff against the base ref, with both "
        "sides of every hunk, removed lines and deleted files included. A file's "
        "current contents show only the new side, so never review them in place "
        "of its diff. Fixture and generated paths are collapsed to one line per "
        "directory and get no diff file.",
    ])

    def group_heading(title: str, count: int, suffix: str = "") -> None:
        noun = "file" if count == 1 else "files"
        lines.extend(["", f"## {title} ({count} {noun}{suffix})", ""])

    for group, title in ((REVIEW_GROUP_SOURCE, "Source"), (REVIEW_GROUP_TESTS, "Tests")):
        members = [entry for entry in entries if entry["group"] == group]
        group_heading(title, len(members))
        if not members:
            lines.append("_None._")
        for entry in members:
            head = f"{entry['number']}. {_code_span(entry['path'])}{_counts_text(entry['counts'])}"
            status = entry["status"]
            if entry["diff"]:
                lines.append(
                    f"{head} — {status}: `{entry['diff']}`, {entry['size']} bytes"
                )
            elif status == DIFF_STATUS_OMITTED and base_ref:
                command = f"git diff {base_ref} -- {shlex.quote(entry['path'])}"
                lines.append(
                    f"{head} — {status}. With a shell, {_code_span(command)} "
                    f"shows it"
                )
            else:
                lines.append(f"{head} — {status}")

    for group, title in (
        (REVIEW_GROUP_FIXTURES, "Fixtures"), (REVIEW_GROUP_GENERATED, "Generated"),
    ):
        members = [row for row in collapsed if row["group"] == group]
        group_heading(title, len(members), ", collapsed" if members else "")
        if not members:
            lines.append("_None._")
            continue
        by_dir: Dict[str, List[Dict[str, Any]]] = {}
        for row in members:
            directory = posixpath.dirname(_normalized_review_path(row["path"]))
            by_dir.setdefault(directory + "/" if directory else "./", []).append(row)
        for directory in sorted(by_dir):
            rows = by_dir[directory]
            noun = "file" if len(rows) == 1 else "files"
            summed: Optional[_Counts] = None
            if failure is None:
                known = [row["counts"] for row in rows]
                if any(a is None or d is None for a, d in known):
                    summed = (None, None)
                else:
                    summed = (sum(a for a, _ in known), sum(d for _, d in known))  # type: ignore[misc]
            lines.append(
                f"- {_code_span(directory)} — {len(rows)} {noun}"
                f"{_counts_text(summed)} — no diff: collapsed ({group})"
            )
    lines.append("")
    return "\n".join(lines)


_SCOPE_ID_LINE = re.compile(r"^- Scope id: `([0-9a-f]+)`\s*$")
_BASE_REF_LINE = re.compile(r"^- Base ref: (.*)$")
_WRITTEN_BY_LINE = re.compile(r"^- Written by: (.+?)\s*$")
_COVERAGE_LINE = re.compile(r"^Diff coverage: (complete|partial)\b")
_OMITTED_LINE = re.compile(r"^- (`+) ?(.*?) ?\1: (.+?)\s*$")
#: A source or test entry naming its diff file, as ``_render_review_scope``
#: writes it: ``<n>. <path>[ (counts)] — full: `diff/<name>.diff`, <size> bytes``.
#: Anchored on the numbered entry and on the end of its line, so an in-scope
#: path that merely looks like a diff file (``diff/fix-1.diff``, in its own
#: code span or in the coverage block) is never read as one the set wrote.
_DIFF_ENTRY_LINE = re.compile(
    r"^\d+\. .* — " + re.escape(DIFF_STATUS_FULL)
    + r": `(" + re.escape(REVIEW_DIFF_DIRNAME) + r"/[A-Za-z0-9._-]+\.diff)`, "
    + r"(\d+) bytes\r?$"
)


def read_review_scope(scope_file: Union[str, Path]) -> Optional[ReviewScopeHeader]:
    """The header and coverage of a ``review-scope.md``; ``None`` if unusable.

    ``None`` for a missing or unreadable file and for one without a scope id or
    a coverage line — an interrupted or hand-cut manifest reads as missing, and
    a reader then regenerates it rather than trusting half a file. The diff
    files are read off the numbered entries alone, each with the size its entry
    records. Decoded with ``surrogateescape``, as :func:`write_review_diff`
    encoded it, so a path that is not valid UTF-8 reads back as given.
    """
    try:
        text = Path(scope_file).read_text(encoding="utf-8", errors="surrogateescape")
    except (OSError, ValueError):
        return None
    scope_id = base_ref = written_by = coverage = None
    base_commit: Optional[str] = None
    omitted: List[Tuple[str, str]] = []
    in_coverage = False
    for line in text.splitlines():
        if in_coverage:
            match = _OMITTED_LINE.match(line)
            if match:
                omitted.append((match.group(2), match.group(3)))
                continue
            in_coverage = False
        if scope_id is None and (match := _SCOPE_ID_LINE.match(line)):
            scope_id = match.group(1)
        elif base_ref is None and (match := _BASE_REF_LINE.match(line)):
            value = match.group(1)
            ref = re.match(r"`([^`]*)`", value)
            base_ref = ref.group(1) if ref else ""
            commit = re.search(r"commit `([0-9a-f]+)`", value)
            base_commit = commit.group(1) if commit else None
        elif written_by is None and (match := _WRITTEN_BY_LINE.match(line)):
            written_by = match.group(1)
        elif coverage is None and (match := _COVERAGE_LINE.match(line)):
            coverage = match.group(1)
            in_coverage = coverage == DIFF_COVERAGE_PARTIAL
    if not scope_id or not coverage:
        return None
    # Split on the "\n" the writer joins with, not `splitlines()`, which also
    # breaks at characters a path may hold (U+2028, \x1c…) and would part an
    # entry from its diff file.
    diff_sizes: Dict[str, int] = {}
    for line in text.split("\n"):
        match = _DIFF_ENTRY_LINE.match(line)
        if match:
            diff_sizes[match.group(1)] = int(match.group(2))
    diff_files = list(diff_sizes)
    return ReviewScopeHeader(
        scope_id=scope_id,
        base_ref=base_ref or "",
        base_commit=base_commit,
        written_by=written_by or "not recorded",
        coverage=coverage,
        omitted=omitted,
        diff_files=diff_files,
        diff_sizes=diff_sizes,
    )


def _diff_file_is_whole(target: Path, size: int) -> bool:
    try:
        return target.is_file() and target.stat().st_size == size
    except OSError:
        return False


def load_review_scope(out_dir: Union[str, Path]) -> Optional[ReviewScopeHeader]:
    """The manifest in ``out_dir`` when the pair is whole, else ``None``.

    Whole means: ``review-scope.md`` parses, ``diff/`` is a directory, and
    every diff file the manifest names exists at the byte size its entry
    records. Anything less — a phase dir from before the manifest existed, one
    cleaned by hand, a write interrupted part way, a diff file truncated or
    edited to another length — reads as missing.
    """
    out = Path(out_dir)
    header = read_review_scope(out / REVIEW_SCOPE_FILENAME)
    if header is None or not (out / REVIEW_DIFF_DIRNAME).is_dir():
        return None
    if not all(
        _diff_file_is_whole(out / rel, header.diff_sizes[rel])
        for rel in header.diff_files
    ):
        return None
    return header


def _diff_set_mismatch(
    out: Path, header: ReviewScopeHeader, entries: Sequence[Dict[str, Any]]
) -> Optional[str]:
    """Why the diff files on disk are not the ones ``entries`` hold, or ``None``.

    A scope id is computed from the diffs just generated, but reviewers read
    the files on disk, so a set is reused only when the manifest names exactly
    this set's files and each holds the very bytes computed for it (size, then
    sha256).
    """
    expected = {entry["diff"]: entry for entry in entries if entry["diff"]}
    if set(header.diff_files) != set(expected):
        return "its manifest names other diff files than this scope has"
    for rel, entry in expected.items():
        try:
            data = (out / rel).read_bytes()
        except OSError as exc:
            return f"{rel} cannot be read ({exc.strerror or exc})"
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            return f"{rel} is not the diff that scope id describes"
    return None


def _contained_diff_target(out_dir: Path, diff_dir: Path, name: str) -> Path:
    """``diff_dir / name``, checked to lie directly inside ``diff/``."""
    target = diff_dir / name
    real_dir = os.path.realpath(diff_dir)
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or os.sep in name
        or os.path.dirname(os.path.realpath(target)) != real_dir
        or os.path.dirname(real_dir) != os.path.realpath(out_dir)
    ):
        raise ReviewDiffContainmentError(
            f"refusing to write diff file {name!r}: its target "
            f"{os.path.realpath(target)} does not lie directly inside {real_dir} "
            f"(the diff file name rule is broken)"
        )
    return target


def _write_diff_file(target: Path, data: bytes) -> None:
    """Write one file of the set. A seam of its own so a test can interrupt it."""
    with open(target, "wb") as handle:
        handle.write(data)


def _reset_diff_dir(diff_dir: Path) -> None:
    """Empty ``diff/`` (or replace whatever stands at that name) and recreate it."""
    if diff_dir.is_symlink() or (diff_dir.exists() and not diff_dir.is_dir()):
        diff_dir.unlink()
    if diff_dir.is_dir():
        for child in diff_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    diff_dir.mkdir(parents=True, exist_ok=True)


def write_review_diff(
    base_ref: Optional[str],
    paths: Iterable[str],
    out_dir: Union[str, Path],
    collapse_paths: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    scope_source: Optional[str] = None,
    written_by: str = WRITTEN_BY_FANOUT,
    cwd: Optional[str] = None,
) -> ReviewDiffSet:
    """Write ``review-scope.md`` and ``diff/`` for a code review into ``out_dir``.

    The manifest lists every in-scope path with its ``git diff --numstat``
    counts, grouped by :func:`classify_review_paths` (the one classifier the
    prompt's file list uses too, so the two can never put a path in different
    groups; no ``git check-attr`` of its own), in :func:`order_review_paths`'
    order, which is why ``<n>`` numbers files source first even when the
    reaggregate hands over an unordered list. Each source and test path gets
    one whole diff file, ``diff/<n>-<slug>.diff`` — no per-file cap, no total
    cap, no chunking (decision 11.4) — or a status saying why it has none:
    ``no diff: no net change since <base_ref>``, or ``omitted: git diff
    failed``. Fixture and generated paths collapse to one line per directory
    (``no diff: collapsed (fixtures|generated)``).

    Git failures degrade the set, never the phase: an empty ``base_ref`` or a
    failed ``--numstat`` omits every source and test path (``Diff failed:`` in
    the manifest), a failed per-path diff omits that path, each prints
    ``[DIFF_FAILED]``, the coverage goes ``partial`` (``[DIFF_PARTIAL]``), and
    this returns normally. Only :class:`ReviewDiffContainmentError` raises.

    The set is computed in memory first. When the manifest on disk carries the
    same scope id (a hash of the base commit — or the failure — and every
    entry's path, status, counts and diff bytes) and the diff files it names
    are exactly this set's, each holding the bytes just computed for it,
    nothing is touched; a diff file truncated or edited since it was written
    fails that check, so a reused set shows reviewers the change its id
    describes. Otherwise the pair is replaced in an order that
    never leaves a manifest naming a file it did not write: delete the
    manifest, empty ``diff/``, write the diffs, then write the manifest last by
    temp file and rename. ``diff/.gitignore`` (``*``) keeps git from tracking
    any of it.

    ``cwd`` is where git runs, and a caller passes the repository root: the
    paths are root-relative, and a path intent-to-add marked for this review
    shows only in that repository's ``git diff``. The default, the process's
    cwd, is right only at the root — from a subdirectory git reads every path
    relative to it, each diff comes back empty, and the manifest would record
    ``no net change`` under ``complete`` coverage for files that did change.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scope_file = out / REVIEW_SCOPE_FILENAME
    diff_dir = out / REVIEW_DIFF_DIRNAME
    base_ref = base_ref or ""

    groups = classify_review_paths(paths, collapse_paths, cwd=cwd)
    ordered = _review_order(groups)
    reviewed = [
        path for path in ordered
        if groups[path] in (REVIEW_GROUP_SOURCE, REVIEW_GROUP_TESTS)
    ]

    failure: Optional[str] = None
    base_commit: Optional[str] = None
    counts: Dict[str, _Counts] = {}
    if not base_ref:
        failure = (
            "git diff: no base ref (neither the requested ref nor HEAD~1 "
            "resolves: an initial commit, or a shallow clone), so there is "
            "nothing to diff against"
        )
    else:
        base_commit = _resolve_base_commit(base_ref, cwd)
        if ordered:
            counts, numstat_error = _numstat_counts(base_ref, ordered, cwd)
            if numstat_error is not None:
                failure = (
                    f"git diff --numstat {base_ref} -- <{len(ordered)} paths>: "
                    f"{numstat_error}"
                )
    if failure is not None:
        print(f"[DIFF_FAILED] {failure}")

    width = len(str(len(reviewed))) if reviewed else 1
    entries: List[Dict[str, Any]] = []
    diffs: Dict[str, bytes] = {}
    omitted: List[Tuple[str, str]] = []
    path_failures: List[str] = []
    for number, path in enumerate(reviewed, 1):
        entry: Dict[str, Any] = {
            "path": path, "group": groups[path], "number": number,
            "counts": None, "diff": None, "size": None, "sha256": None,
        }
        if failure is not None:
            entry["status"] = DIFF_STATUS_OMITTED
            omitted.append((path, DIFF_OMITTED_REASON))
        else:
            entry["counts"] = _entry_counts(path, counts)
            data, error = _git_diff_output(
                ["diff", *_DIFF_FLAGS, base_ref, "--", path], cwd
            )
            if error is not None:
                entry["status"] = DIFF_STATUS_OMITTED
                omitted.append((path, DIFF_OMITTED_REASON))
                path_failures.append(
                    f"git diff {base_ref} -- {shlex.quote(path)}: {error}"
                )
            elif not data:
                entry["status"] = f"no diff: no net change since {base_ref}"
            else:
                name = f"{str(number).zfill(width)}-{review_diff_slug(path)}.diff"
                entry.update(
                    status=DIFF_STATUS_FULL,
                    diff=f"{REVIEW_DIFF_DIRNAME}/{name}",
                    size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
                diffs[name] = data
        entries.append(entry)
    for line in path_failures[:_DIFF_FAILED_PRINT_LIMIT]:
        print(f"[DIFF_FAILED] {line}")
    if len(path_failures) > _DIFF_FAILED_PRINT_LIMIT:
        print(
            f"[DIFF_FAILED] … and {len(path_failures) - _DIFF_FAILED_PRINT_LIMIT} "
            f"more path(s); every one is listed in {scope_file}"
        )

    collapsed = [
        {
            "path": path,
            "group": groups[path],
            "counts": _entry_counts(path, counts) if failure is None else None,
            "status": f"no diff: collapsed ({groups[path]})",
        }
        for path in ordered
        if groups[path] in (REVIEW_GROUP_FIXTURES, REVIEW_GROUP_GENERATED)
    ]

    # Partial exactly when a source or test path has no diff file although it
    # may have changed. A failed set omits every such path, so it is partial
    # whenever there is one; a scope of only fixtures (or nothing) lacks
    # nothing, and its `Diff failed:` line still says what git refused.
    coverage = DIFF_COVERAGE_PARTIAL if omitted else DIFF_COVERAGE_COMPLETE
    fingerprint = [
        base_commit or failure or base_ref,
        [
            [row["path"], row["group"], row["status"], row["counts"], row.get("sha256")]
            for row in [*entries, *collapsed]
        ],
    ]
    scope_id = hashlib.sha256(
        _scope_bytes(json.dumps(fingerprint, ensure_ascii=False))
    ).hexdigest()[:_SCOPE_ID_CHARS]

    if coverage == DIFF_COVERAGE_PARTIAL:
        sample = ", ".join(path for path, _ in omitted[:_DIFF_PARTIAL_SAMPLE])
        if len(omitted) > _DIFF_PARTIAL_SAMPLE:
            sample += f" … and {len(omitted) - _DIFF_PARTIAL_SAMPLE} more"
        print(
            f"[DIFF_PARTIAL] {len(omitted)} of {len(reviewed)} source and test "
            f"files have no diff file ({DIFF_OMITTED_REASON})"
            + (f": {sample}" if sample else "")
            + "; a reviewer that cannot run git diff cannot review them"
        )

    def result(written: bool, recorded_by: str) -> ReviewDiffSet:
        return ReviewDiffSet(
            coverage=coverage,
            omitted=omitted,
            scope_id=scope_id,
            scope_file=str(scope_file.absolute()),
            diff_dir=str(diff_dir.absolute()),
            base_ref=base_ref,
            base_commit=base_commit,
            written_by=recorded_by,
            written=written,
            failure=failure,
        )

    existing = load_review_scope(out)
    if existing is not None and existing.scope_id == scope_id:
        mismatch = _diff_set_mismatch(out, existing, entries)
        if mismatch is None:
            if not (diff_dir / ".gitignore").exists():
                _write_diff_file(
                    _contained_diff_target(out, diff_dir, ".gitignore"),
                    _DIFF_GITIGNORE.encode("utf-8"),
                )
            print(f"Review scope unchanged (scope {scope_id}): {scope_file}")
            return result(False, existing.written_by)
        print(
            f"Review scope {scope_id} is on disk, but {mismatch}; rewriting the set"
        )

    text = _render_review_scope(
        scope_id=scope_id,
        base_ref=base_ref,
        base_commit=base_commit,
        scope_source=scope_source,
        written_by=written_by,
        failure=failure,
        coverage=coverage,
        omitted=omitted,
        reviewed_total=len(reviewed),
        entries=entries,
        collapsed=collapsed,
    )
    # The replacement order is the whole of the guarantee that a manifest on
    # disk only ever names files that exist and that this call wrote: with the
    # manifest gone first and written last, a write interrupted anywhere in
    # between leaves no manifest, which every reader treats as missing.
    scope_file.unlink(missing_ok=True)
    _reset_diff_dir(diff_dir)
    _write_diff_file(
        _contained_diff_target(out, diff_dir, ".gitignore"),
        _DIFF_GITIGNORE.encode("utf-8"),
    )
    for name, data in diffs.items():
        _write_diff_file(_contained_diff_target(out, diff_dir, name), data)
    temp = out / f".{REVIEW_SCOPE_FILENAME}.{os.getpid()}.tmp"
    with open(temp, "wb") as handle:
        handle.write(_scope_bytes(text))
    os.replace(temp, scope_file)
    print(
        f"Review scope written (scope {scope_id}, coverage {coverage}, "
        f"{len(diffs)} diff file(s) for {len(reviewed)} source and test "
        f"file(s)): {scope_file}"
    )
    return result(True, written_by)
