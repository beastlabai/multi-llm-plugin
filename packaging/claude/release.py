#!/usr/bin/env python3
"""The push-time guard and the release helper for the Claude Code plugin.

Claude Code installs read ``master`` and update only when the version goes up, so a
push to ``master`` that moves ``packaging/claude/multi-llm/`` is the release
(``packaging/claude/README.md``, "Release procedure"). ``master`` has no branch
protection, and CI checks a push only after it has landed, when users can already
fetch it. These tools run the same checks before anything leaves this clone.

``pre-push``
    What the git hook runs, with git's ref lines on stdin. A push to
    ``refs/heads/master`` must pass the moved-tree rule against the remote's
    ``master`` and the newest ``claude-v*`` tag, and each generated tree it carries
    must equal a clean rebuild of that same commit (a forgotten ``make
    claude-regen`` publishes nothing new while looking like a release). A
    ``claude-v*`` tag push must name the version of the commit it points at, and
    that commit must already be on the destination's ``master``. Every other push
    passes untouched. A push of ``master`` or a ``claude-v*`` tag that cannot be
    checked (no ``release.py`` in the checkout, no ``uv``) is refused, never waved
    through: ``git push --no-verify`` is the deliberate way past.
``install-hook`` / ``uninstall-hook``
    Write or remove ``.git/hooks/pre-push`` (wherever ``core.hooksPath`` puts it).
    Local to this clone and never committed, like every git hook. Neither touches a
    hook it did not write.
``publish``
    The whole release from the local ``master``: fetch, the pre-push checks plus the
    shared-version history check (``make claude-check-bump``), then, with
    ``--push``, a fast-forward push of ``master`` and the ``claude-v<version>``
    tag. Without ``--push`` it prints what it would do and pushes nothing. It never
    force-pushes. Everything about the remote is asked of the one URL the push goes
    to, so a ``pushurl`` that differs from the fetch URL cannot mislead it.

Run through make: ``make claude-hook-install``, ``make claude-publish`` and
``make claude-publish PUSH=1``.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent


def _load_builder():
    """``build.py`` by path. This directory stays off ``sys.path``: its
    ``build.py`` would answer a later ``import build`` meant for another builder."""
    spec = importlib.util.spec_from_file_location("claude_build", PACKAGE_DIR / "build.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build = _load_builder()

from common import release_history  # noqa: E402  (build.py put packaging/ on sys.path)
from common.payload import PayloadError  # noqa: E402

TAG_PREFIX = build.TAG_PREFIX
MASTER_REF = "refs/heads/master"
TAG_REF_PREFIX = f"refs/tags/{TAG_PREFIX}"

#: The marker that identifies the hook this module writes, and the only one it removes.
HOOK_MARK = "multi-llm release pre-push"

#: Variables git exports to a hook, naming the pushing checkout. A git command run
#: in another worktree would read the wrong repository through them.
GIT_LOCAL_ENV = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR",
)

#: Variables that would point a ``uv run`` at an environment outside the checkout it
#: is run for, which it would then sync. Cleared before every ``uv`` this module or
#: its hook starts.
UV_ENV = ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")

#: Each generated tree a ``master`` commit carries, the builder that must reproduce
#: it from that same commit, and the command that regenerates it. Codex users
#: install from ``release``, so a stale Codex tree is not a public fault, but the
#: ``package`` job fails on it after the push; it is checked for the same price.
GENERATED: Tuple[Tuple[str, str, str], ...] = (
    (build.PLUGIN_RELATIVE, "packaging/claude/build.py", "make claude-regen"),
    ("packaging/codex/multi-llm", "packaging/codex/build.py", "make codex-regen"),
)

#: Lines of a failed rebuild's output quoted in the refusal.
DETAIL_LINES = 12

HOOK_SCRIPT = f"""#!/bin/sh
# {HOOK_MARK}: installed by `make claude-hook-install`. Local to this clone and
# never committed; `make claude-hook-uninstall` removes it.
#
# A push of master or of a claude-v* tag runs the Claude release checks in
# packaging/claude/release.py (packaging/claude/README.md, "Release procedure").
# Every other push passes straight through. When the checks cannot run, a push of
# master or of a claude-v* tag is refused, never waved through: `git push
# --no-verify` is the deliberate way past, and CI still runs the checks, but only
# after the push has landed.
input=$(cat)
case "$input" in
  *" {MASTER_REF} "*|*" {TAG_REF_PREFIX}"*) ;;
  *) exit 0 ;;
esac
refuse() {{
  echo "pre-push: $1, so the Claude release checks cannot run and this push of master or a claude-v* tag is refused. Push with --no-verify to skip them deliberately." >&2
  exit 1
}}
top=$(git rev-parse --show-toplevel 2>/dev/null) || refuse "this is not a checkout with a working tree"
unset {' '.join(GIT_LOCAL_ENV)} {' '.join(UV_ENV)}
cd "$top" || refuse "cannot enter $top"
[ -f packaging/claude/release.py ] || refuse "this checkout has no packaging/claude/release.py (check out a branch that has it)"
command -v uv >/dev/null 2>&1 || refuse "uv is not on PATH"
printf '%s\\n' "$input" | uv run --quiet --frozen --project skills/multi-llm -- \\
  python packaging/claude/release.py pre-push "$1" "$2"
"""

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
StaleCheck = Callable[[Path, str], List[str]]
HistoryCheck = Callable[[Path, str], Tuple[List[str], List[str]]]


class ReleaseError(Exception):
    """A git step this module depends on failed."""


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def _clean_env(extra_unset: Sequence[str] = ()) -> Mapping[str, str]:
    env = dict(os.environ)
    for name in (*GIT_LOCAL_ENV, *extra_unset):
        env.pop(name, None)
    return env


def _git(checkout: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        ["git", *args], cwd=str(checkout), env=_clean_env(),
        capture_output=True, text=True, check=False,
    )


def _out(checkout: Path, *args: str) -> str:
    done = _git(checkout, *args)
    if done.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def is_zero(sha: str) -> bool:
    """git's spelling of "no object": a deletion, or a ref the remote lacks."""
    return bool(sha) and set(sha) == {"0"}


def commit_of(checkout: Path, rev: str) -> Optional[str]:
    """The commit *rev* names (peeled through an annotated tag), or ``None``."""
    done = _git(checkout, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def is_ancestor(checkout: Path, ancestor: str, descendant: str) -> bool:
    done = _git(checkout, "merge-base", "--is-ancestor", ancestor, descendant)
    if done.returncode in (0, 1):
        return done.returncode == 0
    raise ReleaseError(
        f"git merge-base --is-ancestor {ancestor} {descendant} failed: {done.stderr.strip()}"
    )


def _short(sha: Optional[str]) -> str:
    return sha[:12] if sha else "(none)"


def display(url: str) -> str:
    """*url* without any ``user:secret@`` part, for messages."""
    return re.sub(r"(?<=://)[^/@]+@", "", url)


def advertised_refs(checkout: Path, destination: str, *patterns: str) -> Dict[str, str]:
    """``{ref: commit}`` that *destination* advertises now, annotated tags peeled.

    Asked of the destination itself, never of a remote-tracking ref, which says only
    what some earlier fetch from the fetch URL saw.
    """
    done = _git(checkout, "ls-remote", destination, *patterns)
    if done.returncode != 0:
        raise ReleaseError(f"git ls-remote {display(destination)} failed: {done.stderr.strip()}")
    direct: Dict[str, str] = {}
    peeled: Dict[str, str] = {}
    for line in done.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.endswith("^{}"):
            peeled[ref[:-3]] = sha
        elif ref:
            direct[ref] = sha
    return {
        ref: peeled.get(ref, sha) for ref, sha in direct.items()
        if any(fnmatch.fnmatchcase(ref, pattern) for pattern in patterns)
    }


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def stale_tree_problems(
    checkout: Path, commit: str, *, runner: Runner = subprocess.run,
) -> List[str]:
    """Rebuild each generated tree of *commit* in a throwaway worktree and compare.

    The commit's own builder runs, against the commit's own inputs, so neither this
    checkout's working tree nor its version of the builder decides the answer. A
    tree is stale when a source edit was committed without its ``make *-regen``:
    the push would look like a release and deliver nothing.
    """
    scratch = Path(tempfile.mkdtemp(prefix="multi-llm-release-check."))
    worktree = scratch / "tree"
    added = False
    try:
        done = _git(checkout, "worktree", "add", "--detach", "--quiet", str(worktree), commit)
        if done.returncode != 0:
            return [f"could not check out {_short(commit)} to rebuild it: {done.stderr.strip()}"]
        added = True
        # The builders' own dirty switch, and an environment uv would otherwise sync
        # this throwaway checkout into.
        env = _clean_env(("MULTI_LLM_ALLOW_DIRTY", *UV_ENV))
        problems: List[str] = []
        for tree, builder, regen in GENERATED:
            missing = [path for path, present in (
                (f"{tree}/", (worktree / tree).is_dir()),
                (builder, (worktree / builder).is_file()),
            ) if not present]
            if missing:
                problems.append(
                    f"{_short(commit)} carries no {' and no '.join(missing)}, so {tree}/ "
                    "cannot be rebuilt and compared. A master commit carries every "
                    "generated tree and the builder that reproduces it."
                )
                continue
            argv = [
                "uv", "run", "--quiet", "--frozen",
                "--project", str(worktree / "skills" / "multi-llm"),
                "--", "python", str(worktree / builder), "--require-no-diff",
            ]
            try:
                result = runner(argv, cwd=str(worktree), env=env,
                                capture_output=True, text=True, check=False)
            except FileNotFoundError:
                return ["uv is not on PATH, so the generated trees could not be rebuilt"]
            if result.returncode != 0:
                lines = (result.stderr or result.stdout or "").strip().splitlines()
                detail = "\n".join(f"      {line}" for line in lines[-DETAIL_LINES:])
                problems.append(
                    f"{tree}/ at {_short(commit)} is not what a clean build of that "
                    f"commit produces. Run `{regen}` and commit the result with the "
                    f"edit that implies it.\n{detail}".rstrip()
                )
        return problems
    finally:
        if added:
            _git(checkout, "worktree", "remove", "--force", str(worktree))
        shutil.rmtree(scratch, ignore_errors=True)
        if added:
            _git(checkout, "worktree", "prune")


def master_problems(
    checkout: Path, commit: str, base: Optional[str], *,
    stale: StaleCheck = stale_tree_problems,
) -> List[str]:
    """Why *commit* must not become ``master`` over *base*: the moved-tree rule,
    then the clean rebuild."""
    try:
        problems = list(build.release_problems(checkout, candidate=commit, base=base))
    except PayloadError as exc:
        return [str(exc)]
    return problems + stale(checkout, commit)


def tag_problems(
    checkout: Path, destination: str, tag: str, sha: str, masters: Sequence[str],
) -> List[str]:
    """Why ``claude-v*`` tag *tag* at *sha* is not a release record.

    *destination* is where the push goes. Its ``master``, asked of it now, and the
    ``master`` commits the same push sends (*masters*) are what the tagged commit
    must be on.
    """
    version = tag[len(TAG_PREFIX):]
    if not release_history.is_version(version):
        return [f"{tag} does not name a strict semver version"]
    commit = commit_of(checkout, sha)
    if commit is None:
        return [f"{tag} does not point at a commit"]
    problems: List[str] = []
    if build._tree_id(checkout, commit) is None:  # noqa: SLF001
        problems.append(
            f"{tag} points at {_short(commit)}, which carries no "
            f"{build.PLUGIN_RELATIVE}/. A Claude release tag records the master "
            "commit that shipped the plugin."
        )
    else:
        declared = build._version_at(checkout, commit)  # noqa: SLF001
        if declared != version:
            problems.append(
                f"{tag} points at {_short(commit)}, whose version is {declared}. "
                f"Tag that commit {TAG_PREFIX}{declared}, or tag the commit that "
                f"carries {version}."
            )
    where = display(destination)
    try:
        served = advertised_refs(checkout, destination, MASTER_REF).get(MASTER_REF)
    except ReleaseError as exc:
        return problems + [str(exc)]
    heads = list(masters)
    if served and commit_of(checkout, served):
        heads.append(served)
    if any(is_ancestor(checkout, commit, head) for head in heads):
        return problems
    if served and not commit_of(checkout, served):
        problems.append(
            f"master on {where} is at {_short(served)}, which this clone does not have, "
            f"so {tag} cannot be shown to be on it. Fetch it (`git fetch {where} master`) "
            "and push again."
        )
    else:
        problems.append(
            f"{tag} points at {_short(commit)}, which master on {where} does not contain. "
            "Push master first: the tag records what users were served."
        )
    return problems


# ---------------------------------------------------------------------------
# pre-push
# ---------------------------------------------------------------------------

def parse_updates(text: str) -> List[Tuple[str, str, str, str]]:
    """git's pre-push stdin: ``<local ref> <local sha> <remote ref> <remote sha>``."""
    updates = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ReleaseError(f"unexpected pre-push line: {line!r}")
        updates.append((fields[0], fields[1], fields[2], fields[3]))
    return updates


def pre_push(
    checkout: Path, remote: str, updates: Sequence[Tuple[str, str, str, str]], *,
    url: str = "", stale: StaleCheck = stale_tree_problems,
) -> int:
    """The hook's verdict. Messages go to stderr, which git shows the pusher.

    git passes the remote's name (or the URL, for an unnamed one) and the URL it is
    pushing to; the URL is the destination. The ``<remote sha>`` of each line is
    what that destination advertised for the ref when this push connected.
    """
    destination = url or remote
    problems: List[str] = []
    passed: List[str] = []
    masters = [local for _, local, ref, _ in updates if ref == MASTER_REF and not is_zero(local)]
    for _local_ref, local, remote_ref, remote_sha in updates:
        if is_zero(local):
            continue
        if remote_ref == MASTER_REF:
            base = None if is_zero(remote_sha) else remote_sha
            if base and commit_of(checkout, base) is None:
                problems.append(
                    f"master on {display(destination)} is at {_short(base)}, which this "
                    "clone does not have, so this push cannot be compared with what users "
                    f"have now. Fetch it (`git fetch {remote}`), merge, and push again."
                )
                continue
            found = master_problems(checkout, local, base, stale=stale)
            problems += found
            if not found:
                passed.append(f"master {_short(local)}")
        elif remote_ref.startswith(TAG_REF_PREFIX):
            tag = remote_ref[len("refs/tags/"):]
            found = tag_problems(checkout, destination, tag, local, masters)
            problems += found
            if not found:
                passed.append(tag)
    if problems:
        for line in dict.fromkeys(problems):
            print(f"REFUSING: {line}", file=sys.stderr)
        print(
            "pre-push: nothing was pushed. `git push --no-verify` skips these checks; "
            "CI still runs them, but only after the push has landed.",
            file=sys.stderr,
        )
        return 1
    if passed:
        print(f"pre-push: Claude release checks pass for {', '.join(passed)}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# The hook file
# ---------------------------------------------------------------------------

def hook_path(checkout: Path) -> Path:
    """Where git looks for ``pre-push`` in *checkout*, ``core.hooksPath`` included.
    Shared by every worktree of the clone."""
    path = Path(_out(checkout, "rev-parse", "--git-path", "hooks/pre-push"))
    return path if path.is_absolute() else checkout / path


def install_hook(checkout: Path) -> Tuple[int, str]:
    path = hook_path(checkout)
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        if HOOK_MARK not in text:
            return 1, (
                f"{path} already exists and is not this hook, so it was left alone. "
                "Move it aside and install again, or call "
                "`packaging/claude/release.py pre-push \"$1\" \"$2\"` from it, "
                "passing git's stdin through."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HOOK_SCRIPT, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return 0, (
        f"installed {path}: pushes of master and of {TAG_PREFIX}* tags now run the "
        "Claude release checks first. Local to this clone; a new clone needs "
        "`make claude-hook-install` again."
    )


def uninstall_hook(checkout: Path) -> Tuple[int, str]:
    path = hook_path(checkout)
    if not path.exists():
        return 0, f"no pre-push hook at {path}"
    if HOOK_MARK not in path.read_text(encoding="utf-8", errors="replace"):
        return 1, f"{path} is not this hook, so it was left alone"
    path.unlink()
    return 0, f"removed {path}"


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------

def history_problems(checkout: Path, commit: str) -> Tuple[List[str], List[str]]:
    """``make claude-check-bump``'s shared half: every family's tags and npm's
    version list. Returns (what was read, refusals). Reads the registry."""
    try:
        report = release_history.check(checkout, candidate=commit)
    except PayloadError as exc:
        return [], [str(exc)]
    return list(report.lines), list(report.problems)


def publish(
    checkout: Path, *, remote: str = "origin", push: bool = False, fetch: bool = True,
    stale: StaleCheck = stale_tree_problems, history: HistoryCheck = history_problems,
) -> int:
    """Check, then (with *push*) release the local ``master``. Prints its plan.

    Everything about the remote is asked of the one URL a push to *remote* goes to
    (its ``pushurl`` when set), never of a remote-tracking ref, which only says what
    an earlier fetch from the fetch URL saw.
    """
    def refuse(lines: Sequence[str]) -> int:
        for line in dict.fromkeys(lines):
            print(f"REFUSING: {line}", file=sys.stderr)
        print("publish: nothing was pushed.", file=sys.stderr)
        return 1

    branch = _git(checkout, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip()
    if branch != "master":
        return refuse([
            f"publish releases the local master and runs from it; this checkout is on "
            f"{branch or 'a detached HEAD'}. Run it from the checkout that has master."
        ])
    urls = _git(checkout, "remote", "get-url", "--push", "--all", remote)
    destinations = [line for line in urls.stdout.splitlines() if line.strip()]
    if urls.returncode != 0 or not destinations:
        return refuse([f"no remote named {remote}: {urls.stderr.strip()}"])
    if len(destinations) != 1:
        return refuse([
            f"{remote} pushes to {len(destinations)} URLs "
            f"({', '.join(display(url) for url in destinations)}). publish checks and "
            "pushes to exactly one; release to each by hand, or configure one push URL."
        ])
    destination = destinations[0]
    where = display(destination)
    if fetch:
        done = _git(checkout, "fetch", "--tags", remote)
        if done.returncode != 0:
            detail = " ".join(line.strip() for line in done.stderr.splitlines()
                              if "rejected" in line or "fatal" in line or "error" in line)
            return refuse([
                f"git fetch --tags {remote} failed: {detail or done.stderr.strip()}. A tag "
                "this clone holds differently from the remote is a release record that "
                "disagrees; find out which is right before publishing."
            ])

    candidate = _out(checkout, "rev-parse", "--verify", f"{MASTER_REF}^{{commit}}")
    try:
        version = build._version_at(checkout, candidate)  # noqa: SLF001
        served = advertised_refs(checkout, destination, MASTER_REF, f"{TAG_REF_PREFIX}*")
    except (PayloadError, ReleaseError) as exc:
        return refuse([str(exc)])
    tag = f"{TAG_PREFIX}{version}"
    base = served.get(MASTER_REF)
    if base and commit_of(checkout, base) is None:
        _git(checkout, "fetch", "--quiet", destination, MASTER_REF)
        if commit_of(checkout, base) is None:
            return refuse([
                f"master on {where} is at {_short(base)}, which this clone cannot fetch, so "
                "this release cannot be compared with what users have now."
            ])
    if base and not is_ancestor(checkout, base, candidate):
        return refuse([
            f"local master ({_short(candidate)}) does not contain master on {where} "
            f"({_short(base)}). Pull and merge first; publish never force-pushes."
        ])

    problems: List[str] = []
    # The release rule reads this clone's tags; one the destination has and this clone
    # lacks, or has elsewhere, is a release the rule would not see.
    for ref, commit in sorted(served.items()):
        if not ref.startswith(TAG_REF_PREFIX):
            continue
        name = ref[len("refs/tags/"):]
        local = commit_of(checkout, ref)
        if local != commit:
            have = "has no such tag" if local is None else f"has it at {_short(local)}"
            problems.append(
                f"{name} on {where} marks {_short(commit)}, but this clone {have}. Fetch "
                f"the tags from where you push (`git fetch --tags {where}`) so the release "
                "rule sees every release."
            )
    problems += master_problems(checkout, candidate, base, stale=stale)
    read, shared = history(checkout, candidate)
    problems += shared

    local_tag = commit_of(checkout, f"refs/tags/{tag}")
    served_tag = served.get(f"refs/tags/{tag}")
    create_tag = push_tag = False
    if served_tag is None:
        if local_tag is not None:
            if is_ancestor(checkout, local_tag, candidate):
                push_tag = True
            else:
                problems.append(
                    f"{tag} exists in this clone at {_short(local_tag)}, which master does "
                    f"not contain, and {where} does not have it. It was never published: "
                    f"delete it (`git tag -d {tag}`) and run publish again."
                )
        elif build._tree_id(checkout, candidate) is not None:  # noqa: SLF001
            create_tag = push_tag = True

    for line in read:
        print(line)
    print(f"publish: local master {_short(candidate)}, Claude Code plugin {version}, to {where}")
    if base == candidate:
        print(f"  push:   nothing; master there is already {_short(candidate)}")
    else:
        count = _out(checkout, "rev-list", "--count",
                     f"{base}..{candidate}" if base else candidate)
        print(f"  push:   master {_short(base)} -> {_short(candidate)} "
              f"({count} commit(s), fast-forward)")
    if create_tag:
        print(f"  tag:    create {tag} on {_short(candidate)} and push it")
    elif push_tag:
        print(f"  tag:    push {tag} ({_short(local_tag)}), which this clone has and {where} lacks")
    elif served_tag:
        print(f"  tag:    none needed; {tag} is published at {_short(served_tag)}")
    if problems:
        return refuse(problems)
    print("  checks: moved-tree rule, clean rebuild of the generated trees, "
          "shared-version history: pass")
    print("  not run here: `make claude-package-container` (Docker; the real install "
          "and upgrade)")

    if not push:
        print("publish: dry run, nothing pushed. `make claude-publish PUSH=1` does the above.")
        return 0

    if base != candidate:
        done = _git(checkout, "push", remote, f"{candidate}:{MASTER_REF}")
        sys.stderr.write(done.stderr)
        if done.returncode != 0:
            return refuse([f"git push {remote} master failed (exit {done.returncode})"])
    if create_tag:
        made = _git(checkout, "tag", "-a", tag, candidate, "-m", f"Claude Code plugin {version}")
        if made.returncode != 0:
            print(f"REFUSING: git tag {tag} failed: {made.stderr.strip()}", file=sys.stderr)
            print(f"publish: master is pushed but not tagged. Run `make claude-publish "
                  f"PUSH=1` again, or by hand: git tag -a {tag} {candidate} -m "
                  f"\"Claude Code plugin {version}\" && git push {remote} {tag}",
                  file=sys.stderr)
            return 1
    if push_tag:
        done = _git(checkout, "push", remote, f"refs/tags/{tag}")
        sys.stderr.write(done.stderr)
        if done.returncode != 0:
            print(f"publish: master is pushed and {tag} exists here, but pushing it failed. "
                  "Run `make claude-publish PUSH=1` again: it pushes the tag it finds "
                  "missing.", file=sys.stderr)
            return 1
    print(f"publish: released Claude Code plugin {version}. Users update with "
          "`claude plugin marketplace update beastlabai` then "
          "`claude plugin update multi-llm@beastlabai`, then a new session.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release.py", description="The Claude Code plugin's pre-push guard and release helper.",
    )
    parser.add_argument("--repo", type=Path, default=None,
                        help="the checkout to act on (default: this one)")
    sub = parser.add_subparsers(dest="command", required=True)
    hook = sub.add_parser("pre-push", help="the git hook: ref lines on stdin")
    hook.add_argument("remote")
    hook.add_argument("url", nargs="?", default="")
    sub.add_parser("install-hook", help="write .git/hooks/pre-push")
    sub.add_parser("uninstall-hook", help="remove the hook install-hook wrote")
    pub = sub.add_parser("publish", help="check, then with --push release the local master")
    pub.add_argument("--remote", default="origin")
    pub.add_argument("--push", action="store_true",
                     help="push master and the claude-v<version> tag (default: print the plan)")
    pub.add_argument("--no-fetch", action="store_true",
                     help="skip `git fetch --tags` (the history check still needs every tag)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for name in GIT_LOCAL_ENV:
        os.environ.pop(name, None)
    checkout = (args.repo or REPO_ROOT).resolve()
    try:
        if args.command == "pre-push":
            return pre_push(checkout, args.remote, parse_updates(sys.stdin.read()), url=args.url)
        if args.command in ("install-hook", "uninstall-hook"):
            action = install_hook if args.command == "install-hook" else uninstall_hook
            status, message = action(checkout)
            print(message, file=sys.stderr if status else sys.stdout)
            return status
        return publish(checkout, remote=args.remote, push=args.push, fetch=not args.no_fetch)
    except ReleaseError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
