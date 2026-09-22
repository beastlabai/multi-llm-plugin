#!/usr/bin/env python3
"""Seed the container's sample project: a throwaway git repo with a real diff (C7).

``review-code`` and ``apply-code-fixes`` need something to review, so the image
cannot just carry a plan file. This builds, fresh per row:

* a small **Python** source tree — decided rather than discovered, because the
  seed script's tree *is* the diff and the diff is what every provider-lane golden
  is captured over. Changing the language later would not edit a fixture, it would
  re-capture every bundle under ``tests/golden/container/<cli>/``;
* ``plans/auth-feature.md`` — the **existing** ``tests/fixtures/e2e/plans/
  auth-feature.md``, copied verbatim so the container lane and the
  instruction-refactor guard share one fixture and cannot drift;
* one commit, then an uncommitted working diff the plan describes;
* ``.multi-llm/providers.yaml`` — the row's **complete** selection override,
  rendered from ``providers.tmpl.yaml``.

Two properties this script exists to guarantee:

**The commit identity is pinned in all six parts, not just the dates.** A commit
sha hashes author and committer *name and email* as well as the two timestamps,
and C6a rule 4 deny-lists ``~/.gitconfig``. Leaving identity unpinned has two bad
outcomes and no good one: either git guesses a ``user.email`` from the container's
hostname and uid — making the sha a property of the machine — or, with no gitconfig
anywhere on the tmpfs ``$HOME``, the seed commit fails outright ("Author identity
unknown") and takes the whole run with it. Identity travels as ``git -c`` per
invocation, never by writing a ``~/.gitconfig`` the deny-list assertion would then
have to distinguish from a mounted one.

**There is no remote and no way to reach one.** No ``origin``, no credential
helper, ``GIT_TERMINAL_PROMPT=0``. Nothing inside the container can push anywhere,
including the ``agent-write`` rows that are allowed to edit files.

**A one-spec render is no longer BYTE-identical to the pre-change render, and
that is deliberate.** ``--spec`` is a repeatable fan-out now, and the template's
selection slot became a bare one that the renderer fills with a whole YAML flow
list (``models: ["codex:gpt-5.5"]`` where a two-line block sequence used to
stand). A one-element list re-renders through that slot like any other, so the
bytes move even when nothing about the row did. What must not move is the
*parse*: the plain-CI render guards load both documents and compare the loaded
objects, never the text, because byte identity is what was traded away to get a
slot a ``str.replace`` can fill with a sequence at all (``providers.tmpl.yaml``
carries the full reasoning in its header).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The skill root is NOT `HERE.parent.parent` in every layout, and assuming it is
# fails only inside the container. In the repo this file sits at
# `skills/multi-llm/tests/container/`, so two levels up is the skill; in the image
# the rig is copied to `/opt/mllm/rig/` while the skill lives at
# `/opt/mllm/skill/`, and two levels up is `/opt`. `$MLLM_SKILL_DIR` is what the
# image sets, so it wins where it is present.
SKILL_ROOT = Path(os.environ.get("MLLM_SKILL_DIR") or HERE.parent.parent)
PLAN_FIXTURE = SKILL_ROOT / "tests" / "fixtures" / "e2e" / "plans" / "auth-feature.md"
if not PLAN_FIXTURE.is_file():
    # Fall back to the repo layout, then fail with both paths named rather than
    # with a bare FileNotFoundError from inside shutil.copyfile.
    _repo = HERE.parent.parent / "tests" / "fixtures" / "e2e" / "plans" / "auth-feature.md"
    if _repo.is_file():
        SKILL_ROOT, PLAN_FIXTURE = HERE.parent.parent, _repo
TEMPLATE = HERE / "providers.tmpl.yaml"

# Pinned in all six parts. Any change here moves every golden.
GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "multi-llm e2e",
    "GIT_AUTHOR_EMAIL": "e2e@multi-llm.invalid",
    "GIT_COMMITTER_NAME": "multi-llm e2e",
    "GIT_COMMITTER_EMAIL": "e2e@multi-llm.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}

# --------------------------------------------------------------------------- #
# The committed tree: what `review-code` diffs against.
# --------------------------------------------------------------------------- #

COMMITTED = {
    "src/__init__.py": '"""A small auth service, as described in plans/auth-feature.md."""\n',
    "src/models.py": '''"""User and session models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class User:
    """A registered user."""

    id: str
    email: str
    password_hash: str
    created_at: datetime = field(default_factory=_now)
    verified_at: datetime | None = None

    def is_verified(self) -> bool:
        return self.verified_at is not None


@dataclass
class Session:
    """An active session, keyed by its refresh token."""

    id: str
    user_id: str
    refresh_token: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return _now() >= self.expires_at
''',
    "src/auth.py": '''"""Authentication: registration, login, and token issue."""

import hashlib
import re

EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")


def validate_email(email: str) -> bool:
    """Whether ``email`` looks like an address."""
    return bool(EMAIL_RE.match(email or ""))


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Whether ``password`` matches ``stored_hash``."""
    return hash_password(password) == stored_hash
''',
    "src/sessions.py": '''"""Session lifecycle."""

from typing import Dict, Optional

from .models import Session


class SessionStore:
    """An in-memory session table."""

    def __init__(self) -> None:
        self._by_token: Dict[str, Session] = {}

    def add(self, session: Session) -> None:
        self._by_token[session.refresh_token] = session

    def get(self, refresh_token: str) -> Optional[Session]:
        return self._by_token.get(refresh_token)

    def revoke(self, refresh_token: str) -> bool:
        return self._by_token.pop(refresh_token, None) is not None
''',
    "tests/test_auth.py": '''"""Tests for the auth helpers."""

from src.auth import hash_password, validate_email, verify_password


def test_validate_email_accepts_an_address():
    assert validate_email("user@example.com")


def test_validate_email_rejects_a_bare_word():
    assert not validate_email("nope")


def test_password_round_trip():
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong", stored)
''',
    "README.md": """# sample-project

A throwaway fixture repo for the multi-llm container harness. It implements the
first steps of `plans/auth-feature.md` and carries an uncommitted working diff so
`review-code` has something to review.
""",
}

# --------------------------------------------------------------------------- #
# The uncommitted working diff: what `review-code` reviews and
# `apply-code-fixes` is asked to fix. Deliberately reviewable — a plausible
# rate-limiter with real problems in it (unbounded dict, no lockout, a
# comparison that is not constant time) so a fan-out has something to say.
# --------------------------------------------------------------------------- #

WORKING_DIFF = {
    "src/security.py": '''"""Rate limiting and account lockout (work in progress)."""

import time

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 300

_attempts = {}


def record_attempt(email, success):
    """Record a login attempt for ``email``."""
    bucket = _attempts.setdefault(email, [])
    bucket.append((time.time(), success))
    return len(bucket)


def is_locked_out(email):
    """Whether ``email`` has too many recent failures."""
    bucket = _attempts.get(email, [])
    recent = [t for t, ok in bucket if not ok and time.time() - t < WINDOW_SECONDS]
    return len(recent) > MAX_ATTEMPTS


def compare_tokens(a, b):
    """Compare two tokens."""
    return a == b
''',
    "src/auth.py": '''"""Authentication: registration, login, and token issue."""

import hashlib
import re

from .security import is_locked_out, record_attempt

EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")


def validate_email(email: str) -> bool:
    """Whether ``email`` looks like an address."""
    return bool(EMAIL_RE.match(email or ""))


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Whether ``password`` matches ``stored_hash``."""
    return hash_password(password) == stored_hash


def login(email: str, password: str, stored_hash: str) -> bool:
    """Attempt a login, recording the attempt for rate limiting."""
    if is_locked_out(email):
        return False
    ok = verify_password(password, stored_hash)
    record_attempt(email, ok)
    return ok
''',
}


def _git(args, cwd, env):
    """Run one git command with the pinned identity and no terminal prompt."""
    identity = []
    for key, value in (("user.name", GIT_IDENTITY["GIT_AUTHOR_NAME"]),
                       ("user.email", GIT_IDENTITY["GIT_AUTHOR_EMAIL"])):
        identity += ["-c", f"{key}={value}"]
    return subprocess.run(
        ["git", *identity, *args],
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def render_override(specs: list[str] | str, runner: str,
                    runner_cheap: str | None = None) -> str:
    """Render the row's complete selection override from the template.

    ``specs`` is the row's **fan-out**: every ``provider:model`` spec the row
    dispatches to, in order, written into all 22 selection positions the
    template carries. A bare string is accepted and read as a one-element list
    — the single-spec rows predate the fan-out and there is no reason to make
    every one of their callers learn a list.

    ``runner`` and ``runner_cheap`` stay singular: ``driver.subagent_runner`` is
    one slug, and widening it here would multiply the write dispatch the row is
    measuring.
    """
    if isinstance(specs, str):
        specs = [specs]
    specs = list(specs)
    if not specs:
        # Rejected loudly rather than rendered as `[]`. An empty selection list
        # deep-merges onto base as an empty selection, so the row would start,
        # dispatch to nobody, and report a shape no one typed.
        raise ValueError(
            "seed_workspace.render_override: the fan-out is empty. Name at "
            "least one provider:model spec — an empty list renders "
            "`models: []`, which leaves the row with nothing to dispatch to.")

    # `json.dumps`, not an f-string join: a JSON array of JSON strings IS a
    # valid YAML flow sequence of double-quoted scalars, punctuation included,
    # and the quoting is the load-bearing half. Every spec carries a `:` and
    # some carry a `/` (opencode's `stub/stub-model`), and an UNQUOTED `a:b`
    # inside a flow sequence re-reads as the mapping `{a: b}` — a row that
    # rendered that way would merge a dict where a list belongs and fall
    # through to base's roster without a word.
    fan_out = json.dumps(specs)

    text = TEMPLATE.read_text(encoding="utf-8")
    return (
        text.replace("{{fan_out_specs}}", fan_out)
        # `{{runner_cheap}}` is substituted BEFORE `{{runner}}`, and the order
        # is kept deliberately: it is the only order that survives one slot
        # being renamed to a prefix of another, which is exactly how a chained
        # `str.replace` renderer springs a silent leak.
        .replace("{{runner_cheap}}", runner_cheap or runner)
        .replace("{{runner}}", runner)
    )


def seed(root: Path, specs: list[str] | str, runner: str,
         runner_cheap: str | None = None) -> Path:
    """Build the sample project at ``root`` and return it.

    Clears the directory's *contents* rather than the directory. ``rmtree(root)``
    works on a developer's disk and fails on every container row, because
    ``/work/sample-project`` is a **tmpfs mount point** and a mount point cannot be
    removed from inside the container:

        OSError: [Errno 30] Read-only file system: '/work/sample-project'

    That killed every provider row before its CLI ever started.
    """
    root.mkdir(parents=True, exist_ok=True)
    for entry in root.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    env = dict(os.environ)
    env.update(GIT_IDENTITY)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env.pop("GIT_DIR", None)

    for rel, body in COMMITTED.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")

    plans = root / "plans"
    plans.mkdir(exist_ok=True)
    if not PLAN_FIXTURE.is_file():
        raise SystemExit(
            f"seed_workspace: the plan fixture is missing at {PLAN_FIXTURE}. "
            f"Set $MLLM_SKILL_DIR to the skill root (the image sets it to "
            f"/opt/mllm/skill); this script's own directory is {HERE}."
        )
    shutil.copyfile(PLAN_FIXTURE, plans / "auth-feature.md")

    _git(["init", "-q", "-b", "main"], root, env)
    _git(["add", "-A"], root, env)
    _git(["commit", "-q", "-m", "Initial auth service"], root, env)

    # No remote, and nothing that could acquire one.
    _git(["config", "--local", "credential.helper", ""], root, env)

    # The uncommitted working diff, written after the commit on purpose.
    for rel, body in WORKING_DIFF.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")

    config_dir = root / ".multi-llm"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "providers.yaml").write_text(
        render_override(specs, runner, runner_cheap), encoding="utf-8"
    )
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/work/sample-project")
    # Repeatable and order-preserving: one `--spec` per fan-out member. Every
    # one of them lands in all 22 selection positions, in the order given, so
    # `--spec a:b --spec c:d` is a two-provider row rather than two rows.
    parser.add_argument("--spec", action="append", required=True,
                        help="one provider:model spec in the row's fan-out; "
                             "repeatable, order preserved")
    parser.add_argument("--runner", required=True,
                        help="the row's agent-write-capable executor runner spec")
    parser.add_argument("--runner-cheap", default=None,
                        help="cheap-tier runner (defaults to --runner)")
    args = parser.parse_args(argv)
    # `args.spec` is a LIST here (action="append"), forwarded whole.
    root = seed(Path(args.root), args.spec, args.runner, args.runner_cheap)
    print(f"seeded {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
