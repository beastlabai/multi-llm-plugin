#!/usr/bin/env python3
"""The helper layer both release gates share: ``packaging/codex/release_gate.py``
and ``packaging/opencode/release_gate.py``.

Each family keeps its own rows, graders, evidence shape and stamp rules. What is
the same contract in both lives here once:

* **The vocabulary.** :class:`GateError` (a refusal worded for the maintainer
  reading the make output), :data:`STAMP_RE` (a run record's
  ``YYYYmmddTHHMMSSZ`` name prefix), :func:`read_json` (unreadable or invalid is
  ``None``) and :func:`records_for` (a row's run records since a stamp).
* **The session.** :func:`begin_session` deletes the earlier evidence and writes
  a fresh id, so the stamp accepts only evidence from the running gate;
  :func:`current_session` reads it back. The family adds its own fields (the
  opencode candidate commit and its dirtiness).
* **The bounded step.** :func:`run_step` runs one gate step under a wall-clock
  bound with its output kept, and the ``step`` subcommand both gates expose
  (:func:`add_step_parser`, :func:`split_step_command`).

**Stdlib only.** The opencode publishing workflow's ``verify`` job runs that
gate's ``validate-record`` under ``uv run --frozen --no-dev``, which installs the
runtime dependencies alone.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from common import procgroup

#: A run record's name prefix, compared lexically with a ``--since`` stamp.
STAMP_RE = re.compile(r"\d{8}T\d{6}Z")

SESSION_NAME = "session.json"


class GateError(Exception):
    """A refusal, worded for the maintainer reading the make output."""


def read_json(path: Path) -> Any:
    """``path`` parsed, or ``None`` when it is missing, unreadable or not JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any, *, sort_keys: bool = False) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")
    return path


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(moment: datetime.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def compact(moment: datetime.datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def records_for(row_id: str, since: str, records_dir: Path) -> List[Path]:
    """Every record of ``row_id`` started at or after ``since``, oldest first."""
    found = []
    for path in sorted(Path(records_dir).glob(f"*-{row_id}.json")):
        stamp = path.name[: -len(f"-{row_id}.json")]
        if STAMP_RE.fullmatch(stamp) and stamp >= since:
            found.append(path)
    return found


def harness(skill_dir: Path):
    """The container harness module (``tests/container/harness.py``): the run
    record vocabulary. Imported on use, never by the gates' CI paths."""
    if str(skill_dir) not in sys.path:
        sys.path.insert(0, str(skill_dir))
    from tests.container import harness as module

    return module


# ---------------------------------------------------------------------------
# The session: one gate invocation's evidence, never a mix of several
# ---------------------------------------------------------------------------

def begin_session(directory: Path, *,
                  fields: Optional[Callable[[datetime.datetime], Mapping[str, Any]]] = None,
                  clear: Sequence[str] = (), create: Sequence[str] = ()) -> Dict[str, Any]:
    """Start a gate session: earlier evidence deleted, a fresh id written.

    Every ``*.json`` in ``directory`` goes, then the sub-directories ``clear``
    names; ``create`` names the ones made empty again. ``fields(moment)``, read
    after the deletion, adds the family's own facts beside ``id`` and
    ``started_at``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    for name in clear:
        shutil.rmtree(directory / name, ignore_errors=True)
    for name in create:
        (directory / name).mkdir()
    moment = now()
    session: Dict[str, Any] = {"id": uuid.uuid4().hex, "started_at": iso(moment)}
    if fields is not None:
        session.update(fields(moment))
    write_json(directory / SESSION_NAME, session)
    return session


def current_session(directory: Path) -> Optional[Dict[str, Any]]:
    """The running session, or ``None`` when there is none (or it has no id)."""
    session = read_json(Path(directory) / SESSION_NAME)
    return session if isinstance(session, dict) and session.get("id") else None


# ---------------------------------------------------------------------------
# One bounded gate step, with its log kept
# ---------------------------------------------------------------------------

def run_step(name: str, argv: Sequence[str], *, timeout: float, grace: float,
             directory: Path, capture: Optional[Path] = None,
             stdin: Optional[Path] = None) -> int:
    """Run one gate step under a wall-clock bound, its output tee'd to
    ``<directory>/logs/<name>.log``. Returns the step's exit, 124 when the bound
    killed it, 127 when it could not start.

    **Its own session, with no terminal.** The child's stdin is ``/dev/null``
    (or the file ``stdin`` names) and it leads a new session, so a timeout kills
    every descendant in one ``killpg``. That is also what makes the bound safe
    for ``opencode run``, which a plain ``timeout N`` wrapper wedges before it
    creates a session (measured: ``timeout`` moves the child into a background
    process group of the same terminal; with no controlling terminal there is
    nothing to be stopped by). ``capture`` keeps the last non-empty stdout line
    (the host-install path, the image id) for the recipe's next line, since
    each recipe line is its own shell.

    **A grace before SIGKILL.** On the bound the group gets SIGTERM, then
    ``grace`` seconds to empty before SIGKILL (:func:`procgroup.stop`,
    ``until="group"``). The family derives it from what its steps must finish:
    the opencode gate from its probes' sweep bound.
    """
    logs = Path(directory) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    started = time.monotonic()
    last_stdout: List[str] = []
    lock = threading.Lock()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(argv)}\n")
        try:
            source = open(stdin, "rb") if stdin is not None else None
        except OSError as exc:
            print(f"release gate: step {name} could not open its stdin {stdin}: {exc}",
                  file=sys.stderr)
            return 127
        try:
            try:
                child = subprocess.Popen(
                    list(argv), stdin=source if source is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=(os.name == "posix"), close_fds=False)
            except OSError as exc:
                print(f"release gate: step {name} could not start {argv[0]!r}: {exc}",
                      file=sys.stderr)
                return 127
        finally:
            if source is not None:
                source.close()

        def pump(stream, sink, is_stdout: bool) -> None:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace")
                with lock:
                    log.write(text)
                    log.flush()
                    sink.write(text)
                    sink.flush()
                    if is_stdout and text.strip():
                        last_stdout[:] = [text.strip()]

        threads = [threading.Thread(target=pump, args=(child.stdout, sys.stdout, True),
                                    daemon=True),
                   threading.Thread(target=pump, args=(child.stderr, sys.stderr, False),
                                    daemon=True)]
        for thread in threads:
            thread.start()
        code: Optional[int]
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            procgroup.stop(child, grace=grace)
            code = None
        except KeyboardInterrupt:
            procgroup.stop(child, grace=grace)
            raise
        for thread in threads:
            thread.join(timeout=grace)
    elapsed = time.monotonic() - started
    if code is None:
        print(f"release gate: step {name} exceeded its {timeout:g} s bound and was killed "
              f"with its process group; its output is in {log_path}", file=sys.stderr)
        return 124
    if code != 0:
        print(f"release gate: step {name} failed (exit {code}) after {elapsed:.0f} s; its "
              f"output is in {log_path}", file=sys.stderr)
        return code
    if capture is not None:
        if not last_stdout:
            print(f"release gate: step {name} printed nothing on stdout to capture into "
                  f"{capture}; its output is in {log_path}", file=sys.stderr)
            return 1
        Path(capture).parent.mkdir(parents=True, exist_ok=True)
        Path(capture).write_text(last_stdout[0] + "\n", encoding="utf-8")
    return 0


def add_step_parser(sub) -> None:
    """The ``step`` subcommand, as both gates' ``main`` declare it."""
    step = sub.add_parser(
        "step", help="run one bounded gate step, keeping its log",
        usage="release_gate.py step NAME --timeout S [--capture F] [--stdin F] -- CMD...")
    step.add_argument("name")
    step.add_argument("--timeout", type=float, required=True)
    step.add_argument("--capture", type=Path, default=None)
    step.add_argument("--stdin", type=Path, default=None,
                      help="a file for the step's stdin (default: /dev/null)")


def split_step_command(words: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(gate words, step command)``: the command is everything after the first
    ``--`` that follows ``step``, split off before argparse sees it (a REMAINDER
    positional would swallow ``--timeout`` too)."""
    words = list(words)
    if "step" in words and "--" in words[words.index("step"):]:
        cut = words.index("--", words.index("step"))
        return words[:cut], words[cut + 1:]
    return words, []
