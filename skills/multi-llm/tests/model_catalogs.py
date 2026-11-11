"""Committed model catalog snapshots: where they live, how to read and refresh them.

``tests/test_model_catalogs.py`` checks every model id the base
``providers.yaml`` configures against these snapshots, reading committed files
only: no network, no login, no subprocess. A person refreshes them, logged in
to each CLI whose listing needs it, with ``make refresh-model-catalogs``. It is
never run in CI, because most listings need a login or the network, and it is
never run on the dispatch path, so ``status`` stays subprocess-free.

A snapshot is ``tests/fixtures/model_catalogs/<source>.json``::

    {"captured": "YYYY-MM-DD", "cli_version": "...", "command": "...", "ids": [...]}

``ids`` is sorted and unique. kilocode is the one source stored elsewhere: its
capture, ``tests/fixtures/kilocode/model_variants.json``, also records which
variants each model publishes, and :func:`load_snapshot` reads its keys. It is
refreshed by hand with the commands in its ``_provenance`` block.

The contract is written up in
docs/development/providers.md#model-catalog-snapshots.

Imported by test modules and run as ``python -m tests.model_catalogs`` from
``skills/multi-llm``; not a test module itself.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

TESTS_DIR = Path(__file__).resolve().parent
CATALOG_DIR = TESTS_DIR / "fixtures" / "model_catalogs"
KILOCODE_CAPTURE = TESTS_DIR / "fixtures" / "kilocode" / "model_variants.json"

#: A snapshot older than this many days draws a warning, never a failure: a
#: catalog drifts, but a stale snapshot is still the best offline evidence.
STALE_AFTER_DAYS = 45

#: The snapshot file's closed key set.
SNAPSHOT_KEYS = frozenset({"captured", "cli_version", "command", "ids"})

#: How long one listing may run during a refresh.
LISTING_TIMEOUT_S = 120


@dataclass(frozen=True)
class Snapshot:
    """One source's committed catalog."""

    source: str
    captured: _dt.date
    cli_version: str
    command: str
    ids: FrozenSet[str]
    path: Path

    def age_days(self, today: _dt.date) -> int:
        return (today - self.captured).days


def _parse_codex(stdout: str) -> List[str]:
    """``codex debug models`` prints ``{"models": [{"slug": ...}, ...]}``."""
    return [str(m["slug"]) for m in json.loads(stdout)["models"] if m.get("slug")]


def _parse_opencode(stdout: str) -> List[str]:
    """One ``provider/model`` per line, read by the adapter's own parser."""
    from utils.providers.opencode import _parse_model_listing

    slugs = _parse_model_listing(stdout)
    if slugs is None:
        raise ValueError("not an `opencode models` listing")
    return list(slugs)


def _walk(node: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _parse_omp(stdout: str) -> List[str]:
    """``omp models <provider> --json``: the ``selector`` of every entry.

    The selector (``openrouter/openai/gpt-6-luna``) is what ``--model`` takes;
    ``id`` (``openai/gpt-6-luna``) is not.
    """
    return [str(d["selector"]) for d in _walk(json.loads(stdout)) if d.get("selector")]


def _parse_openrouter(body: str) -> List[str]:
    """``GET /api/v1/models`` returns ``{"data": [{"id": ...}, ...]}``."""
    return [str(m["id"]) for m in json.loads(body)["data"] if m.get("id")]


@dataclass(frozen=True)
class Source:
    """A listing ``make refresh-model-catalogs`` can capture."""

    name: str
    parse: Callable[[str], List[str]]
    #: What a capture needs, printed before it runs.
    needs: str
    command: Tuple[str, ...] = ()
    url: str = ""
    version_cmd: Tuple[str, ...] = ()
    #: A throwaway home for the run, so the listing cannot touch the user's own
    #: config (``CODEX_HOME`` for codex).
    scratch_home_env: str = ""

    @property
    def spelled(self) -> str:
        return self.url or " ".join(self.command)


#: Every source the refresh target knows. A source whose output format has not
#: been captured in this repository has no entry: a guessed parser would write
#: a wrong snapshot. Its providers stay in ``UNCHECKED`` with that reason.
SOURCES: Dict[str, Source] = {
    source.name: source
    for source in (
        Source(
            name="codex",
            command=("codex", "debug", "models", "--bundled"),
            version_cmd=("codex", "--version"),
            parse=_parse_codex,
            needs="nothing: --bundled reads the catalog shipped in the binary",
            scratch_home_env="CODEX_HOME",
        ),
        Source(
            name="opencode",
            command=("opencode", "models"),
            version_cmd=("opencode", "--version"),
            parse=_parse_opencode,
            needs="the network (models.dev) and a login for each provider you list",
        ),
        Source(
            name="omp",
            command=("omp", "models", "openrouter", "--json"),
            version_cmd=("omp", "--version"),
            parse=_parse_omp,
            needs="the network (it refreshes ~/.omp/agent/models.db)",
        ),
        Source(
            name="openrouter",
            url="https://openrouter.ai/api/v1/models",
            parse=_parse_openrouter,
            needs="the network; no key",
        ),
    )
}


def snapshot_path(source: str) -> Path:
    if source == "kilocode":
        return KILOCODE_CAPTURE
    return CATALOG_DIR / f"{source}.json"


def _date(text: str) -> _dt.date:
    return _dt.date.fromisoformat(str(text))


def load_snapshot(source: str) -> Optional[Snapshot]:
    """The committed snapshot for ``source``, or ``None`` when there is none."""
    path = snapshot_path(source)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if source == "kilocode":
        provenance = data["_provenance"]
        return Snapshot(
            source=source,
            captured=_date(provenance["captured"]),
            cli_version=str(provenance["cli_version"]),
            command="; ".join(provenance["commands"]),
            ids=frozenset(data["variants"]),
            path=path,
        )
    return Snapshot(
        source=source,
        captured=_date(data["captured"]),
        cli_version=str(data["cli_version"]),
        command=str(data["command"]),
        ids=frozenset(data["ids"]),
        path=path,
    )


def committed_sources() -> List[str]:
    """Every source with a committed snapshot, kilocode's capture included."""
    names = sorted(path.stem for path in CATALOG_DIR.glob("*.json"))
    if KILOCODE_CAPTURE.is_file():
        names.append("kilocode")
    return sorted(set(names))


def stale_message(snapshot: Snapshot, today: _dt.date) -> Optional[str]:
    """A warning for a snapshot older than :data:`STALE_AFTER_DAYS`, else ``None``."""
    age = snapshot.age_days(today)
    if age <= STALE_AFTER_DAYS:
        return None
    how = (
        f"`make refresh-model-catalogs SOURCES={snapshot.source}`"
        if snapshot.source in SOURCES
        else f"the `_provenance` commands in {snapshot.path.name}"
    )
    return (
        f"model catalog snapshot {snapshot.source!r} is {age} days old "
        f"(captured {snapshot.captured.isoformat()}, over {STALE_AFTER_DAYS}); "
        f"refresh it with {how}"
    )


# ---------------------------------------------------------------------------
# The refresh, run by a person
# ---------------------------------------------------------------------------


def _run(argv: Sequence[str], env: Dict[str, str]) -> str:
    completed = subprocess.run(
        list(argv), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=LISTING_TIMEOUT_S, env=env, check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-400:]
        raise RuntimeError(f"`{' '.join(argv)}` exited {completed.returncode}: {tail}")
    return completed.stdout


def capture(source: Source) -> Dict[str, Any]:
    """Run ``source``'s listing and return the snapshot document."""
    if source.url:
        request = urllib.request.Request(
            source.url, headers={"User-Agent": "multi-llm-catalog-refresh"}
        )
        with urllib.request.urlopen(request, timeout=LISTING_TIMEOUT_S) as response:
            body = response.read().decode("utf-8")
        version = "n/a (HTTP API)"
    else:
        binary = source.command[0]
        if shutil.which(binary) is None:
            raise RuntimeError(f"`{binary}` is not on PATH")
        env = dict(os.environ)
        scratch = None
        if source.scratch_home_env:
            scratch = tempfile.mkdtemp(prefix=f"mllm-catalog-{source.name}-")
            env[source.scratch_home_env] = scratch
        try:
            version = _run(source.version_cmd, env).strip().splitlines()[0]
            body = _run(source.command, env)
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
    ids = sorted(set(source.parse(body)))
    if not ids:
        raise RuntimeError("the listing parsed to no ids; nothing written")
    return {
        "captured": _dt.date.today().isoformat(),
        "cli_version": version,
        "command": source.spelled,
        "ids": ids,
    }


def refresh(names: Sequence[str]) -> int:
    """Capture and write each named source; 1 when any failed."""
    failed = []
    for name in names:
        source = SOURCES[name]
        print(f"{name}: needs {source.needs}; running {source.spelled}")
        try:
            document = capture(source)
        except Exception as exc:  # noqa: BLE001 - report every source, then fail
            failed.append(name)
            print(f"{name}: FAILED, snapshot unchanged: {exc}")
            continue
        old = load_snapshot(name)
        before = old.ids if old else frozenset()
        after = frozenset(document["ids"])
        path = snapshot_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(
            f"{name}: {len(after)} ids written to {path.relative_to(TESTS_DIR.parent)} "
            f"(+{len(after - before)} -{len(before - after)})"
        )
    if failed:
        print(f"failed: {', '.join(failed)}")
    print(
        "Review the diff, then run tests/test_model_catalogs.py; move a provider "
        "out of UNCHECKED there once its source has a snapshot."
    )
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI"):
        print("refresh-model-catalogs is run by a person, never in CI (it needs "
              "logins and the network).")
        return 3
    unknown = [name for name in args if name not in SOURCES]
    if unknown:
        print(f"unknown source(s): {', '.join(unknown)}; known: {', '.join(SOURCES)}")
        return 2
    return refresh(args or list(SOURCES))


if __name__ == "__main__":
    raise SystemExit(main())
