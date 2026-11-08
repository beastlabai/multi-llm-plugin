"""The native implement leg's report: a plan whose code task follows the manual one, and the emit and util calls.

``_emit(capsys, *flags)`` runs the native implement emit on ``PLAN_ARG`` and
``_util(capsys, *argv)`` a ``multi_llm.py util`` command, each returning what
they printed.

Fixtures: ``native_repo`` provides ``repo`` (``make_repo`` with that plan and
no host or route in the environment). It is not registered suite-wide, because
``tests.support.gate_fixture`` owns the suite's ``repo``; import it to use it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import multi_llm
from utils import provider_registry as registry

from tests.support.implement_world import PLAN_WITH_CODE_DEPENDENT, make_repo


PLAN_ARG = "plans/checked.md"


@pytest.fixture(name="repo")
def native_repo(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    obj = make_repo(tmp_path, monkeypatch, PLAN_WITH_CODE_DEPENDENT)
    yield obj
    registry._config = None
    registry._config_key = None


def _emit(capsys, *flags: str, yes: bool = True):
    assert multi_llm.main(
        ["implement", PLAN_ARG, "--host", "claude-code",
         *(["--yes"] if yes else ["--no-confirm"]), *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(
        line.split(": ", 1)[1] for line in out.splitlines()
        if line.startswith("Output file: ")
    )
    return json.loads(Path(path).read_text(encoding="utf-8")), out


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err
