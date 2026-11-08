"""``multi_llm.main`` in-process, from a working directory that is not the repository.

``_run(argv, capsys)`` returns ``(exit code, stdout, stderr, result payload)``;
``_clean_env`` drops the host, route and detach variables a run reads.

Fixtures (registered suite-wide by ``tests/conftest.py``): ``workdir``, the
test's ``tmp_path`` as the cwd with those variables cleared, so ``plans/foo.md``
is a relative plan path.
"""

import pytest

import multi_llm
from utils import cli_common, detach


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """cwd at the fixture root, so ``plans/foo.md`` is a relative plan path."""
    monkeypatch.chdir(tmp_path)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _run(argv, capsys):
    """``multi_llm.main(argv)`` → (exit code, stdout, stderr, result payload)."""
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, cli_common.parse_result_line(captured.out)


def _clean_env(monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
