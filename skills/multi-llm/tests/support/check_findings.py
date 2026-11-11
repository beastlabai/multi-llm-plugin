"""A check that reports findings, in the implement world (``tests.support.implement_world``).

``_check_finds(repo, findings=FINDINGS)`` scripts the mock runner's T002 check
to fail with *findings*. No fixtures: the ``repo`` these files use is
``implement_world.implement_repo``.
"""

from __future__ import annotations

import json

from tests.support.implement_world import CHECK_GATE


FINDINGS = [
    "npx html-validate index.html: index.html:1 doctype must be lowercase",
    "npx html-validate index.html: index.html:9 aria-label misused",
]


def _check_finds(repo, findings=FINDINGS) -> None:
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    repo.monkeypatch.setenv("MOCK_LLM_WRITE_CHECK", json.dumps(
        {"passed": False, "findings": list(findings)}
    ))
