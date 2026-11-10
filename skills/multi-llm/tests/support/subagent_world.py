"""The subagent executor's world: mock runners on ``PATH`` and the validation tasks it reads.

``_validation_batch`` writes one materialized validation task as
``utils/validation.py`` does, ``_write_validation_task_file`` a whole task file,
and ``_pid_alive`` asks whether a process still runs.

Fixtures (registered suite-wide by ``tests/conftest.py``): ``runner_env``, a temp
``PATH`` with mock ``codex`` / ``opencode`` / ``claude`` and a driver config.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from utils import subagent_executor as executor
from utils import provider_registry as registry
from utils.subagent_tasks import build_inputs, make_task, make_task_file, write_task_file

from tests.mock_launchers import install_mock_llm, mock_path


def _install_launcher(bin_dir: Path, name: str) -> Path:
    """Put a mock provider CLI called ``name`` on a temp PATH."""
    return install_mock_llm(bin_dir, name)


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """A temp PATH carrying mock `codex`/`opencode`/`claude` plus a driver config.

    Returns a small helper object so each test can pick its runner and its echo
    kind without repeating the wiring.
    """
    bin_dir = tmp_path / "bin"
    for name in ("codex", "opencode", "claude"):
        _install_launcher(bin_dir, name)
    # PATH is REPLACED, not prepended: this suite spawns real subprocesses, and a
    # test that unlinks a mock launcher (the spawn_failed rung) would otherwise
    # fall through to a developer's real `codex`/`claude` and make a live API
    # call. Only the mock bin dir, an interpreter's own dir and where git lives
    # (node too, on Windows) are on it.
    monkeypatch.setenv("PATH", mock_path(bin_dir, ("codex", "opencode", "claude")))
    monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
    monkeypatch.setenv("MULTI_LLM_TEST_FAST_BACKOFF", "1")
    for name in ("codex", "opencode", "claude"):
        resolved = shutil.which(name)
        assert resolved and resolved.startswith(str(bin_dir)), (
            f"{name} resolves to {resolved!r}, outside the mock bin dir — this "
            f"suite must never be able to reach a real provider CLI"
        )
    executor.reset_version_cache()

    class Env:
        bin_dir = None
        config_path = None

        def use(self, runner: str = "codex:gpt-5.5", **driver):
            body = {"subagent_runner": runner, **driver}
            lines = ["driver:"]
            for key, value in body.items():
                rendered = "null" if value is None else (
                    "true" if value is True else
                    "false" if value is False else str(value)
                )
                lines.append(f"  {key}: {rendered}")
            path = tmp_path / "driver-config.yaml"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(path))
            registry._config = None
            self.config_path = path
            return path

        def echo(self, kind: str):
            monkeypatch.setenv("MOCK_LLM_ECHO", kind)

    env = Env()
    env.bin_dir = bin_dir
    yield env
    registry._config = None


def _inputs(phase_dir: Path):
    """The `inputs` block the schema requires for a grouped-suggestion kind."""
    return build_inputs(grouped_file=str(Path(phase_dir) / "grouped.json"))


def _validation_batch(phase_dir: Path, batch_index: int = 0, hashes=("aaaa000000000000",)):
    """One materialized validation task, exactly as `utils/validation.py` writes it."""
    records = [
        {"index": i, "group_hash": h, "group_id": h, "theme": f"Theme {i}",
         "category": "reliability", "models": ["model-a"],
         "suggestions": [{"title": f"S{i}", "desc": "d", "importance": "MEDIUM",
                          "type": "addition"}]}
        for i, h in enumerate(hashes)
    ]
    prompt = (
        "You are validating review suggestions.\n\n"
        + json.dumps(records, indent=2)
        + "\n\nPrint ONE JSON object and nothing else."
    )
    return make_task(
        kind="validation",
        task_id=f"validation-batch-{batch_index}",
        prompt=prompt,
        output_name=f"validation_batch_{batch_index}.json",
        inputs=_inputs(phase_dir),
        phase_dir=str(phase_dir),
        extra={
            "batch_index": batch_index,
            "group_indices": list(range(len(hashes))),
            "group_ids": list(hashes),
            "groups_count": len(hashes),
        },
    )


def _write_validation_task_file(phase_dir: Path, tasks) -> Path:
    document = make_task_file(
        kind="validation",
        tasks=tasks,
        inputs=_inputs(phase_dir),
        task_list_key="batches",
        extra={"total_batches": len(tasks), "phase_dir": str(phase_dir)},
    )
    path = phase_dir / "validation_tasks.json"
    write_task_file(str(path), document)
    return path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True
