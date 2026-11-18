"""What a two-batch apply-code-fixes executor run leaves behind, normalised.

``tests/test_scoped_batch_checks.py`` checks ``driver.batch_checks`` absent or
``full`` changes nothing a default run leaves: it compares
:func:`capture_default_run` with the golden file
``tests/golden/scoped_checks_default_run.json``, which this module captured,
through the ``env`` fixture of ``tests.test_write_executor`` and its
``FakeState``, in a checkout of the commit before the setting existed. The
comparison is of the normalised artifacts below, not of bytes, and the
golden covers the ``FakeState`` path only; the same test module compares
absent with ``full`` on the production ``StateManager`` path, on this commit. So this module uses
nothing that commit lacked. To write the golden again after an intended
change to what a default run leaves, run that test module with
``MULTI_LLM_UPDATE_GOLDEN=1`` and ``-k "golden and absent"``.

The run: two batches on ``src/a.py``, each reporting its known checks, with
``driver.verify_command`` a pytest command (a stand-in ``pytest`` on the mock
PATH that records its arguments), so the setting would narrow it. Kept: every
dispatched prompt, ``state.json``, ``write_batch_report.json``, the summary,
the ``.status.json`` mirror, the result's messages and what ``pytest`` was
run with; temp paths, timestamps, durations and pids normalised.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.mock_launchers import write_launcher
from utils.subagent_tasks import (
    PRIOR_CHANGES_TOKEN,
    build_inputs,
    make_write_task,
    write_output_name,
)
from utils.write_batch_scheduler import dispatched_prompt_path

#: The verify command the run is configured with: a pytest command the
#: setting would narrow.
PYTEST_COMMAND = "pytest -q"

#: What each batch reports about the known checks.
BATCH_CHECK = {
    "passed": True, "findings": [],
    "ran": {"pytest -q tests/test_a.py": "passed"},
    "listed": {PYTEST_COMMAND: "passed"},
}

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?s\b")
#: Keys whose value is a clock reading, a duration, a pid or a run id.
_VOLATILE = re.compile(
    r"(_at|_seconds|_s|_ms|duration|elapsed|pid|pgid|started|finished|"
    r"timestamp|run_id|last_updated|deadline|mtime|eta)$"
)


def install_pytest(bin_dir: Path, tmp_path: Path, exit_code: int = 0) -> Dict[str, Path]:
    """A stand-in ``pytest`` on the mock PATH: it appends its arguments to a
    log, one line per run, and exits with the code in a file beside it."""
    log = tmp_path / "pytest-calls.txt"
    code = tmp_path / "pytest-exit.txt"
    code.write_text(str(exit_code), encoding="utf-8")
    write_launcher(
        bin_dir, "pytest",
        "#!/bin/sh\n"
        f"printf 'pytest %s\\n' \"$*\" >> '{log.as_posix()}'\n"
        f"echo 'collected 3 items'\n"
        f"exit $(cat '{code.as_posix()}')\n",
    )
    return {"log": log, "exit": code}


def pytest_calls(paths: Dict[str, Path]) -> List[str]:
    log = paths["log"]
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def fix_task(env: Any, index: int) -> Dict[str, Any]:
    """Apply-code-fixes batch *index* on ``src/a.py``."""
    return make_write_task(
        kind="apply_code_fixes",
        task_id=f"apply-fix-batch-{index}",
        prompt=f"Apply fix {index} to {env.root / 'src/a.py'}\n\n{PRIOR_CHANGES_TOKEN}\n",
        inputs=build_inputs(plan_file=str(env.plan)),
        batch_index=index,
        prior_changes_id=f"src/a.py#{index:08d}",
        phase_dir=str(env.phase_dir),
        declared_files=[str(env.root / "src/a.py")],
        expected_ids=[f"f{index}"],
        output_name=write_output_name("apply_code_fixes", index),
    )


def program_two_batches(env: Any, monkeypatch: Any) -> None:
    """The mock runner's edits and check report for :func:`fix_task` 0 and 1."""
    env.edits([{"path": "src/a.py", "mode": "patch", "content": "a = 2\n"}])
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["f0", "f1"]))
    monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(BATCH_CHECK))


def _normal_text(text: str, roots: List[str]) -> str:
    for root in roots:
        text = text.replace(root, "<TMP>")
    text = _ISO.sub("<TS>", text)
    return _DURATION.sub("<N>s", text)


def normalised(value: Any, roots: List[str]) -> Any:
    """*value* with temp paths, clock readings and durations made portable."""
    if isinstance(value, dict):
        return {
            key: ("<V>" if _VOLATILE.search(str(key)) and not isinstance(item, (dict, list))
                  else normalised(item, roots))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalised(item, roots) for item in value]
    if isinstance(value, str):
        return _normal_text(value, roots)
    if isinstance(value, float):
        return "<F>"
    return value


def capture_default_run(env: Any, monkeypatch: Any, state: Any,
                        *, extra_driver: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run :func:`fix_task` 0 and 1 with ``verify_command: pytest -q`` (and
    *extra_driver*), and return what the run left behind, normalised."""
    # The census of what the phase changed reads the repository at the cwd.
    monkeypatch.chdir(env.root)
    pytest_paths = install_pytest(env.bin_dir, env.tmp_path)
    env.use(verify_command=f"'{PYTEST_COMMAND}'", **(extra_driver or {}))
    program_two_batches(env, monkeypatch)
    summary = env.phase_dir / "summary.md"
    lines: List[str] = []
    result = env.scheduler(
        [fix_task(env, 0), fix_task(env, 1)], kind="apply_code_fixes",
        state=state, summary_file=str(summary), log=lines.append,
    ).run()
    roots = sorted({str(env.tmp_path), str(Path(env.tmp_path).resolve())}, key=len, reverse=True)

    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    prompts = {
        task_id: _normal_text(
            Path(dispatched_prompt_path(str(env.phase_dir), task_id)).read_text(encoding="utf-8"),
            roots)
        for task_id in ("apply-fix-batch-0", "apply-fix-batch-1")
    }
    return {
        "exit_code": result.exit_code,
        "finalized": result.finalized,
        "prompts": prompts,
        "state": normalised(read_json(Path(state.state_file)), roots),
        "report": normalised(read_json(env.phase_dir / "write_batch_report.json"), roots),
        "summary": _normal_text(summary.read_text(encoding="utf-8"), roots)
        if summary.exists() else None,
        "mirror": normalised(read_json(env.phase_dir / ".status.json"), roots),
        "messages": [_normal_text(line, roots) for line in result.messages],
        "verification": normalised(result.verification, roots),
        "pytest_calls": pytest_calls(pytest_paths),
    }
