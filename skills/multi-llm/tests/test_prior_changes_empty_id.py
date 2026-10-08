"""`util prior-changes append` refuses an empty or blank `--id` (fix round 28,
live run 12 #9).

The decline fixture on Claude Code: the host built the id with
`--id "$(python3 -c …)"`, the substitution printed nothing, the append took
`--id ""`, said "Appended entry /apply-fixes", and `prior_changes.jsonl` held
a stray `"id": ""` row beside the real one. The append now writes nothing
for an empty or blank id, exits 3, and says where the id comes from: the
batch's `prior_changes_id` in the phase's `write_tasks.json` (the task id for
implement), naming the batches listed there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import prior_changes  # noqa: E402
from utils.prior_changes import FILENAME  # noqa: E402

BATCHES = [
    {"task_id": "apply-fix-batch-0", "prior_changes_id": "src/opponent.js#306e1bc3"},
    {"task_id": "apply-fix-batch-1", "prior_changes_id": "src/race.js#86ac6ae7"},
]


@pytest.fixture
def plan_dir(tmp_path: Path) -> Path:
    out = tmp_path / "plans" / "kart-sim"
    (out / "apply-fixes").mkdir(parents=True)
    (out / "apply-fixes" / "write_tasks.json").write_text(json.dumps({
        "tasks_schema_version": 2, "kind": "apply_code_fixes", "batches": BATCHES,
    }), encoding="utf-8")
    return out


def _append(out: Path, entry_id: str, *, phase: str = "apply-fixes") -> int:
    return prior_changes.main([
        "append", "--output-dir", str(out), "--id", entry_id, "--phase", phase,
        "--summary", "Fixed opponentLabel to read opponent.driver.",
    ])


@pytest.mark.parametrize("entry_id", ["", "   ", "\t\n"])
def test_an_empty_or_blank_id_writes_nothing_and_says_where_the_id_comes_from(
    plan_dir: Path, entry_id: str, capsys,
) -> None:
    assert _append(plan_dir, entry_id) == prior_changes.EXIT_REFUSED == 3
    assert not (plan_dir / FILENAME).exists()
    err = capsys.readouterr().err
    assert err.startswith("ERROR: util prior-changes append: --id is empty, so nothing was written")
    assert f"prior_changes_id in {plan_dir / 'apply-fixes' / 'write_tasks.json'}" in err
    assert "for implement, the task id" in err
    assert ("The batches there: apply-fix-batch-0: src/opponent.js#306e1bc3; "
            "apply-fix-batch-1: src/race.js#86ac6ae7.") in err
    assert "Appended entry" not in err


def test_the_executor_phase_word_finds_the_same_task_file(plan_dir: Path, capsys) -> None:
    assert _append(plan_dir, "", phase="apply-code-fixes") == 3
    assert "apply-fix-batch-1: src/race.js#86ac6ae7" in capsys.readouterr().err


def test_an_implement_task_file_lists_its_task_ids_once(tmp_path: Path, capsys) -> None:
    out = tmp_path / "plan"
    (out / "implement").mkdir(parents=True)
    (out / "implement" / "write_tasks.json").write_text(json.dumps({"batches": [
        {"task_id": "T001", "prior_changes_id": "T001"},
        {"task_id": "T002", "prior_changes_id": "T002"},
    ]}), encoding="utf-8")
    assert _append(out, "", phase="implement") == 3
    assert "The batches there: T001; T002." in capsys.readouterr().err


def test_no_task_file_still_names_where_the_id_comes_from(tmp_path: Path, capsys) -> None:
    assert _append(tmp_path, "") == 3
    err = capsys.readouterr().err
    assert "prior_changes_id in" in err and "The batches there" not in err
    assert not (tmp_path / FILENAME).exists()


def test_through_the_dispatcher_the_refusal_is_exit_3(plan_dir: Path, capsys) -> None:
    code = multi_llm.main([
        "util", "prior-changes", "append", "--output-dir", str(plan_dir),
        "--id", "", "--phase", "apply-fixes", "--summary", "x",
    ])
    assert code == 3
    assert not (plan_dir / FILENAME).exists()


def test_a_real_id_still_appends(plan_dir: Path, capsys) -> None:
    assert _append(plan_dir, "src/opponent.js#306e1bc3") == 0
    rows = [json.loads(line) for line in (plan_dir / FILENAME).read_text().splitlines()]
    assert [(row["id"], row.get("task_id")) for row in rows] == [
        ("src/opponent.js#306e1bc3", "apply-fix-batch-0")]
    assert "Appended entry src/opponent.js#306e1bc3/apply-fixes" in capsys.readouterr().err
