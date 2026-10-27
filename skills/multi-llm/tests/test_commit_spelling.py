"""The commit spelling hook (plans/review-commits.md §2.8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import commit_spelling as cs  # noqa: E402

BASE = "a1b2c3d4e5f6" + "0" * 28
HEAD = "e4f5a6b7c8d9" + "1" * 28


def _brief(tmp_path: Path, record) -> Path:
    brief = tmp_path / "plans" / "commit-reviews" / "a1b2c3d4e5f6-e4f5a6b7c8d9.md"
    brief.parent.mkdir(parents=True)
    brief.write_text("<!-- multi-llm:commit-brief v1 -->\n", encoding="utf-8")
    out = brief.parent / brief.stem
    out.mkdir()
    state = {"plan_file": str(brief)}
    if record is not None:
        state["commit_range"] = record
    (out / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return brief


def test_a_plan_run_is_never_mapped(tmp_path):
    brief = _brief(tmp_path, None)
    assert cs.commit_view(brief) is None
    assert cs.spell_args("apply-code-fixes", "p.md", ["--resume"], None) == (
        "apply-code-fixes", ["p.md", "--resume"])


def test_a_head_ref_only_state_is_never_mapped():
    assert cs.commit_view({"head_ref": {"base": BASE, "head": HEAD}}) is None


def test_fronted_words_map_back_and_synthesised_flags_go(tmp_path):
    brief = _brief(tmp_path, {"base": BASE, "head": HEAD, "brief": "x.md"})
    view = cs.commit_view(brief)
    assert view is not None and view.pinned == "a1b2c3d4e5f6..e4f5a6b7c8d9"
    word, rest = cs.spell_args(
        "apply-code-fixes", brief,
        ["--plan-file", str(brief), "--base-ref", BASE, f"--head-ref={HEAD}",
         "--commit-stage", "/x/.commit-stage-1.json", "--focus", "f",
         "--context-plan", "docs/x.md", "--resume", "--models", "a", "b"],
        view)
    assert word == "apply-commit-fixes"
    assert rest == ["a1b2c3d4e5f6..e4f5a6b7c8d9", "--resume", "--models", "a", "b"]
    assert cs.spell_mode("review-code", view) == "review-commits"
    assert cs.spell_mode("improve-commits", view) == "improve-commits"
    assert cs.spell_mode("status", view) == "status"


def test_a_root_range_is_typed_back_as_the_brief_path(tmp_path):
    brief = _brief(tmp_path, {"base": None, "head": HEAD})
    view = cs.commit_view(brief)
    assert view.pinned == str(brief)
    command = cs.spell_command("review-code", brief, "--resume", view=view)
    assert " review-commits " in command and str(brief) in command
    assert "--base-ref" not in command and "--plan-file" not in command


def test_the_row_keyed_mapping_only_touches_a_range_composite():
    assert cs.step_word("improve-commits", "review-code") == "review-commits"
    assert cs.step_word("improve-commits", "apply-code-fixes") == "apply-commit-fixes"
    assert cs.step_word("improve-code", "review-code") == "review-code"
    assert cs.step_word("improve-plan", "review-plan") == "review-plan"
    assert cs.step_word("review-code", "review-code") == "review-code"


def test_a_retry_command_is_spelled_for_a_commit_review_only(tmp_path):
    brief = _brief(tmp_path, {"base": BASE, "head": HEAD})
    flags = ["--models", "a", "--base-ref", BASE, "--head-ref", HEAD, "--yes"]
    command = cs.spell_retry_command(
        "review-code", brief, "--reaggregate", typed_plan="typed.md", flags=flags)
    assert " review-commits a1b2c3d4e5f6..e4f5a6b7c8d9 " in command
    assert "--reaggregate" in command and "--models a" in command
    assert "--base-ref" not in command and "--head-ref" not in command
    assert "typed.md" not in command

    plan = _brief(tmp_path / "plan", None)
    from utils.cli_common import retry_command

    assert cs.spell_retry_command(
        "review-code", plan, "--reaggregate", typed_plan="typed.md", flags=flags,
    ) == retry_command("review-code", "typed.md", "--reaggregate", flags=flags)
