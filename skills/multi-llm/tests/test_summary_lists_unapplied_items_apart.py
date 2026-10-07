"""summary.md lists an item its batch did not apply under its own heading,
never under an applied one (fix round 22, live run 9 #2).

The decline fixture: on Claude Code the host followed the runbook's summary
template, which said each declined fix's line sits "in its batch below", so
the forbidden fix was item 1 of "## Applied Fixes" (with "Not applied" text
beside the applied fix 2); the executor's own summary.md listed it under
batch 0 of "## Applied Changes" the same way. The counts were right on both;
the headings said the opposite.

Both legs now give the declined items their own section, "## Declined by
their batch (not applied)", each with its title, id, batch and reason, and
the items skipped at the user's request theirs, "## Skipped at your request
(not applied)"; a batch's own block says how many of its items are listed
there instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import batch_done  # noqa: E402
from utils.subagent_executor import (  # noqa: E402
    DECLINED_SECTION_HEADING,
    SKIPPED_SECTION_HEADING,
)

from tests.test_write_executor_orchestrators import repo  # noqa: E402,F401
from tests.test_write_batch_declined import REASON, _PairRig  # noqa: E402

INSTRUCTIONS = SKILL_DIR / "instructions"


def _section(summary: str, heading: str):
    """The lines of one `## ` section, heading excluded, up to the next."""
    lines = summary.splitlines()
    start = lines.index(f"## {heading}")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    return lines[start + 1:end]


class TestTheExecutorsSummaryListsThemApart(_PairRig):

    def test_a_declined_item_has_its_own_section_and_no_applied_line(self, repo):
        plan = self._review_pair(repo)
        ids = self._ids(plan)
        declined_id, applied_id = ids[0]
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_DECLINED", json.dumps(
            [{"id": declined_id, "reason": REASON}]))
        assert self._run(repo, plan, [applied_id] + ids[1]) == 0
        summary = self._summary(plan)

        assert "- Suggestions declined by their batch (not applied): 1" in summary
        declined = _section(summary, DECLINED_SECTION_HEADING)
        named = [line for line in declined if f"`{declined_id}`" in line]
        assert len(named) == 1, declined
        assert re.fullmatch(
            rf"- \*\*suggestion A[12]\*\* \(`{declined_id}`, apply-batch-0\) — "
            + re.escape(REASON), named[0]), named[0]
        declined_title = named[0].split("**")[1]

        applied = _section(summary, "Applied Changes")
        assert not any(declined_title in line for line in applied), applied
        assert not any("Not applied" in line for line in applied
                       if line.startswith(("1.", "2.", "3."))), applied
        assert (f'**Not applied**: 1 declined by the batch (listed under '
                f'"{DECLINED_SECTION_HEADING}" below)') in applied
        assert any(line.endswith("- Applied") for line in applied), applied
        assert SKIPPED_SECTION_HEADING not in summary

    def test_nothing_declined_adds_no_section(self, repo):
        plan = self._review(repo, "one")
        ids = self._ids(plan)
        assert self._run(repo, plan, sum(ids, [])) == 0
        summary = self._summary(plan)
        assert DECLINED_SECTION_HEADING not in summary
        assert SKIPPED_SECTION_HEADING not in summary
        assert "**Not applied**" not in summary

    def test_an_item_skipped_at_your_request_has_its_own_section(self, repo, capsys):
        plan = self._review_pair(repo)
        ids = self._ids(plan)
        left_out, applied_id = ids[0]
        attested = [applied_id] + ids[1]
        assert self._run(repo, plan, attested) == 3
        assert self._run(
            repo, plan, attested, "--resume", "--human-task", "apply-batch-0=skip",
        ) == 0
        summary = self._summary(plan)

        skipped = _section(summary, SKIPPED_SECTION_HEADING)
        named = [line for line in skipped if f"`{left_out}`" in line]
        assert len(named) == 1, skipped
        assert named[0].startswith("- **suggestion A")
        assert f"(`{left_out}`, apply-batch-0) — --human-task apply-batch-0=skip" in named[0]
        title = named[0].split("**")[1]
        applied = _section(summary, "Applied Changes")
        assert not any(title in line for line in applied), applied
        assert (f'**Not applied**: 1 skipped at your request (listed under '
                f'"{SKIPPED_SECTION_HEADING}" below)') in applied


class TestTheRunbooksSayTheSame:

    @pytest.mark.parametrize("name,applied_heading", [
        ("apply-code-fixes.md", "## Applied Fixes"),
        ("apply-suggestions.md", "## Applied Changes"),
        ("apply-task-suggestions.md", "## Applied Changes"),
    ])
    def test_the_summary_template_has_the_two_sections(self, name, applied_heading):
        text = (INSTRUCTIONS / name).read_text(encoding="utf-8")
        template = text[text.index(applied_heading):]
        assert f"\n## {DECLINED_SECTION_HEADING}\n" in template
        assert f"\n## {SKIPPED_SECTION_HEADING}\n" in template
        # The template's count line no longer puts the item in its batch.
        assert "each one's line in its batch below reads" not in text
        assert f"listed under `## {DECLINED_SECTION_HEADING}` below" in text
        assert f"never under `{applied_heading}`" in text

    def test_the_record_tells_the_host_where_it_goes(self, tmp_path, capsys):
        from tests.test_batch_done import (
            _materialize, _state, _suggestion_batches,
        )
        from utils.state_manager import StateManager

        plan = tmp_path / "plans" / "feature.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Feature\n", encoding="utf-8")
        out = tmp_path / "plans" / "feature"
        out.mkdir()
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        assert batch_done.main([
            "--phase-dir", str(phase_dir), "--task-id", "apply-batch-1",
            "--status", "completed", "--declined", f"g-rate={REASON}",
        ]) == 0
        said = capsys.readouterr().out
        assert (f'List it in the phase\'s summary.md under "## '
                f'{DECLINED_SECTION_HEADING}", with its id and this reason, '
                f'never under the applied ones') in said
        assert "never count it as applied" in said

        assert batch_done.main([
            "--phase-dir", str(phase_dir), "--task-id", "apply-batch-0",
            "--status", "completed", "--skipped", "g-password",
        ]) == 0
        said = capsys.readouterr().out
        assert (f'List it in the phase\'s summary.md under "## '
                f'{SKIPPED_SECTION_HEADING}", with its id, never under the '
                f'applied ones') in said
        processed = _state(out)["processed_apply-suggestions"]
        assert processed["g-rate"]["status"] == "declined"
        assert processed["g-password"]["status"] == "skipped"
