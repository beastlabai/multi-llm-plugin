"""A declined item's reason quotes the decision that already decides it
(fix round 26).

Live run 11, the Codex decline fixture: apply-fix-batch-0 held a fix asking
to clamp opponent speed to MAX_SPEED. The plan's Decisions section says "Do
not add a clamp, cap or `Math.min(…, MAX_SPEED)` to `nextSpeed`", and the
implementation record shown in the prompt says the speed was "deliberately
NOT clamped to MAX_SPEED, per the plan's Decisions". The batch declined it
with "The implementation record says exceeding MAX_SPEED is an intentional
catch-up rule, so this requires a plan decision." — the plan had decided.

The prompts primed that wording: the `declined` bullet listed "one that
needs a plan decision" as a kind of decline, the code-only rule told a batch
to report a fix that would move the plan as "not applied: needs a plan
decision", and the implementation record said to "say so in your returned
summary". Each now tells the batch to name and quote the decision that
already decides against the item, and to say it needs a plan decision only
when nothing does. Both legs bind these prompts (`with_result_contract`, the
code-fix batch prompt in `orchestrator_output.json`).
"""

from __future__ import annotations

import pytest

from utils.code_fix_batcher import (
    CodeFixBatch,
    code_only_rule,
    format_fix_batch_prompt,
)
from utils.prior_changes import append
from utils.write_batch_scheduler import with_result_contract

QUOTE = "name which one and quote that decision"
ONLY_WHEN = "say it needs a plan decision only when nothing there decides it"


def _flat(text: str) -> str:
    return " ".join(text.split())


class TestTheDeclinedBullet:

    @pytest.mark.parametrize("kind, sources", [
        ("apply_code_fixes", "the plan, its task list or the implementation record"),
        ("apply_suggestions", "the plan or its task list"),
        ("apply_task_suggestions", "the plan or its task list"),
    ])
    def test_it_asks_for_the_decision_quoted(self, kind, sources):
        text = _flat(with_result_contract(kind, {"expected_ids": ["aa", "bb"]}, "P"))
        assert f"when {sources} already decides against the" in text
        assert QUOTE in text
        assert ONLY_WHEN in text
        # The old list offered "needs a plan decision" as a kind of decline.
        assert "one that needs a plan decision" not in text

    def test_implement_has_no_declined_bullet(self):
        text = with_result_contract("implement", {"expected_ids": ["T001"]}, "P")
        assert QUOTE not in text


class TestTheCodeFixPrompt:

    def test_the_code_only_rule_quotes_a_decision_before_a_plan_decision(self):
        rule = _flat(code_only_rule("plans/kart-sim.md"))
        assert (
            "When the plan or its task list already decides against the fix "
            "(a rule it states, such as \"do not add X\"), quote that decision "
            "as the reason; report it as \"not applied: needs a plan "
            "decision\" only when they do not decide it."
        ) in rule
        assert "make no edit for that fix and decline it" in rule

    def test_the_implementation_record_says_to_decline_quoting_it(self, tmp_path):
        """The decline fixture's batch 0: the record says the clamp was left
        out on purpose; the batch is told to decline the fix quoting it."""
        append(str(tmp_path), "T001", "implement",
               "Created src/opponent.js. Opponent speed deliberately NOT clamped "
               "to MAX_SPEED, per the plan's Decisions.")
        batch = CodeFixBatch(file_key="src/opponent.js")
        for title in ("Opponent speed exceeds MAX_SPEED",
                      "Standings label reads a missing name field"):
            batch.add({"file": "src/opponent.js", "title": title,
                       "description": "d", "line_range": [15, 19],
                       "importance": "MEDIUM", "type": "bug"})
        prompt = _flat(format_fix_batch_prompt(
            batch, "plans/kart-sim.md", "HEAD~1", output_dir=str(tmp_path)))
        assert "deliberately NOT clamped" in prompt
        assert (
            "If a finding above contradicts a decision recorded here, do NOT "
            "silently revert it: leave the code as it is and decline that fix, "
            "quoting the recorded decision as its reason."
        ) in prompt
        assert "say so in your returned summary" not in prompt
