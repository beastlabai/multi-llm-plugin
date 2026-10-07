"""Every bounded line that shows a model's words cuts them with ``text_clip.clip``.

Fix round 22 (live run 9, #4). ``status``'s progress line cut a declined fix's
reason mid-word ("… so this fix would reverse that dec)") and on OpenCode left
"(catch-up may reach 60 m/s), )"; the finished run's ``notice`` and
``status``'s "check findings for review-code" line cut T002's finding to
"… but scripts/c)", which a Codex host relayed as "truncated at 'but
scripts/c'". The texts below are the run's own (``/tmp/mllm-race-r9/``).
Each row asserts the shown preview is a word-boundary cut ending in "…",
with no separator before the mark and every bracket paired.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import run_status  # noqa: E402
from utils.text_clip import ELLIPSIS, clip, unclosed  # noqa: E402

CODEX_REASON = (
    "The implementation record deliberately allows the catch-up boost to exceed "
    "MAX_SPEED, so this fix would reverse that decision"
)
OPENCODE_REASON = (
    "Implementation record T001 and the plan's Decisions deliberately leave "
    "nextSpeed unclamped (catch-up may reach 60 m/s), and test 'a trailing "
    "opponent may exceed MAX_SPEED' pins it; adding Math.min(..., MAX_SPEED) "
    "would break it"
)
T002_FINDING = (
    "node scripts/check.mjs exited 1: check failed: retries must be 3 (found 2); "
    "src/config.json has retries 2, but scripts/check.mjs requires 3."
)
LONG_FINDINGS = [T002_FINDING, OPENCODE_REASON, CODEX_REASON]


def assert_clean_cut(line: str, source: str) -> None:
    """*line* shows a cut of *source*: ends of words, a mark, paired brackets."""
    assert ELLIPSIS in line, line
    before = line.split(ELLIPSIS)[0]
    # The last kept word is a whole word of the source.
    last = before.split()[-1].lstrip("(\"'")
    assert re.search(rf"(^|\s|\(|\"){re.escape(last)}(\s|$|[,;:.)\"'])", source), (last, line)
    assert before[-1] not in " ,;:—-/(", line
    assert unclosed(line) == [], line


class TestStatusProgressLine:
    def _line(self, reason: str) -> str:
        mirror = {
            "kind": "apply_code_fixes", "total": 2, "ok": 2,
            "items_total": 3, "items_applied": 2,
            "tasks": [{"task_id": "apply-fix-batch-0", "status": "ok"},
                      {"task_id": "apply-fix-batch-1", "status": "ok"}],
            "declined_items": [{"task_id": "apply-fix-batch-0",
                                "id": "c8fe719158c8d2be", "reason": reason}],
        }
        text = " ".join(run_status._mirror_text("write_subagents", mirror))
        assert "declined by its batch: c8fe7191 in apply-fix-batch-0 (" in text
        return text[text.index("declined by its batch"):]

    def test_codex_run_9_declined_reason(self):
        line = self._line(CODEX_REASON)
        assert "reverse that dec)" not in line
        assert line.endswith("so this fix would reverse that…)"), line
        assert_clean_cut(line, CODEX_REASON)

    def test_opencode_run_9_declined_reason_leaves_no_dangling_comma(self):
        line = self._line(OPENCODE_REASON)
        assert ", )" not in line
        assert_clean_cut(line, OPENCODE_REASON)


class TestStatusCheckFindingLines:
    def _payload(self, key: str, findings) -> dict:
        return {"phases": {"implement": {"status": "in_progress", key: [
            {"id": "T002", "findings": list(findings)}]}}}

    def test_check_findings_for_review_code(self):
        lines = run_status._workflow_trailer_lines(
            self._payload("check_findings", [T002_FINDING]), prefix="")
        line = next(item for item in lines if "check findings for review-code" in item)
        assert "but scripts/c)" not in line
        assert line.endswith("has retries 2, but…)"), line
        assert_clean_cut(line, T002_FINDING)

    def test_a_tasks_own_check_quote(self):
        lines = run_status._workflow_trailer_lines(
            self._payload("batch_check_findings", [T002_FINDING]), prefix="")
        line = next(item for item in lines if "own report says a check" in item)
        assert_clean_cut(line, T002_FINDING)

    def test_the_native_implement_block(self):
        view = {"in_progress_since": None, "completed": False,
                "phase_dir": "plans/p/implement",
                "tally": {"done": 1, "total": 3, "buckets": {}},
                "reasons": {}, "answers": {}, "resumable": False,
                "findings": [{"id": "T002", "findings": LONG_FINDINGS}],
                "batch_checks": [{"id": "T003", "findings": LONG_FINDINGS}]}
        lines = run_status.render_state_implement_lines(view)
        for word in ("check findings T002:", "own check fails T003:"):
            line = next(item for item in lines if word in item)
            assert len(line.split(": ", 1)[1]) <= 300
            assert_clean_cut(line, "; ".join(LONG_FINDINGS))

    def test_the_native_apply_block(self):
        view = {"phase_dir": "plans/p/apply-fixes", "completed": True,
                "batches_recorded": 1, "batches_total": 1,
                "batch_checks": [{"id": "apply-fix-batch-0", "findings": LONG_FINDINGS}]}
        lines = run_status.render_state_apply_lines(view)
        line = next(item for item in lines if "own check fails" in item)
        assert_clean_cut(line, "; ".join(LONG_FINDINGS))


def test_clip_keeps_the_run_9_budgets():
    """The budgets stay what they were: 120 for one finding or reason."""
    assert len(clip(CODEX_REASON, 120)) <= 120
