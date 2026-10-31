"""Reproduce a review-policy change with disjoint reviewer selections (no spend)."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

SOURCE = Path("/home/marc/code/mllm-wt/scope-policy/skills/multi-llm")
sys.path.insert(0, str(SOURCE))
from tests.test_scope_prompts_and_validation import (
    _review_plan_main, _state, _full_prompt, flat, REVIEWER_EXPAND,
)
from utils import scope_policy as sp

root = Path(tempfile.mkdtemp(prefix="scope-policy-mixing-"))
capture = io.StringIO()
with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
    first = _review_plan_main(root, "preserve", models=("cursor-agent:auto",))
    before = _state(first["plan"])
    second = _review_plan_main(root, "expand", models=("cursor-agent:gpt-5",))
    after = _state(first["plan"])
    # Reselect both reviewers without --force: the preserved first result is
    # reused alongside the expanding result under the new single policy.
    third = _review_plan_main(root, None, models=("cursor-agent:auto", "cursor-agent:gpt-5"))
phase = first["phase_dir"]
summary = {
    "root": str(root),
    "first_exit": first["code"],
    "first_policy": sp.round_record(before, "review-plan")["review_policy"],
    "second_exit": second["code"],
    "second_policy": sp.round_record(after, "review-plan")["review_policy"],
    "second_dispatched": [s["model_spec"] for s in second["sent"]],
    "second_expand_guidance": bool(second["sent"]) and REVIEWER_EXPAND in flat(_full_prompt(second["sent"][0])),
    "reviewer_result_files": sorted(p.name for p in phase.glob("cursor*.json")),
    "conflict_reported": "[SCOPE_POLICY_CONFLICT]" in capture.getvalue(),
    "third_exit": third["code"],
    "third_dispatched": [s["model_spec"] for s in third["sent"]],
    "third_group_models": sorted({m for g in json.loads((phase / "grouped.json").read_text())["groups"] for m in g["models"]}),
    "expected": "exit 3 before dispatch when reviewer policy changes in an unfinished round",
}
(Path(__file__).parent / "policy-mixing-transcript.log").write_text(capture.getvalue())
(Path(__file__).parent / "policy-mixing-result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
