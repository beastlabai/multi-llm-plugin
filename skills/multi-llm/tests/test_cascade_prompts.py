"""A cascade stage's forward ledger in the reviewer, validator and judge prompts.

A cascade stage's review round records a ``cascade`` block in its
``.status.json`` (``utils/cascade_provenance.py``). Every prompt that renders
the earlier rounds' ledger reads that block from the review dir and verifies
the frozen file's hash before a byte of it reaches a model; a file changed
under the stage stops the call at exit 3. Without a block, every prompt is
byte-identical to the one rendered before the ledger existed, which the
goldens pin:

* reviewer: ``{prior_rounds}`` shares the ``{scope_policy_guidance}`` line of
  ``plan_review.txt`` and ``code_review.txt``, empty outside a cascade, the
  frozen ``ledger.md`` after its own blank line inside one;
* validator: the frozen ``ledger-validator.md`` inline (never a file pointer,
  docs/development/delegation.md#materialized-task-contract) with the re-raise
  rule, the optional ``reraise_of`` reply field only then, and the ledger's
  sha256 recorded in the task file's ``inputs``;
* the merge keeps ``reraise_of`` on the verdict, puts it on the group as
  ``validation_reraise_of`` and renders ``re-raises <id>`` in both reports;
* judge: the rule-6 sentence in every cascade stage, the frozen
  ``ledger-judge.md`` list when it is not empty.

The cascade block is built here the way a stage builds it: ledger files in an
``inputs/`` dir, ``write_manifest`` last, ``block_from_manifest``, then the
block written into the review dir's ``.status.json``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
from utils import cascade_provenance as cp  # noqa: E402
from utils import review_lane  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import status_file  # noqa: E402
from utils import subagent_prompts as sp  # noqa: E402
from utils.apply_orchestrator_base import ApplyOrchestratorBase  # noqa: E402
from utils.html_report_generator import _build_group_data  # noqa: E402
from utils.human_decisions import build_judge_tasks, cascade_judge_ledger  # noqa: E402
from utils.prompt_channels import STDOUT_OUTPUT_LOCATION, render_working_limits  # noqa: E402
from utils.scope_policy import reviewer_guidance, validator_guidance  # noqa: E402
from utils.subagent_tasks import write_task_file  # noqa: E402
from utils import validation  # noqa: E402
from tests import test_subagent_prompts as golden_inputs  # noqa: E402

GOLDEN_DIR = SKILL_ROOT / "tests" / "golden" / "prompts"
PROMPTS_DIR = SKILL_ROOT / "prompts"
PLAN_PROMPT, CODE_PROMPT = "plan_review.txt", "code_review.txt"

# ---------------------------------------------------------------------------
# A stage's frozen inputs
# ---------------------------------------------------------------------------

#: The reviewer's block: its overflow line names the stage's ledger-full.md by
#: absolute path ({full}), as reviewers may read files.
REVIEWER_LEDGER = (
    "## Earlier review rounds\n"
    "\n"
    "This plan (or code) has already been through 1 automated review round(s) by\n"
    "other models. Their results are below.\n"
    "\n"
    "Rejected or held back in earlier rounds. Do not raise these again unless you\n"
    "have an argument the recorded reason does not answer; if you do, start the\n"
    "finding's description with \"Re-raises R<n>-<m>:\" and say what is new.\n"
    "- R1-1 [HIGH] (codex:cheap) Add retry to uploads — the plan loses a file on 503"
    " — reason: the plan already retries in Step 4\n"
    "\n"
    "Changed by earlier rounds. These edits were made by other models, not by the\n"
    "author. Review them as you review the rest of the work: they can be wrong, and\n"
    "you may propose correcting or reverting them.\n"
    "- A1-1 [MEDIUM] (codex:cheap) Name the config directory — at: ## Step 1\n"
    "\n"
    "2 more entries are in {full}.\n"
)
#: The validator's lists: same entries, an overflow count and no path.
VALIDATOR_LEDGER = (
    "Rejected or held back in earlier rounds:\n"
    "- R1-1 [HIGH] (codex:cheap) Add retry to uploads — the plan loses a file on 503"
    " — reason: the plan already retries in Step 4\n"
    "\n"
    "Changed by earlier rounds:\n"
    "- A1-1 [MEDIUM] (codex:cheap) Name the config directory — at: ## Step 1\n"
    "\n"
    "2 more entries were left out.\n"
)
#: The judge's list: the rejected or held-back entries alone.
JUDGE_LEDGER = (
    "- R1-1 [HIGH] (codex:cheap) Add retry to uploads — the plan loses a file on 503"
    " — reason: the plan already retries in Step 4\n"
)


def make_block(tmp_path: Path, *, stage: int = 2, empty: bool = False) -> Dict[str, Any]:
    """A stage's ``cascade`` block over frozen inputs written under *tmp_path*.

    ``empty`` writes the first stage's ledger files: present and empty.
    """
    inputs = tmp_path / "cascade" / f"stage-{stage}" / "inputs"
    inputs.mkdir(parents=True)
    full = inputs / cp.FULL_LEDGER
    texts = {
        cp.REVIEWER_LEDGER: "" if empty else REVIEWER_LEDGER.format(full=full),
        cp.VALIDATOR_LEDGER: "" if empty else VALIDATOR_LEDGER,
        cp.JUDGE_LEDGER: "" if empty else JUDGE_LEDGER,
        cp.FULL_LEDGER: "" if empty else "every entry\n",
    }
    for name, text in texts.items():
        (inputs / name).write_bytes(text.encode("utf-8"))
    manifest = cp.write_manifest(
        inputs, cascade_id="c0ffee00", mode="improve-plan", stage=stage, of=3,
        model="codex:mid", names=list(texts),
    )
    return cp.block_from_manifest(manifest)


def record_block(review_dir: Path, block: Optional[Dict[str, Any]]) -> None:
    """The review round's checkpoint, holding *block* (``None``: no cascade)."""
    review_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {"round_id": "a" * 16}
    if block is not None:
        data[cp.STATUS_KEY] = block
    status_file.write_status(str(review_dir), data)


def tamper(block: Dict[str, Any], name: str) -> Path:
    path = cp.input_path(block, name)
    path.write_bytes(path.read_bytes() + b"- R9-9 [LOW] (x) slipped in\n")
    return path


def reviewer_text(block: Dict[str, Any]) -> str:
    return cp.ledger_text(block, cp.REVIEWER_LEDGER)


# ---------------------------------------------------------------------------
# Reviewer prompt
# ---------------------------------------------------------------------------

_CONTEXT = {
    PLAN_PROMPT: ({"plan_path": "/work/plans/p.md", "prefix": "p"}, "review-plan"),
    CODE_PROMPT: ({
        "plan_path": "/work/plans/p.md",
        "tracked_files_list": "- src/app.py",
        "scope_source": "the files the implement phase recorded",
        "base_ref": "0123abcd",
        "prefix": "p",
        "implementation_record": "No implementation record was written.",
        "scope_file": "/work/plans/p/code-review/scope.json",
        "diff_dir": "/work/plans/p/code-review/diff",
        "diff_access": "Each file's diff is under /work/plans/p/code-review/diff.",
    }, "code-review"),
}


def render_review(name: str, prior_rounds: str) -> str:
    """*name* as a reviewer receives it, with the ``{prior_rounds}`` value given."""
    context, phase = _CONTEXT[name]
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").format(
        **context, output_json_path="", output_location=STDOUT_OUTPUT_LOCATION,
        working_limits=render_working_limits(1200),
        scope_policy_guidance=reviewer_guidance("expand", phase),
        prior_rounds=prior_rounds,
    )


def review_golden(name: str) -> str:
    """The rendering recorded before ``{prior_rounds}`` existed."""
    return (GOLDEN_DIR / name.replace(".txt", "_rendered.txt")).read_text(encoding="utf-8")


class TestReviewerTemplates:
    @pytest.mark.parametrize("name", [PLAN_PROMPT, CODE_PROMPT])
    def test_the_placeholder_shares_the_scope_line(self, name):
        template = (PROMPTS_DIR / name).read_text(encoding="utf-8")
        assert template.count("{prior_rounds}") == 1
        assert template.count("\n\n{scope_policy_guidance}{prior_rounds}\n\n## Output Format") == 1

    @pytest.mark.parametrize("name", ["task_review.txt", "commit_review.txt"])
    def test_other_review_templates_do_not_take_it(self, name):
        assert "{prior_rounds}" not in (PROMPTS_DIR / name).read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", [PLAN_PROMPT, CODE_PROMPT])
    @pytest.mark.parametrize("ledger", ["", "\n", "  \n\n"])
    def test_no_ledger_renders_todays_bytes(self, name, ledger):
        assert rob.prior_rounds_block(ledger) == ""
        assert render_review(name, rob.prior_rounds_block(ledger)) == review_golden(name)

    @pytest.mark.parametrize("name", [PLAN_PROMPT, CODE_PROMPT])
    def test_the_ledger_follows_the_scope_block_after_one_blank_line(self, name, tmp_path):
        ledger = reviewer_text(make_block(tmp_path))
        rendered = render_review(name, rob.prior_rounds_block(ledger))
        scope = reviewer_guidance("expand", _CONTEXT[name][1])

        assert f"{scope}\n\n{ledger.strip()}\n\n## Output Format" in rendered
        # Nothing else moved: taking the block out gives today's prompt back.
        assert rendered.replace(f"\n\n{ledger.strip()}", "", 1) == review_golden(name)


class TestWithPriorRounds:
    def test_the_factory_gets_the_block(self):
        factory = rob.with_prior_rounds(lambda spec: {"plan_path": spec}, "## Earlier\n\nx\n")
        assert factory("m:x") == {"plan_path": "m:x", "prior_rounds": "\n\n## Earlier\n\nx"}

    def test_no_ledger_adds_an_empty_key(self):
        assert rob.with_prior_rounds(lambda spec: {}, "")("m:x") == {"prior_rounds": ""}

    def test_a_factorys_own_key_wins(self):
        factory = rob.with_prior_rounds(lambda spec: {"prior_rounds": "own"}, "ledger")
        assert factory("m:x")["prior_rounds"] == "own"


class TestRoundPriorRounds:
    def test_no_checkpoint_and_no_block_read_empty(self, tmp_path):
        assert rob.round_prior_rounds(str(tmp_path / "missing")) == ""
        record_block(tmp_path / "review-plan", None)
        assert rob.round_prior_rounds(str(tmp_path / "review-plan")) == ""

    def test_the_first_stage_reads_empty(self, tmp_path):
        record_block(tmp_path / "review-plan", make_block(tmp_path, stage=1, empty=True))
        assert rob.round_prior_rounds(str(tmp_path / "review-plan")) == ""

    def test_a_later_stage_reads_its_frozen_ledger(self, tmp_path):
        block = make_block(tmp_path)
        record_block(tmp_path / "review-plan", block)
        assert rob.round_prior_rounds(str(tmp_path / "review-plan")) == reviewer_text(block)

    def test_a_changed_ledger_stops_at_exit_3(self, tmp_path, capsys):
        block = make_block(tmp_path)
        record_block(tmp_path / "review-plan", block)
        changed = tamper(block, cp.REVIEWER_LEDGER)

        with pytest.raises(SystemExit) as stop:
            rob.round_prior_rounds(str(tmp_path / "review-plan"))
        assert stop.value.code == 3
        err = capsys.readouterr().err
        assert "ERROR: cascade stage input" in err and str(changed) in err


def _carry_block(block: Dict[str, Any]):
    """``review_lane.open_round`` that then records *block*, as a stage's
    review does once its manifest is verified."""
    original = review_lane.open_round

    def opening(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        status_file.update_status(kwargs["phase_dir"], {cp.STATUS_KEY: block})
        return result

    return patch.object(review_lane, "open_round", opening)


SUGGESTION = {
    "title": "Validate the email field",
    "desc": "The plan adds the field but never says how it is validated.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}


class TestReviewPlanReadsTheBlockAfterTheOpeningCheckpoint:
    """``review_plan_orchestrator.main`` for real; the fan-out is stubbed and
    hands back the context factory the slots would format their prompt with."""

    def _run(self, tmp_path, block: Optional[Dict[str, Any]]):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\n## Overview\n\nAdd an email field.\n", encoding="utf-8")
        seen: List[Any] = []
        spec = "cursor-agent:good"

        async def fake_run_all_models(specs, plan_path, template, factory, *args, **kwargs):
            seen.append((template, factory))
            output = json.dumps([SUGGESTION])
            return {spec: {
                "success": True, "output": output, "error": None, "prompt": "",
                "stdout": output, "stderr": "", "duration_seconds": 0.1,
                "model_spec": spec, "source": "file",
            }}

        argv = ["--plan-file", str(plan), "--models", spec, "--skip-validation",
                "--timeout", "5", "--max-parallel", "1"]
        carry = _carry_block(block) if block is not None else contextlib.nullcontext()
        with patch.object(rob, "run_all_models", new=fake_run_all_models), carry:
            try:
                code = asyncio.run(rpo.main(argv))
            except SystemExit as exc:
                code = exc.code
        return code, seen, spec

    def test_outside_a_cascade_the_block_is_empty(self, tmp_path):
        code, seen, spec = self._run(tmp_path, None)
        assert code == 0
        assert seen[0][1](spec)["prior_rounds"] == ""

    def test_a_stages_reviewers_get_its_ledger_after_the_scope_block(self, tmp_path):
        block = make_block(tmp_path / "frozen")
        code, seen, spec = self._run(tmp_path, block)
        assert code == 0
        template, factory = seen[0]
        context = factory(spec)
        assert context["prior_rounds"] == "\n\n" + reviewer_text(block).strip()
        prompt = template.format(
            **context, output_json_path="", output_location=STDOUT_OUTPUT_LOCATION,
            working_limits=render_working_limits(1200),
        )
        assert (
            f"{reviewer_guidance('expand', 'review-plan')}\n\n"
            f"{reviewer_text(block).strip()}\n\n## Output Format"
        ) in prompt

    def test_a_changed_ledger_stops_before_any_model_runs(self, tmp_path):
        block = make_block(tmp_path / "frozen")
        tamper(block, cp.REVIEWER_LEDGER)
        code, seen, _ = self._run(tmp_path, block)
        assert code == 3
        assert seen == []


class TestReviewCodeReadsTheBlockAfterTheOpeningCheckpoint:
    """``code_review_orchestrator.main`` for real on a two-commit repository;
    only ``invoke_with_provider`` is faked, so the real prompt is formatted."""

    @pytest.fixture(autouse=True)
    def _clean_invocation(self):
        """``run_review`` records the invocation as the dispatcher would; clear it
        on both sides, or a later test in this worker replays this argv
        (``cli_common.current_replay_flags``) in the commands it prints."""
        from utils import cli_common

        cli_common.clear_invocation()
        yield
        cli_common.clear_invocation()

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        from tests.test_review_code_head_ref import lines, new_repo

        repo = new_repo(tmp_path)
        repo.write("a.py", lines("a", 20))
        base = repo.commit("base")
        repo.write("a.py", lines("a", 20, marked=(5,)))
        head = repo.commit("feature")
        monkeypatch.chdir(repo.root)
        repo.base, repo.head = base, head
        repo.argv = ["--base-ref", base, "--head-ref", head, "--skip-validation"]
        return repo

    def _review(self, repo, block: Optional[Dict[str, Any]]):
        from tests.test_review_code_head_ref import Reviewer, review_argv, run_review

        reviewer = Reviewer()
        with _carry_block(block) if block is not None else contextlib.nullcontext():
            code = run_review(review_argv(repo, *repo.argv), reviewer)
        return code, reviewer.prompts

    def test_a_stages_reviewers_get_its_ledger_after_the_scope_block(self, repo, tmp_path):
        block = make_block(tmp_path / "frozen")
        code, prompts = self._review(repo, block)
        assert code == 0 and len(prompts) == 1
        assert (
            f"{reviewer_guidance('expand', 'code-review')}\n\n"
            f"{reviewer_text(block).strip()}\n\n## Output Format"
        ) in prompts[0]

    def test_outside_a_cascade_the_scope_block_is_followed_by_the_output_format(self, repo):
        code, prompts = self._review(repo, None)
        assert code == 0
        assert f"{reviewer_guidance('expand', 'code-review')}\n\n## Output Format" in prompts[0]
        assert "Earlier review rounds" not in prompts[0]

    def test_a_changed_ledger_stops_before_any_model_runs(self, repo, tmp_path):
        block = make_block(tmp_path / "frozen")
        tamper(block, cp.REVIEWER_LEDGER)
        code, prompts = self._review(repo, block)
        assert code == 3
        assert prompts == []


# ---------------------------------------------------------------------------
# Validator prompt and materialization
# ---------------------------------------------------------------------------

def _validation_golden(name: str) -> str:
    return (GOLDEN_DIR / f"{name}.txt").read_text(encoding="utf-8")


class TestValidatorPrompt:
    @pytest.mark.parametrize("name, kwargs", [
        ("validation", {}),
        ("revalidation", {"kind": "revalidation"}),
        ("validation_scope_expand",
         {"scope_guidance": validator_guidance("expand", "review-plan")}),
        ("validation_code_review",
         {"extra_instructions": sp.CODE_REVIEW_VALIDATION_GUIDANCE}),
    ])
    @pytest.mark.parametrize("ledger", ["", "\n"])
    def test_no_ledger_renders_todays_bytes(self, name, kwargs, ledger):
        prompt = sp.build_validation_prompt(
            plan_excerpt=golden_inputs.PLAN_EXCERPT,
            suggestion_records=golden_inputs.SUGGESTION_RECORDS,
            prior_rounds=ledger, **kwargs,
        )
        assert prompt == _validation_golden(name)
        assert "reraise_of" not in prompt

    def _with_ledger(self, **kwargs: Any) -> str:
        return sp.build_validation_prompt(
            plan_excerpt=golden_inputs.PLAN_EXCERPT,
            suggestion_records=golden_inputs.SUGGESTION_RECORDS,
            scope_guidance=validator_guidance("expand", "review-plan"),
            prior_rounds=VALIDATOR_LEDGER, **kwargs,
        )

    def test_the_ledger_and_its_rule_follow_the_scope_block(self):
        prompt = self._with_ledger()
        scope = validator_guidance("expand", "review-plan").strip()
        section = sp.prior_rounds_validator_section(VALIDATOR_LEDGER)

        assert section.startswith("## Earlier review rounds\n\n")
        assert VALIDATOR_LEDGER.strip() in section
        assert section.endswith(sp.PRIOR_ROUNDS_VALIDATOR_RULE)
        assert f"{scope}\n\n{section}\n\n## Reply" in prompt
        assert prompt.index(scope) < prompt.index(section) < prompt.index("## Reply")

    def test_the_rule_is_the_one_the_design_states(self):
        assert sp.PRIOR_ROUNDS_VALIDATOR_RULE == (
            'A group that repeats an entry under "Rejected or held back": judge it on its\n'
            'merits, but classify it "invalid" unless it gives a reason the recorded\n'
            'rejection does not answer. Put the entry\'s id in "reraise_of" either way.\n'
            'A group that challenges an entry under "Changed by earlier rounds" is an\n'
            'ordinary finding.'
        )

    def test_the_reply_contract_gains_reraise_of_only_with_a_ledger(self):
        with_ledger = self._with_ledger()
        plain = _validation_golden("validation_scope_expand")
        field = '"scope_reference": "<baseline section, or empty>", "reraise_of": "<R-id, or empty>"}'

        assert field in with_ledger
        assert "6. `reraise_of` is the id of" in with_ledger
        assert "reraise_of" not in plain
        # The contract is today's plus the field and its rule, nothing else.
        unchanged = with_ledger.replace(', "reraise_of": "<R-id, or empty>"', "")
        unchanged = unchanged.replace(sp._RERAISE_RULE_LINE, "")
        section = sp.prior_rounds_validator_section(VALIDATOR_LEDGER)
        assert unchanged.replace(f"\n{section}\n", "", 1) == plain

    def test_revalidation_gets_the_same_section(self):
        prompt = self._with_ledger(kind="revalidation")
        assert sp.prior_rounds_validator_section(VALIDATOR_LEDGER) in prompt
        assert '"reraise_of": "<R-id, or empty>"' in prompt


def _groups() -> List[Dict[str, Any]]:
    return [
        {
            "theme": theme, "category": "reliability", "models": ["codex:mid"],
            "suggestions": [{
                "title": theme, "desc": f"{theme}: what is wrong.",
                "importance": importance, "type": "addition", "reference": "## Step 1",
            }],
        }
        for theme, importance in (
            ("Retry uploads on 503", "HIGH"),
            ("Say where config lives", "LOW"),
        )
    ]


def _materialize(review_dir: Path, **kwargs: Any) -> Dict[str, Any]:
    review_dir.mkdir(parents=True, exist_ok=True)
    return validation.prepare_batched_validation_tasks(
        groups=_groups(), context="# Plan\n\n## Step 1\nUpload the file.\n",
        output_dir=str(review_dir), plan_file="", **kwargs,
    )


class TestValidationMaterialization:
    def test_a_stages_prompts_carry_the_validator_ledger_inline(self, tmp_path):
        block = make_block(tmp_path)
        review = tmp_path / "plan" / "review-plan"
        record_block(review, block)

        document = _materialize(review)

        assert document["batches"]
        for task in document["batches"]:
            assert sp.prior_rounds_validator_section(VALIDATOR_LEDGER) in task["prompt"]
            assert '"reraise_of": "<R-id, or empty>"' in task["prompt"]
            # Inline only: no pointer to the stage's files, not even the one
            # the reviewer's block names.
            assert task["context_mode"] == "inline"
            assert str(cp.inputs_dir(block)) not in task["prompt"]
            assert cp.FULL_LEDGER not in task["prompt"]
            assert task["inputs"][validation.PRIOR_ROUNDS_INPUT_KEY] == (
                block["files"][cp.VALIDATOR_LEDGER])

    def test_the_ledger_hash_is_recorded_with_the_rounds_inputs(self, tmp_path):
        block = make_block(tmp_path)
        review = tmp_path / "plan" / "review-plan"
        record_block(review, block)

        document = _materialize(review)

        assert document["inputs"][validation.PRIOR_ROUNDS_INPUT_KEY] == (
            block["files"][cp.VALIDATOR_LEDGER])
        # The task file schema accepts it, on write and on read.
        write_task_file(str(review / "validation_tasks.json"), document)

    @pytest.mark.parametrize("stage_one", [False, True])
    def test_no_block_and_the_first_stage_keep_todays_task_file(self, tmp_path, stage_one):
        plain = _materialize(tmp_path / "plain" / "review-plan")
        review = tmp_path / "plan" / "review-plan"
        record_block(review, make_block(tmp_path, stage=1, empty=True) if stage_one else None)

        document = _materialize(review)

        assert validation.PRIOR_ROUNDS_INPUT_KEY not in document["inputs"]
        assert [t["prompt"] for t in document["batches"]] == [
            t["prompt"] for t in plain["batches"]]
        assert all("Earlier review rounds" not in t["prompt"] for t in document["batches"])

    def test_a_changed_validator_ledger_stops_at_exit_3(self, tmp_path, capsys):
        block = make_block(tmp_path)
        review = tmp_path / "plan" / "review-plan"
        record_block(review, block)
        changed = tamper(block, cp.VALIDATOR_LEDGER)

        with pytest.raises(SystemExit) as stop:
            _materialize(review)
        assert stop.value.code == 3
        assert str(changed) in capsys.readouterr().err

    def test_nothing_to_validate_reads_no_ledger(self, tmp_path):
        block = make_block(tmp_path)
        review = tmp_path / "plan" / "review-plan"
        record_block(review, block)
        tamper(block, cp.VALIDATOR_LEDGER)

        document = validation.prepare_batched_validation_tasks(
            groups=[], context="# Plan\n", output_dir=str(review))
        assert document["batches"] == []

    def test_revalidation_carries_it_too(self, tmp_path):
        block = make_block(tmp_path)
        review = tmp_path / "plan" / "review-plan"
        record_block(review, block)
        groups = _groups()
        results = [
            {"group_index": i, "status": "validation_failed", "reason": "timeout",
             "confidence": 0.0, "error_type": "timeout", "recoverable": True}
            for i in range(len(groups))
        ]

        document = validation.prepare_batched_revalidation_tasks(
            groups=groups, validation_results=results, context="# Plan\n",
            output_dir=str(review),
        )

        assert document["batches"]
        for task in document["batches"]:
            assert sp.prior_rounds_validator_section(VALIDATOR_LEDGER) in task["prompt"]


# ---------------------------------------------------------------------------
# The merge carries reraise_of
# ---------------------------------------------------------------------------

def _answer_batches(review: Path, document: Dict[str, Any], replies: Dict[int, Dict[str, Any]]) -> None:
    """Write each batch's reply: group N gets ``replies[N]`` merged into a verdict."""
    for batch in document["batches"]:
        groups = []
        for index, group_id in zip(batch["group_indices"], batch["group_ids"]):
            groups.append({
                "group_index": index, "group_hash": group_id, "status": "invalid",
                "reason": "already answered", "confidence": 0.8,
                **replies.get(index, {}),
            })
        (review / batch["output_name"]).write_text(
            json.dumps({"metadata": {"model": "host"}, "groups": groups}), encoding="utf-8")


class TestTheMergeCarriesReraiseOf:
    def _stage_round(self, tmp_path, replies, *, cascade: bool = True):
        review = tmp_path / "plan" / "review-plan"
        record_block(review, make_block(tmp_path) if cascade else None)
        document = _materialize(review)
        _answer_batches(review, document, replies)
        return review, document

    def test_merge_keeps_and_normalizes_it(self, tmp_path):
        review, document = self._stage_round(
            tmp_path, {0: {"reraise_of": " R1-1 "}, 1: {"reraise_of": ""}})

        records = validation.merge_batched_validation_results(
            str(review), document, total_groups=2, groups=_groups())

        assert records[0]["reraise_of"] == "R1-1"
        assert "reraise_of" not in records[1]

    @pytest.mark.parametrize("value, kept", [
        ("R7-7", "R7-7"),            # an id the ledger never listed
        ("see R1-1\nagain", "see R1-1 again"),
        (3, "3"),
        (["R1-1", "R2-4"], "R1-1, R2-4"),
        ({"id": "R1-1"}, '{"id": "R1-1"}'),
        (None, None), ("   ", None), ([], None), (False, None),
    ])
    def test_an_unknown_or_malformed_id_is_kept_as_text(self, value, kept):
        assert validation.normalize_reraise(value) == kept

    def test_a_round_without_a_ledger_drops_it(self, tmp_path):
        review, document = self._stage_round(
            tmp_path, {0: {"reraise_of": "R1-1"}}, cascade=False)

        records = validation.merge_batched_validation_results(
            str(review), document, total_groups=2, groups=_groups())

        assert all("reraise_of" not in record for record in records)

    def test_merge_validation_batches_writes_it_and_puts_it_on_the_group(self, tmp_path):
        review, document = self._stage_round(tmp_path, {0: {"reraise_of": "R1-1"}})
        validation_path = review / "validation.json"

        groups = rob.merge_validation_batches(
            str(review), document, _groups(), str(validation_path))

        saved = json.loads(validation_path.read_text(encoding="utf-8"))["groups"]
        by_index = {entry["group_index"]: entry for entry in saved}
        assert by_index[0]["reraise_of"] == "R1-1"
        assert "reraise_of" not in by_index[1]
        assert groups[0]["validation_reraise_of"] == "R1-1"
        assert "validation_reraise_of" not in groups[1]

    def test_a_reloaded_validation_json_joins_it_again(self, tmp_path):
        review, document = self._stage_round(tmp_path, {1: {"reraise_of": "R1-1"}})
        validation_path = review / "validation.json"
        rob.merge_validation_batches(str(review), document, _groups(), str(validation_path))

        reloaded = validation.load_validation_results(validation_path)
        groups = validation.apply_validation_to_groups(_groups(), reloaded)

        assert groups[1]["validation_reraise_of"] == "R1-1"
        assert "validation_reraise_of" not in groups[0]


class TestReportsRenderReraises:
    def _group(self, **extra: Any) -> Dict[str, Any]:
        group = dict(_groups()[0])
        group.update({"validation_status": "invalid", "validation_reason": "answered"})
        group.update(extra)
        return group

    def test_report_md_names_the_entry_on_the_validation_line(self):
        lines = rob.format_group(self._group(validation_reraise_of="R1-1"), 1)
        line = next(line for line in lines if line.startswith("**Validation:**"))
        assert line.startswith("**Validation:** Invalid (re-raises R1-1) | **Category:**")

    def test_report_md_without_one_keeps_its_line(self):
        lines = rob.format_group(self._group(), 1)
        line = next(line for line in lines if line.startswith("**Validation:**"))
        assert line.startswith("**Validation:** Invalid | **Category:**")

    def test_the_code_review_report_names_it_beside_the_group_verdict(self):
        issue = {"title": "t", "desc": "d", "file": "a.py", "line_range": [1, 2],
                 "type": "bug", "model": "m"}

        def line(**kwargs: Any) -> str:
            lines = cro._format_issue(1, issue, validation_status="invalid",
                                      validation_reason="r", **kwargs)
            return next(text for text in lines if text.startswith("**Validation:**"))

        assert line().startswith("**Validation:** Invalid | **File:**")
        assert line(siblings=[2]).startswith("**Validation:** Invalid (group verdict) |")
        assert line(reraise_of="R1-1").startswith(
            "**Validation:** Invalid (re-raises R1-1) |")
        assert line(siblings=[2], reraise_of="R1-1").startswith(
            "**Validation:** Invalid (group verdict; re-raises R1-1) |")

    def test_the_html_data_carries_it_only_when_present(self):
        assert _build_group_data(self._group(validation_reraise_of="R1-1"), 0)[
            "validationReraiseOf"] == "R1-1"
        assert "validationReraiseOf" not in _build_group_data(self._group(), 0)

    @pytest.mark.parametrize("template", ["report_template.html", "pr_report_template.html"])
    def test_both_html_templates_render_it(self, template):
        text = (SKILL_ROOT / "templates" / template).read_text(encoding="utf-8")
        assert "group.validationReraiseOf ? ' (re-raises ' + escapeHtml(" in text


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

class TestJudgePrompt:
    def test_outside_a_cascade_the_prompt_keeps_its_bytes(self):
        golden = (GOLDEN_DIR / "human_decision_judge.txt").read_text(encoding="utf-8")
        assert sp.build_human_decision_judge_prompt(golden_inputs.JUDGE_ITEMS) == golden
        assert sp.build_human_decision_judge_prompt(
            golden_inputs.JUDGE_ITEMS, cascade_rejected=None) == golden

    def test_the_first_stage_gets_the_rule_6_sentence_alone(self):
        prompt = sp.build_human_decision_judge_prompt(
            golden_inputs.JUDGE_ITEMS, cascade_rejected="")

        assert sp.CASCADE_JUDGE_RULE6_SENTENCE == (
            "In this run each item comes from one model's round, so rule 6 "
            "(model consensus) does not apply.")
        assert sp.HUMAN_DECISION_JUDGE_RULES in prompt
        assert (f"{sp.HUMAN_DECISION_JUDGE_RULES}\n\n### Earlier review rounds\n\n"
                f"{sp.CASCADE_JUDGE_RULE6_SENTENCE}\n\n## Reply") in prompt
        assert "rejected or held back the entries" not in prompt

    def test_a_later_stage_gets_the_rejected_list_and_the_reraise_rule(self):
        prompt = sp.build_human_decision_judge_prompt(
            golden_inputs.JUDGE_ITEMS, cascade_rejected=JUDGE_LEDGER)
        section = sp.cascade_judge_section(JUDGE_LEDGER)

        assert sp.CASCADE_JUDGE_RULE6_SENTENCE in section
        assert sp.CASCADE_JUDGE_RERAISE_RULE in section
        assert section.endswith(JUDGE_LEDGER)
        assert f"{sp.HUMAN_DECISION_JUDGE_RULES}\n\n{section}\n## Reply" in prompt
        # Nothing else moved.
        golden = (GOLDEN_DIR / "human_decision_judge.txt").read_text(encoding="utf-8")
        assert prompt.replace(f"\n{section}", "", 1) == golden

    def test_the_rules_and_their_reference_section_do_not_change(self):
        reference = (SKILL_ROOT / "references" / "human-decision-batch.md").read_text(
            encoding="utf-8")
        assert sp.extract_reference_section(reference, "judge-rules") == (
            sp.render_reference_section("judge-rules"))

    def test_build_judge_tasks_passes_it_to_every_prompt(self, tmp_path):
        items = [dict(golden_inputs.JUDGE_ITEMS[0])]
        plain = build_judge_tasks(items=items, phase_dir=str(tmp_path))
        cascade = build_judge_tasks(items=items, phase_dir=str(tmp_path),
                                    cascade_rejected=JUDGE_LEDGER)

        assert plain["tasks"][0]["prompt"] == sp.build_human_decision_judge_prompt(items)
        assert cascade["tasks"][0]["prompt"] == sp.build_human_decision_judge_prompt(
            items, cascade_rejected=JUDGE_LEDGER)


class TestTheApplyReadsTheJudgeLedgerFromTheReviewRound:
    def test_no_block_is_none_and_the_first_stage_is_empty(self, tmp_path):
        review = tmp_path / "plan" / "review-plan"
        assert cascade_judge_ledger(str(review)) is None
        record_block(review, None)
        assert cascade_judge_ledger(str(review)) is None
        record_block(review, make_block(tmp_path, stage=1, empty=True))
        assert cascade_judge_ledger(str(review)) == ""

    def test_a_later_stage_reads_its_frozen_list(self, tmp_path):
        review = tmp_path / "plan" / "review-plan"
        record_block(review, make_block(tmp_path))
        assert cascade_judge_ledger(str(review)) == JUDGE_LEDGER

    def test_the_apply_stops_at_exit_3_on_a_changed_list(self, tmp_path, capsys):
        review = tmp_path / "plan" / "review-plan"
        block = make_block(tmp_path)
        record_block(review, block)
        changed = tamper(block, cp.JUDGE_LEDGER)
        apply = types.SimpleNamespace(out_dir=str(tmp_path / "plan"), review_subdir="review-plan")

        with pytest.raises(SystemExit) as stop:
            ApplyOrchestratorBase._cascade_judge_ledger(apply)
        assert stop.value.code == 3
        assert str(changed) in capsys.readouterr().err

    def test_the_apply_reads_it_without_writing(self, tmp_path):
        review = tmp_path / "plan" / "review-plan"
        record_block(review, make_block(tmp_path))
        before = (review / ".status.json").read_bytes()
        apply = types.SimpleNamespace(out_dir=str(tmp_path / "plan"), review_subdir="review-plan")

        assert ApplyOrchestratorBase._cascade_judge_ledger(apply) == JUDGE_LEDGER
        assert (review / ".status.json").read_bytes() == before


class TestTheApplyJudgeHookRendersIt:
    """``ApplyOrchestratorBase._handle_human_decisions`` for real, as the
    judging tests drive it: the judge task file it writes carries the stage's
    block, and a changed list stops it before anything is written."""

    def _orchestrator(self, tmp_path):
        import argparse

        from tests.test_human_decisions import GROUP_A, _group
        from utils.state_manager import StateManager

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()

        class _Orchestrator(ApplyOrchestratorBase):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def get_output_path(self):
                return str(phase_dir / "orchestrator_output.json")

        orch = _Orchestrator(argparse.Namespace(plan_file=str(plan), agent_decide=True, yes=False))
        orch.plan_path = str(plan)
        orch.state = StateManager(plan)
        orch.needs_human = [_group(GROUP_A)]
        # Set by the apply's setup in a real run; never the cwd here.
        orch.out_dir = str(tmp_path / "plan")
        return orch, phase_dir, tmp_path / "plan" / "review-plan"

    def _prompts(self, phase_dir: Path) -> List[str]:
        from utils.human_decisions import JUDGE_TASKS_FILENAME
        from utils.subagent_tasks import tasks_of

        document = json.loads((phase_dir / JUDGE_TASKS_FILENAME).read_text(encoding="utf-8"))
        return [task["prompt"] for task in tasks_of(document)]

    def test_outside_a_cascade_the_judge_prompt_has_no_cascade_section(self, tmp_path):
        orch, phase_dir, _ = self._orchestrator(tmp_path)
        orch._handle_human_decisions()
        prompts = self._prompts(phase_dir)
        assert prompts and all(sp.CASCADE_JUDGE_RULE6_SENTENCE not in p for p in prompts)

    def test_a_stage_judge_gets_the_sentence_and_the_list(self, tmp_path):
        orch, phase_dir, review = self._orchestrator(tmp_path)
        record_block(review, make_block(tmp_path / "frozen"))
        orch._handle_human_decisions()
        prompts = self._prompts(phase_dir)
        assert prompts and all(sp.cascade_judge_section(JUDGE_LEDGER) in p for p in prompts)

    def test_a_changed_list_stops_at_exit_3_before_writing_tasks(self, tmp_path):
        from utils.human_decisions import JUDGE_TASKS_FILENAME

        orch, phase_dir, review = self._orchestrator(tmp_path)
        block = make_block(tmp_path / "frozen")
        record_block(review, block)
        tamper(block, cp.JUDGE_LEDGER)

        with pytest.raises(SystemExit) as stop:
            orch._handle_human_decisions()
        assert stop.value.code == 3
        assert not (phase_dir / JUDGE_TASKS_FILENAME).exists()


class TestAJudgeVerdictBelongsToItsStage:
    """Verdict files are keyed by group id, a content hash: a later stage that
    repeats an earlier stage's finding word for word finds the earlier verdict
    at its own verdict file's name. It is reused only within the stage (and
    judge ledger) its task file was written for; the ledger is checked before
    any verdict is read; outside a cascade nothing changes."""

    def _orchestrator(self, tmp_path: Path, *, yes: bool = False):
        import argparse

        from tests.test_human_decisions import GROUP_A, _group
        from utils.state_manager import StateManager

        plan = tmp_path / "plan.md"
        if not plan.exists():
            plan.write_text("# Plan\n", encoding="utf-8")
        phase_dir = tmp_path / "phase"
        phase_dir.mkdir(exist_ok=True)

        class _Orchestrator(ApplyOrchestratorBase):
            phase_name = "apply-suggestions"
            review_subdir = "review-plan"
            item_noun = "suggestion"
            supports_revalidation = False
            supports_skip_flag = False

            def get_output_path(self):
                return str(phase_dir / "orchestrator_output.json")

        orch = _Orchestrator(argparse.Namespace(plan_file=str(plan), agent_decide=True, yes=yes))
        orch.plan_path = str(plan)
        orch.state = StateManager(plan)
        orch.needs_human = [_group(GROUP_A)]
        orch.out_dir = str(tmp_path / "plan")
        return orch, phase_dir, tmp_path / "plan" / "review-plan"

    @staticmethod
    def _verdict(phase_dir: Path, decision: str, reason: str) -> Path:
        from tests.test_human_decisions import GROUP_A
        from utils.human_decisions import verdict_output_name

        path = phase_dir / verdict_output_name(GROUP_A)
        path.write_text(json.dumps({"schema_version": "1.0", "verdicts": [
            {"group_id": GROUP_A, "decision": decision, "reason": reason}]}), encoding="utf-8")
        return path

    @staticmethod
    def _tasks(phase_dir: Path) -> Dict[str, Any]:
        from utils.human_decisions import JUDGE_TASKS_FILENAME

        return json.loads((phase_dir / JUDGE_TASKS_FILENAME).read_text(encoding="utf-8"))

    def test_a_stage_1_approval_is_not_reused_for_the_same_group_in_stage_2(self, tmp_path):
        from tests.test_human_decisions import GROUP_A
        from utils.human_decisions import (
            CASCADE_STAGE_INPUT_KEY, JUDGE_LEDGER_INPUT_KEY, verdict_files)

        orch, phase_dir, review = self._orchestrator(tmp_path)
        record_block(review, make_block(tmp_path / "s1", stage=1, empty=True))
        orch._handle_human_decisions()
        assert self._tasks(phase_dir)["inputs"][CASCADE_STAGE_INPUT_KEY] == "c0ffee00:1"
        approval = self._verdict(phase_dir, "approve", "approved in stage 1")
        # The stage's own verdict decides its group, as before.
        orch, _, _ = self._orchestrator(tmp_path)
        orch._handle_human_decisions()
        assert [g["group_hash"] for g in orch.valid] == [GROUP_A]

        stage_2 = make_block(tmp_path / "s2", stage=2)
        record_block(review, stage_2)
        orch, _, _ = self._orchestrator(tmp_path)
        orch._handle_human_decisions()
        assert orch.valid == [] and [g["group_hash"] for g in orch.needs_human] == [GROUP_A]
        assert not approval.exists() and verdict_files(str(phase_dir)) == []
        retired = sorted(p.name for p in phase_dir.glob("retired_human_decision_batch_*"))
        assert len(retired) == 1 and ".other-stage." in retired[0]
        tasks = self._tasks(phase_dir)
        assert tasks["inputs"][CASCADE_STAGE_INPUT_KEY] == "c0ffee00:2"
        assert tasks["inputs"][JUDGE_LEDGER_INPUT_KEY] == stage_2["files"][cp.JUDGE_LEDGER]
        assert all(t["inputs"][CASCADE_STAGE_INPUT_KEY] == "c0ffee00:2" for t in tasks["tasks"])
        assert sp.cascade_judge_section(JUDGE_LEDGER) in tasks["tasks"][0]["prompt"]

        # Stage 2's judge decides it against stage 2's ledger.
        self._verdict(phase_dir, "skip", "rejected in an earlier round")
        orch, _, _ = self._orchestrator(tmp_path)
        orch._handle_human_decisions()
        assert orch.valid == [] and [g["group_hash"] for g in orch.skipped] == [GROUP_A]

    def test_another_stages_verdict_is_never_reused_under_yes(self, tmp_path):
        """Unattended, a judging pass that found a verdict skips the rest: an
        earlier stage's verdict must not count as one."""
        from tests.test_human_decisions import GROUP_A
        from utils.human_decisions import JUDGE_TASKS_FILENAME

        orch, phase_dir, review = self._orchestrator(tmp_path, yes=True)
        record_block(review, make_block(tmp_path / "s2", stage=2))
        self._verdict(phase_dir, "approve", "an earlier stage's approval")
        orch._handle_human_decisions()
        assert orch.valid == [] and orch.skipped == []
        assert [g["group_hash"] for g in orch.needs_human] == [GROUP_A]
        assert (phase_dir / JUDGE_TASKS_FILENAME).exists()

    def test_a_changed_judge_ledger_stops_before_an_old_verdict_is_read(self, tmp_path):
        from tests.test_human_decisions import GROUP_A
        from utils.human_decisions import JUDGE_TASKS_FILENAME

        orch, phase_dir, review = self._orchestrator(tmp_path)
        block = make_block(tmp_path / "s2", stage=2)
        record_block(review, block)
        approval = self._verdict(phase_dir, "approve", "an earlier stage's approval")
        before = approval.read_bytes()
        tamper(block, cp.JUDGE_LEDGER)

        with pytest.raises(SystemExit) as stop:
            orch._handle_human_decisions()
        assert stop.value.code == 3
        assert orch.valid == [] and [g["group_hash"] for g in orch.needs_human] == [GROUP_A]
        assert approval.read_bytes() == before, "nothing was moved or read into the run"
        assert not (phase_dir / JUDGE_TASKS_FILENAME).exists()
        assert not orch.state.get_all_human_decisions("apply-suggestions")

    def test_outside_a_cascade_an_earlier_verdict_is_reused_as_before(self, tmp_path):
        from tests.test_human_decisions import GROUP_A

        orch, phase_dir, review = self._orchestrator(tmp_path)
        record_block(review, None)
        approval = self._verdict(phase_dir, "approve", "an earlier run's approval")
        orch._handle_human_decisions()
        assert [g["group_hash"] for g in orch.valid] == [GROUP_A]
        assert approval.exists()
        assert not list(phase_dir.glob("retired_*"))

    def test_outside_a_cascade_the_task_file_gains_no_key(self, tmp_path):
        from utils.human_decisions import CASCADE_STAGE_INPUT_KEY, JUDGE_LEDGER_INPUT_KEY

        orch, phase_dir, review = self._orchestrator(tmp_path)
        record_block(review, None)
        orch._handle_human_decisions()
        tasks = self._tasks(phase_dir)
        for inputs in [tasks["inputs"], *(t["inputs"] for t in tasks["tasks"])]:
            assert CASCADE_STAGE_INPUT_KEY not in inputs
            assert JUDGE_LEDGER_INPUT_KEY not in inputs

    def test_build_judge_tasks_records_the_binding_only_when_given(self, tmp_path):
        from utils.human_decisions import CascadeJudge
        from utils.schema_validator import validate_subagent_task_file

        items = [dict(golden_inputs.JUDGE_ITEMS[0])]
        plain = build_judge_tasks(items=items, phase_dir=str(tmp_path))
        again = build_judge_tasks(items=items, phase_dir=str(tmp_path), cascade_inputs=None)
        assert json.dumps(plain, sort_keys=True) == json.dumps(again, sort_keys=True)
        judge = CascadeJudge(ledger=JUDGE_LEDGER, stage="c0ffee00:2", ledger_sha256="ab" * 32)
        bound = build_judge_tasks(items=items, phase_dir=str(tmp_path),
                                  cascade_rejected=judge.ledger,
                                  cascade_inputs=judge.inputs())
        assert judge.binds(bound) and not judge.binds(plain)
        assert bound["inputs"] == {**plain["inputs"], **judge.inputs()}
        assert validate_subagent_task_file(bound) == []
