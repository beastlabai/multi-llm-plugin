"""The two scope-policy prompt checks and the validation-side plumbing.

plans/scope-policy-and-report-tags.md §3.1 (Check 1: reviewers prevent new
scope creep; Check 2: validators independently reject leaked extensions) and
§7's "Default compatibility", "Two prompt checks", "Leaked extension",
"Semantics" and "Authority" bullets. These exercise the rendered prompts the
orchestrators really build and the merge the round really runs; the verdicts
are fixtures, so they prove plumbing, not a model's semantic accuracy.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
import review_tasks_orchestrator as rto  # noqa: E402
from utils import check_reply  # noqa: E402
from utils import review_orchestrator_base as base  # noqa: E402
from utils import scope_policy as sp  # noqa: E402
from utils import scope_round  # noqa: E402
from utils import validation as validation_mod  # noqa: E402
from utils.llm_client import STDOUT_OUTPUT_LOCATION, render_working_limits  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.prompt_loader import clear_cache, load_prompt  # noqa: E402
from utils.schema_validator import drop_non_record_entries  # noqa: E402
from utils.state_manager import generate_group_id  # noqa: E402
from utils.subagent_prompts import (  # noqa: E402
    PLAN_VALIDATION_SCOPE_GUIDANCE,
    build_consolidation_prompt,
    build_salvage_prompt,
    build_validation_prompt,
)
from utils.suggestion_processor import SuggestionGroup  # noqa: E402


def flat(text: str) -> str:
    """Prompt text with its hard wrapping collapsed, for phrase checks."""
    return " ".join(text.split())


#: The operative sentences of each block, checked on rendered prompts.
REVIEWER_PRESERVE = (
    "Do not propose new features or product outcomes, unrelated refactors, or "
    "optional cleanup outside that outcome, even when they would be beneficial."
)
REVIEWER_PRESERVE_COVERAGE = (
    "Keep reporting missing or incomplete agreed requirements",
    "Keep reporting existing scope creep",
    "report the blocker with its evidence",
    "This rule takes precedence over the broader checklists above.",
)
REVIEWER_EXPAND = (
    "This classification is informational: do not leave out a finding, change "
    "its importance, or narrow its fix because it extends the scope or because "
    "its scope is uncertain."
)
VALIDATOR_PRESERVE = "It is rejected for this run, even if its facts are correct or its importance is high."
VALIDATOR_EXPAND = (
    "This classification is informational: a group whose issue is real is "
    "\"valid\" whether or not it extends the scope"
)
#: The existing definition of "valid", unchanged under every policy.
VALID_DEFINITION = '- "valid": the issue is real and should be addressed'


def ns_with_policy(namespace: argparse.Namespace, policy: Any) -> argparse.Namespace:
    """The CLI flag is agent B's; tests set it on the namespace directly."""
    if policy is not None:
        namespace.scope_policy = policy
    return namespace


# ---------------------------------------------------------------------------
# Fixtures: a plan, groups, rendered review prompts
# ---------------------------------------------------------------------------


def _plan(tmp_path: Path, text: str = "# Plan\n\n## Output\nExport the report as CSV.\n") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = tmp_path / "feature.md"
    plan.write_text(text, encoding="utf-8")
    return plan


def _reply(model_spec: str, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    provider, _, model = model_spec.partition(":")
    return {
        "success": True, "data": findings, "source": "file", "output_file": "unused.json",
        "details": {"provider": provider, "model": model or "auto"},
    }


#: A reviewer that mislabels an expanding remedy `in_scope` (the leaked case),
#: plus an ordinary in-scope finding.
LEAKED_FINDINGS = [
    {
        "title": "Also export the report as PDF",
        "desc": "Users will want PDF; add a PDF exporter beside the CSV one.",
        "importance": "high", "reference": "## Output", "type": "addition",
        "scope_relation": "in_scope", "scope_reason": "Required for the export goal.",
    },
    {
        "title": "CSV header row is never specified",
        "desc": ("The plan says CSV but never names the columns of the header row, so "
                 "two implementers would write two different files. The Output section "
                 "is the only place the format is named, and it lists no column, no "
                 "order and no quoting rule; name them there so the export is checkable."),
        "importance": "medium", "reference": "## Output", "type": "clarification",
        "scope_relation": "in_scope", "scope_reason": "Completes the CSV requirement.",
    },
]


def _review_plan_main(tmp_path: Path, policy: Any, *, findings=None,
                      extra_argv=(), models=("cursor-agent:auto",)) -> Dict[str, Any]:
    """Drive review-plan's real main to its validation hand-off; capture prompts."""
    plan = tmp_path / "feature.md"
    if not plan.exists():
        _plan(tmp_path)
    argv = ["--plan-file", str(plan), "--models", *models, "--timeout", "5", *extra_argv]
    sent: List[Dict[str, Any]] = []

    def dispatch(**kwargs):
        sent.append(kwargs)
        return _reply(kwargs["model_spec"], list(findings if findings is not None else LEAKED_FINDINGS))

    original = rpo.parse_args
    _release_run_anchor()
    with patch.object(base, "invoke_with_file_output", side_effect=dispatch), \
            patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), \
            patch.object(rpo, "parse_args", lambda a: ns_with_policy(original(a), policy)):
        try:
            code = asyncio.run(rpo.main(argv))
        except SystemExit as exc:
            code = exc.code
    phase_dir = base.get_phase_dir(plan, "review-plan")
    return {"plan": plan, "phase_dir": phase_dir, "code": code, "sent": sent}


def _release_run_anchor() -> None:
    """One process runs one plan; a test that runs two plans releases the pin
    between them, as the suite's autouse fixture does between tests."""
    from utils import provider_registry

    provider_registry._RUN_ANCHOR = None


def _full_prompt(kwargs: Dict[str, Any]) -> str:
    context = dict(kwargs["prompt_context"])
    context.setdefault("output_json_path", "/unused.json")
    context.setdefault("output_location", STDOUT_OUTPUT_LOCATION)
    context.setdefault("working_limits", render_working_limits(kwargs.get("timeout") or 600))
    return kwargs["prompt_template"].format(**context)


def _state(plan: Path) -> Dict[str, Any]:
    return json.loads((output_dir_path(plan) / "state.json").read_text(encoding="utf-8"))


def _task_doc(phase_dir: Path) -> Dict[str, Any]:
    return json.loads((phase_dir / "validation_tasks.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Check 1: the four reviewer prompts
# ---------------------------------------------------------------------------


class TestReviewPlanAndTasksPrompts:
    """The real fan-out renders the round's block into every reviewer prompt."""

    @pytest.mark.parametrize("policy", [None, "expand"])
    def test_default_and_explicit_expand_add_annotation_only(self, tmp_path, policy):
        run = _review_plan_main(tmp_path, policy)
        prompt = _full_prompt(run["sent"][0])
        text = flat(prompt)
        assert REVIEWER_EXPAND in text
        assert "## Scope Policy: Preserve" not in prompt
        assert REVIEWER_PRESERVE not in text
        assert "takes precedence" not in text
        # The existing checklist and its broad criteria stand verbatim.
        template = load_prompt("plan_review.txt")
        head, _, tail = template.partition("{scope_policy_guidance}")
        for section in ("### Premise & Approach", "### Implementation Quality",
                        "- **Scope appropriateness**: Is this plan solving the right problem",
                        "- **Over-engineering**: Is the solution more complex than necessary?",
                        "## Out of Scope: Task Decomposition"):
            assert section in head and section in prompt
        assert '"scope_relation": "in_scope" | "scope_extension" | "unclear"' in prompt

    def test_omitted_and_explicit_expand_render_the_same_prompt(self, tmp_path):
        first = _full_prompt(_review_plan_main(tmp_path / "a", None)["sent"][0])
        second = _full_prompt(_review_plan_main(tmp_path / "b", "expand")["sent"][0])
        assert first.replace(str(tmp_path / "a"), "<R>") == second.replace(str(tmp_path / "b"), "<R>")

    def test_preserve_adds_the_reviewer_suppression_and_keeps_coverage(self, tmp_path):
        run = _review_plan_main(tmp_path, "preserve")
        prompt = _full_prompt(run["sent"][0])
        text = flat(prompt)
        assert "## Scope Policy: Preserve" in prompt
        assert REVIEWER_PRESERVE in text
        for phrase in REVIEWER_PRESERVE_COVERAGE:
            assert phrase in text
        assert REVIEWER_EXPAND not in text
        # The block follows the checklist it overrides, and the checklist stays.
        assert prompt.index("### Implementation Quality") < prompt.index("## Scope Policy: Preserve")
        assert prompt.index("## Scope Policy: Preserve") < prompt.index("## Output Format")

    def test_review_tasks_prompt_judges_tasks_against_the_plan(self, tmp_path):
        plan = _plan(tmp_path)
        tasks = output_dir_path(plan) / "tasks" / "tasks.md"
        tasks.parent.mkdir(parents=True)
        tasks.write_text("# Tasks\n\n## T001 Export CSV\n", encoding="utf-8")
        sent: List[Dict[str, Any]] = []

        def dispatch(**kwargs):
            sent.append(kwargs)
            return _reply(kwargs["model_spec"], [])

        original = rto.parse_args
        for policy in ("preserve", None):
            sent.clear()
            with patch.object(base, "invoke_with_file_output", side_effect=dispatch), \
                    patch.object(base, "PROVIDER_STAGGER_DELAY", 0.0), \
                    patch.object(rto, "parse_args",
                                 lambda a, p=policy: ns_with_policy(original(a), p)):
                assert asyncio.run(rto.main([
                    "--plan-file", str(plan), "--models", "cursor-agent:auto",
                    "--timeout", "5", "--skip-validation", "--force",
                ] + (["--rerun-all"] if policy is None else []))) == 0
            text = flat(_full_prompt(sent[0]))
            assert "does not make the feature in scope by listing it" in text
            assert "### Step 3: Flag Unmapped Requirements (Coverage Gaps)" in _full_prompt(sent[0])
            assert (REVIEWER_PRESERVE in text) is (policy == "preserve")
            assert (REVIEWER_EXPAND in text) is (policy != "preserve")


def _code_slot(tmp_path, monkeypatch, state, policy) -> str:
    """One real ``run_single_review`` slot; returns the prompt it dispatched."""
    from utils import review_orchestrator_base as rob

    class _NoProbe:
        internal_retries = None

        def sandbox_probe_command(self, profile):
            return None

        def sandbox_fallback_rungs(self, profile):
            return ()

    plan = tmp_path / "plans" / "feature.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("<!-- multi-llm:commit-brief v1 -->\n# Brief\n", encoding="utf-8")
    out = output_dir_path(plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps(state), encoding="utf-8")
    seen: List[Dict[str, Any]] = []

    def dispatch(**kwargs):
        seen.append(kwargs)
        return {"success": True, "data": [], "source": "file", "output_file": "", "details": {}}

    monkeypatch.setattr(rob, "get_provider", lambda _name: _NoProbe())
    monkeypatch.setattr("utils.subagent_executor._SANDBOX_PROBE_CACHE", {})
    monkeypatch.setattr(cro, "invoke_with_file_output", dispatch)
    monkeypatch.setattr(cro, "get_project_root", lambda _p: str(tmp_path))
    kwargs = {"scope_policy": policy} if policy is not None else {}
    success, _issues, error = asyncio.run(cro.run_single_review(
        "cursor-agent:auto", plan, tmp_path / "out", timeout=600,
        changed_files=["src/app.py"], **kwargs,
    ))
    assert success, error
    return _full_prompt(seen[0])


def _commit_record() -> Dict[str, Any]:
    head = "b" * 40
    return {
        "base": "a" * 40, "head": head,
        "commits": [{"sha": head, "date": "2026-10-24", "subject": "Add CSV export",
                     "merge": False, "outside_window": False}],
        "phrase": None, "window": None, "notes": [],
        "size": {"commits": 1, "files": 1, "added": 3, "deleted": 0, "lines": 3, "large": False},
        "brief": "plans/feature.md", "plan": None, "plan_snapshot": None,
        "plan_sha256": None, "focus": None, "drifted": [],
    }


class TestCodeAndCommitReviewPrompts:
    @pytest.mark.parametrize("policy", [None, "expand", "preserve"])
    def test_code_review_prompt(self, tmp_path, monkeypatch, policy):
        prompt = _code_slot(tmp_path, monkeypatch, {"tracked_files": []}, policy)
        text = flat(prompt)
        # Every existing "look for" class is still solicited, scope creep included.
        for bullet in ("- **Bugs**: Logic errors", "- **Scope creep**: Code that goes beyond what was planned",
                       "- **Quality issues**: Performance problems, code smells"):
            assert bullet in prompt
        assert "drift in the implementation cannot authorize itself" in text
        assert (REVIEWER_PRESERVE in text) is (policy == "preserve")
        assert (REVIEWER_EXPAND in text) is (policy != "preserve")
        assert '"scope_relation": exactly "in_scope", "scope_extension", or "unclear"' in prompt

    @pytest.mark.parametrize("policy", [None, "preserve"])
    def test_commit_review_prompt_uses_the_brief_as_baseline(self, tmp_path, monkeypatch, policy):
        prompt = _code_slot(tmp_path, monkeypatch, {"commit_range": _commit_record()}, policy)
        text = flat(prompt)
        assert prompt.startswith("You are reviewing the code changes made by a range of git commits.")
        assert "A focus limits where you look; it does not add goals." in text
        assert "too vague to settle a product-scope question" in text
        assert "- **Scope creep**: Changes that no commit message accounts for" in prompt
        assert (REVIEWER_PRESERVE in text) is (policy == "preserve")


class TestPoliciesAlternateInOneProcess:
    """The template cache holds policy-free text; every rendering is per call."""

    def test_alternating_renders_never_leak(self, tmp_path, monkeypatch):
        clear_cache()
        first = _code_slot(tmp_path / "1", monkeypatch, {"tracked_files": []}, None)
        middle = _code_slot(tmp_path / "2", monkeypatch, {"tracked_files": []}, "preserve")
        last = _code_slot(tmp_path / "3", monkeypatch, {"tracked_files": []}, "expand")
        assert REVIEWER_PRESERVE in flat(middle)
        assert REVIEWER_PRESERVE not in flat(first) and REVIEWER_PRESERVE not in flat(last)
        assert first.replace(str(tmp_path / "1"), "<R>") == last.replace(str(tmp_path / "3"), "<R>")
        for name in ("plan_review.txt", "task_review.txt", "code_review.txt", "commit_review.txt"):
            template = load_prompt(name)
            assert template.count("{scope_policy_guidance}") == 1
            assert "Scope Policy: Preserve" not in template
            assert "Scope Classification" not in template


# ---------------------------------------------------------------------------
# Check 2: the validation routes
# ---------------------------------------------------------------------------


def _groups(findings=None) -> List[Dict[str, Any]]:
    """Two groups, the first the leaked extension, stamped like grouped.json."""
    groups = []
    for finding in (findings or LEAKED_FINDINGS):
        member = dict(finding, source_model="cursor-agent:auto")
        groups.append({
            "category": "completeness", "theme": finding["title"],
            "suggestions": [member], "models": ["cursor-agent:auto"],
            "priority_score": 3.0, "validation_status": None, "validation_reason": None,
        })
    for group in groups:
        group["group_hash"] = generate_group_id(group)
    return groups


BASELINE = sp.build_baseline([sp.baseline_source("plan", sha256="c" * 64, snapshot="validation-snapshot.md")],
                             phase="review-plan")


def _block(policy, *, baseline=BASELINE, phase="review-plan"):
    return scope_round.scope_block(policy, phase, source="explicit", baseline=baseline,
                                   round_baseline_id=sp.baseline_id_of(baseline))


def _materialize(tmp_path: Path, policy, groups=None, **kwargs) -> Dict[str, Any]:
    phase_dir = tmp_path / "review-plan"
    phase_dir.mkdir(parents=True, exist_ok=True)
    return validation_mod.prepare_batched_validation_tasks(
        groups=groups or _groups(), context="# Plan\n\n## Output\nExport CSV.\n",
        output_dir=str(phase_dir), plan_file="", phase="review-plan",
        scope=_block(policy) if policy != "omitted" else None, **kwargs,
    )


class TestBatchedValidationPrompts:
    @pytest.mark.parametrize("policy", ["omitted", "expand"])
    def test_expand_keeps_the_valid_definition_and_adds_annotation_only(self, tmp_path, policy):
        document = _materialize(tmp_path, policy)
        for task in document["batches"]:
            text = flat(task["prompt"])
            assert VALID_DEFINITION in task["prompt"]
            assert VALIDATOR_EXPAND in text
            assert VALIDATOR_PRESERVE not in text
            assert "selected_remedy" not in task["prompt"]
            # The plan rule still fills in: the scope block is its own argument.
            assert PLAN_VALIDATION_SCOPE_GUIDANCE in task["prompt"]
            assert '"scope_relation": "in_scope|scope_extension|unclear"' in task["prompt"]
        assert document["scope"]["policy"] == "expand"

    def test_preserve_adds_independent_rejection_beside_the_same_valid_definition(self, tmp_path):
        document = _materialize(tmp_path, "preserve")
        for task in document["batches"]:
            text = flat(task["prompt"])
            assert VALID_DEFINITION in task["prompt"]
            assert VALIDATOR_PRESERVE in text
            assert "Classify independently of the reviewers' own `scope_relation` tags" in text
            assert 'never mark it "invalid" to keep it out' in text
            assert PLAN_VALIDATION_SCOPE_GUIDANCE in task["prompt"]
            # The validator sees the reviewer's provisional tag it must not trust.
            assert '"scope_relation": "in_scope"' in task["prompt"]
        assert document["scope"]["policy"] == "preserve"
        assert document["scope"]["baseline_id"] == BASELINE["baseline_id"]

    def test_both_invokers_run_the_same_bytes(self, tmp_path):
        """The native host reads `prompt_path`; the executor reads `prompt`."""
        document = _materialize(tmp_path, "preserve")
        for task in document["batches"]:
            assert Path(task["prompt_path"]).read_text(encoding="utf-8") == task["prompt"]

    def test_same_batches_under_both_policies(self, tmp_path):
        """No extra validation round or batch per finding (plan §7)."""
        expand = _materialize(tmp_path / "e", "expand")
        preserve = _materialize(tmp_path / "p", "preserve")
        assert expand["total_batches"] == preserve["total_batches"]
        assert [t["group_ids"] for t in expand["batches"]] == [t["group_ids"] for t in preserve["batches"]]

    def test_a_phase_guidance_keeps_its_place(self):
        prompt = build_validation_prompt(
            plan_excerpt="x", suggestion_records=[], extra_instructions="PHASE RULE",
            scope_guidance=sp.validator_guidance("preserve", "code-review"),
        )
        assert prompt.index("PHASE RULE") < prompt.index("## Scope Policy: Preserve") < prompt.index("## Reply")
        assert PLAN_VALIDATION_SCOPE_GUIDANCE not in prompt


def _round_dir(tmp_path: Path, plan_text: str):
    """A plan whose review-plan round recorded its baseline from its snapshot."""
    plan = _plan(tmp_path, plan_text)
    phase_dir = output_dir_path(plan) / "review-plan"
    phase_dir.mkdir(parents=True)
    data = plan.read_bytes()
    (phase_dir / "validation-snapshot.md").write_bytes(data)
    import hashlib

    baseline = sp.build_baseline([sp.baseline_source(
        "plan", sha256=hashlib.sha256(data).hexdigest(), path=str(plan),
        snapshot="validation-snapshot.md")], phase="review-plan")
    state: Dict[str, Any] = {}
    sp.record_review_round(state, "review-plan", "preserve", "explicit", baseline)
    (output_dir_path(plan) / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return plan, phase_dir, baseline


class TestRevalidationPrompts:
    def _revalidate(self, plan, phase_dir, *, current_text=None, policy=None):
        if current_text is not None:
            plan.write_text(current_text, encoding="utf-8")
        document = validation_mod.bind_document_file(
            str(plan), str(phase_dir), label="plan",
            snapshot_name=validation_mod.REVALIDATION_SNAPSHOT_NAME,
        )
        groups = _groups()
        results = [{"group_index": i, "status": "validation_failed", "group_id": g["group_hash"]}
                   for i, g in enumerate(groups)]
        return validation_mod.prepare_batched_revalidation_tasks(
            groups=groups, validation_results=results, context=document.text,
            output_dir=str(phase_dir), plan_file=str(plan), context_document=document,
            phase="review-plan", scope_policy=policy,
        )

    def test_a_changed_document_gets_both_labelled_contexts(self, tmp_path):
        plan, phase_dir, baseline = _round_dir(tmp_path, "# Plan\n\n## Output\nCSV only.\n")
        document = self._revalidate(plan, phase_dir, current_text="# Plan\n\n## Output\nCSV and PDF.\n")
        prompt = document["batches"][0]["prompt"]
        assert "## Context (Plan content)" in prompt and "CSV and PDF." in prompt
        assert f"## Context ({validation_mod.REVALIDATION_BASELINE_LABEL})" in prompt
        assert "CSV only." in prompt
        assert "original review round's baseline" in flat(prompt)
        # The round recorded `preserve`; omission inherits it.
        assert VALIDATOR_PRESERVE in flat(prompt)
        assert document["scope"]["baseline_id"] == baseline["baseline_id"]
        assert document["scope"]["revalidation_baseline_differs"] is True

    def test_an_unchanged_document_needs_no_second_context(self, tmp_path):
        plan, phase_dir, baseline = _round_dir(tmp_path, "# Plan\n\n## Output\nCSV only.\n")
        document = self._revalidate(plan, phase_dir)
        prompt = document["batches"][0]["prompt"]
        assert validation_mod.REVALIDATION_BASELINE_LABEL not in prompt
        assert "original review round's" not in flat(prompt)
        assert document["scope"]["baseline_id"] == baseline["baseline_id"]

    def test_an_explicit_apply_policy_wins_over_the_record(self, tmp_path):
        plan, phase_dir, _baseline = _round_dir(tmp_path, "# Plan\n")
        prompt = self._revalidate(plan, phase_dir, policy="expand")["batches"][0]["prompt"]
        assert VALIDATOR_PRESERVE not in flat(prompt)
        assert VALIDATOR_EXPAND in flat(prompt)

    def test_a_missing_original_snapshot_binds_the_current_document_only(self, tmp_path):
        plan, phase_dir, baseline = _round_dir(tmp_path, "# Plan\n\n## Output\nCSV only.\n")
        (phase_dir / "validation-snapshot.md").unlink()
        document = self._revalidate(plan, phase_dir, current_text="# Plan\n\nchanged\n")
        assert validation_mod.REVALIDATION_BASELINE_LABEL not in document["batches"][0]["prompt"]
        # Bound to another baseline than the round's: those verdicts read unclear.
        assert document["scope"]["baseline_id"] != baseline["baseline_id"]
        assert document["scope"]["round_baseline_id"] == baseline["baseline_id"]


# ---------------------------------------------------------------------------
# The leaked extension, end to end through the merge
# ---------------------------------------------------------------------------


def _merge(tmp_path: Path, policy: str, verdicts) -> Dict[str, Any]:
    groups = _groups()
    document = _materialize(tmp_path, policy, groups=groups)
    phase_dir = Path(document["phase_dir"])
    (phase_dir / "grouped.json").write_text(json.dumps(groups), encoding="utf-8")
    for task in document["batches"]:
        replies = [verdicts(groups[i], i) for i in task["group_indices"]]
        (phase_dir / task["output_name"]).write_text(
            json.dumps({"metadata": {"model": "validator"}, "groups": replies}), encoding="utf-8")
    validation_path = phase_dir / "validation.json"
    validated = base.merge_validation_batches(str(phase_dir), document, groups, str(validation_path))
    return {"validated": validated, "saved": json.loads(validation_path.read_text(encoding="utf-8")),
            "phase_dir": phase_dir}


def _independent_verdict(group, index):
    """The validator: both findings are real; the PDF remedy extends the plan."""
    extension = "PDF" in group["theme"]
    return {
        "group_index": index, "group_hash": group["group_hash"], "status": "valid",
        "reason": "The finding is accurate.", "confidence": 0.9,
        "scope_relation": "scope_extension" if extension else "in_scope",
        "scope_reason": ("Adds a PDF exporter the plan never asks for." if extension
                         else "Names the columns the CSV requirement needs."),
        "scope_reference": "## Output",
        # A model echo of the host stamps is never trusted.
        "scope_baseline_id": "forged", "scope_proposal_digest": "forged",
    }


class TestLeakedExtension:
    def test_preserve_records_the_policy_rejection_and_keeps_validity(self, tmp_path):
        run = _merge(tmp_path, "preserve", _independent_verdict)
        pdf, csv = run["validated"]
        assert pdf["validation_status"] == "valid" and csv["validation_status"] == "valid"
        # The reviewer's provisional tag stays visible on the member.
        assert pdf["suggestions"][0]["scope_relation"] == "in_scope"
        assert pdf[sp.VERDICT_KEY]["relation"] == "scope_extension"
        assert pdf[sp.DECISION_KEY]["outcome"] == "deferred"
        assert pdf[sp.DECISION_KEY]["label"] == "Scope extension"
        assert csv[sp.DECISION_KEY]["outcome"] == "eligible"
        saved = {r["group_id"]: r for r in run["saved"]["groups"]}
        assert saved[pdf["group_hash"]]["status"] == "valid"
        assert saved[pdf["group_hash"]][sp.DECISION_KEY]["eligible"] is False
        assert saved[pdf["group_hash"]]["scope_baseline_id"] == BASELINE["baseline_id"]
        assert saved[pdf["group_hash"]]["scope_proposal_digest"] == sp.proposal_digest(_groups()[0])
        meta = run["saved"]["metadata"]
        assert meta["scope_policy"] == "preserve"
        assert meta["scope_policy_decisions"]["deferred"] == 1
        assert meta["scope_policy_decisions"]["scope_extension"] == 1
        status = json.loads((run["phase_dir"] / ".status.json").read_text(encoding="utf-8"))
        assert status["scope_policy_decisions"]["deferred"] == 1

    def test_the_gate_still_defers_after_a_validity_override(self, tmp_path):
        """Bulk approval or a judge flips validity, never scope authority."""
        pdf = _merge(tmp_path, "preserve", _independent_verdict)["validated"][0]
        approved = dict(pdf, validation_status="valid", validation_override="valid")
        assert not sp.scope_decision(approved, "preserve", BASELINE["baseline_id"]).eligible

    def test_expand_keeps_the_same_correct_extension_eligible(self, tmp_path):
        run = _merge(tmp_path, "expand", _independent_verdict)
        pdf = run["validated"][0]
        assert pdf["validation_status"] == "valid"
        assert pdf[sp.VERDICT_KEY]["relation"] == "scope_extension"
        assert sp.DECISION_KEY not in pdf
        assert sp.scope_decision(pdf, "expand", BASELINE["baseline_id"]).eligible
        assert "scope_policy_decisions" not in run["saved"]["metadata"]

    def test_expand_and_omission_decide_the_same_validity(self, tmp_path):
        expand = _merge(tmp_path / "e", "expand", _independent_verdict)["validated"]
        omitted = _merge(tmp_path / "o", "omitted", _independent_verdict)["validated"]
        keys = ("validation_status", "validation_reason", "validation_confidence", "group_hash")
        assert [{k: g.get(k) for k in keys} for g in expand] == [{k: g.get(k) for k in keys} for g in omitted]

    def test_a_reply_without_scope_fields_is_a_valid_reply_and_reads_unclear(self, tmp_path):
        def bare(group, index):
            return {"group_index": index, "group_hash": group["group_hash"], "status": "valid",
                    "reason": "ok", "confidence": 0.8}

        run = _merge(tmp_path, "preserve", bare)
        for group in run["validated"]:
            assert group["validation_status"] == "valid"
            assert sp.authority(group, BASELINE["baseline_id"]).relation == "unclear"
            assert group[sp.DECISION_KEY]["outcome"] == "deferred"

    def test_malformed_scope_never_costs_the_verdict(self, tmp_path):
        def malformed(group, index):
            return {"group_index": index, "group_hash": group["group_hash"], "status": "invalid",
                    "reason": "false positive", "confidence": 0.7,
                    "scope_relation": {"weird": True}, "scope_reason": ["x"]}

        run = _merge(tmp_path, "expand", malformed)
        for group in run["validated"]:
            assert group["validation_status"] == "invalid"
            assert group[sp.VERDICT_KEY]["relation"] == "unclear"


# ---------------------------------------------------------------------------
# Authority: hashes only, never positions
# ---------------------------------------------------------------------------


class TestAuthority:
    def test_index_fallback_transfers_validity_but_never_scope(self, tmp_path):
        def unhashed(group, index):
            verdict = _independent_verdict(group, index)
            verdict.pop("group_hash")
            verdict["scope_relation"] = "in_scope"
            verdict["scope_reason"] = "trust me"
            return verdict

        run = _merge(tmp_path, "preserve", unhashed)
        for group in run["validated"]:
            assert group["validation_status"] == "valid"
            assert sp.VERDICT_KEY not in group
            assert group[sp.DECISION_KEY]["outcome"] == "deferred"
        for record in run["saved"]["groups"]:
            assert "scope_proposal_digest" not in record
            assert "scope_baseline_id" not in record

    def test_a_foreign_hash_is_discarded(self, tmp_path):
        def foreign(group, index):
            verdict = _independent_verdict(group, index)
            verdict["group_hash"] = "f" * 16
            return verdict

        for group in _merge(tmp_path, "preserve", foreign)["validated"]:
            assert sp.VERDICT_KEY not in group

    def test_shuffled_records_join_each_group_to_its_own_verdict(self, tmp_path):
        run = _merge(tmp_path, "expand", _independent_verdict)
        records = list(reversed(run["saved"]["groups"]))
        for record in records:
            record["group_index"] = 99  # a position that names nothing
        rejoined = validation_mod.apply_validation_to_groups(_groups(), records)
        by_theme = {g["theme"]: g[sp.VERDICT_KEY]["relation"] for g in rejoined}
        assert by_theme == {"Also export the report as PDF": "scope_extension",
                            "CSV header row is never specified": "in_scope"}

    def test_a_hashless_legacy_record_cannot_establish_authority(self):
        groups = _groups()
        legacy = [{"group_index": 0, "status": "valid", "reason": "ok",
                   "scope_relation": "in_scope", "scope_reason": "r",
                   "scope_proposal_digest": sp.proposal_digest(groups[0]),
                   "scope_baseline_id": BASELINE["baseline_id"]}]
        joined = validation_mod.apply_validation_to_groups(groups, legacy)
        assert joined[0]["validation_status"] == "valid"
        assert sp.VERDICT_KEY not in joined[0]

    def test_an_edited_proposal_reads_stale(self, tmp_path):
        """An edit past the stable id's truncated prefix keeps the hash join
        (and validity) but not the scope verdict's binding."""
        run = _merge(tmp_path, "expand", _independent_verdict)
        records = run["saved"]["groups"]
        edited = _groups()
        edited[1]["suggestions"][0]["desc"] += " " + "x" * 400 + " Also add a PDF exporter."
        assert generate_group_id(edited[1]) == generate_group_id(_groups()[1])
        joined = validation_mod.apply_validation_to_groups(edited, records)
        assert joined[1]["validation_status"] == "valid"
        assert sp.authority(joined[1], BASELINE["baseline_id"]).state == "stale"

    def test_metadata_only_changes_keep_stable_ids(self, tmp_path):
        plain = _groups()
        validated = _merge(tmp_path, "preserve", _independent_verdict)["validated"]
        assert [generate_group_id(g) for g in validated] == [generate_group_id(g) for g in plain]
        normalized = drop_non_record_entries(
            [dict(LEAKED_FINDINGS[0], scope_relation="In-Scope")], "m", noun="suggestions")
        assert normalized[0]["scope_relation"] == "in_scope"
        assert generate_group_id({"theme": "t", "suggestions": normalized}) == generate_group_id(
            {"theme": "t", "suggestions": [{k: v for k, v in LEAKED_FINDINGS[0].items()
                                            if not k.startswith("scope_")}]})


class TestSalvageKeepsTheScopeTags:
    @pytest.mark.parametrize("phase_name", ["review_plan", "review_tasks"])
    def test_a_review_salvage_request_lists_the_scope_tags_as_optional(self, tmp_path, phase_name):
        """The salvage subagent rebuilds a reviewer's answer from the expected
        shape: the provisional tags ride along as optional fields, while the
        required fields (and so what salvage drops) stay as they were."""
        path = base._write_salvage_request(
            model="cursor-agent:auto", sanitized_model="cursor-agent_auto",
            raw_output='[{"title": "t", "desc": "d", "scope_relation": "in_scope",}]',
            phase_dir=str(tmp_path), phase_name=phase_name,
        )
        request = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = request["expected_schema"]
        assert schema["fields"] == ["title", "desc", "importance", "reference", "type"]
        assert schema["optional_fields"] == ["scope_relation", "scope_reason", "scope_reference"]
        prompt = build_salvage_prompt(request)
        shape = prompt[prompt.index("## Expected shape"):prompt.index("## Raw output to salvage")]
        assert '"optional_fields": [' in shape and '"scope_reference"' in shape


class TestSerializers:
    def test_suggestion_group_round_trips_scope_keys(self):
        group = SuggestionGroup("completeness", "Export")
        group.scope_verdict = {"relation": "in_scope", "reason": "r", "proposal_digest": "d"}
        group.scope_policy_decision = {"policy": "preserve", "eligible": True}
        again = SuggestionGroup.from_dict(group.to_dict())
        assert again.scope_verdict == group.scope_verdict
        assert again.scope_policy_decision == group.scope_policy_decision

    def test_a_legacy_group_serializes_as_before(self):
        group = SuggestionGroup("completeness", "Export")
        assert "scope_verdict" not in group.to_dict()
        assert SuggestionGroup.from_dict(group.to_dict()).scope_verdict is None

    def test_check_reply_accepts_replies_with_and_without_scope(self, tmp_path):
        task = {"task_id": "validation-batch-0", "kind": "validation",
                "output_name": "validation_batch_0.json", "group_ids": ["a" * 16, "b" * 16]}
        reply = {"groups": [
            {"group_hash": "a" * 16, "status": "valid", "scope_relation": "garbage"},
            {"group_hash": "b" * 16, "status": "invalid"},
        ]}
        (tmp_path / "validation_batch_0.json").write_text(json.dumps(reply), encoding="utf-8")
        assert check_reply.check_task_reply(task, str(tmp_path)) == []

    def test_the_consolidator_is_no_scope_authority(self):
        """Consolidation records carry titles and descriptions only."""
        from utils.consolidation import build_consolidation_group_records

        groups = _groups()
        records = build_consolidation_group_records(groups, [0, 1], [g["group_hash"] for g in groups])
        prompt = build_consolidation_prompt(
            template=load_prompt("consolidate_suggestions.txt"), group_records=records)
        assert "scope_relation" not in prompt and "scope_extension" not in prompt

    def test_reviewer_schemas_accept_optional_and_malformed_scope(self):
        from utils.schema_validator import validate_document

        for name, finding in (
            ("review_suggestions.schema.json",
             {k: v for k, v in LEAKED_FINDINGS[0].items()}),
            ("code_review_issues.schema.json",
             {"title": "t", "desc": "d", "importance": "low", "file": "a.py", "type": "bug",
              "scope_relation": 7, "scope_reason": None}),
        ):
            schema = json.loads((SKILL_ROOT / "schemas" / name).read_text(encoding="utf-8"))
            assert validate_document([finding], schema) == []
            bare = {k: v for k, v in finding.items() if not k.startswith("scope_")}
            assert validate_document([bare], schema) == []


# ---------------------------------------------------------------------------
# The round: one policy and one baseline for both roles
# ---------------------------------------------------------------------------


class TestTheRoundBindsBothRoles:
    def test_reviewers_and_validators_share_policy_and_baseline(self, tmp_path):
        run = _review_plan_main(tmp_path, "preserve")
        assert run["code"] == 0
        reviewer = flat(_full_prompt(run["sent"][0]))
        assert REVIEWER_PRESERVE in reviewer
        document = _task_doc(run["phase_dir"])
        for task in document["batches"]:
            assert VALIDATOR_PRESERVE in flat(task["prompt"])
            assert Path(task["prompt_path"]).read_text(encoding="utf-8") == task["prompt"]
        record = sp.round_record(_state(run["plan"]), "review-plan")
        assert record["review_policy"] == "preserve"
        assert record["review_policy_source"] == "explicit"
        snapshot = run["phase_dir"] / "validation-snapshot.md"
        import hashlib

        source = record["baseline"]["sources"][0]
        assert source["role"] == "plan" and source["snapshot"] == "validation-snapshot.md"
        assert source["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert document["scope"]["baseline_id"] == record["baseline"]["baseline_id"]
        assert document["scope"]["round_baseline_id"] == record["baseline"]["baseline_id"]

    def test_no_extra_model_calls(self, tmp_path):
        models = ("cursor-agent:auto", "cursor-agent:gpt-5")
        expand = _review_plan_main(tmp_path / "e", None, models=models)
        preserve = _review_plan_main(tmp_path / "p", "preserve", models=models)
        assert len(expand["sent"]) == len(preserve["sent"]) == 2
        assert (_task_doc(expand["phase_dir"])["total_batches"]
                == _task_doc(preserve["phase_dir"])["total_batches"])

    def test_a_resume_inherits_and_an_explicit_change_is_refused(self, tmp_path, capsys):
        first = _review_plan_main(tmp_path, "preserve")
        assert first["code"] == 0
        # Same round, policy omitted: the recorded one stays.
        again = _review_plan_main(tmp_path, None, extra_argv=("--reaggregate",))
        assert sp.round_record(_state(first["plan"]), "review-plan")["review_policy"] == "preserve"
        assert again["code"] in (0, None)
        # An explicit change on a continuation of that round: exit 3, named control.
        capsys.readouterr()
        refused = _review_plan_main(tmp_path, "expand", extra_argv=("--reaggregate",))
        assert refused["code"] == 3
        out = capsys.readouterr().out
        assert "[SCOPE_POLICY_CONFLICT]" in out and "--force --rerun-all" in out

    def test_a_forced_new_review_without_the_flag_is_default_expand(self, tmp_path):
        first = _review_plan_main(tmp_path, "preserve", extra_argv=("--skip-validation",))
        assert first["code"] == 0
        forced = _review_plan_main(tmp_path, None,
                                   extra_argv=("--skip-validation", "--force", "--rerun-all"))
        assert forced["code"] == 0
        record = sp.round_record(_state(first["plan"]), "review-plan")
        assert record["review_policy"] == "expand"
        assert record["review_policy_source"] == "default"
        assert REVIEWER_EXPAND in flat(_full_prompt(forced["sent"][0]))

    @pytest.mark.parametrize("force", [False, True])
    @pytest.mark.parametrize("old,new", [("preserve", "expand"), ("expand", "preserve")])
    def test_a_disjoint_reviewer_selection_cannot_change_policy(self, tmp_path, capsys,
                                                               force, old, new):
        first = _review_plan_main(tmp_path, old, models=("cursor-agent:auto",))
        before = sp.round_record(_state(first["plan"]), "review-plan")
        capsys.readouterr()
        second = _review_plan_main(tmp_path, new, models=("cursor-agent:gpt-5",),
                                  extra_argv=("--force",) if force else ())
        assert second["code"] == 3
        assert second["sent"] == []
        assert "[SCOPE_POLICY_CONFLICT]" in capsys.readouterr().out
        assert sp.round_record(_state(first["plan"]), "review-plan") == before

    def test_a_new_policy_round_retires_unselected_cached_reviewers(self, tmp_path):
        first = _review_plan_main(tmp_path, "preserve", models=("cursor-agent:auto",))
        old_result = first["phase_dir"] / "cursor-agent_auto.json"
        original = old_result.read_bytes()
        second = _review_plan_main(tmp_path, "expand", models=("cursor-agent:gpt-5",),
                                  extra_argv=("--force", "--rerun-all"))
        assert second["code"] == 0
        assert not old_result.exists()
        assert any(p.read_bytes() == original
                   for p in first["phase_dir"].glob("cursor-agent_auto.json.*"))
        third = _review_plan_main(tmp_path, None,
                                 models=("cursor-agent:auto", "cursor-agent:gpt-5"))
        assert third["code"] == 0
        assert [s["model_spec"] for s in third["sent"]] == ["cursor-agent:auto"]
        assert REVIEWER_EXPAND in flat(_full_prompt(third["sent"][0]))

    @pytest.mark.parametrize("policy", [None, "preserve"])
    def test_a_disjoint_selection_can_continue_under_the_original_policy(self, tmp_path, policy):
        first = _review_plan_main(tmp_path, "preserve", models=("cursor-agent:auto",))
        second = _review_plan_main(tmp_path, policy, models=("cursor-agent:gpt-5",))
        assert second["code"] == 0
        assert REVIEWER_PRESERVE in flat(_full_prompt(second["sent"][0]))
        assert (first["phase_dir"] / "cursor-agent_auto.json").exists()

    def test_failed_cache_retirement_refuses_the_new_policy_before_dispatch(self, tmp_path):
        first = _review_plan_main(tmp_path, "preserve", models=("cursor-agent:auto",))
        before = sp.round_record(_state(first["plan"]), "review-plan")
        with patch.object(base._result_files, "set_aside_stale_result", return_value=None):
            second = _review_plan_main(tmp_path, "expand", models=("cursor-agent:gpt-5",),
                                      extra_argv=("--force", "--rerun-all"))
        assert second["code"] == 3
        assert second["sent"] == []
        assert sp.round_record(_state(first["plan"]), "review-plan") == before
        assert (first["phase_dir"] / "cursor-agent_auto.json").exists()

    def test_a_forced_round_never_reuses_results_of_another_policy(self, tmp_path, capsys):
        """A plain --force keeps the policy of the results it reuses (with one
        notice); a different policy typed for that reuse is still refused."""
        models = ("cursor-agent:auto", "cursor-agent:gpt-5")
        first = _review_plan_main(tmp_path, "preserve", models=models[:1],
                                  extra_argv=("--skip-validation",))
        assert first["code"] == 0
        capsys.readouterr()
        reused = _review_plan_main(tmp_path, None, models=models,
                                   extra_argv=("--skip-validation", "--force"))
        assert reused["code"] == 0
        out = capsys.readouterr().out
        notices = [line for line in out.splitlines() if line.startswith("Scope policy notice:")]
        assert len(notices) == 1
        assert "requested under `preserve`" in notices[0]
        assert "--force --rerun-all starts a fresh round under the default `expand`" in notices[0]
        assert "[SCOPE_POLICY_CONFLICT]" not in out
        # Only the missing reviewer ran, and it was asked under the kept policy.
        assert [s["model_spec"] for s in reused["sent"]] == [models[1]]
        assert REVIEWER_PRESERVE in flat(_full_prompt(reused["sent"][0]))
        record = sp.round_record(_state(first["plan"]), "review-plan")
        assert (record["review_policy"], record["review_policy_source"]) == ("preserve", "review")

        refused = _review_plan_main(tmp_path, "expand", models=models,
                                    extra_argv=("--skip-validation", "--force"))
        assert refused["code"] == 3
        assert refused["sent"] == []
        assert "--rerun-all" in capsys.readouterr().out


class TestResolutionUnits:
    def test_round_results_ignore_bookkeeping_and_include_empty_legacy_reviews(self, tmp_path):
        assert not base.round_results_exist(str(tmp_path / "missing"))
        for name in (".status.json", "grouped.json", "validation_tasks.json", "report_data.json",
                     "consolidated_report_data.json", "subagent_batch_report.json"):
            (tmp_path / name).write_text("{}")
        assert not base.round_results_exist(str(tmp_path))
        (tmp_path / "cursor-agent_auto.json").write_text("[]")
        assert base.round_results_exist(str(tmp_path))

    def test_new_round_defaults_to_expand(self):
        assert scope_round.resolve_review_round({}, "review-plan", None) == scope_round.RoundPolicy(
            "expand", "default", True)

    def test_legacy_results_continue_as_expand(self):
        resolved = scope_round.resolve_review_round({}, "review-plan", None, has_results=True)
        assert (resolved.policy, resolved.source, resolved.new_round) == ("expand", "legacy", False)
        with pytest.raises(scope_round.PolicyConflict):
            scope_round.resolve_review_round({}, "review-plan", "preserve", has_results=True)

    def test_a_plain_force_keeps_the_reused_results_policy(self):
        state: Dict[str, Any] = {}
        sp.record_review_round(state, "review-plan", "preserve", "explicit")
        kept = scope_round.resolve_review_round(state, "review-plan", None, force=True,
                                                has_results=True)
        assert (kept.policy, kept.source, kept.new_round) == ("preserve", "review", True)
        assert "`preserve`" in kept.notice and "--force --rerun-all" in kept.notice
        # Typed and different: still refused, naming --rerun-all.
        with pytest.raises(scope_round.PolicyConflict, match="--rerun-all"):
            scope_round.resolve_review_round(state, "review-plan", "expand", force=True,
                                             has_results=True)
        # --rerun-all, or no results to reuse: a fresh default round, no notice.
        for kwargs in ({"has_results": True, "rerun_all": True}, {"has_results": False}):
            fresh = scope_round.resolve_review_round(state, "review-plan", None, force=True,
                                                     **kwargs)
            assert fresh == scope_round.RoundPolicy("expand", "default", True)
        # The same policy typed again: no conflict and no notice.
        same = scope_round.resolve_review_round(state, "review-plan", "preserve", force=True,
                                                has_results=True)
        assert (same.policy, same.source, same.notice) == ("preserve", "explicit", None)

    def test_a_record_with_no_results_yet_takes_the_new_explicit_policy(self):
        state: Dict[str, Any] = {}
        sp.record_review_round(state, "code-review", "preserve", "explicit")
        resolved = scope_round.resolve_review_round(state, "review-code", "expand")
        assert (resolved.policy, resolved.new_round) == ("expand", True)

    def test_validators_baseline_moves_when_the_document_changed(self):
        class Doc:
            def __init__(self, digest):
                self.sha256 = digest
                self.path = None
                self.source_path = None
                self.label = "plan"
                self.in_phase_dir = False

        recorded = sp.build_baseline([sp.baseline_source("plan", sha256="1" * 64),
                                      sp.baseline_source("tasks", sha256="2" * 64)])
        same = scope_round.validators_baseline("review-tasks", recorded, plan_document=Doc("1" * 64))
        moved = scope_round.validators_baseline("review-tasks", recorded, plan_document=Doc("3" * 64))
        assert same["baseline_id"] == recorded["baseline_id"]
        assert moved["baseline_id"] != recorded["baseline_id"]

    def test_expand_decides_nothing(self):
        groups = _groups()
        assert scope_round.annotate_decisions(groups, "expand", BASELINE["baseline_id"]) is None
        assert all(sp.DECISION_KEY not in g for g in groups)

    def test_only_groups_validation_judged_count_as_deferrals(self):
        """A validation_failed group reached no verdict: no decision, no count;
        an invalid one is shown its decision but is not a deferral either."""
        failed, judged, invalid, valid = (_groups() + _groups())[:4]
        failed["validation_status"] = "validation_failed"
        failed[sp.DECISION_KEY] = {"outcome": "deferred"}  # a stale one is dropped
        judged["validation_status"] = "needs-human-decision"
        invalid["validation_status"] = "invalid"
        valid["validation_status"] = "valid"
        counts = scope_round.annotate_decisions([failed, judged, invalid, valid], "preserve",
                                                BASELINE["baseline_id"])
        assert counts == {"eligible": 0, "deferred": 2, "in_scope": 0,
                          "scope_extension": 0, "unclear": 2}
        assert sp.DECISION_KEY not in failed
        assert all(g[sp.DECISION_KEY]["outcome"] == "deferred" for g in (judged, invalid, valid))
        assert scope_round.decisions_line(counts).startswith(
            "[SCOPE_POLICY] preserve: 2 validated group(s) deferred")


class TestCodeReviewRound:
    """review-code resolves, records and hands the same round to its validators."""

    def _main(self, tmp_path, policy, *, state_dict=None, **overrides):
        from unittest.mock import AsyncMock, MagicMock

        plan_path = tmp_path / "plan.md"
        if not plan_path.exists():
            plan_path.write_text("# plan\n\n## Goal\nFix the teleport resync.\n", encoding="utf-8")
        state_dict = state_dict if state_dict is not None else {"phases_completed": {}}
        values = {"tracked_files": [{"path": "index.html"}], "head_before_implement": "abc123",
                  "head_at_start": "abc123", "pre_existing_changes": []}
        fake_state = MagicMock()
        fake_state.get = MagicMock(side_effect=lambda k, default=None: values.get(k, default))
        fake_state.save = MagicMock()
        fake_state.is_phase_completed = MagicMock(return_value=False)
        fake_state.phase_completion_is_stale = MagicMock(return_value=False)
        fake_state.state = state_dict
        issue = {"title": "OOB teleport skips resync", "desc": "Teleport skips the resync.",
                 "type": "bug", "importance": "high", "file": "index.html",
                 "scope_relation": "in_scope", "scope_reason": "Fixes a regression."}
        reviews = AsyncMock(return_value={"cursor-agent:auto": (True, [issue], None)})
        fields = dict(
            plan_file=str(plan_path), models=["cursor-agent:auto"], interactive=False,
            quick=False, timeout=None, max_parallel=2, base_ref=None, skip_validation=False,
            validation_model=None, apply_fixes=False, force=False, rerun_all=False,
            reaggregate=False, report_style="pr",
        )
        fields.update(overrides)
        args = argparse.Namespace(**fields)
        ns_with_policy(args, policy)
        phase = tmp_path / "phase"

        class _Ctx:
            def __init__(self, files, **_kw):
                self.files = list(files)

            def __enter__(self):
                return self.files

            def __exit__(self, *exc):
                return False

        _release_run_anchor()
        with patch.object(cro, "parse_args", return_value=args), \
                patch.object(cro, "get_or_create_state", return_value=fake_state), \
                patch("utils.interactive.resolve_models", return_value=["cursor-agent:auto"]), \
                patch("utils.interactive.is_model_valid", return_value=True), \
                patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
                patch.object(cro, "intent_to_add_untracked", side_effect=_Ctx), \
                patch.object(cro, "run_all_reviews", new=reviews), \
                patch.object(cro, "build_validation_secondary_context", return_value=("", "")), \
                patch.object(cro, "dispatch_subagent_tasks", return_value=False), \
                patch.object(cro, "generate_review_report", return_value="# report\n"), \
                patch.object(cro, "get_phase_dir", return_value=phase):
            try:
                code = asyncio.run(cro.main())
            except SystemExit as exc:
                code = exc.code
        return {"code": code, "state": state_dict, "reviews": reviews, "phase": phase}

    def test_preserve_reaches_both_roles(self, tmp_path):
        run = self._main(tmp_path, "preserve")
        assert run["reviews"].call_args.kwargs.get("scope_policy") == "preserve"
        record = sp.round_record(run["state"], "code-review")
        assert record["review_policy"] == "preserve"
        document = json.loads((run["phase"] / "validation_tasks.json").read_text(encoding="utf-8"))
        assert document["scope"]["policy"] == "preserve"
        assert document["scope"]["phase"] == "code-review"
        assert document["scope"]["baseline_id"] == record["baseline"]["baseline_id"]
        roles = [s["role"] for s in record["baseline"]["sources"]]
        assert roles[0] == "plan" and "diff" in roles
        for task in document["batches"]:
            assert VALIDATOR_PRESERVE in flat(task["prompt"])

    def test_the_default_round_passes_no_policy_keyword(self, tmp_path):
        run = self._main(tmp_path, None)
        assert "scope_policy" not in run["reviews"].call_args.kwargs
        document = json.loads((run["phase"] / "validation_tasks.json").read_text(encoding="utf-8"))
        assert document["scope"]["policy"] == "expand"
        assert VALIDATOR_PRESERVE not in flat(document["batches"][0]["prompt"])

    def test_a_policy_change_on_an_unfinished_round_is_exit_3(self, tmp_path, capsys):
        state: Dict[str, Any] = {"phases_completed": {}}
        sp.record_review_round(state, "code-review", "expand", "default")
        phase = tmp_path / "phase"
        phase.mkdir()
        (phase / "cursor-agent_auto.json").write_text("[]", encoding="utf-8")
        run = self._main(tmp_path, "preserve", state_dict=state, force=True)
        assert run["code"] == 3
        assert run["reviews"].call_count == 0
        assert "[SCOPE_POLICY_CONFLICT]" in capsys.readouterr().out


class TestCodeReviewBaselineUnits:
    def test_a_commit_review_binds_brief_endpoints_and_context_plan(self, tmp_path):
        brief = tmp_path / "brief.md"
        brief.write_text("<!-- multi-llm:commit-brief v1 -->\n# Brief\n", encoding="utf-8")
        record = dict(_commit_record(), plan_sha256="d" * 64,
                      plan_snapshot="plans/x/context-plan.md", focus="retries")

        class State:
            state = {"commit_range": record}

        baseline = cro.code_review_scope_baseline(State(), brief, None, base_ref="a" * 40,
                                                  head_ref="b" * 40)
        roles = {s["role"]: s for s in baseline["sources"]}
        assert set(roles) == {"commit_brief", "commit_messages", "context_plan", "focus"}
        assert roles["commit_messages"]["endpoints"] == {"base": "a" * 40, "head": "b" * 40}
        assert roles["context_plan"]["sha256"] == "d" * 64
        moved = dict(record, head="c" * 40)

        class Moved:
            state = {"commit_range": moved}

        assert cro.code_review_scope_baseline(Moved(), brief, None, base_ref=None)[
            "baseline_id"] != baseline["baseline_id"]


#: (case, phase, text) the rubric must carry, in every role and policy that
#: classifies (plan §7 "Semantics"). Fixture verdicts prove plumbing only; these
#: pin that the cases a model is asked to tell apart are actually described.
RUBRIC = [
    ("required existing work", "review-plan",
     "Completing an existing requirement is in scope even when it is substantial work"),
    ("added-file prerequisite", "code-review", "and so is a prerequisite in another file"),
    ("removing scope creep", "review-plan",
     "removing behaviour or content the baseline never asked for (existing scope creep)"),
    ("justified optional expansion", "review-plan",
     "even when it is worthwhile or needed to make the overall goal achievable"),
    ("unrelated cleanup", "code-review", "adds a capability, outcome or unrelated cleanup"),
    ("missing context", "review-plan", "the baseline does not settle it (missing, truncated or vague intent)"),
    ("mixed remedies", "review-plan",
     "materially different fixes would differ in scope and none is chosen"),
    ("vague commit intent", "commits",
     "Commit messages too vague to settle a product-scope question make the answer \"unclear\""),
    ("task drift", "review-tasks", "does not make the feature in scope by listing it"),
    ("implementation drift", "code-review", "drift in the implementation cannot authorize itself"),
]


class TestSemanticsRubric:
    @pytest.mark.parametrize("policy", ["expand", "preserve"])
    @pytest.mark.parametrize("case,phase,text", RUBRIC, ids=[c for c, _p, _t in RUBRIC])
    def test_the_reviewer_rubric_names_the_case(self, case, phase, text, policy):
        assert text in flat(sp.reviewer_guidance(policy, phase))

    @pytest.mark.parametrize("case,phase,text",
                             [r for r in RUBRIC if r[0] not in ("required existing work",
                                                                 "added-file prerequisite",
                                                                 "removing scope creep",
                                                                 "justified optional expansion",
                                                                 "unrelated cleanup",
                                                                 "missing context",
                                                                 "mixed remedies")],
                             ids=lambda v: v if isinstance(v, str) and " " in v else None)
    def test_the_validator_baseline_names_the_case(self, case, phase, text):
        for policy in ("expand", "preserve"):
            assert text in flat(sp.validator_guidance(policy, phase))

    def test_preserve_validators_settle_mixed_remedies_and_exclusions(self):
        text = flat(sp.validator_guidance("preserve", "review-plan"))
        assert 'any extension among them makes the group "scope_extension"' in text
        assert "Explicit exclusions in the baseline bind under every policy." in text
        assert "removes behaviour the baseline never asked for" in text

    def test_expand_validators_get_the_relation_definitions_too(self):
        text = flat(sp.validator_guidance("expand", "review-plan"))
        for _case, phase, rubric in RUBRIC:
            if phase == "review-plan":
                assert rubric in text

    def test_preserve_reviewers_report_a_blocked_requirement(self):
        text = flat(sp.reviewer_guidance("preserve", "review-plan"))
        assert "do not hide the problem and do not prescribe the expansion as the fix" in text
        assert "Explicit exclusions in the baseline" in text
