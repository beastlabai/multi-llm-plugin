#!/usr/bin/env python3
"""``--cascade`` end to end: the real dispatcher, real phases, one model per stage.

``--improve-plan --cascade`` and ``--improve-code --cascade`` run one
review-then-apply round per model, weakest first, each stage reviewing the work
as the stages before it left it and told what they rejected or changed (the
forward ledger). ``tests/test_improve_cascade_executor.py`` and
``tests/test_improve_cascade_native.py`` pin the stage argv and the native
planner's lines with every phase stubbed or fabricated on disk; this module
runs the same contract the way a user gets it: a ``multi_llm.py`` subprocess
in a throwaway git repository, the suite's mock provider binaries on ``PATH``,
and nothing stubbed inside the skill. It reuses ``tests/test_e2e_improve_modes.py``'s
world (repository, driver config, write plugin) unchanged and adds:

* **one answer per stage.** The scenario tells the stages apart by the ledger
  each stage's reviewer prompt carries ("already been through <n> automated
  review round(s)"), never by the model, so a stage that reviews without its
  ledger gets stage 1's answer and the counts below stop matching;
* **a second plugin** layered over the improve-modes one (``MOCK_LLM_PLUGIN``),
  which can fail every call to one model (a stage whose only reviewer
  produced nothing) and can have the validators reject one theme. When a
  validation prompt carries the ledger and the ledger lists that theme as
  rejected, the scripted verdict names the entry in ``reraise_of``, as the
  validator contract asks.

What is anchored, by weight: the cascade directory on disk (``record.json``, the
pointer, each stage's ``inputs/`` and published ``artifacts/`` with its
``.complete.json`` hashes), the per-round table recounted from the archived
files, the mock call log (which model reviewed when, which prompts carried the
ledger, what a re-run really called), the plan's bytes, and the one
``MULTI_LLM_RESULT`` line.

Covered: three-stage ``improve-plan`` and ``improve-code`` in one executor
invocation; a rejected stage-1 finding raised again in stage 2 (marked
``invalid`` with ``reraise_of``, counted in the Re-raises column); the judge's
prompt in every stage (the rule-6 sentence, and from stage 2 the rejected
list); the finished cascade's command again (exit 2, nothing called, also under
``--detach``) and ``--force --cascade`` (a new cascade directory); a stop in
stage 2's apply and the printed continuation finishing it without calling stage
1's model again, and the next-step lookups while it is stopped; a detached
worker running the whole cascade (each close-out re-entering the run dir's
lock); a middle stage whose one model fails (skipped, the run completes); the
native leg's planner chain as a file-shim host runs it (``util improve-steps``
hands off, each ``[IMPROVE_STEP] … run`` through the real dispatcher, then the
latest ``[IMPROVE_CASCADE_CONTINUE]`` until ``[CASCADE_COMPLETE]``), with the
Stop hook's report check over a transcript of that very chain; and the
container row's scenario, checked against the prompts a real cascade sends.

``status`` after a foreground stop names the stage the cascade continues at,
on an entry of its own (no composite run record), and suggests the cascade's
continuation rather than the next workflow phase.

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_e2e_improve_cascade.py -q
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import PERF_SCALE  # noqa: E402
import scenario_matcher  # noqa: E402
from tests.support.transcripts import Transcript, decision_of, run_check  # noqa: E402
from tests.support.wakeup_world import call  # noqa: E402
from tests.test_e2e_improve_modes import (  # noqa: E402
    CODE_MARKER,
    MULTI_LLM,
    PLAN_MARKER,
    PLAN_REL,
    RESPONSES,
    ROUND_TWO_FINDINGS,
    Run,
    World,
    _stub_uv,
    classify,
)
from utils import cascade_provenance as provenance  # noqa: E402
from utils import report_check  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402

#: The stages, weakest first: three catalog models of one mock CLI, told apart
#: in the call log by their ``--model`` argument.
STAGES = ("cursor-agent:auto", "cursor-agent:composer-2.5", "cursor-agent:gpt-5.6-luna-high")
STAGE_MODEL_ARGS = tuple(spec.split(":", 1)[1] for spec in STAGES)

PLAN_DIRS = ("review-plan", "apply-suggestions")
CODE_DIRS = ("code-review", "apply-fixes")
LEDGER_HEADING = "## Earlier review rounds"
#: What the codex shim bakes into every dispatcher call (hosts/codex/shim.md.tmpl).
CODEX_TOKENS = ("--host", "codex", "--yes", "--exec-subagents")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "e2e"
CHAIN_SCENARIO = FIXTURES / "scenarios" / "container_full_chain.yaml"
#: The container row's scenario (tests/test_container_improve_cascade.py).
CONTAINER_CASCADE_SCENARIO = FIXTURES / "container_scenarios" / "container_improve_cascade.yaml"

# --------------------------------------------------------------------------- #
# The cascade plugin, layered over tests/test_e2e_improve_modes.py's
# --------------------------------------------------------------------------- #

CASCADE_PLUGIN = '''\
"""Scratch mock plugin for tests/test_e2e_improve_cascade.py (see its docstring)."""
import importlib.util
import json
import os
import re
import sys

_spec = importlib.util.spec_from_file_location("cascade_e2e_base_plugin", __BASE__)
_base = importlib.util.module_from_spec(_spec)
sys.modules["cascade_e2e_base_plugin"] = _base
_spec.loader.exec_module(_base)

#: `- R1-2 [HIGH] (model) <title> — <description> — reason: <reason>`
_LEDGER_ENTRY = re.compile(r"^- (R\\d+-\\d+) \\[[A-Z]+\\] \\([^)]*\\) (.+?) — ", re.M)
_REJECTED = "Rejected or held back in earlier rounds:"
_CHANGED = "Changed by earlier rounds:"


def parse_provider_args(provider):
    """The improve-modes seam, plus: every call to CASCADE_E2E_FAIL_MODEL fails."""
    args, prompt = _base.parse_provider_args(provider)
    failing = os.environ.get("CASCADE_E2E_FAIL_MODEL")
    if failing and getattr(args, "model", None) == failing:
        os.environ["MOCK_LLM_FAIL"] = "1"
    return args, prompt


def _themes(prompt):
    """{group_hash: theme} of the groups a validation batch embeds."""
    start = prompt.index("[", prompt.index("## Suggestion groups to validate"))
    groups, _end = json.JSONDecoder().raw_decode(prompt, start)
    return {group["group_hash"]: group.get("theme") or "" for group in groups}


def get_response_content(prompt):
    """Validators reject the CASCADE_E2E_REJECT theme, naming the ledger entry it repeats."""
    content = _base.get_response_content(prompt)
    reject = os.environ.get("CASCADE_E2E_REJECT")
    if not reject or not prompt.startswith("You are validating review suggestions"):
        return content
    document = json.loads(content)
    themes = _themes(prompt)
    ledger = prompt.split(_REJECTED, 1)[1].split(_CHANGED, 1)[0] if _REJECTED in prompt else ""
    repeats = next((m.group(1) for m in _LEDGER_ENTRY.finditer(ledger)
                    if m.group(2).startswith(reject)), None)
    for group in document["groups"]:
        if themes.get(group["group_hash"], "").startswith(reject):
            group["status"] = "invalid"
            group["reason"] = "scripted rejection"
            if repeats:
                group["reraise_of"] = repeats
    return json.dumps(document)
'''

# --------------------------------------------------------------------------- #
# What each stage's reviewer answers
# --------------------------------------------------------------------------- #

PLAN_STAGE_1 = json.loads((RESPONSES / "review_plan" / "raw_suggestions.json").read_text(
    encoding="utf-8"))
#: The stage-1 finding the validators reject; stage 2 raises it again verbatim.
REJECTED_THEME = "Document zero-downtime migration approach"
_REJECTED_FINDING = next(s for s in PLAN_STAGE_1 if s["title"] == REJECTED_THEME)
PLAN_STAGE_2 = [_REJECTED_FINDING, ROUND_TWO_FINDINGS[0]]
PLAN_STAGE_3 = [ROUND_TWO_FINDINGS[1]]

CODE_STAGE_2 = [{
    "title": "Password check is not constant time",
    "desc": "verify_password compares two hex digests with `==`, which returns at the "
            "first differing character; compare them with hmac.compare_digest.",
    "importance": "medium", "file": "src/auth.py", "line_range": [19, 21], "type": "bug",
    "anchor_text": "return hash_password(password) == stored_hash",
}]
CODE_STAGE_3 = [{
    "title": "Lockout window is not configurable",
    "desc": "WINDOW_SECONDS is a module constant; read it from settings so an operator "
            "can tune the lockout without a code change.",
    "importance": "low", "file": "src/security.py", "line_range": [5, 6],
    "type": "improvement", "anchor_text": "WINDOW_SECONDS = 300",
}]

#: An alternative stage 2 and 3 whose findings are all new: every stage's apply
#: dispatches batches, so a stop can land between two of stage 2's.
FRESH_STAGE_2 = list(ROUND_TWO_FINDINGS)
FRESH_STAGE_3 = [{
    "title": "Rate limit the token refresh endpoint",
    "desc": "Step 6 limits login attempts only; a stolen refresh token can be replayed "
            "without limit. Apply the same per-address limit to POST /auth/refresh.",
    "type": "addition", "reference": "### Step 6: Security Measures", "importance": "MEDIUM",
}]

_PLAN_REVIEW = "You are reviewing an implementation plan for completeness"
_CODE_REVIEW = "You are reviewing code changes that implement a feature plan"


def _stage_rules(opening: str, subject: str, answers: Tuple[Path, Path, Path]) -> List[dict]:
    """Three rules for one review prompt: the ledger's round count picks the answer."""
    stage_1, stage_2, stage_3 = answers
    seen = f"This {subject} has already been through"
    return [
        {"pattern": f"(?s){opening}.*{seen} 2 automated review rounds by",
         "fixture": str(stage_3)},
        {"pattern": f"(?s){opening}.*{seen} 1 automated review round by",
         "fixture": str(stage_2)},
        {"pattern": opening, "fixture": str(stage_1)},
    ]


class CascadeWorld(World):
    """``World`` with per-stage review answers and the cascade plugin."""

    def __init__(self, tmp_path: Path, *, working_diff: bool = False,
                 plan_stages: Tuple[list, list] = (PLAN_STAGE_2, PLAN_STAGE_3)):
        self.plan_stages = plan_stages
        super().__init__(tmp_path, working_diff=working_diff)
        plugin = tmp_path / "cascade_mock_plugin.py"
        plugin.write_text(
            CASCADE_PLUGIN.replace("__BASE__", repr(str(tmp_path / "improve_mock_plugin.py"))),
            encoding="utf-8")
        self.runner.mock_plugin = plugin

    def set_review_answers(self, plan_review=None, code_review=None) -> None:
        """The staged scenario (``World.__init__`` calls this once)."""
        def answer(name: str, findings: list) -> Path:
            path = self.tmp_path / f"{name}.json"
            path.write_text(json.dumps(findings), encoding="utf-8")
            return path

        plan = (RESPONSES / "review_plan" / "raw_suggestions.json",
                answer("plan-stage-2", self.plan_stages[0]),
                answer("plan-stage-3", self.plan_stages[1]))
        code = (RESPONSES / "code_review" / "container_issues.json",
                answer("code-stage-2", CODE_STAGE_2), answer("code-stage-3", CODE_STAGE_3))
        self.scenario.write_text(yaml.safe_dump({
            "name": "improve cascade",
            "on_unmatched": "fail",
            "prompts": _stage_rules(_PLAN_REVIEW, "plan", plan)
            + _stage_rules(_CODE_REVIEW, "code", code),
        }, sort_keys=False), encoding="utf-8")

    # -- running --------------------------------------------------------------
    def cascade(self, mode: str, *flags: str, **kwargs) -> Run:
        """``<mode> <plan> --cascade --models <the three stages> --yes --exec-subagents``."""
        return self.run(mode, "--cascade", "--models", *STAGES, "--yes", "--exec-subagents",
                        *flags, **kwargs)

    def dispatch(self, argv: List[str]) -> subprocess.CompletedProcess:
        """One dispatcher call, the way a host's shell tool runs it."""
        return subprocess.run(
            [sys.executable, str(MULTI_LLM), *argv], cwd=self.repo, env=self.env(),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120 * PERF_SCALE,
        )

    # -- reading what a cascade left -------------------------------------------
    def composite_dir(self, mode: str) -> Path:
        return self.out / mode

    def cascade_dirs(self, mode: str) -> List[Path]:
        root = self.composite_dir(mode) / "cascades"
        return sorted(root.iterdir()) if root.is_dir() else []

    def current(self, mode: str) -> Path:
        pointer = json.loads((self.composite_dir(mode) / "cascade-current.json").read_text(encoding="utf-8"))
        return self.composite_dir(mode) / pointer["dir"]

    def reviews(self) -> List[Tuple[str, str]]:
        """``(model argument, prompt)`` of every reviewer call, in call order."""
        found = []
        for entry in self.calls():
            if classify(entry) == "review":
                argv = entry["args"]
                found.append((argv[argv.index("--model") + 1], entry["prompt"]))
        return found

    def reviews_of(self, model_arg: str) -> List[str]:
        return [prompt for model, prompt in self.reviews() if model == model_arg]

    def calls_by_stage(self) -> List[List[dict]]:
        """The call log cut at each reviewer call: stage k's review and everything
        it led to (validation, writes) up to the next stage's review."""
        stages: List[List[dict]] = []
        for entry in self.calls():
            if classify(entry) == "review" or not stages:
                stages.append([])
            stages[-1].append(entry)
        return stages


@pytest.fixture
def world(tmp_path) -> CascadeWorld:
    return CascadeWorld(tmp_path)


@pytest.fixture
def code_world(tmp_path) -> CascadeWorld:
    return CascadeWorld(tmp_path, working_diff=True)


# --------------------------------------------------------------------------- #
# Reading the evidence
# --------------------------------------------------------------------------- #

def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_lines(run: Run) -> List[str]:
    return [line for line in run.stdout.splitlines() if line.startswith("MULTI_LLM_RESULT:")]


def _steps(run: Run) -> List[Tuple[str, List[str]]]:
    """``(mode word, flags after the plan)`` of every ``$ uv run …`` step line, in order."""
    steps = []
    for line in run.stdout.splitlines():
        if line.startswith("$ "):
            tokens = shlex.split(line[2:])
            at = next(i for i, token in enumerate(tokens) if token.endswith("multi_llm.py"))
            steps.append((tokens[at + 1], tokens[at + 3:]))
    return steps


def _after_dispatcher(command: str) -> List[str]:
    """The tokens after ``multi_llm.py`` of a printed command (the dispatcher's own argv)."""
    tokens = shlex.split(command)
    at = next(i for i, token in enumerate(tokens) if token.endswith("multi_llm.py"))
    return tokens[at + 1:]


def _flag_values(tokens: List[str], flag: str) -> List[str]:
    """The values after *flag* up to the next option."""
    if flag not in tokens:
        return []
    out = []
    for token in tokens[tokens.index(flag) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def _marker_lines(text: str, marker: str) -> List[str]:
    return [line[len(marker) + 1:] for line in text.splitlines() if line.startswith(marker + " ")]


def _tree(root: Path) -> Dict[str, bytes]:
    """Every file under *root* but lock files (a status poll may take and drop one)."""
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and not p.name.endswith(".lock")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_rows(text: str) -> Dict[int, List[str]]:
    """``{stage: cells}`` of the per-round table's body rows."""
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells and cells[0].isdigit():
            rows[int(cells[0])] = cells
    return rows


def _reused(artifacts: Path) -> set:
    """The items of the batches the stage's apply reused from an earlier
    stage (its write report's resume-skip rows) instead of dispatching."""
    for report in artifacts.glob("*/write_batch_report.json"):
        return {item for task in _read(report)["tasks"] if task.get("skipped")
                for item in task.get("expected_ids") or []}
    return set()


def _recount(artifacts: Path, review_dir: str) -> Dict[str, int]:
    """The table's item columns, counted straight from one stage's archived files.

    An ``applied`` item whose batch was reused from an earlier stage (a
    finding repeated word for word) is No-op: this stage applied nothing."""
    groups = _read(artifacts / review_dir / "grouped.json")["groups"]
    verdicts = _read(artifacts / review_dir / "validation.json")["groups"]
    records = _read(artifacts / "records.json")
    reused = _reused(artifacts)
    statuses = [
        "no-op" if entry.get("status") == "applied" and item in reused else entry.get("status")
        for item, entry in (records.get("processed") or {}).items()]
    decisions = [entry.get("decision") for entry in (records.get("human_decisions") or {}).values()]
    return {
        "findings": len(groups),
        "valid": sum(v.get("status") == "valid" for v in verdicts),
        "invalid": sum(v.get("status") == "invalid" for v in verdicts),
        "needs_decision": sum(v.get("status") == "needs-human-decision" for v in verdicts),
        "reraises": sum(bool(v.get("reraise_of")) for v in verdicts),
        "applied": statuses.count("applied"),
        "no_op": statuses.count("no-op"),
        "declined": statuses.count("declined"),
        "judge_approved": decisions.count("approved"),
        "judge_salvaged": decisions.count("salvaged"),
        "judge_skipped": decisions.count("skipped"),
    }


def _assert_archived(world: CascadeWorld, mode: str, dirs: Tuple[str, str],
                     cascade_dir: Path, models: Tuple[str, ...]) -> Dict[int, Dict[str, int]]:
    """Every stage published once, its marks true, and every table number its files'.

    Returns each stage's recount so a caller can pin the numbers it scripted.
    """
    review_dir, apply_dir = dirs
    record = _read(cascade_dir / "record.json")
    summary = (cascade_dir / "summary.md").read_text(encoding="utf-8")
    rows = _table_rows(summary)
    assert record["state"] == "finished" and record["mode"] == mode, record
    assert [s["model"] for s in record["stages"]] == list(models), record["stages"]
    assert sorted(rows) == list(range(1, len(models) + 1)), summary
    counts: Dict[int, Dict[str, int]] = {}
    for stage in record["stages"]:
        index = stage["index"]
        home = cascade_dir / f"stage-{index}"
        artifacts = home / "artifacts"
        assert stage["state"] == "done", stage
        assert not (home / "artifacts.partial").exists()

        # -- the frozen inputs and the published mark ---------------------------
        manifest = provenance.load_manifest(home / "inputs" / "manifest.json")
        assert (manifest["cascade_id"], manifest["stage"], manifest["model"]) == (
            record["cascade_id"], index, stage["model"])
        mark = _read(artifacts / ".complete.json")
        assert (mark["cascade_id"], mark["stage"], mark["model"], mark["round_id"]) == (
            record["cascade_id"], index, stage["model"], stage["bound_round"]), mark
        assert mark["files"], mark
        for rel, digest in mark["files"].items():
            assert _sha256(artifacts / rel) == digest, f"stage {index}: {rel} moved"
        assert not [p for p in artifacts.rglob("*") if p.name.endswith(".lock")], (
            "a lock file was archived")

        # -- the archived review round belongs to this stage ---------------------
        block = provenance.read_block(artifacts / review_dir)
        assert block is not None, f"stage {index}: the archived review carries no cascade block"
        assert (block["cascade_id"], block["stage"], block["model"]) == (
            record["cascade_id"], index, stage["model"]), block

        # -- the numbers: files == table == record ---------------------------------
        recount = _recount(artifacts, review_dir)
        counts[index] = recount
        cells = rows[index]
        assert cells[1] == stage["model"], cells
        reraises = "—" if index == 1 else str(recount["reraises"])
        assert cells[2:10] == [
            str(recount["findings"]), str(recount["valid"]), str(recount["invalid"]),
            str(recount["needs_decision"]), reraises, str(recount["applied"]),
            str(recount["no_op"]), str(recount["declined"]),
        ], (index, cells, recount)
        # Held back (column 10) is 0 without --scope-policy preserve.
        assert cells[10] == "0", (index, cells)
        judge = "{judge_approved}/{judge_salvaged}/{judge_skipped}".format(**recount)
        assert cells[11] == judge and cells[13] == "not recorded", (index, cells, judge)
        for key, value in recount.items():
            assert stage["counts"][key] == value, (index, key, stage["counts"])
        assert stage["counts"]["cost"] is None, "the mocks report no cost; never a zero"
        # the write report of this stage's apply holds this stage's batches only
        # (a batch reused from an earlier stage is a resume-skip row, not applied here)
        report = _read(artifacts / apply_dir / "write_batch_report.json")
        applied = [t for t in report["tasks"]
                   if t["write_outcome"] == "applied" and not t.get("skipped")]
        assert sum(len(t["expected_ids"]) for t in applied) == recount["applied"], report
    return counts


# --------------------------------------------------------------------------- #
# 1. Three stages in one invocation
# --------------------------------------------------------------------------- #

class TestThreeStagesInOneInvocation:

    def test_improve_plan_reviews_and_applies_one_model_per_stage(self, world):
        run = world.cascade("improve-plan", env=world.env(CASCADE_E2E_REJECT=REJECTED_THEME))
        assert run.exit_code == 0, run.explain()

        # -- one cascade, finished, three stages archived ---------------------------
        dirs = world.cascade_dirs("improve-plan")
        assert len(dirs) == 1 and world.current("improve-plan") == dirs[0], dirs
        cascade_dir = dirs[0]
        counts = _assert_archived(world, "improve-plan", PLAN_DIRS, cascade_dir, STAGES)
        assert (cascade_dir / "start.md").read_bytes() == world.original_plan
        assert [counts[k]["findings"] for k in (1, 2, 3)] == [
            len(PLAN_STAGE_1), len(PLAN_STAGE_2), len(PLAN_STAGE_3)], (
            "a stage did not get its own answer: a reviewer prompt lacked its ledger")

        # -- the plan: each stage's applied batches, once, in stage order ---------------
        markers = []
        for index in (1, 2, 3):
            report = _read(cascade_dir / f"stage-{index}" / "artifacts" / "apply-suggestions"
                           / "write_batch_report.json")
            markers += [PLAN_MARKER.format(id=t["expected_ids"][0]) for t in report["tasks"]
                        if t["write_outcome"] == "applied"]
            after = (cascade_dir / f"stage-{index}" / "artifacts" / "plan.after.md").read_bytes()
            assert after == world.original_plan + "".join(markers).encode("utf-8"), index
        assert world.plan_bytes == world.original_plan + "".join(markers).encode("utf-8")
        assert len(world.calls_of("write")) == len(markers) == 3 + 1 + 1

        # -- each stage's argv: its one model, a fresh round, bound to its inputs ------
        steps = _steps(run)
        assert [word for word, _flags in steps] == ["review-plan", "apply-suggestions"] * 3, steps
        for index, (model, (_word, review)) in enumerate(zip(STAGES, steps[0::2]), start=1):
            assert _flag_values(review, "--models") == [model], review
            assert review.count("--force") == 1 and review.count("--rerun-all") == 1, review
            manifest = Path(_flag_values(review, "--cascade-stage-file")[0])
            assert manifest.resolve() == (
                cascade_dir / f"stage-{index}" / "inputs" / "manifest.json").resolve()
        for _word, apply in steps[1::2]:
            assert "--resume" not in apply and "--force" not in apply, apply

        # -- the prompts: stage 1 alone has no ledger; later stages carry theirs --------
        assert [model for model, _prompt in world.reviews()] == list(STAGE_MODEL_ARGS)
        by_stage = world.calls_by_stage()
        assert len(by_stage) == 3
        for index, calls in enumerate(by_stage, start=1):
            inputs = cascade_dir / f"stage-{index}" / "inputs"
            review = calls[0]["prompt"]
            validators = [c["prompt"] for c in calls if classify(c) == "validate"]
            assert validators, f"stage {index} validated nothing"
            if index == 1:
                assert (inputs / "ledger.md").read_text(encoding="utf-8") == ""
                assert LEDGER_HEADING not in review
                assert not any(LEDGER_HEADING in prompt for prompt in validators)
                assert not any('"reraise_of"' in prompt for prompt in validators)
                continue
            ledger = (inputs / "ledger.md").read_text(encoding="utf-8")
            assert ledger.startswith(LEDGER_HEADING) and ledger.strip() in review, index
            rounds = "1 automated review round by" if index == 2 else "2 automated review rounds by"
            assert rounds in ledger, ledger
            validator_ledger = (inputs / "ledger-validator.md").read_text(encoding="utf-8")
            for prompt in validators:
                assert validator_ledger.strip() in prompt, index
                assert '"reraise_of"' in prompt, "the re-raise contract travels with the ledger"
                assert str(inputs) not in prompt, "a validator prompt names a file path"
        stage_3_ledger = (cascade_dir / "stage-3" / "inputs" / "ledger.md").read_text(encoding="utf-8")
        assert "- A1-1 " in stage_3_ledger and "- A2-1 " in stage_3_ledger, stage_3_ledger

        # -- the re-raise: rejected in stage 1, raised again in stage 2 ------------------
        rejected = (cascade_dir / "stage-2" / "inputs" / "ledger-validator.md").read_text(encoding="utf-8")
        assert f"- R1-1 [HIGH] (cursor-agent:auto) {REJECTED_THEME} — " in rejected, rejected
        verdicts = _read(cascade_dir / "stage-2" / "artifacts" / "review-plan"
                         / "validation.json")["groups"]
        groups = {g["group_hash"]: g for g in _read(
            cascade_dir / "stage-2" / "artifacts" / "review-plan" / "grouped.json")["groups"]}
        repeated = [v for v in verdicts if groups[v["group_id"]]["theme"] == REJECTED_THEME]
        assert len(repeated) == 1 and repeated[0]["status"] == "invalid", verdicts
        assert repeated[0]["reraise_of"] == "R1-1", repeated
        assert [v.get("reraise_of") for v in verdicts if v is not repeated[0]] == [None]
        assert [counts[k]["reraises"] for k in (1, 2, 3)] == [0, 1, 0]
        assert [counts[k]["invalid"] for k in (1, 2, 3)] == [1, 1, 0]
        report_md = (cascade_dir / "stage-2" / "artifacts" / "review-plan" / "report.md").read_text(encoding="utf-8")
        assert "re-raises R1-1" in report_md

        # -- one report, one result line ---------------------------------------------------
        assert run.stdout.count("### Cascade") == 1, run.stdout
        table = [line for line in (cascade_dir / "summary.md").read_text(encoding="utf-8").splitlines()
                 if line.startswith("|")]
        assert "\n".join(table) in run.stdout, "the report's table is not summary.md's"
        assert len(_result_lines(run)) == 1, _result_lines(run)
        result = run.result
        assert (result["mode"], result["state"], result["exit"], result["next"]) == (
            "improve-plan", "done", 0, None), result
        for path in (cascade_dir / "summary.md", cascade_dir / "ledger.md",
                     cascade_dir / "stage-3"):
            assert str(path) in result["artifacts"], (path, result["artifacts"])
        assert result["user_next_step"]["phase"] == "generate-tasks", result
        state = world.state()
        assert set(state["phases_completed"]) == {"review-plan", "apply-suggestions"}
        assert state["improve_plan"]["current_phase"] == "complete", state["improve_plan"]

    def test_the_judge_of_a_later_stage_is_told_what_earlier_stages_rejected(self, world):
        """Validators leave the first group of each batch to the judge, and reject
        one theme; every stage's judge prompt carries the rule-6 sentence (one
        model per round), and from stage 2 on the stage's frozen rejected list."""
        run = world.cascade("improve-plan", env=world.env(
            CASCADE_E2E_REJECT=REJECTED_THEME, IMPROVE_E2E_VERDICT="needs-human-decision",
            IMPROVE_E2E_JUDGE="1"))
        assert run.exit_code == 0, run.explain()
        cascade_dir = world.current("improve-plan")
        counts = _assert_archived(world, "improve-plan", PLAN_DIRS, cascade_dir, STAGES)
        assert all(counts[k]["judge_approved"] >= 1 for k in (1, 2, 3)), counts

        rule_6 = "rule 6 (model consensus) does not apply"
        rejected_list = "Earlier rounds of this run rejected or held back the entries below."
        for index, calls in enumerate(world.calls_by_stage(), start=1):
            judges = [c["prompt"] for c in calls if classify(c) == "judge"]
            assert judges, f"stage {index}: no judge prompt"
            frozen = (cascade_dir / f"stage-{index}" / "inputs" / "ledger-judge.md").read_text(encoding="utf-8")
            for prompt in judges:
                assert rule_6 in prompt, index
                if index == 1:
                    assert frozen == "" and rejected_list not in prompt
                else:
                    assert rejected_list in prompt and frozen.strip() in prompt, index
                    assert f"- R1-1 [HIGH] (cursor-agent:auto) {REJECTED_THEME}" in prompt
        assert len(_result_lines(run)) == 1 and run.result["state"] == "done"

    def test_improve_code_fixes_the_change_one_model_per_stage(self, code_world):
        world = code_world
        run = world.cascade("improve-code")
        assert run.exit_code == 0, run.explain()

        dirs = world.cascade_dirs("improve-code")
        assert len(dirs) == 1, dirs
        cascade_dir = dirs[0]
        counts = _assert_archived(world, "improve-code", CODE_DIRS, cascade_dir, STAGES)
        assert not (cascade_dir / "start.md").exists(), "improve-code keeps no plan copy"
        assert counts[2]["findings"] == len(CODE_STAGE_2)
        assert counts[3]["findings"] == len(CODE_STAGE_3)

        # -- the source files: one marker per applied batch, each stage's own files ------
        expected = {rel: world.sources_before[rel].decode("utf-8") for rel in world.sources_before}
        for index in (1, 2, 3):
            artifacts = cascade_dir / f"stage-{index}" / "artifacts"
            report = _read(artifacts / "apply-fixes" / "write_batch_report.json")
            touched = set()
            for task in report["tasks"]:
                assert task["write_outcome"] == "applied", task
                expected[task["files_touched"][0]] += CODE_MARKER.format(id=task["expected_ids"][0])
                touched.update(task["files_touched"])
            listed = (artifacts / "files-touched.txt").read_text(encoding="utf-8").split()
            assert set(listed) == touched, (index, listed, touched)
        assert set((cascade_dir / "stage-2" / "artifacts" / "files-touched.txt")
                   .read_text(encoding="utf-8").split()) == {"src/auth.py"}
        for rel, text in expected.items():
            assert (world.repo / rel).read_bytes().decode("utf-8") == text, rel

        # -- prompts: the ledger reaches stage 2 and 3's reviewers, never stage 1's ------
        assert [model for model, _p in world.reviews()] == list(STAGE_MODEL_ARGS)
        prompts = [prompt for _model, prompt in world.reviews()]
        assert LEDGER_HEADING not in prompts[0]
        for index, prompt in ((2, prompts[1]), (3, prompts[2])):
            ledger = (cascade_dir / f"stage-{index}" / "inputs" / "ledger.md").read_text(encoding="utf-8")
            assert ledger.startswith(LEDGER_HEADING) and ledger.strip() in prompt, index
            assert "This code has already been through" in ledger
            assert "changes-so-far.diff" not in ledger, "improve-code freezes no plan diff"
        for word, flags in _steps(run)[0::2]:
            assert word == "review-code" and "--cascade-stage-file" in flags, flags

        assert len(_result_lines(run)) == 1
        assert (run.result["mode"], run.result["state"], run.result["exit"]) == (
            "improve-code", "done", 0), run.result
        assert run.stdout.count("### Cascade") == 1
        assert set(world.state()["phases_completed"]) == {"code-review", "apply-code-fixes"}


# --------------------------------------------------------------------------- #
# 2. A finished cascade: the same command again, then --force --cascade
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestAFinishedCascade:

    def test_the_same_command_is_already_complete_and_force_starts_a_new_cascade(self, world):
        first = world.cascade("improve-plan")
        assert first.exit_code == 0, first.explain()
        cascade_dir = world.current("improve-plan")
        snapshot = _tree(world.composite_dir("improve-plan"))
        calls = len(world.calls())

        for extra in ((), ("--detach",)):
            again = world.cascade("improve-plan", *extra)
            assert again.exit_code == 2, again.explain()
            lines = _marker_lines(again.stdout, "[ALREADY_COMPLETE]")
            assert len(lines) == 1 and lines[0].startswith("cascade finished at "), again.stdout
            assert "(3 stages); --force --cascade starts a new one" in lines[0], lines
            assert len(world.calls()) == calls, f"{extra}: a provider was called"
            assert "$ uv run" not in again.stdout, f"{extra}: a phase was started"
            assert "[DETACHED_RUN]" not in again.stdout
            assert not list(world.repo.rglob("orchestrator-run*.log")), "a worker was launched"
            assert len(_result_lines(again)) == 1
            assert (again.result["state"], again.result["exit"]) == ("nothing-to-do", 2)
            assert _tree(world.composite_dir("improve-plan")) == snapshot, (
                f"{extra}: a finished cascade's re-run wrote something")

        forced = world.cascade("improve-plan", "--force")
        assert forced.exit_code == 0, forced.explain()
        dirs = world.cascade_dirs("improve-plan")
        assert len(dirs) == 2 and cascade_dir in dirs, dirs
        new = world.current("improve-plan")
        assert new != cascade_dir
        assert _read(cascade_dir / "record.json")["state"] == "finished", (
            "the finished cascade's record was rewritten")
        record = _read(new / "record.json")
        assert record["state"] == "finished" and record["cascade_id"] != _read(
            cascade_dir / "record.json")["cascade_id"]
        assert [s["state"] for s in record["stages"]] == ["done"] * 3
        assert all((new / f"stage-{k}" / "artifacts" / ".complete.json").is_file()
                   for k in (1, 2, 3))
        # it starts from the plan as the first cascade left it
        assert (new / "start.md").read_bytes() == (
            cascade_dir / "stage-3" / "artifacts" / "plan.after.md").read_bytes()
        reviews = [model for model, _p in world.reviews()]
        assert reviews == list(STAGE_MODEL_ARGS) * 2, "a stage of the new cascade ran twice"
        # the new cascade starts over: stage 1 reviews without a ledger again
        assert LEDGER_HEADING not in world.reviews()[3][1]
        assert "--force" in _steps(forced)[0][1]
        assert len(_result_lines(forced)) == 1 and forced.result["state"] == "done"


# --------------------------------------------------------------------------- #
# 3. Stopped in stage 2's apply, continued with the printed command
# --------------------------------------------------------------------------- #

@pytest.mark.slow
class TestAnInterruptedCascade:

    def test_the_printed_continuation_finishes_stage_2_and_runs_stage_3(self, tmp_path):
        world = CascadeWorld(tmp_path, plan_stages=(FRESH_STAGE_2, FRESH_STAGE_3))
        # Stage 1 applies four batches; the sixth write call is stage 2's second batch.
        stopped = world.cascade("improve-plan", env=world.env(IMPROVE_E2E_FAIL_CALL="6"))
        assert stopped.exit_code == 3, stopped.explain()
        assert len(_result_lines(stopped)) == 1
        result = stopped.result
        assert (result["state"], result["exit"]) == ("needs-human", 3), result
        assert "apply-suggestions" in result["error"], result
        continuation = _after_dispatcher(result["next"])
        assert continuation[0] == "improve-plan", continuation
        assert continuation[-2:] == ["--cascade", "--resume"], continuation
        assert "--force" not in continuation and "--cascade-stage-file" not in continuation
        assert _flag_values(continuation, "--models") == list(STAGES), (
            "the continuation carries the typed stage list, never one stage's model")

        cascade_dir = world.current("improve-plan")
        record = _read(cascade_dir / "record.json")
        assert record["state"] == "running"
        assert [s["state"] for s in record["stages"]] == ["done", "running", "pending"]
        assert (cascade_dir / "stage-1" / "artifacts" / ".complete.json").is_file()
        assert not (cascade_dir / "stage-2" / "artifacts").exists()
        assert [model for model, _p in world.reviews()] == list(STAGE_MODEL_ARGS[:2])
        writes, validations = len(world.calls_of("write")), len(world.calls_of("validate"))
        assert writes == 6
        calls_before = len(world.calls())

        # The printed command, with the tokens an executor host bakes into every call.
        resumed = _run_tokens(world, continuation + ["--yes", "--exec-subagents"])
        assert resumed.exit_code == 0, resumed.explain()
        steps = _steps(resumed)
        assert [word for word, _f in steps] == [
            "apply-suggestions", "review-plan", "apply-suggestions"], steps
        assert steps[0][1].count("--resume") == 1, "stage 2's apply was not resumed"
        assert _flag_values(steps[1][1], "--models") == [STAGES[2]]

        # Stage 1's model is never called again, nor stage 2's: one review per stage.
        assert [model for model, _p in world.reviews()] == list(STAGE_MODEL_ARGS)
        new = world.calls()[calls_before:]
        assert classify(new[0]) == "write", "the re-run reviewed or validated before the apply"
        first_review = next(i for i, c in enumerate(new) if classify(c) == "review")
        assert {classify(c) for c in new[:first_review]} == {"write"}, (
            "stage 2's review or validation was paid for twice")
        assert len(world.calls_of("validate")) > validations  # stage 3's own

        _assert_archived(world, "improve-plan", PLAN_DIRS, cascade_dir, STAGES)
        report = _read(cascade_dir / "stage-2" / "artifacts" / "apply-suggestions"
                       / "write_batch_report.json")
        assert [t["write_outcome"] for t in report["tasks"]] == ["applied", "applied"], report
        markers = []
        for index in (1, 2, 3):
            stage_report = _read(cascade_dir / f"stage-{index}" / "artifacts"
                                 / "apply-suggestions" / "write_batch_report.json")
            markers += [PLAN_MARKER.format(id=t["expected_ids"][0])
                        for t in stage_report["tasks"]]
        assert world.plan_bytes == world.original_plan + "".join(markers).encode("utf-8"), (
            "a batch landed twice or never")
        assert len(_result_lines(resumed)) == 1
        assert (resumed.result["state"], resumed.result["exit"]) == ("done", 0)


@pytest.mark.slow
class TestADetachedCascade:

    def test_the_worker_runs_every_stage_and_records_each_while_it_runs(self, world):
        """Every tool route appends ``--detach``: the worker holds the composite run
        dir's lock for the whole cascade, so each close-out must re-enter it."""
        from utils import status_file

        env = world.env(MULTI_LLM_UV=str(_stub_uv(world.tmp_path)))
        launch = world.cascade("improve-plan", "--detach", env=env)
        assert launch.exit_code == 0, launch.explain()
        assert "run_id:" in launch.stdout, launch.explain()

        run_dir = world.composite_dir("improve-plan")
        deadline = time.monotonic() + 90 * PERF_SCALE
        run: dict = {}
        stages_seen = set()
        while time.monotonic() < deadline:
            stage = status_file.read_status(str(run_dir)).get("cascade_stage")
            if stage:
                assert stage["of"] == 3 and stage["model"] == STAGES[stage["index"] - 1], stage
                stages_seen.add(stage["index"])
            run = status_file.read_run(str(run_dir))
            if run.get("state") in status_file.TERMINAL_RUN_STATES:
                break
            time.sleep(0.1)
        log = (run_dir / "orchestrator-run.log").read_text(encoding="utf-8")
        assert run.get("state") == "done" and run.get("exit_code") == 0, (run, log[-3000:])
        assert stages_seen, "the checkpoint never named a stage while the cascade ran"
        assert status_file.read_status(str(run_dir)).get("cascade_stage") is None, (
            "a finished cascade still names a stage")
        assert log.count("=== Step 1: review-plan ===") == 3, log[-3000:]
        assert len([line for line in log.splitlines()
                    if line.startswith("MULTI_LLM_RESULT:")]) == 1
        _assert_archived(world, "improve-plan", PLAN_DIRS, world.current("improve-plan"), STAGES)
        assert [model for model, _p in world.reviews()] == list(STAGE_MODEL_ARGS)


def _run_tokens(world: CascadeWorld, argv: List[str]) -> Run:
    """``multi_llm.py <argv>`` as a ``Run`` (the plan is whatever *argv* names)."""
    started = time.monotonic()
    done = world.dispatch(argv)
    return Run(argv=argv, exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr,
               seconds=time.monotonic() - started, result=parse_result_line(done.stdout))


class TestStatusOfAStoppedCascade:

    def test_the_continuation_lookups_name_the_cascade(self, tmp_path):
        """``--after`` and ``--blocked`` answer "continue the composite" with
        ``--cascade --resume`` while the cascade is unfinished, never the next
        workflow phase, though ``state.json`` still holds stage 1's apply stamp."""
        world, _stopped = _stopped_in_stage_2(tmp_path)
        after = _run_tokens(world, ["util", "next-command", PLAN_REL, "--after", "improve-plan"])
        assert after.exit_code == 0, after.explain()
        command = _marker_lines(after.stdout, "[NEXT_COMMAND]")
        assert command == [f"/multi-llm:multi-llm --improve-plan {PLAN_REL} --cascade --resume"], (
            after.stdout)
        why = _marker_lines(after.stdout, "[NEXT_COMMAND_WHY]")
        assert why == [f"continues the unfinished cascade at stage 2 of 2 ({STAGES[1]})"], why
        blocked = _run_tokens(world, ["util", "next-command", PLAN_REL, "--blocked",
                                      "improve-plan"])
        assert _marker_lines(blocked.stdout, "[RESUME_COMMAND]") == command, blocked.stdout

    def test_status_names_the_stage_a_stopped_foreground_cascade_continues_at(self, tmp_path):
        world, _stopped = _stopped_in_stage_2(tmp_path)
        status = world.run("status")
        assert status.exit_code in (0, 2), status.explain()
        assert f"stage 2 of 2 ({STAGES[1]})" in status.stdout, status.stdout
        assert "suggested next: generate-tasks" not in status.stdout, status.stdout
        assert "suggested next: improve-plan" in status.stdout, status.stdout


def _stopped_in_stage_2(tmp_path: Path) -> Tuple[CascadeWorld, Run]:
    """A two-stage cascade stopped by a failed batch in stage 2's apply (foreground)."""
    world = CascadeWorld(tmp_path, plan_stages=(FRESH_STAGE_2, FRESH_STAGE_3))
    stopped = world.run("improve-plan", "--cascade", "--models", *STAGES[:2], "--yes",
                        "--exec-subagents", env=world.env(IMPROVE_E2E_FAIL_CALL="5"))
    assert stopped.exit_code == 3, stopped.explain()
    record = _read(world.current("improve-plan") / "record.json")
    assert record["state"] == "running", record
    assert [s["state"] for s in record["stages"]] == ["done", "running"], record["stages"]
    return world, stopped


# --------------------------------------------------------------------------- #
# 4. A middle stage whose one model produced nothing
# --------------------------------------------------------------------------- #

class TestAMiddleStageWithNoResult:

    def test_it_is_skipped_and_the_cascade_completes(self, world):
        run = world.cascade("improve-plan",
                            env=world.env(CASCADE_E2E_FAIL_MODEL=STAGE_MODEL_ARGS[1]))
        assert run.exit_code == 0, run.explain()

        skipped = _marker_lines(run.stdout, "[CASCADE_STAGE_SKIPPED]")
        assert len(skipped) == 1 and skipped[0].startswith(f"2 {STAGES[1]}: "), run.stdout
        assert len(_result_lines(run)) == 1
        result = run.result
        assert (result["state"], result["exit"], result["next"]) == ("done", 0, None), result
        assert "[CASCADE_STAGE_SKIPPED] 2 " in (result.get("notice") or ""), result
        assert "error" not in result, result

        cascade_dir = world.current("improve-plan")
        record = _read(cascade_dir / "record.json")
        assert record["state"] == "finished"
        assert [s["state"] for s in record["stages"]] == ["done", "skipped", "done"]
        reason = record["stages"][1]["skip_reason"]
        assert reason and reason in skipped[0], (reason, skipped)
        assert not (cascade_dir / "stage-2" / "artifacts").exists()
        row = _table_rows((cascade_dir / "summary.md").read_text(encoding="utf-8"))[2]
        assert row[1] == STAGES[1] and row[2] == f"skipped: {reason}", row

        # stage 3 sees one archived round: stage 1's
        ledger = (cascade_dir / "stage-3" / "inputs" / "ledger.md").read_text(encoding="utf-8")
        assert "1 automated review round by" in ledger and "- A2-" not in ledger, ledger
        # the failing model: its attempt and the one transport retry, nothing more
        reviews = [model for model, _p in world.reviews()]
        assert reviews == [STAGE_MODEL_ARGS[0], STAGE_MODEL_ARGS[1], STAGE_MODEL_ARGS[1],
                           STAGE_MODEL_ARGS[2]], reviews
        assert (cascade_dir / "stage-3" / "artifacts" / ".complete.json").is_file()


class TestALastStageWithNoResult:

    def test_the_reviews_printed_retry_keeps_the_stage_and_the_continuation_reuses_it(
            self, world):
        """The last stage's review produced no result and printed its own
        ``Retry with: … review-plan <plan> … --force``. The review ran inside
        the composite's process, so it has no replay flags of its own; the
        retry still names the stage's one model, so run as printed (with the
        host's unattended tokens) its round keeps the stage's block, and the
        cascade's continuation binds that round without dispatching the
        stage's reviewer again."""
        run = world.cascade("improve-plan",
                            env=world.env(CASCADE_E2E_FAIL_MODEL=STAGE_MODEL_ARGS[2]))
        assert run.exit_code == 3, run.explain()
        retry = [line.split("Retry with: ", 1)[1]
                 for line in (run.stdout + run.stderr).splitlines() if "Retry with: " in line]
        assert len(retry) == 1, run.explain()
        argv = _after_dispatcher(retry[0])
        assert argv[0] == "review-plan" and argv[-1] == "--force", argv
        assert _flag_values(argv, "--models") == [STAGES[2]], argv

        reviews = len(world.reviews())
        done = world.dispatch([*argv, "--yes", "--exec-subagents"])
        assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-2000:])
        assert [model for model, _p in world.reviews()][reviews:] == [STAGE_MODEL_ARGS[2]]
        status = _read(world.out / "review-plan" / ".status.json")
        assert status["models_requested"] == [STAGES[2]]
        assert (status["cascade"]["stage"], status["cascade"]["model"]) == (3, STAGES[2])

        reviews = len(world.reviews())
        again = world.run("improve-plan", "--cascade", "--resume", "--yes", "--exec-subagents")
        assert again.exit_code == 0, again.explain()
        assert len(world.reviews()) == reviews, "the continuation dispatched no reviewer"
        record = _read(world.current("improve-plan") / "record.json")
        assert record["state"] == "finished"
        assert [s["state"] for s in record["stages"]] == ["done", "done", "done"]
        assert record["stages"][2]["bound_round"] == status["round_id"]


# --------------------------------------------------------------------------- #
# 5. The native leg: the planner chain as a file-shim host runs it
# --------------------------------------------------------------------------- #

class _Chain:
    """Runs a native cascade the way section 2C of the instruction core words it,
    keeping each call's command and stdout for the report check."""

    def __init__(self, world: CascadeWorld):
        self.world = world
        self.calls: List[Tuple[str, str, int]] = []

    def run(self, argv: List[str], shown: Optional[str] = None) -> subprocess.CompletedProcess:
        done = self.world.dispatch(argv)
        command = shown or "uv run --project /p -- python /p/multi_llm.py " + shlex.join(argv)
        self.calls.append((command, done.stdout, done.returncode))
        return done

    def printed(self, command: str) -> subprocess.CompletedProcess:
        """A command the planner printed, run as printed (``uv run`` aside)."""
        return self.run(_after_dispatcher(command), shown=command)

    def stage(self, out: str) -> None:
        """Each ``run`` step of a stage block, with its flags line and the host's tokens."""
        for line in out.splitlines():
            if not line.startswith("[IMPROVE_STEP] "):
                continue
            _marker, index, word, decision = line.split(" ", 4)[:4]
            if decision != "run":
                continue
            flags = shlex.split(_marker_lines(out, f"[IMPROVE_STEP_FLAGS] {index}")[0])
            done = self.run([word, PLAN_REL, *CODEX_TOKENS, *flags])
            assert done.returncode == 0, (word, flags, done.stdout[-2000:], done.stderr[-2000:])


@pytest.mark.slow
class TestTheNativeLeg:

    def test_a_forced_cascade_runs_each_stage_once_through_the_printed_commands(
            self, world, capsys, monkeypatch):
        chain = _Chain(world)
        typed = ["--force", "--cascade", "--models", *STAGES]
        planned = chain.run(["util", "improve-steps", PLAN_REL, "--mode", "improve-plan",
                             "--host", "codex", *typed])
        assert planned.returncode == 0, planned.stderr
        assert not world.composite_dir("improve-plan").exists(), "improve-steps wrote something"
        assert not _marker_lines(planned.stdout, "[IMPROVE_STEP]")
        first = _marker_lines(planned.stdout, "[IMPROVE_CASCADE_NEXT]")
        assert len(first) == 1 and len(_marker_lines(planned.stdout, "[IMPROVE_RERUN]")) == 1
        first_argv = _after_dispatcher(first[0])
        assert first_argv[:3] == ["util", "improve-cascade", "next"], first_argv
        assert "--force" in first_argv and "--cascade" in first_argv

        out = chain.printed(first[0])
        continues: List[str] = []
        stages_seen: List[str] = []
        for _ in range(len(STAGES) + 1):
            if out.returncode != 0:
                break
            stages_seen += _marker_lines(out.stdout, "[IMPROVE_CASCADE]")
            assert "[IMPROVE_STEP] 1b consolidate skip (cascade)" in out.stdout, out.stdout
            chain.stage(out.stdout)
            latest = _marker_lines(out.stdout, "[IMPROVE_CASCADE_CONTINUE]")
            assert len(latest) == 1, out.stdout
            continues.append(latest[0])
            out = chain.printed(latest[0])

        # -- it completed, once, with no restart -------------------------------------------
        assert out.returncode == 2, (out.stdout, out.stderr)
        complete = _marker_lines(out.stdout, "[CASCADE_COMPLETE]")
        assert len(complete) == 1 and complete[0].startswith("3 stages; table: "), out.stdout
        assert (world.repo / complete[0].split("table: ", 1)[1]).is_file()
        news = [line for line in stages_seen if line.startswith("new cascade ")]
        assert len(news) == 1, stages_seen
        assert [line for line in stages_seen if line.startswith("stage ")] == [
            f"stage {k} of 3: {model}" for k, model in enumerate(STAGES, start=1)]
        assert len(world.cascade_dirs("improve-plan")) == 1
        cascade_dir = world.current("improve-plan")
        record = _read(cascade_dir / "record.json")
        # Each continue command is bound to the cascade and to the stage it
        # follows (`--after-stage <k>`), and to nothing else: no restart flag
        # and no selector, so following them never starts the cascade over.
        assert len(continues) == 3, continues
        for stage, line in enumerate(continues, start=1):
            cont = _after_dispatcher(line)
            assert _flag_values(cont, "--cascade-id") == [record["cascade_id"]], cont
            assert _flag_values(cont, "--after-stage") == [str(stage)], cont
            for flag in ("--force", "--models", "--cascade"):
                assert flag not in cont, (flag, cont)
        assert [model for model, _p in world.reviews()] == list(STAGE_MODEL_ARGS), (
            "a stage was reviewed twice: the cascade restarted")
        _assert_archived(world, "improve-plan", PLAN_DIRS, cascade_dir, STAGES)
        assert LEDGER_HEADING not in world.reviews()[0][1]
        assert all(LEDGER_HEADING in prompt for _m, prompt in world.reviews()[1:])

        # -- the Stop hook's report check over this very chain ------------------------------
        monkeypatch.setattr(report_check, "SETTLE_BUDGET_S", 0.3, raising=False)
        lookup = chain.run(["util", "next-command", PLAN_REL, "--after", "improve-plan",
                            "--host", "codex"])
        assert lookup.returncode == 0, lookup.stderr
        next_command = _marker_lines(lookup.stdout, "[NEXT_COMMAND]")[0]

        finished = _transcript(chain.calls)
        good = finished.say(f"## Improve Plan Summary\n\n### Cascade\n\n```\n{next_command}\n```\n")
        assert run_check(capsys, good.write(world.tmp_path / "done.jsonl")) == ""

        # Cut after stage 2's apply: held to the latest planner call's continuation.
        cut = _calls_through_stage(chain.calls, 2)
        latest = _marker_lines(next(out for cmd, out, _ in reversed(cut)
                                    if "improve-cascade next" in cmd), "[IMPROVE_RERUN]")[0]
        assert latest.endswith("--improve-plan plans/auth-feature.md --cascade --resume"), latest
        held = report_check.last_block(report_check.read_turn(
            _transcript(cut).write(world.tmp_path / "cut.jsonl")))
        assert held is not None and held.continuation and held.commands() == [latest]
        stopped = _transcript(cut).say(f"## Stopped\n\n```\n{latest}\n```\n")
        assert run_check(capsys, stopped.write(world.tmp_path / "stop.jsonl")) == ""
        inner = [line for cmd, out, _ in cut if "improve-cascade" not in cmd
                 for line in _marker_lines(out, "[NEXT_COMMAND]")]
        assert inner, "the phases printed their own standalone next steps"
        wrong = _transcript(cut).say(f"## Stopped\n\n```\n{inner[-1]}\n```\n")
        reason = decision_of(run_check(capsys, wrong.write(world.tmp_path / "wrong.jsonl")))
        assert f"```\n{latest}\n```" in reason["reason"], reason


def _transcript(calls: List[Tuple[str, str, int]]) -> Transcript:
    t = Transcript().prompt("Use the multi-llm skill with exactly these arguments: "
                            f"--improve-plan {PLAN_REL} --force --cascade --models "
                            + " ".join(STAGES))
    for command, output, _code in calls:
        call(t, command, output)
    return t


def _calls_through_stage(calls: List[Tuple[str, str, int]],
                         stage: int) -> List[Tuple[str, str, int]]:
    """The chain's calls up to and including stage *stage*'s apply step."""
    seen = 0
    for position, (command, _out, _code) in enumerate(calls):
        if _after_dispatcher(command)[0] == "apply-suggestions":
            seen += 1
            if seen == stage:
                return calls[:position + 1]
    raise AssertionError(f"no apply step for stage {stage}")


# --------------------------------------------------------------------------- #
# 6. The container row's scenario, kept honest in every plain run
# --------------------------------------------------------------------------- #

class TestTheContainerCascadeScenario:
    """``tests/test_container_improve_cascade.py`` runs only with ``-m container``;
    this keeps its scenario honest at no cost: the chain's rules verbatim behind
    one front rule, and every prompt a real two-stage cascade sends answered by
    the rule that owns it."""

    FRONT = "../container_scenarios/../responses/review_plan/raw_suggestions.json"
    OWNER = {
        "review": "../responses/review_plan/raw_suggestions.json",
        "validate": "../responses/validation/plan_suggestions_valid.json",
        "write": "../responses/write_batches/apply_suggestions.json",
    }

    def test_it_is_one_ledger_rule_then_the_chain_scenario(self):
        cascade = yaml.safe_load(CONTAINER_CASCADE_SCENARIO.read_text(encoding="utf-8"))
        chain = yaml.safe_load(CHAIN_SCENARIO.read_text(encoding="utf-8"))
        assert cascade["prompts"][1:] == chain["prompts"], (
            "copy container_full_chain.yaml's `prompts:` list across verbatim")
        assert cascade["on_unmatched"] == chain["on_unmatched"] == "fail"
        assert cascade["prompts"][0]["fixture"] == self.FRONT
        scenario = scenario_matcher.load_scenario(CONTAINER_CASCADE_SCENARIO)
        for rule in scenario.rules:
            path = rule.response.path(scenario.dir)
            assert path is not None and path.is_file(), rule.response.fixture
        front = scenario.rules[0].response.path(scenario.dir)
        assert front.read_bytes() == (RESPONSES / "review_plan" / "raw_suggestions.json"
                                      ).read_bytes(), "the front rule must serve the chain's answer"

    def test_each_prompt_of_a_two_stage_cascade_resolves_to_its_owner(self, world):
        run = world.run("improve-plan", "--cascade", "--models", *STAGES[:2], "--yes",
                        "--exec-subagents")
        assert run.exit_code == 0, run.explain()
        scenario = scenario_matcher.load_scenario(CONTAINER_CASCADE_SCENARIO)
        stage, seen = 0, set()
        for entry in world.calls():
            kind = classify(entry)
            prompt = entry["prompt"]
            matches = [rule.response.fixture for rule in scenario.matching_rules(prompt)]
            resolved = scenario_matcher.resolve(scenario, prompt)
            if kind == "review":
                stage += 1
                if stage == 1:
                    assert matches == [self.OWNER["review"]], matches
                else:
                    assert matches == [self.FRONT, self.OWNER["review"]], matches
                    assert resolved.fixture == self.FRONT, resolved
            else:
                assert matches == [self.OWNER[kind]], (kind, matches, prompt[:200])
            seen.add(kind)
        assert stage == 2 and seen == set(self.OWNER), (stage, seen)

        # The verdict sheet joins BY GROUP HASH: the stage's groups must be among
        # its rows, or the container row's validation lands on validation_failed.
        sheet = _read(FIXTURES / "responses" / "validation" / "plan_suggestions_valid.json")
        hashes = {str(row.get("group_hash") or row.get("group_id")) for row in sheet["groups"]}
        stage_1 = world.current("improve-plan") / "stage-1" / "artifacts" / "review-plan"
        grouped = {group["group_hash"] for group in _read(stage_1 / "grouped.json")["groups"]}
        assert len(grouped) == len(PLAN_STAGE_1) and grouped <= hashes, sorted(grouped - hashes)
