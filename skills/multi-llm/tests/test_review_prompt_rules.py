"""The review rules in the plan-review and code-review fan-out prompts.

Plan: sharper-review-prompts. Six wording changes (R1-R6) to
``prompts/plan_review.txt`` and ``prompts/code_review.txt``, and none anywhere
else in the pipeline. The wording is the design, so each class below pins its
own change's sentences verbatim, as the reviewer receives them: the template
rendered with dummy context, the working-limits block and an output-location
block.

Around those pins:

- the wording the changes replaced is gone from the template files, and the
  shared schema's ``desc`` description no longer asks for a fix;
- each prompt still opens with the sentence its container scenario matches;
- the plan prompt's example array is valid review output, with the three
  descriptions the plan wrote (whether those carry a fix was judged once, by
  reading them; this pin makes any later edit fail, so it is read again);
- neither rendered prompt, whole, trips a failure classifier under any
  adapter's pattern sets, since four adapters echo the prompt into the stream
  their run is classified from.
"""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.llm_client import (  # noqa: E402
    FILE_OUTPUT_LOCATION,
    STDOUT_OUTPUT_LOCATION,
    render_working_limits,
)
from utils.providers import PROVIDER_CLASSES  # noqa: E402
from utils.providers.base import classify_streams, match_streams  # noqa: E402

PROMPTS_DIR = SKILL_ROOT / "prompts"
SUGGESTIONS_SCHEMA = SKILL_ROOT / "schemas" / "review_suggestions.schema.json"
CONTAINER_SCENARIO = (
    SKILL_ROOT / "tests" / "fixtures" / "e2e" / "scenarios" / "container_full_chain.yaml"
)

PLAN_PROMPT = "plan_review.txt"
CODE_PROMPT = "code_review.txt"
REVIEW_PROMPTS = (PLAN_PROMPT, CODE_PROMPT)

#: The phase's own prompt-context keys (`review_plan_orchestrator`, and
#: `code_review_orchestrator`'s `prompt_context`), with dummy values. The
#: values are deliberately bland, so a classifier trip below is the
#: template's and not this fixture's.
_CONTEXT = {
    PLAN_PROMPT: {"plan_path": "/work/plans/p.md", "prefix": "p"},
    CODE_PROMPT: {
        "plan_path": "/work/plans/p.md",
        "tracked_files_list": "- src/app.py\n- tests/test_app.py",
        "scope_source": "the files the implement phase recorded",
        "base_ref": "0123abcd",
        "prefix": "p",
        "implementation_record": "No implementation record was written.",
        "scope_file": "/work/plans/p/code-review/scope.json",
        "diff_dir": "/work/plans/p/code-review/diff",
        "diff_access": "Each file's diff is under /work/plans/p/code-review/diff.",
    },
}

#: Where each phase's per-model result file lands, for the file block.
_RESULT_PATH = {
    PLAN_PROMPT: "/work/plans/p/review-plan/claude-code_sonnet.json",
    CODE_PROMPT: "/work/plans/p/code-review/claude-code_sonnet.json",
}

#: The two `{output_location}` blocks `render_output_location` chooses from.
LOCATIONS = ("file", "stdout")


def _template(name):
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _render(name, location="file"):
    """The prompt a reviewer receives: the three keys `invoke_with_file_output`
    injects on top of the phase's own context."""
    path = _RESULT_PATH[name]
    block = (
        FILE_OUTPUT_LOCATION.format(path=path)
        if location == "file"
        else STDOUT_OUTPUT_LOCATION
    )
    return _template(name).format(
        **_CONTEXT[name],
        output_json_path=path,
        output_location=block,
        working_limits=render_working_limits(1200),
    )


def _lines(name):
    return _render(name).splitlines()


def _next_non_blank(lines, index):
    return next(line for line in lines[index + 1:] if line.strip())


# ---------------------------------------------------------------------------
# One class per change, each pinning its own sentences verbatim
# ---------------------------------------------------------------------------

class TestR1PlanReviewSkipsWhatAnImplementerSortsOut:
    """Plan reviewers report what an implementer needs to be told, and the
    `low` rubric and `## Your Task` stop inviting style findings."""

    STEP_2 = "2. Analyze it thoroughly for issues and gaps"
    STEP_3 = "3. Report each issue you find, specifically"
    SKIP_LINE = (
        "Report what an implementer needs to be told. Skip anything an "
        "implementer would sort out without asking, such as a line number that "
        "has moved, a detail they can fill in from the code, or wording and "
        "formatting. If you are unsure whether an implementer could sort a "
        "detail out without asking, report it."
    )
    LOW_RUBRIC = (
        "  - low = a real problem with a small impact (not a style or wording "
        "preference)"
    )

    def test_steps_2_and_3_are_reworded(self):
        lines = _lines(PLAN_PROMPT)
        assert self.STEP_2 in lines
        assert self.STEP_3 in lines

    def test_the_skip_line_follows_the_task_list(self):
        text = _render(PLAN_PROMPT)
        assert self.SKIP_LINE in _lines(PLAN_PROMPT)
        # "Then add this line after the list": below step 3, above the checks.
        assert (
            text.index(self.STEP_3)
            < text.index(self.SKIP_LINE)
            < text.index("## What to Look For")
        )

    def test_the_low_rubric_asks_for_a_real_problem(self):
        assert self.LOW_RUBRIC in _lines(PLAN_PROMPT)

    def test_the_medium_and_high_rubrics_are_unchanged(self):
        lines = _lines(PLAN_PROMPT)
        assert (
            "  - high = critical gaps, security issues, architectural flaws, "
            "blockers"
        ) in lines
        assert (
            "  - medium = missing details, unclear requirements, potential issues"
        ) in lines


class TestR2AlternativesAnswerThePlansReasoning:
    """An alternative to a choice the plan explains says why the plan's stated
    reason does not hold."""

    BULLET = (
        "- **Alternative approaches**: Is there a simpler, more effective, or "
        "fundamentally different way to achieve the same goal? If you propose "
        "an alternative to a choice the plan explains, say why its stated "
        "reason doesn't hold or what it misses."
    )

    def test_the_bullet_carries_the_condition(self):
        assert self.BULLET in _lines(PLAN_PROMPT)


class TestR3FindingsCarryEvidenceNotAFix:
    """`desc` carries the issue, its evidence where it helps, and the intended
    value when the plan shows it; a fix only for a design choice the plan does
    not settle."""

    PLAN_DESC = (
        '- "desc": string (what is wrong and why it matters. Where it helps, '
        "add the evidence: the plan text it rests on, with its section and a "
        "short quote, or a repository location as path:line, only if you "
        "opened that file and checked what is there. If the plan or a file you "
        "opened shows which value or behaviour is intended, say which, for "
        "example which of two numbers that disagree is the right one. You "
        "don't need to propose a fix; whoever applies the finding decides it. "
        "But if the right fix is a design choice the plan does not settle, "
        "propose the one you would make, so that validation can weigh it.)"
    )
    CODE_DESC = (
        '- "desc": string, 2-4 sentences: what is wrong and why it matters, '
        "with the evidence where it helps (what in the cited lines shows it, "
        "another location as path:line, or the plan text the change breaks). "
        "If the plan or the surrounding code shows the intended behaviour, say "
        "what it is. You don't need to propose a fix; whoever applies the "
        "finding decides it. But if the right fix is a design choice that "
        "neither settles, propose the one you would make, so that validation "
        "can weigh it. (required)"
    )
    CODE_SHAPE_DESC = (
        '"desc": "What is wrong and why it matters, with evidence where it helps"'
    )

    def test_the_plan_prompt_desc_rule(self):
        assert self.PLAN_DESC in _lines(PLAN_PROMPT)

    def test_the_code_prompt_desc_rule(self):
        assert self.CODE_DESC in _lines(CODE_PROMPT)

    def test_the_code_prompt_shape_example(self):
        assert self.CODE_SHAPE_DESC in _render(CODE_PROMPT)


class TestR4TestsThatCannotFail:
    """Look for checks that would still pass with the behaviour they are
    offered as evidence for missing or broken."""

    PLAN_BULLET = (
        "- **Missing tests**: Is the testing strategy adequate? Would a test or "
        "acceptance criterion offered as evidence for a behaviour still pass "
        "with that behaviour missing or broken, or is any check unable to fail?"
    )
    CODE_BULLET = (
        "- **Tests that test nothing**: Tests in the change offered as evidence "
        "for a behaviour that would still pass with that behaviour missing or "
        "broken: they never reach the code that implements it, mock out that "
        "code, assert something that cannot fail, or are skipped "
        "unconditionally or for a reason that does not apply"
    )

    def test_the_plan_prompt_missing_tests_bullet(self):
        assert self.PLAN_BULLET in _lines(PLAN_PROMPT)

    def test_the_code_prompt_bullet_follows_missing_functionality(self):
        lines = _lines(CODE_PROMPT)
        index = lines.index(
            "- **Missing functionality**: Plan requirements not implemented"
        )
        assert lines[index + 1] == self.CODE_BULLET


class TestR5ConcreteConsistencyChecks:
    """`Conflicting statements` becomes three concrete checks, and `Naming
    consistency` goes: its "consistent" half is the two-names check."""

    BULLETS = [
        "- **Conflicting statements**: Do parts of the plan contradict each "
        "other? Look especially for numbers or limits that differ between "
        "sections, and for one thing called by two names.",
        "- **Dangling references**: Does the plan refer to a section, decision "
        "or ID of its own that it never defines? A reference to one defined in "
        "a document the plan names, such as another plan, a tasks file or a "
        "file in the repository, is not dangling.",
        "- **Questions already answered**: Does the plan list as open a "
        "question it answers elsewhere?",
    ]

    def test_the_three_bullets_replace_conflicting_statements_in_order(self):
        lines = _lines(PLAN_PROMPT)
        index = lines.index(self.BULLETS[0])
        assert lines[index:index + 3] == self.BULLETS


class TestR6SecretsAndDataExposure:
    """No secret's value in any finding field, and data exposure is a security
    finding in both prompts."""

    PLAN_SECURITY = (
        "- **Security concerns**: Are there potential vulnerabilities? Could "
        "data leak through logs, telemetry, caches, test output or committed "
        "files?"
    )
    CODE_SECURITY = (
        "- **Security issues**: Vulnerabilities, unsafe patterns, and data "
        "exposure through logs, telemetry, caches, test output or committed "
        "files"
    )
    SECRET_LINE = (
        "If a finding involves a secret (a key, token or password), say where "
        "it is and never copy its value into any field. That includes "
        "evidence: cite where the value is, and leave it out of any quote."
    )
    ANCHOR_REQUIREMENT = (
        '- "anchor_text": string, 10-50 char code snippet for locating issue '
        "(required)"
    )
    ANCHOR_SECRET = (
        "  - for a secret, use a snippet without the value, such as the "
        "assignment's left-hand side (\"self.api_key =\"). If no 10-50 char "
        "snippet leaves the value out, a shorter snippet or \"\" is fine; give "
        "line_range so file and line_range locate it"
    )

    def test_the_plan_prompt_security_bullet(self):
        assert self.PLAN_SECURITY in _lines(PLAN_PROMPT)

    def test_the_code_prompt_security_bullet(self):
        assert self.CODE_SECURITY in _lines(CODE_PROMPT)

    @pytest.mark.parametrize("name", REVIEW_PROMPTS)
    def test_the_secret_line_sits_directly_above_the_empty_array_rule(self, name):
        lines = _lines(name)
        assert lines.count(self.SECRET_LINE) == 1
        index = lines.index(self.SECRET_LINE)
        assert _next_non_blank(lines, index).startswith(
            "If no issues are found, return an empty array"
        )

    def test_the_anchor_requirement_is_unchanged_and_gains_the_sub_bullet(self):
        lines = _lines(CODE_PROMPT)
        index = lines.index(self.ANCHOR_REQUIREMENT)
        assert lines[index + 1] == self.ANCHOR_SECRET


# ---------------------------------------------------------------------------
# What the changes replaced
# ---------------------------------------------------------------------------

class TestReplacedWordingIsGone:
    """Read from the template files, so nothing the renderer adds can hide a
    phrase or supply one."""

    @pytest.mark.parametrize("name", REVIEW_PROMPTS)
    @pytest.mark.parametrize(
        "phrase",
        [
            "recommended fix",
            "explaining problem and fix",
            "style suggestions",
            "nice-to-haves",
        ],
    )
    def test_gone_from_both_prompts(self, name, phrase):
        assert phrase not in _template(name)

    @pytest.mark.parametrize(
        "phrase",
        [
            "Provide specific, actionable feedback",
            "areas needing improvement",
            "Naming consistency",
        ],
    )
    def test_gone_from_the_plan_prompt(self, phrase):
        assert phrase not in _template(PLAN_PROMPT)


class TestSuggestionsSchemaDescription:
    """The prompt checks never read the schema, so without this the old
    "recommended fix" description could survive every other test. Plain
    `json.load`, so it never skips for want of `jsonschema`."""

    def test_desc_defers_to_the_phase_prompt(self):
        with SUGGESTIONS_SCHEMA.open(encoding="utf-8") as handle:
            schema = json.load(handle)

        assert schema["items"]["properties"]["desc"]["description"] == (
            "Text explaining the issue, in the form the phase's prompt asks for"
        )

    def test_recommended_fix_is_nowhere_in_the_file(self):
        assert "recommended fix" not in SUGGESTIONS_SCHEMA.read_text(encoding="utf-8")


class TestOpeningSentencesAreUnchanged:
    """The container scenario routes each fan-out turn by its opening
    sentence; a reworded first line would leave the turn unmatched."""

    OPENING = {
        PLAN_PROMPT: (
            "You are reviewing an implementation plan for completeness, "
            "correctness, and clarity."
        ),
        CODE_PROMPT: "You are reviewing code changes that implement a feature plan.",
    }
    SCENARIO_DESCRIPTION = {
        PLAN_PROMPT: "review-plan fan-out: prompts/plan_review.txt",
        CODE_PROMPT: "review-code fan-out: prompts/code_review.txt",
    }

    @pytest.mark.parametrize("name", REVIEW_PROMPTS)
    def test_the_first_line_is_the_scenario_sentence(self, name):
        rows = yaml.safe_load(CONTAINER_SCENARIO.read_text(encoding="utf-8"))["prompts"]
        (row,) = [r for r in rows if r.get("description") == self.SCENARIO_DESCRIPTION[name]]

        assert _template(name).splitlines()[0] == self.OPENING[name]
        # The pattern spells the sentence out, its one period escaped.
        assert row["pattern"].endswith(self.OPENING[name].replace(".", "\\."))
        # The pattern's negative lookahead must not catch our own text either.
        assert re.search(row["pattern"], _render(name))


# ---------------------------------------------------------------------------
# The plan prompt's examples
# ---------------------------------------------------------------------------

#: §2 R3's three example descriptions, verbatim and in order. None of them
#: carries a fix; that was judged by reading them, and this pin makes any edit
#: fail so it is read again. (No keyword check: "Use", "Set" and "The fix is"
#: slip past a word list, and an evidence quote with "should" trips one.)
EXAMPLE_DESCS = [
    "Section 2 caps uploads at 'at most 50 files per request', but the sizing "
    "table in Section 4 assumes 100 per request, so its memory estimate is "
    "twice what Section 2 allows. Section 2 states the cap as a rule and the "
    "table only assumes a figure, so the table's 100 is the wrong one. No step "
    "depends on that estimate, so the impact is small.",
    "Section 3 attaches every retry to an 'on_upload_failed hook', but no "
    "section of the plan defines that hook or says which component calls it. "
    "Every retry step in Section 3 rests on it.",
    "The only acceptance criterion for retries is 'the upload command exits "
    "0', which it already does today. It would pass with retries missing, so "
    "nothing checks that a dropped upload is sent again.",
]


def _plan_examples():
    """The example array, decoded from its opening `[` to its matching `]`.

    Not sliced up to "If no issues are found": R6's secret line sits between
    the array and that sentence, so that span is not JSON.
    """
    text = _render(PLAN_PROMPT)
    start = text.index("[", text.index("Example output:"))
    examples, _end = json.JSONDecoder().raw_decode(text, start)
    return examples


class TestPlanExamplesAreValidOutput:
    def test_the_examples_parse_to_three_findings(self):
        examples = _plan_examples()

        assert isinstance(examples, list)
        assert len(examples) == 3
        assert all(isinstance(example, dict) for example in examples)

    def test_the_descriptions_are_the_plans_verbatim(self):
        assert [example["desc"] for example in _plan_examples()] == EXAMPLE_DESCS

    def test_the_examples_validate_against_the_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SUGGESTIONS_SCHEMA.read_text(encoding="utf-8"))

        jsonschema.validate(_plan_examples(), schema)


# ---------------------------------------------------------------------------
# Echo safety
# ---------------------------------------------------------------------------

class TestRenderedPromptsTripNoFailureClassifier:
    """goose, muse, pi and omp echo the prompt into their stream, and
    the old example's "timeout scenarios" once classified a clean goose run as
    a timeout. The whole final prompt, under every adapter's own pattern sets:
    pi and omp add "No API key found for", cline adds "incorrect api key", and
    R6 put key, token and password wording next to that vocabulary. A trip is
    reworded at its source, never excluded here."""

    @pytest.mark.parametrize("provider", sorted(PROVIDER_CLASSES))
    @pytest.mark.parametrize("location", LOCATIONS)
    @pytest.mark.parametrize("name", REVIEW_PROMPTS)
    def test_neither_stream_classifies(self, name, location, provider):
        cls = PROVIDER_CLASSES[provider]
        patterns = dict(
            auth_patterns=cls.auth_error_patterns,
            rate_limit_patterns=cls.rate_limit_patterns,
            exit_zero_markers=cls.exit_zero_failure_markers,
        )
        rendered = _render(name, location)

        for stdout, stderr in ((rendered, ""), ("", rendered)):
            verdict = classify_streams(stdout, stderr, **patterns)
            # `match_streams` only names the text that fired, for the message.
            assert verdict is None, _describe_trip(stdout, stderr, patterns)


def _describe_trip(stdout, stderr, patterns):
    match = match_streams(stdout, stderr, **patterns)
    if match is None or match.offset is None:
        return match
    text = (stdout if match.channel == "stdout" else stderr).lower()
    return (match.status, match.pattern, text[max(0, match.offset - 60):match.offset + 60])
