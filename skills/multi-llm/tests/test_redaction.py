#!/usr/bin/env python3
"""Redaction over logs and failure artifacts (§4.5), and the caps it must precede (§4.4).

Nine realistic secret shapes planted in provider output landed **unredacted** in
``log_*.txt`` on a *successful* run, in ``failed/*.attemptN.json``, in
``subagent_batch_report.json`` and in the exit-3 stdout block that becomes
``orchestrator-run.log`` under ``--detach``. One boundary closes all four:
``write_safety.redact_text``.

Three ship gates, per §4.5.4:

1. **Recall — 17/17**, asserted shape by shape so a regression names the shape it
   lost rather than a count.
2. **Zero false positives** over the benign corpora — ``redact_text(c) == c`` for
   every golden container turn, every side turn and every e2e response fixture.
   The repo's canonical fixture is an *auth plan*, dense in the vocabulary a
   careless pattern set shreds, and the e2e fixtures **contain code**, which is
   where a delimiter-anchored set without ``_looks_like_credential`` still fails.
   ``utils/**/*.py`` is scanned as a **diagnostic, not a gate** (§4.5.2): it is
   source, never a redaction input, and its two known hits are named there.
3. **No drift from the committed scanner** (§4.5.5) — every credential shape
   ``.gitleaks.toml`` detects is also redacted by ``redact_text``. One-way on
   purpose: ``redact_text`` is a strict superset and its patterns are never
   copied back into the TOML, which would make the scanner red on day one against
   this repo's committed decoys. Vectors are **synthesized**, shared with
   ``test_no_committed_secrets``, and never committed.

And the group gate the two items form together (§14.4): **redaction runs before
truncation**. Both caps cut by construction, and a cut applied first slices a
secret into a remnant no pattern can match and no downstream filter can recover —
the last-20-lines rule decapitates a PEM block's ``-----BEGIN`` header, and a
64 KiB tail keeps a value whose key name has been cut away. Those two are
asserted directly, not inferred from the halves being green.
"""

import json
import random
import string
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import llm_client, subagent_executor, write_safety  # noqa: E402
from utils.write_safety import redact_argv, redact_text  # noqa: E402

# The synthesized-vector table is shared with the committed-secret suite so the
# two cannot drift in their fixtures (§4.5.5). Nothing here is a committed literal
# for a shape that table already covers.
from .test_no_committed_secrets import (  # noqa: E402
    CONFIG_PATH,
    VECTORS,
    _body,
    _run_gitleaks,
    _synthesize,
    requires_gitleaks,
)

SKILL_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The seventeen shapes (§4.5.4 test 1)
# ---------------------------------------------------------------------------

_RNG = random.Random("multi-llm redaction recall")

#: The charset for the two hex vectors below. A 32-char lowercase-hex token
#: carries no vendor prefix, so its charset *is* its shape — and it is also what
#: ``write_safety._looks_like_credential`` gates on (a digit, a case mix, or one
#: of ``-_./+=``), which is why the alphabet is named rather than improvised.
_HEX = string.digits + "abcdef"

_GOOGLE_KEY = "AIzaSy" + _body(_RNG, 33)
_HF_TOKEN = "hf_" + _body(_RNG, 34)
_GITLAB_PAT = "glpat-" + _body(_RNG, 22)
#: ``xoxb-`` + two ten-digit segments + a 24-char body. The segments are drawn
#: from the seeded RNG rather than written out: ``slack-bot-token`` detects the
#: prefix-plus-digits *rendering* on its own, so a committed pair of literal
#: segments is a finding in this file even with the body assembled at runtime.
_SLACK_TOKEN = (
    "xoxb-"
    + _body(_RNG, 10, string.digits)
    + "-"
    + _body(_RNG, 10, string.digits)
    + "-"
    + _body(_RNG, 24)
)
_OPENAI_KEY = "sk-proj-" + _body(_RNG, 48)
_STRIPE_ISH = "sk-live-" + _body(_RNG, 24)
#: The three delimited key/value vectors (P1). Synthesized like every other
#: shape: a 32-char hex literal beside ``token=``, or a passphrase beside
#: ``password =``, is precisely what ``generic-api-key`` fires on, so writing
#: them out would make this file the one committed secret nobody scans. The
#: shapes are preserved exactly — only the concrete characters differ.
_TOKEN_HEX = _body(_RNG, 32, _HEX)
_CLIENT_SECRET_HEX = _body(_RNG, 32, _HEX)
_PASSWORD = "-".join(
    (
        _body(_RNG, 10),
        _body(_RNG, 5, string.ascii_lowercase),
        _body(_RNG, 7, string.ascii_lowercase),
    )
)
_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkNJIn0"
    ".dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
)
_PEM_BODY = [_body(_RNG, 64) for _ in range(24)]
_PEM = "\n".join(
    ["-----BEGIN RSA PRIVATE KEY-----", *_PEM_BODY, "-----END RSA PRIVATE KEY-----"]
)

#: ``(shape, rendered line, the value(s) that must not survive)``.
SHAPES = [
    (
        "openai-project-key",
        f"provider refused the call: invalid api key {_OPENAI_KEY}",
        (_OPENAI_KEY,),
    ),
    (
        "anthropic-api-key",
        f"env ANTHROPIC_API_KEY resolved to {_synthesize('anthropic-api-key', 'recall')}",
        (_synthesize("anthropic-api-key", "recall"),),
    ),
    ("slack-bot-token", f"posting to #ci with {_SLACK_TOKEN}", (_SLACK_TOKEN,)),
    (
        "github-pat",
        f"remote rejected, retrying with {_synthesize('github-pat', 'recall')}",
        (_synthesize("github-pat", "recall"),),
    ),
    ("google-api-key", f"GET /v1/models?key={_GOOGLE_KEY}", (_GOOGLE_KEY,)),
    ("authorization-bearer", f"Authorization: Bearer {_JWT}", (_JWT,)),
    (
        "token-kv",
        f"resolved token={_TOKEN_HEX} from the keyring",
        (_TOKEN_HEX,),
    ),
    (
        "password-kv",
        f"connecting with password = {_PASSWORD}",
        (_PASSWORD,),
    ),
    (
        "client-secret-kv",
        f"oauth exchange failed: client_secret: {_CLIENT_SECRET_HEX}",
        (_CLIENT_SECRET_HEX,),
    ),
    ("aws-access-key-id", "aws sts caller AKIAIOSFODNN7EXAMPLE", ("AKIAIOSFODNN7EXAMPLE",)),
    (
        "aws-secret-access-key",
        "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",),
    ),
    ("gitlab-pat", f"glab auth login --token {_GITLAB_PAT}", (_GITLAB_PAT,)),
    ("huggingface-token", f"HUGGINGFACE_HUB_TOKEN={_HF_TOKEN}", (_HF_TOKEN,)),
    (
        "quoted-json-api-key",
        json.dumps({"api_key": _STRIPE_ISH}),
        (_STRIPE_ISH,),
    ),
    (
        "url-basic-auth",
        "git clone https://ci:s3cr3tP4ssw0rd@git.example.com/acme/repo.git",
        ("s3cr3tP4ssw0rd",),
    ),
    ("pem-private-key", _PEM, tuple(_PEM_BODY)),
    (
        "cursor-identity-envelope",
        json.dumps(
            {
                "userEmail": "ada@example.com",
                "terminalProgram": "iTerm.app",
                "shell": "/bin/zsh",
            }
        ),
        ("ada@example.com", "iTerm.app", "/bin/zsh"),
    ),
]

assert len(SHAPES) == 17, "§4.5.4 names seventeen shapes; the fixture must carry all of them"
assert len(_GOOGLE_KEY) == 39, "P6's bug was a length bug: the fixture must be a real 39-char key"


# ---------------------------------------------------------------------------
# Corpora (§4.5.4 test 2)
# ---------------------------------------------------------------------------


def _benign_corpus():
    """Exactly the files §4.5.4 names, as ``(label, path)``."""
    turns = sorted((SKILL_DIR / "tests/golden/container_turns").glob("*.txt"))
    side = sorted((SKILL_DIR / "tests/golden/container_turns/side_turns").glob("*.txt"))
    responses = sorted(
        p for p in (SKILL_DIR / "tests/fixtures/e2e/responses").rglob("*") if p.is_file()
    )
    assert len(turns) == 12, f"expected 12 container turns, found {len(turns)}"
    assert len(side) == 7, f"expected 7 side turns, found {len(side)}"
    assert len(responses) == 33, f"expected 33 e2e response fixtures, found {len(responses)}"
    return [
        (str(p.relative_to(SKILL_DIR)), p) for p in [*turns, *side, *responses]
    ]


BENIGN_CORPUS = _benign_corpus()


# ---------------------------------------------------------------------------
# 1. Recall
# ---------------------------------------------------------------------------


class TestRecallSeventeenShapes:
    """Shape by shape, so a regression names what it lost."""

    @pytest.mark.parametrize("shape,line,planted", SHAPES, ids=[s[0] for s in SHAPES])
    def test_shape_is_redacted(self, shape, line, planted):
        out = redact_text(line)
        for value in planted:
            assert value not in out, (
                f"{shape}: the planted value survived redaction.\n"
                f"in : {line[:200]!r}\nout: {out[:200]!r}"
            )

    @pytest.mark.parametrize("shape,line,planted", SHAPES, ids=[s[0] for s in SHAPES])
    def test_shape_is_redacted_inside_a_log_line(self, shape, line, planted):
        """The same shape embedded in a rendered log line, not alone on one."""
        wrapped = f"2026-08-16T10:00:00 [gemini] stderr | {line} (attempt 2)"
        out = redact_text(wrapped)
        for value in planted:
            assert value not in out, f"{shape}: survived when embedded in a log line"

    @pytest.mark.parametrize("shape,line,planted", SHAPES, ids=[s[0] for s in SHAPES])
    def test_surrounding_text_survives(self, shape, line, planted):
        """The *value* is replaced, never the whole line — a redacted log stays usable."""
        out = redact_text(line)
        assert out.strip(), f"{shape}: redaction emptied the line"
        assert "<redacted:" in out, f"{shape}: nothing was marked as redacted"

    def test_redaction_is_idempotent(self):
        """A second pass must not re-redact its own markers."""
        for shape, line, _ in SHAPES:
            once = redact_text(line)
            assert redact_text(once) == once, f"{shape}: redaction is not idempotent"

    def test_cursor_privacy_fields_are_redacted_raw_and_embedded(self):
        """§4.5.4's privacy cases: the cursor ``about``/``status`` envelope."""
        envelope = json.dumps(
            {
                "userEmail": "grace@example.org",
                "terminalProgram": "Apple_Terminal",
                "shell": "/usr/bin/fish",
            }
        )
        for text in (envelope, f"cursor-agent about --format json -> {envelope}"):
            out = redact_text(text)
            for value in ("grace@example.org", "Apple_Terminal", "/usr/bin/fish"):
                assert value not in out, f"{value} survived in {text[:60]!r}"


class TestBothShapesTheProbeBoundaryReallyEmits:
    """P13 over the *other* `about` shape, and over a short value.

    `cursor-agent about` is a declared ``setup_probe`` now, so its output
    reaches a rendered status row and an exit-3 refusal tail. A build that does
    not know ``--format`` prints the ``Key  Value`` table
    ``providers.cursor_agent._about_text_pairs`` parses — the shape the
    quoted-JSON rule cannot see at all.
    """

    TEXT_TABLE = (
        "Cursor Agent\n"
        "CLI Version       2026.08.11-e8db854\n"
        "Subscription      pro\n"
        "User Email        grace@example.org\n"
        "Terminal Program\tApple_Terminal\n"
        "Shell: /usr/bin/fish\n"
        "OS Platform       linux\n"
    )

    def test_the_text_table_fallback_is_redacted_too(self):
        out = redact_text(self.TEXT_TABLE)
        for value in ("grace@example.org", "Apple_Terminal", "/usr/bin/fish"):
            assert value not in out, f"{value!r} survived the text-table shape"

    def test_the_text_table_keeps_every_diagnostic_value(self):
        """A row with its version and platform gone is not a usable readout."""
        out = redact_text(self.TEXT_TABLE)
        for value in ("2026.08.11-e8db854", "pro", "linux"):
            assert value in out, f"{value!r} is diagnostic, not identity"
        for label in ("User Email", "Terminal Program", "Shell"):
            assert label in out, f"{label!r} was dropped rather than redacted in place"

    def test_the_text_table_redaction_is_idempotent(self):
        once = redact_text(self.TEXT_TABLE)
        assert redact_text(once) == once

    def test_a_two_character_json_value_is_not_below_the_floor(self):
        """The old `{3,}` bound let a short identity value through."""
        out = redact_text(json.dumps({"shell": "sh", "cliVersion": "2026.08.11"}))
        assert '"sh"' not in out
        assert "2026.08.11" in out

    @pytest.mark.parametrize(
        "line",
        [
            "the shell = bash default is fine",
            "Set email to whatever the reviewer prefers.",
            "  - shell: bash",
            "run `SHELL=/bin/sh make test` first",
        ],
    )
    def test_prose_and_source_keep_the_same_words(self, line):
        """Key-named *and* line-anchored: the reason this rule can exist at all."""
        assert redact_text(line) == line


class TestATokenCountIsNotAToken:
    """The P1 key half matches ``token`` inside every counter claude-code prints.

    Measured 2026-09-17 on a real ``claude -p --output-format json`` failure
    envelope: ``"cacheReadInputTokens":<eight digits>`` was rewritten to
    ``<redacted:secret-kv>`` in ``log_claude-code_*.txt``, which left the log's
    one JSON document unparseable and the salvage gate reading a mangled
    excerpt. A credential is never a bare integer.
    """

    ENVELOPE = (
        '{"type":"result","is_error":true,"api_error_status":429,'
        '"usage":{"input_tokens":1936,"output_tokens":137348,'
        '"cache_read_input_tokens":12345678},'
        '"modelUsage":{"claude-fable-5-1":{"inputTokens":1936,'
        '"outputTokens":137348,"cacheReadInputTokens":12345678,'
        '"cacheCreationInputTokens":1065254}},"total_cost_usd":25.05}'
    )

    def test_an_eight_digit_token_count_survives(self):
        assert redact_text(self.ENVELOPE) == self.ENVELOPE

    def test_the_envelope_still_parses_after_redaction(self):
        assert json.loads(redact_text(self.ENVELOPE))["usage"]["cache_read_input_tokens"] == 12345678

    def test_a_token_that_carries_letters_is_still_redacted(self):
        """The floor is "bare integer", not "starts with a digit"."""
        out = redact_text("token=1234567890abcdef1234567890abcdef")
        assert "1234567890abcdef1234567890abcdef" not in out
        assert "<redacted:secret-kv>" in out


# ---------------------------------------------------------------------------
# 2. Zero false positives
# ---------------------------------------------------------------------------


class TestNoFalsePositivesOnBenignCorpora:
    """The auth-plan prose and the code-carrying e2e fixtures, unchanged."""

    @pytest.mark.parametrize(
        "label,path", BENIGN_CORPUS, ids=[label for label, _ in BENIGN_CORPUS]
    )
    def test_benign_file_is_returned_byte_for_byte(self, label, path):
        content = path.read_text(encoding="utf-8", errors="replace")
        out = redact_text(content)
        if out != content:
            diffs = [
                (i, a, b)
                for i, (a, b) in enumerate(zip(content.splitlines(), out.splitlines()))
                if a != b
            ]
            pytest.fail(
                f"{label}: redaction rewrote {len(diffs)} line(s) of a benign corpus "
                f"file. First: line {diffs[0][0] + 1}\n  in : {diffs[0][1][:200]!r}\n"
                f"  out: {diffs[0][2][:200]!r}"
            )

    #: §4.5.2 records two known hits over ``utils/**/*.py``; the tree carries a
    #: **third** of exactly the same class, and it is not a pattern that got
    #: looser: ``ENV_GIT_CREDENTIAL_HELPER_VAR = "MULTI_LLM_GIT_CREDENTIAL_HELPER"``
    #: has been in provider_registry.py since 2026-06-25, seven weeks before the
    #: measurement, in a tree that already had the 66 files that note counts. All
    #: three are Python constant assignments — source, which is never a
    #: redaction input. A **fourth** joined them when the executor's
    #: nested-exec reason codes were given a name:
    #: ``"nested_exec_credentials": "nested_exec.credentials"`` maps a
    #: refusal code to the capability field it reports about, and the only
    #: thing credential-shaped in it is the word. Recorded here so nobody
    #: re-derives any of them as a regression.
    KNOWN_SOURCE_HITS = 5


    def test_utils_source_tree_is_a_diagnostic_not_a_gate(self):
        """Source is never a redaction input; this is a tripwire on looseness."""
        hits = []
        for path in sorted((SKILL_DIR / "utils").rglob("*.py")):
            content = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(content.splitlines(), 1):
                if redact_text(line) != line:
                    hits.append(f"{path.relative_to(SKILL_DIR)}:{lineno}: {line.strip()[:120]}")
        assert len(hits) <= self.KNOWN_SOURCE_HITS, (
            f"redact_text now fires on {len(hits)} lines of the harness's own source, "
            f"above the {self.KNOWN_SOURCE_HITS} known ones (`token = group_id` in "
            "human_decisions.py, `token_count=token_count` and `token_count=args.tokens` in metrics.py, "
            "`ENV_GIT_CREDENTIAL_HELPER_VAR` in provider_registry.py). This corpus is a "
            "diagnostic rather than a gate, but a count that moved means a pattern got "
            "looser — investigate before shipping:\n  " + "\n  ".join(hits)
        )


# ---------------------------------------------------------------------------
# 3. No drift from the committed scanner (§4.5.5)
# ---------------------------------------------------------------------------


@requires_gitleaks
class TestNoDriftFromTheCommittedScanner:
    """One-way: everything gitleaks catches, ``redact_text`` also redacts.

    The converse is deliberately not asserted and the patterns are never copied
    across — §4.5.1's prefix branches would make the committed scanner red on day
    one against this repo's decoys.
    """

    def test_every_gitleaks_finding_is_absent_from_redact_text(self, tmp_path):
        tree = tmp_path / "tree"
        tree.mkdir()
        planted = {name: _synthesize(name, "redaction-drift-gate") for name in sorted(VECTORS)}
        # No identifier beside the value, exactly as `_plant` does it: the vendor
        # rule must be what fires, and the *pattern* is what has to catch it.
        lines = [f"# {secret}" for secret in planted.values()]
        (tree / "planted.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        findings, _ = _run_gitleaks(tree, CONFIG_PATH, tmp_path / "out")
        assert findings, (
            "gitleaks reported nothing for the synthesized vendor vectors, so this gate "
            "measures nothing. Check the vector table in test_no_committed_secrets."
        )
        covered = set()
        for finding in findings:
            secret = finding["Secret"]
            carrier = lines[finding["StartLine"] - 1]
            assert secret in carrier, "the finding does not line up with the planted line"
            assert secret not in redact_text(carrier), (
                f"gitleaks rule {finding['RuleID']} detects a shape redact_text lets "
                f"through. §4.5.1 has drifted behind the committed scanner — add the "
                f"pattern there (never copy §4.5.1's table into .gitleaks.toml)."
            )
            covered.add(secret)
        assert covered == set(planted.values()), (
            f"gitleaks found {len(covered)} of {len(planted)} synthesized vectors; the "
            "uncovered ones prove nothing about drift"
        )


# ---------------------------------------------------------------------------
# redact_argv is re-pointed through redact_text (§4.5.5)
# ---------------------------------------------------------------------------


class TestRedactArgvRoutesThroughRedactText:
    """The flag-hint pass is kept; the pattern pass closes the inverse hole."""

    def test_flag_hint_pass_is_unchanged(self):
        out = redact_argv(["cli", "--api-key", "whatever-value", "--model", "x"], "")
        assert out == ["cli", "--api-key", "<redacted>", "--model", "x"]
        assert redact_argv(["cli", "--auth-token=abc"], "")[1] == "--auth-token=<redacted>"

    def test_prompt_elision_is_unchanged(self):
        out = redact_argv(["cli", "-p", "THE PROMPT"], "THE PROMPT")
        assert out[2].startswith("<prompt sha256:")

    def test_an_unhinted_argument_carrying_a_secret_is_redacted(self):
        """The inverse hole: a secret in an argument no flag name announces."""
        out = redact_argv(["cli", "--config", f"endpoint=https://x/?key={_GOOGLE_KEY}"], "")
        assert _GOOGLE_KEY not in " ".join(out)

    def test_a_bare_positional_secret_is_redacted(self):
        out = redact_argv(["cli", "exec", f"Authorization: Bearer {_JWT}"], "")
        assert _JWT not in " ".join(out)

    def test_ordinary_arguments_survive(self):
        argv = ["claude", "-p", "--model", "sonnet", "--output-format", "stream-json"]
        assert redact_argv(argv, "") == argv


# ---------------------------------------------------------------------------
# The four sinks (§4.5.3)
# ---------------------------------------------------------------------------


def _all_shape_lines():
    return "\n".join(line for _, line, _ in SHAPES)


def _all_planted():
    return [value for _, _, planted in SHAPES for value in planted]


class TestSinkSuccessLog:
    """``_save_log`` — all nine planted shapes landed here on a *successful* run."""

    def test_no_planted_value_reaches_the_log(self, tmp_path):
        log = tmp_path / "log_gemini_review.txt"
        assert llm_client._save_log(
            log,
            model="gemini/pro",
            prompt=f"review this: {_OPENAI_KEY}",
            stdout=_all_shape_lines(),
            stderr=f"warn: token={_TOKEN_HEX}",
            returncode=0,
            success=True,
            error=f"none (but see {_SLACK_TOKEN})",
        )
        text = log.read_text(encoding="utf-8")
        for value in _all_planted():
            assert value not in text, f"{value[:24]}… reached the success log"


class TestSinkFailedAttemptCapture:
    """``failed/{task_id}.attemptN.json`` — the 64 KiB tail preserved every one."""

    def test_no_planted_value_reaches_the_capture(self, tmp_path):
        path = subagent_executor._write_failure_capture(
            phase_dir=str(tmp_path),
            task={"task_id": "T001", "kind": "validation"},
            attempt_index=1,
            argv=["codex", "exec", "--api-key", _OPENAI_KEY],
            prompt="PROMPT",
            stdout=_all_shape_lines(),
            stderr=f"Authorization: Bearer {_JWT}",
            exit_code=1,
            duration=0.5,
            status="runner_error",
            error=f"the runner printed {_SLACK_TOKEN}",
        )
        assert path
        text = Path(path).read_text(encoding="utf-8")
        for value in _all_planted():
            assert value not in text, f"{value[:24]}… reached failed/*.attemptN.json"


class TestSinkBatchReport:
    """``subagent_batch_report.json`` — six shapes × fifteen rows."""

    def test_no_planted_value_reaches_the_report(self, tmp_path):
        report = subagent_executor.BatchReport(
            phase="plan-review",
            kind="validation",
            phase_dir=str(tmp_path),
            outcomes=[
                subagent_executor.TaskOutcome(
                    task_id="T001",
                    kind="validation",
                    runner="codex:gpt-5",
                    status="runner_error",
                    reason=f"stream tail: {_all_shape_lines()}",
                    attempts=[
                        subagent_executor.Attempt(
                            index=1,
                            status="runner_error",
                            error=f"Authorization: Bearer {_JWT}",
                        )
                    ],
                )
            ],
        )
        text = Path(report.write()).read_text(encoding="utf-8")
        for value in _all_planted():
            assert value not in text, f"{value[:24]}… reached subagent_batch_report.json"


class TestSinkDetachedStdout:
    """The exit-3 stdout block → ``orchestrator-run.log`` under ``--detach``."""

    class _Provider:
        def __init__(self, verdict):
            self._verdict = verdict

        def classify_failure(self, stdout, stderr, exit_code):
            return self._verdict

    def test_auth_failure_stream_tail_is_redacted(self):
        capture = subagent_executor._Capture(
            stdout="", stderr=_all_shape_lines(), exit_code=1
        )
        result = subagent_executor._classify_failed_run(
            self._Provider(subagent_executor.STATUS_AUTH_FAILED), capture
        )
        assert result is not None
        _, reason = result
        for value in _all_planted():
            assert value not in reason, f"{value[:24]}… reached the exit-3 stdout block"

    def test_the_printed_failure_summary_is_redacted_on_every_rung(self):
        """Rungs 4-6 take their reason from an adapter error string, not a tail."""
        report = subagent_executor.BatchReport(
            phase="plan-review",
            kind="validation",
            outcomes=[
                subagent_executor.TaskOutcome(
                    task_id="T001",
                    kind="validation",
                    runner="codex:gpt-5",
                    status="unparseable_result",
                    reason=f"the adapter reported: {_all_shape_lines()}",
                )
            ],
        )
        printed = "\n".join(report.failure_summary())
        for value in _all_planted():
            assert value not in printed, f"{value[:24]}… reached the printed summary"

    def test_rate_limit_stream_tail_is_redacted(self):
        capture = subagent_executor._Capture(
            stdout=_all_shape_lines(), stderr="", exit_code=1
        )
        result = subagent_executor._classify_failed_run(
            self._Provider(subagent_executor.STATUS_RATE_LIMITED), capture
        )
        assert result is not None
        _, reason = result
        for value in _all_planted():
            assert value not in reason


# ---------------------------------------------------------------------------
# L4 — bounded stored output (§4.4)
# ---------------------------------------------------------------------------


class TestSaveLogIsBounded:
    """A single successful ``review-plan`` wrote a 6,254,326 B log. Cap it."""

    def test_a_six_megabyte_stdout_writes_a_small_log(self, tmp_path):
        log = tmp_path / "log_gemini_review.txt"
        stdout = "\n".join(f"event {i}: " + ("x" * 600) for i in range(10_000))
        assert len(stdout) > 6 * 1000 * 1000
        assert llm_client._save_log(
            log, model="gemini/pro", prompt="p", stdout=stdout, stderr=stdout,
            returncode=0, success=True,
        )
        size = log.stat().st_size
        assert size <= 100 * 1024, f"log is {size} B; the ~100 KB total cap did not apply"

    def test_a_single_enormous_line_is_capped_at_64_kib(self, tmp_path):
        log = tmp_path / "log.txt"
        assert llm_client._save_log(
            log, model="m", prompt="p", stdout="J" * (3 * 1024 * 1024), stderr="",
            returncode=0, success=True,
        )
        text = log.read_text(encoding="utf-8")
        longest = max(len(line.encode("utf-8")) for line in text.splitlines())
        assert longest <= 64 * 1024 + 200, (
            f"the longest rendered line is {longest} B; the 64 KiB/line cap did not apply"
        )

    def test_only_the_last_twenty_lines_of_a_stream_are_kept(self, tmp_path):
        log = tmp_path / "log.txt"
        stdout = "\n".join(f"line-{i:04d}" for i in range(500))
        assert llm_client._save_log(
            log, model="m", prompt="p", stdout=stdout, stderr="",
            returncode=0, success=True,
        )
        text = log.read_text(encoding="utf-8")
        assert "line-0499" in text, "the tail of the stream is what a failure needs"
        assert "line-0480" in text, "exactly the last 20 lines, not fewer"
        assert "line-0479" not in text, "exactly the last 20 lines, not more"
        assert "line-0000" not in text, "the last-20-lines rule did not apply"
        assert "line-0100" not in text
        assert "480 earlier line(s) truncated" in text, "the cut must say it happened"

    def test_a_small_log_is_left_alone(self, tmp_path):
        log = tmp_path / "log.txt"
        assert llm_client._save_log(
            log, model="m", prompt="the prompt", stdout="hello\nworld", stderr="oops",
            returncode=1, success=False, error="boom", duration_seconds=1.5,
        )
        text = log.read_text(encoding="utf-8")
        assert "hello\nworld" in text
        assert "oops" in text
        assert "boom" in text


# ---------------------------------------------------------------------------
# Group gate A2 (§14.4) — redaction runs BEFORE truncation
# ---------------------------------------------------------------------------


class TestRedactionRunsBeforeTruncation:
    """A secret straddling the cut is still redacted — the ordering, not the halves.

    Each case is chosen so that truncate-then-redact leaves a remnant **no pattern
    can match**: that is the failure mode the ordering rule exists for, and a case
    whose remnant is still matchable would pass under either order and prove
    nothing.
    """

    def test_a_pem_block_straddling_the_last_twenty_lines_cut(self, tmp_path):
        """The header is what P12 anchors on; last-20 cuts it off, body and all."""
        log = tmp_path / "log.txt"
        stdout = "\n".join(["-----BEGIN RSA PRIVATE KEY-----", *_PEM_BODY,
                            "-----END RSA PRIVATE KEY-----"])
        assert stdout.count("\n") + 1 > 20, "the block must straddle the 20-line cut"
        assert llm_client._save_log(
            log, model="m", prompt="p", stdout=stdout, stderr="",
            returncode=0, success=True,
        )
        text = log.read_text(encoding="utf-8")
        for body_line in _PEM_BODY:
            assert body_line not in text, (
                "private key material survived the last-20-lines cut. Truncation ran "
                "before redaction: the -----BEGIN----- line P12 anchors on was dropped, "
                "so nothing downstream could match the body that remained."
            )

    def test_a_key_value_straddling_the_64_kib_line_cut(self, tmp_path):
        """The key name is what P1 anchors on; a byte tail keeps only the value."""
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        filler = "f" * (64 * 1024 - 12)
        line = f"{filler}aws_secret_access_key={secret}"
        path = subagent_executor._write_failure_capture(
            phase_dir=str(tmp_path),
            task={"task_id": "T001", "kind": "validation"},
            attempt_index=1,
            argv=["codex", "exec"],
            prompt="PROMPT",
            stdout=line,
            stderr="",
            exit_code=1,
            duration=0.5,
            status="runner_error",
            error="failed",
        )
        assert path
        text = Path(path).read_text(encoding="utf-8")
        assert secret not in text, (
            "the value survived the 64 KiB tail. Truncation ran before redaction: the "
            "cut landed inside `aws_secret_access_key=`, leaving `_access_key=<value>` "
            "— a shape SECRET_KEY does not match, so no later pass can recover it."
        )

    def test_a_stream_tail_straddling_the_2_kib_exit_three_cut(self):
        """Same cut, the ``orchestrator-run.log`` sink."""
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        filler = "f" * (2000 - 12)
        capture = subagent_executor._Capture(
            stdout="", stderr=f"{filler}aws_secret_access_key={secret}", exit_code=1
        )
        result = subagent_executor._classify_failed_run(
            TestSinkDetachedStdout._Provider(subagent_executor.STATUS_AUTH_FAILED), capture
        )
        assert result is not None
        assert secret not in result[1]


# ---------------------------------------------------------------------------
# The boundary lives beside redact_argv (§4.5.3)
# ---------------------------------------------------------------------------


def test_redact_text_lives_in_write_safety_beside_redact_argv():
    """One helper, one module — never a fourth pattern language in a new file."""
    assert callable(write_safety.redact_text)
    assert "redact_text" in write_safety.__all__
    assert not (SKILL_DIR / "utils" / "redact.py").exists(), (
        "§4.5.5: redact_text lands beside redact_argv in write_safety.py; splitting "
        "them into separate modules is the shape that subsection argues against"
    )
