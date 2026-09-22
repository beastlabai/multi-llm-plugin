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
import re
import string
import sys
import time
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
    assert len(turns) == 13, f"expected 13 container turns, found {len(turns)}"
    assert len(side) == 7, f"expected 7 side turns, found {len(side)}"
    assert len(responses) == 34, f"expected 34 e2e response fixtures, found {len(responses)}"
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
        """The floor is "bare integer", not "starts with a digit".

        The value is ``_TOKEN_HEX`` for the reason its comment gives: a 32-hex
        literal beside ``token=`` is exactly what ``generic-api-key`` fires
        on, and writing one out here made this file the tracked tree's one
        finding (``test_no_committed_secrets``, 2026-09-17).
        """
        out = redact_text(f"token={_TOKEN_HEX}")
        assert _TOKEN_HEX not in out
        assert "<redacted:secret-kv>" in out


# ---------------------------------------------------------------------------
# 1b. Recall when the key is GLUED to a preceding word character
# ---------------------------------------------------------------------------
#
# Every vendor-prefix pattern in `_REDACTIONS` used to carry a leading `\b`, and
# a key preceded by a word character therefore did not match at all. MEASURED
# 2026-09-22 on a real claude-code transcript: a probe script printed the word
# `SET` with the value glued straight onto it, and the OpenRouter key — the one
# credential this project handles end to end — reached disk intact, because its
# line's key name was the bare word `KEY`, which P1's key half does not match
# either. `redact_text` guards provider logs, the audit record and `redact_argv`,
# so this was a live leak path rather than a test-only curiosity.
#
# The fix is per-prefix rather than a blanket un-anchoring: see the tier comment
# over `_REDACTIONS`. These tests pin both ends of it — every re-anchored prefix
# redacts under every glue it claims, and `TestTheLeftAnchorStillProtectsProse`
# below pins the prose that is the reason bare `sk-` and P2 keep `\b` outright
# and the two `(?<![a-z])` prefixes give up a lowercase-letter glue.

#: The OpenRouter key: this project's own credential, and the shape that leaked.
#: Synthesized like every other vector here — the concrete characters are drawn
#: from the seeded RNG so this file never becomes the one committed secret.
_OPENROUTER_KEY = "sk-or-v1-" + _body(_RNG, 64, _HEX)
_ANTHROPIC_GLUED = _synthesize("anthropic-api-key", "glued")
_GITHUB_GLUED = _synthesize("github-pat", "glued")
_AWS_STS_ID = "ASIA" + _body(_RNG, 16, string.ascii_uppercase + string.digits)
_URL_PASSWORD = _body(_RNG, 16)
_URL_BASIC = f"https://ci:{_URL_PASSWORD}@git.example.com/acme/repo.git"

#: The word characters a rendered line really puts in front of a value when
#: something concatenated without a separator. `SET` is the measured one; the
#: rest are the other three classes of word character.
GLUE = ["SET", "s", "9", "_"]

#: The glue set for the two prefixes guarded by `(?<![a-z])` rather than
#: un-anchored outright. `sk-` ends `task`/`risk`/`disk`/`ask` and `gh?_` is
#: reachable through `laughs_`, so a *lowercase letter* is the one thing that can
#: put either inside an English word — and it is therefore the one glue those two
#: give up. Spelled out here because it is the only place this fix is narrower
#: than an unanchored sweep. See the `_REDACTIONS` tier comment.
GLUE_NON_LOWERCASE = ["SET", "9", "_"]

#: ``(shape, the key, the glue set it must survive)``. Every vendor-prefix row
#: of ``_REDACTIONS`` is represented, some by more than one sub-shape (the
#: ``sk-`` family has four vendor infixes and AWS has two prefixes, and each
#: infix is its own adjudication); ``test_every_redaction_row_is_adjudicated``
#: is what keeps a row from being left out.
GLUED_SHAPES = [
    ("sk-family/anthropic", _ANTHROPIC_GLUED, GLUE_NON_LOWERCASE),
    ("sk-family/openai-project", _OPENAI_KEY, GLUE_NON_LOWERCASE),
    ("sk-family/openrouter", _OPENROUTER_KEY, GLUE_NON_LOWERCASE),
    ("sk-family/stripe-ish-live", _STRIPE_ISH, GLUE_NON_LOWERCASE),
    ("slack", _SLACK_TOKEN, GLUE),
    ("github", _GITHUB_GLUED, GLUE_NON_LOWERCASE),
    ("google-api-key", _GOOGLE_KEY, GLUE),
    ("aws-access-key-id", "AKIAIOSFODNN7EXAMPLE", GLUE),
    ("aws-access-key-id/sts", _AWS_STS_ID, GLUE),
    ("gitlab-pat", _GITLAB_PAT, GLUE),
    ("huggingface", _HF_TOKEN, GLUE),
    ("jwt", _JWT, GLUE),
]

_GLUED_CASES = [
    (f"{shape} after {glue!r}", glue + key, key)
    for shape, key, glue_set in GLUED_SHAPES
    for glue in glue_set
]


class TestAKeyGluedToAPrecedingWordCharacterIsStillRedacted:
    """The left-boundary gap, shape by shape and glue by glue."""

    @pytest.mark.parametrize(
        "case,line,planted", _GLUED_CASES, ids=[c[0] for c in _GLUED_CASES]
    )
    def test_the_glued_key_does_not_survive(self, case, line, planted):
        out = redact_text(line)
        assert planted not in out, (
            f"{case}: a leading `\\b` refused the match, so the key survived.\n"
            f"in : {line[:200]!r}\nout: {out[:200]!r}"
        )

    @pytest.mark.parametrize(
        "case,line,planted", _GLUED_CASES, ids=[c[0] for c in _GLUED_CASES]
    )
    def test_the_glued_key_does_not_survive_inside_a_log_line(self, case, line, planted):
        wrapped = f"2026-09-22T10:00:00 [codex] stderr | env dump: {line} (attempt 2)"
        assert planted not in redact_text(wrapped), f"{case}: survived inside a log line"

    @pytest.mark.parametrize(
        "case,line,planted", _GLUED_CASES, ids=[c[0] for c in _GLUED_CASES]
    )
    def test_glued_redaction_is_idempotent(self, case, line, planted):
        once = redact_text(line)
        assert redact_text(once) == once, f"{case}: redaction is not idempotent"

    def test_no_glue_lets_the_url_basic_auth_password_survive(self):
        """P11's scheme is lowercase, so `\\b` let a lowercase glue through by
        swallowing it as part of the scheme while refusing an uppercase one."""
        for glue in GLUE:
            out = redact_text(glue + _URL_BASIC)
            assert _URL_PASSWORD not in out, f"the password survived after {glue!r}"

    def test_the_measured_transcript_line(self):
        """The exact 2026-09-22 shape, key name and all.

        `KEY` is not a key half P1 matches (`api[_-]?key`/`apikey` need the
        `api`), so before the fix nothing in the table fired on this line and
        the OpenRouter key reached disk verbatim.
        """
        line = f"KEY=SET{_OPENROUTER_KEY}"
        out = redact_text(line)
        assert _OPENROUTER_KEY not in out, "the measured leak reproduced"
        assert "<redacted:sk-family>" in out
        assert out.startswith("KEY=SET"), "the surrounding text must stay diffable"

    def test_every_redaction_row_is_adjudicated(self):
        """A new pattern must land in one of three buckets, on purpose.

        The anchoring decision is the whole substance of this fix, so a row
        added later cannot inherit `\\b` by copy-paste without this going red.
        """
        covered = {shape.split("/")[0] for shape, _, _ in GLUED_SHAPES} | {"url-basic-auth"}
        # Left-anchored deliberately: `bearer`/`basic` end `forbearer` and
        # `ultrabasic`, and un-anchoring P2 would then eat the next eight-character
        # word. See `TestTheLeftAnchorStillProtectsProse`.
        anchored_on_purpose = {"auth-scheme"}
        # Never carried a left boundary a word character could defeat: P12 and
        # P13 anchor on their own literals or on the start of a line, and P1's
        # key half anchors on the START OF THE KEY RUN — `(?<![A-Za-z0-9_.\-])`,
        # which every key name satisfies by definition — so ASCII glue in front
        # of the keyword is inside the run and absorbed: `XXXtoken=…` matched
        # before the 2026-09-22 rewrite and after. That rewrite also dropped
        # P1's `\b`, which is why it now catches a keyword glued to a *non*-ASCII
        # word character too (`ötoken=…`); see TestSecretKvKeyHalfCoverage.
        glue_tolerant = {
            "pem-private-key", "identity-json", "identity-text", "secret-kv",
        }
        names = {name for name, _, _, _ in write_safety._REDACTIONS}
        assert names == covered | anchored_on_purpose | glue_tolerant, (
            "a _REDACTIONS row is not adjudicated for left-anchoring. Add it to "
            "GLUED_SHAPES (re-anchored), to anchored_on_purpose (ambiguous prefix, "
            "with the prose it protects added to PROSE_THE_LEFT_ANCHOR_PROTECTS), "
            "or to glue_tolerant (no left boundary a word character can defeat)."
        )

    def test_p1_absorbs_ascii_glue_without_help(self):
        """P1 was never part of this gap, and the fix did not change it."""
        assert f"token={_TOKEN_HEX}" not in redact_text(f"XXXtoken={_TOKEN_HEX}")
        assert _TOKEN_HEX not in redact_text(f"9token={_TOKEN_HEX}")


# ---------------------------------------------------------------------------
# 1c. P1's key half, rewritten for linearity — and the coverage it kept
# ---------------------------------------------------------------------------
#
# P1 used to read `\b [KEY_CHAR]* KEYWORD [KEY_CHAR]* DELIM (VALUE)`: two
# unbounded greedy runs around the keyword, quadratic on one line of dotted,
# hyphenated or base64url output (see LINEARITY_PAYLOADS for the numbers). The
# rewrite is `(?<!KEY_CHAR) (?=[KEY_CHAR]*? KEYWORD) [KEY_CHAR]* DELIM (VALUE)`
# — the leading run is gone (it never contributed to the replaced group, which
# sits after the keyword), the `\b` went with it (it guarded that run's start,
# and demanding a boundary immediately before the keyword would refuse
# `aws_secret_access_key`), and the keyword test moved into a lookahead so the
# question "does this run hold a keyword?" is asked once per RUN rather than
# once per keyword occurrence. Python lookaheads are atomic, which is what
# makes that hold.
#
# This is a **do-not-weaken** row, so the equivalence is asserted rather than
# argued: the pre-rewrite pattern is spelled out below and both are run over a
# generated corpus, character index by character index. The claim is not "the
# new one is also good" — it is "every character the old one redacted, the new
# one redacts too".

#: P1's key half exactly as it was before 2026-09-22. Spelled out rather than
#: described, so `test_the_rewrite_redacts_everything_the_old_one_did` compares
#: against the real thing. It is deliberately a literal and not a reconstruction
#: from `_KEY_CHAR`/`_SECRET_KEYWORD`: a later edit to those constants must show
#: up here as a difference, not be silently carried into the baseline.
_SECRET_KV_BEFORE_2026_09_22 = re.compile(
    r"(?i)\b[A-Za-z0-9_.\-]*"
    r"(?:secret|passwd|password|token|api[_-]?key|apikey|credential)"
    r"[A-Za-z0-9_.\-]*"
    r"(?:[\"']?\s*[:=]\s*[\"']?)"
    r"([^\s\"',;]{8,})"
)

_SECRET_KV_NOW = {name: pattern for name, pattern, _g, _p in write_safety._REDACTIONS}[
    "secret-kv"
]


def _secret_kv_redacted_indices(pattern, text):
    """The character indices ``pattern`` would replace, predicate included."""
    covered = set()

    def _sub(match):
        if not write_safety._looks_like_credential(match.group(1)):
            return match.group(0)
        start, end = match.span(1)
        covered.update(range(start, end))
        return write_safety._replace_group(match, 1, "secret-kv")

    pattern.sub(_sub, text)
    return covered


#: A credential-shaped value: passes `_looks_like_credential` on every branch
#: that matters, and is not a placeholder.
_KV_VALUE = "Xy9-_.z01234567"

#: ``(case, rendered line)``. The behavioural enumeration the rewrite has to
#: survive: the bare keyword, every joiner a compound key name uses, every glue
#: class in front of the keyword, both delimiters, the quoted-JSON form, and the
#: separator lengths that a *bounded* trailing run — the obvious alternative fix
#: — would have silently stopped redacting.
SECRET_KV_KEY_SHAPES = [
    # Bare keyword, both delimiters, spaced and unspaced.
    ("bare/token-equals", f"token={_KV_VALUE}"),
    ("bare/secret-colon", f"secret: {_KV_VALUE}"),
    ("bare/password-spaced", f"password = {_KV_VALUE}"),
    ("bare/credential", f"credential:{_KV_VALUE}"),
    ("bare/passwd", f"passwd={_KV_VALUE}"),
    ("bare/apikey", f"apikey={_KV_VALUE}"),
    # Compound key names — each joiner the class carries, on both sides.
    ("compound/underscore-both", f"aws_secret_access_key={_KV_VALUE}"),
    ("compound/dash-left", f"x-api-key: {_KV_VALUE}"),
    ("compound/dot-left", f"providers.openrouter.token={_KV_VALUE}"),
    ("compound/dot-right", f"token.value.v2={_KV_VALUE}"),
    ("compound/mixed", f"app-1.auth_token-v2={_KV_VALUE}"),
    ("compound/upper", f"OPENAI_API_KEY={_KV_VALUE}"),
    # Glue: a word character welded straight onto the keyword with no joiner.
    # The old leading run absorbed ASCII glue; these must not regress.
    ("glue/uppercase-word", f"SETtoken={_KV_VALUE}"),
    ("glue/lowercase", f"stoken={_KV_VALUE}"),
    ("glue/digit", f"9token={_KV_VALUE}"),
    ("glue/underscore", f"_token={_KV_VALUE}"),
    ("glue/hyphen", f"-token={_KV_VALUE}"),
    ("glue/dot", f".token={_KV_VALUE}"),
    # Quoted JSON, the shape the optional quotes in `_DELIM` exist for.
    ("quoted/json", f'{{"api_key": "{_KV_VALUE}"}}'),
    ("quoted/single", f"'client_secret' : '{_KV_VALUE}'"),
    # Separator length between the LAST keyword and the delimiter. A bounded
    # trailing run loses these at whatever bound it picked; the lookahead form
    # does not, and that is the reason it was chosen.
    ("length/sep-8", f"secret{'x' * 8}={_KV_VALUE}"),
    ("length/sep-32", f"secret{'x' * 32}={_KV_VALUE}"),
    ("length/sep-64", f"secret{'x' * 64}={_KV_VALUE}"),
    ("length/sep-256", f"secret{'x' * 256}={_KV_VALUE}"),
    ("length/sep-4096", f"secret{'x' * 4096}={_KV_VALUE}"),
    ("length/prefix-4096", f"{'x' * 4096}_token={_KV_VALUE}"),
    # The keyword at the end of, and repeated inside, a long dotted run — the
    # two shapes that were 16 s and 99 s before the rewrite.
    ("run/dotted-tail", f"{'a.b.c.' * 200}token={_KV_VALUE}"),
    ("run/keyword-dense", f"{'token.' * 200}token={_KV_VALUE}"),
    # Embedded in a rendered log line rather than alone on one.
    ("embedded/log-line", f"2026-09-22T10:00:00 [codex] stderr | token={_KV_VALUE} (attempt 2)"),
]

#: The broadening the rewrite bought, stated rather than left to be discovered.
#: Dropping `\b` means a keyword glued to a word character *outside*
#: ``[A-Za-z0-9_]`` is now matched; before, no `\b` existed anywhere in front of
#: it and the row could not fire at all. This is the same gap section 1b closed
#: for the vendor prefixes, and it is the ONLY behavioural difference the
#: differential found over ~10^6 strings.
SECRET_KV_NON_ASCII_GLUE = [
    ("non-ascii/latin-1", f"ötoken={_KV_VALUE}"),
    ("non-ascii/cjk", f"設定token={_KV_VALUE}"),
    ("non-ascii/accented-compound", f"clé_api_key={_KV_VALUE}"),
]


class TestSecretKvKeyHalfCoverage:
    """P1's key half: everything it caught before, it catches now."""

    @pytest.mark.parametrize(
        "case,line", SECRET_KV_KEY_SHAPES, ids=[c for c, _ in SECRET_KV_KEY_SHAPES]
    )
    def test_the_value_is_redacted(self, case, line):
        out = redact_text(line)
        assert _KV_VALUE not in out, (
            f"{case}: the key half stopped matching this shape.\n"
            f"  in : {line[:160]!r}…\n  out: {out[:160]!r}…"
        )
        assert "<redacted:secret-kv>" in out, f"{case}: redacted by the wrong row"

    @pytest.mark.parametrize(
        "case,line", SECRET_KV_KEY_SHAPES, ids=[c for c, _ in SECRET_KV_KEY_SHAPES]
    )
    def test_the_old_pattern_agreed(self, case, line):
        """Each named shape is one the pre-rewrite row also caught.

        Without this the table above could drift into asserting only what the
        new pattern happens to do — the point is that it is a *subset* claim.
        """
        assert _secret_kv_redacted_indices(_SECRET_KV_BEFORE_2026_09_22, line), (
            f"{case}: this shape is not part of the coverage claim — the "
            f"pre-rewrite pattern did not match it either"
        )

    @pytest.mark.parametrize(
        "case,line", SECRET_KV_NON_ASCII_GLUE, ids=[c for c, _ in SECRET_KV_NON_ASCII_GLUE]
    )
    def test_non_ascii_glue_is_the_one_broadening(self, case, line):
        """Broader, on purpose, and in the direction 1b already argued for."""
        assert _KV_VALUE not in redact_text(line), f"{case}: the value survived"
        assert not _secret_kv_redacted_indices(_SECRET_KV_BEFORE_2026_09_22, line), (
            f"{case}: this is supposed to be a case the OLD pattern missed; if it "
            f"matched, the broadening claim is wrong and this row belongs above"
        )

    def test_idempotent_on_every_shape(self):
        for case, line in [*SECRET_KV_KEY_SHAPES, *SECRET_KV_NON_ASCII_GLUE]:
            once = redact_text(line)
            assert redact_text(once) == once, f"{case}: redaction is not idempotent"


def _differential_corpus():
    """A deterministic corpus that exercises the key half from every side."""
    glues = ["", "SET", "s", "9", "_", "-", ".", "ö", "aws_", '"', " ", "("]
    keywords = ["secret", "passwd", "password", "token", "api_key", "api-key",
                "apikey", "credential", "SECRET", "Token", "API_KEY"]
    separators = ["", "_access_key", "-id", ".v2", "X" * 40, "Y" * 200, "_" * 3]
    delimiters = ["=", ":", " = ", " : ", '":"', "':'", "\t=\t"]
    values = ["abcdefgh", "Xy9-_.z01", "12345678", "aGVsbG8gd29ybGQK", "short",
              "password", "os.environ.get", "<redacted:jwt>"]
    for glue in glues:
        for keyword in keywords:
            for separator in separators:
                for delimiter in delimiters:
                    for value in values:
                        line = glue + keyword + separator + delimiter + value
                        yield line
                        yield f"log | {line} | trailing words here"
    rng = random.Random("secret-kv differential")
    alphabet = list("abcSECRETtoken_-.=: \"'\t,;/(){}[]$\\|&*!`<>0123456789ö漢+@%")
    for _ in range(40000):
        yield "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))


class TestTheRewriteRedactsEverythingTheOldOneDid:
    """The do-not-weaken claim, checked rather than asserted in prose."""

    def test_no_character_the_old_pattern_redacted_survives(self):
        lost = []
        ascii_only_differences = []
        for line in _differential_corpus():
            before = _secret_kv_redacted_indices(_SECRET_KV_BEFORE_2026_09_22, line)
            after = _secret_kv_redacted_indices(_SECRET_KV_NOW, line)
            if before - after:
                lost.append((line, sorted(before - after)[:8]))
                if len(lost) > 5:
                    break
            elif before != after and line.isascii():
                ascii_only_differences.append((line, sorted(after - before)[:8]))
                if len(ascii_only_differences) > 5:
                    break
        assert not lost, (
            "the rewrite of P1's key half LOST coverage — these characters were "
            "redacted before and are not now:\n  "
            + "\n  ".join(f"{line!r} at {idx}" for line, idx in lost)
        )
        assert not ascii_only_differences, (
            "the rewrite changed behaviour on an ASCII-only input. The only "
            "behavioural difference it is allowed is the non-ASCII glue case "
            "(see SECRET_KV_NON_ASCII_GLUE):\n  "
            + "\n  ".join(f"{line!r} gained {idx}" for line, idx in ascii_only_differences)
        )


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
    #:
    #: A **fifth and sixth** arrived with the omp adapter, both the same class
    #: again: ``OMP_CREDENTIAL_ENV = "OPENROUTER_API_KEY"`` names the variable
    #: omp's own ``$env`` chain reads — a *name*, never a value, and the reason
    #: the constant exists at all is that this is the one adapter whose
    #: credential is not foreign to it — and
    #: ``credential_helper=credential_helper`` is a keyword forwarded to
    #: ``super().resolve_env``. Neither is a pattern that got looser, and
    #: neither line has ever been a redaction input: this tree is source.
    #:
    #: An **eighth** arrived 2026-09-22 when the nested-exec probe was taught
    #: to read its own reply: ``NESTED_PROBE_TOKEN = "MULTI_LLM_NESTED_EXEC_OK"``
    #: names the sentinel the probe asks a runner to echo back. It is printed
    #: verbatim inside the probe prompt, so it is the opposite of a secret --
    #: the constant exists only so the question and the answer cannot drift
    #: apart. A constant assignment in source again, same class as the rest.
    KNOWN_SOURCE_HITS = 8


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
            "`ENV_GIT_CREDENTIAL_HELPER_VAR` in provider_registry.py, `nested_exec_credentials` in "
            "subagent_executor.py, `OMP_CREDENTIAL_ENV` + `credential_helper=credential_helper` "
            "in providers/omp.py, and `NESTED_PROBE_TOKEN` in subagent_executor.py). This corpus is a "
            "diagnostic rather than a gate, but a count that moved means a pattern got "
            "looser — investigate before shipping:\n  " + "\n  ".join(hits)
        )


#: The prose the left anchor exists for — the other half of 1b, and the reason
#: the fix is per-prefix rather than a blanket un-anchoring. Hand-written rather
#: than harvested, because the benign corpora above are this repo's own fixtures
#: and a pattern can be loose in a way they happen not to exercise. The `ask-…`
#: rows ARE from this tree — `git grep -oE '[A-Za-z0-9_]sk-[A-Za-z0-9_-]{8,}'`
#: over the corpora returns those compounds and nothing else, which is the
#: measured reason bare `sk-` can never be unanchored here; the rest are the
#: constructions each anchoring decision was weighed against.
PROSE_THE_LEFT_ANCHOR_PROTECTS = [
    # Bare `sk-` + 16 body characters, which is why that branch keeps `\b`.
    ("sk-bare/task", "task-oriented-development-workflow"),
    ("sk-bare/risk", "risk-projection-model"),
    ("sk-bare/repo-own", "ask-suggestion-advisory rows join ask-suggestions-batch"),
    ("sk-bare/repo-own-2", "ask-authoring-rules-and-decomposition"),
    ("sk-bare/desk", "desk-projection-alignment-matrix"),
    # A vendor infix reached from an English word ending in `sk` — always with a
    # lowercase letter in front, which is what `(?<![a-z])` refuses.
    ("sk-infix/live", "disk-live-migration-controller"),
    ("sk-infix/live-2", "task-live-reload-middleware-v2"),
    ("sk-infix/ant", "risk-ant-colony-optimization-solver"),
    ("sk-infix/or-v1", "task-or-v1-branch-protection-rules"),
    # `gh[opusr]_` through the `-s` plural, plus a 16-character run.
    ("github/plural", "he laughs_loudlyandforever about the release"),
    # P2's two keywords as ordinary words, each followed by an 8-character one.
    ("auth-scheme/basic", "the algorithm is ultrabasic compared to gradient-descent"),
    ("auth-scheme/bearer", "a forbearer inherited the estate and never changed it"),
    # `ASIA` as a word, and an uppercase run that is not 20 characters of key.
    ("aws/word", "see ASIA PACIFIC REGION NOTES for details"),
    ("aws/uppercase-run", "EURASIA and the PACIFICREGIONXYZ map"),
    # `AIza` needs `AI` upper then `za` lower; a camelCase identifier must not
    # be mistaken for one.
    ("google/camel", "openAIzeroShotClassifierConfigurationBuilderFactory"),
]


class TestTheLeftAnchorStillProtectsProse:
    """Section 1b's inverse: un-anchoring is per-prefix, and here is the cost it
    refuses to pay.

    The naive fix for the glued-key gap is to drop every ``\\b``, and it is
    wrong: an unanchored bare ``sk-`` plus sixteen body characters eats
    ``task-oriented-development-workflow``, and an unanchored vendor infix eats
    ``risk-ant-colony-optimization-solver``. A blanket sweep is defensible in a
    throwaway rig digest and is not defensible in the filter that renders every
    provider log, so each prefix is adjudicated on whether English can reach it.
    """

    @pytest.mark.parametrize(
        "case,line",
        PROSE_THE_LEFT_ANCHOR_PROTECTS,
        ids=[c for c, _ in PROSE_THE_LEFT_ANCHOR_PROTECTS],
    )
    def test_prose_is_returned_byte_for_byte(self, case, line):
        out = redact_text(line)
        assert out == line, (
            f"{case}: the left anchor was widened past what this prefix can "
            f"afford.\n  in : {line!r}\n  out: {out!r}"
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

    def test_a_glued_key_in_an_unhinted_argument_inherits_the_fix(self):
        """`redact_argv` needs no change of its own: every argument that is not
        the prompt and carries no hinted flag goes through `redact_text`, so the
        left-boundary fix reaches argv the moment it lands in the table.

        The flag-hint pass above cannot cover this — it keys on the *flag*, and
        `--config` is not a secret-shaped name.
        """
        out = redact_argv(["cli", "--config", f"KEY=SET{_OPENROUTER_KEY}"], "")
        assert _OPENROUTER_KEY not in " ".join(out)
        assert "<redacted:sk-family>" in out[2]

    def test_a_glued_key_as_a_bare_positional_inherits_the_fix(self):
        out = redact_argv(["cli", "exec", f"probe printed SET{_GOOGLE_KEY}"], "")
        assert _GOOGLE_KEY not in " ".join(out)

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
# The other half of that ordering: redaction must be LINEAR
# ---------------------------------------------------------------------------
#
# Running before truncation is what makes the group gate above work, and it is
# also what makes an unbounded quantifier in `_REDACTIONS` a hang rather than a
# slow function: `llm_client._save_log` calls `_redact(stdout)` on the **whole**
# stream and only then `_bound_stream`s it, so every pattern here is handed
# however many megabytes a provider chose to emit.
#
# MEASURED 2026-09-22, and this is a regression that really shipped for the
# length of one review round. Un-anchoring P11 (section 1b) took it from
# `\b[a-z][a-z0-9+.\-]*://…` to `[a-z][a-z0-9+.\-]*://…`. P11 is the one row
# whose pattern opens with a character class instead of a literal, so CPython
# cannot skip start positions the way it does for `sk-`, `xox`, `gh`, `AIza`,
# `AKIA`, `glpat-`, `hf_` and `eyJ`; without the `\b` the engine tried every
# character in turn, each time running `[a-z0-9+.\-]*` to the end of the buffer
# and backtracking for `://`. Quadratic: 0.16 s at 20 KB, 2.5 s at 80 KB, 40 s
# at 320 KB, and roughly 1600 s on the 2 MB payload
# `tests/test_mock_plugin_hook.py` plants — which is how a redaction change
# came to fail a mock-plugin test with a 120 s timeout and no mention of a
# credential anywhere in it. The fix was to bound the scheme run; this is the
# tripwire that keeps it bounded.
#
# A ceiling, not a benchmark. Every row is under 100 ms per megabyte today, the
# bound is 5 s, and the behaviour it exists to catch is three orders of
# magnitude the wrong side of that — so it cannot flake on a loaded box and
# cannot pass on a quadratic pattern.

#: One megabyte, which is well inside what a provider really emits.
LINEARITY_CHARS = 1024 * 1024

#: Per-row wall-clock ceiling for one pass over ``LINEARITY_CHARS``.
LINEARITY_CEILING_S = 5.0

#: ``(label, repeating unit)``. The shapes are chosen for what they do to a
#: greedy run: one character repeated forever (the incident's own payload),
#: words joined by the hyphens ``_KEY_CHAR`` and P3's body both accept, a
#: mixed-case alphanumeric blob, and the URL-scheme charset.
#:
#: **There is no exception list any more, and the four rows below the first
#: four are why it went.** ``secret-kv`` used to be carved out here, because its
#: ``\b [A-Za-z0-9_.\-]* KEYWORD [A-Za-z0-9_.\-]*`` key half was quadratic at
#: rest — not a consequence of the 1b anchoring work, a shape this repository
#: had been shipping for as long as the table existed. The trigger is a long
#: run with **no whitespace, quote or brace in it** (each of those breaks the
#: run and keeps the row linear) that is densely punctuated with ``.`` or
#: ``-``: those two are the only members of the key-character class that are
#: not *word* characters, so they manufacture a ``\b`` start position every few
#: characters without ever ending the run. MEASURED 2026-09-22 at HEAD, one
#: ``sub`` pass: hyphenated slugs 0.50 s / 1.99 s / 7.96 s at 8/16/32 KB,
#: dotted identifiers 1.03 s / 4.17 s / 16.5 s, base64url 0.12 s / 0.50 s /
#: 1.97 s, and a keyword-dense dotted run (``token.`` repeated) **98.9 s at
#: 8 KB**. Fourfold per doubling is n^2, and a megabyte extrapolates to hours
#: inside a filter that runs on the whole provider stream before any cap.
#:
#: Every one of those shapes is a row here now, un-excepted, and the whole
#: table runs in about 0.04 s per megabyte — see the rewritten ``_SECRET_KEY``
#: for the rule and for why it costs no coverage. A shape nobody measures is
#: where the next one would hide, so the four new rows stay whether or not they
#: are ever slow again.
LINEARITY_PAYLOADS = [
    ("single-character-run", "y"),
    ("hyphenated-words", "abc-def-"),
    ("mixed-alphanumeric", "aB3xY9zQ"),
    ("url-scheme-charset", "a+b.c-d9"),
    # The four shapes that made `secret-kv` quadratic, each a real line of
    # provider output: a base64url blob, a dotted module-path list, a run of
    # hyphenated slugs, and a dotted run that repeats a keyword.
    ("base64url-blob", "aGVsbG8-V29ybGQ_QmFzZTY0-YmxvYl8"),
    ("dotted-identifiers", "a.b.c.d."),
    ("hyphenated-slugs", "alpha-beta-gamma-"),
    ("keyword-dense-dotted-run", "token."),
]

_PAYLOAD_CACHE: dict = {}


def _linearity_payload(unit: str) -> str:
    """Built once and shared: four megabytes of parametrized fixture otherwise."""
    if unit not in _PAYLOAD_CACHE:
        _PAYLOAD_CACHE[unit] = (unit * (LINEARITY_CHARS // len(unit) + 1))[:LINEARITY_CHARS]
    return _PAYLOAD_CACHE[unit]


_LINEARITY_CASES = [
    (f"{name} on {label}", name, pattern, unit)
    for label, unit in LINEARITY_PAYLOADS
    for name, pattern, _group, _predicate in write_safety._REDACTIONS
]


class TestRedactTextIsLinearOnAMegabyteStream:
    """No row may do work proportional to the square of its input.

    Every row against every payload, with **no carve-out**: a row that cannot
    survive a shape is a row to fix, not a row to list here.
    """

    def test_no_row_is_excepted(self):
        """The guard covers the whole table, and that is the point of it.

        ``secret-kv`` was the single named exception until 2026-09-22. An
        exception list is how a quadratic row stays shipped, so the assertion
        that there is none is itself a test.
        """
        rows = {name for name, _, _, _ in write_safety._REDACTIONS}
        covered = {name for _case, name, _pattern, _unit in _LINEARITY_CASES}
        assert covered == rows, (
            f"rows missing from the linearity guard: {sorted(rows - covered)}. "
            "Every _REDACTIONS row runs against every LINEARITY_PAYLOADS shape; "
            "fix the row rather than excepting it."
        )

    @pytest.mark.parametrize(
        "case,name,pattern,unit", _LINEARITY_CASES, ids=[c[0] for c in _LINEARITY_CASES]
    )
    def test_one_row_over_one_megabyte(self, case, name, pattern, unit):
        payload = _linearity_payload(unit)
        started = time.perf_counter()
        pattern.sub(lambda match: match.group(0), payload)
        elapsed = time.perf_counter() - started
        assert elapsed < LINEARITY_CEILING_S, (
            f"{case}: one pass over {LINEARITY_CHARS} characters took {elapsed:.1f}s, "
            f"over the {LINEARITY_CEILING_S}s ceiling. A pattern that opens with a "
            f"character class and then runs an UNBOUNDED quantifier before a required "
            f"literal is quadratic once its left anchor is removed — bound the "
            f"quantifier (P11 is the worked example) rather than raising this ceiling. "
            f"redact_text runs on the whole stream before any cap, so this is a hang "
            f"in every log sink, not a slow test."
        )

    def test_the_mock_plugin_payload_that_caught_this(self):
        """The exact 2 MB shape ``test_mock_plugin_hook`` plants.

        That test installs a scratch adapter emitting ``PAYLOAD_NEEDLE`` plus
        two megabytes of one character, runs a real ``review-plan``, and gives
        the whole thing 120 s. With P11 unbounded, ``_save_log``'s pre-truncation
        ``_redact`` never returned and the run died on that timeout with its
        last line still ``Starting review with file-based output`` — a failure
        that names no credential and looks nothing like a redaction bug.
        """
        payload = "M22-SCRATCH-PLUGIN-PAYLOAD" + "y" * (2 * 1024 * 1024)
        started = time.perf_counter()
        out = redact_text(payload)
        elapsed = time.perf_counter() - started
        assert elapsed < LINEARITY_CEILING_S * 2, (
            f"redact_text took {elapsed:.1f}s on the 2 MB mock-plugin payload; "
            f"test_mock_plugin_hook gives the whole review-plan run 120s"
        )
        assert out == payload, "two megabytes of one character is not a credential"


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
