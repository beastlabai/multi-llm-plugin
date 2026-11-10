"""The generic §6.2 fixture harness, and the roster guard that closes gap 4.

**Why this module exists.** Seven providers ship parser fixtures under
``tests/fixtures/<provider>/``; each set is consumed by a hand-written class in
a ``test_provider_*.py`` file (copilot's is ~450 lines). Nothing discovered fixture
directories and nothing asserted that a provider *had* one — which is exactly
why nine missing sets went unnoticed for as long as they did. Replicating the
hand-written pattern nine more times would leave the tenth omission just as
silent, so this module inverts it: a per-provider ``manifest.yaml`` declares the
rows, this file parametrizes over provider × row, and
:data:`FIXTURES_PENDING` / :data:`MANIFEST_PENDING` are the only two places a
provider may sit without one — by name, with a reason.

**The manifest format** (``tests/fixtures/<provider>/manifest.yaml``)::

    schema_version: 1
    provider: aider            # must be a key of provider_registry._PROVIDERS
    cli_version: "0.86.2"      # the CLI these bytes were captured from
    captured: "2026-08-19"     # ISO date
    method: >-                 # how they were produced, in one paragraph
      §6 step 5 ...
    rows:
      success:
        stdout: success.stdout.txt   # names relative to this directory;
        stderr: success.stderr.txt   # an omitted stream means "empty"
        exit_code: 0
        wall_s: 3.07                 # optional recorded measurements
        requests: 1
        notes: "..."                 # optional free text
        expect:
          success: true              # required: parse_output()["success"]
          data: [...]                # optional: the EXACT unwrapped document
          error_contains: ["..."]    # optional: substrings of ["error"]
          raw_contains: ["..."]      # optional: substrings of the captured bytes
          classify_failure: null     # optional, and `null` is an assertion
          known_wrong:               # optional: the measured verdict is wrong
            intended: rate_limited   #   what classify_failure should give
            reason: "..."            #   why it does not yet
          salvage: answer            # optional: none | partial | answer
          lane_status: {lane_a: ok, lane_b: ok}   # required: see below
      empty:
        not_applicable: "<the measurement that says why>"

``expect.salvage`` labels whether the row's stdout holds answer text a salvage
could recover, judged from the bytes and **never** from ``success`` — a failed
run can still hold an answer worth salvaging, and a timed-out one can hold
none. The three values, graded by each adapter's ``salvage_text`` sweep
(``test_pi_json_mode.py``, ``test_omp_stream_reading.py``,
``test_salvage_text.py``):

- ``none``: transport only (an auth error, a rate limit, an empty stream, a
  reviewer killed while a tool ran). The reduced text must be *rejected* by
  ``has_salvageable_json``.
- ``partial``: a real partial answer on a failed run (an output-limit cut, a
  timeout mid-array). The reduced text must *pass* that gate; these rows are
  what salvage exists for.
- ``answer``: a whole answer. ``extract_json_from_text`` recovers it, and where
  the row pins ``data`` it recovers exactly that.

Only these three values are accepted, so a typo cannot quietly assert nothing
(plan: review-run-scope-salvage-and-failure-visibility §4, T4.1).

``expect.lane_status`` pins the retry-ladder status each lane gives the row,
read without a child process: ``lane_a`` is the review fan-out's
(``utils.provider_outcome.outcome_from_capture``, then what
``invoke_with_file_output`` makes of it when the model wrote no result file,
then ``review_orchestrator_base.attempt_ladder_status``; ``null`` is an error
line nothing recognises), ``lane_b`` the executor's (``_classify_pre_parse``,
the extraction, then ``_classify_failed_run`` for a reclassifiable status; the
task kind's own shape check is left out, since it judges the task, not the
provider). A row whose two statuses lead the ladder to different retry-or-stop
answers carries ``open: <unit>`` naming the unit that will settle it, or
``by_design: <reason>`` for the one permanent difference: an exit 0 of an
adapter that is not ``EXIT0_CLASSIFIABLE``, which the fan-out stops on
(``null``) and the executor retries once (``unparseable_result``). The grader
fails when a marker is missing, when one is left on a row whose lanes agree,
and when ``by_design`` sits on any other shape. ``permission_denied`` from the
fan-out is the deliberate exception
(docs/development/delegation.md#read-and-analyze-executor).
Every graded row carries it, so a parser or classifier change shows here first.

``expect.known_wrong`` marks a ``classify_failure`` pin that records a known
bug rather than the right verdict. ``classify_failure`` stays the measured
value, so the row keeps grading what the adapter does today; ``intended`` names
the verdict it should give, one of :data:`CLASSIFY_VERDICTS`, and ``reason``
says why it does not. When the adapter starts giving ``intended``, the
classification check fails and asks for the marker to go and the pin to move,
so a classifier fix flips each such row deliberately. The marker never skips a
row: its parse, salvage and lane checks run as on any other.

A row carries **either** files+``exit_code``+``expect`` **or**
``not_applicable`` with a measured reason — §6.3's rule, enforced here: *"not
applicable, because …" is an acceptable cell; an untested one is not.* Every
canonical row of §6.2 must appear in every manifest (:data:`CANONICAL_ROWS`);
extra rows beyond the canon are graded the same way. Bytes must be raw CLI
output; a row that is not carries ``synthesized: true`` **and** a
``provenance:`` string saying what was changed and why (copilot's
``multi-assistant.jsonl`` is the precedent).

Unknown keys are rejected at every level. A silently-ignored ``expects:`` typo
would assert nothing at all, which is the failure mode this whole module exists
to remove.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers import failure_classifier as _fc  # noqa: E402
from tests.roster_check import RECIPE  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST_NAME = "manifest.yaml"

#: §6.2's table, verbatim. Every manifest must have a cell for each — filled or
#: measured-not-applicable.
CANONICAL_ROWS = (
    "success",
    "truncated",
    "ephemeral-only",
    "multi-assistant",
    "error-envelope",
    "exit-zero-failure",
    "stdout-error",
    "empty",
    "auth-error",
    "rate-limit",
    "invalid-model",
    "context-overflow",
)

# --------------------------------------------------------------------------- #
# The roster guard's two exemption tables. THIS IS THE GAP-4 FIX: adding a
# provider to `provider_registry._PROVIDERS` without a manifest fails
# `test_every_registry_provider_is_accounted_for` BY NAME. Neither table may be
# widened silently — every entry is asserted to still be true below.
# --------------------------------------------------------------------------- #

#: Group A — fixtures ARE committed, but are still consumed by a hand-written
#: class in a ``test_provider_*.py`` file rather than by a manifest. The value is that
#: class's name and both it and the fixture directory are asserted to exist, so
#: this table cannot be used to park a provider that has no fixtures at all.
#: Migrating one of these to a manifest is a strict improvement and deletes its
#: row here.
#:
#: ``copilot`` and ``vibe`` left this table on 2026-08-20, and what forced them
#: out is worth recording: a hand-written consumer graded their PARSING and
#: nothing graded their classification, so both shipped a *successful* run that
#: `classify_failure` read as ``rate_limited`` — from our own echoed prompt —
#: with no test in the repo able to see it. Their manifests now pin that cell.
#:
#: **This table is now empty.** codex, kilocode, kimi, opencode and qwen-code
#: left it on 2026-11-10 with manifests over captures from
#: tests/fixtures/row_capture/, and their first grading found the same class of
#: gap: kilocode's 401 and 429 classify None (see its manifest's notes). It is
#: kept, like ``FIXTURES_PENDING``, as the named place a provider with
#: fixtures but no manifest has to be listed.
MANIFEST_PENDING: dict[str, str] = {}

#: Group B — no fixture set exists at all (plans/harness-full-coverage.md
#: Wave 5). The value is the measured reason, and the directory is asserted
#: ABSENT: landing a capture forces the entry out of this table rather than
#: leaving it to rot as a false exemption.
#: Providers with NO fixture directory at all, by name and with the measured
#: reason. **This table is now empty, and that is the point of §14.7 gap 4.**
#:
#: It held nine entries on 2026-08-19 — agy, claude-code, cline, cursor-agent,
#: gemini, goose, grok, muse, pi. All nine were captured that day, and the test
#: below is what forced each entry out: it asserts the directory does NOT exist,
#: so landing a capture makes the exemption fail rather than letting it linger.
#:
#: Keeping the table (rather than deleting it with its last row) is deliberate.
#: The next provider added to the registry has no fixtures on its first day, and
#: `test_every_registry_provider_is_accounted_for` will demand it be named here
#: with a reason — which is the whole mechanism that was missing when nine sets
#: went unnoticed.
FIXTURES_PENDING: dict[str, str] = {}

_ALLOWED_TOP = {"schema_version", "provider", "cli_version", "captured", "method",
                "notes", "rows"}
_ALLOWED_ROW = {"stdout", "stderr", "exit_code", "expect", "not_applicable",
                "synthesized", "provenance", "notes", "wall_s", "requests"}
_ALLOWED_EXPECT = {"success", "data", "error_contains", "raw_contains",
                   "classify_failure", "known_wrong", "salvage", "lane_status"}
#: Every verdict ``classify_failure`` gives besides ``None``: the ladder
#: statuses and ``CATEGORY_*`` names ``failure_classifier.classify_streams``
#: returns. ``expect.known_wrong.intended`` must be one of them.
CLASSIFY_VERDICTS = frozenset({
    _fc.STATUS_AUTH_FAILED, _fc.STATUS_CONTEXT_LENGTH_EXCEEDED,
    _fc.STATUS_RATE_LIMITED, _fc.STATUS_RUNNER_ERROR, _fc.STATUS_SPAWN_FAILED,
    _fc.STATUS_TIMEOUT, _fc.CATEGORY_CONNECTION_ERROR, _fc.CATEGORY_SERVER_ERROR,
    _fc.CATEGORY_INVALID_MODEL, _fc.CATEGORY_PERMISSION_DENIED,
    _fc.CATEGORY_SANDBOX_UNAVAILABLE,
})
#: ``expect.known_wrong``'s keys, both required: the verdict the row should
#: get, and why it does not yet.
KNOWN_WRONG_KEYS = frozenset({"intended", "reason"})
#: ``expect.salvage``'s whole vocabulary (see the module docstring).
SALVAGE_LABELS = ("none", "partial", "answer")
#: ``expect.lane_status``'s keys: the two statuses, and the markers of a row
#: whose lanes answer retry-or-stop differently: ``open`` until a unit settles
#: it, ``by_design`` where the difference is permanent.
LANE_STATUS_KEYS = ("lane_a", "lane_b")
LANE_STATUS_OPEN = "open"
LANE_STATUS_BY_DESIGN = "by_design"
LANE_STATUS_MARKERS = (LANE_STATUS_OPEN, LANE_STATUS_BY_DESIGN)
#: The one lane pair ``by_design`` may mark, on an exit 0 of an adapter that is
#: not ``EXIT0_CLASSIFIABLE``: no verdict, so the fan-out stops and the
#: executor keeps its parse failure's one retry.
BY_DESIGN_LANES = {"lane_a": None, "lane_b": "unparseable_result"}
#: The fan-out stops on these where the executor retries once, on purpose
#: (docs/development/delegation.md#read-and-analyze-executor).
LANE_A_STOPS_BY_DESIGN = frozenset({"permission_denied"})


def _load(provider_dir: Path) -> dict:
    with open(provider_dir / MANIFEST_NAME, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _discover() -> dict:
    return {
        path.parent.name: _load(path.parent)
        for path in sorted(FIXTURES.glob(f"*/{MANIFEST_NAME}"))
    }


MANIFESTS = _discover()

#: (provider, row-name) for every declared row of every manifest — the whole
#: parametrization surface. Built at import time so a collection error is a
#: manifest error, reported once, rather than a mystery skip.
ROW_IDS = [
    (provider, row)
    for provider, manifest in MANIFESTS.items()
    for row in (manifest.get("rows") or {})
]


def _row(provider: str, row: str) -> dict:
    return MANIFESTS[provider]["rows"][row]


def known_wrong_problems(expect: dict) -> list:
    """What is wrong with ``expect.known_wrong``: empty when it is absent or
    well formed (see the module docstring)."""
    if "known_wrong" not in expect:
        return []
    marker = expect["known_wrong"]
    if not isinstance(marker, dict):
        return ["expect.known_wrong is a mapping: {intended: <verdict>, reason: <why>}"]
    problems = []
    if set(marker) != KNOWN_WRONG_KEYS:
        problems.append(
            f"expect.known_wrong takes exactly {sorted(KNOWN_WRONG_KEYS)}, "
            f"not {sorted(marker)}"
        )
    if "classify_failure" not in expect:
        problems.append(
            "expect.known_wrong sits beside the measured expect.classify_failure, "
            "which stays pinned"
        )
    intended = marker.get("intended")
    if not isinstance(intended, str) or intended not in CLASSIFY_VERDICTS:
        problems.append(
            f"expect.known_wrong.intended is {intended!r}; it must be one of "
            f"{sorted(CLASSIFY_VERDICTS)}"
        )
    elif intended == expect.get("classify_failure"):
        problems.append(
            "expect.known_wrong.intended is the measured classify_failure, so "
            "nothing is wrong: delete the marker"
        )
    reason = marker.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("expect.known_wrong.reason says why the measured verdict is wrong")
    return problems


def _stream(provider: str, row: dict, key: str) -> str:
    """One captured stream, decoded. An omitted key means the CLI wrote nothing."""
    name = row.get(key)
    if name is None:
        return ""
    return (FIXTURES / provider / name).read_bytes().decode("utf-8")


# --------------------------------------------------------------------------- #
# The roster guard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("provider", sorted(_PROVIDERS))
def test_every_registry_provider_is_accounted_for(provider):
    """A manifest, or a named exemption. Never silence.

    This is §14.7 gap 4's fix. It is an assertion, not a warning: a warning is
    what let nine providers ship with no parser fixtures at all.
    """
    if provider in MANIFESTS:
        return
    reason = MANIFEST_PENDING.get(provider) or FIXTURES_PENDING.get(provider)
    assert reason, (
        f"provider {provider!r} has neither tests/fixtures/{provider}/manifest.yaml "
        f"nor an entry in MANIFEST_PENDING / FIXTURES_PENDING in "
        f"tests/test_provider_fixture_rows.py. Capture its §6.2 rows (offline "
        f"capture: {RECIPE}), or add it to one table WITH the measured reason it "
        f"cannot be captured yet."
    )


@pytest.mark.parametrize("provider", sorted(MANIFEST_PENDING))
def test_group_a_exemptions_really_have_committed_fixtures(provider):
    """Group A means "fixtures exist, manifest pending" — both halves checked."""
    assert provider in _PROVIDERS, f"{provider} is not a registry key"
    assert provider not in MANIFESTS, (
        f"{provider} now has a manifest — delete its MANIFEST_PENDING row"
    )
    directory = FIXTURES / provider
    assert directory.is_dir() and any(directory.iterdir()), (
        f"{provider} claims committed fixtures but {directory} is empty/absent"
    )
    consumer = MANIFEST_PENDING[provider]
    source = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path(__file__).parent.glob("test_provider_*.py"))
    )
    assert f"class {consumer}" in source, (
        f"{provider}'s declared consumer {consumer} is gone from the test_provider_*.py files; "
        f"its fixtures are now ungraded"
    )


@pytest.mark.parametrize("provider", sorted(FIXTURES_PENDING))
def test_group_b_exemptions_are_still_uncaptured(provider):
    """Group B means "no fixtures at all". A capture must move the entry out."""
    assert provider in _PROVIDERS, f"{provider} is not a registry key"
    assert provider not in MANIFESTS, (
        f"{provider} now has a manifest — delete its FIXTURES_PENDING row"
    )
    directory = FIXTURES / provider
    assert not directory.exists(), (
        f"{provider} now has {directory} — write its manifest.yaml, or move the "
        f"entry to MANIFEST_PENDING naming the class that grades it"
    )
    assert len(FIXTURES_PENDING[provider]) > 40, (
        f"{provider}'s exemption reason is too thin to be a measurement"
    )


def test_no_provider_sits_in_both_tables():
    assert not (set(MANIFEST_PENDING) & set(FIXTURES_PENDING))


def test_at_least_one_manifest_exists():
    """Guards the guard: an empty discovery would make every row test vacuous."""
    assert MANIFESTS, "no tests/fixtures/*/manifest.yaml was discovered at all"


# --------------------------------------------------------------------------- #
# Manifest shape
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("provider", sorted(MANIFESTS))
def test_manifest_is_well_formed(provider):
    manifest = MANIFESTS[provider]

    assert set(manifest) <= _ALLOWED_TOP, (
        f"unknown top-level keys: {sorted(set(manifest) - _ALLOWED_TOP)}"
    )
    assert manifest.get("schema_version") == 1
    assert manifest.get("provider") == provider, (
        "the manifest's provider key must equal its directory name"
    )
    assert provider in _PROVIDERS, f"{provider} is not a registry key"
    for key in ("cli_version", "captured", "method"):
        assert str(manifest.get(key) or "").strip(), f"{key} is empty"
    assert isinstance(manifest.get("rows"), dict) and manifest["rows"]


@pytest.mark.parametrize("provider", sorted(MANIFESTS))
def test_manifest_declares_every_canonical_row(provider):
    """§6.2's twelve rows, each filled or measured-not-applicable."""
    missing = [row for row in CANONICAL_ROWS if row not in MANIFESTS[provider]["rows"]]

    assert not missing, (
        f"{provider}'s manifest has no cell for {missing}. A row that cannot be "
        f"produced takes `not_applicable: \"<the measurement>\"`; it is never "
        f"simply absent."
    )


@pytest.mark.parametrize("provider,row_name", ROW_IDS,
                         ids=[f"{p}-{r}" for p, r in ROW_IDS])
def test_row_is_well_formed(provider, row_name):
    row = _row(provider, row_name)

    assert set(row) <= _ALLOWED_ROW, (
        f"unknown row keys: {sorted(set(row) - _ALLOWED_ROW)}"
    )
    if "not_applicable" in row:
        assert set(row) <= {"not_applicable", "notes"}, (
            "a not-applicable row declares no files, no exit code and no expectation"
        )
        assert len(str(row["not_applicable"])) > 40, (
            "'not applicable, because …' — the because is the whole point (§6.3)"
        )
        for name in (f"{row_name}.stdout.txt", f"{row_name}.stderr.txt"):
            assert not (FIXTURES / provider / name).exists(), (
                f"{name} exists but the row is declared not applicable"
            )
        return

    assert row.get("stdout") or row.get("stderr"), (
        "a graded row names at least one captured stream"
    )
    assert isinstance(row.get("exit_code"), int), "exit_code is required and is an int"
    expect = row.get("expect")
    assert isinstance(expect, dict), "a graded row carries an `expect` block"
    assert set(expect) <= _ALLOWED_EXPECT, (
        f"unknown expect keys: {sorted(set(expect) - _ALLOWED_EXPECT)}"
    )
    assert isinstance(expect.get("success"), bool), (
        "expect.success is required — it is the minimum a fixture may pin"
    )
    problems = known_wrong_problems(expect)
    assert not problems, "; ".join(problems)
    if "salvage" in expect:
        assert expect["salvage"] in SALVAGE_LABELS, (
            f"expect.salvage is {expect['salvage']!r}; it must be one of "
            f"{list(SALVAGE_LABELS)}"
        )
    lane_status = expect.get("lane_status")
    assert isinstance(lane_status, dict), (
        "expect.lane_status is required: {lane_a: <status>, lane_b: <status>}"
    )
    assert set(lane_status) <= {*LANE_STATUS_KEYS, *LANE_STATUS_MARKERS}, (
        f"unknown lane_status keys: {sorted(set(lane_status) - {*LANE_STATUS_KEYS, *LANE_STATUS_MARKERS})}"
    )
    assert all(key in lane_status for key in LANE_STATUS_KEYS), (
        "expect.lane_status names both lanes, `null` included"
    )
    if LANE_STATUS_OPEN in lane_status:
        assert re.search(r"\bU\d+[a-z]?\b", str(lane_status[LANE_STATUS_OPEN])), (
            "lane_status.open names the unit that will settle the row"
        )
    assert not all(key in lane_status for key in LANE_STATUS_MARKERS), (
        "a row is open or by design, not both"
    )
    if LANE_STATUS_BY_DESIGN in lane_status:
        assert len(str(lane_status[LANE_STATUS_BY_DESIGN])) > 40, (
            "lane_status.by_design states why the difference is permanent"
        )
    for key in ("stdout", "stderr"):
        if row.get(key):
            assert (FIXTURES / provider / row[key]).is_file(), (
                f"{row[key]} is named by the manifest but not committed"
            )
    if row.get("synthesized"):
        assert str(row.get("provenance") or "").strip(), (
            "a synthesized row must say IN THE MANIFEST what was changed and why"
        )


@pytest.mark.parametrize("provider,row_name", ROW_IDS,
                         ids=[f"{p}-{r}" for p, r in ROW_IDS])
def test_row_parses_as_declared(provider, row_name):
    """The captured bytes through the real adapter's ``parse_output``."""
    row = _row(provider, row_name)
    if "not_applicable" in row:
        pytest.skip(row["not_applicable"])

    adapter = _PROVIDERS[provider]
    stdout = _stream(provider, row, "stdout")
    stderr = _stream(provider, row, "stderr")
    expect = row["expect"]

    result = adapter.parse_output(stdout, stderr)

    assert result.get("success") is expect["success"], (
        f"{provider}/{row_name}: parse_output said "
        f"{result.get('success')!r}, manifest declares {expect['success']!r} "
        f"(error: {str(result.get('error'))[:300]!r})"
    )
    if "data" in expect:
        assert result.get("data") == expect["data"], (
            f"{provider}/{row_name}: the unwrapped document is not the declared one"
        )
    for needle in expect.get("error_contains") or ():
        assert needle in str(result.get("error") or ""), (
            f"{provider}/{row_name}: {needle!r} missing from the parser's error"
        )
    for needle in expect.get("raw_contains") or ():
        assert needle in stdout + stderr, (
            f"{provider}/{row_name}: {needle!r} is not in the captured bytes — the "
            f"upstream wording this fixture exists to pin has changed"
        )


@pytest.mark.parametrize("provider,row_name", ROW_IDS,
                         ids=[f"{p}-{r}" for p, r in ROW_IDS])
def test_row_classifies_as_declared(provider, row_name):
    """``classify_failure``, wherever the row declares a category.

    ``classify_failure: null`` is an assertion in its own right — "nothing was
    recognized here" — and on a success row it is the one that catches an
    adapter reading its own happy path as a failure. A row with
    ``known_wrong`` fails once the adapter gives its ``intended`` verdict.
    """
    row = _row(provider, row_name)
    if "not_applicable" in row:
        pytest.skip(row["not_applicable"])
    if "classify_failure" not in row["expect"]:
        pytest.skip("this row declares no classification")

    adapter = _PROVIDERS[provider]
    verdict = adapter.classify_failure(
        _stream(provider, row, "stdout"),
        _stream(provider, row, "stderr"),
        row["exit_code"],
    )

    marker = row["expect"].get("known_wrong")
    if marker is not None:
        assert verdict != marker["intended"], (
            f"{provider}/{row_name}: classify_failure now gives the intended "
            f"{verdict!r}. Delete the row's expect.known_wrong marker, set "
            f"expect.classify_failure: {verdict!r}, and re-pin expect.lane_status"
        )
    assert verdict == row["expect"]["classify_failure"], (
        f"{provider}/{row_name}: classify_failure gave {verdict!r}, the manifest "
        f"pins {row['expect']['classify_failure']!r}"
    )


#: The rows whose measured verdict carries ``expect.known_wrong``.
KNOWN_WRONG_ROWS = [
    (provider, row_name)
    for provider, row_name in ROW_IDS
    if "known_wrong" in (_row(provider, row_name).get("expect") or {})
]

_GOOD_MARKER = {"intended": "rate_limited", "reason": "the reset defeats the pattern"}


@pytest.mark.parametrize("expect,needle", [
    ({"classify_failure": None, "known_wrong": "rate_limited"}, "is a mapping"),
    ({"classify_failure": None,
      "known_wrong": {**_GOOD_MARKER, "fix": "U94"}}, "takes exactly"),
    ({"classify_failure": None,
      "known_wrong": {"intended": "rate_limited"}}, "takes exactly"),
    ({"known_wrong": _GOOD_MARKER}, "beside the measured"),
    ({"classify_failure": None,
      "known_wrong": {**_GOOD_MARKER, "intended": "rate-limited"}}, "must be one of"),
    ({"classify_failure": None,
      "known_wrong": {**_GOOD_MARKER, "intended": None}}, "must be one of"),
    ({"classify_failure": "rate_limited", "known_wrong": _GOOD_MARKER}, "nothing is wrong"),
    ({"classify_failure": None,
      "known_wrong": {**_GOOD_MARKER, "reason": "  "}}, "reason says why"),
    ({"classify_failure": None,
      "known_wrong": {**_GOOD_MARKER, "reason": None}}, "reason says why"),
], ids=["not-a-mapping", "unknown-key", "no-reason-key", "no-measured-pin",
        "unknown-verdict", "null-verdict", "intended-is-measured", "blank-reason",
        "null-reason"])
def test_a_malformed_known_wrong_marker_is_rejected(expect, needle):
    problems = known_wrong_problems(expect)
    assert any(needle in problem for problem in problems), problems


def test_a_well_formed_known_wrong_marker_passes():
    assert known_wrong_problems({"classify_failure": None, "known_wrong": _GOOD_MARKER}) == []
    assert known_wrong_problems({"classify_failure": None}) == []


@pytest.mark.parametrize("provider,row_name", KNOWN_WRONG_ROWS,
                         ids=[f"{p}-{r}" for p, r in KNOWN_WRONG_ROWS])
def test_a_known_wrong_row_fails_once_its_verdict_is_fixed(
    provider, row_name, monkeypatch
):
    """The classifier fix the marker waits for: the adapter gives ``intended``
    and the row's own classification check asks for the marker to go."""
    intended = _row(provider, row_name)["expect"]["known_wrong"]["intended"]
    monkeypatch.setattr(
        _PROVIDERS[provider], "classify_failure", lambda *_args, **_kwargs: intended
    )

    with pytest.raises(AssertionError, match="Delete the row's expect.known_wrong marker"):
        test_row_classifies_as_declared(provider, row_name)


# --------------------------------------------------------------------------- #
# Both lanes' ladder status, read from the capture alone
# --------------------------------------------------------------------------- #

#: ``invoke_with_file_output``'s hard failures: returned as they are, the file
#: channel never read.
_HARD_FAILURES = frozenset({
    "BINARY_NOT_FOUND", "TIMEOUT", "SUBPROCESS_FAILED", "PROMPT_TOO_LONG",
    "PROMPT_UNSAFE",
})


def _capture(provider: str, row: dict):
    from utils.provider_outcome import Capture

    return Capture(
        stdout=_stream(provider, row, "stdout"),
        stderr=_stream(provider, row, "stderr"),
        exit_code=row["exit_code"],
    )


def _fan_out_view(outcome: dict, provider: str, capture) -> dict:
    """What ``invoke_with_file_output`` returns for ``outcome`` (``capture``'s
    reading) when the model wrote no result file: a hard failure as it is,
    stdout data it accepts as a success, anything else as its "both failed"
    line, ``details`` kept. ``test_the_fan_out_view_is_invoke_with_file_outputs``
    holds this to the real function."""
    from utils.llm_client import _is_valid_parsed_data
    from utils.provider_outcome import FAILURE_CATEGORY_KEY, rejected_answer_category

    if not outcome.get("success") and outcome.get("error_code") in _HARD_FAILURES:
        return outcome
    if _is_valid_parsed_data(outcome.get("data")):
        return {"success": True, "details": outcome.get("details", {})}
    details = outcome.get("details", {})
    if outcome.get("success") and FAILURE_CATEGORY_KEY not in details:
        # Parsed, but not into an answer: the executor's verdict for it.
        details = {**details, FAILURE_CATEGORY_KEY: rejected_answer_category(
            _PROVIDERS[provider], capture.stdout, capture.stderr,
        )}
    return {
        "success": False,
        "error": "JSON parsing failed from both file and stdout",
        "details": details,
    }


def _lane_a_status(provider: str, capture) -> object:
    from utils.provider_outcome import outcome_from_capture
    from utils.review_orchestrator_base import attempt_ladder_status

    outcome = outcome_from_capture(
        _PROVIDERS[provider], capture, provider_name=provider, model="m",
    )
    seen = _fan_out_view(outcome, provider, capture)
    return attempt_ladder_status(seen, seen.get("details") or {})


def _lane_b_status(provider: str, capture) -> object:
    """The executor's reading of one capture, in ``dispatch_task``'s order."""
    from utils import subagent_executor as executor
    from utils.retry_ladder import STATUS_OK

    adapter = _PROVIDERS[provider]
    status, _ = executor._classify_pre_parse(capture)
    if status != STATUS_OK:
        return status
    try:
        executor._extract_result_and_syntax(
            adapter, capture.stdout, capture.stderr, prefer_arrays=True,
            capped="stdout" in capture.capped_streams,
        )
    except executor.ResultRejected as exc:
        status = exc.status
    else:
        return STATUS_OK
    if status in executor.RECLASSIFIABLE_STATUSES:
        reclassified = executor._classify_failed_run(adapter, capture, "m")
        if reclassified is not None:
            status = reclassified[0]
    return status


def _retried(provider: str, status: object) -> bool:
    """Whether one attempt with ``status`` is sent again, on the same ladder
    for both lanes: one retry, the adapter's own retry envelope."""
    from utils.retry_ladder import RetryLadder

    ladder = RetryLadder(
        timeout=100000, retries=1,
        envelope=getattr(_PROVIDERS[provider], "internal_retries", None),
    )
    ladder.next_attempt()
    return ladder.record(status).retry


@pytest.mark.parametrize("provider,row_name", ROW_IDS,
                         ids=[f"{p}-{r}" for p, r in ROW_IDS])
def test_row_reaches_its_lane_status(provider, row_name):
    """Both lanes' status for the captured bytes, against the manifest's pin.

    A changed parser or classifier fails here, naming the row and both
    statuses. ``open`` must sit exactly on the rows whose lanes answer
    retry-or-stop differently, outside the deliberate exception."""
    row = _row(provider, row_name)
    if "not_applicable" in row:
        pytest.skip(row["not_applicable"])
    pinned = row["expect"].get("lane_status") or {}
    capture = _capture(provider, row)
    found = {
        "lane_a": _lane_a_status(provider, capture),
        "lane_b": _lane_b_status(provider, capture),
    }
    assert found == {key: pinned.get(key) for key in LANE_STATUS_KEYS}, (
        f"{provider}/{row_name}: the lanes now say {found}; the manifest pins "
        f"{ {key: pinned.get(key) for key in LANE_STATUS_KEYS} }"
    )
    differs = (
        _retried(provider, found["lane_a"]) != _retried(provider, found["lane_b"])
        and found["lane_a"] not in LANE_A_STOPS_BY_DESIGN
    )
    marker = next((key for key in LANE_STATUS_MARKERS if key in pinned), None)
    if differs:
        assert marker is not None, (
            f"{provider}/{row_name}: the fan-out "
            f"{'retries' if _retried(provider, found['lane_a']) else 'stops'} and "
            f"the executor does not; mark the row `open: <unit>` or fix a lane"
        )
    else:
        assert marker is None, (
            f"{provider}/{row_name}: both lanes now make the same call; delete "
            f"the row's `{marker}: {pinned.get(marker)}` marker"
        )
    if marker == LANE_STATUS_BY_DESIGN:
        assert row["exit_code"] == 0 and found == BY_DESIGN_LANES, (
            f"{provider}/{row_name}: `by_design` marks only an exit 0 the fan-out "
            f"stops on and the executor retries once ({BY_DESIGN_LANES})"
        )
        assert _PROVIDERS[provider].EXIT0_CLASSIFIABLE is False, (
            f"{provider} is EXIT0_CLASSIFIABLE now: re-pin {row_name} and delete "
            f"its `by_design` marker"
        )


@pytest.mark.parametrize("provider,row_name", ROW_IDS,
                         ids=[f"{p}-{r}" for p, r in ROW_IDS])
def test_the_fan_out_view_is_invoke_with_file_outputs(
    provider, row_name, monkeypatch, tmp_path
):
    """:func:`_fan_out_view` against the real ``invoke_with_file_output``, spawn
    stubbed: the fast path above reads the fan-out's status the way the
    fan-out does, or this fails first."""
    import contextlib

    from utils import llm_client
    from utils.review_orchestrator_base import attempt_ladder_status

    row = _row(provider, row_name)
    if "not_applicable" in row:
        pytest.skip(row["not_applicable"])
    capture = _capture(provider, row)
    adapter = _PROVIDERS[provider]

    def spawn(cmd, **_kwargs):
        return subprocess.CompletedProcess(
            cmd, capture.exit_code, capture.stdout, capture.stderr
        )

    @contextlib.contextmanager
    def no_dispatch_context(*_args, **_kwargs):
        class _Dispatch:
            refusal = None
        yield _Dispatch()

    monkeypatch.setattr(llm_client, "_run_provider_cli", spawn)
    monkeypatch.setattr(llm_client, "_resolve_executable", lambda name: ([name], False))
    monkeypatch.setattr(llm_client, "provider_dispatch_context", no_dispatch_context)
    # Through the instance __dict__, whose undo deletes the key: the registry's
    # adapters are process-wide, and undoing an instance setattr leaves the
    # original bound method on the instance, shadowing a later test's
    # class-level patch of the same method.
    monkeypatch.setitem(vars(adapter), "is_available", lambda: True)
    monkeypatch.setitem(
        vars(adapter), "build_command_for_transport", lambda *a, **k: (["/bin/true"], None)
    )
    monkeypatch.setitem(vars(adapter), "resolve_env", lambda *a, **k: {})
    result = llm_client.invoke_with_file_output(
        prompt_template="hi {output_json_path}", model_spec=f"{provider}:m",
        prompt_context={}, output_dir=tmp_path, phase="review-plan", timeout=30,
        prefer_arrays=True, cwd=str(tmp_path),
    )
    assert attempt_ladder_status(result, result.get("details") or {}) == (
        _lane_a_status(provider, capture)
    )
