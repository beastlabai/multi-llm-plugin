"""The generic §6.2 fixture harness, and the roster guard that closes gap 4.

**Why this module exists.** Seven providers ship parser fixtures under
``tests/fixtures/<provider>/``; each set is consumed by a hand-written class in
``test_providers.py`` (copilot's is ~450 lines). Nothing discovered fixture
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
      empty:
        not_applicable: "<the measurement that says why>"

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

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import _PROVIDERS  # noqa: E402

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
#: class in ``test_providers.py`` rather than by a manifest. The value is that
#: class's name and both it and the fixture directory are asserted to exist, so
#: this table cannot be used to park a provider that has no fixtures at all.
#: Migrating one of these to a manifest is a strict improvement and deletes its
#: row here.
MANIFEST_PENDING = {
    "codex": "TestCodexParseRealCaptures",
    "copilot": "TestCopilotProvider",
    "kilocode": "TestKiloCodeDefaultModelSpecs",
    "kimi": "TestKimiProvider",
    "opencode": "TestOpenCodeRealSchema",
    "qwen-code": "TestQwenCodeProvider",
    "vibe": "TestVibeProvider",
}

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
                   "classify_failure"}


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
        f"nor an entry in MANIFEST_PENDING / FIXTURES_PENDING. Capture its §6.2 "
        f"rows, or add it to one table WITH the measured reason it cannot be "
        f"captured yet."
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
    source = (Path(__file__).parent / "test_providers.py").read_text(encoding="utf-8")
    assert f"class {consumer}" in source, (
        f"{provider}'s declared consumer {consumer} is gone from test_providers.py; "
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
    adapter reading its own happy path as a failure.
    """
    row = _row(provider, row_name)
    if "not_applicable" in row:
        pytest.skip(row["not_applicable"])
    if "classify_failure" not in row["expect"]:
        pytest.skip("this row declares no classification")

    adapter = _PROVIDERS[provider]

    assert adapter.classify_failure(
        _stream(provider, row, "stdout"),
        _stream(provider, row, "stderr"),
        row["exit_code"],
    ) == row["expect"]["classify_failure"]
