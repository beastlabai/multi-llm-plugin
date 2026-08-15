"""The fake-binary lane's roster and ``mock_llm.py``'s tables must agree (C5).

The container matrix (``tests/container/redirect.yaml``) tags every CLI with the
lane it runs in. A ``lane: fake`` row is not a description — it is a **promise**
that ``tests/mocks/mock_llm.py`` can stand in for that CLI, and the fake binary
is the only coverage such a row will ever get, because redirection to a local
stub was measured and found impossible for it. This file is what turns that
promise into something a test types out loud, in the same
declaration-vs-implementation spirit as ``tests/test_providers.py``.

**Adding an identity is five parts, and no part is optional:**

1. the name in ``get_provider_name``'s ``known_providers`` **and** in
   ``SkillRunner.PROVIDERS`` (the launcher), handling launcher-name ≠ provider-id
   wherever it applies;
2. an argv parser in ``parse_provider_args`` accepting **exactly** what that
   adapter's ``build_command`` emits, for every profile it declares;
3. a formatter emitting that CLI's real wire format — never the silent
   cursor-agent fallback;
4. a ``VERSION_STRINGS`` entry at or above the adapter's minimum-version gate;
5. ``AUTH_PROBE_ARGS`` + ``AUTH_OK_STRINGS`` for any adapter declaring an
   ``auth_check`` — and for no other, since a probe the real CLI does not offer
   is as much a fabrication as a wire format it never emits.

**The two directions, and why the second one is not C5's literal wording.**
C5 was written before the Phase 1 spike and expected up to seven CLIs to drop
into this lane; it therefore asked for "a roster entry with no mock identity
fails, and a mock identity missing from the roster fails too". The spike measured
**11 of 13 CLIs as redirectable**, so the roster collapsed to exactly two rows
(``cursor-agent`` and ``agy``) while ``mock_llm.py`` still carries six identities
for the *existing* mock-based suite (``tests/test_e2e_*.py`` and friends) — five
of which are now provider-lane CLIs. Implemented literally, the second direction
would fail on ``claude-code``, ``codex``, ``gemini``, ``kilocode`` and
``opencode``: a red test about entirely correct code. The corrected second
direction keeps what that half was for — no identity may be a fiction — by
asserting instead that **every mock identity names a provider this skill really
ships**. An orphan identity for a provider that does not exist is still caught;
a provider-lane CLI that the mock-based suite legitimately emulates is not.

The round trip itself — argv → parser → formatter → the adapter's own
``parse_output`` — lives in ``tests/test_mock_adapter_roundtrip.py``, which runs
over every identity rather than just the roster.

Plain CI: no container, no Docker, no network, no subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import get_provider  # noqa: E402

from .harness import SkillRunner  # noqa: E402
from .mocks import mock_llm  # noqa: E402
from .test_mock_adapter_roundtrip import (  # noqa: E402
    PROMPT,
    agent_argv,
    mock_identities,
    parse_with_mock,
)

SKILL_DIR = Path(__file__).resolve().parent.parent
PROVIDERS_YAML = SKILL_DIR / "providers.yaml"
REDIRECT_YAML = Path(__file__).resolve().parent / "container" / "redirect.yaml"

#: The file every failure message must name, because it is the only place any of
#: the five parts can be added.
MOCK_FILE = "tests/mocks/mock_llm.py"

#: An identity the mock does not know, used to prove a branch is real: the
#: fallback parser accepts only ``--model`` plus one positional, so an argv it
#: swallows is an argv no adapter-specific branch was needed for.
UNKNOWN_IDENTITY = "not-a-provider-the-mock-knows"


def _fake_rows() -> List[Dict[str, object]]:
    data = yaml.safe_load(REDIRECT_YAML.read_text(encoding="utf-8"))
    return [row for row in data["clis"] if row.get("lane") == "fake"]


FAKE_ROWS = _fake_rows()
ROSTER = [str(row["cli"]) for row in FAKE_ROWS]


def _providers_yaml_names() -> List[str]:
    config = yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8")) or {}
    return sorted(config.get("providers") or {})


class TestRosterIsPopulated:
    """A roster that collapsed to nothing would make everything below vacuous."""

    def test_the_matrix_still_has_fake_lane_rows(self):
        assert ROSTER, (
            f"{REDIRECT_YAML} has no `lane: fake` row. Either the matrix moved "
            f"or the lane was retired; a parametrization over an empty roster "
            f"passes without asserting anything."
        )

    def test_every_rostered_cli_is_a_shipped_provider(self):
        """A roster naming a CLI this skill does not configure is a dead promise."""
        shipped = set(_providers_yaml_names())
        unknown = [name for name in ROSTER if name not in shipped]
        assert not unknown, (
            f"{REDIRECT_YAML} rosters {unknown} for the fake lane, but "
            f"providers.yaml configures no such provider."
        )


@pytest.mark.parametrize("identity", ROSTER)
class TestRosteredIdentityHasAllFiveParts:
    """Direction one: every ``lane: fake`` row has a COMPLETE mock identity.

    Each test names one part and the file to edit, because the whole hazard here
    is that a missing part is silent: the run still exits 0, it just answers in
    another CLI's format or refuses an argv that looks like a provider failure.
    """

    def test_part1_launcher_name_resolves_and_is_installed(self, identity):
        provider = get_provider(identity)
        assert provider is not None, f"no adapter ships for the rostered {identity!r}"
        launcher = provider.build_command(PROMPT, "m")[0]

        saved = sys.argv
        sys.argv = [f"/tmp/bin/{launcher}"]
        try:
            detected = mock_llm.get_provider_name()
        finally:
            sys.argv = saved

        assert detected == identity, (
            f"part 1: a launcher named {launcher!r} is detected as {detected!r}. "
            f"Add {identity!r} to get_provider_name()'s known_providers in "
            f"{MOCK_FILE} (or, when the launcher name differs from the provider "
            f"id the way `claude` does, to its special case)."
        )
        assert launcher in SkillRunner.PROVIDERS, (
            f"part 1: {launcher!r} is not in SkillRunner.PROVIDERS "
            f"(tests/harness/skill_runner.py), so no launcher by that name is "
            f"ever placed on PATH and the fake lane cannot run this row at all."
        )

    def test_part2_parser_accepts_exactly_the_adapters_argv(self, identity):
        provider = get_provider(identity)
        for profile in provider.agent_profiles:
            argv = agent_argv(provider, profile, "mock-model", PROMPT)
            try:
                _args, parsed = parse_with_mock(identity, argv)
            except SystemExit as exc:
                pytest.fail(
                    f"part 2: the mock's {identity} parser refused the argv "
                    f"{type(provider).__name__}.build_command emits for profile "
                    f"{profile!r} (argparse exit {exc.code}): {argv}. Fix "
                    f"parse_provider_args() in {MOCK_FILE}."
                )
            assert parsed == PROMPT, (
                f"part 2: the {identity} parser accepted the {profile!r} argv but "
                f"recovered {parsed!r} as the prompt. Fix parse_provider_args() "
                f"in {MOCK_FILE}."
            )

            # And the branch is really this identity's, not the generic
            # `--model` + positional fallback every unknown provider gets.
            with pytest.raises(SystemExit):
                parse_with_mock(UNKNOWN_IDENTITY, argv)

    def test_part3_formatter_is_this_clis_own(self, identity):
        expected_name = f"format_{identity.replace('-', '_')}_output"
        formatter = getattr(mock_llm, expected_name, None)
        assert formatter is not None, (
            f"part 3: {MOCK_FILE} defines no {expected_name}(), so format_output() "
            f"answers {identity!r} in CURSOR-AGENT's wire format and the adapter "
            f"is handed a shape its real CLI never produces."
        )

        content = '[{"id": "roster-1"}]'
        assert mock_llm.format_output(identity, content) == formatter(content), (
            f"part 3: {expected_name}() exists but format_output() does not "
            f"dispatch to it; the identity is silently falling back. Add it to "
            f"the formatters table in {MOCK_FILE}."
        )
        if identity != "cursor-agent":
            assert mock_llm.format_output(identity, content) != (
                mock_llm.format_cursor_agent_output(content)
            ), f"part 3: {identity!r} emits cursor-agent's format verbatim."

    def test_part4_version_string_clears_the_minimum(self, identity):
        provider = get_provider(identity)
        version = mock_llm.VERSION_STRINGS.get(identity)
        assert version and version != "0.0.0", (
            f"part 4: no usable VERSION_STRINGS entry for {identity!r} in "
            f"{MOCK_FILE}, so `{identity} --version` prints the 0.0.0 fallback "
            f"and any minimum-version preflight (D3m) refuses the runner."
        )

        parsed = provider.parse_version(version, "")
        assert parsed, (
            f"part 4: {type(provider).__name__}.parse_version cannot read "
            f"{version!r}, which is what the D3m preflight would record as "
            f"`unknown`."
        )
        if provider.min_version:
            assert _version_tuple(parsed) >= _version_tuple(provider.min_version), (
                f"part 4: VERSION_STRINGS[{identity!r}] is {version!r}, below the "
                f"adapter's min_version {provider.min_version!r}."
            )

    def test_part5_auth_probe_matches_the_adapters_auth_check(self, identity):
        provider = get_provider(identity)
        if not provider.auth_check:
            assert identity not in mock_llm.AUTH_PROBE_ARGS, (
                f"part 5: {MOCK_FILE} answers an auth probe for {identity!r}, but "
                f"the adapter declares no auth_check — the mock would be "
                f"emulating a subcommand the real CLI does not have."
            )
            assert identity not in mock_llm.AUTH_OK_STRINGS
            return

        expected = tuple(provider.auth_check[1:])
        assert mock_llm.AUTH_PROBE_ARGS.get(identity) == expected, (
            f"part 5: the adapter's auth_check is {provider.auth_check}, so "
            f"AUTH_PROBE_ARGS[{identity!r}] must be {expected} in {MOCK_FILE}; a "
            f"probe the mock does not recognise is parsed as a PROMPT and "
            f"answered with a fixture."
        )
        assert mock_llm.AUTH_OK_STRINGS.get(identity), (
            f"part 5: no AUTH_OK_STRINGS entry for {identity!r} in {MOCK_FILE}."
        )
        assert mock_llm.is_auth_probe(list(expected)), (
            f"part 5: is_auth_probe() does not recognise {list(expected)}."
        )


def _version_tuple(version: str):
    """Compare dotted versions numerically, tolerating trailing junk."""
    parts = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class TestEveryMockIdentityIsAShippedProvider:
    """Direction two (see the module docstring for why it is not C5's wording).

    A mock identity is a claim that some provider's CLI behaves this way. An
    identity naming a provider this skill does not ship is a claim about
    nothing: it can never be selected, it can never be round-tripped, and it
    quietly rots until someone reads it as coverage.
    """

    @pytest.mark.parametrize("identity", mock_identities())
    def test_identity_names_a_provider_in_providers_yaml(self, identity: str):
        shipped = _providers_yaml_names()
        assert identity in shipped, (
            f"{MOCK_FILE} carries the identity {identity!r}, which providers.yaml "
            f"does not configure ({shipped}). Either the provider was removed "
            f"and the identity should go with it, or the name is a typo — either "
            f"way nothing can ever dispatch it."
        )

    @pytest.mark.parametrize("identity", mock_identities())
    def test_identity_has_an_adapter_to_round_trip_against(self, identity: str):
        assert get_provider(identity) is not None, (
            f"{MOCK_FILE} carries the identity {identity!r} with no hardcoded "
            f"adapter under utils/providers/, so its wire format is a shape "
            f"nothing in this repo ever parses."
        )

    def test_the_roster_is_a_subset_of_the_identities(self):
        """Direction one, stated once over the whole roster.

        The per-row tests above name the missing part; this one names the row,
        so a roster addition with no mock work at all fails with the roster's
        own vocabulary rather than five separate part failures.
        """
        identities = set(mock_identities())
        missing = [name for name in ROSTER if name not in identities]
        assert not missing, (
            f"{REDIRECT_YAML} promises the fake lane can stand in for {missing}, "
            f"but {MOCK_FILE} carries no identity for them. Add the five parts, "
            f"or drop the row from the roster and record it as uncovered in "
            f"tests/container/README.md."
        )
