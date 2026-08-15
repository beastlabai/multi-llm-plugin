"""Every mock identity must survive a full adapter round trip (C5's gate).

**Why this file exists.** ``tests/mocks/mock_llm.py`` emulates a provider CLI by
name, and *every one of its lookups falls back silently*: ``format_output``
falls back to ``format_cursor_agent_output``, ``parse_provider_args`` to a bare
``--model`` + positional parser, ``VERSION_STRINGS`` to ``"0.0.0"``, and
``AUTH_PROBE_ARGS`` simply does not recognise a probe. So an unimplemented — or
half-implemented — provider does not fail loudly: it emits **another CLI's wire
format**, and the adapter is handed a shape its real CLI never produces. A green
row on that proves the mock talked to itself, which is worse than no row and is
the same class of defect as the ``codex`` ``item.completed`` schema bug this
whole design exists to catch (issue #2).

**What it asserts**, per mock identity and per profile that identity's adapter
declares:

1. ``build_command`` / ``build_agent_command`` renders the argv,
2. the launcher name in ``argv[0]`` really resolves back to that identity,
3. the mock's parser **accepts that argv** and recovers the prompt (an argparse
   ``SystemExit`` fails the case — it is exit 2, indistinguishable from a
   provider failure once it is behind a subprocess),
4. the mock's formatter output is fed to the adapter's own ``parse_output``,
   which must return the content that went in, with ``parse_usage`` agreeing
   with the declared ``usage_support``.

This is ``tests/test_providers.py``'s declaration-vs-implementation discipline
applied to the mock: nothing here is new *behaviour*, it is the first assertion
over behaviour the mock was already assumed to have.

**Scope note.** C5 wrote this test as "parametrized over the roster" (the
``lane: fake`` rows of ``tests/container/redirect.yaml``). It runs over **every**
mock identity instead, deliberately: the roster is two rows after the Phase 1
spike, while five of the six pre-existing identities serve the mock-based e2e
suite and are exactly the ones whose round trip had never been asserted. Running
the wider set is what makes a failure here a real finding about the current mock
rather than a statement about the container matrix.

Plain CI: no container, no Docker, no network, no subprocess.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Set, Tuple

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provider_registry import get_provider  # noqa: E402
from utils.providers.base import (  # noqa: E402
    PROFILE_GENERATE,
    USAGE_NONE,
    LLMProvider,
)

from .mocks import mock_llm  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
PROVIDERS_YAML = SKILL_DIR / "providers.yaml"
MOCK_LLM_PATH = Path(mock_llm.__file__).resolve()

#: The prompt every case round-trips. Deliberately ordinary text: a prompt
#: starting with "-" is a separate (real) hazard tracked in redirect.yaml's
#: opencode row, not this test's subject.
PROMPT = "Review this plan and list every risk you find"

#: The answer the mock is asked to carry, in the shape the skill's own phases
#: exchange, so ``parse_output`` returning it proves the whole envelope was
#: unwrapped rather than merely parsed as JSON.
CONTENT_OBJ: List[Dict[str, object]] = [
    {
        "id": "roundtrip-1",
        "type": "suggestion",
        "description": "The wire format survived the adapter's parser",
        "importance": "MEDIUM",
        "location": "mock_file.py:10",
    }
]
CONTENT = json.dumps(CONTENT_OBJ)

#: Placeholder profile for an identity the mock claims and the registry does not
#: ship. It cannot be round-tripped, so it is carried as a case that FAILS with
#: that reason rather than being dropped (a dropped case is a vacuous pass).
NO_ADAPTER = "<no adapter in the provider registry>"


# ---------------------------------------------------------------------------
# The mock's identity tables. These are function-local by design (they are the
# emulation's own dispatch), so they are read structurally rather than copied
# here — a copy would be a second list to maintain and would agree with the mock
# by construction instead of by assertion.
#
# ``tests/test_fake_lane_roster.py`` imports these helpers.
# ---------------------------------------------------------------------------


def _mock_ast() -> ast.Module:
    return ast.parse(MOCK_LLM_PATH.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_mock_ast()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"{MOCK_LLM_PATH} no longer defines {name}(); this test reads the mock's "
        f"identity tables out of it."
    )


def _assigned_literal(func: ast.FunctionDef, variable: str) -> ast.AST:
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        ):
            return node.value
    raise AssertionError(
        f"{MOCK_LLM_PATH}: {func.name}() no longer assigns {variable}; this test "
        f"reads the mock's identity tables out of it."
    )


def detected_launcher_names() -> List[str]:
    """The launcher names ``get_provider_name`` matches on."""
    return list(ast.literal_eval(
        _assigned_literal(_function("get_provider_name"), "known_providers")
    ))


def formatter_identities() -> Set[str]:
    """The identities ``format_output`` dispatches on (everything else falls back)."""
    node = _assigned_literal(_function("format_output"), "formatters")
    assert isinstance(node, ast.Dict)
    return {ast.literal_eval(key) for key in node.keys}


def parser_identities() -> Set[str]:
    """The identities ``parse_provider_args`` has a real branch for."""
    names: Set[str] = set()
    for node in ast.walk(_function("parse_provider_args")):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id == "provider"):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                names.add(comparator.value)
    return names


def _special_cased_identities() -> Set[str]:
    """Identities ``get_provider_name`` returns directly (the ``claude`` case).

    The claude-code adapter's binary is ``claude``, so its launcher name and its
    provider id differ and it can never appear in ``known_providers``.
    """
    returned: Set[str] = set()
    for node in ast.walk(_function("get_provider_name")):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if isinstance(value, str) and value != "unknown":
                returned.add(value)
    return returned


def mock_identities() -> List[str]:
    """Every provider identity ``mock_llm.py`` claims, from all four tables.

    The union, not any single table: a half-added identity is precisely one that
    appears in some of them, and each missing part is a *silent* fallback.
    """
    return sorted(
        set(detected_launcher_names())
        | _special_cased_identities()
        | set(mock_llm.VERSION_STRINGS)
        | formatter_identities()
        | parser_identities()
    )


# ---------------------------------------------------------------------------
# Round-trip mechanics
# ---------------------------------------------------------------------------


def configured_models(identity: str) -> List[str]:
    """Every model ``providers.yaml`` configures for ``identity``.

    Read from the file rather than through ``load_config`` so collection does
    not depend on the config-layering fixtures, and so the cases are the ones a
    user's *shipped* configuration can really produce — the ``model:effort``
    entries included, which are the specs that change the rendered argv.
    """
    config = yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8")) or {}
    entry = (config.get("providers") or {}).get(identity) or {}
    return list(entry.get("models") or ["mock-model"])


def agent_argv(
    provider: LLMProvider, profile: str, model: str, prompt: str
) -> List[str]:
    """The argv this adapter really renders for ``profile``."""
    if profile == PROFILE_GENERATE:
        return provider.build_command(prompt, model)
    return provider.build_agent_command(prompt, model, profile)


def parse_with_mock(identity: str, argv: Sequence[str]) -> Tuple[object, str]:
    """Run ``mock_llm.parse_provider_args`` against ``argv`` as the CLI would see it.

    The mock's parsers read ``sys.argv`` directly (they are an executable's
    parsers, not a library's), so the invocation has to be staged the same way —
    ``argv[0]`` included, since that is what ``get_provider_name`` keys on.

    Raises:
        SystemExit: whatever argparse did, unswallowed, so callers can report it
            as the exit-2 provider-failure lookalike it becomes in a real run.
    """
    saved = sys.argv
    sys.argv = list(argv)
    try:
        return mock_llm.parse_provider_args(identity)
    finally:
        sys.argv = saved


def _argv_shape(argv: Sequence[str]) -> Tuple[str, ...]:
    """The flags (and subcommand) an argv carries, ignoring every value.

    Used to collapse "eleven configured models that all render the same flags"
    into one case while keeping the ``model:effort`` entries — which really do
    add a flag (``--effort``, ``--variant``, ``-c model_reasoning_effort=…``) —
    as cases of their own.
    """
    flags = tuple(arg for arg in argv if arg.startswith("-"))
    subcommand = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else ""
    return (subcommand,) + flags


class RoundTripCase(NamedTuple):
    """One (identity, profile, argv shape) the round trip must hold for."""

    identity: str
    profile: str
    model: str
    argv: Tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{self.identity}-{self.profile}-{self.model}"


def _cases() -> List[RoundTripCase]:
    cases: List[RoundTripCase] = []
    for identity in mock_identities():
        provider = get_provider(identity)
        if provider is None:
            cases.append(RoundTripCase(identity, NO_ADAPTER, "", ()))
            continue
        for profile in provider.agent_profiles:
            shapes: Set[Tuple[str, ...]] = set()
            for model in configured_models(identity):
                argv = tuple(agent_argv(provider, profile, model, PROMPT))
                shape = _argv_shape(argv)
                if shape in shapes:
                    continue
                shapes.add(shape)
                cases.append(RoundTripCase(identity, profile, model, argv))
    return cases


CASES = _cases()


class TestMockParserAcceptsTheAdaptersArgv:
    """Leg one: the argv the product builds must parse as the mock's own CLI."""

    def test_there_is_something_to_check(self):
        """A parametrization that collapsed to nothing would pass vacuously."""
        assert CASES, "no mock identity produced a case; the tables above moved"

    @pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
    def test_argv_parses_and_the_prompt_comes_back(self, case: RoundTripCase):
        if case.profile == NO_ADAPTER:
            pytest.fail(
                f"mock_llm.py claims the identity {case.identity!r}, but the "
                f"provider registry ships no adapter for it. Either remove the "
                f"identity from tests/mocks/mock_llm.py or add the adapter under "
                f"utils/providers/."
            )
        provider = get_provider(case.identity)

        saved = sys.argv
        sys.argv = list(case.argv)
        try:
            detected = mock_llm.get_provider_name()
        finally:
            sys.argv = saved
        assert detected == case.identity, (
            f"a launcher named {case.argv[0]!r} (which is what "
            f"{type(provider).__name__}.build_command puts on PATH) is detected "
            f"as {detected!r}, not {case.identity!r}. Fix "
            f"get_provider_name()'s known_providers in tests/mocks/mock_llm.py."
        )

        try:
            _args, parsed_prompt = parse_with_mock(case.identity, case.argv)
        except SystemExit as exc:
            pytest.fail(
                f"the mock's {case.identity} parser REFUSED the argv "
                f"{type(provider).__name__}.build_command really emits for "
                f"profile {case.profile!r} (argparse exit {exc.code}): "
                f"{list(case.argv)}. Behind a subprocess that is exit 2 — "
                f"indistinguishable from a provider failure. Fix "
                f"parse_provider_args() in tests/mocks/mock_llm.py."
            )
        assert parsed_prompt == PROMPT, (
            f"the mock's {case.identity} parser accepted the argv but recovered "
            f"{parsed_prompt!r} as the prompt; the run would be answered with "
            f"the wrong (or the default) fixture. Fix parse_provider_args() in "
            f"tests/mocks/mock_llm.py."
        )

    @pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
    def test_stdin_transport_argv_parses_too(self, case: RoundTripCase):
        """The same argv minus the prompt, which is how the executor dispatches.

        Materialized subagent prompts travel on **stdin** (D2d), so the argv the
        mock sees ends in a flag value rather than the prompt. The mock reads the
        prompt from stdin in that case, but only if the argv parsed at all.
        """
        if case.profile == NO_ADAPTER:
            pytest.skip(f"{case.identity}: no adapter (reported by the case above)")
        provider = get_provider(case.identity)
        if not provider.supports_prompt_transport("stdin"):
            pytest.skip(f"{case.identity} declares argv transport only")

        if case.profile == PROFILE_GENERATE:
            argv, stdin_text = provider.build_command_for_transport(
                PROMPT, case.model, "stdin"
            )
        else:
            argv, stdin_text = provider.build_agent_command_for_transport(
                PROMPT, case.model, case.profile, "stdin"
            )
        assert stdin_text == PROMPT and PROMPT not in argv

        try:
            _args, parsed_prompt = parse_with_mock(case.identity, argv)
        except SystemExit as exc:
            pytest.fail(
                f"the mock's {case.identity} parser refused the stdin-transport "
                f"argv for profile {case.profile!r} (argparse exit {exc.code}): "
                f"{list(argv)}. Fix parse_provider_args() in "
                f"tests/mocks/mock_llm.py."
            )
        assert not parsed_prompt, (
            f"the mock recovered a prompt ({parsed_prompt!r}) from an argv that "
            f"carries none; main() would then never read stdin."
        )


class TestMockWireFormatSurvivesTheAdapter:
    """Leg two: the bytes the mock prints must parse as that adapter's own CLI.

    Not parametrized by profile: a CLI's stdout shape is a property of the CLI,
    not of the sandbox policy the profile selected, and every adapter here has
    exactly one ``parse_output``.
    """

    @pytest.mark.parametrize("identity", mock_identities())
    def test_parse_output_recovers_the_content(self, identity: str):
        provider = get_provider(identity)
        if provider is None:
            pytest.fail(
                f"mock_llm.py claims the identity {identity!r} with no adapter in "
                f"the provider registry."
            )

        wire = mock_llm.format_output(identity, CONTENT)
        result = provider.parse_output(wire, "")

        assert result.get("success") is True, (
            f"{type(provider).__name__}.parse_output rejected the wire format "
            f"the mock emits for {identity!r}: {result.get('error')!r}. If the "
            f"identity has no entry in format_output()'s table it silently gets "
            f"CURSOR-AGENT's format — check tests/mocks/mock_llm.py."
        )
        assert result.get("data") == CONTENT_OBJ, (
            f"{type(provider).__name__}.parse_output returned "
            f"{result.get('data')!r} instead of the content the mock was asked "
            f"to carry. The envelope was parsed but not unwrapped, which is what "
            f"another CLI's wire format looks like from here."
        )

    @pytest.mark.parametrize("identity", mock_identities())
    def test_parse_usage_agrees_with_the_declared_support(self, identity: str):
        """D3j: usage is reported or absent, never zero — and never fabricated."""
        provider = get_provider(identity)
        if provider is None:
            pytest.fail(f"mock_llm.py claims {identity!r} with no adapter")

        wire = mock_llm.format_output(identity, CONTENT)
        usage = provider.parse_usage(wire, "")

        if provider.usage_support == USAGE_NONE:
            assert usage is None, (
                f"{identity} declares usage_support='none' but recovered {usage!r} "
                f"from the mock's stream; the declaration and the extractor "
                f"disagree (tests/test_providers.py owns the same invariant from "
                f"the adapter's side)."
            )
            return

        assert isinstance(usage, dict) and usage, (
            f"{identity} declares usage_support={provider.usage_support!r} but "
            f"recovered nothing from the mock's own wire format. The mock's "
            f"formatter must carry the usage block the real CLI emits, or the "
            f"D3j extractor has no negative-free case to prove it on."
        )
        assert any(value for value in usage.values()), (
            f"{identity}: every recovered usage field is falsy ({usage!r}); an "
            f"all-zero report reads as a free run."
        )
