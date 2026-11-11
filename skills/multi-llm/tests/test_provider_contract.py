"""Tests for the contracts every adapter declares (registry-wide).

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.agy import AgyProvider
from utils.providers.aider import AiderProvider
from utils.providers.base import LLMProvider, split_reasoning_effort
from tests.roster_check import assert_covers_registry, assert_pinned_subset
from utils.providers import base as base_module, cursor_agent as cursor_agent_module
from utils.providers import version_gate
from utils.providers.claude_code import ClaudeCodeProvider
from utils.providers.cline import ClineProvider
from utils.providers.codex import CodexProvider
from utils.providers.copilot import CopilotProvider
from utils.providers.fx import FxProvider
from utils.providers.cursor_agent import CursorAgentProvider
from utils.providers.gemini import GeminiProvider
from utils.providers.goose import GooseProvider
from utils.providers.grok import GrokProvider
from utils.providers.kilocode import KiloCodeProvider
from utils.providers.kimi import KimiProvider
from utils.providers.muse import MuseProvider
from utils.providers.omp import OmpProvider
from utils.providers.opencode import OpenCodeProvider
from utils.providers.pi import PiProvider
from utils.providers.qwen_code import QwenCodeProvider
from utils.providers.vibe import VibeProvider
from tests.provider_test_support import _envelope_sweep_names, _registered_provider_names


class TestSplitReasoningEffort:
    """Tests for the split_reasoning_effort helper in providers.base."""

    EFFORTS = frozenset({"low", "medium", "high"})

    def test_effort_suffix_split(self):
        """A whitelisted suffix after the last colon is split off."""
        assert split_reasoning_effort("gpt-5.5:high", self.EFFORTS) == ("gpt-5.5", "high")

    def test_no_colon_passthrough(self):
        """A model without a colon passes through verbatim."""
        assert split_reasoning_effort("gpt-5.5", self.EFFORTS) == ("gpt-5.5", None)

    def test_unknown_suffix_passthrough(self):
        """A suffix not in the whitelist keeps the full model string."""
        assert split_reasoning_effort("gpt-5.5:turbo", self.EFFORTS) == ("gpt-5.5:turbo", None)

    def test_empty_base_passthrough(self):
        """An empty base before the colon (":high") passes through verbatim."""
        assert split_reasoning_effort(":high", self.EFFORTS) == (":high", None)

    def test_empty_suffix_passthrough(self):
        """A trailing colon with no suffix passes through verbatim."""
        assert split_reasoning_effort("m:", self.EFFORTS) == ("m:", None)

    def test_multi_colon_splits_on_last(self):
        """Multi-colon models split on the LAST colon only."""
        assert split_reasoning_effort("a:b:high", self.EFFORTS) == ("a:b", "high")

    def test_whitelist_sensitivity(self):
        """The same string splits or passes through depending on the whitelist."""
        assert split_reasoning_effort("m:max", frozenset({"max"})) == ("m", "max")
        assert split_reasoning_effort("m:max", frozenset({"high"})) == ("m:max", None)


#: §7.6(b)'s guard shape. Two families, both of which move the number §7.5 rule 4
#: subtracts: a *count* of vendor-internal requests (`…MAX_RETRIES`, a backoff or
#: attempt cap) and the *per-request* cutoff those requests are multiplied by
#: (`API_TIMEOUT_MS` — claude-code's `max_wall_s` is literally 4 × this). Both are
#: matched, because either one set without a declared envelope leaves the ladder
#: budgeting for one request against a subprocess that will make several.
#: Deliberately narrow: a generic `*_TIMEOUT` would catch env shaping that says
#: nothing about retries and force a fabricated declaration.
RETRY_SHAPED_ENV_KEY = r"RETRY|RETRIES|BACKOFF|ATTEMPTS?|API_TIMEOUT|REQUEST_TIMEOUT"


def _retry_shaped_env_keys(provider):
    """The retry-shaped variables this adapter's ``get_env`` really sets."""
    import re

    try:
        env = provider.get_env("m")
    except TypeError:  # pragma: no cover - defensive, no adapter needs it today
        env = provider.get_env()
    return sorted(
        key for key in (env or {})
        if re.search(RETRY_SHAPED_ENV_KEY, key.upper())
    )


class TestInternalRetryEnvelopeDeclarations:
    """§7.6(b) — every adapter declares its CLI's internal retry envelope.

    §7.5's arithmetic needs one number per adapter and today that number exists
    only in the plan's measurements. The declaration is the input rule 4
    subtracts: a slot whose remaining budget cannot hold one worst-case attempt
    of the adapter's own vendor envelope is `budget_exhausted` *before* a child
    is spawned, and rule 6(b) refuses an identical rate-limit re-dispatch to an
    adapter that already retried 429s internally.

    Nothing reads it yet — M9's ladder is written and pure, and consumes it
    structurally (`retry_ladder.InternalRetryEnvelopeLike`), which is why these
    tests assert the three attribute *names* rather than an isinstance.
    """

    #: Adapters whose vendor envelope was measured against a fail-forever
    #: stub. `kimi` joined on 0.36.1: 10 attempts, exponential backoff to
    #: ~38 s, 429s retried and `Retry-After` honoured exactly — the row
    #: §6.1.1 explicitly warned must be *confirmed*, not assumed `None`.
    MEASURED = ("claude-code", "codex", "kimi")

    @pytest.fixture
    def registry(self):
        from utils.provider_registry import _PROVIDERS

        return _PROVIDERS

    def test_the_sweep_really_covers_every_registered_adapter(self, registry):
        """The done-when is 'all of them': the sweep is the registry itself.

        No count is pinned: the registry is derived from utils/providers/, and a
        deleted or added adapter is caught by the hand-kept tables it leaves out
        of step (tests/test_provider_roster.py names them).
        """
        assert _envelope_sweep_names() == sorted(registry)

    def test_the_base_default_is_none(self):
        """An adapter that measured nothing declares nothing, by inheriting."""
        assert LLMProvider.internal_retries is None

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_every_adapter_declares_the_attribute(self, registry, name):
        provider = registry[name]

        assert hasattr(provider, "internal_retries")

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_a_declared_envelope_satisfies_the_ladders_protocol(self, registry, name):
        """The seam is structural: three attributes, read off whatever is there."""
        envelope = registry[name].internal_retries
        if envelope is None:
            return

        if envelope.max_requests is None:
            # fx retries without a cap; kilocode's measured chain is per model
            # turn, and a run makes several (its declaration's comment).
            assert name in ("fx", "kilocode"), (
                f"{name} has an unexplained unbounded retry declaration"
            )
            assert envelope.max_wall_s is None
            assert envelope.retries_rate_limit is True
            assert envelope.configurable_by == ()
            return
        assert isinstance(envelope.max_requests, int) and envelope.max_requests >= 1
        assert envelope.max_wall_s is None or envelope.max_wall_s > 0
        assert isinstance(envelope.retries_rate_limit, bool)
        assert isinstance(envelope.configurable_by, tuple)

    @pytest.mark.parametrize(
        "name",
        [
            n for n in _envelope_sweep_names()
            if n not in ("claude-code", "cline", "codex", "copilot", "gemini",
                         "goose", "grok", "kilocode", "kimi",
                         "muse", "omp", "pi", "qwen-code", "vibe", "fx")
        ],
    )
    def test_every_unmeasured_adapter_declares_none(self, registry, name):
        """§7.5 rule 6(c): `None` is 'assume one request', never 'assume the worst'.

        Named without a count for the same reason as the sweep above: the
        unmeasured roster shrinks every time somebody runs §6.1.1's experiments,
        and "the other eleven" was already wrong.
        """
        assert registry[name].internal_retries is None

    # -- the measured rows ------------------------------------------------

    def test_fx_declares_an_unknown_internal_request_bound(self, registry):
        """fx 0.0.14 retries transient failures without a finite request cap."""
        envelope = registry["fx"].internal_retries
        assert envelope is not None
        assert envelope.max_requests is None
        assert envelope.max_wall_s is None
        assert envelope.retries_rate_limit is True

    def _with_timeout(self, monkeypatch, seconds):
        """Pin the resolved per-provider slot timeout the envelope derives from."""
        import utils.provider_registry as registry_module

        monkeypatch.setattr(
            registry_module,
            "get_provider_timeout",
            lambda name, anchor=None, **kw: seconds,
        )

    def test_claude_code_declares_the_post_l6_request_count(self, registry):
        """11 at the CLI default → 4 once L6 sets `CLAUDE_CODE_MAX_RETRIES=3`.

        The shipped number is the post-L6 one because L6 has landed and every
        dispatch carries its env; declaring 11 would budget for a configuration
        this harness never runs.
        """
        from utils.providers.claude_code import CLAUDE_CODE_MAX_RETRIES

        envelope = registry["claude-code"].internal_retries

        assert envelope is not None
        assert envelope.max_requests == 4
        assert envelope.max_requests == CLAUDE_CODE_MAX_RETRIES + 1

    def test_claude_code_retries_rate_limits_internally(self, registry):
        """Which is what makes §7.5 rule 6(b) forbid an identical re-dispatch."""
        assert registry["claude-code"].internal_retries.retries_rate_limit is True

    def test_claude_code_names_both_knobs_l6_sets(self, registry):
        envelope = registry["claude-code"].internal_retries

        assert envelope.configurable_by == ("CLAUDE_CODE_MAX_RETRIES", "API_TIMEOUT_MS")

    @pytest.mark.parametrize("slot_s,expected", [(1800, 1200.0), (600, 480.0)])
    def test_claude_code_max_wall_tracks_the_resolved_timeout(
        self, registry, monkeypatch, slot_s, expected
    ):
        """4 × `API_TIMEOUT_MS`, and `API_TIMEOUT_MS` is itself derived (§7.3).

        The plan's two measured points: **1200 s** at a 1800 s slot and 480 s at
        a 600 s one. A constant here would be wrong at one of them.
        """
        self._with_timeout(monkeypatch, slot_s)

        assert registry["claude-code"].internal_retries.max_wall_s == expected

    def test_claude_code_max_wall_is_four_times_the_emitted_env(
        self, registry, monkeypatch
    ):
        """The declaration and the env are one derivation, not two copies of it."""
        self._with_timeout(monkeypatch, 1800)
        provider = registry["claude-code"]

        env = provider.get_env("sonnet")

        assert provider.internal_retries.max_wall_s == (
            4 * int(env["API_TIMEOUT_MS"]) / 1000.0
        )

    def test_claude_code_declares_no_bound_when_the_slot_omits_the_timeout(
        self, registry, monkeypatch
    ):
        """Below ~120 s `get_env` emits no `API_TIMEOUT_MS`, so the wall is unknown.

        `None` is the honest answer and the one rule 4 handles (no admission
        test); inventing a bound would fail slots that would have succeeded.
        """
        self._with_timeout(monkeypatch, 60)
        provider = registry["claude-code"]

        assert "API_TIMEOUT_MS" not in provider.get_env("sonnet")
        assert provider.internal_retries.max_wall_s is None
        assert provider.internal_retries.max_requests == 4

    def test_cline_declares_its_measured_ladder(self, registry):
        """Six requests on a 500, a 429 or a zero-byte body, 429 retried by
        cline itself, no statable wall (measured on 3.0.70, 2026-10-09:
        `tests/fixtures/cline/manifest.yaml`)."""
        envelope = registry["cline"].internal_retries

        assert envelope is not None
        assert envelope.max_requests == 6
        assert envelope.retries_rate_limit is True
        assert envelope.max_wall_s is None

    def test_codex_declares_thirty_requests(self, registry):
        """(stream+1) × (request+1), defaults 4 and 5 — measured, not inferred."""
        envelope = registry["codex"].internal_retries

        assert envelope is not None
        assert envelope.max_requests == 30

    def test_codex_does_not_retry_a_rate_limit(self, registry):
        """One request, then `exceeded retry limit, last status: 429`.

        So the harness-side backoff is the only one a 429 gets, and rule 6(b)
        must NOT suppress it for this adapter.
        """
        assert registry["codex"].internal_retries.retries_rate_limit is False

    def test_codex_declares_its_measured_wall(self, registry):
        """~25 s on 5xx/stream failures inside the one awaited subprocess."""
        assert registry["codex"].internal_retries.max_wall_s == 25.0

    # -- the guard --------------------------------------------------------

    @pytest.mark.parametrize("name", _envelope_sweep_names())
    def test_a_retry_shaped_env_var_requires_a_declared_envelope(self, registry, name):
        """§7.6(b)'s converse, and the reason the declaration ships with a test.

        An adapter that grows a retry env without updating its envelope is a
        budget bug: §7.5 rule 4 would subtract one request's worth of wall clock
        from the deadline while the child quietly makes several. Red for
        claude-code from the moment L6's `CLAUDE_CODE_MAX_RETRIES` landed.
        """
        provider = registry[name]
        shaped = _retry_shaped_env_keys(provider)
        if not shaped:
            return

        assert provider.internal_retries is not None, (
            f"{name}: get_env sets {shaped} but declares no internal_retries "
            f"envelope — §7.5's budget math would assume one request"
        )

    def test_the_ladder_stays_free_of_provider_imports(self):
        """The structural read IS the seam (§7.6(b), `retry_ladder`'s docstring).

        The ladder is pure by construction; an import of `utils.providers` would
        make the policy object depend on the adapter registry it is meant to be
        handed a value from.
        """
        import ast
        import inspect

        import utils.retry_ladder as ladder

        tree = ast.parse(inspect.getsource(ladder))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")

        assert not [name for name in imported if "provider" in name], (
            f"retry_ladder must read the envelope structurally, never import it "
            f"(found {imported})"
        )


# Per-provider prompt transport table: the transport each adapter's FAN-OUT path
# uses by default. Every row but one declares `argv`, and for those rows what is
# pinned is that the adapter renders the prompt as exactly one argv element, so
# llm_client's Windows argv-length check still applies to that path.
#
# `omp` is the exception and the reason this sentence is no longer "every adapter
# declares argv": it declares `stdin`, because `invoke_with_file_output` — the
# function every Lane A review phase goes through — takes no transport argument
# and forwards none, so an `argv` default would leave every review prompt on the
# path measured failing (200 KB on argv is E2BIG from the kernel before omp
# starts; the same prompt over stdin is answered in one call). A `stdin` row is
# pinned the other way round: the prompt appears on NEITHER argv nor twice.
#
# It is NOT a claim that no other adapter can do better: claude-code, codex,
# opencode and qwen-code each declare
# `supported_prompt_transports = ("argv", "stdin")`, which is the separate member
# a *caller* may ask for. The header used to say "none has a documented stdin or
# prompt-file mode used anywhere in this codebase", which had been false for
# three adapters for some time.
# An adapter that changes its fan-out default must update both its
# prompt_transport and this table.
TRANSPORT_CASES = [
    (AgyProvider, "argv", "Gemini 3.1 Pro (High)"),
    (AiderProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (ClaudeCodeProvider, "argv", "sonnet"),
    (ClineProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (CodexProvider, "argv", "gpt-5.5"),
    # copilot renders the prompt as one argv element like every row here,
    # AND declares a stdin transport a caller may ask for: piped stdin with
    # the `-p` PAIR omitted entirely carried a 30 KB prompt (§6.4, measured).
    (CopilotProvider, "argv", "openai/gpt-4.1-mini"),
    (FxProvider, "stdin", "openai/gpt-5.6-luna"),
    (CursorAgentProvider, "argv", "auto"),
    (GeminiProvider, "argv", "gemini-2.5-pro"),
    # kimi is argv-only by MEASUREMENT, not by default: `-p -` delivers the
    # literal "-" rather than reading stdin, `-p @file` delivers the literal
    # path, and `--prompt-file` does not exist (harness-expansion.md §6.4).
    (KimiProvider, "argv", "moonshotai/kimi-k2-thinking"),
    (GooseProvider, "argv", "openrouter/z-ai/glm-5.2"),
    (GrokProvider, "argv", "grok-4"),
    (KiloCodeProvider, "argv", "openai-native/gpt-5.5"),
    (MuseProvider, "argv", "muse-spark-1.2"),
    # The one `stdin` row, and the first adapter whose fan-out default is not
    # argv. `argv` stays SUPPORTED for a caller that asks for it and renders
    # `-- <prompt>` last; the default is stdin because the harness has no way to
    # ask for it on the path every review prompt takes (§2.5).
    (OmpProvider, "stdin", "openrouter/openai/gpt-5.6-luna"),
    (OpenCodeProvider, "argv", "openai/gpt-5.5"),
    (PiProvider, "argv", "anthropic/claude-opus-4-8"),
    # qwen-code renders the prompt as one argv element like every row here,
    # AND declares a stdin transport a caller may ask for (§6.4, measured).
    (QwenCodeProvider, "argv", "qwen/qwen3.8-max"),
    # vibe renders the prompt as one argv element like every row here, AND
    # declares a stdin transport -- but its stdin form is the odd one out and the
    # difference is load-bearing: `-p` STAYS and merely loses its value, because
    # dropping the flag entirely launches the interactive TUI and hangs forever
    # rather than degrading (§6.4, measured with an 8,427-byte prompt).
    (VibeProvider, "argv", "vibe-default"),
]


class TestPromptTransport:
    """Locks the per-provider prompt transport decision (see TRANSPORT_CASES)."""

    PROMPT = "Analyze this plan carefully"

    @pytest.mark.parametrize(
        "provider_cls,transport,model",
        TRANSPORT_CASES,
        ids=[cls.__name__ for cls, _, _ in TRANSPORT_CASES],
    )
    def test_declared_transport_matches_build_command(
        self, provider_cls, transport, model
    ):
        """prompt_transport is declared as decided AND matches the built argv.

        The assertion is per row rather than the blanket `transport == "argv"`
        it used to be: omp's fan-out default is `stdin`, and grading it against
        argv would have made the first non-argv adapter unrepresentable in the
        table that exists to pin exactly that decision. An `argv` row keeps the
        one-element check; a `stdin` row asserts the mirror image, that the
        prompt reaches argv not at all and `stdin_text` carries it whole —
        omp answers a prompt given on both as TWO user messages, so "not on
        argv" is the assertion and not a tidiness preference.
        """
        provider = provider_cls()

        assert provider.prompt_transport == transport

        if transport == "argv":
            cmd = provider.build_command(self.PROMPT, model)
            # The full prompt travels as exactly ONE argv element (aider
            # prefixes "/ask " but keeps the prompt intact in that element).
            prompt_args = [arg for arg in cmd if self.PROMPT in arg]
            assert len(prompt_args) == 1
            return

        assert transport == "stdin"
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, model, "stdin"
        )
        assert [arg for arg in argv if self.PROMPT in arg] == []
        assert stdin_text == self.PROMPT

    def test_table_covers_every_registered_provider(self):
        """The transport table stays in sync with the provider registry."""
        from utils.provider_registry import _PROVIDERS

        assert_covers_registry(
            {cls.__name__ for cls, _, _ in TRANSPORT_CASES},
            "TRANSPORT_CASES in tests/test_provider_contract.py",
            what="a transport row",
            hint="A row is (AdapterClass, the transport its default build uses, "
            "one real model id).",
            registry={type(p).__name__ for p in _PROVIDERS.values()},
        )


class TestSubagentRunnerDeclarations:
    """Every adapter's D3 capability declarations must match its members.

    The capability data cannot be allowed to drift from the code in either
    direction: an adapter that gains `build_agent_command()` without declaring
    the profile would be rejected as a runner it can actually serve, and one that
    declares a profile it does not implement would be dispatched and fail at
    runtime on a user's host.
    """

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_declared_profiles_match_implemented_members(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        if name == "fx":
            # Fx overrides the dispatcher only to route the generate profile
            # through its specialized CLI command. It delegates agent-readonly
            # to the base implementation and advertises generate alone.
            assert provider.agent_profiles == ("generate",)
            return
        implements = (
            type(provider).build_agent_command is not LLMProvider.build_agent_command
        )
        assert implements == ("agent-readonly" in provider.agent_profiles), (
            f"{name}: build_agent_command implemented={implements} but "
            f"agent_profiles={provider.agent_profiles}"
        )
        # `generate` is every adapter's build_command and is always declared.
        assert "generate" in provider.agent_profiles

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_declared_write_profile_matches_supports_file_edits(self, name):
        """`agent-write` and `supports_file_edits` are one fact, declared twice.

        Both must agree AND the branch must actually render a command: an adapter
        that declares the profile but falls through to the base raise would be
        chosen as a write runner and fail on a user's repository, and one that
        implements it without declaring it would be refused for work it can do.
        """
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import PROFILE_AGENT_WRITE, ProfileUnsupported

        provider = _PROVIDERS[name]
        declares = PROFILE_AGENT_WRITE in provider.agent_profiles
        assert declares == provider.supports_file_edits, (
            f"{name}: agent_profiles declares agent-write={declares} but "
            f"supports_file_edits={provider.supports_file_edits}"
        )
        if declares:
            argv = provider.build_agent_command("p", "m", PROFILE_AGENT_WRITE)
            assert argv and argv[-1] == "p"
            # A non-argv transport is mandatory for the edit-capable invocation
            # (task 6 / D2d): apply and implement prompts are the largest the
            # skill builds.
            stdin_argv, stdin_text = provider.build_agent_command_for_transport(
                "p", "m", PROFILE_AGENT_WRITE, "stdin"
            )
            assert stdin_text == "p" and "p" not in stdin_argv
        else:
            with pytest.raises(ProfileUnsupported):
                provider.build_agent_command("p", "m", PROFILE_AGENT_WRITE)

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_declared_usage_matches_an_implemented_extractor(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        implements = type(provider).parse_usage is not LLMProvider.parse_usage
        assert implements == (provider.usage_support != "none"), (
            f"{name}: parse_usage implemented={implements} but "
            f"usage_support={provider.usage_support!r}"
        )
        if not implements:
            assert provider.parse_usage("anything at all", "") is None

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_a_runner_declaring_a_profile_also_declares_its_version_data(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        if "agent-readonly" not in provider.agent_profiles:
            return
        assert provider.version_cmd, f"{name}: no version_cmd"
        assert provider.min_version, f"{name}: no min_version"
        assert provider.verified_version, f"{name}: no verified_version"


CAPABILITIES_PATH = Path(__file__).parent.parent / "hosts" / "capabilities.yaml"


def _declared_host_runner_cells():
    """Every `runners:` cell in `hosts/capabilities.yaml`, as the YAML declares it.

    Read off the document rather than written out as a roster of hosts and
    runners, so a host or a runner cell added later is swept on the day it is
    added rather than on the day someone remembers to extend a list here.
    """
    import yaml

    document = yaml.safe_load(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    return sorted(
        (host_id, runner, cell.get("usage"))
        for host_id, host in (document.get("hosts") or {}).items()
        for runner, cell in (host.get("runners") or {}).items()
    )


HOST_RUNNER_USAGE_CELLS = _declared_host_runner_cells()


class TestHostCapabilityUsageMatchesTheAdapter:
    """W5: the third declaration of one fact, tied to the other two.

    `TestSubagentRunnerDeclarations` above ties an adapter's `usage_support` to
    whether it implements `parse_usage`. Nothing tied either of those to the
    per-host `runners:` cell in `hosts/capabilities.yaml`, and it drifted for
    exactly that reason: the claude-code host declared
    `cursor-agent: {… usage: none}` long after the adapter had moved to
    `USAGE_PARTIAL`.

    Nothing reads the YAML value at run time today, which made the drift silent
    rather than harmless — the cell is rendered into
    `references/host-capabilities.md`, whose per-runner `usage=` line is what a
    human reads before choosing a runner. (`docs/hosts.md` renders the *default
    runner* column only; the repo README carries neither.)
    """

    @pytest.mark.parametrize(
        "host_id,runner,declared",
        HOST_RUNNER_USAGE_CELLS,
        ids=[f"{host}-{runner}" for host, runner, _ in HOST_RUNNER_USAGE_CELLS],
    )
    def test_the_cell_matches_the_named_adapters_declaration(
        self, host_id, runner, declared
    ):
        from utils.provider_registry import _PROVIDERS

        assert runner in _PROVIDERS, (
            f"{host_id}: `runners:` names {runner!r}, which is not an adapter in "
            f"the registry"
        )
        assert declared == _PROVIDERS[runner].usage_support, (
            f"{host_id}/{runner}: capabilities.yaml declares usage={declared!r} "
            f"but the adapter declares "
            f"usage_support={_PROVIDERS[runner].usage_support!r}. The YAML is the "
            f"rendered doc; the adapter is the code — move the cell."
        )

    def test_every_host_that_declares_runners_is_in_the_sweep(self):
        """A loader that quietly returned nothing would make the sweep vacuous."""
        import yaml

        document = yaml.safe_load(CAPABILITIES_PATH.read_text(encoding="utf-8"))
        with_runners = {
            host_id for host_id, host in document["hosts"].items()
            if host.get("runners")
        }

        assert HOST_RUNNER_USAGE_CELLS, "no runner cells were collected at all"
        assert with_runners == {host for host, _, _ in HOST_RUNNER_USAGE_CELLS}


class TestDeclaredOutputChannel:
    """`supports_file_output`: can this adapter write the result file at all?

    Two adapters run their CLI in a read-only chat mode on purpose, and the
    review prompts used to demand a file write from them regardless. Measured
    2026-08-18 on `cursor-agent:gpt-5.6-sol-xhigh`, three identical runs: the
    envelope came back `{"type":"result","subtype":"success","is_error":false,…}`
    with `result` = "I can't write the requested file while Ask mode is active.
    Switch to Agent mode and resend the request." — no file, no JSON, and the
    slot died on "JSON parsing failed from both file and stdout".
    """

    #: Pinned like the `agent-write` roster is pinned, and for the mirror-image
    #: reason: five of the seven are here because of a *mode the adapter chose*, so
    #: an adapter joining or leaving it is a deliberate change to how that CLI is
    #: invoked, never a passing thought. Those five name the argv atom that causes
    #: it, and `TestReadOnlyArgvMatchesTheDeclaration` in
    #: `test_provider_housekeeping.py` checks that atom against the argv the
    #: adapter really renders — this set is the *what*, that test is the *why*.
    #:
    #:   aider       `/ask <prompt>`         read-only chat mode
    #:   copilot     no `--allow-all-tools`  write requested, then denied
    #:   cursor-agent `--mode ask`           write refusal (measured 2026-08-18)
    #:   qwen-code   `--max-tool-calls 0`    zero tool budget, exit 55, no stdout
    #:   vibe        `--agent plan`          write_file/edit permission `never`
    #:
    #: THE SIXTH HAS NO ATOM, and that is why it went unnoticed longest:
    #:
    #:   gemini      (no atom — a DEFAULT)   ships no `write_file` tool at all
    #:
    #: Measured 2026-08-19 (harness-expansion.md §16.3): under this adapter's argv
    #: gemini-cli advertises nine tools and none of them writes. Handed a
    #: `write_file` call it answers `Tool "write_file" not found in registry`,
    #: writes nothing, and **exits 0 printing the model's own "JSON written to …"
    #: confirmation as its answer** — the one member that reports success for a
    #: write that never happened, where the other five fail loudly. An argv guard
    #: cannot see a default, so its adjudication lives in that same file's third
    #: table, `NO_WRITE_TOOL_AT_ALL`, which feeds the same derived-roster
    #: assertion.
    #:
    #: THE SEVENTH IS NEITHER A MODE NOR A MISSING TOOL (D6, 2026-09-11):
    #:
    #:   codex       `--sandbox workspace-write`  writes only through bubblewrap
    #:
    #: codex CAN write a file — wherever bubblewrap can start. On a stock Ubuntu
    #: 24.04 (`kernel.apparmor_restrict_unprivileged_userns=1`) it cannot, and a
    #: review asked for a file spent seven minutes twice, failed three writes
    #: and returned no JSON, with `--ephemeral` keeping nothing to salvage. The
    #: array was always the final `agent_message`, which `parse_output` reads,
    #: so the fan-out asks for stdout and never bets a slot on the sandbox. Its
    #: adjudication is the `STOPS_FILE_WRITES` row under that atom;
    #: `supports_file_edits` stays True (the agent-write profile really edits,
    #: and the sandbox probe that gates it is D6 step 3).
    CANNOT_WRITE_FILES = {"aider", "codex", "copilot", "cursor-agent", "gemini",
                          "qwen-code", "vibe"}

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_declares_a_bool(self, name):
        from utils.provider_registry import _PROVIDERS

        assert isinstance(_PROVIDERS[name].supports_file_output, bool)

    def test_the_read_only_roster_is_exactly_the_declared_set(self):
        from utils.provider_registry import _PROVIDERS

        declared = {
            name
            for name, provider in _PROVIDERS.items()
            if not provider.supports_file_output
        }
        assert_pinned_subset(
            self.CANNOT_WRITE_FILES,
            declared,
            "TestDeclaredOutputChannel.CANNOT_WRITE_FILES in tests/test_provider_contract.py",
            members="the adapters that declare supports_file_output = False",
            hint="A member's cause is adjudicated in tests/test_provider_housekeeping.py: "
            "a STOPS_FILE_WRITES row for the argv atom that stops the write, or a "
            "NO_WRITE_TOOL_AT_ALL row naming the probe when the CLI ships no write tool.",
        )

    def test_cursor_declares_it_cannot_write_files(self):
        from utils.provider_registry import get_provider

        assert get_provider("cursor-agent").supports_file_output is False

    def test_cursors_declaration_matches_the_argv_that_causes_it(self):
        """The declaration is only true while the write refusal is in the argv.

        `--mode ask` is what refuses the write, and `-f`/`--yolo` is the bypass
        this adapter exists to stop passing (§3.1). Restoring either — "fixing"
        the file write by handing an untrusted prompt write access again — must
        fail here rather than quietly undo the safety decision.

        §5.4.1's per-flag gate is stubbed exactly as in
        `TestCursorAgentProvider.test_cursor_agent_build_command`, so this reads
        the argv rather than the developer's installed cursor build.
        """
        from utils.provider_registry import get_provider

        with patch.object(
            version_gate,
            "read_help_tokens",
            lambda _b, **_kw: frozenset({"--trust", "--mode"}),
        ), patch.object(
            cursor_agent_module.shutil, "which", lambda _n: sys.executable
        ):
            cmd = get_provider("cursor-agent").build_command("prompt", "auto")

        assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "ask"
        assert "--trust" in cmd
        assert "-f" not in cmd and "--force" not in cmd and "--yolo" not in cmd

    def test_a_file_capable_adapter_is_still_the_default(self):
        from utils.provider_registry import get_provider

        assert get_provider("claude-code").supports_file_output is True
        assert get_provider("opencode").supports_file_output is True

    def test_codex_moves_the_review_off_its_sandbox_and_keeps_its_edits(self):
        """D6 step 1: the review channel no longer depends on bubblewrap, and
        nothing else about the adapter moves — the agent-write profile still
        edits, and the `--output-schema` sidecar is still a path."""
        from utils.provider_registry import get_provider

        codex = get_provider("codex")

        assert codex.supports_file_output is False
        assert codex.supports_file_edits is True
        assert codex.structured_output_transport == "file"

    @pytest.mark.parametrize("name", sorted(CANNOT_WRITE_FILES))
    def test_a_read_only_adapter_is_asked_for_stdout(self, name):
        """The declaration is what the prompt is rendered from — not a name
        spelled out in an orchestrator."""
        from utils.prompt_channels import STDOUT_OUTPUT_LOCATION, render_output_location

        block = render_output_location("/out/result.json", f"{name}:some-model")

        assert block == STDOUT_OUTPUT_LOCATION
        assert "/out/result.json" not in block


class TestDeclaredPlanRoute:
    """`wants_inline_plan`: does the ask lane inline the plan's body into this
    adapter's prompt (plans/ask-answer-recovery-and-false-success.md D4.2)?

    Measured 2026-09-26: aider already received the plan by `--read`, so the
    inlined copy only doubled its prompt and fed its mention scan, which
    auto-added every repo path the plan named — a 1.08M-token request.
    """

    #: Pinned like `CANNOT_WRITE_FILES`, and each member names the route that
    #: replaces the inline copy: an adapter declaring `False` without one would
    #: send its model no plan at all. `TestPathOnlyAskPromptHasARoute` in
    #: `test_provider_housekeeping.py` checks the route against the argv the
    #: adapter really renders — this map is the *what*, that test is the *why*.
    #:
    #:   aider   `--read <plan_path>`   every absolute path the prompt names is
    #:                                  attached read-only (`_extract_read_files`)
    PATH_ONLY_PLAN_ROUTES = {"aider": "--read"}

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_declares_a_bool(self, name):
        from utils.provider_registry import _PROVIDERS

        assert isinstance(_PROVIDERS[name].wants_inline_plan, bool)

    def test_the_path_only_roster_is_exactly_the_declared_set(self):
        from utils.provider_registry import _PROVIDERS

        declared = {
            name
            for name, provider in _PROVIDERS.items()
            if not provider.wants_inline_plan
        }
        assert declared == set(self.PATH_ONLY_PLAN_ROUTES)

    def test_inlining_is_still_the_default(self):
        from utils.provider_registry import get_provider

        assert base_module.LLMProvider.wants_inline_plan is True
        assert get_provider("claude-code").wants_inline_plan is True


class TestDeclaredGenerateShellAndReadScope:
    """`runs_shell_in_generate` and `generate_read_scope`: what a fan-out slot
    can do to reach the change (plan: review-run-scope-salvage-and-failure-
    visibility D2.3).

    Measured on the 2026-09-25 review run: the code-review prompt told every
    lane to run `git diff`, and cursor's `--mode ask` lanes have no shell — one
    returned `[]` without ever seeing a diff. `render_diff_access` now renders
    the prompt's `{diff_access}` block from these two declarations, so a wrong
    value tells a lane it has a shell it lacks, or withholds one it has. Each
    value is pinned to its source here; `tests/test_provider_housekeeping.py`'s
    `TestShellDeclarationMatchesTheEvidence` checks the shell half against the
    argv or the named measurement behind it.
    """

    #: D2.3's table. Every adapter not in it is `None` (not measured), and a
    #: value leaves `None` only with a measured source: a log, a probe or argv.
    #:
    #:   pi           True   the incident's `bash` call running pytest (§3.1)
    #:   omp          True   its argv: `--tools read,grep,glob,bash,edit,write`
    #:   codex        True   §7.3's audited stream ran `pwd && rg --files`
    #:   cursor-agent False  ask mode refused a shell in the kimi and grok logs
    #:   claude-code  False  `acceptEdits` denies headless Bash (3 denials)
    #:   qwen-code    False  `--max-tool-calls 0` aborts before any tool runs
    #:   gemini       False  no `run_shell_command` under this argv (0.25.2)
    #:   fx           True   production 0.0.14 --auto probe; shell-reviewed fixtures
    SHELL_IN_GENERATE = {
        "pi": True,
        "omp": True,
        "codex": True,
        "cursor-agent": False,
        "claude-code": False,
        "qwen-code": False,
        "gemini": False,
        "fx": True,
    }
    #: The adapters whose read scope is measured away from the `"cwd"` default.
    #: cursor-agent declares `"cwd"` explicitly (measured, and equal to the
    #: default), so it is not a departure. aider's is read from its argv:
    #: `/ask` has no file tool, and the stdin decline supply refuses the
    #: mention adds that once brought review-code the changed files.
    READ_SCOPE_DEPARTURES = {"qwen-code": "none", "aider": "none"}

    def test_the_base_defaults_assert_nothing(self):
        assert LLMProvider.runs_shell_in_generate is None
        assert LLMProvider.generate_read_scope == "cwd"

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_declares_true_false_or_none(self, name):
        from utils.provider_registry import _PROVIDERS

        value = _PROVIDERS[name].runs_shell_in_generate
        # `is` and `isinstance(bool)`: a truthy 1 or "yes" is not a declaration.
        assert value is None or isinstance(value, bool), (name, value)

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_read_scope_is_one_of_the_three(self, name):
        from utils.prompt_channels import GENERATE_READ_SCOPES
        from utils.provider_registry import _PROVIDERS

        assert GENERATE_READ_SCOPES == ("none", "cwd", "any")
        assert _PROVIDERS[name].generate_read_scope in GENERATE_READ_SCOPES, name

    def test_the_shell_values_are_the_measured_table(self):
        from utils.provider_registry import _PROVIDERS

        declared = {
            name: provider.runs_shell_in_generate
            for name, provider in _PROVIDERS.items()
            if provider.runs_shell_in_generate is not None
        }
        assert declared == self.SHELL_IN_GENERATE

    @pytest.mark.parametrize("name", sorted(_registered_provider_names()))
    def test_every_adapter_outside_the_table_is_unmeasured(self, name):
        from utils.provider_registry import _PROVIDERS

        if name in self.SHELL_IN_GENERATE:
            assert _PROVIDERS[name].runs_shell_in_generate is self.SHELL_IN_GENERATE[name]
        else:
            assert _PROVIDERS[name].runs_shell_in_generate is None, (
                f"{name} declares runs_shell_in_generate="
                f"{_PROVIDERS[name].runs_shell_in_generate!r} with no row in "
                f"D2.3's table — a value leaves None only with a measured source"
            )

    def test_cursor_reads_under_cwd_and_qwen_reads_nothing(self):
        from utils.provider_registry import get_provider

        assert get_provider("cursor-agent").generate_read_scope == "cwd"
        assert get_provider("qwen-code").generate_read_scope == "none"

    def test_every_other_adapter_keeps_the_cwd_default(self):
        from utils.provider_registry import _PROVIDERS

        departures = {
            name: provider.generate_read_scope
            for name, provider in _PROVIDERS.items()
            if provider.generate_read_scope != "cwd"
        }
        assert departures == self.READ_SCOPE_DEPARTURES

    def test_the_two_declarations_are_not_the_agent_readonly_ones(self):
        """codex reads anywhere under `agent-readonly` (`reads_outside_cwd`),
        but nothing measured that for `generate`, so its fan-out slot keeps the
        `"cwd"` default: one profile's measurement never stands in for
        another's."""
        from utils.provider_registry import get_provider

        codex = get_provider("codex")
        assert codex.reads_outside_cwd is True
        assert codex.generate_read_scope == "cwd"


class TestProviderBinaryNotFound:
    """Tests for provider availability when binaries are not found."""

    @patch("shutil.which", return_value=None)
    def test_provider_binary_not_found(self, mock_which):
        """Verify is_available() returns False when binary not in PATH.

        This tests all providers to ensure they properly report unavailability
        when their respective CLI tools are not installed.
        """
        providers = [
            ClaudeCodeProvider(),
            CursorAgentProvider(),
            GeminiProvider(),
            KiloCodeProvider(),
            OpenCodeProvider(),
        ]

        for provider in providers:
            assert provider.is_available() is False, (
                f"{provider.name} should report unavailable when binary not found"
            )

    @patch("shutil.which")
    def test_multiple_providers_availability(self, mock_which):
        """Test mixed availability across providers."""
        # Simulate: cursor-agent available, gemini not, opencode available
        def which_side_effect(binary):
            available = {"cursor-agent": "/usr/bin/cursor-agent", "opencode": "/usr/bin/opencode"}
            return available.get(binary)

        mock_which.side_effect = which_side_effect

        cursor = CursorAgentProvider()
        gemini = GeminiProvider()
        opencode = OpenCodeProvider()

        assert cursor.is_available() is True
        assert gemini.is_available() is False
        assert opencode.is_available() is True


class TestProviderEdgeCases:
    """Edge case tests for all providers."""

    def test_cursor_agent_whitespace_handling(self):
        """Test cursor-agent handles whitespace in output."""
        provider = CursorAgentProvider()
        wrapper = {"type": "result", "result": "  \n  [1, 2, 3]  \n  "}
        stdout = f"  \n  {json.dumps(wrapper)}  \n  "

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_gemini_nested_json_in_response(self):
        """Test Gemini handles deeply nested JSON."""
        provider = GeminiProvider()
        nested_data = {
            "level1": {
                "level2": {
                    "level3": [{"deep": True}]
                }
            }
        }
        wrapper = {"session_id": "s1", "response": json.dumps(nested_data), "stats": {}}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == nested_data

    def test_opencode_large_text_concatenation(self):
        """Test OpenCode handles many text events."""
        provider = OpenCodeProvider()
        # Simulate chunked JSON output across many events
        json_str = json.dumps([{"i": i} for i in range(100)])
        chunk_size = 50
        chunks = [json_str[i:i+chunk_size] for i in range(0, len(json_str), chunk_size)]

        events = [{"type": "text", "part": {"type": "text", "text": chunk}} for chunk in chunks]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert len(result["data"]) == 100

    def test_cursor_agent_handles_wrapper_without_type(self):
        """Test cursor-agent handles JSON wrapper missing 'type' field."""
        provider = CursorAgentProvider()
        # Direct JSON object without "type":"result" wrapper
        data = {"result": [1, 2, 3]}
        stdout = json.dumps(data)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        # Should parse the inner result
        assert result["data"] == [1, 2, 3]

    def test_gemini_handles_response_as_dict(self):
        """Test Gemini when response field is already a dict (not string)."""
        provider = GeminiProvider()
        # If response is somehow already parsed (edge case)
        wrapper = {"session_id": "s1", "response": {"key": "value"}, "stats": {}}
        stdout = json.dumps(wrapper)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == {"key": "value"}

    def test_opencode_handles_non_text_part_type(self):
        """Test OpenCode ignores events with non-text part type."""
        provider = OpenCodeProvider()
        events = [
            {"type": "text", "part": {"type": "image", "data": "base64..."}},
            {"type": "text", "part": {"type": "text", "text": "[1, 2]"}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        result = provider.parse_output(stdout, "")

        assert result["success"] is True
        assert result["data"] == [1, 2]


class TestProviderInvocationErrors:
    """Tests for invoke_with_provider() error handling.

    These tests verify the critical orchestration behavior including timeouts,
    subprocess failures, and structured error output by mocking subprocess.run.
    """

    @pytest.fixture
    def mock_subprocess(self):
        """Create a mock for subprocess.run."""
        with patch("subprocess.run") as mock:
            yield mock

    @pytest.fixture
    def mock_provider_available(self):
        """Mock shutil.which to make providers appear available."""
        with patch("shutil.which", return_value="/usr/bin/mock-binary"):
            yield

    def test_timeout_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Test that subprocess.TimeoutExpired returns TIMEOUT error code.

        When a provider command times out, invoke_with_provider should return
        a structured error response with error_code='TIMEOUT'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        # Simulate timeout - TimeoutExpired requires cmd and timeout args,
        # stdout/stderr are set as attributes
        exc = subprocess.TimeoutExpired(
            cmd=["cursor-agent", "--print", "test"],
            timeout=600
        )
        exc.stdout = "partial output"
        exc.stderr = "timeout stderr"
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
            timeout=600
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert "timed out" in result["error"].lower()
        assert result["details"]["timeout"] == 600
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"

    def test_timeout_with_none_output(self, mock_subprocess, mock_provider_available):
        """Test timeout handling when stdout/stderr are None."""
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        exc = subprocess.TimeoutExpired(
            cmd=["gemini", "test"],
            timeout=900
        )
        exc.stdout = None
        exc.stderr = None
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="gemini:gemini-pro",
            timeout=900
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT
        assert result["details"]["provider"] == "gemini"

    def test_timeout_with_bytes_output(self, mock_subprocess, mock_provider_available):
        """Test timeout handling when stdout/stderr are bytes."""
        from utils.llm_client import invoke_with_provider, ERROR_TIMEOUT
        import subprocess

        exc = subprocess.TimeoutExpired(
            cmd=["opencode", "run"],
            timeout=600
        )
        exc.stdout = b"bytes output"
        exc.stderr = b"bytes error"
        mock_subprocess.side_effect = exc

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="opencode:claude-3"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_TIMEOUT

    def test_subprocess_failed_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Test that non-zero exit code returns SUBPROCESS_FAILED error code.

        When a provider command fails with a non-zero exit code, invoke_with_provider
        should return a structured error response with error_code='SUBPROCESS_FAILED'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_SUBPROCESS_FAILED
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"
        mock_result.stderr = "command failed: invalid arguments"
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_SUBPROCESS_FAILED
        assert "exited with code 1" in result["error"]
        assert result["details"]["exit_code"] == 1
        assert result["details"]["stderr"] == "command failed: invalid arguments"
        assert result["details"]["provider"] == "cursor-agent"

    def test_subprocess_failed_various_exit_codes(self, mock_subprocess, mock_provider_available):
        """Test SUBPROCESS_FAILED with various exit codes."""
        from utils.llm_client import invoke_with_provider, ERROR_SUBPROCESS_FAILED
        from unittest.mock import MagicMock

        for exit_code in [1, 2, 127, 255]:
            mock_result = MagicMock()
            mock_result.returncode = exit_code
            mock_result.stdout = ""
            mock_result.stderr = f"exit code {exit_code}"
            mock_subprocess.return_value = mock_result

            result = invoke_with_provider(
                prompt="Test prompt",
                model_spec="gemini:gemini-pro"
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_SUBPROCESS_FAILED
            assert result["details"]["exit_code"] == exit_code

    def test_binary_not_found_returns_error_code(self):
        """Test that unavailable binary returns BINARY_NOT_FOUND error code.

        When a provider's CLI tool is not found in PATH, invoke_with_provider
        should return a structured error response with error_code='BINARY_NOT_FOUND'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_BINARY_NOT_FOUND

        # Mock shutil.which to return None (binary not found)
        with patch("shutil.which", return_value=None):
            result = invoke_with_provider(
                prompt="Test prompt",
                model_spec="cursor-agent:gpt-4"
            )

            assert result["success"] is False
            assert result["error_code"] == ERROR_BINARY_NOT_FOUND
            assert "not found" in result["error"].lower()
            assert result["details"]["provider"] == "cursor-agent"
            assert result["details"]["model"] == "gpt-4"

    def test_unknown_provider_returns_binary_not_found(self):
        """Test that unknown provider returns BINARY_NOT_FOUND error code."""
        from utils.llm_client import invoke_with_provider, ERROR_BINARY_NOT_FOUND

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="nonexistent-provider:model"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_BINARY_NOT_FOUND
        assert "nonexistent-provider" in result["error"]

    def test_parse_error_returns_error_code(self, mock_subprocess, mock_provider_available):
        """Test that parse failures return PARSE_ERROR error code.

        When output cannot be parsed as valid JSON, invoke_with_provider
        should return a structured error response with error_code='PARSE_ERROR'.
        """
        from utils.llm_client import invoke_with_provider, ERROR_PARSE_ERROR
        from unittest.mock import MagicMock

        # Return valid process result but unparseable output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is not JSON at all, just plain text without any brackets"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is False
        assert result["error_code"] == ERROR_PARSE_ERROR
        assert "error" in result

    def test_successful_invocation(self, mock_subprocess, mock_provider_available):
        """Test successful invocation returns data and details."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        # Return valid cursor-agent JSON wrapper format
        inner_data = [{"title": "Test", "importance": "high"}]
        wrapper = {"type": "result", "result": json.dumps(inner_data)}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(wrapper)
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is True
        assert result["data"] == inner_data
        assert result["details"]["provider"] == "cursor-agent"
        assert result["details"]["model"] == "gpt-4"
        assert "duration_seconds" in result["details"]

    def test_invocation_uses_provider_timeout(self, mock_subprocess, mock_provider_available):
        """Test that invocation uses provider's default timeout when not specified."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
            # No timeout specified - should use provider default
        )

        # Verify subprocess.run was called with correct timeout
        mock_subprocess.assert_called_once()
        call_kwargs = mock_subprocess.call_args
        # cursor-agent default timeout is 1200
        assert call_kwargs.kwargs["timeout"] == 1200

    def test_invocation_uses_custom_timeout(self, mock_subprocess, mock_provider_available):
        """Test that invocation uses custom timeout when specified."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4",
            timeout=300  # Custom timeout
        )

        mock_subprocess.assert_called_once()
        call_kwargs = mock_subprocess.call_args
        assert call_kwargs.kwargs["timeout"] == 300

    def test_duration_tracking(self, mock_subprocess, mock_provider_available):
        """Test that invocation tracks duration in details."""
        from utils.llm_client import invoke_with_provider
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"type": "result", "result": "[]"})
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = invoke_with_provider(
            prompt="Test prompt",
            model_spec="cursor-agent:gpt-4"
        )

        assert result["success"] is True
        assert "duration_seconds" in result["details"]
        assert isinstance(result["details"]["duration_seconds"], float)
        assert result["details"]["duration_seconds"] >= 0
