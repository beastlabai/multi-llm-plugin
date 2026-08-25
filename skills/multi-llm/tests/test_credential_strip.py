"""M3 (donor §2.1) — the credential strip that lets subscription auth win.

The defect this closes was measured on the wire at claude-code 2.1.233: with a
subscription token present, an exported `ANTHROPIC_API_KEY` swaps the request's
`authorization: Bearer …` for `x-api-key: …` and **drops the subscription beta
header entirely**, so the run is billed per token and the subscription is not
used at all. Omnigent calls it "a cost rule, not a security one".

M3 is **three edits, not one**, and this module is the guard on all three:

* the adapter's `get_remove_env()` — the strip itself, gated on a *sanitized*
  `claude auth status` preflight and its decision table;
* `hosts/capabilities.yaml`'s `env_guards.passthrough` — which carried
  `ANTHROPIC_API_KEY` and, because the executor re-admits every passthrough name
  last, silently cancelled the strip on the agent-profile lane while leaving it
  in force on the fan-out lane;
* the re-render of the generated `references/host-capabilities.md`, which
  documented the opposite policy.

M3 also ships the single shared env builder both lanes go through
(`base.build_child_env`), which is what makes **Lane A ≡ Lane B** a property of
the code rather than of two call sites kept in sync by hand — and is the
construction dependency M17 lands inside (atomic group **A6**).

Everything here is hermetic: no test spawns the real `claude`, and the pinned
name lists are re-written in this file rather than imported into the assertion,
so silently editing the adapter's tuples reds the tree instead of passing
vacuously.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import base  # noqa: E402
from utils.providers import claude_code as cc  # noqa: E402
from utils.providers.claude_code import ClaudeCodeProvider  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402

SKILL_DIR = Path(__file__).parent.parent

#: A path `shutil.which` can plausibly have returned. Only its truthiness is
#: read, so it carries no platform assumption.
_RESOLVED = "/usr/bin/claude"


def _registry_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one.

    Resolved at class-definition time because `@pytest.mark.parametrize`
    evaluates its argument while the class body runs — the same constraint that
    forces `test_providers.py`'s local twin at its own class bodies.

    **No test in this module may name an adapter or a count** (M0.a).
    """
    return sorted(_PROVIDERS)


@pytest.fixture(autouse=True)
def _hermetic_gate():
    """No test inherits another's verdict cache, warn-once state — or a real CLI.

    `shutil.which` is stubbed to "not installed" for the whole module, so a test
    that forgets to patch it declines on the missing-binary row instead of
    spawning the developer's own `claude` and reading their real login. A test
    that needs the probe re-patches it inside its own `with`.
    """
    cc.reset_subscription_verdict_cache()
    base.reset_env_policy_warnings()
    with patch.object(cc.shutil, "which", return_value=None):
        yield
    cc.reset_subscription_verdict_cache()
    base.reset_env_policy_warnings()


def _probe(stdout: str, returncode: int = 0):
    """A patched `subprocess.run` returning one canned `claude auth status`."""
    return subprocess.CompletedProcess(
        args=[_RESOLVED, "auth", "status"], returncode=returncode, stdout=stdout, stderr=""
    )


# The four rows below are **verbatim** `claude auth status` output, captured at
# claude-code 2.1.237 on 2026-08-20 (zero network, zero paid tokens). They are
# transcribed rather than invented because the whole point of the sanitized gate
# is that the *unsanitized* shape looks the same as the sanitized one, and a
# hand-written fixture would have hidden the difference the plan was written to
# catch. §2.1 was written against `authMethod: "oauth_token"` at 2.1.233; at
# 2.1.237 that is the ANTHROPIC_AUTH_TOKEN spelling and a real subscription says
# `claude.ai` — see `OAUTH_AUTH_METHODS`' table.
SUBSCRIPTION_JSON = (
    '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty",'
    '"email":"dev@example.com","orgId":"org","orgName":"org",'
    '"subscriptionType":"max"}'
)
#: The same login with `ANTHROPIC_API_KEY` exported — `apiKeySource` appears and
#: `subscriptionType` goes null, while `loggedIn` stays true. This row IS the
#: ambiguity: an unsanitized `loggedIn: true` describes an API-key run.
OVERRIDDEN_JSON = (
    '{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty",'
    '"apiKeySource":"ANTHROPIC_API_KEY","email":null,"orgId":null,'
    '"orgName":null,"subscriptionType":null}'
)
#: `ANTHROPIC_AUTH_TOKEN` exported and nothing else.
ENV_TOKEN_JSON = '{"loggedIn":true,"authMethod":"oauth_token","apiProvider":"firstParty"}'
LOGGED_OUT_JSON = '{"loggedIn":false}'


# ---------------------------------------------------------------------------
# Ship gate 1 — the pinned ten-name constant
# ---------------------------------------------------------------------------


class TestThePinnedNames:
    """§13.2's "10 names, not 1", re-stated here so the adapter cannot drift.

    The lists are pinned to a bundle version. They are **transcribed**, never
    read: the bundle groups these families under minified symbols that are
    re-assigned on every release, so nothing addresses them at run time or at
    build time. A vendor upgrade that introduces an eleventh name is caught by
    the separate, advisory, CI-only drift check — not by this test, and never by
    the dispatch path.
    """

    #: Re-written by hand on purpose (assertion (a)): importing the adapter's
    #: tuple and comparing it to itself would pass however it was edited.
    CREDENTIAL = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS")
    VENDOR_BACKEND = (
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_GOOGLE_CLOUD",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_GATEWAY",
    )

    def test_the_credential_names_are_exactly_the_three(self):
        assert cc._CREDENTIAL_ENV == self.CREDENTIAL

    def test_the_vendor_backend_names_are_exactly_the_seven(self):
        assert cc._VENDOR_BACKEND_ENV == self.VENDOR_BACKEND

    def test_the_two_lists_together_are_the_ten(self):
        """Three credential names plus seven backend names, and no overlap."""
        assert not set(self.CREDENTIAL) & set(self.VENDOR_BACKEND)
        assert len(set(self.CREDENTIAL) | set(self.VENDOR_BACKEND)) == 10

    def test_they_are_two_constants_and_not_one(self):
        """The two gates cannot be conflated by a later edit.

        They are gated on different things — the credential names on the
        sanitized probe, the backend names on *declaration* — so a single merged
        tuple would silently put a user's configured Bedrock route behind a
        billing probe.
        """
        assert cc._CREDENTIAL_ENV is not cc._VENDOR_BACKEND_ENV

    def test_the_backend_list_is_the_declaration_validation_domain(self):
        """Assertion (b): the declared set and the strip set cannot drift apart.

        `declared_env_names` takes the domain as an argument and the adapter
        hands it `_VENDOR_BACKEND_ENV`, so a name outside it can never become
        "declared" and thereby exempt itself from a strip.
        """
        with patch.dict(
            os.environ, {"CLAUDE_CODE_USE_BEDROCK": "1"}, clear=False
        ), patch.object(
            base,
            "_provider_block",
            return_value={"env": {"CLAUDE_CODE_USE_BEDROCK": "1", "NOT_A_BACKEND": "x"}},
        ):
            declared = base.declared_env_names(
                "claude-code", cc._VENDOR_BACKEND_ENV
            )
        assert declared == ("CLAUDE_CODE_USE_BEDROCK",)

    def test_the_never_stripped_names_are_named(self):
        """The three the plan settles by name, so a later edit has to argue.

        `OPENAI_API_KEY` is measured not consulted by claude-code at all and
        stripping it breaks configs declaring `env_key = "OPENAI_API_KEY"`;
        `ANTHROPIC_BASE_URL` is a corporate gateway's only route to a model;
        `CLAUDE_CONFIG_DIR` is where the subscription credential lives.
        """
        for name in ("OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CONFIG_DIR"):
            assert name in cc.NEVER_STRIPPED
        assert set(cc.NEVER_STRIPPED).isdisjoint(cc._CREDENTIAL_ENV)
        assert set(cc.NEVER_STRIPPED).isdisjoint(cc._VENDOR_BACKEND_ENV)


# ---------------------------------------------------------------------------
# Ship gate 2 — the decision table, and what `get_remove_env` returns
# ---------------------------------------------------------------------------


class TestTheDecisionTable:
    """§2.1's table, row by row, with no CLI spawned.

    The rows exist because `apiKeySource` **cannot** be the gate: it names the
    credential that currently wins and reports the same value for the two
    populations that matter — *subscription overridden by env* (the money bug,
    strip is correct) and *API-key-only* (strip yields "Not logged in" in 0.35 s
    and breaks every slot). The gate is the sanitized probe plus this table.
    """

    @pytest.mark.parametrize(
        "returncode,stdout,expected,why",
        [
            (0, SUBSCRIPTION_JSON, cc.STRIP_VERDICT_STRIP, "a subscription survives"),
            (0, LOGGED_OUT_JSON, cc.STRIP_VERDICT_DECLINE, "API-key-only user"),
            (0, OVERRIDDEN_JSON, cc.STRIP_VERDICT_DECLINE, "a key still reaches the CLI"),
            (
                0,
                '{"loggedIn":true,"authMethod":"api_key"}',
                cc.STRIP_VERDICT_DECLINE,
                "key-shaped identity",
            ),
            (
                0,
                '{"loggedIn":true,"authMethod":"claude.ai","subscriptionType":null}',
                cc.STRIP_VERDICT_DECLINE,
                "logged in with no subscription behind it",
            ),
            (1, SUBSCRIPTION_JSON, cc.STRIP_VERDICT_DECLINE, "non-zero exit"),
            (0, "not json at all", cc.STRIP_VERDICT_DECLINE, "undecodable"),
            (0, "", cc.STRIP_VERDICT_DECLINE, "empty output"),
        ],
    )
    def test_each_row(self, returncode, stdout, expected, why):
        assert cc._verdict_from_probe(returncode, stdout)[0] == expected, why

    def test_both_measured_subscription_spellings_are_accepted(self):
        """2.1.237 says `claude.ai`; 2.1.233 (donor §2.1) said `oauth_token`.

        Dropping either is a silent regression in one direction: without
        `claude.ai` the gate never fires on a current CLI, and without
        `oauth_token` it never fires on the version the plan measured. Both are
        only safe *because* the probe is sanitized — the env-token that produces
        `oauth_token` at 2.1.237 has been removed before the probe runs.
        """
        assert cc._verdict_from_probe(0, ENV_TOKEN_JSON)[0] == cc.STRIP_VERDICT_STRIP
        assert cc._verdict_from_probe(0, SUBSCRIPTION_JSON)[0] == cc.STRIP_VERDICT_STRIP

    def test_the_unsanitized_pair_is_diagnostic_only(self):
        """`loggedIn: true` is true in the money-bug row AND the API-key row.

        The two differ only in `apiKeySource`/`subscriptionType`, which is why
        the probe has to be run against the *post-strip* env rather than read for
        "am I logged in".
        """
        import json as _json

        overridden = _json.loads(OVERRIDDEN_JSON)
        clean = _json.loads(SUBSCRIPTION_JSON)
        assert overridden["loggedIn"] == clean["loggedIn"] is True
        assert overridden["authMethod"] == clean["authMethod"]
        assert cc._verdict_from_probe(0, OVERRIDDEN_JSON)[0] == cc.STRIP_VERDICT_DECLINE

    def test_json_embedded_in_a_banner_still_decodes(self):
        """A CLI that banners before its JSON is not an "unknown schema"."""
        verdict, _ = cc._verdict_from_probe(0, "welcome\n" + SUBSCRIPTION_JSON + "\n")
        assert verdict == cc.STRIP_VERDICT_STRIP

    def test_a_missing_binary_declines_without_spawning(self):
        with patch.object(cc.shutil, "which", return_value=None), patch.object(
            cc.subprocess, "run"
        ) as run:
            verdict, reason = cc._subscription_verdict(cc._CREDENTIAL_ENV)
        assert verdict == cc.STRIP_VERDICT_DECLINE
        assert "PATH" in reason
        run.assert_not_called()

    def test_a_gateway_declines_before_any_probe(self):
        """§2.1's Q9, answered: `ANTHROPIC_BASE_URL` present ⇒ decline.

        When the org pays through a gateway the subscription-vs-API-key billing
        question does not apply either way, and stripping would produce a run
        that authenticates against the public API with credentials meant for the
        gateway. It is also what keeps `docs/providers.md`'s endpoint recipes
        working.
        """
        with patch.dict(
            os.environ, {"ANTHROPIC_BASE_URL": "https://gateway.example.com"}, clear=False
        ), patch.object(cc.subprocess, "run") as run:
            verdict, reason = cc._subscription_verdict(cc._CREDENTIAL_ENV)
        assert verdict == cc.STRIP_VERDICT_DECLINE
        assert "ANTHROPIC_BASE_URL" in reason
        run.assert_not_called()


class TestWhatGetRemoveEnvReturns:
    """Assertions (c) and (d) of §2.1's pinned-constants test."""

    ALL_TEN = {name: "x" for name in (*cc._CREDENTIAL_ENV, *cc._VENDOR_BACKEND_ENV)}

    def test_claudecode_is_in_every_outcome(self):
        """(d) — including `strip_vendor_keys: never`.

        That strip is not a policy: without it the child `claude -p` detects the
        parent's `CLAUDECODE=1` and refuses to start, so it is what makes the
        dispatch work at all.
        """
        provider = ClaudeCodeProvider()
        for policy in base.STRIP_VENDOR_KEYS_POLICIES:
            with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
                cc, "strip_vendor_keys_policy", return_value=policy
            ), patch.object(
                cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_STRIP, "")
            ):
                assert "CLAUDECODE" in provider.get_remove_env(), policy

    def test_on_a_strip_verdict_it_is_the_credentials_plus_the_backends(self):
        """(c), strip half."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_STRIP, "ok")
        ):
            removed = set(provider.get_remove_env())
        assert removed == {"CLAUDECODE", *cc._CREDENTIAL_ENV, *cc._VENDOR_BACKEND_ENV}

    def test_on_a_decline_it_is_exactly_the_undeclared_backends(self):
        """(c), decline half — the API-key path is preserved byte-for-byte."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_DECLINE, "no")
        ):
            removed = set(provider.get_remove_env())
        assert removed == {"CLAUDECODE", *cc._VENDOR_BACKEND_ENV}
        assert not removed & set(cc._CREDENTIAL_ENV)

    def test_a_declared_backend_is_never_stripped(self):
        """The configured auth path is the user's own choice, on every lane.

        Stripping a declared `CLAUDE_CODE_USE_*` unconditionally trades a 137 s
        hang for "the user's only route to a model is gone".
        """
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
            cc, "declared_env_names", return_value=("CLAUDE_CODE_USE_BEDROCK",)
        ), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_DECLINE, "no")
        ):
            removed = set(provider.get_remove_env())
        assert "CLAUDE_CODE_USE_BEDROCK" not in removed
        assert "CLAUDE_CODE_USE_VERTEX" in removed

    def test_vendor_backend_keep_survives_strip_vendor_keys_always(self):
        """The two tables are independent of each other."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
            cc, "strip_vendor_keys_policy", return_value=base.STRIP_VENDOR_KEYS_ALWAYS
        ), patch.object(
            cc, "vendor_backend_policy", return_value=base.VENDOR_BACKEND_KEEP
        ):
            removed = set(provider.get_remove_env())
        assert removed == {"CLAUDECODE", *cc._CREDENTIAL_ENV}

    def test_an_undeclared_backend_goes_even_under_strip_vendor_keys_never(self):
        """…and the independence holds in the other direction too."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
            cc, "strip_vendor_keys_policy", return_value=base.STRIP_VENDOR_KEYS_NEVER
        ):
            removed = set(provider.get_remove_env())
        assert removed == {"CLAUDECODE", *cc._VENDOR_BACKEND_ENV}

    def test_only_names_actually_present_are_returned(self):
        """A strip is a statement about *this* environment.

        Popping an absent key is a no-op either way, so this changes no
        behaviour — but it is what lets a run with no vendor credential decide
        nothing and therefore probe nothing.
        """
        provider = ClaudeCodeProvider()
        env = {k: v for k, v in os.environ.items() if k not in self.ALL_TEN}
        with patch.dict(os.environ, env, clear=True), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_STRIP, "ok")
        ):
            assert provider.get_remove_env() == ["CLAUDECODE"]

    def test_always_skips_the_probe_entirely(self):
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.ALL_TEN, clear=False), patch.object(
            cc, "strip_vendor_keys_policy", return_value=base.STRIP_VENDOR_KEYS_ALWAYS
        ), patch.object(cc, "_subscription_verdict") as verdict:
            removed = set(provider.get_remove_env())
        verdict.assert_not_called()
        assert set(cc._CREDENTIAL_ENV) <= removed


# ---------------------------------------------------------------------------
# Ship gate 3 — probe call count and probe timeout
# ---------------------------------------------------------------------------


class TestProbeMechanics:
    """Placement, caching, cost and failure posture (§2.1's probe-mechanics block).

    The rules being pinned: lazily on the first claude-code env build, never at
    import and never per slot; memoised for the process lifetime on the resolved
    binary + `CLAUDE_CONFIG_DIR` + the candidate strip set; `stdin=DEVNULL`,
    captured output, a 5 s timeout, zero network; and **every** failure declines,
    because failing closed on a probe hiccup breaks every claude-code slot for
    API-key-only users.
    """

    KEYED = {"ANTHROPIC_API_KEY": "sk-test"}

    def test_a_forty_slot_run_spawns_exactly_one_probe(self):
        """A 4-model × 10-phase run must not spawn 40 `claude auth status`."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(
            cc.subprocess, "run", return_value=_probe(SUBSCRIPTION_JSON)
        ) as run:
            for _ in range(40):
                provider.get_remove_env()
        assert run.call_count == 1
        assert cc.probe_spawn_count() == 1

    def test_a_run_with_no_vendor_credential_spawns_no_probe_at_all(self):
        """Nothing to decide ⇒ nothing spawned."""
        provider = ClaudeCodeProvider()
        env = {k: v for k, v in os.environ.items() if k not in cc._CREDENTIAL_ENV}
        with patch.dict(os.environ, env, clear=True), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(cc.subprocess, "run") as run:
            for _ in range(10):
                provider.get_remove_env()
        run.assert_not_called()
        assert cc.probe_spawn_count() == 0

    def test_importing_the_adapter_spawns_nothing(self):
        """Never at import: a codex-only run must not probe claude at all."""
        assert cc.probe_spawn_count() == 0

    def test_the_probe_is_contained(self):
        """`stdin=DEVNULL`, captured output, a 5 s timeout — it cannot prompt."""
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(
            cc.subprocess, "run", return_value=_probe(SUBSCRIPTION_JSON)
        ) as run:
            cc._subscription_verdict(cc._CREDENTIAL_ENV)
        kwargs = run.call_args.kwargs
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == cc.AUTH_PROBE_TIMEOUT_S
        assert run.call_args.args[0] == [_RESOLVED, "auth", "status"]

    def test_the_probe_runs_sanitized(self):
        """Rule 1 — the env it is handed is the env the post-strip spawn gets.

        Run with the API-key variables still inherited, `claude auth status`
        reports only *which credential currently wins*; it says nothing about
        whether a usable subscription remains once they are removed. The
        nested-session marker goes too, or a probe run from inside a Claude Code
        session is refused by the same guard a dispatch is.
        """
        with patch.dict(
            os.environ, {**self.KEYED, "CLAUDECODE": "1"}, clear=False
        ), patch.object(cc.shutil, "which", return_value=_RESOLVED), patch.object(
            cc.subprocess, "run", return_value=_probe(SUBSCRIPTION_JSON)
        ) as run:
            cc._subscription_verdict(cc._CREDENTIAL_ENV)
        probe_env = run.call_args.kwargs["env"]
        for name in cc._CREDENTIAL_ENV:
            assert name not in probe_env
        assert "CLAUDECODE" not in probe_env
        # §2.2's global git env, so it cannot hang on a credential helper.
        assert probe_env["GIT_TERMINAL_PROMPT"] == "0"

    def test_a_probe_timeout_declines_and_keeps_the_key(self):
        """The gate fails **open**: an occasional per-token bill beats a dead slot."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(
            cc.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
        ):
            removed = provider.get_remove_env()
        assert "ANTHROPIC_API_KEY" not in removed
        assert removed == ["CLAUDECODE"]

    def test_a_probe_timeout_is_cached_like_any_other_verdict(self):
        """A timeout must not be re-paid once per slot for the rest of the run."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(
            cc.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
        ) as run:
            for _ in range(5):
                provider.get_remove_env()
        assert run.call_count == 1

    def test_an_oserror_declines_rather_than_raising(self):
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(cc.subprocess, "run", side_effect=OSError("boom")):
            assert provider.get_remove_env() == ["CLAUDECODE"]

    def test_the_cache_key_carries_the_candidate_strip_set(self):
        """The verdict is only meaningful for the env it was sanitized against."""
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(
            cc.subprocess, "run", return_value=_probe(SUBSCRIPTION_JSON)
        ) as run:
            cc._subscription_verdict(("ANTHROPIC_API_KEY",))
            cc._subscription_verdict(("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))
        assert run.call_count == 2

    def test_a_decline_warns_once_naming_the_billing_consequence(self, capsys):
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(cc.subprocess, "run", return_value=_probe(LOGGED_OUT_JSON)):
            for _ in range(4):
                provider.get_remove_env()
        err = capsys.readouterr().err
        assert err.count("ANTHROPIC_API_KEY") == 1
        assert "billed per token" in err

    def test_no_credential_no_warning(self, capsys):
        """With nothing to strip there is no billing consequence to name."""
        provider = ClaudeCodeProvider()
        env = {k: v for k, v in os.environ.items() if k not in cc._CREDENTIAL_ENV}
        with patch.dict(os.environ, env, clear=True):
            provider.get_remove_env()
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Ship gate 4 — Lane A ≡ Lane B parity, for BOTH verdicts
# ---------------------------------------------------------------------------


def _lane_a(provider, model="opus", base_env=None):
    """The fan-out lane's composition, through the one shared builder."""
    return base.build_child_env(
        provider,
        provider_env=provider.resolve_env(model, base.PROFILE_GENERATE),
        base_env=base_env,
    )


def _lane_b(provider, model="opus", *, host_guards=(), passthrough=(), base_env=None):
    """The agent-profile lane's composition, through the same builder."""
    return base.build_child_env(
        provider,
        provider_env=provider.resolve_env(model, base.PROFILE_AGENT_READONLY),
        host_guards=host_guards,
        host_passthrough=passthrough,
        base_env=base_env,
    )


class TestLaneAEqualsLaneB:
    """"Lane A ≡ Lane B, by construction" — the parity the split used to break.

    Before M3, Lane A applied only the adapter's `get_remove_env` while Lane B
    subtracted the host's `env_guards` and **re-admitted `passthrough`**, so the
    same claude-code slot stripped `ANTHROPIC_API_KEY` on the fan-out lane and
    inherited it on every agent-profile dispatch. The verdict is now computed
    once, in one builder, and both lanes consume it — so a decline is symmetric
    too.
    """

    KEYED = {"ANTHROPIC_API_KEY": "sk-test", "SOME_UNRELATED": "keep-me"}

    @pytest.mark.parametrize(
        "verdict", [cc.STRIP_VERDICT_STRIP, cc.STRIP_VERDICT_DECLINE]
    )
    def test_the_credential_name_set_agrees_on_both_lanes(self, verdict):
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc, "_subscription_verdict", return_value=(verdict, "measured")
        ):
            a, b = _lane_a(provider), _lane_b(provider)
        for name in (*cc._CREDENTIAL_ENV, *cc._VENDOR_BACKEND_ENV):
            assert (name in a) == (name in b), name

    def test_a_strip_verdict_strips_on_both_lanes(self):
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_STRIP, "ok")
        ):
            a, b = _lane_a(provider), _lane_b(provider)
        assert "ANTHROPIC_API_KEY" not in a
        assert "ANTHROPIC_API_KEY" not in b

    def test_a_decline_re_admits_on_both_lanes(self):
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_DECLINE, "no")
        ):
            a, b = _lane_a(provider), _lane_b(provider)
        assert a["ANTHROPIC_API_KEY"] == "sk-test"
        assert b["ANTHROPIC_API_KEY"] == "sk-test"

    def test_lane_b_does_not_re_probe(self):
        """One verdict per process, consumed by both lanes."""
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc.shutil, "which", return_value=_RESOLVED
        ), patch.object(
            cc.subprocess, "run", return_value=_probe(SUBSCRIPTION_JSON)
        ) as run:
            _lane_a(provider)
            _lane_b(provider)
        assert run.call_count == 1

    def test_a_host_passthrough_can_no_longer_re_admit_a_stripped_credential(self):
        """The half of A6 that lives in `hosts/capabilities.yaml`.

        Asserted against the executor's composed env, not against the YAML: this
        is the mechanism that made the strip a no-op on Lane B, so it is pinned
        where the re-admission happened.
        """
        provider = ClaudeCodeProvider()
        with patch.dict(os.environ, self.KEYED, clear=False), patch.object(
            cc, "_subscription_verdict", return_value=(cc.STRIP_VERDICT_STRIP, "ok")
        ):
            composed = _lane_b(
                provider,
                host_guards=["CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"],
                passthrough=_claude_code_passthrough(),
            )
        assert "ANTHROPIC_API_KEY" not in composed

    @pytest.mark.parametrize("name", _registry_names())
    def test_every_adapter_composes_the_same_removals_on_both_lanes(self, name):
        """Parametrized over the registry (M0.a), never over a roster.

        The parity claim is about the *builder*, so it has to hold for every
        adapter — including ones with a run-scoped context of their own
        (opencode's hermetic `XDG_CONFIG_HOME` + `--pure`), whose delta must
        still win over whatever the inherited env carried.
        """
        provider = _PROVIDERS[name]
        with patch.dict(os.environ, {"MLLM_PARITY_PROBE": "1"}, clear=False):
            a = _lane_a(provider, model="m")
            b = _lane_b(provider, model="m")
        if a is None and b is None:
            return
        assert (a or {}).get("MLLM_PARITY_PROBE") == (b or {}).get("MLLM_PARITY_PROBE")


class TestTheDispatchContextReachesTheStrip:
    """The anchor and the source env are the **dispatch's**, not the process's.

    `build_child_env` already took an `anchor` — the plan this dispatch resolved
    against — and used it for the allowlist decision, but it asked the adapter
    for its removals context-free. For claude-code that hook is a *policy* read:
    `strip_vendor_keys` and `vendor_backend` resolve through the normal config
    layering, so an anchor-less call evaluates them against whatever
    `.multi-llm/providers.yaml` the process-wide run anchor or the CWD happens to
    find — a credential decision the requested project never made. The same hook
    also decides which names are *present*, which can only mean present in the
    environment the child will actually inherit.
    """

    ANCHOR = "/plans/another-project/auth-feature.md"
    KEYED = {"ANTHROPIC_API_KEY": "sk-test"}

    @staticmethod
    def _recording_policies(seen, policy):
        """Both policy reads, patched to record the anchor each was asked with."""

        def _strip(_name, anchor=None):
            seen.append(("strip_vendor_keys", anchor))
            return policy

        def _backend(_name, anchor=None):
            seen.append(("vendor_backend", anchor))
            return base.VENDOR_BACKEND_AUTO

        return (
            patch.object(cc, "strip_vendor_keys_policy", _strip),
            patch.object(cc, "vendor_backend_policy", _backend),
        )

    def test_the_explicit_anchor_governs_the_policy_read(self):
        """The anchor the builder was given is the anchor the policy is read at."""
        seen = []
        strip, backend = self._recording_policies(seen, base.STRIP_VENDOR_KEYS_NEVER)
        with patch.dict(os.environ, self.KEYED, clear=False), strip, backend:
            base.build_child_env(
                ClaudeCodeProvider(), provider_env={}, anchor=self.ANCHOR
            )
        assert seen, "the credential policy was never consulted"
        assert {anchor for _, anchor in seen} == {self.ANCHOR}

    def test_no_anchor_still_means_the_process_default(self):
        """A context-free dispatch is byte-for-byte what it was."""
        seen = []
        strip, backend = self._recording_policies(seen, base.STRIP_VENDOR_KEYS_NEVER)
        with patch.dict(os.environ, self.KEYED, clear=False), strip, backend:
            base.build_child_env(ClaudeCodeProvider(), provider_env={})
        assert {anchor for _, anchor in seen} == {None}

    def test_presence_is_judged_against_the_env_the_child_inherits(self):
        """A credential the *source* carries is stripped, whatever the process has.

        The builder composes from `base_env` when it is given one, so an adapter
        reasoning about `os.environ` instead would answer about an environment
        no child of this dispatch will ever see.
        """
        source = {"PATH": "/usr/bin", **self.KEYED}
        clean = {k: v for k, v in os.environ.items() if k not in cc._CREDENTIAL_ENV}
        with patch.dict(os.environ, clean, clear=True), patch.object(
            cc, "strip_vendor_keys_policy", return_value=base.STRIP_VENDOR_KEYS_ALWAYS
        ):
            composed = base.build_child_env(
                ClaudeCodeProvider(), provider_env={}, base_env=source
            )
        assert composed is not None
        assert "ANTHROPIC_API_KEY" not in composed

    def test_an_adapter_that_takes_no_context_is_still_asked(self):
        """Most removal lists are static, and a zero-parameter override is fine.

        The hook is duck-typed at the builder, so refusing one would turn a
        correct static list into a `TypeError` mid-dispatch.
        """

        class _Static:
            name = "static"

            def get_remove_env(self):
                return ["SOME_NESTING_GUARD"]

        composed = base.build_child_env(
            _Static(),
            provider_env={},
            base_env={"SOME_NESTING_GUARD": "1", "PATH": "/usr/bin"},
            anchor=self.ANCHOR,
        )
        assert composed == {"PATH": "/usr/bin"}


def _claude_code_passthrough():
    """The claude-code host's own `passthrough`, read from the shipped data."""
    from utils import host_capabilities as caps

    return list(caps.env_guards("claude-code")["passthrough"])


class TestTheHostDataHalf:
    """The `hosts/capabilities.yaml` edit and its generated re-render."""

    def test_the_claude_code_host_sanctions_no_vendor_credential(self):
        """A host passthrough is re-admitted **last**, so this cell is a veto.

        Only the `claude-code` host ever declared one, which is why emptying it
        changes exactly one situation: claude-code dispatched from inside
        claude-code.
        """
        sanctioned = set(_claude_code_passthrough())
        assert sanctioned.isdisjoint(cc._CREDENTIAL_ENV)
        assert sanctioned.isdisjoint(cc._VENDOR_BACKEND_ENV)

    def test_no_host_sanctions_a_vendor_credential(self):
        """The rule, not the one cell — a new host cannot re-open this."""
        from utils import host_capabilities as caps

        for host_id in caps.host_ids():
            sanctioned = set(caps.env_guards(host_id)["passthrough"])
            assert sanctioned.isdisjoint(cc._CREDENTIAL_ENV), host_id
            assert sanctioned.isdisjoint(cc._VENDOR_BACKEND_ENV), host_id

    def test_the_generated_reference_is_not_hand_edited(self):
        """The third edit is a **re-render**, and `--check` is its guard.

        `references/host-capabilities.md` documented the opposite policy; hand
        editing it fails CI *and* drifts from the schema, so the recipe is to
        re-run the renderer and commit both outputs.
        """
        rendered = (SKILL_DIR / "references" / "host-capabilities.md").read_text(
            encoding="utf-8"
        )
        heading = "Sanctioned exceptions that survive stripping"
        line = next(l for l in rendered.splitlines() if heading in l)
        assert "ANTHROPIC_API_KEY" not in line
        assert "*none recorded*" in line


class TestThePolicyKeys:
    """`strip_vendor_keys` / `vendor_backend` are provider keys, read by layering."""

    def test_the_policy_vocabularies_are_closed(self):
        assert base.STRIP_VENDOR_KEYS_POLICIES == ("auto", "always", "never")
        assert base.VENDOR_BACKEND_POLICIES == ("auto", "keep", "strip")

    def test_claude_code_ships_auto_and_codex_defaults_to_never(self):
        """codex has no zero-network verdict, so `auto` cannot mean anything there.

        `codex login status` false-negatives for a `CODEX_API_KEY`-only user and
        `codex doctor --json` is advisory, so instead of guessing, codex strips
        nothing unless the user opts in per provider.
        """
        assert base.strip_vendor_keys_policy("claude-code") == "auto"
        assert base.strip_vendor_keys_policy("codex") == "never"

    @pytest.mark.parametrize("name", _registry_names())
    def test_no_adapter_without_a_verdict_defaults_to_stripping(self, name):
        """Registry-parametrized (M0.a): a new adapter cannot inherit a strip.

        A default of `auto` is only defensible for an adapter with a measured
        zero-network probe; everything else must resolve to `never` until one
        exists.
        """
        policy = base.strip_vendor_keys_policy(name)
        if name in base.STRIP_VENDOR_KEYS_DEFAULTS:
            assert policy == base.STRIP_VENDOR_KEYS_DEFAULTS[name]
        else:
            assert policy == base.STRIP_VENDOR_KEYS_NEVER

    def test_an_unknown_value_warns_once_and_falls_back(self, capsys):
        with patch.object(base, "_provider_block", return_value={"strip_vendor_keys": "yes"}):
            for _ in range(3):
                assert base.strip_vendor_keys_policy("claude-code") == "auto"
        err = capsys.readouterr().err
        assert err.count("strip_vendor_keys") == 1

    def test_the_shipped_config_declares_both_keys(self):
        """They are documented where a user would look for them."""
        import yaml

        config = yaml.safe_load((SKILL_DIR / "providers.yaml").read_text(encoding="utf-8"))
        block = config["providers"]["claude-code"]
        assert block["strip_vendor_keys"] == "auto"
        assert block["vendor_backend"] == "auto"
