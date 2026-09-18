"""M17 (donor §2.4) — the deny-by-default env allowlist, and the gate it ships behind.

Today every provider subprocess sees every other provider's API key and every
cloud token in the shell. M17 replaces that with an allowlist: proxy, CA-bundle,
XDG, keyring and per-CLI home variables pass, the rest does not.

**It ships switched off, and this module is why.** §2.4 makes the allowlist
conditional on a per-CLI × per-platform capability matrix, and M0.a makes the
CLI axis the **live registry** rather than the written roster the specification
was drafted against. That correction is security-relevant rather than tidy: a
matrix derived from a stale roster silently omits the newer adapters'
credential, keyring, proxy and CA cells, and each omission is a mid-run auth or
TLS failure of exactly the kind the allowlist exists to prevent. So:

* the matrix is `len(registry) × len(platforms) × len(capabilities)`, computed
  here from the registry and never typed as a number;
* a cell is **green** only when the CLI, launched under `env -i` plus the
  allowlist, reaches that capability at parity with a full-env control —
  answering `--version` is not parity;
* nothing in this pass ran a cell, so the Linux column is not green and
  `env_allowlist` must ship `off`. `TestTheShipGate` is that condition, written
  so it *keeps* holding: the day the column goes green, the gate stops forcing
  `off` and the default may be flipped by a deliberate edit.

M17 lands **inside** M3's shared env builder (atomic group **A6**): a
deny-by-default list on top of the two divergent constructions Lane A and Lane B
used to have would allow what the other denies, silently. So the last class here
is the Lane A ≡ Lane B env-equality half of that pair.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import base  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402

SKILL_DIR = Path(__file__).parent.parent


def _registry_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one.

    Resolved at class-definition time because `@pytest.mark.parametrize`
    evaluates its argument while the class body runs.

    **No test in this module may name an adapter or a count** (M0.a). Where a
    size is asserted it is *derived* — `len(_PROVIDERS) × len(axis)` — so a
    seventeenth or an eighteenth adapter moves the expectation with it instead of
    quietly falling out of the matrix.
    """
    return sorted(_PROVIDERS)


# ---------------------------------------------------------------------------
# Ship gate 1 — the matrix, at current-roster size
# ---------------------------------------------------------------------------


class TestTheCapabilityMatrix:
    """§2.4's condition 2: the list ships only behind this matrix."""

    def test_the_axes_are_the_declared_ones(self):
        assert base.ENV_ALLOWLIST_PLATFORMS == ("linux", "macos", "windows")
        assert base.ENV_ALLOWLIST_CAPABILITIES == (
            "credential",
            "keyring",
            "proxy",
            "ca_bundle",
        )

    def test_the_cli_axis_is_the_registry_and_not_a_roster(self):
        """The whole of M0.a, as one assertion.

        The specification's axis was a written list of names; if this ever stops
        being the registry, four adapters' cells vanish from the matrix and the
        allowlist ships blind for them.
        """
        matrix = base.env_allowlist_matrix()
        assert {cell[0] for cell in matrix} == set(_registry_names())

    def test_the_matrix_is_registry_x_platforms_x_capabilities(self):
        expected = (
            len(_PROVIDERS)
            * len(base.ENV_ALLOWLIST_PLATFORMS)
            * len(base.ENV_ALLOWLIST_CAPABILITIES)
        )
        assert len(base.env_allowlist_matrix()) == expected

    def test_the_linux_column_is_registry_x_capabilities(self):
        """The ship gate's column, sized from the registry rather than typed."""
        expected = len(_PROVIDERS) * len(base.ENV_ALLOWLIST_CAPABILITIES)
        assert len(base.env_allowlist_column("linux")) == expected

    @pytest.mark.parametrize("platform", base.ENV_ALLOWLIST_PLATFORMS)
    def test_every_column_is_the_same_size(self, platform):
        expected = len(_PROVIDERS) * len(base.ENV_ALLOWLIST_CAPABILITIES)
        assert len(base.env_allowlist_column(platform)) == expected

    @pytest.mark.parametrize("name", _registry_names())
    def test_every_registered_adapter_has_a_full_row(self, name):
        """No adapter may be missing a credential, keyring, proxy or CA cell."""
        matrix = base.env_allowlist_matrix()
        for platform in base.ENV_ALLOWLIST_PLATFORMS:
            for capability in base.ENV_ALLOWLIST_CAPABILITIES:
                assert (name, platform, capability) in matrix

    def test_an_unrun_cell_reads_unmeasured_and_never_green(self):
        """A cell is a measurement or it is nothing — never an inference."""
        statuses = set(base.env_allowlist_matrix().values())
        assert statuses <= {base.ENV_ALLOWLIST_GREEN, base.ENV_ALLOWLIST_UNMEASURED}
        assert base.ENV_ALLOWLIST_UNMEASURED in statuses

    def test_no_cell_is_recorded_for_an_unregistered_adapter(self):
        """A measurement for a name nobody registers is a stale roster leaking in."""
        for provider, _platform, _capability in base.ENV_ALLOWLIST_MEASURED:
            assert provider in _PROVIDERS


class TestTheShipGate:
    """`env_allowlist` stays `off` until the Linux column is green (M0.a)."""

    def test_the_default_is_off(self):
        assert base.ENV_ALLOWLIST_DEFAULT == base.ENV_ALLOWLIST_OFF

    def test_the_shipped_config_says_off(self):
        import yaml

        config = yaml.safe_load((SKILL_DIR / "providers.yaml").read_text(encoding="utf-8"))
        value = config["env_allowlist"]
        # YAML 1.1 would resolve a bare `off` to False; either spelling is the
        # documented word, neither is `on`.
        assert value in (base.ENV_ALLOWLIST_OFF, False)

    def test_the_resolved_policy_is_off(self):
        assert base.env_allowlist_policy() == base.ENV_ALLOWLIST_OFF
        assert base.env_allowlist_enabled() is False

    def test_the_gate_itself(self):
        """The condition, not a snapshot of today's answer.

        While any Linux cell is unmeasured the shipped default MUST be `off`, so
        an unrun cell degrades to today's inheritance rather than to a silent
        mid-run auth, proxy or TLS failure. When the column goes green this test
        stops constraining the default and a deliberate edit may flip it.
        """
        if base.env_allowlist_column_is_green("linux"):
            pytest.skip("the Linux column is green; the default may be flipped")
        assert base.ENV_ALLOWLIST_DEFAULT == base.ENV_ALLOWLIST_OFF, (
            "the Linux column of the capability matrix is not green, so the "
            "allowlist may not be on by default — an unmeasured cell is a "
            "mid-run auth or TLS failure waiting for the user who has that "
            "capability."
        )

    def test_a_column_with_one_unmeasured_cell_is_not_green(self):
        """The gate is `all`, never `any`."""
        assert not base.env_allowlist_column_is_green("linux")
        measured = {
            cell: base.ENV_ALLOWLIST_GREEN
            for cell in base.env_allowlist_column("linux")
        }
        with patch.object(base, "ENV_ALLOWLIST_MEASURED", measured):
            assert base.env_allowlist_column_is_green("linux")
        one_short = dict(measured)
        one_short.pop(next(iter(one_short)))
        with patch.object(base, "ENV_ALLOWLIST_MEASURED", one_short):
            assert not base.env_allowlist_column_is_green("linux")

    def test_an_unknown_policy_value_warns_once_and_falls_back_to_off(self, capsys):
        base.reset_env_policy_warnings()
        with patch.object(base, "_merged_config", return_value={"env_allowlist": "yes"}):
            for _ in range(3):
                assert base.env_allowlist_policy() == base.ENV_ALLOWLIST_OFF
        assert capsys.readouterr().err.count("env_allowlist") == 1
        base.reset_env_policy_warnings()

    def test_a_yaml_boolean_is_read_as_the_documented_word(self):
        """YAML 1.1 resolves a bare `on`/`off` to a boolean."""
        with patch.object(base, "_merged_config", return_value={"env_allowlist": False}):
            assert base.env_allowlist_policy() == base.ENV_ALLOWLIST_OFF
        with patch.object(base, "_merged_config", return_value={"env_allowlist": True}):
            assert base.env_allowlist_policy() == base.ENV_ALLOWLIST_ON


# ---------------------------------------------------------------------------
# Ship gate 2 — the list's contents, per adapter, over the registry
# ---------------------------------------------------------------------------


class TestTheListContents:
    """The measured corrections §2.4 makes to the donor's starting list."""

    @pytest.mark.parametrize("name", _registry_names())
    def test_every_registered_adapter_declares_a_profile(self, name):
        """The completeness rule M0.a exists for.

        `env_allowlist_for` falls back for an unknown adapter — the fail-open
        direction — and this test is what stops that fallback from becoming the
        normal case: a newly registered adapter with no row reds here rather than
        shipping without its own home and CA variables.
        """
        assert name in base.ENV_ALLOWLIST_PROFILES

    def test_no_profile_is_declared_for_an_unregistered_adapter(self):
        assert set(base.ENV_ALLOWLIST_PROFILES) == set(_registry_names())

    @pytest.mark.parametrize("name", _registry_names())
    def test_every_adapter_is_in_exactly_one_ca_family(self, name):
        assert base.ENV_ALLOWLIST_PROFILES[name].ca_family in base.CA_BUNDLE_ENV

    def test_the_two_ca_families_partition_the_registry(self):
        """`NODE_EXTRA_CA_CERTS` is not "all these CLIs are Node".

        It covers the npm installs and is useless for the Go/Python/Rust/native
        ones, which need `SSL_CERT_FILE`/`SSL_CERT_DIR`/`REQUESTS_CA_BUNDLE`/
        `CURL_CA_BUNDLE` — none of which the donor's list carried, so a
        corporate-CA user was broken by it as drafted. The partition is asserted
        over the registry rather than by counting either side.
        """
        by_family = {family: set() for family in base.CA_BUNDLE_ENV}
        for name in _registry_names():
            by_family[base.ENV_ALLOWLIST_PROFILES[name].ca_family].add(name)
        assert set().union(*by_family.values()) == set(_registry_names())
        assert sum(len(v) for v in by_family.values()) == len(_PROVIDERS)
        for family, members in by_family.items():
            assert members, f"the {family} CA family is empty"

    @pytest.mark.parametrize("name", _registry_names())
    def test_both_ca_families_are_admitted_for_every_adapter(self, name):
        """A CA variable a CLI ignores is inert; a missing one is a TLS failure.

        The declared family decides which variable is *load-bearing* for that
        adapter's `ca_bundle` cell, not which ones it may see.
        """
        for family in base.CA_BUNDLE_ENV.values():
            for variable in family:
                assert base.env_allowlist_admits(name, variable)

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_measured_omissions_are_all_admitted(self, name):
        """Every correction §2.4's measurement note names, for every adapter."""
        for variable in (
            "SSH_AUTH_SOCK",            # every CLI shells out to git
            "DBUS_SESSION_BUS_ADDRESS",  # goose's org.freedesktop.secrets
            "XDG_RUNTIME_DIR",           # the other half of that pair
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "NODE_OPTIONS",
            "HOME",
            "PATH",
            "TMPDIR",
        ):
            assert base.env_allowlist_admits(name, variable), variable

    @pytest.mark.parametrize("name", _registry_names())
    def test_proxies_pass_in_both_cases(self, name):
        """Lower-case `http_proxy`/`https_proxy` are aider/libcurl's spelling."""
        for variable in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        ):
            assert base.env_allowlist_admits(name, variable), variable

    #: Vendor credentials and cloud tokens, used as the negative probe below.
    #: A CLI seeing its **own** credential variable is correct — that is how it
    #: authenticates — so the assertion is scoped to *foreign* ones by asking
    #: whether the adapter's own prefix family covers the name.
    CREDENTIAL_PROBES = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "DASHSCOPE_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    )

    @pytest.mark.parametrize("name", _registry_names())
    def test_another_providers_credential_never_passes(self, name):
        """The point of the exercise, stated as a negative.

        A vendor credential belonging to some *other* CLI has no business in this
        child's environment, and neither has a cloud token. A credential the
        adapter's own prefix family covers is excluded — `GEMINI_API_KEY` for
        gemini is how gemini authenticates, and denying it would be the mid-run
        auth failure the matrix exists to prevent.
        """
        own = base.ENV_ALLOWLIST_PROFILES[name].prefixes
        for variable in self.CREDENTIAL_PROBES:
            if any(variable.startswith(prefix) for prefix in own):
                continue
            assert not base.env_allowlist_admits(name, variable), variable

    def test_no_adapter_sees_every_credential(self):
        """The negative above must not be vacuous for any adapter.

        Every adapter denies at least most of the probe set; an adapter whose
        prefix family swallowed all of them would pass the test above while
        allowlisting nothing.
        """
        for name in _registry_names():
            denied = [
                v for v in self.CREDENTIAL_PROBES if not base.env_allowlist_admits(name, v)
            ]
            assert len(denied) >= len(self.CREDENTIAL_PROBES) - 1, name

    @pytest.mark.parametrize("name", _registry_names())
    def test_an_adapters_own_prefix_family_passes(self, name):
        """Each harness's own prefix family, per adapter."""
        profile = base.ENV_ALLOWLIST_PROFILES[name]
        for prefix in profile.prefixes:
            assert base.env_allowlist_admits(name, prefix + "SOMETHING")
        for variable in profile.names:
            assert base.env_allowlist_admits(name, variable)

    def test_one_adapters_home_var_does_not_leak_into_another(self):
        """`CODEX_HOME` is codex's; it is not opencode's business.

        Asserted structurally over the registry: for every pair of adapters, a
        name declared by exactly one of them is admitted by that one only.
        """
        shared_base = {n.upper() for n in base.ENV_ALLOWLIST_BASE_NAMES}
        declared = {
            name: {
                variable
                for variable in base.ENV_ALLOWLIST_PROFILES[name].names
                # A name that is also in the shared base list (aider's `COLUMNS`)
                # is admitted everywhere by design, not by leakage.
                if variable.upper() not in shared_base
                and not any(
                    variable.upper().startswith(p.upper())
                    for p in base.ENV_ALLOWLIST_BASE_PREFIXES
                )
            }
            for name in _registry_names()
        }
        shared = {
            variable
            for variable in set().union(*declared.values())
            if sum(variable in names for names in declared.values()) > 1
        }
        for name, names in declared.items():
            for variable in names - shared:
                others = [other for other in declared if other != name]
                admitting = [o for o in others if base.env_allowlist_admits(o, variable)]
                # A sibling may still admit it through its own prefix family —
                # that is by design and only ever widens within one vendor.
                for other in admitting:
                    prefixes = base.ENV_ALLOWLIST_PROFILES[other].prefixes
                    assert any(variable.startswith(p) for p in prefixes), (
                        f"{variable} is declared only by {name} but {other} "
                        f"admits it through the shared base list"
                    )

    def test_filtering_keeps_only_allowed_names(self):
        source = {
            "PATH": "/usr/bin",
            "HOME": "/home/dev",
            "HTTPS_PROXY": "http://proxy:3128",
            "ANTHROPIC_API_KEY": "sk-secret",
            "AWS_SESSION_TOKEN": "aws-secret",
        }
        filtered = base.apply_env_allowlist(source, "codex")
        assert set(filtered) == {"PATH", "HOME", "HTTPS_PROXY"}


# ---------------------------------------------------------------------------
# Ship gate 3 — Lane A ≡ Lane B env equality (the A6 half M17 owns)
# ---------------------------------------------------------------------------


SAMPLE_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/home/dev",
    "HTTPS_PROXY": "http://proxy:3128",
    "SSH_AUTH_SOCK": "/run/ssh",
    "ANTHROPIC_API_KEY": "sk-secret",
    "OPENROUTER_API_KEY": "or-secret",
    "AWS_SESSION_TOKEN": "aws-secret",
}


def _own_credential_env(provider):
    """The env name this adapter declares as its CLI's **own** credential.

    Read off the adapter module's own constant rather than from a table here,
    so the exemption above cannot be granted by editing this file: an adapter
    claims it by declaring it beside the code that emits it.
    """
    module = sys.modules[type(provider).__module__]
    return getattr(module, f"{provider.name.upper().replace('-', '_')}_CREDENTIAL_ENV", None)


@pytest.fixture(autouse=True)
def _no_real_vendor_cli():
    """No sweep below may spawn the developer's own CLI, or read their login.

    `SAMPLE_ENV` carries real credential *names* on purpose, and the builder asks
    each adapter for its removals with **the composed source env** — so an
    adapter whose strip policy is settled by a zero-network preflight would run
    that preflight against this synthetic environment. `shutil.which` is stubbed
    to "not installed" for the duration, which is the missing-binary row every
    such gate declines on, so what these tests measure is the *composition* and
    never the machine they run on. Nothing here resolves a binary for any other
    reason: the sweeps call `resolve_env` and `build_child_env`, never
    `build_command`.
    """
    with patch.object(base.shutil, "which", return_value=None):
        yield


def _both_lanes(provider, source, *, delta=None, host_guards=(), passthrough=()):
    """Both lanes' composed env from **one** adapter delta.

    The delta is taken once and handed to both lanes on purpose. Several
    adapters mint a fresh per-dispatch isolation directory inside `get_env`
    (`COPILOT_HOME`, `KIMI_CODE_HOME`, `QWEN_HOME`, `VIBE_HOME`), so calling
    `resolve_env` twice yields two different temp paths for reasons that have
    nothing to do with lane parity — that is the home-isolation feature working.
    What M3/M17 own, and what these tests are about, is the **composition**: given
    the same delta, the same inherited env and the same host entry, the two lanes
    must produce the same child environment.
    """
    if delta is None:
        # `base_env=source`, because `build_child_env` below is given the same
        # source and the two must reason about one environment. An adapter that
        # passes a credential THROUGH from what the child will inherit (omp, the
        # only one whose CLI reads its key under the unprefixed name) reads it
        # here; every adapter that ignores the keyword — which is all the others
        # — composes exactly the delta it always did.
        delta = provider.resolve_env("m", base.PROFILE_GENERATE, base_env=source)
    return (
        base.build_child_env(provider, provider_env=delta, base_env=source),
        base.build_child_env(
            provider,
            provider_env=delta,
            host_guards=host_guards,
            host_passthrough=passthrough,
            base_env=source,
        ),
    )


class TestLaneAEqualsLaneBUnderTheAllowlist:
    """§2.4's condition 1: one builder, or one lane allows what the other denies.

    The two lanes really do differ — Lane B additionally subtracts the running
    host's `env_guards` — so the claim being pinned is precise: **the allowlist
    decision is identical on both lanes**, and the only names on which the two
    composed environments may differ are the host guards Lane B was handed.
    """

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_two_lanes_agree_with_the_allowlist_on(self, name):
        with patch.object(base, "env_allowlist_enabled", return_value=True):
            a, b = _both_lanes(_PROVIDERS[name], SAMPLE_ENV)
        assert a == b

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_two_lanes_agree_with_the_allowlist_off(self, name):
        with patch.object(base, "env_allowlist_enabled", return_value=False):
            a, b = _both_lanes(_PROVIDERS[name], SAMPLE_ENV)
        assert a == b

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_allowlist_decision_itself_is_lane_independent(self, name):
        """The inherited names admitted are the same set on both lanes.

        Stated over the *inherited* keys rather than the composed env, so a
        per-dispatch isolation path an adapter mints inside `get_env` cannot make
        this pass or fail for the wrong reason.
        """
        provider = _PROVIDERS[name]
        with patch.object(base, "env_allowlist_enabled", return_value=True):
            a = base.build_child_env(
                provider,
                provider_env=provider.resolve_env("m", base.PROFILE_GENERATE),
                base_env=SAMPLE_ENV,
            )
            b = base.build_child_env(
                provider,
                provider_env=provider.resolve_env("m", base.PROFILE_AGENT_READONLY),
                base_env=SAMPLE_ENV,
            )
        assert {k for k in a if k in SAMPLE_ENV} == {k for k in b if k in SAMPLE_ENV}

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_only_lane_difference_is_the_host_guards(self, name):
        provider = _PROVIDERS[name]
        guards = ["CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_SESSION_ID"]
        source = {**SAMPLE_ENV, **{g: "1" for g in guards}}
        with patch.object(base, "env_allowlist_enabled", return_value=True):
            a, b = _both_lanes(provider, source, host_guards=guards)
        assert set(a or {}) - set(b or {}) <= set(guards)
        assert set(b or {}) - set(a or {}) == set()

    @pytest.mark.parametrize("name", _registry_names())
    def test_no_lane_leaks_a_foreign_credential_when_the_allowlist_is_on(self, name):
        """**Foreign** is the load-bearing word, and one adapter needed it said.

        ``OPENROUTER_API_KEY`` is foreign to almost every adapter here: kimi,
        qwen and copilot all carry the same key past the filter, but each under
        its **own** prefixed spelling (``KIMI_MODEL_API_KEY``,
        ``COPILOT_PROVIDER_API_KEY``), so the unprefixed name reaching their
        child really would be a leak. omp is the one adapter whose CLI reads the
        credential under that exact name — there is no other spelling for it to
        use — so passing it through is what its ``get_env`` exists to do, and
        the filter dropping it is the bug rather than the guard.

        The carve-out is therefore **by the adapter's own declared constant**,
        not by a name in a list here, and it is not a free pass: the value has
        to be the SOURCE's, so an adapter minting a credential of its own still
        fails. ``AWS_SESSION_TOKEN`` has no such exemption and never will —
        nothing in this registry speaks to AWS.
        """
        provider = _PROVIDERS[name]
        own = _own_credential_env(provider)
        with patch.object(base, "env_allowlist_enabled", return_value=True):
            for env in _both_lanes(provider, SAMPLE_ENV):
                assert "AWS_SESSION_TOKEN" not in env
                if own == "OPENROUTER_API_KEY":
                    assert env.get("OPENROUTER_API_KEY") == SAMPLE_ENV[
                        "OPENROUTER_API_KEY"
                    ], "a passed-through credential must be the source's own value"
                else:
                    assert "OPENROUTER_API_KEY" not in env

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_adapters_own_delta_always_wins_over_the_filter(self, name):
        """Composition order: the delta is applied **after** the filter.

        This is what keeps a run-scoped redirect intact under an allowlist that
        would not have *inherited* it — opencode's hermetic `XDG_CONFIG_HOME`
        being the live case, but the rule is the adapter's, not opencode's.
        """
        provider = _PROVIDERS[name]
        delta = provider.resolve_env("m", base.PROFILE_GENERATE)
        with patch.object(base, "env_allowlist_enabled", return_value=True):
            composed = base.build_child_env(
                provider, provider_env=delta, base_env=SAMPLE_ENV
            )
        removed = set(base._provider_removals(provider, base_env=SAMPLE_ENV))
        for key, value in delta.items():
            if key in removed:
                continue
            assert composed[key] == value, key


class TestTheBuilderPreservesTodaysBehaviour:
    """With the allowlist off, the builder must be byte-identical to before it."""

    class _Bare:
        name = "bare"

        def get_remove_env(self):
            return []

    def test_no_delta_and_no_removals_returns_none(self):
        """`env=None` is what lets `subprocess` inherit directly."""
        assert (
            base.build_child_env(
                self._Bare(), provider_env={}, base_env=SAMPLE_ENV
            )
            is None
        )

    def test_the_allowlist_forces_a_composed_env_even_with_no_delta(self):
        with patch.object(base, "env_allowlist_enabled", return_value=True):
            composed = base.build_child_env(
                self._Bare(), provider_env={}, base_env=SAMPLE_ENV
            )
        assert composed is not None
        assert "ANTHROPIC_API_KEY" not in composed

    def test_a_passthrough_re_admits_a_host_guard(self):
        """The host-side allowlist still works for what it is actually for."""
        composed = base.build_child_env(
            self._Bare(),
            provider_env={"X": "1"},
            host_guards=["CLAUDECODE", "CLAUDE_SKILL_DIR"],
            host_passthrough=["CLAUDE_SKILL_DIR"],
            base_env={**SAMPLE_ENV, "CLAUDECODE": "1", "CLAUDE_SKILL_DIR": "/skill"},
        )
        assert "CLAUDECODE" not in composed
        assert composed["CLAUDE_SKILL_DIR"] == "/skill"

    @pytest.mark.parametrize("name", _registry_names())
    def test_the_builder_reproduces_the_old_hand_rolled_composition(self, name):
        """Green stays green: the refactor moved code, not behaviour.

        The right-hand side is literally what both spawn sites used to inline —
        merge the delta over the inherited env, then pop the union of the
        adapter's list and the host's guards minus the passthrough. The
        adapter's list is asked for with the same context the builder asks with
        (`base_env=source`), because for an adapter that decides by presence the
        two are the same question and asking it two different ways would compare
        this composition against a different environment's.
        """
        provider = _PROVIDERS[name]
        delta = provider.resolve_env("m", base.PROFILE_AGENT_READONLY)
        guards = ["CLAUDECODE", "CLAUDE_SESSION_ID"]
        passthrough = ["CLAUDE_SESSION_ID"]
        source = {**SAMPLE_ENV, "CLAUDECODE": "1", "CLAUDE_SESSION_ID": "s"}

        expected = {**source, **delta}
        for key in [
            n
            for n in dict.fromkeys(
                [*base._provider_removals(provider, base_env=source), *guards]
            )
            if n not in passthrough
        ]:
            expected.pop(key, None)

        composed = base.build_child_env(
            provider,
            provider_env=delta,
            host_guards=guards,
            host_passthrough=passthrough,
            base_env=source,
        )
        assert composed == expected


class TestItComposesWithTheHermeticAdapter:
    """M4's run-scoped mechanism and M17's filter must not fight each other.

    opencode enters a `ContextVar`-scoped hermetic dispatch at the spawn sites,
    which renders `--pure` **and** an `XDG_CONFIG_HOME` redirect from one piece
    of state. Those are emitted as the adapter's delta, so they survive the
    filter by construction; the config-carrying names it strips are removed
    after it, so the strip is not undone either.
    """

    def test_the_hermetic_redirect_survives_the_filter(self):
        from utils.providers import opencode

        provider = _PROVIDERS["opencode"]
        opencode.reset_hermetic_state()
        try:
            with provider.hermetic_dispatch(base.PROFILE_GENERATE, "m") as refusal:
                assert refusal is None
                delta = provider.resolve_env("m", base.PROFILE_GENERATE)
                with patch.object(base, "env_allowlist_enabled", return_value=True):
                    composed = base.build_child_env(
                        provider,
                        provider_env=delta,
                        base_env={
                            **SAMPLE_ENV,
                            "XDG_CONFIG_HOME": "/home/dev/.config",
                            "OPENCODE_CONFIG": "/home/dev/opencode.json",
                        },
                    )
            assert composed["XDG_CONFIG_HOME"] == delta["XDG_CONFIG_HOME"]
            assert composed["XDG_CONFIG_HOME"] != "/home/dev/.config"
            assert "OPENCODE_CONFIG" not in composed
        finally:
            opencode.reset_hermetic_state()

    def test_outside_a_hermetic_dispatch_nothing_is_redirected(self):
        from utils.providers import opencode

        provider = _PROVIDERS["opencode"]
        opencode.reset_hermetic_state()
        delta = provider.resolve_env("m", base.PROFILE_GENERATE)
        assert "XDG_CONFIG_HOME" not in delta


class TestTheAllowlistIsBuiltInsideTheSharedBuilder:
    """A6's structural half, asserted rather than described.

    If the filter ever moves out of `build_child_env` and into either spawn
    site, the two lanes can diverge again — which is the failure §2.4's
    condition 1 exists to prevent.
    """

    @pytest.mark.parametrize(
        "module_name", ["llm_client.py", "subagent_executor.py"]
    )
    def test_neither_spawn_site_filters_the_env_itself(self, module_name):
        source = (SKILL_DIR / "utils" / module_name).read_text(encoding="utf-8")
        assert "build_child_env" in source, (
            f"{module_name} no longer routes its child env through the one "
            f"shared builder; Lane A and Lane B can diverge again."
        )
        for token in ("apply_env_allowlist", "env_allowlist_admits"):
            assert token not in source, (
                f"{module_name} applies the allowlist itself — it belongs in "
                f"`base.build_child_env`, which both lanes go through."
            )

    def test_the_builder_is_the_only_caller_of_the_filter(self):
        import inspect

        source = inspect.getsource(base.build_child_env)
        assert "apply_env_allowlist" in source
        assert "env_allowlist_enabled" in source
