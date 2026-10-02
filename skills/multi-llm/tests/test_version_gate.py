"""M14 (§5.4 / §5.4.1) — the one version gate, and the four gates it absorbed.

Before M14 the same two rules lived in several places at once, each with the
reason written down beside it:

* the dotted-version **comparison** was in `providers/base._version_below` *and*
  in `subagent_executor._version_tuple`, deliberately duplicated because M14 was
  the row that owned the merge;
* the per-flag **help sniff** was shipped three times — inside L7
  (`--fallback-model`), M7 (`--json-schema`) and M1 (`--trust --mode ask`) —
  because §11.1 needs those tasks severable from §11.2, plus a fourth, purely
  declarative one in codex, whose flags are exempt at its floor (§5.4.1(a)).

This module is what makes "absorbed" a checked statement rather than a claim.
It asserts the merged rule, the floor/ceiling verdict, the stdout **and stderr**
version read §5.4 requires, and the two failure postures the shared flag gate
holds — *decline the flag, keep the run* for an optional flag, *refuse the slot
before spawn* for a load-bearing one.

**Green stays green is the point.** M14 is an absorption refactor: the gate
tests L7, M1 and M7 shipped are re-pointed at the shared implementation and
still assert exactly what they asserted before. This module adds what none of
them could — that there is now only *one* implementation to point at.

Nothing here names an adapter roster or a count (M0.a). The sweeps parametrize
over the live registry; the mechanism tests use a local throwaway adapter, so
they measure the gate rather than whichever CLI a developer happens to have
installed.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import subagent_executor as executor  # noqa: E402
from utils.provider_registry import _PROVIDERS, load_base_config  # noqa: E402
from utils.providers import base as base_module  # noqa: E402
from utils.providers.base import (  # noqa: E402
    VERSION_ABOVE_VERIFIED,
    VERSION_OK,
    VERSION_TOO_LOW,
    VERSION_TOO_NEW,
    VERSION_UNREAD,
    LLMProvider,
    _version_below,
    read_help_tokens,
    reset_flag_gate_cache,
    version_ceiling,
    version_tuple,
    version_verdict,
)


def _registry_names():
    """Every provider id in the registry, so the sweeps cannot miss a new one.

    Resolved at class-definition time because `@pytest.mark.parametrize`
    evaluates its argument while the class body runs. **No test here may name an
    adapter or a count** (M0.a).
    """
    return sorted(_PROVIDERS)


# --------------------------------------------------------------------------- #
# A throwaway adapter, so the gate is measured rather than the developer's CLIs
# --------------------------------------------------------------------------- #


class _Gated(LLMProvider):
    """The smallest adapter that can be version-gated and flag-gated."""

    def __init__(self, binary_name="probe-cli", **declarations):
        self._binary = binary_name
        for key, value in declarations.items():
            setattr(self, key, value)

    @property
    def name(self):
        return "probe-cli"

    @property
    def default_timeout(self):
        return 60

    def is_available(self):
        import shutil

        return shutil.which(self._binary) is not None

    def build_command(self, prompt, model, **_kwargs):
        return [self._binary, prompt]

    def parse_output(self, stdout, stderr):
        return {"content": stdout}


def _write_cli(path: Path, *, stdout: str = "", stderr: str = "", exit_code: int = 0,
               counter: Path | None = None) -> Path:
    """A fake CLI that answers anything with fixed streams and a fixed code.

    The counter, when given, ticks once per invocation — which is how "one help
    read per CLI per process" is observed rather than assumed. It is rendered as
    the empty string when absent, never as ``str(None)``: ``"None"`` is a
    perfectly truthy string and the default would otherwise tick a counter into
    a file literally named ``None``.
    """
    script = path.with_suffix(".py")
    script.write_text(
        textwrap.dedent(
            f"""\
            import sys, pathlib
            counter = {('' if counter is None else str(counter))!r}
            if counter:
                p = pathlib.Path(counter)
                p.write_text(str(int(p.read_text() or 0) + 1) if p.exists() else '1')
            sys.stdout.write({stdout!r})
            sys.stderr.write({stderr!r})
            sys.exit({exit_code})
            """
        ),
        encoding="utf-8",
    )
    path.write_text(
        f"#!/bin/sh\nexec {sys.executable!r} {str(script)!r} \"$@\"\n", encoding="utf-8"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture(autouse=True)
def _clean_gate_state():
    """No memoised help read and no armed warn-once set leaks between tests."""
    reset_flag_gate_cache()
    yield
    reset_flag_gate_cache()


# --------------------------------------------------------------------------- #
# Seam 1 — the comparison, merged
# --------------------------------------------------------------------------- #


class TestOneComparison:
    """`version_tuple` is the whole rule, and both former owners now use it."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2.1.232", (2, 1, 232)),
            ("0.144.0", (0, 144, 0)),
            # Build noise: the leading integer of each component wins, so a
            # date-plus-hash version orders like the date it is.
            ("2026.08.11-e8db854", (2026, 8, 11)),
            ("1.18.12-beta.3", (1, 18, 12, 3)),
            # A component with no leading digit contributes 0 rather than
            # raising — a version comparison may not crash a dispatch path.
            ("v2.0", (0, 0)),
            ("", (0,)),
        ],
    )
    def test_the_parse_is_leading_integers_component_wise(self, text, expected):
        assert version_tuple(text) == expected

    def test_the_availability_label_and_the_preflight_share_one_rule(self):
        """The seam M10 left open, closed and pinned.

        `_version_below` is what `describe_availability` labels a build with;
        `version_verdict` is what the executor refuses one with. Both are
        `version_tuple`, so a build cannot be "too old" for the refusal and fine
        for the report — which is exactly the drift a second copy allows.
        """
        assert base_module._version_below.__module__ == base_module.__name__
        pairs = [
            ("1.9.9", "2.0.0", True),
            ("2.0.0", "2.0.0", False),
            ("2.0.1", "2.0.0", False),
            ("0.100.0", "0.144.0", True),
            ("10.0.0", "9.9.9", False),  # numeric, never lexicographic
            ("2026.08.11-e8db854", "2026.08.12", True),
        ]
        for read, floor, below in pairs:
            assert _version_below(read, floor) is below
            provider = _Gated(min_version=floor)
            verdict = version_verdict(provider, read)
            assert (verdict.state == VERSION_TOO_LOW) is below, (
                f"{read} vs floor {floor}: the label and the refusal disagree"
            )

    def test_the_executor_kept_no_private_copy(self):
        """`subagent_executor._version_tuple` is gone, not shadowed."""
        assert not hasattr(executor, "_version_tuple")
        assert executor.version_verdict is version_verdict


# --------------------------------------------------------------------------- #
# The floor / ceiling / verified verdict
# --------------------------------------------------------------------------- #


class TestTheVerdict:
    """Floor, then ceiling, then the advisory warning — in the preflight's order."""

    def test_an_unread_version_is_never_a_refusal(self):
        """Absence of evidence is not a negative verdict (D3m)."""
        provider = _Gated(min_version="2.0.0", max_version_exclusive="3.0.0")
        for unread in (None, "", "unknown"):
            verdict = version_verdict(provider, unread)
            assert verdict.state == VERSION_UNREAD
            assert verdict.refuses is False
            assert verdict.detail == ""

    def test_below_the_floor_refuses_and_names_both_numbers(self):
        verdict = version_verdict(_Gated(min_version="0.144.0"), "0.100.0")
        assert verdict.state == VERSION_TOO_LOW
        assert verdict.refuses is True
        assert "0.100.0" in verdict.detail and "0.144.0" in verdict.detail

    def test_below_the_floor_names_the_declared_upgrade_command(self):
        """`upgrade_hint` is printed with the floor substituted, never run."""
        provider = _Gated(min_version="1.18.34", upgrade_hint="probe-cli upgrade {version}")
        verdict = version_verdict(provider, "1.18.20")
        assert verdict.refuses is True
        assert verdict.detail.endswith(
            "Upgrade probe-cli to >= 1.18.34 (run: probe-cli upgrade 1.18.34)."
        )
        installed = _Gated(min_version="1.18.34", upgrade_hint="probe-cli upgrade {version}")
        installed.is_available = lambda: True
        report = installed.describe_availability(version="1.18.20")
        assert report.detail.endswith(
            "upgrade probe-cli to >= 1.18.34 (run: probe-cli upgrade 1.18.34)."
        )

    def test_without_an_upgrade_hint_the_sentence_is_unchanged(self):
        verdict = version_verdict(_Gated(min_version="0.144.0"), "0.100.0")
        assert verdict.detail.endswith("Upgrade probe-cli to >= 0.144.0.")

    def test_the_floor_is_inclusive_and_the_ceiling_is_exclusive(self):
        """The declared ceiling is the FIRST refused build, the floor the first allowed."""
        provider = _Gated(min_version="1.0.0", max_version_exclusive="2.0.0")
        assert version_verdict(provider, "1.0.0").state == VERSION_OK
        assert version_verdict(provider, "1.9.9").state == VERSION_OK
        assert version_verdict(provider, "2.0.0").state == VERSION_TOO_NEW
        assert version_verdict(provider, "2.0.1").state == VERSION_TOO_NEW

    def test_the_ceiling_refuses_and_names_the_number(self):
        verdict = version_verdict(_Gated(max_version_exclusive="2.0.0"), "2.3.1")
        assert verdict.refuses is True
        assert "2.3.1" in verdict.detail and "2.0.0" in verdict.detail

    def test_above_verified_warns_and_never_refuses(self):
        """A ceiling and `verified_version` are different claims, and stay so."""
        verdict = version_verdict(_Gated(verified_version="1.18.12"), "99.0.0")
        assert verdict.state == VERSION_ABOVE_VERIFIED
        assert verdict.refuses is False
        assert "newer than the release" in verdict.detail

    def test_the_sentence_carries_the_name_the_caller_asked_for(self):
        """The executor passes the registry key the user typed, not `provider.name`."""
        verdict = version_verdict(
            _Gated(min_version="2.0.0"), "1.0.0", name="some-runner"
        )
        assert verdict.detail.startswith("some-runner 1.0.0 is older")

    def test_no_declared_bound_is_no_verdict(self):
        assert version_verdict(_Gated(), "1.2.3").state == VERSION_OK


# --------------------------------------------------------------------------- #
# §5.4 — the version read takes BOTH streams
# --------------------------------------------------------------------------- #


class TestTheVersionIsReadFromEitherStream:
    """"Some CLIs banner to stderr; opencode is stdout-only" — so read both."""

    def test_a_cli_that_prints_its_version_on_stderr_is_read(self, tmp_path):
        """The §6.1 matrix's named case, end to end through the real probe.

        Nothing about this CLI writes to stdout. Before §5.4 that is a version
        recorded `unknown`, which silently disables the floor for that adapter —
        a gate that cannot read a version cannot refuse an old one.
        """
        binary = _write_cli(
            tmp_path / "stderr-versioned", stdout="", stderr="probe-cli 1.4.2\n"
        )
        provider = _Gated(min_version="1.0.0", version_cmd=(str(binary), "--version"))

        completed = executor._run_probe(list(provider.version_cmd))
        assert completed is not None and completed.stdout == ""
        parsed = provider.parse_version(completed.stdout, completed.stderr)

        assert parsed == "1.4.2"
        assert version_verdict(provider, parsed).state == VERSION_OK

    def test_the_same_cli_below_the_floor_is_refused_from_stderr_alone(self, tmp_path):
        binary = _write_cli(
            tmp_path / "stderr-old", stdout="", stderr="probe-cli 0.9.0\n"
        )
        provider = _Gated(min_version="1.0.0", version_cmd=(str(binary), "--version"))

        completed = executor._run_probe(list(provider.version_cmd))
        verdict = version_verdict(
            provider, provider.parse_version(completed.stdout, completed.stderr)
        )

        assert verdict.state == VERSION_TOO_LOW
        assert verdict.refuses is True

    def test_a_stdout_only_cli_still_reads(self, tmp_path):
        """The other half of the pair, so "both streams" is not "stderr wins"."""
        binary = _write_cli(tmp_path / "stdout-versioned", stdout="1.18.12\n")
        provider = _Gated(version_cmd=(str(binary), "--version"))
        completed = executor._run_probe(list(provider.version_cmd))

        assert provider.parse_version(completed.stdout, completed.stderr) == "1.18.12"

    def test_neither_stream_carrying_a_version_is_unknown_not_a_refusal(self, tmp_path):
        binary = _write_cli(tmp_path / "mute", stdout="", stderr="")
        provider = _Gated(min_version="1.0.0", version_cmd=(str(binary), "--version"))
        completed = executor._run_probe(list(provider.version_cmd))

        parsed = provider.parse_version(completed.stdout, completed.stderr)
        assert parsed is None
        assert version_verdict(provider, parsed).refuses is False


# --------------------------------------------------------------------------- #
# The ceiling as configuration
# --------------------------------------------------------------------------- #


class TestTheCeilingIsReachableFromConfig:
    """`providers.<name>.max_version_exclusive` overrides the class declaration."""

    def test_the_declared_default_is_no_ceiling(self):
        assert LLMProvider.max_version_exclusive == ""

    def test_config_supplies_a_ceiling_the_adapter_does_not_declare(self):
        provider = _Gated()
        with patch.object(
            base_module, "_provider_block",
            lambda name, anchor=None: {"max_version_exclusive": "2.0.0"},
        ):
            assert version_ceiling(provider) == "2.0.0"
            assert version_verdict(provider, "2.1.0").state == VERSION_TOO_NEW

    def test_config_wins_over_the_class_declaration(self):
        provider = _Gated(max_version_exclusive="9.0.0")
        with patch.object(
            base_module, "_provider_block",
            lambda name, anchor=None: {"max_version_exclusive": "2.0.0"},
        ):
            assert version_ceiling(provider) == "2.0.0"

    def test_a_blank_or_unusable_config_value_is_no_ceiling(self):
        provider = _Gated()
        for value in ("", "   ", None, [], {}):
            with patch.object(
                base_module, "_provider_block",
                lambda name, anchor=None, _v=value: {"max_version_exclusive": _v},
            ):
                assert version_ceiling(provider) == ""

    def test_a_non_numeric_ceiling_is_dropped_rather_than_refusing_everything(self):
        """The footgun, closed where the value is read.

        `version_tuple` maps a value with no leading digits to `(0,)`, which sits
        *below* every real build — so a typo compared naively would refuse the
        whole provider. YAML 1.1 resolving a bare `on`/`yes` to a bool is exactly
        how such a value arrives.
        """
        base_module.reset_env_policy_warnings()
        provider = _Gated()
        for value in ("latest", True, "vNext"):
            with patch.object(
                base_module, "_provider_block",
                lambda name, anchor=None, _v=value: {"max_version_exclusive": _v},
            ):
                assert version_ceiling(provider) == ""
                assert version_verdict(provider, "1.2.3").state == VERSION_OK
        base_module.reset_env_policy_warnings()

    def test_a_decorated_ceiling_is_normalized_to_its_numeric_core(self):
        """`v2.0.0`, `2.0.0-rc1` and a build-suffixed date all compare as numbers."""
        base_module.reset_env_policy_warnings()
        provider = _Gated()
        for value, expected in (
            ("v2.0.0", "2.0.0"),
            ("2.0.0-rc1", "2.0.0"),
            ("  2.0.0  ", "2.0.0"),
        ):
            with patch.object(
                base_module, "_provider_block",
                lambda name, anchor=None, _v=value: {"max_version_exclusive": _v},
            ):
                assert version_ceiling(provider) == expected
                assert version_verdict(provider, "1.2.3").state == VERSION_OK
                assert version_verdict(provider, "2.0.0").state == VERSION_TOO_NEW
        with patch.object(
            base_module, "_provider_block",
            lambda name, anchor=None: {"max_version_exclusive": "2026.08.11-e8db854"},
        ):
            assert version_ceiling(provider) == "2026.08.11"
        base_module.reset_env_policy_warnings()

    def test_a_ceiling_that_merely_contains_a_version_is_dropped(self):
        """The footgun one step along: a partial match may not become the ceiling.

        `release-2.0` carries a dotted-numeric token, so a substring check passes
        it — and the string it would return parses as `(0, 0)`, below every real
        build, refusing the whole provider. Only a value that matches the version
        format *whole* becomes a ceiling; the rest warn once and mean "no
        ceiling", exactly like `latest`.
        """
        base_module.reset_env_policy_warnings()
        provider = _Gated()
        for value in ("release-2.0", "1.x", "2.0.0 or later", "build 2.0.0"):
            with patch.object(
                base_module, "_provider_block",
                lambda name, anchor=None, _v=value: {"max_version_exclusive": _v},
            ):
                assert version_ceiling(provider) == ""
                assert version_verdict(provider, "1.2.3").state == VERSION_OK
                assert version_verdict(provider, "9.9.9").state == VERSION_OK
        base_module.reset_env_policy_warnings()

    def test_the_base_config_ships_the_key_discoverably_and_inert(self):
        """One shipped provider writes the key out, empty, so a user can find it.

        The same posture L7's `fallback_models` holds: present in base means
        discoverable without reading code, and empty means today's behaviour,
        byte for byte. It is deliberately **not** asserted for every provider —
        an unmeasured ceiling is not a row worth carrying.
        """
        blocks = load_base_config()["providers"]
        declaring = {
            name: block.get("max_version_exclusive")
            for name, block in blocks.items()
            if "max_version_exclusive" in block
        }
        assert declaring, (
            "no provider block declares `max_version_exclusive`, so the ceiling "
            "is code-only and a user cannot pin a broken vendor build without "
            "editing an adapter"
        )
        for name, value in declaring.items():
            assert not (value or ""), (
                f"providers.{name}.max_version_exclusive ships {value!r}: a live "
                f"ceiling refuses runs, so it may only ship for a build that was "
                f"measured to be broken"
            )

    @pytest.mark.parametrize("name", _registry_names())
    def test_no_shipped_adapter_refuses_its_own_verified_build(self, name):
        """A declared ceiling below the fixture build would refuse every run."""
        provider = _PROVIDERS[name]
        ceiling = version_ceiling(provider)
        if not ceiling:
            return
        verified = (provider.verified_version or "").strip()
        floor = (provider.min_version or "").strip()
        if floor:
            assert version_tuple(floor) < version_tuple(ceiling), (
                f"{name}: floor {floor} is not below ceiling {ceiling}"
            )
        if verified:
            assert version_tuple(verified) < version_tuple(ceiling), (
                f"{name}: the build the fixtures were captured from ({verified}) "
                f"is at or above the ceiling {ceiling}"
            )


# --------------------------------------------------------------------------- #
# §5.4.1 — the absorbed per-flag gate
# --------------------------------------------------------------------------- #


class TestTheSharedFlagGate:
    """One help read, one cache, two postures — and no adapter-local copy."""

    def _installed(self, monkeypatch, binary: Path):
        monkeypatch.setenv(
            "PATH", str(binary.parent) + os.pathsep + os.environ.get("PATH", "")
        )

    def test_the_tokens_come_from_the_binarys_own_help(self, tmp_path):
        binary = _write_cli(
            tmp_path / "probe-cli",
            stdout="Options:\n  --trust  trust it\n  --mode <m>  the mode\n",
        )
        tokens = read_help_tokens(str(binary))
        assert {"--trust", "--mode"} <= tokens

    def test_both_help_streams_are_read_concatenated(self, tmp_path):
        """§5.4's rule applies to the help read too, not only to `--version`."""
        binary = _write_cli(
            tmp_path / "probe-cli",
            stdout="Usage: probe-cli\n",
            stderr="  --only-on-stderr  yes\n",
        )
        assert "--only-on-stderr" in read_help_tokens(str(binary))

    def test_a_flag_is_matched_as_a_whole_word(self, tmp_path):
        """`--fallback-model-chain` does not confirm `--fallback-model`, and
        `--model` does not confirm `--mode`. Confirming the wrong flag is the
        failure this gate exists to prevent."""
        binary = _write_cli(
            tmp_path / "probe-cli",
            stdout="  --fallback-model-chain <list>\n  --model <m>\n",
        )
        tokens = read_help_tokens(str(binary))
        assert "--fallback-model-chain" in tokens and "--model" in tokens
        assert "--fallback-model" not in tokens
        assert "--mode" not in tokens

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"exit_code": 2},          # non-zero exit
            {"stdout": ""},            # empty help
            {"stdout": "   \n"},       # whitespace-only help
        ],
    )
    def test_an_unreadable_help_is_unsupported_never_a_crash(self, tmp_path, kwargs):
        binary = _write_cli(tmp_path / "probe-cli", **kwargs)
        assert read_help_tokens(str(binary)) == frozenset()

    def test_a_missing_binary_is_unsupported(self, tmp_path):
        assert read_help_tokens(str(tmp_path / "never-created")) == frozenset()

    def test_one_help_spawn_per_binary_per_process(self, tmp_path):
        counter = tmp_path / "spawns"
        binary = _write_cli(
            tmp_path / "probe-cli", stdout="  --flag\n", counter=counter
        )
        for _ in range(5):
            assert "--flag" in read_help_tokens(str(binary))
        assert counter.read_text().strip() == "1", (
            "the sniff must be memoised for the process, not re-spawned per "
            "question — a 4-model x 10-phase run would spawn 40 of them"
        )

    def test_the_help_argv_is_part_of_the_cache_key(self, tmp_path):
        """A flag on a subcommand's help is a different read of the same binary."""
        counter = tmp_path / "spawns"
        binary = _write_cli(
            tmp_path / "probe-cli", stdout="  --flag\n", counter=counter
        )
        read_help_tokens(str(binary))
        read_help_tokens(str(binary), help_argv=("run", "--help"))
        assert counter.read_text().strip() == "2"

    def test_the_optional_posture_declines_an_unconfirmed_flag(
        self, tmp_path, monkeypatch
    ):
        binary = _write_cli(tmp_path / "probe-cli", stdout="  --something-else\n")
        self._installed(monkeypatch, binary)
        provider = _Gated(version_cmd=("probe-cli", "--version"))

        assert provider.supports_flag("--wanted") is False
        assert provider.supports_flag("--something-else") is True

    def test_the_optional_posture_declines_when_there_is_no_binary(self, monkeypatch):
        """Undecidable ⇒ False: an unrecognised flag is a dead slot, not a degrade."""
        monkeypatch.setattr("shutil.which", lambda _n: None)
        assert _Gated(version_cmd=("probe-cli", "--version")).supports_flag("--x") is False

    def test_the_load_bearing_posture_reports_exactly_what_is_missing(
        self, tmp_path, monkeypatch
    ):
        binary = _write_cli(tmp_path / "probe-cli", stdout="  --trust\n")
        self._installed(monkeypatch, binary)
        provider = _Gated(version_cmd=("probe-cli", "--version"))

        assert provider.missing_flags(("--trust", "--mode")) == ("--mode",)
        assert provider.missing_flags(("--trust",)) == ()

    def test_the_load_bearing_posture_refuses_nothing_without_a_binary(
        self, monkeypatch
    ):
        """The asymmetry, stated: an absent CLI is `is_available()`'s refusal.

        Turning "not installed" into "version-too-low" would replace a precise
        message with a misleading one, and a binary that cannot be spawned
        cannot reject a flag either.
        """
        monkeypatch.setattr("shutil.which", lambda _n: None)
        provider = _Gated(version_cmd=("probe-cli", "--version"))
        assert provider.missing_flags(("--trust", "--mode")) == ()

    def test_an_unspawnable_resolved_path_also_refuses_nothing(
        self, tmp_path, monkeypatch
    ):
        ghost = tmp_path / "probe-cli"  # never created
        monkeypatch.setattr("shutil.which", lambda _n: str(ghost))
        provider = _Gated(version_cmd=("probe-cli", "--version"))
        assert provider.missing_flags(("--trust",)) == ()
        # …and the optional form still declines, because it is undecidable.
        assert provider.supports_flag("--trust") is False

    def test_the_gate_binary_comes_from_a_declaration(self, monkeypatch):
        """`version_cmd[0]`, then `auth_check[0]` — never a literal in the gate."""
        assert _Gated(version_cmd=("from-version", "--version")).gate_binary_name() == (
            "from-version"
        )
        assert _Gated(auth_check=("from-auth", "status")).gate_binary_name() == "from-auth"
        assert _Gated().gate_binary_name() == ""

    def test_an_adapter_declaring_no_binary_never_spawns(self, monkeypatch):
        monkeypatch.setattr(
            base_module, "read_help_tokens",
            lambda *a, **k: pytest.fail("no probe may run without a declared binary"),
        )
        assert _Gated().supports_flag("--x") is False
        assert _Gated().missing_flags(("--x",)) == ()

    def test_the_probe_env_is_the_shared_builder_plus_the_rendering_pins(self):
        """§5.4.1's containment clause: same env builder, full-width help."""
        env = _Gated().gate_probe_env()
        assert env["TERM"] == "dumb"
        assert env["NO_COLOR"] == "1"
        assert int(env["COLUMNS"]) >= 200, (
            "a truncated help rendering is the failure mode §3.6 was bitten by"
        )

    def test_the_probe_applies_the_adapters_own_removals(self, monkeypatch):
        """A probe run from inside a nested session must not be refused by it."""
        monkeypatch.setenv("SOME_NESTING_GUARD", "1")
        provider = _Gated()
        provider.get_remove_env = lambda: ["SOME_NESTING_GUARD"]  # type: ignore[method-assign]
        assert "SOME_NESTING_GUARD" not in provider.gate_probe_env()


class TestTheExemptionIsDeclaredNotAssumed:
    """§5.4.1(a): a flag verified functionally at the floor takes no probe."""

    def test_an_exempt_flag_is_supported_without_any_probe(self, monkeypatch):
        monkeypatch.setattr(
            base_module, "read_help_tokens",
            lambda *a, **k: pytest.fail("an exempt flag must not spawn a probe"),
        )
        provider = _Gated(
            version_cmd=("probe-cli", "--version"),
            flags_verified_at_floor=("--ephemeral",),
        )
        assert provider.supports_flag("--ephemeral") is True

    def test_an_exempt_flag_is_never_reported_missing(self, tmp_path, monkeypatch):
        binary = _write_cli(tmp_path / "probe-cli", stdout="  --other\n")
        monkeypatch.setenv(
            "PATH", str(binary.parent) + os.pathsep + os.environ.get("PATH", "")
        )
        provider = _Gated(
            version_cmd=("probe-cli", "--version"),
            flags_verified_at_floor=("--ephemeral",),
        )
        assert provider.missing_flags(("--ephemeral", "--absent")) == ("--absent",)

    def test_the_default_is_no_exemption(self):
        assert LLMProvider.flags_verified_at_floor == ()

    @pytest.mark.parametrize("name", _registry_names())
    def test_an_exemption_requires_a_declared_floor_to_be_verified_at(self, name):
        """"Verified at the declared min_version" needs a declared min_version.

        An adapter claiming the exemption with no floor is claiming something
        about a range it never named, which is the prose this declaration
        replaced.
        """
        provider = _PROVIDERS[name]
        if not provider.flags_verified_at_floor:
            return
        assert (provider.min_version or "").strip(), (
            f"{name} declares flags_verified_at_floor="
            f"{provider.flags_verified_at_floor} but no min_version to have "
            f"verified them at"
        )
        for flag in provider.flags_verified_at_floor:
            assert flag.startswith("--"), f"{name}: {flag!r} is not a long option"


class TestTheOneOffsAreGone:
    """The absorption, asserted structurally rather than by reading the diff."""

    def test_no_adapter_keeps_a_private_help_sniff_of_the_absorbed_three(self):
        """L7/M7 (claude) and M1 (cursor) hand their gate to the shared one.

        Named by module rather than by roster on purpose: these are the three
        §5.4.1 one-offs M14 was scheduled to absorb, and this asserts the
        absorption happened. A *fourth* adapter growing its own copy is a
        different question, answered by the next test.
        """
        from utils.providers import claude_code, cursor_agent

        for module in (claude_code, cursor_agent):
            for attribute in ("_read_help_tokens", "_probe_help_tokens",
                              "_help_flag_tokens", "_gateable_binary"):
                assert not hasattr(module, attribute), (
                    f"{module.__name__}.{attribute} is back — the gate has one "
                    f"implementation, in providers/base.py"
                )
        # `supports_flag` is the base's, not an override.
        assert (
            claude_code.ClaudeCodeProvider.supports_flag is LLMProvider.supports_flag
        )
        assert (
            cursor_agent.CursorAgentProvider.missing_flags is LLMProvider.missing_flags
        )

    @pytest.mark.parametrize("name", _registry_names())
    def test_no_adapter_overrides_the_shared_gate(self, name):
        """The gate is implemented once, so no adapter may re-implement it.

        Parametrized over the live registry (M0.a) so an adapter added after
        M14 cannot quietly reopen the seam the task closed.
        """
        provider = _PROVIDERS[name]
        for method in ("supports_flag", "missing_flags", "gate_probe_env",
                       "flag_tokens", "gate_binary"):
            assert getattr(type(provider), method) is getattr(LLMProvider, method), (
                f"{name} overrides {method}: M14's whole point is that there is "
                f"one gate. Declare `flags_verified_at_floor` or `help_argv` "
                f"instead of re-implementing the probe."
            )


# --------------------------------------------------------------------------- #
# The preflight still behaves exactly as it did (green stays green)
# --------------------------------------------------------------------------- #


class TestThePreflightRefusalIsUnchanged:
    """The executor's D3m behaviour, now computed by the shared verdict."""

    def _detect(self, provider_name, monkeypatch, streams):
        executor.reset_version_cache()
        monkeypatch.setattr(
            executor, "_run_probe",
            lambda argv: subprocess.CompletedProcess(argv, 0, *streams),
        )
        return executor._detect_version(
            provider_name, key="driver.subagent_runner",
            spec=f"{provider_name}:m", profile="agent-readonly",
        )

    def test_a_build_below_the_floor_is_refused_before_any_child(self, monkeypatch):
        name = next(
            n for n, p in sorted(_PROVIDERS.items())
            if (p.min_version or "").strip() and p.version_cmd
        )
        with pytest.raises(executor.RunnerRejected) as excinfo:
            self._detect(name, monkeypatch, ("0.0.1", ""))
        assert "0.0.1" in excinfo.value.message
        assert _PROVIDERS[name].min_version in excinfo.value.message

    def test_a_version_only_on_stderr_still_reaches_the_refusal(self, monkeypatch):
        """The stderr case through the *real* call path, not just `parse_version`."""
        name = next(
            n for n, p in sorted(_PROVIDERS.items())
            if (p.min_version or "").strip() and p.version_cmd
        )
        with pytest.raises(executor.RunnerRejected) as excinfo:
            self._detect(name, monkeypatch, ("", "0.0.1"))
        assert "0.0.1" in excinfo.value.message

    def test_a_configured_ceiling_refuses_through_the_same_path(self, monkeypatch):
        name = next(n for n, p in sorted(_PROVIDERS.items()) if p.version_cmd)
        monkeypatch.setattr(
            base_module, "_provider_block",
            lambda block_name, anchor=None: {"max_version_exclusive": "2.0.0"},
        )
        with pytest.raises(executor.RunnerRejected) as excinfo:
            self._detect(name, monkeypatch, ("99.0.0", ""))
        assert "2.0.0" in excinfo.value.message and "99.0.0" in excinfo.value.message

    def test_an_unreadable_version_records_unknown_and_never_blocks(
        self, monkeypatch, capsys
    ):
        name = next(n for n, p in sorted(_PROVIDERS.items()) if p.version_cmd)
        assert self._detect(name, monkeypatch, ("no version here", "")) == "unknown"
        assert "could not determine" in capsys.readouterr().err

    def test_above_verified_warns_once_and_dispatches(self, monkeypatch, capsys):
        name = next(
            n for n, p in sorted(_PROVIDERS.items())
            if (p.verified_version or "").strip() and p.version_cmd
            and not (p.max_version_exclusive or "").strip()
        )
        assert self._detect(name, monkeypatch, ("99.0.0", "")) == "99.0.0"
        err = capsys.readouterr().err
        assert "newer than the release" in err
        assert err.startswith("multi-llm warning: ")
