"""Tests for the omp adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import LLMProvider
from utils.providers import base as base_module, omp as omp_module
from utils.providers.omp import (
    OMP_APPROVAL_MODE,
    OMP_MIN_MAX_TIME_S,
    OMP_REQUIRED_FLAGS,
    OMP_SHUTDOWN_MARGIN_S,
    OMP_TOOLS,
    OVERLAY_PATH as OMP_OVERLAY_PATH,
    OmpFlagsUnconfirmed,
    OmpProvider,
    max_time_value,
)


class TestOmpProvider:
    """oh-my-pi (`omp`), the adapter that introduced the stdin default.

    What this class grades is everything about omp that is *not* a captured
    stream: identity, the setup metadata, the rendered argv, the flag gate that
    refuses to render one, the bypass the guard has to be able to see, the
    file-output declaration and the env delta. The parsing rows — every
    `stopReason`, the denial, the usage sums, `vendor_text`'s reduction — are
    graded from committed bytes under `tests/fixtures/omp/` by the §6.2 manifest
    rig, because a parser graded against strings written beside it is graded
    against its author's memory of the stream.
    """

    MODEL = "openrouter/openai/gpt-5.6-luna"
    PROMPT = "Analyze this plan carefully"

    @pytest.fixture
    def provider(self):
        return OmpProvider()

    @pytest.fixture
    def ungated(self):
        """Render argv from the adapter, never from the developer's PATH.

        omp's §2.11 gate reads the installed binary's `--help` before
        `build_command` will emit anything, which would otherwise make every
        assertion below depend on whether this machine happens to have omp. The
        shared gate resolves its binary through `LLMProvider.gate_binary`, so
        neutralizing that one lookup is enough: with no binary to read,
        `missing_flags` reports nothing missing and the argv is the one the
        adapter would render. The gate itself is graded separately, with the
        help text stubbed rather than the lookup.
        """
        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None):
            yield

    # ── identity ─────────────────────────────────────────────────────────
    def test_identity(self, provider):
        assert provider.name == "omp"
        assert provider.default_timeout == 600
        assert provider.version_cmd == ("omp", "--version")
        assert provider.verified_version == "18.8.3"
        assert provider.usage_support == base_module.USAGE_FULL

    def test_availability_is_the_binarys_own_name_on_path(self, provider):
        """`omp` is both the provider key and the binary, so there is no
        `BINARY_TO_CLI` row to keep in step with this."""
        with patch("shutil.which", return_value="/usr/local/bin/omp") as which:
            assert provider.is_available() is True
        assert which.call_args[0][0] == "omp"

        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_no_version_floor_is_declared_and_none_is_owed(self, provider):
        """A measured-at version is an upper bound on the floor, never the floor.

        The upstream release notes mention every flag this argv needs without
        dating any of their introductions, so a `min_version` here would be a
        number nobody measured. What the adapter needs is not a version but the
        flags, and `OMP_REQUIRED_FLAGS` refuses by the NAME of the one a build
        lacks. `flags_verified_at_floor` stays empty for the same reason: an
        exemption from the gate requires a declared floor to have been verified
        at, and there is none.
        """
        assert provider.min_version == ""
        assert provider.max_version_exclusive == ""
        assert provider.flags_verified_at_floor == ()

    def test_the_declared_transports(self, provider):
        assert provider.supported_prompt_transports == ("argv", "stdin")
        assert provider.prompt_transport == "stdin"
        assert provider.prompt_argv_index == -1
        assert provider.structured_output_support is False

    # ── setup metadata (§2.1) ────────────────────────────────────────────
    def test_no_auth_probe_is_declared_and_none_is_owed(self, provider):
        """omp has no auth-status subcommand, measured on `omp --help` at 18.2.2.

        The three credential-adjacent commands it does have are each
        disqualified for the probe's job: `omp token <provider>` PRINTS the
        credential, so a probe whose success output is the secret would put the
        key into the captured streams a doctor row records; `omp usage` is a
        live usage-limit snapshot over the network for OAuth accounts (a network
        fact, not a credential read); and `omp auth-broker status` reports the
        credential-vault daemon, not whether an OpenRouter key resolves. A fix
        is owed only where a probe is declared, so declaring none is what makes
        `login_cmd` empty rather than missing.
        """
        assert provider.auth_check == ()
        assert provider.login_cmd == ""
        assert provider.setup_probe == ()

        hints = provider.describe_setup()
        assert hints.status_args == ()
        assert hints.login_args == ()

    def test_the_auth_hint_names_the_credential_and_where_it_lives(self, provider):
        hint = provider.describe_setup().auth_hint

        assert "OPENROUTER_API_KEY" in hint
        assert "~/.omp/agent/.env" in hint

    def test_the_install_hint_names_the_measured_route_and_not_the_npm_one(
        self, provider
    ):
        """The npm package needs Bun ≥ 1.3.14 and does not run under node, which
        is why the hint names the release asset and its checksum file instead —
        and why the container lane's `npm ci` layer could never have carried it.
        """
        hint = provider.describe_setup().install_hint

        assert "omp-linux-x64" in hint
        assert "SHA256SUMS.txt" in hint
        assert "npm" not in hint

    @pytest.mark.parametrize("field", ["auth_hint", "install_hint"])
    def test_a_hint_is_one_stripped_line_carrying_nothing_from_this_machine(
        self, provider, field
    ):
        """Both hints are rendered into `--init` output, a log and a failure
        artifact, so neither may carry a value read from this process's
        environment or an absolute home path. The 12-character floor is what
        keeps the sweep about credentials rather than about `LANG`: shorter than
        that a value is not a secret, and longer than that its appearance here
        would be a leak.
        """
        hint = getattr(provider.describe_setup(), field)

        assert hint == hint.strip() and "\n" not in hint
        assert "/home/" not in hint
        # A value the adapter's own source spells out is a constant, not a
        # read: a GitHub runner exports GITHUB_SERVER_URL=https://github.com,
        # which the install hint's public releases URL contains (first CI run
        # of this history, 2026-10-23).
        import inspect

        source = inspect.getsource(type(provider))
        for value in os.environ.values():
            if len(value) >= 12 and value not in source:
                assert value not in hint

    def test_doctors_auth_row_is_unprobed_and_spawns_nothing(self, provider):
        """`unprobed` / ok, and the `_run` tripwire is the half that matters.

        A missing key is not silently assumed good and it is not probed for
        either: it surfaces one dispatch later as omp's own pre-flight `No API
        key found for openrouter.` (exit 1, 0 B stdout, zero requests), which
        `auth_error_patterns` classifies as `auth_failed`. `doctor` is the one
        command allowed to spawn a provider CLI outside a phase, so an adapter
        that declares no probe must cost it no spawn at all.
        """
        from utils import doctor

        def boom(*_args, **_kwargs):
            raise AssertionError("doctor must not spawn anything for omp")

        with patch.object(doctor, "_run", boom):
            row = doctor._probe_auth(provider, timeout=1.0)

        assert row["state"] == "unprobed"
        assert row["detail"] == "no auth_check declared"
        assert row["ok"] is True

    def test_the_executor_preflight_answers_unknown_for_the_same_reason(self):
        """`unknown` is the honest answer for a CLI that declares no probe —
        never an assumption of success, and never a refusal either."""
        from utils import subagent_executor

        subagent_executor._AUTH_CACHE.pop("omp|1", None)
        subagent_executor._AUTH_CACHE.pop("omp|0", None)
        try:
            state = subagent_executor._detect_auth(
                "omp",
                self.MODEL,
                key="k",
                spec=f"omp:{self.MODEL}",
                profile=base_module.PROFILE_AGENT_READONLY,
            )
        finally:
            subagent_executor._AUTH_CACHE.pop("omp|1", None)
            subagent_executor._AUTH_CACHE.pop("omp|0", None)

        assert state == "unknown"

    # ── argv (§2.2) ──────────────────────────────────────────────────────
    def test_build_command_renders_the_measured_argv_in_the_measured_order(
        self, provider, ungated
    ):
        """The order is fixed as written so `expected_argv.yaml`,
        `cli-surface.yaml` and this test agree on ONE rendering."""
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv == [
            "omp", "--no-session", "-p",
            "--mode", "json",
            "--config", str(OMP_OVERLAY_PATH),
            "--approval-mode", "yolo",
            "--tools", "read,grep,glob,bash,edit,write",
            "--cwd", os.getcwd(),
            "--model", self.MODEL,
            "--", self.PROMPT,
        ]

    def test_the_policy_tokens_are_the_adapters_own_constants(self, provider):
        """Spelled out above AND read from the constants here, so a narrowing of
        either one still has to face the literal."""
        assert OMP_APPROVAL_MODE == "yolo"
        assert OMP_TOOLS == (
            "read", "grep", "glob", "bash", "edit", "write"
        )

    @pytest.mark.parametrize(
        "rung", ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_every_rung_of_the_ladder_renders_thinking_before_model(
        self, provider, ungated, rung
    ):
        """Seven rungs, `off` where pi spells it `none`.

        omp does NOT validate `--thinking` — `none` and `ultra` are accepted
        silently and ignored — so this whitelist is the only guard there is, and
        the suffix has to be stripped from the model before `--model` or omp's
        own native `:level` parsing would see it twice.
        """
        argv = provider.build_command(self.PROMPT, f"{self.MODEL}:{rung}")

        assert "--thinking" in argv
        assert argv[argv.index("--thinking") + 1] == rung
        assert argv.index("--thinking") < argv.index("--model")
        assert argv[argv.index("--model") + 1] == self.MODEL

    def test_no_suffix_renders_no_thinking_flag(self, provider, ungated):
        assert "--thinking" not in provider.build_command(self.PROMPT, self.MODEL)

    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/z-ai/glm-5.2:free",
            "bedrock/anthropic.claude-v1:0",
            "openrouter/openai/gpt-5.6-luna:auto",
        ],
    )
    def test_a_suffix_that_is_not_a_rung_stays_in_the_model(
        self, provider, ungated, model
    ):
        """`:free` ids, bedrock `…-v1:0` version suffixes and omp's own eighth
        level `:auto` all pass through as part of the model string — which is
        exactly how `omp:<model>:auto` keeps working without `auto` being
        declared a rung it is not.
        """
        argv = provider.build_command(self.PROMPT, model)

        assert argv[argv.index("--model") + 1] == model
        assert "--thinking" not in argv

    @pytest.mark.parametrize(
        "timeout_s,expected",
        [(None, None), (1200, "1195"), (6, "1"), (5, None), (25, "20")],
    )
    def test_the_max_time_arithmetic(self, timeout_s, expected):
        """qwen's FORMULA, never qwen's helper: outer deadline minus a shutdown
        margin, `None` in ⇒ `None` out so the flag is absent rather than
        defaulted. `6 → 1` is the floor: `OMP_MIN_MAX_TIME_S` is 1 because
        `--max-time 1` was measured cutting cleanly, and copying qwen's 60 s
        minimum would render a `--max-time` larger than a short slice's own
        deadline and turn a clean `aborted` into a SIGTERM.
        """
        assert max_time_value(timeout_s) == expected
        assert OMP_SHUTDOWN_MARGIN_S == 5
        assert OMP_MIN_MAX_TIME_S == 1

    def test_a_deadline_above_the_margin_renders_the_difference(
        self, provider, ungated
    ):
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=1200)

        assert "--max-time" in argv
        assert argv[argv.index("--max-time") + 1] == "1195"
        assert argv.index("--max-time") < argv.index("--cwd")

    def test_an_undersized_deadline_omits_the_flag_and_warns_once(
        self, provider, ungated, capsys
    ):
        """The undersized case is an OMISSION, not qwen's clamp-and-render.

        The retry ladder produces a sub-margin `timeout_s` legitimately — a
        retry's deadline is the slot's REMAINING wall clock, down to a 1.0 s
        floor — and neither of the other two answers is available: clamping
        would put a `--max-time` larger than the outer deadline on argv, which
        is a token that cannot do its job, and refusing would turn the ladder's
        own short slices into failed slots, because a `build_command` raise
        escapes as the slot's error with no retry. So the flag is dropped and
        the argv is BYTE-IDENTICAL to the `timeout_s=None` rendering, with the
        parent's `subprocess.run` deadline left as the only bound.
        """
        capsys.readouterr()
        argv = provider.build_command(
            self.PROMPT, self.MODEL, timeout_s=OMP_SHUTDOWN_MARGIN_S
        )
        warning = capsys.readouterr().err

        assert argv == provider.build_command(self.PROMPT, self.MODEL, timeout_s=None)
        assert "--max-time" not in argv
        assert warning.count("WARNING") == 1
        assert "omp" in warning
        assert "qwen" not in warning

    @pytest.mark.parametrize("timeout_s", [1, 2, 3, 4, 5])
    def test_nothing_is_clamped_upward(self, provider, ungated, timeout_s):
        """A slice too short for the margin renders no bound at all; it never
        renders one LONGER than the deadline it is supposed to sit inside."""
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=timeout_s)

        assert "--max-time" not in argv

    def test_a_slice_too_short_for_the_retry_chain_is_not_undersized(
        self, provider, ungated
    ):
        """`--max-time` bounds the dispatch; it does not promise omp's own
        twelve-request 429 chain fits inside it. A 25 s slot renders 20 s
        exactly, and the slot had no time for that chain whichever way it ended.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL, timeout_s=25)

        assert argv[argv.index("--max-time") + 1] == "20"

    def test_the_overlay_is_absolute_and_ships_with_the_skill(
        self, provider, ungated
    ):
        """A relative `--config` resolves against `--cwd`, not the launch cwd,
        and a missing file is a fast named launch failure (exit 1 in 0.13 s,
        zero requests) — so the path is resolved absolutely here and the file
        has to be in the tree the skill ships.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL)
        rendered = Path(argv[argv.index("--config") + 1])

        assert rendered.is_absolute()
        assert rendered == OMP_OVERLAY_PATH
        assert rendered.is_file()

    def test_cwd_is_rendered_on_every_dispatch(self, provider, ungated):
        """Not a tidiness flag, and not conditional on the spawn directory.

        Launched from `$HOME` without it omp silently relocates to `/tmp` —
        nothing on either channel says so — and a reviewer that moved to `/tmp`
        reads none of the repo's files, has an empty `git diff`, writes
        `{output_json_path}` into `/tmp`, and still exits 0 with a
        plausible-looking answer. The obvious alternative is worse:
        `--allow-home` keeps the cwd AND auto-mounts the MCP servers from
        `~/.cursor/mcp.json`, so it is rendered on no profile.
        """
        argv = provider.build_command(self.PROMPT, self.MODEL, cwd="/srv/project")

        assert argv[argv.index("--cwd") + 1] == "/srv/project"
        assert "--allow-home" not in argv

    def test_cwd_defaults_to_the_process_directory(self, provider, ungated):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[argv.index("--cwd") + 1] == os.getcwd()

    def test_the_stdin_rendering_is_the_argv_minus_its_last_two_tokens(
        self, provider, ungated
    ):
        """That `[:-2]` is omp's own override and not the base helper's `[:-1]`:
        dropping only the prompt would leave a bare `--` on argv."""
        argv = provider.build_command(self.PROMPT, self.MODEL)
        stdin_argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "stdin"
        )

        assert argv[-2:] == ["--", self.PROMPT]
        assert stdin_argv == argv[:-2]
        assert stdin_text == self.PROMPT
        assert stdin_argv[-2:] == ["--model", self.MODEL]

    def test_the_argv_transport_still_renders_the_prompt_last(
        self, provider, ungated
    ):
        argv, stdin_text = provider.build_command_for_transport(
            self.PROMPT, self.MODEL, "argv"
        )

        assert stdin_text is None
        assert argv == provider.build_command(self.PROMPT, self.MODEL)

    def test_an_unsupported_transport_is_refused_by_name(self, provider, ungated):
        with pytest.raises(ValueError) as excinfo:
            provider.build_command_for_transport(
                self.PROMPT, self.MODEL, "prompt-file"
            )

        assert "prompt-file" in str(excinfo.value)

    @pytest.mark.parametrize("prompt", ["@plans/auth.md", "--help me review this"])
    def test_a_sigil_prefixed_prompt_is_literal_on_stdin(
        self, provider, ungated, prompt
    ):
        """On stdin every prompt is literal whatever its first character: the
        bytes never meet a flag parser or omp's `@` attachment at all, which is
        why this transport is the adapter's default. The two argv tests below are
        the other half — there the same two sigils are made literal by `--`."""
        argv, stdin_text = provider.build_command_for_transport(
            prompt, self.MODEL, "stdin"
        )

        assert stdin_text == prompt
        assert prompt not in argv

    def test_a_dash_prefixed_prompt_is_accepted_on_argv_too(
        self, provider, ungated
    ):
        """`--` ends flag parsing, so a positional beginning with `-` is text."""
        prompt = "--help me review this"
        argv = provider.build_command(prompt, self.MODEL)

        assert argv[-2:] == ["--", prompt]

    def test_an_at_prefixed_prompt_is_accepted_on_argv_behind_the_dash_dash(
        self, provider, ungated
    ):
        """An `@`-prefixed prompt is a prompt, and `--` is what keeps it one.

        Measured at 18.2.2 in both directions: after `--`, `@atfile.txt` reaches
        the model as that literal text and the file is not read; the same prompt
        without `--` has the file's contents inlined into the user message. This
        adapter's argv always ends `["--", prompt]`, so the prompt is accepted
        rather than refused — and the two positional assertions are the
        guarantee, not decoration: tidying the `--` away would turn every
        `@`-prefixed prompt into a silent file read, and it fails here first.
        """
        prompt = "@plan.md"
        argv, stdin_text = provider.build_command_for_transport(
            prompt, self.MODEL, "argv"
        )

        assert stdin_text is None
        assert argv[-1] == prompt
        assert argv[-2] == "--"

    # ── the flag gate (§2.11) ────────────────────────────────────────────
    @staticmethod
    def _with_help(tokens):
        """The gate reading a help text this test wrote, on a binary that
        resolves. Both halves are patched because the gate needs a spawnable
        path before it will read anything, and a machine that happens to have
        omp installed must not be what decides the result.
        """
        return (
            patch.object(LLMProvider, "gate_binary", lambda self, **_kw: sys.executable),
            patch.object(
                base_module, "read_help_tokens", lambda _b, **_kw: frozenset(tokens)
            ),
        )

    def test_a_help_text_missing_one_flag_refuses_by_that_flags_name(self, provider):
        """`--mode` has no safe degraded form: without it the parser reads a text
        stream, so the gate is the load-bearing form and not the optional one.
        """
        tokens = set(OMP_REQUIRED_FLAGS) - {"--mode"}
        binary, help_text = self._with_help(tokens)
        with binary, help_text, pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            provider.build_command(self.PROMPT, self.MODEL)

        refusal = excinfo.value
        assert refusal.missing == ("--mode",)
        assert refusal.state == base_module.VERSION_TOO_LOW
        assert "--mode" in str(refusal)

    def test_an_empty_help_names_every_required_flag(self, provider):
        """An empty help is the undecidable case, and the load-bearing form
        declines toward refusing the SLOT rather than toward a degraded argv:
        without `--tools` or `--approval-mode` the run inherits omp's factory
        `yolo` over every tool it has, and without `--config` the retry floor is
        unbounded.
        """
        binary, help_text = self._with_help(())
        with binary, help_text, pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            provider.build_command(self.PROMPT, self.MODEL)

        assert excinfo.value.missing == OMP_REQUIRED_FLAGS
        for flag in OMP_REQUIRED_FLAGS:
            assert flag in str(excinfo.value)

    def test_p_is_confirmed_through_its_long_spelling(self, provider):
        """The shared token regex reads long options only, so `-p` is gated
        through `--print` — the long spelling on the same help line — while the
        argv keeps `-p`."""
        assert "--print" in OMP_REQUIRED_FLAGS
        assert "-p" not in OMP_REQUIRED_FLAGS

        binary, help_text = self._with_help(OMP_REQUIRED_FLAGS)
        with binary, help_text:
            argv = provider.build_command(self.PROMPT, self.MODEL)

        assert "-p" in argv
        assert "--print" not in argv

    def test_a_flag_this_call_does_not_emit_is_never_probed(self, provider):
        """`--thinking` and `--max-time` are appended to the probed set only when
        the call really renders them, so a build lacking `--max-time` still runs
        the dispatches that do not ask for one."""
        binary, help_text = self._with_help(OMP_REQUIRED_FLAGS)
        with binary, help_text:
            provider.build_command(self.PROMPT, self.MODEL)

            with pytest.raises(OmpFlagsUnconfirmed) as excinfo:
                provider.build_command(self.PROMPT, f"{self.MODEL}:high")

        assert excinfo.value.missing == ("--thinking",)

    def test_a_refusal_names_the_binary_and_the_operators_two_moves(self, provider):
        """Raised BEFORE any argv exists — no call ever gets a partial command —
        which is also what lets the exception propagate out of
        `run_single_model`'s ladder loop into its per-model `except`, recorded as
        that slot's error with `ladder.record` never reached: no retry, by
        construction, and the other slots of the fan-out unaffected. It borrows
        the `version-too-low` word as cursor's does, but the claim it makes is
        the stronger one — the name of the flag, not a number.
        """
        binary, help_text = self._with_help(())
        with binary, help_text, pytest.raises(OmpFlagsUnconfirmed) as excinfo:
            provider.build_command(self.PROMPT, self.MODEL)

        refusal = excinfo.value
        assert refusal.binary == sys.executable
        assert "Upgrade omp" in str(refusal)
        assert "drop omp from --models" in str(refusal)

    def test_an_unresolvable_binary_renders_the_argv_without_a_probe(self, provider):
        """"Not installed" is `is_available()`'s refusal to make, with its own
        precise message; a binary that cannot be spawned cannot reject a flag
        either. It is also the read-only scan's precondition — that guard reads
        this repo's source and must never depend on which CLIs a machine has.
        """
        def boom(*_args, **_kwargs):
            raise AssertionError("an unresolvable binary must not be probed")

        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None), \
                patch.object(base_module, "read_help_tokens", boom):
            argv = provider.build_command(self.PROMPT, self.MODEL)

        assert argv[0] == "omp"
        assert argv[-2:] == ["--", self.PROMPT]

    # ── guard visibility (§2.3) ──────────────────────────────────────────
    def test_the_bypass_is_exactly_one_visible_token_and_it_is_declared(
        self, provider, ungated
    ):
        """The token the scan reports is the VALUE, `yolo`, and never the flag.

        `negative_tokens_in_argv` records whole argv elements: with `yolo` in
        `NEGATIVE_VALUE_TOKENS` the pair `--approval-mode yolo` reports the value
        element once and the flag element never (it is in neither
        `NEGATIVE_FLAG_TOKENS` nor `NEGATIVE_TOKEN_PREFIXES`). Both sides of the
        machinery key on that one string, so a row naming the flag as its token
        would fail in both directions at once — `yolo` unexcused, and
        `--approval-mode` "never emitted".
        """
        from tests.test_read_only_modes import negative_tokens_in_argv, read_only_modes

        emitted = negative_tokens_in_argv(
            provider.build_command(self.PROMPT, self.MODEL)
        )

        assert emitted == ["yolo"]

        exception = read_only_modes()["omp"]["bypass_exception"]
        assert exception["token"] == "yolo"
        assert exception["profile"] == base_module.PROFILE_GENERATE
        assert read_only_modes()["omp"]["mechanism"] is None

    def test_the_approval_mode_is_on_argv_and_never_in_the_overlay(self):
        """The adapter cannot rely on a safe default because there is none —
        omp's own factory default IS `yolo` over every tool — and it must not
        rely on the user's `config.yml` either. Putting the mode in the overlay
        would make both irrelevant the same way and be invisible to the read-only
        scan, which is the one thing that may not happen. Read as a document
        rather than as text, so the word appearing in the file's own comments is
        not what this asserts about.
        """
        import yaml

        settings = yaml.safe_load(OMP_OVERLAY_PATH.read_text(encoding="utf-8")) or {}

        def _keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from _keys(value)
            elif isinstance(node, list):
                for value in node:
                    yield from _keys(value)

        assert "approvalMode" not in set(_keys(settings))
        assert "tools" not in set(_keys(settings))
        # What the overlay IS for: the retry bound and the project-MCP
        # suppression, both of which the argv has no flag for.
        assert settings["retry"]["enabled"] is False
        assert settings["mcp"]["enableProjectConfig"] is False

    def test_no_auto_approve_flag_is_ever_rendered(self, provider, ungated):
        argv = provider.build_command(self.PROMPT, self.MODEL)

        assert "--auto-approve" not in argv
        assert "--yolo" not in argv

    # ── the file-output declaration (§2.3) ───────────────────────────────
    def test_the_file_channel_is_declared_and_nothing_adjudicates_it(self, provider):
        """`supports_file_output = True` is a MEASUREMENT, not an inference.

        Under `--approval-mode yolo --tools read,grep,glob,bash,edit,write`
        the `write` and `edit` tools really created and edited files — a file
        appeared, and `p1.txt` went alpha→beta — so the declaration records what
        was observed. The empty atom list below is a separate fact and would be
        poor evidence on its own: `RESTRICTIVE_ATOM_RE` matches only
        `ask|plan|read-only|no-write|dry-run|deny|restrict|sandbox|safe` as whole
        words plus a zero budget, and `read,` is not `read-only` while `edit` and
        `write` are not `no-write`. So omp owns NO row in either adjudication
        table, any row added would be stale on arrival, and the regex is
        deliberately not widened to make one fit — a write-ENABLING flag cannot
        cause the failure that table guards. The day a profile renders a
        narrower `--tools` or an `always-ask` mode, that atom IS visible and the
        adjudication is owed then.
        """
        from tests.test_provider_housekeeping import (
            REVIEWED_STILL_WRITES,
            STOPS_FILE_WRITES,
            restrictive_atoms,
        )

        assert provider.supports_file_output is True
        assert restrictive_atoms("omp") == []
        assert not [key for key in STOPS_FILE_WRITES if key[0] == "omp"]
        assert not [key for key in REVIEWED_STILL_WRITES if key[0] == "omp"]

    def test_the_write_profile_is_not_declared(self, provider):
        """`generate` is the only profile this adapter ships. The accepted
        disposition is scoped to it, and an `agent-write` runner would be a
        separate decision on its own measurement.
        """
        assert provider.agent_profiles == (base_module.PROFILE_GENERATE,)
        assert provider.supports_file_edits is False

    # ── env (§2.10) ──────────────────────────────────────────────────────
    SOURCE_ENV = {
        "HOME": "/home/e2e",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "OPENROUTER_API_KEY": "sk-or-from-the-source-env",
        "GEMINI_API_KEY": "gemini-key",
        "OPENAI_API_KEY": "openai-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
    }

    def _composed(self, provider, source):
        """The child env both ways round the allowlist, as `(on, off)`.

        One composition point — `build_child_env` over `resolve_env` — because
        that is what both spawn sites call, and an adapter whose delta survived
        on one lane and not the other would be a credential that is present or
        absent depending on which phase asked.
        """
        out = []
        for enabled in (True, False):
            with patch.object(
                base_module, "env_allowlist_enabled", return_value=enabled
            ):
                out.append(
                    base_module.build_child_env(
                        provider,
                        provider_env=provider.resolve_env(
                            self.MODEL, base_module.PROFILE_GENERATE, base_env=source
                        ),
                        base_env=source,
                    )
                )
        return out[0], out[1]

    def test_the_credential_survives_the_filter_in_both_modes(
        self, provider, monkeypatch
    ):
        """Under `env_allowlist: on` nothing in omp's profile admits
        `OPENROUTER_API_KEY` — the allowlist requires a credential to start with
        the adapter's own prefix and omp's family is `PI_`/`OMP_` — so the one
        auth source measured as "the env alone works" would be silently dropped
        and omp would fall through its `$env` chain to a file, or exit 1.
        Re-emitting it from `get_env` is the route kimi, qwen and copilot already
        take; omp's own spelling happens to be the variable's real name.

        `get_env` reads that value from `os.environ` rather than from the
        `base_env` it is composed over, so both are set here and agree. The two
        lanes' composed envs are then asserted EQUAL, which is the property that
        matters: the allowlist decision may not change what omp is handed.
        """
        monkeypatch.setenv(
            "OPENROUTER_API_KEY", self.SOURCE_ENV["OPENROUTER_API_KEY"]
        )
        on, off = self._composed(provider, self.SOURCE_ENV)

        for env in (on, off):
            assert env["OPENROUTER_API_KEY"] == "sk-or-from-the-source-env"
            assert env["OMP_SKIP_SETUP"] == "1"
            assert env["HOME"] == "/home/e2e"
        assert on == off

    def test_an_absent_key_is_absent_rather_than_injected_empty(
        self, provider, monkeypatch
    ):
        """Conditional, not `os.environ.get(name, "")`.

        An injected empty string would be PRESENT and wrong to omp's `$env`
        chain, and would shadow the `~/.omp/agent/.env` source a user with no
        exported key relies on — turning a working configuration into a
        pre-flight `No API key found for openrouter.`
        """
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        source = {
            key: value
            for key, value in self.SOURCE_ENV.items()
            if key != "OPENROUTER_API_KEY"
        }

        on, off = self._composed(provider, source)

        for env in (on, off):
            assert "OPENROUTER_API_KEY" not in env
            assert env["OMP_SKIP_SETUP"] == "1"
        assert on == off

    def test_every_foreign_vendor_key_is_stripped_in_both_modes(
        self, provider, monkeypatch
    ):
        """"Adapter wins, ambient loses, no fallback."

        This is what closes the measured hazard on a DEFAULT install, not the
        env allowlist — that ships `off`, so with the filter off the child is the
        source environment plus this adapter's delta minus these removals, and
        without the list a stray `GEMINI_API_KEY` reaches omp on every dispatch.
        The two measured rows are the reason it is a list and not a comment: with
        either of them in the process env a run with no `--model` picked
        `google/gemini-3.1-pro-preview` and spun on a real 429.
        """
        monkeypatch.setenv(
            "OPENROUTER_API_KEY", self.SOURCE_ENV["OPENROUTER_API_KEY"]
        )
        source = {
            **self.SOURCE_ENV,
            **{name: "leaked" for name in omp_module._STRIPPED_VENDOR_VARS},
        }

        on, off = self._composed(provider, source)

        for env in (on, off):
            for name in omp_module._STRIPPED_VENDOR_VARS:
                assert name not in env
        assert on == off

    def test_nothing_the_adapter_injects_is_also_stripped(self, provider):
        """An adapter that strips what it injects is a bug, and the two
        declarations sit far enough apart in the file to make it a likely one."""
        injected = set(provider.get_env(self.MODEL))
        stripped = set(provider.get_remove_env())

        assert "OPENROUTER_API_KEY" not in omp_module._STRIPPED_VENDOR_VARS
        assert injected & stripped == set()
