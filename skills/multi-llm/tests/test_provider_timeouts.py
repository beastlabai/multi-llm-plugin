"""Tests for how the resolved timeout reaches each adapter.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.base import LLMProvider
from utils.providers.omp import OVERLAY_PATH as OMP_OVERLAY_PATH
from tests.roster_check import assert_covers_registry


class TestInvokeWithProviderThreadsTheResolvedTimeout:
    """§9.6.b: the value comes from `invoke_with_provider`'s own computation."""

    @pytest.mark.parametrize("configured,expected", [
        (600, "570s"),
        (1200, "1170s"),
        (2400, "2370s"),
    ])
    def test_the_spawned_argv_carries_the_derived_flag(self, configured, expected):
        from unittest.mock import MagicMock
        from utils.llm_client import invoke_with_provider

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = json.dumps({"status": "SUCCESS", "response": "[]"})
        completed.stderr = ""

        with patch("shutil.which", return_value="/usr/bin/agy"), \
                patch("subprocess.run", return_value=completed) as run:
            invoke_with_provider(
                prompt="Test prompt",
                model_spec="agy:Gemini 3.1 Pro (High)",
                timeout=configured,
            )

        argv = run.call_args.args[0]
        assert argv[argv.index("--print-timeout") + 1] == expected
        # The two deadlines are the same number, one margin apart — never a tie.
        assert run.call_args.kwargs["timeout"] == configured


# --- The off switch: `timeout_s=None` renders today's argv byte-for-byte ----
#
# Captured from the pre-L5 tree, before the keyword existed. This is the whole
# rollback story (§14.4): drop the one call site in `llm_client.py` and every
# adapter is back to exactly these bytes, agy included.
PRE_L5_ARGV_PROMPT = "Analyze this plan carefully"

#: Where a new adapter adds its rows, for the coverage failures.
PRE_L5_ARGV_WHERE = "PRE_L5_ARGV in tests/test_provider_timeouts.py"
PRE_L5_ARGV_HINT = (
    "A new adapter's row is the argv its build_command(PRE_L5_ARGV_PROMPT, "
    "<model>) renders today, and its PRE_L5_ARGV_MODELS row is that model: one "
    "real id from its providers.yaml block. Both tables are also read by "
    "tests/test_structured_output.py."
)


PRE_L5_ARGV_MODELS = {
    "agy": "Gemini 3.1 Pro (High)",
    "aider": "openrouter/z-ai/glm-5.2",
    "claude-code": "sonnet",
    "cline": "openrouter/z-ai/glm-5.2",
    "codex": "gpt-5.5",
    "copilot": "openai/gpt-4.1-mini",
    "cursor-agent": "auto",
    "gemini": "gemini-2.5-pro",
    "goose": "openrouter/z-ai/glm-5.2",
    "grok": "grok-4",
    "fx": "openai/gpt-5.6-luna",
    "kilocode": "openai-native/gpt-5.5",
    "kimi": "moonshotai/kimi-k2-thinking",
    "muse": "muse-spark-1.2",
    # The real default SELECTOR (providers.yaml), not a placeholder: omp
    # matches on `selector` (openrouter/openai/gpt-5.6-luna) and not on `id`
    # (openai/gpt-5.6-luna), and the row is what keeps the difference rendered.
    "omp": "openrouter/openai/gpt-5.6-luna",
    "opencode": "openai/gpt-5.5",
    "pi": "anthropic/claude-opus-4-8",
    "qwen-code": "qwen/qwen3.8-max",
    # vibe's entry is a SEED ALIAS, not a wire slug: it has no --model flag,
    # so this value never reaches argv at all -- it rides VIBE_ACTIVE_MODEL.
    # It is still the real default from providers.yaml, because the housekeeping
    # sweep renders argv from these and a dummy would exercise nothing.
    "vibe": "vibe-default",
}


PRE_L5_ARGV = {
    "agy": [
        "agy", "--new-project", "--dangerously-skip-permissions",
        "--output-format", "json", "--print-timeout", "20m",
        "--model", "Gemini 3.1 Pro (High)", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "aider": [
        "aider", "--model", "openrouter/z-ai/glm-5.2",
        "--message", f"/ask {PRE_L5_ARGV_PROMPT}",
        # No `--yes-always` since plans/ask-answer-recovery-and-false-success.md
        # T4.2: the decline supply on stdin answers aider's prompts instead.
        "--no-auto-commits", "--no-pretty", "--no-stream",
        "--no-check-update", "--no-show-model-warnings", "--no-analytics",
        "--no-gitignore", "--no-show-release-notes", "--no-detect-urls",
        "--no-fancy-input",
        "--chat-history-file", os.devnull,
        "--input-history-file", os.devnull,
    ],
    # §17's write grant is part of the frozen argv as of 2026-08-20: the
    # generate prompt ends in "write your JSON output to this file: <path>"
    # (`supports_file_output = True`), and headless `claude -p` denied the
    # model's own Write without it. Measured on the credentialed live lane.
    "claude-code": [
        "claude", "-p", "--output-format", "json", "--model", "sonnet",
        "--permission-mode", "acceptEdits",
        PRE_L5_ARGV_PROMPT,
    ],
    "cline": [
        "cline", "--json", "-P", "openrouter", "-m", "z-ai/glm-5.2",
        PRE_L5_ARGV_PROMPT,
    ],
    "codex": [
        "codex", "exec", "--sandbox", "workspace-write", "--json",
        "-c", "features.plugins=false",
        "--ephemeral", "--skip-git-repo-check", "--model", "gpt-5.5",
        PRE_L5_ARGV_PROMPT,
    ],
    # No permission flag and no `--add-dir`: §6.5 tier 1, measured sufficient
    # with a clean canary at all four rungs. This entry is what keeps the help
    # text's false "--allow-all-tools is required for non-interactive mode" from
    # being "fixed" back into the argv by a later edit.
    "copilot": [
        "copilot", "--output-format", "json", "--stream", "off",
        "--log-level", "none", "--no-custom-instructions", "--no-ask-user",
        "--no-auto-update", "--disable-builtin-mcps",
        "--model", "openai/gpt-4.1-mini", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "cursor-agent": [
        "cursor-agent", "--print", "--trust", "--mode", "ask",
        "--output-format", "json", "--model", "auto", PRE_L5_ARGV_PROMPT,
    ],
    "gemini": [
        "gemini", "--output-format", "json", "--model", "gemini-2.5-pro",
        PRE_L5_ARGV_PROMPT,
    ],
    "goose": [
        "goose", "run", "--no-session", "-q", "--output-format", "json",
        "--no-profile", "--max-turns", "25", "--with-builtin", "developer",
        "--provider", "openrouter", "--model", "z-ai/glm-5.2",
        "-t", PRE_L5_ARGV_PROMPT,
    ],
    # §9.4 moved grok's `-p <prompt>` pair to the END (it was at index 4). That
    # is an independent normalization, NOT the timeout keyword: `timeout_s` is
    # still inert here, and the rollback story below is unaffected — dropping the
    # keyword's call site returns grok to exactly this argv.
    "grok": [
        "grok", "--no-auto-update", "--always-approve",
        "--output-format", "json", "-m", "grok-4", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "fx": [
        "fx", "ask", "--json", "--no-save", "--auto", "--model",
        "openai/gpt-5.6-luna", "--", PRE_L5_ARGV_PROMPT,
    ],
    "kilocode": [
        "kilocode", "run", "--auto", "-m", "openai-native/gpt-5.5",
        PRE_L5_ARGV_PROMPT,
    ],
    # No model token: kimi selects it through KIMI_MODEL_NAME, and `-m <slug>`
    # is a hard failure ("Model ... is not configured in config.toml"). This row
    # is what keeps that measured fact from being "fixed" back into argv.
    "kimi": [
        "kimi", "--output-format", "stream-json", "-p", PRE_L5_ARGV_PROMPT,
    ],
    "muse": [
        "muse", "exec", "--json", "--disable-approval", "--disable-sandbox",
        "--model", "muse-spark-1.2", PRE_L5_ARGV_PROMPT,
    ],
    # Two values in this row are NOT literals and cannot be: the overlay lives
    # beside the skill and `--cwd` is the spawn directory. Both are still
    # assertions — `--config` must be the absolute overlay THIS build ships, and
    # `--cwd` must be rendered on every dispatch (without it, launched from
    # $HOME omp silently relocates to /tmp and reviews an empty tree) — so they
    # are taken from the adapter's own constant and from `os.getcwd()`, read
    # here at import for the same reason the argv reads it at build time.
    # Everything else is policy and is spelled out: no `--thinking` and no
    # `--max-time` (the model carries no `:effort` and `timeout_s` is None), the
    # approval mode and the six-name allow-list verbatim, and the prompt last
    # behind `--`.
    "omp": [
        "omp", "--no-session", "-p", "--mode", "json",
        "--config", str(OMP_OVERLAY_PATH),
        "--approval-mode", "yolo",
        "--tools", "read,grep,glob,bash,edit,write",
        "--cwd", os.getcwd(),
        "--model", "openrouter/openai/gpt-5.6-luna",
        "--", PRE_L5_ARGV_PROMPT,
    ],
    "opencode": [
        "opencode", "run", "--format", "json", "--print-logs",
        "--log-level", "ERROR", "--model", "openai/gpt-5.5",
        PRE_L5_ARGV_PROMPT,
    ],
    # `--mode json` since plan §7 (T7.5): the JSONL event stream parse_output
    # reads, rendered right after `-p` like omp's.
    "pi": [
        "pi", "--no-session", "-p", "--mode", "json",
        "--model", "anthropic/claude-opus-4-8",
        PRE_L5_ARGV_PROMPT,
    ],
    # No --max-wall-time element at all: that flag is the ONLY row-derived value
    # qwen renders, and `timeout_s=None` must emit nothing rather than a default
    # (§1's guard, and what keeps expected_argv.yaml free of a "sometimes
    # present" case). The prompt is a bare trailing positional -- `-p` is
    # deprecated at 0.21.13.
    "qwen-code": [
        "qwen", "--bare", "--chat-recording", "false", "-o", "json",
        "--max-tool-calls", "0", "--max-session-turns", "25",
        "-m", "qwen/qwen3.8-max", PRE_L5_ARGV_PROMPT,
    ],
    # Note what is NOT here: the model. vibe has no `--model` flag, so the alias
    # rides VIBE_ACTIVE_MODEL and the argv is model-independent -- which is also
    # why vibe needs no normalize_argv.py entry (no model token ever reaches the
    # call log). `--max-price` and `--max-turns` are POLICY constants and are
    # pinned verbatim; `--agent plan` is the §6.5 rung, and the bare form it
    # replaces is DESTRUCTIVE, not tier 1.
    "vibe": [
        "vibe", "--output", "json", "--max-price", "0.25",
        "--max-turns", "4", "--agent", "plan", "-p", PRE_L5_ARGV_PROMPT,
    ],
}


class TestTimeoutKeywordIsInertWhenUnset:
    """`timeout_s=None` is the seam; assert it rather than assume it."""

    @pytest.fixture(autouse=True)
    def ungated(self):
        """Render argv from the adapters, never from the developer's PATH.

        Two rows in the table are gated: omp's §2.11 probe and cursor's
        `--trust`/`--mode` one both read the *installed* binary's `--help`
        before `build_command` will emit anything. Ungated, a machine carrying
        an older omp raises `OmpFlagsUnconfirmed` here instead of comparing
        argv -- a machine-dependent failure in the one sweep whose whole point
        is that these bytes are machine-independent.
        `TestOmpProvider.ungated` neutralizes exactly this lookup for exactly
        this reason, and the shared gate resolves its binary through
        `LLMProvider.gate_binary`, so neutralizing that one lookup is enough:
        with no binary to read, `missing_flags` reports nothing missing and the
        argv is the one the adapter would render. pi's optional `--tools` gate
        (T3.1) reads the same lookup, so its row is the argv WITHOUT the flag;
        the argv with it is pinned in `TestPiToolAllowlist`. Nothing else in the
        table consults the gate from `build_command` (claude's two optional-flag
        reads are on the `schema`/fallback paths this sweep does not take), and
        both gates are graded on their own, with the help text stubbed rather
        than the lookup.
        """
        with patch.object(LLMProvider, "gate_binary", lambda self, **_kw: None):
            yield

    def test_the_table_covers_every_registered_adapter(self):
        assert_covers_registry(
            PRE_L5_ARGV, PRE_L5_ARGV_WHERE, what="a pinned argv row",
            hint=PRE_L5_ARGV_HINT,
        )
        assert_covers_registry(
            PRE_L5_ARGV_MODELS, "PRE_L5_ARGV_MODELS in tests/test_provider_timeouts.py",
            what="a model row", hint=PRE_L5_ARGV_HINT,
        )

    @pytest.mark.parametrize("name", sorted(PRE_L5_ARGV))
    def test_explicit_none_renders_the_pre_l5_argv(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        argv = provider.build_command(
            PRE_L5_ARGV_PROMPT, PRE_L5_ARGV_MODELS[name], timeout_s=None
        )

        assert argv == PRE_L5_ARGV[name]

    @pytest.mark.parametrize("name", sorted(PRE_L5_ARGV))
    def test_omitting_the_keyword_is_the_same_as_passing_none(self, name):
        from utils.provider_registry import _PROVIDERS

        provider = _PROVIDERS[name]
        model = PRE_L5_ARGV_MODELS[name]

        assert provider.build_command(PRE_L5_ARGV_PROMPT, model) == provider.build_command(
            PRE_L5_ARGV_PROMPT, model, timeout_s=None
        )

    @pytest.mark.parametrize("name", sorted(PRE_L5_ARGV))
    def test_the_other_three_builders_are_inert_too(self, name):
        """The keyword lands on all four builders and changes none of them."""
        from utils.provider_registry import _PROVIDERS
        from utils.providers.base import PROFILE_GENERATE

        provider = _PROVIDERS[name]
        model = PRE_L5_ARGV_MODELS[name]

        for transport in ("argv", "stdin"):
            if transport != "argv" and not provider.supports_prompt_transport(transport):
                continue
            assert provider.build_command_for_transport(
                PRE_L5_ARGV_PROMPT, model, transport, timeout_s=None
            ) == provider.build_command_for_transport(
                PRE_L5_ARGV_PROMPT, model, transport
            )

        for profile in provider.agent_profiles:
            if profile == PROFILE_GENERATE:
                continue
            assert provider.build_agent_command(
                PRE_L5_ARGV_PROMPT, model, profile, timeout_s=None
            ) == provider.build_agent_command(PRE_L5_ARGV_PROMPT, model, profile)
            for transport in ("argv", "stdin"):
                if transport != "argv" and not provider.supports_prompt_transport(transport):
                    continue
                assert provider.build_agent_command_for_transport(
                    PRE_L5_ARGV_PROMPT, model, profile, transport, timeout_s=None
                ) == provider.build_agent_command_for_transport(
                    PRE_L5_ARGV_PROMPT, model, profile, transport
                )
