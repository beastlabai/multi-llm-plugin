"""Tests for the Junie adapter.

Every captured stream named here is in ``tests/fixtures/junie/`` (its
``manifest.yaml`` says how it was taken). See docs/development/providers.md.
"""

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import junie as junie_module  # noqa: E402
from utils.providers.junie import JunieProvider  # noqa: E402
from utils.providers.base import build_child_env  # noqa: E402
from utils.retry_ladder import STATUS_AUTH_FAILED  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "junie"
MODEL = "openai/gpt-6-luna"


def _capture(stem):
    out = FIXTURES / f"{stem}.stdout.txt"
    err = FIXTURES / f"{stem}.stderr.txt"
    return (
        out.read_text(encoding="utf-8") if out.exists() else "",
        err.read_text(encoding="utf-8") if err.exists() else "",
    )


@pytest.fixture
def provider():
    return JunieProvider()


@pytest.fixture(autouse=True)
def _remove_roots():
    yield
    junie_module.cleanup_home_dirs()


class TestArgv:
    def test_name_and_binary(self, provider):
        assert provider.name == "junie"
        with patch("shutil.which", return_value="/usr/bin/junie") as which:
            assert provider.is_available() is True
        which.assert_called_once_with("junie")

    def test_every_dispatch_is_read_only_and_closes_repository_config(self, provider):
        argv = provider.build_command("Review", MODEL)
        assert argv[0] == "junie"
        assert argv[argv.index("--allowed-tools") + 1] == (
            "open,open_entire_file,scroll_down,scroll_up,glob_search,grep_search,submit"
        )
        for flag in (
            "--mcp-default-locations", "--skill-default-locations",
            "--command-default-location", "--agent-default-location",
            "--config-default-locations", "--model-default-locations",
            "--share-anonymous-statistics",
        ):
            assert argv[argv.index(flag) + 1] == "false", flag
        assert "--skip-update-check" in argv
        assert argv[argv.index("--output-format") + 1] == "json"
        assert argv[argv.index("--max-steps") + 1] == str(junie_module.JUNIE_MAX_STEPS)
        for write_tool in ("bash", "create", "search_replace", "multi_edit", "spawn_subagent"):
            assert write_tool not in argv[argv.index("--allowed-tools") + 1].split(",")

    def test_the_slug_never_reaches_argv(self, provider):
        argv = provider.build_command("Review", "deepseek/deepseek-v4.1-flash")
        assert argv[argv.index("--model") + 1] == "custom:multi-llm"
        assert not any("deepseek" in token for token in argv)

    def test_the_prompt_is_the_trailing_task_pair(self, provider):
        argv = provider.build_command("Review this", MODEL)
        assert argv[-2:] == ["--task", "Review this"]

    def test_stdin_drops_the_task_pair(self, provider):
        argv, stdin = provider.build_command_for_transport("Review this", MODEL, "stdin")
        assert stdin == "Review this"
        assert "--task" not in argv and "Review this" not in argv
        assert argv == provider.build_command("Review this", MODEL)[:-2]

    def test_file_transport_is_refused(self, provider):
        with pytest.raises(ValueError):
            provider.build_command_for_transport("p", MODEL, "file", "/tmp/p")

    def test_an_effort_suffix_renders_effort(self, provider):
        argv = provider.build_command("p", f"{MODEL}:xhigh")
        assert argv[argv.index("--effort") + 1] == "xhigh"
        assert "--effort" not in provider.build_command("p", MODEL)

    def test_timeout_and_schema_are_inert(self, provider):
        assert provider.build_command("p", MODEL, timeout_s=30) == provider.build_command("p", MODEL)


class TestEnvironment:
    def test_the_profile_holds_the_slug_and_a_key_reference_only(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value")
        env = provider.get_env("deepseek/deepseek-v4.1-flash")
        models = Path(env["JUNIE_MODEL_LOCATIONS"])
        profile_path = models / "multi-llm.json"
        text = profile_path.read_text(encoding="utf-8")
        assert "sk-or-secret-value" not in text
        profile = json.loads(text)
        assert profile == {
            "baseUrl": "https://openrouter.ai/api/v1/chat/completions",
            "id": "deepseek/deepseek-v4.1-flash",
            "apiType": "OpenAICompletion",
            "apiKey": "${OPENROUTER_API_KEY}",
        }
        if os.name == "posix":
            assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(models.parent.stat().st_mode) == 0o700

    def test_an_effort_is_the_profiles_only_effort(self, provider):
        env = provider.get_env(f"{MODEL}:low")
        profile = json.loads((Path(env["JUNIE_MODEL_LOCATIONS"]) / "multi-llm.json").read_text())
        assert profile["id"] == MODEL
        assert profile["efforts"] == ["low"] and profile["defaultEffort"] == "low"

    def test_junie_home_and_user_home_both_point_into_the_root(self, provider, monkeypatch):
        monkeypatch.delenv("JAVA_TOOL_OPTIONS", raising=False)
        env = provider.get_env(MODEL)
        root = Path(env["JUNIE_MODEL_LOCATIONS"]).parent
        assert env["JUNIE_HOME"] == str(root / ".junie")
        assert env["JAVA_TOOL_OPTIONS"] == f"-Duser.home={root}"

    def test_a_users_java_options_are_kept_ahead_of_ours(self, provider, monkeypatch):
        monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Xmx2g -Duser.home=/elsewhere")
        env = provider.get_env(MODEL)
        options = env["JAVA_TOOL_OPTIONS"].split()
        assert options[:2] == ["-Xmx2g", "-Duser.home=/elsewhere"]
        assert options[-1].startswith("-Duser.home=") and options[-1] != "-Duser.home=/elsewhere"

    def test_every_spawn_gets_its_own_root(self, provider):
        first = provider.get_env(MODEL)["JUNIE_HOME"]
        second = provider.get_env(MODEL)["JUNIE_HOME"]
        assert first != second

    def test_inherited_junie_settings_and_java_overrides_are_stripped(self, provider):
        removed = provider.get_remove_env(base_env={
            "JUNIE_MODEL": "sonnet", "JUNIE_TASK": "x", "JUNIE_TRUST_PROJECT": "true",
            "JUNIE_OPENROUTER_API_KEY": "k", "JUNIE_HOME": "/mine", "PATH": "/bin",
        })
        assert {"JUNIE_MODEL", "JUNIE_TASK", "JUNIE_TRUST_PROJECT",
                "JUNIE_OPENROUTER_API_KEY", "_JAVA_OPTIONS"} <= set(removed)
        assert "JUNIE_HOME" not in removed and "PATH" not in removed

    def test_the_child_env_carries_the_root_and_drops_a_users_junie_model(self, provider, monkeypatch):
        monkeypatch.setenv("JUNIE_MODEL", "sonnet")
        monkeypatch.setenv("JUNIE_HOME", "/home/user/.junie")
        env = build_child_env(provider, provider_env=provider.resolve_env(MODEL))
        assert "JUNIE_MODEL" not in env
        assert env["JUNIE_HOME"] != "/home/user/.junie"
        assert env["JUNIE_HOME"].startswith(str(Path(env["JUNIE_MODEL_LOCATIONS"]).parent))


ENVELOPE = {
    "sessionId": "session-1", "taskName": "Review the plan",
    "result": '[{"id": "S001"}]', "changes": [],
    "llmUsage": [{"model": MODEL, "calls": 3, "cost": 0.0, "inputTokens": 300,
                  "cacheInputTokens": 20, "cacheCreateTokens": 0, "outputTokens": 40}],
}


class TestParse:
    def test_a_clean_envelope_is_the_submitted_answer(self, provider):
        result = provider.parse_output(json.dumps(ENVELOPE), "")
        assert result["success"] is True
        assert result["data"] == [{"id": "S001"}]

    def test_errors_fail_even_on_an_answer(self, provider):
        envelope = dict(ENVELOPE, errors=["The run reached its step limit (60 steps)"] * 2)
        result = provider.parse_output(json.dumps(envelope), "")
        assert result["success"] is False
        assert result["error"] == "The run reached its step limit (60 steps)"

    def test_an_empty_result_fails(self, provider):
        result = provider.parse_output(json.dumps(dict(ENVELOPE, result="  ")), "")
        assert result["success"] is False

    def test_plain_text_on_stdout_fails(self, provider):
        result = provider.parse_output("Authenticated successfully\n● unexpected response", "")
        assert result["success"] is False

    def test_an_unset_key_is_named_from_stderr(self, provider):
        stdout, stderr = _capture("missing-key")
        result = provider.parse_output(stdout, stderr)
        assert result["success"] is False
        assert "OPENROUTER_API_KEY" in result["error"]
        assert provider.classify_failure(stdout, stderr, 1) == STATUS_AUTH_FAILED

    def test_salvage_reads_result_only_without_errors(self, provider):
        assert provider.salvage_text(json.dumps(ENVELOPE)) == '[{"id": "S001"}]'
        assert provider.salvage_text(json.dumps(dict(ENVELOPE, errors=["x"]))) == ""
        assert provider.salvage_text("not json") == ""

    def test_usage_sums_tokens_and_never_reports_cost(self, provider):
        envelope = dict(ENVELOPE, llmUsage=ENVELOPE["llmUsage"] * 2)
        usage = provider.parse_usage(json.dumps(envelope), "")
        assert usage == {"input_tokens": 600, "output_tokens": 80,
                         "cached_input_tokens": 40, "total_tokens": 680}
        assert "cost_usd" not in usage

    def test_no_usage_rows_is_absent_usage(self, provider):
        assert provider.parse_usage(json.dumps(dict(ENVELOPE, llmUsage=[])), "") is None
        assert provider.parse_usage("", "") is None

    def test_vendor_text_drops_the_models_words(self, provider):
        envelope = dict(ENVELOPE, taskName="rate limit exceeded", result="401 unauthorized")
        stdout, _ = provider.vendor_text(json.dumps(envelope), "")
        assert "rate limit" not in stdout and "unauthorized" not in stdout


class TestCaptures:
    def test_the_success_capture_is_the_submitted_findings(self, provider):
        stdout, stderr = _capture("success")
        result = provider.parse_output(stdout, stderr)
        assert result["success"] is True
        assert [item["id"] for item in result["data"]] == ["S001", "S002"]

    def test_the_decoy_text_before_a_tool_call_is_not_the_answer(self, provider):
        stdout, stderr = _capture("multi-assistant")
        result = provider.parse_output(stdout, stderr)
        assert result["success"] is True
        assert "D999" not in json.dumps(result["data"])

    def test_the_step_limit_exits_zero_and_still_fails(self, provider):
        stdout, stderr = _capture("exit-zero-failure")
        result = provider.parse_output(stdout, stderr)
        assert result["success"] is False
        assert "step limit" in result["error"]


class TestGates:
    def test_a_build_without_the_read_only_flag_is_refused(self, provider):
        with patch.object(JunieProvider, "missing_flags", return_value=("--allowed-tools",)), \
                patch.object(JunieProvider, "gate_binary", return_value="/opt/junie"):
            with pytest.raises(junie_module.JunieFlagsUnconfirmed, match="--allowed-tools"):
                provider.build_command("p", MODEL)

    def test_the_required_flags_cover_every_restriction_in_the_argv(self, provider):
        argv = provider.build_command("p", MODEL)
        flags = {token for token in argv if token.startswith("--")} - {
            "--skip-update-check", "--share-anonymous-statistics", "--task"}
        assert flags <= set(junie_module.JUNIE_REQUIRED_FLAGS)

    def test_a_root_with_whitespace_is_refused(self, monkeypatch):
        monkeypatch.delenv("JAVA_TOOL_OPTIONS", raising=False)
        with pytest.raises(OSError, match="whitespace"):
            junie_module._java_options("/tmp/a b/root")
        assert junie_module._java_options("/tmp/root") == "-Duser.home=/tmp/root"
