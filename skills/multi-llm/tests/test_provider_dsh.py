"""The dsh adapter: argv, the private DSH_HOME and its patch, and the event stream.

The captured rows (tests/fixtures/dsh/manifest.yaml) are graded by
test_provider_fixture_rows.py; these cases pin what a capture cannot show on
its own: the rendered argv per transport, what the environment and the profile
patch carry for a model spec, and the reading of hand-built edge streams.
"""
import json
import os
from pathlib import Path

import pytest
import yaml

from utils.providers import dsh
from utils.providers.dsh import DshFlagsUnconfirmed, DshProvider

MODEL = "deepseek/deepseek-v4.1-flash-1017"


@pytest.fixture
def provider(monkeypatch):
    p = DshProvider()
    monkeypatch.setattr(p, "missing_flags", lambda flags: ())
    yield p
    dsh.cleanup_home_dirs()


def stream(*events):
    return "\n".join(json.dumps(e) for e in events) + "\n"


def final_ok(text):
    return {"type": "final", "sessionId": "s", "reason": {"kind": "completed"}, "text": text}


def final_err(code, message):
    return {"type": "final", "sessionId": "s", "text": "",
            "reason": {"kind": "error", "error": {"code": code, "message": message}}}


def step_end(i, o):
    return {"type": "status", "phase": "step_end", "turn": 1, "step": 1,
            "usage": {"inputTokens": i, "outputTokens": o, "totalTokens": i + o}}


# -- argv ----------------------------------------------------------------------

def test_argv_is_read_only_and_never_names_a_model(provider):
    argv = provider.build_command("review this", f"{MODEL}:high")
    assert argv == ["dsh", "--profile", "headless", "--json", "--permission-mode", "read-only",
                    "--approval-policy", "never", "--", "review this"]


def test_stdin_transport_reads_the_task_from_dash(provider):
    argv, stdin = provider.build_command_for_transport("-x prompt", MODEL, "stdin")
    assert argv[-1] == "-" and "--" not in argv and stdin == "-x prompt"


def test_argv_transport_keeps_the_separator(provider):
    argv, stdin = provider.build_command_for_transport("--help me", MODEL, "argv")
    assert argv[-2:] == ["--", "--help me"] and stdin is None


def test_unknown_transport_is_refused(provider):
    with pytest.raises(ValueError):
        provider.build_command_for_transport("p", MODEL, "file")


def test_a_dsh_without_the_launcher_flags_is_refused(monkeypatch):
    p = DshProvider()
    monkeypatch.setattr(p, "missing_flags", lambda flags: ("--profile",))
    monkeypatch.setattr(p, "gate_binary", lambda runnable=False: "/usr/bin/dsh")
    with pytest.raises(DshFlagsUnconfirmed, match="distributed shell"):
        p.build_command("p", MODEL)


def test_the_prerelease_version_is_read_whole(provider):
    assert provider.parse_version("0.2.2-rc.5\n", "") == "0.2.2-rc.5"
    assert provider.parse_version("", "dsh 0.3.0\n") == "0.3.0"


# -- environment and the profile patch ------------------------------------------

@pytest.mark.parametrize("spec,model,reasoning", [
    (MODEL, MODEL, ""),
    (f"{MODEL}:high", MODEL, "high"),
    (f"{MODEL}:max", MODEL, "max"),
    (f"{MODEL}:off", MODEL, "off"),
])
def test_the_model_and_its_effort_travel_in_the_environment(provider, spec, model, reasoning):
    env = provider.get_env(spec)
    assert env[dsh.MODEL_ENV] == model
    assert env[dsh.REASONING_ENV] == reasoning
    assert env["DSH_TELEMETRY_MODE"] == "DISABLED"


def test_every_spawn_gets_its_own_private_home_with_the_patch(provider, monkeypatch):
    monkeypatch.setenv("DSH_HOME", "/somewhere/the/user/pinned")
    first, second = provider.get_env(MODEL)["DSH_HOME"], provider.get_env(MODEL)["DSH_HOME"]
    assert first != second
    for home in (first, second):
        assert not home.startswith("/somewhere")
        if os.name == "posix":
            assert (os.stat(home).st_mode & 0o777) == 0o700
        patch = Path(home, "profiles", "headless", "cordis.patch.yml")
        assert patch.read_text(encoding="utf-8") == dsh.DSH_PROFILE_PATCH


class _JsTag(yaml.SafeLoader):
    pass


_JsTag.add_constructor("!!js", lambda loader, node: ("js", loader.construct_scalar(node)))
_JsTag.add_constructor("tag:yaml.org,2002:js", lambda loader, node: ("js", loader.construct_scalar(node)))


def test_the_patch_pins_the_route_and_disables_the_side_requests():
    entries = {e["id"]: e for e in yaml.load(dsh.DSH_PROFILE_PATCH, Loader=_JsTag)}
    for disabled in ("session-title-llm", "session-log-deepseek", "session-telemetry-otel"):
        assert entries[disabled] == {"id": disabled, "disabled": True}
    model = entries["agent-default-model"]["config"]
    assert model["provider"] == "openrouter"
    assert model["model"] == ("js", f"process.env.{dsh.MODEL_ENV}")
    route = entries["llm-pi-ai"]["config"]["providers"]
    assert list(route) == ["openrouter"]  # never deepseek-official
    route = route["openrouter"]
    assert route["baseURL"] == "https://openrouter.ai/api/v1"
    assert route["apiKeyEnv"] == "OPENROUTER_API_KEY"
    assert route["retryPolicy"] == {"mode": "normal", "maxRetries": dsh.DSH_MAX_RETRIES}
    assert route["streamIdleTimeoutMs"] == dsh.DSH_STREAM_IDLE_TIMEOUT_S * 1000
    assert route["models"][0]["maxTokens"] == dsh.DSH_MAX_OUTPUT_TOKENS
    # No credential value anywhere in the file.
    assert "sk-" not in dsh.DSH_PROFILE_PATCH


def test_the_patch_variables_cannot_come_from_a_project_env_file():
    # dsh refuses to start when a project .env sets any DSH_* name.
    assert dsh.MODEL_ENV.startswith("DSH_") and dsh.REASONING_ENV.startswith("DSH_")


def test_ambient_dsh_variables_are_stripped_but_ours_are_kept(provider):
    source = {"DSH_PERMISSION_MODE": "danger-full-access", "DSH_TELEMETRY_OTLP_URL": "x",
              "DSH_HOME": "/h", dsh.MODEL_ENV: "m", "OPENROUTER_API_KEY": "k"}
    removed = provider.get_remove_env(base_env=source)
    assert removed == ["DSH_PERMISSION_MODE", "DSH_TELEMETRY_OTLP_URL"]


def test_the_retry_envelope_is_derived_from_the_pinned_bounds():
    envelope = DshProvider.internal_retries
    assert envelope.max_requests == dsh.DSH_MAX_RETRIES + 1
    assert envelope.max_wall_s >= (dsh.DSH_MAX_RETRIES + 1) * dsh.DSH_STREAM_IDLE_TIMEOUT_S
    assert envelope.retries_rate_limit is True


# -- reading the stream ------------------------------------------------------------

def test_the_answer_is_the_final_message_not_the_narration(provider):
    raw = stream({"type": "text", "text": '[{"id": "D999"}]'},
                 {"type": "tool_call", "callId": "c", "tool": "read", "input": {}},
                 {"type": "text", "text": '[{"id": "S001"}]'},
                 final_ok('[{"id": "S001"}]'))
    assert provider.parse_output(raw, "") == {"success": True, "data": [{"id": "S001"}]}
    assert provider.salvage_text(raw) == '[{"id": "S001"}]'


def test_a_run_without_a_final_record_fails_even_at_exit_zero(provider):
    raw = stream({"type": "session", "sessionId": "s"},
                 {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
                 {"type": "text", "text": "[]"}, step_end(5, 1))
    result = provider.parse_output(raw, "")
    assert result["success"] is False and "without a final event" in result["error"]
    # Cut after the step ended with no tool call: the message is salvage's answer.
    assert provider.salvage_text(raw) == "[]"


def test_a_message_whose_step_never_ended_is_not_salvaged(provider):
    # The multi-assistant capture cut after its decoy: the decoy is narration.
    raw = stream({"type": "status", "phase": "step_start", "turn": 1, "step": 1},
                 {"type": "text", "text": '[{"id": "D999"}]'})
    assert provider.salvage_text(raw) == ""


def test_line_separators_inside_an_answer_do_not_cut_the_record(provider):
    answer = '[{"title": "a\u2028b\u2029c\u0085d"}]'
    line = json.dumps(final_ok(answer), ensure_ascii=False)
    assert "\u2028" in line
    assert provider.parse_output(line + "\n", "")["success"] is True


def test_narration_cut_by_a_tool_call_is_not_salvaged(provider):
    raw = stream({"type": "text", "text": "[1]"},
                 {"type": "tool_call", "callId": "c", "tool": "read", "input": {}})
    assert provider.salvage_text(raw) == ""


def test_a_completed_run_with_no_text_fails(provider):
    result = provider.parse_output(stream(final_ok("  ")), "")
    assert result["success"] is False and "no final message" in result["error"]


def test_salvage_has_no_opinion_on_a_stream_that_is_not_dsh(provider):
    assert provider.salvage_text("plain text\n") is None


@pytest.mark.parametrize("code,message,verdict", [
    ("AUTH", "401: bad key", "auth_failed"),
    ("MISSING_CREDENTIAL", "no credential", "auth_failed"),
    ("RATE_LIMIT", "429: slow down", "rate_limited"),
    ("QUOTA", "402: insufficient credits", "rate_limited"),
    ("CONTEXT_WINDOW_EXCEEDED", "400: too long", "context_length_exceeded"),
    ("UNKNOWN_MODEL", "no configured model", "invalid_model"),
    ("UNSUPPORTED_REASONING_EFFORT", "does not support reasoning effort", "invalid_model"),
    ("INVALID_REQUEST", "404: The model `x` does not exist", "invalid_model"),
    ("INVALID_REQUEST", "400: something else", None),
    ("SERVER", "500", "server_error"),
    ("TIMEOUT", "pi-ai stream idle timeout after 120000ms", "server_error"),
    ("EMPTY_RESPONSE", "no content", "server_error"),
    ("TRANSPORT", "ECONNRESET", "connection_error"),
    ("INVALID_CONFIG", "needs an api", "runner_error"),
])
def test_the_final_records_code_decides_the_verdict(provider, code, message, verdict):
    raw = stream(final_err(code, message))
    assert provider.classify_failure(raw, f"dsh: {code}: {message}\n", 1) == verdict
    assert provider.parse_output(raw, "")["error"] == f"dsh {code}: {message}"


def test_a_preflight_failure_is_read_from_stderr(provider):
    stderr = 'dsh: UNKNOWN_MODEL: pi-ai provider "openrouter" has no configured model "x"\n'
    assert provider.classify_failure("", stderr, 1) == "invalid_model"
    assert provider.describe_failure("", stderr).startswith("dsh UNKNOWN_MODEL:")


def test_a_project_env_refusal_is_a_runner_error(provider):
    stderr = ('Error: dsh: /repo/.env sets "DSH_PERMISSION_MODE", which only the launching '
              'environment may set (it decides how this process starts); export it instead\n'
              "    at readEnvLayer (file:///x.js:1:1)\n")
    assert provider.classify_failure("", stderr, 1) == "runner_error"
    assert provider.describe_failure("", stderr).startswith('dsh: /repo/.env sets "DSH_PERMISSION_MODE"')


def test_the_models_text_never_reaches_the_verdict(provider):
    raw = stream({"type": "text", "text": "401 unauthorized rate limit exceeded"},
                 final_ok('[{"title": "429 rate limit handling"}]'))
    assert provider.classify_failure(raw, "", 0) is None
    vendor, _ = provider.vendor_text(raw, "")
    assert vendor == ""


def test_model_bytes_dumped_on_stderr_never_reach_the_verdict(provider):
    stderr = ("Could not parse message into JSON:\nFrom chunk:\n"
              "401 unauthorized 429 rate limit exceeded insufficient credits\n"
              "dsh: PI_AI_ERROR: Unexpected end of JSON input\n")
    assert provider.classify_failure("", stderr, 1) is None
    _, reduced = provider.vendor_text("", stderr)
    assert reduced == "dsh: PI_AI_ERROR: Unexpected end of JSON input"


def test_a_home_that_cannot_be_made_fails_closed(provider, monkeypatch):
    monkeypatch.setenv("DSH_HOME", "/user/pinned")

    def refuse():
        raise OSError("no space")

    monkeypatch.setattr(dsh._HOMES, "create", refuse)
    assert provider.get_env(MODEL)["DSH_HOME"] == dsh.UNUSABLE_HOME
    assert "DSH_HOME" not in provider.get_remove_env(base_env={"DSH_HOME": "/user/pinned"})


def test_the_key_survives_the_env_allowlist(provider, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert provider.get_env(MODEL)["OPENROUTER_API_KEY"] == "sk-or-test"
    env = provider.resolve_env(MODEL, base_env={"OPENROUTER_API_KEY": "sk-or-other"})
    assert env["OPENROUTER_API_KEY"] == "sk-or-other"
    assert "OPENROUTER_API_KEY" not in provider.resolve_env(MODEL, base_env={})


def test_exit_127_falls_through_to_the_shared_rows(provider):
    assert provider.classify_failure("", "", 127) == "spawn_failed"


def test_usage_sums_every_step_and_is_absent_when_nothing_was_spent(provider):
    raw = stream(step_end(100, 7), {"type": "tool_call", "callId": "c", "tool": "read", "input": {}},
                 step_end(50, 3), final_ok("[]"))
    assert provider.parse_usage(raw, "") == {"input_tokens": 150, "output_tokens": 10, "tool_uses": 1}
    assert provider.parse_usage(stream(step_end(0, 0), final_err("AUTH", "401")), "") is None


def test_the_cached_prefix_is_reported_apart(provider):
    cached = dict(step_end(134, 101))
    cached["usage"] = dict(cached["usage"], cacheReadTokens=5888)
    raw = stream(step_end(5917, 45), cached, final_ok("[]"))
    assert provider.parse_usage(raw, "") == {
        "input_tokens": 6051, "output_tokens": 146, "cached_input_tokens": 5888, "tool_uses": 0}


def test_the_stdin_dash_normalizes_as_a_positional(provider):
    """expected_argv.yaml keys dsh's stdin row on `tail: ["-"]`: the bare
    `-` is the prompt positional, never a flag named `-`."""
    from tests.golden.normalize_argv import normalize_argv

    argv, _ = provider.build_command_for_transport("prompt", MODEL, "stdin")
    view = normalize_argv(argv)
    assert view["tail"] == ["-"]
    assert "-" not in view["options"]
