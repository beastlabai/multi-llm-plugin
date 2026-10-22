"""fx's terminal envelope must not turn transport or denied tools into answers."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from utils.providers.fx import FxProvider
from utils.providers import fx


def envelope(**overrides):
    value = {
        "output": "narration\n[]", "final_output": "[]", "exit_code": 0,
        "tool_calls": [], "usage": {"input_tokens": 20, "output_tokens": 4},
    }
    value.update(overrides)
    return json.dumps(value)


@pytest.fixture
def provider():
    return FxProvider()


def test_only_final_answer_is_parsed(provider):
    raw = envelope(output='[{"wrong": true}]\n[]')
    assert provider.parse_output(raw, "429 unauthorized") == {"success": True, "data": []}
    assert provider.salvage_text(raw) == "[]"
    assert provider.classify_failure(raw, "fx ask: API request failed · HTTP 401") is None


@pytest.mark.parametrize("changes", [
    {"exit_code": True}, {"exit_code": "0"}, {"exit_code": 1},
    {"output": None}, {"final_output": None}, {"final_output": ""},
    {"error": "InvalidChunk"}, {"auth_failure": {"reason": "invalid credentials"}},
    {"error": {}}, {"error": False}, {"error": None}, {"auth_failure": {}},
    {"tool_calls": [{"name": "shell", "status": "error", "error": "rejected"}]},
    {"output": "", "final_output": "Done."},
    {"tool_calls": None}, {"tool_calls": [None]},
    {"tool_calls": [{"name": "shell", "status": "error", "error": {"category": "rejected", "code": "rejected"}}]},
])
def test_terminal_failures_cannot_be_salvaged(provider, changes):
    raw = envelope(**changes)
    result = provider.parse_output(raw, "")
    assert not result["success"]
    assert result["error_code"] == "SUBPROCESS_FAILED"
    assert provider.salvage_text(raw) == ""
    from utils import subagent_executor as executor
    with pytest.raises(executor.ResultRejected):
        executor._extract_result_and_syntax(provider, raw, "", prefer_arrays=True)


@pytest.mark.parametrize("raw", ["", "[]", "null", "{}", '{"output":', envelope() + envelope()])
def test_malformed_envelopes_are_hard_failures(provider, raw):
    assert provider.parse_output(raw, "")["error_code"] == "SUBPROCESS_FAILED"


def test_legitimate_done_is_answer_text(provider):
    assert provider.salvage_text(envelope(output="Done.", final_output="Done.")) == "Done."


def test_answer_format_failure_allows_file_recovery(provider):
    result = provider.parse_output(envelope(output="Written.", final_output="Written."), "")
    assert not result["success"]
    assert "error_code" not in result


def test_unknown_keys_warn_once_without_discarding_answer(provider, monkeypatch):
    monkeypatch.setattr(fx, "_WARNED_KEYS", set())
    with pytest.warns(UserWarning, match="fx version drift.*future_key"):
        assert provider.parse_output(envelope(future_key=1), "")["success"]
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        provider.parse_output(envelope(future_key=2), "")
    assert not caught


@pytest.mark.parametrize("usage,expected", [
    ({"input_tokens": 20, "output_tokens": 4}, {"input_tokens": 20, "output_tokens": 4}),
    ({"input_tokens": None, "output_tokens": None}, None),
    ({"input_tokens": 0, "output_tokens": None}, {"input_tokens": 0}),
    ({"input_tokens": True, "output_tokens": -1}, None),
    (None, None),
])
def test_usage_has_no_invented_cost_or_tokens(provider, usage, expected):
    assert provider.parse_usage(envelope(usage=usage), "") == expected


@pytest.mark.parametrize("status,body,category", [
    (401, "unauthorized", "auth_failed"),
    (403, "Budget exhausted", "rate_limited"),
    (403, "Access blocked by policy", "permission_denied"),
    (502, "Insufficient credits", "rate_limited"),
    (502, "upstream failed", "server_error"),
    (400, "model/name is not a valid model ID", "invalid_model"),
    (502, "No endpoints found", "invalid_model"),
    (400, "maximum context length is 16385 tokens", "context_length_exceeded"),
    (429, "Too many requests", "rate_limited"),
])
def test_anchored_provider_errors(provider, status, body, category):
    line = f"API request failed · HTTP {status} · {body}\n"
    raw = envelope(output=line, final_output="", exit_code=1)
    assert provider.classify_failure(raw, "", 1) == category
    assert provider.classify_failure("", "fx ask: " + line, 1) == category
    assert provider.classify_failure("", "shell says " + line, 1) is None


@pytest.mark.parametrize("error,category", [
    ("MissingCredentials", "auth_failed"),
    ("IncompleteStream", "server_error"),
    ("InvalidChunk", "server_error"),
    ("InvalidToolName", "server_error"),
    ("NonInteractivePermissionRequired", "runner_error"),
    ("InvalidProfileConfiguration", "runner_error"),
])
def test_typed_error_precedes_other_channels(provider, error, category):
    raw = envelope(error=error, exit_code=1, final_output="")
    assert provider.classify_failure(raw, "fx ask: API request failed · HTTP 429", 1) == category


def test_tool_rejection_is_a_structural_failure(provider):
    raw = envelope(tool_calls=[{"name": "shell", "status": "error", "error": {"code": "rejected"}}])
    status, reason = provider.classify_structural(raw, "")
    assert status == "runner_error"
    assert "refused tool shell" in reason


_CAPTURES = json.loads(
    (Path(__file__).parent / "fixtures/fx/captures.json").read_text()
)["cases"]
_CAPTURE_CATEGORIES = {
    "truncated_stream": "server_error", "http_401": "auth_failed",
    "http_402": "server_error", "http_403": "permission_denied",
    "http_404": "server_error", "malformed_sse": "server_error",
    "writable_settings": "runner_error", "tool_write_file_deny": "server_error",
    "invalid-model": "invalid_model", "context-overflow": "context_length_exceeded",
}
_CAPTURE_ANSWERS = {
    "canonical_success": "canonical repository stub answer",
    "scripted_read_file": "scripted read_file terminal",
    "scripted_write_file": "scripted write_file terminal",
    "scripted_shell": "scripted shell terminal",
    "retry_after_short": "wire option probe answer",
}


@pytest.mark.parametrize("capture", _CAPTURES, ids=lambda row: row["name"])
def test_retained_binary_captures(provider, capture):
    name = capture["name"]
    stdout, stderr = capture["stdout"], capture["stderr"]
    assert provider.classify_failure(stdout, stderr, capture["returncode"]) == _CAPTURE_CATEGORIES.get(name)
    assert provider.salvage_text(stdout) == _CAPTURE_ANSWERS.get(name, "")
    result = provider.parse_output(stdout, stderr)
    if name not in _CAPTURE_ANSWERS:
        assert result["error_code"] == "SUBPROCESS_FAILED"
    else:
        # These stub replies are prose, so the file-output caller may still
        # recover its requested result file after the answer-format failure.
        assert "error_code" not in result


@pytest.mark.parametrize("raw", [
    envelope(output="", final_output="Done."),
    envelope(exit_code=True),
    envelope(error="InvalidChunk"),
    envelope(error={}), envelope(error=False), envelope(auth_failure={}),
    envelope(tool_calls=[{"name": "shell", "status": "error", "error": "rejected"}]),
])
def test_file_output_cannot_recover_a_terminal_failure(provider, monkeypatch, tmp_path, raw):
    from utils import llm_client, provider_registry

    monkeypatch.setitem(provider_registry._PROVIDERS, "fx", provider)

    def invoke(prompt, *args, **kwargs):
        # A file written during this invocation is still invalid when fx's
        # terminal envelope says the run failed.
        Path(prompt).write_text('[{"stale_or_incomplete": true}]')
        return provider.parse_output(raw, "")

    monkeypatch.setattr(llm_client, "invoke_with_provider", invoke)
    result = llm_client.invoke_with_file_output(
        prompt_template="{output_json_path}", model_spec="fx:test-model",
        prompt_context={}, output_dir=tmp_path, phase="test",
    )
    assert not result["success"]
    assert result["error_code"] == "SUBPROCESS_FAILED"
    assert result.get("data") is None


@pytest.mark.parametrize("prompt", ["", " \n", "has\0nul", "bad\ud800"])
def test_invalid_prompts_are_refused_before_flag_probe(provider, monkeypatch, prompt):
    monkeypatch.setattr(provider, "missing_flags", lambda flags: pytest.fail("must refuse before probe"))
    with pytest.raises(fx.FxPromptInvalid):
        provider.build_command(prompt, "test/model")


@pytest.mark.parametrize("transport", ["argv", "stdin"])
def test_prompt_transport_is_independent_of_timeout(provider, monkeypatch, transport):
    monkeypatch.setattr(provider, "missing_flags", lambda flags: ())
    prompt = "-literal --model unrelated"
    short = provider.build_command_for_transport(prompt, "test/model", transport, timeout_s=2)
    assert short == provider.build_command_for_transport(prompt, "test/model", transport, timeout_s=1200)
    cmd, stdin = short
    assert cmd[:7] == ["fx", "ask", "--json", "--no-save", "--auto", "--model", "test/model"]
    assert (cmd[7:], stdin) == ((["--", prompt], None) if transport == "argv" else ([], prompt))


@pytest.mark.parametrize("prompt", ["-h", "--help"])
def test_help_prompt_is_sent_on_stdin(provider, monkeypatch, prompt):
    monkeypatch.setattr(provider, "missing_flags", lambda flags: ())
    argv, stdin = provider.build_command_for_transport(prompt, "test/model", "argv")
    assert stdin == prompt
    assert argv[-1] == "test/model"


def test_wrong_binary_is_refused_by_ask_help(provider, monkeypatch, tmp_path):
    executable = tmp_path / "fx"
    args = tmp_path / "args"
    executable.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{args}"\nprintf "Usage: fx [file] [code]\\n"\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    with pytest.raises(fx.FxFlagsUnconfirmed, match="JSON viewer"):
        provider.build_command("hello", "test/model")
    assert args.read_text().splitlines() == ["ask", "--help"]


def test_bare_env_and_command_build_allocate_no_home(provider, monkeypatch):
    monkeypatch.setattr(fx.tempfile, "mkdtemp", lambda **kw: pytest.fail("allocated a HOME"))
    monkeypatch.setattr(provider, "missing_flags", lambda flags: ())
    assert "HOME" not in provider.get_env("test/model")
    assert "HOME" not in provider.resolve_env("test/model", base_env={})
    provider.build_command("hello", "test/model")


def test_concurrent_dispatches_share_private_home_then_remove_it(provider):
    barrier = Barrier(3)

    def slot(i):
        with provider.hermetic_dispatch("generate", f"test/model{i}"):
            home = Path(provider.get_env("m")["HOME"])
            assert home.stat().st_mode & 0o777 == 0o700
            assert (home / ".fx").stat().st_mode & 0o777 == 0o700
            settings = home / ".fx/settings.json"
            assert settings.stat().st_mode & 0o777 == 0o600
            assert json.loads(settings.read_text()) == fx.render_settings()
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)
            return home

    with ThreadPoolExecutor(max_workers=3) as pool:
        homes = list(pool.map(slot, range(3)))
    assert len(set(homes)) == 1
    assert not homes[0].exists()
    assert fx._ACTIVE_HOME.get() is None
    assert fx._CONFIG_HOME_USERS == 0


def test_env_uses_composed_source_and_removes_ambient_controls(provider, monkeypatch, tmp_path):
    from utils.providers.base import build_child_env
    monkeypatch.setenv(fx.FX_CREDENTIAL_ENV, "wrong-ambient-credential")
    real = tmp_path / "real-home"
    real.mkdir()
    (real / ".gitconfig").write_text("[user]\nname = Test\n")
    source = {
        "HOME": str(real), "PATH": "/usr/bin:/test/canary",
        fx.FX_CREDENTIAL_ENV: "dummy-credential", "FX_MODEL": "wrong/model",
        "FX_PERMISSION_MODE": "full-access", "FX_MAX_AGENT_STEPS": "0",
        "FX_TRACE_REQUESTS": "1", "FX_NEW_OVERRIDE": "bad",
        "AI_GATEWAY_API_KEY": "dummy", "VERCEL_OIDC_TOKEN": "dummy",
        "TYPESAFE_API_KEY": "dummy",
    }
    with provider.hermetic_dispatch("generate"):
        delta = provider.resolve_env("test/model", base_env=source)
        child = build_child_env(provider, provider_env=delta, base_env=source)
        home = Path(child["HOME"])
        assert home != real
        assert child[fx.FX_CREDENTIAL_ENV] == "dummy-credential"
        assert child["GIT_CONFIG_GLOBAL"] == str(real / ".gitconfig")
        assert child["XDG_CONFIG_HOME"] == str(real / ".config")
        assert child["MULTI_LLM_FX_PATH"] == source["PATH"]
        assert child["FX_PERMISSION_MODE"] == "auto"
        assert child["FX_MAX_AGENT_STEPS"] == "60"
        assert not set(child) & {"FX_MODEL", "FX_TRACE_REQUESTS", "FX_NEW_OVERRIDE", "AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN", "TYPESAFE_API_KEY"}
        assert fx.FX_CREDENTIAL_ENV not in provider.resolve_env("m", base_env={})
    assert real.exists() and not home.exists()


def test_login_profile_cannot_restore_skill_virtualenv(provider, monkeypatch, tmp_path):
    from utils.providers import base
    skill_env = tmp_path / "skill-env"
    scripts = skill_env / "bin"
    scripts.mkdir(parents=True)
    python = scripts / "python"
    python.write_text("#!/bin/sh\nexit 99\n")
    python.chmod(0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(scripts, target_is_directory=True)
    monkeypatch.setattr(base.sys, "prefix", str(skill_env))
    monkeypatch.setattr(base.sys, "base_prefix", "/usr")
    source = {"HOME": str(tmp_path), "PATH": os.pathsep.join([str(scripts), str(alias), "/usr/bin", "/bin"])}
    with provider.hermetic_dispatch("generate"):
        child = base.build_child_env(provider, provider_env=provider.resolve_env("m", base_env=source), base_env=source)
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", '. "$HOME/.bash_profile"; printf "%s" "$PATH"'],
            env=child, capture_output=True, text=True, check=True,
        )
    assert result.stdout == "/usr/bin:/bin"


@pytest.mark.parametrize("outcome", ["success", "timeout", "exception"])
def test_lane_a_releases_home_after_child_finishes(provider, monkeypatch, outcome):
    from utils import llm_client, provider_registry
    monkeypatch.setitem(provider_registry._PROVIDERS, "fx", provider)
    monkeypatch.setattr(provider, "is_available", lambda: True)
    monkeypatch.setattr(provider, "missing_flags", lambda flags: ())
    seen = []

    def run(cmd, **kwargs):
        home = Path(kwargs["env"]["HOME"])
        seen.append(home)
        assert home.is_dir()
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1, stderr="[notice] ⚠ Rate limited · HTTP 429")
        if outcome == "exception":
            raise OSError("launch failed")
        return subprocess.CompletedProcess(cmd, 0, envelope(), "")

    monkeypatch.setattr(llm_client, "_run_provider_cli", run)
    result = llm_client.invoke_with_provider("hello", "fx:test/model", timeout=1)
    assert result["success"] is (outcome == "success")
    if outcome == "timeout":
        assert result["error_code"] == "TIMEOUT"
    assert seen and all(not home.exists() for home in seen)
    assert fx._ACTIVE_HOME.get() is None


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_lane_b_releases_home_on_error_and_cancellation(provider, monkeypatch, failure):
    from utils import subagent_executor
    seen = []

    def spawn(**kwargs):
        home = Path(fx._ACTIVE_HOME.get())
        seen.append(home)
        assert home.is_dir()
        raise failure("dispatch interrupted")

    monkeypatch.setattr(subagent_executor, "_spawn_in_context", spawn)
    with pytest.raises(failure):
        subagent_executor._spawn_child(provider=provider, model="test/model", prompt="hi", profile="generate", transport="stdin", timeout=1, cwd=None)
    assert seen and not seen[0].exists()
    assert fx._ACTIVE_HOME.get() is None


def test_executor_generate_uses_the_same_transport(provider, monkeypatch):
    monkeypatch.setattr(provider, "missing_flags", lambda flags: ())
    assert provider.build_agent_command_for_transport("hello", "m", "generate", "stdin") == provider.build_command_for_transport("hello", "m", "stdin")
    from utils.providers.base import ProfileUnsupported
    with pytest.raises(ProfileUnsupported):
        provider.build_agent_command_for_transport("hello", "m", "agent-readonly", "stdin")


def test_executor_deadline_precedes_rate_notice_classification():
    from utils.subagent_executor import _Capture, _classify_pre_parse
    raw = (Path(__file__).parent / "fixtures/fx/rate-limit.stderr.txt").read_text()
    status, reason = _classify_pre_parse(_Capture(stderr=raw, exit_code=-15, timed_out=True))
    assert status == "timeout"
    assert "timeout" in reason
