"""The hermes adapter: argv, private homes, the gateway stop and the stream reader.

The captured rows themselves are graded by ``test_provider_fixture_rows.py``
from ``tests/fixtures/hermes/manifest.yaml``; this file covers what a capture
cannot show: the argv per transport and effort, the dispatch lifecycle, and the
hand-made stream shapes the parser must refuse.
"""
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import hermes  # noqa: E402
from utils.providers.base import ProfileUnsupported  # noqa: E402
from utils.providers.types import PROFILE_GENERATE  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "hermes"


def _read(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def provider(monkeypatch):
    p = hermes.HermesProvider()
    monkeypatch.setattr(p, "missing_flags", lambda flags: ())
    return p


POSIX_MODES = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")


@pytest.fixture(autouse=True)
def _homes(tmp_path, monkeypatch):
    # `tempfile.tempdir`, not TMPDIR: Windows reads TMP/TEMP.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield
    hermes.cleanup_home_dirs()


class TestArgv:
    BASE = [
        "hermes", "chat", "--format", "stream-json", "--toolsets", "", "--ignore-rules",
        "--source", "tool", "--provider", "openrouter", "--max-turns", str(hermes.MAX_TURNS),
    ]

    def test_generate_is_toolless_and_trails_the_prompt(self, provider):
        assert provider.build_command("Review it", "openai/gpt-6-luna") == (
            self.BASE + ["-m", "openai/gpt-6-luna", "-q", "Review it"]
        )

    def test_stdin_replaces_the_prompt_pair(self, provider):
        cmd, stdin = provider.build_command_for_transport("Review it", "m/x", "stdin")
        assert cmd == self.BASE + ["-m", "m/x", "--query-file", "-"]
        assert stdin == "Review it"

    def test_a_dash_prompt_always_goes_on_stdin(self, provider):
        cmd, stdin = provider.build_command_for_transport("--help me", "m/x", "argv")
        assert cmd[-2:] == ["--query-file", "-"] and stdin == "--help me"

    def test_argv_transport_keeps_the_prompt(self, provider):
        cmd, stdin = provider.build_command_for_transport("Review it", "m/x", "argv")
        assert cmd[-2:] == ["-q", "Review it"] and stdin is None

    def test_file_transport_is_refused(self, provider):
        with pytest.raises(ValueError):
            provider.build_command_for_transport("p", "m/x", "file", "/tmp/p")

    @pytest.mark.parametrize("suffix,emitted", [
        ("high", "high"), ("none", "none"), ("max", "max"), ("ultra", "max"),
    ])
    def test_effort_suffix_renders_reasoning(self, provider, suffix, emitted):
        cmd = provider.build_command("p", f"openai/gpt-6-luna:{suffix}")
        assert cmd[cmd.index("-m") + 1] == "openai/gpt-6-luna"
        assert cmd[cmd.index("--reasoning") + 1] == emitted

    def test_an_unknown_suffix_stays_in_the_model_id(self, provider):
        cmd = provider.build_command("p", "meta/llama:free")
        assert cmd[cmd.index("-m") + 1] == "meta/llama:free"
        assert "--reasoning" not in cmd

    def test_a_binary_without_the_flags_is_refused(self, monkeypatch):
        p = hermes.HermesProvider()
        monkeypatch.setattr(p, "missing_flags", lambda flags: ("--format",))
        monkeypatch.setattr(p, "gate_binary", lambda runnable=False: "/usr/bin/hermes")
        with pytest.raises(hermes.HermesFlagsUnconfirmed, match="--format"):
            p.build_command("p", "m/x")

    @pytest.mark.skipif(os.name == "nt", reason="a POSIX shell script stands in for hermes")
    @pytest.mark.parametrize("complete", [True, False])
    def test_the_gate_really_probes_chat_help_under_a_private_home(
            self, tmp_path, monkeypatch, complete):
        """No stubbed ``missing_flags``: the gate must find ``hermes`` on PATH
        (there is no ``version_cmd`` to name it), run ``chat --help`` with a
        private HERMES_HOME, and refuse a help text missing a flag."""
        from utils.providers import version_gate

        flags = hermes.HERMES_REQUIRED_FLAGS if complete else hermes.HERMES_REQUIRED_FLAGS[1:]
        bindir = tmp_path / "bin"
        bindir.mkdir()
        seen = tmp_path / "seen"
        script = bindir / "hermes"
        script.write_text("#!/bin/sh\n"
                          f"echo \"$HERMES_HOME $*\" >> '{seen}'\n"
                          f"echo 'usage: hermes chat {' '.join(flags)}'\n")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "users-own-hermes"))
        version_gate.reset_flag_gate_cache()
        p = hermes.HermesProvider()
        try:
            if complete:
                assert p.build_command("p", "m/x")[0] == "hermes"
            else:
                with pytest.raises(hermes.HermesFlagsUnconfirmed, match="--format"):
                    p.build_command("p", "m/x")
        finally:
            version_gate.reset_flag_gate_cache()
        [probe] = seen.read_text().splitlines()
        home, _, argv = probe.partition(" ")
        assert argv == "chat --help"
        assert home != str(tmp_path / "users-own-hermes")
        assert Path(home).name.startswith("multi-llm-hermes-")

    def test_agent_profiles_are_refused(self, provider):
        assert provider.agent_profiles == (PROFILE_GENERATE,)
        with pytest.raises(ProfileUnsupported):
            provider.build_agent_command("p", "m/x", "agent-readonly")


class TestDispatchLifecycle:
    def test_each_dispatch_gets_a_seeded_private_root(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc123")
        stopped = []
        monkeypatch.setattr(hermes, "stop_gateway", lambda root, **_: stopped.append(root))
        with provider.hermetic_dispatch(PROFILE_GENERATE, "m/x") as refusal:
            assert refusal is None
            env = provider.get_env("m/x")
            root = env["HERMES_HOME"]
            assert env["HOME"] == os.path.join(root, "home")
            assert env["XDG_STATE_HOME"].startswith(env["HOME"])
            env_file = Path(root, ".env")
            if os.name != "nt":
                assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
                assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
            lines = env_file.read_text().splitlines()
            assert "OPENROUTER_API_KEY=sk-or-v1-abc123" in lines
            assert f"OPENROUTER_BASE_URL={hermes.BASE_URL}" in lines
            assert json.loads(Path(root, "config.yaml").read_text()) == hermes.HERMES_CONFIG
            with provider.hermetic_dispatch(PROFILE_GENERATE, "m/x"):
                assert provider.get_env("m/x")["HERMES_HOME"] != root
        assert stopped[-1] == root
        assert len(stopped) == 2

    def test_the_gateway_is_stopped_when_the_dispatch_raises(self, provider, monkeypatch):
        stopped = []
        monkeypatch.setattr(hermes, "stop_gateway", lambda root, **_: stopped.append(root))
        with pytest.raises(RuntimeError):
            with provider.hermetic_dispatch(PROFILE_GENERATE, "m/x"):
                raise RuntimeError("spawn failed")
        assert len(stopped) == 1

    def test_a_root_that_cannot_be_made_refuses_the_slot(self, provider, monkeypatch):
        def boom():
            raise OSError("no space")
        monkeypatch.setattr(hermes._HOMES, "create", boom)
        with provider.hermetic_dispatch(PROFILE_GENERATE, "m/x") as refusal:
            assert "private HERMES_HOME" in refusal

    def test_an_unsafe_key_is_not_written(self, tmp_path):
        hermes.seed_home(str(tmp_path), "bad\nkey")
        assert "OPENROUTER_API_KEY" not in (tmp_path / ".env").read_text()

    def test_outside_a_dispatch_one_credential_free_root_serves(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc123")
        env = provider.get_env("m/x")
        assert Path(env["HERMES_HOME"]).name.startswith("multi-llm-hermes-")
        assert "OPENROUTER_API_KEY" not in Path(env["HERMES_HOME"], ".env").read_text()
        assert provider.get_env("m/x")["HERMES_HOME"] == env["HERMES_HOME"]
        assert provider.help_probe_env()["HERMES_HOME"] == env["HERMES_HOME"]

    def test_the_help_probe_runs_in_the_private_root(self, provider):
        assert provider.gate_probe_env()["HERMES_HOME"] == provider.help_probe_env()["HERMES_HOME"]


class TestStopGateway:
    def test_without_a_binary_nothing_is_spawned(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(hermes.shutil, "which", lambda name: None)
        monkeypatch.setattr(hermes.subprocess, "run", lambda *a, **k: calls.append(a))
        hermes.stop_gateway(str(tmp_path))
        assert calls == []

    def test_the_stop_runs_with_its_own_home_over_the_spawn_env(self, tmp_path, monkeypatch):
        seen = {}

        def run(argv, **kwargs):
            seen.update(argv=argv, env=kwargs["env"])
        monkeypatch.setattr(hermes.subprocess, "run", run)
        hermes.stop_gateway(str(tmp_path), binary="/bin/hermes",
                            env={"PATH": "/bin", "XDG_RUNTIME_DIR": "/run/user/1", "HOME": "/real"})
        assert seen["argv"] == ["/bin/hermes", "gateway", "stop", "--yes"]
        assert seen["env"]["HERMES_HOME"] == str(tmp_path)
        assert seen["env"]["HOME"] == str(tmp_path / "home")
        assert seen["env"]["XDG_RUNTIME_DIR"] == "/run/user/1"

    def test_a_failing_stop_never_raises(self, tmp_path, monkeypatch):
        def run(*a, **k):
            raise OSError("gone")
        monkeypatch.setattr(hermes.subprocess, "run", run)
        hermes.stop_gateway(str(tmp_path), binary="/bin/hermes")


class TestEnvironment:
    def test_ambient_hermes_names_and_the_key_are_stripped(self, provider):
        removed = provider.get_remove_env(base_env={
            "HERMES_HOME": "/x", "HERMES_TUI_GATEWAY_URL": "ws://x", "HERMES_FOO": "1", "PATH": "/bin",
        })
        assert "HERMES_HOME" not in removed
        assert {"HERMES_TUI_GATEWAY_URL", "HERMES_FOO", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"} <= set(removed)
        assert "PATH" not in removed


def _stream(*records):
    return "\n".join(json.dumps(r) for r in records) + "\n"


INIT = {"type": "system", "subtype": "init", "model": "m", "session_id": ""}


class TestStreamReader:
    def test_the_final_message_is_the_answer(self, provider):
        out = _read("multi-assistant.stdout.txt")
        parsed = provider.parse_output(out, "")
        assert parsed["success"] and parsed["data"][0]["id"] == "S001"
        assert "D999" not in provider.salvage_text(out)

    def test_a_cut_stream_is_a_failure(self, provider):
        out = _stream(INIT, {"type": "text", "text": "[{\"id\": 1"})
        parsed = provider.parse_output(out, "")
        assert not parsed["success"] and "no result record" in parsed["error"]
        assert provider.salvage_text(out) == ""

    def test_an_error_field_fails_even_at_exit_code_zero(self, provider):
        out = _stream(INIT, {"type": "text", "text": "[]"},
                      {"type": "result", "exit_code": 0, "text": "[]", "error": "HTTP 500: boom"})
        assert not provider.parse_output(out, "")["success"]

    def test_hermes_own_sentence_without_text_records_is_no_reply(self, provider):
        out = _read("exit-zero-failure.stdout.txt")
        parsed = provider.parse_output(out, "")
        assert not parsed["success"] and "no reply" in parsed["error"]
        assert provider.salvage_text(out) == ""

    def test_a_malformed_result_record_is_refused(self, provider):
        out = _stream(INIT, {"type": "text", "text": "x"}, {"type": "result", "exit_code": "0", "text": "x"})
        assert "malformed" in provider.parse_output(out, "")["error"]

    def test_usage_comes_from_the_result_tokens(self, provider):
        assert provider.parse_usage(_read("success.stdout.txt"), "") == {
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cached_input_tokens": 0,
        }
        assert provider.parse_usage(_stream(INIT), "") is None


class TestClassification:
    def test_vendor_text_drops_the_answer_but_keeps_failures(self, provider):
        out, _ = provider.vendor_text(_read("success.stdout.txt"), "")
        assert "Bound the retry ladder" not in out
        failed, _ = provider.vendor_text(_read("auth-error.stdout.txt"), "")
        assert "HTTP 401" in failed and "rejected your API key" in failed

    def test_an_answer_quoting_a_rate_limit_is_not_one(self, provider):
        out = _stream(INIT, {"type": "text", "text": "429 rate limit exceeded is risk R1"},
                      {"type": "result", "exit_code": 0, "text": "429 rate limit exceeded is risk R1"})
        assert provider.classify_failure(out, "", 0) is None
        assert provider.classify_exit_zero(out, "") is None

    @pytest.mark.parametrize("error,expected", [
        ("HTTP 402: Insufficient credits", "rate_limited"),
        ("HTTP 403: Forbidden", "permission_denied"),
        ("HTTP 400: This model's maximum context length is 8192 tokens", "context_length_exceeded"),
        ("HTTP 503: upstream unavailable", "server_error"),
    ])
    def test_the_error_field_decides(self, provider, error, expected):
        out = _stream(INIT, {"type": "result", "exit_code": 1, "text": "x", "error": error})
        assert provider.classify_failure(out, "", 1) == expected

    def test_the_compression_floor_is_not_a_context_overflow(self, provider):
        out = _read("context-overflow.stdout.txt")
        assert provider.classify_failure(out, "", 1) != "context_length_exceeded"
