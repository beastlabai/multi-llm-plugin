"""Non-interactive git env in base ``get_env`` — §2.2 / L9.

An agent-profile run shells out to git. A credential prompt behind
``capture_output=True`` blocks forever and burns the whole slot timeout with two
empty streams; a *GUI-style* credential helper does worse — it burns the timeout
**and** survives the parent's kill as an orphan. Both were measured (75.1 s and
90.1 s) against a private https remote under a scratch ``HOME``.

The fix is two-layered, and the layering is the point:

* **global** (every provider, both lanes): ``GIT_TERMINAL_PROMPT=0`` and
  ``GIT_ASKPASS=""``. Neither suppresses a helper that *answers*, so
  authentication that works today keeps working everywhere.
* **scoped** to ``generate`` / the Lane A fan-out / ``agent-readonly``:
  ``GIT_CONFIG_COUNT`` + ``credential.helper=""``. This one is not free — it
  equally suppresses ``store``, an OS keychain and ``gh auth git-credential`` —
  so it is deliberately **never** applied to ``agent-write``.

This module is §2.2's ship gate, and it is a *behavioural* gate: it drives the
real ``git`` binary against a real credential helper and a real private remote
rather than asserting a dict. Three groups:

1. :class:`TestAnsweringHelper` — a helper that returns credentials promptly.
   ``keep`` authenticates; the wipe fails fast rather than hanging. The cost of
   the wipe is *measured* here, not assumed.
2. :class:`TestHangingHelper` — a helper that never returns. The wipe comes back
   in well under a second and the helper is never even spawned (no orphan);
   ``keep`` genuinely hangs and is bounded only by process-group teardown.
3. :class:`TestGitEnvShape` / :class:`TestCredentialHelperPolicy` — the pure
   assertions, including the **agent-write negative**: the wipe must be absent
   from that profile's env (§14.4's A4 group gate).

**Deviation from §2.2, stated rather than buried:** the remote is served over
``http://127.0.0.1`` instead of ``https://``. Git's credential path (401 →
helper → askpass → terminal) is scheme-independent — the 401 challenge is what
drives it — and a self-signed TLS cert would add a second, unrelated failure
mode (and an ``openssl``/``cryptography`` dependency) to a gate whose whole job
is to be unambiguous about the first.
"""

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import process_control  # noqa: E402
from utils.providers.base import (  # noqa: E402
    GIT_HELPER_AUTO,
    GIT_HELPER_KEEP,
    GIT_HELPER_WIPE,
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
    git_env,
)
from utils.provider_registry import get_provider  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or os.name == "nt",
    reason="the git ship gate needs a POSIX git binary and process groups",
)

#: The only credentials the fixture server accepts.
GIT_USER, GIT_PASSWORD = "gituser", "s3cret"
_BASIC = base64.b64encode(f"{GIT_USER}:{GIT_PASSWORD}".encode()).decode()

#: The wipe's whole promise: fail *fast*. §2.2 measured 0.27 s; the assertion is
#: an order of magnitude looser so a loaded CI box cannot make it flaky, while
#: still being three orders of magnitude below the 90.1 s hang it replaces.
FAST_FAILURE_S = 1.0

#: How long the hanging-helper case waits before concluding git really is stuck.
#: The helper itself sleeps for minutes, so anything that survives this window is
#: hung, not slow.
HANG_WINDOW_S = 2.0


# --------------------------------------------------------------------------- #
# The rig: a bare repo behind HTTP Basic auth, i.e. a "private remote".
# --------------------------------------------------------------------------- #

class _AuthedRepoHandler(BaseHTTPRequestHandler):
    """Serve one bare repo over dumb HTTP, but only with correct credentials.

    The 401 + ``WWW-Authenticate`` challenge is the whole rig: it is what makes
    git consult ``credential.helper``, then ``GIT_ASKPASS``/``core.askPass``,
    then the terminal. Without a challenge none of the machinery under test
    would run at all.
    """

    bare_root: Path = None  # set by the fixture before the server starts

    def log_message(self, *args):  # noqa: D102 - silence the default stderr log
        pass

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.headers.get("Authorization", "") != f"Basic {_BASIC}":
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="git"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        path = self.path.split("?", 1)[0].lstrip("/")
        prefix = "repo.git/"
        if not path.startswith(prefix):
            self.send_error(404)
            return
        target = self.bare_root / path[len(prefix):]
        if not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _git(*args, cwd=None):
    """Run git with the developer's own config firmly out of the way."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True,
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def private_remote():
    """A URL that serves a real git repo and challenges for credentials."""
    root = Path(tempfile.mkdtemp(prefix="multi-llm-gitenv-"))
    try:
        work = root / "work"
        work.mkdir()
        _git("init", "-q", "-b", "main", cwd=work)
        (work / "README.md").write_text("ship gate\n")
        _git("add", "README.md", cwd=work)
        _git("-c", "user.email=t@example.invalid", "-c", "user.name=T",
             "commit", "-qm", "init", cwd=work)
        bare = root / "repo.git"
        _git("clone", "-q", "--bare", str(work), str(bare))
        # Dumb HTTP transport needs the refs/objects indexes materialized.
        _git("--git-dir", str(bare), "update-server-info")

        _AuthedRepoHandler.bare_root = bare
        from tests.container.stub_server import LoopbackHTTPServer

        server = LoopbackHTTPServer(("127.0.0.1", 0), _AuthedRepoHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/repo.git"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _scratch_home(tmp_path: Path, helper_body: str) -> Path:
    """A scratch ``HOME`` whose ``~/.gitconfig`` configures one helper."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    helper = tmp_path / "helper.sh"
    helper.write_text(helper_body)
    helper.chmod(0o755)
    (home / ".gitconfig").write_text(f"[credential]\n\thelper = {helper}\n")
    return home


def answering_helper(tmp_path: Path) -> Path:
    """A helper that returns the right credentials immediately."""
    marker = tmp_path / "answering.invoked"
    return _scratch_home(tmp_path, (
        "#!/bin/sh\n"
        f"echo $$ >> {marker}\n"
        'if [ "$1" = get ]; then\n'
        f"  printf 'username={GIT_USER}\\npassword={GIT_PASSWORD}\\n'\n"
        "fi\n"
    ))


def hanging_helper(tmp_path: Path) -> Path:
    """A helper that records its pid and then never returns.

    This is the GUI-prompt shape §2.2 measured: git blocks on it for the whole
    slot timeout, and killing git alone leaves it running.
    """
    marker = tmp_path / "hanging.pid"
    return _scratch_home(tmp_path, (
        "#!/bin/sh\n"
        f"echo $$ > {marker}\n"
        "sleep 600\n"
    ))


def child_env(home: Path, provider_env: dict) -> dict:
    """Compose the child env exactly as the two spawn sites do."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        **provider_env,
    }


def ls_remote(url: str, env: dict, timeout: float = 30.0):
    """One ``git ls-remote`` against the private remote. Returns (rc, secs, err)."""
    started = time.time()
    proc = subprocess.run(
        ["git", "ls-remote", url], env=env, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=timeout,
    )
    return proc.returncode, time.time() - started, f"{proc.stdout}\n{proc.stderr}"


def readonly_env() -> dict:
    """The env an ``agent-readonly`` dispatch really gets, via the adapter."""
    return get_provider("claude-code").resolve_env("sonnet", PROFILE_AGENT_READONLY)


def write_env() -> dict:
    """The env an ``agent-write`` dispatch really gets, via the adapter."""
    return get_provider("claude-code").resolve_env("sonnet", PROFILE_AGENT_WRITE)


# --------------------------------------------------------------------------- #
# Ship gate (a): a helper that returns credentials promptly.
# --------------------------------------------------------------------------- #

class TestAnsweringHelper:
    """§2.2 ship gate (a) — the cost of the wipe, measured rather than assumed."""

    def test_keep_authenticates(self, private_remote, tmp_path):
        home = answering_helper(tmp_path)
        env = child_env(home, git_env(PROFILE_AGENT_READONLY,
                                      credential_helper=GIT_HELPER_KEEP))
        rc, elapsed, output = ls_remote(private_remote, env)
        assert rc == 0, f"keep must still authenticate: {output}"
        assert "refs/heads/main" in output
        assert (tmp_path / "answering.invoked").exists(), \
            "keep must let the configured helper run"
        assert elapsed < 10.0

    def test_wipe_fails_fast_instead_of_authenticating(self, private_remote, tmp_path):
        home = answering_helper(tmp_path)
        env = child_env(home, git_env(PROFILE_AGENT_READONLY,
                                      credential_helper=GIT_HELPER_AUTO))
        rc, elapsed, output = ls_remote(private_remote, env)
        assert rc != 0, "the wipe must not reach a private remote"
        assert "terminal prompts disabled" in output, output
        assert elapsed < FAST_FAILURE_S, f"expected a fast failure, took {elapsed:.2f}s"
        assert not (tmp_path / "answering.invoked").exists(), \
            "the wipe must stop the helper being invoked at all"

    def test_agent_write_keeps_the_users_helper(self, private_remote, tmp_path):
        """The A4 group gate, live: agent-write authenticates like today."""
        home = answering_helper(tmp_path)
        rc, _elapsed, output = ls_remote(private_remote, child_env(home, write_env()))
        assert rc == 0, f"agent-write must never get the wipe: {output}"
        assert "refs/heads/main" in output

    def test_agent_readonly_via_the_adapter_fails_fast(self, private_remote, tmp_path):
        """The same wipe, reached the way a dispatch reaches it."""
        home = answering_helper(tmp_path)
        rc, elapsed, output = ls_remote(private_remote, child_env(home, readonly_env()))
        assert rc != 0 and "terminal prompts disabled" in output, output
        assert elapsed < FAST_FAILURE_S


# --------------------------------------------------------------------------- #
# Ship gate (b): a helper that hangs.
# --------------------------------------------------------------------------- #

class TestHangingHelper:
    """§2.2 ship gate (b) — the 90.1 s hang, and the orphan it left behind."""

    def test_wipe_returns_fast_and_leaves_no_orphan(self, private_remote, tmp_path):
        home = hanging_helper(tmp_path)
        env = child_env(home, git_env(PROFILE_AGENT_READONLY,
                                      credential_helper=GIT_HELPER_AUTO))
        rc, elapsed, output = ls_remote(private_remote, env, timeout=30.0)
        assert rc != 0
        assert elapsed < FAST_FAILURE_S, \
            f"the hanging helper must not be waited on at all, took {elapsed:.2f}s"
        marker = tmp_path / "hanging.pid"
        assert not marker.exists(), \
            "the wipe must prevent the helper from being spawned in the first place"

    def test_keep_hangs_and_is_bounded_only_by_process_group_teardown(
        self, private_remote, tmp_path
    ):
        """The control: without the wipe the hang is real, and so is the orphan.

        This is what makes the assertion above non-vacuous. ``keep`` is the
        documented escape hatch, and its only bound is the process-group kill —
        never a fast failure.
        """
        home = hanging_helper(tmp_path)
        env = child_env(home, git_env(PROFILE_AGENT_READONLY,
                                      credential_helper=GIT_HELPER_KEEP))
        proc = subprocess.Popen(
            ["git", "ls-remote", private_remote], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True,
            **process_control.new_process_group_kwargs(),
        )
        pgid = process_control.process_group_id(proc.pid)
        marker = tmp_path / "hanging.pid"
        try:
            deadline = time.time() + HANG_WINDOW_S
            while time.time() < deadline and not marker.exists():
                time.sleep(0.05)
            assert marker.exists(), "keep must let the hanging helper run"
            assert proc.poll() is None, \
                "keep must hang on the helper (this is the measured 90.1 s case)"
            helper_pid = int(marker.read_text().strip())
            assert process_control.process_alive(helper_pid)

            # Group teardown, then reap — the order `_kill_timed_out` +
            # `_drain_after_kill` use. Without the reap the leader lingers as a
            # zombie and `process_alive` keeps answering True for it, which is an
            # artifact of this test being the parent, not of the kill.
            verdict = process_control.kill_tree(proc.pid, pgid=pgid, escalate_after_s=2.0)
            assert verdict != "skipped", verdict
            proc.communicate(timeout=10)
            assert proc.poll() is not None, "the leader must be gone after teardown"
            deadline = time.time() + 5.0
            while time.time() < deadline and process_control.process_alive(helper_pid):
                time.sleep(0.05)
            assert not process_control.process_alive(helper_pid), \
                "process-group teardown must take the helper down with git"
        finally:
            if proc.poll() is None:
                process_control.kill_tree(proc.pid, pgid=pgid, escalate_after_s=1.0)
                try:
                    proc.communicate(timeout=10)
                except Exception:  # noqa: BLE001 - teardown must not mask the failure
                    pass


# --------------------------------------------------------------------------- #
# The env itself.
# --------------------------------------------------------------------------- #

WIPE_KEYS = ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0")


class TestGitEnvShape:
    """The global half applies everywhere; the wipe is scoped."""

    @pytest.mark.parametrize(
        "profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE]
    )
    def test_the_global_block_is_on_every_profile(self, profile):
        env = git_env(profile)
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_ASKPASS"] == ""
        assert env["SSH_ASKPASS"] == ""
        assert env["SSH_ASKPASS_REQUIRE"] == "never"
        assert env["GCM_INTERACTIVE"] == "never"

    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    def test_the_wipe_is_scoped_to_the_read_only_lanes(self, profile):
        env = git_env(profile)
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
        assert env["GIT_CONFIG_VALUE_0"] == ""

    def test_agent_write_never_gets_the_wipe(self):
        """§14.4's A4 gate: the negative assertion, in the same file."""
        env = git_env(PROFILE_AGENT_WRITE)
        for key in WIPE_KEYS:
            assert key not in env, f"agent-write must not carry {key}"

    def test_agent_write_never_gets_the_wipe_through_the_adapter(self):
        for key in WIPE_KEYS:
            assert key not in write_env(), f"agent-write must not carry {key}"

    def test_dropped_keys_are_really_dropped(self):
        """§2.2 cut these explicitly; re-adding them would be inert noise."""
        env = git_env(PROFILE_AGENT_READONLY)
        assert "LC_ALL" not in env
        injected = {env.get(f"GIT_CONFIG_KEY_{n}") for n in range(4)}
        assert "credential.interactive" not in injected
        assert "credential.guiPrompt" not in injected

    def test_an_inherited_git_config_injection_is_appended_not_clobbered(self):
        """A caller's own ``GIT_CONFIG_*`` keys survive; ours goes last.

        Last wins, and for a multi-valued key an empty value *resets the list*,
        so ours has to be the final entry to suppress anything below it.
        """
        env = git_env(PROFILE_AGENT_READONLY, base_env={
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "http.sslVerify",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_KEY_1": "core.pager",
            "GIT_CONFIG_VALUE_1": "cat",
        })
        assert env["GIT_CONFIG_COUNT"] == "3"
        assert env["GIT_CONFIG_KEY_2"] == "credential.helper"
        assert env["GIT_CONFIG_VALUE_2"] == ""
        assert "GIT_CONFIG_KEY_0" not in env and "GIT_CONFIG_KEY_1" not in env

    def test_adapter_env_still_wins_over_the_git_block(self):
        """``resolve_env`` composes; the adapter's own keys are last."""
        aider = get_provider("aider")
        env = aider.resolve_env("openrouter/z-ai/glm-5.2", PROFILE_GENERATE)
        assert env["BROWSER"] == "true"
        assert env["COLUMNS"] == "10000"
        assert env["GIT_TERMINAL_PROMPT"] == "0"

    @pytest.mark.parametrize("name", ["aider", "goose", "codex", "opencode", "gemini"])
    def test_every_adapter_gets_the_global_block(self, name):
        env = get_provider(name).resolve_env("m", PROFILE_GENERATE)
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_ASKPASS"] == ""


class TestCredentialHelperPolicy:
    """``git_env.credential_helper`` — the off switch, both directions."""

    def test_keep_removes_the_wipe_everywhere(self):
        for profile in (PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE):
            env = git_env(profile, credential_helper=GIT_HELPER_KEEP)
            for key in WIPE_KEYS:
                assert key not in env
            assert env["GIT_TERMINAL_PROMPT"] == "0"

    def test_wipe_applies_it_everywhere_including_agent_write(self):
        env = git_env(PROFILE_AGENT_WRITE, credential_helper=GIT_HELPER_WIPE)
        assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
        assert env["GIT_CONFIG_VALUE_0"] == ""

    @pytest.mark.config_override
    def test_env_var_overrides_config_for_one_run(self, monkeypatch):
        import utils.provider_registry as registry
        registry._config = None
        monkeypatch.setenv("MULTI_LLM_GIT_CREDENTIAL_HELPER", "keep")
        env = get_provider("claude-code").resolve_env("sonnet", PROFILE_AGENT_READONLY)
        for key in WIPE_KEYS:
            assert key not in env

    @pytest.mark.config_override
    def test_a_global_config_key_switches_it_off(self, tmp_path, monkeypatch):
        cfg = tmp_path / "providers.yaml"
        cfg.write_text("git_env:\n  credential_helper: keep\n")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(cfg))
        import utils.provider_registry as registry
        registry._config = None
        env = get_provider("claude-code").resolve_env("sonnet", PROFILE_AGENT_READONLY)
        for key in WIPE_KEYS:
            assert key not in env

    @pytest.mark.config_override
    def test_a_per_provider_key_beats_the_global_one(self, tmp_path, monkeypatch):
        cfg = tmp_path / "providers.yaml"
        cfg.write_text(
            "git_env:\n  credential_helper: keep\n"
            "providers:\n  codex:\n    git_env:\n      credential_helper: wipe\n"
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(cfg))
        import utils.provider_registry as registry
        registry._config = None
        assert "GIT_CONFIG_COUNT" in get_provider("codex").resolve_env(
            "gpt-5.6-sol", PROFILE_AGENT_READONLY)
        assert "GIT_CONFIG_COUNT" not in get_provider("claude-code").resolve_env(
            "sonnet", PROFILE_AGENT_READONLY)

    @pytest.mark.config_override
    def test_an_unknown_config_value_is_rejected_at_load_time(self, tmp_path, monkeypatch):
        from utils.provider_registry import ConfigError, load_config
        cfg = tmp_path / "providers.yaml"
        cfg.write_text("git_env:\n  credential_helper: keeep\n")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(cfg))
        import utils.provider_registry as registry
        registry._config = None
        with pytest.raises(ConfigError) as excinfo:
            load_config()
        assert "credential_helper" in str(excinfo.value)
        assert "keeep" in str(excinfo.value)

    @pytest.mark.config_override
    def test_an_unknown_env_value_falls_back_to_auto_with_a_warning(
        self, monkeypatch, capsys
    ):
        import utils.provider_registry as registry
        registry._config = None
        monkeypatch.setenv("MULTI_LLM_GIT_CREDENTIAL_HELPER", "nope")
        env = get_provider("claude-code").resolve_env("sonnet", PROFILE_AGENT_READONLY)
        assert env["GIT_CONFIG_KEY_0"] == "credential.helper"
        assert "MULTI_LLM_GIT_CREDENTIAL_HELPER" in capsys.readouterr().err
