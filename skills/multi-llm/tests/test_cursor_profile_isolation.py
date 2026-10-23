"""The per-run cursor profile redirect (§2.3's shape, applied to cursor-agent).

`cursor-agent --print --model <m>` rewrites the user's **global default model**
in `cli-config.json`, and `--trust` writes a durable `.workspace-trusted`
marker. The shipped mitigation was a snapshot/restore guard armed at the first
spawn and replayed at process exit. Measured, that guard is:

* **correct within one process** — two rows share the first (clean) snapshot,
  unrelated keys survive; and
* **broken across two overlapping processes** — ``_ARMED`` is a module-level
  global, so a second orchestrator that arms *after* the first has already
  spawned a cursor row snapshots the **polluted** value as if it were the
  user's, and whichever process exits last writes that back. The user starts on
  ``gpt-5.6-sol`` and ends on ``composer-2.5``, permanently.

The fix is not a better guard, it is **not touching the user's profile at
all**: every run gets its own ``CURSOR_CONFIG_DIR`` + ``CURSOR_DATA_DIR``. That
is safe because cursor resolves credentials through a *third* rule that
consults neither variable (``$XDG_CONFIG_HOME/cursor/auth.json`` on Linux,
verified 0600 on the probe machine) — which is why the redirect does **not**
present as a logged-out CLI, and why ``XDG_CONFIG_HOME`` itself is never
redirected.

Every test here redirects both variables (or ``HOME``/``XDG_CONFIG_HOME``) into
``tmp_path``: nothing in this file may reach the developer's real ``~/.cursor``
or ``~/.config/cursor``. No test spends a cursor call — a "child" is a five-line
function that rewrites the same keys the real CLI does.
"""

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import cursor_agent as cursor_agent_module
from utils.providers.base import PROFILE_GENERATE
from utils.providers.cursor_agent import CursorAgentProvider

#: The user's own settings, as they exist before any multi-llm run.
USER_CONFIG = {
    "version": 1,
    "editor": "vim",
    "model": {"modelId": "gpt-5.6-sol"},
    "selectedModel": {
        "modelId": "gpt-5.6-sol",
        "parameters": [{"id": "reasoning", "value": "xhigh"}],
    },
    "modelSelectionHistory": ["gpt-5.6-sol"],
    "modelParameters": {"gpt-5.6-sol": {"reasoning": "xhigh"}},
    "hasChangedDefaultModel": True,
}


@pytest.fixture
def provider():
    return CursorAgentProvider()


@pytest.fixture(autouse=True)
def _clean_module_state(tmp_path, monkeypatch):
    """No cached help read, no armed guard and no leaked profile dir.

    ``TMPDIR`` is pinned inside ``tmp_path`` so a redirected profile dir is
    visible to the test *and* cannot survive the session even if cleanup
    regressed — which is a property the cleanup tests then assert directly.
    """
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    # `tempfile` caches `gettempdir()` on first use, so the env var alone would
    # not move an already-warmed process. Pin the module attribute too.
    monkeypatch.setattr(cursor_agent_module.tempfile, "tempdir", str(scratch))
    monkeypatch.delenv("CURSOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CURSOR_DATA_DIR", raising=False)
    cursor_agent_module.reset_flag_gate_cache()
    cursor_agent_module.reset_side_effect_guard()
    cursor_agent_module.reset_profile_state()
    yield
    cursor_agent_module.reset_flag_gate_cache()
    cursor_agent_module.reset_side_effect_guard()
    cursor_agent_module.reset_profile_state()


@pytest.fixture
def user_profile(tmp_path, monkeypatch):
    """The *user's own* cursor profile, at the paths cursor really resolves.

    Neither ``CURSOR_CONFIG_DIR`` nor ``CURSOR_DATA_DIR`` is set — setting them
    is the user pinning their own profile, which suppresses the redirect, and
    that is a different test. ``HOME`` and ``XDG_CONFIG_HOME`` are moved into
    ``tmp_path`` instead, so "the user's real config path" in this file is a
    scratch path and the developer's own profile is never reachable.
    """
    home = tmp_path / "home"
    xdg = home / ".config"
    config_dir = xdg / "cursor"     # cli-config.json  ← $XDG_CONFIG_HOME/cursor
    data_dir = home / ".cursor"     # projects/<slug>/ ← ~/.cursor
    (config_dir).mkdir(parents=True)
    (data_dir / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    # Windows' `expanduser` reads USERPROFILE, never HOME: unset, `~` there is
    # the runner's real profile, which both cursor's rules and `child_run` use.
    monkeypatch.setenv("USERPROFILE", str(home))

    config = config_dir / "cli-config.json"
    config.write_text(json.dumps(USER_CONFIG, indent=2) + "\n", encoding="utf-8")
    # cursor's own auth file — the reason the redirect is safe. It follows
    # NEITHER redirected variable, so it must still be reachable afterwards.
    auth = config_dir / "auth.json"
    auth.write_text(json.dumps({"accessToken": "not-a-real-token"}), encoding="utf-8")
    auth.chmod(0o600)
    return {"config_dir": config_dir, "data_dir": data_dir, "config": config, "auth": auth}


# ---------------------------------------------------------------------------
# A "child": what a real cursor-agent run does to whichever profile it is given.
# Resolved from the *merged* env a spawn would really see, using cursor's own
# three rules — never by asking our adapter where it thinks the file is.
# ---------------------------------------------------------------------------


def _resolved_config_dir(env):
    merged = {**os.environ, **(env or {})}
    explicit = (merged.get("CURSOR_CONFIG_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    xdg = (merged.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        return Path(xdg) / "cursor"
    return Path(os.path.expanduser("~")) / ".cursor"


def _resolved_data_dir(env):
    merged = {**os.environ, **(env or {})}
    explicit = (merged.get("CURSOR_DATA_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    return Path(os.path.expanduser("~")) / ".cursor"


def child_run(env, model, *, workspace=None):
    """Rewrite exactly what ``--model``/``--trust`` rewrite, where they write it."""
    config_dir = _resolved_config_dir(env)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "cli-config.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        document = {}
    document["model"] = {"modelId": model}
    document["selectedModel"] = {"modelId": model, "parameters": []}
    document["modelSelectionHistory"] = [model] + [
        m for m in document.get("modelSelectionHistory", []) if m != model
    ]
    document["modelParameters"] = {**document.get("modelParameters", {}), model: {}}
    document["hasChangedDefaultModel"] = True
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    if workspace is not None:
        marker = _resolved_data_dir(env) / "projects" / "slug-of-the-run" / ".workspace-trusted"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "trustedAt": "2026-08-17T09:00:00.000Z",
                    "workspacePath": workspace,
                    "trustMethod": "cli-flag",
                }
            ),
            encoding="utf-8",
        )
    return path


def _selected(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["selectedModel"]["modelId"]


def _write_override(tmp_path, monkeypatch, value):
    """Point ``MULTI_LLM_PROVIDERS_CONFIG`` at a one-key cursor override."""
    import utils.provider_registry as registry

    override = tmp_path / "providers-override.yaml"
    override.write_text(
        "providers:\n  cursor-agent:\n    isolate_profile: " + f'"{value}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
    registry._config = None
    registry._config_key = None
    return override


@pytest.fixture
def redirect_here(tmp_path, monkeypatch):
    """The redirect, on whichever platform runs this file.

    ``auto`` redirects only where cursor's auth rule was measured (Linux); a
    macOS or Windows user gets the same redirect with ``isolate_profile: on``,
    so off Linux the tests of the redirect's own mechanics opt in that way. On
    Linux nothing is written and they keep covering the shipped ``auto``.
    """
    if not sys.platform.startswith(cursor_agent_module._AUTH_UNAFFECTED_PLATFORMS):
        _write_override(tmp_path, monkeypatch, "on")


# ---------------------------------------------------------------------------
# The two cases from the reproduction script, as real tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("redirect_here")
class TestTheReproductionScript:
    """Both cases the probe demonstrated, driven through the real spawn hook."""

    def test_case_a_two_rows_one_process(self, provider, user_profile):
        """Intra-run: two concurrent rows must leave the user on their own model.

        Green before this change (that is what "first arm wins" bought) and it
        must stay green: the redirect may not regress the case the guard got
        right.
        """
        config = user_profile["config"]
        before = config.read_bytes()

        env_one = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        child_run(env_one, "composer-2.5")
        env_two = provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        child_run(env_two, "gemini-3.6-flash-high")

        cursor_agent_module.restore_side_effects()
        cursor_agent_module.cleanup_isolated_profiles()

        assert _selected(config) == "gpt-5.6-sol"
        assert json.loads(config.read_text(encoding="utf-8"))["editor"] == "vim"
        assert config.read_bytes() == before

    def test_case_b_two_overlapping_processes(self, provider, user_profile):
        """Cross-process: the defect. Red under the guard, green under the redirect.

        ``reset_side_effect_guard()`` is a faithful stand-in for a second
        process — ``_ARMED`` is the only cross-row state the guard has, so
        dropping it is exactly what a fresh interpreter presents. Under the
        guard, P2 snapshots P1's pollution as if it were the user's setting and
        restores it last. Under the redirect, neither process ever names the
        user's file.
        """
        config = user_profile["config"]
        before = config.read_bytes()

        # ---- P1 -------------------------------------------------------
        p1_env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        p1_guard = cursor_agent_module.armed_snapshot()
        p1_profiles = cursor_agent_module.isolated_profile_dirs()
        child_run(p1_env, "composer-2.5")

        # ---- P2 is a *different* process ------------------------------
        cursor_agent_module.reset_side_effect_guard()
        cursor_agent_module.reset_run_profile_dir()
        p2_env = provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        p2_guard = cursor_agent_module.armed_snapshot()
        child_run(p2_env, "gemini-3.6-flash-high")

        # ---- they exit in the order that used to lose the user's value -
        if p1_guard is not None:
            p1_guard.restore()
        if p2_guard is not None:
            p2_guard.restore()

        assert _selected(config) == "gpt-5.6-sol", (
            "a second overlapping run snapshotted the first run's pollution and "
            "restored it over the user's own default model"
        )
        assert config.read_bytes() == before
        # ...and the two runs really were isolated from each other, which is the
        # property that makes the ordering above irrelevant.
        assert p1_profiles, "P1 got no profile of its own"
        assert p1_env["CURSOR_CONFIG_DIR"] != p2_env["CURSOR_CONFIG_DIR"]

        cursor_agent_module.cleanup_isolated_profiles()


# ---------------------------------------------------------------------------
# The redirect itself
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("redirect_here")
class TestRedirect:
    def test_resolve_env_names_both_cursor_dirs(self, provider, user_profile):
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert env["CURSOR_CONFIG_DIR"] == env["CURSOR_DATA_DIR"]
        assert Path(env["CURSOR_CONFIG_DIR"]).is_dir()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits: Windows has no 0700 to read back")
    def test_the_profile_dir_is_0700(self, provider, user_profile):
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        mode = stat.S_IMODE(os.stat(env["CURSOR_CONFIG_DIR"]).st_mode)
        assert mode == 0o700, oct(mode)

    def test_xdg_config_home_is_never_redirected(self, provider, user_profile):
        """Redirecting it would move ``auth.json`` and log the CLI out."""
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert "XDG_CONFIG_HOME" not in env
        assert "HOME" not in env
        assert os.environ["XDG_CONFIG_HOME"] == str(user_profile["config_dir"].parent)

    def test_auth_stays_where_cursor_looks_for_it(self, provider, user_profile):
        """The redirect must not present as a logged-out CLI.

        ``auth.json`` follows neither redirected variable — it is
        ``$XDG_CONFIG_HOME/cursor/auth.json`` on Linux — so after the redirect
        it is still exactly where the CLI resolves it, with its mode intact.
        """
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        auth = Path(os.environ["XDG_CONFIG_HOME"]) / "cursor" / "auth.json"
        assert auth == user_profile["auth"]
        assert auth.is_file()
        # Windows keeps only the read-only bit, so the fixture's 0600 reads back 0666.
        assert stat.S_IMODE(auth.stat().st_mode) == (0o666 if os.name == "nt" else 0o600)
        assert json.loads(auth.read_text(encoding="utf-8"))["accessToken"]

    def test_the_guard_is_not_armed_when_the_redirect_is_on(self, provider, user_profile):
        """Nothing to snapshot: the run cannot reach the user's file."""
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert cursor_agent_module.armed_snapshot() is None

    def test_a_trust_marker_lands_in_the_run_profile(self, provider, user_profile, tmp_path):
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        child_run(env, "composer-2.5", workspace=str(tmp_path))

        user_markers = list((user_profile["data_dir"] / "projects").glob("*/.workspace-trusted"))
        assert user_markers == []
        run_markers = list(
            (Path(env["CURSOR_DATA_DIR"]) / "projects").glob("*/.workspace-trusted")
        )
        assert len(run_markers) == 1


@pytest.mark.usefixtures("redirect_here")
class TestUserConfigIsNeverWritten:
    """Asserted against the filesystem, not against our own source."""

    def test_a_read_only_user_profile_does_not_fail_the_run(self, provider, user_profile, tmp_path):
        """The strongest available statement: make writing *impossible*.

        The config file is 0400 and its directory 0500, so an in-place rewrite
        and an atomic ``os.replace`` both raise ``PermissionError``. A run that
        completes under that is a run that never wrote there.
        """
        config = user_profile["config"]
        config_dir = user_profile["config_dir"]
        before = config.read_bytes()
        config.chmod(0o400)
        config_dir.chmod(0o500)
        try:
            env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
            child_run(env, "composer-2.5", workspace=str(tmp_path))
            env_two = provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
            child_run(env_two, "gemini-3.6-flash-high", workspace=str(tmp_path))
            cursor_agent_module.restore_side_effects()
            cursor_agent_module.cleanup_isolated_profiles()
        finally:
            config_dir.chmod(0o700)
            config.chmod(0o600)

        assert config.read_bytes() == before

    def test_no_write_open_ever_names_the_user_profile(self, provider, user_profile, tmp_path, monkeypatch):
        """Belt and braces: record every write-mode open and every replace."""
        real_open = open
        real_replace = os.replace
        written = []

        def recording_open(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                written.append(os.path.abspath(os.fspath(file)))
            return real_open(file, mode, *args, **kwargs)

        def recording_replace(src, dst, *args, **kwargs):
            written.append(os.path.abspath(os.fspath(dst)))
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr("builtins.open", recording_open)
        monkeypatch.setattr(os, "replace", recording_replace)

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        child_run(env, "composer-2.5", workspace=str(tmp_path))
        cursor_agent_module.restore_side_effects()
        cursor_agent_module.cleanup_isolated_profiles()

        monkeypatch.undo()
        user_root = str(user_profile["config_dir"].parent.parent)  # the fake $HOME
        offenders = [p for p in written if p.startswith(user_root + os.sep)]
        # The simulated child writes into the redirected dir; nothing may land
        # under the user's home at all.
        assert offenders == [], offenders


class TestUserSuppliedDirsAreRespected:
    """A pinned profile is the user's decision, and it is never deleted."""

    @pytest.mark.parametrize("variable", ["CURSOR_CONFIG_DIR", "CURSOR_DATA_DIR"])
    def test_a_pinned_dir_suppresses_the_redirect(self, provider, tmp_path, monkeypatch, variable):
        pinned = tmp_path / "pinned"
        (pinned / "projects").mkdir(parents=True)
        monkeypatch.setenv(variable, str(pinned))

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        assert "CURSOR_CONFIG_DIR" not in env
        assert "CURSOR_DATA_DIR" not in env
        assert cursor_agent_module.isolated_profile_dirs() == ()
        # ...and the fallback guard takes over, because the pin is the user's
        # own profile and the residue is back in scope.
        assert cursor_agent_module.armed_snapshot() is not None

    def test_a_pinned_dir_is_never_removed(self, provider, tmp_path, monkeypatch):
        pinned = tmp_path / "pinned"
        pinned.mkdir()
        (pinned / "cli-config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("CURSOR_CONFIG_DIR", str(pinned))
        monkeypatch.setenv("CURSOR_DATA_DIR", str(pinned))

        provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        cursor_agent_module.cleanup_isolated_profiles()

        assert pinned.is_dir()
        assert (pinned / "cli-config.json").exists()

    def test_a_dir_this_process_did_not_create_is_never_removed(self, tmp_path):
        stranger = tmp_path / "someone-elses-profile"
        stranger.mkdir()
        removed = cursor_agent_module.remove_profile_dir(str(stranger))
        assert removed is False
        assert stranger.is_dir()


@pytest.mark.usefixtures("redirect_here")
class TestCleanup:
    def test_the_dir_is_gone_after_a_normal_exit(self, provider, user_profile):
        with cursor_agent_module.isolated_profile_session():
            env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
            created = Path(env["CURSOR_CONFIG_DIR"])
            assert created.is_dir()
        assert not created.exists()

    def test_the_dir_is_gone_after_an_exception(self, provider, user_profile):
        created = None
        with pytest.raises(RuntimeError):
            with cursor_agent_module.isolated_profile_session():
                env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
                created = Path(env["CURSOR_CONFIG_DIR"])
                raise RuntimeError("a row blew up mid-fan-out")
        assert created is not None and not created.exists()

    def test_a_populated_dir_is_still_removed(self, provider, user_profile, tmp_path):
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        child_run(env, "composer-2.5", workspace=str(tmp_path))
        created = Path(env["CURSOR_CONFIG_DIR"])
        assert (created / "cli-config.json").exists()

        cursor_agent_module.cleanup_isolated_profiles()

        assert not created.exists()

    def test_cleanup_is_idempotent_and_never_raises(self, provider, user_profile):
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        created = Path(env["CURSOR_CONFIG_DIR"])
        cursor_agent_module.cleanup_isolated_profiles()
        # Gone underneath us, twice over.
        cursor_agent_module.cleanup_isolated_profiles()
        assert not created.exists()

    def test_the_atexit_hook_is_registered_at_creation(self, provider, user_profile, monkeypatch):
        registered = []
        monkeypatch.setattr(
            cursor_agent_module.atexit, "register", lambda fn, *a, **k: registered.append(fn) or fn
        )
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert cursor_agent_module.cleanup_isolated_profiles in registered

    def test_the_hook_is_registered_once_per_process(self, provider, user_profile, monkeypatch):
        registered = []
        monkeypatch.setattr(
            cursor_agent_module.atexit, "register", lambda fn, *a, **k: registered.append(fn) or fn
        )
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        assert registered.count(cursor_agent_module.cleanup_isolated_profiles) == 1


@pytest.mark.usefixtures("redirect_here")
class TestCleanupRacesTheLeakedWorker:
    """cursor-agent leaks a detached ``worker-server``, and cleanup must survive it.

    Measured on the live probe: the daemon **outlives the CLI**, reparents to
    init, and **inherits the redirect env** — its own ``CURSOR_CONFIG_DIR``
    pointed into the run's scratch directory. So a removal can always be racing
    a live writer. The contract asserted below, in priority order:

    1. **Never raise and never block.** Cleanup runs from ``atexit`` and from a
       ``finally``; raising would fail a run that already succeeded and waiting
       would delay every run for a directory nobody reads.
    2. **Still remove it.** A profile holds a 703 KB ``statsig-cache.json`` plus
       a chat store per run, so leaking it is not acceptable hygiene — hence the
       bounded re-walk rather than a single ``rmtree`` pass.
    3. **Never signal the holder.** Killing the worker belongs to
       ``plans/timeout-orphans-and-reaggregate-determinism.md``. This module
       removes bytes and nothing else.

    Note what is deliberately *not* asserted: on Linux an open fd does not block
    ``rmtree``, so "the dir survived" is a POSIX quirk, not the contract. Every
    test below asserts the contract — gone, or still tracked so the next pass
    retries it — and never the quirk.
    """

    @staticmethod
    def _still_accounted_for(created):
        """Removed, or handed to the next pass. Never silently dropped."""
        gone = not created.exists()
        tracked = str(created) in cursor_agent_module.isolated_profile_dirs()
        assert gone or tracked, (
            "a profile dir that survived cleanup must stay on the tracking list "
            "so the atexit pass retries it"
        )
        return gone

    def test_an_open_handle_in_the_tree_does_not_make_cleanup_raise(
        self, provider, user_profile, tmp_path
    ):
        """A live holder, spelled the way the worker really holds one."""
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        created = Path(env["CURSOR_CONFIG_DIR"])
        child_run(env, "composer-2.5", workspace=str(tmp_path))
        session = created / "chats" / "0123abcd" / "session-1"
        session.mkdir(parents=True)

        handle = open(session / "store.db", "wb")
        try:
            handle.write(b"\x00" * 4096)
            handle.flush()
            removed = cursor_agent_module.cleanup_isolated_profiles()  # must not raise
        finally:
            handle.close()

        assert isinstance(removed, list)
        self._still_accounted_for(created)

    def test_a_file_appearing_mid_removal_is_still_cleaned_up(
        self, provider, user_profile, monkeypatch
    ):
        """``rmtree`` lists a directory, then unlinks what it listed.

        A file the worker creates in the gap survives the walk and keeps its
        parent non-empty. The second pass sees it — which is the whole reason
        one pass is not enough.
        """
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        created = Path(env["CURSOR_CONFIG_DIR"])
        real_rmtree = cursor_agent_module.shutil.rmtree
        passes = []

        def the_worker_writes_after_the_walk(path, *args, **kwargs):
            real_rmtree(path, *args, **kwargs)
            passes.append(path)
            if len(passes) == 1:  # the daemon re-fetches its cache, once
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "statsig-cache.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            cursor_agent_module.shutil, "rmtree", the_worker_writes_after_the_walk
        )

        removed = cursor_agent_module.cleanup_isolated_profiles()

        assert len(passes) >= 2, "one pass cannot beat a writer; the re-walk is the point"
        assert removed == [str(created)]
        assert not created.exists()
        assert cursor_agent_module.isolated_profile_dirs() == ()

    def test_a_holder_that_never_lets_go_is_bounded_not_awaited(
        self, provider, user_profile, monkeypatch
    ):
        """The pathological case: a worker writing on every single pass.

        Cleanup must give up rather than spin or sleep, report honestly that the
        tree is not gone, and leave it tracked for the ``atexit`` pass.
        """
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        created = Path(env["CURSOR_CONFIG_DIR"])
        real_rmtree = cursor_agent_module.shutil.rmtree
        passes = []

        def a_worker_that_keeps_writing(path, *args, **kwargs):
            real_rmtree(path, *args, **kwargs)
            passes.append(path)
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "statsig-cache.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(cursor_agent_module.shutil, "rmtree", a_worker_that_keeps_writing)

        started = time.monotonic()
        removed = cursor_agent_module.cleanup_isolated_profiles()  # must not raise
        elapsed = time.monotonic() - started

        assert removed == [], "a tree that is still there must not be reported as removed"
        assert len(passes) == cursor_agent_module._REMOVE_PASSES, "bounded, never a spin"
        assert elapsed < 5.0, "cleanup must never wait on a daemon"
        assert str(created) in cursor_agent_module.isolated_profile_dirs()
        self._still_accounted_for(created)

    def test_cleanup_never_raises_even_when_rmtree_itself_does(
        self, provider, user_profile, monkeypatch
    ):
        """``ignore_errors=True`` is not trusted; the guarantee is structural."""
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        def boom(*_args, **_kwargs):
            raise OSError("the worker's store.db vanished mid-walk")

        monkeypatch.setattr(cursor_agent_module.shutil, "rmtree", boom)

        assert cursor_agent_module.cleanup_isolated_profiles() == []

    def test_cleanup_never_signals_the_holder(self, provider, user_profile, tmp_path):
        """Reaping the worker is another plan's job — this one removes bytes."""
        killed = []
        original_kill = cursor_agent_module.os.kill
        cursor_agent_module.os.kill = lambda *args, **kwargs: killed.append(args)
        try:
            env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
            child_run(env, "composer-2.5", workspace=str(tmp_path))
            cursor_agent_module.cleanup_isolated_profiles()
        finally:
            cursor_agent_module.os.kill = original_kill
        assert killed == []

    def test_remove_profile_dir_is_best_effort_too(self, provider, user_profile, monkeypatch):
        """The single-dir path shares the contract, not just the batch path."""
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        created = env["CURSOR_CONFIG_DIR"]

        def boom(*_args, **_kwargs):
            raise OSError("a live worker holds it")

        monkeypatch.setattr(cursor_agent_module.shutil, "rmtree", boom)

        assert cursor_agent_module.remove_profile_dir(created) is True  # ours; must not raise
        assert cursor_agent_module.remove_profile_dir(created) is False  # no longer ours


@pytest.mark.usefixtures("redirect_here")
class TestScopeIsParameterized:
    """One decision point: :data:`PROFILE_DIR_SCOPE`. Both values are covered.

    The live probe settled which one ships: cursor's parameters ride on argv and
    the config file is *output*, so a shared file cannot leak effort, thinking,
    fast or max-mode between concurrent rows. Per-spawn would only buy the
    ~3.35 s cold-start penalty per child and hand the leaked ``worker-server`` a
    directory torn down at row end. The switch stays parameterized; the default
    is now evidence-backed.
    """

    def test_the_shipped_scope_is_per_run(self):
        assert cursor_agent_module.PROFILE_DIR_SCOPE == cursor_agent_module.SCOPE_PER_RUN

    def test_per_run_shares_one_dir_across_rows(self, provider, user_profile, monkeypatch):
        monkeypatch.setattr(
            cursor_agent_module, "PROFILE_DIR_SCOPE", cursor_agent_module.SCOPE_PER_RUN
        )
        first = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        second = provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        assert first["CURSOR_CONFIG_DIR"] == second["CURSOR_CONFIG_DIR"]
        assert len(cursor_agent_module.isolated_profile_dirs()) == 1

    def test_per_spawn_gives_each_child_its_own_dir(self, provider, user_profile, monkeypatch):
        monkeypatch.setattr(
            cursor_agent_module, "PROFILE_DIR_SCOPE", cursor_agent_module.SCOPE_PER_SPAWN
        )
        first = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        second = provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        assert first["CURSOR_CONFIG_DIR"] != second["CURSOR_CONFIG_DIR"]
        assert len(cursor_agent_module.isolated_profile_dirs()) == 2

    def test_per_spawn_dirs_are_all_cleaned_up(self, provider, user_profile, monkeypatch):
        monkeypatch.setattr(
            cursor_agent_module, "PROFILE_DIR_SCOPE", cursor_agent_module.SCOPE_PER_SPAWN
        )
        created = [
            Path(provider.resolve_env(m, PROFILE_GENERATE)["CURSOR_CONFIG_DIR"])
            for m in ("composer-2.5", "gemini-3.6-flash-high", "cursor-grok-4.5-high")
        ]
        cursor_agent_module.cleanup_isolated_profiles()
        assert [p for p in created if p.exists()] == []

    def test_concurrent_rows_create_exactly_one_dir_under_per_run(self, provider, user_profile, monkeypatch):
        """§2.3's concurrency gate: N slots, one directory, no race."""
        monkeypatch.setattr(
            cursor_agent_module, "PROFILE_DIR_SCOPE", cursor_agent_module.SCOPE_PER_RUN
        )
        seen = []
        barrier = threading.Barrier(8)

        def row():
            barrier.wait()
            seen.append(provider.resolve_env("composer-2.5", PROFILE_GENERATE)["CURSOR_CONFIG_DIR"])

        threads = [threading.Thread(target=row) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(set(seen)) == 1
        assert len(cursor_agent_module.isolated_profile_dirs()) == 1


@pytest.mark.usefixtures("redirect_here")
class TestSeedingHook:
    """Seeding is one function — and the live probe says it stays empty.

    Measured: with both variables pointed at an **empty** directory, a real
    cursor-agent run exited 0, authenticated, and wrote 0 bytes to stderr. No
    seed is required. The hook survives as the single place a future need would
    land, and the rule below binds it whatever lands there.
    """

    def test_the_seed_hook_runs_once_per_created_dir(self, provider, user_profile, monkeypatch):
        seeded = []
        monkeypatch.setattr(cursor_agent_module, "_seed_profile_dir", seeded.append)
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        assert len(seeded) == 1
        assert seeded == list(cursor_agent_module.isolated_profile_dirs())

    def test_the_seed_never_copies_the_user_config(self, provider, user_profile):
        """Copying it in would reintroduce the leak in reverse."""
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        run_config = Path(env["CURSOR_CONFIG_DIR"]) / "cli-config.json"
        if run_config.exists():
            seeded = json.loads(run_config.read_text(encoding="utf-8"))
            for key in ("model", "selectedModel", "modelSelectionHistory", "modelParameters"):
                assert key not in seeded, f"the user's {key} was copied into the run profile"


# ---------------------------------------------------------------------------
# The off switch, and the fallback it re-enables
# ---------------------------------------------------------------------------


@pytest.mark.config_override
class TestIsolateProfileSwitch:
    def test_default_is_auto(self, tmp_path, monkeypatch):
        """``providers.yaml`` and :data:`ISOLATE_DEFAULT` must ship the same word.

        Not ``on``: only Linux's auth rule was measured, and on an unmeasured
        platform ``on`` would present every cursor slot as logged out, while
        ``auto`` degrades to the snapshot/restore guard that shipped before the
        redirect existed. Asserting both halves is what stops the file and the
        constant from drifting apart.
        """
        assert cursor_agent_module.isolate_profile_setting() == cursor_agent_module.ISOLATE_AUTO
        assert cursor_agent_module.ISOLATE_DEFAULT == cursor_agent_module.ISOLATE_AUTO

    def test_off_reproduces_todays_behaviour_exactly(self, provider, user_profile, tmp_path, monkeypatch):
        """No redirect, guard armed, user's file the one that moves and comes back."""
        _write_override(tmp_path, monkeypatch, "off")
        config = user_profile["config"]
        before = config.read_bytes()

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        assert "CURSOR_CONFIG_DIR" not in env
        assert "CURSOR_DATA_DIR" not in env
        assert cursor_agent_module.isolated_profile_dirs() == ()
        assert cursor_agent_module.armed_snapshot() is not None

        child_run(env, "composer-2.5")
        assert _selected(config) == "composer-2.5", "the run really did hit the user's file"

        cursor_agent_module.restore_side_effects()
        assert config.read_bytes() == before

    def test_off_still_carries_the_cross_process_defect(self, provider, user_profile, tmp_path, monkeypatch):
        """Why ``off`` is not the default, asserted rather than asserted-about.

        The fallback guard is a mitigation, not a fix: two overlapping
        processes still corrupt the user's default model. This test exists so
        that fact is visible in CI rather than in a comment.
        """
        _write_override(tmp_path, monkeypatch, "off")
        config = user_profile["config"]

        p1_env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        p1_guard = cursor_agent_module.armed_snapshot()
        child_run(p1_env, "composer-2.5")

        cursor_agent_module.reset_side_effect_guard()   # a second process
        p2_env = provider.resolve_env("gemini-3.6-flash-high", PROFILE_GENERATE)
        p2_guard = cursor_agent_module.armed_snapshot()
        child_run(p2_env, "gemini-3.6-flash-high")

        p1_guard.restore()
        p2_guard.restore()

        assert _selected(config) == "composer-2.5", (
            "if this ever says gpt-5.6-sol the fallback got better and this test "
            "should be tightened, not deleted"
        )

    def test_on_redirects(self, provider, user_profile, tmp_path, monkeypatch):
        _write_override(tmp_path, monkeypatch, "on")
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert Path(env["CURSOR_CONFIG_DIR"]).is_dir()

    def test_auto_redirects_where_the_auth_rule_is_verified(self, provider, user_profile, tmp_path, monkeypatch):
        _write_override(tmp_path, monkeypatch, "auto")
        monkeypatch.setattr(cursor_agent_module.sys, "platform", "linux")
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert Path(env["CURSOR_CONFIG_DIR"]).is_dir()

    def test_auto_declines_where_it_is_not(self, provider, user_profile, tmp_path, monkeypatch):
        _write_override(tmp_path, monkeypatch, "auto")
        monkeypatch.setattr(cursor_agent_module.sys, "platform", "darwin")
        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)
        assert "CURSOR_CONFIG_DIR" not in env
        assert cursor_agent_module.armed_snapshot() is not None

    def test_an_unknown_value_falls_back_to_the_default(self, tmp_path, monkeypatch):
        _write_override(tmp_path, monkeypatch, "sometimes")
        assert (
            cursor_agent_module.isolate_profile_setting()
            == cursor_agent_module.ISOLATE_DEFAULT
            == cursor_agent_module.ISOLATE_AUTO
        )

    def test_a_yaml_bare_on_off_is_read_as_the_enum(self, tmp_path, monkeypatch):
        """YAML 1.1 turns a bare ``on``/``off`` into a bool; both must still work."""
        import utils.provider_registry as registry

        override = tmp_path / "providers-override.yaml"
        override.write_text(
            "providers:\n  cursor-agent:\n    isolate_profile: off\n", encoding="utf-8"
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
        registry._config = None
        registry._config_key = None
        assert cursor_agent_module.isolate_profile_setting() == cursor_agent_module.ISOLATE_OFF


class TestTheShippedAutoDefaultOnThisPlatform:
    """``auto`` must not be a quiet way of turning the feature off.

    Deliberately **not** ``@pytest.mark.config_override``: the autouse isolation
    fixture unsets ``MULTI_LLM_PROVIDERS_CONFIG`` and neutralizes project-local
    discovery, so what these tests read is the shipped ``providers.yaml`` and
    nothing else. That is the point — the pair below pins that the value the
    file ships really redirects where the auth rule was measured, and really
    falls back to the guard where it was not.
    """

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux is the platform cursor's auth rule was measured on",
    )
    def test_auto_really_redirects_on_linux(self, provider, user_profile):
        """The measured platform: empty redirected dir, run exits 0, auth intact."""
        assert cursor_agent_module.isolate_profile_setting() == cursor_agent_module.ISOLATE_AUTO
        assert "linux" in cursor_agent_module._AUTH_UNAFFECTED_PLATFORMS

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        assert env["CURSOR_CONFIG_DIR"] == env["CURSOR_DATA_DIR"]
        assert Path(env["CURSOR_CONFIG_DIR"]).is_dir()
        assert cursor_agent_module.isolated_profile_dirs() == (env["CURSOR_CONFIG_DIR"],)
        # Redirected, so there is nothing for the fallback guard to snapshot...
        assert cursor_agent_module.armed_snapshot() is None
        # ...and auth still resolves through the rule neither variable moves.
        assert "XDG_CONFIG_HOME" not in env and "HOME" not in env
        assert user_profile["auth"].is_file()

    @pytest.mark.parametrize("platform", ["darwin", "win32"])
    def test_auto_falls_back_to_the_guard_off_linux(
        self, provider, user_profile, monkeypatch, platform
    ):
        """The unmeasured platforms: the guard, i.e. exactly what shipped before.

        This is the whole reason the default is ``auto`` rather than ``on`` — if
        cursor resolves ``auth.json`` differently there, ``on`` would present a
        logged-out CLI, while this path is strictly no worse than the previous
        release.
        """
        monkeypatch.setattr(cursor_agent_module.sys, "platform", platform)

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        assert "CURSOR_CONFIG_DIR" not in env
        assert "CURSOR_DATA_DIR" not in env
        assert cursor_agent_module.isolated_profile_dirs() == ()
        assert cursor_agent_module.armed_snapshot() is not None
        assert env["GIT_TERMINAL_PROMPT"] == "0", "the shared env delta survives"


class TestRefusalFallsBackToTheGuard:
    """A redirect that cannot be built must not silently drop the mitigation."""

    def test_a_failed_mkdtemp_arms_the_guard_instead(self, provider, user_profile, monkeypatch):
        def boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(cursor_agent_module.tempfile, "mkdtemp", boom)

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        assert "CURSOR_CONFIG_DIR" not in env
        assert cursor_agent_module.armed_snapshot() is not None
        assert env["GIT_TERMINAL_PROMPT"] == "0", "the shared env delta survives"


class TestTheFallbackGuardIsArmedWithTheDispatchCwd:
    """A Lane B dispatch runs in an explicit ``cwd``; the guard must record it.

    ``_marker_is_ours`` only removes a ``.workspace-trusted`` whose
    ``workspacePath`` names a workspace *this process ran in*. Arming the
    fallback without the dispatch's ``cwd`` records the **orchestrator's**
    directory instead, so the durable marker the run really wrote survives as if
    the user had trusted that repo on purpose — which also trusts it for their
    own later interactive sessions.
    """

    @staticmethod
    def _force_the_fallback(monkeypatch):
        """Make the redirect unbuildable, whatever this platform's `auto` does."""

        def boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(cursor_agent_module.tempfile, "mkdtemp", boom)

    def test_the_guard_records_the_dispatch_cwd_not_the_process_one(
        self, provider, user_profile, tmp_path, monkeypatch
    ):
        self._force_the_fallback(monkeypatch)
        workspace = tmp_path / "checkout"
        workspace.mkdir()

        provider.resolve_env("composer-2.5", PROFILE_GENERATE, cwd=str(workspace))

        armed = cursor_agent_module.armed_snapshot()
        assert armed is not None
        assert os.path.realpath(str(workspace)) in armed.workspaces
        assert os.path.realpath(os.getcwd()) not in armed.workspaces

    def test_a_marker_written_for_that_cwd_is_removed_at_restore(
        self, provider, user_profile, tmp_path, monkeypatch
    ):
        self._force_the_fallback(monkeypatch)
        workspace = tmp_path / "checkout"
        workspace.mkdir()

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE, cwd=str(workspace))
        child_run(env, "composer-2.5", workspace=str(workspace))

        marker = (
            user_profile["data_dir"] / "projects" / "slug-of-the-run" / ".workspace-trusted"
        )
        assert marker.is_file(), "the child really wrote into the user's profile"

        assert str(marker) in cursor_agent_module.restore_side_effects()
        assert not marker.exists()

    def test_omitting_the_cwd_still_means_this_process_directory(
        self, provider, user_profile, tmp_path, monkeypatch
    ):
        """Lane A's usual case: no explicit cwd, so the child inherits ours."""
        self._force_the_fallback(monkeypatch)

        env = provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        armed = cursor_agent_module.armed_snapshot()
        assert armed is not None
        assert os.path.realpath(os.getcwd()) in armed.workspaces

        child_run(env, "composer-2.5", workspace=os.getcwd())
        marker = (
            user_profile["data_dir"] / "projects" / "slug-of-the-run" / ".workspace-trusted"
        )
        assert str(marker) in cursor_agent_module.restore_side_effects()
        assert not marker.exists()
