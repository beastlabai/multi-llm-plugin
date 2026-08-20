"""vibe's per-slot config root and env surface, asserted rather than assumed (§8.2).

**Why this is a module of its own.** `tests/container/wrapper.py` rebuilds every
child's environment from scratch and points `HOME`/`XDG_*`/`TMPDIR` at a fresh
per-invocation root, so the container lane isolates for free and would stay green
no matter what this adapter does. Production fan-out gets none of that:
`resolve_env` composes on top of the user's real environment, so whatever the
adapter does not inject, strip or redirect is inherited from the user's shell —
and whatever the CLI writes lands in the user's real `$HOME`. vibe writes on
**every** run (session logs, `vibehistory`, `trusted_folders.toml`), so this is
not hypothetical for it.

**The measurement that shaped this module, and it removed work rather than
adding it.** §4.2 assumed vibe's lack of a `--model` flag forced a templated
per-run config file, which is the entire reason §1 grew a `prepare_run(model,
profile, *, run_dir)` hook. The §6 probe measured otherwise:
`VIBE_ACTIVE_MODEL` selects the model from the environment, and two concurrent
slots with **different** models both succeeded while sharing one `VIBE_HOME` and
one read-only `config.toml`. So the catalog is a static committed seed, the only
per-run-varying value is an env var, and there is no file for a race to be about.
`test_two_concurrent_calls_get_two_different_writable_roots` still pins a per-slot
root — vibe's *own* writes need one — but
`test_two_concurrent_slots_select_their_own_models` is the assertion that
replaces the hook, and it is the one that would have justified building it.

**The ordering hazard this module exists to catch, stated once.** Both spawn
sites compose the child env as `{**os.environ, **provider_env}` and *then* pop
every name `get_remove_env()` returns — the strip runs **after** the merge. A
name in both lists is therefore not defence in depth; it is the adapter deleting
its own value. For vibe that would be silent and total: strip
`VIBE_ACTIVE_MODEL` and no model is selected at all, with nothing in the argv to
show why, because vibe has no `--model` flag to disagree with.
`test_nothing_the_adapter_injects_is_also_stripped` is that assertion.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.providers import vibe as vibe_module  # noqa: E402
from utils.providers.vibe import VibeProvider  # noqa: E402

MODEL = "vibe-default"


@pytest.fixture(autouse=True)
def _clean_tracking():
    """Each test owns its own set of created roots."""
    vibe_module.cleanup_home_dirs()
    yield
    vibe_module.cleanup_home_dirs()


@pytest.fixture
def sentinel_home(tmp_path, monkeypatch):
    """A `$HOME` that must still be empty when the test ends."""
    home = tmp_path / "sentinel-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("VIBE_HOME", raising=False)
    return home


class TestVibeHomeIsolation:
    def test_resolve_env_never_yields_a_path_under_the_real_home(self, sentinel_home):
        """(1) The redirect points somewhere else, and `$HOME` stays untouched."""
        env = VibeProvider().resolve_env(MODEL)

        root = env["VIBE_HOME"]
        assert not root.startswith(str(sentinel_home)), root
        assert Path(root).is_dir()
        assert list(sentinel_home.iterdir()) == [], "the adapter wrote into $HOME"

    def test_two_concurrent_calls_get_two_different_writable_roots(
        self, sentinel_home
    ):
        """(2) Two fan-out slots never share one writable config root.

        Run genuinely concurrently rather than in sequence, because the failure
        mode being excluded is a race on a shared module-level directory, not a
        naming collision — and `_PROVIDERS` holds **one** instance shared across
        the whole fan-out.
        """
        provider = VibeProvider()

        with ThreadPoolExecutor(max_workers=2) as pool:
            roots = [
                f.result()["VIBE_HOME"]
                for f in [pool.submit(provider.resolve_env, MODEL) for _ in range(2)]
            ]

        assert len(set(roots)) == 2, roots
        for root in roots:
            assert Path(root).is_dir()
            assert os.access(root, os.W_OK)

    def test_two_concurrent_slots_select_their_own_models(self, sentinel_home):
        """The assertion that replaced `prepare_run` (§4.2's wrinkle, void).

        vibe has no `--model` flag, so §4.2 concluded each slot must be handed a
        templated `config.toml` of its own and §1 grew a file-writing hook for
        it. Measured, the model is an **env var** — so two slots on one shared
        instance simply get two different values, and the "two slots racing one
        `active_model`" failure the hook existed to prevent cannot occur.

        This is the test that would have justified building the hook. It passes
        without one, which is why none ships.
        """
        provider = VibeProvider()

        with ThreadPoolExecutor(max_workers=2) as pool:
            envs = [
                f.result()
                for f in [
                    pool.submit(provider.resolve_env, "alpha-alias"),
                    pool.submit(provider.resolve_env, "beta-alias"),
                ]
            ]

        assert {e["VIBE_ACTIVE_MODEL"] for e in envs} == {"alpha-alias", "beta-alias"}

    def test_the_root_is_private_to_this_user(self, sentinel_home):
        """0700, asserted rather than inherited from `mkdtemp`'s default."""
        root = VibeProvider().resolve_env(MODEL)["VIBE_HOME"]

        assert oct(Path(root).stat().st_mode & 0o777) == "0o700"

    def test_a_user_pinned_root_is_respected_and_never_removed(
        self, tmp_path, monkeypatch
    ):
        """Their directory, their rules — and both container lanes rely on it.

        This is not only politeness for a host user. vibe reads
        `$VIBE_HOME/config.toml`, and the committed `[[providers]]`/`[[models]]`
        catalog is seeded into a root the *lane* chooses — the provider lane
        through the wrapper's `config_roots`, the credentialed lane by rendering
        the same `home_seed/vibe/` tree. An adapter that always minted a fresh
        empty directory would hand vibe a config root with no providers in it.
        """
        pinned = tmp_path / "my-vibe-home"
        pinned.mkdir()
        monkeypatch.setenv("VIBE_HOME", str(pinned))

        env = VibeProvider().resolve_env(MODEL)

        assert env.get("VIBE_HOME") in (None, str(pinned))
        assert vibe_module.isolated_home_dirs() == ()
        vibe_module.cleanup_home_dirs()
        assert pinned.is_dir(), "a user-pinned root was removed"

    def test_the_created_roots_are_gone_after_cleanup(self, sentinel_home):
        """(4) The per-run dir does not survive the process."""
        provider = VibeProvider()
        roots = [provider.resolve_env(MODEL)["VIBE_HOME"] for _ in range(3)]

        vibe_module.cleanup_home_dirs()

        for root in roots:
            assert not Path(root).exists(), root

    def test_the_prefix_makes_a_leak_sweepable(self, sentinel_home):
        """A SIGKILL skips `atexit`, so the fixed prefix is the recovery path."""
        root = VibeProvider().resolve_env(MODEL)["VIBE_HOME"]

        assert Path(root).name.startswith("multi-llm-vibe-")

    def test_cleanup_never_raises_on_an_already_removed_root(self, sentinel_home):
        """Containment must never be the thing that fails a run."""
        import shutil

        root = VibeProvider().resolve_env(MODEL)["VIBE_HOME"]
        shutil.rmtree(root)

        vibe_module.cleanup_home_dirs()  # must not raise


class TestVibeEnvSurface:
    """The B1 selector set and the strip list (§2.2 rule 6, §8.2 test 3)."""

    def test_the_model_surface_is_env_because_there_is_no_flag(self, sentinel_home):
        """`VIBE_ACTIVE_MODEL` is the whole model surface — nothing reaches argv."""
        provider = VibeProvider()
        env = provider.resolve_env(MODEL)
        argv = provider.build_command("PROMPT", MODEL)

        assert env["VIBE_ACTIVE_MODEL"] == MODEL
        assert MODEL not in argv, "vibe has no --model flag; the alias must not be argv"

    def test_the_pinned_inner_bound_matches_the_declared_envelope(self, sentinel_home):
        """The pin and the declaration cannot drift.

        The vendor default is 720 s over 3 attempts — a ~36-minute worst case
        inside one slot — so the pin is what makes the declared `max_wall_s`
        true. Deriving the assertion from the same constants is the point: a
        change to one without the other fails here.
        """
        provider = VibeProvider()
        env = provider.resolve_env(MODEL)

        assert env["VIBE_API_TIMEOUT"] == str(vibe_module.API_TIMEOUT_S)
        envelope = provider.internal_retries
        assert envelope is not None
        assert envelope.max_wall_s == 3 * vibe_module.API_TIMEOUT_S + 2
        assert envelope.max_wall_s < provider.default_timeout

    def test_the_safe_agent_is_pinned_in_env_as_well_as_argv(self, sentinel_home):
        """§6.5's rung, belt-and-braces against an argv regression.

        vibe's `default_agent` is `accept-edits`, whose `AgentSafety` is
        DESTRUCTIVE. Pinning `VIBE_DEFAULT_AGENT` means dropping `--agent plan`
        from the argv could not silently re-arm file edits.
        """
        env = VibeProvider().resolve_env(MODEL)

        assert env["VIBE_DEFAULT_AGENT"] == "plan"
        assert "--agent" in VibeProvider().build_command("PROMPT", MODEL)

    def test_telemetry_and_updates_are_off(self, sentinel_home):
        """Both default TRUE upstream, and the dep tree carries sentry + otel."""
        env = VibeProvider().resolve_env(MODEL)

        assert env["VIBE_ENABLE_TELEMETRY"] == "false"
        assert env["VIBE_ENABLE_UPDATE_CHECKS"] == "false"
        assert env["VIBE_ENABLE_NOTIFICATIONS"] == "false"

    def test_effort_is_emitted_unconditionally(self, sentinel_home):
        """An ambient `VIBE_THINKING` must not reach an unsuffixed spec.

        `get_remove_env` takes no model, so it cannot strip this name only on the
        unsuffixed path — and stripping it always would delete the level on the
        suffixed path, because the strip runs after the merge. Emitting it every
        time is the only shape that makes both paths deterministic.
        """
        provider = VibeProvider()

        assert provider.resolve_env(MODEL)["VIBE_THINKING"] == "off"
        assert provider.resolve_env(f"{MODEL}:high")["VIBE_THINKING"] == "high"

    def test_the_effort_suffix_never_reaches_the_alias(self, sentinel_home):
        """A `:high` glued onto the alias selects no configured model at all."""
        env = VibeProvider().resolve_env(f"{MODEL}:max")

        assert env["VIBE_ACTIVE_MODEL"] == MODEL

    def test_an_unknown_suffix_passes_through_as_part_of_the_alias(self, sentinel_home):
        """`split_reasoning_effort` only splits a whitelisted level.

        An alias that legitimately contains a colon must survive intact, which is
        the whole reason the split is whitelist-driven rather than positional.
        """
        env = VibeProvider().resolve_env("weird:alias")

        assert env["VIBE_ACTIVE_MODEL"] == "weird:alias"
        assert env["VIBE_THINKING"] == "off"

    @pytest.mark.parametrize(
        "name,why",
        [
            ("MISTRAL_API_KEY", "a stale vendor key beside a `mistral`-named provider"),
            ("VIBE_ENABLE_OTEL", "a separate switch the adapter does not set"),
            ("LOG_LEVEL", "vibe reads the BARE name; it changes the output shape"),
        ],
    )
    def test_every_measured_name_is_stripped(self, name, why):
        assert name in VibeProvider().get_remove_env(), why

    def test_nothing_the_adapter_injects_is_also_stripped(self):
        """The ordering assertion, and for vibe it is the sharpest one.

        The strip runs **after** the merge at both spawn sites, so a name in both
        lists is the adapter deleting its own value. `VIBE_ACTIVE_MODEL` is the
        case that matters: strip it and no model is selected at all, with nothing
        in the argv to show why — vibe has no `--model` flag to disagree with.
        """
        provider = VibeProvider()
        injected = set(provider.get_env(MODEL))
        stripped = set(provider.get_remove_env())

        assert injected & stripped == set()
