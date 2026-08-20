"""kimi's per-slot config root, asserted rather than assumed (§8.2).

**Why this is a module of its own, and why it is not covered by the container
lane.** `tests/container/wrapper.py` rebuilds every child's environment from
scratch and points `HOME`/`XDG_*`/`TMPDIR` at a fresh per-invocation root, so
the container lane isolates for free and would stay green no matter what this
adapter does. Production fan-out gets none of that: `resolve_env` composes on
top of the user's real environment, so whatever the adapter does not inject,
strip or redirect is inherited from the user's shell — and whatever the CLI
writes lands in the user's real `$HOME`.

**What kimi actually writes.** Measured at 0.36.1 in the §6 probe: `device_id`,
`config.toml`, `logs/`, `workspaces.json`, `session_index.jsonl`,
`sessions/<ws>/<session>/{state.json,wire.jsonl,logs/}` and a 16-shard
`cache/query-store/` with a write-ahead log per shard. Two concurrent fan-out
slots sharing one root write all of that to the same paths — the corruption
`test_cursor_profile_isolation.py` was written for, arriving through a different
CLI.

The probe measured both halves of the claim this module pins: without
`KIMI_CODE_HOME` the canary is **dirty** (every path above appears under
`$HOME/.kimi-code/`), and with it the canary is **clean**. So the redirect is
load-bearing, not hygiene.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.providers import kimi as kimi_module  # noqa: E402
from utils.providers.kimi import KimiProvider  # noqa: E402

MODEL = "moonshotai/kimi-k2-thinking"


@pytest.fixture(autouse=True)
def _clean_tracking():
    """Each test owns its own set of created roots."""
    kimi_module.cleanup_home_dirs()
    yield
    kimi_module.cleanup_home_dirs()


@pytest.fixture
def sentinel_home(tmp_path, monkeypatch):
    """A `$HOME` that must still be empty when the test ends."""
    home = tmp_path / "sentinel-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("KIMI_CODE_HOME", raising=False)
    return home


class TestKimiHomeIsolation:
    def test_resolve_env_never_yields_a_path_under_the_real_home(self, sentinel_home):
        """(1) The redirect points somewhere else, and `$HOME` stays untouched."""
        env = KimiProvider().resolve_env(MODEL)

        root = env["KIMI_CODE_HOME"]
        assert not root.startswith(str(sentinel_home)), root
        assert Path(root).is_dir()
        assert list(sentinel_home.iterdir()) == [], "the adapter wrote into $HOME"

    def test_two_concurrent_calls_get_two_different_writable_roots(
        self, sentinel_home
    ):
        """(2) The assertion this module exists for.

        Two fan-out slots on kimi — the same module-level singleton, since
        `_PROVIDERS` holds one instance shared across the whole fan-out — must
        never be handed the same config root. Run them genuinely concurrently
        rather than in sequence, because the failure mode being excluded is a
        race on a shared module-level directory, not a naming collision.
        """
        provider = KimiProvider()

        with ThreadPoolExecutor(max_workers=2) as pool:
            roots = [
                f.result()["KIMI_CODE_HOME"]
                for f in [pool.submit(provider.resolve_env, MODEL) for _ in range(2)]
            ]

        assert len(set(roots)) == 2, roots
        for root in roots:
            assert Path(root).is_dir()
            assert os.access(root, os.W_OK)

    def test_the_root_is_private_to_this_user(self, sentinel_home):
        """0700, asserted rather than inherited from `mkdtemp`'s default."""
        root = KimiProvider().resolve_env(MODEL)["KIMI_CODE_HOME"]

        assert oct(Path(root).stat().st_mode & 0o777) == "0o700"

    def test_a_user_pinned_root_is_respected_and_never_removed(
        self, tmp_path, monkeypatch
    ):
        """A `KIMI_CODE_HOME` the user set is theirs: not overridden, not deleted."""
        pinned = tmp_path / "my-kimi-home"
        pinned.mkdir()
        (pinned / "config.toml").write_text("# mine\n", encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(pinned))

        env = KimiProvider().resolve_env(MODEL)

        assert "KIMI_CODE_HOME" not in env, "adapter overrode the user's own root"
        kimi_module.cleanup_home_dirs()
        assert (pinned / "config.toml").exists()

    def test_the_created_roots_are_gone_after_cleanup(self, sentinel_home):
        """(4) The dirs this process created do not outlive it.

        `cleanup_home_dirs` is registered with `atexit` at the first creation,
        so this is the same code path a finished run takes.
        """
        provider = KimiProvider()
        roots = [provider.resolve_env(MODEL)["KIMI_CODE_HOME"] for _ in range(3)]
        assert all(Path(r).is_dir() for r in roots)

        kimi_module.cleanup_home_dirs()

        assert not any(Path(r).exists() for r in roots)
        assert kimi_module.isolated_home_dirs() == ()

    def test_the_prefix_makes_a_leak_sweepable(self, sentinel_home):
        """A SIGKILL skips `atexit`; the fixed prefix is what makes the residue
        findable by name instead of anonymous under TMPDIR."""
        root = KimiProvider().resolve_env(MODEL)["KIMI_CODE_HOME"]

        assert Path(root).name.startswith(kimi_module._HOME_PREFIX)

    def test_cleanup_never_raises_on_an_already_removed_root(self, sentinel_home):
        """Containment must never be the thing that fails a run — and this path
        runs from `atexit`, where an exception is worst of all."""
        import shutil

        root = KimiProvider().resolve_env(MODEL)["KIMI_CODE_HOME"]
        shutil.rmtree(root)

        kimi_module.cleanup_home_dirs()  # must not raise


class TestKimiStrippedNames:
    """(3) `get_remove_env` carries every name in §8.2's Stripped column."""

    def test_every_unset_family_member_is_stripped_by_name(self):
        provider = KimiProvider()
        injected = set(provider.get_env(MODEL))
        stripped = set(provider.get_remove_env())

        for name in kimi_module._KIMI_MODEL_VARS:
            if name in injected:
                continue
            assert name in stripped, f"unset KIMI_MODEL_* survives the shell: {name}"

    def test_the_vendor_and_legacy_names_are_stripped_by_name(self):
        stripped = set(KimiProvider().get_remove_env())

        for name in kimi_module._STRIPPED_VENDOR_VARS:
            assert name in stripped, name

    def test_nothing_the_adapter_injects_is_also_stripped(self):
        """A value we strip must never also be a value we set — that is a
        contradiction the composition order would resolve silently."""
        provider = KimiProvider()

        assert not set(provider.get_env(MODEL)) & set(provider.get_remove_env())
