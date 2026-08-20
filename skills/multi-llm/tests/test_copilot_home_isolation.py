"""copilot's per-slot config root, asserted rather than assumed (§8.2).

**Why this is a module of its own, and why it is not covered by the container
lane.** `tests/container/wrapper.py` rebuilds every child's environment from
scratch and points `HOME`/`XDG_*`/`TMPDIR` at a fresh per-invocation root, so the
container lane isolates for free and would stay green no matter what this adapter
does. Production fan-out gets none of that: `resolve_env` composes on top of the
user's real environment, so whatever the adapter does not inject, strip or
redirect is inherited from the user's shell — and whatever the CLI writes lands
in the user's real `$HOME`.

**The split this adapter has to get right, and it is copilot-specific.** The §6
probe measured that copilot keeps its state in *two* places with *two* different
owners:

* `COPILOT_HOME` — config and state, `$HOME/.copilot` by default. Per slot, and
  that is what this module pins.
* `XDG_CACHE_HOME` — the **209 MB self-extracted package cache**. `COPILOT_HOME`
  does **not** relocate it (a direct answer to §8.2's open question, and the
  answer is no). The adapter deliberately leaves it alone: a per-slot value would
  re-extract 209 MB on every spawn, and a hardcoded absolute path would be wrong
  on a host. The E2E image pins it to a pre-warmed `/opt` path at build time.

So "isolate everything the CLI writes" is the wrong goal here — the cache is
meant to be shared — and `test_the_package_cache_is_left_to_the_environment`
pins the distinction so a later "fix" does not quietly make every spawn 209 MB
more expensive.

**The strip list is load-bearing in a way most adapters' are not.**
`COPILOT_ALLOW_ALL` is the environment twin of `--allow-all-tools`: a user with
it exported promotes the run to §6.5 tier 4 with **no argv change at all**. The
recorded tier is only enforceable because that name is stripped, which is why it
gets its own assertion rather than riding the generic sweep.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.providers import copilot as copilot_module  # noqa: E402
from utils.providers.copilot import CopilotProvider  # noqa: E402

MODEL = "openai/gpt-4.1-mini"


@pytest.fixture(autouse=True)
def _clean_tracking():
    """Each test owns its own set of created roots."""
    copilot_module.cleanup_home_dirs()
    yield
    copilot_module.cleanup_home_dirs()


@pytest.fixture
def sentinel_home(tmp_path, monkeypatch):
    """A `$HOME` that must still be empty when the test ends."""
    home = tmp_path / "sentinel-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    return home


class TestCopilotHomeIsolation:
    def test_resolve_env_never_yields_a_path_under_the_real_home(self, sentinel_home):
        """(1) The redirect points somewhere else, and `$HOME` stays untouched."""
        env = CopilotProvider().resolve_env(MODEL)

        root = env["COPILOT_HOME"]
        assert not root.startswith(str(sentinel_home)), root
        assert Path(root).is_dir()
        assert list(sentinel_home.iterdir()) == [], "the adapter wrote into $HOME"

    def test_two_concurrent_calls_get_two_different_writable_roots(
        self, sentinel_home
    ):
        """(2) The assertion this module exists for.

        Two fan-out slots on copilot — the same module-level singleton, since
        `_PROVIDERS` holds one instance shared across the whole fan-out — must
        never be handed the same config root. Run them genuinely concurrently
        rather than in sequence, because the failure mode being excluded is a
        race on a shared module-level directory, not a naming collision.
        """
        provider = CopilotProvider()

        with ThreadPoolExecutor(max_workers=2) as pool:
            roots = [
                f.result()["COPILOT_HOME"]
                for f in [pool.submit(provider.resolve_env, MODEL) for _ in range(2)]
            ]

        assert len(set(roots)) == 2, roots
        for root in roots:
            assert Path(root).is_dir()
            assert os.access(root, os.W_OK)

    def test_the_root_is_private_to_this_user(self, sentinel_home):
        """0700, asserted rather than inherited from `mkdtemp`'s default."""
        root = CopilotProvider().resolve_env(MODEL)["COPILOT_HOME"]

        assert oct(Path(root).stat().st_mode & 0o777) == "0o700"

    def test_a_user_pinned_root_is_respected_and_never_removed(
        self, tmp_path, monkeypatch
    ):
        """Their directory, their rules — we neither replace it nor delete it."""
        pinned = tmp_path / "my-copilot-home"
        pinned.mkdir()
        monkeypatch.setenv("COPILOT_HOME", str(pinned))

        env = CopilotProvider().resolve_env(MODEL)

        assert env.get("COPILOT_HOME") in (None, str(pinned))
        assert copilot_module.isolated_home_dirs() == ()
        copilot_module.cleanup_home_dirs()
        assert pinned.is_dir(), "a user-pinned root was removed"

    def test_the_created_roots_are_gone_after_cleanup(self, sentinel_home):
        """(4) The per-run dir does not survive the process."""
        provider = CopilotProvider()
        roots = [provider.resolve_env(MODEL)["COPILOT_HOME"] for _ in range(3)]

        copilot_module.cleanup_home_dirs()

        for root in roots:
            assert not Path(root).exists(), root

    def test_the_prefix_makes_a_leak_sweepable(self, sentinel_home):
        """A SIGKILL skips `atexit`, so the fixed prefix is the recovery path."""
        root = CopilotProvider().resolve_env(MODEL)["COPILOT_HOME"]

        assert Path(root).name.startswith("multi-llm-copilot-")

    def test_cleanup_never_raises_on_an_already_removed_root(self, sentinel_home):
        """Containment must never be the thing that fails a run."""
        import shutil

        root = CopilotProvider().resolve_env(MODEL)["COPILOT_HOME"]
        shutil.rmtree(root)

        copilot_module.cleanup_home_dirs()  # must not raise

    def test_the_package_cache_is_left_to_the_environment(self, sentinel_home):
        """`XDG_CACHE_HOME` is deliberately NOT set — see the module docstring.

        `COPILOT_HOME` does not relocate the 209 MB package cache, and isolating
        it per slot would pay that extraction on every spawn.
        """
        env = CopilotProvider().resolve_env(MODEL)

        assert "XDG_CACHE_HOME" not in env


class TestCopilotStrippedNames:
    """(3) `get_remove_env` carries every name in §8.2's Stripped column."""

    @pytest.mark.parametrize(
        "name,why",
        [
            ("COPILOT_GITHUB_TOKEN", "the seat path, first in precedence order"),
            ("GH_TOKEN", "the seat path, second in precedence order"),
            ("GITHUB_TOKEN", "the seat path, third in precedence order"),
            ("COPILOT_PROVIDER_BEARER_TOKEN", "outranks the API key we inject"),
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "setting it auto-enables telemetry"),
            ("COPILOT_OTEL_ENABLED", "the explicit telemetry switch"),
            ("COPILOT_MODEL", "we pass --model; a stale one confuses a bug report"),
            ("COPILOT_CUSTOM_INSTRUCTIONS_DIRS", "reproducibility of a review answer"),
        ],
    )
    def test_every_measured_name_is_stripped(self, name, why):
        assert name in CopilotProvider().get_remove_env(), why

    def test_the_tier_widening_twin_is_stripped(self):
        """The one strip that is a ship-gate rather than hygiene.

        `COPILOT_ALLOW_ALL` promotes the run to §6.5 tier 4 with no argv change,
        so without this the recorded tier is a claim about our flags rather than
        about the child's permissions.
        """
        assert "COPILOT_ALLOW_ALL" in CopilotProvider().get_remove_env()

    def test_nothing_the_adapter_injects_is_also_stripped(self):
        """A value we strip must never also be a value we set — the two would
        race and the winner would depend on composition order."""
        provider = CopilotProvider()
        injected = set(provider.get_env(MODEL))
        stripped = set(provider.get_remove_env())

        assert injected & stripped == set()
