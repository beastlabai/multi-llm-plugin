"""qwen-code's per-slot config root, asserted rather than assumed (§8.2).

**Why this is a module of its own, and why the container lane does not cover
it.** `tests/container/wrapper.py` rebuilds every child's environment from
scratch and points `HOME`/`XDG_*`/`TMPDIR` at a fresh per-invocation root, so the
container lane isolates for free and would stay green whatever this adapter does.
Production fan-out gets none of that: `resolve_env` composes on top of the user's
real environment, so whatever the adapter does not inject, strip or redirect is
inherited from the user's shell.

**§8.2 called this qwen's "sharp open item" and the measurement closed it.**
§3.3 recorded that qwen "writes `$HOME/.qwen/settings.json` and exposes no
config-dir env var", and ranked a per-spawn `HOME` redirect as the likely fix —
with a warning that the gemini-cli lineage "crashes outright if `$HOME` does not
exist". Three of those claims are wrong at 0.21.13, and the probe measured each:

* **`QWEN_HOME` exists and is exact.** With it set, `installation_id`,
  `usage/token-usage-YYYY-MM.jsonl`, `usage_record.jsonl` and
  `tmp/<sha>/scheduled_tasks.lock` land *directly under* it, and `$HOME/.qwen`
  is **not created at all** — verified from both a warm and a freshly wiped
  `$HOME`. So it replaces `$HOME/.qwen`, not `$HOME`, and the redirect's cost
  (re-injecting `GIT_CONFIG_GLOBAL` and friends) is never incurred.
  `XDG_CONFIG_HOME`, `QWEN_CODE_HOME` and `QWEN_CONFIG_DIR` were measured and
  are **not** honoured.
* **`$HOME` need not exist** — a run against a non-existent `HOME` returned rc=0
  and created it.
* **The concurrency corruption was not reproducible** in any of three
  configurations (separate `QWEN_HOME`, shared `QWEN_HOME`, shared real `$HOME`):
  qwen never writes `settings.json` on this path and its per-run writes are
  append-only JSONL.

That last point is why the per-slot root is set anyway rather than dropped: it is
free, it satisfies §8.2's test (2) by construction, and it keeps a fan-out's
`usage_record.jsonl` out of the user's own qwen state. This module pins the
mechanism so a later "simplification" that drops it is a visible test failure
rather than a silent regression in someone's `$HOME`.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.providers import qwen_code as qwen_module  # noqa: E402
from utils.providers.qwen_code import QwenCodeProvider  # noqa: E402

MODEL = "qwen/qwen3.8-max"


@pytest.fixture(autouse=True)
def _clean_tracking():
    """Each test owns its own set of created roots."""
    qwen_module.cleanup_home_dirs()
    yield
    qwen_module.cleanup_home_dirs()


@pytest.fixture
def sentinel_home(tmp_path, monkeypatch):
    """A `$HOME` that must still be empty when the test ends."""
    home = tmp_path / "sentinel-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("QWEN_HOME", raising=False)
    return home


class TestQwenHomeIsolation:
    def test_resolve_env_never_yields_a_path_under_the_real_home(self, sentinel_home):
        """§8.2 test (1): the sentinel `$HOME` must stay empty."""
        env = QwenCodeProvider().get_env(MODEL)

        root = env["QWEN_HOME"]
        assert not root.startswith(str(sentinel_home))
        assert list(sentinel_home.iterdir()) == []

    def test_two_concurrent_calls_get_two_different_writable_roots(
        self, sentinel_home
    ):
        """§8.2 test (2) — the assertion the whole mechanism exists for.

        Two fan-out slots on qwen-code, resolved concurrently, must not share one
        writable root. The probe could not reproduce corruption from sharing, but
        "we could not make it corrupt" is not the same as "it cannot", and the
        per-slot root costs nothing.
        """
        provider = QwenCodeProvider()

        with ThreadPoolExecutor(max_workers=2) as pool:
            roots = [
                f.result()["QWEN_HOME"]
                for f in [pool.submit(provider.get_env, MODEL) for _ in range(2)]
            ]

        assert roots[0] != roots[1]
        assert all(Path(r).is_dir() for r in roots)

    def test_one_instance_serves_every_slot_without_stashing_state(self, sentinel_home):
        """Adapters are module-level singletons shared across a fan-out, so a
        per-run path on `self` would be a data race rather than an API."""
        provider = QwenCodeProvider()

        first = provider.get_env(MODEL)["QWEN_HOME"]
        second = provider.get_env("qwen/qwen3-max")["QWEN_HOME"]

        assert first != second

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits: Windows has no 0700 to read back")
    def test_the_root_is_private_to_this_user(self, sentinel_home):
        root = Path(QwenCodeProvider().get_env(MODEL)["QWEN_HOME"])
        assert oct(root.stat().st_mode & 0o777) == "0o700"

    def test_a_user_pinned_root_is_respected_and_never_removed(
        self, tmp_path, monkeypatch
    ):
        """Their directory, their rules: we neither replace nor delete it."""
        pinned = tmp_path / "my-qwen"
        pinned.mkdir()
        monkeypatch.setenv("QWEN_HOME", str(pinned))

        env = QwenCodeProvider().get_env(MODEL)

        assert "QWEN_HOME" not in env
        assert qwen_module.isolated_home_dirs() == ()

        qwen_module.cleanup_home_dirs()
        assert pinned.is_dir()

    def test_the_created_roots_are_gone_after_cleanup(self, sentinel_home):
        """§8.2 test (4): the per-run dir does not survive the run."""
        provider = QwenCodeProvider()
        roots = [Path(provider.get_env(MODEL)["QWEN_HOME"]) for _ in range(3)]
        assert all(r.is_dir() for r in roots)

        qwen_module.cleanup_home_dirs()

        assert not any(r.exists() for r in roots)
        assert qwen_module.isolated_home_dirs() == ()

    def test_the_prefix_makes_a_leak_sweepable(self, sentinel_home):
        """A SIGKILL skips `atexit`, so the fixed prefix is the recovery path."""
        root = Path(QwenCodeProvider().get_env(MODEL)["QWEN_HOME"])
        assert root.name.startswith("multi-llm-qwen-")

    def test_cleanup_never_raises_on_an_already_removed_root(self, sentinel_home):
        """Containment must never be the thing that fails a run."""
        import shutil

        root = Path(QwenCodeProvider().get_env(MODEL)["QWEN_HOME"])
        shutil.rmtree(root)

        qwen_module.cleanup_home_dirs()  # must not raise

    def test_a_failed_mkdtemp_runs_without_the_root_rather_than_refusing(
        self, sentinel_home, monkeypatch
    ):
        """Refusing a slot over a containment mechanism is the worse trade."""
        def boom(*_a, **_k):
            raise OSError("no space left on device")

        monkeypatch.setattr(qwen_module.tempfile, "mkdtemp", boom)

        env = QwenCodeProvider().get_env(MODEL)

        assert "QWEN_HOME" not in env
        assert env["OPENAI_MODEL"] == MODEL


class TestQwenStrippedNames:
    """§8.2 test (3): every name in the Stripped column, asserted BY NAME."""

    def test_the_vendor_and_legacy_names_are_stripped_by_name(self):
        stripped = set(QwenCodeProvider().get_remove_env())

        for name in (
            "OPENAI_API_BASE",
            "QWEN_API_KEY",
            "QWEN_BASE_URL",
            "QWEN_CODE_MODEL",
            "QWEN_DEFAULT_AUTH_TYPE",
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_BASE_URL",
            "MODELSCOPE_API_KEY",
        ):
            assert name in stripped, name

    def test_nothing_the_adapter_injects_is_also_stripped(self):
        """A value we strip must never also be a value anything branches on."""
        provider = QwenCodeProvider()
        assert set(provider.get_remove_env()).isdisjoint(provider.get_env(MODEL))

    def test_the_openai_trio_survives_a_hostile_parent_environment(self, monkeypatch):
        """§3.3's money bug, in its exact original shape.

        A stale inherited `OPENAI_BASE_URL=https://api.openai.com/v1` COMPLETES
        the trio and would be used, pairing the user's real OpenAI credit with an
        OpenRouter slug. The adapter's values must win, and no `api.openai.com`
        substring may survive anywhere in the composed child env.
        """
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-parent")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        before = dict(os.environ)

        provider = QwenCodeProvider()
        child = dict(os.environ)
        child.update(provider.get_env(MODEL))
        for name in provider.get_remove_env():
            child.pop(name, None)

        assert child["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
        assert child["OPENAI_API_KEY"] == "sk-or-test"
        assert child["OPENAI_MODEL"] == MODEL
        assert not any("api.openai.com" in v for v in child.values())
        # rule 1: the parent process is untouched, so codex's next spawn in the
        # same fan-out still sees its own environment.
        assert dict(os.environ) == before
