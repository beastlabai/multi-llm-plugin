"""The private config roots adapters hand their CLIs (``utils/providers/private_home.py``).

copilot, kimi, qwen-code and vibe each create one ``0700`` root per spawn
through one :class:`PrivateHomes`, and remove only the roots their own
process created. The per-adapter isolation tests
(``test_*_home_isolation.py``) pin what each CLI sees; this file pins the
shared lifecycle, and the fork guard on opencode's shared config home.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import copilot, kimi, opencode, private_home, qwen_code, vibe  # noqa: E402
from utils.providers.private_home import PrivateHomes  # noqa: E402

ADAPTERS = {"copilot": copilot, "kimi": kimi, "qwen_code": qwen_code, "vibe": vibe}


@pytest.fixture
def homes():
    instance = PrivateHomes("multi-llm-test-private-home-")
    yield instance
    instance.cleanup()


def test_create_makes_a_fresh_private_root_under_the_prefix(homes):
    first, second = homes.create(), homes.create()

    assert first != second
    for path in (first, second):
        assert Path(path).is_dir()
        assert Path(path).name.startswith(homes.prefix)
        assert Path(path).parent == Path(tempfile.gettempdir())
        if os.name != "nt":  # POSIX mode bits: Windows keeps only a read-only flag
            assert Path(path).stat().st_mode & 0o777 == 0o700
    assert homes.isolated_home_dirs() == (first, second)


def test_cleanup_removes_every_root_and_never_raises(homes):
    roots = [homes.create() for _ in range(3)]
    (Path(roots[0]) / "sessions").mkdir()
    (Path(roots[0]) / "sessions" / "s.jsonl").write_text("{}\n", encoding="utf-8")
    shutil.rmtree(roots[1])  # already gone: still no exception

    homes.cleanup()

    assert not any(Path(root).exists() for root in roots)
    assert homes.isolated_home_dirs() == ()
    homes.cleanup()  # idempotent


def test_the_atexit_sweep_is_registered_once_at_the_first_creation(homes, monkeypatch):
    registered = []
    monkeypatch.setattr(private_home.atexit, "register", registered.append)

    homes.create()
    homes.create()

    assert registered == [homes.cleanup]


def test_a_root_that_cannot_be_made_private_is_removed_and_raises(homes, monkeypatch):
    made = []
    real_mkdtemp = private_home.tempfile.mkdtemp

    def _mkdtemp(**kwargs):
        made.append(real_mkdtemp(**kwargs))
        return made[-1]

    def _chmod(_path, _mode):
        raise PermissionError("chmod refused")

    monkeypatch.setattr(private_home.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(private_home.os, "chmod", _chmod)

    with pytest.raises(OSError):
        homes.create()

    assert made and not Path(made[0]).exists()
    assert homes.isolated_home_dirs() == ()


def test_a_forked_child_never_removes_its_parents_roots(homes):
    parent_root = homes.create()
    homes._pid = os.getpid() + 1  # this process did not create what is tracked

    assert homes.isolated_home_dirs() == ()
    homes.cleanup()
    assert Path(parent_root).is_dir(), "a child removed its parent's root"

    child_root = homes.create()
    assert homes.isolated_home_dirs() == (child_root,)
    homes.cleanup()
    assert not Path(child_root).exists()
    shutil.rmtree(parent_root)


def test_reset_forgets_without_removing(homes):
    root = homes.create()
    homes.reset()
    assert homes.isolated_home_dirs() == ()
    assert Path(root).is_dir()
    shutil.rmtree(root)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_each_adapter_keeps_its_module_names_on_its_own_instance(name):
    module = ADAPTERS[name]
    assert module._HOMES.prefix == module._HOME_PREFIX
    assert module.isolated_home_dirs == module._HOMES.isolated_home_dirs
    assert module.cleanup_home_dirs == module._HOMES.cleanup
    assert module.reset_home_dirs == module._HOMES.reset


def test_the_adapters_share_no_instance_and_no_prefix():
    assert len({id(module._HOMES) for module in ADAPTERS.values()}) == len(ADAPTERS)
    assert len({module._HOME_PREFIX for module in ADAPTERS.values()}) == len(ADAPTERS)


class TestOpencodeSharedHomeForkGuard:
    """opencode's one shared config home per process: a forked child neither
    reuses nor removes the directory its parent made."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        opencode.sweep_config_home()
        yield
        opencode.sweep_config_home()

    def test_a_child_sweep_leaves_the_parents_directory(self, monkeypatch):
        home = opencode.acquire_config_home()
        monkeypatch.setattr(opencode, "_CONFIG_HOME_PID", os.getpid() + 1)

        opencode.sweep_config_home()

        assert Path(home).is_dir(), "a child removed its parent's config home"
        assert opencode._CONFIG_HOME is None
        shutil.rmtree(home)

    def test_a_child_acquire_makes_its_own_directory(self, monkeypatch):
        parent = opencode.acquire_config_home()
        monkeypatch.setattr(opencode, "_CONFIG_HOME_PID", os.getpid() + 1)

        child = opencode.acquire_config_home()

        assert child != parent
        assert opencode._CONFIG_HOME_USERS == 1
        if os.name != "nt":
            assert Path(child).stat().st_mode & 0o777 == 0o700
        opencode.release_config_home()
        assert not Path(child).exists()
        assert Path(parent).is_dir()
        shutil.rmtree(parent)

    def test_the_owning_process_still_sweeps_its_directory(self):
        home = opencode.acquire_config_home()
        assert opencode.acquire_config_home() == home
        opencode.release_config_home()
        assert Path(home).is_dir(), "removed while a slot still holds it"
        opencode.release_config_home()
        assert not Path(home).exists()
