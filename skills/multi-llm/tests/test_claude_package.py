"""The Claude Code plugin: the generated tree, its builder, and the tree at run time.

A Claude Code install copies ``packaging/claude/multi-llm/`` wholesale with no build
on the user's machine, so whatever that directory holds is what ships. This suite
grades it from several sides, and every row says which input it reads:

* **(src)** the committed sources: ``packaging/claude/build.py``, the shared
  ``payload`` module, the Makefile.
* **(tree)** the **checked-in** plugin. A row may read it on a clean checkout.
* **(build)** ``build.py``'s output in a temp directory, from the session fixture
  ``built_plugin``, with dirty inputs allowed so it grades the working tree. Needs
  git and nothing else.
* **(rig)** ``build()`` over a throwaway git repository under ``tmp_path``, which is
  how a refusal is driven without touching this checkout. Each refusal also proves
  that the previous complete tree survived it byte for byte.
* **(pack)** the actual ``npm pack`` artifact, unpacked, for the rules every final
  distribution owes.

**Where a row runs.** A plain row runs on Linux, macOS and Windows, so it reads file
modes from the git index and never from the disk. A row that needs a POSIX shell
skips on Windows and says so. A row that needs ``uv``, ``make`` or ``npm`` carries
``package`` and runs in the Linux package job, where ``MLLM_REQUIRE_PACKAGE_ROWS``
turns its skip into a failure. The ``claude plugin validate`` row is the exception:
that job has no ``claude`` binary, so it skips by name in every lane.

**The install path has a space in it on purpose.** ``${CLAUDE_SKILL_DIR}`` is
substituted into quoted shell text, and a tree that only works from a path without
spaces would pass every other row here.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
CLAUDE_PKG = REPO_ROOT / "packaging" / "claude"
OPENCODE_PKG = REPO_ROOT / "packaging" / "opencode"
PLUGIN_DIR = CLAUDE_PKG / "multi-llm"
PAYLOAD = "skills/multi-llm"
PLUGIN_RELATIVE = "packaging/claude/multi-llm"
CODEX_RELATIVE = "packaging/codex/multi-llm"

for _path in (str(SKILL_ROOT), str(REPO_ROOT / "packaging")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from common import payload  # noqa: E402
from tests.test_report_check import (  # noqa: E402
    COMMAND, Transcript, block_output, report,
)
from tests.test_hook_transcript_settle import (  # noqa: E402
    BACKGROUND_PHASE, LAUNCHED, LOOKUP, denied, own_call_entry, reply, stop_input,
    tool_result, tool_use, wakeup_input, write_later,
)
from tests.test_hook_transcript_settle import line as transcript_line  # noqa: E402
from tests.test_wakeup_check import hook_input, phase, racing_03  # noqa: E402
from utils import cli_common, unattended  # noqa: E402


def _load_builder(name: str, path: Path):
    """Each host's builder is called ``build.py``, and one pytest session holds them
    all, so each is loaded under a name of its own rather than ``import build``."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


claude_build = _load_builder("claude_build", CLAUDE_PKG / "build.py")
opencode_build = _load_builder(
    "opencode_build_for_the_claude_suite", OPENCODE_PKG / "build.py",
)

#: Set by the package job: a row that would skip fails instead.
REQUIRE_ROWS_ENV = "MLLM_REQUIRE_PACKAGE_ROWS"

PROVIDERS_DOC_URL = "https://github.com/beastlabai/multi-llm-plugin/blob/master/docs/providers.md"
CONFIGURATION_DOC_URL = (
    "https://github.com/beastlabai/multi-llm-plugin/blob/master/docs/configuration.md"
)

posix_shell = pytest.mark.skipif(
    os.name == "nt", reason="drives POSIX shell scripts; Windows has no /bin/sh",
)


def _unavailable(reason: str) -> None:
    if os.environ.get(REQUIRE_ROWS_ENV):
        pytest.fail(f"{REQUIRE_ROWS_ENV} is set but this row cannot run: {reason}")
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _no_ambient_overrides(monkeypatch):
    """A developer's shell must not change what a row renders or builds."""
    for name in (
        unattended.UV_RUN_ARGS_ENV,
        multi_llm.LEGACY_INSTRUCTIONS_ENV,
        claude_build.ALLOW_DIRTY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Reading trees and the git index
# ---------------------------------------------------------------------------

def _index_modes(prefix: str, root: Path = REPO_ROOT) -> Dict[str, str]:
    """``{path: git mode}`` under *prefix*, which is the same answer on every OS."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z", "--", prefix],
        check=True, capture_output=True, text=True,
    ).stdout
    modes: Dict[str, str] = {}
    for record in out.split("\0"):
        if record:
            meta, _, path = record.partition("\t")
            modes[path] = meta.split(" ", 1)[0]
    return modes


def _tracked_below(prefix: str) -> Set[str]:
    """Tracked paths under *prefix*, relative to it: what the Git route carries."""
    strip = prefix.rstrip("/") + "/"
    return {path[len(strip):] for path in _index_modes(prefix)}


def _exec_bit(path: Path) -> bool:
    return os.name != "nt" and bool(path.stat().st_mode & stat.S_IXUSR)


def _snapshot(tree: Path, *, shipped_only: bool = False) -> Dict[str, Tuple[str, bool]]:
    """``{relative: (sha256, executable)}``, so a comparison sees bytes and modes."""
    found: Dict[str, Tuple[str, bool]] = {}
    for path in sorted(Path(tree).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(tree).as_posix()
        if shipped_only and claude_build._is_by_product(relative):  # noqa: SLF001
            continue
        found[relative] = (hashlib.sha256(path.read_bytes()).hexdigest(), _exec_bit(path))
    return found


def _differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> List[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _expected_plugin_files() -> Set[str]:
    """What the plugin must hold, from ``git ls-files`` and the manifest alone, so it
    does not lean on the build's own bookkeeping."""
    staged = payload.expected_payload(
        REPO_ROOT, extra_allowlist=claude_build.LEGACY_ALLOWLIST,
    )
    return set(claude_build.PLUGIN_FILES) | {f"{PAYLOAD}/{name}" for name in staged}


def _copy_shipped(source: Path, destination: Path) -> None:
    """Copy a plugin the way an install does: files only, by-products left behind."""
    for relative in claude_build.shipped_files(source):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def _clean_env(**extra: str) -> Dict[str, str]:
    """The ambient environment minus everything that would redirect ``uv`` or the
    skill. ``MULTI_LLM_UV`` stays: the dispatcher's own uv check reads it."""
    environment = {
        name: value for name, value in os.environ.items()
        if name not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH"}
        and (not name.startswith("MULTI_LLM_") or name == "MULTI_LLM_UV")
    }
    environment["PYTHONUTF8"] = "1"
    environment.update(extra)
    return environment


@pytest.fixture(scope="session")
def built_plugin(tmp_path_factory) -> Path:
    """``build.py``'s output, in a temp directory. Never the checked-in tree."""
    destination = tmp_path_factory.mktemp("claude-plugin") / "multi-llm"
    claude_build.build(destination=destination, allow_dirty=True)
    return destination


@pytest.fixture(params=["checked-in", "rebuilt"])
def finished(request, built_plugin) -> Path:
    """The two finished trees a rule has to hold for."""
    return PLUGIN_DIR if request.param == "checked-in" else built_plugin


@pytest.fixture(scope="module")
def staged_install(tmp_path_factory) -> Path:
    """The plugin as a user has it: copied whole to a path containing spaces."""
    root = tmp_path_factory.mktemp("claude-install") / "my plugin dir"
    _copy_shipped(PLUGIN_DIR, root)
    return root


@pytest.fixture(scope="session")
def opencode_staged(tmp_path_factory) -> Path:
    """The OpenCode builder's staging directory, for the shared-API boundary."""
    destination = tmp_path_factory.mktemp("opencode-staged")
    opencode_build.build(destination=destination, allow_dirty=True)
    return destination


# ===========================================================================
# The finished distribution
# ===========================================================================

class TestFinishedDistribution:
    def test_the_inventory_is_exact_both_ways(self, finished):
        """(tree, build) An unaccounted file ships; a missing one breaks a session."""
        present = set(claude_build.shipped_files(finished))
        expected = _expected_plugin_files()
        assert sorted(expected - present) == [], "missing"
        assert sorted(present - expected) == [], "unexpected"

    def test_the_tracked_tree_is_that_inventory(self):
        """(tree) The Git route serves what the index holds, not what the disk does."""
        tracked = _tracked_below(PLUGIN_RELATIVE)
        expected = _expected_plugin_files()
        assert sorted(expected - tracked) == [], "missing from the index"
        assert sorted(tracked - expected) == [], "tracked but not part of the plugin"

    def test_the_required_files_are_present(self, finished):
        """(tree, build) Named rather than inferred, because the inventory row above
        derives its answer from the same allowlist the build uses."""
        present = set(claude_build.shipped_files(finished))
        required = set(claude_build.PLUGIN_FILES) | {
            f"{PAYLOAD}/{name}"
            for name in (
                *payload.REQUIRED_INVENTORY,
                *claude_build.LEGACY_REQUIRED,
                *payload.target_module_paths(SKILL_ROOT),
            )
        }
        assert sorted(required - present) == []
        for prefix in payload.REQUIRED_TREES:
            assert any(path.startswith(f"{PAYLOAD}/{prefix}") for path in present), prefix

    def test_legacy_holds_exactly_the_named_files(self, finished):
        """(tree, build) A partial ``legacy/`` falls back to the current core with no
        error, so a non-empty directory proves nothing."""
        prefix = f"{PAYLOAD}/legacy/"
        held = {
            path[len(prefix):] for path in claude_build.shipped_files(finished)
            if path.startswith(prefix)
        }
        assert held == set(claude_build.LEGACY_FILES)
        assert len(held) == 17
        assert sum(name.startswith("instructions/") for name in held) == 11
        assert sum(name.startswith("references/") for name in held) == 5
        assert "README.md" in held

    def test_every_file_is_its_authored_source_byte_for_byte(self, finished):
        """(tree, build) Nothing renders the Claude plugin: the root stays the source,
        and a copy that differs is a hand edit a regeneration would discard."""
        sources = dict(claude_build.PLUGIN_FILES)
        for relative in claude_build.shipped_files(finished):
            sources.setdefault(relative, relative)
        changed = [
            relative for relative, source in sources.items()
            if (finished / relative).read_bytes() != (REPO_ROOT / source).read_bytes()
        ]
        assert changed == []

    def test_the_manifest(self, finished):
        """(tree, build) The manifest is the root one verbatim, and its version is the
        owner's: Claude Code caches by it."""
        manifest = json.loads(
            (finished / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        assert manifest["name"] == "multi-llm"
        assert manifest["version"] == claude_build.project_version()
        assert manifest["hooks"] == "./hooks/hooks.json"
        assert (finished / manifest["hooks"]).is_file()
        assert manifest == json.loads(
            (REPO_ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
        )

    def test_the_plugin_carries_no_catalog(self, finished):
        """(tree, build) A catalog inside the generated directory would make it a
        second marketplace rather than a plugin."""
        assert not (finished / ".claude-plugin" / "marketplace.json").exists()

    def test_the_component_rules_hold(self, finished):
        """(tree, build) The build's own assertion, run over the finished tree."""
        assert claude_build.component_problems(
            finished, claude_build.project_version(),
        ) == []

    def test_each_hook_command_resolves_from_the_plugin_root(self, finished):
        """(tree, build) Each command is the quoted ``${CLAUDE_PLUGIN_ROOT}`` spelling
        of a shipped script, and every shipped script is reached by some event."""
        configuration = json.loads(
            (finished / "hooks" / "hooks.json").read_text(encoding="utf-8")
        )
        shipped = claude_build.shipped_files(finished)
        reached = []
        for groups in configuration["hooks"].values():
            for group in groups:
                for hook in group["hooks"]:
                    assert hook["type"] == "command"
                    match = re.fullmatch(
                        r'"\$\{CLAUDE_PLUGIN_ROOT\}/(hooks/[\w.-]+)"', hook["command"],
                    )
                    assert match, hook["command"]
                    assert match.group(1) in shipped, hook["command"]
                    reached.append(match.group(1))
        assert sorted(reached) == sorted(claude_build.HOOK_SCRIPTS)
        assert set(configuration["hooks"]) == {"SessionStart", "Stop", "PreToolUse"}

    def test_the_hook_scripts_are_executable_in_the_index(self):
        """(tree) ``hooks.json`` runs each script by path, so a lost bit is a hook that
        fails in every session. The index is the mode a Git install receives."""
        for prefix in ("hooks", f"{PLUGIN_RELATIVE}/hooks"):
            modes = _index_modes(prefix)
            for script in claude_build.HOOK_SCRIPTS:
                name = script.split("/", 1)[1]
                assert modes[f"{prefix}/{name}"] == "100755", f"{prefix}/{name}"
            assert modes[f"{prefix}/hooks.json"] == "100644"

    def test_every_generated_mode_is_its_source_s(self):
        """(tree) The git tree id is the release comparison, and it covers modes, so a
        generated mode that drifts from its source reads as a shipped change."""
        generated = _index_modes(PLUGIN_RELATIVE)
        sources = {
            **_index_modes(PAYLOAD),
            **_index_modes(".claude-plugin/plugin.json"),
            **_index_modes("hooks"),
            **_index_modes("LICENSE"),
        }
        mapping = dict(claude_build.PLUGIN_FILES)
        drifted = []
        for path, mode in generated.items():
            relative = path[len(PLUGIN_RELATIVE) + 1:]
            if sources.get(mapping.get(relative, relative)) != mode:
                drifted.append((path, mode))
        assert drifted == []
        assert "120000" not in generated.values(), "a symlink is tracked in the plugin"

    def test_no_file_in_the_tree_is_a_symlink(self, finished):
        """(tree, build) A plain install copy silently drops one."""
        linked = [
            path.relative_to(finished).as_posix()
            for path in finished.rglob("*") if path.is_symlink()
            and not claude_build._is_by_product(path.relative_to(finished).as_posix())  # noqa: SLF001
        ]
        assert linked == []

    @posix_shell
    def test_a_rebuild_takes_its_modes_from_the_index(self, built_plugin):
        """(build) Only the bits git records may reach the tree."""
        index = {
            **_index_modes(PAYLOAD),
            **{
                target: _index_modes(source)[source]
                for target, source in claude_build.PLUGIN_FILES.items()
            },
        }
        for relative in claude_build.shipped_files(built_plugin):
            wanted = 0o755 if index[relative] == "100755" else 0o644
            mode = stat.S_IMODE((built_plugin / relative).stat().st_mode)
            assert mode == wanted, relative

    def test_the_by_products_of_running_the_plugin_are_tolerated(self, tmp_path):
        """(tree) Running from the checked-in tree writes a ``.venv`` and
        ``__pycache__``. They are gitignored, so they never ship, and no check may
        read them as part of the plugin."""
        tree = tmp_path / "multi-llm"
        _copy_shipped(PLUGIN_DIR, tree)
        files = claude_build.shipped_files(tree)
        digest = claude_build.digest_of(tree)
        for by_product in (
            f"{PAYLOAD}/.venv/pyvenv.cfg",
            f"{PAYLOAD}/utils/__pycache__/cli_common.cpython-314.pyc",
            f"{PAYLOAD}/.pytest_cache/v/cache/lastfailed",
            f"{PAYLOAD}/.ruff_cache/CACHEDIR.TAG",
        ):
            target = tree / by_product
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("by-product\n", encoding="utf-8")
        assert claude_build.shipped_files(tree) == files
        assert claude_build.digest_of(tree) == digest
        assert claude_build.component_problems(tree, claude_build.project_version()) == []
        staged = payload.expected_payload(
            REPO_ROOT, extra_allowlist=claude_build.LEGACY_ALLOWLIST,
        )
        assert claude_build._inventory_problems(tree, staged) == []  # noqa: SLF001

    def test_a_stray_or_a_missing_file_is_named_both_ways(self, tmp_path):
        """(tree) The inventory check is not a tautology: each direction fires."""
        tree = tmp_path / "multi-llm"
        _copy_shipped(PLUGIN_DIR, tree)
        staged = payload.expected_payload(
            REPO_ROOT, extra_allowlist=claude_build.LEGACY_ALLOWLIST,
        )
        (tree / "hooks" / "stray.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (tree / "LICENSE").unlink()
        problems = claude_build._inventory_problems(tree, staged)  # noqa: SLF001
        assert problems == ["missing: LICENSE", "unexpected: hooks/stray.sh"]

    def test_claude_plugin_validate_accepts_the_tree(self, tmp_path):
        """(tree) The host's own validator, under a throwaway home. CI has no ``claude``
        binary, so this skips by name there rather than failing the package job."""
        claude = shutil.which("claude")
        if claude is None:
            pytest.skip("the claude binary is not on PATH; plugin validate cannot run")
        home = tmp_path / "home"
        home.mkdir()
        result = subprocess.run(
            [claude, "plugin", "validate", str(PLUGIN_DIR)],
            env=dict(
                os.environ, HOME=str(home), USERPROFILE=str(home),
                CLAUDE_CONFIG_DIR=str(home / ".claude"),
            ),
            capture_output=True, encoding="utf-8", errors="replace", timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation passed" in result.stdout + result.stderr


class TestDeterministicRegeneration:
    def test_two_builds_are_byte_identical(self, built_plugin, tmp_path):
        """(build) Mode bits included: the checked-in tree is compared against a clean
        rebuild, so a second answer from the same inputs would fail CI at random."""
        again = tmp_path / "multi-llm"
        claude_build.build(destination=again, allow_dirty=True)
        assert _differences(_snapshot(built_plugin), _snapshot(again)) == []

    def test_a_build_equals_the_checked_in_tree(self, built_plugin):
        """(build vs tree) Bytes and inventory. Built from the working tree, so it goes
        red in the commit that edits a source without regenerating."""
        drift = _differences(
            {name: value[0] for name, value in _snapshot(built_plugin).items()},
            {
                name: value[0]
                for name, value in _snapshot(PLUGIN_DIR, shipped_only=True).items()
            },
        )
        assert drift == [], (
            f"{PLUGIN_RELATIVE} differs from a rebuild; run `make claude-regen`: {drift[:20]}"
        )
        assert claude_build.digest_of(built_plugin) == claude_build.digest_of(PLUGIN_DIR)

    @posix_shell
    def test_the_umask_does_not_reach_the_tree(self, built_plugin, tmp_path):
        """(build) A checkout's mode bits follow its umask, and the tree must not."""
        previous = os.umask(0o077)
        try:
            again = tmp_path / "multi-llm"
            claude_build.build(destination=again, allow_dirty=True)
        finally:
            os.umask(previous)
        assert _differences(_snapshot(built_plugin), _snapshot(again)) == []
        for relative in claude_build.shipped_files(again):
            assert stat.S_IMODE((again / relative).stat().st_mode) in (0o644, 0o755), relative


# ===========================================================================
# The builder's constants and CLI
# ===========================================================================

class TestTheBuilderSources:
    def test_the_legacy_allowlist_is_literal(self):
        """(src) A glob would admit a new file by accident; the exact-inventory check
        names it only because nothing widened the allowlist first."""
        assert len(claude_build.LEGACY_ALLOWLIST) == len(claude_build.LEGACY_FILES) == 17
        for pattern in claude_build.LEGACY_ALLOWLIST:
            assert not set(pattern) & set("*?["), pattern
            assert pattern.startswith(f"{PAYLOAD}/legacy/")

    def test_the_legacy_inventory_is_what_git_tracks(self):
        """(src) The named list and the tracked directory agree, both ways."""
        assert _tracked_below(f"{PAYLOAD}/legacy") == set(claude_build.LEGACY_FILES)

    def test_the_packaging_inputs_name_every_input(self):
        """(src) Everything the build reads outside the payload prefix is on the
        dirty-input list, and the generated tree, which is its output, is not."""
        for source in claude_build.PLUGIN_FILES.values():
            assert source in claude_build.PACKAGING_INPUTS, source
        assert "packaging/claude/build.py" in claude_build.PACKAGING_INPUTS
        assert "packaging/common/payload.py" in claude_build.PACKAGING_INPUTS
        assert not any(
            path.startswith(PLUGIN_RELATIVE) for path in claude_build.PACKAGING_INPUTS
        )

    def test_the_catalog_entry_names_this_tree(self):
        """(src) The build refuses any other source, so the two cannot drift."""
        assert claude_build._catalog_entry_problems(REPO_ROOT) == []  # noqa: SLF001

    def test_loading_the_builder_leaves_its_directory_off_sys_path(self):
        """(src) ``build.py`` on ``sys.path`` would answer a later ``import build`` meant
        for the OpenCode builder, in whichever session loaded this one first."""
        probe = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('claude_build', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules['claude_build'] = module\n"
            "spec.loader.exec_module(module)\n"
            "sys.exit(0 if sys.argv[2] not in sys.path else 9)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, str(CLAUDE_PKG / "build.py"), str(CLAUDE_PKG)],
            env=_clean_env(), capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_allow_dirty_env_is_read_by_exact_spelling(self, monkeypatch):
        """(src) ``main()``'s default, through the parser with ``build()`` stubbed:
        ``=1`` turns it on and nothing else does."""
        seen = []

        def fake_build(*, destination, allow_dirty):
            seen.append(allow_dirty)
            return {"files": 0, "bytes": 0, "destination": destination,
                    "version": "0", "digest": "0"}

        monkeypatch.setattr(claude_build, "build", fake_build)
        for value, expected in (
            ("1", True), ("0", False), ("true", False), ("", False), (None, False),
        ):
            if value is None:
                monkeypatch.delenv(claude_build.ALLOW_DIRTY_ENV, raising=False)
            else:
                monkeypatch.setenv(claude_build.ALLOW_DIRTY_ENV, value)
            assert claude_build.allow_dirty_from_env() is expected, value
            seen.clear()
            assert claude_build.main([]) == 0
            assert seen == [expected], value
        monkeypatch.delenv(claude_build.ALLOW_DIRTY_ENV, raising=False)
        seen.clear()
        assert claude_build.main(["--allow-dirty"]) == 0
        assert seen == [True]

    def test_a_refusal_reaches_the_cli_as_an_error_line(self, monkeypatch, capsys):
        """(src) A refusal is a named message and exit 1, never a traceback."""
        def refuse(**_kwargs):
            raise payload.PayloadError("the refusal text")

        monkeypatch.setattr(claude_build, "build", refuse)
        assert claude_build.main([]) == 1
        assert "ERROR: the refusal text" in capsys.readouterr().err

    def test_print_version_prints_the_owner_s(self, capsys):
        """(src) The release tooling reads the version through this flag."""
        assert claude_build.main(["--print-version"]) == 0
        assert capsys.readouterr().out.strip() == claude_build.project_version()


# ===========================================================================
# The refusals, over a throwaway repository
# ===========================================================================

def _git(repo: Path, *args: str, stdin: Optional[str] = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, encoding="utf-8", input=stdin,
        env=dict(
            os.environ,
            GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
            GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
        ),
    )
    return result.stdout


class Rig:
    """A throwaway repository holding what the build reads, plus a complete tree that
    an earlier clean build left behind, which is what a refusal has to preserve."""

    def __init__(self, base: Path):
        self.base = base
        self.repo = base / "repo"
        self.skill = self.repo / PAYLOAD
        self.out = base / "out" / "multi-llm"

    def build(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("root", self.repo)
        kwargs.setdefault("base", self.skill)
        kwargs.setdefault("destination", self.out)
        return claude_build.build(**kwargs)

    def write(self, relative: str, text: str) -> Path:
        target = self.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return target

    def append(self, relative: str, text: str) -> None:
        target = self.repo / relative
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(text)

    def replace(self, relative: str, old: str, new: str) -> None:
        target = self.repo / relative
        body = target.read_text(encoding="utf-8")
        assert old in body, f"{old!r} is not in {relative}"
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(body.replace(old, new))

    def commit(self, message: str = "edit") -> None:
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "--no-verify", "-m", message)

    def remove(self, relative: str) -> None:
        _git(self.repo, "rm", "-q", "-f", relative)
        _git(self.repo, "commit", "-q", "--no-verify", "-m", f"remove {relative}")

    def track_symlink(self, relative: str, points_at: str) -> None:
        """Commit a symlink through the index alone, which needs no privilege on
        Windows. The working tree is untouched, so reach it with ``allow_dirty``."""
        blob = _git(self.repo, "hash-object", "-w", "--stdin", stdin=points_at).strip()
        _git(self.repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},{relative}")
        _git(self.repo, "commit", "-q", "--no-verify", "-m", f"link {relative}")

    def refused(self, match: str, **kwargs: Any) -> str:
        """The build refuses, the previous tree survives, and no scratch is left."""
        before = _snapshot(self.out)
        with pytest.raises(payload.PayloadError, match=match) as caught:
            self.build(**kwargs)
        assert _differences(before, _snapshot(self.out)) == [], (
            "a refused build changed the previous complete tree"
        )
        assert [path.name for path in self.out.parent.iterdir()] == [self.out.name], (
            "a refused build left scratch beside the tree"
        )
        return str(caught.value)


@pytest.fixture(scope="module")
def rig_template(tmp_path_factory) -> Path:
    """Copied per row, so a row can edit and commit freely. Built once per module,
    because the copy costs a fraction of a clean build."""
    base = tmp_path_factory.mktemp("claude-rig")
    repo = base / "repo"
    payload_files = payload.expected_payload(
        REPO_ROOT, extra_allowlist=claude_build.LEGACY_ALLOWLIST,
    )
    sources = [
        *(f"{PAYLOAD}/{name}" for name in payload_files),
        payload.MANIFEST_RELATIVE,
        *claude_build.PLUGIN_FILES.values(),
        *claude_build.PACKAGING_INPUTS,
        ".claude-plugin/marketplace.json",
        ".gitattributes",
        ".gitignore",
    ]
    for relative in dict.fromkeys(sources):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "add", "-A")
    for script in claude_build.HOOK_SCRIPTS:
        _git(repo, "update-index", "--chmod=+x", script)
        if os.name != "nt":
            os.chmod(repo / script, 0o755)
    _git(repo, "commit", "-q", "--no-verify", "-m", "the plugin's inputs")
    claude_build.build(root=repo, base=repo / PAYLOAD, destination=base / "out" / "multi-llm")
    return base


@pytest.fixture
def rig(rig_template, tmp_path) -> Rig:
    base = tmp_path / "rig"
    shutil.copytree(rig_template, base, symlinks=True)
    return Rig(base)


def _anthropic_shaped() -> str:
    """Assembled at run time, so this file never holds a key-shaped literal."""
    return "sk-" + "ant-" + "a" * 40


class TestTheRig:
    def test_a_clean_rig_builds_the_same_plugin(self, rig, built_plugin):
        """The refusal rows mean nothing unless a clean rig builds what ships, so a
        refusal below is about the row's edit and not about the fixture."""
        assert claude_build.digest_of(rig.out) == claude_build.digest_of(built_plugin)
        report_ = rig.build()
        assert report_["digest"] == claude_build.digest_of(built_plugin)
        assert report_["files"] == len(claude_build.shipped_files(built_plugin))

    def test_a_successful_build_replaces_the_whole_tree(self, rig):
        """A swap, not a merge: a file an earlier build left does not survive."""
        (rig.out / "stale.txt").write_text("left over\n", encoding="utf-8")
        rig.build()
        assert not (rig.out / "stale.txt").exists()
        assert [path.name for path in rig.out.parent.iterdir()] == [rig.out.name]

    def test_base_follows_root(self, rig):
        """Naming another checkout must not read this one's version."""
        rig.replace(
            ".claude-plugin/plugin.json",
            f'"version": "{claude_build.project_version()}"', '"version": "9.9.9"',
        )
        rig.replace(
            f"{PAYLOAD}/pyproject.toml",
            f'version = "{claude_build.project_version()}"', 'version = "9.9.9"',
        )
        rig.commit("bump both ends")
        assert claude_build.build(root=rig.repo, destination=rig.out)["version"] == "9.9.9"

    def test_the_catalog_check_can_be_left_off(self, rig):
        """A fixture repository without a catalog can still build."""
        rig.remove(".claude-plugin/marketplace.json")
        rig.refused("marketplace.json does not parse")
        assert rig.build(check_catalog=False)["files"] > 0


class TestCredentialRefusals:
    @pytest.mark.parametrize("relative", [
        "hooks/doctor.sh",
        "hooks/report-check.sh",
        "hooks/wakeup-check.sh",
        "hooks/hooks.json",
        ".claude-plugin/plugin.json",
        "LICENSE",
    ])
    def test_a_credential_string_outside_the_payload(self, rig, relative):
        """(rig) These files never pass through ``payload.stage``, so only the sweep
        over the complete plugin root can see them. ``LICENSE`` has no suffix, and a
        complete sweep cannot skip a file for that."""
        rig.append(relative, f"\n# {_anthropic_shaped()}\n")
        rig.commit()
        rig.refused(rf"{re.escape(relative)}: Anthropic key")

    def test_a_credential_string_in_the_payload(self, rig):
        """(rig) The same sweep, through the shared staging path."""
        rig.append(f"{PAYLOAD}/SKILL.md", f"\n{_anthropic_shaped()}\n")
        rig.commit()
        rig.refused("SKILL.md: Anthropic key")

    @pytest.mark.parametrize("name, shape", [
        ("hooks/id_deploy", "credential-shaped filename"),
        ("hooks/e2e.env", "credential filename"),
        ("hooks/signing.pem", "credential-shaped filename"),
    ])
    def test_a_credential_file_in_the_plugin_root(self, rig, monkeypatch, name, shape):
        """(rig) The explicit file map is what stops most of these, so the row extends
        it: the sweep is the rule that holds the day someone adds an entry."""
        rig.write(name, "not really a key\n")
        rig.commit()
        monkeypatch.setattr(
            claude_build, "PLUGIN_FILES", {**claude_build.PLUGIN_FILES, name: name},
        )
        rig.refused(rf"{re.escape(name)}: {shape}")

    def test_a_credential_file_in_the_payload(self, rig):
        """(rig) ``utils/**`` is an allowlist glob, so the name alone must refuse it."""
        rig.write(f"{PAYLOAD}/utils/id_backup.py", "# nothing\n")
        rig.commit()
        rig.refused("id_backup.py: credential-shaped filename")

    def test_an_unnamed_file_beside_the_hooks_does_not_ship(self, rig):
        """(rig) The counter-case: the map is explicit, so a file nobody named stays
        out and the build has nothing to refuse."""
        rig.write("hooks/.env", "TOKEN=x\n")
        _git(rig.repo, "add", "-f", "hooks/.env")
        rig.commit()
        rig.build()
        assert "hooks/.env" not in claude_build.shipped_files(rig.out)


class TestLegacyInventoryRefusals:
    @pytest.mark.parametrize("name", claude_build.LEGACY_FILES)
    def test_a_missing_legacy_file(self, rig, name):
        """(rig) Each of the 17: a missing frozen copy would silently serve the current
        core under ``MULTI_LLM_LEGACY_INSTRUCTIONS=1``."""
        rig.remove(f"{PAYLOAD}/legacy/{name}")
        message = rig.refused("the legacy inventory is not the one this plugin ships")
        assert f"missing: legacy/{name}" in message

    def test_a_legacy_file_gone_from_the_working_tree_only(self, rig):
        """(rig) ``--allow-dirty`` reads the disk, so a tracked file deleted but not yet
        ``git rm``-ed is as missing as an untracked one, and gets the same name."""
        (rig.skill / "legacy" / "references" / "uv-check.md").unlink()
        message = rig.refused("the legacy inventory", allow_dirty=True)
        assert "missing: legacy/references/uv-check.md" in message

    @pytest.mark.parametrize("name", [
        "instructions/new-mode.md", "references/new.md", "NOTES.md",
    ])
    def test_an_extra_tracked_legacy_file(self, rig, name):
        """(rig) The list is exact both ways: an extra file is not silently left out."""
        rig.write(f"{PAYLOAD}/legacy/{name}", "frozen\n")
        rig.commit()
        message = rig.refused("the legacy inventory is not the one this plugin ships")
        assert f"not in LEGACY_FILES: legacy/{name}" in message


class TestVersionRefusals:
    def test_the_root_manifest_behind_the_owner(self, rig):
        """(rig) Claude Code caches by the manifest's version, so a stale one means
        nobody who has it receives the bytes."""
        rig.replace(
            ".claude-plugin/plugin.json",
            f'"version": "{claude_build.project_version()}"', '"version": "0.0.1"',
        )
        rig.commit()
        message = rig.refused("says version '0.0.1'")
        assert claude_build.project_version() in message

    def test_the_owner_moved_without_the_manifest(self, rig):
        """(rig) The other direction: ``project_version`` reads *base*."""
        rig.replace(
            f"{PAYLOAD}/pyproject.toml",
            f'version = "{claude_build.project_version()}"', 'version = "9.9.9"',
        )
        rig.commit()
        rig.refused("says '9.9.9'")


class TestBudgetRefusals:
    def test_over_the_byte_budget(self, rig):
        """(rig) A real oversized file rather than a patched limit, so the refusal is
        the one a stray artifact would get. ``assets/icons/**`` is an allowlist glob."""
        _files, max_bytes = payload.DISTRIBUTION_BUDGETS["claude-code"]
        target = rig.write(f"{PAYLOAD}/assets/icons/huge.bin", "")
        target.write_bytes(b"\0" * (max_bytes + 1))
        rig.commit()
        message = rig.refused("the claude-code distribution is over budget")
        assert f"unpacked bytes (budget {max_bytes})" in message

    def test_over_the_file_budget(self, rig):
        """(rig) An accidental test tree is many small files as often as one big one."""
        max_files, _bytes = payload.DISTRIBUTION_BUDGETS["claude-code"]
        have = len(claude_build.shipped_files(rig.out))
        for index in range(max_files + 1 - have):
            rig.write(f"{PAYLOAD}/assets/icons/pad-{index}.svg", "<svg/>\n")
        rig.commit()
        message = rig.refused("the claude-code distribution is over budget")
        assert f"{max_files + 1} files (budget {max_files})" in message


class TestDirtyInputRefusals:
    @pytest.mark.parametrize("relative", [
        *claude_build.PACKAGING_INPUTS,
        f"{PAYLOAD}/multi_llm.py",
        f"{PAYLOAD}/legacy/README.md",
        payload.MANIFEST_RELATIVE,
    ])
    def test_a_dirty_input_without_allow_dirty(self, rig, relative):
        """(rig) Every packaging input is outside ``stage``'s own prefix or inside it,
        and neither side may ship uncommitted state from a clean build."""
        rig.append(relative, "\n")
        rig.refused("refusing to stage from a dirty tree")

    def test_an_untracked_file_without_allow_dirty(self, rig):
        """(rig) A clean build has no notion of "harmless" scratch under the payload."""
        rig.write(f"{PAYLOAD}/notes.tmp", "scratch\n")
        rig.refused("refusing to stage from a dirty tree")

    def test_allow_dirty_takes_a_modified_input(self, rig):
        """(rig) The counter-case: ``make claude-regen`` runs before the commit, so its
        output has to carry the edit."""
        rig.append("hooks/doctor.sh", "\n# edited\n")
        rig.build(allow_dirty=True)
        assert (rig.out / "hooks" / "doctor.sh").read_text(encoding="utf-8").endswith(
            "# edited\n"
        )

    @pytest.mark.parametrize("relative", [
        f"{PAYLOAD}/utils/new_module.py",
        f"{PAYLOAD}/references/new.md",
        f"{PAYLOAD}/legacy/instructions/new-mode.md",
        f"{PAYLOAD}/legacy/scratch.txt",
    ])
    def test_an_untracked_admitted_input_with_allow_dirty(self, rig, relative):
        """(rig) Staging reads ``git ls-files``, so the file would be silently absent
        and a clean build of the commit could not reproduce the tree once it is added.
        ``legacy/`` is refused whole, since an unlisted file there is an inventory fault
        whichever way it arrives."""
        rig.write(relative, "new\n")
        message = rig.refused("untracked input", allow_dirty=True)
        assert relative in message

    def test_an_untracked_plugin_input_with_allow_dirty(self, rig):
        """(rig) The same rule for an input outside ``skills/multi-llm``."""
        _git(rig.repo, "rm", "-q", "--cached", "hooks/wakeup-check.sh")
        _git(rig.repo, "commit", "-q", "--no-verify", "-m", "untrack a hook")
        message = rig.refused("untracked input", allow_dirty=True)
        assert "hooks/wakeup-check.sh" in message

    def test_scratch_the_allowlist_does_not_admit_is_tolerated(self, rig):
        """(rig) The counter-case: ``--allow-dirty`` is for working trees, which hold
        scratch and by-products, and neither reaches the plugin."""
        rig.write(f"{PAYLOAD}/notes.tmp", "scratch\n")
        rig.write(f"{PAYLOAD}/.venv/pyvenv.cfg", "x\n")
        rig.write(f"{PAYLOAD}/utils/__pycache__/x.pyc", "x\n")
        rig.build(allow_dirty=True)
        held = set(claude_build.shipped_files(rig.out))
        assert f"{PAYLOAD}/notes.tmp" not in held
        assert not any("__pycache__" in path or ".venv" in path for path in _snapshot(rig.out))


class TestMissingInputRefusals:
    @pytest.mark.parametrize("name", [
        "SKILL.md", "providers.yaml", "utils/landlock_exec.py",
    ])
    def test_a_missing_required_resource(self, rig, name):
        """(rig) A short payload looks exactly like a successful build without this."""
        rig.remove(f"{PAYLOAD}/{name}")
        message = rig.refused("the staged payload is missing resources")
        assert name in message

    def test_a_missing_mode_target(self, rig):
        """(rig) A dispatch row naming a file the tree lost fails the build, not the
        user's first run of that mode."""
        rig.remove(f"{PAYLOAD}/utils/doctor.py")
        rig.refused("modes.py names target_module")

    def test_a_tracked_payload_file_gone_from_the_working_tree(self, rig):
        """(rig) Under ``--allow-dirty`` this is a named refusal, not a traceback."""
        (rig.skill / "providers.yaml").unlink()
        message = rig.refused("tracked but missing from the working tree", allow_dirty=True)
        assert "providers.yaml" in message

    @pytest.mark.parametrize("relative", sorted(claude_build.PLUGIN_FILES.values()))
    def test_a_plugin_input_gone_from_the_working_tree(self, rig, relative):
        """(rig) Each file the plugin carries outside the payload."""
        (rig.repo / relative).unlink()
        message = rig.refused("a tracked plugin input is missing", allow_dirty=True)
        assert relative in message


class TestSymlinkRefusals:
    def test_a_tracked_symlink_in_the_payload(self, rig):
        """(rig) An install copy drops it and leaves the parent empty at exit 0."""
        rig.track_symlink(f"{PAYLOAD}/utils/link.py", "doctor.py")
        message = rig.refused("payload containing symlinks", allow_dirty=True)
        assert "utils/link.py" in message

    def test_a_tracked_symlink_in_legacy(self, rig):
        """(rig) ``stage`` reads the whole prefix, so the Claude-only files are covered
        even though the shared allowlist never names them."""
        rig.remove(f"{PAYLOAD}/legacy/references/uv-check.md")
        rig.track_symlink(f"{PAYLOAD}/legacy/references/uv-check.md", "../../references/uv-check.md")
        rig.refused("the legacy inventory|payload containing symlinks", allow_dirty=True)

    @pytest.mark.parametrize("relative", sorted(claude_build.PLUGIN_FILES.values()))
    def test_a_tracked_symlink_among_the_plugin_files(self, rig, relative):
        """(rig) ``stage`` never sees these, and the copy follows a link, so without its
        own check the tree would ship whatever the link points at."""
        rig.track_symlink(relative, "../elsewhere")
        message = rig.refused("symlinked inputs", allow_dirty=True)
        assert relative in message

    @posix_shell
    def test_a_committed_symlink_reaches_the_clean_build_s_refusal(self, rig):
        """(rig) The real thing on disk, in a clean tree, naming a file outside the
        repository: the case that would ship a stranger's bytes."""
        outside = rig.base / "outside.txt"
        outside.write_text("not this repository's licence\n", encoding="utf-8")
        (rig.repo / "LICENSE").unlink()
        (rig.repo / "LICENSE").symlink_to(outside)
        rig.commit()
        message = rig.refused("symlinked inputs")
        assert "LICENSE" in message

    @posix_shell
    def test_a_plugin_input_turned_into_a_symlink_on_disk(self, rig):
        """(rig) The index still says "regular file", so only the disk can tell."""
        (rig.repo / "hooks" / "hooks.json").unlink()
        (rig.repo / "hooks" / "hooks.json").symlink_to(rig.skill / "SKILL.md")
        message = rig.refused("symlinked inputs", allow_dirty=True)
        assert "hooks/hooks.json" in message

    @posix_shell
    def test_a_plugin_input_reached_through_a_symlinked_directory(self, rig):
        """(rig) Every file under ``hooks/`` is a regular file in the index and at its
        own path on disk, so only a check of the whole path catches the directory."""
        outside = rig.base / "outside-hooks"
        shutil.move(str(rig.repo / "hooks"), str(outside))
        (rig.repo / "hooks").symlink_to(outside)
        (outside / "doctor.sh").write_text("#!/bin/sh\n# not this repository\n", encoding="utf-8")
        message = rig.refused("symlinked inputs", allow_dirty=True)
        assert "hooks/doctor.sh" in message

    @posix_shell
    def test_a_payload_file_reached_through_a_symlinked_directory(self, rig):
        """(rig) The same for the shared staging path: a directory in the payload."""
        outside = rig.base / "outside-references"
        shutil.move(str(rig.skill / "references"), str(outside))
        (rig.skill / "references").symlink_to(outside)
        message = rig.refused("a symlink on disk at", allow_dirty=True)
        assert f"{PAYLOAD}/references" in message

    @posix_shell
    def test_a_payload_reached_through_a_symlinked_skill_directory(self, rig):
        """(rig) The whole skill swapped for a link, the highest component on the path."""
        outside = rig.base / "outside-skill"
        shutil.move(str(rig.skill), str(outside))
        rig.skill.symlink_to(outside)
        rig.refused("a symlink on disk at", allow_dirty=True)

    @posix_shell
    def test_a_payload_file_turned_into_a_symlink_on_disk(self, rig):
        """(rig) The same for the payload, under ``--allow-dirty``."""
        (rig.skill / "SKILL.md").unlink()
        (rig.skill / "SKILL.md").symlink_to(rig.skill / "providers.yaml")
        rig.refused("a regular file in the index and a symlink on disk", allow_dirty=True)


def _edit_json(rig: Rig, relative: str, change) -> None:
    path = rig.repo / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    rig.write(relative, json.dumps(document, indent=2) + "\n")


def _set_first_command(command: str):
    def change(document: Dict[str, Any]) -> None:
        document["hooks"]["Stop"][0]["hooks"][0]["command"] = command
    return change


class TestComponentRefusals:
    @pytest.mark.parametrize("command, match", [
        ("/usr/local/bin/report-check.sh", r"does not resolve from \$\{CLAUDE_PLUGIN_ROOT\}"),
        ('"${HOME}/report-check.sh"', r"does not resolve from \$\{CLAUDE_PLUGIN_ROOT\}"),
        ('"${CLAUDE_PLUGIN_ROOT}/hooks/absent.sh"', "does not ship: hooks/absent.sh"),
        ('"${CLAUDE_PLUGIN_ROOT}/../outside.sh"', r"does not ship: \.\./outside\.sh"),
        ('"${CLAUDE_PLUGIN_ROOT}/hooks"', "does not ship: hooks"),
    ])
    def test_a_hook_command_that_leaves_the_plugin(self, rig, command, match):
        """(rig) An install copies this without complaint and a session then fails."""
        _edit_json(rig, "hooks/hooks.json", _set_first_command(command))
        rig.commit()
        rig.refused(match)

    def test_a_hook_command_reaching_a_file_beside_the_tree(self, tmp_path):
        """(tree) ``../x`` resolves here when a file happens to be there, and to nothing
        in an install, so the disk cannot be what answers."""
        tree = tmp_path / "multi-llm"
        _copy_shipped(PLUGIN_DIR, tree)
        (tmp_path / "outside.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        path = tree / "hooks" / "hooks.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        _set_first_command('"${CLAUDE_PLUGIN_ROOT}/../outside.sh"')(document)
        path.write_text(json.dumps(document), encoding="utf-8")
        problems = claude_build.component_problems(tree, claude_build.project_version())
        assert any("../outside.sh" in line for line in problems), problems

    @pytest.mark.parametrize("change, match", [
        (lambda m: m.update(name="other"), "names 'other'"),
        (lambda m: m.update(hooks="./hooks/other.json"), "points hooks at"),
        (lambda m: m.pop("hooks"), "points hooks at"),
    ])
    def test_a_bad_manifest(self, rig, change, match):
        """(rig) The fields the install resolves against the directory it copied."""
        _edit_json(rig, ".claude-plugin/plugin.json", change)
        rig.commit()
        rig.refused(match)

    @pytest.mark.parametrize("relative, match", [
        (".claude-plugin/plugin.json", "plugin.json does not parse"),
        ("hooks/hooks.json", "hooks/hooks.json does not parse"),
    ])
    def test_an_unparseable_configuration(self, rig, relative, match):
        """(rig) The host reads both at install and at session start."""
        rig.write(relative, "{not json\n")
        rig.commit()
        rig.refused(match)

    def test_a_configuration_with_no_command_hook(self, rig):
        """(rig) A tree that registers nothing has quietly lost all three hooks."""
        rig.write("hooks/hooks.json", '{"hooks": {}}\n')
        rig.commit()
        rig.refused("registers no command hook")

    @posix_shell
    def test_a_hook_script_that_lost_its_bit(self, rig):
        """(rig) From the index, as the copy into the tree sets it."""
        _git(rig.repo, "update-index", "--chmod=-x", "hooks/doctor.sh")
        os.chmod(rig.repo / "hooks" / "doctor.sh", 0o644)
        rig.commit()
        rig.refused("hooks/doctor.sh is not executable")

    def test_a_catalog_naming_another_source(self, rig):
        """(rig) The entry that decides what an install copies."""
        rig.replace(
            ".claude-plugin/marketplace.json", f'"./{PLUGIN_RELATIVE}"', '"./"',
        )
        rig.commit()
        rig.refused(f"must have source './{PLUGIN_RELATIVE}'")


class TestTheSwap:
    def test_a_failed_rename_puts_the_previous_tree_back(self, rig):
        """(rig) The last step can fail too: the old tree is moved aside first, so it
        has to come back. A scratch path that does not exist makes the rename fail
        without patching anything."""
        before = _snapshot(rig.out)
        with pytest.raises(OSError):
            payload.swap_tree(rig.base / "never-created", rig.out)
        assert _differences(before, _snapshot(rig.out)) == []

    def test_a_first_build_needs_no_previous_tree(self, rig):
        """(rig) A fresh clone has no destination to retire."""
        fresh = rig.base / "fresh" / "multi-llm"
        rig.build(destination=fresh)
        assert claude_build.digest_of(fresh) == claude_build.digest_of(rig.out)


class TestCommittedDrift:
    def test_by_products_are_not_drift_and_an_edit_is(self, rig):
        """(rig) ``make claude-package`` fails on drift. A developer's ``.venv`` must not
        trip it, and a hand edit in the generated tree must."""
        rig.build(destination=rig.repo / PLUGIN_RELATIVE)
        rig.commit("generated tree")
        assert claude_build.committed_drift(rig.repo) == ()

        by_product = rig.repo / PLUGIN_RELATIVE / PAYLOAD / ".venv" / "pyvenv.cfg"
        by_product.parent.mkdir(parents=True)
        by_product.write_text("x\n", encoding="utf-8")
        assert claude_build.committed_drift(rig.repo) == ()

        rig.append(f"{PLUGIN_RELATIVE}/LICENSE", "\nedited\n")
        drift = claude_build.committed_drift(rig.repo)
        assert len(drift) == 1 and drift[0].endswith("LICENSE")


# ===========================================================================
# The shared-API boundary
# ===========================================================================

class TestTheSharedBoundary:
    def test_extra_allowlist_admits_for_that_call_only(self, rig, tmp_path):
        """(rig) Another host's build straight after a Claude one must see nothing of
        it, and the parsed allowlist must not have grown."""
        allowlist = payload.runtime_allowlist(rig.repo)

        def staged(name: str, **kwargs: Any) -> Tuple[str, ...]:
            return payload.stage(rig.repo, tmp_path / name / "multi-llm", **kwargs)

        legacy = tuple(f"legacy/{name}" for name in claude_build.LEGACY_FILES)
        first = staged("first", extra_allowlist=claude_build.LEGACY_ALLOWLIST)
        assert set(legacy) <= set(first)
        second = staged("second")
        assert not [path for path in second if path.startswith("legacy/")]
        assert not (tmp_path / "second" / "multi-llm" / "legacy").exists()
        assert set(first) - set(second) == set(legacy)
        third = staged(
            "third", extra_allowlist=(f"{PAYLOAD}/legacy/README.md",),
        )
        assert set(third) - set(second) == {"legacy/README.md"}
        assert payload.runtime_allowlist(rig.repo) == allowlist

    def test_extra_allowlist_admits_and_does_not_require(self, rig, tmp_path):
        """(rig) A caller that needs the file present says so through *required_extra*,
        which is why the Claude build passes both."""
        rig.remove(f"{PAYLOAD}/legacy/README.md")
        staged = payload.stage(
            rig.repo, tmp_path / "a" / "multi-llm",
            extra_allowlist=claude_build.LEGACY_ALLOWLIST,
        )
        assert "legacy/README.md" not in staged
        with pytest.raises(payload.PayloadError, match="legacy/README.md"):
            payload.stage(
                rig.repo, tmp_path / "b" / "multi-llm",
                extra_allowlist=claude_build.LEGACY_ALLOWLIST,
                required_extra=claude_build.LEGACY_REQUIRED,
            )

    def test_required_extra_does_not_admit(self, rig, tmp_path):
        """(rig) The other half: requiring a file is not admitting it."""
        with pytest.raises(payload.PayloadError, match="legacy/README.md"):
            payload.stage(
                rig.repo, tmp_path / "a" / "multi-llm",
                required_extra=claude_build.LEGACY_REQUIRED,
            )

    def test_expected_payload_defaults_to_no_legacy(self):
        """(src) ``scan_tarball`` grades the npm artifact against this default."""
        default = payload.expected_payload(REPO_ROOT)
        assert not [path for path in default if path.startswith("legacy/")]
        extended = payload.expected_payload(
            REPO_ROOT, extra_allowlist=claude_build.LEGACY_ALLOWLIST,
        )
        assert set(extended) - set(default) == {
            f"legacy/{name}" for name in claude_build.LEGACY_FILES
        }

    def test_the_manifest_s_allowlist_names_no_legacy(self):
        """(src) The container image and every other host read this same list."""
        assert not [p for p in payload.runtime_allowlist(REPO_ROOT) if "legacy" in p]

    def test_the_codex_tree_carries_no_legacy(self):
        """(tree) The legacy core names Claude Code's tools, which Codex does not have."""
        tracked = _tracked_below(CODEX_RELATIVE)
        assert tracked, "the Codex tree is tracked"
        assert not [path for path in tracked if "legacy" in path.split("/")]
        assert not (REPO_ROOT / CODEX_RELATIVE / PAYLOAD / "legacy").exists()

    def test_the_opencode_staging_path_carries_no_legacy(self, opencode_staged):
        """(build) The same for the npm route, from its real builder."""
        assert (opencode_staged / PAYLOAD / "SKILL.md").is_file()
        assert not (opencode_staged / PAYLOAD / "legacy").exists()
        assert not [
            path for path in payload.file_inventory(opencode_staged)
            if "legacy" in path.split("/")
        ]


# ===========================================================================
# Exclusions over the final distributions
# ===========================================================================

#: A path segment that marks a whole directory as not for shipping.
_EXCLUDED_SEGMENTS = frozenset({
    "tests", "test", "fixtures", "golden", "goldens", "mocks", "plans", "docs",
    "coding_planning", "scratch", "tmp", ".git", ".github", ".multi-llm", ".gate",
    "__pycache__", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "node_modules", "legacy",
})

#: A file name that is development material wherever it appears.
_EXCLUDED_NAMES = frozenset({
    "AGENTS.md", "CLAUDE.md", "TODO.md", "SECURITY.md", "CONTRIBUTING.md", "Makefile",
    "conftest.py", "pytest.ini", "tox.ini", ".gitignore", ".gitattributes",
    ".dockerignore", ".gitleaks.toml", ".package-check.stamp", "package-lock.json",
    ".DS_Store",
})

_EXCLUDED_GLOBS = (
    "test_*.py", "*_test.py", "*.pyc", "*.pyo", "*.orig", "*.rej", "*.bak", "*.tmp",
    "*.log", "*.golden", "*.tgz", "*.swp", "*~", "README*",
)

#: The named exceptions, and the only ones: each is a path, never a pattern, so a new
#: file under ``legacy/`` is not excused by this list.
CLAUDE_EXCEPTIONS = frozenset(f"{PAYLOAD}/legacy/{name}" for name in claude_build.LEGACY_FILES)
CODEX_EXCEPTIONS: frozenset = frozenset()
#: npm packs the root README whatever ``files`` says, and it is the package's page.
OPENCODE_EXCEPTIONS = frozenset({"README.md"})


def _exclusion_problems(paths: Iterable[str], exceptions: Iterable[str]) -> List[str]:
    allowed = set(exceptions)
    problems = []
    for relative in sorted(paths):
        if relative in allowed:
            continue
        parts = relative.split("/")
        name = parts[-1]
        if (
            _EXCLUDED_SEGMENTS.intersection(parts[:-1])
            or name in _EXCLUDED_NAMES
            or any(fnmatch.fnmatchcase(name, glob) for glob in _EXCLUDED_GLOBS)
        ):
            problems.append(relative)
    return problems


@pytest.fixture(scope="session")
def packed_npm(tmp_path_factory) -> Tuple[Path, Path]:
    """The actual ``npm pack`` artifact, and its unpacked ``package/`` directory.

    ``make opencode-package`` builds, then packs inside the package directory. This
    does the same in a temp directory so a row leaves nothing in the checkout, which
    is why it passes ``--ignore-scripts``: ``prepack`` only re-runs the build above,
    through a relative path that does not resolve from here.
    """
    if not shutil.which("npm"):
        _unavailable("npm is not on PATH; this row needs `npm pack`")
    if sys.platform.startswith("win"):
        _unavailable("the package job is Linux-only by decision")
    staged = tmp_path_factory.mktemp("opencode-pack-input")
    opencode_build.build(destination=staged, allow_dirty=True)
    destination = tmp_path_factory.mktemp("opencode-pack-output")
    packed = subprocess.run(
        ["npm", "pack", "--ignore-scripts", "--pack-destination", str(destination)],
        cwd=str(staged), capture_output=True, text=True, timeout=600,
    )
    assert packed.returncode == 0, packed.stdout + packed.stderr
    archives = list(destination.glob("*.tgz"))
    assert len(archives) == 1, f"expected one tarball, got {archives}"
    payload.scan_tarball(archives[0])
    unpacked = tmp_path_factory.mktemp("opencode-unpacked")
    with tarfile.open(archives[0], "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        if hasattr(tarfile, "data_filter"):
            handle.extractall(unpacked, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            handle.extractall(unpacked, members=members)
    return archives[0], unpacked / "package"


class TestExclusions:
    @pytest.mark.parametrize("relative", [
        f"{PAYLOAD}/tests/test_x.py",
        f"{PAYLOAD}/utils/test_helpers.py",
        f"{PAYLOAD}/tests/fixtures/a.json",
        f"{PAYLOAD}/fixtures/a.json",
        f"{PAYLOAD}/golden/a.txt",
        f"{PAYLOAD}/plans/my-plan.md",
        f"{PAYLOAD}/docs/x.md",
        "docs/providers.md",
        f"{PAYLOAD}/AGENTS.md",
        "CLAUDE.md",
        f"{PAYLOAD}/README.md",
        f"{PAYLOAD}/.git/config",
        f"{PAYLOAD}/.gitignore",
        f"{PAYLOAD}/.venv/pyvenv.cfg",
        f"{PAYLOAD}/utils/__pycache__/a.pyc",
        f"{PAYLOAD}/.pytest_cache/CACHEDIR.TAG",
        f"{PAYLOAD}/.ruff_cache/x",
        f"{PAYLOAD}/.multi-llm/providers.yaml",
        f"{PAYLOAD}/scratch/out.txt",
        f"{PAYLOAD}/notes.tmp",
        f"{PAYLOAD}/run.log",
        f"{PAYLOAD}/legacy/README.md",
    ])
    def test_the_checker_flags_each_class(self, relative):
        """(src) Each class is caught, so the rows below are not vacuous."""
        assert _exclusion_problems([relative], CODEX_EXCEPTIONS) == [relative]

    def test_an_exception_is_a_path_and_not_a_pattern(self):
        """(src) The Claude exceptions excuse the named files and nothing else there."""
        extra = f"{PAYLOAD}/legacy/instructions/new-mode.md"
        assert _exclusion_problems(CLAUDE_EXCEPTIONS, CLAUDE_EXCEPTIONS) == []
        assert _exclusion_problems([extra], CLAUDE_EXCEPTIONS) == [extra]
        nested = f"{PAYLOAD}/legacy/README.md/x"
        assert _exclusion_problems([nested], CLAUDE_EXCEPTIONS) == [nested]

    def test_the_claude_tree(self):
        """(tree) What the Git route serves, so a by-product on disk is not at issue."""
        assert _exclusion_problems(
            _tracked_below(PLUGIN_RELATIVE), CLAUDE_EXCEPTIONS,
        ) == []

    def test_the_rebuilt_claude_tree(self, built_plugin):
        """(build) The same rule over what a build writes."""
        assert _exclusion_problems(
            claude_build.shipped_files(built_plugin), CLAUDE_EXCEPTIONS,
        ) == []

    def test_every_claude_exception_is_still_in_use(self):
        """(tree) An exception nothing uses is one a later file could slip through."""
        assert CLAUDE_EXCEPTIONS <= _tracked_below(PLUGIN_RELATIVE)

    def test_the_codex_tree(self):
        """(tree) Codex has no exception at all, ``legacy/`` included."""
        assert _exclusion_problems(
            _tracked_below(CODEX_RELATIVE), CODEX_EXCEPTIONS,
        ) == []

    @pytest.mark.package
    def test_the_npm_artifact(self, packed_npm):
        """(pack) What npm wrote, not what the staging directory held. ``README.md`` is
        the package's own page and the only exception."""
        _archive, package = packed_npm
        files = payload.file_inventory(package)
        assert _exclusion_problems(files, OPENCODE_EXCEPTIONS) == []
        assert "README.md" in files

    @pytest.mark.package
    def test_the_npm_artifact_holds_only_what_files_names(self, packed_npm):
        """(pack) The build scripts and sources sit beside the staged tree on disk, and
        only the package's ``files`` key keeps them out."""
        _archive, package = packed_npm
        assert {path.name for path in package.iterdir()} == {
            "dist", "skills", "README.md", "LICENSE", "package.json",
        }


# ===========================================================================
# Budgets
# ===========================================================================

def _measure(tree: Path) -> Tuple[int, int]:
    files = claude_build.shipped_files(tree)
    return len(files), sum(files.values())


class TestBudgets:
    def test_every_distribution_has_a_budget(self):
        """(src) A distribution with no budget has no limit at all."""
        assert set(payload.DISTRIBUTION_BUDGETS) == {"claude-code", "codex", "opencode"}

    def test_claude_stays_at_or_below_its_review_target(self):
        """(src) Raising this past 12,000,000 is a decision, not a side effect."""
        assert payload.DISTRIBUTION_BUDGETS["claude-code"][1] <= 12_000_000

    @pytest.mark.parametrize("distribution", sorted(payload.DISTRIBUTION_BUDGETS))
    def test_a_failing_budget_names_the_distribution_and_the_totals(self, distribution):
        """(src) "Too big" is no use on its own: the reader needs which and by how much."""
        max_files, max_bytes = payload.DISTRIBUTION_BUDGETS[distribution]
        assert payload.budget_problem(distribution, max_files, max_bytes) is None
        by_files = payload.budget_problem(distribution, max_files + 1, max_bytes)
        assert distribution in by_files
        assert f"{max_files + 1} files (budget {max_files})" in by_files
        by_bytes = payload.budget_problem(distribution, max_files, max_bytes + 1)
        assert distribution in by_bytes
        assert f"{max_bytes + 1} unpacked bytes (budget {max_bytes})" in by_bytes
        with pytest.raises(payload.PayloadError, match=distribution):
            payload.check_budget(distribution, max_files + 1, max_bytes + 1)

    def test_an_unknown_distribution_has_no_quiet_pass(self):
        """(src) A misspelled name must not read as under budget."""
        assert "no budget" in payload.budget_problem("claude", 1, 1)

    def test_the_claude_tree_is_within_budget(self):
        """(tree) The tree a Claude Code install copies."""
        files, size = _measure(PLUGIN_DIR)
        assert payload.budget_problem("claude-code", files, size) is None

    def test_the_codex_tree_is_within_budget(self):
        """(tree) The tree a Codex install copies."""
        files, size = _measure(REPO_ROOT / CODEX_RELATIVE)
        assert payload.budget_problem("codex", files, size) is None

    @pytest.mark.package
    def test_the_npm_artifact_is_within_budget(self, packed_npm):
        """(pack) The tarball's unpacked contents, never a staging directory holding
        build scripts, which would measure something nobody installs."""
        _archive, package = packed_npm
        files = payload.file_inventory(package)
        assert payload.budget_problem("opencode", len(files), sum(files.values())) is None


# ===========================================================================
# The staged hooks, against stubs
# ===========================================================================

def _stub_uv(directory: Path, *, execute: bool) -> Path:
    """A ``uv`` that logs its argv, and optionally runs the rest on this test's Python
    so the staged ``multi_llm.py`` itself answers."""
    directory.mkdir(parents=True, exist_ok=True)
    stub = directory / "uv"
    body = "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARGV_LOG\"\n"
    if execute:
        body += (
            "while [ \"$#\" -gt 0 ] && [ \"$1\" != \"--\" ]; do shift; done\n"
            "shift\n"
            "shift\n"
            "exec \"$REAL_PYTHON\" \"$@\"\n"
        )
    stub.write_text(body, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub


def _hook_env(tmp_path: Path, install: Optional[Path], uv: Optional[Path]) -> Dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "ARGV_LOG": str(tmp_path / "argv.txt"),
        "REAL_PYTHON": sys.executable,
    }
    if install is not None:
        environment["CLAUDE_PLUGIN_ROOT"] = str(install)
    if uv is not None:
        environment["MULTI_LLM_UV"] = str(uv)
    return environment


def _run_hook(
    tmp_path: Path, install: Path, script: str, text: str, uv: Optional[Path],
) -> Tuple[subprocess.CompletedProcess, Path]:
    completed = subprocess.run(
        ["/bin/sh", str(install / "hooks" / script)],
        input=text, env=_hook_env(tmp_path, install, uv),
        capture_output=True, encoding="utf-8", timeout=60,
    )
    return completed, tmp_path / "argv.txt"


def _stop_input(path: Path, *, active: bool = False) -> str:
    return json.dumps({
        "session_id": "s1", "transcript_path": str(path), "cwd": str(path.parent),
        "hook_event_name": "Stop", "stop_hook_active": active,
    })


def _dispatcher_argv(skill: Path, *words: str) -> List[str]:
    return [
        "run", *unattended.DEFAULT_UV_RUN_ARGS, "--project", str(skill), "--",
        "python", f"{skill}/multi_llm.py", *words,
    ]


@posix_shell
class TestTheStagedDoctorHook:
    def test_it_runs_the_doctor_through_the_resolved_uv(self, staged_install, tmp_path):
        """(tree) The argv keeps the spaced path as one word each time."""
        skill = staged_install / PAYLOAD
        completed, log = _run_hook(
            tmp_path, staged_install, "doctor.sh", "", _stub_uv(tmp_path / "bin", execute=False),
        )
        assert completed.returncode == 0, completed.stderr
        assert log.read_text(encoding="utf-8").splitlines() == _dispatcher_argv(
            skill, "doctor", "--warn-only",
        )

    def test_it_is_silent_and_exits_0_without_uv(self, staged_install, tmp_path):
        """(tree) A SessionStart hook that exited 2 would block the session."""
        completed, _log = _run_hook(tmp_path, staged_install, "doctor.sh", "", None)
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""

    def test_it_is_silent_and_exits_0_outside_a_plugin_tree(self, tmp_path, staged_install):
        """(tree) A half-removed install must not disturb the session either."""
        environment = _hook_env(tmp_path, tmp_path / "nowhere", None)
        completed = subprocess.run(
            ["/bin/sh", str(staged_install / "hooks" / "doctor.sh")],
            env=environment, capture_output=True, encoding="utf-8", timeout=60,
        )
        assert completed.returncode == 0 and completed.stdout == ""


@posix_shell
class TestTheStagedReportCheckHook:
    def test_a_miss_prints_exactly_one_json_object(self, staged_install, tmp_path):
        """(tree) The whole of stdout is the decision: Claude Code parses it as JSON only
        when it is nothing else."""
        skill = staged_install / PAYLOAD
        path = Transcript().prompt("go").bash(block_output()).say("Done.").write(
            tmp_path / "my session.jsonl",
        )
        completed, log = _run_hook(
            tmp_path, staged_install, "report-check.sh", _stop_input(path),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0, completed.stderr
        decision = json.loads(completed.stdout)
        assert decision["decision"] == "block"
        assert COMMAND in decision["reason"]
        assert cli_common.RESULT_MARKER not in completed.stdout
        assert log.read_text(encoding="utf-8").splitlines() == _dispatcher_argv(
            skill, "util", "report-check", "--hook-input", "-",
        )

    def test_a_copied_command_prints_nothing(self, staged_install, tmp_path):
        """(tree) The counter-case, with the check really having run."""
        path = Transcript().prompt("go").bash(block_output()).say(report(COMMAND)).write(
            tmp_path / "t.jsonl",
        )
        completed, log = _run_hook(
            tmp_path, staged_install, "report-check.sh", _stop_input(path),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == ""
        assert log.exists(), "the check itself must have run"

    def test_stop_hook_active_exits_before_uv(self, staged_install, tmp_path):
        """(tree) It blocks at most once per turn, so it cannot loop."""
        path = Transcript().prompt("go").bash(block_output()).say("Done.").write(
            tmp_path / "t.jsonl",
        )
        completed, log = _run_hook(
            tmp_path, staged_install, "report-check.sh", _stop_input(path, active=True),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_transcript_without_the_marker_exits_before_uv(self, staged_install, tmp_path):
        """(tree) A session that never ran the dispatcher pays two greps and no uv."""
        path = Transcript().prompt("hello").say("Hi.").write(tmp_path / "t.jsonl")
        completed, log = _run_hook(
            tmp_path, staged_install, "report-check.sh", _stop_input(path),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0 and completed.stdout == ""
        assert not log.exists()

    @pytest.mark.parametrize("text", [
        "", "not json", '{"stop_hook_active": false}',
        '{"transcript_path": "/nonexistent/t.jsonl", "stop_hook_active": false}',
    ])
    def test_no_transcript_exits_0_silently(self, staged_install, tmp_path, text):
        """(tree) Every error is a silent pass."""
        completed, log = _run_hook(
            tmp_path, staged_install, "report-check.sh", text,
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_transcript_file_written_after_the_hook_started(self, staged_install, tmp_path):
        """(tree) In a session that young the whole file is missing at ``Stop``."""
        path = tmp_path / "t.jsonl"
        transcript = Transcript().prompt("go").bash(block_output()).say("Stopped. Done.")
        write_later(path, "".join(transcript_line(entry) for entry in transcript.entries))
        completed, _log = _run_hook(
            tmp_path, staged_install, "report-check.sh",
            json.dumps(stop_input(path, "Stopped. Done.")),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["decision"] == "block"

    def test_a_dispatcher_call_whose_block_is_not_on_file_yet_goes_on(
        self, staged_install, tmp_path,
    ):
        """(tree) Claude Code writes the transcript in batches, so the call can be on
        file while its block is not."""
        path = tmp_path / "t.jsonl"
        path.write_text(
            transcript_line(Transcript().prompt("go").entries[0])
            + transcript_line(tool_use("toolu_1", LOOKUP, "msg_1")),
            encoding="utf-8",
        )
        write_later(path, transcript_line(tool_result("toolu_1", block_output())) + transcript_line(reply("Done.")))
        completed, _log = _run_hook(
            tmp_path, staged_install, "report-check.sh",
            json.dumps(stop_input(path, "Done.")),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["decision"] == "block"

    def test_a_bash_call_without_the_dispatcher_exits_before_uv(self, staged_install, tmp_path):
        """(tree) An ordinary session's tool calls cost it no uv start."""
        path = tmp_path / "t.jsonl"
        path.write_text(
            transcript_line(Transcript().prompt("list the files").entries[0])
            + transcript_line(tool_use("toolu_1", "ls", "msg_1"))
            + transcript_line(tool_result("toolu_1", "a.txt\n"))
            + transcript_line(reply("One file.")),
            encoding="utf-8",
        )
        completed, log = _run_hook(
            tmp_path, staged_install, "report-check.sh",
            json.dumps(stop_input(path, "One file.")),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_failing_util_exits_0_silently(self, staged_install, tmp_path):
        """(tree) A broken check must not turn into a blocked turn."""
        path = Transcript().prompt("go").bash(block_output()).say("Done.").write(
            tmp_path / "t.jsonl",
        )
        broken = tmp_path / "broken-uv"
        broken.write_text("#!/bin/sh\necho boom >&2\necho 'not json'\nexit 7\n", encoding="utf-8")
        broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
        completed, _log = _run_hook(
            tmp_path, staged_install, "report-check.sh", _stop_input(path), broken,
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""


@posix_shell
class TestTheStagedWakeupHook:
    def test_a_refusal_is_exactly_one_json_object(self, staged_install, tmp_path):
        """(tree) A scheduled wakeup while a phase this session started is running."""
        skill = staged_install / PAYLOAD
        path = racing_03(Transcript()).write(tmp_path / "my session.jsonl")
        completed, log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh", json.dumps(hook_input(path)),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""
        decision = json.loads(completed.stdout)
        assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert log.read_text(encoding="utf-8").splitlines() == _dispatcher_argv(
            skill, "util", "wakeup-check", "--hook-input", "-",
        )

    def test_a_finished_phase_prints_nothing(self, staged_install, tmp_path):
        """(tree) The counter-case, with the check really having run."""
        transcript = racing_03(Transcript())
        phase(transcript, "generate-tasks", block_output(), " --reaggregate")
        path = transcript.write(tmp_path / "t.jsonl")
        completed, log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh", json.dumps(hook_input(path)),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0 and completed.stdout == ""
        assert log.exists(), "the check itself must have run"

    def test_another_tool_exits_before_uv(self, staged_install, tmp_path):
        """(tree) The hook never affects anything but the two scheduling tools."""
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        completed, log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh",
            json.dumps(hook_input(path, tool="Bash")),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_session_that_never_ran_the_dispatcher_exits_before_uv(
        self, staged_install, tmp_path,
    ):
        """(tree) It never touches the user's own wakeups."""
        path = Transcript().prompt("remind me later").say("Sure.").write(tmp_path / "t.jsonl")
        completed, log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh", json.dumps(hook_input(path)),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""
        assert not log.exists()

    def test_a_phase_launched_in_the_same_message_reaches_the_util(
        self, staged_install, tmp_path,
    ):
        """(tree) The launch and the scheduling call arrive in one write, after the hook
        started, so it has to wait for its own call before it greps."""
        path = tmp_path / "t.jsonl"
        path.write_text(transcript_line(Transcript().prompt("go").entries[0]), encoding="utf-8")
        launch = tool_use("toolu_1", BACKGROUND_PHASE, "msg_1", run_in_background=True)
        own = own_call_entry()
        own["message"]["id"] = "msg_1"
        write_later(path, transcript_line(launch) + transcript_line(own) + transcript_line(tool_result("toolu_1", LAUNCHED)))
        completed, _log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh", json.dumps(wakeup_input(path)),
            _stub_uv(tmp_path / "bin", execute=True),
        )
        assert completed.returncode == 0, completed.stderr
        assert "review-tasks phase" in denied(completed.stdout)

    def test_without_uv_it_allows_silently(self, staged_install, tmp_path):
        """(tree) It fails open."""
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        completed, _log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh", json.dumps(hook_input(path)), None,
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""

    def test_a_failing_util_allows_silently(self, staged_install, tmp_path):
        """(tree) It fails open here too."""
        path = racing_03(Transcript()).write(tmp_path / "t.jsonl")
        broken = tmp_path / "broken-uv"
        broken.write_text("#!/bin/sh\necho boom >&2\necho 'not json'\nexit 7\n", encoding="utf-8")
        broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
        completed, _log = _run_hook(
            tmp_path, staged_install, "wakeup-check.sh", json.dumps(hook_input(path)), broken,
        )
        assert completed.returncode == 0
        assert completed.stdout == "" and completed.stderr == ""


@posix_shell
class TestTheHookCommandsAsConfigured:
    def _commands(self, install: Path) -> Dict[str, str]:
        configuration = json.loads((install / "hooks" / "hooks.json").read_text(encoding="utf-8"))
        return {
            event: groups[0]["hooks"][0]["command"]
            for event, groups in configuration["hooks"].items()
        }

    def test_each_command_runs_from_a_path_with_spaces(self, staged_install, tmp_path):
        """(tree) The literal ``hooks.json`` string through a shell, as Claude Code runs
        it. That needs the quoting to survive the spaces and the script to keep its
        executable bit, which no row calling ``sh <script>`` would notice."""
        skill = staged_install / PAYLOAD
        transcript = Transcript().prompt("go").bash(block_output()).say("Done.").write(
            tmp_path / "stop.jsonl",
        )
        wakeup = racing_03(Transcript()).write(tmp_path / "wakeup.jsonl")
        scenarios = {
            "SessionStart": ("", ["doctor", "--warn-only"]),
            "Stop": (_stop_input(transcript), ["util", "report-check", "--hook-input", "-"]),
            "PreToolUse": (
                json.dumps(hook_input(wakeup)), ["util", "wakeup-check", "--hook-input", "-"],
            ),
        }
        stub = _stub_uv(tmp_path / "bin", execute=False)
        for event, command in self._commands(staged_install).items():
            text, words = scenarios[event]
            log = tmp_path / "argv.txt"
            if log.exists():
                log.unlink()
            completed = subprocess.run(
                ["/bin/sh", "-c", command], input=text,
                env=_hook_env(tmp_path, staged_install, stub),
                capture_output=True, encoding="utf-8", timeout=60,
            )
            assert completed.returncode == 0, f"{event}: {completed.stderr}"
            assert log.read_text(encoding="utf-8").splitlines() == _dispatcher_argv(
                skill, *words,
            ), event

    @pytest.mark.parametrize("script, text_kind, words", [
        ("doctor.sh", "none", ["doctor", "--warn-only"]),
        ("report-check.sh", "stop", ["util", "report-check", "--hook-input", "-"]),
        ("wakeup-check.sh", "wakeup", ["util", "wakeup-check", "--hook-input", "-"]),
    ])
    def test_a_script_finds_uv_in_its_documented_home(
        self, staged_install, tmp_path, script, text_kind, words,
    ):
        """(tree) A hook starts with the host's PATH, which often lacks ``~/.local/bin``."""
        if shutil.which("uv", path="/usr/bin:/bin"):
            pytest.skip("uv is in /usr/bin or /bin here, so PATH answers before the home")
        _stub_uv(tmp_path / ".local" / "bin", execute=False)
        if text_kind == "stop":
            path = Transcript().prompt("go").bash(block_output()).say("Done.").write(
                tmp_path / "t.jsonl",
            )
            text = _stop_input(path)
        elif text_kind == "wakeup":
            text = json.dumps(hook_input(racing_03(Transcript()).write(tmp_path / "t.jsonl")))
        else:
            text = ""
        completed, log = _run_hook(tmp_path, staged_install, script, text, None)
        assert completed.returncode == 0, completed.stderr
        assert log.read_text(encoding="utf-8").splitlines() == _dispatcher_argv(
            staged_install / PAYLOAD, *words,
        )

    def test_a_script_finds_its_own_plugin_without_the_variable(self, staged_install, tmp_path):
        """(tree) The fallback when ``CLAUDE_PLUGIN_ROOT`` is unset, from ``$0``."""
        environment = _hook_env(tmp_path, None, _stub_uv(tmp_path / "bin", execute=False))
        completed = subprocess.run(
            ["/bin/sh", str(staged_install / "hooks" / "doctor.sh")],
            env=environment, capture_output=True, encoding="utf-8", timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines() == (
            _dispatcher_argv(staged_install / PAYLOAD, "doctor", "--warn-only")
        )


# ===========================================================================
# The staged runtime, with the real uv
# ===========================================================================

def _real_uv() -> str:
    if os.name == "nt":
        _unavailable("the package job is Linux-only by decision")
    uv = unattended.find_uv()
    if uv is None:
        _unavailable("uv is not installed")
    return uv


def _uv_cli(
    skill: Path, *words: str, cwd: Path, env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    """The dispatcher, the way every shipped text spells it."""
    return subprocess.run(
        [
            _real_uv(), "run", *unattended.DEFAULT_UV_RUN_ARGS, "--project", str(skill), "--",
            "python", str(skill / "multi_llm.py"), *words,
        ],
        cwd=str(cwd), env=_clean_env(**dict(env or {})),
        capture_output=True, encoding="utf-8", timeout=300,
    )


def _framed(mode_name: str, core: str) -> str:
    """Everything ``util instructions`` prints for *core*, up to the result line."""
    body = core if core.endswith("\n") else core + "\n"
    return (
        multi_llm._truncation_notice(mode_name, core)  # noqa: SLF001
        + body
        + f"{multi_llm.CORE_END_MARKER} {mode_name}\n"
    )


def _result_line(stdout: str, framed: str) -> Dict[str, Any]:
    """The served text is exactly *framed*, then one result line, and nothing else."""
    assert stdout.startswith(framed), "the served core differs from the staged file"
    rest = stdout[len(framed):]
    assert rest.startswith(cli_common.RESULT_MARKER), rest[:200]
    assert rest.count("\n") == 1, rest[:400]
    return json.loads(rest[len(cli_common.RESULT_MARKER):])


@pytest.mark.package
class TestTheStagedRuntime:
    def test_the_flags_are_the_ones_the_shipped_text_spells(self):
        """(src) The hooks and ``SKILL.md`` type them out, so this row and they can only
        agree through the owner."""
        assert unattended.DEFAULT_UV_RUN_ARGS == ("--frozen", "--no-dev")

    def test_the_tree_syncs_the_runtime_set_only(self, staged_install, tmp_path):
        """(tree, uv) ``--frozen --no-dev`` from the install path installs pyyaml and
        nothing from the dev group, and the lock survives the trip."""
        uv = _real_uv()
        result = subprocess.run(
            [uv, "sync", "--frozen", "--no-dev", "--dry-run",
             "--project", str(staged_install / PAYLOAD)],
            env=_clean_env(UV_PROJECT_ENVIRONMENT=str(tmp_path / "venv")),
            capture_output=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        listing = result.stdout + result.stderr
        names = {m.lower() for m in re.findall(r"^\s*\+\s+([A-Za-z0-9_.-]+)==", listing, re.M)}
        assert names == {"pyyaml"}, listing

    def test_cli_help(self, staged_install, tmp_path):
        """(tree, uv) Every mode resolves its summary from the staged ``modes.py``."""
        result = _uv_cli(staged_install / PAYLOAD, "--help", cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage: multi_llm.py <mode>" in result.stdout
        for mode in mode_table.MODES:
            assert mode.name in result.stdout, mode.name

    def test_instructions_render_for_a_normal_mode(self, staged_install, tmp_path):
        """(tree, uv) The current core, with the spaced path substituted everywhere."""
        skill = staged_install / PAYLOAD
        result = _uv_cli(skill, "util", "instructions", "review-plan", cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[LEGACY_INSTRUCTIONS]" not in result.stderr
        core = cli_common.render_instruction_text(
            (skill / "instructions" / "review-plan.md").read_text(encoding="utf-8"), skill,
        )
        line = _result_line(result.stdout, _framed("review-plan", core))
        assert line["artifacts"] == [str(skill / "instructions" / "review-plan.md")]
        assert f'--project "{skill}"' in core
        assert f'"{skill}/multi_llm.py"' in core
        for token in ("{SKILL_DIR}", "${CLAUDE_SKILL_DIR}", "{UV_RUN_FLAGS}"):
            assert token not in core, token

    def test_init_reports_and_writes_an_inert_config(self, staged_install, tmp_path):
        """(tree, uv) With no provider CLI on PATH, ``--init`` writes a config that
        inherits every default, at the git root of the disposable repository."""
        skill = staged_install / PAYLOAD
        _uv_cli(skill, "--help", cwd=tmp_path)  # the venv exists before PATH narrows
        repo, path_dir = self._disposable_repo(tmp_path)
        result = _uv_cli(skill, "init", cwd=repo, env={"PATH": str(path_dir)})
        assert result.returncode == 0, result.stdout + result.stderr
        config = repo / ".multi-llm" / "providers.yaml"
        assert f"Wrote {config}" in result.stdout
        assert "No supported provider CLIs found on PATH" in result.stdout
        text = config.read_text(encoding="utf-8")
        assert text.startswith("# .multi-llm/providers.yaml")
        assert "# <<< multi-llm:init-managed >>>" in text
        document = yaml.safe_load(text)
        assert not document.get("providers") and not document.get("default_provider")
        assert _git(repo, "status", "--porcelain").splitlines() == ["?? .multi-llm/"]

    def test_init_enables_a_detected_provider(self, staged_install, tmp_path):
        """(tree, uv) The same file with one CLI found: its block is uncommented."""
        skill = staged_install / PAYLOAD
        _uv_cli(skill, "--help", cwd=tmp_path)
        repo, path_dir = self._disposable_repo(tmp_path)
        stub = path_dir / "codex"
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
        result = _uv_cli(skill, "init", cwd=repo, env={"PATH": str(path_dir)})
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Detected providers: codex" in result.stdout
        document = yaml.safe_load(
            (repo / ".multi-llm" / "providers.yaml").read_text(encoding="utf-8")
        )
        assert document["default_provider"] == "codex"
        assert list(document["providers"]) == ["codex"]

    def test_init_template_only_writes_the_staged_template(self, staged_install, tmp_path):
        """(tree, uv) The template is resolved from the install, not from the cwd."""
        skill = staged_install / PAYLOAD
        repo, _path_dir = self._disposable_repo(tmp_path)
        result = _uv_cli(skill, "init", "--template-only", cwd=repo)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (repo / ".multi-llm" / "providers.yaml").read_bytes() == (
            skill / "templates" / "config" / "providers.override.yaml"
        ).read_bytes()

    def test_running_changes_no_shipped_file(self, staged_install, tmp_path):
        """(tree, uv) The install stays the tree it was: a by-product is fine, a rewritten
        ``uv.lock`` is a plugin that drifts from its own cached version."""
        skill = staged_install / PAYLOAD
        assert _uv_cli(skill, "--help", cwd=tmp_path).returncode == 0
        assert (skill / ".venv").is_dir(), "the run was expected to build its venv in place"
        assert claude_build.digest_of(staged_install) == claude_build.digest_of(PLUGIN_DIR)

    def test_the_real_hooks_answer_through_the_real_uv(self, staged_install, tmp_path):
        """(tree, uv) No stub at all: each decision comes from the staged dispatcher,
        run by the flags the script spells out, from the path with spaces."""
        skill = staged_install / PAYLOAD
        _uv_cli(skill, "--help", cwd=tmp_path)
        environment = _clean_env(
            PATH=f"{Path(_real_uv()).parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
            CLAUDE_PLUGIN_ROOT=str(staged_install),
        )
        environment.pop("MULTI_LLM_UV", None)

        def hook(script: str, text: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["/bin/sh", str(staged_install / "hooks" / script)],
                input=text, env=environment, capture_output=True,
                encoding="utf-8", timeout=120,
            )

        stop = Transcript().prompt("go").bash(block_output()).say("Done.").write(
            tmp_path / "stop.jsonl",
        )
        answered = hook("report-check.sh", _stop_input(stop))
        assert answered.returncode == 0, answered.stderr
        assert json.loads(answered.stdout)["decision"] == "block"

        wakeup = racing_03(Transcript()).write(tmp_path / "wakeup.jsonl")
        answered = hook("wakeup-check.sh", json.dumps(hook_input(wakeup)))
        assert answered.returncode == 0, answered.stderr
        decision = json.loads(answered.stdout)
        assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"

        assert hook("doctor.sh", "").returncode == 0

    @staticmethod
    def _disposable_repo(tmp_path: Path) -> Tuple[Path, Path]:
        """A git repository, and a PATH holding git alone, so the detection a row sees
        does not depend on which provider CLIs this machine has."""
        repo = tmp_path / "disposable repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
        path_dir = tmp_path / "path"
        path_dir.mkdir()
        (path_dir / "git").symlink_to(shutil.which("git"))
        return repo, path_dir


# ===========================================================================
# The legacy switch
# ===========================================================================

#: ``{mode word: instruction file}`` for every mode with a frozen copy. Mapped through
#: the mode table, because a file name is not always its mode word.
LEGACY_MODES: Dict[str, str] = {
    mode.name: mode.instruction_file for mode in mode_table.MODES
    if mode.instruction_file
    and f"instructions/{mode.instruction_file}" in claude_build.LEGACY_FILES
}

#: Modes with a core file that postdate the frozen copy.
UNFROZEN_MODES: Tuple[str, ...] = tuple(sorted(
    mode.name for mode in mode_table.MODES
    if mode.instruction_file and mode.name not in LEGACY_MODES
))

#: A dispatcher-driven mode, which has no core file at all.
NO_CORE_MODE = "status"


class TestTheLegacyModeMap:
    def test_every_frozen_file_has_its_mode(self):
        """(src) Eleven files and eleven modes, or a row below quietly grades fewer."""
        frozen = {
            name[len("instructions/"):] for name in claude_build.LEGACY_FILES
            if name.startswith("instructions/")
        }
        assert set(LEGACY_MODES.values()) == frozen
        assert len(LEGACY_MODES) == 11

    def test_a_file_name_is_not_always_the_mode_word(self):
        """(src) The two whose word differs are the ones a name match would lose."""
        assert LEGACY_MODES["full"] == "full-workflow.md"
        assert LEGACY_MODES["init"] == "init-config.md"
        for word, name in LEGACY_MODES.items():
            assert mode_table.instruction_path(word) == f"instructions/{name}"

    def test_the_readme_names_the_modes_with_no_frozen_copy(self):
        """(src) It is what the stderr notice sends an operator to read."""
        text = (SKILL_ROOT / "legacy" / "README.md").read_text(encoding="utf-8")
        assert UNFROZEN_MODES
        for word in UNFROZEN_MODES:
            assert f"`{word}`" in text, word


@pytest.mark.package
class TestTheLegacySwitch:
    @pytest.mark.parametrize("word", sorted(LEGACY_MODES))
    def test_the_switch_serves_the_frozen_core(self, staged_install, tmp_path, word):
        """(tree, uv) The whole of stdout is the staged frozen file after the normal
        rendering, inside the CLI's framing. Success or non-empty output cannot pass
        this, and neither can serving the current core: the two differ."""
        skill = staged_install / PAYLOAD
        name = LEGACY_MODES[word]
        result = _uv_cli(
            skill, "util", "instructions", word, cwd=tmp_path,
            env={multi_llm.LEGACY_INSTRUCTIONS_ENV: "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr

        notice = [
            line for line in result.stderr.splitlines()
            if line.startswith("[LEGACY_INSTRUCTIONS]")
        ]
        assert len(notice) == 1, result.stderr
        assert f"from instructions/{name} because" in notice[0]
        assert "see legacy/README.md" in notice[0]
        assert "[LEGACY_INSTRUCTIONS]" not in result.stdout, "stdout may be piped into a prompt"

        frozen = skill / "legacy" / "instructions" / name
        core = cli_common.render_instruction_text(frozen.read_text(encoding="utf-8"), skill)
        line = _result_line(result.stdout, _framed(word, core))
        assert line["artifacts"] == [str(frozen)]
        assert line["exit"] == 0

        current = cli_common.render_instruction_text(
            (skill / "instructions" / name).read_text(encoding="utf-8"), skill,
        )
        assert core != current, f"{name}: the comparison cannot tell the two cores apart"
        for token in ("{SKILL_DIR}", "${CLAUDE_SKILL_DIR}", "{UV_RUN_FLAGS}"):
            assert token not in core, f"{name}: {token} reached the served text"

    @pytest.mark.parametrize("word", sorted(LEGACY_MODES))
    def test_without_the_variable_the_current_core_is_served(self, staged_install, tmp_path, word):
        """(tree, uv) The switch is off unless asked, so a default install reads no
        frozen text. Any value but ``1`` is off too."""
        skill = staged_install / PAYLOAD
        name = LEGACY_MODES[word]
        result = _uv_cli(
            skill, "util", "instructions", word, cwd=tmp_path,
            env={multi_llm.LEGACY_INSTRUCTIONS_ENV: "true"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[LEGACY_INSTRUCTIONS]" not in result.stderr
        current = cli_common.render_instruction_text(
            (skill / "instructions" / name).read_text(encoding="utf-8"), skill,
        )
        _result_line(result.stdout, _framed(word, current))

    @pytest.mark.parametrize("word", UNFROZEN_MODES)
    def test_a_mode_with_no_frozen_copy_prints_no_notice(self, staged_install, tmp_path, word):
        """(tree, uv) It serves the current core, exactly as without the variable, and
        claims nothing about a file it did not read."""
        skill = staged_install / PAYLOAD
        result = _uv_cli(
            skill, "util", "instructions", word, cwd=tmp_path,
            env={multi_llm.LEGACY_INSTRUCTIONS_ENV: "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[LEGACY_INSTRUCTIONS]" not in result.stderr
        assert "[LEGACY_INSTRUCTIONS]" not in result.stdout
        path = skill / mode_table.instruction_path(word)
        current = cli_common.render_instruction_text(path.read_text(encoding="utf-8"), skill)
        assert _result_line(result.stdout, _framed(word, current))["artifacts"] == [str(path)]

    def test_a_mode_with_no_core_file_prints_no_notice(self, staged_install, tmp_path):
        """(tree, uv) The dispatcher-driven branch never reads a file at all."""
        result = _uv_cli(
            staged_install / PAYLOAD, "util", "instructions", NO_CORE_MODE, cwd=tmp_path,
            env={multi_llm.LEGACY_INSTRUCTIONS_ENV: "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[LEGACY_INSTRUCTIONS]" not in result.stderr
        assert f"{multi_llm.CORE_END_MARKER} {NO_CORE_MODE}" in result.stdout


# ===========================================================================
# Legacy implement: the post-implementation test selection
# ===========================================================================

STEP_8 = "8. **Run post-implementation regression tests:**"
UNAVAILABLE = "plugin self-tests: unavailable (development tests are not shipped)"
NOT_RUN = "tests not run (no test command available)"


def _serve_implement(skill: Path) -> str:
    """The served text, so what runs below carries the path the renderer substituted.
    On this test's interpreter rather than through uv, so these rows can run plain."""
    result = subprocess.run(
        [sys.executable, str(skill / "multi_llm.py"), "util", "instructions", "implement"],
        cwd=str(skill.parent),
        env=_clean_env(**{multi_llm.LEGACY_INSTRUCTIONS_ENV: "1"}),
        capture_output=True, encoding="utf-8", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[LEGACY_INSTRUCTIONS]" in result.stderr
    return result.stdout


def _selection_block(served: str, *, as_served: bool = False) -> str:
    """Step 8's fenced ``bash`` block. *as_served* keeps the list item's indentation,
    which is what an agent has if it pastes the lines as they stand."""
    start = served.index(STEP_8)
    match = re.compile(r"^   ```bash\n(.*?)^   ```\n", re.S | re.M).search(served, start)
    assert match, "step 8 has no fenced bash block"
    body = match.group(1)
    if as_served:
        return body
    return "".join(line[3:] if line.startswith("   ") else line for line in body.splitlines(True))


class _Selection:
    """Runs step 8's block in a fixture project, with recording stubs first on PATH."""

    def __init__(self, tmp_path: Path):
        self.project = tmp_path / "consuming project"
        self.project.mkdir()
        self.bin = tmp_path / "stub bin"
        self.bin.mkdir()
        self.log = tmp_path / "stub.log"
        self._runner("pytest")
        self._runner("npm")
        self._runner("project-check")
        self._stub(
            "uv",
            "while [ \"$#\" -gt 0 ] && [ \"$1\" != \"--\" ]; do shift; done\n"
            "[ \"$#\" -gt 0 ] && shift\n"
            "if [ \"${1:-}\" = python ]; then shift; exec \"$REAL_PYTHON\" \"$@\"; fi\n"
            "echo \"stub uv ran\"\n",
        )

    def _stub(self, name: str, rest: str) -> None:
        path = self.bin / name
        path.write_text(
            "#!/bin/sh\n"
            f"{{ printf '%s\\037' '{name}' \"$@\"; printf '\\036'; }} >> \"$STUB_LOG\"\n"
            + rest,
            encoding="utf-8",
        )
        path.chmod(0o755)

    def _runner(self, name: str) -> None:
        self._stub(name, f"echo 'stub {name} ran'\n")

    def fail_uv(self) -> None:
        """A ``uv`` that cannot run anything: the config read comes back empty."""
        self._stub("uv", "echo 'uv: broken' >&2\nexit 1\n")

    def configure(self, text: str) -> None:
        config = self.project / ".multi-llm" / "providers.yaml"
        config.parent.mkdir(exist_ok=True)
        config.write_text(text, encoding="utf-8")

    def verify_command(self, command: str) -> None:
        self.configure(yaml.safe_dump({"driver": {"verify_command": command}}))

    def run(self, block: str, *, modifies: Optional[str] = None) -> subprocess.CompletedProcess:
        environment = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(self.project),
            "STUB_LOG": str(self.log),
            "REAL_PYTHON": sys.executable,
        }
        if modifies is not None:
            environment["PLAN_MODIFIES_SKILL"] = modifies
        return subprocess.run(
            [shutil.which("bash") or "bash", "-c", block], cwd=str(self.project),
            env=environment, capture_output=True, encoding="utf-8", timeout=120,
        )

    def calls(self) -> List[List[str]]:
        if not self.log.exists():
            return []
        records = self.log.read_text(encoding="utf-8").split("\x1e")
        return [record.split("\x1f")[:-1] for record in records if record]

    def names(self) -> List[str]:
        return [call[0] for call in self.calls()]

    def assert_never_touched_the_skill_tests(self, skill: Path) -> None:
        for call in self.calls():
            for word in call:
                if word.startswith("import pathlib"):
                    continue
                assert f"{skill}/tests" not in word, call
        assert "tests/" not in " ".join(
            word for call in self.calls() for word in call if not word.startswith("import ")
        )


@pytest.fixture
def selection(tmp_path) -> _Selection:
    return _Selection(tmp_path)


@pytest.fixture(scope="module")
def staged_block(staged_install) -> str:
    return _selection_block(_serve_implement(staged_install / PAYLOAD))


@pytest.fixture(scope="module")
def staged_served(staged_install) -> str:
    return _serve_implement(staged_install / PAYLOAD)


def _make_source_checkout(root: Path) -> Path:
    """The staged tree plus a ``tests/`` directory, which is the only difference
    between a source checkout and an install that the block may look at."""
    _copy_shipped(PLUGIN_DIR, root)
    skill = root / PAYLOAD
    (skill / "tests").mkdir()
    (skill / "tests" / "test_stub.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    return skill


@pytest.fixture(scope="module")
def source_checkout(tmp_path_factory) -> Path:
    return _make_source_checkout(tmp_path_factory.mktemp("claude-source") / "source checkout")


@posix_shell
class TestTheImplementSelection:
    def _self_test_argv(self, skill: Path) -> List[str]:
        return [
            "uv", "run", "--project", str(skill), "--", "pytest",
            f"{skill}/tests/", "-v", "--tb=short",
        ]

    def _config_read(self, skill: Path, call: List[str]) -> None:
        """The config read asks for the runtime set only, so reading one YAML key never
        needs the dev group from the network."""
        assert call[:-1] == [
            "uv", "run", *unattended.DEFAULT_UV_RUN_ARGS, "--project", str(skill), "--",
            "python", "-c",
        ], call
        assert call[-1].startswith("import pathlib, yaml"), call

    @pytest.mark.parametrize("as_served", [False, True], ids=["dedented", "as-served"])
    def test_a_skill_plan_in_a_source_checkout_runs_the_self_tests(
        self, selection, source_checkout, as_served,
    ):
        """The existing route stays for a checkout that has the tests, ahead of any
        command the project configures."""
        selection.verify_command("project-check --all")
        block = _selection_block(_serve_implement(source_checkout), as_served=as_served)
        result = selection.run(block, modifies="1")
        assert result.returncode == 0, result.stderr
        assert selection.calls() == [self._self_test_argv(source_checkout)]
        assert "stub uv ran" in result.stdout
        assert UNAVAILABLE not in result.stdout
        assert NOT_RUN not in result.stdout

    @pytest.mark.parametrize("as_served", [False, True], ids=["dedented", "as-served"])
    def test_a_skill_plan_in_a_staged_install_records_and_runs_the_project_s(
        self, selection, staged_install, as_served,
    ):
        """An install has no tests to run, and says so, then runs what the project asked
        for. The block has to work however it was pasted: a read that depended on the
        list item's indentation would lose the configured command to an
        ``IndentationError`` and report it as ``tests not run``."""
        skill = staged_install / PAYLOAD
        selection.verify_command("project-check --all")
        block = _selection_block(_serve_implement(skill), as_served=as_served)
        result = selection.run(block, modifies="1")
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == UNAVAILABLE
        assert "stub project-check ran" in result.stdout
        assert NOT_RUN not in result.stdout
        assert selection.names() == ["uv", "project-check"]
        self._config_read(skill, selection.calls()[0])
        assert selection.calls()[1] == ["project-check", "--all"]
        selection.assert_never_touched_the_skill_tests(skill)

    def test_a_configured_pytest_is_the_project_s_and_not_the_skill_s(
        self, selection, staged_install, staged_block,
    ):
        """A project that configures pytest gets exactly its own argv."""
        selection.verify_command("pytest -q project_tests")
        result = selection.run(staged_block, modifies="1")
        assert result.returncode == 0, result.stderr
        assert selection.calls()[1:] == [["pytest", "-q", "project_tests"]]
        for call in selection.calls():
            assert not any(f"{staged_install / PAYLOAD}/tests" in word for word in call)

    @pytest.mark.parametrize("modifies", ["1", None], ids=["skill-plan", "other-plan"])
    @pytest.mark.parametrize("files, runner, argv", [
        ({"package.json": '{"scripts": {"test": "x"}}\n'}, "npm", ["npm", "test"]),
        ({"pytest.ini": "[pytest]\n"}, "pytest", ["pytest", "-v", "--tb=short"]),
        ({"conftest.py": ""}, "pytest", ["pytest", "-v", "--tb=short"]),
        (
            {"pyproject.toml": "[tool.pytest.ini_options]\n"},
            "pytest", ["pytest", "-v", "--tb=short"],
        ),
    ], ids=["package-json", "pytest-ini", "conftest", "pyproject"])
    def test_with_no_configured_command_it_detects_a_runner(
        self, selection, staged_install, staged_block, files, runner, argv, modifies,
    ):
        """The runbook's existing detection, in order, with nothing configured."""
        for name, body in files.items():
            (selection.project / name).write_text(body, encoding="utf-8")
        result = selection.run(staged_block, modifies=modifies)
        assert result.returncode == 0, result.stderr
        assert f"stub {runner} ran" in result.stdout
        assert selection.names() == ["uv", runner]
        assert selection.calls()[1] == argv
        assert NOT_RUN not in result.stdout
        assert (UNAVAILABLE in result.stdout) is (modifies == "1")
        selection.assert_never_touched_the_skill_tests(staged_install / PAYLOAD)

    def test_npm_is_detected_ahead_of_pytest(self, selection, staged_block):
        """The order the runbook has always given."""
        (selection.project / "package.json").write_text(
            '{"scripts": {"test": "x"}}\n', encoding="utf-8",
        )
        (selection.project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        selection.run(staged_block)
        assert selection.names() == ["uv", "npm"]

    def test_a_configured_command_beats_a_detected_runner(self, selection, staged_block):
        """The project said what it wants, so nothing is guessed."""
        selection.verify_command("project-check")
        (selection.project / "package.json").write_text(
            '{"scripts": {"test": "x"}}\n', encoding="utf-8",
        )
        (selection.project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        selection.run(staged_block)
        assert selection.names() == ["uv", "project-check"]

    @pytest.mark.parametrize("config", [
        "driver: {}\n",
        "driver:\n  verify_command: ''\n",
        "defaults: {}\n",
        "",
        "{not: [valid\n",
    ], ids=["no-key", "empty-key", "no-driver", "empty-file", "malformed"])
    def test_a_config_with_no_usable_command_falls_back(self, selection, staged_block, config):
        """A config file that says nothing, or cannot be read, is not a test command,
        and it must not stop the detection behind it."""
        selection.configure(config)
        (selection.project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        result = selection.run(staged_block)
        assert selection.names() == ["uv", "pytest"], result.stderr

    @pytest.mark.parametrize("modifies", ["1", "0", None], ids=["skill-plan", "zero", "unset"])
    def test_no_command_at_all_is_reported_as_not_run(
        self, selection, staged_install, staged_block, modifies,
    ):
        """Never reported as passed, and no runner is called to find out."""
        result = selection.run(staged_block, modifies=modifies)
        assert result.returncode == 0, result.stderr
        assert NOT_RUN in result.stdout
        assert selection.names() == ["uv"]
        self._config_read(staged_install / PAYLOAD, selection.calls()[0])
        assert (UNAVAILABLE in result.stdout) is (modifies == "1")

    @pytest.mark.parametrize("files", [
        {"package.json": '{"scripts": {"build": "x"}}\n'},
        {"pyproject.toml": "[project]\nname = 'x'\n"},
    ], ids=["package-json-without-test", "pyproject-without-pytest"])
    def test_a_project_file_that_names_no_runner_is_not_a_command(
        self, selection, staged_block, files,
    ):
        """Having the file is not enough: it has to declare the runner."""
        for name, body in files.items():
            (selection.project / name).write_text(body, encoding="utf-8")
        result = selection.run(staged_block)
        assert NOT_RUN in result.stdout
        assert selection.names() == ["uv"]

    def test_a_uv_that_cannot_run_falls_back_honestly(self, selection, staged_block):
        """The configured command is out of reach, so the result is the detection's, or
        ``tests not run``. It must not be a silent success."""
        selection.fail_uv()
        selection.verify_command("project-check")
        result = selection.run(staged_block)
        assert NOT_RUN in result.stdout
        assert selection.names() == ["uv"]

    @pytest.mark.parametrize("fixture_name", ["staged_install", "source_checkout"])
    def test_a_non_skill_plan_runs_the_configured_command(
        self, selection, request, fixture_name,
    ):
        """Whatever the install looks like, and with no self-test record."""
        located = request.getfixturevalue(fixture_name)
        skill = located / PAYLOAD if fixture_name == "staged_install" else located
        selection.verify_command("project-check --all")
        result = selection.run(_selection_block(_serve_implement(skill)))
        assert result.returncode == 0, result.stderr
        assert selection.names() == ["uv", "project-check"]
        assert selection.calls()[1] == ["project-check", "--all"]
        assert UNAVAILABLE not in result.stdout
        selection.assert_never_touched_the_skill_tests(skill)

    def test_a_configured_pipeline_keeps_its_quoting(self, selection, staged_block):
        """The documented shape is a shell string, so it goes through ``eval`` whole."""
        selection.verify_command("project-check 'one word' && project-check second")
        result = selection.run(staged_block)
        assert result.returncode == 0, result.stderr
        assert selection.calls()[1:] == [
            ["project-check", "one word"], ["project-check", "second"],
        ]

    def test_an_apostrophe_in_the_path_does_not_break_the_block(self, selection, tmp_path):
        """A rendered path lands inside the block's own quoting. Inside single quotes an
        apostrophe in a home directory ends the string and the command is garbage."""
        skill = _make_source_checkout(tmp_path / "o'brien source")
        block = _selection_block(_serve_implement(skill))
        result = selection.run(block, modifies="1")
        assert result.returncode == 0, result.stderr
        assert selection.calls() == [self._self_test_argv(skill)]

    def test_the_served_text_keeps_the_conditions(self, staged_install, staged_served):
        """The executed rows read the block; this holds the prose around it, which is
        what tells an agent how to set the variable and report the result."""
        skill = staged_install / PAYLOAD
        served = staged_served
        assert f"set `PLAN_MODIFIES_SKILL=1` if the plan modifies `{skill}/` files" in served
        assert f'[ "${{PLAN_MODIFIES_SKILL:-0}}" = 1 ] && [ -d "{skill}/tests" ]' in served
        assert f'echo "{UNAVAILABLE}"' in served
        assert f'echo "{NOT_RUN}"' in served
        assert "A printed `tests not run` line is the result to report. Never report it as passed." in served
        assert served.count(NOT_RUN) >= 2, "the summary template keeps the same wording"

        block = _selection_block(served)
        config = block.index(".multi-llm/providers.yaml")
        assert block.index('"verify_command"') > config
        npm = block.index("npm test")
        pytest_detected = block.rindex("pytest -v --tb=short")
        assert config < npm < pytest_detected < block.index('eval "$TEST_COMMAND"')

    def test_the_self_tests_are_cited_nowhere_else(self, staged_install, staged_served):
        """A second unguarded citation would send an install to pytest on a directory
        it does not have."""
        skill = staged_install / PAYLOAD
        outside = staged_served.replace(_selection_block(staged_served, as_served=True), "")
        assert f"{skill}/tests" not in outside

    def test_the_block_is_the_authored_one(self):
        """Parity: the build copies it, and the source checkout runs the same text."""
        authored = (SKILL_ROOT / "legacy" / "instructions" / "implement.md").read_bytes()
        shipped = (PLUGIN_DIR / PAYLOAD / "legacy" / "instructions" / "implement.md").read_bytes()
        assert authored == shipped


@pytest.mark.package
class TestTheImplementSelectionWithRealUv:
    def test_the_config_read_runs_for_real_from_the_install(self, staged_install, tmp_path):
        """(tree, uv) The stubbed rows cannot show that the runtime set can import yaml
        from a path with spaces. Only the project's command is a stub here."""
        uv = _real_uv()
        skill = staged_install / PAYLOAD
        _uv_cli(skill, "--help", cwd=tmp_path)
        selection = _Selection(tmp_path)
        (selection.bin / "uv").unlink()
        selection.verify_command("project-check --all")
        served = _uv_cli(
            skill, "util", "instructions", "implement", cwd=tmp_path,
            env={multi_llm.LEGACY_INSTRUCTIONS_ENV: "1"},
        )
        assert served.returncode == 0, served.stdout + served.stderr
        result = subprocess.run(
            [shutil.which("bash"), "-c", _selection_block(served.stdout)],
            cwd=str(selection.project),
            env=_clean_env(
                PATH=os.pathsep.join((str(selection.bin), str(Path(uv).parent), "/usr/bin", "/bin")),
                STUB_LOG=str(selection.log),
                PLAN_MODIFIES_SKILL="1",
            ),
            capture_output=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert UNAVAILABLE in result.stdout
        assert selection.calls() == [["project-check", "--all"]], result.stderr


# ===========================================================================
# Runtime references
# ===========================================================================

_SKILL_TOKEN = r"(?:\$\{CLAUDE_SKILL_DIR\}|\$\{MULTI_LLM_SKILL_DIR\}|\$MULTI_LLM_SKILL_DIR|\{SKILL_DIR\})"
_TOKEN_CITATION = re.compile(_SKILL_TOKEN + r"/([\w./<>-]*)")
_BARE_CITATION = re.compile(
    r"(?<![\w./}$<-])((?:references|instructions|prompts|schemas|templates|hosts|utils|assets)"
    r"/[\w./<>-]*\w)"
)
_HOOK_CITATION = re.compile(r"\$(skill|root)/([\w./-]+)")
_URL = re.compile(r"https?://[^\s)>\"'`]+")

#: Paths the running skill creates, or that only a source checkout has. Named one by
#: one, against the file that cites them, so neither can spread unnoticed.
EXAMPLE_OF_A_RUNTIME_PATH = {
    (f"{PAYLOAD}/legacy/instructions/implement.md", "state/abc123.json"),
}
SOURCE_CHECKOUT_ONLY = {
    (f"{PAYLOAD}/legacy/instructions/implement.md", "tests"),
    (f"{PAYLOAD}/legacy/instructions/implement.md", "tests/"),
}
#: A citation with ``<mode>`` in it, which stands for every mode with a core file.
MODE_PLACEHOLDER = "instructions/<mode>.md"

#: Plan names the shipped text uses as examples of the *user's* plan. Anything else
#: under ``plans/`` is a citation of a working note, which is never published.
EXAMPLE_PLANS = frozenset({"my-feature", "foo", "x"})

#: Shapes that can only be this repository's own development material. A hit counts
#: only when it resolves here, so ``tests/race.test.mjs``, an example of a consuming
#: project's file, is not one.
_DEV_CITATION = re.compile(
    r"(?<![\w./}$-])("
    r"tests/[\w./-]*\w|docs/[\w./-]*\w|packaging/[\w./-]*\w|\.github/[\w./-]*\w"
    r"|plans/[\w.-]+\.md|AGENTS\.md|CLAUDE\.md"
    r")"
)


def _runtime_text(tree: Path) -> Dict[str, str]:
    """``{plugin-relative: text}`` for what a session reads or runs."""
    skill = tree / PAYLOAD
    paths = [
        skill / "SKILL.md",
        *(path for folder in ("instructions", "references", "legacy")
          for path in (skill / folder).rglob("*")),
        *(tree / "hooks").iterdir(),
    ]
    return {
        path.relative_to(tree).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(paths) if path.is_file()
    }


def _token_citations(text: str) -> Set[str]:
    return {match.group(1).rstrip(".") for match in _TOKEN_CITATION.finditer(text)}


def _resolves(skill: Path, cited: str) -> bool:
    return (skill / cited).exists()


def _resolves_bare(skill: Path, cited: str, *, in_legacy: bool) -> bool:
    """A bare citation may name a module attribute, ``utils/unattended.find_uv``, and a
    frozen file's relative citation may mean its own frozen sibling."""
    roots = [skill, skill / "legacy"] if in_legacy else [skill]
    for root in roots:
        if (root / cited).exists():
            return True
        stem, dot, _attribute = cited.rpartition(".")
        if dot and (root / f"{stem}.py").is_file():
            return True
    return False


class TestRuntimeReferences:
    def test_every_cited_skill_path_resolves(self):
        """(tree) A citation is read at run time, where a missing file is an agent stuck
        mid-phase. The exceptions are named above and cannot grow by accident."""
        skill = PLUGIN_DIR / PAYLOAD
        unresolved = []
        for relative, text in _runtime_text(PLUGIN_DIR).items():
            for cited in _token_citations(text):
                if cited == MODE_PLACEHOLDER or _resolves(skill, cited):
                    continue
                if (relative, cited) in EXAMPLE_OF_A_RUNTIME_PATH | SOURCE_CHECKOUT_ONLY:
                    continue
                unresolved.append((relative, cited))
        assert unresolved == []

    def test_the_mode_placeholder_stands_for_real_files(self):
        """(tree) It cannot be checked as a path, so it is checked as what it means."""
        skill = PLUGIN_DIR / PAYLOAD
        assert "`${CLAUDE_SKILL_DIR}/instructions/<mode>.md`" in (skill / "SKILL.md").read_text(
            encoding="utf-8",
        )
        for mode in mode_table.MODES:
            if mode.instruction_file:
                assert (skill / "instructions" / mode.instruction_file).is_file(), mode.name

    def test_the_named_exceptions_are_still_needed(self):
        """(tree) An exception nothing cites is one a later citation could hide behind,
        and one that resolves was never an exception."""
        skill = PLUGIN_DIR / PAYLOAD
        text = _runtime_text(PLUGIN_DIR)
        for relative, cited in EXAMPLE_OF_A_RUNTIME_PATH | SOURCE_CHECKOUT_ONLY:
            assert cited in _token_citations(text[relative]), (relative, cited)
            assert not _resolves(skill, cited), (relative, cited)

    def test_the_source_checkout_branch_stays_guarded(self):
        """(tree) The ``tests/`` exception exists for one guarded branch, so its citations
        must all sit in the block whose condition checks for the directory."""
        text = (PLUGIN_DIR / PAYLOAD / "legacy" / "instructions" / "implement.md").read_text(
            encoding="utf-8",
        )
        block = text[text.index(STEP_8):]
        block = block[:block.index("   ```\n", block.index("   ```bash\n") + 1)]
        assert text.count("${CLAUDE_SKILL_DIR}/tests") == block.count("${CLAUDE_SKILL_DIR}/tests") == 2
        assert '[ -d "${CLAUDE_SKILL_DIR}/tests" ]' in block

    def test_every_bare_skill_path_resolves(self):
        """(tree) The same rule for a citation without the token, which reads as a path
        relative to the skill. ``references/uv-check.md`` must exist, or the file
        citing it is wrong."""
        skill = PLUGIN_DIR / PAYLOAD
        unresolved = []
        for relative, text in _runtime_text(PLUGIN_DIR).items():
            in_legacy = relative.startswith(f"{PAYLOAD}/legacy/")
            for match in _BARE_CITATION.finditer(text):
                cited = match.group(1).rstrip(".")
                if cited == MODE_PLACEHOLDER:
                    continue
                if not _resolves_bare(skill, cited, in_legacy=in_legacy):
                    unresolved.append((relative, cited))
        assert sorted(set(unresolved)) == []

    def test_every_path_a_hook_script_builds_resolves(self):
        """(tree) ``$skill`` and ``$root`` are how the scripts reach the dispatcher."""
        for script in claude_build.HOOK_SCRIPTS:
            text = (PLUGIN_DIR / script).read_text(encoding="utf-8")
            cited = _HOOK_CITATION.findall(text)
            assert cited, script
            for variable, relative in cited:
                base = PLUGIN_DIR if variable == "root" else PLUGIN_DIR / PAYLOAD
                assert (base / relative).exists(), (script, variable, relative)

    def test_no_runtime_text_cites_a_development_only_path(self):
        """(tree) An installed plugin has no ``tests/``, ``docs/`` or ``plans/``, so a
        citation of one points at nothing a reader of the install can open."""
        development_only = []
        for relative, text in _runtime_text(PLUGIN_DIR).items():
            stripped = _URL.sub("", text)
            for match in _DEV_CITATION.finditer(stripped):
                cited = match.group(1).rstrip(".")
                if cited.startswith("plans/"):
                    if Path(cited).stem not in EXAMPLE_PLANS:
                        development_only.append((relative, cited))
                    continue
                if (REPO_ROOT / cited).exists() or (SKILL_ROOT / cited).exists():
                    development_only.append((relative, cited))
        assert sorted(set(development_only)) == []

    def test_the_checker_finds_a_development_only_citation(self):
        """(src) Each shape it is meant to catch, so an empty answer above means
        something. ``${CLAUDE_SKILL_DIR}/tests`` is the guarded branch's, not this rule's."""
        for sample, cited in (
            ("see `tests/test_x.py`.", "tests/test_x.py"),
            ("(plans/some-note.md T11)", "plans/some-note.md"),
            ("See docs/hosts.md for", "docs/hosts.md"),
            ("read AGENTS.md first", "AGENTS.md"),
            ("in packaging/claude/build.py", "packaging/claude/build.py"),
        ):
            assert [m.group(1) for m in _DEV_CITATION.finditer(sample)] == [cited], sample
        assert not list(_DEV_CITATION.finditer('"${CLAUDE_SKILL_DIR}/tests/" -v'))

    @pytest.mark.parametrize("relative, url, label", [
        (f"{PAYLOAD}/SKILL.md", PROVIDERS_DOC_URL, "the provider documentation"),
        (
            f"{PAYLOAD}/instructions/init-config.md",
            CONFIGURATION_DOC_URL,
            "the configuration documentation",
        ),
    ])
    def test_each_documentation_link_is_at_its_site(self, relative, url, label):
        """(tree) A user of the install has no repository to read ``docs/`` from, so
        these two are the only pointers that may leave the plugin."""
        text = (PLUGIN_DIR / relative).read_text(encoding="utf-8")
        assert f"[{label}]({url})" in text
        stripped = _URL.sub("", text)
        assert "docs/providers.md" not in stripped
        assert "docs/configuration.md" not in stripped

    def test_no_runtime_text_keeps_the_old_repository_pointer(self):
        """(tree) Outside its URL, neither document's path appears anywhere a session
        reads."""
        for relative, text in _runtime_text(PLUGIN_DIR).items():
            stripped = _URL.sub("", text)
            assert "docs/providers.md" not in stripped, relative
            assert "docs/configuration.md" not in stripped, relative

    @pytest.mark.parametrize("relative", ["docs/providers.md", "docs/configuration.md"])
    def test_the_documents_exist_in_the_source_repository(self, relative):
        """(src) A public link target, so moving one breaks every installed copy."""
        assert (REPO_ROOT / relative).is_file()
        assert relative in _index_modes(relative)

    def test_no_payload_carries_a_copy_of_the_documents(self):
        """(tree) The links are the route; a second copy would be a second owner."""
        for tree in (PLUGIN_RELATIVE, CODEX_RELATIVE):
            assert not [path for path in _tracked_below(tree) if "docs" in path.split("/")]


# ===========================================================================
# Make wiring
# ===========================================================================

def _makefile_recipes() -> Dict[str, List[str]]:
    recipes: Dict[str, List[str]] = {}
    current = None
    for line in (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+):(?!=)", line)
        if match:
            current = match.group(1)
            recipes[current] = []
        elif line.startswith("\t") and current:
            recipes[current].append(line.strip())
        elif line.strip() and not line.startswith(("#", " ")):
            current = None
    return recipes


class TestMake:
    TARGETS = (
        "claude-regen", "claude-package", "claude-check-release",
        "claude-check-bump", "claude-package-check-ci",
    )

    def test_the_targets_exist_and_are_phony(self):
        """(src) Each is a verb, never a file, and each has a help line."""
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        phony = " ".join(re.findall(r"^\.PHONY:(.*)$", makefile.replace("\\\n", " "), re.M))
        recipes = _makefile_recipes()
        for target in self.TARGETS:
            assert target in recipes, target
            assert re.search(rf"(?<![\w-]){re.escape(target)}(?![\w-])", phony), target
            assert f"make {target}" in makefile, f"{target} has no help line"

    def test_the_clean_build_ignores_every_dirty_opt_in(self):
        """(src) ``claude-package`` is what CI trusts, so a developer's exported variable
        must not turn it into the regeneration."""
        recipes = _makefile_recipes()
        assert any("--allow-dirty" in line for line in recipes["claude-regen"])
        package = " ".join(recipes["claude-package"])
        assert "--allow-dirty" not in package
        assert "MULTI_LLM_ALLOW_DIRTY= " in package
        assert "--require-no-diff" in package

    def test_the_ci_tier_runs_both_selections(self):
        """(src) Both selections, the second turning a skip into a failure, and its
        clean build unless ``ALLOW_DIRTY`` asks otherwise."""
        recipe = "\n".join(_makefile_recipes()["claude-package-check-ci"])
        assert '-m "not live and not container and not package"' in recipe
        assert 'MLLM_REQUIRE_PACKAGE_ROWS=1 $(CLAUDE_PACKAGE_PYTEST) -m "package and not container"' in recipe
        assert "$(MAKE) claude-package" in recipe
        assert "claude-check-release" in recipe

    def test_the_pytest_list_names_this_route_s_suite(self):
        """(src) ``claude-package-check-ci`` runs what this variable names, and only that."""
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        block = re.search(r"^CLAUDE_PACKAGE_PYTEST :=((?:.*\\\n)*.*)$", makefile, re.M)
        assert block
        assert "skills/multi-llm/tests/test_claude_package.py" in block.group(1)
        assert "test_opencode_package.py" not in block.group(1)

    def test_at_least_one_package_row_is_selected(self):
        """(src) pytest exits 5 on an empty selection, and the CI tier's second pass would
        go red on that for no reason a reader could find."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
             "-m", "package and not container", str(Path(__file__))],
            cwd=str(SKILL_ROOT), env=_clean_env(),
            capture_output=True, encoding="utf-8", timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "test_claude_package.py::" in result.stdout

    @pytest.mark.package
    def test_the_targets_on_a_throwaway_repository(self, rig, tmp_path):
        """(rig, make, uv) The real recipes over the real builder: a clean tree passes, a
        dirty input is refused whatever the environment says, the regeneration takes
        it, and a hand edit in the generated tree is drift."""
        if not shutil.which("make"):
            _unavailable("make is not on PATH")
        if not shutil.which("uv") and unattended.find_uv() is None:
            _unavailable("uv is not installed")
        repo = rig.repo
        shutil.copy2(REPO_ROOT / "Makefile", repo / "Makefile")
        for relative in _index_modes("packaging/common"):
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, target)
        rig.build(destination=repo / PLUGIN_RELATIVE)
        rig.commit("the generated tree and the recipes")

        def make(target: str, **env: str) -> subprocess.CompletedProcess:
            environment = _clean_env(
                UV_PROJECT_ENVIRONMENT=str(tmp_path / "venv"),
                PYTHONDONTWRITEBYTECODE="1",
                **env,
            )
            environment.pop("MAKEFLAGS", None)
            environment.pop("MAKELEVEL", None)
            uv = unattended.find_uv()
            environment["PATH"] = f"{Path(uv).parent}{os.pathsep}{environment.get('PATH', '')}"
            return subprocess.run(
                ["make", "--no-print-directory", target], cwd=str(repo), env=environment,
                capture_output=True, encoding="utf-8", timeout=420,
            )

        def output(result: subprocess.CompletedProcess) -> str:
            return result.stdout + result.stderr

        clean = make("claude-package")
        assert clean.returncode == 0, output(clean)
        assert "clean rebuild equals the committed tree" in clean.stdout

        generated = repo / PLUGIN_RELATIVE / "hooks" / "doctor.sh"
        committed = generated.read_bytes()
        rig.append("hooks/doctor.sh", "\n# edited\n")
        refused = make("claude-package", MULTI_LLM_ALLOW_DIRTY="1", ALLOW_DIRTY="1")
        assert refused.returncode != 0, output(refused)
        assert "refusing to stage from a dirty tree" in output(refused)
        assert generated.read_bytes() == committed

        regenerated = make("claude-regen")
        assert regenerated.returncode == 0, output(regenerated)
        assert generated.read_bytes().endswith(b"# edited\n")
        rig.commit("the edit and its output")
        assert make("claude-package").returncode == 0

        rig.append(f"{PLUGIN_RELATIVE}/LICENSE", "\nedited by hand\n")
        rig.commit("a hand edit in the generated tree")
        drifted = make("claude-package")
        assert drifted.returncode != 0, output(drifted)
        assert "differs from the committed tree" in output(drifted)
