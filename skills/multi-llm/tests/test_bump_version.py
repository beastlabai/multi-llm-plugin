"""``make bump-version``: the shared version, its mirrors, and the regeneration.

Scope: ``packaging/common/bump_version.py``. The rows copy this repository's own
mirror files into a throwaway repository, so a pattern that stops matching the real
files fails here rather than at release time. The regeneration is stubbed except in
:class:`TestTheRealFiles`, which plans against this checkout without writing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGING = REPO_ROOT / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

from common import bump_version, release_history  # noqa: E402

CURRENT = release_history.version_in((REPO_ROOT / bump_version.OWNER).read_text(encoding="utf-8"))


def _next_patch(version: str) -> str:
    major, minor, patch = re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()
    return f"{major}.{minor}.{int(patch) + 1}"


NEXT = _next_patch(CURRENT)


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "init.defaultBranch=master", "-c", "tag.gpgsign=false",
         "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         *args],
        cwd=str(cwd), capture_output=True, text=True)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout


@pytest.fixture
def root(tmp_path) -> Path:
    """The real mirror files, in a repository of their own."""
    for mirror in bump_version.MIRRORS:
        target = tmp_path / mirror.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / mirror.path).read_bytes())
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "mirrors")
    return tmp_path


def _snapshot(root: Path) -> dict:
    return {m.path: (root / m.path).read_bytes() for m in bump_version.MIRRORS}


def _opencode_package(text: str) -> dict:
    def walk(node):
        if isinstance(node, dict):
            if node.get("name") == bump_version.NPM_NAME and "public_version" in node:
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
    found = list(walk(yaml.safe_load(text)))
    assert len(found) == 1
    return found[0]


class TestTheBump:
    def test_every_mirror_moves_and_nothing_else_does(self, root):
        before = _snapshot(root)
        assert bump_version.bump(root, NEXT, regenerate=False) == 0
        for mirror in bump_version.MIRRORS:
            old = before[mirror.path].decode("utf-8")
            new = (root / mirror.path).read_text(encoding="utf-8")
            changed = bump_version.changed_lines(old, new)
            assert changed, mirror.path
            for was, now in changed:
                assert was.replace(CURRENT, NEXT) == now, (mirror.path, was, now)
            assert len(old.splitlines()) == len(new.splitlines())

    def test_each_reader_now_sees_the_new_version(self, root):
        public_before = _opencode_package(
            (root / "skills/multi-llm/hosts/capabilities.yaml").read_text(encoding="utf-8")
        )["public_version"]
        bump_version.bump(root, NEXT, regenerate=False)
        read = lambda p: (root / p).read_text(encoding="utf-8")  # noqa: E731
        assert release_history.version_in(read(bump_version.OWNER)) == NEXT
        assert re.search(rf'name = "multi-llm"\nversion = "{re.escape(NEXT)}"',
                         read("skills/multi-llm/uv.lock"))
        assert json.loads(read(".claude-plugin/plugin.json"))["version"] == NEXT
        assert json.loads(read("packaging/opencode/package.json"))["version"] == NEXT
        package = _opencode_package(read("skills/multi-llm/hosts/capabilities.yaml"))
        assert package["version"] == NEXT
        assert package["public_version"] == public_before
        readme = read("packaging/opencode/README.md")
        assert f"{bump_version.NPM_NAME}@{NEXT}" in readme
        assert f"{bump_version.NPM_NAME}@{CURRENT}" not in readme

    def test_a_dry_run_writes_nothing(self, root, capsys):
        before = _snapshot(root)
        assert bump_version.bump(root, NEXT, dry_run=True) == 0
        assert _snapshot(root) == before
        out = capsys.readouterr().out
        assert f"{CURRENT} -> {NEXT}" in out and "dry run" in out and "claude-regen" in out

    def test_the_regeneration_runs_in_the_procedure_s_order(self, root):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs["cwd"]))
            return subprocess.CompletedProcess(argv, 0)

        assert bump_version.bump(root, NEXT, runner=runner) == 0
        assert [argv for argv, _ in calls] == [list(a) for a in bump_version.REGENERATE]
        assert calls[0][0][-1].endswith("render_capabilities.py")
        assert calls[-1][0][-1] == "claude-regen"
        assert {cwd for _, cwd in calls} == {str(root)}

    def test_a_failed_step_stops_and_names_what_is_left(self, root, capsys):
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 2 if argv[-1] == "codex-regen" else 0)

        assert bump_version.bump(root, NEXT, runner=runner) == 1
        assert calls[-1][-1] == "codex-regen" and len(calls) == 2
        err = capsys.readouterr().err
        assert "codex-regen" in err and "claude-regen" in err
        assert f"already say {NEXT}" in err


class TestRefusals:
    @pytest.mark.parametrize("version", ["0.1", "v1.0.0", "1.0.0.0", "next"])
    def test_a_version_that_is_not_strict_semver(self, root, version, capsys):
        before = _snapshot(root)
        assert bump_version.bump(root, version, regenerate=False) == 1
        assert "not a strict semver version" in capsys.readouterr().err
        assert _snapshot(root) == before

    @pytest.mark.parametrize("version", [CURRENT, f"{CURRENT}+rebuild", "0.0.1"])
    def test_a_version_that_is_not_newer(self, root, version, capsys):
        assert bump_version.bump(root, version, regenerate=False) == 1
        assert f"not newer than the current version {CURRENT}" in capsys.readouterr().err

    @pytest.mark.parametrize("family", ["claude-v", "codex-v", "opencode-v"])
    def test_a_version_any_family_has_tagged(self, root, family, capsys):
        _git(root, "tag", "-a", f"{family}{NEXT}", "-m", "x")
        before = _snapshot(root)
        assert bump_version.bump(root, NEXT, regenerate=False) == 1
        assert f"{NEXT} is tagged" in capsys.readouterr().err
        assert _snapshot(root) == before

    def test_a_drifted_mirror_is_named_and_nothing_is_written(self, root, capsys):
        manifest = root / ".claude-plugin/plugin.json"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(f'"{CURRENT}"', '"0.0.9"'),
            encoding="utf-8")
        before = _snapshot(root)
        assert bump_version.bump(root, NEXT, regenerate=False) == 1
        err = capsys.readouterr().err
        assert ".claude-plugin/plugin.json: expected the version" in err
        assert "nothing was written" in err
        assert _snapshot(root) == before

    def test_public_version_alone_is_not_the_package_version(self, root, capsys):
        """A capabilities.yaml whose package.version drifted must not have its
        public_version rewritten instead."""
        path = root / "skills/multi-llm/hosts/capabilities.yaml"
        text = path.read_text(encoding="utf-8")
        text = re.sub(rf'(name: {bump_version.NPM_NAME}\n\s+version: )"[^"]+"',
                      r'\1"0.0.9"', text)
        path.write_text(text, encoding="utf-8")
        assert bump_version.bump(root, NEXT, regenerate=False) == 1
        assert "capabilities.yaml: expected the version" in capsys.readouterr().err

    def test_one_drifted_readme_pin_is_not_rewritten_around(self, root, capsys):
        """Review finding: one pin at the current version, the other behind."""
        path = root / "packaging/opencode/README.md"
        text = path.read_text(encoding="utf-8")
        spec = f"{bump_version.NPM_NAME}@{CURRENT}"
        assert text.count(spec) >= 2
        head, tail = text.split(spec, 1)
        path.write_text(head + f"{bump_version.NPM_NAME}@0.0.9" + tail, encoding="utf-8")
        assert bump_version.bump(root, NEXT, regenerate=False) == 1
        assert "also pins 0.0.9" in capsys.readouterr().err

    @pytest.mark.parametrize("text, want", [
        ("x opencode-multi-llm@1.2.3\n", "x opencode-multi-llm@1.2.4\n"),
        ("end opencode-multi-llm@1.2.3.", "end opencode-multi-llm@1.2.4."),
        ('["opencode-multi-llm@1.2.3"]', '["opencode-multi-llm@1.2.4"]'),
    ])
    def test_readme_specs_move_as_whole_tokens(self, text, want):
        assert bump_version._npm_readme(text, "1.2.3", "1.2.4") == want

    @pytest.mark.parametrize("text", [
        "opencode-multi-llm@1.2.30", "opencode-multi-llm@1.2.3-rc1",
        "opencode-multi-llm@1.2.3.4", "nothing pinned",
    ])
    def test_a_longer_or_missing_token_is_refused(self, text):
        with pytest.raises(bump_version.BumpError):
            bump_version._npm_readme(text, "1.2.3", "1.2.4")

    def test_every_problem_is_reported_at_once(self, root, capsys):
        for relative in (".claude-plugin/plugin.json", "packaging/opencode/package.json"):
            path = root / relative
            path.write_text(path.read_text(encoding="utf-8").replace(f'"{CURRENT}"', '"0.0.9"'),
                            encoding="utf-8")
        assert bump_version.bump(root, CURRENT, regenerate=False) == 1
        err = capsys.readouterr().err
        assert "not newer" in err and "plugin.json" in err and "package.json" in err


class TestTheRealFiles:
    def test_this_checkout_s_mirrors_all_carry_the_current_version(self):
        """Plans, never writes: every pattern still finds its line in the real files."""
        old, edits = bump_version.plan(REPO_ROOT, "999.0.0")
        assert old == CURRENT
        assert set(edits) == {m.path for m in bump_version.MIRRORS}

    def test_the_mirrors_are_the_ones_the_bump_procedure_lists(self):
        readme = (PACKAGING / "claude" / "README.md").read_text(encoding="utf-8")
        procedure = readme.split("## Bump procedure", 1)[1].split("\n## ", 1)[0]
        for mirror in bump_version.MIRRORS:
            assert f"`{mirror.path}`" in procedure, mirror.path
        assert "make bump-version" in procedure
