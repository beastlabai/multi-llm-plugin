"""Claude Code installs follow ``master``, so a moved plugin tree is a release.

Scope: ``release_problems`` and ``release_tags`` in ``packaging/claude/build.py``,
the CLI flags over them, and the CI wiring that holds a pull request to the rule.
Beside ``test_codex_package.py::test_the_bump_rule_reads_git_both_ways`` in style:
each row builds a throwaway repository with this one's shape where the rule looks
(the version owner, the generated tree) and tags it as a release history would be
tagged. Nothing here touches the network or this repository's own tags.

**The whole tree is the rule, not the shared digest.** Hook scripts and the licence
sit outside :func:`release_history.shared_payload_digest`, and Claude Code caches by
manifest version, so an edit to either under an unmoved version reaches nobody who
has that version. The rows for those two assert both halves: the digest did not
move, and the rule refuses anyway.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGING = REPO_ROOT / "packaging"
CLAUDE_BUILD = PACKAGING / "claude" / "build.py"
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

from common import payload, release_history  # noqa: E402
from common.payload import PayloadError  # noqa: E402

_spec = importlib.util.spec_from_file_location("claude_build_for_release", CLAUDE_BUILD)
claude_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claude_build)

TREE = claude_build.PLUGIN_RELATIVE
HOOK = "hooks/doctor.sh"
LICENCE = "LICENSE"
SHIPPED = "skills/multi-llm/multi_llm.py"

#: The allowlist the shared digest reads, in the two headings the real manifest has.
MANIFEST = """\
# fixture manifest
--- the product under test ---
skills/multi-llm/*.py
skills/multi-llm/pyproject.toml
--- the rig that drives it ---
skills/multi-llm/tests/**
"""

_GIT_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
    GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
         "-c", "init.defaultBranch=master", *args],
        cwd=str(cwd), env=_GIT_ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout


class Repo:
    """A throwaway repository holding a version owner and a generated plugin tree.

    :meth:`commit` writes the version owner and the shared file, in the source and
    in the tree, so a default commit is a shipped-byte edit with a version on it.
    ``.gitattributes`` pins ``eol=lf`` as the real one does, so a Windows checkout's
    bytes are the committed ones.
    """

    def __init__(self, root: Path, *, with_tree: bool = True):
        self.root = root
        root.mkdir(parents=True)
        _git(root, "init", "-q")
        self.write(".gitattributes", "* text=auto eol=lf\n")
        self.write(payload.MANIFEST_RELATIVE, MANIFEST)
        self.with_tree = with_tree
        if with_tree:
            self.write_tree(HOOK, "#!/bin/sh\necho doctor\n")
            self.write_tree(LICENCE, "licence\n")

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def write_tree(self, relative: str, text: str) -> None:
        self.write(f"{TREE}/{relative}", text)

    def commit(self, version: str, body: str = "a", *, tag: str = "",
               lightweight: bool = False) -> str:
        """Commit *version* with *body* as the shared file's content."""
        pyproject = f'[project]\nname = "multi-llm"\nversion = "{version}"\n'
        shipped = f"BODY = {body!r}\n"
        self.write(claude_build.VERSION_PATH, pyproject)
        self.write(SHIPPED, shipped)
        if self.with_tree:
            self.write_tree("skills/multi-llm/pyproject.toml", pyproject)
            self.write_tree(SHIPPED, shipped)
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "--allow-empty", "-m", f"{version} {body}")
        if tag:
            self.tag(tag, lightweight=lightweight)
        return self.head()

    def tag(self, name: str, ref: str = "HEAD", *, lightweight: bool = False) -> None:
        if lightweight:
            _git(self.root, "tag", name, ref)
        else:
            _git(self.root, "tag", "-a", name, "-m", name, ref)

    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD").strip()

    def digest(self, ref: str = "HEAD") -> str:
        return release_history.shared_payload_digest(
            self.root, release_history.resolve_commit(self.root, ref))

    def problems(self, **kwargs) -> tuple:
        return claude_build.release_problems(self.root, **kwargs)


@pytest.fixture
def repo(tmp_path) -> Repo:
    return Repo(tmp_path / "repo")


def _one(problems: tuple) -> str:
    assert len(problems) == 1, problems
    return problems[0]


# ===========================================================================
# Against the newest claude-v* tag
# ===========================================================================

class TestAgainstTheNewestTag:
    def test_no_tag_and_no_base_is_the_empty_case(self, repo):
        """The state before the first Claude release: nothing to compare."""
        repo.commit("0.1.0")
        repo.commit("0.1.0", "b")
        assert claude_build.release_tags(repo.root) == ()
        assert repo.problems() == ()

    def test_the_tagged_commit_itself_passes(self, repo):
        repo.commit("0.1.0", tag="claude-v0.1.0")
        assert repo.problems() == ()

    def test_an_unmoved_tree_under_the_same_version_passes(self, repo):
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.write("docs/notes.md", "a development edit\n")
        repo.commit("0.1.0")
        assert repo.problems() == ()

    def test_a_moved_tree_under_the_same_version_is_refused(self, repo):
        """The tag, the version and the remedy, all named: this line is what a
        contributor reads on a red pull request."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.commit("0.1.0", "b")
        line = _one(repo.problems())
        assert f"{TREE}/ moved since claude-v0.1.0" in line
        assert "the version is 0.1.0, not newer than 0.1.0" in line
        assert f"Bump {claude_build.VERSION_PATH}" in line
        assert "`make claude-regen`" in line
        assert "claude-v<version>" in line
        assert "packaging/claude/README.md" in line

    @pytest.mark.parametrize("relative", [HOOK, LICENCE], ids=["hook", "licence"])
    def test_an_edit_outside_the_shared_digest_is_refused(self, repo, relative):
        """The shared check is satisfied by this commit; this rule is not."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.write_tree(relative, "edited after the release\n")
        repo.commit("0.1.0")
        assert repo.digest() == repo.digest("claude-v0.1.0")
        assert "moved since claude-v0.1.0" in _one(repo.problems())

    def test_an_executable_bit_alone_is_refused(self, repo):
        """A hook loses or gains the mode ``hooks.json`` depends on, with no byte
        moving: no digest over contents would see it, the tree id does."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        path = f"{TREE}/{HOOK}"
        _git(repo.root, "update-index", "--chmod=+x", path)
        _git(repo.root, "commit", "-q", "-m", "mode only")
        assert (_git(repo.root, "rev-parse", f"HEAD:{path}")
                == _git(repo.root, "rev-parse", f"claude-v0.1.0:{path}"))
        assert "moved since claude-v0.1.0" in _one(repo.problems())

    def test_a_moved_tree_under_a_newer_version_passes(self, repo):
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.commit("0.1.1", "b")
        assert repo.problems() == ()

    def test_a_moved_tree_under_an_older_version_is_refused(self, repo):
        repo.commit("0.2.0", tag="claude-v0.2.0")
        repo.commit("0.1.9", "b")
        line = _one(repo.problems())
        assert "moved since claude-v0.2.0" in line
        assert "the version is 0.1.9, not newer than 0.2.0" in line

    def test_build_metadata_is_not_a_newer_version(self, repo):
        """semver gives ``+build`` no precedence, and the shared rule reads
        ``0.1.0+rebuild`` as ``0.1.0``."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.commit("0.1.0+rebuild", "b")
        assert "not newer than 0.1.0" in _one(repo.problems())

    def test_a_prerelease_of_the_released_version_is_not_newer(self, repo):
        repo.commit("0.2.0", tag="claude-v0.2.0")
        repo.commit("0.2.0-rc.1", "b")
        assert "not newer than 0.2.0" in _one(repo.problems())

    def test_an_annotated_and_a_lightweight_tag_read_alike(self, tmp_path):
        for lightweight in (False, True):
            repo = Repo(tmp_path / f"repo-{lightweight}")
            repo.commit("0.1.0", tag="claude-v0.1.0", lightweight=lightweight)
            repo.commit("0.1.0", "b")
            assert "moved since claude-v0.1.0" in _one(repo.problems()), lightweight

    def test_newest_is_semver_s_newest_not_the_tag_list_s(self, repo):
        """``0.9.0`` sorts after ``0.10.0`` as text, so a text sort would hold the
        candidate to the older release and pass an edit under ``0.10.0``."""
        repo.commit("0.9.0", tag="claude-v0.9.0")
        repo.commit("0.10.0", "b", tag="claude-v0.10.0")
        repo.commit("0.10.0", "c")
        assert claude_build.release_tags(repo.root) == ("claude-v0.9.0", "claude-v0.10.0")
        assert "moved since claude-v0.10.0" in _one(repo.problems())

    def test_a_prerelease_tag_sorts_before_its_release(self, repo):
        repo.commit("0.2.0-rc.1", tag="claude-v0.2.0-rc.1")
        repo.commit("0.2.0", "b", tag="claude-v0.2.0")
        assert claude_build.release_tags(repo.root) == (
            "claude-v0.2.0-rc.1", "claude-v0.2.0")

    def test_other_families_tags_are_not_this_rule_s(self, repo):
        """A Codex-only release says nothing about what Claude users hold."""
        repo.commit("0.1.0", tag="codex-v0.1.0")
        repo.commit("0.1.0", "b", tag="opencode-v0.1.0")
        repo.commit("0.1.0", "c")
        assert claude_build.release_tags(repo.root) == ()
        assert repo.problems() == ()

    def test_a_candidate_without_the_tree_is_refused(self, tmp_path):
        bare = Repo(tmp_path / "repo", with_tree=False)
        bare.commit("0.1.0")
        assert "carries no packaging/claude/multi-llm/" in _one(bare.problems())
        assert "`make claude-regen`" in _one(bare.problems())

    def test_a_named_candidate_is_read_rather_than_head(self, repo):
        first = repo.commit("0.1.0", tag="claude-v0.1.0")
        moved = repo.commit("0.1.0", "b")
        assert repo.problems(candidate=first) == ()
        assert repo.problems(candidate=moved) != ()


class TestWhatTheTagsCanSay:
    def test_a_malformed_tag_raises_by_name(self, repo):
        """Skipped, it would hide the release it marks. The moved-tree rule cannot
        order a name that is not a version, so it says so."""
        repo.commit("0.1.0")
        repo.tag("claude-vnext", lightweight=True)
        with pytest.raises(PayloadError, match="claude-vnext"):
            claude_build.release_tags(repo.root)
        with pytest.raises(PayloadError, match="claude-vnext"):
            repo.problems()

    def test_a_tag_on_a_commit_without_the_tree_is_refused(self, tmp_path):
        """Read as "nothing to compare", the mistake disarms the rule for every
        later edit: the newest release no longer holds anything to anything."""
        mistagged = Repo(tmp_path / "repo", with_tree=False)
        mistagged.commit("0.1.0", tag="claude-v0.1.0")
        mistagged.write_tree(HOOK, "#!/bin/sh\n")
        mistagged.commit("0.1.0")
        line = _one(mistagged.problems())
        assert "claude-v0.1.0" in line and "carries no packaging/claude/multi-llm/" in line


# ===========================================================================
# Against a pull request's target
# ===========================================================================

class TestAgainstABase:
    def test_a_same_version_edit_with_no_tag_at_all_is_refused(self, repo):
        """The rule's reach before the first tag: only the base can see it."""
        base = repo.commit("0.1.0")
        repo.commit("0.1.0", "b")
        assert repo.problems() == ()
        line = _one(repo.problems(base=base))
        assert f"moved since {base}" in line and "not newer than 0.1.0" in line

    def test_a_same_version_edit_on_an_untagged_bump_is_refused(self, repo):
        """master carries a bump whose release tag is not cut: the tag sees a newer
        version and passes, and only the base sees the edit on top of it."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        base = repo.commit("0.1.1", "b")
        repo.commit("0.1.1", "c")
        assert repo.problems() == ()
        assert f"moved since {base}" in _one(repo.problems(base=base))

    def test_a_bump_passes(self, repo):
        base = repo.commit("0.1.0")
        repo.commit("0.1.1", "b")
        assert repo.problems(base=base) == ()

    def test_an_unmoved_tree_passes(self, repo):
        base = repo.commit("0.1.0")
        repo.write("docs/notes.md", "a development edit\n")
        repo.commit("0.1.0")
        assert repo.problems(base=base) == ()

    def test_a_base_lacking_the_tree_is_the_first_release_commit(self, tmp_path):
        """The commit that adds the tree has nothing to be compared with."""
        repo = Repo(tmp_path / "repo", with_tree=False)
        base = repo.commit("0.1.0")
        repo.with_tree = True
        repo.write_tree(HOOK, "#!/bin/sh\n")
        repo.commit("0.1.0")
        assert repo.problems(base=base) == ()

    def test_a_base_that_does_not_resolve_raises_by_name(self, repo):
        """It is not a base lacking the tree. Read that way, a misspelt or unfetched
        target passes every edit, and the job goes green having compared nothing."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        with pytest.raises(PayloadError, match="origin/no-such-branch"):
            repo.problems(base="origin/no-such-branch")

    def test_both_comparisons_report(self, repo):
        repo.commit("0.1.0", tag="claude-v0.1.0")
        base = repo.commit("0.1.0", "b")
        repo.commit("0.1.0", "c")
        problems = repo.problems(base=base)
        assert len(problems) == 2, problems
        assert "moved since claude-v0.1.0" in problems[0]
        assert f"moved since {base}" in problems[1]


# ===========================================================================
# A clone that cannot answer
# ===========================================================================

class TestIncompleteClones:
    """A shallow clone fetches no older tag, so the empty case would pass it."""

    @pytest.fixture
    def origin(self, tmp_path) -> Repo:
        origin = Repo(tmp_path / "origin")
        origin.commit("0.1.0", tag="claude-v0.1.0")
        origin.commit("0.1.0", "b")
        return origin

    def test_a_complete_clone_sees_the_release(self, origin, tmp_path):
        clone = tmp_path / "full"
        _git(tmp_path, "clone", "-q", origin.root.as_uri(), str(clone))
        assert "moved since claude-v0.1.0" in _one(claude_build.release_problems(clone))

    def test_a_shallow_clone_is_refused(self, origin, tmp_path):
        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "-q", "--depth", "1", origin.root.as_uri(), str(clone))
        assert claude_build.release_tags(clone) == ()
        assert "is a shallow clone" in _one(claude_build.release_problems(clone))


# ===========================================================================
# The CLI, through main()
# ===========================================================================

class TestCheckReleaseCli:
    """Exit status is what the make target and the job read."""

    @pytest.fixture(autouse=True)
    def no_registry(self, monkeypatch):
        """``--check-release`` promises git alone, on a pull request's runner."""
        def refuse(*_args, **_kwargs):
            raise AssertionError("--check-release asked the registry")
        monkeypatch.setattr(release_history, "_run_npm", refuse)
        monkeypatch.setattr(release_history, "npm_versions", refuse)
        monkeypatch.setattr(release_history, "resolve_registry", refuse)

    def _run(self, repo: Repo, *extra: str) -> int:
        return claude_build.main(["--check-release", "--repo", str(repo.root), *extra])

    def test_a_passing_tree_exits_zero(self, repo, capsys):
        repo.commit("0.1.0", tag="claude-v0.1.0")
        assert self._run(repo) == 0
        assert "the Claude release rules hold" in capsys.readouterr().out

    def test_a_refusal_exits_one_on_stderr(self, repo, capsys):
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.commit("0.1.0", "b")
        assert self._run(repo) == 1
        captured = capsys.readouterr()
        assert "REFUSING: " in captured.err and "moved since claude-v0.1.0" in captured.err
        assert "the Claude release rules hold" not in captured.out

    def test_the_base_flag_reaches_the_rule(self, repo, capsys):
        base = repo.commit("0.1.0")
        repo.commit("0.1.0", "b")
        assert self._run(repo) == 0
        capsys.readouterr()
        assert self._run(repo, "--base", base) == 1
        assert f"moved since {base}" in capsys.readouterr().err

    def test_the_candidate_flag_judges_that_commit_rather_than_head(self, repo, capsys):
        """The pre-push hook judges the commit being pushed, which need not be HEAD."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        moved = repo.commit("0.1.0", "b")
        repo.commit("0.1.1", "c")
        assert self._run(repo) == 0
        capsys.readouterr()
        assert self._run(repo, "--candidate", moved) == 1
        assert "moved since claude-v0.1.0" in capsys.readouterr().err

    def test_a_malformed_tag_exits_one_naming_it(self, repo, capsys):
        repo.commit("0.1.0")
        repo.tag("claude-vnext", lightweight=True)
        assert self._run(repo) == 1
        assert "ERROR: tag claude-vnext does not name" in capsys.readouterr().err

    def test_a_base_that_does_not_resolve_exits_one_naming_it(self, repo, capsys):
        repo.commit("0.1.0")
        assert self._run(repo, "--base", "origin/no-such-branch") == 1
        assert "origin/no-such-branch" in capsys.readouterr().err

    def test_a_directory_that_is_no_repository_exits_one(self, tmp_path, capsys):
        assert claude_build.main(
            ["--check-release", "--repo", str(tmp_path)]) == 1
        assert "ERROR: " in capsys.readouterr().err


class TestCheckBumpCli:
    """``--check-bump``: this rule and the shared one, each refusal printed."""

    @pytest.fixture(autouse=True)
    def registry(self, monkeypatch):
        """npm has never heard of the package: its documented E404."""
        calls: List[List[str]] = []

        def absent(argv, _timeout):
            calls.append(list(argv))
            error = {"code": "E404", "summary": f"Not Found - {argv[2]}", "detail": ""}
            return subprocess.CompletedProcess(
                list(argv), 1, json.dumps({"error": error}), "npm error code E404\n")

        monkeypatch.setattr(release_history, "_run_npm", absent)
        monkeypatch.setattr(release_history, "npm_package_name", lambda: "fixture-package")
        self.calls = calls

    @staticmethod
    def _run(repo: Repo, *extra: str) -> int:
        return claude_build.main(["--check-bump", "--repo", str(repo.root), *extra])

    def test_a_bumped_candidate_exits_zero_having_read_the_registry(self, repo, capsys):
        repo.commit("0.1.0", tag="codex-v0.1.0")
        repo.commit("0.1.1", "b", tag="claude-v0.1.1")
        repo.commit("0.1.2", "c")
        assert self._run(repo) == 0
        out = capsys.readouterr().out
        assert "the Claude release rules hold" in out
        assert "claude-v0.1.1" in out
        assert self.calls, "--check-bump never asked the registry"

    def test_both_rules_report_a_same_version_shared_edit(self, repo, capsys):
        repo.commit("0.1.0", tag="codex-v0.1.0")
        repo.commit("0.1.1", "b", tag="claude-v0.1.1")
        repo.commit("0.1.1", "c")
        assert self._run(repo) == 1
        err = capsys.readouterr().err
        assert "moved since claude-v0.1.1" in err
        assert "claude-v0.1.1 carries 0.1.1 with another shared-payload digest" in err

    def test_a_hook_edit_is_this_rule_s_alone(self, repo, capsys):
        repo.commit("0.1.0", tag="codex-v0.1.0")
        repo.commit("0.1.1", "b", tag="claude-v0.1.1")
        repo.write_tree(HOOK, "edited after the release\n")
        repo.commit("0.1.1", "b")
        assert self._run(repo) == 1
        err = capsys.readouterr().err
        assert "moved since claude-v0.1.1" in err
        assert "shared-payload digest" not in err

    def test_a_shared_refusal_with_this_rule_satisfied_still_exits_one(self, repo, capsys):
        """The tree moved under a newer version, so this rule passes; the version
        is older than one another family released, so the shared rule does not."""
        repo.commit("0.1.0", tag="claude-v0.1.0")
        repo.commit("0.1.3", "b", tag="codex-v0.1.3")
        _git(repo.root, "tag", "codex-v0.1.0", "claude-v0.1.0")
        repo.commit("0.1.2", "c")
        assert self._run(repo) == 1
        err = capsys.readouterr().err
        assert "older than the newest release 0.1.3 (codex-v0.1.3)" in err
        assert "moved since" not in err

    def test_a_shallow_clone_is_refused_once(self, repo, tmp_path, capsys):
        repo.commit("0.1.0", tag="codex-v0.1.0")
        repo.commit("0.1.1", "b")
        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "-q", "--depth", "1", repo.root.as_uri(), str(clone))
        assert claude_build.main(["--check-bump", "--repo", str(clone)]) == 1
        assert capsys.readouterr().err.count("is a shallow clone") == 1

    def test_another_registry_exits_one_naming_it(self, repo, capsys):
        repo.commit("0.1.0", tag="codex-v0.1.0")
        assert self._run(repo, "--registry", "https://registry.example.invalid") == 1
        assert "registry.example.invalid" in capsys.readouterr().err


# ===========================================================================
# The CI wiring
# ===========================================================================

def _makefile() -> str:
    return (REPO_ROOT / "Makefile").read_text(encoding="utf-8")


def _recipe(target: str) -> str:
    match = re.search(rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)*)", _makefile(), re.M)
    assert match, f"no {target} target"
    return match.group(1)


def _package_step(job: str = "package") -> Optional[dict]:
    document = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
    for step in document["jobs"][job]["steps"]:
        if "make claude-package-check-ci" in (step.get("run") or ""):
            return step
    return None


class TestCiWiring:
    def test_the_package_job_runs_the_tier(self):
        step = _package_step()
        assert step is not None, "the package job does not run the Claude tier"
        assert step["run"].strip() == "make claude-package-check-ci"

    def test_the_tier_runs_beside_the_other_two(self):
        document = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
        runs = [step.get("run") for step in document["jobs"]["package"]["steps"]]
        for target in ("opencode-package-check-ci", "codex-package-check-ci",
                       "claude-package-check-ci"):
            assert f"make {target}" in runs, target

    def test_the_three_os_job_does_not(self):
        """The clean rebuild needs the Linux job's guarantees, as the other two
        tiers do. The plain rows in this file run there on their own."""
        assert _package_step("test") is None

    def test_the_base_is_the_pull_request_target_or_the_commit_before_a_push(self):
        expression = _package_step()["env"]["CLAUDE_RELEASE_BASE"]
        assert expression == (
            "${{ github.event_name == 'pull_request' "
            "&& format('origin/{0}', github.base_ref) || github.event.before }}"
        )

    def test_the_base_reaches_the_script_through_the_environment(self):
        """A branch name is text someone else chose, so no expression is
        interpolated into the shell script."""
        assert "${{" not in _package_step()["run"]

    def test_the_checkout_holds_the_tags_and_the_target(self):
        """``fetch-depth: 0`` is what fetches every tag and every branch. Without
        it the newest ``claude-v*`` tag and ``origin/<target>`` are simply absent."""
        document = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
        checkout = next(step for step in document["jobs"]["package"]["steps"]
                        if str(step.get("uses", "")).startswith("actions/checkout@"))
        assert checkout["with"]["fetch-depth"] == 0

    @pytest.mark.parametrize("target", [
        "claude-package-check-ci", "claude-package", "claude-check-release",
    ])
    def test_nothing_in_the_tier_asks_a_registry(self, target):
        recipe = _recipe(target)
        for word in ("check-bump", "npm", "registry", "REGISTRY"):
            assert word not in recipe, f"{target} names {word}"

    def test_the_rule_runs_before_the_rows(self):
        recipe = _recipe("claude-package-check-ci")
        assert recipe.index("claude-check-release") < recipe.index("CLAUDE_PACKAGE_PYTEST")

    def test_the_tier_runs_the_rule(self):
        assert "--check-release" in _recipe("claude-check-release")
        assert "claude-check-release" in _recipe("claude-package-check-ci")


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("make") is None,
    reason="the job that runs this target is Linux-only, and needs make",
)
class TestTheTargetsOverARealMake:
    """The recipes run for real, with the two tools they call replaced by a stand-in
    that logs its arguments, because the promise is about what reaches the builder."""

    @pytest.fixture
    def run(self, tmp_path):
        log = tmp_path / "argv.jsonl"
        stand_in = tmp_path / "stand_in.py"
        stand_in.write_text(
            "import json, sys\n"
            f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        command = f"{sys.executable} {stand_in}"

        def go(target: str, environment: dict, *make_args: str) -> List[List[str]]:
            env = {key: value for key, value in os.environ.items()
                   if key not in ("CLAUDE_RELEASE_BASE", "MAKEFLAGS", "MFLAGS", "ALLOW_DIRTY")}
            env.update(environment)
            done = subprocess.run(
                ["make", "--no-print-directory", target, f"CLAUDE_BUILD={command}",
                 f"CLAUDE_PACKAGE_PYTEST={command}", *make_args],
                cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
            )
            assert done.returncode == 0, done.stdout + done.stderr
            return [json.loads(line) for line in log.read_text("utf-8").splitlines()]

        return go

    @staticmethod
    def _release_call(calls: List[List[str]]) -> List[str]:
        matching = [call for call in calls if "--check-release" in call]
        assert len(matching) == 1, calls
        return matching[0]

    def test_the_tier_passes_the_base_through_its_recursive_make(self, run):
        calls = run("claude-package-check-ci", {"CLAUDE_RELEASE_BASE": "origin/master"})
        assert self._release_call(calls) == ["--check-release", "--base", "origin/master"]
        assert calls[0] == ["--require-no-diff"], "the clean rebuild runs first"

    def test_a_push_passes_no_base_at_all(self, run):
        """The empty string the expression gives on a push is not a ref to compare."""
        calls = run("claude-package-check-ci", {"CLAUDE_RELEASE_BASE": ""})
        assert self._release_call(calls) == ["--check-release"]

    def test_an_unset_base_passes_none_either(self, run):
        assert self._release_call(run("claude-check-release", {})) == ["--check-release"]

    def test_the_make_command_line_works_too(self, run):
        """How a maintainer runs the same check by hand."""
        calls = run("claude-check-release", {}, "CLAUDE_RELEASE_BASE=origin/master")
        assert self._release_call(calls) == ["--check-release", "--base", "origin/master"]

    @pytest.mark.parametrize("template", [
        "origin/`touch$IFS{sentinel}`",      # a valid ref name: only shell syntax
        "origin/$(touch$IFS{sentinel})",     # a valid ref name: the $(...) spelling
        "origin/$(shell touch {sentinel})",  # make's own function, were it expanded
    ], ids=["backticks", "command-substitution", "make-shell"])
    def test_a_hostile_branch_name_is_only_ever_an_argument(self, run, tmp_path, template):
        """``env:`` in the workflow keeps it out of the job's script; the recipe has
        to keep it out of make's text and the shell's too, or that is only moved."""
        sentinel = tmp_path / "executed"
        hostile = template.format(sentinel=sentinel)
        calls = run("claude-package-check-ci", {"CLAUDE_RELEASE_BASE": hostile})
        assert self._release_call(calls) == ["--check-release", "--base", hostile]
        assert not sentinel.exists(), "the branch name ran"


# ===========================================================================
# The two documents that state these rules
# ===========================================================================

README = REPO_ROOT / "packaging" / "claude" / "README.md"
HOSTS = REPO_ROOT / "docs" / "development" / "hosts.md"
BOTH_DOCUMENTS = pytest.mark.parametrize("document", [README, HOSTS], ids=["readme", "hosts"])


def _flat(path: Path) -> str:
    """The document on one line: a phrase may be wrapped anywhere."""
    return " ".join(path.read_text(encoding="utf-8").split())


class TestTheDocuments:
    """Each of these is a fact copied out of the code or a procedure a release
    depends on, so a change on either side has to reach the other."""

    @BOTH_DOCUMENTS
    def test_the_online_links_are_the_shipped_ones(self, document):
        text = document.read_text("utf-8")
        for name, source in (("providers", "SKILL.md"),
                             ("configuration", "instructions/init-config.md")):
            url = f"https://github.com/beastlabai/multi-llm-plugin/blob/master/docs/{name}.md"
            assert url in text, url
            assert url in (SKILL_ROOT / source).read_text("utf-8"), source

    def test_the_readme_quotes_the_budget_that_is_enforced(self):
        files, size = payload.DISTRIBUTION_BUDGETS["claude-code"]
        text = _flat(README)
        assert "DISTRIBUTION_BUDGETS" in text and "packaging/common/payload.py" in text
        assert f"{files} files and {size:,} unpacked bytes" in text

    @BOTH_DOCUMENTS
    def test_the_bump_procedure_names_the_whole_chain(self, document):
        text = _flat(document)
        for needle in (
            "skills/multi-llm/pyproject.toml", "skills/multi-llm/uv.lock",
            ".claude-plugin/plugin.json", "skills/multi-llm/hosts/capabilities.yaml",
            "packaging/opencode/package.json", "packaging/opencode/README.md",
            "hosts/render_capabilities.py", "make codex-regen", "make claude-regen",
            "one commit", "A root manifest bump alone",
        ):
            assert needle in text, needle

    @BOTH_DOCUMENTS
    def test_the_release_procedure_and_its_exception_are_stated(self, document):
        text = _flat(document)
        for needle in (
            "make claude-check-bump", "claude-v<version>", "immutable",
            "off `master`", "is the release",
            "any commit that moves `packaging/claude/multi-llm/` under an unmoved version "
            "fails the check",
            "unfinished payload changes stay off `master` until ready to release",
        ):
            assert needle in text, needle

    def test_the_readme_lists_every_command(self):
        text = README.read_text("utf-8")
        for target in ("claude-regen", "claude-package", "claude-package-check-ci",
                       "claude-check-release", "claude-check-bump"):
            assert f"`make {target}`" in text, target
            assert re.search(rf"^{target}:", _makefile(), re.M), target

    def test_the_refusal_points_at_a_file_that_says_what_to_do(self):
        """The moved-tree refusal sends a contributor to the README by path."""
        assert README.is_file()
        assert "## Bump procedure" in README.read_text("utf-8")
