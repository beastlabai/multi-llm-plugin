"""The shared-version release-history check, over fixture histories.

Scope: ``packaging/common/release_history.py`` and its two entry points,
``packaging/codex/build.py --check-bump`` and ``packaging/opencode/build.py
--check-bump`` (``make opencode-check-bump``), from
``plans/opencode-plugin-publish.md`` step 2. Beside
``test_codex_package.py::test_the_bump_rule_reads_git_both_ways`` in style: each
row builds a throwaway repository shaped like this one where the check looks
(the allowlist manifest, the version owner, a payload, Codex's checked-in
digest) and tags it the way a release history would be tagged.

**Both real CLIs are driven, in subprocesses.** Each run is a fresh interpreter
that loads :mod:`common.release_history`, replaces its one npm runner
(``_run_npm``) with the row's canned registry answer, then runs the builder's
``__main__`` with ``--check-bump --repo <fixture>``. ``--repo`` is a real CLI
option (the clone whose history is read); the npm stand-in is the test's alone
and no environment variable reaches it, so nothing a release path inherits can
fake a registry answer. A fresh interpreter per run also keeps the two builders'
module names (each imports its own ``release_gate``) out of each other's way.

**None of this touches the network or this repository's own tags.** The check
reads the real history only on the release paths: it is deliberately out of
the ``tests`` workflow (``TestPlacement``), because after a release ``master``
may change shipped bytes while still saying the released version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
PACKAGING = REPO_ROOT / "packaging"

if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

from common import payload, release_history  # noqa: E402

ENTRY_POINTS: Dict[str, Path] = {
    "codex": PACKAGING / "codex" / "build.py",
    "opencode": PACKAGING / "opencode" / "build.py",
}
BOTH = pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))

PACKAGE = release_history.npm_package_name()
REGISTRY = release_history.resolve_registry()
CODEX_DIGEST = "packaging/codex/multi-llm/dist/tree.sha256"

#: The fixture's allowlist: the same two headings the real manifest carries, so
#: :func:`payload.runtime_allowlist` parses it unmodified.
MANIFEST = """\
# fixture manifest
--- the product under test ---
skills/multi-llm/*.py
skills/multi-llm/pyproject.toml
skills/multi-llm/utils/**
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


class History:
    """A throwaway repository with this one's shape where the check looks.

    Each :meth:`commit` writes the version owner and the payload body, and moves
    Codex's checked-in ``dist/tree.sha256`` with them as ``make codex-regen``
    would, so Codex's own-tree rules see a realistic tree beside the shared one.
    ``.gitattributes`` pins ``eol=lf`` as the real one does, so a Windows
    checkout's bytes are the committed ones.
    """

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True)
        _git(root, "init", "-q")
        self.write(".gitattributes", "* text=auto eol=lf\n")
        self.write(payload.MANIFEST_RELATIVE, MANIFEST)
        self.write("skills/multi-llm/utils/helper.py", "HELPER = 1\n")
        self.write("skills/multi-llm/tests/test_rig.py", "# rig, not payload\n")

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def commit(self, version: str, body: str, *, tag: str = "", rig: str = "",
               lightweight: bool = False) -> str:
        self.write(release_history.VERSION_PATH,
                   f'[project]\nname = "multi-llm"\nversion = "{version}"\n')
        self.write("skills/multi-llm/multi_llm.py", f"BODY = {body!r}\n")
        self.write(CODEX_DIGEST,
                   hashlib.sha256(f"{version}:{body}".encode()).hexdigest() + "\n")
        if rig:
            self.write("skills/multi-llm/tests/test_rig.py", rig)
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

    def checkout(self, ref: str) -> None:
        _git(self.root, "checkout", "-q", ref)

    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD").strip()

    def digest(self, ref: str = "HEAD") -> str:
        return release_history.shared_payload_digest(
            self.root, release_history.resolve_commit(self.root, ref))


# ---------------------------------------------------------------------------
# The registry, canned in npm's measured shapes (npm 10.9.3, 2026-10-25)
# ---------------------------------------------------------------------------

def listed(*versions: str) -> Dict[str, object]:
    return {"exit": 0, "stdout": json.dumps(list(versions), indent=2) + "\n", "stderr": ""}


def absent(package: str = PACKAGE) -> Dict[str, object]:
    """What ``npm view <never-published> versions --json`` prints, verbatim."""
    error = {
        "code": "E404",
        "summary": f"Not Found - GET {REGISTRY}/{package} - Not found",
        "detail": f"'{package}@*' is not in this registry.\n\nNote that you can "
                  "also install from a\ntarball, folder, http url, or git url.",
    }
    return {"exit": 1, "stdout": json.dumps({"error": error}, indent=2) + "\n",
            "stderr": "npm error code E404\n"}


def failed(code: str, summary: str) -> Dict[str, object]:
    error = {"code": code, "summary": summary, "detail": ""}
    return {"exit": 1, "stdout": json.dumps({"error": error}, indent=2) + "\n",
            "stderr": f"npm error code {code}\n"}


def runner(answer: Dict[str, object], calls: Optional[List] = None):
    def run(argv: Sequence[str], timeout: float):
        if calls is not None:
            calls.append((tuple(argv), timeout))
        return subprocess.CompletedProcess(
            list(argv), answer["exit"], answer["stdout"], answer["stderr"])
    return run


_DRIVER = r'''
import json, runpy, subprocess, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
sys.path.insert(0, spec["packaging"])
from common import release_history

def answer(argv, timeout):
    with open(spec["log"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": list(argv), "timeout": timeout}) + "\n")
    return subprocess.CompletedProcess(list(argv), spec["exit"], spec["stdout"], spec["stderr"])

release_history._run_npm = answer
sys.argv = [spec["builder"], *spec["args"]]
runpy.run_path(spec["builder"], run_name="__main__")
'''


class Run:
    def __init__(self, done: subprocess.CompletedProcess, log: Path):
        self.code = done.returncode
        self.stdout = done.stdout
        self.stderr = done.stderr
        self.npm = ([json.loads(line) for line in log.read_text("utf-8").splitlines()]
                    if log.is_file() else [])

    @property
    def refusals(self) -> List[str]:
        return [line[len("REFUSING: "):] for line in self.stderr.splitlines()
                if line.startswith("REFUSING: ")]

    def __repr__(self) -> str:
        return f"exit {self.code}\n--- stdout\n{self.stdout}\n--- stderr\n{self.stderr}"


@pytest.fixture
def cli(tmp_path):
    """``cli(entry, repo, answer, *extra)``: the real ``build.py --check-bump``."""
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    count = [0]

    def run(entry: str, repo: Path, answer: Dict[str, object], *extra: str) -> Run:
        count[0] += 1
        log = tmp_path / f"npm-{count[0]}.jsonl"
        spec = tmp_path / f"spec-{count[0]}.json"
        spec.write_text(json.dumps({
            "packaging": str(PACKAGING), "builder": str(ENTRY_POINTS[entry]),
            "args": ["--check-bump", "--repo", str(repo), *extra],
            "log": str(log), **answer,
        }), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(driver), str(spec)], cwd=str(tmp_path),
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        return Run(done, log)

    return run


def _refused(result: Run, *needles: str) -> None:
    assert result.code == 1, result
    joined = "\n".join(result.refusals)
    for needle in needles:
        assert needle in joined, f"{needle!r} not refused:\n{result!r}"


def _passed(result: Run) -> None:
    assert result.code == 0 and not result.refusals, result


# ===========================================================================
# The digest: one commit's allowlist over one commit's bytes
# ===========================================================================

class TestSharedPayloadDigest:
    def test_it_is_the_working_tree_rule_on_a_clean_checkout(self, tmp_path):
        """:func:`payload.tree_digest` over :func:`payload.expected_payload`'s
        files, as a clean checkout of the commit stages them."""
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a")
        files = payload.expected_payload(h.root)
        assert files == ("multi_llm.py", "pyproject.toml", "utils/helper.py")
        staged = tmp_path / "staged"
        for name in files:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(h.root / "skills" / "multi-llm" / name, target)
        assert h.digest() == payload.tree_digest(staged)

    def test_it_never_reads_the_working_tree(self, tmp_path):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a")
        before = h.digest()
        h.write("skills/multi-llm/multi_llm.py", "BODY = 'uncommitted'\n")
        h.write(payload.MANIFEST_RELATIVE, MANIFEST.replace(
            "skills/multi-llm/*.py\n", "skills/multi-llm/*.py\nskills/multi-llm/tests/**\n"))
        assert h.digest() == before

    def test_only_admitted_files_move_it(self, tmp_path):
        """The rig, Codex's own tree and OpenCode's manifest are outside it."""
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a")
        before = h.digest()
        h.write("skills/multi-llm/tests/test_rig.py", "# a rig edit\n")
        h.write(CODEX_DIGEST, "f" * 64 + "\n")
        h.write("packaging/opencode/package.json", '{"name": "x", "version": "9.9.9"}\n')
        _git(h.root, "add", "-A")
        _git(h.root, "commit", "-q", "-m", "distribution files only")
        assert h.digest() == before
        h.write("skills/multi-llm/utils/helper.py", "HELPER = 2\n")
        _git(h.root, "commit", "-q", "-am", "a shared edit")
        assert h.digest() != before

    def test_the_allowlist_is_read_from_the_same_commit(self, tmp_path):
        """A commit whose manifest admits more digests more; the older commit
        still digests what *its* manifest admitted."""
        h = History(tmp_path / "repo")
        h.write("skills/multi-llm/notes.txt", "tracked, not yet shipped\n")
        first = h.commit("0.1.0", "a")
        first_digest = h.digest()
        h.write(payload.MANIFEST_RELATIVE, MANIFEST.replace(
            "skills/multi-llm/pyproject.toml\n",
            "skills/multi-llm/pyproject.toml\nskills/multi-llm/notes.txt\n"))
        _git(h.root, "commit", "-q", "-am", "ship the notes")
        assert h.digest() != first_digest
        assert h.digest(first) == first_digest

    def test_an_admitted_symlink_is_refused(self, tmp_path):
        """Staged as a git entry, not created on disk, so it runs everywhere."""
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a")
        blob = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"], cwd=str(h.root), input="helper.py",
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        _git(h.root, "update-index", "--add", "--cacheinfo",
             f"120000,{blob},skills/multi-llm/utils/link.py")
        _git(h.root, "commit", "-q", "-m", "a symlink")
        with pytest.raises(release_history.ReleaseHistoryError, match="link.py"):
            h.digest()


# ===========================================================================
# Versions and the registry, unit-level
# ===========================================================================

class TestVersions:
    def test_precedence_is_semver_s(self):
        ordered = ["0.1.0-rc.1", "0.1.0-rc.2", "0.1.0-rc.10", "0.1.0-rc.a", "0.1.0",
                   "0.1.1", "0.2.0", "1.0.0"]
        assert sorted(reversed(ordered), key=release_history.version_key) == ordered

    def test_build_metadata_is_not_a_new_version(self):
        assert release_history.release_version("0.1.2+rebuild") == "0.1.2"
        assert (release_history.version_key("0.1.2+rebuild")
                == release_history.version_key("0.1.2"))

    def test_the_codex_build_reads_the_version_through_the_same_parse(self):
        text = (PACKAGING / "codex" / "build.py").read_text(encoding="utf-8")
        assert "release_history.version_in(" in text
        assert r"^\[project\]" not in text, "a second pyproject parse is back in codex/build.py"


class TestRegistry:
    def test_the_query_names_the_single_registry_and_bounds_itself(self):
        calls: List = []
        release_history.npm_versions(PACKAGE, REGISTRY, runner=runner(listed("0.1.1"), calls))
        (argv, timeout), = calls
        assert argv[:4] == ("npm", "view", PACKAGE, "versions")
        assert argv[argv.index("--registry") + 1] == REGISTRY
        assert "--json" in argv and "--prefer-online" in argv
        retries = [a for a in argv if a.startswith("--fetch-retries=")]
        assert retries and 0 <= int(retries[0].split("=")[1]) <= 3, argv
        assert any(a.startswith("--fetch-timeout=") for a in argv), argv
        assert 0 < timeout <= 120

    def test_a_list_is_the_versions(self):
        answer = release_history.npm_versions(
            PACKAGE, REGISTRY, runner=runner(listed("0.1.1", "0.1.2")))
        assert answer.versions == ("0.1.1", "0.1.2") and not answer.absent

    def test_a_bare_string_is_one_version(self):
        bare = {"exit": 0, "stdout": '"0.1.1"\n', "stderr": ""}
        assert release_history.npm_versions(
            PACKAGE, REGISTRY, runner=runner(bare)).versions == ("0.1.1",)

    def test_e404_on_the_package_name_is_no_versions(self):
        answer = release_history.npm_versions(PACKAGE, REGISTRY, runner=runner(absent()))
        assert answer.absent and answer.versions == ()

    @pytest.mark.parametrize("answer, code", [
        (failed("E500", f"500 Internal Server Error - GET {REGISTRY}/{PACKAGE}"), "E500"),
        (failed("E403", f"403 Forbidden - GET {REGISTRY}/{PACKAGE}"), "E403"),
        (failed("E429", f"429 Too Many Requests - GET {REGISTRY}/{PACKAGE}"), "E429"),
        (failed("ECONNREFUSED", "FetchError: connect ECONNREFUSED"), "ECONNREFUSED"),
        (absent("some-other-package"), "E404"),
        ({"exit": 1, "stdout": "", "stderr": "npm error code ETIMEDOUT\n"}, "ETIMEDOUT"),
        ({"exit": 0, "stdout": "{}", "stderr": ""}, "exited 0"),
        ({"exit": 0, "stdout": '["latest"]', "stderr": ""}, "not semver"),
    ], ids=["E500", "E403", "E429", "ECONNREFUSED", "E404-other-name", "stderr-only",
            "no-list", "not-semver"])
    def test_anything_else_is_an_unknown(self, answer, code):
        with pytest.raises(release_history.ReleaseHistoryError, match=re.escape(code)):
            release_history.npm_versions(PACKAGE, REGISTRY, runner=runner(answer))

    @pytest.mark.skipif(os.name == "nt", reason="the stand-in npm is a POSIX shell script")
    def test_the_real_runner_reads_a_real_process(self, tmp_path, monkeypatch):
        """``_run_npm`` itself: PATH resolution, the scratch cwd, the budget."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        npm = bin_dir / "npm"
        npm.write_text(
            "#!/bin/sh\n"
            'if [ "$NPM_STUB_SLEEP" ]; then sleep "$NPM_STUB_SLEEP"; fi\n'
            "printf '[\"0.1.1\"]\\n'\n", encoding="utf-8")
        npm.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        assert release_history.npm_versions(PACKAGE, REGISTRY).versions == ("0.1.1",)
        monkeypatch.setenv("NPM_STUB_SLEEP", "5")
        with pytest.raises(release_history.ReleaseHistoryError, match="did not answer"):
            release_history._run_npm(release_history.npm_view_argv(PACKAGE, REGISTRY), 0.5)  # noqa: SLF001

    def test_npm_missing_is_an_unknown(self, monkeypatch):
        monkeypatch.setattr(release_history.shutil, "which", lambda name: None)
        with pytest.raises(release_history.ReleaseHistoryError, match="not on PATH"):
            release_history.npm_versions(PACKAGE, REGISTRY)


# ===========================================================================
# The histories the plan names, through both real CLIs
# ===========================================================================

class TestNewestReleaseOnlyUnderOpenCode:
    """``codex-v0.1.0``, then ``opencode-v0.1.1``; Codex never released 0.1.1."""

    @pytest.fixture
    def history(self, tmp_path):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.1", "b", tag="opencode-v0.1.1")
        h.commit("0.1.1", "c")  # a shared fix on master, still saying 0.1.1
        return h

    @BOTH
    def test_a_later_shared_edit_at_the_same_version_is_refused(self, history, cli, entry):
        result = cli(entry, history.root, listed("0.1.1"))
        _refused(result,
                 "opencode-v0.1.1 carries 0.1.1 with another shared-payload digest",
                 history.digest("opencode-v0.1.1"), history.digest())

    def test_codex_s_own_rules_alone_would_have_passed_it(self, history, cli):
        """The gap the shared check closes: the newest codex-v* tag is 0.1.0,
        so the own-tree rule sees a bumped version and is satisfied."""
        refusals = cli("codex", history.root, listed("0.1.1")).refusals
        assert len(refusals) == 1 and "opencode-v0.1.1" in refusals[0], refusals

    @BOTH
    def test_the_tagged_commit_itself_passes(self, history, cli, entry):
        history.checkout("opencode-v0.1.1")
        _passed(cli(entry, history.root, listed("0.1.1")))

    def test_the_tagged_commit_passes_as_a_candidate_too(self, history, cli):
        _passed(cli("opencode", history.root, listed("0.1.1"), "--candidate", "opencode-v0.1.1"))

    @BOTH
    def test_a_bump_to_the_next_version_passes(self, history, cli, entry):
        history.commit("0.1.2", "c")
        _passed(cli(entry, history.root, listed("0.1.1")))


class TestNewestReleaseOnlyUnderCodex:
    """The mirror: ``codex-v0.1.1`` exists, OpenCode never published."""

    @pytest.fixture
    def history(self, tmp_path):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.1", "b", tag="codex-v0.1.1")
        return h

    @BOTH
    def test_a_later_shared_edit_at_the_same_version_is_refused(self, history, cli, entry):
        history.commit("0.1.1", "c")
        result = cli(entry, history.root, absent())
        _refused(result, "codex-v0.1.1 carries 0.1.1 with another shared-payload digest",
                 history.digest("codex-v0.1.1"), history.digest())

    def test_codex_s_own_tree_rule_still_runs_beside_it(self, history, cli):
        history.commit("0.1.1", "c")
        _refused(cli("codex", history.root, absent()),
                 "the plugin digest moved since codex-v0.1.1",
                 "codex-v0.1.1 carries 0.1.1")

    @BOTH
    def test_the_tagged_commit_itself_passes(self, history, cli, entry):
        history.commit("0.1.1", "c")
        history.checkout("codex-v0.1.1")
        _passed(cli(entry, history.root, absent()))

    @BOTH
    def test_a_same_payload_release_by_the_other_family_passes(self, history, cli, entry):
        """OpenCode may release 0.1.1 from a commit whose shared payload is
        codex-v0.1.1's: a rig-only commit, here."""
        history.commit("0.1.1", "b", rig="# a rig edit\n", tag="opencode-v0.1.1")
        _passed(cli(entry, history.root, listed("0.1.1")))

    @BOTH
    def test_a_bump_to_the_next_version_passes(self, history, cli, entry):
        history.commit("0.1.2", "c")
        _passed(cli(entry, history.root, absent()))


@BOTH
def test_an_unreleased_version_older_than_the_newest_release_is_refused(tmp_path, cli, entry):
    h = History(tmp_path / "repo")
    h.commit("0.1.0", "a", tag="codex-v0.1.0")
    h.commit("0.1.2", "b", tag="opencode-v0.1.2")  # 0.1.1 skipped: legal
    h.commit("0.1.1", "c")
    _refused(cli(entry, h.root, listed("0.1.2")),
             "the candidate's version 0.1.1 is unreleased but older than the newest "
             "release 0.1.2 (opencode-v0.1.2)")


class TestNpmVersionWithoutATag:
    @pytest.fixture
    def history(self, tmp_path):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        return h

    @BOTH
    def test_it_is_refused_by_name(self, history, cli, entry):
        history.commit("0.1.2", "c")
        _refused(cli(entry, history.root, listed("0.1.1")),
                 f"npm lists {PACKAGE}@0.1.1", "no opencode-v0.1.1 tag names it")

    def test_a_candidate_at_that_version_cannot_be_compared(self, history, cli):
        history.commit("0.1.1", "c")
        _refused(cli("opencode", history.root, listed("0.1.1")),
                 "npm lists but no tag names")

    def test_a_codex_tag_does_not_account_for_an_npm_version(self, history, cli):
        """npm publishes from ``opencode-v*`` tags alone."""
        history.commit("0.1.1", "b", tag="codex-v0.1.1")
        _refused(cli("opencode", history.root, listed("0.1.1")),
                 "no opencode-v0.1.1 tag names it")


class TestRegistryErrors:
    @pytest.fixture
    def history(self, tmp_path):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.1", "b")
        return h

    @BOTH
    @pytest.mark.parametrize("answer, code", [
        (failed("E500", f"500 Internal Server Error - GET {REGISTRY}/{PACKAGE}"), "E500"),
        (failed("ECONNREFUSED", "FetchError: connect ECONNREFUSED"), "ECONNREFUSED"),
        (absent("another-package"), "E404"),
    ], ids=["E500", "ECONNREFUSED", "E404-other-name"])
    def test_an_error_other_than_e404_refuses(self, history, cli, entry, answer, code):
        result = cli(entry, history.root, answer)
        _refused(result, f"npm could not list {PACKAGE} versions", f"code {code}")
        assert result.npm and result.npm[0]["argv"][:4] == ["npm", "view", PACKAGE, "versions"]

    @BOTH
    def test_e404_is_the_green_answer(self, history, cli, entry):
        result = cli(entry, history.root, absent())
        _passed(result)
        assert f"{PACKAGE} is not on {REGISTRY} (E404)" in result.stdout

    def test_another_registry_is_refused_naming_both(self, history, cli):
        result = cli("opencode", history.root, absent(), "--registry", "https://example.invalid")
        _refused(result, "https://example.invalid", REGISTRY)
        assert not result.npm, "npm was asked before the registry was checked"


class TestFirstRelease:
    """Only ``codex-v0.1.0``; ``opencode-multi-llm`` is not on npm at all."""

    @pytest.fixture
    def history(self, tmp_path):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        return h

    @BOTH
    def test_a_bumped_candidate_passes(self, history, cli, entry):
        history.commit("0.1.1", "b")
        result = cli(entry, history.root, absent())
        _passed(result)
        assert "0.1.1 is newer than every released version (newest: 0.1.0)" in result.stdout

    @BOTH
    def test_an_unbumped_candidate_with_other_shared_bytes_is_refused(self, history, cli, entry):
        history.commit("0.1.0", "b")
        _refused(cli(entry, history.root, absent()),
                 "codex-v0.1.0 carries 0.1.0 with another shared-payload digest",
                 history.digest("codex-v0.1.0"), history.digest())

    @BOTH
    def test_an_unbumped_candidate_with_the_same_shared_bytes_passes(self, history, cli, entry):
        history.commit("0.1.0", "a", rig="# a rig-only edit\n")
        _passed(cli(entry, history.root, absent()))


@BOTH
def test_a_tag_whose_name_is_not_its_commit_s_version_is_refused(tmp_path, cli, entry):
    h = History(tmp_path / "repo")
    h.commit("0.1.0", "a", tag="codex-v0.1.0")
    h.commit("0.1.1", "b", tag="opencode-v0.1.2", lightweight=True)
    h.commit("0.1.3", "c")
    _refused(cli(entry, h.root, listed("0.1.2")),
             "opencode-v0.1.2 names 0.1.2, but its commit",
             f"{release_history.VERSION_PATH} says 0.1.1")


class TestTwoTagsOfOneVersion:
    @BOTH
    def test_different_shared_payloads_are_refused(self, tmp_path, cli, entry):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.1", "b", tag="codex-v0.1.1")
        h.commit("0.1.1", "c", tag="opencode-v0.1.1")
        first, second = h.digest("codex-v0.1.1"), h.digest("opencode-v0.1.1")
        h.commit("0.1.2", "d")
        _refused(cli(entry, h.root, listed("0.1.1")),
                 "version 0.1.1 names more than one shared payload",
                 f"codex-v0.1.1 (commit", first, f"opencode-v0.1.1 (commit", second)

    @BOTH
    def test_one_shared_payload_on_two_commits_passes(self, tmp_path, cli, entry):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.1", "b", tag="codex-v0.1.1")
        h.commit("0.1.1", "b", rig="# packaging-only follow-up\n", tag="opencode-v0.1.1")
        h.commit("0.1.2", "d")
        _passed(cli(entry, h.root, listed("0.1.1")))


class TestIncompleteClones:
    """A clone that cannot answer for the history is refused, never read as
    one with no releases: that is the state Codex's own rules pass."""

    @pytest.fixture
    def origin(self, tmp_path):
        h = History(tmp_path / "origin")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.0", "b")  # unbumped, so a complete history refuses it
        return h

    @BOTH
    def test_a_complete_clone_sees_the_release(self, origin, cli, entry, tmp_path):
        clone = tmp_path / "full"
        _git(tmp_path, "clone", "-q", origin.root.as_uri(), str(clone))
        _refused(cli(entry, clone, absent()), "codex-v0.1.0 carries 0.1.0")

    @BOTH
    def test_a_shallow_clone_is_refused(self, origin, cli, entry, tmp_path):
        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "-q", "--depth", "1", origin.root.as_uri(), str(clone))
        _refused(cli(entry, clone, absent()), "is a shallow clone")

    @BOTH
    def test_a_tagless_clone_is_refused(self, origin, cli, entry, tmp_path):
        clone = tmp_path / "tagless"
        _git(tmp_path, "clone", "-q", "--no-tags", origin.root.as_uri(), str(clone))
        _refused(cli(entry, clone, absent()), "has no codex-v0.1.0 tag")

    def test_codex_s_own_rules_pass_a_tagless_clone(self, origin, tmp_path):
        """Why the gap check exists: measured, not assumed."""
        import importlib.util

        clone = tmp_path / "tagless"
        _git(tmp_path, "clone", "-q", "--no-tags", origin.root.as_uri(), str(clone))
        spec = importlib.util.spec_from_file_location(
            "codex_build_for_release_history", str(ENTRY_POINTS["codex"]))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.release_tags(clone) == ()
        assert module.bump_history_problems(clone) == ()

    def test_a_malformed_tag_is_refused_not_skipped(self, tmp_path, cli):
        h = History(tmp_path / "repo")
        h.commit("0.1.0", "a", tag="codex-v0.1.0")
        h.commit("0.1.1", "b", tag="opencode-vnext", lightweight=True)
        _refused(cli("opencode", h.root, absent()),
                 "tag opencode-vnext does not name a strict semver version")


# ===========================================================================
# Where the check runs, and where it must not
# ===========================================================================

def _makefile_recipe(makefile: str, target: str) -> str:
    match = re.search(rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)*)", makefile, re.M)
    assert match, f"no {target} target"
    return match.group(1)


class TestPlacement:
    MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    def test_opencode_check_bump_runs_the_entry_point_on_the_single_registry(self):
        recipe = _makefile_recipe(self.MAKEFILE, "opencode-check-bump")
        assert "$(OPENCODE_PKG)/build.py --check-bump" in recipe
        assert "--registry $(OPENCODE_REGISTRY)" in recipe
        assert "ALLOW_DIRTY" not in recipe and "$(OPENCODE_BUILD)" not in recipe

    def test_it_is_phony_documented_and_follows_the_registry_target(self):
        joined = self.MAKEFILE.replace("\\\n", " ")
        phony = " ".join(re.findall(r"^\.PHONY:(.*)$", joined, re.M))
        assert re.search(r"(?<![\w-])opencode-check-bump(?![\w-])", phony)
        assert "make opencode-check-bump" in self.MAKEFILE
        assert (self.MAKEFILE.index("\nopencode-print-registry:")
                < self.MAKEFILE.index("\nopencode-check-bump:"))

    def test_the_tests_workflow_never_runs_it(self):
        """A legal same-version edit on master must not turn it red."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
        assert "--check-bump" not in workflow and "check-bump" not in workflow
        for target in ("opencode-package-check-ci", "codex-package-check-ci"):
            assert target in workflow
            recipe = _makefile_recipe(self.MAKEFILE, target)
            assert "check-bump" not in recipe, f"{target} runs the release-history check"
