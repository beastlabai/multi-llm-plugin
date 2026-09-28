"""`write_review_diff`: the scope manifest and per-file diffs a code review reads.

A code reviewer used to be told only to run `git diff`, and a lane with no shell
never saw a diff at all (plan: review-run-scope-salvage-and-failure-visibility
§2.1). The orchestrator now writes `review-scope.md` plus one whole diff per
source or test file into the phase dir before dispatch (D2.2). Every test here
runs against a real temp git repository with the user's own git config shut
out, because the property under test is "the file holds exactly the bytes git
prints", and a `color.ui=always` in someone's `~/.gitconfig` would change them.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import git_utils  # noqa: E402
from utils.git_utils import (  # noqa: E402
    DIFF_COVERAGE_COMPLETE,
    DIFF_COVERAGE_PARTIAL,
    REVIEW_SCOPE_FILENAME,
    WRITTEN_BY_REAGGREGATE,
    ReviewDiffContainmentError,
    intent_to_add_untracked,
    load_review_scope,
    read_review_scope,
    write_review_diff,
)


# ---------------------------------------------------------------------------
# A hermetic repository
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
    ).stdout


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)


def _write(repo: Path, rel: str, text: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with a base commit, cwd'd into (as `review-code` runs)."""
    for key, value in {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _commit(root, "root")
    monkeypatch.chdir(root)
    return root


def _out(repo: Path) -> Path:
    """The phase dir, inside the repository the way a plan's output dir is."""
    return repo / "plans" / "feature" / "code-review"


_ENTRY = re.compile(r"^(\d+)\. (`+) ?(.*?) ?\2")


def _entries(manifest: str) -> dict:
    """``{path: entry line}`` for every numbered (source/test) entry."""
    found = {}
    for line in manifest.splitlines():
        match = _ENTRY.match(line)
        if match:
            found[match.group(3)] = line
    return found


def _diff_files(out: Path) -> list:
    return sorted(p.name for p in (out / "diff").iterdir() if p.name != ".gitignore")


# ---------------------------------------------------------------------------
# What the set holds
# ---------------------------------------------------------------------------


class TestGroupsAndCollapsing:
    def test_the_four_groups_and_collapsed_directories(self, repo, capsys):
        _write(repo, ".gitattributes", "gen/** linguist-generated\n")
        for rel in ("src/app.py", "tests/test_app.py", "tests/fixtures/opencode/a.json",
                    "tests/fixtures/opencode/b.json", "gen/mirror.py"):
            _write(repo, rel, "one\n")
        _commit(repo, "base")
        for rel in ("src/app.py", "tests/test_app.py", "tests/fixtures/opencode/a.json",
                    "tests/fixtures/opencode/b.json", "gen/mirror.py"):
            _write(repo, rel, "one\ntwo\n")

        result = write_review_diff(
            "HEAD", ["gen/mirror.py", "tests/fixtures/opencode/b.json",
                     "tests/fixtures/opencode/a.json", "tests/test_app.py", "src/app.py"],
            _out(repo),
        )

        manifest = (_out(repo) / REVIEW_SCOPE_FILENAME).read_text(encoding="utf-8")
        assert result.coverage == DIFF_COVERAGE_COMPLETE
        assert "## Source (1 file)" in manifest
        assert "## Tests (1 file)" in manifest
        assert "## Fixtures (2 files, collapsed)" in manifest
        assert "## Generated (1 file, collapsed)" in manifest
        # One line per directory, with its file count, and no diff file.
        assert ("- `tests/fixtures/opencode/` — 2 files (+2 -0) — no diff: "
                "collapsed (fixtures)") in manifest
        assert "- `gen/` — 1 file (+1 -0) — no diff: collapsed (generated)" in manifest
        assert _diff_files(_out(repo)) == ["1-src-app.py.diff", "2-tests-test_app.py.diff"]
        assert "[DIFF_" not in capsys.readouterr().out

    def test_a_configured_collapse_glob_makes_a_lockfile_generated(self, repo):
        _write(repo, "poetry.lock", "a\n")
        _write(repo, "app.py", "a\n")
        _commit(repo, "base")
        _write(repo, "poetry.lock", "b\n")
        _write(repo, "app.py", "b\n")

        write_review_diff(
            "HEAD", ["poetry.lock", "app.py"], _out(repo),
            {"generated": ["**/*.lock"]},
        )

        manifest = (_out(repo) / REVIEW_SCOPE_FILENAME).read_text(encoding="utf-8")
        assert "- `./` — 1 file (+1 -1) — no diff: collapsed (generated)" in manifest
        assert "poetry.lock" not in _entries(manifest)
        assert _diff_files(_out(repo)) == ["1-app.py.diff"]

    def test_no_entry_lacks_both_a_diff_path_and_a_reason(self, repo):
        _write(repo, "src/changed.py", "a\n")
        _write(repo, "src/untouched.py", "a\n")
        _write(repo, "tests/fixtures/f.json", "a\n")
        _commit(repo, "base")
        _write(repo, "src/changed.py", "b\n")
        _write(repo, "tests/fixtures/f.json", "b\n")

        write_review_diff(
            "HEAD", ["src/changed.py", "src/untouched.py", "tests/fixtures/f.json"],
            _out(repo),
        )

        manifest = (_out(repo) / REVIEW_SCOPE_FILENAME).read_text(encoding="utf-8")
        entries = _entries(manifest)
        assert set(entries) == {"src/changed.py", "src/untouched.py"}
        for line in entries.values():
            assert "`diff/" in line or " — no diff: " in line or " — omitted: " in line
        assert "no diff: no net change since HEAD" in entries["src/untouched.py"]
        assert "no diff: collapsed (fixtures)" in manifest

    def test_every_source_and_test_entry_names_its_diff_file_and_size(self, repo):
        for rel in ("lib/a.py", "lib/b.py", "tests/test_a.py"):
            _write(repo, rel, "x\n")
        _commit(repo, "base")
        for rel in ("lib/a.py", "lib/b.py", "tests/test_a.py"):
            _write(repo, rel, "y\n")

        write_review_diff("HEAD", ["lib/a.py", "lib/b.py", "tests/test_a.py"], _out(repo))

        out = _out(repo)
        manifest = (out / REVIEW_SCOPE_FILENAME).read_text(encoding="utf-8")
        for path, line in _entries(manifest).items():
            rel = re.search(r"`(diff/[^`]+)`, (\d+) bytes", line)
            assert rel, line
            assert (out / rel.group(1)).stat().st_size == int(rel.group(2))
            assert " — full: " in line


class TestDiffsAreWholeAndExact:
    def test_a_one_megabyte_diff_is_one_file_equal_to_git_diff(self, repo):
        _write(repo, "big.py", "")
        _commit(repo, "base")
        body = "".join(f"line {i:07d} " + "x" * 50 + "\n" for i in range(17000))
        _write(repo, "big.py", body)
        assert len(body) > 1_000_000

        result = write_review_diff("HEAD", ["big.py"], _out(repo))

        expected = _git_bytes(repo, "diff", "HEAD", "--", "big.py")
        written = (_out(repo) / "diff" / "1-big.py.diff").read_bytes()
        assert written == expected
        manifest = Path(result.scope_file).read_text(encoding="utf-8")
        assert f"— full: `diff/1-big.py.diff`, {len(expected)} bytes" in manifest

    def test_a_deleted_file_shows_every_removed_line_in_order(self, repo):
        lines = [f"removed line {i}" for i in range(3000)]
        _write(repo, "gone.py", "\n".join(lines) + "\n")
        _commit(repo, "base")
        (repo / "gone.py").unlink()

        write_review_diff("HEAD", ["gone.py"], _out(repo))

        diff = (_out(repo) / "diff" / "1-gone.py.diff").read_text(encoding="utf-8")
        removed = [ln[1:] for ln in diff.splitlines()
                   if ln.startswith("-") and not ln.startswith("---")]
        assert removed == lines
        assert "deleted file mode" in diff

    def test_a_new_untracked_file_is_diffed_via_intent_to_add(self, repo):
        _commit(repo, "base")
        _write(repo, "src/new_module.py", "def f():\n    return 1\n")

        with intent_to_add_untracked(["src/new_module.py"]):
            write_review_diff("HEAD", ["src/new_module.py"], _out(repo))

        diff = (_out(repo) / "diff" / "1-src-new_module.py.diff").read_text(encoding="utf-8")
        assert "new file mode" in diff
        assert "+def f():" in diff
        # The intent-to-add entry is gone again once the block exits.
        assert "src/new_module.py" in _git(repo, "ls-files", "--others", "--exclude-standard")

    def test_a_file_with_the_minus_diff_attribute(self, repo):
        _write(repo, ".gitattributes", "*.dat -diff\n")
        _write(repo, "blob.dat", "a\n")
        _commit(repo, "base")
        _write(repo, "blob.dat", "b\n")

        result = write_review_diff("HEAD", ["blob.dat"], _out(repo))

        manifest = Path(result.scope_file).read_text(encoding="utf-8")
        line = _entries(manifest)["blob.dat"]
        assert "(binary)" in line and " — full: " in line
        diff = (_out(repo) / "diff" / "1-blob.dat.diff").read_text(encoding="utf-8")
        assert "Binary files" in diff
        assert diff.encode("utf-8") == _git_bytes(repo, "diff", "HEAD", "--", "blob.dat")


class TestTheOrderIsReviewOrder:
    @pytest.mark.parametrize("given", [
        ["tests/test_a.py", "utils/z.py"],
        ["utils/z.py", "tests/test_a.py"],
    ])
    def test_source_is_numbered_before_tests_whatever_the_input_order(self, repo, given):
        _write(repo, "tests/test_a.py", "a\n")
        _write(repo, "utils/z.py", "a\n")
        _commit(repo, "base")
        _write(repo, "tests/test_a.py", "b\n")
        _write(repo, "utils/z.py", "b\n")

        write_review_diff("HEAD", given, _out(repo))

        assert _diff_files(_out(repo)) == ["1-utils-z.py.diff", "2-tests-test_a.py.diff"]


# ---------------------------------------------------------------------------
# Nothing leaves diff/
# ---------------------------------------------------------------------------


def _tree(root: Path) -> set:
    return {p.relative_to(root) for p in root.rglob("*")}


class TestHostilePathsStayInsideTheDiffDir:
    HOSTILE = [
        "dir/..sneaky..py",
        ".leading-dot.py",
        "with space/and more.py",
        "ünïcödé/файл.py",
        "deep/" + "n" * 120 + ".py",
    ]

    def test_every_diff_file_lands_directly_in_diff_with_a_unique_name(self, repo):
        for rel in self.HOSTILE:
            _write(repo, rel, "a\n")
        _commit(repo, "base")
        for rel in self.HOSTILE:
            _write(repo, rel, "b\n")
        before = _tree(repo)

        result = write_review_diff("HEAD", self.HOSTILE, _out(repo))

        out = _out(repo)
        names = _diff_files(out)
        assert len(names) == len(self.HOSTILE) == len(set(names))
        for name in names:
            assert re.fullmatch(r"\d+-[A-Za-z0-9._-]+\.diff", name), name
            assert len(name) < 100
        # Nothing new outside the phase dir: every added path is under it (or
        # is one of the directories leading to it).
        added = _tree(repo) - before - {Path("plans"), Path("plans/feature")}
        assert all(str(path).startswith("plans/feature/code-review") for path in added)
        assert result.coverage == DIFF_COVERAGE_COMPLETE
        # Each file still holds its own path's diff, found through the manifest.
        manifest = Path(result.scope_file).read_text(encoding="utf-8")
        for path, line in _entries(manifest).items():
            rel = re.search(r"`(diff/[^`]+)`", line).group(1)
            assert (out / rel).read_bytes() == _git_bytes(repo, "diff", "HEAD", "--", path)

    def test_a_path_outside_the_repository_fails_the_set_and_writes_nothing(
        self, repo, capsys
    ):
        _write(repo, "a.py", "a\n")
        _commit(repo, "base")
        _write(repo, "a.py", "b\n")

        result = write_review_diff("HEAD", ["a.py", "../outside.py"], _out(repo))

        # git refuses the pathspec, which is the `--numstat` failure D2.2
        # defines as failing the whole set: reported, and nothing written.
        assert "[DIFF_FAILED] git diff --numstat" in capsys.readouterr().out
        assert result.coverage == DIFF_COVERAGE_PARTIAL
        assert ("../outside.py", "git diff failed") in result.omitted
        assert _diff_files(_out(repo)) == []
        assert not (repo.parent / "outside.py").exists()

    def test_a_broken_slug_rule_raises_instead_of_writing_outside(self, repo, monkeypatch):
        _write(repo, "a.py", "a\n")
        _commit(repo, "base")
        _write(repo, "a.py", "b\n")
        monkeypatch.setattr(git_utils, "review_diff_slug", lambda _path: "../../escape")

        with pytest.raises(ReviewDiffContainmentError):
            write_review_diff("HEAD", ["a.py"], _out(repo))

        assert not list(repo.rglob("*escape*"))
        # The manifest is written last, so a raise part way leaves none.
        assert not (_out(repo) / REVIEW_SCOPE_FILENAME).exists()


# ---------------------------------------------------------------------------
# A git failure degrades the set, never the phase
# ---------------------------------------------------------------------------


class TestGitFailuresDegradeTheSet:
    @pytest.mark.parametrize("base_ref", ["no-such-ref", "", "HEAD~1"])
    def test_the_whole_set_is_omitted_and_the_call_returns(self, repo, capsys, base_ref):
        # `repo` has exactly one commit, so `HEAD~1` does not resolve either.
        _write(repo, "src/a.py", "a\n")
        _write(repo, "tests/test_a.py", "a\n")

        result = write_review_diff(base_ref, ["src/a.py", "tests/test_a.py"], _out(repo))

        out = capsys.readouterr().out
        assert "[DIFF_FAILED]" in out
        assert "[DIFF_PARTIAL] 2 of 2 source and test files" in out
        assert result.coverage == DIFF_COVERAGE_PARTIAL
        assert sorted(result.omitted) == [
            ("src/a.py", "git diff failed"), ("tests/test_a.py", "git diff failed"),
        ]
        manifest = Path(result.scope_file).read_text(encoding="utf-8")
        header_end = manifest.index("- Written by:")
        after_header = manifest[header_end:].splitlines()[2]
        assert after_header.startswith("Diff failed: ")
        assert "Diff coverage: partial — 2 of 2 source and test files omitted" in manifest
        for line in _entries(manifest).values():
            assert "omitted: git diff failed" in line
        assert _diff_files(_out(repo)) == []
        assert load_review_scope(_out(repo)).coverage == DIFF_COVERAGE_PARTIAL

    def test_one_failed_path_omits_that_path_alone(self, repo, capsys, monkeypatch):
        _write(repo, "a.py", "a\n")
        _write(repo, "b.py", "a\n")
        _commit(repo, "base")
        _write(repo, "a.py", "b\n")
        _write(repo, "b.py", "b\n")
        real = git_utils._git_diff_output

        def flaky(args, cwd):
            if args[0] == "diff" and args[-1] == "b.py" and "--numstat" not in args:
                return None, "fatal: simulated"
            return real(args, cwd)

        monkeypatch.setattr(git_utils, "_git_diff_output", flaky)

        result = write_review_diff("HEAD", ["a.py", "b.py"], _out(repo))

        out = capsys.readouterr().out
        assert "[DIFF_FAILED] git diff HEAD -- b.py: fatal: simulated" in out
        assert result.omitted == [("b.py", "git diff failed")]
        assert _diff_files(_out(repo)) == ["1-a.py.diff"]
        manifest = Path(result.scope_file).read_text(encoding="utf-8")
        assert "`git diff HEAD -- b.py` shows it" in manifest


# ---------------------------------------------------------------------------
# Retention, reuse and interruption
# ---------------------------------------------------------------------------


class TestAChangedScopeReplacesTheWholeSet:
    def test_no_earlier_file_survives_and_git_ignores_the_set(self, repo):
        for rel in ("src/a.py", "src/b.py", "src/c.py", "tests/test_c.py"):
            _write(repo, rel, "one\n")
        _commit(repo, "base")
        for rel in ("src/a.py", "src/b.py", "src/c.py", "tests/test_c.py"):
            _write(repo, rel, "two\n")
        out = _out(repo)
        first = write_review_diff(
            "HEAD", ["src/a.py", "src/b.py", "src/c.py", "tests/test_c.py"], out,
        )
        first_files = set(_diff_files(out))

        # Fewer paths, another order, and one changed content.
        _write(repo, "src/c.py", "three\n")
        second = write_review_diff("HEAD", ["tests/test_c.py", "src/c.py"], out)

        names = _diff_files(out)
        assert names == ["1-src-c.py.diff", "2-tests-test_c.py.diff"]
        assert not (first_files - set(names)) & set(os.listdir(out / "diff"))
        manifest = Path(second.scope_file).read_text(encoding="utf-8")
        named = set(re.findall(r"`diff/([^`]+)`", manifest))
        assert named == set(names)
        assert second.scope_id != first.scope_id
        assert (out / "diff" / ".gitignore").read_text(encoding="utf-8") == "*\n"
        ignored = _git(repo, "check-ignore", *[f"{out}/diff/{n}" for n in names])
        assert len(ignored.splitlines()) == len(names)
        untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
        assert "/diff/" not in untracked
        # The manifest itself stays visible to git, like report.md.
        assert REVIEW_SCOPE_FILENAME in untracked


class TestAnUnchangedTreeIsLeftUntouched:
    def test_the_same_scope_id_and_no_file_rewritten(self, repo, capsys):
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        first = write_review_diff("HEAD", ["src/a.py"], out)
        files = [out / REVIEW_SCOPE_FILENAME, *(out / "diff").iterdir()]
        for path in files:
            os.utime(path, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))

        second = write_review_diff("HEAD", ["src/a.py"], out)

        assert second.scope_id == first.scope_id
        assert second.written is False and first.written is True
        for path in files:
            assert path.stat().st_mtime_ns == 1_000_000_000_000_000_000, path
        assert "Review scope unchanged" in capsys.readouterr().out

    def test_the_written_by_of_the_manifest_on_disk_is_reported(self, repo):
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        write_review_diff("HEAD", ["src/a.py"], out, written_by=WRITTEN_BY_REAGGREGATE)

        again = write_review_diff("HEAD", ["src/a.py"], out)

        assert again.written_by == WRITTEN_BY_REAGGREGATE

    @pytest.mark.parametrize("edit", ["same size", "truncated"])
    def test_a_diff_file_changed_on_disk_is_rewritten(self, repo, capsys, edit):
        # The id is computed from the diff just generated; reviewers read the
        # file on disk. A reuse that only checked the file exists would hand
        # them bytes the id never described.
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        first = write_review_diff("HEAD", ["src/a.py"], out)
        target = out / "diff" / "1-src-a.py.diff"
        good = target.read_bytes()
        target.write_bytes(good[:-2] + b"X\n" if edit == "same size" else good[:10])
        capsys.readouterr()

        second = write_review_diff("HEAD", ["src/a.py"], out)

        assert second.scope_id == first.scope_id
        assert second.written is True
        assert target.read_bytes() == good == _git_bytes(repo, "diff", "HEAD", "--", "src/a.py")
        assert "Review scope unchanged" not in capsys.readouterr().out


class TestAnInterruptedWriteLeavesNoManifest:
    def test_a_raise_after_diff_is_emptied_leaves_no_review_scope_md(self, repo, monkeypatch):
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        write_review_diff("HEAD", ["src/a.py"], out)
        assert (out / REVIEW_SCOPE_FILENAME).exists()
        _write(repo, "src/a.py", "three\n")
        real = git_utils._write_diff_file

        def fail_on_a_diff(target, data):
            if str(target).endswith(".diff"):
                raise OSError("disk full")
            real(target, data)

        monkeypatch.setattr(git_utils, "_write_diff_file", fail_on_a_diff)

        with pytest.raises(OSError):
            write_review_diff("HEAD", ["src/a.py"], out)

        assert not (out / REVIEW_SCOPE_FILENAME).exists()
        assert load_review_scope(out) is None


class TestTheManifestReadsBack:
    def test_the_header_and_coverage_round_trip(self, repo):
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        result = write_review_diff("HEAD", ["src/a.py"], out, scope_source="tracked")

        header = read_review_scope(out / REVIEW_SCOPE_FILENAME)

        assert header.scope_id == result.scope_id
        assert header.base_ref == "HEAD"
        assert header.base_commit == _git(repo, "rev-parse", "HEAD").strip()
        assert header.written_by == "fan-out"
        assert header.coverage == DIFF_COVERAGE_COMPLETE
        assert header.diff_files == ["diff/1-src-a.py.diff"]

    def test_a_missing_diff_file_makes_the_pair_incomplete(self, repo):
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        write_review_diff("HEAD", ["src/a.py"], out)
        (out / "diff" / "1-src-a.py.diff").unlink()

        assert read_review_scope(out / REVIEW_SCOPE_FILENAME) is not None
        assert load_review_scope(out) is None

    def test_a_truncated_diff_file_makes_the_pair_incomplete(self, repo):
        _write(repo, "src/a.py", "one\n")
        _commit(repo, "base")
        _write(repo, "src/a.py", "two\n")
        out = _out(repo)
        write_review_diff("HEAD", ["src/a.py"], out)
        target = out / "diff" / "1-src-a.py.diff"
        size = target.stat().st_size
        target.write_bytes(target.read_bytes()[:10])

        header = read_review_scope(out / REVIEW_SCOPE_FILENAME)
        assert header.diff_sizes == {"diff/1-src-a.py.diff": size}
        assert load_review_scope(out) is None

    def test_an_in_scope_path_shaped_like_a_diff_file_is_not_one_the_set_names(
        self, repo, capsys
    ):
        # `diff/fix-1.diff` under review renders as a code span in its own
        # entry; read as a diff file, it never exists, so the pair read as
        # incomplete and was rewritten on every resume and every reaggregate.
        for rel in ("src/a.py", "diff/fix-1.diff"):
            _write(repo, rel, "one\n")
        _commit(repo, "base")
        for rel in ("src/a.py", "diff/fix-1.diff"):
            _write(repo, rel, "two\n")
        out = _out(repo)
        first = write_review_diff("HEAD", ["src/a.py", "diff/fix-1.diff"], out)

        header = load_review_scope(out)

        assert header is not None
        assert header.diff_files == ["diff/1-diff-fix-1.diff.diff", "diff/2-src-a.py.diff"]
        capsys.readouterr()
        again = write_review_diff("HEAD", ["src/a.py", "diff/fix-1.diff"], out)
        assert again.written is False and again.scope_id == first.scope_id
        assert "Review scope unchanged" in capsys.readouterr().out


class TestANonUtf8PathName:
    RAW = b"caf\xff.py"

    def test_its_counts_land_on_its_entry_and_the_set_reads_back(self, repo):
        # Python hands such a name over as a lone surrogate (argv, os.listdir);
        # `-z` numstat must key it the same way, and the scope id and manifest
        # must encode it rather than raise.
        path = os.fsdecode(self.RAW)
        try:
            (repo / path).write_bytes(b"one\n")
        except (OSError, UnicodeEncodeError):
            pytest.skip("this filesystem refuses a file name that is not UTF-8")
        _commit(repo, "base")
        (repo / path).write_bytes(b"one\ntwo\n")
        out = _out(repo)

        first = write_review_diff("HEAD", [path], out)

        manifest = (out / REVIEW_SCOPE_FILENAME).read_bytes()
        assert b"1. `caf\xff.py` (+1 -0) \xe2\x80\x94 full: `diff/1-caf-.py.diff`" in manifest
        assert (out / "diff" / "1-caf-.py.diff").read_bytes() == _git_bytes(
            repo, "diff", "HEAD", "--", path
        )
        header = load_review_scope(out)
        assert header is not None and header.scope_id == first.scope_id
        again = write_review_diff("HEAD", [path], out)
        assert again.written is False and again.scope_id == first.scope_id
