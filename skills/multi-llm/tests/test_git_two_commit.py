"""Two-commit diffs in ``utils/git_utils.py`` (plans/review-commits.md §2.5, §4 T2).

A commit review compares two commits, ``base`` and ``head``, and never the
working tree. ``head_ref=None`` keeps today's "base against the working tree"
everywhere, byte for byte, so the golden below was captured from the code as
it was before ``head_ref`` existed and is compared on every run. Every test
runs against a real temp repository with the user's git config shut out and
fixed commit dates, so SHAs, diffs and manifests are the same on every
machine.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import git_utils  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "git_two_commit_head_none.json"

#: The environment every repository here is built and read under.
_HERMETIC_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000",
}
_UNSET_ENV = ("GIT_DEFAULT_HASH", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


def _write(repo: Path, rel: str, data: "str | bytes") -> Path:
    """Write bytes, never text: a text write on Windows would add CRs."""
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _init(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "commit.gpgsign", "false")
    return root


# ---------------------------------------------------------------------------
# The golden scenario: one root commit, one change commit, a dirty tree
# ---------------------------------------------------------------------------

#: Every path the golden scope names, in an order no rule sorts by.
GOLDEN_PATHS: List[str] = [
    "tests/fixtures/data/a.json",
    "src/app.py",
    "gen/mirror.py",
    "src/gone.py",
    "src/new.py",
    "tests/test_app.py",
    "src/same.py",
    "blob.dat",
    "src/sp ace.py",
    "src/new_name.py",
    "docs/readme.md",
]


def build_golden_repo(root: Path) -> Dict[str, str]:
    """The golden repository: ``c0`` (root), ``c1`` (the change), a dirty tree.

    Returns ``{"c0": sha, "c1": sha}``. ``c1`` modifies, adds, deletes and
    renames; the tree on top of it carries an unstaged edit, a staged edit
    and an untracked file, which a ``head_ref=None`` diff includes.
    """
    _init(root)
    _write(root, ".gitattributes", "gen/** linguist-generated\n")
    _write(root, "src/app.py", "def app():\n    return 1\n")
    _write(root, "src/gone.py", "import os\n\nGONE = os.sep\n")
    _write(root, "src/same.py", "SAME = True\n")
    _write(root, "src/old_name.py", "".join(f"line {n}\n" for n in range(1, 21)))
    _write(root, "src/sp ace.py", "x = 1\n")
    _write(root, "tests/test_app.py", "def test_app():\n    assert True\n")
    _write(root, "tests/fixtures/data/a.json", '{"a": 1}\n')
    _write(root, "gen/mirror.py", "MIRROR = 1\n")
    _write(root, "blob.dat", b"\x00\x01\x02binary\n")
    _write(root, "docs/readme.md", "# Readme\n")
    c0 = _commit(root, "root")

    _write(root, "src/app.py", "def app():\n    return 2\n\n\ndef more():\n    return 3\n")
    (root / "src/gone.py").unlink()
    _write(root, "src/new.py", "NEW = 'brand new module'\n")
    _git(root, "mv", "src/old_name.py", "src/new_name.py")
    _write(root, "src/sp ace.py", "x = 2\n")
    _write(root, "tests/test_app.py", "def test_app():\n    assert app() == 2\n")
    _write(root, "tests/fixtures/data/a.json", '{"a": 2}\n')
    _write(root, "gen/mirror.py", "MIRROR = 2\n")
    _write(root, "blob.dat", b"\x00\x01\x02binary, changed\n")
    c1 = _commit(root, "change")

    # The dirty tree: an unstaged edit, a staged edit, an untracked file.
    _write(root, "src/app.py", "def app():\n    return 2\n\n\ndef more():\n    return 4\n")
    _write(root, "docs/readme.md", "# Readme\n\nStaged.\n")
    _git(root, "add", "docs/readme.md")
    _write(root, "scratch.txt", "untracked\n")
    return {"c0": c0, "c1": c1}


_WRITTEN_LINE = re.compile(r"^- Written: .*$", re.MULTILINE)


def _normalize(text: str, root: Path) -> str:
    """Machine-specific bits out: the write time and the temp path."""
    text = _WRITTEN_LINE.sub("- Written: <WRITTEN>", text)
    for spelling in {str(root), root.as_posix(), os.path.realpath(root)}:
        text = text.replace(spelling, "<ROOT>")
    return text.replace("\\", "/")


def _diff_set_record(result: Any, out: Path, root: Path) -> Dict[str, Any]:
    manifest = (out / git_utils.REVIEW_SCOPE_FILENAME).read_bytes().decode("utf-8")
    diffs = {
        path.name: path.read_bytes().decode("utf-8")
        for path in sorted((out / git_utils.REVIEW_DIFF_DIRNAME).iterdir())
    }
    fields = result._asdict()
    for key in ("scope_file", "diff_dir"):
        fields[key] = _normalize(fields[key], root)
    fields["omitted"] = [list(pair) for pair in fields["omitted"]]
    # Only today's fields: the golden predates any field added since.
    keep = ("coverage", "omitted", "scope_id", "scope_file", "diff_dir", "base_ref",
            "base_commit", "written_by", "written", "failure")
    return {
        "manifest": _normalize(manifest, root),
        "diffs": diffs,
        "result": {key: fields[key] for key in keep},
    }


def _header_record(header: Any) -> Dict[str, Any]:
    fields = header._asdict()
    keep = ("scope_id", "base_ref", "base_commit", "written_by", "coverage",
            "omitted", "diff_files", "diff_sizes")
    record = {key: fields[key] for key in keep}
    record["omitted"] = [list(pair) for pair in record["omitted"]]
    return record


def capture_head_none(root: Path, capsys_read) -> Dict[str, Any]:
    """Every ``head_ref=None`` output the golden pins, called as today's
    callers call them (no ``head_ref`` keyword at all). ``capsys_read()``
    returns and clears what was printed so far."""
    record: Dict[str, Any] = {}
    capsys_read()

    out = root / "plans" / "feature" / "code-review"
    first = git_utils.write_review_diff(
        "HEAD~1", GOLDEN_PATHS, out, scope_source="tracked files", cwd=str(root),
    )
    record["write"] = _diff_set_record(first, out, root)
    record["write"]["stdout"] = _normalize(capsys_read(), root)

    again = git_utils.write_review_diff(
        "HEAD~1", GOLDEN_PATHS, out, scope_source="tracked files", cwd=str(root),
    )
    record["write_again"] = {
        "result": _diff_set_record(again, out, root)["result"],
        "stdout": _normalize(capsys_read(), root),
    }

    header = git_utils.read_review_scope(out / git_utils.REVIEW_SCOPE_FILENAME)
    record["read_back"] = _header_record(header)

    for label, base in (("no_base", ""), ("bad_base", "no-such-ref")):
        target = root / "plans" / label / "code-review"
        result = git_utils.write_review_diff(base, GOLDEN_PATHS, target, cwd=str(root))
        record[label] = _diff_set_record(result, target, root)
        record[label]["stdout"] = _normalize(capsys_read(), root)

    real = git_utils._git_diff_output

    def flaky(args, cwd):
        if args[0] == "diff" and args[-1] == "src/new.py" and "--numstat" not in args:
            return None, "fatal: simulated"
        return real(args, cwd)

    git_utils._git_diff_output = flaky
    try:
        target = root / "plans" / "one_failed" / "code-review"
        result = git_utils.write_review_diff("HEAD~1", GOLDEN_PATHS, target, cwd=str(root))
    finally:
        git_utils._git_diff_output = real
    record["one_failed"] = _diff_set_record(result, target, root)
    record["one_failed"]["stdout"] = _normalize(capsys_read(), root)

    hunks = git_utils.capture_diff_hunks("HEAD~1", GOLDEN_PATHS)
    record["hunks"] = json.loads(json.dumps(hunks))
    record["changed"] = git_utils.get_files_changed_since_ref("HEAD~1")
    record["groups"] = git_utils.classify_review_paths(GOLDEN_PATHS, cwd=str(root))
    capsys_read()
    return record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermetic(monkeypatch):
    for key, value in _HERMETIC_ENV.items():
        monkeypatch.setenv(key, value)
    for key in _UNSET_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def golden_repo(tmp_path, monkeypatch, hermetic):
    root = tmp_path / "repo"
    shas = build_golden_repo(root)
    monkeypatch.chdir(root)
    return root, shas


def _plain_diff(root: Path, *args: str) -> bytes:
    """What git itself prints for a per-path diff, the bytes a diff file holds."""
    return subprocess.run(
        ["git", "diff", "--no-color", "--no-ext-diff", "--no-renames", *args],
        cwd=root, check=True, capture_output=True,
    ).stdout


def _manifest(result: Any) -> str:
    return Path(result.scope_file).read_bytes().decode("utf-8")


def _status_of(manifest: str, path: str) -> str:
    for line in manifest.split("\n"):
        if re.match(rf"^\d+\. `{re.escape(path)}`", line):
            return line
    raise AssertionError(f"{path} has no entry in the manifest")


# ---------------------------------------------------------------------------
# head_ref=None: byte-identical to the code before head_ref existed
# ---------------------------------------------------------------------------


class TestHeadNoneGolden:
    def test_every_head_none_output_matches_the_golden(self, golden_repo, capsys):
        root, shas = golden_repo
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        # The repository itself must rebuild to the same commits, or every
        # comparison below would fail for a reason that is not the code's.
        assert shas == golden["shas"]

        record = capture_head_none(root, lambda: capsys.readouterr().out)

        for key in ("write", "write_again", "read_back", "no_base", "bad_base",
                    "one_failed", "hunks", "changed", "groups"):
            assert record[key] == golden[key], key

    def test_a_working_tree_set_records_no_head(self, golden_repo):
        root, _ = golden_repo
        out = root / "plans" / "wt" / "code-review"
        result = git_utils.write_review_diff("HEAD~1", ["src/app.py"], out, cwd=str(root))
        assert result.head_ref is None and result.head_commit is None
        header = git_utils.load_review_scope(out)
        assert header.head_ref is None and header.head_commit is None
        assert "- Head ref:" not in _manifest(result)


# ---------------------------------------------------------------------------
# A two-commit diff: base head, whatever the working tree holds
# ---------------------------------------------------------------------------


class TestTwoCommitDiff:
    def test_the_set_ignores_a_dirty_tree(self, golden_repo, capsys):
        root, shas = golden_repo
        c0, c1 = shas["c0"], shas["c1"]
        out = root / "plans" / "range" / "code-review"

        result = git_utils.write_review_diff(
            c0, GOLDEN_PATHS, out, scope_source="commit range", cwd=str(root),
            head_ref=c1,
        )

        assert result.failure is None and result.coverage == "complete"
        assert (result.base_commit, result.head_ref, result.head_commit) == (c0, c1, c1)
        manifest = _manifest(result)
        # Every diff file is git's own `base head` diff: the unstaged edit
        # (`return 4`) and the staged one (docs/readme.md) never enter it.
        app = (out / "diff" / "3-src-app.py.diff").read_bytes()
        assert app == _plain_diff(root, c0, c1, "--", "src/app.py")
        assert b"return 4" not in app and b"return 3" in app
        assert _status_of(manifest, "docs/readme.md").endswith(
            f"— no diff: no net change in {c0[:7]}..{c1[:7]}"
        )
        assert "(+5 -1)" in _status_of(manifest, "src/app.py")
        assert f"- Base ref: `{c0}` (commit `{c0}`)" in manifest
        assert f"- Head ref: `{c1}` (commit `{c1}`)" in manifest
        # The wording no longer says the working tree is the new side.
        assert "current contents show only the new side" not in manifest
        assert f"commit `{c1[:7]}`" in manifest and "not the working tree" in manifest

        header = git_utils.load_review_scope(out)
        assert (header.base_ref, header.base_commit) == (c0, c0)
        assert (header.head_ref, header.head_commit) == (c1, c1)
        assert header.scope_id == result.scope_id

        again = git_utils.write_review_diff(
            c0, GOLDEN_PATHS, out, scope_source="commit range", cwd=str(root),
            head_ref=c1,
        )
        assert again.written is False and again.scope_id == result.scope_id

    def test_the_changed_files_and_hunks_ignore_a_dirty_tree(self, golden_repo):
        root, shas = golden_repo
        c0, c1 = shas["c0"], shas["c1"]

        changed = git_utils.get_files_changed_since_ref(c0, c1, cwd=str(root))
        # The rename is listed under its new path only; the staged readme and
        # the untracked scratch file are not in the range.
        assert sorted(changed) == [
            "blob.dat", "gen/mirror.py", "src/app.py", "src/gone.py",
            "src/new.py", "src/new_name.py", "src/sp ace.py",
            "tests/fixtures/data/a.json", "tests/test_app.py",
        ]

        hunks = git_utils.capture_diff_hunks(c0, GOLDEN_PATHS, head_ref=c1, cwd=str(root))
        assert "_notices" not in hunks  # no "Diff includes uncommitted changes"
        added = [
            line["content"] for hunk in hunks["src/app.py"]["hunks"]
            for line in hunk["lines"] if line["type"] == "add"
        ]
        assert "    return 3" in added and "    return 4" not in added
        assert "docs/readme.md" not in hunks
        gone = hunks["src/gone.py"]
        assert gone["deleted"] is True
        assert gone["last_known_content"] == "import os\n\nGONE = os.sep\n"

    def test_the_head_none_hunks_still_say_the_tree_is_dirty(self, golden_repo):
        root, _ = golden_repo
        hunks = git_utils.capture_diff_hunks("HEAD~1", ["src/app.py"])
        assert hunks["_notices"] == ["Diff includes uncommitted changes"]

    def test_the_omitted_file_command_names_both_commits(
        self, golden_repo, capsys, monkeypatch,
    ):
        root, shas = golden_repo
        c0, c1 = shas["c0"], shas["c1"]
        real = git_utils._git_diff_output

        def flaky(args, cwd):
            if "diff" in args and args[-1] == "src/new.py" and "--numstat" not in args:
                return None, "fatal: simulated"
            return real(args, cwd)

        monkeypatch.setattr(git_utils, "_git_diff_output", flaky)
        capsys.readouterr()
        result = git_utils.write_review_diff(
            c0, GOLDEN_PATHS, root / "plans" / "f" / "code-review", cwd=str(root),
            head_ref=c1,
        )
        out = capsys.readouterr().out
        assert f"[DIFF_FAILED] git diff {c0} {c1} -- src/new.py: fatal: simulated" in out
        assert result.omitted == [("src/new.py", "git diff failed")]
        assert (
            f"With a shell, `git diff {c0} {c1} -- src/new.py` shows it"
            in _status_of(_manifest(result), "src/new.py")
        )

    def test_a_failed_numstat_names_both_sides(self, golden_repo, capsys):
        root, shas = golden_repo
        result = git_utils.write_review_diff(
            shas["c0"], ["src/app.py"], root / "plans" / "g" / "code-review",
            cwd=str(root), head_ref="no-such-head",
        )
        assert result.failure.startswith(
            f"git diff --numstat {shas['c0']} no-such-head -- <1 paths>:"
        )
        assert result.head_commit is None
        assert "- Head ref: `no-such-head` (does not resolve to a commit)" in _manifest(result)

    def test_the_scope_id_changes_with_head(self, golden_repo):
        root, shas = golden_repo
        c0, c1 = shas["c0"], shas["c1"]
        # A later head whose tree is c1's: the same diff bytes, another pair.
        _git(root, "stash", "-q", "--include-untracked")
        c2 = _commit(root, "nothing")
        paths = ["src/app.py", "src/new.py"]

        def scope(head, name):
            return git_utils.write_review_diff(
                c0, paths, root / "plans" / name / "code-review", cwd=str(root),
                head_ref=head,
            )

        at_c1, at_c2 = scope(c1, "c1"), scope(c2, "c2")
        assert at_c1.scope_id != at_c2.scope_id
        assert sorted(os.listdir(Path(at_c1.diff_dir))) == sorted(os.listdir(Path(at_c2.diff_dir)))
        for name in os.listdir(Path(at_c1.diff_dir)):
            assert (Path(at_c1.diff_dir) / name).read_bytes() == (
                Path(at_c2.diff_dir) / name
            ).read_bytes()
        # Stable for the same pair, and never the working-tree set's id.
        assert scope(c1, "c1-again").scope_id == at_c1.scope_id
        assert git_utils.write_review_diff(
            c0, paths, root / "plans" / "wt" / "code-review", cwd=str(root),
        ).scope_id != at_c2.scope_id


# ---------------------------------------------------------------------------
# A root range: the empty tree as the base
# ---------------------------------------------------------------------------


@pytest.fixture
def root_repo(tmp_path, monkeypatch, hermetic):
    root = _init(tmp_path / "repo")
    _write(root, "src/a.py", "one\ntwo\n")
    _write(root, "tests/test_a.py", "def test_a():\n    pass\n")
    _write(root, "README.md", "# R\n")
    c0 = _commit(root, "root")
    _write(root, "src/a.py", "one\ntwo\nthree\n")
    c1 = _commit(root, "second")
    monkeypatch.chdir(root)
    return root, c0, c1


class TestEmptyTreeBase:
    def test_every_file_is_added_and_nothing_fails(self, root_repo, capsys):
        root, c0, c1 = root_repo
        empty = git_utils.empty_tree_id(str(root))
        paths = ["src/a.py", "tests/test_a.py", "README.md", "absent.py"]
        out = root / "plans" / "root" / "code-review"

        result = git_utils.write_review_diff(empty, paths, out, cwd=str(root), head_ref=c0)

        printed = capsys.readouterr().out
        assert "[DIFF_FAILED]" not in printed
        assert result.failure is None and result.base_commit is None
        manifest = _manifest(result)
        assert "Diff failed:" not in manifest
        assert (
            f"- Base ref: `{empty}` (empty tree: the range starts at the root commit)"
            in manifest
        )
        assert f"- Head ref: `{c0}` (commit `{c0}`)" in manifest
        assert "(+2 -0) — full:" in _status_of(manifest, "src/a.py")
        assert "(+1 -0) — full:" in _status_of(manifest, "README.md")
        assert _status_of(manifest, "absent.py").endswith(
            f"— no diff: no net change in root..{c0[:7]}"
        )
        for diff in Path(result.diff_dir).glob("*.diff"):
            data = diff.read_bytes()
            assert b"new file mode" in data and b"\n-" not in data.split(b"@@", 1)[1]

        header = git_utils.read_review_scope(Path(result.scope_file))
        assert (header.base_ref, header.base_commit) == (empty, None)
        assert (header.head_ref, header.head_commit) == (c0, c0)

        following = git_utils.write_review_diff(
            c0, paths, root / "plans" / "next" / "code-review", cwd=str(root),
            head_ref=c1,
        )
        assert following.scope_id != result.scope_id

    def test_the_empty_tree_is_never_resolved_as_a_commit(self, root_repo, monkeypatch):
        root, c0, _ = root_repo
        empty = git_utils.empty_tree_id(str(root))
        real = git_utils._resolve_base_commit
        asked: List[str] = []

        def spy(ref, cwd):
            asked.append(ref)
            return real(ref, cwd)

        monkeypatch.setattr(git_utils, "_resolve_base_commit", spy)
        git_utils.write_review_diff(
            empty, ["src/a.py"], root / "plans" / "r" / "code-review", cwd=str(root),
            head_ref=c0,
        )
        assert asked == [c0]  # the head alone

    def test_the_other_readers_take_the_empty_tree(self, root_repo):
        root, c0, _ = root_repo
        empty = git_utils.empty_tree_id(str(root))
        assert sorted(git_utils.get_files_changed_since_ref(empty, c0, cwd=str(root))) == [
            "README.md", "src/a.py", "tests/test_a.py",
        ]
        hunks = git_utils.capture_diff_hunks(empty, ["src/a.py"], head_ref=c0, cwd=str(root))
        kinds = {line["type"] for hunk in hunks["src/a.py"]["hunks"] for line in hunk["lines"]}
        assert kinds == {"add"}
        assert git_utils.rename_map(empty, c0, cwd=str(root)) == {}

    def test_a_truncated_id_is_not_the_empty_tree(self, root_repo):
        root, c0, _ = root_repo
        empty = git_utils.empty_tree_id(str(root))
        assert git_utils.is_empty_tree_id(empty, str(root))
        assert not git_utils.is_empty_tree_id(empty[:12], str(root))
        assert not git_utils._is_empty_tree_base(c0, str(root))


# ---------------------------------------------------------------------------
# blobs_in_commit agrees with paths_in_commit
# ---------------------------------------------------------------------------


@pytest.fixture
def blob_repo(tmp_path, monkeypatch, hermetic):
    root = _init(tmp_path / "repo")
    _write(root, "a.py", "a\n")
    _write(root, "pkg/mod.py", "m\n")
    newline_path = None
    if os.name != "nt":
        newline_path = "odd\nname.py"
        _write(root, newline_path, "n\n")
    sha = _commit(root, "c")
    monkeypatch.chdir(root)
    return root, sha, newline_path


class TestBlobsInCommit:
    def test_keys_equal_paths_in_commit_and_ids_equal_rev_parse(self, blob_repo):
        root, sha, newline_path = blob_repo
        asked = ["a.py", "pkg/mod.py", "pkg", "missing.py", newline_path or "odd\nname.py"]

        blobs = git_utils.blobs_in_commit(sha, asked, cwd=str(root))
        present = git_utils.paths_in_commit(sha, asked, cwd=str(root))

        assert set(blobs) == present == {"a.py", "pkg/mod.py", "pkg"}
        for path, object_id in blobs.items():
            assert object_id == _git(root, "rev-parse", f"{sha}:{path}").strip()

    def test_a_missing_ref_is_none_for_both(self, blob_repo):
        root, _, _ = blob_repo
        ref = "no-such-ref"
        assert git_utils.blobs_in_commit(ref, ["a.py"], cwd=str(root)) is None
        assert git_utils.paths_in_commit(ref, ["a.py"], cwd=str(root)) is None

    def test_the_empty_tree_classifies_paths_but_is_not_a_guard_head(self, blob_repo):
        root, _, _ = blob_repo
        ref = git_utils.empty_tree_id(str(root))
        assert git_utils.paths_in_commit(ref, ["a.py"], cwd=str(root)) == set()
        assert git_utils.blobs_in_commit(ref, ["a.py"], cwd=str(root)) is None

    def test_no_paths_is_empty_for_both(self, blob_repo):
        root, sha, _ = blob_repo
        assert git_utils.blobs_in_commit(sha, [], cwd=str(root)) == {}
        assert git_utils.paths_in_commit(sha, [], cwd=str(root)) == set()


# ---------------------------------------------------------------------------
# read_at_commit and paths in git's form
# ---------------------------------------------------------------------------


class TestReadAtCommit:
    def test_reads_the_commit_not_the_disk(self, blob_repo):
        root, sha, _ = blob_repo
        _write(root, "a.py", "edited on disk\n")
        assert git_utils.read_at_commit(sha, "a.py", cwd=str(root)) == b"a\n"
        assert git_utils.read_at_commit(sha, "./a.py", cwd=str(root)) == b"a\n"

    def test_absent_directory_and_bad_ref_are_none(self, blob_repo):
        root, sha, _ = blob_repo
        assert git_utils.read_at_commit(sha, "missing.py", cwd=str(root)) is None
        assert git_utils.read_at_commit(sha, "pkg", cwd=str(root)) is None
        assert git_utils.read_at_commit("no-such-ref", "a.py", cwd=str(root)) is None
        assert git_utils.read_at_commit(sha, "", cwd=str(root)) is None

    def test_an_absolute_path_is_read_repository_relative(self, blob_repo):
        root, sha, _ = blob_repo
        absolute = str(root / "pkg" / "mod.py")
        assert git_utils.git_path(absolute, cwd=str(root)) == "pkg/mod.py"
        assert git_utils.read_at_commit(sha, absolute, cwd=str(root)) == b"m\n"

    @pytest.mark.skipif(os.name != "nt", reason="backslash is a separator only on Windows")
    def test_windows_backslash_and_absolute_paths_read_the_posix_path(self, blob_repo):
        root, sha, _ = blob_repo
        assert git_utils.git_path("pkg\\mod.py") == "pkg/mod.py"
        assert git_utils.read_at_commit(sha, "pkg\\mod.py", cwd=str(root)) == b"m\n"
        assert git_utils.read_at_commit(
            sha, str(root / "pkg" / "mod.py"), cwd=str(root)
        ) == b"m\n"

    @pytest.mark.skipif(os.name == "nt", reason="a backslash cannot be in a Windows file name")
    def test_posix_literal_backslash_file_name_is_read_by_that_name(
        self, tmp_path, monkeypatch, hermetic,
    ):
        root = _init(tmp_path / "repo")
        _write(root, "a\\b.py", "literal\n")
        _write(root, "a/b.py", "nested\n")
        sha = _commit(root, "c")
        assert git_utils.git_path("a\\b.py") == "a\\b.py"
        assert git_utils.read_at_commit(sha, "a\\b.py", cwd=str(root)) == b"literal\n"
        assert git_utils.read_at_commit(sha, "a/b.py", cwd=str(root)) == b"nested\n"


# ---------------------------------------------------------------------------
# capture_file_snapshots_at: the report's snapshots, read at head
# ---------------------------------------------------------------------------


class TestCaptureFileSnapshotsAt:
    def test_same_shape_as_the_disk_snapshot_and_read_at_the_commit(
        self, tmp_path, monkeypatch, hermetic,
    ):
        root = _init(tmp_path / "repo")
        _write(root, "lf.py", "one\ntwo\n")
        _write(root, "crlf.txt", b"one\r\ntwo\r\n")
        _write(root, "bin.dat", b"\x00\x01")
        _write(root, "gone.py", "g\n")
        sha = _commit(root, "c")
        monkeypatch.chdir(root)
        disk = git_utils.capture_file_snapshots(["lf.py", "crlf.txt", "bin.dat"])
        (root / "gone.py").unlink()
        _commit(root, "drop")
        _write(root, "lf.py", "edited\n")

        at = git_utils.capture_file_snapshots_at(
            sha, ["lf.py", "crlf.txt", "bin.dat", "./lf.py", "missing.py"], cwd=str(root),
        )

        assert at["lf.py"] == disk["lf.py"] == ["one", "two"]
        assert at["crlf.txt"] == disk["crlf.txt"] == ["one", "two"]
        assert at["./lf.py"] == ["one", "two"]
        assert "bin.dat" not in at and "missing.py" not in at
        head = _git(root, "rev-parse", "HEAD").strip()
        assert "gone.py" not in git_utils.capture_file_snapshots_at(head, ["gone.py"], cwd=str(root))
        assert git_utils.capture_file_snapshots_at("no-such-ref", ["lf.py"], cwd=str(root)) == {}


# ---------------------------------------------------------------------------
# rename_map and deleting_commit
# ---------------------------------------------------------------------------


class TestRenameMapAndDeletingCommit:
    def test_rename_map_maps_old_to_new(self, golden_repo):
        root, shas = golden_repo
        assert git_utils.rename_map(shas["c0"], shas["c1"], cwd=str(root)) == {
            "src/old_name.py": "src/new_name.py",
        }
        assert git_utils.rename_map("no-such-ref", shas["c1"], cwd=str(root)) is None

    def test_deleting_commit(self, golden_repo):
        root, shas = golden_repo
        c0, c1 = shas["c0"], shas["c1"]
        assert git_utils.deleting_commit(c0, c1, "src/gone.py", cwd=str(root)) == c1
        # A rename's old path was deleted by the commit that moved it.
        assert git_utils.deleting_commit(c0, c1, "src/old_name.py", cwd=str(root)) == c1
        # A root range has no `<base>..` side: None and the empty tree alike.
        assert git_utils.deleting_commit(None, c1, "src/gone.py", cwd=str(root)) == c1
        empty = git_utils.empty_tree_id(str(root))
        assert git_utils.deleting_commit(empty, c1, "src/gone.py", cwd=str(root)) == c1
        # Nothing in the range deleted it, or the range does not reach it.
        assert git_utils.deleting_commit(c0, c1, "src/app.py", cwd=str(root)) is None
        assert git_utils.deleting_commit(c1, c1, "src/gone.py", cwd=str(root)) is None
        assert git_utils.deleting_commit(c0, "no-such-ref", "src/gone.py", cwd=str(root)) is None


# ---------------------------------------------------------------------------
# check-attr --source, and its pre-2.40 fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def attr_repo(tmp_path, monkeypatch, hermetic):
    """``head``'s ``.gitattributes`` marks ``gen/**`` generated; the working
    tree's no longer does."""
    root = _init(tmp_path / "repo")
    _write(root, ".gitattributes", "gen/** linguist-generated\n")
    _write(root, "gen/mirror.py", "M = 1\n")
    _write(root, "src/a.py", "a\n")
    head = _commit(root, "c")
    _write(root, ".gitattributes", "")
    monkeypatch.chdir(root)
    return root, head


def _stub_check_attr(monkeypatch, returncode: int, stderr: str) -> List[List[str]]:
    """Every ``check-attr`` argv that carries ``--source`` fails as told; the
    rest runs the real git. Returns the list of check-attr argvs seen."""
    real = git_utils.subprocess.run
    seen: List[List[str]] = []

    def run(argv, *args, **kwargs):
        if isinstance(argv, list) and "check-attr" in argv:
            seen.append(list(argv))
            if any(arg.startswith("--source=") for arg in argv):
                return subprocess.CompletedProcess(argv, returncode, "", stderr)
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(git_utils.subprocess, "run", run)
    return seen


class TestCheckAttrSource:
    PATHS = ["gen/mirror.py", "src/a.py"]

    def test_the_attributes_are_read_as_head_has_them(self, attr_repo):
        root, head = attr_repo
        at_head = git_utils.classify_review_paths(self.PATHS, cwd=str(root), source=head)
        on_disk = git_utils.classify_review_paths(self.PATHS, cwd=str(root))
        assert at_head["gen/mirror.py"] == "generated"
        assert on_disk["gen/mirror.py"] == "source"

    def test_an_old_git_keeps_the_pinned_attributes_with_one_warning(
        self, attr_repo, monkeypatch, capsys,
    ):
        root, head = attr_repo
        _write(root, ".gitattributes", "src/** linguist-generated\n")
        version = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        seen = _stub_check_attr(
            monkeypatch, 129,
            f"error: unknown option `source={head}'\nusage: git check-attr [-a | --all | <attr>...]\n",
        )

        groups = git_utils.classify_review_paths(self.PATHS, cwd=str(root), source=head)

        assert groups == {"gen/mirror.py": "generated", "src/a.py": "source"}
        assert len(seen) == 2 and not any(a.startswith("--source") for a in seen[1])
        err = capsys.readouterr().err
        assert err.count("Warning:") == 1
        assert version in err
        assert f"attributes from {head[:7]} through the pinned attribute fallback" in err

    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("state", ["unstaged", "staged", "committed"])
    def test_pinned_diff_ignores_later_attributes(self, tmp_path, hermetic, monkeypatch,
                                                 legacy, state):
        from utils.commit_brief import render_brief_header
        from utils.commit_range import measure_range, resolve_range

        root = _init(tmp_path / "repo")
        _write(root, ".gitattributes", "src/** diff\n")
        _write(root, "src/a.py", "value = 1\n")
        _write(root, "src/gone.py", "old = 1\n")
        base = _commit(root, "base")
        _write(root, "src/a.py", "value = 2\n")
        (root / "src/gone.py").unlink()
        head = _commit(root, "reviewed")
        paths = ["src/a.py", "src/gone.py"]
        cr = resolve_range(head, cwd=str(root))
        header_kwargs = dict(cwd=str(root), focus=None, plan=None, plan_snapshot=None,
                             plan_sha256=None)
        before = git_utils.write_review_diff(base, paths, tmp_path / "before",
                                              cwd=str(root), head_ref=head)
        before_hunks = git_utils.capture_diff_hunks(base, paths, cwd=str(root), head_ref=head)
        before_size = measure_range(cr, cwd=str(root))
        before_header = render_brief_header(cr, **header_kwargs)

        _write(root, ".gitattributes", "src/** -diff linguist-generated\n")
        if state != "unstaged":
            _git(root, "add", ".gitattributes")
        if state == "committed":
            _git(root, "commit", "-qm", "later attributes")
        index_before = (root / ".git/index").read_bytes()
        attributes_before = (root / ".gitattributes").read_bytes()
        scratch_indexes = []
        if legacy:
            real_run = subprocess.run

            def old_git(argv, **kwargs):
                if any(arg.startswith(("--source=", "--attr-source=")) for arg in argv):
                    message = "unknown option: --attr-source\n"
                    if not (kwargs.get("text") or kwargs.get("encoding")):
                        message = message.encode()
                    return subprocess.CompletedProcess(argv, 129, message[:0], message)
                if "read-tree" in argv:
                    scratch_indexes.append(kwargs["env"]["GIT_INDEX_FILE"])
                return real_run(argv, **kwargs)

            monkeypatch.setattr(subprocess, "run", old_git)

        after = git_utils.write_review_diff(base, paths, tmp_path / "after",
                                             cwd=str(root), head_ref=head)
        assert after.scope_id == before.scope_id
        assert after.coverage == "complete"
        for first in (tmp_path / "before/diff").iterdir():
            assert (tmp_path / "after/diff" / first.name).read_bytes() == first.read_bytes()
        assert git_utils.capture_diff_hunks(base, paths, cwd=str(root), head_ref=head) == before_hunks
        assert measure_range(cr, cwd=str(root)) == before_size
        assert render_brief_header(cr, **header_kwargs) == before_header
        assert (root / ".git/index").read_bytes() == index_before
        assert (root / ".gitattributes").read_bytes() == attributes_before
        if legacy:
            assert scratch_indexes
            assert all(not Path(index).exists() for index in scratch_indexes)

    def test_any_other_failure_classifies_nothing_by_attribute(
        self, attr_repo, monkeypatch, capsys,
    ):
        root, head = attr_repo
        seen = _stub_check_attr(monkeypatch, 128, "fatal: bad --source\n")

        groups = git_utils.classify_review_paths(self.PATHS, cwd=str(root), source=head)

        assert groups == {"gen/mirror.py": "source", "src/a.py": "source"}
        assert len(seen) == 1  # no retry without --source
        err = capsys.readouterr().err
        assert err.count("Warning:") == 1
        assert "git check-attr linguist-generated failed (fatal: bad --source)" in err

    def test_the_two_commit_set_collapses_by_heads_attributes(self, attr_repo):
        root, head = attr_repo
        _write(root, "gen/mirror.py", "M = 2\n")
        _write(root, "src/a.py", "b\n")
        later = _commit(root, "later")
        result = git_utils.write_review_diff(
            head, self.PATHS, root / "plans" / "a" / "code-review", cwd=str(root),
            head_ref=later,
        )
        manifest = _manifest(result)
        # `later` no longer marks gen/** (its .gitattributes is empty).
        assert "## Generated (0 files)" in manifest
        early = git_utils.write_review_diff(
            git_utils.empty_tree_id(str(root)), self.PATHS,
            root / "plans" / "b" / "code-review", cwd=str(root), head_ref=head,
        )
        assert "- `gen/` — 1 file (+1 -0) — no diff: collapsed (generated)" in _manifest(early)


# ---------------------------------------------------------------------------
# paths_changed_since: THE change test (drift == apply guard)
# ---------------------------------------------------------------------------

SCOPE = [
    "a.py", "gone.py", "keep.py", "run.sh", "thing", "moved.py", "new.py",
    "notes.txt",
]


@pytest.fixture
def change_repo(tmp_path, monkeypatch, hermetic):
    """``base`` → ``head``: head edits a.py, deletes gone.py, adds new.py.
    The working tree starts equal to ``head``."""
    root = _init(tmp_path / "repo")
    _write(root, ".gitattributes", "* text=auto\n")
    for rel in ("a.py", "gone.py", "keep.py", "thing", "moved.py"):
        _write(root, rel, f"{rel}\n")
    _write(root, "run.sh", "echo hi\n")
    _write(root, "notes.txt", "one\ntwo\n")
    _commit(root, "base")
    _write(root, "a.py", "a.py, edited in the range\n")
    (root / "gone.py").unlink()
    _write(root, "new.py", "new\n")
    head = _commit(root, "head")
    monkeypatch.chdir(root)
    return root, head


def _changed(root: Path, head: str, paths=SCOPE) -> set:
    return git_utils.paths_changed_since(head, paths, cwd=str(root))


def _edit(root, rel):
    _write(root, rel, "edited after head\n")


def _rm(root, rel):
    (root / rel).unlink()


class TestPathsChangedSince:
    def test_a_tree_equal_to_head_matches(self, change_repo):
        root, head = change_repo
        # gone.py is absent at head and absent on disk: it matches.
        assert _changed(root, head) == set()

    def test_edited(self, change_repo):
        root, head = change_repo
        _edit(root, "a.py")
        assert _changed(root, head) == {"a.py"}

    def test_staged_and_committed_since_head(self, change_repo):
        root, head = change_repo
        _edit(root, "keep.py")
        _git(root, "add", "keep.py")
        assert _changed(root, head) == {"keep.py"}
        _commit(root, "later")
        assert _changed(root, head) == {"keep.py"}

    @pytest.mark.skipif(os.name == "nt", reason="no executable bit on Windows")
    def test_a_mode_only_change(self, change_repo):
        root, head = change_repo
        os.chmod(root / "run.sh", 0o755)
        assert _changed(root, head) == {"run.sh"}

    def test_deleted_on_disk(self, change_repo):
        root, head = change_repo
        _rm(root, "keep.py")
        assert _changed(root, head) == {"keep.py"}

    def test_recreated_untracked(self, change_repo):
        root, head = change_repo
        _write(root, "gone.py", "back\n")
        assert _changed(root, head) == {"gone.py"}

    def test_recreated_and_added_with_intent_to_add(self, change_repo):
        root, head = change_repo
        _write(root, "gone.py", "back\n")
        _git(root, "add", "-N", "gone.py")
        assert _changed(root, head) == {"gone.py"}

    def test_a_skip_worktree_entry_edited(self, change_repo):
        root, head = change_repo
        _git(root, "update-index", "--skip-worktree", "keep.py")
        _edit(root, "keep.py")
        # git diff never reads that disk copy; only the ls-files tag tells.
        assert subprocess.run(
            ["git", "diff", "--name-only", head, "--", "keep.py"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout == ""
        assert _changed(root, head) == {"keep.py"}

    def test_an_assume_unchanged_entry(self, change_repo):
        root, head = change_repo
        _git(root, "update-index", "--assume-unchanged", "keep.py")
        assert _changed(root, head) == {"keep.py"}

    def test_a_directory_where_a_file_was(self, change_repo):
        root, head = change_repo
        _rm(root, "thing")
        _write(root, "thing/inner.py", "x\n")
        _write(root, "gone.py/inner.py", "x\n")  # absent at head, a directory now
        assert _changed(root, head) == {"thing", "gone.py"}

    def test_a_disk_entry_that_is_neither_file_nor_symlink_is_changed(self, change_repo):
        """What only the kind check sees: a directory at ``head`` with the same
        directory on disk, and a submodule checked out at the very commit
        ``head`` records. ``git diff <head>`` lists neither."""
        root, _ = change_repo
        _write(root, "pkg/inner.py", "x\n")
        nested = _init(root / "sub")
        _write(nested, "s.py", "s\n")
        sub_sha = _commit(nested, "sub")
        _git(root, "update-index", "--add", "--cacheinfo", f"160000,{sub_sha},sub")
        _git(root, "add", "pkg")
        _git(root, "commit", "-q", "-m", "dir and gitlink")
        head = _git(root, "rev-parse", "HEAD").strip()
        assert subprocess.run(
            ["git", "diff", "--name-only", head, "--", "pkg", "sub"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout == ""
        assert _changed(root, head, ["pkg", "sub", "keep.py"]) == {"pkg", "sub"}

    def test_git_mv_of_a_scope_file_is_listed_at_its_scope_path(self, change_repo):
        root, head = change_repo
        _git(root, "mv", "moved.py", "elsewhere.py")
        assert _changed(root, head) == {"moved.py"}

    def test_a_move_onto_a_deleted_scope_path_lists_both(self, change_repo):
        root, head = change_repo
        _git(root, "config", "diff.renames", "true")
        _git(root, "mv", "moved.py", "gone.py")
        # What plain `--name-only` would print: the rename's new path alone.
        paired = subprocess.run(
            ["git", "diff", "--name-only", head, "--", "moved.py", "gone.py"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
        assert paired == ["gone.py"]
        assert _changed(root, head) == {"moved.py", "gone.py"}

    def test_a_crlf_checkout_of_an_unchanged_file_matches(self, change_repo):
        root, head = change_repo
        _write(root, "notes.txt", b"one\r\ntwo\r\n")
        assert (root / "notes.txt").read_bytes() != _git(root, "show", f"{head}:notes.txt").encode()
        assert _changed(root, head) == set()

    def test_a_failed_read_marks_every_path_changed(self, change_repo, capsys):
        root, _ = change_repo
        assert _changed(root, "no-such-head", ["a.py", "keep.py"]) == {"a.py", "keep.py"}
        assert capsys.readouterr().err.count("Warning:") == 1

    def test_the_result_holds_the_strings_as_given(self, change_repo):
        root, head = change_repo
        _edit(root, "a.py")
        assert _changed(root, head, ["./a.py", "keep.py", ""]) == {"./a.py"}
        assert _changed(root, head, []) == set()

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="only a Linux file system holds a name that is not valid UTF-8",
    )
    def test_a_name_that_is_not_utf8_matches_head_while_unchanged(self, change_repo):
        """A name that is not valid UTF-8 arrives as its ``surrogateescape``
        string; the presence read must ask git about those very bytes, not
        about a ``?`` a lossy encode would put there, or the unchanged file
        reads as absent at ``head`` and so as changed forever."""
        root, _ = change_repo
        name = os.fsdecode(b"caf\xe9.py")
        try:
            _write(root, name, "x\n")
        except OSError as exc:
            pytest.skip(f"this file system refuses a non-UTF-8 name ({exc})")
        head = _commit(root, "a non-UTF-8 name")
        expected = _git(root, "rev-parse", f"{head}:{name}").strip()

        assert git_utils.paths_in_commit(head, [name, "keep.py"], cwd=str(root)) == {
            name, "keep.py",
        }
        assert git_utils.blobs_in_commit(head, [name], cwd=str(root)) == {name: expected}
        assert _changed(root, head, [name, "keep.py"]) == set()
        _write(root, name, "edited\n")
        assert _changed(root, head, [name, "keep.py"]) == {name}

    def test_a_string_no_bytes_carry_is_absent_not_a_raise(self, change_repo):
        root, head = change_repo
        assert git_utils.paths_in_commit(head, ["\ud800.py", "keep.py"], cwd=str(root)) == {
            "keep.py",
        }

    @pytest.mark.skipif(os.name == "nt", reason="':' cannot be in a Windows file name")
    def test_pathspec_magic_in_a_name_is_a_file_name(self, change_repo, capsys):
        root, head = change_repo
        # Without literal pathspecs git refuses the whole call on the magic
        # (`:(bad)` is no pathspec magic it knows), and every path would read
        # as changed. Literally, it is a name absent at head and on disk.
        assert _changed(root, head, [":(bad)x.py", "*.py", "keep.py"]) == set()
        assert capsys.readouterr().err == ""
