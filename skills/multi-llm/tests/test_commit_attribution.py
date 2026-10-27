"""Tests for ``utils/commit_attribution.py``: which commit introduced each
finding of a commit review (plans/review-commits.md §2.6, §4 T4).

Every fixture is a real scratch repository, and every ``commit_range`` record
is the one :func:`utils.commit_range.resolve_range` builds for it, so the
first-parent list (merges included) is the production one. Only the blame
timeout stubs a seam (``git_argv`` for the blame call), as §4 T4 says.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import commit_attribution as ca  # noqa: E402
from utils.commit_attribution import (  # noqa: E402
    BLAME_TIMEOUT_S,
    INTRODUCED_BY,
    INTRODUCED_BY_SOURCE,
    SOURCE_BLAME,
    SOURCE_MODEL,
    SOURCE_MODEL_DRIFTED,
    SOURCE_NONE,
    anchor_at_head,
    attribute_groups,
    attribute_review_groups,
    group_location,
    parse_porcelain,
)
from utils.commit_range import resolve_range  # noqa: E402
from utils.state_manager import generate_group_id, stamp_stable_ids  # noqa: E402


@pytest.fixture(autouse=True)
def _git_env(monkeypatch):
    """No user or system git config, a fixed identity, no inherited redirection."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.invalid"),
                        ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.invalid")):
        monkeypatch.setenv(name, value)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY"):
        monkeypatch.delenv(name, raising=False)


class Repo:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")

    @property
    def cwd(self) -> str:
        return str(self.root)

    def git(self, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout.strip()

    def write(self, name: str, content: Any) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    def commit(self, subject: str, files: Optional[Dict[str, Any]] = None,
               remove: Optional[List[str]] = None) -> str:
        for name, content in (files or {}).items():
            self.write(name, content)
        for name in remove or []:
            self.git("rm", "-q", name)
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", subject)
        return self.git("rev-parse", "HEAD")


def lines(*values: str) -> str:
    return "".join(f"{v}\n" for v in values)


TEN = [f"line {i}" for i in range(1, 11)]


def with_line(base: List[str], number: int, text: str) -> List[str]:
    out = list(base)
    out[number - 1] = text
    return out


def record_for(repo: Repo, spec: Optional[str], **extra: Any) -> Dict[str, Any]:
    record = resolve_range(spec, cwd=repo.cwd).to_record()
    record.setdefault("drifted", [])
    record.update(extra)
    return record


def group(file: Optional[str], line_range: Any, *, anchor: str = "", claim: Any = None,
          theme: str = "A finding") -> Dict[str, Any]:
    suggestion: Dict[str, Any] = {"title": theme, "desc": f"{theme} details", "type": "bug",
                                  "importance": "high", "anchor_text": anchor}
    if file is not None:
        suggestion["file"] = file
    suggestion["line_range"] = line_range
    if claim is not None:
        suggestion[INTRODUCED_BY] = claim
    return {"theme": theme, "category": "bug", "models": ["m:a"], "suggestions": [suggestion]}


def outcome(g: Dict[str, Any]):
    return g[INTRODUCED_BY], g[INTRODUCED_BY_SOURCE]


@pytest.fixture
def linear(tmp_path):
    """base, then c1 (rewrites line 3 of a.py), then c2 (rewrites line 5)."""
    repo = Repo(Path(os.path.realpath(tmp_path)) / "repo")
    base = repo.commit("base", {"a.py": lines(*TEN), "keep.py": lines(*TEN)})
    c1 = repo.commit("c1: change line three", {"a.py": lines(*with_line(TEN, 3, "three"))})
    c2 = repo.commit("c2: change line five",
                     {"a.py": lines(*with_line(with_line(TEN, 3, "three"), 5, "five"))})
    return repo, base, c1, c2


# ---------------------------------------------------------------------------
# Blame
# ---------------------------------------------------------------------------


class TestBlame:
    def test_blame_picks_the_newest_in_range_commit(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [3, 5], anchor="three")]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_BLAME)
        assert result.blame_calls == 1

    def test_a_single_line_goes_to_the_commit_that_wrote_it(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [3, 3]), group("a.py", [5, 5])]
        attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c1, SOURCE_BLAME)
        assert outcome(groups[1]) == (c2, SOURCE_BLAME)

    def test_two_groups_on_one_file_make_one_blame_call(self, linear, monkeypatch):
        repo, base, c1, c2 = linear
        calls = []
        real = ca._run_blame

        def spy(argv, cwd, timeout):
            calls.append(list(argv))
            return real(argv, cwd, timeout)

        monkeypatch.setattr(ca, "_run_blame", spy)
        groups = [group("a.py", [3, 3]), group("a.py", [5, 6]), group("keep.py", [1, 2])]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert result.blame_calls == 2          # a.py once, keep.py once
        a_calls = [argv for argv in calls if argv[-1] == "a.py"]
        assert len(a_calls) == 1
        argv = a_calls[0]
        assert argv.count("-L") == 2
        assert "3,3" in argv and "5,6" in argv
        assert f"{base}..{c2}" in argv
        for flag in ("--porcelain", "--first-parent", "--root"):
            assert flag in argv

    def test_an_older_line_takes_the_models_claim_as_model(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [1, 2], claim=c1[:7]), group("a.py", [1, 2])]
        attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c1, SOURCE_MODEL)
        assert outcome(groups[1]) == (None, SOURCE_NONE)

    def test_a_range_ending_past_the_last_line_is_clamped_and_still_blamed(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [5, 400])]
        attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_BLAME)

    def test_a_root_range_blames_head_alone_and_keeps_the_root_commit(self, tmp_path):
        repo = Repo(Path(os.path.realpath(tmp_path)) / "repo")
        root = repo.commit("root", {"a.py": lines(*TEN)})
        second = repo.commit("second", {"a.py": lines(*with_line(TEN, 2, "two"))})
        # A root range: every commit up to `second`, base None.
        record = {"base": None, "head": second, "drifted": [],
                  "commits": [{"sha": second, "subject": "second"},
                              {"sha": root, "subject": "root"}]}
        groups = [group("a.py", [1, 1]), group("a.py", [2, 2])]
        attribute_groups(groups, record, cwd=repo.cwd)
        assert outcome(groups[0]) == (root, SOURCE_BLAME)
        assert outcome(groups[1]) == (second, SOURCE_BLAME)


# ---------------------------------------------------------------------------
# A merge in the range
# ---------------------------------------------------------------------------


@pytest.fixture
def merged(tmp_path):
    """base; main gets m1 (line 2); side branch s1 rewrites line 7; a --no-ff
    merge brings it onto main."""
    repo = Repo(Path(os.path.realpath(tmp_path)) / "repo")
    base = repo.commit("base", {"a.py": lines(*TEN)})
    repo.git("checkout", "-q", "-b", "side")
    s1 = repo.commit("s1: side change", {"a.py": lines(*with_line(TEN, 7, "seven"))})
    repo.git("checkout", "-q", "main")
    m1 = repo.commit("m1: main change", {"other.py": "x = 1\n"})
    repo.git("merge", "-q", "--no-ff", "-m", "merge side", "side")
    merge = repo.git("rev-parse", "HEAD")
    return repo, base, s1, m1, merge


class TestAMergeInTheRange:
    def test_the_list_is_first_parent(self, merged):
        repo, base, s1, m1, merge = merged
        record = record_for(repo, f"{base}..{merge}")
        assert [c["sha"] for c in record["commits"]] == [merge, m1]

    def test_blame_attributes_a_side_line_to_the_listed_merge(self, merged):
        repo, base, s1, m1, merge = merged
        groups = [group("a.py", [7, 7], anchor="seven")]
        attribute_groups(groups, record_for(repo, f"{base}..{merge}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (merge, SOURCE_BLAME)

    def test_a_claim_naming_the_side_commit_maps_to_the_merge(self, merged):
        repo, base, s1, m1, merge = merged
        # Line 1 predates the range, so blame names no listed commit.
        groups = [group("a.py", [1, 1], claim=s1[:10]), group(None, None, claim=s1)]
        attribute_groups(groups, record_for(repo, f"{base}..{merge}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (merge, SOURCE_MODEL)
        assert outcome(groups[1]) == (merge, SOURCE_MODEL)

    def test_a_claim_naming_an_older_side_commit_maps_to_its_merge(self, tmp_path, monkeypatch):
        """The first of two commits on a merged side branch is no parent of any
        first-parent commit; it still maps to the merge that brought it in.
        A branch merged twice maps each side commit to the OLDEST merge that
        reaches it, and each distinct side commit costs one walk per pass."""
        repo = Repo(Path(os.path.realpath(tmp_path)) / "repo")
        base = repo.commit("base", {"a.py": lines(*TEN)})
        repo.git("checkout", "-q", "-b", "side")
        s0 = repo.commit("s0: side, first", {"side.py": "s0\n"})
        s1 = repo.commit("s1: side, second", {"side.py": "s0\ns1\n"})
        repo.git("checkout", "-q", "main")
        m1 = repo.commit("m1: main change", {"other.py": "x = 1\n"})
        repo.git("merge", "-q", "--no-ff", "-m", "merge side once", "side")
        first_merge = repo.git("rev-parse", "HEAD")
        repo.git("checkout", "-q", "side")
        s2 = repo.commit("s2: side, after the first merge", {"side.py": "s0\ns1\ns2\n"})
        repo.git("checkout", "-q", "main")
        repo.git("merge", "-q", "--no-ff", "-m", "merge side again", "side")
        second_merge = repo.git("rev-parse", "HEAD")
        record = record_for(repo, f"{base}..{second_merge}")
        assert [c["sha"] for c in record["commits"]] == [second_merge, first_merge, m1]

        walks = []
        real = ca._run_git

        def spy(*args, **kwargs):
            if "--ancestry-path" in args:
                walks.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(ca, "_run_git", spy)
        # Line 1 predates the range, so every group falls to the model's claim.
        groups = [group("a.py", [1, 1], claim=s0), group("a.py", [1, 1], claim=s0[:9]),
                  group("a.py", [1, 1], claim=s1[:7]), group(None, None, claim=s2),
                  group("a.py", [1, 1], claim=base)]
        attribute_groups(groups, record, cwd=repo.cwd)
        assert outcome(groups[0]) == (first_merge, SOURCE_MODEL)
        assert outcome(groups[1]) == (first_merge, SOURCE_MODEL)
        assert outcome(groups[2]) == (first_merge, SOURCE_MODEL)
        assert outcome(groups[3]) == (second_merge, SOURCE_MODEL)
        assert outcome(groups[4]) == (None, SOURCE_NONE)
        assert len(walks) == 3, walks   # s0 (named twice), s1, s2

    def test_the_claim_maps_to_the_merge_not_to_a_later_commit(self, merged):
        repo, base, s1, m1, merge = merged
        after = repo.commit("after the merge", {"later.py": "y = 2\n"})
        record = record_for(repo, f"{base}..{after}")
        assert [c["sha"] for c in record["commits"]] == [after, merge, m1]
        groups = [group("a.py", [1, 1], claim=s1[:7]), group("a.py", [7, 7])]
        attribute_groups(groups, record, cwd=repo.cwd)
        assert outcome(groups[0]) == (merge, SOURCE_MODEL)
        assert outcome(groups[1]) == (merge, SOURCE_BLAME)


# ---------------------------------------------------------------------------
# The model's claim
# ---------------------------------------------------------------------------


class TestTheClaim:
    def test_out_of_range_shas_are_refused(self, linear):
        repo, base, c1, c2 = linear
        claims = [base[:7], base, "deadbee", "not-a-sha", "c1", 42, c1[:6]]
        groups = [group("a.py", [1, 1], claim=claim) for claim in claims]
        attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert [outcome(g) for g in groups] == [(None, SOURCE_NONE)] * len(claims)

    def test_a_claim_in_backticks_or_with_a_subject_still_reads(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [1, 1], claim=f"`{c1[:8]}`"),
                  group("a.py", [1, 1], claim=f"{c2[:7]} c2: change line five")]
        attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c1, SOURCE_MODEL)
        assert outcome(groups[1]) == (c2, SOURCE_MODEL)

    def test_an_ambiguous_prefix_within_the_range_is_refused(self):
        a = "abcdef1" + "0" * 33
        b = "abcdef1" + "1" * 33
        resolver = ca._ClaimResolver([a, b], None, b, None)
        assert resolver.resolve("abcdef1") is None
        assert resolver.resolve("abcdef10") == a

    def test_the_first_claim_that_resolves_wins(self, linear):
        repo, base, c1, c2 = linear
        g = group("a.py", [1, 1], claim="deadbee")
        g["suggestions"].append({"title": "x", "desc": "y", "type": "bug",
                                 INTRODUCED_BY: c1[:7]})
        attribute_groups([g], record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(g) == (c1, SOURCE_MODEL)

    def test_a_previous_passs_output_is_never_read_as_a_claim(self, linear):
        repo, base, c1, c2 = linear
        g = group("a.py", [1, 1])
        g[INTRODUCED_BY], g[INTRODUCED_BY_SOURCE] = c1, SOURCE_MODEL
        attribute_groups([g], record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(g) == (None, SOURCE_NONE)


# ---------------------------------------------------------------------------
# Blame that cannot run
# ---------------------------------------------------------------------------


class TestBlameThatCannotRun:
    def test_a_file_deleted_at_head_falls_back_to_the_claim_then_none(self, linear, capsys):
        repo, base, c1, c2 = linear
        c3 = repo.commit("c3: drop keep.py", remove=["keep.py"])
        groups = [group("keep.py", [2, 3], claim=c1[:7]), group("keep.py", [2, 3])]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c3}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c1, SOURCE_MODEL)
        assert outcome(groups[1]) == (None, SOURCE_NONE)
        assert result.blame_calls == 0
        out = capsys.readouterr().out
        assert "keep.py" in out and f"absent at {c3[:7]}" in out

    def test_a_binary_file_falls_back(self, linear, capsys):
        repo, base, c1, c2 = linear
        c3 = repo.commit("c3: binary", {"blob.bin": b"abc\x00def\nmore\n"})
        groups = [group("blob.bin", [1, 1], claim=c3[:7]), group("blob.bin", [1, 1])]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c3}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c3, SOURCE_MODEL)
        assert outcome(groups[1]) == (None, SOURCE_NONE)
        assert result.blame_calls == 0
        assert "blob.bin (binary file)" in capsys.readouterr().out

    def test_a_range_starting_past_the_last_line_falls_back_and_spares_the_others(
        self, linear, capsys
    ):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [11, 12], claim=c1[:7]), group("a.py", [40, 41]),
                  group("a.py", [5, 5])]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c1, SOURCE_MODEL)
        assert outcome(groups[1]) == (None, SOURCE_NONE)
        # git refuses a whole call for one bad -L; the others are still blamed.
        assert outcome(groups[2]) == (c2, SOURCE_BLAME)
        assert result.blame_calls == 1
        assert "past the last line (10)" in capsys.readouterr().out

    def test_a_call_that_times_out_falls_back_and_the_run_completes(
        self, linear, monkeypatch, capsys
    ):
        repo, base, c1, c2 = linear
        assert BLAME_TIMEOUT_S == 60
        monkeypatch.setattr(ca, "BLAME_TIMEOUT_S", 0.5)
        real_argv = ca.git_argv

        def slow_blame(*args, cwd=None):
            if args and args[0] == "blame":
                return [sys.executable, "-c", "import time; time.sleep(30)"]
            return real_argv(*args, cwd=cwd)

        monkeypatch.setattr(ca, "git_argv", slow_blame)
        groups = [group("a.py", [5, 5], claim=c1[:7]), group("a.py", [3, 3])]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c1, SOURCE_MODEL)
        assert outcome(groups[1]) == (None, SOURCE_NONE)
        assert result.blame_calls == 1
        assert "git blame passed 0.5s" in capsys.readouterr().out

    def test_a_failing_call_falls_back(self, linear, monkeypatch):
        repo, base, c1, c2 = linear
        real_argv = ca.git_argv

        def broken(*args, cwd=None):
            if args and args[0] == "blame":
                return [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(2)"]
            return real_argv(*args, cwd=cwd)

        monkeypatch.setattr(ca, "git_argv", broken)
        groups = [group("a.py", [5, 5], claim=c2[:7])]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_MODEL)
        assert ("a.py", "git blame failed (boom)") in result.skipped

    def test_a_group_without_a_location_takes_the_claim(self, linear):
        repo, base, c1, c2 = linear
        groups = [group(None, None, claim=c2[:7]), group("a.py", None), group("a.py", "x")]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert [outcome(g) for g in groups] == [
            (c2, SOURCE_MODEL), (None, SOURCE_NONE), (None, SOURCE_NONE)]
        assert result.blame_calls == 0


# ---------------------------------------------------------------------------
# A drifted file
# ---------------------------------------------------------------------------


@pytest.fixture
def drifted(linear):
    """a.py edited on disk after head: three lines inserted above line 5, so
    head's line 5 ("five") sits at line 8 of the disk copy."""
    repo, base, c1, c2 = linear
    disk = with_line(TEN, 3, "three")
    disk = disk[:4] + ["new a", "new b", "new c"] + disk[4:]
    disk[7] = "five"
    repo.write("a.py", lines(*disk))
    assert (repo.root / "a.py").read_text().splitlines()[7] == "five"
    return linear


class TestADriftedFile:
    def test_disk_numbers_whose_anchor_is_not_at_head_take_the_claim_as_model_drifted(
        self, drifted, capsys
    ):
        repo, base, c1, c2 = drifted
        record = record_for(repo, f"{base}..{c2}", drifted=["a.py"])
        groups = [group("a.py", [8, 8], anchor="five", claim=c2[:7]),
                  group("a.py", [8, 8], anchor="five", claim="deadbee"),
                  group("a.py", [8, 8], anchor="", claim=c1[:7])]
        result = attribute_groups(groups, record, cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_MODEL_DRIFTED)
        assert outcome(groups[1]) == (None, SOURCE_NONE)
        assert outcome(groups[2]) == (c1, SOURCE_MODEL_DRIFTED)
        assert result.blame_calls == 0
        assert result.drifted == ("a.py",)
        out = capsys.readouterr().out
        assert "a.py" in out and "model-drifted" in out and "anchor_text not at lines 8-8" in out

    def test_heads_numbers_are_blamed(self, drifted):
        repo, base, c1, c2 = drifted
        record = record_for(repo, f"{base}..{c2}", drifted=["a.py"])
        groups = [group("a.py", [5, 5], anchor="five", claim=c1[:7]),
                  group("a.py", [3, 5], anchor="three line 4\n   five")]
        attribute_groups(groups, record, cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_BLAME)
        # Whitespace runs collapse on both sides and the anchor may span lines.
        assert outcome(groups[1]) == (c2, SOURCE_BLAME)

    def test_a_file_edited_only_after_launch_gets_the_same_check(self, drifted):
        repo, base, c1, c2 = drifted
        record = record_for(repo, f"{base}..{c2}")   # drifted: [] at launch
        assert record["drifted"] == []
        groups = [group("a.py", [8, 8], anchor="five", claim=c2[:7]),
                  group("a.py", [5, 5], anchor="five")]
        result = attribute_groups(groups, record, cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_MODEL_DRIFTED)
        assert outcome(groups[1]) == (c2, SOURCE_BLAME)
        assert result.drifted == ("a.py",)

    def test_an_undrifted_file_is_blamed_without_an_anchor(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [5, 5], anchor="")]
        result = attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        assert outcome(groups[0]) == (c2, SOURCE_BLAME)
        assert result.drifted == ()


# ---------------------------------------------------------------------------
# Off the group hash; the call site's entry
# ---------------------------------------------------------------------------


class TestTheGroupHash:
    def test_attribution_never_moves_the_group_hash(self, linear):
        repo, base, c1, c2 = linear
        groups = [group("a.py", [3, 3], claim=c2[:7], theme="One"),
                  group("a.py", [1, 1], claim=c1[:7], theme="Two")]
        stamp_stable_ids(groups)
        before = [(g["group_hash"], generate_group_id(g)) for g in groups]
        attribute_groups(groups, record_for(repo, f"{base}..{c2}"), cwd=repo.cwd)
        after = [(g["group_hash"], generate_group_id(g)) for g in groups]
        assert before == after
        restamped = [dict(g) for g in groups]
        stamp_stable_ids(restamped)
        assert [g["group_hash"] for g in restamped] == [h for h, _ in before]
        # Sibling fields only: the title, description and theme are untouched.
        assert groups[0]["theme"] == "One"
        assert groups[0]["suggestions"][0]["desc"] == "One details"

    def test_two_models_that_disagree_still_group_together(self):
        from utils import group_similar_suggestions

        issue = {"title": "Off by one", "desc": "The loop ends early.", "importance": "high",
                 "file": "a.py", "line_range": [3, 3], "type": "bug"}
        mine = dict(issue, source_model="m:a", introduced_by="aaaaaaa")
        theirs = dict(issue, source_model="m:b", introduced_by="bbbbbbb")
        grouped = group_similar_suggestions([mine, theirs])
        assert len(grouped) == 1


class TestTheEntryPoint:
    def test_no_commit_range_attributes_nothing(self, linear):
        groups = [group("a.py", [3, 3], claim="abc1234")]
        assert attribute_review_groups(groups, {"head_ref": {"head": "x"}}, repo_root=None) is None
        assert INTRODUCED_BY not in groups[0]

    def test_a_state_with_commit_range_attributes_every_group(self, linear, capsys):
        repo, base, c1, c2 = linear
        state = {"commit_range": record_for(repo, f"{base}..{c2}")}
        groups = [group("a.py", [3, 3]), group("a.py", [1, 1])]
        result = attribute_review_groups(groups, state, repo_root=repo.cwd)
        assert [o.source for o in result.outcomes] == [SOURCE_BLAME, SOURCE_NONE]
        assert "2 group(s) attributed: 1 blame, 1 none" in capsys.readouterr().out

    def test_an_unexpected_failure_leaves_every_group_unattributed(self, linear, monkeypatch,
                                                                    capsys):
        repo, base, c1, c2 = linear

        def broken(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(ca, "attribute_groups", broken)
        groups = [group("a.py", [3, 3], claim=c1[:7])]
        result = attribute_review_groups(groups, {"commit_range": record_for(
            repo, f"{base}..{c2}")}, repo_root=repo.cwd)
        assert outcome(groups[0]) == (None, SOURCE_NONE)
        assert [o.source for o in result.outcomes] == [SOURCE_NONE]
        assert "disk on fire" in capsys.readouterr().out


class TestHelpers:
    def test_parse_porcelain_reads_headers_only(self):
        sha = "a" * 40
        other = "b" * 40
        text = "\n".join([
            f"{sha} 1 1 2", "author x", f"summary {other} 9 9", "filename f", "\tcontent",
            f"{sha} 2 2", f"\t{other} 3 3",
            f"{other} 5 3 1", "boundary", "filename f", "\told",
        ])
        assert parse_porcelain(text) == {1: sha, 2: sha, 3: other}

    def test_anchor_at_head(self):
        text = ["def f():", "    return  1", "", "x = 2"]
        assert anchor_at_head("return 1", text, 1, 2)
        assert anchor_at_head("f():\n return 1", text, 1, 2)
        assert not anchor_at_head("x = 2", text, 1, 2)
        assert not anchor_at_head("   ", text, 1, 4)
        assert not anchor_at_head("x = 2", text, 5, 4)

    def test_group_location_reads_the_primary_suggestion(self):
        g = group("pkg/a.py", [9, 4], anchor="x")
        assert group_location(g) == ("pkg/a.py", (4, 9), "x")
        assert group_location(group("a.py", [0, 2]))[1] == (1, 2)
        assert group_location(group("a.py", [True, 2]))[1] is None
        bare = {"file": "b.py", "line_range": [2, 3], "anchor_text": "y"}
        assert group_location(bare) == ("b.py", (2, 3), "y")
