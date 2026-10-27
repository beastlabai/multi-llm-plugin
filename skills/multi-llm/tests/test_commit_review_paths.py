"""The commit-review path layer (plans/review-commits.md §2.2, §2.4, §2.8).

``utils/commit_review.py``'s helpers that decide where a commit brief lives,
whether a file is one, which brief a pinned range names, and which paths are a
commit review's own files. Plus the class drop those feed:
``code_review_orchestrator._drop_workflow_paths`` and finalize_tracking's
``_drop_workflow_files`` leave every brief and report folder out of the review
scope and ``tracked_files``, whatever run is reviewing.

Every fixture is a throwaway repository under ``tmp_path``. The default suite
pins project discovery to "none" (``conftest._isolate_provider_config``), so
the folder is the shipped ``plans/commit-reviews`` unless a test says
otherwise.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import code_review_orchestrator as cro
import finalize_tracking
from utils import commit_review
from utils.commit_review import (
    BRIEF_END_MARKER,
    BRIEF_MARKER,
    CommitReviewError,
    brief_for_pinned,
    brief_path_for,
    commit_review_artifact,
    commit_reviews_folder_rel,
    document_label,
    ensure_commit_reviews_dir,
    is_commit_brief,
    matches_brief_path,
    not_ignored_warning,
    output_dir_for_brief,
    repo_root_of,
)
from utils.output_handler import output_dir_path
from utils.provider_registry import ConfigError

BASE = "0123456789ab" + "c" * 28
HEAD = "fedcba987654" + "d" * 28
SLUG = "0123456789ab-fedcba987654"
ROOT_SLUG = "root-fedcba987654"
FOLDER = "plans/commit-reviews"


def _can_symlink() -> bool:
    with tempfile.TemporaryDirectory() as d:
        try:
            os.symlink(os.path.join(d, "t"), os.path.join(d, "l"))
            return True
        except (OSError, NotImplementedError):
            return False


_requires_symlinks = pytest.mark.skipif(
    not _can_symlink(), reason="requires symlink-creation privilege"
)


def _git(repo, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True,
    ).stdout


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with a tracked ``plans/`` and no commit-reviews folder; cwd."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root / "plans" / "feature.md", "# Feature\n")
    _write(root / "src" / "app.py", "a = 1\n")
    _write(root / "README.md", "# Readme\n")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "init")
    monkeypatch.chdir(root)
    return Path(os.path.realpath(root))


# ---------------------------------------------------------------------------
# Where a brief lives
# ---------------------------------------------------------------------------


class TestBriefPaths:
    def test_repo_root_of_a_file_a_directory_and_the_cwd(self, repo):
        assert repo_root_of() == repo
        assert repo_root_of(repo / "src") == repo
        assert repo_root_of(repo / "src" / "app.py") == repo
        # A brief about to be written: its folder does not exist yet.
        assert repo_root_of(repo / FOLDER / f"{SLUG}.md") == repo

    def test_repo_root_of_outside_a_repository_is_a_refusal(self, tmp_path):
        loose = tmp_path / "loose"
        loose.mkdir()
        with pytest.raises(CommitReviewError) as exc:
            repo_root_of(loose)
        assert exc.value.exit_code == 3
        assert "not in a git repository" in exc.value.message

    def test_brief_path_for_a_range_and_a_root_range(self, repo):
        assert brief_path_for(BASE, HEAD, anchor=repo) == repo / FOLDER / f"{SLUG}.md"
        assert brief_path_for(None, HEAD, anchor=repo) == repo / FOLDER / f"{ROOT_SLUG}.md"
        # Anchored on a subdirectory: the same root, the same brief.
        assert brief_path_for(BASE, HEAD, anchor=repo / "src") == repo / FOLDER / f"{SLUG}.md"

    @pytest.mark.parametrize("base, head", [
        ("HEAD~1", HEAD), (BASE, "main"), (BASE, ""), ("", HEAD),
    ])
    def test_brief_path_for_refuses_endpoints_that_are_not_shas(self, repo, base, head):
        with pytest.raises(ValueError):
            brief_path_for(base, head, anchor=repo)

    def test_the_output_dir_is_any_plans_rule(self, repo):
        brief = brief_path_for(BASE, HEAD, anchor=repo)
        assert output_dir_for_brief(brief) == repo / FOLDER / SLUG
        assert output_dir_for_brief(brief) == output_dir_path(brief)
        root_brief = brief_path_for(None, HEAD, anchor=repo)
        assert output_dir_for_brief(root_brief) == repo / FOLDER / ROOT_SLUG

    def test_the_folder_relative_to_the_root(self, repo):
        assert commit_reviews_folder_rel(repo) == FOLDER

    def test_the_folder_is_none_when_the_accessor_raises(self, repo, monkeypatch):
        def refuse(anchor=None):
            raise ConfigError("multi-llm config error: code_review.commit_reviews_dir …")

        monkeypatch.setattr(commit_review, "get_commit_reviews_dir", refuse)
        assert commit_reviews_folder_rel(repo) is None


class TestPatternsMatchTheRangeGrammar:
    def test_the_local_patterns_equal_commit_ranges(self):
        """Kept here so this module never imports the range grammar; the two
        copies must stay one text."""
        commit_range = pytest.importorskip("utils.commit_range")
        assert commit_review.SLUG_PATTERN == commit_range.SLUG_PATTERN
        assert commit_review.PINNED_PATTERN == commit_range.PINNED_PATTERN

    def test_the_markers(self):
        assert BRIEF_MARKER == "<!-- multi-llm:commit-brief v1 -->"
        assert BRIEF_END_MARKER == "<!-- /multi-llm:commit-brief -->"


# ---------------------------------------------------------------------------
# Is this file a brief? (§2.2, "A brief path is accepted too, and only a brief")
# ---------------------------------------------------------------------------


class TestIsCommitBrief:
    def test_the_marker_on_line_one_anywhere(self, repo):
        brief = _write(repo / "docs" / "copied-brief.md",
                       f"{BRIEF_MARKER}\n# Commit brief\n{BRIEF_END_MARKER}\n")
        assert is_commit_brief(brief)
        assert is_commit_brief("docs/copied-brief.md")

    def test_the_marker_with_a_crlf_line_end(self, repo):
        brief = repo / "docs" / "crlf.md"
        brief.parent.mkdir(parents=True)
        brief.write_bytes(f"{BRIEF_MARKER}\r\n# x\r\n".encode("utf-8"))
        assert is_commit_brief(brief)

    def test_a_brief_in_the_folder_with_its_header_cut_away(self, repo):
        brief = _write(repo / FOLDER / f"{SLUG}.md", "# What is left\n")
        root_brief = _write(repo / FOLDER / f"{ROOT_SLUG}.md", "")
        assert is_commit_brief(brief)
        assert is_commit_brief(f"{FOLDER}/{SLUG}.md")
        assert is_commit_brief(root_brief)

    def test_relative_to_the_cwd_from_a_subdirectory(self, repo, monkeypatch):
        _write(repo / FOLDER / f"{SLUG}.md", "# cut\n")
        monkeypatch.chdir(repo / "src")
        assert is_commit_brief(f"../{FOLDER}/{SLUG}.md")
        assert is_commit_brief(f"{FOLDER}/{SLUG}.md", cwd=repo)

    def test_a_file_that_quotes_the_marker_below_line_one(self, repo):
        plan = _write(repo / "plans" / "review-commits.md",
                      f"# Plan\n\nThe brief starts with `{BRIEF_MARKER}`:\n{BRIEF_MARKER}\n")
        assert not is_commit_brief(plan)

    def test_a_plan_and_a_readme(self, repo):
        assert not is_commit_brief(repo / "plans" / "feature.md")
        assert not is_commit_brief("README.md")

    @pytest.mark.parametrize("name", [
        "notes.md", f"{SLUG}.markdown", f"{SLUG}.md.bak", "0123456789ab-fedcba98765.md",
        "ROOT-fedcba987654.md", "0123456789AB-fedcba987654.md",
    ])
    def test_a_file_in_the_folder_without_the_slug_shape(self, repo, name):
        path = _write(repo / FOLDER / name, "# not generated\n")
        assert not is_commit_brief(path)

    def test_a_brief_shape_in_another_folder(self, repo):
        assert not is_commit_brief(_write(repo / "plans" / f"{SLUG}.md", "# x\n"))
        assert not is_commit_brief(_write(repo / FOLDER / "sub" / f"{SLUG}.md", "# x\n"))

    def test_a_path_that_is_not_a_file(self, repo):
        missing = repo / FOLDER / f"{SLUG}.md"
        assert not is_commit_brief(missing)
        # ...while the shape alone still matches, for [COMMIT_REVIEW_NOT_FOUND].
        assert matches_brief_path(missing)
        assert matches_brief_path(f"{FOLDER}/{SLUG}.md")
        assert not matches_brief_path("plans/feature.md")
        (repo / FOLDER / ROOT_SLUG).mkdir(parents=True)
        assert not is_commit_brief(repo / FOLDER / ROOT_SLUG)

    def test_when_the_accessor_raises_the_marker_alone_decides(self, repo, monkeypatch):
        cut = _write(repo / FOLDER / f"{SLUG}.md", "# cut\n")
        marked = _write(repo / FOLDER / f"{ROOT_SLUG}.md", f"{BRIEF_MARKER}\n")

        def refuse(anchor=None):
            raise ConfigError("multi-llm config error: code_review.commit_reviews_dir …")

        monkeypatch.setattr(commit_review, "get_commit_reviews_dir", refuse)
        assert not is_commit_brief(cut)
        assert is_commit_brief(marked)

    @pytest.mark.config_override
    def test_a_configured_folder(self, repo):
        import utils.provider_registry as registry

        _write(repo / ".multi-llm" / "providers.yaml",
               "code_review:\n  commit_reviews_dir: reviews/commits\n")
        registry._config = None
        registry._config_key = None
        assert is_commit_brief(_write(repo / "reviews" / "commits" / f"{SLUG}.md", "# x\n"))
        assert not is_commit_brief(_write(repo / FOLDER / f"{SLUG}.md", "# x\n"))


# ---------------------------------------------------------------------------
# A pinned range names its brief (§2.2, "`status` takes the pinned spelling")
# ---------------------------------------------------------------------------


class TestBriefForPinned:
    def test_the_pinned_spelling_names_the_brief_whether_or_not_it_exists(self, repo):
        expected = repo / FOLDER / f"{SLUG}.md"
        assert brief_for_pinned("0123456789ab..fedcba987654") == expected
        assert not expected.exists()
        assert brief_for_pinned(" 0123456789ab..fedcba987654\n") == expected

    def test_from_a_subdirectory_it_is_the_same_brief(self, repo, monkeypatch):
        monkeypatch.chdir(repo / "src")
        assert brief_for_pinned("0123456789ab..fedcba987654") == repo / FOLDER / f"{SLUG}.md"
        assert brief_for_pinned("0123456789ab..fedcba987654", cwd=repo / "src") == (
            repo / FOLDER / f"{SLUG}.md"
        )

    @pytest.mark.parametrize("text", [
        "HEAD~2..HEAD", "root..fedcba9", "0123456789ab...fedcba987654",
        "0123456789AB..fedcba987654", "0123456789abc..fedcba987654",
        "0123456789a..fedcba987654", "0123456789ab..", "last 2 commits",
        f"{FOLDER}/{SLUG}.md", "plans/feature.md", "", "0123456789ab-fedcba987654",
    ])
    def test_other_text_is_not_a_pinned_range(self, repo, text):
        assert brief_for_pinned(text) is None


# ---------------------------------------------------------------------------
# A commit review's own files, by name (§2.4, "Out of scope for every review")
# ---------------------------------------------------------------------------


class TestCommitReviewArtifact:
    @pytest.mark.parametrize("path", [
        f"{FOLDER}/{SLUG}.md",
        f"{FOLDER}/{ROOT_SLUG}.md",
        f"{FOLDER}/{SLUG}/state.json",
        f"{FOLDER}/{SLUG}/code-review/report.md",
        f"{FOLDER}/{SLUG}/code-review/restarted-20261026-100000/grouped.json",
        f"{FOLDER}/{ROOT_SLUG}/context-plan.md",
        f"{FOLDER}/{SLUG}/",
        f"./{FOLDER}/{SLUG}.md",
    ])
    def test_briefs_and_everything_under_a_slug_folder(self, path):
        assert commit_review_artifact(path, FOLDER)
        assert commit_review_artifact(path, FOLDER + "/")

    @pytest.mark.parametrize("path", [
        f"{FOLDER}/notes.md",
        f"{FOLDER}/README.md",
        f"{FOLDER}/scripts/{SLUG}.md",
        f"{FOLDER}/{SLUG}.md.bak",
        f"{FOLDER}/{SLUG}.txt",
        f"{FOLDER}/not-a-slug/state.json",
        f"{FOLDER}/.gitignore",
        FOLDER,
        f"{FOLDER}/",
        f"plans/{SLUG}.md",
        f"plans/commit-reviews-old/{SLUG}.md",
        f"src/{FOLDER}/{SLUG}.md",
        "",
    ])
    def test_anything_else_is_not(self, path):
        assert not commit_review_artifact(path, FOLDER)

    def test_an_empty_folder_matches_nothing(self):
        assert not commit_review_artifact(f"{SLUG}.md", "")
        assert not commit_review_artifact(f"{SLUG}/state.json", "")

    def test_it_reads_the_string_never_the_disk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no repository, nothing on disk
        assert commit_review_artifact(f"reviews/{SLUG}/state.json", "reviews")


# ---------------------------------------------------------------------------
# The folder ignores itself (§2.4)
# ---------------------------------------------------------------------------


class TestEnsureCommitReviewsDir:
    def test_a_new_folder_gets_its_gitignore_and_git_status_is_unchanged(self, repo):
        before = _git(repo, "status", "--porcelain")

        folder, created = ensure_commit_reviews_dir(repo)

        assert (folder, created) == (repo / FOLDER, True)
        assert (folder / ".gitignore").read_text(encoding="utf-8") == "*\n"
        _write(folder / f"{SLUG}.md", f"{BRIEF_MARKER}\n")
        _write(folder / SLUG / "state.json", "{}\n")
        assert _git(repo, "status", "--porcelain") == before
        assert not (repo / "plans" / ".gitignore").exists()

    def test_a_new_parent_gets_no_gitignore(self, repo):
        _git(repo, "rm", "-q", "-r", "plans")
        _git(repo, "commit", "--quiet", "-m", "no plans")
        before = _git(repo, "status", "--porcelain")

        folder, created = ensure_commit_reviews_dir(repo / "src")

        assert created and folder == repo / FOLDER
        assert sorted(os.listdir(repo / "plans")) == ["commit-reviews"]
        assert _git(repo, "status", "--porcelain") == before

    def test_an_existing_folder_is_left_alone(self, repo):
        (repo / FOLDER).mkdir(parents=True)

        assert ensure_commit_reviews_dir(repo) == (repo / FOLDER, False)
        assert not (repo / FOLDER / ".gitignore").exists()

    def test_a_deleted_gitignore_is_not_restored(self, repo):
        folder, _ = ensure_commit_reviews_dir(repo)
        (folder / ".gitignore").unlink()

        assert ensure_commit_reviews_dir(repo) == (folder, False)
        assert not (folder / ".gitignore").exists()

    def test_a_file_in_the_folders_place_is_a_refusal(self, repo):
        _write(repo / FOLDER, "not a folder\n")
        with pytest.raises(CommitReviewError, match="is not a directory"):
            ensure_commit_reviews_dir(repo)

    @_requires_symlinks
    def test_a_folder_symlinked_out_of_the_repository_writes_nothing(
        self, repo, tmp_path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(str(outside), str(repo / FOLDER))

        with pytest.raises(ConfigError) as exc:
            ensure_commit_reviews_dir(repo)
        assert os.path.realpath(outside) in str(exc.value)
        assert list(outside.iterdir()) == []


class TestNotIgnoredWarning:
    LINE = (
        f"[COMMIT_REVIEWS_NOT_IGNORED] {FOLDER} is not ignored by git: its "
        f"briefs and reports show in git status and can be committed; add "
        f"{FOLDER}/ to .gitignore unless committing them is intended"
    )

    def test_no_folder_yet_says_nothing(self, repo):
        assert not_ignored_warning(repo / FOLDER, repo / FOLDER / f"{SLUG}.md") is None

    def test_a_folder_the_plugin_created_says_nothing(self, repo):
        folder, _ = ensure_commit_reviews_dir(repo)
        assert not_ignored_warning(folder, folder / f"{SLUG}.md") is None

    def test_an_existing_folder_git_does_not_ignore(self, repo):
        (repo / FOLDER).mkdir(parents=True)
        assert not_ignored_warning(repo / FOLDER, repo / FOLDER / f"{SLUG}.md") == self.LINE

    def test_a_deleted_gitignore_warns(self, repo):
        folder, _ = ensure_commit_reviews_dir(repo)
        (folder / ".gitignore").unlink()
        assert not_ignored_warning(folder, folder / f"{SLUG}.md") == self.LINE

    def test_a_folder_the_repository_ignores_says_nothing(self, repo):
        _write(repo / ".gitignore", f"/{FOLDER}/\n")
        (repo / FOLDER).mkdir(parents=True)
        assert not_ignored_warning(repo / FOLDER, repo / FOLDER / f"{SLUG}.md") is None

    def test_a_tracked_brief_counts_as_not_ignored(self, repo):
        _write(repo / ".gitignore", f"/{FOLDER}/\n")
        brief = _write(repo / FOLDER / f"{SLUG}.md", f"{BRIEF_MARKER}\n")
        _git(repo, "add", "-f", str(brief))
        assert not_ignored_warning(repo / FOLDER, brief) == self.LINE

    def test_it_writes_nothing(self, repo):
        (repo / FOLDER).mkdir(parents=True)
        not_ignored_warning(repo / FOLDER, repo / FOLDER / f"{SLUG}.md")
        assert os.listdir(repo / FOLDER) == []


# ---------------------------------------------------------------------------
# "Commit brief", never "plan" (§2.8)
# ---------------------------------------------------------------------------


class TestDocumentLabel:
    def test_a_state_with_a_commit_range_names_the_commit_brief(self):
        assert document_label({"commit_range": {"base": None, "head": HEAD}}) == "commit brief"

    @pytest.mark.parametrize("state", [None, {}, {"commit_range": None}, {"plan_path": "x.md"}])
    def test_anything_else_is_a_plan(self, state):
        assert document_label(state) == "plan"

    def test_a_state_manager(self, repo):
        from utils.state_manager import StateManager

        plan = repo / "plans" / "feature.md"
        manager = StateManager(plan)
        assert document_label(manager) == "plan"
        manager.state["commit_range"] = {"base": BASE, "head": HEAD}
        assert document_label(manager) == "commit brief"


# ---------------------------------------------------------------------------
# The class drop (§2.4, "Out of scope for every review, as a class")
# ---------------------------------------------------------------------------


REVIEW_FILES = [
    f"{FOLDER}/{SLUG}.md",
    f"{FOLDER}/{SLUG}/code-review/report.md",
    f"{FOLDER}/{SLUG}/state.json",
]
NOTES = f"{FOLDER}/notes.md"


@pytest.fixture
def committed_review(repo):
    """An un-ignored earlier review committed beside a user's ``notes.md``.

    Returns ``(repo, base)``: *base* is the commit before the one that added
    the review files, the brief, its ``state.json`` and its report, the notes
    and a source change.
    """
    base = _git(repo, "rev-parse", "HEAD").strip()
    _write(repo / REVIEW_FILES[0], f"{BRIEF_MARKER}\n# brief\n{BRIEF_END_MARKER}\n")
    _write(repo / REVIEW_FILES[1], "# Code review\n")
    _write(repo / REVIEW_FILES[2], "{}\n")
    _write(repo / NOTES, "# My notes about reviews\n")
    _write(repo / "src" / "app.py", "a = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "commit a review")
    return repo, base


class TestTheClassDropInTheReviewScope:
    def test_drop_workflow_paths_keeps_only_the_users_file(self, committed_review, capsys):
        repo, _base = committed_review
        paths = REVIEW_FILES + [NOTES, "src/app.py"]

        kept = cro._drop_workflow_paths(paths, [], repo_root=str(repo))

        assert kept == [NOTES, "src/app.py"]
        out = capsys.readouterr().out
        assert "Left out 3 of the workflow's own file(s)" in out
        for path in REVIEW_FILES:
            assert path in out
        assert NOTES not in out

    def test_beside_the_runs_own_workflow_entries_and_counted_with_them(
        self, committed_review, capsys
    ):
        repo, _base = committed_review
        kept = cro._drop_workflow_paths(
            REVIEW_FILES + [NOTES, "plans/feature.md"], ["plans/feature.md"],
            repo_root=str(repo),
        )
        assert kept == [NOTES]
        assert "Left out 4 of the workflow's own file(s)" in capsys.readouterr().out

    def test_a_diff_across_the_commit_with_a_state_lacking_state_file(
        self, committed_review, capsys
    ):
        """A plain `--review-code --base-ref` over the commit: the git-diff
        fallback, a plain dict state (no `state_file` attribute) and a plan
        unrelated to the committed review."""
        repo, base = committed_review
        state = {"tracked_files": [], "pre_existing_changes": []}

        scope = cro.resolve_review_scope(
            state, base, anchor=str(repo / "plans" / "feature.md"), repo_root=str(repo),
        )

        assert scope.source == "git_diff"
        assert sorted(scope.paths) == [NOTES, "src/app.py"]
        out = capsys.readouterr().out
        assert "Left out 3 of the workflow's own file(s)" in out

    def test_tracked_files_naming_the_review_keep_only_the_users_file(
        self, committed_review, capsys
    ):
        repo, _base = committed_review
        state = {"tracked_files": [
            {"path": path, "task_id": "implementation"}
            for path in REVIEW_FILES + [NOTES, "src/app.py"]
        ]}

        scope = cro.resolve_review_scope(state, "HEAD~1", repo_root=str(repo))

        assert scope.source == "tracked_files"
        assert sorted(scope.paths) == [NOTES, "src/app.py"]

    def test_when_the_accessor_raises_nothing_extra_is_dropped(
        self, committed_review, monkeypatch, capsys
    ):
        repo, _base = committed_review

        def refuse(anchor=None):
            raise ConfigError("multi-llm config error: code_review.commit_reviews_dir …")

        monkeypatch.setattr(commit_review, "get_commit_reviews_dir", refuse)
        paths = REVIEW_FILES + [NOTES]
        assert cro._drop_workflow_paths(paths, [], repo_root=str(repo)) == paths
        assert "Left out" not in capsys.readouterr().out


class TestTheClassDropInTrackedFiles:
    @pytest.fixture
    def state_path(self, committed_review, tmp_path):
        """A run's state file outside the repository (not itself a dirty path),
        whose plan is unrelated to the committed review."""
        repo, base = committed_review
        path = tmp_path / "run" / "state.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "plan_path": str(repo / "plans" / "feature.md"),
            "pre_existing_changes": [],
            "head_before_implement": base,
        }), encoding="utf-8")
        return path

    @staticmethod
    def _recorded(state_path):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return [entry["path"] for entry in state["tracked_files"]]

    @pytest.mark.parametrize("only_touched", [False, True], ids=["census", "only-touched"])
    def test_given_all_four_only_the_notes_are_recorded(
        self, state_path, capsys, only_touched
    ):
        argv = ["--state-file", str(state_path)]
        if only_touched:
            argv.append("--only-touched")
        for path in REVIEW_FILES + [NOTES]:
            argv += ["--touched", path]

        assert finalize_tracking.main(argv) == 0

        assert self._recorded(state_path) == [NOTES]
        out = capsys.readouterr().out
        assert "Left out 3 of the workflow's own file(s)" in out

    def test_only_the_review_files_is_refused_as_workflow_files(self, state_path, capsys):
        argv = ["--state-file", str(state_path), "--only-touched"]
        for path in REVIEW_FILES:
            argv += ["--touched", path]

        assert finalize_tracking.main(argv) == 1
        assert "only the workflow's own files" in capsys.readouterr().err

    def test_when_the_accessor_raises_nothing_extra_is_dropped(
        self, committed_review, monkeypatch
    ):
        def refuse(anchor=None):
            raise ConfigError("multi-llm config error: code_review.commit_reviews_dir …")

        monkeypatch.setattr(commit_review, "get_commit_reviews_dir", refuse)
        paths = REVIEW_FILES + [NOTES]
        assert finalize_tracking._drop_workflow_files(paths, [], lambda _line: None) == paths
