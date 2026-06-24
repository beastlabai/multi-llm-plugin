"""Git utilities for the multi-llm skill."""

import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

class GitError(Exception):
    """Raised when a git operation fails."""
    pass

def _run_git(*args: str, check: bool = True) -> Tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout, result.stderr, result.returncode

def get_project_root(from_path: str) -> Optional[str]:
    """Detect project root via git rev-parse --show-toplevel.

    Args:
        from_path: A file path within the project (used to determine
                   which directory to run git from).

    Returns:
        Absolute path to the project root, or None if not in a git repo.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
        cwd=os.path.dirname(from_path) or "."
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def get_project_root_from_dir(directory: Optional[str] = None) -> Optional[str]:
    """Detect the git project root by running git from a *directory* directly.

    Unlike :func:`get_project_root`, which takes a *file* path and uses its
    ``dirname`` to choose where to run git, this takes a directory and runs
    ``git rev-parse --show-toplevel`` there with no path manipulation. Pass a
    directory (e.g. CWD or a target-repo dir) directly without the brittle
    synthetic-path trick (``get_project_root(os.path.join(cwd, "_"))``) that
    relies on ``dirname`` stripping a fake segment.

    Args:
        directory: Directory to run git discovery from. Defaults to CWD.

    Returns:
        Absolute path to the project root, or None if not in a git repo.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
        cwd=directory or "."
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def get_modified_files() -> List[str]:
    """Get list of modified and untracked files."""
    stdout, _, _ = _run_git("status", "--porcelain")
    files = []
    for line in stdout.strip().split("\n"):
        if line:
            status = line[:2]
            filename = line[3:]
            if status.strip():
                files.append(filename)
    return files

def get_staged_files() -> List[str]:
    """Get list of currently staged files."""
    stdout, _, _ = _run_git("diff", "--cached", "--name-only")
    return [f for f in stdout.strip().split("\n") if f]

def stage_files(files: List[str]) -> None:
    """Stage specific files."""
    if files:
        _run_git("add", "--", *files)

def unstage_files(files: List[str]) -> None:
    """Unstage specific files."""
    if files:
        _run_git("reset", "HEAD", "--", *files)


@contextmanager
def intent_to_add_untracked(file_paths: List[str]) -> Iterator[List[str]]:
    """Temporarily mark untracked files as intent-to-add so `git diff` picks them up.

    `git diff` (all variants -- `git diff`, `git diff HEAD`, `git diff A..B`) silently
    ignores fully-untracked files. For code review flows that rely on git diff to
    surface implementation changes, this means new files never show up. Marking them
    with `git add -N` creates an empty index entry so diffs treat them as full-file
    additions without staging any content.

    On exit, any files this context manager added are reset back to untracked via
    `git reset HEAD -- <file>`. Files that were already in the index (staged, or
    already intent-to-add by the user) are never touched.

    Args:
        file_paths: Candidate files to consider. Files not currently untracked are
            skipped. A file listed here but missing from the working tree is also
            skipped.

    Yields:
        The list of files this context manager actually marked (may be empty).
    """
    touched: List[str] = []
    if file_paths:
        try:
            stdout, _, rc = _run_git(
                "ls-files", "--others", "--exclude-standard", "--", *file_paths,
                check=False,
            )
            if rc == 0:
                touched = [line for line in stdout.splitlines() if line.strip()]
        except Exception as exc:
            print(
                f"Warning: could not list untracked files for intent-to-add: {exc}",
                file=sys.stderr,
            )
            touched = []

        if touched:
            try:
                _run_git("add", "-N", "--", *touched)
                print(
                    f"Marked {len(touched)} untracked file(s) as intent-to-add "
                    f"so review can see their contents"
                )
            except GitError as exc:
                print(
                    f"Warning: git add -N failed ({exc}); untracked files may be "
                    f"invisible to reviewers",
                    file=sys.stderr,
                )
                touched = []
    try:
        yield touched
    finally:
        if touched:
            try:
                _run_git("reset", "HEAD", "--", *touched)
            except GitError as exc:
                print(
                    f"Warning: failed to reset intent-to-add entries for "
                    f"{len(touched)} file(s): {exc}. Run `git reset HEAD -- "
                    f"{' '.join(touched)}` to clean up manually.",
                    file=sys.stderr,
                )

def get_staged_diff() -> str:
    """Get diff of all staged changes."""
    stdout, _, _ = _run_git("diff", "--cached")
    return stdout

def get_file_diff(file_path: str, base_ref: Optional[str] = None) -> str:
    """Get diff for a specific file."""
    if base_ref:
        stdout, _, _ = _run_git("diff", base_ref, "--", file_path)
    else:
        stdout, _, _ = _run_git("diff", "--", file_path)
    return stdout

def get_current_head() -> str:
    """Get current HEAD commit hash."""
    stdout, _, _ = _run_git("rev-parse", "HEAD")
    return stdout.strip()

def get_branch_name() -> Optional[str]:
    """Get current branch name, or None if detached HEAD."""
    stdout, _, code = _run_git("symbolic-ref", "--short", "HEAD", check=False)
    if code == 0:
        return stdout.strip()
    return None

def is_clean_working_tree() -> bool:
    """Check if working tree is clean (no uncommitted changes)."""
    _, _, code = _run_git("diff", "--quiet", check=False)
    if code != 0:
        return False
    _, _, code = _run_git("diff", "--cached", "--quiet", check=False)
    return code == 0

def get_diff_since_ref(base_ref: str) -> str:
    """Get diff of all changes since a reference commit."""
    stdout, _, _ = _run_git("diff", base_ref)
    return stdout

def get_files_changed_since_ref(base_ref: str) -> List[str]:
    """Get list of files changed since a reference commit."""
    stdout, _, _ = _run_git("diff", "--name-only", base_ref)
    return [f for f in stdout.strip().split("\n") if f]


def validate_git_ref(ref: str) -> str:
    """Validate a git ref using git rev-parse --verify.

    If the ref is invalid, falls back to HEAD~1. If HEAD~1 is also invalid
    (or the original ref was already HEAD~1), returns an empty string.

    Args:
        ref: The git ref to validate (e.g., a commit hash, branch name, HEAD~1).

    Returns:
        The validated ref if valid, 'HEAD~1' as fallback, or empty string
        if both the ref and HEAD~1 are invalid.
    """
    if not ref:
        return ""

    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            check=True,
            capture_output=True,
            text=True,
        )
        return ref
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"Warning: git ref '{ref}' is invalid, falling back to HEAD~1",
            file=sys.stderr,
        )

    # If the ref was already HEAD~1, don't redundantly retry it
    if ref == "HEAD~1":
        print(
            "Warning: HEAD~1 is not a valid git ref (shallow clone or initial commit?), "
            "using empty base_ref",
            file=sys.stderr,
        )
        return ""

    # Fallback to HEAD~1
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD~1"],
            check=True,
            capture_output=True,
            text=True,
        )
        return "HEAD~1"
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Warning: HEAD~1 is also not a valid git ref (shallow clone or initial commit?), "
            "using empty base_ref",
            file=sys.stderr,
        )
        return ""


# ---------------------------------------------------------------------------
# Diff hunk capture and file content capture utilities
# ---------------------------------------------------------------------------

# Budget constants for embedded context data
_MAX_LINES_PER_FILE = 200
_MAX_TOTAL_BYTES = 500 * 1024  # 500KB
_GIT_TIMEOUT_SECONDS = 10


def _is_binary_file(path: str) -> bool:
    """Heuristic check for binary file content.

    Reads the first 8KB and looks for null bytes.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except (OSError, IOError):
        return False


def _parse_unified_diff(diff_text: str) -> Dict[str, Any]:
    """Parse unified diff text for a single file into structured hunk data.

    Returns a dict with keys:
      - hunks: list of hunk objects (header + lines)
      - old_path / new_path: for renamed files
      - binary: True if the diff indicates a binary file
      - deleted: True if the file was deleted
      - truncated_at: int if lines were capped

    Each hunk contains:
      - header: the @@ line
      - lines: list of {type, old_line, new_line, content}
    """
    lines = diff_text.split("\n")
    result: Dict[str, Any] = {
        "hunks": [],
        "old_path": None,
        "new_path": None,
        "binary": False,
        "deleted": False,
    }

    total_content_lines = 0
    truncated = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect binary file marker
        if line.startswith("Binary files") or line.startswith("GIT binary patch"):
            result["binary"] = True
            return result

        # Detect rename
        if line.startswith("rename from "):
            result["old_path"] = line[len("rename from "):]
        elif line.startswith("rename to "):
            result["new_path"] = line[len("rename to "):]

        # Detect old/new paths from --- / +++ headers
        if line.startswith("--- a/"):
            result["old_path"] = result["old_path"] or line[6:]
        elif line.startswith("--- /dev/null"):
            # New file – no old path
            pass
        elif line.startswith("+++ b/"):
            result["new_path"] = result["new_path"] or line[6:]
        elif line.startswith("+++ /dev/null"):
            result["deleted"] = True

        # Parse hunk header
        hunk_match = re.match(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line
        )
        if hunk_match:
            old_start = int(hunk_match.group(1))
            new_start = int(hunk_match.group(3))
            hunk_header_text = line

            hunk: Dict[str, Any] = {
                "header": hunk_header_text,
                "lines": [],
            }

            old_line = old_start
            new_line = new_start

            i += 1
            while i < len(lines):
                hline = lines[i]

                # Stop at next hunk, next file header, or end
                if (
                    hline.startswith("@@ ")
                    or hline.startswith("diff --git")
                    or hline.startswith("--- a/")
                    or hline.startswith("--- /dev/null")
                ):
                    break

                if truncated:
                    i += 1
                    continue

                if total_content_lines >= _MAX_LINES_PER_FILE:
                    truncated = True
                    result["truncated_at"] = _MAX_LINES_PER_FILE
                    i += 1
                    continue

                if hline.startswith("+"):
                    hunk["lines"].append({
                        "type": "add",
                        "old_line": None,
                        "new_line": new_line,
                        "content": hline[1:],
                    })
                    new_line += 1
                    total_content_lines += 1
                elif hline.startswith("-"):
                    hunk["lines"].append({
                        "type": "remove",
                        "old_line": old_line,
                        "new_line": None,
                        "content": hline[1:],
                    })
                    old_line += 1
                    total_content_lines += 1
                elif hline.startswith(" "):
                    hunk["lines"].append({
                        "type": "context",
                        "old_line": old_line,
                        "new_line": new_line,
                        "content": hline[1:],
                    })
                    old_line += 1
                    new_line += 1
                    total_content_lines += 1
                elif hline.startswith("\\ No newline at end of file"):
                    # Marker line – skip, don't count toward budget
                    pass
                else:
                    # Unknown line in hunk (e.g., empty trailing line) – skip
                    pass

                i += 1

            result["hunks"].append(hunk)
            continue  # Don't increment i again – already positioned

        i += 1

    return result


def _split_diff_by_file(raw_diff: str) -> Dict[str, str]:
    """Split a multi-file unified diff into per-file chunks.

    Returns a dict mapping file paths to their diff text.
    The key is the new-side path (b/ path) from the diff header.
    """
    file_diffs: Dict[str, str] = {}
    current_path: Optional[str] = None
    current_lines: List[str] = []

    for line in raw_diff.split("\n"):
        if line.startswith("diff --git"):
            # Save previous file's diff
            if current_path is not None:
                file_diffs[current_path] = "\n".join(current_lines)

            # Extract path from "diff --git a/path b/path"
            parts = line.split(" b/", 1)
            current_path = parts[1] if len(parts) > 1 else None
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last file
    if current_path is not None:
        file_diffs[current_path] = "\n".join(current_lines)

    return file_diffs


def capture_diff_hunks(
    base_ref: str,
    file_paths: List[str],
) -> Dict[str, Any]:
    """Capture structured diff hunks for specified files against a base ref.

    Runs ``git diff base_ref -- <files>`` and parses the unified diff output
    into structured per-file data consumed by the PR-style report template.

    Args:
        base_ref: Git ref to diff against (commit SHA, branch name, HEAD~N).
        file_paths: List of file paths to capture diffs for.  Only files in
            this list are processed (scoped to suggestion-referenced files).

    Returns:
        A dict with:
          - Per file-path key: structured hunk data (see ``_parse_unified_diff``).
          - ``_notices``: list of notice strings (e.g., uncommitted changes).
          - ``_error``: error string when git is unavailable.

    Edge cases handled:
      - Binary files → ``{binary: true}``
      - Renamed files → ``old_path`` and ``new_path`` included
      - Deleted files → ``deleted: true``, content retrieved via ``git show``
      - Invalid base_ref → empty dict with warning
      - Git not available → empty dict with notice
      - 10-second timeout enforced
      - 200-line per-file cap
      - 500KB total budget
    """
    if not file_paths:
        return {}

    result: Dict[str, Any] = {}
    notices: List[str] = []

    # --- Check for uncommitted changes ---
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        if status_result.returncode == 0 and status_result.stdout.strip():
            notices.append("Diff includes uncommitted changes")
    except FileNotFoundError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (not installed or not in PATH)",
            file=sys.stderr,
        )
        return result
    except subprocess.TimeoutExpired:
        notices.append("git status timed out")
    except OSError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (OS error)",
            file=sys.stderr,
        )
        return result

    # --- Run git diff ---
    try:
        diff_result = subprocess.run(
            ["git", "diff", base_ref, "--"] + list(file_paths),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (not installed or not in PATH)",
            file=sys.stderr,
        )
        return result
    except subprocess.TimeoutExpired:
        notices.append("diff capture timed out")
        result["_notices"] = notices
        return result
    except OSError:
        result["_notices"] = ["Git not available -- code context shown from file content only."]
        print(
            "Warning: git is not available (OS error)",
            file=sys.stderr,
        )
        return result

    if diff_result.returncode != 0:
        # Invalid base_ref or other git error
        print(
            f"Warning: git diff failed for base_ref '{base_ref}': "
            f"{diff_result.stderr.strip()}",
            file=sys.stderr,
        )
        result["_notices"] = notices
        return result

    raw_diff = diff_result.stdout

    # --- Split and parse per-file diffs ---
    per_file_diffs = _split_diff_by_file(raw_diff)
    total_bytes = 0

    start_time = time.monotonic()

    for fpath in file_paths:
        # Timeout check
        if time.monotonic() - start_time > _GIT_TIMEOUT_SECONDS:
            notices.append("diff capture timed out")
            break

        file_diff_text = per_file_diffs.get(fpath)
        if file_diff_text is None:
            # File may not have changes in the diff – skip silently
            continue

        # Budget check
        diff_bytes = len(file_diff_text.encode("utf-8", errors="replace"))
        if total_bytes + diff_bytes > _MAX_TOTAL_BYTES:
            result[fpath] = {"truncated": True}
            notices.append(
                f"Size budget exceeded at {fpath} – remaining files omitted"
            )
            break

        total_bytes += diff_bytes

        parsed = _parse_unified_diff(file_diff_text)
        result[fpath] = parsed

    # --- Handle deleted files: retrieve last-known content ---
    for fpath in file_paths:
        if fpath in result and isinstance(result[fpath], dict) and result[fpath].get("deleted"):
            try:
                show_result = subprocess.run(
                    ["git", "show", f"{base_ref}:{fpath}"],
                    capture_output=True,
                    text=True,
                    timeout=_GIT_TIMEOUT_SECONDS,
                )
                if show_result.returncode == 0:
                    result[fpath]["last_known_content"] = show_result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass  # Best effort – content unavailable

    if notices:
        result["_notices"] = notices

    return result


def capture_file_snapshots(
    file_paths: List[str],
) -> Dict[str, List[str]]:
    """Capture file content for multiple files at a point in time.

    Call this at the START of a review phase (before model invocations begin)
    to snapshot file content.  The returned dict can be passed as
    ``file_snapshots`` to ``generate_html_report()`` and ultimately to
    ``capture_file_context()``, ensuring the report reflects file state at
    review start rather than at report-generation time.

    Args:
        file_paths: List of file paths (absolute or relative to cwd) to
            snapshot.

    Returns:
        Dict mapping each readable, non-binary file path to a list of its
        lines (with trailing newlines stripped).  Files that do not exist,
        are binary, or cannot be read are silently omitted.
    """
    snapshots: Dict[str, List[str]] = {}
    for fpath in file_paths:
        if not fpath or not os.path.isfile(fpath):
            continue
        if _is_binary_file(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                snapshots[fpath] = [
                    ln.rstrip("\n").rstrip("\r") for ln in f.readlines()
                ]
        except (OSError, IOError):
            continue
    return snapshots


def capture_file_context(
    file_path: str,
    line_range: List[int],
    context_lines: int = 5,
    file_snapshots: Optional[Dict[str, List[str]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Read lines around a line range from a file.

    Returns an array of ``{line_number: int, content: str}`` objects covering
    ``context_lines`` before and after the specified ``line_range``.

    When ``file_snapshots`` is provided and contains ``file_path``, the
    pre-captured snapshot is used instead of reading the file from disk.
    This allows callers to capture file content at the START of a review
    phase (via ``capture_file_snapshots()``) and avoid stale data if files
    are modified between review start and report generation.

    When no snapshot is available, falls back to reading the file from disk
    at call time.  For accurate context when files may have been modified
    after review start, prefer providing ``file_snapshots`` or using
    ``capture_diff_hunks`` with a ``base_ref`` (which uses git history).

    Args:
        file_path: Path to the file on disk (absolute or relative to cwd).
        line_range: Two-element list ``[start_line, end_line]`` (1-indexed,
            inclusive).
        context_lines: Number of extra lines to include before ``start_line``
            and after ``end_line``.  Defaults to 5.
        file_snapshots: Optional dict mapping file paths to pre-captured
            line lists (as returned by ``capture_file_snapshots()``).  When
            the file is found in this dict, its snapshot is used instead of
            reading from disk.

    Returns:
        List of ``{line_number, content}`` dicts, or ``None`` if the file
        does not exist or is binary.
    """
    if not file_path or not line_range:
        return None

    # Try snapshot first
    if file_snapshots and file_path in file_snapshots:
        all_lines = file_snapshots[file_path]
    else:
        # Fall back to reading from disk
        if not os.path.isfile(file_path):
            return None

        # Check for binary content
        if _is_binary_file(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = [
                    ln.rstrip("\n").rstrip("\r") for ln in f.readlines()
                ]
        except (OSError, IOError):
            return None

    if not all_lines:
        return None

    start_line = line_range[0]
    end_line = line_range[1] if len(line_range) > 1 else start_line

    # Clamp to valid range
    context_start = max(1, start_line - context_lines)
    context_end = end_line + context_lines

    # Clamp end to file length
    context_end = min(context_end, len(all_lines))

    result = []
    for i in range(context_start - 1, context_end):
        if i < 0 or i >= len(all_lines):
            continue
        result.append({
            "line_number": i + 1,
            "content": all_lines[i],
        })

    return result if result else None
