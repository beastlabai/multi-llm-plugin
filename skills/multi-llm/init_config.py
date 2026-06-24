#!/usr/bin/env python3
"""Opt-in scaffolder for a per-project multi-llm config override.

Writes a fully-commented ``<dir>/.multi-llm/providers.yaml`` stub that lets a
repository override multi-llm's *selection* defaults (default_provider,
defaults.models, defaults.quick_models, defaults.modes) without editing the
installed plugin. The override file is optional and auto-discovered from the git
root at run time; this script just creates the commented starting point.

Usage (the way the skill invokes it):

    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py \\
        [--dir PATH] [--force] [--gitignore]

By default the file is left *trackable* (commit it for a team-wide, repo-standard
selection). Pass --gitignore to instead keep it as a developer-local, untracked
override.
"""
import argparse
import os
import sys
from pathlib import Path

# The script ships inside the skill but runs with the *user's* repo as CWD.
# Make the skill's own packages importable regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).parent))

from utils.git_utils import get_project_root_from_dir  # noqa: E402

# Template is resolved strictly relative to THIS file, never CWD / --project dir
# (both diverge from the install location). If this moves, update the Section 4
# packaging-guard test in tests/test_provider_registry.py to match.
TEMPLATE_PATH = Path(__file__).parent / "templates" / "config" / "providers.override.yaml"

CONFIG_DIRNAME = ".multi-llm"
CONFIG_FILENAME = "providers.yaml"
GITIGNORE_ENTRY = ".multi-llm/"


def resolve_target_dir(requested: "str | None") -> Path:
    """Resolve the directory to scaffold into.

    Defaults to the git root; falls back to CWD (with a printed notice) outside a
    repo. An explicit --dir is honored verbatim.
    """
    if requested:
        return Path(requested).expanduser().resolve()

    root = get_project_root_from_dir(os.getcwd())
    if root:
        return Path(root)

    cwd = Path(os.getcwd()).resolve()
    print(
        f"NOTE: not inside a git repository — using the current directory: {cwd}\n"
        f"      (auto-discovery only finds the override inside a git repo; outside "
        f"one, point MULTI_LLM_PROVIDERS_CONFIG at it instead).",
        file=sys.stderr,
    )
    return cwd


def _gitignore_has_entry(gitignore_path: Path) -> bool:
    """Return True if .gitignore already ignores the .multi-llm/ directory."""
    if not gitignore_path.exists():
        return False
    target = GITIGNORE_ENTRY.strip().strip("/")
    for raw in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lstrip("/").rstrip("/") == target:
            return True
    return False


def append_gitignore(target_dir: Path) -> "str | None":
    """Idempotently append the ignore entry to ``<target_dir>/.gitignore``.

    Creates the file if absent. Returns a message describing what happened, or
    None if no change was needed.
    """
    gitignore_path = target_dir / ".gitignore"
    if _gitignore_has_entry(gitignore_path):
        return None

    existing = ""
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"

    block = "# multi-llm per-project config (developer-local, untracked)\n" + GITIGNORE_ENTRY + "\n"
    gitignore_path.write_text(existing + block, encoding="utf-8")
    return f"Added '{GITIGNORE_ENTRY}' to {gitignore_path}"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="init_config.py",
        description=(
            "Scaffold a commented per-project multi-llm config override at "
            "<dir>/.multi-llm/providers.yaml. The file is optional and overrides "
            "only selection defaults (providers/models picked for a run)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help="Target directory (defaults to the git root; falls back to CWD outside a repo).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing .multi-llm/providers.yaml (refuses by default).",
    )
    parser.add_argument(
        "--gitignore",
        action="store_true",
        help=(
            "Also append '.multi-llm/' to the repo's .gitignore (idempotent) so the "
            "override stays a developer-local, untracked file. Default: leave it trackable."
        ),
    )
    args = parser.parse_args(argv)

    if not TEMPLATE_PATH.exists():
        print(
            f"ERROR: config template not found at {TEMPLATE_PATH}\n"
            f"       The plugin install looks corrupt or partial; reinstall the skill.",
            file=sys.stderr,
        )
        return 1

    target_dir = resolve_target_dir(args.dir)
    config_dir = target_dir / CONFIG_DIRNAME
    config_path = config_dir / CONFIG_FILENAME

    if config_path.exists() and not args.force:
        print(
            f"ERROR: {config_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {config_path}")

    # Tracking-state handling: do nothing to .gitignore unless --gitignore.
    if args.gitignore:
        result = append_gitignore(target_dir)
        if result:
            print(result)
        else:
            print(f"'{GITIGNORE_ENTRY}' already present in .gitignore — no change.")
        print("This file is IGNORED by git (developer-local, untracked).")
    else:
        print(
            "This file is TRACKED by git by default — commit it for a team-wide, "
            "repo-standard selection.\n"
            "Re-run with --gitignore (or use MULTI_LLM_PROVIDERS_CONFIG for an "
            "out-of-tree path) to keep personal, per-developer preferences untracked."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
