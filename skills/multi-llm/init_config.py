#!/usr/bin/env python3
"""Opt-in scaffolder for a per-project multi-llm config override.

On a TTY this walks the user through an **interactive** setup: it detects which
provider CLIs are installed, lets them pick the default + ``--quick`` model panels
(curated first, full CLI catalog opt-in behind "Show all…"), and writes the chosen
*selection* keys into ``<git-root>/.multi-llm/providers.yaml``.

Off a TTY — or with ``--template-only`` / ``--non-interactive`` — it falls back to
copying the fully-commented template stub verbatim (the historical behavior), so a
repository can still hand-edit ``default_provider`` / ``defaults.models`` /
``defaults.quick_models`` / ``defaults.modes`` without editing the installed plugin.
The override file is optional and auto-discovered from the git root at run time.

Usage (the way the skill invokes it):

    uv run --project ${CLAUDE_SKILL_DIR} -- python ${CLAUDE_SKILL_DIR}/init_config.py \\
        [--dir PATH] [--force] [--gitignore] \\
        [--template-only] [--non-interactive] [--timeout SECONDS]

By default the file is left *trackable* (commit it for a team-wide, repo-standard
selection). Pass --gitignore to instead keep it as a developer-local, untracked
override.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

# The script ships inside the skill but runs with the *user's* repo as CWD.
# Make the skill's own packages importable regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).parent))

from utils.git_utils import get_project_root_from_dir  # noqa: E402
from utils.interactive import (  # noqa: E402
    ACTION_ENTER_MANUAL,
    ACTION_SHOW_ALL,
    is_tty,
    prompt_text,
    prompt_yes_no,
    select_multi,
    select_one,
    select_with_actions,
)
from utils.provider_registry import get_provider, load_base_config  # noqa: E402

# Template is resolved strictly relative to THIS file, never CWD / --project dir
# (both diverge from the install location). If this moves, update the Section 4
# packaging-guard test in tests/test_provider_registry.py to match.
TEMPLATE_PATH = Path(__file__).parent / "templates" / "config" / "providers.override.yaml"

CONFIG_DIRNAME = ".multi-llm"
CONFIG_FILENAME = "providers.yaml"
GITIGNORE_ENTRY = ".multi-llm/"

# Delimited region of the template that the interactive writer replaces. The
# header comments (and the MODE SHADOWING note) live OUTSIDE these markers and are
# copied verbatim, so a re-init never clobbers them. Keep in sync with the markers
# in templates/config/providers.override.yaml.
MARKER_START = "# <<< multi-llm:init-managed >>>"
MARKER_END = "# <<< /multi-llm:init-managed >>>"

# Sentinel labels shown in the per-provider picker first pass.
SHOW_ALL_LABEL_FMT = "⤵  Show all models ({name} models)…"
MANUAL_LABEL = "✎  Enter a model id manually…"

# Above this many rows, the no-gum/no-fzf numbered fallback warns before dumping a
# full "Show all…" catalog (the only place a long flat list can appear — §4).
LONG_LIST_WARN_THRESHOLD = 40


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


def _report_gitignore(args: argparse.Namespace, target_dir: Path) -> None:
    """Shared tracking-state messaging for both the template and interactive paths."""
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


# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------


def _base_catalog_specs(providers_cfg: dict) -> set:
    """Set of every ``provider:model`` spec present in the base catalog."""
    specs = set()
    for name, cfg in providers_cfg.items():
        for model in (cfg.get("models", []) or []):
            specs.add(f"{name}:{model}")
    return specs


def _shadowed_modes(base: dict, target_dir: Path) -> "list[str]":
    """Modes whose mode-specific list will still win over the globals we write.

    The plan (§Out of scope, acceptance criteria) wants the confirm step to warn
    about ``defaults.modes`` present in the **base** config OR in any discovered
    parent override on the resolution path — the project-local
    ``.multi-llm/providers.yaml`` and the ``MULTI_LLM_PROVIDERS_CONFIG`` env layer.
    Reading only ``load_base_config()`` misses modes a user set via those override
    layers, which still shadow the freshly-written globals at runtime.

    We read the same override layers ``load_config`` discovers (anchored on
    ``target_dir``) directly, so a pre-existing project file or env override that
    carries ``defaults.modes`` is surfaced too. Best-effort: any failure falls back
    to the base-config modes alone (never raises).
    """
    modes: set = set((base.get("defaults", {}) or {}).get("modes", {}) or {})
    try:
        import utils.provider_registry as registry

        permissive = registry._truthy_env(registry.ENV_PERMISSIVE_VAR)
        for layer_name, path in registry._override_paths(anchor=str(target_dir)):
            layer = registry._load_override_layer(layer_name, path, permissive=permissive)
            if not isinstance(layer, dict):
                continue
            layer_modes = (layer.get("defaults", {}) or {}).get("modes", {}) or {}
            if isinstance(layer_modes, dict):
                modes |= set(layer_modes)
    except Exception:
        # Override-layer probing is advisory; fall back to base-config modes only.
        pass
    return sorted(modes)


def _available_providers(providers_cfg: dict) -> "list[str]":
    """Base-config provider names whose CLI is installed, in declaration order."""
    available = []
    for name in providers_cfg:  # dict preserves base-config declaration order
        provider = get_provider(name)
        if provider is not None and provider.is_available():
            available.append(name)
    return available


def _have_fuzzy_picker() -> bool:
    """True if gum or fzf is available (so long lists get a fuzzy filter)."""
    return bool(shutil.which("gum") or shutil.which("fzf"))


def _manual_entry_loop(provider_name: str) -> "list[str]":
    """Collect free-text bare model ids for ``provider_name`` (blank to finish)."""
    entries: list[str] = []
    print(f"Enter {provider_name} model ids one per line (blank line to finish):")
    while True:
        value = prompt_text(f"  {provider_name} model id:")
        if not value:
            break
        entries.append(value)
    return entries


def _pick_provider_models(provider, curated: "list[str]", timeout: int) -> "tuple[list[str], bool]":
    """Two-tier picker for one provider → (chosen BARE model ids, manual-origin flag).

    The second element is True when the ids came from the "Enter manually…"
    free-text sentinel; those ids are always flagged "(unverified id)" at confirm
    time per §5a, even when they happen to exist in the base catalog.

    First pass shows the curated rows plus sentinels ("Show all…" only when the
    provider can list, "Enter manually…" always). Sentinels are exclusive
    (select_with_actions enforces the precedence); a cancel skips the provider.
    """
    show_all_label = (
        SHOW_ALL_LABEL_FMT.format(name=provider.name) if provider.can_list_models else None
    )
    result = select_with_actions(
        list(curated),
        f"Select {provider.name} models (curated):",
        show_all_label=show_all_label,
        manual_label=MANUAL_LABEL,
    )

    if result.cancelled:
        return [], False

    if result.action == ACTION_SHOW_ALL:
        listing = provider.list_models(curated, timeout=timeout)
        if listing.note:
            print(f"  note: {listing.note}")
        full = listing.models or list(curated)
        if not _have_fuzzy_picker() and len(full) > LONG_LIST_WARN_THRESHOLD:
            print(
                f"  (no gum/fzf installed — showing all {len(full)} {provider.name} "
                f"models as a numbered list; install gum or fzf for fuzzy filtering)"
            )
        picks = select_multi(full, f"Select {provider.name} models (type to filter):")
        return list(picks), False

    if result.action == ACTION_ENTER_MANUAL:
        return _manual_entry_loop(provider.name), True

    return list(result.selected), False


def _collect_default_models(
    available: "list[str]", providers_cfg: dict, timeout: int
) -> "tuple[list[str], set]":
    """Run the per-provider picker across all available providers.

    Returns ``(ordered specs, manual_specs)`` where ``manual_specs`` is the subset
    of specs that came from a free-text "Enter manually…" entry — always flagged
    "(unverified id)" at confirm time regardless of base-catalog membership.
    """
    specs: list[str] = []
    seen = set()
    manual_specs: set = set()
    for name in available:
        provider = get_provider(name)
        curated = providers_cfg[name].get("models", []) or []
        bare_ids, manual = _pick_provider_models(provider, curated, timeout)
        for bare in bare_ids:
            spec = f"{name}:{bare}"
            if spec not in seen:
                specs.append(spec)
                seen.add(spec)
            if manual:
                manual_specs.add(spec)
    return specs, manual_specs


def _collect_default_models_guarded(
    available: "list[str]", providers_cfg: dict, timeout: int
) -> "tuple[list[str], set] | None":
    """Collect default models with the zero-selection guard (§3 step 3).

    Returns ``(specs, manual_specs)``, or None to signal an abort (caller exits 1)
    when the user selects nothing even after a retry. Never returns empty specs.
    """
    for attempt in range(2):
        specs, manual_specs = _collect_default_models(available, providers_cfg, timeout)
        if specs:
            return specs, manual_specs
        if attempt == 0:
            print(
                "\nYou didn't select any models. An empty default panel can't be "
                "written (it would brick every run). Let's try again — pick at least "
                "one model from any provider.\n",
                file=sys.stderr,
            )
    return None


def _choose_quick_models(default_specs: "list[str]") -> "list[str]":
    """Propose + confirm the ``--quick`` panel (a ≤2 positional subset of defaults).

    Never returns an empty list silently: if the user trims everything, they must
    explicitly confirm disabling ``--quick`` (writing ``quick_models: []``);
    otherwise the proposal is re-offered.
    """
    proposed = default_specs[:2]
    while True:
        print("\nProposed --quick panel (lightweight subset of your defaults):")
        for spec in proposed:
            print(f"  - {spec}")
        if prompt_yes_no("Use these as your --quick panel?", default=True):
            quick = list(proposed)
        else:
            quick = list(
                select_multi(default_specs, "Select --quick models (subset of your defaults):")
            )

        if quick:
            return quick

        print(
            "An empty --quick panel makes `--quick` error at runtime.",
            file=sys.stderr,
        )
        if prompt_yes_no(
            "Disable --quick entirely for this repo (write quick_models: [])?",
            default=False,
        ):
            return []
        # else: re-offer the proposal


def _choose_default_provider(available: "list[str]", default_specs: "list[str]") -> str:
    """Single-select ``default_provider``, defaulting to the first pick's provider."""
    first_provider = default_specs[0].split(":", 1)[0]
    default = first_provider if first_provider in available else available[0]
    if len(available) == 1:
        return available[0]
    chosen = select_one(available, "Choose the default provider:", default=default)
    return chosen if chosen in available else default


def _confirm_write(
    default_provider: str,
    models: "list[str]",
    quick_models: "list[str]",
    base_catalog: set,
    shadowed_modes: "list[str]",
    manual_specs: "set | None" = None,
) -> bool:
    """Show the pre-write summary (§5a), flag unverified ids, gate the write."""
    manual_specs = manual_specs or set()

    def _is_unverified(spec: str) -> bool:
        # A manually-entered id is always "unverified" per §5a, even when it
        # happens to exist in the base catalog (the user typed it free-form).
        return spec in manual_specs or spec not in base_catalog

    def _annotate(spec: str) -> str:
        return f"{spec}          (unverified id)" if _is_unverified(spec) else spec

    print("\nAbout to write .multi-llm/providers.yaml:\n")
    print(f"  default_provider: {default_provider}")
    print("  defaults.models:")
    for spec in models:
        print(f"    - {_annotate(spec)}")
    print("  defaults.quick_models:")
    if quick_models:
        for spec in quick_models:
            print(f"    - {_annotate(spec)}")
    else:
        print("    [] (--quick disabled for this repo)")

    if shadowed_modes:
        print(
            "\nNote: a mode-specific model list already exists for: "
            + ", ".join(shadowed_modes)
            + ".\n      Those per-mode lists still WIN over the globals above for "
            "those modes."
        )

    if any(_is_unverified(spec) for spec in (models + quick_models)):
        print("\n  (unverified id) = manual/Show-all pick not in the base catalog "
              "(check for typos; it will still run).")

    return prompt_yes_no("\nWrite this config?", default=False)


# ---------------------------------------------------------------------------
# Rendering + writing
# ---------------------------------------------------------------------------


def render_managed_block(
    default_provider: str, models: "list[str]", quick_models: "list[str]"
) -> str:
    """Render the selection-key payload as a YAML block (no marker lines).

    Uses ``yaml.safe_dump`` so every id (including ``/``- or ``:``-bearing specs)
    is correctly quoted; ``sort_keys=False`` preserves
    default_provider → models → quick_models order. ``quick_models`` is always
    emitted (an explicit ``[]`` means "--quick disabled").
    """
    payload = {
        "default_provider": default_provider,
        "defaults": {
            "models": list(models),
            "quick_models": list(quick_models),
        },
    }
    return yaml.safe_dump(payload, default_flow_style=False, sort_keys=False).rstrip("\n")


def splice_managed_block(template_text: str, managed_body: str) -> str:
    """Replace the delimited region of ``template_text`` with ``managed_body``.

    Header comments (and the MODE SHADOWING note) outside the markers are kept
    verbatim. Raises ValueError if the markers are missing/misordered so a corrupt
    template fails loudly rather than silently producing a stub.
    """
    start = template_text.find(MARKER_START)
    end = template_text.find(MARKER_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError("init-managed markers missing or misordered in template")
    start_line_end = template_text.find("\n", start) + 1
    before = template_text[:start_line_end]
    after = template_text[end:]
    notice = (
        "# Written by `multi-llm --init`. This block is REGENERATED on every\n"
        "# `--init --force`; put any hand-maintained keys OUTSIDE the markers.\n"
        "#\n"
        "# These are GLOBAL selection defaults; a `defaults.modes.<mode>` entry (if\n"
        "# any) still wins for that mode. See the MODE SHADOWING note above.\n"
    )
    return before + notice + managed_body + "\n" + after


def _parse_check(text: str) -> "str | None":
    """Pre-write gate: the generated text must parse to a selection-keys mapping.

    Returns an error string (refuse to write) or None on success.
    """
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return f"generated YAML does not parse: {e}"
    if not isinstance(parsed, dict) or "defaults" not in parsed:
        return "generated YAML is missing the expected selection keys"
    return None


def _recheck_merged(target_dir: Path, expected_models: "list[str]", config_path: Path) -> None:
    """Post-write best-effort reload: confirm the chosen specs actually resolve.

    Reloads the now-written file through ``load_config`` against a **fresh cache**
    (defeating the per-process memoization that would otherwise serve a stale
    pre-write config). When ``target_dir`` is a git root the file is discovered and
    its merged ``defaults.models`` should equal the chosen specs; a mismatch means
    a higher-precedence ``MULTI_LLM_PROVIDERS_CONFIG`` env override is shadowing it,
    which is worth a warning (the file is still valid and written). Never raises.

    Config discovery anchors on the **git root**, not ``target_dir``. When the
    written file is NOT the file discovery would pick up (the ``--dir <subdir>``
    case, or the outside-a-repo CWD fallback), the merged ``defaults.models`` falls
    back to the base catalog and would never equal the chosen specs — a mismatch
    that has nothing to do with an env override. So the shadowing warning only
    fires when the discovered project-config path is exactly the file we wrote.
    """
    try:
        import utils.provider_registry as registry

        registry._config = None
        registry._config_key = None
        merged = registry.load_config(anchor=str(target_dir))
        discovered = registry._find_project_config(anchor=str(target_dir))
        registry._config = None
        registry._config_key = None
        # Only diagnose shadowing when the written file is the one discovery picks
        # up; otherwise 'resolved' is the base catalog, not our file, and the
        # mismatch would misattribute an env override that may not exist.
        if discovered is None or discovered.resolve() != config_path.resolve():
            return
        resolved = list((merged.get("defaults", {}) or {}).get("models", []) or [])
        # Compare unconditionally: an env override that resolves defaults.models to
        # [] still shadows the chosen specs, so the empty-list case must warn too
        # (a truthiness guard would silently accept it).
        if resolved != list(expected_models):
            print(
                "WARNING: a higher-precedence override (e.g. MULTI_LLM_PROVIDERS_CONFIG) "
                "shadows the written defaults.models at runtime:\n"
                f"         resolved={resolved}\n"
                f"         written={expected_models}",
                file=sys.stderr,
            )
    except Exception as e:  # never let the recheck crash a completed init
        print(f"NOTE: skipped post-write config recheck ({e}).", file=sys.stderr)


def _splice_source(config_path: Path) -> str:
    """Pick the text to splice the managed block into.

    The template contract (see templates/config/providers.override.yaml) promises
    that everything OUTSIDE the markers — hand-maintained keys, comments — survives a
    re-init. To honor that, a re-init over an existing config splices into the
    EXISTING file when it already carries both markers, preserving its
    outside-the-region content verbatim. The bundled template is used only for a
    first-time write (file absent) or a degraded/legacy file missing the markers
    (where there is no managed region to preserve and we reset to the stub).
    """
    if config_path.exists():
        try:
            existing = config_path.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing is not None and MARKER_START in existing and MARKER_END in existing:
            return existing
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def _write_interactive(
    args: argparse.Namespace,
    target_dir: Path,
    config_dir: Path,
    config_path: Path,
    default_provider: str,
    models: "list[str]",
    quick_models: "list[str]",
) -> int:
    """Render, validate, and write the interactive selection; report + gitignore."""
    # Re-init over an existing marked config preserves its outside-the-markers edits
    # (template contract); first-write / marker-less files use the bundled template.
    source_text = _splice_source(config_path)
    managed = render_managed_block(default_provider, models, quick_models)
    text = splice_managed_block(source_text, managed)

    # Pre-write gate: never write unparseable YAML.
    error = _parse_check(text)
    if error:
        print(f"ERROR: refusing to write an invalid config — {error}", file=sys.stderr)
        return 1

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    # Post-write: confirm the chosen specs resolve through load_config (fresh cache).
    _recheck_merged(target_dir, models, config_path)
    print(f"\nWrote {config_path}")
    print(f"  default_provider: {default_provider}")
    print(f"  defaults.models: {len(models)} model(s)")
    print(
        "  defaults.quick_models: "
        + (f"{len(quick_models)} model(s)" if quick_models else "[] (--quick disabled)")
    )
    print(
        "  (Validated: the file parses and its defaults.models resolve via "
        "load_config.)\n"
        "  These globals apply only to modes WITHOUT their own "
        "defaults.modes.<mode> list.\n"
        "  Smoke-test it, e.g.:  /multi-llm:multi-llm --review-plan <plan> --quick"
    )
    _report_gitignore(args, target_dir)
    return 0


def run_interactive(args: argparse.Namespace, target_dir: Path, config_dir: Path,
                    config_path: Path) -> int:
    """Drive the interactive init flow (§3). Returns a process exit code."""
    base = load_base_config()
    providers_cfg = base.get("providers", {}) or {}
    available = _available_providers(providers_cfg)

    if not available:
        print(
            "No supported provider CLIs were detected on this machine.\n"
            "multi-llm needs at least one of these installed and on PATH:\n"
            "  - cursor-agent      (https://cursor.com)\n"
            "  - claude            (Claude Code)\n"
            "  - codex, gemini, opencode, kilocode\n"
            "Install one and re-run `--init`. To scaffold a commented stub now "
            "instead, re-run with --template-only.",
            file=sys.stderr,
        )
        return 1

    base_catalog = _base_catalog_specs(providers_cfg)
    shadowed_modes = _shadowed_modes(base, target_dir)

    print(
        "Detected installed providers: " + ", ".join(available) + "\n"
        "Pick the models for your default panel (curated list shown first; choose "
        '"Show all…" to browse the full CLI catalog).'
    )

    while True:  # confirm/edit loop — no file is written until confirmed
        collected = _collect_default_models_guarded(available, providers_cfg, args.timeout)
        if collected is None:
            print(
                "Aborting: no default models were selected. Re-run `--init` and pick "
                "at least one model.",
                file=sys.stderr,
            )
            return 1
        models, manual_specs = collected
        quick_models = _choose_quick_models(models)
        default_provider = _choose_default_provider(available, models)

        if _confirm_write(
            default_provider, models, quick_models, base_catalog, shadowed_modes, manual_specs
        ):
            break
        if not prompt_yes_no(
            "Edit your selections and try again? (No = quit without writing)",
            default=True,
        ):
            print("No changes written.")
            return 0

    return _write_interactive(
        args, target_dir, config_dir, config_path, default_provider, models, quick_models
    )


def write_template_only(args: argparse.Namespace, target_dir: Path, config_dir: Path,
                        config_path: Path) -> int:
    """Copy the commented template stub verbatim (historical behavior)."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {config_path}")
    _report_gitignore(args, target_dir)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="init_config.py",
        description=(
            "Set up a per-project multi-llm config override at "
            "<dir>/.multi-llm/providers.yaml. On a TTY this is interactive "
            "(detect installed CLIs, pick default + quick model panels); off a TTY "
            "or with --template-only / --non-interactive it copies a commented stub."
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
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Skip the interactive picker; write the commented template stub verbatim.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt (CI/unattended). Implies --template-only; also implied off a TTY.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Seconds to wait for a provider's `models` listing command (default: 10).",
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

    # Interactive only on a real TTY and when not explicitly opted out.
    use_interactive = not (args.template_only or args.non_interactive) and is_tty()
    if use_interactive:
        return run_interactive(args, target_dir, config_dir, config_path)
    return write_template_only(args, target_dir, config_dir, config_path)


if __name__ == "__main__":
    raise SystemExit(main())
