"""Every spelling of the uv candidate list agrees with the one that runs.

**Why this is its own module.** The list lives in every place the ``READERS``
table below names, and the guard that existed covered one of them,
one-directionally, with no ordering. It could not sit in
``tests/test_unattended.py`` (whose corpus is rooted at the skill dir and would
never see ``packaging/``) nor in ``tests/test_instruction_portability.py``
(whose manifest carries placeholder *counts*, a different property).

**The two halves, and the second is the one that was missing.** A table of known
readers proves each one agrees; it proves nothing about an *unlisted* reader added
somewhere else. So the discovery half greps the whole tracked tree and requires
the hit set to **equal** the table's path set. Without it the table is a list
somebody remembers to update; with it, a new reader turns this red on the pull
request that adds it.

**Why it matters more than it looks.** The two ``.exe`` entries are the entire
Windows story: the official installer creates them, Git Bash is the only
supported Windows shell, and a reader that drops them recreates precisely the
Windows failure the npm route exists to remove. ``detach.resolve_uv()`` shipped
exactly that defect — a two-entry probe with no ``.exe``, no Homebrew, no
``/usr/local``, no ``/opt/local`` — on the code path that ``detach: required``
puts *every* fan-out mode through, and it named that shortened list back to the
user in its own error text. It could therefore refuse on a machine where the
foreground preflight had just succeeded.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import unattended  # noqa: E402

from tests.generated_distributions import (  # noqa: E402
    CLAUDE,
    CODEX,
    DISTRIBUTIONS,
    PAYLOAD,
    authored_source,
)


def _shell_form(candidate: str) -> str:
    """A candidate as a POSIX shell command line spells it.

    ``~`` becomes ``"$HOME"`` **and gains quotes**, because a home directory may
    contain a space and an unquoted ``$HOME`` would split. An absolute candidate
    is spelled bare, exactly as the shipped ``ls`` lines do.
    """
    if candidate.startswith("~/"):
        return f'"$HOME/{candidate[2:]}"'
    return candidate


#: The single source, and the two derived spellings every reader uses.
CANDIDATES = unattended.UV_CANDIDATES
LS_LINE = " ".join(_shell_form(candidate) for candidate in CANDIDATES)

#: Every reader below, each with the *form* it carries the list in.
#:
#: ``ls`` — the literal ``ls <paths> 2>/dev/null`` probe an agent runs.
#: ``python`` — the tuple itself.
#: ``bracket`` — the ``~/.local/bin/uv[.exe]`` prose form, which collapses the
#: two ``.exe`` pairs into bracketed optionals.
#: ``generated`` — emitted into a build artifact from the Python owner.
READERS = {
    "utils/unattended.py": "python",
    "utils/detach.py": "python",
    "hosts/snippets/uv-preflight.md": "ls",
    "hosts/codex/shim.md.tmpl": "ls",
    "hosts/opencode/shim.md.tmpl": "ls",
    "SKILL.md": "ls",
    "references/uv-check.md": "ls",
    "hosts/uv-missing.txt": "bracket",
    # The Claude Code plugin's SessionStart hook (D6 step 5) resolves uv the
    # same way before running `doctor --warn-only`; it lives at the repo root
    # beside `.claude-plugin/`, outside the skill tree.
    "hooks/doctor.sh": "ls",
    # The plugin's Stop hook (plans/next-phase-command.md T11) resolves uv the
    # same way, after the grep that lets a session with no phase result skip
    # it; it sits beside doctor.sh, the second reader outside the skill tree.
    "hooks/report-check.sh": "ls",
    # The plugin's PreToolUse hook (fix round 26) resolves uv the same way,
    # after the greps that let any other tool call, and a session that never
    # ran the dispatcher, skip it.
    "hooks/wakeup-check.sh": "ls",
    # The codex plugin's generated surface (plans/codex-native-plugin.md, D3e),
    # committed because the Git route serves the checked-in tree: the list the
    # server's uv preflight reads, emitted from `unattended.UV_CANDIDATES`.
    "packaging/codex/multi-llm/dist/cli-surface.json": "generated",
}

#: Where each reader lives, relative to the repo root.
_ROOTS = {name: SKILL_ROOT for name in READERS}
_ROOTS["hooks/doctor.sh"] = REPO_ROOT
_ROOTS["hooks/report-check.sh"] = REPO_ROOT
_ROOTS["hooks/wakeup-check.sh"] = REPO_ROOT
_ROOTS["packaging/codex/multi-llm/dist/cli-surface.json"] = REPO_ROOT


def _files_naming_a_candidate(root: Path) -> list:
    """Repo-relative paths under *root* that name a uv candidate, non-readers aside."""
    # The patterns are derived from the list, never typed: a bare `bin/uv`
    # grep also finds the container image's own `/opt/mllm/bin/uv` wrapper,
    # which is a different thing entirely and would make this row a
    # permanent false positive with an ever-growing exclusion list.
    patterns = []
    for candidate in CANDIDATES:
        stem = candidate.removeprefix("~")
        if stem.endswith(".exe"):
            continue  # covered by its non-.exe sibling
        patterns += ["-e", stem]
    try:
        listing = subprocess.run(
            # `--untracked`: a reader is caught on the working tree that
            # adds it, before it is staged, and ignored files stay out.
            ["git", "-C", str(root), "grep", "-l", "-F", "--untracked",
             *patterns, "--", "skills/multi-llm", "packaging", "hooks"],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:  # pragma: no cover - environment guard
        pytest.skip("git is not on PATH")
    if listing.returncode not in (0, 1):
        pytest.skip(f"git grep failed: {listing.stderr.strip()}")

    found = []
    for line in listing.stdout.splitlines():
        if not line.strip():
            continue
        path = Path(line)
        # Tracked but not runtime, and deliberately outside the contract:
        # this module and the package suite both *quote* the list in order
        # to assert about it.
        if path.name in (
            "test_uv_candidates.py", "test_opencode_package.py",
            "test_unattended.py", "test_detach.py", "test_shim_correctness.py",
            "test_install_host.py", "test_instruction_portability.py",
            # Quotes a recorded dispatcher argv in a fixture.
            "test_opencode_registry_check.py",
        ):
            continue
        # The step-5 registry-install probe's stock-container recipe
        # INSTALLS uv at /usr/local/bin, the way a first-time user's system
        # package would; it never searches the list, so it is no reader.
        # The plugin it then exercises is, and is in READERS.
        if path.as_posix() == "packaging/opencode/registry_check.py":
            continue
        # A committed run record QUOTES what ran on a user's machine (the
        # codex-v0.1.0 network check's dispatcher argv was
        # `/home/checker/.local/bin/uv`); it is evidence, never a reader of
        # the list, and editing it to satisfy this scan would falsify it.
        if path.parent.as_posix().endswith("tests/container/records"):
            continue
        found.append(path.as_posix())
    return found


def _reader_name(path: str) -> str:
    """The ``READERS`` key a discovered file is graded as.

    A generated copy is the *same* reader as its source, so it answers to that
    source's row rather than being listed once per tree. That holds only because
    the bytes match: the build-parity rows compare each tree to a fresh build, and
    :class:`TestGeneratedCopies` compares every graded copy here.
    A file with no authored source keeps its own path, so it has to have a row of
    its own or the scan fails on it by name.
    """
    graded = Path(authored_source(path) or path)
    try:
        return graded.relative_to(PAYLOAD).as_posix()
    except ValueError:
        return graded.as_posix()


class TestTheListItself:
    def test_the_two_exe_entries_are_present(self):
        """They are the entire Windows story, so they are asserted by name."""
        assert "~/.local/bin/uv.exe" in CANDIDATES
        assert "~/.cargo/bin/uv.exe" in CANDIDATES

    def test_every_entry_is_absolute_or_home_anchored(self):
        for candidate in CANDIDATES:
            assert candidate.startswith(("~/", "/")), candidate

    def test_the_order_is_the_probe_order(self):
        """Home-relative first, then the system locations.

        Order is part of the contract, not an accident: a user's own install
        must win over a system one, or a stale ``/usr/local/bin/uv`` shadows the
        uv they just installed.
        """
        home = [c for c in CANDIDATES if c.startswith("~/")]
        system = [c for c in CANDIDATES if not c.startswith("~/")]
        assert list(CANDIDATES) == home + system


class TestEveryReaderAgrees:
    @pytest.mark.parametrize("relative", sorted(READERS))
    def test_every_uv_candidate_spelling_agrees_with_the_source(self, relative):
        """Element for element, in order — not "each entry appears somewhere".

        The guard this replaces asserted only that each entry appeared
        *somewhere* in one file, and asserted no order at all. A reader that
        listed six of seven in a scrambled order passed it.
        """
        path = _ROOTS[relative] / relative
        assert path.is_file(), f"{relative} has moved; fix the table, not the test"
        text = path.read_text(encoding="utf-8")
        form = READERS[relative]

        if form == "python":
            # The tuple is the source in one file and is *consumed* in the
            # other. Either way, no second literal list may exist beside it.
            listed = re.findall(r'"(~/[^"]*bin/uv(?:\.exe)?)"', text)
            if relative == "utils/unattended.py":
                assert listed == [c for c in CANDIDATES if c.startswith("~/")]
            else:
                assert not listed, (
                    f"{relative} carries its own literal candidate paths "
                    f"({listed}); it must consume unattended.UV_CANDIDATES"
                )
            return

        if form == "ls":
            assert LS_LINE in text, (
                f"{relative}'s `ls` probe does not match the source list.\n"
                f"expected: {LS_LINE}\n"
                "Every candidate, in probe order, `$HOME`-spelled and quoted."
            )
            return

        if form == "generated":
            import json

            assert json.loads(text)["uv"]["candidates"] == list(CANDIDATES), (
                f"{relative} was emitted from a different list; rebuild it "
                "(`make codex-regen`)"
            )
            return

        # bracket: the prose form, where the two `.exe` pairs collapse.
        for candidate in CANDIDATES:
            bracketed = candidate.replace(".exe", "[.exe]")
            base = bracketed.replace("[.exe]", "")
            assert bracketed in text or f"{base}[.exe]" in text, (
                f"{relative} does not mention {candidate}"
            )

    def test_the_error_text_of_the_detached_probe_names_the_full_list(self):
        """``resolve_uv``'s refusal names every place it looked.

        It used to name a two-entry list — in code *and* again in its message —
        so a user was told uv was missing from two locations while the
        foreground preflight had checked seven.
        """
        from utils import detach

        with pytest.MonkeyPatch.context() as patch:
            # The refusal has to be forced by the *probe*, not by the
            # environment and not by the list. Redirecting `MULTI_LLM_UV`,
            # `PATH` and `HOME` leaves `/opt/homebrew/bin/uv`,
            # `/usr/local/bin/uv` and `/opt/local/bin/uv` — which no env var can
            # move — so on a machine carrying uv in one of them (common on macOS
            # and in CI images) the probe succeeds and there is no refusal to
            # read. Emptying `UV_CANDIDATES` is not the fix either: the message
            # is rendered from that same tuple, so it would empty the very text
            # under assertion. Stubbing the probe forces the refusal on every
            # host and leaves the list intact — and because `resolve_uv` looks
            # `find_uv` up on the module at call time, a delegation that stopped
            # delegating still fails here.
            patch.setattr(unattended, "find_uv", lambda: None)
            with pytest.raises(detach.DetachError) as excinfo:
                detach.resolve_uv()
        message = str(excinfo.value)
        for candidate in CANDIDATES:
            assert candidate in message, (
                f"the detached probe's refusal does not name {candidate}"
            )

    def test_the_two_probes_resolve_identically(self):
        """``detach.resolve_uv()`` and ``unattended.find_uv()`` are one probe.

        Not "similar": ``resolve_uv`` delegates, so a machine on which one
        succeeds and the other refuses is not reachable.
        """
        from utils import detach

        found = unattended.find_uv()
        if found is None:
            with pytest.raises(detach.DetachError):
                detach.resolve_uv()
        else:
            assert detach.resolve_uv() == found


class TestNoUnlistedReader:
    def test_no_unlisted_file_names_a_uv_candidate(self):
        """The discovery half: the hit set must **equal** the table's path set.

        This is what today's guard most lacked. A table proves its own rows
        agree and says nothing about an unlisted reader added elsewhere — so the
        whole tracked tree is grepped and an unlisted hit is a failure by name.
        """
        hits = {_reader_name(path) for path in _files_naming_a_candidate(REPO_ROOT)}

        expected = set(READERS)
        unlisted = hits - expected
        assert not unlisted, (
            f"{len(unlisted)} file(s) name a uv candidate path but are not in "
            f"READERS: {sorted(unlisted)}\n"
            "Add each to the table with the form it carries the list in — or, "
            "better, make it consume unattended.UV_CANDIDATES."
        )
        vanished = expected - hits
        assert not vanished, (
            f"READERS names {sorted(vanished)}, which no longer mention a uv "
            "candidate path. A row that outlives its reader is how a table stops "
            "being a guard."
        )

    @pytest.mark.parametrize("planted, graded_as", [
        # A new script beside the hooks is not one of the three known copies.
        (f"{CLAUDE.root}hooks/new-probe.sh", f"{CLAUDE.root}hooks/new-probe.sh"),
        # Outside everything the Claude plugin declares.
        (f"{CLAUDE.root}bin/probe.sh", f"{CLAUDE.root}bin/probe.sh"),
        # Under the payload it answers to its authored name, which has no row.
        (f"{CLAUDE.root}{PAYLOAD}utils/new_reader.py", "utils/new_reader.py"),
        (f"{CODEX.root}{PAYLOAD}utils/new_reader.py", "utils/new_reader.py"),
        # Codex's skill is rendered, so a ladder there is new text in that one
        # file. It must not pass as the authored SKILL.md, which is a known reader.
        (CODEX.manifest, CODEX.manifest),
        # Codex ships no hooks, so this copies nothing: the authored file of the
        # same name is a reader, this would not be.
        (f"{CODEX.root}hooks/doctor.sh", f"{CODEX.root}hooks/doctor.sh"),
        # A tree the shared list does not name is not excused by its shape.
        ("packaging/other/multi-llm/hooks/doctor.sh",
         "packaging/other/multi-llm/hooks/doctor.sh"),
    ])
    def test_an_unknown_reader_under_a_generated_root_still_fails(
        self, tmp_path, planted, graded_as,
    ):
        """Discovery stays a failure for anything the mappings do not account for.

        Run through the same scan as the row above, in a scratch repository, because
        planting a real-shaped reader in the working tree is the one thing a guard
        over the working tree must not do.
        """
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        target = tmp_path / planted
        target.parent.mkdir(parents=True)
        target.write_text(f"ls {LS_LINE} 2>/dev/null\n", encoding="utf-8")

        hits = {_reader_name(path) for path in _files_naming_a_candidate(tmp_path)}
        assert hits - set(READERS) == {graded_as}, hits

    @pytest.mark.parametrize("planted, graded_as", [
        (f"{CLAUDE.root}hooks/doctor.sh", "hooks/doctor.sh"),
        (f"{CLAUDE.root}hooks/report-check.sh", "hooks/report-check.sh"),
        (f"{CLAUDE.root}hooks/wakeup-check.sh", "hooks/wakeup-check.sh"),
        (CLAUDE.manifest, "SKILL.md"),
        (f"{CLAUDE.root}{PAYLOAD}references/uv-check.md", "references/uv-check.md"),
        (f"{CLAUDE.root}{PAYLOAD}utils/unattended.py", "utils/unattended.py"),
        (f"{CODEX.root}{PAYLOAD}references/uv-check.md", "references/uv-check.md"),
    ])
    def test_a_known_copy_answers_to_its_authored_row(self, tmp_path, planted, graded_as):
        """The other half of the row above: the same scan accepts a real copy."""
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        target = tmp_path / planted
        target.parent.mkdir(parents=True)
        target.write_text(f"ls {LS_LINE} 2>/dev/null\n", encoding="utf-8")

        hits = {_reader_name(path) for path in _files_naming_a_candidate(tmp_path)}
        assert hits == {graded_as}
        assert graded_as in READERS


def _load_builder(host: str):
    """A host's ``build.py``, loaded under a name no other suite imports it by."""
    name = f"{host}_build_for_uv_candidates"
    spec = importlib.util.spec_from_file_location(
        name, str(REPO_ROOT / "packaging" / host / "build.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestGeneratedCopies:
    """What lets the scan above grade a copy as its source, checked directly."""

    def test_every_graded_copy_is_its_source_byte_for_byte(self):
        """A stale copy would otherwise pass: the reader rows read the authored file.

        ``READERS`` opens ``hooks/doctor.sh`` and never the copy under
        ``packaging/claude/``, so a tree left behind by an edit to the ladder ships
        the old list while every row above stays green.
        """
        stale = [
            f"{path} differs from {source}"
            for path in _files_naming_a_candidate(REPO_ROOT)
            for source in [authored_source(path)]
            if source is not None
            and (REPO_ROOT / path).read_bytes() != (REPO_ROOT / source).read_bytes()
        ]
        assert not stale, (
            f"{stale}: each is graded as its authored source, which is only sound "
            "while the bytes match. Run `make codex-regen claude-regen`."
        )

    def test_claude_carries_every_authored_reader(self):
        """Exactly the authored readers, so a reader dropped from the copy is seen.

        The discovery row cannot see that: the authored file still names the list,
        so the hit set is unchanged while the plugin stops shipping the file. The
        one row left out is Codex's rendered surface, which Claude has no copy of.
        """
        graded = {
            _reader_name(path)
            for path in _files_naming_a_candidate(REPO_ROOT)
            if path.startswith(CLAUDE.root)
        }
        assert graded == set(READERS) - {f"{CODEX.root}dist/cli-surface.json"}

    @pytest.mark.parametrize(
        "name", sorted(name for name in READERS if name.startswith("packaging/")),
    )
    def test_a_reader_with_its_own_generated_row_is_a_declared_render(self, name):
        """A generated file earns its own row only by being rendered, not copied.

        Otherwise a copy could be listed a second time here, and the table would
        go back to needing one row per tree.
        """
        assert authored_source(name) is None
        assert any(
            name.startswith(tree.root) and name[len(tree.root):] in tree.rendered
            for tree in DISTRIBUTIONS
        ), f"{name} is not declared in generated_distributions.py as rendered"

    def test_the_shared_list_matches_what_the_builders_ship(self, monkeypatch):
        """The list is typed out on purpose, so it is held to the two builders.

        A file a builder starts writing outside the payload would otherwise have no
        mapping here. It would then fail the scan under its own path if it named a
        candidate, with nothing telling the next reader that the list is the fix.
        """
        # The builders put their own directories on the path, for their own imports.
        monkeypatch.setattr(sys, "path", list(sys.path))
        claude = _load_builder("claude")
        assert CLAUDE.root == f"{claude.PLUGIN_RELATIVE}/"
        assert dict(CLAUDE.copied) == dict(claude.PLUGIN_FILES)
        assert CLAUDE.rendered == {}

        codex = _load_builder("codex")
        assert CODEX.root == f"{codex.PLUGIN_RELATIVE}/"
        outside_payload = {
            name for name in (*CODEX.copied, *CODEX.rendered)
            if not name.startswith(PAYLOAD)
        }
        assert outside_payload == set(codex.PLUGIN_FILES)
        assert CODEX.rendered[f"{PAYLOAD}SKILL.md"] == f"{PAYLOAD}{codex.SKILL_TEMPLATE}"

    @pytest.mark.parametrize("tree", DISTRIBUTIONS, ids=lambda tree: tree.root)
    def test_every_mapped_file_exists(self, tree):
        """A mapping that outlives its file is how a list stops being a guard."""
        for relative in (*tree.copied, *tree.rendered, tree.version_manifest):
            assert (REPO_ROOT / tree.root / relative).is_file(), (
                f"{tree.root}{relative} is mapped but missing; run `{tree.regen}`"
            )
        for source in (*tree.copied.values(), *tree.rendered.values()):
            assert (REPO_ROOT / source).is_file(), source
        assert (REPO_ROOT / tree.manifest).is_file()

    @pytest.mark.parametrize("tree", DISTRIBUTIONS, ids=lambda tree: tree.root)
    def test_every_mapped_copy_is_its_source_byte_for_byte(self, tree):
        """The non-payload copies, which the candidate scan alone would not reach."""
        for relative, source in tree.copied.items():
            assert (REPO_ROOT / tree.root / relative).read_bytes() == (
                REPO_ROOT / source
            ).read_bytes(), f"{tree.root}{relative} differs; run `{tree.regen}`"

    @pytest.mark.parametrize("path, source", [
        (f"{CLAUDE.root}hooks/hooks.json", "hooks/hooks.json"),
        (f"{CLAUDE.root}.claude-plugin/plugin.json", ".claude-plugin/plugin.json"),
        (CLAUDE.manifest, f"{PAYLOAD}SKILL.md"),
        (f"{CODEX.root}mcp/server.py", "packaging/codex/mcp_server.py"),
        (CODEX.manifest, None),
        (f"{CODEX.root}dist/cli-surface.json", None),
        (f"{CLAUDE.root}unknown.txt", None),
        (f"{PAYLOAD}SKILL.md", None),
        # The trailing slash on each root is what keeps a sibling directory out.
        ("packaging/claude/multi-llm-old/hooks/doctor.sh", None),
    ])
    def test_authored_source(self, path, source):
        assert authored_source(path) == source
