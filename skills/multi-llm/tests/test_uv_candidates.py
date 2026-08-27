"""Every spelling of the uv candidate list agrees with the one that runs.

**Why this is its own module.** The list lives in nine places, and the guard that
existed covered one of them, one-directionally, with no ordering. It could not
sit in ``tests/test_unattended.py`` (whose corpus is rooted at the skill dir and
would never see ``packaging/``) nor in ``tests/test_instruction_portability.py``
(whose manifest carries placeholder *counts*, a different property).

**The two halves, and the second is the one that was missing.** A table of known
readers proves each one agrees; it proves nothing about a *tenth* reader added
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

import re
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
REPO_ROOT = SKILL_ROOT.parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import unattended  # noqa: E402


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

#: The nine readers, each with the *form* it carries the list in.
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
    "legacy/references/uv-check.md": "ls",
    "hosts/uv-missing.txt": "bracket",
}

#: Where each reader lives, relative to the repo root.
_ROOTS = {name: SKILL_ROOT for name in READERS}


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
        agree and says nothing about a tenth reader added elsewhere — so the
        whole tracked tree is grepped and an unlisted hit is a failure by name.
        """
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
                ["git", "-C", str(REPO_ROOT), "grep", "-l", "-F", *patterns, "--",
                 "skills/multi-llm", "packaging"],
                capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError:  # pragma: no cover - environment guard
            pytest.skip("git is not on PATH")
        if listing.returncode not in (0, 1):
            pytest.skip(f"git grep failed: {listing.stderr.strip()}")

        hits = set()
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
            ):
                continue
            try:
                hits.add(str(path.relative_to("skills/multi-llm")))
            except ValueError:
                hits.add(str(path))

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
