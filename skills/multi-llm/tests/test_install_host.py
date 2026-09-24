#!/usr/bin/env python3
"""The install lifecycle (D7a-D7c) — hermetic, no host CLI, POSIX *and* Windows.

Every test here runs against a **fixture skill dir containing a space** with
``install_paths`` redirected to a temp ``$HOME``/``%USERPROFILE%`` and a temp
project root, so no test on any platform writes into a real config dir. The one
thing this suite structurally cannot prove is that the *host* discovers what the
installer wrote — only a live host can show that, so it is the live tier's row and
nothing here stands in for it.

What is asserted:

* a rendered **copy** (never a link) at the preferred path of each scope, per
  shipped host, with the baked prefix absolute, quoted, resolving to the fixture
  dir, and no placeholder or unfilled slot surviving;
* every installed shim still passing the shipped-shim CI guard
  (``tests/test_shim_correctness.py``), run over the *installed file* — so the
  installer's notion of a valid shim cannot drift from CI's;
* ``--check``'s five assertions, each failing on its own, including the uv-absent
  path with a stubbed probe — and both halves of row 4, the ``shim_version``
  counter and the ``template_sha256`` the header has always recorded, in that
  order of precedence;
* every shipped ``hosts/*/shim.md.tmpl`` pinned to its digest, so a template edit
  arrives as a failing test naming the file rather than as a fleet of installed
  shims that quietly stopped matching it;
* idempotency (a second run is byte-identical and writes nothing), local-edit
  refusal and ``--force``'s ``.local.bak``, rollback (previous bytes restored, or
  no file left behind), a foreign host id refused, precedence reporting;
* ``uninstall-host`` — ``--dry-run`` changes nothing, only files carrying the
  ``multi_llm_shim`` block are removed, compatibility locations and edited shims
  are reported and kept;
* ``install_paths`` pairwise disjointness across installable hosts, so a shim
  installed for one host can neither collide with nor shadow another's;
* the Windows seam (``%USERPROFILE%``/XDG token expansion, the Git Bash
  prerequisite, no symlink API anywhere in the installer) — **asserted**, not
  skipped, on POSIX too, because a skipped seam is an unverified one;
* the **generated host-install page and the front page that points at it**: the
  block is byte-identical to a fresh render, a missing or marker-less
  ``docs/hosts.md`` is a ``--check`` failure rather than a skipped step, the
  opening sentence is the one composed from ``_installable_prose()`` at every
  arity, and every host fact the generated page states — host ids, one
  ``install-host <slug>`` line each, the printed shim paths, the reload
  instruction, the baked-in ``--yes``, the runner slugs and the headless
  separator — is compared against ``hosts/capabilities.yaml`` and, for the
  paths, against a real ``install-host --project`` run. The README names the
  installable hosts and links here; it does not keep a second copy of those
  recipes.
"""

import ast
import copy
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(SKILL_ROOT / "hosts"))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
import render_capabilities as renderer  # noqa: E402
from utils import host_capabilities as caps  # noqa: E402
from utils import host_install as hi  # noqa: E402

#: The repository (and published-plugin) root: the directory holding `README.md`
#: and `docs/`. Anchored on the skill root, never derived from the depth of
#: `renderer.HOST_DOC` — see :func:`readme_path`.
REPO_ROOT = SKILL_ROOT.parent.parent

#: The fixture skill dir name: a space in every segment that can carry one, so a
#: quoting bug cannot hide behind a clean path (D1b/D7c — spaces are the
#: assumption, not the exception).
FIXTURE_DIRNAME = "multi llm fixture/skill dir"

#: What a fixture skill dir needs for rendering and checking: the capability data,
#: the templates, the snippets, the uv message, and the reference host's shim.
FIXTURE_CONTENT = ("hosts", "SKILL.md", "references", "schemas")

INSTALLABLE = sorted(
    host_id for host_id in caps.host_ids()
    if caps.entry(host_id)["installable"]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skill(tmp_path):
    """A skill dir with a space in its path, carrying only what install needs."""
    root = tmp_path / FIXTURE_DIRNAME
    root.mkdir(parents=True)
    for name in FIXTURE_CONTENT:
        source = SKILL_ROOT / name
        if source.is_dir():
            shutil.copytree(source, root / name)
        else:
            shutil.copy2(source, root / name)
    caps.clear_cache()
    yield root
    caps.clear_cache()


@pytest.fixture
def home(tmp_path):
    """A temp per-user root, reachable through either platform's variable."""
    path = tmp_path / "home dir"
    path.mkdir()
    return path


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "project root"
    path.mkdir()
    return path


@pytest.fixture
def env(home):
    """The environment mapping the path expansion is driven from.

    Both ``HOME`` and ``USERPROFILE`` are set to the same temp directory, which is
    what a Windows machine running Git Bash looks like — so the same fixture
    exercises the POSIX and the Windows expansion path.
    """
    return {"HOME": str(home), "USERPROFILE": str(home)}


@pytest.fixture
def ready(monkeypatch):
    """Row 5's prerequisites, stubbed: uv resolves, and the runner is installed,
    authenticated and current.

    None of the three is a property of the machine running the tests — CI has no
    codex or opencode binary, and a developer's may have one that is logged out —
    and row 5 has its own dedicated tests below, so every other test stubs them
    rather than asserting the state of the test runner. Stubbing the probe also
    keeps the suite hermetic: the real one shells out to `<cli> --version` and the
    adapter's `auth_check`.
    """
    monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(hi, "runner_available", lambda provider: True)
    monkeypatch.setattr(
        hi, "probe_runner",
        lambda spec, **kwargs: {"auth": "ok", "version": "9.9.9", "problem": ""},
    )


def install(host_id, skill, env, project, **kwargs):
    return hi.install(
        host_id, base=str(skill), env=env, project_root=project, **kwargs
    )


def check(host_id, skill, env, project, **kwargs):
    return hi.check(
        host_id, base=str(skill), env=env, project_root=project, **kwargs
    )


def target(host_id, skill, env, project, scope=hi.DEFAULT_SCOPE):
    return hi.target_location(
        host_id, scope, base=str(skill), env=env, project_root=project
    ).path


# ---------------------------------------------------------------------------
# 1. The rendered copy, per shipped host and scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host_id", INSTALLABLE)
@pytest.mark.parametrize("scope", hi.WRITABLE_SCOPES)
class TestRenderedCopy:
    def test_a_copy_is_written_to_the_preferred_path_of_the_scope(
        self, host_id, scope, skill, env, project, ready, capsys
    ):
        assert install(host_id, skill, env, project, scope=scope) == 0
        path = target(host_id, skill, env, project, scope)
        assert path.is_file()
        assert not path.is_symlink(), (
            "the installed artifact is a rendered copy: a link would install a "
            "template no host can run, and would drag the Windows "
            "symlink-privilege question in with it"
        )
        assert "[SHIM_INSTALLED]" in capsys.readouterr().out

    def test_the_baked_prefix_is_the_fixture_skill_dir_quoted(
        self, host_id, scope, skill, env, project, ready
    ):
        install(host_id, skill, env, project, scope=scope)
        text = target(host_id, skill, env, project, scope).read_text(
            encoding="utf-8"
        )
        prefix = hi.baked_prefix(text)
        assert prefix == str(skill)
        assert Path(prefix).is_absolute()
        assert f'--project "{skill}"' in text, (
            "every baked prefix is emitted quoted: the fixture path contains "
            "spaces, which is the ordinary case on Windows"
        )

    def test_nothing_is_left_unresolved(
        self, host_id, scope, skill, env, project, ready
    ):
        install(host_id, skill, env, project, scope=scope)
        text = target(host_id, skill, env, project, scope).read_text(
            encoding="utf-8"
        )
        for token in hi.FORBIDDEN_IN_INSTALLED:
            assert token not in text, token
        assert not hi._SLOT_RE.findall(text)
        for slot in caps.VERB_SLOTS.values():
            assert slot not in text

    def test_the_three_verb_sentences_come_from_the_data(
        self, host_id, scope, skill, env, project, ready
    ):
        install(host_id, skill, env, project, scope=scope)
        text = target(host_id, skill, env, project, scope).read_text(
            encoding="utf-8"
        )
        for sentence in caps.verb_sentences(host_id, str(skill)):
            assert text.count(sentence) == 1


class TestSlotInterpolation:
    """Every slot the renderer knows, on a template that carries all of them.

    The two shipped shims deliberately carry neither shared snippet: a 100-line
    rendered budget cannot hold `uv-preflight.md`'s inlined 31-line install message
    *and* the poll loop, so both name `hosts/uv-missing.txt` and print it exactly
    as the reference shim does. The interpolation itself is still part of the
    contract, so it is exercised here on a fixture template rather than left
    untested until a host uses it.
    """

    HOST = "codex"

    def test_a_snippet_slot_is_inlined_verbatim(self, skill):
        rendered = hi.render(
            self.HOST, base=str(skill),
            template="{{poll_loop}}\n--host codex\n",
        )
        snippet = (skill / "hosts" / "snippets" / "poll-loop.md").read_text(
            encoding="utf-8"
        )
        head = snippet.splitlines()[0]
        assert head in rendered
        assert "## Polling a detached run" in rendered
        assert "{{poll_loop}}" not in rendered

    def test_a_snippets_own_slots_are_filled_too(self, skill):
        rendered = hi.render(
            self.HOST, base=str(skill), template="{{uv_preflight}}\n",
        )
        message = (skill / "hosts" / "uv-missing.txt").read_text(encoding="utf-8")
        assert message.rstrip("\n") in rendered, (
            "a partial may itself carry the uv-message slot, so snippets expand "
            "before the install-time slots do"
        )
        assert not hi._SLOT_RE.findall(rendered)

    def test_a_snippet_slot_carries_the_baked_prefix(self, skill):
        rendered = hi.render(
            self.HOST, base=str(skill), skill="/other skill dir",
            template="{{poll_loop}}\n",
        )
        assert '--project "/other skill dir"' in rendered
        assert "{SKILL_DIR}" not in rendered

    def test_the_message_file_slot_is_the_baked_absolute_path(self, skill):
        rendered = hi.render(
            self.HOST, base=str(skill), template='cat "{{uv_missing_file}}"\n'
        )
        assert rendered == f'cat "{skill / "hosts" / "uv-missing.txt"}"\n'

    @pytest.mark.parametrize("host_id", INSTALLABLE)
    def test_every_shipped_shim_prints_the_single_sourced_message(
        self, host_id, skill, env, project, ready
    ):
        install(host_id, skill, env, project)
        text = target(host_id, skill, env, project).read_text(encoding="utf-8")
        assert str(skill / "hosts" / "uv-missing.txt") in text
        assert "hosts/uv-missing.txt" in (
            SKILL_ROOT / caps.entry(host_id)["shim"]["template"]
        ).read_text(encoding="utf-8"), (
            "the wording lives in exactly one place and a shim may name it or "
            "interpolate it, never paraphrase it"
        )


@pytest.mark.parametrize("host_id", INSTALLABLE)
class TestInstalledShimPassesTheShippedGuard:
    """The installed file is held to the same assertions CI holds the template to.

    Imported and driven directly rather than re-listed: an installer whose output
    would fail the shipped-shim guard is an installer producing a shim CI would
    reject, and the two must never be able to disagree.
    """

    def test_every_guard_assertion_passes_on_the_installed_file(
        self, host_id, skill, env, project, ready, monkeypatch
    ):
        from tests import test_shim_correctness as guard

        install(host_id, skill, env, project)
        path = target(host_id, skill, env, project)
        # The guard renders against a fixture skill dir containing a space; here
        # the installed file already carries one, so point the guard at it. The
        # property is unchanged — an absolute prefix that survives a space.
        monkeypatch.setattr(guard, "FIXTURE_SKILL_DIR", str(skill))
        instance = guard.TestShim()
        methods = [
            name for name in dir(instance)
            if name.startswith("test_")
        ]
        assert len(methods) >= 12, methods
        for name in methods:
            getattr(instance, name)(host_id, path)

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda t: t.replace("command -v uv", "true"),
            lambda t: t + "\n{{ask_user}}\n",
            lambda t: t.replace("**Spawn subagents**", "**Spawn helpers**"),
            lambda t: t.replace("--host ", "--host not-a-"),
        ],
    )
    def test_what_the_installer_rejects_the_guard_rejects_too(
        self, host_id, skill, env, project, ready, monkeypatch, mutation
    ):
        """Two-way agreement: neither side may accept what the other refuses.

        The positive direction is the test above (an installed shim passes every
        guard assertion). This is the negative one, and together they are what
        "the installer's notion of a valid shim cannot drift from CI's" means.

        One asymmetry is deliberate and therefore not listed: a surviving
        ``{SKILL_DIR}`` token. The guard *renders* what it reads (a template
        legitimately carries the token), while the installer checks the rendered
        output — where the same token can only be a leftover.
        """
        from tests import test_shim_correctness as guard

        install(host_id, skill, env, project)
        path = target(host_id, skill, env, project)
        path.write_text(mutation(path.read_text(encoding="utf-8")),
                        encoding="utf-8")
        assert hi.shape_problems(
            host_id, path.read_text(encoding="utf-8"), str(skill)
        ), "the installer accepted a mutation it should have rejected"

        monkeypatch.setattr(guard, "FIXTURE_SKILL_DIR", str(skill))
        instance = guard.TestShim()
        failures = []
        for name in [n for n in dir(instance) if n.startswith("test_")]:
            try:
                getattr(instance, name)(host_id, path)
            except AssertionError:
                failures.append(name)
        assert failures, "the shipped-shim guard accepted what the installer refused"

    def test_the_installed_shim_fits_the_line_budget(
        self, host_id, skill, env, project, ready
    ):
        install(host_id, skill, env, project)
        path = target(host_id, skill, env, project)
        entry = caps.entry(host_id, str(skill))
        cap = int(entry.get("shim_max_lines") or 100)
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= cap, (
            f"{path} renders {len(text.splitlines())} lines against a cap of {cap}"
        )


# ---------------------------------------------------------------------------
# 1b. The template bodies are pinned, so a change to one is never silent
# ---------------------------------------------------------------------------

#: The ``SHIM_VERSION`` the digests below were recorded against, and each shipped
#: template's sha256 at that moment. Pinned rather than computed, for the same
#: reason `references/host-capabilities.md` is byte-compared against a fresh
#: render: a guard that derives its expectation from the file it is guarding
#: asserts nothing. A template edit therefore lands as a failing test naming the
#: file, which is where the author is asked whether the edit also moved the
#: rendered *shape* and so owes ``SHIM_VERSION`` a bump.
PINNED_SHIM_VERSION = 4

PINNED_TEMPLATE_SHA256 = {
    "codex":
        "47c85ee8341f9db4c1483db7bee3dcb698afbe9799809efe530263ed264783e6",
    "opencode":
        "8d091e60a543975c9bc21fb2e9b59ea267340741f12701a68166f5d0487a7c3b",
}


class TestTemplateBodiesArePinnedToTheirGeneration:
    """A shipped template's bytes are part of the install contract (D7b).

    Two things depend on them and neither can see an edit on its own.
    ``--check`` row 4 compares the ``template_sha256`` a shim recorded when it was
    written against the template on disk, so every template edit silently
    invalidates every shim already installed from it — a fact worth landing on the
    author's terminal rather than on a user's. And ``SHIM_VERSION`` is the
    hand-maintained half of that row: it is one integer for **every** host, moved
    by a change to the rendered shape, and it is the only signal for the changes a
    template digest cannot see (a snippet, a verb sentence, `render` itself). It
    has been missed before — the ``--exec-subagents`` that OpenCode's dispatcher
    line gained left the counter at 1 — which is what this pin exists to stop.
    """

    def test_the_pin_covers_exactly_the_installable_hosts(self):
        assert sorted(PINNED_TEMPLATE_SHA256) == INSTALLABLE, (
            "a host shipping a shim template must pin its digest here: add the "
            f"missing entry to PINNED_TEMPLATE_SHA256 in {Path(__file__).name}."
        )

    @pytest.mark.parametrize("host_id", INSTALLABLE)
    def test_the_template_still_hashes_to_its_pinned_digest(self, host_id):
        template = hi.template_path(host_id)
        actual = hi._sha256(template.read_text(encoding="utf-8"))
        relative = template.relative_to(SKILL_ROOT)
        assert actual == PINNED_TEMPLATE_SHA256[host_id], (
            f"you changed {relative}. Every shim already installed from it now "
            f"fails `install-host {host_id} --check` row 4, which is correct and "
            "intended — this test is the notice, not the objection. Do two "
            f"things: (1) if the edit changed the rendered *shape* (a new step, a "
            "new slot, a changed command form), bump SHIM_VERSION in "
            "utils/host_install.py and PINNED_SHIM_VERSION here to the same "
            f"number; (2) set PINNED_TEMPLATE_SHA256['{host_id}'] in "
            f"{Path(__file__).name} to {actual}."
        )

    def test_the_pinned_generation_is_the_shipped_one(self):
        assert PINNED_SHIM_VERSION == hi.SHIM_VERSION, (
            f"SHIM_VERSION moved to {hi.SHIM_VERSION} without this block being "
            "revisited. Re-read every digest above against the templates as they "
            f"stand and set PINNED_SHIM_VERSION to {hi.SHIM_VERSION}: the pin is "
            "a statement about which template bodies belong to which generation, "
            "so a bump that leaves it behind makes it a statement about neither."
        )


# ---------------------------------------------------------------------------
# 2. --check: five assertions, and how each one fails
# ---------------------------------------------------------------------------

class TestCheckAssertions:
    HOST = "codex"

    def test_all_five_pass_after_an_install(self, skill, env, project, ready):
        install(self.HOST, skill, env, project)
        result = check(self.HOST, skill, env, project)
        assert [row.number for row in result.rows] == [1, 2, 3, 4, 5]
        assert result.ok, [(r.number, r.detail) for r in result.failures]

    def test_row_one_fails_when_no_shim_is_installed(
        self, skill, env, project, ready
    ):
        result = check(self.HOST, skill, env, project)
        assert not result.ok
        row = result.row(hi.ROW_PRESENT)
        assert not row.ok and "no shim found" in row.detail
        assert "install-host" in row.fix

    def test_row_two_fails_when_the_skill_dir_moved(
        self, skill, env, project, ready, tmp_path
    ):
        install(self.HOST, skill, env, project)
        moved = tmp_path / "moved skill dir"
        moved.mkdir()
        for name in FIXTURE_CONTENT:
            source = skill / name
            if source.is_dir():
                shutil.copytree(source, moved / name)
            else:
                shutil.copy2(source, moved / name)
        caps.clear_cache()
        result = check(self.HOST, moved, env, project)
        row = result.row(hi.ROW_PREFIX)
        assert not row.ok and "moved" in row.detail
        assert "install-host" in row.fix

    def test_row_two_fails_when_the_baked_prefix_no_longer_exists(
        self, skill, env, project, ready, tmp_path
    ):
        """The updated-or-removed-plugin case: same repair line, own message."""
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        gone = tmp_path / "deleted skill dir"
        text = path.read_text(encoding="utf-8").replace(str(skill), str(gone))
        path.write_text(text, encoding="utf-8")
        result = check(self.HOST, skill, env, project)
        row = result.row(hi.ROW_PREFIX)
        assert not row.ok and "no longer exists" in row.detail

    def test_row_three_fails_on_a_broken_shape(self, skill, env, project, ready):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        text = path.read_text(encoding="utf-8").replace("command -v uv", "true")
        path.write_text(text, encoding="utf-8")
        row = check(self.HOST, skill, env, project).row(hi.ROW_SHAPE)
        assert not row.ok and "preflight" in row.detail

    @pytest.mark.parametrize(
        "mutation,expected",
        [
            (lambda t: t.replace("---\n", "", 1), "frontmatter"),
            (lambda t: t.replace("--host codex", "--host opencode"), "bakes --host"),
            (lambda t: t.replace("--yes", ""), "never passes --yes"),
            (lambda t: t + "\n{SKILL_DIR}\n", "survived rendering"),
            (lambda t: t + "\n{{ask_user}}\n", "never filled"),
        ],
    )
    def test_row_three_names_what_broke(self, skill, env, project, ready,
                                        mutation, expected):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")
        row = check(self.HOST, skill, env, project).row(hi.ROW_SHAPE)
        assert not row.ok and expected in row.detail, row.detail

    def test_row_four_fails_on_an_older_generation(self, skill, env, project,
                                                   ready):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        text = path.read_text(encoding="utf-8").replace(
            f'"shim_version": {hi.SHIM_VERSION}', '"shim_version": 0'
        )
        path.write_text(text, encoding="utf-8")
        row = check(self.HOST, skill, env, project).row(hi.ROW_GENERATION)
        assert not row.ok and "older multi-llm" in row.detail
        assert "install-host" in row.fix

    def test_row_four_fails_on_a_file_we_never_wrote(self, skill, env, project,
                                                     ready):
        path = target(self.HOST, skill, env, project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            hi.render(self.HOST, base=str(skill)), encoding="utf-8"
        )
        row = check(self.HOST, skill, env, project).row(hi.ROW_GENERATION)
        assert not row.ok and hi.HEADER_KEY in row.detail

    def test_row_four_names_the_template_a_pristine_shim_came_from(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        row = check(self.HOST, skill, env, project).row(hi.ROW_GENERATION)
        assert row.ok
        assert "current template" in row.detail, row.detail

    def test_row_four_fails_when_the_template_body_has_moved(
        self, skill, env, project, ready
    ):
        """The shim is untouched; the template it was rendered from is not.

        This is the shape the defect really had: `with_header` has recorded a
        `template_sha256` since D7 and nothing read it, so a shim rendered from a
        superseded body kept its valid shape, its honest body hash and a
        `shim_version` that had never moved — and reported "generation current"
        while missing whatever the template edit had added.
        """
        install(self.HOST, skill, env, project)
        template = hi.template_path(self.HOST, str(skill))
        template.write_text(
            template.read_text(encoding="utf-8") + "\nA later revision.\n",
            encoding="utf-8",
        )
        result = check(self.HOST, skill, env, project)
        row = result.row(hi.ROW_GENERATION)
        assert not row.ok, row.detail
        assert "superseded" in row.detail and "template body has changed" in row.detail
        assert str(caps.entry(self.HOST, str(skill))["shim"]["template"]) in row.detail
        assert "install-host" in row.fix
        # Only row 4: the file on disk is still a byte-valid shim of ours, which
        # is exactly why no other row could have caught this.
        assert [r.number for r in result.failures] == [hi.ROW_GENERATION]

    def test_row_four_fails_a_header_that_records_no_template_digest(
        self, skill, env, project, ready
    ):
        """A missing field is its own failure, not an older generation.

        `shim_version` has never left 1, so "this predates the field" is not a
        reading `--check` can fall back on: every install-host that has shipped
        writes `template_sha256`, and a header without one was hand-assembled or
        truncated. Treating it as absent-therefore-fine would pass precisely the
        file nothing else can vouch for.
        """
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        stripped = re.sub(
            r', "template_sha256": "[0-9a-f]+"', "",
            path.read_text(encoding="utf-8"), count=1,
        )
        assert "template_sha256" not in stripped
        path.write_text(stripped, encoding="utf-8")
        row = check(self.HOST, skill, env, project).row(hi.ROW_GENERATION)
        assert not row.ok
        assert "records no template_sha256" in row.detail, row.detail
        assert "install-host" in row.fix

    def test_row_four_reports_the_coarser_generation_first(
        self, skill, env, project, ready
    ):
        """Both stale: the `shim_version` miss is the message, not the digest.

        A shim from an older multi-llm was rendered by an older *renderer* too, so
        its template digest is stale as a consequence rather than as the finding.
        Naming the consequence would send the reader at the template.
        """
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                f'"shim_version": {hi.SHIM_VERSION}', '"shim_version": 0'
            ),
            encoding="utf-8",
        )
        template = hi.template_path(self.HOST, str(skill))
        template.write_text(
            template.read_text(encoding="utf-8") + "\nA later revision.\n",
            encoding="utf-8",
        )
        row = check(self.HOST, skill, env, project).row(hi.ROW_GENERATION)
        assert not row.ok
        assert "older multi-llm" in row.detail and "superseded" not in row.detail

    def test_a_host_with_no_template_has_no_template_to_have_moved(self, skill):
        """`shim.template: null` is a legitimate empty, never a stale render.

        The reference host's shim is the plugin's own SKILL.md, which nobody
        renders — `_template_sha` returns "" for it, and a digest comparison
        against "" would fail the one host that cannot have one.
        """
        assert hi._template_sha("claude-code", str(skill)) == ""
        assert hi._template_problem("claude-code", {}, str(skill)) == ""

    def test_row_five_fails_with_the_single_sourced_uv_text(
        self, skill, env, project, monkeypatch, capsys
    ):
        """D6c layer 1: the install-time half of the uv bootstrap.

        Layer 3 cannot cover this — every command runs through `uv run`, so the
        shell fails with exit 127 before any Python could print the message — and
        the wording is the file's, byte for byte, never a paraphrase.
        """
        monkeypatch.setattr(hi, "runner_available", lambda provider: True)
        monkeypatch.setenv("PATH", str(skill))  # a PATH with no uv on it
        monkeypatch.delenv("MULTI_LLM_UV", raising=False)
        monkeypatch.setattr(hi, "find_uv", lambda: None)
        row = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)
        assert not row.ok
        expected = (skill / "hosts" / "uv-missing.txt").read_text(
            encoding="utf-8"
        ).rstrip("\n")
        assert expected in row.detail
        assert "[UV_MISSING]" in row.detail

    def test_row_five_fails_when_the_runner_cli_is_missing(
        self, skill, env, project, monkeypatch
    ):
        monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
        monkeypatch.setattr(hi, "runner_available", lambda provider: False)
        row = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)
        assert not row.ok
        runner, _ = hi.configured_runner(
            self.HOST, base=str(skill), project_root=project
        )
        assert runner in row.detail

    def test_row_five_checks_the_runner_that_would_really_be_dispatched(
        self, skill, env, project, monkeypatch, tmp_path
    ):
        """The layered runner is what runs, so it is what --check probes (D3o).

        Reading the capability entry's `default_subagent_runner` instead would
        green-light a project configured for another runner while that one is
        absent, logged out or too old — and fail a project over a default nothing
        on it would ever dispatch. The layer that supplied the value is reported
        with it, because "I set it and nothing changed" is otherwise a support
        question.
        """
        override = tmp_path / "providers.yaml"
        override.write_text(
            'driver:\n  subagent_runner: "opencode:opencode/sonnet"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(override))
        monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
        probed = []
        monkeypatch.setattr(
            hi, "runner_available", lambda provider: probed.append(provider)
        )
        row = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)
        entry_default = caps.entry(self.HOST, str(skill))["default_subagent_runner"]
        assert probed == ["opencode"], probed
        assert not row.ok
        assert "opencode:opencode/sonnet" in row.detail
        assert "supplied by env" in row.detail
        assert entry_default not in row.detail

    def test_row_five_reports_auth_and_version_not_just_presence(
        self, skill, env, project, monkeypatch
    ):
        """Installed is not authenticated (D3c), and `which` cannot tell them apart.

        A logged-out or too-old CLI passes detection and then fails once per task
        in the first batch it is dispatched into, so the probe the executor runs
        per phase is the probe --check runs here.
        """
        monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
        monkeypatch.setattr(hi, "runner_available", lambda provider: True)
        monkeypatch.setattr(
            hi, "probe_runner",
            lambda spec, **kwargs: {
                "auth": "unknown", "version": "unknown",
                "problem": f"Refusing to dispatch: {spec} (reason: not_authenticated)",
            },
        )
        row = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)
        assert not row.ok and "not_authenticated" in row.detail

        monkeypatch.setattr(
            hi, "probe_runner",
            lambda spec, **kwargs: {
                "auth": "ok", "version": "1.2.3", "problem": "",
            },
        )
        row = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)
        assert row.ok, row.detail
        assert "auth=ok" in row.detail and "version=1.2.3" in row.detail
        assert "nested_exec=" in row.detail

    def test_row_five_prints_the_adapters_own_metadata_readout(
        self, skill, env, project, monkeypatch
    ):
        """M13 (§5.2): the readout is demoted out of the auth path, not deleted.

        cursor-agent's `about --format json` reports a plan tier no other probe
        reports, and `--check` is the surface that shows it. It is *context*:
        it never changes the row's verdict, which is what the second half
        asserts by flipping the readout on and off over the same facts.
        """
        monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
        monkeypatch.setattr(hi, "runner_available", lambda provider: True)
        facts = {"auth": "ok", "version": "1.2.3", "problem": ""}
        monkeypatch.setattr(hi, "probe_runner", lambda spec, **kw: dict(facts))
        without = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)

        monkeypatch.setattr(
            hi, "probe_runner",
            lambda spec, **kw: dict(facts, metadata="plan-tier=pro"),
        )
        row = check(self.HOST, skill, env, project).row(hi.ROW_ENVIRONMENT)

        assert row.ok is without.ok is True, row.detail
        assert "plan-tier=pro" in row.detail
        assert "plan-tier" not in without.detail
        # The verdict is the same sentence either way; only the facts grew.
        assert "auth=ok" in row.detail and "state=" in row.detail

    def test_the_runner_probe_reuses_the_init_detection(self):
        """One detector, not two: `--init` and `--check` ask the same question."""
        import init_config

        source = Path(hi.__file__).read_text(encoding="utf-8")
        assert "provider_registry import get_provider" in source
        assert "is_available()" in source
        assert hi.runner_available("definitely-not-a-provider") is False
        assert init_config._available_providers({}) == []

    def test_the_reload_state_is_reported_and_does_not_fail(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        result = check(self.HOST, skill, env, project)
        assert result.ok
        assert any("[NOT_YET_LOADED]" in note for note in result.notes)

    def test_a_non_installable_host_runs_rows_three_to_five(
        self, skill, env, project, ready
    ):
        """`install-host claude-code` is --check-only (D5c)."""
        result = check("claude-code", skill, env, project)
        assert [row.number for row in result.rows] == [3, 4, 5]
        assert result.ok, [(r.number, r.detail) for r in result.failures]
        assert result.row(hi.ROW_GENERATION).detail.startswith("n/a")

    def test_installing_the_reference_host_is_refused(self, skill, env, project,
                                                      ready):
        with pytest.raises(hi.Refused) as excinfo:
            install("claude-code", skill, env, project)
        assert "--check-only" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Lifecycle: idempotency, upgrade, local edits, rollback
# ---------------------------------------------------------------------------

class TestLifecycle:
    HOST = "opencode"

    def test_a_second_run_is_byte_identical_and_writes_nothing(
        self, skill, env, project, ready, capsys
    ):
        assert install(self.HOST, skill, env, project) == 0
        path = target(self.HOST, skill, env, project)
        first = path.read_bytes()
        stamp = path.stat().st_mtime_ns
        capsys.readouterr()
        assert install(self.HOST, skill, env, project) == 0
        assert path.read_bytes() == first
        assert "[SHIM_UNCHANGED]" in capsys.readouterr().out
        assert path.stat().st_mtime_ns == stamp, (
            "an unchanged re-install rewrites nothing: an installer whose second "
            "run differs from its first cannot be driven from a post-update hook"
        )

    def test_the_render_is_deterministic_across_processes(self, skill):
        assert hi.render_installed(self.HOST, base=str(skill)) == (
            hi.render_installed(self.HOST, base=str(skill))
        )

    def test_no_temp_file_or_backup_survives_a_successful_install(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        leftovers = [
            child.name for child in path.parent.iterdir()
            if child.name != path.name
        ]
        assert leftovers == [], leftovers

    def test_a_relocated_skill_dir_is_repaired_by_re_rendering(
        self, skill, env, project, ready, tmp_path
    ):
        install(self.HOST, skill, env, project)
        moved = tmp_path / "relocated skill dir"
        shutil.copytree(skill, moved)
        caps.clear_cache()
        assert not check(self.HOST, moved, env, project).ok
        assert install(self.HOST, moved, env, project) == 0
        result = check(self.HOST, moved, env, project)
        assert result.ok
        assert result.row(hi.ROW_PREFIX).detail == str(moved)

    def test_an_older_generation_is_repaired_by_re_rendering(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        stale = path.read_text(encoding="utf-8").replace(
            f'"shim_version": {hi.SHIM_VERSION}', '"shim_version": 0'
        )
        # Re-hash so the file is an honest older generation, not a local edit.
        path.write_text(
            hi.with_header(hi.body_of(stale), self.HOST, base=str(skill)).replace(
                f'"shim_version": {hi.SHIM_VERSION}', '"shim_version": 0'
            ),
            encoding="utf-8",
        )
        assert install(self.HOST, skill, env, project) == 0
        assert check(self.HOST, skill, env, project).ok

    def test_a_local_edit_is_refused_without_force(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        edited = path.read_text(encoding="utf-8") + "\nmy own note\n"
        path.write_text(edited, encoding="utf-8")
        with pytest.raises(hi.Refused) as excinfo:
            install(self.HOST, skill, env, project)
        assert "local edits" in str(excinfo.value)
        assert "--force" in str(excinfo.value)
        assert path.read_text(encoding="utf-8") == edited, "the edit survived"

    def test_force_overwrites_and_preserves_the_edit(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        edited = path.read_text(encoding="utf-8") + "\nmy own note\n"
        path.write_text(edited, encoding="utf-8")
        assert install(self.HOST, skill, env, project, force=True) == 0
        backup = Path(str(path) + ".local.bak")
        assert backup.read_text(encoding="utf-8") == edited
        assert path.read_text(encoding="utf-8") == hi.render_installed(
            self.HOST, base=str(skill)
        )

    def test_an_unrecognized_file_is_never_clobbered(
        self, skill, env, project, ready
    ):
        path = target(self.HOST, skill, env, project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# my own command\n", encoding="utf-8")
        with pytest.raises(hi.Refused) as excinfo:
            install(self.HOST, skill, env, project)
        assert hi.HEADER_KEY in str(excinfo.value)
        assert path.read_text(encoding="utf-8") == "# my own command\n"

    def test_forcing_over_an_unrecognized_file_preserves_it(
        self, skill, env, project, ready
    ):
        """Whatever --force had to override is kept, exactly as promised."""
        path = target(self.HOST, skill, env, project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# my own command\n", encoding="utf-8")
        assert install(self.HOST, skill, env, project, force=True) == 0
        assert Path(str(path) + ".local.bak").read_text(encoding="utf-8") == (
            "# my own command\n"
        )
        assert hi.parse_header(path.read_text(encoding="utf-8"))["host"] == self.HOST

    def test_a_target_owned_by_another_host_is_refused(
        self, skill, env, project, ready
    ):
        """Two hosts writing one path is a conflict, reported not resolved."""
        path = target(self.HOST, skill, env, project)
        path.parent.mkdir(parents=True, exist_ok=True)
        other = hi.render_installed("codex", base=str(skill))
        path.write_text(other, encoding="utf-8")
        with pytest.raises(hi.Refused) as excinfo:
            install(self.HOST, skill, env, project)
        message = str(excinfo.value)
        assert "'codex'" in message and f"'{self.HOST}'" in message
        assert path.read_text(encoding="utf-8") == other

    def test_a_failed_verification_restores_the_previous_bytes(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        previous = path.read_bytes()
        # Force the *next* render to differ, then make verification fail.
        monkeypatch.setattr(
            hi, "render_installed",
            lambda host_id, base=None: hi.with_header(
                hi.render(host_id, base=base) + "\nchanged\n", host_id, base=base
            ),
        )
        monkeypatch.setattr(
            hi, "check",
            lambda *a, **k: hi.CheckResult(
                host_id=self.HOST,
                rows=[hi.Row(hi.ROW_SHAPE, False, "forced failure")],
            ),
        )
        with pytest.raises(hi.Refused) as excinfo:
            install(self.HOST, skill, env, project)
        assert "rolled back" in str(excinfo.value)
        assert path.read_bytes() == previous, "the previous shim was not restored"
        capsys.readouterr()

    def test_a_failed_first_install_leaves_no_file(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        path = target(self.HOST, skill, env, project)
        monkeypatch.setattr(
            hi, "check",
            lambda *a, **k: hi.CheckResult(
                host_id=self.HOST,
                rows=[hi.Row(hi.ROW_SHAPE, False, "forced failure")],
            ),
        )
        with pytest.raises(hi.Refused) as excinfo:
            install(self.HOST, skill, env, project)
        assert "nothing was left behind" in str(excinfo.value)
        assert not path.exists()
        capsys.readouterr()

    def test_an_environment_failure_keeps_the_written_shim(
        self, skill, env, project, monkeypatch, capsys
    ):
        """Row 5 reports; it does not roll back a byte-correct shim.

        Deleting a correct file cannot make a missing CLI appear, and the user's
        next step is to install that CLI rather than to re-render.
        """
        monkeypatch.setattr(hi, "find_uv", lambda: None)
        monkeypatch.setattr(hi, "runner_available", lambda provider: True)
        with pytest.raises(hi.Refused) as excinfo:
            install(self.HOST, skill, env, project)
        assert "installed and correct" in str(excinfo.value)
        assert target(self.HOST, skill, env, project).is_file()
        capsys.readouterr()


# ---------------------------------------------------------------------------
# 4. Scopes, precedence and compatibility locations
# ---------------------------------------------------------------------------

class TestScopeAndPrecedence:
    HOST = "codex"

    def test_the_two_scopes_write_two_different_files(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project, scope=hi.SCOPE_GLOBAL)
        install(self.HOST, skill, env, project, scope=hi.SCOPE_PROJECT)
        globally = target(self.HOST, skill, env, project, hi.SCOPE_GLOBAL)
        locally = target(self.HOST, skill, env, project, hi.SCOPE_PROJECT)
        assert globally != locally
        assert globally.read_bytes() == locally.read_bytes()

    def test_check_reports_every_copy_in_precedence_order(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project, scope=hi.SCOPE_GLOBAL)
        install(self.HOST, skill, env, project, scope=hi.SCOPE_PROJECT)
        result = check(self.HOST, skill, env, project)
        scopes = [copy.scope for copy in result.copies]
        assert scopes == [hi.SCOPE_PROJECT, hi.SCOPE_GLOBAL]
        assert result.loaded.scope == hi.SCOPE_PROJECT, (
            "project scope wins on every host surveyed, and --check names the "
            "copy that will actually load"
        )

    def test_a_global_install_warns_about_a_shadowing_project_shim(
        self, skill, env, project, ready, capsys
    ):
        install(self.HOST, skill, env, project, scope=hi.SCOPE_PROJECT)
        capsys.readouterr()
        install(self.HOST, skill, env, project, scope=hi.SCOPE_GLOBAL)
        assert "[SHIM_SHADOWED]" in capsys.readouterr().out

    def test_a_compatibility_location_is_found_but_never_written(
        self, skill, env, project, ready, capsys
    ):
        compat = [
            location
            for location in hi.locations(
                self.HOST, base=str(skill), env=env, project_root=project
            )
            if not location.writable
        ]
        assert compat, "this host declares no compatibility location to test"
        legacy = compat[0]
        legacy.directory.mkdir(parents=True, exist_ok=True)
        legacy.path.write_text(
            hi.render_installed(self.HOST, base=str(skill)), encoding="utf-8"
        )
        install(self.HOST, skill, env, project, scope=hi.SCOPE_GLOBAL)
        out = capsys.readouterr().out
        assert str(legacy.path) in out
        assert legacy.path.is_file()
        written = target(self.HOST, skill, env, project, hi.SCOPE_GLOBAL)
        assert written != legacy.path

    def test_the_file_name_comes_from_the_entry_not_from_a_convention(
        self, skill
    ):
        assert hi.shim_filename("codex", str(skill)) == "SKILL.md"
        assert hi.shim_filename("opencode", str(skill)) == "multi-llm.md"


class TestInstallPathsAreDisjoint:
    """A shim installed for one host must not collide with or shadow another's."""

    def test_preferred_paths_are_pairwise_disjoint(self, env, project):
        seen = {}
        for host_id in INSTALLABLE:
            for scope in hi.WRITABLE_SCOPES:
                path = hi.target_location(
                    host_id, scope, env=env, project_root=project
                ).path
                assert path not in seen, (
                    f"{host_id} and {seen.get(path)} both install to {path}: "
                    "overwriting would swap the baked --host, and with it the "
                    "whole capability profile, out from under the other host"
                )
                seen[path] = host_id

    def test_no_hosts_target_directory_contains_anothers(self, env, project):
        directories = {
            host_id: {
                hi.target_location(
                    host_id, scope, env=env, project_root=project
                ).directory
                for scope in hi.WRITABLE_SCOPES
            }
            for host_id in INSTALLABLE
        }
        for host_id, mine in directories.items():
            for other_id, theirs in directories.items():
                if host_id == other_id:
                    continue
                for one in mine:
                    for two in theirs:
                        assert one != two and two not in one.parents, (
                            f"{other_id}'s directory {two} contains {host_id}'s "
                            f"{one}; an install for one host would sit inside the "
                            "other's tree"
                        )

    def test_every_installed_shim_names_only_its_own_host(
        self, skill, env, project, ready
    ):
        for host_id in INSTALLABLE:
            install(host_id, skill, env, project)
        for host_id in INSTALLABLE:
            text = target(host_id, skill, env, project).read_text(
                encoding="utf-8"
            )
            assert hi.parse_header(text)["host"] == host_id
            for other in INSTALLABLE:
                if other != host_id:
                    assert f"--host {other}" not in text


# ---------------------------------------------------------------------------
# 5. uninstall-host
# ---------------------------------------------------------------------------

class TestUninstall:
    HOST = "codex"

    def uninstall(self, skill, env, project, **kwargs):
        return hi.uninstall(
            self.HOST, base=str(skill), env=env, project_root=project, **kwargs
        )

    def test_dry_run_lists_and_changes_nothing(self, skill, env, project, ready,
                                               capsys):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        capsys.readouterr()
        assert self.uninstall(skill, env, project, dry_run=True) == 0
        out = capsys.readouterr().out
        assert "[SHIM_WOULD_REMOVE]" in out and str(path) in out
        assert path.is_file()

    def test_it_removes_only_what_carries_our_block(self, skill, env, project,
                                                    ready, capsys):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        assert self.uninstall(skill, env, project) == 0
        assert not path.exists()
        assert "[SHIM_REMOVED]" in capsys.readouterr().out

    def test_an_empty_skill_directory_it_created_is_pruned(
        self, skill, env, project, ready
    ):
        install(self.HOST, skill, env, project)
        directory = target(self.HOST, skill, env, project).parent
        assert directory.name == "multi-llm"
        self.uninstall(skill, env, project)
        assert not directory.exists()

    def test_a_shared_command_directory_is_left_alone(
        self, skill, env, project, ready
    ):
        hi.install(
            "opencode", base=str(skill), env=env, project_root=project,
            scope=hi.SCOPE_PROJECT,
        )
        directory = hi.target_location(
            "opencode", hi.SCOPE_PROJECT, base=str(skill), env=env,
            project_root=project,
        ).directory
        hi.uninstall(
            "opencode", base=str(skill), env=env, project_root=project
        )
        assert directory.is_dir(), (
            "a shared commands/ directory is not ours to remove even when empty"
        )

    def test_a_file_of_someone_elses_is_reported_not_removed(
        self, skill, env, project, ready, capsys
    ):
        path = target(self.HOST, skill, env, project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# not ours\n", encoding="utf-8")
        assert self.uninstall(skill, env, project) == 2
        out = capsys.readouterr().out
        assert "[SHIM_KEPT]" in out and hi.HEADER_KEY in out
        assert path.is_file()

    def test_an_edited_shim_needs_force(self, skill, env, project, ready,
                                       capsys):
        install(self.HOST, skill, env, project)
        path = target(self.HOST, skill, env, project)
        path.write_text(
            path.read_text(encoding="utf-8") + "\nmine\n", encoding="utf-8"
        )
        assert self.uninstall(skill, env, project) == 2
        assert "local edits" in capsys.readouterr().out
        assert path.is_file()
        assert self.uninstall(skill, env, project, force=True) == 0
        assert not path.exists()

    def test_a_compatibility_location_is_reported_not_removed(
        self, skill, env, project, ready, capsys
    ):
        compat = [
            location
            for location in hi.locations(
                self.HOST, base=str(skill), env=env, project_root=project
            )
            if not location.writable
        ][0]
        compat.directory.mkdir(parents=True, exist_ok=True)
        compat.path.write_text(
            hi.render_installed(self.HOST, base=str(skill)), encoding="utf-8"
        )
        assert self.uninstall(skill, env, project) == 2
        assert "compatibility location" in capsys.readouterr().out
        assert compat.path.is_file()

    def test_nothing_to_remove_is_exit_two(self, skill, env, project, ready,
                                           capsys):
        assert self.uninstall(skill, env, project) == 2
        assert "[SHIM_NOT_FOUND]" in capsys.readouterr().out

    def test_a_scope_selector_limits_the_candidates(self, skill, env, project,
                                                    ready):
        install(self.HOST, skill, env, project, scope=hi.SCOPE_GLOBAL)
        install(self.HOST, skill, env, project, scope=hi.SCOPE_PROJECT)
        assert self.uninstall(
            skill, env, project, scopes=(hi.SCOPE_PROJECT,)
        ) == 0
        assert not target(
            self.HOST, skill, env, project, hi.SCOPE_PROJECT
        ).exists()
        assert target(self.HOST, skill, env, project, hi.SCOPE_GLOBAL).is_file()


# ---------------------------------------------------------------------------
# 6. The Windows seam (asserted, never skipped)
# ---------------------------------------------------------------------------

class TestWindowsSeam:
    def test_userprofile_expands_when_home_is_absent(self, tmp_path):
        root = tmp_path / "Users" / "First Last"
        expanded = hi.expand_path_tokens(
            "$HOME/.codex/skills/multi-llm", env={"USERPROFILE": str(root)}
        )
        assert expanded == Path(os.path.abspath(root / ".codex/skills/multi-llm"))

    def test_the_userprofile_token_itself_is_understood(self, tmp_path):
        root = tmp_path / "Users" / "First Last"
        assert hi.expand_path_tokens(
            "%USERPROFILE%/.codex", env={"USERPROFILE": str(root)}
        ) == Path(os.path.abspath(root / ".codex"))

    def test_home_wins_when_both_are_set(self, tmp_path):
        """Git Bash sets HOME; native Windows Python sees USERPROFILE."""
        assert hi.user_home(
            {"HOME": str(tmp_path / "bash"), "USERPROFILE": str(tmp_path / "win")}
        ) == tmp_path / "bash"

    def test_xdg_is_honoured_and_falls_back_to_dot_config(self, tmp_path):
        assert hi.expand_path_tokens(
            "$XDG_CONFIG_HOME/opencode/commands",
            env={"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "cfg")},
        ) == Path(os.path.abspath(tmp_path / "cfg/opencode/commands"))
        assert hi.expand_path_tokens(
            "$XDG_CONFIG_HOME/opencode/commands", env={"HOME": str(tmp_path)}
        ) == Path(os.path.abspath(tmp_path / ".config/opencode/commands"))

    def test_the_xdg_fallback_collapses_onto_the_home_entry(self, env, project):
        """Two global entries, one path when XDG is unset — reported once."""
        found = hi.locations("opencode", env=env, project_root=project)
        globals_ = [
            location for location in found
            if location.scope == hi.SCOPE_GLOBAL and location.writable
        ]
        assert len(globals_) == 1, [str(g.path) for g in globals_]

    def test_a_relative_entry_resolves_against_the_project_root(self, tmp_path):
        root = tmp_path / "my project"
        assert hi.expand_path_tokens(
            ".opencode/commands", env={}, project_root=root
        ) == Path(os.path.abspath(root / ".opencode/commands"))

    def test_git_bash_is_the_same_hard_prerequisite(self, skill, env, project,
                                                    monkeypatch):
        install(self.__class__ and "codex", skill, env, project) if False else None
        monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
        monkeypatch.setattr(hi, "runner_available", lambda provider: True)
        monkeypatch.setattr(hi.shutil, "which", lambda name: None)
        row = hi._environment_row("codex", base=str(skill), windows=True)
        assert not row.ok
        assert "Git for Windows" in row.detail
        assert "git-scm.com/download/win" in row.detail

    def test_the_git_bash_message_agrees_with_the_reference_doc(self):
        doc = (SKILL_ROOT / "references" / "uv-check.md").read_text(
            encoding="utf-8"
        )
        assert "git-scm.com/download/win" in doc
        assert "Git for Windows" in hi.GIT_BASH_MISSING

    def test_the_installer_never_reaches_for_a_symlink_api(self):
        """No symlink, no junction, no elevation — enforced, not just documented."""
        tree = ast.parse(
            Path(hi.__file__).read_text(encoding="utf-8"), filename=hi.__file__
        )
        offenders = [
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Attribute, ast.Name))
            and (getattr(node, "attr", None) or getattr(node, "id", None))
            in ("symlink", "symlink_to", "hardlink_to", "link", "mklink")
        ]
        assert not offenders, offenders

    def test_the_write_is_a_replace_into_the_target_directory(self):
        source = Path(hi.__file__).read_text(encoding="utf-8")
        assert "os.replace" in source
        assert "os.fsync" in source
        assert 'dir=str(target.parent)' in source, (
            "the temp file must live in the target directory, which is what makes "
            "os.replace atomic on Windows too"
        )
        assert 'os.fdopen(handle, "wb")' in source, (
            "the write is byte mode — identical bytes on every platform, which "
            "idempotency and the body hash both depend on, and which a backup or "
            "a rollback needs too: round-tripping a user's file through a text "
            "stream would rewrite its line endings"
        )
        assert ', "w",' not in source, (
            "a text-mode stream translates newlines on Windows; encode instead"
        )


# ---------------------------------------------------------------------------
# 7. The CLI surface, through the dispatcher
# ---------------------------------------------------------------------------

class TestCli:
    def test_the_modes_are_dispatchable_and_route_to_the_installer(self):
        for word in ("install-host", "uninstall-host"):
            mode = mode_table.resolve(word)
            assert mode.unavailable is None
            assert mode.target_module == "utils.host_install"
            assert mode.requires_uv is False, (
                "install-host --check IS layer 1 of the uv bootstrap: failing at "
                "the layer-3 backstop would replace a five-row prerequisite report "
                "with a one-line refusal"
            )

    def test_install_and_check_through_the_dispatcher(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        monkeypatch.chdir(project)
        monkeypatch.setenv("HOME", env["HOME"])
        monkeypatch.setenv("USERPROFILE", env["USERPROFILE"])
        monkeypatch.setattr(hi, "skill_dir", lambda: skill)
        assert multi_llm.main(["install-host", "codex"]) == 0
        out = capsys.readouterr().out
        assert "[SHIM_INSTALLED]" in out
        assert multi_llm.main(["install-host", "codex", "--check"]) == 0
        assert multi_llm.main(["install-host", "opencode", "--check"]) == 3
        err = capsys.readouterr().err
        assert "[INSTALL_CHECK_FAILED]" in err

    def test_check_all_covers_every_installed_host(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        monkeypatch.chdir(project)
        monkeypatch.setenv("HOME", env["HOME"])
        monkeypatch.setenv("USERPROFILE", env["USERPROFILE"])
        monkeypatch.setattr(hi, "skill_dir", lambda: skill)
        assert multi_llm.main(["install-host", "--check", "--all"]) == 2
        capsys.readouterr()
        for host_id in INSTALLABLE:
            assert multi_llm.main(["install-host", host_id]) == 0
        capsys.readouterr()
        assert multi_llm.main(["install-host", "--check", "--all"]) == 0
        out = capsys.readouterr().out
        for host_id in INSTALLABLE:
            assert f"host={host_id}" in out

    def test_all_and_a_host_id_together_are_refused(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        monkeypatch.setattr(hi, "skill_dir", lambda: skill)
        assert multi_llm.main(["install-host", "codex", "--all"]) == 3
        assert "not both" in capsys.readouterr().err

    def test_a_missing_host_id_is_exit_three(self, capsys):
        assert multi_llm.main(["install-host"]) == 3
        assert "needs a host id" in capsys.readouterr().err

    def test_an_unknown_host_is_exit_three_naming_the_known_ones(self, capsys):
        assert multi_llm.main(["install-host", "no-such-host"]) == 3
        err = capsys.readouterr().err
        assert "unknown host" in err and "codex" in err

    def test_the_result_line_is_emitted_and_records_the_artifact(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        from utils.cli_common import parse_result_line

        monkeypatch.chdir(project)
        monkeypatch.setenv("HOME", env["HOME"])
        monkeypatch.setenv("USERPROFILE", env["USERPROFILE"])
        monkeypatch.setattr(hi, "skill_dir", lambda: skill)
        assert multi_llm.main(["install-host", "codex"]) == 0
        payload = parse_result_line(capsys.readouterr().out)
        assert payload["mode"] == "install-host"
        assert payload["exit"] == 0 and payload["next"] is None
        assert payload["artifacts"] == [
            str(target("codex", skill, env, project))
        ]

    def test_uninstall_through_the_dispatcher(
        self, skill, env, project, ready, monkeypatch, capsys
    ):
        monkeypatch.chdir(project)
        monkeypatch.setenv("HOME", env["HOME"])
        monkeypatch.setenv("USERPROFILE", env["USERPROFILE"])
        monkeypatch.setattr(hi, "skill_dir", lambda: skill)
        assert multi_llm.main(["install-host", "codex"]) == 0
        capsys.readouterr()
        assert multi_llm.main(["uninstall-host", "codex", "--dry-run"]) == 0
        assert target("codex", skill, env, project).is_file()
        assert multi_llm.main(["uninstall-host", "codex"]) == 0
        assert not target("codex", skill, env, project).exists()
        assert multi_llm.main(["uninstall-host", "codex"]) == 2

    def test_the_install_modes_are_not_user_facing(self):
        hint = mode_table.render_argument_hint()
        assert "--install-host" not in hint
        assert "--uninstall-host" not in hint


# ---------------------------------------------------------------------------
# 8. Packaging and the generated documentation
# ---------------------------------------------------------------------------

def readme_path() -> Path:
    """The front page, resolved from the repo root.

    Deliberately *not* ``renderer.host_doc_path().parent.parent / "README.md"``:
    that expression is the repo root only while ``HOST_DOC`` happens to be
    exactly two segments deep. Move the generated section one level down —
    ``docs/hosts/index.md``, the obvious next step once ``docs/hosts/`` grows a
    second page — and the same walk lands on ``docs/README.md``, which today
    raises and, the day that file exists, silently guards the wrong document.
    ``HOST_DOC`` is free to move by one line (D2); the repo root is not.
    """
    return REPO_ROOT / "README.md"


def readme_body() -> str:
    return readme_path().read_text(encoding="utf-8")


def host_doc_body() -> str:
    """The generated host-install page, where the `$SKILL` recipes now live."""
    return renderer.host_doc_path().read_text(encoding="utf-8")


#: The README numbers its headings (``## 1. Installation``, ``### 1.2 Codex``);
#: the number is optional here so a renumbering never moves a guard.
_HEADING_NUMBER = r"(?:\d+(?:\.\d+)*\.?\s+)?"


def readme_heading_re(heading: str) -> "re.Pattern":
    """``## Installation`` -> a pattern matching it with or without its number."""
    hashes, _, title = heading.partition(" ")
    return re.compile(rf"^{hashes} {_HEADING_NUMBER}{re.escape(title)}", re.M)


def readme_section(body: str, heading: str) -> str:
    """One ``##`` section of the README, its heading line included."""
    lines = body.splitlines()
    pattern = readme_heading_re(heading)
    start = next(i for i, line in enumerate(lines) if pattern.match(line))
    end = next(
        (
            i for i, line in enumerate(lines[start + 1:], start + 1)
            if line.startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def path_spellings(path: str) -> set:
    """Every spelling of one ``install_paths`` entry a README may legitimately use.

    The data stores ``$HOME``; prose addressed to a human writes ``~``. Nothing
    else is normalized — in particular ``$XDG_CONFIG_HOME`` has no second
    spelling, which is exactly why naming its ``~/.config`` fallback *as the
    destination* was wrong rather than merely informal (D5).
    """
    return {path, path.replace("$HOME", "~"), path.replace("~", "$HOME")}


def mentions(text: str, path: str) -> bool:
    """Is *path* named as a path of its own, rather than as the tail of a longer one?

    ``.codex/skills/multi-llm/SKILL.md`` is a suffix of
    ``~/.codex/skills/multi-llm/SKILL.md``, so a plain substring test would let
    the ``--project`` half of the sentence vanish behind the global half and
    still pass.
    """
    return any(
        re.search(r"(?<![\w/~$.-])" + re.escape(spelling), text)
        for spelling in path_spellings(path)
    )


class TestPackagingAndDocs:
    #: Every ``SKILL.md`` the repository may carry outside ``.claude/`` (a
    #: developer's own workflow skills live there and ship nowhere): the core
    #: skill, and the codex plugin's generated thin skill (plans/codex-native-
    #: plugin.md, D1/D7), which sits under ``packaging/`` precisely so the
    #: skill tree still holds exactly one.
    PERMITTED_SKILL_MANIFESTS = frozenset({
        "skills/multi-llm/SKILL.md",
        "packaging/codex/multi-llm/skills/multi-llm/SKILL.md",
    })

    def test_exactly_the_permitted_skill_manifests_ship(self):
        """Repo-rooted and ``git ls-files``-based, against an explicit allow-set.

        Tracked *and* untracked-but-not-ignored, so a generated tree that is
        about to be committed is graded before the commit rather than after it.
        The skill-tree half still holds on its own: a second ``SKILL.md`` under
        ``skills/multi-llm/`` is a second auto-discoverable skill to every host
        that scans that directory.
        """
        import subprocess

        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
             "--", ":(glob)**/SKILL.md", ":(exclude).claude/**"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        manifests = {path for path in listed.split("\0") if path}
        assert manifests == self.PERMITTED_SKILL_MANIFESTS, (
            f"unexpected: {sorted(manifests - self.PERMITTED_SKILL_MANIFESTS)}; "
            f"missing: {sorted(self.PERMITTED_SKILL_MANIFESTS - manifests)}"
        )
        in_skill_tree = [
            path.relative_to(SKILL_ROOT).as_posix()
            for path in SKILL_ROOT.rglob("SKILL.md")
            if ".venv" not in path.parts and "tests" not in path.parts
        ]
        assert in_skill_tree == ["SKILL.md"], in_skill_tree

    def test_the_templates_are_tmpl_suffixed_and_declared(self):
        for host_id in INSTALLABLE:
            template = caps.entry(host_id)["shim"]["template"]
            assert template.endswith(".md.tmpl"), template
            assert (SKILL_ROOT / template).is_file()

    def test_the_reference_matrix_is_byte_identical_to_a_fresh_render(self):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        current = (SKILL_ROOT / renderer.OUTPUT).read_text(encoding="utf-8")
        assert current == renderer.render()

    def test_the_host_doc_section_is_byte_identical_to_a_fresh_render(self):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        host_doc = renderer.host_doc_path()
        assert host_doc.is_file(), host_doc
        body = host_doc.read_text(encoding="utf-8")
        assert renderer.HOST_DOC_BEGIN in body and renderer.HOST_DOC_END in body
        assert renderer.host_doc_with_section(body) == body, (
            f"{renderer.HOST_DOC}'s host-install section is generated from "
            "hosts/capabilities.yaml — re-run hosts/render_capabilities.py and "
            "commit the result; a hand-maintained copy is a CI failure"
        )

    def test_the_host_doc_section_carries_the_quickstarts_and_the_matrix(self):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        section = renderer.render_host_doc_section()
        assert "# Hosting multi-llm in other harnesses" in section
        assert "## After a plugin update" in section
        assert "install-host --check --all" in section
        for host_id in caps.host_ids():
            assert f"`{host_id}`" in section
        for host_id in INSTALLABLE:
            assert f"install-host {host_id} --check" in section

    def test_the_readme_points_at_the_host_doc_rather_than_carrying_it(self):
        """The section moved out of the README; the pointer is what replaced it.

        A trim that dropped the link would leave the two installable hosts with
        no route from the front page, which is the failure this catches — not
        the wording, only that the README still names the generated document and
        no longer holds a second, hand-maintained copy of its heading.

        The README is resolved by :func:`readme_path` — from the repo root, not
        by walking up from the generated file — so ``HOST_DOC`` can move without
        dragging this guard onto another document with it.
        """
        body = readme_body()
        assert renderer.HOST_DOC.replace(os.sep, "/") in body
        assert renderer.HOST_DOC_BEGIN not in body
        # The $SKILL recipes moved with the rest of the install page; a second
        # copy on the front page is the hand-maintained drift this guard exists
        # to prevent.
        assert "SKILL=$(claude plugin list" not in body
        assert "SKILL=$(dirname" not in body

    def test_the_readme_is_still_found_when_the_host_doc_moves_a_level_down(
        self, monkeypatch
    ):
        """D2's promise, made testable: retargeting the section is *one* line.

        This script keys on the two markers and never on the filename, so
        ``docs/hosts/index.md`` is a one-line edit to ``HOST_DOC``. It only stays
        one line while nothing else derives the repo root from that constant's
        depth: the previous spelling of the test above resolved
        ``host_doc_path().parent.parent / "README.md"``, which under this
        monkeypatch is ``docs/README.md`` — a second edit in a second file today,
        and a guard silently pointed at the wrong document the day that file
        exists.
        """
        monkeypatch.setattr(
            renderer, "HOST_DOC", os.path.join("docs", "hosts", "index.md")
        )
        assert readme_path() == REPO_ROOT / "README.md"
        assert readme_path().is_file()
        assert renderer.host_doc_path() == REPO_ROOT / "docs" / "hosts" / "index.md"
        assert renderer.host_doc_path().parent.parent / "README.md" != readme_path(), (
            "the README lookup is deriving the repo root from HOST_DOC's depth "
            "again; anchor it on REPO_ROOT so the constant stays free to move"
        )

    def test_the_check_guard_covers_both_generated_files(self, monkeypatch,
                                                         capsys):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        assert renderer.main(["--check"]) == 0
        monkeypatch.setattr(
            renderer, "render_host_doc_section", lambda base=None: "drifted\n"
        )
        assert renderer.main(["--check"]) == 1
        assert renderer.HOST_DOC in capsys.readouterr().err


class TestTheOpeningSentenceOfTheGeneratedPage:
    """`_installable_prose()` at every arity, and the sentence built from it.

    Two halves that do not overlap. The parametrized rows pin the **join**; the
    sentence tests pin that `render_host_doc_section()` actually composes its
    opening from the helper. Neither is what the shipped assertions above make:
    those check that each current host id appears somewhere in the section as an
    isolated `` `codex` `` token, which a mangled join satisfies too (D4).
    """

    @pytest.mark.parametrize(
        "host_ids,expected",
        [
            ([], "no other host yet"),
            (["a"], "`a`"),
            (["a", "b"], "`a` and `b`"),
            (["a", "b", "c"], "`a`, `b` and `c`"),
        ],
    )
    def test_the_helper_renders_one_shape_per_arity(self, host_ids, expected):
        assert renderer._installable_prose(host_ids) == expected

    def test_the_rendered_opening_is_the_sentence_composed_from_the_helper(self):
        installable = [
            host_id for host_id in sorted(caps.host_ids())
            if caps.entry(host_id)["installable"]
        ]
        expected = (
            "multi-llm ships as a Claude Code plugin, and it runs unchanged "
            f"inside {renderer._installable_prose(installable)} as well."
        )
        section = renderer.render_host_doc_section()
        assert section.splitlines()[2].startswith(expected), (
            "the generated page's first sentence is not the one composed from "
            "_installable_prose(); a hand-written host list there is the exact "
            "staleness this whole file exists to prevent"
        )

    def test_the_empty_list_gets_its_own_sentence_not_an_empty_slot(
        self, monkeypatch
    ):
        """The one arity the D4 sentence cannot absorb.

        Flipping every ``installable:`` in ``hosts/capabilities.yaml`` to
        ``false`` is a data-only change — no code edit, nothing to review in a
        diff of ``.py`` files — so the sentence it produces has to be true on
        its own. "…it runs unchanged inside no other host yet as well" is not:
        the helper's contract is the *list*, and the repair belongs to the
        sentence around it.
        """
        data = copy.deepcopy(caps.load())
        for host in data["hosts"].values():
            host["installable"] = False
        monkeypatch.setattr(caps, "load", lambda base=None: data)

        section = renderer.render_host_doc_section()
        assert section.splitlines()[2].startswith(renderer._opening_sentence([]))
        assert "no other host is installable yet" in section
        assert "inside no other host yet as well" not in section
        assert "no other host yet" not in section, (
            "the helper's bare noun phrase reached the page: it is a list, not "
            "a sentence, and interpolating it states something false"
        )


class TestTheHostDocMustExistToBeUpToDate:
    """Absence is staleness, not a skipped step.

    Both the ``--check`` path and the write path used to wrap the host doc in
    ``if host_doc.is_file()``, which was harmless while the target was the
    always-present ``README.md``. Against ``docs/hosts.md`` it means "the
    install page for two of the three hosts does not exist" prints as "… are up
    to date." (§3, Goal).
    """

    def test_a_missing_host_doc_fails_check_and_names_it(
        self, tmp_path, monkeypatch, capsys
    ):
        missing = tmp_path / "docs" / "hosts.md"
        monkeypatch.setattr(renderer, "host_doc_path", lambda: missing)
        problem = renderer.host_doc_problem(missing)
        assert problem and "does not exist" in problem

        assert renderer.main(["--check"]) == 1
        captured = capsys.readouterr()
        assert renderer.HOST_DOC in captured.err
        assert str(missing) in captured.err
        assert "up to date" not in captured.out

    def test_a_marker_less_host_doc_fails_the_same_way(
        self, tmp_path, monkeypatch, capsys
    ):
        doc = tmp_path / "hosts.md"
        doc.write_text(
            "# Hosting multi-llm in other harnesses\n\nhand-written, no markers\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(renderer, "host_doc_path", lambda: doc)
        problem = renderer.host_doc_problem(doc)
        assert problem and "markers" in problem

        assert renderer.main(["--check"]) == 1
        captured = capsys.readouterr()
        assert renderer.HOST_DOC in captured.err
        assert renderer.HOST_DOC_BEGIN in captured.err, (
            "the failure has to be a usable instruction: both marker lines, "
            "since this script fills a block and never creates the page"
        )
        assert "up to date" not in captured.out

    def test_the_write_path_refuses_rather_than_creating_the_page(
        self, tmp_path, monkeypatch, capsys
    ):
        """A first write into a missing file must not no-op in silence.

        The renderer owns the block between the two markers and nothing else —
        the comment header above and the footer link row below are hand-written
        and have no source to be rendered from — so it reports and exits 1.
        """
        missing = tmp_path / "docs" / "hosts.md"
        monkeypatch.setattr(renderer, "host_doc_path", lambda: missing)
        monkeypatch.setattr(renderer, "OUTPUT", str(tmp_path / "reference.md"))

        assert renderer.main([]) == 1
        assert not missing.exists(), "the write path conjured the page"
        assert renderer.HOST_DOC_BEGIN in capsys.readouterr().err


class TestTheHostDocFlag:
    def test_the_deprecated_readme_spelling_still_selects_the_host_doc(
        self, capsys
    ):
        """`--host-doc`'s former name, kept as an alias (§3) and now exercised.

        A compatibility shim with no guard is the kind of thing the next
        refactor removes by accident; one assertion is the whole price of
        keeping the promise that anything still typing `--readme` works.
        """
        assert renderer.main(["--stdout", "--host-doc"]) == 0
        current = capsys.readouterr().out
        assert renderer.main(["--stdout", "--readme"]) == 0
        deprecated = capsys.readouterr().out
        assert deprecated == current == renderer.render_host_doc_section()
        assert deprecated != renderer.render(), (
            "--readme parsed, but selected the reference matrix: the alias has "
            "to reach the same dest as --host-doc, not merely be accepted"
        )


class TestThePublishedSkillOneLiner:
    """Every `$SKILL` route the host-install page publishes, asserted as a shape.

    None of this is covered by the sanity check the page prints beside them:
    that gates on the *resolved* `$SKILL`, so a glob that terminates on a
    directory, or one that matched some other marketplace's plugin, passes it.
    Measured on a real machine, four cache copies of this plugin existed and
    three of them — including the newest by mtime — did **not** contain
    `multi_llm.py`, which is what makes the terminator load-bearing rather than
    decorative (D6). Read that measurement narrowly: those three are versions
    predating the dispatcher (added in `e4e43a9`), and the fourth carries one
    only because `link-skill.sh` symlinks it at a dev checkout. On an ordinary
    machine every installed version ships a dispatcher and `ls -t` picks the
    genuinely newest, so the terminator is a guard against *stale versions*,
    not proof that the newest copy is usually unusable. The defect `ls -t`
    cannot fix at all is the one the query route addresses: a directory mtime
    does not know which install is enabled, or which project it belongs to.

    The block publishes three routes, each with its own way of rotting:

    * the **checkout** assignment, which the gate's own error message tells the
      reader to fall back to — so it has to exist above that message;
    * the **supported query**, `claude plugin list --json`, which asks Claude
      Code for its own install path instead of re-deriving the cache layout;
    * the **glob**, kept as the route for a machine with no JSON parser. `jq`
      is not part of Git for Windows, and Git Bash is a hard prerequisite on
      Windows, so a `jq`-only block would be a block Windows readers cannot run.
    """

    def section(self) -> str:
        return host_doc_body()

    def skill_routes(self) -> list:
        """Every published `SKILL=` assignment, line continuations folded in."""
        lines = self.section().splitlines()
        routes = []
        for index, line in enumerate(lines):
            if not line.startswith("SKILL="):
                continue
            block = [line]
            while block[-1].rstrip().endswith("\\"):
                index += 1
                block.append(lines[index])
            routes.append("\n".join(block))
        return routes

    def marketplace(self) -> str:
        """The marketplace the README's own `/plugin install` line names."""
        installs = set(re.findall(r"/plugin install multi-llm@(\S+)", readme_body()))
        assert len(installs) == 1, (
            f"the README installs from {sorted(installs)}; a route that pins a "
            "marketplace can only pin one"
        )
        return installs.pop()

    def one_liner(self) -> str:
        """The cache-glob route — the one with the `ls -t` picker."""
        globbing = [route for route in self.skill_routes() if "ls -t" in route]
        assert len(globbing) == 1, globbing
        return globbing[0]

    def plugin_query(self) -> str:
        """The supported route: Claude Code reporting its own install path."""
        asking = [
            route for route in self.skill_routes() if "claude plugin list" in route
        ]
        assert len(asking) == 1, (
            "the host-install page publishes exactly one `claude plugin list` "
            f"route; found {len(asking)}"
        )
        return asking[0]

    def test_it_globs_the_plugin_cache_and_terminates_at_the_dispatcher(self):
        line = self.one_liner()
        assert "plugins/cache/" in line, (
            "the one-liner no longer reads the Claude Code plugin cache, which "
            "is the only thing it was ever resolving"
        )
        match = re.search(r"ls -t (\S+)", line)
        assert match, line
        glob = match.group(1)
        assert glob.endswith("/multi_llm.py"), (
            f"the cache glob ends at {glob!r}: `ls -t | head -1` picks the "
            "newest *match*, and the newest version directory need not contain "
            "the dispatcher at all — terminating on multi_llm.py is what makes "
            "the pick a tree that actually runs"
        )
        assert "| head -1" in line
        assert line.startswith("SKILL=$(dirname "), (
            "$SKILL is a directory, so the dispatcher path the glob resolves "
            "has to be handed to dirname"
        )

    def test_the_marketplace_segment_is_the_one_the_readme_installs_from(self):
        marketplace = self.marketplace()
        assert f"plugins/cache/{marketplace}/" in self.one_liner(), (
            f"the cache glob does not pin the '{marketplace}' marketplace the "
            "README installs from. A wildcard there matches every marketplace "
            "on the machine, so a fork, a mirror or a private marketplace "
            "shipping a plugin named multi-llm wins the `ls -t` whenever its "
            "copy was written last — a different project's code at the end of "
            "$SKILL, and the sanity check below it passes"
        )

    def test_the_supported_query_asks_claude_code_for_its_own_install_path(self):
        """The point of the route: `installPath`, not a re-derived layout.

        `plugins/cache/<marketplace>/<plugin>/<version>` is Claude Code's own
        business, and the glob works only because it happens to match today's
        shape. `claude plugin list --json` reports the directory outright, so a
        route built on `installPath` survives a layout change the glob cannot —
        and reports two things the filesystem cannot be asked for at all,
        whether an install is enabled and which project it belongs to.
        """
        query = self.plugin_query()
        assert "claude plugin list --json" in query, (
            "the supported route no longer asks `claude plugin list --json`. "
            "That query, not the cache layout, is what makes this route the "
            "authoritative one"
        )
        assert ".installPath" in query, (
            "the query no longer reads the reported `installPath`; rebuilding "
            "the path out of `id` and `version` puts the cache layout back "
            "into the host-install page, which is what the glob below already does"
        )
        assert "plugins/cache" not in query, (
            "the supported route hard-codes the plugin cache layout, so it "
            "buys nothing over the glob it is published above"
        )
        assert f'"multi-llm@{self.marketplace()}"' in query, (
            f"the query does not select the 'multi-llm@{self.marketplace()}' "
            "plugin id the README's own `/plugin install` line names. Every "
            "row in that list is some plugin's install, so an unfiltered pick "
            "puts another plugin's directory at the end of $SKILL"
        )

    def test_the_query_prefers_an_enabled_install_rather_than_filtering_on_it(self):
        """`enabled` is a ranking key here, and may not become a filter.

        Measured against a real `claude plugin list --json`: the flag is
        answered for the *current directory*, not for the install, and every
        row reports `enabled: false` when the command runs somewhere Claude
        Code has no project — a bare directory, a fresh clone, `/tmp`. A
        `select(.enabled)` therefore resolves nothing on exactly the machines
        whose reader is standing outside a project, which a command the README
        says to run "from a shell" invites.
        """
        query = self.plugin_query()
        selects = re.findall(r"select\(([^)]*)\)", query)
        assert selects, f"the query filters nothing: {query}"
        for predicate in selects:
            assert "enabled" not in predicate, (
                "`enabled` is being used as a filter. It is reported per "
                "current directory and reads false for every row outside a "
                f"Claude Code project, so `select({predicate})` resolves "
                "nothing there — rank on it instead of dropping rows"
            )
        ranking = re.search(r"sort_by\(([^)]*)\)", query)
        assert ranking and "enabled" in ranking.group(1), (
            "nothing prefers an enabled install any more. Several installs of "
            "this plugin coexist on one machine — one per project — so the "
            "pick becomes whichever row the tie-break happens to reach first"
        )

    def test_the_query_probes_the_dispatcher_before_accepting_a_candidate(self):
        """The check that survives both traps at once.

        Ranking alone cannot be right: the highest-ranked row may belong to
        another project, and — measured — a version directory older than the
        dispatcher carries no `multi_llm.py` at all, so three of four cache
        copies on the machine this was written on would have failed the gate.
        Probing each candidate on disk and taking the first that answers is
        what turns a ranked list into a usable tree.
        """
        query = self.plugin_query()
        assert re.search(r'-f\s+"\$\w+/skills/multi-llm/multi_llm\.py"', query), (
            "the query accepts a candidate without testing that its tree "
            "carries multi_llm.py. Rows are ranked, not verified: a version "
            "predating the dispatcher, or an install belonging to a different "
            "project, ranks like any other"
        )

    def test_a_parser_free_route_is_published_beside_the_jq_one(self):
        """`jq` is a dependency, so it cannot be the only way through.

        Git for Windows ships no `jq`, and the README makes Git Bash a hard
        prerequisite on Windows — a block whose only `$SKILL` route needs a
        JSON parser is a block those readers cannot run at all.
        """
        section = self.section()
        assert "jq" in section, (
            "the supported query's `jq` dependency is not named anywhere in "
            "the host-install page; a reader without it gets `jq: command "
            "not found` and an empty $SKILL with no idea what to do next"
        )
        parser_free = [
            route
            for route in self.skill_routes()
            if "jq" not in route and "SKILL=$(" in route
        ]
        assert parser_free, (
            "every resolving `$SKILL` route now needs `jq`. Keep one that does "
            "not — the cache glob needs only `ls`, `head` and `dirname`, all of "
            "which Git Bash bundles"
        )

    def test_the_checkout_route_the_failure_message_names_is_published(self):
        """The gate's advice has to point at a line the block actually carries."""
        section = self.section()
        advice = re.findall(r"""set (SKILL=[^\s"']+)""", section)
        assert advice, (
            "the sanity check no longer tells a reader with an unresolved "
            "$SKILL what to set instead"
        )
        for suggestion in set(advice):
            assert any(
                route.startswith(suggestion) for route in self.skill_routes()
            ), (
                f"the failure message tells the reader to `set {suggestion}`, "
                "which is not one of the routes the block publishes"
            )


class TestTheReadmeStatesEveryInstallableHostsFacts:
    """D8 extended: the front page names the hosts; the generated page has the facts.

    `_installable_prose()` keeps the *generated* page's opening honest. The
    README still has to name every installable host (the pointer paragraph and
    the Installation heading) or a new reader never reaches `docs/hosts.md`.
    The install recipes themselves — `install-host <slug>`, shim paths, reload,
    `--yes`, runner slugs, the headless separator — live on that generated
    page, so a fourth `installable: true` host joins them from the data.

    Six comparisons against loaded YAML — one per row of D8's table — and no
    prose frozen around them.
    """

    def section(self) -> str:
        return readme_section(readme_body(), "## Installation")

    def host_doc(self) -> str:
        return host_doc_body()

    # Row 1: the host ids.
    def test_the_pointer_paragraph_and_one_heading_per_installable_host(
        self
    ):
        body = readme_body()
        intro = body[:readme_heading_re("## Installation").search(body).start()]
        pointing = [
            paragraph for paragraph in intro.split("\n\n")
            if "install-host" in paragraph
        ]
        assert len(pointing) == 1, (
            "exactly one paragraph above ## Installation promises the other "
            f"hosts; found {len(pointing)}"
        )
        for host_id in INSTALLABLE:
            assert host_id in pointing[0].lower(), (
                f"the README's opening promise never names {host_id}, which "
                "hosts/capabilities.yaml declares installable"
            )

        headings = [
            line for line in self.section().splitlines() if line.startswith("### ")
        ]
        for host_id in INSTALLABLE:
            naming = [line for line in headings if host_id in line.lower()]
            assert len(naming) == 1, (
                f"every installable host gets exactly one Installation "
                f"subsection of its own; {host_id} is named by {naming} among "
                f"the headings {headings}"
            )

    # Row 2: `install-host <slug>` for every installable host, no extras.
    def test_one_install_host_command_line_per_installable_host(self):
        slugs = re.findall(
            r'multi_llm\.py" install-host ([a-z][\w-]*)', self.host_doc()
        )
        assert set(slugs) == set(INSTALLABLE), (
            "the host-install page runs `install-host <slug>` for each "
            f"installable host: capabilities.yaml declares {sorted(INSTALLABLE)}, "
            f"the page runs {sorted(set(slugs))}"
        )

    # Row 3: the printed shim paths, from `install_paths` + `shim.installed`.
    def test_the_printed_shim_paths_are_the_preferred_ones(self):
        section = self.host_doc()
        for host_id in INSTALLABLE:
            entry = caps.entry(host_id)
            filename = os.path.basename(entry["shim"]["installed"])
            for scope in ("global", "project"):
                preferred = f"{entry['install_paths'][scope][0]}/{filename}"
                assert mentions(section, preferred), (
                    f"the host-install page does not print {host_id}'s {scope} "
                    f"shim path {preferred} — install_paths.{scope}[0] joined "
                    "with shim.installed is the file install-host writes"
                )

    def test_a_later_install_path_may_only_appear_as_a_fallback(self):
        """`global[-1]` is where an already-drifted fact came from.

        The data carries OpenCode's second global entry as the fallback for an
        unset ``$XDG_CONFIG_HOME``; the installer writes the *first*. Printing
        the fallback as the destination told every reader with the variable set
        that their shim landed somewhere it did not, and left them unable to
        find the install ``--check`` reports as present (D5).
        """
        section = self.host_doc()
        for host_id in INSTALLABLE:
            entry = caps.entry(host_id)
            filename = os.path.basename(entry["shim"]["installed"])
            for later in entry["install_paths"]["global"][1:]:
                for spelling in path_spellings(f"{later}/{filename}"):
                    for line in section.splitlines():
                        if spelling not in line:
                            continue
                        assert "fall" in line.lower(), (
                            f"{host_id}'s {spelling} is a later install_paths."
                            "global entry — the unset-$XDG_CONFIG_HOME fallback "
                            "— and the host-install page states it as the "
                            "destination install-host writes"
                        )

    @pytest.mark.parametrize("host_id", INSTALLABLE)
    def test_the_printed_project_path_is_the_file_install_host_really_writes(
        self, host_id, skill, env, project, ready
    ):
        """The one row with a second source of truth: the artifact itself.

        Reading ``install_paths`` on both sides would only prove the page and
        the YAML agree with each other; this proves both agree with what the
        installer put on disk, so a renderer that stopped honouring
        ``install_paths`` fails here too.
        """
        install(host_id, skill, env, project, scope=hi.SCOPE_PROJECT)
        written = target(host_id, skill, env, project, hi.SCOPE_PROJECT)
        relative = written.relative_to(project).as_posix()
        assert mentions(self.host_doc(), relative), (
            f"install-host --project wrote {relative}; the host-install page "
            "prints some other path"
        )

    # Row 4: "start a new <host> session" ← `reload`.
    def test_the_reload_instruction_matches_the_reload_field(self):
        wording = {
            "new-session": "start a new {host} session",
            "restart": "restart",
            "none": "no reload step",
        }
        section = self.host_doc().lower()
        for host_id in INSTALLABLE:
            value = caps.entry(host_id)["reload"]
            assert value in wording, (
                f"{host_id} declares reload: {value}, which the host-install "
                "page has no sentence for — state it where the install is, per D5"
            )
            expected = wording[value].format(host=host_id)
            assert expected in section, (
                f"{host_id} is reload: {value}, and the host-install page "
                f"never tells the reader to {expected!r}"
            )

    # Row 5: the baked-in `--yes` claim ← `ask`.
    def test_the_baked_in_yes_claim_tracks_the_ask_field(self):
        section = self.host_doc()
        for host_id in INSTALLABLE:
            assert caps.entry(host_id)["ask"] != "structured", (
                f"{host_id} now declares a structured question tool, so the "
                "page's 'the shim bakes in --yes' bullet is false: unattended "
                "is no longer the default there, and the bullet has to be "
                "rewritten rather than merely re-read"
            )
        assert "`--yes`" in section
        assert "structured question tool" in section
        for host_id in caps.host_ids():
            if caps.entry(host_id)["ask"] == "structured":
                assert host_id.replace("-", " ") in section.lower(), (
                    f"{host_id} is the host that bullet excepts, and the "
                    "host-install page never names it"
                )

    # Row 6: runner slug and headless separator ← the data, where declared.
    def test_the_runner_slugs_and_the_headless_separator_come_from_the_data(self):
        section = self.host_doc()
        for host_id in INSTALLABLE:
            entry = caps.entry(host_id)
            runner = entry["default_subagent_runner"]
            assert f"`{runner}`" in section, (
                f"{host_id}'s default_subagent_runner is {runner}; the "
                "host-install page names a different one"
            )
            headless = entry.get("headless_invocation")
            if headless:
                assert headless in section, (
                    f"{host_id} declares a headless_invocation ({headless}) "
                    "because the obvious headless form fails confusingly rather "
                    "than loudly; the host-install page has to carry it verbatim"
                )
