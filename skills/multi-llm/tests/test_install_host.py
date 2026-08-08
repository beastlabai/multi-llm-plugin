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
  path with a stubbed probe;
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
  skipped, on POSIX too, because a skipped seam is an unverified one.
"""

import ast
import os
import shutil
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from utils import host_capabilities as caps  # noqa: E402
from utils import host_install as hi  # noqa: E402

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
    """Row 5's prerequisites, stubbed: uv resolves and the runner CLI exists.

    Neither is a property of the machine running the tests — CI has no codex or
    opencode binary — and row 5 has its own dedicated tests below, so every other
    test stubs it rather than asserting the state of the test runner.
    """
    monkeypatch.setattr(hi, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(hi, "runner_available", lambda provider: True)


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
        runner = caps.entry(self.HOST, str(skill))["default_subagent_runner"]
        assert runner in row.detail

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
        assert 'newline="\\n"' in source, (
            "identical bytes on every platform; idempotency and the body hash both "
            "depend on it"
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

class TestPackagingAndDocs:
    def test_exactly_one_auto_discoverable_skill_manifest_still_ships(self):
        manifests = [
            path.relative_to(SKILL_ROOT).as_posix()
            for path in SKILL_ROOT.rglob("SKILL.md")
            if ".venv" not in path.parts and "tests" not in path.parts
        ]
        assert manifests == ["SKILL.md"], manifests

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

    def test_the_readme_section_is_byte_identical_to_a_fresh_render(self):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        readme = renderer.readme_path()
        assert readme.is_file(), readme
        body = readme.read_text(encoding="utf-8")
        assert renderer.README_BEGIN in body and renderer.README_END in body
        assert renderer.readme_with_section(body) == body, (
            "the README's host-install section is generated from "
            "hosts/capabilities.yaml — re-run hosts/render_capabilities.py and "
            "commit the result; a hand-maintained copy is a CI failure"
        )

    def test_the_readme_section_carries_the_quickstarts_and_the_matrix(self):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        section = renderer.render_readme_section()
        assert "## Hosting multi-llm in other harnesses" in section
        assert "### After a plugin update" in section
        assert "install-host --check --all" in section
        for host_id in caps.host_ids():
            assert f"`{host_id}`" in section
        for host_id in INSTALLABLE:
            assert f"install-host {host_id} --check" in section

    def test_the_check_guard_covers_both_generated_files(self, monkeypatch,
                                                         capsys):
        sys.path.insert(0, str(SKILL_ROOT / "hosts"))
        import render_capabilities as renderer

        assert renderer.main(["--check"]) == 0
        monkeypatch.setattr(
            renderer, "render_readme_section", lambda base=None: "drifted\n"
        )
        assert renderer.main(["--check"]) == 1
        assert "README.md" in capsys.readouterr().err
