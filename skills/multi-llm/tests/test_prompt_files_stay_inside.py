"""The kept prompt files never leave the repository (fix round 17).

The Fable review of round 16 (noise #1): a jailed repo-wide codex batch can
write the phase dir, ``plans/<stem>/<phase>`` lying inside the root grant, and
the plugin that keeps each dispatched prompt under ``{phase}/prompts/`` is not
jailed. Round 16 checked ``prompts`` once, by path: a phase dir replaced by a
link was resolved and accepted, and a ``prompts`` swapped for a link between
the check and the write was followed, so the plugin wrote a prompt file
outside the repository. On the native leg the host's own subagents can write
the phase dir too, and ``util write-prompt`` wrote by path through a link at
the file's own name, overwriting the file it named.

Now the phase dir is pinned before any batch runs and every file is written
relative to descriptors opened without following a link
(``utils/prompt_files.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from tests.test_write_prompt import (  # noqa: E402
    _implement_batches, _materialize, _suggestion_batches, plan_dir,  # noqa: F401
)
from utils import subagent_executor as executor  # noqa: E402
from utils import write_prompt  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the attacks need symlinks; Windows runs no jail"
)

APPLIED_S0 = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
              "summary": "added the step"}


def _swap_for_link(path: Path, target: Path) -> Path:
    """What a jailed batch does: move the directory aside, link its name."""
    moved = path.with_name(path.name + ".moved")
    os.rename(path, moved)
    os.symlink(target, path, target_is_directory=True)
    return moved


# ===========================================================================
# The executor leg: WriteBatchScheduler._keep_dispatched_prompt
# ===========================================================================

class TestTheExecutorLeg:

    def test_a_phase_dir_replaced_by_a_link_after_the_pin_is_not_followed(
        self, env, tmp_path
    ):
        """(a) The batch replaces ``plans/<stem>/<phase>`` itself with a link
        out of the repository once the run started."""
        outside = tmp_path / "outside"
        outside.mkdir()
        lines = []
        scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
        _swap_for_link(env.phase_dir, outside)

        scheduler._keep_dispatched_prompt("apply-batch-0", 1, "PROMPT", "PROMPT")

        assert list(outside.iterdir()) == []
        assert any("could not keep apply-batch-0's dispatched prompt" in line
                   and "symlink" in line for line in lines), lines

    def test_a_phase_dir_that_resolves_outside_the_repository_is_never_pinned(
        self, env, tmp_path
    ):
        """A link left by an earlier run's batch: the phase dir already leads
        out of the repository when this run builds its scheduler."""
        outside = tmp_path / "outside"
        outside.mkdir()
        _swap_for_link(env.phase_dir, outside)
        lines = []
        scheduler = env.scheduler([env.apply_task(0)], log=lines.append)

        scheduler._keep_dispatched_prompt("apply-batch-0", 1, "PROMPT", "PROMPT")

        assert list(outside.iterdir()) == []
        assert any("not inside the repository" in line for line in lines), lines

    def test_a_phase_dir_replaced_by_another_folder_is_refused(self, env):
        """The pinned identity: a real folder moved into the phase dir's
        place is not the folder the run started with."""
        lines = []
        scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
        os.rename(env.phase_dir, env.phase_dir.with_name("aside"))
        env.phase_dir.mkdir()

        scheduler._keep_dispatched_prompt("apply-batch-0", 1, "PROMPT", "PROMPT")

        assert not (env.phase_dir / "prompts").exists()
        assert any("no longer the folder this run started with" in line
                   for line in lines), lines

    def test_prompts_swapped_for_a_link_while_the_file_is_written_is_not_followed(
        self, env, tmp_path, monkeypatch
    ):
        """(b) The TOCTOU: ``prompts`` becomes a link between the check and
        the write (here, the moment the temporary file is created)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (env.phase_dir / "prompts").mkdir()
        real_open = os.open
        swapped = []

        def swap_then_open(path, *args, **kwargs):
            if os.path.basename(str(path)).startswith(".prompt-") and not swapped:
                swapped.append(_swap_for_link(env.phase_dir / "prompts", outside))
            return real_open(path, *args, **kwargs)

        lines = []
        scheduler = env.scheduler([env.apply_task(0)], log=lines.append)
        monkeypatch.setattr(os, "open", swap_then_open)
        scheduler._keep_dispatched_prompt("apply-batch-0", 1, "PROMPT", "PROMPT")
        monkeypatch.undo()

        assert swapped, "the swap ran"
        assert list(outside.iterdir()) == []
        # It landed in the folder that was opened, wherever that is now.
        assert (swapped[0] / "apply-batch-0.prompt.txt").read_text() == "PROMPT"

    def test_a_link_or_hard_link_at_the_file_name_is_replaced_never_written_through(
        self, env, tmp_path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "victim"
        victim.write_text("secret", encoding="utf-8")
        prompts = env.phase_dir / "prompts"
        prompts.mkdir()
        os.symlink(victim, prompts / "apply-batch-0.prompt.txt")
        os.link(victim, prompts / "apply-batch-1.prompt.txt")
        scheduler = env.scheduler([env.apply_task(0)])

        scheduler._keep_dispatched_prompt("apply-batch-0", 1, "P0", "P0")
        scheduler._keep_dispatched_prompt("apply-batch-1", 1, "P1", "P1")

        assert victim.read_text() == "secret"
        assert not (prompts / "apply-batch-0.prompt.txt").is_symlink()
        assert (prompts / "apply-batch-0.prompt.txt").read_text() == "P0"
        assert (prompts / "apply-batch-1.prompt.txt").read_text() == "P1"

    def test_a_whole_run_still_keeps_its_prompt(self, env):
        env.edits([_patch_edit(env.plan)], result=APPLIED_S0)
        result = env.scheduler([env.apply_task(0)]).run()
        assert result.outcomes[0].status == executor.STATUS_OK
        assert (env.phase_dir / "prompts" / "apply-batch-0.prompt.txt").is_file()


# ===========================================================================
# The native leg: util write-prompt
# ===========================================================================

def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


class TestTheNativeLeg:

    def _phase(self, plan_dir):
        plan, out = plan_dir
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        return phase_dir

    def _main(self, phase_dir, task_id="apply-batch-0"):
        return write_prompt.main(["--phase-dir", str(phase_dir), "--task-id", task_id])

    def test_a_link_at_the_prompt_file_name_is_replaced_not_written_through(
        self, plan_dir, tmp_path, capsys
    ):
        phase_dir = self._phase(plan_dir)
        victim = tmp_path / "victim.txt"
        victim.write_text("the user's own file\n", encoding="utf-8")
        (phase_dir / "prompts").mkdir()
        target = phase_dir / "prompts" / "apply-batch-0.prompt.txt"
        os.symlink(victim, target)

        assert self._main(phase_dir) == 0

        assert victim.read_text() == "the user's own file\n"
        assert not target.is_symlink()
        assert target.read_text(encoding="utf-8") == write_prompt.render_write_prompt(
            phase_dir, "apply-batch-0")[0]

    def test_a_prompts_link_is_refused_and_nothing_lands_outside(
        self, plan_dir, tmp_path, capsys
    ):
        phase_dir = self._phase(plan_dir)
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, phase_dir / "prompts", target_is_directory=True)

        assert self._main(phase_dir) == 1

        assert list(outside.iterdir()) == []
        err = capsys.readouterr().err
        assert "cannot write" in err and "is a symlink" in err

    def test_a_phase_dir_link_is_refused(self, plan_dir, tmp_path, capsys):
        """The phase dir itself is a link: its task file is read through it,
        but nothing is written through it."""
        phase_dir = self._phase(plan_dir)
        outside = tmp_path / "outside"
        os.rename(phase_dir, outside)
        os.symlink(outside, phase_dir, target_is_directory=True)

        assert self._main(phase_dir) == 1

        assert not (outside / "prompts").exists()
        assert "is a symlink" in capsys.readouterr().err

    def test_a_folder_swapped_between_the_read_and_the_write_is_refused(
        self, plan_dir, tmp_path, monkeypatch, capsys
    ):
        """The plan's output folder becomes a link to a folder elsewhere
        holding a phase dir of the same name, after the task file was read."""
        plan, out = plan_dir
        _git_init(plan.parent.parent)
        phase_dir = self._phase(plan_dir)
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "apply-suggestions").mkdir(parents=True)
        real_render = write_prompt.render_write_prompt

        def render_then_swap(*args, **kwargs):
            rendered = real_render(*args, **kwargs)
            _swap_for_link(out, elsewhere)
            return rendered

        monkeypatch.setattr(write_prompt, "render_write_prompt", render_then_swap)

        assert self._main(phase_dir) == 1

        assert not (elsewhere / "apply-suggestions" / "prompts").exists()
        assert "cannot write" in capsys.readouterr().err

    def test_a_check_tree_record_is_never_written_through_a_link(self, plan_dir, tmp_path):
        plan, out = plan_dir
        _git_init(plan.parent.parent)
        phase_dir, _tasks = _materialize(
            plan, out, "implement", _implement_batches(), "implement")
        victim = tmp_path / "victim.txt"
        victim.write_text("keep me", encoding="utf-8")
        (phase_dir / "prompts").mkdir()
        # The temporary name round 16's record wrote through.
        os.symlink(victim, phase_dir / "prompts" / "T001.tree.json.tmp")
        os.symlink(victim, phase_dir / "prompts" / "T001.tree.json")

        record = write_prompt.record_check_tree(phase_dir, "T001")

        assert victim.read_text() == "keep me"
        assert record is not None and not record.is_symlink()
        assert json.loads(record.read_text())["task_id"] == "T001"

    def test_discarding_a_check_tree_record_never_reaches_through_a_link(
        self, plan_dir, tmp_path
    ):
        phase_dir = self._phase(plan_dir)
        outside = tmp_path / "outside"
        outside.mkdir()
        kept = outside / "T001.tree.json"
        kept.write_text("{}", encoding="utf-8")
        os.symlink(outside, phase_dir / "prompts", target_is_directory=True)

        write_prompt.discard_check_tree(phase_dir, "T001")

        assert kept.exists()
