#!/usr/bin/env python3
"""`--exec-subagents` end to end: one invocation, through to the report (D3b).

This is the read leg the suite could not cover before: a real
``review_plan_orchestrator.py`` subprocess fans out to mock providers, writes
``validation_tasks.json``, and then — instead of exiting at
``[VALIDATION_PENDING]`` for a host to drive a second turn — dispatches the batch
through the executor, merges it and finishes the phase.

The assertions that matter are comparative, not absolute:

* the **self-contained** run's artifacts equal the **host-driven** run's, where
  the host-driven leg is played by ``SkillRunner.simulate_subagents`` exactly as
  every other e2e test plays it (D3a: whichever invoker ran, reaggregation sees
  byte-identical files);
* with the flag **off**, stdout and every artifact are byte-identical to what
  shipped before the executor existed — the executor is strictly additive until a
  host asks for it.

Usage:
    uv run -- pytest tests/test_e2e_exec_subagents.py -v
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import FixtureManager, SkillRunner  # noqa: E402

PLAN_BODY = """# Executor Test Plan

## Overview
A plan whose review must reach a validated report in one invocation.

## Step 1
Upload the file.

## Step 2
Write the config.
"""


def _driver_config(tmp_path: Path, skill_dir: Path, runner: str) -> Path:
    """A full base config re-assertion plus a `driver.subagent_runner` override.

    The harness normally pins ``MULTI_LLM_PROVIDERS_CONFIG`` at the skill's own
    base ``providers.yaml`` so a dogfooding checkout's
    ``.multi-llm/providers.yaml`` cannot reroute mock model specs. Overriding that
    var with a *driver-only* file would give that project layer back, so this
    writes base verbatim with one key changed — same isolation, one difference.
    """
    base = yaml.safe_load((skill_dir / "providers.yaml").read_text(encoding="utf-8"))
    base.setdefault("driver", {})["subagent_runner"] = runner
    path = tmp_path / "driver-providers.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return path


def _ask_mock_codex(runner: SkillRunner, prompt: str):
    """Run one materialized prompt through the mock `codex` and recover its answer.

    This is the *callee* half of D3a played literally: the agent prints a JSON
    object on stdout, and the caller (the invoker) is the one that persists it.
    Recovery goes through the adapter's ``parse_output`` first, exactly as the
    executor does, because the mock emits codex's NDJSON event stream and not
    plain text.
    """
    import subprocess

    from utils.provider_registry import get_provider
    from utils.subagent_executor import extract_result

    completed = subprocess.run(
        [sys.executable, str(runner.mock_llm_path), "exec", "-s", "read-only",
         "--json", "--model", "gpt-5.5"],
        input=prompt, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "MULTI_LLM_TEST_MODE": "1",
             "MOCK_LLM_PROVIDER": "codex", "MOCK_LLM_ECHO": "validation"},
    )
    return extract_result(
        get_provider("codex"), completed.stdout, completed.stderr,
        prefer_arrays=False,
    )


def _normalize(text: str) -> str:
    """Strip run-varying noise (paths, timings, ids) from stdout for comparison."""
    text = re.sub(r"/tmp/[^\s\"']+", "<path>", text)
    text = re.sub(r"\d+\.\d+s", "<dur>", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+", "<ts>", text)
    return text


@pytest.fixture
def exec_runner(tmp_path, mock_provider):
    """A SkillRunner whose driver dispatches subagents to the mock `codex`."""
    runner = SkillRunner(tmp_path, extra_env=mock_provider.get_env())
    config = _driver_config(tmp_path, runner.skill_dir, "codex:gpt-5.5")
    runner.extra_env.update({
        "MULTI_LLM_PROVIDERS_CONFIG": str(config),
        # Safe to set for the whole run: the echo returns None for a prompt with
        # no group hashes, so the model fan-out still gets its scenario answer and
        # only the materialized validation prompts are echoed.
        "MOCK_LLM_ECHO": "validation",
    })
    return runner


class TestSelfContainedRun:
    """D3b: the same process continues past the marker and finishes the phase."""

    def test_one_invocation_reaches_a_validated_report(
        self, exec_runner, fixture_manager, mock_provider, assertions
    ):
        mock_provider.set_scenario("happy_path")
        exec_runner.extra_env.update(mock_provider.get_env())
        exec_runner.extra_env["MOCK_LLM_ECHO"] = "validation"
        plan = fixture_manager.create_plan("exec-plan", PLAN_BODY)

        result = exec_runner.run_orchestrator(
            "review_plan", plan.plan_path,
            "--models", "cursor-agent",
            "--exec-subagents",
            timeout=120,
        )
        assert result.exit_code == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert result.mock_was_invoked()

        phase_dir = plan.output_dir / "review-plan"
        # The marker is still printed — informational on this path — but the run
        # did NOT stop there.
        assert "[VALIDATION_PENDING]" in result.stdout or \
               "[VALIDATION_BATCHES_PENDING]" in result.stdout
        assert "[exec-subagents]" in result.stdout

        # The batch really was dispatched through the provider adapters...
        batch_files = sorted(phase_dir.glob("validation_batch_*.json"))
        assert batch_files, list(phase_dir.iterdir())
        # ...merged...
        validation = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        groups = validation["groups"] if isinstance(validation, dict) else validation
        assert groups and all(g["status"] == "valid" for g in groups)
        # F3: the mock validator's own self-report is `"model": "mock"`
        # (`echo_subagent_answer`) — a free-hand string exactly like the one a
        # real subagent wrote. `--exec-subagents` dispatched this batch
        # itself, so `metadata.model`/`metadata.runner` must name the
        # executor's own dispatch record (the driver override below,
        # "codex:gpt-5.5") instead of trusting the self-report.
        assert validation["metadata"]["model"] == "codex:gpt-5.5"
        assert validation["metadata"]["runner"] == "codex:gpt-5.5"
        # ...and the phase completed, report included.
        assert list(phase_dir.glob("*report*.md")) or (phase_dir / "report.html").exists()
        state = json.loads((plan.output_dir / "state.json").read_text(encoding="utf-8"))
        assert "review-plan" in json.dumps(state)

        # The executor's own bookkeeping is present and every task succeeded.
        report = json.loads(
            (phase_dir / "subagent_batch_report.json").read_text(encoding="utf-8")
        )
        assert report["invoker"] == "executor"
        assert report["tasks"] and all(t["status"] == "ok" for t in report["tasks"])
        assert all(t["profile"] == "agent-readonly" for t in report["tasks"])
        assert report["runner_versions"] == {"codex:gpt-5.5": "0.160.1"}

        # ...and the same counts are mirrored into .status.json, so a detached or
        # CI run is diagnosable without any stdout at all (D4a).
        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        mirrored = status["progress"]["subagents"]
        assert mirrored["ok"] == mirrored["total"] == len(report["tasks"])
        assert mirrored["failed_task_ids"] == []
        assert mirrored["runner_versions"] == {"codex:gpt-5.5": "0.160.1"}

        # No follow-up is owed: the run is finished when the process exits.
        marker = [
            line for line in result.stdout.splitlines()
            if line.startswith("MULTI_LLM_RESULT:")
        ]
        if marker:  # only the dispatcher route emits it
            assert json.loads(marker[-1].split(":", 1)[1])["next"] is None

    def test_matches_the_host_driven_run_artifact_for_artifact(
        self, tmp_path, fixture_manager, mock_provider, assertions
    ):
        """D3a: whichever invoker ran, reaggregation sees the same files.

        The host-driven leg is the one every other e2e test uses — fan out, exit
        at the marker, ``simulate_subagents`` writes each ``output_path``,
        ``--reaggregate`` finishes — so this compares the new path against the
        established one rather than against a hand-written expectation.
        """
        mock_provider.set_scenario("happy_path")

        # --- host-driven (today's behavior) ---
        native_runner = SkillRunner(tmp_path / "native", extra_env=mock_provider.get_env())
        native_fixtures = FixtureManager(tmp_path / "native")
        native_plan = native_fixtures.create_plan("plan", PLAN_BODY)
        first = native_runner.run_orchestrator(
            "review_plan", native_plan.plan_path, "--models", "cursor-agent",
            timeout=120,
        )
        assert first.exit_code == 0, first.stderr
        native_phase = native_plan.output_dir / "review-plan"
        # The "subagent" is the SAME mock binary the executor dispatches, driven
        # from the SAME materialized prompt — so any difference in the persisted
        # file is a difference in the invoker, which is exactly what this test is
        # for. (Left to its default builder the harness would answer as a
        # different agent, and `metadata.model` alone would differ.)
        written = SkillRunner.simulate_subagents(
            native_phase / "validation_tasks.json",
            result_for=lambda task: _ask_mock_codex(native_runner, task["prompt"]),
        )
        assert written
        second = native_runner.run_orchestrator(
            "review_plan", native_plan.plan_path, "--models", "cursor-agent",
            "--reaggregate", timeout=120,
        )
        assert second.exit_code == 0, second.stderr

        # --- self-contained (--exec-subagents) ---
        exec_runner = SkillRunner(tmp_path / "exec", extra_env=mock_provider.get_env())
        exec_runner.extra_env.update({
            "MULTI_LLM_PROVIDERS_CONFIG": str(
                _driver_config(tmp_path, exec_runner.skill_dir, "codex:gpt-5.5")
            ),
            "MOCK_LLM_ECHO": "validation",
        })
        exec_fixtures = FixtureManager(tmp_path / "exec")
        exec_plan = exec_fixtures.create_plan("plan", PLAN_BODY)
        third = exec_runner.run_orchestrator(
            "review_plan", exec_plan.plan_path, "--models", "cursor-agent",
            "--exec-subagents", timeout=120,
        )
        assert third.exit_code == 0, f"{third.stdout}\n{third.stderr}"
        exec_phase = exec_plan.output_dir / "review-plan"

        # The validation batch each invoker persisted is byte-identical, which is
        # what makes everything downstream of it identical too.
        assert (exec_phase / "validation_batch_0.json").read_bytes() == (
            native_phase / "validation_batch_0.json"
        ).read_bytes()

        # And the merged validation the report is built from agrees.
        def _groups(path):
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["groups"] if isinstance(data, dict) else data

        exec_groups = _groups(exec_phase / "validation.json")
        native_groups = _groups(native_phase / "validation.json")
        assert [
            (g["group_index"], g["status"], g.get("group_id")) for g in exec_groups
        ] == [
            (g["group_index"], g["status"], g.get("group_id")) for g in native_groups
        ]


class TestFlagOffIsUnchanged:
    """With the flag off nothing about a run changes."""

    def test_stdout_and_artifacts_are_identical_without_the_flag(
        self, tmp_path, fixture_manager, mock_provider
    ):
        """Two runs, one of which merely *could* have used the executor."""
        mock_provider.set_scenario("happy_path")
        outputs = {}
        for label in ("plain", "with-driver-config"):
            runner = SkillRunner(tmp_path / label, extra_env=mock_provider.get_env())
            if label == "with-driver-config":
                # A configured runner must change nothing until the flag is passed.
                runner.extra_env["MULTI_LLM_PROVIDERS_CONFIG"] = str(
                    _driver_config(tmp_path, runner.skill_dir, "codex:gpt-5.5")
                )
                runner.extra_env["MOCK_LLM_ECHO"] = "validation"
            fixtures = FixtureManager(tmp_path / label)
            plan = fixtures.create_plan("plan", PLAN_BODY)
            result = runner.run_orchestrator(
                "review_plan", plan.plan_path, "--models", "cursor-agent",
                timeout=120,
            )
            assert result.exit_code == 0, result.stderr
            phase_dir = plan.output_dir / "review-plan"
            outputs[label] = {
                "stdout": _normalize(result.stdout),
                "files": sorted(p.name for p in phase_dir.iterdir()),
                "tasks": json.loads(
                    (phase_dir / "validation_tasks.json").read_text(encoding="utf-8")
                ),
            }

        assert outputs["plain"]["stdout"] == outputs["with-driver-config"]["stdout"]
        assert outputs["plain"]["files"] == outputs["with-driver-config"]["files"]
        # No executor artifacts anywhere.
        assert "subagent_batch_report.json" not in outputs["plain"]["files"]
        assert "failed" not in outputs["plain"]["files"]
        # The marker path still owes a follow-up command.
        assert "_PENDING]" in outputs["plain"]["stdout"]
        assert "--reaggregate" in outputs["plain"]["stdout"]

    def test_no_validation_batch_is_written_without_the_flag(
        self, tmp_path, fixture_manager, mock_provider
    ):
        mock_provider.set_scenario("happy_path")
        runner = SkillRunner(tmp_path, extra_env=mock_provider.get_env())
        runner.extra_env["MULTI_LLM_PROVIDERS_CONFIG"] = str(
            _driver_config(tmp_path, runner.skill_dir, "codex:gpt-5.5")
        )
        plan = fixture_manager.create_plan("no-flag", PLAN_BODY)
        result = runner.run_orchestrator(
            "review_plan", plan.plan_path, "--models", "cursor-agent", timeout=120,
        )
        assert result.exit_code == 0
        phase_dir = plan.output_dir / "review-plan"
        assert list(phase_dir.glob("validation_batch_*.json")) == []
        assert not (phase_dir / "validation.json").exists()


class TestPartialBatchStopsThePhase:
    """D3i/D3k: a failed task blocks reaggregation and exits 3, even under --yes."""

    def test_exit_3_and_no_report(
        self, tmp_path, fixture_manager, mock_provider
    ):
        mock_provider.set_scenario("happy_path")
        runner = SkillRunner(tmp_path, extra_env=mock_provider.get_env())
        runner.extra_env.update({
            "MULTI_LLM_PROVIDERS_CONFIG": str(
                _driver_config(tmp_path, runner.skill_dir, "codex:gpt-5.5")
            ),
            # No echo: the mock answers the validation prompt with the default
            # review-shaped fixture, which is not a validation document.
        })
        plan = fixture_manager.create_plan("partial", PLAN_BODY)
        result = runner.run_orchestrator(
            "review_plan", plan.plan_path, "--models", "cursor-agent",
            "--exec-subagents", "--yes", timeout=120,
        )
        assert result.exit_code == 3, f"{result.stdout}\n{result.stderr}"
        combined = result.stdout + result.stderr
        assert "refuses to reaggregate" in combined
        assert "Under --yes this is still exit 3" in combined

        phase_dir = plan.output_dir / "review-plan"
        assert not (phase_dir / "validation.json").exists()
        # Diagnostics survive the process, and nothing was salvaged instead.
        report = json.loads(
            (phase_dir / "subagent_batch_report.json").read_text(encoding="utf-8")
        )
        assert all(t["status"] != "ok" for t in report["tasks"])
        assert all(t["rung"] in (1, 2, 3, 4, 5, 6, 7) for t in report["tasks"])
        assert (phase_dir / "failed").is_dir()
        assert list((phase_dir / "failed").glob("*.json"))

        # The same statuses are mirrored, so a detached run is diagnosable from
        # `.status.json` alone.
        status = json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))
        mirrored = status["progress"]["subagents"]
        assert mirrored["ok"] == 0
        assert mirrored["failed_task_ids"]
        assert set(mirrored["counts"]) == {
            t["status"] for t in report["tasks"]
        }
