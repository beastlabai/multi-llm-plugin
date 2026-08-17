"""A one-off fake adapter is a *plugin*, never an edit to the shared mock (M22).

**The incident this file exists for** (§13.1). A probe needed a fake provider
that emits a multi-megabyte payload. The fake lane had no supported way to
inject one, so the obvious move was taken: the tracked
``tests/mocks/mock_llm.py`` was **overwritten in place** (1275 lines -> 40),
which broke ``tests/test_fake_lane_roster.py`` and
``tests/test_mock_adapter_roundtrip.py`` by collection error until
``git checkout --`` restored it. The lesson is a rig gap, not an accident: four
more tasks owe a fake-provider probe of exactly that shape.

**The hook.** ``MOCK_LLM_PLUGIN=<path>.py`` names a throwaway module *outside
the tree*. ``mock_llm.py`` imports it after its own tables are built and lets it
replace **exactly four seams** — ``parse_provider_args``, ``format_output``,
``get_response_content`` and ``VERSION_STRINGS`` — and nothing else. It is
honoured **only** under ``MULTI_LLM_TEST_MODE=1``, the same gate the binary
itself refuses to run without, so a stray variable in a developer's shell can
never reroute a real run. ``SkillRunner(mock_plugin=...)`` threads the path into
the child environment beside ``MOCK_LLM_SCENARIO`` / ``MOCK_LLM_PROVIDER`` /
``MOCK_LLM_CALL_LOG``.

**What is asserted here**, in the order that matters:

1. the headline probe — a scratch plugin emitting a multi-megabyte payload,
   driven through a real fake-lane orchestrator run, with
   ``tests/mocks/mock_llm.py`` proven **byte-unchanged** across the run;
2. the gate — the plugin is not even *imported* without
   ``MULTI_LLM_TEST_MODE=1`` (observed through an import-time side effect, not
   inferred);
3. the seam list — the four are honoured, a fifth name is refused **loudly**
   rather than silently ignored, and a plugin that overrides nothing at all is
   refused too (a silent no-op probe is how the incident's class of mistake
   hides);
4. ``SkillRunner`` really exports the variable, and only when asked;
5. green-stays-green — with ``MOCK_LLM_PLUGIN`` unset every mock identity emits
   **byte-identical** output, so the hook costs the existing suite nothing.

No ``utils/`` module is involved: this task cannot change product behaviour by
construction.

Plain CI: no container, no Docker, no network. Subprocesses only (the fake
binary and one orchestrator run) — the hook is a property of the child process,
so asserting it in-process alone would prove nothing about a real run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from .harness import FixtureManager, SkillRunner  # noqa: E402
from .mocks import mock_llm  # noqa: E402
from .test_e2e_mock_infrastructure import prompt_argv  # noqa: E402
from .test_mock_adapter_roundtrip import mock_identities  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
MOCK_LLM_PATH = Path(mock_llm.__file__).resolve()

#: Two megabytes and change. "Multi-megabyte" is the point: the probe that
#: caused the incident needed an output far past anything a fixture file in the
#: tree would hold, which is exactly why editing the shared mock looked like the
#: only way to get one.
PAYLOAD_BYTES = 2 * 1024 * 1024

#: A needle inside the payload, so a failure message can say whether the payload
#: arrived truncated or not at all.
PAYLOAD_NEEDLE = "M22-SCRATCH-PLUGIN-PAYLOAD"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The one volatile field any identity's wire format carries: opencode stamps
#: every event with a millisecond epoch clock (``format_opencode_output``), so
#: two renders a millisecond apart differ in bytes for a reason that has nothing
#: to do with this hook. Nothing else in any formatter reads a clock, and
#: :meth:`TestUnpluggedOutputIsByteIdentical.test_only_opencode_needs_the_clock_exemption`
#: pins that so the exemption cannot silently widen.
_EPOCH_MS = re.compile(r"\b\d{13}\b")


def normalize_clock(text: str) -> str:
    return _EPOCH_MS.sub("<ms>", text)


def write_plugin(tmp_path: Path, body: str, name: str = "scratch_adapter.py") -> Path:
    """Write a throwaway adapter under ``tmp_path`` and return its path.

    The location is the assertion: a plugin lives in a pytest ``tmp_path``,
    never in the repository, so the probe leaves no diff behind to forget to
    revert.
    """
    plugin_path = tmp_path / name
    plugin_path.write_text(body, encoding="utf-8")
    return plugin_path


def huge_payload_plugin_source() -> str:
    """A scratch adapter whose answer is a multi-megabyte suggestion list.

    The suggestion carries the field names ``review_orchestrator_base``'s own
    ``REQUIRED_FIELDS`` really validate (``title``/``desc``/``importance``/
    ``reference``/``type``), so the payload is *kept* rather than dropped as an
    invalid suggestion — the probe is about size surviving the pipeline, and a
    silently discarded suggestion would prove nothing about it.
    """
    return (
        "import json\n"
        "\n"
        f"PAYLOAD = {PAYLOAD_NEEDLE!r} + 'y' * {PAYLOAD_BYTES}\n"
        "\n"
        "\n"
        "def get_response_content(prompt):\n"
        "    return json.dumps([\n"
        "        {\n"
        "            'title': 'A multi-megabyte suggestion',\n"
        "            'desc': PAYLOAD,\n"
        "            'importance': 'MEDIUM',\n"
        "            'reference': 'Overview',\n"
        "            'type': 'clarification',\n"
        "        }\n"
        "    ])\n"
    )


def sentinel_plugin_source(sentinel: Path) -> str:
    """A scratch adapter that records the *fact of its import* on disk.

    The gate is "not honoured without ``MULTI_LLM_TEST_MODE=1``", and the
    strongest reading of that is "not even imported": a plugin's module body is
    arbitrary code. Writing the sentinel at import time makes the difference
    observable instead of inferred from an absent effect.
    """
    return (
        "import pathlib\n"
        "\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('imported\\n', encoding='utf-8')\n"
        "\n"
        "\n"
        "def get_response_content(prompt):\n"
        "    return '[]'\n"
    )


class TestScratchPluginProbe:
    """The headline case: the probe that caused the incident, done supportably."""

    def test_multi_megabyte_payload_arrives_and_the_tracked_mock_is_untouched(
        self,
        tmp_path: Path,
        fixture_manager: FixtureManager,
    ):
        """Red until the hook exists; the point of the whole task.

        A scratch plugin outside the tree makes a fake provider emit ~2 MB, a
        real ``review-plan`` run carries it end to end, and
        ``tests/mocks/mock_llm.py`` is byte-identical afterwards.
        """
        before = _digest(MOCK_LLM_PATH)

        plugin = write_plugin(tmp_path, huge_payload_plugin_source())
        runner = SkillRunner(tmp_path=tmp_path, mock_plugin=plugin)

        plan = fixture_manager.create_plan(
            "m22-huge-output",
            "# Huge Output Plan\n\n## Overview\nOne probe, one throwaway adapter.\n",
        )

        result = runner.run_orchestrator(
            "review_plan",
            plan.plan_path,
            "--models",
            "cursor-agent",
            "--skip-validation",
            timeout=120,
        )

        assert result.success, (
            f"review-plan failed with the scratch plugin installed "
            f"(exit {result.exit_code}).\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
        assert result.mock_was_invoked(), (
            "the mock was never invoked; the plugin cannot have been exercised"
        )

        review_dir = plan.output_dir / "review-plan"
        assert review_dir.exists(), f"no review-plan/ under {plan.output_dir}"

        carriers = [
            path
            for path in sorted(review_dir.iterdir())
            if path.is_file() and PAYLOAD_NEEDLE in path.read_text(
                encoding="utf-8", errors="replace"
            )
        ]
        assert carriers, (
            f"the {PAYLOAD_BYTES}-byte payload never reached any artifact in "
            f"{review_dir} (files: {[p.name for p in review_dir.iterdir()]}). "
            f"Either MOCK_LLM_PLUGIN was not threaded into the child env, or "
            f"get_response_content was not overridden."
        )

        # The per-model result file. Its exact spelling follows the resolved
        # model spec (`cursor-agent.json`, `cursor-agent_<model>.json`, ...),
        # which this task does not own, so match the model prefix rather than
        # pin a name — the assertion is that the payload survived into a
        # *persisted result*, not into a log.
        model_results = [
            path
            for path in carriers
            if path.suffix == ".json" and path.name.startswith("cursor-agent")
        ]
        assert model_results, (
            f"the payload landed in {[p.name for p in carriers]} but in no "
            f"per-model result file; a payload that only reached the raw log "
            f"never survived the parse."
        )
        payload = PAYLOAD_NEEDLE + "y" * PAYLOAD_BYTES
        for path in model_results:
            raw = path.read_text(encoding="utf-8")
            assert payload in raw, (
                f"the payload arrived truncated: {path.name} is {len(raw)} "
                f"characters and does not contain the full "
                f"{PAYLOAD_BYTES}-byte run."
            )

        assert _digest(MOCK_LLM_PATH) == before, (
            f"{MOCK_LLM_PATH} changed during the probe. The whole point of "
            f"MOCK_LLM_PLUGIN is that a one-off adapter is a throwaway file "
            f"outside the tree, never a diff against the shared mock."
        )

    def test_the_plugin_really_lives_outside_the_tree(self, tmp_path: Path):
        """A probe that wrote its adapter into ``tests/`` would prove nothing."""
        plugin = write_plugin(tmp_path, huge_payload_plugin_source())
        assert SKILL_DIR not in plugin.resolve().parents, (
            f"{plugin} is inside the skill tree; a scratch adapter must be a "
            f"throwaway file in tmp_path."
        )


class TestTheGateIsTestMode:
    """``MULTI_LLM_TEST_MODE=1`` or the plugin does not exist."""

    def test_the_plugin_is_not_imported_without_test_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        sentinel = tmp_path / "imported.txt"
        plugin = write_plugin(tmp_path, sentinel_plugin_source(sentinel))

        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        monkeypatch.delenv("MULTI_LLM_TEST_MODE", raising=False)

        assert mock_llm.load_mock_plugin() is None
        assert not sentinel.exists(), (
            "the plugin's module body ran without MULTI_LLM_TEST_MODE=1; the "
            "gate must be checked before the import, not after it."
        )

    def test_the_plugin_is_imported_under_test_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        sentinel = tmp_path / "imported.txt"
        plugin = write_plugin(tmp_path, sentinel_plugin_source(sentinel))

        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")

        module = mock_llm.load_mock_plugin()
        assert module is not None
        assert sentinel.read_text(encoding="utf-8") == "imported\n"

    def test_no_plugin_variable_is_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("MOCK_LLM_PLUGIN", raising=False)
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        assert mock_llm.load_mock_plugin() is None

    def test_the_binary_refuses_and_imports_nothing_without_test_mode(
        self, tmp_path: Path
    ):
        """The same gate observed through the real child process."""
        sentinel = tmp_path / "imported.txt"
        plugin = write_plugin(tmp_path, sentinel_plugin_source(sentinel))

        env = os.environ.copy()
        env.pop("MULTI_LLM_TEST_MODE", None)
        env["MOCK_LLM_PLUGIN"] = str(plugin)
        env["MOCK_LLM_PROVIDER"] = "cursor-agent"

        completed = subprocess.run(
            [sys.executable, str(MOCK_LLM_PATH), "a prompt"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 1
        assert "MULTI_LLM_TEST_MODE" in completed.stderr
        assert not sentinel.exists(), (
            "the fake binary imported the plugin before refusing to run; the "
            "gate must come first."
        )


class TestExactlyFourSeams:
    """Four names are honoured. A fifth is a loud refusal, not a silent no-op."""

    def test_the_seam_list_is_the_documented_four(self):
        assert tuple(mock_llm.PLUGIN_SEAMS) == (
            "parse_provider_args",
            "format_output",
            "get_response_content",
            "VERSION_STRINGS",
        )

    def test_every_seam_is_installed_over_the_mocks_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        plugin = write_plugin(
            tmp_path,
            "import argparse\n"
            "\n"
            "VERSION_STRINGS = {'cursor-agent': '9.9.9 (scratch)'}\n"
            "\n"
            "\n"
            "def parse_provider_args(provider):\n"
            "    return argparse.Namespace(), 'from the plugin'\n"
            "\n"
            "\n"
            "def format_output(provider, content):\n"
            "    return 'PLUGIN:' + content\n"
            "\n"
            "\n"
            "def get_response_content(prompt):\n"
            "    return 'scratch content'\n",
        )
        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")

        namespace: Dict[str, object] = dict(vars(mock_llm))
        installed = mock_llm.install_mock_plugin(namespace)

        assert sorted(installed) == sorted(mock_llm.PLUGIN_SEAMS)
        assert namespace["get_response_content"]("anything") == "scratch content"
        assert namespace["format_output"]("cursor-agent", "x") == "PLUGIN:x"
        assert namespace["VERSION_STRINGS"] == {"cursor-agent": "9.9.9 (scratch)"}
        assert namespace["parse_provider_args"]("cursor-agent")[1] == "from the plugin"

    def test_a_partial_plugin_overrides_only_what_it_declares(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        plugin = write_plugin(
            tmp_path, "def get_response_content(prompt):\n    return 'only this'\n"
        )
        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")

        namespace: Dict[str, object] = dict(vars(mock_llm))
        installed = mock_llm.install_mock_plugin(namespace)

        assert installed == ["get_response_content"]
        assert namespace["format_output"] is mock_llm.format_output
        assert namespace["VERSION_STRINGS"] is mock_llm.VERSION_STRINGS

    def test_a_plugin_that_overrides_nothing_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        plugin = write_plugin(tmp_path, "def helper():\n    return 1\n")
        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")

        with pytest.raises(mock_llm.MockPluginError) as excinfo:
            mock_llm.install_mock_plugin(dict(vars(mock_llm)))
        assert "parse_provider_args" in str(excinfo.value)

    def test_a_fifth_seam_is_refused_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A name the hook cannot honour must fail, not be quietly ignored."""
        plugin = write_plugin(
            tmp_path,
            "def get_response_content(prompt):\n"
            "    return '[]'\n"
            "\n"
            "\n"
            "def default_response_content(provider):\n"
            "    return 'this seam is not offered'\n",
        )
        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(plugin))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")

        with pytest.raises(mock_llm.MockPluginError) as excinfo:
            mock_llm.install_mock_plugin(dict(vars(mock_llm)))
        assert "default_response_content" in str(excinfo.value)

    def test_a_missing_plugin_file_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MOCK_LLM_PLUGIN", str(tmp_path / "nope.py"))
        monkeypatch.setenv("MULTI_LLM_TEST_MODE", "1")
        with pytest.raises(mock_llm.MockPluginError):
            mock_llm.load_mock_plugin()

    def test_the_binary_reports_a_bad_plugin_as_a_test_authoring_error(
        self, tmp_path: Path
    ):
        """Exit 9, the code this mock already reserves for a rig mistake."""
        plugin = write_plugin(tmp_path, "def helper():\n    return 1\n")

        env = os.environ.copy()
        env["MULTI_LLM_TEST_MODE"] = "1"
        env["MOCK_LLM_PLUGIN"] = str(plugin)
        env["MOCK_LLM_PROVIDER"] = "cursor-agent"

        completed = subprocess.run(
            [sys.executable, str(MOCK_LLM_PATH), "a prompt"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == mock_llm.EXIT_TEST_AUTHORING_ERROR
        assert "MOCK_LLM_PLUGIN" in completed.stderr

    def test_version_strings_is_a_seam_the_binary_honours(self, tmp_path: Path):
        """``--version`` is answered before argv parsing, so it must see the table."""
        plugin = write_plugin(
            tmp_path, "VERSION_STRINGS = {'cursor-agent': '9.9.9 (scratch)'}\n"
        )

        env = os.environ.copy()
        env["MULTI_LLM_TEST_MODE"] = "1"
        env["MOCK_LLM_PLUGIN"] = str(plugin)
        env["MOCK_LLM_PROVIDER"] = "cursor-agent"

        completed = subprocess.run(
            [sys.executable, str(MOCK_LLM_PATH), "--version"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "9.9.9 (scratch)"


class TestSkillRunnerThreadsIt:
    """The parameter is the supported way to reach the child's environment."""

    def test_mock_plugin_lands_in_the_child_env(self, tmp_path: Path):
        plugin = write_plugin(tmp_path, "def get_response_content(p):\n    return '[]'\n")
        runner = SkillRunner(tmp_path=tmp_path, mock_plugin=plugin)
        env = runner._build_env()
        assert env["MOCK_LLM_PLUGIN"] == str(plugin)
        # beside, not instead of, the variables that were already threaded
        assert env["MULTI_LLM_TEST_MODE"] == "1"
        assert env["MOCK_LLM_CALL_LOG"] == str(runner.call_log_path)

    def test_no_variable_without_the_parameter(self, tmp_path: Path):
        runner = SkillRunner(tmp_path=tmp_path)
        assert "MOCK_LLM_PLUGIN" not in runner._build_env()

    def test_extra_env_still_wins(self, tmp_path: Path):
        plugin = write_plugin(tmp_path, "def get_response_content(p):\n    return '[]'\n")
        other = write_plugin(
            tmp_path, "def get_response_content(p):\n    return '[]'\n", "other.py"
        )
        runner = SkillRunner(
            tmp_path=tmp_path,
            mock_plugin=plugin,
            extra_env={"MOCK_LLM_PLUGIN": str(other)},
        )
        assert runner._build_env()["MOCK_LLM_PLUGIN"] == str(other)


class TestUnpluggedOutputIsByteIdentical:
    """Green-stays-green: with no plugin, every identity emits what it always did."""

    #: Pinned through MOCK_LLM_FIXTURE so the comparison is about the wire
    #: format and not about which fixture the default path chose.
    CONTENT = json.dumps(
        [
            {
                "id": "byte-identical-1",
                "type": "suggestion",
                "description": "unchanged by the hook",
                "importance": "MEDIUM",
                "location": "mock_file.py:10",
            }
        ]
    )

    def _run(self, identity: str, tmp_path: Path, plugin: Path = None) -> str:
        fixture = tmp_path / "fixture.json"
        fixture.write_text(self.CONTENT, encoding="utf-8")

        env = os.environ.copy()
        env.pop("MOCK_LLM_SCENARIO", None)
        env.pop("MOCK_LLM_CALL_LOG", None)
        env.pop("MOCK_LLM_PLUGIN", None)
        env["MULTI_LLM_TEST_MODE"] = "1"
        env["MOCK_LLM_PROVIDER"] = identity
        env["MOCK_LLM_FIXTURE"] = str(fixture)
        if plugin is not None:
            env["MOCK_LLM_PLUGIN"] = str(plugin)

        completed = subprocess.run(
            [sys.executable, str(MOCK_LLM_PATH)] + prompt_argv(identity, "a prompt"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    def test_there_are_at_least_six_identities(self):
        """A parametrization that collapsed would make the next test vacuous."""
        assert len(mock_identities()) >= 6, mock_identities()

    @pytest.mark.parametrize("identity", mock_identities())
    def test_identity_output_is_unchanged_without_a_plugin(
        self, identity: str, tmp_path: Path
    ):
        """The child's bytes equal what the module's own untouched tables render.

        Not a tautology: the fake binary runs the hook (it is installed inside
        ``main()``), while ``mock_llm.format_output`` here is the *unpatched*
        module this pytest process imported. A hook that leaked — installed from
        a stray environment variable, or mutating anything on the default path —
        shows up as a diff.
        """
        stdout = self._run(identity, tmp_path)
        expected = mock_llm.format_output(identity, self.CONTENT) + "\n"
        assert normalize_clock(stdout) == normalize_clock(expected), (
            f"the fake binary's bytes for {identity!r} no longer match the "
            f"module's own untouched tables; MOCK_LLM_PLUGIN was unset, so the "
            f"hook must have changed the default path."
        )

    @pytest.mark.parametrize("identity", mock_identities())
    def test_only_opencode_needs_the_clock_exemption(
        self, identity: str, tmp_path: Path
    ):
        """Every other identity is compared byte-for-byte, with no normalizer."""
        stdout = self._run(identity, tmp_path)
        if identity == "opencode":
            assert normalize_clock(stdout) != stdout, (
                "opencode stopped emitting a millisecond clock; drop it from "
                "the exemption instead of keeping a normalizer that no longer "
                "normalizes anything."
            )
            return
        assert normalize_clock(stdout) == stdout, (
            f"{identity!r} now emits a 13-digit epoch-ms field, so the test "
            f"above silently stopped comparing it byte-for-byte. Either pin the "
            f"clock in the formatter or widen the exemption deliberately."
        )
        assert stdout == mock_llm.format_output(identity, self.CONTENT) + "\n"

    @pytest.mark.parametrize("identity", mock_identities())
    def test_the_comparison_has_teeth(self, identity: str, tmp_path: Path):
        """The same run WITH a plugin must differ, or the check above is vacuous."""
        plugin = write_plugin(
            tmp_path,
            "def format_output(provider, content):\n"
            "    return 'SCRATCH-ADAPTER:' + provider\n",
        )
        assert self._run(identity, tmp_path, plugin).strip() == (
            f"SCRATCH-ADAPTER:{identity}"
        )


def test_no_utils_module_is_involved():
    """M22 is test-rig only: the hook lives entirely under ``tests/``.

    Named as an assertion because "no product behaviour changed" is the
    property that makes this task safe to land first, ahead of the four probes
    that owe it.
    """
    hook_files: List[Path] = [
        MOCK_LLM_PATH,
        SKILL_DIR / "tests" / "harness" / "skill_runner.py",
    ]
    for path in hook_files:
        assert (SKILL_DIR / "tests") in path.resolve().parents, path
