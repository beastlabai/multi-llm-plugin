"""M1 (§3.1 / §5.4.1): cursor-agent's read-only argv, its flag gate, and its residue.

Three separate contracts live here, all of them cursor-only:

1. **The argv.** ``cursor-agent --print --trust --mode ask --output-format json
   --model <m> <prompt>`` — ``-f`` (``--force``/``--yolo``, "Run Everything") is
   gone, and the two flags that replace it are inseparable (atomic group **A5**).
2. **The gate**, which since M14 is the *shared* one in
   ``utils/providers/base.py`` rather than a copy in the adapter. cursor declares
   ``version_cmd = ("cursor-agent", "--version")`` (M14's measurement) but no
   ``min_version``: no floor has been measured for these two flags, so the
   binary's own ``--help`` is what confirms them, once per process, before every
   dispatch. An unconfirmed token refuses the slot as ``version-too-low`` naming
   the flags — never a degraded argv, because the only degradation available
   *is* the write bypass this change removes. The tests below therefore patch
   ``base.read_help_tokens``, the one implementation, and the load-bearing
   posture they assert (unconfirmed ⇒ refuse; no runnable binary ⇒ no refusal)
   now lives in ``LLMProvider.missing_flags``.
3. **The residue.** ``--trust`` writes a durable marker and ``--model`` rewrites
   the user's default model, both in the user's own profile. Neither is ours to
   keep, so both are snapshotted before the spawn and restored after.

**Recorded evidence — the two §3.1 ship gates, measured 2026-08-16 with three
paid cursor calls. They are recorded here, not re-run: §14.5 tiers a live cursor
call T4 and forbids it as a CI gate.**

*Q12 — does ``--mode ask`` keep its read tools?* **Yes, confirmed at tool level.**
Under ``--print --trust --mode ask --output-format json``, cursor's own agent
transcript (``~/.cursor/projects/<slug>/agent-transcripts/*/*.jsonl``) carried
``{"type":"tool_use","name":"Grep"}`` and ``{"type":"tool_use","name":"Read"}``,
and the run located a file by an unguessable token it was never given. So ask
mode covers **every** cursor Lane A phase — ``review-code``/``review-tasks``
included — and neither a per-phase narrowing nor the ``--mode plan`` fallback is
needed.

*Q11 — does ``--trust`` persist?* **Yes.** The run left
``<data>/projects/<slug>/.workspace-trusted`` (206 B,
``{"trustedAt":…,"workspacePath":…,"trustMethod":"cli-flag"}``) behind; seven
such markers, never garbage-collected, already existed on the probe machine.
A marker also trusts that workspace for the user's own later *interactive*
cursor-agent sessions, which run with full tool permissions — which is why
:class:`CursorResidueSnapshot` removes the ones a run creates.

*Q11 bonus, not in the plan and larger than the marker:* ``cursor-agent --print
--model <m>`` **rewrites the user's global default model** — ``cli-config.json``'s
``model`` / ``selectedModel`` moved to the model the run passed and
``modelSelectionHistory`` was reordered. A fan-out of several cursor rows would
otherwise leave the user's interactive CLI on whichever row finished last.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers import base as base_module
from utils.providers import cursor_agent as cursor_agent_module
from utils.providers.base import PROFILE_GENERATE
from utils.providers.cursor_agent import (
    CURSOR_REQUIRED_FLAGS,
    VERSION_TOO_LOW,
    CursorAgentProvider,
    CursorFlagsUnconfirmed,
)

#: The one shipping argv. Any other shape is a regression, in either direction.
EXPECTED_ARGV = [
    "cursor-agent",
    "--print",
    "--trust",
    "--mode",
    "ask",
    "--output-format",
    "json",
    "--model",
    "gpt-5",
    "the prompt",
]

CONFIRMED = frozenset({"--print", "--output-format", "--model", "--mode", "--trust"})

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"

#: Something ``shutil.which`` can plausibly have returned: the gate only reads a
#: binary it could actually spawn, so a made-up path would skip it entirely.
_RUNNABLE = sys.executable


@pytest.fixture
def provider():
    return CursorAgentProvider()


@pytest.fixture(autouse=True)
def _clean_module_state():
    """No cached help read and no armed guard leaks between tests."""
    base_module.reset_flag_gate_cache()
    cursor_agent_module.reset_side_effect_guard()
    yield
    base_module.reset_flag_gate_cache()
    cursor_agent_module.reset_side_effect_guard()


@pytest.fixture
def confirmed(monkeypatch):
    """Both tokens confirmed, with no binary and no spawn."""
    monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: _RUNNABLE)
    monkeypatch.setattr(base_module, "read_help_tokens", lambda _b, **_kw: CONFIRMED)


def _write_help_binary(
    path: Path, help_text: str, *, exit_code: int = 0, counter: Path | None = None
) -> Path:
    """A fake CLI that answers ``--help`` with ``help_text`` and nothing else."""
    # Render the no-counter case as the empty string, not as ``str(None)``: the
    # generated script guards on truthiness, and ``"None"`` is a perfectly
    # truthy string, so the default used to tick a counter into a file literally
    # named ``None`` in whatever the CWD happened to be.
    body = (
        "import sys, pathlib\n"
        f"counter = {('' if counter is None else str(counter))!r}\n"
        "if counter:\n"
        "    p = pathlib.Path(counter)\n"
        "    p.write_text(str(int(p.read_text() or 0) + 1) if p.exists() else '1')\n"
        f"sys.stdout.write({help_text!r})\n"
        f"sys.exit({exit_code})\n"
    )
    script = path.with_suffix(".py")
    script.write_text(body, encoding="utf-8")
    path.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(script)!r} \"$@\"\n", encoding="utf-8")
    path.chmod(0o755)
    return path


REAL_HELP_EXCERPT = (
    "Options:\n"
    "  -p, --print                  Print responses to console\n"
    "  --output-format <format>     Output format (only works with --print)\n"
    "  --mode <mode>                Start in the given execution mode. plan:\n"
    "                               read-only/planning. ask: Q&A style.\n"
    "                               (choices: \"plan\", \"ask\")\n"
    "  --model <model>              Model to use\n"
    "  -f, --force                  Force allow commands unless explicitly denied\n"
    "  --trust                      Trust the current workspace without prompting\n"
)


class TestCursorArgv:
    """The exact-``==`` argv assertion (§3.1's fixture set)."""

    def test_exact_argv_when_both_tokens_confirmed(self, provider, confirmed):
        assert provider.build_command("the prompt", "gpt-5") == EXPECTED_ARGV

    def test_no_write_bypass_token_survives(self, provider, confirmed):
        """§3.5's negative set: no ``--yolo``-class token in cursor's argv."""
        cmd = provider.build_command("the prompt", "gpt-5")
        for token in ("-f", "--force", "--yolo", "--auto-review", "--approve-mcps"):
            assert token not in cmd

    def test_no_write_bypass_token_in_the_adapter_source(self):
        source = Path(cursor_agent_module.__file__).read_text(encoding="utf-8")
        for token in ('"-f"', '"--force"', '"--yolo"'):
            assert token not in source, f"{token} still rendered by the adapter"

    def test_mode_ask_is_not_narrowed_to_a_phase_subset(self, provider, confirmed):
        """Q12 answered: ask mode kept its read tools, so no phase gets a different argv."""
        first = provider.build_command("review this diff", "gpt-5")
        second = provider.build_command("answer this question", "gpt-5")
        assert first[1:6] == second[1:6] == ["--print", "--trust", "--mode", "ask", "--output-format"]


class TestFlagGate:
    """§5.4.1's load-bearing gate: confirm both tokens, or refuse the slot."""

    def test_missing_mode_refuses_the_slot(self, provider, monkeypatch):
        monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: _RUNNABLE)
        monkeypatch.setattr(
            base_module, "read_help_tokens",
            lambda _b, **_kw: frozenset({"--print", "--trust"}),
        )
        with pytest.raises(CursorFlagsUnconfirmed) as excinfo:
            provider.build_command("p", "m")
        assert excinfo.value.state == VERSION_TOO_LOW
        assert excinfo.value.missing == ("--mode",)
        message = str(excinfo.value)
        assert "--mode" in message and "--trust" in message and VERSION_TOO_LOW in message

    def test_missing_trust_refuses_the_slot(self, provider, monkeypatch):
        monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: _RUNNABLE)
        monkeypatch.setattr(
            base_module, "read_help_tokens",
            lambda _b, **_kw: frozenset({"--print", "--mode"}),
        )
        with pytest.raises(CursorFlagsUnconfirmed) as excinfo:
            provider.build_command("p", "m")
        assert excinfo.value.missing == ("--trust",)

    def test_undecidable_help_refuses_rather_than_degrading(self, provider, monkeypatch):
        monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: _RUNNABLE)
        monkeypatch.setattr(base_module, "read_help_tokens", lambda _b, **_kw: frozenset())
        with pytest.raises(CursorFlagsUnconfirmed) as excinfo:
            provider.build_command("p", "m")
        assert set(excinfo.value.missing) == set(CURSOR_REQUIRED_FLAGS)

    def test_a_refusal_never_emits_a_partial_argv(self, provider, monkeypatch):
        """The refusal must not be reachable *around*: no argv comes back at all."""
        monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: _RUNNABLE)
        monkeypatch.setattr(base_module, "read_help_tokens", lambda _b, **_kw: frozenset({"--trust"}))
        try:
            cmd = provider.build_command("p", "m")
        except CursorFlagsUnconfirmed:
            cmd = None
        assert cmd is None

    def test_help_is_read_from_the_real_binary(self, tmp_path):
        binary = _write_help_binary(tmp_path / "cursor-agent", REAL_HELP_EXCERPT)
        tokens = base_module.read_help_tokens(str(binary))
        assert "--trust" in tokens
        assert "--mode" in tokens
        assert "--output-format" in tokens

    def test_model_does_not_confirm_mode(self, tmp_path):
        """``--model`` must not be read as ``--mode``: whole-word tokens only."""
        binary = _write_help_binary(tmp_path / "cursor-agent", "  --model <model>  Model to use\n")
        tokens = base_module.read_help_tokens(str(binary))
        assert "--model" in tokens
        assert "--mode" not in tokens

    def test_nonzero_help_exit_is_unsupported(self, tmp_path):
        binary = _write_help_binary(tmp_path / "cursor-agent", REAL_HELP_EXCERPT, exit_code=1)
        assert base_module.read_help_tokens(str(binary)) == frozenset()

    def test_empty_help_is_unsupported(self, tmp_path):
        binary = _write_help_binary(tmp_path / "cursor-agent", "")
        assert base_module.read_help_tokens(str(binary)) == frozenset()

    def test_missing_binary_is_unsupported(self, tmp_path):
        assert base_module.read_help_tokens(str(tmp_path / "nope")) == frozenset()

    def test_one_help_spawn_per_binary_per_process(self, tmp_path):
        counter = tmp_path / "spawns"
        binary = _write_help_binary(tmp_path / "cursor-agent", REAL_HELP_EXCERPT, counter=counter)
        for _ in range(4):
            assert "--trust" in base_module.read_help_tokens(str(binary))
        assert counter.read_text().strip() == "1"

    def test_absent_cli_renders_argv_without_a_probe(self, provider, monkeypatch):
        """No binary is ``is_available()``'s refusal, not the gate's.

        ``detach._spec_binary`` asks the adapter for ``build_command(...)[0]``
        purely to learn the binary *name* for the "not installed" pre-flight
        message; refusing there would replace a precise message with silence.
        """
        monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: None)
        monkeypatch.setattr(
            base_module,
            "read_help_tokens",
            lambda _b, **_kw: pytest.fail("no help probe may run without a binary"),
        )
        assert provider.build_command("the prompt", "gpt-5") == EXPECTED_ARGV

    def test_unrunnable_cli_renders_argv_without_a_probe(self, provider, monkeypatch, tmp_path):
        """A resolved path that cannot be spawned cannot reject a flag either."""
        ghost = tmp_path / "cursor-agent"  # never created
        monkeypatch.setattr(cursor_agent_module.shutil, "which", lambda _n: str(ghost))
        monkeypatch.setattr(
            base_module,
            "read_help_tokens",
            lambda _b, **_kw: pytest.fail("no help probe may run without a runnable binary"),
        )
        assert provider.build_command("the prompt", "gpt-5") == EXPECTED_ARGV


class TestFakeLaneAcceptsTheNewArgv:
    """§14.1's trap: an unlisted flag is argparse exit 2, read as a provider failure."""

    def _run_mock(self, args, tmp_path):
        env = dict(os.environ, MULTI_LLM_TEST_MODE="1", MOCK_LLM_PROVIDER="cursor-agent")
        return subprocess.run(
            [sys.executable, str(MOCK_LLM), *args],
            capture_output=True, text=True, env=env, timeout=30, cwd=str(tmp_path),
        )

    def test_mock_accepts_the_shipping_argv(self, tmp_path):
        result = self._run_mock(EXPECTED_ARGV[1:], tmp_path)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["type"] == "result"

    def test_mock_help_lists_both_gated_flags(self, tmp_path):
        """The gate sniffs whichever binary the lane resolves — including the mock."""
        result = self._run_mock(["--help"], tmp_path)
        assert result.returncode == 0, result.stderr
        assert "--trust" in result.stdout
        assert "--mode" in result.stdout


@pytest.fixture
def cursor_home(tmp_path, monkeypatch):
    """A scratch ``CURSOR_CONFIG_DIR``/``CURSOR_DATA_DIR``, resolved exactly as the CLI does."""
    home = tmp_path / "cursor-home"
    (home / "projects").mkdir(parents=True)
    monkeypatch.setenv("CURSOR_CONFIG_DIR", str(home))
    monkeypatch.setenv("CURSOR_DATA_DIR", str(home))
    return home


def _marker(home: Path, slug: str, workspace: str, method: str = "cli-flag") -> Path:
    path = home / "projects" / slug / ".workspace-trusted"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"trustedAt": "2026-08-16T19:24:17.655Z", "workspacePath": workspace, "trustMethod": method}
        ),
        encoding="utf-8",
    )
    return path


ORIGINAL_CONFIG = {
    "version": 1,
    "permissions": {"allow": ["Read(**)"], "deny": []},
    "model": {"modelId": "gpt-5.6-sol", "displayName": "GPT-5.6 Sol 272K Extra High"},
    "selectedModel": {"modelId": "gpt-5.6-sol", "parameters": [{"k": "effort", "v": "xhigh"}]},
    "modelSelectionHistory": ["gpt-5.6-sol", "composer-2.5"],
    "hints": True,
}


class TestResidueCleanup:
    """Q11's two side effects: the trust marker and the rewritten default model."""

    def test_a_marker_this_run_created_is_removed(self, cursor_home, tmp_path):
        workspace = str(tmp_path)
        snapshot = cursor_agent_module.snapshot_side_effects(workspace=workspace)
        marker = _marker(cursor_home, "slug-of-the-run", workspace)

        snapshot.restore()

        assert not marker.exists()
        assert marker.parent.is_dir(), "only the marker is removed, never the project dir"

    def test_a_pre_existing_marker_is_untouched(self, cursor_home, tmp_path):
        workspace = str(tmp_path)
        marker = _marker(cursor_home, "slug-of-the-run", workspace)
        before = marker.read_bytes()

        snapshot = cursor_agent_module.snapshot_side_effects(workspace=workspace)
        snapshot.restore()

        assert marker.exists()
        assert marker.read_bytes() == before

    def test_a_marker_for_another_workspace_is_untouched(self, cursor_home, tmp_path):
        workspace = str(tmp_path)
        snapshot = cursor_agent_module.snapshot_side_effects(workspace=workspace)
        other = _marker(cursor_home, "somebody-elses-repo", "/somewhere/else/entirely")

        snapshot.restore()

        assert other.exists()

    def test_a_marker_the_user_wrote_by_hand_is_untouched(self, cursor_home, tmp_path):
        """Only ``trustMethod: cli-flag`` is ours; anything else stays.

        Observed, not assumed: a ``--trust`` marker carries
        ``"trustMethod": "cli-flag"``, and a marker acquired any other way omits
        the field entirely (a 2026-08-04 marker on the probe machine has only
        ``trustedAt`` and ``workspacePath``).
        """
        workspace = str(tmp_path)
        snapshot = cursor_agent_module.snapshot_side_effects(workspace=workspace)
        interactive = _marker(cursor_home, "slug-of-the-run", workspace, method="interactive-prompt")
        legacy = cursor_home / "projects" / "older-build" / ".workspace-trusted"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(
            json.dumps({"trustedAt": "2026-08-04T10:02:16.327Z", "workspacePath": workspace}),
            encoding="utf-8",
        )

        snapshot.restore()

        assert interactive.exists()
        assert legacy.exists()

    def test_the_three_model_keys_are_restored(self, cursor_home):
        config = cursor_home / "cli-config.json"
        config.write_text(json.dumps(ORIGINAL_CONFIG, indent=2) + "\n", encoding="utf-8")

        snapshot = cursor_agent_module.snapshot_side_effects()

        # What a `--model composer-2.5` run leaves behind.
        rewritten = json.loads(config.read_text(encoding="utf-8"))
        rewritten["model"] = {"modelId": "composer-2.5", "displayName": "Composer 2.5"}
        rewritten["selectedModel"] = {"modelId": "composer-2.5", "parameters": []}
        rewritten["modelSelectionHistory"] = ["composer-2.5", "gpt-5.6-sol"]
        rewritten["statsigLastUpdated"] = 1786908330202  # cursor's own bookkeeping
        config.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")

        snapshot.restore()

        after = json.loads(config.read_text(encoding="utf-8"))
        assert after["model"] == ORIGINAL_CONFIG["model"]
        assert after["selectedModel"] == ORIGINAL_CONFIG["selectedModel"]
        assert after["modelSelectionHistory"] == ORIGINAL_CONFIG["modelSelectionHistory"]
        assert after["statsigLastUpdated"] == 1786908330202, "unrelated keys are not reverted"
        assert after["permissions"] == ORIGINAL_CONFIG["permissions"]

    def test_model_parameters_and_the_changed_flag_are_restored_too(self, cursor_home):
        """The two keys the first cut of the guard leaked.

        ``modelParameters`` is keyed **per model id**, so a run that adds its
        own model's entry moves the mapping even though the user's own entry is
        untouched — restoring it means putting the whole mapping back, not
        merging. ``hasChangedDefaultModel`` is a bare flag cursor flips the
        first time ``--model`` moves the default, and a run that flips it tells
        the user's next interactive session that *they* changed it.
        """
        config = cursor_home / "cli-config.json"
        original = {
            **ORIGINAL_CONFIG,
            "modelParameters": {"gpt-5.6-sol": {"reasoning": "xhigh"}},
            "hasChangedDefaultModel": False,
        }
        config.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

        snapshot = cursor_agent_module.snapshot_side_effects()

        rewritten = json.loads(config.read_text(encoding="utf-8"))
        rewritten["modelParameters"] = {
            "gpt-5.6-sol": {"reasoning": "xhigh"},
            "composer-2.5": {"reasoning": "none"},
        }
        rewritten["hasChangedDefaultModel"] = True
        config.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")

        snapshot.restore()

        after = json.loads(config.read_text(encoding="utf-8"))
        assert after["modelParameters"] == original["modelParameters"]
        assert after["hasChangedDefaultModel"] is False

    def test_a_model_parameters_key_the_user_never_had_is_removed_again(self, cursor_home):
        config = cursor_home / "cli-config.json"
        original = {"version": 1, "hints": True}
        config.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

        snapshot = cursor_agent_module.snapshot_side_effects()
        config.write_text(
            json.dumps(
                {**original, "modelParameters": {"composer-2.5": {}}, "hasChangedDefaultModel": True},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        snapshot.restore()

        after = json.loads(config.read_text(encoding="utf-8"))
        assert "modelParameters" not in after
        assert "hasChangedDefaultModel" not in after

    def test_restore_is_byte_for_byte_when_only_the_model_keys_moved(self, cursor_home):
        config = cursor_home / "cli-config.json"
        original = json.dumps(ORIGINAL_CONFIG, indent=4) + "\n"
        config.write_text(original, encoding="utf-8")

        snapshot = cursor_agent_module.snapshot_side_effects()
        rewritten = json.loads(original)
        rewritten["model"] = {"modelId": "composer-2.5", "displayName": "Composer 2.5"}
        config.write_text(json.dumps(rewritten), encoding="utf-8")

        snapshot.restore()

        assert config.read_text(encoding="utf-8") == original

    def test_an_untouched_config_is_not_rewritten(self, cursor_home):
        config = cursor_home / "cli-config.json"
        config.write_text(json.dumps(ORIGINAL_CONFIG, indent=2) + "\n", encoding="utf-8")
        before = config.stat().st_mtime_ns

        snapshot = cursor_agent_module.snapshot_side_effects()
        snapshot.restore()

        assert config.stat().st_mtime_ns == before

    def test_a_key_the_user_never_had_is_removed_again(self, cursor_home):
        config = cursor_home / "cli-config.json"
        original = {"version": 1, "hints": True}
        config.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

        snapshot = cursor_agent_module.snapshot_side_effects()
        config.write_text(
            json.dumps({**original, "model": {"modelId": "composer-2.5"}}, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot.restore()

        assert "model" not in json.loads(config.read_text(encoding="utf-8"))

    def test_an_unparseable_config_is_left_alone(self, cursor_home):
        config = cursor_home / "cli-config.json"
        config.write_text(json.dumps(ORIGINAL_CONFIG), encoding="utf-8")
        snapshot = cursor_agent_module.snapshot_side_effects()
        config.write_text("{ this is not json", encoding="utf-8")

        snapshot.restore()

        assert config.read_text(encoding="utf-8") == "{ this is not json"

    def test_restore_never_raises(self, cursor_home, tmp_path):
        snapshot = cursor_agent_module.snapshot_side_effects(workspace=str(tmp_path))
        # Everything the snapshot recorded has gone away underneath it.
        import shutil as _shutil

        _shutil.rmtree(cursor_home)
        snapshot.restore()

    def test_restore_is_idempotent(self, cursor_home, tmp_path):
        workspace = str(tmp_path)
        snapshot = cursor_agent_module.snapshot_side_effects(workspace=workspace)
        marker = _marker(cursor_home, "slug-of-the-run", workspace)
        snapshot.restore()
        snapshot.restore()
        assert not marker.exists()


class TestGuardWiring:
    """The guard is armed at the one composition point both spawn sites call."""

    def test_resolve_env_arms_the_guard(self, provider, cursor_home):
        assert cursor_agent_module.armed_snapshot() is None

        provider.resolve_env("gpt-5", PROFILE_GENERATE)

        assert cursor_agent_module.armed_snapshot() is not None

    def test_arming_twice_keeps_the_first_snapshot(self, provider, cursor_home):
        """Concurrent rows must not let a later row's reading become the baseline."""
        config = cursor_home / "cli-config.json"
        config.write_text(json.dumps(ORIGINAL_CONFIG, indent=2) + "\n", encoding="utf-8")

        provider.resolve_env("gpt-5", PROFILE_GENERATE)
        first = cursor_agent_module.armed_snapshot()

        rewritten = {**ORIGINAL_CONFIG, "model": {"modelId": "composer-2.5"}}
        config.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
        provider.resolve_env("composer-2.5", PROFILE_GENERATE)

        assert cursor_agent_module.armed_snapshot() is first

        cursor_agent_module.restore_side_effects()
        after = json.loads(config.read_text(encoding="utf-8"))
        assert after["model"] == ORIGINAL_CONFIG["model"]

    def test_resolve_env_still_returns_the_shared_env_delta(self, provider, cursor_home):
        env = provider.resolve_env("gpt-5", PROFILE_GENERATE)
        assert env["GIT_TERMINAL_PROMPT"] == "0"
