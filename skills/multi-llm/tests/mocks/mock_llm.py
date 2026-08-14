#!/usr/bin/env python3
"""Unified mock LLM binary for testing.

This mock serves as a unified replacement for all provider-specific mocks.
It detects which provider is being emulated via sys.argv[0] (symlinked names)
and outputs in the correct format for each provider.

Supported Providers:
    - cursor-agent: {"type": "result", "result": "..."}
    - gemini: {"session_id": "...", "response": "...", "stats": {...}}
    - opencode: NDJSON events (step_start, text, step_finish)
    - codex: NDJSON events (thread.started, turn.started, item.completed, turn.completed)
    - kilocode: Direct JSON output
    - claude-code (launcher name ``claude``): {"type": "result", "result": "...",
      "usage": {...}, "total_cost_usd": ...}

Environment Variables (Legacy - for backward compatibility):
    MULTI_LLM_TEST_MODE: Must be set to "1" for the mock to run (safety check)
    MOCK_LLM_FIXTURE: Path to a fixture file to use as the response
    MOCK_LLM_FAIL: If set to "1", exit with an error
    MOCK_LLM_TIMEOUT: If set to "1", simulate timeout (quick exit with error)
    MOCK_OUTPUT_PATH: Path to write output file to (alternative to parsing from prompt)

Environment Variables (New - for scenario-based testing):
    MOCK_LLM_SCENARIO: Path to scenario YAML file for pattern-based responses
    MOCK_LLM_CONFIG: JSON string with dynamic configuration
    MOCK_LLM_CALL_LOG: Path to JSONL file for logging all invocations
    MOCK_LLM_VERSION: Version string `--version` prints (default: the per-provider
        release each shipped parser fixture was captured from). Drives the D3m
        minimum-version refusal path in tests.
    MOCK_LLM_LOGGED_OUT: When "1", an `auth_check` probe (D3c: `codex login
        status`, `opencode auth list`) answers with a login notice and a non-zero
        exit instead of a credential listing. Drives the executor preflight's
        refusal of a logged-out runner *before* it spawns any child.
    MOCK_LLM_TOUCH: Path the mock writes to before answering, so the D3e
        read-only sentinel test can prove a runner that ignores its profile is
        detectable (a genuinely read-only runner never sets this).
    MOCK_LLM_EDITS: JSON list of ``{path, mode: create|patch|delete, content}``
        applied **relative to the invocation cwd** before the mock prints its
        result. This is the write path (D3g): it is what makes the four outcomes
        the executor must distinguish constructible —
          * a real edit + ``status: applied``      -> applied
          * no edit  + ``status: applied``         -> clean failure
          * a partial edit + ``status: failed``    -> partial application
          * ``MOCK_LLM_PROSE=1``                   -> schema rejection
        An entry resolving outside the invocation cwd is a **test-authoring error**
        the mock rejects loudly (exit 9): the sandbox assertions must never be
        satisfied by a mock that simply could not escape.
    MOCK_LLM_WRITE_RESULT: JSON object the mock prints as the D3g batch result
        (``status``/``ids``/``files_touched``/``summary``/``notes``). Defaults are
        derived from MOCK_LLM_EDITS so the common case needs only that.
    MOCK_LLM_WRITE_IDS: JSON array of the ids the derived result attests to. A
        compliant runner reports which of the batch's ``expected_ids`` it applied
        — a batch that moves bytes and attests to none of them is a *partial
        application*, not a success — and the mock cannot see the task, so the
        scenario supplies them. Extra ids are ignored by the classifier, so one
        superset covers a multi-batch run. Unset (the default) is the negative
        case: the unattested reply.
    MOCK_LLM_PROSE: When "1", print prose instead of result JSON — the negative
        case for "a runner that returns prose fails the task instead of being
        counted as applied".

Precedence:
    1. Legacy variables take precedence when set (backward compatibility)
    2. Scenario-based variables used when legacy variables are not set
    3. MOCK_LLM_CALL_LOG is always respected regardless of mode
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def bootstrap_streams() -> None:
    """Force UTF-8 stdout/stderr, mirroring utils.stream_bootstrap.

    The orchestrator captures this process's output with ``encoding="utf-8"``.
    On Windows a piped stdout otherwise defaults to the locale codec (cp1252),
    so a fixture/response containing an em dash or any non-cp1252 character
    would either raise UnicodeEncodeError here or arrive as mojibake there.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def get_provider_name() -> str:
    """Detect provider name from sys.argv[0], falling back to MOCK_LLM_PROVIDER.

    POSIX launchers are symlinks/copies named exactly 'cursor-agent',
    'gemini', ... so sys.argv[0] carries the provider name. Windows cannot
    execute an extensionless shebang script, so its launchers either run a
    '<provider>.py' copy (argv[0] still carries the name — hence the suffix is
    stripped before matching) or run mock_llm.py directly and pass the identity
    in MOCK_LLM_PROVIDER (python always sets sys.argv[0] to the script path, so
    there is no argv[0] to key on in that case).

    Returns:
        The provider name (cursor-agent, gemini, opencode, codex, kilocode)
        or 'unknown' if not detected.
    """
    invoked = Path(sys.argv[0])
    # Match the bare launcher name and the suffix-stripped name, so both
    # `cursor-agent` (POSIX symlink) and `cursor-agent.py` (Windows copy) work.
    invoked_names = (invoked.name, invoked.stem)

    # Handle both direct name and any suffix/prefix variations
    known_providers = ["cursor-agent", "gemini", "opencode", "codex", "kilocode"]

    # The claude-code adapter's binary is `claude`, so the launcher name and the
    # provider id differ for exactly this one.
    for invoked_name in invoked_names:
        if invoked_name == "claude" or invoked_name.endswith("claude"):
            return "claude-code"

    for provider in known_providers:
        for invoked_name in invoked_names:
            if invoked_name == provider or invoked_name.endswith(provider):
                return provider

    # Fallback: if invoked directly as mock_llm.py, check for provider env var
    if os.environ.get("MOCK_LLM_PROVIDER"):
        return os.environ["MOCK_LLM_PROVIDER"]

    return "unknown"


def load_scenario(scenario_path: str) -> Optional[Dict[str, Any]]:
    """Load a scenario YAML file.

    Args:
        scenario_path: Path to the scenario YAML file.

    Returns:
        Parsed scenario dict or None if loading fails.
    """
    try:
        import yaml
    except ImportError:
        # yaml not available, return None
        return None

    try:
        with open(scenario_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None


def match_prompt_pattern(prompt: str, scenario: Dict[str, Any]) -> Optional[str]:
    """Match prompt against scenario patterns and return fixture path.

    Args:
        prompt: The prompt text to match.
        scenario: The loaded scenario dict.

    Returns:
        Path to fixture file or None if no match.
    """
    prompts = scenario.get("prompts", [])
    for entry in prompts:
        pattern = entry.get("pattern", "")
        fixture = entry.get("fixture", "")
        if pattern and fixture:
            try:
                if re.search(pattern, prompt, re.IGNORECASE):
                    return fixture
            except re.error:
                # Invalid regex, skip
                continue
    return None


def _unique(values: List[str]) -> List[str]:
    """De-duplicate while keeping first-seen order."""
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def echo_subagent_answer(kind: str, prompt: str) -> Optional[str]:
    """Answer a materialized subagent prompt by echoing the ids it embeds.

    Selected with ``MOCK_LLM_ECHO=<kind>``. A materialized read/analyze prompt
    carries every input it needs as *values* (``context_mode: inline``), so a
    schema-valid answer can be derived from the prompt alone — which is exactly
    what makes an executor dispatch comparable to the in-process native-invoker
    simulation: both sides derive the same answer from the same prompt, one
    through a real subprocess and the provider adapter stack, the other in
    process. The shapes mirror
    ``SkillRunner.simulate_subagents``'s default builder.

    Returns None for an unknown kind **and whenever the prompt carries none of
    the ids that kind needs**, so the caller falls through to the normal response
    precedence. That is what makes ``MOCK_LLM_ECHO`` safe to set for a whole
    orchestrator run: the model fan-out's own review prompts contain no group
    hashes, so they still get the scenario/fixture answer, and only the
    materialized subagent prompts are echoed.
    """
    if kind in ("validation", "revalidation"):
        hashes = _unique(re.findall(r'"group_hash":\s*"([0-9a-fA-F]+)"', prompt))
        if not hashes:
            return None
        indices = _unique(re.findall(r'"index":\s*(\d+)', prompt))
        return json.dumps({
            "groups": [
                {
                    "group_index": int(indices[position]) if position < len(indices) else position,
                    "group_hash": group_hash,
                    "status": "valid",
                    "reason": "simulated",
                    "confidence": 1.0,
                }
                for position, group_hash in enumerate(hashes)
            ],
            "metadata": {
                "model": "mock",
                "timestamp": "1970-01-01T00:00:00",
                "schema_version": "2.1",
            },
        })

    if kind == "human_decision_judge":
        group_ids = _unique(re.findall(r'"group_id":\s*"([0-9a-fA-F]+)"', prompt))
        if not group_ids:
            return None
        return json.dumps({
            "schema_version": "1.0",
            "verdicts": [
                {
                    "group_id": group_id,
                    "decision": "approve",
                    "reason": "simulated approval",
                }
                for group_id in group_ids
            ],
        })

    if kind == "consolidation":
        group_ids = _unique(re.findall(r'"group_id":\s*"([0-9a-fA-F]+)"', prompt))
        if not group_ids:
            return None
        indices = _unique(re.findall(r'"index":\s*(\d+)', prompt))
        return json.dumps({
            "verification": "passed",
            "clusters": [],
            "singletons": [
                {
                    "group_id": group_id,
                    "group_index": int(indices[position]) if position < len(indices) else position,
                    "reasoning": "simulated singleton",
                }
                for position, group_id in enumerate(group_ids)
            ],
        })

    if kind == "task_generation":
        # The one kind whose context is the repository, so there are no ids in the
        # prompt to echo — the marker is the prompt's own opening sentence, and the
        # answer is a fixed two-task graph that satisfies both the schema and the
        # semantic rules (unique T### ids, declared dependencies, acyclic, one
        # dependency-free task). Deterministic on purpose: the executor leg and the
        # native-invoker leg must persist byte-identical documents.
        if "You are decomposing" not in prompt:
            return None
        return json.dumps({
            "schema_version": "1.0",
            "plan_preamble": (
                "Simulated preamble: implement the plan in two steps, foundation "
                "first, then the wiring that consumes it."
            ),
            "tasks": [
                {
                    "id": "T001",
                    "title": "Create the foundation module",
                    "description": "Add the module the plan names, exporting the "
                                   "interface T002 consumes.",
                    "depends_on": [],
                    "files_to_create": ["src/foundation.py"],
                    "files_to_modify": [],
                    "acceptance_criteria": [
                        "src/foundation.py exports the documented interface",
                    ],
                    "estimated_complexity": "low",
                    "executor_role": "agent",
                },
                {
                    "id": "T002",
                    "title": "Wire the foundation into the entry point",
                    "description": "Import T001's interface from the entry point "
                                   "and call it where the plan says.",
                    "depends_on": ["T001"],
                    "files_to_create": [],
                    "files_to_modify": ["src/main.py"],
                    "acceptance_criteria": [
                        "src/main.py calls the interface T001 exported",
                    ],
                    "estimated_complexity": "medium",
                    "executor_role": "agent",
                },
            ],
        })

    if kind == "salvage":
        if "JSON extraction specialist" not in prompt:
            return None
        return json.dumps([
            {
                "title": "Salvaged suggestion",
                "desc": "Recovered from an unparseable model reply.",
                "importance": "MEDIUM",
                "reference": "## Step 1",
                "type": "addition",
            }
        ])
    return None


# ---------------------------------------------------------------------------
# The write path (D3g): real edits under cwd, then the batch result JSON
# ---------------------------------------------------------------------------

EXIT_TEST_AUTHORING_ERROR = 9


def apply_mock_edits() -> List[Dict[str, Any]]:
    """Apply ``MOCK_LLM_EDITS`` relative to the invocation cwd, before answering.

    Returns the list of applied entries (empty when the variable is unset), so the
    default batch result can name the files it actually touched.

    An entry that resolves outside the invocation cwd exits **9** rather than being
    clamped: the executor's containment and sandbox assertions must be satisfied by
    a mock that genuinely *could* have escaped and was stopped, never by one that
    was structurally incapable of it. Exit 9 is distinct from every provider-failure
    code so a test-authoring mistake cannot masquerade as a runner error.
    """
    raw = os.environ.get("MOCK_LLM_EDITS")
    if not raw:
        return []
    try:
        edits = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"MOCK EDITS ERROR: MOCK_LLM_EDITS is not JSON ({exc})", file=sys.stderr)
        sys.exit(EXIT_TEST_AUTHORING_ERROR)
    if not isinstance(edits, list):
        print("MOCK EDITS ERROR: MOCK_LLM_EDITS must be a JSON list", file=sys.stderr)
        sys.exit(EXIT_TEST_AUTHORING_ERROR)

    cwd = Path.cwd().resolve()
    applied: List[Dict[str, Any]] = []
    for entry in edits:
        if not isinstance(entry, dict) or not entry.get("path"):
            print(
                f"MOCK EDITS ERROR: each entry needs a `path`; got {entry!r}",
                file=sys.stderr,
            )
            sys.exit(EXIT_TEST_AUTHORING_ERROR)
        target = Path(entry["path"])
        if not target.is_absolute():
            target = cwd / target
        resolved = target.resolve()
        try:
            inside = resolved.is_relative_to(cwd)
        except AttributeError:  # pragma: no cover - Python < 3.9
            inside = os.path.commonpath([str(resolved), str(cwd)]) == str(cwd)
        if not inside:
            print(
                f"MOCK EDITS ERROR: {entry['path']!r} resolves to {resolved}, "
                f"outside the invocation cwd {cwd}. That is a test-authoring "
                f"error: the write-containment assertions must be satisfied by a "
                f"mock that could have escaped, not by one that could not.",
                file=sys.stderr,
            )
            sys.exit(EXIT_TEST_AUTHORING_ERROR)

        mode = str(entry.get("mode") or "patch")
        if mode == "delete":
            try:
                resolved.unlink()
            except FileNotFoundError:
                pass
        elif mode == "create":
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(str(entry.get("content") or ""), encoding="utf-8")
        elif mode == "patch":
            resolved.parent.mkdir(parents=True, exist_ok=True)
            existing = (
                resolved.read_text(encoding="utf-8") if resolved.is_file() else ""
            )
            resolved.write_text(
                existing + str(entry.get("content") or ""), encoding="utf-8"
            )
        else:
            print(
                f"MOCK EDITS ERROR: unknown mode {mode!r} (create|patch|delete)",
                file=sys.stderr,
            )
            sys.exit(EXIT_TEST_AUTHORING_ERROR)
        applied.append({
            "path": os.path.relpath(str(resolved), str(cwd)).replace(os.sep, "/"),
            "mode": mode,
        })
    return applied


def write_batch_result(applied: List[Dict[str, Any]]) -> str:
    """The D3g batch result JSON this invocation prints on stdout.

    ``MOCK_LLM_WRITE_RESULT`` wins entirely (so a test can claim ``applied`` while
    editing nothing, or ``failed`` while editing something); otherwise the shape is
    derived from what was actually edited. Usage fields appear **only** when the
    scenario sets them, which is what gives the "no fabricated token counts"
    assertion a negative case.
    """
    override = os.environ.get("MOCK_LLM_WRITE_RESULT")
    if override:
        return override
    touched = [entry["path"] for entry in applied if entry["mode"] != "delete"]
    touched += [entry["path"] for entry in applied if entry["mode"] == "delete"]
    try:
        ids = json.loads(os.environ.get("MOCK_LLM_WRITE_IDS") or "[]")
    except json.JSONDecodeError:
        ids = []
    return json.dumps({
        "status": "applied" if applied else "no-op",
        "ids": [str(item) for item in ids] if isinstance(ids, list) else [],
        "files_touched": touched,
        "summary": (
            f"edited {len(touched)} file(s): {', '.join(touched)}"[:200]
            if touched else "nothing needed changing"
        ),
    })


def get_response_content(prompt: str) -> str:
    """Determine response content based on environment and prompt.

    Precedence:
    1. MOCK_LLM_FIXTURE (legacy)
    2. MOCK_LLM_ECHO (materialized-subagent answer derived from the prompt)
    3. MOCK_LLM_SCENARIO pattern matching
    4. MOCK_LLM_CONFIG
    5. Default response

    Args:
        prompt: The prompt text (used for pattern matching).

    Returns:
        Response content string.
    """
    # Legacy: MOCK_LLM_FIXTURE takes highest precedence
    fixture_path = os.environ.get("MOCK_LLM_FIXTURE")
    if fixture_path and Path(fixture_path).exists():
        with open(fixture_path, "r", encoding="utf-8") as f:
            return f.read()

    # D3: answer a materialized subagent prompt from the ids it embeds.
    echo_kind = os.environ.get("MOCK_LLM_ECHO")
    if echo_kind:
        echoed = echo_subagent_answer(echo_kind, prompt)
        if echoed is not None:
            return echoed

    # New: MOCK_LLM_SCENARIO for pattern-based responses
    scenario_path = os.environ.get("MOCK_LLM_SCENARIO")
    if scenario_path:
        scenario = load_scenario(scenario_path)
        if scenario:
            fixture_rel = match_prompt_pattern(prompt, scenario)
            if fixture_rel:
                # Resolve fixture path relative to scenario file location
                scenario_dir = Path(scenario_path).parent
                fixture_full = scenario_dir / fixture_rel
                if fixture_full.exists():
                    with open(fixture_full, "r", encoding="utf-8") as f:
                        return f.read()

    # New: MOCK_LLM_CONFIG for dynamic configuration
    config_json = os.environ.get("MOCK_LLM_CONFIG")
    if config_json:
        try:
            config = json.loads(config_json)
            if "response" in config:
                response = config["response"]
                if isinstance(response, str):
                    return response
                return json.dumps(response)
        except json.JSONDecodeError:
            pass

    # Default response based on provider
    provider = get_provider_name()
    return json.dumps([
        {
            "id": f"mock-{provider}-1",
            "type": "suggestion",
            "description": f"Mock suggestion from {provider}",
            "importance": "MEDIUM",
            "location": "mock_file.py:10"
        }
    ])


def write_output_file(prompt: str, content: str) -> None:
    """Write response content to output file if specified.

    Args:
        prompt: The prompt text (may contain OUTPUT_FILE directive).
        content: The response content to write.
    """
    output_path = None

    # Check environment variable first
    if os.environ.get("MOCK_OUTPUT_PATH"):
        output_path = Path(os.environ["MOCK_OUTPUT_PATH"])
    elif "OUTPUT_FILE:" in prompt:
        # Extract output file path from prompt
        for line in prompt.split("\n"):
            if "OUTPUT_FILE:" in line:
                output_path = Path(line.split("OUTPUT_FILE:")[1].strip())
                break

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)


def log_call(provider: str, args: List[str], prompt: str) -> None:
    """Log the mock call to JSONL file if configured.

    Args:
        provider: The detected provider name.
        args: Command line arguments.
        prompt: The prompt text.
    """
    log_path = os.environ.get("MOCK_LLM_CALL_LOG")
    if not log_path:
        return

    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provider": provider,
        "argv": sys.argv,
        "args": args,
        "prompt": prompt,
        # The assertion surface for D3e/D3h: `cwd` proves the executor pinned the
        # resolved repo root rather than inheriting one, and `prompt_transport`
        # proves a large prompt actually left argv rather than merely fitting.
        "cwd": os.getcwd(),
        "prompt_transport": "argv" if prompt in args else "stdin",
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith("MOCK_LLM_") or k == "MULTI_LLM_TEST_MODE"
        }
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def format_cursor_agent_output(content: str) -> str:
    """Format output for cursor-agent provider.

    Format: {"type": "result", "result": "<content>"}
    """
    return json.dumps({"type": "result", "result": content})


def format_gemini_output(content: str) -> str:
    """Format output for gemini provider.

    Format: {"session_id": "...", "response": "<content>", "stats": {...}}
    """
    return json.dumps({
        "session_id": "mock-session-12345",
        "response": content,
        "stats": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
    })


def format_claude_code_output(content: str) -> str:
    """Format output for the claude-code provider (binary name ``claude``).

    Mirrors `claude -p --output-format json`: the answer is wrapped in a
    ``result`` envelope that also carries the ``usage`` block and
    ``total_cost_usd`` the D3j extractor reads. ``parse_output`` unwraps
    ``result`` and discards the rest, so keeping the envelope complete here is
    what makes the usage extractor testable at all.
    """
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": content,
        "session_id": "mock-session-claude",
        "num_turns": 1,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 0,
        },
        "total_cost_usd": 0.0123,
    })


def format_opencode_output(content: str) -> str:
    """Format output for opencode provider.

    Mirrors the real `opencode run --format json` stream. Every event carries
    a millisecond `timestamp` and `sessionID` envelope plus a `part` object;
    note that the top-level type uses underscores (`step_start`) while
    `part.type` uses hyphens (`step-start`). Only parts whose `time.end` is
    set are emitted as `text` events, which is how the CLI excludes the user's
    own prompt echo and the mid-stream `text-start` marker.
    """
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    session_id = "ses_mock000000000000000000000"
    message_id = "msg_mock000000000000000000000"

    def envelope(event_type: str, **rest: object) -> dict:
        return {"type": event_type, "timestamp": now_ms, "sessionID": session_id, **rest}

    events = [
        envelope(
            "step_start",
            part={
                "id": "prt_mock_step_start",
                "sessionID": session_id,
                "messageID": message_id,
                "type": "step-start",
            },
        ),
        envelope(
            "text",
            part={
                "id": "prt_mock_text",
                "sessionID": session_id,
                "messageID": message_id,
                "type": "text",
                "text": content,
                "time": {"start": now_ms, "end": now_ms},
            },
        ),
        envelope(
            "step_finish",
            part={
                "id": "prt_mock_step_finish",
                "sessionID": session_id,
                "messageID": message_id,
                "type": "step-finish",
                # The real StepFinishPart carries `reason`, `cost` and a
                # `tokens: {input, output, reasoning, cache: {read, write}}`
                # object (verified against the part schema in the 1.18.x
                # binary's embedded bundle). The D3j extractor sums these across
                # steps, so a mock without them could not exercise it.
                "reason": "stop",
                "cost": 0.0042,
                "tokens": {
                    "input": 200,
                    "output": 60,
                    "reasoning": 5,
                    "cache": {"read": 40, "write": 0},
                },
            },
        ),
    ]
    return "\n".join(json.dumps(event) for event in events)


def format_codex_output(content: str) -> str:
    """Format output for codex provider.

    Format: NDJSON event stream, mirroring real `codex exec --json` output
    (captured from codex-cli 0.144.0):

        thread.started -> turn.started -> item.completed(agent_message)
        -> turn.completed

    The assistant reply lives in the nested item, NOT in a top-level "text"
    event. Keep this in step with the real CLI: a mock shaped to match the
    parser instead of the CLI is what let issue #2 ship unnoticed.
    """
    events = [
        {"type": "thread.started", "thread_id": "mock-thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "agent_message",
                "text": content,
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 12,
                "reasoning_output_tokens": 0,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


def format_kilocode_output(content: str) -> str:
    """Format output for kilocode provider.

    Format: Direct JSON output (the content itself, or wrapped if needed)
    The kilocode parser expects direct JSON that can be parsed.
    """
    # kilocode just outputs the content directly
    # If content is already valid JSON, return as-is
    # Otherwise, wrap it
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        return json.dumps({"output": content})


def format_output(provider: str, content: str) -> str:
    """Format output according to provider's expected wire format.

    Args:
        provider: The provider name.
        content: The response content.

    Returns:
        Formatted output string.
    """
    formatters = {
        "cursor-agent": format_cursor_agent_output,
        "gemini": format_gemini_output,
        "opencode": format_opencode_output,
        "codex": format_codex_output,
        "kilocode": format_kilocode_output,
        "claude-code": format_claude_code_output,
    }

    formatter = formatters.get(provider, format_cursor_agent_output)
    return formatter(content)


def parse_provider_args(provider: str) -> tuple[argparse.Namespace, str]:
    """Parse command-line arguments based on provider.

    Each provider has slightly different argument patterns.

    Args:
        provider: The provider name.

    Returns:
        Tuple of (parsed args namespace, prompt string)
    """
    if provider == "cursor-agent":
        parser = argparse.ArgumentParser(description="Mock cursor-agent CLI")
        parser.add_argument("--print", dest="print_mode", action="store_true")
        parser.add_argument("-f", "--force", action="store_true")
        parser.add_argument("--output-format", dest="output_format", default="json")
        parser.add_argument("--model", default="auto")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    elif provider == "gemini":
        parser = argparse.ArgumentParser(description="Mock gemini CLI")
        parser.add_argument("--output-format", dest="output_format", default="json")
        parser.add_argument("--model", default="gemini-2.0-flash")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    elif provider == "opencode":
        parser = argparse.ArgumentParser(description="Mock opencode CLI")
        subparsers = parser.add_subparsers(dest="command")
        run_parser = subparsers.add_parser("run")
        run_parser.add_argument("--format", dest="output_format", default="json")
        run_parser.add_argument("--model", default="claude-sonnet")
        # The rest of what OpenCodeProvider.build_command actually emits. The
        # mock must accept the real argv or the e2e path silently degrades to
        # an argparse exit-2 that looks like a provider failure.
        run_parser.add_argument("--print-logs", dest="print_logs", action="store_true")
        run_parser.add_argument("--log-level", dest="log_level", default=None)
        run_parser.add_argument("--variant", default=None)
        # The `agent-readonly` profile passes `--agent plan` (D3e). Accepting it
        # here is what keeps the read-only dispatch from degrading into an
        # argparse exit-2 that looks like a provider failure.
        run_parser.add_argument("--agent", default=None)
        run_parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        if args.command != "run":
            print("ERROR: Only 'run' subcommand is supported", file=sys.stderr)
            sys.exit(1)
        return args, args.prompt

    elif provider == "codex":
        parser = argparse.ArgumentParser(description="Mock codex CLI")
        subparsers = parser.add_subparsers(dest="command")
        exec_parser = subparsers.add_parser("exec")
        # Real codex spells this `-s, --sandbox <SANDBOX_MODE>`. --full-auto is
        # deliberately NOT accepted: codex 0.146.0 keeps it only as a hidden
        # deprecation trap, so rejecting it here catches a regression that
        # reintroduces the old flag.
        exec_parser.add_argument("-s", "--sandbox", default=None)
        exec_parser.add_argument("--json", action="store_true")
        exec_parser.add_argument("--model", default="gpt-4")
        exec_parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        if args.command != "exec":
            print("ERROR: Only 'exec' subcommand is supported", file=sys.stderr)
            sys.exit(1)
        return args, args.prompt

    elif provider == "claude-code":
        parser = argparse.ArgumentParser(description="Mock claude CLI")
        parser.add_argument("-p", "--print", dest="print_mode", action="store_true")
        parser.add_argument("--output-format", dest="output_format", default="json")
        parser.add_argument("--model", default="sonnet")
        parser.add_argument("--effort", default=None)
        # The `agent-readonly` profile's tool policy (D3e). Real `claude` accepts
        # both spellings of each flag; the adapter emits the hyphenated ones.
        parser.add_argument("--allowed-tools", "--allowedTools", dest="allowed_tools",
                            default=None)
        parser.add_argument("--disallowed-tools", "--disallowedTools",
                            dest="disallowed_tools", default=None)
        # The `agent-write` profile's pre-approval (D3g). Accepting it here is what
        # keeps an edit-capable dispatch from degrading into an argparse exit-2
        # that looks like a provider failure. `--dangerously-skip-permissions` is
        # deliberately NOT accepted: no profile may emit it, so a regression that
        # introduces it fails loudly here.
        parser.add_argument("--permission-mode", dest="permission_mode", default=None)
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    elif provider == "kilocode":
        parser = argparse.ArgumentParser(description="Mock kilocode CLI")
        parser.add_argument("--auto", action="store_true")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    else:
        # Unknown provider - basic argument parsing
        parser = argparse.ArgumentParser(description="Mock LLM CLI")
        parser.add_argument("--model", default="default")
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt


#: What `<provider> --version` prints, mirroring each real CLI's own format.
#: These are the releases the shipped parser fixtures were captured from, so a
#: mock run passes the D3m minimum-version gate the way a supported CLI does.
VERSION_STRINGS = {
    "codex": "codex-cli 0.146.0",
    "opencode": "1.18.12",
    "claude-code": "2.1.224 (Claude Code)",
    "cursor-agent": "2026.02.01-abcdef",
    "gemini": "0.9.0",
    "kilocode": "1.4.0",
}


def print_version(provider: str) -> None:
    """Answer `--version` the way the emulated CLI does (D3m preflight)."""
    override = os.environ.get("MOCK_LLM_VERSION")
    print(override if override else VERSION_STRINGS.get(provider, "0.0.0"))


#: The `auth_check` subcommand each runner-eligible adapter declares, minus the
#: launcher name (`utils/providers/*.py`). The D3c preflight probes it once per
#: distinct runner per run, so the mock must answer it as its own step — before
#: prompt parsing, the call log and the edit/sentinel side effects, none of which
#: a login-status probe performs on a real CLI.
AUTH_PROBE_ARGS = {
    "codex": ("login", "status"),
    "opencode": ("auth", "list"),
}

#: What each probe prints when the CLI is logged in.
AUTH_OK_STRINGS = {
    "codex": "Logged in using ChatGPT",
    "opencode": "Credentials\n  anthropic  oauth",
}


def is_auth_probe(argv: List[str]) -> bool:
    """Whether this invocation is an adapter's `auth_check`, not a dispatch."""
    positional = tuple(arg for arg in argv if not arg.startswith("-"))
    return any(
        positional[: len(expected)] == expected
        for expected in AUTH_PROBE_ARGS.values()
    )


def print_auth_status(provider: str) -> int:
    """Answer an `auth_check` probe; return the exit code the CLI would use.

    ``MOCK_LLM_LOGGED_OUT=1`` is the negative case: a login notice on stderr plus
    a non-zero exit, which is what the executor's preflight must refuse *before*
    it spawns the first child. The notice is deliberately one of the generic
    phrases in ``auth_error_patterns`` rather than a claim about any real CLI's
    exact wording.
    """
    if os.environ.get("MOCK_LLM_LOGGED_OUT") == "1":
        print("Not logged in. Run the login command first.", file=sys.stderr)
        return 1
    print(AUTH_OK_STRINGS.get(provider, "Logged in"))
    return 0


def read_stdin_prompt() -> str:
    """Read the prompt from stdin (the executor's non-argv transport, D2d).

    Every runner-eligible CLI reads its prompt from stdin when no positional
    prompt is given, and the executor uses exactly that transport for
    materialized prompts, so a mock that only looked at argv would see an empty
    prompt and silently answer the default fixture.
    """
    try:
        if sys.stdin is None or sys.stdin.closed:
            return ""
        return sys.stdin.read()
    except (OSError, ValueError):
        return ""


def main() -> int:
    """Main entry point for the unified mock LLM binary.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    bootstrap_streams()

    # Safety check: only run in test mode
    if os.environ.get("MULTI_LLM_TEST_MODE") != "1":
        print(
            "ERROR: mock_llm.py should only be used in test mode. "
            "Set MULTI_LLM_TEST_MODE=1 to enable.",
            file=sys.stderr,
        )
        return 1

    # Detect provider from invocation name
    provider = get_provider_name()

    # `--version` is answered before any provider argument parsing: the D3m
    # preflight runs it once per distinct runner, and every real CLI accepts it
    # without a subcommand.
    if "--version" in sys.argv[1:] or "-V" in sys.argv[1:]:
        print_version(provider)
        return 0

    # An `auth_check` probe (D3c) is answered the same way, and for the same
    # reason: the preflight runs it once per distinct runner, and it is not a
    # dispatch — so it must not touch the sentinel, apply mock edits or land in
    # the call log.
    if is_auth_probe(sys.argv[1:]):
        return print_auth_status(provider)

    # Parse arguments based on provider
    args, prompt = parse_provider_args(provider)
    if not prompt:
        prompt = read_stdin_prompt()

    # A genuinely read-only runner can never do this; the D3e sentinel test uses
    # it to prove the check has teeth rather than passing vacuously.
    touch_path = os.environ.get("MOCK_LLM_TOUCH")
    if touch_path:
        Path(touch_path).parent.mkdir(parents=True, exist_ok=True)
        Path(touch_path).write_text("written by the mock runner\n", encoding="utf-8")

    # Log the call (always, regardless of mode)
    log_call(provider, sys.argv[1:], prompt)

    # The write path (D3g): edits land BEFORE the result is printed, so the
    # executor's post-run snapshot comparison sees exactly what a real
    # edit-capable dispatch would have left behind.
    applied_edits = apply_mock_edits()
    if applied_edits or os.environ.get("MOCK_LLM_WRITE_RESULT") or os.environ.get(
        "MOCK_LLM_PROSE"
    ):
        if os.environ.get("MOCK_LLM_PROSE") == "1":
            # The schema-rejection case: a runner that narrates instead of
            # answering must fail the task, never be counted as applied.
            content = (
                "I went ahead and made the changes you asked for. Everything "
                "looks good now! Let me know if you want anything else."
            )
        else:
            content = write_batch_result(applied_edits)
        print(format_output(provider, content))
        return 0

    # Check for simulated failures (legacy)
    if os.environ.get("MOCK_LLM_FAIL") == "1":
        print("ERROR: Simulated failure", file=sys.stderr)
        return 1

    # Check for simulated timeout (legacy)
    # Instead of sleeping forever (which would hang tests), we exit with
    # a recognizable timeout error pattern after a brief delay
    if os.environ.get("MOCK_LLM_TIMEOUT") == "1":
        # Small delay to simulate some processing, then fail with timeout error
        time.sleep(0.05)  # 50ms
        print("ERROR: Simulated timeout - operation timed out", file=sys.stderr)
        return 124  # Standard timeout exit code

    # Get response content
    content = get_response_content(prompt)

    # Write to output file if specified
    write_output_file(prompt, content)

    # Format and print output according to provider's wire format
    output = format_output(provider, content)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
