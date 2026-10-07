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
    - agy: single JSON envelope {"conversation_id": ..., "status": "SUCCESS",
      "response": "...", "duration_seconds": ..., "num_turns": ..., "usage": {...}}

Every identity here is five parts, and no part is optional (C5): the name in
``get_provider_name``/``SkillRunner.PROVIDERS``, an argv parser matching that
adapter's ``build_command`` exactly, a formatter emitting the CLI's *real* wire
format, a ``VERSION_STRINGS`` entry, and — only for an adapter declaring an
``auth_check`` — an ``AUTH_PROBE_ARGS``/``AUTH_OK_STRINGS`` pair. Every one of
those lookups falls back **silently** (see ``format_output``,
``parse_provider_args``, ``VERSION_STRINGS`` and ``AUTH_PROBE_ARGS`` below), so a
half-added identity does not fail loudly — it emits another CLI's wire format
and hands the adapter a shape its real CLI never produces. ``tests/
test_mock_adapter_roundtrip.py`` and ``tests/test_fake_lane_roster.py`` are what
turn that from a convention into an assertion.

Environment Variables (Legacy - for backward compatibility):
    MULTI_LLM_TEST_MODE: Must be set to "1" for the mock to run (safety check)
    MOCK_LLM_FIXTURE: Path to a fixture file to use as the response
    MOCK_LLM_FAIL: If set to "1", exit with an error
    MOCK_LLM_TIMEOUT: If set to "1", simulate timeout (quick exit with error)
    MOCK_OUTPUT_PATH: Path to write output file to (alternative to parsing from prompt)

Environment Variables (New - for scenario-based testing):
    MOCK_LLM_SCENARIO: Path to scenario YAML file for pattern-based responses.
        What that file *means* is defined once, in ``tests/scenario_matcher.py``,
        which this binary imports and the container lane's HTTP stub imports too
        (C4) — so the two can never disagree about which fixture answers which
        prompt. A scenario declaring ``on_unmatched: fail`` makes an unanswered
        prompt exit 8 with the prompt's sha256 on stderr instead of falling
        through to the generic default below.
    MOCK_LLM_TURN_INDEX: The zero-based per-session turn index the matcher's
        second selection stage uses (pattern match -> turn index -> declared
        fallback). The fake binary has no session of its own, so the caller
        declares it; the HTTP stub derives the same number from the per-exec id
        in its base URL. Ignored by any scenario that declares no ``turns:``.
    MOCK_LLM_CONFIG: JSON string with dynamic configuration. ``"response"`` is
        the answer text. ``"error"`` (with an optional ``"exit_code"``, default
        1) makes the run FAIL the way the provider's real CLI reports a failure
        (``format_failure``) instead of answering with the error text: a
        classifier that reads only what the vendor wrote (``vendor_text``) must
        find it where the vendor puts it, never in the model's answer.
    MOCK_LLM_CALL_LOG: Path to JSONL file for logging all invocations
    MOCK_LLM_PLUGIN: Path to a **throwaway** one-off fake adapter (M22). See
        ``load_mock_plugin``/``install_mock_plugin`` at the bottom of this file:
        it is imported *after* every table here is built, honoured only under
        MULTI_LLM_TEST_MODE=1, and may replace exactly the four seams named in
        ``PLUGIN_SEAMS`` — anything else is a loud refusal, not a silent no-op.
        A probe that needs one fake provider to behave unusually (a huge
        payload, a malformed envelope, a version below a gate) writes a scratch
        module OUTSIDE the tree and points this at it, instead of editing this
        shared file in place — which is a real incident, not a hypothetical
        (plans/harness-invocation-improvements.md §13.1: this file was
        overwritten 1275 lines -> 40 and broke two suites by collection error).
        ``SkillRunner(mock_plugin=...)`` is the harness-side way to set it.
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
    MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR: "honour" or "ignore" — agy only. Selects
        one of the two §9.6.e stubs behind `--print-timeout`: the child that
        reaches its own derived cutoff and shuts down cleanly before the outer
        deadline, or the one that ignores the flag and is orphaned by the
        leader-only kill. See `run_print_timeout_stub`.
    MOCK_LLM_PID_FILE: Path the print-timeout stubs write their descendant's pid
        to, so a test can assert nothing outlived the run rather than infer it.
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
    MOCK_LLM_EDITS_IF_PROMPT_HAS: a substring; when set, MOCK_LLM_EDITS apply
        only to an invocation whose prompt contains it (a check task and an
        edit task in one resumed run: the check must move nothing while the
        task after it edits for real). The derived batch result follows what
        was actually applied, so a gated-off call reports ``no-op``.
    MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS: a substring; an invocation whose prompt
        contains it prints a ``status: failed`` batch result (summary from
        MOCK_LLM_WRITE_FAIL_SUMMARY, notes from MOCK_LLM_WRITE_FAIL_NOTES)
        while every other invocation keeps the derived result.
    MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS: a substring; an invocation whose prompt
        contains it prints a ``status: no-op`` batch result carrying the
        ``check`` object in MOCK_LLM_WRITE_CHECK (JSON; default: two findings)
        — a check task that ran and reported problems.
    MOCK_LLM_WRITE_BATCH_CHECK: JSON ``check`` object an applied batch's
        derived result carries (the checks it ran on its own edits; fix round
        14), only when the prompt holds MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS
        if that is set
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
    MOCK_LLM_REPAIR_RESULT: The text the mock prints when the prompt is a write
        batch's report re-ask (``REPORT_REASK_MARKER``, the ladder's rule 7 for an
        apply kind). A well-behaved runner edits nothing on that turn, so the
        mock applies no ``MOCK_LLM_EDITS`` then — unless
    MOCK_LLM_REPAIR_EDITS: is "1": the misbehaving runner that re-applies the
        batch while it was only asked to restate its report.
    MOCK_LLM_REPAIR_RESTATE: is "1" (and MOCK_LLM_REPAIR_RESULT unset): the
        well-behaved runner that restates its report from the previous reply
        the re-ask QUOTES — status, ``ids`` and ``files_touched`` read off the
        quote, so whatever the quote cut is missing from the restatement. An
        item the quote names by position ("Suggestion 1") is restated as the
        id the re-ask lists beside that number, when it lists any.

Precedence:
    1. Legacy variables take precedence when set (backward compatibility)
    2. Scenario-based variables used when legacy variables are not set
    3. MOCK_LLM_CALL_LOG is always respected regardless of mode
"""

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import re
import sys
import time
import types
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
        The provider name (cursor-agent, gemini, opencode, codex, kilocode, agy)
        or 'unknown' if not detected.
    """
    invoked = Path(sys.argv[0])
    # Match the bare launcher name and the suffix-stripped name, so both
    # `cursor-agent` (POSIX symlink) and `cursor-agent.py` (Windows copy) work.
    invoked_names = (invoked.name, invoked.stem)

    # Handle both direct name and any suffix/prefix variations
    known_providers = [
        "cursor-agent", "gemini", "opencode", "codex", "kilocode", "agy",
    ]

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


class ScenarioRefused(Exception):
    """A scenario declaring ``on_unmatched: fail`` refused to answer this prompt.

    Wraps whatever ``scenario_matcher`` raised so ``main`` can turn it into one
    exit code without importing the matcher's exception classes at module level.
    The message it carries is the matcher's, which names the scenario and the
    prompt's sha256 and **never** the prompt body.
    """


def _scenario_matcher():
    """Import ``tests/scenario_matcher.py``, or exit naming where it was looked for.

    That module is the one definition of what a scenario means (C4), shared with
    the container lane's HTTP stub, so this file imports it instead of keeping a
    second copy. Two locations are searched, because a provider launcher is
    either a symlink to this file or a copy of it: ``tests/`` — the grandparent
    of the real ``tests/mocks/mock_llm.py``, which is what ``__file__`` resolves
    to in place *and* through a symlink — and the launcher's own directory,
    where a copier drops the matcher beside the copy.

    Lazy rather than module-level, because ``mocks/__init__.py`` imports this
    file as a library and a copied launcher that never sets
    ``MOCK_LLM_SCENARIO`` must keep working wherever it landed. Not finding it
    is fatal on purpose: the only alternative is a second, drifting copy of the
    matcher here, which is precisely what extracting it prevents.
    """
    import importlib.util

    if "scenario_matcher" in sys.modules:
        return sys.modules["scenario_matcher"]

    here = Path(__file__).resolve()
    searched = [here.parent.parent, here.parent]
    for directory in searched:
        module_path = directory / "scenario_matcher.py"
        if not module_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("scenario_matcher", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["scenario_matcher"] = module
        spec.loader.exec_module(module)
        return module

    print(
        "MOCK SCENARIO ERROR: MOCK_LLM_SCENARIO is set but scenario_matcher.py "
        "is in neither " + " nor ".join(str(path) for path in searched) + ". "
        "That module is the one definition of what a scenario means, so this "
        "binary will not guess at one: a launcher that is a copy of mock_llm.py "
        "rather than a symlink to it must carry scenario_matcher.py beside it.",
        file=sys.stderr,
    )
    sys.exit(EXIT_TEST_AUTHORING_ERROR)


def scenario_turn_index() -> Optional[int]:
    """The per-session turn index ``MOCK_LLM_TURN_INDEX`` declares, if any.

    ``None`` (unset, or unparseable) means "this caller keeps no turn counter",
    which skips the matcher's turn stage entirely. A fake CLI has no session, so
    the number can only come from whoever spawned it.
    """
    raw = os.environ.get("MOCK_LLM_TURN_INDEX")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
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
                    "subagent_type": None,
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
                    "subagent_type": None,
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

#: A scenario declaring ``on_unmatched: fail`` had nothing to answer with. Its
#: own code, distinct from a provider failure (1) and from a test-authoring
#: error (9), because it is neither: the rig is wired correctly and the CLI ran
#: — the *scenario* does not cover the prompt the run actually produced.
EXIT_UNMATCHED_PROMPT = 8


def apply_mock_edits(prompt: str = "") -> List[Dict[str, Any]]:
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
    gate = os.environ.get("MOCK_LLM_EDITS_IF_PROMPT_HAS")
    if gate and gate not in (prompt or ""):
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


#: The opening of ``subagent_prompts.build_schema_repair_prompt``'s write form
#: (a test pins the two together; the mock is standalone and imports nothing).
REPORT_REASK_MARKER = "You are re-sending the report of a batch of edits."


#: A line of the re-ask's "The batch's ids" block: ``- suggestion 1 = `<id>```
#: (``task 1`` for an implement re-ask, fix round 14).
_LISTED_ID = re.compile(r"^- (?:suggestion|fix|task) (\d+) = `([^`]+)`$", re.MULTILINE)

#: An item named by its position, the way the batch prompts' own "Return
#: Format" asks for ("Suggestion 1: …").
_POSITIONAL_ID = re.compile(r"^\s*(?:suggestion|fix|task|item)\s*#?\s*(\d+)\s*$", re.I)
_POSITIONAL_LINE = re.compile(
    r"^[\s*#-]*((?:suggestion|fix)\s+\d+)\s*\**\s*:(.*)$", re.I | re.MULTILINE
)

#: How a positional line opens when its item was NOT applied ("Suggestion 2:
#: skipped — the file does not exist"). A runner restating its report does
#: not claim such an item (fix round 12: the mock restated it as applied).
#: Read at the start of the line's text only, so "added the note (the skip
#: link stays)" is still an applied item.
_NOT_APPLIED = re.compile(
    r"(?:skip(?:ped)?|not\s+applied|could\s*n[o']t\s+(?:be\s+)?appl\w*|"
    r"cannot\s+(?:be\s+)?appl\w*|unable\s+to|did\s*n[o']t\s+appl\w*|"
    r"failed|declined|rejected|no\s+change)\b",
    re.I,
)


def _applied_positions(quoted: str) -> List[str]:
    """The positional names a prose reply says it applied, in order."""
    return [
        name for name, rest in _POSITIONAL_LINE.findall(quoted)
        if not _NOT_APPLIED.match(rest.lstrip(" \t*_-—–:([`"))
    ]


def restate_report(prompt: str) -> str:
    """The report a runner restates from the reply its re-ask quotes.

    Reads only what sits between the re-ask's ``<<<PREVIOUS REPLY`` fences —
    a runner told to "take every value from it" has nothing else — so a quote
    that cut the report's start restates a report without its ids. A prose
    reply with no ``ids`` list restates the items it numbered ("Suggestion 1:
    …") under those names, as opencode/big-pickle did (racing run 4,
    2026-10-05) — only the ones it says it applied: "Suggestion 2: skipped —
    the file does not exist" is not claimed (fix round 12). When the re-ask
    lists the batch's ids, a name that is a position is written as the id
    listed beside that number; any other name is kept as it was quoted.
    """
    opener, closer = "<<<PREVIOUS REPLY\n", "\nPREVIOUS REPLY>>>"
    start = prompt.find(opener)
    end = prompt.rfind(closer)
    quoted = prompt[start + len(opener):end] if 0 <= start < end else ""
    after = prompt[end:] if 0 <= start < end else ""

    def last_list(key: str) -> List[Any]:
        found = re.findall(r'"%s"\s*:\s*(\[[^\]]*\])' % key, quoted)
        try:
            return json.loads(found[-1]) if found else []
        except json.JSONDecodeError:
            return []

    ids = last_list("ids") or _applied_positions(quoted)
    listed = {int(n): item for n, item in _LISTED_ID.findall(after)}

    def spelled(name: Any) -> Any:
        if str(name) in listed.values():
            return name
        position = _POSITIONAL_ID.match(str(name))
        if position and int(position.group(1)) in listed:
            return listed[int(position.group(1))]
        return name

    status = re.findall(r'"status"\s*:\s*"([a-z-]+)"', quoted)
    return json.dumps({
        "status": status[-1] if status else "applied",
        "ids": [spelled(name) for name in ids],
        "files_touched": last_list("files_touched"),
        "summary": "restated from the previous reply",
    })


def write_batch_result(applied: List[Dict[str, Any]], prompt: str = "") -> str:
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
    failing = os.environ.get("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS")
    if failing and failing in (prompt or ""):
        # One batch of a multi-batch run fails (a check that found errors)
        # while its siblings keep the derived result: the shape a stopped
        # implement is tested against.
        return json.dumps({
            "status": "failed",
            "ids": [],
            "files_touched": [],
            "summary": os.environ.get("MOCK_LLM_WRITE_FAIL_SUMMARY")
            or "the check found errors",
            "notes": os.environ.get("MOCK_LLM_WRITE_FAIL_NOTES") or "",
        })
    touched = [entry["path"] for entry in applied if entry["mode"] != "delete"]
    touched += [entry["path"] for entry in applied if entry["mode"] == "delete"]
    try:
        ids = json.loads(os.environ.get("MOCK_LLM_WRITE_IDS") or "[]")
    except json.JSONDecodeError:
        ids = []
    checking = os.environ.get("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS")
    if checking and checking in (prompt or "") and not applied:
        check = json.loads(os.environ.get("MOCK_LLM_WRITE_CHECK") or json.dumps({
            "passed": False,
            "findings": [
                "npx html-validate index.html: index.html:1 doctype must be lowercase",
                "npx html-validate index.html: index.html:9 aria-label misused",
            ],
        }))
        return json.dumps({
            # `failed` beside findings is how racing run 3's runners answered
            # a check that ran and failed (fix round 12).
            "status": os.environ.get("MOCK_LLM_WRITE_CHECK_STATUS") or "no-op",
            "ids": [str(item) for item in ids] if isinstance(ids, list) else [],
            "files_touched": [],
            "summary": "the checks ran",
            "check": check,
        })
    derived = {
        "status": "applied" if applied else "no-op",
        "ids": [str(item) for item in ids] if isinstance(ids, list) else [],
        "files_touched": touched,
        "summary": (
            f"edited {len(touched)} file(s): {', '.join(touched)}"[:200]
            if touched else "nothing needed changing"
        ),
    }
    # Fix round 18: the items a batch declines (MOCK_LLM_WRITE_DECLINED, a
    # JSON list of {"id", "reason"}), each only on a batch whose prompt
    # names its id — the result contract spells every id of the batch.
    declined = json.loads(os.environ.get("MOCK_LLM_WRITE_DECLINED") or "[]")
    if declined:
        derived["declined"] = [
            entry for entry in declined
            if isinstance(entry, dict) and str(entry.get("id")) in (prompt or "")
        ]
    # Fix round 14: a code batch reporting the checks it ran on its edits
    # (MOCK_LLM_WRITE_BATCH_CHECK, JSON), only on a batch whose prompt holds
    # MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS when that is set.
    own_check = os.environ.get("MOCK_LLM_WRITE_BATCH_CHECK")
    only_if = os.environ.get("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS")
    if own_check and applied and (not only_if or only_if in (prompt or "")):
        derived["check"] = json.loads(own_check)
    return json.dumps(derived)


def default_response_content(provider: str) -> str:
    """The generic answer an invocation nothing else claimed falls back to.

    Kept here rather than in ``scenario_matcher`` on purpose: it is derived from
    the *invoker's* identity (``argv[0]``), which is the one thing the shared
    matcher's API may not depend on — the HTTP stub has no ``argv[0]`` to read.
    It is also the reason ``on_unmatched: fail`` exists: this list is shaped
    enough to look like a passing review phase while proving nothing.
    """
    return json.dumps([
        {
            "id": f"mock-{provider}-1",
            "type": "suggestion",
            "description": f"Mock suggestion from {provider}",
            "importance": "MEDIUM",
            "location": "mock_file.py:10"
        }
    ])


def resolve_scenario_content(prompt: str) -> Optional[str]:
    """Answer ``prompt`` from ``MOCK_LLM_SCENARIO``, or ``None`` for no answer.

    Everything about *what a scenario means* — the YAML shape, fixture paths
    relative to the scenario file, the pattern -> turn -> fallback order — lives
    in ``scenario_matcher``, so this function is only the env-to-parameter
    adapter the shared module deliberately refuses to be.

    Raises:
        ScenarioRefused: the scenario declares ``on_unmatched: fail`` and had
            nothing to answer with (or named a fixture that is not on disk).
    """
    scenario_path = os.environ.get("MOCK_LLM_SCENARIO")
    if not scenario_path:
        return None

    matcher = _scenario_matcher()
    try:
        scenario = matcher.load_scenario(scenario_path)
        if scenario is None:
            return None
        resolution = scenario.resolve(
            prompt,
            provider=get_provider_name(),
            turn_index=scenario_turn_index(),
        )
    except matcher.ScenarioError as exc:
        raise ScenarioRefused(str(exc)) from exc
    return resolution.content if resolution is not None else None


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

    Raises:
        ScenarioRefused: see :func:`resolve_scenario_content`. Note this is
            raised *past* MOCK_LLM_CONFIG and the default: a scenario that
            declared it must answer is not silently outranked by a looser
            source lower in the precedence list.
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
    scenario_content = resolve_scenario_content(prompt)
    if scenario_content is not None:
        return scenario_content

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
    return default_response_content(get_provider_name())


#: The first line of the file block the three shipped review prompts render for
#: every adapter declaring ``supports_file_output`` (``llm_client``'s
#: ``FILE_OUTPUT_LOCATION``). Matching the SHIPPED instruction — rather than the
#: ``OUTPUT_FILE:`` directive no orchestrator has ever emitted — is what makes
#: the file channel load-bearing on this lane: before this, every fake-lane
#: result arrived through `invoke_with_file_output`'s stdout fallback, so a
#: provider that cannot write a file was indistinguishable from one that can and
#: the whole primary channel was dead code suite-wide.
FILE_INSTRUCTION_RE = re.compile(
    r"^CRITICAL: Write your JSON output to this file: (?P<path>\S.*?)\s*$",
    re.MULTILINE,
)

#: Mock identities whose real adapter declares ``supports_file_output = False``
#: because its CLI runs in a read-only mode (cursor `--mode ask`). Kept as a
#: literal rather than imported from `utils.providers`: this file is a stand-in
#: for a third-party binary and must not depend on the product it fakes — the
#: pin that keeps the two in step is
#: `test_providers.TestDeclaredOutputChannel.CANNOT_WRITE_FILES`, and
#: `test_fake_lane_output_channel.py` asserts this set against it.
#:
#: The other three declared read-only adapters (`aider` `/ask`, `qwen-code`
#: `--max-tool-calls 0`, `vibe` `--agent plan`) have no identity here — all
#: three are `lane: provider` CLIs whose real binary is in the container image —
#: so this set is cursor alone today. `test_fake_lane_output_channel.py` asserts
#: exactly that intersection, in both directions.
#: UPDATED 2026-09-11 (D6): `codex` declares False too and is deliberately NOT
#: here. Its CLI writes wherever bubblewrap can start; the declaration moves the
#: review off a sandbox that a stock Ubuntu 24.04 cannot start, so this identity
#: keeps writing when a prompt asks it to — which is what the real binary does
#: on a working machine. `test_fake_lane_output_channel.py` names that
#: exception (`DECLARED_FALSE_BUT_WRITES_WHEN_ASKED`).
#: UPDATED 2026-08-19: `gemini` joined, and it does NOT refuse in words.
#: Measured (harness-expansion.md §14.7 gap 5): gemini-cli ships no `write_file`
#: tool under this adapter's argv, so a write is answered
#: `Tool "write_file" not found in registry`, nothing is written — and gemini
#: EXITS 0 printing the model's own "JSON written to ..." confirmation as its
#: answer. So the fake lane must not give it cursor's refusal sentence: the
#: dangerous shape here is a run that claims the file exists. Each identity
#: carries the text its real CLI was measured producing.
READ_ONLY_REPLIES = {
    "cursor-agent": (
        "I can't write the requested file while Ask mode is active. "
        "Switch to Agent mode and resend the request."
    ),
    "gemini": (
        "JSON written to the requested output file. The review is complete "
        "and the results have been saved."
    ),
}

READ_ONLY_IDENTITIES = frozenset(READ_ONLY_REPLIES)

#: What cursor-agent really answers when a prompt demands a file write under
#: `--mode ask`. Measured 2026-08-18 on `cursor-agent:gpt-5.6-sol-xhigh`, three
#: identical runs: a **successful** envelope (`"subtype": "success"`,
#: `"is_error": false` — so `exit_zero_failure_markers` does not match it and
#: the slot is not classified as a failure) whose entire `result` is this
#: sentence, and no file anywhere. Reproduced on this lane so that a prompt
#: regression demanding a file from a read-only adapter fails a test here
#: instead of a user's review run.
ASK_MODE_REFUSAL = (
    "I can't write the requested file while Ask mode is active. "
    "Switch to Agent mode and resend the request."
)


def shipped_file_instruction(prompt: str) -> Optional[Path]:
    """The file the SHIPPED review prompt orders the model to write, if any.

    This is the one spelling the product really renders
    (``llm_client.FILE_OUTPUT_LOCATION``), and therefore the only one that means
    "an adapter was asked for the file channel". The refusal gate keys on it
    alone — `MOCK_OUTPUT_PATH` and the legacy `OUTPUT_FILE:` directive are both
    a *test author* saying "put a file here", which is a property of the harness
    rather than of the product's prompt, and a read-only identity must still
    honour those (a fixture that sets one is staging a file, not modelling a
    refusal).
    """
    match = FILE_INSTRUCTION_RE.search(prompt)
    return Path(match.group("path")) if match else None


def demanded_output_path(prompt: str) -> Optional[Path]:
    """The file this prompt orders the model to write, in either spelling."""
    shipped = shipped_file_instruction(prompt)
    if shipped:
        return shipped

    if "OUTPUT_FILE:" in prompt:
        for line in prompt.split("\n"):
            if "OUTPUT_FILE:" in line:
                return Path(line.split("OUTPUT_FILE:")[1].strip())

    return None


def write_output_file(prompt: str, content: str) -> None:
    """Write response content to output file if one was asked for.

    Args:
        prompt: The prompt text (may carry a file instruction).
        content: The response content to write.
    """
    if os.environ.get("MOCK_OUTPUT_PATH"):
        output_path = Path(os.environ["MOCK_OUTPUT_PATH"])
    else:
        output_path = demanded_output_path(prompt)

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

    Format: {"type": "result", "result": "<content>"} plus the rest of the
    nine-key envelope the real CLI emits. `parse_output` unwraps `result` and
    discards everything else, but the `usage` block is what the D3j extractor
    reads (camelCase here, unlike claude-code's snake_case one), so keeping the
    envelope complete is what makes `usage_support = partial` testable. There is
    deliberately no cost field: the real envelope carries none.
    """
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 1234,
        "duration_api_ms": 1234,
        "result": content,
        "session_id": "mock-session-12345",
        "request_id": "mock-request-12345",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 50,
            "cacheReadTokens": 20,
            "cacheWriteTokens": 0,
        },
    })


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


def configured_failure() -> Optional[tuple]:
    """``(message, exit_code)`` from ``MOCK_LLM_CONFIG``'s ``"error"``, or ``None``."""
    config_json = os.environ.get("MOCK_LLM_CONFIG")
    if not config_json:
        return None
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(config, dict) or not isinstance(config.get("error"), str):
        return None
    exit_code = config.get("exit_code", 1)
    return config["error"], exit_code if isinstance(exit_code, int) else 1


def format_failure(provider: str, message: str) -> tuple:
    """``(stdout, stderr)`` of a failed run, in the provider's real shape.

    codex reports a failed turn on stdout as a top-level ``error`` event and a
    ``turn.failed`` event, stderr empty — the codex-cli 0.146.0 invalid-key
    capture (``tests/test_failure_classifier.py``'s ``CODEX_TURN_FAILED_401``)
    and ``tests/fixtures/codex/0.156.1/SCHEMA.md``'s top-level events. Every
    other identity puts the sentence on stderr, with nothing on stdout.
    """
    if provider == "codex":
        events = [
            {"type": "thread.started", "thread_id": "mock-thread-1"},
            {"type": "turn.started"},
            {"type": "error", "message": message},
            {"type": "turn.failed", "error": {"message": message}},
        ]
        return "\n".join(json.dumps(event) for event in events), ""
    return "", message


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


def format_agy_output(content: str) -> str:
    """Format output for the agy provider (Google Antigravity CLI).

    Mirrors `agy --print --output-format json`: ONE JSON object on ONE stdout
    line, the answer in `response`, and `status` carrying SUCCESS/ERROR —
    `AgyProvider.parse_output` reads exactly those two fields and treats any
    non-SUCCESS status as a failure carrying `error`.

    PROVENANCE, because C5's one non-negotiable rule is "captured from the real
    binary, never invented". A live capture of a SUCCESS envelope is **not
    possible**, and the reason is stronger than the one first written here.

    This docstring used to say agy's credentials live in the OS keyring "not in
    `$HOME`", so a print-mode run would spend the developer's own Google quota.
    That claim is RETRACTED — `tests/container/redirect.yaml`'s agy row carries
    the retraction — because it holds only where **D-Bus is reachable**:
    headless, agy logs `Failed to load token from keyring, falling back to file`
    and opens `$HOME/.gemini/antigravity-cli/antigravity-oauth-token`, which is
    seedable. What actually makes a success capture impossible is that **agy's
    auth resolves before its first RPC**: re-measured 2026-08-19 with
    `CLOUD_CODE_URL`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` and
    `GOOGLE_GEMINI_BASE_URL` all pointed at a live stub, the run produced a
    byte-identical failure envelope with ZERO requests in the stub's log. No
    environment variable selects an API-key path a stub could answer, so there is
    no credential-free route to a successful run at any price.

    The shape below is therefore the repo's existing record of real runs —
    `AgyProvider`'s docstring and `tests/test_providers.TestAgyProvider._envelope`
    — with every key corroborated **offline** against the installed binary itself
    (agy 1.1.13, `agy --version`), whose Go JSON struct tags carry `ConversationID
    json:"conversation_id"`, `Status json:"status"`, `Response json:"response"`,
    `DurationSeconds json:"duration_seconds"`, `NumTurns json:"num_turns"` (which
    occurs exactly once in the whole binary) and `Usage json:"usage"`.

    It is corroborated a second way now: `tests/fixtures/agy/` (captured
    2026-08-19, agy 1.1.15, `--network none`, no credential) holds one graded row
    and eleven written reasons, and that row is the real binary's **ERROR**
    envelope: `conversation_id, status, response, error, duration_seconds,
    num_turns, usage` on one line, in that order — this formatter's six keys in
    the same order, plus the `error` a SUCCESS envelope does not carry. A failure
    capture cannot supply a success body, but it does confirm the field set and
    the ordering below against 1.1.15 rather than 1.1.13.

    The `usage` block is inert here — `AgyProvider` declares `usage: none` and
    implements no `parse_usage` — and is present only because the real envelope
    carries it. Nothing in this repo reads it, and a future extractor must be
    written against a real capture, not against these numbers.
    """
    return json.dumps({
        "conversation_id": "conv-mock-agy",
        "status": "SUCCESS",
        "response": content,
        "duration_seconds": 4.2,
        "num_turns": 1,
        "usage": {"input_tokens": 50, "output_tokens": 50},
    })


def format_output(provider: str, content: str) -> str:
    """Format output according to provider's expected wire format.

    Args:
        provider: The provider name.
        content: The response content.

    Returns:
        Formatted output string.

    NOTE the fallback: an unknown provider is answered in **cursor-agent's** wire
    format rather than refused, so a half-added identity hands its adapter a
    shape its real CLI never emits and the row still goes green.
    ``tests/test_fake_lane_roster.py`` asserts every rostered identity is in this
    table for exactly that reason.
    """
    formatters = {
        "cursor-agent": format_cursor_agent_output,
        "gemini": format_gemini_output,
        "opencode": format_opencode_output,
        "codex": format_codex_output,
        "kilocode": format_kilocode_output,
        "claude-code": format_claude_code_output,
        "agy": format_agy_output,
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
        # `-f`/`--yolo` are still what the real CLI accepts, so the mock still
        # accepts them — but the adapter no longer emits them (M1/§3.1).
        parser.add_argument("-f", "--force", "--yolo", action="store_true")
        # What `CursorAgentProvider.build_command` really emits now. Both are
        # load-bearing twice over: an unrecognised flag is an argparse **exit 2
        # that reads as a provider failure** (the same trap this file already
        # records for opencode's `--agent`), and M1's §5.4.1 capability gate
        # sniffs the *help text* of whichever binary the lane resolves — this
        # one included — so a flag missing from `--help` is silently dropped and
        # the fixtures then assert an argv the real CLI would never emit.
        # argparse renders both into `--help` from these two lines; the help
        # strings are there so a truncated rendering still shows the tokens.
        parser.add_argument(
            "--trust",
            action="store_true",
            help="Trust the current workspace without prompting",
        )
        parser.add_argument(
            "--mode",
            choices=["plan", "ask"],
            default=None,
            help="Start in the given execution mode (plan | ask)",
        )
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
        # `-c key=value` config overrides. Real codex takes these (the Phase 1
        # spike redirected the binary with `-c model_providers.*`), and the
        # adapter emits `-c model_reasoning_effort=<effort>` for every
        # `model:effort` spec — `codex:gpt-5.5:high` is configured in
        # providers.yaml — so an argv the product really builds was an argparse
        # exit 2 here until this line existed.
        exec_parser.add_argument("-c", "--config", action="append", default=[])
        # M2: both are emitted UNCONDITIONALLY by `CodexProvider.build_command`,
        # on every profile and both transports, so every mock codex dispatch
        # carries them. Missing here they are an argparse exit 2 —
        # indistinguishable from a provider failure once it is behind a
        # subprocess, which is the trap this file's comments record twice.
        # `--ephemeral` suppresses the rollout file; `--skip-git-repo-check`
        # opens codex's trust gate (real codex outside a git repo exits 1 with
        # zero requests without it). Both are booleans and neither takes a value.
        exec_parser.add_argument("--ephemeral", action="store_true")
        exec_parser.add_argument("--skip-git-repo-check", dest="skip_git_repo_check",
                                 action="store_true")
        # §4.2's `--output-schema <FILE>`. codex's half of the contract asymmetry
        # takes a PATH, not inline JSON. It carries no §5.4.1 gate (verified at
        # the declared floor 0.144.0), so the help text is not load-bearing the
        # way claude's is — but the parser entry is: without it the argv the
        # product really builds is an argparse exit 2 that reads as a provider
        # failure, which is the trap this file's comments record twice.
        exec_parser.add_argument(
            "--output-schema",
            dest="output_schema",
            default=None,
            help="Path to a JSON Schema file constraining the final response",
        )
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
        # §3.6's `--fallback-model <m1,m2>`, emitted only from a configured
        # `providers.claude-code.fallback_models` list. It has to be here TWICE
        # over: as a parser entry, or the argv the product really builds is an
        # argparse exit 2 that reads as a provider failure; and in the HELP TEXT,
        # because the flag carries a §5.4.1 capability gate that sniffs this
        # mock's own `--help` — a flag missing from the help is silently dropped
        # one step earlier, and the fixture then asserts an argv the real CLI
        # would never emit. argparse renders the `help=` below into `--help`, so
        # the two cannot drift apart.
        parser.add_argument(
            "--fallback-model",
            dest="fallback_model",
            default=None,
            help=(
                "Enable automatic fallback to specified model(s) when the "
                "default model is overloaded or not available. Accepts a "
                "comma-separated list to try each in order. (only works with "
                "--print)"
            ),
        )
        # §4.2's `--json-schema <inline json>`, emitted only when a caller
        # attaches a schema. Exactly the same two-places rule as
        # `--fallback-model` above, for exactly the same two reasons: as a parser
        # entry, or an argv the product really builds is an argparse exit 2 that
        # reads as a provider failure; and in the HELP TEXT, because the flag
        # carries a §5.4.1 capability gate that sniffs THIS mock's `--help` — a
        # flag missing from the help is silently dropped one step earlier, and
        # the fixture then asserts an argv the real CLI would never emit.
        parser.add_argument(
            "--json-schema",
            dest="json_schema",
            default=None,
            help="JSON Schema for structured output (only works with --print)",
        )
        parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        return args, args.prompt

    elif provider == "kilocode":
        # `KiloCodeProvider.build_command` emits
        #   kilocode run --auto -m <provider/model> [--variant <effort>] <prompt>
        # and the real CLI agrees: tests/container/runs/surface/cli-surface.json
        # records subcommand `run` and options `--auto` / `-m` probed against
        # kilocode 7.4.19. This parser previously accepted neither `run` nor
        # `-m`, so the adapter's own argv was an argparse exit 2 here —
        # indistinguishable from a provider failure. `--json` is a real flag the
        # adapter deliberately never emits (it streams every reasoning token as
        # a full-content-so-far line); it stays accepted because it is a CLI
        # feature, not a deprecation trap like codex's `--full-auto`.
        parser = argparse.ArgumentParser(description="Mock kilocode CLI")
        subparsers = parser.add_subparsers(dest="command")
        run_parser = subparsers.add_parser("run")
        run_parser.add_argument("--auto", action="store_true")
        run_parser.add_argument("--json", action="store_true")
        run_parser.add_argument("-m", "--model", default="")
        run_parser.add_argument("--variant", default=None)
        run_parser.add_argument("prompt", nargs="?", default="")
        args = parser.parse_args()
        if args.command != "run":
            print("ERROR: Only 'run' subcommand is supported", file=sys.stderr)
            sys.exit(1)
        return args, args.prompt

    elif provider == "agy":
        # `AgyProvider.build_command` emits, in this order:
        #   agy --new-project --dangerously-skip-permissions
        #       --output-format json --print-timeout <duration> --model "<Display Name>"
        #       -p "<prompt>"
        # Every flag verified against `agy --help` (agy 1.1.13, offline), where
        # `-p` is the short alias of `--print`/`--prompt` and carries the prompt
        # as its VALUE — agy has no positional prompt at all, which is why this
        # parser declares none.
        #
        # allow_abbrev=False is load-bearing, not tidiness. agy's own Go flag
        # parser does no abbreviation matching, and argparse's default would
        # silently accept three real agy flags that no profile may emit:
        # `--mode accept-edits|plan` as an abbreviation of `--model` (the
        # adapter declares no agent profile, so an execution-mode flag here is a
        # regression), and `--print`/`--prompt` as abbreviations of
        # `--print-timeout`. `--continue`/`-c`/`--conversation` are likewise
        # undeclared on purpose: `--new-project` exists precisely because print
        # mode otherwise reuses a previous project whose workspace may be bound
        # to a DIFFERENT directory, so a resumed conversation is the silent
        # wrong-repo bug the adapter documents.
        parser = argparse.ArgumentParser(
            description="Mock agy CLI", allow_abbrev=False
        )
        parser.add_argument("--new-project", dest="new_project", action="store_true")
        parser.add_argument(
            "--dangerously-skip-permissions", dest="skip_permissions",
            action="store_true",
        )
        parser.add_argument("--output-format", dest="output_format", default="json")
        parser.add_argument("--print-timeout", dest="print_timeout", default=None)
        parser.add_argument("--model", default=None)
        # L11 #3: `agy --effort low|medium|high` is a real flag (`agy --help`,
        # agy 1.1.13) that the adapter now emits for a `model:effort` spec. Note
        # `allow_abbrev=False` above means this does NOT open the door to
        # `--eff`; and it must stay distinct from `--model`, which is why the
        # adapter passes the level on its own flag rather than in the name.
        parser.add_argument("--effort", default=None)
        parser.add_argument("-p", "--print", "--prompt", dest="prompt", default="")
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
    "codex": "codex-cli 0.160.1",
    "opencode": "1.18.35",
    # Moved 2.1.224 -> 2.1.232 with the adapter, 2026-08-20: the shipped
    # claude-code fixtures were captured at 2.1.232 and the image installs it
    # (cli-versions.lock). This entry and `ClaudeCodeProvider.verified_version`
    # are one fact written twice, and test_subagent_executor.py::
    # test_the_mock_version_matches_each_verified_version is what keeps them so.
    "claude-code": "2.1.232 (Claude Code)",
    "cursor-agent": "2026.02.01-abcdef",
    "gemini": "0.9.0",
    "kilocode": "1.4.0",
    # Captured from the installed binary: `agy --version` prints a bare dotted
    # version and nothing else (agy 1.1.13). `AgyProvider` declares no
    # `version_cmd`, so no preflight probes this today; the entry exists because
    # the fallback below is "0.0.0", which would fail a minimum-version gate the
    # moment agy grows one.
    "agy": "1.1.13",
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
#:
#: Entries are added for an adapter declaring an ``auth_check`` and for NO other:
#: an identity with a fabricated probe here would answer a subcommand its real
#: CLI does not have. ``agy`` is the current example of the empty case — its auth
#: is a Google OAuth session (the OS keyring where D-Bus is reachable, a token
#: file under ``$HOME/.gemini/antigravity-cli/`` where it is not) with no
#: `login status`-class subcommand at all, so `AgyProvider.auth_check` is empty
#: and this table stays silent about it. That empty case is not a missing part:
#: ``test_fake_lane_roster.py``'s part 5 asserts the absence in both directions.
AUTH_PROBE_ARGS = {
    "codex": ("login", "status"),
    "opencode": ("auth", "list"),
    # M13 (§5.2): cursor-agent's probe is `status --format json`, not `about`.
    # `about` round-trips the network for its `userEmail`/`subscriptionTier`, so
    # it cannot tell offline-logged-in from logged-out; `status` reads the local
    # token. The flag is part of the argv because the verdict is a JSON *field*
    # — this CLI exits 0 in every auth state.
    "cursor-agent": ("status", "--format", "json"),
}

#: What each probe prints when the CLI is logged in. cursor-agent's row is a
#: JSON object rather than a sentence for the same reason its argv carries
#: `--format json`: its adapter reads a field and never an exit code, so a
#: prose "Logged in" would be answered `None` (unknown) by the real classifier.
AUTH_OK_STRINGS = {
    "codex": "Logged in using ChatGPT",
    "opencode": "Credentials\n  anthropic  oauth",
    "cursor-agent": '{"isAuthenticated": true, "message": "Logged in"}',
}


def is_auth_probe(argv: List[str]) -> bool:
    """Whether this invocation is an adapter's `auth_check`, not a dispatch.

    Both sides are reduced to their positionals before comparison, because a
    declared probe may carry flags (`status --format json`) while this function
    is handed a real argv where those flags sit anywhere. Reducing only the
    argv would make such a probe unrecognisable — and an unrecognised probe is
    parsed as a *prompt* and answered with a fixture.
    """
    positional = tuple(arg for arg in argv if not arg.startswith("-"))
    return any(
        positional[: len(expected)] == expected
        for expected in (
            tuple(arg for arg in probe if not arg.startswith("-"))
            for probe in AUTH_PROBE_ARGS.values()
        )
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


#: What real bubblewrap prints on a stock Ubuntu 24.04 with
#: `kernel.apparmor_restrict_unprivileged_userns=1` and no profile granting
#: /usr/bin/bwrap the `userns` right — measured, not paraphrased.
BWRAP_BROKEN_LINE = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"


def run_codex_sandbox(argv: List[str]) -> int:
    """Emulate `codex sandbox [-c key=value…] -- <cmd…>` (D6 step 3).

    The real subcommand starts bubblewrap under the named policy and runs
    ``<cmd>`` inside it. Here the command runs as an ordinary child in the
    inherited cwd — a mock has no sandbox to offer — and its exit code is the
    verdict, unless ``MOCK_LLM_SANDBOX_BROKEN=1`` replays the machine this was
    written on: the bwrap line on stderr, exit 1, and the command never runs.

    ``MOCK_LLM_SANDBOX_RUNS_FULL_ACCESS=1`` (plan step 3b, 2026-10-02) refines
    the replay the way the real binary behaves: ``danger-full-access`` needs no
    bubblewrap, so under that policy the command runs (inside whatever jail the
    plugin wrapped this process in), while ``workspace-write`` and
    ``read-only`` still die on the bwrap line. Opt-in, so every test written
    against the plain replay keeps its walk.
    """
    if os.environ.get("MOCK_LLM_SANDBOX_BROKEN") == "1":
        full_access = any(
            "sandbox_mode" in arg and "danger-full-access" in arg for arg in argv
        )
        if not (full_access and os.environ.get("MOCK_LLM_SANDBOX_RUNS_FULL_ACCESS") == "1"):
            print(BWRAP_BROKEN_LINE, file=sys.stderr)
            return 1
    if "--" not in argv:
        print("error: codex sandbox needs `-- <command>`", file=sys.stderr)
        return 2
    command = argv[argv.index("--") + 1:]
    if not command:
        print("error: codex sandbox needs a command after `--`", file=sys.stderr)
        return 2
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL).returncode
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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


# ---------------------------------------------------------------------------
# MOCK_LLM_PLUGIN: the supported injection point for a ONE-OFF fake adapter.
#
# A probe that needs a fake provider to behave unusually used to have exactly
# one move available: edit this file. That is not a hypothetical — it happened
# (plans/harness-invocation-improvements.md §13.1), it replaced 1275 lines with
# 40, and it broke tests/test_fake_lane_roster.py and
# tests/test_mock_adapter_roundtrip.py by collection error until `git checkout`
# undid it. So the injection point is a **file outside the tree**, named by an
# environment variable, and everything about it is deliberately narrow:
#
#   * it is honoured only under MULTI_LLM_TEST_MODE=1 — the same gate the binary
#     itself refuses to run without, checked BEFORE the import, so a stray
#     variable in a shell can never make a real invocation import a scratch
#     module;
#   * it is imported AFTER every table above is built, so an override replaces a
#     value rather than being replaced by one;
#   * it may replace exactly four seams and no others. A name it cannot honour
#     is a loud refusal (exit 9, the code this mock already reserves for a rig
#     mistake) rather than a silent no-op, because a probe that quietly did
#     nothing is the same failure the incident was: an assertion about a mock
#     talking to itself.
#
# What it is NOT: a way to add an identity. An identity is five parts (see this
# module's docstring and tests/test_fake_lane_roster.py) and belongs here, in
# the tree, asserted. A plugin is one throwaway probe's private business.
# ---------------------------------------------------------------------------

#: Exactly what a plugin may override. Each is looked up through module globals
#: at call time, so replacing the binding is enough.
PLUGIN_SEAMS = (
    "parse_provider_args",
    "format_output",
    "get_response_content",
    "VERSION_STRINGS",
)

#: The name the imported plugin module is given. Used to tell a function the
#: plugin **defined** from one it merely imported (reusing a formatter from this
#: file is legitimate; redefining one is a seam that does not exist).
PLUGIN_MODULE_NAME = "mock_llm_plugin"


class MockPluginError(Exception):
    """MOCK_LLM_PLUGIN named something this hook cannot honour.

    Always a test-authoring error (exit 9), never a simulated provider failure:
    the probe asked for behaviour it will not get, and finding that out from a
    wrong-looking assertion three files away is exactly what this hook exists to
    prevent.
    """


def _mock_owned_names() -> set:
    """Every non-seam name this module owns, for the collision check below."""
    owned = set()
    for name, value in vars(sys.modules[__name__]).items():
        if name.startswith("_") or name in PLUGIN_SEAMS:
            continue
        if isinstance(value, types.ModuleType):
            continue
        if name.isupper() or getattr(value, "__module__", None) == __name__:
            owned.add(name)
    return owned


def load_mock_plugin() -> Optional[types.ModuleType]:
    """Import ``MOCK_LLM_PLUGIN``, or return None when there is nothing to load.

    Returns None both when the variable is unset and when
    ``MULTI_LLM_TEST_MODE`` is not ``"1"`` — the gate is checked *before* the
    import, so a plugin's module body (arbitrary code) never runs outside test
    mode.

    Raises:
        MockPluginError: the path does not exist, or the module raised while
            importing. Both are the probe's own mistake and must be loud.
    """
    plugin_env = os.environ.get("MOCK_LLM_PLUGIN")
    if not plugin_env:
        return None
    if os.environ.get("MULTI_LLM_TEST_MODE") != "1":
        return None

    plugin_path = Path(plugin_env)
    if not plugin_path.is_file():
        raise MockPluginError(
            f"MOCK_LLM_PLUGIN={plugin_env} is not a file. A one-off fake adapter "
            f"is a throwaway .py module (normally under a pytest tmp_path), "
            f"never an edit to {Path(__file__).name}."
        )

    spec = importlib.util.spec_from_file_location(PLUGIN_MODULE_NAME, plugin_path)
    if spec is None or spec.loader is None:
        raise MockPluginError(
            f"MOCK_LLM_PLUGIN={plugin_env} is not importable as a Python module."
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - the probe's own code, reported as-is
        raise MockPluginError(
            f"MOCK_LLM_PLUGIN={plugin_env} raised while importing: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return module


def _plugin_declared_names(plugin: types.ModuleType) -> set:
    """The public names a plugin **defines**, minus the seams.

    A callable imported from elsewhere is excluded, so a plugin may legitimately
    reuse this module's own helpers (``from mock_llm import
    format_codex_output``) while *redefining* one is caught.
    """
    declared = set()
    for name, value in vars(plugin).items():
        if name.startswith("_") or name in PLUGIN_SEAMS:
            continue
        if isinstance(value, types.ModuleType):
            continue
        if callable(value) and getattr(value, "__module__", None) != PLUGIN_MODULE_NAME:
            continue
        declared.add(name)
    return declared


def install_mock_plugin(
    namespace: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Override the seams ``MOCK_LLM_PLUGIN`` declares; return their names.

    Args:
        namespace: The mapping to write the overrides into. Defaults to this
            module's own globals (which is what ``main`` wants); tests pass a
            copy so they can assert what moved without patching the module.

    Returns:
        The overridden seam names, in ``PLUGIN_SEAMS`` order. Empty when no
        plugin is configured (or test mode is off).

    Raises:
        MockPluginError: the plugin declares no seam at all (a probe that
            silently does nothing), or declares a name this hook cannot honour.
    """
    plugin = load_mock_plugin()
    if plugin is None:
        return []

    declared = _plugin_declared_names(plugin)
    unhonourable = sorted(declared & _mock_owned_names())
    if unhonourable:
        raise MockPluginError(
            f"MOCK_LLM_PLUGIN defines {', '.join(unhonourable)}, which this hook "
            f"does not override. A plugin may replace exactly: "
            f"{', '.join(PLUGIN_SEAMS)}. Anything else has to be expressed "
            f"through one of those four (or belong in {Path(__file__).name} as a "
            f"real identity, with the five parts an identity needs)."
        )

    overridden = [name for name in PLUGIN_SEAMS if hasattr(plugin, name)]
    if not overridden:
        raise MockPluginError(
            f"MOCK_LLM_PLUGIN overrides nothing: it defines none of "
            f"{', '.join(PLUGIN_SEAMS)}, so the run would silently be the "
            f"ordinary mock. Check the seam name for a typo."
        )

    target = globals() if namespace is None else namespace
    for name in overridden:
        target[name] = getattr(plugin, name)
    return overridden


#: Go duration spellings `AgyProvider.build_command` can emit: whole seconds
#: (`1170s`, what §9.6.c renders) plus the `19m30s` fallback form the plan keeps
#: in reserve if the real binary ever rejects bare seconds.
_GO_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_go_duration(text: Optional[str]) -> Optional[int]:
    """Seconds in a Go duration string, or ``None`` when it is not one."""
    if not text:
        return None
    match = _GO_DURATION_RE.match(str(text))
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def run_print_timeout_stub(provider: str, args: Any) -> Optional[int]:
    """The two agy stubs behind ``--print-timeout`` (§9.6.e), or ``None``.

    Selected with ``MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR``:

    * ``honour`` — the child agy is supposed to be. It starts a descendant,
      reaches its own ``--print-timeout`` cutoff, shuts that descendant down and
      writes its ``status: ERROR`` envelope, all *before* the parent's
      ``subprocess.run`` deadline. The harness then reports agy's own reason
      instead of a bare TIMEOUT, and nothing is left running. This is the stub
      L5 makes reachable: it only works because the derived inner cutoff is
      strictly below the outer one.
    * ``ignore`` — the disobedient twin. It starts the same descendant and then
      sleeps past *any* outer deadline. ``subprocess.run``'s kill reaches the
      leader only, so the descendant survives — the orphan owned by
      ``plans/timeout-orphans-and-reaggregate-determinism.md``. L5 cannot fix
      this one, which is why its test ships xfail.

    ``MOCK_LLM_PID_FILE`` receives the descendant's pid, so a test can assert
    liveness rather than infer it. Returns ``None`` when neither stub is
    selected, so the normal answer path continues untouched.
    """
    import subprocess

    behaviour = os.environ.get("MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR")
    if not behaviour:
        return None
    if behaviour not in ("honour", "ignore"):
        print(
            f"MOCK STUB ERROR: MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR={behaviour!r} "
            f"is not 'honour' or 'ignore'",
            file=sys.stderr,
        )
        return EXIT_TEST_AUTHORING_ERROR
    if provider != "agy":
        # A silent no-op here would let a test believe it exercised the stub.
        print(
            f"MOCK STUB ERROR: MOCK_LLM_PRINT_TIMEOUT_BEHAVIOUR is agy-only "
            f"(--print-timeout is agy's flag); got provider {provider!r}",
            file=sys.stderr,
        )
        return EXIT_TEST_AUTHORING_ERROR

    rendered = getattr(args, "print_timeout", None)
    inner_s = parse_go_duration(rendered)
    if behaviour == "honour" and inner_s is None:
        print(
            f"MOCK STUB ERROR: --print-timeout {rendered!r} is not a Go "
            f"duration, so the honouring stub has no cutoff to honour",
            file=sys.stderr,
        )
        return EXIT_TEST_AUTHORING_ERROR

    # A real descendant, so "no child survives" is observed and not inferred.
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"]
    )
    pid_file = os.environ.get("MOCK_LLM_PID_FILE")
    if pid_file:
        Path(pid_file).write_text(str(sleeper.pid), encoding="utf-8")

    if behaviour == "ignore":
        time.sleep(600)
        return 0

    time.sleep(inner_s)
    sleeper.terminate()
    try:
        sleeper.wait(timeout=5)
    except subprocess.TimeoutExpired:
        sleeper.kill()
        sleeper.wait()
    # agy's own envelope, in agy's own shape: `AgyProvider.parse_output` reads
    # `status` and reports `error` as the failure reason.
    print(json.dumps({
        "conversation_id": "conv-mock-agy",
        "status": "ERROR",
        "response": "",
        "error": f"print-timeout of {rendered} reached",
        "duration_seconds": float(inner_s),
        "num_turns": 1,
        "usage": {"input_tokens": 50, "output_tokens": 0},
    }))
    return 0


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

    # A one-off fake adapter (M22), installed before ANY seam is consulted —
    # `--version` reads VERSION_STRINGS below, so a later install would be a
    # seam the plugin was promised and did not get.
    try:
        install_mock_plugin()
    except MockPluginError as exc:
        print(f"MOCK PLUGIN ERROR: {exc}", file=sys.stderr)
        return EXIT_TEST_AUTHORING_ERROR

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

    # codex's model-free sandbox self-test (D6 step 3): `codex sandbox -c
    # 'sandbox_mode="…"' -- <cmd>` runs <cmd> under the same bubblewrap policy
    # a dispatch would use. Not a dispatch either — it must not land in the
    # call log — so it is answered here, before the exec parser.
    if provider == "codex" and sys.argv[1:2] == ["sandbox"]:
        return run_codex_sandbox(sys.argv[2:])

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

    # A write batch's report re-ask (the ladder's rule 7): a well-behaved
    # runner restates its report and edits nothing.
    if REPORT_REASK_MARKER in (prompt or "") and os.environ.get("MOCK_LLM_EDITS"):
        applied = (
            apply_mock_edits() if os.environ.get("MOCK_LLM_REPAIR_EDITS") == "1"
            else []
        )
        content = os.environ.get("MOCK_LLM_REPAIR_RESULT") or (
            restate_report(prompt)
            if os.environ.get("MOCK_LLM_REPAIR_RESTATE") == "1"
            else write_batch_result(applied)
        )
        print(format_output(provider, content))
        return 0

    # The write path (D3g): edits land BEFORE the result is printed, so the
    # executor's post-run snapshot comparison sees exactly what a real
    # edit-capable dispatch would have left behind.
    applied_edits = apply_mock_edits(prompt)
    gated_write = bool(
        (os.environ.get("MOCK_LLM_EDITS") and os.environ.get("MOCK_LLM_EDITS_IF_PROMPT_HAS"))
        or os.environ.get("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS")
        or os.environ.get("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS")
    )
    if applied_edits or gated_write or os.environ.get("MOCK_LLM_WRITE_RESULT") or os.environ.get(
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
            content = write_batch_result(applied_edits, prompt)
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

    # MOCK_LLM_CONFIG's "error": the run fails in the provider's own shape.
    failure = configured_failure()
    if failure is not None:
        message, exit_code = failure
        stdout, stderr = format_failure(provider, message)
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        return exit_code

    # L5 (§9.6.e): the two agy stubs that decide whether a child outlives the
    # outer deadline. Placed after the call log so the argv is recorded either
    # way, and before any response formatting because neither stub answers the
    # prompt — they exist to be *slow* in one of two specific ways.
    stub_exit = run_print_timeout_stub(provider, args)
    if stub_exit is not None:
        return stub_exit

    # Get response content. A scenario that declared `on_unmatched: fail` and
    # found nothing to say is a **failed invocation**, not a generic answer:
    # both invokers of the shared matcher inherit that strictness together, so
    # the fake binary exits non-zero here for the same prompt the HTTP stub
    # would answer with a machine-readable error. The reason names the scenario
    # and the prompt's sha256 — never the prompt body.
    try:
        content = get_response_content(prompt)
    except ScenarioRefused as exc:
        print(f"MOCK SCENARIO ERROR: {exc}", file=sys.stderr)
        return EXIT_UNMATCHED_PROMPT

    # A read-only identity handed a "write this file" instruction answers the
    # way the real CLI was measured answering it: a SUCCESSFUL envelope carrying
    # a refusal sentence, and no file. This is the fake lane's half of the
    # `supports_file_output` contract — the shipped prompts render the stdout
    # block for these identities, so reaching this branch means a prompt, a
    # template or a declaration regressed, and the row that regressed it fails
    # here rather than 450 s into a user's review.
    if provider in READ_ONLY_IDENTITIES and shipped_file_instruction(prompt):
        print(format_output(provider, READ_ONLY_REPLIES[provider]))
        return 0

    # Write to output file if specified
    write_output_file(prompt, content)

    # Format and print output according to provider's wire format
    output = format_output(provider, content)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
