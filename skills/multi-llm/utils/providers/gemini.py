"""Gemini CLI provider implementation."""
import json
import shutil
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, StructuredSchema


class GeminiProvider(LLMProvider):

    #: MEASURED FALSE 2026-08-19, closing harness-expansion.md §14.7 gap 5.
    #:
    #: This adapter shipped the default `True` and was the last unmeasured one.
    #: Under its own argv — no `--yolo`, no `--approval-mode` — gemini-cli 0.25.2
    #: advertises nine tools and **`write_file` is not among them** (nor
    #: `replace`, nor `run_shell_command`). Handed a `write_file` functionCall it
    #: answers `Tool "write_file" not found in registry`, writes nothing, and
    #: **exits 0 printing the model's own "JSON written to ..." confirmation as
    #: its answer** — so the fan-out recorded a sentence about a file instead of
    #: the review, with `success: True`. That is a worse failure than the four
    #: argv-restricted adapters, which at least refuse in words.
    #:
    #: The restriction is a DEFAULT, not a flag, so the argv guard in
    #: tests/test_provider_housekeeping.py could not see it; that file now carries
    #: NO_WRITE_TOOL_AT_ALL for exactly this shape. Probe:
    #: plans/harness-expansion/probe-out/gemini/gap5_probe.py.
    supports_file_output = False
    """Provider for Gemini CLI tool.

    Output shape on stdout: pretty-printed
    ``{"session_id": …, "response": "<the answer>", "stats": {…}}``.
    :meth:`parse_output` unwraps ``response`` and parses it as JSON, falling back
    to :func:`extract_json_from_text` for JSON embedded in prose.

    Everything below was **measured** by the §6 step-5 probe on 2026-08-19
    against ``gemini 0.25.2`` driven offline at a localhost stub
    (``--network none``, no credential); the raw captures are
    ``tests/fixtures/gemini/`` and its README carries the full record.

    **stdout is empty on every failure.** 401, 429, 404, 400 context-length,
    5xx-forever, a truncated stream, a zero-byte response and a 200 carrying an
    error object all produce **0 bytes on stdout** and put the whole diagnosis on
    stderr — the exact inverse of the opencode trap. Since ``parse_output`` reads
    only ``stdout``, the operator-facing error for all eight is the identical,
    contentless ``JSON parse error: Expecting value: line 1 column 1 (char 0)``.
    ``classify_failure`` does read both streams, which is why four of them still
    land on a category.

    **stderr carries a machine-readable envelope whose message is destroyed.**
    After the JS prose and a Node stack, gemini prints
    ``{"session_id", "error": {"type", "message", "code"}}`` — and ``message`` is
    always the literal string ``[object Object]`` (upstream stringification bug
    at 0.25.2). ``code`` is real, and the **process exit code is that code
    truncated to 8 bits**: 401 → 145, 400 → 144, 500 → 244, and a generic
    ``code: 1`` → 1. An exit code from this CLI is a status, not a category;
    ``!= 0`` is the only safe reading of it.

    **Internal retries are real and expensive.** A fail-forever 429 ran **10
    requests over 243 s** inside one awaited subprocess, and a fail-forever 500
    the same (10 / 232 s); a truncated stream, a 401, a 404 and a 400 are each
    **one** request, terminal. The envelope is not declared in
    :attr:`internal_retries` yet — declaring it is an adapter change this capture
    deliberately did not make.

    **``supports_file_output`` is inherited as ``True`` and does not hold.**
    Under the argv this adapter emits — no ``--yolo``, no ``--approval-mode`` —
    gemini declares exactly nine tools to the model (``list_directory``,
    ``read_file``, ``search_file_content``, ``glob``, ``activate_skill``,
    ``save_memory``, ``google_web_search``, ``write_todos``,
    ``delegate_to_agent``) and **``write_file`` is not among them**. Answering a
    ``write_file`` call gets ``Tool "write_file" not found in registry``, the
    file is never created, and gemini **exits 0** printing the model's
    ``JSON written to …`` confirmation as its answer — a success report for a
    write that did not happen. That puts this adapter in the same class as
    ``cursor-agent --mode ask`` and ``aider /ask``, and the fix is the
    declaration (``supports_file_output = False``, so
    ``llm_client.render_output_location`` asks for the array on stdout), never a
    ``--yolo``-class argv bypass. Left unchanged here because the roster in
    ``tests/test_provider_housekeeping.py`` is keyed on restriction-shaped argv
    atoms and gemini's argv carries none — its read-only tool set is the
    *default*. See ``tests/fixtures/gemini/README.md`` for the two probe runs
    that measured it (§14.7 gap 5).
    """

    version_cmd = ("gemini", "--version")
    #: 0.25.2 is where every fixture in ``tests/fixtures/gemini/`` was captured —
    #: the §6 step-5 probe, ``--network none``, the stub and the failserver both
    #: running inside the container, no credential — and the only release this
    #: parser has been run against. A pin to re-confirm rather than a range to
    #: trust: nothing older has been measured, and the field the parser reads
    #: (``response``) sits beside a ``stats`` block upstream reshapes freely.
    #: ``tests/container/cli-versions.lock`` pins the same string.
    #:
    #: ``gemini --version`` prints a bare ``0.25.2`` on stdout, which the base
    #: :meth:`parse_version` reads without an override.
    #: NO FLOOR, deliberately. `min_version` was set to 0.25.2 on 2026-08-19 — the
    #: version the §16 fixtures were captured against — and removed the same day
    #: for the reason §5.4.1(a) of harness-invocation-improvements.md states: a
    #: MEASURED-AT version is an upper bound on the true floor, never the floor.
    #: This is a PRE-EXISTING adapter, so a floor pinned to the release that
    #: happened to be in the image newly refuses installs that worked yesterday,
    #: with exit 3 and no evidence that 0.25.2 is where anything actually broke.
    #: (copilot/kimi/qwen-code/vibe keep theirs because they landed *together
    #: with* their probes; muse, cursor-agent and agy leave it empty for this
    #: same reason.) `verified_version` below is warning-only and carries the
    #: measurement instead.
    verified_version = "0.25.2"

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_timeout(self) -> int:
        return 900  # Gemini can be slower

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # gemini --output-format json --model <model> "<prompt>"
        return ["gemini", "--output-format", "json", "--model", model, prompt]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        try:
            wrapper = json.loads(stdout)
            # Gemini returns: {"session_id": "...", "response": "...", "stats": {...}}
            if isinstance(wrapper, dict) and "response" in wrapper:
                response = wrapper["response"]
                # Try to parse the response as JSON if it looks like JSON
                if isinstance(response, str):
                    response = response.strip()
                    if response.startswith(('[', '{')):
                        try:
                            return {"success": True, "data": json.loads(response)}
                        except json.JSONDecodeError:
                            pass
                    # Apply JSON extraction fallback for non-JSON starting responses
                    return extract_json_from_text(response, prefer_arrays=True)
                return {"success": True, "data": response}
            return {"success": True, "data": wrapper}
        except json.JSONDecodeError as e:
            # Try fallback extraction on raw stdout
            fallback = extract_json_from_text(stdout, prefer_arrays=True)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": f"JSON parse error: {e}", "raw": stdout, "data": None}
