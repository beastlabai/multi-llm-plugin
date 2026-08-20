"""Muse Code CLI provider implementation with JSONL event stream parsing."""
import json
import shutil
from typing import Any, Dict, Iterator, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, StructuredSchema, split_reasoning_effort

# Reasoning effort values accepted by `muse exec --reasoning-effort`
# (verified live against Muse Code 0.1.0-R708.1). Unlike most providers here,
# muse HARD-ERRORS on an unrecognised value ("unsupported reasoning effort
# `x`; expected none|minimal|low|medium|high|xhigh|ultra") rather than
# ignoring it, so the whitelist has to match the CLI exactly.
REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "ultra"}
)


class MuseProvider(LLMProvider):
    """Provider for Meta's Muse Code CLI tool.

    ``muse exec --json`` emits JSONL records on stdout. Each record is an
    envelope — ``{"sequence": N, "record_type": ..., "payload_type": ...,
    "payload": {...}}`` — and only three payload types matter here:

    - ``run.output.delta``  — raw streaming text chunks. These are token-level
      fragments, NOT message boundaries: a single assistant message arrives
      split across many deltas.
    - ``tool.result``       — a completed tool call, which is the only usable
      marker between assistant messages (see parse_output).
    - ``run.terminal.*``    — the closing record, carrying ``terminal``
      (completed/failed), ``reason``, and ``text``.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``muse-spark-1.2:xhigh``), translated to ``--reasoning-effort <effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name. Without a suffix muse does NOT use
    the ``--reasoning-effort`` default advertised in ``--help``; it uses
    ``reasoning_effort`` from the user's ~/.config/muse/settings.json.

    ``--disable-approval`` is mandatory, not a convenience: muse's default
    approval policy is ``on-request``, and a tool call the policy judges risky
    (verified with ``rm -rf``) blocks on a prompt that headless exec can never
    answer — the run writes no further output and burns the entire timeout.
    The shell sandbox is deliberately left ON (muse's default): it already
    permits writes under the workspace root, which is where every review
    output path lives.

    Workspace trust is also left at muse's default. An untrusted workspace
    only skips project-local instructions, skills, and hooks — file reads and
    writes still work — so this defers to the user's own ``muse`` trust store
    rather than granting trust from inside a review run.
    """

    # §7.2 — per-adapter pattern overrides, EXTENDING the base sets.
    #
    # muse's failures are structured: exit 1 with stderr carrying only "run
    # ended with Failed", and the real cause on stdout in the
    # `run.terminal.failed` record's `reason` — which is why `describe_failure`
    # exists. Two recorded shapes the base English lists do not carry:
    #
    # * an unfunded account gives `API error 402 … Billing verification failed …
    #   (after 10 provider attempts)`. That is a credit/quota condition, so it
    #   belongs on the rate-limit rung (backoff, budget not consumed) rather
    #   than on the generic transport rung.
    # * `API error 401` is muse's auth spelling; the bare token `401` is not in
    #   the base auth set at all (it is only a *context* word there).
    #
    # MEASURED 2026-08-19 AND NOT YET FIXED — the two entries below did not fire
    # on the one 401 this repo has actually captured. `tests/fixtures/muse/`'s
    # `auth-error` row is a real 401 from `failserver.py --mode 401`, and muse
    # SWALLOWS the vendor text: the strings `401`, `API error` and
    # `Incorrect API key provided` appear **nowhere** in the 17 KB capture.
    # What muse writes instead is its own gloss, in the `run.terminal.failed`
    # record's `reason` — "your API key from META_API_KEY was rejected — update
    # or unset it" — which matches none of the twelve patterns this tuple then
    # holds, so a rejected muse credential classifies as **None** and lands on
    # the transport rung with a full retry budget. The one-line fix is a marker
    # for muse's own wording (`was rejected` / `api key from`), the way vibe's
    # `conversation context exceeds` was added; it is left undone here on
    # purpose so the committed fixture pins today's behaviour and the fix is a
    # visible edit to both this tuple and that manifest. The entries are NOT
    # removed: they are still what the live 402 shape's sibling statuses
    # spell, and nothing has measured a live 401 against a real Meta endpoint.
    auth_error_patterns = LLMProvider.auth_error_patterns + (
        "api error 401",
        "api error 403",
    )
    rate_limit_patterns = LLMProvider.rate_limit_patterns + (
        "api error 402",
        "billing verification failed",
        "api error 429",
    )

    # A THIRD MEASURED SHAPE, recorded rather than fixed, and it is a base-table
    # ORDERING fact rather than a missing pattern here. muse prefixes every
    # vendor error it does not gloss with `API error <status>: <vendor text>`,
    # and `\bapi error\b` is a member of `base.SERVER_ERROR_MARKERS`, which
    # `classify_streams` consults BEFORE `INVALID_MODEL_MARKERS`. So muse's 404
    # — which is otherwise ideal, keeping the vendor sentence verbatim and
    # NAMING the model ("API error 404: The model `muse-spark-9.9-nonexistent`
    # does not exist or you do not have access to it.") — classifies as
    # `server_error`, a retryable transport category, instead of the terminal
    # `invalid_model` whose fix is a config edit. Pinned by
    # `tests/fixtures/muse/`'s `invalid-model` row, which asserts the
    # classification the code produces today. The 402/429 rungs above are
    # unaffected: rate limits are checked before the server table.

    #: The build every fixture in ``tests/fixtures/muse/`` was captured from —
    #: the §6 step-5 probe, ``--network none``, ``stub_server.py`` and
    #: ``failserver.py`` both running inside the container, no credential. Ten
    #: of §6.2's twelve rows — plus two extra truncation rows — are raw
    #: ``muse exec --json`` bytes from this build, so it is what the JSONL
    #: envelope shape above is *known* to be true of, not a range to trust.
    #:
    #: ``min_version`` is deliberately left empty rather than pinned to the same
    #: string, and the reason is that this adapter declares no ``version_cmd``:
    #: ``_detect_version`` only compares against a floor when a probe actually
    #: read a version, so a ``min_version`` here would be enforced by nothing
    #: while reading, in every listing, as a gate that exists. Declaring the
    #: probe would take a ``parse_version`` override first — ``muse --version``
    #: prints ``Muse Code 0.1.0 (0.1.0-R708.1)``, two version strings on one
    #: line, and the base parser reads neither of them the way this field
    #: spells it. ``tests/container/cli-versions.lock`` pins the same build.
    verified_version = "0.1.0-R708.1"

    @property
    def name(self) -> str:
        return "muse"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("muse") is not None

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # muse exec --json --disable-approval --model <model>
        #     [--reasoning-effort <effort>] "<prompt>"
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = [
            "muse", "exec", "--json", "--disable-approval",
            "--model", base_model,
        ]
        if effort is not None:
            cmd += ["--reasoning-effort", effort]
        cmd.append(prompt)
        return cmd

    @staticmethod
    def _iter_events(stdout: str) -> Iterator[Dict[str, Any]]:
        """Yield ``(payload_type, payload)``-bearing records of the JSONL stream."""
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
            if isinstance(event, dict) and isinstance(event.get("payload"), dict):
                yield event

    @staticmethod
    def _direct_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse ``text`` as one complete JSON document, or return None."""
        if text.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(text)}
            except json.JSONDecodeError:
                pass
        return None

    def _scan(self, stdout: str) -> Dict[str, Any]:
        """Collect the answer text and terminal state from the event stream.

        ``tail`` holds the deltas emitted after the LAST ``tool.result``, which
        is the closest thing muse gives to "the final assistant message" — see
        parse_output for why that matters.
        """
        deltas: List[str] = []
        tail: List[str] = []
        terminal: Optional[str] = None
        reason: Optional[str] = None
        terminal_text: Optional[str] = None
        task_failures: List[str] = []

        for event in self._iter_events(stdout):
            payload_type = event.get("payload_type") or ""
            payload = event["payload"]

            if payload_type == "run.output.delta":
                text = payload.get("text")
                if isinstance(text, str):
                    deltas.append(text)
                    tail.append(text)
            elif payload_type == "tool.result":
                # A tool call closes the preceding assistant message; anything
                # buffered so far belongs to an earlier turn.
                tail.clear()
            elif payload_type.startswith("run.terminal."):
                terminal = payload.get("terminal")
                if isinstance(payload.get("reason"), str) and payload["reason"]:
                    reason = payload["reason"]
                if isinstance(payload.get("text"), str):
                    terminal_text = payload["text"]
            elif payload_type == "task.lifecycle.failed":
                detail = payload.get("event")
                if isinstance(detail, dict):
                    message = detail.get("reason")
                    if isinstance(message, str) and message:
                        task_failures.append(message)

        return {
            # The terminal record's text is authoritative — the deltas are
            # ephemeral status records that a future build could drop.
            "full": terminal_text if terminal_text is not None else "".join(deltas),
            "tail": "".join(tail),
            "terminal": terminal,
            "reason": reason,
            "task_failures": task_failures,
        }

    def describe_failure(self, stdout: str, stderr: str) -> Optional[str]:
        """Recover the reason for a non-zero exit from the event stream.

        On a failed run muse writes only "run ended with Failed" to stderr and
        leaves the actual cause in the ``run.terminal.failed`` record on
        stdout. Verified against a live out-of-credits account, which exited 1
        with that stderr and a terminal reason of "API error 402 [...]:
        Billing verification failed. [...] (after 10 provider attempts)".
        """
        scan = self._scan(stdout)
        if scan["reason"]:
            return scan["reason"]
        # No terminal record (killed mid-run, say) — fall back to whatever
        # task-level failures were reported.
        return "; ".join(scan["task_failures"]) or None

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse the JSONL event stream emitted by ``muse exec --json``.

        The terminal record's ``text`` is the concatenation of EVERY assistant
        message in the run, glued with no separator — a run that narrates
        before answering yields "Looks good so far.[{...}]", and one that
        shows a draft yields "Draft: [{...draft...}][{...answer...}]". Feeding
        that whole string to the extractor returns the FIRST balanced
        structure, i.e. the draft, so the closing message has to be isolated
        first.

        muse publishes no message-boundary event (``run.output.delta`` carries
        token fragments, verified by splitting one message across three SSE
        deltas), but ``tool.result`` reliably separates turns. The deltas after
        the last tool result are therefore tried first, and the full text only
        as a fallback — for single-turn runs the two are identical.

        All of that is now a **committed capture** rather than a description:
        ``tests/fixtures/muse/multi-assistant.stdout.txt`` is raw JSONL from a
        run whose terminal ``text`` reads ``Draft: [{…DRAFT…}][{…S001…}]``, and
        the manifest asserts both halves — ``extract_json_from_text`` over the
        whole string returns the DRAFT, the tail after the last ``tool.result``
        returns the answer. It is the only test in the repo that would notice
        this rule being removed, which is why it took a probe-local stub driving
        a real ``read_file`` call to produce (see that directory's README).

        MEASURED 2026-08-19, recorded rather than fixed: **a FAILED run whose
        salvaged text happens to parse is returned as a success.** The first
        branch below fails a non-``completed`` run only ``and not full``, so with
        any text present control falls through to ``_direct_json(tail)`` and
        returns before the terminal state is consulted again.
        ``tests/fixtures/muse/truncated-json-answer.stdout.txt`` is the capture:
        exit **1**, ``run.terminal.failed`` reading "model failed after 10
        attempts: stream ended before terminal event", and this method answering
        ``success: True`` with the array. ``llm_client`` is safe by accident (it
        short-circuits on ``returncode != 0`` and never calls this), but
        ``subagent_executor`` is not — it treats the exit code as evidence rather
        than verdict and accepts a schema-valid result from a non-zero exit. The
        fix is one clause; the fixture pins today's behaviour until it lands.

        Residual limit: two JSON documents inside the SAME closing turn (a draft
        and an answer with no tool call between them) still resolve to the first,
        since nothing separates them. Preferring the last candidate instead just
        swaps that for the opposite failure — an answer followed by an
        illustrative snippet — so this keeps the extractor's first-wins rule,
        which the "return ONLY a JSON array" prompts make the safer bet.
        """
        scan = self._scan(stdout)
        full = scan["full"].strip()
        tail = scan["tail"].strip()

        if scan["terminal"] not in (None, "completed") and not full:
            reason = scan["reason"] or "; ".join(scan["task_failures"])
            error = f"Muse run {scan['terminal']}"
            if reason:
                error = f"{error}: {reason}"
            return {"success": False, "error": error, "raw": stdout, "data": None}

        if not full:
            error = "No output events found in output"
            reason = self.describe_failure(stdout, stderr)
            if reason:
                error = f"Muse reported an error: {reason}"
            return {"success": False, "error": error, "raw": stdout, "data": None}

        # Exact parses first — they are unambiguous — closing message before
        # the whole transcript.
        for candidate in (tail, full):
            if candidate:
                direct = self._direct_json(candidate)
                if direct is not None:
                    return direct

        result: Dict[str, Any] = {}
        for candidate in (tail, full):
            if not candidate:
                continue
            # Try to extract JSON from the text (may be in code blocks)
            result = extract_json_from_text(candidate, prefer_arrays=True)
            if result.get("success"):
                return result

        # Nothing parsed out of the salvaged text of a run that did not reach
        # "completed" — the terminal reason is the real diagnosis, so keep it
        # rather than reporting only "no valid JSON found".
        if scan["terminal"] not in (None, "completed") and scan["reason"]:
            result = dict(result)
            result["error"] = "{} (muse run {}: {})".format(
                result.get("error", "Failed to parse response"),
                scan["terminal"], scan["reason"],
            )
        return result
