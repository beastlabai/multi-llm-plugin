"""Aider CLI provider implementation with ANSWER-marker text parsing."""
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, StructuredSchema, split_reasoning_effort

# Reasoning effort values passed to aider's --reasoning-effort, which is an
# unvalidated pass-through (verified on aider 0.86.2): for openrouter/ models
# aider always accepts it and sends OpenRouter's reasoning.effort param;
# other models warn-and-ignore.
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})

# aider prints the model's reply after this marker (with --no-pretty).
# Reasoning models emit an optional "► **THINKING**" block first, which may
# contain draft/decoy JSON — only content after the LAST ANSWER marker is
# the actual reply.
ANSWER_MARKER = "► **ANSWER**"

# Absolute file paths mentioned in the prompt (e.g. the plan file) are
# passed to aider via --read so the model actually sees their contents.
# aider's own file-mention auto-add splits the message on whitespace and
# only matches repo-RELATIVE paths / bare basenames, so the absolute paths
# used by the orchestrator prompts never trigger it.
# Two-branch alternation: Windows drive-letter paths (backslash included in
# the character class so C:\Users\foo\bar.md matches in full) and POSIX
# paths (branch byte-identical to the historical pattern). Known accepted
# limitations: UNC paths (\\server\share\...) and paths containing spaces
# are not matched — extraction is best-effort enrichment behind an
# existence check, so a miss just means no extra --read context.
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/][\w.\-/\\]+|/[\w.\-/]+)")
_MAX_READ_FILES = 20


class AiderProvider(LLMProvider):
    """Provider for the Aider CLI tool (https://aider.chat/).

    Aider runs one-shot headless via --message with a battery of flags that
    disable interactivity, litter files, update checks, and streaming. The
    prompt is prefixed with "/ask " so aider stays in read-only chat mode:
    in edit mode aider refuses to write any absolute path outside the target
    git repo ("is not in the subpath of ..."), so the orchestrator's
    file-based {output_json_path} can never be written reliably — stdout is
    the ONLY output path for this provider (the file read always misses and
    invoke_with_file_output falls back to stdout).

    Output shape on stdout (stderr only carries a harmless non-terminal
    warning): 4-6 banner lines (Aider version, Model, Git repo, Repo-map),
    then for reasoning models an optional "► **THINKING**" block, then
    "► **ANSWER**", then the reply, then usually a trailing
    "Tokens: ... Cost: ..." line.

    Wrap gotcha: with --no-pretty aider still hard-wraps output at the
    terminal width (~80 columns), inserting literal newlines inside long
    JSON strings, which breaks json.loads (and extract_json_from_text does
    not repair broken strings). get_env() therefore sets COLUMNS=10000,
    which removes the wrapping entirely (verified empirically).

    get_env() also sets BROWSER=true: on rate-limit give-up aider offers to
    open the OpenRouter settings URL and --yes-always auto-accepts, which
    would launch a real browser; BROWSER=true makes Python's webbrowser a
    no-op.

    Input files: aider's file-mention auto-add only matches repo-relative
    paths / bare basenames, never the absolute paths the orchestrator
    prompts use, so build_command scans the prompt for existing absolute
    file paths (e.g. the plan file) and passes them via --read (read-only
    context files, allowed outside the repo root).

    Model names are litellm model specs passed VERBATIM to --model (e.g.
    "openrouter/z-ai/glm-5.2") — no partition/splitting. Auth is env-only:
    openrouter/ models need OPENROUTER_API_KEY in the environment (the
    subprocess inherits it).

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openrouter/z-ai/glm-5.2:high``), translated to
    ``--reasoning-effort <effort>`` right after the --model pair. Valid
    efforts are listed in REASONING_EFFORTS; anything else passes through
    verbatim as the model name (keeping ``:free``-style ids intact).

    Gotcha: aider exits 0 EVEN on auth failure and rate-limit give-up (the
    error text, e.g. "litellm.AuthenticationError ...", goes to stdout,
    usually without an ANSWER marker) — parse_output treats "no JSON
    extractable" as failure and surfaces a stdout snippet as the error.

    File-write emulation quirk: because the orchestrator prompt demands
    file-based output while /ask mode is read-only, some models answer
    with a pseudo write envelope like {"file_path": ..., "content":
    "<the JSON as a string>"} (observed live with kimi-k2.7-code).
    parse_output unwraps that envelope and returns the parsed content.
    """

    #: §9.4: aider has **no standalone prompt element**. The prompt ships inside
    #: `"/ask " + prompt` after `--message` (the prefix is what keeps aider in
    #: read-only chat mode — see the class docstring), so `argv[-1] == prompt`
    #: is false and `cmd[:-1]` would truncate an unrelated flag. Declaring
    #: `None` is why §9.4 could not be an assertion: aider is the counterexample.
    #: An adapter declaring `None` must stay argv-only with no agent profile —
    #: both of those paths drop `argv[-1]` — which aider is and does.
    prompt_argv_index = None

    #: The class docstring's "stdout is the ONLY output path for this provider",
    #: stated as data instead of prose. ``/ask`` keeps aider read-only, and in
    #: edit mode it refuses any absolute path outside the target git repo
    #: anyway, so the file the review prompts name can never be written. Asking
    #: for stdout directly is what removes the pseudo-write envelope some models
    #: answered the impossible instruction with (``{"file_path": …, "content":
    #: "<the JSON as a string>"}``); :meth:`parse_output` still unwraps that
    #: envelope, because a model can invent one from any prompt.
    supports_file_output = False

    version_cmd = ("aider", "--version")
    #: 0.86.2 is where every fixture in ``tests/fixtures/aider/`` was captured —
    #: the §6 step-5 probe, ``--network none``, the stub and the failserver both
    #: running inside the container, no credential — and the release
    #: ``tests/container/cli-versions.lock`` pins. It is a pin to re-confirm
    #: rather than a range to trust: the two things the parser reads are both
    #: aider-internal strings upstream is free to restyle
    #: (``reasoning_tags.REASONING_END``, which is where ``► **ANSWER**`` comes
    #: from, and the ``Tokens:``/``Cost:`` accounting lines the answer is trimmed
    #: of).
    #:
    #: **There is deliberately no ``min_version``.** A draft of this block
    #: declared ``min_version = "0.86.2"`` on the strength of "nothing older has
    #: been run against this parser", which is exactly the error
    #: `harness-invocation-improvements.md` §5.4.1(a) names: a *measured-at*
    #: version is an upper bound on the true floor, never the floor itself, so
    #: adopting it as a floor refuses working installs to buy nothing. aider is a
    #: **pre-existing** adapter — unlike copilot, kimi, qwen-code and vibe, which
    #: landed together with their probes and have no older users to refuse — and
    #: nothing measured says 0.85 fails. ``min_version`` is a hard exit-3 refusal
    #: in ``subagent_executor._detect_version``; ``verified_version`` is a
    #: one-line warning. The capture supports the warning and not the refusal, so
    #: only the warning is declared. ``muse``, ``cursor-agent`` and ``agy``
    #: already leave the floor empty for the mirror-image reason.
    #:
    #: (Today the refusal would not even fire: this adapter declares only the
    #: ``generate`` profile, ``_detect_version`` runs solely inside
    #: ``preflight_runner``, and that raises ``missing_profile`` first. That
    #: makes a wrong floor *latent* rather than harmless — it would go live, with
    #: no measurement behind it, on the day this adapter becomes runner-eligible.)
    #:
    #: ``version_cmd`` stays, because it is the only thing that can ever make
    #: ``verified_version`` speak: with no probe the detected version is
    #: ``unknown`` and both the floor and the warning are skipped. Confirmed
    #: against the pinned binary 2026-08-20 (``docker run --rm --network none
    #: --entrypoint /opt/mllm/real/aider mllm-e2e:local --version``): it prints
    #: exactly ``aider 0.86.2`` on stdout, which the base :meth:`parse_version`
    #: reads without an override.
    verified_version = "0.86.2"

    # §7.2 — the auth override plus the exit-0 sniff, folded out of the class
    # docstring's "Gotcha" and into the classifier so the D3k ladder sees them.
    #
    # Measured on aider 0.86.2, 2026-08-16, `--api-key openrouter=sk-invalid`,
    # exit **0**:
    #   litellm.AuthenticationError: AuthenticationError: OpenrouterException -
    #   {"error":{"message":"Missing Authentication header","code":401}}
    #   The API provider is not able to authenticate you. Check your API key.
    # Not one word of that matches the base auth set — "Check your API key" is
    # not "invalid api key", and `401` is only a context word — so a logged-out
    # aider classified as *nothing* and spent the transport retry budget on it.
    auth_error_patterns = LLMProvider.auth_error_patterns + (
        "not able to authenticate",
        "missing authentication header",
    )

    #: The residual marker: any other litellm exception aider prints while still
    #: exiting 0. It says "this run failed" and nothing more, which is exactly
    #: why :func:`classify_streams` consults it last.
    exit_zero_failure_markers = (
        re.compile(r"\blitellm\.\w*error\b"),
        re.compile(r"\blitellm\.exceptions\b"),
    )

    #: litellm's model-cost-map refresh, and the one line of aider's stdout that
    #: must never reach the failure classifier.
    #:
    #: MEASURED on the 2026-08-19 credentialed live row. litellm fetches
    #: ``model_prices_and_context_window.json`` from raw.githubusercontent.com at
    #: IMPORT time — before aider parses a flag, so ``--no-check-update`` and
    #: ``--no-analytics`` cannot reach it — with a 5 s timeout, and on ANY
    #: exception falls back to ``model_prices_and_context_window_backup.json``,
    #: which ships inside the package (1.28 MB, already on disk). The refresh is
    #: therefore best-effort by upstream's own design and its failure is not the
    #: run's failure.
    #:
    #: But aider prints that failure to **stdout** (stderr stays empty) and the
    #: text ends in ``Name or service not known``, which is a literal entry in
    #: :data:`base.CONNECTION_ERROR_MARKERS`. So on any host that cannot reach
    #: raw.githubusercontent.com — an air-gapped build box, a corporate proxy, a
    #: container with an egress allowlist — a *successful* aider run was
    #: classified ``connection_error``: the model answered, the JSON parsed, the
    #: credit was spent, and the D3k ladder took the rung for a transport fault.
    #: Measured exactly that way: exit 0, ``["ok"]`` parsed, $0.00028 charged,
    #: classification ``connection_error``.
    #:
    #: Excised by FILENAME rather than by hostname: the file name is unique to
    #: this fetch, while the host also serves things a real failure could mention.
    _TOLERATED_STDOUT_LINES = (
        re.compile(r"model_prices_and_context_window\.json"),
    )

    @classmethod
    def _drop_tolerated_lines(cls, stream: str) -> str:
        """Remove upstream's best-effort-fetch noise before classification."""
        if not stream:
            return stream
        return "\n".join(
            line for line in stream.splitlines()
            if not any(p.search(line) for p in cls._TOLERATED_STDOUT_LINES)
        )

    def classify_failure(
        self, stdout: str, stderr: str, exit_code: int = 0
    ) -> Optional[str]:
        """As the base, minus litellm's tolerated cost-map refresh line.

        The adapter supplies the data and ``classify_streams`` owns the rule
        (§7.2), so this override changes neither — it only declines to hand the
        rule a line upstream has already decided is non-fatal. Everything else
        aider prints, including every genuine ``litellm.*Error``, still reaches
        the classifier unchanged.
        """
        return super().classify_failure(
            self._drop_tolerated_lines(stdout),
            self._drop_tolerated_lines(stderr),
            exit_code,
        )

    @property
    def name(self) -> str:
        return "aider"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return shutil.which("aider") is not None

    @staticmethod
    def _extract_read_files(prompt: str) -> List[str]:
        """Existing absolute file paths mentioned in the prompt, deduped.

        These are passed via --read (read-only context; allowed OUTSIDE the
        repo root, unlike editable files). Non-existent paths — notably the
        not-yet-written {output_json_path} — are skipped, as are
        directories and special files (os.devnull is not a regular file).
        """
        found: List[str] = []
        for match in _ABS_PATH_RE.findall(prompt):
            candidate = match.rstrip(".,;:")
            if candidate in found:
                continue
            try:
                if Path(candidate).is_file():
                    found.append(candidate)
            except OSError:
                continue
            if len(found) >= _MAX_READ_FILES:
                break
        return found

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        # aider --model <litellm-model-id> --message "/ask <prompt>" + flags.
        # Every flag is load-bearing: --yes-always (no interactive prompts),
        # --no-auto-commits (no commits in the target repo), --no-pretty /
        # --no-stream (plain non-ANSI output), --no-check-update /
        # --no-show-release-notes / --no-analytics (no network/update side
        # channels), --no-show-model-warnings / --no-detect-urls /
        # --no-fancy-input (no extra prompts), --no-gitignore (don't touch
        # the target repo's .gitignore), --chat-history-file /
        # --input-history-file os.devnull (no litter files). The "/ask "
        # prefix keeps aider in read-only chat mode (see class docstring).
        # Do NOT pass --no-git: in a git repo aider sends a repo-map and
        # --yes-always auto-adds in-repo files the model requests; without
        # git the model is blind. Absolute paths mentioned in the prompt
        # (the plan file etc.) are additionally passed via --read, because
        # aider's mention auto-add never matches absolute paths.
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        cmd = [
            "aider",
            "--model",
            base_model,
        ]
        if effort is not None:
            cmd += ["--reasoning-effort", effort]
        cmd += [
            "--message",
            "/ask " + prompt,
            "--yes-always",
            "--no-auto-commits",
            "--no-pretty",
            "--no-stream",
            "--no-check-update",
            "--no-show-model-warnings",
            "--no-analytics",
            "--no-gitignore",
            "--no-show-release-notes",
            "--no-detect-urls",
            "--no-fancy-input",
            "--chat-history-file",
            os.devnull,
            "--input-history-file",
            os.devnull,
        ]
        for path in self._extract_read_files(prompt):
            cmd += ["--read", path]
        return cmd

    def get_env(self, model: str) -> Dict[str, str]:
        # BROWSER=true: neuter webbrowser on rate-limit give-up (see class
        # docstring). COLUMNS=10000: disable the ~80-column hard wrap that
        # inserts literal newlines inside long JSON strings.
        return {"BROWSER": "true", "COLUMNS": "10000"}

    @staticmethod
    def _unwrap_file_write_envelope(data: Any) -> Any:
        """Unwrap a pseudo file-write envelope emitted in read-only mode.

        The prompt asks the model to WRITE the JSON to a file; /ask mode
        cannot, so some models emulate the write (both shapes observed
        live with kimi-k2.7-code):

        - {"file_path": "...", "content": "<json string>"} — a pseudo
          write-file tool call; parse the content.
        - {"code": "<python that dumps the json>"} — a hallucinated
          code_interpreter call whose source embeds the JSON array as a
          literal; extract it from the code text.

        Return the parsed payload in those cases, the data unchanged
        otherwise.
        """
        if (
            isinstance(data, dict)
            and isinstance(data.get("content"), str)
            and set(data.keys()) <= {"file_path", "path", "filename", "file", "content"}
        ):
            content = data["content"].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                fallback = extract_json_from_text(content, prefer_arrays=True)
                if fallback.get("success"):
                    return fallback["data"]
        if (
            isinstance(data, dict)
            and isinstance(data.get("code"), str)
            and set(data.keys()) <= {"file_path", "path", "filename", "file", "code"}
        ):
            fallback = extract_json_from_text(data["code"], prefer_arrays=True)
            if fallback.get("success"):
                return fallback["data"]
        return data

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse plain-text output from the Aider CLI.

        1. Take the text after the LAST "► **ANSWER**" marker (THINKING
           blocks before it may contain decoy JSON)
        2. Strip trailing "Tokens: ..." / "Cost: ..." accounting lines
        3. Parse the answer as JSON, with fallback extraction for code
           blocks and JSON embedded in prose; unwrap a pseudo file-write
           envelope ({"file_path": ..., "content": "<json>"})
        4. No marker at all -> fallback extraction on full stdout; nothing
           extractable -> failure with a stdout snippet as the error (aider
           exits 0 even on auth errors and rate-limit give-ups)
        """
        stdout = stdout.strip()
        if not stdout:
            return {"success": False, "error": "Empty output from aider", "raw": stdout, "data": None}

        if ANSWER_MARKER in stdout:
            answer = stdout.rsplit(ANSWER_MARKER, 1)[1]
            answer = "\n".join(
                line for line in answer.splitlines()
                if not line.strip().startswith(("Tokens:", "Cost:"))
            ).strip()
            if not answer:
                return {"success": False, "error": "Empty answer after ANSWER marker", "raw": stdout, "data": None}
            if answer.startswith(("[", "{")):
                try:
                    data = json.loads(answer)
                except json.JSONDecodeError:
                    pass
                else:
                    return {"success": True, "data": self._unwrap_file_write_envelope(data)}
            result = extract_json_from_text(answer, prefer_arrays=True)
            if result.get("success"):
                result["data"] = self._unwrap_file_write_envelope(result["data"])
                return result
            return {
                "success": False,
                "error": f"No JSON found in aider answer: {answer[:500]}",
                "raw": stdout,
                "data": None,
            }

        # No ANSWER marker (e.g. auth error / rate-limit give-up printed to
        # stdout with exit code 0): try extraction on the full stdout.
        fallback = extract_json_from_text(stdout, prefer_arrays=True)
        if fallback.get("success"):
            fallback["data"] = self._unwrap_file_write_envelope(fallback["data"])
            return fallback
        # THE SNIPPET IS TAKEN AFTER THE TOLERATED LINES ARE DROPPED, and that is
        # the difference between a usable diagnosis and 500 characters of noise.
        # Offline, the first ~460 bytes of aider's stdout are litellm's cost-map
        # DNS failure plus the banner, so a failed AUTH run used to report
        # `No JSON found in aider output: HTTPSConnectionPool(host=
        # 'raw.githubusercontent.com'...` and never reached the
        # `litellm.AuthenticationError` further down — measured on the 2026-08-19
        # fixture capture. `_drop_tolerated_lines` already exists for
        # `classify_failure`; the error text needs it for the same reason.
        diagnosis = self._drop_tolerated_lines(stdout).strip() or stdout
        return {
            "success": False,
            "error": f"No JSON found in aider output: {diagnosis[:500]}",
            "raw": stdout,
            "data": None,
        }
