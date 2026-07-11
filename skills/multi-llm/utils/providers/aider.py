"""Aider CLI provider implementation with ANSWER-marker text parsing."""
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..json_extractor import extract_json_from_text
from .base import LLMProvider, split_reasoning_effort

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

    def build_command(self, prompt: str, model: str) -> List[str]:
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
        return {
            "success": False,
            "error": f"No JSON found in aider output: {stdout[:500]}",
            "raw": stdout,
            "data": None,
        }
