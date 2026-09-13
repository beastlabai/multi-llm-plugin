#!/usr/bin/env python3
"""
Ask orchestrator: a lightweight, read-only Q&A side-channel for the multi-llm skill.

Unlike the review/implement orchestrators, this mode does NOT force model output
into a JSON suggestion schema and does NOT group/validate/consolidate. It takes a
plan file plus a free-text question, fans the question out to each configured LLM
with the plan inlined as read-only context, captures each model's raw markdown
answer, and aggregates the answers into a single ``answers.md`` file.

Output layout (per question):
    {plan}/ask/<slug>-<hash8>/
    ├── answers.md            # aggregated markdown answers (regenerated each run)
    ├── answer_<model>.md     # per-model raw markdown answer
    ├── log_<model>.txt       # per-model provider debug log
    ├── error_<model>.log     # per-model error log (only on failure)
    └── .status.json          # resume detection + collision/freshness guard

Slug-collision semantics (see ``slugify_question``):
- **Same question text → same slug → same dir → resume** (deliberate sharing):
  existing per-model answers are reused and ``answers.md`` is regenerated.
- **Different question text → different slug → different dir** (no sharing):
  because ``<hash8>`` derives from the full original question, two different
  questions cannot normally share a directory, so there is no silent
  answer-mixing.
- **Astronomically-unlikely hash + prefix collision → suffixed sibling dir**
  (``<slug>-2/``), never a silent overwrite: the verbatim-question guard
  (``.status.json`` ``question`` field) catches a stored question that differs
  byte-for-byte from the current one and routes to a sibling directory instead
  of regenerating another question's ``answers.md``.

No ``state.json`` phase tracking — ask is a read-only side-channel, not a
workflow phase.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.stream_bootstrap import bootstrap_streams
from utils import cli_common
from utils.cli_common import (
    build_parent_parser,
    dispatcher_prefix,
)
from utils.output_handler import get_phase_dir, slugify_question
from utils.prompt_loader import load_prompt
from utils.interactive import resolve_models
from utils import unattended as unattended_mod
from utils.unattended import NeedsHuman
from utils.provider_registry import (
    get_all_model_specs,
    get_provider,
    get_provider_timeout,
    is_model_valid,
    parse_model_spec,
    set_run_anchor,
)
from utils.json_extractor import sanitize_model_name
from utils.llm_client import invoke_with_provider
from utils.git_utils import get_project_root
from utils.detach import record_output_files, report_progress
from utils.review_orchestrator_base import (
    maybe_detach,
    read_plan,
    run_models_concurrent,
    update_status,
    write_status,
)

# Prompt template filename
ASK_PROMPT_FILE = "ask.txt"

# Conservatively below Linux MAX_ARG_STRLEN (~128 KB) so the rendered prompt,
# delivered as a single argv element, never fails with E2BIG. Leaves headroom
# for the rest of the template.
MAX_INLINE_PLAN_BYTES = 100 * 1024

# Hard ceiling for the fully-rendered prompt (delivered as one argv element).
# Conservatively below Linux MAX_ARG_STRLEN (~128 KB). The plan-inlining guard
# (MAX_INLINE_PLAN_BYTES) only measures plan_content; this catches the case where
# the question (or other template fields) push the rendered prompt over the argv
# ceiling even when the plan itself is small, which would otherwise fail E2BIG.
MAX_PROMPT_BYTES = 124 * 1024


# ---------------------------------------------------------------------------
# Question input resolution
# ---------------------------------------------------------------------------

def resolve_question(args: argparse.Namespace) -> str:
    """Resolve the verbatim question from exactly one of the mutually-exclusive
    inputs (``--question`` / ``--question-file`` / ``--question-env``).

    The resolved question is the **verbatim original text** — no shell
    processing, no re-quoting, no stripping — so that all three sources yield
    byte-identical results and the slug/hash and stored ``question`` key use the
    exact original string.
    """
    if args.question is not None:
        return args.question
    if args.question_file is not None:
        try:
            with open(args.question_file, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, IOError) as e:
            print(f"ERROR: could not read --question-file '{args.question_file}': {e}")
            sys.exit(1)
    if args.question_env is not None:
        value = os.environ.get(args.question_env)
        if value is None:
            print(f"ERROR: --question-env '{args.question_env}' is not set in the environment")
            sys.exit(1)
        return value
    # argparse(required=True) on the group guarantees one of the above.
    print("ERROR: no question provided")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------

def compute_plan_hash(plan_content: str) -> str:
    """Return the first 12 hex chars of sha1 over the plan content."""
    return hashlib.sha1(plan_content.encode("utf-8")).hexdigest()[:12]


def build_inline_plan(plan_content: str, plan_path: str) -> Tuple[str, bool]:
    """Return (content_for_prompt, truncated).

    Inlines the full plan when it fits under :data:`MAX_INLINE_PLAN_BYTES`.
    Above the threshold, truncates the inlined copy and appends a clear marker
    pointing at ``plan_path`` (agentic providers can still read the full file
    from disk; the path is always present in the prompt). This is a documented,
    recoverable degradation rather than a hard ``E2BIG`` failure.
    """
    encoded = plan_content.encode("utf-8")
    if len(encoded) <= MAX_INLINE_PLAN_BYTES:
        return plan_content, False
    truncated = encoded[:MAX_INLINE_PLAN_BYTES].decode("utf-8", errors="ignore")
    marker = (
        f"\n\n[... plan truncated at {MAX_INLINE_PLAN_BYTES // 1024} KB; "
        f"read the full plan at {plan_path} ...]\n"
    )
    return truncated + marker, True


# ---------------------------------------------------------------------------
# The two per-model fields of ``prompts/ask.txt``
#
# Same decision as the three JSON review prompts', taken the same way: from the
# adapter's declared
# :attr:`~utils.providers.base.LLMProvider.supports_file_output` rather than from
# a provider name spelled out in this orchestrator
# (:func:`utils.llm_client.render_output_location` is the review-side twin). The
# wording cannot be shared with that function's constants — those ask for a raw
# JSON array and ask mode's answer is markdown — but the roster, the rule and
# the unknown-provider posture are the same one: an adapter this build no longer
# ships renders the file variant, which is what this template did unconditionally
# before the fields existed.
#
# Six shipped adapters declare ``supports_file_output = False`` and were being
# told to write a file they can only answer with a refusal sentence. Nothing
# downstream moves: they already land on :func:`recover_answer_text`, which
# persists the stdout answer to ``answer_path``, so ``source: "fallback"`` stays
# the measured outcome and aggregation still reads a file.
#
# The stdout variant is the shorter of the two on both fields, deliberately: the
# rendered template is the fixed overhead ``render_prompt_guarded`` subtracts
# from :data:`MAX_PROMPT_BYTES`, so this fix can only grow the question budget,
# never shrink it. ``tests/test_ask.py`` pins that as an assertion.
# ---------------------------------------------------------------------------

#: The read-only bullet for a file-capable adapter. Byte-identical to the text
#: the template carried inline before the field existed.
FILE_READONLY_WRITE_RULE = (
    "Do NOT modify, create, or delete any file except the single answer output\n"
    "  file specified below ({path}). In particular, NEVER edit the plan\n"
    "  file or any other file in the repository."
)

#: The same bullet for an adapter that cannot write: naming an exception for a
#: file the next section has just said not to write would contradict it.
STDOUT_READONLY_WRITE_RULE = (
    "Do NOT modify, create, or delete any file. In particular, NEVER edit the\n"
    "  plan file or any other file in the repository."
)

#: What a file-capable adapter is asked for. Byte-identical to the text the
#: template carried inline before the field existed, so the rendered prompt of
#: every adapter that declares ``supports_file_output`` is unchanged to the
#: character.
FILE_OUTPUT_LOCATION = (
    "## Output Location\n"
    "Write your answer as MARKDOWN (not JSON) to this exact absolute file path:\n"
    "\n"
    "{path}\n"
    "\n"
    "- Write to the path exactly as given above. Do NOT reinterpret, relativize, or\n"
    "  change it. It is an absolute path; write the file there.\n"
    "- Write ONLY your markdown answer to that file (no JSON wrapper, no code-fence\n"
    "  around the whole answer).\n"
    "- ALSO print your full answer to stdout. Stdout is the fallback that is used if\n"
    "  the file write fails, so the two should contain the same answer."
)

#: What an adapter that declares ``supports_file_output = False`` is asked for
#: instead: stdout is the channel it *does* have, it is the one
#: :func:`recover_answer_text` already falls back to, and the answer lands in
#: exactly the same ``answer_<model>.md`` either way.
STDOUT_OUTPUT_LOCATION = (
    "## Output Location\n"
    "Print your full answer to stdout as your final message, in MARKDOWN (not JSON).\n"
    "Do NOT write it to a file — you are running in a read-only mode and any file\n"
    "write will be refused. Print the answer itself and nothing else: no JSON\n"
    "wrapper, no code-fence around the whole answer."
)


def render_output_fields(provider_name: Optional[str], output_md_path: str) -> Dict[str, str]:
    """The ``{readonly_write_rule}`` and ``{output_location}`` fields for
    ``provider_name``'s rendered ask prompt.

    ``provider_name`` is the already-resolved first half of the model spec (see
    :func:`~utils.provider_registry.parse_model_spec`); ``None`` — or a name this
    build ships no adapter for — renders the file variant, which is what every
    caller got before these fields existed.
    """
    provider = get_provider(provider_name) if provider_name else None
    if provider is not None and not provider.supports_file_output:
        return {
            "readonly_write_rule": STDOUT_READONLY_WRITE_RULE,
            "output_location": STDOUT_OUTPUT_LOCATION,
        }
    return {
        "readonly_write_rule": FILE_READONLY_WRITE_RULE.format(path=output_md_path),
        "output_location": FILE_OUTPUT_LOCATION.format(path=output_md_path),
    }


def render_prompt(
    template: str,
    question: str,
    plan_path: str,
    plan_content: str,
    output_md_path: str,
    provider_name: Optional[str] = None,
) -> str:
    """Render the ask prompt. Only template literals need ``{{`` escaping;
    substituted values (e.g. plan content with literal braces) pass through
    untouched because ``str.format`` only parses the template's own braces —
    which is also why the two per-model fields are formatted *before* they are
    substituted rather than by this call.

    ``provider_name`` selects the output-channel variant
    (:func:`render_output_fields`); omitted, the prompt asks for a file.
    """
    return template.format(
        question=question,
        plan_path=plan_path,
        plan_content=plan_content,
        **render_output_fields(provider_name, output_md_path),
    )


def render_prompt_guarded(
    template: str,
    question: str,
    plan_path: str,
    plan_content: str,
    output_md_path: str,
    provider_name: Optional[str] = None,
) -> Tuple[str, bool]:
    """Render the ask prompt and guarantee the result stays under
    :data:`MAX_PROMPT_BYTES` (the single-argv ceiling), returning
    ``(rendered_prompt, prompt_truncated)``.

    :func:`build_inline_plan` only bounds ``plan_content``; a large question (or
    other template fields) can still push the *fully rendered* prompt over
    ``MAX_ARG_STRLEN`` and cause an ``E2BIG`` failure even when the plan is
    small. This measures the rendered prompt and, when oversized, truncates the
    question (the only remaining caller-controlled field) with a clear marker so
    the invocation still runs instead of hard-failing. ``plan_path`` and
    ``output_md_path`` are never truncated — the model needs them intact to read
    the full plan and write its answer.

    ``provider_name`` is threaded through to :func:`render_prompt` so the
    overhead measured here is the overhead of the variant actually dispatched —
    the stdout variant is the shorter of the two, so a read-only adapter's
    question budget can only be larger than a file-capable one's.
    """
    rendered = render_prompt(
        template, question, plan_path, plan_content, output_md_path, provider_name
    )
    if len(rendered.encode("utf-8")) <= MAX_PROMPT_BYTES:
        return rendered, False

    # The fixed overhead is everything except the question. Compute how many
    # question bytes we can keep so the rendered prompt fits under the ceiling.
    marker = "\n\n[... question truncated to fit the argv size limit ...]"
    overhead = len(
        render_prompt(
            template, marker, plan_path, plan_content, output_md_path, provider_name
        ).encode("utf-8")
    )
    budget = MAX_PROMPT_BYTES - overhead
    if budget < 0:
        budget = 0
    q_bytes = question.encode("utf-8")[:budget]
    truncated_question = q_bytes.decode("utf-8", errors="ignore") + marker
    rendered = render_prompt(
        template, truncated_question, plan_path, plan_content, output_md_path, provider_name
    )
    return rendered, True


# ---------------------------------------------------------------------------
# Answer capture helpers
# ---------------------------------------------------------------------------

def read_nonempty_answer(path: str) -> Optional[str]:
    """Return the answer file's content if it exists and is readable with
    non-whitespace content, else None.

    This is the single, consistent definition of "a usable answer file" used
    by resume, the live run, and the hard-failure-with-file policy, so a
    truncated/empty/unreadable file never counts as success.
    """
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                return text
    except (OSError, IOError):
        return None
    return None


def looks_like_ndjson(text: str) -> bool:
    """Heuristic: True if *text* looks like an NDJSON event stream.

    Several providers (e.g. codex's "No text events" failure) set ``raw`` to the
    full NDJSON event stream. We must never paste that verbatim into
    ``answers.md`` as if it were the answer. A genuine markdown answer contains
    prose lines that do not parse as JSON, so the first non-JSON line short
    circuits this to False.
    """
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    parsed_dicts = 0
    typed_events = 0
    for ln in lines[:25]:
        try:
            obj = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(obj, dict):
            return False
        parsed_dicts += 1
        if "type" in obj or "msg" in obj or "event" in obj:
            typed_events += 1
    return parsed_dicts >= 2 and typed_events >= 1


def _unwrap_provider_envelope(text: str) -> str:
    """Unwrap a provider's JSON result envelope to its inner answer string.

    Several agentic providers print their answer wrapped in a single JSON object
    on stdout: claude-code/cursor-agent use ``{"type":"result","result":"..."}``
    and gemini uses ``{"session_id":...,"response":"...","stats":{...}}``. When
    the answer is recovered from the log (the dict-``data``-with-no-``raw`` case),
    that stdout is the *envelope*, not the answer — pasting it verbatim into
    ``answers.md`` would embed escaped JSON instead of the model's markdown. If
    *text* is exactly such an envelope with a non-empty string inner field,
    return the inner string (the real answer); otherwise return *text* unchanged.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text
    if not isinstance(obj, dict):
        return text
    for key in ("result", "response"):
        inner = obj.get(key)
        if isinstance(inner, str) and inner.strip():
            return inner
    return text


def _read_stdout_from_log(log_file: Optional[str]) -> Optional[str]:
    """Extract the captured STDOUT section from a per-model debug log, or None.

    The log written by ``utils.llm_client._save_log`` embeds the full captured
    stdout between a ``STDOUT`` banner (a ``-`` rule, the word ``STDOUT``, a ``-``
    rule) and the following ``STDERR`` banner. For agentic providers
    (claude-code/cursor-agent/gemini) whose answer contains brackets/fences,
    ``parse_output`` returns ``success`` with a ``dict`` ``data`` and no ``raw``,
    so the real answer is unreachable from ``result`` alone — but it still lives
    in this log's stdout section, the documented last-resort recovery source.

    Because that captured stdout is the provider's *envelope*
    (``{"type":"result","result":"..."}`` / ``{"response":"..."}``) rather than
    the bare answer, it is unwrapped via :func:`_unwrap_provider_envelope` so the
    model's actual markdown — not escaped JSON — lands in ``answers.md``.
    """
    if not log_file:
        return None
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_text = f.read()
    except (OSError, IOError):
        return None

    lines = log_text.split("\n")
    stdout_lines: List[str] = []
    in_stdout = False
    for i, line in enumerate(lines):
        if not in_stdout:
            # The STDOUT banner is "----..." / "STDOUT" / "----..." — detect the
            # "STDOUT" label flanked by dashed rule lines.
            if line.strip() == "STDOUT" and i > 0 and set(lines[i - 1].strip()) == {"-"}:
                in_stdout = True
            continue
        # Skip the dashed rule line immediately after the STDOUT label.
        if not stdout_lines and set(line.strip()) == {"-"} and line.strip():
            continue
        # The STDERR banner (dashed rule then "STDERR") ends the stdout section.
        if line.strip() == "STDERR" and stdout_lines and set(stdout_lines[-1].strip()) == {"-"}:
            stdout_lines.pop()  # drop the trailing dashed rule
            break
        stdout_lines.append(line)

    if not in_stdout:
        return None
    captured = "\n".join(stdout_lines).strip()
    if not captured or captured == "(empty)":
        return None
    if looks_like_ndjson(captured):
        return None
    unwrapped = _unwrap_provider_envelope(captured).strip()
    return unwrapped or None


def recover_answer_text(result: Dict[str, Any], log_file: Optional[str] = None) -> Optional[str]:
    """Recover a usable plain-text answer from a provider result, or None.

    Safer than the naive "``data`` if str else ``raw``" rule:
    - Use ``result["data"]`` only when it is a non-empty string. A ``dict``/
      ``list`` ``data`` is an erroneously-extracted JSON fragment (``parse_output``
      runs ``extract_json_from_text`` on the inner text, so a good markdown
      answer that merely contains brackets/code-fences gets reduced to a
      fragment) — treated as a capture failure, never embedded as-is.
    - Otherwise fall back to ``result.get("raw")`` (the real stdout for the
      no-JSON parse path) only when it is a non-empty string that is not an
      NDJSON event stream.
    - Last resort: when neither ``data`` nor ``raw`` is usable (the agentic
      ``dict``-``data``-with-no-``raw`` case), recover the captured stdout from
      the per-model log file so a good answer that exists only there is not
      silently dropped.
    """
    data = result.get("data")
    if isinstance(data, str) and data.strip():
        return data
    raw = result.get("raw")
    if isinstance(raw, str) and raw.strip() and not looks_like_ndjson(raw):
        return raw
    return _read_stdout_from_log(log_file)


def write_error_log(question_dir: str, model_spec: str, error: str, result: Dict[str, Any]) -> str:
    """Write a per-model error log (same convention as ``save_model_result``)."""
    sanitized = sanitize_model_name(model_spec)
    path = os.path.join(question_dir, f"error_{sanitized}.log")
    details = result.get("details", {}) if isinstance(result, dict) else {}
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_spec}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error}\n")
            stderr_text = details.get("stderr") if isinstance(details, dict) else None
            if stderr_text:
                f.write(f"\nstderr:\n{stderr_text}\n")
    except (OSError, IOError):
        pass
    return path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def demote_headings(text: str) -> str:
    """Shift every ATX markdown heading down one level (outside fenced code
    blocks), capping at level 6.

    Per-model answers commonly contain their own ``#``/``##`` headings. Demoting
    them keeps them strictly *below* the per-model ``# provider:model`` section
    header in ``answers.md`` so it stays unambiguous where one model's answer
    ends and the next begins. Headings inside fenced code blocks (e.g. shell
    comments) are left untouched.
    """
    import re as _re

    out: List[str] = []
    in_fence = False
    fence_marker: Optional[str] = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker is not None and stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = None
            out.append(line)
            continue
        if not in_fence:
            m = _re.match(r'^(#{1,6})(\s.*)?$', line)
            if m and len(m.group(1)) < 6:
                out.append("#" + line)
                continue
        out.append(line)
    return "\n".join(out)


def collect_completed(
    question_dir: str, model_specs: List[str]
) -> List[Tuple[str, str]]:
    """Read-only scan of ``answer_<model>.md`` files.

    Returns the list of ``(spec, answer_text)`` pairs for models that produced
    a usable (non-empty) answer, in the resolved ``model_specs`` order. This is
    the file-based source of truth for which models succeeded, and lets callers
    derive ``completed`` without writing ``answers.md``.
    """
    sections: List[Tuple[str, str]] = []
    for spec in model_specs:
        sanitized = sanitize_model_name(spec)
        answer_path = os.path.join(question_dir, f"answer_{sanitized}.md")
        text = read_nonempty_answer(answer_path)
        if text is not None:
            sections.append((spec, text.strip()))
    return sections


def write_answers_md(
    question_dir: str,
    question: str,
    plan_path: str,
    model_specs: List[str],
    failed_models: Dict[str, str],
    timestamp: str,
) -> Tuple[str, List[str]]:
    """Fully (re)generate ``answers.md`` from all current ``answer_<model>.md``
    files plus the failed-models map.

    Returns (answers_md_path, completed_specs). Section ordering is
    deterministic (the resolved ``model_specs`` order), so repeated
    regeneration over identical inputs is byte-identical. ``completed`` is the
    file-based source of truth for which models succeeded; ``failed_models``
    enumerates the rest.
    """
    sections = collect_completed(question_dir, model_specs)
    completed: List[str] = [spec for spec, _ in sections]

    lines: List[str] = ["# Multi-LLM Ask", ""]
    lines.append("**Question:**")
    lines.append("")
    for q_line in question.split("\n"):
        lines.append(f"> {q_line}")
    lines.append("")
    lines.append(f"**Plan:** {plan_path}")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Models asked:** {', '.join(model_specs)}")
    lines.append(f"**Answered:** {len(completed)}/{len(model_specs)}")
    lines.append("")

    for spec, text in sections:
        lines.append("---")
        lines.append("")
        lines.append(f"# {spec}")
        lines.append("")
        lines.append(demote_headings(text))
        lines.append("")

    if failed_models:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Models")
        lines.append("")
        for spec in model_specs:
            if spec in failed_models:
                lines.append(f"- **{spec}**: {failed_models[spec]}")
        lines.append("")

    answers_md = os.path.join(question_dir, "answers.md")
    with open(answers_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return answers_md, completed


# ---------------------------------------------------------------------------
# Question directory resolution (collision / resume guard)
# ---------------------------------------------------------------------------

def resolve_question_dir(phase_dir: str, slug: str, question: str) -> Tuple[str, bool]:
    """Resolve the per-question output directory, honoring the verbatim-question
    collision guard.

    Returns (dir_path, is_resume).
    - If the slug dir does not exist → fresh dir.
    - If it exists and its stored ``.status.json`` ``question`` byte-for-byte
      equals the current question (or no status is stored) → resume into it.
    - If it exists with a *different* stored question (astronomically-unlikely
      hash+prefix collision) → route to a suffixed sibling (``<slug>-2``, ...),
      never overwriting another question's answers.
    """
    base = os.path.join(phase_dir, slug)
    candidate = base
    n = 1
    while True:
        if not os.path.exists(candidate):
            return candidate, False
        if os.path.isdir(candidate):
            status_path = os.path.join(candidate, ".status.json")
            stored_q: Optional[str] = None
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r", encoding="utf-8") as f:
                        stored_q = json.load(f).get("question")
                except (json.JSONDecodeError, OSError, IOError):
                    stored_q = None
            if stored_q is None or stored_q == question:
                return candidate, True
        # Different question (or a non-dir collision) → try a sibling.
        n += 1
        candidate = f"{base}-{n}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    try:
        available_models = get_all_model_specs()
    except FileNotFoundError:
        available_models = ["(providers.yaml not found - run with --models to specify)"]
    except Exception as e:
        available_models = [f"(config error: {type(e).__name__} - check providers.yaml)"]

    # Printed commands, so resolved from `__file__` and never a skill-dir
    # placeholder (see review_plan_orchestrator.parse_args for why).
    run = dispatcher_prefix()
    parser = argparse.ArgumentParser(
        description="Ask each configured LLM a free-text question about a plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[build_parent_parser()],
        epilog=f"""
Available models (provider:model format):
  {', '.join(available_models[:10])}{'...' if len(available_models) > 10 else ''}

Examples:
  {run} \\
    ask plans/my-plan.md "Is the rollback strategy sufficient?"

  # Question passed via a temp file (no question bytes transit the shell):
  {run} \\
    ask plans/my-plan.md --question-file /tmp/q.txt --quick
        """,
    )

    parser.add_argument(
        "--plan-file",
        required=True,
        help="Path to the implementation plan markdown file",
    )

    # The question is supplied via exactly one of three mutually-exclusive,
    # shell-safe mechanisms so user-controlled free text never has to be
    # interpolated into a shell command line.
    qgroup = parser.add_mutually_exclusive_group(required=True)
    qgroup.add_argument(
        "--question",
        default=None,
        help="The question as a single argv value (caller controls quoting)",
    )
    qgroup.add_argument(
        "--question-file",
        default=None,
        help="Read the verbatim question from a file (preferred shell-safe mechanism)",
    )
    qgroup.add_argument(
        "--question-env",
        default=None,
        help="Read the verbatim question from the named environment variable",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models in provider:model format (bare names use default provider)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Use quick_models from providers.yaml (lightweight)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override timeout per model in seconds (default: per-provider timeout)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum number of parallel model invocations (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass resume guards (e.g. act on a stale-plan warning); already-answered "
             "models are kept and only missing answers re-run. For a full re-ask of every "
             "model, also pass --rerun-all.",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="Re-run every model from scratch, discarding any existing per-model result files "
             "(default: resume — skip models that already have results).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def main(argv: Optional[List[str]] = None) -> int:
    # Line buffering + UTF-8/replace stream encoding (Windows-safe output);
    # see utils/stream_bootstrap.py for the full rationale.
    bootstrap_streams()

    args = parse_args(argv)

    # Resolve the verbatim question and require non-empty content.
    question = resolve_question(args)
    if not question.strip():
        print('ERROR: question is empty. Provide a non-empty question, e.g. '
              '--question "Is the rollback strategy sufficient?"')
        return 1

    # Validate plan file. Use realpath (not abspath) so symlinked plan paths
    # match the canonical path agentic CLIs resolve on disk, consistent with
    # output_md_path and the instruction files.
    plan_path = os.path.realpath(args.plan_file)
    if not os.path.isfile(plan_path):
        print(f"ERROR: Plan file not found: {args.plan_file}")
        print(f"       Resolved path: {plan_path}")
        return 1

    # §9.3.a/b: the anchor capture point — absolute, before anything can chdir
    # and before any provider read that decides routing.
    set_run_anchor(plan_path)

    plan_content = read_plan(plan_path)
    plan_hash = compute_plan_hash(plan_content)

    # Mutual exclusivity check (mirror review orchestrators).
    if args.quick and args.interactive:
        print("ERROR: --quick and --interactive are mutually exclusive.")
        return 1

    # Resolve models. D6 row 2: `defaults.models` is the auto-selected answer and
    # an unresolvable set is exit 3 — `ask` never renders a picker unattended.
    unattended = bool(getattr(args, "yes", False))
    try:
        model_specs = resolve_models(
            cli_models=args.models,
            interactive=args.interactive,
            quick=args.quick,
            mode="ask",
            anchor=plan_path,  # per-project config discovery follows the plan-derived root
            unattended=unattended,
        )
    except NeedsHuman as e:
        return unattended_mod.fail_closed(e.message, site_id=e.site)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        available = get_all_model_specs(anchor=plan_path)
        print(f"Use --models flag. Available: {', '.join(available[:5])}...")
        return 1

    if not model_specs:
        print("ERROR: No models selected.")
        return 1

    # Model validation = warn-and-proceed (match existing orchestrators).
    invalid_models = [m for m in model_specs if not is_model_valid(m, anchor=plan_path)]
    if invalid_models:
        print(f"WARNING: Unknown models (proceeding anyway): {', '.join(invalid_models)}")
        # Anchored to match the rejection above (§9.3.c).
        available = get_all_model_specs(anchor=plan_path)
        print(f"Available models: {', '.join(available[:10])}...")

    # Compute slug + resolve the per-question directory (collision/resume guard).
    slug = slugify_question(question)
    phase_dir = str(get_phase_dir(Path(plan_path), "ask"))
    question_dir, is_resume = resolve_question_dir(phase_dir, slug, question)
    os.makedirs(question_dir, exist_ok=True)

    # Plan-freshness check on resume (warn only; --force is how to act on it).
    if is_resume:
        status_path = os.path.join(question_dir, ".status.json")
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    prior = json.load(f)
                if prior.get("plan_hash") and prior.get("plan_hash") != plan_hash:
                    print("WARNING: plan file has changed since these answers were "
                          "generated; existing answers may be stale — re-run with "
                          "--rerun-all to refresh every model's answer.")
            except (json.JSONDecodeError, OSError, IOError):
                pass

    template = load_prompt(ASK_PROMPT_FILE)
    inline_plan, truncated = build_inline_plan(plan_content, plan_path)
    if truncated:
        print(f"WARNING: plan exceeds {MAX_INLINE_PLAN_BYTES // 1024} KB; inlined copy "
              f"truncated (a marker points models at the full plan path).")

    project_root = get_project_root(plan_path) or os.path.dirname(plan_path) or "."
    skip_existing = not args.rerun_all

    print(f"Question slug: {slug}")
    print(f"Output directory: {question_dir}")
    print(f"Models: {', '.join(model_specs)}")
    print(f"Plan: {plan_path} ({len(plan_content)} bytes)")
    if args.rerun_all:
        print("Rerun-all: re-asking all models (ignoring existing answer files)")
    print("")

    # --- the detach boundary -------------------------------------------------
    # `ask` runs live at {plan}/ask/<slug>-<hash8>/, so the run this fork owns is
    # this question's dir — which is also why `status` needs `--question`/`--run-id`
    # to pick one. Everything above (question resolution, the collision/resume
    # guard, model resolution, plan hashing) stays in the foreground.
    detach_outcome = maybe_detach(
        args=args,
        mode="ask",
        phase_dir=question_dir,
        plan_file=plan_path,
        phase="ask",
        ask_slug=slug,
        model_specs=model_specs,
    )
    if detach_outcome is not None:
        sys.exit(detach_outcome)

    write_status(question_dir, {
        "phase": "ask",
        "state": "models_running",
        "started_at": datetime.now().isoformat(),
        "models_requested": list(model_specs),
        "question": question,
        "question_slug": slug,
        "plan_hash": plan_hash,
        "answers_md": os.path.join(question_dir, "answers.md"),
    })
    if unattended and not args.models and not args.quick:
        # D6 row 2, recorded rather than merely applied (see review base).
        unattended_mod.record_resolution(
            question_dir,
            unattended_mod.row(2).site_id,
            "defaults.models: " + ", ".join(model_specs),
        )
    report_progress(
        question_dir, step="fanout",
        models_total=len(model_specs), models_done=0,
    )

    def skip_existing_answer(model_spec: str, index: int) -> Optional[Dict[str, Any]]:
        # Cheap, synchronous resume check (runs before stagger/semaphores):
        # short-circuit models that already have a non-empty answer so resume
        # stays fast. Returns None when the model must (re)run.
        if not skip_existing:
            return None
        sanitized = sanitize_model_name(model_spec)
        answer_path = os.path.realpath(os.path.join(question_dir, f"answer_{sanitized}.md"))
        if read_nonempty_answer(answer_path) is not None:
            print(f"[SKIP] {model_spec} - already has a non-empty answer")
            return {"model": model_spec, "success": True, "error": None, "source": "resume"}
        return None

    async def ask_one(model_spec: str, index: int) -> Dict[str, Any]:
        provider_name, _ = parse_model_spec(model_spec, anchor=plan_path)
        sanitized = sanitize_model_name(model_spec)
        # output_md_path MUST be absolute: provider subprocesses run with
        # cwd=project_root, so a relative path could escape the question dir.
        answer_path = os.path.realpath(os.path.join(question_dir, f"answer_{sanitized}.md"))
        log_file = os.path.join(question_dir, f"log_{sanitized}.txt")

        if skip_existing:
            # The pre-stagger skip_predicate already short-circuited models with a
            # usable answer; reaching here means the file is missing/empty, so the
            # model re-runs (warn if a stale empty file is present).
            if os.path.exists(answer_path):
                print(f"WARNING: answer_{sanitized}.md is empty/unreadable; re-running model")
        else:
            # --rerun-all: drop any stale answer so only content written during THIS
            # run can satisfy the hard-failure-with-file policy below.
            try:
                os.remove(answer_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass

        # provider_name (resolved above) selects the output-channel variant: an
        # adapter declaring supports_file_output = False is asked for stdout
        # rather than for a file it would answer with a refusal sentence.
        prompt, prompt_truncated = render_prompt_guarded(
            template, question, plan_path, inline_plan, answer_path, provider_name
        )
        if prompt_truncated:
            print(f"WARNING: rendered prompt for {model_spec} exceeded "
                  f"{MAX_PROMPT_BYTES // 1024} KB; question truncated to fit the argv "
                  f"size limit (plan path preserved so the model can read the full plan).")
        per_model_timeout = (
            int(args.timeout) if args.timeout is not None
            else get_provider_timeout(provider_name, anchor=plan_path)
        )

        result = await asyncio.to_thread(
            invoke_with_provider,
            prompt=prompt,
            model_spec=model_spec,
            timeout=per_model_timeout,
            log_file=log_file,
            cwd=project_root,
            anchor=plan_path,
        )

        # 1. Primary + hard-failure-with-file policy: a non-empty answer file is
        #    success regardless of how the subprocess exited (agentic CLIs often
        #    write the answer mid-run and then time out during further
        #    exploration). This keeps run-1 and a resumed run classifying the
        #    same artifact identically.
        if read_nonempty_answer(answer_path) is not None:
            return {"model": model_spec, "success": True, "error": None, "source": "file"}

        # 2. Fallback: recover a genuine plain-text answer from the result and
        #    persist it so resume skips this model next time. Success requires
        #    the recovered text to actually land on disk (success ⟺ non-empty
        #    answer file), so aggregation (file-based) always sees it.
        recovered = recover_answer_text(result, log_file)
        if recovered is not None:
            try:
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(recovered)
                return {"model": model_spec, "success": True, "error": None, "source": "fallback"}
            except (OSError, IOError) as e:
                error = f"Recovered answer but failed to persist it: {e}"
                write_error_log(question_dir, model_spec, error, result)
                return {"model": model_spec, "success": False, "error": error, "source": "failed"}

        # 3. Failure: no usable answer file and nothing recoverable.
        error = result.get("error") or "No usable answer produced"
        write_error_log(question_dir, model_spec, error, result)
        return {"model": model_spec, "success": False, "error": error, "source": "failed"}

    results = await run_models_concurrent(
        model_specs, ask_one, args.timeout, args.max_parallel,
        skip_predicate=skip_existing_answer,
        anchor=plan_path,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read-only scan to determine which models succeeded, so we can build the
    # failed map before writing answers.md exactly once (no transient on-disk
    # version missing the ## Failed Models section).
    completed = [spec for spec, _ in collect_completed(question_dir, model_specs)]
    completed_set = set(completed)

    # Every requested model is either completed (has a usable answer file) or
    # failed; build the failed map for the report from that invariant so
    # ## Failed Models and the live results agree.
    failed_models: Dict[str, str] = {}
    for spec in model_specs:
        if spec in completed_set:
            continue
        res = results.get(spec, {})
        reason = (res.get("error") if isinstance(res, dict) else None) \
            or "No answer produced (model did not complete)"
        failed_models[spec] = reason

    # Write answers.md once, with the resolved failed-models section.
    answers_md, completed = write_answers_md(
        question_dir, question, plan_path, model_specs, failed_models, timestamp
    )

    if completed and not failed_models:
        state = "completed"
    elif completed:
        state = "partial"
    else:
        state = "failed"

    # A detached ask hands its answers back through `status`, not through a
    # stdout nobody read (no-op in the foreground).
    record_output_files(question_dir, answers=answers_md)
    report_progress(
        question_dir, step="complete", models_total=len(model_specs),
        models_done=len(completed),
    )

    update_status(question_dir, {
        "state": state,
        "models_completed": completed,
        "models_failed": failed_models,
        "answers_md": answers_md,
    })

    print("")
    print(f"Ask complete: {len(completed)}/{len(model_specs)} models answered "
          f"({len(failed_models)} failed).")
    if failed_models:
        for spec, reason in failed_models.items():
            print(f"  - FAILED {spec}: {reason}")
    # D6 row 3: a partial answer set is usable output — marker + `state:
    # "partial"` + exit 0. Zero answers has nothing to aggregate, so it is exit 3
    # naming the retry command rather than the old bare exit 1.
    row3 = unattended_mod.report_partial_results(
        completed=completed,
        failed=list(failed_models),
        phase_dir=question_dir,
        # D2 row 6: the live question positionally (never a recorded
        # --question), the caller's flags, then the verb.
        retry_command=cli_common.retry_command(
            "ask", plan_path, question, "--force",
            flags=cli_common.current_replay_flags(mode="ask"),
        ),
        unattended=unattended,
    )

    # Final stdout line: the answers.md path (so the instruction layer/Claude
    # can report it).
    print(answers_md)

    return row3


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
