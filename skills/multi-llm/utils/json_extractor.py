"""Robust JSON extraction with multiple fallback strategies.

This module provides a robust way to extract JSON from text that may contain
extra content, duplicate JSON structures, or other issues that break simple
regex-based extraction.

The key insight is that JSON has balanced brackets. We find all complete JSON
structures by tracking bracket depth, handling strings correctly (not counting
brackets inside "..."), and returning each structure individually.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in filenames.

    Replaces all non-alphanumeric characters except hyphen and underscore with underscore.

    Args:
        model: Model specification (e.g., 'cursor-agent:gpt-5.2-high')

    Returns:
        Sanitized string safe for filenames (e.g., 'cursor-agent_gpt-5_2-high')
    """
    return re.sub(r'[^a-zA-Z0-9\-_]', '_', model)


def _warn_stem_collision(
    phase_dir: str, stem: str, specs: List[str], retained: str
) -> None:
    """Name the specs that share one stem, and what the report will claim instead.

    Split out of :func:`build_unsanitize_map` so the loop stays a loop and this
    stays prose: the operator-facing half of a collision is the *consequence*,
    not the detection, and it takes more words than the detection does.

    **stderr, not stdout.** The reaggregate path's stdout is its
    machine-readable channel — the ``[PARTIAL_RESULTS: ...]`` and
    ``[REAGGREGATE_AFTER_SALVAGE]`` markers a host parses — and a degradation
    nobody chose does not belong in it. Same rule, and for the same reason, as
    ``warn_stem_attribution`` (``code_review_orchestrator.py``), which says so at
    length for the sibling case.

    **A plain print rather than that shared helper**, because the helper lives in
    an orchestrator and this is a leaf module every orchestrator imports;
    reaching back up for it would make ``utils.json_extractor`` depend on
    ``code_review_orchestrator``. A ``WARNING:``-headlined stderr print is what
    the rest of the package already does for exactly this class of
    "continuing, but degraded" notice (``utils/status_file.py``,
    ``utils/tasks_file.py``, ``utils/filtering.py``).
    """
    listed = ", ".join(repr(spec) for spec in specs)
    print(
        f"WARNING: {phase_dir}/.status.json lists {len(specs)} distinct model "
        f"specs that sanitize to one filename stem {stem!r}: {listed}.\n"
        f"  Every group attributed to {stem} is reported as {retained!r} — the "
        f"spec listed last, which need not be the model that produced it. "
        f"sanitize_model_name flattens every separator a spec carries and "
        f"nothing can put one back, so this phase dir cannot say which of those "
        f"models {stem}.json came from; during the fan-out they wrote that one "
        f"file between them, so at most one of their answers survived and the "
        f"resume predicate would have skipped the second as already done.\n"
        f"  resolve_models (utils/interactive.py) refuses a colliding pair "
        f"before dispatch, so this checkpoint predates that gate or was written "
        f"by hand. Recover by editing models_requested / models_completed down "
        f"to the model this dir's results actually came from, or re-run the "
        f"fan-out with --force once the selected specs no longer collide.",
        file=sys.stderr,
    )


def build_unsanitize_map(phase_dir: str) -> Dict[str, str]:
    """Build a mapping from sanitized model names back to original specs.

    Reads ``.status.json`` in *phase_dir* and uses ``models_requested`` /
    ``models_completed`` to create a ``sanitize(original) -> original`` dict.

    **The inverse exists only while the map is one-to-one, and this is the last
    place that can still notice it is not.** :func:`sanitize_model_name` is lossy
    in one direction only — every separator a spec can carry (``:``, ``.``,
    ``/``) collapses to the same ``_`` — so ``p:a.b`` and ``p:a_b`` are two
    models with one stem, ``p_a_b``, and so are
    ``openrouter:qwen/qwen3-max`` and ``openrouter:qwen.qwen3-max``.
    ``resolve_models`` (``utils/interactive.py``) refuses such a pair before
    anything is dispatched, which is where refusing costs a correct run nothing.
    But ``--reaggregate`` resolves nothing at all and only reads this file, so
    the gate never runs for it: a checkpoint written before the gate existed, or
    written by hand — which is a recovery ``warn_stem_attribution``
    (``code_review_orchestrator.py``) actively recommends — can still hold a
    colliding pair. Left alone the loop below is a plain dict write, the spec
    listed *last* silently wins, and every group under that stem is attributed
    to a model that may not have produced it. That is a **misattribution, not an
    omission**: the report names a model that did not say the thing, and reads
    as plausible either way.

    **The collision is reported and the run continues, deliberately.** The same
    posture ``warn_stem_attribution`` takes for the sibling case — a phase dir
    whose checkpoint records no models at all still reaggregates, warning that
    attribution has fallen back to stems — because the two degradations have to
    tell the operator one story rather than two. Raising here would also throw
    away a fan-out that already completed in order to protect the *naming* of
    its results, which is the wrong trade at this end of the run; fail-closed
    belongs at ``resolve_models``, before the spend.

    Returns:
        Dict mapping sanitized names to original model specs. On a collision the
        last spec still wins — the mapping is unchanged — but one warning per
        colliding stem goes to stderr first.
        Empty dict if ``.status.json`` is missing or unreadable.
    """
    status_path = Path(phase_dir) / ".status.json"
    if not status_path.exists():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    mapping: Dict[str, str] = {}
    # Distinct specs per stem, first-seen order, accumulated across BOTH keys
    # before anything is reported. Both halves of that matter:
    #
    #  * a run that resumed lists its models under ``models_requested`` and again
    #    under ``models_completed``, so a colliding spec's twin is as likely to
    #    sit in the other key as beside it — detecting per key would miss the
    #    commonest real shape;
    #  * warning inside the loop would chain a three-way collision into two
    #    pairwise warnings, the second of which has already forgotten the first
    #    spec. One warning per stem naming every spec is the thing an operator
    #    can act on.
    #
    # Repeats of the *same* spec are not a collision — that is just the union of
    # the two keys doing its job — so identity, not arrival, is what counts.
    by_stem: Dict[str, List[str]] = {}
    for key in ("models_requested", "models_completed"):
        for original in data.get(key, []):
            stem = sanitize_model_name(original)
            seen = by_stem.setdefault(stem, [])
            if original not in seen:
                seen.append(original)
            mapping[stem] = original
    for stem, specs in by_stem.items():
        if len(specs) > 1:
            _warn_stem_collision(phase_dir, stem, specs, mapping[stem])
    return mapping


def find_json_candidates(text: str, char_type: str = "array") -> List[str]:
    """Find all potential JSON structures using balanced bracket matching.

    Args:
        text: Text to search
        char_type: 'array' for [...], 'object' for {...}, 'both' for either

    Returns:
        List of candidate JSON strings, ordered by position in text
    """
    candidates: List[Tuple[int, str]] = []
    chars: List[Tuple[str, str]] = []

    if char_type in ("array", "both"):
        chars.append(("[", "]"))
    if char_type in ("object", "both"):
        chars.append(("{", "}"))

    for start_char, end_char in chars:
        i = 0
        while i < len(text):
            if text[i] == start_char:
                depth = 1
                in_string = False
                escape_next = False
                j = i + 1

                while j < len(text) and depth > 0:
                    c = text[j]
                    if escape_next:
                        escape_next = False
                    elif c == "\\" and in_string:
                        escape_next = True
                    elif c == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if c == start_char:
                            depth += 1
                        elif c == end_char:
                            depth -= 1
                    j += 1

                if depth == 0:
                    candidates.append((i, text[i:j]))
            i += 1

    # Sort by position, return just the strings
    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates]


def _candidate_spans(text: str, char_type: str) -> List[Tuple[int, int, str]]:
    """Recover each candidate's ``(start, end, text)`` span, in position order.

    :func:`find_json_candidates` returns the candidate strings only, but the
    last-document rule has to know which candidates sit *inside* another one —
    reversing a flat list would return the inner ``[1, 2]`` of ``[{"a": [1, 2]}]``.
    At most one balanced run begins at any index, so candidate starts are
    strictly increasing and a forward-only search recovers every span without
    a second scan of the text.
    """
    spans: List[Tuple[int, int, str]] = []
    pos = 0
    for candidate in find_json_candidates(text, char_type):
        start = text.find(candidate, pos)
        if start < 0:  # pragma: no cover - defensive; find() cannot miss here
            continue
        spans.append((start, start + len(candidate), candidate))
        pos = start + 1
    return spans


def extract_json_from_text(text: str, prefer_arrays: bool = True) -> Dict[str, Any]:
    """Extract JSON with multiple fallback strategies.

    **The last document wins, at every level** — the last envelope record (the
    adapters' own rule), then the last top-level candidate inside that record's
    text — with shape preference consulted before position, and a contained
    candidate never outranking the one that contains it. Models routinely emit a
    draft or a decoy before the real answer in one turn; the answer is the one
    that comes last.

    Four clauses make that concrete:

    1. **Top-level first.** A candidate contained in another candidate's span is
       ranked *after* every top-level one rather than dropped, so an inner
       structure is still salvageable when the outer one does not parse.
    2. **Shape before position.** ``prefer_arrays`` still tries every array
       before every object; last-wins applies *within* each group, so a trailing
       illustrative object never displaces a preceding array.
    3. **Non-empty before empty.** The last candidate that parses to a non-empty
       value, else the last that parses at all.
    4. **Code fences follow the same rule.** The last fenced block that parses
       to a non-empty value wins, else the last that parses at all, so the
       fenced path and the bare-text path are on one rule.

    Strategies in order:
    1. Code blocks (```json ... ```), last non-empty parseable one
    2. Balanced bracket matching - find all candidates, ranked by the rule above
    3. Try parsing each candidate in rank order

    Args:
        text: Text that may contain JSON
        prefer_arrays: If True, try array candidates before objects

    Returns:
        Dict with 'success', 'data'/'error', optionally 'raw'
    """
    text = text.strip()

    # Strategy 1: Code blocks - try both json-tagged and untagged.
    # Clause 4: the LAST fenced block that parses wins - and clause 3 binds the
    # fenced path too, so a trailing empty fence never displaces an earlier
    # non-empty one. Walk the fences in reverse, keep the first empty document
    # seen as the fallback, and serve it only when no fence carries a value.
    fence_matches = list(
        re.finditer(r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text)
    )
    empty_fence = None
    for match in reversed(fence_matches):
        try:
            fence_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if fence_data:
            return {"success": True, "data": fence_data}
        if empty_fence is None:
            empty_fence = fence_data
    if empty_fence is not None:
        return {"success": True, "data": empty_fence}

    # Strategy 2: Balanced bracket extraction.
    # Clause 2: shape before position - every array is tried before every
    # object when prefer_arrays is set, and last-wins applies within a group.
    if prefer_arrays:
        groups = [_candidate_spans(text, "array"), _candidate_spans(text, "object")]
    else:
        groups = [_candidate_spans(text, "both")]

    # A candidate sits inside another one iff some earlier-starting candidate
    # ends at or after it — balanced runs nest, they never partially overlap —
    # so one pass over the position-sorted union settles containment across both
    # groups without comparing every pair.
    contained_spans = set()
    furthest_end = -1
    for start, end, _ in sorted(span for group in groups for span in group):
        if end <= furthest_end:
            contained_spans.add((start, end))
        else:
            furthest_end = end

    # Clause 1: within each group, every top-level candidate ranks ahead of
    # every contained one; within each tier the LAST by position ranks first.
    candidates: List[str] = []
    for group in groups:
        for contained in (False, True):
            candidates.extend(
                span[2]
                for span in sorted(group, key=lambda s: s[0], reverse=True)
                if ((span[0], span[1]) in contained_spans) is contained
            )

    # Strategy 3: Try parsing each candidate in rank order.
    # Clause 3: the highest-ranked candidate that parses to a NON-EMPTY value,
    # else the highest-ranked that parses at all.
    empty_fallback = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if data:
            return {"success": True, "data": data}
        if empty_fallback is None:
            empty_fallback = data
    if empty_fallback is not None:
        return {"success": True, "data": empty_fallback}

    return {
        "success": False,
        "error": "No valid JSON found in text",
        "raw": text,
        "data": None,
    }


#: ANSI colour/OSC escapes, removed before anything goes looking for a bracket.
#: A provider CLI paints its own diagnostics — kilocode's dead-account failure
#: arrives as ``\x1b[91m\x1b[1mError: \x1b[0mUser not found.`` — and every one of
#: those ``\x1b[`` sequences is an opening bracket with no JSON behind it. This
#: is a second copy of the pattern ``providers/cursor_agent`` keeps for reading
#: *its* human-readable output, and deliberately so: ``json_extractor`` is below
#: the provider layer and importing upward would invert the dependency.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

#: The field names a model *answer* record carries and a failure envelope does
#: not. Both shipped answer schemas require ``title``, ``desc`` and
#: ``importance`` on every record (``schemas/review_suggestions.schema.json``
#: adds ``reference``, ``schemas/code_review_issues.schema.json`` adds
#: ``line_range``, ``anchor_text`` and ``suggested_fix``), so a real record trips
#: this set on its first key. The two required names deliberately left out are
#: ``type`` and ``file``: they are shared with the very shapes this set is here
#: to reject — ``{"type": "error", "message": …}`` is a provider event record
#: verbatim — and leaving them out costs nothing, because no record can carry
#: them without also carrying the three above. ``severity`` is the one name from
#: outside the schemas: it is the near-miss a model writes in place of
#: ``importance``, and repairing a near-miss is exactly what salvage is for.
_ANSWER_FIELD_NAMES = frozenset(
    {
        "title",
        "desc",
        "importance",
        "severity",
        "reference",
        "line_range",
        "anchor_text",
        "suggested_fix",
    }
)

#: A JSON structure that was **started**: ``[`` or ``{`` followed by a token
#: only JSON puts in that position — a string or a nested structure. Truncation
#: is why balanced matching alone is not enough: ``[{"title": "x"`` never closes,
#: so :func:`find_json_candidates` cannot see it, and a half-written array is
#: precisely what salvage is for. Requiring that token is what keeps ``[ERROR]``,
#: ``[91m`` and ``[docs]`` — brackets from log levels, escapes and prose — out,
#: and with them ``[kilocode:openrouter/moonshotai/kimi-k3]`` and
#: ``{code: 401, message: "x"}``: a colon-tagged log prefix (this tool's own
#: ``[provider:model]`` format among them) and a brace dict written the way a
#: language that is not JSON writes one. An unquoted key is not one of the
#: admissible tokens, mechanical damage though it is: a model emitting JSON
#: quotes its keys, ``json.loads`` refuses an unquoted one in any case, and every
#: truncation this clause exists to catch opens ``[{"``, ``{"`` or ``[[`` — so
#: the only documents ``name:`` after a bracket ever brings in are log lines.
_JSON_STRUCTURE_OPENING = re.compile(r'[\[{]\s*["\[{]')


def is_refused_result_envelope(data: Any) -> bool:
    """True when ``data`` is a result envelope the CLI itself flagged as failed.

    ``claude -p --output-format json`` and ``cursor-agent --output-format json``
    answer a failed run with the **same** envelope a successful one gets —
    ``{"type": "result", …}`` — and say so in two fields, ``is_error`` and
    ``terminal_reason``. The verdict rule is
    :func:`utils.providers.result_envelope.envelope_verdict`'s and is consulted
    there rather than re-spelled here; the late import is because that module
    imports :func:`extract_json_from_text` from this one.

    Why the gate needs it: that envelope's bookkeeping carries empty arrays —
    ``permission_denials: []`` at the top level and ``usage.iterations: []``
    one key down — and both :func:`is_answer_shaped` (an object with an
    array-valued key) and clause 1 of :func:`has_salvageable_json` (a nested
    candidate that parses to ``[]``) read those as an answer the model wrote.
    Measured 2026-09-17 on ``claude-code:claude-fable-5-1:high``: a 491 s
    review that ended on a 429 usage cap (``api_error_status: 429``,
    ``result: "You've reached your Fable limit…"``) printed ``[SALVAGE_NEEDED]``
    and handed the host that envelope as a document to repair. Every failure
    capture under ``tests/fixtures/claude-code/`` has the same shape.
    """
    if not isinstance(data, dict):
        return False
    try:
        from .providers.result_envelope import envelope_verdict
    except ImportError:  # the flat import style ``utils/llm_client._redact`` also allows
        from utils.providers.result_envelope import envelope_verdict
    return bool(envelope_verdict(data))


def is_answer_shaped(data: Any) -> bool:
    """True when a parsed value could be the answer, or hold it.

    An array is an answer whatever it contains, ``[]`` included. An object
    qualifies two ways: one array-valued key (the model wrapped its array), or
    one key the answer schemas require (the model emitted a single record where
    an array was asked for, or truncated the array away around one). Anything
    else — an error envelope, a provider event, a bare scalar map — is a
    document that parsed and still holds nothing for a repair pass to find.

    Public because it is the rule at **both** ends of salvage, and one rule
    spelled twice is two rules waiting to disagree:

    * the **gate** — :func:`has_salvageable_json` — asks it of a candidate found
      in the raw output: *is there an answer here to ask a repair pass for?* A
      ``{"error": {"code": 401, …}}`` body parses and holds nothing, so no
      salvage is dispatched and the operator is sent to the provider's auth
      rather than to a JSON repair.
    * the **persist step** — ``utils/subagent_executor._salvage_gave_up`` — asks
      it of what the salvage subagent came back with: *is what came back an
      answer?* A salvage task's ``output_name`` **is** the model's own result
      file, so a subagent that faithfully "recovered" the relayed error envelope
      out of a truncated stream would have that envelope written over the model's
      review, and reaggregation would count a model that never produced a
      parseable word as a participant.

    The same shapes fail at both ends for the same reason, which is why the
    predicate is shared rather than reimplemented at the second one.
    """
    if isinstance(data, list):
        return True
    if isinstance(data, dict):
        # The CLI's own verdict outranks the shape: a refused result envelope
        # carries ``permission_denials: []`` and holds nothing to repair.
        if is_refused_result_envelope(data):
            return False
        if any(isinstance(value, list) for value in data.values()):
            return True
        return not _ANSWER_FIELD_NAMES.isdisjoint(data)
    return False


def has_salvageable_json(text: str) -> bool:
    """True when *text* still holds a model answer a salvage pass could repair.

    Salvage repairs *mechanical* damage to an answer the model did produce —
    truncation, a markdown fence, prose around the array, a trailing comma. It
    cannot conjure an answer that was never produced, and its prompt ends "if
    nothing recoverable is present, print an empty {type}"
    (``subagent_prompts.build_salvage_prompt``). So dispatching salvage at output
    with no JSON in it at all buys exactly one thing: an empty array written to
    that model's result file, where reaggregation reads it as "this model
    reviewed the diff and found nothing". A hard credential failure
    (``Error: User not found.`` on a non-zero exit) is that shape, and it is also
    the case where a ``[SALVAGE_NEEDED]`` marker sends the operator to a JSON
    repair when the fix is the provider's auth.

    Two ways text qualifies, in order:

    1. **A balanced candidate that parses *into an answer's shape*.** A fenced
       array, an array with prose either side of it, an object where an array
       was wanted, or the inner object of an array truncated after it —
       :func:`find_json_candidates` finds all of them. Parsing separates a real
       structure from ``[ERROR]``, but it does not separate an answer from a
       failure, because a failure is frequently well-formed JSON:
       ``{"error": {"code": 401, "message": "User not found"}}`` and the
       ``{"type": "error", "message": …}`` record a provider CLI prints as an
       event both parse perfectly and hold nothing to repair. So the parsed
       value has to be an **array** (``[]`` included — a model that found
       nothing wrote an answer), an **object with an array-valued key** (a model
       wrapped its array in an object), or an **object naming a field the answer
       schemas require** (:data:`_ANSWER_FIELD_NAMES`). Objects are not rejected
       wholesale: a wrapped array is genuine salvage.
    2. **Failing that, a structure that was started.** Clause 1 cannot see a
       truncated document, because an unclosed bracket is not a candidate. Only
       openings *outside* every candidate clause 1 parsed and turned down count
       here — the ``{`` an error envelope opens is a document that closed and
       said nothing, not a truncation — which is what lets a reply carrying both
       (an envelope, then a half-written array) still qualify on the array.

    The expected type is deliberately not a parameter: a model that wrapped the
    array in an object (``{"suggestions": [ …``) is the ordinary salvage case,
    not a second failure mode, so both shapes count.

    One document is turned down by *verdict* rather than by shape: a result
    envelope the CLI itself flagged as failed
    (:func:`is_refused_result_envelope`). Its empty bookkeeping arrays are what
    made every committed claude-code failure capture — and the 2026-09-17 usage
    cap — pass both clauses, and the candidates nested inside its span are the
    same document, not a second one, so they are skipped rather than parsed.

    Args:
        text: Raw model output — whichever stream the failure left behind.

    Returns:
        True when a salvage subagent would have something to work with. False
        means this is not a parse failure at all, and the caller must fail the
        model cleanly instead (the D6 row-3 partial-results path already covers
        a model that produced nothing).
    """
    body = _ANSI_ESCAPE.sub("", text or "")
    if not body.strip():
        return False

    # Clause 1. A candidate that parses but holds no answer is *turned down*,
    # never returned on, so a reply that opens with an error envelope is still
    # read to the end. Its span is kept: balanced runs nest or stay disjoint and
    # arrive in increasing start order, so extending the open interval whenever
    # the next span begins inside it leaves a short, sorted, disjoint list of
    # the regions clause 2 must not read a truncation into.
    settled: List[Tuple[int, int]] = []
    # The end of the last refused result envelope seen. Candidates arrive in
    # increasing start order and nest or stay disjoint, so everything that
    # starts before this offset is *inside* that envelope — its ``usage`` block,
    # its ``iterations: []``, its ``permission_denials: []`` — and is the
    # failure report's own bookkeeping, not a second document. It is scoped to
    # the refused envelope on purpose: an ordinary turned-down object still lets
    # its children speak, because ``{"data": {"issues": [...]}}`` is a doubly
    # wrapped answer and the outer object is not the one that knows.
    refused_until = -1
    for start, end, candidate in _candidate_spans(body, "both"):
        if start < refused_until:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if is_answer_shaped(data):
            return True
        if is_refused_result_envelope(data):
            refused_until = end
        if settled and start <= settled[-1][1]:
            settled[-1] = (settled[-1][0], max(settled[-1][1], end))
        else:
            settled.append((start, end))

    # Clause 2. Both sequences ascend, so one pointer over *settled* walks them
    # together rather than re-scanning the list per opening.
    index = 0
    for match in _JSON_STRUCTURE_OPENING.finditer(body):
        opening = match.start()
        while index < len(settled) and settled[index][1] <= opening:
            index += 1
        if index == len(settled) or opening < settled[index][0]:
            return True

    return False


def generate_output_path(
    out_dir: Union[str, Path],
    prefix: str,
    phase: str,
    model_spec: str
) -> Path:
    """Generate output file path for JSON results in phase subdirectory.

    Creates a path for LLM JSON output using the pattern:
    {out_dir}/{phase_dir}/{model}.json

    Phase mapping (internal to kebab-case):
    - plan_review -> review-plan
    - code_review -> code-review

    Args:
        out_dir: Base output directory (plan's output folder)
        prefix: File prefix (unused in new structure, kept for compatibility)
        phase: Operation phase (e.g., 'code_review', 'plan_review')
        model_spec: Model specification (e.g., 'cursor-agent:auto')

    Returns:
        Path object for the output file
    """
    out_dir = Path(out_dir)

    # Map internal phase names to kebab-case directory names
    phase_dir_map = {
        'plan_review': 'review-plan',
        'code_review': 'code-review',
        'task_review': 'review-tasks',
    }
    phase_dir_name = phase_dir_map.get(phase, phase.replace('_', '-'))

    # Create phase subdirectory
    phase_dir = out_dir / phase_dir_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    safe_model = sanitize_model_name(model_spec)
    filename = f"{safe_model}.json"

    return phase_dir / filename


def read_json_from_file(
    file_path: Union[str, Path],
    prefer_arrays: bool = True
) -> Dict[str, Any]:
    """Read and parse JSON from a file with fallback extraction.

    Attempts to read and parse JSON from a file. If direct parsing fails,
    falls back to using extract_json_from_text() to handle code blocks,
    extra content, or other formatting issues.

    Args:
        file_path: Path to the JSON file
        prefer_arrays: If True, prefer array candidates over objects

    Returns:
        Dict with keys:
        - 'success': bool indicating if JSON was successfully parsed
        - 'data': Parsed JSON data (if successful)
        - 'error': Error message (if failed)
        - 'source': Where the JSON came from ('file', 'file_extracted', 'missing', 'error', 'empty')
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "source": "missing",
            "data": None,
        }

    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except (IOError, OSError, UnicodeDecodeError) as e:
        return {
            "success": False,
            "error": f"Read error: {type(e).__name__}: {e}",
            "source": "error",
            "data": None,
        }

    if not content:
        return {
            "success": False,
            "error": "File is empty",
            "source": "empty",
            "data": None,
        }

    # Try direct JSON parse first
    try:
        data = json.loads(content)
        return {"success": True, "data": data, "source": "file"}
    except json.JSONDecodeError:
        pass

    # Fall back to extraction (handles code blocks, etc.)
    result = extract_json_from_text(content, prefer_arrays=prefer_arrays)
    if result.get("success"):
        result["source"] = "file_extracted"
    else:
        result["source"] = "file_extraction_failed"
    return result
