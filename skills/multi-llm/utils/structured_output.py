"""Native structured output — the policy half (§4.2, §4.2.1, §4.2.2).

The adapters own the *argv* (``schema=`` on the four builders, plus
``structured_output_support``); this module owns everything else, because
§4.2.1 puts the decision in the **caller**: ``parse_output`` keeps its
stdout-only signature and never learns about schemas, and no adapter can
therefore silently upgrade prose to validated output.

Four things live here:

1. **The phase → schema-file table** (:data:`PHASE_SCHEMA_FILE`) §4.2.1 says
   lives beside the phase definitions, closed over every phase ``modes.py``
   declares. A phase either names a file under ``skills/multi-llm/schemas/`` or
   is explicitly ``None``; there is no third state and no inline schema at a
   call site.
2. **The transport preconditions** (:func:`native_transport_error`). Measured,
   per CLI, and the reason this is not simply "hand the adapter the file".
3. **The sidecar** (:func:`schema_sidecar`) — codex takes a *path*, so the
   caller writes one temp file per invocation and removes the directory in a
   ``finally`` that covers timeout, kill-path teardown and exception alike.
   Adapters are module-level singletons shared across a concurrent fan-out, so
   they may not hold this state (the same constraint that sank §4.1's T9).
4. **The failure policy** (:func:`classify_attempt`) — §4.2.1's five branches,
   with the one rule that makes the item worth shipping stated as a rule:
   *no combination of exit 0, ``subtype: success`` and ``is_error: false`` is by
   itself a success for a schema-carrying phase.*

Measured 2026-08-16 against the real ``claude`` 2.1.233 binary driven at a
localhost Anthropic stub (zero paid tokens; the captured envelopes are the
fixtures in ``tests/test_structured_output.py``):

* a conforming tool call ⇒ ``exit 0``, ``subtype: success`` **and** a
  ``structured_output`` key, with ``result`` holding the same object serialized;
* a non-conforming tool call ⇒ ``exit 1``, ``subtype:
  error_max_structured_output_retries``, **no** ``structured_output``;
* a prose answer that never calls the tool ⇒ ``exit 0``, ``subtype: success``,
  ``is_error: false``, **no** ``structured_output`` key at all and ``result``
  holding raw, unvalidated text. That third shape is the silent-corruption bug
  this module exists to remove.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .json_extractor import extract_json_from_text
from .providers.base import (
    MAX_STRUCTURED_OUTPUT_RETRIES,
    STRUCTURED_OUTPUT_RETRY_ENV,
    LLMProvider,
    StructuredSchema,
)
from .schema_validator import SCHEMA_DIR, validate_document

__all__ = [
    "MAX_STRUCTURED_OUTPUT_RETRIES",
    "MARKER_NATIVE",
    "MARKER_RECOVERED",
    "MARKER_UNVALIDATED",
    "FAILURE_SCHEMA_VIOLATION",
    "FAILURE_CONFIG_ERROR",
    "FAILURE_TIMEOUT",
    "PHASE_SCHEMA_FILE",
    "SchemaConfigError",
    "StructuredResult",
    "classify_attempt",
    "load_phase_schema",
    "native_transport_error",
    "schema_for_dispatch",
    "schema_sidecar",
    "structured_output_env",
]


# ─────────────────────────────────────────────────────────────────────────────
# The retry budget (§4.2.1), pinned in the child env rather than left to the CLI
# ─────────────────────────────────────────────────────────────────────────────

# ``MAX_STRUCTURED_OUTPUT_RETRIES`` and ``STRUCTURED_OUTPUT_RETRY_ENV`` are
# imported from ``providers.base`` above and re-exported here through
# ``__all__``: an adapter's ``get_env`` is what emits the variable and cannot
# import this module (the reverse edge would be a cycle through
# ``utils.providers.__init__``), so there is one definition and two readers.
# §4.2.1's rationale ("2× a normal call") is an upper bound that is never
# reached: the measured meaning of ``=1`` is one attempt total, so the pin is
# cheaper than the section assumed, not more expensive.


def structured_output_env() -> Dict[str, str]:
    """The child env pin (§4.2.1). One place, so both lanes agree."""
    return {STRUCTURED_OUTPUT_RETRY_ENV: str(MAX_STRUCTURED_OUTPUT_RETRIES)}


# ─────────────────────────────────────────────────────────────────────────────
# Markers and failure reasons
# ─────────────────────────────────────────────────────────────────────────────

#: ``structured_output`` was present and conformant.
MARKER_NATIVE = "native"
#: The document was recovered from ``result`` **and validated against the same
#: schema file**, with one WARN naming the phase and model.
MARKER_RECOVERED = "recovered"
#: No schema was in play at all (an adapter without ``structured_output_support``,
#: a declined §5.4.1 gate, or a phase with no schema file): the prompt-level path
#: ran and nothing validated it. Never confused with ``recovered``.
MARKER_UNVALIDATED = "unvalidated"

#: Handed to the existing retry ladder (§7.1) like any other parse failure.
FAILURE_SCHEMA_VIOLATION = "schema_violation"
#: A schema this run can never satisfy — never retried and never billed.
FAILURE_CONFIG_ERROR = "config_error"
#: §4.2.1 case 5: a timeout is a timeout, never a schema violation.
FAILURE_TIMEOUT = "timeout"


class SchemaConfigError(Exception):
    """A schema file is missing, unreadable or not a JSON object."""


# ─────────────────────────────────────────────────────────────────────────────
# §4.2.2's phase → schema-file table
# ─────────────────────────────────────────────────────────────────────────────

#: Closed over every phase ``modes.py`` declares, plus the two sub-phases that
#: own a JSON contract without owning a mode word (``consolidation``,
#: ``task-decomposition``). ``None`` is a *decision*, not an omission: the
#: free-text phases (``ask``) and the write phases (whose product is a file diff
#: rather than a document) are explicitly out of scope, because attaching a
#: schema there would convert a legitimate prose answer into a failure.
#:
#: ``review-plan`` and ``review-tasks`` were §4.2.2's honest gap — "no file
#: today". They now share ``review_suggestions.schema.json``, transcribed
#: verbatim from the two shipped prompt contracts (which are byte-identical in
#: field set, enum values and the "exactly these fields" rule). One file, not
#: two, because it is one document; two copies would be free to drift.
PHASE_SCHEMA_FILE: Dict[str, Optional[str]] = {
    # ---- the five JSON-document phases §4.2.1 names ------------------------
    "review-plan": "review_suggestions.schema.json",
    "review-tasks": "review_suggestions.schema.json",
    "code-review": "code_review_issues.schema.json",
    "generate-tasks": "generated_tasks.schema.json",
    "consolidation": "consolidation_batch.schema.json",
    # ---- adjacent documents that are already validated on write -----------
    "task-decomposition": "task_decomposition.schema.json",
    # ---- explicitly out of scope (§4.2.1, "Affected phases") --------------
    "ask": None,
    "implement": None,
    "apply-suggestions": None,
    "apply-task-suggestions": None,
    "apply-code-fixes": None,
    "full": None,
}


def load_phase_schema(
    phase: str, schema_dir: Optional[Path] = None
) -> Optional[StructuredSchema]:
    """The schema file for ``phase``, read from disk, or ``None``.

    ``None`` means the phase is deliberately on the prompt-level path. A phase
    that names a file whose contents cannot be read is a **configuration error**
    naming the file — never a silently dropped flag, because the same file is
    the source for §4.2.1 case 2's validation and a run that quietly lost it
    would validate nothing while reporting ``recovered``.
    """
    name = PHASE_SCHEMA_FILE.get(phase)
    if not name:
        return None
    root = Path(schema_dir) if schema_dir is not None else SCHEMA_DIR
    path = root / name
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaConfigError(
            f"phase {phase!r} declares schema {name!r}, but {path} does not "
            f"exist. Schemas are files under skills/multi-llm/schemas/; an "
            f"inline schema at the call site is not an alternative (§4.2.2)."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaConfigError(
            f"phase {phase!r} declares schema {name!r}, but {path} could not be "
            f"read as JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise SchemaConfigError(
            f"phase {phase!r} declares schema {name!r}, but {path} is not a JSON "
            f"object"
        )
    return StructuredSchema(document=document, name=name)


# ─────────────────────────────────────────────────────────────────────────────
# Transport preconditions — measured, per CLI
# ─────────────────────────────────────────────────────────────────────────────

def _strict_openai_errors(
    node: Any,
    path: str = "<root>",
    root: Optional[Dict[str, Any]] = None,
    seen: Optional[set] = None,
) -> List[str]:
    """Where ``node`` breaks OpenAI's *strict* structured-output subset.

    Measured 2026-08-16 against codex-cli 0.146.0: the flag becomes
    ``response_format`` / ``text.format.schema`` named ``codex_output_schema``,
    and the API answers a plain draft-07 schema with ``invalid_json_schema: …
    'additionalProperties' is required to be supplied and to be false``. So the
    constraint is the vendor's, not ours.

    **Necessary conditions, not a complete strict-mode validator.** It encodes
    the two rules the measurement showed; a schema that passes here may still be
    refused by the API for a keyword this does not model. That refusal is not a
    silent one — it arrives as ``invalid_json_schema`` on the stream and
    :func:`classify_attempt` turns it into :data:`FAILURE_CONFIG_ERROR`, naming
    the file, never retried and never billed. Over-approximating the other way
    (declaring more schemas transportable than are) is the failure mode this
    ordering avoids.

    Local ``$ref`` pointers are followed, because that is where the real
    violations hide: ``generated_tasks.schema.json``'s root is strict-clean and
    its ``$defs`` are not.
    """
    errors: List[str] = []
    if not isinstance(node, dict):
        return errors
    if root is None:
        root = node
    if seen is None:
        seen = set()

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in seen or not ref.startswith("#/"):
            return errors
        seen = seen | {ref}
        target: Any = root
        for part in ref[2:].split("/"):
            if not isinstance(target, dict) or part not in target:
                return errors
            target = target[part]
        return _strict_openai_errors(target, path, root, seen)

    node_type = node.get("type")
    if node_type == "object" or "properties" in node:
        if node.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be present and false")
        properties = node.get("properties") or {}
        required = set(node.get("required") or ())
        missing = sorted(set(properties) - required)
        if missing:
            errors.append(
                f"{path}: every property must be required; missing "
                f"{', '.join(missing)}"
            )
        for name, sub in properties.items():
            errors.extend(_strict_openai_errors(sub, f"{path}.{name}", root, seen))
    items = node.get("items")
    if isinstance(items, dict):
        errors.extend(_strict_openai_errors(items, f"{path}[]", root, seen))
    for keyword in ("allOf", "anyOf", "oneOf"):
        for index, sub in enumerate(node.get(keyword) or ()):
            errors.extend(
                _strict_openai_errors(sub, f"{path}.{keyword}[{index}]", root, seen)
            )
    return errors


def native_transport_error(
    provider: LLMProvider, schema: StructuredSchema
) -> Optional[str]:
    """Why ``provider`` cannot carry ``schema`` natively, or ``None``.

    **Not an invented restriction and not a second schema.** Both rules below
    were measured, and both resolve the way a declined §5.4.1 gate does: the flag
    is dropped, the phase runs its prompt-level JSON path, and §4.2.1's recovery
    branch marks the result ``recovered``/``unvalidated`` — never ``native``.
    Emitting a flag the CLI will reject is the one outcome neither posture may
    produce.

    * **claude** — ``--json-schema`` becomes a 25th tool named
      ``StructuredOutput`` whose ``input_schema`` is the schema **verbatim**
      (measured: the stub receives it unwrapped). A tool input is an object, so
      an array-rooted schema can never be satisfied: measured, the model's array
      answer was rejected client-side on every attempt and the run ended on
      ``error_max_structured_output_retries`` having burned the whole budget.
      ``code_review_issues.schema.json`` and ``review_suggestions.schema.json``
      are array-rooted, so those phases stay on the prompt-level path until the
      documents themselves change — which is a contract change for every
      consumer, not something M7 may do at a call site.
    * **codex** — ``--output-schema`` is OpenAI *strict* structured output, a
      proper subset of draft-07 (object root, ``additionalProperties: false`` on
      every object, every property in ``required``).
    """
    if not getattr(provider, "structured_output_support", False):
        return (
            f"{provider.name} declares no structured_output_support, so "
            f"{schema.name} is not passed to it"
        )
    document = dict(schema.document)
    if document.get("type") != "object":
        return (
            f"{provider.name} cannot carry {schema.name}: its root type is "
            f"{document.get('type')!r} and the flag's transport requires an "
            f"object"
        )
    if provider.name == "codex":
        errors = _strict_openai_errors(document)
        if errors:
            return (
                f"codex cannot carry {schema.name}: OpenAI's strict "
                f"structured-output subset rejects it — " + "; ".join(errors[:3])
            )
    return None


#: One WARN per (provider, phase) per process, not one per dispatch.
_DECLINED: set = set()


def reset_declined_cache() -> None:
    """Forget the per-process WARN bookkeeping (tests only)."""
    _DECLINED.clear()


def schema_for_dispatch(
    provider: LLMProvider, phase: str, *, warn: bool = True
) -> Tuple[Optional[StructuredSchema], str]:
    """``(schema_or_None, marker_when_no_schema)`` — the pre-dispatch read.

    §4.2.2: *the caller reads the flag before dispatch.* ``False`` (or a schema
    this CLI cannot carry, or a phase with no schema file) means the phase runs
    its prompt-level JSON path and the result is marked ``recovered`` /
    ``unvalidated`` per §4.2.1 — never ``native``.
    """
    schema = load_phase_schema(phase)
    if schema is None:
        return None, MARKER_UNVALIDATED
    reason = native_transport_error(provider, schema)
    if reason is None:
        return schema, MARKER_NATIVE
    key = (provider.name, phase)
    if warn and key not in _DECLINED:
        _DECLINED.add(key)
        print(
            f"[WARN] structured output: {reason}. The {phase} phase falls back "
            f"to prompt-level JSON plus schema-validated recovery; the result is "
            f"marked '{MARKER_RECOVERED}', never '{MARKER_NATIVE}'.",
            file=sys.stderr,
        )
    # The file still validates the recovered document, so the phase is
    # `recovered`-capable rather than merely `unvalidated`.
    return None, MARKER_RECOVERED


# ─────────────────────────────────────────────────────────────────────────────
# codex's sidecar — written by the caller, removed in a `finally`
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def schema_sidecar(schema: StructuredSchema) -> Iterator[StructuredSchema]:
    """One ``mkdtemp`` per invocation carrying ``schema``, removed on the way out.

    The directory is removed in a ``finally`` so it is cleaned up on timeout, on
    kill-path teardown and on exception alike (§4.2.1 case 5). A temp dir that
    cannot be created or written fails the attempt **before** spawn rather than
    silently dropping the flag.
    """
    directory: Optional[str] = None
    try:
        try:
            directory = tempfile.mkdtemp(prefix="multi-llm-schema-")
            path = os.path.join(directory, schema.name or "schema.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(dict(schema.document), handle, sort_keys=True)
        except OSError as exc:
            raise SchemaConfigError(
                f"could not materialize {schema.name} for the CLI's "
                f"--output-schema: {exc}. Nothing was dispatched."
            ) from exc
        yield schema.with_path(path)
    finally:
        if directory:
            shutil.rmtree(directory, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# §4.2.1 — the failure policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StructuredResult:
    """One attempt's verdict.

    ``marker`` is set **only** on acceptance (``native`` / ``recovered``);
    ``failure_reason`` is set only on refusal. They are never both set, which is
    what stops a caller from reading a violation as a degraded success.
    """

    marker: Optional[str] = None
    data: Any = None
    failure_reason: Optional[str] = None
    warning: Optional[str] = None
    retryable: bool = True


#: claude's own refusals, verbatim at 2.1.233. Both are emitted **before any
#: request** (0 requests, nothing billed), and both name the flag.
_SCHEMA_CONFIG_MARKERS = (
    "--json-schema is not valid json",
    "--json-schema is not a valid json schema",
    "--output-schema",
    "invalid_json_schema",
    "invalid schema for response_format",
)

#: claude's terminal subtype when the model called the tool and never satisfied
#: it. The CLI already spent its (pinned) budget, so **we** do not re-run it.
_MAX_RETRIES_SUBTYPE = "error_max_structured_output_retries"


def _envelope(stdout: str) -> Optional[Dict[str, Any]]:
    """The result envelope, from stdout only (§4.2.1, "Parser inputs")."""
    text = (stdout or "").strip()
    if text.startswith("{"):
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            return candidate
    # NDJSON (codex): the last event carrying the key wins.
    found: Optional[Dict[str, Any]] = None
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "structured_output" in event:
            found = event
    return found


def _validate(document: Any, schema: StructuredSchema) -> List[str]:
    return validate_document(document, dict(schema.document))


def classify_attempt(
    schema: Optional[StructuredSchema],
    *,
    stdout: str,
    stderr: str = "",
    exit_code: Optional[int] = 0,
    timed_out: bool = False,
    phase: str = "",
    model: str = "",
) -> StructuredResult:
    """§4.2.1's decision, applied to one attempt's captured streams.

    Ordered deliberately: a timeout is classified before anything reads the
    streams (case 5), and a configuration error before anything reads the
    envelope (case 4), because both would otherwise be laundered into the
    ``structured_output``-absent branch and re-run by the ladder.
    """
    if timed_out:
        return StructuredResult(
            failure_reason=FAILURE_TIMEOUT,
            retryable=True,
            warning=None,
        )

    where = " ".join(part for part in (phase, model) if part)
    haystack = f"{stdout or ''}\n{stderr or ''}".lower()
    if any(marker in haystack for marker in _SCHEMA_CONFIG_MARKERS):
        name = schema.name if schema is not None else "the configured schema"
        return StructuredResult(
            failure_reason=FAILURE_CONFIG_ERROR,
            retryable=False,
            warning=(
                f"structured output: the CLI refused {name} before making any "
                f"request{f' ({where})' if where else ''}. Fix the schema file; "
                f"this is never retried and nothing was billed."
            ),
        )

    envelope = _envelope(stdout)

    # Case 3 of §4.2.1's numbering: the CLI spent its own budget.
    if envelope is not None and envelope.get("subtype") == _MAX_RETRIES_SUBTYPE:
        return StructuredResult(
            failure_reason=FAILURE_SCHEMA_VIOLATION,
            retryable=False,
            warning=(
                f"structured output: the CLI exhausted its "
                f"{STRUCTURED_OUTPUT_RETRY_ENV} budget"
                f"{f' ({where})' if where else ''}; not re-run here."
            ),
        )

    if schema is None:
        # No schema was in play: nothing to validate, nothing to claim.
        return StructuredResult(marker=MARKER_UNVALIDATED, data=None)

    if exit_code not in (0, None):
        return StructuredResult(failure_reason=FAILURE_SCHEMA_VIOLATION)

    native = envelope.get("structured_output") if envelope else None
    if native is not None:
        # Case 4: defence in depth. The CLI validates client-side, so this
        # should be unreachable — and if it is reached it takes case 2's failure
        # branch rather than being trusted.
        errors = _validate(native, schema)
        if not errors:
            return StructuredResult(marker=MARKER_NATIVE, data=native)
        return StructuredResult(
            failure_reason=FAILURE_SCHEMA_VIOLATION,
            warning=(
                f"structured output: `structured_output` was present but did "
                f"not validate against {schema.name}: {errors[0]}"
            ),
        )

    # Case 2: absent, exit 0. Recover, then validate against the SAME file.
    result_text = envelope.get("result") if envelope else None
    if not isinstance(result_text, str):
        result_text = stdout or ""
    recovered = extract_json_from_text(result_text, prefer_arrays=True)
    if not recovered.get("success"):
        return StructuredResult(failure_reason=FAILURE_SCHEMA_VIOLATION)
    document = recovered.get("data")
    errors = _validate(document, schema)
    if errors:
        return StructuredResult(failure_reason=FAILURE_SCHEMA_VIOLATION)
    return StructuredResult(
        marker=MARKER_RECOVERED,
        data=document,
        warning=(
            f"structured output: no `structured_output` key"
            f"{f' ({where})' if where else ''}; the document was recovered from "
            f"the prose answer and validated against {schema.name}. Marked "
            f"'{MARKER_RECOVERED}'."
        ),
    )
