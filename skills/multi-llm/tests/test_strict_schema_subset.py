"""Every schema the plugin hands a provider as structured output fits OpenAI's
strict subset, and the gate that decides "hand it over or decline" knows the
rules the API enforces.

Measured 2026-10-04 (codex-cli 0.160.0, gpt-6-luna, the three-host re-run):
``generated_tasks.schema.json`` declared ``"schema_version": {"const": "1.0"}``
with no ``type``. The local strict check (``_strict_openai_errors``) only knew
``additionalProperties: false`` and "every property required", so it passed the
file, codex sent it as ``text.format.schema``, and OpenAI answered every
generate-tasks dispatch with ``invalid_json_schema … In context=('properties',
'schema_version'), schema must have a 'type' key`` (HTTP 400). Rung 7, never
retried: every Codex-runner generate-tasks died. No offline test could see the
API's check, so these tests carry it:

* :class:`TestProviderFacingSchemas` walks what each structured-output adapter
  is really handed (the executor's kind → phase table, through the same
  ``schema_for_dispatch`` the dispatch calls) and fails on a strict violation,
  with an independent "every node is typed" walker beside the production check;
* :class:`TestStrictRules` pins each rule of the production check to a minimal
  schema, so weakening the check is a red test rather than a 400 in a live run;
* :class:`TestGeneratedTasksMeaningIsUnchanged` holds the validator's verdicts
  on the schema's own documents where they were.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import structured_output as so  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.provider_registry import _PROVIDERS  # noqa: E402
from utils.providers.base import StructuredSchema  # noqa: E402
from utils.schema_validator import SCHEMA_DIR, validate_document  # noqa: E402


# --------------------------------------------------------------------------- #
# An independent reading of "every schema node is typed"
# --------------------------------------------------------------------------- #

def _untyped_nodes(node, path="<root>"):
    """Schema positions with neither ``type``, ``$ref`` nor ``anyOf``.

    Deliberately not the production walker: the rule that failed live is
    re-derived here from the document alone, so a regression in
    ``_strict_openai_errors`` cannot also hide the schema it should catch.
    """
    found = []
    if not isinstance(node, dict):
        return [path]
    if "$ref" not in node and "anyOf" not in node and "type" not in node:
        found.append(path)
    for name, sub in (node.get("properties") or {}).items():
        found.extend(_untyped_nodes(sub, f"{path}.{name}"))
    if isinstance(node.get("items"), dict):
        found.extend(_untyped_nodes(node["items"], f"{path}[]"))
    for index, sub in enumerate(node.get("anyOf") or ()):
        found.extend(_untyped_nodes(sub, f"{path}.anyOf[{index}]"))
    for container in ("$defs", "definitions"):
        for name, sub in (node.get(container) or {}).items():
            found.extend(_untyped_nodes(sub, f"{path}.{container}.{name}"))
    return found


def _handed_over():
    """``{(provider, phase): schema_name}`` for every native hand-over today.

    Lane B is the only lane that dispatches a schema (Lane A's ``schema_phase``
    has no caller), and it reaches ``schema_for_dispatch`` exactly through
    ``SCHEMA_PHASE_BY_KIND``. Every adapter that declares structured-output
    support is asked, so a newly supporting adapter joins this census without
    an edit here.
    """
    so.reset_declined_cache()
    handed = {}
    for phase in sorted(set(executor.SCHEMA_PHASE_BY_KIND.values())):
        for name, provider in sorted(_PROVIDERS.items()):
            if not getattr(provider, "structured_output_support", False):
                continue
            schema, _marker = so.schema_for_dispatch(provider, phase, warn=False)
            if schema is not None:
                handed[(name, phase)] = schema.name
    so.reset_declined_cache()
    return handed


def _load(name):
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


class TestProviderFacingSchemas:
    def test_codex_is_handed_generated_tasks_natively(self):
        """The fix must not turn into a decline: codex carries the file.

        A gate that simply refused the schema would also stop the 400 — and
        silently put every codex generate-tasks back on a `recovered` parse.
        """
        handed = _handed_over()
        assert handed.get(("codex", "generate-tasks")) == "generated_tasks.schema.json"

    def test_every_schema_handed_to_codex_is_strict_clean(self):
        handed = _handed_over()
        codex_schemas = {
            name for (provider, _phase), name in handed.items()
            if provider == "codex"
        }
        assert codex_schemas, "codex is handed no schema at all"
        for name in sorted(codex_schemas):
            document = _load(name)
            assert so._strict_openai_errors(document) == [], name
            assert _untyped_nodes(document) == [], name

    def test_every_provider_facing_schema_is_strict_clean_or_named_here(self):
        """Every schema any adapter is handed meets the subset — with one
        named exception that codex is never handed.

        ``consolidation_batch.schema.json`` reaches claude-code and qwen-code
        (neither enforces OpenAI's subset: one is an Anthropic tool input, the
        other a plain function tool) and is declined for codex by the gate, so
        a consolidation batch on codex runs the prompt-level path marked
        `recovered`. Making it strict means requiring its optional failure-path
        keys (`error`/`fallback`/`groups`) as nullable — which every reply would
        then carry as `null`, and which `consolidation.py` reads as "this batch
        failed" by key presence. That is a contract change for the batch file,
        not a schema edit, so it is listed here rather than made in passing.
        """
        exceptions = {"consolidation_batch.schema.json"}
        handed = _handed_over()
        for (provider, phase), name in sorted(handed.items()):
            document = _load(name)
            errors = so._strict_openai_errors(document)
            if name in exceptions:
                assert provider != "codex", (provider, phase, name)
                assert errors, f"{name} is strict-clean now; drop the exception"
                continue
            assert errors == [], (provider, phase, name, errors)
            assert _untyped_nodes(document) == [], (provider, phase, name)

    def test_the_gate_declines_an_untyped_const_instead_of_handing_it_over(self):
        """The live failure, replayed at the gate: declined, never dispatched."""
        codex = _PROVIDERS["codex"]
        broken = StructuredSchema(
            document={
                "type": "object",
                "additionalProperties": False,
                "required": ["schema_version"],
                "properties": {"schema_version": {"const": "1.0"}},
            },
            name="broken.schema.json",
        )
        reason = so.native_transport_error(codex, broken)
        assert reason and "schema_version: schema must have a 'type' key" in reason


# --------------------------------------------------------------------------- #
# The production check, rule by rule
# --------------------------------------------------------------------------- #

def _wrap(prop, **root_extra):
    document = {
        "type": "object",
        "additionalProperties": False,
        "required": ["p"],
        "properties": {"p": prop},
    }
    document.update(root_extra)
    return document


class TestStrictRules:
    def test_a_clean_schema_has_no_errors(self):
        assert so._strict_openai_errors(_wrap({"type": "string"})) == []
        assert so._strict_openai_errors(_wrap({
            "anyOf": [{"type": "string"}, {"type": "null"}],
        })) == []
        assert so._strict_openai_errors(_wrap({
            "type": ["string", "null"], "enum": ["a", None],
        })) == []

    @pytest.mark.parametrize("prop,needle", [
        ({"const": "1.0"}, "schema must have a 'type' key"),
        ({"enum": ["low", "high"]}, "schema must have a 'type' key"),
        ({"description": "anything"}, "schema must have a 'type' key"),
        ({"type": "decimal"}, "unsupported type"),
        ({"type": "array"}, "needs one 'items' schema"),
        ({"type": "array", "items": [{"type": "string"}]}, "needs one 'items' schema"),
        ({"type": "array", "items": {"enum": ["x"]}}, "<root>.p[]: schema must have"),
        ({"type": "object", "properties": {"a": {"type": "string"}},
          "required": ["a"]}, "additionalProperties must be present and false"),
        ({"type": "object", "additionalProperties": False,
          "properties": {"a": {"type": "string"}}}, "every property must be required"),
        ({"type": "string", "allOf": [{"type": "string"}]}, "'allOf' is not supported"),
        ({"type": "string", "oneOf": [{"type": "string"}]}, "'oneOf' is not supported"),
        ({"type": "string", "not": {"const": "x"}}, "'not' is not supported"),
        ({"type": "string", "if": {"const": "x"}, "then": {}},
         "'if' is not supported"),
        ({"type": "object", "additionalProperties": False, "properties": {},
          "patternProperties": {"^x": {"type": "string"}}},
         "'patternProperties' is not supported"),
        ({"type": "array", "items": {"type": "string"}, "uniqueItems": True},
         "'uniqueItems' is not supported"),
        ({"type": "string", "format": "uri"}, "format 'uri' is not supported"),
        ({"$ref": "other.json#/x"}, "only local $ref"),
        ({"$ref": "#/$defs/missing"}, "does not resolve"),
        ({"anyOf": [{"const": "a"}, {"type": "null"}]},
         "<root>.p.anyOf[0]: schema must have a 'type' key"),
    ])
    def test_each_rule_is_enforced(self, prop, needle):
        errors = so._strict_openai_errors(_wrap(prop))
        assert any(needle in error for error in errors), errors

    def test_a_ref_is_followed_into_its_definition(self):
        document = _wrap(
            {"$ref": "#/$defs/x"},
            **{"$defs": {"x": {"enum": ["a"]}}},
        )
        errors = so._strict_openai_errors(document)
        assert errors == ["<root>.p: schema must have a 'type' key"]

    def test_an_unreferenced_definition_is_checked_too(self):
        document = _wrap({"type": "string"}, **{"$defs": {"x": {"const": 1}}})
        errors = so._strict_openai_errors(document)
        assert errors == ["<root>.$defs.x: schema must have a 'type' key"]

    def test_a_recursive_ref_terminates(self):
        document = _wrap(
            {"$ref": "#/$defs/node"},
            **{"$defs": {"node": {
                "type": "object",
                "additionalProperties": False,
                "required": ["next"],
                "properties": {"next": {"anyOf": [
                    {"$ref": "#/$defs/node"}, {"type": "null"},
                ]}},
            }}},
        )
        assert so._strict_openai_errors(document) == []

    def test_the_root_may_not_be_an_anyof(self):
        errors = so._strict_openai_errors({"anyOf": [_wrap({"type": "string"})]})
        assert "<root>: the root may not be an anyOf" in errors

    def test_length_and_pattern_keywords_are_not_refused(self):
        """Accepted live in strict mode on 2026-10-04 once `type` was added
        (`minLength` on three fields, `pattern` on the task id). OpenAI's own
        pages disagree about them, so declining them would take the native
        path away on the strength of a doc the API contradicts."""
        assert so._strict_openai_errors(_wrap({
            "type": "string", "minLength": 1, "pattern": "^T[0-9]{3}$",
        })) == []
        assert so._strict_openai_errors(_wrap({
            "type": "array", "items": {"type": "string"}, "minItems": 1,
        })) == []


# --------------------------------------------------------------------------- #
# The validator's meaning did not move
# --------------------------------------------------------------------------- #

_TASK = {
    "id": "T001",
    "title": "Set up the module",
    "description": "Create the module skeleton.",
    "depends_on": [],
    "files_to_create": ["src/a.py"],
    "files_to_modify": [],
    "acceptance_criteria": ["`python -c 'import a'` exits 0"],
    "estimated_complexity": "low",
    "executor_role": "agent",
    "subagent_type": None,
}
_DOC = {"schema_version": "1.0", "plan_preamble": "Context.", "tasks": [_TASK]}


class TestGeneratedTasksMeaningIsUnchanged:
    """`type` beside `const`/`enum` adds a check every accepted value already
    passed: the same documents validate, the same ones fail."""

    @pytest.fixture
    def schema(self):
        return _load("generated_tasks.schema.json")

    def test_the_canonical_document_still_validates(self, schema):
        assert validate_document(_DOC, schema) == []

    @pytest.mark.parametrize("mutate", [
        lambda d: d.update(schema_version="2.0"),
        lambda d: d.update(schema_version=1.0),
        lambda d: d["tasks"][0].update(estimated_complexity="extreme"),
        lambda d: d["tasks"][0].update(estimated_complexity=1),
        lambda d: d["tasks"][0].update(executor_role="robot"),
        lambda d: d["tasks"][0].update(executor_role=None),
    ])
    def test_the_values_it_refused_are_still_refused(self, schema, mutate):
        document = copy.deepcopy(_DOC)
        mutate(document)
        assert validate_document(document, schema) != []

    @pytest.mark.parametrize("value", ["low", "medium", "high"])
    def test_every_complexity_still_validates(self, schema, value):
        document = copy.deepcopy(_DOC)
        document["tasks"][0]["estimated_complexity"] = value
        assert validate_document(document, schema) == []
