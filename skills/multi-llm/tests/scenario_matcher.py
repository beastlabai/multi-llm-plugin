#!/usr/bin/env python3
"""One definition of what a scenario *means*, shared by every invoker (C4).

``tests/fixtures/e2e/scenarios/*.yaml`` maps a prompt to the canned answer a
model would have given for it. Two very different processes have to agree on
that mapping, byte for byte:

* ``tests/mocks/mock_llm.py`` — the fake provider CLI. It learns its identity
  from ``argv[0]`` and its scenario from ``MOCK_LLM_SCENARIO``.
* ``tests/container/stub_server.py`` — the HTTP stub the *real* CLIs are
  redirected at inside the container (a later phase of
  ``plans/containerized-harness-e2e.md``, a local-only working note that is not
  part of this repo). It has no ``argv[0]`` and no
  environment of its own; it has a request.

If those two grew their own copies of the matching logic they would drift, and
the container lane would then be testing a second source of truth for what a
scenario means. So the logic lives here — and **nothing in this module's API
reads ``sys.argv`` or ``os.environ``**. Every input is a parameter, because the
stub has neither to read.

Selection order, in the plan's words: *explicit pattern match → per-session turn
index → the scenario's declared fallback*.

1. **pattern** — the first ``prompts:`` entry whose ``pattern`` matches
   (``re.search``, case-insensitive) and that actually names a response. First
   match wins; a matching entry whose fixture file is missing does **not**
   advance to the next candidate (it is a scenario-authoring error, not a near
   miss), which is the behaviour ``mock_llm.get_response_content`` has always
   had.
2. **turn** — the scenario-level ``turns:`` list, indexed by the caller's
   per-session turn index. HTTP carries no session identity, so the *caller*
   owns the counter (the stub keys it on the per-exec id baked into the base
   URL); this module only reads the number it is handed.
3. **fallback** — the scenario-level ``fallback:`` entry.

...and if none of the three answers:

* by default, :func:`resolve` returns ``None`` — "this scenario has nothing to
  say about this prompt" — and the caller decides what that means. For
  ``mock_llm.py`` that is today's fall-through to ``MOCK_LLM_CONFIG`` and then
  to its generic provider default;
* under the scenario-level ``on_unmatched: fail`` key, it raises
  :class:`UnmatchedPrompt` instead. That key exists because a generic default is
  *shaped* enough to look like a passing review phase while proving nothing: on
  the container lane it would mean a real CLI never exercising the fixture its
  row claims to cover. Both invokers of this module inherit the strictness
  together — the fake binary exits non-zero, the stub answers a machine-readable
  error — rather than one of them staying lenient.

A raised error carries the prompt's **sha256 and byte length, never its body**:
a prompt is tens of KB of plan text, and the same rule already governs the
wrapper call log and the write-path audit log.

Scenario keys this module does **not** own are left alone in
:attr:`Scenario.raw`: ``default_delay_ms``, ``timeout``, ``rate_limit``,
``malformed_responses``, ``provider_overrides``, ``logging`` and friends are
rendering/injection concerns belonging to whichever invoker implements them.
In particular ``provider_overrides.*.wire_format`` is *not* consulted here — the
fake binary picks its wire format from ``argv[0]`` and the stub from the request
surface, and neither may take it from the scenario without the two disagreeing.

Dependencies: stdlib + pyyaml, nothing else. This module is imported inside the
container image (it is named in ``tests/container/context_manifest.txt``), so it
must not reach for anything the product's own environment does not carry.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ON_UNMATCHED_DEFAULT",
    "ON_UNMATCHED_FAIL",
    "ON_UNMATCHED_VALUES",
    "MissingFixture",
    "PromptRule",
    "Resolution",
    "ResponseSpec",
    "Scenario",
    "ScenarioConfigError",
    "ScenarioError",
    "ScenarioMatcher",
    "UnmatchedPrompt",
    "load_scenario",
    "matching_rules",
    "prompt_digest",
    "resolve",
    "resolve_response",
]


#: Value of the scenario-level ``on_unmatched`` key that keeps the historical,
#: lenient behaviour: an unanswered prompt resolves to ``None`` and the caller
#: falls through to whatever default it had. This is also the behaviour when the
#: key is absent, which is what makes the key purely additive.
ON_UNMATCHED_DEFAULT = "default"

#: Value that turns an unanswered prompt into a raised :class:`UnmatchedPrompt`.
ON_UNMATCHED_FAIL = "fail"

ON_UNMATCHED_VALUES = (ON_UNMATCHED_DEFAULT, ON_UNMATCHED_FAIL)

#: Where a :class:`Resolution` came from, in selection order.
SOURCE_PATTERN = "pattern"
SOURCE_TURN = "turn"
SOURCE_FALLBACK = "fallback"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ScenarioError(Exception):
    """Base for every refusal this module raises."""


class ScenarioConfigError(ScenarioError):
    """The scenario file itself is wrong — a bad key, an uncompilable pattern.

    Raised at load time so an authoring mistake surfaces where it was made
    rather than as a mysteriously unanswered prompt three phases later. Only
    reachable for keys this module owns, so a scenario that predates them
    cannot trip it.
    """


class UnmatchedPrompt(ScenarioError):
    """No rule, turn or fallback answered, and the scenario said that must fail.

    Carries the prompt's digest and length rather than its bytes: callers log
    this, and a prompt is plan text.
    """

    def __init__(
        self,
        *,
        scenario_name: str,
        scenario_path: Path,
        prompt_sha256: str,
        prompt_bytes: int,
        provider: str | None = None,
        turn_index: int | None = None,
    ) -> None:
        self.scenario_name = scenario_name
        self.scenario_path = scenario_path
        self.prompt_sha256 = prompt_sha256
        self.prompt_bytes = prompt_bytes
        self.provider = provider
        self.turn_index = turn_index
        super().__init__(str(self))

    def __str__(self) -> str:
        provider = self.provider or "-"
        turn = "-" if self.turn_index is None else self.turn_index
        return (
            f"scenario {self.scenario_name!r} ({self.scenario_path}) declares "
            f"on_unmatched: {ON_UNMATCHED_FAIL} and nothing answered prompt "
            f"sha256={self.prompt_sha256} ({self.prompt_bytes} bytes, "
            f"provider={provider}, turn={turn})"
        )


class MissingFixture(ScenarioError):
    """A rule matched but the fixture it names is not on disk.

    Only raised under ``on_unmatched: fail``. Without that key a missing fixture
    falls through exactly as it always has — several shipped scenarios name
    fixtures that do not resolve, and fixing them is not this refactor's job.
    Under the strict key, falling through would be precisely the leniency the
    key exists to remove: the row would answer with a generic default while
    claiming to cover a named fixture.
    """

    def __init__(
        self,
        *,
        scenario_name: str,
        scenario_path: Path,
        fixture: str,
        fixture_path: Path,
        source: str,
        prompt_sha256: str,
        prompt_bytes: int,
    ) -> None:
        self.scenario_name = scenario_name
        self.scenario_path = scenario_path
        self.fixture = fixture
        self.fixture_path = fixture_path
        self.source = source
        self.prompt_sha256 = prompt_sha256
        self.prompt_bytes = prompt_bytes
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"scenario {self.scenario_name!r} ({self.scenario_path}) matched by "
            f"{self.source}, but its fixture {self.fixture!r} does not exist at "
            f"{self.fixture_path} (prompt sha256={self.prompt_sha256}, "
            f"{self.prompt_bytes} bytes)"
        )


# --------------------------------------------------------------------------- #
# Prompt identity
# --------------------------------------------------------------------------- #


def prompt_digest(prompt: str) -> str:
    """The prompt's sha256 hex digest — the only form of it that gets logged."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_bytes(prompt: str) -> int:
    return len(prompt.encode("utf-8"))


# --------------------------------------------------------------------------- #
# The pieces of a scenario
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResponseSpec:
    """Where one answer's bytes come from: a fixture file, or inline content.

    ``fixture`` is stored exactly as the YAML wrote it — a path relative to the
    **scenario file**, which is the only anchor both invokers share (the stub's
    cwd is the container's, the fake binary's is the orchestrator's workspace).
    """

    fixture: str | None = None
    content: str | None = None
    description: str | None = None

    def path(self, scenario_dir: Path) -> Path | None:
        """The fixture's location, or ``None`` for an inline spec."""
        if self.fixture is None:
            return None
        return scenario_dir / self.fixture


@dataclass(frozen=True)
class PromptRule:
    """One ``prompts:`` entry that can actually answer something.

    Entries carrying only an *injection* directive (``action: timeout``,
    ``malformed: truncated_object``) name no response and are dropped at load —
    the historical matcher skipped them inside its loop for the same reason, and
    dropping them at parse time keeps "first match wins" meaning the same thing.
    """

    pattern: str
    response: ResponseSpec
    provider_filter: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def matches(self, prompt: str, provider: str | None = None) -> bool:
        """Whether this rule answers ``prompt`` for ``provider``.

        A rule scoped with ``provider_filter`` is not considered by a caller
        that does not know which provider it is answering for
        (``provider=None``): a provider-independent query may not be answered
        with a provider-specific fixture.
        """
        if self.provider_filter is not None and self.provider_filter != provider:
            return False
        try:
            return bool(re.search(self.pattern, prompt, re.IGNORECASE))
        except re.error:  # pragma: no cover - load() rejects these for strict scenarios
            return False


@dataclass(frozen=True)
class Resolution:
    """What answered, and with what — everything a log line or an assertion needs."""

    content: str
    source: str
    description: str | None = None
    fixture: str | None = None
    fixture_path: Path | None = None
    pattern: str | None = None
    turn_index: int | None = None


@dataclass(frozen=True)
class Scenario:
    """A parsed scenario file. Construct it with :func:`load_scenario`."""

    path: Path
    name: str
    on_unmatched: str
    rules: tuple[PromptRule, ...]
    turns: tuple[ResponseSpec, ...]
    fallback: ResponseSpec | None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        """The directory every fixture path in this scenario is relative to."""
        return self.path.parent

    @property
    def strict(self) -> bool:
        """Whether an unanswered prompt raises instead of returning ``None``."""
        return self.on_unmatched == ON_UNMATCHED_FAIL

    def matching_rules(self, prompt: str, provider: str | None = None) -> list[PromptRule]:
        """Every rule that matches, in file order — not just the winning one.

        Only the first is ever used to answer; the rest are what a non-overlap
        assertion needs (a scenario whose patterns cross-match hands one phase's
        answer to another phase's validator).
        """
        return [rule for rule in self.rules if rule.matches(prompt, provider)]

    def resolve(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        turn_index: int | None = None,
    ) -> Resolution | None:
        """Resolve ``prompt`` to its canned answer. See :func:`resolve`."""
        return resolve(self, prompt, provider=provider, turn_index=turn_index)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _as_response_spec(value: Any) -> ResponseSpec | None:
    """Build a :class:`ResponseSpec` from a YAML node, or ``None`` if it names none.

    A bare string is shorthand for ``{fixture: <string>}``, which is how the
    scenario-level ``turns:``/``fallback:`` lists read best when every entry is
    just a file.
    """
    if isinstance(value, str):
        return ResponseSpec(fixture=value)
    if not isinstance(value, Mapping):
        return None
    # An empty `fixture` names no file — the historical matcher required a
    # truthy one. An empty `content` is a real answer (a CLI that returns
    # nothing is a case worth encoding), so it is tested against None instead.
    fixture = value.get("fixture") or None
    content = value.get("content")
    if fixture is None and content is None:
        return None
    description = value.get("description")
    return ResponseSpec(
        fixture=str(fixture) if fixture is not None else None,
        content=str(content) if content is not None else None,
        description=str(description) if description is not None else None,
    )


def _parse_on_unmatched(raw: Mapping[str, Any], scenario_path: Path) -> str:
    declared = raw.get("on_unmatched", ON_UNMATCHED_DEFAULT)
    if declared is None:
        return ON_UNMATCHED_DEFAULT
    value = str(declared).strip().lower()
    if value not in ON_UNMATCHED_VALUES:
        raise ScenarioConfigError(
            f"{scenario_path}: on_unmatched must be one of "
            f"{', '.join(ON_UNMATCHED_VALUES)} (got {declared!r})"
        )
    return value


def _parse_rules(
    raw: Mapping[str, Any], scenario_path: Path, strict: bool
) -> tuple[PromptRule, ...]:
    entries = raw.get("prompts") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ScenarioConfigError(f"{scenario_path}: `prompts` must be a list")

    rules: list[PromptRule] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        pattern = entry.get("pattern")
        if not pattern:
            continue
        response = _as_response_spec(entry)
        if response is None:
            # An injection-only entry (`action:`/`malformed:`): it names no
            # response, so it can never answer. Historically skipped mid-loop.
            continue
        try:
            re.compile(str(pattern))
        except re.error as exc:
            if strict:
                raise ScenarioConfigError(
                    f"{scenario_path}: pattern {pattern!r} does not compile ({exc}). "
                    f"A scenario declaring on_unmatched: {ON_UNMATCHED_FAIL} cannot "
                    f"carry a rule that silently never matches."
                ) from exc
            # Lenient scenarios have always skipped an uncompilable pattern.
            continue
        provider_filter = entry.get("provider_filter")
        rules.append(
            PromptRule(
                pattern=str(pattern),
                response=response,
                provider_filter=str(provider_filter) if provider_filter else None,
                raw=entry,
            )
        )
    return tuple(rules)


def _parse_turns(raw: Mapping[str, Any], scenario_path: Path) -> tuple[ResponseSpec, ...]:
    entries = raw.get("turns") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ScenarioConfigError(f"{scenario_path}: `turns` must be a list")
    specs: list[ResponseSpec] = []
    for position, entry in enumerate(entries):
        spec = _as_response_spec(entry)
        if spec is None:
            raise ScenarioConfigError(
                f"{scenario_path}: turns[{position}] names neither `fixture` nor "
                f"`content`, so it can never answer a turn"
            )
        specs.append(spec)
    return tuple(specs)


def load_scenario(scenario_path: str | Path) -> Scenario | None:
    """Parse a scenario file.

    Returns ``None`` — never raises — when the scenario is simply *unavailable*:
    pyyaml is not importable, the path does not exist or does not read, the YAML
    does not parse, or it is not a mapping. That tolerance is
    ``mock_llm.load_scenario``'s and is deliberate: an unreadable scenario has
    always degraded to "no scenario", and a fake CLI that died on it would turn
    an unrelated environment problem into a provider failure.

    :class:`ScenarioConfigError` *is* raised when the file parses but the keys
    this module owns are wrong. That is a different thing — the scenario is
    present and says something invalid — and only reachable through keys no
    pre-existing scenario carries.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a hard dependency of the suite
        return None

    path = Path(scenario_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(raw, Mapping):
        return None

    on_unmatched = _parse_on_unmatched(raw, path)
    strict = on_unmatched == ON_UNMATCHED_FAIL
    fallback = _as_response_spec(raw.get("fallback"))
    return Scenario(
        path=path,
        name=str(raw.get("name") or path.stem),
        on_unmatched=on_unmatched,
        rules=_parse_rules(raw, path, strict),
        turns=_parse_turns(raw, path),
        fallback=fallback,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _read(spec: ResponseSpec, scenario: Scenario) -> str | None:
    """The spec's bytes, or ``None`` when its fixture is not on disk."""
    if spec.content is not None:
        return spec.content
    fixture_path = spec.path(scenario.dir)
    if fixture_path is None or not fixture_path.exists():
        return None
    with open(fixture_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _candidates(
    scenario: Scenario, prompt: str, provider: str | None, turn_index: int | None
) -> Iterator[tuple[str, ResponseSpec, PromptRule | None]]:
    """The selection order, as (source, spec, rule) — at most one is consumed."""
    for rule in scenario.rules:
        if rule.matches(prompt, provider):
            # First match wins, and a first match that cannot be read is a
            # scenario-authoring error rather than a near miss: it does not
            # advance to the next candidate.
            yield SOURCE_PATTERN, rule.response, rule
            return

    if turn_index is not None and 0 <= turn_index < len(scenario.turns):
        yield SOURCE_TURN, scenario.turns[turn_index], None
        return

    if scenario.fallback is not None:
        yield SOURCE_FALLBACK, scenario.fallback, None


def resolve(
    scenario: Scenario,
    prompt: str,
    *,
    provider: str | None = None,
    turn_index: int | None = None,
) -> Resolution | None:
    """Resolve ``prompt`` against ``scenario``: pattern → turn → fallback.

    Args:
        scenario: A :func:`load_scenario` result.
        prompt: The prompt text, exactly as the model would have received it.
        provider: The provider identity answering, when the caller knows it
            (``cursor-agent``, ``codex``, …). Used only to honour a rule's
            ``provider_filter`` and to annotate a refusal — never to pick a wire
            format, which is the invoker's own business.
        turn_index: The **caller-owned**, zero-based turn counter for this
            session. HTTP has no session identity, so the stub derives it from
            the per-exec id in the base URL and the fake binary from
            ``MOCK_LLM_TURN_INDEX``; this module only reads the number.

    Returns:
        The :class:`Resolution` that answered, or ``None`` when the scenario has
        nothing to say and does not declare ``on_unmatched: fail``.

    Raises:
        UnmatchedPrompt: nothing answered and ``on_unmatched: fail``.
        MissingFixture: something matched, its fixture is absent, and
            ``on_unmatched: fail``.
    """
    for source, spec, rule in _candidates(scenario, prompt, provider, turn_index):
        content = _read(spec, scenario)
        if content is None:
            if scenario.strict:
                fixture_path = spec.path(scenario.dir)
                raise MissingFixture(
                    scenario_name=scenario.name,
                    scenario_path=scenario.path,
                    fixture=spec.fixture or "",
                    fixture_path=fixture_path if fixture_path is not None else scenario.dir,
                    source=source,
                    prompt_sha256=prompt_digest(prompt),
                    prompt_bytes=_prompt_bytes(prompt),
                )
            return None
        return Resolution(
            content=content,
            source=source,
            description=spec.description,
            fixture=spec.fixture,
            fixture_path=spec.path(scenario.dir),
            pattern=rule.pattern if rule is not None else None,
            turn_index=turn_index if source == SOURCE_TURN else None,
        )

    if scenario.strict:
        raise UnmatchedPrompt(
            scenario_name=scenario.name,
            scenario_path=scenario.path,
            prompt_sha256=prompt_digest(prompt),
            prompt_bytes=_prompt_bytes(prompt),
            provider=provider,
            turn_index=turn_index,
        )
    return None


def resolve_response(
    scenario_path: str | Path,
    prompt: str,
    *,
    provider: str | None = None,
    turn_index: int | None = None,
) -> Resolution | None:
    """:func:`resolve`, for a caller holding a path rather than a :class:`Scenario`.

    The one-call entry point: ``(scenario_path, prompt)`` is all it needs. A
    caller answering many prompts against one scenario should
    :func:`load_scenario` once and call :meth:`Scenario.resolve` instead — this
    re-reads and re-parses the YAML on every call.
    """
    scenario = load_scenario(scenario_path)
    if scenario is None:
        return None
    return resolve(scenario, prompt, provider=provider, turn_index=turn_index)


def matching_rules(
    scenario: Scenario, prompt: str, provider: str | None = None
) -> list[PromptRule]:
    """Every rule matching ``prompt``, in file order. See :meth:`Scenario.matching_rules`."""
    return scenario.matching_rules(prompt, provider)


class ScenarioMatcher:
    """A scenario held open by a long-lived caller, e.g. the HTTP stub.

    The functional API above is shaped for a one-shot process: a fake CLI reads
    one prompt, answers it and exits, so re-reading the YAML costs nothing. A
    server is the other case — it loads once at startup and then answers
    thousands of requests across concurrent sessions — and it wants two things
    the loose functions deliberately do not give it:

    * **loading is an error, not a shrug.** :func:`load_scenario` returns
      ``None`` for an unavailable scenario because a fake CLI degrading to "no
      scenario" is better than a fake CLI dying on an unrelated environment
      problem. A server handed ``--scenario`` and unable to read it must not
      come up serving defaults, so :meth:`load` raises instead.
    * **``turn`` rather than ``turn_index``.** The stub's own vocabulary for the
      per-session counter it keeps.

    Everything else delegates: this class holds a :class:`Scenario` and calls
    :func:`resolve`. It is a calling convention, never a second implementation.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    @classmethod
    def load(cls, scenario_path: str | Path) -> ScenarioMatcher:
        """Load ``scenario_path``, raising when it cannot be read.

        Raises:
            ScenarioConfigError: the scenario is unavailable (missing, not
                readable, not YAML, not a mapping) or invalid.
        """
        scenario = load_scenario(scenario_path)
        if scenario is None:
            raise ScenarioConfigError(
                f"{scenario_path}: scenario could not be loaded (missing, "
                f"unreadable, or not a YAML mapping)"
            )
        return cls(scenario)

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def strict(self) -> bool:
        return self.scenario.strict

    def resolve(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        turn: int | None = None,
    ) -> Resolution | None:
        """:func:`resolve`, with the stub's spelling of the turn counter."""
        return resolve(self.scenario, prompt, provider=provider, turn_index=turn)

    def matching_rules(self, prompt: str, provider: str | None = None) -> list[PromptRule]:
        """See :meth:`Scenario.matching_rules`."""
        return self.scenario.matching_rules(prompt, provider)
