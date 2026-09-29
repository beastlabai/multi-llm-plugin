"""M15 / §3.5 — the read-only mode table as data, and the `--yolo`-class negative set.

Three guards, and they answer three different questions:

* **The table covers the registry.** ``providers.yaml``'s ``read_only_modes:``
  block carries one row per registered adapter, checked against
  ``provider_registry._PROVIDERS`` rather than against a roster written here.
  Nothing in this module names an adapter or a count — the specification it
  implements was written for thirteen adapters and the registry holds
  seventeen, and a literal list is precisely how four of them fell out of it
  (``harness-residual-work.md`` M0.a).

* **An unprobed row is `null`, never a guess.** ``mechanism:`` and ``probed:``
  move together in both directions, a probed row's tokens really do appear in
  the argv that adapter renders, and a ``null`` row has not quietly acquired
  some other CLI's read-only flag. The rule matters more than the arithmetic:
  three rows were measured, every other row is the probe queue, and an
  adapter's flags, help text or name are not a measurement.

* **The negative set.** No adapter emits a `--yolo`-class write bypass from
  ``build_command`` (the ``generate`` profile — the Lane A review/ask fan-out)
  or ``get_env``, and none from ``agent-readonly`` either. §3.5 is emphatic
  that this guard **reads this repo's own source, never a vendor's help text**:
  it is verifiable offline, cannot be invalidated by a vendor release, and says
  nothing about what any CLI's surface does or does not still offer. Orca's
  claims about opencode and kilo having removed the Claude-style skip flag are
  unvalidated donor material and are deliberately not encoded anywhere here.

The one escape hatch is §3.5's own: an adapter whose only unattended mode is
such a flag gets an explicit ``null`` row **plus a named
``bypass_exception:``** naming the token, the profile, why it survives and what
retires it. It is not a way to wave a flag through — the guard is equally red on
an exception whose token is no longer emitted, so an exception cannot outlive
the flag it excuses, and the declaration lives in the base ``providers.yaml``
layer only, where no user override can reach it.
"""
import ast
import re
from pathlib import Path
from unittest import mock

import pytest

import utils.providers.base as base_module
from utils.provider_registry import _PROVIDERS, load_base_config
from utils.providers.base import (
    PROFILE_AGENT_READONLY,
    PROFILE_AGENT_WRITE,
    PROFILE_GENERATE,
)

PROMPT = "Analyze this plan carefully"

#: The block's closed key set. An unknown key is a typo that would otherwise
#: sit in the file doing nothing, which is the failure mode a table of records
#: is most exposed to.
ROW_KEYS = frozenset({"mechanism", "probed", "bypass_exception"})

#: A named exception's closed key set. Each field is load-bearing: without
#: `retires_when` the entry is "it was already there" with extra words.
EXCEPTION_KEYS = frozenset({"token", "profile", "reason", "retires_when"})


# ---------------------------------------------------------------------------
# The `--yolo`-class negative set (§3.5), as code rather than config
# ---------------------------------------------------------------------------
#
# Deliberately NOT read from `providers.yaml`: the vocabulary of the guard is
# policy and must not be reachable by any config layer, or a project-local
# override could narrow the set until the scan passed. Only the *exceptions*
# are data, and those are read from the base layer alone.

#: Whole argv tokens. `-f` is cursor's retired write bypass (§3.1), kept in the
#: set so it cannot come back under any adapter.
NEGATIVE_FLAG_TOKENS = frozenset(
    {"--yolo", "-f", "--force", "--yes-always", "--auto-approve"}
)

#: Prefix family — `--dangerously-skip-permissions`,
#: `--dangerously-bypass-approvals-and-sandbox`, and anything upstream names next.
NEGATIVE_TOKEN_PREFIXES = ("--dangerously",)

#: Bypasses that ride as a *value* rather than as a flag
#: (`--permission-mode bypassPermissions`, `--sandbox danger-full-access`,
#: `--approval-mode yolo`), so they are matched wherever they sit.
#: `danger-full-access` joined the set with the D6 override (2026-09-17): it is
#: codex's own third sandbox policy, and before it was here the guard was blind
#: to it — a value, not a `--dangerously` prefix. The default argv of every
#: profile must still never carry it; the one place it may appear is the
#: uncontained rung of the dispatch-time ladder `DISPATCH_TIME_OVERRIDE`
#: declares below, and that class checks every rung from both sides.
#:
#: `yolo` joined on the same reading (2026-09-17, the omp adapter): it is the
#: value half of `--approval-mode yolo`, and without it in this set the pair
#: would have passed the scan unseen — `--approval-mode` is in neither
#: `NEGATIVE_FLAG_TOKENS` nor `NEGATIVE_TOKEN_PREFIXES`, and a guard that cannot
#: see the token is not a guard. Unlike `danger-full-access` it is rendered in a
#: DEFAULT argv, which is exactly why it owes a declared
#: `read_only_modes.omp.bypass_exception` naming the value rather than the flag:
#: both halves of the machinery key on this one string.
NEGATIVE_VALUE_TOKENS = frozenset(
    {"bypassPermissions", "danger-full-access", "yolo"}
)

#: The ONE dispatch-time override of the negative set, declared as data with the
#: same four fields a `bypass_exception:` row carries — token, profile(s), why it
#: survives, what retires it — because §3.5's rule is "never a silent
#: precedent", and this is not a `read_only_modes:` row: codex's row is a
#: *probed* mechanism (`--sandbox read-only`), and the table's own invariant
#: (`test_an_exception_row_is_a_null_row`) rightly refuses a row that claims
#: both a mechanism and an exception. What differs from a `bypass_exception:`
#: is where the token lives: never in a rendered default, only behind a kwarg
#: (`sandbox_fallback=<rung id>`) that a gate sets after all four of its
#: conditions hold.
#:
#: Since the landlock rung (2026-09-22) the override is a LADDER, in walk order,
#: and each rung carries its own disposition. Rung 1 (`landlock`) is a real
#: kernel boundary and carries NO negative token — it renders `--sandbox
#: read-only` plus the per-dispatch feature pair, and the negative-set scan must
#: stay silent on it; its hazard is the other one, `workspace-write` beside the
#: feature (the exit-101 panic), so the rung pins `read-only` as its value.
#: Rung 2 (`unsandboxed`) is the D6 bypass and its token is the negative-set
#: `danger-full-access`. `TestTheOneDispatchTimeOverride` reads every rung from
#: both sides — the default argv stays clean of it, the override renders exactly
#: this rewrite on exactly these profiles, the write profile raises — and pins
#: this declaration against the adapter's own `SANDBOX_FALLBACK_RUNGS`.
DISPATCH_TIME_OVERRIDE = {
    "provider": "codex",
    "profiles": (PROFILE_GENERATE, PROFILE_AGENT_READONLY),
    "rungs": (
        {
            "rung": "landlock",
            "rewrite": (
                "--sandbox", "read-only", "-c", "features.use_legacy_landlock=true",
            ),
            "token": None,
            "contained": True,
            "reason": (
                "landlock rung (2026-09-22): codex 0.155.1 still ships a second "
                "Linux sandbox backend that needs no user namespace, so on a "
                "machine where bubblewrap cannot start a read-only phase runs "
                "under a real kernel boundary — every write denied, no network — "
                "instead of none. Taken only when its pinned probe exits 0 with "
                "its canary write DENIED, and only under `auto` or `contained`."
            ),
            "retires_when": (
                "upstream removes `[features] use_legacy_landlock` (deprecated at "
                "0.155.1; `codex features list` no longer naming it is the "
                "signal) — the probe then fails, the rung is deleted, and the "
                "ladder collapses to rung 2 alone."
            ),
        },
        {
            "rung": "unsandboxed",
            "rewrite": ("--sandbox", "danger-full-access"),
            "token": "danger-full-access",
            "contained": False,
            "reason": (
                "D6 (2026-09-17): on a machine where bubblewrap cannot start, "
                "codex — the one reviewer that insists on its own OS sandbox — "
                "runs its read-only phases without it under "
                "`providers.codex.sandbox_fallback: auto` (the default; "
                "`contained` and `off` never reach it), the footing every other "
                "reviewer already has there; the D6 probe verdict must be "
                "`unavailable` (never `unknown`), rung 1's probe must have "
                "failed, and the profile read-only. A by-hand run under this "
                "value found 2 unique validated-high issues the sandboxed "
                "reviewers did not."
            ),
            "retires_when": (
                "the AppArmor repair `multi_llm.py doctor` prints is the norm on "
                "the hosts this runs on, or codex ships a sandbox that does not "
                "need a user namespace — then the key and the rewrite go together."
            ),
        },
    ),
}

#: The rung ids, in walk order — the parametrize axis below.
OVERRIDE_RUNGS = tuple(row["rung"] for row in DISPATCH_TIME_OVERRIDE["rungs"])


def _override_rung(rung):
    return next(row for row in DISPATCH_TIME_OVERRIDE["rungs"] if row["rung"] == rung)


#: The env case: a variable whose value is the bypass.
NEGATIVE_ENV = {"GOOSE_MODE": "auto"}


def _canonical(token):
    """`--flag=value` is the same bypass as `--flag value`."""
    head, sep, tail = token.partition("=")
    return head, (tail if sep else None)


def negative_tokens_in_argv(argv):
    """Every `--yolo`-class token in one rendered argv, in order."""
    found = []
    for element in argv:
        head, tail = _canonical(element)
        if head in NEGATIVE_FLAG_TOKENS:
            found.append(head)
        elif any(head.startswith(prefix) for prefix in NEGATIVE_TOKEN_PREFIXES):
            found.append(head)
        for candidate in (element, tail):
            if candidate is not None and candidate in NEGATIVE_VALUE_TOKENS:
                found.append(candidate)
    return found


def negative_tokens_in_env(env):
    """Every `--yolo`-class bypass in one adapter's env delta."""
    found = []
    for name, value in (env or {}).items():
        text = str(value)
        if NEGATIVE_ENV.get(name) == text:
            found.append(f"{name}={text}")
        elif text in NEGATIVE_VALUE_TOKENS or text in NEGATIVE_FLAG_TOKENS:
            found.append(f"{name}={text}")
    return found


# ---------------------------------------------------------------------------
# Rendering every adapter, from the registry
# ---------------------------------------------------------------------------


def _registered_provider_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one.

    A local twin of the same helper in `tests/test_providers.py`, for the reason
    recorded there: `@pytest.mark.parametrize` evaluates its argument while the
    class body runs, so the roster has to be importable at that moment.
    """
    return sorted(_PROVIDERS)


def _model_for(name):
    """A real model slug for this adapter, taken from the config it ships with.

    Not a hand-written table and not a shared placeholder: several adapters split
    on `/` or on a trailing `:effort`, so a dummy would render an argv shape the
    real one never has. Reading `providers.<name>.models` keeps this derived from
    the same file the roster comes from.
    """
    block = (load_base_config().get("providers") or {}).get(name) or {}
    models = block.get("models") or []
    return models[0] if models else "test-model"


@pytest.fixture(autouse=True)
def _adapter_reads_no_installed_binary():
    """Render argv from the adapter, never from the developer's PATH.

    cursor's §5.4.1 capability gate reads the *installed* cursor-agent's `--help`
    before `build_command` will emit `--trust --mode ask`, and raises
    `CursorFlagsUnconfirmed` when it cannot confirm both tokens. That makes this
    module's result depend on which CLIs the machine happens to have — the one
    thing a source-scanning guard must not do. Neutralizing the gate's binary
    lookup is enough: with no binary to read, `build_command` skips the probe and
    returns the argv it would render. Every other adapter is untouched by this.

    Since M14 the lookup is the *shared* gate's (`LLMProvider.gate_binary`), so
    that is what is neutralized — one implementation, one patch point.
    """
    with mock.patch.object(base_module.LLMProvider, "gate_binary", lambda self, **_kw: None):
        yield


def _generate_argv(name):
    return _PROVIDERS[name].build_command(PROMPT, _model_for(name))


def _profile_argv(name, profile):
    """The argv for `profile`, or None when this adapter does not declare it."""
    provider = _PROVIDERS[name]
    if profile not in provider.agent_profiles:
        return None
    return provider.build_agent_command(PROMPT, _model_for(name), profile)


def _env_for(name):
    return _PROVIDERS[name].get_env(_model_for(name))


def read_only_modes():
    """The `read_only_modes:` block from the BASE layer only.

    Base, never `load_config()`: a per-project `.multi-llm/providers.yaml` or a
    `MULTI_LLM_PROVIDERS_CONFIG` deep-merges over base like any other layer, so
    reading the merged view would let a user's file add a `bypass_exception:` and
    silence this guard. What ships is what is checked.
    """
    return load_base_config().get("read_only_modes") or {}


def _rows_with_exceptions():
    return {
        name: row["bypass_exception"]
        for name, row in read_only_modes().items()
        if isinstance(row, dict) and row.get("bypass_exception") is not None
    }


def _exception_token_for(name, profile):
    """The one token an exception excuses **on this surface**, or nothing.

    `bypass_exception.profile` is the profile the disposition was written for,
    and binding the allowed set to it is what keeps that schema-required field
    load-bearing rather than decorative. Keyed on the adapter alone, an entry
    recorded for `agent-write` — the one profile §3.5 permits a bypass on —
    would silence the `generate` and `get_env` sweeps, which scan a different
    lane entirely, and the widest disposition in the file would become the
    quietest. Both shipped exceptions declare `profile: generate`, so nothing is
    currently mis-exempted; this makes that a fact the guard checks rather than
    one it happens to be handed.
    """
    declared = _rows_with_exceptions().get(name)
    if declared is None or declared.get("profile") != profile:
        return set()
    return {declared["token"]}


class TestTheTableIsOneRowPerRegisteredAdapter:
    """The row set comes from the registry, not from a list written by hand."""

    def test_the_row_set_is_exactly_the_registry(self):
        table = set(read_only_modes())
        registry = set(_registered_provider_names())

        assert table == registry, (
            f"read_only_modes: is missing {sorted(registry - table)} and carries "
            f"{sorted(table - registry)} for adapters the registry does not have. "
            f"Every registered adapter needs a row on the day it is registered — "
            f"an unprobed one is `mechanism: null`, never a guess."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_the_row_declares_only_known_keys(self, name):
        row = read_only_modes()[name]

        assert isinstance(row, dict), f"{name}: row must be a mapping, got {row!r}"
        unknown = set(row) - ROW_KEYS
        assert not unknown, (
            f"{name}: unknown key(s) {sorted(unknown)} in its read_only_modes row. "
            f"Known keys: {sorted(ROW_KEYS)}."
        )
        assert "mechanism" in row and "probed" in row, (
            f"{name}: both `mechanism:` and `probed:` must be present — an absent "
            f"key and an explicit `null` read the same to a human and differently "
            f"to a parser."
        )


#: The one measured mechanism flag whose meaning is not fixed across CLIs, and
#: therefore the only one the borrow guard below reads as a whole
#: `(flag, value)` atom rather than as a bare flag.
#:
#: `--mode` is overloaded: cursor-agent's `--mode ask` is a PERMISSION TIER,
#: omp's `--mode json` is an OUTPUT FORMAT selector, and they share nothing but
#: the four characters. Matching the bare flag would have refused omp's argv for
#: borrowing a mechanism it does not have. Every other measured flag keeps the
#: bare-flag check, which is strictly the stricter reading: `--sandbox`'s only
#: measured use is a permission tier (codex's `read-only`), so an unprobed
#: adapter rendering `--sandbox` under ANY value has borrowed another CLI's
#: containment mechanism and should be probed before it ships — pairing it with
#: `read-only` would let `--sandbox <anything-else>` through unseen. A flag is
#: added here only once a second CLI is measured using it for something that is
#: not a read-only mode.
OVERLOADED_MECHANISM_FLAGS = frozenset({"--mode"})


class TestAnUnprobedRowIsNull:
    """§3.5's rule of record: a row is measured or it is `null`."""

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_a_mechanism_and_its_probe_move_together(self, name):
        row = read_only_modes()[name]
        mechanism, probed = row["mechanism"], row["probed"]

        if mechanism is None:
            assert probed is None, (
                f"{name}: `mechanism: null` with a `probed:` provenance. A probe "
                f"that found a read-only mode belongs in `mechanism:`; one that "
                f"found none is not this row's evidence."
            )
            return

        assert isinstance(mechanism, list) and mechanism, (
            f"{name}: `mechanism:` must be the list of argv tokens the adapter "
            f"renders, got {mechanism!r}."
        )
        assert isinstance(probed, str) and probed.strip(), (
            f"{name}: a non-null `mechanism:` needs its measurement's provenance "
            f"in `probed:`. An unprobed row ships as null, never as a guess — not "
            f"from the adapter's flags, its docs or its name."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_a_probed_mechanism_is_really_in_the_argv(self, name, tmp_path, monkeypatch):
        """A recorded mechanism is a fact about this repo's code, not a note.

        An entry is a whole argv token, with two narrow extensions
        (plans/ask-answer-recovery-and-false-success.md T4.2), because aider's
        measured mechanism has neither part as a token of its own:

        * ``message:<prefix>`` — ``--message``'s rendered value starts with
          ``<prefix>`` (aider's ``/ask ``, which rides inside that value);
        * ``stdin:<answer>`` — on the argv transport,
          ``build_command_for_transport`` returns a ``stdin_text`` that is one
          or more lines, every one exactly ``<answer>`` (aider's decline supply;
          its count is pinned in ``tests/test_aider_provider.py``).

        Neither starts with ``-``, so :meth:`_borrowable` never turns one into a
        flag another row could be accused of borrowing.
        """
        mechanism = read_only_modes()[name]["mechanism"]
        if mechanism is None:
            pytest.skip(f"{name} is a null row — the probe queue, not a claim")

        generate = _generate_argv(name)
        rendered = set(generate)
        readonly = _profile_argv(name, PROFILE_AGENT_READONLY)
        if readonly is not None:
            rendered |= set(readonly)

        def present(entry):
            if entry.startswith("message:"):
                prefix = entry[len("message:"):]
                values = [
                    generate[index + 1]
                    for index, token in enumerate(generate[:-1])
                    if token == "--message"
                ]
                return any(value.startswith(prefix) for value in values)
            if entry.startswith("stdin:"):
                # A hermetic cwd and $HOME: aider refuses before spawn when a
                # config file there turns yes-always on (D4.2), and this guard
                # is about the adapter's code, not the machine it runs on.
                monkeypatch.setenv("HOME", str(tmp_path))
                _argv, stdin_text = _PROVIDERS[name].build_command_for_transport(
                    PROMPT, _model_for(name), "argv", cwd=str(tmp_path)
                )
                answer = entry[len("stdin:"):]
                return (
                    isinstance(stdin_text, str)
                    and stdin_text.endswith("\n")
                    and set(stdin_text[:-1].split("\n")) == {answer}
                )
            return entry in rendered

        missing = [token for token in mechanism if not present(token)]
        assert not missing, (
            f"{name}: read_only_modes records {mechanism} but the adapter renders "
            f"none of {missing} in its `generate` or `agent-readonly` argv. Either "
            f"the mechanism moved and the row is now fiction, or the row was never "
            f"true."
        )

    @staticmethod
    def _borrowable(table, name):
        """Every other row's mechanism, as bare flags plus the one exception.

        A mechanism is recorded as the argv tokens the adapter renders, and two
        of the three measured rows are a flag with the *value* that selects the
        mode: `--mode ask`, `--sandbox read-only`. The flag alone is the borrow
        this guard looks for, because a permission tier's *other* values are
        the ones nobody measured: an unprobed adapter rendering `--sandbox`
        under any value at all has copied codex's containment mechanism, and
        pairing the flag with `read-only` would wave every other value through.

        `OVERLOADED_MECHANISM_FLAGS` is the narrow carve-out, and it holds one
        flag. omp forced it: it renders `--mode json`, an OUTPUT FORMAT
        selector with nothing of cursor-agent's `--mode ask` permission tier in
        it but the four characters, so matching the bare flag would have
        refused an argv that borrowed nothing. The relaxation is scoped to that
        flag alone rather than applied registry-wide, so every other measured
        flag — claude-code's bare pair included — is matched exactly as
        strictly as it was before omp arrived.
        """
        atoms = set()
        for other, row in table.items():
            if other == name or not row["mechanism"]:
                continue
            tokens = list(row["mechanism"])
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if not token.startswith("-"):
                    index += 1
                    continue
                nxt = tokens[index + 1] if index + 1 < len(tokens) else None
                if token in OVERLOADED_MECHANISM_FLAGS and (
                    nxt is not None and not nxt.startswith("-")
                ):
                    atoms.add((token, nxt))
                    index += 2
                else:
                    atoms.add((token,))
                    index += 1
        return atoms

    @staticmethod
    def _atoms_in(argv):
        """The rendered argv as `(flag,)` and `(flag, value)` atoms.

        Both forms for every flag, deliberately: which of the two a borrow is
        judged on is `_borrowable`'s decision, and recording both here is what
        lets one intersection answer for a bare-flag row and for an overloaded
        one without the caller branching per flag.
        """
        atoms = set()
        for index, token in enumerate(argv):
            if not token.startswith("-"):
                continue
            atoms.add((token,))
            nxt = argv[index + 1] if index + 1 < len(argv) else None
            if nxt is not None and not nxt.startswith("-"):
                atoms.add((token, nxt))
        return atoms

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_a_null_row_did_not_acquire_an_invented_read_only_flag(self, name):
        """An unprobed CLI keeps its current argv rather than borrowing a flag.

        The vocabulary checked here is the *other* rows' measured mechanisms —
        `--sandbox`, `--allowed-tools`, `--disallowed-tools` as bare flags, and
        `--mode ask` as a whole atom because `--mode` is the one flag two CLIs
        measurably use for different things (see :meth:`_borrowable` and
        `OVERLOADED_MECHANISM_FLAGS`). An unprobed adapter that suddenly renders
        one of them has had another CLI's mechanism copied onto it, which is the
        specific failure §3.5 names: a flag invented for a row nobody measured.
        If the mechanism is genuinely this CLI's own, probe it and move the row
        — that is the whole point of the queue.
        """
        table = read_only_modes()
        if table[name]["mechanism"] is not None:
            pytest.skip(f"{name} is a probed row and owns its own mechanism")

        # Each profile's argv is read on its own rather than concatenated: a
        # join would invent a `(flag, value)` pair straddling the seam between
        # the last token of one and the first of the next.
        rendered = set(self._atoms_in(_generate_argv(name)))
        readonly = _profile_argv(name, PROFILE_AGENT_READONLY)
        if readonly is not None:
            rendered |= self._atoms_in(readonly)

        borrowed = self._borrowable(table, name)
        found = sorted(" ".join(atom) for atom in borrowed & rendered)
        assert not found, (
            f"{name} is an unprobed (`null`) row but its argv carries {found}, "
            f"which is another adapter's measured read-only mechanism. Probe this "
            f"CLI and record its own row, or drop the flag."
        )


class TestTheYoloClassNegativeSet:
    """No adapter reaches a write bypass from a read-only lane.

    Scanned per §3.5: `build_command` (the `generate` profile — the Lane A
    review/ask fan-out, which consumes documents and never diffs) and `get_env`,
    plus `agent-readonly`, which AGENTS.md already covers verbatim. `agent-write`
    is the one profile the rule permits a bypass on, and only for a runner with
    no native sandbox; it has its own test below.
    """

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_generate_carries_no_bypass_token(self, name):
        allowed = _exception_token_for(name, PROFILE_GENERATE)

        found = [
            token
            for token in negative_tokens_in_argv(_generate_argv(name))
            if token not in allowed
        ]
        assert not found, (
            f"{name}: build_command emits {found} on the `generate` profile. "
            f"`generate` is a read/analyze dispatch — every phase that reaches it "
            f"consumes a document, never a diff — so a write bypass there grants "
            f"edit access no phase asked for. Use the runner's own read-only "
            f"mechanism, or, if it has none, record a `null` row plus a named "
            f"`bypass_exception:` in providers.yaml's read_only_modes table. An "
            f"exception declared for another profile does not reach this one."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_get_env_carries_no_bypass_token(self, name):
        # `get_env` takes no profile: one env delta rides every dispatch this
        # adapter makes, `generate` included. So only a `generate`-scoped
        # exception can excuse a bypass here — one written for `agent-write`
        # would still be reachable from the read/analyze lane, which is the
        # thing this class exists to refuse.
        allowed = _exception_token_for(name, PROFILE_GENERATE)

        found = [
            token
            for token in negative_tokens_in_env(_env_for(name))
            if token not in allowed
        ]
        assert not found, (
            f"{name}: get_env carries {found}. The env is the same lane as the "
            f"argv — `GOOSE_MODE=auto` is a bypass exactly like `--yolo` is — and "
            f"it is not profile-scoped, so only a `generate` exception excuses it."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_an_exception_only_excuses_the_profile_it_declares(self, name):
        """`profile:` decides which sweep an exception reaches, not just reads.

        Mutation-shaped, and derived from what ships rather than from a case
        written here: each declared exception is re-scoped to `agent-write` and
        at least one of the two sweeps above must go red again. With the allowed
        set keyed on the adapter alone this test fails — which is precisely the
        hole: an exception recorded for the one profile §3.5 permits a bypass on
        would silence the two lanes where it permits none.
        """
        declared = _rows_with_exceptions().get(name)
        if declared is None:
            pytest.skip(f"{name} declares no bypass exception")

        rescoped = {name: {**declared, "profile": PROFILE_AGENT_WRITE}}
        went_red = []
        with mock.patch(f"{__name__}._rows_with_exceptions", lambda: rescoped):
            for sweep in (
                self.test_generate_carries_no_bypass_token,
                self.test_get_env_carries_no_bypass_token,
            ):
                try:
                    sweep(name)
                except AssertionError:
                    went_red.append(sweep.__name__)

        assert went_red, (
            f"{name}: its bypass_exception was re-scoped to "
            f"{PROFILE_AGENT_WRITE!r} and both the `generate` and `get_env` "
            f"sweeps stayed green, so {declared['token']!r} is exempted by the "
            f"adapter's name alone and `profile:` decides nothing."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_agent_readonly_carries_no_bypass_token(self, name):
        argv = _profile_argv(name, PROFILE_AGENT_READONLY)
        if argv is None:
            pytest.skip(f"{name} does not declare agent-readonly")

        found = negative_tokens_in_argv(argv)
        assert not found, (
            f"{name}: its agent-readonly argv emits {found}. AGENTS.md covers this "
            f"profile verbatim — the runner's own denial mechanism is the contract "
            f"and a bypass beside it cancels the profile."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_the_agent_write_exemption_is_still_unused(self, name):
        """§3.5 permits a bypass here and no shipped adapter needs one.

        The permission is narrow: `agent-write` may carry a `--yolo`-class flag
        **only** where the runner has no native sandbox or permission mode, and
        the three adapters pinned to that profile all have one (`claude-code`
        `--permission-mode acceptEdits`, `codex` `--sandbox workspace-write`,
        `opencode` `--agent build`). So the row is empty, and this test says so
        rather than leaving it inferred. A genuinely sandbox-less write runner
        arriving later is the one case where changing this test is the right
        move — together with AGENTS.md's pin, never quietly.
        """
        argv = _profile_argv(name, PROFILE_AGENT_WRITE)
        if argv is None:
            pytest.skip(f"{name} does not declare agent-write")

        found = negative_tokens_in_argv(argv)
        assert not found, (
            f"{name}: its agent-write argv emits {found}. That is permitted only "
            f"for a runner with no native sandbox or permission mode; this one has "
            f"one, and the write boundary is write_safety plus the runner's own."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_an_exception_declares_every_field_that_makes_it_a_disposition(self, name):
        declared = _rows_with_exceptions().get(name)
        if declared is None:
            pytest.skip(f"{name} declares no bypass exception")

        assert set(declared) == EXCEPTION_KEYS, (
            f"{name}: bypass_exception keys {sorted(declared)} != "
            f"{sorted(EXCEPTION_KEYS)}. Every field is load-bearing: without "
            f"`retires_when` the entry is 'it was already there' with extra words."
        )
        for key in sorted(EXCEPTION_KEYS):
            assert isinstance(declared[key], str) and declared[key].strip(), (
                f"{name}: bypass_exception.{key} is empty."
            )
        assert declared["profile"] in {
            PROFILE_GENERATE,
            PROFILE_AGENT_READONLY,
            PROFILE_AGENT_WRITE,
        }, f"{name}: bypass_exception.profile is {declared['profile']!r}."

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_an_exception_row_is_a_null_row(self, name):
        """§3.5's disposition is the `null` row **and** the named exception.

        A row that claims a probed read-only mechanism and still needs a write
        bypass beside it is a contradiction: the mechanism would be the reason
        the bypass is unnecessary.
        """
        row = read_only_modes()[name]
        if row.get("bypass_exception") is None:
            pytest.skip(f"{name} declares no bypass exception")

        assert row["mechanism"] is None, (
            f"{name}: declares both a probed mechanism {row['mechanism']} and a "
            f"bypass exception. Once the mechanism is measured the exception is "
            f"what it retires into — the two cannot both be true."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_no_exception_outlives_the_flag_it_excuses(self, name):
        """The escape hatch closes itself.

        Without this, an exception survives the removal of its flag and quietly
        pre-authorizes the *next* person to add one back. With it, retiring the
        flag forces the entry out of `providers.yaml` in the same change.
        """
        declared = _rows_with_exceptions().get(name)
        if declared is None:
            pytest.skip(f"{name} declares no bypass exception")

        emitted = set(negative_tokens_in_argv(_generate_argv(name)))
        emitted |= set(negative_tokens_in_env(_env_for(name)))
        readonly = _profile_argv(name, PROFILE_AGENT_READONLY)
        if readonly is not None:
            emitted |= set(negative_tokens_in_argv(readonly))

        assert declared["token"] in emitted, (
            f"{name}: providers.yaml still excuses {declared['token']!r} but the "
            f"adapter no longer emits it. Delete the bypass_exception — a stale "
            f"exception is a standing permission nobody re-reviewed."
        )

    def test_the_guard_reads_this_repos_source_and_no_vendor_surface(self):
        """§3.5's provenance split, asserted rather than left to a comment.

        The shipping guard is internal: it scans what *we* emit. The donor's
        upstream claims ("opencode and kilo removed the Claude-style skip flag",
        "Orca actively strips it") were never probed and must not be encoded as
        an assertion about anyone's CLI — a vendor re-adding `--yolo` upstream
        changes nothing about what this repo emits.
        """
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        reaches_outside = imported & {
            "subprocess", "shutil", "socket", "urllib", "http", "requests",
        }
        assert not reaches_outside, (
            f"this guard imports {sorted(reaches_outside)}, which is how it would "
            f"start reading a vendor's help text or a network surface instead of "
            f"this repo's source. §3.5 keeps the two apart on purpose: what we "
            f"emit is ours to assert and permanent by construction; what a CLI "
            f"offers is a vendor's to change. The advisory route for the latter "
            f"already exists — cli-surface.yaml's container help probe."
        )


class TestTheOneDispatchTimeOverride:
    """`DISPATCH_TIME_OVERRIDE`, a ladder, checked rung by rung from both sides.

    D6 (2026-09-17) made `danger-full-access` the one dispatch-time override;
    the landlock rung (2026-09-22) put a contained rung above it. The sweeps
    above prove the default argv of every profile is bypass-free —
    `danger-full-access` is in `NEGATIVE_VALUE_TOKENS`, so they would go red
    the day it leaked into a default. This class proves the other half for
    EVERY rung: the rewrite exists, renders exactly the declared atoms on
    exactly the declared profiles, is unreachable from the write profile, is
    absent from every default argv, and is a no-op on every adapter that is not
    the declared one — so the ladder cannot quietly grow a second adapter, a
    third profile or a third rung without this file changing. Rung 1 gets the
    one assertion that is its own: it ALWAYS renders `--sandbox read-only`
    beside its feature and never `workspace-write`, because that pairing is the
    exit-101 panic.
    """

    FEATURE = "features.use_legacy_landlock"

    @staticmethod
    def _override_argv(name, profile, rung, transport="argv"):
        provider = _PROVIDERS[name]
        if profile == PROFILE_GENERATE:
            argv, _ = provider.build_command_for_transport(
                PROMPT, _model_for(name), transport, sandbox_fallback=rung
            )
            return argv
        if profile not in provider.agent_profiles:
            return None
        argv, _ = provider.build_agent_command_for_transport(
            PROMPT, _model_for(name), profile, transport, sandbox_fallback=rung
        )
        return argv

    @staticmethod
    def _default_argv(name, profile):
        if profile == PROFILE_GENERATE:
            return _generate_argv(name)
        return _profile_argv(name, profile)

    def _mentions_feature(self, argv):
        return any(self.FEATURE in element for element in argv)

    def test_the_declaration_carries_every_field_of_a_disposition(self):
        assert set(DISPATCH_TIME_OVERRIDE) == {"provider", "profiles", "rungs"}
        assert set(DISPATCH_TIME_OVERRIDE["profiles"]) == {
            PROFILE_GENERATE, PROFILE_AGENT_READONLY,
        }, "the override may only ever cover the two read-only profiles"
        for row in DISPATCH_TIME_OVERRIDE["rungs"]:
            assert set(row) == {
                "rung", "rewrite", "token", "contained", "reason", "retires_when",
            }, row["rung"]
            for key in ("rung", "reason", "retires_when"):
                assert isinstance(row[key], str) and row[key].strip(), (row["rung"], key)
            assert row["rewrite"][:1] == ("--sandbox",), row["rung"]

    def test_only_the_uncontained_rung_owes_a_negative_token(self):
        """The guard has to be able to see the thing it is excusing — and a
        real boundary is not a bypass, so the contained rung excuses nothing."""
        for row in DISPATCH_TIME_OVERRIDE["rungs"]:
            if row["contained"]:
                assert row["token"] is None, row["rung"]
                assert negative_tokens_in_argv(list(row["rewrite"])) == [], row["rung"]
            else:
                assert row["token"] in NEGATIVE_VALUE_TOKENS, row["rung"]
                assert negative_tokens_in_argv(list(row["rewrite"])) == [row["token"]]

    def test_the_declaration_matches_the_base_class_tuple(self):
        """One tuple, declared once: the adapters read it, this file pins it."""
        assert tuple(DISPATCH_TIME_OVERRIDE["profiles"]) == tuple(
            base_module.SANDBOX_FALLBACK_PROFILES
        )
        assert PROFILE_AGENT_WRITE not in base_module.SANDBOX_FALLBACK_PROFILES

    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    def test_the_declaration_matches_the_adapters_ladder(self, profile):
        """Same rungs, same order, same atoms, same containment — as the adapter
        declares them, so neither side can move without the other."""
        name = DISPATCH_TIME_OVERRIDE["provider"]
        declared = _PROVIDERS[name].sandbox_fallback_rungs(profile)
        assert tuple(r.name for r in declared) == OVERRIDE_RUNGS
        for rung in declared:
            row = _override_rung(rung.name)
            assert tuple(rung.argv_rewrite) == row["rewrite"], rung.name
            assert rung.contained is row["contained"], rung.name
        assert _PROVIDERS[name].sandbox_fallback_rungs(PROFILE_AGENT_WRITE) == ()

    @pytest.mark.parametrize("transport", ["argv", "stdin"])
    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    @pytest.mark.parametrize("rung", OVERRIDE_RUNGS)
    def test_the_override_renders_the_declared_rewrite_on_a_read_only_profile(
        self, rung, profile, transport
    ):
        row = _override_rung(rung)
        argv = self._override_argv(
            DISPATCH_TIME_OVERRIDE["provider"], profile, rung, transport
        )

        expected = [] if row["token"] is None else [row["token"]]
        assert negative_tokens_in_argv(argv) == expected
        # The `--sandbox` value rides in the same slot the profile's own row
        # occupies, so the override is a value change and not a new flag...
        at = argv.index("--sandbox")
        assert argv[at:at + 2] == list(row["rewrite"][:2])
        # ...and any further atoms (rung 1's `-c` pair) appear exactly once,
        # adjacent, never anywhere else.
        extra = list(row["rewrite"][2:])
        if extra:
            hits = [i for i in range(len(argv) - len(extra) + 1)
                    if argv[i:i + len(extra)] == extra]
            assert len(hits) == 1, (rung, profile, argv)
        else:
            assert not self._mentions_feature(argv), (rung, profile, argv)

    @pytest.mark.parametrize("transport", ["argv", "stdin"])
    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    def test_rung_one_always_pairs_its_feature_with_read_only(self, profile, transport):
        """`workspace-write` + the feature is the exit-101 panic, and `generate`'s
        own row IS `workspace-write` — so the rung must overwrite it, always."""
        argv = self._override_argv(
            DISPATCH_TIME_OVERRIDE["provider"], profile, "landlock", transport
        )
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert "workspace-write" not in argv
        assert self._mentions_feature(argv)

    @pytest.mark.parametrize(
        "profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY, PROFILE_AGENT_WRITE]
    )
    def test_the_default_argv_carries_no_rung(self, profile):
        """Mutation-shaped: drop the kwarg and every rung must vanish — neither
        the bypass token nor the landlock feature is in any default argv."""
        argv = self._default_argv(DISPATCH_TIME_OVERRIDE["provider"], profile)
        assert negative_tokens_in_argv(argv) == []
        assert not self._mentions_feature(argv)

    def test_generate_keeps_workspace_write_on_the_healthy_path(self):
        """The other side of rung 1's `read-only`: it is rendered only on a
        dispatch that took the rung. `generate`'s default argv and its table
        row both still say `workspace-write`, with no feature beside it — the
        rung cannot leak into the healthy-machine path or into
        `_SANDBOX_BY_PROFILE`."""
        from utils.providers import codex as codex_module

        argv = _generate_argv(DISPATCH_TIME_OVERRIDE["provider"])
        assert argv[argv.index("--sandbox") + 1] == "workspace-write"
        assert not self._mentions_feature(argv)
        assert codex_module._SANDBOX_BY_PROFILE[PROFILE_GENERATE] == "workspace-write"

    @pytest.mark.parametrize("rung", OVERRIDE_RUNGS)
    def test_the_write_profile_raises_rather_than_rendering_it(self, rung):
        name = DISPATCH_TIME_OVERRIDE["provider"]
        with pytest.raises(
            ValueError, match="never runs the 'agent-write' profile on a sandbox fallback"
        ):
            self._override_argv(name, PROFILE_AGENT_WRITE, rung)
        # And the policy hook answers None for it whatever the config says.
        assert _PROVIDERS[name].sandbox_fallback_policy(PROFILE_AGENT_WRITE) is None

    @pytest.mark.parametrize("rung", OVERRIDE_RUNGS)
    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_no_other_adapter_changes_its_argv_under_a_rung(self, name, rung):
        """The kwarg is accepted everywhere and acted on in exactly one place."""
        if name == DISPATCH_TIME_OVERRIDE["provider"]:
            pytest.skip("the declared adapter — covered above")
        assert self._override_argv(name, PROFILE_GENERATE, rung) == _generate_argv(name)
        readonly = self._override_argv(name, PROFILE_AGENT_READONLY, rung)
        assert readonly == _profile_argv(name, PROFILE_AGENT_READONLY)
        assert _PROVIDERS[name].sandbox_fallback_policy(PROFILE_GENERATE) is None

    def test_the_policy_defaults_to_auto_and_reads_a_bare_boolean_as_its_word(
        self, monkeypatch, capsys
    ):
        """`auto` ships; YAML's `off`/`false` is `off` and `on`/`true` is
        `auto`, silently — the key exists to be turned off, so there is no
        spelling of "on" that needs a warning. The retired `external` is not
        an alias: it warns like any other unknown word and lands on the
        default, which is the behaviour it used to ask for."""
        from utils.providers import codex as codex_module

        base_module.reset_env_policy_warnings()
        monkeypatch.setattr(
            codex_module, "_provider_block", lambda name, anchor=None: {}
        )
        assert codex_module.sandbox_fallback_policy() == "auto"
        assert base_module.SANDBOX_FALLBACK_POLICIES == ("off", "auto", "contained")
        for value, expected, warns in (
            ("auto", "auto", False), ("AUTO", "auto", False),
            ("off", "off", False), (False, "off", False), (True, "auto", False),
            # The landlock rung's third value: not a word YAML coerces, so it
            # has no bool leg and resolves verbatim.
            ("contained", "contained", False), ("Contained", "contained", False),
            ("external", "auto", True), ("garbage", "auto", True),
        ):
            base_module.reset_env_policy_warnings()
            capsys.readouterr()
            monkeypatch.setattr(
                codex_module, "_provider_block",
                lambda name, anchor=None, v=value: {"sandbox_fallback": v},
            )
            assert codex_module.sandbox_fallback_policy() == expected, value
            err = capsys.readouterr().err
            assert ("providers.codex.sandbox_fallback" in err) is warns, (value, err)


class TestTheLandlockPairInTheGoldenNormalizer:
    """`tests/golden/normalize_argv.py` classifies rung 1's `-c` pair explicitly.

    The container lane's argv golden reduces run-varying values to placeholders;
    a policy token reduced to one makes the golden row vacuous (§14.1). Rung 1's
    `features.use_legacy_landlock=true` is policy in both halves, so it must
    survive normalization byte-for-byte — beside codex's row-derived `-c`
    overrides, which still collapse — and it must be a named entry rather than
    whatever `_classify`'s generic split happens to do with it.
    """

    def test_the_pair_is_kept_verbatim_and_named(self):
        from .golden import normalize_argv as golden

        assert "features.use_legacy_landlock" in golden._KV_POLICY_KEYS
        assert "features.use_legacy_landlock" not in golden._KV_ROW_DERIVED_KEYS
        argv, _ = _PROVIDERS["codex"].build_agent_command_for_transport(
            PROMPT, _model_for("codex"), PROFILE_AGENT_READONLY, "stdin",
            sandbox_fallback="landlock",
        )
        view = golden.normalize_argv(argv)
        dash_c = view["options"]["-c"]
        values = dash_c if isinstance(dash_c, list) else [dash_c]
        assert "features.use_legacy_landlock=true" in values
        assert view["options"]["--sandbox"] == "read-only"

    def test_a_different_value_does_not_normalize_to_the_same_row(self):
        from .golden.normalize_argv import normalize_argv

        on = normalize_argv(["codex", "exec", "-c", "features.use_legacy_landlock=true"])
        off = normalize_argv(["codex", "exec", "-c", "features.use_legacy_landlock=false"])
        assert on != off


class TestAgentsMdCarriesTheProfileScopedRule:
    """AGENTS.md is where the next adapter author reads the rule, so it must say it.

    §3.5's gap was that the shipped bullet is scoped to `agent-readonly` and says
    nothing about `generate` — which is how a draft could put `--yolo`-class
    flags on the fan-out without contradicting the letter of the rule while
    inverting its point. A plan file is not a fix for that; the amendment is.
    """

    @staticmethod
    def _bullet():
        path = Path(__file__).resolve().parent.parent / "AGENTS.md"
        text = path.read_text(encoding="utf-8")
        bullets = [
            line
            for line in text.splitlines()
            if "A read/analyze dispatch never gets write access" in line
        ]
        assert len(bullets) == 1, (
            f"expected exactly one read/analyze bullet in {path}, found "
            f"{len(bullets)}"
        )
        return bullets[0]

    def test_the_rule_is_extended_from_agent_readonly_to_generate(self):
        bullet = self._bullet()

        for phrase in ("generate", "build_command", "profile-scoped"):
            assert phrase in bullet, (
                f"AGENTS.md's read/analyze bullet does not mention {phrase!r}. The "
                f"rule as written covers `agent-readonly` only, and that gap is "
                f"what §3.5 exists to close."
            )

    def test_the_bullet_names_every_token_of_the_negative_set(self):
        bullet = self._bullet()

        for token in sorted(NEGATIVE_FLAG_TOKENS | NEGATIVE_VALUE_TOKENS):
            assert f"`{token}`" in bullet, (
                f"AGENTS.md's read/analyze bullet does not name `{token}`. An "
                f"adapter author reads this list, not the test module."
            )
        for prefix in NEGATIVE_TOKEN_PREFIXES:
            assert f"`{prefix}" in bullet
        for name, value in NEGATIVE_ENV.items():
            assert f"`{name}={value}`" in bullet

    def test_the_bullet_carries_the_null_row_plus_named_exception_disposition(self):
        bullet = self._bullet()

        assert "read_only_modes" in bullet
        assert re.search(r"`?None`? row|`null` row", bullet), (
            "AGENTS.md must say what happens to a runner whose only unattended "
            "mode is a bypass: an explicit None row, never a silent precedent."
        )
        assert "named exception" in bullet
        assert Path(__file__).name in bullet, (
            "AGENTS.md should name the guard, so the rule and its enforcement are "
            "one click apart."
        )
