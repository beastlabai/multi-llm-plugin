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
#: (`--permission-mode bypassPermissions`, `--sandbox danger-full-access`), so
#: they are matched wherever they sit. `danger-full-access` joined the set with
#: the D6 override (2026-09-17): it is codex's own third sandbox policy, and
#: before it was here the guard was blind to it — a value, not a `--dangerously`
#: prefix. The default argv of every profile must still never carry it; the one
#: place it may appear is the dispatch-time rewrite `DISPATCH_TIME_OVERRIDE`
#: declares below, and that class checks the rewrite from both sides.
NEGATIVE_VALUE_TOKENS = frozenset({"bypassPermissions", "danger-full-access"})

#: The ONE dispatch-time override of the negative set, declared as data with the
#: same four fields a `bypass_exception:` row carries — token, profile(s), why it
#: survives, what retires it — because §3.5's rule is "never a silent
#: precedent", and this is not a `read_only_modes:` row: codex's row is a
#: *probed* mechanism (`--sandbox read-only`), and the table's own invariant
#: (`test_an_exception_row_is_a_null_row`) rightly refuses a row that claims
#: both a mechanism and an exception. What differs from a `bypass_exception:`
#: is where the token lives: never in a rendered default, only behind a kwarg
#: (`sandbox_fallback=True`) that a gate sets after all four of its conditions
#: hold. `TestTheOneDispatchTimeOverride` reads it from both sides — the
#: default argv stays clean, the override renders exactly this token on
#: exactly these profiles, and the write profile raises rather than renders.
DISPATCH_TIME_OVERRIDE = {
    "provider": "codex",
    "token": "danger-full-access",
    "profiles": (PROFILE_GENERATE, PROFILE_AGENT_READONLY),
    "reason": (
        "D6 (2026-09-17): on a machine where bubblewrap cannot start, the "
        "operator's `providers.codex.sandbox_fallback: external` attests an "
        "external boundary; the D6 probe verdict must be `unavailable` (never "
        "`unknown`) and the profile read-only. A by-hand run under this value "
        "found 2 unique validated-high issues the sandboxed reviewers did not."
    ),
    "retires_when": (
        "the AppArmor repair `multi_llm.py doctor` prints is the norm on the "
        "hosts this runs on, or codex ships a sandbox that does not need a "
        "user namespace — then the key and the rewrite go together."
    ),
}

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
    def test_a_probed_mechanism_is_really_in_the_argv(self, name):
        """A recorded mechanism is a fact about this repo's code, not a note."""
        mechanism = read_only_modes()[name]["mechanism"]
        if mechanism is None:
            pytest.skip(f"{name} is a null row — the probe queue, not a claim")

        rendered = set(_generate_argv(name))
        readonly = _profile_argv(name, PROFILE_AGENT_READONLY)
        if readonly is not None:
            rendered |= set(readonly)

        missing = [token for token in mechanism if token not in rendered]
        assert not missing, (
            f"{name}: read_only_modes records {mechanism} but the adapter renders "
            f"none of {missing} in its `generate` or `agent-readonly` argv. Either "
            f"the mechanism moved and the row is now fiction, or the row was never "
            f"true."
        )

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_a_null_row_did_not_acquire_an_invented_read_only_flag(self, name):
        """An unprobed CLI keeps its current argv rather than borrowing a flag.

        The vocabulary checked here is the *other* rows' measured flag tokens —
        `--mode`, `--sandbox`, `--allowed-tools`, `--disallowed-tools` today. An
        unprobed adapter that suddenly renders one of them has had another CLI's
        mechanism copied onto it, which is the specific failure §3.5 names: a
        flag invented for a row nobody measured. If the flag is genuinely this
        CLI's own, probe it and move the row — that is the whole point of the
        queue.
        """
        table = read_only_modes()
        if table[name]["mechanism"] is not None:
            pytest.skip(f"{name} is a probed row and owns its own mechanism")

        borrowed = {
            token
            for other, row in table.items()
            if other != name and row["mechanism"]
            for token in row["mechanism"]
            if token.startswith("-")
        }

        argv = list(_generate_argv(name))
        readonly = _profile_argv(name, PROFILE_AGENT_READONLY)
        if readonly is not None:
            argv += readonly

        found = sorted(borrowed.intersection(argv))
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
    """`DISPATCH_TIME_OVERRIDE`, checked from both sides (D6, 2026-09-17).

    The sweeps above prove the default argv of every profile is bypass-free —
    `danger-full-access` is in `NEGATIVE_VALUE_TOKENS`, so they would go red
    the day it leaked into a default. This class proves the other half: the
    rewrite exists, renders exactly the declared token on exactly the declared
    profiles, is unreachable from the write profile, and is a no-op on every
    adapter that is not the declared one — so the override cannot quietly grow
    a second adapter or a third profile without this file changing.
    """

    @staticmethod
    def _override_argv(name, profile):
        provider = _PROVIDERS[name]
        if profile == PROFILE_GENERATE:
            argv, _ = provider.build_command_for_transport(
                PROMPT, _model_for(name), "argv", sandbox_fallback=True
            )
            return argv
        if profile not in provider.agent_profiles:
            return None
        argv, _ = provider.build_agent_command_for_transport(
            PROMPT, _model_for(name), profile, "argv", sandbox_fallback=True
        )
        return argv

    def test_the_declared_token_is_in_the_negative_set(self):
        """The guard has to be able to see the thing it is excusing."""
        assert DISPATCH_TIME_OVERRIDE["token"] in NEGATIVE_VALUE_TOKENS

    def test_the_declaration_carries_every_field_of_a_disposition(self):
        assert set(DISPATCH_TIME_OVERRIDE) == {
            "provider", "token", "profiles", "reason", "retires_when",
        }
        for key in ("provider", "token", "reason", "retires_when"):
            value = DISPATCH_TIME_OVERRIDE[key]
            assert isinstance(value, str) and value.strip(), key
        assert set(DISPATCH_TIME_OVERRIDE["profiles"]) == {
            PROFILE_GENERATE, PROFILE_AGENT_READONLY,
        }, "the override may only ever cover the two read-only profiles"

    def test_the_declaration_matches_the_base_class_tuple(self):
        """One tuple, declared once: the adapters read it, this file pins it."""
        assert tuple(DISPATCH_TIME_OVERRIDE["profiles"]) == tuple(
            base_module.SANDBOX_FALLBACK_PROFILES
        )
        assert PROFILE_AGENT_WRITE not in base_module.SANDBOX_FALLBACK_PROFILES

    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    def test_the_override_renders_the_declared_token_on_a_read_only_profile(self, profile):
        name = DISPATCH_TIME_OVERRIDE["provider"]
        argv = self._override_argv(name, profile)

        assert negative_tokens_in_argv(argv) == [DISPATCH_TIME_OVERRIDE["token"]]
        # And ONLY as the `--sandbox` value: the same slot the profile's own
        # row occupies, so the override is a value change and not a new flag.
        assert argv[argv.index("--sandbox") + 1] == DISPATCH_TIME_OVERRIDE["token"]

    @pytest.mark.parametrize("profile", [PROFILE_GENERATE, PROFILE_AGENT_READONLY])
    def test_the_default_argv_of_the_same_profile_stays_clean(self, profile):
        """Mutation-shaped: drop the kwarg and the token must vanish."""
        name = DISPATCH_TIME_OVERRIDE["provider"]
        if profile == PROFILE_GENERATE:
            argv = _generate_argv(name)
        else:
            argv = _profile_argv(name, profile)
        assert negative_tokens_in_argv(argv) == []

    def test_the_write_profile_raises_rather_than_rendering_it(self):
        name = DISPATCH_TIME_OVERRIDE["provider"]
        with pytest.raises(ValueError, match="never runs the 'agent-write' profile unsandboxed"):
            self._override_argv(name, PROFILE_AGENT_WRITE)
        # And the policy hook answers None for it whatever the config says.
        assert _PROVIDERS[name].sandbox_fallback_policy(PROFILE_AGENT_WRITE) is None

    @pytest.mark.parametrize("name", _registered_provider_names())
    def test_no_other_adapter_changes_its_argv_under_the_flag(self, name):
        """The kwarg is accepted everywhere and acted on in exactly one place."""
        if name == DISPATCH_TIME_OVERRIDE["provider"]:
            pytest.skip("the declared adapter — covered above")
        assert self._override_argv(name, PROFILE_GENERATE) == _generate_argv(name)
        readonly = self._override_argv(name, PROFILE_AGENT_READONLY)
        assert readonly == _profile_argv(name, PROFILE_AGENT_READONLY)
        assert _PROVIDERS[name].sandbox_fallback_policy(PROFILE_GENERATE) is None

    def test_the_policy_defaults_to_off_and_refuses_a_bare_boolean(self, monkeypatch):
        """`off` ships; `on`/`true` is not a spelling of the attestation."""
        from utils.providers import codex as codex_module

        base_module.reset_env_policy_warnings()
        monkeypatch.setattr(
            codex_module, "_provider_block", lambda name, anchor=None: {}
        )
        assert codex_module.sandbox_fallback_policy() == "off"
        for value, expected in (
            ("external", "external"), ("EXTERNAL", "external"), (False, "off"),
            (True, "off"), ("on", "off"), ("yes", "off"), ("garbage", "off"),
        ):
            base_module.reset_env_policy_warnings()
            monkeypatch.setattr(
                codex_module, "_provider_block",
                lambda name, anchor=None, v=value: {"sandbox_fallback": v},
            )
            assert codex_module.sandbox_fallback_policy() == expected, value


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
