"""M12 (§6.1) — the canonical reasoning-effort ladder and its per-provider subsets.

**Every sweep in this module is parametrized over the live provider registry**
(:func:`_registry_provider_names`), never over a written-down roster of adapter
names and never against a count of them. That is M0.a's binding rule for the
whole of Group M, and it is not a style preference: the specifications these
tests implement were written against a smaller registry than the one that ships,
so any assertion phrased "across all N adapters" is
under-specified rather than merely stale. A new adapter is picked up here the
moment it is registered, and :class:`TestTheSweepsCannotMissANewAdapter` reads
this module's own AST to prove that no sweep below quietly grew a literal list.

The one place a proper noun is allowed is a *model* id in a regression test for
a specific measurement (:class:`TestPerModelCaps`). A model is not an adapter
roster — a measurement about one model cannot be made registry-derived without
losing the thing it records.
"""
import ast
import importlib
import pathlib

import pytest

from utils.providers.base import (
    LADDER_POSITION,
    LADDER_RUNGS,
    REASONING_LADDER,
    SECOND_AXIS_VARIANTS,
    UNADVERTISED_RUNGS,
    NO_REASONING_LADDER,
    ReasoningLadder,
    split_reasoning_effort,
)


def _registry_provider_names():
    """Every provider id in the registry, so the sweeps below cannot miss a new one.

    Deliberately the same shape (and the same docstring sentence) as
    ``tests/test_providers.py::_registered_provider_names`` — that helper's
    docstring *is* M0.a's requirement, and a second module implementing the same
    rule differently would be the drift the rule exists to prevent. It is a
    twin rather than an import because ``@pytest.mark.parametrize`` evaluates its
    argument while a class body runs, so the callable has to be importable and
    cheap at collection time.
    """
    from utils.provider_registry import _PROVIDERS

    return sorted(_PROVIDERS)


def _adapter_module(name):
    """The adapter module backing registry id ``name``."""
    from utils.provider_registry import _PROVIDERS

    return importlib.import_module(type(_PROVIDERS[name]).__module__)


def _ladder(name):
    """The ``EFFORT_LADDER`` registry id ``name`` declares."""
    return getattr(_adapter_module(name), "EFFORT_LADDER", None)


def _declared_whitelist(name):
    """The module's ``REASONING_EFFORTS``, or ``None`` when it declares none."""
    return getattr(_adapter_module(name), "REASONING_EFFORTS", None)


def _base_config():
    from utils.provider_registry import load_base_config

    return load_base_config()


def _configured_specs(name):
    """Every model spec the shipped config can dispatch to provider ``name``.

    The per-provider catalog plus the rows of ``defaults.models`` /
    ``defaults.quick_models`` addressed to it, each stripped of its
    ``<provider>:`` prefix so what is left is the model spec the adapter sees.
    """
    config = _base_config()
    providers = config.get("providers") or {}
    defaults = config.get("defaults") or {}

    specs = [
        spec
        for spec in ((providers.get(name) or {}).get("models") or [])
        if isinstance(spec, str)
    ]
    for key in ("models", "quick_models"):
        for spec in defaults.get(key) or []:
            if isinstance(spec, str) and spec.startswith(f"{name}:"):
                specs.append(spec.split(":", 1)[1])
    return specs


class TestTheLadderItself:
    """Rule 1: seven rungs, one total order, and nothing else joins it."""

    def test_the_ladder_is_seven_rungs_weakest_to_strongest(self):
        """The canonical vocabulary, stated once for the whole tree."""
        assert REASONING_LADDER == (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )

    def test_rungs_are_unique(self):
        """A repeated rung would make ``LADDER_POSITION`` silently lossy."""
        assert len(set(REASONING_LADDER)) == len(REASONING_LADDER)

    def test_membership_set_matches_the_order(self):
        assert LADDER_RUNGS == frozenset(REASONING_LADDER)

    def test_positions_are_the_ladder_order(self):
        """``LADDER_POSITION`` is the order, not a second opinion about it."""
        assert [rung for rung, _ in sorted(LADDER_POSITION.items(), key=lambda kv: kv[1])] == list(
            REASONING_LADDER
        )

    def test_unadvertised_rungs_are_rungs(self):
        """Rule 3's two names are on the ladder — they are unadvertised, not off it."""
        assert UNADVERTISED_RUNGS <= LADDER_RUNGS

    def test_second_axis_variants_are_not_rungs(self):
        """Rule 2: ``thinking``/``instant`` select a MODE, so they are off-ladder."""
        assert not (SECOND_AXIS_VARIANTS & LADDER_RUNGS)

    def test_the_two_alias_words_are_not_rungs(self):
        """``off`` and ``ultra`` are per-CLI spellings, never ladder members.

        If either were admitted as a rung, ``off`` would sort below ``none`` and
        ``ultra`` above ``max`` and the order would stop being a measurement of
        anything.
        """
        assert "off" not in LADDER_RUNGS
        assert "ultra" not in LADDER_RUNGS

    def test_the_undeclared_default_is_empty_and_not_a_decline(self):
        """A forgetful new adapter must fail both branches of the sweep below.

        ``NO_REASONING_LADDER`` accepts nothing *and* gives no reason, so it can
        neither pass as "declares a subset" nor as "declines with a reason".
        """
        assert NO_REASONING_LADDER.accepted == frozenset()
        assert NO_REASONING_LADDER.declined == ""


class TestEveryRegisteredProviderDeclaresItsPosition:
    """The registry sweep: every adapter, with no second roster written down."""

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_provider_declares_an_effort_ladder(self, provider_name):
        """Every adapter states a position, including "this CLI has none"."""
        ladder = _ladder(provider_name)

        assert isinstance(ladder, ReasoningLadder), (
            f"{provider_name} declares no EFFORT_LADDER. Every registered adapter "
            "states its position on the canonical ladder (M12/§6.1) — an adapter "
            "whose CLI has no effort mechanism declares "
            "`ReasoningLadder(provider=…, declined='<the measurement>')`."
        )

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_provider_field_matches_the_registry_key(self, provider_name):
        """The ladder names itself with the id the registry knows it by."""
        assert _ladder(provider_name).provider == provider_name

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_declaration_and_whitelist_are_one_fact(self, provider_name):
        """``REASONING_EFFORTS`` is *derived* from the ladder, never restated.

        The whole point of M12: an adapter that hand-writes a whitelist beside
        its declaration has two owners for one fact and nothing pinning them
        together. Either the module derives its whitelist from the ladder, or it
        declines an effort mechanism outright and declares neither.
        """
        ladder = _ladder(provider_name)
        whitelist = _declared_whitelist(provider_name)

        if whitelist is None:
            assert ladder.declined, (
                f"{provider_name} declares no REASONING_EFFORTS and gives no "
                "reason. Declining an effort mechanism is a measurement and it "
                "is written down: set `declined=` to what was measured."
            )
            assert ladder.accepted == frozenset()
        else:
            assert not ladder.declined, (
                f"{provider_name} declares REASONING_EFFORTS *and* declines an "
                "effort mechanism. Exactly one of those is true."
            )
            assert ladder.accepted == frozenset(whitelist), (
                f"{provider_name}'s REASONING_EFFORTS is not its ladder's accepted "
                f"set: ladder={sorted(ladder.accepted)} whitelist={sorted(whitelist)}. "
                "Derive it (`REASONING_EFFORTS = EFFORT_LADDER.accepted`) rather "
                "than typing it out."
            )

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_every_rung_declared_is_a_ladder_rung(self, provider_name):
        """No adapter invents an eighth rung for itself."""
        ladder = _ladder(provider_name)

        assert set(ladder.rungs) <= LADDER_RUNGS, (
            f"{provider_name} declares {sorted(set(ladder.rungs) - LADDER_RUNGS)} as "
            "rungs. A name the ladder does not carry is either a local spelling "
            "(`spellings=`) or a second-axis variant (`variants=`) — never a rung."
        )

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_rungs_are_stated_in_ladder_order_without_repeats(self, provider_name):
        """A declaration that reorders the ladder reads as a different order."""
        ladder = _ladder(provider_name)
        positions = [LADDER_POSITION[rung] for rung in ladder.rungs]

        assert positions == sorted(positions)
        assert len(set(ladder.rungs)) == len(ladder.rungs)

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_no_accepted_name_is_unexplained(self, provider_name):
        """Every accepted name is a rung, a declared spelling, or a declared variant.

        This is what stops the ladder becoming decorative — a name that slipped
        into a whitelist with no declaration behind it has no answer to "which
        rung is that", which is the state the whole task replaces.
        """
        ladder = _ladder(provider_name)
        explained = set(ladder.rungs) | set(ladder.local_names) | set(ladder.variants)

        assert ladder.accepted <= explained

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_local_spellings_name_a_rung_and_are_not_rungs(self, provider_name):
        """A spelling is a CLI's word for a rung, so it maps to one and is not one."""
        ladder = _ladder(provider_name)

        for local, canonical in ladder.spellings:
            assert canonical in LADDER_RUNGS, (
                f"{provider_name} spells {local!r} for {canonical!r}, which is not "
                "a rung."
            )
            assert local not in LADDER_RUNGS, (
                f"{provider_name} declares the rung {local!r} as a local spelling "
                "of something else."
            )

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_inbound_aliases_are_accepted_but_never_emitted(self, provider_name):
        """An inbound alias is rewritten before argv; it may never come back out."""
        ladder = _ladder(provider_name)

        for local, canonical in ladder.inbound_aliases:
            assert canonical in ladder.rungs, (
                f"{provider_name} rewrites {local!r} to {canonical!r}, a rung it "
                "does not accept — the rewrite would emit a value the CLI refuses."
            )
            assert local not in ladder.accepted, (
                f"{provider_name} lists {local!r} both as an inbound alias and as "
                "an emitted name. An alias is rewritten or it is accepted, not both."
            )
            assert ladder.resolve(local) == canonical

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_off_ladder_accepted_names_are_declared_second_axis_variants(
        self, provider_name
    ):
        """Rule 2, from the other side: an off-ladder name is a mode, not a rung.

        The accepted-set model must let variant names *through* rather than
        rejecting them — but only names the shared table recognises as a second
        axis, so a typo cannot enter the tree disguised as one.
        """
        ladder = _ladder(provider_name)
        off_ladder = ladder.accepted - LADDER_RUNGS - set(ladder.local_names)

        assert off_ladder <= SECOND_AXIS_VARIANTS
        assert set(ladder.variants) <= SECOND_AXIS_VARIANTS

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_variants_are_never_normalized_onto_a_rung(self, provider_name):
        """A second-axis name has no rung, and asking for one answers ``None``."""
        ladder = _ladder(provider_name)

        for variant in ladder.variants:
            assert ladder.normalize(variant) is None
            assert ladder.accepts(variant)
            assert ladder.resolve(variant) == variant


class TestTheLadderFeedsTheExistingSplitter:
    """`split_reasoning_effort` is load-bearing elsewhere and does not move."""

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_every_accepted_name_splits_off_a_model_spec(self, provider_name):
        """The derived whitelist is still exactly what the splitter consumes."""
        ladder = _ladder(provider_name)

        for name in sorted(ladder.accepted_inbound):
            assert split_reasoning_effort(f"a-model:{name}", ladder.accepted_inbound) == (
                "a-model",
                name,
            )

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_an_unaccepted_suffix_stays_part_of_the_model_name(self, provider_name):
        """Unchanged pass-through behaviour: no coercion happens at split time."""
        ladder = _ladder(provider_name)
        spec = "a-model:not-an-effort"

        assert split_reasoning_effort(spec, ladder.accepted_inbound) == (spec, None)

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_a_declining_adapter_splits_nothing(self, provider_name):
        """An adapter with no effort mechanism keeps every colon in the model id."""
        ladder = _ladder(provider_name)
        if not ladder.declined:
            pytest.skip("declares an effort mechanism")

        for rung in REASONING_LADDER:
            spec = f"vendor/a-model:{rung}"
            assert split_reasoning_effort(spec, ladder.accepted_inbound) == (spec, None)


class TestCoercionOnlyFiresWhenTheRawValueIsUnsupported:
    """§6.1's coercion rule, which is a rule about when NOT to coerce."""

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_an_accepted_name_is_returned_untouched(self, provider_name):
        """A value the CLI already takes is never rewritten into another one."""
        ladder = _ladder(provider_name)

        for name in sorted(ladder.accepted):
            assert ladder.resolve(name) == name

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_render_round_trips_through_normalize(self, provider_name):
        """Rung → this CLI's word → back to the same rung, for every rung it has."""
        ladder = _ladder(provider_name)

        for rung in REASONING_LADDER:
            spelled = ladder.render(rung)
            if spelled is None:
                continue
            assert spelled in ladder.accepted
            assert ladder.normalize(spelled) == rung
            assert ladder.resolve(rung) == spelled

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_a_rung_this_cli_lacks_is_refused_not_rounded(self, provider_name):
        """A silent downgrade is the worse failure, so an absent rung is refused.

        Almost no CLI here reports the effort it actually ran at, so a value
        quietly rounded onto a neighbouring rung is invisible in the output.
        """
        ladder = _ladder(provider_name)

        for rung in REASONING_LADDER:
            if ladder.render(rung) is None:
                assert ladder.resolve(rung) is None

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_an_unknown_word_is_refused(self, provider_name):
        ladder = _ladder(provider_name)

        assert ladder.resolve("bogus") is None
        assert ladder.resolve("") is None

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_rung_down_never_selects_an_unadvertised_rung(self, provider_name):
        """Rule 3: an unadvertised rung is reachable only by a user naming it."""
        ladder = _ladder(provider_name)

        for rung in REASONING_LADDER:
            weaker = ladder.rung_down(rung)
            if weaker is None:
                continue
            assert weaker in ladder.rungs
            assert weaker not in UNADVERTISED_RUNGS
            assert LADDER_POSITION[weaker] < LADDER_POSITION[rung]

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_rung_down_from_an_off_ladder_name_is_none(self, provider_name):
        """Rule 2 again: a mode has no neighbour to step down to."""
        ladder = _ladder(provider_name)

        for variant in sorted(SECOND_AXIS_VARIANTS):
            assert ladder.rung_down(variant) is None


class TestTheUnionOverTheRegistry:
    """Whole-registry properties, derived rather than listed."""

    def test_no_registered_provider_invents_a_rung(self):
        union = set()
        for name in _registry_provider_names():
            union |= set(_ladder(name).rungs)

        assert union <= LADDER_RUNGS

    def test_every_rung_is_claimed_by_at_least_one_registered_provider(self):
        """A rung nothing accepts is a rung nothing measured.

        The ladder is a union of measurements, not an aspiration: if this goes
        red the honest fix is to drop the rung, not to add it to an adapter.
        """
        union = set()
        for name in _registry_provider_names():
            union |= set(_ladder(name).rungs)

        assert LADDER_RUNGS - union == set()

    def test_every_off_ladder_name_in_the_tree_is_declared_somewhere(self):
        """Across the registry, only spellings and second-axis variants are off-ladder."""
        for name in _registry_provider_names():
            ladder = _ladder(name)
            off_ladder = ladder.accepted_inbound - LADDER_RUNGS

            assert off_ladder <= set(ladder.local_names) | set(ladder.aliases) | SECOND_AXIS_VARIANTS

    def test_the_two_alias_pairs_run_in_both_directions(self):
        """§6.1's "same pair of names, opposite direction", asserted structurally.

        Derived from the registry rather than from the two adapter names the
        plan happens to mention: whichever adapters they are, an outbound
        spelling and an inbound alias for the same rung must not coexist in one
        declaration, and each must resolve the way its direction says.
        """
        outbound = {}
        inbound = {}
        for name in _registry_provider_names():
            ladder = _ladder(name)
            for local, canonical in ladder.spellings:
                outbound.setdefault((local, canonical), []).append(name)
                # Outbound only where the canonical word is not itself accepted:
                # a CLI that takes both gets the canonical one, and the local
                # spelling survives because a user may have typed it.
                expected = canonical if canonical in ladder.rungs else local
                assert ladder.resolve(canonical) == expected
                assert ladder.resolve(local) == local
            for local, canonical in ladder.inbound_aliases:
                inbound.setdefault((local, canonical), []).append(name)
                assert ladder.resolve(local) == canonical
            overlap = set(ladder.local_names) & set(ladder.aliases)
            assert not overlap, (
                f"{name} declares {sorted(overlap)} as both an emitted spelling and "
                "an inbound-only alias — one declaration cannot run both directions."
            )

        assert outbound, "no adapter declares a local spelling any more"
        assert inbound, "no adapter declares an inbound-only alias any more"


class TestPerModelCaps:
    """§6.1's per-MODEL half: a cap is a fact about a model, never about a provider."""

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_a_cap_narrows_and_never_widens(self, provider_name):
        """Every capped name is one the provider accepts in the first place."""
        ladder = _ladder(provider_name)

        for model, allowed in ladder.model_caps:
            assert set(allowed) <= ladder.accepted, (
                f"{provider_name}'s cap for {model!r} lists "
                f"{sorted(set(allowed) - ladder.accepted)}, which this CLI does not "
                "accept at all. A cap narrows an accepted set; it cannot widen one."
            )
            assert ladder.accepted_for(model) == frozenset(allowed) & ladder.accepted

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_an_uncapped_model_gets_the_whole_accepted_set(self, provider_name):
        ladder = _ladder(provider_name)

        assert ladder.accepted_for("some/uncapped-model") == ladder.accepted
        assert ladder.accepted_for(None) == ladder.accepted

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_a_capped_name_is_refused_for_that_model_only(self, provider_name):
        ladder = _ladder(provider_name)

        for model, allowed in ladder.model_caps:
            for name in sorted(ladder.accepted - frozenset(allowed)):
                assert ladder.refuses_for_model(model, name)
                assert ladder.resolve(name, model) is None
                assert not ladder.accepts(name, model)
                # The cap is about the model, not about the provider.
                assert ladder.accepts(name)

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_a_cap_applies_through_a_spec_that_still_carries_its_suffix(
        self, provider_name
    ):
        """Callers pass the spec they have; the suffix is split off first."""
        ladder = _ladder(provider_name)

        for model, allowed in ladder.model_caps:
            for name in sorted(ladder.accepted):
                assert ladder.accepted_for(f"{model}:{name}") == ladder.accepted_for(model)

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_no_configured_spec_asks_for_a_capped_effort(self, provider_name):
        """The shipped config is graded against the caps, not against a comment.

        This is what "refused in code rather than by a YAML comment" buys: a
        config that re-adds a suffix to a model measured not to survive one goes
        red here instead of burning a slot's whole timeout at run time.
        """
        ladder = _ladder(provider_name)
        offenders = []

        for spec in _configured_specs(provider_name):
            base, effort = split_reasoning_effort(spec, ladder.accepted_inbound)
            if effort is None:
                continue
            if ladder.resolve(effort, base) is None:
                offenders.append(f"  {provider_name}:{spec} — {base} accepts "
                                 f"{sorted(ladder.accepted_for(base))}")

        assert not offenders, (
            "a configured model spec asks for a reasoning effort its model is "
            "declared not to accept:\n" + "\n".join(offenders)
        )

    @pytest.mark.parametrize("provider_name", _registry_provider_names())
    def test_a_declining_adapter_carries_no_effort_shaped_suffix(self, provider_name):
        """The silent failure a decline exists to prevent, checked in the config.

        An adapter with no effort mechanism parses no suffix, so a spec ending in
        an effort-looking word reaches the backend verbatim as part of the model
        id and comes back as "model not found" with nothing pointing at the
        suffix.
        """
        ladder = _ladder(provider_name)
        if not ladder.declined:
            pytest.skip("declares an effort mechanism")

        effort_shaped = LADDER_RUNGS | SECOND_AXIS_VARIANTS | {"off", "ultra"}
        offenders = [
            spec
            for spec in _configured_specs(provider_name)
            if spec.rpartition(":")[0] and spec.rpartition(":")[2] in effort_shaped
        ]

        assert not offenders, (
            f"{provider_name} parses no `:effort` suffix ({ladder.declined}), so "
            f"these specs would reach the backend with the suffix glued to the "
            f"model id: {offenders}"
        )

    def test_the_measured_opencode_variant_hang_is_declared_in_code(self):
        """Regression: the one combination measured to HANG rather than fail.

        `opencode run --variant` on this model writes zero bytes to either
        stream, never reaches session creation, and burns the whole slot timeout,
        where the same model with no suffix answers in about three seconds. It
        lived as a prose comment in `providers.yaml`; M12 moved it into the
        adapter's own declaration.

        This is the one test in the module that names a provider, and it names a
        *model* with it: M0.a forbids an acceptance test standing on a roster of
        adapter names, not a regression test recording a single measurement.
        """
        ladder = _ladder("opencode")
        model = "openrouter/deepseek/deepseek-v4-flash-0731"

        assert ladder.accepted_for(model) == frozenset(), (
            "the deepseek `--variant` hang is no longer declared. If it was "
            "re-measured as fixed, say so where the cap is declared."
        )
        for name in sorted(ladder.accepted):
            assert ladder.resolve(name, model) is None


class TestTheSweepsCannotMissANewAdapter:
    """M0.a, asserted against this module's own source rather than promised.

    Donor §11.2 and every acceptance row under it were written against a fixed
    roster of adapter names the registry has already outgrown. The rule that
    replaced them is that no Group M acceptance test may name a literal adapter
    list or a count — so this module proves it about itself, mechanically,
    instead of relying on nobody pasting one in later.
    """

    @staticmethod
    def _module_tree():
        source = pathlib.Path(__file__).read_text(encoding="utf-8")
        return source, ast.parse(source)

    def test_every_parametrized_sweep_draws_from_the_registry(self):
        """Each ``provider_name`` sweep's argument is the registry helper's call."""
        _, tree = self._module_tree()
        seen = 0

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "parametrize"):
                continue
            argnames = node.args[0]
            assert isinstance(argnames, ast.Constant), ast.dump(argnames)
            assert argnames.value == "provider_name", (
                "this module parametrizes over provider ids and nothing else; "
                f"found {argnames.value!r}"
            )
            argvalues = node.args[1]
            assert isinstance(argvalues, ast.Call), (
                "a parametrized sweep must call the registry helper, not carry a "
                f"literal: {ast.dump(argvalues)}"
            )
            called = argvalues.func
            assert isinstance(called, ast.Name) and called.id == "_registry_provider_names", (
                "every sweep draws its ids from `_registry_provider_names()` so a "
                f"newly registered adapter is picked up automatically: {ast.dump(called)}"
            )
            seen += 1

        assert seen, "no registry-parametrized sweep found — did they all become literals?"

    def test_no_sweep_carries_a_literal_roster(self):
        """No list/tuple/set literal of provider ids appears anywhere in the module."""
        _, tree = self._module_tree()
        known = set(_registry_provider_names())

        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                continue
            values = [
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            hits = known.intersection(values)
            assert len(hits) <= 1, (
                f"a literal roster of adapter ids appears in this module: {sorted(hits)}. "
                "Parametrize over `_registry_provider_names()` instead — a written-down "
                "list is exactly what a new adapter falls out of."
            )

    def test_the_module_states_no_adapter_count(self):
        """No integer literal here equals the number of registered adapters.

        A count is a roster with the names left out: it goes stale the same way
        and it fails the same silent way. ``assert len(names) == <n>`` is the
        shape this catches, and the count it compares against is read from the
        registry rather than written down — including by this test.
        """
        _, tree = self._module_tree()
        registry_size = len(_registry_provider_names())

        offenders = sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and node.value == registry_size
        )

        assert not offenders, (
            f"an integer literal equal to the registered-adapter count appears on "
            f"line(s) {offenders}. The count comes from the registry at run time "
            "or it is not stated at all."
        )
