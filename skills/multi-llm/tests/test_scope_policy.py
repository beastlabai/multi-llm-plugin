"""The shared scope-policy contract (plans/scope-policy-and-report-tags.md §3).

These pin observable rules of :mod:`utils.scope_policy`: vocabulary, the
never-raising normalization, baseline/proposal binding, authority, the gate's
pass-through under ``expand``, and the role- and policy-specific prompt blocks.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import scope_policy as sp
from utils.state_manager import generate_group_id


def _group(desc="Add retries to the upload loop.", **extra):
    group = {
        "theme": "Upload retries",
        "category": "completeness",
        "suggestions": [
            {"title": "Retry uploads", "desc": desc, "type": "addition",
             "reference": "## 3", "importance": "medium"},
        ],
    }
    group.update(extra)
    return group


BASELINE = sp.build_baseline([sp.baseline_source("plan", sha256="ab" * 32, path="plans/x.md")])
BID = BASELINE["baseline_id"]


def _bound(group, relation, reason="because", **record_extra):
    record = {"scope_relation": relation, "scope_reason": reason, **record_extra}
    group = dict(group)
    group[sp.VERDICT_KEY] = sp.make_verdict(record, group, baseline_id=BID)
    return group


class TestVocabulary:
    def test_labels_are_the_three_report_labels(self):
        assert sp.LABELS == {
            "in_scope": "Within scope",
            "scope_extension": "Scope extension",
            "unclear": "Scope unclear",
        }

    def test_default_is_expand(self):
        assert sp.DEFAULT_POLICY == "expand"
        assert sp.effective(None) == "expand"
        assert sp.effective("") == "expand"

    @pytest.mark.parametrize("raw,expected", [
        ("preserve", "preserve"), (" Expand ", "expand"), ("PRESERVE", "preserve"),
        ("", None), (None, None), ("restrict", None), (3, None),
    ])
    def test_normalize_policy(self, raw, expected):
        assert sp.normalize_policy(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("in_scope", "in_scope"), ("in-scope", "in_scope"), ("Scope Extension", "scope_extension"),
        ("extension", "scope_extension"), ("UNCLEAR", "unclear"), ("bogus", "unclear"),
        (None, "unclear"), (5, "unclear"), (["in_scope"], "unclear"),
    ])
    def test_normalize_relation(self, raw, expected):
        assert sp.normalize_relation(raw) == expected


class TestNormalizeScopeFields:
    def test_malformed_never_raises_and_is_unclear(self):
        for record in (None, [], "x", {"scope_relation": {"a": 1}}, {"scope_reason": 7}):
            fields = sp.normalize_scope_fields(record)
            assert fields["scope_relation"] == "unclear"

    def test_definite_relation_without_reason_is_unclear(self):
        assert sp.normalize_scope_fields({"scope_relation": "in_scope"})["scope_relation"] == "unclear"
        assert sp.normalize_scope_fields(
            {"scope_relation": "in_scope", "scope_reason": "completes R1"})["scope_relation"] == "in_scope"

    def test_reason_is_flattened_and_cannot_form_headings_or_checkboxes(self):
        fields = sp.normalize_scope_fields({
            "scope_relation": "scope_extension",
            "scope_reason": "## Heading\n- [x] adds PDF\nexport",
            "scope_reference": "### Plan: Output",
        })
        assert "\n" not in fields["scope_reason"]
        assert not fields["scope_reason"].startswith("#")
        assert "[x]" not in fields["scope_reason"].split(" ")[0]
        assert not fields["scope_reference"].startswith("#")

    def test_reason_is_bounded(self):
        fields = sp.normalize_scope_fields({"scope_relation": "unclear", "scope_reason": "word " * 500})
        assert len(fields["scope_reason"]) <= sp.REASON_LIMIT + 5


class TestBaseline:
    def test_id_ignores_display_paths(self):
        a = sp.build_baseline([sp.baseline_source("plan", sha256="1" * 64, path="a/plan.md")])
        b = sp.build_baseline([sp.baseline_source("plan", sha256="1" * 64, path="/elsewhere/plan.md")])
        assert a["baseline_id"] == b["baseline_id"]

    def test_id_changes_with_content_or_endpoints(self):
        a = sp.build_baseline([sp.baseline_source("plan", sha256="1" * 64)])
        b = sp.build_baseline([sp.baseline_source("plan", sha256="2" * 64)])
        c = sp.build_baseline([sp.baseline_source("commit_brief", sha256="1" * 64,
                                                  endpoints={"head": "abc"})])
        d = sp.build_baseline([sp.baseline_source("commit_brief", sha256="1" * 64,
                                                  endpoints={"head": "abd"})])
        assert a["baseline_id"] != b["baseline_id"]
        assert c["baseline_id"] != d["baseline_id"]

    def test_id_is_order_independent(self):
        s1 = sp.baseline_source("plan", sha256="1" * 64)
        s2 = sp.baseline_source("tasks", sha256="2" * 64)
        assert sp.build_baseline([s1, s2])["baseline_id"] == sp.build_baseline([s2, s1])["baseline_id"]

    def test_unreadable_roles(self):
        record = sp.build_baseline([sp.baseline_source("plan", sha256=None)])
        assert sp.baseline_unreadable_roles(record) == ["plan"]


class TestProposalBinding:
    def test_metadata_only_changes_keep_digest_and_stable_id(self):
        group = _group()
        digest = sp.proposal_digest(group)
        gid = generate_group_id(group)
        tagged = _bound(group, "in_scope")
        tagged["suggestions"] = [dict(tagged["suggestions"][0], scope_relation="in_scope",
                                      scope_reason="x", id="abc", suggestion_hash="h",
                                      source_model="m")]
        tagged["group_hash"] = gid
        tagged["validation_status"] = "valid"
        assert sp.proposal_digest(tagged) == digest
        assert generate_group_id(tagged) == gid

    def test_edit_beyond_the_id_hash_prefix_changes_digest_not_id(self):
        base = "x" * 250
        group = _group(desc=base)
        edited = _group(desc=base + " and also add PDF export")
        assert generate_group_id(group) == generate_group_id(edited)
        assert sp.proposal_digest(group) != sp.proposal_digest(edited)

    def test_removing_a_member_changes_digest(self):
        group = _group()
        group["suggestions"].append({"title": "Also export PDF", "desc": "Add PDF.", "type": "addition"})
        fewer = dict(group, suggestions=group["suggestions"][:1])
        assert sp.proposal_digest(group) != sp.proposal_digest(fewer)


class TestAuthority:
    def test_bound_verdict_is_authoritative(self):
        auth = sp.authority(_bound(_group(), "in_scope"), BID)
        assert auth.authoritative and auth.relation == "in_scope"

    def test_missing_verdict_is_unclear(self):
        auth = sp.authority(_group(), BID)
        assert auth.relation == "unclear" and auth.state == "missing"

    def test_reviewer_tag_alone_is_never_authority(self):
        group = _group()
        group["suggestions"][0].update(scope_relation="in_scope", scope_reason="trust me")
        assert sp.authority(group, BID).relation == "unclear"

    def test_other_baseline_is_unclear(self):
        auth = sp.authority(_bound(_group(), "in_scope"), "0" * 16)
        assert auth.relation == "unclear" and auth.state == "other-baseline"

    def test_edited_description_is_stale(self):
        group = _bound(_group(), "in_scope")
        group["suggestions"] = [dict(group["suggestions"][0], desc="Now add PDF export too.")]
        auth = sp.authority(group, BID)
        assert auth.relation == "unclear" and auth.state == "stale"

    def test_reply_without_scope_fields_is_missing(self):
        group = dict(_group())
        group[sp.VERDICT_KEY] = sp.make_verdict({"status": "valid"}, group, baseline_id=BID)
        assert sp.authority(group, BID).state == "missing"

    def test_selected_remedy_only_when_allowed(self):
        group = dict(_group())
        record = {"scope_relation": "in_scope", "scope_reason": "r", "selected_remedy": "Retry once."}
        assert sp.SELECTED_REMEDY_FIELD not in sp.make_verdict(record, group, baseline_id=BID)
        group[sp.VERDICT_KEY] = sp.make_verdict(record, group, baseline_id=BID, allow_selected_remedy=True)
        auth = sp.authority(group, BID)
        assert auth.authoritative and auth.selected_remedy == "Retry once."


class TestGate:
    @pytest.mark.parametrize("relation", ["in_scope", "scope_extension", "unclear", None])
    def test_expand_is_a_pass_through(self, relation):
        group = _group() if relation is None else _bound(_group(), relation)
        decision = sp.scope_decision(group, "expand", BID)
        assert decision.eligible and decision.outcome == "eligible"
        assert sp.scope_decision(group, None, BID).eligible

    def test_expand_passes_stale_and_other_baseline(self):
        group = _bound(_group(), "scope_extension")
        group["suggestions"] = [dict(group["suggestions"][0], desc="edited")]
        assert sp.scope_decision(group, "expand", "f" * 16).eligible

    def test_preserve_passes_only_bound_in_scope(self):
        assert sp.scope_decision(_bound(_group(), "in_scope"), "preserve", BID).eligible
        for group in (_bound(_group(), "scope_extension"), _bound(_group(), "unclear"), _group()):
            decision = sp.scope_decision(group, "preserve", BID)
            assert not decision.eligible
            assert decision.outcome == "deferred"
            assert decision.reason.startswith(sp.DEFERRED_OUTCOME)

    def test_preserve_missing_explains_recovery(self):
        decision = sp.scope_decision(_group(), "preserve", BID)
        assert "fresh classified review" in decision.reason

    def test_partition(self):
        groups = [_bound(_group(), "in_scope"), _bound(_group("other"), "scope_extension")]
        eligible, deferred = sp.partition(groups, "preserve", BID)
        assert len(eligible) == 1 and len(deferred) == 1
        assert deferred[0][sp.DECISION_KEY]["label"] == "Scope extension"
        eligible, deferred = sp.partition(groups, "expand", BID)
        assert len(eligible) == 2 and not deferred


class TestCombinedRelation:
    def test_any_extension_wins(self):
        assert sp.combined_relation(["in_scope", "scope_extension", "unclear"]) == "scope_extension"

    def test_unknown_member_makes_unclear(self):
        assert sp.combined_relation(["in_scope", "garbage"]) == "unclear"

    def test_all_in_scope(self):
        assert sp.combined_relation(["in_scope", "in_scope"]) == "in_scope"

    def test_empty_is_unclear(self):
        assert sp.combined_relation([]) == "unclear"


def _flat(text):
    """Prompt text with its line wrapping collapsed, for phrase checks."""
    return " ".join(text.split())


class TestPromptGuidance:
    PHASES = ["review-plan", "review-tasks", "code-review", "commits"]

    @pytest.mark.parametrize("phase", PHASES)
    def test_expand_reviewer_is_informational_without_suppression(self, phase):
        text = _flat(sp.reviewer_guidance("expand", phase))
        assert "informational" in text
        assert "do not leave out a finding" in text
        assert "Do not propose new features" not in text
        assert "takes precedence" not in text
        assert "Preserve" not in text

    @pytest.mark.parametrize("phase", PHASES)
    def test_preserve_reviewer_suppresses_but_keeps_coverage(self, phase):
        text = _flat(sp.reviewer_guidance("preserve", phase))
        assert "Do not propose new features" in text
        assert "takes precedence over the broader checklists" in text
        assert "existing scope creep" in text
        assert "report the blocker" in text
        assert "Returning no findings is acceptable" in text

    @pytest.mark.parametrize("phase", PHASES)
    def test_expand_validator_keeps_valid_definition(self, phase):
        text = _flat(sp.validator_guidance("expand", phase))
        assert "informational" in text
        assert "rejected for this run" not in text
        assert "selected_remedy" not in text
        assert "scope_relation" in text

    @pytest.mark.parametrize("phase", PHASES)
    def test_preserve_validator_rejects_independently(self, phase):
        text = _flat(sp.validator_guidance("preserve", phase))
        assert "rejected for this run" in text
        assert "never mark it \"invalid\"" in text
        assert "independently of the reviewers" in text
        assert "selected_remedy" in text

    def test_alternating_renders_do_not_leak(self):
        first = sp.reviewer_guidance("expand", "review-plan")
        sp.reviewer_guidance("preserve", "review-plan")
        assert sp.reviewer_guidance("expand", "review-plan") == first

    def test_phase_baselines_differ(self):
        texts = {sp.baseline_description(p) for p in self.PHASES}
        assert len(texts) == 4
        assert "drift" in sp.baseline_description("code-review")
        assert "focus" in sp.baseline_description("commits")

    def test_revalidation_note(self):
        assert "original review round's" in _flat(sp.validator_guidance(
            "expand", "review-plan", revalidation_baseline_differs=True))

    def test_writer_guidance_is_empty_under_expand(self):
        assert sp.writer_guidance("expand") == ""
        assert sp.writer_guidance(None) == ""
        assert "decline that item" in sp.writer_guidance("preserve")


class TestResolution:
    def test_order(self):
        assert sp.resolve_policy(explicit="preserve", continuation="expand") == ("preserve", "explicit")
        assert sp.resolve_policy(continuation="expand", recorded="preserve") == ("expand", "continuation")
        assert sp.resolve_policy(forwarded="preserve", recorded="expand") == ("preserve", "workflow")
        assert sp.resolve_policy(recorded="preserve") == ("preserve", "review")
        assert sp.resolve_policy() == ("expand", "default")

    def test_invalid_values_are_skipped(self):
        assert sp.resolve_policy(explicit="bogus", recorded="preserve") == ("preserve", "review")


class TestStateRecord:
    def test_review_then_apply_override_survives(self):
        state = {}
        sp.record_review_round(state, "review-plan", "preserve", "explicit", BASELINE)
        sp.record_apply_policy(state, "apply-suggestions", "expand", "explicit")
        policies = sp.round_policies(state, "apply-suggestions")
        assert policies["review_policy"] == "preserve"
        assert policies["apply_policy"] == "expand"
        assert policies["baseline_id"] == BID

    def test_new_round_drops_old_override(self):
        state = {}
        sp.record_review_round(state, "code-review", "preserve", "explicit")
        sp.record_apply_policy(state, "apply-code-fixes", "expand", "explicit")
        sp.record_review_round(state, "review-code", "expand", "default")
        assert sp.round_policies(state, "apply-fixes")["apply_policy"] is None

    def test_baseline_is_never_rebased(self):
        state = {}
        sp.update_round_baseline(state, "review-plan", BASELINE)
        other = sp.build_baseline([sp.baseline_source("plan", sha256="9" * 64)])
        sp.update_round_baseline(state, "review-plan", other)
        assert sp.round_policies(state, "review-plan")["baseline_id"] == BID

    def test_legacy_round_reads_expand(self):
        policies = sp.round_policies({}, "review-tasks")
        assert policies["review_policy"] == "expand"
        assert policies["review_policy_source"] == "legacy"


class TestReportVocabulary:
    def test_markdown_tag(self):
        tag = sp.markdown_tag("scope_extension", "Adds `PDF`\nexport", provisional=True)
        assert tag.startswith("`Scope extension` (reviewer, provisional) — ")
        assert "\n" not in tag and "`PDF`" not in tag

    def test_policy_line_names_review_policy_when_overridden(self):
        line = sp.policy_line("preserve", "expand")
        assert "Scope policy: expand" in line and "requested under preserve" in line
        assert "not solicited" in line

    def test_counts_use_distinct_groups(self):
        g = _bound(_group(), "scope_extension")
        g["group_hash"] = "h1"
        counts = sp.count_relations([g, dict(g)], BID)
        assert counts["scope_extension"] == 1


class TestStampedRecords:
    def test_stamp_discards_model_echoes_and_binds(self):
        group = _group()
        record = {"group_hash": "h", "status": "valid", "scope_relation": "in_scope",
                  "scope_reason": "r", "scope_baseline_id": "forged", "scope_proposal_digest": "forged"}
        stamped = sp.stamp_validation_record(record, group, baseline_id=BID)
        assert stamped["scope_baseline_id"] == BID
        assert stamped["scope_proposal_digest"] == sp.proposal_digest(group)
        assert stamped["status"] == "valid"

    def test_apply_time_join_uses_the_stamped_digest(self):
        group = _group()
        stamped = sp.stamp_validation_record(
            {"scope_relation": "in_scope", "scope_reason": "r"}, group, baseline_id=BID)
        later = dict(group, suggestions=[dict(group["suggestions"][0], desc="now also add PDF export")])
        sp.attach_verdict(later, stamped, hash_matched=True)
        assert sp.authority(later, BID).state == "stale"
        same = dict(group)
        sp.attach_verdict(same, stamped, hash_matched=True)
        assert sp.authority(same, BID).authoritative

    def test_index_fallback_never_transfers_authority(self):
        group = _group()
        stamped = sp.stamp_validation_record(
            {"scope_relation": "in_scope", "scope_reason": "r"}, group, baseline_id=BID)
        target = dict(group, **{sp.VERDICT_KEY: {"relation": "in_scope"}})
        sp.attach_verdict(target, stamped, hash_matched=False)
        assert sp.VERDICT_KEY not in target
        assert sp.scope_decision(target, "preserve", BID).outcome == "deferred"

    def test_unstamped_legacy_record_has_no_authority(self):
        assert sp.verdict_from_record({"status": "valid", "scope_relation": "in_scope",
                                       "scope_reason": "r"}) is None

    def test_selected_remedy_round_trip(self):
        group = _group()
        stamped = sp.stamp_validation_record(
            {"scope_relation": "in_scope", "scope_reason": "r", "selected_remedy": "Retry once."},
            group, baseline_id=BID, allow_selected_remedy=True)
        target = dict(group)
        sp.attach_verdict(target, stamped, hash_matched=True)
        auth = sp.authority(target, BID)
        assert auth.authoritative and auth.selected_remedy == "Retry once."
        expand = sp.stamp_validation_record(
            {"scope_relation": "in_scope", "scope_reason": "r", "selected_remedy": "Retry once."},
            group, baseline_id=BID)
        assert "selected_remedy" not in expand
