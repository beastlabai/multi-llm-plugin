"""Consolidation reads the group record ``grouped.json`` really holds.

``SuggestionGroup.to_dict`` writes no group-level ``reference``,
``importance`` or ``type``, and a suggestion's text is ``desc``. The
consolidation step read all four from fields that do not exist, so every
singleton came out LOW (a HIGH finding shown as LOW and sorted last), every
group record told the model MEDIUM, and the content measure saw only the
theme. Its own 12-char id also gave same-content groups one id, so one of
them vanished from the consolidated output.

Every test here starts from groups the plugin writes: ``group_similar_suggestions``
-> ``to_dict`` -> ``stamp_stable_ids`` -> ``grouped.json``, then the real
``consolidate_mode`` / ``reaggregate_consolidation_mode`` legs with the batch
reply a subagent would write (docs/development/review-results.md,
"Consolidation batches").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

import review_plan_orchestrator as rpo  # noqa: E402
from utils import consolidation as cons  # noqa: E402
from utils.html_report_generator import CONSOLIDATED_REPORT_DATA  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.report_markdown import checkbox_line, MARK_VALID, SKIP_GROUP  # noqa: E402
from utils.state_manager import save_groups_payload, stamp_stable_ids  # noqa: E402
from utils.suggestion_processor import group_similar_suggestions  # noqa: E402

UNKNOWN = "<img src=x onerror=alert(1)>"


def _suggestion(title: str, desc: str, importance: str, model: str) -> Dict[str, Any]:
    return {
        "title": title, "desc": desc, "importance": importance,
        "reference": "## Task 1: Uploads", "type": "modification",
        "source_model": model,
    }


SUGGESTIONS = [
    _suggestion("Retry the upload call on transient 503 responses",
                "The upload call has no retry, so a transient 503 loses the file.",
                "HIGH", "model-a"),
    _suggestion("Document which directory the config file is read from",
                "The plan never says where the configuration lives on disk.",
                "LOW", "model-b"),
    _suggestion("Rename the cache flag to match the CLI spelling",
                "The flag is called --no-cache in one place and --nocache in another.",
                "MEDIUM", "model-c"),
]


def _production_groups(suggestions=SUGGESTIONS) -> List[Dict[str, Any]]:
    groups = [g.to_dict() for g in group_similar_suggestions(
        [dict(s) for s in suggestions])]
    stamp_stable_ids(groups)
    return groups


def _by_level(groups) -> Dict[str, int]:
    """Group index of the group holding each level's suggestion."""
    return {g["suggestions"][0]["importance"]: i for i, g in enumerate(groups)}


def _phase(tmp_path: Path, groups, statuses=None) -> tuple:
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\n## Task 1: Uploads\nUpload files.\n", encoding="utf-8")
    phase_dir = output_dir_path(plan) / "review-plan"
    phase_dir.mkdir(parents=True)
    (phase_dir / "grouped.json").write_text(
        json.dumps(save_groups_payload(groups), indent=2), encoding="utf-8")
    statuses = statuses or ["valid"] * len(groups)
    (phase_dir / "validation.json").write_text(json.dumps({"groups": [
        {"group_index": i, "group_hash": g["group_hash"], "status": status,
         "reason": ""}
        for i, (g, status) in enumerate(zip(groups, statuses))
    ]}), encoding="utf-8")
    return plan, phase_dir


def _prepare(plan: Path, phase_dir: Path) -> Dict[str, Any]:
    rpo.consolidate_mode(str(phase_dir), str(plan))
    return json.loads((phase_dir / "consolidation_tasks.json").read_text(encoding="utf-8"))


def _reply(phase_dir: Path, batch: Dict[str, Any], clusters, singletons) -> None:
    """The batch reply a subagent writes, copying ids and indices from its input."""
    ids = dict(zip(batch["group_indices"], batch["group_ids"]))
    (phase_dir / f"consolidation_batch_{batch['batch_index']}.json").write_text(json.dumps({
        "verification": "passed",
        "clusters": [{
            "title": title, "description": f"{title}.", "importance": level,
            "type": "modification",
            "underlying_group_ids": [ids[i] for i in members],
            "underlying_group_indices": list(members), "reasoning": "Related.",
        } for title, level, members in clusters],
        "singletons": [{"group_id": ids[i], "group_index": i, "reasoning": "Alone."}
                       for i in singletons],
    }), encoding="utf-8")


def _outputs(plan: Path, phase_dir: Path):
    rpo.reaggregate_consolidation_mode(str(phase_dir), str(plan), "plan", [])
    consolidated = json.loads((phase_dir / "consolidated.json").read_text(encoding="utf-8"))
    report = (phase_dir / "consolidated-report.md").read_text(encoding="utf-8")
    page = json.loads((phase_dir / CONSOLIDATED_REPORT_DATA).read_text(encoding="utf-8"))
    return consolidated["consolidated_groups"], report, page["consolidatedGroups"]


def _md_sections(report: str) -> List[str]:
    return report.split("\n## CG")[1:]


class TestTheGroupShapeIsWhatTheGrouperWrites:

    def test_no_group_carries_the_fields_consolidation_used_to_read(self):
        groups = _production_groups()
        assert len(groups) == 3
        for group in groups:
            assert not {"reference", "importance", "type"} & set(group)
            for suggestion in group["suggestions"]:
                assert "desc" in suggestion and "description" not in suggestion


class TestAHighGroupStaysHigh:

    def test_in_the_prompt_records(self, tmp_path):
        groups = _production_groups()
        level = _by_level(groups)
        plan, phase_dir = _phase(tmp_path, groups)
        document = _prepare(plan, phase_dir)
        (batch,) = document["batches"]
        records = cons.build_consolidation_group_records(
            groups, batch["group_indices"], cons.consolidation_group_ids(groups))
        assert {r["index"]: r["importance"] for r in records} == {
            level["HIGH"]: "HIGH", level["MEDIUM"]: "MEDIUM", level["LOW"]: "LOW"}
        assert '"importance": "HIGH"' in batch["prompt"]

    def test_as_a_singleton_in_consolidated_json_and_both_reports(self, tmp_path):
        groups = _production_groups()
        level = _by_level(groups)
        plan, phase_dir = _phase(tmp_path, groups)
        (batch,) = _prepare(plan, phase_dir)["batches"]
        _reply(phase_dir, batch,
               clusters=[("Config and flag naming", "LOW", (level["LOW"], level["MEDIUM"]))],
               singletons=[level["HIGH"]])
        entries, report, page = _outputs(plan, phase_dir)

        high = [e for e in entries if e["underlying_group_indices"] == [level["HIGH"]]]
        assert [e["importance"] for e in high] == ["HIGH"]
        # Sorted first, as CG1, on both pages.
        assert page[0]["displayIndex"] == 1 and page[0]["isSingleton"] is True
        assert [u["groupIndex"] for u in page[0]["underlyingGroups"]] == [level["HIGH"] + 1]
        assert page[0]["importance"] == "HIGH"
        assert "**Importance:** HIGH" in _md_sections(report)[0]
        # The cluster the model called LOW holds a MEDIUM group.
        cluster = [e for e in entries if not e["is_singleton"]]
        assert [e["importance"] for e in cluster] == ["MEDIUM"]

    def test_inside_a_cluster_the_model_called_medium(self, tmp_path):
        groups = _production_groups()
        level = _by_level(groups)
        plan, phase_dir = _phase(tmp_path, groups)
        (batch,) = _prepare(plan, phase_dir)["batches"]
        _reply(phase_dir, batch,
               clusters=[("Uploads and config", "MEDIUM", (level["HIGH"], level["LOW"]))],
               singletons=[level["MEDIUM"]])
        entries, report, page = _outputs(plan, phase_dir)
        cluster = [e for e in entries if not e["is_singleton"]]
        assert [e["importance"] for e in cluster] == ["HIGH"]
        assert page[0]["isSingleton"] is False and page[0]["importance"] == "HIGH"
        assert "**Importance:** HIGH" in _md_sections(report)[0]

    def test_the_singleton_level_is_its_members_highest_in_either_case(self):
        groups = _production_groups([
            _suggestion("Retry the upload call", "No retry.", "low", "model-a"),
            _suggestion("Retry the upload call", "No retry at all.", "high", "model-b"),
        ])
        assert len(groups) == 1
        entry = cons._make_singleton_consolidated("abc", 0, groups[0])
        assert entry["importance"] == "HIGH"


class TestTheSuggestionTextIsDesc:

    def test_the_content_measure_counts_it(self):
        (group,) = _production_groups(SUGGESTIONS[:1])
        assert cons._measure_group_content_length(group) == (
            len(group["theme"]) + len(SUGGESTIONS[0]["desc"]))

    def test_an_oversized_solo_group_is_truncated_in_its_prompt(self):
        long_text = "word " * 400
        (group,) = _production_groups([
            _suggestion("Retry the upload call", long_text, "HIGH", "model-a")])
        ids = cons.consolidation_group_ids([group])
        (record,) = cons.build_consolidation_group_records(
            [group], [0], ids, char_budget=500)
        text = record["suggestions"][0]["description"]
        assert "[...truncated...]" in text and len(text) < len(long_text)
        # The group itself is untouched.
        assert group["suggestions"][0]["desc"] == long_text

    def test_the_helper_splits_on_content_but_batches_are_cut_by_count(self, tmp_path):
        """The budget reads ``desc`` now, so it would split; batch formation
        does not apply it (``CONSOLIDATION_BATCH_CHAR_BUDGET``), so batches
        stay as every recorded consolidation formed them."""
        big = "x" * (cons.CONSOLIDATION_CHAR_BUDGET // 2 + 100)
        groups = [{
            "theme": f"Theme {i}", "models": ["m"], "category": "modification",
            "suggestions": [{"title": f"T{i}", "desc": big, "importance": "HIGH"}],
        } for i in range(4)]
        stamp_stable_ids(groups)
        split = cons._split_into_budget_batches(
            [0, 1, 2, 3], groups, 4, cons.CONSOLIDATION_CHAR_BUDGET)
        assert split == [[0], [1], [2], [3]]

        assert cons.CONSOLIDATION_BATCH_CHAR_BUDGET is None
        result = cons.prepare_consolidation_tasks(
            groups, {"_uncategorized": [0, 1, 2, 3]}, str(tmp_path), "plan.md")
        assert [b["group_indices"] for b in result["batches"]] == [[0, 1, 2, 3]]


def _assert_ids_and_indices_agree(entries, groups) -> None:
    """Each entry's indices are the groups its ids name (the decision join
    reads the ids, the pages the indices), and no group is shown twice."""
    ids = cons.consolidation_group_ids(groups)
    for entry in entries:
        assert [ids[i] for i in entry["underlying_group_indices"]] == entry[
            "underlying_group_ids"]
        assert entry["is_singleton"] is (len(entry["underlying_group_ids"]) == 1)
    shown = [i for e in entries for i in e["underlying_group_indices"]]
    assert len(shown) == len(set(shown))


def _copies(n: int = 2, others: int = 1) -> List[Dict[str, Any]]:
    """*n* same-content groups (one theme, one title), then *others* other
    groups."""
    copy = {
        "theme": "Retry the upload call", "models": ["model-a"],
        "category": "modification",
        "suggestions": [{"title": "Retry the upload call", "desc": "No retry.",
                         "importance": "HIGH", "type": "modification",
                         "reference": "## Task 1: Uploads", "source_model": "model-a"}],
    }
    other = {
        "theme": "Document the config path", "models": ["model-b"],
        "category": "clarification",
        "suggestions": [{"title": "Document the config path", "desc": "Unstated.",
                         "importance": "LOW", "type": "clarification",
                         "reference": "## Task 2", "source_model": "model-b"}],
    }
    extra = []
    for k in range(1, others):
        more = json.loads(json.dumps(other))
        more["theme"] = more["suggestions"][0]["title"] = f"Name the log file {k}"
        extra.append(more)
    groups = [json.loads(json.dumps(copy)) for _ in range(n)] + [other][:others] + extra
    stamp_stable_ids(groups)
    return groups


def _plain(groups, indices) -> List[str]:
    return [cons.generate_group_id(groups[i]) for i in indices]


def _tasks(groups, *batches, ids=None) -> Dict[str, Any]:
    """A task file whose batches hold *batches* (lists of group indices),
    listed under *ids* (default: the ids the plugin assigns now)."""
    ids = ids or cons.consolidation_group_ids(groups)
    return {
        "batches": [{"batch_index": n, "section_key": "_uncategorized",
                     "group_indices": list(indices),
                     "group_ids": [ids[i] for i in indices]}
                    for n, indices in enumerate(batches)],
        "singleton_sections": {},
    }


def _write_reply(phase_dir: Path, index: int, clusters=(), singletons=(), **extra):
    """A batch reply: *clusters* are (ids, indices) pairs, *singletons* ids."""
    (phase_dir / f"consolidation_batch_{index}.json").write_text(json.dumps({
        "clusters": [{"title": f"Cluster {n}", "description": "d",
                      "importance": "LOW", "type": "clarification",
                      "underlying_group_ids": list(ids),
                      "underlying_group_indices": list(indices), "reasoning": ""}
                     for n, (ids, indices) in enumerate(clusters)],
        "singletons": [{"group_id": gid, "group_index": 0, "reasoning": ""}
                       for gid in singletons],
        **extra,
    }), encoding="utf-8")


def _shown(entries) -> List[List[int]]:
    return sorted(e["underlying_group_indices"] for e in entries)


class TestSameContentGroupsStayApart:

    def test_each_gets_its_own_id_and_ordinary_ids_do_not_move(self):
        groups = _copies(3)
        ids = cons.consolidation_group_ids(groups)
        assert len(set(ids)) == 4
        assert all(len(i) == 12 and int(i, 16) >= 0 for i in ids)
        # The first copy and the ordinary group keep the plain content id.
        assert ids[0] == cons.generate_group_id(groups[0])
        assert ids[3] == cons.generate_group_id(groups[3])
        assert cons.generate_group_id(groups[1]) == ids[0]
        # Deterministic: a re-read of the same groups gives the same ids.
        assert cons.consolidation_group_ids(json.loads(json.dumps(groups))) == ids

    def test_two_copies_are_two_entries(self, tmp_path):
        groups = _copies(2)
        plan, phase_dir = _phase(tmp_path, groups)
        (batch,) = _prepare(plan, phase_dir)["batches"]
        assert len(set(batch["group_ids"])) == 3
        # Two copies merge into one cluster only when the model says so; here
        # it keeps them apart, and the third group joins one of them.
        _reply(phase_dir, batch, clusters=[("Upload and config", "HIGH", (1, 2))],
               singletons=[0])
        entries, _report, _page = _outputs(plan, phase_dir)
        covered = sorted(i for e in entries for i in e["underlying_group_indices"])
        assert covered == [0, 1, 2]
        singles = [e for e in entries if e["is_singleton"]]
        assert [e["underlying_group_indices"] for e in singles] == [[0]]
        assert len({e["consolidated_id"] for e in entries}) == len(entries) == 2

    def test_all_singletons_keep_every_copy(self, tmp_path):
        groups = _copies(2)
        ids = cons.consolidation_group_ids(groups)
        tasks = {
            "batches": [{"batch_index": 0, "section_key": "_uncategorized",
                         "group_indices": [0, 1, 2], "group_ids": ids}],
            "singleton_sections": {},
        }
        (tmp_path / "consolidation_batch_0.json").write_text(json.dumps({
            "clusters": [],
            "singletons": [{"group_id": gid, "group_index": i, "reasoning": ""}
                           for i, gid in enumerate(ids)],
        }), encoding="utf-8")
        entries, _failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        assert sorted(e["underlying_group_indices"][0] for e in entries) == [0, 1, 2]

    def test_a_task_file_from_before_still_covers_every_copy(self, tmp_path):
        """An in-flight round prepared before copies had ids of their own
        listed both under the first one's; the merge still emits both."""
        groups = _copies(2)
        plain = cons.generate_group_id(groups[0])
        other = cons.generate_group_id(groups[2])
        tasks = {
            "batches": [{"batch_index": 0, "section_key": "_uncategorized",
                         "group_indices": [0, 1, 2],
                         "group_ids": [plain, plain, other]}],
            "singleton_sections": {},
        }
        (tmp_path / "consolidation_batch_0.json").write_text(json.dumps({
            "clusters": [],
            "singletons": [{"group_id": plain, "group_index": 1, "reasoning": ""},
                           {"group_id": other, "group_index": 2, "reasoning": ""}],
        }), encoding="utf-8")
        entries, _failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        covered = sorted(i for e in entries for i in e["underlying_group_indices"])
        assert len(entries) == 3 and len(set(covered)) == 3

    def test_a_cluster_from_such_a_task_file_holds_every_copy_listed_under_its_id(
        self, tmp_path,
    ):
        """The reply's index cannot say which copy it meant (a reply's
        indices are never read), so the id names every copy its batch was
        given under it, as a decision on it covered them then."""
        groups = _copies(2)
        plain = cons.generate_group_id(groups[0])
        other = cons.generate_group_id(groups[2])
        tasks = {
            "batches": [{"batch_index": 0, "section_key": "_uncategorized",
                         "group_indices": [0, 1, 2],
                         "group_ids": [plain, plain, other]}],
            "singleton_sections": {},
        }
        (tmp_path / "consolidation_batch_0.json").write_text(json.dumps({
            "clusters": [{"title": "Uploads and config", "description": "d",
                          "importance": "LOW", "type": "modification",
                          "underlying_group_ids": [plain, other],
                          "underlying_group_indices": [1, 2], "reasoning": ""}],
            "singletons": [],
        }), encoding="utf-8")
        entries, _failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        covered = sorted(i for e in entries for i in e["underlying_group_indices"])
        assert covered == [0, 1, 2]
        (cluster,) = [e for e in entries if not e["is_singleton"]]
        assert cluster["underlying_group_indices"] == [0, 1, 2]
        assert cluster["importance"] == "HIGH"
        _assert_ids_and_indices_agree(entries, groups)

    def _decide(self, tmp_path, groups, entries, skipped_cid, marked=True):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        grouped = phase_dir / "grouped.json"
        grouped.write_text(json.dumps(save_groups_payload(groups)), encoding="utf-8")
        metadata = {
            "schema_version": "1.0", "total_original_groups": 3,
            "total_consolidated": len(entries), "merged_count": 0,
            "singleton_count": len(entries), "consolidation_ratio": 1.0,
            "timestamp": "2026-11-08T00:00:00",
            "plan_hash": cons._compute_file_hash(str(plan)),
            "grouped_hash": cons._compute_file_hash(str(grouped)),
            "plan_hash_algorithm": "sha256", "skipped_report": False,
            "sections_processed": 1,
        }
        payload = {"consolidated_groups": entries, "metadata": metadata}
        if marked:
            payload["group_id_format"] = cons.GROUP_ID_FORMAT
        (phase_dir / "consolidated.json").write_text(json.dumps(payload),
                                                     encoding="utf-8")
        (phase_dir / "consolidated-report.md").write_text("\n".join(
            [f"## CG{i + 1} [{e['consolidated_id']}]: {e['title']}\n"
             + checkbox_line(SKIP_GROUP).replace("[ ]", "[x]" if e["consolidated_id"]
                                                 == skipped_cid else "[ ]")
             for i, e in enumerate(entries)]), encoding="utf-8")
        skipped, _overrides = cons.load_merged_suggestions(
            str(phase_dir), groups, str(plan))
        return skipped

    @staticmethod
    def _entry(gid, idx):
        return {
            "consolidated_id": gid, "display_index": 1, "title": "T",
            "description": "", "importance": "HIGH", "reference": "",
            "type": "modification", "underlying_group_indices": [idx],
            "underlying_group_ids": [gid], "model_count": 1,
            "original_suggestion_count": 1, "is_singleton": True, "reasoning": "",
        }

    def test_a_decision_on_one_copy_decides_only_that_copy(self, tmp_path):
        groups = _copies(2)
        ids = cons.consolidation_group_ids(groups)
        entries = [self._entry(gid, i) for i, gid in enumerate(ids)]
        assert self._decide(tmp_path, groups, entries, ids[0]) == {0}

    def test_a_file_from_before_decides_every_copy_as_it_did(self, tmp_path):
        groups = _copies(2)
        plain = cons.generate_group_id(groups[0])
        other = cons.generate_group_id(groups[2])
        entries = [self._entry(plain, 1), self._entry(other, 2)]
        assert self._decide(tmp_path, groups, entries, plain, marked=False) == {0, 1}

    def test_a_marked_file_naming_only_the_first_copy_decides_only_it(self, tmp_path):
        """The shape a file from before has (no later copy's id), but marked:
        its later copy was left out of the consolidation, not filed under
        the first one's id."""
        groups = _copies(2)
        plain = cons.generate_group_id(groups[0])
        other = cons.generate_group_id(groups[2])
        entries = [self._entry(plain, 0), self._entry(other, 2)]
        assert self._decide(tmp_path, groups, entries, plain) == {0}


def _tick(report: Path, cid: str, *labels: str) -> None:
    """Tick *labels* in the consolidated report's block for *cid* only."""
    parts = report.read_text(encoding="utf-8").split("\n## CG")
    (n,) = [n for n, part in enumerate(parts)
            if n and f"[{cid}]:" in part.split("\n", 1)[0]]
    for label in labels:
        assert checkbox_line(label) in parts[n]
        parts[n] = parts[n].replace(
            checkbox_line(label), checkbox_line(label).replace("[ ]", "[x]"), 1)
    report.write_text("\n## CG".join(parts), encoding="utf-8")


class TestADecisionReachesOnlyTheGroupsItsEntryShows:
    """Same-content groups with different validation statuses: the excluded
    copy is in no entry, so no decision may reach it. A new file has the
    shape of a file from before (no later copy's id) when the later copy is
    the one left out; its ``group_id_format`` tells the two apart."""

    @pytest.mark.parametrize("excluded", [1, 0])
    def test_skip_and_mark_valid_reach_only_the_kept_copy(self, tmp_path, excluded):
        groups = _copies(2, others=2)
        kept = 1 - excluded
        statuses = ["valid", "valid", "needs-human-decision", "valid"]
        statuses[excluded] = "invalid"
        plan, phase_dir = _phase(tmp_path, groups, statuses)
        (batch,) = _prepare(plan, phase_dir)["batches"]
        assert sorted(batch["group_indices"]) == sorted([kept, 2, 3])
        _reply(phase_dir, batch, clusters=[("Config and log", "LOW", (2, 3))],
               singletons=[kept])
        entries, _report, _page = _outputs(plan, phase_dir)
        assert _shown(entries) == sorted([[2, 3], [kept]])
        written = json.loads((phase_dir / "consolidated.json").read_text(encoding="utf-8"))
        assert written["group_id_format"] == cons.GROUP_ID_FORMAT

        (entry,) = [e for e in entries if e["underlying_group_indices"] == [kept]]
        _tick(phase_dir / "consolidated-report.md", entry["consolidated_id"],
              SKIP_GROUP, MARK_VALID)
        skipped, overrides = cons.load_merged_suggestions(
            str(phase_dir), groups, str(plan))
        assert skipped == {kept}
        assert overrides == {kept: "valid"}


class TestAnOldTaskFileShowsOnlyTheGroupsItAssigned:
    """A task file written before copies had ids of their own listed each
    copy under the plain id; its ``group_indices`` say which copy a batch was
    given, and a reply to that batch is read through them."""

    @pytest.mark.parametrize("assigned", [1, 0])
    def test_the_copy_it_did_not_assign_stays_out(self, tmp_path, assigned):
        groups = _copies(2)
        tasks = _tasks(groups, [assigned, 2], ids=_plain(groups, range(3)))
        _write_reply(tmp_path, 0, singletons=_plain(groups, [assigned, 2]))
        entries, _failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        assert _shown(entries) == sorted([[2], [assigned]])
        _assert_ids_and_indices_agree(entries, groups)

    def test_a_cluster_naming_the_plain_id_holds_the_assigned_copy(self, tmp_path):
        groups = _copies(2)
        tasks = _tasks(groups, [1, 2], ids=_plain(groups, range(3)))
        _write_reply(tmp_path, 0, clusters=[(_plain(groups, [1, 2]), [1, 2])])
        entries, _failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        assert _shown(entries) == [[1, 2]]
        _assert_ids_and_indices_agree(entries, groups)

    def test_a_failed_batch_shows_only_its_own_groups(self, tmp_path):
        groups = _copies(2)
        tasks = _tasks(groups, [1, 2], ids=_plain(groups, range(3)))
        entries, failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        assert failures["batches"] == [0]
        assert _shown(entries) == [[1], [2]]

    def test_copies_in_two_batches_each_answer_for_their_own(self, tmp_path):
        groups = _copies(2, others=2)
        tasks = _tasks(groups, [0, 2], [1, 3], ids=_plain(groups, range(4)))
        _write_reply(tmp_path, 0, singletons=_plain(groups, [0, 2]))
        _write_reply(tmp_path, 1, clusters=[(_plain(groups, [1, 3]), [1, 3])])
        entries, _failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        assert _shown(entries) == [[0], [1, 3], [2]]
        _assert_ids_and_indices_agree(entries, groups)


class TestAReplyNamesOnlyItsOwnBatchsGroups:
    """A reply's ids are checked against the batch the task file gave it
    before any is claimed, and every index comes from an accepted id: a
    guessed id, a group the batch was not given, or a wrong index never
    changes which groups an entry shows (REVIEW-13)."""

    @staticmethod
    def _merge(tmp_path, groups, tasks):
        entries, failures = cons.merge_consolidation_results(str(tmp_path), tasks, groups)
        _assert_ids_and_indices_agree(entries, groups)
        return entries, failures

    def test_an_unknown_id_is_dropped_with_its_index(self, tmp_path):
        groups = _copies(1, others=3)
        ids = cons.consolidation_group_ids(groups)
        _write_reply(tmp_path, 0, clusters=[([ids[0], "ffffffffffff"], [0, 1])])
        entries, _ = self._merge(tmp_path, groups, _tasks(groups, [0, 1]))
        # What is left of the cluster is one group; the other is still shown.
        assert _shown(entries) == [[0], [1]]
        assert all(e["is_singleton"] for e in entries)

    def test_a_group_no_batch_was_given_stays_out(self, tmp_path):
        groups = _copies(1, others=3)  # group 2 and 3: excluded by validation
        ids = cons.consolidation_group_ids(groups)
        _write_reply(tmp_path, 0, clusters=[([ids[0], ids[2]], [0, 2])],
                     singletons=[ids[1], ids[3]])
        entries, _ = self._merge(tmp_path, groups, _tasks(groups, [0, 1]))
        assert _shown(entries) == [[0], [1]]

    def test_another_batchs_group_stays_with_its_own_batch(self, tmp_path):
        groups = _copies(1, others=3)
        ids = cons.consolidation_group_ids(groups)
        _write_reply(tmp_path, 0, clusters=[([ids[0], ids[2]], [0, 2])],
                     singletons=[ids[1]])
        _write_reply(tmp_path, 1, clusters=[([ids[2], ids[3]], [2, 3])])
        entries, _ = self._merge(tmp_path, groups, _tasks(groups, [0, 1], [2, 3]))
        assert _shown(entries) == [[0], [1], [2, 3]]

    def test_wrong_indices_are_not_read(self, tmp_path):
        groups = _copies(1, others=3)
        ids = cons.consolidation_group_ids(groups)
        _write_reply(tmp_path, 0, clusters=[([ids[0], ids[1]], [3, 2, 1])])
        entries, _ = self._merge(tmp_path, groups, _tasks(groups, [0, 1]))
        assert _shown(entries) == [[0, 1]]

    def test_an_id_that_is_not_a_string_is_dropped(self, tmp_path):
        groups = _copies(1, others=3)
        ids = cons.consolidation_group_ids(groups)
        _write_reply(tmp_path, 0,
                     clusters=[([ids[0], [ids[1]], {"id": ids[1]}, 7, None], [0, 1])],
                     singletons=[[ids[1]]])
        entries, _ = self._merge(tmp_path, groups, _tasks(groups, [0, 1]))
        assert _shown(entries) == [[0], [1]]

    @pytest.mark.parametrize("named", [lambda ids: [ids[2]], lambda ids: [[ids[2]]], lambda ids: 7])
    def test_an_error_reply_shows_its_own_batchs_groups(self, tmp_path, named):
        groups = _copies(1, others=3)
        ids = cons.consolidation_group_ids(groups)
        (tmp_path / "consolidation_batch_0.json").write_text(json.dumps(
            {"error": "refused", "groups": named(ids)}), encoding="utf-8")
        entries, failures = self._merge(tmp_path, groups, _tasks(groups, [0, 1]))
        assert failures["batches"] == [0]
        assert _shown(entries) == [[0], [1]]


class TestTheConsolidatedPageShowsWhatPythonSends:

    def test_an_unknown_importance_reads_medium(self, tmp_path):
        groups = _production_groups(SUGGESTIONS[:1])
        groups[0]["suggestions"][0]["importance"] = UNKNOWN
        entry = cons._make_singleton_consolidated(
            cons.consolidation_group_ids(groups)[0], 0, groups[0])
        entry.update(consolidated_id=cons.consolidation_group_ids(groups)[0],
                     display_index=1, importance=UNKNOWN)
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        cons.generate_consolidated_html([entry], groups, str(phase_dir), str(plan),
                                        ["model-a"])
        (cg,) = json.loads((phase_dir / CONSOLIDATED_REPORT_DATA).read_text(
            encoding="utf-8"))["consolidatedGroups"]
        assert cg["importance"] == "MEDIUM"
        assert [s["importance"] for s in cg["underlyingGroups"][0]["suggestions"]] == [
            "MEDIUM"]
        report = cons.generate_consolidated_report([entry], groups, str(phase_dir), "plan")
        assert "**Importance:** MEDIUM" in Path(report).read_text(encoding="utf-8")

    @pytest.mark.parametrize("level", [7, None, {"x": 1}])
    def test_a_non_string_importance_does_not_break_the_page(self, tmp_path, level):
        groups = _production_groups(SUGGESTIONS[:1])
        groups[0]["suggestions"][0]["importance"] = level
        entry = cons._make_singleton_consolidated(
            cons.consolidation_group_ids(groups)[0], 0, groups[0])
        entry.update(consolidated_id=cons.consolidation_group_ids(groups)[0],
                     display_index=1)
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        cons.generate_consolidated_html([entry], groups, str(phase_dir), str(plan),
                                        ["model-a"])
        (cg,) = json.loads((phase_dir / CONSOLIDATED_REPORT_DATA).read_text(
            encoding="utf-8"))["consolidatedGroups"]
        assert cg["importance"] == "MEDIUM"
