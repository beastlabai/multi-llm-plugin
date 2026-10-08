"""A validation reply is one object, printed once (fix round 26).

Live run 11, Claude Code review-plan, validation batch 5: the Sonnet
subagent printed its object with the two groups' hashes swapped, then "Wait,
the hashes above are swapped relative to the groups; corrected reply:" and a
second, correct object. The host wrote the reply to the batch file,
`util check-reply` said "reply is not valid JSON (Extra data at line 3
column 1)", and one re-ask produced the single object.

Decided: no permissive parse. The draft's two hashes covered the batch
exactly, so a parse that keeps "the one object that validates" would have
found two; telling them apart needs the echoed `group_index`, which the merge
treats as advisory because correct replies mis-echo it (run 11 itself: three
verdicts on OpenCode). The host's runbook already says to re-ask and never to
keep part of a reply. Instead the contract says to settle every verdict and
check the hashes before printing, and that a neighbouring group's hash moves
the verdict, and `check-reply` names a reply with several objects so the
re-ask says what to fix.
"""

from __future__ import annotations

import json

from utils import check_reply
from utils.subagent_prompts import _VALIDATION_REPLY_CONTRACT

GROUP_10 = "fda2c710694e5aaa"
GROUP_11 = "2a67675417917338"


def _object(first_hash: str, second_hash: str) -> str:
    return json.dumps({
        "metadata": {"model": "claude-sonnet-5-5", "timestamp": "2026-10-08T00:00:00Z",
                     "schema_version": "2.1"},
        "groups": [
            {"group_index": 10, "group_hash": first_hash, "status": "valid",
             "reason": "The plan gives no start grid.", "confidence": 0.85},
            {"group_index": 11, "group_hash": second_hash, "status": "valid",
             "reason": "The plan never says a network is needed.", "confidence": 0.55},
        ],
    })


#: The reply as the host wrote it (run 11), reasons shortened.
RUN_11_REPLY = (
    _object(GROUP_11, GROUP_10)
    + "\n\nWait, the hashes above are swapped relative to the groups; corrected reply:\n\n"
    + _object(GROUP_10, GROUP_11) + "\n"
)


def _problems(tmp_path, text: str):
    path = tmp_path / "validation_batch_5.json"
    path.write_text(text, encoding="utf-8")
    task = {"task_id": "validation-batch-5", "kind": "validation",
            "output_path": str(path), "group_ids": [GROUP_10, GROUP_11]}
    return check_reply.check_task_reply(task, str(tmp_path))


class TestTheContract:

    def test_it_says_to_check_the_hashes_and_print_once(self):
        text = " ".join(_VALIDATION_REPLY_CONTRACT.split())
        assert (
            "Settle every verdict, and check each result's `group_hash` against "
            "its group above, before you print anything, then print the object "
            "once: a reply that adds a correction, a note or a second object "
            "after the first does not parse."
        ) in text

    def test_it_says_a_neighbouring_hash_moves_the_verdict(self):
        text = " ".join(_VALIDATION_REPLY_CONTRACT.split())
        assert (
            "a wrong hash discards the result, and the hash of another group "
            "in this batch puts your verdict on that group."
        ) in text


class TestCheckReplyNamesIt:

    def test_the_run_11_reply_is_named_as_two_objects(self, tmp_path):
        (problem,) = _problems(tmp_path, RUN_11_REPLY)
        assert problem.startswith("reply holds 2 JSON objects, not one (Extra data at line 3")
        assert "re-ask for the whole reply, printed once; never keep one of them yourself" in problem

    def test_neither_object_is_kept(self, tmp_path):
        """The check never rewrites the file: the batch stays invalid until
        the subagent's own single reply replaces it."""
        _problems(tmp_path, RUN_11_REPLY)
        assert (tmp_path / "validation_batch_5.json").read_text() == RUN_11_REPLY

    def test_one_object_with_prose_after_it_is_plain_invalid_json(self, tmp_path):
        (problem,) = _problems(tmp_path, _object(GROUP_10, GROUP_11) + "\nDone {see above}.")
        assert problem.startswith("reply is not valid JSON (Extra data")

    def test_the_corrected_reply_alone_is_ok(self, tmp_path):
        assert _problems(tmp_path, _object(GROUP_10, GROUP_11)) == []
