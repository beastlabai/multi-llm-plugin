"""``multi_llm.py util`` in-process, and the flags of a stop's first run.

``multi_llm_util(argv)`` runs ``multi_llm.main(["util", *argv])``. No fixtures.
"""

from __future__ import annotations


#: The answers the check-gate first stop offers, in the order a user weighs
#: them: the combined go-on-without-fixing command, then T005's (refused before
#: dispatch), then T002's (its findings hold T007).
FIRST_STOP_FLAGS = [
    ["--resume", "--human-task", "T002=done", "--human-task", "T005=skip"],
    ["--resume", "--human-task", "T005=skip"],
    ["--resume", "--human-task", "T005=done"],
    ["--resume", "--human-task", "T002=done"],
    ["--resume", "--human-task", "T002=skip"],
    ["--resume", "--human-task", "T002=retry"],
]


def multi_llm_util(argv) -> int:
    import multi_llm

    return multi_llm.main(["util", *argv])
