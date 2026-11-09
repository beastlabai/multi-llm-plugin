"""``utils/text_fmt.py``: ``plural``, and the commit modules that use it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import commit_apply, commit_brief, commit_range, commit_report  # noqa: E402
from utils.text_fmt import plural  # noqa: E402


@pytest.mark.parametrize("count, expected", [
    (0, "0 files"), (1, "1 file"), (2, "2 files"), (-1, "-1 files"),
])
def test_a_regular_plural(count, expected):
    assert plural(count, "file") == expected


def test_an_irregular_plural():
    assert plural(1, "batch", "batches") == "1 batch"
    assert plural(3, "batch", "batches") == "3 batches"
    assert plural(3, "batch", plural="batches") == "3 batches"


@pytest.mark.parametrize("module", [commit_apply, commit_brief, commit_range, commit_report],
                         ids=lambda module: module.__name__)
def test_the_commit_modules_share_it(module):
    assert module._plural is plural
