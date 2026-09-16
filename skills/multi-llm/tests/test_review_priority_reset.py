"""A reaggregated review must replace the previous run's finding count."""

from utils import cli_common
from utils.review_orchestrator_base import record_high_priority_groups
from utils.status_file import read_status


def test_clean_reaggregation_clears_previous_high_findings(tmp_path, monkeypatch):
    context = cli_common.ResultContext(mode="review-plan", phase="review-plan")
    monkeypatch.setattr(cli_common, "_ACTIVE", context)
    groups = [{"suggestions": [{"importance": "HIGH"}]}]
    assert record_high_priority_groups(str(tmp_path), groups) == 1
    assert read_status(str(tmp_path))["high_priority_groups"] == {"count": 1}
    assert context.high_priority_groups == {"count": 1}

    assert record_high_priority_groups(str(tmp_path), []) == 0
    assert not read_status(str(tmp_path)).get("high_priority_groups")
    assert context.high_priority_groups is None
