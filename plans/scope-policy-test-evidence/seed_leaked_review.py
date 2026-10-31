"""Seed two known reviewer candidates; validation and application run live."""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/marc/code/mllm-wt/scope-policy/skills/multi-llm")
from tests.test_scope_prompts_and_validation import _review_plan_main

root = Path('/tmp/scope-policy-live-20261031/leaked-fixture')
capture = io.StringIO()
with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
    result = _review_plan_main(root, 'preserve')
(Path(__file__).parent / 'leaked-review-seed.log').write_text(capture.getvalue())
print(json.dumps({'plan': str(result['plan']), 'exit': result['code'],
                  'reviewer': 'deterministic fixture; not a live model'}))
