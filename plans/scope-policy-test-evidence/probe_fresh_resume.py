"""Probe the effective policy on a new default run and its continuation."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/marc/code/mllm-wt/scope-policy/skills/multi-llm')
import generate_tasks_orchestrator as gto
from utils.state_manager import StateManager
from utils import scope_policy_runs as runs

root = Path(tempfile.mkdtemp(prefix='scope-policy-fresh-resume-'))
plan = root/'feature.md'
plan.write_text('# Feature\nExport CSV.\n')
phase = str(root/'feature'/'generate-tasks')
capture = io.StringIO()
with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
    manager = StateManager(plan)
    manager.save()
    def resolve(*flags):
        args = gto.parse_args(['--plan-file', str(plan), '--yes', *flags])
        return gto._resolve_scope_policy(args, plan, phase)
    first = resolve('--scope-policy', 'preserve')
    forced = resolve('--force')
    # A partial/failed fresh run has no phase completion yet.
    resumed = resolve()
summary = {
    'root': str(root), 'first': first, 'forced_default_new_run': forced,
    'continuation': resumed, 'expected_continuation': 'expand',
    'record': runs.run_record(StateManager(plan).state, 'generate-tasks'),
}
out = Path(__file__).parent
(out/'fresh-resume-transcript.log').write_text(capture.getvalue())
(out/'fresh-resume-result.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
