"""Assert packaged/browser test outcomes, preserve artifacts, tally known usage."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/tmp/scope-policy-world-20261031')
OUT = Path(__file__).parent


def load(path):
    return json.loads(path.read_text())


def envelope(name):
    return json.loads(load(OUT/(name+'.json'))['result']['content'][0]['text'])


def output_json(name):
    text = envelope(name)['output']
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]


for name in ('world-code-review', 'world-code-apply', 'world-generation-resume',
             'world-review-new-round', 'world-review-reselect'):
    data = envelope(name)
    assert data['ok'] is True and data['exit'] == 0, (name, data)
for name in ('world-review-interrupt', 'world-generation-preserve-interrupt',
             'world-generation-fresh-interrupt'):
    assert 'confirmed aborted' in envelope(name)['message'], name
conflict = envelope('world-review-policy-conflict')
assert conflict['exit'] == 3 and '[SCOPE_POLICY_CONFLICT]' in conflict['output']
phase = ROOT/'review-resume/feature/review-plan'
assert (phase/'claude-code_haiku.json.stale-earlier-run').exists()
review_log = (phase/'orchestrator-run.log').read_text()
assert '[SKIP] claude-code:haiku:low' in review_log
assert '[claude-code:haiku] Starting review' in review_log
generation = ROOT/'generation-resume/feature'
state = load(generation/'state.json')
assert state['scope_policy_runs']['generate-tasks']['closed_at']
tasks = load(generation/'generate-tasks/task_generation_tasks.json')
assert all('## Scope Policy: Preserve' not in t['prompt']
           for t in tasks.get('tasks', tasks.get('batches', [])))
assert 'scope-policy: expand' in (generation/'generate-tasks/orchestrator-run.log').read_text()
assert 'FAIL' in (OUT/'world-app-tests-before.log').read_text()
assert '\nOK\n' in (OUT/'world-app-tests-after.log').read_text()

edited = output_json('world-browser-apply')
assert edited['to_apply'] == []
assert edited['summary']['user_skipped_count'] == 1
assert edited['scope_policy']['deferred_items'][0]['authority'] == 'stale'
overridden = output_json('world-browser-override-apply')
assert overridden['to_apply'] == []
assert overridden['scope_policy']['deferred_count'] == 2
assert {g['relation'] for g in overridden['scope_policy']['deferred_items']} == {'unclear','scope_extension'}
expanded = envelope('world-browser-expand-apply')
assert 'Valid (to apply): 2' in expanded['stderr']
baseline = envelope('world-baseline-dry-run')
assert expanded['exit'] == baseline['exit'] == 0
assert expanded['ok'] is baseline['ok'] is False
assert expanded['message'] == baseline['message']
assert 'no run id' in expanded['message']
assert load(OUT/'world-browser-errors.json')['data']['errors'] == []

# Count each final completed call once. The browser fixture contains copied
# earlier usage and is deliberately excluded; interrupted calls lack final usage.
usage_sources = {}
for relative in ('expenses/feature/code-review/.status.json',
                 'review-resume/feature/review-plan/.status.json'):
    usage_sources[relative] = sum(v.get('cost_usd', 0)
                                  for v in load(ROOT/relative).get('usage', {}).values())
for relative in ('expenses/feature/code-review/subagent_batch_report.json',
                 'expenses/feature/apply-fixes/subagent_batch_report.json',
                 'generation-resume/feature/generate-tasks/subagent_batch_report.json'):
    usage_sources[relative] = load(ROOT/relative)['usage']['cost_usd']
additional = round(sum(usage_sources.values()), 8)
original = load(OUT/'live-results.json')['reported_cost_usd']

dest = OUT/'world-artifacts'
for project in ('expenses','review-resume','generation-resume','browser-roundtrip'):
    for path in (ROOT/project).rglob('*'):
        if not path.is_file() or any(part in ('.git','.venv','__pycache__') for part in path.parts):
            continue
        target = dest/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
source_record = load(OUT/'fix-source-record.json')
source = Path(source_record['worktree'])
for name, digest in source_record['runtime_sha256'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    assert (ROOT/'installed-plugin'/name).read_bytes() == (source/name).read_bytes()
summary = {
    'temporary_root': str(ROOT),
    'package_entrypoint': 'Copied regenerated Codex MCP package, real stdio JSON-RPC client; no actual Codex UI session',
    'model': 'claude-code:haiku (also :low for the second reviewer slot)',
    'application_tests': {'before': '1 failed, 2 passed', 'after': '3 passed'},
    'policy_resume_checks': 'passed',
    'browser_engine': 'Chrome via agent-browser after T3 preview action timeouts',
    'browser_scope_checks': 'passed: skips, description edits, stale binding, validity override, explicit expand eligibility',
    'browser_errors': [],
    'additional_finding': {'severity': 'P2', 'title': 'MCP package misreports successful dry runs as errors',
                           'reproduced_on_original_HEAD_package': True,
                           'message': expanded['message'], 'implementation_changed': False},
    'additional_reported_cost_usd': additional,
    'cumulative_reported_cost_usd': round(original + additional, 8),
    'usage_sources': usage_sources,
    'completed_calls_with_usage': 8,
    'interrupted_calls_without_final_usage': 3,
    'cost_note': 'Reported completed-call usage only; interrupted calls may incur additional unreported usage. Excludes the parent Codex conversation and is not a verified bill.',
}
(OUT/'world-results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
