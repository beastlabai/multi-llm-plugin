"""Copy the small, relevant live artifacts and assert the observed outcomes."""
import json
import shutil
from pathlib import Path

ROOT = Path('/tmp/scope-policy-live-20261031')
OUT = Path(__file__).parent
DEST = OUT/'live-artifacts'
FILES = {
    'state.json', 'validation.json', 'grouped.json', 'report.md', 'report.html',
    'report_data.json', 'validation_tasks.json', 'validation-snapshot.md',
    'subagent_batch_report.json', 'write_batch_report.json', 'write_tasks.json',
    'orchestrator_output.json', 'summary.md', 'user_selections.json', '.status.json',
    'claude-code_haiku.json', 'cursor-agent_auto.json',
}

def load(path):
    return json.loads(path.read_text())

def items(data):
    return data.get('groups', []) if isinstance(data, dict) else data

cost = 0.0
review_summaries = {}
for name in ('preserve', 'expand', 'leaked-fixture/feature'):
    phase = ROOT/name/'review-plan'
    verdicts = load(phase/'validation.json')['groups']
    relations = {}
    for group in verdicts:
        relation = group['scope_relation']
        relations[relation] = relations.get(relation, 0) + 1
    review_summaries[name] = {
        'verdict_count': len(verdicts), 'scope_relations': relations,
        'validator_runners': load(phase/'subagent_batch_report.json')['runner_versions'],
    }
    for value in load(phase/'.status.json').get('usage', {}).values():
        cost += value.get('cost_usd', 0)

for path in ROOT.rglob('subagent_batch_report.json'):
    cost += load(path).get('usage', {}).get('cost_usd', 0)

leak = ROOT/'leaked-fixture/feature'
verdicts = load(leak/'review-plan/validation.json')['groups']
assert {v['scope_relation'] for v in verdicts} == {'in_scope', 'scope_extension'}
pdf = next(v for v in verdicts if v['scope_relation'] == 'scope_extension')
assert pdf['scope_policy_decision']['eligible'] is False
output = load(leak/'apply-suggestions/orchestrator_output.json')
assert len(output['to_apply']) == 1
assert 'PDF' not in output['to_apply'][0]['title']
assert 'PDF' not in (ROOT/'leaked-fixture/feature.md').read_text()
assert len(load(leak/'state.json')['scope_policy']['review-plan']['deferrals']) == 1

before = (ROOT/'preserve-before-apply.md').read_text()
after = (ROOT/'preserve.md').read_text()
assert before != after
assert 'csv module' in after
assert 'Keep the current CLI arguments. Existing callers must keep working.' in after
tasks = load(ROOT/'preserve/apply-suggestions/write_tasks.json')['batches']
assert all(t['scope']['policy'] == 'preserve' for t in tasks)
assert all(i['relation'] == 'in_scope' for t in tasks for i in t['scope']['items'])
assert 'Suggestions applied: 1' in (OUT/'live-preserve-write.log').read_text()
assert 'Valid (to apply): 7' in (OUT/'live-expand-apply-dry-run.log').read_text()
assert review_summaries['expand']['scope_relations']['unclear'] == 3

for path in ROOT.rglob('*'):
    if path.is_file() and (path.name in FILES or path.name.endswith('.prompt.txt')):
        target = DEST/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
for name in ('preserve.md','preserve-before-apply.md','expand.md','leaked-fixture/feature.md',
             '.multi-llm/providers.yaml'):
    target = DEST/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/name, target)

summary = {
    'temporary_project': str(ROOT), 'models': ['claude-code:haiku'],
    'reviews': review_summaries,
    'bounded_write': {'applied': 1, 'inherited_policy': 'preserve'},
    'leaked_candidate': {'reviewer': 'seeded fixture', 'live_validation': True,
                         'pdf_relation': pdf['scope_relation'], 'pdf_status': pdf['status'],
                         'pdf_deferred': True, 'eligible_header_remedy': 'writer declined: missing plan decision'},
    'reported_cost_usd': round(cost, 8),
    'cost_note': 'Sum of provider-reported usage, not an independently verified bill.',
}
(OUT/'live-results.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
