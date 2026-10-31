# Scope-policy and report-tag implementation test

Related: [implementation plan](scope-policy-and-report-tags.md) and
[test evidence](scope-policy-test-evidence/).

Tested 2026-10-31. **Both confirmed P2 defects have been fixed** in the same
`feat/scope-policy` worktree; the fix verification is recorded below. The original ordinary CI
suite and all focused scope-policy tests passed. Cheap live-model checks exercised
reviewing, validation, report generation, default application eligibility and a
bounded write. Subsequent packaged/browser checks passed the scope-policy cases
and found a pre-existing P2 dry-run reporting issue. **That issue is now fixed in
both package routes**; its verification is recorded at the end of this report.

The staged implementation is in `/home/marc/code/mllm-wt/scope-policy`, branch
`feat/scope-policy`, based on `c91e8d2937ecc543d7cd94fb8a3dff7c4a8df46a`.
The main checkout had no staged changes. Tested index tree:
`00d2631ca24c556e937c43f4d13d17e7e5ed9400`.
The implementation and its staging were unchanged at the end of the original testing pass.
The subsequent fixes are included with the implementation in the same feature change.
QA artifacts were initially saved untracked in the main checkout. They now live under
`plans/` alongside the implementation plan, as requested by the user.

## Findings

### P2 — A disjoint reviewer selection bypasses the review-policy conflict guard

Location: [review_orchestrator_base.py:572](../skills/multi-llm/utils/review_orchestrator_base.py#L572),
used at [review_orchestrator_base.py:7912](../skills/multi-llm/utils/review_orchestrator_base.py#L7912)
and [code_review_orchestrator.py:6074](../skills/multi-llm/code_review_orchestrator.py#L6074).

`requested_results_exist()` checks only the models selected on this invocation.
If another model already produced a result, changing the selection makes
`has_results` false. `resolve_review_round()` then accepts a different explicit
policy as a fresh round without retiring those older results.

Reproduced through the real review-plan main function with mocked reviewer
transport and pending validation:

1. Reviewer A runs under `preserve`, leaving an unfinished round and its result.
2. Select only reviewer B and request `expand`, without a new-round flag.
   Actual: exit 0, an expanding reviewer is dispatched, and the recorded policy
   changes to `expand`. Expected: exit 3 before dispatch.
3. Select A and B with policy omitted. Actual: exit 0, neither reviewer runs,
   and the resulting groups include both reviewers' cached outputs under the
   single `expand` policy.

This violates plan §4's prohibition on combining reviewer outputs requested
under different policies. It also makes the recorded reviewer-policy statement
inaccurate for some findings. The implementation notes already listed this as
an unverified concern; it is now confirmed.

Evidence: [result](scope-policy-test-evidence/policy-mixing-result.json),
[full transcript](scope-policy-test-evidence/policy-mixing-transcript.log),
[three-step reproducer](scope-policy-test-evidence/probe_policy_mixing.py),
[failing regression](scope-policy-test-evidence/test_policy_mixing_regression.py).

Suggested correction: establish whether the recorded round has any reviewer
outputs independently of the current selection, then refuse the policy change
or retire all previous-round outputs through the existing new-round mechanism.

### P2 — Resuming a fresh default run restores the previous run's policy

Locations: [implement_orchestrator.py:2733](../skills/multi-llm/implement_orchestrator.py#L2733)
and [generate_tasks_orchestrator.py:659](../skills/multi-llm/generate_tasks_orchestrator.py#L659).

The fresh-run path correctly ignores the previous live choice for the immediate
invocation. However, it only persists an explicitly typed policy, leaving the
previous `scope_policy_runs` record live when the new run uses the default.
An unfinished new run's next continuation therefore inherits that older record.

The implementation integration regression invokes `implement.main()` and
captures the real task handoff, replacing only the write execution:

1. Start unfinished implementation with explicit `preserve`.
2. Start over with `--fresh` and no scope-policy option. Actual: new task prompts
   correctly use default `expand`.
3. Continue with `--resume` and no policy option. Actual: task prompts contain
   the `preserve` guidance again, and stdout says it retained the recorded policy.
   Expected: continue the fresh run under `expand`.

The generation policy resolver independently reproduces the same sequence with
`--force`: `preserve → expand → preserve`. This silently narrows resumed work
and breaks the policy persistence/default-compatibility contract. The existing
fresh-run tests stop after step 2.

Evidence: [generation result](scope-policy-test-evidence/fresh-resume-result.json),
[generation reproducer](scope-policy-test-evidence/probe_fresh_resume.py),
[implementation regression](scope-policy-test-evidence/test_fresh_resume_regression.py),
[combined failing-test output](scope-policy-test-evidence/additional-regressions.log).

Suggested correction: retire or replace the old run's choice when a new default
run begins, so its continuation cannot discover an earlier run's live policy.

## Verification

- **Focused scope-policy suite:** 461 passed in 6.50 seconds. Covered the nine
  new scope-policy/prompt/report test files. [Log](scope-policy-test-evidence/focused-tests.log).
- **Full ordinary CI suite:** 26,666 passed, 281 skipped, 26 warnings in
  357.44 seconds, using four pytest workers and excluding `live`, `container`,
  `package` and `codex_jail` markers. [Log](scope-policy-test-evidence/ordinary-ci-tests.log).
- **Additional regression tests:** both fail as expected on the staged source;
  neither invokes a model. These expose the two findings above and were kept
  outside the implementation's existing test tree.
- **Generated documentation:** `hosts/render_capabilities.py --check` passed.
- **Codex payload consistency:** all 63 changed source payload files checked
  match their tracked Codex mirrors. [Source hashes and index identity](scope-policy-test-evidence/staged-source-record.json).
- **Whitespace check:** `git diff --cached --check` exits 2 for an extra blank
  line at EOF in `docs/development/host-reporting.md:209` and
  `docs/development/write-scheduler.md:706`. These are minor cleanup items.

Run the additional regressions from this checkout:

```bash
PYTHONPATH=/home/marc/code/mllm-wt/scope-policy/skills/multi-llm \
uv run --project /home/marc/code/mllm-wt/scope-policy/skills/multi-llm -- \
pytest -p tests.conftest \
  plans/scope-policy-test-evidence/test_fresh_resume_regression.py \
  plans/scope-policy-test-evidence/test_policy_mixing_regression.py -q
```

## Live test

Temporary Git project: `/tmp/scope-policy-live-20261031`. All reviewer, validator
and writer calls explicitly used `claude-code:haiku`; runtime usage identified
the served model as `claude-haiku-5-5`. There were twelve live model calls.
Total provider-reported usage was **$0.06423612**; this is the usage metadata,
not an independently verified bill. CLI version: 2.1.316. The harness warned
that its parser fixtures were captured against 2.1.293; parsing completed.

The tests used the staged source orchestrators' supported internal testing seam,
not the installed plugin's separate cached source. The small
[phase harness](scope-policy-test-evidence/run_source_phase.py) imports each
orchestrator's `main()` and preserves its exit status. It does not exercise the
installed MCP tool, native host UI, detach, or a full multi-phase workflow live.
Those contracts were covered only by the existing ordinary tests where selected.

- A real `preserve` review and live validation completed. Reports retain
  separate reviewer and authoritative group tags. The validation results were
  bound to the recorded baseline.
- A real `expand` review and live validation completed. Its subsequent apply
  dry run retained all seven valid groups, including three whose scope was
  `unclear`, without adding scope deferrals.
- A deterministic reviewer fixture deliberately mislabeled PDF export as
  `in_scope`. The real Haiku validator independently classified it as
  `scope_extension`, retaining its `needs-human-decision` validity status.
  `preserve` deferred it before judging/writing; the candidate remains in the
  Markdown and HTML reports, and the persisted deferral remains in state.
- In that seeded case, the only eligible header clarification reached the real
  writer. The writer declined it because the minimal plan did not define enough
  fields to choose columns. No change was applied; the outcome is recorded as
  declined/no-op, and PDF was not added.
- A separate real apply selected the CSV-quoting correction from the live
  `preserve` review. With the policy option omitted, it inherited `preserve`,
  materialized a bound `in_scope` write task and applied exactly one suggestion.
  The plan now specifies proper CSV escaping and retains its CLI constraints.
  [Applied diff](scope-policy-test-evidence/live-preserve-write.diff).

Live evidence: [structured results](scope-policy-test-evidence/live-results.json),
[preserve review log](scope-policy-test-evidence/live-preserve.log),
[expand review log](scope-policy-test-evidence/live-expand.log),
[expand apply dry run](scope-policy-test-evidence/live-expand-apply-dry-run.log),
[leaked-candidate validation](scope-policy-test-evidence/live-leaked-validation.log),
[leaked-candidate apply](scope-policy-test-evidence/live-leaked-apply-preserve.log),
[successful bounded write](scope-policy-test-evidence/live-preserve-write.log).
Copied inputs, task files, state, reports and provider usage are under
[live-artifacts](scope-policy-test-evidence/live-artifacts/).

The original pass ran no container/package release gate or browser rendering test. The HTML
template/selection tests in the focused suite passed, including their Node
execution. The live writer edited a plan document; it did not implement or test
a CSV application. The original testing pass made no implementation fixes.

## Fix verification

Both issues were corrected in `/home/marc/code/mllm-wt/scope-policy` at the user's request.

- The review-policy guard now checks the entire round's cached reviewer results,
  including unselected reviewers, using the shared result-file filter. A policy
  change without the required new-round controls is refused before dispatch.
- When an authorized new round changes policy, all earlier result caches are
  renamed under the existing `.stale-earlier-run` convention. This prevents a later
  selection from reusing a reviewer requested under the earlier policy. A failed
  retirement refuses the new round before its policy is recorded.
- Generation, implementation, full and improve workflows now share
  `scope_policy_runs.record_run_start`. A default fresh run closes the previous
  choice, so its continuation cannot restore that older policy. Dry-run policy
  resolution leaves the stored choice intact.
- Regression coverage was added to the implementation's test suite for both
  policy-change directions, disjoint selections, old-cache retirement, retirement
  failure, legitimate continuation under the same policy, fresh-run resumes and
  dry runs. Existing fresh/full/composite tests now continue through an interrupted
  fresh run's next resume, which the original tests did not check.
- Codex and OpenCode payloads were regenerated. Both match all eight changed
  runtime source files. The two documentation EOF whitespace findings were also
  removed.

Completed checks:

- The two original external reproducers now pass:
  [original regression verification](scope-policy-test-evidence/fix-original-regressions.log).
- The expanded scope-policy suite passes: **473 passed**:
  [focused test output](scope-policy-test-evidence/fix-focused-tests.log).
- The full ordinary CI rerun passes: **26,678 passed, 281 skipped, 26 warnings**
  in 361.98 seconds, with the same marker exclusions and four workers as the
  original run. [Full-suite output](scope-policy-test-evidence/fix-ordinary-ci-tests.log).
- Generated host documentation is current; both `git diff --check` and
  `git diff HEAD --check` pass.

Final source hashes still match the tested bytes and regenerated Codex payload.
The fixes were unstaged at verification; they are included with the feature change.
No additional live model calls were needed for the fixes.
[Fix source hashes](scope-policy-test-evidence/fix-source-record.json),
[source/test/documentation patch](scope-policy-test-evidence/fix-source-changes.patch),
[Codex regeneration](scope-policy-test-evidence/fix-codex-regeneration.log),
[OpenCode regeneration](scope-policy-test-evidence/fix-opencode-regeneration.log).

## Subsequent real-world checks

Ran against a copy of the regenerated Codex package installed at
`/tmp/scope-policy-world-20261031/installed-plugin`, with four small Git projects
beside it. The [QA client](scope-policy-test-evidence/run_packaged_mcp.py) drove
the package's actual stdio MCP server, dispatcher, detached workers and provider
CLIs. It supplied roots and session metadata itself; this was not an interactive
Codex application session or a marketplace-install test. All new model calls used
Haiku, including a `haiku:low` spelling for the second reviewer slot.

The following scope-policy checks passed:

- **Real application review/apply:** reviewed a planted CSV escaping regression
  in `expenses.py`. Before applying, the application's special-character test
  failed while its header and API-signature tests passed. The package's live
  review and validation found two valid in-scope regressions and one unclear
  compatibility concern. The preserve apply fixed the valid items and deferred
  the unclear one. All three application tests then passed, and the public
  `export_expenses(records)` signature remained unchanged.
- **Interrupted review and different reviewer:** cancelled a real detached
  preserve review after its reviewer wrote a result. MCP confirmed it aborted.
  Selecting only `haiku:low` with expand was refused at exit 3 with
  `[SCOPE_POLICY_CONFLICT]`. An authorized new round retired the earlier cache;
  subsequently selecting both reviewers reused only the new-policy slot and
  dispatched the retired reviewer again. This directly exercises the first fix
  through the packaged boundary.
- **Fresh generation and resume:** started generation under preserve, cancelled
  it, started over with force and no policy, cancelled again, then resumed with
  no policy. Both cancellations were confirmed aborted. The old preserve record
  was closed, the resumed prompt contained no preserve guidance, and generation
  completed under expand with a task list. This exercises the second fix through
  the real packaged generation route. Implementation's own resume path remains
  covered by the automated integration regression, not a further live writer run.
- **Browser export and apply:** used a fresh fixture derived from the earlier
  live reviewer/validator artifacts. Chrome displayed the tags and provisional
  versus authoritative distinction. Checked a group skip, changed an in-scope
  description to add PDF export, downloaded `user_selections.json`, and supplied
  that exact download to the package. The engine respected the skip and deferred
  the edited group as stale/unclear. Then removed the skip and selected a valid
  override for the PDF candidate: preserve still deferred both groups. An explicit
  expand dry run admitted both candidates. PR and flat views rendered, the edited
  tag visibly warned about stale classification, and Chrome reported no page
  errors. The dry-run eligibility results were correct despite the wrapper
  reporting problem below.

T3 preview reads/navigation worked, but its action calls repeatedly timed out
with `Preview automation click timed out after 15000ms.` The browser interaction
checks therefore used the `agent-browser` skill and its own isolated Chrome
session. The preview tab, Chrome session and temporary HTTP server were closed.
No implementation files changed during these additional tests.

Evidence: [structured results and costs](scope-policy-test-evidence/world-results.json),
[application tests before](scope-policy-test-evidence/world-app-tests-before.log),
[application tests after](scope-policy-test-evidence/world-app-tests-after.log),
[review RPC result](scope-policy-test-evidence/world-code-review.json),
[apply RPC result](scope-policy-test-evidence/world-code-apply.json),
[policy conflict](scope-policy-test-evidence/world-review-policy-conflict.json),
[fresh generation state](scope-policy-test-evidence/world-generation-fresh-state.json),
[generation resume](scope-policy-test-evidence/world-generation-resume.json),
[reviewer reselection](scope-policy-test-evidence/world-review-reselect.json),
[downloaded edited selections](scope-policy-test-evidence/world-browser-selections.json),
[downloaded valid override](scope-policy-test-evidence/world-browser-valid-override-selections.json),
[edited PR screenshot](scope-policy-test-evidence/world-browser-edited.png),
[flat screenshot](scope-policy-test-evidence/world-browser-flat.png),
[browser errors](scope-policy-test-evidence/world-browser-errors.json).
Project inputs, worker logs, task files, state and reports are copied under
[world-artifacts](scope-policy-test-evidence/world-artifacts/).

### Additional P2 — MCP wrapper reports successful dry runs as errors (pre-existing)

Location: [mcp_server.py:2499](../packaging/codex/mcp_server.py#L2499),
the launch-banner guard in `Server.run_workflow`.

The MCP wrapper always requests detached execution for apply modes. An apply
dry run completes in the foreground and prints its correct eligibility/batch
preview without a detached-run banner. The wrapper then returns
`ok: false, state: "error", exit: 0` with this message:

> the launch reported no run id on its banner, so the run cannot be polled or aborted safely. Refusing to adopt whatever run is newest.

This occurred for both preserve and expand dry runs. The engine did not apply
any changes and its policy decisions were correct; the wrapper's success/error
reporting is the problem. The same request against the original main-checkout
package reproduced the identical error, confirming it predates this feature.
The temporary baseline virtual environment created by that control run was
removed after the test. The original real-world test pass made no fix for this
separate packaging issue; the subsequent fix is recorded below.

Evidence: [preserve edited-selection request](scope-policy-test-evidence/world-browser-apply.json),
[preserve validity-override request](scope-policy-test-evidence/world-browser-override-apply.json),
[expand request](scope-policy-test-evidence/world-browser-expand-apply.json),
[original-package control](scope-policy-test-evidence/world-baseline-dry-run.json).

### Updated usage

The additional eight completed model calls reported **$0.05945721**. Added to the
earlier $0.06423612, the recorded completed-call usage is **$0.12369333**.
Three intentionally interrupted calls did not return final usage metadata, so
that figure excludes any usage they incurred and is not a complete billing
total. Copied usage from the earlier browser fixture was excluded from the new
sum. These figures also exclude the parent Codex conversation and are not an
independently verified bill. All model calls in this pass were Haiku.

## Packaged dry-run reporting fix

Fixed the additional P2 in the same `feat/scope-policy` worktree. Both the Codex
MCP server and OpenCode package supervisor now accept a completed foreground
preview without a detached-run banner when the invocation explicitly requests
one of the declared preview flags. The shared CLI surface supplies those flags
from `cli_common.FOREGROUND_PREVIEW_FLAGS`; surface schema v7 keeps both package
supervisors in sync with that metadata.

The change preserves the supervision checks:

- A valid structured dispatcher result is still required.
- Ordinary launches without a banner are still refused.
- A result that says it is running or carries a continuation still requires a
  banner, even with a preview flag.
- A preview that does produce a banner is polled using that banner's run id.
- A successful preview reports that invocation's result without marking the
  apply phase complete, creating write tasks, or applying changes.

Added regression tests for foreground previews, misleading flag-like text,
missing structured results, missing launch banners and detached previews in both
package routes. A real-dispatcher test covers expand and preserve while asserting
that the plan bytes and phase completion remain unchanged.

Verification:

- The MCP/surface test pass completed with **428 passed**. The final additional
  handshake guard cases pass in the focused rerun: **12 passed**.
  [MCP/surface log](scope-policy-test-evidence/dry-run-fix-mcp-tests.log),
  [final handshake regression log](scope-policy-test-evidence/dry-run-fix-handshake-tests.log).
- OpenCode state-machine tests: **40 passed**.
  [OpenCode test log](scope-policy-test-evidence/dry-run-fix-opencode-tests.log).
- Codex and OpenCode packages regenerated; Node syntax, generated host docs and
  `git diff HEAD --check` pass.
- Replayed the saved browser fixture against a fresh copy of the final
  regenerated MCP package under both preserve and expand. Both return
  `ok: true, state: "done", exit: 0`, with no run id or abort command. The
  apply phase remains incomplete and no write task was created.
  [Preserve result](scope-policy-test-evidence/dry-run-fix-preserve.json),
  [expand result](scope-policy-test-evidence/dry-run-fix-expand.json).

The final full ordinary CI rerun passes: **26,699 passed, 281 skipped, 26 warnings**
in 357.45 seconds, with four workers and the same integration-marker exclusions.
[Full-suite log](scope-policy-test-evidence/dry-run-fix-ordinary-ci-tests.log).
Final source hashes still match the tested bytes. No additional live model calls
were made. The implementation, fixes, planning documents and evidence are included together in the feature change.
[Final fix source record](scope-policy-test-evidence/dry-run-fix-source-record.json),
[source/test patch](scope-policy-test-evidence/dry-run-fix-source-changes.patch),
[Codex regeneration](scope-policy-test-evidence/dry-run-fix-codex-build.log),
[OpenCode regeneration](scope-policy-test-evidence/dry-run-fix-opencode-build.log).

## Final integration checks

The implementation plan, this report and their evidence now live together under
`plans/`, with reciprocal links between the two documents. Unrelated planning
notes remain ignored. The repository's tracked-plan guard permits these two
documents and their evidence directory; credential scanning still covers them.

After staging the complete change, the documentation-link and repository-policy
checks passed: **69 passed**. [Final check log](scope-policy-test-evidence/precommit-checks.log).
The final staged diff also passes `git diff --cached --check`. No additional
model calls were needed for these integration checks.
