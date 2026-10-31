# Workflows in depth

Where each mode writes its output, what `--full` and the review-and-apply modes
do on every run, every option, how ambiguous findings get decided, and how the
scope policy keeps a run to its agreed scope. Commit reviews have a page of their
own, [Commit reviews](commit-reviews.md). Back to the [README](../README.md).

---

## Plans and outputs

A **plan** is just a markdown file describing the change you want to make - a feature spec, refactor outline, or design doc. You write it (or have Claude draft it), then point multi-llm at it. There's no required template; the more concrete the plan, the more useful the reviews. Most workflows start from a plan, with two exceptions: `--ask` only needs a plan to give the models context for your question, and the commit modes (`--review-commits`, `--apply-commit-fixes`, `--improve-commits`) need none - they take a plan only as optional read-only context (`--context-plan <path>`, the **context plan**), and in place of a plan they write a generated **commit brief**: the range's commits, their messages and its diff stat, which the review reads as the statement of intent ([Commit reviews](commit-reviews.md)).

The path you pass (e.g. `plans/my-feature.md`) is always a **file**. From it, multi-llm derives a sibling **output directory** in the same folder, named after the plan file without its extension - so `plans/my-feature.md` produces `plans/my-feature/`. All workflow state and generated artifacts live there:

| Path | Written by |
| --- | --- |
| `plans/my-feature/state.json` | every mode (tracks workflow progress) |
| `plans/my-feature/review-plan/` | `--review-plan`, `--apply-suggestions` |
| `plans/my-feature/tasks/` | `--generate-tasks`, `--review-tasks` |
| `plans/my-feature/implement/summary.md` | `--implement` |
| `plans/my-feature/code-review/report.md` | `--review-code`, `--apply-code-fixes` |
| `plans/my-feature/ask/<question-slug>/answers.md` | `--ask` |
| `plans/my-feature/improve-plan/`, `plans/my-feature/improve-code/` | `--improve-plan`, `--improve-code`: the run's own folder (`.status.json`, log and lock); the two phases it runs write their usual folders |
| `plans/commit-reviews/<base>-<head>.md` | `--review-commits`, `--apply-commit-fixes`, `--improve-commits`: the commit brief, one per range |
| `plans/commit-reviews/<base>-<head>/code-review/report.md` (and `report.html`) | `--review-commits`, `--apply-commit-fixes` |
| `plans/commit-reviews/<base>-<head>/improve-commits/` | `--improve-commits`: the run's own folder, as for `--improve-code` |

The commit rows are the one exception to the sibling rule: a commit brief is named after its range (`<base>` and `<head>` are 12-character commit ids; a range that starts at the repository's first commit is `root-<head>`), and its folder beside it holds the `state.json` and phase folders a plan's would. The folder is `code_review.commit_reviews_dir`, `plans/commit-reviews` by default, relative to the repository root ([Per-project configuration](configuration.md#good-to-know)). When multi-llm creates it, it writes a `.gitignore` holding `*` inside it, so briefs and reports never show in `git status`; delete that file to commit them.

Throughout the README and these docs, `{plan}/...` refers to that derived directory. Because everything is stored next to the plan, a workflow is fully portable - move or commit the plan folder and its state travels with it.

## Running the pipeline

- **`--full`** chains the phases end to end: review-plan -> apply-suggestions -> generate-tasks -> review-tasks -> apply-task-suggestions -> implement -> review-code -> apply-code-fixes. It pauses at each apply step to let you approve `needs-human-decision` findings; pass `--yes` (alias `--non-interactive`) to run the whole pipeline fully unattended — zero prompts: non-interactive model selection, a judge agent decides every `needs-human-decision` item, and review-tasks runs automatically. For finer control, add just `--no-confirm` and/or `--agent-decide` (see [below](#letting-the-agent-decide-ambiguous-findings)) instead.
- **Applying code fixes** can happen two ways. Run `--apply-code-fixes` as a standalone pass to apply a previous review's fixes - this is the phase that handles `needs-human-decision` items (prompts, salvage, HTML badges). Or, to apply only the clearly-valid fixes inline during the review itself, add `--apply-fixes` to a `--review-code` run.

## Review and apply in one command

`--improve-plan` and `--improve-code` each run one review and then the apply of its findings, unattended, as one command:

| Command | Runs | The review gets | The apply gets |
| --- | --- | --- | --- |
| `--improve-plan <plan>` | `--review-plan`, then `--apply-suggestions` | `--yes`, plus your `--models` or `--quick` | `--yes --agent-decide --no-confirm` |
| `--improve-code <plan>` | `--review-code`, then `--apply-code-fixes` | `--yes`, plus your `--models` or `--quick` and your `--files` / `--files-from` / `--base-ref` | `--yes --agent-decide --no-confirm`, plus `--base-ref` (yours, else the one the review used) |

- **Models.** The review picks its own list, exactly as when you run it alone: with no flag, `defaults.modes.review-plan` (`--improve-plan`) or `defaults.modes.code-review` (`--improve-code`) when you set one, else `defaults.models`; with `--quick`, that entry's `quick` list, else `defaults.quick_models`. The `--improve-*` modes have no model list of their own, so a `defaults.modes.improve-plan` or `improve-code` entry is read by nothing ([Per-project configuration](configuration.md#good-to-know)). Nothing asks you to choose; `--models` wins over `--quick`.
- **Consolidation.** `--improve-plan` runs review-plan's consolidation pass between the two steps when the review recommends it, as `--full` does.
- **Re-runs and `--resume`.** Running the command again skips a step already done for the review on record and continues the one that stopped: an unfinished review runs its missing models, and an interrupted apply continues with `--resume` whether or not you type it. Once both steps are done for that review, it runs nothing, prints `[ALREADY_COMPLETE]` and the next command, and exits 2.
- **`--force`** starts a new review round: every model reviews the plan (or the code) as it reads now (`--force --rerun-all`, which replaces the previous round's review files), and the apply then applies that round's findings. A stopped run names the command that continues it, without `--force`; typing `--force` again starts yet another round. `--improve-code` also reviews again with no flag when a newer `--implement` ran after the review, or when the `--files` / `--files-from` you type differ from the files that review covered. A list on stdin (`--files-from -`) is not compared: the completed review stands, and the run says so on a `NOTE:` line (`does not compare --files-from - with the scope on record; the completed code review stands`); pass the list with `--files` or `--files-from <file>`, or add `--force`, to review it.
- **`--interactive` is refused.** To pick models or approve items yourself, run the two phases separately.
- **`--full` remains the whole workflow**; these two run only its first two or its last two phases.

**Before you run it.** Both modes edit files with nobody asked: the mode word is the opt-in, and there is no confirmation and no extra flag.

- *What gets written, and how to undo it.* `--improve-plan` edits the plan in place; the plan as it was before the apply stays in `{plan}/apply-suggestions/backup.md`. After a `--force` round that file is the plan as that round's apply found it, and the earlier round's backup stays beside it as `backup.<sha8>.md`. **`--improve-code` edits source files anywhere in the repository, unattended.** When the run uses `--exec-subagents` (the Codex and OpenCode default) each batch keeps a pre-batch snapshot that `multi_llm.py util restore-batch` restores; nothing backs up the repository as a whole, so git is the way back.
- *Who decides.* A judge agent decides every `needs-human-decision` item, and nobody is asked. The run's summary names the judge and each decision it made.
- *What your tree should be.* Commit or stash first, so `git diff` shows exactly what the run changed and `git restore` undoes it. Nothing refuses a dirty tree. When the apply runs with `--exec-subagents` (the Codex and OpenCode default), it records your uncommitted paths in `{plan}/state.json` (`pre_existing_changes`) as its baseline and names them at its start (`dirty tree at start: N uncommitted path(s) recorded as the baseline …`, the first ten listed), unless the plan already holds a baseline, which it keeps and only counts (an earlier `--implement` records one, before it changed anything). Without `--exec-subagents` (the Claude Code default, where the host runs the batches), the apply records nothing about your tree and says nothing about it.
- *What it costs.* One review call per model in the review's list (`defaults.models`, or the `--quick` or `--models` list), then validation, judging, and one write dispatch per batch. `--quick` is the cheap way to run it; `--force` pays for a whole new review round. `--improve-plan` adds one consolidation pass (one subagent call per consolidation batch) when the review recommends it - 15 or more actionable groups - and that pass changes nothing that gets applied.

## Options

These flags combine with any mode and are **position-independent** - they may appear before or after the plan path (and, for `--ask`, around the question). The one exception is a commit mode's range, which goes right after the mode word, before any flag ([Commit reviews](commit-reviews.md)). Not every flag applies to every mode.

| Flag | Effect |
| --- | --- |
| `--models <provider:model> ...` | Use exactly these models (variadic), overriding YAML defaults. Each must exist in `providers.yaml`. |
| `--quick` | Use the smaller `quick_models` subset for a faster  run. In `--improve-plan` / `--improve-code` / `--improve-commits` it goes to the review step. |
| `--interactive` | Force two-step model selection (provider, then models), ignoring YAML defaults. |
| `--no-confirm` | Skip confirmation prompts - for unattended/silent runs. |
| `--dry-run` | Show what would happen (tasks/batches) without making changes. Applies to `--implement` and the apply modes, `--apply-commit-fixes` included. |
| `--agent-decide` | In the apply modes, let a judge agent decide every `needs-human-decision` finding instead of prompting you (see [below](#letting-the-agent-decide-ambiguous-findings)). Implied by `--yes`, which the Codex and OpenCode shims bake in. `--claude-decide` / `--let-claude-decide` are accepted legacy spellings. |
| `--force` | **Meaning depends on mode.** In the fan-out review/ask modes (`--review-plan`, `--review-tasks`, `--review-code`, `--review-commits`, `--ask`), resume an interrupted run: keep already-completed per-model results, run only the missing models, and bypass the completed-phase / partial-completion guards. In the apply modes, it's an alias for `--yes` (confirm bulk approval). In `--improve-plan` / `--improve-code` / `--improve-commits`, it starts a new review round, then applies that round's findings ([above](#review-and-apply-in-one-command)). |
| `--rerun-all` | In the fan-out review/ask modes, re-run every model from scratch, discarding any existing per-model results. Combine with `--force` (`--force --rerun-all`) for a fresh full re-run of an already-completed phase. |
| `--focus "<text>"` | Steer a commit review toward a priority (not a filter). Taken by the three commit modes only. Recorded with the commit brief: a later run of the same range keeps it when you leave it out, and a different value (or `--focus ""`, which clears it) restarts the review round. `--apply-commit-fixes` keeps the value the review on record used ([Commit reviews](commit-reviews.md)). |
| `--context-plan <path>` | Give a commit review a plan as read-only context, and check the commits against it too. Taken by the three commit modes only. Copied into the review's folder when the review starts and recorded like `--focus`; type it again to review against an edited file. `--apply-commit-fixes` keeps the context plan the review on record used, and never edits it. |
| `--scope-policy expand\|preserve` | Whether the run may carry changes beyond the agreed scope. `expand`, the default for a new run, keeps today's behaviour and only tags each finding's scope; `preserve` asks the reviewers for in-scope corrections only, and validation and the apply hold every extension back. Taken by the review, apply and commit modes, `--generate-tasks`, `--implement`, the `--improve-*` modes and `--full`, not by `--ask`, `--status` or `--init`. Left out, a continuation keeps the policy on record ([below](#scope-policy)). |

The apply and implement modes accept additional approval flags (`--yes`, `--approve-all`, `--resume`, `--task`, and more); run `multi_llm.py <mode> --help` for the authoritative list, or read the per-mode files in `skills/multi-llm/instructions/`.

## Letting the agent decide ambiguous findings

When the models disagree or a suggestion is borderline, validation marks it `needs-human-decision`. In the apply modes (`--apply-suggestions`, `--apply-code-fixes`, `--apply-commit-fixes`, `--apply-task-suggestions`) Claude Code prompts you to approve or skip each one through its structured `AskUserQuestion` tool - it is the only host that has one, so on Codex and OpenCode the baked-in `--yes` resolves these without asking ([Hosts](hosts.md)). Every such prompt also offers a **Let the agent decide** option, per item or for a whole batch, that hands just that judgment to a judge agent instead of you.

To skip the prompts entirely and have a judge agent decide **every** `needs-human-decision` finding up front, pass `--agent-decide` - which is what the baked-in `--yes` already does on Codex and OpenCode. The judge is not necessarily Claude: on a host with native subagents (Claude Code) it is the host's own agent, and under `--exec-subagents` (the Codex and OpenCode default) it is the model your session is running on, e.g. `codex:<chat model>` on Codex. A `driver.subagent_runner` you set explicitly (in `<git-root>/.multi-llm/providers.yaml` or the file `MULTI_LLM_PROVIDERS_CONFIG` names) overrides that, and the host's floor runner (`codex:gpt-6-astra` on Codex) is used only when neither is available. The summary names it on a `judge:` line. `--improve-plan`, `--improve-code` and `--improve-commits` imply `--agent-decide` (with `--yes` and `--no-confirm`) for their apply step on every host.

```text
/multi-llm:multi-llm --apply-code-fixes plans/my-feature.md --agent-decide --no-confirm
```

Unlike the blanket approve/skip flags, "Let the agent decide" is a **per-item** judgment that **salvages** partially-valid findings - it trims each to its worthwhile core and applies that, skipping only findings with nothing worth keeping. The generated HTML report badges every finding **Approved / Salvaged / Skipped**. Add `--no-confirm` (as above) for a fully unattended run.

## Scope policy

`--scope-policy expand|preserve` decides whether a run may carry changes that go beyond the **agreed scope**: the goals, requirements and limits it was given. Whichever policy a run uses, every report tags every finding with its scope.

**`expand`, the default, changes no decision.** A new run without the flag is `expand`, and it reviews, validates, judges, selects and applies exactly as before. Reviewers still report the features, improvements, refactors and cleanup they find, and a warranted extension stays eligible under the usual validity, priority and selection rules. The tags are informational: nothing is suppressed, rejected, deferred or asked because a finding reads `Scope extension` or `Scope unclear`. `expand` is permission to extend, not an instruction to: it never adds work, and it never overrides a limit the plan states, such as "do not change the API".

**`preserve` adds three checks, each independent of the one before it.**

1. **Reviewers are asked not to propose extensions.** They report only findings whose fix stays within the agreed outcome - no new features, unrelated refactors or optional cleanup, however beneficial - and keep reporting missing or incomplete requirements, the bugs and security regressions the work introduced, and existing scope creep: removing something the plan never asked for is within scope. A requirement that cannot be met without going beyond the scope is reported as a blocker naming the decision you need to make, neither hidden nor quietly expanded.
2. **Validators reject any extension that got through.** Each group's whole proposed fix is classified again against the agreed scope, independently of the reviewers' own tags, and an extension or an unclear fix is rejected for this run. Correctness stays separate: a real finding whose fix extends the scope stays `valid` and stays in the report, marked `Deferred: scope preservation requested`; it is never turned into a false positive. The review prints one `[SCOPE_POLICY] preserve: …` line counting them.
3. **The apply defers everything not validated as within scope.** Just before batching, the apply checks each item again: only a finding validated `Within scope` against this review round's scope, and unchanged since, is applied. Everything else is **deferred** - kept apart from applied, skipped and invalid findings, named on `[SCOPE_DEFERRED]` lines and in `summary.md`'s "Deferred by scope policy" section - and the phase completes with its eligible work done. No flag lets a deferred item through: not `--yes`, `--no-confirm`, `--agent-decide`, a bulk approval, `--force` or `--accept-stale-consolidation`; `--min-priority` still applies on top. Writers are told to decline an item that turns out to need an extension. When an agreed requirement itself cannot be met inside the scope, the apply stops at exit 3 with `[SCOPE_BLOCKER]`, and the phases that depend on it refuse to start until you decide: amend the plan and review it again, or apply with `--scope-policy expand`.

Under `preserve`, `--generate-tasks` writes tasks only for what the plan asks for, and `--implement` tells each task's model to add no outcome beyond its task. The review modes stay read-only under either policy; the `--improve-*` modes edit their target unattended, as always.

**What the scope is measured against.** The plan's own goals, requirements and limits, for a plan review. For a task review, the plan the tasks came from: a task cannot bring a feature into scope by listing it. For a code review, the accepted plan and the decisions recorded in it; the diff and the task list show what was built, not what was asked for, so drift in the implementation cannot authorize itself, while fixing a regression the change introduced is within scope. For a commit review, the commit brief ([Commit reviews](commit-reviews.md)). A review round records its scope once, by hash, in `{plan}/state.json`, and keeps it: a plan edited while the round is under way does not move it (validation warns `[SCOPE_BASELINE_MOVED]`), a revalidation after an apply judges correctness against the current plan but scope against the round's, and the next round reads the edited plan. A limit that lives only in your conversation reaches no model: write it into the plan, or the `--context-plan` of a commit review.

**The tags in the reports.** Every `report.md` and `report.html` - plan, tasks, code and commit reviews, and the consolidated reports - uses three labels:

| Tag | The proposed fix |
| --- | --- |
| `Within scope` | completes, corrects or clarifies what the scope already asks for, even when that is substantial work: a prerequisite in another file, a fix for a regression the work introduced, the removal of existing scope creep |
| `Scope extension` | adds a capability, an outcome or unrelated cleanup the scope does not ask for, or changes an agreed outcome, even when it is worthwhile |
| `Scope unclear` | the scope does not settle it (missing or vague intent), the fixes on offer differ in scope and none was chosen, or there is no valid classification on record |

A tag describes the proposed fix, not whether the finding is real or how important it is, and it is shown apart from validity, selection and outcome. A finding's own `**Scope:**` line is its reviewer's tag, marked provisional; the group's `**Group scope:**` line is the validated one, and the one the apply reads. Each report's header carries a `**Scope policy:**` line, `**Scope tags:**` counts (per distinct group, not per finding) and a legend. Under `preserve` each group also shows a `**Scope outcome:**`, `Eligible under scope preservation` or `Deferred: scope preservation requested`, and the policy line says that additional scope was not solicited, so its extension count covers only the findings the reviewers returned. A consolidated group reads `Scope extension` when any group under it is one, else `Scope unclear` when any is unclear, and each group under it is still decided on its own. `report.html` shows the same tags as badges.

**Examples.** These are the skill's arguments: start each line with `/multi-llm:multi-llm` in Claude Code, `$multi-llm:multi-llm` in Codex, or `/multi-llm` in OpenCode. Each standalone phase assumes the phases before it have run; the `--full` line is the alternative to running them one by one.

```text
# New plan review: the default behaviour, scope tags are informational
--review-plan plans/feature.md

# Plan review and application with a preserved scope
--review-plan plans/feature.md --scope-policy preserve
--apply-suggestions plans/feature.md --scope-policy preserve

# Task generation, review, correction, and implementation
--generate-tasks plans/feature.md --scope-policy preserve
--review-tasks plans/feature.md --scope-policy preserve
--apply-task-suggestions plans/feature.md --scope-policy preserve
--implement plans/feature.md --scope-policy preserve

# Code review and application in a single unattended command
--improve-code plans/feature.md --scope-policy preserve

# Alternative: preserve scope throughout the full unattended pipeline
--full plans/feature.md --scope-policy preserve --yes

# Commit review and application, without requiring a plan
--improve-commits main..HEAD --scope-policy preserve

# Explicitly allow expansion when applying remaining code-review findings
--apply-code-fixes plans/feature.md --scope-policy expand
```

The `--apply-*` lines would keep their review's `preserve` without the flag; it is spelled out so each line says what it does.

**Which policy a run uses.**

- **A typed `--scope-policy` wins.** An empty or unknown value stops at exit 3 before anything runs. The policy is never a question: under `--yes`, which the Codex and OpenCode entry points pass, one nobody typed is recorded as `[AUTO_RESOLVED] scope-policy: …`.
- **Left out on a new run, it is `expand`.** A new review round, `--generate-tasks` and `--implement` each start their own run, so to preserve the scope through the standalone phases type the flag on each one, as above. The next-phase command a run prints never adds it.
- **Left out on a continuation, it is the one on record.** A review's `--reaggregate` keeps its round's policy; an apply, and any revalidation it runs, keep the review round's (or the apply's own typed override, below); a re-run or `--resume` of a stopped `--generate-tasks`, `--implement`, `--improve-*` or `--full` run keeps the value typed when it started. Repeating the same value is harmless. A review's `--force` resume of the missing models keeps the policy the results it reuses were requested under, and says so in one `Scope policy notice:` line; `--force --rerun-all` starts a fresh round instead, under `expand` or the policy you type.
- **`--improve-*` and `--full` resolve it once** and pass that one value to every step, including the apply `--full` runs itself when a prerequisite is missing. With none typed they pass none, and each step resolves on its own as above. `--force` starts them over, without the earlier choice.

**Changing it after the review.**

- **On an apply, a typed policy overrides the review's** and is kept for that apply's later runs. It changes only which of the findings the review returned may be applied: `--apply-code-fixes plans/feature.md --scope-policy expand` after a `preserve` review applies what that review deferred, and never applies a landed batch twice. It cannot recover suggestions the `preserve` reviewers were never asked for, nor change their tags; the reports name both policies. A completed `--improve-*` run typed again with a different policy runs only its apply again, when that apply deferred something. Changing the policy part-way through an apply rebuilds only the batches that never started; a batch of the earlier run that did not finish stops it at exit 3, naming the ways out.
- **A review round keeps the policy its reviewers were asked under.** A different one typed for a round already under way stops at exit 3 with `[SCOPE_POLICY_CONFLICT]`, and running a completed review again does not start a new round. To ask the reviewers again - for the broader suggestions a `preserve` round never solicited, say - start a new round with the existing controls and the policy you want: `--review-code plans/feature.md --force --rerun-all --scope-policy expand`, or `--improve-code plans/feature.md --force --scope-policy expand`.

**Old and unclear findings.**

- **A review from before `--scope-policy` existed reads as `expand`.** Its findings show `Scope unclear` and apply exactly as before, with no migration and no model call.
- **Under `preserve`, a finding with no valid classification is deferred,** whether it comes from such an old review or from a validation reply that carried none, with the way forward: run a fresh review, or apply with `--scope-policy expand`.
- **A finding changed after validation reads `Scope unclear`** - its description edited in `report.md` or `report.html`, one of its members skipped, or its fix trimmed by the judge - and its tag is marked possibly stale. Under `expand` that changes nothing; under `preserve` the finding is deferred.
- **Missing or vague intent gives `Scope unclear`, not an invented requirement.** Under `preserve` that defers the fix, so state the limits you care about in the plan.

---

Back to the [README](../README.md) · [Commit reviews](commit-reviews.md) · [Per-project configuration](configuration.md)
