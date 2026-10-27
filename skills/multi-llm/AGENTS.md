# Multi-LLM Skill — Development Guide

This guide applies when developing the plugin under `skills/multi-llm/`.
Multi-LLM coordinates review, planning, and implementation through provider
CLIs. Orchestrators prepare work for host-native subagents or the built-in
executor; keep execution mechanics in the shared executor and scheduler.

Paths below are relative to this skill directory unless stated otherwise.
Detailed implementation contracts and regression history live in
[`docs/development/`](../../docs/development/). Read the relevant topic before
changing that subsystem; the routing list at the end names each topic.
`CLAUDE.md` imports this file, so keep it as the shared development entrypoint.

## Development commands

Run these from the **repository root**. Use `uv run --project`, not
`--directory`; the latter has caused duplicated paths.

```bash
# Focused test file
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_filtering.py -v

# Ordinary CI suite, excluding explicitly provisioned integrations
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests -m "not live and not container and not package and not codex_jail"

# Error-handling tests
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests -m robustness

# Check generated host documentation
uv run --project skills/multi-llm -- python skills/multi-llm/hosts/render_capabilities.py --check
```

Choose checks that exercise the changed behavior. The
[development documentation](../../docs/development.md) covers the container
harness and plugin validation; `pyproject.toml` defines the test markers and
normal exclusions. Tests marked `live` need provider CLIs and API access;
container, package, and Codex-jail tests have separate prerequisites.

## Ownership and source of truth

- **`multi_llm.py`** is the public entrypoint. It normalizes positional arguments
  and lazily calls a mode's `main(argv) -> int` in-process. Direct orchestrator
  invocation remains supported for tests and internal re-entry.
- **`modes.py`** owns mode words, aliases, phase names, dispatch targets,
  positional arguments, and composite sequences. Add a mode there, then update
  the generated surfaces. A mode and phase can differ: `review-code` maps to
  `code-review`.
- **`utils/cli_common.py`** owns the shared argparse parent, command builders,
  invocation record, and `MULTI_LLM_RESULT` emitter. Do not redefine shared flags
  in an orchestrator. Regenerate `tests/golden/*_help.txt` when shared help
  changes; a new flag can also require updating the literal argument-hint
  trailer.
- **`instructions/*.md`** owns the host-neutral workflow. `SKILL.md` and
  `hosts/*/shim.md.tmpl` adapt it to hosts. The core's verbs are *spawn subagents*,
  *ask the user*, and *run detached*. Keep concrete host tool names in capability
  data and shims.
- **`hosts/capabilities.yaml`** owns host mechanisms and installation metadata;
  read it through `utils/host_capabilities.py`. Its renderer owns
  `references/host-capabilities.md` and the generated part of
  `docs/hosts.md`. Edit the source and regenerate those documents.
- **`utils/provider_registry.py`** owns configuration layering and typed driver
  accessors. `utils/providers/` owns provider commands, profiles, output channels,
  and usage parsing. Configuration `command` values are metadata; executable
  provider binaries come from built-in adapters.
- **`utils/subagent_executor.py`** invokes materialized tasks;
  **`utils/write_batch_scheduler.py`** owns write scheduling and continuation.
  Both host-native and executor paths must preserve the same task and result
  contracts.
- **`utils/process_control.py`** owns platform-specific process, lock, and
  teardown primitives. Keep `fcntl`, `msvcrt`, and `ctypes` process internals
  there.

Imports must be side-effect free: parsing, `sys.argv` reads, and I/O belong in
`main`, not module scope. Multiple dispatch targets run in one interpreter.

## CLI, paths, and lifecycle contracts

The invocation shape is:

```text
uv run --project <skill-dir> -- python <skill-dir>/multi_llm.py <mode> [positionals] [flags]
```

The mode must be `argv[1]`. Forward remaining arguments as a Python list;
re-parsing a command string loses quoting and paths with spaces. Runtime
invocations obtain their `--frozen --no-dev` flags from
`utils/unattended.uv_run_args()`. Instruction templates use `{UV_RUN_FLAGS}` and
`{SKILL_DIR}`; avoid duplicating those decisions in prose or host wrappers.

Exit codes are uniform: **0** completed invocation, **1** unexpected error,
**2** nothing to do/already complete, **3** needs human input or cannot proceed.
Findings are artifacts and stdout markers, not errors. For example, a completed
code review returns 0 despite HIGH findings; the explicit `--fail-on-high` gate
returns 3. Every invocation emits exactly one `MULTI_LLM_RESULT` line, including
on failure. Composite workflows call phase functions directly rather than
nesting `multi_llm.main` and its result context.

Interactivity and host selection are explicit. Do not infer workflow behavior
from a TTY, `CI`, or a host executable. `--yes` and `--interactive` conflict.
New decision gates need a row in `utils/unattended.ROWS`, a real parser flag in
`DECISION_FLAGS`, instructions for the affected mode and full workflow, and a
recorded resolution when automatically decided. Prerequisite checks belong in
`check_workflow_prerequisites.py`; `--yes` does not bypass them.

Detach only after preflight, before invoking a model. A detached child re-enters
through `uv run --project`, preserving the lockfile and project environment.
`--detach` implies `--yes` and must not be combined with a host's background
primitive. Keep workflow state in `state.json` distinct from process/run state
in `.status.json`; a successful exit or pending marker alone does not prove
that a workflow phase completed.

Keep output paths plan-relative. Preserve symlinks when resolving the skill
directory (`abspath`, not `realpath`). Durable records store a mode, plan path,
and replay flags as a list, never a command containing an absolute plugin-cache
path. Reconstruct commands through the shared builders using the running skill
and the host's command spelling. Preserve the user's plan spelling in displayed
follow-up commands.

## Execution and state invariants

Materialized task files are the delegation contract. Preserve their schemas,
context declarations, output paths, and validation in both execution paths.
Validation uses full model capability; the cheap coordinator role must not
silently become the validator.

**Read/analyze dispatch must preserve its read-only profile.** This applies to
`generate`/`build_command` as well as `agent-readonly`. Unsupported profiles must
be refused explicitly. Do not add a permission bypass to make a provider run;
profile-scoped exceptions and sandbox fallback conditions are specified in the
[provider contract](../../docs/development/providers.md#read-only-provider-contracts)
and guarded by `tests/test_read_only_modes.py`. `agent-write` also requires an
explicit declaration consistent with real file-edit support and prompt
transport. Keep runner environments isolated from the skill's Python venv.

Write retries must account for edits already made. A batch that changed files
before failing is a partial application; do not blindly dispatch its original
prompt again. Use the existing write-safety and backup mechanisms for salvage
and modification, respecting the scope-specific behavior described in the
[write safety](../../docs/development/write-safety.md) and provider references.

Preserve these distinctions in state and reporting:

- A check with findings is `check-findings`, not `verified`. Accepted findings
  remain explicit context for dependent tasks; unrelated failures remain
  failures.
- Deferred manual tasks and tasks waiting on them remain owed. If a manual task
  blocks every remaining code task, implementation must report the gate.
- Declined, partially applied, verified no-op, and completed items have distinct
  recording rules. Resume counts describe the run's items, including earlier
  attempts, rather than only the newly materialized batch file.
- Only `implement` and `apply-code-fixes` update `tracked_files`. Record completed
  batches even when a later batch stops. Exclude the workflow's own plan, task
  list, and artifacts from code tracking and inferred review scope.
- Running a batch and verifying its changes are separate facts. Report checks
  that failed or did not run; do not imply testing from a completed phase.

## Review artifacts and user-visible results

Group IDs are stable 16-character content hashes produced through
`state_manager.py`, never random IDs. Validation statuses are `valid`, `invalid`,
`needs-human-decision`, and `validation_failed`; importance is HIGH, MEDIUM, LOW.
Code-review reports number issues, but decisions act on groups: resolve through
stable group hashes, never a displayed issue's list position.

Aggregation must tolerate malformed entries without losing other reviewers'
results. Use `schema_validator.drop_non_record_entries` before stamping result
records. Reaggregation must exclude non-model JSON artifacts and respect the
recorded reviewer identities and current-run file baselines. A clean review
with zero findings still writes the empty artifacts its apply phase needs.

`report_data.json` preserves the inputs needed to regenerate HTML decisions.
Keep phase/directory aliases consistent across `output_handler`, state readers,
and prior-change records. Provider logs must identify the actual provider,
final outcome, and output source; unavailable usage is absent, not zero.

Result fields have different purposes: `next` is a mechanical continuation;
`user_next_step` is a command for the user. Recommendations must not become
commands the host automatically executes. Preserve supplied progress lines and
ready Markdown sections through host relay. Manual tasks can accompany either
a next-command block or a completed-workflow block. Use `utils/text_clip.clip`
for bounded model text, not raw string slicing.

## Read the relevant detailed contract

These references preserve implementation details and the incidents behind the
rules. Each has a contents list; load the sections relevant to the change.

- **CLI parsing, modes, utilities, result fields:**
  [Dispatcher and result protocol](../../docs/development/dispatcher.md).
- **Background execution, locks, uv, retry persistence:**
  [Detached runs and recovery](../../docs/development/runtime.md).
- **Decision flags, interactivity, prerequisite gates:**
  [Unattended decisions](../../docs/development/unattended.md).
- **Shims, capabilities, installation, packaging, version updates:**
  [Host integration](../../docs/development/hosts.md).
- **Task generation, full/composite workflows, commit reviews, ask, implementation state:**
  [Workflow orchestration](../../docs/development/workflows.md).
- **Task schemas, validation context, executor dispatch, runner pinning:**
  [Delegation and validation](../../docs/development/delegation.md).
- **Write retries, partial/declined results, batch counts and parsing:**
  [Write scheduling](../../docs/development/write-scheduler.md).
- **Containment, backups, rollback, code tracking:**
  [Write boundaries and tracking](../../docs/development/write-safety.md).
- **Check findings, accepted checks, human-task answers and dependencies:**
  [Checks and human task gates](../../docs/development/checks.md).
- **Adapters, configuration, permission profiles, output and environments:**
  [Provider contracts](../../docs/development/providers.md).
- **Grouping, snapshots, aggregation, result recovery, report decisions:**
  [Review artifacts and reports](../../docs/development/review-results.md).
- **Status views, progress, ETA, command rendering:**
  [Status views and progress](../../docs/development/status.md).
- **Tool supervision, stop hooks, report relay, native apply counts:**
  [Host result delivery](../../docs/development/host-reporting.md).

## Maintaining this guide

Keep broadly applicable rules and routing here. Put subsystem details and dated
regression narratives in the relevant reference, with links to owning code and
existing tests. Avoid appending another incident's full history to a rule.
Update affected links and documentation guards when moving a contract; the
reference should remain reachable without loading the entire development manual.
