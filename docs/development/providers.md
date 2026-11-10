# Provider contracts and configuration

Read before changing provider adapters, permission profiles, sandbox fallbacks, runner configuration, output channels, usage, or child environments.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Adding a provider adapter](#adding-a-provider-adapter)
- [The provider base modules](#the-provider-base-modules)
- [Configuration and runner resolution](#configuration-and-runner-resolution)
  - [The overall deadline](#the-overall-deadline)
- [Read-only provider contracts](#read-only-provider-contracts)
- [Model catalog snapshots](#model-catalog-snapshots)
- [Provider output channels](#provider-output-channels)
- [Capturing a CLI's streams](#capturing-a-clis-streams)
  - [A CLI that loops](#a-cli-that-loops)
- [Reading a capture](#reading-a-capture)
  - [Launch diagnostics](#launch-diagnostics)
- [Usage reporting](#usage-reporting)
- [Child Python environments](#child-python-environments)
- [Private config roots](#private-config-roots)
- [Provider outcome logs](#provider-outcome-logs)
- [Version floors and measured versions](#version-floors-and-measured-versions)

## Adding a provider adapter

The roster is derived. `utils/providers/__init__.py` discovers every concrete `LLMProvider`
subclass defined in `utils/providers/`, and `provider_registry._PROVIDERS` instantiates them,
sorted by `name`; `utils.providers.__all__` and `init_config.py` derive their lists from it. Adding
the module registers the adapter, and no code names it anywhere else. Adapter behaviour stays
in code, not in `providers.yaml`, because a cloned repository can layer its own config over
that file.

What cannot be derived stays hand-kept: measured config rows, pinned test tables, fixtures
and docs. Each one has a test that compares it with the registry and fails naming the table,
its file and the row to add or remove (`tests/roster_check.py`;
`tests/test_provider_roster.py` holds the config, report and docs checks).

**Measure first, offline.** Spend nothing: run the CLI's `--version` and `--help`, then point
it at the localhost stubs with a fake key (`stub-key-not-a-credential`) and a throwaway
`HOME` plus the CLI's own config directory. Use `tests/container/stub_server.py --port P
--log L --scenario S` for answers and `tests/container/failserver.py --port P --mode M` for
failures (`401`, `404`, `429`, `429ra`, `500`, `ctx`, `empty`, `stall` and the truncation
modes). The reproducible place to run them is the pinned image (`tests/container/Dockerfile`
at the `cli-versions.lock` pins, `--network none`). The `method:` paragraph of each
`tests/fixtures/<provider>/manifest.yaml` records the exact vehicle a sibling used; copy the
closest one. `tests/fixtures/row_capture/` is the general one: a driver that takes one row
per container for any wrapped CLI (`run_row.sh <cli> <row> <stub|fail|script> [mode]`), with
its argv held to `build_command` by `test_row_capture_kit.py`; add the CLI to its `CLIS`.

Then work through the steps in order. Each test named fails until its step is done.

| Step | Where | What to add | Fails until then |
| --- | --- | --- | --- |
| 1 | `utils/providers/<name>.py` | One concrete `LLMProvider` subclass with the four abstract members: `name`, `is_available` (`shutil.which` of the binary), `build_command`, `parse_output`. Do not override `default_timeout`: the timeout is the `providers.yaml` row of step 2, and every adapter inherits the one fallback, `DEFAULT_TIMEOUT_S`. Add `EFFORT_LADDER = ReasoningLadder(provider=..., rungs=...)` and `REASONING_EFFORTS = EFFORT_LADDER.accepted`; a CLI with no effort control declares `ReasoningLadder(provider=..., declined="<the measurement>")` and no `REASONING_EFFORTS`. Decide each declaration the argv shows: `prompt_argv_index`, `supported_prompt_transports`, `agent_profiles`, `supports_file_output`, `runs_shell_in_generate`, `internal_retries`; and from the captures, `LAUNCH_DIAGNOSTICS` ([launch diagnostics](#launch-diagnostics)) and, for a JSON envelope, `ENVELOPE_SIGNATURE`. Copy the closest sibling: `grok.py` for a JSON envelope, `kilocode.py` for plain stdout. | `test_reasoning_effort_ladder.py`; `test_provider_registry.py` (no `default_timeout` override); `test_provider_housekeeping.py`: the prompt index against the argv, a `STOPS_FILE_WRITES` or `REVIEWED_STILL_WRITES` row for each restriction-shaped argv atom, a `SHELL_EVIDENCE` row for a non-`None` shell declaration |
| 2 | `providers.yaml`, `tests/read_only_modes.yaml` | A `providers.<name>` block (`default_timeout`, `max_concurrent`, `models`, the first being the default) in `providers.yaml`, inserted where it should sit in `--init`'s order; and a `read_only_modes.<name>` row in `tests/read_only_modes.yaml`: `mechanism: null` and `probed: null` until probed ([read-only contracts](#read-only-provider-contracts)). If the CLI lists its models, add a catalog route ([model catalog snapshots](#model-catalog-snapshots)); otherwise an `UNCHECKED` row naming why. | `test_provider_roster.py`, `test_read_only_modes.py`, `test_model_catalogs.py` |
| 3 | `templates/config/providers.override.yaml` | Regenerate it: from `skills/multi-llm/`, `uv run --project . -- python -c "import init_config as c; c.TEMPLATE_PATH.write_text(c.build_template_text())"`. | `test_provider_registry.py::TestTemplateBaseParity` |
| 4 | `ENV_ALLOWLIST_PROFILES` in `utils/providers/child_env.py` | A `RunnerEnvProfile`: CA family (`node` or `native`), the env prefixes the CLI owns, its home and config variable names. | `test_env_allowlist.py` |
| 5 | Test tables | `MODELS` in `test_provider_housekeeping.py`; `PRE_L5_ARGV` and `PRE_L5_ARGV_MODELS` in `test_provider_timeouts.py` (also read by `test_structured_output.py`); `TRANSPORT_CASES` in `test_provider_contract.py`; a `ROWS` entry in `test_answer_text.py`. Each holds a real model id from step 2, never a placeholder. The `ROWS` entry passes only when the answer is readable: a clean `{"type": "result"}` or `{"response": ...}` envelope, plain stdout (`answer_framing = "plain"`), or a `salvage_text` hook. | those files |
| 6 | `TestDeclaredOutputChannel.CANNOT_WRITE_FILES` in `test_provider_contract.py`; `NO_WRITE_TOOL_AT_ALL` in `test_provider_housekeeping.py` | **Only for an adapter declaring `supports_file_output = False`** (it answers on stdout): its name in `CANNOT_WRITE_FILES`. **Also, only when no argv atom causes the declaration** (step 1 added no `STOPS_FILE_WRITES` row, because the CLI ships no write tool): a `NO_WRITE_TOOL_AT_ALL` row naming the probe that measured it ([output channels](#provider-output-channels)). | `test_provider_contract.py::TestDeclaredOutputChannel`; `test_provider_housekeeping.py::TestReadOnlyArgvMatchesTheDeclaration` |
| 7 | `tests/fixtures/<name>/` | `manifest.yaml` and the captured streams, one cell per canonical row (`success`, `truncated`, ... `context-overflow`; a measured `not_applicable` counts), each graded row with its `expect.lane_status` ([reading a capture](#reading-a-capture); the failing test prints the values). Until they are captured, a `FIXTURES_PENDING` row in `test_provider_fixture_rows.py` with the measured reason. | `test_provider_fixture_rows.py` |
| 8 | `_PROVIDER_ICON_KEYS` in `utils/html_report_generator.py` | A row naming an `_ICON_DEFS` key: the CLI's own mark, or a letter icon `(None, letter, colour)`. A bundled mark also goes in `docs/attribution.md`. | `test_provider_roster.py` |
| 9 | Docs | A row in the `docs/providers.md` table (key, binary, link). In the repository root, the harness link and count in `README.md` and the count in `.claude-plugin/plugin.json` and `marketplace.json`. The provider→`binary` list and the declaration-order list in `instructions/init-config.md`, and the base-order list in `docs/configuration.md`. The "Not affiliated" names in `docs/attribution.md` are not checked. | `test_provider_roster.py` |
| 10 | `tests/test_provider_<name>.py` | The adapter's own cases: argv per profile, parse edge cases, failure classification over the fixtures. | (new) |
| 11 | Packaging | `git add -N` the new files, because payload builds refuse untracked inputs. Then `make claude-regen codex-regen` and `git add -N` the copies of the new files the trees gained. A release also needs a version bump ([package distribution](hosts.md#package-distribution)). | `test_claude_package.py`, `test_codex_package.py` |

The container lanes are optional and separate: a `redirect.yaml` row, a `cli-versions.lock` pin,
`expected_argv.yaml` (`seed_expected_argv.py`), a golden bundle (`seed_container_golden.py`)
and a host assignment in `host_matrix.yaml`, as `tests/container/README.md` describes. A CLI
given a provider-lane row also needs its golden baseline (`test_golden_container_bundle.py`),
and a `lane: fake` row needs its `mock_llm.py` identity (`test_fake_lane_roster.py`).

Verify with the roster's own files before the broad run:

```bash
uv run --project skills/multi-llm -- pytest -q -p no:cacheprovider -n 4 --dist loadfile \
  skills/multi-llm/tests/test_provider_roster.py skills/multi-llm/tests/test_provider_housekeeping.py \
  skills/multi-llm/tests/test_provider_contract.py skills/multi-llm/tests/test_read_only_modes.py \
  skills/multi-llm/tests/test_env_allowlist.py skills/multi-llm/tests/test_reasoning_effort_ladder.py \
  skills/multi-llm/tests/test_provider_timeouts.py skills/multi-llm/tests/test_structured_output.py \
  skills/multi-llm/tests/test_answer_text.py skills/multi-llm/tests/test_provider_fixture_rows.py \
  skills/multi-llm/tests/test_provider_registry.py
```

## The provider base modules

`utils/providers/base.py` is `LLMProvider` and its fallback timeout, `DEFAULT_TIMEOUT_S`.
What adapters and callers share besides the class lives in seven modules beside it, one job
each, listed lowest first: a module imports only the ones above it in this table, and none
imports `base` at run time.

| Module | What it holds |
| --- | --- |
| `types.py` | The three invocation profiles, the structured-output retry budget, the usage and config-auth words, `InternalRetryEnvelope`, `StructuredSchema` and its refusals, `SetupHints`, `Availability` and `AvailabilityReport`. |
| `reasoning.py` | The reasoning-effort ladder and `ReasoningLadder`, plus `split_reasoning_effort`. |
| `failure_classifier.py` | The marker tables, `classify_streams` and `match_streams`, `exit_zero_failure_category`, and `LLMProvider`'s two default pattern sets (`AUTH_ERROR_PATTERNS`, `RATE_LIMIT_PATTERNS`). |
| `stdout_shape.py` | `envelope_signature`, `read_stdout`, `structural_stdout` and `diagnostic_streams` ([reading a capture](#reading-a-capture)). |
| `child_env.py` | Git's non-interactive environment, the strip and backend policies, the env allowlist and its matrix, and `build_child_env`. |
| `version_gate.py` | `version_tuple`, `version_verdict` and the help-token gate (`read_help_tokens`). |
| `sandbox_rungs.py` | The sandbox fallback's policies and rungs, the write posture, and the `[SANDBOX_*]` notices. |

The classifier is pure: it imports no provider module, and no `utils` module but
`retry_ladder` (the status vocabulary) and `process_state` (the cache registry), both
standard-library only, so it can be read and tested alone. `stdout_shape` stays out of it
because it needs `json_extractor`'s ANSI pattern.

`base` re-exports every name of the seven modules, **for reading only** (its `_MOVED`
table): `base.X` and `from .base import X` return the owner's current object, so no importer
changed. A patch or assignment of a moved name on `base` reaches no code, because `LLMProvider`
and the owners read each name through its owner (`failure_classifier.classify_streams`).
Patch a moved name where it lives; `tests/test_provider_base_moved_names.py` fails any test
that patches one on `base`, and pins the table, the layering and the classifier's purity.

## Configuration and runner resolution

**Config layering** (`provider_registry.load_config`): base `providers.yaml` → auto-discovered
`<git-root>/.multi-llm/providers.yaml` → `MULTI_LLM_PROVIDERS_CONFIG` env override. The auto-discovered layer deep-merges
its **full** contents — including a `providers:` block — over base, identical to the env
layer (no longer filtered to selection keys). Deep-merge, lists replace, `None`
skipped. Discovery anchors on the optional `anchor` (plan path threaded from
`resolve_models`) then CWD. Present-but-invalid explicit
overrides fail fast (`ConfigError`), a file that is not UTF-8 text included (`yaml_io.load_file`
raises it as a parse error); `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` restores warn-and-skip. A
merged `providers:` block that is not a mapping (a list, a scalar) is replaced by the shipped one
with a warning, as a single malformed provider entry gets its base definition back: otherwise every
per-provider read would fall back to an adapter default that no layer states.

**Config caching.** The merged config is cached per process, one entry per set of layer files
a load read, keyed on each file's path and stat signature (device, inode, size, mtime, ctime of
the file it leads to) plus the effective permissive flag (`_ConfigCache`). `/repo` and `/repo/plans`
share an entry; two repositories, or a repository and a directory outside any, never do; and a
layer that changes, appears or disappears is read again on the next access from any anchor, so
a lowered `max_concurrent*` cap reaches the ordinary accessors at once. A load is cached only when
every file it read was settled (older than `yaml_io.SETTLE_S`) before the read and kept its
signature through it. Each load is one snapshot: the merged dict plus the layer that supplied
each `driver.*` value, so a value and its provenance never come from two loads. Code that runs
or logs a value with its source reads both in one call (`get_verify_settings` for
`driver.verify_command`, `_driver_view` inside the registry). Code that reads several values for
one decision loads once (`load_config_snapshot(anchor=...)`) and passes `snapshot=` to the accessors
that take it, so the values come from the same files for one git lookup; the write scheduler does
this once per batch. A snapshot answers for the anchor it was loaded with, so a read takes
`snapshot=` or `anchor=`, never both: `require_one_config_source` raises on the pair rather than
let a snapshot of one repository answer a read anchored on another, since checking that the two
agree would cost the git lookup the snapshot saves. The anchor guard in
`tests/test_provider_registry.py` counts a `snapshot=` as anchored only when it traces to an
anchored `load_config_snapshot` in the same function or arrives through a reader's own `snapshot`
parameter (`SNAPSHOT_READERS`). Which layer files a load reads is
never cached: every access asks git for the project root (`git rev-parse --show-toplevel`, about
1.5 ms) and checks whether `.multi-llm/providers.yaml` is there. git's answer depends on more than
a process can cheaply watch (a re-pointed checkout symlink, a `.git` created or removed,
`core.worktree` in a gitfile target's `config` or in `config.worktree`, `safe.directory` and the
rest of the global config), and the project layer's `verify_command` is executed, so a remembered
answer could run another repository's command. `registry._config = None` drops everything
(`set_run_anchor`, `--init`, the tests).

**YAML parsing** goes through `utils/yaml_io.py` and nowhere else
(`tests/test_yaml_io.py` enforces it): `SAFE_LOADER` is libyaml's `CSafeLoader` when the PyYAML
wheel has it, else `SafeLoader`, about 0.6 ms against 16 ms for `providers.yaml`. The two
build equal documents for every YAML file in the repository; only parse-error marks differ
(libyaml's carry no snippet). `load_file` caches a parse per resolved path, valid while the open
file's `fstat` (device, inode, size, mtime, ctime) matches. It returns deep copies, and admits a
parse only when that signature was settled (older than `SETTLE_S`) when taken, before the read,
and is unchanged after it, because coarse timestamps let a same-size rewrite keep its signature. `host_capabilities.load` keeps its own per-root cache and parses
under a lock, so concurrent first readers share one parse.

Config `command` is metadata-only and NEVER executed (binaries are hardcoded in
`utils/providers/`); a merged provider name with no hardcoded adapter is ignored at runtime.
Scaffold via `init_config.py` (`--init`, auto-detects installed CLIs).

**`driver:` is the fourth top-level block** (`subagent_runner`,
`subagent_runner_cheap`, `max_concurrent_subagents`, `max_concurrent_write_subagents`, `subagent_timeout_s`,
`subagent_retries`, `verify_command`, `verify_timeout_s`, `write_paths`),
fully populated in base so nothing has to guess, read only through the typed accessors
(`get_driver_config`, `get_subagent_runner`, `get_subagent_timeout`, `get_max_concurrent_subagents`,
`get_max_concurrent_write_subagents`, `get_verify_command`, `get_verify_timeout_s`, `get_write_paths`,
`get_subagent_retries`, `get_driver_provenance`) and never by indexing `config["driver"]`.

`verify_command` is the one key whose value is **executed** — a shell string, by the write
scheduler only, after a code phase's last batch — so it is optional (`None` = this
repo declares no check) and `docs/configuration.md`'s safety section names the carve-out. The two
write keys are separate from their read siblings on purpose: write concurrency defaults to **1**
because the batchers partition by artifact, and `write_paths` is a nested
`allow`/`deny` mapping the `--init` template renders
nested too.

Validation is fail-fast and only for an **explicit override layer**: an unknown key under
`driver:` is rejected rather than ignored (a typo is not a silent no-op), a runner spec
must be an explicit `provider:model` whose provider has a built-in adapter — **raising, where
the fan-out path warns-and-filters**, because a filtered runner degrades to "no runner configured"
and resurfaces mid-phase. The `--init` template carries the same block fully commented
(byte-diffed against base by CI, so a base default change forces a regeneration).

**`resolve_subagent_runner`'s ladder (D3o) is explicit `env`/`project` →
`session` → the host's `default_subagent_runner` →
`base`/`default`.** The `session` rung is the model the
host session itself is running: `MULTI_LLM_HOST_SESSION_MODEL` (`provider_registry.HOST_SESSION_MODEL_ENV`, set by the opencode
plugin from the session's newest assistant message, `providerID/modelID` **and** that message's
`variant` as an `:effort` suffix — a session opened at
`--variant low` used to dispatch every subagent at the model's default effort, and the
adapter's per-model cap (`opencode.EFFORT_LADDER.model_caps`) is asked at the one place `--variant` is
emitted, so the model measured to *hang* on one loses the suffix with a warning whether it came from
the picker or from a typed `--models` spec) spelled `<session_runner_provider>:<value>`, and only
when the host's capability entry names a `session_runner_provider` (`opencode` on opencode;
`codex` on codex, where the plugin's MCP server exports the per-call turn metadata's
`model` — validated as a plain id, no colon or whitespace, re-read every call — at
codex's configured default effort, since that metadata carries no effort key (measured codex
0.156.1, 2026-09-25); `null` on claude-code, where nothing hands the dispatcher the
session's model).

It is implicit — below an operator's opinion, above the credential-free host floor — and it is the
one rung that **falls through at preflight instead of refusing**: `task_batch._preflight_batch_runners` prints
`[exec-subagents] session model <spec> failed preflight (<reason_code>); falling back to <spec> (<layer>)`, resolves again with `skip_session=True`, and records
`runner_fallback: {from, reason, to, layer}` in `subagent_batch_report.json`; every other rung's refusal is unchanged; the
nested-exec probe's deadline is `NESTED_PROBE_TIMEOUT_S = 30 s` (was an unmeasured 120 s: the deadline only
decides how long an *inconclusive* probe stalls the batch — the refusing verdicts land before the
first token — and a slow session model paid the full 120 s to record the same
`unknown`); `run_tasks` dispatches longest prompt first
(`_dispatch_order`, stable on file order; ids, `batch_index`, output paths, the
report's `tasks[]` and the runner listing keep file order, and a `dispatch order (longest first): …`
line prints only when it differs) so the biggest batch never runs alone at the end — on the
2026-09-13 run file-order dispatch left a 1076 s batch running solo for the last ten minutes; and
the session's model is exempt from the advisory "not in the configured catalog" warning — by rung
(`session`) **and by model** (`_is_session_model`: the same spec reaches
`preflight_runner` as the generate-tasks orchestrator's `driver.subagent_runner` and as a
re-invocation's `tasks[<id>].runner (recorded from a previous attempt)`, and each printed the warning the rung had just been excused
from), because that catalog is the operator's fan-out list and the live `_detect_model` leg
still judges the session's model; with no session model set the warning is exactly as it was.

Measured 2026-09-11: a `--quick` review from an opencode session on
`opencode/nemotron-3-ultra-free` validated on `opencode/big-pickle` because nothing named the session's
model, which is the question this rung answers.

### The overall deadline

Each slot's budget (its provider's timeout, or the run's `--timeout`) starts once the slot holds
its provider's `max_concurrent` semaphore and then the global `--max-parallel` one, so it can
wait behind both, and the two waits add up. Until a slot of provider `p` starts, at every moment
either the global gate is full or every slot holding `p`'s gate is running, so it starts by
`(W − W_p) / max_parallel + (W_p − its budget) / p's gate` (`W` sums every slot's budget, `W_p`
those of `p`'s slots). The fan-out's overall deadline (`concurrency.overall_deadline`, read by
`review_orchestrator_base.run_models_concurrent`) is the latest such finish, plus one largest
budget of headroom (kill grace, drain, a launch turn waited for inside the gates) and the launch
stagger. One slot is `2 × budget + stagger`; kilocode's six default models at `max_concurrent: 2`
come to 5400 s, and the shipped default model list (eleven slots, `--max-parallel 5`) to 7444 s.
`tests/test_overall_deadline.py` runs mixed schedules through the real gates on a virtual clock
and checks every one finishes inside it. The deadline harvests every finished slot before it
cancels the rest. The executor's batch passes its own budget, one task budget per task, because
backpressure can narrow a provider to one slot mid-batch (`task_batch`), and
`run_tasks_concurrent` counts that as one slot.

## Read-only provider contracts

**A read/analyze dispatch never gets write access.** An adapter opts into `agent-readonly` by
implementing `build_agent_command(prompt, model, profile)` with the runner's **own** denial mechanism (codex
`--sandbox read-only`, opencode `--agent plan`, claude
`--allowed-tools`/`--disallowed-tools`) and declaring it in `agent_profiles`. There is
no fabricated read-only mode: an adapter without one raises `ProfileUnsupported` and is refused as
a runner with the reason named. `tests/test_provider_contract.py` (and `tests/test_subagent_executor.py`) cross-check
declaration against implementation in both directions, for profiles **and** for
`parse_usage`/`usage_support`, so the capability data cannot drift from the code.

The write profile follows the identical rule one level up: `agent-write` must agree with
`supports_file_edits`, must render a real command, and must offer a non-argv transport — and the
set of adapters declaring it is pinned to exactly `claude-code`, `codex`,
`opencode`, because widening it hands pre-approved edit access to a user's repository.
**The same rule binds `generate`, and the rule is profile-scoped rather than
flag-scoped** (§3.5): `build_command` *is* the Lane A review/ask fan-out, and every phase
that reaches it — `ask`, `review-plan`, `review-tasks`,
`review-code`, task generation, consolidation — consumes a *document*, never a diff.

So no adapter may reach a write bypass (`--yolo`,
`-f`/`--force`, `--yes-always`, `--auto-approve`,
`bypassPermissions`, `--dangerously-*`, `GOOSE_MODE=auto`) from it, which is exactly
what cursor's `-f` did before `--trust --mode ask` replaced it.

`danger-full-access` — codex's third sandbox policy, a *value* rather than a flag — is in the same
negative set, and it is the **one dispatch-time override** the guard admits (D6, 2026-09-17): never
in any rendered default, only through the adapter's `apply_sandbox_fallback` behind a
`sandbox_fallback=<rung id>` kwarg that `sandbox_refusal` (fan-out) and `_detect_sandbox`
(executor) set when all four hold — the D6 probe verdict is `unavailable` (never
`unknown`), `providers.codex.sandbox_fallback` permits the rung, the profile is
`generate` or `agent-readonly`, and the adapter declares the hook (codex alone) —
while `agent-write` raises rather than renders it on every rung but two narrow ones.

Since 2026-09-30 (plan codex-sandbox-unavailable-findings-and-fixes step 3, option (b), a user
decision) `agent-write` renders `danger-full-access` inside the plugin's Landlock jail
and nowhere else by default: its ladder is `WRITE_FALLBACK_RUNGS = (PLUGIN_LANDLOCK_RUNG,)`, reached through
`write_rung_allowed` (`auto` alone; `WRITE_RUNG_OPT_IN` in
`utils/providers/codex.py` is the code-level opt-in), only when codex's own `workspace-write` probe
said `unavailable`, and only with a grant whose scope the rung serves —
`plan-file` for `apply-suggestions` (one file rule on the plan: in-place edits only,
so `sed -i`, temp-file-and-rename saves and `mv` are denied) and
`tasks` for `apply-task-suggestions` (a dir rule on the default `plans/<stem>/tasks/`,
else a file rule on the tasks file); `repo-wide` for `apply-code-fixes` and
`implement` since 2026-10-02 (plan step 3b, a user decision), and since 2026-10-05 (step
3b revised, another user decision: "it's ok to give .git and .multi-llm directory access; backing up
directories before each batch is too resource intensive") **one required dir rule on the repository
root** (`landlock_dispatch.repo_wide_entry`) — the whole repository is writable inside the jail,
`.git`, `.codex`, `.multi-llm` and this run's output folder
included, nothing is backed up or restored around a batch, and the developer message tells the model
to leave those alone (instruction-level only; `REPO_WIDE_HANDS_OFF_NAMES`).

The per-entry enumeration it replaces (`repo_wide_plan`, the protected set,
`extra_protected_paths`, the F6 refusal of a declared new top-level entry, `_refuse_declared_sets_before_batch_one`)
kept the root unruled, so the racing re-run's single root-level `index.html` was refused
before any dispatch (`plans/2026-10-04-racing-rerun/FINDINGS-codex.md` issues 2–3); Landlock is allow-only and a rule inherits
down the tree, so a root rule cannot exclude `.git`.

What is left is detective: `driver.write_paths` pre-dispatch and post-run (a jailed write to
`.multi-llm/providers.yaml` lands and the batch then fails on the deny rule), the undeclared-write sweep,
and one warn-only `[GIT_CONFIG_CHANGED]` line per batch whose `.git/config` hash or
`.git/hooks/` listing moved (`git_utils.git_metadata_fingerprint`, taken before dispatch, no copies,
never a failed batch); the plugin's own git calls after a batch run nothing the batch could have
planted — every runtime git argv is `git_utils.git_argv` (`-c core.fsmonitor=false -c core.hooksPath=/dev/null -c log.showSignature=false`, the diff
prefixes pinned
(`diff.noprefix`/`mnemonicPrefix`/`srcPrefix`/`dstPrefix`/`relative`,
since a planted prefix emptied `capture_diff_hunks` — review fixes 2026-10-05),
`--no-ext-diff --no-textconv` on diffs, and per call — read from the repository the command runs in, so
pass `cwd=` — an empty
`-c filter.<n>.clean=`/`smudge=`/`process=` plus `required=false` for
every filter driver the repo's **own** config defines (scope
`local`/`worktree`, includes resolved; a global `filter.lfs` is
the user's and stays) and `core.attributesFile=/dev/null` when that config names one, because
`status`, `diff`, `add` and `hash-object`
all run a clean filter bound by `.git/info/attributes` (measured git 2.43); `tests/test_git_utils.py`
fails any other `["git", …]` literal, `_hardened_argv`'s own excepted), while the
user's own git still honours the planted config, which the line exists to say; the fingerprint also
covers `.git/info/attributes`, `.git/info/exclude`, `config.worktree` and every
`include.path` target.

The rung probe hands over no target in the tree — its `$TMPDIR` write is the control (a
control on the first sorted directory failed on plain permissions and wrote into the user's tree on
every `status` poll, 2026-10-02) — and moves the canary's negative controls to a
directory outside the repository (`$MULTI_LLM_CANARY_WORKDIR`, beside the slot `TMPDIR`),
because the root rule makes the in-repo `.multi-llm/tmp` writable; step 1b's measurement
(2026-10-02, codex 0.160.0) adds `UV_CACHE_DIR` and a prepended `PYTEST_ADDOPTS`
`cache_dir` under the slot's `TMPDIR` for this scope (prepended to an
inherited `PYTEST_ADDOPTS` too, `merge_env_redirects`, never replacing it), while
`/tmp` and `/dev/shm` stay denied and the developer message says so.

A failed rung probe is refused, never walked on.

The one other write render is F5 (step 6, also a user decision): with the codex chat's own
`danger-full-access` posture (`MULTI_LLM_HOST_SANDBOX_POSTURE`, exported per call by the codex MCP server,
never inherited) and only after the jail could not take a `plan-file` or
`tasks` dispatch — a failed, absent or skipped rung; `repo-wide` never
reaches the uncontained rung by consent (`WRITE_SCOPES_BY_CONSENT`: step 3b put it on the jail alone,
the safer reading) — `write_consent_allows` lets `agent-write` take the uncontained rung
under `auto`, printing `[SANDBOX_BYPASSED]` naming F5; never under
`contained` or `off`, never ahead of the jail, never on a walk that
measured nothing.

Since 2026-09-22 that override is the **last rung of a ladder** (two rungs until 2026-09-30, three
since), not the only one: rung 1 `landlock` renders `--sandbox read-only -c features.use_legacy_landlock=true` — codex's
second Linux backend, which needs no user namespace, taken only when its pinned probe's canary write
is *denied* — and is a real kernel boundary, reported `[SANDBOX_FALLBACK]` /
`sandbox: fallback-landlock` in `models_served` / `sandboxed by fallback rung: landlock` on the report's model
labels / `sandbox-fallback landlock <specs>` in `status`, never `[SANDBOX_BYPASSED]`, and exempt
from the porcelain check, which `review_orchestrator_base.porcelain_check_applies` fires for the uncontained rung id alone.

Since plan §4 (validation-context-by-path-and-codex-0156-fallout, 2026-09-24) that check compares
per-path baselines (`porcelain_snapshot`: `git status --porcelain -z --untracked-files=all`, one `git hash-object --no-filters --stdin-paths` over
the dirty set minus the plugin's own trees, blob id / `symlink:` /
`absent` / `unreadable` plus the exec bit, degrading to path + status above
50 MB), and `unexpected_writes` walks the union of both, so a write to an already-dirty file, a
deleted untracked file and a restored dirty file are all seen; its failure names how many other
boundary-less slots of the fan-out, **or any other process writing to this repository**, could also
have written the change, because it cannot tell who did.

The one exit is the slot's own audited action stream (option B: `audit_readonly_stream` on the
adapter — codex only, `utils/providers/codex_stream_audit.py`, a fail-closed classifier over every
`command_execution`/`file_change`/`mcp_tool_call`/`collab_tool_call`/`web_search`
item of the full in-memory stream, run in `llm_client` before the excerpt is bounded,
ANDed across every attempt): a clean combined verdict downgrades the failure to a
`[SANDBOX_BYPASSED] … not attributed` warning on `progress.models[spec].warning`.

That exemption is **on** (`readonly_audit.STREAM_AUDIT_EXEMPTION_ENABLED = True`, enabled 2026-09-25 after the §6.2 live capture
classified `web_search`'s `other`/`text_result` fields); setting
it `False` keeps the verdict computed and recorded on every uncontained attempt but
makes it change nothing; the last rung `unsandboxed` is `danger-full-access`.

Between them, since 2026-09-30, sits `plugin-landlock` (plan
codex-sandbox-unavailable-findings-and-fixes §6 F1, step 2), walked only after codex's own probe
said `unavailable`: `utils/landlock_exec.py`, run as `sys.executable -I -S`, applies a
Landlock ruleset to itself and `exec`s codex under `--sandbox danger-full-access` with the
allow-list `utils/landlock_dispatch.py` builds (`$CODEX_HOME`, a per-attempt 0700
`TMPDIR` under the parent's temp dir, `/dev/null`, optional
`/dev/ptmx` + `/dev/pts`) and the env redirects under that
`TMPDIR` step 1c measured (`GIT_OPTIONAL_LOCKS=0`, `PYTHONPYCACHEPREFIX`,
`npm_config_cache`); below Landlock ABI 3, or with no Landlock at all (the Docker image), the
wrapper exits 126 and the walk moves on.

Its own probe runs `codex sandbox -c sandbox_mode="danger-full-access" -- sh -c <canary>` inside the jail with negative, truncation and positive
controls.

Its grade is `incidental`, never contained (`contained=False`,
`SANDBOX_CONTAINMENT_INCIDENTAL`): it stops stray content writes, not a model set on escaping — pathname Unix
sockets (`systemd-run --user`, D-Bus, Docker) write through it, `$CODEX_HOME` is
writable, `chmod`/`utimes` pass and the network is not confined — so
`contained` never reaches it (`sandbox_rung_allowed`), `auto` does, and
every surface says which it is: `[SANDBOX_INCIDENTAL]` (never `[SANDBOX_BYPASSED]` or
`[SANDBOX_FALLBACK]`), `sandbox: fallback-plugin-landlock` in `models_served`, `stray writes stopped by plugin-landlock; not a boundary against the model` on
the report's model labels, `stray-writes-only plugin-landlock <specs>` in `status`.

A fan-out slot on it skips the porcelain check (`porcelain_check_applies` still fires for
`unsandboxed` alone, by decision) and runs the mode check (`[MODE_CHECK]`: a
flipped executable bit on a tracked file, fail-closed unless the stream audit clears it) and the
home check (`[CODEX_HOME_CHECK]`: a change to `$CODEX_HOME`'s config, instruction or
plugin files — fail-closed in the fan-out, warn-only in the executor lane). A fan-out
attempt that raises returns no `details["home_check"]`, so the lane (`dispatch_with_ladder` on
the review lanes, the single dispatch `run_single_model` makes with no output directory, ask's
single-shot `ask_one`) compares the home of every jailed child the attempt armed (`llm_client.armed_homes_window`, through
`review_orchestrator_base.raised_attempt_home_records`) against the invoker's own
before-snapshot, and a home it cannot compare is reported as not checked and fails the slot
closed.

The four fan-out lanes (ask's single-shot slot, review-code's `run_single_review`, and both lanes
of `run_single_model`) run the gate and these checks through one object,
`utils/slot_guard.SlotGuard`. `begin()` asks `sandbox_refusal`, prints the refusal or the rung's
one line and takes the porcelain and mode before-snapshots, all ahead of the dispatch and each in
a worker thread. `finish()` runs, after the last attempt, the porcelain verdict, the jail's own
lines (for ask; the ladder loop prints them per attempt) and the mode and home checks, and returns
the one reason the slot fails with. Each lane keeps only what it does with that reason, and
`tests/test_slot_guard_parity.py` pins every outcome on every lane. The porcelain check's ignore
list (`fanout_ignore_prefixes`) follows the plan's output folder through symlinks, and an
unsandboxed slot can re-point that folder during its window. With the folder inside
`.multi-llm/`, the re-pointing is itself exempt, and either list alone would exempt a tracked
tree: one taken before the window exempts the target of a symlink removed in it, one taken after
exempts the target of a symlink planted in it. So every lane resolves the list in `begin()` and
again in `finish()`, before the after snapshot, and a path is left out only where both lists
leave it out (`common_ignore_prefixes`). When they differ, one `[SANDBOX_BYPASSED]` line names
the folders only one of them covers (`bypass_ignore_drift`), printed under the slot's name and
noted for the closing summary; it fails nothing by itself. A folder that appears during the run
resolves to the path it had before, so a normal run sees no difference, and its after snapshot and
comparison are the plain ones. The before snapshot keeps the output folder's entries apart (a
`porcelain_snapshot` zone, hashed under a cap of its own, never printed), compared only when the
lists differ. Then the after snapshot hashes the paths outside both lists and the folders under
each list alone under caps of their own, so a large file in a folder that one list leaves out
cannot switch off the content comparison of the paths both lists audit, and
`unexpected_writes_across_drift` compares them: a path that appears or disappears counts; outside
both lists, a status change counts, and so does a content change where both sides were hashed
(a lane's own list could have hashed them or not, so both comparisons run); under one list alone,
the same where both sides were hashed, and the path itself otherwise. So the verdict names every
change either list alone would have seen, a dirty file put back to clean under the before list is
named, and one left alone there is not.

A trust-only `[projects."…"]` table never fails it (codex adds one per session directory, and
concurrent sessions share the home), but every one besides the run's own cwd that appeared during
the run and is still in `config.toml` when the closing summary prints is named there, as
added by this run or by another codex session running at the same time (`landlock_dispatch.TrustEntriesNote`, fix
round 17: a jailed batch can write `$CODEX_HOME`, so a planted `[projects."/"]` must
not pass unnamed; one gone again by then, run 6's sibling noise, is not named; removing one is
offered as the user's own call, never an order; fix round 23 dropped round 22's
`/proc` attribution, since any process can be named codex and a planted
`[projects."/"]` read "nothing needs doing": the note reads no other process's state).

It compares each watched tree file by content hash plus inode, size, mode and mtime
(`landlock_dispatch._tree_entry`), never by ctime: a plugin's uv `.venv` under
`plugins/cache/` is hardlinks into the uv cache, so any `uv run` elsewhere on
the machine moves those inodes' ctime — on 2026-09-30 that failed a codex review-plan slot closed on
23 untouched pyyaml files.

A line naming more than five changed paths (`HOME_CHANGE_LISTED`) is summarized — count, shared
directory, three examples — and names a file under `$XDG_STATE_HOME/multi-llm/codex-home-check/` that lists them all
(`landlock_dispatch.describe_changed`, both lanes; one 0600 file per record named by its hash, the newest
`HOME_CHANGE_LIST_KEEP` kept, and every name and key on the line and in the file passed through
`render_name`, which escapes a non-printable one repr-style so a planted newline cannot
forge a marker line); the verdict does not move.

Parent-side reads of `$CODEX_HOME` files go through `landlock_dispatch._read_watched` /
`_open_and_read`: `O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC`, an `fstat` check for a regular file
carrying the `lstat`'ed inode, and a read capped at 8 MiB (`_WATCHED_READ_LIMIT`),
because a plain `open()` hangs on a FIFO a jailed turn swapped in and follows a
swapped symlink. A watched file that is a symlink records `symlink:<dest>|<target sha256 or absent/replaced/unreadable>`, so both a retarget
and an edit of the target's content are changes; a non-regular, swapped or unreadable result is a
change, never "unchanged".

A codex-internal EACCES is a `[JAIL_DENIED]` failure, never retried and never moved to
another rung, whose line names the path, the allow-list and both codex versions;
`jail_denied_edits` reads a command's denials only from lines where the denied tool names itself
at column 0 (`_DENIAL_TOOL_PREFIX`: the measured bash/dash, coreutils, git, sed, tee, Python and
node shapes), so a test suite's own `E   PermissionError: [Errno 13] …` report is never a denied edit (fix round
4); and `MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock` skips this rung alone for one process (`skip_rung_requested`; any
other value is rejected at start-up with exit 3).

The per-slot `[SANDBOX_BYPASSED]` line keeps its fixed stem and appends each rung the walk tried
with its outcome and probe tail (`fallback rung 'landlock' unavailable (codex 0.156.1: "…")`, or `not tried: unusable since codex 0.156.1 (panics; measured; installed: 0.159.3)`). One
`[SANDBOX_RUNG_RETIRED]` line per provider and rung is printed after the fan-out
(`announce_retired_rungs`; the executor prints its own once per process) when the rung was measured
retired this run — skipped by `removed_at`, or its probe printed the removal panic —
whatever the previous run recorded, and also on the old transition (the previous run of the phase
recorded `fallback-<rung>`, this one `bypassed`; review phases only, since
`ask` records no `models_served`). A timeout, canary-setup failure or other
probe error is named as itself and never earns it.

The key gates every rung: `auto` (default) takes rung 1 then rung 2,
`contained` takes rung 1 and refuses rather than reach rung 2, `off`
takes none and refuses — exactly what it always meant; a project layer, starting from the widest
value, can only narrow. **⚠ Never enable landlock globally**: `codex features enable use_legacy_landlock` (or
`[features] use_legacy_landlock = true` in `~/.codex/config.toml`) makes every `workspace-write` dispatch panic
with exit 101, `agent-write` included, and no per-dispatch `-c` can clear
it — the flag is only safe as rung 1's per-dispatch `-c` beside
`--sandbox read-only`, and `doctor` warns when it finds the global one.

Rung 1 serves read-only profiles only and is **retired on codex 0.156.1+**: upstream marked the
feature deprecated (carried as `SandboxRung.deprecated_at` = 0.155.1, render-only), and on 0.156.1 its
probe exits 101 with `filesystem-restricted execution requires bubblewrap to isolate app-server sockets` (carried as `removed_at` = 0.156.1, measured
2026-09-24 on this configuration — not an upstream changelog entry; classified by that text as its
own `retired` walk outcome and by `describe_failure`, never by exit 101, which
constraint 2's global panic shares).

At or above `removed_at` the per-run walk does not probe rung 1 at all
(`skipped`), so on this machine `danger-full-access` is again the only read-phase
rung under `auto`; `doctor` still probes, and re-measuring moves or
drops `removed_at` (owner: whoever bumps the pinned codex version).

The AppArmor repair `doctor` prints remains the system-level fix for
`agent-write` and gives back codex's own `workspace-write`, a real boundary with a
confined network, which the jail's grants are not (where the jail probes ok, `doctor`
heads it as a system-level alternative); `tests/test_read_only_modes.py::TestTheOneDispatchTimeOverride` reads that declaration from both
sides. omp carries its bypass as a value too — `--approval-mode yolo`, so the token the scan reports
is the bare `yolo` and never the flag — but it is **not** a dispatch-time override:
it is rendered on every `generate` dispatch, because every narrower approval mode denies
`bash` headless and the code-review lane shells out to `git diff` (under
`--approval-mode write` the model answered `git version 2.43.0` after three denials, having run
nothing), and it is bounded by an explicit `--tools read,grep,glob,bash,edit,write` allow-list where a tool outside
the list is *absent* rather than denied.

Accepted by the repository owner on 2026-09-17 for `generate` only.

A runner whose only unattended mode is such a flag gets an explicit `None` row in
the `read_only_modes:` table (`skills/multi-llm/tests/read_only_modes.yaml`) **and a named exception recorded beside it**
(`bypass_exception:` — the token, the profile, why it survives, what retires it), never a silent
precedent; `agy`'s `--dangerously-skip-permissions` and omp's `yolo` are the
two entries that disposition currently holds; the first is queued behind a probe, and omp's is the
accepted one. (`aider`'s `--yes-always` was the third until
plans/ask-answer-recovery-and-false-success.md D4.2 retired it: aider's read-only mechanism is now
`/ask ` plus a stdin decline supply, `AIDER_DECLINE_LINES` × `s`,
because `--yes-always` also answered aider's own over-limit guard and its mention auto-add,
and an inherited `yes-always` from `.aider.conf.yml`/`.env` refuses
the slot before spawn.) `--yolo`-class flags are reserved for a future
`agent-write` runner that has **no** native sandbox or permission mode, and none of the
three pinned above needs one, so that row is empty today.

`tests/test_read_only_modes.py` is the guard and it reads this repo's own source, never a vendor's help text:
it scans every registered adapter's `build_command`/`get_env` (plus
`agent-readonly`) and goes red on any such token that is not a declared exception — and
equally red on a declared exception whose token is gone, so an exception cannot outlive the flag it
excuses. The table sat in the base `providers.yaml` until 2026-11-08, with the guard reading the base
layer alone so no project layer could add an exception; nothing at run time ever read it. It now
lives under `tests/`, which keeps that guarantee because no config layer is looked up there, takes
11 KB out of the shipped file, and stays outside `code_review.collapse_paths`' fixture globs, so a
changed `bypass_exception:` reaches reviewers as a whole diff. A `read_only_modes:` key back in
`providers.yaml` fails the guard.

**What the profile cannot contain, the prompt has to say.** `agent-write` containment is
file-granular and repo-rooted — and for opencode deliberately outside `HERMETIC_PROFILES`
(`utils/providers/opencode.py`) — so it saw nothing of the write subagent that stood up a local
`http.server`, drove headless Chrome at it for ten of an eleven-minute task, left the
server orphaned, and signed off with `killall -9 … python3`, killing the detached orchestrator that
had dispatched it (F2, 2026-09-16).

The write-batch scheduler's shared result-contract seam (`BatchPromptBuilder.with_result_contract`,
`utils/write_batch_scheduler.py`) therefore names
`kill`/`pkill`/`killall` and background servers,
watchers and browsers as forbidden on every dispatched write batch (fix round 21 dropped its "verify
with static checks only", which contradicted the checks an apply-code-fixes batch is told to run)
(N3, 2026-09-16: `prompts/implementation_task.txt`'s own `## Constraints` used to carry a second copy of
the same rule, which also stranded the template's terminal "Proceed with the implementation." ahead
of the appended contract — both are gone from the template now, so the shared seam is the only place
any write kind picks the rule up), and `implement_orchestrator.ACCEPTANCE_CRITERIA_FRAMING` heads the pasted criteria with whose
job they are: a plan's criteria are routinely *manual* verification steps, and nothing marks them as
such — `executor_role == "human"` is task-level, not criterion-level.

The process-safety rule is advisory, not a guard: nothing in-process can stop a runner that ignores
it (`SIGKILL` cannot be caught, and the runner is the same user), so the process-group
isolation only protects the runner from the executor's kill, never the reverse — what makes a killed
run survivable is that `--resume` reads the scheduler's verdict and re-runs the check it
owes, not that the kill was prevented (Fable review, 2026-09-17).

Since N4 (2026-09-16), a human task's own materialized entry skips the framing, the
`## Constraints` block and the proceed line entirely: `implement_orchestrator.build_human_task_prompt` renders its
checklist plus one "this task is deferred to a human; nothing dispatches it" line instead, because a
task whose whole job is the browser/manual work those sections forbid should not read a prompt that
forbids it.

## Model catalog snapshots

`is_model_valid` compares a spec with `providers.yaml` itself, so it cannot see a retired id
(`cursor-agent:cursor-grok-4.7-high` once passed the whole suite while not existing), and at run
time only opencode's `_detect_model` asks the CLI. `tests/test_model_catalogs.py` closes the gap
offline: every model id the base config names (`providers.<name>.models`, `defaults.*`, the
`driver` runners), with its `:effort` split off the way `build_command` splits it (the ladder's
`accepted_inbound`, which only codex widens), must appear in a committed snapshot of its CLI's
catalog. It reads committed files only, so it runs in CI with no credentials.

- **Snapshots** are `skills/multi-llm/tests/fixtures/model_catalogs/<source>.json`:
  `{captured, cli_version, command, ids}`, ids sorted and unique. kilocode's existing capture
  (`tests/fixtures/kilocode/model_variants.json`, which also records each model's variants) feeds
  the same check through `tests/model_catalogs.load_snapshot`.
- **Routes** (`ROUTES` in the test) map a provider to its source and an id into that catalog:
  verbatim for codex and opencode, `openrouter/<slug>` to `<slug>` for cline, goose, aider and pi,
  plain slugs for kimi, qwen-code, copilot and fx, omp's selectors for omp.
- **Gaps are named, never silent.** `UNCHECKED = {provider: reason}` holds every adapter without
  a committed snapshot, and `UNCHECKED_MODELS` every id outside a checked source's coverage
  (kilocode's native `openai/` and `google/` backends). Both directions are tested: an adapter in
  neither table fails, and so does an `UNCHECKED` row whose source now has a snapshot or an
  `UNCHECKED_MODELS` row that no longer applies.
- **Refresh** with `make refresh-model-catalogs [SOURCES=...]`, run by a person logged in to the
  CLIs it lists, never in CI (the script refuses when `CI` is set) and never on the dispatch
  path, so `status` stays subprocess-free. Sources are defined in `tests/model_catalogs.py`
  `SOURCES` with their parser; a CLI whose output format has not been captured here (cursor-agent,
  agy, grok) has no source, because a guessed parser writes a wrong snapshot. Review the diff,
  then move the providers the new snapshot serves out of `UNCHECKED`.
- **Staleness** is a warning, not a failure: a snapshot over 45 days old prints one in the test
  run's summary naming the refresh command.

First capture (2026-11-08): only codex, from `codex debug models --bundled`, which reads the
catalog shipped in the binary with no login and no network; with kilocode's capture that makes
two checked providers. opencode, omp and the OpenRouter-served adapters need a network capture,
and cursor-agent, agy and grok a logged-in one plus a parser.

## Provider output channels

**The fan-out asks each adapter for the channel it actually has** (`supports_file_output`). The
three JSON review prompts end in a `{output_location}` block rendered by `prompt_channels.render_output_location`
from the adapter's own declaration, never from a provider name branched on in an orchestrator:
`True` (the default) renders the byte-identical "write your JSON output to this
file: …" block, `False` asks for the raw array on stdout. The `False`
roster is pinned to exactly seven: **five of them are one shape, the sixth is another, and the
seventh is a third**.

Five run their CLI in a mode that cannot write and refuse in words: `cursor-agent`
`--mode ask`, `aider` `/ask`, `qwen-code`
`--max-tool-calls 0` (a **zero** tool budget — the abort is `FatalBudgetExceededError`, exit 55, and
zero bytes on stdout, so there is not even a fallback left), `vibe`
`--agent plan` (upstream's read-only profile, `write_file`/`edit`
permission `never`) and, since 2026-08-19, `copilot` — whose flagging
atom is `--no-ask-user` but whose real mechanism is the flag it deliberately does *not* ship:
with no `--allow-all-tools`, the model calls `create` and the CLI answers
`{"message": "Permission denied and could not request permission from user", "code": "denied"}`, and under `-p` no approval channel exists for it to arrive
on.

A measured `cursor-agent` slot is the archetype: `is_error: false` carrying only "I
can't write the requested file while Ask mode is active", then a failure on `JSON parsing failed from both file and stdout`.
**`gemini` is the sixth and it is worse**: under this adapter's argv (no
`--yolo`, no `--approval-mode`) gemini-cli ships **no `write_file` tool
at all** — nine tools, none of them a write — so handed one it errors internally
(`Tool "write_file" not found in registry`), writes nothing, and **exits 0 printing the model's own "JSON written to …"
confirmation as its answer**. A refusal fails loudly; this reports success for a write that never
happened, so the slot would record the confirmation sentence as the review.

**`codex` is the seventh (D6, 2026-09-11) and its reason is the machine, not a
mode**: `codex exec` writes through bubblewrap, and on a stock Ubuntu 24.04
(`kernel.apparmor_restrict_unprivileged_userns=1`) bwrap cannot start (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`), so a review asked for a file
spent seven minutes twice, failed three writes and returned no JSON — while the array was its final
`agent_message` all along, which `parse_output` reads.

The fan-out asks it for stdout; `supports_file_edits` stays `True` because the
agent-write profile really edits, and the model-free sandbox probe that gates it is D6 step 3. muse,
the other bubblewrap user, runs `--disable-sandbox` instead (beside `--disable-approval`, and
never `--yolo`, which also grants workspace trust). **The prompt is the fix, never the
argv**: restoring `-f`/`--yolo` to buy the file write back is the
write bypass §3.1 removed.

The one argv that *does* move is the D6 override the read/analyze bullet names (2026-09-17): under
`providers.codex.sandbox_fallback: auto` — the default, because codex is the one reviewer that insists on its own
sandbox and the other four already run these phases with the operator's shell rights (muse always
`--disable-sandbox`; pi, cursor-agent and claude-code never had one) — and only once the probe
has said `unavailable`, a read-only codex dispatch (`generate` or
`agent-readonly`) first tries the contained `landlock` rung
(`--sandbox read-only -c features.use_legacy_landlock=true`, kernel-denied writes, reported `[SANDBOX_FALLBACK]`; not even probed on
codex 0.156.1+, where it is measured unavailable — `removed_at`), then (since 2026-09-30)
the plugin's own Landlock jail, `plugin-landlock` (`--sandbox danger-full-access` under the ruleset
`utils/landlock_exec.py` applies before it `exec`s codex; the `incidental`
grade, reported `[SANDBOX_INCIDENTAL]`, with the mode and home checks in place of the porcelain
check — the read/analyze bullet has the detail), and only when that rung's probe fails too, or it is
absent or skipped, runs `--sandbox danger-full-access` in the real repo root — no worktree snapshot, because
the measured by-hand run escaped one in 6 commands to read the untracked plan and the ignored
`node_modules`, and found 2 unique validated-high issues the sandboxed reviewers did not —
prints one `[SANDBOX_BYPASSED]` line per slot, records `sandbox: bypassed` in
`.status.json`'s `models_served` (so `--reaggregate` keeps it), labels the
report `(ran unsandboxed: sandbox_fallback=auto)`, and fails the slot by path when `git status --porcelain` changed across
it; `contained` stops at the landlock rung and refuses rather than bypass (it never takes
the plugin's jail either, whose grade is not a real OS boundary), and `off`
restores the refusal outright.

It never buys the file write back (the fan-out still asks codex for stdout), and this fan-out
override never reaches a write profile: the write phases dispatch through the executor, where since
2026-09-30 `agent-write` may take the `plugin-landlock` rung alone, for the
`plan-file` and `tasks` scopes under `auto`, or the
uncontained rung only with F5's consent (the read/analyze bullet has the rule). Nothing on the parse
side moves — `invoke_with_file_output` still reads the file first, and `save_model_result` writes
the per-model result file from the returned data whichever channel supplied it.

The roster is not maintained by hand: `tests/test_provider_housekeeping.py::TestReadOnlyArgvMatchesTheDeclaration` reads every adapter's rendered argv,
and any restriction-shaped atom in it (`ask`, `plan`,
`read-only`, `no-write`, `sandbox`, or a zero tool/write budget)
must be adjudicated in one of two tables — one that forces `supports_file_output = False`, one recording
that it was checked and still writes. A new flag nobody classified fails by name, so the question is
answered when the flag is introduced rather than when a user's fan-out dies on it.

**An argv guard is blind to a *default*, which is how `gemini` sat at
`True` with a clean argv until it was measured**, so a further table sits beside
those two: `NO_WRITE_TOOL_AT_ALL`, keyed by provider rather than by atom, whose rows are
measurements naming their probe rather than opinions, and which feeds the same derived-roster
assertion.

## Capturing a CLI's streams

Both spawn lanes read a provider CLI's stdout and stderr through one helper,
`utils/process_capture.py` (`PipeCapture`): the review fan-out in `llm_client._run_provider_cli`
and the executor in `dispatch_spawn._spawn_in_context`. Each lane still owns its spawn
(`process_control.popen_tree`), its process group (Lane A shares the terminal's so a Ctrl-C reaches
the CLI; the executor gives each runner its own), its Windows Job Object, its timeout and its
teardown. The capture only replaces what `Popen.communicate` did with the CLI's pipes:

- **Bounded per stream** at `process_capture.MAX_STREAM_BYTES` (32 MiB, the executor's
  `MAX_STREAM_BYTES`). Bytes past the bound are still read, so the CLI never blocks on a full
  pipe, then dropped and counted; one `[STREAM_CAP]` line names the stream. A capped stdout is
  never parsed and its head is never an answer: both lanes refuse it with the same sentence
  (`process_capture.oversized_stream_message`) at the same rung, `unparseable_result` (rung 5,
  retried). The executor refuses it before the schema verdict reads it; the fan-out's exit-0
  refusal is a parse failure carrying `details.stream_capped`, `failure_category:
  unparseable_result` (so the review ladder retries it as the executor does) and `incomplete`, so
  ask's `answer_finished` keeps the head as partial evidence and counts no answer from that run.
  A capped stderr is kept truncated. The fan-out's read-only audit is never clean for a capped
  stdout, since it saw only the head.
- **The same text below the bound.** The kept bytes are decoded once with the stream's encoding
  and error handler, with `communicate`'s text-mode newline translation, so parsers see the same
  string. Stdin is encoded as `communicate` encodes it (on Windows, as the text stdin's
  `TextIOWrapper` writes it: `\n` becomes `\r\n`).
- **Stdin fails as `communicate` fails.** The input is checked and encoded in the caller's thread
  before any feed starts, so a type or encoding error raises there instead of the child reading an
  empty prompt. A feed ignores only what `communicate` ignores, `BrokenPipeError` and, on the
  blocking (Windows) path, the `EINVAL` of `Popen._stdin_write`; any other write error is raised
  once by the capture's `communicate` after the CLI exits, unless that call had already timed out.
- **The wait ends with the CLI.** The capture takes all three pipes off the `Popen`, so
  `Popen.communicate` serves only as the wait, under the caller's timeout, and returns when the
  process exits. Stdin is fed from a thread of the capture's own and is not part of the wait: once
  the CLI has exited, what it left unread is not sent (a background child holding stdin unread
  cannot keep a finished CLI waiting to the deadline). The output pipes then get at most
  `process_capture.DRAIN_S` (5 s; `LANE_A_DRAIN_S` and the executor's `TIMEOUT_DRAIN_S` are this
  value) to reach end of file. Past that the capture prints `[PIPE_HELD]` and returns what the CLI
  wrote with its real exit code. A timeout still raises `TimeoutExpired` with the output so far,
  and the lane's post-stop drain collects the rest.
- **Nothing is stopped or cut off for a survivor.** A held pipe is never closed under its writer,
  which would end it with `SIGPIPE`/`EPIPE`: its drain keeps reading and drops what it reads until
  the last writer closes it. A CLI that exits on its own leaves what it started running.
- **A closed capture is let go.** The drain and feed threads hold only their pipe, an end-of-file
  flag and, until `close()`, the buffer they fill; never the capture or the `Popen`. A survivor
  that holds a pipe for hours keeps a drain thread and its descriptors, not the call's buffers.
- **The prompt goes with the CLI.** The encoded input belongs to the capture, and a feed takes it
  one chunk at a time. When the CLI exits the capture drops what is left, so a feed stuck writing
  to a stdin a survivor never reads holds one chunk (64 KiB at most) and its raw descriptor, never
  the prompt or a write buffer.

POSIX drains both output pipes on one thread with a poll selector, and feeds stdin in pieces the
pipe has room for, so the feed stops as soon as the CLI is gone. Windows cannot select on pipes,
so each pipe gets a blocking thread, stdin's writing 64 KiB chunks with a stop check between them.
None is closed from another thread: Windows' anonymous pipes are synchronous and the C runtime
locks a descriptor for the whole of a pending read or write, so a close from elsewhere waits for
the write instead of ending it (`Popen.communicate` on Windows likewise leaves such a writer
running). A blocking thread whose pipe is still held finishes when the last process on the other
end closes it. A `Popen` stand-in without
real pipes, as the tests use, has nothing to take, and its own `communicate` answers. Tests:
`tests/test_process_capture.py` (both drain shapes on POSIX, the Windows one included),
`TestLaneACaptureEndsWithTheCliAndIsBounded` in `tests/test_llm_client.py`, the capped-stdout row
in `tests/test_lane_b_structured_output.py`, and the held-pipe and capture-bound rows beside the
surviving-grandchild test in `tests/test_subagent_executor.py`.

### A CLI that loops

opencode 1.19.5 never exits when the provider cuts a stream before its finish chunk, or answers
HTTP 200 with an error body. `run`'s session loop leaves only on a finish other than `tool-calls`
or `unknown` (read from the binary's embedded bundle), and both failures finish a step as
`unknown` with nothing printed. So it re-sends the same request with no backoff until something
kills it. Offline against `tests/container/failserver.py`, the `truncated` and `error-envelope`
captures made 3,887 and 3,456 requests in 240 s, about 16 empty steps a second, 2.7 MB of stdout,
stderr empty. Only the slot's deadline stopped them. Against a real provider that is a whole slot
timeout of requests, each possibly billed for its prompt.

An adapter can therefore declare a **stdout watch**, `LLMProvider.stream_watch()`, and opencode
does (`opencode.LoopWatch`):

- **One mechanism, both lanes.** Each lane asks the adapter once per attempt
  (`process_capture.provider_watch`) and hands the watch to its `PipeCapture`. The drain feeds it
  each stdout chunk as it is read. When it returns a reason, `PipeCapture.communicate` prints one
  `[STREAM_STOP]` line and raises `StreamStopped`, a `TimeoutExpired`. Each lane's own timeout
  handler then stops the CLI exactly as at the deadline (Lane A's `_stop_and_note` sweep, the
  executor's `_kill_timed_out`, each with its Windows Job Object), and its bounded drain collects
  the rest. There is no stop code of its own. The wait polls the watch every
  `process_capture.WATCH_POLL_S` (0.1 s) only while a watch is live. An adapter without one is
  waited on exactly as before, and Lane A passes `stream_watch=` to `_run_provider_cli` only when
  there is one.
- **The verdict.** The attempt is `timed_out` with `stream_stop` set to the watch's reason. Both
  lanes record it as **`runner_error`**: the CLI was answering a failed provider response, not
  running out of time. The failure line is the reason plus how soon the stop came
  (`process_capture.stopped_early_text`), for example `opencode looped: 20 steps in a row within
  0.7s ended with finish reason "unknown" and printed no text or tool call, and opencode re-sends
  the request after such a step without ever exiting (...). Stopped after 1.2s rather than at its
  600s timeout`. The fan-out keeps the reason as `details.stream_stop` and keeps `error_code:
  TIMEOUT`, because the plugin cut the run short and the salvage gate must refuse a salvaged `[]`
  from it as from a deadline kill. The executor records `runner_error` with the same reason,
  measured against its per-task timeout, and a write batch it stopped is never reported as
  having run out of time.
- **No identical retry.** A `runner_error` normally earns a transport retry. A loop stop does not:
  the CLI had already re-sent the same request 20 times by itself, and nothing in the captures
  shows a fresh process changing the failing response. Each lane hands the ladder the explicit
  `stream_stop` metadata (`details["stream_stop"]` on the fan-out, `_Capture.stream_stop` in the
  executor) as `RetryLadder.record(..., stream_stopped=True)`. That is rule 10, which stops the slot
  (`stream_stopped`) the way rule 6(b) refuses to repeat a rate-limit chain a CLI already ran, and
  leaves `driver.subagent_retries` unspent. The status and its failure line stay as above.
- **opencode's signal.** A streak of `opencode.LOOP_EMPTY_STEPS` (20) steps, each a `step_start`
  and a `step_finish` with reason `unknown` and nothing between them. A step counts once, and
  only when it is a well-formed `step_start` (a `step-start` part with a session and a message
  id, as 1.19.5 prints every one) closed by the `step_finish` of the same session and message. A
  boundary out of order (a finish with no open step or for another one, a second start, a
  malformed start or finish) ends the streak, and nothing counts until the next well-formed
  start. Any other event (`text`,
  `tool_use`, `reasoning`, `error`, a type 1.19.5 does not emit), a line that is not an event or is
  longer than `LOOP_LINE_MAX` (64 KiB), or any other finish reason ends the streak. Tokens are not
  read: the looped steps carry none, and a loop that bills tokens is the costlier one to leave
  running. A stream the watch does not recognise never trips it, so a release that changes these
  events falls back to the deadline.
- **Why 20.** In 11,187 steps of 1,607 real opencode sessions in a developer's local session
  store (up to 132 steps in one session, 28 sessions past 48 steps), 13 steps finished `unknown`,
  none with a text or tool part (reasoning and snapshot patches only, which `run --format json`
  does not print). The longest streak was 2. In 10 of the 11 sessions an ordinary step came next
  (`tool-calls` or `stop`): a real provider hiccup that opencode's re-send recovers from. The
  eleventh session ended on a step that never finished. 20 keeps a 10x margin over that. On the
  captures it stops the cut stream 0.7 s and the error body 1.6 s into the stream (opencode's own
  timestamps), after 20 requests instead of about 3,900. A slow loop, where each failed step
  first spends a minute reasoning, can still reach the deadline first. That is no worse than
  before, and it is the case where a recovery is likeliest.
- **A watch must stay cheap and safe.** It runs on the drain thread, keeps a counter, the open
  step's ids and one partial line, and a raise drops it (the wait then runs to the deadline).
  Before acting on a stop, the wait checks whether the CLI has exited. One that already has is
  reaped and returns its own ending, so a watch that tripped on the early steps of a run that
  then answered and exited stops nothing, and `stream_stop` stays `None`.

Tests: `tests/test_opencode_loop_watch.py` (the watch over every committed opencode stream, the
two loop heads in `tests/fixtures/opencode/loop/`, a 60-step legitimate run with short `unknown`
streaks, unpaired, duplicate and malformed boundaries, chunking down to one byte; a run that
exited 0 after the watch tripped, on both drains; both lanes stopping a looping fake CLI through
their real kill path, and the verdicts they record), the watch rows in
`tests/test_process_capture.py`, rule 10 in `tests/test_retry_ladder.py`, and the no-second-dispatch
row on both lanes in `tests/test_lane_a_ladder.py`.

## Reading a capture

`llm_client.invoke_with_provider` is three steps: `prepare_dispatch` (provider, timeout, transport,
phase schema, jail grant, argv, child env, and every refusal before a spawn), the spawn
(`_run_dispatch`), and the reading of what it captured (`provider_outcome.outcome_from_capture`).
The reading spawns nothing and touches no file itself, so a fixture's bytes become the result a
caller sees without a child process or a patch: build a `provider_outcome.Capture` (the streams,
how the run ended, the duration, the capped streams and the bound they were capped at) and hand it
to `outcome_from_capture`.

- **One tail.** A non-zero exit, an exit 0, a deadline kill and a launch failure each start the
  result their own way (the error line, the parse, the timeout line), then every child that ran
  goes through the same tail: read-only audit, `stream_capped`, `failure_category`, the
  `sandbox` word, the jail read-back, then on a failure `stdout_excerpt` and `salvage_text`, then
  `answer_text` and `usage`. A key the reading has nothing for is absent, never empty; a launch
  failure carries none of them. Each path keeps the key order it has always had, and the order of
  its calls: the jail read-back samples live state, so a non-zero exit takes it before the audit
  (its keys still land after `sandbox`), a timeout and an exit 0 after it.
- **Full stream first, redacted before bounded.** The audit, the salvage and answer hooks and
  usage read the whole stream; any stream text that leaves is redacted, then cut.
- **An envelope is read by where it sits, in one place.** An adapter whose JSON mode prints one
  envelope around its answer (claude-code, cursor-agent, gemini, grok, agy, goose) declares how it
  opens (`ENVELOPE_SIGNATURE`, the opening keys of its captured envelopes,
  `stdout_shape.envelope_signature`). `stdout_shape.read_stdout` is the one rule, read by the parser and by
  `stdout_shape.diagnostic_streams`. Provenance is position, not keys: a model writes inside the answer
  string, never the start of stdout, and every committed capture is exactly one envelope. After ANSI
  colour and leading whitespace, stdout opens with a run of complete JSON values, one per line
  (`raw_decode` where each starts, so one linear pass), then any trailing text. One of those values
  opening on the signature is the **envelope** (the last such; JSON log lines and trailing text are
  the CLI's), unwrapped from its answer fields as a whole stdout would be. The signature opening
  there with no complete value is **cut**: a parse failure marked `incomplete`, before any text
  extraction could serve its bookkeeping or goose's echoed user turn. The opening **anywhere else**,
  found by a plain regex so no quote or bracket can hide it (a banner ahead of the envelope, an
  answer quoting one), is **ambiguous**: a parse failure, one bounded retry. A stdout with no
  opening goes to the adapter's ordinary extraction, unchanged. A whole envelope without its answer
  field fails too. **The cost:** text ahead of the envelope cannot be told from a model quoting
  one (an example error envelope behind a line of prose, or in an unfinished fence, would otherwise
  decide `auth_failed`), so a CLI release, wrapper or runtime that starts printing a banner on
  stdout breaks that adapter's parse: every slot fails, and the ladder dispatches it once more,
  until the banner's measured form is added to the rule. No capture shows one today.
- **One reduction for failure verdicts and their quotes.** The six adapters' `vendor_text` is
  `stdout_shape.diagnostic_streams`, so the classifier on exit 0 and on any other exit, and the executor's
  failure quote, read one text: for an envelope, each signature record as the adapter's
  `envelope_diagnostics` keeps it (its `ENVELOPE_ANSWER_FIELDS` out, the vendor's refusal fields
  such as `is_error`, `subtype` and `error` in; claude-code keeps a `result` beside `terminal_reason:
  "api_error"`, the CLI's own `API Error:` sentence; goose drops its echoed user turns), plus the
  other JSON lines and the trailing text; for a cut or ambiguous stdout, nothing of stdout, so only
  stderr is read, under the adapter's [launch diagnostics](#launch-diagnostics); for a plain stdout
  (a CLI's plain-text refusal), all of it.
- **kilocode's diagnosis is its painted line.** Its stdout is the answer and its stderr also
  carries the banner and a tool trace naming what the model read, so its `vendor_text` keeps only
  the stderr lines that open on its bold-red `Error: ` paint (a reset ahead of it allowed), colour
  off. An uncoloured run (`NO_COLOR` reaches the child) therefore gives no verdict from its text:
  its exit alone decides, one bounded retry, until a capture shows how kilocode frames an error
  without colour, since an uncoloured `Error:` line cannot be told from a tool's output or a
  multi-line tool argument. The cause OpenRouter's gateway gives, `Provider returned error`, is
  read from the JSON body on the lines after it as a structured code: `error.code` 429 is
  `rate_limited`, in a rate limit's place in the tables' order, and the body's message and
  metadata never reach a table (`KiloCodeProvider.classify_failure`).
- **One parse result.** Every `parse_output` returns `{"success": True, "data": …}` or
  `{"success": False, "error": …, "raw": …, "data": None}`, adapter keys after those four, built
  through `providers.parse_kit` (`ok`, `fail`, `answer_from_text`, `loads_or_extract`,
  `direct_json`). A failure's `raw` picks the executor's rung: non-empty text is
  `unparseable_result`, none is `runner_error`, so each adapter passes the text it means. Error
  sentences stay the adapter's own: the fixture manifests grade them.
- **The jail stays the caller's.** Its read-back needs the attempt's live context, so it comes in
  as a callable (`JailReadBack`); a jail denial replaces the error line, carries no category tag
  and wins over any category.
- **Logs.** The caller passes the log writer in (`log`, `LogWriter`; `_log_outcome`), and the
  reading calls it once, where each path has always logged, so a reading that raises later still
  leaves the log: a hard failure once its error line is final, before the excerpt, salvage,
  answer and usage steps; an exit 0 after its salvage, before its answer text, unless the caller
  defers it (`details["_deferred_log"]`, see [provider outcome logs](#provider-outcome-logs)).
  The prepared dispatch's contexts are closed in a `finally` on every outcome.
- **The schema verdict** (`schema_phase`, W1: deferred by the owner, no production caller) sits
  behind `_phase_schema` before the spawn and `provider_outcome._schema_verdict` after it.

The executor's `dispatch_spawn._Capture` carries the same stream fields but stays mutable (the
spawn fills it in steps, the jail and the parse write to it afterwards), and its readers read a
status, not this result, so it shares the fields and its readers accept a `Capture`
(`tests/test_provider_outcome.py`). Every graded fixture row pins both lanes' ladder status as
`expect.lane_status` (`tests/test_provider_fixture_rows.py`): `lane_a` from `outcome_from_capture`
as the review fan-out reads it with no result file, `lane_b` from the executor's readers. A row
whose two statuses retry differently carries `open: <unit>` until that unit settles it, or
`by_design: <reason>` for the one permanent difference: an exit 0 of an adapter that is not
`EXIT0_CLASSIFIABLE`, which the fan-out stops on and the executor retries once. A row whose
measured `expect.classify_failure` is a known bug keeps that measured pin and adds
`expect.known_wrong: {intended: <verdict>, reason: <why>}`; the grader fails once the adapter
gives `intended`, so the fix deletes the marker and re-pins the row.

### Launch diagnostics

Two verdicts fail the executor's whole batch from text alone: `sandbox_unavailable`
(`failure_classifier.SANDBOX_MARKERS`) and `spawn_failed`'s stderr rows (`failure_classifier.STDERR_LAUNCH_FAILURE_MARKERS`).
So they are read only from text that a model or a tool cannot write:

- **Anchored lines.** `SANDBOX_MARKERS` is bubblewrap's measured refusals, each anchored at a line
  start on its exact prefix and listed beside the capture it came from. A line that only mentions
  bubblewrap, or an unmeasured refusal, is no sandbox verdict and keeps its verdict and retries,
  but the failure message still names the sandbox: any `bwrap: ` line on the CLI's stderr (a log
  prefix ahead of it, another wording) adds `failure_classifier.sandbox_hint`'s sentence pointing
  at `multi_llm.py doctor`, to the fan-out's error line beside its category tag
  (`provider_outcome._sandbox_hint_line`) and as the executor's `runner_error` reason, one
  transport retry (`attempt_classifier._sandbox_hinted`). The preflight probe still decides by exit
  code.
- **The CLI's own refusal, or no progress.** `spawn_failed`'s stderr row is two tables.
  `CLI_REFUSAL_MARKERS` are a CLI's words about its own argv or config (omp's), batch-fatal
  wherever they appear on its stderr. `SHELL_REFUSAL_MARKERS` (`command not found`,
  `exec format error`, cmd.exe's `is not recognized`) name whichever command a shell could not
  run, which may be a hook's, a wrapper's or an MCP server's on the same stderr. The classifier
  reads both as `spawn_failed`; both lanes keep a shell phrase's verdict only for a run that made
  no progress, nothing on stdout and nothing else the adapter recognises on stderr, and otherwise
  take the adapter's reading of the same capture with only those phrases blanked
  (`failure_classifier.launch_refusal`: a rate limit, or `None` and one transport retry; see
  [the executor](delegation.md#read-and-analyze-executor)).
- **stderr, where the CLI wrote it.** Each adapter declares `LAUNCH_DIAGNOSTICS` with the captures
  it rests on. `stderr`: everything there is the CLI's, its launcher's or the OS's own. `none`:
  stderr also relays a model's or a tool's words on lines that cannot be told apart, so neither
  row reads it (`LLMProvider.launch_diagnostics` hands `classify_streams` an empty
  `launch_stderr`). Exit 127 stays `spawn_failed` everywhere. Today `none` is kilocode (its tool
  trace), fx (its shell tool's command and output), gemini (`Error executing tool …`), opencode
  (an auto-rejected permission quoting the model's tool argument) and pi (its SDK's dump of an
  unparsed chunk). `tests/test_launch_diagnostics.py` checks that every adapter declares it.
- **stdout is never read for them.** No provider run merges stderr into stdout: Lane A captures
  the two pipes apart, the executor and the sandbox probes give each its own pipe, and the
  `2>&1` in the code is the orchestrator's own detach log. The write lane's verify command merges
  them, but no classifier reads it. A vendor refusal that reached us only on stdout would be a
  bounded retry: at most one extra dispatch per lane, and a sandbox that refused spends no model
  tokens. Matching it there would hand a model's answer the power to stop the batch.
- **The plugin's own refusals are structural.** The plugin's jail refuses with an exit code
  (`landlock_exec`'s 125 and 126 with its `landlock_exec:` note, read by
  `landlock_dispatch.wrapper_evidence` before any output is parsed), a spawn that fails is a
  launch error, and the sandbox ladder judges its probes by exit code. None of them, and no
  `sandbox_fallback` decision, reads these markers.

## Usage reporting

**Usage is reported or absent, never zero** (`parse_usage`, D3j): claude-code reads the
`result` envelope `parse_output` discards, codex the terminal
`turn.completed` event, opencode the **sum** of every `step_finish` part (reading the
last step alone under-reports a multi-step run). Every other adapter declares `usage: none`
and records duration only; a batch mixing the two reports the partial sum plus
`tasks_without_usage`.

## Child Python environments

**A runner never inherits the skill's own Python environment** (fix round 8). `uv run --project <skill dir>`
exports `VIRTUAL_ENV` naming the skill's venv, and a forwarded or image-wide
`UV_PROJECT_ENVIRONMENT` can name it too; `providers.child_env.build_child_env` (both lanes, the codex jail, the
verification command) drops either when it resolves to the running interpreter's
`sys.prefix` (`skill_environment_names`), so a subagent's
`uv run`/`uv pip`/`poetry` in the user's repository uses
the repository's own environment; a value naming another environment is kept.

Never export `UV_PROJECT_ENVIRONMENT` from a host route to move the skill's venv: round 7 did
(631bf9f, reverted) and a jailed `uv run` in the user's repository failed with
Permission denied, an unjailed one installed into the skill's venv. The same builder takes the
venv's `bin` (`Scripts` on Windows) off a runner's
`PATH` (`skill_environment_path`, fix round 9): with neither name set and no
`.venv` in the repository, uv uses the first `python` on
`PATH`, so `uv pip install` in an empty repository still targeted the skill's
venv and a runner's `python`/`pytest` were the skill's.

Runner CLIs are unaffected (both lanes resolve `argv[0]` against the dispatcher's own
`PATH` before the spawn, and the venv holds none); the suite's mock launchers get
the test interpreter from `python3`/`python` shims in
`SkillRunner`'s `bin/`.

## Private config roots

copilot, kimi, qwen-code and vibe write CLI state (sessions, caches, logs) under the user's home,
so each spawn gets a fresh `0700` root instead (`providers.private_home.PrivateHomes`), handed to
the CLI through its own variable (`COPILOT_HOME`, `KIMI_CODE_HOME`, `QWEN_HOME`, `VIBE_HOME`). Two
concurrent slots never share one writable root. A root the user pinned through that variable is
used as it is and never removed.

- **A root lives as long as its dispatch, and its CLI.** Both spawn sites enter
  `llm_client.provider_dispatch_context`, which is also a `private_home.dispatch_homes` scope:
  closing it removes the roots the adapter's `get_env` made inside it, once every child the
  dispatch started has ended. The scope holds a `process_control.child_ledger`, which
  `popen_tree` adds each child to. A child has ended when it was seen to exit, or when its spawner
  confirmed the stop (`process_control.note_stop`): the CLI gone and every process group the stop
  reached `killed` or `gone`, U77's rule (`process_control.unconfirmed_kill`; the executor's
  `_kill_timed_out` applies it, its own zombie counting as stopped, and Lane A's
  `llm_client._stop_and_note` applies it to the sweep's groups). Any other stop (`survived`,
  `unresolved`, `skipped`, a kill that raised, a Windows job not ended) keeps the roots, `0700` and
  tracked, so whatever is still running can go on writing there. Roots are attributed through a
  context variable, so concurrent slots never release each other's. A root made outside a dispatch
  (an auth probe, `doctor`), one kept for a stop not confirmed, and one that could not be removed
  wait for `PrivateHomes.cleanup`, registered with `atexit`, or, after a crash, for the sweep below.
  A CLI that exits on its own has ended: what it leaves running is not followed.
- **Orphans are swept by owner, never by age.** A root is named
  `<prefix><pid>-<start>-<pidns>-<random>`: the creating process's pid, start time with its source
  (`proc.8812345`, `process_control.sourced_start_time`: `/proc` ticks, `ps` text or a Windows
  creation time) and PID namespace (`na` off Linux, which has none; `unknown` on Linux when it
  cannot be read). The first root an adapter makes in a process removes that adapter's roots whose
  owner is dead: both namespaces known and the same, and the pid gone or live under another start
  time read from the same source (a pid alone can be reused). An owner that is alive, whose start
  time cannot be read now or was read from another source (Linux falls back to `ps` when `/proc`
  cannot be read), or whose namespace is another or unknown on either side (two sandboxes sharing
  one temp directory) keeps its roots, and so does a root named before owners were recorded
  (`<prefix><random>`). Only real directories this user owns are removed, never through a symlink.
  The jail's scratch directories use the same pid and start-time rule
  (`landlock_dispatch.sweep_slot_leftovers`).

## Provider outcome logs

**A provider log describes the invocation's final outcome, and names its provider.**
`llm_client._save_log` writes `<PROVIDER> LOG` from the spec's provider prefix
(`CLAUDE-CODE LOG`, `CURSOR-AGENT LOG`, …) with
`Success:`/`Error:` for the outcome the caller actually got, plus
`Result source: file|file_extracted|stdout_fallback` and, when the stdout extractor lost, either `Stdout parse: failed (<reason>)` (a real
double failure) or `Stdout: not used (file output)` when `result_source` is one of
`FILE_WRITTEN_SOURCES` — a file-output provider was told not to print JSON on stdout at all, so its
failed stdout parse is expected, not a fault worth flagging beside `Success: True` (F9.2,
2026-09-16).

`invoke_with_file_output` defers every exit-0 write, not only a parse failure (`_defer_parse_failure_log` →
`details["_deferred_log"]`), and writes once, after the file-vs-stdout channel is resolved, via
`_finalize_file_output_log` — until 2026-09-11 a claude-code review that completed through the file
channel with 24 suggestions logged `Success: False` / `No valid JSON found in text` under a header
that called it cursor-agent; until F13 (2026-09-16) a *clean* stdout parse wrote its log
immediately, before the channel was known, so `Result source:` was silently absent for exactly
the runs that used the stdout fallback. Hard failures (timeout, non-zero exit, launch failure) still
write immediately — no channel choice to wait for.

The per-model fan-out's `Output file: <path> (source: …)` line (`llm_client.describe_output_file`) only says "Output file"
when `source` is `file`/`file_extracted`; a
`stdout_fallback` source — the model never wrote there — says "Result will be saved to" instead
(F14). The `save_agent_log` that once lived in `review_orchestrator_base` had no callers and is
gone.

## Version floors and measured versions

**A floor (`min_version`) is declared only where the break was measured; `verified_version` is
the release the captures were taken on and only warns.** A version a capture was taken on is an
upper bound on the true floor, never the floor: pinning `min_version` to it would refuse installs
that work, with exit 3 and no evidence that anything broke there. An adapter whose probes landed
with it keeps a floor (copilot, kimi, qwen-code, vibe); an adapter that existed before its probes
leaves `min_version` empty (grok, muse, cursor-agent, agy) and carries the measurement in
`verified_version`. `tests/container/cli-versions.lock` pins the same release for the container.

Code comments next to these two values state this rule and point here. The measurement history
lives in this section, not in the adapter.

### grok

- 0.2.93 (2026-08-19): every fixture in `tests/fixtures/grok/` was first captured on it with the
  step-5 probe (`--network none`, the stub and the failserver both inside the container, no
  credential). A `min_version` of 0.2.93 was set that day and removed the same day for the reason
  above.
- 1.0.46 (2026-10-08): every row re-captured on it with the same rig in the pinned image, and
  `cli-versions.lock` moved with it. The parser needed no change; the three `no_visible_content`
  rows and the pre-flight row are byte-identical. What moved: five requests per answered exec (a
  root probe and a reply-summary call on the `-m` model), the title model `grok-4.6`, `stopReason`
  `end_turn`, a 429 passing the vendor's own words (with the wait), no `Request URL` or
  response-header lines, and a 5xx or zero-byte response retried for about 1,340 s instead of
  335 s. Re-captured again the same day with `get_env`'s two switches in force: four requests per
  answered exec, and every retry chain short.
- `grok --version` prints `grok 1.0.46 (<commit>)` on stdout; the base `parse_version` reads its
  first dotted token without an override. The two shapes the parser depends on (`{"text": …}` for
  an answer, `{"type": "error", …}` for a failure) are vendor-internal envelopes, so the pin is
  one to re-confirm rather than a range to trust.
