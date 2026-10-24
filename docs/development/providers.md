# Provider contracts and configuration

Read before changing provider adapters, permission profiles, sandbox fallbacks, runner configuration, output channels, usage, or child environments.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Configuration and runner resolution](#configuration-and-runner-resolution)
- [Read-only provider contracts](#read-only-provider-contracts)
- [Provider output channels](#provider-output-channels)
- [Usage reporting](#usage-reporting)
- [Child Python environments](#child-python-environments)
- [Provider outcome logs](#provider-outcome-logs)

## Configuration and runner resolution

**Config layering** (`provider_registry.load_config`): base `providers.yaml` → auto-discovered
`<git-root>/.multi-llm/providers.yaml` → `MULTI_LLM_PROVIDERS_CONFIG` env override. The auto-discovered layer deep-merges
its **full** contents — including a `providers:` block — over base, identical to the env
layer (no longer filtered to selection keys). Deep-merge, lists replace, `None`
skipped. Discovery anchors on the optional `anchor` (plan path threaded from
`resolve_models`) then CWD; cache keyed on the resolved anchor. Present-but-invalid explicit
overrides fail fast (`ConfigError`); `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` restores warn-and-skip.

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
one rung that **falls through at preflight instead of refusing**: `subagent_executor._preflight_batch_runners` prints
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

## Read-only provider contracts

**A read/analyze dispatch never gets write access.** An adapter opts into `agent-readonly` by
implementing `build_agent_command(prompt, model, profile)` with the runner's **own** denial mechanism (codex
`--sandbox read-only`, opencode `--agent plan`, claude
`--allowed-tools`/`--disallowed-tools`) and declaring it in `agent_profiles`. There is
no fabricated read-only mode: an adapter without one raises `ProfileUnsupported` and is refused as
a runner with the reason named. `tests/test_providers.py` (and `tests/test_subagent_executor.py`) cross-check
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
plugin files — fail-closed in the fan-out, warn-only in the executor lane).

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
`providers.yaml`'s `read_only_modes:` table **and a named exception recorded beside it**
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
excuses.

**What the profile cannot contain, the prompt has to say.** `agent-write` containment is
file-granular and repo-rooted — and for opencode deliberately outside `HERMETIC_PROFILES`
(`utils/providers/opencode.py`) — so it saw nothing of the write subagent that stood up a local
`http.server`, drove headless Chrome at it for ten of an eleven-minute task, left the
server orphaned, and signed off with `killall -9 … python3`, killing the detached orchestrator that
had dispatched it (F2, 2026-09-16).

The write-batch scheduler's shared result-contract seam (`_with_result_contract`,
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

## Provider output channels

**The fan-out asks each adapter for the channel it actually has** (`supports_file_output`). The
three JSON review prompts end in a `{output_location}` block rendered by `llm_client.render_output_location`
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
`UV_PROJECT_ENVIRONMENT` can name it too; `providers.base.build_child_env` (both lanes, the codex jail, the
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
