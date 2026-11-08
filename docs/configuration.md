# Per-project configuration

Giving one repository its own provider/model defaults, what each config layer
overrides, and why an auto-loaded override is safe. The base layer itself is
documented in [Providers](providers.md). Back to the [README](../README.md).

---

The shipped [`providers.yaml`](../skills/multi-llm/providers.yaml) is the
**base** layer shared by every repo. To give one repository its own
provider/model defaults — without editing the installed plugin — add an
optional, auto-discovered override file at
`<git-root>/.multi-llm/providers.yaml`. When it's absent, behavior is unchanged.

It deep-merges its full contents over the base and inherits everything you omit,
so it can stay tiny — change just the *selection* keys (`default_provider`,
`defaults.models`, `defaults.quick_models`, `defaults.modes`) and leave the rest:

```yaml
# .multi-llm/providers.yaml — only what this repo changes
default_provider: claude-code
defaults:
  models:
    - claude-code:fable
    - cursor-agent:composer-2.5
  quick_models:
    - claude-code:fable
```

Create one with `/multi-llm:multi-llm --init`. This is **fully automatic and
zero-prompt** — it works the same in a terminal, inside Claude Code, or in CI. It
scans your `PATH` for the supported provider CLIs and writes a preconfigured
override: starting from an inert template, it **uncomments** the full `providers:`
block, `defaults.models` / `quick_models` entries, and `default_provider` for each
CLI it detects, leaving the rest commented. `default_provider` is set to the
**first detected provider** in base order (claude-code, cursor-agent, gemini,
grok, fx, opencode, codex, kilocode, cline, goose, aider, agy, muse, pi, kimi,
qwen-code, copilot, vibe, omp). With **no CLIs installed** it writes the inert
template (which inherits the built-in defaults) and exits 0 with an install
notice — it never errors. Pass `--template-only` to skip detection and write the
pristine commented stub for hand-editing. Add `--gitignore` to keep the file
developer-local, and `--force` to overwrite an existing one — overwriting first
backs up the previous file alongside as `providers.yaml.bak.<timestamp>` (and
prints the path), so you can port any custom settings into the regenerated file.

## Good to know

- **Precedence (low → high):** base `providers.yaml` → project-local
  `.multi-llm/providers.yaml` → `MULTI_LLM_PROVIDERS_CONFIG=/path.yaml` env override.
  Each layer is deep-merged over the one below.
- **Which models a run uses.** A model is referenced as `provider:model` (e.g.
  `codex:gpt-5.5`); a bare model name uses `default_provider`. Selection
  priority, highest first: `--models`, then `--interactive`, then `--quick`,
  then `defaults.models` from YAML, then interactive fallback.
- **Lists replace, they don't append.** Set `defaults.models` and you get exactly
  that list. Omitting or blanking a key inherits the base; an explicit empty list
  `[]` is the only way to clear one — and an empty `defaults.models`/`quick_models`
  falls back to interactive selection (or fails in unattended runs like `--full`,
  `--improve-plan` and `--improve-code`, whose review always runs under `--yes`).
- **Per-mode lists win.** Modes with a base `defaults.modes.<mode>` entry (e.g.
  `--review-plan`, `--review-code`) ignore a project-wide `defaults.models` —
  override `defaults.modes.<mode>` to change those. The `--improve-*` modes, like
  `--full`, have no `defaults.modes` entry of their own: their review reads
  `defaults.modes.review-plan` (`--improve-plan`) or `defaults.modes.code-review`
  (`--improve-code`; the key is `code-review`, not `review-code`), so that is the
  entry to set to change an improve run's reviewers. An `improve-plan` or
  `improve-code` entry is read by nothing.
- **Requires a git repo.** Discovery resolves the git root, so there's one override
  per repo, shared repo-wide (no per-subdirectory configs). Non-git projects must
  use `MULTI_LLM_PROVIDERS_CONFIG` instead.
- **Commit or ignore?** Commit it for a team-wide standard; ignore it (`--gitignore`)
  or use the env override for personal preferences.
- **`driver.verify_command` is this repo's own check**, and the one `driver.*` key
  worth setting per project: a shell string (`pnpm typecheck && pnpm lint:fix`,
  `cargo check`, `make test`) that the code-editing phases — `implement` and
  `apply-code-fixes` — run **once**, from the repository root, after their last
  batch applied. Left unset, those phases say so in one line rather than completing
  unchecked; a non-zero exit reverts nothing (every batch was already verified
  against its own snapshot) but ends the phase at exit 3 with the command and the
  tail of its output. `driver.verify_timeout_s` bounds it (default 600 s), and
  the bound is on the command's whole process group rather than on the
  `/bin/sh` that started it, so a hung `tsc` is killed rather than orphaned.
  The markdown apply phases never run it.
  Set or not, every `apply-code-fixes` batch is also told the checks this
  repository is known to have (this command, the plain test, lint, type-check
  or build commands the finished tasks of the task list implement ran name,
  `package.json`'s test script) and asked to run them after its edits, in its
  own sandbox and never as root, skipping any it judges unsafe, and say what
  each did. A task-named command that could deploy, migrate, seed, delete,
  reach the network or start a server or watcher is never listed, and neither
  is one added to the task list after implement ran. The plugin itself runs
  none of them; a failing one is reported as that batch's failing check, and
  when this key is unset the closing line says what each batch reported for
  each command.
- **`progress.interval_s` / `progress.poll_s` price two different cadences.**
  `interval_s` (default 30 s) is how often a running fan-out prints a
  `[PROGRESS]` ticker line — a stdout cadence only, so `interval_s: 0` is the
  documented **disable form**: the lines stop, while the run's status file still
  records every transition and `--wait` is untouched. `poll_s` (default 90 s) is
  how often the agent host polls that run, and the resolved `--wait` default — a
  poll costs a **tool turn**, which is why it is a separate knob rather than the
  same number. It is **clamped under the host's shell cap** (the tightest
  surveyed ceiling is OpenCode's 120 s, less a 30 s margin — which is where 90
  comes from), so raising it past that cap buys nothing. Both floors are 5 s;
  `poll_s` has **no** disable form (`0` is rejected, since it would mean a poll
  that returns instantly rather than one switched off), and an unknown key under
  `progress:` is rejected at load rather than silently keeping the default.
- **`code_review.collapse_paths` decides which changed files reviewers skim.**
  `review-code` sorts its scope into source, tests, fixtures and generated
  files, in that order, and writes a whole per-file diff for every source and
  test file to its phase dir (`code-review/review-scope.md` and
  `code-review/diff/`). Fixtures and generated files are listed one line per
  directory, with no diff. A path is generated when `.gitattributes` gives it
  `linguist-generated`, or when it matches a `generated` glob here; it is a
  fixture when it matches a `fixtures` glob. The defaults are `generated: []`
  and `fixtures: ["**/tests/fixtures/**", "**/tests/golden/**"]`; a project can
  add, say, `**/*.lock` or `dist/**` so a huge machine-written file is never
  handed to reviewers as a diff. A malformed value is rejected at load.
- **`code_review.commit_reviews_dir` is where commit reviews live.**
  `--review-commits`, `--apply-commit-fixes` and `--improve-commits` write each
  range's commit brief to `<dir>/<slug>.md` (the slug is `<base12>-<head12>`,
  or `root-<head12>` for a range that starts at the first commit) and its
  `state.json`, phase folders and reports to `<dir>/<slug>/`. The default is
  `plans/commit-reviews`.
  - **Relative to the repository root**, whichever layer sets it: never to the
    current directory or to the config file's folder, so a
    `MULTI_LLM_PROVIDERS_CONFIG` file kept elsewhere and a run started from a
    subdirectory name the same folder.
  - **Refused at load**, in permissive mode too, naming the key and the value:
    anything that is not a string, an empty string, an absolute path (a
    leading `/`, or a drive prefix such as `C:`; `\` is read as `/`), a path
    with a `..` segment, and `.` (the repository root itself). `null` keeps the
    default. A folder that a symlinked component resolves outside the
    repository is refused too, naming the resolved path, before a commit mode
    writes anything.
  - **A new folder ignores itself.** When the plugin creates the folder, it
    writes `<dir>/.gitignore` holding `*`, so briefs and reports never show in
    `git status` and a `git add -A` cannot stage them. Only the folder itself
    gets one, never a parent created with it such as `plans/`.
  - **To commit reviews, delete that `.gitignore`**, or point the key at a
    tracked folder. The plugin never writes a `.gitignore` into a folder it did
    not create and never restores one you deleted. While the folder exists and
    git does not ignore the brief (a tracked brief counts as not ignored), a
    commit mode prints `[COMMIT_REVIEWS_NOT_IGNORED] <dir> is not ignored by
    git: …` before it writes anything, naming `<dir>/` as the entry to add to
    `.gitignore` if committing them is not intended, and the run continues.
  - **Briefs and report folders are never code under review.** Every review,
    whatever its mode or plan, leaves `<dir>/<slug>.md` and everything under
    `<dir>/<slug>/` out of its scope and out of `tracked_files`, also when a
    reviewed range holds a commit that added them, and its `Left out N of the
    workflow's own file(s)` line names them. Anything else you keep in the
    folder (a README, notes, a script) is reviewed like any other file.

## Environment variables

These are the `MULTI_LLM_*` variables a person may set. Codex starts the
plugin's tool server with a stripped environment: of these variables, only the
ones the plugin's `.mcp.json` lists in `env_vars` arrive from the environment
Codex itself was started in. The "Codex manifest" column says which. The server
then starts every run through a `bash -lc` login shell, which reads
`~/.bash_profile` or `~/.profile`, so a variable exported there reaches the run
whether the manifest lists it or not, and overrides a forwarded value (see
[hosts](hosts.md)). `MULTI_LLM_CODEX_UNSANDBOXED_WRITES` is the exception: the
server reads it itself before any shell starts, so only a value in Codex's own
environment counts. The variables the plugin sets for itself (handshakes,
probes, host inputs) and the build and test switches are listed, with their
owners, in `skills/multi-llm/utils/env_vars.py`.

<!-- BEGIN GENERATED: environment-variables (utils/env_vars.py) -->
<!-- Generated from the `setting` rows of utils/env_vars.py. Edit the row, then run `python utils/env_vars.py`. -->

| Variable | What it does | Codex manifest |
|---|---|---|
| `MULTI_LLM_PROVIDERS_CONFIG` | Path to a `providers.yaml` deep-merged over the project and user config, above both. A malformed file fails fast (see the next row). | forwarded |
| `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE` | `1` (also `true`, `yes`, `on`) turns a malformed explicit override file into a warning and skips it instead of failing fast. | forwarded |
| `MULTI_LLM_GIT_CREDENTIAL_HELPER` | One-run override of the git credential-helper policy for runner children: `auto` (default), `wipe` or `keep`. Wins over both config layers; anything else warns and uses `auto`. | forwarded |
| `MULTI_LLM_HOST` | Host id used when `--host` is not passed (for a host that calls `multi_llm.py` without a rendered shim). Default `claude-code`; an unknown id exits 3. | forwarded |
| `MULTI_LLM_UV` | Explicit path to `uv`, tried before `PATH` and the documented probe chain. Also used to re-run a detached worker. | forwarded |
| `MULTI_LLM_UV_RUN_ARGS` | Replaces the flags passed to `uv run` (split on whitespace, no shell). Unset or empty keeps the default; `--no-sync` is correct only against an already-built `.venv`. | forwarded |
| `MULTI_LLM_MAX_DETACHED` | Ceiling on concurrent detached runs per project (default 4, minimum 1). The `--max-detached` flag beats it. | forwarded |
| `MULTI_LLM_HEARTBEAT_INTERVAL_S` | Seconds between a detached worker's heartbeats (default 10, minimum 0.05). Mostly for tests and slow disks. | forwarded |
| `MULTI_LLM_HANDSHAKE_TIMEOUT_S` | Seconds `--detach` waits for the worker to confirm it started (default 10, minimum 0.1) before reporting the launch as failed. | forwarded |
| `MULTI_LLM_VALIDATION_CONTEXT` | How much plan and code each validation batch embeds: unset (whole plan plus the code its issues cite), `cited` (plan scoped per batch) or `full` (round-wide blocks in every batch). Anything else reads as `full`. | forwarded |
| `MULTI_LLM_CODEX_UNSANDBOXED_WRITES` | Codex plugin only: `1` lifts the tool's own refusal of write-capable modes under a read-only or unreadable session posture. It never changes which sandbox the runners use. Set it in the environment that starts Codex. | forwarded |
| `MULTI_LLM_CODEX_SKIP_RUNG` | `plugin-landlock` skips that sandbox rung alone for the process; any other value is rejected at start-up. Use it to tell a jail denial from a codex fault. | not forwarded |
<!-- END GENERATED: environment-variables -->

## Safety

A repo's override is auto-loaded, so a cloned repo can change which models a run
selects — but never *what executes* on the model path: provider binaries are
hardcoded and the config `command` field is never run, in any layer. The one key
that *is* executed is `driver.verify_command`, and only by a write phase you asked
for, after it edited that same repo with an agent CLI — the run prints the command
and the config layer it came from before it starts, and again with its exit code
once it ends. Treat an untrusted repo's override the way you would its `Makefile`
— and the blast radius is not only the override file: the *scripts the command
invokes* (`package.json`'s `typecheck`, a `Makefile` target) are files the write
subagent was just allowed to edit, and they run outside whatever sandbox the
agent CLI had. The check runs with the host's own session variables stripped and,
when `env_allowlist` is on, with the same deny-by-default environment a runner
gets; otherwise it inherits the environment you launched the run with.
One credential makes that filter's shape load-bearing for a user's own setup:
`omp` is the first adapter whose key is admitted by neither its own `PI_`/`OMP_`
prefix nor the shared base list, so the adapter passes `OPENROUTER_API_KEY`
through **explicitly** — a filtered run still carries it — and omp then resolves
it through its own chain: the process environment, then a `.env` in the directory
the run starts from (a checked-out repo's own key therefore outranks the one you
exported), then `~/.omp/agent/.env`, then `~/.omp/.env` or `~/.env`, of which
`~/.omp/agent/.env` is the source that depends on nothing in the parent
environment and the one to use when the key is not exported in the shell the run
starts in.

`fx` has a separate context boundary worth understanding before enabling it.
The adapter supplies a managed `HOME`, so fx ignores the user's existing fx
configuration and settings. Its context setup suppresses `AGENTS.md` injection;
skill catalog metadata may still be visible, while the `skill` loading tool is
denied. This does not provide an OS sandbox or block web access. In `--auto`,
shell commands may trigger an additional, billed reviewer call; that reviewer's
usage is not included in fx's reported token counts or cost because fx returns
no cost field. The reviewer may deny a shell command, and its usage can vary.
See [the fx provider notes](providers.md#the-provider-table).

The override now deep-merges its full contents (including a `providers:` block)
over base, exactly like the `MULTI_LLM_PROVIDERS_CONFIG` env layer — but since
`command` is documentation-only, a merged block can only retune metadata
(timeouts, model lists), and a provider name with no hardcoded adapter is
ignored at runtime. A present-but-malformed explicit override fails fast with a
clear error naming the file; set `MULTI_LLM_PROVIDERS_CONFIG_PERMISSIVE=1` to
warn-and-skip instead.

One provider key does change *how* a binary executes:
`providers.codex.sandbox_fallback` (`auto` by default; `contained` or `off` to
narrow it). codex is the only reviewer that insists on its own OS sandbox —
`codex exec` runs the model's shell through bubblewrap, and on a stock Ubuntu
24.04 bwrap cannot start (`multi_llm.py doctor` prints the AppArmor repair).
Once the sandbox probe has failed, codex's **read-only** work (the review/ask
fan-out and the executor's validation/consolidation/salvage kinds) walks a
three-rung ladder. Rung 1, `landlock`, runs `--sandbox read-only` with a
per-dispatch `-c features.use_legacy_landlock=true` — codex's second Linux
backend, which needs no user namespace — and is taken only when its probe's
canary write is denied by the kernel; the slot stays sandboxed and says
so (`[SANDBOX_FALLBACK]` on stdout, `sandboxed by fallback rung: landlock` on
the report's model labels, `sandbox-fallback landlock <spec>` in `status`).
Rung 2, `plugin-landlock` (since 2026-09-30), is the plugin's own Landlock jail
around codex, described [below](#the-plugins-landlock-jail): it stops stray
writes, not the model, so its grade is `incidental`, never contained, and its
slots say so (`[SANDBOX_INCIDENTAL]` on stdout, `stray writes stopped by
plugin-landlock; not a boundary against the model` on the report's model labels,
`stray-writes-only plugin-landlock <spec>` in `status`). Rung 3 runs
`--sandbox danger-full-access` in the real
repo root, i.e. with the same rights as your shell — the footing cursor-agent,
pi, claude-code and muse have always had in those phases — and every artifact
of such a slot says so (`[SANDBOX_BYPASSED]` on stdout,
`ran unsandboxed: sandbox_fallback=auto` on the report's model labels, a
`git status` check that fails the slot by path if the tree changed). `auto`
takes rung 1, then rung 2, then rung 3; `contained` takes rung 1 only and
otherwise refuses — rung 2 is not the real OS boundary that value promises, so
`contained` never takes it; `off` restores the refusal: codex is left out on
such a machine until the repair is installed. A run that falls to rung 3 where
the previous run of the phase took rung 1 says so once, on its own
`[SANDBOX_RUNG_RETIRED]` line (one that left rung 2, on a
`[SANDBOX_RUNG_CHANGED]` line) — never a second `[SANDBOX_BYPASSED]`, which
stays one line per slot. doctor's BROKEN row names the value in force and the
rung it selected.

The write phases follow a narrower rule. Where codex's own `workspace-write`
sandbox starts, they use it, whatever the key says. Where it cannot start, they
run on rung 2 alone, and only under `auto`: `apply-suggestions` (scope
`plan-file`: the jail lets codex write the plan file and nothing else in the
repository, in place), `apply-task-suggestions` (scope `tasks`: the default
`plans/<stem>/tasks/` directory, else the tasks file) and, since 2026-10-02,
`apply-code-fixes` and `implement` (scope `repo-wide`: since 2026-10-05 one
rule on the repository root, so the whole repository is writable inside the
jail — `.git`, `.codex`, `.multi-llm` and the run's own output folder
included. Nothing is backed up or restored around a batch; the model is told
to leave those alone, `driver.write_paths` is still enforced before and after
each batch, the sweep names undeclared writes, and one warn-only
`[GIT_CONFIG_CHANGED]` line says when `.git/config` or `.git/hooks/` changed
during a batch — inspect them before your next git command, since your own
git honours whatever was planted; the plugin's own git calls do not. Until
then the grant enumerated entries outside that protected set and refused a
task declaring a new top-level file before dispatch, which is why a plan whose
whole deliverable was one root-level `index.html` never ran on Codex). Each
batch on the rung prints
the stray-writes-only line first, which also says network access is not
confined there (codex's own `workspace-write` confines it). `contained` and
`off` refuse them, and a write never falls to rung 1 or to rung 3 — with one
exception: when the codex chat that started the run is itself in
`danger-full-access`, a plan-file or tasks write the jail could not take runs
on rung 3 under `auto`, because you already gave that chat unconfined write
access; a repo-wide batch never takes rung 3, whatever the chat's posture. The
refusal leads with the way out that needs no change to the
machine — set `driver.subagent_runner` to a runner that is not Codex (for
example `claude-code:<model>`) in `.multi-llm/providers.yaml` — then says why
codex cannot run here and why the jail did not take the job, and ends "Nothing
was dispatched." The AppArmor repair `doctor` prints is the system-level
alternative: it needs `/etc` and sudo, and gives back codex's own
`workspace-write` for every write phase.

**Never enable landlock globally.** `codex features enable
use_legacy_landlock` makes every `workspace-write` dispatch panic (exit 101),
write phases included; the flag is only safe as the per-dispatch `-c` rung 1
adds, and `doctor` warns if your codex config sets it. Rung 1 is read-only only
and rides a feature codex marked deprecated (measured at 0.155.1) and that has
been unusable since 0.156.1 (the rung's probe panics with exit 101; measured,
while codex 0.160.0 still lists the feature): on a current codex the walk skips
it by version, `auto` goes on to rung 2 and `contained` refuses.

Because the default is already the widest value, a cloned repository's
`.multi-llm/providers.yaml` can only narrow the fallback — there is nothing for
an untrusted override to widen here. What rung 3 gives up, in plain words: for
a codex read-only slot, no OS boundary of any kind between the model's shell and
this machine (reads and writes anywhere you can, network from model commands,
and no prompt-injection resistance for a validation batch beyond the `git
status` check). That is the footing the other reviewers already run on
(measured 2026-09-17: the host's own shell tool runs in the same namespaces as
your terminal), which is why `auto` keeps it as the last rung rather than
refusing; set `contained` to keep the boundary and drop the bypass.

**codex's trust entries.** codex adds `[projects."<repo>"] trust_level =
"trusted"` to your `$CODEX_HOME/config.toml` whenever it runs with write access
(`workspace-write` or `danger-full-access`, jailed or not) in a repository you
never trusted — the codex children a multi-llm run starts included — which
would make your later codex sessions honour that repository's project config.
Since 2026-09-30 a run records the file's state before its first codex child
starts and, when it ends — normally, on an error, on Ctrl-C or on SIGTERM —
removes only the tables it added that still hold exactly that line, leaving
every other byte and the file's mode as they were. A table changed during the run is left, with a warning. A SIGKILLed run,
or two overlapping runs in one repository, can still leave an entry behind.

### The plugin's Landlock jail

Rung 2 (since 2026-09-30) confines codex from outside: a small wrapper,
`utils/landlock_exec.py`, applies a Landlock ruleset to its own process and then
`exec`s codex under `--sandbox danger-full-access`. Inside it, codex may write
only to `$CODEX_HOME`, a fresh 0700 `TMPDIR` made for each attempt under your
temp directory (and removed afterwards), `/dev/null` and, where they exist,
`/dev/ptmx` and `/dev/pts`. `GIT_OPTIONAL_LOCKS=0`, `PYTHONPYCACHEPREFIX` and
`npm_config_cache` send git's, Python's and npm's incidental writes into that
`TMPDIR`; a repo-wide batch also gets `UV_CACHE_DIR` and a pytest `cache_dir`
there, so `uv run pytest` and `npm test` complete inside the jail, while
`/tmp` and `/dev/shm` stay denied. A write phase adds its grant: the plan
file, the tasks directory or file, or — for `apply-code-fixes` and
`implement` — one rule on the repository root (since 2026-10-05; Landlock is
allow-only and a rule inherits down the tree, so that rule covers `.git` and
`.multi-llm` too). The allow-list is not configurable.

- **When it is used.** Only after codex's own sandbox probe says `unavailable`;
  a sandbox that starts is used unchanged. The kernel must offer Landlock ABI 3
  (Linux 6.2) or newer. Without it (an older kernel, or the Docker image) the
  wrapper exits 126, the rung is recorded absent, and `auto` moves on to rung 3.
  Its own probe runs `codex sandbox -c sandbox_mode="danger-full-access" -- sh
  -c <canary>` inside the jail and passes only when a write outside the
  allow-list and a truncation of an existing file are both denied and a write
  inside it lands.
- **What it does not stop.** Writes through Unix sockets (`systemd-run --user`,
  D-Bus, Docker), mode and timestamp changes on existing files, writes to
  `$CODEX_HOME`, and the network. It stops stray writes, not a model set on
  escaping, which is why no artifact calls it contained or sandboxed.
- **What checks the rest.** A review slot on the rung skips the `git status`
  check and runs two narrower ones. `[MODE_CHECK]` fails the slot when a tracked
  file's executable bit flipped, and `[CODEX_HOME_CHECK]` fails it when
  `$CODEX_HOME`'s config, instruction or plugin files changed, unless the slot's
  own audited command stream clears it. In the executor lane (validation, task
  generation, the write phases) the home check only warns.
- **A denied edit is never recorded as applied.** Codex can finish its turn
  normally with every write denied, so the scheduler reads the turn's stream:
  a batch whose edit was denied fails, and one that moved some bytes is a
  partial application (exit 3) that keeps its pre-batch snapshot and leaves
  the phase open. The plan-file grant allows in-place edits only: `sed -i`, a
  save through a temporary file and a rename, and `mv` onto the plan are
  denied.

#### When the jail denies a turn

A codex upgrade that writes a new path, or a tool the probe never started, can
make codex itself fail inside the jail with a permission error. That slot or
dispatch fails with one `[JAIL_DENIED]` line, in the run log and again in the
run's closing summary. It names the denied path (or says codex did not name
one), the allow-list, and the installed and measured-through codex versions,
flagging a newer installed codex as the likely cause. It is not a model error:
it is never retried and never moved to another rung. The line prints the next
steps:

1. **Compare.** `multi_llm.py doctor` shows a `plugin-landlock` row under a
   BROKEN codex line: the probe result, the resolved base allow-list, both
   codex versions and the switch in step 3.
2. **Report.** Open an issue at
   [github.com/beastlabai/multi-llm-plugin/issues](https://github.com/beastlabai/multi-llm-plugin/issues)
   with the `[JAIL_DENIED]` line and that row. A missing entry is fixed in the
   plugin, never by your config.
3. **Run the job anyway.** First, point the failing job at a runner that is not
   Codex (for example `claude-code:<model>` in `.multi-llm/providers.yaml`;
   `driver.subagent_runner` for editing jobs). That widens nothing. Otherwise,
   for one run, set `MULTI_LLM_CODEX_SKIP_RUNG=plugin-landlock`. It skips this
   rung alone, and nothing about it outlives the process:
   - a read-only phase under `auto` walks on to rung 3 (`[SANDBOX_BYPASSED]`
     and the `git status` check; the line names the variable);
   - `contained` and `off` refuse as before, and so does a write phase
     (`The plugin's Landlock jail does not take this job: skipped by
     MULTI_LLM_CODEX_SKIP_RUNG`), unless the full-access chat exception above
     applies;
   - any other value stops every mode at start-up with exit 3, naming the one
     accepted value. `doctor` reports it on its row instead, and shows the rung
     as skipped while the variable is set.

## Base config reference

The shipped [`providers.yaml`](../skills/multi-llm/providers.yaml) keeps one
short comment per key: the values it takes and what you must know to set one.
This section carries the reasons. Every key here can be set in any layer
(see [Good to know](#good-to-know) for the order); a key the base does not
declare merges and is then ignored.

### Provider blocks

Each `providers.<name>` block is metadata for one built-in adapter. The binary,
every flag and every argument live in the adapter under
`skills/multi-llm/utils/providers/`. Two keys that used to sit here,
`command:` and `supports_json_output:`, were removed because nothing read
them; a `command:` in any layer still merges and still never changes which
program runs.

- **`default_timeout`** is the slot timeout in seconds, and the only
  per-provider number. Several adapters derive an inner deadline from it and
  hold it under the slot, so the CLI reports its own error before multi-llm
  kills it: agy's `--print-timeout` and qwen-code's `--max-wall-time` sit 30 s
  under it, and claude-code's `API_TIMEOUT_MS` is a quarter of it less 30 s,
  capped at 300 s, so its retries fit inside the slot. Raising the key raises
  those too. The shipped values carry measurements: muse's 1800 s is generous
  because muse retries a failing call up to 10 times with exponential backoff
  (a 429 alone waits 60 s), so an unfunded account burns minutes before it
  reports the 402, and the slot holds one whole chain; copilot's 1200 s
  clears its own ~86 s rate-limit chain; kimi's bounds a stalled upstream that
  only kimi's own ~300 s request timeout ends; qwen-code's bounds its
  unbounded 429 and stall paths through the derived `--max-wall-time`; vibe's
  clears its declared inner bound (3 × 90 s plus backoff). A provider with no
  row gets `DEFAULT_TIMEOUT_S` (1200 s, in `utils/providers/base.py`); the
  adapters declare no timeout of their own, so there is no second number to
  disagree with this one.
- **`max_concurrent`** caps that provider's parallel slots. cline's `-m`
  rewrites the saved default in `~/.cline`, which is why cline is capped.
- **`models`** is the `--interactive` picker's menu, the first entry being the
  provider's default ([a menu, not a gate](providers.md#the-provider-table)).
  Keep each CLI's own spelling: a model OpenRouter serves may not exist in a
  native CLI's catalog, and the reverse. The catalogs list with
  `codex debug models`, `cursor-agent models`, `agy models`, `grok models`,
  `opencode models`, `kilocode models <backend> --verbose`,
  `omp models openrouter --json` and `pi --list-models`; OpenRouter slugs are
  at `https://openrouter.ai/api/v1/models`. The repository's
  `tests/test_model_catalogs.py` checks configured ids against committed
  catalog snapshots where one exists.

**The `:effort` suffix.** Most providers accept `model:effort`, split on the
last colon against that provider's accepted names; any other suffix stays part
of the model name, so `:free` ids and Bedrock `...-v1:0` ids pass through
whole. The accepted names come from one canonical ladder (none, minimal, low,
medium, high, xhigh, max) in `utils/providers/base.py`, of which each adapter
declares its subset as `EFFORT_LADDER`; `off` is a CLI's spelling of `none`
(goose, pi, vibe), `ultra` of `max` (muse; codex accepts it and sends `max`),
and `thinking` / `instant` (kilocode, opencode) are a mode rather than an
amount and pass through uncompared. Where a provider differs:

| Provider | The suffix becomes | What an unsupported value does |
| --- | --- | --- |
| claude-code | `--effort <level>` | the CLI downgrades it per model, silently |
| cursor-agent | no suffix: effort is part of the model name (`-xhigh` variants) | stays in the name |
| gemini | no suffix and no per-run control: define a custom alias with a `thinkingConfig` under `modelConfigs.customAliases` in `~/.gemini/settings.json` (or the project's `.gemini/settings.json`) and list the alias as the model | stays in the name |
| codex | `-c model_reasoning_effort=<level>` | `max` is advertised per model by `codex debug models` (`supported_reasoning_levels`); `none`/`minimal` are accepted on the wire but advertised by no catalog entry, so pass one only when you mean it |
| grok | `--reasoning-effort <level>` | per-model support varies |
| opencode | `run --variant <name>` (names are provider-specific) | resolved or ignored server-side; one model that hangs on a variant is declared in the adapter and loses the suffix with a warning |
| kilocode | `run --variant <name>` | **ignored silently**: variants are per model and drift, so check each one against `tests/fixtures/kilocode/model_variants.json` |
| cline | `--thinking <level>` (does not touch the saved default) | cline exits 1; `xhigh` may be downgraded to `high` on some backends |
| goose | `GOOSE_THINKING_EFFORT` in the child's environment (goose has no flag) | ignored silently, as is any effort for an OpenRouter model goose does not class as reasoning-capable |
| aider | `--reasoning-effort <value>` | passed unvalidated: `openrouter/` models accept it as OpenRouter's `reasoning.effort`, others warn and ignore it |
| agy | `--effort low\|medium\|high`, beside the tier its display name already carries | stays in the name, which agy then rejects out loud |
| muse | `--reasoning-effort <level>` | muse hard-errors; without a suffix muse reads `reasoning_effort` from `~/.config/muse/settings.json`, not its `--help` default |
| pi, omp | `--thinking <level>`, stripped before `--model` so the CLI's own `:level` parsing never sees it | omp validates nothing, so the whitelist is the only guard; omp's own `:auto` stays in the name and reaches omp |
| kimi, qwen-code | no suffix: a third `:` segment is part of the id | qwen-code sends it to the backend verbatim, which answers "model not found"; kimi's `KIMI_MODEL_THINKING_EFFORT` is inert on the wire at 0.36.1 |
| copilot | the effort setting (all seven accepted at 1.0.80) | which ones a backend honours is server-side |
| vibe | `VIBE_THINKING` (vibe has no flag) | split anyway, since an alias with `:high` glued on selects no model |

### Version ceilings

Every supported build of every CLI is decided by one gate in
`utils/providers/base.py`. The floor (`min_version`) and the build the
fixtures came from (`verified_version`) are declared on the adapter, next to
the parser they protect. The ceiling, `max_version_exclusive`, is declared on
the adapter too and can be overridden per provider in config, because a
vendor can ship a broken build between two releases of this plugin and
pinning it must not need a code edit. Below the floor, or at or above the
ceiling, a slot is refused before any child starts, naming the number; above
`verified_version` is a warning, never a block.

A number ships with a comment naming the feature it protects (or the breakage
it avoids), never alone: a ceiling with no reason cannot be retired, because
nobody can tell what would retire it. A version measured once is an upper
bound on the true floor, so a flag verified on one recent build is gated per
flag (the CLI's own `--help`, read once per process) rather than by raising
the floor. The key is written out on opencode, shipped empty (no ceiling),
because opencode's npm `latest` has been a prerelease: e.g.
`max_version_exclusive: "2.0.0"` refuses every slot on 2.0.0 and later.

### claude-code keys

- **`fallback_models`** (and **`fallback_models_default`** for primaries
  with no entry) map a primary model to the stand-ins claude-code's
  `--fallback-model` uses when the primary is overloaded (three consecutive
  529s) or unavailable (a 404 falls back on the first error). It ships empty,
  and empty emits no flag: nothing here knows which model should stand in for
  which, and an implicit list would answer with a model nobody configured.
  The list is joined verbatim in your order. The CLI accepts a bad list
  silently, so it is validated at load: an entry that repeats, is empty, names
  a model not in `models:`, or is its own primary fails the run naming the
  key. A fired fallback is recorded as `served_model`, read from the
  envelope's `modelUsage`.
- **`strip_vendor_keys: auto | always | never`**. Measured at claude-code
  2.1.233: with a subscription login present, an exported `ANTHROPIC_API_KEY`
  swaps the request's subscription bearer token for `x-api-key` and the run is
  billed per token. `ANTHROPIC_AUTH_TOKEN` overrides it the same way, and
  `ANTHROPIC_CUSTOM_HEADERS` is a third path. `auto` strips those three only
  on an affirmative subscription verdict from a sanitized `claude auth status`
  (run once per process with the three already removed, no network, about
  0.3 s, and never for a run with no claude-code slot or no such variable
  set). Every other outcome keeps them, with one warning naming the billing
  consequence, because failing closed would break every claude-code slot of an
  API-key-only user. `always` strips without asking; `never` keeps them.
  `ANTHROPIC_BASE_URL` is never stripped, and its presence makes `auto` keep
  the keys: a gateway user's credentials belong to the gateway
  ([endpoint recipes](providers.md#pointing-a-provider-cli-at-a-different-endpoint-glm--zai-and-the-general-case)).
  `OPENAI_API_KEY`, `CLAUDE_CONFIG_DIR` and `CODEX_HOME` are never stripped.
- **`vendor_backend: auto | keep | strip`** is a second, independent gate
  over the seven `CLAUDE_CODE_USE_*` names (BEDROCK, VERTEX, FOUNDRY,
  ANTHROPIC_AWS, GOOGLE_CLOUD, MANTLE, GATEWAY). An inherited one is worse than
  a billing bug: `CLAUDE_CODE_USE_BEDROCK` hung for 137 s with no request and
  no output, waiting on AWS credentials. `auto` strips one that arrived by
  inheritance, meaning one not named in this provider's own `env:` block
  (`env: {CLAUDE_CODE_USE_BEDROCK: "1"}` declares it yours; the seven names
  are the only ones such a declaration may name). `keep` never strips it, its
  hang then bounded by the slot timeout; `strip` always does. The two gates
  are independent: `vendor_backend: keep` still gets the credential row
  evaluated, and `strip_vendor_keys: never` still removes an undeclared
  backend.

### cursor-agent keys

**`isolate_profile: auto | on | off`** gives each run its own cursor profile:
`CURSOR_CONFIG_DIR` and `CURSOR_DATA_DIR` pointed at a 0700 temporary
directory removed at exit. Without it, `--model` rewrites your global default
model in `cli-config.json` and `--trust` writes a lasting
`.workspace-trusted` marker; a snapshot-and-restore guard puts them back, but
only within one process, so two overlapping runs can leave your default model
changed for good. `auto` redirects only where cursor's login is verified not
to follow those two variables, which is Linux today (measured: `auth.json`
lives under `$XDG_CONFIG_HOME/cursor/`, and a run against an empty redirected
directory authenticated). `on` redirects everywhere; where cursor finds its
login differently, every cursor slot then looks logged out, which is why it is
not the default. `off` keeps your profile, for a project rule set or an MCP
config cursor must see. Setting `CURSOR_CONFIG_DIR` or `CURSOR_DATA_DIR`
yourself also turns the redirect off, and that profile is never written or
deleted.

### opencode keys

- **`hermetic: on | auto | off`**. `on` and `auto` run opencode under an
  empty, run-scoped `XDG_CONFIG_HOME` plus `--pure`, and move
  `OPENCODE_TEST_HOME` there too, so neither your
  `~/.config/opencode/opencode.json`, `~/.opencode/` nor the user-level skills
  under `~/.claude/skills` change the model, agent or permissions a dispatch
  uses. `XDG_DATA_HOME` (stored logins, `opencode auth list`) and
  `XDG_CACHE_HOME` (the model catalog) are kept. It applies to fan-out, `ask`
  and `agent-readonly` slots; `agent-write` inherits your config. The cost: an
  MCP server opencode starts also sees the empty config home; `off` restores
  inheritance for that case. Quote the value: YAML 1.1 reads a bare `on` or
  `off` as a boolean.
- **`config_providers`** is a verbatim opencode `provider` map, without
  credentials (those stay in `auth.json`). A hermetic run cannot see a provider
  defined only in your own opencode config, so declare it here; a slot naming
  an undeclared one is refused before spawn rather than failing to resolve the
  model. Built-in providers need nothing here. A declared provider passes the
  login preflight wherever its key lives: in `auth.json` under its id, in an
  `{env:VAR}` reference resolved in the dispatch's own environment (so
  `env_allowlist` applies), or nowhere (a keyless local server); the preflight
  reads `opencode debug config` before refusing. A built-in provider
  overridden without a key, or with an `{env:VAR}` that is unset, is still
  refused.
- **`opencode/*` models** are opencode Zen's server-side catalog: models.dev
  marks one deprecated and `opencode models` stops listing it, with no opencode
  release involved. `opencode/nemotron-3-ultra-free` went that way by
  2026-11-06.

### codex keys

- **`strip_vendor_keys`** is deliberately unset for codex, which means
  `never`. codex has no zero-network login verdict (`codex login status`
  reports a `CODEX_API_KEY`-only user as logged out, and `codex doctor --json`
  is advisory), and `auto` behaves as `never` for any adapter without one.
  `always` would leave a `CODEX_API_KEY`-only user without a credential, and
  codex does not fail fast then: it sends the request with no
  `Authorization` header and the turn dies at the server. `OPENAI_API_KEY` is
  never stripped: codex does not read it, and a custom provider declaring
  `env_key = "OPENAI_API_KEY"` needs it.
- **`sandbox_fallback: auto | contained | off`** is described in full under
  [Safety](#safety). A probe verdict of `unavailable` (never `unknown`) starts
  the ladder; "read-only phases" means the review and ask fan-out plus the
  executor's validation, revalidation, consolidation, salvage, task generation
  and human-decision judge. YAML booleans read as the word they spell:
  `off`/`false` is `off`, `on`/`true` is `auto`. doctor's BROKEN row names the
  value in force and the rung its probe chose.

### Model names per provider

- **cline, goose:** `<backend>/<model-id>`, split at the first `/` into the
  backend (`-P` / `--provider`) and the model (`-m` / `--model`). goose reads
  its key from the environment only (`OPENROUTER_API_KEY` for `openrouter/`).
- **aider:** a litellm spec, passed whole to `--model`; `openrouter/` models
  need `OPENROUTER_API_KEY` exported.
- **pi:** pi model patterns, passed whole; pi resolves `provider/id` itself.
  Keys come from provider variables (`OPENROUTER_API_KEY`,
  `ANTHROPIC_API_KEY`, ...) or pi's `/login`.
- **agy:** the exact display names `agy models` prints, spaces and parentheses
  included. An unknown name is refused with an error listing the valid ones
  (verified 2026-07-29; older builds fell back to the default model
  silently). It signs in with Google once (`~/.gemini/antigravity-cli/`); no
  API key.
- **kilocode:** `backend/provider/model`. Prime and ultraspeed SKUs are
  separately priced speed variants, not replacements for the standard ids.
- **kimi, qwen-code, copilot, fx:** OpenRouter slugs on each CLI's generic
  bring-your-own-key backend. A Kimi Code subscription, a Qwen first-party plan
  and a Copilot seat each have a different catalog, and their model names do
  not transfer. kimi takes the slug in `KIMI_MODEL_NAME`, never on its command
  line. copilot's 1.0.80 probe recorded a completions-parser issue with
  `reasoning_content` deltas (copilot-cli#4196), with
  `COPILOT_PROVIDER_WIRE_API=responses` as the stub-tested fallback; the
  shipped entry's model availability is checked, its compatibility with that
  parser is not.
- **vibe:** aliases, not slugs. vibe has no `--model` flag; each entry names a
  `[[models]]` alias in `$VIBE_HOME/config.toml` whose own `name` is the
  OpenRouter slug, selected through `VIBE_ACTIVE_MODEL`. A host needs that
  alias defined; the container lanes render it from the selected model.
- **omp:** selectors, passed whole to `--model`: copy the `selector` field of
  `omp models openrouter --json` (`openrouter/openai/gpt-6-luna`), not its `id`
  (`openai/gpt-6-luna`). Where omp finds its key is under [Safety](#safety).
- **muse:** model ids from `https://dev.meta.ai/docs/models` or
  `muse model-profile show <id>`; Wonder Full needs a Power Usage or Maximum
  Usage subscription. Login is stored (`muse login` / `muse auth set`).
- **claude-code:** moving aliases (`sonnet`, `opus`, ...) and explicit ids
  (`https://code.claude.com/docs/en/model-config`).
- **gemini:** `https://ai.google.dev/gemini-api/docs/models`; an API key and
  an OAuth or Code Assist login can see different models.

### `git_env`

Every provider CLI shells out to git, and a credential prompt behind captured
pipes blocks forever: measured against a private HTTPS remote, a terminal
prompt burned 75 s with two empty streams, and a GUI-style credential helper
90 s, surviving the parent's kill as an orphan. Two layers apply:

- **Always**, for every provider: `GIT_TERMINAL_PROMPT=0` and an empty
  `GIT_ASKPASS` (plus `SSH_ASKPASS`, `SSH_ASKPASS_REQUIRE` and
  `GCM_INTERACTIVE`, inert on Linux). A helper that answers still works.
- **Scoped**, set by `credential_helper`: `credential.helper=""` through
  `GIT_CONFIG_COUNT`. `auto` applies it to the read-only lanes only (the
  `generate` fan-out and `agent-readonly`), never to `agent-write`, which
  legitimately runs clone, fetch and commit against your own remotes. `keep`
  never applies it, for a read-only lane that needs a private remote (a
  private plan repository, a submodule fetch); a helper that hangs is then
  bounded only by the slot timeout. `wipe` applies it everywhere, at the cost
  of `agent-write` reaching private HTTPS remotes.

Set it globally or per provider under `providers.<name>.git_env`;
`MULTI_LLM_GIT_CREDENTIAL_HELPER` overrides both for one run, and an unknown
value in either layer is rejected at load. The accepted cost of `auto`: in the
read-only lanes a private HTTPS clone fails in about 0.3 s with
`terminal prompts disabled`, the CLI reports its own tool error, and the model
continues.

### `env_allowlist`

Every provider child sees every other provider's API key and every cloud
token in your shell. `on` replaces that with a deny-by-default list (proxy,
CA bundle, XDG, keyring and each CLI's own home variables pass) built in the
one shared child-environment builder, so both dispatch lanes get the same
list. It ships `off` because the list is only as good as the per-CLI,
per-platform matrix behind it (credential, keyring, proxy and CA bundle for
every registered adapter on three platforms), and a cell is green only when
the CLI under `env -i` plus the list reaches that capability as it does with
the full environment. No cell has been measured, so turning it on risks a
mid-run login or TLS failure on any CLI; `tests/test_env_allowlist.py` keeps
the default `off` while a Linux cell is unmeasured. Quote the value: YAML 1.1
reads a bare `on` or `off` as a boolean.
