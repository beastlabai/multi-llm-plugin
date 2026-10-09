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
grok, opencode, codex, kilocode, kimi, qwen, copilot, cline, goose, aider, agy, pi, muse, vibe, omp). With **no CLIs installed** it writes the inert
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
an untrusted override to widen here. See the key's comment in `providers.yaml`
for what each rung gives up.

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
