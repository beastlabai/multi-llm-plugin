# opencode capture rig (plan §4.1, §4.2 step 1, T4.0, T4.6-T4.8, and the parser's streams)

Plan: `plans/opencode-validator-false-rate-limit-and-denied-turn.md`.

Captured 2026-09-25, re-captured on every opencode bump, **offline, $0**. The real `opencode run` binary talks to a
localhost chat-completions stub (`stub.py`) configured as a custom `stub`
provider. It runs under a cleared environment with a scratch HOME/XDG tree,
`TMPDIR` inside the scratch tree, stdin `/dev/null`,
`OPENCODE_DISABLE_MODELS_FETCH=1`, `OPENCODE_DISABLE_AUTOUPDATE=1` and
`OPENCODE_DISABLE_PROJECT_CONFIG=1`. No model was called. Nothing under the
user's `~/.config/opencode`, `~/.local/share/opencode` or `/etc` was read or
written. The managed-config layer is reached through opencode's own
`OPENCODE_TEST_MANAGED_CONFIG_DIR` override. The org-config layer is reached by
seeding the scratch data-dir DB with an account whose URL is the stub.

**Versions.** opencode supports its newest release only, and that release is
both `min_version` and `verified_version` (the pin). The committed directory is
the pin's, every case: **1.18.35** (2026-10-08, 72 cases). A bump
(`make opencode-bump VERSION=<x.y.z>`, which drives `run.sh`) captures the new
release into `<new version>/` and deletes the old directory: git history keeps
the earlier captures. On 1.18.34 the 68 permission cases gave a `summary.tsv`
identical to 1.18.32's; on 1.18.35 all 72 cases gave one identical to 1.18.34's
(timestamps and session ids aside), with the same event keys in every stream. While the floor was lower, 1.17.15 and 1.18.32 ran every
case and 1.18.18, 1.18.23 and 1.18.30 the T4.0 subset; the tables below keep
those results where they say so.

Reproduce: `./run.sh [--bin PATH] [CASE ...]` (a thin wrapper over
`capture.py`). The case table is `capture.py:_cases()`. Output goes to
`<version>/`. No `timeout` wrapper is used. Every case ends on its own, because
the stub answers any follow-up with a final text. `capture.py` kills a run only
past a 90 s deadline and records `exit=KILLED`; no run needed that.

## Files per case (`<version>/CASE.*`)

| File | Content |
|---|---|
| `CASE.jsonl` | stdout of `opencode run --format json --print-logs --log-level ERROR` |
| `CASE.stderr.txt` | stderr, verbatim (ANSI colour codes included) |
| `CASE.exit.txt` | `exit=N` |
| `CASE.requests.jsonl` | every HTTP request the stub saw: POSTs with `tools` (bool) and the `role:"tool"` results they carried, plus the org `GET /api/config` |
| `CASE.config.json` | what the case set: argv, env, `OPENCODE_CONFIG_CONTENT`, the global `opencode.json`, the managed file, the org config and the scripted tool call |
| `CASE.debug_config.json` | `opencode [--pure] debug config` under the same env (T8.1 can take its fixture from here, §9 item 6) |
| `summary.tsv` | one row per case, derived from the files above. `capture.py --summarize-only` rebuilds it without re-running |

**Redaction.** Only three things are rewritten: the scratch root becomes
`/rig`, the stub port becomes `PORT`, and the host name becomes `rig-host`.
The host name appears only in the cases whose read of `/etc/hostname`
completed. `capture.py` then refuses to finish if the invoking user's home, the
user name, the scratch path or the host name is still present in any file it
wrote. The stub's `apiKey` is the literal `"x"`. The account tokens are
`rig-access`/`rig-refresh`.

**Reading an outcome.** Every case exits 0 except `t40_badperm_plan`,
`auth_error` and `invalid_model`.
`summary.tsv` column `tool_posts` counts the stub requests that carried tools:

- `2` means the tool result went back to the model. The last `step_finish`
  `part.reason` is then `stop` and the text is `FINAL ANSWER`.
- `1` means the turn ended on the tool call. The last reason is `tool-calls`,
  and no second request was made.

## The parser's stream cases

Four cases script a whole turn through `stub.py`'s `turns` (one entry per
model step: optional text, tool calls and usage) rather than one tool call. They
are what `OpenCodeProvider`'s parser and classifier are graded on
(`tests/opencode_fixtures.py` names them):

| Case | Shape |
|---|---|
| `run_review` | a healthy review turn on `build`: four steps (read, read + bash, write, closing text), two text parts, finish `stop`. The per-step tokens are the D3 gate's real 1.18.30 `opencode/big-pickle` reviewer's |
| `denied_turn` | run b38de8bb's shape on `plan`: a completed read whose output quotes plan text about rate limits (`rate limit`, `rate_limit_hit`, `429`), then a bash on `/etc/hostname` that `run` auto-rejects: exit 0, two `step_finish tool-calls`, no final answer. b38de8bb itself (1.18.32, a redacted real run) is in git history |
| `auth_error` | a declared built-in `google` provider with no key: one `ProviderAuthError` event on stdout, exit 1, no request to any provider |
| `invalid_model` | `--model opencode/sonnet`: the `UnknownError` wrapper on stdout and the `ProviderModelNotFoundError` "Did you mean" cause on stderr, exit 1 |

`auth_error` and `invalid_model` run with `HTTP(S)_PROXY` pointed at a closed
local port, so any egress they attempted would die; neither attempts one.

## Stream shape (identical on 1.17.15, 1.18.32, 1.18.34 and 1.18.35)

- The top-level `type` is one of `step_start`, `tool_use`, `step_finish`,
  `text` or `error`. The payload is under `part`, whose `part.type` is
  `step-start`, `tool`, `step-finish` or `text`. A top-level `reason` never
  exists; read `part.reason`.
- For a `tool_use` event, `part.state` keys are `status`, `input`, `time`, and
  either `output` (`completed`) or `error` (`error`).
- **Auto-rejected (`ask`)**: `part.state.error` is exactly
  `The user rejected permission to use this specific tool call.`. It does
  **not** name the permission or the pattern. Only `part.state.input` (e.g.
  `filePath`) is there. So T4.3's `<permission> <pattern>` can come only from
  the stderr line. Without that line, the reason has to omit the parenthetical
  (§4.2 step 4, last paragraph).
- **Denied by a `deny` rule**: `part.state.error` starts
  `The user has specified a rule which prevents you from using this specific tool call. Here are some of the relevant rules [...]`,
  followed by the matching rules as JSON, **in evaluation order, with absolute
  patterns**. This is how the built-in globs below were read.
- **stderr for an `ask`**: one line,
  `! permission requested: <permission> (<patterns, comma-joined>); auto-rejecting`,
  with ANSI bold/colour around the `!`. Examples:
  `external_directory (/etc/*)`, `external_directory (/rig/outside/*)` and
  `read (.env)`. The external-directory pattern is the parent directory plus
  `/*`. That is why a file-path allow (`s41_denyallow_file_read`) never
  matches.

## `permission.asked`, `auto-rejecting` and `blocked → stop` (READ from both binaries)

- `run`'s event loop handles each `permission.asked` event for its session. If
  `--auto` (or `--yolo` / `--dangerously-skip-permissions`) was passed, it
  replies `once`. Otherwise it prints the stderr line above and replies
  `reject`.
- In `SessionProcessor`, `shouldBreak = config.experimental?.continue_loop_on_deny !== true`.
  When a tool fails with `Permission.RejectedError` or `Question.RejectedError`,
  `blocked = shouldBreak`. A blocked step returns `"stop"`, the session goes
  idle, `run` breaks, and the process exits **0**. The stream ends
  `step_finish` / `tool-calls` (`s41_ask_read`, `t40_none_*`).
- A `deny` rule throws `PermissionDeniedError` instead, which never sets
  `blocked`. The error goes back to the model and the loop continues
  (`s41_deny_*`, `t40_block_*`).
- With `continue_loop_on_deny: true`, stderr still carries `auto-rejecting`,
  but the loop continues to `FINAL ANSWER` / `stop` (`s41_cont_read`,
  `t40_cont_*`). This is T4.3's pinned negative: stderr alone must never fire
  the check.

## How rules resolve

Evaluation is `findLast` over a flat rule list: the last rule matching both
the permission and the pattern wins, and the default is `ask`. The per-agent
list is built in this order (READ, and visible in every deny error):

1. **Shared defaults.** `"*":"allow"`, `doom_loop:"ask"`,
   `read:{"*":"allow","*.env":"ask","*.env.*":"ask","*.env.example":"allow"}`,
   and `external_directory: {"*":"ask", <tool-output>/*:"allow", <tmp>/*:"allow", <each skill dir>/*:"allow", <each config reference>/*:"allow"}`.
2. **The agent's own overrides.** `plan` adds `external_directory {<data>/plans/*:"allow"}`
   (plus edit/question/task rules). `build` adds no `external_directory` rule.
3. **The user's merged top-level `permission`.**
4. **`agent.<name>.permission`.**
5. **An appended `external_directory {<tool-output>/*:"allow"}`**, skipped only
   when a rule already denies exactly that glob.

Config layers deep-merge, and **a key that already exists keeps its first
position**. The merge order, READ from 1.18.32:

1. well-known remote configs;
2. global `config.json`, `opencode.json`, `opencode.jsonc`;
3. `OPENCODE_CONFIG`;
4. project files (unless `OPENCODE_DISABLE_PROJECT_CONFIG`);
5. `.opencode` dirs and `OPENCODE_CONFIG_DIR`;
6. **`OPENCODE_CONFIG_CONTENT`**;
7. **active-org account config**;
8. **managed dir** (`/etc/opencode` on Linux, or `$OPENCODE_TEST_MANAGED_CONFIG_DIR`);
9. macOS managed preferences (a no-op on Linux);
10. **`OPENCODE_PERMISSION`**, merged into top-level `permission`;
11. the legacy `tools` block, merged underneath `permission`.

Rows 7 and 8 load on 1.17.15 as well (measured below).

## T4.0 version matrix: config acceptance and effect (measured while the floor was 1.17.15)

The config under test is the §4.2 step 1 block under `agent.<plan|build>.permission`,
with and without `experimental.continue_loop_on_deny: true`. The target is
`/rig/outside/f.txt`; `t40_env_*` read `<cwd>/.env` instead.

| Case (plan and build) | 1.17.15 | 1.18.18 | 1.18.23 | 1.18.30 | 1.18.32 |
|---|---|---|---|---|---|
| `t40_none` (no content) | auto-reject, ends `tool-calls` | same | same | same | same |
| `t40_cont` (key only) | auto-reject, **continues** to `stop` | same | same | same | same |
| `t40_block` | deny returned, `stop` | same | same | same | same |
| `t40_blockcont` | deny returned, `stop` | same | same | same | same |
| `q2_ustar_agentlayer` | deny returned, `stop` | same | same | same | same |
| `t40_env_none` / `t40_env_block` | auto-reject `read (.env)` / deny returned, `stop` | — | — | — | same |

- **Accepted, not ignored, not warned about.** On every build run:
  `external_directory`, `doom_loop` and `read` are named schema keys, and
  `agent.<name>.permission` is a schema field. Stderr is empty on every block
  case. `debug config` prints the block and `experimental.continue_loop_on_deny: true`
  back unchanged. The deny takes effect.
- **What rejection would look like** (the controls):
  - `t40_badperm_plan`: an invalid action is **fatal**. It exits 1 with
    `Error: Configuration is invalid at OPENCODE_CONFIG_CONTENT ↳ Expected PermissionActionConfig, got "maybe" agent.plan.permission.external_directory`,
    and no request reaches the stub.
  - `t40_badpermname_plan`: an unknown permission name is accepted silently.
    The schema is `StructWithRest`, and `debug config` keeps the name.
  - `t40_badexp_plan`: an unknown `experimental` key is **dropped silently**.
    It exits 0 with no warning, and `debug config` shows `experimental: {}`.
  So on a build that lacked `continue_loop_on_deny`, the key would be inert,
  not fatal. The schema entry, `continue_loop_on_deny: optional(Boolean)`, is
  present in the 1.17.15 binary.
- **Result.** `continue_loop_on_deny` had its effect on every build from 1.17.15
  to 1.18.35. With the floor at the newest release, the range is the support
  window (`CONTINUE_LOOP_ON_DENY_RANGE = (min_version, verified_version)`), and
  `t40_cont_*` / `s41_cont_read` re-measure it at every bump. The permission
  block needs **no version gate**.

## Every case (1.17.15, 1.18.32, 1.18.34 and 1.18.35 agree on every row; `summary.tsv`)

`/etc/hostname` is read only where the plan names it (§4.1 and `q2_ustar_*`).
Every other case reads a scratch file. "Our block" is the §4.2 step 1 block
under `agent.<agent>.permission` in `OPENCODE_CONFIG_CONTENT`.

**§4.1's table** (`--agent plan --pure`, config in `OPENCODE_CONFIG_CONTENT`):

| Case | Config | Outcome |
|---|---|---|
| `s41_ask_read` / `s41_ask_bash` | none | auto-reject → `tool-calls`, 1 request. **The b38de8bb signature** |
| `s41_deny_read` / `s41_deny_bash` | `permission.external_directory: "deny"` | deny returned → `stop` |
| `s41_cont_read` | `experimental.continue_loop_on_deny` | auto-reject, turn continues → `stop` |
| `s41_denyallow_read` | `{"*":"deny","/etc/*":"allow"}` | read completed |
| `s41_denyallow_file_read` | `{"*":"deny","/etc/hostname":"allow"}` | deny: a file path is not a directory glob |

**§4.2 step 1** (both agents; the user config is the scratch global `opencode.json`):

| Case | Outcome |
|---|---|
| `q2_ustar_agentlayer_{plan,build}`: user top-level `{"external_directory":"ask","*":"allow"}` + our block | **deny returned → `stop`** (the pinned case for T4.1/T4.7) |
| `q2_ustar_toplevel_{plan,build}` (control: the same block top-level) | read **completed**. The user wins, as in the pre-flight |
| `builtin_tmp_ctrl_*` / `builtin_tmp_*` | completed / **deny returned**: shadowed |
| `builtin_plansdir_ctrl_plan` / `builtin_plansdir_plan` | completed / **deny returned**: shadowed |
| `builtin_plansdir_ctrl_build` / `builtin_plansdir_build` | auto-reject (`build` has no plans allow) / deny returned |
| `builtin_skilldir_ctrl_*` / `builtin_skilldir_*` (skill at `$HOME/.claude/skills/rigskill/`) | completed / **deny returned**: shadowed |
| `builtin_toolout_*` | completed. The appended `<tool-output>/*` allow survives |

**The built-in allows are shadowed by our `"*":"deny"`** on both agents and
both versions. The deny error lists the resolved rules. For `builtin_tmp_plan`
on 1.18.32, the `external_directory` rules in order are:

```
"*":ask, /rig/data/opencode/tool-output/*:allow, /rig/tmp/opencode/*:allow,
/rig/data/opencode/plans/*:allow, "*":deny, /rig/data/opencode/tool-output/*:allow
```

`builtin_tmp_build` has the same list without the plans row. `builtin_skilldir_*`
adds `/rig/home/.claude/skills/rigskill/*:allow` after the tmp row. So the globs
to re-emit after `"*":"deny"` are:

- `<os.tmpdir()>/opencode/*`, where `os.tmpdir()` is `$TMPDIR`, else `$TMP`,
  else `$TEMP`, else `/tmp`;
- `<data>/plans/*` (plan only), where `<data>` is `$XDG_DATA_HOME/opencode`,
  else `~/.local/share/opencode`;
- every discovered skill directory plus `/*`. These are environment-dependent,
  and skills are discovered under `--pure` too;
- every config-reference path plus `/*`.

`<data>/tool-output/*` needs no re-emit.

**T4.7 precedence** (user `{"<R>/shared/*":"allow","<R>/shared/private/*":"deny"}`
top-level, both agents):

| Case | Outcome |
|---|---|
| `t47_shared_*`: the rules re-listed in the agent layer after `"*":"deny"`, read `shared/f.txt` | completed |
| `t47_private_*`: same, read `shared/private/f.txt` | deny returned → `stop` |
| `t47_shared_toplevel_*` (control: not re-listed) | deny returned: the user's allow is shadowed |

**T4.7's merge over an inherited `OPENCODE_CONFIG_CONTENT`** (1.18.32 only;
`build`, no `--pure`). The content's own `agent.build.permission` carries a
permission-level `"*":"allow"` beside our three rows:

| Case | Outcome |
|---|---|
| `t47_inherited_star_build`: the user's `"*"` first, our rows after it (what `merge_permission_content` emits) | deny returned → `stop` |
| `t47_inherited_star_ctrl_build` (control: our rows first, the user's `"*"` last) | **read completed**: the later `"*"` wins by `findLast` |

So within one layer, key order decides, and the adapter re-appends its rows
at the end of the agent's `permission` map rather than updating them in place.

**A user's own agent layer** (user `agent.<a>.permission.external_directory` in
the global file, our block in the content):

- `useragent_star_*` (`{"*":"allow"}`): deny returned. Our content overwrites
  the value of the same `"*"` key in place.
- `useragent_dir_*` (`{"<R>/outside/*":"allow"}`): deny returned. The user's
  key is earlier and our `"*"` is later, so the user's agent-layer allow is
  shadowed exactly like a top-level one. T4.7 must re-list agent-layer entries
  too.
- The §4.2 step 1 note ("can still be shadowed by a user's own
  `agent.<name>.permission.external_directory`") does **not** hold for a
  global-file user. Only a layer merged *after* the content can shadow ours.

**Re-listing does not survive a file's own agent layer; a trailing key does**
(1.18.32). A key a config *file* already holds keeps the file's position in
the merge, so re-ordering the content cannot move it. opencode matches a
rule's permission as a wildcard (`Permission.evaluate`), so the adapter
repeats its `external_directory` list under a new last key,
`"external_directory*"` (`opencode.TRAILING_EXTERNAL_KEY`):

| case (user `agent.<a>.permission` in the global file) | content | outcome |
|---|---|---|
| `useragent_relist_ctrl_*` (`external_directory` `{shared/*: allow, shared/private/*: deny}`) | the re-listed list under `external_directory` only | **denied**: the file's `shared/*` stays ahead of our `"*"` |
| `useragent_relist_*` (same) | the same list, repeated under the trailing key | read completed |
| `useragent_relist_private_*` (same) | same, reading `shared/private/` | deny returned, `stop` |
| `useragent_ustar_ctrl_*` (`{"external_directory":"ask","*":"allow"}`) | our block under `external_directory` only | **read completed**: the file's later `"*"` beats our deny |
| `useragent_ustar_*` (same) | our list repeated under the trailing key | deny returned, `stop` |

**T4.8's rig case:** `write_outofcwd_build` (`build`, no `--pure`, our block):
the out-of-cwd read returns the deny to the model and finishes `stop`.

**Layers after `OPENCODE_CONFIG_CONTENT`** (hermetic shape: `--pure`, empty
global file apart from the stub provider, `OPENCODE_DISABLE_PROJECT_CONFIG=1`):

| Case | Outcome |
|---|---|
| `managed_agent_plan`: managed `opencode.json` sets `agent.plan.permission.external_directory {"*":"allow"}` | **read completed.** The managed layer shadows our deny. It overwrites the value of our `"*"` key in place |
| `managed_toplevel_plan`: managed top-level `{"*":"allow"}` | deny returned. The top-level layer is beaten by our agent layer |
| `envperm_plan`: `OPENCODE_PERMISSION={"external_directory":{"*":"allow"}}` | deny returned. It merges into top-level `permission` (layer 3) |
| `org_agent_plan`: active org account in the data-dir DB; the org config sets the same `agent.plan` allow | **read completed.** The org layer loads (the stub saw `GET /api/config` with `x-org-id` and a bearer) and shadows our deny |
| `org_agent_dbmem_plan`: same DB, `OPENCODE_DB=:memory:` | deny returned. No account, so no org fetch |

- **The managed file** is at `/etc/opencode/opencode.json[c]` on Linux. The
  path comes from `managedConfigDir()`, which is
  `process.env.OPENCODE_TEST_MANAGED_CONFIG_DIR || "/etc/opencode"`, on 1.17.15
  and 1.18.32 alike. Nothing gates it: not `--pure`, not the XDG redirect, not
  `OPENCODE_DISABLE_PROJECT_CONFIG`. The only env lever is that test hook.
- **The org config** needs three things: an active account plus
  `active_org_id` in `<data>/opencode.db` (the `account` and `account_state`
  tables), a token that is not expired (or a refresh), and network access to
  the account URL. The hermetic env passes `XDG_DATA_HOME` through, so a user
  logged in to an org gets the org layer in a hermetic slot. A fetch failure is
  swallowed (`logDebug`).
- **What disables each.** The org layer: `OPENCODE_DB` pointing at another
  database (`:memory:` measured working for `run` on both versions; the side
  effect is that no session is persisted). The managed layer:
  `OPENCODE_TEST_MANAGED_CONFIG_DIR` pointed at an empty directory, which is
  READ, not measured, and is a test-only hook.
- Both layers are administrator or organisation policy. Overriding them from
  the plugin is a design decision this rig does not make.

## Carried over from `plans/…/investigation/`

The §4.1 cases re-run the investigation's `oc/` configurations (`ask`,
`bashask`, `deny`, `bashdeny`, `cont`, `denyallow`, `denyallow2`). The
§4.2/§4.4 cases extend `preflight-2026-09-25/oc-permissions/` with the
agent-layer delivery. The stub is the same turn logic, with its spec moved into
a JSON file and an org-config endpoint added.

## The `debug config` fixture (plan §8, T8.1)

`CASE.debug_config.json` above is this rig's per-case record of the resolved
config. The §8 reader (`opencode._read_declared_providers`) is graded on a
separate, model-free capture instead:
`tests/fixtures/opencode/debug_config.<pin>.json`, produced by
`tests/fixtures/opencode/debug_config_capture.py` with the same scratch-HOME
procedure (cleared env, `OPENCODE_DISABLE_MODELS_FETCH=1`, stdin `/dev/null`,
no stub needed because `debug config` makes no model call). It carries one case
per §8.2 verdict-table row (custom with an inline key, keyless, `{env:}` unset
and set, built-in override with and without a key, nothing declared), plus a
commented `opencode.jsonc`, the project layer with and without
`OPENCODE_DISABLE_PROJECT_CONFIG`, `OPENCODE_CONFIG`, the user's
`external_directory` rules on all three layers (T8.2's second half, for T4.7),
and the hermetic `OPENCODE_CONFIG_CONTENT` with and without `--pure`, carrying
the §4.2 permission block under `agent.plan.permission` so the fixture shows the
reducer ignoring it. Every `apiKey` is a `sk-canary-not-a-key-<n>` canary.
**1.18.34 and 1.18.35 mask every declared `apiKey` as `"***"`** (an empty one and an unset
`{env:}` reference included); 1.18.32 printed them in plaintext, `""` for the
unset reference. `auth_list.<pin>.json` (`auth_list_capture.py`, same
procedure) is the `opencode auth list` envelope the auth probe is graded on.
The in-image repeat through `tests/container/wrapper.py` is not captured yet.
