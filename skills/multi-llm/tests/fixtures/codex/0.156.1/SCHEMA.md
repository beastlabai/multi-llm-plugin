# codex-cli 0.156.1 `codex exec --json` item schema (plan §6.2)

Captured 2026-09-24 against **codex-cli 0.156.1**, offline: the real binary was
pointed at a localhost Responses-API stub (`capture/stub.py`) that scripts one
tool call per turn, with a throwaway `CODEX_HOME` and a throwaway git repo as
cwd, `--sandbox danger-full-access` (bubblewrap cannot start on the capture
machine). Zero API quota. `capture/run.sh NAME capture/script_X.json --model
gpt-5.5 --sandbox danger-full-access` reproduces a stream (`--model gpt-5.5`
makes codex advertise the `apply_patch` custom tool; the fallback model id does
not). Absolute paths of the throwaway tree are rewritten to `/work/tree`, and the
one PTY prompt that printed `user@host` is scrubbed; nothing else was edited.

The `web_search_*` and `code_mode_writes*` fixtures are the exception. They come
from a **live** capture made on 2026-09-25 (see "`web_search` and its
`action.type` / `results[].type` values" below), and their stub replays are
marked `_replay`.

| Fixture | What it drives |
|---|---|
| `read_only_clean.jsonl` | `sed -n`, `cat`, `rg … \| head`, `git log`, a non-zero `ls … 2>/dev/null`, `nl … \| sed -n` — the clean positive control |
| `quoting_render.jsonl` | how codex renders `$`, `$(…)`, backticks, `login:false` (`-c`) and `shell:/bin/sh` into `command` |
| `writes_patches_sessions.jsonl` | reads, a non-zero exit, `echo > f`, a heredoc write, a `python3 -c` write, `apply_patch` as the freeform tool, `apply_patch <<EOF` through the shell, and two exec sessions left open at turn end |
| `exec_session.jsonl` | an exec session that completes (`sleep; echo > s.txt`), and a PTY `bash` driven via `write_stdin` (`echo tty > t.txt`) |
| `mcp_tool_call.jsonl` | a stdio MCP server's `write_file` tool (`capture/mcp.py`) that really wrote `m.txt` |
| `collab_spawn_agent.jsonl` | `spawn_agent`; the sub-agent then ran `echo sub > sub.txt` |
| `run_0c477c4d_partial.jsonl` | the items the persisted `log_codex_gpt-6-astra_xhigh.txt` of run `0c477c4d` still holds (live `gpt-6-astra`, 0.156.1) |
| `web_search_hosted.jsonl` | **live** (2026-09-25, `gpt-5.5`): the hosted `web_search` tool doing `search`, `open_page`, `find_in_page` |
| `web_search_hosted.sse.json` | the raw Responses SSE `web_search_call` items behind it (`added` + `done`), recorded by `capture/proxy.py` |
| `web_search_hosted_replay.jsonl` | the same items replayed offline: `capture/script_w.json` through `stub.py` |
| `web_search_probe_replay.jsonl` | offline probe (`capture/script_w_probe.json`): an unknown server action, `results` on a hosted item, a failed call, an action with no query |
| `web_search_code_mode.jsonl` | **live** (2026-09-25, `gpt-6-astra`, code mode): `tools.web__run` doing `search`, `open_page`, `find_in_page`, with `text_result` results |
| `web_search_code_mode.rollout.jsonl` | the matching `~/.codex/sessions` rollout lines: the `exec` JS cell and its three `Extension`/`web.search` items |
| `code_mode_writes.jsonl` (+ `.rollout.jsonl`) | **live** (2026-09-25, `gpt-6-astra`, `danger-full-access` in an empty scratch dir): one `exec` cell calling `apply_patch`, `exec_command("echo hi > b.txt")`, `exec_command("cat a.txt b.txt")` |
| `code_mode_mcp.jsonl` (+ `.rollout.jsonl`) | **live** (2026-09-25, `gpt-6-astra`, `danger-full-access` in an empty scratch dir): one `exec` cell calling the `capfs` MCP tool `write_file` (`capture/mcp.py`), which really wrote `m.txt` |

## Top-level events

`thread.started`, `turn.started`, `item.started`, `item.completed`,
`turn.completed` (carries `usage`). The binary's tag table also names
`item.updated`, `turn.failed`, `error`. Unchanged from 0.146.0.

## `command_execution`

```json
{"id":"item_0","type":"command_execution","command":"/bin/bash -lc \"sed -n '1,2p' AGENTS.md\"",
 "aggregated_output":"line one\nline two\n","exit_code":0,"status":"completed"}
```

- Fields: `id`, `type`, `command` (**string**), `aggregated_output` (string),
  `exit_code` (int; `null` on `item.started`), `status` (`in_progress` on
  started; `completed` for exit 0, **`failed`** for a non-zero exit — `ls
  does-not-exist` is `exit_code: 2, status: "failed"`).
- Emitted as an `item.started` / `item.completed` pair with the same id.
- `command` is **one string, not an argv array**, and always carries **one
  shell wrapper**: `<shell> -lc <quoted cmd>`. `<shell>` is `/bin/bash` by
  default, `/usr/bin/sh` when the model passes `shell: /bin/sh`; the flag is
  `-lc`, or `-c` when the model passes `login: false`. The quoting is POSIX
  (single quotes, or double quotes with `\"` escapes, or a concatenation such as
  `'line"'$'"'`), and `shlex.split(command)` recovers exactly `[shell, flag,
  cmd]` in every captured case (`quoting_render.jsonl`).
- The inner string is exactly what the shell runs: `$(…)`, backticks and
  heredocs reach it unexpanded (`cat $(echo AGENTS.md)` is rendered with the
  `$(` intact).
- There is **no** `workdir`, `tty` or `shell` field on the item.
- codex exports `PAGER=cat`, `GIT_PAGER=cat` and `GH_PAGER=cat` to every exec
  (measured in-session), so `git log`/`show`/`diff` never start a pager even
  under a PTY.

## Exec sessions (unified exec: `exec_command` + `write_stdin`)

There is **no separate item type**. An `exec_command` that is still running
when its yield time expires is a `command_execution` whose `item.started` has
no `item.completed` yet. `write_stdin` produces **no item at all**:

- If the session finishes before the turn ends, its `item.completed` arrives
  (after later items) with the **original** `command` only
  (`/bin/bash -lc bash`); text typed through `write_stdin` appears at most in
  `aggregated_output` (PTY echo), never in `command`. `exec_session.jsonl`'s
  PTY `bash` wrote `t.txt` this way.
- If the turn ends first, the `item.started` is never completed
  (`writes_patches_sessions.jsonl` items 8 and 9).

So the items do **not** carry every command a session ran. The audit's answer:
an unmatched `item.started` fails (unfinished action), and a completed
session's `command` is classified like any other — an interactive program
(`bash`, `python`, …) is not on the allow-list.

## `file_change`

```json
{"id":"item_6","type":"file_change","changes":[{"path":"/work/tree/AGENTS.md","kind":"update"},
 {"path":"/work/tree/a.txt","kind":"add"}],"status":"completed"}
```

- Fields: `id`, `type`, `changes` (list of `{path: absolute string, kind:
  add|update|delete}`), `status`. Started/completed pair.
- Produced by the freeform `apply_patch` custom tool **and** by `apply_patch
  <<'EOF' … EOF` run through `exec_command` — codex intercepts the latter and
  emits a `file_change` instead of a `command_execution`.
- A shell redirect, a heredoc (`cat > h.txt <<'EOF'`), or `python3 -c
  "open(...).write(...)"` is **not** a `file_change`: it is a plain
  `command_execution` whose `command` carries the write.

## `mcp_tool_call`

```json
{"id":"item_1","type":"mcp_tool_call","server":"capfs","tool":"write_file",
 "arguments":{"path":"m.txt","text":"x"},
 "result":{"content":[{"type":"text","text":"wrote m.txt"}],"structured_content":null},
 "error":null,"status":"completed"}
```

- Fields: `id`, `type`, `server`, `tool`, `arguments` (object), `result`
  (`null` on started), `error`, `status`. Started/completed pair.
- The model calls it as function `write_file` in namespace `mcp__capfs`; with
  `--model gpt-5.5` MCP tools are deferred behind `tool_search`.

## `collab_tool_call` (new since the 0.146.0 capture)

```json
{"id":"item_1","type":"collab_tool_call","tool":"spawn_agent","sender_thread_id":"…",
 "receiver_thread_ids":["…"],"prompt":"SUBAGENT-TASK write a file",
 "agents_states":{"…":{"status":"pending_init","message":null}},"status":"completed"}
```

- `tool` is `spawn_agent` / `send_input` / `wait_agent` / `close_agent` /
  `resume_agent` (the `multi_agent_v1` namespace).
- **The sub-agent's own items never reach the parent's stream**: the spawned
  agent ran `echo sub > sub.txt` (the file exists) and no `command_execution`
  for it appears. A slot that spawned an agent could have written anything.

## `agent_message`, `reasoning`, `error`

Unchanged: `agent_message` / `reasoning` carry `text`; the nested `error` item
(`"Model metadata for `stub-model` not found…"`) is non-fatal and appears on
every run with a model id codex has no metadata for. None of them is a tool
action.

## `web_search` and its `action.type` / `results[].type` values

The §1 evidence table called `find_in_page`, `text_result` and `other` "new item
types". **They are not item types.** They are field values inside a
`web_search` item: `action.type` is one of `search`, `open_page`,
`find_in_page`, `other`, and `results[].type` is `text_result`.

### Live capture (2026-09-25)

Two live `codex exec --json --skip-git-repo-check` runs from an empty scratch
dir, using codex's own ChatGPT login (subscription, no per-token billing), with
the prompt "search the web for the current Python release, then open the
python.org downloads page, and find the word 'Release' on that page. Do not run
any shell commands. Answer in one line":

- `gpt-5.5`, `--sandbox read-only`, routed through `capture/proxy.py` (a logging
  passthrough to `chatgpt.com/backend-api/codex` configured as a custom
  `model_providers.cap` with `requires_openai_auth=true`). The model used the
  **hosted** `web_search` Responses tool. → `web_search_hosted.jsonl` +
  `web_search_hosted.sse.json`.
- `gpt-6-astra`, `--sandbox read-only`, on the built-in provider. This model
  runs in **code mode**: the one model-facing tool is a JavaScript `exec` cell,
  which called `tools.web__run({search_query|open|find: …})` three times. The
  custom provider does not expose `web__run`, so this run has no proxy log.
  The rollout file holds the cell and the three items. → `web_search_code_mode.jsonl`
  + `web_search_code_mode.rollout.jsonl`. Run `0c477c4d` took this route: its
  `exec-…` call id matches.

Edits: the final `message` text in the stub script is `[]`. `results` lists are
cut to their first 2 entries, and `thumbnail_url` is shortened. In
`code_mode_writes*`, the scratch path is rewritten to `/work/tree`. The duplicate
`id` keys are kept byte for byte. Nothing else was edited.

```json
{"type":"item.started","item":{"id":"item_0","type":"web_search","id":"ws_0077…","query":"","action":{"type":"other"}}}
{"type":"item.completed","item":{"id":"item_0","type":"web_search","id":"ws_0077…","query":"current Python release python.org downloads",
 "action":{"type":"search","query":"current Python release python.org downloads","queries":["current Python release python.org downloads"]}}}
{"type":"item.completed","item":{"id":"item_2","type":"web_search","id":"exec-0de7…","query":"current Python release site:python.org",
 "action":{"type":"search","query":"current Python release site:python.org"},
 "results":[{"type":"text_result","domain":"www.python.org","ref_id":"turn0search0","snippet":"* Latest Python 3 Release - Python 3.14.7",
  "title":"Python Source Releases | Python.org","url":"https://www.python.org/getit/source/"}]}}
```

- `web_search` fields: `id` (**emitted twice**: `item_N`, then the call id,
  `ws_…` on the hosted route and `exec-…` on the code-mode route. A JSON parser
  keeps the last one, which is the same on the started and the completed line),
  `type`, `query` (a string, `""` on started), `action`, and `results` (only on
  the code-mode route). There is no `status` field. A hosted call the server
  marks `failed` still arrives as a plain `item.completed`
  (`web_search_probe_replay.jsonl` `ws_probe_3`).
- Every `item.started` carries `action: {"type": "other"}` and `query: ""`.

### Each value: what produces it, and can it touch the working tree?

| Value | Produced by | Touches the tree? |
|---|---|---|
| `action.type: search` (`query`, optional `queries`) | the server's hosted `web_search` tool (`web_search_call` output item, `action.type: search`), or the code-mode `web__run({search_query})` call | **No.** A network search. `query` is text, not a path |
| `action.type: open_page` (`url`) | the hosted tool fetching a page, or `web__run({open})` | **No.** A network fetch of a public URL |
| `action.type: find_in_page` (`url`, `pattern`) | the hosted tool searching a page it already fetched, or `web__run({find})` | **No.** A text search inside the fetched page |
| `action.type: other` | **codex itself**, in two places: (1) on every `item.started`, as a placeholder while the call runs (the SSE `output_item.added` has no action yet), and (2) on `item.completed`, as codex's serde catch-all (`WebSearchAction::Other`) for a server action it cannot name. The stub probe sent `{"type":"not_a_known_action","url":…}`, and codex emitted `{"type":"other"}` with the fields dropped | **No.** It is still a `web_search` item, and neither producer runs anything locally |
| `results[].type: text_result` (`domain`, `ref_id`, `snippet`, `title`, `url`, optional `thumbnail_url`) | only the code-mode `web__run` route: page text the network search returned, which codex relays to the model | **No.** Data relayed to the model. codex never acts on it |

Supporting facts:

- The binary's `WebSearchAction` enum has only `search {query, queries}`,
  `open_page {url}` and `find_in_page {url, pattern}`, plus the `other`
  fallback. The string `text_result` does not appear in the binary at all:
  codex passes the code-mode results through as opaque JSON.
- The hosted route never carries `results`. When the stub attached `results`
  to a `web_search_call`, codex dropped them (`ws_probe_2`).
- Nothing in any of these fields names a local path. No local process, file
  write or MCP call is involved: the hosted call runs on the OpenAI server, and
  `web__run` is a network tool.
- The audit therefore classifies every `web_search` whose `action.type` is one
  of the four values above, and whose `results` is absent or a list of
  `text_result`, as inert. Any other action or result type still fails closed.

## Code mode (`gpt-6-astra`): the `exec` cell emits no item

On `gpt-6-astra`, 0.156.1 advertises a code-mode tool set: a freeform `exec`
tool whose input is JavaScript run in a V8 isolate ("no Node, no file system,
no network access"). All other tools are reachable only as `await tools.X(…)`
from inside that cell: `exec_command`, `write_stdin`, `apply_patch`, `web__run`,
`view_image`, `image_gen__imagegen`, the MCP resource tools, goals, and so on.
The `exec` `custom_tool_call` itself produces **no** `--json` item. Every nested
call captured so far does:

- `code_mode_writes.jsonl`: one cell called `tools.apply_patch(…Add File a.txt…)`,
  then `tools.exec_command({cmd:"echo hi > b.txt"})`, then `cat a.txt b.txt`. The
  stream has exactly a `file_change`, then two `command_execution` items (same
  `/bin/bash -lc` wrapper as the direct route). Both files were really written.
- `web_search_code_mode.jsonl`: `tools.web__run` → `web_search` items.
- `code_mode_mcp.jsonl`: the cell looked the tool up in `ALL_TOOLS` and called
  `tools[name]({path:"m.txt",text:"x"})`. → one `mcp_tool_call` item
  (`server: capfs`, `tool: write_file`), the same shape as the direct route.

Not captured: a nested `image_gen__imagegen` call from inside a cell. `write_stdin` emits no item on either route (see "Exec
sessions").

## `run_0c477c4d_partial.jsonl`

Lines 117-125 of the persisted log, which is cut at 46,080 B
(`[stream truncated at 46080 B]`) and keeps only the last lines of the stream.
Kept: item_27 (completed only — its start was before the cut), items 28-29
(started + completed), item_30 (the `web_search` above), item_31 (started only).
item_31's completed line is itself cut mid-`aggregated_output` in the log and
is omitted; there is no `turn.completed`. Every `aggregated_output` of a
`command_execution` is replaced by `[redacted for the fixture: N chars of
private-repository file content]`. Commands keep their exact structure —
verbs, flags, quoting, `;` chains, pipes, sed ranges and rg pattern shapes —
but every private path, file name and identifier is swapped for a neutral
stand-in (`src/db/schema/10-core.sql`, `src/services/records/…`); ids,
statuses and exit codes are verbatim. The only `AGENTS.md` read the surviving tail holds is item_28
(`sed -n '360,440p' src/db/AGENTS.md`); any earlier read of
`src/AGENTS.md` itself fell off the cut.

## Item types with no capture

`todo_list` (the plan tool is not advertised by 0.156.1 on these models) and
`image_generation` were not produced. The audit treats them, and every other
unlisted type, as "could have written".
