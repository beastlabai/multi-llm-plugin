<!-- SHARED SHIM SNIPPET (D5/D4f) — interpolated into every rendered shim whose
     host resolves *run detached* to `--detach`, never a second file an agent is
     told to open. Edited here once; it still counts against every shim's rendered
     line budget, because an agent reads it.

     No slots. Nothing host-specific may enter this file: the cap that motivates
     the loop is the host entry's timeout ceiling, and whether the loop is used at
     all is the *run detached* verb slot. A snippet that would need
     an `if` is two snippets, or it is not a snippet. -->

## Polling a detached run

`--detach` returns immediately with the run's log path and a `status` command.
Poll with that command, one call per turn:

```bash
uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" status "<plan>" --view run --wait
```

Read the **exit code**, which is the loop condition — never the prose:

- **0** — the run finished. Read the markers and output paths from the log path
  `status` printed.
- **2** — still running (this also covers "no run recorded yet"). Poll again.
- **3** — the run needs a human. The message names the flag or command that
  answers it; do not retry the same invocation.
- **1** — the run state is unreadable. Report it; do not guess.

Three rules make this safe on a host that caps shell time:

- **One `status` call per shell command.** A `status` call is a file read plus a
  liveness probe and finishes in well under a second. `--wait` (90 s default,
  chosen to sit under the tightest documented host cap) compresses the wait into
  one turn.
- **Never spell the wait as `sleep <n>; status` in one call.** That collapses the
  wait into a single shell command and reintroduces exactly the cap kill
  `--detach` exists to avoid; the waiting belongs *between* calls, in your own
  turn loop.
- **`status` takes no lock and writes nothing**, so it is safe while the run owns
  the phase dir. To stop a run, the only kill path is
  `... multi_llm.py status "<plan>" --abort`.
