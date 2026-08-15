# `home_seed/<id>/` — minimum synthesized config, value-free and tracked

Every container row starts each identity at its **first-run posture** in a fresh
tmpfs `$HOME`, which is the point: it is the state a real user's machine is never
in, and therefore the one nothing else tests. These templates are the measured
minimum that lets each CLI start non-interactively, pass the executor's own
credential probe, and run without stopping to ask for approval.

**They contain no values.** Every credential field is an empty string or a
`{{placeholder}}` the entrypoint or the PATH wrapper substitutes at row start —
with the *fake* per-exec stub session key on the provider lane, and on the host
lane with whatever `creds.yaml` resolved. That is why `.gitignore` anchors its
credential rules by path (`**/.codex/auth.json`) and then negates this whole
directory back in: a bare `*auth.json` would silently untrack these files —
`git add` appears to succeed, `git ls-files` never lists them, the committed-secret
scanner never walks them, and the build context ships without them — surfacing
three phases later as an opencode row that dies on `not_authenticated`.

## Measured, not assumed (Phase 1)

| id | file | why it is needed |
| --- | --- | --- |
| `codex` | `.codex/config.toml` | selects the stub provider; **`$CODEX_HOME` must also point here and the directory must exist** — codex exits 1 with "CODEX_HOME points to … but that path does not exist", and a *relative* `$HOME` makes it resolve `~/.codex` against nothing and fall straight through to `api.openai.com` |
| `codex` | `.codex/auth.json` | `codex login status` prints "Not logged in" and exits **1** without it, and `classify_auth_probe` reads that as `False` — so `install-host codex` fails row 5 and `preflight_runner` refuses the batch before the first stub call |
| `opencode` | `.local/share/opencode/auth.json` | where `opencode auth list` reads credentials from |
| `claude-code` | `.claude.json` | onboarding + a trust entry for `/work/sample-project`; the real `~/.claude.json` is deny-listed (87 KB of per-project history and MCP definitions that routinely embed third-party keys), so this is synthesized instead of mounted |

**One measured surprise worth knowing before you trust an opencode row:** a
logged-out `opencode auth list` prints "0 credentials" and exits **0**, which
matches none of the adapter's `auth_error_patterns` — so `classify_auth_probe`
returns `True` and the executor preflight admits the batch. Unlike codex, opencode
does *not* fail closed on a missing credential; its `install-host` row 5 reports
`auth=ok` on a completely fresh `$HOME`. A negative auth row on opencode therefore
cannot expect a `not_authenticated` refusal.
