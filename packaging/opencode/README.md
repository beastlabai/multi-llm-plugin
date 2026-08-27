# opencode-multi-llm

Run multi-LLM plan reviews, task generation, implementation and code reviews from
inside OpenCode — several LLM CLIs in parallel, driven by one command.

## Install

Add the plugin to your OpenCode config (`~/.config/opencode/opencode.json`) and
restart:

```json
{ "plugin": ["opencode-multi-llm@0.1.0"] }
```

Then `/multi-llm plans/my-feature.md` in a session.

**Pin the version.** An unpinned entry resolves **once and never again**: the
cache key is the literal spec string, and the installer early-returns whenever
the package directory already exists. `"opencode-multi-llm"` resolves to
`@latest` on the first start and then freezes there forever. A semver range
freezes identically. Upgrading means **bumping the pin and restarting** — that is
the whole update story, and there is no `opencode plugin update`.

Requires **OpenCode 1.18.18 or newer** (declared in `engines`, and enforced: an
older host refuses the plugin with a clear compatibility error) and
[uv](https://docs.astral.sh/uv/) on the machine.

## First run

**The first `/multi-llm` after an install or an upgrade needs network, or a warm
`~/.cache/uv`.** The tool runs the packaged Python through `uv run --frozen
--no-dev`, which installs the single runtime dependency (`pyyaml`) into a venv
inside the installed package on first use. After that the route works offline.
This is the difference between "the first `/multi-llm` is slow, or fails behind a
proxy" and "the plugin is broken".

Each installed version pays for its own venv, and the old one is never reclaimed
— see *Uninstalling* below.

## Troubleshooting

### Nothing happened, and there is no error

**Every failure on this route exits 0.** The plugin stays listed in `plugin[]`,
the run reports success, and `--print-logs --log-level DEBUG` writes a **0-byte**
log. So do not diagnose with a presence check or an exit code — neither proves
anything. Check for a **mutated key** instead:

```bash
opencode debug config | jq '.command["multi-llm"] != null'
```

`true` means the plugin loaded and its `config` hook ran. `false` means it did
not, whatever `plugin[]` says.

### Clearing the cache — do it **online**

There is no `opencode plugin update` and no cache-clear command. The only support
hatch is deleting the package's cache directory and restarting:

```bash
rm -rf ~/.cache/opencode/packages/opencode-multi-llm@0.1.0
```

Spell the path from the **normalized** spec, which is not always the string in
your config: `"opencode-multi-llm"` caches at
`packages/opencode-multi-llm@latest`, and a scoped name nests as
`packages/@scope/name@version/node_modules/@scope/name/`.

Online this takes about a second and self-heals, and it is also the only thing
that ever moves an unpinned entry to a newer version. **Offline it turns a
working plugin into a missing one** — roughly 70 s burned per fetch chain, and the
plugin comes back only if `~/.npm/_cacache` still holds the packument and
tarball. Otherwise it is silently dropped at exit 0.

One thing that hatch is *not* for: the spec directory's own `package.json` is an
**output** of the last resolve and is rewritten by the next one, so hand-editing
it changes nothing. The authoritative installed version is
`node_modules/opencode-multi-llm/package.json`.

### `ModuleNotFoundError: No module named 'yaml'`

You set `MULTI_LLM_UV_RUN_ARGS` to something containing `--no-sync` against a
tree whose venv was never built. See the next section — that is not a bug in the
plugin, and uv reports no error of its own.

## `MULTI_LLM_UV_RUN_ARGS`

Overrides the flags passed to `uv run`. The default is `--frozen --no-dev`.

**It replaces the list; it never appends.** That is what makes `--no-sync`
expressible at all — appended, it would sit beside the very sync you are trying
to suppress. Unset, empty or whitespace-only means the default. It is split on
whitespace with no quote processing and no shell.

Because it replaces, **it can legitimately drop `--no-dev`**, and that is the one
documented way back into a first run that installs 17 packages instead of one,
and into an offline posture that fails on a test-only dependency.

Two failure signatures, which look nothing alike:

- `--no-sync` **against a tree with no venv**: uv creates an empty environment,
  runs against it anyway, and the dispatcher dies with `ModuleNotFoundError: No
  module named 'yaml'` at exit 1. No uv-level error at all, and the empty venv
  persists so every later run repeats it. `--no-sync` is correct only against a
  venv something else already built — a container image, a pre-warmed lane —
  and never on a first run.
- **The offline default**: fails *inside* uv at the download step, with
  `Network connectivity is disabled, but the requested data wasn't found in the
  cache for: pyyaml==6.0.3`.

## Uninstalling

Delete the `plugin` entry from `opencode.json`. That is all this package ever
writes — no shim, no file under `~/.config/opencode/commands/`, no provenance
header anywhere.

**But deleting the line is not deleting the install.** The cache directory
`~/.cache/opencode/packages/<spec>` stays on disk, and nothing ever reclaims it.
A later reinstall of that same spec hits the installer's early return and
silently runs the **old** payload — so a reinstall meant to pick up a newer
version needs the `rm -rf` above first, online. Deleting the cache directory is
optional but is not automatic.

## Caveats

**Windows is untested on this route.** The uv probe carries the two `.exe`
locations the official Windows installer creates, and paths are composed with
Node's `path` module, so the spelling is right — but nothing downstream of that
probe has been measured on Windows: argv spawning, `~` expansion, the
backslash-separated skill directory travelling into `--project` and every Python
path argument, the detached worker's process-group signalling, and where OpenCode
roots its cache. macOS is in the same position. This release is validated on
Linux.

**Stream parsing is best-effort above opencode 1.18.12.** The shipped parser
fixtures were captured from that release, and this package's declared floor
(1.18.18) sits above it, so this holds for every install on this route. Output
that does not match the captured shapes degrades to a best-effort read rather
than failing.

## License

MIT — see `LICENSE`.
