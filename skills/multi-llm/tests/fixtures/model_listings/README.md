# Model-listing CLI fixtures + parsing rules

Captured output of the three listing CLIs (`cursor-agent models`, `opencode
models`, `kilocode models`) plus synthetic edge cases, used to drive the
fixture-based parser tests for each adapter's `list_models()` (see
`utils/providers/base.py` and the per-adapter parsers).

The captures were taken with the subprocess environment neutralized
(`TERM=dumb NO_COLOR=1 PAGER=cat`, `stdin=/dev/null`) — exactly how
`base._run_models_command()` invokes them at runtime.

## Common id-format contract (all adapters)

`ModelListing.models` / `recommended` hold **bare model ids** — the exact string
each adapter's `build_command()` passes to `--model`/`-m`. The init flow prefixes
the provider name to form a `provider:model` spec, and `parse_model_spec` splits on
the **first** colon. Therefore:

- A bare id **may** contain `/` (opencode/kilocode namespaced ids are fine).
- A bare id **must not** contain `:` — every parser **skips** any listed id with a
  `:` so the `provider:model` round-trip stays unambiguous.

Parsers also: detect a JSON shape (leading `[`/`{` → list of strings, or list of
`{"id"/"name": ...}` objects), strip ANSI escapes, skip blank/header/footer lines,
and on anything unexpected return an empty list so the adapter falls back to
curated with a `note`.

## Per-provider rules

### cursor-agent (`cursor_agent_success.txt`, `cursor_agent_ansi.txt`)
Plain text. Layout: a `Available models` header, blank lines, then one model per
line as `"<id> - <Description>"`, and a trailing `Tip: use --model <id> ...` line.
Rule: for each line, match `^(?P<id>\S+)\s+-\s+\S` and take group 1 as the bare id;
skip the header (`Available models` has no ` - `), blanks, the `Tip:` footer (no
` - ` after `Tip:` and it contains a `:`), and any id containing `:`.

### opencode (`opencode_success.txt`)
Plain text, one `provider/model` per line, no header/blank/footer lines.
Rule: each non-empty stripped line is a bare id; skip ids containing whitespace or
`:`.

### kilocode (`kilocode_success.txt`)
Plain text, one id per line (664 in the wild; the fixture is a representative
subset). Ids look like `kilo/anthropic/claude-opus-4.8`, `kilo/~google/...`, or
`openrouter/moonshotai/kimi-k2`. ~33 of the real lines carry a `:free` /
`:discounted` suffix (`kilo/.../foo:free`) — these are **skipped** per the id-format
contract. No header/blank/footer lines.
Rule: each non-empty stripped line is a bare id; skip ids containing whitespace or
`:` (which drops the `:free`/`:discounted` variants).

### codex / gemini / claude-code
No listing command (`can_list_models = False`); they use the default
`list_models()` → curated only. No fixtures needed.

## Edge-case fixtures (synthetic)

- `empty.txt` — empty output (CLI ran but printed nothing) → `[]` → curated fallback.
- `auth_error.txt` — an auth/error message on stdout → no parseable ids → curated.
- `cursor_agent_ansi.txt` — ANSI-colored cursor output → ANSI stripped, ids parsed.
- `json_array_of_strings.json` — JSON list-of-strings shape.
- `json_array_of_objects.json` — JSON list-of-`{id,name}` objects shape.
