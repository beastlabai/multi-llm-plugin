# Providers

Which code harnesses and models multi-llm calls, how a model is named, and how to
point a provider CLI at a different endpoint. Back to the [README](../README.md).

---

## The provider table

**Edit [`skills/multi-llm/providers.yaml`](../skills/multi-llm/providers.yaml) before using the plugin** - it defines which code harnesses and models multi-llm calls by default. Each provider maps to a CLI binary that must be installed and authenticated separately:

| Provider key | CLI binary | Install / docs |
| --- | --- | --- |
| `claude-code` | `claude` | [Claude Code](https://code.claude.com/docs/en/) |
| `cursor-agent` | `cursor-agent` | [Cursor Agent CLI](https://cursor.com/docs/cli/overview) |
| `gemini` | `gemini` | [Gemini CLI](https://geminicli.com/docs/get-started/) |
| `grok` | `grok` | [Grok Build](https://docs.x.ai/build) |
| `codex` | `codex` | [Codex CLI](https://developers.openai.com/codex/cli) |
| `opencode` | `opencode` | [OpenCode CLI](https://opencode.ai/docs/cli/) |
| `kilocode` | `kilocode` | [Kilo Code CLI](https://kilo.ai/docs/cli) |
| `cline` | `cline` | [Cline](https://docs.cline.bot/) |
| `goose` | `goose` | [goose](https://block.github.io/goose/) |
| `aider` | `aider` | [Aider](https://aider.chat/) |
| `agy` | `agy` | [Antigravity CLI](https://antigravity.google/docs/cli/using) |
| `pi` | `pi` | [Pi](https://pi.dev/docs/latest/usage) |
| `muse` | `muse` | [Muse Code](https://dev.meta.ai/docs/muse-code/extending) |
| `kimi` | `kimi` | [Kimi Code](https://code.kimi.com/docs) - argv-only: a large prompt exceeds the Windows command-line cap and is refused before spawn, so this provider serves every fan-out mode on POSIX only. Also note: at 0.36.1 the CLI contacts `telemetry-logs.kimi.com` on every run even though multi-llm sets `KIMI_DISABLE_TELEMETRY=1` (measured; the opt-out does not gate the connection attempt) |
| `qwen-code` | `qwen` | [Qwen Code](https://qwenlm.github.io/qwen-code-docs/) - the provider key names the harness, the binary is `qwen`. Supports a stdin prompt transport (so a multi-KB prompt is never measured against the Windows command-line cap) and `--json-schema` structured output. Note: at 0.21.13 the CLI **exits 0 on every API error** (401/404/400/500 alike) with the diagnosis on stdout as `[API Error: ...]` and stderr empty - multi-llm detects that marker, so a failed run is reported as a failure rather than as the model's answer. It also runs with `--max-tool-calls 0`, so a tool call is refused before it executes rather than touching your checkout - measured at ~29% of live runs, which the retry ladder normally absorbs at the cost of a second call |
| `copilot` | `copilot` | [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli) - driven **BYOK**, so it needs no GitHub account, no PAT and no Copilot seat: pointing it at an OpenRouter key is the whole setup, and multi-llm strips any `GH_TOKEN`/`GITHUB_TOKEN` you already export so a stale one cannot re-enable the GitHub side underneath. Supports a stdin prompt transport (so a multi-KB prompt is never measured against the Windows command-line cap) and the `:effort` suffix. Two measured notes: the shipped model list is **non-reasoning slugs only**, because a stream carrying `reasoning_content` breaks the CLI's own completions parser; and an empty provider 200 makes the CLI **exit 0 with an empty answer and no error record**, which multi-llm detects structurally rather than by a marker |
| `vibe` | `vibe` | [Mistral Vibe](https://docs.mistral.ai/vibe/) - installed from PyPI as `mistral-vibe`, and it needs Python 3.12+. It has **no `--model` flag**: the entries above are aliases defined in `$VIBE_HOME/config.toml`, and multi-llm selects one through `VIBE_ACTIVE_MODEL`, so pointing vibe at a different model means editing that file as well as this list. Supports a stdin prompt transport (so a multi-KB prompt is never measured against the Windows command-line cap) and the `:effort` suffix, which rides the config's `thinking` key rather than a flag. One measured note worth knowing: vibe's own default agent is `accept-edits`, which auto-approves file writes, so multi-llm always passes `--agent plan` (read-only) and pins `VIBE_DEFAULT_AGENT` to match |

A model is referenced as `provider:model` (e.g. `codex:gpt-5.5`). A bare model name uses `default_provider` from `providers.yaml`.

> **Heads up - defaults:** the shipped `defaults.models` list references a specific set of models across `cursor-agent`, `kilocode`, and `claude-code`. If you don't have those CLIs installed and authenticated, either pass `--models` explicitly, run with `--interactive` to choose, or edit `defaults.models` in `providers.yaml` to match the providers you have. The safest zero-setup choice is `claude-code` models, which every Claude Code user already has.

Model selection priority (highest first): `--models`, then `--interactive`, then `--quick`, then `defaults.models` (from YAML), then interactive fallback.

## Pointing a provider CLI at a different endpoint (GLM / Z.ai, and the general case)

**No config layer sets environment for a provider** - `providers.yaml`, the per-project override and `MULTI_LLM_PROVIDERS_CONFIG` carry `default_timeout`, `max_concurrent` and `models` and nothing else, and a key the base does not declare merges and is then ignored, so there is no `env:` block to write. Endpoint and credential wiring is **shell configuration this plugin neither owns nor validates**: export the variables in your shell, your `.envrc`, or per invocation (`env VAR=... multi-llm ...`). They reach the CLI because each spawn hands the child the inherited environment plus the adapter's own delta, and no adapter touches the `ANTHROPIC_*` namespace below.

GLM (Z.ai) is a worked example, because GLM models are already first-class here - `z-ai/glm-5.2` is the default model for five providers. There are three recipes, at **two different evidence levels**, and the difference is stated rather than left to be assumed.

**R1 - GLM over the OpenRouter lane (recommended, live-verified).** Runs `claude-code` against a GLM slug on an OpenRouter key:

```bash
export ANTHROPIC_BASE_URL=https://openrouter.ai/api   # the bare /api origin - Claude Code appends /v1/messages itself
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=                             # present AND empty - this is what keeps the run off your own Anthropic account
# then select a GLM slug, e.g.:  --models claude-code:z-ai/glm-5.2
```

Two caveats on this path, both **measured** rather than theoretical:

- **`total_cost_usd` is wrong on any non-first-party base URL.** Claude Code prices the tokens as if they were first-party Anthropic (`"provider": "firstParty"`), so a run reporting `total_cost_usd: 0.2061` really spent `$0.0204`. Only the key's own usage delta is a spend figure on this path.
- **A GLM slug is not an Anthropic model,** and OpenRouter's own cookbook warns that Claude Code "may not work correctly with other providers". A tool-calling failure on this recipe may be the model rather than the wiring.

**R2 - the direct Z.ai Anthropic endpoint (documented, not live-verified (vendor docs)).** Needs a Z.ai Coding Plan or pay-as-you-go key:

```bash
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
export ANTHROPIC_AUTH_TOKEN=<your Z.ai key>
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2-air
```

**R3 - the Z.ai OpenAI-compatible coding endpoint (documented, not live-verified (vendor docs)).** For `opencode` / `goose` / `cline` / `kilocode`, point each CLI's own base-URL and key variables (see that CLI's documentation - this plugin does not set them) at `https://api.z.ai/api/coding/paas/v4`. Note that the **OpenRouter** half of R3 needs no endpoint recipe at all: the shipped `kilocode`, `cline`, `goose`, `aider` and `pi` blocks already list `openrouter/z-ai/glm-5.2` entries, so "GLM through those CLIs" is a model-selection line, not an endpoint recipe.

Two rules span all three recipes:

- **Pin models with `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, never `ANTHROPIC_MODEL` / `ANTHROPIC_SMALL_FAST_MODEL`.** The latter two are stripped from every detached run, so a recipe built on them works interactively and then silently falls back to the vendor default under `--detach`.
- **Z.ai restricts *Coding Plan subscription* keys to its officially supported tool list.** Pay-as-you-go keys, or the existing OpenRouter lanes, are the safe default - a second reason R1 is the recommended path, on top of being the one that is actually verified here.
