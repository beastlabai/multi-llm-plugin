/**
 * opencode-multi-llm — the OpenCode plugin entry point.
 *
 * WHAT THIS FILE IS. The whole plugin. It injects the `/multi-llm` slash command
 * and registers a model-callable `multi_llm` tool whose `execute` runs the
 * skill's Python dispatcher, supervises a detached run to a terminal result, and
 * hands back a structured answer.
 *
 * WHY IT EXISTS. The file route ships a markdown shim, and a markdown shim is
 * prose a model may improvise past — the host lane recorded one completing a
 * workflow "while IMPROVISING the invocation … dropping every baked flag and
 * omitting --detach" on a host whose capability entry declares `detach:
 * required`. Code is not prose. Every flag that was an instruction is now
 * something this file appends, and every rule that was a sentence is a check
 * that runs before the spawn.
 *
 * FOUR RULES GOVERN EVERY LINE HERE, and each one is load-bearing:
 *
 *   1. **No bare specifiers.** A runtime `import` that cannot resolve next to
 *      this file kills the plugin *completely silently* — exit 0, still listed in
 *      `plugin[]`, config unmutated, and a 0-byte debug log. Only `node:`
 *      builtins. This, not the `dependencies` field, is what kills a plugin
 *      offline.
 *   2. **No authored prose.** Every mode word, alias, summary, flag name, exit
 *      code and message this file shows a model is read from
 *      `dist/cli-surface.json` or from the packaged skill tree. The single owner
 *      of each stays in Python. The only text authored here is the connective
 *      scaffold around generated values.
 *   3. **Every export must be a plugin factory.** MEASURED against opencode
 *      1.18.18 and 1.18.21: the loader iterates *every* named export of this
 *      module and calls each one as a plugin factory. A single export that is
 *      not a function makes it refuse the **whole module** —
 *      `failed to load plugin … error="Plugin export is not a function"` — and
 *      that is another silent exit-0 death: still listed in `plugin[]`, config
 *      unmutated. Two exports of the *same* factory get it invoked twice. So
 *      this file exports exactly one thing, `MultiLlmPlugin`, and hangs
 *      `execute`/`__testing` on it as properties, which the loader never walks.
 *   4. **Never throw.** Not at plugin init, not out of `execute`. Nothing
 *      validated the payload at install time — opencode's embedded installer runs
 *      with `ignoreScripts: true`, so `postinstall` never runs — so a missing
 *      skill dir is a warned, degraded plugin, never a crashed OpenCode startup.
 *      A throw out of `execute` is a silent exit-0 death.
 *
 * THE SCHEMA IS PLAIN JSON SCHEMA, NOT ZOD, and that is a real trade. opencode
 * takes its zod path only if *every* arg value is a zod schema, else it falls
 * back to `{type:"object", properties, required: Object.keys(properties)}`.
 * Consequences: every declared arg becomes required (there is no way to express
 * an optional one), and `parameters` becomes `Unknown` so the host does no
 * runtime validation at all. **A model can send `{"mode": 42}`.** Hence
 * `validateArgs` below, which hand-checks every argument before anything spawns.
 *
 * PLATFORM. Validated on Linux and only there. Paths are composed with
 * `node:path` throughout so the separator follows the host, and the uv probe
 * carries the two `.exe` entries the official Windows installer creates — but
 * nothing downstream of that probe has been measured on Windows or macOS on this
 * route or on the file route.
 */

import { spawn } from "node:child_process"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

// ---------------------------------------------------------------------------
// Where we are. The load-bearing insight of this whole route.
// ---------------------------------------------------------------------------

/**
 * The plugin knows where it is; the shim never could.
 *
 * `import.meta.dirname` resolves to the package's real on-disk directory — the
 * loader resolves `main`/`exports` to an absolute path and `pathToFileURL`s it,
 * so this is never opencode's `/$bunfs/`. A package upgrade lands in a new cache
 * directory and this follows it: no baked prefix, no re-render, no
 * `install-host --all` after an update, and no staleness to check.
 */
const PACKAGE_DIR = path.resolve(import.meta.dirname, "..")
const SKILL_DIR = path.join(PACKAGE_DIR, "skills", "multi-llm")
const DIST_DIR = path.join(PACKAGE_DIR, "dist")

const COMMAND_NAME = "multi-llm"

/**
 * The tool's name for the one case where there is no surface to read it from:
 * the warning that says the tool is *not* registered. Everywhere a tool is
 * registered or named in a result it is `surface.tool_name`, whose owner is
 * `modes.TOOL_NAME`; this literal only ever reaches stderr.
 */
const DEGRADED_TOOL_NAME = "multi_llm"

/**
 * The placeholder the build leaves wherever a generated artifact would
 * otherwise carry an absolute skill dir. Resolving it is this file's job and
 * only this file's: the package lands in whatever cache directory npm gives it,
 * so a path baked at build time is a path that is wrong on every machine.
 */
const SKILL_DIR_TOKEN = "{SKILL_DIR}"

/**
 * Bumped in lockstep with `packaging/common/surface.py`'s
 * SURFACE_SCHEMA_VERSION. 4: `tool_name`, `tool_description` and
 * `argument_descriptions` are read from the surface rather than composed here.
 * 5: `dispatcher.route_env` (the route signal's name and value, set on every
 * child this file spawns, so a finished phase's next command is spelled for
 * this route) and each mode row's `optional_why`.
 */
const SUPPORTED_SURFACE_VERSION = 5

// ---------------------------------------------------------------------------
// The generated surface
// ---------------------------------------------------------------------------

/**
 * Read `dist/cli-surface.json`, or null.
 *
 * Everything the model reads comes from here. A payload/plugin mismatch is a
 * named error rather than a TypeError inside a `.map`, which is what the version
 * check buys.
 */
function loadSurface() {
  try {
    const raw = fs.readFileSync(path.join(DIST_DIR, "cli-surface.json"), "utf8")
    const surface = JSON.parse(raw)
    if (surface?.schema_version !== SUPPORTED_SURFACE_VERSION) {
      warn(
        `cli-surface.json declares schema_version ${surface?.schema_version}, ` +
          `but this plugin build understands ${SUPPORTED_SURFACE_VERSION}. ` +
          `The package is inconsistent; reinstall it.`,
      )
      return null
    }
    // Schema v5's route signal, set on every child. A surface without a usable
    // one is refused whole, like a version mismatch, never half-used: spawning
    // without it would print a finished phase's next command in the file-shim
    // spelling on this route.
    const routeEnv = surface.dispatcher?.route_env
    if (!isNonEmptyString(routeEnv?.name) || !isNonEmptyString(routeEnv?.value)) {
      warn(
        `cli-surface.json declares no usable dispatcher.route_env (a non-empty ` +
          `name and value). The package is inconsistent; reinstall it.`,
      )
      return null
    }
    return surface
  } catch (error) {
    warn(`cannot read the generated CLI surface: ${describe(error)}`)
    return null
  }
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.length > 0
}

function warn(message) {
  // stderr, never stdout: a host may be reading this process's stdout.
  console.error(`[multi-llm] ${message}`)
}

function describe(error) {
  if (!error) return "unknown error"
  return error.stack || error.message || String(error)
}

// ---------------------------------------------------------------------------
// The model-facing definition, read whole from the generated surface
// ---------------------------------------------------------------------------

/**
 * The tool's argument schema: four required strings/arrays whose descriptions
 * are the surface's `argument_descriptions`, byte for byte.
 *
 * Only the JSON-Schema *types* are this file's — they are the shape `execute`
 * validates, not words a model reads. The description text (the mode
 * vocabulary, the plan and question rules, the reserved-flag list, and any
 * host-keyed clause such as this host's `@` sigil) is rendered in Python by
 * `packaging/common/surface.py` from the mode table, so this file holds no
 * sentence of it. `mode` is a plain string, not an enum: the fallback schema
 * is `{type, properties, required}` and has no `enum` to carry one.
 */
function toolArgs(surface) {
  const described = surface.argument_descriptions
  const text = (name) => {
    const value = described[name]
    if (typeof value !== "string") {
      throw new TypeError(`argument_descriptions.${name} is not a string`)
    }
    return value
  }
  return {
    mode: { type: "string", description: text("mode") },
    plan: { type: "string", description: text("plan") },
    question: { type: "string", description: text("question") },
    extra_flags: {
      type: "array",
      items: { type: "string" },
      description: text("extra_flags"),
    },
  }
}

/** The surface's `tool_description`, checked for shape and nothing else. */
function toolDescription(surface) {
  if (typeof surface.tool_description !== "string") {
    throw new TypeError("tool_description is not a string")
  }
  return surface.tool_description
}

// ---------------------------------------------------------------------------
// Reserved-flag matching — argparse's rule, not a literal-token scan
// ---------------------------------------------------------------------------

/**
 * Whether option-name token `head` would bind to option `name`.
 *
 * A transliteration of `cli_common._flag_matches`, and the suite asserts the two
 * agree over a shared case table rather than trusting this comment. Matching
 * literal spellings does not close the guard: **no parser in the skill passes
 * `allow_abbrev=False`**, so argparse binds `--ho`, `--ye` and `--det` to their
 * full options — a guard that scanned exact spellings was bypassable by typing
 * three fewer characters — and `--host=codex` hides its head behind an `=`.
 */
function flagMatches(head, name, exactOptions) {
  if (head === name) return true
  if (!name.startsWith("--")) return false // short options never abbreviate
  if (!head.startsWith("--") || head.length <= 2) return false
  // A real option spelled in full is not an abbreviation of a longer one:
  // argparse resolves an exact match before it considers any abbreviation.
  if (exactOptions.includes(head)) return false
  return name.startsWith(head)
}

/** The reserved name a token would bind to, or null. */
function reservedHit(token, surface) {
  const head = token.split("=", 1)[0]
  const { exact, named_exception: exception, prefixes, exact_options: exacts } =
    surface.reserved_flags
  for (const prefix of prefixes) {
    // Reserved by *prefix*, not by the two flags that exist today, so a third
    // hidden re-entry flag is covered the day it lands rather than the day
    // someone remembers. `--internal-worker` turns a launch into an
    // unsupervised in-process worker; `--internal-run-id` claims another run.
    if (head.startsWith(prefix)) return prefix + "*"
  }
  for (const name of [...exact, ...exception]) {
    if (flagMatches(head, name, exacts)) return name
  }
  return null
}

// ---------------------------------------------------------------------------
// Argument validation — every rule, before anything spawns
// ---------------------------------------------------------------------------

/**
 * *args* with a left-out `question` read as `""`, its empty value.
 *
 * Only `ask` takes a question, and the property's own description says it is
 * `""` for every other mode; a host model that left it out of an
 * `apply-suggestions` call got `"question" must be a string, got undefined`
 * and had to retry (racing re-run 3, opencode 02b). A question left out of an
 * `ask` call is still refused by the arity rule, which names what to send.
 */
function withAbsentQuestion(args) {
  if (args && typeof args === "object" && !Array.isArray(args) && !("question" in args)) {
    return { ...args, question: "" }
  }
  return args
}

/** `{ ok: true, mode }` or `{ ok: false, error }`. Never throws. */
function validateArgs(args, surface) {
  if (args === null || typeof args !== "object" || Array.isArray(args)) {
    return fail(`arguments must be an object, got ${typeName(args)}.`)
  }
  const { mode, plan, question, extra_flags: extraFlags } = args

  if (typeof mode !== "string" || mode.length === 0) {
    return fail(
      `"mode" must be a non-empty string, got ${typeName(mode)}. Legal words: ` +
        surface.modes.flatMap((m) => [m.name, ...m.aliases]).join(", "),
    )
  }
  const row = resolveMode(mode, surface)
  if (!row) {
    return fail(
      `unknown mode "${mode}". Legal words: ` +
        surface.modes.flatMap((m) => [m.name, ...m.aliases]).join(", ") +
        `. Nothing is prefix-matched or guessed.`,
    )
  }
  if (typeof plan !== "string") {
    return fail(`"plan" must be a string, got ${typeName(plan)}.`)
  }
  if (typeof question !== "string") {
    const asking = surface.modes
      .filter((m) => m.positionals.includes(surface.positional_kinds.question))
      .map((m) => m.name)
    return fail(
      `"question" must be a string: the question for ` +
        `${asking.join(" or ") || "no mode"}, "" (or the key left out) for ` +
        `every other mode. Got ${typeName(question)}.`,
    )
  }
  if (!Array.isArray(extraFlags) || extraFlags.some((t) => typeof t !== "string")) {
    return fail(
      `"extra_flags" must be an array of strings (use [] for none), got ` +
        `${typeName(extraFlags)}.`,
    )
  }

  // Arity is the mode's own declaration, never a JS judgement. This is what
  // retires the flat "an empty plan is invalid" rule, which `init` — a mode
  // declaring no positional at all — would have had to violate.
  const takesPlan = row.positionals.includes(surface.positional_kinds.plan)
  const takesQuestion = row.positionals.includes(
    surface.positional_kinds.question,
  )
  if (takesPlan && plan.trim() === "") {
    return fail(`mode "${row.name}" takes a plan path and "plan" was empty.`)
  }
  if (!takesPlan && plan !== "") {
    return fail(
      `mode "${row.name}" declares no plan positional, so "plan" must be "".`,
    )
  }
  if (takesQuestion && question.trim() === "") {
    return fail(
      `mode "${row.name}" takes a question and "question" was empty: send ` +
        `the user's question as "question".`,
    )
  }
  if (!takesQuestion && question !== "") {
    return fail(
      `mode "${row.name}" declares no question positional, so "question" must ` +
        `be "".`,
    )
  }
  if (takesPlan && plan.startsWith("-")) {
    return fail(
      `"plan" looks like a flag ("${plan}"). The dispatcher takes leading bare ` +
        `words as positionals and forwards everything from the first flag on, ` +
        `so this would arrive with one positional missing.`,
    )
  }
  if (takesQuestion && question.startsWith("-")) {
    // The dispatcher's grammar takes leading bare words as positionals and
    // forwards everything from the first flag on verbatim, so such a question
    // ends the positional scan. Lifting this is a dispatcher-grammar change,
    // never a JS workaround.
    return fail(
      `"question" may not begin with "-" — the dispatcher would read it as a ` +
        `flag and the mode would arrive one positional short. A question that ` +
        `merely contains a flag is fine; only the first character matters.`,
    )
  }

  const split = splitExtraFlags(extraFlags, surface)
  if (split.refused) {
    const { token, hit } = split.refused
    if (surface.reserved_flags.named_exception.includes(hit)) {
      return fail(interactiveRefusal(hit, token))
    }
    return fail(
      `"${token}" is reserved: this tool composes ${hit} itself, so a call may ` +
        `not duplicate, contradict or restate it. Remove it from extra_flags.`,
    )
  }

  return { ok: true, mode: row, extraFlags: split.kept, dropped: split.dropped }
}

/**
 * `extra_flags` split into what is forwarded, what is dropped as a no-op, and
 * the first token that is refused.
 *
 * A token naming a flag this tool composes is a no-op when the surface lists
 * it as `reserved_flags.redundant` — a valueless one (`--yes` in every
 * spelling, `--exec-subagents`) always, a valued one (`--host`) only with the
 * value this tool composes, read from the `=` form or the next element. Every
 * other reserved token is refused, as is a valued one missing its value or
 * naming another value. A surface without `redundant` drops nothing, so every
 * reserved token is refused exactly as before.
 */
function splitExtraFlags(extraFlags, surface) {
  const redundant = surface.reserved_flags.redundant ?? { flags: [], valued: {} }
  const kept = []
  const dropped = []
  for (let index = 0; index < extraFlags.length; index += 1) {
    const token = extraFlags[index]
    const hit = reservedHit(token, surface)
    if (!hit) {
      kept.push(token)
      continue
    }
    if ((redundant.flags ?? []).includes(hit) && !token.includes("=")) {
      dropped.push(token)
      continue
    }
    const value = redundant.valued?.[hit]
    if (typeof value === "string") {
      const inline = token.includes("=")
      const given = inline ? token.slice(token.indexOf("=") + 1) : extraFlags[index + 1]
      if (given === value) {
        dropped.push(inline ? token : `${token} ${given}`)
        if (!inline) index += 1
        continue
      }
    }
    return { kept, dropped, refused: { token, hit } }
  }
  return { kept, dropped, refused: null }
}

/**
 * The refusal for `--interactive`/`-i`, which must name the alternatives.
 *
 * Interactive mode is unsupported on this route, and refused rather than
 * substituted. With `--detach` the dispatcher rejects the pair at parse time;
 * without it the forced model picker has no TTY here, and what happens then
 * depends on which picker binary is on the user's PATH — a silently empty model
 * set where gum/fzf is installed, a hard error where neither is. An
 * environment-dependent failure is the one shape an acceptance criterion must
 * not have, so the flag is refused before any spawn.
 */
function interactiveRefusal(name, token) {
  return (
    `"${token}" is refused: interactive model selection is unsupported on this ` +
    `route. The tool spawns the dispatcher with no terminal on its stdin, so ` +
    `${name} would force a model picker that cannot prompt. Choose models ` +
    `non-interactively instead — --models "provider:model,…" or --quick in ` +
    `extra_flags, or defaults.models in providers.yaml — or use the file-route ` +
    `shim (install-host) from a real terminal, where interactive selection ` +
    `still works.`
  )
}

function resolveMode(word, surface) {
  return (
    surface.modes.find((mode) => mode.name === word) ||
    surface.modes.find((mode) => mode.aliases.includes(word)) ||
    null
  )
}

function typeName(value) {
  if (value === null) return "null"
  if (Array.isArray(value)) return "array"
  return typeof value
}

function fail(message) {
  return { ok: false, error: message }
}

// ---------------------------------------------------------------------------
// uv preflight — the probe, branch for branch
// ---------------------------------------------------------------------------

/**
 * The absolute path of a usable uv, or null.
 *
 * Mirrors `unattended.find_uv()` branch for branch, not merely list for list:
 * `$MULTI_LLM_UV` → PATH → the candidate list. An override that is not a file
 * falls through to PATH rather than failing. The candidate list itself comes
 * from the generated surface, so it cannot be a shortened restatement — the two
 * `.exe` entries are the entire Windows story, and a reimplementation that
 * dropped them would recreate precisely the failure this route exists to remove.
 */
function findUv(surface, env) {
  const override = env[surface.uv.env]
  if (override && isFile(override)) return path.resolve(override)

  const onPath = whichUv(env)
  if (onPath) return onPath

  for (const candidate of surface.uv.candidates) {
    const expanded = expandHome(candidate, env)
    if (isFile(expanded)) return expanded
  }
  return null
}

function whichUv(env) {
  const raw = pathVariable(env)
  if (!raw) return null
  const names = process.platform === "win32" ? ["uv.exe", "uv"] : ["uv"]
  for (const dir of raw.split(path.delimiter)) {
    if (!dir) continue
    for (const name of names) {
      const candidate = path.join(dir, name)
      if (isFile(candidate) && isExecutable(candidate)) {
        return path.resolve(candidate)
      }
    }
  }
  return null
}

/**
 * The PATH of an environment object, under whatever name the platform gave it.
 *
 * `process.env` itself is case-insensitive on Windows, but the copy `execute`
 * takes of it is an ordinary object, and it keeps each name as the system
 * spelled it: `Path`. `env.PATH` on that copy is `undefined`, so the probe
 * would walk no directory and report a uv that is on the PATH as missing —
 * silently, for every Windows install the candidate list does not name (winget,
 * scoop, pip). The child gets the copy whole, so it never needed this.
 */
function pathVariable(env) {
  if (process.platform !== "win32") return env.PATH || ""
  const name = Object.keys(env).find((key) => key.toUpperCase() === "PATH")
  return (name && env[name]) || ""
}

function isFile(target) {
  try {
    return fs.statSync(target).isFile()
  } catch {
    return false
  }
}

function isExecutable(target) {
  try {
    fs.accessSync(target, fs.constants.X_OK)
    return true
  } catch {
    return false
  }
}

/**
 * `~` and `~/…` against the home directory. No shell will do it now.
 *
 * It takes the environment rather than reading `process.env` directly, because
 * the uv probe is handed an environment and every branch of that probe must
 * resolve against the *same* one — a candidate list expanded against a different
 * home than the PATH it was compared to is a probe that cannot be reasoned
 * about, or tested.
 */
function expandHome(target, env = process.env) {
  const home = env.HOME || env.USERPROFILE || os.homedir()
  if (target === "~") return home
  if (target.startsWith("~/") || target.startsWith("~\\")) {
    return path.join(home, target.slice(2))
  }
  return target
}

/**
 * The uv-missing message, read from the packaged tree.
 *
 * Never a string literal here: the wording lives in `hosts/uv-missing.txt` and
 * that is its single owner, exactly as it is for the file route's shim.
 */
function uvMissingText(surface) {
  const file = path.join(SKILL_DIR, ...surface.uv.missing_message_file.split("/"))
  try {
    // Line endings read the way Python's text mode reads them (CRLF is LF), so a
    // CRLF checkout cannot leave a `\r` where the trailing newline was trimmed.
    return fs.readFileSync(file, "utf8").replace(/\r\n/g, "\n").replace(/\n+$/, "")
  } catch {
    // A partial install is the only way to get here, so say so rather than
    // duplicating the text this file deliberately does not carry.
    return (
      `[UV_MISSING] uv is required and was not found, and this install is ` +
      `missing ${file}. Install uv ` +
      `(https://docs.astral.sh/uv/getting-started/installation/) and retry. ` +
      `There is no bare-python fallback.`
    )
  }
}

// ---------------------------------------------------------------------------
// argv composition
// ---------------------------------------------------------------------------

/**
 * The dispatcher argv, in the one order the grammar accepts.
 *
 * mode word → the mode's positionals **in declared order** → the baked
 * `--host <id> --yes --exec-subagents` → `--detach` for a supports_detach mode
 * only → extra_flags verbatim.
 *
 * Positionals must precede every flag or the dispatcher's `_split_positionals`
 * stops early, and an empty plan or question contributes **zero** argv elements
 * — never an empty-string element, which would read as a bare word and fail the
 * arity check.
 */
function composeDispatcherArgv(row, values, surface) {
  // The word the caller sent, **not** `row.name`: aliases travel unexpanded.
  // `modes.ALIAS_FLAGS` prepends `--abort` for abort/cancel/stop, so expanding
  // `abort` to `status` here would compose a plain status *view* of a run the
  // caller asked to stop — the flag the alias contributes would never be
  // injected. The plugin never expands an alias itself and never appends the
  // flag one already contributes.
  const argv = [values.word ?? row.name]
  for (const kind of row.positionals) {
    if (kind === surface.positional_kinds.plan && values.plan) {
      // The user's own spelling when the caller kept one (`planArg`), else the
      // resolved path: the dispatcher runs in the anchor, so a relative plan
      // names the same file, and it is the spelling the run records as
      // `plan_arg` and a finished phase's next command is printed with.
      argv.push(values.planArg || values.plan)
    } else if (kind === surface.positional_kinds.question && values.question) {
      argv.push(values.question)
    }
  }
  argv.push(...surface.baked_flags)
  if (row.supports_detach) argv.push(surface.detach_flag)
  argv.push(...values.extraFlags)
  return argv
}

/**
 * The full spawn argv: uv, its flags, `--project`, then the dispatcher's.
 *
 * The uv flags come from the generated surface, whose owner is
 * `unattended.uv_run_args()`. `--no-dev` is the load-bearing one: `--frozen`
 * suppresses only the lockfile staleness check and does not stop the sync, and
 * `uv run` syncs the dev group by default — so without it a user's first
 * dispatch pulls 17 packages and every offline posture fails outright.
 */
function composeSpawnArgv(uv, dispatcherArgv, surface, env) {
  const override = env[surface.uv.run_args_env]
  // Split on ASCII whitespace with no quote processing and no shell — no uv flag
  // contains a space. It **replaces** the default list and never appends, which
  // is what makes `--no-sync` expressible at all: appended, it would sit beside
  // the very sync the setter is trying to suppress. Unset, empty or
  // whitespace-only means the default, so a stray `export VAR=` cannot silently
  // re-enable the dev-group sync.
  const uvFlags =
    override && override.trim() !== ""
      ? override.trim().split(/\s+/)
      : surface.uv.default_run_args
  return [
    "run",
    ...uvFlags,
    "--project",
    SKILL_DIR,
    "--",
    "python",
    path.join(SKILL_DIR, surface.dispatcher.script),
    ...dispatcherArgv,
  ]
}

/**
 * The workspace the child runs in, and what a relative plan is relative to.
 *
 * `ctx.worktree ?? ctx.directory`, in that order: the worktree is the checkout
 * the session is editing and `directory` the project it was opened from, and
 * they differ exactly when the session runs in a git worktree, where the
 * worktree is the right answer. Neither is in the published ToolContext type, so
 * a missing or non-absolute value falls through, finally to `process.cwd()`.
 *
 * This is not cosmetic. `<git-root>/.multi-llm/providers.yaml` discovery walks up
 * from here, and `init` resolves its target from the cwd alone — spawned from the
 * package cache it would scaffold inside `~/.cache/opencode/packages/…` and every
 * later run would silently fall back to the shipped defaults. The anchor actually
 * used is echoed in the result, because a wrong one is otherwise invisible: it
 * surfaces as a model-selection bug, not as a config error.
 */
function resolveAnchor(ctx) {
  for (const candidate of [ctx?.worktree, ctx?.directory]) {
    if (typeof candidate === "string" && candidate && path.isAbsolute(candidate)) {
      return candidate
    }
  }
  return process.cwd()
}

/**
 * How long the session-model lookup may take before the run goes on without it.
 *
 * It is a nicety, not a prerequisite: the ladder has three more rungs under it,
 * so a slow or wedged `client` must cost a moment and never a dispatch.
 */
const SESSION_MODEL_BUDGET_MS = 3000

/**
 * The model the OpenCode session is itself running on, as `provider/model[:variant]`.
 *
 * WHY THE PLUGIN HAS TO ASK. `resolve_subagent_runner`'s ladder had no rung for
 * "what the operator selected": an explicit `driver.subagent_runner` won, else
 * the host's shipped credential-free floor did. So a run launched from a session
 * whose picker said one model dispatched all of its batches on another, with
 * nothing on screen to say so. The `ToolContext` carries no model — only
 * `sessionID`, `messageID`, `agent`, `directory`, `worktree` and `abort` — but
 * the plugin factory's own input carries the SDK `client`, and the session's
 * newest assistant message records the `providerID`/`modelID` it was produced
 * with. That is the observation; what it *means* is Python's to decide, which is
 * why this returns a string and sets no policy.
 *
 * THE VARIANT IS PART OF THE SELECTION. The same `info` carries `variant` — the
 * reasoning effort the picker is set to — and dropping it made a session opened
 * with `--variant low` dispatch every one of its subagents at the model's
 * *default* effort, so the run's cost and latency silently diverged from what
 * the operator chose and the echoed `session_model` told only two thirds of the
 * truth. It rides along as the adapter's own `model:effort` suffix, which is the
 * spelling `utils/providers/opencode.py` already splits back off and renders as
 * `run --variant <name>`; still no policy here — which models may not take one
 * is a measurement, and it lives with the adapter that emits the flag. The one
 * thing checked is *shape*: a variant carrying `:`, `/` or whitespace could not
 * be split back off as a suffix and would corrupt the model id instead, so such
 * a word is left off rather than glued on.
 *
 * Best effort by construction, and every failure is swallowed (rule 4): an old
 * host with no `client`, a route that moved, a hang. `Promise.race` against a
 * timer rather than an `AbortSignal`, because the budget has to hold even
 * against a client that ignores one — and the timer is deliberately NOT
 * `unref`'d: a hung client leaves nothing else on the loop, and an unref'd timer
 * would let node drain and exit with the await still pending instead of letting
 * the budget expire. `clearTimeout` in the `finally` is what keeps the ordinary
 * path from paying for it.
 */
async function resolveSessionModel(client, ctx, budgetMs = SESSION_MODEL_BUDGET_MS) {
  const sessionID = ctx?.sessionID
  if (typeof sessionID !== "string" || !sessionID) return null
  const fetch = client?.session?.messages
  if (typeof fetch !== "function") return null
  let timer = null
  try {
    const answered = await Promise.race([
      fetch.call(client.session, { path: { id: sessionID } }),
      new Promise((resolve) => {
        timer = setTimeout(() => resolve(null), budgetMs)
      }),
    ])
    const messages = Array.isArray(answered?.data)
      ? answered.data
      : Array.isArray(answered)
        ? answered
        : []
    // Newest first: the run is about to dispatch on whatever the session is
    // using *now*, which is the last assistant turn, not the first.
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const info = messages[i]?.info
      if (!info || info.role !== "assistant") continue
      const provider = typeof info.providerID === "string" ? info.providerID.trim() : ""
      const model = typeof info.modelID === "string" ? info.modelID.trim() : ""
      if (!provider || !model) continue
      const variant = typeof info.variant === "string" ? info.variant.trim() : ""
      const suffixable = variant && !/[\s:/]/.test(variant)
      return suffixable ? `${provider}/${model}:${variant}` : `${provider}/${model}`
    }
    return null
  } catch (error) {
    // Never fatal: the ladder's remaining rungs are the fallback, and they are
    // the behaviour every run had before this lookup existed.
    warn(`could not read the session model: ${describe(error)}`)
    return null
  } finally {
    if (timer) clearTimeout(timer)
  }
}

/**
 * A plan path normalized against the anchor. Resolving is not quoting.
 *
 * One leading `@` comes off first: it is this host's file-attachment sigil, so
 * `--review-plan @plans/x.md` is the ordinary way a user names a file here and
 * the text reaches the model with the sigil still on it. Left in place it
 * resolves to `<workspace>/@plans/x.md` and the run dies on "Plan file not
 * found", with nothing in the error naming the character at fault. Exactly one,
 * and only at the front: a path may legitimately contain `@` anywhere else, and
 * `@@x` is not a doubled sigil. A bare `"@"` keeps its sigil rather than
 * becoming the empty string, so the positional still reaches the dispatcher and
 * the refusal names what the user typed instead of a missing argument.
 */
function normalizePlan(plan, anchor) {
  const spelled = planSpelling(plan)
  if (!spelled) return ""
  if (path.isAbsolute(spelled)) return spelled
  return path.resolve(anchor, spelled)
}

/**
 * Name a plan typed from the repository root by the root's file, when the
 * dispatcher's result line says it ran from there.
 *
 * The dispatcher runs a command whose relative paths name nothing from its cwd
 * (the anchor, here) from the root of the repository that cwd is in (fix round
 * 26, live run 11), and its result line names that root (the surface's
 * `dispatcher.ran_from_root_key`). So the polls, the abort and the options this
 * server composes name the file that run reads, never a path under the anchor
 * that does not exist. Fix round 27 (the Fable review of round 26): this server
 * walked to the nearest `.git` itself, which named the root's plan where the
 * dispatcher refused to move (`GIT_DIR` set, a planted `core.worktree`, a
 * planted nested `.git`); it now takes the dispatcher's own answer, off a line
 * it reads anyway, and nothing else. A root that is not absolute or that the
 * anchor is not inside is ignored, as is a plan typed absolute. The codex
 * server's `plan_from_root` is its port.
 */
function planFromRoot(result, values, anchor, surface) {
  const key = surface?.dispatcher?.ran_from_root_key
  const root = key && result && typeof result === "object" ? result[key] : null
  const typed = values?.planArg
  if (
    typeof root !== "string" || !path.isAbsolute(root) ||
    typeof typed !== "string" || !typed || path.isAbsolute(typed)
  ) {
    return
  }
  let here
  try {
    here = realPath(anchor)
  } catch {
    return
  }
  const base = path.resolve(root)
  const prefix = base.endsWith(path.sep) ? base : base + path.sep
  if (here !== base && !here.startsWith(prefix)) return
  values.plan = path.resolve(base, typed)
}

/**
 * The plan as the user spelled it: the sigil off and `~` expanded, **nothing
 * resolved**.
 *
 * `normalizePlan` makes the path absolute so this server can name the file, and
 * that is also the string the dispatcher used to receive, record as `plan_arg`
 * and print in every next command (2026-10-02, issue #15: a user who typed
 * `plans/racing-game.md` was told to type `/multi-llm --implement
 * /tmp/…/plans/racing-game.md`). The child runs in the anchor, so a relative
 * spelling resolves to the same file there, and the launch passes this one.
 * Everything this server composes for itself (the polls, the abort) keeps the
 * absolute path; the tool calls it hands back carry this one (`handBackTyped`).
 */
function planSpelling(plan) {
  if (!plan) return ""
  const sigilless =
    plan.startsWith("@") && plan.length > 1 ? plan.slice(1) : plan
  return expandHome(sigilless)
}

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

/** Retained stdout budget. What is bounded is what is *kept*, never the scan. */
const STDOUT_TAIL_BYTES = 64 * 1024
const STDERR_TAIL_BYTES = 8 * 1024
/**
 * The incremental line buffer's own cap. The retained tail is bounded, but the
 * buffer feeding it is only bounded by the child's newlines, and a child that
 * emits none must not be able to grow it without limit. Deliberately larger
 * than the retained tail so a legitimate single-line `MULTI_LLM_RESULT:`
 * payload still arrives whole; past it the line's *head* is what survives,
 * which is where the marker is.
 */
const STDOUT_LINE_BYTES = 4 * STDOUT_TAIL_BYTES
/** SIGTERM → SIGKILL grace, shared by the abort and the timeout paths. */
const KILL_GRACE_MS = 5000

/**
 * Run one child to completion. Resolves; never rejects.
 *
 * Asynchronous `spawn`, an argv array, `shell: false`. Never `exec`, never a
 * concatenated string, never a shell — a shell re-opens exactly the injection
 * surface this design closes, and only an async spawn can be interrupted by an
 * abort at all. `plan`, `question` and `extra_flags` are all model-controlled;
 * on a composed command string the reserved-flag check would be the only thing
 * standing between a model and a `;` inside a plan path, which is not a load a
 * flag check can carry.
 *
 * **The whole stream is read.** The payload is the *last* `MULTI_LLM_RESULT:`
 * line and an arbitrarily long body can precede it, so a fixed-size tail read is
 * not a legal implementation — a body larger than the tail would still contain
 * the result line while the tail no longer did. Every line is scanned; what is
 * retained is the newest result candidate plus the preceding text to a budget.
 */
function runChild({
  uv, argv, cwd, env, signal, marker, banner: fields, timeoutMs,
}) {
  const bannerPattern = new RegExp(
    `^\\s*(${(fields ?? ["run_id"]).join("|")}):\\s*(.+?)\\s*$`,
  )
  return new Promise((resolve) => {
    let child
    try {
      child = spawn(uv, argv, {
        cwd,
        env,
        shell: false,
        // Its own process group, so a kill can reach the whole tree: `uv run`
        // is a wrapper and the python grandchild is what actually holds stdout
        // open, so signalling the direct child alone leaves it alive with
        // `close` — the event this promise resolves on — never firing.
        detached: true,
        // No console window for that new group on Windows.
        windowsHide: true,
        // /dev/null, the same thing the detached worker already gets. Any code
        // path that does reach a prompt gets EOF and a visible failure rather
        // than blocking forever on a pipe nobody writes to.
        stdio: ["ignore", "pipe", "pipe"],
      })
    } catch (error) {
      resolve({ spawnError: describe(error), code: null, result: null })
      return
    }

    let stdoutTail = ""
    let stderrTail = ""
    let pending = ""
    let overflowed = false
    let truncated = false
    let lastResult = null
    let lastResultRaw = null
    const banner = {}

    const onLine = (line) => {
      const trimmed = line.trim()
      if (trimmed.startsWith(marker)) {
        lastResultRaw = trimmed.slice(marker.length).trim()
        try {
          lastResult = JSON.parse(lastResultRaw)
        } catch {
          lastResult = null // malformed; the raw text is kept for the report
        }
      }
      // The banner is where run_id comes from: the MULTI_LLM_RESULT payload has
      // no run_id key at all, and the launch's `next` carries --run-id only for
      // `ask` runs. Any other derivation — "the newest run in the phase dir" —
      // would adopt, poll and, worse, abort a concurrent run under the same plan.
      // The field names are generated data, whose owner is detach.BANNER_FIELDS.
      const field = bannerPattern.exec(line)
      if (field && banner[field[1]] === undefined) banner[field[1]] = field[2]

      stdoutTail += line + "\n"
      if (stdoutTail.length > STDOUT_TAIL_BYTES) {
        // Drop from the *beginning*, never the end: the last line is the one
        // that must survive, so what goes is the oldest text. `slice` counts
        // UTF-16 units, so the cut can land between a surrogate pair and leave
        // a lone half at the front — cosmetic damage to text already marked
        // truncated, never to the result line.
        truncated = true
        stdoutTail = stdoutTail.slice(stdoutTail.length - STDOUT_TAIL_BYTES)
      }
    }

    child.stdout.setEncoding("utf8")
    child.stdout.on("data", (chunk) => {
      let text = chunk
      if (overflowed) {
        // Still inside an over-long line: everything up to its newline is
        // already past the cap, so it is dropped rather than buffered. The head
        // is what gets flushed as the line, which is what keeps the marker
        // check honest — `MULTI_LLM_RESULT:` is a prefix.
        const end = text.indexOf("\n")
        if (end < 0) return
        onLine(pending)
        pending = ""
        overflowed = false
        text = text.slice(end + 1)
      }
      pending += text
      let index
      while ((index = pending.indexOf("\n")) >= 0) {
        onLine(pending.slice(0, index))
        pending = pending.slice(index + 1)
      }
      if (pending.length > STDOUT_LINE_BYTES) {
        // Never an unbounded buffer: a child emitting one enormous unterminated
        // line would otherwise exhaust the host process's memory, the scan
        // being incremental but the buffer under it not.
        truncated = true
        overflowed = true
        pending = pending.slice(0, STDOUT_LINE_BYTES)
      }
    })
    child.stderr.setEncoding("utf8")
    child.stderr.on("data", (chunk) => {
      // Never parsed, never searched for the result marker — stdout is the
      // payload channel — but it is what makes a failure diagnosable: the
      // dispatcher's tracebacks, argparse's usage errors and uv's own complaints
      // all land here and none of them reach stdout.
      stderrTail = (stderrTail + chunk).slice(-STDERR_TAIL_BYTES)
    })

    // SIGTERM, then SIGKILL after a grace, to the whole process group — the
    // same escalation the dispatcher's own abort path runs, and for the same
    // reason: this promise resolves on `close`, which waits for the stdio pipes
    // as well as for the exit, so a grandchild that outlived the TERM still
    // holding stdout would hang the caller indefinitely.
    let killTimer = null
    const signalTree = (name) => {
      const pid = child.pid
      if (pid === undefined) return
      if (process.platform === "win32") {
        const killLeader = () => {
          try { child.kill("SIGKILL") } catch { /* already gone */ }
        }
        if (name === "SIGTERM") {
          // taskkill needs the live wrapper to find its descendants. Killing
          // uv first strands Python, which can keep running the phase.
          try {
            const killer = spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
              shell: false, windowsHide: true, stdio: "ignore",
              timeout: KILL_GRACE_MS, killSignal: "SIGKILL",
            })
            killer.once("error", killLeader)
            killer.once("close", (code) => { if (code !== 0) killLeader() })
          } catch {
            killLeader()
          }
        } else {
          killLeader()
        }
        return
      }
      try {
        process.kill(-pid, name) // negative pid: the group `detached` created
      } catch {
        try {
          child.kill(name) // no group, or already reaped — the child alone
        } catch {
          /* already gone */
        }
      }
    }
    const killTree = () => {
      if (killTimer) return // already escalating
      signalTree("SIGTERM")
      killTimer = setTimeout(() => {
        signalTree("SIGKILL")
        // Past the kill, an fd some survivor still holds is the only thing that
        // could keep `close` from firing; dropping our ends of the pipes settles
        // it either way.
        child.stdout?.destroy()
        child.stderr?.destroy()
      }, KILL_GRACE_MS)
      killTimer.unref?.()
    }

    let aborted = false
    let timedOut = false
    const onAbort = () => {
      aborted = true
      killTree()
    }
    if (signal) {
      if (signal.aborted) onAbort()
      else signal.addEventListener("abort", onAbort, { once: true })
    }
    let timer = null
    if (timeoutMs) {
      // The abort dispatch is the caller that sets one, and it is the call that
      // must not hang: a bare SIGTERM with no escalation and no group reach
      // would leave `abortAndReport` waiting on a child whose grandchild still
      // holds stdout — a paid fan-out left running with nobody reading it.
      //
      // Recorded as well as signalled. `killTree` escalates to SIGKILL and drops
      // our pipe ends, so the promise always settles — but it settles looking
      // like any other early death: no exit code of the child's own, no result
      // line. `timedOut` is what tells the caller which of those it is holding,
      // so a report can say the bound was hit rather than imply the dispatcher
      // answered.
      timer = setTimeout(() => {
        timedOut = true
        killTree()
      }, timeoutMs)
      timer.unref?.()
    }

    const finish = (code, spawnError) => {
      if (timer) clearTimeout(timer)
      if (killTimer) clearTimeout(killTimer)
      if (signal) signal.removeEventListener?.("abort", onAbort)
      if (pending) onLine(pending)
      resolve({
        code,
        spawnError: spawnError ?? null,
        result: lastResult,
        resultRaw: lastResultRaw,
        stdout: stdoutTail,
        stderr: stderrTail,
        truncated,
        banner,
        aborted,
        timedOut,
        argv: [uv, ...argv],
      })
    }

    child.on("error", (error) => finish(null, describe(error)))
    child.on("close", (code) => finish(code ?? null, null))
  })
}

// ---------------------------------------------------------------------------
// The execute state machine
// ---------------------------------------------------------------------------

/**
 * The two explicit bounds on the poll loop: a wall-clock budget and a cap on
 * `next` continuations.
 *
 * Never an unbounded loop, and never a silent give-up that leaves the worker
 * running — on exhausting either, `execute` aborts the run it claimed and
 * returns a needs-human result naming the run id, the log path and the literal
 * abort command.
 *
 * The budget is deliberately generous: a tool `execute` is bound by no timeout
 * here (a probe ran 400 s uninterrupted with the model continuing normally), so
 * this is a runaway backstop rather than a deadline a real fan-out should ever
 * approach. It is emphatically *not* derived from the host's
 * `timeout_ceiling_s`: that is the documented **shell** cap, it binds no tool
 * execution, and reading a host capability into a poll duration is precisely how
 * the shell-cap workaround would get transliterated back in. That is a statement
 * about *this* bound and not about the `--wait` the poll child resolves for
 * itself: the dispatcher does clamp that one against the same ceiling, and the
 * poll loop's comment says why it is accepted there. The difference is who
 * computes it — nothing here reads a capability, and no duration is spelled in
 * this file.
 *
 * Both are overridable **only** through the environment, and only so the suite
 * can drive the exhaustion branch without waiting hours for it. There is no
 * argument, no config key and no model-reachable surface: a bound a caller could
 * lower is a bound a model could lower.
 */
function pollBudgetMs(env) {
  return positiveInt(env.MLLM_TEST_POLL_BUDGET_MS, 6 * 60 * 60 * 1000)
}

function nextContinuationCap(env) {
  return positiveInt(env.MLLM_TEST_NEXT_CAP, 8)
}

/** How long the best-effort abort dispatch is given before it is given up on. */
const ABORT_DISPATCH_TIMEOUT_MS = 30_000

function positiveInt(raw, fallback) {
  const parsed = Number.parseInt(raw ?? "", 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

/**
 * The toast's severity word. An SDK enum value (`"info" | "success" |
 * "warning" | "error"`), like the `"assistant"` the session-model lookup
 * matches on — not a word this file authors about the run.
 */
const TOAST_VARIANT = "info"

/**
 * How long a relayed toast asks to stay on screen, in milliseconds.
 *
 * `tui.showToast`'s `duration` is optional and opencode's own schema defaults
 * it to 5 000 ms, which is what this relay was sending by omitting it: a
 * five-second flash once per poll. Measured 2026-09-21 on a 363 s package-route
 * run — five toasts, four of them ~90 s apart — that is a 5.5 % duty cycle
 * inside a six-minute silence, and the user saw exactly one of the five: the
 * launch one, which lands about a second after they press enter. The channel
 * was never the problem; the dwell was. A toast that lives until its successor
 * replaces it turns a flash into continuous progress.
 *
 * The number is **measured, not configured**, which is what keeps it out of the
 * rule this file is held to. `--wait`'s duration is a fact Python owns (it
 * resolves the project's poll cadence and clamps it to this host's foreground
 * cap), and composing it here is exactly what the poll loop below refuses to
 * do — but the interval between two relays is something *this process*
 * observes, and after the first poll it is that duration by construction. So
 * every relay past the first asks for the gap it just watched plus a margin,
 * and nothing here has to know what the cadence is.
 *
 * The two literals are this file's own display facts, in the sense
 * `TOAST_VARIANT` is: `FIRST_HOLD` covers the launch seam, which fires before
 * any gap exists and must survive until the first poll returns — one `--wait`,
 * which Python clamps under this host's documented ~120 s foreground cap — and
 * `MAX_HOLD` bounds how long a stale toast can outlive a loop that stopped,
 * since nothing clears a toast when `execute` returns.
 */
const TOAST_FIRST_HOLD_MS = 120_000
const TOAST_GAP_MARGIN_MS = 15_000
const TOAST_MAX_HOLD_MS = 5 * 60 * 1000

/** The gap this call just watched, plus a margin. Stamps the clock it reads. */
function toastHoldMs(memory) {
  const now = Date.now()
  const previous = memory?.lastRelayAt ?? null
  if (memory) memory.lastRelayAt = now
  const gap = previous ? now - previous : 0
  if (!Number.isFinite(gap) || gap <= 0) return TOAST_FIRST_HOLD_MS
  return Math.min(gap + TOAST_GAP_MARGIN_MS, TOAST_MAX_HOLD_MS)
}

/** One relay counter, where the memory may legitimately be absent. */
function countRelay(memory, key) {
  if (memory?.relays && typeof memory.relays[key] === "number") {
    memory.relays[key] += 1
  }
}

/**
 * The relay's own tally, for the result line. Numbers, never a sentence.
 *
 * It exists because two very different failures leave identical evidence, and
 * a forensic pass over a real run could not separate them: `showToast`
 * persists nothing in `opencode.db`, a rejected POST used to be swallowed by a
 * bare `.catch(() => {})`, and `warn` is `console.error` on a stderr
 * `opencode.log` does not capture (grep it for this plugin's prefix and you get
 * nothing). "The relay fired five times and you missed four" and "the relay
 * fired five times and every POST failed" produced the same artifacts: none.
 *
 * `attempted` is decisive on its own and is exact when this is read.
 * `delivered` and `rejected` settle asynchronously — nothing awaits the SDK
 * call, which is what fire-and-forget costs — so the **last** relay of a call
 * may still be in flight here and land in neither. They are a floor, not a
 * census, and `attempted - delivered - rejected <= 1` is the shape to read them
 * by. `unavailable` counts the relays this host had no toast route for at all,
 * which is the third diagnosis and the only one that was ever visible.
 */
function relayCounts(memory) {
  return memory?.relays ? { ...memory.relays } : null
}

/**
 * One fire-and-forget progress update, on every channel this host offers.
 *
 * WHY THERE IS A RELAY AT ALL. On the package route there is no agent in the
 * loop: `execute` supervises the run itself and returns exactly once, so the
 * poll loop below already holds every number a user wants mid-run — which
 * models are running, how many have finished — and is otherwise silent for as
 * long as the fan-out takes. There is no `tool.execute.progress` hook to hang
 * this on either; `tool.execute.after` fires only once the result exists.
 *
 * WHY BOTH CHANNELS FIRE, AND WHICH ONE THE DESIGN RESTS ON. `ctx.metadata` is
 * `ToolContext`'s own fire-and-forget callback, callable any number of times
 * during `execute`. It attaches the update to *this* tool part rather than to
 * the session and costs no round trip, so it is the cheap and correct **record**
 * of what the loop knew and when. What it is not is a mid-run *render*, and
 * that is now measured rather than doubted: on a 5 m 40 s four-model run
 * (2026-09-21) opencode's `event` table carried that tool part's whole life as
 * `pending` → `running` → `completed` on three contiguous per-session `seq`
 * values, with no gap — across four `relay()` calls, no `ctx.metadata` call
 * produced a persisted part update at all. It is kept because it costs nothing
 * and another renderer may yet paint it; nothing here depends on it.
 * `client.tui.showToast` is the live channel, and what it carries is the *same
 * update* rendered for a human rather than serialized (see WHAT TRAVELS): it is
 * session-global, noisier and costs an SDK round trip, but it is the channel
 * that is displayed, and it is what keeps the package route from being silent
 * until the fan-out answers. A context with no `metadata` is one channel down;
 * a caller with no `client` is the relay's only channel gone, which is why that
 * one is said out loud rather than passed over, and a channel that throws never
 * suppresses the other one.
 *
 * WHAT TRAVELS. The update object as handed in, unreworded — the dispatcher's
 * own `progress` object, the line(s) the dispatcher already *rendered* from it,
 * and the identifiers the tool's result already spells (`run_id`, `log`,
 * `polls`). Nothing is scraped out of rendered text and nothing is summarised
 * into a sentence: a regex over a human line, or a phrase composed here, would
 * be the first break in the rule that this file holds no mode word, flag or
 * message of its own. The toast has to fill a `message` field, and what fills
 * it is one of the strings the dispatcher supplied, passed through byte for
 * byte — `MULTI_LLM_RESULT.progress_lines`, composed by the same renderers that
 * write the human `status` view. This file selects which of the supplied
 * strings to show and remembers which it has shown; it authors none of them.
 * With none supplied — an older dispatcher, or a run with no per-model map —
 * the object goes over as JSON exactly as it always did.
 *
 * Never throws and never blocks (rule 4), and returns whether *anything* was
 * relayed. A relay is a nicety on top of a run that is already paid for; a host
 * whose channel rejects must cost a warning line, never the fan-out.
 *
 * `memory` is the caller's per-call relay state (see `relayMemory`) and is
 * optional: without it nothing is deduplicated and the absent-channel warning
 * is not rate-limited, which is the right default for a single direct call.
 */
function relayProgress(ctx, client, update, memory, title) {
  let relayed = false
  countRelay(memory, "attempted")
  try {
    if (typeof ctx?.metadata === "function") {
      ctx.metadata({ metadata: update })
      relayed = true
    }
  } catch (error) {
    warn(`could not relay progress through the tool context: ${describe(error)}`)
  }
  try {
    const toast = client?.tui?.showToast
    if (typeof toast === "function") {
      const answered = toast.call(client.tui, {
        body: {
          // The tool's own registration key (`surface.tool_name`, handed in by
          // `execute`) — not a word this file authors about the run. It is what
          // makes this toast identifiable among whatever else the host is
          // showing.
          title,
          message: toastMessage(update, memory),
          variant: TOAST_VARIANT,
          // Omitted, this defaulted to a five-second flash once per poll. See
          // `toastHoldMs`.
          duration: toastHoldMs(memory),
        },
      })
      // Nothing awaits the SDK call — that is what "fire and forget" costs — so
      // its rejection has to be handled here or it surfaces as an unhandled
      // rejection in the process `execute` is running in. It is *counted*
      // rather than swallowed: a bare `.catch(() => {})` is what made a failed
      // POST and a displayed-and-missed toast leave identical evidence.
      if (answered && typeof answered.then === "function") {
        answered.then(
          () => countRelay(memory, "delivered"),
          () => countRelay(memory, "rejected"),
        )
      } else {
        countRelay(memory, "delivered")
      }
      relayed = true
    } else {
      countRelay(memory, "unavailable")
      if (!memory?.warnedNoToast) {
        // The absence is said out loud, because it is otherwise
        // indistinguishable from the failure it looks like. `showToast` leaves
        // no trace in `opencode.db` and this function's `catch` only fires when
        // a channel *throws*, so a host that simply does not carry the route
        // produced exactly the same evidence as one that displayed the toast
        // and was missed: none. Once per call rather than once per poll — a run
        // that polls twenty times has one missing channel, not twenty. The
        // counter above is the half a user can hand back; this line only
        // reaches a maintainer watching the process's stderr, which
        // `opencode.log` does not capture.
        if (memory) memory.warnedNoToast = true
        warn(
          "no toast channel on this host (client.tui.showToast is absent), so " +
            "nothing this run relays mid-run can be displayed",
        )
      }
    }
  } catch (error) {
    countRelay(memory, "rejected")
    warn(`could not relay progress to the TUI: ${describe(error)}`)
  }
  return relayed
}

/**
 * The relay's memory for one `execute` call.
 *
 * Two facts: which supplied lines have already been shown, and whether the
 * missing-channel warning has been said. Per **call**, which is not the same
 * thing as per run, and the difference is worth stating in both directions.
 *
 * It is not per *module*, and that is the load-bearing half. This module is
 * imported once for a whole OpenCode session, so a Set declared up there would
 * show one run's roster on the session's first tool call and suppress it on
 * every call after — for runs it has never seen. A second tool call is a second
 * run, whose roster is news again.
 *
 * It is not per *run* either, and that is a real gap rather than a
 * simplification. One `execute` can span several runs: a mechanical
 * continuation (`--reaggregate`, `--consolidate`) is driven through the same
 * launch-and-poll cycle and claims a **new** run id, while this state is
 * created once above that loop. A continuation's roster is byte-identical to
 * the one already shown — same models, same budgets, it is the same fan-out
 * being finished off — so it is suppressed, and that run's first poll falls
 * back to the serialized object. Accepted rather than overlooked: the roster
 * answers "what is this fan-out over", the answer has not changed, and
 * resetting per cycle would repeat a line the user is already looking at in
 * order to announce a phase they never asked for. The continuation's
 * `just finished:` lines are news and are shown as usual.
 */
function relayMemory() {
  return {
    shown: new Set(),
    warnedNoToast: false,
    // When the last relay went out, so the next one can ask to stay up for the
    // gap it just watched (`toastHoldMs`).
    lastRelayAt: null,
    // What the relay did, for the result line (`relayCounts`).
    relays: { attempted: 0, delivered: 0, rejected: 0, unavailable: 0 },
  }
}

/**
 * The toast's `message`: the dispatcher's own lines, or the update as JSON.
 *
 * The one piece of judgement this file is allowed over the relay's text, and it
 * is selection, never composition. A supplied line is shown once and then
 * suppressed — the roster is byte-identical on every poll of one run, so
 * repeating it would bury the line that *is* news under the line that is not —
 * and equality is the whole test: nothing here reads a line's content, so no
 * wording, prefix or marker from the Python side is spelled on this side.
 *
 * Several fresh lines are shown together, joined by newlines, in the order the
 * dispatcher supplied them. Falling back to the serialized object when nothing
 * is fresh keeps the relay's cadence exactly as it was — one update per
 * non-terminal poll, never a silent one — which is what tells a live loop apart
 * from a stalled one.
 */
function toastMessage(update, memory) {
  const supplied = Array.isArray(update?.lines) ? update.lines : []
  const fresh = []
  for (const line of supplied) {
    if (typeof line !== "string" || !line) continue
    if (memory?.shown?.has(line)) continue
    memory?.shown?.add(line)
    fresh.push(line)
  }
  return fresh.length ? fresh.join("\n") : JSON.stringify(update)
}

/**
 * S1–S5. Runs to a terminal result and returns exactly once.
 *
 * `execute` is not resumable and hands back no continuation token; re-attachment
 * is a fresh tool call with `mode: "status"`. Nothing is returned mid-run — the
 * alternative design (return as soon as the launch detaches, hand the model a
 * run id, make it poll) would put a bare run id in front of the model as an
 * intermediate result and make every long run cost a round of turns.
 *
 * What `--detach` buys, given the tool blocks anyway, is **durability, not
 * latency**: the worker is a daemon rather than this process's child, so a
 * crashed session or an abandoned poll leaves the run alive and re-attachable
 * instead of killing a paid fan-out mid-flight.
 */
async function runWorkflow({
  row, values, surface, ctx, client, env, uv, anchor,
  relayState = relayMemory(),
}) {
  const signal = ctx?.abort
  const codes = surface.dispatcher.exit_codes
  const banner = surface.dispatcher.banner
  // `timeoutMs` is optional and only the poll loop passes one: what it passes
  // is the *remaining* poll budget, so the child cannot outlive the bound the
  // loop is keeping. `runChild` turns it into the same TERM-then-KILL
  // escalation to the whole process group that the abort dispatch gets.
  const spawnOne = (dispatcherArgv, timeoutMs) =>
    runChild({
      uv,
      argv: composeSpawnArgv(uv, dispatcherArgv, surface, env),
      cwd: anchor,
      env,
      signal,
      marker: surface.dispatcher.result_marker,
      banner: surface.dispatcher.banner.fields,
      timeoutMs,
    })
  // The abort dispatch runs **without** the cancellation signal, and that is the
  // whole point of it: the signal is what triggered the abort, so passing it
  // would kill the one child whose job is to stop the detached worker — leaving
  // a paid fan-out running with nobody reading it, which is exactly the outcome
  // "aborted, not abandoned" exists to prevent. It is bounded by its own timeout
  // instead.
  const spawnUncancellable = (dispatcherArgv) =>
    runChild({
      uv,
      argv: composeSpawnArgv(uv, dispatcherArgv, surface, env),
      cwd: anchor,
      env,
      marker: surface.dispatcher.result_marker,
      banner: surface.dispatcher.banner.fields,
      timeoutMs: ABORT_DISPATCH_TIMEOUT_MS,
    })

  // The run this call is watching *right now*, the call whose plan names it,
  // and whether this call is what claimed it.
  //
  // All four are per-cycle rather than per-call, because a mechanical
  // continuation of a detach-capable mode is composed with `--detach` exactly
  // as the first launch is: it claims a **new** run, and from that moment the
  // run a cancellation must abort, and the run the report must name, is the new
  // one. Pinning them to the launch would abort a run that already finished
  // while the live re-aggregate ran on unattended.
  //
  // `logPath` comes from the banner's log line, and from a re-attach's too:
  // `render_run_text` prints `log: <path>` in the same shape the banner does,
  // so the one parse already running serves both. Best-effort either way — a
  // missing log path costs a line of the report, never a poll or an abort.
  let claimed = false
  let runId = null
  let logPath = null
  let watched = values
  let polls = 0
  // The mid-run relay, closed over exactly the four fields above so a caller
  // cannot hand it a run id that disagrees with the one this call is watching.
  // `child` is whichever dispatcher child just answered; a launch carries no
  // `progress` (its result line is a handshake) and a poll of a run that
  // predates the tracker carries none either, so the key is read optionally and
  // a missing payload relays as `null` rather than throwing inside `execute`.
  // It still relays: the identifiers and the poll count are themselves the
  // signal that the loop is alive, and dropping the update would make the
  // cadence depend on the run's age.
  //
  // `lines` is the same payload's own rendered text, read the same optional
  // way and for the same reason — a dispatcher that predates the key, or a
  // phase with no per-model map, supplies none. It is what the toast displays,
  // and it exists because a toast's `message` is a string while this file may
  // not author one: the sentence is composed where the numbers are, by the
  // renderers that also write the human `status` view, and travels ready to
  // show. It rides inside the update rather than beside it so the redaction
  // below covers it too — a provider's own error sentence reaches
  // `just finished:` exactly as it reaches `models[*].error`.
  //
  // The update goes out through the same redaction the terminal result
  // applies, and for the same reason: `progress` is free text other writers
  // composed — a provider's own sentence in `models[*].error`, mirrored by the
  // executor — so a runnable dispatcher command can reach this object exactly
  // as it reaches `child.result`. Redacting on one exit and not the other
  // would enforce the no-runnable-string rule for the model reading the result
  // and not for the model reading the relay. No `next` string is threaded in:
  // the relay names no field a pointer could point at, so the shape pass is
  // the whole of it.
  const relay = (child) =>
    relayProgress(
      ctx,
      client,
      redactCommand(
        {
          progress: child?.result?.progress ?? null,
          lines: child?.result?.progress_lines ?? null,
          run_id: runId,
          log: logPath,
          polls,
        },
        null,
        surface,
      ),
      relayState,
      surface.tool_name,
    )

  // Every internally-composed call — the poll, the abort — goes through the
  // same composition as the launch, so it carries the same appended flags and
  // the same argv rules. Never a shell string, and never a second code path.
  // The plan defaults to the *watched* call's rather than the original's: once
  // a continuation has claimed a run, that continuation's plan is the one
  // naming it.
  const composeStatus = (extraFlags, plan = watched.plan) => {
    const statusRow = resolveMode(surface.dispatcher.poll[0], surface)
    return composeDispatcherArgv(
      statusRow,
      { word: statusRow.name, plan, question: "", extraFlags },
      surface,
    )
  }

  // The exit taken when this call stops before the run does.
  //
  // A run this call **claimed** is aborted, not abandoned. A run it merely
  // re-attached to is not this call's to end — abandoning a watch is not a
  // request to stop the work, and the run was already running with nobody
  // reading it before this call began — so that one is reported, with its id,
  // its log and the abort command, and left alive for a human to decide.
  const stopReport = (reason) =>
    claimed
      ? abortAndReport({
          reason, runId, logPath, anchor, surface, env, uv, values: watched,
          spawnOne: spawnUncancellable,
          compose: composeStatus,
        })
      : orphanReport({
          reason, runId, logPath, anchor, surface, values: watched,
        })

  const budgetMs = pollBudgetMs(env)
  // One deadline for the whole call, never one per cycle: a continuation chain
  // that restarted the clock each time would silently multiply the runaway
  // backstop by the continuation cap.
  const deadline = Date.now() + budgetMs
  const budgetExhausted =
    `the poll budget of ${Math.round(budgetMs / 1000)}s was exhausted`

  /**
   * S1–S3 for one dispatcher call: launch, claim whatever it detached, poll it
   * to a terminal result.
   *
   * **Every** launch goes through here — the caller's, and each mechanical
   * continuation's. A continuation is re-composed by the same
   * `composeDispatcherArgv`, so a detach-capable mode gets `--detach` appended
   * exactly as the first launch does; awaiting only its launcher and treating
   * that handshake as the answer would report `state: done` for a re-aggregate
   * that has not started work, and leave its worker running with nobody reading
   * it — the "abandoned, not aborted" outcome this whole path exists to
   * prevent.
   *
   * Answers a discriminated outcome rather than a bare result:
   *   `terminal`    — a terminal payload the caller may keep or continue from
   *   `return`      — the whole call ends, with this result
   *   `spawn-error` — uv never started, and what that costs is the caller's
   *                   decision rather than this function's
   */
  const cycle = async (cycleRow, cycleValues) => {
    // S1 — the launch child.
    const launchArgv = composeDispatcherArgv(cycleRow, cycleValues, surface)
    const launch = await spawnOne(launchArgv)
    if (launch.spawnError) {
      return {
        kind: "spawn-error",
        result: errorResult(
          launch,
          `could not start uv: ${launch.spawnError}`,
          anchor,
          { surface },
        ),
      }
    }
    // Fix round 27: the plan the dispatcher read, before any poll, abort or
    // option names it.
    planFromRoot(launch.result, cycleValues, anchor, surface)
    // The run in view when this cycle began, which is what a launch that
    // claimed nothing still has to report: on the first cycle that is no run at
    // all, and on a continuation it is the finished run that spawned it.
    const foreground = {
      anchor, surface, runId, logPath, polls,
      values: cycleValues,
    }

    if (cycleRow.supports_detach) {
      const launched = launch.banner[banner.run_id_field] || null
      if (launched) {
        runId = launched
        logPath = launch.banner[banner.log_field] || null
        watched = cycleValues
        claimed = true
        // The banner is the first moment this call knows anything, and the
        // first poll below can take a whole `--wait` to come back, so this seam
        // is the difference between "it started, here is the log" arriving now
        // and arriving a poll from now. Deliberately after the assignments and
        // before the failure checks: a launch that claimed a run and then died
        // still started one, and the update naming it is what a human needs to
        // go read the log. Guarded on the cancellation signal rather than
        // moved below those checks, because the two exits are not the same
        // exit: a launch that *died* owes the reader the identifiers, while a
        // launch the caller *cancelled* owes them nothing — an update emitted
        // once the signal has fired is an update after cancellation, and this
        // loop goes silent from that moment.
        if (!signal?.aborted) relay(launch)
      }
      if (launch.aborted || launch.code !== codes.ok) {
        // `detach.launch()` daemonises the worker and prints the banner *before*
        // the launcher exits, so a launcher that was killed or that failed on its
        // way out can still have left a live run behind it. When the banner named
        // one, it is aborted here rather than left to spend provider budget with
        // nobody reading it: returning an error while the fan-out runs on is the
        // "abandoned, not aborted" outcome this whole path exists to prevent, and
        // a cancelled launch is no exception to it.
        if (launched) {
          return {
            kind: "return",
            result: stopReport(
              launch.aborted
                ? "the call was cancelled while the launch was still running"
                : `the launch exited ${launch.code} after reporting a run`,
            ),
          }
        }
        // No banner, so nothing was claimed: a worker dying inside the launch
        // window is exit 3 with the log tail already attached, and a launcher
        // killed before it forked left nothing to stop.
        return { kind: "return", result: terminalResult(launch, foreground) }
      }
      if (!launched) {
        return {
          kind: "return",
          result: errorResult(
            launch,
            "the launch reported no run_id on its banner, so the run cannot be " +
              "polled or aborted safely. Refusing to adopt whatever run is newest.",
            anchor,
            // The banner it *did* print carries runnable commands; `surface` is
            // what lets them be redacted out of the reported output.
            { surface },
          ),
        }
      }
    } else {
      // A status re-attach and an init are ordinary foreground children: the mode
      // declares supports_detach false and the dispatcher refuses --detach on it
      // with exit 3. One of them is not necessarily *terminal*, though — a
      // re-attach to a run still in flight exits 2, and returning that straight
      // out ends a call that "blocks to a terminal result exactly like any other
      // call" and reports it as `ok: true`, which tells a model an unfinished run
      // succeeded. So a caller-pinned view of a live run enters the same bounded
      // loop below; everything else — an init, a snapshot naming no run, an
      // abort, a view of a run already terminal — returns here as before.
      const adopted = reattachedRunId(launch, cycleValues, surface)
      if (!adopted) {
        return { kind: "return", result: terminalResult(launch, foreground) }
      }
      runId = adopted
      logPath = launch.banner[banner.log_field] || null
      watched = cycleValues
      // Adopted, never claimed: this call did not start that run, so abandoning
      // the watch is not a request to stop it.
      claimed = false
      // No relay seam here, and that is a gap named rather than a decision
      // defended. The detach branch above relays its launch child because the
      // banner is the first moment it knows anything; this branch does not, so
      // a `mode: "status"` re-attach whose caller composed `--wait` drops
      // whatever that wait observed: the dispatcher rendered it onto *this*
      // child's result line (`--wait` is the only invocation that can honestly
      // write `just finished:`) and the loop below relays only the polls it
      // issues itself. The shape is pre-existing; what it costs changed, since
      // it used to drop a JSON blob nobody would have read and now drops a
      // composed sentence. Bounded, though: a run still in flight loses one
      // `just finished:` line and nothing lasting — the first poll this loop
      // issues carries the roster and every later piece of news.
    }

    // S2/S3 — the poll loop. One blocking child per iteration; the JS side
    // invents no cadence of its own, because `--wait` already polls internally
    // and returns exit 2 while non-terminal.
    //
    // `--wait` is a store_true and the bare flag is still all that is passed —
    // but what the flag *means* is no longer one fixed default. The dispatcher
    // resolves the duration from the project's configured poll cadence and then
    // clamps the resolved value to this host's documented foreground cap less a
    // margin, and that clamp fires here: `composeStatus` appends the same baked
    // flags every dispatch gets, so the `--host` it reads is always on the
    // argv. A configured 300 s cadence therefore polls at the clamped figure on
    // this route.
    //
    // That is accepted rather than worked around, and for a reason that is this
    // loop's own rather than the shell cap's. The poll duration is the relay's
    // granularity: the seam below fires once per non-terminal poll, so a long
    // cadence is a run that reports in rarely. It is *only* the relay that
    // pays — cancellation and the call budget do not reach here to wait out an
    // iteration: `spawnOne` threads the abort signal into `runChild`, whose
    // `onAbort` kills the child's process group the moment it fires, and the
    // poll is spawned with the *remaining* budget as its timeout. The clamp
    // bounds exactly that, so what it costs is a configured cadence's length
    // and what it buys is a live relay. Composing an explicit
    // `--timeout` here would buy none of that and would move a duration into
    // this file, which is the one thing it may not hold: the number stays a
    // fact Python owns, and the shipped cadence is unchanged either way.
    const pollArgv = composeStatus([
      ...surface.dispatcher.poll.slice(1),
      surface.dispatcher.run_id_option,
      runId,
    ])
    let poll = null
    while (true) {
      if (signal?.aborted) {
        return { kind: "return", result: stopReport("the call was cancelled") }
      }
      const remaining = deadline - Date.now()
      if (remaining <= 0) {
        return { kind: "return", result: stopReport(budgetExhausted) }
      }
      // The check above bounds the *loop*; this argument is what bounds the
      // call. Awaiting an unbounded child and only testing the clock before the
      // next one makes the budget a bound on iterations, not on time: a `uv`
      // that never starts moving, or a status child whose grandchild still
      // holds stdout open, blocks here past six hours and forever, with the
      // deadline it sailed past never reached again. `runChild` escalates
      // SIGTERM to SIGKILL over the whole process group, so the await always
      // comes back.
      poll = await spawnOne(pollArgv, remaining)
      polls += 1
      if (poll.spawnError) {
        return {
          kind: "return",
          result: errorResult(poll, `could not poll: ${poll.spawnError}`, anchor, {
            runId, logPath, surface, values: watched,
          }),
        }
      }
      if (poll.aborted) {
        return { kind: "return", result: stopReport("the call was cancelled") }
      }
      if (poll.timedOut) {
        // Killed on the budget, so it has no exit code of its own — and null is
        // not `nothing_to_do`, so falling through would break out of the loop
        // and report the corpse as the run's terminal answer. It is the budget
        // running out, and it takes the budget's exit: the claimed run is
        // aborted, a merely watched one is reported alive.
        return { kind: "return", result: stopReport(budgetExhausted) }
      }
      // 2 is "non-terminal" — and also, deliberately, "no run in view at all",
      // because 0 would read as finished. The --run-id pin is what distinguishes
      // them; a loop that adopted the newest run would poll a stranger
      // *successfully*, which is the worse failure.
      if (poll.code !== codes.nothing_to_do) break
      // The per-iteration seam, and deliberately *below* the break. The
      // terminal poll's numbers ride out in the result this call is about to
      // return, so relaying it here too would emit the same state twice and put
      // an update *after* the answer. Every non-terminal poll relays exactly
      // once instead: a run that polls five times yields five updates, which is
      // the only thing that tells a live loop apart from a stalled one.
      relay(poll)
    }

    const terminal = terminalResult(poll, {
      anchor, surface, runId, logPath, polls, values: watched,
    })
    // `argv` names what this cycle ran: its own launch, never the internal
    // `status … --wait --run-id` poll whose answer this is. The model was told
    // to report `argv` as "what was being run" (it is told the phase and the
    // plan since fix round 19), and on Codex racing re-run 5 the host told the
    // user the poll was "the dispatcher's final argv" (fix round 14). Set here,
    // per cycle, so a continuation names its own launch.
    terminal.argv = launch.argv
    // The launch's own result line is a handshake, not the answer: it says
    // state: done, exit: 0 and would report "done" for a fan-out that has not
    // started work. The terminal payload is the final poll's.
    terminal.launch_state = launch.result?.state ?? null
    return { kind: "terminal", terminal }
  }

  const first = await cycle(row, values)
  if (first.kind !== "terminal") return first.result
  let terminal = first.terminal

  // S4 — `next`, and D3a's split. A **mechanical** continuation (a re-aggregate
  // or a consolidate: the same phase finishing itself off) is parsed back into
  // (mode, plan, flags) and re-composed as argv here, through the same
  // validator and the same appended flags — and then driven through the whole
  // launch-and-poll machine above, because that re-composition is what appends
  // `--detach` to it. Everything else is returned as a tool-call object and
  // never re-executed. The cap is what stops a chain.
  const cap = nextContinuationCap(env)
  let followed = 0
  while (
    terminal.ok &&
    followed < cap &&
    isMechanical(terminal.next, surface) &&
    !signal?.aborted
  ) {
    const args = terminal.next.args
    // "The same validator" is meant literally, and it is this call rather than
    // the composition below that makes it true. `parseDispatcherCommand` has
    // already dropped the flags this tool composes itself, so nothing the
    // dispatcher emits today fails here — but arity, a `-`-prefixed
    // positional and the extra_flags element types are `validateArgs`'s rules
    // and **nothing** else enforces them, so a `next` whose shape ever changed
    // would otherwise reach `spawn` unchecked on a path no caller can see.
    // It resolves the mode row on the way through, which is why no separate
    // `resolveMode` runs here: an unknown mode is one of its refusals.
    const checked = validateArgs(args, surface)
    if (!checked.ok) {
      // Refused — and not handed back as a runnable continuation either. A
      // tool call this same validator would reject at the door is not one to
      // invite the model to send, so it is reported exactly as an unparseable
      // `next` is. That also makes `isMechanical` answer false below, so the
      // cap check does not mistake a refusal for a chain cut short.
      terminal.next = {
        unparseable: true,
        note:
          "The dispatcher returned a follow-up this plugin refused to run: " +
          `${checked.error} It is reported rather than executed. Do not run ` +
          "it in a shell.",
      }
      break
    }
    followed += 1
    const outcome = await cycle(checked.mode, {
      word: args.mode,
      plan: args.plan,
      // The dispatcher's `next` names the plan absolutely; when that is the
      // plan this call was typed with, the continuation keeps the user's
      // spelling so its own next-phase command does too.
      planArg: samePlanFile(args.plan, values.plan) ? values.planArg : undefined,
      question: args.question,
      extraFlags: args.extra_flags,
    })
    // uv failing to start is no statement about the phase that just finished:
    // the standing terminal is still the answer, and its `next` still spells the
    // continuation as a tool call the model can send back itself.
    if (outcome.kind === "spawn-error") break
    if (outcome.kind === "return") return outcome.result
    terminal = outcome.terminal
    terminal.continuations = followed
  }
  if (followed >= cap && isMechanical(terminal.next, surface)) {
    // Exhausting the cap takes the same path as exhausting the budget: stop the
    // run this call claimed rather than leave a worker with nobody reading it,
    // and report one it only re-attached to rather than end work it did not
    // start.
    return stopReport(`the continuation cap of ${cap} was reached`)
  }
  return terminal
}

/**
 * Whether a parsed `next` is a continuation `execute` may follow itself.
 *
 * Only the markers the generated surface names, and only a mode the table
 * knows. Anything else — including anything unparseable — is reported and
 * handed back, never run.
 */
function isMechanical(next, surface) {
  if (!next || next.unparseable || !next.args) return false
  const markers = surface.dispatcher.mechanical_continuations ?? []
  return next.args.extra_flags.some((token) =>
    markers.includes(token.split("=", 1)[0]),
  )
}

/**
 * The run a foreground call re-attached to, when that run is still going.
 *
 * Exit 2 means two different things on a status view — "still running" and "no
 * such run in view at all" — and what separates them is not the exit code but
 * the payload's `next`: `run_status.main` records a follow-up **only** when the
 * view it rendered was non-empty and non-terminal, and that follow-up is the
 * status command for the run it saw, carrying that run's full id. A `next` that
 * parses back into a poll-mode call with a run-id pin is therefore the
 * dispatcher itself saying "this run is alive; poll it again with this".
 *
 * Two conditions, and the first is what keeps this from adopting a stranger.
 * The **caller** must have named a run: with no pin, a plan holding several
 * live runs records a follow-up for whichever phase dir sorted first, and
 * polling — then, at the budget, aborting — a run nobody asked about is exactly
 * the adoption the launch path refuses. Given a pin, the id returned is still
 * the dispatcher's rather than the caller's, because `--run-id` accepts a
 * unique *prefix* and every later poll should carry the whole thing.
 *
 * Everything else answers null: an init, a snapshot naming no run, an abort or
 * a sweep (neither records a follow-up at all), a view of a run already
 * terminal, a view naming no run.
 */
function reattachedRunId(child, values, surface) {
  if (child.code !== surface.dispatcher.exit_codes.nothing_to_do) return null
  if (!pinnedRunId(values.extraFlags, surface)) return null
  const next = child.result?.next
  if (typeof next !== "string" || !next) return null
  const parsed = parseDispatcherCommand(next, surface)
  const pollMode = resolveMode(surface.dispatcher.poll[0], surface)
  if (!parsed || !pollMode || parsed.mode !== pollMode.name) return null
  return pinnedRunId(parsed.extra_flags, surface)
}

/**
 * The value bound to the run-id option in a token list, or null.
 *
 * `flagMatches` rather than an equality test, for the same reason the reserved
 * check uses it: no parser in the skill passes `allow_abbrev=False`, so a
 * caller's abbreviation binds to the option just as the full spelling does and
 * a scan that missed it would read a pinned call as an unpinned one.
 */
function pinnedRunId(tokens, surface) {
  const name = surface.dispatcher.run_id_option
  const exacts = surface.reserved_flags.exact_options
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]
    const cut = token.indexOf("=")
    const head = cut < 0 ? token : token.slice(0, cut)
    if (!flagMatches(head, name, exacts)) continue
    if (cut >= 0) return token.slice(cut + 1) || null
    const value = tokens[index + 1]
    return value !== undefined && !value.startsWith("-") ? value : null
  }
  return null
}

/**
 * A claimed detached run is aborted, not abandoned.
 *
 * `status <plan> --abort --run-id <id>` is the documented *only* kill path: it
 * runs an identity probe, then a process-group SIGTERM, then SIGKILL after a
 * grace. This never signals the worker pid itself — the worker is not our child,
 * the pid may have been recycled, and that identity probe is exactly the guard
 * against killing a stranger. Best-effort and bounded: if it fails, the result
 * carries the run id, the log path and the literal abort command so a human can
 * finish. An orphan is reported, never hidden.
 */
async function abortAndReport({
  reason, runId, logPath, anchor, surface, values, spawnOne, compose,
}) {
  const abortArgv = compose([
    ...surface.dispatcher.abort.slice(1),
    surface.dispatcher.run_id_option,
    runId,
  ])
  const codes = surface.dispatcher.exit_codes
  // What `abort_dispatched` claims is that the run **stopped**, not that a child
  // started. `status --abort` exits `ok` when it terminated a live process group
  // and `nothing_to_do` when there was nothing left to stop — already terminal,
  // or the pid was gone and the record was closed out — and both of those mean
  // the fan-out is no longer spending provider budget. `needs_human` is the
  // survivor case (the pid outlived the SIGKILL window, or the identity probe
  // was inconclusive and the abort refused to guess) and `error` is a failure;
  // neither is a statement that the run ended.
  const stopped = (code) => code === codes.ok || code === codes.nothing_to_do
  let dispatched = false
  let dispatchTimedOut = false
  try {
    const out = await spawnOne(abortArgv)
    dispatchTimedOut = !!out.timedOut
    // The MULTI_LLM_RESULT line is required as well as the code, because an exit
    // status alone is not the dispatcher's own verdict. This dispatch is bounded
    // by `ABORT_DISPATCH_TIMEOUT_MS`, and a child killed on that timeout — or one
    // that fell over before reaching its terminal line — can close without ever
    // saying what happened to the run. Reporting `abort_dispatched: true` for one
    // of those tells the reader a paid fan-out was stopped when nothing confirmed
    // it: the same "abandoned, not aborted" outcome wearing the opposite label,
    // and worse than the honest `NOT confirmed` because it stops them looking.
    dispatched =
      !out.spawnError &&
      stopped(out.code) &&
      !!out.result &&
      stopped(out.result.exit)
  } catch {
    dispatched = false
  }
  return {
    state: "needs-human",
    ok: false,
    reason,
    run_id: runId,
    log: logPath,
    abort_dispatched: dispatched,
    abort_command: {
      tool: surface.tool_name,
      args: abortArgs(values, runId, surface),
    },
    anchor,
    message:
      `${reason}. The detached run ${runId} was ${
        dispatched ? "confirmed aborted" : "NOT confirmed aborted"
      }. ` +
      // Why nothing confirmed it, when the reason is the bound rather than the
      // dispatcher: the abort child hit `ABORT_DISPATCH_TIMEOUT_MS` and was
      // killed, so its silence says nothing about the run's state.
      (dispatchTimedOut ? "The abort dispatch timed out and was killed. " : "") +
      (logPath ? `Log: ${logPath}. ` : "") +
      `If it is still alive, stop it with the abort tool call in this result.`,
  }
}

/**
 * A run this call watched but never claimed: reported, and left alone.
 *
 * The mirror of `abortAndReport` for a re-attach. Abandoning a *watch* is not a
 * request to end the work — the run was already going with nobody reading it
 * before this call began, and returning it to that state is the conservative
 * answer, where killing a paid fan-out some other call launched is not. So
 * nothing is dispatched, and the result still carries the run id, the log path
 * and the abort command spelled as a tool call, because a run left alive has to
 * be a run the reader can find and stop.
 */
function orphanReport({ reason, runId, logPath, anchor, surface, values }) {
  return {
    state: "needs-human",
    ok: false,
    reason,
    run_id: runId,
    log: logPath,
    abort_dispatched: false,
    abort_command: {
      tool: surface.tool_name,
      args: abortArgs(values, runId, surface),
    },
    anchor,
    message:
      `${reason}. The run ${runId} was not started by this call, so it was ` +
      `left running rather than stopped. ` +
      (logPath ? `Log: ${logPath}. ` : "") +
      `Watch it again by sending this call again, or stop it with the abort ` +
      `tool call in this result.`,
  }
}

/**
 * The abort spelled as a tool call the model could send back.
 *
 * The **explicit** `status … --abort` spelling, taken from generated data
 * rather than the `abort` alias: both are legal and they are the same run, but
 * the alias is a mode word and this file holds none. Deriving it also means a
 * renamed alias cannot silently turn this into a call the dispatcher rejects.
 */
function abortArgs(values, runId, surface) {
  const [word, ...flags] = surface.dispatcher.abort
  return {
    mode: word,
    plan: values.plan,
    question: "",
    extra_flags: [...flags, surface.dispatcher.run_id_option, runId],
  }
}

/** S4 — the terminal result, and what happens to `next`. */
function terminalResult(child, context) {
  const {
    anchor, surface, runId = null, logPath = null, polls = 0,
    values = null,
  } = context
  const codes = surface.dispatcher.exit_codes
  const base = {
    exit: child.code,
    anchor,
    argv: child.argv,
    run_id: runId,
    log: logPath,
    polls,
    stdout_truncated: child.truncated || undefined,
  }

  if (child.code === codes.needs_human) {
    // Exit 3 is not an error but a needs-human result. Under the baked --yes
    // nothing asks a question on the default path, but every fail-closed site
    // exits 3 — and when a numbered option block *is* present it is returned
    // verbatim, never synthesized, because it went into this process's captured
    // stdout rather than into the transcript.
    //
    // Verbatim stops at the runnable strings inside it, and that is D3c's
    // fourth part rather than an exception to its first: a numbered option's
    // tail *is* a full dispatcher command, and a block that says "copy exactly
    // one option's command" beside a message that says "do not re-run the
    // command yourself" is an invitation to paste one into a shell. So every
    // runnable command here — an option's tail, a banner's `status:`/`abort:`
    // line, the payload's own `next` — is redacted where it stood and
    // re-spelled as a tool call by rule (a). Following still stops.
    //
    // It stops at the `MULTI_LLM_RESULT:` envelope too, which was never part of
    // the block a human is being asked to read and comes back parsed in
    // `result` instead.
    //
    // Each redacted dispatcher command is re-spelled the way this route's user
    // types it (`dispatcher.user_command`), with the plan as this call typed
    // it, rather than replaced by a removal note: the line that named the
    // retry still names it, in the only spelling a person can use. A surface
    // without the key, or a command that does not parse, keeps the note.
    const respell = respellerFor(surface, values)
    const payload = redactCommand(child.result ?? null, null, surface, respell)
    const output = humanOutput(child, null, surface, respell)
    const stderr = redactCommand(child.stderr, null, surface, respell)
    return {
      ...base,
      state: "needs-human",
      ok: false,
      result: payload,
      // Lifted off the *redacted* payload, never off `child.result`: a copy
      // taken from the raw one would put back, in its own field, the very
      // command the field beside it just removed.
      error: payload?.error ?? null,
      output,
      stderr: stderr || undefined,
      next: null, // following `next` stops here
      // The retries the child printed, then — always last, and only on this
      // branch — the one option that is a legal exit rather than a retry. One
      // key, `human`, read off the surface: a consumer branching on
      // `"tool" in option` takes the human branch, and `isMechanical` already
      // answers false for anything without `args`, so the continuation loop
      // can never auto-follow it. Exit 1 (`errorResult`) does not get it: an
      // error result with nothing scraped still reports no `options` key.
      options: [
        ...toolCallsFor(child, surface, values),
        { human: surface.needs_human_option },
      ],
      nested_exec: nestedExecReason(child, surface),
      // Composed from the surface's template and clauses, never authored
      // here: this file holds no message of its own (the same rule that keeps
      // the uv preflight text in `hosts/uv-missing.txt`).
      message: needsHumanMessage({ payload, output, stderr, values, surface }),
    }
  }

  if (child.code === codes.nothing_to_do && child.result) {
    const rawNext = child.result.next
    // Exit 2 is emphatically *not* an error on this route: it is "nothing to
    // do / already complete". In the poll loop it means "keep polling" and
    // never reaches here; on a foreground child it is a legitimate terminal
    // answer, and reporting it as an error would tell a model a completed phase
    // had failed.
    const result = redactCommand(child.result, rawNext, surface)
    return {
      ...base,
      state: child.result.state ?? "nothing-to-do",
      ok: true,
      result,
      artifacts: child.result.artifacts ?? [],
      next: nextAsToolCall(rawNext, surface),
      output: humanOutput(child, rawNext, surface, null, carriesNextStep(child)),
      ...liftedUserNextStep(result),
    }
  }

  if (child.code !== codes.ok || !child.result) {
    return errorResult(
      child,
      child.result
        ? `the dispatcher exited ${child.code}`
        : child.resultRaw
          ? `the dispatcher's ${surface.dispatcher.result_marker} line was not ` +
            `valid JSON: ${child.resultRaw}`
          : `the dispatcher exited ${child.code} with no ` +
            `${surface.dispatcher.result_marker} line`,
      anchor,
      // `values` is threaded so an exit-1 poll — `status` itself failing, which
      // is never a statement about the run — can still hand back the abort
      // command for the run that is *still alive*.
      { runId, logPath, surface, values },
    )
  }

  // `next` is never handed back as a shell command — and "never" has to mean
  // the *whole* result, not just the `next` key. The dispatcher's own stdout
  // carries the runnable string inside its MULTI_LLM_RESULT line, and the
  // parsed payload carries it again, so both are redacted on the way out.
  // Otherwise the tool hands a model a working shell command in two places
  // while carefully re-spelling it as a tool call in a third, and the model's
  // natural move is to run the one it can paste.
  const raw = child.result.next
  const result = redactCommand(child.result, raw, surface)
  return {
    ...base,
    state: child.result.state ?? "done",
    ok: true,
    result,
    artifacts: child.result.artifacts ?? [],
    next: nextAsToolCall(raw, surface),
    output: humanOutput(child, raw, surface, null, carriesNextStep(child)),
    ...liftedUserNextStep(result),
  }
}

/**
 * `{ user_next_step }` off the *redacted* payload, or `{}` when it has none.
 *
 * Exit 0 and exit 2 only: a finished phase or a terminal poll, and a phase
 * already complete. The value is the payload's own, byte for byte, the
 * completed form (`command: null`, `complete: true`) included, under the same
 * key name, so a model reading the top-level keys cannot miss it. A payload
 * without the key lifts nothing: an absent key keeps meaning "no answer", never
 * "workflow complete". It is the user's to type, never a call: `isMechanical`
 * reads `next` alone, so the continuation loop never follows it.
 *
 * Moved, not copied (fix round 26, live run 11): the section the model copies
 * was in the result twice, and the decline run printed "Workflow complete."
 * twice, plain and then fenced. One copy, in the field that owns it, as
 * `humanOutput` drops the parsed result line. `payload` is `redactCommand`'s
 * own copy, so the child's parsed result is untouched.
 */
function liftedUserNextStep(payload) {
  const step = payload?.user_next_step
  if (step === undefined || step === null) return {}
  delete payload.user_next_step
  return { user_next_step: step }
}

/**
 * The exit-3 `message`, filled from the surface's template.
 *
 * `needs_human_message` is a template with `{phase}` and `{error}` tokens plus
 * the clause alternatives they are filled with; this function only *selects*
 * a clause for each and substitutes. It authors no wording, not even a
 * fallback, so the one sentence a model reads on the path this incident turned
 * on is Python's.
 *
 * The phase word comes from the **launch**, not from the payload: the call's
 * validated mode word, resolved through the surface's mode table. A poll's
 * result line is `status`'s own and declares no phase, so on a detached
 * worker's death the payload cannot name the phase that died; `result.phase`
 * is consulted only to *confirm* what the launch says and never to overrule
 * it. "Did not complete" is claimed only when the result line's `state` is the
 * needs-human one — and a completed `review-code --fail-on-high`, which
 * carries that same state, requires both the launch flag and reported HIGH
 * findings before it gets the completed-review clause instead.
 *
 * With no result line at all — a child that died before it emitted one, or a
 * line that was not JSON — nothing names the phase, and the last non-blank
 * line of the *redacted* output (else stderr) stands in for the error line.
 * Never a line off `child.stdout`: a tail taken from the raw stream would put
 * a runnable command back into the message beside the fields that just
 * removed it.
 */
function needsHumanMessage({ payload, output, stderr, values, surface }) {
  const text = surface.needs_human_message
  const fill = (clause, vars) =>
    clause.replace(/\{(phase|error)\}/g, (_, key) => vars[key] ?? "")

  const row = values?.word ? resolveMode(values.word, surface) : null
  const launched = row?.phase ?? null
  const gate = surface.dispatcher.review_gate
  const held =
    row?.name === gate.mode && (values?.extraFlags ?? []).includes(gate.flag) &&
    Number.isInteger(payload?.high_priority_groups?.count) &&
    payload.high_priority_groups.count > 0

  let phase
  if (!payload) {
    phase = text.phase_unknown
  } else if (launched && held) {
    phase = fill(text.review_completed, { phase: launched })
  } else if (launched) {
    // Only the needs-human word is "did not complete"; every other word the
    // surface names — `partial` included, a run that finished — is not.
    const incomplete = payload.state === surface.dispatcher.states.needs_human
    phase = fill(incomplete ? text.phase_incomplete : text.phase_known, {
      phase: launched,
    })
  } else {
    phase = text.phase_unnamed
  }

  let line = null
  if (payload) {
    line = typeof payload.error === "string" && payload.error.trim()
      ? payload.error.trim()
      : null
  } else {
    line = lastNonBlankLine(output) ?? lastNonBlankLine(stderr)
  }
  const error = line
    ? fill(text.error_known, { error: `"${line}"` })
    : text.error_unknown

  return fill(text.template, { phase, error })
}

function lastNonBlankLine(text) {
  if (typeof text !== "string") return null
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean)
  return lines.length ? lines[lines.length - 1] : null
}

/** Where a redacted `next` went, and where any other redacted command went. */
const NEXT_MARKER = "<see the `next` field of this result for the tool call>"
const RUNNABLE_MARKER =
  "<a runnable command was removed; commands come back only as tool calls>"

/**
 * Replace every runnable dispatcher command with a pointer.
 *
 * Applied to the parsed payload, to the captured stdout and to the captured
 * stderr alike, on **every** branch, because a command a model can paste is a
 * command a model may paste — and a shell invocation reaches none of the guards
 * in this file. Two passes, because a command arrives here two ways: as the
 * exact `next` string, whose pointer can name the field that replaced it, and
 * as whatever else the child printed — a needs-human option block's tails, a
 * launch banner's `status:`/`abort:` lines — which is matched by shape.
 *
 * The walk is structural rather than a scrub of `JSON.stringify` output: each
 * command is redacted in the string that actually holds it, so no escaping of
 * the surrounding encoding can hide a copy from the pattern or be broken by the
 * replacement.
 */
function redactCommand(value, command, surface = null, respell = null) {
  return scrubValue(
    value,
    commandForms(command),
    surface ? runnablePattern(surface) : null,
    respell,
  )
}

/**
 * Every spelling of one command a captured string may hold.
 *
 * `next` reaches this file **decoded** — `JSON.parse` produced it — but the text
 * it was parsed out of holds it *escaped*, and so does any other JSON the child
 * printed: a worker's own `MULTI_LLM_RESULT:` line echoed inside a fan-out's
 * output, a payload quoted back in an error. Only `"` and `\` differ between
 * the two spellings, which is exactly the Windows-path and quoted-argument case
 * — `C:\skill\multi_llm.py` is written `C:\\skill\\multi_llm.py` there — and
 * splitting on the decoded form alone leaves that copy whole and runnable. The
 * shape pattern is no backstop for it either: an escaped `\"` ends the pattern's
 * run, so a command whose script path sits behind one is never matched at all.
 *
 * Escaped form first: it is the longer of the two, so replacing it first is what
 * keeps a shorter overlapping match from eating half of it.
 */
function commandForms(command) {
  if (typeof command !== "string" || !command) return []
  const escaped = JSON.stringify(command).slice(1, -1)
  return escaped === command ? [command] : [escaped, command]
}

/**
 * The child's stdout as a reader reads it: redacted, and with the dispatcher's
 * own envelope line dropped.
 *
 * The `MULTI_LLM_RESULT:` line is machine plumbing, not part of what the child
 * rendered, and it is already handed back parsed in `result` — so leaving it in
 * `output` returns the same payload twice, the second copy being a JSON blob
 * whose `next` a model reads as a command that merely happens to be redacted.
 * One copy, in the field that owns it.
 *
 * It is dropped only when it *parsed*. An unparseable line is the diagnosis
 * rather than duplication: the parse failed, so nothing else in the result
 * carries it, and dropping it would delete the only evidence of what the child
 * actually printed.
 *
 * With `carried` (the result carries the lifted `user_next_step`), the block's
 * lines that step says itself are dropped too (fix round 27, the Fable review
 * of round 26: the manual note and the optional whys reached the model twice,
 * in `output` and in the step's `markdown`): a line that is one of the
 * surface's `dispatcher.next_step_markers`, or starts with one and a space. A
 * `[RUNNER_WARNING]` line, which the step does not carry, stays. The codex
 * server's `human_output` is its port.
 */
function humanOutput(child, command, surface, respell = null, carried = false) {
  const text = redactCommand(child?.stdout, command, surface, respell)
  const marker = surface?.dispatcher?.result_marker
  if (typeof text !== "string" || !text || !marker || !child?.result) {
    return text
  }
  const said = carried ? surface?.dispatcher?.next_step_markers ?? [] : []
  return text
    .split("\n")
    .filter((line) => !line.trim().startsWith(marker))
    .filter(
      (line) =>
        !said.some((block) => line === block || line.startsWith(block + " ")),
    )
    .join("\n")
    .trimEnd()
}

/** Whether `liftedUserNextStep` lifts a step off this child's parsed result. */
function carriesNextStep(child) {
  const step = child?.result?.user_next_step
  return step !== undefined && step !== null
}

function scrubValue(value, commands, pattern, respell = null) {
  if (typeof value === "string") {
    let once = value
    for (const form of commands) once = once.split(form).join(NEXT_MARKER)
    if (!pattern) return once
    return once.replace(
      pattern,
      (match, offset, whole) =>
        (respell && !cutMidToken(match, offset, whole) && respell(match)) ||
        RUNNABLE_MARKER,
    )
  }
  if (Array.isArray(value)) {
    return value.map((item) => scrubValue(item, commands, pattern, respell))
  }
  if (value && typeof value === "object") {
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      out[key] = scrubValue(item, commands, pattern, respell)
    }
    return out
  }
  return value
}

/**
 * The function that re-spells one redacted dispatcher command for the user,
 * or null when the surface carries no `dispatcher.user_command`.
 *
 * The command is parsed into tool-call arguments first (`nextAsToolCall`, so
 * the flags this tool composes are already gone), its plan put back the way
 * this call typed it, and the result spelled through the route's own template.
 * A command that does not parse answers null, and the caller keeps the note.
 */
function respellerFor(surface, values) {
  if (!surface?.dispatcher?.user_command) return null
  return (command) => {
    const call = nextAsToolCall(command, surface)
    if (!call?.args) return null
    return userSpelling(
      { ...call.args, plan: typedPlan(call.args.plan, values) },
      surface,
    )
  }
}

/** *plan* as this call typed it, when it names the call's own plan file. */
function typedPlan(plan, values) {
  return values?.planArg && plan && samePlanFile(plan, values.plan)
    ? values.planArg
    : plan
}

/** The result keys that hand the model a tool call to send back. */
const HANDED_BACK_CALLS = ["next", "abort_command"]

/**
 * Every tool call *outcome* hands back names the plan as this call typed it.
 *
 * This plugin names the plan absolutely inside its own loop (the polls, the
 * abort dispatch, a mechanical continuation), and those spellings used to leave
 * in a returned `next` and `abort_command`. The call the model sent back then
 * *typed* the absolute path, so its resume command and its re-spelled output
 * lines carried it too (racing re-run 3, 2026-10-05, on the codex route). The
 * call runs in the same anchor, so the typed spelling names the same file.
 * The port of `hand_back_typed`.
 */
function handBackTyped(outcome, values, surface) {
  for (const key of HANDED_BACK_CALLS) {
    const call = outcome?.[key]
    if (!call?.args || typeof call.args.plan !== "string") continue
    const plan = typedPlan(call.args.plan, values)
    if (plan === call.args.plan) continue
    const args = { ...call.args, plan }
    outcome[key] = { ...call, args }
    if (typeof call.human === "string") {
      outcome[key].human = describeCall(args, surface)
    }
  }
}

/**
 * Whether two plan paths name one file, symlinks followed on both sides.
 *
 * The call's plan is made absolute without following symlinks
 * (`normalizePlan`) and the dispatcher prints some paths through
 * `Path.resolve()`, so under a symlinked workspace the same file came back
 * spelled two ways and the re-spelled commands and options carried the
 * absolute real path instead of the plan as typed (2026-10-05). Two absolute
 * paths are compared by their real paths; anything else only by equality.
 */
function samePlanFile(left, right) {
  if (left === right) return true
  if (!left || !right || !path.isAbsolute(left) || !path.isAbsolute(right)) {
    return false
  }
  return realPathOf(left) === realPathOf(right)
}

/** `realPath`, else the real directory plus the name, else as given. */
function realPathOf(target) {
  try {
    return realPath(target)
  } catch {
    try {
      return path.join(realPath(path.dirname(target)), path.basename(target))
    } catch {
      return path.resolve(target)
    }
  }
}

/**
 * The real path the dispatcher names a file by. Throws as `fs.realpathSync`.
 *
 * On Windows that is `fs.realpathSync.native`: the dispatcher prints paths
 * through `Path.resolve()` and pins its root with `os.path.realpath`, which
 * there answer the file's on-disk case, its long name for an 8.3 one and the
 * real drive behind a `subst`; the JS walk only follows links. So an anchor
 * spelled `c:\users\me\proj` (a shell `cd` keeps the case it was typed in)
 * never equalled the dispatcher's `C:\Users\Me\proj`, and every command
 * handed back named the plan absolutely. The codex port compares with
 * `os.path.realpath` already. Elsewhere, and when the native call fails (a
 * volume it cannot query), the JS walk as before.
 */
function realPath(target) {
  if (process.platform === "win32") {
    try {
      return fs.realpathSync.native(target)
    } catch {
      // The JS walk below, which also throws for a path that does not exist.
    }
  }
  return fs.realpathSync(target)
}

/**
 * Tool-call arguments spelled the way this route's user types them, or null.
 *
 * The template and its slot are `dispatcher.user_command`, the mode word is
 * the row's `user_flag`, and every value is quoted by `quoteForReader`: the
 * pieces `next_phase.resume_command` builds the same string from, which the
 * suite holds equal. This file spells none of them.
 */
function userSpelling(args, surface) {
  const spell = surface?.dispatcher?.user_command
  const row = resolveMode(args.mode, surface)
  if (!spell || typeof spell.template !== "string" || !row?.user_flag) return null
  const words = [row.user_flag]
  if (args.plan) words.push(quoteForReader(args.plan))
  if (args.question) words.push(quoteForReader(args.question))
  for (const flag of args.extra_flags ?? []) words.push(quoteForReader(flag))
  return spell.template.split(spell.args_slot).join(words.join(" "))
}

/**
 * A value quoted for the model that reads a user-spelled command, never for a
 * shell: `next_phase.quote_plan_arg`'s rule, which the suite holds this to over
 * a shared case table. Bare without whitespace or a quote; single quotes around
 * a value holding `"` but not `'`; double quotes otherwise. Nothing escaped.
 */
function quoteForReader(value) {
  const hasDouble = value.includes('"')
  const hasSingle = value.includes("'")
  if (hasDouble && !hasSingle) return `'${value}'`
  if (hasDouble || hasSingle || /\s/.test(value)) return `"${value}"`
  return value
}

/**
 * A runnable dispatcher command, matched by shape rather than by equality.
 *
 * `cli_common.dispatcher_command()` renders one line — `uv run … --project
 * <dir> -- python <dir>/<script> <mode> …` — POSIX-quoted with *single* quotes,
 * so a match may stop at a line break or at a double quote and still have taken
 * the whole command. Stopping at the quote is what lets one pattern scrub a
 * copy embedded in a JSON payload without eating the payload around it. The
 * script name is generated data; the binary names are the ones `whichUv`
 * probes.
 */
function runnablePattern(surface) {
  const script = escapeRegExp(surface.dispatcher.script)
  // A command ends at the line, a double quote or a backtick: the exit-3
  // option blocks quote the command inline (``re-invoke with `uv run …` — it
  // re-dispatches …``), and a scrape that ran to the end of the line handed
  // back the closing backtick and the rest of the sentence as extra flags.
  // Except `"'"`: the middle of `cli_common._quote`'s `'"'"'`, its only
  // spelling of an apostrophe — a question holding one used to be cut there
  // (2026-10-05).
  const body = `(?:[^\\n"\`]|"'")*`
  return new RegExp(
    `[^\\s"\`]*\\buv(?:\\.exe)?\\b\\s+run\\b${body}${script}${body}`,
    "g",
  )
}

/**
 * Whether a `"` cut this match where its last token may go on.
 *
 * The pattern stops at a double quote, and every command the dispatcher prints
 * is POSIX-quoted (`cli_common._quote`), so a stop right after a closing quote
 * (or an escaping backslash) can sit inside one shell word: the parse then
 * succeeds on the part before the cut, and the re-spelled command — and the
 * option — carry a truncated value (an `ask` question with a `"` in it,
 * 2026-10-05). Such a match keeps the pointer and comes back unparseable; a
 * cut inside an open quote already fails to parse.
 */
function cutMidToken(match, offset, whole) {
  return (
    typeof whole === "string" &&
    whole[offset + match.length] === '"' &&
    (match.endsWith("'") || match.endsWith("\\"))
  )
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

/**
 * Every runnable command a child printed, spelled as a tool call.
 *
 * Rule (a) applied to the text the tool captured rather than to the `next` key.
 * A command redacted out of a needs-human block or an error's output is handed
 * back in the only form this plugin ever hands one back, in the order it
 * appeared — so nothing is dropped by the redaction and nothing is pasteable.
 */
function toolCallsFor(child, surface, values = null) {
  if (!child || !surface) return []
  const raw = child.result?.next
  const commands = typeof raw === "string" && raw ? [raw] : []
  const captured = `${child.stdout || ""}\n${child.stderr || ""}`
  for (const found of captured.matchAll(runnablePattern(surface))) {
    // A command a quote cut short is reported, never offered: its parse
    // would carry the truncated value (`cutMidToken`).
    commands.push(
      cutMidToken(found[0], found.index, captured) ? null : found[0],
    )
  }
  const seen = new Set()
  const calls = []
  for (const command of commands) {
    const call =
      command === null ? unparseableCall() : nextAsToolCall(command, surface)
    // Never a `null` beside the human entry: the needs-human branch appends
    // that entry after this list, and "the last option" has to mean it.
    if (!call) continue
    if (call.args) {
      // The plan as this call typed it, when it is this call's plan: the
      // dispatcher prints it absolute (racing re-run 2026-10-04).
      call.args.plan = typedPlan(call.args.plan, values)
      call.human = describeCall(call.args, surface)
    }
    // One option per distinct call, not per printed line: a failed run's own
    // retry hint and `status`'s `resume with:` line differ only in flags this
    // tool composes, and the same option was listed twice.
    const key = JSON.stringify(call.args ?? call)
    if (seen.has(key)) continue
    seen.add(key)
    calls.push(call)
  }
  return calls
}

/**
 * Re-spell a `MULTI_LLM_RESULT.next` string as a tool call.
 *
 * The tool output must contain no runnable `uv run` string at all — none is
 * composed on the way in, and none is handed back on the way out. Handed a
 * shell command, a model's natural move is to run it in bash, which bypasses
 * every guard in this file.
 */
function nextAsToolCall(next, surface) {
  if (!next || typeof next !== "string") return null
  const parsed = parseDispatcherCommand(next, surface)
  if (!parsed) return unparseableCall()
  return {
    tool: surface.tool_name,
    args: parsed,
    human: describeCall(parsed, surface),
  }
}

/** What a follow-up this plugin cannot parse into a tool call comes back as. */
function unparseableCall() {
  return {
    unparseable: true,
    note:
      "The dispatcher returned a follow-up this plugin could not parse into " +
      "a tool call. It is reported rather than executed. Do not run it in a " +
      "shell.",
  }
}

/**
 * Parse `uv run … multi_llm.py <mode> <positionals> <flags>` back into args.
 *
 * Deliberately narrow: it takes only what it can name, and anything it cannot is
 * reported rather than executed.
 */
function parseDispatcherCommand(command, surface) {
  const tokens = splitPosix(command)
  if (!tokens) return null
  const scriptAt = tokens.findIndex((t) =>
    t.endsWith(surface.dispatcher.script),
  )
  if (scriptAt < 0 || scriptAt + 1 >= tokens.length) return null
  const rest = tokens.slice(scriptAt + 1)
  const row = resolveMode(rest[0], surface)
  if (!row) return null

  const positionals = []
  const flags = []
  let seenFlag = false
  for (const token of rest.slice(1)) {
    if (token.startsWith("-")) seenFlag = true
    if (seenFlag) flags.push(token)
    else positionals.push(token)
  }
  const takesPlan = row.positionals.includes(surface.positional_kinds.plan)
  const takesQuestion = row.positionals.includes(
    surface.positional_kinds.question,
  )
  // Drop the flags this tool composes itself — **and each one's value**. A
  // filter that dropped only the flag token would leave `opencode` behind from
  // `--host opencode`, which then reads as a bare word the dispatcher rejects.
  // A value is any following token that does not itself look like a flag, and
  // an `=` form carries its value already.
  const kept = []
  for (let index = 0; index < flags.length; index += 1) {
    const token = flags[index]
    if (!reservedHit(token, surface)) {
      kept.push(token)
      continue
    }
    const inline = token.includes("=")
    const next = flags[index + 1]
    if (!inline && next !== undefined && !next.startsWith("-")) index += 1
  }
  const args = {
    mode: row.name,
    plan: takesPlan ? (positionals[0] ?? "") : "",
    question: takesQuestion ? (positionals[1] ?? "") : "",
    extra_flags: kept,
  }
  if (takesPlan && !args.plan) return null
  return args
}

/** POSIX word splitting, quotes only. Nothing is executed; this only reads. */
function splitPosix(text) {
  const out = []
  let current = ""
  let quote = null
  let any = false
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]
    if (quote) {
      if (character === quote) quote = null
      else if (quote === '"' && character === "\\" && index + 1 < text.length) {
        index += 1
        current += text[index]
      } else current += character
      any = true
      continue
    }
    if (character === "'" || character === '"') {
      quote = character
      any = true
      continue
    }
    if (character === "\\" && index + 1 < text.length) {
      index += 1
      current += text[index]
      any = true
      continue
    }
    if (/\s/.test(character)) {
      if (any) out.push(current)
      current = ""
      any = false
      continue
    }
    current += character
    any = true
  }
  if (quote) return null // unbalanced
  if (any) out.push(current)
  return out
}

function describeCall(args, surface) {
  const parts = [args.mode]
  if (args.plan) parts.push(args.plan)
  if (args.extra_flags.length) parts.push(...args.extra_flags)
  return `call ${surface.tool_name} with ${parts.join(" ")}`
}

/**
 * The executor's refusals are structured, not crashes.
 *
 * `nested_exec_denied`, `nested_exec_spawn`, `nested_exec_network` and
 * `nested_exec_credentials` each carry their own explanation text. They surface
 * as a named reason with the capability field they came from — never a stack
 * trace, never a bare non-zero exit, never a silent exit 0.
 */
function nestedExecReason(child, surface) {
  const haystack = `${child.stdout || ""}\n${child.stderr || ""}`
  for (const [code, field] of Object.entries(
    surface?.nested_exec_reasons ?? {},
  )) {
    if (haystack.includes(code)) {
      return { reason: code, capability_field: field, log: child.banner?.log ?? null }
    }
  }
  return undefined
}

/** A failed child is a result, not an exception. */
function errorResult(child, message, anchor, extra = {}) {
  const { runId = null, logPath = null, surface = null, values = null } = extra
  // The same whole-result invariant as the terminal branches: a failing child's
  // captured output routinely carries runnable commands — a launch banner's
  // `status:`/`abort:` lines, a refusal that names its own fix — and an error is
  // no licence to hand one to a model. Redacted here, returned as tool calls.
  const options = toolCallsFor(child, surface, values)
  const out = {
    state: "error",
    ok: false,
    exit: child?.code ?? null,
    argv: child?.argv ?? null,
    anchor,
    message,
    stderr: redactCommand(child?.stderr, null, surface) || undefined,
    output: humanOutput(child, null, surface) || undefined,
    options: options.length ? options : undefined,
    nested_exec: child ? nestedExecReason(child, surface) : undefined,
  }
  if (runId) {
    // `status` itself failing is never a statement about the run: the claimed
    // run is still alive, so the result carries what a human needs to stop it.
    out.run_id = runId
    out.log = logPath
    if (surface && values) {
      out.abort_command = {
        tool: surface.tool_name,
        args: abortArgs(values, runId, surface),
      }
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// The core instructions, surfaced by the tool rather than by prose
// ---------------------------------------------------------------------------

/**
 * `util instructions <mode> --host <id>` — the core plus the three resolved
 * capability verbs.
 *
 * The command body is one line now, so the only place the model used to read the
 * workflow and the verb resolutions is gone. Nothing is copied into JS and
 * nothing is copied into the command body: the single owner still produces the
 * text, just at run time and through the tool.
 */
async function fetchInstructions({ row, surface, ctx, env, uv, anchor }) {
  if (!row.name) return null
  const argv = [
    ...surface.dispatcher.instructions,
    row.name,
    surface.dispatcher.host_option,
    surface.host,
  ]
  const child = await runChild({
    uv,
    argv: composeSpawnArgv(uv, argv, surface, env),
    cwd: anchor,
    env,
    signal: ctx?.abort,
    marker: surface.dispatcher.result_marker,
    banner: surface.dispatcher.banner.fields,
  })
  if (child.spawnError || child.code !== surface.dispatcher.exit_codes.ok) {
    return null
  }
  // Strip the trailing result line: it is the dispatcher's own envelope, not
  // part of the core text.
  return child.stdout
    .split("\n")
    .filter((line) => !line.trim().startsWith(surface.dispatcher.result_marker))
    .join("\n")
    .trimEnd()
}

/**
 * `util next-command <plan> --blocked <mode> --host <id>`: the command that
 * resumes a blocked write phase, as the result line's `user_resume` (its
 * `command`, the resume's `flags`, the stop's `answers`) plus the `word` and
 * `plan` it was asked about, or `null`.
 *
 * Asked after an exit 3 so the user is left with something to type: the
 * dispatcher's own output names the retry as a raw `uv run` argv (redacted
 * here) and the options as tool calls, and neither is a command a person types.
 * The string is built in Python from the host's `next_command` template for this
 * route, so this file spells none of it, and it is the user's to type: nothing
 * here (or in `isMechanical`) ever runs it. A surface without the `resume` entry,
 * a spawn failure, a non-zero exit or an empty answer all give `null`: an
 * advisory lookup never changes the result it decorates. The mode asked about
 * is `blockedWord`'s: a `status`/`abort` call reports on a run and is no phase.
 */
async function fetchResumeCommand({
  values, surface, ctx, env, uv, anchor, outcome = null,
}) {
  const resume = surface.dispatcher.resume
  const plan = values.planArg || values.plan
  const word = blockedWord(values, outcome, surface)
  if (!resume || !plan || !word) return null
  const argv = [
    ...resume.argv,
    plan,
    resume.option,
    word,
    surface.dispatcher.host_option,
    surface.host,
  ]
  const child = await runChild({
    uv,
    argv: composeSpawnArgv(uv, argv, surface, env),
    cwd: anchor,
    env,
    signal: ctx?.abort,
    marker: surface.dispatcher.result_marker,
    banner: surface.dispatcher.banner.fields,
  })
  if (child.spawnError || child.code !== surface.dispatcher.exit_codes.ok) {
    return null
  }
  const step = child.result?.[resume.result_key]
  const command = step?.command
  return typeof command === "string" && command.trim()
    ? { ...step, command, word, plan }
    : null
}

/**
 * The stop's line without the ways forward its section carries, in the
 * `error`'s place (fix round 28, live run 12 #5: the hosts relayed the whole
 * `error` line, whose "To go on (your call): … add --human-task …" restated
 * every answer the section then gave as commands).
 *
 * `util next-command --blocked` hands back the line it read off the stop's
 * record (`error_line`) and that line without the section's ways forward
 * (`error_beside_section`). Swapped only while the result's `error` is the
 * recorded line word for word, so a stop the record does not describe keeps
 * its own line; `result.error` follows (the stop title reads it) and
 * `message` is composed again around it. The port of the codex server's
 * `relay_stop_beside_section`.
 */
function relayStopBesideSection(values, outcome, step, surface) {
  const line = step?.error_line
  const beside = step?.error_beside_section
  const payload = outcome?.result
  if (
    typeof line !== "string" || typeof beside !== "string" || !beside.trim() ||
    !payload || typeof payload !== "object"
  ) {
    return
  }
  const respell = respellerFor(surface, values)
  const current = outcome.error
  const recorded = redactCommand(line, null, surface, respell)
  if (
    typeof current !== "string" || typeof recorded !== "string" ||
    current.trim() !== recorded.trim()
  ) {
    return
  }
  const shown = redactCommand(beside, null, surface, respell)
  outcome.error = shown
  payload.error = shown
  outcome.message = needsHumanMessage({
    payload,
    output: outcome.output,
    stderr: outcome.stderr,
    values,
    surface,
  })
}

/**
 * Lead `options` with the resume and every answer the stop offers.
 *
 * Fix round 14 (live run 5): `resume_command` was a plain `--resume` while the
 * first option replayed `--human-task` answers, and on a first stop `message`
 * and `options` offered only the bare resume, the answers sitting in `error`.
 * `util next-command --blocked` now hands back the resume's own `flags` and
 * the stop's `answers` (each a command the user types, its flags and what it
 * does); here the resume becomes the first option — the same command as
 * `resume_command` by construction — each answer the next ones, in order, and
 * the scraped options follow minus any repeat, the human note still last. The
 * answers go to `resume_answers` and, through the surface's `answers` clause
 * filled with the dispatcher's `answers_text`, into `message`. This file
 * spells none of it. The port of the codex server's `attach_answers`.
 *
 * The option calls name the lookup's `mode`, the mode the resume runs, and
 * the word asked about only when the lookup names none: a composite asked
 * about answers with an inner phase's resume and answers (plans/improve-plan-
 * and-improve-code.md §2.6), and a call built from the asked word would hand
 * `--human-task` to a parser that rejects it.
 */
function attachAnswers(step, outcome, surface) {
  const named = typeof step.mode === "string" && step.mode ? step.mode : step.word
  const row = resolveMode(named, surface)
  if (!row) return
  const flagsOf = (value) =>
    Array.isArray(value) &&
    value.every((flag) => typeof flag === "string" && flag)
      ? [...value]
      : null
  const callFor = (extraFlags) => {
    const args = {
      mode: row.name, plan: step.plan, question: "", extra_flags: extraFlags,
    }
    return { tool: surface.tool_name, args, human: describeCall(args, surface) }
  }
  const lead = []
  const flags = flagsOf(step.flags)
  if (flags) lead.push(callFor(flags))
  const answers = []
  for (const entry of Array.isArray(step.answers) ? step.answers : []) {
    const entryFlags = flagsOf(entry?.flags)
    if (
      !entryFlags || typeof entry.command !== "string" ||
      !entry.command.trim() || typeof entry.why !== "string"
    ) {
      continue
    }
    answers.push({ command: entry.command, why: entry.why })
    lead.push(callFor(entryFlags))
  }
  if (answers.length) {
    outcome.resume_answers = answers
    const clause = surface.needs_human_message?.answers
    const text = step.answers_text
    if (
      typeof clause === "string" && typeof text === "string" && text.trim() &&
      !("resume_markdown" in outcome)
    ) {
      outcome.message = `${outcome.message} ${clause.split("{answers}").join(text)}`
    }
  }
  if (!lead.length) return
  const seen = new Set(lead.map((call) => JSON.stringify(call.args)))
  const rest = (outcome.options ?? []).filter(
    (option) => !(option?.args && seen.has(JSON.stringify(option.args))),
  )
  outcome.options = [...lead, ...rest]
}

/**
 * The mode `fetchResumeCommand` asks about: the call's own when it is a phase,
 * else the phase of the run the call reported on.
 *
 * A `status` or `abort` call on a failed run re-spells status's `resume with:`
 * line into the host's command, and asking `--blocked status` — no phase, so no
 * answer — then put "give the user no command" beside it (2026-10-05). The run
 * is named by the result line (a poll pinned to one run carries its `mode` and
 * `phase`), else by the one phase the output re-invokes; several, or none, name
 * nothing.
 */
function blockedWord(values, outcome, surface) {
  const isPhase = (word) =>
    typeof word === "string" && !!resolveMode(word, surface)?.phase
  if (isPhase(values?.word)) return values.word
  const result = outcome?.result
  for (const word of [result?.mode, result?.phase]) {
    if (isPhase(word)) return word
  }
  const named = new Set(
    (outcome?.options ?? [])
      .map((option) => option?.args?.mode)
      .filter(isPhase)
      .map((word) => resolveMode(word, surface).name),
  )
  return named.size === 1 ? [...named][0] : null
}

// ---------------------------------------------------------------------------
// The plugin
// ---------------------------------------------------------------------------

/**
 * The `config` hook's command injection.
 *
 * OpenCode merges every `{command,commands}/**\/*.md` file into a single
 * `config.command` map first, then hands that same object **by reference** to
 * each plugin's `config` hook. Hooks run in load order, return nothing, and
 * neither their output nor the merge is validated or checked for collisions —
 * so there is no host-level precedence between the two routes and the hook's own
 * assignment operator decides. Plain assignment is pinned deliberately: the npm
 * route wins, and a file shim at `~/.config/opencode/commands/multi-llm.md`
 * becomes inert — still on disk, never invoked.
 *
 * Nothing in OpenCode says so. One entry in the map, no duplicate, no WARN, no
 * ERROR, and no mention of the name at DEBUG. Since the hook sees the file entry
 * *before* overwriting it, this plugin is the only place that collision can ever
 * be reported — so it reports it.
 *
 * It deliberately does not touch `cfg.skills`, though it could: opencode's skill
 * frontmatter schema silently rejects `argument-hint`, which this SKILL.md
 * carries, so it would never list — and a host already carrying its own copy of
 * the tree would then discover two `multi-llm` skills.
 */
function injectCommand(cfg, surface) {
  const commandPath = path.join(DIST_DIR, "command.md")
  const raw = fs.readFileSync(commandPath, "utf8")
  const body = resolveSkillDir(raw)
  const { description, template } = splitFrontmatter(body)

  cfg.command = cfg.command || {}
  if (cfg.command[COMMAND_NAME]) {
    const removal = uninstallCommand(surface)
    warn(
      `a "/${COMMAND_NAME}" command already exists — almost certainly the ` +
        `file-route shim written by install-host. This plugin's command ` +
        `replaces it wholesale, so that file is now dead weight.` +
        (removal ? ` Remove it with: ${removal}` : ""),
    )
  }
  cfg.command[COMMAND_NAME] = { template, description }
}

/** Resolve every `{SKILL_DIR}` in a generated artifact to this package's tree. */
function resolveSkillDir(text) {
  return text.split(SKILL_DIR_TOKEN).join(SKILL_DIR)
}

/**
 * The runnable command that removes the shim the warning just found, or `""`.
 *
 * Generated, like everything else a user is shown: `host_install.uninstall_line()`
 * is the single owner of that spelling — it is the same string `--check` prints
 * for the both-routes posture — and it reaches here through the surface with its
 * skill dir still a token. The warning is the only load-time report that a file
 * shim has gone inert, so the repair it names has to be one a user can paste;
 * composed in JS it drifted into `multi_llm.py uninstall-host <host>`, which has
 * no uv prefix, no resolved path and a placeholder where the host goes.
 *
 * A surface that is missing or too old to carry the field yields `""` and the
 * warning simply stops one sentence early — never a half-spelled command.
 */
function uninstallCommand(surface) {
  const line = surface?.uninstall_command
  return typeof line === "string" && line ? resolveSkillDir(line) : ""
}

/** Split `---\n…\n---\n` frontmatter, returning its description and the body. */
function splitFrontmatter(text) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text)
  if (!match) return { description: "", template: text }
  const described = /^description:\s*(.*)$/m.exec(match[1])
  let description = described ? described[1].trim() : ""
  if (
    (description.startsWith('"') && description.endsWith('"')) ||
    (description.startsWith("'") && description.endsWith("'"))
  ) {
    description = description.slice(1, -1)
  }
  return { description, template: match[2] }
}

/**
 * The `tool` hook's value: a map of tool id to definition, or `{}`.
 *
 * A **map**, not a registering function. `Hooks.tool` is typed
 * `{ [id: string]: ToolDefinition }`, and opencode's tool registry collects
 * plugin tools with `for (const [id, def] of Object.entries(plugin.tool ?? {}))`
 * — a function has no enumerable own entries, so a `tool: async (registry) => …`
 * hook is never called, never throws, and registers nothing. The failure is
 * silent and total: opencode starts clean, `/multi-llm` still injects, and
 * `multi_llm` is simply absent from the model's tool list, which is exactly the
 * exit-0 death rule 4 exists to prevent. Building the map eagerly here is also
 * what lets a bad surface be *reported*: there is no later hook call to report
 * from.
 *
 * `toolDescription`/`toolArgs` run inside the try for the reason the config
 * hook has one. `loadSurface` checks the schema *version* and nothing else, so a
 * payload that declares the right version with the wrong shape —
 * `argument_descriptions` absent, `tool_name` not a string — throws a TypeError
 * out of them. An
 * unregistered tool is a warned, degraded plugin; a throw out of the factory
 * takes OpenCode's whole plugin load with it.
 */
function buildToolHook(surface, client) {
  if (!surface) {
    warn(
      `the ${DEGRADED_TOOL_NAME} tool is not registered: the generated CLI ` +
        `surface could not be read. The package looks partial; reinstall it.`,
    )
    return {}
  }
  try {
    const name = surface.tool_name
    if (typeof name !== "string" || !name) {
      throw new TypeError("tool_name is not a non-empty string")
    }
    return {
      [name]: {
        description: toolDescription(surface),
        args: toolArgs(surface),
        // `ToolResult` is `string | { output, title?, metadata?, attachments? }`
        // and opencode reads a non-string result as exactly those four keys.
        // Handing back the raw envelope would therefore keep only its `output`
        // — the human stdout — and drop `state`, `ok`, `next`, `anchor` and
        // `result` on the floor, silently turning a structured contract into a
        // prose blob (and an error branch, which has no `output` at all, into
        // an `undefined` one). One JSON string keeps every field.
        execute: async (args, ctx) =>
          JSON.stringify(await execute(args, ctx, surface, client)),
      },
    }
  } catch (error) {
    warn(
      `the ${DEGRADED_TOOL_NAME} tool is not registered: the generated CLI ` +
        `surface is not the shape this plugin build expects. The package looks ` +
        `partial; reinstall it. ${describe(error)}`,
    )
    return {}
  }
}

// The factory input is opencode's own `PluginInput` — `{ client, app, $,
// directory, worktree, … }`. Destructured with a default so a host that calls
// the factory with no argument at all (every one of this suite's `await
// MultiLlmPlugin()` calls, and any older loader) still gets a working plugin:
// `client` is then `undefined`, `resolveSessionModel` returns null, and the run
// resolves its runner exactly as it did before this existed (rule 4).
export const MultiLlmPlugin = async ({ client } = {}) => {
  // Read once at load. A null surface degrades the plugin rather than crashing
  // OpenCode; `execute` reports it, and the command still injects.
  const surface = loadSurface()

  return {
    config: async (cfg) => {
      try {
        injectCommand(cfg, surface)
      } catch (error) {
        // Never throw at plugin init. A hook that throws is logged and
        // swallowed by OpenCode anyway, and the alternative is a degraded
        // plugin that looks identical to a working one.
        warn(`could not inject /${COMMAND_NAME}: ${describe(error)}`)
      }
    },

    tool: buildToolHook(surface, client),

    // The stop, on the tool's own line (fix round 16): see `stopTitle`.
    "tool.execute.after": async (input, output) => {
      try {
        titleTheStop(input, output, surface)
      } catch (error) {
        warn(`could not title the stop on the tool part: ${describe(error)}`)
      }
    },
  }
}

/**
 * The tool part's title when a call ends at a stop, or null.
 *
 * OpenCode racing run 6 (fix round 16): implement stopped at T006, the result
 * held the stop and `resume_command`, and the session's free model never wrote
 * its reply — it sat idle for 28 minutes until it was killed, so the user got
 * nothing. A plugin tool's title is the one thing opencode shows of a tool
 * call without the model: `opencode run` prints it on the tool's own line
 * (`⚙ multi_llm <title>`, in place of the call's arguments) the moment the call
 * returns and `--format json` carries it as `state.title` (measured offline
 * on 1.18.34 with a stub provider and a model that never replies; the TUI's
 * tool header reads the same title, which was not measured). `tool.execute.after` is where a plugin sets it;
 * `execute` keeps returning one JSON string.
 *
 * Only a finished stop is titled: a failed or needs-human result that carries
 * a `resume_command` or a `no_command_reason` (a run left alive, a handoff or
 * an abort report carries neither). Every word is the surface's `stop_title`
 * or the result's own: the run's last error line (`result.error`, the line the
 * exit-3 `message` quotes) and `resume_command`.
 *
 * Fix round 17: the error line carries a task title from tasks.md and the
 * runner's own summary, so both parts are shown without control characters
 * (`plainTitleText`: an ANSI sequence could erase or rewrite the tool line,
 * the resume command included), and the error line is cut at the surface's
 * `max_line` characters, ending in its `cut`; the resume command is appended
 * after it and never cut.
 */
function stopTitle(outcome, surface) {
  const text = surface?.stop_title
  if (!text || typeof text.template !== "string") return null
  if (!outcome || typeof outcome !== "object" || outcome.ok !== false) return null
  const command =
    typeof outcome.resume_command === "string" && plainTitleText(outcome.resume_command)
      ? plainTitleText(outcome.resume_command)
      : null
  if (!command && typeof outcome.no_command_reason !== "string") return null
  const raw = outcome.result?.error
  const own = typeof raw === "string" ? plainTitleText(raw) : ""
  const line = own ? capTitleLine(own, text) : text.no_line
  if (typeof line !== "string") return null
  let title = text.template.split("{line}").join(line)
  if (command && typeof text.resume === "string") {
    title += text.resume.split("{command}").join(command)
  }
  return title
}

/**
 * `value` as one plain line for a tool title: whitespace runs one space, and
 * no C0 or C1 control, DEL, or bidirectional override or isolate left (the
 * set `cli_common.plain_line` removes from the result's `error`).
 */
function plainTitleText(value) {
  return String(value)
    .replace(/[\u0000-\u001f\u007f-\u009f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g, (ch) =>
      /[\t\n\v\f\r\u0085]/.test(ch) ? " " : "",
    )
    .replace(/\s+/g, " ")
    .trim()
}

/** Characters a cut never ends on before its mark (`text_clip.DANGLING`). */
const TITLE_DANGLING = new Set(Array.from(" \t,;:.!?/\\|&+=*-\u2013\u2014\u00b7"))
/** Bracket pairs a cut closes again after its mark, and the marks that toggle. */
const TITLE_PAIRS = { "(": ")", "[": "]", "{": "}", "\u201c": "\u201d" }
const TITLE_CLOSING = Object.fromEntries(
  Object.entries(TITLE_PAIRS).map(([opener, closer]) => [closer, opener]),
)
const TITLE_TOGGLES = new Set(['"', "`"])
const titleSpace = (ch) => /\s/u.test(ch)
const titleWord = (ch) => /[\p{L}\p{N}]/u.test(ch)

/**
 * The marks `chars` opens and never closes, outermost first, as
 * `[index, mark]` (`text_clip.unclosed`): brackets pair by kind, `"` and
 * backticks toggle, and a `'` counts only where it cannot be an apostrophe.
 */
function unclosedMarks(chars) {
  const stack = []
  const close = (mark) => {
    for (let depth = stack.length - 1; depth >= 0; depth--) {
      if (stack[depth][1] === mark) {
        stack.length = depth
        return true
      }
    }
    return false
  }
  chars.forEach((ch, index) => {
    if (Object.hasOwn(TITLE_PAIRS, ch)) stack.push([index, ch])
    else if (Object.hasOwn(TITLE_CLOSING, ch)) close(TITLE_CLOSING[ch])
    else if (TITLE_TOGGLES.has(ch)) {
      if (!close(ch)) stack.push([index, ch])
    } else if (ch === "'") {
      const before = index ? chars[index - 1] : " "
      const after = index + 1 < chars.length ? chars[index + 1] : " "
      if (stack.some(([, mark]) => mark === "'") && !titleSpace(before) && !titleWord(after)) {
        close("'")
      } else if (!titleWord(before) && !titleSpace(after)) {
        stack.push([index, ch])
      }
    }
  })
  return stack
}

/**
 * `line` cut to the surface's `max_line` characters, ending in its `cut`.
 *
 * Fix round 22 (live run 9, #4): the same rule as the dispatcher's
 * `text_clip.clip`, which cuts every other preview of the model's words: at a
 * word boundary (a token longer than a third of the room is cut inside), no
 * separator before the mark, and every bracket or quote the kept part opened
 * closed again after it. It was a bare cut mid-word.
 */
function capTitleLine(line, text) {
  const max = text.max_line
  if (!Number.isInteger(max) || max <= 0) return line
  const chars = Array.from(line)
  if (chars.length <= max) return line
  const cut = typeof text.cut === "string" ? text.cut : ""
  let room = max - Array.from(cut).length
  while (room > 0) {
    let kept = chars.slice(0, room)
    if (!titleSpace(chars[room])) {
      for (let index = kept.length - 1; index >= 0; index--) {
        if (titleSpace(kept[index])) {
          if (3 * index >= 2 * room) kept = kept.slice(0, index)
          break
        }
      }
    }
    let stack
    for (;;) {
      while (kept.length && TITLE_DANGLING.has(kept[kept.length - 1])) kept.pop()
      stack = unclosedMarks(kept)
      if (stack.length && stack[stack.length - 1][0] === kept.length - 1) {
        kept.pop()
        continue
      }
      break
    }
    if (!kept.length) return cutTitleInside(chars, max, cut)
    const closers = stack.slice().reverse().map(([, mark]) => TITLE_PAIRS[mark] ?? mark)
    const clipped = kept.join("") + cut + closers.join("")
    const length = Array.from(clipped).length
    if (length <= max) return clipped
    room -= length - max
  }
  return Array.from(cut).slice(0, max).join("")
}

/**
 * `chars` cut inside within `max`, for a head the word-boundary cut left
 * nothing of (`text_clip._cut_inside`, fix round 23: a line of openers was
 * cut to the bare mark): the first characters that fit beside the mark and
 * the closers of what they open, a trailing separator dropped unless that
 * leaves nothing.
 */
function cutTitleInside(chars, max, cut) {
  let room = max - Array.from(cut).length
  while (room > 0) {
    const head = chars.slice(0, room)
    let kept = head.slice()
    while (kept.length && TITLE_DANGLING.has(kept[kept.length - 1])) kept.pop()
    if (!kept.length) {
      kept = head.slice()
      while (kept.length && titleSpace(kept[kept.length - 1])) kept.pop()
    }
    if (!kept.length) break
    const closers = unclosedMarks(kept).reverse().map(([, mark]) => TITLE_PAIRS[mark] ?? mark)
    const clipped = kept.join("") + cut + closers.join("")
    const length = Array.from(clipped).length
    if (length <= max) return clipped
    // Each character dropped takes at most one closer with it.
    room -= Math.floor((length - max + 1) / 2)
  }
  return Array.from(cut).slice(0, max).join("")
}

/** `tool.execute.after`: title this tool's stop, leave every other call alone. */
function titleTheStop(input, output, surface) {
  if (!surface || input?.tool !== surface.tool_name) return
  if (!output || typeof output.output !== "string") return
  let outcome
  try {
    outcome = JSON.parse(output.output)
  } catch {
    return
  }
  const title = stopTitle(outcome, surface)
  if (title) output.title = title
}


/** The whole contract, in one function. Returns; never throws. */
async function execute(args, ctx, surface, client) {
  try {
    const env = { ...process.env }
    const anchor = resolveAnchor(ctx)

    args = withAbsentQuestion(args)
    const checked = validateArgs(args, surface)
    if (!checked.ok) {
      return { state: "error", ok: false, anchor, message: checked.error }
    }
    const row = checked.mode

    const uv = findUv(surface, env)
    if (!uv) {
      // Refuse rather than fall back to bare `python`: the skill's dependencies
      // are pinned in uv.lock and resolved by `uv run` from pyproject.toml, so a
      // bare interpreter turns a clear "install uv" into a ModuleNotFoundError
      // or, worse, a silently wrong dependency version mid-fan-out.
      return finishOutcome({
        state: "needs-human",
        ok: false,
        anchor,
        message: uvMissingText(surface),
      }, checked, surface)
    }
    // Propagate the binary the preflight approved, so the detached worker
    // re-resolves to it instead of re-probing a PATH it may not share.
    env[surface.uv.env] = uv
    // The route signal, on the environment the launcher, every poll, the abort
    // dispatch and the instructions fetch all run with: the dispatcher spells a
    // finished phase's `user_next_step` command for this route. Both strings
    // are the surface's (`loadSurface` refused a surface without them); this
    // file spells neither.
    env[surface.dispatcher.route_env.name] = surface.dispatcher.route_env.value

    // The session's own model, on the environment the launcher AND every poll
    // run with — the detached worker inherits it through `utils/detach.py`'s
    // `child_env`, which copies `os.environ`. The variable's name is generated
    // (Python owns it); this file only supplies the value, and only when the
    // lookup succeeded, so an unanswered lookup is byte-for-byte the old
    // behaviour rather than an empty variable the ladder has to special-case.
    const sessionModel = await resolveSessionModel(client, ctx)
    const sessionModelEnv = surface.dispatcher?.session_model_env
    if (sessionModel && sessionModelEnv) env[sessionModelEnv] = sessionModel

    const values = {
      // The word as sent, so an alias reaches the dispatcher unexpanded.
      word: args.mode,
      plan: normalizePlan(args.plan, anchor),
      planArg: planSpelling(args.plan),
      question: args.question,
      // What `validateArgs` forwards: a restated flag this tool composes with
      // the same meaning is dropped here, quietly (run 6, #10a/#14).
      extraFlags: checked.extraFlags ?? args.extra_flags,
    }

    // What the relay has already said, and what it did: see `relayMemory`,
    // where both halves of "per call" are written down — not per module (this
    // module outlives the session's every tool call), and deliberately not per
    // *run* either, so a mechanical continuation inherits it and its identical
    // roster stays unrepeated. Created here rather than inside the loop because
    // *this* is the frame that reads it back onto the answer.
    const relayState = relayMemory()
    const outcome = await runWorkflow({
      row, values, surface, ctx, client, env, uv, anchor, relayState,
    })
    handBackTyped(outcome, values, surface)
    // Numbers, not a sentence (`relayCounts`): what the mid-run channel was
    // asked to do, on the one surface a user can hand back.
    const relays = relayCounts(relayState)
    if (relays) outcome.relays = relays

    // The core instructions and the three resolved verb sentences, for a call
    // whose mode *and* whose outcome leave the model work after the dispatcher
    // returns — the predicate below implements both halves of that sentence.
    if (outcome.ok && rowNeedsInstructions(row, outcome, surface)) {
      const instructions = await fetchInstructions({
        row, surface, ctx, env, uv, anchor,
      })
      if (instructions) outcome.instructions = instructions
    }
    // A blocked run owes the user a command they can type: the same phase,
    // resumed, in this route's own spelling. Asked of the dispatcher (which
    // owns the spelling), after the result is composed, and only for an exit 3.
    if (
      outcome.state === surface.dispatcher.states.needs_human &&
      outcome.exit === surface.dispatcher.exit_codes.needs_human
    ) {
      const resume = await fetchResumeCommand({
        values, surface, ctx, env, uv, anchor, outcome,
      })
      if (resume) {
        outcome.resume_command = resume.command
        const text = surface.needs_human_message
        const why = typeof resume.why === "string" ? resume.why.trim() : ""
        let clause = text.resume
        if (why && typeof text.resume_why === "string") {
          // Fix round 22: what the plain resume does, from the stop.
          outcome.resume_why = why
          clause = text.resume_why.split("{why}").join(why)
        }
        if (
          typeof resume.markdown === "string" && resume.markdown.trim() &&
          typeof text.resume_markdown === "string"
        ) {
          // Fix round 24: the ways forward as one written section, whose
          // clause stands for the resume's and the answers'.
          outcome.resume_markdown = resume.markdown
          clause = text.resume_markdown
          relayStopBesideSection(values, outcome, resume, surface)
        }
        outcome.message = `${outcome.message} ${clause}`
        attachAnswers(resume, outcome, surface)
      }
    }
    // Echoed for the same reason `anchor` is: a wrong one is otherwise
    // invisible, surfacing as "why did it review with THAT model" rather than
    // as anything anyone can point at.
    if (sessionModel) outcome.session_model = sessionModel
    return finishOutcome(outcome, checked, surface)
  } catch (error) {
    // The last line of defence for rule 4. A throw here would be a silent
    // exit-0 death with a 0-byte log; a result is at least readable.
    return {
      state: "error",
      ok: false,
      message: `the multi_llm tool failed unexpectedly: ${describe(error)}`,
    }
  }
}

/**
 * The two things every outcome past argument validation owes the model.
 *
 * A failed or needs-human result says explicitly what the user can run
 * (racing re-run 2026-10-04: a result with no command key at all, and the host
 * model composed a "next phase" of its own): `user_next_step` is null — a phase
 * that did not finish has no next phase — and, unless the result carries a
 * `resume_command`, `no_command_reason` says there is none to give; the
 * dispatcher's exit-3 message gains the surface's `no_command` clause. A flag
 * `validateArgs` dropped from `extra_flags` (one this tool composes, restated
 * with the same meaning) is never named back: run 6 (#10a/#14) relayed the old
 * note to the user as "`--yes` was dropped from your flags" on every call.
 * Every sentence is the surface's; a surface lacking one only loses that
 * sentence.
 *
 * Two results get no reason (2026-10-05): one that names a run and no `exit`
 * (`abortAndReport`, `orphanReport`: the run outlived the call, so "did not
 * finish" is not yet true), and a `status`/`abort` call whose output shows a
 * re-spelled command re-invoking a phase (`showsAPhaseCommand`), which the
 * reason would contradict.
 */
function finishOutcome(outcome, checked, surface) {
  return wayForwardFirst(finishFields(outcome, checked, surface))
}

function finishFields(outcome, checked, surface) {
  if (outcome.ok !== false) return outcome
  outcome.user_next_step = null
  if (outcome.resume_command) return outcome
  const live = !!outcome.run_id && (outcome.exit ?? null) === null
  const report = !!checked?.mode && !checked.mode.phase
  if (live || (report && showsAPhaseCommand(outcome, surface))) return outcome
  if (typeof surface?.no_command_reason === "string") {
    outcome.no_command_reason = surface.no_command_reason
  }
  const clause = surface?.needs_human_message?.no_command
  if (
    typeof clause === "string" &&
    outcome.state === surface.dispatcher.states.needs_human &&
    outcome.exit === surface.dispatcher.exit_codes.needs_human
  ) {
    outcome.message = `${outcome.message} ${clause}`
  }
  return outcome
}

/** The keys that tell the user what to type, right after `ok` and `state`. */
const WAY_FORWARD_KEYS = [
  "user_next_step", "resume_command", "resume_why", "resume_answers",
  "resume_markdown", "no_command_reason",
]

/**
 * The calls a result hands back, led after those when set (fix round 8): a
 * stopped watch's way forward is its `abort_command`, a running one's its
 * `next`, and a needs-human result's `options` trailed `result`, `output` and
 * `stderr`. A null one keeps its place.
 */
const HANDED_BACK_LEAD_KEYS = ["next", "abort_command", "options"]

/** The keys placed last: the long instructions text. */
const LAST_KEYS = ["instructions"]

/**
 * The result ordered for a reader that sees only part of it: `ok`, `state`,
 * the way forward, `message`, everything else, and the instructions last.
 *
 * A host shows its model a cut of a large tool result. Measured on the codex
 * route (racing re-run 3, 2026-10-05): the model read a 50 355-character
 * result whose middle was dropped, `resume_command` sat in that middle, and
 * the model told the user there was no resume command. A cut that keeps the
 * head, the tail or both still shows the way forward when it leads the result
 * and the long text trails it. The port of the codex server's `envelope` order.
 */
function wayForwardFirst(outcome) {
  const ordered = {}
  for (const key of ["ok", "state", ...WAY_FORWARD_KEYS]) {
    if (key in outcome) ordered[key] = outcome[key]
  }
  for (const key of HANDED_BACK_LEAD_KEYS) {
    if (outcome[key] != null) ordered[key] = outcome[key]
  }
  if ("message" in outcome) ordered.message = outcome.message
  for (const [key, value] of Object.entries(outcome)) {
    if (!(key in ordered) && !LAST_KEYS.includes(key)) ordered[key] = value
  }
  for (const key of LAST_KEYS) {
    if (key in outcome) ordered[key] = outcome[key]
  }
  return ordered
}

/**
 * Whether the result's own text shows a re-spelled command that re-invokes a
 * phase: an `options` call on a phase mode, spelled for the user
 * (`userSpelling`), found in `output` or `stderr` as it was rendered.
 */
function showsAPhaseCommand(outcome, surface) {
  const text = [outcome.output, outcome.stderr]
    .filter((part) => typeof part === "string")
    .join("\n")
  if (!text) return false
  return (outcome.options ?? []).some((option) => {
    if (!option?.args || !resolveMode(option.args.mode, surface)?.phase) return false
    const spelled = userSpelling(option.args, surface)
    return !!spelled && text.includes(spelled)
  })
}

/**
 * Whether this call's *outcome* leaves the model work the instructions describe.
 *
 * The mode is only half the question, and reading it alone is what put 22 KB of
 * "run this in bash" into the result of a finished run: measured 2026-09-16, a
 * `review-plan` the plugin had driven to completion with `--exec-subagents`
 * returned `next: null`, `state: "done"` — provably nothing left to do — beside
 * the full rendered runbook, whose fifteen `uv run …` lines the same result's
 * own message forbids running. So the mode has to support detach *and* the
 * outcome has to be unfinished.
 *
 * "Unfinished" is enumerated positively — a `next` the model is expected to
 * send back, or the needs-human state — rather than as "any state that is not
 * the finished one", because that negation leaks the same 22 KB back through
 * every *other* word an `ok: true` result can carry, and exit 2 carries three
 * of them: "this phase has already been completed, use --force" comes back
 * `ok: true`, `state: "nothing-to-do"`, `next: null` and owes the model
 * nothing, as do the poll vocabulary's `running` and `died`, and exit 0's
 * `partial` (a finished run with a failed model slot: terminal, `next: null`;
 * the model names the failures from `progress.models`, it runs nothing). The
 * words that do mean work are the generated surface's, never spelled here.
 */
function rowNeedsInstructions(row, outcome, surface) {
  if (!row.supports_detach) return false
  const states = surface.dispatcher.states
  return (
    (outcome?.next ?? null) !== null || outcome?.state === states.needs_human
  )
}

// Exported for the test suite, which grades this file against its Python owners
// rather than against an expectation someone typed from memory.
const __testing = {
  PACKAGE_DIR,
  SKILL_DIR,
  composeDispatcherArgv,
  composeSpawnArgv,
  capTitleLine,
  expandHome,
  findUv,
  handBackTyped,
  flagMatches,
  injectCommand,
  isMechanical,
  loadSurface,
  nextAsToolCall,
  normalizePlan,
  planFromRoot,
  planSpelling,
  fetchResumeCommand,
  finishOutcome,
  parseDispatcherCommand,
  quoteForReader,
  splitExtraFlags,
  toolCallsFor,
  userSpelling,
  redactCommand,
  respellerFor,
  typedPlan,
  relayCounts,
  relayMemory,
  relayProgress,
  reservedHit,
  resolveAnchor,
  resolveSessionModel,
  rowNeedsInstructions,
  runChild,
  resolveMode,
  splitFrontmatter,
  splitPosix,
  terminalResult,
  toastHoldMs,
  toastMessage,
  toolArgs,
  toolDescription,
  uvMissingText,
  validateArgs,
  wayForwardFirst,
  withAbsentQuestion,
}

// Hung on the factory rather than exported, because opencode calls every export
// of this module as a plugin factory and refuses the whole module over one that
// is not a function (rule 3). Function properties are not exports, so the test
// suite reaches these through a plain `import` and the loader never sees them.
MultiLlmPlugin.execute = execute
MultiLlmPlugin.__testing = __testing
