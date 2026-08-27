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
const TOOL_NAME = "multi_llm"

/**
 * The placeholder the build leaves wherever a generated artifact would
 * otherwise carry an absolute skill dir. Resolving it is this file's job and
 * only this file's: the package lands in whatever cache directory npm gives it,
 * so a path baked at build time is a path that is wrong on every machine.
 */
const SKILL_DIR_TOKEN = "{SKILL_DIR}"

/** Bumped in lockstep with build.py's SURFACE_SCHEMA_VERSION. */
const SUPPORTED_SURFACE_VERSION = 1

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
    return surface
  } catch (error) {
    warn(`cannot read the generated CLI surface: ${describe(error)}`)
    return null
  }
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
// The model-facing definition, assembled from the generated surface
// ---------------------------------------------------------------------------

/**
 * The tool description: a fixed lede, then one generated line per exposed mode.
 *
 * The lede is the *only* authored sentence in the entire definition, and it does
 * not live here — it is `modes.TOOL_DESCRIPTION_LEDE`, rendered into the surface
 * file beside the modes. It carries the one thing no generated line can: that
 * this tool is the only way to run multi-llm on this host, and that its
 * dispatcher is not to be reached through bash.
 *
 * Examples travel inside the description because the JSON-Schema fallback has
 * nowhere to put an `examples` key the host would render.
 */
function buildDescription(surface) {
  const lines = [surface.description_lede, "", "Modes:"]
  for (const mode of surface.modes) {
    const names = mode.aliases.length
      ? `${mode.name} | ${mode.aliases.join(" | ")}`
      : mode.name
    lines.push(`  ${names} — ${mode.summary}`)
  }
  lines.push("")
  lines.push(`The default mode is ${surface.default_mode}.`)
  lines.push("")
  lines.push("Examples:")
  for (const example of workedExamples(surface)) {
    lines.push(`  ${JSON.stringify(example)}`)
  }
  return lines.join("\n")
}

/**
 * The four worked spellings, **selected by shape rather than by name**.
 *
 * Between them they cover every arity rule: a mode with one positional, a mode
 * with none (three empties), a mode with two (a question), and an alias
 * travelling unexpanded beside extra flags. Naming the modes here would put mode
 * words back into this file, which is the one thing it must not hold — so each
 * example is found by what the generated table says the mode *declares*, and a
 * renamed mode changes the examples without an edit.
 *
 * Examples travel inside the description because the JSON-Schema fallback has
 * nowhere to put an `examples` key the host would render.
 */
function workedExamples(surface) {
  const PLAN = surface.positional_kinds.plan
  const QUESTION = surface.positional_kinds.question
  const examples = []
  const spell = (mode, plan, question, extra_flags) => ({
    mode, plan, question, extra_flags,
  })

  // 1. The documented default: one plan, nothing else.
  const fallback = resolveMode(surface.default_mode, surface)
  if (fallback) examples.push(spell(fallback.name, "plans/x.md", "", []))

  // 2. A mode declaring no positional at all — the three-empties spelling,
  //    which is the one a flat "an empty plan is invalid" rule would forbid.
  const nullary = surface.modes.find((mode) => mode.positionals.length === 0)
  if (nullary) examples.push(spell(nullary.name, "", "", []))

  // 3. A mode declaring a question as well as a plan.
  const asking = surface.modes.find(
    (mode) =>
      mode.positionals.includes(PLAN) && mode.positionals.includes(QUESTION),
  )
  if (asking) {
    examples.push(
      spell(asking.name, "plans/x.md", "Why is the ceiling 120 s?", []),
    )
  }

  // 4. An alias, unexpanded, carrying extra flags.
  const aliased = surface.modes.find(
    (mode) => mode.aliases.length > 0 && mode.positionals.includes(PLAN),
  )
  if (aliased) {
    examples.push(
      spell(aliased.aliases[0], "plans/x.md", "", [
        surface.dispatcher.run_id_option,
        "<id>",
      ]),
    )
  }
  return examples
}

/**
 * Argument descriptions, generated per property.
 *
 * `mode` is a plain string, not an enum: the fallback schema is
 * `{type, properties, required}` and has no `enum` to carry one, so legality is
 * the pre-spawn check against the generated table and the *description* is what
 * conveys the vocabulary. A consequence of the no-zod trade, not a preference.
 */
function buildArgs(surface) {
  const words = surface.modes.flatMap((mode) => [mode.name, ...mode.aliases])
  const planModes = surface.modes
    .filter((mode) => mode.positionals.includes(surface.positional_kinds.plan))
    .map((mode) => mode.name)
  const questionModes = surface.modes
    .filter((mode) =>
      mode.positionals.includes(surface.positional_kinds.question),
    )
    .map((mode) => mode.name)
  const reserved = [
    ...surface.reserved_flags.exact,
    ...surface.reserved_flags.named_exception,
    ...surface.reserved_flags.prefixes.map((prefix) => `${prefix}*`),
  ]
  return {
    mode: {
      type: "string",
      description:
        `One of: ${words.join(", ")}. An alias travels unexpanded — do not ` +
        `expand it and do not add the flag it already contributes. Never ` +
        `empty. The default when the user named no mode is ` +
        `${surface.default_mode}.`,
    },
    plan: {
      type: "string",
      description:
        `The plan path, as exactly one argv element — never split, never ` +
        `shell-quoted, never escaped. Strip at most one layer of matching ` +
        `quotes from what the user typed and send the value. A path ` +
        `containing spaces needs no escaping. Empty ("") when and only when ` +
        `the mode declares no plan positional. Modes taking a plan: ` +
        `${planModes.join(", ")}. Relative paths resolve against the ` +
        `workspace.`,
    },
    question: {
      type: "string",
      description:
        `The free-text question, as exactly one argv element — newlines ` +
        `preserved verbatim, no truncation, no escaping. Empty ("") for every ` +
        `mode except: ${questionModes.join(", ") || "(none)"}. A question ` +
        `whose first character is "-" is refused, because the dispatcher's ` +
        `grammar would read it as a flag and end the positional scan; a ` +
        `question merely containing a flag is fine.`,
    },
    extra_flags: {
      type: "array",
      items: { type: "string" },
      description:
        `Additional dispatcher flags, one token per array element — ` +
        `["--view", "run"], never ["--view run"]. Use [] for none. Passed ` +
        `verbatim, in order, never reordered or deduplicated. These are ` +
        `rejected because this tool composes them itself: ` +
        `${reserved.join(", ")} — including their abbreviations and "=" forms.`,
    },
  }
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
  for (const [name, value] of [["plan", plan], ["question", question]]) {
    if (typeof value !== "string") {
      return fail(`"${name}" must be a string, got ${typeName(value)}.`)
    }
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
    return fail(`mode "${row.name}" takes a question and "question" was empty.`)
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

  for (const token of extraFlags) {
    const hit = reservedHit(token, surface)
    if (!hit) continue
    if (surface.reserved_flags.named_exception.includes(hit)) {
      return fail(interactiveRefusal(hit, token))
    }
    return fail(
      `"${token}" is reserved: this tool composes ${hit} itself, so a call may ` +
        `not duplicate, contradict or restate it. Remove it from extra_flags.`,
    )
  }

  return { ok: true, mode: row }
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
  const raw = env.PATH || ""
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
    return fs.readFileSync(file, "utf8").replace(/\n+$/, "")
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
      argv.push(values.plan)
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

/** A plan path normalized against the anchor. Resolving is not quoting. */
function normalizePlan(plan, anchor) {
  if (!plan) return ""
  const expanded = expandHome(plan)
  return path.isAbsolute(expanded) ? expanded : path.resolve(anchor, expanded)
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
 * the shell-cap workaround would get transliterated back in.
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
async function runWorkflow({ row, values, surface, ctx, env, uv, anchor }) {
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
    }

    // S2/S3 — the poll loop. One blocking child per iteration; the JS side
    // invents no cadence of its own, because `--wait` already polls internally
    // and returns exit 2 while non-terminal. `--wait` is a store_true: the bare
    // flag, never a --timeout, so the duration is the dispatcher's own default.
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
    }

    // No argv is threaded in: `terminalResult` names the argv of the child it
    // was handed, which is the only argv that ran. Passing one alongside it
    // invites a caller to pass the wrong one — the poll's on a continuation's
    // result — on a path whose whole contract is that the result names what
    // ran.
    const terminal = terminalResult(poll, {
      anchor, surface, runId, logPath, polls, values: watched,
    })
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
    abort_command: { tool: TOOL_NAME, args: abortArgs(values, runId, surface) },
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
    abort_command: { tool: TOOL_NAME, args: abortArgs(values, runId, surface) },
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
    const payload = redactCommand(child.result ?? null, null, surface)
    return {
      ...base,
      state: "needs-human",
      ok: false,
      result: payload,
      // Lifted off the *redacted* payload, never off `child.result`: a copy
      // taken from the raw one would put back, in its own field, the very
      // command the field beside it just removed.
      error: payload?.error ?? null,
      output: humanOutput(child, null, surface),
      stderr: redactCommand(child.stderr, null, surface) || undefined,
      next: null, // following `next` stops here
      options: toolCallsFor(child, surface),
      nested_exec: nestedExecReason(child, surface),
      message:
        "The dispatcher needs a human. Its output is included as it was " +
        "rendered, except that every runnable command in it was replaced by a " +
        "pointer and spelled as a tool call in `options`, in the order it " +
        "appeared, and its result line is in `result` instead. Do not " +
        "paraphrase the output; to enact one of its commands, send back that " +
        "tool call rather than running anything yourself.",
    }
  }

  if (child.code === codes.nothing_to_do && child.result) {
    const rawNext = child.result.next
    // Exit 2 is emphatically *not* an error on this route: it is "nothing to
    // do / already complete". In the poll loop it means "keep polling" and
    // never reaches here; on a foreground child it is a legitimate terminal
    // answer, and reporting it as an error would tell a model a completed phase
    // had failed.
    return {
      ...base,
      state: child.result.state ?? "nothing-to-do",
      ok: true,
      result: redactCommand(child.result, rawNext, surface),
      artifacts: child.result.artifacts ?? [],
      next: nextAsToolCall(rawNext, surface),
      output: humanOutput(child, rawNext, surface),
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
  return {
    ...base,
    state: child.result.state ?? "done",
    ok: true,
    result: redactCommand(child.result, raw, surface),
    artifacts: child.result.artifacts ?? [],
    next: nextAsToolCall(raw, surface),
    output: humanOutput(child, raw, surface),
  }
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
function redactCommand(value, command, surface = null) {
  return scrubValue(
    value,
    commandForms(command),
    surface ? runnablePattern(surface) : null,
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
 */
function humanOutput(child, command, surface) {
  const text = redactCommand(child?.stdout, command, surface)
  const marker = surface?.dispatcher?.result_marker
  if (typeof text !== "string" || !text || !marker || !child?.result) {
    return text
  }
  return text
    .split("\n")
    .filter((line) => !line.trim().startsWith(marker))
    .join("\n")
    .trimEnd()
}

function scrubValue(value, commands, pattern) {
  if (typeof value === "string") {
    let once = value
    for (const form of commands) once = once.split(form).join(NEXT_MARKER)
    return pattern ? once.replace(pattern, RUNNABLE_MARKER) : once
  }
  if (Array.isArray(value)) {
    return value.map((item) => scrubValue(item, commands, pattern))
  }
  if (value && typeof value === "object") {
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      out[key] = scrubValue(item, commands, pattern)
    }
    return out
  }
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
  return new RegExp(
    `[^\\s"]*\\buv(?:\\.exe)?\\b\\s+run\\b[^\\n"]*${script}[^\\n"]*`,
    "g",
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
function toolCallsFor(child, surface) {
  if (!child || !surface) return []
  const raw = child.result?.next
  const commands = typeof raw === "string" && raw ? [raw] : []
  const captured = `${child.stdout || ""}\n${child.stderr || ""}`
  commands.push(...(captured.match(runnablePattern(surface)) ?? []))
  const seen = new Set()
  const calls = []
  for (const command of commands) {
    if (seen.has(command)) continue
    seen.add(command)
    calls.push(nextAsToolCall(command, surface))
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
  if (!parsed) {
    return {
      unparseable: true,
      note:
        "The dispatcher returned a follow-up this plugin could not parse into " +
        "a tool call. It is reported rather than executed. Do not run it in a " +
        "shell.",
    }
  }
  return { tool: TOOL_NAME, args: parsed, human: describeCall(parsed) }
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

function describeCall(args) {
  const parts = [args.mode]
  if (args.plan) parts.push(args.plan)
  if (args.extra_flags.length) parts.push(...args.extra_flags)
  return `call ${TOOL_NAME} with ${parts.join(" ")}`
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
  const options = toolCallsFor(child, surface)
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
        tool: TOOL_NAME,
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
 * `buildDescription`/`buildArgs` run inside the try for the reason the config
 * hook has one. `loadSurface` checks the schema *version* and nothing else, so a
 * payload that declares the right version with the wrong shape — `modes` not an
 * array, `reserved_flags` absent — throws a TypeError out of them. An
 * unregistered tool is a warned, degraded plugin; a throw out of the factory
 * takes OpenCode's whole plugin load with it.
 */
function buildToolHook(surface) {
  if (!surface) {
    warn(
      `the ${TOOL_NAME} tool is not registered: the generated CLI surface ` +
        `could not be read. The package looks partial; reinstall it.`,
    )
    return {}
  }
  try {
    return {
      [TOOL_NAME]: {
        description: buildDescription(surface),
        args: buildArgs(surface),
        // `ToolResult` is `string | { output, title?, metadata?, attachments? }`
        // and opencode reads a non-string result as exactly those four keys.
        // Handing back the raw envelope would therefore keep only its `output`
        // — the human stdout — and drop `state`, `ok`, `next`, `anchor` and
        // `result` on the floor, silently turning a structured contract into a
        // prose blob (and an error branch, which has no `output` at all, into
        // an `undefined` one). One JSON string keeps every field.
        execute: async (args, ctx) =>
          JSON.stringify(await execute(args, ctx, surface)),
      },
    }
  } catch (error) {
    warn(
      `the ${TOOL_NAME} tool is not registered: the generated CLI surface ` +
        `is not the shape this plugin build expects. The package looks ` +
        `partial; reinstall it. ${describe(error)}`,
    )
    return {}
  }
}

export const MultiLlmPlugin = async () => {
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

    tool: buildToolHook(surface),
  }
}


/** The whole contract, in one function. Returns; never throws. */
async function execute(args, ctx, surface) {
  try {
    const env = { ...process.env }
    const anchor = resolveAnchor(ctx)

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
      return {
        state: "needs-human",
        ok: false,
        anchor,
        message: uvMissingText(surface),
      }
    }
    // Propagate the binary the preflight approved, so the detached worker
    // re-resolves to it instead of re-probing a PATH it may not share.
    env[surface.uv.env] = uv

    const values = {
      // The word as sent, so an alias reaches the dispatcher unexpanded.
      word: args.mode,
      plan: normalizePlan(args.plan, anchor),
      question: args.question,
      extraFlags: args.extra_flags,
    }

    const outcome = await runWorkflow({
      row, values, surface, ctx, env, uv, anchor,
    })

    // The core instructions and the three resolved verb sentences, for any mode
    // where the model still has work after the dispatcher returns.
    if (outcome.ok && rowNeedsInstructions(row)) {
      const instructions = await fetchInstructions({
        row, surface, ctx, env, uv, anchor,
      })
      if (instructions) outcome.instructions = instructions
    }
    return outcome
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

/** Modes whose work continues after the dispatcher returns. */
function rowNeedsInstructions(row) {
  return row.supports_detach
}

// Exported for the test suite, which grades this file against its Python owners
// rather than against an expectation someone typed from memory.
const __testing = {
  PACKAGE_DIR,
  SKILL_DIR,
  buildArgs,
  buildDescription,
  composeDispatcherArgv,
  composeSpawnArgv,
  expandHome,
  findUv,
  flagMatches,
  injectCommand,
  loadSurface,
  nextAsToolCall,
  normalizePlan,
  parseDispatcherCommand,
  reservedHit,
  resolveAnchor,
  runChild,
  resolveMode,
  splitFrontmatter,
  splitPosix,
  uvMissingText,
  validateArgs,
  workedExamples,
}

// Hung on the factory rather than exported, because opencode calls every export
// of this module as a plugin factory and refuses the whole module over one that
// is not a function (rule 3). Function properties are not exports, so the test
// suite reaches these through a plain `import` and the loader never sees them.
MultiLlmPlugin.execute = execute
MultiLlmPlugin.__testing = __testing
