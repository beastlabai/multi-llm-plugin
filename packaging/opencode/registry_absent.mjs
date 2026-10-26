#!/usr/bin/env node
/**
 * Row 9 — `opencode-multi-llm@<version>` must be ABSENT from the single
 * registry. The one implementation every row-9 site calls: the full gate
 * (`make opencode-package-check`, through `make opencode-registry-absent`), the
 * workflow's `gate`, `verify` and `publish` jobs, and the laptop first-release
 * target (`first_release.py`) immediately before upload.
 *
 *     node packaging/opencode/registry_absent.mjs --registry <url>
 *         [--package-json <path>] [--out <json>]
 *
 * WHY ONE FILE. Row 9 used to be two hand-maintained copies, a shell recipe in
 * the Makefile and a step in the publishing workflow, each deciding on npm's
 * error code with its own inline parser. A copy that drifts into accepting a
 * non-`E404` unknown is exactly the failure this row exists to prevent, and two
 * copies are one more place for that drift than one. The tiers still run
 * separately (CI consumes no stamp); they now share the script, not only its
 * reasoning.
 *
 * THREE OUTCOMES, NOT TWO, and this is the whole reason the probe is not a bare
 * `if npm view ...`:
 *
 *   * npm exits 0 — the version is there. `published`: refuse. npm forbids
 *     republishing a version, so a botched release costs a number permanently.
 *   * npm reports an `.error.code` of exactly `E404` — `absent`, the one green
 *     answer, and also the right green for a name never published at all.
 *   * anything else — a network failure, an `E429`, a registry outage, an
 *     `E401`/`ENEEDAUTH`, a killed probe, output that is not npm's JSON —
 *     `unknown`: refuse. Every one of those also exits non-zero, so read as a
 *     two-way test they would all come back "absent" and the gate would pass on
 *     a probe that never answered.
 *
 * `--json` puts the code on STDOUT as `.error.code`. Measured 2026-10-25 on npm
 * 10.9.3 and 11.5.1 (the publish job's pin) for `E404`, `E401`, `ECONNREFUSED`
 * and `ENOTFOUND`; stderr carries only the human-readable `npm error` lines and
 * is never parsed. `E404` has two shapes: a name never published says `Not
 * Found - GET <url> - Not found`, and a missing version of a package that
 * exists (every release after the first) says `No match found for version
 * <v>` (measured with `npm view express@0.0.0 version --json`, both npms; only
 * the `detail` wording differs). Never `npm publish --dry-run`, which does not
 * ask.
 *
 * ONE REGISTRY, CHECKED, NEVER INHERITED. npm's registry is configurable — a
 * user `.npmrc`, `npm_config_registry`, a scoped registry — and an inherited
 * setting could otherwise make this row accept `E404` from a different server
 * than the one publication targets. So:
 *
 *   * `--registry` is required, and it must equal the Makefile's
 *     `OPENCODE_REGISTRY` line, parsed here with the same strict rule
 *     `registry.py` applies (exactly one `OPENCODE_REGISTRY := <https url>`
 *     line). The Makefile stays the only place the URL is written; this script
 *     reads it rather than taking a second `--expected-registry` it would have
 *     to trust.
 *   * It is passed to npm explicitly, which outranks both the environment and
 *     every `.npmrc` (measured against two localhost stubs, and pinned by
 *     `test_opencode_registry_absent.py`). A scoped name would escape that —
 *     `@scope:registry` outranks `--registry` — so a scoped name is refused.
 *   * When npm's `E404` names the URL it fetched (`Not Found - GET <url>`), that
 *     URL must lie under the single registry, or the answer is `unknown`. Only
 *     the whole-package shape names one: once the package exists, a missing
 *     VERSION is `E404` with `No match found for version <v>` and no URL at
 *     all (measured, above), so that binding cannot apply there.
 *
 * NOTHING OF THE CALLER'S npm CONFIGURATION. `--registry` alone does not make
 * the answer the single registry's: a user or global `.npmrc` can still carry
 * a `proxy`, an `https-proxy`, a `cafile` or `strict-ssl=false`, and any of
 * them can route the request to, or accept a certificate from, a server that
 * is not the registry, whose `E404` would then pass for an absence. So npm
 * gets an empty `--userconfig` and an empty `--globalconfig` (command-line
 * flags, which outrank every file), and the child's environment drops every
 * inherited `npm_config_*` variable, which would otherwise outrank those empty
 * files. The standard proxy variables (`HTTPS_PROXY` …) still pass: they say
 * how this machine reaches the network at all, as row 7's isolation keeps them.
 *
 * NOR A PROJECT'S. npm has a third file layer, the project's `.npmrc`, and it
 * outranks both empty files. npm finds the project by walking up from its cwd
 * to the first directory holding a `package.json` or a `node_modules`, and
 * falls back to the cwd itself, so the caller's directory (the repository root
 * under `make`), any ancestor of it, and — now that the user file is moved —
 * a home holding `~/node_modules` or `~/package.json` each had their `.npmrc`
 * read as project config (npm skips `~/.npmrc` there only while it IS the user
 * file). Reviewed and reproduced: a cwd `.npmrc` naming a proxy that answers
 * 404 to everything turned row 9 green. So npm runs in the scratch directory
 * below, which holds neither, with `--prefix=<that directory>`. Measured on
 * npm 10.9.3 and 11.5.1, alike: a command-line `--prefix` IS the project root
 * (npm skips the walk entirely), while a cwd alone is not enough — from an
 * empty scratch dir npm still walks up, so a `package.json` and `.npmrc` above
 * it (a `TMPDIR` inside a checkout) were still read. Both are passed: the
 * flag pins the root, the cwd keeps nothing else npm resolves against the
 * caller's directory.
 *
 * BOUNDED. A fresh `--cache` per probe (no packument served from an earlier
 * read, nothing written to the caller's cache), `--fetch-retries 2` with short
 * back-offs, a 30 s `--fetch-timeout`, and a hard wall-clock kill on the child,
 * so an offline or black-holed registry ends in `unknown` instead of a hang.
 *
 * EVIDENCE. Prints the registry URL and npm's code. With `--out`, writes the
 * row-9 evidence (`{row, pass, registry, name, version, npm_exit, code,
 * outcome}`, the contract's shape) on every outcome, and removes any file
 * already at that path first, so a stale green record can never outlive a
 * refusal. Exit 0 iff `absent`; 1 for `published` or `unknown`; 2 for a usage
 * error.
 *
 * Node built-ins only, so the `publish` job's third-party code stays the
 * builder's locked runtime dependencies. Linux is the release path's platform;
 * on Windows `npm` is a `.cmd` shim that Node will not spawn without a shell,
 * and this script does not open one.
 */

import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "..", "..");
const MAKEFILE = join(REPO_ROOT, "Makefile");
const DEFAULT_MANIFEST = join(HERE, "package.json");

// registry.py's rule, restated: the suite runs both parsers over the same
// Makefiles and requires the same answer. `[ \t]` rather than `\s`, so a
// definition can never borrow its value from the next line.
const DEFINITION = /^OPENCODE_REGISTRY[ \t]*:=[ \t]*(\S+)[ \t]*$/gm;
const URL_RULE = /^https:\/\/[A-Za-z0-9.-]+(?::\d+)?\/?$/;

// npm's own name rule for an unscoped package, and a semver version. Checked
// before either reaches npm's argv, so a manifest can never smuggle a flag in.
const NAME_RULE = /^[a-z0-9~-][a-z0-9._~-]*$/;
const VERSION_RULE = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

// Every npm call this script makes is bounded by these. The kill is the outer
// bound: three attempts at the fetch timeout plus two capped back-offs fit
// inside it, so npm normally reports its own code before the kill fires.
const FETCH_FLAGS = [
  "--fetch-retries=2",
  "--fetch-retry-mintimeout=2000",
  "--fetch-retry-maxtimeout=10000",
  "--fetch-timeout=30000",
];
const KILL_AFTER_MS = 150_000;

const USAGE =
  "usage: node packaging/opencode/registry_absent.mjs --registry <url> " +
  "[--package-json <path>] [--out <json>]";

class Refusal extends Error {}

function parseArgs(argv) {
  const args = { registry: null, packageJson: DEFAULT_MANIFEST, out: null };
  const take = (i, flag) => {
    if (i + 1 >= argv.length) throw new Refusal(`${flag} needs a value\n${USAGE}`);
    return argv[i + 1];
  };
  for (let i = 0; i < argv.length; i += 1) {
    const raw = argv[i];
    const eq = raw.indexOf("=");
    const flag = raw.startsWith("--") && eq > 0 ? raw.slice(0, eq) : raw;
    const inline = raw.startsWith("--") && eq > 0 ? raw.slice(eq + 1) : null;
    const value = () => {
      if (inline !== null) return inline;
      const v = take(i, flag);
      i += 1;
      return v;
    };
    if (flag === "--registry") args.registry = value();
    else if (flag === "--package-json") args.packageJson = value();
    else if (flag === "--out") args.out = value();
    else if (flag === "-h" || flag === "--help") {
      process.stdout.write(`${USAGE}\n`);
      process.exit(0);
    } else throw new Refusal(`unknown argument ${JSON.stringify(raw)}\n${USAGE}`);
  }
  if (!args.registry) throw new Refusal(`--registry is required\n${USAGE}`);
  return args;
}

// registry.py's `normalize`, restated: ONE trailing slash removed, nothing
// more forgiving. Surrounding whitespace or a second slash stays, compares
// unequal to the single registry, and is refused as another registry.
function normalize(url) {
  const text = String(url);
  return text.endsWith("/") ? text.slice(0, -1) : text;
}

/** The URL the Makefile defines, without a trailing slash — or a refusal. */
export function singleRegistry(makefile = MAKEFILE) {
  const text = readFileSync(makefile, "utf8");
  const found = [...text.matchAll(DEFINITION)].map((m) => m[1]);
  if (found.length !== 1) {
    throw new Error(
      `${makefile}: expected exactly one \`OPENCODE_REGISTRY := <url>\` line, found ${found.length}`,
    );
  }
  if (!URL_RULE.test(found[0])) {
    throw new Error(`${makefile}: OPENCODE_REGISTRY ${JSON.stringify(found[0])} is not an https origin`);
  }
  return normalize(found[0]);
}

function readManifest(path) {
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(`${path}: cannot read the package manifest (${error.message})`);
  }
  const { name, version } = manifest ?? {};
  if (typeof name !== "string" || !NAME_RULE.test(name)) {
    throw new Error(
      `${path}: name ${JSON.stringify(name)} is not an unscoped npm name. A scoped ` +
        "name would let an `@scope:registry` setting outrank --registry, so row 9 refuses it.",
    );
  }
  if (typeof version !== "string" || !VERSION_RULE.test(version)) {
    throw new Error(`${path}: version ${JSON.stringify(version)} is not a semver version`);
  }
  return { name, version };
}

/** npm's argv for the probe. Exported for nothing but the suite's reading. */
export function npmArgv(name, version, registry, { cache, userconfig, globalconfig, prefix }) {
  return [
    "view", `${name}@${version}`, "version", "--json",
    `--registry=${registry}`,
    `--cache=${cache}`,
    `--userconfig=${userconfig}`,
    `--globalconfig=${globalconfig}`,
    `--prefix=${prefix}`,
    "--prefer-online",
    "--no-update-notifier",
    ...FETCH_FLAGS,
  ];
}

/**
 * Decide the outcome from what npm did. Pure, so the three-way rule has one
 * spelling: `published` on exit 0, `absent` on exactly `E404`, `unknown` on
 * everything else. An `E404` that names the URL it fetched must name one under
 * the single registry; only the whole-package shape does (`Not Found - GET
 * <url>`, a name never published). A missing version of a package that exists
 * is `No match found for version <v>`, with no URL to check, and is accepted
 * on the code alone: which server answered is what the argv and the empty
 * config layers above settle, not this summary.
 */
export function classify({ status, stdout, registry }) {
  if (status === 0) {
    return { outcome: "published", code: null, why: `npm exited 0 and reported ${JSON.stringify(String(stdout ?? "").trim())}` };
  }
  let parsed = null;
  try {
    parsed = JSON.parse(String(stdout ?? ""));
  } catch {
    return { outcome: "unknown", code: null, why: "npm's output is not the JSON `--json` promises" };
  }
  const code = parsed && typeof parsed === "object" && parsed.error && typeof parsed.error.code === "string"
    ? parsed.error.code
    : null;
  if (code !== "E404") {
    return { outcome: "unknown", code, why: `npm reported code ${code ?? "<none>"}; only E404 is a green answer` };
  }
  const summary = typeof parsed.error.summary === "string" ? parsed.error.summary : "";
  // The whole-package shape only. The later-release shape (`No match found
  // for version X`) carries no `GET`, so nothing here binds it to a server.
  const fetched = /\bGET\s+(\S+)/.exec(summary);
  if (fetched && !fetched[1].startsWith(`${registry}/`)) {
    return {
      outcome: "unknown",
      code,
      why: `the E404 came from ${fetched[1]}, not from the single registry ${registry}`,
    };
  }
  return { outcome: "absent", code, why: "E404 from the single registry" };
}

function writeEvidence(out, evidence) {
  if (!out) return;
  mkdirSync(dirname(resolve(out)), { recursive: true });
  writeFileSync(out, `${JSON.stringify(evidence, null, 2)}\n`, "utf8");
}

function main(argv) {
  let args;
  try {
    args = parseArgs(argv);
  } catch (error) {
    process.stderr.write(`REFUSING (row 9): ${error.message}\n`);
    return 2;
  }
  if (args.out) rmSync(args.out, { force: true });

  const given = normalize(args.registry);
  const evidence = {
    row: "9", pass: false, registry: given, name: null, version: null,
    npm_exit: null, code: null, outcome: "unknown",
  };

  let registry;
  try {
    registry = singleRegistry();
    ({ name: evidence.name, version: evidence.version } = readManifest(args.packageJson));
  } catch (error) {
    writeEvidence(args.out, evidence);
    process.stderr.write(`REFUSING (row 9): ${error.message}\n`);
    return 1;
  }
  if (given !== registry) {
    writeEvidence(args.out, evidence);
    process.stderr.write(
      `REFUSING (row 9): registry ${args.registry} is not the release path's single ` +
        `registry ${registry} (Makefile OPENCODE_REGISTRY).\n`,
    );
    return 1;
  }

  const spec = `${evidence.name}@${evidence.version}`;
  // One fresh directory: npm's cwd and project root, its cache and the two
  // empty config files, removed together whatever npm did. Nothing here is a
  // `package.json` or a `node_modules`, and no `.npmrc` is written, so the
  // project layer `--prefix` pins it to is empty too.
  const scratch = mkdtempSync(join(tmpdir(), "mllm-row9-npm-"));
  const paths = {
    cache: join(scratch, "cache"),
    userconfig: join(scratch, "npmrc-user"),
    globalconfig: join(scratch, "npmrc-global"),
    prefix: scratch,
  };
  const env = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.toLowerCase().startsWith("npm_config_")),
  );
  let child;
  try {
    writeFileSync(paths.userconfig, "", "utf8");
    writeFileSync(paths.globalconfig, "", "utf8");
    child = spawnSync("npm", npmArgv(evidence.name, evidence.version, registry, paths), {
      encoding: "utf8",
      cwd: scratch,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: KILL_AFTER_MS,
      killSignal: "SIGKILL",
      maxBuffer: 16 * 1024 * 1024,
    });
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }

  let decided;
  if (child.error) {
    const reason = child.error.code === "ETIMEDOUT"
      ? `npm did not answer within ${KILL_AFTER_MS / 1000} s and was killed`
      : `npm could not be run (${child.error.message})`;
    decided = { outcome: "unknown", code: null, why: reason };
  } else if (child.status === null) {
    decided = { outcome: "unknown", code: null, why: `npm was killed by ${child.signal}` };
  } else {
    decided = classify({ status: child.status, stdout: child.stdout, registry });
  }

  evidence.npm_exit = child.status ?? null;
  evidence.code = decided.code;
  evidence.outcome = decided.outcome;
  evidence.pass = decided.outcome === "absent";
  writeEvidence(args.out, evidence);

  process.stdout.write(
    `row 9: registry ${registry} answered code ${decided.code ?? "<none>"} ` +
      `(npm exit ${evidence.npm_exit ?? "<none>"}) for ${spec}: ${decided.outcome}\n`,
  );
  if (decided.outcome === "absent") {
    process.stdout.write(`row 9: ${spec} is absent from ${registry} (E404)\n`);
    return 0;
  }
  if (decided.outcome === "published") {
    process.stderr.write(
      `REFUSING (row 9): ${spec} is already on ${registry} (${decided.why}). npm forbids ` +
        "republishing a version: bump packaging/opencode/package.json.\n",
    );
    return 1;
  }
  process.stderr.write(
    `REFUSING (row 9): could not determine whether ${spec} is on ${registry}: ${decided.why}. ` +
      "That is an UNKNOWN, not an absence; re-run once the registry answers.\n",
  );
  const body = String(child.stdout ?? "").trim();
  if (body) process.stderr.write(`${body}\n`);
  return 1;
}

// Imported by nothing at runtime; the guard keeps `import()` side-effect free
// for a reader that wants `classify` or `npmArgv` without running a probe.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
