// Publish the npm CLI layer's executables into /opt/mllm/real (Dockerfile step 5).
//
// WHY NOT `node_modules/.bin`. npm links a package's `bin` entries when it
// unpacks the tarball, which is BEFORE that package's own postinstall runs — so a
// CLI whose launcher is *created* by its postinstall never gets a `.bin` link at
// all. Measured on this pin set: `cline@3.0.55` ships no `bin/cline` in its
// tarball and writes it (plus a 149 MB `bin/.cline` payload) from
// `postinstall.mjs`, so `node_modules/.bin/cline` is absent while the executable
// is right there. Reading each package's own `bin` map after the install is over
// is the resolution that does not depend on npm's internal ordering.
//
// It is also self-maintaining: the wanted names come from argv, the packages come
// from package.json, and a new CLI is one dependency plus one name.
//
// Every wanted name must resolve to a file that EXISTS, or this exits non-zero
// and fails the build. A missing CLI has to be a red build here rather than a
// `no real binary at /opt/mllm/real/<cli>` from the wrapper, four phases later,
// inside a container nobody can attach to.

import fs from "node:fs";
import path from "node:path";

const [projectDir, realDir, ...wanted] = process.argv.slice(2);
if (!projectDir || !realDir || wanted.length === 0) {
  console.error("usage: link_cli_bins.mjs <npm-project-dir> <real-dir> <binary>...");
  process.exit(2);
}

const modulesDir = path.join(projectDir, "node_modules");
const manifest = JSON.parse(
  fs.readFileSync(path.join(projectDir, "package.json"), "utf8"),
);
const want = new Set(wanted);
const linked = new Map();

for (const dep of Object.keys(manifest.dependencies ?? {})) {
  const packageJson = path.join(modulesDir, dep, "package.json");
  if (!fs.existsSync(packageJson)) {
    console.error(`no package.json for dependency ${dep}`);
    process.exit(1);
  }
  const pkg = JSON.parse(fs.readFileSync(packageJson, "utf8"));
  // `bin` is either a string (named after the package) or a name -> path map.
  const bins =
    typeof pkg.bin === "string"
      ? { [pkg.name.split("/").pop()]: pkg.bin }
      : pkg.bin ?? {};

  for (const [name, rel] of Object.entries(bins)) {
    if (!want.has(name)) continue;
    const target = path.resolve(path.join(modulesDir, dep, rel));
    if (!fs.existsSync(target)) {
      console.error(`${dep} declares bin ${name} -> ${rel}, which does not exist`);
      process.exit(1);
    }
    fs.chmodSync(target, 0o755);
    const link = path.join(realDir, name);
    fs.rmSync(link, { force: true });
    fs.symlinkSync(target, link);
    linked.set(name, target);
    console.log(`${link} -> ${target}   (${dep}@${pkg.version})`);
  }
}

const missing = [...want].filter((name) => !linked.has(name));
if (missing.length > 0) {
  console.error(
    `the npm layer produced no executable for: ${missing.join(", ")}`,
  );
  process.exit(1);
}
