# The Claude Code plugin tree

For maintainers. Users install through the marketplace, as
[the installation guide](../../docs/installation.md) describes.

`multi-llm/` beside this file is the plugin Claude Code installs. `build.py` generates it, and
it is checked in.

## What the tree is, and why it is checked in

`.claude-plugin/marketplace.json` names `./packaging/claude/multi-llm` as the plugin's `source`.
Installing copies that directory into the plugin cache with no build on the user's machine, so
the bytes committed here are the bytes a user receives. Every commit that changes a shipped file
therefore owes a regenerated tree in the same commit.

The repository root stays the authored source. Edit `.claude-plugin/plugin.json`, `hooks/` and
`skills/multi-llm/` where they are, never their copies here. `claude --plugin-dir <checkout>`
still loads the root directly during development, but that layout is not what users get.

| In the tree | Comes from |
| --- | --- |
| `.claude-plugin/plugin.json` | the root manifest, verbatim |
| `hooks/hooks.json`, `hooks/doctor.sh`, `hooks/report-check.sh`, `hooks/wakeup-check.sh` | the root `hooks/`, verbatim, with the mode git records |
| `LICENSE` | the repository root |
| `skills/multi-llm/` | the shared runtime allowlist, the product-under-test section of `skills/multi-llm/tests/container/context_manifest.txt` |

## Commands

| Command | What it does | When |
| --- | --- | --- |
| `make claude-regen` | Rewrites the tree from your working copy. It tolerates modified inputs and refuses untracked ones. | After editing a shipped file, before the commit. |
| `make claude-package` | Rebuilds from clean inputs only, and the result must equal the committed tree. | After the commit. |
| `make claude-package-check-ci` | The CI tier: `claude-package`, `claude-check-release`, then the plain and `package`-marked rows. No Docker, no `claude` binary, no registry. `ALLOW_DIRTY=1` swaps `claude-regen` in for a local run mid-edit. | The `package` job runs it on every pull request and every push to `master`. |
| `make claude-package-container` | The plugin as a user gets it, in the test image: the sparse registration and install, a fresh session's own account of what it loaded, the hooks really running, the published `/multi-llm:multi-llm` string, the dispatcher from the cache copy, and the documented update. Needs Docker. No key, no network, $0. | Before a release. `make e2e-container` runs it too. |
| `make claude-check-release` | The moved-tree rule below. Reads committed history only, so commit first. `CLAUDE_RELEASE_BASE=<ref>` also compares against that ref, as a pull request does against its target. | Before asking for a merge. |
| `make claude-check-bump` | The moved-tree rule plus the shared-version check over every `claude-v*`, `codex-v*` and `opencode-v*` tag and npm's version list. Reads the registry, and refuses a shallow or tagless clone. | Before publishing. Never in the `tests` workflow. |
| `make bump-version VERSION=x.y.z` | The bump procedure below in one step: the owner, every mirror, the capability docs and both regenerated trees. Plans every edit before writing any, refuses a mirror that does not already say the current version, and a version that is not newer than the current one and every release tag. `DRY_RUN=1` prints the edits only. Commits nothing. | With the change that needs a release, before the commit. |
| `make claude-hook-install` | Writes a local `pre-push` hook (never committed; `make claude-hook-uninstall` removes it). A push of `master` must then pass the moved-tree rule against the remote's `master` and a clean rebuild of both generated trees from the pushed commit, in a throwaway worktree. A `claude-v*` tag push must name its commit's version, on a commit the destination's `master` has (asked of the destination, not a tracking ref). Other pushes pass untouched. A push of `master` or a `claude-v*` tag it cannot check (no `release.py` in the checkout, no `uv`) is refused, not waved through; `git push --no-verify` skips it deliberately. | Once per clone. |
| `make claude-publish` | From `master`: fetches, runs the hook's checks plus `claude-check-bump`'s shared-version check, and prints the push and tag it would make. Everything about the remote is asked of the one URL the push goes to (`pushurl` when set; several are refused). `PUSH=1` then pushes `master` as a fast-forward, creates the `claude-v<version>` tag if needed and pushes it if the destination lacks it, so a rerun finishes a half-done release. Never forces. Does not run Docker. | To release. |

## Whole-plugin staging and verification

`payload.stage` covers `skills/multi-llm/` alone, which is not enough for a plugin whose manifest
and hooks live outside that directory. So `build()` assembles the complete plugin in a scratch
directory beside the destination and checks it as one tree. A failure at any step leaves the
previous complete plugin in place.

1. Refuse dirty inputs (`make claude-regen` tolerates modified ones, never untracked ones).
2. Stage the shared allowlist. The required inventory is the static list and every
   `target_module` in `modes.py`.
3. Copy the manifest, hooks and licence from their authored sources, with the mode the git index
   records rather than whatever the checkout's umask produced.
4. Sweep the complete plugin root for credentials. `payload.stage`'s own sweep saw only the
   payload.
5. Assert the result: exactly the expected files, both ways; the manifest's name, its version
   against `skills/multi-llm/pyproject.toml`, and its `hooks` pointer; every hook command
   resolving from `${CLAUDE_PLUGIN_ROOT}` to a file in the tree; each hook script executable;
   `SKILL.md` present; the marketplace entry naming this tree; the budget below.
6. Swap the finished tree into place.

## The two online documentation links

`SKILL.md` and `instructions/init-config.md` point at the user documentation by URL:

- `https://github.com/beastlabai/multi-llm-plugin/blob/master/docs/providers.md`
- `https://github.com/beastlabai/multi-llm-plugin/blob/master/docs/configuration.md`

`docs/` is in no host's payload. It is written to be read on the repository's site, a copy would
add the same bytes to all three distributions, and a repository-relative pointer resolves neither
in an installed plugin nor in a user's project. These two links are the explicit exception to the
rule that every runtime reference resolves locally. Reading them needs the network; running the
plugin does not. They are public link targets, so those two paths stay where they are.

## Budgets

`DISTRIBUTION_BUDGETS` in `packaging/common/payload.py` is the one place they live. The
`claude-code` entry allows 310 files and 11,500,000 unpacked bytes, and `build()` refuses a tree
over either before anything is swapped in. The tree measured 243 files and about 10.53 MB when
the file budget was last raised (2026-11-08, for the code-health module splits). An over-budget tree usually means development files reached the allowlist, so
find out what grew first. Raising the budget is a reviewed edit to that table.

The budget counts installed bytes. It says nothing about what a marketplace clone downloads.

## Bump procedure

`skills/multi-llm/pyproject.toml` owns the version. All of this goes in **one commit**.
`make bump-version VERSION=x.y.z` does steps 1 to 3; they are listed so that a failure
part-way can be finished by hand:

1. Set the new version in `skills/multi-llm/pyproject.toml`. It must be newer than every version
   any family has released.
2. Synchronize the mirrors:
   - `skills/multi-llm/uv.lock`: `uv lock --project skills/multi-llm`
   - the root `.claude-plugin/plugin.json`
   - `skills/multi-llm/hosts/capabilities.yaml`: the `opencode` entry's `package.version`
   - `packaging/opencode/package.json`
   - `packaging/opencode/README.md`: the pinned spec, in both places it appears
3. Regenerate, in this order:

   ```bash
   uv run --project skills/multi-llm -- python skills/multi-llm/hosts/render_capabilities.py
   make codex-regen
   make claude-regen
   ```

**A root manifest bump alone does nothing for users.** Claude Code reads the manifest inside the
installed tree and keys its cache on that version (`plugins/cache/beastlabai/multi-llm/<version>/`
under its config directory). A root edit with no `make claude-regen` updates neither the installed
manifest nor that cache directory, and anyone already holding the old version keeps it.

## Release procedure

The Claude release is the bumped `master` commit containing the regenerated tree. There is no
registry, no release branch and no workflow: a `claude-v*` tag triggers nothing.

1. Prepare the candidate on a branch, following the bump procedure, with the source changes and
   every regenerated mirror in the same commit.
2. Run `make claude-package-check-ci` and `make claude-package-container`. `make claude-publish`
   (step 4) runs `make claude-check-bump`'s checks itself, from a full clone with every tag.
3. Keep the candidate off `master` until it is verified as an installed plugin, a fresh install
   and an upgrade over a cached older version, under a throwaway `CLAUDE_CONFIG_DIR`. Verify again
   if the candidate changes.
4. Merge it into the local `master`, then run `make claude-publish` there and read its plan.
   `make claude-publish PUSH=1` pushes `master`. **That push is the release**: the next
   `marketplace update` serves it. A merge that rewrites the commit produces a different
   candidate, so it needs verifying again.
5. Record that exact commit, as part of publishing it. `make claude-publish PUSH=1` does this
   right after the push; by hand it is:

   ```bash
   git tag -a claude-v<version> <commit> -m "Claude Code plugin <version>"
   git push origin claude-v<version>
   ```

   The tag is immutable. It is never moved or deleted, and a mistake in a release is fixed by a new
   version. The moved-tree rule compares against the newest tag, and the shared-version check holds
   later Codex and OpenCode releases of this version to its shared bytes, so a commit that is
   released but untagged is invisible to both.

The Codex and OpenCode release procedures are unchanged: they keep their own tags, gates and
promotion. A family may release a version the others have not, but a version one family has
released names one shared payload for all three.

## The exception to the post-release policy

After a Codex or OpenCode release, `master` may change shipped bytes while still saying the
released version (the post-release policy in `packaging/common/release_history.py`). Those users
install from the `release` ref or a pinned npm version, so such an edit is unreleased development.

Claude Code installs follow `master` and cache by version, so here a same-version edit is not
harmless. Fresh installs would get the new bytes and everyone holding that version would keep the
old ones, under one version number.

**The consequence: after the first Claude tag, any commit that moves `packaging/claude/multi-llm/`
under an unmoved version fails the check, so unfinished payload changes stay off `master` until
ready to release.** Nearly every runtime edit moves that tree, so a pull request that changes a
shipped file also carries a bump and is itself a release. Pull requests are held to this from the
moment the tree is on `master`, tag or no tag, because the check also compares with their target.

The rule is `packaging/claude/build.py --check-release`, and it runs on pull requests in the
`package` job. It compares git tree ids, so it covers the whole tree: hooks and the executable
bit, neither of which the shared-payload digest sees. The version must be strictly newer
than the compared commit's, and build metadata does not count. The shared-version check itself
stays out of the `tests` workflow.

A push to `master` is compared with the commit `master` was at before it, so a same-version edit
pushed directly on top of a bump whose tag is not cut yet is still refused. That refusal comes after
the push has landed, so run `make claude-check-release` first, or use a pull request. After a force
push the earlier commit is not fetched, and the check refuses by name for that one run.

`make claude-hook-install` moves that refusal before the push: the hook runs the same rule against
the remote's `master`, and also rebuilds both generated trees from the pushed commit, which the
rule alone cannot see (a source edit committed without `make claude-regen` moves nothing and
delivers nothing). The hook is local to each clone, and `--no-verify` skips it, so CI stays the
backstop.
