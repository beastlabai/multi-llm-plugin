# `codex plugin list --json` fixture — recorded

`plugin-list.json` is what `tests/test_codex_route.py` replays through a fake
`codex` binary to drive `host_install._codex_route()`'s step 1 (D5).

**Provenance.** Recorded verbatim, not written by hand: it is the
`codex plugin list --json` document Phase 1's zero-quota container row 5
(`codex-package-seed`) captured on codex-cli 0.156.1 in the seeded home, run
`20260923T191255Z` — `tests/container/runs/20260923T191255Z/codex-package-seed/codex-plugin-list.json`,
graded by `tests/container/records/20260923T191255Z-codex-package-seed.json`
(a local run record, not in the repo).
Nothing is normalised: the paths are the image's (`/opt/mllm/codex-marketplace`),
and the resolver reads none of them.

**What the shape says.** A top-level `{"installed": [...], "available": [...]}`
object; `available` is empty. The `multi-llm@beastlabai` entry carries
`pluginId`, `name`, `marketplaceName`, `version`, `installed`, `enabled`,
`installPolicy`, `authPolicy`, a `source` object (`{"source": "local", "path"}`
on a local-marketplace install) **and** a `marketplaceSource` object
(`{"sourceType": "local", "source"}`). There is **no `installedPath`** on a
list entry (R25/R40): the resolver computes the cache directory from
`marketplaceName` and `version` and stats it, and reads neither source object.

It replaced a provisional stand-in (a hand-written `{source: "git", id}` entry
beside a real curated-remote row) that predated the container capture. Rows
that need another entry — a catalog-only `available` row, a second marketplace —
add it to a copy of this document in the test, never to this file.
