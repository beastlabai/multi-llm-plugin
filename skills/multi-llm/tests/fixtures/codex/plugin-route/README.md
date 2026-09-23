# `codex plugin list --json` fixture — PROVISIONAL

`plugin-list.json` is what `tests/test_codex_route.py` replays through a fake
`codex` binary to drive `host_install._codex_route()`'s step 1 (D5).

**Provenance, precisely.** The document's *shape* — the top-level
`{"installed": [...], "available": [...]}` object and the per-entry keys
`pluginId`, `name`, `marketplaceName`, `version`, `installed`, `enabled`,
`source`, `installPolicy`, `authPolicy` — was read off a real
`codex plugin list --json` on codex-cli 0.156.1 (2026-09-23, a developer
machine, the curated remote plugins it had installed by default; the first
entry is one of them, verbatim). The `multi-llm@beastlabai` entry and the
`available` row are **written by hand** in that shape: no `multi-llm` plugin
was installed when it was captured.

Note that 0.156.1 spells the source object `source`, where R25/R40 (0.155.1)
recorded `marketplaceSource`; the resolver reads neither.

**Replace it** with the document Phase 1's container package row records
(`codex plugin list --json` in the seeded home, beside its `plugin add
--json`) the day that row commits one — that capture is the authority, and
this file is only a stand-in until it exists.
