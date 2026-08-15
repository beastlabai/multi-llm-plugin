"""Reduce a wrapper call-log record to a comparable flag view (C8).

**What this is coverage of, stated plainly: our own `build_command()`.** Both
sides of the comparison originate in this repository, so it cannot detect an
upstream change — if a CLI renames a flag, the wrapper keeps logging the argv
*we* generate, byte-identical to the expected table, and the assertion stays
green while the real binary rejects the call. The upstream half is
`cli-surface.yaml`'s help probe. Read this as a regression guard: a refactor that
drops `--json`, reorders a positional, loses a separator token or leaks a stray
flag fails here at the argv instead of inside a decoded stream three layers down.

**A full-argv golden would be unusable and would not be drift detection either.**
The prompt positional is routinely tens of KB of plan text, and the rest of the
record is volatile by construction: the model ids a row pins, the per-invocation
`$HOME` and config roots, the sample project's absolute root, the phase dir. Such
a table would churn on every fixture edit, every mount-point change and every
added row while proving nothing about `build_command()`.

So a record reduces to four things:

* **the binary's basename** — `/opt/mllm/real/codex` → `codex`; the wrapper tree's
  layout is not part of any contract;
* **the subcommand chain** — leading non-flag tokens before the first `-`-prefixed
  one, verbatim and in order (`exec`, `run`, `login status`);
* **the option set** — one entry per flag, compared as a set, each value
  *classified* rather than copied. **Policy values are kept verbatim, because they
  are the assertion**: `--sandbox workspace-write`, `--format json`, `--json`,
  `--agent build`, `--permission-mode acceptEdits`, `--log-level ERROR`. Values
  derived from the row's own inputs become a typed placeholder — `--model
  <model>`, `-c model_reasoning_effort=<effort>`, any token resolving to a path
  under the tmpfs roots → `<path>`. A flag that legitimately repeats (codex's
  `-c`) collects its normalized values into a list, so a lost repetition still
  fails;
* **the positional tail** — the prompt collapses to `<prompt>`, and bare separator
  tokens (`--`) survive as themselves, since dropping or misplacing one is exactly
  the class of bug this exists to catch.

Order matters inside the subcommand chain and the tail; two independent flags
trading places is not a regression, a missing flag or a changed policy value is.
Redaction already happened at write time, so normalization never re-reads a
credential.
"""

from __future__ import annotations

import re

__all__ = ["normalize_argv", "PLACEHOLDER_PROMPT", "PLACEHOLDER_PATH"]

PLACEHOLDER_PROMPT = "<prompt>"
PLACEHOLDER_PATH = "<path>"
PLACEHOLDER_MODEL = "<model>"

#: Already collapsed by the wrapper at write time.
_PROMPT_RE = re.compile(r"^<prompt:sha256:[0-9a-f]+:len=\d+>$")
_REDACTED_RE = re.compile(r"^<redacted:sha256:[0-9a-f]+>$")

#: Anything under a run-time root is per-invocation and cannot be compared.
_PATH_PREFIXES = ("/work/", "/home/e2e", "/tmp/", "/opt/mllm/skill/",
                  "/run/creds/")

#: Flags whose value is drawn from the row rather than from policy, each mapped to
#: a placeholder naming WHAT it is. A single `<value>` would compare equal across
#: `--model` and `--base-url`, so a refactor that passed the base URL to `--model`
#: would normalize identically and pass.
_ROW_DERIVED = {
    "--model": PLACEHOLDER_MODEL,
    "-m": PLACEHOLDER_MODEL,
    "--provider": "<provider>",
    "-P": "<provider>",
    "--base-url": "<base-url>",
    "--endpoint": "<base-url>",
    "-e": "<base-url>",
}

#: `key=value` config overrides whose VALUE is row-derived but whose KEY is the
#: assertion (codex's `-c model_reasoning_effort=high`).
_KV_ROW_DERIVED_KEYS = {"model_reasoning_effort", "model", "base_url"}


def _classify(token: str) -> str:
    """One argv value → the thing worth comparing."""
    if _PROMPT_RE.match(token):
        return PLACEHOLDER_PROMPT
    if _REDACTED_RE.match(token):
        return "<redacted>"
    if token.startswith(_PATH_PREFIXES) or token.startswith("/tmp"):
        return PLACEHOLDER_PATH
    if "=" in token:
        key, _, value = token.partition("=")
        if key in _KV_ROW_DERIVED_KEYS:
            return f"{key}=<{key.split('_')[-1]}>"
        return f"{key}={_classify(value)}" if value else token
    return token


def normalize_argv(argv: list[str]) -> dict:
    """``{binary, subcommands, options, tail}`` for one call-log record."""
    if not argv:
        return {"binary": None, "subcommands": [], "options": {}, "tail": []}

    binary = argv[0].rsplit("/", 1)[-1]
    rest = list(argv[1:])

    subcommands: list[str] = []
    index = 0
    while index < len(rest) and not rest[index].startswith("-"):
        subcommands.append(rest[index])
        index += 1

    options: dict[str, object] = {}
    #: Flags whose repeats all normalized to the same value. Tracked beside
    #: `options` rather than encoded into it, so the stored value stays the bare
    #: placeholder — see the render step below.
    collapsed: set[str] = set()
    tail: list[str] = []
    while index < len(rest):
        token = rest[index]
        if token == "--":
            # A bare separator survives as itself: dropping or misplacing one is
            # precisely the bug this check exists to catch (opencode's yargs).
            tail.append("--")
            index += 1
            continue
        if not token.startswith("-"):
            tail.append(_classify(token))
            index += 1
            continue

        if "=" in token and token.startswith("--"):
            flag, _, value = token.partition("=")
            normalized = _ROW_DERIVED.get(flag) or _classify(value)
        else:
            flag = token
            nxt = rest[index + 1] if index + 1 < len(rest) else None
            if nxt is not None and not nxt.startswith("-"):
                normalized = _ROW_DERIVED.get(flag) or _classify(nxt)
                index += 1
            else:
                normalized = None      # valueless flag
        index += 1

        if flag in options:
            # A repeatable flag collects into a list, so a lost repetition fails
            # — but only where the COUNT is a property of command construction.
            #
            # codex's `-c model_reasoning_effort=…` repeats because we chose to
            # pass another override: the count is ours, and losing one is a
            # regression worth failing on.
            #
            # aider's `--read` repeats once per ABSOLUTE PATH MENTIONED IN THE
            # PROMPT (its mention auto-add never matches an absolute path), so
            # the count is a property of the fixture text, not of
            # `build_command()`. Asserting it would make this table churn on
            # every prompt edit while proving nothing. So a repeat whose values
            # all normalize to the SAME placeholder collapses to `<x>+`, which
            # still distinguishes "absent" from "present one or more times".
            #
            # `<x>+` is COUNT-INDEPENDENT, so the marker is never stored in the
            # value: a stored `<path>+` read back as an ordinary value compares
            # distinct against the next `<path>`, and the third occurrence would
            # degrade the collapse into a two-element list — the normalized view
            # would then vary with the repetition count the collapse exists to
            # hide, and `--read` would churn on the fixture's path count.
            existing = options[flag]
            values = ([*existing] if isinstance(existing, list) else [existing])
            values.append(normalized)
            distinct = set(values)
            if len(distinct) == 1 and isinstance(values[0], str):
                options[flag] = values[0]
                collapsed.add(flag)
            else:
                options[flag] = values
                collapsed.discard(flag)
        else:
            options[flag] = normalized

    rendered = {flag: (f"{value}+" if flag in collapsed else value)
                for flag, value in options.items()}

    return {"binary": binary, "subcommands": subcommands,
            "options": rendered, "tail": tail}
