"""Kilo Code CLI provider implementation."""
import json
import re
import shutil
from typing import Any, Dict, List, Optional

from ..json_extractor import extract_json_from_text
from .base import (
    REASONING_LADDER,
    LLMProvider,
    ReasoningLadder,
    StructuredSchema,
    split_reasoning_effort,
)

# Reasoning variant names accepted by `kilocode run --variant` (verified on
# kilocode 7.4.1). Variants are model-specific: effort words for
# OpenAI-family/glm models, "thinking"/"instant" for kimi/minimax; unknown
# variants are silently ignored by the CLI.
#
# `thinking`/`instant` are the second axis, not two more rungs — they select a
# MODE, so they are declared as `variants` and pass through without ever being
# compared to (or coerced onto) a rung.
#
# No `model_caps` here on purpose. kilocode's per-model narrowing is *captured*
# rather than hand-written — `tests/fixtures/kilocode/model_variants.json`, from
# `kilocode models <backend> --verbose` — and `tests/test_providers.py` already
# grades every configured spec against it. Restating a slice of that capture
# here would give the same fact two owners and one of them no refresh command.
EFFORT_LADDER = ReasoningLadder(
    provider="kilocode",
    rungs=REASONING_LADDER,
    variants=("instant", "thinking"),
)
REASONING_EFFORTS = EFFORT_LADDER.accepted

#: The two names the Kilo Code CLI installs under, in preference order. Upstream
#: ships both spellings; this adapter used to know only the first, so on a
#: machine where the binary landed as `kilo` the provider was not merely degraded
#: — `is_available()` said False and it was invisible to the fan-out. Order is
#: load-bearing: where both resolve, `kilocode` wins, which is what keeps today's
#: argv byte-identical everywhere the canonical name is installed (the container
#: image installs `@kilocode/cli` as `binary: kilocode`).
BINARY_NAMES = ("kilocode", "kilo")

#: What `build_command` emits when neither name resolves. `is_available()` is
#: False in that case, so the provider is filtered out of the fan-out before any
#: spawn and this value never reaches a subprocess — but `build_command` is still
#: called on a binary-free host by the argv-contract tests, the housekeeping
#: `MODELS` table and `seed_expected_argv.py`, all of which pin the canonical
#: spelling. Falling back to it is what makes the change a pure no-op wherever a
#: `kilo`-only install is not in play.
CANONICAL_BINARY = BINARY_NAMES[0]


def _resolve_binary() -> str:
    """The Kilo Code binary name to place at ``argv[0]``.

    Returns the first of :data:`BINARY_NAMES` that ``shutil.which`` resolves, so
    the argv names a binary that is really there rather than a literal this
    module hardcoded. With neither installed it returns
    :data:`CANONICAL_BINARY` — see that constant for why the nothing-installed
    case has a value at all.
    """
    for name in BINARY_NAMES:
        if shutil.which(name) is not None:
            return name
    return CANONICAL_BINARY


class KiloCodeProvider(LLMProvider):
    """Provider for Kilo Code CLI tool.

    Kilo Code uses `run --auto` for non-interactive mode with plain text output.
    The --json flag is avoided because it streams every reasoning token as a
    full-content-so-far JSON line, producing massive output (~2.6MB for ~50KB
    of actual content). Model selection is via `-m provider/model` flag.

    Model strings support an optional ``model[:effort]`` suffix (e.g.
    ``openai-native/gpt-5.5:high``), translated to ``--variant <effort>``.
    Valid efforts are listed in REASONING_EFFORTS; anything else passes
    through verbatim as the model name (keeping ``:free``-style ids intact).

    The CLI ships under **two** binary names, ``kilocode`` and ``kilo``
    (:data:`BINARY_NAMES`). Both ``is_available()`` and ``build_command`` go
    through :func:`_resolve_binary`, so a ``kilo``-only install is a working
    provider rather than an invisible one, and ``argv[0]`` always names a binary
    the resolver actually found.
    """

    @property
    def name(self) -> str:
        return "kilocode"

    @property
    def default_timeout(self) -> int:
        return 600

    def is_available(self) -> bool:
        return any(shutil.which(name) is not None for name in BINARY_NAMES)

    def build_command(
        self,
        prompt: str,
        model: str,
        *,
        timeout_s: Optional[int] = None,
        schema: Optional[StructuredSchema] = None,
    ) -> List[str]:
        binary = _resolve_binary()
        base_model, effort = split_reasoning_effort(model, REASONING_EFFORTS)
        if effort is not None:
            return [binary, "run", "--auto", "-m", base_model, "--variant", effort, prompt]
        return [binary, "run", "--auto", "-m", model, prompt]

    def parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse output from Kilo Code CLI.

        Strips ANSI escape codes from plain text output, then attempts
        JSON extraction from code blocks or embedded JSON in text.
        """
        # Strip ANSI escape codes from plain text output
        stdout = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        stdout = stdout.strip()

        if not stdout:
            return {"success": False, "error": "Empty output", "raw": stdout, "data": None}

        # Try direct JSON parsing
        if stdout.startswith(('[', '{')):
            try:
                return {"success": True, "data": json.loads(stdout)}
            except json.JSONDecodeError:
                pass

        # Fall back to extraction from text/code blocks
        return extract_json_from_text(stdout, prefer_arrays=True)
