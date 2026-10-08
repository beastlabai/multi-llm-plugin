#!/usr/bin/env python3
"""How the Claude Code shim tells its host to run a dispatcher call (fix round
26, live run 11 #13).

Racing 03: the host ran `cd plans/racing-game/generate-tasks && grep …` to read
a phase file; Claude Code's Bash keeps that directory for every later call, so
its next `--reaggregate` resolved `plans/racing-game.md` inside the phase folder
and failed `Plan file not found`. The stop fixture's 01: the host chained a
stray `uv run --project ".../skills/multi-llm/skills/multi-llm" -- true` before
a `util write-prompt`, and racing 02/05 held the dispatcher in a shell variable
(`M="uv run …"; $M util batch-done …`), which no allow rule matches. SKILL.md's
section 6 now says to run every call from the session's starting directory,
never to `cd`, and to run each call alone and exactly as written.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL_ROOT = Path(__file__).parent.parent
SKILL_MD = SKILL_ROOT / "SKILL.md"


def _section_six_lead() -> str:
    text = SKILL_MD.read_text(encoding="utf-8")
    section = text.split("## 6. The one dispatcher command", 1)[1]
    lead = section.split("```", 1)[0]
    return " ".join(lead.split())


class TestSectionSixSaysHowToRunACall:

    def test_from_the_starting_directory_and_never_cd(self):
        lead = _section_six_lead()
        assert "every `util` command a core names starts the same way" in lead
        assert "from the directory this session started in (normally the repository root)" in lead
        assert "never `cd` in a Bash call" in lead
        assert "keeps the new directory for every later call" in lead

    def test_alone_and_exactly_as_written(self):
        lead = _section_six_lead()
        assert "alone in its Bash call and spelled exactly as written here" in lead
        assert "the skill directory as printed" in lead
        assert "nothing chained before or after it" in lead
        assert "no shell variable standing in for it" in lead


class TestTheCoreNeverAsksForWhatTheShimForbids:
    """A guard on the rule's other side: no core instruction chains a
    dispatcher call or changes directory, so the shim's rule never contradicts
    a step the host is told to run."""

    CHAINED = re.compile(r"(&&|;|\|\|)\s*(PYTHONUNBUFFERED=1\s+)?uv run\b")
    CD = re.compile(r"(^|[\s;&|(`])cd\s+[\"'$A-Za-z{/.~]")

    def test_no_core_instruction_chains_a_dispatcher_call_or_cds(self):
        offenders = []
        for path in sorted((SKILL_ROOT / "instructions").glob("*.md")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self.CHAINED.search(line) or self.CD.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()[:100]}")
        assert not offenders, offenders
