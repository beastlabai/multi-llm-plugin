"""The frozen `legacy/` core still closes the two phases the Python moved.

Reviewer finding 3 (fix round 6, 2026-10-05): since 4dbcae2 the write
phases' native emit path records the phase *started* and only the host's
closing step (`--mark-completed`, `finalize_tracking.py --mark-completed`)
records it completed. The legacy copies were written against the emit-time
completion that commit removed — `legacy/instructions/apply-suggestions.md`
had no closing step at all and `legacy/instructions/implement.md` called
`finalize_tracking.py` without the flag — so a host served the legacy core
(`MULTI_LLM_LEGACY_INSTRUCTIONS=1`) could no longer complete either phase.

The choice was between editing the frozen copies and making the orchestrators
keep emit-time completion when the variable is set. The README names the
escape hatch a *read-path* switch ("same dispatcher, same modes, same exit
codes"), so a write-path branch keyed on it would break the README's own
rule harder than two closing calls do; the two calls are the minimal edit
and are declared in `legacy/README.md`.
"""

from pathlib import Path

LEGACY = Path(__file__).parent.parent / "legacy"


def test_legacy_apply_suggestions_closes_the_phase():
    text = (LEGACY / "instructions" / "apply-suggestions.md").read_text(encoding="utf-8")
    closing = (
        'uv run --project "${CLAUDE_SKILL_DIR}" -- python '
        '"${CLAUDE_SKILL_DIR}/apply_suggestions_orchestrator.py" '
        '--plan-file "$PLAN_PATH" --mark-completed'
    )
    assert closing in text
    # After the summary report, before the JSON reference — the native
    # core's own placement (§8).
    assert text.index("### 7. Generate Summary Report") < text.index(closing)
    assert text.index(closing) < text.index("## JSON Output Structure")


def test_legacy_implement_closes_the_phase():
    text = (LEGACY / "instructions" / "implement.md").read_text(encoding="utf-8")
    closing = (
        'uv run --project "${CLAUDE_SKILL_DIR}" -- python '
        '"${CLAUDE_SKILL_DIR}/finalize_tracking.py" --state-file "$STATE_FILE" '
        '--mark-completed'
    )
    assert closing in text
    # The step-7 census call stays as it was (no flag: the phase is not done
    # there), and the closing call sits after the plan update, before the
    # report.
    assert '"${CLAUDE_SKILL_DIR}/finalize_tracking.py" --state-file "$STATE_FILE"\n' in text
    assert text.index("10. **Update the original plan:**") < text.index(closing)
    assert text.index(closing) < text.index("11. **Report results**")


def test_the_readme_declares_the_back_port():
    text = (LEGACY / "README.md").read_text(encoding="utf-8")
    assert "--mark-completed" in text
