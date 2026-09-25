"""The shared pi-family stream reader (``utils/providers/pi_family_stream.py``).

T7.3 of ``plans/opencode-validator-false-rate-limit-and-denied-turn.md`` lifted
omp's JSONL reader into this module so pi can read the same stream without an
import cycle (§7.2, option R3). The acceptance test of the lift is elsewhere:
every omp test and fixture row passes with **zero** edits to its expectations.
What this file pins is what those tests cannot see:

* **the dependency order** ``base`` → ``pi_family_stream`` → ``pi`` → ``omp``,
  in a fresh interpreter, so no earlier import masks a cycle;
* that neither adapter keeps a private copy of the moved helpers;
* that the dialect-built sentences are omp's old sentences byte for byte;
* the two pieces omp does not use yet: the envelope test (§7.3 step 1) and
  ``sum_usage``'s compaction leg.
"""
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils.providers import omp as omp_module  # noqa: E402
from utils.providers import pi as pi_module  # noqa: E402
from utils.providers import pi_family_stream as pfs  # noqa: E402
from utils.providers.omp import OMP_DIALECT, OmpProvider  # noqa: E402

MODULE_PATH = SKILL_ROOT / "utils" / "providers" / "pi_family_stream.py"
OMP_FIXTURES = SKILL_ROOT / "tests" / "fixtures" / "omp"
PI_PROBE = (
    SKILL_ROOT / "tests" / "fixtures" / "pi" / "json-mode"
    / "usage-probe-0.87.1.stdout.jsonl"
)

#: A pi-shaped dialect for the helpers pi will use (T7.4 declares the real one).
PI_LIKE = pfs.StreamDialect(
    product="pi",
    verified_version="0.87.1",
    stream_version=3,
    stop_reasons=("stop", "length", "toolUse", "error", "aborted"),
    compaction_event="compaction_end",
    compaction_usage=True,
)


# --------------------------------------------------------------------------- #
# Import order
# --------------------------------------------------------------------------- #

#: Run in a fresh interpreter. ``utils`` and ``utils.providers`` are registered
#: as bare namespace packages first, because the real ``utils.providers``
#: ``__init__`` imports every adapter module (its roster is discovered), which
#: would load pi and omp before the module under test and hide the order.
_ISOLATED_IMPORT = textwrap.dedent(
    """
    import importlib, json, sys, types
    from pathlib import Path
    root = Path(sys.argv[1])
    for name, d in (("utils", root / "utils"),
                    ("utils.providers", root / "utils" / "providers")):
        m = types.ModuleType(name)
        m.__path__ = [str(d)]
        sys.modules[name] = m
    out = {}
    order = sys.argv[2].split(",")
    mods = {}
    for step in order:
        mods[step] = importlib.import_module("utils.providers." + step)
        out[step] = sorted(k for k in sys.modules if k.startswith("utils.providers."))
    if "pi_family_stream" in mods and "pi" in mods:
        out["pi_cap_shared"] = mods["pi"]._cap is mods["pi_family_stream"]._cap
    if "pi_family_stream" in mods and "omp" in mods:
        out["omp_one_line_shared"] = (
            mods["omp"]._one_line is mods["pi_family_stream"]._one_line
        )
    print(json.dumps(out))
    """
)


def _import_in_fresh_interpreter(*steps: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATED_IMPORT, str(SKILL_ROOT), ",".join(steps)],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestImportOrder:
    """``base`` / ``json_extractor`` → ``pi_family_stream`` → ``pi`` → ``omp``."""

    def test_the_shared_module_imports_alone_without_either_adapter(self):
        out = _import_in_fresh_interpreter("pi_family_stream")
        loaded = out["pi_family_stream"]
        assert "utils.providers.pi" not in loaded
        assert "utils.providers.omp" not in loaded

    def test_shared_then_pi_then_omp_in_one_interpreter(self):
        out = _import_in_fresh_interpreter("pi_family_stream", "pi", "omp")
        assert "utils.providers.omp" not in out["pi"]
        assert out["pi_cap_shared"] is True
        assert out["omp_one_line_shared"] is True

    @pytest.mark.parametrize("module", ["pi", "omp"])
    def test_each_adapter_imports_alone(self, module):
        out = _import_in_fresh_interpreter(module)
        assert f"utils.providers.{module}" in out[module]
        assert "utils.providers.pi_family_stream" in out[module]

    def test_the_source_never_names_pi_or_omp_in_an_import(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                imported.append(base)
                imported.extend(f"{base}.{alias.name}" for alias in node.names)
        for name in imported:
            last = name.rsplit(".", 1)[-1]
            assert last not in ("pi", "omp"), f"pi_family_stream imports {name}"
        relative = {
            "." * n.level + (n.module or "")
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level
        }
        assert relative <= {".base", "..json_extractor"}, relative


class TestNoPrivateCopies:
    """Neither adapter keeps its own copy of a moved helper."""

    @pytest.mark.parametrize("name", ["_cap", "_one_line", "_MAX_DIAGNOSIS_CHARS"])
    def test_pi_reexports_the_shared_helper(self, name):
        assert getattr(pi_module, name) is getattr(pfs, name)

    @pytest.mark.parametrize(
        "name",
        [
            "_cap", "_one_line", "_events", "_stream_version", "_assistant_ends",
            "_tool_ends", "_result_text", "_answer_text", "_unclosed_container",
            "_retry_give_up", "_first_line", "_ASSISTANT_MESSAGE_REQUIRED_KEYS",
        ],
    )
    def test_omp_uses_the_shared_helper(self, name):
        assert getattr(omp_module, name) is getattr(pfs, name)

    def test_from_pi_import_cap_still_works(self):
        from utils.providers.pi import _cap  # noqa: F401


# --------------------------------------------------------------------------- #
# omp's sentences, byte for byte
# --------------------------------------------------------------------------- #

class TestOmpDialectSentences:
    """The literals omp.py built before the lift, copied from its old source."""

    @pytest.mark.parametrize("version", [None, 3, 4])
    def test_version_clause(self, version):
        seen = 3 if version is None else version
        assert omp_module._version_clause(version) == (
            f"stream version {seen}; the parser was written against omp 18.2.2's "
            f"version 3 stream"
        )

    def test_unrecognized_stop_reason(self):
        final = {"stopReason": "deferred", "errorMessage": "  some\n detail "}
        assert pfs.unrecognized_stop_reason(final, 3, OMP_DIALECT) == (
            "unrecognized stopReason 'deferred' on the last assistant message "
            "(omp 18.2.2 emits stop/toolUse/error/aborted; stream version 3; the "
            "parser was written against omp 18.2.2's version 3 stream); some detail"
        )

    def test_stream_version_warning(self, capsys):
        omp_module._warn_stream_version(9731)
        omp_module._warn_stream_version(9731)
        err = capsys.readouterr().err
        assert err == (
            "WARNING: omp stream version 9731; the parser was written against "
            "3 (omp 18.2.2).\n"
        )

    def test_the_compaction_leg_is_off_for_omp(self):
        assert OMP_DIALECT.compaction_event == "auto_compaction_end"
        assert OMP_DIALECT.compaction_usage is False

    def test_a_pi_dialect_speaks_for_pi(self):
        assert pfs.version_clause(None, PI_LIKE) == (
            "stream version 3; the parser was written against pi 0.87.1's "
            "version 3 stream"
        )
        fault = pfs.shape_fault([{"role": "assistant", "content": []}], 3, dialect=PI_LIKE)
        assert fault == (
            "assistant message_end without stopReason (stream version 3; the "
            "parser was written against pi 0.87.1's version 3 stream)"
        )


# --------------------------------------------------------------------------- #
# The envelope test (§7.3 step 1) — unused by omp for now
# --------------------------------------------------------------------------- #

_HEADER = json.dumps({"type": "session", "version": 3})


class TestHasStreamEnvelope:
    @pytest.mark.parametrize(
        "stdout",
        [
            "",
            "   \n",
            "OK",
            '{"answer": "OK"}',
            '[{"file": "a.py"}]',
            '```json\n[{"file": "a.py"}]\n```',
            '{"type": "bug", "file": "a.py"}',
            '{"type": 3}',
            "Here is the review.\n" + '{"answer": "OK"}',
        ],
    )
    def test_not_a_stream(self, stdout):
        assert pfs.has_stream_envelope(stdout, PI_LIKE) is False

    @pytest.mark.parametrize(
        "stdout",
        [
            _HEADER,
            _HEADER + "\n" + '{"type": "agent_start"}',
            # cut mid-line after the header: stays on the stream parser
            _HEADER + '\n{"type": "message_end", "mess',
            # junk between events
            _HEADER + "\nsome banner\n" + '{"type": "agent_start"}',
            # a banner ahead of a stream whose header was lost
            'Working...\n{"type": "turn_start"}',
            '{"type": "tool_execution_end", "toolName": "read"}',
            '{"type": "auto_compaction_end"}',
        ],
    )
    def test_a_stream(self, stdout):
        assert pfs.has_stream_envelope(stdout, PI_LIKE) is True

    def test_every_committed_omp_stdout_with_events_has_an_envelope(self):
        checked = 0
        for path in sorted(OMP_FIXTURES.glob("*.stdout.txt")):
            text = path.read_text(encoding="utf-8")
            events, _ = pfs._events(text)
            if events:
                checked += 1
                assert pfs.has_stream_envelope(text, OMP_DIALECT), path.name
        assert checked

    def test_the_pi_probe_has_an_envelope(self):
        text = PI_PROBE.read_text(encoding="utf-8")
        assert pfs.has_stream_envelope(text, PI_LIKE)


# --------------------------------------------------------------------------- #
# sum_usage
# --------------------------------------------------------------------------- #

class TestSumUsage:
    def test_omp_parse_usage_is_sum_usage_under_the_omp_dialect(self):
        provider = OmpProvider()
        checked = 0
        for path in sorted(OMP_FIXTURES.glob("*.stdout.txt")):
            text = path.read_text(encoding="utf-8")
            events, _ = pfs._events(text)
            expected = pfs.sum_usage(events, OMP_DIALECT) if events else None
            assert provider.parse_usage(text, "") == expected, path.name
            checked += 1
        assert checked

    def test_the_pi_probe_sums_to_the_recorded_numbers(self):
        events, _ = pfs._events(PI_PROBE.read_text(encoding="utf-8"))
        usage = pfs.sum_usage(events, PI_LIKE)
        assert usage["input_tokens"] == 1445
        assert usage["output_tokens"] == 3
        assert usage["cached_input_tokens"] == 256
        assert usage["total_tokens"] == 1704
        assert round(usage["cost_usd"], 6) == 0.000219

    def test_no_events_is_none(self):
        assert pfs.sum_usage([], PI_LIKE) is None

    @staticmethod
    def _overflow_stream(compaction_type):
        usage = {"input": 10, "output": 2, "cacheRead": 0, "totalTokens": 12,
                 "cost": {"total": 0.001}}
        return [
            {"type": "session", "version": 3},
            {"type": "message_end",
             "message": {"role": "assistant", "content": [], "stopReason": "stop",
                         "usage": usage}},
            {"type": compaction_type, "reason": "overflow",
             "result": {"summary": "s", "usage": {"input": 100, "output": 20,
                                                  "totalTokens": 120,
                                                  "cost": {"total": 0.01}}}},
            {"type": compaction_type, "reason": "overflow", "aborted": True},
        ]

    def test_the_compaction_leg_adds_the_summary_call_when_on(self):
        usage = pfs.sum_usage(self._overflow_stream("compaction_end"), PI_LIKE)
        assert usage["input_tokens"] == 110
        assert usage["output_tokens"] == 22
        assert usage["total_tokens"] == 132
        assert usage["cost_usd"] == pytest.approx(0.011)
        assert usage["cached_input_tokens"] == 0

    def test_the_compaction_leg_is_ignored_when_off(self):
        stream = self._overflow_stream("auto_compaction_end")
        usage = pfs.sum_usage(stream, OMP_DIALECT)
        assert usage["input_tokens"] == 10
        assert usage["total_tokens"] == 12

    def test_absent_keys_stay_absent(self):
        events = [
            {"type": "message_end",
             "message": {"role": "assistant", "content": [], "stopReason": "error",
                         "usage": {"input": 0, "output": 0}}},
        ]
        usage = pfs.sum_usage(events, PI_LIKE)
        assert usage == {"input_tokens": 0, "output_tokens": 0}
