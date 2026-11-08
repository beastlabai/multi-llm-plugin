"""The JSON readers outside ``utils.json_extractor`` read model output in
linear time and return exactly what their regexes returned.

Three readers recovered JSON with searches that are quadratic on untrusted
text: ``review_orchestrator_base.extract_json_array`` (what
``save_model_result`` parses a plan or task review with, after the provider's
deadline), ``schema_validator.validate_json_output`` and
``suggestion_processor.extract_suggestions_from_review``. They now call
``first_fenced_block`` and ``first_to_last_bracket``. Each is compared with a
copy of itself as it was, on the extractor tests' corpus and on shapes of its
own, and timed on a megabyte in a child process under a timeout, so a
regression fails instead of hanging the suite.
"""

import json
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_json_extractor import (  # noqa: E402
    _FENCE_ALPHABETS,
    _FENCE_EDGE_CASES,
    _FIRST_READER_EDGE_CASES,
    _WALK_ALPHABETS,
    _document_mix,
    _fence_mix,
    _fenced_fixture_texts,
)
from utils import schema_validator, suggestion_processor  # noqa: E402
from utils.review_orchestrator_base import extract_json_array  # noqa: E402
from utils.schema_validator import drop_non_record_entries, validate_json_output  # noqa: E402
from utils.suggestion_processor import extract_suggestions_from_review  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent


# --- The oracles: the three readers as they were, regexes and all. --------------


def _old_extract_json_array(text):
    try:
        wrapper = json.loads(text)
        if isinstance(wrapper, dict) and 'result' in wrapper:
            text = wrapper['result']
    except (json.JSONDecodeError, TypeError):
        pass

    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if code_block_match:
        return code_block_match.group(1)

    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        return array_match.group(0)

    return None


def _old_validate_json_output(output, schema_name):
    parsed = None

    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError:
        pass

    if parsed is None:
        code_block_match = re.search(
            r'```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```',
            output
        )
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

    if parsed is None:
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            match = re.search(pattern, output)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    break
                except json.JSONDecodeError:
                    continue

    if parsed is None:
        return False, {}, ["Could not extract valid JSON from output"]

    try:
        schema = schema_validator.load_schema(schema_name)
    except FileNotFoundError as e:
        return False, parsed, [str(e)]

    errors = schema_validator.validate_against_schema(parsed, schema)

    return len(errors) == 0, parsed, errors


def _old_extract_suggestions_from_review(review_text, model_name):
    suggestions = []

    json_match = re.search(r'```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```', review_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, list):
                for item in drop_non_record_entries(
                    parsed, model_name, noun="suggestions"
                ):
                    item["source_model"] = model_name
                    suggestions.append(item)
                return suggestions
        except json.JSONDecodeError:
            pass

    suggestion_patterns = [
        r'(?:Suggestion|Issue|Problem|Concern|Improvement)[\s:]+(.+?)(?=\n\n|\n-|\n\d+\.|\Z)',
        r'-\s+(.+?)(?=\n-|\n\n|\Z)',
        r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|\Z)',
    ]

    for pattern in suggestion_patterns:
        matches = re.findall(pattern, review_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            text = match.strip()
            if len(text) > 20:
                suggestions.append({
                    "title": text[:100] + "..." if len(text) > 100 else text,
                    "desc": text,
                    "importance": suggestion_processor._infer_importance(text),
                    "type": suggestion_processor._infer_type(text),
                    "source_model": model_name,
                })

    return suggestions


def _outcome(reader, *args):
    """What a reader returned, or the kind of error it raised: a ``TypeError``
    for a result that is not a string is part of the contract kept."""
    try:
        return "returned", reader(*args)
    except Exception as error:  # noqa: BLE001 - the error is the outcome compared
        return "raised", type(error)


# --- The corpus. ------------------------------------------------------------------

#: Review prose the suggestion fallback reads: list markers, numbered items,
#: keywords, digit runs and the blank lines that end an item.
_PROSE_ALPHABET = (
    "1", "12", ".", ". ", " ", "\n", "\n\n", "-", "- ", "Issue", "issue:",
    "Suggestion: ", "Concern ", "x", "word ", "1. ", "\n1.", "\n-", "critical ",
    "add a section ", "remove ", "```", "```json\n", "[", "]", "{", "}",
)


def _random_strings(seed, count=1500):
    rng = random.Random(seed)
    alphabets = _FENCE_ALPHABETS + _WALK_ALPHABETS + (_PROSE_ALPHABET,)
    texts = []
    for index in range(count):
        alphabet = alphabets[index % len(alphabets)]
        length = rng.choice([rng.randint(0, 8), rng.randint(0, 40), rng.randint(0, 120)])
        texts.append("".join(rng.choice(alphabet) for _ in range(length)))
    return texts


def _mixes(seed, count=600):
    rng = random.Random(seed)
    return [(_fence_mix if index % 2 else _document_mix)(rng) for index in range(count)]


def _corpus(seed):
    texts = list(_FENCE_EDGE_CASES) + list(_FIRST_READER_EDGE_CASES)
    texts += _random_strings(seed) + _mixes(seed + 1) + _fenced_fixture_texts()
    return texts


def _suggestions_array(rng):
    """A reviewer's array with fences, brackets and numbered prose inside its
    strings, compact or indented, sometimes fenced and sometimes cut short."""
    suggestions = [
        {
            "title": rng.choice(["t", "```[", "[1] x", "1. y"]),
            "desc": "".join(rng.choice(_PROSE_ALPHABET) for _ in range(rng.randint(0, 30))),
            "importance": rng.choice(["high", "LOW", "medium", "urgent"]),
            "reference": rng.choice(["## A", "## A / ## B", "[x]"]),
            "type": rng.choice(["addition", "deletion", "rewrite"]),
        }
        for _ in range(rng.randint(0, 3))
    ]
    text = json.dumps(suggestions, indent=rng.choice([None, 2]))
    roll = rng.random()
    if roll < 0.25:
        text = "```json\n" + text + "\n```"
    elif roll < 0.4:
        text = text[: rng.randint(0, len(text))]
    return text


# --- extract_json_array: what save_model_result parses a review with. -------------


class TestExtractJsonArrayMatchesItsRegexes:

    def test_corpus(self):
        for text in _corpus(4242):
            assert _outcome(extract_json_array, text) == _outcome(
                _old_extract_json_array, text), f"text={text!r}"

    def test_cursor_agent_wrappers(self):
        """A ``{"result": ...}`` wrapper is unwrapped first; a result that is
        not a string raised ``TypeError`` from the search and still does."""
        texts = [json.dumps(value) for value in (
            {"result": 5}, {"result": None}, {"result": ["x"]}, {"result": {"a": [1]}},
            {"result": True}, {"type": "result"}, {"result": ""},
        )]
        for text in _random_strings(77, count=600) + _mixes(78, count=200):
            texts.append(json.dumps({"result": text}))
            texts.append(json.dumps({"type": "result", "subtype": "success", "result": text}))
        for text in texts:
            assert _outcome(extract_json_array, text) == _outcome(
                _old_extract_json_array, text), f"text={text!r}"

    def test_review_arrays(self):
        rng = random.Random(5150)
        for _ in range(1500):
            text = _suggestions_array(rng)
            assert _outcome(extract_json_array, text) == _outcome(
                _old_extract_json_array, text), f"text={text!r}"

    @pytest.mark.parametrize("value", [None, 5, b"[1]", b'{"result": "[1]"}', ["[1]"]])
    def test_anything_but_text(self, value):
        assert _outcome(extract_json_array, value) == _outcome(_old_extract_json_array, value)


class TestSaveModelResultReadsADeepReplyAsUnparsed:
    """A reply nested deeper than the JSON parser's stack raised
    ``RecursionError`` out of ``save_model_result`` and ended the round's save
    loop; from about 1,000 brackets on Python 3.10 and 3.11. It is now a reply
    that does not parse: the model failed, its raw output is logged, and the
    salvage gate decides as for any other unparsed reply."""

    @pytest.mark.parametrize("output", [
        "[" * (1024 * 1024),
        "[" * (1024 * 1024) + "]" * (1024 * 1024),
        '[{"title": "t", "x": ' + "[" * (1024 * 1024) + "]",
    ], ids=["unclosed", "closed", "inside-a-suggestion"])
    def test_a_deep_reply_fails_the_model_without_raising(self, tmp_path, capsys, output):
        from utils.review_orchestrator_base import save_model_result

        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        saved = save_model_result(
            "p", "m:x", True, output, None, str(tmp_path), str(phase_dir), "review_tasks")
        assert saved is False
        assert not (phase_dir / "m_x.json").exists()
        log = next(phase_dir.glob("error_*.log")).read_text(encoding="utf-8")
        assert "recursion" in log.lower() or "stack" in log.lower()
        assert "Parse error:" in log or "No JSON in response:" in log
        assert "Encode error" not in log


def _deep_suggestion(depth):
    """A valid suggestion with an extra field nested *depth* arrays deep."""
    suggestion = {"title": "t", "desc": "d", "importance": "high",
                  "reference": "## A", "type": "addition"}
    return json.dumps([suggestion])[:-2] + ', "x": ' + "[" * depth + "]" * depth + "}]"


class TestSaveModelResultNamesAnEncoderFailure:
    """A reply that parses can still be too deep for the encoder that writes
    ``{model}.json``: on Python 3.12 the decoder holds about 10,000 levels and
    the encoder about 1,000, and on 3.13 and later ``json.dump`` (but not
    ``json.dumps``) stops at about 1,000. That wrote a truncated result file and
    logged the reply as one that does not parse."""

    def _save(self, tmp_path, output):
        from utils.review_orchestrator_base import save_model_result

        phase_dir = tmp_path / "phase"
        phase_dir.mkdir()
        saved = save_model_result(
            "p", "m:x", True, output, None, str(tmp_path), str(phase_dir), "review_tasks")
        return saved, phase_dir

    def test_an_exhausted_encoder_leaves_no_partial_file(self, tmp_path, monkeypatch, capsys):
        """Both encoders fail as the real ones do: ``dump`` after writing a
        prefix to the file, ``dumps`` before returning anything."""
        output = _deep_suggestion(20)
        real_dump, real_dumps = json.dump, json.dumps

        def deep(obj):
            return isinstance(obj, list) and any(
                isinstance(item, dict) and "x" in item for item in obj)

        def dump(obj, fp, *args, **kwargs):
            if deep(obj):
                fp.write('[\n  {\n    "title": "t",\n    "x": [[[')
                raise RecursionError("maximum recursion depth exceeded")
            return real_dump(obj, fp, *args, **kwargs)

        def dumps(obj, *args, **kwargs):
            if deep(obj):
                raise RecursionError("maximum recursion depth exceeded")
            return real_dumps(obj, *args, **kwargs)

        monkeypatch.setattr(json, "dump", dump)
        monkeypatch.setattr(json, "dumps", dumps)
        saved, phase_dir = self._save(tmp_path, output)

        assert saved is False
        # No result file, partial or whole, and no salvage request: the reply
        # parsed, so there is nothing for a repair to fix.
        assert sorted(p.name for p in phase_dir.iterdir()) == ["error_m_x.log"]
        log = (phase_dir / "error_m_x.log").read_text(encoding="utf-8")
        assert "Encode error: the reply parsed" in log
        assert "maximum recursion depth exceeded" in log
        assert "Parse error" not in log and "No JSON in response" not in log
        assert "nested too deep to save" in capsys.readouterr().out

    def test_a_reply_this_python_parses_is_saved_whole_or_not_at_all(self, tmp_path, capsys):
        """The real encoders on the reviewer's shape (an extra field 1,100
        arrays deep): every Python since 3.12 parses it, and ``json.dump``'s
        encoder could not write it on any of them."""
        output = _deep_suggestion(1100)
        try:
            parsed = json.loads(output)
        except RecursionError:
            pytest.skip("this Python's decoder stops short of 1,100 levels")
        saved, phase_dir = self._save(tmp_path, output)

        result = phase_dir / "m_x.json"
        try:
            json.dumps(parsed, indent=2)
        except RecursionError:
            assert saved is False and not result.exists()
            log = (phase_dir / "error_m_x.log").read_text(encoding="utf-8")
            assert "Encode error: the reply parsed" in log
        else:
            assert saved is True
            parsed[0]["importance"] = "HIGH"
            assert json.loads(result.read_text(encoding="utf-8")) == parsed


# --- The two readers no review path reaches (see the module docstrings). ----------


class TestValidateJsonOutputMatchesItsRegexes:

    @pytest.mark.parametrize("schema_name", [
        "code_review_issues.schema.json", "state_file.schema.json", "no_such.schema.json",
    ])
    def test_corpus(self, schema_name):
        for text in _corpus(9001):
            assert _outcome(validate_json_output, text, schema_name) == _outcome(
                _old_validate_json_output, text, schema_name), f"text={text!r}"

    def test_review_arrays(self):
        rng = random.Random(6006)
        for _ in range(1000):
            text = "Here it is: " + _suggestions_array(rng) + rng.choice(["", " done {"])
            assert _outcome(validate_json_output, text, "code_review_issues.schema.json") == (
                _outcome(_old_validate_json_output, text, "code_review_issues.schema.json")
            ), f"text={text!r}"


class TestExtractSuggestionsMatchesItsRegexes:

    def test_corpus(self, capsys):
        for text in _corpus(31415):
            assert _outcome(extract_suggestions_from_review, text, "m:x") == _outcome(
                _old_extract_suggestions_from_review, text, "m:x"), f"text={text!r}"

    def test_review_prose_and_arrays(self, capsys):
        rng = random.Random(2718)
        for _ in range(1500):
            parts = [
                "".join(rng.choice(_PROSE_ALPHABET) for _ in range(rng.randint(0, 60))),
                _suggestions_array(rng) if rng.random() < 0.3 else "",
                "".join(rng.choice(_PROSE_ALPHABET) for _ in range(rng.randint(0, 60))),
            ]
            text = "".join(parts)
            assert _outcome(extract_suggestions_from_review, text, "m:x") == _outcome(
                _old_extract_suggestions_from_review, text, "m:x"), f"text={text!r}"


# --- Bounded time, in a child process. --------------------------------------------


def _run_timed(body, what):
    script = textwrap.dedent(f"""
        import contextlib, io, json, os, sys, tempfile, time
        sys.path.insert(0, {str(SKILL_ROOT)!r})
        size = 1024 * 1024
    """) + textwrap.dedent(body) + textwrap.dedent("""
        print(json.dumps({"elapsed": elapsed}))
    """)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{what} took over 60 s on 1 MiB: a quadratic search is back "
                    "(see utils/json_extractor.py first_fenced_block, first_to_last_bracket)")
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1])["elapsed"] < 20


class TestTheReadersAreLinear:
    """At 40 KB these took about 0.3 to 1.1 s each, and four times that for
    every doubling. A megabyte now takes milliseconds."""

    def test_save_model_result_saves_a_megabyte_reply_in_bounded_time(self):
        """The reachable one: ``save_model_result`` parses a plan or task
        review synchronously, outside the provider's timeout."""
        _run_timed("""
            from utils.review_orchestrator_base import save_model_result
            suggestion = {"title": "t", "desc": "```[" * (size // 4), "importance": "high",
                          "reference": "## A", "type": "addition"}
            outputs = [
                json.dumps([suggestion]),
                json.dumps({"desc": "[" * size}),
                json.dumps({"type": "result", "result": "```[" * (size // 4)}),
                "```json\\n[" * (size // 9),
                "[" * size,
            ]
            with tempfile.TemporaryDirectory() as root:
                started = time.perf_counter()
                for index, output in enumerate(outputs):
                    phase_dir = os.path.join(root, str(index))
                    os.makedirs(phase_dir)
                    with contextlib.redirect_stdout(io.StringIO()):
                        save_model_result("p", "m:x", True, output, None, root,
                                          phase_dir, "review_tasks")
                elapsed = time.perf_counter() - started
        """, "save_model_result")

    def test_the_unreached_readers_read_a_megabyte_in_bounded_time(self):
        _run_timed("""
            from utils.schema_validator import validate_json_output
            from utils.suggestion_processor import extract_suggestions_from_review
            texts = ["x" + "```[" * (size // 4), "x" + "[" * size, "x" + "{" * size,
                     "x" + "```{" * (size // 4), "x" + "1" * size,
                     "1. x" + "\\n" + "1" * size]
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()), \\
                    contextlib.redirect_stderr(io.StringIO()):
                for text in texts:
                    validate_json_output(text, "code_review_issues.schema.json")
                    extract_suggestions_from_review(text, "m:x")
            elapsed = time.perf_counter() - started
        """, "validate_json_output or extract_suggestions_from_review")
