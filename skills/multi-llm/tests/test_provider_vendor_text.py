"""Tests for the vendor-text haystack contract across adapters.

One of the per-provider test files; helpers that several of them share live in
``provider_test_support.py``. See docs/development/providers.md.
"""

import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.providers.aider import AiderProvider
from utils.providers.base import CATEGORY_INVALID_MODEL, LLMProvider
from utils.retry_ladder import (
    STATUS_AUTH_FAILED,
    STATUS_CONTEXT_LENGTH_EXCEEDED,
    STATUS_RATE_LIMITED,
)
from utils.providers import base as base_module
from utils.providers.goose import GooseProvider
from tests.provider_test_support import C1_FIXTURES, _c1_adapter, _c1_capture, _c1_without_the_hook
from tests.source_cache import python_files, source_text, source_tree


#: The whole committed goose corpus and its verdict on both sides of the fix.
#: **This tuple is C1's acceptance test**: fourteen rows, and exactly one cell
#: moves. `stdout` is `None` for the two rows the manifest declares
#: not-applicable (goose exits 0 on every measured failure, and its envelope is
#: never empty), which classify as `None` on empty streams either way. Every
#: goose `*.stderr.txt` is 0 bytes, so stderr is "" throughout.
C1_GOOSE_CORPUS = (
    # row,                stdout file,                    before,          after
    ("success",           "success.stdout.txt",           None,            None),
    ("truncated",         "truncated.stdout.txt",         None,            None),
    ("ephemeral-only",    "ephemeral-only.stdout.txt",    None,            None),
    ("multi-assistant",   "multi-assistant.stdout.txt",   None,            None),
    ("error-envelope",    "error-envelope.stdout.txt",    None,            None),
    ("exit-zero-failure", "exit-zero-failure.stdout.txt", "server_error",  "server_error"),
    ("stdout-error",      None,                           None,            None),
    ("empty",             None,                           None,            None),
    ("auth-error",        "auth-error.stdout.txt",        "auth_failed",   "auth_failed"),
    ("rate-limit",        "rate-limit.stdout.txt",        "rate_limited",  "rate_limited"),
    ("invalid-model",     "invalid-model.stdout.txt",     "invalid_model", "invalid_model"),
    # Review 24 pins #4: the shared table learned goose 1.53.0's `effective`.
    ("context-overflow",  "context-overflow.stdout.txt",  "context_length_exceeded", "context_length_exceeded"),
    ("zero-body",         "zero-body.stdout.txt",         None,            None),
    ("prompt-echo",       "prompt-echo.stdout.txt",       "timeout",       None),
)


#: The four adapters measured to echo our prompt, with the committed capture of a
#: run that SUCCEEDED and the verdict that capture drew before the fix. muse's
#: `None` is the "one plan word away" case: it echoes, nothing in its committed
#: prompt happens to match a table, and its own `auth_error_patterns` already
#: carry `api error 401` — which is why C1 lands before the auth-table widening.
C1_ECHOING_SUCCESS_CAPTURES = (
    ("goose", "prompt-echo.stdout.txt", "timeout"),
    ("copilot", "success.jsonl", "rate_limited"),
    ("vibe", "success-json.json", "rate_limited"),
    ("muse", "success.stdout.txt", None),
)


def _c1_goose_with_plan_sentence(sentence):
    """goose's real cut-stream capture, with one sentence added to OUR prompt.

    The bytes stay a real goose envelope: the sentence is appended to the echoed
    user turn, which is exactly where a plan discussing 401s or rate limits would
    have put it. Everything goose itself wrote is untouched.
    """
    envelope = json.loads(_c1_capture("goose", "truncated.stdout.txt"))
    for message in envelope["messages"]:
        if message.get("role") == "user":
            message["content"][0]["text"] += "\n\n" + sentence
            break
    return json.dumps(envelope, indent=2, ensure_ascii=False)


class TestVendorTextContract:
    """The hook itself: identity by default, applied in exactly one place."""

    def test_the_base_hook_is_identity(self):
        """An adapter that does not override it keeps today's bytes exactly."""
        stdout, stderr = LLMProvider.vendor_text(
            AiderProvider(), "some stdout", "some stderr"
        )
        assert (stdout, stderr) == ("some stdout", "some stderr")

    @pytest.mark.parametrize("provider", [
        "agy", "aider", "claude-code", "cline", "cursor-agent",
        "gemini", "grok", "kimi", "qwen-code",
    ])
    def test_every_non_echoing_adapter_still_gets_the_whole_capture(self, provider):
        """The nine adapters no committed capture shows echoing.

        "Non-echoing" means **echoes neither the prompt nor the repo**. opencode
        left this roster (T2.2): it does not echo our prompt, but its
        ``--format json`` stdout carries tool output, and a validator's ``grep``
        of a plan discussing rate limits was recorded ``rate_limited``. pi left
        it with ``--mode json`` (T7.4): its stream echoes our prompt, pi's own
        system prompt and every tool result, so ``PiProvider.vendor_text``
        reduces it (``tests/fixtures/pi/markers-in-tool-result``). kilocode is
        not in it either: its stdout is the plain answer and its stderr a tool
        trace naming what the model read, so ``KiloCodeProvider.vendor_text``
        keeps only its painted ``Error:`` lines
        (``TestKiloCodeReadsOnlyItsPaintedErrorLines``).

        The default being identity is what makes "fix the class for the adapters
        that echo" a bounded claim rather than a rewrite of every classifier.
        """
        text = "Ran into this error: 429 rate limit exceeded"
        assert _c1_adapter(provider).vendor_text(text, "stderr text") == (
            text, "stderr text"
        )

    def test_the_contract_is_documented_where_an_override_author_will_read_it(self):
        """Both rules that make a reduction safe are in the hook's own docstring."""
        doc = (LLMProvider.vendor_text.__doc__ or "").lower()
        assert "only remove" in doc, "the removal-only rule is not documented"
        assert "unchanged" in doc and "never" in doc, (
            "the degrade-to-its-input rule is not documented"
        )

    @staticmethod
    def _c1_base_source():
        """base.py's text, read from disk rather than through ``inspect``.

        ``inspect.getsource`` resolves a slice by the *imported* object's
        ``co_firstlineno`` against the *current* file, so an unrelated edit to
        this module while the suite runs makes it return the wrong lines. The
        source is parsed here instead, which has no such coupling.
        """
        from utils.providers import base as base_module

        return source_text(base_module.__file__), source_tree(base_module.__file__)

    @staticmethod
    def _c1_body_of(tree, name, cls=None):
        import ast

        scope = tree
        if cls is not None:
            scope = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == cls
            )
        function = next(
            node for node in scope.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return ast.dump(ast.Module(body=function.body, type_ignores=[]))

    def test_classify_failure_applies_it(self):
        """One call, immediately before the rule runs."""
        _text, tree = self._c1_base_source()
        body = self._c1_body_of(tree, "classify_failure", cls="LLMProvider")
        assert "'vendor_text'" in body, (
            "classify_failure no longer applies the hook"
        )

    def test_nothing_else_applies_it(self):
        """§7.2 depends on `classify_streams` staying the adapter-free rule.

        Exactly two appliers (T2.5): `classify_failure`, to classify, and the
        executor's `_classify_failed_run` (in `attempt_classifier.py`), to
        quote — never to classify.
        """
        import ast

        from utils.providers import failure_classifier

        text, tree = self._c1_base_source()
        classifier_tree = source_tree(failure_classifier.__file__)

        assert text.count("self.vendor_text(") == 1, (
            "the hook is applied in more than one place"
        )
        assert "'vendor_text'" not in self._c1_body_of(classifier_tree, "classify_streams")
        assert "'vendor_text'" not in self._c1_body_of(classifier_tree, "match_streams")
        assert "'vendor_text'" not in self._c1_body_of(
            tree, "classify_auth_probe", cls="LLMProvider"
        )

        utils_dir = Path(base_module.__file__).parent.parent
        executor_path = utils_dir / "attempt_classifier.py"
        executor_text = source_text(executor_path)
        assert executor_text.count("vendor_text(") == 1, (
            "attempt_classifier.py applies the hook more than once"
        )
        executor_tree = source_tree(executor_path)
        applier = next(
            node for node in executor_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_classify_failed_run"
        )
        assert "'vendor_text'" in ast.dump(applier), (
            "the executor's one application is not inside _classify_failed_run"
        )

        for path in python_files(utils_dir):
            if path in (executor_path, Path(base_module.__file__)):
                continue
            calls = [
                line for line in path.read_text(encoding="utf-8").splitlines()
                if "vendor_text(" in line
                and "def vendor_text(" not in line
                # An override delegating to its own base is not a new applier.
                and "super().vendor_text(" not in line
            ]
            assert not calls, f"{path.name} applies vendor_text: {calls}"

    def test_a_reduction_that_empties_the_haystack_is_not_what_the_base_does(self):
        """The degrade rule, stated as behaviour rather than as prose.

        An empty haystack classifies as `None`, so a hook that returned "" on a
        parse failure would launder a genuine 401 into a transport-shaped retry.
        This is the shape that must NOT be written; it is here so the reason is
        executable.
        """
        class _Launderer(GooseProvider):
            def vendor_text(self, stdout, stderr):
                return "", ""

        # goose 1.53.0 also tags this 401 with its error `kind`, which
        # `GooseProvider.classify_failure` reads off stdout before any table
        # (review 24, pins #4); `other` takes that away, so the verdict below
        # is the tables' alone, as it was when this test was written.
        genuine_401 = _c1_capture("goose", "auth-error.stdout.txt").replace(
            '"kind": "authentication"', '"kind": "other"'
        )
        assert '"kind": "other"' in genuine_401
        assert GooseProvider().classify_failure(genuine_401, "", 0) == \
            STATUS_AUTH_FAILED
        assert _Launderer().classify_failure(genuine_401, "", 0) is None


class TestVendorTextOnTheFourEchoingAdapters:
    """The measured set: goose, copilot, vibe, muse."""

    @pytest.mark.parametrize("provider,capture,before", C1_ECHOING_SUCCESS_CAPTURES)
    def test_a_successful_run_classifies_as_nothing(self, provider, capture, before):
        """A run that produced an answer is not a transport fault."""
        stdout = _c1_capture(provider, capture)
        assert _c1_adapter(provider).classify_failure(stdout, "", 0) is None

    @pytest.mark.parametrize("provider,capture,before", C1_ECHOING_SUCCESS_CAPTURES)
    def test_and_that_is_a_change(self, provider, capture, before, monkeypatch):
        """The same bytes, on the pre-C1 haystack, drew `before`."""
        stdout = _c1_capture(provider, capture)
        _c1_without_the_hook(monkeypatch, provider)
        assert _c1_adapter(provider).classify_failure(stdout, "", 0) == before

    @pytest.mark.parametrize("provider", ["goose", "copilot", "vibe", "muse"])
    def test_the_reduction_only_ever_removes(self, provider):
        """Every committed capture: a pure deletion, and no stderr.

        "Shorter" is the weak half of the contract and it is not enough on its
        own: three of the four hooks re-render a parsed envelope, so a rendering
        that inserted bytes the capture never carried — muse re-serialized with
        json's default ``", "``/``": "`` separators against a COMPACT stream —
        stays under the length bound purely because the dropped prompt is
        bigger than what the rendering added. The subsequence check is the
        contract as written: every character of the reduction appears in the
        capture, in order, so the hook can only have removed.
        """
        adapter = _c1_adapter(provider)
        for path in sorted((C1_FIXTURES / provider).iterdir()):
            if path.suffix not in (".txt", ".jsonl", ".json"):
                continue
            raw = path.read_bytes().decode("utf-8")
            reduced, stderr = adapter.vendor_text(raw, "a stderr diagnosis")
            assert len(reduced) <= len(raw), (
                f"{provider}/{path.name}: the reduction GREW the capture"
            )
            remaining = iter(raw)
            assert all(char in remaining for char in reduced), (
                f"{provider}/{path.name}: the reduction INSERTED bytes the "
                f"capture never carried"
            )
            assert stderr == "a stderr diagnosis", (
                f"{provider}/{path.name}: no shipped reduction touches stderr"
            )

    @pytest.mark.parametrize("provider", ["goose", "copilot", "vibe", "muse"])
    def test_a_broken_envelope_degrades_to_its_input(self, provider):
        """Not `""` — see `test_a_reduction_that_empties_the_haystack…`."""
        for junk in ("", "   ", "not json at all", '{"messages": "not a list"}',
                     '{"unclosed": ', "\x00\x01"):
            assert _c1_adapter(provider).vendor_text(junk, "err") == (junk, "err")

    @pytest.mark.parametrize("provider,capture,expected", [
        ("goose", "auth-error.stdout.txt", STATUS_AUTH_FAILED),
        ("goose", "rate-limit.stdout.txt", STATUS_RATE_LIMITED),
        ("goose", "exit-zero-failure.stdout.txt", "server_error"),
        ("goose", "invalid-model.stdout.txt", CATEGORY_INVALID_MODEL),
        ("copilot", "auth-error.jsonl", STATUS_AUTH_FAILED),
        ("copilot", "rate-limit.jsonl", STATUS_RATE_LIMITED),
        ("copilot", "invalid-model.jsonl", CATEGORY_INVALID_MODEL),
        ("copilot", "context-overflow.jsonl", STATUS_CONTEXT_LENGTH_EXCEEDED),
    ])
    def test_the_vendors_own_diagnosis_still_reaches_the_tables(
        self, provider, capture, expected
    ):
        """The reduction removes our turn and nothing the adapter said.

        goose's `Ran into this error:`, its quoted `500 Internal Server Error`
        status line and its `401 Unauthorized` gloss all live in the ASSISTANT
        message; copilot's whole diagnosis is a `session.error` record. Both are
        on the far side of the filter.
        """
        stdout = _c1_capture(provider, capture)
        assert _c1_adapter(provider).classify_failure(stdout, "", 0) == expected


class TestACutInsideCopilotsEchoIsStillDropped:
    """The residual a whole-line parse leaves: a stream cut MID-record.

    copilot's echo is one `user.message` line of 61 751 bytes — the largest in
    the stream by two orders of magnitude — so a kill lands inside it more often
    than anywhere else, and `json.loads` rejects the fragment. A filter that
    dropped only what PARSED would hand the prompt bytes straight back to the
    tables on the failed-run path `_classify_failed_run` reads.
    `truncated.jsonl` cannot show this: that capture's cut fell on a record
    boundary, so nothing committed pinned the shape until
    `truncated-mid-echo.jsonl`.
    """

    CUT = "truncated-mid-echo.jsonl"

    def test_the_committed_mid_echo_cut_classifies_as_nothing(self):
        """A half-written stream says nothing about *why* it stopped."""
        stdout = _c1_capture("copilot", self.CUT)
        assert _c1_adapter("copilot").classify_failure(stdout, "", 137) is None

    def test_and_that_is_a_change(self, monkeypatch):
        """The same bytes on the pre-C1 haystack — and on a line-parse-only one.

        `rate_limited` is one of the two classes the executor forwards, so this
        is the live half: it would have backed the whole batch off on the
        strength of a sentence in the plan under review.
        """
        stdout = _c1_capture("copilot", self.CUT)
        _c1_without_the_hook(monkeypatch, "copilot")
        assert _c1_adapter("copilot").classify_failure(stdout, "", 137) == \
            STATUS_RATE_LIMITED

    def test_the_prompt_bytes_are_gone_from_the_haystack(self):
        """Stated over the bytes, not over the verdict.

        The capture still carries our plan text — the fixture is worthless if a
        later edit removes it from the bytes instead of from the haystack.
        """
        stdout = _c1_capture("copilot", self.CUT)
        assert "rate-limit patterns" in stdout
        reduced, _stderr = _c1_adapter("copilot").vendor_text(stdout, "")
        assert "rate-limit" not in reduced
        assert "user.message" not in reduced

    def test_a_cut_record_of_ANY_OTHER_type_is_still_passed_through(self):
        """Only the echo's opener is dropped, and only at the line start.

        A fragment of an `assistant.message` is the vendor's own half-written
        text and stays in the haystack; so does a line that merely mentions the
        echo's type later on. Nothing about a rejected line is *parsed* — the
        rule is a prefix match used to remove, and never to read evidence.
        """
        adapter = _c1_adapter("copilot")
        vendor_fragment = (
            '{"type":"assistant.message","data":{"content":"429 rate limit'
        )
        mentions_it_later = (
            '{"note":"the record after this one is a "type":"user.message" '
            'and it hit a rate-limit'
        )
        for line in (vendor_fragment, mentions_it_later):
            stdout = _c1_capture("copilot", "success.jsonl") + "\n" + line
            reduced, _stderr = adapter.vendor_text(stdout, "")
            assert line in reduced, "a line that is not the echo was dropped"
            assert adapter.classify_failure(stdout, "", 137) == \
                STATUS_RATE_LIMITED

    def test_a_cut_too_early_to_name_the_type_is_passed_through(self):
        """It carries no prompt bytes, so there is nothing to remove."""
        opener = '{"type":"user.mess'
        reduced, _stderr = _c1_adapter("copilot").vendor_text(opener, "err")
        assert (reduced, _stderr) == (opener, "err")

    def test_the_record_boundary_capture_is_untouched_by_the_new_rule(self):
        """`truncated.jsonl` is the other half of the pair and does not move."""
        stdout = _c1_capture("copilot", "truncated.jsonl")
        adapter = _c1_adapter("copilot")
        assert adapter.classify_failure(stdout, "", 137) is None
        reduced, _stderr = adapter.vendor_text(stdout, "")
        assert "model.call_start" in reduced


class TestTheGooseCorpusMovesExactlyOneCell:
    """C1's acceptance test, run against every committed goose row."""

    @pytest.mark.parametrize("row,capture,before,after", C1_GOOSE_CORPUS)
    def test_each_row_classifies_as_the_table_declares(
        self, row, capture, before, after
    ):
        stdout = _c1_capture("goose", capture) if capture else ""
        assert GooseProvider().classify_failure(stdout, "", 0) == after

    @pytest.mark.parametrize("row,capture,before,after", C1_GOOSE_CORPUS)
    def test_each_row_classified_as_the_table_says_it_used_to(
        self, row, capture, before, after, monkeypatch
    ):
        stdout = _c1_capture("goose", capture) if capture else ""
        _c1_without_the_hook(monkeypatch, "goose")
        assert GooseProvider().classify_failure(stdout, "", 0) == before

    def test_exactly_one_of_the_fourteen_moved(self):
        moved = [row for row, _, before, after in C1_GOOSE_CORPUS if before != after]
        assert len(C1_GOOSE_CORPUS) == 14
        assert moved == ["prompt-echo"], (
            "C1 is a one-cell change to this corpus; anything else is a "
            "reduction that removed more than our prompt"
        )

    def test_the_table_and_the_fixture_manifest_cannot_drift(self):
        """The rig grades the same fourteen verdicts this table asserts."""
        import yaml

        manifest = yaml.safe_load(
            (C1_FIXTURES / "goose" / "manifest.yaml").read_text(encoding="utf-8")
        )
        rows = manifest["rows"]
        assert set(rows) == {row for row, _, _, _ in C1_GOOSE_CORPUS}
        for row, _capture, _before, after in C1_GOOSE_CORPUS:
            declared = rows[row]
            if "not_applicable" in declared:
                continue
            assert declared["expect"]["classify_failure"] == after, (
                f"goose/{row}: the manifest and this table disagree"
            )

    def test_the_prompt_echo_row_was_not_inverted(self):
        """The run always succeeded; only the classifier's answer was wrong."""
        import yaml

        manifest = yaml.safe_load(
            (C1_FIXTURES / "goose" / "manifest.yaml").read_text(encoding="utf-8")
        )
        expect = manifest["rows"]["prompt-echo"]["expect"]
        assert expect["success"] is True
        assert expect["classify_failure"] is None
        raw = _c1_capture("goose", "prompt-echo.stdout.txt")
        assert "address timeout scenarios" in raw, (
            "the row stops pinning anything if the echoed prompt is edited out "
            "of the bytes instead of out of the haystack"
        )


class TestCopilotAndVibeAreWatchedByTheRigFromNowOn:
    """The two halves of this defect that no fixture row watched."""

    @pytest.mark.parametrize("provider,capture", [
        ("copilot", "success.jsonl"),
        ("vibe", "success-json.json"),
    ])
    def test_a_manifest_exists_and_pins_the_success_row(self, provider, capture):
        import yaml

        path = C1_FIXTURES / provider / "manifest.yaml"
        assert path.is_file(), (
            f"{provider} ships captures with no manifest, which is how a green "
            f"run classified `rate_limited` unnoticed"
        )
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        success = manifest["rows"]["success"]
        assert success["stdout"] == capture
        assert success["expect"]["success"] is True
        assert success["expect"]["classify_failure"] is None

    @pytest.mark.parametrize("provider", ["copilot", "vibe"])
    def test_neither_sits_in_the_manifest_pending_table_any_more(self, provider):
        from tests.test_provider_fixture_rows import (
            FIXTURES_PENDING,
            MANIFEST_PENDING,
        )

        assert provider not in MANIFEST_PENDING
        assert provider not in FIXTURES_PENDING


class TestAiderStillComposesWithTheHook:
    """`aider`'s narrowing override is the precedent, and it is unchanged."""

    COST_MAP_NOISE = (
        "Error: Unable to fetch model_prices_and_context_window.json: "
        "Name or service not known\n"
    )

    def test_the_tolerated_line_is_still_dropped(self):
        """A successful aider run on a host with no egress is not a failure."""
        assert AiderProvider().classify_failure(
            self.COST_MAP_NOISE + '["ok"]', "", 0
        ) is None

    def test_a_genuine_failure_still_survives_both_reductions(self):
        assert AiderProvider().classify_failure(
            self.COST_MAP_NOISE + "litellm.AuthenticationError: unauthorized (401)",
            "", 1,
        ) == STATUS_AUTH_FAILED

    def test_the_override_still_delegates_the_rule_to_super(self):
        """Two reductions, both removal-only, so the order does not matter.

        The override narrows, ``super()`` then applies the hook. Read off the
        file rather than through ``inspect``, for the reason
        ``TestVendorTextContract._c1_base_source`` gives.
        """
        import ast
        import sys

        module = sys.modules[AiderProvider.__module__]
        tree = source_tree(module.__file__)
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AiderProvider"
        )
        method = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "classify_failure"
        )
        dumped = ast.dump(ast.Module(body=method.body, type_ignores=[]))
        assert "'super'" in dumped and "'classify_failure'" in dumped


class TestEchoedPlanTextNoLongerDecidesTheBatch:
    """The integration half: `_classify_failed_run`, on a FAILED run.

    This is the path the green-run measurements above are only a proxy for.
    `_classify_failed_run` is consulted for a run with no usable result, and it
    forwards exactly two of the classifier's answers — `auth_failed` (rung 1,
    the whole batch fails) and `rate_limited` (rung 3, backoff). Those are
    precisely the two an echoed plan sentence could manufacture.
    """

    AUTH_SENTENCE = "The plan says: the API returns 401 unauthorized on error."
    RATE_SENTENCE = "The plan says: retries must cover rate-limit responses."

    def _capture(self, sentence, exit_code=0):
        from utils.subagent_executor import _Capture

        return _Capture(
            stdout=_c1_goose_with_plan_sentence(sentence), stderr="",
            exit_code=exit_code,
        )

    @pytest.mark.parametrize("exit_code", [0, 1])
    @pytest.mark.parametrize("sentence_attr", ["AUTH_SENTENCE", "RATE_SENTENCE"])
    def test_a_cut_stream_stays_transport_shaped(self, sentence_attr, exit_code):
        from utils.subagent_executor import _classify_failed_run

        capture = self._capture(getattr(self, sentence_attr), exit_code)
        assert _classify_failed_run(GooseProvider(), capture) is None

    @pytest.mark.parametrize("sentence_attr,was", [
        ("AUTH_SENTENCE", STATUS_AUTH_FAILED),
        ("RATE_SENTENCE", STATUS_RATE_LIMITED),
    ])
    def test_and_before_C1_it_did_not(self, sentence_attr, was, monkeypatch):
        """The same bytes, the same call, on the pre-C1 haystack.

        On a non-zero exit, where :meth:`classify_failure` and its hook decide.
        goose is not ``EXIT0_CLASSIFIABLE``, so an exit 0 is never classified,
        hook or not (``test_exit_zero_prose_matrix.py``).
        """
        from utils.subagent_executor import _classify_failed_run

        capture = self._capture(getattr(self, sentence_attr), exit_code=1)
        _c1_without_the_hook(monkeypatch, "goose")
        verdict = _classify_failed_run(GooseProvider(), capture)
        assert verdict is not None and verdict[0] == was

    def test_the_cut_stream_itself_is_unclassified_either_way(self):
        """The control: goose's real `truncated` capture, no plan sentence."""
        from utils.subagent_executor import _Capture, _classify_failed_run

        capture = _Capture(
            stdout=_c1_capture("goose", "truncated.stdout.txt"),
            stderr="",
            exit_code=0,
        )
        assert _classify_failed_run(GooseProvider(), capture) is None


class TestTheNestedProbePromptIsCleanByConstruction:
    """The preflight probe is safe because of what its prompt IS, not the hook.

    `NESTED_PROBE_PROMPT`'s auth verdict raises `RunnerRejected` and refuses the
    host outright, so an echo of it carrying a marker would be the worst
    instance of this defect. It cannot: the prompt is one clean token.
    """

    def test_the_prompt_matches_no_table_of_any_adapter(self):
        from utils.provider_registry import _PROVIDERS
        from utils.subagent_executor import NESTED_PROBE_PROMPT

        for name, adapter in sorted(_PROVIDERS.items()):
            assert adapter.classify_failure(NESTED_PROBE_PROMPT, "", 0) is None, (
                f"{name} reads the nested-exec probe prompt itself as a failure"
            )

    def test_an_adapter_echoing_it_verbatim_still_classifies_nothing(self):
        """The echo, not just the prompt — which is what a runner returns."""
        from utils.subagent_executor import NESTED_PROBE_PROMPT

        echo = f"{NESTED_PROBE_PROMPT}\n{NESTED_PROBE_PROMPT}"
        assert GooseProvider().classify_failure(echo, "", 0) is None

    def test_it_stays_one_token_with_no_marker_vocabulary_in_it(self):
        from utils.providers import base as base_module
        from utils.subagent_executor import NESTED_PROBE_PROMPT

        haystack = NESTED_PROBE_PROMPT.lower()
        tables = (
            base_module.CONTEXT_OVERFLOW_MARKERS,
            base_module.TIMEOUT_MARKERS,
            base_module.CONNECTION_ERROR_MARKERS,
            base_module.SERVER_ERROR_MARKERS,
            base_module.INVALID_MODEL_MARKERS,
            base_module.SHARED_RATE_LIMIT_MARKERS,
        )
        for table in tables:
            assert not base_module._matches_marker(table, haystack)
        for _category, markers in base_module.LAUNCH_FAILURE_MARKERS:
            assert not base_module._matches_marker(markers, haystack)
        for markers in base_module.STDERR_LAUNCH_FAILURE_MARKERS.values():
            assert not base_module._matches_marker(markers, haystack)


class TestTheAuthProbeHaystackIsUntouched:
    """`classify_auth_probe` is deliberately out of scope (base.py's second
    `f"{stdout}\\n{stderr}"`).

    An auth probe never carries our prompt, so narrowing its haystack fixes
    nothing — and it is the one verdict allowed to refuse a run outright.
    """

    @pytest.mark.parametrize("provider,capture", [
        ("goose", "auth-error.stdout.txt"),
        ("copilot", "auth-error.jsonl"),
    ])
    def test_a_real_credential_notice_is_still_read_as_not_authenticated(
        self, provider, capture
    ):
        stdout = _c1_capture(provider, capture)
        assert _c1_adapter(provider).classify_auth_probe(stdout, "", 0) is False

    def test_the_probe_reads_the_whole_capture_where_the_classifier_does_not(self):
        """One set of bytes, two deliberately different answers.

        The auth-shaped sentence is in the ECHOED user turn. `classify_failure`
        no longer sees it; `classify_auth_probe` still does, and that is the
        conservative direction — the probe's only pattern-drawn verdict is the
        one that refuses.
        """
        stdout = _c1_goose_with_plan_sentence(
            "The plan says: the API returns 401 unauthorized on error."
        )
        adapter = GooseProvider()
        assert adapter.classify_failure(stdout, "", 0) is None
        assert adapter.classify_auth_probe(stdout, "", 0) is False
