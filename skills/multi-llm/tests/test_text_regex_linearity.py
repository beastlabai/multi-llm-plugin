"""The plan, task and review text readers of code-health unit U69 return what
their regexes returned, in linear time.

Each reader below parsed model output (a ``generate-tasks`` task list, a
reviewer's ``reference``, a write subagent's check findings, a plan an edit
batch rewrote) with a regex that is quadratic or worse on one run of
whitespace: retried at every line start of a blank run, or splitting one run
between two adjacent ``\\s*``. ``SECTION_PATTERN`` was about n^3.5 (185 s for a
1 KB reference) and the IMPL_SUMMARY marker cubic (97 s at 8 KB). Each is
compared, in the way its caller reads it, with a copy of itself as it was, on
hand-written edges, seeded random text and the repository's fixtures, then
timed on a megabyte in a child process under a timeout.
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

from update_plan_tasks import generate_tasks_markdown  # noqa: E402
from utils import plan_updater, suggestion_batcher, suggestion_processor  # noqa: E402
from utils import task_decomposer as td  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402

SKILL_ROOT = Path(__file__).parent.parent
FIXTURES = SKILL_ROOT / "tests" / "fixtures"


def _outcome(reader, *args):
    try:
        return "returned", reader(*args)
    except Exception as error:  # noqa: BLE001 - the error is the outcome compared
        return "raised", type(error)


def _soup(rng, alphabet, count, longest=(6, 20, 60)):
    texts = []
    for _ in range(count):
        length = rng.randint(0, rng.choice(longest))
        texts.append("".join(rng.choice(alphabet) for _ in range(length)))
    return texts


#: Whitespace the regexes' ``\s`` and ``str.isspace`` must agree on.
_SPACES = (" ", "  ", "\t", "\n", "\n\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x85",
           "\xa0", " ", "　")


# =================================================================================
# task_decomposer: the task list implement and check-tasks parse
# =================================================================================

_I = re.IGNORECASE
_OLD_HUMAN_TASK_PATTERNS = (
    re.compile(r'(?:^|\n)\s*(?:[-*]\s+)?\*{0,2}type\*{0,2}\s*:\s*\*{0,2}manual\b', _I),
    re.compile(r'(?:^|\n)\s*(?:[-*]\s+)?\*{0,2}executor(?:\s+role)?\*{0,2}\s*:\s*'
               r'\*{0,2}human\b', _I),
    re.compile(r'(?:^|\n)\s*(?:[-*]\s+)?\*{0,2}subagent(?:\s+type)?\*{0,2}\s*:\s*'
               r'\*{0,2}human\b', _I),
)
_LABEL = r"(?:dependenc(?:y|ies)|depends?\s+on)"
_OLD_DEPENDENCIES_LINE = re.compile(
    r"^\s*(?:[-*]\s+)?(?:\*{2}\s*" + _LABEL + r"\s*(?::\s*\*{2}|\*{2}\s*:)|" + _LABEL
    + r"\s*:)(?P<ids>[^\n]*)$", _I | re.MULTILINE)
_OLD_ID_LIST = r"\s*T?\d+(?:\s*-\s*T?\d+)?(?:\s*,\s*T?\d+(?:\s*-\s*T?\d+)?)*\s*,?\s*"
_OLD_TASK_SECTION_PATTERN = (
    r'(?:#{2,3})\s*(?:Step|Task)?\s*([T]?\d+):?\s*(.+?)'
    r'(?=\n#{2,3}\s*(?:Step|Task)?\s*[T]?\d|\n## |\Z)'
)
_OLD_FILES_LINE = re.compile(
    r'^\s*(?:\*{2}\s*files?\s+to\s+(modify|create)\s*\*{2}'
    r'|[-*]\s+files?\s+to\s+(modify|create))\s*:\s*(?P<paths>[^\n]*)$', _I | re.MULTILINE)
_OLD_COMPLEXITY_LINE = re.compile(
    r'^\s*(?:\*{2}\s*complexity\s*\*{2}|[-*]\s+complexity)\s*:\s*'
    r'\**\s*(?P<level>low|medium|high)\b', _I | re.MULTILINE)


def _old_explicit_files(description):
    found = {}
    for match in _OLD_FILES_LINE.finditer(description):
        action = (match.group(1) or match.group(2) or "").lower()
        paths = found.setdefault(action, [])
        for raw in match.group("paths").split(","):
            item = raw.strip().strip("`\"'").strip()
            if not item or item.lower().strip("()") in ("none", "n/a", "-", "—"):
                continue
            if item not in paths:
                paths.append(item)
    return found or None


def _old_extract_files(description, action):
    explicit = _old_explicit_files(description)
    if explicit is not None:
        return list(explicit.get(action, []))
    files = []
    all_paths = re.findall(r'[`"]?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[`"]?', description)
    for path in all_paths:
        context_start = max(0, description.find(path) - 50)
        context = description[context_start:description.find(path) + len(path) + 50].lower()
        if action == "create" and any(w in context for w in ["create", "new", "add", "write"]):
            if path not in files:
                files.append(path)
        elif action == "modify" and any(w in context for w in ["modify", "update", "edit", "change"]):
            if path not in files:
                files.append(path)
    return files


def _old_extract_acceptance_criteria(description):
    heading_pattern = r'(?:^|\n)\s*\*{0,2}acceptance\s+criteria\*{0,2}\s*:?\s*\n'
    match = re.search(heading_pattern, description, re.IGNORECASE)
    if not match:
        return []
    rest = description[match.end():]
    next_section = re.search(r'\n\s*\*{2}[A-Za-z]', rest)
    if next_section:
        rest = rest[:next_section.start()]
    criteria = []
    for line in rest.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        bullet_match = re.match(r'^[-*]\s+(?:\[.\]\s+)?(.+)$', stripped)
        if bullet_match:
            item = bullet_match.group(1).strip()
            if item:
                criteria.append(item)
            continue
        numbered_match = re.match(r'^\d+[.)]\s+(.+)$', stripped)
        if numbered_match:
            item = numbered_match.group(1).strip()
            if item:
                criteria.append(item)
            continue
    return criteria


def _old_strip_metadata_sections(description):
    section_names = [
        r'files?\s+to\s+modify', r'files?\s+to\s+create', r'dependenc(?:y|ies)',
        r'depends?\s+on', r'complexity', r'subagent(?:\s+type)?',
        r'executor(?:\s+role)?', r'acceptance\s+criteria',
    ]
    result = description
    for name in section_names:
        pattern = (
            r'(?:(?<=\n)|(?:^))\s*'
            r'(?:\*{2}' + name + r'\*{2}'
            r'|[-*]\s+' + name + r')'
            r'\s*:?\s*'
            r'[^\n]*'
            r'(?:\n(?!\s*\n)(?!\s*\*{2}[A-Za-z])(?!\s*#{2,})[^\n]*)*'
        )
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


#: Pieces of a task list, the renderer's metadata lines among them.
_TASK_ALPHABET = _SPACES + (
    "## ", "### ", "##", "#", "Task ", "Step ", "task", "STEP", "T", "T001", "T002", "1", "23",
    ": ", ":", "Title", "**", "*", "- ", "* ", "-", "—", "(", ")", ";", ",", ".", "/", "`",
    "**Dependencies**: ", "**Depends on**:", "**Dependencies:**", "Depends on: ", "dependency:",
    "None", "N/A", "T001, T002", "T001-T003", "1-2", "(see Step 2)",
    "**Files to modify**: ", "- Files to create: ", "files to modify", "a.py", "src/b.ts",
    "`c.md`", "x.tar.gz", "none",
    "**Complexity**: ", "- complexity:", "low", "high", "medium", "**", "complex",
    "**Type**: Manual", "**Executor**: human", "- **Subagent type**: human", "manual",
    "**Acceptance Criteria**:", "acceptance criteria", "- [ ] item", "- [x] done", "1. item",
    "2) two", "create ", "modify ", "update ", "x", "word ", "é", "٣",
)


def _task_id(raw):
    raw = str(raw)
    return raw if raw.startswith("T") else f"T{int(raw):03d}"


def _rendered_task_lists():
    """``tasks.md`` as ``update_plan_tasks`` renders it, from the fixtures'
    task lists and a few made here (human tasks, empty file lists)."""
    rendered = []
    for path in (FIXTURES / "sample_tasks.json",
                 FIXTURES / "e2e/responses/generate_tasks/tasks.json",
                 FIXTURES / "e2e/responses/generate_tasks/tasks_v1.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = data["tasks"] if isinstance(data, dict) else data
        for task in tasks:
            task["id"] = _task_id(task["id"])
            task["depends_on"] = [_task_id(d) for d in task.get("depends_on", [])]
            task.setdefault("files_to_modify", task.get("context_files", []))
            task.setdefault("files_to_create", task.get("output_files", []))
        rendered.append(generate_tasks_markdown(tasks, path.name))
    made = [
        {"id": "T001", "title": "Ask the user", "description": "Grant access.\n\n\n  ",
         "executor_role": "human", "acceptance_criteria": ["done", "  spaced  "]},
        {"id": "T002", "title": "Build", "depends_on": ["T001"],
         "description": "Write src/a.py and update docs/b.md.\n\n**Note**: x",
         "files_to_modify": [], "estimated_complexity": "low"},
    ]
    rendered.append(generate_tasks_markdown(made, "made.md"))
    return rendered


def _task_corpus(seed):
    rng = random.Random(seed)
    texts = _rendered_task_lists()
    texts += [(FIXTURES / "sample_plan.md").read_text(encoding="utf-8"),
              (FIXTURES / "e2e/plans/auth-feature.md").read_text(encoding="utf-8")]
    # Each rendered list with its lines' whitespace runs stretched.
    texts += [text.replace("\n", "\n\n \n") for text in texts[:4]]
    texts += _soup(rng, _TASK_ALPHABET, 4000, longest=(8, 30, 120))
    return texts


class TestTaskDecomposerReadsWhatItsRegexesRead:

    def test_manual_markers(self):
        for text in _task_corpus(1):
            for old, new in zip(_OLD_HUMAN_TASK_PATTERNS, td.HUMAN_TASK_PATTERNS):
                a, b = old.search(text), new.search(text)
                assert (a is None) == (b is None) and (a is None or a.end() == b.end()), text

    def test_dependency_lines(self):
        for text in _task_corpus(2):
            assert [(m.group("ids"), m.end()) for m in td.DEPENDENCIES_LINE.finditer(text)] == [
                (m.group("ids"), m.end()) for m in _OLD_DEPENDENCIES_LINE.finditer(text)], text

    def test_id_only_bullets(self):
        rng = random.Random(3)
        alphabet = _SPACES + ("T", "1", "01", "-", ",", " , ", " - ", "x", "T002")
        for text in _soup(rng, alphabet, 20000) + _task_corpus(3)[:50]:
            assert bool(td._ID_LIST.fullmatch(text)) == bool(re.fullmatch(_OLD_ID_LIST, text)), text

    def test_task_sections(self):
        flags = re.DOTALL | re.IGNORECASE
        for text in _task_corpus(4):
            assert re.findall(td.TASK_SECTION_PATTERN, text, flags) == re.findall(
                _OLD_TASK_SECTION_PATTERN, text, flags), text
            assert plan_updater.has_implementation_tasks(text) == (
                plan_updater.tasks_marker(text) is not None
                or plan_updater.GENERATED_TASKS_START in text
                or bool(re.findall(_OLD_TASK_SECTION_PATTERN, text, flags)))

    def test_files_and_complexity_lines(self):
        for text in _task_corpus(5):
            assert [(m.group(1), m.group(2), m.group("paths"), m.end())
                    for m in td.TaskDecomposer._FILES_LINE.finditer(text)] == [
                (m.group(1), m.group(2), m.group("paths"), m.end())
                for m in _OLD_FILES_LINE.finditer(text)], text
            a = _OLD_COMPLEXITY_LINE.search(text)
            b = td.TaskDecomposer._COMPLEXITY_LINE.search(text)
            assert (a and (a.group("level"), a.end())) == (b and (b.group("level"), b.end())), text

    def test_files_from_the_narrative(self):
        decomposer = td.TaskDecomposer()
        rng = random.Random(6)
        paths = ("a.py", "src/b.ts", "x.tar.gz", ".hidden", "no_ext", "a..b", "-x.y-", "c.md")
        alphabet = _SPACES + paths + ("create ", "modify ", "`", '"', "/", ".", "-", "_", "x")
        for text in _task_corpus(6) + _soup(rng, alphabet, 4000):
            for action in ("modify", "create"):
                assert decomposer._extract_files(text, action) == _old_extract_files(
                    text, action), text

    def test_acceptance_criteria_and_the_stripped_narrative(self):
        decomposer = td.TaskDecomposer()
        for text in _task_corpus(7):
            assert decomposer._extract_acceptance_criteria(text) == (
                _old_extract_acceptance_criteria(text)), text
            assert decomposer._strip_metadata_sections(text) == (
                _old_strip_metadata_sections(text)), text


# =================================================================================
# suggestion_batcher and suggestion_processor: a reviewer's `reference`
# =================================================================================

_OLD_SECTION_PATTERN = re.compile(
    r'(?:###?\s*)?(?:Step\s*)?(\d+)?[:\s.-]*(.+?)(?:\s*$)', re.IGNORECASE)


def _old_normalize_section_reference(reference):
    if not reference:
        return "unknown"
    reference = reference.strip()
    file_match = suggestion_batcher.FILE_PATTERN.search(reference)
    if file_match:
        return f"file:{file_match.group(1).lower()}"
    section_match = _OLD_SECTION_PATTERN.match(reference)
    if section_match:
        step_num = section_match.group(1)
        section_name = section_match.group(2).strip().lower()
        if step_num:
            return f"step_{step_num}"
        normalized = re.sub(r'[^\w\s]', '', section_name)
        normalized = re.sub(r'\s+', '_', normalized)
        return f"section:{normalized[:30]}"
    normalized = re.sub(r'[^\w\s]', '', reference.lower())
    normalized = re.sub(r'\s+', '_', normalized)
    return f"ref:{normalized[:30]}"


def _fixture_references():
    references = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("reference", "section") and isinstance(item, str):
                    references.append(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in (FIXTURES / "review_plan_turbo_boost.json",
                 *sorted((FIXTURES / "e2e/responses").rglob("*.json"))):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    assert len(references) > 50
    return references


_REFERENCE_ALPHABET = _SPACES + (
    "#", "##", "###", "Step", "step", "STEP", "ſtep", "1", "12", "٣", ":", ".", "-",
    "/", " / ", ",", " , ", "x", "Name", "a b", "T", "3.", "File: a.py", "(", ")",
)


def _reference_corpus(seed):
    rng = random.Random(seed)
    edges = ["", " ", "\n", "Step 1", "### Step 3: Create", "## 2. x", "x  \ny", "##  \nx",
             "### Step  12:  name  ", "a , ,b", "a / / b", "a /, b", " / x", "x / ", ",",
             "## Design / ### 7. Lap", "### 5. Visual/Audio Effects (Juice)"]
    return edges + _fixture_references() + _soup(rng, _REFERENCE_ALPHABET, 30000, (5, 10, 25))


class TestReferencesReadAsTheirRegexesRead:

    def test_section_parts(self):
        for text in _reference_corpus(11):
            match = _OLD_SECTION_PATTERN.match(text)
            assert suggestion_batcher._section_parts(text) == (
                match and (match.group(1), match.group(2))), repr(text)
            assert suggestion_batcher.normalize_section_reference(text) == (
                _old_normalize_section_reference(text)), repr(text)

    def test_reference_split(self):
        old = re.compile(r"\s+/\s+|\s*,\s*")
        for text in _reference_corpus(12):
            assert suggestion_processor._split_reference(text) == old.split(text), repr(text)
            tokens = {part.strip().lstrip("#").strip().lower()
                      for part in old.split(text) if part.strip()}
            assert suggestion_processor._reference_tokens({"reference": text}) == tokens


# =================================================================================
# write_batch_scheduler: a write subagent's check findings
# =================================================================================

_OLD_FINDING_EXIT = re.compile(
    r"\(\s*(?:exit(?:ed|s)?|returned)(?:\s+(?:with|code|status))*\s*[:=]?\s*-?\d+\s*\)"
    r"|\b(?:exit(?:ed|s)?|returned)(?:\s+with)?\s+(?:code|status)\s*[:=]?\s*-?\d+\b"
    r"|\b(?:exit(?:ed|s)?|returned)(?:\s+with)?\s*[:=]?\s*-?\d+"
    r"(?=\s*(?:$|[),;:!?]|\.(?!\d)|(?:and|but|with|then)\b))"
)
_OLD_FINDING_ACCEPTANCE_NOTE = re.compile(
    r"\s+\((?:the )?(?:problems?|findings?) the user accepted from "
    r"(?P<check_id>T\d+)\)\.?\s*$"
)
_FINDING_ALPHABET = _SPACES + (
    "(", ")", "exit", "exited", "exits", "returned", "with", "code", "status", ":", "=", "-",
    "1", "23", ".", ",", "and", "then", "x", "npm test", "check failed:",
    " (problems the user accepted from T1)", " (the findings the user accepted from T12).",
    "problem", "finding", " the user accepted from ", "T3", "the ",
)


class TestFindingsReadAsTheirRegexesRead:

    def test_exit_clauses_and_acceptance_notes(self):
        rng = random.Random(21)
        for text in _soup(rng, _FINDING_ALPHABET, 30000, (6, 15, 40)):
            for variant in (text, " ".join(text.lower().split())):
                assert wbs._FINDING_EXIT.sub(" ", variant) == _OLD_FINDING_EXIT.sub(
                    " ", variant), repr(variant)
            a = _OLD_FINDING_ACCEPTANCE_NOTE.search(text)
            b = wbs._FINDING_ACCEPTANCE_NOTE.search(text)
            assert (a and (a.span(), a.group("check_id"))) == (
                b and (b.span(), b.group("check_id"))), repr(text)


# =================================================================================
# plan_updater: the plan finalisation rewrites
# =================================================================================

_OLD_IMPL_SUMMARY = re.compile(plan_updater.IMPL_SUMMARY_REFERENCE_PATTERN)
_OLD_IMPL_SECTION = re.compile(
    r'## Implementation Summary\s*\n.*?' + plan_updater.IMPL_SUMMARY_REFERENCE_PATTERN,
    re.DOTALL)
_OLD_TASK_HEADER = r'(?:#{2,3})\s*(?:Task\s+)?([T]?\d+):?\s*(.+?)(?=\n)'


def _old_insert_implementation_summary_reference(content, summary_file):
    reference = plan_updater.IMPL_SUMMARY_REFERENCE_TEMPLATE.format(summary_file=summary_file)
    reference_section = f"""
## Implementation Summary

See [{summary_file}]({summary_file}) for the implementation summary including files modified and any deviations from the plan.

{reference}
"""
    if _OLD_IMPL_SUMMARY.search(content):
        replacement = reference_section.strip()
        return _OLD_IMPL_SECTION.sub(lambda _m: replacement, content)
    return content.rstrip() + "\n" + reference_section


def _old_get_implementation_summary_path(content):
    match = _OLD_IMPL_SUMMARY.search(content)
    return match.group(1).strip() if match else None


def _old_task_headers(content):
    return re.findall(_OLD_TASK_HEADER, content, re.IGNORECASE)


_PLAN_ALPHABET = _SPACES + (
    "<!-- IMPL_SUMMARY:", "-->", "->", "-", ">", "a", "b.md", "## Implementation Summary",
    "<!--", "x y", "## Task 1:", "### T2", "## 3", ":", "Title", "#", "\\p",
)


def _plan_corpus(seed):
    rng = random.Random(seed)
    plans = [(FIXTURES / "sample_plan.md").read_text(encoding="utf-8"),
             (FIXTURES / "e2e/plans/auth-feature.md").read_text(encoding="utf-8")]
    edges = [
        "<!-- IMPL_SUMMARY: a.md -->", "<!-- IMPL_SUMMARY:   -->", "<!-- IMPL_SUMMARY:\n\n -->",
        "<!-- IMPL_SUMMARY: --> b -->", "<!-- IMPL_SUMMARY: a <!-- IMPL_SUMMARY: b\nc -->",
        "<!-- IMPL_SUMMARY:a\n-->", "## Implementation Summary\n\nold\n<!-- IMPL_SUMMARY: o.md -->",
        "## Implementation Summary <!-- IMPL_SUMMARY: o.md -->",
        "## Implementation Summary\n<!-- IMPL_SUMMARY:\n -->\n## Implementation Summary\n"
        "<!-- IMPL_SUMMARY: z -->",
    ]
    marked = [plan + "\n## Implementation Summary\n\nSee x.\n\n<!-- IMPL_SUMMARY: s.md -->\n"
              for plan in plans]
    return plans + marked + edges + _soup(rng, _PLAN_ALPHABET, 20000, (6, 15, 40))


class TestPlanUpdaterReadsWhatItsRegexesRead:

    def test_the_summary_marker(self):
        for text in _plan_corpus(31):
            old = _OLD_IMPL_SUMMARY.search(text)
            assert plan_updater._ImplSummaryMarkers(text).search() == (
                old and (old.start(), old.end(), old.start(1), old.end(1))), repr(text)
            assert plan_updater.get_implementation_summary_path(text) == (
                _old_get_implementation_summary_path(text)), repr(text)

    def test_the_summary_section_rewrite(self):
        for text in _plan_corpus(32):
            assert plan_updater._ImplSummaryMarkers(text).sections() == [
                m.span() for m in _OLD_IMPL_SECTION.finditer(text)], repr(text)
            for summary in ("s.md", "..\\.multi-llm\\p\\sum.md"):
                assert _outcome(plan_updater.insert_implementation_summary_reference,
                                text, summary) == _outcome(
                    _old_insert_implementation_summary_reference, text, summary), repr(text)

    def test_the_task_list(self):
        for text in _plan_corpus(33) + _task_corpus(34)[:2000]:
            assert [(task["id"], task["title"]) for task in plan_updater.extract_task_list(text)] == [
                (f"T{raw}" if not raw.startswith("T") else raw, title.strip())
                for raw, title in _old_task_headers(text)], repr(text)


# =================================================================================
# Bounded time, in a child process
# =================================================================================


def _run_timed(body, what):
    script = textwrap.dedent(f"""
        import contextlib, io, json, sys, time
        sys.path.insert(0, {str(SKILL_ROOT)!r})
        size = 1024 * 1024
    """) + textwrap.dedent(body) + textwrap.dedent("""
        print(json.dumps({"elapsed": elapsed}))
    """)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{what} took over 60 s on 1 MiB: a quadratic regex is back "
                    "(see the rule over task_decomposer.HUMAN_TASK_PATTERNS)")
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1])["elapsed"] < 20


class TestTheReadersAreLinear:
    """At 32 KB of the shape named, each of these took 0.5 to 12 s with its
    regex, four times that per doubling (``SECTION_PATTERN`` and the
    IMPL_SUMMARY marker far worse). A megabyte now takes well under a second."""

    def test_parse_plan_reads_a_megabyte_task_list_in_bounded_time(self):
        """``implement`` parses a generated task list synchronously."""
        _run_timed("""
            from utils.task_decomposer import TaskDecomposer
            head = "## T001: Title\\n"
            pumps = [
                "x" + "\\n" * size + "x",
                "x" + "\\n " * (size // 2) + "x",
                "##" + " " * size + "x",
                "**Complexity**:" + " " * size + "x",
                "**Acceptance Criteria**" + " " * size + "x",
                "**Acceptance Criteria**:\\n- a" + "\\n" * size + "x",
                "**Dependencies**:\\n- 1" + " " * size + "x",
                "a" * size,
                "a/b-c_d" * (size // 7),
                "- Files to modify" + "\\n" * size,
            ]
            started = time.perf_counter()
            for pump in pumps:
                TaskDecomposer().parse_plan(head + pump)
            elapsed = time.perf_counter() - started
        """, "TaskDecomposer.parse_plan")

    def test_a_megabyte_reference_is_grouped_in_bounded_time(self):
        """``batch_suggestions`` keys and ``group_suggestions`` weighs by the
        reviewer's ``reference``."""
        _run_timed("""
            from utils.suggestion_batcher import normalize_section_reference
            from utils.suggestion_processor import compute_section_weights
            references = ["x" + " " * size + "\\ny", "##" + " " * size + "x\\ny",
                          "### Step" + " " * size + "1" * size + "\\nz",
                          "x" + " " * size + "y", "x" + " ," * (size // 2) + "y"]
            started = time.perf_counter()
            for reference in references:
                normalize_section_reference(reference)
            compute_section_weights([{"reference": r} for r in references])
            elapsed = time.perf_counter() - started
        """, "normalize_section_reference or _reference_tokens")

    def test_a_megabyte_finding_is_matched_in_bounded_time(self):
        """``split_accepted_findings`` reads the findings a write subagent
        reported, verbatim."""
        _run_timed("""
            from utils.write_batch_scheduler import split_accepted_findings
            findings = ["npm test: x" + " " * size + "y",
                        "npm test: exit" + "\\n" * size + "x",
                        "npm test: (exit" + " " * size + "x",
                        "npm test: exit code" + " " * size + "x"]
            accepted = {"T001": ["npm test: x y"]}
            started = time.perf_counter()
            split_accepted_findings(findings, accepted)
            elapsed = time.perf_counter() - started
        """, "split_accepted_findings")

    def test_a_megabyte_plan_is_finalised_in_bounded_time(self):
        """Finalisation reads and rewrites the summary reference in a plan the
        edit batches may have rewritten."""
        _run_timed("""
            from utils.plan_updater import (
                extract_task_list, get_implementation_summary_path,
                insert_implementation_summary_reference)
            plans = [
                "<!-- IMPL_SUMMARY:" + " " * size + "x",
                "<!-- IMPL_SUMMARY:x" * (size // 19),
                "## Implementation Summary\\n<!-- IMPL_SUMMARY:" + " " * size + "x",
                "<!-- IMPL_SUMMARY: a --> ## Implementation Summary" + "\\n" * size + "x",
                "<!-- IMPL_SUMMARY: a -->" + "## Implementation Summary\\n" * (size // 26),
                "## 1:" + " " * size + "x",
                "## 1: x" * (size // 7),
            ]
            started = time.perf_counter()
            for plan in plans:
                get_implementation_summary_path(plan)
                insert_implementation_summary_reference(plan, "s.md")
                extract_task_list(plan)
            elapsed = time.perf_counter() - started
        """, "plan_updater's summary or task readers")
