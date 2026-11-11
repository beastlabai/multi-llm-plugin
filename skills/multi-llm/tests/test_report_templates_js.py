"""The report pages' JavaScript, run.

Every report page (``report.html`` in the flat and PR styles,
``consolidated-report.html``) is its template plus the code the three share,
``templates/shared/report_core.js`` and ``report_core.css``, inlined at
generation (docs/development.md#shared-report-code). These tests generate each
page from fixtures and run its code:

* every inline script on every page parses, and no page redefines a shared
  function (Node);
* the escapers turn hostile text into text and whole attribute values, read
  back by an HTML parser and, for an inline handler, by the JS engine (Node);
* a selection survives export, reload and the Python reader the apply phase
  uses, and a hostile stored one restores nothing it should not (Node, with a
  stand-in for the few browser objects the selection code touches);
* the real vendored marked and DOMPurify, in headless Chrome, turn hostile
  markdown into the markdown vocabulary only, and nothing fires.

The Node checks skip without ``node`` and the browser ones without Chrome, so
no CI leg newly needs either; the generator checks at the top always run.
``tests/test_html_report_scripts.py`` holds the static checks on the sources.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import commit_report  # noqa: E402
from utils import html_report_generator as hrg  # noqa: E402
from utils.consolidation import (  # noqa: E402
    generate_consolidated_html,
    generate_consolidated_id,
    generate_group_id,
)
from utils.html_report_generator import (  # noqa: E402
    SHARED_ASSETS,
    SLOT_CORE_SCRIPT,
    SLOT_CORE_STYLE,
    embed_report_data,
    generate_html_report,
    load_report_template,
    script_json,
)
from utils.report_parser import (  # noqa: E402
    load_consolidated_html_selections,
    load_html_selections,
)
from utils.state_manager import stamp_stable_ids  # noqa: E402

SHARED = Path(__file__).parent.parent / "templates" / "shared"
NODE = shutil.which("node")
CHROME = next((p for p in map(shutil.which, ("google-chrome", "google-chrome-stable",
                                             "chromium", "chromium-browser")) if p), None)

needs_node = pytest.mark.skipif(NODE is None, reason="needs node on PATH")
needs_chrome = pytest.mark.skipif(CHROME is None, reason="needs Chrome or Chromium on PATH")


# ---------------------------------------------------------------------------
# The pages
# ---------------------------------------------------------------------------

PAGE_KINDS = ("flat/review-plan", "pr/review-plan", "flat/commit-review", "pr/commit-review",
              "consolidated")
HEAD = "c" * 40


def _groups(code: bool) -> List[Dict[str, Any]]:
    def suggestion(title: str, desc: str) -> Dict[str, Any]:
        s = {"title": title, "desc": desc, "description": desc, "type": "modification",
             "importance": "HIGH", "model": "m:a", "source_model": "m:a"}
        if code:
            s.update(file="a.py", line_range=[3, 4], anchor_text="x = 1")
        else:
            s["reference"] = "Task 1"
        return s
    groups = [
        {"theme": "First", "category": "bug", "reference": "Task 1", "models": ["m:a"],
         "importance": "HIGH", "validation_status": "valid", "validation_reason": "ok",
         "suggestions": [suggestion("First finding", "Do **this**.")]},
        {"theme": "Second", "category": "bug", "reference": "Task 1", "models": ["m:a"],
         "importance": "LOW", "validation_status": "needs-human-decision",
         "validation_reason": "unsure", "suggestions": [suggestion("Second finding", "Maybe.")]},
    ]
    stamp_stable_ids(groups)
    return groups


def _commit_review() -> Dict[str, Any]:
    return commit_report.report_payload({
        "base": "b" * 40, "head": HEAD,
        "commits": [{"sha": HEAD, "date": "2026-11-01", "subject": "Fix it",
                     "merge": False, "outside_window": False}],
        "phrase": "last commit", "window": None, "notes": [],
        "size": {"commits": 1, "files": 1, "added": 2, "deleted": 1, "lines": 3, "large": False},
        "brief": "plans/commit-reviews/x.md", "drifted": ["a.py"],
    }, drifted=["a.py"], scope=["a.py"])


def build_page(kind: str, tmp_path: Path) -> Tuple[str, Path, str]:
    """``(html, phase_dir, plan_path)`` for one page kind."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\n## Task 1\n\nDo it.\n", encoding="utf-8")
    if kind == "consolidated":
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        groups = _groups(code=False)
        ids = [generate_group_id(g) for g in groups]
        consolidated = [{
            "consolidated_id": generate_consolidated_id(ids), "display_index": 1,
            "title": "Combined", "description": "Both **at once**.", "importance": "HIGH",
            "reference": "Task 1", "type": "modification", "underlying_group_indices": [0, 1],
            "underlying_group_ids": ids, "model_count": 1, "is_singleton": False,
        }]
        path = generate_consolidated_html(consolidated, groups, str(phase_dir), str(plan), ["m:a"])
        return Path(path).read_text(encoding="utf-8"), phase_dir, str(plan)
    style, review = kind.split("/")
    phase_dir = tmp_path / f"{style}-{review}"
    phase_dir.mkdir()
    commit = review == "commit-review"
    html = generate_html_report(
        groups=_groups(code=commit), plan_path=plan, phase_dir=phase_dir,
        phase_type="code-review" if commit else "review-plan", models=["m:a"],
        template_style=style, commit_review=_commit_review() if commit else None)
    return html, phase_dir, str(plan)


class _Scripts(HTMLParser):
    """Each ``<script>`` element's attributes and text, in page order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scripts: List[Tuple[Dict[str, str], str]] = []
        self._open: Any = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._open = ({k: v or "" for k, v in attrs}, [])

    def handle_endtag(self, tag):
        if tag == "script" and self._open is not None:
            self.scripts.append((self._open[0], "".join(self._open[1])))
            self._open = None

    def handle_data(self, data):
        if self._open is not None:
            self._open[1].append(data)


def page_scripts(html: str) -> List[Dict[str, str]]:
    """``[{name, text}]``: every inline script, named by what it is."""
    parser = _Scripts()
    parser.feed(html)
    parser.close()
    out = []
    for index, (attrs, text) in enumerate(parser.scripts):
        name = attrs.get("data-vendored") or attrs.get("data-shared") or f"page-script-{index}"
        out.append({"name": name, "text": text})
    return out


def embedded_data(html: str) -> Dict[str, Any]:
    match = re.search(r"(?:const|var) reportData = (\{.*?\n\});", html, re.S)
    return json.loads(match.group(1).replace("<\\/", "</").replace("<\\!--", "<!--"))


# ---------------------------------------------------------------------------
# The generator inlines the shared files (no node needed)
# ---------------------------------------------------------------------------

class TestTheSharedFilesAreInlined:

    @pytest.mark.parametrize("kind", PAGE_KINDS)
    def test_each_page_carries_each_file_once_and_in_order(self, kind, tmp_path):
        html, _phase_dir, _plan = build_page(kind, tmp_path)
        for slot in SHARED_ASSETS:
            assert slot not in html
        css = (SHARED / "report_core.css").read_text(encoding="utf-8")
        js = (SHARED / "report_core.js").read_text(encoding="utf-8")
        assert html.count('<style data-shared="report_core.css">') == 1 and html.count(css) == 1
        assert html.count('<script data-shared="report_core.js">') == 1 and html.count(js) == 1
        # Its styles come before the page's (the page wins a tie), its script
        # after the vendored libraries and before the page's.
        assert html.index('data-shared="report_core.css"') < html.index("<style>")
        assert html.rindex("data-vendored=") < html.index('data-shared="report_core.js"')
        assert html.index('data-shared="report_core.js"') < html.rindex("<script>")

    def test_report_data_is_never_taken_for_a_slot(self, tmp_path):
        """Slots are filled before the data goes in: a description spelling
        a slot stays data, script-escaped, and the page has one copy."""
        template = load_report_template("flat")
        data = {"groups": [{"theme": SLOT_CORE_SCRIPT, "suggestions": [
            {"description": SLOT_CORE_STYLE + " " + SLOT_CORE_SCRIPT}]}]}
        html = embed_report_data(data, template)
        assert html.count('<script data-shared="report_core.js">') == 1
        assert html.count('<style data-shared="report_core.css">') == 1
        assert SLOT_CORE_SCRIPT.replace("<!--", "<\\!--") in html

    @pytest.fixture
    def shared_copy(self, tmp_path, monkeypatch):
        copy = tmp_path / "shared"
        shutil.copytree(SHARED, copy)
        monkeypatch.setattr(hrg, "_SHARED_DIR", copy)
        return copy

    def test_a_missing_file_fails_like_a_missing_template(self, shared_copy, tmp_path):
        (shared_copy / "report_core.js").unlink()
        with pytest.raises(FileNotFoundError) as caught:
            embed_report_data({}, load_report_template("pr"))
        assert caught.value.filename == str(shared_copy / "report_core.js")
        html, _phase_dir, _plan = build_page("flat/review-plan", tmp_path)
        assert "Template Not Found" in html and "report_core.js" in html

    @pytest.mark.parametrize("bad", ["var s = '</script>';", "/* <!-- */", "a { b: '</style>' }",
                                     "var p = 'REPORT_DATA_PLACEHOLDER';"])
    def test_a_file_that_would_break_its_element_is_refused(self, shared_copy, bad):
        (shared_copy / "report_core.css").write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match="cannot be inlined"):
            embed_report_data({}, load_report_template("consolidated"))

    def test_the_shipped_files_hold_nothing_refused(self):
        for _slot, (filename, _element) in SHARED_ASSETS.items():
            text = (SHARED / filename).read_text(encoding="utf-8")
            assert not hrg._SHARED_FORBIDDEN.search(text), filename


# ---------------------------------------------------------------------------
# Running page code under Node
# ---------------------------------------------------------------------------

#: The browser objects the code under test touches, and no more: storage, the
#: download link, Blob/URL, and a document whose lookups find nothing (each
#: page's render code is not run here; tests/test_scope_report_tags.py does).
#: Scripts run in one vm context, in page order, as a browser runs them.
_HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
const ctx = vm.createContext({ __host: { out: function (v) { out.push(JSON.stringify(v)); } } });
vm.runInContext(String.raw`
var __store = {}, __blobs = [], __downloads = [], __errors = [], __storageThrows = false;
var console = { log: function () {}, warn: function () {},
                error: function () { __errors.push(Array.prototype.join.call(arguments, ' ')); } };
var localStorage = {
  getItem: function (k) { if (__storageThrows) throw new Error('denied');
                          return Object.prototype.hasOwnProperty.call(__store, k) ? __store[k] : null; },
  setItem: function (k, v) { if (__storageThrows) throw new Error('denied'); __store[k] = String(v); },
  removeItem: function (k) { delete __store[k]; }
};
function El(tag) {
  this.tagName = String(tag || '').toUpperCase(); this.style = {}; this.dataset = {};
  this.classList = { add: function () {}, remove: function () {}, toggle: function () { return false; },
                     contains: function () { return false; } };
}
El.prototype = {
  appendChild: function (c) { return c; }, removeChild: function (c) { return c; },
  setAttribute: function () {}, getAttribute: function () { return null; },
  addEventListener: function () {}, querySelector: function () { return null; },
  querySelectorAll: function () { return []; },
  click: function () { if (this.tagName === 'A') __downloads.push({ name: this.download, href: this.href }); }
};
var document = {
  readyState: 'loading', body: new El('body'), documentElement: new El('html'),
  addEventListener: function () {}, getElementById: function () { return null; },
  querySelector: function () { return null; }, querySelectorAll: function () { return []; },
  createElement: function (t) { return new El(t); }
};
var window = this; window.addEventListener = function () {};
function Blob(parts) { this.text = parts.join(''); __blobs.push(this.text); }
var URL = { createObjectURL: function () { return 'blob:' + (__blobs.length - 1); },
            revokeObjectURL: function () {} };
function setTimeout() { return 0; }
function __out(v) { __host.out(v); }
`, ctx, { filename: 'harness.js' });
vm.runInContext('__store = ' + JSON.stringify(input.store || {}) + ';', ctx);
let core = null;
for (const s of input.scripts) {
  vm.runInContext(s.text, ctx, { filename: s.name });
  if (s.name === 'report_core.js') {
    core = {};
    for (const name of input.coreFunctions) core[name] = ctx[name];
  }
}
// A page that declares a shared function again replaces the shared one.
const redefined = core ? input.coreFunctions.filter(function (n) { return ctx[n] !== core[n]; }) : null;
vm.runInContext(input.body, ctx, { filename: 'body.js' });
process.stdout.write(JSON.stringify({ out: out.map(JSON.parse), redefined: redefined }) + '\n');
"""

#: Every top-level function report_core.js declares.
CORE_FUNCTIONS = re.findall(r"^function (\w+)\(", (SHARED / "report_core.js").read_text(
    encoding="utf-8"), re.M)


def run_node(scripts: List[Dict[str, str]], body: str, store: Dict[str, str] = None) -> Dict:
    """Run *scripts* in page order in one context, then *body*, which calls
    ``__out(value)``; returns ``{out: [values], redefined: [names]}``."""
    if NODE is None:
        pytest.skip("node is not on PATH")
    with tempfile.TemporaryDirectory() as scratch:
        harness = Path(scratch) / "harness.js"
        harness.write_text(_HARNESS, encoding="utf-8")
        data = Path(scratch) / "input.json"
        data.write_text(json.dumps({"scripts": scripts, "body": body, "store": store or {},
                                    "coreFunctions": CORE_FUNCTIONS}), encoding="utf-8")
        done = subprocess.run([NODE, str(harness), str(data)], capture_output=True, text=True,
                              encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr[-3000:]
    # Split on "\n" only: the values may hold U+2028, which splitlines() breaks on.
    return json.loads(done.stdout.strip().split("\n")[-1])


def core_only() -> List[Dict[str, str]]:
    return [{"name": "report_core.js",
             "text": (SHARED / "report_core.js").read_text(encoding="utf-8")}]


@needs_node
class TestEveryPageScriptRuns:

    def test_the_shared_script_declares_functions_only(self):
        """Nothing runs at load: the file is safe to inline anywhere."""
        assert len(CORE_FUNCTIONS) > 40
        out = run_node(core_only(), "__out(Object.keys(__store).length + __errors.length);")
        assert out["out"] == [0]

    @pytest.mark.parametrize("kind", PAGE_KINDS)
    def test_every_script_parses_and_loads_and_none_redefines_a_shared_function(self, kind, tmp_path):
        html, _phase_dir, _plan = build_page(kind, tmp_path)
        scripts = page_scripts(html)
        assert [s["name"] for s in scripts][-2:] == ["report_core.js", "page-script-" + str(len(scripts) - 1)]
        out = run_node(scripts, "__out([typeof renderMarkdown, typeof exportSelections, "
                                "typeof selectionsFile, typeof getStorageKey, typeof init]);")
        assert out["out"] == [["function"] * 5]
        assert out["redefined"] == []


#: Text that tries to leave its element, its attribute (either quote) or an
#: inline handler's JS string, or to name a URL scheme that runs script.
HOSTILE_VALUES = [
    "<script>alert(1)</script>",
    '" onerror="alert(2)" x="',
    "' onmouseover='alert(3)' x='",
    '"><img src=x onerror=alert(4)>',
    "javascript:alert(5)",
    "a &amp; b &lt; c </span> d",
    "');alert(6);//",
    "back\\slash \\' and\nnew line \u2028 \u2029 end",
    "</textarea><!-- -->",
    "",
    "plain",
]


class _Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: List[Tuple[str, Dict[str, str]]] = []
        self.text: List[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, {k: v for k, v in attrs}))

    def handle_data(self, data):
        self.text.append(data)

    def handle_comment(self, data):
        self.tags.append(("!--", {}))


@needs_node
class TestTheEscapers:

    def test_text_and_attribute_values_come_back_whole(self):
        out = run_node(core_only(), "__out(%s.map(function (v) {"
                       " return '<span title=\"' + escapeAttr(v) + '\" data-q=\\'' + escapeAttr(v) +"
                       " '\\'>' + escapeHtml(v) + '</span>'; }));" % json.dumps(HOSTILE_VALUES))
        for value, markup in zip(HOSTILE_VALUES, out["out"][0]):
            parsed = _Markup()
            parsed.feed(markup)
            parsed.close()
            assert parsed.tags == [("span", {"title": value, "data-q": value})], (value, markup)
            assert "".join(parsed.text) == value
            assert not re.search(r"[<>\"']", markup.split(">", 1)[1].rsplit("<", 1)[0])

    def test_an_inline_handler_gets_the_value_back_as_one_string(self):
        made = run_node(core_only(), "__out(%s.map(function (v) {"
                        " return '<button onclick=\"pick(\\'' + escapeJsAttr(v) + '\\')\">x</button>'; }));"
                        % json.dumps(HOSTILE_VALUES))["out"][0]
        handlers = []
        for markup in made:
            parsed = _Markup()
            parsed.feed(markup)
            parsed.close()
            (tag, attrs), = parsed.tags
            assert tag == "button" and set(attrs) == {"onclick"}, markup
            handlers.append(attrs["onclick"])
        # The handler text as the browser hands it to the JS engine.
        out = run_node([], "var picked = []; function pick(v) { picked.push(v); }"
                           " %s.forEach(function (code) { new Function('pick', code)(pick); });"
                           " __out(picked);" % json.dumps(handlers))
        assert out["out"] == [HOSTILE_VALUES]

    def test_badge_classes_and_scope_values_stay_on_their_lists(self):
        out = run_node(core_only(), "__out(%s.map(function (v) { return [getImportanceBadgeClass(v),"
                       " getValidationBadgeClass(v), scopeRelation(v)]; }));" % json.dumps(HOSTILE_VALUES))
        for imp, val, rel in out["out"][0]:
            assert imp in ("badge-importance-high", "badge-importance-medium",
                           "badge-importance-low", "badge-category")
            assert val == "badge-category"
            assert rel == "unclear"


# ---------------------------------------------------------------------------
# A selection round trip
# ---------------------------------------------------------------------------

#: What each page's init seeds before it loads selections (the flat and PR
#: pages key their storage by phase and plan; the consolidated one by plan).
_SEED = """
if (Object.prototype.hasOwnProperty.call(selections, 'plan_path')) {
  selections.plan_path = reportData.planPath || '';
  selections.phase = reportData.phase || '';
}
"""


def _choose(data: Dict[str, Any], kind: str) -> Tuple[str, str]:
    """The group id a page's selections key, and a suggestion id (or '')."""
    if kind == "consolidated":
        return data["consolidatedGroups"][0]["consolidatedId"], ""
    group = data["groups"][0]
    return group["groupHash"], group["suggestions"][0]["id"]


@needs_node
class TestASelectionRoundTrip:

    @pytest.mark.parametrize("kind", PAGE_KINDS)
    def test_export_reload_and_the_apply_readers(self, kind, tmp_path):
        html, phase_dir, plan = build_page(kind, tmp_path)
        scripts = page_scripts(html)
        data = embedded_data(html)
        gid, sid = _choose(data, kind)
        choose = _SEED + """
var gid = %s, sid = %s;
selections.skipped_groups[gid] = true;
selections.validation_overrides[gid] = 'invalid';
selections._labels[gid] = 'G1';
if (selections.edited_descriptions) {
  selections.skipped_suggestions[sid] = true;
  selections.edited_descriptions[sid] = 'Edited "text" <b>';
}
exportSelections();
__out({ downloads: __downloads, file: JSON.parse(__blobs[__blobs.length - 1]),
        key: getStorageKey(), store: __store, errors: __errors });
""" % (json.dumps(gid), json.dumps(sid))
        first = run_node(scripts, choose)["out"][0]
        assert first["errors"] == []
        (download,) = first["downloads"]
        exported = first["file"]
        stored = json.loads(first["store"][first["key"]])
        assert stored["last_file_export_at"] == exported["exported_at"]
        if kind == "consolidated":
            assert download["name"] == "consolidated_user_selections.json"
            # The C-level file: decisions on consolidated groups only.
            assert set(exported) == {"plan_path", "format_version", "exported_at",
                                     "skipped_groups", "validation_overrides", "_labels"}
        else:
            assert download["name"] == "user_selections.json"
            assert exported["edited_descriptions"] == {sid: 'Edited "text" <b>'}
            assert exported["skipped_suggestions"] == {sid: True}
            assert exported["phase"] == data["phase"] and exported == stored
        assert exported["skipped_groups"] == {gid: True}
        assert exported["validation_overrides"] == {gid: "invalid"}
        assert exported["plan_path"] == data["planPath"]

        # A reload restores what was stored.
        reload = _SEED + "loadSelections(); __out(selections);"
        restored = run_node(scripts, reload, store=first["store"])["out"][0]
        for key in ("skipped_groups", "validation_overrides", "_labels", "last_file_export_at"):
            assert restored[key] == stored[key], key

        # The file is what the apply phase reads.
        (phase_dir / download["name"]).write_text(json.dumps(exported), encoding="utf-8")
        reader = load_consolidated_html_selections if kind == "consolidated" else load_html_selections
        loaded = reader(phase_dir, plan_path=data["planPath"])
        assert loaded is not None and gid in loaded["skipped_groups"]
        assert loaded["validation_overrides"][gid] == "invalid"

    @pytest.mark.parametrize("kind", PAGE_KINDS)
    def test_a_hostile_stored_selection_restores_only_what_the_page_declares(self, kind, tmp_path):
        html, _phase_dir, _plan = build_page(kind, tmp_path)
        scripts = page_scripts(html)
        key_of = run_node(scripts, _SEED + "__out(getStorageKey());")["out"][0]
        stored = {
            "skipped_groups": ["g1", "g2"],                       # a v1 list
            "validation_overrides": {"g1": "claude_decide"},      # before the judge rename
            "_labels": "not an object",
            "scope_policy": "expand", "scopeRelation": "in_scope", "selected-remedy": "x",
            "unknown_field": 1,
        }
        restored = run_node(scripts, _SEED + "loadSelections(); __out(selections);",
                            store={key_of: json.dumps(stored)})["out"][0]
        assert restored["skipped_groups"] == {"g1": True, "g2": True}
        assert restored["validation_overrides"] == {"g1": "judge_decide"}
        assert restored["_labels"] == {}
        assert not [k for k in restored if "scope" in k.lower() or "remedy" in k.lower()]
        assert "unknown_field" not in restored

    @pytest.mark.parametrize("store", ["{not json", "null", "[1, 2]", "7"])
    def test_a_broken_store_leaves_the_selection_alone(self, store, tmp_path):
        html, _phase_dir, _plan = build_page("pr/review-plan", tmp_path)
        scripts = page_scripts(html)
        key_of = run_node(scripts, _SEED + "__out(getStorageKey());")["out"][0]
        before = run_node(scripts, _SEED + "__out(selections);")["out"][0]
        after = run_node(scripts, _SEED + "loadSelections(); __out(selections);",
                         store={key_of: store})["out"][0]
        assert after == before

    def test_storage_that_refuses_does_not_stop_the_export(self, tmp_path):
        html, _phase_dir, _plan = build_page("flat/review-plan", tmp_path)
        out = run_node(page_scripts(html), _SEED + "__storageThrows = true; loadSelections();"
                       " exportSelections(); __out({ downloads: __downloads.length, errors: __errors });")
        assert out["out"][0]["downloads"] == 1
        assert len(out["out"][0]["errors"]) == 2  # one load, one save, each reported


# ---------------------------------------------------------------------------
# The real sanitiser, in a browser
# ---------------------------------------------------------------------------

#: Markdown a model controls. Each tries a script, a handler, a URL scheme, an
#: element outside the markdown vocabulary, a form control the export code
#: queries, or a sanitiser bypass (mutation XSS, entity-encoded schemes).
HOSTILE_MARKDOWN = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(2)>",
    "[click](javascript:alert(3))",
    "[click](JaVaScRiPt:alert(4))",
    "[data](data:text/html;base64,PHNjcmlwdD5hbGVydCg1KTwvc2NyaXB0Pg==)",
    '<a href="https://example.com/" onclick="alert(6)">x</a>',
    '[q](https://example.com/" onmouseover="alert(7))',
    '[t](https://example.com/ "title\\" onmouseover=\\"alert(8)")',
    '<div style="background:url(javascript:alert(9))" class="skip-checkbox" id="export-btn"'
    ' data-cg-id="forged">x</div>',
    "<svg onload=alert(10)><circle/></svg>",
    '<iframe src="javascript:alert(11)"></iframe>',
    '<input type="checkbox" class="skip-checkbox" data-cg-id="forged">',
    '<form action="https://evil.example/"><button formaction="javascript:alert(12)">go</button></form>',
    "<math><mtext><table><mglyph><style><img src=x onerror=alert(13)>",
    "<details open ontoggle=alert(14)><summary>s</summary>x</details>",
    '<a href="&#106;avascript:alert(15)">e</a>',
    "```html\n<script>alert(16)</script>\n```",
    "<style>body { display: none }</style>",
    '<a href="vbscript:msgbox(17)">v</a>',
    "<noscript><p title=\"</noscript><img src=x onerror=alert(18)>\"></noscript>",
]
BENIGN_MARKDOWN = ("| a | b |\n|:-|-:|\n| 1 | 2 |\n\n```py\nx = 1\n```\n\n"
                   "[docs](https://example.com/docs) and **bold** <kbd>K</kbd>")

#: renderMarkdown's own vocabulary (report_core.js).
_ALLOWED_TAGS = {"a", "abbr", "b", "blockquote", "br", "code", "dd", "del", "details", "div", "dl",
                 "dt", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "ins", "kbd", "li",
                 "mark", "ol", "p", "pre", "q", "s", "samp", "small", "span", "strong", "sub",
                 "summary", "sup", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "u", "ul",
                 "var"}
_ALLOWED_ATTRS = {"align", "colspan", "href", "open", "rowspan", "start", "title"}
_SAFE_HREF = re.compile(r"^(?:https?:|mailto:|#|/|\.|[^:]*$)", re.I)

#: Runs first in <head>: a handler that fires leaves its argument on <html>.
_ALERT_TRAP = ("<script>window.alert = window.msgbox = function (m) { document.documentElement"
               ".setAttribute('data-alerted', String(m)); };</script>")


def _render_cases_script(cases: List[str]) -> str:
    """Run after the page's own scripts: each case through the page's
    renderMarkdown into a ``<section data-case>`` under ``#__md``."""
    return ("<script>(function () { var root = document.createElement('div'); root.id = '__md';"
            " var cases = " + script_json(cases) + ";"
            " cases.forEach(function (text, i) { var s = document.createElement('section');"
            " s.setAttribute('data-case', String(i)); s.innerHTML = renderMarkdown(text);"
            " root.appendChild(s); }); document.body.appendChild(root); })();</script>")


class _Cases(HTMLParser):
    """Each case section's descendants: (tag, attrs), and text."""

    #: Elements with no end tag in the serialised DOM.
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
            "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cases: Dict[int, Dict[str, Any]] = {}
        self.alerted = None
        self._case = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if tag == "html":
            self.alerted = a.get("data-alerted")
        if self._case is not None:
            self._depth += tag not in self.VOID
            self.cases[self._case]["tags"].append((tag, a))
        elif tag == "section" and "data-case" in a:
            self._case = int(a["data-case"])
            self._depth = 0
            self.cases[self._case] = {"tags": [], "text": []}

    def handle_startendtag(self, tag, attrs):
        if self._case is not None:
            self.cases[self._case]["tags"].append((tag, {k: v or "" for k, v in attrs}))

    def handle_endtag(self, tag):
        if self._case is None:
            return
        if self._depth == 0 and tag == "section":
            self._case = None
        else:
            self._depth -= 1

    def handle_data(self, data):
        if self._case is not None:
            self.cases[self._case]["text"].append(data)


def rendered_cases(html: str, scratch: Path, cases: List[str]) -> _Cases:
    if CHROME is None:
        pytest.skip("no Chrome or Chromium on PATH")
    page = scratch / "page.html"
    head_at = html.index("<head>") + len("<head>")
    body_end = html.rindex("</body>")
    page.write_text(html[:head_at] + _ALERT_TRAP + html[head_at:body_end]
                    + _render_cases_script(cases) + html[body_end:], encoding="utf-8")
    done = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={scratch / 'chrome-profile'}",
         "--host-resolver-rules=MAP * ~NOTFOUND", "--virtual-time-budget=5000",
         "--dump-dom", page.as_uri()],
        # A page that runs a script can open a dialog no trap catches (one in
        # a frame) and hang: that is a failure too.
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr[-3000:]
    parsed = _Cases()
    parsed.feed(done.stdout)
    parsed.close()
    return parsed


@needs_chrome
@pytest.mark.parametrize("kind", ["flat/review-plan", "pr/commit-review", "consolidated"])
class TestTheRealSanitiser:

    def test_hostile_markdown_keeps_only_the_markdown_vocabulary(self, kind, tmp_path):
        html, _phase_dir, _plan = build_page(kind, tmp_path)
        parsed = rendered_cases(html, tmp_path, HOSTILE_MARKDOWN + [BENIGN_MARKDOWN])
        assert parsed.alerted is None
        assert sorted(parsed.cases) == list(range(len(HOSTILE_MARKDOWN) + 1))
        for index, case in parsed.cases.items():
            for tag, attrs in case["tags"]:
                assert tag in _ALLOWED_TAGS, (index, tag, attrs)
                assert set(attrs) <= _ALLOWED_ATTRS, (index, tag, attrs)
                href = re.sub(r"[\x00-\x20]", "", attrs.get("href", ""))
                assert _SAFE_HREF.match(href), (index, href)
        # A fenced block is shown as code, not run.
        fence = parsed.cases[HOSTILE_MARKDOWN.index("```html\n<script>alert(16)</script>\n```")]
        assert "<script>alert(16)</script>" in "".join(fence["text"])
        assert ("code", {}) in fence["tags"]
        # The vocabulary a plan uses survives: a table with its alignment, a
        # code block, a link and inline markup.
        benign = parsed.cases[len(HOSTILE_MARKDOWN)]["tags"]
        assert ("th", {"align": "left"}) in benign and ("td", {"align": "right"}) in benign
        assert ("pre", {}) in benign and ("code", {}) in benign
        assert ("a", {"href": "https://example.com/docs"}) in benign
        assert ("strong", {}) in benign and ("kbd", {}) in benign
