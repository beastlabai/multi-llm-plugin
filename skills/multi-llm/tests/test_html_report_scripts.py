"""The report pages' scripts: vendored, pinned, inlined, and model text sanitised.

Every report page (flat, PR, consolidated) gets ``marked`` and ``DOMPurify``
inlined from ``templates/vendor/`` at generation, so a page loads no script
from the network and reads the same offline. Model and plan text goes through
one ``renderMarkdown`` (identical in the three templates): marked renders it,
DOMPurify keeps the markdown vocabulary, and without both libraries the text is
escaped. See docs/development.md#vendored-report-scripts.

The Node checks run the real vendored ``marked`` and the templates' own
``renderMarkdown``; DOMPurify needs a DOM that Node lacks, so they check that
its output is what the page returns and what profile it is given, not what it
strips. They skip without ``node``; the static checks still run.

Every other field (status, importance, model, id, label, path, reference) goes
through the block's escapers or badge allow-lists. A static scan holds each
template to that, and the browser checks load hostile metadata into all three
pages in headless Chrome, offline, and read the rendered DOM. They skip
without Chrome.
"""

import hashlib
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

from utils import html_report_generator as hrg  # noqa: E402
from utils.consolidation import generate_consolidated_html  # noqa: E402
from utils.html_report_generator import (  # noqa: E402
    SLOT_VENDOR_SCRIPTS,
    VENDORED_SCRIPTS,
    generate_html_report,
    regenerate_report_with_human_decisions,
)

SKILL = Path(__file__).parent.parent
TEMPLATES = SKILL / "templates"
VENDOR = TEMPLATES / "vendor"
TEMPLATE_NAMES = ("report_template.html", "pr_report_template.html",
                  "consolidated_report_template.html")
NODE = shutil.which("node")

#: A description a model controls: an event handler, a script URL, a script
#: element, a form control the page's own export code would query, and a style.
HOSTILE = (
    "Fix it <img src=x onerror=alert(1)> now.\n\n"
    "[click me](javascript:alert(1))\n\n"
    "<script>alert(2)</script>\n\n"
    '<input type="checkbox" data-cg-id="forged" class="skip-checkbox">\n\n'
    "<style>body { display: none }</style>"
)


# ---------------------------------------------------------------------------
# Building the three pages
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\n## Task 1\n\nDo it.\n", encoding="utf-8")
    return plan


def _groups() -> List[Dict[str, Any]]:
    return [{
        "theme": "Hostile theme",
        "category": "bug",
        "reference": "Task 1",
        "models": ["model-a"],
        "importance": "HIGH",
        "suggestions": [{
            "title": "Hostile finding",
            "desc": HOSTILE,
            "description": HOSTILE,
            "type": "modification",
            "importance": "HIGH",
            "model": "model-a",
        }],
    }]


def flat_or_pr_page(tmp_path: Path, style: str) -> Tuple[str, Path]:
    phase_dir = tmp_path / f"review-plan-{style}"
    phase_dir.mkdir()
    html = generate_html_report(groups=_groups(), plan_path=_plan(tmp_path),
                                phase_dir=phase_dir, phase_type="review-plan",
                                models=["model-a"], template_style=style)
    return html, phase_dir


def consolidated_page(tmp_path: Path) -> str:
    from utils.consolidation import generate_consolidated_id, generate_group_id

    phase_dir = tmp_path / "review-plan"
    phase_dir.mkdir()
    groups = _groups()
    ids = [generate_group_id(g) for g in groups]
    consolidated = [{
        "consolidated_id": generate_consolidated_id(ids),
        "display_index": 1,
        "title": "Combined",
        "description": HOSTILE,
        "importance": "HIGH",
        "reference": "Task 1",
        "type": "modification",
        "underlying_group_indices": [0],
        "underlying_group_ids": ids,
        "model_count": 1,
        "is_singleton": True,
    }]
    path = generate_consolidated_html(consolidated, groups, str(phase_dir),
                                      str(_plan(tmp_path)), ["model-a"])
    return Path(path).read_text(encoding="utf-8")


@pytest.fixture(params=["flat", "pr", "consolidated"])
def page(request, tmp_path) -> str:
    if request.param == "consolidated":
        return consolidated_page(tmp_path)
    return flat_or_pr_page(tmp_path, request.param)[0]


# ---------------------------------------------------------------------------
# What a page loads
# ---------------------------------------------------------------------------

class _Page(HTMLParser):
    """Every element with its attributes, and each script's text. Script text
    is raw data to the tokenizer, so markup inside the data JSON is not a tag."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: List[Tuple[str, Dict[str, str]]] = []
        self.scripts: List[Tuple[Dict[str, str], str]] = []
        self._script: Any = None

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        self.tags.append((tag, a))
        if tag == "script":
            self._script = (a, [])

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append((self._script[0], "".join(self._script[1])))
            self._script = None

    def handle_data(self, data):
        if self._script is not None:
            self._script[1].append(data)


def parse(html: str) -> _Page:
    parsed = _Page()
    parsed.feed(html)
    parsed.close()
    return parsed


#: An exact version in a CDN path: ``pkg@1.2.3/``.
_EXACT_VERSION = re.compile(r"@\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?/")
_INTEGRITY = re.compile(r"^(sha256|sha384|sha512)-[A-Za-z0-9+/]+={0,2}$")
_URL_IMPORT = re.compile(r"""\bimport\b[^;'"]*?['"]((?:https?:)?//[^'"]+)['"]|\bimport\(\s*['"]((?:https?:)?//)""")


def unpinned_external_scripts(html: str) -> List[str]:
    """Each script a page would fetch without an exact version and an integrity
    hash: a ``<script src>``, a ``<link>`` preload of one, or an ES-module
    import from a URL (which no ``integrity`` attribute covers)."""
    found = []
    parsed = parse(html)
    for tag, attrs in parsed.tags:
        url = attrs.get("src") if tag == "script" else (
            attrs.get("href") if tag == "link" and attrs.get("rel") in ("modulepreload", "preload")
            else None)
        if not url:
            continue
        if not (_EXACT_VERSION.search(url) and _INTEGRITY.match(attrs.get("integrity", ""))
                and "crossorigin" in attrs):
            found.append(url)
    for _attrs, text in parsed.scripts:
        found += [m.group(1) or m.group(2) for m in _URL_IMPORT.finditer(text)]
    return found


class TestTheUnpinnedScriptCheck:
    """The check itself, on the shapes the templates used to carry."""

    def test_it_flags_the_old_cdn_tags(self):
        old = ('<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>'
               '<script type="module">import { marked } from '
               '"https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";</script>')
        assert unpinned_external_scripts(old) == [
            "https://cdn.jsdelivr.net/npm/marked/marked.min.js",
            "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js",
        ]

    def test_a_version_without_integrity_is_still_flagged(self):
        tag = '<script src="https://cdn.jsdelivr.net/npm/dompurify@3.4.17/dist/purify.min.js"></script>'
        assert unpinned_external_scripts(tag) != []

    def test_a_pinned_tag_with_integrity_passes(self):
        tag = ('<script src="https://cdn.jsdelivr.net/npm/dompurify@3.4.17/dist/purify.min.js" '
               'integrity="sha384-AAAA" crossorigin="anonymous"></script>')
        assert unpinned_external_scripts(tag) == []


class TestThePagesLoadNoScriptFromTheNetwork:

    def test_no_template_names_an_external_script(self):
        for name in TEMPLATE_NAMES:
            text = (TEMPLATES / name).read_text(encoding="utf-8")
            assert unpinned_external_scripts(text) == [], name
            assert not [a for t, a in parse(text).tags if t == "script" and "src" in a], name
            assert text.count(SLOT_VENDOR_SCRIPTS) == 1, name

    def test_a_generated_page_has_no_unpinned_or_external_script(self, page):
        assert unpinned_external_scripts(page) == []
        parsed = parse(page)
        assert [a for t, a in parsed.tags if t == "script" and "src" in a] == []
        assert [a for t, a in parsed.tags if t == "link"] == []
        assert SLOT_VENDOR_SCRIPTS not in page

    def test_both_libraries_are_inlined_before_the_page_script(self, page):
        scripts = parse(page).scripts
        vendored = [a.get("data-vendored") for a, _ in scripts if "data-vendored" in a]
        assert vendored == [f"{pkg}@{ver}" for _f, pkg, ver, _h in VENDORED_SCRIPTS]
        # The page's own script (the one defining renderMarkdown) comes last,
        # so both globals exist when it starts.
        last_attrs, last_text = scripts[-1]
        assert last_attrs == {} and "function renderMarkdown(" in last_text
        # No wait on a library event: init runs at once, also offline.
        assert "marked-ready" not in page and "waitForDeps" not in page

    def test_a_regenerated_page_carries_them_too(self, tmp_path):
        _html, phase_dir = flat_or_pr_page(tmp_path, "pr")
        path, _matched = regenerate_report_with_human_decisions(phase_dir, {})
        text = path.read_text(encoding="utf-8")
        assert unpinned_external_scripts(text) == []
        assert [a.get("data-vendored") for a, _ in parse(text).scripts if "data-vendored" in a] \
            == [f"{pkg}@{ver}" for _f, pkg, ver, _h in VENDORED_SCRIPTS]


# ---------------------------------------------------------------------------
# The vendored files and their pins
# ---------------------------------------------------------------------------

class TestTheVendoredFiles:

    @pytest.mark.parametrize("entry", VENDORED_SCRIPTS, ids=lambda e: e[1])
    def test_each_file_matches_its_pin(self, entry):
        filename, package, version, sha256 = entry
        raw = (VENDOR / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha256
        banner = raw[:400].decode("utf-8")
        assert {"marked": f"marked v{version}", "dompurify": f"DOMPurify {version}"}[package] in banner
        text = raw.decode("utf-8")
        # Inlined into a <script> and filled after the other slots: nothing in
        # it may end the element or look like a slot or the data placeholder.
        assert not re.search(r"</?script", text, re.I)
        assert "<!--MLLM:" not in text and "REPORT_DATA_PLACEHOLDER" not in text

    def test_each_library_ships_its_licence(self):
        assert "Permission is hereby granted" in (VENDOR / "marked.LICENSE.txt").read_text(
            encoding="utf-8")
        assert "Apache License" in (VENDOR / "dompurify.LICENSE.txt").read_text(encoding="utf-8")
        assert sorted(p.name for p in VENDOR.iterdir()) == sorted(
            [e[0] for e in VENDORED_SCRIPTS] + ["marked.LICENSE.txt", "dompurify.LICENSE.txt"])

    def test_the_source_map_comment_is_dropped(self):
        html = hrg.vendor_scripts_html()
        assert "sourceMappingURL" not in html
        assert html.count("</script>") == len(VENDORED_SCRIPTS)


class TestAFileOffItsPinIsNotInlined:

    @pytest.fixture
    def vendor_copy(self, tmp_path, monkeypatch):
        copy = tmp_path / "vendor"
        shutil.copytree(VENDOR, copy)
        monkeypatch.setattr(hrg, "_VENDOR_DIR", copy)
        monkeypatch.setattr(hrg, "_vendor_scripts_cache", None)
        return copy

    def test_an_edited_library_is_left_out(self, vendor_copy, caplog):
        purify = vendor_copy / "purify.min.js"
        purify.write_bytes(purify.read_bytes() + b"\n;window.pwned = 1;\n")
        html = hrg.vendor_scripts_html()
        assert 'data-vendored="marked@' in html
        assert "dompurify@" not in html and "pwned" not in html
        assert "not inlined" in caplog.text and "pinned" in caplog.text

    def test_a_missing_library_is_left_out(self, vendor_copy, caplog):
        (vendor_copy / "marked.umd.js").unlink()
        html = hrg.vendor_scripts_html()
        assert "marked@" not in html and 'data-vendored="dompurify@' in html
        assert "not inlined" in caplog.text

    def test_a_library_holding_a_script_tag_is_left_out(self, vendor_copy, monkeypatch):
        bad = vendor_copy / "marked.umd.js"
        bad.write_bytes(b"var s = '</script><img src=x onerror=alert(1)>';\n")
        monkeypatch.setattr(hrg, "VENDORED_SCRIPTS", (
            ("marked.umd.js", "marked", "0.0.0", hashlib.sha256(bad.read_bytes()).hexdigest()),))
        assert hrg.vendor_scripts_html() == ""


# ---------------------------------------------------------------------------
# One renderer, in every template
# ---------------------------------------------------------------------------

_BLOCK = re.compile(r"    // BEGIN SAFE MARKDOWN.*?// END SAFE MARKDOWN\n", re.S)


def safe_markdown_block(text: str) -> str:
    blocks = _BLOCK.findall(text)
    assert len(blocks) == 1, "expected one SAFE MARKDOWN block"
    return blocks[0]


class TestOneRendererEverywhere:

    def test_the_block_is_identical_in_the_three_templates(self):
        blocks = {name: safe_markdown_block((TEMPLATES / name).read_text(encoding="utf-8"))
                  for name in TEMPLATE_NAMES}
        assert len(set(blocks.values())) == 1, list(blocks)

    @pytest.mark.parametrize("name", TEMPLATE_NAMES)
    def test_markdown_is_parsed_only_inside_it(self, name):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        outside = _BLOCK.sub("", text)
        assert text.count("function renderMarkdown(") == 1
        assert "function renderMarkdown(" not in outside
        for raw_call in ("marked.parse", "markedParse", "marked(", "DOMPurify"):
            assert raw_call not in outside, (name, raw_call)

    def test_the_block_sanitises_what_marked_returns(self):
        block = safe_markdown_block((TEMPLATES / TEMPLATE_NAMES[0]).read_text(encoding="utf-8"))
        assert "DOMPurify.sanitize(marked.parse(String(text), { breaks: true })," in block
        assert "DOMPurify.isSupported === true" in block
        assert "escapeHtml(text)" in block


class TestHostileTextStaysData:
    """The data path: a hostile description reaches the page only as JSON
    inside the data script, never as markup, and is stored unchanged."""

    def test_no_live_markup_from_the_description(self, page):
        parsed = parse(page)
        for tag, attrs in parsed.tags:
            assert not (tag == "img" and "onerror" in attrs), attrs
            assert not attrs.get("href", "").lower().startswith("javascript:"), attrs
            assert attrs.get("data-cg-id") != "forged", attrs
            assert "alert(" not in "".join(v for k, v in attrs.items() if k.startswith("on")), attrs
        # The text sits in the data script, script-escaped.
        data_scripts = [t for _a, t in parsed.scripts if "reportData" in t and HOSTILE[:30] in t]
        assert data_scripts
        assert "<\\/script>" in data_scripts[0] and "</script>" not in data_scripts[0]

    def test_the_stored_description_is_the_model_text_unchanged(self, tmp_path):
        _html, phase_dir = flat_or_pr_page(tmp_path, "pr")
        stored = json.loads((phase_dir / "report_data.json").read_text(encoding="utf-8"))
        texts = [s.get("description") for g in stored["groups"] for s in g["suggestions"]]
        assert HOSTILE in texts


# ---------------------------------------------------------------------------
# The renderer under Node, with the real vendored marked
# ---------------------------------------------------------------------------

#: The block's escapers are plain string functions: Node needs no ``document``.
_NODE_PRELUDE = r"""
const vm = require('vm');
const fs = require('fs');
"""


def run_renderer(template: str, body: str) -> Any:
    """Load the vendored libraries as classic scripts (globals, as a browser
    does) and the template's SAFE MARKDOWN block (which holds the escapers),
    then run *body*, which prints one JSON value."""
    if NODE is None:
        pytest.skip("node is not on PATH; the static checks above still ran")
    text = (TEMPLATES / template).read_text(encoding="utf-8")
    page_js = safe_markdown_block(text)
    files = {e[1]: str(VENDOR / e[0]) for e in VENDORED_SCRIPTS}
    script = (
        _NODE_PRELUDE
        + f"vm.runInThisContext(fs.readFileSync({json.dumps(files['marked'])}, 'utf8'));\n"
        + f"const PURIFY = {json.dumps(files['dompurify'])};\n"
        + f"vm.runInThisContext({json.dumps(page_js)});\n"
        + f"const HOSTILE = {json.dumps(HOSTILE)};\n"
        + body
    )
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "render.js"
        path.write_text(script, encoding="utf-8")
        done = subprocess.run([NODE, str(path)], capture_output=True, text=True,
                              encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


_FORBIDDEN_TAGS = {"script", "style", "img", "input", "form", "button", "textarea", "select",
                   "option", "iframe", "frame", "object", "embed", "svg", "math", "link", "meta",
                   "base", "template", "video", "audio", "source", "picture", "dialog"}
_FORBIDDEN_ATTRS = {"style", "class", "id", "name", "src", "srcset", "srcdoc", "action",
                    "formaction", "target", "xlink:href", "background", "poster"}


@pytest.mark.parametrize("template", TEMPLATE_NAMES)
class TestTheRendererUnderNode:

    def test_marked_output_goes_through_sanitize_with_the_markdown_profile(self, template):
        out = run_renderer(template, r"""
var calls = [];
globalThis.DOMPurify = { isSupported: true, sanitize: function (dirty, cfg) {
  calls.push([dirty, cfg]); return 'SANITISED'; } };
var result = renderMarkdown(HOSTILE);
console.log(JSON.stringify({ result: result, calls: calls }));
""")
        assert out["result"] == "SANITISED"
        (dirty, cfg), = out["calls"]
        # marked itself passes the hostile markup through: the sanitiser is
        # what stands between it and innerHTML.
        assert "<img src=x onerror=alert(1)>" in dirty
        assert 'href="javascript:alert(1)"' in dirty
        tags = set(cfg["ALLOWED_TAGS"])
        assert {"p", "br", "a", "code", "pre", "table", "thead", "tbody", "tr", "th", "td",
                "ul", "ol", "li", "blockquote", "strong", "em", "h1", "h3"} <= tags
        assert not tags & _FORBIDDEN_TAGS, tags & _FORBIDDEN_TAGS
        attrs = set(cfg["ALLOWED_ATTR"])
        assert "href" in attrs and "align" in attrs
        assert not attrs & _FORBIDDEN_ATTRS, attrs & _FORBIDDEN_ATTRS
        assert not [a for a in attrs if a.startswith("on")]
        assert cfg["ALLOW_DATA_ATTR"] is False and cfg["ALLOW_ARIA_ATTR"] is False
        assert set(cfg) == {"ALLOWED_TAGS", "ALLOWED_ATTR", "ALLOW_DATA_ATTR", "ALLOW_ARIA_ATTR"}

    def test_tables_code_and_line_breaks_reach_the_sanitiser(self, template):
        out = run_renderer(template, r"""
globalThis.DOMPurify = { isSupported: true, sanitize: function (dirty) { return dirty; } };
console.log(JSON.stringify(renderMarkdown(
  'a\nb\n\n| x | y |\n|:-|-:|\n| 1 | 2 |\n\n```py\nx = 1\n```\n')));
""")
        assert "a<br>b" in out
        assert '<th align="left">x</th>' in out and "<td" in out
        assert '<pre><code class="language-py">x = 1' in out

    @pytest.mark.parametrize("setup", [
        "delete globalThis.DOMPurify;",
        "globalThis.DOMPurify = { isSupported: false, sanitize: function (x) { return x; } };",
        "globalThis.DOMPurify = { sanitize: function (x) { return x; } };",
        "globalThis.DOMPurify = { isSupported: true, sanitize: function () { throw new Error('x'); } };",
        # The real vendored DOMPurify without a DOM: a factory with no sanitize.
        "vm.runInThisContext(fs.readFileSync(PURIFY, 'utf8'));",
        "globalThis.DOMPurify = { isSupported: true, sanitize: function (x) { return x; } };"
        " delete globalThis.marked;",
    ], ids=["no-purify", "unsupported", "no-flag", "throws", "real-purify-no-dom", "no-marked"])
    def test_without_a_working_sanitiser_the_text_is_escaped(self, template, setup):
        out = run_renderer(template, setup + r"""
console.log(JSON.stringify(renderMarkdown(HOSTILE)));
""")
        assert out.startswith("<p>") and out.endswith("</p>")
        assert "<img" not in out and "<a " not in out and "<script" not in out
        assert "<input" not in out and "<style" not in out
        assert "&lt;img src=x onerror=alert(1)&gt;" in out
        assert out.count("<br>") == HOSTILE.count("\n")


# ---------------------------------------------------------------------------
# The escapers and badge allow-lists in the block
# ---------------------------------------------------------------------------

#: Every class name a badge helper may return.
_BADGE_CLASSES = {"badge-importance-high", "badge-importance-medium", "badge-importance-low",
                  "badge-validation-valid", "badge-validation-invalid",
                  "badge-validation-needs-human", "badge-validation-failed", "badge-category"}


@pytest.mark.parametrize("template", TEMPLATE_NAMES)
class TestTheBlockEscapers:

    def test_escape_html_escapes_all_five_characters(self, template):
        out = run_renderer(template, r"""
console.log(JSON.stringify([escapeHtml('a&b<c>d"e\'f'), escapeAttr('"\''), escapeHtml(null),
  escapeHtml(undefined), escapeHtml(0), escapeHtml(false)]));
""")
        assert out == ["a&amp;b&lt;c&gt;d&quot;e&#39;f", "&quot;&#39;", "", "", "0", "false"]

    def test_badge_classes_come_from_a_fixed_list(self, template):
        out = run_renderer(template, r"""
var hostile = ['x" onmouseover="alert(1)', '<img src=x onerror=alert(1)>', 'high', 'Valid', 7, {},
  null, undefined, 'valid', 'needs-human-decision', 'validation_failed', 'LOW'];
console.log(JSON.stringify(hostile.map(function (v) {
  return [getImportanceBadgeClass(v), getValidationBadgeClass(v)]; })));
""")
        assert {cls for pair in out for cls in pair} <= _BADGE_CLASSES
        assert out[2][0] == "badge-importance-high" and out[11][0] == "badge-importance-low"
        assert [pair[1] for pair in out[8:11]] == [
            "badge-validation-valid", "badge-validation-needs-human", "badge-validation-failed"]


#: A JS string literal that leaves an HTML attribute value open, then ``+``.
_OPEN_ATTRIBUTE = re.compile(r"""'((?:\\.|[^'\\\n])*)'\s*\+\s*""")
#: What may follow such a literal: an escaper, a badge allow-list, the scope
#: block's own helpers, or a ternary between literals.
_ATTR_SAFE = re.compile(
    r"(?:escapeAttr|escapeJsAttr|scopeAttr|getValidationBadgeClass|getImportanceBadgeClass)\("
    r"|SCOPE_CLASSES\[|\((?:is\w+|hasSuggestions|cg\.isSingleton) \? '")
#: Local names the templates build only from literals (class lists) or numbers.
_ATTR_LITERAL_NAMES = {"cls", "dotClass", "itemClass", "lineClass", "compClass",
                       "unanchoredImpClass", "consensusClass", "skippedClass", "typedClass",
                       "confidenceClass", "confidencePercent", "dClass", "impClass",
                       "validationColorClass", "cardClasses", "groupNum"}


def open_attribute_values(source: str) -> List[Tuple[int, str, bool]]:
    """Each value concatenated into a still-open double-quoted attribute:
    (line, the expression's start, whether it sits inside an inline handler's
    single-quoted JS string). CSS attribute selectors are not markup."""
    sites = []
    for match in _OPEN_ATTRIBUTE.finditer(source):
        literal = match.group(1)
        tail = literal[literal.rfind(">") + 1:]
        if not re.search(r'=\s*"[^"]*$', tail) or re.search(r'\[[\w-]+="$', tail):
            continue
        in_js_string = bool(re.search(r"""\\'[^'"]*$""", tail[tail.rfind('="'):]))
        expr = source[match.end():match.end() + 80].split("\n")[0]
        sites.append((source.count("\n", 0, match.start()) + 1, expr, in_js_string))
    return sites


class TestEveryAttributeValueIsEscaped:
    """The sites the review found (statuses, importances, ids, model names
    concatenated raw into attributes and class names) and any new one fail
    here: each value takes an escaper or a fixed list."""

    @pytest.mark.parametrize("name", TEMPLATE_NAMES)
    def test_no_raw_value_in_an_attribute(self, name):
        source = (TEMPLATES / name).read_text(encoding="utf-8")
        sites = open_attribute_values(source)
        assert len(sites) > 20, "the scan found no attribute sites: the pattern broke"
        raw = [(line, expr) for line, expr, _js in sites
               if not _ATTR_SAFE.match(expr)
               and re.match(r"[\w$]+", expr).group(0) not in _ATTR_LITERAL_NAMES]
        assert raw == [], raw
        handlers = [(line, expr) for line, expr, js in sites
                    if js and not expr.startswith("escapeJsAttr(")]
        assert handlers == [], handlers

    def test_the_scan_catches_the_shapes_the_review_named(self):
        bad = ("html += '<span class=\"badge ' + valBadgeClass + '\">';\n"
               "x = ' data-models=\"' + groupModels.join(',') + '\">';\n"
               "y = ' onchange=\"toggle(\\'' + escapeAttr(id) + '\\')\"';\n")
        sites = open_attribute_values(bad)
        assert [expr.split(" ")[0] for _l, expr, _js in sites] == [
            "valBadgeClass", "groupModels.join(',')", "escapeAttr(id)"]
        assert [js for _l, _e, js in sites] == [False, False, True]

    @pytest.mark.parametrize("name", TEMPLATE_NAMES)
    def test_one_escaper_lives_in_the_block(self, name):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        outside = _BLOCK.sub("", text)
        for fn in ("escapeHtml", "escapeAttr", "escapeJsAttr", "getImportanceBadgeClass",
                   "getValidationBadgeClass"):
            assert text.count(f"function {fn}(") == 1, (name, fn)
            assert f"function {fn}(" not in outside, (name, fn)


# ---------------------------------------------------------------------------
# Model-controlled metadata reaching each page, rendered by a real browser
# ---------------------------------------------------------------------------

#: Fields a reviewer or validator controls outside the markdown bodies. Each
#: tries to leave its attribute (a handler, a forged selection id), open a
#: tag, or break out of an inline handler's JS string; the importance hides
#: its handler in entities that survive uppercasing (every page now reads a
#: value naming no level as MEDIUM, so it never reaches the page).
EVIL_STATUS = ('x" onmouseover="alert(11)" data-cg-id="forged">'
               '<img src=x onerror=alert(12)><input type="checkbox" data-cg-id="forged">')
EVIL_IMPORTANCE = "<img src=x onerror=&#97;&#108;&#101;&#114;&#116;(13)>"
EVIL_MODEL = 'm"><img src=x onerror=alert(14)>'
EVIL_PATH = "src/a\"b'c<img src=x onerror=alert(15)>.py"
EVIL_REFERENCE = 'Task 1" onmouseover="alert(16)" x="<img src=x onerror=alert(17)>'
EVIL_ID = "h');alert(18);//\"><img src=x onerror=alert(19)>"
EVIL_LABEL = "G1<img src=x onerror=alert(20)>"
EVIL_CG_ID = 'cg" data-cg-id="forged'

CHROME = next((p for p in map(shutil.which, ("google-chrome", "google-chrome-stable",
                                             "chromium", "chromium-browser")) if p), None)

#: Runs first in <head>: any handler that does fire leaves its argument on <html>.
_ALERT_TRAP = ("<script>window.alert = function (m) { document.documentElement"
               ".setAttribute('data-alerted', String(m)); };</script>")


def evil_groups(phase: str) -> List[Dict[str, Any]]:
    suggestion = {
        "title": "Evil finding", "desc": "Plain text.", "type": "modification",
        "importance": EVIL_IMPORTANCE, "source_model": EVIL_MODEL,
        "suggestion_hash": EVIL_ID, "display_label": EVIL_LABEL + "S1",
    }
    if phase == "code-review":
        suggestion.update(file=EVIL_PATH, line_range=[3, 4])
    else:
        suggestion["reference"] = EVIL_REFERENCE
    return [{
        "theme": "Evil theme", "category": "bug", "reference": EVIL_REFERENCE,
        "models": [EVIL_MODEL], "importance": EVIL_IMPORTANCE,
        "group_hash": EVIL_ID + "g", "display_label": EVIL_LABEL,
        "validation_status": EVIL_STATUS, "validation_reason": "Because.",
        "validation_confidence": 0.5,
        "suggestions": [suggestion],
    }]


def evil_page(kind: str, tmp_path: Path) -> str:
    if kind == "consolidated":
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        groups = evil_groups("review-plan")
        consolidated = [{
            "consolidated_id": EVIL_CG_ID, "display_index": 1, "title": "Combined",
            "description": "Plain text.", "importance": EVIL_IMPORTANCE,
            "reference": EVIL_REFERENCE, "type": "modification",
            "underlying_group_indices": [0], "underlying_group_ids": [EVIL_ID + "g"],
            "model_count": 1, "is_singleton": True,
        }]
        validation = [{"group_hash": EVIL_ID + "g", "status": EVIL_STATUS,
                       "reason": "Because.", "confidence": 0.5}]
        path = generate_consolidated_html(consolidated, groups, str(phase_dir),
                                          str(_plan(tmp_path)), [EVIL_MODEL],
                                          validation=validation)
        return Path(path).read_text(encoding="utf-8")
    style, phase = kind.split("/")
    phase_dir = tmp_path / f"{phase}-{style}"
    phase_dir.mkdir()
    return generate_html_report(groups=evil_groups(phase), plan_path=_plan(tmp_path),
                                phase_dir=phase_dir, phase_type=phase,
                                models=[EVIL_MODEL], template_style=style)


def rendered_dom(html: str, scratch: Path) -> str:
    """The page's DOM once its script has run, from headless Chrome with no
    network (every host lookup fails)."""
    if CHROME is None:
        pytest.skip("no Chrome or Chromium on PATH; the static checks still ran")
    page = scratch / "page.html"
    page.write_text(html.replace("<head>", "<head>" + _ALERT_TRAP, 1), encoding="utf-8")
    done = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={scratch / 'chrome-profile'}",
         "--host-resolver-rules=MAP * ~NOTFOUND", "--virtual-time-budget=5000",
         "--dump-dom", page.as_uri()],
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


class _Dom(HTMLParser):
    """Each element's attributes, and the text outside scripts and styles."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: List[Tuple[str, Dict[str, str]]] = []
        self.text: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, {k: v or "" for k, v in attrs}))
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.text.append(data)


_JS_TOKEN = re.compile(r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|[A-Za-z_$][\w$]*|\S""", re.S)
_JS_ESCAPE = {"\\": "\\", "'": "'", '"': '"', "n": "\n", "r": "\r"}


def handler_code_names(code: str) -> List[str]:
    """The identifiers an inline handler's code names outside its string
    literals: a value that broke out of its string shows up here."""
    return [t for t in _JS_TOKEN.findall(code) if re.fullmatch(r"[A-Za-z_$][\w$]*", t)]


def js_string_arguments(code: str) -> List[str]:
    out = []
    for token in _JS_TOKEN.findall(code):
        if token[0] in "'\"":
            out.append(re.sub(r"\\u(2028|2029)|\\(.)", lambda m: chr(int(m.group(1), 16))
                              if m.group(1) else _JS_ESCAPE.get(m.group(2), m.group(2)),
                              token[1:-1]))
    return out


PAGE_KINDS = ["flat/review-plan", "flat/code-review", "pr/review-plan", "pr/code-review",
              "consolidated"]


@pytest.mark.parametrize("kind", PAGE_KINDS)
class TestHostileMetadataStaysText:
    """Statuses, importances, model names, ids, labels, file paths and
    references reach the rendered page as text and whole attribute values."""

    def dom(self, kind, tmp_path) -> _Dom:
        parsed = _Dom()
        parsed.feed(rendered_dom(evil_page(kind, tmp_path), tmp_path))
        parsed.close()
        return parsed

    def test_no_live_markup_and_nothing_fires(self, kind, tmp_path):
        parsed = self.dom(kind, tmp_path)
        assert parsed.tags[0][0] == "html" and "data-alerted" not in parsed.tags[0][1]
        handlers = 0
        for tag, attrs in parsed.tags:
            assert not (tag == "img" and "model-icon" not in attrs.get("class", "")), attrs
            assert not {"onmouseover", "onerror", "x"} & set(attrs), (tag, attrs)
            assert attrs.get("data-cg-id") != "forged", (tag, attrs)
            for name, value in attrs.items():
                if name.startswith("on"):
                    handlers += "(" in value
                    assert "alert" not in handler_code_names(value), (tag, name, value)
        if kind != "consolidated":
            assert handlers > 3
        # The page rendered (an empty page passes the checks above), and the
        # hostile values are its text.
        text = "".join(parsed.text)
        # An importance naming no level reads as MEDIUM on every page
        # (utils/importance.py), so the hostile value is replaced, not shown.
        for value in (EVIL_STATUS, "MEDIUM", EVIL_REFERENCE if "review-plan" in kind
                      or kind == "consolidated" else EVIL_PATH):
            assert value in text, (value, text[:2000])
        assert EVIL_IMPORTANCE not in text and EVIL_IMPORTANCE.upper() not in text
        if kind != "consolidated":
            assert EVIL_LABEL in text and EVIL_MODEL in text

    def test_attribute_values_come_back_whole(self, kind, tmp_path):
        parsed = self.dom(kind, tmp_path)
        attrs = [a for _t, a in parsed.tags]
        if kind == "consolidated":
            cards = [a for a in attrs if "group-card" in a.get("class", "").split()]
            assert len(cards) == 1
            assert cards[0]["data-cg-id"] == EVIL_CG_ID
            assert cards[0]["data-reference"] == EVIL_REFERENCE
            assert cards[0]["data-validation"] == EVIL_STATUS
            assert cards[0]["data-importance"] == "MEDIUM"
            return
        assert {a["data-validation"] for a in attrs
                if "data-validation" in a and "data-group-id" in a} == {EVIL_STATUS}
        assert {a["data-group-id"] for a in attrs if "data-group-id" in a} == {EVIL_ID + "g"}
        assert {a["data-suggestion-id"] for a in attrs if "data-suggestion-id" in a} == {EVIL_ID}
        if kind.startswith("flat/"):
            assert [a["data-models"] for a in attrs if "data-models" in a] == [EVIL_MODEL]
        # An inline handler gets the id back whole, as one string argument.
        skip = [a["onchange"] for a in attrs if "Skip" in a.get("onchange", "")
                and "Suggestion" in a["onchange"]]
        assert skip and all(js_string_arguments(code) == [EVIL_ID] for code in skip), skip

    def test_the_selection_controls_are_the_pages_own(self, kind, tmp_path):
        parsed = self.dom(kind, tmp_path)
        boxes = [a for t, a in parsed.tags if t == "input" and a.get("type") == "checkbox"]
        if kind == "consolidated":
            assert [a.get("data-cg-id") for a in boxes if "data-cg-id" in a] == [EVIL_CG_ID]
            selects = [a for t, a in parsed.tags if t == "select" and "data-cg-id" in a]
            assert [a["data-cg-id"] for a in selects] == [EVIL_CG_ID]
        else:
            assert not [a for a in boxes if "data-cg-id" in a]


# ---------------------------------------------------------------------------
# Reviewer identity: the slot names the reviewer, not the reply
# ---------------------------------------------------------------------------

class TestTheSlotNamesTheReviewer:
    """A code-review reply that claims its own ``source_model`` does not get
    to name its reviewer: grouping attributes by it and the reports print it,
    so the fan-out's aggregation stamps the slot's model over it, as
    ``--reaggregate`` already does."""

    def test_a_claimed_source_model_is_overwritten(self, tmp_path):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        import code_review_orchestrator as cro
        from tests.test_code_review_orchestrator import _make_args
        from utils.state_manager import load_groups_payload

        plan_path = tmp_path / "plan.md"
        plan_path.write_text("# plan\n", encoding="utf-8")
        slot = "cursor-agent:auto"
        issue = {"title": "Claimed reviewer", "desc": "A finding.", "type": "bug",
                 "importance": "high", "file": "a.py", "source_model": EVIL_MODEL}
        state = MagicMock()
        state.get = MagicMock(side_effect=lambda k, default=None: {
            "tracked_files": [{"path": "a.py"}], "head_before_implement": "abc123",
            "head_at_start": "abc123", "pre_existing_changes": [],
        }.get(k, default))
        state.state = {"phases_completed": {}}
        state.is_phase_completed = MagicMock(return_value=False)

        with patch.object(cro, "parse_args", return_value=_make_args(plan_path)), \
             patch.object(cro, "get_or_create_state", return_value=state), \
             patch("utils.interactive.resolve_models", return_value=[slot]), \
             patch("utils.interactive.is_model_valid", return_value=True), \
             patch.object(cro, "validate_git_ref", side_effect=lambda r: r or "HEAD~1"), \
             patch.object(cro, "run_all_reviews",
                          new=AsyncMock(return_value={slot: (True, [issue], None)})), \
             patch.object(cro, "generate_review_report", return_value="# report\n"), \
             patch.object(cro, "get_phase_dir", return_value=tmp_path / "phase"):
            asyncio.run(cro.main())

        grouped_path = cro.get_output_paths(plan_path, "code_review_grouped",
                                            phase="code-review")
        groups = load_groups_payload(json.loads(grouped_path.read_text(encoding="utf-8")))
        assert groups
        members = [s for g in groups for s in g.get("suggestions", g.get("issues", []))]
        assert [s["source_model"] for s in members] == [slot]
        assert [g["models"] for g in groups] == [[slot]]
        assert EVIL_MODEL not in grouped_path.read_text(encoding="utf-8")
