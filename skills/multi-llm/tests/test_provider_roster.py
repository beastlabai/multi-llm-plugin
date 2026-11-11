"""Every hand-kept adapter list agrees with the registry.

The registry (``utils.provider_registry._PROVIDERS``) is derived from the modules
in ``utils/providers/``, so it is the one roster. What cannot be derived from it
stays hand-kept: per-adapter config blocks, the report's icon rows, the user
docs and the published adapter counts. Each test here compares one such list
with the registry and fails naming the rows to add or remove and the file they
live in. Tables owned by other test files (models, pinned argv, env profiles,
read-only rows, fixtures) are checked there, through ``tests/roster_check.py``.
The order to fill them in is docs/development/providers.md#adding-a-provider-adapter.
"""

import ast
import importlib
import json
import re
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from tests.roster_check import RECIPE, assert_covers_registry, registered_names
from utils.provider_registry import _PROVIDERS, load_base_config

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_DIR.parents[1]


def _read(relative):
    """A repository file's text, relative to the repo root."""
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _flat(text):
    """Collapse runs of whitespace, so a list wrapped over lines parses as one."""
    return re.sub(r"\s+", " ", text)


def _probed_binaries(name):
    """The executable names the adapter's ``is_available()`` looks up on PATH."""
    asked = []

    def which(binary, *_args, **_kwargs):
        asked.append(binary)
        return None

    with mock.patch("shutil.which", side_effect=which):
        _PROVIDERS[name].is_available()
    return asked


def _base_declaration_order():
    """Registered adapters in providers.yaml's ``providers:`` order.

    That is the order ``--init`` walks to choose ``default_provider``.
    """
    registered = set(registered_names())
    return [
        name for name in (load_base_config().get("providers") or {})
        if name in registered
    ]


def _assert_same_order(listed, where):
    expected = _base_declaration_order()
    assert_covers_registry(listed, where, what="an entry", registry=expected)
    assert listed == expected, (
        f"{where} lists the providers out of order. It must follow the "
        f"`providers:` block of skills/multi-llm/providers.yaml, which --init "
        f"walks to choose default_provider:\n  {', '.join(expected)}\n"
        f"  recipe: {RECIPE}"
    )


class TestConfiguration:
    def test_providers_yaml_has_a_block_per_adapter(self):
        assert_covers_registry(
            load_base_config().get("providers") or {},
            "the providers: block of skills/multi-llm/providers.yaml",
            what="a provider block",
            hint="A block holds default_timeout, max_concurrent and models (the "
            "first is the default). Then regenerate "
            "templates/config/providers.override.yaml (init_config.py, "
            "'Template generator').",
        )

    def test_the_registry_is_not_retyped(self):
        """``_PROVIDERS`` is built from ``PROVIDER_CLASSES``; an adapter class
        imported by name into provider_registry.py is a second roster."""
        source = (SKILL_DIR / "utils" / "provider_registry.py").read_text(
            encoding="utf-8"
        )
        adapter_modules = {
            cls.__module__.rsplit(".", 1)[-1]
            for cls in (type(p) for p in _PROVIDERS.values())
        }
        imported = sorted(
            node.module.rsplit(".", 1)[-1]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.rsplit(".", 1)[-1] in adapter_modules
        )
        assert not imported, (
            f"utils/provider_registry.py imports adapter module(s) {imported} "
            f"by name. Registration is discovery (utils/providers/__init__.py): "
            f"delete the import and the hand-written row."
        )


@pytest.fixture
def extra_adapter_module(tmp_path, monkeypatch):
    """Write a module that discovery walks as if it sat in ``utils/providers/``.

    Discovery reads the package's ``__path__``, so a directory appended to it is
    walked like the package's own and its modules import as
    ``utils.providers.<name>``. The path is restored and each module is dropped
    from ``sys.modules`` afterwards.
    """
    import utils.providers as providers_pkg

    monkeypatch.setattr(
        providers_pkg, "__path__", [*providers_pkg.__path__, str(tmp_path)]
    )
    added = []

    def write(module_name, source):
        (tmp_path / f"{module_name}.py").write_text(
            textwrap.dedent(source), encoding="utf-8"
        )
        added.append(f"{providers_pkg.__name__}.{module_name}")
        importlib.invalidate_caches()

    yield write
    for qualified in added:
        sys.modules.pop(qualified, None)


def _registry_from_discovery():
    """Run discovery and registry construction again, as importing them does."""
    import utils.providers as providers_pkg
    from utils import provider_registry

    return provider_registry._build_registry(
        providers_pkg._discover_provider_classes()
    )


class TestRegistryConstruction:
    """An adapter module registers by existing, and a collision refuses to load."""

    def test_a_new_module_is_registered_beside_the_built_ins(self, extra_adapter_module):
        """The control for the two refusals below: the extra module is walked."""
        extra_adapter_module("zzz_extra", '''
            from utils.providers import codex

            class ZzzExtraProvider(codex.CodexProvider):
                @property
                def name(self):
                    return "zzz-extra"
        ''')

        registry = _registry_from_discovery()

        assert type(registry["zzz-extra"]).__module__ == "utils.providers.zzz_extra"
        assert type(registry["codex"]).__module__ == "utils.providers.codex"
        assert list(registry) == sorted(registry)

    def test_a_reused_class_name_refuses_to_load(self, extra_adapter_module):
        """A second ``CodexProvider`` must not replace the first one unseen.

        The replacement keeps the provider name ``codex``, so the registry's
        name check never sees two claimants: discovery has to refuse it.
        """
        extra_adapter_module("zzz_collision", '''
            from utils.providers import codex

            class CodexProvider(codex.CodexProvider):
                def build_command(self, *args, **kwargs):
                    return ["not-codex"]
        ''')

        with pytest.raises(
            RuntimeError,
            match=r"two adapter classes are named 'CodexProvider': "
            r"utils\.providers\.codex\.CodexProvider and "
            r"utils\.providers\.zzz_collision\.CodexProvider",
        ):
            _registry_from_discovery()

    def test_a_reused_provider_name_refuses_to_load(self, extra_adapter_module):
        extra_adapter_module("zzz_rename", '''
            from utils.providers import codex

            class OtherCodexProvider(codex.CodexProvider):
                pass
        ''')

        with pytest.raises(
            RuntimeError,
            match=r"two adapters claim the provider name 'codex': "
            r"CodexProvider and OtherCodexProvider",
        ):
            _registry_from_discovery()


class TestReportIcons:
    def test_every_adapter_has_a_provider_icon_row(self):
        from utils import html_report_generator as report

        assert_covers_registry(
            report._PROVIDER_ICON_KEYS,
            "_PROVIDER_ICON_KEYS in skills/multi-llm/utils/html_report_generator.py",
            what="an icon row",
            hint="The value is a key of _ICON_DEFS: the CLI's own mark, or a "
            "letter-icon row `(None, letter, colour)` when no first-party SVG "
            "is reachable (docs/attribution.md lists the marks).",
        )

    def test_every_icon_row_names_a_defined_icon(self):
        from utils import html_report_generator as report

        undefined = {
            name: key
            for name, key in report._PROVIDER_ICON_KEYS.items()
            if key not in report._ICON_DEFS
        }
        assert not undefined, (
            f"_PROVIDER_ICON_KEYS names icon keys _ICON_DEFS does not define: "
            f"{undefined}. Add the _ICON_DEFS row in "
            f"skills/multi-llm/utils/html_report_generator.py."
        )


class TestInitConfigInstructions:
    INSTRUCTIONS = "skills/multi-llm/instructions/init-config.md"

    def _binary_map(self):
        text = _flat(_read(self.INSTRUCTIONS))
        found = re.search(r"supported provider CLIs \((.*?)\), copies", text)
        assert found, f"{self.INSTRUCTIONS}: the provider→binary list moved"
        return dict(re.findall(r"([a-z][a-z-]*)→`([^`]+)`", found.group(1)))

    def test_the_binary_list_names_every_adapter(self):
        assert_covers_registry(
            self._binary_map(),
            f"the provider→`binary` list in {self.INSTRUCTIONS}",
            what="an entry",
            hint="Spell it name→`binary`, naming the binary its is_available() looks up.",
        )

    @pytest.mark.parametrize("name", registered_names())
    def test_each_binary_is_the_one_the_adapter_looks_up(self, name):
        listed = self._binary_map().get(name)
        if listed is None:
            pytest.skip("the coverage test above names the missing entry")
        probed = _probed_binaries(name)
        assert listed in probed, (
            f"{self.INSTRUCTIONS} says {name} runs `{listed}`, but its "
            f"is_available() looks up {probed}"
        )

    def test_the_declaration_order_is_the_config_order(self):
        text = _flat(_read(self.INSTRUCTIONS))
        found = re.search(r"declaration order\*\* \(([^)]*)\)", text)
        assert found, f"{self.INSTRUCTIONS}: the declaration-order list moved"
        listed = [n.strip() for n in found.group(1).split(",")]
        _assert_same_order(
            listed, f"the 'base declaration order' list in {self.INSTRUCTIONS}"
        )


class TestUserDocs:
    def test_configuration_doc_lists_the_init_order(self):
        text = _flat(_read("docs/configuration.md"))
        found = re.search(r"in base order \(([^)]*)\)", text)
        assert found, "docs/configuration.md: the 'in base order (...)' list moved"
        listed = [n.strip() for n in found.group(1).split(",")]
        _assert_same_order(listed, "the 'in base order' list in docs/configuration.md")

    def _provider_table(self):
        rows = re.findall(
            r"^\| `([a-z][a-z-]*)` \| `([^`]+)` \|", _read("docs/providers.md"), re.M
        )
        return dict(rows)

    def test_the_provider_table_has_a_row_per_adapter(self):
        assert_covers_registry(
            self._provider_table(),
            "the provider table in docs/providers.md",
            hint="Columns: `key` | `binary` | install link, then any measured "
            "behaviour a user must know.",
        )

    @pytest.mark.parametrize("name", registered_names())
    def test_the_provider_table_names_the_binary_the_adapter_looks_up(self, name):
        listed = self._provider_table().get(name)
        if listed is None:
            pytest.skip("the coverage test above names the missing row")
        probed = _probed_binaries(name)
        assert listed in probed, (
            f"docs/providers.md says {name} runs `{listed}`, but its "
            f"is_available() looks up {probed}"
        )

    def test_the_readme_counts_and_links_every_harness(self):
        text = _flat(_read("README.md"))
        found = re.search(r"\((\d+) supported CLIs: (.*?) itself\)", text)
        assert found, "README.md: the '(N supported CLIs: ...)' list moved"
        stated, links = int(found.group(1)), found.group(2).count("](")
        count = len(_PROVIDERS)
        assert (stated, links) == (count, count), (
            f"README.md's harness list says {stated} supported CLIs and links "
            f"{links}; the registry has {count}. Update the number and add or "
            f"remove the harness's link. recipe: {RECIPE}"
        )

    @pytest.mark.parametrize(
        "manifest", [".claude-plugin/plugin.json", ".claude-plugin/marketplace.json"]
    )
    def test_the_plugin_manifests_count_every_harness(self, manifest):
        stated = [
            int(n)
            for n in re.findall(r"(\d+) coding harnesses", json.dumps(json.loads(_read(manifest))))
        ]
        assert stated, f"{manifest}: the 'N coding harnesses' description moved"
        assert set(stated) == {len(_PROVIDERS)}, (
            f"{manifest} says {stated} coding harnesses; the registry has "
            f"{len(_PROVIDERS)}. Update the description. recipe: {RECIPE}"
        )
