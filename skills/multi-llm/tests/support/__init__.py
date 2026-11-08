"""Fixtures and helpers that more than one test file uses.

A test file imports shared helpers from here, never from another ``test_*.py``
file: a test module changes with its own tests, and one used as a library broke
the files that imported it without either file's tests saying so. Each module
is one world (the check-gate repository, the implement world, the report-check
transcripts, ...), described in its docstring.

Two kinds of module, and ``tests/test_support_package.py`` checks both:

* **Registered.** ``tests/conftest.py`` lists it in ``pytest_plugins``, so its
  fixtures exist for every test: a test names them as parameters and imports
  nothing for them. It defines no autouse fixture, no fixture name that another
  registered module or the conftest defines, and imports no fixture (that would
  register it suite-wide too). A test file that defines a fixture of the same
  name overrides it, as usual.
* **Not registered**, because a fixture name it needs is taken (several worlds
  have a ``repo``), or a fixture is autouse. Every fixture there is
  ``@pytest.fixture(name="repo") def implement_repo(...)``: a test file imports
  the function name (``# noqa: F401 - the repo fixture``), which gives it the
  ``repo`` fixture without binding a module-level ``repo`` that each test
  parameter would shadow (ruff F811). Importing the function always binds its
  fixture in that file: to reuse its body under another name, go through the
  module (``write_orchestrators.orchestrator_repo.__wrapped__``).

``gitrepo`` holds the hermetic git helpers (no developer ``~/.gitconfig``).
"""
