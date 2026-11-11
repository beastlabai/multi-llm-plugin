"""Per-file seconds from pytest JUnit files: the CI legs' timing record.

Both CI test jobs upload a JUnit file (the ``junit-test-<os>`` and
``junit-test-parallel-<os>`` artifacts). A parallel leg's console log cannot be
profiled by file, because its workers interleave their progress, so this reads
the testcase times instead and sums them by test file.

The sums are per-test time, not wall time: under xdist the workers overlap, and
setup a module fixture does once is charged to whichever test triggered it.
Take a leg's wall time from the job itself.

Use (from the repository root)::

    python skills/multi-llm/tests/tools/junit_times.py junit.xml [more.xml ...] [--top N] [--json]

Several files are summed (the shards of one leg, say). ``--json`` prints
``{"<test file>": seconds}`` for every file, for comparing two runs in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Iterable, Tuple


def file_of(case: ET.Element) -> str:
    """The test file a testcase came from, as ``test_name.py``.

    pytest's JUnit ``classname`` is the dotted module path plus any class
    (``tests.test_foo.TestBar``); the file is the first ``test_*`` part. A case
    with no such part (a collection error, say) keeps its classname whole.
    """
    classname = case.get("classname") or ""
    for part in classname.split("."):
        if part.startswith("test_"):
            return f"{part}.py"
    return classname or "<unknown>"


def per_file(paths: Iterable[str]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Summed seconds and testcase counts per test file over every JUnit file in *paths*."""
    seconds: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            name = file_of(case)
            seconds[name] += float(case.get("time") or 0.0)
            counts[name] += 1
    return dict(seconds), dict(counts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("junit", nargs="+", help="JUnit XML files written by pytest --junitxml")
    parser.add_argument("--top", type=int, default=30, help="files to list (default 30)")
    parser.add_argument("--json", action="store_true", help="print every file's seconds as JSON")
    args = parser.parse_args(argv)

    seconds, counts = per_file(args.junit)
    if args.json:
        json.dump({name: round(value, 3) for name, value in sorted(seconds.items())},
                  sys.stdout, indent=1)
        print()
        return 0
    total = sum(seconds.values())
    print(f"{len(seconds)} files, {sum(counts.values())} tests, {total:.0f} s summed")
    for name in sorted(seconds, key=lambda key: -seconds[key])[:args.top]:
        share = seconds[name] / total * 100 if total else 0.0
        print(f"{seconds[name]:9.1f} s {share:5.1f}% {counts[name]:6d} tests  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
