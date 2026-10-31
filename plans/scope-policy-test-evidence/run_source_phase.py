"""Exercise source orchestrators directly, the repository's supported test seam.

This deliberately does not invoke the installed plugin or its dispatcher: the
staged source under test lives in a different worktree from that installation.
"""
import asyncio
import importlib
import json
import sys

SOURCE = "/home/marc/code/mllm-wt/scope-policy/skills/multi-llm"
sys.path.insert(0, SOURCE)

if __name__ == "__main__":
    module = importlib.import_module(sys.argv[1])
    try:
        result = module.main(sys.argv[2:])
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        code = result if isinstance(result, int) else 0
    except SystemExit as exc:
        code = exc.code or 0
    print("QA_SOURCE_PHASE_EXIT: " + json.dumps({"exit": code}))
    sys.exit(code)
