"""Drive the regenerated, copied MCP package over its real stdio protocol.

The repository's Client supplies JSON-RPC transport only; no provider, server,
dispatcher, executor or filesystem operations are mocked. Host roots/posture
are supplied explicitly by this QA client, not by an actual Codex session.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/marc/code/mllm-wt/scope-policy/skills/multi-llm')
from tests.test_codex_mcp_server import Client, turn_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--plugin-root', default='/tmp/scope-policy-world-20261031/installed-plugin')
    parser.add_argument('--out', required=True)
    parser.add_argument('--cancel-after', type=float)
    parser.add_argument('--cancel-on-file')
    parser.add_argument('--args-json', required=True)
    args = parser.parse_args()
    project = Path(args.project)
    out = Path(args.out)
    client = Client(Path(args.plugin_root),
                    dict(os.environ), cwd=project, roots=[project])
    try:
        client.initialize()
        rid = client.start_call(json.loads(args.args_json), token='qa-world',
                                meta=turn_meta([project], 'danger-full-access', model='gpt-6-luna'))
        if args.cancel_after is not None or args.cancel_on_file:
            deadline = time.monotonic() + (args.cancel_after or 120)
            while time.monotonic() < deadline and not client.has_response(rid):
                if args.cancel_on_file and Path(args.cancel_on_file).exists():
                    break
                with client.cond:
                    client.cond.wait(timeout=min(1, max(0, deadline - time.monotonic())))
            client.notify('notifications/cancelled', {'requestId': rid, 'reason': 'QA interruption'})
        answer = client.response(rid, timeout=900)
        out.with_suffix('.json').write_text(json.dumps(answer, indent=2)+'\n')
        result = answer.get('result', {})
        payload = json.loads(result['content'][0]['text']) if result.get('content') else answer
        print(json.dumps({k:payload.get(k) for k in ('ok','exit','state','message','anchor')}, indent=2), flush=True)
    finally:
        client.close()
        out.with_suffix('.rpc.jsonl').write_bytes(b''.join(client.raw_lines))
        out.with_suffix('.stderr.log').write_text(client.stderr)


if __name__ == '__main__':
    main()
