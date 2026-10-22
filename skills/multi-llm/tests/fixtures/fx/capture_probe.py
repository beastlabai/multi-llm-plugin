"""Credential-free real-binary tests of the production adapter in Docker."""
import concurrent.futures
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, '/opt/mllm/skill')
from utils.providers import fx
from utils.providers.base import build_child_env

requests = []
counts = {}
lock = threading.Lock()


class Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        model = body['model']
        reviewer = any(t.get('function', {}).get('name') == 'permission_decision' for t in body.get('tools', []))
        with lock:
            requests.append({'model': model, 'reviewer': reviewer, 'path': self.path, 'body': body})
            turn = counts.get(model, 0)
            if not reviewer: counts[model] = turn + 1
        if model == 'rate-limit':
            self.send_response(429); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(b'{"error":{"message":"Too many requests","code":429}}')
            return
        if model == 'auth':
            self.send_response(401); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid API key","code":401}}')
            return
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        if model == 'empty-signal':
            time.sleep(6)
            return
        calls = []
        content = '[]'
        if reviewer:
            decision = 'caution' if model == 'held-shell' else 'clear'
            arguments = '{' if model == 'malformed-review' else json.dumps({'decision': decision, 'rationale': 'local probe'})
            calls = [{'index': 0, 'id': 'review', 'type': 'function', 'function': {'name': 'permission_decision', 'arguments': arguments}}]
        elif model in ('shell-a', 'shell-b', 'held-shell', 'malformed-review') and turn == 0:
            calls = [{'index': 0, 'id': 'call1', 'type': 'function', 'function': {'name': 'shell', 'arguments': json.dumps({'request': {'action': 'run', 'command': 'pwd; command -v fx-canary; printf FX_SHELL_OK'}})}}]
            content = 'Checking the shell.\n'
        elif model == 'multi-assistant' and turn == 0:
            calls = [{'index': 0, 'id': 'call1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': 'readme'})}}]
            content = 'Reading the file first.\n'
        elif model == 'empty':
            content = ''
        delta = {'content': content}
        if calls: delta['tool_calls'] = calls
        chunks = [
            {'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
            {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if calls else 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}},
        ]
        try:
            for chunk in chunks:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass


server = http.server.ThreadingHTTPServer(('127.0.0.1', 18882), Stub)
threading.Thread(target=server.serve_forever, daemon=True).start()
original = fx.render_settings
fx.render_settings = lambda: original('http://127.0.0.1:18882/s/adapter/v1')
workspace = Path('/tmp/fx-adapter-work'); workspace.mkdir()
(workspace / 'readme').write_text('fixture contents\n')
canary = Path('/tmp/fx-canary-bin'); canary.mkdir()
(canary / 'fx-canary').write_text('#!/bin/sh\nexit 0\n'); (canary / 'fx-canary').chmod(0o755)
os.environ.update(PATH='/tmp/fx:' + str(canary) + ':' + os.environ['PATH'], OPENROUTER_API_KEY='dummy-only', HOME='/tmp')
provider = fx.FxProvider()


def run(model):
    start = time.monotonic()
    with provider.hermetic_dispatch('generate', model):
        env = build_child_env(provider, provider_env=provider.resolve_env(model))
        home = env['HOME']
        cmd, stdin = provider.build_command_for_transport('Return JSON [] after the requested action.', model, 'stdin')
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=workspace)
        timed_out = False
        try: stdout, stderr = process.communicate(stdin, timeout=2 if model in ('rate-limit', 'empty-signal') else 12)
        except subprocess.TimeoutExpired:
            timed_out = True; process.terminate(); stdout, stderr = process.communicate(timeout=10)
        with lock:
            records = [r for r in requests if r['model'] == model]
        return {'name': model, 'stdout': stdout, 'stderr': stderr, 'returncode': process.returncode, 'timeout': timed_out, 'wall_s': round(time.monotonic()-start, 3), 'requests': len(records), 'reviewer_requests': sum(r['reviewer'] for r in records), 'home': home, 'parsed': provider.parse_output(stdout, stderr), 'request_models': [r['body']['model'] for r in records], 'tool_results': [m.get('content') for r in records for m in r['body'].get('messages', []) if m.get('role') == 'tool']}


with concurrent.futures.ThreadPoolExecutor(2) as pool:
    results = list(pool.map(run, ['shell-a', 'shell-b']))
assert results[0]['home'] == results[1]['home'], 'overlapping slots did not share HOME'
for name in ('success', 'multi-assistant', 'empty', 'auth', 'held-shell', 'malformed-review', 'rate-limit', 'empty-signal'):
    results.append(run(name))
assert fx._CONFIG_HOME is None
print(json.dumps(results))
