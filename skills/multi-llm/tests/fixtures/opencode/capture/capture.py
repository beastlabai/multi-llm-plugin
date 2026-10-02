#!/usr/bin/env python3
"""opencode permission capture rig (plan §4.1, §4.2 step 1, T4.0, T4.6-T4.8).

Drives the real ``opencode run`` offline against ``stub.py`` (a localhost
chat-completions provider) from a scratch HOME/XDG under a cleared environment,
stdin ``/dev/null``, ``OPENCODE_DISABLE_MODELS_FETCH=1``. Zero model calls,
zero spend, nothing read from or written to the user's own opencode config or
data, nothing under ``/etc`` written (the managed-config layer is reached
through opencode's own ``OPENCODE_TEST_MANAGED_CONFIG_DIR`` override).

usage: capture.py [--bin PATH] [--scratch DIR] [--keep] [CASE ...]

Writes, per case, into ``<this dir>/<opencode version>/``:
  CASE.jsonl          stdout of ``opencode run --format json``
  CASE.stderr.txt     stderr (``--print-logs --log-level ERROR``)
  CASE.exit.txt       ``exit=N`` (or ``exit=KILLED`` past the deadline)
  CASE.requests.jsonl what reached the stub (one line per HTTP request)
  CASE.config.json    everything the case set: argv, env, the global
                      opencode.json, OPENCODE_CONFIG_CONTENT, managed file,
                      org config
  CASE.debug_config.json  ``opencode debug config`` under the same env
Every absolute scratch path is rewritten to ``/rig``, the stub's port to
``PORT``, the invoking user's home to ``/home/user`` and the host name to
``rig-host``; nothing else is edited. ``summary.tsv`` (one row per case) is
regenerated from the files, so it can be re-derived without re-running.
"""
import argparse
import copy
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEADLINE_S = 90

#: The shipped §4.2 step 1 block (T4.1 / T4.7), verbatim.
BLOCK = {
    "external_directory": {"*": "deny"},
    "read": {"*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"},
    "doom_loop": "deny",
}
CONT = {"experimental": {"continue_loop_on_deny": True}}
#: A later layer that tries to re-open every external read for one agent.
HOSTILE_AGENT = {"agent": {"plan": {"permission": {"external_directory": {"*": "allow"}}}}}


def agent_block(agent, ext=None, cont=False):
    perm = copy.deepcopy(BLOCK)
    if ext is not None:
        perm["external_directory"] = ext
    out = {"agent": {agent: {"permission": perm}}}
    if cont:
        out.update(copy.deepcopy(CONT))
    return out


def read(target):
    return {"tool": "read", "args": {"filePath": target}}


def bash(target):
    return {"tool": "bash", "args": {"command": "cat " + target, "description": "show file"}}


OUT = "{R}/outside/f.txt"  # an out-of-cwd file that is not a host file
#: Any egress a non-stub case might attempt dies on a closed local port; the
#: stub itself is reached directly.
NO_EGRESS = {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
             "NO_PROXY": "127.0.0.1,localhost"}
TINY_PLAN = "# Tiny plan\n\nAdd a `--verbose` flag to the CLI and log each step.\n"
REVIEW_OUT = "{R}/work/plans/tiny/review-plan/opencode_stub_m.json"
REVIEW_JSON = json.dumps([{
    "title": "Missing problem statement", "importance": "medium",
    "description": "The plan names the flag but not the problem it solves.",
    "suggestion": "State which user need --verbose answers."}], indent=2)
#: Plan text that talks about rate limits: in a tool output it must never decide
#: a verdict (run b38de8bb was recorded `rate_limited` on exactly this).
RATE_LIMIT_PLAN = (
    "# Search plan\n\n"
    "The public search endpoint enforces a rate limit of 30 requests per minute per client.\n"
    "Each hit is recorded with rate_limit_hit(client_id) in the history table.\n"
    "A client over the limit gets HTTP 429 with a Retry-After header.\n")
SHARED_USER = {"permission": {"external_directory": {
    "{R}/shared/*": "allow", "{R}/shared/private/*": "deny"}}}
SHARED_EXT = {"*": "deny", "{R}/shared/*": "allow", "{R}/shared/private/*": "deny"}
#: The adapter's trailing permission key (opencode.TRAILING_EXTERNAL_KEY).
TRAILING_KEY = "external_directory*"


def shared_agent(agent):
    """SHARED_USER's rules in a config file's own agent layer."""
    return {"agent": {agent: copy.deepcopy(SHARED_USER)}}


def trailing_block(agent, ext):
    """agent_block with the external list repeated under the trailing key, last."""
    out = agent_block(agent, ext=copy.deepcopy(ext))
    out["agent"][agent]["permission"][TRAILING_KEY] = copy.deepcopy(ext)
    return out


def _cases():
    c = {}

    def add(name, agent, call, content=None, user=None, pure=True, env=None,
            managed=None, org=None, note="", **stream):
        c[name] = dict(agent=agent, call=call, content=content, user=user or {},
                       pure=pure, env=env or {}, managed=managed, org=org, note=note,
                       model=stream.pop("model", "stub/m"), prompt=stream.pop("prompt", None),
                       files=stream.pop("files", {}), script=stream)

    # §4.1's table: `--agent plan --pure`, config in OPENCODE_CONFIG_CONTENT.
    add("s41_ask_read", "plan", read("/etc/hostname"), note="default ask: auto-reject ends the turn (b38de8bb signature)")
    add("s41_ask_bash", "plan", bash("/etc/hostname"), note="same, via bash")
    add("s41_deny_read", "plan", read("/etc/hostname"), {"permission": {"external_directory": "deny"}}, note="deny returns to the model")
    add("s41_deny_bash", "plan", bash("/etc/hostname"), {"permission": {"external_directory": "deny"}}, note="same, via bash")
    add("s41_cont_read", "plan", read("/etc/hostname"), CONT, note="continue_loop_on_deny: auto-rejected, turn continues")
    add("s41_denyallow_read", "plan", read("/etc/hostname"),
        {"permission": {"external_directory": {"*": "deny", "/etc/*": "allow"}}}, note="directory-glob allow after deny: read completes")
    add("s41_denyallow_file_read", "plan", read("/etc/hostname"),
        {"permission": {"external_directory": {"*": "deny", "/etc/hostname": "allow"}}}, note="a file path is not a directory glob: no match")

    for a in ("plan", "build"):
        # T4.0: the T4.1 content with and without the experimental key.
        add(f"t40_none_{a}", a, read(OUT), note="no content: baseline ask")
        add(f"t40_cont_{a}", a, read(OUT), CONT, note="experimental key alone")
        add(f"t40_block_{a}", a, read(OUT), agent_block(a), note="the T4.1 block under agent.<a>.permission")
        add(f"t40_blockcont_{a}", a, read(OUT), agent_block(a, cont=True), note="block + experimental key")
        add(f"t40_env_none_{a}", a, read("{R}/work/.env"), note=".env inside cwd, no content")
        add(f"t40_env_block_{a}", a, read("{R}/work/.env"), agent_block(a), note=".env inside cwd, block")
        # §4.2 step 1: the pinned case, plus the top-level control from the pre-flight.
        add(f"q2_ustar_agentlayer_{a}", a, read("/etc/hostname"), agent_block(a),
            {"permission": {"external_directory": "ask", "*": "allow"}}, pure=False,
            note="user top-level ask+*:allow vs our agent-layer deny (pinned)")
        add(f"q2_ustar_toplevel_{a}", a, read("/etc/hostname"), {"permission": copy.deepcopy(BLOCK)},
            {"permission": {"external_directory": "ask", "*": "allow"}}, pure=False,
            note="control: the same block top-level loses to the user")
        # Built-in external_directory allows under our "*": "deny".
        add(f"builtin_tmp_ctrl_{a}", a, read("{TMP}/f.txt"), note="control: built-in tmp allow, no content")
        add(f"builtin_tmp_{a}", a, read("{TMP}/f.txt"), agent_block(a), note="tmp glob under the block")
        add(f"builtin_plansdir_ctrl_{a}", a, read("{DATA}/plans/f.md"), note="control: built-in <data>/plans allow")
        add(f"builtin_plansdir_{a}", a, read("{DATA}/plans/f.md"), agent_block(a), note="<data>/plans under the block")
        add(f"builtin_skilldir_ctrl_{a}", a, read("{R}/home/.claude/skills/rigskill/SKILL.md"),
            note="control: a discovered skill dir's built-in allow, no content")
        add(f"builtin_skilldir_{a}", a, read("{R}/home/.claude/skills/rigskill/SKILL.md"), agent_block(a),
            note="a discovered skill dir under the block")
        add(f"builtin_toolout_{a}", a, read("{DATA}/tool-output/f.txt"), agent_block(a), note="<data>/tool-output under the block")
        # T4.7 precedence: a user allow with a deny exception, re-listed in the agent layer.
        add(f"t47_shared_{a}", a, read("{R}/shared/f.txt"), agent_block(a, ext=SHARED_EXT), SHARED_USER, pure=False,
            note="re-listed user allow: read completes")
        add(f"t47_private_{a}", a, read("{R}/shared/private/f.txt"), agent_block(a, ext=SHARED_EXT), SHARED_USER, pure=False,
            note="re-listed user deny exception: returns to the model")
        add(f"t47_shared_toplevel_{a}", a, read("{R}/shared/f.txt"), agent_block(a), SHARED_USER, pure=False,
            note="control: user allow NOT re-listed is shadowed by our deny")
        # A user's own agent-layer rule for the same agent.
        add(f"useragent_star_{a}", a, read(OUT), agent_block(a),
            {"agent": {a: {"permission": {"external_directory": {"*": "allow"}}}}}, pure=False,
            note="user agent.<a> '*':allow (same key) vs our content")
        add(f"useragent_dir_{a}", a, read("{R}/outside/f.txt"), agent_block(a),
            {"agent": {a: {"permission": {"external_directory": {"{R}/outside/*": "allow"}}}}}, pure=False,
            note="user agent.<a> dir allow (earlier key) vs our '*':deny")
        # T4.7 over a config FILE's own agent layer: the file's keys keep their
        # positions in the merge, so the adapter repeats the list under the
        # trailing "external_directory*" key (merge_permission_content).
        add(f"useragent_relist_ctrl_{a}", a, read("{R}/shared/f.txt"), agent_block(a, ext=SHARED_EXT),
            shared_agent(a), pure=False,
            note="control: file agent.<a> dir allow stays ahead of our re-listed '*':deny")
        add(f"useragent_relist_{a}", a, read("{R}/shared/f.txt"), trailing_block(a, SHARED_EXT),
            shared_agent(a), pure=False, note="the same, list repeated under the trailing key: read completes")
        add(f"useragent_relist_private_{a}", a, read("{R}/shared/private/f.txt"), trailing_block(a, SHARED_EXT),
            shared_agent(a), pure=False, note="the deny exception under the trailing key: returns to the model")
        add(f"useragent_ustar_ctrl_{a}", a, read(OUT), agent_block(a),
            {"agent": {a: {"permission": {"external_directory": "ask", "*": "allow"}}}}, pure=False,
            note="control: file agent.<a> ask + later '*':allow beats our deny (silent allow)")
        add(f"useragent_ustar_{a}", a, read(OUT), trailing_block(a, BLOCK["external_directory"]),
            {"agent": {a: {"permission": {"external_directory": "ask", "*": "allow"}}}}, pure=False,
            note="the same, list repeated under the trailing key: deny returned")

    # T4.8's rig case: agent-write shape (no --pure, user config kept).
    add("write_outofcwd_build", "build", read(OUT), agent_block("build"), pure=False,
        note="agent-write: out-of-cwd read returns to the model, finish stop")
    # T4.7's merge over an INHERITED content whose own agent layer carries a
    # permission-level "*": the adapter re-appends our rows after the user's
    # keys (merge_permission_content); the control keeps the user's "*" last.
    add("t47_inherited_star_build", "build", read(OUT),
        {"agent": {"build": {"permission": dict([("*", "allow")] + list(copy.deepcopy(BLOCK).items()))}}},
        pure=False, note="inherited agent.build '*':allow, our rows appended after it")
    add("t47_inherited_star_ctrl_build", "build", read(OUT),
        {"agent": {"build": {"permission": dict(list(copy.deepcopy(BLOCK).items()) + [("*", "allow")])}}},
        pure=False, note="control: the user's '*':allow after our rows")

    # Layers after OPENCODE_CONFIG_CONTENT, hermetic shape (--pure, XDG redirect,
    # OPENCODE_DISABLE_PROJECT_CONFIG=1 — the rig always sets the latter).
    add("managed_agent_plan", "plan", read(OUT), agent_block("plan"),
        env={"OPENCODE_TEST_MANAGED_CONFIG_DIR": "{R}/managed"}, managed=HOSTILE_AGENT,
        note="managed opencode.json agent.plan '*':allow after our content")
    add("managed_toplevel_plan", "plan", read(OUT), agent_block("plan"),
        env={"OPENCODE_TEST_MANAGED_CONFIG_DIR": "{R}/managed"},
        managed={"permission": {"external_directory": {"*": "allow"}}},
        note="managed top-level '*':allow vs our agent layer")
    add("envperm_plan", "plan", read(OUT), agent_block("plan"),
        env={"OPENCODE_PERMISSION": json.dumps({"external_directory": {"*": "allow"}})},
        note="OPENCODE_PERMISSION top-level '*':allow vs our agent layer")
    add("org_agent_plan", "plan", read(OUT), agent_block("plan"), org=HOSTILE_AGENT,
        note="active-org account config (from the data-dir DB) agent.plan '*':allow")
    add("org_agent_dbmem_plan", "plan", read(OUT), agent_block("plan"), org=HOSTILE_AGENT,
        env={"OPENCODE_DB": ":memory:"}, note="same account DB, OPENCODE_DB=:memory:")

    # The parser's stream fixtures (the adapter's `verified_version` grades on
    # these). Multi-step turns scripted through stub.py's `turns`; usage per
    # step is (input, output, cached) as opencode reports it in `step_finish`.
    add("run_review", "build", None, pure=True, files={"plans/tiny.md": TINY_PLAN},
        prompt="Review the plan at {R}/work/plans/tiny.md and write the JSON array "
               "of findings to " + REVIEW_OUT + ".",
        turns=[
            {"calls": [read("{R}/work/plans/tiny.md")],
             "usage": {"input": 9324, "output": 92}},
            {"text": "Now let me explore the codebase to understand what context might be relevant.",
             "calls": [read("{R}/work"),
                       {"tool": "bash", "args": {
                           "command": "ls -la {R}/work/plans/tiny/ 2>/dev/null || echo \"Directory does not exist\"",
                           "description": "List the review directory"}}],
             "usage": {"input": 346, "output": 245, "cached": 9280}},
            {"calls": [{"tool": "write", "args": {"filePath": REVIEW_OUT, "content": REVIEW_JSON}}],
             "usage": {"input": 461, "output": 1329, "cached": 9600}},
        ],
        final="JSON written to " + REVIEW_OUT, final_usage={"input": 1358, "output": 98, "cached": 10048},
        note="a healthy review turn: four steps, four tool calls, two text parts, finish stop")
    add("denied_turn", "plan", None, pure=True, files={"docs/plan.md": RATE_LIMIT_PLAN},
        prompt="Validate the plan's claims against the repository.",
        turns=[
            {"text": "I'll read the plan sections and check the rate limit claims.",
             "calls": [read("{R}/work/docs/plan.md")]},
            {"text": "I'll read the host name the plan's deployment section names.",
             "calls": [bash("/etc/hostname")]},
        ],
        note="b38de8bb's shape: plan text about rate limits in a completed read, then an "
             "auto-rejected bash ends the turn")
    add("auth_error", "plan", None, pure=True, model="google/gemini-2.5-flash",
        prompt="Reply with the single word OK.", user={"provider": {"google": {}}},
        env=NO_EGRESS, note="a declared built-in provider with no key: ProviderAuthError, no request")
    add("invalid_model", "plan", None, pure=True, model="opencode/sonnet",
        prompt="Reply with the single word OK.", env=NO_EGRESS,
        note="a model the catalog does not carry: the UnknownError wrapper, cause on stderr")

    # Controls: what a schema rejection looks like on each build.
    add("t40_badperm_plan", "plan", read(OUT), {"agent": {"plan": {"permission": {"external_directory": "maybe"}}}},
        note="control: invalid permission action")
    add("t40_badexp_plan", "plan", read(OUT), {"experimental": {"no_such_key": True}},
        note="control: unknown experimental key")
    add("t40_badpermname_plan", "plan", read(OUT), {"agent": {"plan": {"permission": {"no_such_permission": "deny"}}}},
        note="control: unknown permission name")
    return c


CASES = _cases()


def _sub(obj, m):
    if isinstance(obj, str):
        for k, v in m.items():
            obj = obj.replace("{" + k + "}", v)
        return obj
    if isinstance(obj, list):
        return [_sub(x, m) for x in obj]
    if isinstance(obj, dict):
        return {_sub(k, m): _sub(v, m) for k, v in obj.items()}
    return obj


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _hostname():
    try:
        return open("/etc/hostname").read().strip()
    except OSError:
        return socket.gethostname()


class Redactor:
    def __init__(self, root, port):
        self.pairs = [(os.path.realpath(root), "/rig"), (root, "/rig")]
        home = os.path.expanduser("~")
        if home and home != "/":
            self.pairs.append((home, "/home/user"))
        self.port = str(port)
        self.host = _hostname()

    def __call__(self, text):
        for a, b in self.pairs:
            text = text.replace(a, b)
        text = text.replace("127.0.0.1:" + self.port, "127.0.0.1:PORT")
        if self.host:
            text = re.sub(r"\b%s\b" % re.escape(self.host), "rig-host", text)
        return text


def _seed_org(binary, env, cwd, port):
    """Create the data-dir DB, then insert an active account + org whose URL is the stub."""
    subprocess.run([binary, "session", "list"], env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    dbs = [os.path.join(dp, f) for dp, _, fs in os.walk(env["XDG_DATA_HOME"]) for f in fs if f.endswith(".db")]
    if not dbs:
        raise SystemExit("org seed: no opencode DB was created")
    now = int(time.time() * 1000)
    con = sqlite3.connect(dbs[0])
    con.execute("insert into account(id,email,url,access_token,refresh_token,token_expiry,time_created,time_updated)"
                " values(?,?,?,?,?,?,?,?)",
                ("acc_rig", "rig@example.invalid", "http://127.0.0.1:%d" % port, "rig-access", "rig-refresh",
                 now + 10 * 365 * 86400 * 1000, now, now))
    con.execute("insert into account_state(id,active_account_id,active_org_id) values(1,'acc_rig','org_rig')")
    con.commit()
    con.close()


def run_case(binary, version, name, spec, scratch, outdir):
    R = os.path.join(scratch, version, name)
    shutil.rmtree(R, ignore_errors=True)
    for d in ("home/.claude/skills/rigskill", "cfg/opencode", "data", "cache", "state", "tmp", "work", "outside",
              "shared/private", "managed"):
        os.makedirs(os.path.join(R, d), exist_ok=True)
    DATA = os.path.join(R, "data", "opencode")
    TMP = os.path.join(R, "tmp", "opencode")
    for d in (os.path.join(DATA, "plans"), os.path.join(DATA, "tool-output"), TMP):
        os.makedirs(d, exist_ok=True)
    for p, body in ((os.path.join(R, "outside/f.txt"), "outside file\n"),
                    (os.path.join(R, "shared/f.txt"), "shared file\n"),
                    (os.path.join(R, "shared/private/f.txt"), "private file\n"),
                    (os.path.join(TMP, "f.txt"), "tmp file\n"),
                    (os.path.join(DATA, "plans/f.md"), "# plan file\n"),
                    (os.path.join(DATA, "tool-output/f.txt"), "tool output file\n"),
                    (os.path.join(R, "work/.env"), "SECRET=rig-not-a-secret\n"),
                    (os.path.join(R, "home/.claude/skills/rigskill/SKILL.md"),
                     "---\nname: rigskill\ndescription: capture-rig skill\n---\nrig skill\n"),
                    (os.path.join(R, "work/README.md"), "rig\n")):
        with open(p, "w") as f:
            f.write(body)
    port = _free_port()
    m = {"R": R, "DATA": DATA, "TMP": TMP, "PORT": str(port)}
    for rel, body in spec["files"].items():
        path = os.path.join(R, "work", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(_sub(body, m))
    subprocess.run(["git", "init", "-q", "."], cwd=os.path.join(R, "work"), check=True)

    call = _sub(spec["call"], m)
    content = _sub(spec["content"], m)
    user = {"$schema": "https://opencode.ai/config.json",
            "provider": {"stub": {"npm": "@ai-sdk/openai-compatible", "name": "stub",
                                  "options": {"baseURL": "http://127.0.0.1:%d/v1" % port, "apiKey": "x"},
                                  "models": {"m": {"name": "m", "tool_call": True}}}},
            "model": "stub/m", "small_model": "stub/m"}
    for key, value in _sub(spec["user"], m).items():
        if key == "provider":
            user["provider"].update(value)
        else:
            user[key] = value
    with open(os.path.join(R, "cfg/opencode/opencode.json"), "w") as f:
        json.dump(user, f, indent=1)
    managed = _sub(spec["managed"], m)
    if managed is not None:
        with open(os.path.join(R, "managed/opencode.json"), "w") as f:
            json.dump(managed, f, indent=1)
    stub_spec = dict(call or {})
    stub_spec.update(_sub(spec["script"], m))
    if spec["org"] is not None:
        stub_spec["org_config"] = _sub(spec["org"], m)
    with open(os.path.join(R, "stub.json"), "w") as f:
        json.dump(stub_spec, f)

    env = {"PATH": "/usr/bin:/bin", "HOME": R + "/home", "XDG_CONFIG_HOME": R + "/cfg",
           "XDG_DATA_HOME": R + "/data", "XDG_CACHE_HOME": R + "/cache", "XDG_STATE_HOME": R + "/state",
           "TMPDIR": R + "/tmp", "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
           "OPENCODE_DISABLE_PROJECT_CONFIG": "1"}
    env.update(_sub(spec["env"], m))
    if content is not None:
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(content)
    cwd = os.path.join(R, "work")
    reqlog = os.path.join(R, "requests.jsonl")
    open(reqlog, "w").close()
    stub = subprocess.Popen([sys.executable, os.path.join(HERE, "stub.py"), str(port),
                             os.path.join(R, "stub.json"), reqlog])
    try:
        time.sleep(0.5)
        if spec["org"] is not None:
            _seed_org(binary, {k: v for k, v in env.items() if k != "OPENCODE_DB"}, cwd, port)
        if spec["prompt"] is not None:
            prompt = _sub(spec["prompt"], m)
        else:
            prompt = "please read " + (call["args"].get("filePath") or call["args"]["command"])
        pure = ["--pure"] if spec["pure"] else []
        argv = [binary, "run"] + pure + ["--agent", spec["agent"], "--format", "json", "--print-logs",
                                         "--log-level", "ERROR", "--model", spec["model"], prompt]
        with open(os.path.join(R, "stdout.jsonl"), "w") as so, open(os.path.join(R, "stderr.txt"), "w") as se:
            p = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=so, stderr=se)
            try:
                rc = "exit=%d" % p.wait(timeout=DEADLINE_S)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                rc = "exit=KILLED"
        dbg = subprocess.run([binary] + pure + ["debug", "config"], env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=60)
    finally:
        stub.kill()
        stub.wait()

    red = Redactor(R, port)
    os.makedirs(outdir, exist_ok=True)

    written = []

    def put(suffix, text):
        path = os.path.join(outdir, name + suffix)
        with open(path, "w") as f:
            f.write(red(text))
        written.append(path)

    put(".jsonl", open(os.path.join(R, "stdout.jsonl")).read())
    put(".stderr.txt", open(os.path.join(R, "stderr.txt")).read())
    put(".exit.txt", rc + "\n")
    put(".requests.jsonl", open(reqlog).read())
    put(".debug_config.json", dbg.stdout if dbg.returncode == 0 else
        json.dumps({"exit": dbg.returncode, "stderr": dbg.stderr[-2000:]}, indent=1) + "\n")
    shown_env = {k: v for k, v in env.items() if k not in ("PATH",)}
    put(".config.json", json.dumps({
        "opencode_version": version, "note": spec["note"], "argv": ["opencode"] + argv[1:],
        "env": shown_env, "OPENCODE_CONFIG_CONTENT": content, "global_opencode_json": user,
        "managed_opencode_json": managed, "org_config": stub_spec.get("org_config"),
        "stub_tool_call": call, "stub_script": _sub(spec["script"], m) or None}, indent=1) + "\n")
    leaks = [os.path.expanduser("~"), os.path.basename(os.path.expanduser("~")), scratch, red.host]
    for path in written:
        text = open(path).read()
        for leak in leaks:
            if leak and len(leak) > 2 and leak in text:
                raise SystemExit("redaction failed: %r still in %s" % (leak, path))
    return rc


def summarize(outdir):
    rows = []
    for fn in sorted(os.listdir(outdir)):
        if not fn.endswith(".exit.txt"):
            continue
        name = fn[: -len(".exit.txt")]
        rc = open(os.path.join(outdir, fn)).read().strip()
        reqs = [json.loads(l) for l in open(os.path.join(outdir, name + ".requests.jsonl")) if l.strip()]
        posts = [r for r in reqs if "i" in r]
        tool_posts = [r for r in posts if r["tools"]]
        org_get = any(r.get("GET", "").startswith("/api/config") for r in reqs)
        err = open(os.path.join(outdir, name + ".stderr.txt")).read()
        rej = "auto-rejecting" in err
        tool = finish = ""
        for line in open(os.path.join(outdir, name + ".jsonl")):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            p = e.get("part") or {}
            if e.get("type") == "tool_use":
                st = p.get("state", {})
                tool = st.get("status", "")
                if tool == "error":
                    msg = str(st.get("error", ""))
                    tool += ":" + ("rejected" if "rejected" in msg else "deny-rule" if "specified a rule" in msg else "other")
            elif e.get("type") == "step_finish":
                finish = p.get("reason", "")
            elif e.get("type") == "error":
                finish = "ERROR-EVENT"
        first_err = re.sub(r"\x1b\[[0-9;]*m", "", next((l for l in err.splitlines() if l.strip()), ""))[:200]
        rows.append("\t".join([name, rc, "tool_posts=%d" % len(tool_posts), "tool=" + (tool or "-"),
                               "finish=" + (finish or "-"), "auto_rejecting=%s" % rej,
                               "org_get=%s" % org_get, first_err]))
    with open(os.path.join(outdir, "summary.tsv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.expanduser("~/.opencode/bin/opencode"))
    ap.add_argument("--scratch")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("cases", nargs="*")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="oc-version-") as vh:
        version = subprocess.run([a.bin, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                 env={"PATH": "/usr/bin:/bin", "HOME": vh, "XDG_CONFIG_HOME": vh + "/c",
                                      "XDG_DATA_HOME": vh + "/d", "XDG_CACHE_HOME": vh + "/k",
                                      "XDG_STATE_HOME": vh + "/s", "OPENCODE_DISABLE_AUTOUPDATE": "1"}).stdout.strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit("could not read the opencode version: %r" % version)
    outdir = os.path.join(HERE, version)
    if not a.summarize_only:
        scratch = a.scratch or tempfile.mkdtemp(prefix="oc-capture-")
        names = a.cases or list(CASES)
        for n in names:
            if n not in CASES:
                raise SystemExit("unknown case " + n)
            print(version, n, run_case(a.bin, version, n, CASES[n], scratch, outdir), flush=True)
        if not a.keep and not a.scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    for r in summarize(outdir):
        print(r)


if __name__ == "__main__":
    main()
