`multi-assistant.stdout.txt` is SYNTHESIZED, not captured, and is labelled so
deliberately. The committed `stub_server.py` never emits tool calls (an explicit
non-goal in its docstring), so a run with two assistant turns cannot be produced
offline. Every record in the file uses the grammar verified against kimi 0.36.1 in
`success-stream-json.stdout.txt`.

What it must pin: step 1 of the §3.4 unwrap selects the **last** record whose
`role == "assistant"` and reads its `content` — not the first, not a
concatenation of both (the muse trap), and not the trailing
`session.resume_hint` meta record, **which also carries a `content` key** and is
the specific trap this CLI adds.
