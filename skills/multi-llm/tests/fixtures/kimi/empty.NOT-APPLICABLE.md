`empty` (zero bytes on both channels) was never produced by kimi 0.36.1, nor by
2.1.1 when every row was re-captured on 2026-10-08.
Every invocation that reaches the model — success or failure — emits at least
`{"role":"meta","type":"system.version","version":"2.1.1"}` on stdout (58 bytes;
the 0.36.1 line was 59) before it sends a request.
The STALL case (experiment 4) is not a zero-byte run either, though the 0.36.1
probe recorded it as one: that probe read stdout only at exit, and its 420 s
bound killed the run first. Captured to a file at 2.1.1, the `system.version`
line is on stdout from the start, and a stalled request ends at kimi's own ~300 s
request timeout — `provider.connection_error: Request timed out.`, exit 1 at
302 s with the adapter's pinned single attempt (the probe's own 0.36.1 server log
shows the same ~301 s per attempt). The parser must therefore treat
'stdout parses but carries no role=="assistant" record' as the failure shape,
which `ephemeral-only.stdout.txt` pins.
