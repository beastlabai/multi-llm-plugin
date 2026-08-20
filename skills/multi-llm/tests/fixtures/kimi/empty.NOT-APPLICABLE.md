`empty` (zero bytes on both channels) was never produced by kimi 0.36.1.
Every invocation that reaches the model — success or failure — emits at least
`{"role":"meta","type":"system.version","version":"0.36.1"}` on stdout (59 bytes).
The only zero-byte shape measured is the STALL case (experiment 4), where the
process never terminates at all and so is killed by the outer bound rather than
producing a parseable empty run. The parser must therefore treat
'stdout parses but carries no role=="assistant" record' as the failure shape,
which `ephemeral-only.stdout.txt` pins.
