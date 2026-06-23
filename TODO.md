# TODO

1. Port plugin to other code harnesses (Codex, OpenCode, etc.) — currently tailored to Claude Code only

2. Add an install/configure command that detects available harness CLIs, discovers supported models from each, and lets users assign one or default models per mode (review, ask, apply, etc.) and update the providers.yaml accordingly

3. Improve LLM prompts for better suggestions and fixes — customize prompt wording per code harness, since some harnesses require instructions to be formulated in a specific way to get useful output

4. Add support for more code harnesses as LLM providers — extend `providers.yaml` and the provider registry beyond the current set (Cursor Agent, Gemini CLI, Codex, OpenCode, Kilocode, Claude Code)

5. Add a test phase that creates and/or reviews a test plan, runs it, fixes any failures, then sends test results and applied fixes to other LLMs for review — apply their suggestions (alternate tests, refinements to fixes, etc.) and loop until all tests and suggested tests pass
