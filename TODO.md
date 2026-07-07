# TODO

- Port plugin to other code harnesses (Codex, OpenCode, etc.) — currently tailored to Claude Code only

- Improve LLM prompts for better suggestions and fixes. Also, customize prompt wording per code harness, since some harnesses require instructions to be formulated in a specific way to get useful output

- Add support for more code harnesses as LLM providers — extend `providers.yaml` and the provider registry beyond the current set e.g. Verdent

- Add a test phase that creates and/or reviews a test plan, runs it, fixes any failures, then sends test results and applied fixes to other LLMs for review — apply their suggestions (alternate tests, refinements to fixes, etc.) and loop until all tests and suggested tests pass
