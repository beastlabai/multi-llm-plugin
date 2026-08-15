# TODO

- Port plugin to other code harnesses (Codex, OpenCode, etc.) — currently tailored to Claude Code only.

- Improve LLM prompts for better suggestions and fixes. Also, customize prompt wording per code harness, since some harnesses require instructions to be formulated in a specific way to get useful output.

- Add support for more code harnesses as LLM providers — extend `providers.yaml` and the provider registry beyond the current set e.g. Verdent.

- Add a test phase that creates and/or reviews a test plan, runs it, fixes any failures, then sends test results and applied fixes to other LLMs for review — apply their suggestions (alternate tests, refinements to fixes, etc.) and loop until all tests and suggested tests pass.

- Speed up the slow --apply-suggestions phase which is quite slow since it's not parallelized (to avoid editing overlapping sections of the plan) and only small batches of suggestions are applied at a time (to focus the LLM on only these suggestions so as to not overwhelm it). However, maybe there's a way to speed this up somehow. Try to do the same for --apply-code-fixes phase.

-- Add a final phase at the end of --review-plan or --review-code phases that feeds in all the suggestions made by other coding harnesses on the same run to each coding harnesses to see if they can add more suggestions (not validate other suggestions, but use them as a basis to reveal other aspects they had not considered).