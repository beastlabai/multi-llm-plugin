# Attribution and third-party terms

Trademark, licensing and brand-mark notes for the harnesses and providers
multi-llm talks to. Back to the [README](../README.md).

---

## Not affiliated

BeastLab.ai provides this plugin as-is, with no warranties or liabilities of any kind. BeastLab.ai is not affiliated with, endorsed by, or sponsored by any of the code harnesses (Cursor, Claude Code, Codex, OpenCode, Gemini CLI, Grok Build, Kilo Code, Cline, goose, Aider, Antigravity CLI, Pi, Muse Code, Kimi Code, Qwen Code, GitHub Copilot CLI, Mistral Vibe, oh-my-pi, etc.) or third-party LLM providers (Anthropic, OpenAI, Google, xAI, Moonshot AI, Z.ai/Zhipu, MiniMax, NVIDIA, Alibaba/Qwen, DeepSeek, Mistral AI, Xiaomi, Arcee, etc.) referenced in this project. Trademarks and product names belong to their respective owners.

## One harness is not open source

**GitHub Copilot CLI is proprietary and is used under the GitHub Copilot CLI EULA.** multi-llm ships an *adapter* for it, never the binary — you install it yourself and accept GitHub's terms directly. The project's own end-to-end test image installs it to exercise that adapter, and that image is build-local by construction: its tags carry no registry component, so `docker push` fails outright, and CI builds no image at all.

## Brand marks in HTML reports

HTML reports embed small brand marks to label which harness and which model produced each suggestion — nominative use only, and no endorsement is implied. Each bundled mark is fetched from its vendor's own favicon or brand source (see the provenance comment beside every entry in `utils/html_report_generator.py`'s `_ICON_DEFS`), except Primer Octicons' `copilot-16`, which is MIT-licensed. Where no first-party mark is reachable the report renders a generated letter badge instead of a redrawn or re-hosted logo.
