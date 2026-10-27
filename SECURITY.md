# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub's vulnerability reporting form](https://github.com/beastlabai/multi-llm-plugin/security/advisories/new).
You can also find it under **Security → Report a vulnerability** in this repository.
Please keep vulnerability details out of public issues and pull requests until
the maintainers have had a chance to investigate and coordinate a fix.

Include the affected version or commit, operating system, host and provider CLI
versions, reproduction steps, expected and observed behaviour, and the potential
impact. Use dummy credentials and remove real tokens, private prompts and other
sensitive information from logs or examples.

## Supported versions

Security reports are accepted for the latest `master` revision and Codex plugin
releases from `v0.1.0` onward. Fixes are developed on `master` and shipped in a
current release; a report involving an older release may require upgrading to
confirm the issue or receive the fix.

## Scope

This policy covers the plugin's own code, including its packaging, configuration
handling, credential redaction and invocation of provider CLIs. Problems in how
the plugin configures or invokes a third-party CLI are in scope.

Vulnerabilities within third-party CLIs, model services and dependencies are
handled by their respective maintainers. Report those through the upstream
project's security process; include the impact on this plugin when relevant.

The plugin's execution permissions and sandbox fallbacks depend on the host,
provider and configuration. See [provider configuration](docs/providers.md) and
[sandbox fallback policies](docs/configuration.md) for their documented limits.
