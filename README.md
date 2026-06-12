# dd-ai-devx-evals

> Config-driven evaluation harness for agentic LLM runs, reporting to Datadog
> LLM Observability Experiments.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

`dd-ai-devx-evals` runs an evaluation matrix of **model × scenario × task** through
provider-native agentic SDKs (Anthropic via [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/),
OpenAI via [`openai-codex`](https://pypi.org/project/openai-codex/)), exposes
MCP servers and skills to the model, and reports every run to
[Datadog LLM Observability Experiments](https://docs.datadoghq.com/llm_observability/).

> **Status:** under active construction. This README is a placeholder that will
> be completed once the implementation lands.

## License

Licensed under the [Apache License 2.0](./LICENSE).
