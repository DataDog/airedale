---
name: example-skill
description: >-
  An example Agent Skill demonstrating the SKILL.md package format.
  Both the Claude Agent SDK and OpenAI Codex consume this same format.
---

# Example Skill

This skill demonstrates the Agent-Skills SKILL.md format used by
`dd-ai-devx-evals`. Place any skill directory under `scenario.skills` in your
`experiment.toml` and the harness will stage it for the active engine.

## When to use this skill

Use this skill as a template when creating new Agent Skills for your evaluation
scenarios. Adapt the frontmatter `name` and `description` fields, then replace
this content with the actual guidance for the model.

## Instructions

1. Read the task prompt carefully before selecting tools.
2. Prefer MCP server tools that directly answer the question over general search.
3. Synthesize a concise, evidence-backed answer from the retrieved information.
4. If no relevant tool is available, state that explicitly rather than guessing.
