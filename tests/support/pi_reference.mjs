// Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
//
// This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

// Runs code of an installed pi release, so that tests can compare airedale with pi.
// Usage: node pi_reference.mjs <pi-coding-agent package dir>  (request as JSON on stdin)
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const base = process.argv[2];
const request = JSON.parse(readFileSync(0, "utf8"));
const load = (path) => import(pathToFileURL(`${base}/dist/${path}`).href);

// Files are written to a temporary directory: the requests hold the content, not paths, so that they are
// the same on every machine (the answers are recorded in tests/fixtures/pi_reference).
const sandbox = mkdtempSync(join(tmpdir(), "pi-reference-"));
process.on("exit", () => rmSync(sandbox, { recursive: true, force: true }));

if (request.action === "mcp") {
  // The merge of the global mcp.json (the object ``global_file``) and the repo .pi/mcp.json (the bytes
  // ``project_b64``, or no file).
  const { loadMcpConfig } = await load("extensions/mcp/config.js");
  const agentDir = join(sandbox, "agent");
  const cwd = join(sandbox, "cwd");
  mkdirSync(agentDir);
  mkdirSync(join(cwd, ".pi"), { recursive: true });
  writeFileSync(join(agentDir, "mcp.json"), JSON.stringify(request.global_file));
  if (request.project_b64 !== null && request.project_b64 !== undefined) {
    writeFileSync(join(cwd, ".pi", "mcp.json"), Buffer.from(request.project_b64, "base64"));
  }
  const result = loadMcpConfig({ agentDir, cwd, projectTrusted: true });
  console.log(
    JSON.stringify({
      servers: result.servers.map((server) => ({ name: server.name, scope: server.scope, config: server.config })),
      error_count: result.errors.length,
    }),
  );
} else if (request.action === "skills") {
  // The name that pi gives the skill in ``<dir>/project/SKILL.md`` with this content (none: pi does not load it).
  const { loadSkills } = await load("core/skills.js");
  const file = join(sandbox, "project", "SKILL.md");
  mkdirSync(join(sandbox, "project"));
  writeFileSync(file, request.content);
  const result = loadSkills({ cwd: sandbox, agentDir: sandbox, skillPaths: [file], includeDefaults: false });
  console.log(JSON.stringify({ names: result.skills.map((skill) => skill.name) }));
} else if (request.action === "names") {
  // The MCP tool names, with servers registered in the given order. Same
  // collision rule as the registration loop of the MCP extension.
  const { createMcpToolName } = await load("extensions/mcp/tools.js");
  const owners = new Map();
  const result = [];
  for (const [server, tools] of request.servers) {
    const plain = [...new Set(tools)].map((tool) => createMcpToolName(server, tool));
    const current = new Set();
    for (const tool of tools) {
      const owner = `${server}\0${tool}`;
      const name = createMcpToolName(
        server,
        tool,
        (candidate) =>
          (owners.has(candidate) && owners.get(candidate) !== owner) ||
          current.has(candidate) ||
          plain.indexOf(candidate) !== plain.lastIndexOf(candidate),
      );
      owners.set(name, owner);
      current.add(name);
      result.push({ server, tool, name });
    }
  }
  console.log(JSON.stringify(result));
} else if (request.action === "urls") {
  // What JavaScript's URL (the one pi calls) makes of each string.
  console.log(
    JSON.stringify(
      request.urls.map((value) => {
        if (!URL.canParse(value)) return null;
        const url = new URL(value);
        return {
          scheme: url.protocol.slice(0, -1),
          hostname: url.hostname,
          port: url.port,
          pathname: url.pathname,
          search: url.search,
          hash: url.hash,
        };
      }),
    ),
  );
} else if (request.action === "retry") {
  // The events of pi's agent loop (mock model) for a response that failed with a
  // tool call with id "", and a retry that uses the id "" again.
  Date.now = () => 1_700_000_000_000; // The same events on every run.
  const modules = `${base}/../`;
  const { agentLoop } = await import(pathToFileURL(`${modules}pi-agent-core/dist/agent-loop.js`).href);
  const { AssistantMessageEventStream } = await import(
    pathToFileURL(`${modules}pi-ai/dist/utils/event-stream.js`).href
  );
  const model = { id: "claude-sonnet-4-6", provider: "anthropic", api: "anthropic-messages" };
  const usage = {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  const message = (content, stopReason) => ({
    role: "assistant",
    api: model.api,
    provider: model.provider,
    model: model.id,
    content,
    stopReason,
    usage,
    timestamp: Date.now(),
    ...(stopReason === "error" ? { errorMessage: "overloaded" } : {}),
  });
  const call = (name) => ({ type: "toolCall", id: "", name, arguments: {} });
  const context = {
    messages: [],
    tools: [
      {
        name: "worked",
        label: "worked",
        description: "test",
        parameters: { type: "object", properties: {} },
        execute: async () => ({ content: [{ type: "text", text: "worked result" }], details: {} }),
      },
    ],
  };
  const events = [];
  const rounds = [
    [message([call("failed")], "error")],
    [message([call("worked")], "toolUse"), message([{ type: "text", text: "done" }], "stop")],
  ];
  for (const responses of rounds) {
    const streamFn = () => {
      const response = responses.shift();
      const stream = new AssistantMessageEventStream();
      stream.push(
        response.stopReason === "error"
          ? { type: "error", reason: "error", error: response }
          : { type: "done", reason: response.stopReason, message: response },
      );
      return stream;
    };
    const prompt = [{ role: "user", content: "run", timestamp: Date.now() }];
    for await (const event of agentLoop(prompt, context, { model, convertToLlm: (m) => m }, undefined, streamFn)) {
      events.push(event);
    }
  }
  events.push({ type: "agent_settled", aborted: false });
  console.log(JSON.stringify(events));
} else if (request.action === "parallel") {
  // The events of pi's agent loop (mock model) for one turn with two parallel tool calls that both have
  // the id "": the first call (slow) waits until the loop has emitted the end record of the second call (fast),
  // so the end records come in the other order. A gate, not a timer: no scheduling delay changes the order.
  Date.now = () => 1_700_000_000_000;
  const modules = `${base}/../`;
  const { agentLoop } = await import(pathToFileURL(`${modules}pi-agent-core/dist/agent-loop.js`).href);
  const { AssistantMessageEventStream } = await import(
    pathToFileURL(`${modules}pi-ai/dist/utils/event-stream.js`).href
  );
  const model = { id: "claude-sonnet-4-6", provider: "anthropic", api: "anthropic-messages" };
  const usage = {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  const message = (content, stopReason) => ({
    role: "assistant",
    api: model.api,
    provider: model.provider,
    model: model.id,
    content,
    stopReason,
    usage,
    timestamp: Date.now(),
  });
  let openGate;
  const gate = new Promise((resolve) => {
    openGate = resolve;
  });
  const tool = (name, wait) => ({
    name,
    label: name,
    description: "test",
    parameters: { type: "object", properties: {} },
    execute: async () => {
      if (wait) await wait;
      return { content: [{ type: "text", text: `${name} result` }], details: {} };
    },
  });
  const context = { messages: [], tools: [tool("slow", gate), tool("fast", null)] };
  const responses = [
    message(
      [
        { type: "toolCall", id: "", name: "slow", arguments: {} },
        { type: "toolCall", id: "", name: "fast", arguments: {} },
      ],
      "toolUse",
    ),
    message([{ type: "text", text: "done" }], "stop"),
  ];
  const streamFn = () => {
    const response = responses.shift();
    const stream = new AssistantMessageEventStream();
    stream.push({ type: "done", reason: response.stopReason, message: response });
    return stream;
  };
  const events = [];
  const prompt = [{ role: "user", content: "run", timestamp: Date.now() }];
  for await (const event of agentLoop(prompt, context, { model, convertToLlm: (m) => m }, undefined, streamFn)) {
    events.push(event);
    if (event.type === "tool_execution_end" && event.toolName === "fast") openGate();
  }
  events.push({ type: "agent_settled", aborted: false });
  // The duration of a tool depends on the machine: the same events on every run.
  const zeroDurations = (value) => {
    if (Array.isArray(value)) value.forEach(zeroDurations);
    else if (value !== null && typeof value === "object") {
      for (const key of Object.keys(value)) {
        if (key === "durationMs") value[key] = 0;
        else zeroDurations(value[key]);
      }
    }
  };
  zeroDurations(events);
  console.log(JSON.stringify(events));
}
