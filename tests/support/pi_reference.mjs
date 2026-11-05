// Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
//
// This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

// Runs code of an installed pi release, so that tests can compare airedale with pi.
// Usage: node pi_reference.mjs <pi-coding-agent package dir>  (request as JSON on stdin)
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const base = process.argv[2];
const request = JSON.parse(readFileSync(0, "utf8"));
const load = (path) => import(pathToFileURL(`${base}/dist/${path}`).href);

if (request.action === "mcp") {
  // The merge of the global mcp.json (agentDir) and the repo .pi/mcp.json (cwd).
  const { loadMcpConfig } = await load("extensions/mcp/config.js");
  const result = loadMcpConfig({ agentDir: request.agentDir, cwd: request.cwd, projectTrusted: true });
  console.log(
    JSON.stringify({
      servers: result.servers.map((server) => ({ name: server.name, scope: server.scope, config: server.config })),
      errors: result.errors,
    }),
  );
} else if (request.action === "skills") {
  // The skills that pi loads from the given files, and its parse of the last one.
  const { loadSkills } = await load("core/skills.js");
  const { parseFrontmatter } = await load("utils/frontmatter.js");
  const result = loadSkills({
    cwd: request.cwd,
    agentDir: request.cwd,
    skillPaths: request.paths,
    includeDefaults: false,
  });
  let parsed;
  try {
    parsed = parseFrontmatter(readFileSync(request.paths.at(-1), "utf8")).frontmatter;
  } catch (error) {
    parsed = { parseError: error.message };
  }
  console.log(JSON.stringify({ names: result.skills.map((skill) => skill.name), diagnostics: result.diagnostics, parsed }));
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
}
