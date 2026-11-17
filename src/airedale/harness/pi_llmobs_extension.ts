// Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
//
// This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

/**
 * Airedale LLMObs collector extension for pi.
 *
 * Airedale gives this file to pi with `-e`. It does not make spans and it does
 * not send data out of the pi process. For each model request of the agent
 * loop, it appends 2 session entries. In JSON mode, pi writes them to stdout as
 * `entry_appended` records, in order with the other records. Airedale reads
 * them and makes the LLMObs spans (see `pi_spans.py`).
 *
 * - `airedale.llmobs.request`: `{v, seq, t, prefix, appended, count}`. `t` is
 *   the time (epoch ms) when pi sends the request (before the provider retries
 *   of its HTTP client). The request messages are the first `prefix` messages
 *   of the previous sent request, then `appended`. `count` is the number of
 *   messages. They are the pi messages after the `context` and
 *   `context_with_system` handlers of the extensions loaded before this one.
 *   When the new messages are too large, the entry has `skipped: true` and
 *   `appended: null`, and the next entry has the full list (`prefix` 0).
 * - `airedale.llmobs.response`: `{v, seq, t, status, requestId}`. `t` is the
 *   time when the response headers of a successful response arrive.
 *
 * Opaque provider data (signatures), tool `details` and image data are
 * removed, so that the stream stays small.
 *
 * The handlers never change the request and never throw: a problem here must
 * not change the run. Only type imports are used, so pi can load this file
 * without `node_modules`.
 */

const REQUEST_ENTRY = "airedale.llmobs.request";
const RESPONSE_ENTRY = "airedale.llmobs.response";
const DATA_VERSION = 1;
// Opaque pi fields that are removed: tool `details` on a message, and the
// provider signatures on its content blocks. The data of an image content
// block is replaced by its size. Only these places are changed: tool call
// arguments can have keys and objects of the same shape, and they are kept.
const MESSAGE_OPAQUE_KEYS = new Set(["details"]);
const BLOCK_OPAQUE_KEYS = new Set(["textSignature", "thinkingSignature", "thoughtSignature"]);
// Maximum size (UTF-16 code units of the serialized JSON) of the new messages
// of one request entry. Above it, the entry has `skipped: true` and no
// messages, so that one entry can never get near the stdout record limit of
// airedale (64 MiB). The next request then sends its full message list.
// `AIREDALE_PI_LLMOBS_MAX_APPENDED_CHARS` changes it (for tests).
const MAX_APPENDED_CHARS =
	Number((globalThis as Api).process?.env?.AIREDALE_PI_LLMOBS_MAX_APPENDED_CHARS) || 8 * 1024 * 1024;

interface PendingRequest {
	seq: number;
	messages: string[];
	sent: boolean;
	responded: boolean;
}

// The pi API types are not available without node_modules.
type Api = any;

export default function airedaleLlmobsCollector(pi: Api): void {
	let seq = 0;
	// The serialized messages of the last request that was sent.
	let lastSent: string[] = [];
	let pending: PendingRequest | undefined;

	pi.on("context_with_system", (event: { messages?: unknown[] }) => {
		try {
			const messages = Array.isArray(event?.messages) ? event.messages : [];
			seq += 1;
			pending = { seq, messages: messages.map(serialize), sent: false, responded: false };
		} catch {
			pending = undefined;
		}
		return undefined;
	});

	pi.on("before_provider_request", () => {
		try {
			// Other model calls (for example compaction) also use this hook.
			// Only the first request after a `context_with_system` belongs to it.
			if (pending === undefined || pending.sent) return undefined;
			pending.sent = true;
			const current = pending.messages;
			let prefix = 0;
			while (prefix < lastSent.length && prefix < current.length && lastSent[prefix] === current[prefix]) {
				prefix += 1;
			}
			const t = Date.now();
			const added = current.slice(prefix);
			let size = 0;
			for (const text of added) size += text.length;
			if (size > MAX_APPENDED_CHARS) {
				// Airedale does not know this list: the next change must be a full list.
				lastSent = [];
				pi.appendEntry(REQUEST_ENTRY, {
					v: DATA_VERSION,
					seq: pending.seq,
					t,
					prefix: 0,
					appended: null,
					count: current.length,
					skipped: true,
				});
				return undefined;
			}
			lastSent = current;
			pi.appendEntry(REQUEST_ENTRY, {
				v: DATA_VERSION,
				seq: pending.seq,
				t,
				prefix,
				appended: added.map((text) => JSON.parse(text)),
				count: current.length,
			});
		} catch {
			// Telemetry only.
		}
		return undefined;
	});

	pi.on("after_provider_response", (event: { status?: number; headers?: Record<string, string> }) => {
		try {
			if (pending === undefined || !pending.sent || pending.responded) return;
			pending.responded = true;
			pi.appendEntry(RESPONSE_ENTRY, {
				v: DATA_VERSION,
				seq: pending.seq,
				t: Date.now(),
				status: typeof event?.status === "number" ? event.status : null,
				requestId: requestId(event?.headers),
			});
		} catch {
			// Telemetry only.
		}
	});
}

function serialize(message: unknown): string {
	return JSON.stringify(withoutOpaqueFields(message));
}

function withoutOpaqueFields(message: unknown): unknown {
	if (!isRecord(message)) return message;
	const copy: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(message)) {
		if (MESSAGE_OPAQUE_KEYS.has(key)) continue;
		copy[key] = key === "content" && Array.isArray(value) ? value.map(blockWithoutOpaqueFields) : value;
	}
	return copy;
}

function blockWithoutOpaqueFields(block: unknown): unknown {
	if (!isRecord(block)) return block;
	if (block.type === "image" && typeof block.data === "string") {
		return { type: "image", mimeType: block.mimeType, bytes: block.data.length };
	}
	const copy: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(block)) {
		if (!BLOCK_OPAQUE_KEYS.has(key)) copy[key] = value;
	}
	return copy;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function requestId(headers: Record<string, string> | undefined): string | null {
	if (headers === null || typeof headers !== "object") return null;
	for (const [name, value] of Object.entries(headers)) {
		if (/(^|-)request-id$/i.test(name) && typeof value === "string" && value) return value;
	}
	return null;
}
