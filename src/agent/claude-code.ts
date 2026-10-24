/**
 * The claude-code engine: a pi-ai provider whose models run through the
 * local Claude Code CLI (`claude -p`, headless) instead of a provider API
 * key. It exists so a Claude subscription logged in to the CLI can drive
 * the app with no API key. The CLI owns its own login; this module only
 * spawns it as a subprocess and never reads or passes any credential.
 *
 * Map:
 *   1. The catalog: provider id, the three models (CLI aliases).
 *   2. The prompt builder (pure): pi transcript → system prompt text with
 *      the tool protocol + one user message of content blocks (the
 *      conversation so far, images inline).
 *   3. The reply parser (pure): model text → text + toolCall blocks.
 *   4. The CLI runner: spawn, stream-json in/out, text deltas.
 *   5. The provider: createProvider wiring, keyless auth.
 *
 * Why a text protocol: `claude -p` with its own tools disabled has no way
 * to declare the app's tools natively, so tools are declared as JSON
 * Schema in the system prompt and the model answers with
 * <tool_call>{"name":…,"arguments":{…}}</tool_call> blocks, which the
 * parser turns into pi toolCall content. One CLI run = one model response;
 * pi's agent loop executes the tools and calls again.
 */

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
	createAssistantMessageEventStream,
	createProvider,
	getCurrentSystemPrompt,
	getCurrentTools,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Message,
	type Model,
	type Provider,
	type SimpleStreamOptions,
	type TextContent,
	type Tool,
	type ToolCall,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";

// ─── 1. The catalog ─────────────────────────────────────────────────────────

export const CLAUDE_CODE_PROVIDER_ID = "claude-code";
export const CLAUDE_CODE_API = "claude-code-cli";

/** The CLI's own model aliases (`claude --model <alias>`). */
const MODEL_DEFS = [
	{ id: "sonnet", name: "Claude Sonnet (Claude Code)" },
	{ id: "opus", name: "Claude Opus (Claude Code)" },
	{ id: "haiku", name: "Claude Haiku (Claude Code)" },
] as const;

export function claudeCodeModels(): Model<typeof CLAUDE_CODE_API>[] {
	return MODEL_DEFS.map((def) => ({
		id: def.id,
		name: def.name,
		api: CLAUDE_CODE_API,
		provider: CLAUDE_CODE_PROVIDER_ID,
		baseUrl: "local:claude-cli",
		// No thinking ladder: the CLI picks its own effort; "off" is honest.
		reasoning: false,
		input: ["text", "image"],
		// On the subscription: no per-token charge.
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 32_000,
	}));
}

// ─── 2. The prompt builder ──────────────────────────────────────────────────

/** One Anthropic-style content block for the CLI's stream-json input. */
export type CliContentBlock =
	| { type: "text"; text: string }
	| { type: "image"; source: { type: "base64"; media_type: string; data: string } };

export interface ClaudeCodePrompt {
	/** Passed via --system-prompt-file (replaces Claude Code's own prompt). */
	system: string;
	/** The one user message sent on stdin. */
	content: CliContentBlock[];
}

const PROTOCOL = `# How to use tools

You are running inside an application that executes tools for you. You
cannot run anything yourself; the application runs the tools listed below.

To call a tool, write a block in exactly this form, with valid JSON that
matches the tool's parameters schema:

<tool_call>{"name": "<tool name>", "arguments": { ... }}</tool_call>

Rules:
- You may write normal text before your tool calls. You may make several
  tool calls in one message.
- After your last tool call, end your message. The application runs the
  tools and shows you the results in the next turn.
- Never write tool results yourself and never write <turn> tags.
- When you don't need a tool, just answer in plain text (Markdown is fine).`;

function renderTools(tools: Tool[]): string {
	if (tools.length === 0) return "";
	const decls = tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters }));
	return `\n\n# Available tools\n\n${JSON.stringify(decls, null, 2)}`;
}

function toolCallTag(call: ToolCall): string {
	return `<tool_call>${JSON.stringify({ name: call.name, arguments: call.arguments })}</tool_call>`;
}

/**
 * Pure: the pi transcript → the CLI's system prompt + one user message.
 * System messages replay into one prompt (pi's own helpers); every other
 * message renders in order as a <turn> block, prior tool calls in the same
 * tag form the model is asked to write, tool results with their call id.
 */
export function buildClaudeCodePrompt(context: TranscriptContext): ClaudeCodePrompt {
	const messages = context.messages;
	const basePrompt = getCurrentSystemPrompt(messages);
	const tools = getCurrentTools(messages);
	const system = [basePrompt, tools.length > 0 ? PROTOCOL + renderTools(tools) : ""]
		.filter((s) => s.length > 0)
		.join("\n\n");

	const content: CliContentBlock[] = [];
	let buf = "The conversation so far:\n\n";
	const flush = () => {
		if (buf.length > 0) content.push({ type: "text", text: buf });
		buf = "";
	};
	const pushImage = (mimeType: string, data: string) => {
		flush();
		content.push({ type: "image", source: { type: "base64", media_type: mimeType, data } });
	};

	for (const message of messages as Message[]) {
		if (message.role === "system") continue;
		if (message.role === "user") {
			buf += `<turn role="user">\n`;
			if (typeof message.content === "string") buf += message.content;
			else
				for (const block of message.content) {
					if (block.type === "text") buf += block.text;
					else pushImage(block.mimeType, block.data);
				}
			buf += `\n</turn>\n\n`;
		} else if (message.role === "assistant") {
			buf += `<turn role="assistant">\n`;
			for (const block of message.content) {
				if (block.type === "text") buf += block.text + "\n";
				else if (block.type === "toolCall") buf += toolCallTag(block) + "\n";
				// thinking blocks are the model's private notes; not replayed
			}
			buf += `</turn>\n\n`;
		} else if (message.role === "toolResult") {
			buf += `<turn role="tool_result" tool="${message.toolName}" call_id="${message.toolCallId}"${message.isError ? ` is_error="true"` : ""}>\n`;
			for (const block of message.content) {
				if (block.type === "text") buf += block.text;
				else pushImage(block.mimeType, block.data);
			}
			buf += `\n</turn>\n\n`;
		}
	}
	buf += "Write the assistant's next message now.";
	flush();
	return { system, content };
}

// ─── 3. The reply parser ────────────────────────────────────────────────────

const OPEN = "<tool_call>";
const CLOSE = "</tool_call>";

export interface ParsedToolCall {
	name: string;
	arguments: Record<string, unknown>;
	/** Set when the block's JSON was unusable; arguments is then {}. */
	argumentsError?: string;
}

export interface ParsedReply {
	text: string;
	toolCalls: ParsedToolCall[];
}

function stripFence(s: string): string {
	const t = s.trim();
	const m = /^```(?:json)?\s*([\s\S]*?)\s*```$/.exec(t);
	return m?.[1] ?? t;
}

function parseCallBody(body: string): ParsedToolCall {
	const raw = stripFence(body);
	try {
		const obj = JSON.parse(raw) as unknown;
		if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("not a JSON object");
		const rec = obj as Record<string, unknown>;
		const name = typeof rec.name === "string" ? rec.name : "";
		if (!name) throw new Error('missing "name"');
		let args: unknown = rec.arguments ?? rec.input ?? {};
		if (typeof args === "string") args = JSON.parse(args) as unknown;
		if (!args || typeof args !== "object" || Array.isArray(args)) {
			return { name, arguments: {}, argumentsError: `"arguments" must be a JSON object; got: ${raw}` };
		}
		return { name, arguments: args as Record<string, unknown> };
	} catch (err) {
		const nameGuess = /"name"\s*:\s*"([^"]+)"/.exec(raw)?.[1] ?? "invalid_tool_call";
		return {
			name: nameGuess,
			arguments: {},
			argumentsError: `tool_call block was not valid JSON (${err instanceof Error ? err.message : String(err)}); the block was: ${raw}`,
		};
	}
}

/** Pure: the model's full reply → visible text + tool calls, in order. An
 *  unclosed final <tool_call> (a cut-off reply) still parses as a call. */
export function parseClaudeCodeReply(reply: string): ParsedReply {
	let text = "";
	const toolCalls: ParsedToolCall[] = [];
	let i = 0;
	while (i < reply.length) {
		const open = reply.indexOf(OPEN, i);
		if (open === -1) {
			text += reply.slice(i);
			break;
		}
		text += reply.slice(i, open);
		const close = reply.indexOf(CLOSE, open + OPEN.length);
		const body = reply.slice(open + OPEN.length, close === -1 ? reply.length : close);
		toolCalls.push(parseCallBody(body));
		i = close === -1 ? reply.length : close + CLOSE.length;
	}
	return { text: text.replace(/\n{3,}/g, "\n\n").trim(), toolCalls };
}

/** Pure, for live streaming: the part of a partial reply that is safe to
 *  show — text outside tool_call blocks, holding back an unclosed block and
 *  any trailing prefix of "<tool_call>" that may still become one. */
export function visibleStreamText(partial: string): string {
	let out = "";
	let i = 0;
	while (i < partial.length) {
		const open = partial.indexOf(OPEN, i);
		if (open === -1) {
			out += partial.slice(i);
			break;
		}
		out += partial.slice(i, open);
		const close = partial.indexOf(CLOSE, open + OPEN.length);
		if (close === -1) return out;
		i = close + CLOSE.length;
	}
	for (let k = Math.min(OPEN.length - 1, out.length); k > 0; k--) {
		if (OPEN.startsWith(out.slice(-k))) return out.slice(0, -k);
	}
	return out;
}

// ─── 4. The CLI runner ──────────────────────────────────────────────────────

/** The binary to spawn; tests point this at a fake. */
export function claudeBin(): string {
	return process.env.LH_CLAUDE_BIN || "claude";
}

/** The CLI flags for one headless run (no prompt text, no secrets). */
export function claudeArgs(modelId: string, systemPromptFile: string): string[] {
	return [
		"-p",
		"--model",
		modelId,
		"--system-prompt-file",
		systemPromptFile,
		"--tools",
		"",
		"--setting-sources",
		"",
		"--strict-mcp-config",
		"--disable-slash-commands",
		"--no-session-persistence",
		"--input-format",
		"stream-json",
		"--output-format",
		"stream-json",
		"--verbose",
		"--include-partial-messages",
	];
}

/** Child env: the parent's, minus anything that would switch the CLI from
 *  its own subscription login to API billing. Nothing is ever added. */
function childEnv(): NodeJS.ProcessEnv {
	const env = { ...process.env };
	delete env.ANTHROPIC_API_KEY;
	delete env.ANTHROPIC_AUTH_TOKEN;
	delete env.CLAUDECODE;
	return env;
}

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function usageFrom(raw: unknown): Usage {
	const u = (raw ?? {}) as Record<string, unknown>;
	const n = (k: string) => (typeof u[k] === "number" ? (u[k] as number) : 0);
	const input = n("input_tokens");
	const output = n("output_tokens");
	const cacheRead = n("cache_read_input_tokens");
	const cacheWrite = n("cache_creation_input_tokens");
	return { ...ZERO_USAGE, input, output, cacheRead, cacheWrite, totalTokens: input + output + cacheRead + cacheWrite };
}

const TIMEOUT_MS = 10 * 60 * 1000;

/** One model response through `claude -p`. Never throws: every failure
 *  ends the stream with an error AssistantMessage. */
export function streamClaudeCode(
	model: Model<string>,
	context: TranscriptContext,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const partial: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: { ...ZERO_USAGE },
		stopReason: "pending",
		timestamp: Date.now(),
	};
	const fail = (reason: "error" | "aborted", message: string) => {
		const err: AssistantMessage = { ...partial, stopReason: reason, errorMessage: message };
		stream.push({ type: "error", reason, error: err });
		stream.end(err);
	};

	queueMicrotask(() => {
		let dir: string | undefined;
		try {
			if (options?.signal?.aborted) return fail("aborted", "Request was aborted");
			const prompt = buildClaudeCodePrompt(context);
			dir = fs.mkdtempSync(path.join(os.tmpdir(), "lh-claude-code-"));
			const sysFile = path.join(dir, "system-prompt.md");
			fs.writeFileSync(sysFile, prompt.system);
			const args = claudeArgs(model.id, sysFile);
			const bin = claudeBin();
			if (process.env.NODE_ENV !== "test")
				console.info(`[claude-code] ${bin} ${args.map((a) => (a === "" ? '""' : a)).join(" ")} (cwd ${dir})`);
			const child = spawn(bin, args, { cwd: dir, env: childEnv(), stdio: ["pipe", "pipe", "pipe"] });
			const cleanup = () => {
				if (dir) fs.rmSync(dir, { recursive: true, force: true });
				dir = undefined;
			};

			let raw = ""; // all text deltas so far
			let emitted = ""; // visible text already streamed
			let textOpen = false;
			let result: Record<string, unknown> | undefined;
			let stderr = "";
			let lineBuf = "";
			let settled = false;
			partial.content = [];
			stream.push({ type: "start", partial });

			const textBlock = (): TextContent => partial.content[0] as TextContent;
			const pushVisible = () => {
				const visible = visibleStreamText(raw);
				if (!visible.startsWith(emitted) || visible.length <= emitted.length) return;
				const delta = visible.slice(emitted.length);
				if (!textOpen) {
					if (delta.trim().length === 0) return; // don't open on leading whitespace
					partial.content = [{ type: "text", text: "" }];
					textOpen = true;
					stream.push({ type: "text_start", contentIndex: 0, partial });
				}
				emitted = visible;
				textBlock().text += delta;
				stream.push({ type: "text_delta", contentIndex: 0, delta, partial });
			};
			const onLine = (line: string) => {
				if (!line.trim()) return;
				let ev: Record<string, unknown>;
				try {
					ev = JSON.parse(line) as Record<string, unknown>;
				} catch {
					return;
				}
				if (ev.type === "stream_event") {
					const inner = ev.event as { type?: string; delta?: { type?: string; text?: string } } | undefined;
					if (inner?.type === "content_block_delta" && inner.delta?.type === "text_delta" && inner.delta.text) {
						raw += inner.delta.text;
						pushVisible();
					}
				} else if (ev.type === "result") {
					result = ev;
				}
			};

			const timer = setTimeout(() => child.kill("SIGTERM"), TIMEOUT_MS);
			const onAbort = () => child.kill("SIGTERM");
			options?.signal?.addEventListener("abort", onAbort, { once: true });

			child.stdout.setEncoding("utf8");
			child.stdout.on("data", (chunk: string) => {
				lineBuf += chunk;
				let nl: number;
				while ((nl = lineBuf.indexOf("\n")) !== -1) {
					onLine(lineBuf.slice(0, nl));
					lineBuf = lineBuf.slice(nl + 1);
				}
			});
			child.stderr.setEncoding("utf8");
			child.stderr.on("data", (chunk: string) => {
				stderr = (stderr + chunk).slice(-4000);
			});
			child.stdin.on("error", () => {}); // a CLI that died early: reported on close
			child.stdin.end(JSON.stringify({ type: "user", message: { role: "user", content: prompt.content } }) + "\n");

			const finish = (code: number | null, spawnError?: Error) => {
				if (settled) return;
				settled = true;
				clearTimeout(timer);
				options?.signal?.removeEventListener("abort", onAbort);
				if (lineBuf) onLine(lineBuf);
				cleanup();
				if (options?.signal?.aborted) return fail("aborted", "Request was aborted");
				if (spawnError) return fail("error", `could not run the Claude Code CLI (${bin}): ${spawnError.message}`);
				if (result) partial.usage = usageFrom(result.usage);
				if (!result || result.is_error === true) {
					const why =
						(result && typeof result.result === "string" && result.result) ||
						(result && Array.isArray(result.errors) && result.errors.join("; ")) ||
						stderr.trim() ||
						`claude exited with code ${code}`;
					return fail("error", `Claude Code CLI: ${why}`);
				}
				const fullText = typeof result.result === "string" && result.result.length > 0 ? result.result : raw;
				const parsed = parseClaudeCodeReply(fullText);
				const content: AssistantMessage["content"] = [];
				if (parsed.text) content.push({ type: "text", text: parsed.text });
				for (const call of parsed.toolCalls) {
					const block: ToolCall = {
						type: "toolCall",
						id: `toolu_cc_${randomUUID().replace(/-/g, "").slice(0, 20)}`,
						name: call.name,
						arguments: call.arguments as ToolCall["arguments"],
						...(call.argumentsError ? { argumentsError: call.argumentsError } : {}),
					};
					content.push(block);
				}
				// Close the live text block with the authoritative text.
				if (textOpen) {
					textBlock().text = parsed.text;
					stream.push({ type: "text_end", contentIndex: 0, content: parsed.text, partial });
				} else if (parsed.text) {
					partial.content = [{ type: "text", text: "" }];
					stream.push({ type: "text_start", contentIndex: 0, partial });
					textBlock().text = parsed.text;
					stream.push({ type: "text_delta", contentIndex: 0, delta: parsed.text, partial });
					stream.push({ type: "text_end", contentIndex: 0, content: parsed.text, partial });
				}
				content.forEach((block, k) => {
					if (block.type !== "toolCall") return;
					partial.content = [...partial.content, { ...block, arguments: {} }];
					stream.push({ type: "toolcall_start", contentIndex: k, partial });
					stream.push({ type: "toolcall_delta", contentIndex: k, delta: JSON.stringify(block.arguments), partial });
					partial.content[k] = block;
					stream.push({ type: "toolcall_end", contentIndex: k, toolCall: block, partial });
				});
				const stopReason = parsed.toolCalls.length > 0 ? "toolUse" : "stop";
				const message: AssistantMessage = { ...partial, content, stopReason };
				stream.push({ type: "done", reason: stopReason, message });
				stream.end(message);
			};
			child.on("error", (err) => finish(null, err));
			child.on("close", (code) => finish(code));
		} catch (err) {
			if (dir) fs.rmSync(dir, { recursive: true, force: true });
			fail("error", `claude-code engine: ${err instanceof Error ? err.message : String(err)}`);
		}
	});
	return stream;
}

// ─── 5. The provider ────────────────────────────────────────────────────────

/** The keyless provider: auth always resolves (the CLI holds its own
 *  login), and both stream entry points run the CLI. */
export function claudeCodeProvider(): Provider {
	return createProvider({
		id: CLAUDE_CODE_PROVIDER_ID,
		name: "Claude Code (subscription)",
		auth: { apiKey: { name: "Claude Code CLI login", resolve: async () => ({ auth: {}, source: "claude CLI" }) } },
		models: claudeCodeModels(),
		api: {
			stream: (model, context, options) => streamClaudeCode(model, context, options),
			streamSimple: (model, context, options) => streamClaudeCode(model, context, options),
		},
	}) as Provider;
}
