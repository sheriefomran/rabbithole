/**
 * claude-code engine — the prompt builder, the tool-call parser, the live
 * text holdback, and one whole run against a FAKE `claude` binary (a small
 * node script that records its argv and stdin and replies in the CLI's
 * stream-json shape). No real model is ever called here.
 */

import test, { describe } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { getSupportedThinkingLevels, normalizeContext, type AssistantMessage, type Context, type Tool } from "@earendil-works/pi-ai";
import {
	buildClaudeCodePrompt,
	claudeArgs,
	claudeCodeModels,
	parseClaudeCodeReply,
	streamClaudeCode,
	visibleStreamText,
} from "../src/agent/claude-code.ts";
import { piModels } from "../src/agent/models.ts";

const quizTool: Tool = {
	name: "ask_user_question",
	description: "Ask the learner a question.",
	parameters: { type: "object", properties: { question: { type: "string" } }, required: ["question"] } as never,
};

function ctx(c: Partial<Context>): ReturnType<typeof normalizeContext> {
	return normalizeContext({ messages: [], ...c } as Context);
}

describe("buildClaudeCodePrompt", () => {
	test("the system prompt carries the app prompt, the tool protocol and the tools as JSON Schema", () => {
		const p = buildClaudeCodePrompt(
			ctx({ systemPrompt: "You are the prober.", tools: [quizTool], messages: [{ role: "user", content: "hi", timestamp: 1 }] }),
		);
		assert.ok(p.system.startsWith("You are the prober."));
		assert.ok(p.system.includes("<tool_call>"));
		assert.ok(p.system.includes('"name": "ask_user_question"'));
		assert.ok(p.system.includes('"required": ['));
	});

	test("with no tools the protocol is left out", () => {
		const p = buildClaudeCodePrompt(ctx({ systemPrompt: "Plain.", messages: [{ role: "user", content: "hi", timestamp: 1 }] }));
		assert.equal(p.system, "Plain.");
	});

	test("the conversation renders in order: user, assistant text + tool call, tool result", () => {
		const assistant: AssistantMessage = {
			role: "assistant",
			content: [
				{ type: "text", text: "Let me check." },
				{ type: "toolCall", id: "call_1", name: "ask_user_question", arguments: { question: "What is MPC?" } },
			],
			api: "x",
			provider: "x",
			model: "x",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			stopReason: "toolUse",
			timestamp: 2,
		};
		const p = buildClaudeCodePrompt(
			ctx({
				systemPrompt: "S",
				tools: [quizTool],
				messages: [
					{ role: "user", content: "teach me MPC", timestamp: 1 },
					assistant,
					{
						role: "toolResult",
						toolCallId: "call_1",
						toolName: "ask_user_question",
						content: [{ type: "text", text: "A receding-horizon controller." }],
						isError: false,
						timestamp: 3,
					},
				],
			}),
		);
		assert.equal(p.content.length, 1);
		const text = (p.content[0] as { text: string }).text;
		const u = text.indexOf('<turn role="user">\nteach me MPC');
		const a = text.indexOf('<turn role="assistant">\nLet me check.');
		const call = text.indexOf('<tool_call>{"name":"ask_user_question","arguments":{"question":"What is MPC?"}}</tool_call>');
		const r = text.indexOf('<turn role="tool_result" tool="ask_user_question" call_id="call_1">\nA receding-horizon controller.');
		assert.ok(u >= 0 && a > u && call > a && r > call, text);
		assert.ok(text.endsWith("Write the assistant's next message now."));
	});

	test("images ride inline as image blocks between text blocks", () => {
		const p = buildClaudeCodePrompt(
			ctx({
				systemPrompt: "S",
				messages: [
					{
						role: "user",
						content: [
							{ type: "text", text: "look:" },
							{ type: "image", mimeType: "image/png", data: "AAAA" },
							{ type: "text", text: "what is it?" },
						],
						timestamp: 1,
					},
				],
			}),
		);
		assert.deepEqual(
			p.content.map((b) => b.type),
			["text", "image", "text"],
		);
		assert.deepEqual(p.content[1], { type: "image", source: { type: "base64", media_type: "image/png", data: "AAAA" } });
	});
});

describe("parseClaudeCodeReply", () => {
	test("plain text has no tool calls", () => {
		assert.deepEqual(parseClaudeCodeReply("  Hello there.  "), { text: "Hello there.", toolCalls: [] });
	});

	test("text then two tool calls, in order", () => {
		const r = parseClaudeCodeReply(
			'Here goes.\n<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n<tool_call>{"name":"b","arguments":{}}</tool_call>',
		);
		assert.equal(r.text, "Here goes.");
		assert.deepEqual(r.toolCalls, [
			{ name: "a", arguments: { x: 1 } },
			{ name: "b", arguments: {} },
		]);
	});

	test("a fenced JSON body still parses", () => {
		const r = parseClaudeCodeReply('<tool_call>\n```json\n{"name":"a","arguments":{"q":"?"}}\n```\n</tool_call>');
		assert.deepEqual(r.toolCalls, [{ name: "a", arguments: { q: "?" } }]);
	});

	test("an unclosed final block (cut-off reply) still parses as a call", () => {
		const r = parseClaudeCodeReply('Text <tool_call>{"name":"a","arguments":{"k":"v"}}');
		assert.equal(r.text, "Text");
		assert.deepEqual(r.toolCalls, [{ name: "a", arguments: { k: "v" } }]);
	});

	test("broken JSON becomes a call with argumentsError, so the loop reports it back to the model", () => {
		const r = parseClaudeCodeReply('<tool_call>{"name": "a", "arguments": {oops}}</tool_call>');
		assert.equal(r.toolCalls.length, 1);
		assert.equal(r.toolCalls[0]?.name, "a");
		assert.deepEqual(r.toolCalls[0]?.arguments, {});
		assert.match(r.toolCalls[0]?.argumentsError ?? "", /not valid JSON/);
	});
});

describe("visibleStreamText", () => {
	test("holds back an open block and a partial opening tag", () => {
		assert.equal(visibleStreamText("Hello <tool"), "Hello ");
		assert.equal(visibleStreamText('Hello <tool_call>{"name":'), "Hello ");
		assert.equal(visibleStreamText('Hi <tool_call>{}</tool_call> after'), "Hi  after");
		assert.equal(visibleStreamText("a < b"), "a < b");
	});
});

describe("catalog", () => {
	test("the claude-code provider is registered keyless with sonnet, opus and haiku", async () => {
		assert.deepEqual(
			claudeCodeModels().map((m) => m.id),
			["sonnet", "opus", "haiku"],
		);
		const m = piModels().getModel("claude-code", "sonnet");
		assert.ok(m);
		assert.ok(m.input.includes("image"));
		assert.ok(await piModels().getAuth("claude-code"));
	});

	test("the CLI runs with its own tools, settings, MCP and sessions off", () => {
		const args = claudeArgs("sonnet", "/tmp/s.md");
		assert.deepEqual(args.slice(0, 5), ["-p", "--model", "sonnet", "--system-prompt-file", "/tmp/s.md"]);
		assert.equal(args[args.indexOf("--tools") + 1], "");
		assert.equal(args[args.indexOf("--setting-sources") + 1], "");
		for (const f of ["--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"]) assert.ok(args.includes(f), f);
		assert.ok(!args.includes("--effort"));
	});

	test("the app's thinking level becomes the CLI's --effort", () => {
		const args = claudeArgs("opus", "/tmp/s.md", "high");
		assert.equal(args[args.indexOf("--effort") + 1], "high");
		const model = piModels().getModel("claude-code", "opus")!;
		assert.deepEqual(getSupportedThinkingLevels(model), ["low", "medium", "high", "xhigh", "max"]);
	});
});

// ─── one whole run against a fake claude ───────────────────────────────────

function fakeClaude(reply: string, opts: { isError?: boolean } = {}): { bin: string; dir: string } {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lh-fake-claude-"));
	const bin = path.join(dir, "claude");
	const deltas = reply.match(/[\s\S]{1,7}/g) ?? [];
	const lines = [
		{ type: "system", subtype: "init", tools: [] },
		...deltas.map((t) => ({ type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: t } } })),
		{
			type: "result",
			is_error: opts.isError ?? false,
			result: reply,
			usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 100, cache_creation_input_tokens: 20 },
		},
	];
	fs.writeFileSync(
		bin,
		`#!/usr/bin/env node
const fs = require("fs");
let input = "";
process.stdin.on("data", (c) => (input += c));
process.stdin.on("end", () => {
	const sysFile = process.argv[process.argv.indexOf("--system-prompt-file") + 1];
	fs.writeFileSync(${JSON.stringify(path.join(dir, "seen.json"))}, JSON.stringify({ argv: process.argv.slice(2), stdin: input, system: fs.readFileSync(sysFile, "utf8"), cwd: process.cwd() }));
	for (const l of ${JSON.stringify(lines)}) process.stdout.write(JSON.stringify(l) + "\\n");
});
`,
	);
	fs.chmodSync(bin, 0o755);
	return { bin, dir };
}

async function runFake(reply: string, opts: { isError?: boolean } = {}) {
	const { bin, dir } = fakeClaude(reply, opts);
	const prev = process.env.LH_CLAUDE_BIN;
	process.env.LH_CLAUDE_BIN = bin;
	try {
		const model = piModels().getModel("claude-code", "haiku")!;
		const s = streamClaudeCode(
			model,
			ctx({ systemPrompt: "You are the prober.", tools: [quizTool], messages: [{ role: "user", content: "hi", timestamp: 1 }] }),
		);
		const events: string[] = [];
		const deltas: string[] = [];
		for await (const ev of s) {
			events.push(ev.type);
			if (ev.type === "text_delta") deltas.push(ev.delta);
		}
		const message = await s.result();
		const seen = JSON.parse(fs.readFileSync(path.join(dir, "seen.json"), "utf8"));
		return { events, deltas, message, seen };
	} finally {
		if (prev === undefined) delete process.env.LH_CLAUDE_BIN;
		else process.env.LH_CLAUDE_BIN = prev;
		fs.rmSync(dir, { recursive: true, force: true });
	}
}

describe("streamClaudeCode (fake claude)", () => {
	test("a tool-call reply streams the text, ends with toolUse, and never shows the raw tag", async () => {
		const { events, deltas, message, seen } = await runFake(
			'First, a question.\n<tool_call>{"name":"ask_user_question","arguments":{"question":"What is a terminal cost?"}}</tool_call>',
		);
		assert.equal(message.stopReason, "toolUse");
		assert.equal(message.content[0]?.type, "text");
		assert.equal((message.content[0] as { text: string }).text, "First, a question.");
		const call = message.content[1];
		assert.ok(call && call.type === "toolCall");
		assert.equal(call.name, "ask_user_question");
		assert.deepEqual(call.arguments, { question: "What is a terminal cost?" });
		assert.match(call.id, /^toolu_cc_/);
		assert.ok(!deltas.join("").includes("<tool"), deltas.join(""));
		assert.deepEqual(events.slice(0, 2), ["start", "text_start"]);
		assert.ok(events.includes("toolcall_end"));
		assert.equal(events.at(-1), "done");
		// usage passes through; cost stays zero (subscription)
		assert.equal(message.usage.input, 10);
		assert.equal(message.usage.cacheRead, 100);
		assert.equal(message.usage.cost.total, 0);
		// what the CLI was given
		assert.ok(seen.argv.includes("--no-session-persistence"));
		assert.ok(seen.system.includes("You are the prober."));
		const sent = JSON.parse(seen.stdin.trim());
		assert.equal(sent.type, "user");
		assert.ok(sent.message.content[0].text.includes('<turn role="user">\nhi'));
		assert.notEqual(path.resolve(seen.cwd), path.resolve(process.cwd()), "runs in a scratch dir, not the app's");
	});

	test("a plain reply ends with stop", async () => {
		const { message } = await runFake("Just text.");
		assert.equal(message.stopReason, "stop");
		assert.deepEqual(message.content, [{ type: "text", text: "Just text." }]);
	});

	test("a CLI error becomes an error message, never a throw", async () => {
		const { message, events } = await runFake("usage limit reached", { isError: true });
		assert.equal(message.stopReason, "error");
		assert.match(message.errorMessage ?? "", /usage limit reached/);
		assert.equal(events.at(-1), "error");
	});

	test("a missing binary becomes an error message", async () => {
		const prev = process.env.LH_CLAUDE_BIN;
		process.env.LH_CLAUDE_BIN = "/nonexistent/claude-binary";
		try {
			const s = streamClaudeCode(piModels().getModel("claude-code", "haiku")!, ctx({ messages: [{ role: "user", content: "hi", timestamp: 1 }] }));
			const message = await s.result();
			assert.equal(message.stopReason, "error");
			assert.match(message.errorMessage ?? "", /could not run the Claude Code CLI/);
		} finally {
			if (prev === undefined) delete process.env.LH_CLAUDE_BIN;
			else process.env.LH_CLAUDE_BIN = prev;
		}
	});
});

// ─── the zero-keys gates with a keyless provider (fork) ─────────────────────

import { hasKeylessProvider } from "../components/setup-cta.tsx";
import type { ModelCatalogResponse } from "../src/core/index.ts";

describe("hasKeylessProvider", () => {
	const cat = (providers: { id: string; envVar: string; keyPresent: boolean }[]) =>
		({ models: [], defaults: {}, rolePins: {}, slots: {}, providers }) as unknown as ModelCatalogResponse;
	test("the claude-code provider (no envVar, always present) counts", () => {
		assert.equal(hasKeylessProvider(cat([{ id: "claude-code", envVar: "", keyPresent: true }])), true);
	});
	test("keyed providers never count, present or not; a loading catalog never counts", () => {
		assert.equal(hasKeylessProvider(cat([{ id: "openrouter", envVar: "OPENROUTER_API_KEY", keyPresent: true }])), false);
		assert.equal(hasKeylessProvider(cat([{ id: "anthropic", envVar: "ANTHROPIC_API_KEY", keyPresent: false }])), false);
		assert.equal(hasKeylessProvider(undefined), false);
	});
});
