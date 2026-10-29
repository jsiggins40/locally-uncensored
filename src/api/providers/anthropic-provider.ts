/**
 * Anthropic Provider — Claude API
 *
 * Uses the Anthropic Messages API which has a different format from OpenAI:
 * - System prompt is a separate `system` param, not a message
 * - SSE events use `event: content_block_delta` format
 * - Tool calling uses `tool_use` content blocks
 * - Anthropic itself: hardcoded model list. A compatible gateway (Atlas Cloud,
 *   DeepSeek /anthropic, ...): its own /v1/models, Claude list as fallback
 */

import type {
  ProviderClient, ProviderModel, ProviderConfig, ChatMessage, ChatOptions,
  ChatStreamChunk, ToolCall, ToolDefinition,
} from './types'
import { ProviderError } from './types'
import { parseSSEWithEvents } from '../sse'
import { localFetch, localFetchStream, isDirectFetchAllowed, hostnameOf, ensureProxyAllowsHost } from '../backend'

// ── Anthropic API Types ────────────────────────────────────────

interface AnthropicStreamEvent {
  type: string
  index?: number
  content_block?: { type: string; id?: string; name?: string; input?: any; text?: string }
  delta?: { type: string; text?: string; partial_json?: string; thinking?: string }
  message?: { id: string; usage?: { input_tokens: number; output_tokens: number } }
  // message_delta events carry cumulative output usage at the event top level.
  usage?: { input_tokens?: number; output_tokens?: number }
}

interface AnthropicResponse {
  content: {
    type: 'text' | 'tool_use'
    text?: string
    id?: string
    name?: string
    input?: Record<string, any>
  }[]
  stop_reason?: string
  usage?: { input_tokens: number; output_tokens: number }
}

// ── Known Claude models ────────────────────────────────────────

const CLAUDE_MODELS: ProviderModel[] = [
  { id: 'claude-opus-4-20250514', name: 'Claude Opus 4', provider: 'anthropic', providerName: 'Anthropic', contextLength: 200000, supportsTools: true, supportsVision: true },
  { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4', provider: 'anthropic', providerName: 'Anthropic', contextLength: 200000, supportsTools: true, supportsVision: true },
  { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', provider: 'anthropic', providerName: 'Anthropic', contextLength: 200000, supportsTools: true, supportsVision: true },
]

// ── Prompt caching (2.6.6 A8) ──────────────────────────────────
//
// Prompt caching is GA on `anthropic-version: 2023-06-01`, the version this
// provider already pins. The old opt-in `anthropic-beta:
// prompt-caching-2024-07-31` is no longer required and is NOT sent: an
// unknown beta value is a needless failure surface on the proxies people
// front this provider with (LiteLLM, claude-relay-server, opencode-zen).
//
// The API allows at most 4 breakpoints per request. We place at most 3, in
// render order (tools → system → messages):
//   1. the last tool definition  → caches [tools]
//   2. the system block          → caches [tools + system]
//   3. the last STABLE message   → caches [tools + system + settled history]
// Layered on purpose: when the system prompt moves, the tools prefix still
// reads from cache.
//
// "Stable" means the youngest message the NEXT request will send unchanged,
// so the current turn's message is deliberately skipped. Marking it would
// write a fresh entry on every step and never read one back, because A1
// decay and A3 compaction still rewrite the tail of the history.

const CACHE_CONTROL = { type: 'ephemeral' } as const

/** A message content as blocks, so a marker has something to ride on. */
function asContentBlocks(content: string | Record<string, any>[]): Record<string, any>[] {
  return typeof content === 'string' ? [{ type: 'text', text: content }] : content
}

/**
 * Stamp the three ephemeral breakpoints onto a finished request body. Called
 * from both request paths (chatStream and chatWithTools) after the body is
 * fully built, so vision, tool and plain-text variants all carry the markers.
 */
function applyCacheControl(body: Record<string, any>): void {
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    body.tools[body.tools.length - 1].cache_control = { ...CACHE_CONTROL }
  }

  if (typeof body.system === 'string' && body.system) {
    body.system = [{ type: 'text', text: body.system, cache_control: { ...CACHE_CONTROL } }]
  }

  const messages: Record<string, any>[] = body.messages
  if (messages.length >= 2) {
    const stable = messages[messages.length - 2]
    stable.content = asContentBlocks(stable.content)
    const lastBlock = stable.content[stable.content.length - 1]
    if (lastBlock) lastBlock.cache_control = { ...CACHE_CONTROL }
  }
}

// ── Provider Implementation ────────────────────────────────────

export class AnthropicProvider implements ProviderClient {
  readonly id = 'anthropic' as const

  constructor(private config: ProviderConfig) {}

  private get baseUrl(): string {
    return this.config.baseUrl.replace(/\/+$/, '')
  }

  /**
   * Bug O — v2.4.7. When users point the Anthropic provider at a proxy
   * (claude-relay-server, LiteLLM, opencode-zen, etc.) they sometimes
   * configure the baseUrl with `/v1` already included. Pre-v2.4.7 we always
   * appended `/v1/messages`, producing `https://proxy.example/v1/v1/messages`
   * which 404s silently. Strip a trailing `/v1` so users can paste whichever
   * shape their proxy docs use.
   */
  private messagesUrl(): string {
    const base = this.baseUrl
    if (/\/v1$/i.test(base)) {
      return `${base}/messages`
    }
    return `${base}/v1/messages`
  }

  /** True when the endpoint is Anthropic's own API, not a compatible gateway. */
  private get isAnthropicHost(): boolean {
    return hostnameOf(this.baseUrl) === 'api.anthropic.com'
  }

  /**
   * Gateways that speak the Messages API (Atlas Cloud, DeepSeek's /anthropic,
   * opencode-zen, claude-relay-server) are public hosts outside the pinned CSP
   * in src-tauri/tauri.conf.json, so a direct webview fetch to them is killed
   * before it reaches the network. Same rule as the OpenAI provider: anything
   * the CSP does not list takes the Rust proxy, which also sidesteps CORS.
   */
  private get useLocalProxy(): boolean {
    return !isDirectFetchAllowed(hostnameOf(this.baseUrl))
  }

  private get headers(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-api-key': this.config.apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    }
    // Gateways document their key as Claude Code's ANTHROPIC_AUTH_TOKEN, which
    // goes out as a Bearer token, and some accept only that. Anthropic itself
    // reads x-api-key, so its requests stay exactly as they were.
    if (!this.isAnthropicHost && this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`
    }
    return headers
  }

  /** POST to the Messages endpoint over whichever transport this host needs. */
  private async postMessages(body: Record<string, any>, signal?: AbortSignal, stream = false): Promise<Response> {
    const init = { method: 'POST', headers: this.headers, body: JSON.stringify(body), signal }
    if (!this.useLocalProxy) return fetch(this.messagesUrl(), init)
    await ensureProxyAllowsHost(this.baseUrl)
    return (stream ? localFetchStream : localFetch)(this.messagesUrl(), init)
  }

  async *chatStream(
    model: string,
    messages: ChatMessage[],
    options?: ChatOptions,
  ): AsyncGenerator<ChatStreamChunk> {
    const { system, anthropicMessages } = this.convertMessages(messages)

    const body: Record<string, any> = {
      model,
      messages: anthropicMessages,
      max_tokens: options?.maxTokens && options.maxTokens > 0 ? options.maxTokens : 4096,
      stream: true,
    }

    if (system) body.system = system
    if (options?.temperature !== undefined) body.temperature = options.temperature
    if (options?.topP !== undefined) body.top_p = options.topP
    if (options?.topK !== undefined) body.top_k = options.topK
    // Claude Extended Thinking (Sonnet 3.7+, Opus 4). Opt-in: only when the
    // user actually toggled Thinking ON. Default stays OFF, so toggle OFF
    // simply omits the field. 5000 tokens is a sensible default budget —
    // the model may produce less, but won't exceed it.
    if (options?.thinking === true) {
      body.thinking = { type: 'enabled', budget_tokens: 5000 }
    }

    // Streaming tool turn (same conversion as chatWithTools below). The
    // stream parser already accumulates tool_use blocks via input_json_delta
    // and flushes them into the done-chunk's toolCalls.
    if (options?.tools?.length) {
      body.tools = options.tools.map(t => ({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters,
      }))
    }

    applyCacheControl(body)

    let res = await this.postMessages(body, options?.signal, true)

    // Retry without extended thinking if the model rejects it (e.g. older
    // Claude versions don't support `thinking`).
    if (!res.ok && res.status === 400 && 'thinking' in body) {
      delete body.thinking
      res = await this.postMessages(body, options?.signal, true)
    }

    if (!res.ok) {
      throw await this.parseError(res)
    }

    // Track tool use blocks being built
    const toolUseBlocks: Map<number, { id: string; name: string; input: string }> = new Map()
    // Real token usage: message_start carries input_tokens, message_delta the
    // cumulative output_tokens. Surfaced on the done chunk exactly like the
    // OpenAI provider so the TokenCounter gets a real anchor on every backend.
    let inputTokens = 0
    let outputTokens = 0

    for await (const { data } of parseSSEWithEvents<AnthropicStreamEvent>(res)) {
      if (options?.signal?.aborted) break

      switch (data.type) {
        case 'message_start': {
          const u = data.message?.usage
          if (u?.input_tokens) inputTokens = u.input_tokens
          break
        }

        case 'content_block_start': {
          if (data.content_block?.type === 'tool_use') {
            toolUseBlocks.set(data.index!, {
              id: data.content_block.id || '',
              name: data.content_block.name || '',
              input: '',
            })
          }
          break
        }

        case 'content_block_delta': {
          const dtype = (data.delta as any)?.type
          if (dtype === 'text_delta' && (data.delta as any).text) {
            yield { content: (data.delta as any).text, done: false }
          } else if (dtype === 'thinking_delta' && (data.delta as any).thinking) {
            // Claude Extended Thinking stream — route to `thinking` so the
            // ThinkingBlock UI picks it up (same field as Ollama's native).
            yield { content: '', thinking: (data.delta as any).thinking, done: false }
          } else if (dtype === 'input_json_delta' && (data.delta as any).partial_json) {
            const block = toolUseBlocks.get(data.index!)
            if (block) block.input += (data.delta as any).partial_json
          }
          break
        }

        case 'message_delta': {
          // End of message — flush tool calls. Map Anthropic's stop_reason
          // onto the unified finishReason ('max_tokens' → 'length', the key
          // the chat layer uses to explain thought-only/truncated turns).
          const stopReason = (data.delta as any)?.stop_reason as string | undefined
          if (data.usage?.output_tokens) outputTokens = data.usage.output_tokens
          const toolCalls = this.flushToolUseBlocks(toolUseBlocks)
          yield {
            content: '',
            toolCalls: toolCalls.length ? toolCalls : undefined,
            done: true,
            finishReason: stopReason === 'max_tokens' ? 'length' : (stopReason || 'stop'),
            promptEvalCount: inputTokens || undefined,
            evalCount: outputTokens || undefined,
          }
          return
        }

        case 'message_stop': {
          const toolCalls2 = this.flushToolUseBlocks(toolUseBlocks)
          yield {
            content: '', toolCalls: toolCalls2.length ? toolCalls2 : undefined, done: true, finishReason: 'stop',
            promptEvalCount: inputTokens || undefined, evalCount: outputTokens || undefined,
          }
          return
        }
      }
    }

    // Stream ended without an explicit message_delta/message_stop — the
    // connection was cut mid-generation (same semantics as the OpenAI
    // provider's missing-[DONE] path).
    const toolCalls = this.flushToolUseBlocks(toolUseBlocks)
    yield {
      content: '', toolCalls: toolCalls.length ? toolCalls : undefined, done: true, finishReason: 'disconnect',
      promptEvalCount: inputTokens || undefined, evalCount: outputTokens || undefined,
    }
  }

  async chatWithTools(
    model: string,
    messages: ChatMessage[],
    tools: ToolDefinition[],
    options?: ChatOptions,
  ): Promise<{ content: string; toolCalls: ToolCall[]; promptEvalCount?: number; evalCount?: number }> {
    const { system, anthropicMessages } = this.convertMessages(messages)

    const body: Record<string, any> = {
      model,
      messages: anthropicMessages,
      max_tokens: options?.maxTokens && options.maxTokens > 0 ? options.maxTokens : 4096,
    }

    if (system) body.system = system
    if (options?.temperature !== undefined) body.temperature = options.temperature
    if (options?.topP !== undefined) body.top_p = options.topP
    if (options?.topK !== undefined) body.top_k = options.topK
    // Same extended-thinking gate as chatStream.
    if (options?.thinking === true) {
      body.thinking = { type: 'enabled', budget_tokens: 5000 }
    }

    // Convert OpenAI tool format to Anthropic format
    if (tools.length > 0) {
      body.tools = tools.map(t => ({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters,
      }))
    }

    applyCacheControl(body)

    let res = await this.postMessages(body, options?.signal)

    // Retry without extended thinking if the model rejects it.
    if (!res.ok && res.status === 400 && 'thinking' in body) {
      delete body.thinking
      res = await this.postMessages(body, options?.signal)
    }

    if (!res.ok) {
      throw await this.parseError(res)
    }

    const data: AnthropicResponse = await res.json()

    let content = ''
    const toolCalls: ToolCall[] = []

    for (const block of data.content) {
      if (block.type === 'text') {
        content += block.text || ''
      } else if (block.type === 'tool_use') {
        toolCalls.push({
          id: block.id,
          function: {
            name: block.name!,
            arguments: (typeof block.input === 'object' && block.input) ? block.input : {},
          },
        })
      }
    }

    // Real usage (audit B7): without it the TokenCounter stayed on the
    // char/4 estimate for every Anthropic agent turn.
    return {
      content,
      toolCalls,
      promptEvalCount: data.usage?.input_tokens || undefined,
      evalCount: data.usage?.output_tokens || undefined,
    }
  }

  async listModels(): Promise<ProviderModel[]> {
    if (this.isAnthropicHost) return [...CLAUDE_MODELS]

    // A gateway serves its own catalogue (Atlas Cloud fronts DeepSeek, GLM,
    // Qwen, ...), so the Claude list would offer IDs it cannot route. Ask its
    // /v1/models; both the Anthropic and the OpenAI shape carry `data[].id`.
    // A gateway without the endpoint keeps the Claude list it had before.
    try {
      const url = this.messagesUrl().replace(/\/messages$/, '/models')
      if (this.useLocalProxy) await ensureProxyAllowsHost(this.baseUrl)
      const res = this.useLocalProxy
        ? await localFetch(url, { headers: this.headers })
        : await fetch(url, { headers: this.headers })
      if (!res.ok) return [...CLAUDE_MODELS]
      const data = await res.json()
      const entries: any[] = Array.isArray(data?.data) ? data.data : []
      const models: ProviderModel[] = entries
        .filter(m => typeof m?.id === 'string' && m.id)
        .map(m => ({
          id: m.id,
          name: m.display_name || m.name || m.id,
          provider: 'anthropic' as const,
          providerName: this.config.name || 'Anthropic',
          supportsTools: true,
        }))
      return models.length > 0 ? models : [...CLAUDE_MODELS]
    } catch {
      return [...CLAUDE_MODELS]
    }
  }

  async checkConnection(): Promise<boolean> {
    if (!this.config.apiKey) return false

    try {
      // Send a minimal request to verify the API key
      const res = await this.postMessages({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 1,
        messages: [{ role: 'user', content: 'hi' }],
      })
      // 200 or 400 (bad request but auth worked) both mean the key is valid
      return res.status !== 401 && res.status !== 403
    } catch {
      return false
    }
  }

  async getContextLength(model: string): Promise<number> {
    const known = CLAUDE_MODELS.find(m => model.includes(m.id.split('-').slice(0, 2).join('-')))
    return known?.contextLength || 200000
  }

  // ── Message conversion ───────────────────────────────────────

  private convertMessages(messages: ChatMessage[]): {
    system: string
    anthropicMessages: Record<string, any>[]
  } {
    let system = ''
    const anthropicMessages: Record<string, any>[] = []

    for (const msg of messages) {
      if (msg.role === 'system') {
        // Anthropic: system goes in a separate parameter
        system += (system ? '\n\n' : '') + msg.content
        continue
      }

      if (msg.role === 'tool') {
        // Anthropic tool results are user messages with tool_result content blocks
        anthropicMessages.push({
          role: 'user',
          content: [{
            type: 'tool_result',
            tool_use_id: msg.tool_call_id || 'unknown',
            content: msg.content,
          }],
        })
        continue
      }

      if (msg.role === 'assistant' && msg.tool_calls?.length) {
        // Assistant with tool calls → include tool_use content blocks
        const content: any[] = []
        if (msg.content) content.push({ type: 'text', text: msg.content })
        for (const tc of msg.tool_calls) {
          content.push({
            type: 'tool_use',
            id: tc.id || `toolu_${Math.random().toString(36).slice(2, 11)}`,
            name: tc.function.name,
            input: tc.function.arguments,
          })
        }
        anthropicMessages.push({ role: 'assistant', content })
        continue
      }

      // Regular user/assistant message — with optional images
      if (msg.images?.length && msg.role === 'user') {
        const content: any[] = []
        for (const img of msg.images) {
          content.push({
            type: 'image',
            source: { type: 'base64', media_type: img.mimeType, data: img.data },
          })
        }
        content.push({ type: 'text', text: msg.content })
        anthropicMessages.push({ role: 'user', content })
      } else {
        anthropicMessages.push({ role: msg.role, content: msg.content })
      }
    }

    // Anthropic requires messages to alternate user/assistant.
    // Merge consecutive same-role messages. Block-array contents merge too
    // (audit B7): the agent loop pushes one tool-result message PER call, and
    // Anthropic requires all tool_result blocks answering one assistant turn
    // to arrive in ONE user message. The old string-only merge left two
    // parallel tool calls as two consecutive user messages, which the API
    // rejects — so any multi-tool batch broke the whole Anthropic agent path.
    const asBlocks = (content: string | Record<string, any>[]): Record<string, any>[] =>
      typeof content === 'string' ? [{ type: 'text', text: content }] : content
    const merged: Record<string, any>[] = []
    for (const msg of anthropicMessages) {
      const last = merged[merged.length - 1]
      if (!last || last.role !== msg.role) {
        merged.push(msg)
        continue
      }
      if (typeof last.content === 'string' && typeof msg.content === 'string') {
        last.content += '\n\n' + msg.content
      } else {
        last.content = [...asBlocks(last.content), ...asBlocks(msg.content)]
      }
    }

    return { system, anthropicMessages: merged }
  }

  // ── Tool call helpers ────────────────────────────────────────

  private flushToolUseBlocks(blocks: Map<number, { id: string; name: string; input: string }>): ToolCall[] {
    if (blocks.size === 0) return []

    const calls: ToolCall[] = []
    for (const [, block] of blocks) {
      let args: Record<string, any> = {}
      try { args = JSON.parse(block.input) } catch { /* empty */ }

      calls.push({
        id: block.id,
        function: { name: block.name, arguments: args },
      })
    }
    blocks.clear()
    return calls
  }

  // ── Error parsing ────────────────────────────────────────────

  private async parseError(res: Response): Promise<ProviderError> {
    let message = 'Anthropic: Request failed'
    let code: string = 'network'

    try {
      const data = await res.json()
      if (data.error?.message) message = data.error.message
    } catch { /* use default */ }

    if (res.status === 401 || res.status === 403) {
      code = 'auth'
      message = 'Invalid Anthropic API key. Check Settings > Providers.'
    } else if (res.status === 429) {
      code = 'rate_limit'
      message = 'Rate limited by Anthropic. Wait a moment and try again.'
    } else if (res.status === 404) {
      code = 'not_found'
    } else if (res.status === 529) {
      code = 'overloaded'
      message = 'Anthropic API is overloaded. Try again in a few seconds.'
    }

    return new ProviderError(message, 'anthropic', code, res.status)
  }
}
