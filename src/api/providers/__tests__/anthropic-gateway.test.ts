/**
 * The Anthropic provider pointed at a Messages-API gateway (Atlas Cloud, the
 * endpoint Claude Code reaches through ANTHROPIC_BASE_URL). Anthropic's own
 * host must keep sending exactly what it sent before; a gateway gets the key
 * as a Bearer token too and lists its own models.
 *
 * Run: npx vitest run src/api/providers/__tests__/anthropic-gateway.test.ts
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { AnthropicProvider } from '../anthropic-provider'
import { PROVIDER_PRESETS } from '../types'
import type { ProviderConfig } from '../types'

function config(over: Partial<ProviderConfig> = {}): ProviderConfig {
  return {
    id: 'anthropic', name: 'Atlas Cloud', enabled: true,
    baseUrl: 'https://api.atlascloud.ai', apiKey: 'atlas-test-key', isLocal: false,
    ...over,
  }
}

function stubFetch(respond: (url: string) => any) {
  const calls: { url: string; init: any }[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: any, init: any) => {
    calls.push({ url: String(url), init })
    return respond(String(url))
  }))
  return calls
}

const okMessage = {
  ok: true, status: 200,
  json: async () => ({ content: [{ type: 'text', text: 'ok' }], usage: { input_tokens: 1, output_tokens: 1 } }),
}

afterEach(() => vi.unstubAllGlobals())

describe('Atlas Cloud preset', () => {
  it('rides the anthropic slot on the gateway root, as cloud with a key', () => {
    const preset = PROVIDER_PRESETS.find(p => p.id === 'atlascloud')
    expect(preset).toBeDefined()
    expect(preset!.providerId).toBe('anthropic')
    expect(preset!.baseUrl).toBe('https://api.atlascloud.ai')
    expect(preset!.isLocal).toBe(false)
    expect(preset!.placeholder).toBeDefined()
  })
})

describe('AnthropicProvider against a gateway', () => {
  it('posts to <base>/v1/messages with the key as x-api-key and Bearer', async () => {
    const calls = stubFetch(() => okMessage)
    await new AnthropicProvider(config()).chatWithTools(
      'deepseek-ai/deepseek-v4-pro', [{ role: 'user', content: 'hi' }], [],
    )
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe('https://api.atlascloud.ai/v1/messages')
    expect(calls[0].init.headers['x-api-key']).toBe('atlas-test-key')
    expect(calls[0].init.headers['Authorization']).toBe('Bearer atlas-test-key')
    expect(JSON.parse(calls[0].init.body).model).toBe('deepseek-ai/deepseek-v4-pro')
  })

  it('leaves Anthropic itself on x-api-key only', async () => {
    const calls = stubFetch(() => okMessage)
    await new AnthropicProvider(config({ name: 'Anthropic', baseUrl: 'https://api.anthropic.com', apiKey: 'sk-ant-x' }))
      .chatWithTools('claude-haiku-4-5-20251001', [{ role: 'user', content: 'hi' }], [])
    expect(calls[0].url).toBe('https://api.anthropic.com/v1/messages')
    expect(calls[0].init.headers['x-api-key']).toBe('sk-ant-x')
    expect(calls[0].init.headers['Authorization']).toBeUndefined()
  })

  it("lists the gateway's own models under its name", async () => {
    const calls = stubFetch(() => ({
      ok: true, status: 200,
      json: async () => ({ data: [{ id: 'deepseek-ai/deepseek-v4-pro' }, { id: 'zai-org/glm-5.1', display_name: 'GLM 5.1' }, { id: '' }] }),
    }))
    const models = await new AnthropicProvider(config()).listModels()
    expect(calls[0].url).toBe('https://api.atlascloud.ai/v1/models')
    expect(calls[0].init.headers['Authorization']).toBe('Bearer atlas-test-key')
    expect(models.map(m => m.id)).toEqual(['deepseek-ai/deepseek-v4-pro', 'zai-org/glm-5.1'])
    expect(models[1].name).toBe('GLM 5.1')
    expect(models.every(m => m.provider === 'anthropic' && m.providerName === 'Atlas Cloud')).toBe(true)
  })

  it('takes a base that already ends in /v1', async () => {
    const calls = stubFetch(() => ({ ok: true, status: 200, json: async () => ({ data: [{ id: 'm' }] }) }))
    await new AnthropicProvider(config({ baseUrl: 'https://gw.example/v1/' })).listModels()
    expect(calls[0].url).toBe('https://gw.example/v1/models')
  })

  it('falls back to the Claude list when the gateway has no models endpoint', async () => {
    stubFetch(() => ({ ok: false, status: 404, json: async () => ({}) }))
    const models = await new AnthropicProvider(config()).listModels()
    expect(models.length).toBeGreaterThan(0)
    expect(models.every(m => m.id.startsWith('claude-'))).toBe(true)
  })

  it('never asks Anthropic for a model list', async () => {
    const calls = stubFetch(() => { throw new Error('no network expected') })
    const models = await new AnthropicProvider(config({ baseUrl: 'https://api.anthropic.com' })).listModels()
    expect(calls).toHaveLength(0)
    expect(models.every(m => m.id.startsWith('claude-'))).toBe(true)
  })
})
