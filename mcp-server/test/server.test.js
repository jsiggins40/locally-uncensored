// End-to-end tests: a real MCP client talks to the server over an in-memory
// transport, and the server talks over real HTTP to fake backends.
// Run with `npm test` (node:test, no extra dependencies).

import { test, before, after } from 'node:test'
import assert from 'node:assert/strict'
import http from 'node:http'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js'
import { createServer, defaultSize } from '../src/server.js'
import { parseBackends } from '../src/config.js'
import { readComboOptions } from '../src/comfyui.js'

// 1x1 transparent PNG
const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
  'base64',
)

let fake
let baseUrl
const seen = { chat: [], prompts: [], historyPolls: 0 }

function json(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(body))
}

before(async () => {
  fake = http.createServer(async (req, res) => {
    let body = ''
    for await (const chunk of req) body += chunk
    const url = new URL(req.url, 'http://x')
    // OpenAI compatible chat backend
    if (url.pathname === '/v1/models') return json(res, 200, { data: [{ id: 'qwen-test' }, { id: 'llama-test' }] })
    if (url.pathname === '/v1/chat/completions') {
      const payload = JSON.parse(body)
      seen.chat.push(payload)
      return json(res, 200, {
        choices: [{ message: { role: 'assistant', content: `echo: ${payload.messages.at(-1).content}` }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 3, completion_tokens: 2 },
      })
    }
    // ComfyUI
    if (url.pathname === '/system_stats') {
      return json(res, 200, { system: { comfyui_version: '0.3.99' }, devices: [{ name: 'cuda:0 Fake', vram_total: 12 * 1024 ** 3, vram_free: 10 * 1024 ** 3 }] })
    }
    if (url.pathname === '/object_info/CheckpointLoaderSimple') {
      return json(res, 200, { CheckpointLoaderSimple: { input: { required: { ckpt_name: [['sdxl_base.safetensors', 'sd15.safetensors'], {}] } } } })
    }
    if (url.pathname === '/object_info/KSampler') {
      return json(res, 200, { KSampler: { input: { required: { sampler_name: ['COMBO', { options: ['euler', 'dpmpp_2m'] }], scheduler: [['normal', 'karras']] } } } })
    }
    if (url.pathname.startsWith('/object_info/')) return json(res, 404, {})
    if (url.pathname === '/prompt') {
      seen.prompts.push(JSON.parse(body))
      return json(res, 200, { prompt_id: 'job-1', number: 1, node_errors: {} })
    }
    if (url.pathname === '/history/job-1') {
      seen.historyPolls++
      if (seen.historyPolls === 1) return json(res, 200, {})
      return json(res, 200, {
        'job-1': {
          status: { status_str: 'success', completed: true, messages: [] },
          outputs: { 9: { images: [{ filename: 'LU_MCP_00001_.png', subfolder: '', type: 'output' }] } },
        },
      })
    }
    if (url.pathname === '/history/job-bad') {
      return json(res, 200, {
        'job-bad': {
          status: { status_str: 'error', completed: false, messages: [['execution_error', { node_type: 'KSampler', exception_message: 'CUDA out of memory' }]] },
          outputs: {},
        },
      })
    }
    if (url.pathname === '/view') {
      res.writeHead(200, { 'Content-Type': 'image/png' })
      return res.end(PNG)
    }
    json(res, 404, { error: 'not found' })
  })
  await new Promise((r) => fake.listen(0, '127.0.0.1', r))
  baseUrl = `http://127.0.0.1:${fake.address().port}`
})

after(() => fake.close())

async function connect(overrides = {}) {
  const config = {
    backends: [
      { id: 'fake', name: 'Fake', baseUrl: `${baseUrl}/v1` },
      { id: 'dead', name: 'Dead', baseUrl: 'http://127.0.0.1:9/v1' },
    ],
    comfyUrl: baseUrl,
    probeTimeoutMs: 1000,
    timeoutMs: 10_000,
    maxInlineImageBytes: 1_000_000,
    ...overrides,
  }
  const [clientT, serverT] = InMemoryTransport.createLinkedPair()
  const server = createServer(config)
  await server.connect(serverT)
  const client = new Client({ name: 'test', version: '0' })
  await client.connect(clientT)
  return client
}

const textOf = (result) => result.content.filter((c) => c.type === 'text').map((c) => c.text).join('\n')

test('lists all tools', async () => {
  const client = await connect()
  const { tools } = await client.listTools()
  assert.deepEqual(tools.map((t) => t.name).sort(), [
    'comfyui_generate_image',
    'comfyui_get_result',
    'comfyui_list_models',
    'comfyui_run_workflow',
    'list_local_models',
    'local_chat',
    'lu_status',
  ])
})

test('lu_status reports online and offline backends and ComfyUI', async () => {
  const client = await connect()
  const result = await client.callTool({ name: 'lu_status', arguments: {} })
  const status = JSON.parse(textOf(result))
  assert.equal(status.backends.find((b) => b.id === 'fake').online, true)
  assert.deepEqual(status.backends.find((b) => b.id === 'fake').models, ['qwen-test', 'llama-test'])
  assert.equal(status.backends.find((b) => b.id === 'dead').online, false)
  assert.equal(status.comfyui.online, true)
  assert.equal(status.comfyui.version, '0.3.99')
})

test('local_chat picks the backend serving the model and sends system + history', async () => {
  const client = await connect()
  const result = await client.callTool({
    name: 'local_chat',
    arguments: { prompt: 'hi', system: 'be brief', model: 'llama-test', history: [{ role: 'user', content: 'earlier' }] },
  })
  assert.equal(result.isError, undefined)
  assert.match(textOf(result), /echo: hi/)
  const sent = seen.chat.at(-1)
  assert.equal(sent.model, 'llama-test')
  assert.deepEqual(sent.messages.map((m) => m.role), ['system', 'user', 'user'])
})

test('local_chat names the problem for an unknown model', async () => {
  const client = await connect()
  const result = await client.callTool({ name: 'local_chat', arguments: { prompt: 'hi', model: 'nope' } })
  assert.equal(result.isError, true)
  assert.match(textOf(result), /No online backend serves "nope"/)
})

test('local_chat reports an offline backend instead of hanging', async () => {
  const client = await connect()
  const result = await client.callTool({ name: 'local_chat', arguments: { prompt: 'hi', backend: 'dead' } })
  assert.equal(result.isError, true)
  assert.match(textOf(result), /offline/)
})

test('comfyui_list_models reads both dropdown shapes and survives missing nodes', async () => {
  const client = await connect()
  const models = JSON.parse(textOf(await client.callTool({ name: 'comfyui_list_models', arguments: {} })))
  assert.deepEqual(models.checkpoints, ['sdxl_base.safetensors', 'sd15.safetensors'])
  assert.deepEqual(models.samplers, ['euler', 'dpmpp_2m'])
  assert.deepEqual(models.schedulers, ['normal', 'karras'])
  assert.deepEqual(models.loras, [])
})

test('comfyui_generate_image submits a txt2img graph, waits, and returns the image', async () => {
  seen.historyPolls = 0
  const client = await connect()
  const result = await client.callTool({ name: 'comfyui_generate_image', arguments: { prompt: 'a red fox', seed: 42 } })
  assert.equal(result.isError, undefined, textOf(result))
  const image = result.content.find((c) => c.type === 'image')
  assert.equal(image.mimeType, 'image/png')
  assert.equal(Buffer.from(image.data, 'base64').length, PNG.length)
  const graph = seen.prompts.at(-1).prompt
  assert.equal(graph['4'].inputs.ckpt_name, 'sdxl_base.safetensors')
  assert.equal(graph['5'].inputs.width, 1024)
  assert.equal(graph['3'].inputs.seed, 42)
  assert.equal(graph['6'].inputs.text, 'a red fox')
  assert.ok(seen.historyPolls >= 2, 'polled until the job finished')
})

test('comfyui_generate_image with wait=false returns the prompt id, get_result fetches it', async () => {
  seen.historyPolls = 1 // next poll answers "done"
  const client = await connect()
  const queued = JSON.parse(textOf(await client.callTool({ name: 'comfyui_generate_image', arguments: { prompt: 'x', wait: false } })))
  assert.equal(queued.prompt_id, 'job-1')
  const done = await client.callTool({ name: 'comfyui_get_result', arguments: { prompt_id: 'job-1' } })
  assert.ok(done.content.some((c) => c.type === 'image'))
})

test('oversized images are linked, not inlined', async () => {
  seen.historyPolls = 1
  const client = await connect({ maxInlineImageBytes: 10 })
  const result = await client.callTool({ name: 'comfyui_get_result', arguments: { prompt_id: 'job-1' } })
  assert.equal(result.content.some((c) => c.type === 'image'), false)
  assert.match(textOf(result), /\/view\?filename=LU_MCP_00001_.png/)
})

test('a failed ComfyUI run surfaces the node error', async () => {
  const client = await connect()
  const result = await client.callTool({ name: 'comfyui_get_result', arguments: { prompt_id: 'job-bad' } })
  assert.equal(result.isError, true)
  assert.match(textOf(result), /KSampler: CUDA out of memory/)
})

test('helpers', () => {
  assert.equal(defaultSize('ponyDiffusionV6XL.safetensors'), 1024)
  assert.equal(defaultSize('Realistic_Vision_V6.0.safetensors'), 512)
  assert.deepEqual(parseBackends('a=http://h:1/v1/, b=http://h:2/v1'), [
    { id: 'a', name: 'a', baseUrl: 'http://h:1/v1' },
    { id: 'b', name: 'b', baseUrl: 'http://h:2/v1' },
  ])
  assert.equal(parseBackends(''), null)
  assert.throws(() => parseBackends('nourl'), /id=url/)
  assert.equal(readComboOptions(['INT', {}]), null)
  assert.deepEqual(readComboOptions(['COMBO', {}]), [])
})

test('loadConfig treats empty values and unfilled .mcpb placeholders as defaults', async () => {
  const { loadConfig, DEFAULT_BACKENDS } = await import('../src/config.js')
  const config = loadConfig({ LU_BACKENDS: '${user_config.backends}', LU_COMFYUI_URL: '', LU_TIMEOUT_MS: '${user_config.timeout_ms}' })
  assert.equal(config.backends, DEFAULT_BACKENDS)
  assert.equal(config.comfyUrl, 'http://127.0.0.1:8188')
  assert.equal(config.timeoutMs, 300_000)
  assert.equal(loadConfig({ LU_COMFYUI_URL: 'http://pc:8189/' }).comfyUrl, 'http://pc:8189')
})
