/**
 * Atlas Cloud image client (Seedream 5.0 Pro): reply parsing for both shapes
 * Atlas documents, the size fallback, and the key riding the chat preset.
 *
 * Run: npx vitest run src/api/__tests__/atlas-images.test.ts
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import {
  parseSubmitId,
  parsePrediction,
  explainAtlasStatus,
  submitAtlasImage,
  waitForAtlasImage,
  getAtlasApiKey,
  atlasModelForIntent,
  parseUploadUrl,
  uploadAtlasImage,
  ATLAS_IMAGE_MODELS,
} from '../atlas-images'
import { useProviderStore } from '../../stores/providerStore'

const reply = (status: number, body: unknown) => ({
  ok: status >= 200 && status < 300,
  status,
  text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
})

afterEach(() => vi.unstubAllGlobals())

describe('ATLAS_IMAGE_MODELS', () => {
  it('offers Seedream 5.0 Pro only', () => {
    expect(ATLAS_IMAGE_MODELS.map((m) => m.id)).toEqual(['bytedance/seedream-v5.0-pro/text-to-image'])
  })
})

describe('atlasModelForIntent', () => {
  it('maps one pick to its text-to-image and edit endpoints', () => {
    const pick = 'bytedance/seedream-v5.0-pro/text-to-image'
    expect(atlasModelForIntent(pick, 'image')).toBe(pick)
    expect(atlasModelForIntent(pick, 'edit')).toBe('bytedance/seedream-v5.0-pro/edit')
    expect(atlasModelForIntent(pick, 'video')).toBeNull()
    expect(atlasModelForIntent('something-else', 'image')).toBeNull()
    expect(atlasModelForIntent(null, 'image')).toBeNull()
  })
})

describe('parseUploadUrl', () => {
  it('reads the URL spellings an upload reply may use', () => {
    expect(parseUploadUrl({ data: { download_url: 'https://a/1.png' } })).toBe('https://a/1.png')
    expect(parseUploadUrl({ data: { url: 'https://a/2.png' } })).toBe('https://a/2.png')
    expect(parseUploadUrl({ url: 'https://a/3.png' })).toBe('https://a/3.png')
    expect(parseUploadUrl({ data: {} })).toBeNull()
  })
})

describe('uploadAtlasImage', () => {
  it('sends the decoded image as multipart field "file" and returns its URL', async () => {
    const calls: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      calls.push({ url, init })
      return reply(200, { data: { download_url: 'https://tmp.example/s.jpg' } })
    }))
    const url = await uploadAtlasImage('data:image/jpeg;base64,/9j/4AAQ', 'k')
    expect(url).toBe('https://tmp.example/s.jpg')
    const file = (calls[0].init.body as FormData).get('file') as File
    expect(file.name).toBe('source.jpg')
    expect(file.type).toBe('image/jpeg')
    expect(new Uint8Array(await file.arrayBuffer()).slice(0, 3)).toEqual(new Uint8Array([0xff, 0xd8, 0xff]))
  })

  it('refuses something that is not a base64 data URL', async () => {
    await expect(uploadAtlasImage('blob:abc', 'k')).rejects.toThrow(/could not be read/)
  })
})

describe('parseSubmitId', () => {
  it('reads data.id and the flat spellings', () => {
    expect(parseSubmitId({ code: 200, data: { id: 'p1' } })).toBe('p1')
    expect(parseSubmitId({ predictionId: 'p2' })).toBe('p2')
    expect(parseSubmitId({ data: {} })).toBeNull()
  })
})

describe('parsePrediction', () => {
  it('reads the data.status + data.outputs shape', () => {
    expect(parsePrediction({ data: { status: 'processing' } })).toEqual({ kind: 'pending' })
    expect(parsePrediction({ data: { status: 'completed', outputs: ['https://cdn.example/a.png'] } }))
      .toEqual({ kind: 'done', url: 'https://cdn.example/a.png' })
    expect(parsePrediction({ data: { status: 'succeeded', outputs: ['https://cdn.example/b.png'] } }))
      .toEqual({ kind: 'done', url: 'https://cdn.example/b.png' })
  })

  it('reads the top-level status + result.output shape', () => {
    expect(parsePrediction({ status: 'completed', result: { output: 'https://cdn.example/c.png' } }))
      .toEqual({ kind: 'done', url: 'https://cdn.example/c.png' })
  })

  it('turns a failure into its message, and a finish without output into a failure', () => {
    expect(parsePrediction({ data: { status: 'failed', error: 'prompt rejected' } }))
      .toEqual({ kind: 'failed', error: 'prompt rejected' })
    expect(parsePrediction({ data: { status: 'completed', outputs: [] } }).kind).toBe('failed')
  })
})

describe('explainAtlasStatus', () => {
  it('names the fix for the common failures', () => {
    expect(explainAtlasStatus(401, '')).toMatch(/API key/)
    expect(explainAtlasStatus(402, '')).toMatch(/balance/)
    expect(explainAtlasStatus(500, '{"message":"boom"}')).toBe('Atlas Cloud answered 500: boom')
  })
})

describe('submitAtlasImage', () => {
  it('posts model, prompt and W*H with the Bearer key', async () => {
    const calls: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      calls.push({ url, init })
      return reply(200, { data: { id: 'pred-1' } })
    }))
    const id = await submitAtlasImage({ model: 'm', prompt: 'a cat', width: 1024, height: 1536 }, 'k')
    expect(id).toBe('pred-1')
    expect(calls[0].url).toBe('https://api.atlascloud.ai/api/v1/model/generateImage')
    expect(calls[0].init.headers.Authorization).toBe('Bearer k')
    expect(JSON.parse(calls[0].init.body)).toEqual({ model: 'm', prompt: 'a cat', size: '1024*1536' })
  })

  it('retries once without a size when Atlas refuses the request', async () => {
    const bodies: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init: any) => {
      bodies.push(JSON.parse(init.body))
      return bodies.length === 1 ? reply(400, { message: 'invalid size' }) : reply(200, { data: { id: 'pred-2' } })
    }))
    expect(await submitAtlasImage({ model: 'm', prompt: 'p', width: 512, height: 512 }, 'k')).toBe('pred-2')
    expect(bodies[1]).toEqual({ model: 'm', prompt: 'p' })
  })

  it('does not retry an auth failure', async () => {
    const fetchMock = vi.fn(async () => reply(401, {}))
    vi.stubGlobal('fetch', fetchMock)
    await expect(submitAtlasImage({ model: 'm', prompt: 'p', width: 1, height: 1 }, 'k')).rejects.toThrow(/API key/)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('waitForAtlasImage', () => {
  it('polls the prediction endpoint until the image is there', async () => {
    const urls: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      urls.push(url)
      return reply(200, { data: { status: 'completed', outputs: ['https://cdn.example/x.png'] } })
    }))
    expect(await waitForAtlasImage('pred 1', 'k')).toBe('https://cdn.example/x.png')
    expect(urls[0]).toBe('https://api.atlascloud.ai/api/v1/model/prediction/pred%201')
  })

  it('surfaces a failed render', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => reply(200, { data: { status: 'failed', error: 'nope' } })))
    await expect(waitForAtlasImage('p', 'k')).rejects.toThrow('nope')
  })
})

describe('getAtlasApiKey', () => {
  beforeEach(() => {
    useProviderStore.getState().setProviderApiKey('anthropic', '')
  })

  it('is the anthropic slot key only while that slot points at Atlas Cloud', () => {
    const st = useProviderStore.getState()
    st.setProviderConfig('anthropic', { name: 'Atlas Cloud', baseUrl: 'https://api.atlascloud.ai' })
    st.setProviderApiKey('anthropic', 'atlas-key')
    expect(getAtlasApiKey()).toBe('atlas-key')

    st.setProviderConfig('anthropic', { name: 'Anthropic', baseUrl: 'https://api.anthropic.com' })
    expect(getAtlasApiKey()).toBe('')
  })
})
