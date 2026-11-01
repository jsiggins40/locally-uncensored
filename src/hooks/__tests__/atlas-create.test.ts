/**
 * An Image run with Seedream picked goes to Atlas Cloud and lands in the
 * gallery; without the pick (or without a key) the local engine keeps it.
 *
 * Run: npx vitest run src/hooks/__tests__/atlas-create.test.ts
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { runAtlasImage, activeAtlasImageModel } from '../useAtlasCreate'
import { useCreateStore } from '../../stores/createStore'
import { useProviderStore } from '../../stores/providerStore'

const SEEDREAM = 'bytedance/seedream-v5.0-pro/text-to-image'
const SEEDREAM_EDIT = 'bytedance/seedream-v5.0-pro/edit'
const PNG_DATA_URL = 'data:image/png;base64,iVBORw0KGgo='

function setAtlasKey(key: string) {
  const st = useProviderStore.getState()
  st.setProviderConfig('anthropic', { name: 'Atlas Cloud', baseUrl: 'https://api.atlascloud.ai' })
  st.setProviderApiKey('anthropic', key)
}

beforeEach(() => {
  setAtlasKey('atlas-key')
  useCreateStore.setState({
    backend: 'local', mode: 'image', imageSubMode: 'text2img', removebg: false, utilityOp: null, cloudOp: null,
    atlasImageModel: SEEDREAM, prompt: 'a red fox in snow', width: 1024, height: 1024, source: null,
    isGenerating: false, error: null, gallery: [],
  } as any)
})

afterEach(() => vi.unstubAllGlobals())

describe('activeAtlasImageModel', () => {
  it('is the pick on the local Image intent, and its edit endpoint on Edit', () => {
    expect(activeAtlasImageModel()).toBe(SEEDREAM)
    useCreateStore.setState({ imageSubMode: 'img2img' } as any)
    expect(activeAtlasImageModel()).toBe(SEEDREAM_EDIT)
  })

  it('leaves the run local without a pick, without a key, or on another intent', () => {
    useCreateStore.setState({ atlasImageModel: null } as any)
    expect(activeAtlasImageModel()).toBeNull()

    useCreateStore.setState({ atlasImageModel: SEEDREAM, mode: 'video' } as any)
    expect(activeAtlasImageModel()).toBeNull()

    useCreateStore.setState({ mode: 'image', backend: 'cloud' } as any)
    expect(activeAtlasImageModel()).toBeNull()

    useCreateStore.setState({ backend: 'local' } as any)
    setAtlasKey('')
    expect(activeAtlasImageModel()).toBeNull()
  })
})

describe('runAtlasImage', () => {
  it('submits, polls and adds the image to the gallery', async () => {
    const urls: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      urls.push(url)
      if (url.endsWith('/generateImage')) {
        return { ok: true, status: 200, text: async () => JSON.stringify({ data: { id: 'pred-9' } }) }
      }
      if (url.includes('/prediction/')) {
        return { ok: true, status: 200, text: async () => JSON.stringify({ data: { status: 'completed', outputs: ['https://cdn.example/fox.png'] } }) }
      }
      // Browser dev mode brings the bytes home through the download proxy.
      return { ok: true, status: 200, arrayBuffer: async () => new Uint8Array([0x89, 0x50, 0x4e, 0x47]).buffer }
    }))

    await runAtlasImage()

    const s = useCreateStore.getState()
    expect(s.error).toBeNull()
    expect(s.isGenerating).toBe(false)
    expect(s.gallery).toHaveLength(1)
    expect(s.gallery[0]).toMatchObject({ id: 'atlas-pred-9', type: 'image', model: 'Seedream 5.0 Pro', prompt: 'a red fox in snow', intent: 'image' })
    expect(s.gallery[0].dataUrl).toMatch(/^blob:/)
    expect(urls[2]).toContain(encodeURIComponent('https://cdn.example/fox.png'))
  })

  it('uploads the source, edits it at its own size and tags the result as an edit', async () => {
    const calls: { url: string; init: any }[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      calls.push({ url, init })
      if (url.endsWith('/uploadMedia')) {
        return { ok: true, status: 200, text: async () => JSON.stringify({ data: { download_url: 'https://tmp.example/src.png' } }) }
      }
      if (url.endsWith('/generateImage')) {
        return { ok: true, status: 200, text: async () => JSON.stringify({ data: { id: 'pred-e' } }) }
      }
      if (url.includes('/prediction/')) {
        return { ok: true, status: 200, text: async () => JSON.stringify({ data: { status: 'completed', outputs: ['https://cdn.example/edited.png'] } }) }
      }
      return { ok: true, status: 200, arrayBuffer: async () => new Uint8Array([0x89, 0x50]).buffer }
    }))
    useCreateStore.setState({
      imageSubMode: 'img2img', prompt: 'make it night',
      source: { filename: '', url: PNG_DATA_URL, width: 800, height: 600 },
    } as any)

    await runAtlasImage()

    expect(calls[0].url).toBe('https://api.atlascloud.ai/api/v1/model/uploadMedia')
    expect(calls[0].init.headers.Authorization).toBe('Bearer atlas-key')
    expect(calls[0].init.body).toBeInstanceOf(FormData)
    expect(JSON.parse(calls[1].init.body)).toEqual({
      model: SEEDREAM_EDIT, prompt: 'make it night', images: ['https://tmp.example/src.png'], size: '800*600',
    })
    const s = useCreateStore.getState()
    expect(s.error).toBeNull()
    expect(s.gallery[0]).toMatchObject({ intent: 'edit', width: 800, height: 600 })
  })

  it('asks for a source on Edit before any request', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    useCreateStore.setState({ imageSubMode: 'img2img' } as any)
    await runAtlasImage()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(useCreateStore.getState().error).toBe('Add a source image first.')
  })

  it('refuses a prompt the app-wide safety gate blocks, before any request', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    useCreateStore.setState({ prompt: 'csam' } as any)
    await runAtlasImage()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(useCreateStore.getState().error).toBeTruthy()
  })

  it('shows the Atlas error and leaves the gallery alone when the render fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) =>
      url.endsWith('/generateImage')
        ? { ok: true, status: 200, text: async () => JSON.stringify({ data: { id: 'p' } }) }
        : { ok: true, status: 200, text: async () => JSON.stringify({ data: { status: 'failed', error: 'render failed upstream' } }) },
    ))
    await runAtlasImage()
    const s = useCreateStore.getState()
    expect(s.error).toBe('render failed upstream')
    expect(s.gallery).toHaveLength(0)
    expect(s.isGenerating).toBe(false)
  })
})
