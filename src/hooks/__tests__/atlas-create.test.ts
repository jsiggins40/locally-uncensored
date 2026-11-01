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

function setAtlasKey(key: string) {
  const st = useProviderStore.getState()
  st.setProviderConfig('anthropic', { name: 'Atlas Cloud', baseUrl: 'https://api.atlascloud.ai' })
  st.setProviderApiKey('anthropic', key)
}

beforeEach(() => {
  setAtlasKey('atlas-key')
  useCreateStore.setState({
    backend: 'local', mode: 'image', imageSubMode: 'text2img', removebg: false, utilityOp: null, cloudOp: null,
    atlasImageModel: SEEDREAM, prompt: 'a red fox in snow', width: 1024, height: 1024,
    isGenerating: false, error: null, gallery: [],
  } as any)
})

afterEach(() => vi.unstubAllGlobals())

describe('activeAtlasImageModel', () => {
  it('is the pick on the local Image intent with a key', () => {
    expect(activeAtlasImageModel()).toBe(SEEDREAM)
  })

  it('leaves the run local without a pick, without a key, or off the Image intent', () => {
    useCreateStore.setState({ atlasImageModel: null } as any)
    expect(activeAtlasImageModel()).toBeNull()

    useCreateStore.setState({ atlasImageModel: SEEDREAM, imageSubMode: 'img2img' } as any)
    expect(activeAtlasImageModel()).toBeNull()

    useCreateStore.setState({ imageSubMode: 'text2img', backend: 'cloud' } as any)
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
