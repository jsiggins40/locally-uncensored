/**
 * Atlas Cloud image generation and editing (Seedream 5.0 Pro), bring-your-own-key.
 *
 * Rides the key of the Atlas Cloud chat preset: when the `anthropic` slot points
 * at api.atlascloud.ai, its key also unlocks Atlas's image API. Atlas renders
 * asynchronously: POST generateImage returns a prediction id, and the
 * prediction endpoint is polled until it reports an output URL.
 *
 * Atlas documents two response shapes (`data.status` + `data.outputs[]` on the
 * English reference, top-level `status` + `result.output` on some localized
 * pages), so the parsers below accept both and fail loudly on anything else.
 */

import { localFetch, hostnameOf, ensureProxyAllowsHost, isTauri, backendCall, fetchExternalBytes } from './backend'
import { useProviderStore } from '../stores/providerStore'

export const ATLAS_API_BASE = 'https://api.atlascloud.ai'
const ATLAS_HOST = 'api.atlascloud.ai'

export interface AtlasImageModel {
  /** Text-to-image endpoint; also the id the Create pick is stored under. */
  id: string
  /** The same model's image-edit endpoint, for the Edit intent. */
  editId?: string
  label: string
}

/** The models offered in the Create picker. Seedream 5.0 Pro only, by choice. */
export const ATLAS_IMAGE_MODELS: AtlasImageModel[] = [
  { id: 'bytedance/seedream-v5.0-pro/text-to-image', editId: 'bytedance/seedream-v5.0-pro/edit', label: 'Seedream 5.0 Pro' },
]

export function atlasImageModelById(id: string | null | undefined): AtlasImageModel | undefined {
  return id ? ATLAS_IMAGE_MODELS.find((m) => m.id === id) : undefined
}

/** The endpoint a pick runs on for this intent, or null when it has none. */
export function atlasModelForIntent(pickId: string | null | undefined, intent: string): string | null {
  const m = atlasImageModelById(pickId)
  if (!m) return null
  if (intent === 'image') return m.id
  if (intent === 'edit') return m.editId ?? null
  return null
}

/** The Atlas key, or '' when the anthropic slot is not set up for Atlas Cloud. */
export function getAtlasApiKey(): string {
  const st = useProviderStore.getState()
  const slot = st.providers.anthropic
  if (!slot || hostnameOf(slot.baseUrl) !== ATLAS_HOST) return ''
  return st.getProviderApiKey('anthropic')
}

export class AtlasImageError extends Error {
  readonly status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.name = 'AtlasImageError'
    this.status = status
  }
}

type PredictionState =
  | { kind: 'pending' }
  | { kind: 'done'; url: string }
  | { kind: 'failed'; error: string }

/** The prediction id from a generateImage reply. */
export function parseSubmitId(body: any): string | null {
  const id = body?.data?.id ?? body?.data?.prediction_id ?? body?.id ?? body?.predictionId ?? body?.prediction_id
  return typeof id === 'string' && id ? id : null
}

/** Where a poll reply says the prediction is. */
export function parsePrediction(body: any): PredictionState {
  const data = body?.data ?? body
  const status = String(data?.status ?? body?.status ?? '').toLowerCase()
  const outputs = data?.outputs ?? data?.output ?? body?.result?.output ?? body?.result?.outputs
  const url = Array.isArray(outputs) ? outputs.find((o) => typeof o === 'string' && o) : typeof outputs === 'string' ? outputs : undefined
  if (status === 'completed' || status === 'succeeded' || status === 'success') {
    return url ? { kind: 'done', url } : { kind: 'failed', error: 'Atlas Cloud finished the render but returned no image.' }
  }
  if (status === 'failed' || status === 'error' || status === 'canceled' || status === 'cancelled') {
    const err = data?.error ?? body?.error ?? body?.message
    return { kind: 'failed', error: typeof err === 'string' && err ? err : 'Atlas Cloud could not render this image.' }
  }
  return { kind: 'pending' }
}

/** Plain-language message for an HTTP failure from Atlas. */
export function explainAtlasStatus(status: number, body: string): string {
  let detail = ''
  try {
    const j = JSON.parse(body)
    detail = j?.message ?? j?.error?.message ?? j?.error ?? j?.msg ?? ''
  } catch { detail = body.slice(0, 200) }
  if (status === 401 || status === 403) return 'Atlas Cloud rejected the API key. Check it in Settings, AI Backends, Atlas Cloud.'
  if (status === 402) return 'Your Atlas Cloud balance is empty. Top up your credit at atlascloud.ai.'
  if (status === 429) return 'Atlas Cloud is rate limiting this key. Wait a moment and try again.'
  return `Atlas Cloud answered ${status}${detail ? `: ${detail}` : ''}`
}

async function atlasRequest(path: string, key: string, init: { method?: string; body?: string } = {}): Promise<any> {
  await ensureProxyAllowsHost(ATLAS_API_BASE)
  const res = await localFetch(`${ATLAS_API_BASE}${path}`, {
    method: init.method ?? 'GET',
    body: init.body,
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
  })
  const text = await res.text()
  if (!res.ok) throw new AtlasImageError(explainAtlasStatus(res.status, text), res.status)
  try {
    return JSON.parse(text)
  } catch {
    throw new AtlasImageError('Atlas Cloud sent a reply that is not JSON.')
  }
}

export interface AtlasImageRequest {
  model: string
  prompt: string
  width: number
  height: number
  /** Edit models: URLs of the source image(s), from uploadAtlasImage. */
  images?: string[]
}

/**
 * Submit a render and return its prediction id. Sends the picked size as
 * `W*H`; if Atlas refuses the request (400/422, e.g. a size outside the
 * model's presets) it is sent once more without a size, so the model's own
 * default applies instead of the run failing.
 */
export async function submitAtlasImage(req: AtlasImageRequest, key: string): Promise<string> {
  const base = { model: req.model, prompt: req.prompt, ...(req.images ? { images: req.images } : {}) }
  const post = (body: object) =>
    atlasRequest('/api/v1/model/generateImage', key, { method: 'POST', body: JSON.stringify(body) })
  let reply: any
  try {
    reply = await post({ ...base, size: `${req.width}*${req.height}` })
  } catch (err) {
    if (!(err instanceof AtlasImageError) || (err.status !== 400 && err.status !== 422)) throw err
    reply = await post(base)
  }
  const id = parseSubmitId(reply)
  if (!id) throw new AtlasImageError('Atlas Cloud accepted the render but returned no prediction id.')
  return id
}

/** The temporary URL from an uploadMedia reply. */
export function parseUploadUrl(body: any): string | null {
  const d = body?.data ?? body
  const url = d?.download_url ?? d?.url ?? d?.file_url ?? body?.url
  return typeof url === 'string' && url ? url : null
}

/**
 * Upload an edit source (a data: URL, as the Create store holds it) to Atlas
 * and return the URL the edit model reads. Desktop goes through Rust, since the
 * CSP blocks the host and the generic proxy carries text only.
 */
export async function uploadAtlasImage(dataUrl: string, key: string): Promise<string> {
  const m = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(dataUrl)
  if (!m || !m[2]) throw new AtlasImageError('The source image could not be read for upload.')
  const mime = m[1] || 'image/png'
  const ext = mime === 'image/jpeg' ? 'jpg' : mime === 'image/webp' ? 'webp' : 'png'
  const filename = `source.${ext}`
  let text: string
  if (isTauri()) {
    try {
      text = await backendCall<string>('atlas_upload_media', { apiKey: key, filename, mime, dataB64: m[3] })
    } catch (err) {
      const hit = /^HTTP (\d{3}): ([\s\S]*)$/.exec(String(err))
      throw new AtlasImageError(hit ? explainAtlasStatus(Number(hit[1]), hit[2]) : String(err), hit ? Number(hit[1]) : undefined)
    }
  } else {
    const form = new FormData()
    form.append('file', new Blob([base64ToBytes(m[3])], { type: mime }), filename)
    const res = await fetch(`${ATLAS_API_BASE}/api/v1/model/uploadMedia`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}` },
      body: form,
    })
    text = await res.text()
    if (!res.ok) throw new AtlasImageError(explainAtlasStatus(res.status, text), res.status)
  }
  let url: string | null = null
  try { url = parseUploadUrl(JSON.parse(text)) } catch { /* handled below */ }
  if (!url) throw new AtlasImageError('Atlas Cloud accepted the upload but returned no file URL.')
  return url
}

const POLL_INTERVAL_MS = 2000
const POLL_TIMEOUT_MS = 5 * 60_000

/** Poll a prediction until it has an image URL. Throws on failure, timeout or abort. */
export async function waitForAtlasImage(
  id: string,
  key: string,
  opts: { signal?: AbortSignal; onTick?: (elapsedSec: number) => void } = {},
): Promise<string> {
  const started = Date.now()
  for (;;) {
    if (opts.signal?.aborted) throw new AtlasImageError('aborted')
    const state = parsePrediction(await atlasRequest(`/api/v1/model/prediction/${encodeURIComponent(id)}`, key))
    if (state.kind === 'done') return state.url
    if (state.kind === 'failed') throw new AtlasImageError(state.error)
    const elapsed = Date.now() - started
    if (elapsed > POLL_TIMEOUT_MS) throw new AtlasImageError('Atlas Cloud is taking longer than five minutes. The render may still finish in your Atlas account.')
    opts.onTick?.(Math.round(elapsed / 1000))
    await new Promise<void>((resolve) => {
      const t = setTimeout(resolve, POLL_INTERVAL_MS)
      opts.signal?.addEventListener('abort', () => { clearTimeout(t); resolve() }, { once: true })
    })
  }
}

/**
 * Bring a finished render home. Desktop: saved to disk by Rust (the CSP blocks
 * the CDN in an <img>, and the link expires), returned with its path so the
 * gallery re-reads it after a restart. Browser dev mode: in memory only.
 */
export async function fetchAtlasResult(url: string): Promise<{ dataUrl: string; localPath?: string; filename: string }> {
  if (isTauri()) {
    const localPath = await backendCall<string>('save_remote_image', { url })
    const b64 = await backendCall<string>('read_media_file', { path: localPath })
    const filename = localPath.split(/[\\/]/).pop() || 'atlas.png'
    return { dataUrl: base64ToBlobUrl(b64, mimeFor(filename)), localPath, filename }
  }
  const bytes = await fetchExternalBytes(url)
  const filename = 'atlas.png'
  return { dataUrl: URL.createObjectURL(new Blob([bytes])), filename }
}

function mimeFor(filename: string): string {
  if (/\.jpe?g$/i.test(filename)) return 'image/jpeg'
  if (/\.webp$/i.test(filename)) return 'image/webp'
  return 'image/png'
}

function base64ToBytes(b64: string): Uint8Array<ArrayBuffer> {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

function base64ToBlobUrl(b64: string, mime: string): string {
  return URL.createObjectURL(new Blob([base64ToBytes(b64)], { type: mime }))
}
