// ComfyUI: list models, submit a workflow, wait for it, fetch the outputs.
// Same protocol the app uses in src/api/comfyui.ts: POST /prompt only enqueues,
// completion is found by polling /history/<id> with short requests.

import { randomUUID } from 'node:crypto'
import { request, HttpError } from './http.js'

const POLL_INTERVAL_MS = 1000
const SHORT_TIMEOUT_MS = 15_000

/**
 * Options of one ComfyUI dropdown. ComfyUI serves two shapes, often in the same
 * install: legacy `[[...options], cfg]` and newer `["COMBO", { options }]`.
 * Port of readComboOptions in src/api/comfyui-enum.ts.
 */
export function readComboOptions(spec) {
  if (!Array.isArray(spec)) return Array.isArray(spec?.options) ? spec.options.filter((v) => typeof v === 'string') : null
  const head = spec[0]
  if (Array.isArray(head)) return head.filter((v) => typeof v === 'string')
  if (typeof head === 'string') {
    if (Array.isArray(spec[1]?.options)) return spec[1].options.filter((v) => typeof v === 'string')
    return head.toUpperCase() === 'COMBO' ? [] : null
  }
  return null
}

export async function comfyStatus(config, signal) {
  try {
    const stats = await request(`${config.comfyUrl}/system_stats`, { timeoutMs: config.probeTimeoutMs, signal })
    return {
      online: true,
      url: config.comfyUrl,
      version: stats?.system?.comfyui_version,
      devices: (stats?.devices ?? []).map((d) => ({
        name: d.name,
        vramTotalGB: d.vram_total ? +(d.vram_total / 1024 ** 3).toFixed(1) : undefined,
        vramFreeGB: d.vram_free ? +(d.vram_free / 1024 ** 3).toFixed(1) : undefined,
      })),
    }
  } catch (err) {
    return { online: false, url: config.comfyUrl, error: err.message }
  }
}

/** Reads the dropdown `input` of `node`; [] when the node is not installed. */
export async function nodeOptions(config, node, input, signal) {
  let info
  try {
    info = await request(`${config.comfyUrl}/object_info/${node}`, { timeoutMs: SHORT_TIMEOUT_MS, signal })
  } catch (err) {
    if (err instanceof HttpError && err.status === 404) return []
    throw err
  }
  const spec = info?.[node]?.input?.required?.[input] ?? info?.[node]?.input?.optional?.[input]
  return readComboOptions(spec) ?? []
}

export async function listComfyModels(config, signal) {
  const [checkpoints, diffusionModels, loras, samplers, schedulers] = await Promise.all([
    nodeOptions(config, 'CheckpointLoaderSimple', 'ckpt_name', signal),
    nodeOptions(config, 'UNETLoader', 'unet_name', signal),
    nodeOptions(config, 'LoraLoader', 'lora_name', signal),
    nodeOptions(config, 'KSampler', 'sampler_name', signal),
    nodeOptions(config, 'KSampler', 'scheduler', signal),
  ])
  return { checkpoints, diffusionModels, loras, samplers, schedulers }
}

/**
 * The classic text to image graph. It needs an all-in-one checkpoint (SD 1.5,
 * SDXL, Pony, most community merges) that carries its own CLIP and VAE. Split
 * models (Flux, Wan, HiDream...) need their own graph: use comfyui_run_workflow.
 */
export function buildTxt2ImgWorkflow(p) {
  const seed = p.seed ?? Math.floor(Math.random() * 2 ** 32)
  const workflow = {
    4: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: p.checkpoint } },
    5: { class_type: 'EmptyLatentImage', inputs: { width: p.width, height: p.height, batch_size: 1 } },
    6: { class_type: 'CLIPTextEncode', inputs: { text: p.prompt, clip: ['4', 1] } },
    7: { class_type: 'CLIPTextEncode', inputs: { text: p.negativePrompt ?? '', clip: ['4', 1] } },
    3: {
      class_type: 'KSampler',
      inputs: {
        seed,
        steps: p.steps,
        cfg: p.cfg,
        sampler_name: p.sampler,
        scheduler: p.scheduler,
        denoise: 1,
        model: ['4', 0],
        positive: ['6', 0],
        negative: ['7', 0],
        latent_image: ['5', 0],
      },
    },
    8: { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
    9: { class_type: 'SaveImage', inputs: { filename_prefix: 'LU_MCP', images: ['8', 0] } },
  }
  return { workflow, seed }
}

export async function submitWorkflow(config, workflow, signal) {
  const clientId = randomUUID()
  const res = await request(`${config.comfyUrl}/prompt`, {
    method: 'POST',
    body: { prompt: workflow, client_id: clientId },
    timeoutMs: 30_000,
    signal,
  })
  if (res?.error || (res?.node_errors && Object.keys(res.node_errors).length > 0)) {
    throw new Error(`ComfyUI rejected the workflow: ${JSON.stringify(res.error ?? res.node_errors).slice(0, 800)}`)
  }
  if (!res?.prompt_id) throw new Error(`ComfyUI returned no prompt_id: ${JSON.stringify(res).slice(0, 300)}`)
  return res.prompt_id
}

/** Every file a finished history entry produced (images, gifs, videos, audio). */
export function collectOutputs(entry) {
  const files = []
  for (const [nodeId, out] of Object.entries(entry?.outputs ?? {})) {
    for (const key of ['images', 'gifs', 'videos', 'audio']) {
      for (const f of out?.[key] ?? []) {
        if (f?.filename) files.push({ nodeId, filename: f.filename, subfolder: f.subfolder ?? '', type: f.type ?? 'output' })
      }
    }
  }
  return files
}

/**
 * One look at a job. `done` is false while it is queued or running.
 * Throws when ComfyUI reports the run as failed.
 */
export async function checkJob(config, promptId, signal) {
  const history = await request(`${config.comfyUrl}/history/${encodeURIComponent(promptId)}`, {
    timeoutMs: SHORT_TIMEOUT_MS,
    signal,
  })
  const entry = history?.[promptId]
  if (!entry) return { done: false }
  const status = entry.status
  if (status?.status_str === 'error') {
    const errMsg = (status.messages ?? []).find((m) => m[0] === 'execution_error')?.[1]
    const detail = errMsg ? `${errMsg.node_type}: ${errMsg.exception_message}` : 'see the ComfyUI console'
    throw new Error(`ComfyUI run ${promptId} failed (${detail})`)
  }
  if (status && status.completed === false) return { done: false }
  return { done: true, outputs: collectOutputs(entry) }
}

export async function waitForJob(config, promptId, { signal, onTick } = {}) {
  const deadline = Date.now() + config.timeoutMs
  let ticks = 0
  while (Date.now() < deadline) {
    const job = await checkJob(config, promptId, signal)
    if (job.done) return job.outputs
    await onTick?.(++ticks)
    await sleep(POLL_INTERVAL_MS, signal)
  }
  throw new Error(
    `ComfyUI job ${promptId} is still running after ${Math.round(config.timeoutMs / 1000)} s. ` +
    'It keeps rendering in ComfyUI; check it later with comfyui_get_result.',
  )
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error('Cancelled while waiting for ComfyUI'))
    const onAbort = () => {
      clearTimeout(timer)
      reject(new Error('Cancelled while waiting for ComfyUI'))
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

export function viewUrl(config, file) {
  const q = new URLSearchParams({ filename: file.filename, subfolder: file.subfolder, type: file.type })
  return `${config.comfyUrl}/view?${q}`
}

/** Downloads one output; returns base64 + mime, or null when it is too big to inline. */
export async function fetchOutput(config, file, signal) {
  const res = await request(viewUrl(config, file), { timeoutMs: 60_000, signal, raw: true })
  const buf = Buffer.from(await res.arrayBuffer())
  const mimeType = res.headers.get('content-type')?.split(';')[0] || guessMime(file.filename)
  if (buf.length > config.maxInlineImageBytes) return { tooLarge: true, bytes: buf.length, mimeType }
  return { data: buf.toString('base64'), bytes: buf.length, mimeType }
}

function guessMime(name) {
  const ext = name.split('.').pop()?.toLowerCase()
  return { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif' }[ext] ?? 'application/octet-stream'
}
