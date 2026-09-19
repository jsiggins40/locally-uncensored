/**
 * FLUX.1 Kontext — instruction image editing lane.
 *
 * LU's Edit tab could only ever repaint pixels: img2img re-noises the whole
 * frame at denoise 0.7, inpaint repaints inside a painted mask. Kontext is a
 * different thing — the source rides in as CONDITIONING (ReferenceLatent) and
 * the prompt is read as an instruction, so "make the jacket red" changes the
 * jacket and leaves the rest of the photo alone.
 *
 * The trap this lane exists to close: Kontext is FLUX.1 architecture, so
 * classifyModel's generic `includes('flux')` claimed it, routed it onto the
 * plain text-to-image path, and the source image was silently dropped — the
 * user got back an unrelated fresh image with no error anywhere.
 *
 * Run: npx vitest run src/api/__tests__/flux-kontext-workflow.test.ts
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock only the live-fetch boundary; every pure helper stays real.
vi.mock('../comfyui-nodes', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../comfyui-nodes')>()
  return { ...actual, getAllNodeInfo: vi.fn() }
})
vi.mock('../comfyui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../comfyui')>()
  return { ...actual, findMatchingCLIP: vi.fn(), findMatchingVAE: vi.fn(), findFluxCLIPPair: vi.fn() }
})

import { classifyModel, isKontextModel, isImageModelType, findMatchingCLIP, findMatchingVAE, findFluxCLIPPair } from '../comfyui'
import { determineStrategy, buildDynamicWorkflow } from '../dynamic-workflow'
import { getImageBundles } from '../discover'
import { getAllNodeInfo } from '../comfyui-nodes'
import type { CategorizedNodes, AvailableModels } from '../comfyui-nodes'

const KONTEXT = 'flux1-dev-kontext_fp8_scaled.safetensors'

/** Minimal live /object_info for a current ComfyUI holding a Kontext model. */
const KONTEXT_NODES: Record<string, unknown> = {
  UNETLoader: { input: { required: { unet_name: [[]] } } },
  DualCLIPLoader: { input: { required: {} } },
  CLIPLoader: { input: { required: { clip_name: [[]] } } },
  VAELoader: { input: { required: { vae_name: [[]] } } },
  CLIPTextEncode: { input: { required: {} } },
  ConditioningZeroOut: { input: { required: {} } },
  FluxGuidance: { input: { required: {} } },
  ReferenceLatent: { input: { required: {} } },
  FluxKontextImageScale: { input: { required: { image: [[]] } } },
  LoadImage: { input: { required: { image: [[]] } } },
  VAEEncode: { input: { required: {} } },
  EmptySD3LatentImage: { input: { required: {} } },
  KSampler: { input: { required: {} } },
  VAEDecode: { input: { required: {} } },
  SaveImage: { input: { required: {} } },
}

const editParams = {
  model: KONTEXT,
  prompt: 'give him a leather jacket', negativePrompt: '',
  sampler: 'euler', scheduler: 'simple',
  width: 1024, height: 1024, steps: 20, cfgScale: 1, seed: 7, batchSize: 1,
  inputImage: 'photo.png',
}

interface WfNode { class_type: string; inputs: Record<string, unknown> }

const node = (wf: Record<string, unknown>, type: string) =>
  (Object.entries(wf) as [string, WfNode][]).find(([, n]) => n.class_type === type)


function nodes(): CategorizedNodes {
  return {
    loaders: ['UNETLoader', 'CheckpointLoaderSimple', 'CLIPLoader', 'DualCLIPLoader', 'VAELoader', 'LoadImage'],
    samplers: ['KSampler'],
    latentInit: ['EmptyLatentImage', 'EmptySD3LatentImage'],
    textEncoders: ['CLIPTextEncode', 'ConditioningZeroOut'],
    decoders: ['VAEDecode'],
    savers: ['SaveImage'],
    videoSavers: [],
    motion: [],
  }
}

const models: AvailableModels = {
  checkpoints: [], unets: [KONTEXT],
  vaes: ['ae.safetensors'], clips: ['t5xxl_fp8_e4m3fn.safetensors', 'clip_l.safetensors'],
  motionModels: [],
}

// ── Recognition ─────────────────────────────────────────────────────────

describe('Kontext recognition', () => {
  it('spots a Kontext checkpoint by name', () => {
    expect(isKontextModel(KONTEXT)).toBe(true)
    expect(isKontextModel('flux1-kontext-dev-Q4_K_M.gguf')).toBe(true)
    expect(isKontextModel('Flux-Kontext-Uncensored.safetensors')).toBe(true)
  })

  it('does not claim plain FLUX, and survives a missing name', () => {
    expect(isKontextModel('flux1-dev-fp8.safetensors')).toBe(false)
    expect(isKontextModel('')).toBe(false)
    expect(isKontextModel(null)).toBe(false)
    expect(isKontextModel(undefined)).toBe(false)
  })

  // Deliberate: Kontext shares FLUX.1's autoencoder and text encoders, so
  // findMatchingVAE / findMatchingCLIP must keep resolving it as FLUX. Giving
  // it its own ModelType would have meant a new branch in both resolvers with
  // nothing different to say.
  it('still classifies as flux, so the VAE/CLIP resolvers need no new branch', () => {
    expect(classifyModel(KONTEXT)).toBe('flux')
    expect(isImageModelType(classifyModel(KONTEXT))).toBe(true)
  })
})

// ── Strategy routing ────────────────────────────────────────────────────

describe('Kontext strategy routing', () => {
  it('routes to the edit lane when a source image is staged', () => {
    const r = determineStrategy('flux', false, nodes(), models, KONTEXT, true)
    expect(r.strategy).toBe('unet_flux_kontext')
  })

  // Kontext generates from a bare prompt perfectly well; with nothing to edit
  // the reference graph would have no reference.
  it('falls back to plain text-to-image FLUX with no source image', () => {
    const r = determineStrategy('flux', false, nodes(), models, KONTEXT, false)
    expect(r.strategy).toBe('unet_flux')
  })

  it('leaves plain FLUX on the text-to-image path even with a source image', () => {
    const r = determineStrategy('flux', false, nodes(), models, 'flux1-dev-fp8.safetensors', true)
    expect(r.strategy).toBe('unet_flux')
  })

  it('reports the missing loader honestly instead of guessing', () => {
    const bare = { ...nodes(), loaders: ['CheckpointLoaderSimple'] }
    const r = determineStrategy('flux', false, bare, models, KONTEXT, true)
    expect(r.strategy).toBe('unavailable')
    expect(r.reason).toContain('Kontext')
  })

  // The old callers pass neither argument. They must keep the old behaviour.
  it('is backward compatible when the caller passes no model name', () => {
    expect(determineStrategy('flux', false, nodes(), models).strategy).toBe('unet_flux')
  })
})

// ── Bundle ──────────────────────────────────────────────────────────────

describe('Kontext bundle', () => {
  const bundle = getImageBundles().find((b) => b.name.includes('Kontext'))

  it('is offered in the image bundles', () => {
    expect(bundle).toBeDefined()
  })

  it('ships the model plus the three shared FLUX components', () => {
    const files = bundle!.files
    expect(files.map((f) => f.subfolder)).toEqual(
      expect.arrayContaining(['diffusion_models', 'vae', 'text_encoders']),
    )
    // The filename is what routes the lane — if a repack renames it without
    // 'kontext' the graph silently degrades to text-to-image again.
    const model = files.find((f) => f.subfolder === 'diffusion_models')!
    expect(isKontextModel(model.filename)).toBe(true)
  })

  it('reuses the exact FLUX.1 encoder + VAE files, so the dev bundle shares them', () => {
    const flux = getImageBundles().find((b) => b.name.includes('FLUX.1 [dev]'))!
    const shared = (name: string) => flux.files.find((f) => f.filename === name)
    for (const f of bundle!.files.filter((f) => f.subfolder !== 'diffusion_models')) {
      expect(shared(f.filename!)?.downloadUrl).toBe(f.downloadUrl)
    }
  })
})

// ── The graph itself ────────────────────────────────────────────────────

describe('buildDynamicWorkflow — Kontext edit graph', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getAllNodeInfo).mockResolvedValue(KONTEXT_NODES as never)
    vi.mocked(findMatchingVAE).mockResolvedValue('ae.safetensors')
    vi.mocked(findFluxCLIPPair).mockResolvedValue({ t5: 't5xxl_fp8_e4m3fn.safetensors', clipL: 'clip_l.safetensors' })
    vi.mocked(findMatchingCLIP).mockResolvedValue('t5xxl_fp8_e4m3fn.safetensors')
  })

  it('wires LoadImage → FluxKontextImageScale → VAEEncode → ReferenceLatent', async () => {
    const wf = await buildDynamicWorkflow({ ...editParams } as never)

    const [loadId, load] = node(wf, 'LoadImage')!
    const [scaleId, scale] = node(wf, 'FluxKontextImageScale')!
    const [encId, enc] = node(wf, 'VAEEncode')!
    const [, ref] = node(wf, 'ReferenceLatent')!
    const [, textEnc] = node(wf, 'CLIPTextEncode')!

    expect(load.inputs.image).toBe('photo.png')
    expect(scale.inputs.image).toEqual([loadId, 0])
    expect(enc.inputs.pixels).toEqual([scaleId, 0])
    // The reference is the encoded SOURCE, and the conditioning it decorates is
    // the prompt — the two halves of "apply this instruction to that image".
    expect(ref.inputs.latent).toEqual([encId, 0])
    expect(textEnc.inputs.text).toBe('give him a leather jacket')
    expect(ref.inputs.conditioning).toBeDefined()
  })

  it('samples from the encoded source at denoise 1.0, not a partial re-noise', async () => {
    const wf = await buildDynamicWorkflow({ ...editParams } as never)
    const [encId] = node(wf, 'VAEEncode')!
    const [, sampler] = node(wf, 'KSampler')!

    // The source latent is the starting point, so the edit keeps the frame.
    expect(sampler.inputs.latent_image).toEqual([encId, 0])
    // Kontext is NOT img2img: the source is not partially destroyed first.
    expect(sampler.inputs.denoise).toBe(1.0)
    const types = (Object.values(wf) as WfNode[]).map((n) => n.class_type)
    expect(types).not.toContain('EmptySD3LatentImage')
    expect(types).not.toContain('EmptyLatentImage')
    // The negative encoder is replaced in place, so no stranded CLIPTextEncode
    // rides along in the submitted graph.
    expect(types.filter((t) => t === 'CLIPTextEncode')).toHaveLength(1)
    expect(node(wf, 'ConditioningZeroOut')![1].inputs.conditioning).toBeDefined()
  })

  it('moves edit strength onto FluxGuidance and pins the sampler CFG to 1', async () => {
    const wf = await buildDynamicWorkflow({ ...editParams } as never)
    const [guidId, guid] = node(wf, 'FluxGuidance')!
    const [refId] = node(wf, 'ReferenceLatent')!
    const [, sampler] = node(wf, 'KSampler')!

    expect(guid.inputs.conditioning).toEqual([refId, 0])
    expect(guid.inputs.guidance).toBe(2.5)
    expect(sampler.inputs.positive).toEqual([guidId, 0])
    // Applying the guidance twice — once here, once as CFG — washes the edit out.
    expect(sampler.inputs.cfg).toBe(1.0)
  })

  it('lets the CFG slider drive the guidance when the user raises it', async () => {
    const wf = await buildDynamicWorkflow({ ...editParams, cfgScale: 4 } as never)
    expect(node(wf, 'FluxGuidance')![1].inputs.guidance).toBe(4)
    expect(node(wf, 'KSampler')![1].inputs.cfg).toBe(1.0)
  })

  it('degrades instead of 400ing when the ComfyUI has no FluxKontextImageScale', async () => {
    const older = { ...KONTEXT_NODES }
    delete older.FluxKontextImageScale
    vi.mocked(getAllNodeInfo).mockResolvedValue(older as never)

    const wf = await buildDynamicWorkflow({ ...editParams } as never)
    const [loadId] = node(wf, 'LoadImage')!
    // Encodes the raw image rather than refusing the edit.
    expect(node(wf, 'VAEEncode')![1].inputs.pixels).toEqual([loadId, 0])
    expect(node(wf, 'ReferenceLatent')).toBeDefined()
  })

  it('refuses with a readable reason when ReferenceLatent is missing', async () => {
    const older = { ...KONTEXT_NODES }
    delete older.ReferenceLatent
    vi.mocked(getAllNodeInfo).mockResolvedValue(older as never)

    await expect(buildDynamicWorkflow({ ...editParams } as never))
      .rejects.toThrow(/ReferenceLatent/)
  })

  it('tells a Kontext user to clear the mask, not to go find a checkpoint', async () => {
    await expect(buildDynamicWorkflow({ ...editParams, maskImage: 'mask.png' } as never))
      .rejects.toThrow(/no mask/)
    // The generic inpaint guard's advice would be wrong here.
    await expect(buildDynamicWorkflow({ ...editParams, maskImage: 'mask.png' } as never))
      .rejects.not.toThrow(/pick a checkpoint model/i)
  })

  // The regression this whole lane exists to prevent.
  it('never silently drops the source image the way plain FLUX did', async () => {
    const wf = await buildDynamicWorkflow({ ...editParams } as never)
    expect(node(wf, 'LoadImage')).toBeDefined()
    expect(node(wf, 'ReferenceLatent')).toBeDefined()
  })
})
