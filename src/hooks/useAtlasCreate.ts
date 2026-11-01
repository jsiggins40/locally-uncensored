// Atlas Cloud twin of useCreate's generate/cancel for the Image intent, picked
// per model (createStore.atlasImageModel) rather than per backend axis: the
// global Local/Cloud switch stays about LU Cloud, and an Atlas render needs
// neither ComfyUI nor an LU account, only the user's own Atlas key. Store
// choreography mirrors useCloudCreate so every downstream surface (progress,
// gallery, lightbox) works unchanged.

import { useCallback } from 'react'
import { useCreateStore } from '../stores/createStore'
import { useProviderStore } from '../stores/providerStore'
import {
  atlasImageModelById,
  getAtlasApiKey,
  submitAtlasImage,
  waitForAtlasImage,
  fetchAtlasResult,
  AtlasImageError,
} from '../api/atlas-images'
import { checkPromptSafety, SAFETY_BLOCK_MESSAGE } from '../lib/render/safety'

let activeAbort: AbortController | null = null

/** True while an Atlas render this session started is still in flight. */
export const hasActiveAtlasRun = (): boolean => activeAbort !== null

/** The Atlas model the next Image run goes to, or null for the local engine. */
export function activeAtlasImageModel(): string | null {
  const s = useCreateStore.getState()
  if (s.backend !== 'local' || s.intent() !== 'image' || !getAtlasApiKey()) return null
  return atlasImageModelById(s.atlasImageModel)?.id ?? null
}

/** Reactive twin of activeAtlasImageModel for render-time gates (Stage's
 *  missing-models card, Composer's negative-prompt toggle). */
export function useAtlasImagePickActive(): boolean {
  const backend = useCreateStore((s) => s.backend)
  const intent = useCreateStore((s) => s.intent())
  const atlasImageModel = useCreateStore((s) => s.atlasImageModel)
  const hasKey = useProviderStore((s) => !!s.providers.anthropic && !!getAtlasApiKey())
  return backend === 'local' && intent === 'image' && hasKey && !!atlasImageModelById(atlasImageModel)
}

/** One Atlas Image run from the current Create state (exported for tests). */
export async function runAtlasImage(): Promise<void> {
  const s = useCreateStore.getState()
  if (s.isGenerating) return
  s.setError(null)

  const model = atlasImageModelById(s.atlasImageModel)
  if (!model) return
  const key = getAtlasApiKey()
  if (!key) {
    s.setError('Add your Atlas Cloud key first: Settings, AI Backends, Add Provider, Atlas Cloud.')
    return
  }
  const prompt = s.prompt.trim()
  if (!prompt) {
    s.setError('Please enter a prompt.')
    return
  }
  // The app's own non-negotiable gate, the same one the local and LU Cloud
  // paths run. Adult content passes it; sexualized minors never do.
  if (checkPromptSafety(prompt).blocked) {
    s.setError(SAFETY_BLOCK_MESSAGE)
    return
  }

  const ac = new AbortController()
  activeAbort = ac
  s.setIsGenerating(true)
  s.setProgressPhase('queued')
  s.setProgress(5, 'Sending to Atlas Cloud…')

  try {
    const id = await submitAtlasImage({ model: model.id, prompt, width: s.width, height: s.height }, key)
    const url = await waitForAtlasImage(id, key, {
      signal: ac.signal,
      onTick: (sec) => {
        const st = useCreateStore.getState()
        st.setProgressPhase('sampling')
        st.setProgress(Math.min(90, 15 + sec * 2), `Rendering on Atlas Cloud… ${sec}s`)
      },
    })
    useCreateStore.getState().setProgress(95, 'Saving the image…')
    const saved = await fetchAtlasResult(url)

    const st = useCreateStore.getState()
    st.setProgressPhase('complete')
    st.setProgress(100, 'Complete!')
    st.addToGallery({
      id: `atlas-${id}`,
      type: 'image',
      filename: saved.filename,
      subfolder: '',
      prompt,
      negativePrompt: '',
      model: model.label,
      modelType: 'unknown',
      seed: 0,
      steps: 0,
      cfgScale: 0,
      sampler: '',
      scheduler: '',
      width: s.width,
      height: s.height,
      batchSize: 1,
      createdAt: Date.now(),
      dataUrl: saved.dataUrl,
      localPath: saved.localPath,
      intent: 'image',
    })
  } catch (err) {
    // Stopping the poll is not an error. Atlas has no documented cancel, so
    // a render already running there may still finish (and bill).
    if (!(err instanceof AtlasImageError && err.message === 'aborted')) {
      useCreateStore.getState().setError(err instanceof Error ? err.message : String(err))
    }
  } finally {
    activeAbort = null
    const st = useCreateStore.getState()
    st.setIsGenerating(false)
    st.setProgress(0)
  }
}

/** Stop polling the in-flight run, if any. */
export function cancelAtlasImage(): void {
  activeAbort?.abort()
}

export function useAtlasCreate() {
  const generate = useCallback(() => runAtlasImage(), [])
  const cancel = useCallback(() => cancelAtlasImage(), [])
  return { generate, cancel }
}
