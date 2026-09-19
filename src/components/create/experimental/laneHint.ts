// The caption under the prompt field for lanes that take no prompt — shared,
// pure, testable.
//
// David 2026-07-26, watching a Motion Control run: the card kept saying "add a
// character image and a driving dance/pose video above, then hit Create" while
// both chips were filled and the sampler was already counting steps. The
// caption was rendered on `!needsPrompt` alone, so it never reacted to anything
// the user did.
import type { CreateIntent } from '../../../stores/createStore'
import { isKontextModel } from '../../../api/comfyui'

/** Captions that name the inputs a lane is waiting for. These go stale the
 *  moment the chips are filled, so they hide once the lane reports ready. */
const INPUT_HINT_INTENTS: ReadonlySet<CreateIntent> =
  new Set(['character', 'lipsync', 'motion'] as CreateIntent[])

/** Caption for prompt-less intents so each reads honestly (the old copy said
 *  "remove the background" for every one of them — misleading the eraser into
 *  a guaranteed "Paint a mask first" error). */
export function noPromptHint(id: CreateIntent): string {
  switch (id) {
    case 'upscale':
      return 'No prompt needed. Just hit Create to enhance the image.'
    case 'eraser':
      return 'No prompt needed. Paint a mask over the object to remove, then hit Create.'
    case 'character':
      return 'Add 4 to 30 photos of one person or character above, pick a trigger word, then hit Create to train.'
    case 'lipsync':
      return 'Add the portrait (or clip) and a voice above, then hit Create to make it speak.'
    case 'motion':
      return 'Add a character image and a driving dance/pose video above, then hit Create.'
    default:
      return 'No prompt needed. Just hit Create to remove the background.'
  }
}

/**
 * Should the caption be on screen right now?
 *
 * Never while a run is going (nothing there is an instruction any more), and
 * never for an "add X above" caption whose inputs are already in. The action
 * captions (cutout, upscale, eraser) stay: they describe the click rather than
 * a missing input, and the eraser's mask is deliberately not part of
 * `specialReady`, so hiding on readiness would drop the one line that tells
 * people to paint a mask.
 */
export function shouldShowLaneHint(opts: {
  /** meta.needsPrompt || characterUse — a lane with a prompt field shows none of this. */
  needPrompt: boolean
  /** A run is in flight. */
  isGenerating: boolean
  intent: CreateIntent
  /** Per-intent input readiness, as computed by the Composer. */
  specialReady: boolean
}): boolean {
  if (opts.needPrompt) return false
  if (opts.isGenerating) return false
  return !(INPUT_HINT_INTENTS.has(opts.intent) && opts.specialReady)
}

/** The caption that takes the Edit-strength slider's place on a Kontext model.
 *  It has to say what replaces the knob, not just that the knob is gone. */
export const KONTEXT_NO_STRENGTH_HINT =
  'Kontext edits from your description — say what to change, there is no strength to set.'

/**
 * Does the Edit lane's strength slider actually drive anything right now?
 *
 * Everywhere else in the Edit tab it does: image-to-image re-noises the frame
 * to `denoise` and inpainting repaints the mask at it. FLUX.1 Kontext is the
 * one model where it does not — the source rides in as conditioning through
 * ReferenceLatent and buildDynamicWorkflow samples at denoise 1.0, never
 * reading the slider at all. Leaving the knob on screen there is a control
 * that moves and changes nothing, which reads as "the edit is too weak, turn
 * it up" when the real answer is to rewrite the instruction.
 *
 * Cloud keeps the slider unconditionally: the Kontext lane is local-only, and
 * the cloud edit models are the ones that do read a strength.
 */
export function editLaneHasStrength(backend: string, imageModel: string): boolean {
  if (backend === 'cloud') return true
  return !isKontextModel(imageModel)
}
