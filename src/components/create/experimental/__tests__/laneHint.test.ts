/**
 * Regression: the Motion Control card kept printing "add a character image and
 * a driving dance/pose video above, then hit Create" while both chips were
 * filled and the sampler was already counting steps (David, 2026-07-26, live
 * run on the 2.5.9 ship exe). The caption hung off `!needsPrompt` alone, so it
 * never reacted to the inputs or to a run being in flight.
 *
 * These pin the rule in both directions, including the two cases that must NOT
 * change: the eraser caption is the only place that mentions painting a mask,
 * and a mask is deliberately not part of `specialReady`.
 */
import { describe, it, expect } from 'vitest'
import { noPromptHint, shouldShowLaneHint, editLaneHasStrength, KONTEXT_NO_STRENGTH_HINT } from '../laneHint'
import type { CreateIntent } from '../../../../stores/createStore'

const show = (
  intent: CreateIntent,
  o: Partial<{ needPrompt: boolean; isGenerating: boolean; specialReady: boolean }> = {},
) =>
  shouldShowLaneHint({
    needPrompt: false,
    isGenerating: false,
    specialReady: false,
    intent,
    ...o,
  })

describe('shouldShowLaneHint', () => {
  it('shows the motion caption while the inputs are still missing', () => {
    expect(show('motion', { specialReady: false })).toBe(true)
  })

  it('hides the motion caption once both chips are in', () => {
    expect(show('motion', { specialReady: true })).toBe(false)
  })

  it('hides every caption while a run is in flight', () => {
    // The exact case David watched: ready inputs plus a running sampler.
    expect(show('motion', { specialReady: true, isGenerating: true })).toBe(false)
    // And the incomplete case too, because mid run it is not an instruction.
    expect(show('motion', { specialReady: false, isGenerating: true })).toBe(false)
    expect(show('eraser', { isGenerating: true })).toBe(false)
  })

  it('applies the same rule to the other two input captions', () => {
    for (const id of ['character', 'lipsync'] as CreateIntent[]) {
      expect(show(id, { specialReady: false })).toBe(true)
      expect(show(id, { specialReady: true })).toBe(false)
    }
  })

  it('keeps the action captions up even when the lane reports ready', () => {
    // These describe the click, not a missing input. The eraser one carries the
    // only "paint a mask" instruction in the UI and specialReady is true for it
    // as soon as an image is loaded, so readiness must not hide it.
    for (const id of ['eraser', 'upscale', 'removebg'] as CreateIntent[]) {
      expect(show(id, { specialReady: true })).toBe(true)
    }
  })

  it('shows nothing at all for lanes that have a prompt field', () => {
    expect(show('image', { needPrompt: true })).toBe(false)
    expect(show('motion', { needPrompt: true, specialReady: false })).toBe(false)
  })
})

describe('noPromptHint', () => {
  it('names the inputs each lane is waiting for', () => {
    expect(noPromptHint('motion')).toContain('driving dance/pose video')
    expect(noPromptHint('lipsync')).toContain('voice')
    expect(noPromptHint('character')).toContain('4 to 30 photos')
  })

  it('keeps the eraser on its own line instead of the cutout default', () => {
    // The old copy said "remove the background" for every prompt-less lane,
    // which walked eraser users straight into "Paint a mask first".
    expect(noPromptHint('eraser')).toContain('Paint a mask')
    expect(noPromptHint('eraser')).not.toContain('remove the background')
    expect(noPromptHint('upscale')).toContain('enhance the image')
    expect(noPromptHint('removebg')).toContain('remove the background')
  })
})

/**
 * The Edit tab's strength slider is real on every lane but one. FLUX.1 Kontext
 * takes the source in as conditioning (ReferenceLatent) and buildDynamicWorkflow
 * samples it at denoise 1.0 — the slider is never read, so a user turning it up
 * to "make the edit stronger" changes nothing and has no way to know why.
 */
describe('editLaneHasStrength', () => {
  it('drops the slider for a local Kontext model, whatever the repack is called', () => {
    expect(editLaneHasStrength('local', 'flux1-dev-kontext_fp8_scaled.safetensors')).toBe(false)
    expect(editLaneHasStrength('local', 'Flux-Kontext-Uncensored.safetensors')).toBe(false)
    expect(editLaneHasStrength('local', 'flux1-kontext-dev-Q4_K_M.gguf')).toBe(false)
  })

  it('keeps it for every other local edit model, where denoise really drives the run', () => {
    expect(editLaneHasStrength('local', 'flux1-dev-fp8.safetensors')).toBe(true)
    expect(editLaneHasStrength('local', 'juggernautXL_v9.safetensors')).toBe(true)
    expect(editLaneHasStrength('local', '')).toBe(true)
  })

  // The Kontext lane is local-only; the cloud edit models do read a strength,
  // so a cloud model that happens to carry the word must not lose its slider.
  it('never touches the cloud lane', () => {
    expect(editLaneHasStrength('cloud', 'flux-kontext-pro')).toBe(true)
  })

  it('says what replaces the knob, not just that it is gone', () => {
    expect(KONTEXT_NO_STRENGTH_HINT).toContain('say what to change')
  })
})
