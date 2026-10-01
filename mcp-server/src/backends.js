// Local chat backends: discovery and chat completions over the OpenAI
// compatible API every supported backend serves.

import { request } from './http.js'

/**
 * Asks one backend for its models. Never throws: an offline backend is a normal
 * state, reported as `online: false` with the reason.
 */
export async function probeBackend(backend, config, signal) {
  try {
    const data = await request(`${backend.baseUrl}/models`, { timeoutMs: config.probeTimeoutMs, signal })
    const models = Array.isArray(data?.data) ? data.data.map((m) => m.id).filter(Boolean) : []
    return { ...backend, online: true, models }
  } catch (err) {
    return { ...backend, online: false, models: [], error: err.message }
  }
}

export async function probeAll(config, signal) {
  return Promise.all(config.backends.map((b) => probeBackend(b, config, signal)))
}

export function findBackend(config, id) {
  const backend = config.backends.find((b) => b.id === id)
  if (!backend) {
    const known = config.backends.map((b) => b.id).join(', ')
    throw new Error(`Unknown backend "${id}". Known backends: ${known}`)
  }
  return backend
}

/**
 * Resolves which backend and model a chat goes to. An explicit backend wins;
 * otherwise the first online backend that serves the model (or, with no model
 * given, the first online backend with any model) is used.
 */
export async function resolveTarget(config, { backend: backendId, model }, signal) {
  if (backendId) {
    const probed = await probeBackend(findBackend(config, backendId), config, signal)
    if (!probed.online) throw new Error(`Backend "${backendId}" is offline: ${probed.error}`)
    const chosen = model || probed.models[0]
    if (!chosen) throw new Error(`Backend "${backendId}" has no model loaded`)
    return { backend: probed, model: chosen }
  }
  const online = (await probeAll(config, signal)).filter((b) => b.online)
  if (online.length === 0) {
    throw new Error('No local backend is reachable. Start Locally Uncensored, Ollama, LM Studio or another backend first.')
  }
  if (model) {
    const host = online.find((b) => b.models.includes(model))
    if (!host) {
      const available = online.map((b) => `${b.id}: ${b.models.join(', ') || '(none)'}`).join('; ')
      throw new Error(`No online backend serves "${model}". Available: ${available}`)
    }
    return { backend: host, model }
  }
  const host = online.find((b) => b.models.length > 0)
  if (!host) throw new Error('Backends are online but none reports a model')
  return { backend: host, model: host.models[0] }
}

/**
 * One non streaming chat completion. Returns the answer text, plus the model's
 * separate reasoning when the backend reports one (llama.cpp, LM Studio, vLLM).
 */
export async function chatCompletion(config, { backend, model, messages, temperature, maxTokens }, signal) {
  const body = { model, messages, stream: false }
  if (temperature !== undefined) body.temperature = temperature
  if (maxTokens !== undefined) body.max_tokens = maxTokens
  const data = await request(`${backend.baseUrl}/chat/completions`, {
    method: 'POST',
    body,
    timeoutMs: config.timeoutMs,
    signal,
  })
  const message = data?.choices?.[0]?.message
  if (!message) throw new Error(`${backend.id} returned no message: ${JSON.stringify(data).slice(0, 300)}`)
  return {
    content: typeof message.content === 'string' ? message.content : '',
    reasoning: message.reasoning_content || message.reasoning || undefined,
    finishReason: data.choices[0].finish_reason,
    usage: data.usage,
  }
}
