// Runtime configuration, read from environment variables so the same code works
// from the Claude Desktop developer config (`env` block), an .mcpb bundle
// (`user_config`), or a shell.

/**
 * The local chat backends LU auto-detects, with their default addresses.
 * Mirrors the local entries of PROVIDER_PRESETS in src/api/providers/types.ts.
 * Every backend is reached through its OpenAI compatible `/v1` API; Ollama
 * serves one next to its native API.
 */
export const DEFAULT_BACKENDS = [
  { id: 'builtin', name: 'LU Built-in Engine', baseUrl: 'http://127.0.0.1:8127/v1' },
  { id: 'ollama', name: 'Ollama', baseUrl: 'http://localhost:11434/v1' },
  { id: 'lmstudio', name: 'LM Studio', baseUrl: 'http://localhost:1234/v1' },
  { id: 'vllm', name: 'vLLM', baseUrl: 'http://localhost:8000/v1' },
  { id: 'llamacpp', name: 'llama.cpp / LocalAI / TGI', baseUrl: 'http://localhost:8080/v1' },
  { id: 'koboldcpp', name: 'KoboldCpp', baseUrl: 'http://localhost:5001/v1' },
  { id: 'textgen', name: 'text-generation-webui / TabbyAPI', baseUrl: 'http://localhost:5000/v1' },
  { id: 'jan', name: 'Jan', baseUrl: 'http://localhost:1337/v1' },
  { id: 'gpt4all', name: 'GPT4All', baseUrl: 'http://localhost:4891/v1' },
  { id: 'aphrodite', name: 'Aphrodite', baseUrl: 'http://localhost:2242/v1' },
  { id: 'sglang', name: 'SGLang', baseUrl: 'http://localhost:30000/v1' },
]

const trimSlash = (url) => url.replace(/\/+$/, '')

/**
 * Parses LU_BACKENDS: a comma separated list of `id=url` pairs, e.g.
 * `ollama=http://192.168.1.50:11434/v1,lmstudio=http://localhost:1234/v1`.
 * When set it REPLACES the default list, so only the listed backends are probed.
 */
export function parseBackends(raw) {
  if (!raw || !raw.trim()) return null
  const out = []
  for (const part of raw.split(',')) {
    const entry = part.trim()
    if (!entry) continue
    const eq = entry.indexOf('=')
    if (eq <= 0) throw new Error(`LU_BACKENDS entry "${entry}" is not of the form id=url`)
    const id = entry.slice(0, eq).trim()
    const baseUrl = trimSlash(entry.slice(eq + 1).trim())
    new URL(baseUrl) // throws on a malformed URL, which is what we want at startup
    out.push({ id, name: id, baseUrl })
  }
  return out
}

export function loadConfig(env = process.env) {
  // An empty value, or a `${user_config.x}` placeholder an .mcpb host left
  // unfilled, means "use the default".
  const get = (key) => {
    const v = env[key]?.trim()
    return v && !v.startsWith('${') ? v : undefined
  }
  return {
    backends: parseBackends(get('LU_BACKENDS')) ?? DEFAULT_BACKENDS,
    comfyUrl: trimSlash(get('LU_COMFYUI_URL') ?? 'http://127.0.0.1:8188'),
    // How long a probe waits before calling a backend offline.
    probeTimeoutMs: Number(get('LU_PROBE_TIMEOUT_MS')) || 1500,
    // How long a chat completion or a ComfyUI render may take.
    timeoutMs: Number(get('LU_TIMEOUT_MS')) || 300_000,
    // Images larger than this are returned as a link instead of inline base64,
    // because a multi-megabyte tool result bloats Claude's context.
    maxInlineImageBytes: Number(get('LU_MAX_INLINE_IMAGE_BYTES')) || 1_000_000,
  }
}
