// Small fetch wrapper: timeout, cancellation from the MCP client, and error
// messages that name the URL instead of a bare "fetch failed".

export class HttpError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

/**
 * @param {string} url
 * @param {{ method?: string, body?: unknown, timeoutMs: number, signal?: AbortSignal, raw?: boolean }} opts
 */
export async function request(url, { method = 'GET', body, timeoutMs, signal, raw = false }) {
  const signals = [AbortSignal.timeout(timeoutMs)]
  if (signal) signals.push(signal)
  let res
  try {
    res = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.any(signals),
    })
  } catch (err) {
    if (signal?.aborted) throw new Error(`Request to ${url} was cancelled`)
    if (err?.name === 'TimeoutError') throw new Error(`${url} did not answer within ${timeoutMs} ms`)
    const cause = err?.cause?.code || err?.cause?.message || err?.message
    throw new Error(`Cannot reach ${url} (${cause}). Is the backend running?`)
  }
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.text()).slice(0, 500)
    } catch {
      // body unreadable, the status is enough
    }
    throw new HttpError(`${method} ${url} returned ${res.status}${detail ? `: ${detail}` : ''}`, res.status)
  }
  if (raw) return res
  return res.json()
}
