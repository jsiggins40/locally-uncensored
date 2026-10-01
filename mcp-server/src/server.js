// The MCP server: tool definitions on top of backends.js and comfyui.js.
// Kept separate from index.js so tests can connect it to an in-memory transport.

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'
import { probeAll, probeBackend, findBackend, resolveTarget, chatCompletion } from './backends.js'
import {
  comfyStatus,
  listComfyModels,
  buildTxt2ImgWorkflow,
  submitWorkflow,
  waitForJob,
  checkJob,
  fetchOutput,
  viewUrl,
} from './comfyui.js'

export const SERVER_NAME = 'locally-uncensored'
export const SERVER_VERSION = '0.1.0'

const READ_ONLY = { readOnlyHint: true, openWorldHint: false }
const ACTS_LOCALLY = { readOnlyHint: false, destructiveHint: false, openWorldHint: false }

const text = (value) => ({ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) })

/** Wraps a handler so a thrown error becomes an MCP tool error the model can read. */
const safe = (fn) => async (args, extra) => {
  try {
    return await fn(args, extra)
  } catch (err) {
    return { isError: true, content: [text(err?.message ?? String(err))] }
  }
}

/** Sends MCP progress notifications when the client asked for them. */
function progressReporter(extra) {
  const token = extra?._meta?.progressToken
  if (token === undefined) return undefined
  return async (progress, message) => {
    try {
      await extra.sendNotification({ method: 'notifications/progress', params: { progressToken: token, progress, message } })
    } catch {
      // A lost progress update must never fail the render.
    }
  }
}

/**
 * SD 1.5 checkpoints render 512 px natively and tile into duplicates at 1024;
 * SDXL-family checkpoints are the other way round. Guessed from the file name,
 * the caller can always pass an explicit size.
 */
export function defaultSize(checkpoint) {
  return /xl|pony|illustrious|noob|playground|juggernaut|sd3/i.test(checkpoint) ? 1024 : 512
}

async function renderOutputs(config, promptId, outputs, signal) {
  if (outputs.length === 0) {
    return { content: [text(`ComfyUI job ${promptId} finished but produced no files. Does the workflow end in a Save node?`)] }
  }
  const content = []
  const files = []
  for (const file of outputs) {
    const entry = { filename: file.filename, subfolder: file.subfolder, type: file.type, url: viewUrl(config, file) }
    files.push(entry)
    if (!/\.(png|jpe?g|webp|gif)$/i.test(file.filename)) continue
    const fetched = await fetchOutput(config, file, signal)
    if (fetched.tooLarge) {
      entry.inline = `skipped, ${fetched.bytes} bytes is over LU_MAX_INLINE_IMAGE_BYTES`
      continue
    }
    content.push({ type: 'image', data: fetched.data, mimeType: fetched.mimeType })
  }
  content.push(text({ prompt_id: promptId, files, note: 'Files are saved in the ComfyUI output folder; url opens them while ComfyUI runs.' }))
  return { content }
}

export function createServer(config) {
  const server = new McpServer(
    { name: SERVER_NAME, version: SERVER_VERSION },
    {
      instructions:
        'Gives access to AI models running locally on this computer through Locally Uncensored: ' +
        'chat backends (LU built-in engine, Ollama, LM Studio, llama.cpp and more) and ComfyUI for images. ' +
        'Call lu_status first to see what is running. Everything stays on this machine.',
    },
  )

  server.registerTool(
    'lu_status',
    {
      title: 'Local AI status',
      description: 'Shows which local chat backends and ComfyUI are running, with the models each one serves.',
      inputSchema: {},
      annotations: READ_ONLY,
    },
    safe(async (_args, extra) => {
      const [backends, comfyui] = await Promise.all([probeAll(config, extra.signal), comfyStatus(config, extra.signal)])
      return {
        content: [
          text({
            backends: backends.map((b) => (b.online
              ? { id: b.id, name: b.name, url: b.baseUrl, online: true, models: b.models }
              : { id: b.id, name: b.name, url: b.baseUrl, online: false })),
            comfyui,
          }),
        ],
      }
    }),
  )

  server.registerTool(
    'list_local_models',
    {
      title: 'List local chat models',
      description: 'Lists the chat models served by the online local backends, or by one backend when `backend` is given.',
      inputSchema: {
        backend: z.string().optional().describe('Backend id from lu_status, e.g. "ollama" or "builtin". Omit for all.'),
      },
      annotations: READ_ONLY,
    },
    safe(async ({ backend }, extra) => {
      const probed = backend
        ? [await probeBackend(findBackend(config, backend), config, extra.signal)]
        : (await probeAll(config, extra.signal)).filter((b) => b.online)
      if (probed.length === 0) return { content: [text('No local backend is online.')] }
      return {
        content: [text(probed.map((b) => (b.online
          ? { backend: b.id, models: b.models }
          : { backend: b.id, online: false, error: b.error })))],
      }
    }),
  )

  server.registerTool(
    'local_chat',
    {
      title: 'Ask a local model',
      description:
        'Sends a prompt to a model running locally (nothing leaves this computer) and returns its answer. ' +
        'Without `model` and `backend` the first online backend and its first model are used.',
      inputSchema: {
        prompt: z.string().min(1).describe('The user message to send.'),
        system: z.string().optional().describe('Optional system prompt.'),
        model: z.string().optional().describe('Model id as listed by list_local_models.'),
        backend: z.string().optional().describe('Backend id as listed by lu_status.'),
        history: z
          .array(z.object({ role: z.enum(['user', 'assistant']), content: z.string() }))
          .optional()
          .describe('Earlier turns of the conversation, oldest first, for a multi-turn exchange.'),
        temperature: z.number().min(0).max(2).optional(),
        max_tokens: z.number().int().positive().optional(),
      },
      annotations: ACTS_LOCALLY,
    },
    safe(async (args, extra) => {
      const target = await resolveTarget(config, args, extra.signal)
      const messages = []
      if (args.system) messages.push({ role: 'system', content: args.system })
      messages.push(...(args.history ?? []), { role: 'user', content: args.prompt })
      const result = await chatCompletion(
        config,
        { backend: target.backend, model: target.model, messages, temperature: args.temperature, maxTokens: args.max_tokens },
        extra.signal,
      )
      const meta = { backend: target.backend.id, model: target.model, finish_reason: result.finishReason, usage: result.usage }
      const content = [text(result.content || '(the model returned an empty answer)')]
      if (result.reasoning) content.push(text(`Reasoning:\n${result.reasoning}`))
      content.push(text(meta))
      return { content }
    }),
  )

  server.registerTool(
    'comfyui_list_models',
    {
      title: 'List ComfyUI models',
      description: 'Lists the checkpoints, diffusion models, LoRAs, samplers and schedulers installed in ComfyUI.',
      inputSchema: {},
      annotations: READ_ONLY,
    },
    safe(async (_args, extra) => ({ content: [text(await listComfyModels(config, extra.signal))] })),
  )

  server.registerTool(
    'comfyui_generate_image',
    {
      title: 'Generate an image locally',
      description:
        'Renders an image with ComfyUI from a text prompt using an all-in-one checkpoint (SD 1.5, SDXL, Pony and similar). ' +
        'Split models such as Flux or Wan need their own graph: use comfyui_run_workflow. ' +
        'Size defaults to 1024 for SDXL-family checkpoints and 512 otherwise.',
      inputSchema: {
        prompt: z.string().min(1),
        negative_prompt: z.string().optional(),
        checkpoint: z.string().optional().describe('Checkpoint file from comfyui_list_models. Defaults to the first one.'),
        width: z.number().int().min(64).max(4096).multipleOf(8).optional(),
        height: z.number().int().min(64).max(4096).multipleOf(8).optional(),
        steps: z.number().int().min(1).max(150).default(25),
        cfg: z.number().min(0).max(30).default(7),
        sampler: z.string().default('euler'),
        scheduler: z.string().default('normal'),
        seed: z.number().int().min(0).optional(),
        wait: z
          .boolean()
          .default(true)
          .describe('false returns the prompt_id right away; fetch the image later with comfyui_get_result.'),
      },
      annotations: ACTS_LOCALLY,
    },
    safe(async (args, extra) => {
      let checkpoint = args.checkpoint
      if (!checkpoint) {
        const { checkpoints } = await listComfyModels(config, extra.signal)
        checkpoint = checkpoints[0]
        if (!checkpoint) throw new Error('ComfyUI has no checkpoint installed. Install one in the LU Model Manager first.')
      }
      const size = defaultSize(checkpoint)
      const { workflow, seed } = buildTxt2ImgWorkflow({
        prompt: args.prompt,
        negativePrompt: args.negative_prompt,
        checkpoint,
        width: args.width ?? size,
        height: args.height ?? size,
        steps: args.steps,
        cfg: args.cfg,
        sampler: args.sampler,
        scheduler: args.scheduler,
        seed: args.seed,
      })
      const promptId = await submitWorkflow(config, workflow, extra.signal)
      const settings = { checkpoint, seed, width: args.width ?? size, height: args.height ?? size }
      if (!args.wait) return { content: [text({ prompt_id: promptId, status: 'queued', ...settings })] }
      const progress = progressReporter(extra)
      const outputs = await waitForJob(config, promptId, {
        signal: extra.signal,
        onTick: progress && ((n) => progress(n, `Rendering (${n} s)`)),
      })
      const result = await renderOutputs(config, promptId, outputs, extra.signal)
      result.content.push(text(settings))
      return result
    }),
  )

  server.registerTool(
    'comfyui_run_workflow',
    {
      title: 'Run a ComfyUI workflow',
      description:
        'Runs any ComfyUI workflow in API format (ComfyUI: Workflow > Export (API)). Use it for Flux, video and other graphs ' +
        'that comfyui_generate_image does not cover. Returns the produced files.',
      inputSchema: {
        workflow: z.record(z.string(), z.any()).describe('The API-format workflow: an object of node id to { class_type, inputs }.'),
        wait: z.boolean().default(true),
      },
      annotations: ACTS_LOCALLY,
    },
    safe(async ({ workflow, wait }, extra) => {
      const promptId = await submitWorkflow(config, workflow, extra.signal)
      if (!wait) return { content: [text({ prompt_id: promptId, status: 'queued' })] }
      const progress = progressReporter(extra)
      const outputs = await waitForJob(config, promptId, {
        signal: extra.signal,
        onTick: progress && ((n) => progress(n, `Running (${n} s)`)),
      })
      return renderOutputs(config, promptId, outputs, extra.signal)
    }),
  )

  server.registerTool(
    'comfyui_get_result',
    {
      title: 'Get a ComfyUI result',
      description: 'Checks a ComfyUI job started with wait=false and returns its files once it has finished.',
      inputSchema: { prompt_id: z.string().min(1) },
      annotations: READ_ONLY,
    },
    safe(async ({ prompt_id }, extra) => {
      const job = await checkJob(config, prompt_id, extra.signal)
      if (!job.done) return { content: [text({ prompt_id, status: 'running or queued' })] }
      return renderOutputs(config, prompt_id, job.outputs, extra.signal)
    }),
  )

  return server
}
