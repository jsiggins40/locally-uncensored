# Locally Uncensored MCP server

A local [Model Context Protocol](https://modelcontextprotocol.io) server that lets **Claude Desktop** use the AI that runs on your own machine through Locally Uncensored (LU):

- **Chat backends**: LU's built-in engine, Ollama, LM Studio, vLLM, llama.cpp, KoboldCpp, Jan, GPT4All and the other OpenAI-compatible servers LU detects.
- **ComfyUI**: lists your models, renders images and runs any workflow you export.

The server runs as a local process that Claude Desktop starts over stdio. It only talks to `localhost` (or the URLs you configure). Your prompts to the local models and the images go nowhere else. **The tool results do go back to Claude** (a cloud model), like anything else in a Claude conversation.

## Tools

| Tool | What it does |
|---|---|
| `lu_status` | Which backends and ComfyUI are up, and which models each one serves |
| `list_local_models` | Chat models per backend |
| `local_chat` | Sends a prompt (plus optional system prompt and history) to a local model and returns the answer |
| `comfyui_list_models` | Checkpoints, diffusion models, LoRAs, samplers, schedulers |
| `comfyui_generate_image` | Text to image with an all-in-one checkpoint (SD 1.5, SDXL, Pony…). The image comes back inline |
| `comfyui_run_workflow` | Runs any API-format workflow (Flux, Wan video, your own graphs) |
| `comfyui_get_result` | Picks up a job that was started with `wait: false` |

## Install into Claude Desktop (developer config)

Requirements: Node.js 20 or newer and Claude Desktop on macOS or Windows. LU (or Ollama, LM Studio, ComfyUI…) has to be running when you use the tools; the server itself starts fine without them.

```bash
cd mcp-server
npm install
npm test                      # optional: 12 end-to-end tests against fake backends
npm run install:claude        # writes the entry into claude_desktop_config.json
```

Then **quit Claude Desktop completely** (also from the tray or menu bar) and start it again. The tools appear under the tools/connectors button in the chat box. Claude asks for permission the first time each tool runs.

The installer:

- finds the config file: `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS and `%APPDATA%\Claude\claude_desktop_config.json` on Windows. For the Microsoft Store build it uses the redirected copy under `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\`;
- keeps every other server and setting already in that file, and saves a timestamped `.bak-…` copy first;
- writes the **absolute path of `node`**. Claude Desktop is a GUI app. On macOS it does not see the shell `PATH` that nvm or Homebrew set up, so a bare `"command": "node"` often fails with `spawn node ENOENT`.

Options: `--dry-run` (print, write nothing), `--uninstall`, `--name <key>`, `--config <path>`, `--env KEY=VALUE` (repeatable). Run `node scripts/install-claude-desktop.mjs --help` for the full list.

### By hand instead

Open Claude Desktop → **Settings → Developer → Edit Config**, and add the entry from [`claude_desktop_config.example.json`](claude_desktop_config.example.json) with your real paths (`which node` / `where node`). On Windows, escape the backslashes in JSON: `"C:\\Program Files\\nodejs\\node.exe"`.

### Adding Microsoft 365 alongside (Softeria `ms-365-mcp-server`)

```bash
npm run install:claude -- --with-ms365          # personal account, read-only
npm run install:claude -- --ms365-org           # work/school account: Teams, SharePoint, shared mailboxes
npm run install:claude -- --ms365-write         # drop --read-only (can send mail, edit files…)
```

The entry defaults to `--read-only` on purpose: a model that can send e-mail as you should be an explicit choice. On first use, ask Claude to log in to Microsoft 365. The server then shows a device-code link to sign in. Work tenants may need an admin to consent to the Graph permissions; see that project's README for `--list-permissions` and `--allowed-scopes`.

## Configuration

Set these in the `env` block of the config entry (or with `--env` on the installer):

| Variable | Default | Meaning |
|---|---|---|
| `LU_COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI address (LU's port setting, if you changed it) |
| `LU_BACKENDS` | every LU default port | `id=url` list that **replaces** the default probe list, e.g. `ollama=http://192.168.1.50:11434/v1,lmstudio=http://localhost:1234/v1` |
| `LU_TIMEOUT_MS` | `300000` | Longest a chat answer or a render may take |
| `LU_PROBE_TIMEOUT_MS` | `1500` | How long a backend probe waits before calling it offline |
| `LU_MAX_INLINE_IMAGE_BYTES` | `1000000` | Bigger images come back as a ComfyUI `/view` link instead of inline |

## One-click bundle (.mcpb) instead of the developer config

`manifest.json` describes the same server as a Claude Desktop Extension:

```bash
npm install && npm run pack:mcpb     # → locally-uncensored.mcpb
```

Double-click the file (or drag it into **Settings → Extensions**). Claude Desktop then runs it with its built-in Node and asks for the settings above in a form. The developer config is better while you are changing the code, because an edit only needs a Claude Desktop restart. The bundle is better for handing to someone else.

## Things to know

- **Long renders.** Claude Desktop can give up on a tool call that runs for minutes. The server sends MCP progress notifications while ComfyUI works, but if your renders are slow, ask Claude to use `wait: false` and then `comfyui_get_result`.
- **`comfyui_generate_image` uses the classic SD graph** (checkpoint → CLIP → KSampler → VAE). Flux, Wan and other split models need `comfyui_run_workflow` with a workflow exported via *Workflow → Export (API)*.
- **`comfyui_run_workflow` runs whatever graph it is given.** Some ComfyUI custom nodes can execute code or touch files. Read the workflow in the approval prompt before you allow it.
- **Logs**: Claude Desktop writes the server's stderr to `~/Library/Logs/Claude/mcp-server-locally-uncensored.log` on macOS and `%APPDATA%\Claude\logs\` on Windows.
- **Debug without Claude**: `npm run inspect` opens the MCP Inspector against the server.
