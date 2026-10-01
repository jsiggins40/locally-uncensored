#!/usr/bin/env node
// Registers this MCP server in Claude Desktop's developer config
// (Settings > Developer > Edit Config opens the same file).
//
//   node scripts/install-claude-desktop.mjs              add or update the entry
//   node scripts/install-claude-desktop.mjs --dry-run    print the result, write nothing
//   node scripts/install-claude-desktop.mjs --uninstall  remove the entry
//
// Options:
//   --name <key>          key under mcpServers (default: locally-uncensored)
//   --config <path>       config file to edit, overrides the OS default
//   --env KEY=VALUE       env var for the server, repeatable (e.g. LU_COMFYUI_URL=http://127.0.0.1:8189)
//   --with-ms365          also add Softeria's ms-365-mcp-server, read-only
//   --ms365-org           ...in org mode (Teams, SharePoint, shared mailboxes; work accounts)
//   --ms365-write         ...without --read-only
//
// Existing servers and settings in the file are kept, and the previous file is
// saved next to it as claude_desktop_config.json.bak-<timestamp>.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const entryPoint = path.resolve(here, '..', 'src', 'index.js')

function parseArgs(argv) {
  const opts = { name: 'locally-uncensored', env: {}, dryRun: false, uninstall: false, ms365: false, ms365Org: false, ms365Write: false }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    const value = () => {
      const v = argv[++i]
      if (v === undefined) throw new Error(`${arg} needs a value`)
      return v
    }
    if (arg === '--dry-run') opts.dryRun = true
    else if (arg === '--uninstall') opts.uninstall = true
    else if (arg === '--name') opts.name = value()
    else if (arg === '--config') opts.config = value()
    else if (arg === '--with-ms365') opts.ms365 = true
    else if (arg === '--ms365-org') opts.ms365Org = opts.ms365 = true
    else if (arg === '--ms365-write') opts.ms365Write = opts.ms365 = true
    else if (arg === '--env') {
      const pair = value()
      const eq = pair.indexOf('=')
      if (eq <= 0) throw new Error(`--env expects KEY=VALUE, got "${pair}"`)
      opts.env[pair.slice(0, eq)] = pair.slice(eq + 1)
    } else if (arg === '--help' || arg === '-h') {
      console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1, 20).map((l) => l.replace(/^\/\/ ?/, '')).join('\n'))
      process.exit(0)
    } else throw new Error(`Unknown option ${arg} (try --help)`)
  }
  return opts
}

/**
 * Where Claude Desktop reads its config. On Windows the Microsoft Store build
 * runs packaged and its %APPDATA% is redirected into the package folder, so
 * that copy wins when it exists.
 */
export function defaultConfigPath(platform = process.platform, env = process.env, home = os.homedir()) {
  if (platform === 'darwin') return path.join(home, 'Library', 'Application Support', 'Claude', 'claude_desktop_config.json')
  if (platform === 'win32') {
    const local = env.LOCALAPPDATA || path.join(home, 'AppData', 'Local')
    const packages = path.join(local, 'Packages')
    try {
      for (const dir of fs.readdirSync(packages)) {
        if (!/^(AnthropicPBC\.)?Claude_/i.test(dir)) continue
        const candidate = path.join(packages, dir, 'LocalCache', 'Roaming', 'Claude')
        if (fs.existsSync(candidate)) return path.join(candidate, 'claude_desktop_config.json')
      }
    } catch {
      // no Packages folder: regular installer build
    }
    return path.join(env.APPDATA || path.join(home, 'AppData', 'Roaming'), 'Claude', 'claude_desktop_config.json')
  }
  // Linux has no official build; community builds use the XDG location.
  return path.join(env.XDG_CONFIG_HOME || path.join(home, '.config'), 'Claude', 'claude_desktop_config.json')
}

/**
 * npm's npx next to the node running this script. Claude Desktop is a GUI app
 * and on macOS does not inherit the shell PATH (nvm, Homebrew), so a bare
 * "npx" often fails with "spawn npx ENOENT". Absolute paths avoid that.
 */
function findNpxCli() {
  const nodeDir = path.dirname(process.execPath)
  const candidates = [
    path.join(nodeDir, 'node_modules', 'npm', 'bin', 'npx-cli.js'), // Windows
    path.join(nodeDir, '..', 'lib', 'node_modules', 'npm', 'bin', 'npx-cli.js'), // macOS / Linux
  ]
  return candidates.find((p) => fs.existsSync(p))
}

export function buildEntries(opts) {
  const entries = {
    [opts.name]: {
      // Absolute node path for the same PATH reason as findNpxCli.
      command: process.execPath,
      args: [entryPoint],
      ...(Object.keys(opts.env).length ? { env: opts.env } : {}),
    },
  }
  if (opts.ms365) {
    const flags = [...(opts.ms365Org ? ['--org-mode'] : []), ...(opts.ms365Write ? [] : ['--read-only'])]
    const npxCli = findNpxCli()
    entries.ms365 = npxCli
      ? {
          command: process.execPath,
          args: [npxCli, '-y', '@softeria/ms-365-mcp-server', ...flags],
          // npx starts the package through its #!/usr/bin/env node shim, which
          // needs node on PATH as well. Windows installs node on the system
          // PATH, so only macOS / Linux get a short fixed one.
          ...(process.platform === 'win32'
            ? {}
            : { env: { PATH: [path.dirname(process.execPath), '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin'].join(':') } }),
        }
      : { command: 'npx', args: ['-y', '@softeria/ms-365-mcp-server', ...flags] }
  }
  return entries
}

function readConfig(file) {
  if (!fs.existsSync(file)) return {}
  const raw = fs.readFileSync(file, 'utf8')
  if (!raw.trim()) return {}
  try {
    return JSON.parse(raw)
  } catch (err) {
    throw new Error(`${file} is not valid JSON (${err.message}). Fix or move it, then run this again; nothing was changed.`)
  }
}

function main() {
  const opts = parseArgs(process.argv.slice(2))
  const file = opts.config ? path.resolve(opts.config) : defaultConfigPath()
  const config = readConfig(file)
  config.mcpServers ??= {}

  if (opts.uninstall) {
    const removed = [opts.name, ...(opts.ms365 ? ['ms365'] : [])].filter((k) => k in config.mcpServers)
    for (const k of removed) delete config.mcpServers[k]
    if (removed.length === 0) {
      console.log(`Nothing to remove: no "${opts.name}" entry in ${file}`)
      return
    }
    console.log(`Removing ${removed.join(', ')} from ${file}`)
  } else {
    const entries = buildEntries(opts)
    Object.assign(config.mcpServers, entries)
    console.log(`Adding ${Object.keys(entries).join(', ')} to ${file}`)
  }

  const out = `${JSON.stringify(config, null, 2)}\n`
  if (opts.dryRun) {
    console.log('\n--dry-run, would write:\n')
    console.log(out)
    return
  }
  fs.mkdirSync(path.dirname(file), { recursive: true })
  if (fs.existsSync(file)) {
    const backup = `${file}.bak-${new Date().toISOString().replace(/[:.]/g, '-')}`
    fs.copyFileSync(file, backup)
    console.log(`Backup: ${backup}`)
  }
  // Write then rename, so a crash never leaves Claude Desktop a half-written file.
  const tmp = `${file}.tmp-${process.pid}`
  fs.writeFileSync(tmp, out)
  fs.renameSync(tmp, file)
  console.log('\nDone. Quit Claude Desktop completely (tray / menu bar too) and start it again to load the change.')
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main()
  } catch (err) {
    console.error(`Error: ${err.message}`)
    process.exit(1)
  }
}
