#!/usr/bin/env node
// Entry point Claude Desktop launches: MCP over stdio.
// stdout carries the protocol, so every log line goes to stderr; Claude Desktop
// writes stderr to its log folder (mcp-server-locally-uncensored.log).

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { loadConfig } from './config.js'
import { createServer, SERVER_NAME, SERVER_VERSION } from './server.js'

const config = loadConfig()
const server = createServer(config)
await server.connect(new StdioServerTransport())

console.error(
  `[${SERVER_NAME} ${SERVER_VERSION}] ready. Backends: ${config.backends.map((b) => b.id).join(', ')}; ComfyUI: ${config.comfyUrl}`,
)
