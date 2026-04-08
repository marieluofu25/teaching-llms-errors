import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import type { Connect, Plugin, ViteDevServer } from 'vite'
import { defineConfig } from 'vite'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const PIPELINE_MOUNT = '/pipeline_2026'

function contentType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.json') return 'application/json'
  if (ext === '.png') return 'image/png'
  if (ext === '.html') return 'text/html; charset=utf-8'
  if (ext === '.csv') return 'text/csv; charset=utf-8'
  if (ext === '.svg') return 'image/svg+xml'
  return 'application/octet-stream'
}

function attachPipelineStatic(middlewares: Connect.Server) {
  const root = path.resolve(__dirname, '../../pipeline_2026')
  middlewares.use(PIPELINE_MOUNT, (req, res, next) => {
    const raw = (req.url ?? '/').split('?')[0]
    let rel = ''
    try {
      rel = decodeURIComponent(raw)
    } catch {
      res.statusCode = 400
      res.end('Bad request')
      return
    }
    rel = rel.replace(/^\/+/, '')
    if (rel.includes('\0')) {
      res.statusCode = 400
      res.end('Bad request')
      return
    }

    const base = path.resolve(root)
    const resolved = path.resolve(base, rel)
    if (resolved !== base && !resolved.startsWith(base + path.sep)) {
      res.statusCode = 403
      res.end('Forbidden')
      return
    }

    fs.stat(resolved, (err, st) => {
      if (err || !st.isFile()) {
        res.statusCode = 404
        res.end('Not found')
        return
      }
      res.setHeader('Content-Type', contentType(resolved))
      fs.createReadStream(resolved).pipe(res)
    })
  })
}

function pipelineDataPlugin(): Plugin {
  return {
    name: 'pipeline-data',
    configureServer(server: ViteDevServer) {
      attachPipelineStatic(server.middlewares)
    },
    configurePreviewServer(server) {
      attachPipelineStatic(server.middlewares)
    },
  }
}

export default defineConfig({
  plugins: [react(), pipelineDataPlugin()],
})
