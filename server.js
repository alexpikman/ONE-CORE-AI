import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(ROOT, "public");
const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 5772);
const OLLAMA_HOST = (process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/$/, "");
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || "llama3.2";
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 120000);
const MAX_REQUEST_BYTES = Number(process.env.MAX_REQUEST_BYTES || 1024 * 1024);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon"
};

function sendJson(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  });
  res.end(JSON.stringify(data));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", chunk => {
      body += chunk;
      if (Buffer.byteLength(body) > MAX_REQUEST_BYTES) {
        reject(Object.assign(new Error("Request body is too large"), { statusCode: 413 }));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(Object.assign(new Error("Invalid JSON body"), { statusCode: 400 }));
      }
    });
    req.on("error", reject);
  });
}

async function ollamaFetch(endpoint, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(`${OLLAMA_HOST}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) }
    });
  } finally {
    clearTimeout(timer);
  }
}

async function getOllamaStatus() {
  try {
    const response = await ollamaFetch("/api/tags");
    if (!response.ok) throw new Error(`Ollama returned HTTP ${response.status}`);
    const data = await response.json();
    return { connected: true, models: (data.models || []).map(model => ({
      name: model.name,
      size: model.size,
      modified_at: model.modified_at,
      details: model.details || {}
    })) };
  } catch (error) {
    return { connected: false, models: [], error: error.name === "AbortError" ? "Ollama request timed out" : error.message };
  }
}

async function handleChat(req, res) {
  const body = await readJson(req);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const model = typeof body.model === "string" && body.model.trim() ? body.model.trim() : DEFAULT_MODEL;
  if (!messages.length || messages.some(item => !item || !["system", "user", "assistant"].includes(item.role) || typeof item.content !== "string")) {
    return sendJson(res, 400, { error: "messages must contain valid role/content objects" });
  }

  let upstream;
  try {
    upstream = await ollamaFetch("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        options: {
          temperature: Number.isFinite(body.temperature) ? body.temperature : 0.7
        }
      })
    });
  } catch (error) {
    return sendJson(res, 502, { error: error.name === "AbortError" ? "Ollama timed out" : `Cannot reach Ollama at ${OLLAMA_HOST}: ${error.message}` });
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return sendJson(res, upstream.status || 502, { error: detail || `Ollama returned HTTP ${upstream.status}` });
  }

  res.writeHead(200, {
    "Content-Type": "application/x-ndjson; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "X-Content-Type-Options": "nosniff"
  });

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      for (const line of text.split("\n")) {
        if (!line.trim()) continue;
        let chunk;
        try { chunk = JSON.parse(line); } catch { continue; }
        const event = chunk.message?.content
          ? { type: "token", content: chunk.message.content }
          : chunk.error
            ? { type: "error", error: chunk.error }
            : null;
        if (event) res.write(JSON.stringify(event) + "\n");
        if (chunk.done) {
          res.write(JSON.stringify({ type: "done", model: chunk.model || model, total_duration: chunk.total_duration }) + "\n");
        }
      }
    }
  } catch (error) {
    if (!res.destroyed) res.write(JSON.stringify({ type: "error", error: error.message }) + "\n");
  } finally {
    if (!res.destroyed) res.end();
    reader.releaseLock();
  }
}

async function serveStatic(req, res, pathname) {
  const requested = pathname === "/" ? "/index.html" : decodeURIComponent(pathname);
  const resolved = path.resolve(PUBLIC_DIR, "." + requested);
  if (!resolved.startsWith(PUBLIC_DIR + path.sep) && resolved !== path.join(PUBLIC_DIR, "index.html")) {
    return sendJson(res, 403, { error: "Forbidden" });
  }
  try {
    const data = await readFile(resolved);
    res.writeHead(200, {
      "Content-Type": mimeTypes[path.extname(resolved)] || "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
      "Cache-Control": "no-cache"
    });
    res.end(data);
  } catch {
    sendJson(res, 404, { error: "Not found" });
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  try {
    if (req.method === "GET" && url.pathname === "/api/health") {
      const status = await getOllamaStatus();
      return sendJson(res, 200, {
        ok: true,
        app: "ONE CORE AI",
        port: PORT,
        provider: "ollama",
        model: DEFAULT_MODEL,
        ...status
      });
    }
    if (req.method === "GET" && url.pathname === "/api/models") {
      const status = await getOllamaStatus();
      return sendJson(res, status.connected ? 200 : 503, status);
    }
    if (req.method === "POST" && url.pathname === "/api/chat") {
      return await handleChat(req, res);
    }
    if (req.method === "GET" && !url.pathname.startsWith("/api/")) {
      return await serveStatic(req, res, url.pathname);
    }
    sendJson(res, 404, { error: "API route not found" });
  } catch (error) {
    if (!res.headersSent) sendJson(res, error.statusCode || 500, { error: error.message || "Internal server error" });
    else if (!res.destroyed) res.end();
  }
});

server.listen(PORT, HOST, () => {
  console.log(`ONE CORE AI listening at http://${HOST}:${PORT}`);
  console.log(`Ollama endpoint: ${OLLAMA_HOST}`);
});
