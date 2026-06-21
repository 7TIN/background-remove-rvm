import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { serveStatic } from "hono/bun";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const app = new Hono();

// ── Config ───────────────────────────────────────────────────────────────
const STORAGE_DIR = path.resolve(process.cwd(), "storage");
const MAX_BODY_SIZE = 512 * 1024 * 1024;
const RVM_API_URL = process.env.RVM_API_URL?.replace(/\/$/, "");

if (!RVM_API_URL) {
    console.error("❌ ERROR: RVM_API_URL env var not set!");
    console.error("   Add to .env: RVM_API_URL=https://xxxx.ngrok-free.app");
    process.exit(1);
}

console.log(`[config] Remote GPU server: ${RVM_API_URL}`);

await mkdir(STORAGE_DIR, { recursive: true });

// ── Health check ─────────────────────────────────────────────────────────
app.get("/", (c) => c.json({
    message: "Clip Farm API",
    status: 200,
    rvm_server: RVM_API_URL,
}));

// ── GPU health proxy ─────────────────────────────────────────────────────
app.get("/gpu-health", async (c) => {
    try {
        const res = await fetch(`${RVM_API_URL}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json() as Record<string, unknown>;
        return c.json({ ok: res.ok, ...data });
    } catch (e: any) {
        return c.json({ ok: false, error: e.message }, 502);
    }
});

// ── Matting endpoint ──────────────────────────────────────────────────────
app.post(
    "/matting",
    bodyLimit({
        maxSize: MAX_BODY_SIZE,
        onError: (c) => c.json({ error: "File too large (max 512MB)" }, 413),
    }),
    async (c) => {
        const startTime = Date.now();
        const jobId = crypto.randomUUID();

        try {
            // ── Parse incoming file ──────────────────────────────────────
            const body = await c.req.parseBody();
            const file = body["video"];

            if (!(file instanceof File)) {
                return c.json({ error: "Expected 'video' field with a file" }, 400);
            }

            const allowed = ["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"];
            if (!allowed.includes(file.type)) {
                return c.json({ error: `Unsupported type: ${file.type}` }, 400);
            }

            const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
            console.log(`\n[matting:${jobId}] ▶ START`);
            console.log(`[matting:${jobId}] File: ${file.name} (${fileSizeMB} MB, ${file.type})`);
            console.log(`[matting:${jobId}] Forwarding to GPU: ${RVM_API_URL}/matting`);

            // ── Build FormData correctly ──────────────────────────────────
            // Re-create FormData with a proper File object so boundary is fresh.
            // Do NOT re-use the parsed File directly — Bun can lose the boundary.
            const fileBuffer = await file.arrayBuffer();
            const freshFile = new File([fileBuffer], file.name, { type: file.type });

            const formData = new FormData();
            formData.append("video", freshFile);

            // ── Heartbeat logger while we wait ────────────────────────────
            const heartbeat = setInterval(() => {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
                console.log(`[matting:${jobId}] ⏳ Still processing... ${elapsed}s elapsed`);
            }, 10_000);

            // 10 minute timeout (CPU is very slow)
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 10 * 60 * 1000);

            console.log(`[matting:${jobId}] Request sent, waiting for GPU response...`);

            let response: Response;
            try {
                response = await fetch(`${RVM_API_URL}/matting`, {
                    method: "POST",
                    body: formData,
                    signal: controller.signal,
                    // Do NOT set Content-Type header — let fetch set it with the boundary
                });
            } finally {
                clearTimeout(timeout);
                clearInterval(heartbeat);
            }

            if (!response.ok) {
                const error = await response.text();
                console.error(`[matting:${jobId}] ❌ GPU error ${response.status}: ${error}`);
                return c.json({ error: "GPU failed", detail: error, status: response.status }, 500);
            }

            // ── Save result ───────────────────────────────────────────────
            const outputPath = path.join(STORAGE_DIR, `${jobId}_output.webm`);
            const buffer = Buffer.from(await response.arrayBuffer());
            await writeFile(outputPath, buffer);

            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            console.log(`[matting:${jobId}] ✅ Done in ${duration}s — ${(buffer.length / 1024 / 1024).toFixed(2)} MB output`);

            return c.json({
                success: true,
                jobId,
                input: { name: file.name, size: file.size, type: file.type },
                output: {
                    url: `/storage/${jobId}_output.webm`,
                    size: buffer.length,
                },
                duration: `${duration}s`,
            });

        } catch (err: any) {
            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            if (err.name === "AbortError") {
                console.error(`[matting:${jobId}] ❌ Timeout after ${duration}s`);
                return c.json({ error: "GPU timed out (10 min limit)", jobId, duration: `${duration}s` }, 504);
            }
            console.error(`[matting:${jobId}] ❌ Failed after ${duration}s:`, err.message);
            return c.json({ error: "Processing failed", detail: err.message, jobId, duration: `${duration}s` }, 500);
        }
    }
);

app.use("/storage/*", serveStatic({ root: "." }));

const port = Number(process.env.PORT || 3001);
Bun.serve({
    fetch: app.fetch,
    port,
    maxRequestBodySize: MAX_BODY_SIZE,
});

console.log(`\n✅ Server: http://localhost:${port}`);
console.log(`🔗 GPU:    ${RVM_API_URL}`);
console.log(`🩺 Check GPU health: http://localhost:${port}/gpu-health\n`);