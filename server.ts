import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { serveStatic } from "hono/bun";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const app = new Hono();

// ── Config ───────────────────────────────────────────────────────────────
const PYTHON_BIN = process.env.PYTHON_BIN || path.resolve(process.cwd(), ".venv/Scripts/python.exe");
const STORAGE_DIR = path.resolve(process.cwd(), "storage");
const RVM_SCRIPT = path.resolve(process.cwd(), "rvm_matting.py");
const MAX_BODY_SIZE = 512 * 1024 * 1024; // 512MB
const PROCESS_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes max for CPU processing

// ── Ensure storage dir exists ────────────────────────────────────────────
await mkdir(STORAGE_DIR, { recursive: true });

// ── Validate Python environment (runs once at startup) ─────────────────
async function validatePythonEnvironment() {
    const proc = Bun.spawn(
        [PYTHON_BIN, "-c", "import torch, av, numpy, tqdm; print('RVM deps OK')"],
        { stdout: "pipe", stderr: "pipe" }
    );
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    if (exitCode !== 0) {
        throw new Error(`Python RVM environment invalid:\n${stderr || stdout}`);
    }
    console.log("[startup]", stdout.trim());
}

// ── Run one-time RVM setup (clone repo, download model) ────────────────
async function setupRVM() {
    console.log("[startup] Running RVM one-time setup...");
    const proc = Bun.spawn(
        [PYTHON_BIN, "-c", `
import sys
sys.path.insert(0, ".")
from rvm_matting import setup
setup()
print("RVM setup complete")
        `.trim()],
        { stdout: "pipe", stderr: "pipe", cwd: process.cwd() }
    );
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    if (exitCode !== 0) {
        throw new Error(`RVM setup failed:\n${stderr || stdout}`);
    }
    console.log("[startup]", stdout.trim());
}

// ── Health check ─────────────────────────────────────────────────────────
app.get("/", (c) => c.json({ message: "Clip Farm API", status: 200 }));

// ── Upload + matting endpoint ────────────────────────────────────────────
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
            // Parse multipart upload
            const body = await c.req.parseBody();
            const file = body["video"];

            if (!(file instanceof File)) {
                return c.json({ error: "Expected 'video' field with a file" }, 400);
            }

            // Validate video type
            const allowedTypes = ["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"];
            if (!allowedTypes.includes(file.type)) {
                return c.json({ error: `Unsupported video type: ${file.type}` }, 400);
            }

            const ext = path.extname(file.name) || ".mp4";
            const inputPath = path.join(STORAGE_DIR, `${jobId}_input${ext}`);
            const outputPath = path.join(STORAGE_DIR, `${jobId}_output.webm`);

            // Save uploaded file
            const buffer = Buffer.from(await file.arrayBuffer());
            await writeFile(inputPath, buffer);
            console.log(`[matting:${jobId}] Saved input: ${inputPath} (${buffer.length} bytes)`);

            // ── Run RVM matting with REAL-TIME streaming logs ────────────
            console.log(`[matting:${jobId}] Starting Python RVM process...`);
            console.log(`[matting:${jobId}] WARNING: CPU processing is SLOW. First model load takes ~60s. A 5MB video takes ~5-15 min on CPU.`);

            const proc = Bun.spawn(
                [PYTHON_BIN, RVM_SCRIPT, inputPath, outputPath],
                {
                    stdout: "pipe",
                    stderr: "pipe",
                    env: { 
                        ...process.env, 
                        PYTHONUNBUFFERED: "1",
                        PYTHONIOENCODING: "utf-8",
                    },
                    cwd: process.cwd(),
                }
            );

            // Stream logs in REAL-TIME (don't wait for process to finish)
            const stdoutChunks: string[] = [];
            const stderrChunks: string[] = [];

            const stdoutPromise = (async () => {
                const reader = proc.stdout.getReader();
                const decoder = new TextDecoder();
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const text = decoder.decode(value, { stream: true });
                        stdoutChunks.push(text);
                        // Print immediately
                        process.stdout.write(text);
                    }
                } catch (e) {
                    // Stream closed
                }
            })();

            const stderrPromise = (async () => {
                const reader = proc.stderr.getReader();
                const decoder = new TextDecoder();
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const text = decoder.decode(value, { stream: true });
                        stderrChunks.push(text);
                        // Print immediately to stderr
                        process.stderr.write(text);
                    }
                } catch (e) {
                    // Stream closed
                }
            })();

            // Wait for process with timeout
            const timeout = new Promise<never>((_, reject) => {
                setTimeout(() => {
                    try { proc.kill(); } catch {}
                    reject(new Error(`Processing timed out after ${PROCESS_TIMEOUT_MS/60000} minutes`));
                }, PROCESS_TIMEOUT_MS);
            });

            let exitCode: number;
            try {
                exitCode = await Promise.race([proc.exited, timeout]);
            } catch (timeoutErr: any) {
                await stdoutPromise.catch(() => {});
                await stderrPromise.catch(() => {});
                throw timeoutErr;
            }

            // Wait for streams to finish draining
            await stdoutPromise.catch(() => {});
            await stderrPromise.catch(() => {});

            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            const stdout = stdoutChunks.join("");
            const stderr = stderrChunks.join("");

            if (exitCode !== 0) {
                console.error(`[matting:${jobId}] Failed after ${duration}s (exit ${exitCode})`);
                await Bun.file(inputPath).delete().catch(() => {});
                await Bun.file(outputPath).delete().catch(() => {});
                return c.json(
                    {
                        error: "Matting processing failed",
                        detail: stderr || stdout || "Unknown error",
                        jobId,
                        duration: `${duration}s`,
                        exitCode,
                    },
                    500
                );
            }

            console.log(`[matting:${jobId}] Success in ${duration}s`);

            // Cleanup input file (keep output)
            await Bun.file(inputPath).delete().catch(() => {});

            // Check output exists
            const outputExists = await Bun.file(outputPath).exists();
            if (!outputExists) {
                return c.json({ error: "Output file was not created", jobId }, 500);
            }

            const outputSize = (await Bun.file(outputPath).stat())?.size || 0;

            return c.json({
                success: true,
                jobId,
                input: { name: file.name, size: buffer.length, type: file.type },
                output: {
                    path: outputPath,
                    url: `/storage/${jobId}_output.webm`,
                    size: outputSize,
                },
                duration: `${duration}s`,
            });

        } catch (err: any) {
            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            console.error(`[matting:${jobId}] Exception after ${duration}s:`, err);
            return c.json(
                {
                    error: "Request processing failed",
                    detail: err.message,
                    jobId,
                    duration: `${duration}s`,
                },
                500
            );
        }
    }
);

// ── Serve output files statically ────────────────────────────────────────
app.use("/storage/*", serveStatic({ root: "." }));

// ── Startup ──────────────────────────────────────────────────────────────
const port = Number(process.env.PORT || 3001);

await validatePythonEnvironment();
await setupRVM();

Bun.serve({
    fetch: app.fetch,
    port,
    maxRequestBodySize: MAX_BODY_SIZE,
});

console.log(`Clip Farm API running on http://localhost:${port}`);
console.log(`NOTE: This server uses CPU for video matting. Processing is SLOW (~5-15 min for a 5MB video).`);
console.log(`For production, use a GPU server or consider a lighter model.`);