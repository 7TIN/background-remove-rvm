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

        // Generate unique IDs
        const jobId = crypto.randomUUID();
        const ext = path.extname(file.name) || ".mp4";
        const inputPath = path.join(STORAGE_DIR, `${jobId}_input${ext}`);
        const outputPath = path.join(STORAGE_DIR, `${jobId}_output.webm`);

        // Save uploaded file
        const buffer = Buffer.from(await file.arrayBuffer());
        await writeFile(inputPath, buffer);
        console.log(`[matting:${jobId}] Saved input: ${inputPath} (${buffer.length} bytes)`);

        // Run RVM matting
        const proc = Bun.spawn(
            [PYTHON_BIN, RVM_SCRIPT, inputPath, outputPath],
            {
                stdout: "pipe",
                stderr: "pipe",
                env: { ...process.env, PYTHONUNBUFFERED: "1" },
                cwd: process.cwd(),
            }
        );

        const [stdout, stderr, exitCode] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        if (exitCode !== 0) {
            console.error(`[matting:${jobId}] Failed after ${duration}s:\n${stderr}`);
            // Cleanup input on failure
            await Bun.file(inputPath).delete().catch(() => {});
            return c.json(
                {
                    error: "Matting processing failed",
                    detail: stderr || stdout,
                    jobId,
                    duration: `${duration}s`,
                },
                500
            );
        }

        console.log(`[matting:${jobId}] Success in ${duration}s`);
        console.log(`[matting:${jobId}] stdout:`, stdout.trim());

        // Cleanup input file (keep output)
        await Bun.file(inputPath).delete().catch(() => {});

        return c.json({
            success: true,
            jobId,
            input: { name: file.name, size: buffer.length, type: file.type },
            output: {
                path: outputPath,
                url: `/storage/${jobId}_output.webm`,
            },
            duration: `${duration}s`,
            logs: stdout.trim(),
        });
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