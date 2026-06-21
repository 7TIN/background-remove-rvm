import { Hono } from "hono";

const app = new Hono();

app.get("/", (e) => (
    e.json({
        message : "server",
        status : 200,
    })
))

Bun.serve({
  fetch: app.fetch,
  port: 3000,
});

console.log(`server running on http://localhost:${3000}`);
