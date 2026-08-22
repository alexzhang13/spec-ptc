// Main thread: stdin/stdout JSON-lines bridge between the Python driver and
// the worker. Worker blocks on SharedArrayBuffers; we resolve them from
// Python's replies (claim-or-run, speculative dispatch, forcing).
const worker = new Worker(new URL("./worker.js", import.meta.url).href);
const pendingSab = new Map();   // reqId -> sab
let nextReq = 1;

function send(obj) { process.stdout.write(JSON.stringify(obj) + "\n"); }

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.op === "exec_done" || m.op === "ns") { send(m); return; }
  // tool traffic: forward to Python, remember the sab to unblock later
  const reqId = nextReq++;
  pendingSab.set(reqId, m.sab);
  send({ op: m.op, id: reqId, ...m.payload });
};

const dec = new TextDecoder();
let buf = "";
process.stdin.on("data", (chunk) => {
  buf += dec.decode(chunk);
  let nl;
  while ((nl = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
    if (!line.trim()) continue;
    const m = JSON.parse(line);
    if (m.op === "exec" || m.op === "snapshot_ns") { worker.postMessage(m); continue; }
    if (m.op === "reply") {                     // unblock a waiting worker call
      const sab = pendingSab.get(m.id); pendingSab.delete(m.id);
      const flag = new Int32Array(sab, 0, 2);
      const bytes = new TextEncoder().encode(JSON.stringify(m.data));
      new Uint8Array(sab, 8, bytes.length).set(bytes);
      flag[1] = bytes.length;
      Atomics.store(flag, 0, 1);
      Atomics.notify(flag, 0);
    }
  }
});
