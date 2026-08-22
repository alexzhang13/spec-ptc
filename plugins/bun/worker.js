// Worker: executes REPL code in a persistent namespace. llm_query blocks the
// worker (Atomics.wait) while the main thread talks to the Python driver.
// mode "real": force immediately (waiting mode — claim happens Python-side).
// mode "shadow": dispatch immediately, return a lazy Proxy; forcing waits.

const state = { ns: {}, mode: "real" };

function callHost(op, payload) {
  const sab = new SharedArrayBuffer(4 + 4 + 1 << 20); // flag, len, 1MB data
  const flag = new Int32Array(sab, 0, 2);
  postMessage({ op, payload, sab });
  Atomics.wait(flag, 0, 0);                     // block until host writes
  const len = flag[1];
  const bytes = new Uint8Array(sab, 8, len);
  return JSON.parse(new TextDecoder().decode(bytes));
}

function lazyProxy(dispatchId) {
  let forced = null, has = false;
  const force = () => {
    if (!has) { forced = callHost("tool_force", { id: dispatchId }).result; has = true; }
    return forced;
  };
  return new Proxy({}, {
    get(_, prop) {
      if (prop === Symbol.toPrimitive || prop === "toString" || prop === "valueOf")
        return () => force();
      if (prop === "__isSpecValue") return true;
      const v = force();
      const out = v[prop];
      return typeof out === "function" ? out.bind(v) : out;
    },
    has(_, p) { return p in Object(force()); },
  });
}

function makeTools() {
  return {
    llm_query: (prompt) => {
      prompt = String(prompt); // forces proxies flowing into args
      if (state.mode === "shadow") {
        const { id } = callHost("tool_dispatch", { tool: "llm_query", args: [prompt] });
        return lazyProxy(id);
      }
      return callHost("tool_call", { tool: "llm_query", args: [prompt] }).result;
    },
  };
}

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.op === "exec") {
    state.mode = msg.mode || "real";
    let error = null, stdout = [];
    const tools = makeTools();
    const print = (...a) => stdout.push(a.map(String).join(" "));
    try {
      const fn = new Function("ns", "llm_query", "print",
        `with (ns) { ${msg.code}\n } return ns;`);
      fn(state.ns, tools.llm_query, print);
    } catch (e) { error = String(e); }
    postMessage({ op: "exec_done", id: msg.id, error, stdout: stdout.join("\n") });
  } else if (msg.op === "snapshot_ns") {
    let snap = {};
    try { snap = JSON.parse(JSON.stringify(state.ns)); } catch (e) {}
    postMessage({ op: "ns", id: msg.id, ns: snap });
  }
};
