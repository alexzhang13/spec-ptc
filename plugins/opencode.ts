// OpenCode plugin: resolve tool calls against the spec-ptc daemon (miss -> run normally).
import * as net from "node:net";

function rpc(msg: object, socket = "/tmp/spec-ptc.sock"): Promise<any> {
  return new Promise((resolve, reject) => {
    const c = net.createConnection(socket, () => c.write(JSON.stringify(msg) + "\n"));
    let buf = "";
    c.on("data", (d) => {
      buf += d.toString();
      const nl = buf.indexOf("\n");
      if (nl >= 0) { c.end(); resolve(JSON.parse(buf.slice(0, nl))); }
    });
    c.on("error", reject);
  });
}

export const specPtc = {
  turnBegin: (vars: object) => rpc({ op: "turn_begin", vars }),
  feed: (delta: string) => rpc({ op: "feed", delta }),
  // in your tool executor:  const hit = await specPtc.resolve("llm_query", [prompt]);
  resolve: async (tool: string, args: any[]) => {
    const r = await rpc({ op: "resolve", tool, args });
    return r.status === "hit" ? r.result : null;
  },
  turnEnd: () => rpc({ op: "turn_end" }),
};
