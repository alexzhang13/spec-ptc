"""Claude Code plugin: a PreToolUse hook that short-circuits speculated calls."""

import json
import sys

from plugins.client import SpecClient


def main() -> None:
    event = json.load(sys.stdin)  # CC hook payload
    tool = event.get("tool_name", "")
    args = list(event.get("tool_input", {}).values())
    try:
        hit = SpecClient().resolve(tool, args)
    except Exception:
        hit = None  # daemon down -> normal path
    if hit is None:
        print(json.dumps({}))  # proceed: CC runs the tool itself
    else:  # short-circuit with the speculated result
        print(json.dumps({"decision": "block", "reason": f"spec-ptc claimed result: {hit}"}))


if __name__ == "__main__":
    main()
