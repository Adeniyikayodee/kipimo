"""The field agent's tools over MCP, so any MCP-capable agent can drive the live system.

uv run python -m kipimo.agent.mcp_server
"""

import argparse
import inspect
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from kipimo import config
from kipimo.agent import policies
from kipimo.agent.mission import FIELD_PROMPT as PROMPT
from kipimo.agent.mission import Mission
from kipimo.agent.tools import SCHEMAS, Tools
from kipimo.episode import Episode

TYPES = {"string": str, "number": float, "boolean": bool, "integer": int}


def _tool(tools: Tools, schema: dict, left: list[int]):
    """An MCP tool built from the agent's own schema, so the two can never drift apart."""
    name = schema["name"]

    def fn(**args) -> dict:
        if tools.done or left[0] <= 0:
            return {"error": "the mission is over or out of tool calls"}
        left[0] -= 1
        out = tools.call(name, args)
        if name == "report":
            policies.go_home(tools.st)
        return out

    params = [
        inspect.Parameter(
            k,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=list[str] if v["type"] == "array" else TYPES[v["type"]],
        )
        for k, v in schema["input_schema"]["properties"].items()
    ]
    fn.__signature__ = inspect.Signature(params, return_annotation=dict)
    fn.__annotations__ = {p.name: p.annotation for p in params} | {"return": dict}
    fn.__name__ = name
    return fn


def build(scenario: str = "demo_flood", out: Path | None = None) -> tuple[MCPServer, Tools]:
    ep = Episode(config.load(scenario), out=out)
    ep.bus.emit("EPISODE_START", policy="mcp", seed=ep.cfg.seed)
    st = policies.build(ep)
    Mission(st, {})  # the same adjuster, farmer desk and texts as an agent mission, on rules
    tools, left = Tools(st), [ep.cfg.agent.max_tool_calls]
    server = MCPServer("kipimo", instructions=PROMPT)
    for s in SCHEMAS:
        server.add_tool(_tool(tools, s, left), name=s["name"], description=s["description"])
    return server, tools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="demo_flood")
    ap.add_argument("--out", type=Path, default=Path("events.jsonl"))
    a = ap.parse_args()
    build(a.scenario, a.out)[0].run("stdio")


if __name__ == "__main__":
    main()
