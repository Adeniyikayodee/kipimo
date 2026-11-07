import asyncio

from kipimo.agent.mcp_server import build
from kipimo.agent.tools import SCHEMAS


def test_mcp_exposes_the_same_tools():
    server, tools = build()

    async def go():
        listed = {t.name for t in await server.list_tools()}
        res = await server.call_tool("score_payout", {"parcel_id": "P07"})
        return listed, res

    listed, res = asyncio.run(go())
    assert listed == {s["name"] for s in SCHEMAS}
    assert "P07" in str(res)
    assert tools.st.ep.bus.events[-1]["kind"] == "EPISODE_START"


def test_mcp_claims_face_the_adjuster_and_a_budget():
    server, tools = build()
    assert tools.st.engine.reviewer is not None

    async def spend():
        n = tools.st.ep.cfg.agent.max_tool_calls + 1
        return [await server.call_tool("get_situation", {}) for _ in range(n)]

    assert "out of tool calls" in str(asyncio.run(spend())[-1])
