"""验收项：token 计费统计 / pydantic 校验 / 流式 SSE。"""

import httpx
import respx

DS_URL = "https://api.deepseek.com/chat/completions"


@respx.mock
async def test_usage_accumulates(client):
    respx.post(DS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "你好"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 2000},
            },
        )
    )
    await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    r = await client.get("/v1/usage")
    data = r.json()
    row = data["by_model"]["deepseek-chat"]
    assert row["requests"] == 1
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 2000
    # 0.27/M * 1000 + 1.10/M * 2000 = 0.00027 + 0.0022
    assert abs(data["total_cost_usd"] - 0.002470) < 1e-6


async def test_validation_422(client):
    r = await client.post("/v1/chat", json={"messages": []})
    assert r.status_code == 422
    r = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": ""}]})
    assert r.status_code == 422


async def test_unknown_provider_400(client):
    r = await client.post(
        "/v1/chat",
        headers={"X-Provider": "gpt"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400


@respx.mock
async def test_stream_sse(client):
    sse = (
        'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(DS_URL).mock(
        return_value=httpx.Response(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )
    )
    async with client.stream(
        "POST", "/v1/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    ) as r:
        assert r.status_code == 200
        body = (await r.aread()).decode()
    assert '"content": "你"' in body or '"content":"你"' in body.replace(" ", "")
    assert '"type": "done"' in body.replace('":"', '": "') or "done" in body
    # 流式的用量也要入账
    usage = (await client.get("/v1/usage")).json()
    assert usage["by_model"]["deepseek-chat"]["input_tokens"] == 5
