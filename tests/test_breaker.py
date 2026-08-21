"""验收项：重试（429/5xx 指数退避 ×2）与熔断（3 连败 -> 30s 内 503 降级）。"""

import httpx
import respx

DS_URL = "https://api.deepseek.com/chat/completions"


@respx.mock
async def test_retry_then_breaker_opens(client):
    route = respx.post(DS_URL).mock(return_value=httpx.Response(500, text="boom"))
    # 第 1 个请求：1 原始 + 2 重试 = 3 次尝试全失败 -> 502，同时熔断器计满 3 次开闸
    r1 = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r1.status_code == 502
    assert r1.json()["code"] == "upstream_error"
    assert route.call_count == 3
    # 第 2 个请求：熔断开启，不打上游，直接 503 降级
    r2 = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r2.status_code == 503
    assert r2.json()["code"] == "circuit_open"
    assert route.call_count == 3  # 上游没有收到新请求


@respx.mock
async def test_4xx_no_retry(client):
    """400 类错误是我们的问题，重试无意义：只打一次，不计熔断。"""
    route = respx.post(DS_URL).mock(return_value=httpx.Response(400, text="bad request"))
    r = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert route.call_count == 1
    # 熔断未触发：再来一次仍然打到上游
    await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert route.call_count == 2


@respx.mock
async def test_breaker_isolated_per_provider(client):
    """deepseek 熔断不应影响 claude 通道。"""
    respx.post(DS_URL).mock(return_value=httpx.Response(500, text="boom"))
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    r = await client.post(
        "/v1/chat",
        headers={"X-Provider": "claude"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
