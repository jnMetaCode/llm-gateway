"""验收项：Function Calling 完整回环（两家 provider 各测一遍）。"""

import json

import httpx
import respx

DS_URL = "https://api.deepseek.com/chat/completions"
CL_URL = "https://api.anthropic.com/v1/messages"


def _ds_tool_call_resp():
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": json.dumps({"expression": "2+3*4"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _ds_final_resp():
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": "结果是 14"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 7},
        },
    )


@respx.mock
async def test_deepseek_tool_loop(client):
    route = respx.post(DS_URL).mock(side_effect=[_ds_tool_call_resp(), _ds_final_resp()])
    r = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "算 2+3*4"}], "tools": ["calculator"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == "结果是 14"
    assert body["rounds"] == 2
    assert body["tool_calls"][0]["name"] == "calculator"
    assert body["tool_calls"][0]["result"] == "14"
    assert body["usage"]["input_tokens"] == 35
    assert body["usage"]["output_tokens"] == 12
    assert route.call_count == 2
    # 第二次请求必须带工具结果回填（role=tool）
    second_body = json.loads(route.calls[1].request.content)
    msgs = second_body["messages"]
    assert any(m.get("role") == "tool" and m.get("content") == "14" for m in msgs)


@respx.mock
async def test_claude_tool_loop(client):
    first = httpx.Response(
        200,
        json={
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "weather", "input": {"city": "北京"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 6},
        },
    )
    second = httpx.Response(
        200,
        json={
            "content": [{"type": "text", "text": "北京今天晴，32°C。"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 10},
        },
    )
    route = respx.post(CL_URL).mock(side_effect=[first, second])
    r = await client.post(
        "/v1/chat",
        headers={"X-Provider": "claude"},
        json={"messages": [{"role": "user", "content": "北京天气"}], "tools": ["weather"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "32°C" in body["content"]
    assert body["tool_calls"][0]["result"].startswith("晴")
    # 第二次请求必须带 tool_result block
    second_body = json.loads(route.calls[1].request.content)
    last_msg = second_body["messages"][-1]
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["type"] == "tool_result"


@respx.mock
async def test_unknown_tool_400(client):
    r = await client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "tools": ["rm_rf"]},
    )
    assert r.status_code == 400
    assert "未知工具" in r.json()["detail"]
