"""Anthropic Claude 适配层（Messages API）。

与 OpenAI 协议的三个关键差异（面试常问）：
1. system 不在 messages 里，是独立字段；
2. 工具调用/结果都是 content block（tool_use / tool_result），不是独立角色；
3. 流式是命名事件（message_start / content_block_delta / message_delta ...）。
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import Settings
from ..schemas import Message
from .base import Provider, ProviderReply, ToolCallReq, UpstreamError

_API_VERSION = "2023-06-01"


def _raise_for_status(status: int, body: str) -> None:
    if status == 429 or status >= 500:
        raise UpstreamError(f"Claude {status}: {body[:200]}", retryable=True, status=status)
    if status >= 400:
        raise UpstreamError(f"Claude {status}: {body[:200]}", retryable=False, status=status)


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.model = settings.anthropic_model

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": _API_VERSION,
        }

    def init_history(self, messages: list[Message]) -> dict[str, Any]:
        system = "\n".join(m.content for m in messages if m.role == "system")
        msgs = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        return {"system": system, "messages": msgs}

    def _payload(self, history: dict[str, Any], max_tokens: int, temperature: float) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": history["messages"],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if history["system"]:
            payload["system"] = history["system"]
        return payload

    async def chat(
        self, history: Any, tool_specs: list[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> ProviderReply:
        payload = self._payload(history, max_tokens, temperature)
        if tool_specs:
            payload["tools"] = [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "input_schema": s["parameters"],
                }
                for s in tool_specs
            ]
        try:
            r = await self._client.post(
                f"{self._settings.anthropic_base_url}/v1/messages",
                json=payload,
                headers=self._headers,
                timeout=self._settings.upstream_timeout,
            )
        except httpx.TimeoutException as e:
            raise UpstreamError("Claude 超时", retryable=True) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Claude 网络错误: {e}", retryable=True) from e
        _raise_for_status(r.status_code, r.text)
        data = r.json()
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [
            ToolCallReq(id=b["id"], name=b["name"], arguments=b.get("input") or {})
            for b in data["content"]
            if b["type"] == "tool_use"
        ]
        usage = data.get("usage") or {}
        return ProviderReply(
            text=text,
            tool_calls=calls,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw=data["content"],
        )

    def append_tool_exchange(
        self, history: Any, reply: ProviderReply, results: list[tuple[ToolCallReq, str]]
    ) -> None:
        history["messages"].append({"role": "assistant", "content": reply.raw})
        history["messages"].append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": result}
                    for call, result in results
                ],
            }
        )

    async def stream(
        self, history: Any, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._payload(history, max_tokens, temperature)
        payload["stream"] = True
        input_tokens = 0
        try:
            async with self._client.stream(
                "POST",
                f"{self._settings.anthropic_base_url}/v1/messages",
                json=payload,
                headers=self._headers,
                timeout=self._settings.upstream_timeout,
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")
                    _raise_for_status(r.status_code, body)
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    etype = event.get("type")
                    if etype == "message_start":
                        input_tokens = event["message"].get("usage", {}).get("input_tokens", 0)
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield {"type": "delta", "content": delta["text"]}
                    elif etype == "message_delta":
                        out = event.get("usage", {}).get("output_tokens", 0)
                        yield {"type": "usage", "input": input_tokens, "output": out}
                    elif etype == "message_stop":
                        break
        except httpx.TimeoutException as e:
            raise UpstreamError("Claude 流式超时", retryable=True) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Claude 流式网络错误: {e}", retryable=True) from e
