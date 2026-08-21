"""DeepSeek 适配层（OpenAI 兼容协议）。"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import Settings
from ..schemas import Message
from .base import Provider, ProviderReply, ToolCallReq, UpstreamError


def _raise_for_status(status: int, body: str) -> None:
    if status == 429 or status >= 500:
        raise UpstreamError(f"DeepSeek {status}: {body[:200]}", retryable=True, status=status)
    if status >= 400:
        raise UpstreamError(f"DeepSeek {status}: {body[:200]}", retryable=False, status=status)


class DeepSeekProvider(Provider):
    name = "deepseek"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.model = settings.deepseek_model

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._settings.deepseek_api_key}"}

    def init_history(self, messages: list[Message]) -> list[dict[str, Any]]:
        return [m.model_dump() for m in messages]

    async def chat(
        self, history: Any, tool_specs: list[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> ProviderReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": history,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tool_specs:
            payload["tools"] = [{"type": "function", "function": s} for s in tool_specs]
        try:
            r = await self._client.post(
                f"{self._settings.deepseek_base_url}/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=self._settings.upstream_timeout,
            )
        except httpx.TimeoutException as e:
            raise UpstreamError("DeepSeek 超时", retryable=True) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"DeepSeek 网络错误: {e}", retryable=True) from e
        _raise_for_status(r.status_code, r.text)
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = [
            ToolCallReq(
                id=c["id"],
                name=c["function"]["name"],
                arguments=json.loads(c["function"]["arguments"] or "{}"),
            )
            for c in (msg.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return ProviderReply(
            text=msg.get("content") or "",
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=msg,
        )

    def append_tool_exchange(
        self, history: Any, reply: ProviderReply, results: list[tuple[ToolCallReq, str]]
    ) -> None:
        history.append(reply.raw)  # 原生 assistant 消息自带 tool_calls
        for call, result in results:
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})

    async def stream(
        self, history: Any, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            # async with 保证：客户端断开导致本生成器被取消时，上游连接一并关闭
            async with self._client.stream(
                "POST",
                f"{self._settings.deepseek_base_url}/chat/completions",
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
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta") or {}
                        if delta.get("content"):
                            yield {"type": "delta", "content": delta["content"]}
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        yield {
                            "type": "usage",
                            "input": u.get("prompt_tokens", 0),
                            "output": u.get("completion_tokens", 0),
                        }
        except httpx.TimeoutException as e:
            raise UpstreamError("DeepSeek 流式超时", retryable=True) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"DeepSeek 流式网络错误: {e}", retryable=True) from e
