"""Provider 抽象：把两家 API 的差异挡在适配层内，上层只看统一结构。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Message


class UpstreamError(Exception):
    """上游调用失败。retryable=True（429/5xx/超时）才值得重试与计入熔断。"""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class ToolCallReq:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderReply:
    text: str
    tool_calls: list[ToolCallReq] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None  # 保留原生 assistant 消息，回填历史时按各家格式使用


class Provider(ABC):
    name: str
    model: str

    @abstractmethod
    def init_history(self, messages: list[Message]) -> Any:
        """把统一消息格式转为该家的原生对话历史（后续被原地追加）。"""

    @abstractmethod
    async def chat(
        self, history: Any, tool_specs: list[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> ProviderReply: ...

    @abstractmethod
    def append_tool_exchange(
        self, history: Any, reply: ProviderReply, results: list[tuple[ToolCallReq, str]]
    ) -> None:
        """把「模型要调工具 -> 我们执行的结果」按该家格式回填进历史。"""

    @abstractmethod
    def stream(
        self, history: Any, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[dict[str, Any]]:
        """产出 {"type": "delta", "content": str} 与 {"type": "usage", ...} 事件。"""
