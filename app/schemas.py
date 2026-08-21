"""对外 API 的请求/响应模型。

Java 对照：pydantic BaseModel ≈ DTO + Bean Validation（@Valid @NotNull @Size）。
校验失败时 FastAPI 自动返回 422 + 逐字段错误详情，无需手写异常处理。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    tools: list[str] | None = None  # 本地工具名，如 ["calculator", "weather"]
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ChatResponse(BaseModel):
    provider: str
    model: str
    content: str
    tool_calls: list[ToolCallOut] = []
    usage: Usage
    rounds: int = 1


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None
