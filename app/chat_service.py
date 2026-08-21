"""编排层：工具调用循环 + 熔断重试 + 计费。上层路由只做协议转换。"""

from . import tools
from .config import Settings
from .providers.base import Provider
from .resilience import CircuitBreaker, guarded_call
from .schemas import ChatRequest, ChatResponse, ToolCallOut, Usage
from .usage import UsageStore


async def run_chat(
    provider: Provider,
    req: ChatRequest,
    breaker: CircuitBreaker,
    usage_store: UsageStore,
    settings: Settings,
) -> ChatResponse:
    tool_specs = tools.specs_for(req.tools)  # 未知工具名 -> KeyError -> 路由层转 400
    history = provider.init_history(req.messages)

    executed: list[ToolCallOut] = []
    total_in = total_out = 0
    content = ""
    rounds = 0

    # Function Calling 回环：模型要求调工具 -> 本地执行 -> 结果回填 -> 再问模型，
    # 直到模型给出最终回答或达到轮数上限（防模型无限要工具）。
    for _ in range(settings.max_tool_rounds):
        rounds += 1
        reply = await guarded_call(
            breaker,
            lambda: provider.chat(
                history, tool_specs, max_tokens=req.max_tokens, temperature=req.temperature
            ),
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
        )
        total_in += reply.input_tokens
        total_out += reply.output_tokens
        content = reply.text
        if not reply.tool_calls:
            break
        results = [(c, tools.execute(c.name, c.arguments)) for c in reply.tool_calls]
        executed.extend(
            ToolCallOut(name=c.name, arguments=c.arguments, result=r) for c, r in results
        )
        provider.append_tool_exchange(history, reply, results)
    else:
        content = (content or "") + "\n[已达工具调用轮数上限，返回当前结果]"

    cost = await usage_store.record(provider.model, total_in, total_out)
    return ChatResponse(
        provider=provider.name,
        model=provider.model,
        content=content,
        tool_calls=executed,
        usage=Usage(input_tokens=total_in, output_tokens=total_out, cost_usd=cost),
        rounds=rounds,
    )
