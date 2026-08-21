"""FastAPI 入口。

Java 对照：create_app() ≈ Spring Boot 的 Application + 手动装配 Bean；
lifespan ≈ @PostConstruct/@PreDestroy；Depends ≈ 构造器注入；
exception_handler ≈ @ControllerAdvice + @ExceptionHandler。
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .chat_service import run_chat
from .config import settings
from .providers.base import Provider, UpstreamError
from .providers.claude import ClaudeProvider
from .providers.deepseek import DeepSeekProvider
from .resilience import CircuitBreaker, CircuitOpenError
from .schemas import ChatRequest, ChatResponse, ErrorResponse
from .usage import UsageStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    client = httpx.AsyncClient()  # 复用连接池：每请求新建 client 是常见性能错误
    app.state.providers = {
        "deepseek": DeepSeekProvider(client, settings),
        "claude": ClaudeProvider(client, settings),
    }
    app.state.breakers = {
        name: CircuitBreaker(settings.breaker_fail_threshold, settings.breaker_open_seconds)
        for name in app.state.providers
    }
    app.state.usage = UsageStore(settings.prices)
    try:
        yield
    finally:
        await client.aclose()


def get_provider(
    request: Request,
    x_provider: Annotated[str, Header(description="deepseek 或 claude")] = "deepseek",
) -> tuple[Provider, CircuitBreaker]:
    providers = request.app.state.providers
    if x_provider not in providers:
        raise HTTPException(
            status_code=400,
            detail=f"未知 provider「{x_provider}」，可用: {', '.join(providers)}",
        )
    return providers[x_provider], request.app.state.breakers[x_provider]


ProviderDep = Annotated[tuple[Provider, CircuitBreaker], Depends(get_provider)]


def create_app() -> FastAPI:
    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(CircuitOpenError)
    async def _circuit_open(_: Request, exc: CircuitOpenError) -> JSONResponse:
        body = ErrorResponse(error=str(exc), code="circuit_open", detail="上游降级中，30s 后重试")
        return JSONResponse(status_code=503, content=body.model_dump())

    @app.exception_handler(UpstreamError)
    async def _upstream(_: Request, exc: UpstreamError) -> JSONResponse:
        body = ErrorResponse(
            error=str(exc),
            code="upstream_error",
            detail=f"retryable={exc.retryable}, status={exc.status}",
        )
        return JSONResponse(status_code=502, content=body.model_dump())

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, dep: ProviderDep, request: Request) -> ChatResponse:
        provider, breaker = dep
        try:
            return await run_chat(provider, req, breaker, request.app.state.usage, settings)
        except KeyError as e:  # 未知工具名
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/chat/stream")
    async def chat_stream(
        req: ChatRequest, dep: ProviderDep, request: Request
    ) -> StreamingResponse:
        provider, breaker = dep
        await breaker.before_call()  # 熔断中：直接 503，不建流
        history = provider.init_history(req.messages)

        async def gen() -> AsyncIterator[str]:
            # 客户端断开时 Starlette 会取消本生成器（CancelledError），
            # provider.stream 内的 async with 随之关闭上游连接 —— 取消的传播链。
            usage_ev = None
            try:
                async for ev in provider.stream(
                    history, max_tokens=req.max_tokens, temperature=req.temperature
                ):
                    if ev["type"] == "usage":
                        usage_ev = ev
                        continue
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except UpstreamError as e:
                await breaker.record_failure()
                err = {"type": "error", "code": "upstream_error", "error": str(e)}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                return
            await breaker.record_success()
            usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            if usage_ev:
                store: UsageStore = request.app.state.usage
                cost = await store.record(provider.model, usage_ev["input"], usage_ev["output"])
                usage = {
                    "input_tokens": usage_ev["input"],
                    "output_tokens": usage_ev["output"],
                    "cost_usd": cost,
                }
            yield f"data: {json.dumps({'type': 'done', 'usage': usage}, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/usage")
    async def usage(request: Request) -> dict:
        return await request.app.state.usage.summary()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
