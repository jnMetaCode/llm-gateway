"""熔断 + 重试。

Java 对照：CircuitBreaker ≈ Resilience4j 的三态熔断器（CLOSED/OPEN/HALF_OPEN），
guarded_call ≈ Retry + CircuitBreaker 装饰器组合。这里手写是为了面试能讲清状态机。
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .providers.base import UpstreamError

T = TypeVar("T")


class CircuitOpenError(Exception):
    """熔断开启中：不打上游，直接降级。"""


class CircuitBreaker:
    """三态熔断器。

    CLOSED --连续失败达阈值--> OPEN --冷却期满--> HALF_OPEN(放 1 个试探)
    试探成功 -> CLOSED；试探失败 -> 立刻回 OPEN。
    """

    def __init__(self, fail_threshold: int, open_seconds: float) -> None:
        self._fail_threshold = fail_threshold
        self._open_seconds = open_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()  # 并发请求下保护计数器（asyncio 单线程但有交错点）

    async def before_call(self) -> None:
        async with self._lock:
            if self._opened_at is None:
                return
            if time.monotonic() - self._opened_at < self._open_seconds:
                raise CircuitOpenError("上游熔断中，请稍后重试")
            # 冷却期满 -> 半开：只留 1 次失败额度，失败立刻重开
            self._opened_at = None
            self._failures = self._fail_threshold - 1

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self._fail_threshold:
                self._opened_at = time.monotonic()


async def guarded_call(
    breaker: CircuitBreaker,
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
) -> T:
    """熔断 + 指数退避重试的组合调用。

    只有「可重试」的上游错误（429/5xx/超时）计入熔断失败并触发重试；
    4xx 说明是我们的请求有问题，重试也没用，直接上抛。
    """
    last: UpstreamError | None = None
    for i in range(attempts):
        await breaker.before_call()
        try:
            result = await fn()
        except UpstreamError as e:
            if not e.retryable:
                raise
            await breaker.record_failure()
            last = e
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2**i))
            continue
        await breaker.record_success()
        return result
    assert last is not None
    raise last
