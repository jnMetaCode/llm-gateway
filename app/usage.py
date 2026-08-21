"""调用量与费用统计（内存版，进程重启清零 —— 规格明确不做持久化）。"""

import asyncio
from collections import defaultdict
from typing import Any


class UsageStore:
    def __init__(self, prices: dict[str, tuple[float, float]]) -> None:
        self._prices = prices
        self._by_model: dict[str, dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )
        self._lock = asyncio.Lock()

    def cost_of(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p_in, p_out = self._prices.get(model, (0.0, 0.0))
        return round(input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out, 6)

    async def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = self.cost_of(model, input_tokens, output_tokens)
        async with self._lock:
            row = self._by_model[model]
            row["requests"] += 1
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["cost_usd"] = round(row["cost_usd"] + cost, 6)
        return cost

    async def summary(self) -> dict[str, Any]:
        async with self._lock:
            by_model = {m: dict(v) for m, v in self._by_model.items()}
        return {
            "by_model": by_model,
            "total_requests": sum(v["requests"] for v in by_model.values()),
            "total_cost_usd": round(sum(v["cost_usd"] for v in by_model.values()), 6),
        }
