# llm-gateway

统一代理 DeepSeek / Claude 的 LLM 网关：**SSE 流式 · Function Calling 回环 · token 计费 · 熔断降级 · 指数退避重试**。

> Built in public：规格驱动开发的示范项目——先写验收标准，AI 实现，人工逐条验收。**10/10 测试 + Docker 验收。**

**三个值得看的实现**：
① **SSE 断连的取消传播链**——客户端断开 → 生成器被取消 → `async with` 退出 → 上游连接关闭。
没有这条链，上游会继续吐 token 而无人接收，是流式场景最隐蔽的烧钱方式。
② **三态熔断器**（CLOSED/OPEN/HALF_OPEN）**按 provider 隔离**——一家挂了不拖死另一家，有测试锁死。
③ **4xx 不重试也不计入熔断**——4xx 是请求方的问题，计入会误伤健康的上游；只有 429/5xx/超时才退避重试。

## 快速开始

```bash
uv sync                      # 安装依赖（含 dev）
cp .env.example .env         # 填入 GW_DEEPSEEK_API_KEY / GW_ANTHROPIC_API_KEY
uv run uvicorn app.main:app --reload
```

跑测试（无需任何 key，上游全部 mock）：

```bash
uv run pytest -q
uv run ruff check .
```

## API 示例

```bash
# 非流式（默认 deepseek；换模型加 -H "X-Provider: claude"）
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "算一下 2+3*4"}],
  "tools": ["calculator"]
}' | jq

# SSE 流式
curl -sN localhost:8000/v1/chat/stream -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "用一句话介绍北京"}]
}'

# 用量与费用
curl -s localhost:8000/v1/usage | jq
```

## 架构

```
Route(main.py) ── 协议/校验/错误映射
   └─ ChatService(chat_service.py) ── 工具回环 + guarded_call(熔断×重试) + 计费
        ├─ Provider 适配层(providers/) ── DeepSeek(OpenAI 协议) / Claude(Messages API)
        ├─ CircuitBreaker(resilience.py) ── CLOSED → OPEN → HALF_OPEN 状态机，按 provider 隔离
        ├─ tools.py ── 本地工具注册表（ast 白名单计算器 / mock 天气）
        └─ UsageStore(usage.py) ── 内存计数 + 价格表估费
```

## 设计决策（面试可讲）

1. **为什么计费放网关层**：业务方无感、跨模型统一口径、限额与告警有单一落点。
2. **熔断为什么按 provider 隔离**：DeepSeek 挂了不该拖死 Claude 通道（`test_breaker_isolated_per_provider` 验证）。
3. **4xx 不重试不计熔断**：4xx 是请求方问题，重试只会浪费配额；只有 429/5xx/超时值得退避重试。
4. **SSE 断连取消链**：客户端断开 → Starlette 取消响应生成器（CancelledError）→ provider 内 `async with client.stream` 退出 → 上游连接关闭。没有这条链，上游会一直吐 token 白烧钱。
5. **calculator 用 ast 白名单不用 eval**：工具参数来自模型输出，等于不可信输入。

## Java 程序员对照表

| 本项目 | Java 世界 |
|---|---|
| pydantic BaseModel | DTO + Bean Validation |
| Depends / Annotated | Spring 构造器注入 |
| lifespan | @PostConstruct / @PreDestroy |
| exception_handler | @ControllerAdvice |
| asyncio + httpx.AsyncClient | CompletableFuture / WebClient |
| CircuitBreaker (手写) | Resilience4j |
| ASGITransport 测试 | MockMvc |

## 已知边界（刻意不做）

无鉴权、无持久化（用量在内存）、流式不重试（失败即报错事件）、流式不支持工具调用。

---

### 关于这一组项目

这是三套**评估驱动**的 AI 应用系统，同期开源，可以单独用也可以对照看：

| | 做什么 | 关键实测 |
|---|---|---|
| [repo-rag](https://github.com/jnMetaCode/repo-rag) | 中文知识库 RAG：结构分块 + 两层拒答 + 引用溯源 | hit@1 95.8% · faithfulness 0.981 |
| [orchestrator-lg](https://github.com/jnMetaCode/orchestrator-lg) | 自研 DAG 引擎迁到 LangGraph：checkpoint + 可持久化审批中断 | 7/7 测试 · YAML 零改动兼容 |
| [llm-gateway](https://github.com/jnMetaCode/llm-gateway) | 多模型网关：SSE 取消链 + 三态熔断 + token 计费 | 10/10 测试 · Docker |

共同的方法论：**先建评估集，再写优化**——每个技术决策都由实测数据推导，包括那些「该做但做了反而更差」的决策。

### 关于作者

[@jnMetaCode](https://github.com/jnMetaCode) · 11 年 IT、8 年技术团队管理 · 公众号 **AI不止语**
其他开源：[agency-agents-zh](https://github.com/jnMetaCode/agency-agents-zh)（19.8k★，267 个 AI 专家角色 × 18 类工具链）·
[superpowers-zh](https://github.com/jnMetaCode/superpowers-zh)（7.8k★）· [agency-orchestrator](https://github.com/jnMetaCode/agency-orchestrator)（2.1k★，本项目的上游）

> 在看北京的 AI 技术负责人 / 交付负责人 / 技术合伙人机会 · jnMetaCode@qq.com
