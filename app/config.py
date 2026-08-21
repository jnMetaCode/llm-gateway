"""集中配置：环境变量以 GW_ 前缀注入（.env 亦可）。

Java 对照：约等于 @ConfigurationProperties(prefix = "gw") + application.yml。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GW_", extra="ignore")

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-5"

    upstream_timeout: float = 30.0
    retry_max_attempts: int = 3  # 1 次原始请求 + 最多 2 次重试
    retry_base_delay: float = 0.5  # 指数退避基数：0.5s, 1.0s
    breaker_fail_threshold: int = 3
    breaker_open_seconds: float = 30.0

    max_tool_rounds: int = 5

    # 价格表：美元 / 百万 token（input, output）。示例值，按官网核对后更新。
    prices: dict[str, tuple[float, float]] = {
        "deepseek-chat": (0.27, 1.10),
        "claude-sonnet-5": (3.00, 15.00),
    }


settings = Settings()
