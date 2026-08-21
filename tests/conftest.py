import httpx
import pytest

from app.config import settings
from app.main import create_app


@pytest.fixture(autouse=True)
def fast_settings(monkeypatch):
    """测试不等真实退避延迟，也不需要真实 key。"""
    monkeypatch.setattr(settings, "retry_base_delay", 0.01)
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")


@pytest.fixture
async def client():
    app = create_app()
    # ASGITransport: 不起真实端口，直接把请求打进 ASGI 应用（≈ Spring 的 MockMvc）
    # ASGITransport 不跑 lifespan，手动进入 lifespan 上下文完成装配
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
