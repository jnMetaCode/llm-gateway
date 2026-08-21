"""本地工具注册表：模型通过 Function Calling 调用这里的函数。

安全注意：calculator 用 ast 白名单求值，绝不能用 eval() —— 面试常问点。
"""

import ast
import operator
from typing import Any

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculator(expression: str) -> str:
    """四则运算（+ - * / % **），基于 ast 白名单，防注入。"""
    value = _safe_eval(ast.parse(expression, mode="eval"))
    # 整数结果去掉小数点，例如 14.0 -> 14
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


_MOCK_WEATHER = {
    "北京": "晴，32°C，西南风 2 级",
    "上海": "多云转雷阵雨，29°C",
    "深圳": "大雨，27°C",
}


def weather(city: str) -> str:
    """Mock 天气查询：真实项目里替换为外部 API 调用。"""
    return _MOCK_WEATHER.get(city, f"未收录城市「{city}」，仅支持: {', '.join(_MOCK_WEATHER)}")


# 工具名 -> (函数, JSON Schema 描述)。两家模型的工具协议不同，
# 这里只存中立的 schema，由各 provider 适配层转成自家格式。
REGISTRY: dict[str, tuple[Any, dict[str, Any]]] = {
    "calculator": (
        calculator,
        {
            "name": "calculator",
            "description": "计算一个数学表达式，支持 + - * / % ** 和括号",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 2+3*4"}},
                "required": ["expression"],
            },
        },
    ),
    "weather": (
        weather,
        {
            "name": "weather",
            "description": "查询中国城市当日天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                "required": ["city"],
            },
        },
    ),
}


def specs_for(names: list[str] | None) -> list[dict[str, Any]]:
    """把请求里的工具名转成 schema 列表；未知名称直接抛错（宁可 400 不可静默忽略）。"""
    if not names:
        return []
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"未知工具: {', '.join(unknown)}，可用: {', '.join(REGISTRY)}")
    return [REGISTRY[n][1] for n in names]


def execute(name: str, arguments: dict[str, Any]) -> str:
    """执行工具。异常不上抛——把错误文本还给模型，让它自行修正参数重试。"""
    fn = REGISTRY[name][0]
    try:
        return fn(**arguments)
    except Exception as e:  # noqa: BLE001 - 工具失败属于业务内预期路径
        return f"工具执行失败: {e}"
