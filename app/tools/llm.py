"""
LLM 工具 — 提供 ChatOpenAI 实例

为什么单独一个文件？
- 所有 Agent 都需要用 LLM，但创建方式只需要定义一次
- 如果以后要换模型（比如从 Qwen 换成 DeepSeek），只改这一个文件
- 相当于 Java 里的 @Bean 工厂方法

为什么用 langchain-openai 的 ChatOpenAI？
- 它兼容所有 OpenAI 格式的 API（OpenAI / DeepSeek / 通义千问等）
- 只需要改 base_url 就能切换服务商

Multi-Agent 重构变更：
- 新增 get_model_with_tools() 方法，支持 bind_tools
- 新增 max_tokens / request_timeout 参数，防止 API 超时和无限等待
- 保留原 get_chat_model() 用于简单调用
"""

from typing import Optional, Sequence
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from app.config import settings

# 模块级缓存，避免重复创建
_chat_model: ChatOpenAI | None = None

# 默认的 max_tokens：按 Agent 复杂度分档
DEFAULT_MAX_TOKENS = 2048        # Supervisor / Weather / Hotel 默认
PLANNER_MAX_TOKENS = 6144        # Planner 需要生成大段 JSON，给更多输出空间
BUDGET_MAX_TOKENS = 3072         # Budget 中等复杂度

# HTTP 请求超时（秒）：给 qwen 模型足够的推理时间
DEFAULT_REQUEST_TIMEOUT = 120    # 默认 120 秒
PLANNER_REQUEST_TIMEOUT = 180    # Planner 需要更长时间


def get_chat_model(
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> ChatOpenAI:
    """
    获取 ChatOpenAI 实例（单例，不带工具绑定）

    注意：单例模式下 max_tokens / request_timeout 仅在首次创建时生效。
    如果需要不同配置，请使用 get_model_with_tools()。

    Args:
        max_tokens: 模型最大输出 token 数
        request_timeout: HTTP 请求超时秒数

    Returns:
        配置好的 ChatOpenAI
    """
    global _chat_model

    if _chat_model is None:
        _chat_model = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=0.7,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )

    return _chat_model


def get_model_with_tools(
    tools: Sequence[BaseTool] | None = None,
    temperature: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> ChatOpenAI:
    """
    获取绑定了工具的 ChatOpenAI 实例

    每次调用返回一个新的 model 副本（因为 bind_tools 会修改实例状态，
    不同 Agent 需要绑定不同的工具集，所以不能缓存同一个实例）。

    Args:
        tools: 要绑定的工具列表，为空则获取不绑定工具的 model
        temperature: 温度参数，为 None 则使用默认值 0.7
        max_tokens: 模型最大输出 token 数
        request_timeout: HTTP 请求超时秒数

    Returns:
        绑定了工具的 ChatOpenAI 实例
    """
    temp = temperature if temperature is not None else 0.7

    model_kwargs: dict = {
        "api_key": settings.openai_api_key,
        "base_url": settings.openai_base_url,
        "model": settings.openai_model,
        "temperature": temp,
        "max_tokens": max_tokens,
        "request_timeout": request_timeout,
    }

    model = ChatOpenAI(**model_kwargs)

    if tools:
        return model.bind_tools(tools)

    return model


def reset_chat_model():
    """重置模型缓存（测试或配置变更时使用）"""
    global _chat_model
    _chat_model = None
