"""
LangGraph 状态定义 — 所有 Agent 共享的数据结构

在 LangGraph 中，State 就像流水线上的托盘：
- 每个 Agent 从 State 里读取自己需要的数据
- 处理完后把结果写回 State
- Supervisor 根据 State 决定下一个 Agent

扩展说明（Multi-Agent 重构）：
- 新增 messages 字段用于 Agent 间共享对话历史（operator.add 自动追加）
- 新增 next_agent / last_agent 用于 Supervisor 路由决策
- 新增 retry_count / max_iterations / iteration_count 用于循环控制
- 新增 phase 用于标记当前工作流阶段
- 新增 agent_outputs 用于 Supervisor 快速查阅各 Agent 输出摘要
"""

from typing import Annotated, TypedDict, Any
from operator import add
from langchain_core.messages import BaseMessage
from app.schemas.models import TripRequest, TripPlan


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """合并两个字典，right 的键值覆盖 left 的同名键"""
    merged = dict(left)
    merged.update(right)
    return merged


class TripState(TypedDict):
    """
    旅行规划工作流的共享状态

    数据流向：
    ┌─────────────────┐
    │  request         │ ← API 路由写入（用户的原始请求）
    │  messages        │ ← Agent 间共享对话历史（Annotated[str, add] 自动追加）
    │  next_agent      │ ← Supervisor 输出的路由信号
    │  last_agent      │ ← 上一个执行的 Agent 名称（用于重试）
    │  retry_count     │ ← 当前步骤的重试计数
    │  phase           │ ← 当前阶段 (collect / plan / review / done)
    │  raw_attractions │ ← POI Agent 写入
    │  raw_weather     │ ← Weather Agent 写入
    │  raw_hotels      │ ← Hotel Agent 写入
    │  raw_plan_text   │ ← Planner Agent 写入（LLM 生成的原始文本）
    │  trip_plan       │ ← Finalize 节点写入（最终结构化结果）
    │  agent_outputs   │ ← 各 Agent 输出摘要（Supervisor 快速查阅）
    │  error           │ ← 任何 Agent 出错时写入
    │  iteration_count │ ← 全局迭代计数器（防止无限循环）
    │  max_iterations  │ ← 最大迭代次数（默认 20）
    └─────────────────┘
    """

    # ===== 输入：用户请求（API 路由在调用 graph 之前填入）=====
    request: TripRequest

    # ===== Agent 通信 =====
    messages: Annotated[list[BaseMessage], add]
    next_agent: str
    last_agent: str
    agent_outputs: Annotated[dict[str, str], _merge_dicts]

    # ===== 中间结果：各 Agent 依次填入 =====
    raw_attractions: list[dict]
    raw_weather: list[dict]
    raw_hotels: list[dict]
    raw_plan_text: str
    attraction_photos: Annotated[dict[str, str], _merge_dicts]

    # ===== 最终输出 =====
    trip_plan: TripPlan | None

    # ===== 流程控制 =====
    phase: str
    retry_count: int
    iteration_count: int
    max_iterations: int

    # ===== 错误信息 =====
    error: str
