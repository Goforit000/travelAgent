"""
LangGraph 状态定义 — 所有 Agent 共享的数据结构

Planner-Centric Pipeline 架构：
- 确定性管道：initialize → data_collection → planner → budget → (条件) → finalize
- 路由由 workflow_router 执行
- Agent 之间通过 State 传递结构化数据
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
    │  request         │ ← API 路由写入
    │  messages        │ ← Agent 间共享对话历史
    │  agent_outputs   │ ← 各 Agent 输出摘要
    │  raw_attractions │ ← POI Agent 写入
    │  raw_weather     │ ← Weather Agent 写入
    │  raw_hotels      │ ← Hotel Agent 写入
    │  raw_plan_text   │ ← Planner Agent 写入
    │  trip_plan       │ ← Finalize 节点写入
    │  phase           │ ← 当前阶段
    │  revision_round  │ ← 预算超限修正轮数
    │  error           │ ← 错误信息
    └─────────────────┘
    """

    # ===== 输入 =====
    request: TripRequest

    # ===== Agent 通信 =====
    messages: Annotated[list[BaseMessage], add]
    agent_outputs: Annotated[dict[str, str], _merge_dicts]

    # ===== 中间结果 =====
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
    revision_round: int

    # ===== 错误信息 =====
    error: str
