"""
LangGraph 工作流组装 — Planner-Centric Pipeline 架构

确定性管道 + 唯一条件回路（预算超限回退修正）

  编排图（StateGraph）

  START
    │
    ▼
  initialize_node          ← 初始化 phase、计数器、清空累积器
    │
    ▼
  data_collection_node     ← 并行执行 4 个 Agent（ThreadPoolExecutor, 4 线程）
    │                         POI + Weather + Hotel + Transport
    │                         失败自动重试 1 轮
    ▼
  planner_node             ← 单次 LLM：生成完整行程 JSON
    │                        （若 phase="review" 则进入修订模式）
    ▼
  budget_node              ← ReAct：计算预算 & 提出省钱建议
    │
    ├── 超预算且 revision_round < 3 → 回到 planner_node（修订循环）
    │
    └── 否则 → finalize_node → END

规则：
1. data_collection_node 内部并行 POI+Weather+Hotel+Transport，完成后直接进 planner
2. planner → budget 链式执行
3. budget → workflow_router 硬编码条件路由（无 LLM）
4. finalize → END
"""

from langgraph.graph import StateGraph, START, END
from app.graph.state import TripState
from app.graph.nodes import (
    initialize_node,
    data_collection_node,
    planner_node,
    budget_node,
    finalize_node,
)
from app.graph.routing import workflow_router


def build_trip_workflow() -> StateGraph:
    """构建 Planner-Centric 旅行规划工作流"""
    graph = StateGraph(TripState)

    # 注册节点
    graph.add_node("initialize_node", initialize_node)
    graph.add_node("data_collection_node", data_collection_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("budget_node", budget_node)
    graph.add_node("finalize_node", finalize_node)

    # 确定性管道
    graph.add_edge(START, "initialize_node")
    graph.add_edge("initialize_node", "data_collection_node")
    graph.add_edge("data_collection_node", "planner_node")
    graph.add_edge("planner_node", "budget_node")

    # 唯一条件边：预算超限回路
    graph.add_conditional_edges(
        "budget_node",
        workflow_router,
        {
            "planner_node": "planner_node",
            "finalize_node": "finalize_node",
        },
    )

    # 终止
    graph.add_edge("finalize_node", END)

    return graph.compile()


# 模块级单例
_workflow: StateGraph | None = None


def get_workflow() -> StateGraph:
    """获取工作流实例（单例）"""
    global _workflow
    if _workflow is None:
        _workflow = build_trip_workflow()
        print("✅ LangGraph Planner-Centric 工作流构建完成")
        print(f"   管道: init → data_collection(POI+Weather+Hotel+Transport并行) → planner → budget → finalize")
        print(f"   条件回路: budget → workflow_router → planner(修正) ─→ budget → finalize")
        print(f"   修正上限: revision_round < 3, 死循环保护: iteration >= max_iterations")
    return _workflow
