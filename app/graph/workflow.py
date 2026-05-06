"""
LangGraph 工作流组装 — 多 Agent 协作的 Supervisor 架构（P1 优化版）

优化变更：
1. Supervisor 减频：仅 3 个阶段切换点介入（collect / plan / finalize）
2. 数据收集并行：data_collection_node 内 ThreadPoolExecutor 并行执行 POI+Weather+Hotel
3. 链式规划：planner → budget 直接串联，中间不回 Supervisor

拓扑结构：

                    START
                      │
                      ▼
               initialize_node
                      │
                      ▼
               supervisor_node ◄──────────────────────┐
                      │                               │
                      │ (add_conditional_edges)        │
                      │ supervisor_router()            │
                      │                               │
          ┌───────────┼───────────────┐               │
          ▼           ▼               ▼               │
   data_collection  planner      finalize_node        │
   _node            _node            │                │
   (POI+Weather     │               END               │
    +Hotel 并行)    │                                 │
          │         ▼                                 │
          │    budget_node                            │
          │         │                                 │
          └─────────┴─────────────────────────────────┘
               (add_edge, 无条件返回 supervisor)

规则：
1. data_collection_node 并行执行 POI+Weather+Hotel，完成后返回 supervisor
2. planner_node → budget_node 链式执行，plan 阶段内不回 supervisor
3. budget_node 完成后返回 supervisor（此时 planner+budget 都已完成）
4. 单 Agent 节点（poi/weather/hotel）保留用于错误重试
"""

from langgraph.graph import StateGraph, START, END
from app.graph.state import TripState
from app.graph.nodes import (
    initialize_node,
    supervisor_node,
    data_collection_node,
    planner_node,
    budget_node,
    poi_node,
    weather_node,
    hotel_node,
    finalize_node,
)
from app.graph.routing import supervisor_router


def build_trip_workflow() -> StateGraph:
    """
    构建多 Agent 协作旅行规划工作流（P1 优化版）

    流程：
    START → initialize → supervisor → (条件路由)
      ├─ "collect"  → data_collection_node → supervisor
      ├─ "plan"     → planner_node → budget_node → supervisor
      ├─ "poi"      → poi_node → supervisor (错误重试)
      ├─ "weather"  → weather_node → supervisor (错误重试)
      ├─ "hotel"    → hotel_node → supervisor (错误重试)
      └─ "finalize" → finalize_node → END

    关键变更：
    - data_collection_node 内部并行执行 POI/Weather/Hotel，完成后统一返回
    - planner_node → budget_node 直接链式调用
    """
    # 1. 创建 StateGraph
    graph = StateGraph(TripState)

    # 2. 注册所有节点
    graph.add_node("initialize_node", initialize_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("data_collection_node", data_collection_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("budget_node", budget_node)
    graph.add_node("poi_node", poi_node)
    graph.add_node("weather_node", weather_node)
    graph.add_node("hotel_node", hotel_node)
    graph.add_node("finalize_node", finalize_node)

    # 3. 连接边

    # 3a. 入口
    graph.add_edge(START, "initialize_node")
    graph.add_edge("initialize_node", "supervisor_node")

    # 3b. Supervisor 条件路由
    graph.add_conditional_edges(
        "supervisor_node",
        supervisor_router,
        {
            "data_collection_node": "data_collection_node",
            "planner_node": "planner_node",
            "budget_node": "budget_node",
            "poi_node": "poi_node",
            "weather_node": "weather_node",
            "hotel_node": "hotel_node",
            "finalize_node": "finalize_node",
            "supervisor_node": "supervisor_node",
        },
    )

    # 3c. 数据收集完成后 → 返回 Supervisor
    graph.add_edge("data_collection_node", "supervisor_node")

    # 3d. 规划链：planner → budget（中间不回 Supervisor）
    graph.add_edge("planner_node", "budget_node")
    # budget 完成后 → 返回 Supervisor 做最终决策
    graph.add_edge("budget_node", "supervisor_node")

    # 3e. 单 Agent 重试节点完成后 → 返回 Supervisor
    graph.add_edge("poi_node", "supervisor_node")
    graph.add_edge("weather_node", "supervisor_node")
    graph.add_edge("hotel_node", "supervisor_node")

    # 3f. finalize → END
    graph.add_edge("finalize_node", END)

    # 4. 编译
    app = graph.compile()

    return app


# 模块级单例
_workflow: StateGraph | None = None


def get_workflow() -> StateGraph:
    """获取工作流实例（单例）"""
    global _workflow
    if _workflow is None:
        _workflow = build_trip_workflow()
        print("✅ LangGraph 多 Agent 工作流构建完成 (P1 优化版)")
        print(f"   节点: initialize → supervisor → data_collection|planner→budget|finalize")
        print(f"   路由: supervisor_router (3 阶段决策, 错误重试/降级)")
        print(f"   循环上限: max_iterations=15")
        print(f"   数据收集: POI+Weather+Hotel 并行 (ThreadPoolExecutor)")
        print(f"   规划链: planner → budget (无中间 Supervisor)")
        print(f"   Supervisor 调用次数: 3 次 (vs 原 6 次)")
    return _workflow
