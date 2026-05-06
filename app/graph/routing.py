"""
LangGraph 条件路由 — 根据 Supervisor 的决策和错误状态决定下一个节点

P1 优化：Supervisor 减频
- Supervisor 只做阶段级路由（collect / plan / finalize），不再每个 Agent 后都调用
- "collect" → data_collection_node（内部并行运行 POI + Weather + Hotel）
- "plan"   → planner_node → budget_node（链式，中间不回 Supervisor）
- "finalize" → finalize_node → END

错误重试/降级逻辑保留在路由层。
"""

from app.graph.state import TripState

# next_agent / last_agent 值 → LangGraph 节点名称 映射表
AGENT_TO_NODE: dict[str, str] = {
    "poi": "poi_node",
    "weather": "weather_node",
    "hotel": "hotel_node",
    "planner": "planner_node",
    "budget": "budget_node",
    "collect": "data_collection_node",
    "finalize": "finalize_node",
}


def supervisor_router(state: TripState) -> str:
    """
    Supervisor 的条件路由函数

    被 LangGraph 的 add_conditional_edges 调用。

    返回必须是已注册的节点名称字符串。
    """
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)
    last_agent = state.get("last_agent", "")
    next_agent = state.get("next_agent", "")

    # ——— 分支 0：迭代上限保护（最高优先级）———
    if iteration_count >= max_iterations:
        print(f"  [Router] 迭代次数 {iteration_count} >= {max_iterations}，强制 → finalize_node")
        return "finalize_node"

    # ——— 分支 1：错误重试 ———
    if error and retry_count < 3 and last_agent:
        retry_node = AGENT_TO_NODE.get(last_agent)
        if retry_node:
            new_count = retry_count + 1
            print(f"  [Router] 错误重试: {last_agent} → {retry_node} (第 {new_count} 次)")
            return retry_node

    # ——— 分支 2：错误降级 ———
    if error and retry_count >= 3:
        print(f"  [Router] 错误降级: {last_agent} 已达最大重试次数，返回 Supervisor 重新决策")
        return "supervisor_node"

    # ——— 分支 3：正常路由 ———
    target_node = AGENT_TO_NODE.get(next_agent)

    if target_node is None:
        print(f"  [Router] next_agent='{next_agent}' 无效，返回 Supervisor 重新决策")
        return "supervisor_node"

    print(f"  [Router] next_agent='{next_agent}' → {target_node}")
    return target_node
