"""
工作流条件路由 — 硬编码确定性规则（无 LLM）

Planner-Centric Pipeline 的唯一条件分支：
  budget_node 完成后 → workflow_router() → planner_node | finalize_node

规则：
1. 预算超限 + revision_round < 3 → 回退到 planner_node 修正
2. iteration_count >= max_iterations → 强制 finalize_node（防死循环）
3. 其他情况 → finalize_node
"""

from app.graph.state import TripState

MAX_REVISION_ROUNDS = 3

def workflow_router(state: TripState) -> str:
    """
    预算节点后的硬编码条件路由

    判断逻辑（按优先级）：
    1. 全局死循环保护
    2. 预算超限 + 有修正配额 → 回退 planner
    3. 否则 → 结束到 finalize
    """
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    revision_round = state.get("revision_round", 0)

    # Guard: 死循环保护
    if iteration_count >= max_iterations:
        print(f"  [Router] 迭代 {iteration_count} >= {max_iterations}，强制 → finalize_node")
        return "finalize_node"

    # Guard: 预算超限 + 未达修正上限
    agent_outputs = state.get("agent_outputs", {})
    budget_value = agent_outputs.get("budget_agent", "")
    overshoot = isinstance(budget_value, str) and budget_value.startswith("OVERSHOOT|")

    if overshoot and revision_round < MAX_REVISION_ROUNDS:
        print(f"  [Router] 预算超限! revision_round={revision_round} < {MAX_REVISION_ROUNDS} → planner_node")
        return "planner_node"

    print(f"  [Router] → finalize_node")
    return "finalize_node"
