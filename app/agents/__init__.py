"""
Agent 模块 — 多 Agent 协作系统

继承关系：
  BaseAgent              ← Weather、Transport、Planner（不需要 ReAct 的 Agent）
    └── ReActAgent       ← POI、Hotel、Budget（需要 ReAct 循环的 Agent）
"""