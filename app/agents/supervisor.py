"""
Supervisor Agent — 工作流的总监督者，负责阶段级路由决策

P1 优化变更：
- Supervisor 只做阶段级决策，不再每个 Agent 后都调用
- 3 个路由值：collect / planner / finalize
- "collect" → data_collection_node（内部并行 POI+Weather+Hotel）
- "planner"    → planner_node → budget_node（链式，中间不回 Supervisor）
- "finalize" → finalize_node → END

决策频率：原 ~6 次/请求 → 现 ~3 次/请求
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.llm import get_chat_model


class SupervisorAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "supervisor"

    @property
    def description(self) -> str:
        return (
            "总监督者。阶段级路由决策：决定工作流进入收集、规划还是完成阶段。"
            "不调用任何工具，仅做路由决策。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是旅行规划工作流的总监督者（Supervisor）。你只做阶段级路由决策。

## 可路由的阶段
| 阶段 | 路由值 | 触发的节点 | 说明 |
|------|--------|-----------|------|
| 数据收集 | "collect" | data_collection_node | 并行执行 POI+Weather+Hotel |
| 行程规划 | "planner" | planner_node → budget_node | 链式执行，中间不回 Supervisor |
| 最终处理 | "finalize" | finalize_node | 解析 JSON → 配图 → 兜底 → 结束 |
| 重试 POI | "poi" | poi_node | 仅重试景点搜索 |
| 重试 Weather | "weather" | weather_node | 仅重试天气查询 |
| 重试 Hotel | "hotel" | hotel_node | 仅重试酒店搜索 |
| 重试 Planner | "planner" | planner_node | 仅重试行程规划 |
| 重试 Budget | "budget" | budget_node | 仅重试预算计算 |

## 决策规则（严格按优先级）

### 规则 0：防死循环（最高优先级）
如果 iteration_count >= max_iterations：
→ next_agent = "finalize"

### 规则 1：错误重试
如果 error 非空 且 retry_count < 3：
→ next_agent = last_agent（重试失败的 Agent）

### 规则 2：错误降级
如果 error 非空 且 retry_count >= 3：
→ 清空 error，按顺序进入下一阶段：
  - 如果 last_agent 是数据收集阶段的 Agent (poi/weather/hotel) → next_agent = "planner"
  - 如果 last_agent 是规划阶段的 Agent (planner/budget) → next_agent = "finalize"

### 规则 3：数据收集阶段
如果 raw_attractions / raw_weather / raw_hotels 中任意一个为空，
且 agent_outputs 中不存在对应的 Agent 完成标记：
→ next_agent = "collect"

### 规则 4：规划阶段
如果数据收集完成（三个字段都有数据或对应 agent_outputs 已标记），
且 raw_plan_text 为空或 agent_outputs 中不存在 "budget_agent"：
→ next_agent = "planner"

### 规则 5：预算超限回退
如果 agent_outputs 中 budget_agent 的值以 "OVERSHOOT|" 开头：
→ next_agent = "planner"，phase = "review"
这是最高优先级的规划阶段规则。收到此信号立即路由到 planner，不要犹豫。

### 规则 6：完成
如果 agent_outputs 中同时存在 "planner_agent" 和 "budget_agent"，
且 budget_agent 不是 "OVERSHOOT|" 开头：
→ next_agent = "finalize"

### 规则 7：兜底
以上规则都不匹配：
→ next_agent = "finalize"

## 输出格式

严格输出纯 JSON（不要 markdown 代码块）：
{"next_agent": "collect|finalize|poi|weather|hotel|planner|budget", "reason": "简短理由", "phase": "collect|planner|done"}
"""

    @property
    def tools(self) -> list:
        return []

    @property
    def max_steps(self) -> int:
        return 5

    def run(self, state: TripState) -> dict:
        """Supervisor 主入口 — 单次 LLM 调用进行阶段路由决策"""
        model = get_chat_model()
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_context_message(state)),
        ]

        print(f"\n{'=' * 50}")
        print(f"👁️  {self.name} 正在决策...")
        print(f"{'=' * 50}")

        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        print(f"  [{self.name}] 决策结果: {content[:300]}")

        return self._parse_decision(state, content)

    def _build_context_message(self, state: TripState) -> str:
        """构造当前 State 的摘要"""
        request = state.get("request")
        city = getattr(request, "city", "") if request else ""
        travel_days = getattr(request, "travel_days", 0) if request else 0

        raw_attractions = state.get("raw_attractions", [])
        raw_weather = state.get("raw_weather", [])
        raw_hotels = state.get("raw_hotels", [])
        raw_plan_text = state.get("raw_plan_text", "")
        error = state.get("error", "")
        phase = state.get("phase", "collect")
        last_agent = state.get("last_agent", "")
        retry_count = state.get("retry_count", 0)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)
        agent_outputs = state.get("agent_outputs", {})

        data_ready = bool(raw_attractions and raw_weather and raw_hotels)
        budget_value = agent_outputs.get("budget_agent", "")
        budget_overshoot = budget_value.startswith("OVERSHOOT|") if isinstance(budget_value, str) else False
        plan_ready = bool(raw_plan_text and "budget_agent" in agent_outputs and not budget_overshoot)

        # 提取预算超限详情用于醒目提示
        overshoot_warning = ""
        if budget_overshoot:
            try:
                payload_json = budget_value.split("|", 1)[1]
                payload = json.loads(payload_json)
                target = payload.get("target_budget", 0)
                overshoot = payload.get("overshoot_amount", 0)
                total = payload.get("budget", {}).get("total", 0)
                revision_round = state.get("revision_round", 0)
                overshoot_warning = (
                    f"\n\n{'⚠️' * 3} 预算超限警报 {'⚠️' * 3}\n"
                    f"总费用 {total} 元 > 目标预算 {target} 元（超支 {overshoot} 元）\n"
                    f"当前修正轮数: {revision_round}/2\n"
                    f"请立即路由到 planner 进行第 {revision_round + 1} 次预算修正！\n"
                    f"{'⚠️' * 15}\n"
                )
            except Exception:
                overshoot_warning = "\n\n⚠️ 预算超限! Budget Agent 检测到费用超出目标预算。请立即路由到 planner。\n"

        return f"""请根据状态做出阶段路由决策：{overshoot_warning}

=== 数据状态 ===
景点: {'✅ ' + str(len(raw_attractions)) + ' 条' if raw_attractions else '❌ 未收集'}
天气: {'✅ ' + str(len(raw_weather)) + ' 天' if raw_weather else '❌ 未收集'}
酒店: {'✅ ' + str(len(raw_hotels)) + ' 条' if raw_hotels else '❌ 未收集'}
行程文本: {'✅ ' + str(len(raw_plan_text)) + ' 字符' if raw_plan_text else '❌ 未生成'}
数据收集完成: {'是' if data_ready else '否'}
规划+预算完成: {'是' if plan_ready else '否'}

=== Agent 执行状态 ===
{json.dumps({k: v[:80] for k, v in agent_outputs.items()}, ensure_ascii=False, indent=2) if agent_outputs else '(空)'}

=== 控制 ===
phase: {phase}, last_agent: {last_agent or '无'}, retry: {retry_count}, iter: {iteration_count}/{max_iterations}
error: {'"' + error[:100] + '"' if error else '无'}

请按决策规则输出路由 JSON。"""

    def _parse_decision(self, state: TripState, llm_content: str) -> dict:
        """解析 LLM 输出，提取 next_agent"""
        next_agent = "finalize"
        reason = ""
        new_phase = state.get("phase", "collect")
        error = state.get("error", "")

        parsed = False
        for attempt in [llm_content, self._extract_json(llm_content) or ""]:
            if not attempt:
                continue
            try:
                data = json.loads(attempt)
                next_agent = data.get("next_agent", "finalize")
                reason = data.get("reason", "")
                new_phase = data.get("phase", new_phase)
                parsed = True
                break
            except (json.JSONDecodeError, Exception):
                continue

        if not parsed:
            next_agent = self._hardcoded_fallback(state)
            reason = "LLM 解析失败，使用硬编码规则"

        valid_agents = {"collect", "finalize", "poi", "weather", "hotel", "planner", "budget"}
        if next_agent not in valid_agents:
            print(f"  ⚠️ 非法路由值 '{next_agent}'，使用 fallback")
            next_agent = self._hardcoded_fallback(state)

        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 15)
        if iteration_count >= max_iterations:
            next_agent = "finalize"
            reason = f"达到最大迭代次数 {max_iterations}，强制结束"
            new_phase = "done"

        print(f"  [{self.name}] → 路由到: {next_agent} ({reason})")
        print(f"{'=' * 50}\n")

        last_agent = state.get("last_agent", "")
        new_retry_count = state.get("retry_count", 0)
        if next_agent != last_agent:
            new_retry_count = 0
            error = ""

        return {
            "next_agent": next_agent,
            "last_agent": next_agent,
            "phase": new_phase,
            "retry_count": new_retry_count,
            "error": error,
            "iteration_count": iteration_count + 1,
            "agent_outputs": {"supervisor": f"→ {next_agent}: {reason}"},
        }

    def _hardcoded_fallback(self, state: TripState) -> str:
        """硬编码兜底阶段路由"""
        error = state.get("error", "")
        retry_count = state.get("retry_count", 0)
        last_agent = state.get("last_agent", "")
        agent_outputs = state.get("agent_outputs", {})

        # 错误重试
        if error and retry_count < 3 and last_agent:
            return last_agent

        # 错误降级
        if error and retry_count >= 3:
            if last_agent in ("poi", "weather", "hotel", "collect"):
                return "plan"
            if last_agent in ("planner", "budget"):
                return "finalize"
            return "finalize"

        # 正常阶段判断
        raw_attractions = state.get("raw_attractions", [])
        raw_weather = state.get("raw_weather", [])
        raw_hotels = state.get("raw_hotels", [])
        raw_plan_text = state.get("raw_plan_text", "")

        data_ready = bool(raw_attractions and raw_weather and raw_hotels)
        poi_done = "poi_agent" in agent_outputs
        weather_done = "weather_agent" in agent_outputs
        hotel_done = "hotel_agent" in agent_outputs
        planner_done = "planner_agent" in agent_outputs
        budget_done = "budget_agent" in agent_outputs

        # 检测预算超限信号（最高优先级的规划阶段规则）
        budget_value = agent_outputs.get("budget_agent", "")
        budget_overshoot = isinstance(budget_value, str) and budget_value.startswith("OVERSHOOT|")
        revision_round = state.get("revision_round", 0)
        if budget_overshoot and revision_round < 2:
            return "planner"

        # 数据收集
        if not (data_ready or (poi_done and weather_done and hotel_done)):
            return "collect"

        # 行程规划
        if not (raw_plan_text and planner_done and budget_done):
            return "planner"

        # 完成
        return "finalize"
