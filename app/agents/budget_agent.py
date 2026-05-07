"""
Budget Agent — 负责计算旅行计划预算，超限时生成削减建议（不修改行程）

职责边界：
1. 读取 State 中的 raw_plan_text
2. 调用 calculate_budget_tool 精确计算各项费用
3. 对比 target_budget：
   - 未超限 → 输出 "OK" + 预算汇总
   - 超限 → 调用 suggest_savings_tool 生成削减建议 → 输出 "OVERSHOOT|{json}"
     （由 Supervisor 路由给 Planner Agent 执行修正，Budget 不修改 JSON）

设计理由：
- Budget Agent 的职责是"计算 + 建议"，不应越界修改行程 JSON
- 行程 JSON 修改由 Planner Agent 负责（它掌握完整 schema）
- Supervisor 作为唯一路由决策点，统一处理超限回退逻辑
"""

import json
from langchain_core.tools import BaseTool
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.budget_tools import calculate_budget_tool, suggest_savings_tool
from app.tools.llm import BUDGET_MAX_TOKENS


class BudgetAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "budget_agent"

    @property
    def description(self) -> str:
        return (
            "预算分析师。计算旅行计划费用，若超限生成削减建议（不修改行程）。"
            "负责填充 agent_outputs（含超限信号或预算汇总）。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是专业的旅行预算分析师。你的任务：

## 
1. 从上下文提取旅行计划 JSON，调用 calculate_budget_tool 计算当前预算
2. 分析预算合理性（门票异常、日均餐饮是否合理、酒店费用是否匹配类型）
3. 根据目标预算做判断：

### 情况 A：无目标预算（target_budget = 0）或未超限
输出：
{"status": "ok", "budget": {"total_attractions": N, "total_hotels": N, "total_meals": N, "total_transportation": N, "total": N}, "analysis": "预算分析总结"}

### 情况 B：超限（总费用 > target_budget）
1. 调用 suggest_savings_tool 获取削减建议
2. 输出：
{"status": "overshoot", "budget": {"total_attractions": N, ...}, "target_budget": N, "overshoot_amount": N, "savings_suggestions": [{"category": "...", "current": N, "suggested": N, "saving": N, "description": "..."}], "analysis": "超限分析总结"}

重要：
- 不修改旅行计划 JSON
- 削减建议必须具体可执行（明确哪个类别降多少、替换什么）
- 如果计算工具返回了 warnings，在 analysis 中提及"""

    @property
    def tools(self) -> list[BaseTool]:
        return [calculate_budget_tool, suggest_savings_tool]

    @property
    def max_steps(self) -> int:
        return 5

    @property
    def max_tokens(self) -> int:
        return BUDGET_MAX_TOKENS

    def _build_context_message(self, state: TripState) -> str:
        """构造预算计算的上下文信息"""
        request = state.get("request")
        raw_plan_text = state.get("raw_plan_text", "")

        if request is None:
            return "错误：State 中缺少 request 字段"

        city = getattr(request, "city", "未知")
        accommodation = getattr(request, "accommodation", "经济型酒店")
        travel_days = getattr(request, "travel_days", 1)
        target_budget = getattr(request, "target_budget", 0) or 0

        hotel_cost_map = {
            "经济型酒店": 150, "舒适型酒店": 350, "豪华酒店": 800, "民宿": 200,
        }
        hotel_cost_per_night = hotel_cost_map.get(accommodation, 300)

        transport_cost_map = {
            "自驾": 100, "公共交通": 30, "步行": 5, "混合": 50,
        }
        transportation = getattr(request, "transportation", "公共交通")
        transport_cost_per_day = transport_cost_map.get(transportation, 50)

        budget_line = f"目标预算上限：{target_budget} 元" if target_budget > 0 else "目标预算上限：无限制"

        return f"""请计算以下旅行计划的预算：

目的地城市：{city}
旅行天数：{travel_days} 天
住宿偏好：{accommodation}（估算每晚 {hotel_cost_per_night} 元）
交通方式：{transportation}（估算每天 {transport_cost_per_day} 元）
{budget_line}

=== 旅行计划 JSON ===
{raw_plan_text}

请严格按照 system prompt 执行：
1. 调用 calculate_budget_tool(trip_plan_json=上面的JSON, hotel_cost_per_night={hotel_cost_per_night}, transport_cost_per_day={transport_cost_per_day})
2. 如果目标预算 > 0 且总费用 > 目标预算：
   → 调用 suggest_savings_tool(current_budget_json=上一步结果, target_budget={target_budget}, travel_days={travel_days})
   → 输出 overshoot 格式
3. 否则输出 ok 格式"""

    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list,
    ) -> dict:
        """
        从 LLM 最终输出中提取预算状态和削减建议

        - ok 状态 → agent_outputs["budget_agent"] = 预算分析文本
        - overshoot 状态 → agent_outputs["budget_agent"] = "OVERSHOOT|{json}"
        """
        data: dict = {}
        try:
            data = json.loads(llm_content)
        except (json.JSONDecodeError, Exception):
            extracted = self._extract_json(llm_content)
            if extracted:
                try:
                    data = json.loads(extracted)
                except Exception:
                    data = {}

        if not isinstance(data, dict):
            data = {}

        status = data.get("status", "ok")
        budget = data.get("budget", {})
        total = budget.get("total", 0)
        analysis = data.get("analysis", "")

        output: dict = {
            "agent_outputs": {},
        }

        if status == "overshoot":
            # 构造超限信号：Supervisor 通过 "OVERSHOOT|" 前缀检测
            target_budget = data.get("target_budget", 0)
            overshoot_amount = data.get("overshoot_amount", 0)
            savings_suggestions = data.get("savings_suggestions", [])

            overshoot_payload = {
                "target_budget": target_budget,
                "overshoot_amount": overshoot_amount,
                "budget": budget,
                "savings_suggestions": savings_suggestions,
                "analysis": analysis,
            }
            payload_json = json.dumps(overshoot_payload, ensure_ascii=False)

            print(f"  [budget_agent] ⚠️ 预算超限! total={total} > target={target_budget}, overshoot={overshoot_amount}")
            print(f"  [budget_agent] 生成 {len(savings_suggestions)} 条削减建议")

            output["agent_outputs"][self.name] = f"OVERSHOOT|{payload_json}"
            output["phase"] = "review"
        else:
            print(f"  [budget_agent] ✅ 预算正常: total={total}")
            output["agent_outputs"][self.name] = (
                analysis or f"预算计算完成，总计 {total} 元（未超限）"
            )

        # 如果 trip_plan 已存在且有预算数据，合并进去
        trip_plan = state.get("trip_plan")
        if trip_plan is not None and budget:
            from app.schemas.models import Budget
            try:
                trip_plan.budget = Budget(**budget)
                output["trip_plan"] = trip_plan
            except Exception:
                pass

        return output
