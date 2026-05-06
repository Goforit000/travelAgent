"""
Budget Agent — 负责计算旅行计划预算并给出优化建议

职责：
1. 读取 State 中的 raw_plan_text 或 trip_plan
2. 调用 calculate_budget_tool 精确计算各项费用
3. 如果预算超限（对比用户的预算期望），调用 suggest_savings_tool 生成削减方案
4. 将预算信息写入 trip_plan.budget 或 agent_outputs
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
            "预算分析师。精确计算旅行计划的各项费用（门票、酒店、餐饮、交通），"
            "如超限则提供可行的削减方案。负责填充 budget 相关字段。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是专业的旅行预算分析师。你的任务是：
1. 接收旅行计划，调用 calculate_budget_tool 逐项计算费用
2. 分析预算的合理性：
   - 门票价格是否异常（如单人门票超过 200 元需要标记）
   - 餐饮日均费用是否合理（50-150元/天正常）
   - 酒店费用是否与住宿类型匹配
3. 如果用户有预算上限且总费用超限，调用 suggest_savings_tool 生成削减方案
4. 如果没有明确的预算上限，仅计算并给出优化建议
工作流程：
1. 先从上下文中提取 trip_plan_json（完整的旅行计划 JSON）
2. 调用 calculate_budget_tool(trip_plan_json=trip_plan_json, ...) 计算
3. 如果结果显示存在 overshoot，调用 suggest_savings_tool 获取削减建议
4. 如果没有超限，直接输出预算汇总

输出格式（必须是有效 JSON）：
{
  "status": "done",
  "budget": {
    "total_attractions": N,
    "total_hotels": N,
    "total_meals": N,
    "total_transportation": N,
    "total": N
  },
  "analysis": "预算分析总结",
  "suggestions": [...]
}

如果计算工具返回了 warnings 数组，必须在 analysis 中提及这些问题。"""

    @property
    def tools(self) -> list[BaseTool]:
        return [calculate_budget_tool, suggest_savings_tool]

    @property
    def max_steps(self) -> int:
        return 5  # 计算 + 分析 + 可能的削减方案

    @property
    def max_tokens(self) -> int:
        return BUDGET_MAX_TOKENS  # 3072，预算分析中等复杂度

    def _build_context_message(self, state: TripState) -> str:
        """构造预算计算的上下文信息"""
        request = state.get("request")
        raw_plan_text = state.get("raw_plan_text", "")

        if request is None:
            return "错误：State 中缺少 request 字段"

        city = getattr(request, "city", "未知")
        accommodation = getattr(request, "accommodation", "经济型酒店")
        travel_days = getattr(request, "travel_days", 1)

        # 根据住宿类型估算每晚酒店费用
        hotel_cost_map = {
            "经济型酒店": 150,
            "舒适型酒店": 350,
            "豪华酒店": 800,
            "民宿": 200,
        }
        hotel_cost_per_night = hotel_cost_map.get(accommodation, 300)

        # 根据住宿类型推断交通方式
        transport_cost_map = {
            "自驾": 100,
            "公共交通": 30,
            "步行": 5,
            "混合": 50,
        }
        transportation = getattr(request, "transportation", "公共交通")
        transport_cost_per_day = transport_cost_map.get(transportation, 50)

        context = f"""请计算以下旅行计划的预算：

目的地城市：{city}
旅行天数：{travel_days} 天
住宿偏好：{accommodation}（估算每晚 {hotel_cost_per_night} 元）
交通方式：{transportation}（估算每天 {transport_cost_per_day} 元）

=== 旅行计划 JSON ===
{raw_plan_text}

请严格按照 system prompt 中的工作流程执行：
1. 首先调用 calculate_budget_tool，传入参数：
   - trip_plan_json: 上面的旅行计划 JSON
   - hotel_cost_per_night: {hotel_cost_per_night}
   - transport_cost_per_day: {transport_cost_per_day}
2. 分析计算结果，如果存在 overshoot 且用户有预算限制，调用 suggest_savings_tool
3. 最终输出完整的预算分析 JSON"""

        return context

    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list,
    ) -> dict:
        """
        从 LLM 最终输出中提取预算信息

        预算数据将写入 agent_outputs，由 finalize 节点合并到 trip_plan.budget
        """
        budget_data: dict = {}
        analysis = ""
        suggestions: list = []

        try:
            data = json.loads(llm_content)
            budget_data = data.get("budget", {})
            analysis = data.get("analysis", "")
            suggestions = data.get("suggestions", [])
        except (json.JSONDecodeError, Exception):
            extracted = self._extract_json(llm_content)
            if extracted:
                try:
                    data = json.loads(extracted)
                    budget_data = data.get("budget", {})
                    analysis = data.get("analysis", "")
                    suggestions = data.get("suggestions", [])
                except Exception:
                    pass

        output: dict = {
            "agent_outputs": {
                self.name: analysis or f"预算计算完成，总计 {budget_data.get('total', 'N/A')} 元",
            },
        }

        # 如果 trip_plan 已存在，将 budget 合并进去
        trip_plan = state.get("trip_plan")
        if trip_plan is not None and budget_data:
            from app.schemas.models import Budget
            try:
                trip_plan.budget = Budget(**budget_data)
                output["trip_plan"] = trip_plan
            except Exception:
                pass

        return output
