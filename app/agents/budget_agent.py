"""
Budget Agent — 负责计算旅行计划预算，超限时生成削减建议（不修改行程）

职责边界：
1. 读取 State 中的 raw_plan_text
2. 调用 calculate_budget_tool 精确计算各项费用
3. 对比 target_budget：
   - 未超限 → 输出 "OK" + 预算汇总
   - 超限 → 调用 suggest_savings_tool 生成削减建议 → 输出 "OVERSHOOT|{json}"
     （由 workflow_router 路由给 Planner Agent 执行修正，Budget 不修改 JSON）

设计理由：
- Budget Agent 的职责是"计算 + 建议"，不应越界修改行程 JSON
- 行程 JSON 修改由 Planner Agent 负责（它掌握完整 schema）
- workflow_router 作为确定性路由，统一处理超限回退逻辑
"""

import json
from langchain_core.tools import BaseTool
from app.agents.base import ReActAgent
from app.graph.state import TripState
from app.tools.budget_tools import calculate_budget_tool, suggest_savings_tool
from app.tools.llm import BUDGET_MAX_TOKENS


class BudgetAgent(ReActAgent):
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
{"status": "ok", "budget": {"total_attractions": N, "total_hotels": N, "total_meals": N, "total_transportation": N, "total_intercity_transport": N, "total": N}, "analysis": "预算分析总结"}

### 情况 B：超限（总费用 > target_budget）
1. 调用 suggest_savings_tool 获取削减建议
2. 输出：
{"status": "overshoot", "budget": {"total_attractions": N, "total_hotels": N, "total_meals": N, "total_transportation": N, "total_intercity_transport": N, "total": N}, "people_count": N, "target_budget_per_person": N, "target_budget_total": N, "target_budget": N, "overshoot_amount": N, "savings_suggestions": [{"category": "...", "current": N, "suggested": N, "saving": N, "description": "..."}], "analysis": "超限分析总结"}

重要：
- 不修改旅行计划 JSON
- Planner JSON 中的市内交通字段语义：budget.total_transportation 是目的地城市内全程交通单价；非自驾为每人这些天费用，自驾为每辆车这些天费用。
- 如果 Planner 未提供 budget.total_transportation，calculate_budget_tool 会按 transport_cost_per_day × travel_days 估算全程交通单价；非自驾再乘人数，自驾再乘车辆数。
- intercity_transport.total_cost 是出发地到目的地的往返交通团队总费用，计入 budget.total_intercity_transport。
- 跨城往返交通方式由用户固定选择，超预算时只能建议用户手动调整，不能要求 Planner 擅自切换。
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
        departure_city = getattr(request, "departure_city", "") or "未填写"
        accommodation = getattr(request, "accommodation", "经济型酒店")
        travel_days = getattr(request, "travel_days", 1)
        people_count = max(1, int(getattr(request, "people_count", 1) or 1))
        intercity_mode = getattr(request, "intercity_transport_mode", "high_speed_rail")
        target_budget_per_person = getattr(request, "target_budget", 0) or 0
        target_budget_total = target_budget_per_person * people_count if target_budget_per_person > 0 else 0

        hotel_cost_map = {
            "经济型酒店": 150, "舒适型酒店": 350, "豪华酒店": 800, "民宿": 200,
        }
        hotel_cost_per_night = hotel_cost_map.get(accommodation, 300)

        transport_cost_map = {
            "自驾": 100, "公共交通": 30, "步行": 0, "混合": 50,
        }
        transportation = getattr(request, "transportation", "公共交通")
        transport_cost_per_day = transport_cost_map.get(transportation, 50)

        budget_line = (
            f"人均预算上限：{target_budget_per_person} 元；团队总预算上限：{target_budget_total} 元"
            if target_budget_per_person > 0
            else "目标预算上限：无限制"
        )

        return f"""请计算以下旅行计划的预算：

目的地城市：{city}
出发地城市：{departure_city}
旅行天数：{travel_days} 天
出行人数：{people_count} 人
跨城往返交通方式：{intercity_mode}（用户固定选择，Planner 不应自动修改）
住宿偏好：{accommodation}（估算每晚 {hotel_cost_per_night} 元）
交通方式：{transportation}（缺失时按非自驾每人每天 / 自驾每辆车每天 {transport_cost_per_day} 元估算）
{budget_line}

费用字段语义：
- attractions[].ticket_price：单人票价。
- meals[].estimated_cost：单人单餐费用。
- hotel.estimated_cost：每间每晚费用。
- budget.total_transportation：目的地城市内全程交通单价；非自驾为每人这些天费用，自驾为每辆车这些天费用。
- 如果 Planner 未提供 budget.total_transportation，则使用 transport_cost_per_day × travel_days 估算全程交通单价。
- intercity_transport.total_cost：出发地到目的地往返交通团队总费用。

=== 旅行计划 JSON ===
{raw_plan_text}

请严格按照 system prompt 执行：
1. 调用 calculate_budget_tool(trip_plan_json=上面的JSON, hotel_cost_per_night={hotel_cost_per_night}, transport_cost_per_day={transport_cost_per_day}, people_count={people_count}, transportation="{transportation}")
2. 如果团队总预算上限 > 0 且总费用 > 团队总预算上限：
   → 调用 suggest_savings_tool(current_budget_json=上一步结果, target_budget={target_budget_total}, travel_days={travel_days}, people_count={people_count})
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
        request = state.get("request")
        people_count = max(1, int(getattr(request, "people_count", 1) or 1)) if request else 1
        target_budget_per_person = getattr(request, "target_budget", 0) or 0 if request else 0
        target_budget_total = target_budget_per_person * people_count if target_budget_per_person > 0 else 0

        output: dict = {
            "agent_outputs": {},
        }

        if status != "overshoot" and target_budget_total > 0 and total > target_budget_total:
            status = "overshoot"
            data["target_budget_per_person"] = target_budget_per_person
            data["target_budget_total"] = target_budget_total
            data["target_budget"] = target_budget_total
            data["people_count"] = people_count
            data["overshoot_amount"] = total - target_budget_total
            data.setdefault("savings_suggestions", [])

        if status == "overshoot":
            # 构造超限信号：Supervisor 通过 "OVERSHOOT|" 前缀检测
            target_budget = data.get("target_budget_total", data.get("target_budget", target_budget_total))
            target_budget_per_person = data.get("target_budget_per_person", target_budget_per_person)
            people_count = data.get("people_count", people_count)
            overshoot_amount = data.get("overshoot_amount", 0)
            savings_suggestions = data.get("savings_suggestions", [])

            overshoot_payload = {
                "target_budget": target_budget,
                "target_budget_per_person": target_budget_per_person,
                "target_budget_total": target_budget,
                "people_count": people_count,
                "intercity_transport_mode": getattr(request, "intercity_transport_mode", "high_speed_rail") if request else "high_speed_rail",
                "total_intercity_transport": budget.get("total_intercity_transport", 0),
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
            updated_raw_plan_text = self._replace_budget_in_raw_plan_text(
                state.get("raw_plan_text", ""),
                budget,
            )
            if updated_raw_plan_text:
                output["raw_plan_text"] = updated_raw_plan_text

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

    def _replace_budget_in_raw_plan_text(self, raw_plan_text: str, budget: dict) -> str:
        """
        未超预算时，将 Planner 原始 JSON 中的 budget 汇总替换为 Budget Tool 核算值。

        只替换顶层 budget 对象的五个汇总字段，不修改 days / attractions /
        meals / hotel 等行程结构，避免 Budget Agent 越界重规划行程。
        """
        if not raw_plan_text or not isinstance(budget, dict):
            return ""

        extracted = self._extract_json(raw_plan_text)
        if not extracted:
            return ""

        try:
            plan_data = json.loads(extracted)
        except Exception as e:
            print(f"  [budget_agent] ⚠️ raw_plan_text budget 回填失败，JSON 解析异常: {e}")
            return ""

        if not isinstance(plan_data, dict):
            return ""

        plan_data["budget"] = {
            "total_attractions": int(budget.get("total_attractions", 0) or 0),
            "total_hotels": int(budget.get("total_hotels", 0) or 0),
            "total_meals": int(budget.get("total_meals", 0) or 0),
            "total_transportation": int(budget.get("total_transportation", 0) or 0),
            "total_intercity_transport": int(budget.get("total_intercity_transport", 0) or 0),
            "total": int(budget.get("total", 0) or 0),
        }

        print("  [budget_agent] ✓ raw_plan_text.budget 已替换为 Budget Tool 核算结果")
        return json.dumps(plan_data, ensure_ascii=False, indent=2)
