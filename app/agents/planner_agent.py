"""
Planner Agent — 负责综合所有数据，生成完整的结构化旅行计划

设计思路：
1. 单次 LLM 调用完成规划 — prompt 包含完整 JSON schema + 格式化数据 + 任务要求
2. 数据用编号文本格式传递（非 json.dumps），大幅减少 token 消耗
3. 算法逻辑（聚类、酒店匹配、格式化）已抽离至 app.utils.planner_utils
4. 括号计数法精确提取嵌套 JSON
"""

import json
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.llm import get_chat_model, PLANNER_MAX_TOKENS, PLANNER_REQUEST_TIMEOUT
from app.utils.planner_utils import (
    format_attractions,
    format_weather,
    format_intercity_transport,
    mode_label,
)


PLANNER_SYSTEM_PROMPT = """你是一个专业的旅行规划师。根据以下信息生成详细的旅行计划。

## 景点与酒店
- 每天从同一区域分组（"### 区域名"）中选 2-4 个景点，使用景点列表中提供的真实坐标，禁止跨区选景点
- 每天酒店从当天景点区域的"附近酒店"列表中选取，不同区域的天必须选不同酒店
- 每天三餐各一（早/午/晚），生成合理的预算估算

## 跨城交通
- 用户未填写出发地，intercity_transport 必须设为 null
- 无需遵守首末日到达/出发时间约束，每天正常安排景点即可

## 首末日时间约束（有出发地时必须遵守）
根据去程到达时间安排第一天：
  - 早于 10:00 → 全天 3-4 个景点
  - 10:00-14:00 → 仅下午 1-2 个景点，不安排上午景点
  - 晚于 14:00 → 傍晚 1 个景点或自由活动，首日 description 写"抵达日"
根据返程出发时间安排最后一天：
  - 晚于 18:00 → 全天 2-3 个景点
  - 14:00-18:00 → 上午/中午 1-2 个景点，须靠近出发站/机场
  - 早于 14:00 → 1 个近处景点或不安排，末日 description 写"返程日"
- 必须保留 Transport Agent 的 intercity_transport 结构，不得修改跨城交通方式
- budget.total_transportation = 城市内交通费用，budget.total_intercity_transport = 跨城往返交通费用，不得混淆

## 预算核算（单价制）
- 所有费用填基础单价，禁止乘以出行人数/房间数/车辆数
- ticket_price = 单人票价，meals[].estimated_cost = 单人单餐，hotel.estimated_cost = 每间每晚
- budget.total_transportation = 城市内全程交通单价（非自驾=每人这些天费用，自驾=每车这些天费用）
- budget.total_intercity_transport = intercity_transport.total_cost（跨城往返团队总费用）
- 考虑出行人数，不安排过密路线

## 室内备用计划
- 顶层 indoor_backup_attractions 必须生成，整个行程共 2-3 个（不是每天）
- 限博物馆/美术馆/科技馆/展览馆/商场/书店/室内市集等室内场所
- 不能和正式景点重名，不参与预算和路线，不需要 image_url
- 可用 POI 不足时可根据城市常识补全

严格按以下 JSON 格式输出，禁止 markdown 代码块和额外文字：

{
  "city": "城市名称",
  "departure_city": "出发地城市",
  "start_date": "开始日期",
  "end_date": "结束日期",
  "people_count": 1,
  "indoor_backup_attractions": [
    {
      "name": "室内备用景点名称",
      "address": "地址",
      "description": "简要说明",
      "category": "博物馆/美术馆/商场/展馆",
      "reason": "适合作为室内备用计划的原因",
      "estimated_duration": 90,
      "ticket_price": 0
    }
  ],
  "intercity_transport_mode": "driving | high_speed_rail | flight",
  "intercity_transport": {
    "mode": "driving | high_speed_rail | flight",
    "outbound": {
      "direction": "outbound",
      "origin": "出发城市",
      "destination": "目的地城市",
      "date": "YYYY-MM-DD",
      "mode": "driving | high_speed_rail | flight",
      "duration_minutes": 0,
      "distance_km": 0,
      "estimated_cost": 0,
      "route_summary": "去程说明",
      "notes": ["说明"]
    },
    "return_trip": {
      "direction": "return",
      "origin": "目的地城市",
      "destination": "出发城市",
      "date": "YYYY-MM-DD",
      "mode": "driving | high_speed_rail | flight",
      "duration_minutes": 0,
      "distance_km": 0,
      "estimated_cost": 0,
      "route_summary": "返程说明",
      "notes": ["说明"]
    },
    "total_cost": 0,
    "total_duration_minutes": 0,
    "summary": "往返交通摘要",
    "warnings": []
  },
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0（必须从0开始）,
      "description": "当日行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397, "latitude": 39.916},
        "price_range": "价格范围",
        "rating": "评分"（number类型）,
        "distance": "距景点距离",
        "type": "酒店类型",
        "estimated_cost": 300
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "景点地址",
          "location": {"longitude": 116.397, "latitude": 39.916},
          "visit_duration": 120,
          "description": "景点描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐名称", "description": "描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐名称", "description": "描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐名称", "description": "描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD", "day_weather": "晴", "night_weather": "多云",
      "day_temp": 25, "night_temp": 15, "wind_direction": "南风", "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体旅行建议",
  "budget": {
    "total_attractions": 0,
    "total_hotels": 0,
    "total_meals": 0,
    "total_transportation": 0,
    "total_intercity_transport": 0,
    "total": 0
  }
}
每个字段必填，禁止省略或将嵌套对象简化为字符串或数字。"""


class PlannerAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "planner_agent"

    @property
    def description(self) -> str:
        return (
            "行程规划专家。单次 LLM 调用综合景点、天气、酒店和用户偏好，"
            "生成结构化的每日行程计划。负责填充 raw_plan_text 字段。"
        )

    @property
    def system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    @property
    def max_tokens(self) -> int:
        return PLANNER_MAX_TOKENS

    @property
    def request_timeout(self) -> int:
        return PLANNER_REQUEST_TIMEOUT

    # ===== OVERSHOOT 解析（_build_human_message 和 _get_revision_prompt 共用）=====

    def _parse_overshoot_payload(self, state: TripState) -> Optional[dict]:
        """从 state 中解析 Budget Agent 的 OVERSHOOT 反馈负载"""
        agent_outputs = state.get("agent_outputs", {})
        budget_value = agent_outputs.get("budget_agent", "")
        if not isinstance(budget_value, str) or not budget_value.startswith("OVERSHOOT|"):
            return None
        try:
            payload_json = budget_value.split("|", 1)[1]
            return json.loads(payload_json)
        except Exception:
            return None

    # ===== 核心执行 =====

    def run(self, state: TripState) -> dict:
        request = state.get("request")
        if request is None:
            return {
                "raw_plan_text": "",
                "error": "State 中缺少 request 字段",
                "agent_outputs": {self.name: "失败: 缺少 request"},
            }

        phase = state.get("phase", "")
        is_revision = (phase == "review")

        if is_revision:
            system_prompt = _get_revision_prompt(state, self._parse_overshoot_payload(state))
        else:
            system_prompt = self.system_prompt

        context = self._build_human_message(request, state)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ]

        action = "修正行程（预算约束）" if is_revision else "生成行程"
        print(f"\n{'=' * 50}")
        print(f"🤖 {self.name} 单次 LLM 调用{action}...")
        print(f"{'=' * 50}")

        model = get_chat_model(
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )
        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        print(f"  [{self.name}] LLM 返回 {len(content)} 字符")

        raw_plan_text = _parse_response(content)
        error = ""

        test_json = self._extract_json(raw_plan_text)
        if test_json:
            try:
                data = json.loads(test_json)
                if "days" not in data or len(data.get("days", [])) == 0:
                    error = "生成的行程 JSON 缺少 days 字段"
            except Exception:
                error = "生成的行程 JSON 解析失败"
        else:
            error = "未能从 LLM 输出中提取有效 JSON"

        print(f"{'=' * 50}\n")

        agent_msg = f"行程修正完成 ({len(raw_plan_text)} 字符)" if is_revision else f"行程生成完成 ({len(raw_plan_text)} 字符)"
        print(f"行程信息：{raw_plan_text}")

        return {
            "raw_plan_text": raw_plan_text,
            "error": error,
            "agent_outputs": {self.name: agent_msg},
        }

    def _build_human_message(self, request, state: TripState) -> str:
        city = getattr(request, "city", "未知")
        departure_city = getattr(request, "departure_city", "") or ""
        start_date = getattr(request, "start_date", "")
        end_date = getattr(request, "end_date", "")
        travel_days = getattr(request, "travel_days", 1)
        people_count = max(1, int(getattr(request, "people_count", 1) or 1))
        intercity_mode = getattr(request, "intercity_transport_mode", "high_speed_rail")
        transportation = getattr(request, "transportation", "公共交通")
        accommodation = getattr(request, "accommodation", "经济型酒店")
        preferences = getattr(request, "preferences", [])
        free_text = getattr(request, "free_text_input", "")

        raw_attractions = state.get("raw_attractions", [])
        raw_weather = state.get("raw_weather", [])
        raw_hotels = state.get("raw_hotels", [])
        raw_intercity_transport = state.get("raw_intercity_transport", {})

        pois_info = format_attractions(raw_attractions, travel_days, raw_hotels)
        weather_info = format_weather(raw_weather, travel_days)
        intercity_info = format_intercity_transport(raw_intercity_transport)

        prompt = f"""## 基本信息
- 出发地城市: {departure_city if departure_city else '未填写'}
- 城市: {city}
- 日期范围: {start_date} 至 {end_date}
- 旅行天数: {travel_days} 天
- 出行人数: {people_count} 人
- 往返交通方式: {mode_label(intercity_mode)}（用户指定，禁止擅自更改）
- 交通方式: {transportation}
- 住宿偏好: {accommodation}
- 用户偏好: {', '.join(preferences) if preferences else '无特殊偏好'}"""

        if free_text:
            prompt += f"\n- 额外要求: {free_text}"

        prompt += f"""

## 可用景点 — 已按区域分组（每区含附近酒店）
{pois_info}

## 出发地到目的地往返交通
{intercity_info}

## 天气预报
{weather_info}"""

        # 预算修正模式附加信息
        phase = state.get("phase", "")
        if phase == "review":
            raw_plan_text = state.get("raw_plan_text", "")
            revision_round = state.get("revision_round", 0)
            payload = self._parse_overshoot_payload(state)

            budget_info_text = ""
            if payload:
                target = payload.get("target_budget_total", payload.get("target_budget", 0))
                per_person = payload.get("target_budget_per_person", 0)
                people = payload.get("people_count", people_count)
                overshoot = payload.get("overshoot_amount", 0)
                suggestions = payload.get("savings_suggestions", [])
                print(f"budget的建议：{suggestions}")
                budget_info_text = f"\n## 预算约束（第 {revision_round + 1} 次修正）\n"
                budget_info_text += f"- 目标总预算上限: {target} 元（必须严格遵守）\n"
                budget_info_text += f"- 出行人数: {people} 人；人均预算上限: {per_person} 元\n"
                budget_info_text += f"- 当前超支金额: {overshoot} 元\n"
                budget_info_text += f"- 已修正轮数: {revision_round}\n"
                budget_info_text += f"\n### Budget Agent 的削减建议（按此执行）:\n"
                for s in suggestions:
                    cat = s.get("category", "")
                    curr = s.get("current", 0)
                    suggested = s.get("suggested", 0)
                    saving = s.get("saving", 0)
                    desc = s.get("description", "")
                    budget_info_text += f"- [{cat}] {curr}元 → {suggested}元（节省 {saving}元）: {desc}\n"
            else:
                budget_info_text = f"\n## 预算约束\n目标预算: 请参考 Budget Agent 的建议削减费用。\n"

            prompt += f"""

## 当前行程计划（需要修正费用以符合预算）===
{raw_plan_text}
{budget_info_text}"""

        return prompt


# ============================================================
# 模块级辅助函数
# ============================================================

def _get_revision_prompt(state: TripState, payload: Optional[dict] = None) -> str:
    """构造预算修正的系统提示词，payload 由 PlannerAgent._parse_overshoot_payload() 提供"""
    target_budget = 0
    target_budget_per_person = 0
    people_count = 1
    overshoot_amount = 0
    savings_text = ""

    if payload:
        target_budget = payload.get("target_budget_total", payload.get("target_budget", 0))
        target_budget_per_person = payload.get("target_budget_per_person", 0)
        people_count = payload.get("people_count", 1)
        overshoot_amount = payload.get("overshoot_amount", 0)
        suggestions = payload.get("savings_suggestions", [])
        for s in suggestions:
            savings_text += f"- {s.get('category', '')}: {s.get('description', '')}\n"

    return f"""你是旅行规划师，正在根据预算反馈重新调整行程。

## 预算约束（必须遵守）
- 目标总预算上限: {target_budget} 元
- 出行人数: {people_count} 人
- 人均预算上限: {target_budget_per_person} 元
- 当前超支金额: {overshoot_amount} 元
- 调整后总费用必须 ≤ {target_budget} 元

## Budget Agent 的削减建议（严格按此执行）
{savings_text if savings_text else '请根据常识削减费用'}

## 调整规则
1. 保持景点名称、地址、坐标不变（这些是真实数据）
2. 只改动费用相关字段: ticket_price, estimated_cost, hotel.type, hotel.estimated_cost, hotel.price_range, meals[].estimated_cost
3. ticket_price 必须是单人票价，不能乘以出行人数
4. meals[].estimated_cost 必须是单人单餐费用，不能乘以出行人数
5. hotel.estimated_cost 必须是每间每晚费用，不能乘以房间数
6. budget.total_transportation 必须是目的地城市内全程交通单价：非自驾为每人这些天费用，自驾为每辆车这些天费用
7. budget.total_intercity_transport 必须保留跨城往返交通团队总费用，不得擅自改变用户选择的跨城交通方式
8. 酒店降级示例: "豪华酒店"→"舒适型酒店"，hotel.estimated_cost 800→350
9. 景点削减: 将部分 ticket_price>100 的付费景点替换为 ticket_price=0 的免费景点
10. 餐饮调整: 晚餐 estimated_cost 80→40，午餐 50→30
11. 交通优化: 如需可改目的地内 transportation 字段或降低 budget.total_transportation 全程交通单价，但禁止更改 intercity_transport_mode
12. 每天仍保持 2-3 个景点，三餐不缺
13. budget.total 可以作为草稿，但所有单价必须能让 Budget Agent 复核后的团队总费用 ≤ {target_budget}
14. 景点仍须来自同一区域分组
15. 不同区域的天必须选不同酒店，酒店须与当天景点区域匹配
16. 必须保留顶层 indoor_backup_attractions，整个行程 2-3 个室内备用景点，不能和任何正式景点重名，不参与预算核算，不用为了降预算删除

## 输出格式
与首次规划完全相同，输出完整 JSON（包含 city/departure_city/start_date/end_date/people_count/indoor_backup_attractions/intercity_transport_mode/intercity_transport/days/weather_info/overall_suggestions/budget 全部字段）。
不要添加 markdown 代码块，直接输出纯 JSON。"""


def _parse_response(response: str) -> str:
    """括号计数法提取 JSON"""
    text = response.strip()

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    open_brace = text.find("{")
    if open_brace == -1:
        return text

    brace_count = 0
    close_brace = open_brace
    for i, ch in enumerate(text[open_brace:], open_brace):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                close_brace = i + 1
                break

    return text[open_brace:close_brace]
