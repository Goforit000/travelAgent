"""
Planner Agent — 负责综合所有数据，生成完整的结构化旅行计划

设计思路（参照参考代码）：
1. 单次 LLM 调用完成规划 — prompt 包含完整 JSON schema + 格式化数据 + 任务要求
2. 数据用编号文本格式传递（非 json.dumps），大幅减少 token 消耗
3. prompt 内联完整 JSON 模板，LLM 明确知道每个必填字段
4. 括号计数法精确提取嵌套 JSON
5. Fallback 复用真实 POI 数据
"""

import json
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.llm import get_chat_model, PLANNER_MAX_TOKENS, PLANNER_REQUEST_TIMEOUT


PLANNER_SYSTEM_PROMPT = """你是一个专业的旅行规划师。

根据以下信息，生成一份详细的旅行计划。

## 任务要求
1. 从可用景点中选择合适的景点，每天安排 2-3 个
2. 使用景点列表中提供的真实坐标
3. 为每天安排早餐、午餐、晚餐（各一餐）
4. 推荐合适的酒店
5. 生成合理的预算估算

请严格按照以下 JSON 格式输出，不要添加任何其他内容，不要用 markdown 代码块:

{
  "city": "城市名称",
  "start_date": "开始日期",
  "end_date": "结束日期",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "当日行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397, "latitude": 39.916},
        "price_range": "价格范围",
        "rating": "评分",
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
    "total": 0
  }
}

重要：每个字段都必须填写，禁止省略。禁止将嵌套对象简化为字符串或数字。"""


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
    def tools(self) -> list[BaseTool]:
        return []  # 单次生成，不需要工具

    @property
    def max_steps(self) -> int:
        return 1  # 单次 LLM 调用

    @property
    def max_tokens(self) -> int:
        return PLANNER_MAX_TOKENS

    @property
    def request_timeout(self) -> int:
        return PLANNER_REQUEST_TIMEOUT

    # ===== 覆盖 run()：单次 LLM 调用，不使用 ReAct 循环 =====

    def run(self, state: TripState) -> dict:
        """
        单次 LLM 调用生成行程计划

        跳过 BaseAgent 的 ReAct 循环，直接用 SystemPrompt + 格式化数据
        构造一个完整的 prompt，一次性生成旅行计划 JSON。
        """
        request = state.get("request")
        if request is None:
            return {
                "raw_plan_text": "",
                "error": "State 中缺少 request 字段",
                "agent_outputs": {self.name: "失败: 缺少 request"},
            }

        # 格式化数据为人类可读文本
        context = self._build_human_message(request, state)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=context),
        ]

        print(f"\n{'=' * 50}")
        print(f"🤖 {self.name} 单次 LLM 调用生成行程...")
        print(f"{'=' * 50}")

        model = get_chat_model(
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )
        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        print(f"  [{self.name}] LLM 返回 {len(content)} 字符")

        # 解析响应
        raw_plan_text = _parse_response(content, request)
        error = ""

        # 如果解析出的 JSON 不完整，记录错误
        test_json = _extract_json(raw_plan_text)
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

        return {
            "raw_plan_text": raw_plan_text,
            "error": error,
            "agent_outputs": {
                self.name: f"行程生成完成 ({len(raw_plan_text)} 字符)",
            },
        }

    def _build_human_message(self, request, state: TripState) -> str:
        """构造格式化的人类可读上下文"""
        city = getattr(request, "city", "未知")
        start_date = getattr(request, "start_date", "")
        end_date = getattr(request, "end_date", "")
        travel_days = getattr(request, "travel_days", 1)
        transportation = getattr(request, "transportation", "公共交通")
        accommodation = getattr(request, "accommodation", "经济型酒店")
        preferences = getattr(request, "preferences", [])
        free_text = getattr(request, "free_text_input", "")

        raw_attractions = state.get("raw_attractions", [])
        raw_weather = state.get("raw_weather", [])
        raw_hotels = state.get("raw_hotels", [])

        # 格式化数据
        pois_info = _format_attractions(raw_attractions, travel_days)
        weather_info = _format_weather(raw_weather, travel_days)
        hotels_info = _format_hotels(raw_hotels)

        prompt = f"""## 基本信息
- 城市: {city}
- 日期范围: {start_date} 至 {end_date}
- 旅行天数: {travel_days} 天
- 交通方式: {transportation}
- 住宿偏好: {accommodation}
- 用户偏好: {', '.join(preferences) if preferences else '无特殊偏好'}"""

        if free_text:
            prompt += f"\n- 额外要求: {free_text}"

        prompt += f"""

## 可用景点 (POI)
{pois_info}

## 天气预报
{weather_info}

## 可用酒店
{hotels_info}"""

        return prompt


# ============================================================
# 数据格式化函数（模块级，可复用）
# ============================================================

def _format_attractions(attractions: list[dict], travel_days: int) -> str:
    """格式化景点为编号文本列表 — 人类可读，LLM 易于理解"""
    if not attractions:
        return "暂无景点信息，请根据常识为城市生成合适的景点。"

    # 按评分排序，取 top N
    def sort_key(item: dict) -> float:
        rating = item.get("rating", 0)
        if isinstance(rating, str):
            try:
                return float(rating)
            except (ValueError, TypeError):
                return 0.0
        if isinstance(rating, (int, float)):
            return float(rating)
        return 0.0

    sorted_items = sorted(attractions, key=sort_key, reverse=True)
    limit = travel_days * 3 + 2
    if len(sorted_items) > limit:
        print(f"  [planner_agent] 景点从 {len(sorted_items)} 条裁剪到 {limit} 条")
        sorted_items = sorted_items[:limit]

    lines = []
    for i, poi in enumerate(sorted_items, 1):
        name = poi.get("name", "未知")
        address = poi.get("address", "")
        lng = poi.get("longitude", 0)
        lat = poi.get("latitude", 0)
        rating = poi.get("rating", "")

        lines.append(f"{i}. {name}")
        if address:
            lines.append(f"   地址: {address}")
        lines.append(f"   坐标: ({lng}, {lat})")
        if rating:
            lines.append(f"   评分: {rating}")
        # 门票价格（如有）
        ticket = poi.get("ticket_price", 0)
        if ticket:
            lines.append(f"   门票: {ticket}元")
        # 描述（如有）
        desc = poi.get("description", "")
        if desc:
            lines.append(f"   简介: {desc}")
        lines.append("")

    return "\n".join(lines)


def _format_weather(weather: list[dict], travel_days: int) -> str:
    """格式化天气为文本"""
    if not weather:
        return "暂无天气信息"

    lines = []
    for w in weather[:travel_days]:
        date = w.get("date", "")
        day_w = w.get("day_weather", "")
        night_w = w.get("night_weather", "")
        day_t = w.get("day_temp", "0")
        night_t = w.get("night_temp", "0")
        wind_d = w.get("wind_direction", "")
        wind_p = w.get("wind_power", "")

        # 温度可能带单位，格式化为纯数字
        for temp_val in [day_t, night_t]:
            if isinstance(temp_val, str):
                temp_val = temp_val.replace("°C", "").replace("℃", "").replace("°", "").strip()

        lines.append(f"{date}: {day_w} {day_t}°C / {night_w} {night_t}°C  {wind_d} {wind_p}")

    return "\n".join(lines)


def _format_hotels(hotels: list[dict]) -> str:
    """格式化酒店为编号文本列表"""
    if not hotels:
        return "暂无酒店信息，请根据常识推荐合适的酒店。"

    limit = 5
    items = hotels[:limit]

    lines = []
    for i, h in enumerate(items, 1):
        name = h.get("name", "未知")
        address = h.get("address", "")
        rating = h.get("rating", "")
        price = h.get("price_range", "")
        htype = h.get("type", "")
        lng = h.get("longitude", 0)
        lat = h.get("latitude", 0)

        lines.append(f"{i}. {name}")
        if address:
            lines.append(f"   地址: {address}")
        if rating:
            lines.append(f"   评分: {rating}")
        if price:
            lines.append(f"   价格: {price}")
        if htype:
            lines.append(f"   类型: {htype}")
        lines.append(f"   坐标: ({lng}, {lat})")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# JSON 解析（括号计数法）
# ============================================================

def _parse_response(response: str, request) -> str:
    """
    从 LLM 响应中提取 JSON 字符串

    使用括号计数法精确匹配嵌套 JSON 边界，
    比 rfind("}") 更可靠地处理多层嵌套。
    """
    text = response.strip()

    # 去除 markdown 代码块
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

    # 括号计数法查找完整 JSON
    open_brace = text.find("{")
    if open_brace == -1:
        return text  # 无 JSON，返回原始文本

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


def _extract_json(text: str) -> str | None:
    """从文本中提取 JSON 字符串（备用方法）"""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "{" in text and "}" in text:
        brace_count = 0
        start = text.find("{")
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        return text[start:end] if end > start else None
    return None
