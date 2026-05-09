"""
Planner Agent — 负责综合所有数据，生成完整的结构化旅行计划

设计思路：
1. 单次 LLM 调用完成规划 — prompt 包含完整 JSON schema + 格式化数据 + 任务要求
2. 数据用编号文本格式传递（非 json.dumps），大幅减少 token 消耗
3. 景点按空间距离贪心聚类分组，区域数 ≈ travel_days，同天景点不跨区
4. prompt 内联完整 JSON 模板，LLM 明确知道每个必填字段
5. 括号计数法精确提取嵌套 JSON
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
1. 从可用景点中选择合适的景点，每天安排 2-4 个
2. 使用景点列表中提供的真实坐标
3. 为每天安排早餐、午餐、晚餐（各一餐）
4. 每天的酒店从当天景点区域下的"附近酒店"中选取。
   不同区域的天必须选不同的酒店（即使房价相近）。
5. 生成合理的预算估算
6. 每天的景点必须来自同一区域分组（同一"### 区域名"下的景点）。
   不要跨区域选景点。景点已按空间距离分组并附带了附近酒店列表。

请严格按照以下 JSON 格式输出，不要添加任何其他内容，不要用 markdown 代码块:

{
  "city": "城市名称",
  "start_date": "开始日期",
  "end_date": "结束日期",
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
        return []

    @property
    def max_steps(self) -> int:
        return 1

    @property
    def max_tokens(self) -> int:
        return PLANNER_MAX_TOKENS

    @property
    def request_timeout(self) -> int:
        return PLANNER_REQUEST_TIMEOUT

    # ===== 覆盖 run()：单次 LLM 调用 =====

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
            system_prompt = _get_revision_prompt(state)
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

        pois_info = _format_attractions(raw_attractions, travel_days, raw_hotels)
        weather_info = _format_weather(raw_weather, travel_days)

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

## 可用景点 — 已按区域分组（每区含附近酒店）
{pois_info}

## 天气预报
{weather_info}"""

        # 预算修正模式附加信息
        phase = state.get("phase", "")
        if phase == "review":
            raw_plan_text = state.get("raw_plan_text", "")
            agent_outputs = state.get("agent_outputs", {})
            budget_value = agent_outputs.get("budget_agent", "")
            revision_round = state.get("revision_round", 0)

            budget_info_text = ""
            if isinstance(budget_value, str) and budget_value.startswith("OVERSHOOT|"):
                try:
                    payload_json = budget_value.split("|", 1)[1]
                    payload = json.loads(payload_json)
                    target = payload.get("target_budget", 0)
                    overshoot = payload.get("overshoot_amount", 0)
                    suggestions = payload.get("savings_suggestions", [])
                    print(f"budget的建议：{suggestions}")
                    budget_info_text = f"\n## 预算约束（第 {revision_round + 1} 次修正）\n"
                    budget_info_text += f"- 目标总预算上限: {target} 元（必须严格遵守）\n"
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
                except Exception:
                    budget_info_text = f"\n## 预算约束\n目标预算: 请参考 Budget Agent 的建议削减费用。\n"

            prompt += f"""

## 当前行程计划（需要修正费用以符合预算）===
{raw_plan_text}
{budget_info_text}"""

        return prompt


# ============================================================
# 数据格式化函数
# ============================================================

def _format_attractions(attractions: list[dict], travel_days: int, hotels: list[dict] | None = None) -> str:
    """格式化景点为按区域分组的编号文本 — 每区附加最近的酒店，LLM 直接在本区内选择"""
    if not attractions:
        return "暂无景点信息，请根据常识为城市生成合适的景点。"

    clusters = _greedy_cluster(attractions, target_clusters=max(1, travel_days))
    print(f"  [planner_agent] 贪心聚类: {len(attractions)} 个景点 → {len(clusters)} 个区域")

    # 为每个 cluster 挂载最近的酒店
    if hotels:
        for cluster in clusters:
            cluster["hotels"] = _find_nearest_hotels(cluster, hotels, top_n=3)
        hotel_count = sum(len(c.get("hotels", [])) for c in clusters)
        print(f"  [planner_agent] 酒店挂载: {len(hotels)} 个酒店 → {hotel_count} 个分配")

    lines: list[str] = []
    global_index = 0
    for cluster in clusters:
        name = cluster["name"]
        items = cluster["items"]
        nearby_hotels = cluster.get("hotels", [])
        hotel_count_str = f" + {len(nearby_hotels)} 家附近酒店" if nearby_hotels else ""

        if len(clusters) > 1:
            lines.append(f"### {name}（{len(items)} 个景点{hotel_count_str}）")
        else:
            lines.append(f"### 主要景点（{len(items)} 个景点{hotel_count_str}）")

        for poi in items:
            global_index += 1
            poi_name = poi.get("name", "未知")
            lng = poi.get("longitude", 0)
            lat = poi.get("latitude", 0)
            rating = poi.get("rating", 0)
            address = poi.get("address", "")
            ticket = poi.get("ticket_price", 0)
            desc = poi.get("description", "")

            lines.append(f"  {global_index}. {poi_name} | 坐标({lng}, {lat})")
            if rating:
                lines.append(f"     评分: {rating}")
            if address:
                lines.append(f"     地址: {address}")
            if ticket:
                lines.append(f"     门票: {ticket}元")
            if desc:
                lines.append(f"     简介: {desc}")

        # 输出本区域附近酒店
        if nearby_hotels:
            lines.append(f"  附近酒店（供本区域选择）:")
            for h in nearby_hotels:
                hotel_name = h.get("name", "未知")
                h_lng = h.get("longitude", 0)
                h_lat = h.get("latitude", 0)
                h_rating = h.get("rating", 0)
                h_price = h.get("price_range", "")
                h_type = h.get("type", "")
                h_addr = h.get("address", "")
                h_dist = h.get("_distance_km", 0)

                lines.append(f"    - {hotel_name} | 坐标({h_lng}, {h_lat}) | 距区域中心 {h_dist:.1f}km")
                if h_rating:
                    lines.append(f"      评分: {h_rating}")
                if h_price:
                    lines.append(f"      价格: {h_price}")
                if h_type:
                    lines.append(f"      类型: {h_type}")
                if h_addr:
                    lines.append(f"      地址: {h_addr}")
        lines.append("")

    return "\n".join(lines)


def _find_nearest_hotels(cluster: dict, hotels: list[dict], top_n: int = 3) -> list[dict]:
    """找到离 cluster 中心最近的 top_n 个酒店"""
    import math

    # 计算 cluster 中心
    items = cluster["items"]
    if not items:
        return []
    center_lng = sum(float(h.get("longitude", 0)) for h in items) / len(items)
    center_lat = sum(float(h.get("latitude", 0)) for h in items) / len(items)

    def dist_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlng / 2) ** 2)
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    scored: list[dict] = []
    for h in hotels:
        h_lng = float(h.get("longitude", 0))
        h_lat = float(h.get("latitude", 0))
        if h_lng == 0 and h_lat == 0:
            continue
        d = dist_km(center_lng, center_lat, h_lng, h_lat)
        entry = dict(h)
        entry["_distance_km"] = d
        scored.append(entry)

    scored.sort(key=lambda x: x["_distance_km"])
    return scored[:top_n]


def _greedy_cluster(
    attractions: list[dict],
    target_clusters: int = 3,
    min_threshold_km: float = 2.0,
    max_threshold_km: float = 15.0,
) -> list[dict]:
    """
    贪心空间聚类：将景点按地理位置分组，区域数 ≈ travel_days

    算法：遍历景点，计算与各 cluster 中心距离，
    < threshold → 加入最近 cluster，≥ threshold → 新建 cluster。
    如果 cluster 数 > target_clusters，逐步增大 threshold 合并。
    """
    import math

    if not attractions:
        return [{"name": "默认区域", "items": []}]

    def dist_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlng / 2) ** 2)
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    threshold = min_threshold_km
    clusters: list[dict] = []

    while threshold <= max_threshold_km:
        clusters = []
        for attr in attractions:
            lng = float(attr.get("longitude", 0))
            lat = float(attr.get("latitude", 0))
            if lng == 0 and lat == 0:
                continue

            best_idx = -1
            best_dist = float("inf")
            for idx, c in enumerate(clusters):
                d = dist_km(lng, lat, c["center_lng"], c["center_lat"])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_idx >= 0 and best_dist < threshold:
                c = clusters[best_idx]
                c["items"].append(attr)
                n = len(c["items"])
                c["center_lng"] = (c["center_lng"] * (n - 1) + lng) / n
                c["center_lat"] = (c["center_lat"] * (n - 1) + lat) / n
            else:
                clusters.append({
                    "center_lng": lng,
                    "center_lat": lat,
                    "items": [attr],
                })

        if len(clusters) <= target_clusters:
            break
        threshold += 2.0

    # 命名：取 cluster 内评分最高的景点名 + "及周边"
    for c in clusters:
        items = c["items"]
        if items:
            best = max(items, key=lambda x: (
                float(x.get("rating", 0)) if isinstance(x.get("rating"), (int, float))
                else (float(str(x.get("rating", "0")))
                      if str(x.get("rating", "")).replace(".", "", 1).replace("-", "", 1).isdigit()
                      else 0)
            ))
            c["name"] = best.get("name", "未知") + "及周边"
        else:
            c["name"] = "无景点"
        del c["center_lng"], c["center_lat"]

    clusters.sort(key=lambda c: len(c["items"]), reverse=True)
    return clusters


def _format_weather(weather: list[dict], travel_days: int) -> str:
    if not weather:
        return "暂无天气信息"

    lines = []
    for w in weather[:travel_days]:
        date = w.get("date", "")
        day_w = w.get("day_weather", "")
        night_w = w.get("night_weather", "")
        day_t = str(w.get("day_temp", "0")).replace("°C", "").replace("℃", "").replace("°", "").strip()
        night_t = str(w.get("night_temp", "0")).replace("°C", "").replace("℃", "").replace("°", "").strip()
        wind_d = w.get("wind_direction", "")
        wind_p = w.get("wind_power", "")

        lines.append(f"{date}: {day_w} {day_t}°C / {night_w} {night_t}°C  {wind_d} {wind_p}")

    return "\n".join(lines)


def _format_hotels(hotels: list[dict]) -> str:
    if not hotels:
        return "暂无酒店信息，请根据常识推荐合适的酒店。"

    # 按空间聚类分组，与景点区域格式一致（LLM 可据此为每天选对应区域的酒店）
    clusters = _greedy_cluster(hotels, target_clusters=min(len(hotels), 3), min_threshold_km=2.0, max_threshold_km=5.0)
    print(f"  [planner_agent] 酒店聚类: {len(hotels)} 个酒店 → {len(clusters)} 个区域")

    lines: list[str] = []
    for cluster in clusters:
        name = cluster["name"]
        items = cluster["items"]
        if len(clusters) > 1:
            lines.append(f"### {name}（{len(items)} 个酒店）")
        else:
            lines.append(f"### 酒店（{len(items)} 个）")

        for h in items:
            hotel_name = h.get("name", "未知")
            address = h.get("address", "")
            rating = h.get("rating", 0)
            price = h.get("price_range", "")
            htype = h.get("type", "")
            lng = h.get("longitude", 0)
            lat = h.get("latitude", 0)

            lines.append(f"  - {hotel_name} | 坐标({lng}, {lat})")
            if rating:
                lines.append(f"    评分: {rating}")
            if price:
                lines.append(f"    价格: {price}")
            if htype:
                lines.append(f"    类型: {htype}")
            if address:
                lines.append(f"    地址: {address}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# JSON 解析 + 修正 prompt
# ============================================================

def _get_revision_prompt(state: TripState) -> str:
    agent_outputs = state.get("agent_outputs", {})
    budget_value = agent_outputs.get("budget_agent", "")
    target_budget = 0
    overshoot_amount = 0
    savings_text = ""

    if isinstance(budget_value, str) and budget_value.startswith("OVERSHOOT|"):
        try:
            payload_json = budget_value.split("|", 1)[1]
            payload = json.loads(payload_json)
            target_budget = payload.get("target_budget", 0)
            overshoot_amount = payload.get("overshoot_amount", 0)
            suggestions = payload.get("savings_suggestions", [])
            for s in suggestions:
                savings_text += f"- {s.get('category', '')}: {s.get('description', '')}\n"
        except Exception:
            pass

    return f"""你是旅行规划师，正在根据预算反馈重新调整行程。

## 预算约束（必须遵守）
- 目标总预算上限: {target_budget} 元
- 当前超支金额: {overshoot_amount} 元
- 调整后总费用必须 ≤ {target_budget} 元

## Budget Agent 的削减建议（严格按此执行）
{savings_text if savings_text else '请根据常识削减费用'}

## 调整规则
1. 保持景点名称、地址、坐标不变（这些是真实数据）
2. 只改动费用相关字段: ticket_price, estimated_cost, hotel.type, hotel.estimated_cost, hotel.price_range, meals[].estimated_cost
3. 酒店降级示例: "豪华酒店"→"舒适型酒店"，estimated_cost 800→350
4. 景点削减: 将部分 ticket_price>100 的付费景点替换为 ticket_price=0 的免费景点
5. 餐饮调整: 晚餐 estimated_cost 80→40，午餐 50→30
6. 交通优化: 如需可改 transportation 字段
7. 每天仍保持 2-3 个景点，三餐不缺
8. budget 字段必须重新计算，total 必须 ≤ {target_budget}
9. 景点仍须来自同一区域分组
10. 不同区域的天必须选不同酒店，酒店须与当天景点区域匹配

## 输出格式
与首次规划完全相同，输出完整 JSON（包含 city/start_date/end_date/days/weather_info/overall_suggestions/budget 全部字段）。
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
