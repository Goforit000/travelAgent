"""
Hotel Agent — 负责智能搜索和推荐目的地酒店

职责：
1. 分析用户的住宿偏好（如"经济型酒店"、"豪华酒店"、"民宿"）
2. 制定多区域搜索策略，确保空间多样性
3. 调用 search_hotels_tool 执行搜索
4. 控制数量在 travel_days * 3 以内
5. 将酒店数据格式化后写入 raw_hotels
"""

import json
from langchain_core.tools import BaseTool
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.amap import search_hotels_tool


class HotelAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "hotel_agent"

    @property
    def description(self) -> str:
        return (
            "酒店推荐专家。根据用户的住宿偏好和预算智能多区域搜索酒店，"
            "确保空间多样性，负责填充 raw_hotels 字段。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是专业的酒店推荐专家。你的任务是：

1. 根据用户的住宿偏好（accommodation）制定多区域搜索策略
2. 调用 search_hotels_tool 搜索对应类型的酒店
3. 确保搜索结果覆盖目的地的 2-3 个不同区域，不要全集中在同一区域
4. 控制去重后酒店总数不超过 target_count，达到后立即停止搜索

搜索策略指南（保证空间多样性）：
- 第一次：用住宿偏好直接搜索（如"经济型酒店"）
- 如果结果集中在同一区域（酒店间距离 <2km），换区域关键词搜索
- 区域关键词示例：
  * 北京的"海淀区酒店"、"朝阳区酒店"、"西城区酒店"
  * 上海的"浦东酒店"、"静安酒店"、"徐汇酒店"
  * 一般城市"市中心酒店"、"火车站附近酒店"、"大学城附近酒店"
- 至少搜索 2 次，确保覆盖 2-3 个不同区域

重要规则：
- 必须至少搜索 1 次（不同关键词或不同区域），保证空间覆盖
- 如果已收集的去重后酒店数达到 target_count，立即停止搜索
- 如果结果不足（< 3 条），换一个相关关键词或另一个区域补搜
- 按出行人数调整住宿搜索偏好：1-2 人优先普通房型；3-4 人优先家庭房、套房、民宿；5 人以上优先民宿、公寓酒店、套房或多房间酒店
- 输出必须是有效 JSON，格式为:
  {"summary": "搜索总结", "keywords_used": ["关键词1"], "total_found": N, "hotels": [...]}
  其中 hotels 是去重合并后的酒店列表"""

    @property
    def tools(self) -> list[BaseTool]:
        return [search_hotels_tool]

    @property
    def max_steps(self) -> int:
        return 5

    def _build_context_message(self, state: TripState) -> str:
        """构造酒店搜索的上下文信息"""
        request = state.get("request")
        if request is None:
            return "错误：State 中缺少 request 字段"

        city = getattr(request, "city", "未知")
        accommodation = getattr(request, "accommodation", "经济型酒店")
        travel_days = getattr(request, "travel_days", 1)
        people_count = max(1, int(getattr(request, "people_count", 1) or 1))
        free_text = getattr(request, "free_text_input", "")
        target_count = travel_days * 5
        if people_count <= 2:
            room_hint = "1-2 人：优先普通大床房/双床房，兼顾交通便利"
        elif people_count <= 4:
            room_hint = "3-4 人：优先家庭房、套房、民宿，可搜索“家庭房”“套房”“民宿”"
        else:
            room_hint = "5 人以上：优先民宿、公寓酒店、套房或多房间酒店，可搜索“民宿”“公寓酒店”“多房间”"

        context = f"""请为以下旅行需求搜索酒店：

目的地城市：{city}
住宿偏好：{accommodation}
出行人数：{people_count} 人
旅行天数：{travel_days} 天"""

        if free_text:
            context += f"\n用户额外要求：{free_text}"

        context += f"""

目标收集酒店数量：{target_count} 个（每天 1 个 × {travel_days} 天 + 备用）。
住宿房型策略：{room_hint}
收集到 {target_count} 个去重酒店后立即停止搜索，多搜无益。

请首先用住宿偏好 "{accommodation}" 作为关键词调用 search_hotels_tool 进行第一次搜索。
搜索完成后：
1. 评估结果的区域分布是否足够多样化（至少覆盖 2 个不同区域）
2. 如果集中在同一区域，换一个区域关键词再次搜索
3. 如果去重后数量已达 target_count，停止并输出结果
确保搜索到的酒店覆盖 {city} 的 2-3 个不同区域。"""

        return context

    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list,
    ) -> dict:
        """从 LLM 最终输出中提取酒店数据，写入 raw_hotels"""
        request = state.get("request")
        travel_days = getattr(request, "travel_days", 1) if request else 1
        target_count = travel_days * 5

        try:
            data = json.loads(llm_content)
            hotels = data.get("hotels", [])
        except (json.JSONDecodeError, Exception):
            extracted = self._extract_json(llm_content)
            if extracted:
                try:
                    data = json.loads(extracted)
                    hotels = data.get("hotels", [])
                except Exception:
                    hotels = []
            else:
                hotels = []

        if not hotels:
            return {
                "raw_hotels": [],
                "agent_outputs": {self.name: "搜索完成但未找到酒店数据"},
            }

        # 去重（按 name 去重）
        seen_names: set[str] = set()
        deduped: list[dict] = []
        for hotel in hotels:
            name = hotel.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(hotel)

        # 按评分降序
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

        deduped.sort(key=sort_key, reverse=True)

        # 截断到目标数量
        if len(deduped) > target_count:
            print(f"  [hotel_agent] 酒店从 {len(deduped)} 条截断到 {target_count} 条")
            deduped = deduped[:target_count]

        summary = f"共搜索到 {len(deduped)} 个去重酒店（目标 {target_count} 个）"

        return {
            "raw_hotels": deduped,
            "agent_outputs": {self.name: summary},
        }
