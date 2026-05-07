"""
Hotel Agent — 负责智能搜索和推荐目的地酒店

职责：
1. 分析用户的住宿偏好（如"经济型酒店"、"豪华酒店"、"民宿"）
2. 如有预算限制，在搜索时考虑价格因素
3. 调用 search_hotels_tool 执行搜索
4. 如果搜索结果不理想，调整关键词或增加搜索范围
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
            "酒店推荐专家。根据用户的住宿偏好和预算智能搜索酒店，"
            "支持按价位筛选和多次搜索，负责填充 raw_hotels 字段。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是专业的酒店推荐专家。你的任务是：

1. 根据用户的住宿偏好（accommodation）制定搜索策略
2. 调用 search_hotels_tool 搜索对应类型的酒店
3. 如果第一次搜索结果不够理想（< 3 条），尝试调整搜索关键词
4. 综合评估搜索结果，选出最适合用户的酒店

住宿类型与搜索关键词映射：
- "经济型酒店" → 搜索"经济型酒店"、"快捷酒店"、"青年旅舍"
- "舒适型酒店" → 搜索"舒适型酒店"、"商务酒店"、"三星级酒店"
- "豪华酒店" → 搜索"豪华酒店"、"五星级酒店"、"度假酒店"
- "民宿" → 搜索"民宿"、"客栈"、"短租公寓"

重要规则：
- 必须至少搜索一次
- 首次搜索用用户的住宿偏好作为关键词
- 如果结果不足（< 3 条），换一个相关关键词补搜
- 输出必须是有效 JSON，格式为:
  {"summary": "搜索总结", "keywords_used": ["关键词1"], "total_found": N, "hotels": [...]}
  其中 hotels 是去重合并后的酒店列表
  """

    @property
    def tools(self) -> list[BaseTool]:
        return [search_hotels_tool]

    @property
    def max_steps(self) -> int:
        return 5  # 酒店搜索可能需要多轮

    def _build_context_message(self, state: TripState) -> str:
        """构造酒店搜索的上下文信息"""
        request = state.get("request")
        if request is None:
            return "错误：State 中缺少 request 字段"

        city = getattr(request, "city", "未知")
        accommodation = getattr(request, "accommodation", "经济型酒店")
        free_text = getattr(request, "free_text_input", "")

        context = f"""请为以下旅行需求搜索酒店：

目的地城市：{city}
住宿偏好：{accommodation}"""

        if free_text:
            context += f"\n用户额外要求：{free_text}"

        context += f"""

请首先用住宿偏好 "{accommodation}" 作为关键词调用 search_hotels_tool 进行第一次搜索。
搜索完成后评估结果数量，如果不足 3 条，换一个相关关键词补搜。"""

        return context

    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list,
    ) -> dict:
        """从 LLM 最终输出中提取酒店数据，写入 raw_hotels"""
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

        summary = f"共搜索到 {len(deduped)} 个去重酒店"

        return {
            "raw_hotels": deduped,
            "agent_outputs": {self.name: summary},
        }
