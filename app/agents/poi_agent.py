"""
POI Agent — 负责智能搜索目的地的景点信息

职责：
1. 分析用户的旅行偏好（如"历史文化"、"自然风光"、"美食"等）
2. 制定搜索策略：将偏好拆分为多轮关键词搜索
3. 调用 search_attractions_tool 执行搜索
4. 评估搜索结果是否充分，不够则补搜
5. 合并去重所有搜索结果，写入 raw_attractions
"""

import json
from langchain_core.tools import BaseTool
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.amap import search_attractions_tool


class POIAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "poi_agent"

    @property
    def description(self) -> str:
        return (
            "景点搜索专家。根据用户的旅行偏好智能拆分搜索关键词，"
            "支持多轮搜索并合并去重结果。负责填充 raw_attractions 字段。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是专业的景点搜索专家。你的任务是：

1. 分析用户的旅行偏好（preferences），制定搜索策略
2. 调用 search_attractions_tool 对每个偏好关键词分别搜索
3. 如果第一次搜索返回的结果不够丰富（少于5条），考虑换一个近义词或更具体的关键词再搜
4. 确保搜索结果覆盖了用户的所有偏好类别
5. 在所有搜索完成后，整理并汇总所有找到的景点

搜索策略指南：
- 偏好"历史文化" → 搜索"历史文化"、"博物馆"、"古迹"、"寺庙"
- 偏好"自然风光" → 搜索"自然风光"、"公园"、"山"、"湖"
- 偏好"美食" → 搜索"美食街"、"特色小吃"、"夜市"
- 偏好"购物" → 搜索"购物中心"、"步行街"、"特产"
- 偏好"艺术" → 搜索"美术馆"、"艺术区"、"画廊"
- 偏好"休闲" → 搜索"咖啡馆"、"茶馆"、"SPA"

重要规则：
- 必须至少搜索一次，最多搜索 4 次（不同关键词）
- 每次搜索后评估结果，如果返回为空或很少（< 3 条），立即换关键词补搜
- 如果你认为搜索结果已经充分覆盖了所有偏好，直接输出最终结果
- 输出必须是有效 JSON，格式为:
  {"summary": "搜索总结", "keywords_used": ["关键词1", "关键词2"], "total_found": N, "attractions": [...]}
  其中 attractions 是去重合并后的景点列表（使用之前所有搜索的原始结果）"""

    @property
    def tools(self) -> list[BaseTool]:
        return [search_attractions_tool]

    @property
    def max_steps(self) -> int:
        return 5  # POI 搜索可能需要多轮

    def _build_context_message(self, state: TripState) -> str:
        """构造 POI 搜索的上下文信息"""
        request = state.get("request")
        if request is None:
            return "错误：State 中缺少 request 字段"

        city = getattr(request, "city", "未知")
        preferences = getattr(request, "preferences", [])
        free_text = getattr(request, "free_text_input", "")

        context = f"""请为以下旅行需求搜索景点：

目的地城市：{city}
旅行偏好：{', '.join(preferences) if preferences else '无特别偏好（搜索"热门景点"）'}"""

        if free_text:
            context += f"\n用户额外要求：{free_text}"

        context += f"""

请首先分析偏好列表，然后对每个偏好分别调用 search_attractions_tool。
当前偏好关键词列表：{json.dumps(preferences, ensure_ascii=False)}

如果偏好为空，请用"热门景点"作为关键词搜索。"""

        return context

    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list,
    ) -> dict:
        """从 LLM 最终输出中提取景点数据，写入 raw_attractions"""
        try:
            data = json.loads(llm_content)
            attractions = data.get("attractions", [])
        except (json.JSONDecodeError, Exception):
            # 尝试用 _extract_json 提取
            extracted = self._extract_json(llm_content)
            if extracted:
                try:
                    data = json.loads(extracted)
                    attractions = data.get("attractions", [])
                except Exception:
                    attractions = []
            else:
                attractions = []

        if not attractions:
            return {
                "raw_attractions": [],
                "agent_outputs": {self.name: "搜索完成但未找到景点数据"},
            }

        # 去重（按 name 去重，保留首次出现）
        seen_names: set[str] = set()
        deduped: list[dict] = []
        for attr in attractions:
            name = attr.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(attr)

        summary = f"共搜索到 {len(deduped)} 个去重景点"

        return {
            "raw_attractions": deduped,
            "agent_outputs": {self.name: summary},
        }
