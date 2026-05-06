"""
Weather Agent — 负责查询目的地的天气预报

职责：
1. 根据用户选择的目的地城市查询天气
2. 处理城市名称（如用户输入简称或别名，尽量用标准名称查询）
3. 将天气数据格式化后写入 raw_weather
"""

import json
from langchain_core.tools import BaseTool
from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.amap import query_weather_tool


class WeatherAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "weather_agent"

    @property
    def description(self) -> str:
        return (
            "天气查询助手。根据目的地城市查询未来几天的天气预报，"
            "负责填充 raw_weather 字段。"
        )

    @property
    def system_prompt(self) -> str:
        return """你是天气查询助手。你的任务非常简单：

1. 根据用户提供的目的地城市，调用 query_weather_tool 查询天气
2. 确保使用正确的城市名称（如用户输入"帝都"，应转换为"北京"）
3. 查询完成后，输出查询结果摘要

输出格式（必须是有效 JSON）：
{"city": "城市名", "days_forecast": N, "summary": "天气概况描述", "weather_data": [...]}

其中 weather_data 是原始天气数据。

如果城市名看起来有歧义或可能不存在，请在调用工具之前先确认标准城市名称。"""

    @property
    def tools(self) -> list[BaseTool]:
        return [query_weather_tool]

    @property
    def max_steps(self) -> int:
        return 2  # 天气查询通常 1 次工具调用即可

    def _build_context_message(self, state: TripState) -> str:
        """构造天气查询的上下文信息"""
        request = state.get("request")
        if request is None:
            return "错误：State 中缺少 request 字段"

        city = getattr(request, "city", "未知")
        start_date = getattr(request, "start_date", "")
        end_date = getattr(request, "end_date", "")
        travel_days = getattr(request, "travel_days", 0)

        return f"""请查询以下城市的天气预报：

目的地城市：{city}
旅行日期：{start_date} 至 {end_date}（共 {travel_days} 天）

请调用 query_weather_tool 工具查询 {city} 的天气，确保城市名称正确后再调用。"""

    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list,
    ) -> dict:
        """从 LLM 最终输出中提取天气数据，写入 raw_weather"""
        try:
            data = json.loads(llm_content)
            weather_data = data.get("weather_data", [])
        except (json.JSONDecodeError, Exception):
            extracted = self._extract_json(llm_content)
            if extracted:
                try:
                    data = json.loads(extracted)
                    weather_data = data.get("weather_data", [])
                except Exception:
                    weather_data = []
            else:
                weather_data = []

        if not weather_data:
            return {
                "raw_weather": [],
                "agent_outputs": {self.name: "查询完成但未获取到天气数据"},
            }

        summary = f"获取到 {len(weather_data)} 天天气数据"

        return {
            "raw_weather": weather_data,
            "agent_outputs": {self.name: summary},
        }
