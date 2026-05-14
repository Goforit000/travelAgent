"""
Weather Agent — 直接调用高德 API 查询天气预报，无需 LLM 参与

职责：
1. 从 State 读取 city + travel_days
2. 直接调用 get_weather(city)（底层 HTTP 函数）
3. 按旅行天数截取结果
4. 写入 raw_weather

相比 ReAct 模式：减少 2 次 LLM 调用，耗时从 5-10s 降到 <1s。
纠错能力：高德 API 对模糊城市名（如"帝都"→"北京"）有内部纠错，
返回空列表时 Supervisor 可降级跳过。
"""

from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.amap import get_weather


class WeatherAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "weather_agent"

    @property
    def description(self) -> str:
        return (
            "天气查询助手。直接调用高德 API 获取目的地天气预报，按旅行天数截取。"
            "负责填充 raw_weather 字段。"
        )

    def run(self, state: TripState) -> dict:
        """
        直接调用 get_weather()，跳过 LLM ReAct 循环

        流程：
        1. 从 request 提取 city, travel_days
        2. 调用 get_weather(city)（底层 HTTP，不抛异常）
        3. 截取 [:travel_days]
        4. 返回结果
        """
        request = state.get("request")
        if request is None:
            return {
                "raw_weather": [],
                "error": "State 中缺少 request 字段",
                "agent_outputs": {self.name: "失败: 缺少 request"},
            }

        city = getattr(request, "city", "")
        travel_days = getattr(request, "travel_days", 0)

        if not city:
            return {
                "raw_weather": [],
                "agent_outputs": {self.name: "失败: city 为空"},
            }

        print(f"\n{'=' * 50}")
        print(f"🌤️  {self.name} 直接查询天气: city={city}, days={travel_days}")
        print(f"{'=' * 50}")

        # 直接调用底层 HTTP 函数（不走 @tool / LLM）
        all_weather = get_weather(city)

        if not all_weather:
            print(f"  [{self.name}] ⚠️ 未获取到天气数据")
            return {
                "raw_weather": [],
                "agent_outputs": {self.name: f"未获取到 {city} 的天气数据"},
            }

        # 按旅行天数截取
        weather = all_weather[:travel_days] if travel_days > 0 else all_weather
        print(f"  [{self.name}] ✅ 获取 {len(weather)} 天天气")

        return {
            "raw_weather": weather,
            "agent_outputs": {self.name: f"获取 {city} {len(weather)} 天天气"},
        }
