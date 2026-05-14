"""
Transport Agent — 负责出发地到目的地的跨城往返交通规划

该 Agent 不使用 ReAct 循环，采用单次工具调用：
1. 读取用户选择的跨城交通方式
2. 调用 estimate_intercity_transport
3. 将结构化结果写入 raw_intercity_transport
"""

from app.graph.state import TripState
from app.tools.intercity_tools import estimate_intercity_transport


class TransportAgent:
    """跨城往返交通规划 Agent"""

    @property
    def name(self) -> str:
        return "transport_agent"

    @property
    def description(self) -> str:
        return "跨城往返交通规划专家。根据用户选择的自驾/高铁/飞机生成出发地到目的地的往返路线和费用估算。"

    def run(self, state: TripState) -> dict:
        """执行跨城往返交通规划"""
        request = state.get("request")
        if request is None:
            return {
                "raw_intercity_transport": {},
                "agent_outputs": {self.name: "失败: 缺少 request"},
                "error": "Transport Agent 缺少 request",
            }

        departure_city = getattr(request, "departure_city", None)
        destination_city = getattr(request, "city", "")
        start_date = getattr(request, "start_date", "")
        end_date = getattr(request, "end_date", "")
        people_count = max(1, int(getattr(request, "people_count", 1) or 1))
        mode = getattr(request, "intercity_transport_mode", "high_speed_rail")

        print(f"\n{'=' * 50}")
        print(f"🚄 {self.name} 开始规划跨城往返交通")
        print(f"  {departure_city or '未填写出发地'} → {destination_city}, mode={mode}")
        print(f"  日期: {start_date} → {end_date}, 人数: {people_count}")
        print(f"{'=' * 50}")

        try:
            plan = estimate_intercity_transport(
                departure_city=departure_city,
                destination_city=destination_city,
                start_date=start_date,
                end_date=end_date,
                people_count=people_count,
                mode=mode,
            )
            summary = plan.get("summary", "跨城交通规划完成")
            print(f"  [transport_agent] {summary}")
            print(f"{'=' * 50}\n")
            return {
                "raw_intercity_transport": plan,
                "agent_outputs": {self.name: summary},
            }
        except Exception as e:
            print(f"  [transport_agent] ⚠️ 工具异常，使用空兜底: {e}")
            fallback = {
                "mode": mode,
                "outbound": None,
                "return_trip": None,
                "total_cost": 0,
                "total_duration_minutes": 0,
                "summary": "跨城交通规划失败，已交由 Planner 根据常识补全。",
                "warnings": [f"Transport Agent 异常: {str(e)}"],
            }
            return {
                "raw_intercity_transport": fallback,
                "agent_outputs": {self.name: fallback["summary"]},
                "error": f"Transport Agent: {str(e)}",
            }
