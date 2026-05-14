"""
Transport Agent — 负责出发地到目的地的跨城往返交通规划

根据用户选择的交通方式，分发到对应的工具模块：
- driving       → driving_tools.plan_driving()
- high_speed_rail → train_tools.plan_train()
- flight        → flight_tools.plan_flight()
"""

from app.agents.base import BaseAgent
from app.graph.state import TripState
from app.tools.driving_tools import plan_driving
from app.tools.train_tools import plan_train
from app.tools.flight_tools import plan_flight
from app.tools.amap import geocode_city
from app.utils.transport_utils import normalize_city


class TransportAgent(BaseAgent):
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

        origin = normalize_city(departure_city)
        destination = normalize_city(destination_city)

        # —— 前置校验 ——
        if not origin:
            empty_plan = {
                "mode": mode,
                "outbound": None,
                "return_trip": None,
                "total_cost": 0,
                "total_duration_minutes": 0,
                "summary": "未填写出发城市，跳过跨城往返交通规划。",
                "warnings": ["未填写出发城市，Planner 可只生成目的地城市内行程。"],
            }
            return {
                "raw_intercity_transport": empty_plan,
                "agent_outputs": {self.name: empty_plan["summary"]},
            }

        if not destination:
            empty_plan = {
                "mode": mode,
                "outbound": None,
                "return_trip": None,
                "total_cost": 0,
                "total_duration_minutes": 0,
                "summary": "未填写目的地城市，无法生成跨城交通规划。",
                "warnings": ["目的地城市为空。"],
            }
            return {
                "raw_intercity_transport": empty_plan,
                "agent_outputs": {self.name: empty_plan["summary"]},
            }

        if origin == destination:
            empty_plan = {
                "mode": mode,
                "outbound": None,
                "return_trip": None,
                "total_cost": 0,
                "total_duration_minutes": 0,
                "summary": "出发城市与目的地城市相同，不需要跨城往返交通。",
                "warnings": ["出发城市与目的地城市相同，跨城交通费用按 0 元处理。"],
            }
            return {
                "raw_intercity_transport": empty_plan,
                "agent_outputs": {self.name: empty_plan["summary"]},
            }

        # —— 地理编码（供自驾使用） ——
        origin_coord = geocode_city(origin)
        dest_coord = geocode_city(destination)

        # —— 按模式分发 ——
        try:
            if mode == "driving":
                plan = plan_driving(
                    departure_city=origin,
                    destination_city=destination,
                    start_date=start_date,
                    end_date=end_date,
                    people_count=people_count,
                    origin_coord=origin_coord,
                    destination_coord=dest_coord,
                )
            elif mode == "flight":
                plan = plan_flight(
                    departure_city=origin,
                    destination_city=destination,
                    start_date=start_date,
                    end_date=end_date,
                    people_count=people_count,
                )
            else:  # high_speed_rail 或未知模式默认高铁
                plan = plan_train(
                    departure_city=origin,
                    destination_city=destination,
                    start_date=start_date,
                    end_date=end_date,
                    people_count=people_count,
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
