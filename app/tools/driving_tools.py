"""
自驾跨城交通规划 — 基于高德驾车路径规划。

提供：
    plan_driving(departure_city, destination_city, start_date, end_date, people_count, ...) → dict
"""

from __future__ import annotations

import math
from typing import Any, Literal

from app.tools.amap import AMAP_BASE_URL, amap_get, format_amap_point, geocode_city
from app.utils.transport_utils import (
    build_plan_from_segments,
    create_estimated_fallback,
    normalize_city,
    to_float,
)

AMAP_DRIVING_URL = f"{AMAP_BASE_URL}/direction/driving"


def call_amap_driving_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> dict[str, Any] | None:
    """调用高德驾车路径规划接口。"""
    data = amap_get(
        AMAP_DRIVING_URL,
        params={
            "origin": format_amap_point(origin),
            "destination": format_amap_point(destination),
            "extensions": "all",
            "strategy": "0",
        },
        timeout=12,
    )
    paths = data.get("route", {}).get("paths", [])
    return paths[0] if paths else None


def normalize_driving_segment(
    path: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
) -> dict[str, Any]:
    """将高德驾车路线归一化为单程跨城交通结构。"""
    vehicles = max(1, math.ceil(max(1, people_count) / 4))
    distance_km = round(to_float(path.get("distance")) / 1000, 1)
    duration_minutes = max(1, math.ceil(to_float(path.get("duration")) / 60))
    tolls = to_float(path.get("tolls"))
    if tolls <= 0:
        from app.utils.transport_utils import DEFAULT_DRIVING_TOLL_PER_KM
        tolls = distance_km * DEFAULT_DRIVING_TOLL_PER_KM
    from app.utils.transport_utils import FUEL_COST_PER_KM_PER_VEHICLE
    fuel_cost = distance_km * FUEL_COST_PER_KM_PER_VEHICLE * vehicles
    estimated_cost = int(round(tolls + fuel_cost))

    return {
        "direction": direction,
        "origin": origin_city,
        "destination": destination_city,
        "date": date,
        "mode": "driving",
        "duration_minutes": duration_minutes,
        "distance_km": distance_km,
        "estimated_cost": estimated_cost,
        "route_summary": f"{origin_city} 自驾前往 {destination_city}",
        "notes": [f"数据来源：高德驾车路径规划；按 {vehicles} 辆车估算。"],
        "service_no": None,
        "carrier": None,
        "departure_place": origin_city,
        "arrival_place": destination_city,
        "departure_time": None,
        "arrival_time": None,
        "price_per_person": None,
        "data_source": "amap_driving",
        "is_estimated": False,
    }


def plan_driving(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    origin_coord: tuple[float, float] | None = None,
    destination_coord: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """
    生成自驾往返方案。

    参数：
        departure_city: 出发城市
        destination_city: 目的城市
        start_date: 出发日期 YYYY-MM-DD
        end_date: 返程日期 YYYY-MM-DD
        people_count: 人数
        origin_coord: 出发城市坐标（可选，不提供则自动地理编码）
        destination_coord: 目的城市坐标（可选）

    返回：统一跨城交通方案 dict。
    """
    origin = normalize_city(departure_city)
    destination = normalize_city(destination_city)
    people = max(1, int(people_count or 1))

    origin_coord = origin_coord or geocode_city(origin)
    destination_coord = destination_coord or geocode_city(destination)

    if not origin_coord or not destination_coord:
        return create_estimated_fallback(
            departure_city=origin,
            destination_city=destination,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode="driving",
            warnings=["高德城市地理编码失败，已使用估算生成跨城交通。"],
            origin_coord=origin_coord,
            destination_coord=destination_coord,
        )

    try:
        outbound_path = call_amap_driving_route(origin_coord, destination_coord)
        return_path = call_amap_driving_route(destination_coord, origin_coord)
        if not outbound_path or not return_path:
            raise RuntimeError("高德未返回可用驾车路线。")

        outbound = normalize_driving_segment(
            outbound_path, origin, destination, start_date, "outbound", people,
        )
        return_trip = normalize_driving_segment(
            return_path, destination, origin, end_date, "return", people,
        )
        return build_plan_from_segments("driving", outbound, return_trip, [])
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return create_estimated_fallback(
            departure_city=origin,
            destination_city=destination,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode="driving",
            warnings=[f"高德驾车路径规划失败，已使用估算。原因：{exc}"],
            origin_coord=origin_coord,
            destination_coord=destination_coord,
        )
