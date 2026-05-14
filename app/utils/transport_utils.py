"""
跨城交通公共工具函数 — 供 driving_tools / train_tools / flight_tools 共享。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from app.tools.amap import geocode_city

IntercityMode = Literal["driving", "high_speed_rail", "flight"]

MODE_LABELS: dict[str, str] = {
    "driving": "自驾",
    "high_speed_rail": "高铁/动车",
    "flight": "飞机",
}

FUEL_COST_PER_KM_PER_VEHICLE = 0.75
DEFAULT_DRIVING_TOLL_PER_KM = 0.45
DEFAULT_LOCAL_TRANSFER_COST_PER_PERSON = 60


def normalize_city(city: str | None) -> str:
    """标准化城市名称。"""
    return (city or "").strip()


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    """根据经纬度计算球面距离，单位为公里。"""
    lng1, lat1 = origin
    lng2, lat2 = destination
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fallback_distance_km(origin_city: str, destination_city: str) -> float:
    """当地理编码失败时返回保守默认距离。"""
    if not origin_city or not destination_city or origin_city == destination_city:
        return 0.0
    return 600.0


def estimate_distance_km(
    origin_city: str,
    destination_city: str,
    origin_coord: tuple[float, float] | None = None,
    destination_coord: tuple[float, float] | None = None,
) -> tuple[float, list[str]]:
    """优先用坐标估算距离，失败时返回默认距离。"""
    warnings: list[str] = []
    origin = origin_coord or geocode_city(origin_city)
    destination = destination_coord or geocode_city(destination_city)

    if origin and destination:
        distance = haversine_km(origin, destination) * 1.18
        return round(distance, 1), warnings

    warnings.append("城市坐标解析失败，已使用默认距离估算跨城交通。")
    return fallback_distance_km(origin_city, destination_city), warnings


def time_reasonableness_score(
    departure_time: str | None,
    arrival_time: str | None,
    direction: Literal["outbound", "return"],
) -> int:
    """方向感知的时间合理性评分。去程偏上午出发，返程偏下午出发，到达均不宜太晚。"""
    if not departure_time or not arrival_time:
        return 0
    try:
        dep_hour = int(str(departure_time)[:2])
        arr_hour = int(str(arrival_time)[:2])
    except (ValueError, TypeError):
        return 0

    score = 0

    if direction == "outbound":
        # 去程：偏好上午/早上出发，到达不宜太晚
        if 6 <= dep_hour <= 8:
            score += 100  # 早上出发最佳
        elif 9 <= dep_hour <= 10:
            score += 70  # 上午出发良好
        elif 11 <= dep_hour <= 12:
            score += 30  # 午前可接受
        elif 13 <= dep_hour <= 15:
            score -= 30  # 下午出发浪费半天
        elif 16 <= dep_hour <= 19:
            score -= 80  # 傍晚出发浪费一天
        elif 20 <= dep_hour <= 22:
            score -= 150  # 夜间到达太晚
        else:
            score -= 250  # 凌晨出发不合理

        # 去程到达时间
        if arr_hour <= 5:
            score -= 200  # 凌晨到达
        elif 6 <= arr_hour <= 10:
            score += 30  # 上午到达可开始游玩
        elif 11 <= arr_hour <= 14:
            score += 20  # 中午到达
        elif 15 <= arr_hour <= 18:
            score -= 10  # 下午到达略晚
        elif 19 <= arr_hour <= 22:
            score -= 60  # 晚间到达不便
        else:
            score -= 150  # 深夜到达
    else:
        # 返程：偏好下午/晚上出发，最大化最后一天游玩时间
        if 6 <= dep_hour <= 9:
            score -= 80  # 早上走浪费最后一天
        elif 10 <= dep_hour <= 12:
            score += 10  # 上午走尚可
        elif 13 <= dep_hour <= 15:
            score += 60  # 午后出发良好
        elif 16 <= dep_hour <= 18:
            score += 100  # 下午出发最好，玩到最后一刻
        elif 19 <= dep_hour <= 20:
            score += 80  # 傍晚出发良好
        elif 21 <= dep_hour <= 22:
            score += 20  # 夜间出发可接受
        else:
            score -= 100  # 凌晨出发不合理

        # 返程到达时间
        if arr_hour <= 5:
            score -= 200  # 凌晨到家
        elif 6 <= arr_hour <= 8:
            score += 10
        elif 9 <= arr_hour <= 20:
            score += 30  # 白天到家舒适
        elif 21 <= arr_hour <= 23:
            score += 20  # 夜间到家尚可
        else:
            score -= 80  # 深夜到家

    return score


def parse_time_minutes(departure_time: str | None, arrival_time: str | None) -> int:
    if not departure_time or not arrival_time:
        return 0
    try:
        if "T" in departure_time and "T" in arrival_time:
            start = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(arrival_time.replace("Z", "+00:00"))
            return max(0, int((end - start).total_seconds() // 60))
        start = datetime.strptime(departure_time[:5], "%H:%M")
        end = datetime.strptime(arrival_time[:5], "%H:%M")
        minutes = int((end - start).total_seconds() // 60)
        return minutes if minutes >= 0 else minutes + 24 * 60
    except Exception:
        return 0


def build_plan_from_segments(
    mode: IntercityMode,
    outbound: dict[str, Any],
    return_trip: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """根据去程和返程片段生成往返交通结构。"""
    total_cost = int(outbound.get("estimated_cost", 0) or 0) + int(return_trip.get("estimated_cost", 0) or 0)
    total_duration = int(outbound.get("duration_minutes", 0) or 0) + int(return_trip.get("duration_minutes", 0) or 0)
    mode_label = MODE_LABELS.get(mode, mode)
    summary = (
        f"{mode_label}往返方案：{outbound.get('route_summary', '')}；"
        f"{return_trip.get('route_summary', '')}。团队总费用约 {total_cost} 元。"
    )
    return {
        "mode": mode,
        "outbound": outbound,
        "return_trip": return_trip,
        "total_cost": total_cost,
        "total_duration_minutes": total_duration,
        "summary": summary,
        "warnings": warnings,
    }


def fallback_one_way_ticket_cost(mode: str, distance_km: float) -> int:
    """估算单人单程公共交通费用。"""
    if mode == "flight":
        return max(450, int(distance_km * 0.75)) + DEFAULT_LOCAL_TRANSFER_COST_PER_PERSON
    return max(80, int(distance_km * 0.38))


def fallback_one_way_duration(mode: str, distance_km: float) -> int:
    """估算单程耗时。"""
    if mode == "driving":
        return int(max(60, distance_km / 85 * 60))
    if mode == "flight":
        return int(150 + max(60, distance_km / 750 * 60))
    return int(45 + max(45, distance_km / 240 * 60))


def create_estimated_fallback(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    mode: IntercityMode,
    warnings: list[str] | None = None,
    origin_coord: tuple[float, float] | None = None,
    destination_coord: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """生成估算兜底方案，不伪造真实班次和真实时间。"""
    people = max(1, int(people_count or 1))
    merged_warnings = list(warnings or [])
    distance_km, distance_warnings = estimate_distance_km(
        departure_city,
        destination_city,
        origin_coord=origin_coord,
        destination_coord=destination_coord,
    )
    merged_warnings.extend(distance_warnings)

    vehicles = max(1, math.ceil(people / 4))
    one_way_duration = fallback_one_way_duration(mode, distance_km)
    mode_label = MODE_LABELS.get(mode, mode)

    if mode == "driving":
        one_way_cost = int(round(distance_km * (FUEL_COST_PER_KM_PER_VEHICLE * vehicles + DEFAULT_DRIVING_TOLL_PER_KM)))
    else:
        one_way_cost = int(fallback_one_way_ticket_cost(mode, distance_km) * people)

    def segment(direction: Literal["outbound", "return"], origin: str, destination: str, date: str) -> dict[str, Any]:
        return {
            "direction": direction,
            "origin": origin,
            "destination": destination,
            "date": date,
            "mode": mode,
            "duration_minutes": one_way_duration,
            "distance_km": round(distance_km, 1),
            "estimated_cost": one_way_cost,
            "route_summary": f"{origin} → {destination} {mode_label}估算方案",
            "notes": ["估算数据：未获取到真实班次、真实时间或真实价格。"],
            "service_no": None,
            "carrier": None,
            "departure_place": None,
            "arrival_place": None,
            "departure_time": None,
            "arrival_time": None,
            "price_per_person": None,
            "data_source": "estimated",
            "is_estimated": True,
        }

    outbound = segment("outbound", departure_city, destination_city, start_date)
    return_trip = segment("return", destination_city, departure_city, end_date)
    total_cost = one_way_cost * 2
    total_duration = one_way_duration * 2

    return {
        "mode": mode,
        "outbound": outbound,
        "return_trip": return_trip,
        "total_cost": total_cost,
        "total_duration_minutes": total_duration,
        "summary": f"跨城交通选择 {mode_label}，未获取到完整真实班次，已使用估算兜底。",
        "warnings": merged_warnings,
    }
