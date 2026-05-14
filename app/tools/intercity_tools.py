"""
跨城往返交通规划工具。

规则：
- 自驾：使用高德驾车路径规划。
- 高铁：使用 12306 MCP 查询真实车次/票价/时间。
- 飞机：使用高德公交换乘航班段。
- 第三方 API 不可用时返回估算兜底。
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any, Literal

from langchain_core.tools import tool

from app.tools.amap import AMAP_BASE_URL, amap_get, format_amap_point, geocode_city

AMAP_DRIVING_URL = f"{AMAP_BASE_URL}/direction/driving"

IntercityMode = Literal["driving", "high_speed_rail", "flight"]

MODE_LABELS: dict[str, str] = {
    "driving": "自驾",
    "high_speed_rail": "高铁/动车",
    "flight": "飞机",
}

FUEL_COST_PER_KM_PER_VEHICLE = 0.75
DEFAULT_DRIVING_TOLL_PER_KM = 0.45
DEFAULT_LOCAL_TRANSFER_COST_PER_PERSON = 60


def _debug(message: str) -> None:
    """输出跨城交通调试日志。"""
    print(f"[intercity_tools][debug] {message}")


def _normalize_city(city: str | None) -> str:
    """标准化城市名称。"""
    return (city or "").strip()


def _to_float(value: Any, default: float = 0.0) -> float:
    """安全转换为 float。"""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """安全转换为 int。"""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
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


def _fallback_distance_km(origin_city: str, destination_city: str) -> float:
    """当地理编码失败时返回保守默认距离。"""
    if not origin_city or not destination_city or origin_city == destination_city:
        return 0.0
    return 600.0


def _estimate_distance_km(
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
        distance = _haversine_km(origin, destination) * 1.18
        return round(distance, 1), warnings

    warnings.append("城市坐标解析失败，已使用默认距离估算跨城交通。")
    return _fallback_distance_km(origin_city, destination_city), warnings


def _call_amap_driving_route(
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


def _time_reasonableness_score(
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
            score += 70   # 上午出发良好
        elif 11 <= dep_hour <= 12:
            score += 30   # 午前可接受
        elif 13 <= dep_hour <= 15:
            score -= 30   # 下午出发浪费半天
        elif 16 <= dep_hour <= 19:
            score -= 80   # 傍晚出发浪费一天
        elif 20 <= dep_hour <= 22:
            score -= 150  # 夜间到达太晚
        else:
            score -= 250  # 凌晨出发不合理

        # 去程到达时间
        if arr_hour <= 5:
            score -= 200  # 凌晨到达
        elif 6 <= arr_hour <= 10:
            score += 30   # 上午到达可开始游玩
        elif 11 <= arr_hour <= 14:
            score += 20   # 中午到达
        elif 15 <= arr_hour <= 18:
            score -= 10   # 下午到达略晚
        elif 19 <= arr_hour <= 22:
            score -= 60   # 晚间到达不便
        else:
            score -= 150  # 深夜到达
    else:
        # 返程：偏好下午/晚上出发，最大化最后一天游玩时间
        if 6 <= dep_hour <= 9:
            score -= 80   # 早上走浪费最后一天
        elif 10 <= dep_hour <= 12:
            score += 10   # 上午走尚可
        elif 13 <= dep_hour <= 15:
            score += 60   # 午后出发良好
        elif 16 <= dep_hour <= 18:
            score += 100  # 下午出发最好，玩到最后一刻
        elif 19 <= dep_hour <= 20:
            score += 80   # 傍晚出发良好
        elif 21 <= dep_hour <= 22:
            score += 20   # 夜间出发可接受
        else:
            score -= 100  # 凌晨出发不合理

        # 返程到达时间
        if arr_hour <= 5:
            score -= 200  # 凌晨到家
        elif 6 <= arr_hour <= 8:
            score += 10
        elif 9 <= arr_hour <= 20:
            score += 30   # 白天到家舒适
        elif 21 <= arr_hour <= 23:
            score += 20   # 夜间到家尚可
        else:
            score -= 80   # 深夜到家

    return score

def _parse_time_minutes(departure_time: str | None, arrival_time: str | None) -> int:
    """根据时间字符串估算分钟差，失败时返回 0。"""
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


def _normalize_driving_segment(
    path: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
) -> dict[str, Any]:
    """将高德驾车路线归一化为单程跨城交通结构。"""
    vehicles = max(1, math.ceil(max(1, people_count) / 4))
    distance_km = round(_to_float(path.get("distance")) / 1000, 1)
    duration_minutes = max(1, math.ceil(_to_float(path.get("duration")) / 60))
    tolls = _to_float(path.get("tolls"))
    if tolls <= 0:
        tolls = distance_km * DEFAULT_DRIVING_TOLL_PER_KM
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

def _fallback_one_way_ticket_cost(mode: str, distance_km: float) -> int:
    """估算单人单程公共交通费用。"""
    if mode == "flight":
        return max(450, int(distance_km * 0.75)) + DEFAULT_LOCAL_TRANSFER_COST_PER_PERSON
    return max(80, int(distance_km * 0.38))


def _fallback_one_way_duration(mode: str, distance_km: float) -> int:
    """估算单程耗时。"""
    if mode == "driving":
        return int(max(60, distance_km / 85 * 60))
    if mode == "flight":
        return int(150 + max(60, distance_km / 750 * 60))
    return int(45 + max(45, distance_km / 240 * 60))


def _create_estimated_fallback(
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
    distance_km, distance_warnings = _estimate_distance_km(
        departure_city,
        destination_city,
        origin_coord=origin_coord,
        destination_coord=destination_coord,
    )
    merged_warnings.extend(distance_warnings)

    vehicles = max(1, math.ceil(people / 4))
    one_way_duration = _fallback_one_way_duration(mode, distance_km)
    mode_label = MODE_LABELS.get(mode, mode)

    if mode == "driving":
        one_way_cost = int(round(distance_km * (FUEL_COST_PER_KM_PER_VEHICLE * vehicles + DEFAULT_DRIVING_TOLL_PER_KM)))
    else:
        one_way_cost = int(_fallback_one_way_ticket_cost(mode, distance_km) * people)

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


def _build_plan_from_segments(
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


def _try_driving_plan(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    origin_coord: tuple[float, float],
    destination_coord: tuple[float, float],
) -> dict[str, Any]:
    """尝试生成真实高德自驾往返方案。"""
    outbound_path = _call_amap_driving_route(origin_coord, destination_coord)
    return_path = _call_amap_driving_route(destination_coord, origin_coord)
    if not outbound_path or not return_path:
        raise RuntimeError("高德未返回可用驾车路线。")

    outbound = _normalize_driving_segment(
        outbound_path,
        departure_city,
        destination_city,
        start_date,
        "outbound",
        people_count,
    )
    return_trip = _normalize_driving_segment(
        return_path,
        destination_city,
        departure_city,
        end_date,
        "return",
        people_count,
    )
    return _build_plan_from_segments("driving", outbound, return_trip, [])


def _is_high_speed_train(ticket: dict[str, Any]) -> bool:
    """判断是否为高铁/动车/城际（G/D/C 字头）。"""
    train_no = str(ticket.get("train_no", "") or "")
    return bool(re.match(r"^[GDC]", train_no, re.IGNORECASE))


def _train_duration_minutes(ticket: dict[str, Any]) -> int:
    """解析 12306 车票持续时间（格式可能是 HH:MM 或分钟数）。"""
    duration = ticket.get("duration", "")
    if isinstance(duration, (int, float)):
        return int(duration)
    if isinstance(duration, str) and ":" in duration:
        parts = duration.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    return _to_int(duration, 0)


def _train_second_class_price(source: dict[str, Any]) -> float:
    """提取二等座票价，支持嵌套 prices 结构，无则返回 0。"""
    # 先尝试嵌套 prices 字段
    prices = source.get("prices", {})
    if isinstance(prices, dict):
        for key, val in prices.items():
            if "二等" in key or "second" in key.lower() or "edz" in key.lower():
                p = _to_float(val)
                if p > 0:
                    return p
        # prices 里随便取第一个 > 0 的值作为最低价
        for val in prices.values():
            p = _to_float(val)
            if p > 0:
                return p
    # 顶层直接查找
    for key in ("second_class_price", "price_2nd", "二等座", "second_price",
                "edz_price", "EDZ", "second_seat_price", "ZE", "ze_price"):
        val = source.get(key)
        if val is not None and _to_float(val) > 0:
            return _to_float(val)
    # 遍历顶层字段
    for key, val in source.items():
        if isinstance(val, (int, float)) and val > 0:
            key_lower = key.lower()
            if "二等" in key or "second" in key_lower or "edz" in key_lower:
                return _to_float(val)
    # 兜底
    for key in ("price", "ticket_price", "total_price"):
        val = source.get(key)
        if val is not None and _to_float(val) > 0:
            return _to_float(val)
    return 0.0


def _normalize_train_result(
    ticket: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
    price_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将 12306 车票结果归一化为单程跨城交通结构。"""
    train_no = str(ticket.get("train_no", "") or ticket.get("name", "") or "").strip()
    departure_station = str(ticket.get("from_station_name", "") or ticket.get("start_station", "") or origin_city)
    arrival_station = str(ticket.get("to_station_name", "") or ticket.get("end_station", "") or destination_city)
    departure_time = str(ticket.get("start_time", "") or "")
    arrival_time = str(ticket.get("arrive_time", "") or "")
    duration_minutes = _train_duration_minutes(ticket)
    # 优先从 price_info 取票价，其次从 ticket 本身取
    price_source = price_info if price_info else ticket
    price_per_person = _train_second_class_price(price_source)
    estimated_cost = int(round(price_per_person * max(1, people_count))) if price_per_person > 0 else 0

    route_summary = (
        f"{train_no or '高铁/动车'}: {departure_station}"
        f"{' ' + departure_time if departure_time else ''} → {arrival_station}"
        f"{' ' + arrival_time if arrival_time else ''}"
    )

    return {
        "direction": direction,
        "origin": origin_city,
        "destination": destination_city,
        "date": date,
        "mode": "high_speed_rail",
        "duration_minutes": duration_minutes,
        "distance_km": 0,
        "estimated_cost": estimated_cost,
        "route_summary": route_summary,
        "notes": ["数据来源：12306 官方实时车次。"],
        "service_no": train_no or None,
        "carrier": "铁路",
        "departure_place": departure_station,
        "arrival_place": arrival_station,
        "departure_time": departure_time or None,
        "arrival_time": arrival_time or None,
        "price_per_person": round(price_per_person, 2) if price_per_person > 0 else None,
        "data_source": "12306",
        "is_estimated": False,
    }


def _select_train(
    tickets: list[dict[str, Any]],
    direction: Literal["outbound", "return"],
) -> dict[str, Any] | None:
    """从 12306 车票列表中选出最合适的高铁/动车（方向感知时间评分）。"""
    high_speed = [t for t in tickets if _is_high_speed_train(t)]
    print(f"[intercity_tools][TRACE] _select_train direction={direction}, "
          f"总车次={len(tickets)}, 高铁/动车={len(high_speed)}")
    if tickets:
        print(f"[intercity_tools][TRACE]   第一条车次全部字段: {json.dumps(tickets[0], ensure_ascii=False)}")

    if not high_speed:
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for ticket in high_speed:
        dep_time = str(ticket.get("start_time", "") or "")
        arr_time = str(ticket.get("arrive_time", "") or "")
        time_score = _time_reasonableness_score(dep_time, arr_time, direction)
        duration = _train_duration_minutes(ticket)
        train_no = ticket.get("train_no", "")
        # 类型加分: G=30, D=20, C=10
        prefix = train_no[0].upper() if train_no else ""
        type_bonus = 30 if prefix == "G" else (20 if prefix == "D" else (10 if prefix == "C" else 0))
        total = time_score + type_bonus
        scored.append((total, ticket))
        print(f"[intercity_tools][TRACE]   车次: {train_no}, dep={dep_time}, arr={arr_time}, "
              f"duration={duration}min, type_bonus={type_bonus}, time_score={time_score}, "
              f"total={total}, price={_train_second_class_price(ticket)}")

    scored.sort(key=lambda item: (-item[0], _train_duration_minutes(item[1])))
    best_score, best_ticket = scored[0]
    print(f"[intercity_tools][TRACE] 最高分车次: {best_ticket.get('train_no')}, "
          f"score={best_score}, {'✅ 选中' if best_score >= -50 else '❌ 分数过低'}")

    if best_score < -200:
        return None
    return best_ticket


def _try_railway_plan(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    origin_coord: tuple[float, float],
    destination_coord: tuple[float, float],
) -> dict[str, Any]:
    """使用 12306 MCP 查询真实高铁/动车往返方案。"""
    from app.tools.train_tools import query_tickets, resolve_station_telecode

    print(f"[intercity_tools][TRACE] _try_railway_plan(12306) 开始: {departure_city} → {destination_city}")

    origin_name = _normalize_city(departure_city)
    dest_name = _normalize_city(destination_city)

    from_code = resolve_station_telecode(origin_name)
    to_code = resolve_station_telecode(dest_name)
    print(f"[intercity_tools][TRACE] 车站电报码: {origin_name}→{from_code}, {dest_name}→{to_code}")

    if not from_code or not to_code:
        raise RuntimeError(f"12306 车站解析失败: {departure_city}={from_code}, {destination_city}={to_code}")

    outbound_tickets = query_tickets(from_code, to_code, start_date)
    return_tickets = query_tickets(to_code, from_code, end_date)

    outbound_train = _select_train(outbound_tickets, "outbound")
    return_train = _select_train(return_tickets, "return")
    print(f"[intercity_tools][TRACE] 去程选中: {outbound_train is not None}, 返程选中: {return_train is not None}")

    if not outbound_train or not return_train:
        raise RuntimeError("12306 未返回可用高铁/动车车次。")

    # 分别查询票价（query-tickets 不带价格，需单独调 query-ticket-price）
    from app.tools.train_tools import query_ticket_price
    outbound_code = str(outbound_train.get("train_no", "") or "")
    return_code = str(return_train.get("train_no", "") or "")
    outbound_price = query_ticket_price(from_code, to_code, start_date, outbound_code) if outbound_code else {}
    return_price = query_ticket_price(to_code, from_code, end_date, return_code) if return_code else {}
    print(f"[intercity_tools][TRACE] 票价: 去程={outbound_price}, 返程={return_price}")

    outbound = _normalize_train_result(
        outbound_train, origin_name, dest_name, start_date, "outbound", people_count, outbound_price,
    )
    return_trip = _normalize_train_result(
        return_train, dest_name, origin_name, end_date, "return", people_count, return_price,
    )
    return _build_plan_from_segments("high_speed_rail", outbound, return_trip, [])


def _select_best_google_flight(
    flights: list[dict[str, Any]],
    direction: Literal["outbound", "return"],
) -> dict[str, Any] | None:
    """从 Google Flights 结果中选最优航班（方向感知时间评分 + 价格）。"""
    scored: list[tuple[int, dict[str, Any]]] = []
    for f in flights:
        if not f.get("flights"):
            continue
        print(f"[intercity_tools][TRACE]   Google Flights 航班: {json.dumps(f, ensure_ascii=False)}")
        first_seg = f["flights"][0]
        last_seg = f["flights"][-1]
        dep_time = first_seg.get("departure_time", "")
        arr_time = last_seg.get("arrival_time", "")
        time_score = _time_reasonableness_score(dep_time, arr_time, direction)
        # 直飞加分
        direct_bonus = 100 if len(f["flights"]) == 1 else 0
        # 价格：有真实价格加 200 分并轻微偏好低价；无价格重罚 500 分
        price = f.get("price", 0) or 0
        if price > 0:
            price_score = 200 - int(price / 100)  # 有价格 = 基础 200 分 + 低价偏好
        else:
            price_score = -500  # 无价格 = 严重惩罚，排到最后
        total = time_score + direct_bonus + price_score
        scored.append((total, f))

    if not scored:
        return None

    scored.sort(key=lambda item: -item[0])
    best_score, best = scored[0]
    _debug(f"Google Flights 筛选: best_score={best_score}, "
           f"price={best.get('price')}, segments={len(best.get('flights', []))}")
    return best


def _normalize_google_flight(
    flight: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
) -> dict[str, Any]:
    """将 Google Flights 结果归一化为单程跨城交通结构。"""
    segments = flight.get("flights", [])
    airlines = flight.get("airlines", [])
    airline_str = "/".join(airlines) if airlines else "航班"
    price_per_person = flight.get("price_per_person", 0) or 0
    people = max(1, int(people_count))

    if segments:
        first = segments[0]
        last = segments[-1]
        departure_place = first["from_airport"]["name"]
        arrival_place = last["to_airport"]["name"]
        departure_time = first.get("departure_time", "")
        arrival_time = last.get("arrival_time", "")
        duration_minutes = sum(s.get("duration_minutes", 0) for s in segments)
        service_no = f"{airline_str} {first.get('from_airport', {}).get('code', '')}→{last.get('to_airport', {}).get('code', '')}"
    else:
        departure_place = origin_city
        arrival_place = destination_city
        departure_time = ""
        arrival_time = ""
        duration_minutes = 0
        service_no = airline_str

    estimated_cost = int(round(price_per_person * people)) if price_per_person > 0 else 0

    route_summary = (
        f"{service_no}: {departure_place}"
        f"{' ' + departure_time if departure_time else ''} → {arrival_place}"
        f"{' ' + arrival_time if arrival_time else ''}"
    )

    notes = ["数据来源：Google Flights。"]
    if len(segments) > 1:
        notes.append(f"经停 {len(segments) - 1} 站")

    return {
        "direction": direction,
        "origin": origin_city,
        "destination": destination_city,
        "date": date,
        "mode": "flight",
        "duration_minutes": duration_minutes,
        "distance_km": 0,
        "estimated_cost": estimated_cost,
        "route_summary": route_summary,
        "notes": notes,
        "service_no": service_no,
        "carrier": airline_str,
        "departure_place": departure_place,
        "arrival_place": arrival_place,
        "departure_time": departure_time or None,
        "arrival_time": arrival_time or None,
        "price_per_person": round(price_per_person, 2) if price_per_person > 0 else None,
        "data_source": "google_flights",
        "is_estimated": False,
    }


def _try_flight_plan(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    origin_coord: tuple[float, float],
    destination_coord: tuple[float, float],
) -> dict[str, Any]:
    """使用 Google Flights 查询真实航班往返方案。"""
    from app.tools.flight_tools import resolve_airport_code, query_flights

    _debug(f"Google Flights 查询: {departure_city} → {destination_city}")

    origin_code = resolve_airport_code(departure_city)
    dest_code = resolve_airport_code(destination_city)
    _debug(f"机场代码: {departure_city}→{origin_code}, {destination_city}→{dest_code}")

    if not origin_code or not dest_code:
        raise RuntimeError(f"机场代码解析失败: {departure_city}={origin_code}, {destination_city}={dest_code}")

    outbound_flights = query_flights(origin_code, dest_code, start_date, people_count=people_count)
    return_flights = query_flights(dest_code, origin_code, end_date, people_count=people_count)

    outbound_best = _select_best_google_flight(outbound_flights, "outbound")
    return_best = _select_best_google_flight(return_flights, "return")
    _debug(f"去程选中: {outbound_best is not None}, 返程选中: {return_best is not None}")

    if not outbound_best or not return_best:
        raise RuntimeError("Google Flights 未返回可用航班。")

    outbound = _normalize_google_flight(
        outbound_best, departure_city, destination_city, start_date, "outbound", people_count,
    )
    return_trip = _normalize_google_flight(
        return_best, destination_city, departure_city, end_date, "return", people_count,
    )
    return _build_plan_from_segments("flight", outbound, return_trip, [])


def estimate_intercity_transport(
    departure_city: str | None,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    mode: IntercityMode,
) -> dict[str, Any]:
    """
    生成出发地到目的地的跨城往返交通方案。

    高铁和航班只展示主交通段；不展示地铁/公交接驳。
    """
    origin = _normalize_city(departure_city)
    destination = _normalize_city(destination_city)
    people = max(1, int(people_count or 1))
    selected_mode: IntercityMode = mode if mode in ("driving", "high_speed_rail", "flight") else "high_speed_rail"
    _debug(
        f"跨城交通入口: departure_city={origin or None}, destination_city={destination or None}, "
        f"start_date={start_date}, end_date={end_date}, people_count={people}, mode={selected_mode}"
    )

    if not origin:
        return {
            "mode": selected_mode,
            "outbound": None,
            "return_trip": None,
            "total_cost": 0,
            "total_duration_minutes": 0,
            "summary": "未填写出发城市，跳过跨城往返交通规划。",
            "warnings": ["未填写出发城市，Planner 可只生成目的地城市内行程。"],
        }

    if not destination:
        return {
            "mode": selected_mode,
            "outbound": None,
            "return_trip": None,
            "total_cost": 0,
            "total_duration_minutes": 0,
            "summary": "未填写目的地城市，无法生成跨城交通规划。",
            "warnings": ["目的地城市为空。"],
        }

    if origin == destination:
        return {
            "mode": selected_mode,
            "outbound": None,
            "return_trip": None,
            "total_cost": 0,
            "total_duration_minutes": 0,
            "summary": "出发城市与目的地城市相同，不需要跨城往返交通。",
            "warnings": ["出发城市与目的地城市相同，跨城交通费用按 0 元处理。"],
        }

    warnings: list[str] = []
    origin_coord = geocode_city(origin)
    destination_coord = geocode_city(destination)
    print(f"[intercity_tools][TRACE] 地理编码: origin={origin} → {origin_coord}, destination={destination} → {destination_coord}")
    if selected_mode in ("driving", "high_speed_rail", "flight") and (not origin_coord or not destination_coord):
        print(f"[intercity_tools][TRACE] ❌ 地理编码失败，走估算兜底")
        warnings.append("高德城市地理编码失败，已使用估算生成跨城交通。")
        return _create_estimated_fallback(
            departure_city=origin,
            destination_city=destination,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode=selected_mode,
            warnings=warnings,
            origin_coord=origin_coord,
            destination_coord=destination_coord,
        )

    try:
        if selected_mode == "driving":
            return _try_driving_plan(origin, destination, start_date, end_date, people, origin_coord, destination_coord)  # type: ignore[arg-type]
        if selected_mode == "high_speed_rail":
            return _try_railway_plan(origin, destination, start_date, end_date, people, origin_coord, destination_coord)  # type: ignore[arg-type]
        return _try_flight_plan(origin, destination, start_date, end_date, people, origin_coord, destination_coord)  # type: ignore[arg-type]
    except Exception as exc:
        import traceback
        print(f"[intercity_tools][TRACE] ❌ 异常: {exc}")
        traceback.print_exc()
        if selected_mode == "flight":
            warnings.append(f"高德未返回可用航班段，已使用估算。原因：{exc}")
        elif selected_mode == "high_speed_rail":
            warnings.append(f"12306 未返回可用高铁/动车车次，已使用估算。原因：{exc}")
        else:
            warnings.append(f"高德驾车路径规划失败，已使用估算。原因：{exc}")

        return (
            _create_estimated_fallback(
            departure_city=origin,
            destination_city=destination,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode=selected_mode,
            warnings=warnings,
            origin_coord=origin_coord,
            destination_coord=destination_coord,
        ))
