"""
跨城往返交通规划工具。

规则：
- 自驾：使用高德驾车路径规划。
- 高铁：使用高德公交换乘结果中的 railway 主交通段，不展示地铁/公交接驳。
- 飞机：使用高德公交换乘航班段，不再依赖 Amadeus。
- 第三方 API 不可用时返回估算兜底，但不伪造车次、航班号和真实时间。
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
AMAP_TRANSIT_URL = f"{AMAP_BASE_URL}/direction/transit/integrated"

IntercityMode = Literal["driving", "high_speed_rail", "flight"]

MODE_LABELS: dict[str, str] = {
    "driving": "自驾",
    "high_speed_rail": "高铁/动车",
    "flight": "飞机",
}

HIGH_SPEED_RAIL_TYPE_CODES = {"2011", "2012", "2013"}
HIGH_SPEED_RAIL_KEYWORDS = ("高铁", "动车", "城际", "G", "D", "C")

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


def _first_item(value: Any) -> Any:
    """兼容高德字段可能返回列表或单个对象的情况。"""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _ensure_list(value: Any) -> list[Any]:
    """把对象、列表或空值统一整理为列表。"""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_text(value: Any) -> str:
    """递归提取嵌套对象中的文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_extract_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_extract_text(item) for item in value.values())
    return str(value)


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


def _call_amap_transit_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    origin_city: str,
    destination_city: str,
    strategy: str = "0",
) -> list[dict[str, Any]]:
    """
    调用高德公交换乘接口（单次策略）。

    高铁场景：只提取 railway 主交通段，地铁/公交接驳会被丢弃。
    航班场景：提取 buslines 中的航班段。
    """
    _debug(
        "调用高德公交换乘路径规划: "
        f"origin={format_amap_point(origin)}, destination={format_amap_point(destination)}, "
        f"city={origin_city}, cityd={destination_city}, strategy={strategy}"
    )
    data = amap_get(
        AMAP_TRANSIT_URL,
        params={
            "origin": format_amap_point(origin),
            "destination": format_amap_point(destination),
            "city": origin_city,
            "cityd": destination_city,
            "extensions": "all",
            "strategy": strategy,
            "nightflag": "0",
        },
        timeout=12,
    )
    route = data.get("route", {})
    transits = [item for item in route.get("transits", []) if isinstance(item, dict)]
    print(f"[intercity_tools][TRACE] 高德公交换乘原始响应(strategy={strategy}): transits={len(transits)}, "
          f"distance={route.get('distance')}, taxi_cost={route.get('taxi_cost')}")
    if not transits:
        print(f"[intercity_tools][TRACE] ⚠️ 高德公交换乘返回 transits 为空！route keys: {list(route.keys())}")
    for index, transit in enumerate(transits[:5], 1):
        railways = _iter_railways(transit)
        seg_types: list[str] = []
        for seg in _ensure_list(transit.get("segments")):
            if isinstance(seg, dict):
                has_bus = "bus" in seg and seg.get("bus", {}).get("buslines")
                has_railway = "railway" in seg and seg.get("railway")
                has_walking = "walking" in seg
                types = []
                if has_bus: types.append("bus")
                if has_railway: types.append("railway")
                if has_walking: types.append("walking")
                seg_types.append("+".join(types) if types else "empty")
        print(f"[intercity_tools][TRACE]   换乘候选 {index}: duration={transit.get('duration')}s, "
              f"cost={transit.get('cost')}, segments={seg_types}, "
              f"railways={len(railways)}, rail_names={[_railway_service_no(r) for r in railways]}, "
              f"rail_types={[_railway_type(r) for r in railways]}")
    return transits


def _call_amap_transit_multi_strategy(
    origin: tuple[float, float],
    destination: tuple[float, float],
    origin_city: str,
    destination_city: str,
) -> list[dict[str, Any]]:
    """
    多策略合并调用高德公交换乘接口。

    高铁跨城场景下，单策略可能只返回 1-2 个换乘方案，
    用 0（最快）、2（最少换乘）、1（最经济）三种策略合并去重，增加候选数。
    """
    all_transits: dict[str, dict[str, Any]] = {}  # 用 route_summary 去重
    for strategy in ("0", "2", "1"):
        try:
            transits = _call_amap_transit_route(origin, destination, origin_city, destination_city, strategy=strategy)
            for t in transits:
                # 用 duration+cost 组合作为简易去重 key
                key = f"{t.get('duration')}_{t.get('cost')}"
                if key not in all_transits:
                    all_transits[key] = t
        except Exception as e:
            print(f"[intercity_tools][TRACE] 策略 {strategy} 调用失败: {e}")
    merged = list(all_transits.values())
    print(f"[intercity_tools][TRACE] 多策略合并后 transits 总数: {len(merged)}")
    return merged


def _iter_railways(transit: dict[str, Any]) -> list[dict[str, Any]]:
    """从高德换乘方案的 segments[].railway 中提取铁路段。"""
    railways: list[dict[str, Any]] = []
    for segment in _ensure_list(transit.get("segments")):
        if not isinstance(segment, dict):
            continue
        for railway in _ensure_list(segment.get("railway")):
            if isinstance(railway, dict) and railway:
                railways.append(railway)
    return railways


def _railway_service_no(railway: dict[str, Any]) -> str:
    """提取高铁/动车车次。"""
    for key in ("trip", "name", "id"):
        value = railway.get(key)
        if value:
            return str(value)
    return ""


def _railway_type(railway: dict[str, Any]) -> str:
    """提取铁路类型。"""
    for key in ("type", "typecode", "vehicle_type"):
        value = railway.get(key)
        if value:
            return str(value)
    return ""


def _stop_name(stop: Any) -> str:
    """提取站点名称。"""
    stop = _first_item(stop)
    if isinstance(stop, dict):
        return str(stop.get("name", "") or stop.get("station", ""))
    return str(stop) if stop else ""


def _first_non_empty(*values: Any) -> str | None:
    """返回第一个非空字符串。"""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _stop_time(stop: Any, railway: dict[str, Any], prefixes: tuple[str, ...]) -> str | None:
    """从站点或 railway 对象中提取出发/到达时间。"""
    stop = _first_item(stop)
    candidates: list[Any] = []
    if isinstance(stop, dict):
        candidates.extend(
            [
                stop.get("time"),
                stop.get("start_time"),
                stop.get("end_time"),
                stop.get("departure_time"),
                stop.get("arrival_time"),
            ]
        )
    for prefix in prefixes:
        candidates.extend(
            [
                railway.get(f"{prefix}_time"),
                railway.get(f"{prefix}time"),
                railway.get(prefix),
            ]
        )
    return _first_non_empty(*candidates)


def _railway_min_ticket_cost(railway: dict[str, Any]) -> float:
    """从 railway.spaces 中读取最低票价。"""
    costs: list[float] = []
    for space in _ensure_list(railway.get("spaces")):
        if not isinstance(space, dict):
            continue
        cost = _to_float(space.get("cost"))
        if cost > 0:
            costs.append(cost)
    return min(costs) if costs else 0.0


def _railway_distance_km(railway: dict[str, Any]) -> float:
    """提取铁路段距离，单位公里。"""
    distance = _to_float(railway.get("distance"))
    if distance > 10000:
        return round(distance / 1000, 1)
    return round(distance, 1)


def _railway_score(railway: dict[str, Any]) -> int:
    """给 railway 段打分，用于选择高铁/动车/城际主段。"""
    service_no = _railway_service_no(railway)
    railway_type = _railway_type(railway)
    text = _extract_text(railway)
    score = 30
    if railway_type in HIGH_SPEED_RAIL_TYPE_CODES:
        score += 120
    if re.match(r"^[GDC]\d+", service_no):
        score += 100
    if any(keyword in text for keyword in HIGH_SPEED_RAIL_KEYWORDS):
        score += 60
    if "铁路" in text or "火车" in text:
        score += 20
    return score


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
            score += 80   # 午后出发最佳
        elif 16 <= dep_hour <= 18:
            score += 100  # 下午出发最好，玩到最后一刻
        elif 19 <= dep_hour <= 20:
            score += 60   # 傍晚出发良好
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
            score += 10   # 夜间到家尚可
        else:
            score -= 80   # 深夜到家

    return score


def _select_railway_plan(
    transits: list[dict[str, Any]],
    direction: Literal["outbound", "return"] = "outbound",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """从高德换乘候选中选择最合适的 railway 主段（方向感知时间评分）。"""
    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for transit in transits:
        duration = _to_int(transit.get("duration"), 10**9)
        for railway in _iter_railways(transit):
            candidates.append((_railway_score(railway), duration, transit, railway))

    # 综合 type_score + 方向感知时间合理性评分，duration 做 tiebreaker
    scored_candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    print(f"[intercity_tools][TRACE] _select_railway_plan direction={direction}, raw_candidates={len(candidates)}")
    for type_score, duration, transit, railway in candidates:
        departure = _stop_time(railway.get("departure_stop"), railway, ("departure", "start"))
        arrival = _stop_time(railway.get("arrival_stop"), railway, ("arrival", "end"))
        time_bonus = _time_reasonableness_score(departure, arrival, direction)
        total_score = type_score + time_bonus
        scored_candidates.append((total_score, duration, transit, railway))
        print(f"[intercity_tools][TRACE]   候选: service_no={_railway_service_no(railway)}, "
              f"type={_railway_type(railway)}, type_score={type_score}, "
              f"dep={departure}, arr={arrival}, time_bonus={time_bonus}, "
              f"total={total_score}, duration={duration}min")

    _debug(
        "高铁/动车 railway 筛选分数（含时间合理性）: "
        + json.dumps(
            [
                {
                    "total_score": s,
                    "duration": d,
                    "service_no": _railway_service_no(r),
                    "type": _railway_type(r),
                    "cost": _railway_min_ticket_cost(r),
                    "departure": _stop_name(r.get("departure_stop")),
                    "arrival": _stop_name(r.get("arrival_stop")),
                }
                for s, d, _t, r in scored_candidates[:8]
            ],
            ensure_ascii=False,
        )
    )

    if not scored_candidates:
        print(f"[intercity_tools][TRACE] ❌ 无 railway 候选，scored_candidates 为空")
        return None
    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    score, duration, transit, railway = scored_candidates[0]
    print(f"[intercity_tools][TRACE] 最高分候选: service_no={_railway_service_no(railway)}, "
          f"score={score}, threshold=30, {'✅ 通过' if score >= 30 else '❌ 不通过'}")
    if score < 30:  # 只要有 railway 基础分即可入围，时间分只影响排序
        return None
    _debug(f"高铁/动车 railway 命中: service_no={_railway_service_no(railway)}, score={score}, duration={duration}")
    return transit, railway


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


def _normalize_railway_segment(
    transit: dict[str, Any],
    railway: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
) -> dict[str, Any]:
    """将高德 railway 主段归一化为单程跨城交通结构。"""
    service_no = _railway_service_no(railway) or None
    departure_stop = railway.get("departure_stop")
    arrival_stop = railway.get("arrival_stop")
    departure_place = _stop_name(departure_stop) or origin_city
    arrival_place = _stop_name(arrival_stop) or destination_city
    departure_time = _stop_time(departure_stop, railway, ("departure", "start"))
    arrival_time = _stop_time(arrival_stop, railway, ("arrival", "end"))
    price_per_person = _railway_min_ticket_cost(railway)
    if price_per_person <= 0:
        price_per_person = _to_float(transit.get("cost"))
    distance_km = _railway_distance_km(railway)
    if distance_km <= 0:
        distance_km = 0
    duration_minutes = _parse_time_minutes(departure_time, arrival_time) or max(1, math.ceil(_to_float(transit.get("duration")) / 60))
    estimated_cost = int(round(price_per_person * max(1, people_count))) if price_per_person > 0 else 0

    route_summary = (
        f"{service_no or '高铁/动车'}: {departure_place}"
        f"{' ' + departure_time if departure_time else ''} → {arrival_place}"
        f"{' ' + arrival_time if arrival_time else ''}"
    )

    return {
        "direction": direction,
        "origin": origin_city,
        "destination": destination_city,
        "date": date,
        "mode": "high_speed_rail",
        "duration_minutes": duration_minutes,
        "distance_km": round(distance_km, 1),
        "estimated_cost": estimated_cost,
        "route_summary": route_summary,
        "notes": ["数据来源：高德公交换乘 railway 主交通段；已删除地铁/公交接驳。"],
        "service_no": service_no,
        "carrier": "铁路",
        "departure_place": departure_place,
        "arrival_place": arrival_place,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "price_per_person": round(price_per_person, 2) if price_per_person > 0 else None,
        "data_source": "amap_railway",
        "is_estimated": False,
    }


def _iter_flights(transit: dict[str, Any]) -> list[dict[str, Any]]:
    """从高德公交换乘方案的 segments 中提取飞机/航班段"""
    flights: list[dict[str, Any]] = []
    for segment in _ensure_list(transit.get("segments")):
        if not isinstance(segment, dict):
            continue
        for busline in _ensure_list(segment.get("bus", {}).get("buslines", [])):
            if not isinstance(busline, dict):
                continue
            bus_type = str(busline.get("type", ""))
            bus_name = str(busline.get("name", ""))
            if "飞机" in bus_type or "航班" in bus_type or "air" in bus_type.lower() or "飞机" in bus_name or "航班" in bus_name:
                flights.append(busline)
        # 也检查 railway 段中是否有航班标识
        for railway in _ensure_list(segment.get("railway")):
            if isinstance(railway, dict) and railway:
                rtype = str(railway.get("type", ""))
                if "飞机" in rtype or "航班" in rtype:
                    flights.append(railway)
    return flights


def _flight_score(busline: dict[str, Any]) -> int:
    """给航班段打分：优先直飞、时间合理的"""
    score = 50
    bus_type = str(busline.get("type", ""))
    bus_name = str(busline.get("name", ""))
    if "直飞" in bus_type or "直飞" in bus_name:
        score += 100
    if "经停" in bus_type or "经停" in bus_name or "中转" in bus_name:
        score -= 30
    return score


def _select_flight_plan(
    transits: list[dict[str, Any]],
    direction: Literal["outbound", "return"] = "outbound",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """从高德换乘候选中选择最合适的航班段（方向感知时间评分）"""
    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for transit in transits:
        duration = _to_int(transit.get("duration"), 10**9)
        for flight in _iter_flights(transit):
            departure = str(flight.get("departure_time", "") or "")
            arrival = str(flight.get("arrival_time", "") or "")
            time_bonus = _time_reasonableness_score(departure, arrival, direction)
            total = _flight_score(flight) + time_bonus
            candidates.append((total, duration, transit, flight))

    _debug(
        "航班筛选分数: "
        + json.dumps(
            [
                {
                    "score": s,
                    "duration": d,
                    "name": f.get("name"),
                    "type": f.get("type"),
                    "departure_stop": f.get("departure_stop", {}).get("name") if isinstance(f.get("departure_stop"), dict) else f.get("departure_stop"),
                    "arrival_stop": f.get("arrival_stop", {}).get("name") if isinstance(f.get("arrival_stop"), dict) else f.get("arrival_stop"),
                }
                for s, d, _t, f in candidates[:8]
            ],
            ensure_ascii=False,
        )
    )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, duration, transit, flight = candidates[0]
    if score < 30:
        return None
    _debug(f"航班命中: name={flight.get('name')}, score={score}, duration={duration}")
    return transit, flight


def _normalize_flight_segment(
    transit: dict[str, Any],
    flight: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
) -> dict[str, Any]:
    """将高德航班段归一化为单程跨城交通结构"""
    service_no = str(flight.get("name", "") or flight.get("trip", "") or "").strip() or None
    departure_stop = flight.get("departure_stop")
    arrival_stop = flight.get("arrival_stop")
    departure_place = _stop_name(departure_stop) or origin_city
    arrival_place = _stop_name(arrival_stop) or destination_city
    departure_time = str(flight.get("departure_time", "") or flight.get("start_time", "") or "")
    arrival_time = str(flight.get("arrival_time", "") or flight.get("end_time", "") or "")
    price_per_person = _to_float(flight.get("cost") or flight.get("price") or transit.get("cost"))
    distance_km = _to_float(flight.get("distance"))
    if distance_km > 10000:
        distance_km = round(distance_km / 1000, 1)
    duration_minutes = _parse_time_minutes(departure_time or None, arrival_time or None) or max(1, math.ceil(_to_float(transit.get("duration")) / 60))
    estimated_cost = int(round(price_per_person * max(1, people_count))) if price_per_person > 0 else 0

    route_summary = (
        f"{service_no or '航班'}: {departure_place}"
        f"{' ' + departure_time if departure_time else ''} → {arrival_place}"
        f"{' ' + arrival_time if arrival_time else ''}"
    )

    return {
        "direction": direction,
        "origin": origin_city,
        "destination": destination_city,
        "date": date,
        "mode": "flight",
        "duration_minutes": duration_minutes,
        "distance_km": round(distance_km, 1),
        "estimated_cost": estimated_cost,
        "route_summary": route_summary,
        "notes": ["数据来源：高德公交换乘航班段。"],
        "service_no": service_no,
        "carrier": "航班",
        "departure_place": departure_place,
        "arrival_place": arrival_place,
        "departure_time": departure_time or None,
        "arrival_time": arrival_time or None,
        "price_per_person": round(price_per_person, 2) if price_per_person > 0 else None,
        "data_source": "amap_flight",
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


def _try_railway_plan(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    origin_coord: tuple[float, float],
    destination_coord: tuple[float, float],
) -> dict[str, Any]:
    """尝试生成只包含高铁/动车主段的往返方案（多策略合并，只提取 railway 主段）。"""
    print(f"[intercity_tools][TRACE] _try_railway_plan 开始: {departure_city} → {destination_city}")
    outbound_transits = _call_amap_transit_multi_strategy(origin_coord, destination_coord, departure_city, destination_city)
    print(f"[intercity_tools][TRACE] 去程 transits 数量(多策略合并): {len(outbound_transits)}")
    return_transits = _call_amap_transit_multi_strategy(destination_coord, origin_coord, destination_city, departure_city)
    print(f"[intercity_tools][TRACE] 返程 transits 数量(多策略合并): {len(return_transits)}")
    outbound_selection = _select_railway_plan(outbound_transits, "outbound")
    print(f"[intercity_tools][TRACE] 去程 railway 选中: {outbound_selection is not None}")
    return_selection = _select_railway_plan(return_transits, "return")
    print(f"[intercity_tools][TRACE] 返程 railway 选中: {return_selection is not None}")
    if not outbound_selection or not return_selection:
        raise RuntimeError("高德未返回可用 railway 高铁/动车主段。")

    outbound_transit, outbound_railway = outbound_selection
    return_transit, return_railway = return_selection
    outbound = _normalize_railway_segment(
        outbound_transit, outbound_railway,
        departure_city, destination_city, start_date, "outbound", people_count,
    )
    return_trip = _normalize_railway_segment(
        return_transit, return_railway,
        destination_city, departure_city, end_date, "return", people_count,
    )
    return _build_plan_from_segments("high_speed_rail", outbound, return_trip, [])


def _try_flight_plan(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    origin_coord: tuple[float, float],
    destination_coord: tuple[float, float],
) -> dict[str, Any]:
    """使用高德公交换乘 API 提取航班往返方案"""
    outbound_transits = _call_amap_transit_route(origin_coord, destination_coord, departure_city, destination_city)
    return_transits = _call_amap_transit_route(destination_coord, origin_coord, destination_city, departure_city)
    outbound_selection = _select_flight_plan(outbound_transits, "outbound")
    return_selection = _select_flight_plan(return_transits, "return")
    if not outbound_selection or not return_selection:
        raise RuntimeError("高德未返回可用航班段。")

    outbound_transit, outbound_flight = outbound_selection
    return_transit, return_flight = return_selection
    outbound = _normalize_flight_segment(
        outbound_transit, outbound_flight,
        departure_city, destination_city, start_date, "outbound", people_count,
    )
    return_trip = _normalize_flight_segment(
        return_transit, return_flight,
        destination_city, departure_city, end_date, "return", people_count,
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
            warnings.append(f"高德未返回可用 railway 高铁/动车主段，已使用估算。原因：{exc}")
        else:
            warnings.append(f"高德驾车路径规划失败，已使用估算。原因：{exc}")

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


@tool
def estimate_intercity_transport_tool(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
    mode: IntercityMode,
) -> str:
    """
    查询或估算出发地到目的地的跨城往返交通方案。

    参数：
        departure_city: 出发城市。
        destination_city: 目的地城市。
        start_date: 去程日期，格式为 YYYY-MM-DD。
        end_date: 返程日期，格式为 YYYY-MM-DD。
        people_count: 出行人数。
        mode: driving、high_speed_rail 或 flight。

    返回：
        JSON 字符串。真实班次不可用时返回估算兜底，但不伪造班次和时间。
    """
    result = estimate_intercity_transport(
        departure_city=departure_city,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        people_count=people_count,
        mode=mode,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
