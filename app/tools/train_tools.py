"""
12306 火车票查询工具 — 直接调用 12306 官方 API + 高铁规划层。

底层 API：
    search_stations(keyword)     → list[dict]
    query_tickets(from, to, date) → list[dict]
    query_ticket_price(from, to, date, train_code) → dict
    resolve_station_telecode(city) → str | None

高层规划：
    plan_train(departure_city, destination_city, start_date, end_date, people_count) → dict
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal, Optional

import httpx

from app.tools.station_service import station_service
from app.utils.transport_utils import (
    build_plan_from_segments,
    create_estimated_fallback,
    normalize_city,
    time_reasonableness_score,
    to_float,
    to_int,
)
from app.utils.date_utils import validate_date_not_past

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 12306 API 常量和请求头
# ═══════════════════════════════════════════════════════════════

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

HTTP_URLS = {
    "init": "https://kyfw.12"
            "306.cn/otn/leftTicket/init",
    "query_left_ticket": "https://kyfw.12306.cn/otn/leftTicket/queryG",
    "query_price": "https://kyfw.12306.cn/otn/leftTicketPrice/queryAllPublicPrice",
}

HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://kyfw.12306.cn/otn/leftTicket/init",
    "Host": "kyfw.12306.cn",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "keep-alive",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://kyfw.12306.cn",
}

# 模块首次使用时自动加载车站数据
stations_loaded = False


def init_stations() -> None:
    """懒加载车站数据（从 app/tools/station_name.js）。"""
    global stations_loaded
    if stations_loaded:
        return
    from app.utils.config import settings

    path = settings.station_data_path or None
    station_service.load_stations(path)
    stations_loaded = True
    logger.info(f"[train_tools] 车站数据已加载: {len(station_service.stations)} 个")


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def ensure_telecode(val: str) -> Optional[str]:
    """车站名/三字码 → 三字码（电报码）。"""
    if not val:
        return None
    if val.isalpha() and val.isupper() and len(val) == 3:
        return val
    return station_service.get_station_code(val)


def parse_ticket_string(ticket_str: str, query: dict) -> Optional[dict]:
    """解析 12306 单条车票字符串。"""
    parts = ticket_str.split('|')
    if len(parts) < 35:
        return None
    return {
        "train_no": parts[3],
        "start_time": parts[8],
        "arrive_time": parts[9],
        "duration": parts[10],
        "business_seat_num": parts[32] or "",
        "first_class_num": parts[31] or "",
        "second_class_num": parts[30] or "",
        "advanced_soft_sleeper_num": parts[21] or "",
        "soft_sleeper_num": parts[23] or "",
        "dongwo_num": parts[33] or "",
        "hard_sleeper_num": parts[28] or "",
        "soft_seat_num": parts[24] or "",
        "hard_seat_num": parts[29] or "",
        "no_seat_num": parts[26] or "",
        "from_station": query["from_station"],
        "to_station": query["to_station"],
        "train_date": query["train_date"],
    }


def http_client() -> httpx.Client:
    """创建带默认配置的 httpx 同步客户端。"""
    return httpx.Client(follow_redirects=False, timeout=10, verify=False)


# ═══════════════════════════════════════════════════════════════
# 公开 API（保持与旧 train_tools 相同的函数签名）
# ═══════════════════════════════════════════════════════════════

def search_stations(keyword: str) -> list[dict[str, str]]:
    """
    模糊搜索火车站。

    返回: [{"name": "北京", "code": "BJP", "pinyin": "beijing", "py_short": "bj", "city": "北京"}, ...]
    """
    init_stations()

    result = station_service.search_stations(keyword, limit=15)
    stations = []
    for s in result.stations:
        stations.append({
            "name": s.name,
            "code": s.code,
            "pinyin": s.pinyin,
            "py_short": s.py_short,
            "city": s.city or "",
        })

    logger.debug(f"[train_tools] search-stations({keyword!r}) → {len(stations)} 条")
    return stations


def resolve_station_telecode(city: str) -> Optional[str]:
    """
    城市名 → 首选火车站电报码。

    搜索城市对应的车站，优先选主站（不带方位后缀的）。
    例如 "北京" → "BJP"（北京站），而非 "BXP"（北京西站）。
    """
    init_stations()

    stations = search_stations(city)
    if not stations:
        logger.warning(f"[train_tools] 车站解析失败: {city}")
        return None

    # 优先精确匹配城市名
    for s in stations:
        if s.get("name") == city or s.get("city") == city:
            code = s.get("code", "")
            logger.debug(f"[train_tools] 车站匹配: {city} → {s.get('name')}({code})")
            return code

    # 退而取第一条
    first = stations[0]
    code = first.get("code", "")
    logger.debug(f"[train_tools] 车站匹配(退而): {city} → {first.get('name')}({code})")
    return code


def query_tickets(
    from_station: str,
    to_station: str,
    date: str,
) -> list[dict[str, Any]]:
    """
    查询指定日期两站之间的所有车次。

    参数:
        from_station: 出发站名或三位电报码（如 "北京" 或 "BJP"）
        to_station: 到达站名或三位电报码
        date: 乘车日期 YYYY-MM-DD

    返回: 车次列表，每项含 train_no / from_station_name / to_station_name /
          start_time / arrive_time / duration / seats
    """
    init_stations()

    # 日期校验
    is_valid, error_msg = validate_date_not_past(date)
    if not is_valid:
        logger.warning(f"[train_tools] 日期校验失败: {error_msg}")
        return []

    # 车站名 → 电报码
    from_code = ensure_telecode(from_station)
    to_code = ensure_telecode(to_station)
    if not from_code or not to_code:
        logger.warning(f"[train_tools] 车站解析失败: {from_station}→{from_code}, {to_station}→{to_code}")
        return []

    url_init = HTTP_URLS["init"]
    url_query = HTTP_URLS["query_left_ticket"]
    headers = HTTP_HEADERS.copy()

    tickets_data = []
    max_retries = 3

    for attempt in range(max_retries):
        try:
            with http_client() as client:
                # 先访问 init 页面获取 cookie
                client.get(url_init, headers=headers)

                params = {
                    "leftTicketDTO.train_date": date,
                    "leftTicketDTO.from_station": from_code,
                    "leftTicketDTO.to_station": to_code,
                    "purpose_codes": "ADULT",
                }
                resp = client.get(url_query, headers=headers, params=params)

                if resp.status_code != 200:
                    logger.warning(f"[train_tools] 12306 返回异常状态码: {resp.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return []

                data = resp.json().get("data", {})
                tickets_data = data.get("result", [])
                break

        except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
            logger.warning(f"[train_tools] 网络请求失败 ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return []
        except Exception as e:
            logger.error(f"[train_tools] 查询车票异常: {e}")
            return []

    # 解析结果
    tickets = []
    for ticket_str in tickets_data:
        ticket = parse_ticket_string(ticket_str, {
            "from_station": from_station,
            "to_station": to_station,
            "train_date": date,
        })
        if not ticket:
            continue

        # 从原始串中取实际站点三字码，再反查中文名
        parts = ticket_str.split('|')
        from_code_actual = parts[6] if len(parts) > 6 else None
        to_code_actual = parts[7] if len(parts) > 7 else None

        from_station_obj = station_service.get_station_by_code(from_code_actual) if from_code_actual else None
        to_station_obj = station_service.get_station_by_code(to_code_actual) if to_code_actual else None
        from_station_name = from_station_obj.name if from_station_obj else (from_code_actual or from_station)
        to_station_name = to_station_obj.name if to_station_obj else (to_code_actual or to_station)

        # 构建座位信息
        seats = {}
        if ticket.get("business_seat_num"):
            seats["business"] = ticket["business_seat_num"]
        if ticket.get("first_class_num"):
            seats["first_class"] = ticket["first_class_num"]
        if ticket.get("second_class_num"):
            seats["second_class"] = ticket["second_class_num"]
        if ticket.get("advanced_soft_sleeper_num"):
            seats["advanced_soft_sleeper"] = ticket["advanced_soft_sleeper_num"]
        if ticket.get("soft_sleeper_num"):
            seats["soft_sleeper"] = ticket["soft_sleeper_num"]
        if ticket.get("hard_sleeper_num"):
            seats["hard_sleeper"] = ticket["hard_sleeper_num"]
        if ticket.get("soft_seat_num"):
            seats["soft_seat"] = ticket["soft_seat_num"]
        if ticket.get("hard_seat_num"):
            seats["hard_seat"] = ticket["hard_seat_num"]
        if ticket.get("no_seat_num"):
            seats["no_seat"] = ticket["no_seat_num"]
        if ticket.get("dongwo_num"):
            seats["dongwo"] = ticket["dongwo_num"]

        tickets.append({
            "train_no": ticket["train_no"],
            "from_station_name": from_station_name,
            "to_station_name": to_station_name,
            "start_time": ticket["start_time"],
            "arrive_time": ticket["arrive_time"],
            "duration": ticket["duration"],
            "seats": seats,
        })

    logger.info(f"[train_tools] query-tickets({from_station}→{to_station}, {date}) → {len(tickets)} 条")
    if tickets:
        logger.debug(f"[train_tools]   第一条: {tickets[0]}")
    return tickets


def query_ticket_price(
    from_station: str,
    to_station: str,
    train_date: str,
    train_code: str,
) -> dict[str, Any]:
    """
    查询指定车次的票价信息。

    返回: {"prices": {"二等座": "23.0", "一等座": "45.5", ...}, ...}
          或空 dict（查询失败时）
    """
    init_stations()

    # 车站名 → 电报码
    from_code = ensure_telecode(from_station)
    to_code = ensure_telecode(to_station)
    if not from_code or not to_code:
        logger.warning(f"[train_tools] 票价查询车站解析失败: {from_station}→{from_code}, {to_station}→{to_code}")
        return {}

    url_init = HTTP_URLS["init"]
    url_price = HTTP_URLS["query_price"]
    headers = HTTP_HEADERS.copy()

    params = {
        "leftTicketDTO.train_date": train_date,
        "leftTicketDTO.from_station": from_code,
        "leftTicketDTO.to_station": to_code,
        "purpose_codes": "ADULT",
    }

    max_retries = 3
    json_data = None

    for attempt in range(max_retries):
        try:
            with http_client() as client:
                client.get(url_init, headers=headers)
                resp = client.get(url_price, headers=headers, params=params)

                if resp.status_code != 200:
                    logger.warning(f"[train_tools] 票价查询异常状态码: {resp.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {}

                json_data = resp.json()
                break

        except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
            logger.warning(f"[train_tools] 票价查询网络失败 ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return {}
        except Exception as e:
            logger.error(f"[train_tools] 票价查询异常: {e}")
            return {}

    if not json_data or "data" not in json_data:
        return {}

    price_map = {
        "wz_price": "无座",
        "yz_price": "硬座",
        "yw_price": "硬卧",
        "rw_price": "软卧",
        "gr_price": "高级软卧",
        "ze_price": "二等座",
        "zy_price": "一等座",
        "swz_price": "商务座",
        "tdz_price": "特等座",
        "dw_price": "动卧",
    }

    for item in json_data.get("data", []):
        dto = item.get("queryLeftNewDTO", {})
        current_code = dto.get("station_train_code", "")

        if train_code and current_code != train_code:
            continue

        prices = {}
        for key, name in price_map.items():
            price_val = dto.get(key)
            if price_val and price_val != "--" and price_val != "":
                try:
                    if isinstance(price_val, str) and price_val.isdigit():
                        # 12306 返回的价格最后一位是角，例如 "00230" 表示 23.0 元
                        price_int = int(price_val)
                        price_str = str(price_int)
                        if len(price_str) == 1:
                            formatted = "0." + price_str
                        else:
                            formatted = price_str[:-1] + "." + price_str[-1]
                        prices[name] = formatted
                    else:
                        prices[name] = str(price_val)
                except (ValueError, TypeError):
                    prices[name] = str(price_val)

        result = {
            "prices": prices,
            "train_no": dto.get("train_no"),
            "train_code": current_code,
            "from_station_name": dto.get("from_station_name"),
            "to_station_name": dto.get("to_station_name"),
        }
        logger.info(f"[train_tools] query-ticket-price({train_code}) → prices={prices}")
        return result

    return {}


def list_tools() -> list[dict[str, Any]]:
    """列出所有可用工具（调试用，兼容旧接口）。"""
    return [
        {"name": "query-tickets", "description": "查询 12306 余票/车次/座席/时刻"},
        {"name": "query-ticket-price", "description": "查询火车票价信息"},
        {"name": "search-stations", "description": "智能车站搜索"},
    ]


# ═══════════════════════════════════════════════════════════════
# 高铁规划层（从 intercity_tools 迁入）
# ═══════════════════════════════════════════════════════════════

def is_high_speed_train(ticket: dict[str, Any]) -> bool:
    """判断是否为高铁/动车/城际（G/D/C 字头）。"""
    train_no = str(ticket.get("train_no", "") or "")
    return bool(re.match(r"^[GDC]", train_no, re.IGNORECASE))


def train_duration_minutes(ticket: dict[str, Any]) -> int:
    """解析 12306 车票持续时间（格式可能是 HH:MM 或分钟数）。"""
    duration = ticket.get("duration", "")
    if isinstance(duration, (int, float)):
        return int(duration)
    if isinstance(duration, str) and ":" in duration:
        parts = duration.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    return to_int(duration, 0)


def train_second_class_price(source: dict[str, Any]) -> float:
    # 先尝试嵌套 prices 字段
    prices = source.get("prices", {})
    if isinstance(prices, dict):
        for key, val in prices.items():
            if "二等" in key or "second" in key.lower() or "edz" in key.lower():
                p = to_float(val)
                if p > 0:
                    return p
        for val in prices.values():
            p = to_float(val)
            if p > 0:
                return p
    # 顶层直接查找
    for key in ("second_class_price", "price_2nd", "二等座", "second_price",
                "edz_price", "EDZ", "second_seat_price", "ZE", "ze_price"):
        val = source.get(key)
        if val is not None and to_float(val) > 0:
            return to_float(val)
    # 遍历顶层字段
    for key, val in source.items():
        if isinstance(val, (int, float)) and val > 0:
            key_lower = key.lower()
            if "二等" in key or "second" in key_lower or "edz" in key_lower:
                return to_float(val)
    # 兜底
    for key in ("price", "ticket_price", "total_price"):
        val = source.get(key)
        if val is not None and to_float(val) > 0:
            return to_float(val)
    return 0.0

def select_train(
    tickets: list[dict[str, Any]],
    direction: Literal["outbound", "return"],
) -> dict[str, Any] | None:
    """从 12306 车票列表中选出最合适的高铁/动车（方向感知时间评分）。"""
    high_speed = [t for t in tickets if is_high_speed_train(t)]
    print(f"[train_tools] _select_train direction={direction}, "
          f"总车次={len(tickets)}, 高铁/动车={len(high_speed)}")
    if tickets:
        print(f"[train_tools]   第一条车次全部字段: {json.dumps(tickets[0], ensure_ascii=False)}")

    if not high_speed:
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for ticket in high_speed:
        dep_time = str(ticket.get("start_time", "") or "")
        arr_time = str(ticket.get("arrive_time", "") or "")
        time_score = time_reasonableness_score(dep_time, arr_time, direction)
        duration = train_duration_minutes(ticket)
        train_no = ticket.get("train_no", "")
        prefix = train_no[0].upper() if train_no else ""
        type_bonus = 30 if prefix == "G" else (20 if prefix == "D" else (10 if prefix == "C" else 0))
        total = time_score + type_bonus
        scored.append((total, ticket))
        print(f"[train_tools]   车次: {train_no}, dep={dep_time}, arr={arr_time}, "
              f"duration={duration}min, type_bonus={type_bonus}, time_score={time_score}, "
              f"total={total}, price={train_second_class_price(ticket)}")

    scored.sort(key=lambda item: (-item[0], train_duration_minutes(item[1])))
    best_score, best_ticket = scored[0]
    print(f"[train_tools] 最高分车次: {best_ticket.get('train_no')}, "
          f"score={best_score}, {'✅ 选中' if best_score >= -50 else '❌ 分数过低'}")

    if best_score < -200:
        return None
    return best_ticket


def normalize_train_result(
    ticket: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
    price_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_no = str(ticket.get("train_no", "") or ticket.get("name", "") or "").strip()
    departure_station = str(ticket.get("from_station_name", "") or ticket.get("start_station", "") or origin_city)
    arrival_station = str(ticket.get("to_station_name", "") or ticket.get("end_station", "") or destination_city)
    departure_time = str(ticket.get("start_time", "") or "")
    arrival_time = str(ticket.get("arrive_time", "") or "")
    duration_minutes = train_duration_minutes(ticket)
    price_source = price_info if price_info else ticket
    price_per_person = train_second_class_price(price_source)
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


def plan_train(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
) -> dict[str, Any]:
    """
    生成高铁/动车往返方案（基于 12306 实时数据）。

    参数：
        departure_city: 出发城市
        destination_city: 目的城市
        start_date: 出发日期 YYYY-MM-DD
        end_date: 返程日期 YYYY-MM-DD
        people_count: 人数

    返回：统一跨城交通方案 dict。
    """
    origin_name = normalize_city(departure_city)
    dest_name = normalize_city(destination_city)
    people = max(1, int(people_count or 1))

    print(f"[train_tools] plan_train(12306) 开始: {origin_name} → {dest_name}")

    from_code = resolve_station_telecode(origin_name)
    to_code = resolve_station_telecode(dest_name)
    print(f"[train_tools] 车站电报码: {origin_name}→{from_code}, {dest_name}→{to_code}")

    if not from_code or not to_code:
        return create_estimated_fallback(
            departure_city=origin_name,
            destination_city=dest_name,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode="high_speed_rail",
            warnings=[f"12306 车站解析失败: {origin_name}={from_code}, {dest_name}={to_code}"],
        )

    try:
        outbound_tickets = query_tickets(from_code, to_code, start_date)
        return_tickets = query_tickets(to_code, from_code, end_date)

        outbound_train = select_train(outbound_tickets, "outbound")
        return_train = select_train(return_tickets, "return")
        print(f"[train_tools] 去程选中: {outbound_train is not None}, 返程选中: {return_train is not None}")

        if not outbound_train or not return_train:
            raise RuntimeError("12306 未返回可用高铁/动车车次。")

        outbound_code = str(outbound_train.get("train_no", "") or "")
        return_code = str(return_train.get("train_no", "") or "")
        outbound_price = query_ticket_price(from_code, to_code, start_date, outbound_code) if outbound_code else {}
        return_price = query_ticket_price(to_code, from_code, end_date, return_code) if return_code else {}
        print(f"[train_tools] 票价: 去程={outbound_price}, 返程={return_price}")

        outbound = normalize_train_result(
            outbound_train, origin_name, dest_name, start_date, "outbound", people, outbound_price,
        )
        return_trip = normalize_train_result(
            return_train, dest_name, origin_name, end_date, "return", people, return_price,
        )
        return build_plan_from_segments("high_speed_rail", outbound, return_trip, [])
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return create_estimated_fallback(
            departure_city=origin_name,
            destination_city=dest_name,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode="high_speed_rail",
            warnings=[f"12306 未返回可用高铁/动车车次，已使用估算。原因：{exc}"],
        )

