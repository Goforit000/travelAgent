"""
Google Flights 航班查询工具 — 封装 fast_flights 库 + 航班规划层。

底层 API：
    search_airports(city)     → list[dict]  城市名→机场列表
    resolve_airport_code(city) → str | None  城市→首选机场三字码
    query_flights(from_code, to_code, date, **) → list[dict]  查航班

高层规划：
    plan_flight(departure_city, destination_city, start_date, end_date, people_count) → dict

数据来源：Google Flights（通过 fast_flights 抓取），无需 API Key。
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Literal, Optional

from app.utils.transport_utils import (
    build_plan_from_segments,
    create_estimated_fallback,
    normalize_city,
    time_reasonableness_score,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 城市中英文映射 + 机场数据库
# ═══════════════════════════════════════════════════════════════

# 常用城市中文名 → 首选机场三字码
CITY_CN_TO_CODE: dict[str, str] = {
    "北京": "PEK",
    "上海": "PVG",
    "广州": "CAN",
    "深圳": "SZX",
    "成都": "CTU",
    "重庆": "CKG",
    "杭州": "HGH",
    "武汉": "WUH",
    "西安": "XIY",
    "南京": "NKG",
    "天津": "TSN",
    "苏州": "SHA",  # 苏州无机场，走上海虹桥
    "长沙": "CSX",
    "郑州": "CGO",
    "青岛": "TAO",
    "大连": "DLC",
    "厦门": "XMN",
    "昆明": "KMG",
    "三亚": "SYX",
    "海口": "HAK",
    "贵阳": "KWE",
    "哈尔滨": "HRB",
    "沈阳": "SHE",
    "长春": "CGQ",
    "济南": "TNA",
    "福州": "FOC",
    "合肥": "HFE",
    "南昌": "KHN",
    "南宁": "NNG",
    "兰州": "LHW",
    "太原": "TYN",
    "石家庄": "SJW",
    "乌鲁木齐": "URC",
    "呼和浩特": "HET",
    "银川": "INC",
    "西宁": "XNN",
    "拉萨": "LXA",
    "宁波": "NGB",
    "温州": "WNZ",
    "珠海": "ZUH",
    "桂林": "KWL",
    "丽江": "LJG",
    "香港": "HKG",
    "台北": "TPE",
    "澳门": "MFM",
    # 日韩
    "东京": "NRT",
    "大阪": "KIX",
    "首尔": "ICN",
    "釜山": "PUS",
    # 东南亚
    "曼谷": "BKK",
    "新加坡": "SIN",
    "吉隆坡": "KUL",
    "河内": "HAN",
    "胡志明": "SGN",
    "马尼拉": "MNL",
    "雅加达": "CGK",
    # 欧美
    "伦敦": "LHR",
    "巴黎": "CDG",
    "纽约": "JFK",
    "洛杉矶": "LAX",
    "旧金山": "SFO",
    "悉尼": "SYD",
    "墨尔本": "MEL",
    "迪拜": "DXB",
    "莫斯科": "SVO",
    "法兰克福": "FRA",
    "阿姆斯特丹": "AMS",
}

# 英文城市名 → 首选机场三字码（非中国城市直接映射）
CITY_EN_TO_CODE: dict[str, str] = {
    "tokyo": "NRT", "osaka": "KIX", "seoul": "ICN",
    "bangkok": "BKK", "singapore": "SIN", "kuala lumpur": "KUL",
    "london": "LHR", "paris": "CDG", "new york": "JFK",
    "los angeles": "LAX", "san francisco": "SFO",
    "sydney": "SYD", "melbourne": "MEL",
    "hong kong": "HKG", "taipei": "TPE", "macau": "MFM",
    "dubai": "DXB", "milan": "MXP", "rome": "FCO",
    "barcelona": "BCN", "amsterdam": "AMS", "frankfurt": "FRA",
    "munich": "MUC", "zurich": "ZRH", "vienna": "VIE",
    "istanbul": "IST", "moscow": "SVO", "delhi": "DEL",
    "mumbai": "BOM", "jakarta": "CGK", "manila": "MNL",
    "ho chi minh": "SGN", "hanoi": "HAN",
}

# 城市英文名 → 中文名（用于 airports.csv 的 name 字段反向匹配）
CITY_EN_TO_CN: dict[str, str] = {
    "beijing": "北京", "shanghai": "上海", "guangzhou": "广州",
    "shenzhen": "深圳", "chengdu": "成都", "chongqing": "重庆",
    "hangzhou": "杭州", "wuhan": "武汉", "xian": "西安",
    "nanjing": "南京", "tianjin": "天津", "changsha": "长沙",
    "zhengzhou": "郑州", "qingdao": "青岛", "dalian": "大连",
    "xiamen": "厦门", "kunming": "昆明", "sanya": "三亚",
    "haikou": "海口", "guiyang": "贵阳", "harbin": "哈尔滨",
    "shenyang": "沈阳", "changchun": "长春", "jinan": "济南",
    "fuzhou": "福州", "hefei": "合肥", "nanchang": "南昌",
    "nanning": "南宁", "lanzhou": "兰州", "taiyuan": "太原",
    "shijiazhuang": "石家庄", "urumqi": "乌鲁木齐",
    "hohhot": "呼和浩特", "yinchuan": "银川", "xining": "西宁",
    "lhasa": "拉萨", "ningbo": "宁波", "wenzhou": "温州",
    "zhuhai": "珠海", "guilin": "桂林", "lijiang": "丽江",
}

airports_loaded = False
airports_by_code: dict[str, dict] = {}
airports_list: list[dict] = []


def load_airports() -> None:
    global airports_loaded, airports_by_code, airports_list
    if airports_loaded:
        return

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airports.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"[flight_tools] airports.csv 不存在: {csv_path}")
        airports_loaded = True
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            if not code:
                continue
            info = {
                "code": code,
                "name": (row.get("name") or "").strip(),
                "city": (row.get("city") or "").strip(),
                "country": (row.get("country_id") or "").strip(),
                "location": (row.get("location") or "").strip(),
            }
            airports_by_code[code] = info
            airports_list.append(info)

    airports_loaded = True
    logger.info(f"[flight_tools] 已加载 {len(airports_by_code)} 个机场")


def match_airport_name(name_en: str, keyword_lower: str) -> bool:
    """检查机场英文名是否匹配关键词（英/中）。"""
    name_lower = name_en.lower()
    if keyword_lower in name_lower:
        return True
    # 英文名 → 中文名 → 匹配
    for en_city, cn_city in CITY_EN_TO_CN.items():
        if cn_city == keyword_lower and en_city in name_lower:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════

def search_airports(keyword: str) -> list[dict[str, str]]:
    """
    按城市名搜索机场。

    支持中文城市名（如"北京"、"上海"）和英文名（如"Taipei"、直接三字码）。
    返回机场列表，每项含 code / name / city / country。
    """
    load_airports()

    keyword_lower = keyword.strip().lower()
    results: list[dict] = []

    # 先直接匹配三字码
    if keyword_lower.upper() in airports_by_code:
        results.append(airports_by_code[keyword_lower.upper()])

    # 匹配机场英文名
    for info in airports_list:
        if match_airport_name(info["name"], keyword_lower):
            results.append(info)

    # 去重，优先中国机场
    seen = set()
    unique: list[dict] = []
    for r in results:
        if r["code"] not in seen:
            seen.add(r["code"])
            unique.append(r)

    # 中国机场优先
    cn = [r for r in unique if r["country"] == "CN"]
    other = [r for r in unique if r["country"] != "CN"]
    unique = cn + other

    logger.debug(f"[flight_tools] search_airports({keyword!r}) → {len(unique)} 个")
    return unique


def fmt_time(t) -> str:
    """从 tuple/list 安全提取 HH:MM 时间字符串，缺失时返回空字符串。"""
    if not t or len(t) < 2:
        return ""
    try:
        return f"{int(t[0]):02d}:{int(t[1]):02d}"
    except (ValueError, TypeError, IndexError):
        return ""


def resolve_airport_code(city: str) -> Optional[str]:
    """
    城市名 → 首选机场三字码（IATA code）。

    优先查中文名映射表，其次搜索 airports.csv。
    """
    city_clean = city.strip()
    if not city_clean:
        return None

    # 直接是三字码
    if city_clean.isalpha() and city_clean.isupper() and len(city_clean) == 3:
        load_airports()
        if city_clean in airports_by_code:
            return city_clean

    # 去掉"市"后缀
    if city_clean.endswith("市"):
        city_clean = city_clean[:-1]

    # 查中文映射表
    if city_clean in CITY_CN_TO_CODE:
        code = CITY_CN_TO_CODE[city_clean]
        logger.debug(f"[flight_tools] 机场匹配(映射表): {city} → {code}")
        return code

    # 查英文映射表
    city_lower = city_clean.lower()
    if city_lower in CITY_EN_TO_CODE:
        code = CITY_EN_TO_CODE[city_lower]
        logger.debug(f"[flight_tools] 机场匹配(英→码): {city} → {code}")
        return code

    # 英文名反查中文再查映射
    if city_lower in CITY_EN_TO_CN:
        cn = CITY_EN_TO_CN[city_lower]
        if cn in CITY_CN_TO_CODE:
            code = CITY_CN_TO_CODE[cn]
            logger.debug(f"[flight_tools] 机场匹配(英→中): {city} → {code}")
            return code

    # 搜索 airports.csv
    airports = search_airports(city_clean)
    if not airports:
        logger.warning(f"[flight_tools] 机场解析失败: {city}")
        return None

    # 中国机场：优先不包含"Z"开头的（IATA码 vs ICAO码）
    cn_airports = [a for a in airports if a["country"] == "CN"]
    if cn_airports:
        iata = [a for a in cn_airports if not a["code"].startswith("Z")]
        if iata:
            iata.sort(key=lambda a: len(a["name"]))
            best = iata[0]
            logger.debug(f"[flight_tools] 机场匹配: {city} → {best['name']}({best['code']})")
            return best["code"]

    best = airports[0]
    logger.debug(f"[flight_tools] 机场匹配: {city} → {best['name']}({best['code']})")
    return best["code"]


def query_flights(
    from_airport: str,
    to_airport: str,
    date: str,
    *,
    people_count: int = 1,
    seat: str = "economy",
    trip: str = "one-way",
    language: str = "zh-CN",
    currency: str = "CNY",
    max_stops: int | None = None,
) -> list[dict[str, Any]]:
    """
    查询 Google Flights 航班。

    参数：
        from_airport: 出发机场三字码（如 "PEK"）
        to_airport: 到达机场三字码（如 "SHA"）
        date: 出发日期 YYYY-MM-DD
        people_count: 乘客数
        seat: 舱位 economy / premium-economy / business / first
        trip: 行程类型 one-way / round-trip
        language: 界面语言
        currency: 价格币种

    返回：航班列表，每项含：
        price: 总价（指定币种）
        price_per_person: 人均价
        airlines: 航司代码列表
        flights: [{
            from_airport: {code, name},
            to_airport: {code, name},
            departure_time: "HH:MM",
            arrival_time: "HH:MM",
            duration_minutes: int,
            plane_type: str,
        }, ...]
        carbon_emission: int (grams)
    """
    load_airports()

    try:
        from app.tools.fast_flights import (
            FlightQuery,
            Passengers,
            create_query,
            get_flights,
        )
    except ImportError as e:
        logger.error(f"[flight_tools] fast_flights 依赖未安装: {e}")
        return []

    try:
        query = create_query(
            flights=[
                FlightQuery(
                    date=date,
                    from_airport=from_airport,
                    to_airport=to_airport,
                    max_stops=max_stops,
                )
            ],
            seat=seat,
            trip=trip,
            passengers=Passengers(adults=max(1, int(people_count))),
            language=language,
            currency=currency,
        )

        result = get_flights(query)
    except Exception as e:
        logger.error(f"[flight_tools] Google Flights 查询失败: {e}")
        return []

    if not result:
        logger.info(f"[flight_tools] query-flights({from_airport}→{to_airport}, {date}) → 0 条")
        return []

    flights_list = []
    for flights_obj in result:
        segments = []
        for sf in flights_obj.flights:
            # 安全提取 HH:MM 时间字符串
            dep_time = fmt_time(sf.departure.time)
            arr_time = fmt_time(sf.arrival.time)

            segments.append({
                "from_airport": {
                    "code": sf.from_airport.code,
                    "name": sf.from_airport.name,
                },
                "to_airport": {
                    "code": sf.to_airport.code,
                    "name": sf.to_airport.name,
                },
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "duration_minutes": sf.duration,
                "plane_type": sf.plane_type,
            })

        price = flights_obj.price  # 总价（可能为 0 如果无票）

        pp = max(1, int(people_count))
        flights_list.append({
            "price": price,
            "price_per_person": round(price / pp, 2) if price else 0,
            "airlines": flights_obj.airlines,
            "flights": segments,
            "carbon_emission": flights_obj.carbon.emission if flights_obj.carbon else 0,
            "trip_type": flights_obj.type,
        })

    logger.info(f"[flight_tools] query-flights({from_airport}→{to_airport}, {date}) → {len(flights_list)} 条")
    if flights_list:
        first = flights_list[0]
        logger.debug(f"[flight_tools]   第一条: price={first['price']}, airlines={first['airlines']}, "
                     f"segments={len(first['flights'])}")

    return flights_list


# ═══════════════════════════════════════════════════════════════
# 航班规划层（从 intercity_tools 迁入）
# ═══════════════════════════════════════════════════════════════

def select_best_google_flight(
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
        time_score = time_reasonableness_score(dep_time, arr_time, direction)
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
    print(f"[flight_tools] Google Flights 筛选: best_score={best_score}, "
          f"price={best.get('price')}, segments={len(best.get('flights', []))}")
    return best


def normalize_google_flight(
    flight: dict[str, Any],
    origin_city: str,
    destination_city: str,
    date: str,
    direction: Literal["outbound", "return"],
    people_count: int,
) -> dict[str, Any]:
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


def plan_flight(
    departure_city: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    people_count: int,
) -> dict[str, Any]:
    """
    生成航班往返方案（基于 Google Flights 实时数据）。

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

    print(f"[flight_tools] plan_flight(Google Flights): {origin_name} → {dest_name}")

    origin_code = resolve_airport_code(origin_name)
    dest_code = resolve_airport_code(dest_name)
    print(f"[flight_tools] 机场代码: {origin_name}→{origin_code}, {dest_name}→{dest_code}")

    if not origin_code or not dest_code:
        return create_estimated_fallback(
            departure_city=origin_name,
            destination_city=dest_name,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode="flight",
            warnings=[f"机场代码解析失败: {origin_name}={origin_code}, {dest_name}={dest_code}"],
        )

    try:
        outbound_flights = query_flights(origin_code, dest_code, start_date, people_count=people)
        return_flights = query_flights(dest_code, origin_code, end_date, people_count=people)

        outbound_best = select_best_google_flight(outbound_flights, "outbound")
        return_best = select_best_google_flight(return_flights, "return")
        print(f"[flight_tools] 去程选中: {outbound_best is not None}, 返程选中: {return_best is not None}")

        if not outbound_best or not return_best:
            raise RuntimeError("Google Flights 未返回可用航班。")

        outbound = normalize_google_flight(
            outbound_best, origin_name, dest_name, start_date, "outbound", people,
        )
        return_trip = normalize_google_flight(
            return_best, dest_name, origin_name, end_date, "return", people,
        )
        return build_plan_from_segments("flight", outbound, return_trip, [])
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return create_estimated_fallback(
            departure_city=origin_name,
            destination_city=dest_name,
            start_date=start_date,
            end_date=end_date,
            people_count=people,
            mode="flight",
            warnings=[f"Google Flights 未返回可用航班，已使用估算。原因：{exc}"],
        )
