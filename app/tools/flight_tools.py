"""
Google Flights 航班查询工具 — 封装 fast_flights 库。

提供：
    search_airports(city)     → list[dict]  城市名→机场列表
    resolve_airport_code(city) → str | None  城市→首选机场三字码
    query_flights(from_code, to_code, date, **) → list[dict]  查航班

数据来源：Google Flights（通过 fast_flights 抓取），无需 API Key。
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 城市中英文映射 + 机场数据库
# ═══════════════════════════════════════════════════════════════

# 常用城市中文名 → 首选机场三字码
_CITY_CN_TO_CODE: dict[str, str] = {
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
_CITY_EN_TO_CODE: dict[str, str] = {
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
_CITY_EN_TO_CN: dict[str, str] = {
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

_airports_loaded = False
_airports_by_code: dict[str, dict] = {}
_airports_list: list[dict] = []


def _load_airports() -> None:
    global _airports_loaded, _airports_by_code, _airports_list
    if _airports_loaded:
        return

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airports.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"[flight_tools] airports.csv 不存在: {csv_path}")
        _airports_loaded = True
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
            _airports_by_code[code] = info
            _airports_list.append(info)

    _airports_loaded = True
    logger.info(f"[flight_tools] 已加载 {len(_airports_by_code)} 个机场")


def _match_airport_name(name_en: str, keyword_lower: str) -> bool:
    """检查机场英文名是否匹配关键词（英/中）。"""
    name_lower = name_en.lower()
    if keyword_lower in name_lower:
        return True
    # 英文名 → 中文名 → 匹配
    for en_city, cn_city in _CITY_EN_TO_CN.items():
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
    _load_airports()

    keyword_lower = keyword.strip().lower()
    results: list[dict] = []

    # 先直接匹配三字码
    if keyword_lower.upper() in _airports_by_code:
        results.append(_airports_by_code[keyword_lower.upper()])

    # 匹配机场英文名
    for info in _airports_list:
        if _match_airport_name(info["name"], keyword_lower):
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


def _fmt_time(t) -> str:
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
        _load_airports()
        if city_clean in _airports_by_code:
            return city_clean

    # 去掉"市"后缀
    if city_clean.endswith("市"):
        city_clean = city_clean[:-1]

    # 查中文映射表
    if city_clean in _CITY_CN_TO_CODE:
        code = _CITY_CN_TO_CODE[city_clean]
        logger.debug(f"[flight_tools] 机场匹配(映射表): {city} → {code}")
        return code

    # 查英文映射表
    city_lower = city_clean.lower()
    if city_lower in _CITY_EN_TO_CODE:
        code = _CITY_EN_TO_CODE[city_lower]
        logger.debug(f"[flight_tools] 机场匹配(英→码): {city} → {code}")
        return code

    # 英文名反查中文再查映射
    if city_lower in _CITY_EN_TO_CN:
        cn = _CITY_EN_TO_CN[city_lower]
        if cn in _CITY_CN_TO_CODE:
            code = _CITY_CN_TO_CODE[cn]
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
    _load_airports()

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
            dep_time = _fmt_time(sf.departure.time)
            arr_time = _fmt_time(sf.arrival.time)

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
