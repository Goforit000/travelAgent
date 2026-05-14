"""
高德地图工具 — 封装高德地图 Web API，同时提供底层函数和 @tool 装饰的 Agent 工具

底层函数（供代码直接调用）：
- search_pois()  : 关键词搜索 POI
- get_weather()  : 查询城市天气预报

@tool 装饰的工具（供 Agent ReAct 循环中的 LLM Tool Calling 调用）：
- search_attractions_tool : 搜索景点
- search_hotels_tool      : 搜索酒店
- query_weather_tool      : 查询天气
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.tools import tool

from app.utils.config import settings

# 高德地图 Web API 的基础地址
AMAP_BASE_URL = "https://restapi.amap.com/v3"

def amap_get(url: str, params: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    """
    统一调用高德 Web 服务接口。
    参数：
        url: 高德接口地址。
        params: 请求参数，函数会自动补充 key 和 output=json。
        timeout: 请求超时时间，单位秒。
    返回：
        高德返回的 JSON 字典。
    异常：
        当 HTTP 请求失败或高德返回 status != 1 时抛出异常。
    """
    request_params = {
        "key": settings.amap_api_key,
        "output": "json",
        **params,
    }
    response = httpx.get(url, params=request_params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "1":
        info = data.get("info") or data.get("infocode") or "unknown error"
        raise RuntimeError(f"高德接口调用失败: {info}")
    return data

def geocode_city(city: str) -> tuple[float, float] | None:
    """
    将城市名称解析为高德经纬度。
    参数：
        city: 城市名称，例如“沈阳”“北京”。
    返回：
        解析成功时返回 (longitude, latitude)，失败时返回 None。
    """
    try:
        data = amap_get(
            f"{AMAP_BASE_URL}/geocode/geo",
            params={"address": city},
            timeout=8,
        )
        geocodes = data.get("geocodes", [])
        if not geocodes:
            return None

        location = geocodes[0].get("location", "")
        if "," not in location:
            return None

        lng, lat = location.split(",", 1)
        return float(lng), float(lat)
    except Exception as exc:
        print(f"高德城市地理编码失败: city={city}, error={exc}")
        return None


def format_amap_point(point: tuple[float, float]) -> str:
    """
    将经纬度元组格式化为高德路径规划接口需要的字符串。
    参数：
        point: (longitude, latitude) 经纬度元组。
    返回：
        形如 "116.397428,39.90923" 的字符串。
    """
    return f"{point[0]:.6f},{point[1]:.6f}"


# ============================================================
# 底层 HTTP 函数（供代码直接调用，不抛异常，失败返回空列表）
# ============================================================

def search_pois(
    keywords: str,
    city: str,
    citylimit: bool = True,
    offset: int = 5,
) -> list[dict]:
    """
    搜索 POI（兴趣点）— 通用方法，搜景点、酒店、餐厅都用它

    Args:
        keywords: 搜索关键词，如"历史文化"、"酒店"、"美食"
        city: 城市名，如"北京"
        citylimit: 是否限制在城市范围内搜索
        offset: 返回数量上限（最多 20）

    Returns:
        POI 列表，每个 POI 是一个 dict，包含 name/address/location 等字段
        调用失败返回空列表（不抛异常，让上层决定如何处理）
    """
    try:
        data = amap_get(
            f"{AMAP_BASE_URL}/place/text",
            params={
                "keywords": keywords,
                "city": city,
                "citylimit": str(citylimit).lower(),
                "offset": min(offset, 20),
            },
            timeout=10,
        )

        pois = []
        for item in data.get("pois", []):
            loc_str = item.get("location", "")
            lng, lat = 0.0, 0.0
            if loc_str and "," in loc_str:
                parts = loc_str.split(",")
                lng, lat = float(parts[0]), float(parts[1])

            pois.append({
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "address": item.get("address", ""),
                "longitude": lng,
                "latitude": lat,
                "tel": item.get("tel", ""),
                "rating": item.get("biz_ext", {}).get("rating", 0),
            })

        return pois

    except Exception as e:
        print(f"❌ 高德POI搜索异常: {e}")
        return []

def get_weather(city: str) -> list[dict]:
    """
    查询城市天气预报

    Args:
        city: 城市名或城市编码，如 "北京"

    Returns:
        天气预报列表（最多4天），每条包含 date/day_weather/night_weather/temps 等
        如果失败返回空列表
    """
    try:
        data = amap_get(
            f"{AMAP_BASE_URL}/weather/weatherInfo",
            params={
                "key": settings.amap_api_key,
                "city": city,
                "extensions": "all",  # "all" 返回预报，"base" 只返回实况
                "output": "json",
            },
            timeout=10,
        )
        if data.get("status") != "1":
            print(f"⚠️ 高德天气查询失败: {data.get('info', 'unknown error')}")
            return []

        # 从 forecasts 里提取每日天气
        forecasts = data.get("forecasts", [])
        if not forecasts:
            return []

        casts = forecasts[0].get("casts", [])
        weather_list = []
        for cast in casts:
            weather_list.append({
                "date": cast.get("date", ""),
                "day_weather": cast.get("dayweather", ""),
                "night_weather": cast.get("nightweather", ""),
                "day_temp": cast.get("daytemp", "0"),
                "night_temp": cast.get("nighttemp", "0"),
                "wind_direction": cast.get("daywind", ""),
                "wind_power": cast.get("daypower", ""),
            })

        return weather_list

    except Exception as e:
        print(f"❌ 高德天气查询异常: {e}")
        return []

# ============================================================
# @tool 装饰的 Agent 工具（供 LLM Tool Calling 使用）
# ============================================================

@tool
def search_attractions_tool(
    keywords: str,
    city: str,
    offset: int = 10,
) -> str:
    """
    在高德地图中搜索景点信息。

    根据给定的关键词和城市搜索景点，返回景点名称、地址、经纬度、评分等数据。
    当需要获取某个城市的真实景点数据时调用此工具。

    Args:
        keywords: 搜索关键词，例如 "历史文化"、"自然风光"、"博物馆"、"寺庙"。
                  如果用户有多个偏好，建议分多次调用，每次使用不同的关键词。
        city: 目标城市名称，例如 "北京"、"上海"、"杭州"。
        offset: 返回的景点数量上限，默认为 10，最大 20。

    Returns:
        JSON 字符串，包含景点列表。每个景点包含：
        - name: 景点名称
        - address: 详细地址
        - longitude: 经度
        - latitude: 纬度
        - type: 景点类别
        - rating: 评分
        如果没有找到结果或调用失败，返回空列表的 JSON。
    """
    results = search_pois(keywords=keywords, city=city, offset=offset)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def search_hotels_tool(
    keywords: str,
    city: str,
    offset: int = 10,
) -> str:
    """
    在高德地图中搜索酒店信息。

    根据给定的住宿类型关键词和城市搜索酒店，返回酒店名称、地址、经纬度等数据。
    当需要获取某个城市的酒店数据时调用此工具。

    Args:
        keywords: 酒店类型关键词，例如 "经济型酒店"、"舒适型酒店"、"豪华酒店"、"民宿"。
        city: 目标城市名称，例如 "北京"、"上海"。
        offset: 返回的酒店数量上限，默认为 10，最大 20。

    Returns:
        JSON 字符串，包含酒店列表。每个酒店包含：
        - name: 酒店名称
        - address: 详细地址
        - longitude: 经度
        - latitude: 纬度
        - type: 酒店类型
        - rating: 评分
        如果没有找到结果或调用失败，返回空列表的 JSON。
    """
    results = search_pois(keywords=keywords, city=city, offset=offset)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def query_weather_tool(
    city: str,
) -> str:
    """
    查询指定城市的天气预报。

    获取未来几天的天气信息，包括白天/夜间天气状况、温度、风向风力等。
    当需要获取旅行目的地天气信息时调用此工具。

    Args:
        city: 目标城市名称，例如 "北京"、"上海"。
              支持中文城市名或行政区划代码（如 "110000" 代表北京）。

    Returns:
        JSON 字符串，包含天气预报列表。每天包含：
        - date: 日期 (YYYY-MM-DD)
        - day_weather: 白天天气（如"晴"、"多云"、"小雨"）
        - night_weather: 夜间天气
        - day_temp: 白天温度（字符串，可能带单位如"25"或"25°C"）
        - night_temp: 夜间温度
        - wind_direction: 风向（如"南风"、"北风"）
        - wind_power: 风力等级（如"1-3级"）
        如果查询失败，返回空列表的 JSON。
    """
    results = get_weather(city=city)
    return json.dumps(results, ensure_ascii=False, indent=2)
