"""
Planner 数据准备工具 — 空间聚类、酒店匹配、文本格式化

从 planner_agent.py 中抽离的纯函数，不依赖 LLM 或 Agent 框架。
"""

import json
from typing import Optional
from app.utils.transport_utils import haversine_km


# ============================================================
# 空间聚类
# ============================================================

def greedy_cluster(
    attractions: list[dict],
    target_clusters: int = 3,
    min_threshold_km: float = 3.0,
    max_threshold_km: float = 10.0,
    max_intra_cluster_km: float = 8.0,  #同一天游玩的景点之间最远有多远
) -> list[dict]:
    """
    贪心空间聚类：将景点按地理位置分组

    改进点（相比旧版）：
    1. 先按坐标排序，确保空间相邻的景点在列表中相邻，消除输入顺序依赖
    2. 使用固定种子坐标（首个景点）而非滚动质心，消除雪球漂移效应
    3. 新景点加入前检查与簇内所有点的最大距离，超过硬上限则拒绝合并
    4. target_clusters 作为软约束——若数据天然分散，允许超出
    """
    if not attractions:
        return [{"name": "默认区域", "items": []}]

    # 改进1: 按坐标排序，稳定相邻关系
    sorted_attrs = sorted(attractions, key=lambda a: (
        round(float(a.get("longitude", 0)), 3),
        round(float(a.get("latitude", 0)), 3),
    ))

    threshold = min_threshold_km
    clusters: list[dict] = []

    while threshold <= max_threshold_km:
        clusters = []
        for attr in sorted_attrs:
            lng = float(attr.get("longitude", 0))
            lat = float(attr.get("latitude", 0))
            if lng == 0 and lat == 0:
                continue

            # 找最近的 cluster（以簇内最近点为参照）
            best_idx = -1
            best_dist = float("inf")
            for idx, c in enumerate(clusters):
                for item in c["items"]:
                    ilng = float(item.get("longitude", 0))
                    ilat = float(item.get("latitude", 0))
                    d = haversine_km((lng, lat), (ilng, ilat))
                    if d < best_dist:
                        best_dist = d
                        best_idx = idx

            if best_idx >= 0 and best_dist < threshold:
                c = clusters[best_idx]
                # 改进3: 检查与簇内所有点的最大距离
                would_exceed = False
                for item in c["items"]:
                    ilng = float(item.get("longitude", 0))
                    ilat = float(item.get("latitude", 0))
                    if haversine_km((lng, lat), (ilng, ilat)) > max_intra_cluster_km:
                        would_exceed = True
                        break

                if would_exceed:
                    clusters.append({
                        "seed_lng": lng,
                        "seed_lat": lat,
                        "items": [attr],
                    })
                else:
                    c["items"].append(attr)
            else:
                # 改进2: 固定种子坐标，不滚动
                clusters.append({
                    "seed_lng": lng,
                    "seed_lat": lat,
                    "items": [attr],
                })

        # 改进4: target_clusters 作为软约束——如果阈值已达合理上限，不再强制合并
        if len(clusters) <= target_clusters:
            break
        threshold += 2.0

    # 命名：取 cluster 内评分最高的景点名 + "及周边"
    for c in clusters:
        items = c["items"]
        if items:
            best = max(items, key=lambda x: (
                float(x.get("rating", 0)) if isinstance(x.get("rating"), (int, float))
                else (float(str(x.get("rating", "0")))
                      if str(x.get("rating", "")).replace(".", "", 1).replace("-", "", 1).isdigit()
                      else 0)
            ))
            c["name"] = best.get("name", "未知") + "及周边"
        else:
            c["name"] = "无景点"
        del c["seed_lng"], c["seed_lat"]

    clusters.sort(key=lambda c: len(c["items"]), reverse=True)
    return clusters


# ============================================================
# 酒店匹配
# ============================================================

def find_nearest_hotels(cluster: dict, hotels: list[dict], top_n: int = 3) -> list[dict]:
    """找到离 cluster 内景点最近的 top_n 个酒店，按最近距离（主）+ 平均距离（辅）排序"""
    items = cluster["items"]
    if not items:
        return []

    scored: list[dict] = []
    for h in hotels:
        h_lng = float(h.get("longitude", 0))
        h_lat = float(h.get("latitude", 0))
        if h_lng == 0 and h_lat == 0:
            continue

        dists = []
        for item in items:
            ilng = float(item.get("longitude", 0))
            ilat = float(item.get("latitude", 0))
            if ilng == 0 and ilat == 0:
                continue
            dists.append(haversine_km((h_lng, h_lat), (ilng, ilat)))

        if not dists:
            continue

        entry = dict(h)
        entry["_min_distance_km"] = round(min(dists), 1)
        entry["_avg_distance_km"] = round(sum(dists) / len(dists), 1)
        entry["_max_distance_km"] = round(max(dists), 1)
        scored.append(entry)

    scored.sort(key=lambda x: (x["_min_distance_km"], x["_avg_distance_km"]))
    return scored[:top_n]


# ============================================================
# 数据格式化（供 LLM prompt 使用）
# ============================================================

def format_attractions(attractions: list[dict], travel_days: int, hotels: Optional[list[dict]] = None) -> str:
    """格式化景点为按区域分组的编号文本 — 每区附加最近的酒店，LLM 直接在本区内选择"""
    if not attractions:
        return "暂无景点信息，请根据常识为城市生成合适的景点。"

    clusters = greedy_cluster(attractions, target_clusters=max(1, travel_days))
    print(f"  [planner_agent] 贪心聚类: {len(attractions)} 个景点 → {len(clusters)} 个区域")

    # 为每个 cluster 挂载最近的酒店
    if hotels:
        for cluster in clusters:
            cluster["hotels"] = find_nearest_hotels(cluster, hotels, top_n=3)
        hotel_count = sum(len(c.get("hotels", [])) for c in clusters)
        print(f"  [planner_agent] 酒店挂载: {len(hotels)} 个酒店 → {hotel_count} 个分配")

    lines: list[str] = []
    global_index = 0
    for cluster in clusters:
        name = cluster["name"]
        items = cluster["items"]
        nearby_hotels = cluster.get("hotels", [])
        hotel_count_str = f" + {len(nearby_hotels)} 家附近酒店" if nearby_hotels else ""

        if len(clusters) > 1:
            lines.append(f"### {name}（{len(items)} 个景点{hotel_count_str}）")
        else:
            lines.append(f"### 主要景点（{len(items)} 个景点{hotel_count_str}）")

        for poi in items:
            global_index += 1
            poi_name = poi.get("name", "未知")
            lng = poi.get("longitude", 0)
            lat = poi.get("latitude", 0)
            rating = poi.get("rating", 0)
            address = poi.get("address", "")
            ticket = poi.get("ticket_price", 0)
            desc = poi.get("description", "")

            lines.append(f"  {global_index}. {poi_name} | 坐标({lng}, {lat})")
            if rating:
                lines.append(f"     评分: {rating}")
            if address:
                lines.append(f"     地址: {address}")
            if ticket:
                lines.append(f"     门票: {ticket}元")
            if desc:
                lines.append(f"     简介: {desc}")

        # 输出本区域附近酒店
        if nearby_hotels:
            lines.append(f"  附近酒店（供本区域选择）:")
            for h in nearby_hotels:
                hotel_name = h.get("name", "未知")
                h_lng = h.get("longitude", 0)
                h_lat = h.get("latitude", 0)
                h_rating = h.get("rating", 0)
                h_price = h.get("price_range", "")
                h_type = h.get("type", "")
                h_addr = h.get("address", "")
                max_dist = h.get("_max_distance_km", 0)
                avg_dist = h.get("_avg_distance_km", 0)
                min_dist = h.get("_min_distance_km", 0)

                lines.append(f"    - {hotel_name} | 坐标({h_lng}, {h_lat}) | 距景点 最近{min_dist:.1f}km / 平均{avg_dist:.1f}km / 最远{max_dist:.1f}km")
                if h_rating:
                    lines.append(f"      评分: {h_rating}")
                if h_price:
                    lines.append(f"      价格: {h_price}")
                if h_type:
                    lines.append(f"      类型: {h_type}")
                if h_addr:
                    lines.append(f"      地址: {h_addr}")
        lines.append("")

    return "\n".join(lines)


def format_weather(weather: list[dict], travel_days: int) -> str:
    if not weather:
        return "暂无天气信息"

    lines = []
    for w in weather[:travel_days]:
        date = w.get("date", "")
        day_w = w.get("day_weather", "")
        night_w = w.get("night_weather", "")
        day_t = str(w.get("day_temp", "0")).replace("°C", "").replace("℃", "").replace("°", "").strip()
        night_t = str(w.get("night_temp", "0")).replace("°C", "").replace("℃", "").replace("°", "").strip()
        wind_d = w.get("wind_direction", "")
        wind_p = w.get("wind_power", "")

        lines.append(f"{date}: {day_w} {day_t}°C / {night_w} {night_t}°C  {wind_d} {wind_p}")

    return "\n".join(lines)


def format_intercity_transport(intercity_transport: dict) -> str:
    """格式化跨城往返交通信息，明确标注首末日时间约束"""
    if not isinstance(intercity_transport, dict) or not intercity_transport:
        return "暂无跨城往返交通数据。如果用户未填写出发地，则 intercity_transport 可为 null。"

    if not intercity_transport.get("outbound") or not intercity_transport.get("return_trip"):
        summary = intercity_transport.get("summary", "暂无跨城往返交通数据")
        warnings = intercity_transport.get("warnings", [])
        warning_text = "；".join(warnings) if warnings else "无"
        return f"{summary}\n警告: {warning_text}\n如果无有效去程/返程对象，intercity_transport 可为 null。"

    # 提取关键时间约束
    outbound = intercity_transport.get("outbound", {})
    return_trip = intercity_transport.get("return_trip", {})
    outbound_arrival = outbound.get("arrival_time", "未知")
    return_departure = return_trip.get("departure_time", "未知")

    time_hint = (
        f"\n\n⚠️ 首末日时间约束（必须遵守）：\n"
        f"- 去程到达时间: {outbound_arrival}，第一天行程从到达之后开始，到达当天不要安排到达时间之前的景点。\n"
        f"- 返程出发时间: {return_departure}，最后一天行程必须在出发前结束，且最后一个景点须靠近出发站/机场。\n"
    )

    text = json.dumps(intercity_transport, ensure_ascii=False, indent=2)
    return (
        "以下是 Transport Agent 生成的结构化往返交通数据。"
        "请原样保留字段结构和用户选择的 mode，并写入最终 JSON 的 intercity_transport。"
        f"{time_hint}\n"
        f"{text}"
    )


def mode_label(mode: str) -> str:
    """跨城交通方式显示名"""
    labels = {
        "driving": "自驾",
        "high_speed_rail": "高铁",
        "flight": "飞机",
    }
    return labels.get(mode, mode or "未指定")
