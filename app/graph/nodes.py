"""
LangGraph 节点函数 — 将 Agent 包装为 StateGraph 可调用的节点

Planner-Centric Pipeline 架构：
  initialize → data_collection(POI+Weather+Hotel+Transport并行) → planner → budget → (条件) → finalize
路由由 workflow_router 执行
"""

import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.graph.state import TripState
from app.tools.unsplash import get_photo_url

# ——— Agent 实例（模块级单例）———
from app.agents.poi_agent import POIAgent
from app.agents.weather_agent import WeatherAgent
from app.agents.hotel_agent import HotelAgent
from app.agents.transport_agent import TransportAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.budget_agent import BudgetAgent

_poi_agent = POIAgent()
_weather_agent = WeatherAgent()
_hotel_agent = HotelAgent()
_transport_agent = TransportAgent()
_planner_agent = PlannerAgent()
_budget_agent = BudgetAgent()


# ============================================================
# 节点 0：初始化
# ============================================================

def initialize_node(state: TripState) -> dict:
    """工作流入口节点 — 设置初始状态"""
    request = state.get("request")
    if request is None:
        return {
            "error": "缺少 request 参数",
            "phase": "done",
        }

    city = getattr(request, "city", "")
    travel_days = getattr(request, "travel_days", 0)

    print(f"\n{'=' * 60}")
    print(f"🚀 初始化旅行规划: {city} {travel_days} 天")
    print(f"{'=' * 60}")

    return {
        "phase": "collect",
        "retry_count": 0,
        "iteration_count": 0,
        "max_iterations": 10,
        "revision_round": 0,
        "error": "",
        "agent_outputs": {},
        "raw_attractions": state.get("raw_attractions", []),
        "raw_weather": state.get("raw_weather", []),
        "raw_hotels": state.get("raw_hotels", []),
        "raw_intercity_transport": state.get("raw_intercity_transport", {}),
        "raw_plan_text": state.get("raw_plan_text", ""),
        "trip_plan": state.get("trip_plan"),
        "attraction_photos": state.get("attraction_photos", {}),
    }


# ============================================================
# 节点 1：数据收集（并行 POI + Weather + Hotel + Transport）
# ============================================================

def data_collection_node(state: TripState) -> dict:
    """
    数据收集节点 — 并行运行 POI、Weather、Hotel、Transport 四个 Agent

    - 并行执行，总耗时 = max(单 Agent 耗时)
    - 初次失败或返回空 → 自动重试 1 次（调用单 Agent 节点函数）
    - 重试仍失败/空 → 交由 Planner LLM 常识补全
    """
    print(f"\n{'=' * 60}")
    print(f"📦 [data_collection_node] 并行数据收集开始 (POI + Weather + Hotel + Transport)")
    print(f"{'=' * 60}")

    results: dict = {
        "raw_attractions": state.get("raw_attractions", []),
        "raw_weather": state.get("raw_weather", []),
        "raw_hotels": state.get("raw_hotels", []),
        "raw_intercity_transport": state.get("raw_intercity_transport", {}),
        "agent_outputs": dict(state.get("agent_outputs", {})),
        "error": "",
    }

    # —— 第一轮：并行执行 ——
    def run_poi():
        try:
            print(f"  📍 POI Agent 开始...")
            return _poi_agent.run(state)
        except Exception as e:
            print(f"  ❌ POI Agent 异常: {e}")
            return {"error": f"POI Agent: {e}", "agent_outputs": {"poi_agent": f"失败: {e}"}}

    def run_weather():
        try:
            print(f"  🌤️ Weather Agent 开始...")
            return _weather_agent.run(state)
        except Exception as e:
            print(f"  ❌ Weather Agent 异常: {e}")
            return {"error": f"Weather Agent: {e}", "agent_outputs": {"weather_agent": f"失败: {e}"}}

    def run_hotel():
        try:
            print(f"  🏨 Hotel Agent 开始...")
            return _hotel_agent.run(state)
        except Exception as e:
            print(f"  ❌ Hotel Agent 异常: {e}")
            return {"error": f"Hotel Agent: {e}", "agent_outputs": {"hotel_agent": f"失败: {e}"}}

    def run_transport():
        try:
            print(f"  🚄 Transport Agent 开始...")
            return _transport_agent.run(state)
        except Exception as e:
            print(f"  ❌ Transport Agent 异常: {e}")
            return {"error": f"Transport Agent: {e}", "agent_outputs": {"transport_agent": f"失败: {e}"}}

    round1_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_poi): "poi",
            executor.submit(run_weather): "weather",
            executor.submit(run_hotel): "hotel",
            executor.submit(run_transport): "transport",
        }
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                round1_results[agent_name] = future.result()
            except Exception as e:
                print(f"  ❌ {agent_name} 线程异常: {e}")
                round1_results[agent_name] = {"error": f"{agent_name}: {e}"}

    # 合并第一轮结果 + 记录需要重试的 Agent
    retry_agents: list[str] = []
    for agent_name in ["poi", "weather", "hotel", "transport"]:
        agent_result = round1_results.get(agent_name, {})
        if "agent_outputs" in agent_result:
            results["agent_outputs"].update(agent_result["agent_outputs"])
        key_map = {
            "poi": "raw_attractions",
            "weather": "raw_weather",
            "hotel": "raw_hotels",
            "transport": "raw_intercity_transport",
        }
        data_key = key_map[agent_name]
        if data_key in agent_result:
            results[data_key] = agent_result[data_key]
            if isinstance(agent_result[data_key], dict):
                count = 1 if agent_result[data_key] else 0
            else:
                count = len(agent_result[data_key])
            print(f"  ✅ 第1轮 {agent_name} 完成: {count} 条")
        else:
            results[data_key] = {} if agent_name == "transport" else []
            print(f"  ⚠️ 第1轮 {agent_name} 失败或无数据")

        # 判断是否需要重试：异常 或 返回空数据
        has_error = bool(agent_result.get("error"))
        has_data = bool(agent_result.get(data_key))
        if has_error or not has_data:
            retry_agents.append(agent_name)

    # —— 第二轮：逐一重试失败的 Agent ——
    retry_map = {
        "poi": (poi_node, "raw_attractions"),
        "weather": (weather_node, "raw_weather"),
        "hotel": (hotel_node, "raw_hotels"),
        "transport": (transport_node, "raw_intercity_transport"),
    }
    errors: list[str] = []
    for agent_name in retry_agents:
        print(f"  🔄 重试 {agent_name}...")
        node_func, data_key = retry_map[agent_name]
        try:
            retry_result = node_func(state)
            if "agent_outputs" in retry_result:
                results["agent_outputs"].update(retry_result["agent_outputs"])
            if data_key in retry_result and retry_result[data_key]:
                results[data_key] = retry_result[data_key]
                print(f"  ✅ 重试 {agent_name} 成功: {len(retry_result[data_key])} 条")
            else:
                print(f"  ⚠️ 重试 {agent_name} 仍无数据，交由 Planner 常识补全")
                if retry_result.get("error"):
                    errors.append(retry_result["error"])
        except Exception as e:
            print(f"  ❌ 重试 {agent_name} 异常: {e}，交由 Planner 常识补全")
            errors.append(f"重试{agent_name}: {e}")

    if errors:
        results["error"] = "; ".join(errors)
        print(f"  ⚠️ 数据收集错误: {results['error']}")

    total_attractions = len(results["raw_attractions"])
    total_hotels = len(results["raw_hotels"])
    total_weather = len(results["raw_weather"])
    total_transport = 1 if results.get("raw_intercity_transport") else 0

    print(f"  📊 收集汇总: {total_attractions} 景点, {total_weather} 天天气, {total_hotels} 酒店, {total_transport} 个跨城交通方案")
    print(f"{'=' * 60}\n")

    return results


# ============================================================
# 节点 3-4：规划 + 预算（链式执行）
# ============================================================

def planner_node(state: TripState) -> dict:
    """
    Planner Agent 节点 — 生成行程计划

    注意：此节点完成后不返回 Supervisor，由 workflow 层直接链到 budget_node。
    """
    print(f"\n📋 [planner_node] 激活 Planner Agent")
    try:
        result = _planner_agent.run(state)
        result["error"] = ""
        result["retry_count"] = 0
        return result
    except Exception as e:
        error_msg = f"Planner Agent 执行异常: {str(e)}"
        print(f"❌ [planner_node] {error_msg}")
        return {
            "error": error_msg,
            "agent_outputs": {"planner_agent": f"执行失败: {str(e)}"},
        }


def budget_node(state: TripState) -> dict:
    """
    Budget Agent 节点 — 计算预算

    此节点是 planner → budget 链的最后一环，完成后由 workflow_router 决定下一步。
    若检测到超限信号 (OVERSHOOT|)，递增 revision_round。
    """
    print(f"\n💰 [budget_node] 激活 Budget Agent")
    try:
        result = _budget_agent.run(state)
        result["error"] = ""
        result["retry_count"] = 0

        # 检测超限信号 → 递增修正轮数
        agent_outputs = result.get("agent_outputs", {})
        budget_value = agent_outputs.get("budget_agent", "")
        if isinstance(budget_value, str) and budget_value.startswith("OVERSHOOT|"):
            current_round = state.get("revision_round", 0)
            result["revision_round"] = current_round + 1
            print(f"  [budget_node] ⚠️ 超限! revision_round {current_round} → {current_round + 1}")

        return result
    except Exception as e:
        error_msg = f"Budget Agent 执行异常: {str(e)}"
        print(f"❌ [budget_node] {error_msg}")
        return {
            "error": error_msg,
            "agent_outputs": {"budget_agent": f"执行失败: {str(e)}"},
        }


# ============================================================
# 节点 5：保留的单 Agent 节点（用于错误重试时单独重跑）
# ============================================================

def poi_node(state: TripState) -> dict:
    """POI Agent 节点（用于错误重试时单独调用）"""
    print(f"\n📍 [poi_node] 激活 POI Agent (重试)")
    try:
        result = _poi_agent.run(state)
        result["error"] = ""
        result["retry_count"] = 0
        return result
    except Exception as e:
        return {
            "error": f"POI Agent: {str(e)}",
            "agent_outputs": {"poi_agent": f"重试失败: {str(e)}"},
        }


def weather_node(state: TripState) -> dict:
    """Weather Agent 节点（用于错误重试时单独调用）"""
    print(f"\n🌤️ [weather_node] 激活 Weather Agent (重试)")
    try:
        result = _weather_agent.run(state)
        result["error"] = ""
        result["retry_count"] = 0
        return result
    except Exception as e:
        return {
            "error": f"Weather Agent: {str(e)}",
            "agent_outputs": {"weather_agent": f"重试失败: {str(e)}"},
        }


def hotel_node(state: TripState) -> dict:
    """Hotel Agent 节点（用于错误重试时单独调用）"""
    print(f"\n🏨 [hotel_node] 激活 Hotel Agent (重试)")
    try:
        result = _hotel_agent.run(state)
        result["error"] = ""
        result["retry_count"] = 0
        return result
    except Exception as e:
        return {
            "error": f"Hotel Agent: {str(e)}",
            "agent_outputs": {"hotel_agent": f"重试失败: {str(e)}"},
        }


def transport_node(state: TripState) -> dict:
    """Transport Agent 节点（用于错误重试时单独调用）"""
    print(f"\n🚄 [transport_node] 激活 Transport Agent (重试)")
    try:
        result = _transport_agent.run(state)
        result["error"] = ""
        result["retry_count"] = 0
        return result
    except Exception as e:
        return {
            "raw_intercity_transport": {},
            "error": f"Transport Agent: {str(e)}",
            "agent_outputs": {"transport_agent": f"重试失败: {str(e)}"},
        }


# ============================================================
# 节点 6：Finalize — 最终处理
# ============================================================

def finalize_node(state: TripState) -> dict:
    """
    最终处理节点 — 解析行程 → 配图 → 全局兜底

    这是用户请求的最后一道保障：无论前面多少个 Agent 失败，
    这个节点必须保证返回一个可用的 TripPlan。
    """
    from app.schemas.models import TripPlan, DayPlan, Attraction, Meal, Location

    print(f"\n{'=' * 60}")
    print(f"🏁 [finalize_node] 最终处理开始")
    print(f"{'=' * 60}")

    trip_plan = state.get("trip_plan")
    raw_plan_text = state.get("raw_plan_text", "")
    error = state.get("error", "")
    attraction_photos = state.get("attraction_photos", {})

    # ——— 步骤 1：解析行程 ———
    if trip_plan is None and raw_plan_text:
        try:
            json_str = _extract_json(raw_plan_text)
            data = json.loads(json_str)
            trip_plan = TripPlan(**data)
            print(f"  ✅ 行程解析成功: {len(trip_plan.days)} 天")
        except Exception as e:
            print(f"  ⚠️ 行程解析失败: {e}，使用备用计划")
            trip_plan = None

    # ——— 步骤 2：兜底 ———
    if trip_plan is None:
        request = state.get("request")
        if request is not None:
            trip_plan = _create_fallback_plan(request, state)
            error = (error + "; " if error else "") + "使用备用计划"
            print(f"  🔄 已生成备用计划")
        else:
            error = "无法生成旅行计划：缺少 request"

    # ——— 步骤 3：配图 ———
    if trip_plan is not None:
        request = state.get("request")
        if request is not None and hasattr(trip_plan, "people_count"):
            try:
                trip_plan.people_count = max(1, int(getattr(request, "people_count", 1) or 1))
            except Exception:
                trip_plan.people_count = 1

        new_photos: dict[str, str] = {}
        for day in trip_plan.days:
            for attraction in day.attractions:
                if attraction.image_url:
                    continue
                name = attraction.name
                query = f"{name} {trip_plan.city} China landmark"
                print(f"  📷 搜索图片: {name}")
                url = get_photo_url(query)
                if url:
                    attraction.image_url = url
                    new_photos[name] = url
                    print(f"     ✅ 找到图片")
                else:
                    print(f"     ⚠️ 未找到图片")

        if new_photos:
            attraction_photos = {**attraction_photos, **new_photos}

    print(f"{'=' * 60}")
    print(f"🏁 [finalize_node] 完成")
    print(f"{'=' * 60}\n")

    return {
        "trip_plan": trip_plan,
        "attraction_photos": attraction_photos,
        "error": error,
        "phase": "done",
    }


# ============================================================
# 辅助函数
# ============================================================

def _extract_json(text: str) -> str:
    """从文本中提取 JSON 字符串 — 委托给共享工具函数"""
    from app.utils.json_utils import extract_json
    result = extract_json(text)
    if result is None:
        raise ValueError("文本中未找到 JSON 数据")
    return result


def _create_fallback_plan(request, state) -> "TripPlan":
    """
    备用计划生成 — 优先使用 Agent 已收集的真实数据

    优先链：
    raw_attractions 有数据 → 用真实 POI
    raw_hotels 有数据 → 填充真实酒店
    raw_weather 有数据 → 填充真实天气
    都没有 → 硬编码兜底
    """
    from app.schemas.models import (
        TripPlan,
        DayPlan,
        Attraction,
        Meal,
        Location,
        Hotel,
        WeatherInfo,
        IntercityTransportPlan,
        IndoorBackupAttraction,
    )

    start = datetime.strptime(request.start_date, "%Y-%m-%d")
    raw_attractions = state.get("raw_attractions", [])
    raw_hotels = state.get("raw_hotels", [])
    raw_weather = state.get("raw_weather", [])
    raw_intercity_transport = state.get("raw_intercity_transport", {})

    # 构建真实景点列表
    real_attractions: list[Attraction] = []
    for attr in raw_attractions:
        lng = attr.get("longitude", 0)
        lat = attr.get("latitude", 0)
        real_attractions.append(Attraction(
            name=attr.get("name", f"{request.city}景点"),
            address=attr.get("address", f"{request.city}市"),
            location=Location(
                longitude=lng if lng else 116.40,
                latitude=lat if lat else 39.90,
            ),
            visit_duration=120,
            description=attr.get("description", f"{request.city}推荐景点"),
            category=attr.get("type", "景点"),
            ticket_price=attr.get("ticket_price", 0),
        ))

    # 构建真实酒店
    real_hotel: Hotel | None = None
    if raw_hotels:
        h = raw_hotels[0]
        real_hotel = Hotel(
            name=h.get("name", "推荐酒店"),
            address=h.get("address", ""),
            location=Location(
                longitude=h.get("longitude", 0) or 116.40,
                latitude=h.get("latitude", 0) or 39.90,
            ),
            price_range="200-500元",
            rating=h.get("rating", 0),
            distance="距市中心2公里",
            type=h.get("type", "酒店"),
            estimated_cost=300,
        )

    # 构建真实天气
    weather_info: list[WeatherInfo] = []
    for w in raw_weather[:request.travel_days]:
        weather_info.append(WeatherInfo(
            date=w.get("date", ""),
            day_weather=w.get("day_weather", ""),
            night_weather=w.get("night_weather", ""),
            day_temp=w.get("day_temp", "0"),
            night_temp=w.get("night_temp", "0"),
            wind_direction=w.get("wind_direction", ""),
            wind_power=w.get("wind_power", ""),
        ))

    # 如果没有真实景点，回退到硬编码
    if not real_attractions:
        for j in range(3):
            real_attractions.append(Attraction(
                name=f"{request.city}热门景点{j + 1}",
                address=f"{request.city}市",
                location=Location(longitude=116.40 + j * 0.02, latitude=39.90 + j * 0.02),
                visit_duration=120,
                description=f"{request.city}的推荐游览地点",
            ))

    # 按天分配景点
    days = []
    attr_per_day = max(2, min(3, len(real_attractions) // request.travel_days))
    attr_idx = 0

    for i in range(request.travel_days):
        current = start + timedelta(days=i)
        day_attractions = []
        for _ in range(attr_per_day):
            if attr_idx < len(real_attractions):
                day_attractions.append(real_attractions[attr_idx])
                attr_idx += 1
            else:
                day_attractions.append(Attraction(
                    name=f"{request.city}其他景点",
                    address=f"{request.city}市",
                    location=Location(longitude=116.40, latitude=39.90),
                    visit_duration=120,
                    description="自由探索",
                ))

        days.append(DayPlan(
            date=current.strftime("%Y-%m-%d"),
            day_index=i,
            description=f"第{i + 1}天：探索{request.city}",
            transportation=request.transportation,
            accommodation=request.accommodation,
            hotel=real_hotel if real_hotel and i == 0 else None,
            attractions=day_attractions,
            meals=[
                Meal(type="breakfast", name="当地早餐", estimated_cost=30),
                Meal(type="lunch", name="当地午餐", estimated_cost=50),
                Meal(type="dinner", name="当地晚餐", estimated_cost=80),
            ],
        ))

    real_poi_count = len(raw_attractions)
    fallback_desc = (
        f"这是{request.city}{request.travel_days}日游的行程。"
        f"由于智能规划过程遇到问题，此行程为备用方案（复用了已收集的 {real_poi_count} 个真实景点）。"
        f"建议提前查看各景点开放时间，并根据实际天气情况调整出行计划。"
    )

    intercity_transport = None
    if isinstance(raw_intercity_transport, dict) and raw_intercity_transport.get("outbound") and raw_intercity_transport.get("return_trip"):
        try:
            intercity_transport = IntercityTransportPlan(**raw_intercity_transport)
        except Exception:
            intercity_transport = None

    used_names = {attr.name for day in days for attr in day.attractions}
    indoor_candidates = [
        {
            "name": f"{request.city}博物馆",
            "address": f"{request.city}市中心",
            "description": "适合了解当地历史文化的室内场馆。",
            "category": "博物馆",
            "reason": "室内参观受天气影响小，适合作为临时替代安排。",
            "estimated_duration": 90,
            "ticket_price": 0,
        },
        {
            "name": f"{request.city}美术馆",
            "address": f"{request.city}市区",
            "description": "以艺术展览和文化展示为主的室内空间。",
            "category": "美术馆",
            "reason": "动线集中、节奏轻松，适合雨天或高温天气调整。",
            "estimated_duration": 90,
            "ticket_price": 0,
        },
        {
            "name": f"{request.city}大型商业中心",
            "address": f"{request.city}核心商圈",
            "description": "集合购物、餐饮和休闲的综合室内场所。",
            "category": "商场",
            "reason": "餐饮和休息设施完善，适合作为全天候备用地点。",
            "estimated_duration": 120,
            "ticket_price": 0,
        },
    ]
    indoor_backup_attractions = [
        IndoorBackupAttraction(**item)
        for item in indoor_candidates
        if item["name"] not in used_names
    ][:3]

    return TripPlan(
        city=request.city,
        departure_city=getattr(request, "departure_city", None),
        start_date=request.start_date,
        end_date=request.end_date,
        people_count=max(1, int(getattr(request, "people_count", 1) or 1)),
        indoor_backup_attractions=indoor_backup_attractions,
        intercity_transport_mode=getattr(request, "intercity_transport_mode", None),
        intercity_transport=intercity_transport,
        days=days,
        weather_info=weather_info,
        overall_suggestions=fallback_desc,
    )
