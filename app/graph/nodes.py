"""
LangGraph 节点函数 — 将 Agent 包装为 StateGraph 可调用的节点

P1 优化变更：
- 新增 data_collection_node：并行运行 POI + Weather + Hotel，减少 4 次 Supervisor 调用
- planner_node → budget_node 链式执行（workflow 层处理），中间不回 Supervisor
- Supervisor 仅在 3 个阶段切换点介入：collect → plan → finalize

拓扑结构（优化后）：
  initialize → supervisor → (条件路由)
    ├─ "collect"  → data_collection_node → supervisor
    ├─ "plan"     → planner_node → budget_node → supervisor
    └─ "finalize" → finalize_node → END
"""

import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.graph.state import TripState
from app.tools.unsplash import get_photo_url

# ——— Agent 实例（模块级单例）———
from app.agents.supervisor import SupervisorAgent
from app.agents.poi_agent import POIAgent
from app.agents.weather_agent import WeatherAgent
from app.agents.hotel_agent import HotelAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.budget_agent import BudgetAgent

_supervisor = SupervisorAgent()
_poi_agent = POIAgent()
_weather_agent = WeatherAgent()
_hotel_agent = HotelAgent()
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
            "next_agent": "finalize",
        }

    city = getattr(request, "city", "")
    travel_days = getattr(request, "travel_days", 0)

    print(f"\n{'=' * 60}")
    print(f"🚀 初始化旅行规划: {city} {travel_days} 天")
    print(f"{'=' * 60}")

    return {
        "phase": "collect",
        "next_agent": "",
        "last_agent": "",
        "retry_count": 0,
        "iteration_count": 0,
        "max_iterations": 15,  # P1 优化：减少迭代上限（Supervisor 调用减少）
        "revision_round": 0,
        "error": "",
        "agent_outputs": {},
        "raw_attractions": state.get("raw_attractions", []),
        "raw_weather": state.get("raw_weather", []),
        "raw_hotels": state.get("raw_hotels", []),
        "raw_plan_text": state.get("raw_plan_text", ""),
        "trip_plan": state.get("trip_plan"),
        "attraction_photos": state.get("attraction_photos", {}),
    }


# ============================================================
# 节点 1：Supervisor
# ============================================================

def supervisor_node(state: TripState) -> dict:
    """
    Supervisor 节点 — 阶段级路由决策

    P1 优化：Supervisor 只在 3 个阶段切换点被调用：
    1. 工作流开始 → 决定 "collect"
    2. 数据收集完成 → 决定 "plan"
    3. 规划+预算完成 → 决定 "finalize"
    """
    return _supervisor.run(state)


# ============================================================
# 节点 2：数据收集（P1 新增 — 并行 POI + Weather + Hotel）
# ============================================================

def data_collection_node(state: TripState) -> dict:
    """
    数据收集节点 — 并行运行 POI、Weather、Hotel 三个 Agent

    P1 优化核心：
    - 三个 Agent 之间无数据依赖（都只需要 request.city）
    - 用 ThreadPoolExecutor 并行执行，总耗时 = max(单 Agent 耗时) 而非 sum
    - 三个 Agent 全部完成后，汇总结果一次性返回给 Supervisor
    - 避免了中间 4 次 Supervisor 调用（原来每个 Agent 前后各一次）

    错误处理：
    - 单个 Agent 失败不影响其他 Agent 继续执行
    - 所有失败信息汇总到 error 字段，Supervisor 据此决定降级策略
    """
    print(f"\n{'=' * 60}")
    print(f"📦 [data_collection_node] 并行数据收集开始 (POI + Weather + Hotel)")
    print(f"{'=' * 60}")

    results: dict = {
        "raw_attractions": state.get("raw_attractions", []),
        "raw_weather": state.get("raw_weather", []),
        "raw_hotels": state.get("raw_hotels", []),
        "agent_outputs": dict(state.get("agent_outputs", {})),
        "error": "",
    }
    errors: list[str] = []

    # 定义三个 Agent 的执行函数（捕获异常到返回值中）
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

    # 并行执行
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_poi): "poi",
            executor.submit(run_weather): "weather",
            executor.submit(run_hotel): "hotel",
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                agent_result = future.result()

                # 合并 agent_outputs
                if "agent_outputs" in agent_result:
                    results["agent_outputs"].update(agent_result["agent_outputs"])

                # 合并数据
                if agent_name == "poi" and "raw_attractions" in agent_result:
                    results["raw_attractions"] = agent_result["raw_attractions"]
                    count = len(agent_result["raw_attractions"])
                    print(f"  ✅ POI Agent 完成: {count} 个景点")

                elif agent_name == "weather" and "raw_weather" in agent_result:
                    results["raw_weather"] = agent_result["raw_weather"]
                    count = len(agent_result["raw_weather"])
                    print(f"  ✅ Weather Agent 完成: {count} 天天气")

                elif agent_name == "hotel" and "raw_hotels" in agent_result:
                    results["raw_hotels"] = agent_result["raw_hotels"]
                    count = len(agent_result["raw_hotels"])
                    print(f"  ✅ Hotel Agent 完成: {count} 个酒店")

                # 收集错误
                if agent_result.get("error"):
                    errors.append(agent_result["error"])

            except Exception as e:
                print(f"  ❌ {agent_name} 线程异常: {e}")
                errors.append(f"{agent_name}: {e}")

    # 汇总错误
    if errors:
        results["error"] = "; ".join(errors)
        print(f"  ⚠️ 数据收集阶段错误: {results['error']}")

    total_attractions = len(results["raw_attractions"])
    total_hotels = len(results["raw_hotels"])
    total_weather = len(results["raw_weather"])

    print(f"  📊 收集汇总: {total_attractions} 景点, {total_weather} 天天气, {total_hotels} 酒店")
    print(f"{'=' * 60}\n")

    return results


# ============================================================
# 节点 3-4：规划 + 预算（链式执行，中间不回 Supervisor）
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
            "last_agent": "planner",
            "agent_outputs": {"planner_agent": f"执行失败: {str(e)}"},
        }


def budget_node(state: TripState) -> dict:
    """
    Budget Agent 节点 — 计算预算

    此节点是 planner → budget → supervisor 链的最后一环。
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
            "last_agent": "budget",
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
            "last_agent": "poi",
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
            "last_agent": "weather",
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
            "last_agent": "hotel",
            "agent_outputs": {"hotel_agent": f"重试失败: {str(e)}"},
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
        "next_agent": "",
    }


# ============================================================
# 辅助函数
# ============================================================

def _extract_json(text: str) -> str:
    """从文本中提取 JSON 字符串（括号计数法）"""
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    if "{" in text and "}" in text:
        start = text.find("{")
        brace_count = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        return text[start:end] if end > start else text

    raise ValueError("文本中未找到 JSON 数据")


def _create_fallback_plan(request, state) -> "TripPlan":
    """
    备用计划生成 — 优先使用 Agent 已收集的真实数据

    优先链：
    raw_attractions 有数据 → 用真实 POI
    raw_hotels 有数据 → 填充真实酒店
    raw_weather 有数据 → 填充真实天气
    都没有 → 硬编码兜底
    """
    from app.schemas.models import TripPlan, DayPlan, Attraction, Meal, Location, Hotel, WeatherInfo

    start = datetime.strptime(request.start_date, "%Y-%m-%d")
    raw_attractions = state.get("raw_attractions", [])
    raw_hotels = state.get("raw_hotels", [])
    raw_weather = state.get("raw_weather", [])

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

    return TripPlan(
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        days=days,
        weather_info=weather_info,
        overall_suggestions=fallback_desc,
    )
