"""
预算计算工具 — 供 Budget Agent 在 ReAct 循环中调用

用途：
- calculate_budget_tool : 精确计算行程总预算（门票+酒店+餐饮+交通）
- suggest_savings_tool  : 当预算超限时，分析并给出可行的削减方案

类型防御：所有嵌套字段访问前均用 isinstance 检查，防止 LLM 将对象简化为字符串。
"""

import json
from langchain_core.tools import tool


@tool
def calculate_budget_tool(
    trip_plan_json: str,
    hotel_cost_per_night: int = 0,
    transport_cost_per_day: int = 50,
) -> str:
    """
    精确计算旅行计划的总预算。

    解析行程计划 JSON，逐项汇总以下费用：
    1. 景点门票费用（所有天、所有景点的 ticket_price 之和）
    2. 酒店住宿费用（hotel.estimated_cost × 住宿天数，或使用 hotel_cost_per_night 估算）
    3. 餐饮费用（所有天、所有餐的 estimated_cost 之和）
    4. 交通费用（transport_cost_per_day × 旅行天数）

    当 Budget Agent 需要计算或复核行程预算时调用此工具。

    Args:
        trip_plan_json: 完整的旅行计划 JSON 字符串。
                        可以是纯 JSON 或包含 ```json 代码块的文本。
                        必须包含 days 数组，每个 day 包含 attractions、meals、hotel 字段。
        hotel_cost_per_night: 如果行程中没有指定酒店费用，使用此参数作为每晚酒店估算费用。
                              默认为 0（不额外估算）。
        transport_cost_per_day: 每日交通费用估算，默认为 50 元/天。

    Returns:
        JSON 字符串，包含详细预算明细：
        - total_attractions: 门票总费用
        - total_hotels: 酒店总费用
        - total_meals: 餐饮总费用
        - total_transportation: 交通总费用
        - total: 总费用
        - breakdown: 逐日明细列表
        - warnings: 预算异常警告列表
    """
    defaults = {
        "total_attractions": 0, "total_hotels": 0,
        "total_meals": 0, "total_transportation": 0, "total": 0,
    }

    # 提取 JSON
    try:
        if isinstance(trip_plan_json, str):
            json_str = _extract_json(trip_plan_json)
            plan = json.loads(json_str)
        else:
            plan = trip_plan_json
    except Exception as e:
        return json.dumps({
            **defaults,
            "error": f"无法解析行程 JSON: {str(e)}",
        }, ensure_ascii=False)

    # 类型防御：plan 必须是 dict
    if not isinstance(plan, dict):
        return json.dumps({
            **defaults,
            "error": f"JSON 根必须为对象，实际类型: {type(plan).__name__}",
        }, ensure_ascii=False)

    days = plan.get("days", [])

    # 类型防御：days 必须是 list
    if not isinstance(days, list):
        return json.dumps({
            **defaults,
            "error": f"days 必须为数组，实际类型: {type(days).__name__}",
        }, ensure_ascii=False)

    travel_days = len(days)

    total_attractions = 0
    total_hotels = 0
    total_meals = 0
    warnings: list[dict] = []
    daily_breakdown: list[dict] = []

    for day in days:
        # 类型防御：day 必须是 dict
        if not isinstance(day, dict):
            warnings.append({
                "type": "invalid_day",
                "message": f"day 必须为对象，实际类型: {type(day).__name__}，跳过",
            })
            continue

        day_num = day.get("day_index", 0) + 1
        day_costs = {
            "day": day_num,
            "date": day.get("date", ""),
            "attractions": 0,
            "hotel": 0,
            "meals": 0,
        }

        # 景点门票
        attractions = day.get("attractions", [])
        if isinstance(attractions, list):
            for attr in attractions:
                if not isinstance(attr, dict):
                    warnings.append({
                        "type": "invalid_attraction",
                        "day": day_num,
                        "message": f"景点必须为对象，实际类型: {type(attr).__name__}",
                    })
                    continue

                price = attr.get("ticket_price", 0)
                if isinstance(price, (int, float)) and price >= 0:
                    total_attractions += price
                    day_costs["attractions"] += price
                else:
                    warnings.append({
                        "type": "invalid_ticket_price",
                        "day": day_num,
                        "attraction": attr.get("name", "未知"),
                        "price": price,
                    })
        else:
            warnings.append({
                "type": "invalid_attractions",
                "day": day_num,
                "message": f"attractions 必须为数组，实际类型: {type(attractions).__name__}",
            })

        # 餐饮
        meals = day.get("meals", [])
        if isinstance(meals, list):
            for meal in meals:
                if not isinstance(meal, dict):
                    warnings.append({
                        "type": "invalid_meal",
                        "day": day_num,
                        "message": f"餐饮项必须为对象，实际类型: {type(meal).__name__}",
                    })
                    continue

                cost = meal.get("estimated_cost", 0)
                if isinstance(cost, (int, float)) and cost >= 0:
                    total_meals += cost
                    day_costs["meals"] += cost
                else:
                    warnings.append({
                        "type": "invalid_meal_cost",
                        "day": day_num,
                        "meal_name": meal.get("name", "未知"),
                        "cost": cost,
                    })
        else:
            warnings.append({
                "type": "invalid_meals",
                "day": day_num,
                "message": f"meals 必须为数组，实际类型: {type(meals).__name__}",
            })

        # 酒店
        hotel = day.get("hotel")
        if hotel is not None:
            if isinstance(hotel, dict):
                hotel_cost = hotel.get("estimated_cost", 0)
                if isinstance(hotel_cost, (int, float)) and hotel_cost > 0:
                    total_hotels += hotel_cost
                    day_costs["hotel"] = hotel_cost
            else:
                warnings.append({
                    "type": "invalid_hotel",
                    "day": day_num,
                    "message": f"hotel 必须为对象，实际类型: {type(hotel).__name__}",
                })

        daily_breakdown.append(day_costs)

    # 如果行程中没有任何酒店费用，用估算值
    if total_hotels == 0 and hotel_cost_per_night > 0:
        total_hotels = hotel_cost_per_night * travel_days
        warnings.append({
            "type": "hotel_estimated",
            "message": f"行程中未包含酒店费用，使用估算值 {hotel_cost_per_night}元/晚 × {travel_days}晚",
        })

    total_transportation = transport_cost_per_day * travel_days
    total = total_attractions + total_hotels + total_meals + total_transportation

    # 异常检测
    if total_attractions == 0:
        warnings.append({"type": "zero_attractions", "message": "门票总费用为 0，请确认景点数据是否完整"})
    if total_meals == 0:
        warnings.append({"type": "zero_meals", "message": "餐饮总费用为 0，请确认餐饮数据是否完整"})
    if total_meals > 0 and travel_days > 0:
        avg_meal_per_day = total_meals / travel_days
        if avg_meal_per_day < 50:
            warnings.append({"type": "low_meal_budget", "message": f"日均餐饮仅 {avg_meal_per_day:.0f} 元，可能偏低"})
        if avg_meal_per_day > 500:
            warnings.append({"type": "high_meal_budget", "message": f"日均餐饮 {avg_meal_per_day:.0f} 元，可能偏高"})

    return json.dumps({
        "total_attractions": total_attractions,
        "total_hotels": total_hotels,
        "total_meals": total_meals,
        "total_transportation": total_transportation,
        "total": total,
        "breakdown": daily_breakdown,
        "warnings": warnings,
    }, ensure_ascii=False, indent=2)


@tool
def suggest_savings_tool(
    current_budget_json: str,
    target_budget: int = 0,
    travel_days: int = 1,
) -> str:
    """
    分析当前预算并提供可行的削减方案。

    当 Budget Agent 发现旅行总预算超出用户预期时调用此工具。
    工具会分析各费用类别占比，按优先级给出削减建议：
    1. 酒店：替换为更经济的住宿（如豪华→舒适，舒适→经济）
    2. 景点：减少付费景点，增加免费景点
    3. 餐饮：调整三餐标准
    4. 交通：优化出行方式

    Args:
        current_budget_json: 预算明细 JSON 字符串（通常来自 calculate_budget_tool 的输出）。
                             必须包含 total_attractions / total_hotels / total_meals /
                             total_transportation / total 字段。
        target_budget: 用户的目标预算上限（元），为 0 表示无上限约束。
        travel_days: 旅行天数，用于计算日均费用。

    Returns:
        JSON 字符串，包含削减建议：
        - current_total: 当前总费用
        - target_budget: 目标预算
        - overshoot: 超出金额（0 表示未超限）
        - suggestions: 削减建议列表，每条包含 category / current / suggested / saving / description
        - adjusted_total: 按建议调整后的预估总费用
    """
    try:
        if isinstance(current_budget_json, str):
            budget = json.loads(current_budget_json)
        else:
            budget = current_budget_json
    except Exception as e:
        return json.dumps({
            "error": f"无法解析预算 JSON: {str(e)}",
            "suggestions": [],
        }, ensure_ascii=False)

    if not isinstance(budget, dict):
        return json.dumps({
            "error": f"预算数据必须为对象，实际类型: {type(budget).__name__}",
            "suggestions": [],
        }, ensure_ascii=False)

    current_total = budget.get("total", 0)
    overshoot = max(0, current_total - target_budget) if target_budget > 0 else 0

    suggestions: list[dict] = []

    # 如果无预算上限或无超限，只给出优化建议
    if overshoot == 0 and target_budget > 0:
        return json.dumps({
            "current_total": current_total,
            "target_budget": target_budget,
            "overshoot": 0,
            "message": "预算在目标范围内，无需削减",
            "suggestions": [],
            "adjusted_total": current_total,
        }, ensure_ascii=False, indent=2)

    # 分析各费用占比
    total_attractions = budget.get("total_attractions", 0)
    total_hotels = budget.get("total_hotels", 0)
    total_meals = budget.get("total_meals", 0)
    total_transportation = budget.get("total_transportation", 0)

    # 按金额从高到低排序，优先削减占比最大的类别
    categories = [
        ("hotels", total_hotels, "酒店住宿"),
        ("attractions", total_attractions, "景点门票"),
        ("meals", total_meals, "餐饮费用"),
        ("transportation", total_transportation, "交通费用"),
    ]
    categories.sort(key=lambda x: x[1], reverse=True)

    remaining_overshoot = overshoot
    adjusted_total = current_total

    for cat_key, cat_amount, cat_name in categories:
        if cat_amount <= 0 or remaining_overshoot <= 0:
            continue

        if cat_key == "hotels":
            saving = min(cat_amount, remaining_overshoot)
            savings_pct = min(0.4, saving / cat_amount) if cat_amount > 0 else 0
            actual_saving = int(cat_amount * savings_pct)
            suggestions.append({
                "category": "酒店住宿",
                "current": cat_amount,
                "suggested": cat_amount - actual_saving,
                "saving": actual_saving,
                "description": (
                    f"将酒店标准下调一档（如豪华→舒适→经济），"
                    f"预计节省 {actual_saving} 元。也可以考虑选择距市中心稍远的酒店，"
                    f"价格通常更低但需权衡交通成本。"
                ),
            })
            remaining_overshoot -= actual_saving
            adjusted_total -= actual_saving

        elif cat_key == "attractions":
            saving = min(cat_amount, remaining_overshoot)
            daily_avg = cat_amount / travel_days if travel_days > 0 else cat_amount
            suggest_reduce = max(1, int(remaining_overshoot / daily_avg)) if daily_avg > 0 else 1
            actual_saving = min(cat_amount, daily_avg * suggest_reduce)
            suggestions.append({
                "category": "景点门票",
                "current": cat_amount,
                "suggested": cat_amount - int(actual_saving),
                "saving": int(actual_saving),
                "description": (
                    f"建议将 {suggest_reduce} 天的付费景点替换为免费景点"
                    f"（如公园、博物馆免费日、步行街），预计节省 {int(actual_saving)} 元。"
                ),
            })
            remaining_overshoot -= int(actual_saving)
            adjusted_total -= int(actual_saving)

        elif cat_key == "meals":
            daily_meal_avg = cat_amount / travel_days if travel_days > 0 else cat_amount
            saving_per_day = min(daily_meal_avg * 0.3, daily_meal_avg - 30)
            actual_saving = int(saving_per_day * travel_days)
            actual_saving = min(actual_saving, remaining_overshoot)
            actual_saving = max(0, int(actual_saving))
            suggestions.append({
                "category": "餐饮费用",
                "current": cat_amount,
                "suggested": cat_amount - actual_saving,
                "saving": actual_saving,
                "description": (
                    f"日均餐饮从 {int(daily_meal_avg)} 元降至 {int(daily_meal_avg - saving_per_day)} 元"
                    f"（减少高档餐厅、多尝试当地小吃），预计节省 {actual_saving} 元。"
                ),
            })
            remaining_overshoot -= actual_saving
            adjusted_total -= actual_saving

        elif cat_key == "transportation":
            saving = min(cat_amount, remaining_overshoot)
            actual_saving = min(cat_amount, int(cat_amount * 0.3))
            suggestions.append({
                "category": "交通费用",
                "current": cat_amount,
                "suggested": cat_amount - actual_saving,
                "saving": actual_saving,
                "description": (
                    f"多使用公共交通（地铁/公交）代替打车/租车，"
                    f"预计节省 {actual_saving} 元。"
                ),
            })
            remaining_overshoot -= actual_saving
            adjusted_total -= actual_saving

    return json.dumps({
        "current_total": current_total,
        "target_budget": target_budget if target_budget > 0 else None,
        "overshoot": overshoot,
        "suggestions": suggestions,
        "adjusted_total": adjusted_total,
        "remaining_overshoot": max(0, adjusted_total - target_budget) if target_budget > 0 else 0,
    }, ensure_ascii=False, indent=2)


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
