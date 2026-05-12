"""
FastAPI 路由 — 对外暴露 HTTP 接口

职责：
1. 接收前端请求 → Pydantic 自动校验
2. 调用 LangGraph 工作流
3. 支持两种模式：
   - POST /trip/plan       : 普通 JSON 响应
   - POST /trip/plan/stream: SSE 流式响应
4. 历史计划存储与查询
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.models import TripRequest, TripPlanResponse
from app.graph.workflow import get_workflow
from app.tools.unsplash import get_photo_url
from app.storage import sqlite_store

router = APIRouter(prefix="/api", tags=["旅行规划"])

# 节点名称 → 用户可见的 Agent 名称 映射
NODE_TO_AGENT_NAME: dict[str, str] = {
    "initialize_node": "初始化",
    "data_collection_node": "数据收集 (景点+天气+酒店)",
    "poi_node": "景点搜索",
    "weather_node": "天气查询",
    "hotel_node": "酒店推荐",
    "planner_node": "行程规划",
    "budget_node": "预算计算",
    "finalize_node": "完成处理",
}

NODE_PROGRESS: dict[str, int] = {
    "initialize_node": 5,
    "data_collection_node": 40,
    "planner_node": 75,
    "budget_node": 90,
    "finalize_node": 100,
}


def _sse_event(event_type: str, data: dict | str) -> str:
    if isinstance(data, dict):
        data_str = json.dumps(data, ensure_ascii=False)
    else:
        data_str = data
    return f"event: {event_type}\ndata: {data_str}\n\n"


async def event_generator(initial_state: dict) -> str:
    workflow = get_workflow()
    last_progress = 0
    current_node = ""

    try:
        async for event in workflow.astream_events(initial_state, version="v2"):
            event_kind = event.get("event", "")
            event_name = event.get("name", "")
            event_data = event.get("data", {})

            if event_kind == "on_chain_start" and event_name in NODE_TO_AGENT_NAME:
                current_node = event_name
                agent_name = NODE_TO_AGENT_NAME.get(event_name, event_name)
                yield _sse_event("agent_start", {"agent": event_name, "name": agent_name})

            elif event_kind == "on_tool_start" and event_name:
                if not event_name.startswith("__"):
                    yield _sse_event("tool_start", {"tool": event_name, "agent": current_node})

            elif event_kind == "on_tool_end" and event_name:
                if not event_name.startswith("__"):
                    yield _sse_event("tool_end", {"tool": event_name, "agent": current_node})

            elif event_kind == "on_chain_end" and event_name in NODE_TO_AGENT_NAME:
                if event_name in NODE_PROGRESS:
                    pct = NODE_PROGRESS[event_name]
                    if pct > last_progress:
                        last_progress = pct
                        yield _sse_event("progress", {
                            "percent": pct, "node": event_name,
                            "name": NODE_TO_AGENT_NAME.get(event_name, event_name),
                            "message": f"{NODE_TO_AGENT_NAME.get(event_name, event_name)} 完成",
                        })
                yield _sse_event("agent_end", {"agent": event_name, "name": NODE_TO_AGENT_NAME.get(event_name, event_name)})

            elif event_kind == "on_chain_end" and event_name == "LangGraph":
                final_state = event_data.get("output", {})
                trip_plan = final_state if isinstance(final_state, dict) else {}
                if isinstance(trip_plan, dict):
                    plan_data = trip_plan.get("trip_plan")
                    error_msg = trip_plan.get("error", "")
                else:
                    plan_data = None
                    error_msg = ""

                trip_id = ""
                if plan_data is not None:
                    request = initial_state.get("request")
                    people_count = max(1, int(getattr(request, "people_count", 1) or 1)) if request else 1
                    if hasattr(plan_data, "people_count"):
                        try:
                            plan_data.people_count = people_count
                        except Exception:
                            pass
                    elif isinstance(plan_data, dict):
                        plan_data["people_count"] = people_count

                    try:
                        if request:
                            trip_id = sqlite_store.save_trip(request, plan_data)
                    except Exception as e:
                        print(f"⚠️ 保存计划失败: {e}")

                    try:
                        plan_dict = plan_data.model_dump() if hasattr(plan_data, "model_dump") else plan_data
                    except Exception:
                        plan_dict = str(plan_data)

                    yield _sse_event("done", {
                        "success": True,
                        "message": "旅行计划生成成功" if not error_msg else f"计划已生成（{error_msg}）",
                        "data": plan_dict,
                        "trip_id": trip_id,
                    })
                else:
                    yield _sse_event("error", {
                        "success": False,
                        "message": error_msg or "未能生成旅行计划",
                    })

    except Exception as e:
        print(f"❌ SSE 事件流异常: {e}")
        yield _sse_event("error", {"success": False, "message": f"生成旅行计划失败: {str(e)}"})


# ============================================================
# 端点 1: 普通 JSON 响应
# ============================================================

@router.post("/trip/plan", response_model=TripPlanResponse)
async def plan_trip(request: TripRequest):
    try:
        workflow = get_workflow()
        initial_state = {
            "request": request,
            "messages": [],
            "retry_count": 0,
            "phase": "collect",
            "max_iterations": 10,
            "iteration_count": 0,
            "agent_outputs": {},
            "raw_attractions": [],
            "raw_weather": [],
            "raw_hotels": [],
            "raw_plan_text": "",
            "trip_plan": None,
            "attraction_photos": {},
            "error": "",
        }

        result = workflow.invoke(initial_state)
        trip_plan = result.get("trip_plan")
        error = result.get("error", "")

        # 自动存档
        if trip_plan is not None:
            if hasattr(trip_plan, "people_count"):
                try:
                    trip_plan.people_count = max(1, int(getattr(request, "people_count", 1) or 1))
                except Exception:
                    pass
            try:
                sqlite_store.save_trip(request, trip_plan)
            except Exception as e:
                print(f"⚠️ 保存计划失败: {e}")

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功" if not error else f"计划已生成（{error}）",
            data=trip_plan,
        )
    except Exception as e:
        print(f"❌ 规划失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {str(e)}")


# ============================================================
# 端点 2: SSE 流式响应
# ============================================================

@router.post("/trip/plan/stream")
async def plan_trip_stream(request: TripRequest):
    initial_state = {
        "request": request,
        "messages": [],
        "retry_count": 0,
        "phase": "collect",
        "max_iterations": 10,
        "iteration_count": 0,
        "agent_outputs": {},
        "raw_attractions": [],
        "raw_weather": [],
        "raw_hotels": [],
        "raw_plan_text": "",
        "trip_plan": None,
        "attraction_photos": {},
        "error": "",
    }

    return StreamingResponse(
        event_generator(initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 端点 3: 历史计划
# ============================================================

@router.get("/history")
async def list_history(limit: int = 20):
    """获取历史计划列表（不含完整 JSON）"""
    try:
        trips = sqlite_store.list_trips(limit)
        return {"success": True, "data": trips}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@router.get("/history/{trip_id}")
async def get_history(trip_id: str):
    """获取单个历史计划的完整数据"""
    try:
        trip = sqlite_store.get_trip(trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail="计划不存在")
        return {"success": True, "data": trip["trip_data"], "city": trip["city"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取计划失败: {str(e)}")


@router.delete("/history/{trip_id}")
async def delete_history(trip_id: str):
    """删除指定历史计划"""
    try:
        deleted = sqlite_store.delete_trip(trip_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="计划不存在")
        return {"success": True, "message": "已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ============================================================
# 端点 4: 健康检查 + 景点图片
# ============================================================

@router.get("/trip/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner"}


@router.get("/poi/photo")
async def get_attraction_photo(name: str):
    """
    获取景点图片
    前端在渲染结果页时，会为每个景点请求这个接口获取配图。
    """
    try:
        photo_url = get_photo_url(f"{name} China landmark")
        if not photo_url:
            photo_url = get_photo_url(name)
        return {"success": True, "message": "获取图片成功", "data": {"name": name, "photo_url": photo_url}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")