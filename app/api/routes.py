"""
FastAPI 路由 — 对外暴露 HTTP 接口

职责：
1. 接收前端请求 → Pydantic 自动校验
2. 调用 LangGraph 工作流
3. 支持两种模式：
   - POST /trip/plan       : 普通 JSON 响应（兼容旧版）
   - POST /trip/plan/stream: SSE 流式响应（真实进度推送）

不包含任何业务逻辑。业务全在 graph 层。
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.models import TripRequest, TripPlanResponse
from app.graph.workflow import get_workflow
from app.tools.unsplash import get_photo_url

router = APIRouter(prefix="/api", tags=["旅行规划"])

# 节点名称 → 用户可见的 Agent 名称 映射
NODE_TO_AGENT_NAME: dict[str, str] = {
    "initialize_node": "初始化",
    "supervisor_node": "调度中心",
    "data_collection_node": "数据收集 (景点+天气+酒店)",
    "poi_node": "景点搜索",
    "weather_node": "天气查询",
    "hotel_node": "酒店推荐",
    "planner_node": "行程规划",
    "budget_node": "预算计算",
    "finalize_node": "完成处理",
}

# 节点 → 完成后对应的进度百分比（P1 优化：3 阶段）
NODE_PROGRESS: dict[str, int] = {
    "initialize_node": 5,
    "data_collection_node": 40,
    "planner_node": 75,
    "budget_node": 90,
    "finalize_node": 100,
}


def _sse_event(event_type: str, data: dict | str) -> str:
    """将事件和 data 组装为 SSE 格式"""
    if isinstance(data, dict):
        data_str = json.dumps(data, ensure_ascii=False)
    else:
        data_str = data
    return f"event: {event_type}\ndata: {data_str}\n\n"


async def event_generator(initial_state: dict) -> str:
    """
    LangGraph 事件 → SSE 事件流的异步生成器

    使用 astream_events(version="v2") 捕获所有节点和工具事件，
    转换为前端可消费的 SSE 事件格式。

    发送的事件类型：
    - agent_start : 节点开始执行
    - tool_start  : Agent 内部工具开始调用
    - progress    : 进度更新（当前节点 + 百分比）
    - agent_end   : 节点执行完成
    - done        : 最终 TripPlan 数据
    - error       : 运行异常
    """
    workflow = get_workflow()
    last_progress = 0
    current_node = ""

    try:
        async for event in workflow.astream_events(initial_state, version="v2"):
            event_kind = event.get("event", "")
            event_name = event.get("name", "")
            event_data = event.get("data", {})

            # ——— 节点开始 ———
            if event_kind == "on_chain_start" and event_name in NODE_TO_AGENT_NAME:
                current_node = event_name
                agent_name = NODE_TO_AGENT_NAME.get(event_name, event_name)
                yield _sse_event("agent_start", {
                    "agent": event_name,
                    "name": agent_name,
                })

            # ——— 工具开始（Agent 内部的 Tool Calling）———
            elif event_kind == "on_tool_start" and event_name:
                tool_name = event_name
                # 过滤掉 LangGraph 内部工具
                if not event_name.startswith("__"):
                    yield _sse_event("tool_start", {
                        "tool": tool_name,
                        "agent": current_node,
                    })

            # ——— 工具结束 ———
            elif event_kind == "on_tool_end" and event_name:
                if not event_name.startswith("__"):
                    yield _sse_event("tool_end", {
                        "tool": event_name,
                        "agent": current_node,
                    })

            # ——— 节点结束 ———
            elif event_kind == "on_chain_end" and event_name in NODE_TO_AGENT_NAME:
                # 更新进度
                if event_name in NODE_PROGRESS:
                    pct = NODE_PROGRESS[event_name]
                    if pct > last_progress:
                        last_progress = pct
                        yield _sse_event("progress", {
                            "percent": pct,
                            "node": event_name,
                            "name": NODE_TO_AGENT_NAME.get(event_name, event_name),
                            "message": f"{NODE_TO_AGENT_NAME.get(event_name, event_name)} 完成",
                        })

                yield _sse_event("agent_end", {
                    "agent": event_name,
                    "name": NODE_TO_AGENT_NAME.get(event_name, event_name),
                })

            # ——— LangGraph 整体流结束 ———
            elif event_kind == "on_chain_end" and event_name == "LangGraph":
                final_state = event_data.get("output", {})
                trip_plan = final_state if isinstance(final_state, dict) else {}

                # 如果 output 本身就是 State，提取 trip_plan
                if isinstance(trip_plan, dict):
                    plan_data = trip_plan.get("trip_plan")
                    error_msg = trip_plan.get("error", "")
                else:
                    plan_data = None
                    error_msg = ""

                if plan_data is not None:
                    # Pydantic model → dict
                    try:
                        plan_dict = plan_data.model_dump() if hasattr(plan_data, "model_dump") else plan_data
                    except Exception:
                        plan_dict = str(plan_data)

                    yield _sse_event("done", {
                        "success": True,
                        "message": "旅行计划生成成功" if not error_msg else f"计划已生成（{error_msg}）",
                        "data": plan_dict,
                    })
                else:
                    yield _sse_event("error", {
                        "success": False,
                        "message": error_msg or "未能生成旅行计划",
                    })

    except Exception as e:
        print(f"❌ SSE 事件流异常: {e}")
        yield _sse_event("error", {
            "success": False,
            "message": f"生成旅行计划失败: {str(e)}",
        })


# ============================================================
# 端点 1: 普通 JSON 响应（兼容旧版前端）
# ============================================================

@router.post("/trip/plan", response_model=TripPlanResponse)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划（普通 JSON 响应）

    前端 POST JSON → Pydantic 自动解析为 TripRequest
    → 调用 LangGraph 工作流 → 返回 TripPlanResponse
    """
    try:
        workflow = get_workflow()

        initial_state = {
            "request": request,
            "messages": [],
            "next_agent": "",
            "last_agent": "",
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

        print(f"\n{'=' * 50}")
        print(f"🚀 开始规划: {request.city} {request.travel_days}天")
        print(f"{'=' * 50}")

        result = workflow.invoke(initial_state)

        print(f"{'=' * 50}")
        print(f"✅ 规划完成")
        print(f"{'=' * 50}\n")

        trip_plan = result.get("trip_plan")
        error = result.get("error", "")

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功" if not error else f"计划已生成（{error}）",
            data=trip_plan,
        )

    except Exception as e:
        print(f"❌ 规划失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {str(e)}")


# ============================================================
# 端点 2: SSE 流式响应（新版前端，真实进度推送）
# ============================================================

@router.post("/trip/plan/stream")
async def plan_trip_stream(request: TripRequest):
    """
    生成旅行计划（SSE 流式响应）

    使用 LangGraph astream_events 推送实时进度事件到前端。

    事件类型：
    - agent_start : {"agent": "poi_node", "name": "景点搜索"}
    - tool_start  : {"tool": "search_attractions_tool", "agent": "poi_node"}
    - tool_end    : {"tool": "search_attractions_tool", "agent": "poi_node"}
    - progress    : {"percent": 25, "node": "poi_node", "name": "景点搜索", "message": "..."}
    - agent_end   : {"agent": "poi_node", "name": "景点搜索"}
    - done        : {"success": true, "message": "...", "data": TripPlan}
    - error       : {"success": false, "message": "..."}
    """
    initial_state = {
        "request": request,
        "messages": [],
        "next_agent": "",
        "last_agent": "",
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

    print(f"\n{'=' * 50}")
    print(f"🚀 开始流式规划: {request.city} {request.travel_days}天")
    print(f"{'=' * 50}")

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
# 端点 3: 健康检查
# ============================================================

@router.get("/trip/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "trip-planner"}


# ============================================================
# 端点 4: 景点图片
# ============================================================

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

        return {
            "success": True,
            "message": "获取图片成功",
            "data": {
                "name": name,
                "photo_url": photo_url,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")