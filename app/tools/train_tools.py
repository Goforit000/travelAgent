"""
12306 MCP 火车票查询工具 — 封装 MCP Streamable HTTP 客户端调用

通过 JSON-RPC 调用 mcp-server-12306 的 tools/call 接口，获取官方 12306 实时车次数据。
默认服务地址: http://localhost:8001/mcp（与 TripPlanner 的 8000 端口错开）。

MCP 2025-03-26 Streamable HTTP 协议要求:
  1. POST initialize → 获取 Mcp-Session-Id
  2. POST tools/call（带 Mcp-Session-Id header）→ 调用工具
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.utils.config import settings

# 12306 MCP 服务地址（可通过环境变量 MCP_12306_URL 覆盖）
MCP_12306_URL: str = getattr(settings, "mcp_12306_url", "") or "http://localhost:8001/mcp"

_request_id = 0
_session_id: str | None = None


def list_tools() -> list[dict[str, Any]]:
    """列出 12306 MCP 服务器所有可用工具（调试用）。"""
    global _request_id, _session_id
    if not _session_id:
        _mcp_initialize()

    _request_id += 1
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": _request_id,
    }
    print("[train_tools][TRACE] 调用 tools/list...")
    response = httpx.post(
        MCP_12306_URL,
        json=payload,
        headers={"Content-Type": "application/json", "Mcp-Session-Id": _session_id},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    tools_data = data.get("result", {}).get("tools", [])
    for t in tools_data:
        print(f"[train_tools][TRACE]   工具: {t.get('name')} — {t.get('description', '')[:80]}")
    return tools_data


def _mcp_initialize() -> str:
    """发送 initialize 请求，获取 Mcp-Session-Id。"""
    global _request_id, _session_id
    _request_id += 1

    payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "TripPlanner", "version": "1.0"},
        },
        "id": _request_id,
    }

    print("[train_tools][TRACE] MCP initialize...")
    response = httpx.post(
        MCP_12306_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    response.raise_for_status()

    session_id = response.headers.get("Mcp-Session-Id")
    if not session_id:
        raise RuntimeError("MCP initialize 未返回 Mcp-Session-Id")

    _session_id = session_id
    print(f"[train_tools][TRACE] MCP session 已建立: {_session_id[:12]}...")

    # 发送 initialized 通知
    _request_id += 1
    notify = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    httpx.post(
        MCP_12306_URL,
        json=notify,
        headers={"Content-Type": "application/json", "Mcp-Session-Id": _session_id},
        timeout=10,
    )

    # 自动发现可用工具名
    _request_id += 1
    tools_resp = httpx.post(
        MCP_12306_URL,
        json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": _request_id},
        headers={"Content-Type": "application/json", "Mcp-Session-Id": _session_id},
        timeout=15,
    )
    if tools_resp.status_code == 200:
        tools_data = tools_resp.json()
        tools = tools_data.get("result", {}).get("tools", [])
        print(f"[train_tools][TRACE] 发现 {len(tools)} 个 MCP 工具:")
        for t in tools:
            name = t.get("name", "?")
            desc = (t.get("description", "") or "")[:80]
            params = list(t.get("inputSchema", {}).get("properties", {}).keys())
            print(f"[train_tools][TRACE]   {name}({', '.join(params)}) — {desc}")

    return _session_id


def _mcp_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    JSON-RPC 调用 mcp-server-12306 的 tools/call。

    自动管理 MCP session：首次调用时自动 initialize，
    后续调用复用 Mcp-Session-Id。
    """
    global _request_id, _session_id

    # 确保已初始化 session
    if not _session_id:
        _mcp_initialize()

    _request_id += 1
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
        "id": _request_id,
    }

    print(f"[train_tools][TRACE] MCP 调用: {tool_name}({json.dumps(arguments, ensure_ascii=False)})")

    try:
        response = httpx.post(
            MCP_12306_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Mcp-Session-Id": _session_id,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.ConnectError:
        raise RuntimeError(f"无法连接 12306 MCP 服务 ({MCP_12306_URL})，请确认已启动 mcp-server-12306")
    except Exception as e:
        raise RuntimeError(f"12306 MCP 请求失败: {e}")

    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"MCP 错误: {err.get('message', err)}")

    # MCP 返回格式: result.content[].text 包含 JSON 字符串
    result = data.get("result", {})
    content = result.get("content", [])
    print(f"[train_tools][TRACE] MCP result content 条数: {len(content)}, "
          f"result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    for idx, item in enumerate(content):
        print(f"[train_tools][TRACE]   content[{idx}]: type={item.get('type')}, text_len={len(item.get('text', ''))}, text_preview={str(item.get('text', ''))[:300]}")
        if item.get("type") == "text":
            text = item.get("text", "{}")
            try:
                parsed = json.loads(text)
                print(f"[train_tools][TRACE]   content[{idx}] 解析后: keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}, preview={str(parsed)[:300]}")
                return parsed
            except json.JSONDecodeError:
                return {"raw": text}

    return result


def search_stations(keyword: str) -> list[dict[str, str]]:
    """
    模糊搜索火车站。

    返回: [{"name": "北京", "telecode": "BJP", "pinyin": "beijing", "city": "北京"}, ...]
    """
    result = _mcp_call("search-stations", {"query": keyword})
    print(f"[train_tools][TRACE] search-stations 原始返回: type={type(result).__name__}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}, preview={str(result)[:500]}")
    # 尝试多种可能的返回格式
    stations = result.get("stations", result.get("data", result.get("result", [])))
    if not stations and isinstance(result, list):
        stations = result
    if not stations:
        for key in ("items", "results", "station_list"):
            val = result.get(key, [])
            if val:
                stations = val
                break
    print(f"[train_tools][TRACE] search-stations 解析后: {len(stations)} 个车站")
    if stations:
        print(f"[train_tools][TRACE]   第一条: {stations[0]}")
    return stations


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

    返回: 车次列表，每项含 train_no/start_time/arrive_time/duration/座席/票价等
    """
    result = _mcp_call("query-tickets", {
        "from_station": from_station,
        "to_station": to_station,
        "train_date": date,
    })
    print(f"[train_tools][TRACE] query-tickets 原始返回: type={type(result).__name__}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}, preview={str(result)[:500]}")
    tickets = result.get("tickets", result.get("data", result.get("result", [])))
    if not tickets and isinstance(result, list):
        tickets = result
    if not tickets:
        for key in ("items", "trains", "results", "ticket_list"):
            val = result.get(key, [])
            if val:
                tickets = val
                break
    print(f"[train_tools][TRACE] query-tickets({from_station}→{to_station}, {date}) 返回 {len(tickets)} 条")
    if tickets:
        print(f"[train_tools][TRACE]   第一条: {tickets[0]}")
    return tickets


def query_ticket_price(
    from_station: str,
    to_station: str,
    train_date: str,
    train_code: str,
) -> dict[str, Any]:
    """
    查询指定车次的票价信息。

    返回: 票价字典，含各席别价格（如 second_class_price、first_class_price 等）。
    """
    result = _mcp_call("query-ticket-price", {
        "from_station": from_station,
        "to_station": to_station,
        "train_date": train_date,
        "train_code": train_code,
    })
    print(f"[train_tools][TRACE] query-ticket-price({train_code}) 返回: "
          f"keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}, preview={str(result)[:300]}")
    # 可能嵌套在某个 key 下
    price_info = result.get("price", result.get("data", result))
    if isinstance(price_info, list) and price_info:
        price_info = price_info[0]
    return price_info


def resolve_station_telecode(city: str) -> str | None:
    """
    城市名 → 首选火车站电报码。

    搜索城市对应的车站，优先选主站（不带方位后缀的）。
    例如 "北京" → "BJP"（北京站），而非 "BXP"（北京西站）。
    """
    stations = search_stations(city)
    if not stations:
        return None

    # 优先精确匹配城市名的站（如 "北京" 对应 "北京" 站）
    for s in stations:
        if s.get("name") == city or s.get("city") == city:
            code = s.get("code", "") or s.get("telecode", "")
            print(f"[train_tools][TRACE] 车站匹配: {city} → {s.get('name')}({code})")
            return code

    # 退而取第一条
    first = stations[0]
    code = first.get("code", "") or first.get("telecode", "")
    print(f"[train_tools][TRACE] 车站匹配(退而): {city} → {first.get('name')}({code})")
    return code