"""
历史计划 SQLite 存储 — 零依赖，零配置

数据库文件：项目根目录 data/trips.db（自动创建）
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "trips.db"


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表（首次调用时建表）"""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id          TEXT PRIMARY KEY,
            city        TEXT NOT NULL,
            start_date  TEXT NOT NULL,
            end_date    TEXT NOT NULL,
            travel_days INTEGER NOT NULL,
            people_count INTEGER DEFAULT 1,
            preferences TEXT DEFAULT '[]',
            budget_total REAL,
            trip_data   TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(trips)").fetchall()]
    if "people_count" not in columns:
        conn.execute("ALTER TABLE trips ADD COLUMN people_count INTEGER DEFAULT 1")
    conn.commit()
    conn.close()


def save_trip(request, trip_plan) -> str:
    """
    保存旅行计划

    Args:
        request: TripRequest 对象
        trip_plan: TripPlan 对象

    Returns:
        新记录的 UUID
    """
    init_db()
    trip_id = uuid.uuid4().hex[:12]
    conn = _get_conn()
    people_count = max(1, int(getattr(request, "people_count", 1) or 1))
    if hasattr(trip_plan, "people_count"):
        try:
            trip_plan.people_count = people_count
        except Exception:
            pass

    if hasattr(trip_plan, "model_dump"):
        trip_data = trip_plan.model_dump()
        trip_data["people_count"] = people_count
        trip_json = json.dumps(trip_data, ensure_ascii=False)
    else:
        trip_json = json.dumps({"people_count": people_count}, ensure_ascii=False)

    conn.execute(
        """
        INSERT INTO trips (id, city, start_date, end_date, travel_days, people_count, preferences, budget_total, trip_data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trip_id,
            getattr(request, "city", ""),
            getattr(request, "start_date", ""),
            getattr(request, "end_date", ""),
            getattr(request, "travel_days", 0),
            people_count,
            json.dumps(getattr(request, "preferences", []), ensure_ascii=False),
            trip_plan.budget.total if (trip_plan and trip_plan.budget) else None,
            trip_json,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return trip_id


def list_trips(limit: int = 20) -> list[dict]:
    """
    查询历史计划列表（不含完整 trip_data）

    Returns:
        [{id, city, start_date, end_date, travel_days, preferences, budget_total, created_at}, ...]
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, city, start_date, end_date, travel_days, people_count, preferences, budget_total, created_at FROM trips ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["preferences"] = json.loads(d["preferences"])
        except (json.JSONDecodeError, TypeError):
            d["preferences"] = []
        result.append(d)
    return result


def get_trip(trip_id: str) -> dict | None:
    """获取完整旅行计划 JSON"""
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT trip_data, city FROM trips WHERE id = ?", (trip_id,)).fetchone()
    conn.close()
    if not row:
        return None
    try:
        data = json.loads(row["trip_data"])
        return {"trip_data": data, "city": row["city"]}
    except json.JSONDecodeError:
        return None


def delete_trip(trip_id: str) -> bool:
    """删除指定计划"""
    init_db()
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM trips WHERE id = ?", (trip_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
