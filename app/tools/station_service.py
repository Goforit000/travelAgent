"""
车站数据服务 — 从 12306 station_name.js 解析全国火车站列表。
支持按名称、三字码、拼音、简拼搜索。

来源：mcp-server-12306/src/mcp_12306/services/station_service.py
改造：aiofiles → 同步 open，适配 TripPlanner app 路径。
"""

import os
import re
import logging
from typing import Optional


class Station:
    def __init__(self, name, code, pinyin, py_short, num, city=None):
        self.name = name
        self.code = code
        self.pinyin = pinyin
        self.py_short = py_short
        self.num = num
        self.city = city

    def __repr__(self):
        return f"Station(name={self.name}, code={self.code}, pinyin={self.pinyin}, city={self.city})"


class StationSearchResult:
    def __init__(self, stations):
        self.stations = stations


class StationService:
    def __init__(self):
        self.stations: list[Station] = []

    def load_stations(self, path: Optional[str] = None) -> None:
        """
        解析 12306 原始 JS，提取站点及所属城市信息。
        自动检测并修复字段顺序异常的数据行。
        """
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "station_name.js")

        if not os.path.exists(path):
            logging.warning(f"车站数据文件不存在: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        m = re.search(r"var station_names ?= ?'(.*?)';", content)
        if not m:
            m = re.search(r"'(@[^']+)';", content)
        if not m:
            logging.error("未能解析到站点 JS 内容")
            return

        data = m.group(1)
        stations_raw = [s for s in data.split('@') if s]
        result = []

        for st in stations_raw:
            parts = st.split('|')
            if len(parts) < 8:
                logging.warning(f"字段数异常，跳过：{st}")
                continue

            name = parts[1].strip()
            code = parts[2].strip()
            pinyin = parts[3].strip()
            py_short = parts[4].strip()
            num = parts[5].strip()
            city = parts[7].strip()

            def is_code(val):
                return val.isalpha() and val.isupper() and len(val) == 3
            def is_pinyin(val):
                return val.isalpha() and val.islower() and len(val) >= 2
            def is_py_short(val):
                return val.isalpha() and val.islower() and 1 <= len(val) <= 8

            if not is_code(code):
                for idx in range(1, min(5, len(parts))):
                    if is_code(parts[idx]):
                        code = parts[idx]
                        break

            if not is_pinyin(pinyin):
                for idx in range(1, min(6, len(parts))):
                    if is_pinyin(parts[idx]):
                        pinyin = parts[idx]
                        break

            if not is_py_short(py_short):
                for idx in range(1, min(7, len(parts))):
                    if is_py_short(parts[idx]):
                        py_short = parts[idx]
                        break

            result.append(Station(name, code, pinyin, py_short, num, city))

        self.stations = result
        logging.info(f"已加载 {len(self.stations)} 个车站")

    def get_station_by_name(self, name: str) -> Optional[Station]:
        name = name.strip()
        if name.endswith("站") and len(name) > 2:
            name = name[:-1]
        for s in self.stations:
            if s.name.strip() == name:
                return s
        return None

    def get_station_by_code(self, code: str) -> Optional[Station]:
        for s in self.stations:
            if s.code == code:
                return s
        return None

    def search_stations(self, query: str, limit: int = 10) -> StationSearchResult:
        query = query.strip().lower()
        if query.endswith("站") and len(query) > 2:
            query = query[:-1]
        results = []
        matched_ids = set()

        # 1. 精确匹配
        for s in self.stations:
            if (query == s.name.strip().lower()
                or query == s.code.lower()
                or query == s.pinyin.lower()
                or query == s.py_short.lower()):
                results.append(s)
                matched_ids.add(id(s))
                if len(results) >= limit:
                    return StationSearchResult(results)

        # 2. 模糊匹配（含 city）
        for s in self.stations:
            if id(s) in matched_ids:
                continue
            if (query in s.name.strip().lower()
                or query in s.pinyin.lower()
                or query in s.py_short.lower()
                or query in s.code.lower()
                or (s.city and query in s.city.lower())):
                results.append(s)
                if len(results) >= limit:
                    break

        return StationSearchResult(results)

    def get_station_code(self, query: str) -> Optional[str]:
        """城市名/站名/三字码 → 三字码"""
        if not query:
            return None
        q = query.strip()
        if q.endswith("站") and len(q) > 2:
            q = q[:-1]

        for s in self.stations:
            if q == s.name:
                return s.code
        for s in self.stations:
            if q == s.code:
                return s.code
        return None


# 全局单例，应用启动时初始化
station_service = StationService()
