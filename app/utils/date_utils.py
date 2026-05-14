"""
日期工具 — 来源：mcp-server-12306/src/mcp_12306/utils/date_utils.py
"""

from datetime import datetime, date, timedelta
import re


def validate_date(date_str: str) -> bool:
    """验证日期格式 YYYY-MM-DD"""
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(pattern, date_str):
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_date_not_past(date_str: str) -> tuple[bool, str]:
    """
    验证日期格式并检查是否在有效范围内（今天到 14 天后）。
    12306 提前 14 天售票。
    返回: (是否有效, 错误信息或空字符串)
    """
    if not validate_date(date_str):
        return False, "日期格式错误，请使用 YYYY-MM-DD 格式"

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        query_date = dt.date()
        today = date.today()
        max_date = today + timedelta(days=14)

        if query_date < today:
            return False, f"出发日期不能早于今天（{today.strftime('%Y-%m-%d')}），12306 无法查询历史日期"
        if query_date > max_date:
            return False, f"出发日期不能晚于 {max_date.strftime('%Y-%m-%d')}，12306 仅支持提前 14 天购票"

        return True, ""
    except Exception as e:
        return False, f"日期校验失败: {str(e)}"
