"""
JSON 提取工具 — 从 LLM 返回的文本中提取 JSON 字符串

所有需要从文本中提取 JSON 的模块统一导入此函数，消除重复代码。
"""


def extract_json(text: str) -> str | None:
    """
    从 LLM 返回的文本中提取 JSON 字符串

    支持三种格式：
    1. ```json ... ``` 代码块
    2. ``` ... ``` 无语言标记的代码块
    3. 直接的 { ... } JSON（括号计数法精确匹配嵌套）

    Args:
        text: 可能包含 JSON 的原始文本

    Returns:
        提取到的 JSON 字符串，未找到时返回 None
    """
    if not text:
        return None

    # 情况1：```json ... ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # 情况2：``` ... ```
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # 情况3：括号计数法精确匹配嵌套 JSON
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
        return text[start:end] if end > start else None

    return None
