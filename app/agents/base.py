"""
Agent 基类层 — BaseAgent（最小接口） + ReActAgent（ReAct 循环）

设计原则：
- BaseAgent：只定义 name、description、run() 三个核心接口，适配所有 Agent
- ReActAgent：在 BaseAgent 基础上增加 ReAct 循环、工具绑定、LLM 调用能力

继承关系：
  BaseAgent              ← Weather、Transport、Planner（不需要 ReAct 的 Agent）
    └── ReActAgent       ← POI、Hotel、Budget（需要 ReAct 循环的 Agent）
"""

import json
from abc import ABC, abstractmethod
from typing import Sequence
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from app.graph.state import TripState
from app.tools.llm import get_model_with_tools


class MaxStepsExceededError(Exception):
    """当 ReActAgent 内部循环超过 max_steps 时抛出"""
    def __init__(self, agent_name: str, max_steps: int):
        super().__init__(f"Agent '{agent_name}' 超过最大步骤数 {max_steps}")
        self.agent_name = agent_name
        self.max_steps = max_steps


class BaseAgent(ABC):
    """
    Agent 最小接口 — 所有 Agent 的抽象父类

    子类必须实现：
    - name: str            Agent 标识名称
    - description: str     Agent 职责描述
    - run(state) → dict    执行入口，返回 State 更新字典
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 的唯一标识名称，如 'poi_agent'"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 职责描述"""
        ...

    @abstractmethod
    def run(self, state: TripState) -> dict:
        """
        Agent 主入口 — 子类各自实现执行逻辑

        可以是 ReAct 循环、单次 LLM 调用、直接 API 调用、确定性分发等。
        返回需要更新的 State 字段字典。
        """
        ...

    # ===== 辅助方法 =====

    def _get_attr_from_state(self, state: TripState, key: str, default=None):
        """安全地从 state 中获取字段值"""
        return state.get(key, default)

    def _extract_json(self, text: str) -> str | None:
        """从 LLM 返回的文本中提取 JSON 字符串"""
        from app.utils.json_utils import extract_json
        return extract_json(text)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


class ReActAgent(BaseAgent):
    """
    ReAct 循环 Agent — 在 BaseAgent 基础上增加 LLM + 工具调用能力

    子类必须额外实现：
    - system_prompt: str   系统提示词（定义角色和行为）

    子类可选覆盖：
    - tools() → list       返回可调用的工具列表
    - max_steps: int       内部 ReAct 循环最大步数（默认 5）
    - temperature: float   LLM 温度参数（默认 0.7）
    - max_tokens: int      LLM 最大输出 token 数
    - request_timeout: int HTTP 请求超时秒数
    - _build_context_message(state) → str  构造上下文消息
    - _parse_final_output(state, content, messages) → dict  解析最终输出
    """

    # ===== 必须由子类覆盖 =====
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """系统提示词，定义 Agent 的角色、行为规范和输出格式"""
        ...

    # ===== 可选覆盖 =====
    @property
    def tools(self) -> Sequence[BaseTool]:
        """该 Agent 可调用的工具列表"""
        return []

    @property
    def max_steps(self) -> int:
        """内部 ReAct 循环最大步数，防止 LLM 死循环"""
        return 5

    @property
    def temperature(self) -> float:
        """LLM 温度参数"""
        return 0.7

    @property
    def max_tokens(self) -> int:
        """LLM 最大输出 token 数"""
        from app.tools.llm import DEFAULT_MAX_TOKENS
        return DEFAULT_MAX_TOKENS

    @property
    def request_timeout(self) -> int:
        """HTTP 请求超时秒数"""
        from app.tools.llm import DEFAULT_REQUEST_TIMEOUT
        return DEFAULT_REQUEST_TIMEOUT

    # ===== 模型获取 =====
    def _get_model(self) -> ChatOpenAI:
        """获取绑定了工具的 ChatOpenAI 实例"""
        tool_list = list(self.tools) if self.tools else None
        return get_model_with_tools(
            tools=tool_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )

    # ===== ReAct 循环核心 =====
    def run(self, state: TripState) -> dict:
        """
        ReAct 循环主入口

        流程：
        1. 从 state 中提取相关字段，构造初始 messages
        2. ReAct 循环：调用 LLM → 有 tool_calls 则执行工具并继续 → 无则提取最终回答
        3. 达到 max_steps 上限 → 抛出 MaxStepsExceededError
        """
        model = self._get_model()
        messages = self._build_initial_messages(state)

        print(f"\n{'=' * 50}")
        print(f"🤖 {self.name} 开始执行")
        print(f"{'=' * 50}")

        for step in range(1, self.max_steps + 1):
            print(f"\n  [{self.name}] 步骤 {step}/{self.max_steps}")

            response = model.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None)

            if tool_calls:
                # ——— 有工具调用：执行工具并追加结果 ———
                print(f"  [{self.name}] LLM 请求调用 {len(tool_calls)} 个工具")
                tool_messages = self._execute_tool_calls(tool_calls)
                messages.extend(tool_messages)
                continue
            else:
                # ——— 有工具调用：执行工具并追加结果 ———
                content = response.content if hasattr(response, "content") else str(response)
                print(f"  [{self.name}] LLM 给出最终回答 ({len(content)} 字符)")
                print(f"{'=' * 50}\n")
                return self._parse_final_output(state, content, messages)

        print(f"  ⚠️ [{self.name}] 超出最大步骤数 {self.max_steps}，强制终止")
        print(f"{'=' * 50}\n")
        raise MaxStepsExceededError(self.name, self.max_steps)

    # ===== 消息构建 =====
    def _build_initial_messages(self, state: TripState) -> list[BaseMessage]:
        """
        构造 ReAct 循环的初始消息列表

        默认：SystemMessage（角色定义） + HumanMessage（上下文数据）
        子类可覆盖以自定义消息构造逻辑。
        """
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
        ]
        context = self._build_context_message(state)
        if context:
            messages.append(HumanMessage(content=context))
        return messages

    def _build_context_message(self, state: TripState) -> str:
        """构造上下文消息 — 子类覆盖以提供自定义的上下文数据"""
        return ""

    # ===== 工具执行 =====
    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolMessage]:
        """
        执行 LLM 请求的工具调用

        兼容两种格式：
        1. dict 格式：{"name": "tool_name", "args": {...}, "id": "call_xxx"}
        2. ToolCall 对象：有 .name, .args, .id 属性
        """
        tool_messages: list[ToolMessage] = []
        tool_map = {tool.name: tool for tool in self.tools}

        for tc in tool_calls:
            if isinstance(tc, dict):
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                call_id = tc.get("id", "")
            else:
                tool_name = getattr(tc, "name", "")
                tool_args = getattr(tc, "args", {})
                call_id = getattr(tc, "id", "")

            print(f"  [{self.name}] → 执行工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})")

            try:
                tool = tool_map.get(tool_name)
                if tool is None:
                    result = f"错误：未找到工具 '{tool_name}'，可用工具：{list(tool_map.keys())}"
                    print(f"  [{self.name}] ✗ 工具未找到: {tool_name}")
                else:
                    result = tool.invoke(tool_args)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False, indent=2)
                    print(f"  [{self.name}] ✓ 工具执行成功 ({len(str(result))} 字符)")

            except Exception as e:
                result = f"工具执行异常: {str(e)}"
                print(f"  [{self.name}] ✗ 工具执行异常: {e}")

            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=call_id,
                name=tool_name,
            ))

        return tool_messages

    # ===== 输出解析 =====
    def _parse_final_output(
        self,
        state: TripState,
        llm_content: str,
        messages: list[BaseMessage],
    ) -> dict:
        """
        解析 LLM 最终输出，转换为 State 更新字典
        子类必须覆盖此方法以实现自己的输出提取逻辑。
        """
        return {}

    def __repr__(self) -> str:
        tools_count = len(self.tools) if self.tools else 0
        return f"<{self.__class__.__name__} name={self.name} tools={tools_count}>"
