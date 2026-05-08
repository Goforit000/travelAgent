"""
Agent 基类 — 所有 Agent 的抽象父类

设计原则：
1. 有些Agent 是一个独立的 ReAct 循环：
   - 思考 (Think)：LLM 分析当前状态，决定调用工具 or 给出最终回答
   - 行动 (Act)：执行工具调用，将结果追加到消息历史
   - 观察 (Observe)：LLM 看到工具结果，继续决策
   - 循环直到 LLM 给出最终回答或达到 max_steps 上限

2. 不使用 LangGraph 预置的 create_react_agent：
   - 手动控制每一步，保持架构透明度
   - 每个 Agent 的循环完全可控（步骤数、错误处理、日志输出）

3. 工具调用链路：
   LLM 输出 → tool_calls → _execute_tool → ToolMessage → messages.append → LLM 再次决策
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
    """当 Agent 内部 ReAct 循环超过 max_steps 时抛出"""
    def __init__(self, agent_name: str, max_steps: int):
        super().__init__(f"Agent '{agent_name}' 超过最大步骤数 {max_steps}")
        self.agent_name = agent_name
        self.max_steps = max_steps


class BaseAgent(ABC):
    """
    Agent 基类 — 所有 Agent 的抽象父类

    子类必须实现：
    - name: str            Agent 标识名称
    - description: str     Agent 职责描述（供 Supervisor 路由参考）
    - system_prompt: str   系统提示词（定义角色和行为）

    子类可选覆盖：
    - tools() → list       返回该 Agent 可调用的工具列表
    - max_steps: int       内部 ReAct 循环最大步数（默认 5）
    - temperature: float   LLM 温度参数（默认 0.7）

    子类必须实现：
    - run(state) → dict    执行入口，返回 State 更新字典
    """

    # ===== 必须由子类覆盖 =====
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 的唯一标识名称，如 'poi_agent'"""
        ...
    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 职责描述，供 Supervisor 路由决策时参考"""
        ...
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """系统提示词，定义 Agent 的角色、行为规范和输出格式"""
        ...

    # ===== 可选覆盖 =====
    @property
    def tools(self) -> Sequence[BaseTool]:
        """该 Agent 可调用的工具列表，子类覆盖以绑定工具"""
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
        """LLM 最大输出 token 数，子类可按需覆盖"""
        from app.tools.llm import DEFAULT_MAX_TOKENS
        return DEFAULT_MAX_TOKENS

    @property
    def request_timeout(self) -> int:
        """HTTP 请求超时秒数，子类可按需覆盖"""
        from app.tools.llm import DEFAULT_REQUEST_TIMEOUT
        return DEFAULT_REQUEST_TIMEOUT

    # ===== 模型获取 =====
    def _get_model(self) -> ChatOpenAI:
        """
        获取绑定了工具的 ChatOpenAI 实例

        如果 tools 为空列表，则获取不绑定工具的普通 model；
        否则为 model.bind_tools(tools) 后的实例。
        max_tokens / request_timeout 由子类的属性决定。
        """
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
        Agent 主入口 — 执行 ReAct 循环
        流程：
        1. 从 state 中提取相关字段，构造初始 messages
        2. 进入 ReAct 循环：
           a. 调用 LLM
           b. 如果有 tool_calls → 执行工具 → 追加结果 → 继续循环
           c. 如果无 tool_calls → 提取最终回答 → 跳出循环
        3. 如果达到 max_steps 上限 → 强制返回当前结果

        Args:
            state: 当前工作流共享状态

        Returns:
            需要更新的 State 字段字典

        Raises:
            MaxStepsExceededError: 当循环步数超过 max_steps
        """
        model = self._get_model()
        messages = self._build_initial_messages(state)

        print(f"\n{'=' * 50}")
        print(f"🤖 {self.name} 开始执行")
        print(f"{'=' * 50}")

        for step in range(1, self.max_steps + 1):
            print(f"\n  [{self.name}] 步骤 {step}/{self.max_steps}")

            # 调用 LLM
            response = model.invoke(messages)
            messages.append(response)

            # 检查是否有 tool_calls
            tool_calls = getattr(response, "tool_calls", None)

            if tool_calls:
                # ——— 有工具调用：执行工具并追加结果 ———
                print(f"  [{self.name}] LLM 请求调用 {len(tool_calls)} 个工具")
                tool_messages = self._execute_tool_calls(tool_calls)
                messages.extend(tool_messages)

                # 继续循环，让 LLM 看到工具结果后再次决策
                continue

            else:
                # ——— 无工具调用：LLM 给出了最终回答 ———
                content = response.content if hasattr(response, "content") else str(response)
                print(f"  [{self.name}] LLM 给出最终回答 ({len(content)} 字符)")
                print(f"{'=' * 50}\n")

                return self._parse_final_output(state, content, messages)

        # ——— 超出 max_steps ———
        print(f"  ⚠️ [{self.name}] 超出最大步骤数 {self.max_steps}，强制终止")
        print(f"{'=' * 50}\n")

        raise MaxStepsExceededError(self.name, self.max_steps)

    # ===== 消息构建 =====
    def _build_initial_messages(self, state: TripState) -> list[BaseMessage]:
        """
        构造 ReAct 循环的初始消息列表

        默认实现：
        1. SystemMessage（Agent 的角色定义）
        2. 从 state 中提取相关数据作为 HumanMessage（上下文）
        3. 避免重复追加 state 中已有的 messages

        子类可以覆盖此方法以自定义消息构造逻辑。
        """
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
        ]

        # 构造上下文 HumanMessage
        context = self._build_context_message(state)
        if context:
            messages.append(HumanMessage(content=context))

        return messages

    def _build_context_message(self, state: TripState) -> str:
        """
        构造上下文消息 — 子类覆盖以提供自定义的上下文字符串
        默认返回空字符串（不追加额外上下文）。
        """
        return ""

    # ===== 工具执行 =====

    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolMessage]:
        """
        执行 LLM 请求的工具调用

        需要处理 LangChain 工具调用的两种格式：
        1. dict 格式：{"name": "tool_name", "args": {...}, "id": "call_xxx"}
        2. ToolCall 对象：有 .name, .args, .id 属性

        Args:
            tool_calls: LLM 返回的工具调用列表

        Returns:
            对应的 ToolMessage 列表
        """
        tool_messages: list[ToolMessage] = []
        tool_map = {tool.name: tool for tool in self.tools}

        for tc in tool_calls:
            # 兼容两种格式
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
                    # 将结果转为字符串（有些工具返回 dict）
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
        解析 LLM 的最终输出，转换为 State 更新字典
        子类必须覆盖此方法以实现自己的输出提取逻辑。
        默认实现：返回空字典（不更新任何 State 字段）。

        Args:
            state: 当前工作流状态
            llm_content: LLM 最终回答的文本内容
            messages: 完整的消息历史

        Returns:
            State 更新字典
        """
        return {}

    # ===== 辅助方法 =====

    def _get_attr_from_state(self, state: TripState, key: str, default=None):
        """安全地从 state 中获取字段值"""
        return state.get(key, default)

    def _extract_json(self, text: str) -> str | None:
        """从 LLM 返回的文本中提取 JSON 字符串 — 委托给共享工具函数"""
        from app.utils.json_utils import extract_json
        return extract_json(text)

    def __repr__(self) -> str:
        tools_count = len(self.tools) if self.tools else 0
        return f"<{self.__class__.__name__} name={self.name} tools={tools_count}>"
