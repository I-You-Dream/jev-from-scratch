"""决策引擎：state + questions → typed answers。

把 primitives 的单问题读出封装成一次并行调用，输出符合 schema 的 DecisionResponse。
当前为占位实现，Phase 2 在云服务器上完成。
"""

from __future__ import annotations

from .schema import DecisionRequest, DecisionResponse, Meta


class JevEngine:
    """基于 Qwen3.5-4B 的决策引擎。

    对外契约与 TypeSafe Jev API 形状对齐，便于后续替换为真实服务。
    """

    def __init__(self, model_path: str, device: str = "auto", temperature: float = 1.0):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        # Phase 1 在云服务器上加载模型
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """加载模型与分词器（4bit 量化可选）。"""
        raise NotImplementedError("Phase 1 实现：transformers 加载 Qwen3.5-4B")

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """对同一 state 上的一组问题并行评估，返回类型化答案。

        需保证：答案 id 与问题 id 对齐，类型一致，概率已归一化。
        """
        raise NotImplementedError("Phase 2 实现：批量读出 + 组装 DecisionResponse")

    def decide_batch(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        """批量决策，用于评测与吞吐测试。"""
        raise NotImplementedError("Phase 3 实现")


def format_state(state: str | dict) -> str:
    """把 state 渲染成模型可读的 prompt 片段。

    dict 型 state 的字段拼接顺序需稳定，保证读出可复现。
    """
    raise NotImplementedError("Phase 2 实现")
