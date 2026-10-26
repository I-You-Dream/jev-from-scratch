"""三种决策原语的读出/评分逻辑。

技术路线：在 Qwen3.5-4B 上做**候选似然读出**（零训练），
将候选项的对数似然经长度归一化 + softmax 映射为三种原语的概率输出。

详细设计见 docs/design.md §4。当前为占位实现，Phase 2 在云服务器上完成。
"""

from __future__ import annotations

from .schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


def readout_noul(model, tokenizer, state: str | dict, question: NoulQuestion) -> NoulAnswer:
    """读出是/否命题为真的概率。

    实现思路：视作 2 选项 Choice（"是"/"否"），取"是"侧归一化概率为 p_true。
    """
    raise NotImplementedError("Phase 2 实现：候选似然读出 + 温度缩放")


def readout_choice(model, tokenizer, state: str | dict, question: ChoiceQuestion) -> ChoiceAnswer:
    """读出 N 选 1 分类结果。

    实现思路：对每个选项构造简短补全，取对数似然 → 长度归一化 → softmax。
    """
    raise NotImplementedError("Phase 2 实现：候选似然读出 + 归一化")


def readout_score(model, tokenizer, state: str | dict, question: ScoreQuestion) -> ScoreAnswer:
    """读出有序等级评分。

    实现思路：对每个等级标签做似然读出并归一化，再按序号加权得 expected_score。
    """
    raise NotImplementedError("Phase 2 实现：等级似然读出 + 期望插值")


def length_normalize(log_likelihood: float, n_tokens: int) -> float:
    """长度归一化的对数似然，消除长候选项的系统性偏差。"""
    raise NotImplementedError("Phase 2 实现")


def softmax_probs(scores: list[float], temperature: float = 1.0) -> list[float]:
    """带温度的 softmax，把打分转成归一化概率。"""
    raise NotImplementedError("Phase 2 实现")
