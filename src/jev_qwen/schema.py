"""类型化问题与答案的契约定义。

对应 Jev 的三种决策原语：noul / choice / score。
本模块是全项目的唯一事实源：engine、primitives、scripts 都依赖这里的类型。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ============================================================
# 1. 通用类型
# ============================================================


class PrimitiveType(str, Enum):
    """三种决策原语。"""

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


# 概率：0 到 1 闭区间
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


def _require_text(value: str) -> str:
    """prompt/标签必须是非空白字符串。"""
    if not value.strip():
        raise ValueError("文本字段不能为空或纯空白")
    return value.strip()


class Meta(BaseModel):
    """响应元信息，用于可观测性与治理留痕。"""

    model: str = Field(description="决策模型标识，如 jev-qwen-readout-v0")
    latency_ms: float | None = Field(default=None, description="端到端延迟（毫秒）")
    calibration: str | None = Field(
        default=None, description="使用的校准方法，如 temperature_scaling"
    )


# ============================================================
# 2. 问题定义
# ============================================================


class NoulQuestion(BaseModel):
    """是/否判断问题。

    p_true≈0.5 表示"是/否各半"的不确定，不是"程度中等"。
    """

    type: Literal[PrimitiveType.NOUL] = PrimitiveType.NOUL
    id: str = Field(description="问题唯一标识")
    prompt: str = Field(min_length=1, description="自然语言命题，如：该工单在请求退款吗？")

    @model_validator(mode="after")
    def _check_prompt(self) -> "NoulQuestion":
        object.__setattr__(self, "prompt", _require_text(self.prompt))
        return self


class ChoiceQuestion(BaseModel):
    """N 选 1 分类/路由问题。

    正确答案必须落在 options 内；若真实类别缺失，概率仍被迫落在剩余选项上。
    """

    type: Literal[PrimitiveType.CHOICE] = PrimitiveType.CHOICE
    id: str
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, description="互斥且穷尽的候选选项")

    @model_validator(mode="after")
    def _check_options(self) -> "ChoiceQuestion":
        """prompt 与选项必须非空，且选项互不重复。"""
        object.__setattr__(self, "prompt", _require_text(self.prompt))
        cleaned = [opt.strip() for opt in self.options]
        if any(not opt for opt in cleaned):
            raise ValueError("选项不能为空字符串")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"选项存在重复: {self.options}")
        return self


class ScoreQuestion(BaseModel):
    """有序等级评分问题。

    levels 有序但不一定等距；expected_score 是按序号概率加权的期望值。
    """

    type: Literal[PrimitiveType.SCORE] = PrimitiveType.SCORE
    id: str
    prompt: str = Field(min_length=1)
    levels: list[str] = Field(min_length=2, description="有序等级标签，从低到高")

    @model_validator(mode="after")
    def _check_levels(self) -> "ScoreQuestion":
        """prompt 与等级标签必须非空，且标签互不重复。"""
        object.__setattr__(self, "prompt", _require_text(self.prompt))
        cleaned = [lv.strip() for lv in self.levels]
        if any(not lv for lv in cleaned):
            raise ValueError("等级标签不能为空字符串")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"等级标签存在重复: {self.levels}")
        return self


# 统一的问题联合类型：按 type 字段自动解析
Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


# ============================================================
# 3. 答案定义
# ============================================================


class NoulAnswer(BaseModel):
    """是/否判断结果：命题为真的概率。"""

    type: Literal[PrimitiveType.NOUL] = PrimitiveType.NOUL
    id: str
    p_true: Probability = Field(description="命题为真的概率 ∈ [0,1]")


class ChoiceAnswer(BaseModel):
    """分类/路由结果：选中项 + 各选项概率 + 置信度。

    confidence 表示概率分布的集中程度，不代表整体流程正确性。
    """

    type: Literal[PrimitiveType.CHOICE] = PrimitiveType.CHOICE
    id: str
    selected: str = Field(description="概率最高的选项")
    probabilities: dict[str, Probability] = Field(description="每个选项的概率，总和为 1")
    confidence: Probability = Field(description="分布集中程度，通常取 max 概率")

    @model_validator(mode="after")
    def _check_probs(self) -> "ChoiceAnswer":
        """概率必须归一化，且 selected 必须落在 probabilities 的键中。"""
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"选项概率之和应为 1，实际为 {total}")
        if self.selected not in self.probabilities:
            raise ValueError(f"selected {self.selected!r} 不在 probabilities 中")
        return self


class ScoreAnswer(BaseModel):
    """评分结果：期望分数 + 各等级概率 + 置信度。"""

    type: Literal[PrimitiveType.SCORE] = PrimitiveType.SCORE
    id: str
    expected_score: float = Field(description="按等级序号概率加权的期望分")
    probabilities: dict[str, Probability] = Field(description="每个等级的概率，总和为 1")
    confidence: Probability = Field(description="分布集中程度")

    @model_validator(mode="after")
    def _check_probs(self) -> "ScoreAnswer":
        """概率必须归一化。"""
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"等级概率之和应为 1，实际为 {total}")
        return self


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


# ============================================================
# 4. 请求 / 响应信封
# ============================================================


class DecisionRequest(BaseModel):
    """决策请求：一段状态 + 一组带类型的问题。

    同一 state 上的多个问题并行评估。
    """

    state: str | dict = Field(description="非结构化上下文：纯文本，或文本字段构成的对象")
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "DecisionRequest":
        """问题 id 必须唯一，便于答案对齐。"""
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"问题 id 存在重复: {ids}")
        return self


class DecisionResponse(BaseModel):
    """决策响应：与请求问题一一对应的类型化答案。"""

    answers: list[Answer] = Field(min_length=1)
    meta: Meta

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "DecisionResponse":
        """答案 id 必须唯一。"""
        ids = [a.id for a in self.answers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"答案 id 存在重复: {ids}")
        return self

    def get(self, question_id: str) -> Answer:
        """按问题 id 取答案，便于调用方代码消费。"""
        for ans in self.answers:
            if ans.id == question_id:
                return ans
        raise KeyError(f"未找到 id 为 {question_id!r} 的答案")


__all__ = [
    "PrimitiveType",
    "Probability",
    "Meta",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "Question",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "Answer",
    "DecisionRequest",
    "DecisionResponse",
]
