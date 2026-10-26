"""训练样本构建：raw 中间格式 → 项目契约格式（state + questions + labels）。

本模块是 build_dataset.py（批量转换）与 finetune.py（训练/推理渲染）的共享层，
核心约定：**训练与推理必须用同一套渲染函数**，否则模型看到的输入分布会漂移。

三块能力：
    1. 负采样  —— 给 choice 构造「1 正确 + k 干扰」候选集，含低置信度样本
    2. 模板    —— 多套措辞轮换，降低对单一表达的过拟合
    3. 渲染    —— state + question → 模型输入文本；label → 监督目标文本
"""

from __future__ import annotations

import random
from typing import Sequence

from .schema import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

# ============================================================
# 1. 渲染模板
# ============================================================

# 模型输入的组装模板。多套轮换可降低措辞过拟合（design.md 提到的模板敏感性）。
INPUT_TEMPLATES = [
    "状态：\n{state}\n\n问题：{prompt}\n{body}\n答案：",
    "【状态】\n{state}\n\n【问题】{prompt}\n{body}\n【答案】",
    "以下是需要判断的内容。\n{state}\n\n{prompt}\n{body}\n请作答：",
]

# choice 的问题措辞，按数据源分组（build_dataset 阶段随机选一套）
CHOICE_PROMPTS = {
    "banking77": ["该请求属于哪类意图？", "这条客服消息对应哪个意图？", "应归入哪一类请求？"],
    "ag_news": ["该新闻属于哪个主题？", "这则报道的主题是？", "新闻应归入哪个类别？"],
    "yahoo_topics": ["该问题属于哪个主题？", "这个提问对应哪个话题领域？", "应归入哪一类问题？"],
    "goemotions": ["该评论主要表达了哪种情绪？", "这段话的情绪是？", "最贴近的情绪标签是？"],
}

# score 的等级标签（有序，从低到高）
YELP_LEVELS = ["1星", "2星", "3星", "4星", "5星"]

# 「其他」选项：故意构造无正确候选的样本，逼模型给出低置信度分布
OTHER_OPTION = "其他"


def render_state(state: str | dict) -> str:
    """把 state 渲染成文本片段。

    dict 型 state 的字段按 key 排序拼接，保证同一输入永远渲染出同一文本
    （可复现性要求，见 engine.format_state 的设计说明）。
    """
    if isinstance(state, str):
        return state.strip()
    parts = []
    for key in sorted(state.keys()):
        value = str(state[key]).strip()
        if value:
            parts.append(f"{key}：{value}")
    return "\n".join(parts)


def _format_options(options: Sequence[str], numbered: bool = True) -> str:
    """把候选列表渲染成选项块。numbered=True 用 A/B/C 编号。"""
    if numbered:
        lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    else:
        lines = [f"- {opt}" for opt in options]
    return "选项：\n" + "\n".join(lines)


def _format_levels(levels: Sequence[str]) -> str:
    """把有序等级渲染成等级块。"""
    lines = [f"{i + 1}. {lv}" for i, lv in enumerate(levels)]
    return "等级：\n" + "\n".join(lines)


def render_input(state: str | dict, question: Question, template_idx: int = 0) -> str:
    """渲染模型输入文本（训练与推理共用，务必保持一致）。

    Args:
        state: 非结构化上下文
        question: 类型化问题（noul / choice / score）
        template_idx: 输入模板序号，多套轮换降低过拟合
    """
    template = INPUT_TEMPLATES[template_idx % len(INPUT_TEMPLATES)]
    prompt = question.prompt

    if isinstance(question, ChoiceQuestion):
        body = _format_options(question.options)
    elif isinstance(question, ScoreQuestion):
        body = _format_levels(question.levels)
    else:  # NoulQuestion：只有命题，无候选列表
        body = ""

    return template.format(state=render_state(state), prompt=prompt, body=body).strip()


def target_text(question: Question, label) -> str:
    """监督目标文本：正确答案本身（模型要学会给它更高的似然）。

    noul 的标签是 bool，渲染成「是 / 否」；choice/score 的标签即选项/等级文本。
    """
    if isinstance(question, NoulQuestion):
        return "是" if label else "否"
    return str(label)


# ============================================================
# 2. 负采样
# ============================================================


def sample_options(
    correct: str,
    pool: Sequence[str],
    n_distractors: int = 5,
    other_rate: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[list[str], str]:
    """为 choice 样本构造候选集：1 个正确 + n_distractors 个干扰。

    以 other_rate 的概率返回「全干扰 + 其他」并把标签设为「其他」——
    这类样本教模型：真答案不在候选里时要把概率压低，对应 design.md §2.2
    提到的最大设计风险。

    Returns:
        (options, label)：options 顺序已打乱；label 必定在 options 内
    """
    rng = rng or random.Random()
    pool = [x for x in pool if x != correct]

    # 低置信度样本：正确答案缺席，只能选「其他」
    if pool and rng.random() < other_rate:
        distractors = rng.sample(pool, min(n_distractors, len(pool)))
        options = distractors + [OTHER_OPTION]
        rng.shuffle(options)
        return options, OTHER_OPTION

    n = min(n_distractors, len(pool))
    distractors = rng.sample(pool, n) if n > 0 else []
    options = [correct] + distractors
    rng.shuffle(options)
    return options, correct


# ============================================================
# 3. 样本构建
# ============================================================


def build_choice_sample(
    raw: dict,
    label_pool: Sequence[str],
    sample_id: str,
    rng: random.Random,
    n_distractors: int = 5,
    other_rate: float = 0.05,
) -> dict | None:
    """一条 choice 原始记录 → 训练样本（state + questions + labels）。"""
    prompts = CHOICE_PROMPTS.get(raw["source"], ["该样本属于哪个类别？"])
    prompt = rng.choice(prompts)

    options, label = sample_options(
        correct=raw["label"],
        pool=label_pool,
        n_distractors=n_distractors,
        other_rate=other_rate,
        rng=rng,
    )
    question = ChoiceQuestion(id="q1", prompt=prompt, options=options)
    sample = {
        "id": sample_id,
        "state": raw["text"],
        "questions": [question.model_dump()],
        "labels": {"q1": label},
    }
    return _validated(sample)


def build_score_sample(raw: dict, sample_id: str, rng: random.Random) -> dict | None:
    """一条 score 原始记录 → 训练样本。levels 固定为 1–5 星有序等级。"""
    prompts = ["该评论的评分是几星？", "这条评论对应哪个星级？", "给出的评分等级是？"]
    question = ScoreQuestion(
        id="q1",
        prompt=rng.choice(prompts),
        levels=list(YELP_LEVELS),
    )
    sample = {
        "id": sample_id,
        "state": raw["text"],
        "questions": [question.model_dump()],
        "labels": {"q1": raw["label"]},
    }
    return _validated(sample)


def build_noul_sample(raw: dict, sample_id: str, rng: random.Random) -> dict | None:
    """一条 noul 原始记录 → 训练样本。

    boolq：标签直接是是非；snli 只保留蕴含(真)/矛盾(假)，
    中立样本本就该让模型不确定，硬标 true/false 反而教坏校准，直接丢弃。
    """
    if raw["source"] == "boolq":
        label = raw["label"] == "是"
        prompt = "所述问题的答案是肯定的吗？"
    elif raw["source"] == "snli":
        if raw["label"] == "neutral":
            return None
        label = raw["label"] == "entailment"
        prompt = "根据前提，假设成立吗？"
    else:
        return None

    question = NoulQuestion(id="q1", prompt=prompt)
    sample = {
        "id": sample_id,
        "state": raw["text"],
        "questions": [question.model_dump()],
        "labels": {"q1": label},
    }
    return _validated(sample)


def _validated(sample: dict) -> dict | None:
    """用契约层校验样本结构；不合规返回 None，避免脏数据进训练集。"""
    try:
        DecisionRequest.model_validate(
            {"state": sample["state"], "questions": sample["questions"]}
        )
    except Exception:
        return None
    return sample


__all__ = [
    "CHOICE_PROMPTS",
    "INPUT_TEMPLATES",
    "OTHER_OPTION",
    "YELP_LEVELS",
    "build_choice_sample",
    "build_noul_sample",
    "build_score_sample",
    "render_input",
    "render_state",
    "sample_options",
    "target_text",
]
