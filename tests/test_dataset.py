"""dataset 样本构建的单元测试。

纯逻辑测试，不依赖 torch / 模型，验证：
    - 负采样：候选集包含正确标签、「其他」低置信度样本
    - 渲染：state 稳定输出、三种原语的输入格式
    - 契约：构建出的样本能通过 schema 校验

运行（服务器或本地均可）：
    PYTHONPATH=src python -m pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import random

import pytest

from jev_qwen.dataset import (
    OTHER_OPTION,
    YELP_LEVELS,
    build_choice_sample,
    build_noul_sample,
    build_score_sample,
    render_input,
    render_state,
    sample_options,
    target_text,
)
from jev_qwen.schema import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)


# ============================================================
# 渲染
# ============================================================


class TestRenderState:
    def test_plain_text(self):
        assert render_state(" 一段文本 ") == "一段文本"

    def test_dict_fields_sorted_stable(self):
        # 字段拼接顺序必须稳定，同一输入永远渲染出同一文本（可复现性）
        a = render_state({"body": "正文", "title": "标题"})
        b = render_state({"title": "标题", "body": "正文"})
        assert a == b
        assert "title：标题" in a
        assert a.index("body") < a.index("title")  # 按 key 排序：body 在前


class TestRenderInput:
    def _choice(self):
        return ChoiceQuestion(id="q1", prompt="属于哪类？", options=["A", "B", "C"])

    def test_choice_contains_options(self):
        text = render_input("状态内容", self._choice(), template_idx=0)
        assert "状态内容" in text
        assert "属于哪类？" in text
        assert "A. A" in text and "C. C" in text

    def test_score_contains_levels(self):
        q = ScoreQuestion(id="q1", prompt="评分？", levels=YELP_LEVELS)
        text = render_input("内容", q)
        assert "1星" in text and "5星" in text

    def test_noul_has_no_options_block(self):
        q = NoulQuestion(id="q1", prompt="成立吗？")
        text = render_input("内容", q)
        assert "成立吗？" in text
        assert "选项：" not in text

    def test_template_rotation_changes_wording(self):
        text0 = render_input("s", self._choice(), template_idx=0)
        text1 = render_input("s", self._choice(), template_idx=1)
        assert text0 != text1  # 多模板轮换


class TestTargetText:
    def test_noul_bool(self):
        q = NoulQuestion(id="q1", prompt="p")
        assert target_text(q, True) == "是"
        assert target_text(q, False) == "否"

    def test_choice_and_score_pass_through(self):
        cq = ChoiceQuestion(id="q1", prompt="p", options=["x", "y"])
        sq = ScoreQuestion(id="q1", prompt="p", levels=YELP_LEVELS)
        assert target_text(cq, "x") == "x"
        assert target_text(sq, "3星") == "3星"


# ============================================================
# 负采样
# ============================================================


class TestSampleOptions:
    def test_correct_in_options(self):
        rng = random.Random(0)
        options, label = sample_options("cat", ["dog", "bird", "fish", "snake"], 3, 0.0, rng)
        assert label == "cat"
        assert "cat" in options
        assert len(options) == 4  # 1 正确 + 3 干扰

    def test_no_duplicate_options(self):
        rng = random.Random(1)
        options, _ = sample_options("a", list("bcdefgh"), 5, 0.0, rng)
        assert len(options) == len(set(options))

    def test_other_sample_when_pool_excludes_correct(self):
        # other_rate=1.0 强制走「全干扰 + 其他」路径
        rng = random.Random(2)
        options, label = sample_options("cat", ["dog", "bird", "fish"], 2, 1.0, rng)
        assert label == OTHER_OPTION
        assert OTHER_OPTION in options
        assert "cat" not in options

    def test_distractor_count_clamped_to_pool(self):
        rng = random.Random(3)
        options, _ = sample_options("a", ["b"], 10, 0.0, rng)
        assert len(options) == 2  # 池子只够 1 个干扰


# ============================================================
# 样本构建（含契约校验）
# ============================================================


class TestBuildSamples:
    def test_choice_sample_passes_schema(self):
        rng = random.Random(0)
        raw = {"text": "我的卡还没到", "label": "card_arrival",
               "source": "banking77", "split": "train",
               "primitive": "choice", "industry": "客服"}
        sample = build_choice_sample(raw, ["card_arrival", "age_limit", "atm_support"],
                                     "banking77-0", rng, n_distractors=2)
        assert sample is not None
        req = DecisionRequest.model_validate(
            {"state": sample["state"], "questions": sample["questions"]}
        )
        assert isinstance(req.questions[0], ChoiceQuestion)
        assert sample["labels"]["q1"] in sample["questions"][0]["options"]

    def test_score_sample_levels_ordered(self):
        rng = random.Random(0)
        raw = {"text": "好评", "label": "5星", "source": "yelp_full", "split": "train",
               "primitive": "score", "industry": "电商"}
        sample = build_score_sample(raw, "yelp-0", rng)
        assert sample["labels"]["q1"] == "5星"
        assert sample["questions"][0]["levels"] == YELP_LEVELS

    def test_boolq_noul_label(self):
        rng = random.Random(0)
        raw = {"text": "段落\n问题：xxx", "label": "是", "source": "boolq",
               "split": "train", "primitive": "noul", "industry": "知识"}
        sample = build_noul_sample(raw, "boolq-0", rng)
        assert sample["labels"]["q1"] is True

    def test_snli_neutral_dropped(self):
        # 中立样本硬标 true/false 会教坏校准，必须丢弃
        rng = random.Random(0)
        raw = {"text": "前提：p\n假设：h", "label": "neutral", "source": "snli",
               "split": "train", "primitive": "noul", "industry": "推理"}
        assert build_noul_sample(raw, "snli-0", rng) is None

    def test_snli_contradiction_false(self):
        rng = random.Random(0)
        raw = {"text": "前提：p\n假设：h", "label": "contradiction", "source": "snli",
               "split": "train", "primitive": "noul", "industry": "推理"}
        sample = build_noul_sample(raw, "snli-1", rng)
        assert sample["labels"]["q1"] is False
