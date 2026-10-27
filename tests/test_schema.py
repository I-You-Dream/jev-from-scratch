"""schema 契约的单元测试。

纯 pydantic，不依赖 torch / 模型：
    PYTHONPATH=src python -m pytest tests/test_schema.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jev_qwen.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    Meta,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


# ============================================================
# 问题定义
# ============================================================


class TestNoulQuestion:
    def test_valid(self):
        q = NoulQuestion(id="q1", prompt="该工单在请求退款吗？")
        assert q.type.value == "noul"
        assert q.id == "q1"

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            NoulQuestion(id="q1", prompt="")
        with pytest.raises(ValidationError):
            NoulQuestion(id="q1", prompt="   ")


class TestChoiceQuestion:
    def test_valid(self):
        q = ChoiceQuestion(id="q1", prompt="路由到哪？", options=["物流", "退款", "其他"])
        assert len(q.options) == 3

    def test_too_few_options(self):
        with pytest.raises(ValidationError):
            ChoiceQuestion(id="q1", prompt="p", options=["仅一个"])

    def test_duplicate_options_rejected(self):
        with pytest.raises(ValidationError, match="重复"):
            ChoiceQuestion(id="q1", prompt="p", options=["A", "A", "B"])

    def test_blank_option_rejected(self):
        with pytest.raises(ValidationError):
            ChoiceQuestion(id="q1", prompt="p", options=["A", "  "])


class TestScoreQuestion:
    def test_valid(self):
        q = ScoreQuestion(id="q1", prompt="紧急程度", levels=["低", "中", "高", "紧急"])
        assert len(q.levels) == 4

    def test_duplicate_levels_rejected(self):
        with pytest.raises(ValidationError, match="重复"):
            ScoreQuestion(id="q1", prompt="p", levels=["低", "低"])


# ============================================================
# 答案定义
# ============================================================


class TestNoulAnswer:
    def test_valid(self):
        a = NoulAnswer(id="q1", p_true=0.87)
        assert a.p_true == pytest.approx(0.87)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            NoulAnswer(id="q1", p_true=1.5)
        with pytest.raises(ValidationError):
            NoulAnswer(id="q1", p_true=-0.1)


class TestChoiceAnswer:
    def test_valid(self):
        a = ChoiceAnswer(
            id="q1",
            selected="退款",
            probabilities={"物流": 0.2, "退款": 0.7, "其他": 0.1},
            confidence=0.7,
        )
        assert a.selected == "退款"

    def test_probs_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="之和"):
            ChoiceAnswer(
                id="q1",
                selected="A",
                probabilities={"A": 0.5, "B": 0.2},
                confidence=0.5,
            )

    def test_selected_must_be_in_probs(self):
        with pytest.raises(ValidationError, match="不在"):
            ChoiceAnswer(
                id="q1",
                selected="C",
                probabilities={"A": 0.5, "B": 0.5},
                confidence=0.5,
            )


class TestScoreAnswer:
    def test_valid(self):
        a = ScoreAnswer(
            id="q1",
            expected_score=2.4,
            probabilities={"低": 0.05, "中": 0.3, "高": 0.5, "紧急": 0.15},
            confidence=0.5,
        )
        assert a.expected_score == pytest.approx(2.4)

    def test_probs_must_sum_to_one(self):
        with pytest.raises(ValidationError):
            ScoreAnswer(
                id="q1",
                expected_score=1.0,
                probabilities={"低": 0.9, "高": 0.2},
                confidence=0.9,
            )


# ============================================================
# 请求 / 响应信封
# ============================================================


class TestDecisionRequest:
    def _questions(self):
        return [
            NoulQuestion(id="q1", prompt="在退款吗？"),
            ChoiceQuestion(id="q2", prompt="路由？", options=["A", "B"]),
            ScoreQuestion(id="q3", prompt="紧急？", levels=["低", "高"]),
        ]

    def test_valid_mixed_primitives(self):
        req = DecisionRequest(state={"title": "没发货"}, questions=self._questions())
        assert len(req.questions) == 3
        # 联合类型能正确解析回各自具体类型
        assert isinstance(req.questions[0], NoulQuestion)
        assert isinstance(req.questions[1], ChoiceQuestion)
        assert isinstance(req.questions[2], ScoreQuestion)

    def test_state_as_plain_text(self):
        req = DecisionRequest(state="一段纯文本", questions=[self._questions()[0]])
        assert req.state == "一段纯文本"

    def test_duplicate_question_ids_rejected(self):
        qs = [
            NoulQuestion(id="q1", prompt="p1"),
            NoulQuestion(id="q1", prompt="p2"),
        ]
        with pytest.raises(ValidationError, match="重复"):
            DecisionRequest(state="s", questions=qs)

    def test_empty_questions_rejected(self):
        with pytest.raises(ValidationError):
            DecisionRequest(state="s", questions=[])


class TestDecisionResponse:
    def test_valid_and_get(self):
        resp = DecisionResponse(
            answers=[
                NoulAnswer(id="q1", p_true=0.9),
                ChoiceAnswer(
                    id="q2",
                    selected="A",
                    probabilities={"A": 0.8, "B": 0.2},
                    confidence=0.8,
                ),
            ],
            meta=Meta(model="jev-qwen-readout-v0", latency_ms=120.5),
        )
        assert resp.get("q1").p_true == pytest.approx(0.9)
        assert resp.get("q2").selected == "A"

    def test_get_missing_id_raises(self):
        resp = DecisionResponse(
            answers=[NoulAnswer(id="q1", p_true=0.9)],
            meta=Meta(model="m"),
        )
        with pytest.raises(KeyError):
            resp.get("nope")

    def test_duplicate_answer_ids_rejected(self):
        with pytest.raises(ValidationError, match="重复"):
            DecisionResponse(
                answers=[
                    NoulAnswer(id="q1", p_true=0.9),
                    NoulAnswer(id="q1", p_true=0.1),
                ],
                meta=Meta(model="m"),
            )


# ============================================================
# 端到端契约往返
# ============================================================


class TestRoundTrip:
    def test_request_json_roundtrip(self):
        req = DecisionRequest(
            state={"title": "订单没发货", "body": "能退款吗？"},
            questions=[
                NoulQuestion(id="q1", prompt="在请求退款吗？"),
                ChoiceQuestion(id="q2", prompt="路由？", options=["物流", "退款"]),
            ],
        )
        restored = DecisionRequest.model_validate_json(req.model_dump_json())
        assert restored.model_dump() == req.model_dump()

    def test_response_json_roundtrip(self):
        resp = DecisionResponse(
            answers=[ScoreAnswer(
                id="q1",
                expected_score=1.2,
                probabilities={"低": 0.8, "高": 0.2},
                confidence=0.8,
            )],
            meta=Meta(model="m", latency_ms=88.0),
        )
        restored = DecisionResponse.model_validate_json(resp.model_dump_json())
        assert restored.answers[0].probabilities == {"低": 0.8, "高": 0.2}
