"""概率校准工具。

原生 LLM 的似然通常不校准，需要后处理。核心方法是温度缩放
（temperature scaling）：在验证集上拟合一个温度参数 T，使
softmax(logits / T) 与经验频率对齐。

当前为占位实现，Phase 3 在云服务器上完成。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """校准拟合结果。"""

    method: str
    temperature: float
    ece_before: float
    ece_after: float


def fit_temperature(
    logits: list[list[float]], labels: list[int]
) -> CalibrationResult:
    """在验证集上拟合温度参数，最小化 NLL。

    Args:
        logits: 每个样本对各候选的原始打分
        labels: 每个样本的正确候选下标
    """
    raise NotImplementedError("Phase 3 实现：NLL 最小化拟合温度")


def apply_temperature(scores: list[float], temperature: float) -> list[float]:
    """按温度缩放打分并 softmax 归一化。"""
    raise NotImplementedError("Phase 3 实现")


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 15
) -> float:
    """计算 ECE：概率与经验频率的分桶偏差。"""
    raise NotImplementedError("Phase 3 实现")


def brier_score(probs: list[list[float]], labels: list[int]) -> float:
    """计算 Brier Score：概率的均方误差，同时衡量准确与校准。"""
    raise NotImplementedError("Phase 3 实现")
