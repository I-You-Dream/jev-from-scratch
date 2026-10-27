"""概率校准工具。

原生 LLM 的似然通常不校准，需要后处理。核心方法是温度缩放
（temperature scaling）：在验证集上拟合一个温度参数 T，使
softmax(logits / T) 与经验频率对齐。

对决策模型而言校准是命脉：说 0.8 把握就该有约 80% 正确。
"""

from __future__ import annotations

import math


def fit_temperature(logits: list[list[float]], labels: list[int]) -> float:
    """在验证集上拟合最优温度 T，最小化 NLL。

    T > 1 让分布更平（压过自信），T < 1 更尖（提欠自信）。
    网格搜索 [0.5, 3.0]，步长 0.01。

    Args:
        logits: 每个样本对各候选的原始打分（未 softmax）
        labels: 每个样本的正确候选下标
    Returns:
        使验证集 NLL 最小的温度
    """
    best_T, best_nll = 1.0, float("inf")
    T = 0.5
    while T <= 3.0001:
        nll = 0.0
        for scores, y in zip(logits, labels):
            scaled = [s / T for s in scores]
            mx = max(scaled)
            exps = [math.exp(s - mx) for s in scaled]
            nll += -math.log(exps[y] / sum(exps))
        if nll < best_nll:
            best_nll, best_T = nll, T
        T += 0.01
    return best_T


def apply_temperature(scores: list[float], temperature: float) -> list[float]:
    """按温度缩放打分并 softmax 归一化。"""
    scaled = [s / temperature for s in scores]
    mx = max(scaled)
    exps = [math.exp(s - mx) for s in scaled]
    Z = sum(exps)
    return [e / Z for e in exps]


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 15
) -> float:
    """ECE：置信度分桶后，置信度均值与经验准确率的加权偏差。"""
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, ok))
    total = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += len(bucket) / total * abs(avg_conf - avg_acc)
    return ece


def brier_score(probs: list[list[float]], labels: list[int]) -> float:
    """Brier Score：概率与 one-hot 标签的均方误差，同时衡量准确与校准。"""
    if not probs:
        return 0.0
    total = 0.0
    for dist, y in zip(probs, labels):
        n = len(dist)
        s = sum((p - (1.0 if i == y else 0.0)) ** 2 for i, p in enumerate(dist))
        total += s / n
    return total / len(probs)


__all__ = [
    "apply_temperature",
    "brier_score",
    "expected_calibration_error",
    "fit_temperature",
]
