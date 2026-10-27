#!/usr/bin/env python3
"""对比多个决策模型版本的质量与延迟。

对比三个版本（各自可选，缺哪个跳哪个）：
    base   零训练基线 —— 原始 Qwen3.5-4B 的似然读出
    lora   QLoRA 微调 —— 基座 + LoRA 适配器（finetune.py 默认模式）
    full   全参微调 —— bf16 全参数训练后的完整模型

用法（服务器 GPU 机器）：
    python scripts/compare_models.py \
        --base-model /root/autodl-tmp/models/Qwen3.5-4B \
        --lora-adapter /root/autodl-tmp/models/jev-qlora-v0 \
        --full-model /root/autodl-tmp/models/jev-full-ft-v0 \
        --test data/samples/test.jsonl --limit 300

输出（落盘）：
    evaluation_results.json   结构化指标（总体 + 按原语分列）
    evaluation_summary.md    人读对比表

指标口径见 docs/design.md §5：
    accuracy argmax 是否命中 / ECE 校准误差 / MCE 最差分桶偏差
    brier    概率的均方误差 / latency  端到端耗时分位数
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_qwen.dataset import render_input, target_text  # noqa: E402
from jev_qwen.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

QUESTION_ADAPTER = TypeAdapter(Question)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多版本决策模型对比评测")
    p.add_argument("--base-model", required=True, help="基座模型路径")
    p.add_argument("--lora-adapter", default=None, help="LoRA 适配器目录（QLoRA 版本）")
    p.add_argument("--full-model", default=None, help="全参微调后的完整模型目录")
    p.add_argument("--test", default="data/samples/test.jsonl", help="测试样本")
    p.add_argument("--limit", type=int, default=300,
                   help="抽样多少条测试（候选似然读出较慢），默认 %(default)s")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax 温度；1.0 = 不做校准后处理，看原始校准度")
    p.add_argument("--output", default="evaluation_results.json")
    p.add_argument("--report", default="evaluation_summary.md")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# 指标（口径对齐 docs/design.md §5）
# ============================================================


def expected_calibration_error(confidences: list[float], correct: list[bool],
                               n_bins: int = 15) -> float:
    """ECE：置信度分桶后，置信度均值与经验准确率的加权偏差。"""
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
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


def max_calibration_error(confidences: list[float], correct: list[bool],
                          n_bins: int = 15) -> float:
    """MCE：最差分桶的偏差。"""
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, ok))
    worst = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        worst = max(worst, abs(avg_conf - avg_acc))
    return worst


def brier_score(probs: list[dict[str, float]], labels: list[str]) -> float:
    """多分类 Brier：概率向量与 one-hot 标签的均方误差。"""
    if not probs:
        return 0.0
    total = 0.0
    for dist, label in zip(probs, labels):
        s = sum((p - (1.0 if k == label else 0.0)) ** 2 for k, p in dist.items())
        total += s / max(len(dist), 1)
    return total / len(probs)


def percentile(values: list[float], q: float) -> float:
    """简单分位数（排序插值）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# ============================================================
# 似然读出（与 design.md §4 的零训练路线同口径）
# ============================================================


def candidates_of(question: Question) -> list[str]:
    """取出该问题的全部候选文本（noul 恒为 是/否）。"""
    if isinstance(question, NoulQuestion):
        return ["是", "否"]
    if isinstance(question, ChoiceQuestion):
        return list(question.options)
    return list(question.levels)


def candidate_scores(model, tokenizer, prompt: str, candidates: list[str],
                     max_len: int, device: str) -> list[float]:
    """每个候选取长度归一化的对数似然。

    长度归一化是必须的：候选 token 数不同，原始似然不可比。
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_ids = list(prompt_ids)

    scores = []
    for cand in candidates:
        cand_ids = tokenizer(cand, add_special_tokens=False).input_ids
        input_ids = (prompt_ids + cand_ids)[:max_len]
        n_cand = max(1, min(len(cand_ids), len(input_ids) - len(prompt_ids)))

        with torch.no_grad():
            logits = model(
                torch.tensor([input_ids], device=device)
            ).logits[0]

        # 每个位置预测下一个 token；只取候选段的 logprob
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        token_lps = [
            log_probs[start + i, input_ids[len(prompt_ids) + i]].item()
            for i in range(n_cand)
        ]
        scores.append(sum(token_lps) / len(token_lps))
    return scores


def predict_sample(model, tokenizer, sample: dict, max_len: int,
                   temperature: float, device: str) -> dict:
    """单样本推理：候选似然 → softmax → 概率分布与预测。"""
    question = QUESTION_ADAPTER.validate_python(sample["questions"][0])
    label = target_text(question, sample["labels"][question.id])
    prompt = render_input(sample["state"], question, template_idx=0)
    cands = candidates_of(question)

    t0 = time.perf_counter()
    scores = candidate_scores(model, tokenizer, prompt, cands, max_len, device)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 带温度的 softmax 归一化
    scaled = [s / temperature for s in scores]
    mx = max(scaled)
    exps = [math.exp(s - mx) for s in scaled]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    dist = dict(zip(cands, probs))
    best = max(dist, key=dist.get)

    return {
        "id": sample["id"],
        "primitive": question.type.value,
        "label": label,
        "predicted": best,
        "confidence": dist[best],
        "correct": best == label,
        "probabilities": dist,
        "latency_ms": latency_ms,
    }


# ============================================================
# 模型加载与评测主流程
# ============================================================


def load_variant(name: str, args, device: str):
    """加载一个待测版本：base / lora（基座+适配器）/ full（完整微调模型）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if name == "base":
        path, adapter = args.base_model, None
    elif name == "lora":
        path, adapter = args.base_model, args.lora_adapter
    else:
        path, adapter = args.full_model, None

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def evaluate_variant(name: str, model, tokenizer, samples: list[dict],
                     args) -> tuple[list[dict], dict]:
    """跑完一个版本的全部样本，返回逐样本结果与聚合指标。"""
    records = []
    for i, sample in enumerate(samples):
        rec = predict_sample(
            model, tokenizer, sample, args.max_len, args.temperature,
            device=model.device,
        )
        records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"    [{name}] {i + 1}/{len(samples)}")

    def aggregate(rows: list[dict]) -> dict:
        if not rows:
            return {}
        confs = [r["confidence"] for r in rows]
        correct = [r["correct"] for r in rows]
        lat = [r["latency_ms"] for r in rows]
        return {
            "n": len(rows),
            "accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
            "ece": expected_calibration_error(confs, correct),
            "mce": max_calibration_error(confs, correct),
            "brier": brier_score([r["probabilities"] for r in rows],
                                 [r["label"] for r in rows]),
            "latency_ms_p50": percentile(lat, 0.5),
            "latency_ms_p95": percentile(lat, 0.95),
        }

    metrics = {"overall": aggregate(records)}
    # 按原语分列——混在一起看不出模型是在哪类决策上失灵
    for prim in ("choice", "score", "noul"):
        sub = [r for r in records if r["primitive"] == prim]
        if sub:
            metrics[prim] = aggregate(sub)
    return records, metrics


def render_report(all_metrics: dict, n_samples: int) -> str:
    """生成 markdown 对比表。"""
    lines = [
        "# 决策模型对比评测",
        "",
        f"测试样本：{n_samples} 条（来自 data/samples/test.jsonl）",
        "",
        "## 总体指标",
        "",
        "| 版本 | 准确率 | ECE | MCE | Brier | p50 延迟(ms) | p95 延迟(ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, metrics in all_metrics.items():
        m = metrics.get("overall", {})
        lines.append(
            f"| {name} | {m.get('accuracy', 0):.3f} | {m.get('ece', 0):.3f} | "
            f"{m.get('mce', 0):.3f} | {m.get('brier', 0):.3f} | "
            f"{m.get('latency_ms_p50', 0):.1f} | {m.get('latency_ms_p95', 0):.1f} |"
        )

    for prim in ("choice", "score", "noul"):
        if not any(prim in m for m in all_metrics.values()):
            continue
        lines += ["", f"## 按原语：{prim}", "",
                  "| 版本 | 准确率 | ECE | MCE | Brier | n |",
                  "|---|---|---|---|---|---|"]
        for name, metrics in all_metrics.items():
            m = metrics.get(prim, {})
            if not m:
                continue
            lines.append(
                f"| {name} | {m.get('accuracy', 0):.3f} | {m.get('ece', 0):.3f} | "
                f"{m.get('mce', 0):.3f} | {m.get('brier', 0):.3f} | {m.get('n', 0)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    test_path = Path(args.test)
    samples = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        # 按原语分层抽样：直接取前 N 条会全是一种原语（choice 排在文件前面）
        from collections import defaultdict

        by_prim: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_prim[s["questions"][0]["type"]].append(s)
        per = max(1, args.limit // 3)
        samples = []
        for prim in ("choice", "noul", "score"):
            samples.extend(by_prim[prim][:per])
    print(f"测试样本：{len(samples)} 条")

    variants: list[tuple[str, str]] = [("base", args.base_model)]
    if args.lora_adapter:
        variants.append(("lora", args.lora_adapter))
    if args.full_model:
        variants.append(("full", args.full_model))
    print(f"对比版本：{[v[0] for v in variants]}")

    all_metrics: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}

    for name, path in variants:
        print(f"\n=== 加载版本 [{name}] ← {path}")
        model, tokenizer = load_variant(name, args, device="auto")
        records, metrics = evaluate_variant(name, model, tokenizer, samples, args)
        all_metrics[name] = metrics
        all_records[name] = records
        print(f"    accuracy={metrics['overall']['accuracy']:.3f} "
              f"ece={metrics['overall']['ece']:.3f}")
        # 顺序评测：测完释放，避免三个模型同时占显存
        del model
        torch.cuda.empty_cache()

    # 落盘：结构化结果 + 人读报告
    output = Path(args.output)
    output.write_text(
        json.dumps(
            {
                "test_samples": len(samples),
                "temperature": args.temperature,
                "metrics": all_metrics,
                "predictions": all_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = Path(args.report)
    report.write_text(render_report(all_metrics, len(samples)), encoding="utf-8")

    print(f"\n结果已落盘：\n  {output}\n  {report}")
    print("\n" + render_report(all_metrics, len(samples)))


if __name__ == "__main__":
    main()
