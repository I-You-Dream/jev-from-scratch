#!/usr/bin/env python3
"""把 data/raw/ 中间格式转成 data/samples/ 训练格式。

用法（在项目根目录、服务器上执行）：
    python scripts/build_dataset.py                        # 默认 data/raw → data/samples
    python scripts/build_dataset.py --limit-per-source 3000 --seed 42

输出（符合 data/samples/README.md 的契约）：
    data/samples/train.jsonl    训练用（QLoRA）
    data/samples/val.jsonl      温度缩放拟合用（Phase 3）
    data/samples/test.jsonl     最终评测用

每行一个样本：
    {"id": ..., "state": ..., "questions": [...], "labels": {...}}

转换规则：
    choice（banking77/ag_news/yahoo_topics/goemotions）
        → 负采样候选集（1 正确 + k 干扰，少量「全干扰+其他」低置信度样本）
    score（yelp_full）
        → 有序等级 1–5 星
    noul（boolq/snli）
        → 是/否命题；snli 中立样本丢弃（硬标真/假会教坏校准）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# 允许直接以脚本方式运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_qwen.dataset import (  # noqa: E402
    build_choice_sample,
    build_noul_sample,
    build_score_sample,
)

# 数据源 → 构建器；转换逻辑在 jev_qwen.dataset 中（训练/推理共用）
BUILDERS = {
    "choice": build_choice_sample,
    "score": build_score_sample,
    "noul": build_noul_sample,
}

# 原始 split → 输出 split。缺失的划分由 train 切分补齐
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建微调训练数据集")
    p.add_argument("--input", default="data/raw", help="fetch_data.py 的输出目录")
    p.add_argument("--output", default="data/samples", help="训练样本输出目录")
    p.add_argument("--limit-per-source", type=int, default=None,
                   help="每个数据源最多采样多少条（含各 split），默认全部")
    p.add_argument("--distractors", type=int, default=5,
                   help="choice 候选集中的干扰项个数，默认 %(default)s")
    p.add_argument("--other-rate", type=float, default=0.05,
                   help="choice 低置信度样本（全干扰+其他）的比例，默认 %(default)s")
    p.add_argument("--seed", type=int, default=42, help="随机种子，默认 %(default)s")
    return p.parse_args()


def load_raw(input_dir: Path) -> dict[str, list[dict]]:
    """读取全部 <source>_<split>.jsonl，按数据源归组。"""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    by_source[json.loads(line)["source"]].append(json.loads(line))
    return by_source


def convert(
    raw_records: list[dict],
    primitive: str,
    label_pool: list[str],
    rng: random.Random,
    limit: int | None,
    n_distractors: int,
    other_rate: float,
) -> list[dict]:
    """把一个数据源的原始记录全部转成训练样本。"""
    builder = BUILDERS[primitive]
    samples: list[dict] = []

    records = raw_records[:limit] if limit else raw_records
    for i, rec in enumerate(records):
        sample_id = f"{rec['source']}-{i:06d}"
        if primitive == "choice":
            sample = builder(
                rec, label_pool, sample_id, rng,
                n_distractors=n_distractors, other_rate=other_rate,
            )
        else:
            sample = builder(rec, sample_id, rng)
        if sample is not None:
            # 保留 split 归属，后面按此分文件
            sample["_split"] = SPLIT_MAP.get(rec["split"], "train")
            samples.append(sample)
    return samples


def write_split(samples: list[dict], output: Path) -> dict[str, int]:
    """按 split 分文件写盘（去掉内部字段 _split），返回各 split 条数。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        buckets[s.pop("_split")].append(s)

    counts = {}
    for split, rows in buckets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_dir = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    by_source = load_raw(input_dir)
    if not by_source:
        sys.exit(f"错误：{input_dir} 下没有可用的 raw 数据，请先运行 fetch_data.py")

    print(f"输入：{input_dir}")
    print(f"输出：{output}")
    print(f"数据源：{', '.join(sorted(by_source))}\n")

    all_samples: list[dict] = []
    for source, records in sorted(by_source.items()):
        primitive = records[0]["primitive"]
        # 候选标签池：同数据源全部不同标签，用于负采样抽干扰项
        label_pool = sorted({r["label"] for r in records})
        before = len(all_samples)
        all_samples.extend(convert(
            records, primitive, label_pool, rng,
            limit=args.limit_per_source,
            n_distractors=args.distractors,
            other_rate=args.other_rate,
        ))
        print(f"  [{source:<14}] {primitive:<6} 标签 {len(label_pool):>3} 个，"
              f"样本 {len(all_samples) - before:>6} 条")

    counts = write_split(all_samples, output)
    print("\n========== 落盘 ==========")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} {counts.get(split, 0):>6} 条 → {output / (split + '.jsonl')}")
    print(f"合计 {len(all_samples)} 条")

    # 按原语统计，检查三类决策任务是否都覆盖到了
    by_primitive: dict[str, int] = defaultdict(int)
    for s in all_samples:
        by_primitive[s["questions"][0]["type"]] += 1
    print("原语分布：", dict(by_primitive))


if __name__ == "__main__":
    main()
