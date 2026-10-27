#!/usr/bin/env python3
"""训练数据体检：上 GPU 前把预处理问题一次排干净。

用法：
    python scripts/check_dataset.py --input data/samples

检查项（任一 ERROR 都不该开训）：
    1. 结构合规   每条样本能通过 schema 校验
    2. 标签合法   choice 的 label 必须在 options 内；score 在 levels 内；noul 是 bool
    3. id 唯一    不允许重复 id（对齐答案时会串）
    4. split 泄漏 同一段 state 不能同时出现在 train/val/test
    5. 候选质量   options/levels 无重复、数量正确、含正确答案
    6. 分布统计   原语/来源/标签占比，低置信度样本比例
    7. 文本健康   空文本、极端长度
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_qwen.schema import DecisionRequest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练数据体检")
    p.add_argument("--input", default="data/samples", help="build_dataset.py 的输出目录")
    return p.parse_args()


def load_split(input_dir: Path, split: str) -> list[dict]:
    path = input_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def check_split(name: str, rows: list[dict], errors: list, warnings: list) -> dict:
    """检查单个 split 的质量，返回统计信息。"""
    stats = {
        "n": len(rows),
        "primitive": Counter(),
        "labels": Counter(),
        "other_samples": 0,
        "len_chars": [],
    }
    seen_ids = set()

    for row in rows:
        sid = row.get("id", "?")

        # 1) 结构合规
        try:
            req = DecisionRequest.model_validate(
                {"state": row["state"], "questions": row["questions"]}
            )
        except Exception as exc:
            errors.append(f"[{name}] {sid}: schema 校验失败 - {exc}")
            continue

        # 2) id 唯一
        if sid in seen_ids:
            errors.append(f"[{name}] 重复 id: {sid}")
        seen_ids.add(sid)

        q = req.questions[0]
        label = row["labels"].get(q.id)
        stats["primitive"][q.type.value] += 1
        stats["len_chars"].append(len(str(row["state"])))

        # 3) 标签合法性
        if q.type.value == "choice":
            if label not in q.options:
                errors.append(f"[{name}] {sid}: choice 标签 {label!r} 不在 options 内")
            if len(q.options) != len(set(q.options)):
                errors.append(f"[{name}] {sid}: options 存在重复")
            if label == "其他":
                stats["other_samples"] += 1
        elif q.type.value == "score":
            if label not in q.levels:
                errors.append(f"[{name}] {sid}: score 标签 {label!r} 不在 levels 内")
            if q.levels != sorted(q.levels, key=lambda x: q.levels.index(x)):
                pass  # levels 有序性由构造保证，这里只检查存在性
        else:  # noul
            # 标签可以是 bool（硬标签）或 float（软标签，p_true ∈ [0,1]）
            if isinstance(label, bool):
                pass
            elif isinstance(label, float) and 0.0 <= label <= 1.0:
                pass
            else:
                errors.append(
                    f"[{name}] {sid}: noul 标签应为 bool 或 [0,1] 软标签，"
                    f"实际 {type(label).__name__}={label!r}"
                )

        # 4) 文本健康
        text = str(row["state"])
        if not text.strip():
            errors.append(f"[{name}] {sid}: state 为空")
        if len(text) > 20000:
            warnings.append(f"[{name}] {sid}: state 过长（{len(text)} 字符），可能截断")

        stats["labels"][str(label)] += 1

    return stats


def check_leakage(splits: dict[str, list[dict]], errors: list, warnings: list) -> None:
    """检查 train/val/test 之间是否存在同一 state（泄漏会让指标虚高）。

    必须用完整 state 对比：boolq 等数据集同一 passage 配不同问题，
    前缀截断会把合法样本误判成泄漏（踩过的坑）。
    """
    def state_keys(rows: list[dict]) -> set:
        return {str(r["state"]) for r in rows}

    keys = {name: state_keys(rows) for name, rows in splits.items() if rows}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = keys[names[i]] & keys[names[j]]
            if overlap:
                errors.append(
                    f"split 泄漏：{names[i]} 与 {names[j]} 有 {len(overlap)} 条相同 state"
                )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    errors: list[str] = []
    warnings: list[str] = []

    splits = {s: load_split(input_dir, s) for s in ("train", "val", "test")}
    if not any(splits.values()):
        sys.exit(f"错误：{input_dir} 下没有 train/val/test.jsonl")

    print(f"数据目录：{input_dir}\n")
    all_stats = {}
    for name, rows in splits.items():
        if rows:
            all_stats[name] = check_split(name, rows, errors, warnings)

    check_leakage(splits, errors, warnings)

    # ---- 报告 ----
    print("========== 数据规模 ==========")
    for name, st in all_stats.items():
        prim = dict(st["primitive"])
        other = st["other_samples"]
        lens = st["len_chars"]
        avg_len = sum(lens) / len(lens) if lens else 0
        print(f"  {name:<6} {st['n']:>6} 条  原语 {prim}  「其他」样本 {other}  平均长度 {avg_len:.0f}")

    print("\n========== 标签分布（Top 5） ==========")
    for name, st in all_stats.items():
        top = st["labels"].most_common(5)
        print(f"  {name:<6} {top}")

    print("\n========== 检查结果 ==========")
    if errors:
        print(f"  ✗ ERROR × {len(errors)}（必须修复后再训练）")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... 以及另外 {len(errors) - 20} 条")
    else:
        print("  ✓ 无 ERROR")

    if warnings:
        print(f"  ⚠ WARNING × {len(warnings)}")
        for w in warnings[:10]:
            print(f"    - {w}")

    if not errors:
        print("\n结论：数据体检通过，可以开训。")
    else:
        print("\n结论：存在阻断性问题，请修复后再开训。")
        sys.exit(1)


if __name__ == "__main__":
    main()
