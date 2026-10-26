#!/usr/bin/env python3
"""下载微调数据集，并统一成中间格式 JSONL。

用法（在项目根目录执行）：
    python scripts/fetch_data.py                               # 下载全部数据集
    python scripts/fetch_data.py --datasets banking77 snli      # 只下载指定的
    python scripts/fetch_data.py --output /root/autodl-tmp/data/raw --limit 2000

输出：
    <output>/<name>_<split>.jsonl，每行一个统一记录：
    {
      "text": "...",          # 模型输入文本（各数据集已适配）
      "label": "...",         # 字符串标签（选项 / 等级 / 是-否）
      "source": "banking77",  # 来源数据集
      "split": "train",       # 原始划分
      "primitive": "choice",  # 对应决策原语
      "industry": "客服"       # 行业标签，用于配比与分行业评测
    }

这些中间文件由 scripts/build_dataset.py 进一步转成
data/samples/ 约定的 state + questions + labels 训练格式。

网络提示：
    - HuggingFace 直连困难时先执行 export HF_ENDPOINT=https://hf-mirror.com
    - 或 source /etc/network_turbo 开启学术加速（AutoDL 系实例）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

# ============================================================
# 各数据集的字段适配：把原始记录映射成 {"text":..., "label":...}
# 第二个参数 label_names 是 ClassLabel 名称表（可能为 None）
# ============================================================


def _norm_banking77(rec: dict, label_names: list[str] | None) -> dict:
    """Banking77：客服查询文本 + 意图标签（77 类）。"""
    return {"text": rec["text"], "label": label_names[rec["label"]]}


def _norm_ag_news(rec: dict, label_names: list[str] | None) -> dict:
    """AG News：新闻标题+正文 → 4 大主题。"""
    return {"text": rec["text"], "label": label_names[rec["label"]]}


def _norm_yahoo_topics(rec: dict, label_names: list[str] | None) -> dict:
    """Yahoo Answers：问题标题+描述 → 10 大主题。

    不拼接 best_answer —— 那是答案，放进输入会造成标签泄漏。
    """
    title = (rec.get("question_title") or "").strip()
    content = (rec.get("question_content") or "").strip()
    text = f"{title}\n{content}".strip()
    return {"text": text, "label": label_names[rec["topic"]]}


def _norm_goemotions(rec: dict, label_names: list[str] | None) -> dict:
    """GoEmotions：Reddit 评论 → 情绪标签。

    原始是多标签（labels 列表）；中间格式只保留主标签（第一个），
    完整多标签信息留待 build_dataset 阶段按需展开。
    """
    labels = rec.get("labels") or []
    if not labels:
        return None  # 无标签记录直接跳过
    return {"text": rec["text"], "label": label_names[labels[0]]}


def _norm_yelp_full(rec: dict, label_names: list[str] | None) -> dict:
    """Yelp 评论星级：1–5 星有序等级。

    原始标签是 0–4，映射成人可读的「N 星」，便于 score 原语直接当等级名。
    """
    star = rec["label"] + 1
    return {"text": rec["text"], "label": f"{star}星"}


def _norm_snli(rec: dict, label_names: list[str] | None) -> dict:
    """SNLI：自然语言推理 → 蕴含/中立/矛盾。

    过滤掉标注不一致的记录（label == -1）。
    """
    if rec["label"] == -1:
        return None
    # premise/hypothesis 分开保留，build_dataset 阶段构造「假设成立吗」命题
    return {
        "text": f"前提：{rec['premise']}\n假设：{rec['hypothesis']}",
        "label": label_names[rec["label"]],
    }


def _norm_boolq(rec: dict, label_names: list[str] | None) -> dict:
    """BoolQ：段落+问题 → 是/否。标签不走 ClassLabel，直接字符串化。"""
    text = f"{rec['passage']}\n问题：{rec['question']}"
    return {"text": text, "label": "是" if rec["answer"] else "否"}


# ============================================================
# 数据集注册表（对应 docs/data-plan.md 的行业 × 原语矩阵）
# ============================================================

DatasetNorm = Callable[[dict, list[str] | None], dict | None]

DATASETS: dict[str, dict[str, Any]] = {
    "banking77": {
        "hf_id": "PolyAI/banking77",
        "primitive": "choice",
        "industry": "客服",
        "description": "银行客服细粒度意图（77 类）",
        "normalize": _norm_banking77,
    },
    "ag_news": {
        "hf_id": "fancyzhx/ag_news",
        "primitive": "choice",
        "industry": "新闻",
        "description": "新闻主题 4 分类（World/Sports/Business/SciTech）",
        "normalize": _norm_ag_news,
    },
    "yahoo_topics": {
        "hf_id": "fancyzhx/yahoo_answers_topics",
        "primitive": "choice",
        "industry": "论坛",
        "description": "Yahoo 问答 10 大主题",
        "normalize": _norm_yahoo_topics,
    },
    "goemotions": {
        "hf_id": "SamLowe/goemotions",
        "primitive": "choice",
        "industry": "社交",
        "description": "Reddit 评论细粒度情绪（27 类）",
        "normalize": _norm_goemotions,
    },
    "yelp_full": {
        "hf_id": "Yelp/yelp_review_full",
        "primitive": "score",
        "industry": "电商",
        "description": "Yelp 评论 1–5 星有序评分",
        "normalize": _norm_yelp_full,
    },
    "snli": {
        "hf_id": "stanfordnlp/snli",
        "primitive": "noul",
        "industry": "推理",
        "description": "自然语言推理（蕴含/中立/矛盾）",
        "normalize": _norm_snli,
    },
    "boolq": {
        "hf_id": "google/boolq",
        "primitive": "noul",
        "industry": "知识",
        "description": "段落+问题的是非判断",
        "normalize": _norm_boolq,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载微调数据集（统一中间格式）")
    p.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                   default=None, help="只下载指定数据集，默认全部")
    p.add_argument("--output", default="data/raw",
                   help="输出目录（务必指向数据盘），默认 %(default)s")
    p.add_argument("--limit", type=int, default=None,
                   help="每个 split 最多保存多少条（调试用），默认不限")
    return p.parse_args()


def extract_label_names(dataset_dict) -> list[str] | None:
    """从数据集 features 提取 ClassLabel 名称表；无分类标签则返回 None。"""
    first_split = next(iter(dataset_dict.values()))
    for field in ("label", "labels", "topic"):
        feat = first_split.features.get(field)
        if feat is None:
            continue
        # Sequence(ClassLabel) 场景（如 goemotions 的多标签）也带 names
        if hasattr(feat, "names"):
            return list(feat.names)
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            return list(feat.feature.names)
    return None


def write_jsonl(records: list[dict], path: Path) -> None:
    """把记录列表写成 JSONL（UTF-8，保证中文可读）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fetch_one(name: str, cfg: dict, output: Path, limit: int | None) -> int:
    """下载单个数据集、适配字段并落盘，返回保存的总条数。"""
    # 延迟导入：只跑 --help 时不必加载 datasets
    from datasets import load_dataset

    print(f"\n=== [{name}] {cfg['description']} ===")
    print(f"HF 仓库：{cfg['hf_id']}  |  原语：{cfg['primitive']}  |  行业：{cfg['industry']}")

    try:
        ds = load_dataset(cfg["hf_id"])
    except Exception as exc:  # 单个失败不阻塞其他数据集
        print(f"下载失败（跳过）：{exc}")
        return 0

    label_names = extract_label_names(ds)
    normalize: DatasetNorm = cfg["normalize"]

    total = 0
    for split, dataset in ds.items():
        records: list[dict] = []
        for rec in dataset:
            item = normalize(dict(rec), label_names)
            if item is None:
                continue
            item.update({
                "source": name,
                "split": split,
                "primitive": cfg["primitive"],
                "industry": cfg["industry"],
            })
            records.append(item)
            if limit is not None and len(records) >= limit:
                break

        if not records:
            continue

        out_path = output / f"{name}_{split}.jsonl"
        write_jsonl(records, out_path)
        total += len(records)
        print(f"  {split:>10}: {len(records):>6} 条 → {out_path}")

    return total


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 只检测不设置：是否配了镜像影响能否直连下载
    if not os.environ.get("HF_ENDPOINT"):
        print("提示：未设置 HF_ENDPOINT。国内环境建议先执行：")
        print("      export HF_ENDPOINT=https://hf-mirror.com\n")

    names = args.datasets or list(DATASETS)
    print(f"输出目录：{output}")
    print(f"数据集：{', '.join(names)}")

    summary: dict[str, int] = {}
    for name in names:
        summary[name] = fetch_one(name, DATASETS[name], output, args.limit)

    print("\n========== 汇总 ==========")
    for name, count in summary.items():
        mark = "✓" if count else "✗"
        print(f"  {mark} {name:<14} {count:>7} 条")
    print(f"合计 {sum(summary.values())} 条 → {output}")

    failed = [n for n, c in summary.items() if not c]
    if failed:
        print(f"\n以下数据集未取到数据：{failed}")
        print("常见原因：网络不通（试 hf-mirror / network_turbo）或数据集 ID 变更。")
        sys.exit(1)


if __name__ == "__main__":
    main()
