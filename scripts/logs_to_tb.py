#!/usr/bin/env python3
"""把 trainer_state.json 里的历史 loss 重放成 TensorBoard 事件。

用途：训练时若没开 TensorBoard 回调，事后从 checkpoint 的
trainer_state.json 补出曲线（历史 log_history 完整保留在其中）。

用法：
    python scripts/logs_to_tb.py \
        --state models/jev-qlora-v0/checkpoint-876/trainer_state.json \
        --out models/jev-qlora-v0/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把 trainer_state.json 转成 TensorBoard 事件")
    p.add_argument("--state", required=True, help="trainer_state.json 路径")
    p.add_argument("--out", required=True, help="TB 事件输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    logs = state.get("log_history", [])
    if not logs:
        print("log_history 为空，无可转换内容")
        return

    from torch.utils.tensorboard import SummaryWriter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out))

    for entry in logs:
        step = entry.get("step", 0)
        # 训练损失用 loss，验证损失用 eval_loss，分开打标方便对比
        if "loss" in entry and isinstance(entry["loss"], (int, float)):
            writer.add_scalar("train/loss", entry["loss"], step)
        if "eval_loss" in entry and isinstance(entry["eval_loss"], (int, float)):
            writer.add_scalar("eval/loss", entry["eval_loss"], step)
        if "learning_rate" in entry and isinstance(entry["learning_rate"], (int, float)):
            writer.add_scalar("train/learning_rate", entry["learning_rate"], step)

    writer.close()
    print(f"已转换 {len(logs)} 条记录 → {out}")


if __name__ == "__main__":
    main()
