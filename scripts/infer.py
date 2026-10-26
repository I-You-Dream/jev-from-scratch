"""推理入口脚本。

用法（云服务器阶段）：
    python scripts/infer.py --model ./models/Qwen3.5-4B --input data/samples/demo.jsonl

输入：JSONL，每行一个 DecisionRequest 的 JSON（state + questions）
输出：JSONL，每行一个 DecisionResponse 的 JSON（answers + meta）

当前为占位脚本，Phase 2 实现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接以脚本方式运行（无需 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_qwen.engine import JevEngine  # noqa: E402
from jev_qwen.schema import DecisionRequest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jev 式决策推理")
    p.add_argument("--model", required=True, help="模型路径，如 ./models/Qwen3.5-4B")
    p.add_argument("--input", required=True, help="输入 JSONL 路径")
    p.add_argument("--output", default="predictions.jsonl", help="输出 JSONL 路径")
    p.add_argument("--temperature", type=float, default=1.0, help="校准温度")
    p.add_argument("--quantize", choices=["none", "4bit", "8bit"], default="none")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = JevEngine(model_path=args.model, temperature=args.temperature)
    engine.load()

    n_ok = 0
    with open(args.input, encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            request = DecisionRequest.model_validate_json(line)
            response = engine.decide(request)
            fout.write(response.model_dump_json() + "\n")
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"已处理 {n_ok} 条", file=sys.stderr)

    print(f"完成：{n_ok} 条决策 → {args.output}")


if __name__ == "__main__":
    main()
