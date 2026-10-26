"""评测脚本：准确率 / 校准度(ECE) / Brier / 延迟。

用法（云服务器阶段）：
    python scripts/evaluate.py --pred predictions.jsonl --gold data/samples/test.jsonl

指标按原语类型（noul / choice / score）分别统计，输出 JSON + 文本报表。
当前为占位脚本，Phase 3 实现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_qwen.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
)
from jev_qwen.schema import DecisionResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="决策质量评测")
    p.add_argument("--pred", required=True, help="预测结果 JSONL（infer.py 输出）")
    p.add_argument("--gold", required=True, help="标注数据 JSONL")
    p.add_argument("--output", default="eval_report.json", help="报表输出路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO Phase 3：对齐 pred/gold，按原语分组计算指标
    #   - choice/noul: 准确率、ECE、MCE、Brier
    #   - score: 序号期望的相关性、各等级 ECE
    #   - 全部: 延迟分位数 (p50/p95)
    print("评测脚本尚未实现（Phase 3），见 docs/roadmap.md", file=sys.stderr)
    raise NotImplementedError("Phase 3 实现")


if __name__ == "__main__":
    main()
