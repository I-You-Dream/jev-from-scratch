#!/usr/bin/env python3
"""下载 Qwen3.5-4B 模型权重到本地目录。

用法（在项目根目录执行）：
    python scripts/fetch_model.py                              # 默认 ModelScope → ./models/Qwen3.5-4B
    python scripts/fetch_model.py --output /root/autodl-tmp/models/Qwen3.5-4B
    python scripts/fetch_model.py --source huggingface         # 改走 HuggingFace

注意：
    1. 模型约 8GB（bf16），输出目录务必指向**数据盘**（如 /root/autodl-tmp/models），
       实例重建时系统盘会被清空。
    2. 国内优先 ModelScope；走 HuggingFace 时可配合镜像：
       HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_model.py --source huggingface
    3. 下载完成后自动校验 config / tokenizer / 权重文件是否齐全。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 默认模型仓库 ID（ModelScope 与 HuggingFace 同名）
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"

# 必须存在的配置文件，缺任意一个即视为下载不完整
REQUIRED_FILES = ("config.json", "tokenizer_config.json")

# 权重文件通配模式（safetensors 分片或旧版 bin），命中任一即可
WEIGHT_GLOBS = ("*.safetensors", "*.bin")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载 Qwen3.5-4B 模型权重")
    p.add_argument("--model", default=DEFAULT_MODEL_ID,
                   help="模型仓库 ID，默认 %(default)s")
    p.add_argument("--output", default="models/Qwen3.5-4B",
                   help="模型输出目录（务必指向数据盘），默认 %(default)s")
    p.add_argument("--source", choices=["modelscope", "huggingface"],
                   default="modelscope",
                   help="下载源，默认 ModelScope（国内直连快）")
    return p.parse_args()


def download_modelscope(model_id: str, output: Path) -> None:
    """通过 modelscope CLI 下载模型到指定目录。

    使用 CLI 而非 SDK，是因为 --local_dir 能精确控制落盘位置，
    避免 modelscope 缓存目录结构带来的路径混乱。
    """
    cmd = [
        sys.executable, "-m", "modelscope.cli.cli",
        "download", "--model", model_id, "--local_dir", str(output),
    ]
    print(f"执行：{' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ModelScope 下载失败（exit {proc.returncode}）。"
            f"可尝试 --source huggingface 换源。"
        )


def download_huggingface(model_id: str, output: Path) -> None:
    """通过 huggingface_hub 下载模型到指定目录。

    直接用 Python API（local_dir 参数精确落盘），
    国内环境请先 export HF_ENDPOINT=https://hf-mirror.com。
    """
    # 延迟导入：走 ModelScope 路径时不需要这个依赖
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT")
    if not endpoint:
        print("提示：未设置 HF_ENDPOINT，国内直连 HuggingFace 可能超时；"
              "可先 export HF_ENDPOINT=https://hf-mirror.com")

    print(f"从 HuggingFace 下载 {model_id} → {output}")
    snapshot_download(repo_id=model_id, local_dir=str(output))


def verify_model(output: Path) -> bool:
    """校验模型目录完整性：配置文件齐全，且至少存在一个权重分片。"""
    if not output.is_dir():
        print(f"错误：输出目录不存在：{output}")
        return False

    missing = [name for name in REQUIRED_FILES if not (output / name).exists()]
    if missing:
        print(f"错误：缺少配置文件 {missing}")
        return False

    # 权重可能被切成多个分片，只需确认有落盘即可
    weight_files = []
    for pattern in WEIGHT_GLOBS:
        weight_files.extend(output.glob(pattern))
    if not weight_files:
        print(f"错误：未找到权重文件（{', '.join(WEIGHT_GLOBS)}）")
        return False

    total_gb = sum(f.stat().st_size for f in weight_files) / 1024**3
    print(f"校验通过：{len(weight_files)} 个权重文件，共 {total_gb:.2f} GB")
    return True


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # HF 缓存默认在系统盘 ~/.cache/huggingface，重定向到数据盘避免重建丢失
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(output.parent / ".hf_cache")

    # 无 token / 代理环境下 Xet 存储会 401，回退经典下载路径
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print(f"模型：{args.model}")
    print(f"目标：{output}")
    print(f"来源：{args.source}")
    print("提示：请确认目标目录在数据盘上（实例重建不丢失）\n")

    if args.source == "modelscope":
        download_modelscope(args.model, output)
    else:
        download_huggingface(args.model, output)

    if not verify_model(output):
        sys.exit(1)

    print(f"\n完成：模型已就绪 → {output}")
    print("显存参考：4B 模型 bf16 推理约 9–10GB；4bit 量化约 3–4GB。")


if __name__ == "__main__":
    main()
