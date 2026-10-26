#!/usr/bin/env python3
"""QLoRA 微调：让 Qwen3.5-4B 的候选似然读出更有判别力。

用法（GPU 机器上执行）：
    python scripts/finetune.py \
        --model /root/autodl-tmp/models/Qwen3.5-4B \
        --train data/samples/train.jsonl \
        --val data/samples/val.jsonl \
        --output /root/autodl-tmp/models/jev-qlora-v0

    # 不加载模型，只验证数据管线（无 GPU 也能跑）
    python scripts/finetune.py --model ... --train ... --dry-run

设计（与 docs/design.md §4 的概率读出路线一致）：
    - 训练形态：**似然读出微调**——纯 LoRA，不改模型结构
    - 训练目标：正确候选文本上的交叉熵，把正确答案的似然相对拉高
    - 校准**不在训练里做**：交给 temperature scaling 后处理（calibration.py）
      硬标签 CE 天然导致过度自信，这是分工的理由
    - 推理时：对每个候选取似然 → 长度归一化 → softmax，与零训练基线同接口
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_qwen.dataset import render_input, target_text  # noqa: E402
from jev_qwen.schema import Question  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

# torch 改为延迟导入：小内存容器（cgroup 2GB）上 dry-run 只验数据管线时
# 不必加载 torch（import 本身就吃 1~2GB），正式训练路径才 import

# Question 是 Union 类型，用 TypeAdapter 还原成带类型的对象
QUESTION_ADAPTER = TypeAdapter(Question)

# Qwen 系列的 LoRA 目标模块：注意力 + MLP 投影层
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA 微调决策读出能力")
    p.add_argument("--model", required=True, help="基座模型路径（fetch_model.py 的输出）")
    p.add_argument("--train", default="data/samples/train.jsonl", help="训练样本")
    p.add_argument("--val", default="data/samples/val.jsonl", help="验证样本（用于 eval loss）")
    p.add_argument("--output", default="models/jev-qlora-v0", help="LoRA 适配器输出目录")
    # ---- 训练超参 ----
    p.add_argument("--epochs", type=float, default=2.0, help="训练轮数，默认 %(default)s")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率（QLoRA 常用 1e-4~2e-4）")
    p.add_argument("--batch-size", type=int, default=2, help="单卡 batch，默认 %(default)s")
    p.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    p.add_argument("--max-len", type=int, default=512, help="序列最大长度（决策样本较短）")
    # ---- LoRA 超参 ----
    p.add_argument("--lora-r", type=int, default=16, help="LoRA 秩，默认 %(default)s（16~32）")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha，默认 %(default)s")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    # ---- 模式 ----
    p.add_argument("--dry-run", action="store_true",
                   help="只验证数据管线并打印样本，不加载模型")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


# ============================================================
# 数据管线
# ============================================================


class DecisionDataset:
    """把训练样本编码成 input_ids / labels / attention_mask。

    labels 的约定：prompt 部分为 -100（不计损失），
    只有目标答案的 token 参与交叉熵——这正是「让正确候选似然更高」的训练信号。

    不继承 torch.Dataset：实现 __len__/__getitem__ 即可被 Trainer 消费，
    同时让 dry-run 路径完全绕开 torch（小内存容器可跑）。
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_len: int, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: list[dict] = []

        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        rng = random.Random(seed)
        self.encoded = [self._encode(rec, rng) for rec in self.records]

    def _encode(self, sample: dict, rng: random.Random) -> dict:
        """单条样本 → 模型输入。prompt/target 分别 tokenize 后拼接。"""
        question = QUESTION_ADAPTER.validate_python(sample["questions"][0])
        label = sample["labels"][question.id]

        # 多模板轮换：训练时随机挑一套，降低对单一措辞的过拟合
        template_idx = rng.randrange(3)
        prompt_text = render_input(sample["state"], question, template_idx)

        # chat template 加上角色头；add_generation_prompt 留出 assistant 生成位
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        # 目标文本不再加特殊 token（prompt 侧已带好），末尾补 eos 标记结束
        target_ids = self.tokenizer(
            target_text(question, label) + self.tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids

        input_ids = prompt_ids + target_ids
        input_ids = input_ids[: self.max_len]  # 超长截断（决策样本一般很短）
        labels = [-100] * len(prompt_ids) + target_ids
        labels = labels[: self.max_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict:
        return self.encoded[idx]


class PadCollator:
    """把变长样本 pad 成 batch。labels 用 -100 填充，保证不进损失。"""

    def __init__(self, pad_token_id: int):
        import torch  # noqa: F401  # 确认训练环境有 torch，dry-run 不会走到这
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


# ============================================================
# 模型与训练
# ============================================================


def load_model_and_tokenizer(model_path: str, args: argparse.Namespace):
    """加载 4bit 量化基座 + LoRA 适配器。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # NF4 4bit 量化：4B 模型显存占用从 ~10GB 降到 ~3-4GB
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists():
        sys.exit(f"错误：训练数据不存在 {train_path}，请先运行 build_dataset.py")

    # ---- dry-run：只验数据管线，适合无 GPU 时检查模板/标签是否正确 ----
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ds = DecisionDataset(train_path, tokenizer, args.max_len, seed=args.seed)
        print(f"dry-run：训练样本 {len(ds)} 条已编码成功")
        for i in range(min(2, len(ds))):
            rec = ds.records[i]
            enc = ds[i]
            n_prompt = enc["labels"].count(-100)
            print(f"\n--- 样本 {rec['id']} ---")
            print(f"原语：{rec['questions'][0]['type']}  标签：{rec['labels']}")
            print(f"token 数：{len(enc['input_ids'])}（prompt {n_prompt} + target "
                  f"{len(enc['input_ids']) - n_prompt}）")
            decoded = tokenizer.decode(
                [t for t in enc["input_ids"] if t != tokenizer.pad_token_id],
                skip_special_tokens=False,
            )
            print(f"模型可见输入：\n{decoded[:500]}")
        return

    # ---- 正式训练 ----
    from transformers import Trainer, TrainingArguments

    model, tokenizer = load_model_and_tokenizer(args.model, args)

    train_ds = DecisionDataset(train_path, tokenizer, args.max_len, seed=args.seed)
    eval_ds = (
        DecisionDataset(val_path, tokenizer, args.max_len, seed=args.seed)
        if val_path.exists()
        else None
    )
    print(f"训练样本 {len(train_ds)} 条"
          + (f"，验证 {len(eval_ds)} 条" if eval_ds else ""))

    output_dir = Path(args.output).resolve()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,          # 用时间换显存
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        report_to="none",                     # 不接 wandb/tensorboard
        seed=args.seed,
        remove_unused_columns=False,          # 我们的字段自己管理
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    result = trainer.train()
    print(f"\n训练完成：train loss = {result.training_loss:.4f}")

    # 只保存 LoRA 适配器（几十 MB），不复制整个基座
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 记录训练配置，便于复现与对比实验
    config_path = output_dir / "finetune_config.json"
    config_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"适配器与配置已保存 → {output_dir}")


if __name__ == "__main__":
    main()
