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
from jev_qwen.schema import NoulQuestion, Question  # noqa: E402
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
    p.add_argument("--lr", type=float, default=None,
                   help="学习率；缺省时全参 1e-5、LoRA 2e-4")
    p.add_argument("--batch-size", type=int, default=2, help="单卡 batch，默认 %(default)s")
    p.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    p.add_argument("--max-len", type=int, default=512, help="序列最大长度（决策样本较短）")
    # ---- 精度与训练模式 ----
    p.add_argument("--quantize", choices=["4bit", "none"], default="4bit",
                   help="4bit=QLoRA 省显存；none=bf16 全精度（显存够时更快更准）")
    p.add_argument("--full-finetune", action="store_true",
                   help="全参数微调（不用 LoRA），精度上限最高")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="关闭梯度检查点，用显存换速度")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="标签平滑系数，对抗硬标签 CE 过自信，默认 %(default)s")
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
        self.encoded: list[dict] = []
        for rec in self.records:
            # 软标签样本会展开成两条，所以 _encode 返回 list
            self.encoded.extend(self._encode(rec, rng))

    def _tokenize_prompt(self, prompt_text: str) -> list[int]:
        """渲染 prompt 并套 chat template，返回 prompt 侧 token id。"""
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        # transformers 5.x 返回 BatchEncoding，旧版返回 list[int]，统一成 list
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        return list(prompt_ids)

    def _pack(self, prompt_ids: list[int], target_text_: str, weight: float) -> dict:
        """把 prompt + 目标文本拼成一条编码样本，附带样本权重。"""
        target_ids = self.tokenizer(
            target_text_ + self.tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids
        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "weight": weight,
        }

    def _encode(self, sample: dict, rng: random.Random) -> list[dict]:
        """单条样本 → 一条或多条编码（软标签展开成两条）。"""
        question = QUESTION_ADAPTER.validate_python(sample["questions"][0])
        label = sample["labels"][question.id]

        # 多模板轮换：训练时随机挑一套，降低对单一措辞的过拟合
        prompt_text = render_input(sample["state"], question, rng.randrange(3))
        prompt_ids = self._tokenize_prompt(prompt_text)

        # 软标签：noul 的 float label（p_true），展开成「是」「否」两条带权重
        if isinstance(question, NoulQuestion) and isinstance(label, float):
            return [
                self._pack(prompt_ids, "是", label),
                self._pack(prompt_ids, "否", 1.0 - label),
            ]
        return [self._pack(prompt_ids, target_text(question, label), 1.0)]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict:
        return self.encoded[idx]


class PadCollator:
    """把变长样本 pad 成 batch。labels 用 -100 填充，保证不进损失。

    同时收集每个样本的 weight（软标签样本展开后两条各带 0.5），
    供 SoftLabelTrainer 做加权 loss。
    """

    def __init__(self, pad_token_id: int):
        import torch  # noqa: F401  # 确认训练环境有 torch，dry-run 不会走到这
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": [], "weight": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["weight"].append(f.get("weight", 1.0))
        return {k: torch.tensor(v) for k, v in batch.items()}


def make_soft_label_trainer(Trainer, label_smoothing: float = 0.0):
    """构造支持样本级软标签权重 + 标签平滑的 Trainer 子类。

    延迟接收 Trainer 基类：transformers.Trainer 依赖 torch，
    必须在正式训练路径 import（dry-run 只跑 tokenizer，不加载 torch）。

    默认 Trainer 的 loss 是对展平 token 求 CE 均值，无法表达「这条样本
    只算 0.5 权重」；这里改为逐样本算 CE，再按 weight 加权平均。
    label_smoothing > 0 时把硬标签 1 软化成 1-ε，其余概率摊到 ε——
    对抗硬标签 CE 的过度自信，配合 neutral 软标签构成校准闭环。
    """

    class SoftLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            import torch

            weights = inputs.pop("weight", None)  # [batch]
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()

            log_probs = torch.log_softmax(shift_logits, dim=-1)  # [b, s, V]
            # 目标 token 的负 log 概率；-100 先 clamp 到 0 避免越界，再 mask
            safe_labels = shift_labels.clamp(min=0)
            nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

            if label_smoothing > 0:
                # 平滑项：对 vocab 均匀分布的负 log 概率均值
                smooth = -log_probs.mean(dim=-1)
                per_pos = (1 - label_smoothing) * nll + label_smoothing * smooth
            else:
                per_pos = nll

            # 逐样本平均：只对有效 token（非 -100）取均值
            valid = (shift_labels != -100).float()
            per_sample = (per_pos * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

            if weights is not None:
                weights = weights.float().to(per_sample.device)
                loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)
            else:
                loss = per_sample.mean()

            return (loss, outputs) if return_outputs else loss

    return SoftLabelTrainer


# ============================================================
# 模型与训练
# ============================================================


def make_tb_callback(TrainerCallback):
    """构造 TensorBoard 回调：把训练指标写到指定目录。

    继承 TrainerCallback 基类（它提供 on_init_end 等全部事件的默认空实现，
    否则 Trainer 调用未定义事件会 AttributeError）。

    transformers 5.x 移除了 logging_dir 参数，report_to 目录行为也不确定；
    干脆自己用 SummaryWriter 直写，目录可控、软链接方便。
    """

    class TBCallback(TrainerCallback):
        def __init__(self, log_dir: str):
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """每 logging_steps 触发一次，把标量指标写进 TB。"""
            if not logs:
                return
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            self.writer.close()

    return TBCallback


def load_model_and_tokenizer(model_path: str, args: argparse.Namespace):
    """按训练模式加载基座：4bit 量化（省显存）或 bf16 全精度（快且准）。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.quantize == "4bit":
        # NF4 4bit 量化：4B 模型显存占用从 ~10GB 降到 ~3-4GB
        from transformers import BitsAndBytesConfig

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
            dtype=torch.bfloat16,
        )
    else:
        # bf16 全精度：无量化/反量化开销，数值精度最好；显存 ~10GB 起
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
        )

    if args.full_finetune:
        # 全参数微调：精度上限最高，显存需 ~50GB+（含 AdamW 状态）
        model.train()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"全参数微调：{trainable:,} / {total:,} 可训练（100%）")
    else:
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
    from transformers import Trainer, TrainingArguments, TrainerCallback

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

    # 学习率默认值按模式：全参用小 lr 防灾难性遗忘，LoRA 可以激进些
    lr = args.lr if args.lr is not None else (1e-5 if args.full_finetune else 2e-4)

    # warmup 占总步数 3%。transformers 5.x 移除了 warmup_ratio，自己算
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    warmup_steps = int(steps_per_epoch * args.epochs * 0.03)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        # 梯度检查点省显存但变慢；显存富余时关掉换速度
        gradient_checkpointing=not args.no_grad_checkpoint,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        report_to="none",                     # 不接 wandb/tensorboard
        seed=args.seed,
        remove_unused_columns=False,          # 我们的字段自己管理
    )

    # TensorBoard 日志写进 output_dir/runs，之后软链接到 /root/tf-logs 即可在面板看曲线
    tb_log_dir = output_dir / "runs"
    SoftLabelTrainer = make_soft_label_trainer(Trainer, args.label_smoothing)
    TBCallback = make_tb_callback(TrainerCallback)
    trainer = SoftLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[TBCallback(str(tb_log_dir))],
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
