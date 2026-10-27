# jev-qwen-decision

基于 **Qwen3.5-4B** 复刻 **Jev（System One）决策模型**：不生成文本，只输出带校准概率的类型化答案。

## 什么是校准决策模型

Jev 是 TypeSafe AI 于 2026-09-15 发布的首个「System One 决策模型」：

- **不生成文本**：对预先定义好的问题返回**带校准概率的类型化答案**
- **并行评估**：同一段状态（state）上的多个问题一次调用全部得出
- **类型安全**：合法输出空间在调用前由 schema 预先定义，不可能返回 schema 之外的值
- **快速廉价**：官方口径 70–500ms 延迟、输入 $0.042/百万 token、输出免费

「System One / Two」源自卡尼曼《思考，快与慢》：
系统 1 快速直觉，系统 2 缓慢审慎。现有 LLM 逐 token 生成、依赖思维链，更像系统 2；
Jev 专攻系统 1 式的**有界、高速、可供软件直接消费**的判断。

## 三种决策原语（Primitives）

| 原语 | 问题形态 | 返回 | 典型用途 |
|---|---|---|---|
| **Noul** | 是/否判断（"这张工单是在请求退款吗？"） | `[0,1]` 上的概率 | 标记、门控、独立判据 |
| **Choice** | N 个已定义选项中选一个（"这归哪个团队？"） | 选项 + 各选项概率 + 置信度 | 路由、分类、工具选择 |
| **Score** | 有序等级评分（"这起事故有多严重？"） | 期望分数 + 各等级概率 + 置信度 | 优先级、风险、质量评估 |

三种问题可以混在同一次请求里，共享同一段状态。

## 与传统 LLM 的差异

| 维度 | 传统 LLM | 本项目 |
|---|---|---|
| 输出 | 自由文本（需解析校验） | 强类型结构化数值 |
| 采样 | 自回归逐 token 生成 | 候选似然读出 |
| 训练目标 | RLHF / RLVR | 校准决策（概率要"说到做到"） |
| 幻觉 | 存在跑飞风险 | 结构上不可能输出 schema 外的值 |
| 适用 | 写作、推理、解释 | 分类、路由、打分、门控 |

> **心智模型**：把它当成"带概率的智能 if 语句"，而不是全知助手。
> 它可能返回"错误的合法值"（在选项里选错），但不会越界；也不会算术、数数、比日期。

## 技术方案

### 推理：候选似然读出（零训练）

Qwen 是自回归生成模型，本项目不训练生成、而是**从候选 token 的似然读出概率**：

```
state + question + option_i → P(option_i | prompt)  对每个候选取补全对数似然
                            → 长度归一化 → softmax → 原语概率输出
```

noul 视作「是/否」二选一，score 对每个等级标签同样处理。详细设计见 [docs/design.md](docs/design.md)。

### 微调：三种训练形态

| 形态 | 量化 | 可训练参数 | 说明 |
|---|---|---|---|
| QLoRA | 4bit NF4 | LoRA r16（0.5%） | 低显存（~6GB） |
| **bf16 LoRA** | 无 | LoRA r16 | **校准最优**（~20GB） |
| 全参 | 无 | 全部（42 亿） | 准确率最高（~70GB） |

### 校准：对抗过自信的三板斧

硬标签交叉熵天然把模型推向过自信（说 0.95 实际只对 70%）。本项目用三手段构成校准闭环：

1. **软标签**（数据端）：SNLI 中立样本标注为 `p_true=0.5`，教模型学会「不确定」
2. **标签平滑**（训练端）：硬标签 1 软化成 `1-ε`（ε=0.1）
3. **温度缩放**（推理端）：验证集上拟合温度 T，把置信度拉回真实频率

## 实验结果

300 条分层测试集（choice/score/noul 各 100），温度缩放后：

| 版本 | 准确率 | **ECE** | Brier |
|---|---|---|---|
| 零训练基线 | 32.7% | 0.260 | 0.233 |
| QLoRA（4bit） | 78.3% | 0.077 | 0.075 |
| **bf16 LoRA** | 79.0% | **0.045** | 0.071 |
| 全参微调 | **81.0%** | 0.087 | 0.070 |

三个结论：

- **校准闭环达标**：ECE 从 0.39 降到 0.045（目标 < 0.05）
- **量化损失精度**：4bit 的 ECE 0.077 比 bf16 的 0.045 差 1.7 倍——决策模型里量化省显存不划算
- **全参的悖论**：准确率最高但校准最差，多花算力买 2% 准确率却牺牲校准

完整过程与踩坑见 [docs/tutorial.md](docs/tutorial.md)。

## 快速开始

```bash
# 1. 拉数据（各源限量，streaming）
python scripts/fetch_data.py --limit 3000

# 2. 构建训练集（负采样 + 软标签 + 原语配平）
python scripts/build_dataset.py --balance --seed 42

# 3. 体检
python scripts/check_dataset.py

# 4. 微调（非量化 LoRA，推荐）
python scripts/finetune.py \
    --model ./models/Qwen3.5-4B \
    --train data/samples/train.jsonl --val data/samples/val.jsonl \
    --output ./models/jev-lora-bf16-v0 \
    --quantize none --label-smoothing 0.1

# 5. 评测（含温度缩放）
python scripts/compare_models.py \
    --base-model ./models/Qwen3.5-4B \
    --lora-adapters 'bf16=./models/jev-lora-bf16-v0' \
    --test data/samples/test.jsonl --val data/samples/val.jsonl \
    --fit-temp
```

显存参考：4B 模型 bf16 推理约 9–10GB；QLoRA 训练约 6GB，非量化 LoRA 约 20GB，全参约 70GB。

## 目录结构

```
├── src/jev_qwen/
│   ├── schema.py         # 类型化问答契约（Pydantic，唯一事实源）
│   ├── dataset.py        # 样本构建：模板渲染、负采样、软标签
│   ├── calibration.py    # 温度缩放 / ECE / Brier
│   ├── primitives.py     # 候选似然读出（占位，Phase 2）
│   └── engine.py         # 决策引擎（占位，Phase 2）
├── scripts/
│   ├── fetch_data.py     # 拉取开源数据集
│   ├── build_dataset.py  # 构建训练集（负采样/软标签/配平）
│   ├── check_dataset.py  # 数据体检
│   ├── finetune.py       # QLoRA / 非量化 LoRA / 全参微调
│   ├── compare_models.py # 多版本对比评测（含温度缩放）
│   └── logs_to_tb.py     # 训练日志转 TensorBoard
├── tests/                # 单元测试（schema + dataset）
├── docs/
│   ├── design.md         # 决策原语契约
│   ├── tutorial.md       # 完整教程（含踩坑记录）
│   └── roadmap.md        # 实施路线图
└── data/samples/         # 样本格式约定（*.jsonl 不入库）
```

## License

个人实验项目。
