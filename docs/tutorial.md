# 教程：用 Qwen3.5-4B 复刻「System One 决策模型」

本教程完整记录从零复刻 Jev 式决策模型的全过程：数据获取 → 预处理 → 微调 → 校准 → 评测。
核心目标不是「让模型更会分类」，而是**让模型输出校准的概率**——说 0.8 把握，长期频率上就该有约 80% 正确。

> 前置阅读：[design.md](design.md)（决策原语契约）、[roadmap.md](roadmap.md)（阶段规划）。

---

## 目录

1. [目标：什么是「校准的决策模型」](#1-目标什么是校准的决策模型)
2. [环境准备](#2-环境准备)
3. [数据获取](#3-数据获取)
4. [数据预处理](#4-数据预处理)
5. [微调](#5-微调)
6. [校准](#6-校准)
7. [评测与最终结果](#7-评测与最终结果)
8. [踩坑记录（重点）](#8-踩坑记录重点)

---

## 1. 目标：什么是「校准的决策模型」

传统 LLM 逐 token 生成自由文本，输出需要解析校验，还可能跑飞。决策模型（Jev 风格）反过来：

- **不生成文本**：对预先定义的问题返回类型化答案
- **输出空间事先定义**：答案只能落在 schema 声明的候选集内，结构上不可能越界
- **概率必须校准**：核心价值是「说到做到」

三种决策原语：

| 原语 | 问题形态 | 返回 |
|---|---|---|
| noul | 是/否判断 | `p_true ∈ [0,1]` |
| choice | N 选 1 | 选项 + 各选项概率 + 置信度 |
| score | 有序等级评分 | 期望分 + 各等级概率 + 置信度 |

**最重要的心智模型**：把它当成「带概率的智能 if 语句」，而不是全知助手。它可能返回「错误的合法值」（在候选里选错），但不会越界；也不会算术、数数、比日期。

### 1.1 校准 vs 准确率：为什么 loss 低是坏事

这是本教程最核心的概念。看两个模型：

- 模型甲：对正确答案给 95% 概率，train loss 0.05，**但这是过自信**（说 95% 实际只对 70%）
- 模型乙：对正确答案给 70% 概率，train loss 更高，**但说 70% 就真对 70%**

决策模型的命脉是乙，不是甲。**硬标签交叉熵天然把模型推向过自信**——这是整条技术路线要对抗的核心矛盾。

对抗手段（三者叠加构成「校准闭环」）：

1. **软标签**（数据端）：模糊样本标注为 `p_true≈0.5`，教模型学会「不确定」
2. **标签平滑**（训练端）：硬标签 1 软化成 `1-ε`
3. **温度缩放**（推理端）：在验证集拟合温度 T，把置信度拉回真实频率

---

## 2. 环境准备

### 2.1 服务器

- AutoDL / 商汤类 GPU 实例：显存 ≥24GB（LoRA 用 24G 就够，全参微调 4B 需 ≥80GB）
- **数据盘挂载在 `/root/autodl-tmp`**：一切持久化产物放这里，系统盘实例重建即丢

```bash
# 磁盘布局（重要）
/root/autodl-tmp          # 数据盘，持久化，放代码/环境/模型/数据
/root/autodl-tmp/envs/jev # Python 环境（conda create -p 指定到数据盘）
```

### 2.2 依赖安装

```bash
# 用清华源装（国内快），缓存也重定向到数据盘
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
# bitsandbytes 0.50 需要额外的 kernels 包（4bit 量化用）
pip install 'kernels>=0.11.1'
```

### 2.3 网络（国内服务器的关键）

| 场景 | 方案 |
|---|---|
| GitHub / HuggingFace 直连不通 | `source /etc/network_turbo`（AutoDL 学术加速） |
| pip 装包 | 走清华源（**要关掉 network_turbo**，加速反而拖慢 pip） |
| HF 下载报 401 | `export HF_HUB_DISABLE_XET=1`（禁用 Xet 存储回退 LFS） |

---

## 3. 数据获取

### 3.1 行业 × 原语矩阵

练手不需要单一领域的海量数据，关键是**行业分布均衡**，让模型学到「决策能力」而非「领域知识」。

| 行业 | choice（分类/路由） | score（有序等级） | noul（是/否） |
|---|---|---|---|
| 客服 | Banking77（77 意图） | — | — |
| 新闻 | AG News（4 主题） | — | — |
| 推理 | — | — | SNLI（蕴含判断） |
| 知识 | — | — | BoolQ（是非问答） |
| 电商 | — | Yelp 星级（1–5） | — |

### 3.2 拉取命令

```bash
# 全部源各取 3000 条（streaming，不落全量缓存）
python scripts/fetch_data.py --limit 3000
```

**关键教训：练手项目所有数据源统一限量**。全量拉 snli 56 万条既浪费流量，又会在 `build_dataset` 时因 2GB 内存 OOM。

### 3.3 数据集选择的坑

- `PolyAI/banking77` 是老式 loading-script 数据集，**datasets 5.x 已不支持**（报 `Dataset scripts are no longer supported`），改用社区转换的 Parquet 版 `mteb/banking77`
- mteb 版把意图名放在 `label_text` 字段而非 ClassLabel 名称表，取标签要读对字段

---

## 4. 数据预处理

### 4.1 转换规则（`build_dataset.py`）

raw 中间格式 `{text, label, source, split, primitive, industry}` → 契约格式 `{id, state, questions, labels}`。

#### 负采样（choice 专用）

不把全部 77 个意图塞进候选集（既不真实也难训练），而是每条样本抽 **1 正确 + 5 干扰**：

```bash
python scripts/build_dataset.py --distractors 5 --other-rate 0.05
```

`other-rate`：5% 概率生成「全干扰 + 其他」的低置信度样本，教模型「真答案不在候选里时压低概率」。

#### neutral 软标签（noul 专用，最重要的设计）

SNLI 有蕴含/中立/矛盾三标签。**中立样本不丢弃、不硬标**，而是转成 `p_true=0.5` 的软标签：

```json
{"id": "snli-xxx", "labels": {"q1": 0.5}}
```

这是「学会说不确定」的核心训练信号——Jev 最值钱的能力。

#### 原语配平

各源数量悬殊（snli 56 万 vs yelp 限量 3000），不配平会导致 train/val/test 的原语占比漂移、eval 被大原语主导：

```bash
python scripts/build_dataset.py --balance --seed 42
```

配平后每个 split 内三原语等量（本教程最终 6000/6000/6000，共 18000 条）。

### 4.2 数据体检（上 GPU 前必做）

```bash
python scripts/check_dataset.py
# → reports/check_report.json（落盘）
```

检查项：schema 合规、标签合法性、id 唯一、split 泄漏、候选质量、分布统计。

**踩过的坑**：泄漏检查用 state 前 200 字符对比会误报——boolq 同一 passage 配不同问题，前缀相同但完整 state 不同。必须用**完整 state 精确对比**。

---

## 5. 微调

三种训练形态（`finetune.py` 支持）：

| 形态 | 量化 | 参数 | 显存 | 适用 |
|---|---|---|---|---|
| QLoRA | 4bit NF4 | LoRA r16（0.5%） | ~6GB | 低显存 |
| **bf16 LoRA** | 无 | LoRA r16 | ~20GB | **推荐（校准最好）** |
| 全参 | 无 | 全部 | ~70GB | 追求准确率 |

### 5.1 训练命令

```bash
# 非量化 LoRA（推荐）
python scripts/finetune.py \
    --model /root/autodl-tmp/models/Qwen3.5-4B \
    --train data/samples/train.jsonl \
    --val data/samples/val.jsonl \
    --output /root/autodl-tmp/models/jev-lora-bf16-v0 \
    --quantize none \
    --batch-size 8 --grad-accum 2 --epochs 2 \
    --label-smoothing 0.1

# 全参微调（lr 用小值防灾难遗忘）
python scripts/finetune.py ... \
    --quantize none --full-finetune \
    --epochs 1 --lr 1e-5 --label-smoothing 0.1
```

### 5.2 关键超参

| 参数 | 值 | 说明 |
|---|---|---|
| LoRA r / alpha | 16 / 32 | 标准 1:2 |
| lr | LoRA 2e-4；全参 1e-5 | 全参小 lr 防灾难遗忘 |
| label_smoothing | 0.1 | 对抗过自信 |
| batch | 8 × grad_accum 2 = 16 | 等效 batch |
| epochs | LoRA 2；全参 1 | 全参 7000 条数据 2 轮必过拟合 |

### 5.3 训练目标（软标签 + 标签平滑的实现）

`finetune.py` 里的 `SoftLabelTrainer` 做了两件默认 Trainer 做不了的事：

1. **逐样本加权 CE**：软标签样本（`p_true=0.5`）展开成「是」「否」两条、各带权重 0.5
2. **标签平滑**：硬标签 1 软化成 `1-ε`，其余概率摊到 `ε`

```python
# 关键：labels 必须 pop 出来，不能传给 model
# 否则 Qwen 内部会自己再算 loss 并 logits.float()，全参训练直接爆显存
labels = inputs.pop("labels")
outputs = model(**inputs)  # 只传 input_ids + attention_mask
```

---

## 6. 校准

### 6.1 温度缩放（`calibration.py`）

模型输出的候选似然 `scores`，除以温度 T 再 softmax。T>1 让分布更平（压过自信）。

在验证集上网格搜索最优 T（最小化 NLL）：

```python
def fit_temperature(logits, labels):
    best_T, best_nll = 1.0, float("inf")
    for T in ...:  # 网格 [0.5, 3.0]
        nll = 每个样本 -log(softmax(logits/T)[label])
        ...
    return best_T
```

### 6.2 评测时自动拟合

```bash
python scripts/compare_models.py \
    --base-model .../Qwen3.5-4B \
    --lora-adapters 'v1=.../jev-qlora-v1,bf16=.../jev-lora-bf16-v0' \
    --full-model .../jev-full-ft-v0/checkpoint-534 \
    --test data/samples/test.jsonl --val data/samples/val.jsonl \
    --limit 300 --fit-temp
```

---

## 7. 评测与最终结果

### 7.1 指标口径

| 指标 | 含义 | 目标 |
|---|---|---|
| accuracy | argmax 是否命中标签 | 越高越好 |
| **ECE** | 置信度与经验频率的分桶偏差（「说到做到」程度） | **< 0.05** |
| MCE | 最差分桶偏差 | 越低越好 |
| Brier | 概率均方误差（准确+校准综合） | 越低越好 |

### 7.2 最终结果（300 条分层测试，温度缩放后）

| 版本 | 量化 | 参数 | 准确率 | **ECE** | Brier |
|---|---|---|---|---|---|
| base 零训练 | — | — | 32.7% | 0.260 | 0.233 |
| QLoRA | 4bit | LoRA | 78.3% | 0.077 | 0.075 |
| **bf16 LoRA** | 无 | LoRA | 79.0% | **0.045** | 0.071 |
| 全参 | 无 | 全部 | **81.0%** | 0.087 | 0.070 |

按原语（bf16 LoRA 版）：choice 89%、score 64%、noul 84%。

### 7.3 五个核心结论

1. **训练大幅有效**：准确率 32.7% → 79%，choice 从 14%（比随机猜还差）到 90%
2. **校准闭环达标**：ECE 从 0.39 一路降到 0.045（软标签 + 标签平滑 + 温度缩放）
3. **量化损失精度**：4bit 的 ECE 0.077 比 bf16 的 0.045 差 1.7 倍——**对决策模型，量化省显存不划算**
4. **全参的悖论**：准确率最高但 ECE 最差，多花算力买到 2% 准确率却牺牲校准
5. **综合最优 = bf16 非量化 LoRA**：校准最好、准确率接近全参、成本只有零头

---

## 8. 踩坑记录（重点）

这一节是本教程最有价值的部分——都是真实踩过、排掉的坑。

### 8.1 transformers 5.x API 变化（5 个，最难缠）

升级到 transformers 5.20 后，多个 API 静默变化，报错信息又都不直观：

| 坑 | 报错 | 修法 |
|---|---|---|
| `warmup_ratio` 被移除 | `unexpected keyword argument 'warmup_ratio'` | 自己算 `warmup_steps = steps * 0.03` |
| `apply_chat_template` 返回类型变 | `TypeError: can only concatenate list` | 判断返回值，统一取 `.input_ids` |
| `logging_dir` 被移除 | 参数不存在 | 自定义 `TBCallback` 直写 SummaryWriter |
| `compute_loss` 签名变 | `unexpected keyword 'num_items_in_batch'` | 签名加上该参数（接受但忽略） |
| `torch_dtype` 改为 `dtype` | deprecation warning | 用 `dtype=` |

### 8.2 datasets 5.x 挖弃 loading-script 数据集

`PolyAI/banking77` 报 `Dataset scripts are no longer supported`。改用 `mteb/banking77`（Parquet 版），且注意意图名在 `label_text` 字段。

### 8.3 HF Xet 存储 401

无 token / 走代理时，新版 Xet 后端报 `cas-server.xethub.hf.co ... 401`。修法 `HF_HUB_DISABLE_XET=1`。

### 8.4 缓存写错位置

HF 默认缓存 `~/.cache/huggingface` 在系统盘，实例重建即丢。修法：`HF_HOME` 重定向到数据盘。

### 8.5 低配实例的 2GB 内存陷阱

**GPU 释放后，实例内存配额会从 110GB 掉回 2GB**。`import torch` 就要 1-2GB，直接 OOM（exit 137）。修法：`finetune.py` 延迟导入 torch（dry-run 只验数据管线不加载 torch）。

### 8.6 全参训练 CUDA OOM

`compute_loss` 把 `labels` 传给 `model(**inputs)`，Qwen 内部自己算 loss 时 `logits.float()`——`[batch, seq, 15万vocab]` 转 float32 是显存黑洞。LoRA 版显存富余没暴露，全参版 83GB 直接爆。修法：`labels = inputs.pop("labels")`，模型只出 logits。

### 8.7 磁盘被 optimizer.pt 撑爆

全参训练 checkpoint 里 `optimizer.pt` 高达 16.8GB（AdamW 状态）。50G 数据盘直接满。训完即删（除非要续训）。

### 8.8 本地内存压力杀后台任务

训练用本地 SSH 后台会话跑，本地 Claude Code 内存不足时会话被杀 → 远程 `save_pretrained` 没执行完。修法：**训练一律 `nohup ... > log 2>&1 &` + 服务器端日志文件**，不依赖本地 SSH 会话。

### 8.9 评测抽样 bug

`--limit 300` 直接取前 300 条，而 test.jsonl 里 choice 排在前面 → 300 条全是 choice，score/noul 没测到。修法：按原语分层抽样。

### 8.10 长文本越界

noul/score 的长 state 让 prompt 超 max_len，`start = len(prompt_ids)-1` 越界 `IndexError`。修法：prompt 截断到 `max_len-1` 给候选留空间。

### 8.11 网络（国内服务器的日常）

- GitHub 直连不通 → `source /etc/network_turbo`
- 本地 push GitHub 断 → 走本机代理（netstat 查端口，如 `https_proxy=http://127.0.0.1:7897 git push`）
- SSH 间歇断连 → 用 `ConnectionAttempts` 多重试 + 后台重试循环
- AutoDL 关机后 host:port 变 → 重连时用 machine_id 认机器，别当新机器配

---

## 附录：命令速查

```bash
# 数据链路
python scripts/fetch_data.py --limit 3000          # 拉数据（streaming 限量）
python scripts/build_dataset.py --balance --seed 42 # 构建+配平
python scripts/check_dataset.py                     # 体检

# 训练
python scripts/finetune.py --model ... --quantize none \
    --train data/samples/train.jsonl --val data/samples/val.jsonl \
    --output ... --label-smoothing 0.1

# 评测（温度缩放）
python scripts/compare_models.py --base-model ... \
    --lora-adapters 'v1=...,bf16=...' --full-model ... \
    --test ... --val ... --fit-temp

# 看训练曲线
python scripts/logs_to_tb.py --state .../trainer_state.json --out .../runs
# TensorBoard：服务器自带，端口 6007，logdir /root/tf-logs（软链接训练 runs 目录）
```
