# jev-qwen-decision

基于 **Qwen3.5-4B** 复刻 **Jev（System One）决策模型**的实验项目。

> 本仓库当前阶段仅作**代码暂存**用途：本机不装环境、不下载模型、不跑训练。
> 完整流程将在云服务器上执行，见 [docs/roadmap.md](docs/roadmap.md)。

## Jev 是什么

Jev 是 TypeSafe AI 于 2026-09-15 发布的首个「System One 决策模型」：

- **不生成文本**：对预先定义好的问题返回**带校准概率的类型化答案**
- **并行评估**：同一段状态（state）上的多个问题一次调用全部得出
- **类型安全**：合法输出空间在调用前由 schema 预先定义，不可能返回 schema 之外的值
- **快速廉价**：官方口径 70–500ms 延迟、输入 $0.042/百万 token、输出免费

「System One / Two」名称源自卡尼曼《思考，快与慢》：
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

| 维度 | 传统 LLM | Jev / 本项目目标 |
|---|---|---|
| 输出 | 自由文本（需解析校验） | 强类型结构化数值 |
| 采样 | 自回归逐 token 生成 | 并行读出（单次前向） |
| 训练目标 | RLHF / RLVR | 校准决策（概率要"说到做到"） |
| 幻觉 | 存在跑飞风险 | 结构上不可能输出 schema 外的值 |
| 适用 | 写作、推理、解释 | 分类、路由、打分、门控 |

> **心智模型**：把它当成"带概率的智能 if 语句"，而不是全知助手。
> 它可能返回"错误的合法值"（在选项里选错），但不会越界；也不会算术、数数、比日期。

## 技术路线（候选）

复刻 Jev 有三条路线，按投入从小到大：

1. **概率读出（零训练）** —— 首选起步
   在 Qwen3.5-4B 上对候选项做似然读出（candidate likelihood readout），
   将选项 logits 映射为 choice/score/noul 三种原语的概率输出。
2. **QLoRA 微调**
   用合成/标注决策数据微调，让模型学会给出校准概率。
3. **RLCD 式校准训练**（Reinforcement Learning for Calibrated Decisions）
   完整复刻需要大量算力与数据，暂不实施，仅在 docs 中记录思路。

详细设计见 [docs/design.md](docs/design.md)。

## 目录结构

```
jev-qwen-decision/
├── README.md                 # 本文件：项目说明、Jev 概念、技术路线
├── .gitignore                # 忽略模型权重、数据集、缓存、虚拟环境
├── requirements.txt          # 依赖清单（仅列包名，不安装；云服务器阶段使用）
├── docs/
│   ├── design.md             # 决策原语设计与 API 契约草案
│   └── roadmap.md            # 云服务器阶段实施路线图
├── src/
│   └── jev_qwen/
│       ├── __init__.py
│       ├── schema.py         # 类型化问题与答案的 Pydantic 模型（核心契约）
│       ├── primitives.py     # 三种决策原语的读出/评分逻辑（占位）
│       ├── engine.py         # 决策引擎：state + questions → typed answers（占位）
│       └── calibration.py    # 概率校准工具（占位）
├── scripts/
│   ├── infer.py              # 推理入口脚本（占位）
│   └── evaluate.py           # 评测脚本：准确率 / 校准度(ECE) / 延迟（占位）
├── tests/
│   ├── __init__.py
│   └── test_schema.py        # schema 单元测试（纯 pydantic，不依赖模型）
└── data/
    └── samples/              # 示例决策样本（state + questions + labels）
        └── README.md         # 数据格式说明
```

## 云服务器阶段准备清单（仅文档，本机不执行）

```bash
# 1. 安装依赖（Linux + CUDA）
pip install -r requirements.txt

# 2. 下载模型（ModelScope 或 HuggingFace）
modelscope download --model Qwen/Qwen3.5-4B --local_dir ./models/Qwen3.5-4B
# 或
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./models/Qwen3.5-4B

# 3. 跑通概率读出基线
python scripts/infer.py --model ./models/Qwen3.5-4B --input data/samples/demo.jsonl
```

**显存参考**：4B 模型 bf16 推理约需 9–10GB；4bit 量化后约 3–4GB。
训练（QLoRA）另需优化器状态，建议 ≥16GB（如 RTX 4090 / A10 / L20）。

## 本地可运行的部分

不装 torch 也能验证契约层：

```bash
python -m pytest tests/test_schema.py -v
```

## License

待定（个人实验项目）。
