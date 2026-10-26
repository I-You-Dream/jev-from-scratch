# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目状态（重要）

本仓库当前**只作代码暂存**：本机不装 torch、不下载模型、不跑训练。完整流程在云服务器上执行（见 `docs/roadmap.md`）。当前处于 **Phase 0**：`schema.py` 契约层已完成并有测试覆盖；`primitives.py` / `engine.py` / `calibration.py` / `scripts/*` 全部是 `NotImplementedError` 占位。改动这些占位时，保持 docstring 里标注的 Phase 编号与 roadmap 一致。

## 常用命令

```bash
# 运行全部契约层测试（唯一本地可跑的部分，只需 pydantic，不需要 torch）
PYTHONPATH=src python -m pytest tests/test_schema.py -v

# 运行单个测试类 / 单个用例
PYTHONPATH=src python -m pytest tests/test_schema.py::TestChoiceAnswer -v
PYTHONPATH=src python -m pytest tests/test_schema.py::TestChoiceAnswer::test_probs_must_sum_to_one -v
```

注意：`PYTHONPATH=src` 是**必需的**——项目是 src 布局但没有 pyproject.toml / setup.py，包不可 pip 安装，README 里写的裸 `python -m pytest ...` 会 `ModuleNotFoundError: No module named 'jev_qwen'`。`scripts/*.py` 则在文件头自行 `sys.path.insert` 了 `src`，可直接 `python scripts/infer.py` 运行（云服务器阶段）。

云服务器阶段才执行的命令（本机不要跑）：

```bash
pip install -r requirements.txt
modelscope download --model Qwen/Qwen3.5-4B --local_dir ./models/Qwen3.5-4B
python scripts/infer.py --model ./models/Qwen3.5-4B --input data/samples/demo.jsonl
python scripts/evaluate.py --pred predictions.jsonl --gold data/samples/test.jsonl
```

## 架构

复刻目标是 Jev 式「System One 决策模型」：对一段 state 上的多个问题，**不生成文本**，只返回 schema 允许空间内的类型化概率答案。三种决策原语：

- **noul** — 是/否命题 → `p_true ∈ [0,1]`（0.5 表示不确定，不是程度中等）
- **choice** — N 选 1 → `selected` + 归一化 `probabilities` + `confidence`
- **score** — 有序等级 → 按序号加权的 `expected_score` + `probabilities` + `confidence`

### 分层与数据流

```
schema.py        契约层（唯一事实源，Phase 0 已完成）
    ↓
primitives.py    候选似然读出：单问题 → 单答案（Phase 2）
    ↓
engine.py        JevEngine：DecisionRequest → DecisionResponse（Phase 2）
    ↓
calibration.py   温度缩放 / ECE / Brier（Phase 3）
    ↓
scripts/infer.py, evaluate.py    批量推理与评测入口
```

`src/jev_qwen/schema.py` 是全项目唯一提前敲定的部分：engine、primitives、scripts、tests 全部依赖这里的类型。**修改 schema 即修改全项目契约**，需同步更新 `tests/test_schema.py`、`docs/design.md` 的 JSON 示例和 `data/samples/README.md` 的样本格式。

### 契约层的关键不变量

- 合法输出空间在调用前由问题 schema 定义（options/levels），答案不可能落在候选集外；若真实类别缺失，概率仍被迫摊在剩余选项上——这是设计上的已知风险（见 `docs/design.md` §2.2）。
- `ChoiceAnswer` / `ScoreAnswer` 的 `probabilities` 必须归一化到和为 1（容差 1e-3），`selected` 必须在键中——由 `model_validator` 强制，测试断言这些校验。
- `confidence` 是概率分布的集中程度（通常取 max 概率），**不代表整体正确性**。
- 请求/响应的 `id` 必须唯一，答案与问题按 `id` 对齐（`DecisionResponse.get(id)` 消费）。
- `Question` / `Answer` 是按 `type` 字段判别的联合类型，解析后仍是各自的 Pydantic 具体类。

### 零训练路线（Phase 2 核心）

从 Qwen3.5-4B 候选补全的对数似然读出概率：对每个选项/等级算 log-likelihood → **长度归一化**（消除长候选项偏差，必做）→ 带温度 softmax。noul 视作 2 选项 choice（"是"/"否"）。一次前向只能读出一个候选的似然，N 个候选需 N 次前向或 batch——这是与 Jev 真正并行输出头的根本差距，消除它需要改模型结构，**明确不做**（见 roadmap「明确不做」）。原生似然不校准，需温度缩放后处理。

## 数据与产物约定

- `data/**/*.jsonl` 不入库，示例用 `.jsonl.example` 后缀；样本格式与标签约定见 `data/samples/README.md`。
- 模型权重（`models/`）、checkpoint、评测产物均被 `.gitignore` 忽略，不要提交。
- `format_state()`（engine.py）渲染 dict 型 state 时字段拼接顺序必须稳定，保证读出可复现。
- 治理约束（`docs/design.md` §6）：模型只**为策略提供信息**，阈值、权限、回退、副作用留在调用方代码里；`Meta` 里要留模型版本与延迟。
