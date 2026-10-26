# 实施路线图

> 当前状态：**Phase 0 完成**（本地仅建骨架，不装环境）。
> Phase 1 及之后在云服务器上执行。

## Phase 0 —— 项目骨架（本机，已完成）

- [x] 建立目录结构与文档
- [x] 定义 `schema.py` 类型契约（choice / score / noul）
- [x] schema 单元测试（纯 pydantic，可本地运行）
- [ ] 其余模块占位（primitives / engine / calibration / scripts）

**验收**：`python -m pytest tests/test_schema.py -v` 全绿。

## Phase 1 —— 环境与模型（云服务器）

- [ ] 准备 Linux + CUDA 机器（建议 ≥16GB 显存，4090/A10/L20）
- [ ] `pip install -r requirements.txt`
- [ ] 下载 Qwen3.5-4B 到 `./models/Qwen3.5-4B`（ModelScope 优先，国内快）
- [ ] 冒烟测试：加载模型、跑通一次 `tokenizer.apply_chat_template` + 前向

**验收**：能在 Python 交互里加载模型并对一段文本拿到 logits。

## Phase 2 —— 概率读出基线（零训练）

- [ ] 实现 `primitives.py`：候选似然读出 + 长度归一化 + softmax
- [ ] 实现三种原语的映射（noul 当 2 选项 choice 处理）
- [ ] 实现 `engine.py`：`state + questions → typed answers`
- [ ] prompt 模板工程：为三种原语各设计 2–3 套模板，做敏感性对比
- [ ] 准备首批标注数据 `data/samples/`（每类 ≥50 条）

**验收**：`scripts/infer.py` 能对 JSONL 输入输出合规的 JSON 响应，schema 校验通过。

## Phase 3 —— 校准与评估

- [ ] 实现 `calibration.py`：温度缩放（temperature scaling）拟合
- [ ] 实现 `scripts/evaluate.py`：准确率 / ECE / MCE / Brier / 延迟
- [ ] 建立 held-out 测试集，出第一版基线报表
- [ ] 影子模式实验：真实流量上比对模型决策与人工标注

**验收**：出一份可复现的基线报告（准确率 + ECE + 延迟，按原语分列）。

## Phase 4 —— QLoRA 微调（可选，视 Phase 3 结果决定）

若概率读出的校准度不够：

- [ ] 构造训练数据：state + question + 正确选项/等级
- [ ] QLoRA 微调（`peft` + `bitsandbytes` 4bit，LoRA r=16~32）
- [ ] 训练目标：候选上的交叉熵 + 校准正则
- [ ] 对比微调前后的准确率 / ECE / 延迟

**验收**：微调后 ECE 显著下降且准确率不劣化。

## Phase 5 —— 服务化与灰度（长期）

- [ ] 批量推理优化（vLLM / 批处理读出）
- [ ] 封装为 HTTP 服务（FastAPI），与 TypeSafe Jev API 形状对齐以便替换
- [ ] 置信度门控：低置信度走人工或回退到生成式 LLM
- [ ] 金丝雀发布 + 监控看板（校准漂移告警）

**落地姿势**：影子 → 校准 → 置信度门控 → 金丝雀 → 扩展，
置于中立接口之后，配确定性护栏与人工申诉通道。

## 明确不做

- **RLCD 式端到端校准强化学习**：算力与数据需求大，收益不确定，仅在 docs 记录思路。
- **改模型结构加并行决策头**：能消除与 Jev 的架构差距，但不再是"基于 Qwen3.5-4B 复刻"，
  而是训练新模型，超出本项目范围。

## 云服务器参考

| 项目 | 推荐配置 |
|---|---|
| 推理（4bit） | ≥8GB 显存 |
| 推理（bf16） | ≥12GB 显存 |
| QLoRA 训练 | ≥16GB 显存，建议 24GB |
| 磁盘 | 模型 8GB + 数据/产物 20GB+ |
| 系统 | Ubuntu 22.04 / CUDA 12.x / Python 3.10+ |

国内平台参考：AutoDL、恒源云、阿里云 PAI（ModelScope 下载快）。
