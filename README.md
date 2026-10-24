# Fine-Tuning

## 项目目的

围绕 Qwen1.5-14B 的单卡微调，研究 LoRA、QLoRA 与 FlashAttention-2 的组合方式，比较不同注意力实现、梯度检查点和优化器对显存、吞吐及收敛的影响，并通过中文问答与摘要任务评估微调效果。

项目采用配置驱动的实验流程，将模型构建、数据处理、性能测量、训练评测和结果汇总组织在同一套工具中，便于复现与对比。

## 项目内容

- **模型与微调**：统一模型工厂，支持 bf16 LoRA、NF4 量化、双重量化、LoRA 参数校验及梯度检查点控制。
- **数据处理**：中文任务模板、分组去重、训练/验证/测试划分、回答区间标签与定长基准数据。
- **性能实验**：eager、SDPA、FlashAttention-2 对照，序列长度扫描，以及优化器步时与显存测量。
- **训练与评测**：检查点保存和恢复、基座与微调模型对比、问答 EM/F1、摘要 ROUGE。
- **结果与资料**：JSONL 原始日志、CSV/Excel 汇总、图表、实验报告，以及 QLoRA、FlashAttention-2 和 GPU 测量资料。

## 项目结构与导航

```text
configs/                 公共参数与 E0–E6 实验配置
  bench/                 性能对照与消融实验
  diagnostics/           数值一致性与 OOM 边界诊断
  inspect/ train/ eval/   模型检查、训练和评测配置
src/llm_ft_repro/         核心实现
  model_factory.py       模型加载、量化、LoRA 和 GC
  data.py                数据准备、划分和标签构造
  bench.py telemetry.py  计时、显存与 GPU 采样
  train.py eval.py       训练、恢复与任务评测
  analysis.py evidence.py 结果汇总、日志和溯源
  artifacts.py preflight.py 检查点完整性与输入预检
  config.py cli.py orchestrate.py 配置、命令行与进程编排
scripts/                 启动入口、配置生成与模型文件校验
tests/                   数据、检查点与统计完整性测试
docs/                    设计、使用手册和实验协议
  research/              论文、源码与官方资料索引
data/                    数据来源和审查模板
docker/                  容器构建与依赖清单
results/raw/             原始实验日志
results/summary/         派生统计与报告
figures/                 图表
```

| 想了解的内容 | 入口 |
| --- | --- |
| 实现原理与设计 | [架构说明](docs/architecture.md) |
| 如何运行实验 | [运行手册](docs/runbook.md) · [命令行入口](src/llm_ft_repro/cli.py) |
| 模型与微调配置 | [模型工厂](src/llm_ft_repro/model_factory.py) · [公共配置](configs/base.yaml) |
| 数据格式与评测口径 | [数据协议](docs/data.md) · [评测实现](src/llm_ft_repro/eval.py) |
| 性能测量与结果追踪 | [基准协议](docs/benchmark_protocol.md) · [日志与溯源](docs/evidence.md) |
| 环境与容器配置 | [环境配置](docs/env.md) · [容器说明](docker/README.md) |
| QLoRA 与 FlashAttention-2 资料 | [研究索引](docs/research/README.md) · [阅读路线](docs/research/reading-path.md) |
