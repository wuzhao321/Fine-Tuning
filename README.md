# llm-ft-repro

Qwen1.5-14B 单卡 LoRA / QLoRA + FlashAttention-2 复现实验代码。目标是把**配置 → 原始日志 → 校验 → 汇总 → 报告与简历**连成可核查的证据链。

**当前状态：代码与文档初版，尚未进行 GPU 验证。** 本轮按要求优先开发，不安装训练环境，不下载模型，不租用 GPU，不执行训练。依赖版本、数据集选择和实际实验结果尚待确定，文档中的理论数值均不是实测。

## 从这里开始

在仓库根目录运行以下命令，无需安装 PyTorch，也不访问网络：

```bash
python3 -B scripts/repro.py validate
python3 -B scripts/repro.py plan E2
python3 -B scripts/repro.py preflight --config configs/train/e5_qlora.yaml
python3 -B scripts/repro.py summarize
```

仓库中的 YAML 使用 JSON 子集，规划与默认配置读取可仅用标准库；修改为普通 YAML 后需要 PyYAML。GPU 运行环境的项目要求是 Python 3.10+。`plan` 永远只展示计划；`matrix` 也默认不执行，只有显式加 `--execute` 才启动子进程。

当前默认配置的 preflight 会列出未固定的模型 revision、尚未准备的数据等阻塞项，并返回退出码 2；这是预期状态。它不下载模型、不导入 PyTorch、不探测 GPU。`ready=true` 只说明本地输入检查通过，不代表 GPU 环境已验收。

## 已实现的入口

| 命令 | 功能 |
| --- | --- |
| `validate` | 检查继承后完整配置、字段类型、固定协议和 LoRA 参数 |
| `preflight --config … [--resume …]` | 汇总本地输入缺口，检查完整数据包、adapter 或恢复文件 |
| `inspect --config …` | E0：模型结构、参数量、精度审计、最小前反向 |
| `prepare-data --config …` | 许可来源检查、分组去重、三份划分、模板化与长度统计 |
| `pack-data --config …` | 中文语料转同源定长基准 token 流，无 padding |
| `bench --config … --repeat 0` | 单进程单配置基准、CUDA 同步、NVML、OOM 日志 |
| `matrix E1 --execute` | 串行启动新进程、重复测量、累计时间账本 |
| `train --config … [--resume …]` | E5：相同模型入口、回答区间 loss、验证集、检查点恢复 |
| `eval --config …` | E6：留出测试集生成、QA EM/F1、中文 ROUGE、逐样本输出 |
| `compare-numerics LEFT RIGHT --atol … --rtol …` | eager / FA2 的前 20 步损失比较，阈值需事先确定 |
| `summarize [--reviews …] [--excel] [--figures]` | 审计表、可引用结果、溯源表、图表、报告和简历草稿 |
| `review-template --output …` | 从真实 summary 生成带分组键、文件摘要和待审项目的审查草稿 |

## 目录

```text
configs/                       公共配置及 E0–E6；共 34 份（含 base）
  bench/                       E1–E4 正式矩阵
  diagnostics/                 OOM 降级、无 dropout 数值比较
  inspect/ train/ eval/        E0、E5、E6
src/llm_ft_repro/
  config.py                    严格配置校验与显式继承
  model_factory.py             唯一模型与优化器入口
  data.py                      数据处理与回答区间标签
  bench.py telemetry.py        计时、显存、NVML
  train.py eval.py             训练与评测
  evidence.py analysis.py      证据写入、审查门、统计与报告
  artifacts.py preflight.py    检查点完整性记录、本地输入预检
  orchestrate.py cli.py         进程编排、预算与命令行
scripts/                       无安装启动器、配置生成、模型文件哈希
tests/                         数据泄漏、损坏文件、统计复算等少量离线回归
docs/                          设计、运行手册、协议、环境、数据、报告模板
data/                          来源声明模板；处理后大文件忽略跟踪
docker/                        待选定版本后的构建方案，不含虚构锁文件
results/raw/                   只追加日志，每次尝试独立文件
results/summary/               可重新生成的派生结果
figures/                       项目级图表归档位置
```

## 阅读顺序

1. [实现设计与取舍](docs/architecture.md)：固定模型规格、精度策略、测量边界。
2. [运行手册](docs/runbook.md)：后续 GPU 主机上的命令顺序。
3. [基准协议](docs/benchmark_protocol.md) 与 [证据规则](docs/evidence.md)。
4. [环境状态](docs/env.md)、[数据格式](docs/data.md)、[待办与验收](docs/roadmap.md)。
5. [第二轮完善记录](docs/iteration-02.md)：本地预检、检查点完整性、证据复算和进程关联。
6. [QLoRA / FlashAttention-2 研究资料库](docs/research/README.md)：系统检索的一手资料、阅读路线、版本源码与项目影响。

2026-10-24 文献研究发现：现有 `mfu()` 将 GC 重算计入 MFU 分子，需区分任务 FLOPs 与含重算的分析利用率；60% 只能作为项目复核门槛。详见 [研究后的指标修订建议](docs/research/project-implications.md)。目前没有实测结果，本轮检索没有修改训练代码或配置。

原始计划书保存在 [docs/source-plan.md](docs/source-plan.md)。其中嵌入的 Agent Prompt 是参考规格，本次没有执行其安装、租机或逐阶段审批流程。仓库采用标准 `src/llm_ft_repro` 包布局，实现职责对应原计划的 `src/model_factory` 等模块。

**结果声明：** 尚无 A100 实测数据。任何复现实验结论都必须写明硬件、日期和公开替代数据，不能当作历史实习期间的实测。
