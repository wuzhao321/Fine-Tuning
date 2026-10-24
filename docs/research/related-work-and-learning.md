# 扩展资料与学习路线

检索日期：2026-10-24（Asia/Shanghai）。本页补充两篇核心专题的前置知识、邻近方法和官方教学资料。论文条目若标“摘要”，表示本轮只核实了原作者摘要及元信息，没有声称精读或复现全文。方法优劣限于各原论文的实验条件。

## LoRA、量化与可选研究分支

| 编号 | 一手资料 | 本轮阅读深度 | 应解决的问题与本项目边界 |
| --- | --- | --- | --- |
| R01 | [LoRA 原论文，2021](https://arxiv.org/abs/2106.09685) | 摘要 | 冻结 W、训练低秩增量的起点。参数少不代表中间激活不需要反向传播。 |
| R02 | [Microsoft loralib](https://github.com/microsoft/LoRA) | README | 对照原始实现的 Linear、缩放与合并行为；PEFT 是另一套实现，默认项需分别核实。 |
| R03 | [8-bit Optimizers，2021](https://arxiv.org/abs/2110.02861) | 摘要 | 分块量化优化器统计量的研究出处；“状态为 8-bit”与“支持 paged”是两种机制。 |
| R04 | [LLM.int8()，2022](https://arxiv.org/abs/2208.07339) | 摘要 | 向量量化与离群特征混合精度；不能用它解释本项目 NF4 的编码规则。 |
| R05 | [GPTQ，2022](https://arxiv.org/abs/2210.17323) | 摘要 | 近似二阶信息驱动的训练后权重量化；不是直接替换 NF4 的相同实验。 |
| R06 | [AWQ，2023](https://arxiv.org/abs/2306.00978) | 摘要 | 利用激活统计保护重要权重通道的 weight-only 量化；原文推理性能不证明训练吞吐。 |
| R07 | [SmoothQuant，2022](https://arxiv.org/abs/2211.10438) | 摘要 | 离线平滑激活离群值，研究 W8A8；与 QLoRA 的权重压缩、bf16 计算不同。 |
| R08 | [LoftQ，2023](https://arxiv.org/abs/2310.08659) | 摘要 | 联合考虑量化与 LoRA 初始化以降低重构差异；启用后初始化协议已经变化。 |
| R09 | [LQ-LoRA，2023](https://arxiv.org/abs/2311.12023) | 摘要 | 把权重分解成高精度低秩部分与冻结量化部分，研究可变 bit/block 配置。 |
| R10 | [rsLoRA，2023](https://arxiv.org/abs/2312.03732) | 摘要 | 研究 rank 与缩放的关系；若改为 alpha/sqrt(r)，就不再是项目既定 alpha/r 对照。 |
| R11 | [DoRA，2024](https://arxiv.org/abs/2402.09353) | 摘要 | 将幅值和方向分开适配；会改变参数组成，不能继续套用 104,857,600 断言。 |
| R12 | [PiSSA，2024](https://arxiv.org/abs/2404.02948) | 摘要 | 利用权重主奇异分量初始化适配器；作为初始化研究分支，而非当前默认配置。 |
| R13 | [LoRA+，2024](https://arxiv.org/abs/2402.12354) | 摘要 | 研究 A/B 使用不同学习率；与本项目统一学习率不同，需新增单独实验编号。 |

这些资料用于理解设计空间。本轮没有把任何变体引入训练代码，也没有根据跨论文数字给它们排序。

## 从数学到 GPU kernel

| 编号 | 一手资料 | 本轮阅读深度 | 阅读目标 |
| --- | --- | --- | --- |
| R14 | [Online normalizer calculation for softmax，2018](https://arxiv.org/html/1805.02867v2) | 正文 §2–§3 | 在线更新最大值与归一化分母；是理解分块 softmax 的前置。 |
| R15 | [Self-attention Does Not Need O(n²) Memory，2021](https://arxiv.org/abs/2112.05682) | 摘要 | 理解二次计算量与注意力存储量可以分开讨论。 |
| R16 | [Training Deep Nets with Sublinear Memory Cost，2016](https://arxiv.org/abs/1604.06174) | 摘要 | 激活重算以计算换存储的经典来源；论文的时间开销不是本项目 GC 的保证值。 |
| R17 | [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | 官方教程 | 计算朴素 softmax 的 HBM 读写量，再看 fusion 如何减少中间结果。 |
| R18 | [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 官方教程 | tile、program ID、stride、L2 复用、autotune；为 FA2 的分块 GEMM 建立直觉。 |
| R19 | [Triton Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | 官方教程 | 反向 VJP 与并行归约的教学例子。Qwen 用 RMSNorm，不能把此 LayerNorm 公式直接替换进去。 |
| R20 | [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | 官方教程/源码 | 对照 m/l/acc、causal stage、前反向分块；教程代码和生产 flash-attn 包不是同一个实现。 |
| R21 | [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) | 官方硬件文档 | SM、寄存器、shared memory、occupancy 和异步拷贝，面向 A100 的约束解释。 |
| R22 | [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) | 官方文档 | 区分 requires_grad、no_grad 和 eval；冻结参数与保留输入梯度不是同一件事。此处记录实际访问版本，不是项目锁定版本。 |

建议在纸上完成一次在线 softmax 的分块合并：每块保留最大值 m、指数和 l，以及对 V 的未归一化加权和 u。两块合并时，先取共同最大值，再按 exp(m_old - m_new) 重标定旧统计量，最后输出 u/l。它解释了为什么可以不把整个概率矩阵写回 HBM；完整 attention 的正确反向、mask 和 dropout 仍需阅读 FA2 专题所列正文与实现。来源：[在线 softmax](https://arxiv.org/html/1805.02867v2)、[Triton attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。

## 课程与实践入口

| 编号 | 一手资料 | 本轮阅读深度 | 对应学习任务 |
| --- | --- | --- | --- |
| R23 | [Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) | 官方课程目录 | Lecture 2 的资源核算 → Lecture 5 GPU → Lecture 6 kernel/Triton；其他课程主题可后读。 |
| R24 | [CS336 Assignment 2 官方仓库](https://github.com/stanford-cs336/assignment2-systems) | README/目录 | 从计时和 profiling 过渡到 Triton attention；作业中多卡部分不属于本项目。 |

## FA2 之后的路线与全词表 loss

| 编号 | 一手资料 | 本轮阅读深度 | 关联与边界 |
| --- | --- | --- | --- |
| R25 | [FlashAttention-3，2024](https://arxiv.org/abs/2407.08608) | 摘要 | 关注 Hopper 的异步执行与低精度硬件，原论文 H100 收益不能照搬到 A100。 |
| R26 | [FlashAttention-4，2026](https://arxiv.org/abs/2603.05451) | 摘要 | 了解面向新硬件的算法/流水线共同设计；本项目仍围绕 A100 与 FA2。 |
| R27 | [Liger Kernel，2024](https://arxiv.org/abs/2410.10989) | 摘要 | 除 attention 外，融合 norm、激活、loss 等也可能影响端到端训练；论文百分比不是预期成绩。 |
| R28 | [Liger 官方实现](https://github.com/linkedin/Liger-Kernel) | README | 定位 fused linear cross entropy 等模块与集成接口，先核查版本支持与数值一致性。 |
| R29 | [Cut Your Losses，2024](https://arxiv.org/abs/2411.09009) | 摘要 | 大词表 logits/loss 可成为显存瓶颈；研究不完整物化 logits 的交叉熵计算。 |
| R30 | [Apple CCE 官方实现](https://github.com/apple-aiml-research/ml-cross-entropy) | README | 对照接口、梯度过滤选项及适配范围，不能只根据论文结论假定逐位等价。 |

**项目推断：** 对 Qwen 的 152064 词表，FA2 减少 attention 中间张量后，输出 logits/loss 可能成为重要峰值来源。是否由其主导必须通过显存快照和 profiler 判断。若后续采用 fused loss，单列实验，重新检查 loss、梯度与峰值，避免把多项优化的收益都归到 FA2。依据：[CCE](https://arxiv.org/abs/2411.09009)、[Liger 实现](https://github.com/linkedin/Liger-Kernel)。
