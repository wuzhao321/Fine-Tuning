# FlashAttention-2：原理、实现边界与本项目研究笔记

资料核对日期：2026-10-24。本页收录 18 项一手资料，重点为 A100 上的 Qwen1.5-14B 单卡训练。原论文、历史版本源码和当前官方文档分别标明。`v2.6.3`、`v4.46.3` 是用于定位行为的阅读版本，并非已验证的项目依赖组合。`docker/requirements.in` 是依赖声明，实际兼容范围应结合选定版本的锁文件确认。

以下“来源结论”是资料中的陈述；“项目推断 / 建议”来自资料与项目代码的结合，均不代表已有 GPU 实测。

## 先建立正确的概念

- **FA2 是精确 dense attention 的高效计算方式。** “精确”指没有引入稀疏化或低秩近似；有限精度计算的归约次序改变，不能承诺与 eager 逐位相同。[原论文 F01](https://arxiv.org/pdf/2205.14135)、[SDPA 数值说明 F10](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- **注意力计算量仍随长度呈二次增长。** 避免保存完整的分数 / 概率矩阵，主要改善中间存储和 IO；不能据此宣称完整训练显存或训练耗时都随长度线性增长。[FA2 正文 F02](https://tridao.me/publications/flash2/flash2.pdf)
- **`sdpa` 是接口选择，底层会 dispatch。** PyTorch 2.2 已集成 FA2；`sdpa` 与外部 `flash-attn` 路径可能都执行 FlashAttention 算法。[发布说明 F09](https://docs.pytorch.org/blog/pytorch2-2/)
- **QLoRA 的 4-bit 权重不意味着 4-bit 注意力。** 项目模型工厂配置 NF4 权重、bf16 计算及 autocast；FA2 接收到的 Q/K/V 仍需满足对应实现的 dtype 条件。[FA2 接口 F07](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/flash_attn/flash_attn_interface.py)、[Qwen 集成 F15](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py)

## 18 项一手资料与精读入口

### F01 · FlashAttention 原论文，2022

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) · **阅读深度：正文 §2–3、重算说明及算法。**

**来源结论：** 分块把 Q/K/V 从 HBM 搬至 SRAM，融合分数、softmax 与输出计算，避免完整中间矩阵往返 HBM。反向重算分数和概率，用增加部分运算换取更少 IO。IO 复杂度的优势有 SRAM 大小假设。

**项目推断：** E2 / E3 应把“减少保存激活”和“减小权重存储”分别归因；FA2 的反向重算属于注意力内部机制，不能据此认定配置中的 decoder gradient checkpointing 已启用。

### F02 · FlashAttention-2 原论文，2023

[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf) · **阅读深度：正文 §2–4、Algorithm 1–2、并行与实验部分。**

**来源结论：** 减少非矩阵 FLOPs，沿序列维度增加 thread block 并行，改善 warp 分工；前向仍为 O(N²d)，额外保存的 logsumexp 为 O(N)。论文报告 A100 注意力内核可达约 50%–73% 理论峰值、特定 GPT 训练最高 72% MFU；这些来自不同测量对象。

**项目推断：** 论文数值不是本项目 QLoRA 的预期值，也不能把 60% 解释为普适物理上限。本项目采用的 4N / 6N 估算口径还需单独说明。

### F03 · Online normalizer 的原始论文，2018

[Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867) · **阅读深度：正文 §3、Algorithm 3、并行合并规则。**

**来源结论：** 同时维护运行最大值与归一化分母；遇到新的最大值时，重缩放已有分母。这样无需先完整扫一遍输入再计算指数和归一化。

**学习用推导：** 两块分别保存 `(m₁,l₁)`、`(m₂,l₂)`，则 `m=max(m₁,m₂)`，`l=exp(m₁−m)l₁+exp(m₂−m)l₂`。已有统计能合并，因此 softmax 可按块处理。这个规则先解释归一化；再阅读 F02 如何同步维护未归一化的注意力输出，最后除以分母。

### F04 · 作者的 FA2 讲解，2023

[Tri Dao / Stanford CRFM: FlashAttention-2](https://crfm.stanford.edu/2023/07/17/flash2.html) · **阅读深度：全文与工作划分图说明。**

**来源结论：** 前向从 sliced-K 改为 sliced-Q：warps 各处理不同 Q 行，共享 K/V，减少跨 warp 中间结果求和。小 batch / 长序列时新增序列并行可改善占用率。其注意力对比使用 A100 80GB SXM4、特定长度和 head dimension。

**项目推断：** 本项目 `batch=1, heads=40` 的长序列点适合观察这类收益；这只是机制假设，最终速度仍需测量。

### F05 · 作者仓库 README，固定版本 v2.6.3

[Dao-AILab / flash-attention v2.6.3 README](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/README.md) · **阅读深度：依赖、CUDA 支持、API、changelog。**

**来源结论：** 此版本列出 CUDA ≥11.6、PyTorch ≥1.12、Linux；CUDA FA2 支持 Ampere / Ada / Hopper、fp16 / bf16、head dimension ≤256。API 提供 dropout、causal、局部窗口与确定性反向。v2.1 把不等长 Q/K 的 causal 对齐改成右下角。

**项目推断：** A100 + head dimension 128 在文档的架构 / 形状范围内；这不等于任意 torch、CUDA、编译器组合都已兼容。

### F06 · 作者仓库 README，当前 main

[Dao-AILab / flash-attention main README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md) · **阅读深度：当前 NVIDIA 支持、依赖、FA3 / FA4 分支说明。**

**来源结论：** 检索时 NVIDIA CUDA 路径要求 CUDA ≥12.0、整体要求 PyTorch ≥2.2。FA3 面向 Hopper；FA4 的 CuTeDSL 路径面向 Hopper / Blackwell。相同仓库内有多个实现与不同硬件要求。

**项目建议：** 引用依赖要求时附 tag 或 commit；不能把 F05 的历史下限和当前 main 的要求拼成一个安装方案。A100 主线仍围绕 FA2，不把其他 GPU 的数字作为本机证据。

### F07 · 作者仓库 Python / autograd 接口，v2.6.3

[flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/flash_attn/flash_attn_interface.py) · **阅读深度：FlashAttnFunc、varlen 函数、反向保存对象与 RNG。**

**来源结论：** 普通 Q/K/V 布局为 `(batch, sequence, heads, head_dim)`；varlen 使用扁平 token 轴、累计长度 `cu_seqlens` 和最大长度。autograd 保存 Q/K/V、输出、logsumexp 和 RNG 状态供反向使用，支持 dropout 与确定性反向参数。

**项目建议：** 日志区分真实 Q/K/V dtype 与量化权重 dtype；变长路径需要记录累计长度 / padding 比例；不要假定直接调用此 API 可传任意四维 mask。

### F08 · 作者的数值参考测试，v2.6.3

[tests/test_flash_attn.py](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/tests/test_flash_attn.py) · **阅读深度：参考实现、输出 / 梯度误差检查、dtype / dropout 参数化。**

**来源结论：** 测试对照高精度参考及 PyTorch 低精度实现，检查输出和梯度误差；部分断言按参考误差的倍数设置边界，而非要求逐位相同。

**项目建议：** 现有 loss 轨迹诊断可以保留；内核诊断增加输出、dQ/dK/dV 或 adapter 梯度误差更有解释力。不要把上游测试中的经验倍数照搬成所有模型的统一容差。

### F09 · PyTorch 2.2 发布说明，2024-01-30

[PyTorch 2.2: FlashAttention-v2 integration, AOTInductor](https://docs.pytorch.org/blog/pytorch2-2/) · **阅读深度：FA2 集成部分。**

**来源结论：** `torch.nn.functional.scaled_dot_product_attention` 自 2.2 集成 FA2。

**项目建议：** E2 的实验名应使用“eager / SDPA 自动选择 / 外部 flash-attn 路径”。如果想回答“FA 算法带来多少收益”，需另有禁用 flash 的 SDPA 对照和内核证据，不能只比较两个字符串。

### F10 · PyTorch SDPA API，检索解析到 2.14

[scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) · **阅读深度：参数、dispatch、数值与 dropout 说明。**

**来源结论：** 按输入自动选实现；布尔 mask 的 True 表示参与 attention；float mask 加到分数上。`dropout_p` 按传入值生效，评估时须传 0。不等长 `is_causal` 采用左上对齐；不同 fused backend 可有数值差异，math 后端对低精度输入保留 float 中间量。

**项目推断：** 不能把底层 SDPA 和外部 FA2 的不等长 causal 直接视为同义；模型适配层的 cache / mask 逻辑必须一并审查。

### F11 · PyTorch SDPA 后端控制 API，2.14

[torch.nn.attention.sdpa_kernel](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html) · **阅读深度：上下文管理器与后端枚举示例。**

**来源结论：** 可限定 SDPA 使用 FLASH_ATTENTION、MATH、EFFICIENT_ATTENTION 等后端；当前版本也提供优先级参数。上下文结束后恢复先前状态。

**项目建议：** 诊断时仅允许一个后端，遇到不支持的输入保留错误 / 原因。正式基准保留自动选择组，再把强制后端组作为单独消融；接口随版本变化，不能直接假定旧 torch 支持当前全部参数。

### F12 · PyTorch SDPA 官方教程

[Implementing High-Performance Transformers with SDPA](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) · **阅读深度：显式 dispatch、计时、因果注意力与 profiler 示例。**

**来源结论：** 展示按后端分别计时、处理 fused kernel 不可用的情况，并用 profiler 观察实际操作。不同输入形状可能有不同速度和支持情况。

**项目建议：** 增加独立注意力 microbenchmark：固定 `heads=40, head_dim=128, causal=True, bf16`，逐档扫描长度与 batch，区分 forward、backward 与完整 step。微基准用于解释 E2，不替代训练吞吐结果。

### F13 · PyTorch activation checkpoint 官方文档，2.5

[torch.utils.checkpoint](https://docs.pytorch.org/docs/2.5/checkpoint.html) · **阅读深度：reentrant / non-reentrant 差异、RNG 保持、输入梯度条件。**

**来源结论：** checkpoint 在反向重新执行前向片段；默认处理 RNG 状态使 dropout 与不重算时一致，但有开销。reentrant 要求至少一个输入和输出需要梯度；non-reentrant 没有这一要求，并可提前停止所需重算。

**项目推断：** FA2 内部重算与 decoder GC 可叠加，增加的计算不是相同范围。冻结基座训练中，必须检查 adapter 梯度连通；GC 开销应实测，不能固定套用某个百分比。

### F14 · PyTorch profiler 官方文档，2.14

[torch.profiler](https://docs.pytorch.org/docs/2.14/profiler.html) · **阅读深度：CPU / CUDA activities、schedule、trace 导出、开销说明。**

**来源结论：** 可记录 CPU / CUDA 活动、按训练步调度采样并导出 trace；记录 shapes / stacks 可能引入额外开销，保留 tensor 引用还可能影响执行。

**项目建议：** 单独产生少量“确认后端”的诊断 trace，附版本、配置哈希和 kernel 名；吞吐和峰值显存仍用原正式规程。`model.config._attn_implementation` 只能证明请求路径，不能独立证明执行了哪一个 CUDA kernel。

### F15 · Transformers Qwen2 attention 源码，v4.46.3

[modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py) · **阅读深度：Qwen2FlashAttention2、Qwen2SdpaAttention、attention 类映射。**

**来源结论：** 此版本两条路径分别调用 FA helper 与 PyTorch SDPA；SDPA 遇 `output_attentions=True` 回退手工实现。FA2 路径在 Q/K/V 意外变成 fp32 时按 autocast / 量化前 dtype 等信息回转；训练与评估使用不同 attention dropout 参数。

**项目建议：** 项目 RMSNorm / adapter fp32 策略使实际 Q/K/V dtype 值得记录；检查所选安装版本的同名源码，避免把这个历史版本的行为当成当前运行事实。

### F16 · Transformers FA helper 源码，v4.46.3

[modeling_flash_attention_utils.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_flash_attention_utils.py) · **阅读深度：unpadding、position_ids、varlen dispatch、确定性参数。**

**来源结论：** 2D padding mask 路径执行 unpad → varlen → pad；无 padding 也可凭重置的 position_ids 构造独立序列边界以防跨样本 attention，否则走普通函数。该版本可从 `FLASH_ATTENTION_DETERMINISTIC` 取得确定性选项。

**项目推断：** 把文本拼接后切成等长块，与保留独立样本边界的 packing 是不同训练语义；只有 EOS 而没有边界元数据，不会自动阻止后一个样本关注前一个样本。

### F17 · Transformers 当前 attention interface 官方文档

[Attention backends](https://huggingface.co/docs/transformers/attention_interface) · **阅读深度：后端注册、mask 格式、四维自定义 mask 限制。**

**来源结论：** eager 接受加性 float mask；SDPA 接受 boolean / float mask；FA2 使用 2D padding mask，不直接接受任意 raw 4D mask。2D 的 1 / 0 约定不能直接当作四维加性 mask。

**项目建议：** 将来添加样本隔离 packing 时，不能把 eager 的 dense block-diagonal mask 原样传给 FA2；需要匹配 varlen / 边界适配，且保证不同组计算的是同一种注意力问题。

### F18 · NVIDIA GPU 性能背景

[GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) · **阅读深度：GPU 执行模型、算术强度、性能限制。**

**来源结论：** GPU 工作由 SM、warp、存储层级组成；实际吞吐受形状、并行度、内存带宽及算术强度影响，理论峰值不能保证达到。指南举例列出 A100 的 SM、HBM 与 dense 峰值。

**项目建议：** 报告 A100 的完整型号、容量、SXM / PCIe、时钟和功耗限制。用“理论 FLOPs 估算的 MFU”描述现有指标，避免等同于 profiler / 硬件计数器测得的利用率。

## 项目实现与证据要求

以下对照项目实现与技术资料，列出各项设计的适用边界和证据要求。

| 项目实现 | 研究判断 / 下一步 | 对应资料 |
| --- | --- | --- |
| `model_factory.py` 断言 hidden=5120、heads=40、KV heads=40 | head dimension=128；该模型是 MHA，不能把其他 Qwen 型号的 GQA 结论直接套入 | F05、F07；结构数值来自项目代码 |
| 模型工厂要求请求的 attention 实现与 config 一致 | 继续保留；增加真实 kernel 证据后才能声称后端已确认 | F09、F14、F15 |
| `bench.py` 用 bf16 autocast，权重可 NF4，norm / adapter 部分 fp32 | 分别记录权重、计算和 Q/K/V dtype；避免把所有张量描述为 4-bit | F07、F15 |
| E2 包含 eager / sdpa / FA2 × GC | 能比较实际集成路径；要区分算法收益，另需受控后端组 | F09–F12 |
| 基准 token 数据是拼接流切块，无 padding / position_ids 边界输入 | 适合固定工作量性能对比；不能称为样本隔离 packing 或 padding-free SFT | F16–F17 |
| SFT collator 输出 padding mask 与 `labels=-100` | padding mask 控制 attention；`-100` 只控制 loss。两者职责不同，文档应明确 | F15–F17；collator 行为来自项目代码 |
| `gc.use_reentrant` 由配置控制，reentrant 时启用输入梯度 | 与冻结基座的梯度条件相符；FA2 内部重算不能替代该检查 | F13 |
| 数值诊断把 LoRA dropout 设为 0 | 再核实模型 attention dropout 实际值为 0；不同路径同随机种子不等于同 dropout mask | F07、F10、F15 |
| 正式计时覆盖完整 optimizer step，3 个独立进程重复 | 保留端到端口径；microbenchmark、profiler trace、正式训练三种产物分开解释 | F12、F14 |
| 当前 MFU >0.60 阻止引用 | 可作为人工复核触发器；不能写成“任何 A100 训练均不可能超过 60%” | F02、F18 |

## 需要重点辨别的交互

### Attention dropout、LoRA dropout 与评估

LoRA dropout 是 adapter 分支里的随机操作，attention dropout 位于 softmax 概率后；它们需要分别记录。外部 FA2 与 SDPA 都按调用参数执行 attention dropout，因此 `model.eval()` 的效果依赖模型适配层是否正确传 0。当前对照诊断只在项目配置中显式关闭了 LoRA dropout；模型 config 的 attention dropout 仍应作为运行证据读取。[F07](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/flash_attn/flash_attn_interface.py)、[F10](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[F15](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py)

### 因果遮罩、缓存与变长

训练中 Q/K 等长时，左上 / 右下 causal 对齐一致。生成中 Q 很短、K 包含历史缓存，不等长语义就必须检查；可用一个手工构造的短矩阵解释预期可见范围。padding unpad 负责去掉填充值；独立样本的边界仍要由累计长度或适配层提供。不能把“去 padding”推导成“自动阻止跨样本信息流”。[F05](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/README.md)、[F10](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[F16](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_flash_attention_utils.py)

### FA2 与 gradient checkpointing

FA2 反向依据保存的统计重建局部 attention 概率；decoder GC 则可重新执行更大前向片段，包括投影、归一化与 MLP。二者节省的张量和重算范围有重叠，收益与开销不能相加套公式。冻结基座也需要沿输入方向传播梯度，才能训练更早层的 adapter。现有梯度连通检查及 GC 开关日志应保留。[F01](https://arxiv.org/pdf/2205.14135)、[F13](https://docs.pytorch.org/docs/2.5/checkpoint.html)

## 推荐学习顺序与可交付物

1. **先读 F03 → F01：** 能手算两个分块的最大值、分母合并；画出 eager 保存的 `N×N` 张量与 FA 保存的逐行统计。
2. **再读 F04 → F02：** 能解释减少非矩阵运算、增加序列并行、sliced-Q 三件事分别改善什么；分清存储复杂度和计算复杂度。
3. **读 F09 → F10 → F11 → F15：** 写出本项目 eager / SDPA / 外部 FA2 的调用路径，列出自动选择和回退条件。
4. **读 F05 → F06 → F07：** 形成针对选定 tag 的版本与形状记录，明确架构、dtype 与输入约束。
5. **读 F16 → F17：** 给连续流 packing、样本隔离 packing、padding mask 和 loss mask 各画一个短例子。
6. **读 F08 → F13：** 整理输出 / 梯度容差、dropout 与 GC 条件。有限步 loss 接近仅是模型级诊断证据。
7. **读 F12 → F14 → F18：** 设计独立的内核确认任务，保存 trace，并依据实际测量记录性能。

优先的文档与代码改进项：在运行元数据中加入真实 attention dropout、Q/K/V dtype 和 attention 类；增加可选的 SDPA 单后端诊断入口；增加单独保存 profiler trace 的方案；明确 benchmark 连续流与 SFT 样本边界的定义。
