# Qwen1.5-14B：QLoRA / FlashAttention-2 的集成与测量资料

资料核对日期：2026-10-24。本文收录官方模型仓库、版本化源码、原论文和厂商文档，说明 Qwen1.5-14B 的工程集成、指标定义与测量方法。

标记约定：**来源结论**是原资料直接支持的事实；**项目推导**是结合配置进行的计算；**待验证**是尚未通过本项目运行确认的行为。文中的历史版本是阅读源码的样本，不是本项目已锁定或已验证的环境。`main`、`stable`、无版本文档会变化。

## 首先掌握的结论

- 本模型虽然使用 `Qwen2ForCausalLM`，但 14B 权重配置的 Q 头和 KV 头均为 40，使用 MHA；不能将 Qwen2 系列的 GQA、滑窗、许可证或性能直接套用到它。
- QLoRA 的 4-bit 权重存储、bf16 运算、fp32 adapter、fp32 normalization 和 loss 可以同时存在；只报告“bf16 训练”不足以描述实际行为。
- `attn_implementation="sdpa"` 也可能进入 PyTorch 的 FlashAttention-2 内核。配置值只能证明请求路径，不能替代实际 kernel 证据。
- **现有 `mfu()` 按 GC 开关将系数从 4 改为 6，混入了激活重算成本。PaLM 定义的 MFU 排除重算；该指标的命名与计算需要区分任务 FLOPs 和重算成本。**
- `max_memory_allocated()` 只覆盖 PyTorch allocator；bitsandbytes 的 managed memory 及 CUDA 上下文等不能由它完整解释。NVML 整卡显存、allocator 峰值和 UVM 迁移是不同证据。
- 模型卡的 `transformers>=4.37.0` 是架构支持说明，不能证明项目使用的 `dtype`、`processing_class`、Trainer loss 归一化和恢复 API 全部兼容。

## 资料目录：模型身份、集成与版本

以下条目说明资料用途与适用边界；论文基准和官方示例的性能结果仅适用于其原始实验条件。

| ID | 一手资料与版本 | 支持的结论及边界 | 对应位置 |
| --- | --- | --- | --- |
| E01 | [Qwen1.5-14B config.json](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json)，模型仓库 main | **来源结论**：40 层、hidden 5120、intermediate 13696、40 Q/KV 头、vocab 152064、不共享词表权重、无滑窗。head dimension 可推得 128。应再固定具体 revision。 | E0 结构断言与参数量 |
| E02 | [Qwen1.5-14B 模型卡](https://huggingface.co/Qwen/Qwen1.5-14B)，Model Details / Requirements | **来源结论**：Qwen1.5 是 Qwen2 beta；本阶段除 32B 外未启用 GQA，未采用 SWA/full 混合；原生 Transformers 支持。基座与 Chat 有区别。 | 防止混淆模型身份 |
| E03 | [Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/)，Qwen 团队，2024-02-04 | **来源结论**：官方发布说明给出模型家族、32K 支持和 Transformers 集成背景。发布性能来自其评测设置，不能外推为单卡微调吞吐。 | 背景与报告引用 |
| E04 | [Qwen Technical Report](https://arxiv.org/abs/2309.16609)，2023 | **来源结论**：Qwen 原代模型的技术报告；是家族背景资料。它早于 Qwen1.5，不能覆盖本 checkpoint 的全部配置。 | 架构历史与训练背景 |
| E05 | [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)，2024 | **来源结论**：后续 Qwen2 系列报告明确以前代 Qwen1.5 为对比对象；两者不是同一权重发布。Qwen2 的结构与评测不能替代 E01。 | 后续工作与差异说明 |
| E06 | [Qwen1.5-14B LICENSE](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/LICENSE)，仓库许可证文本 | **来源结论**：本仓库标记 Tongyi Qianwen 许可。条款涉及再分发时提供协议、保留通知及修改声明，也包含使用限制。交付模型或衍生材料前应读取所固定 revision 的原文；不能把它标成 Apache-2.0。 | 模型来源说明 |
| E07 | [PEFT k-bit 准备函数源码](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/utils/other.py)，v0.17.1，`prepare_model_for_kbit_training` | **来源结论**：冻结基座；将符合条件的非 `Params4bit` fp16/bf16 参数转 fp32；GC 默认开启，并区分 reentrant。函数名和旧 docstring 不能替代实现。此版本不是 TF v5 推荐搭配。 | 模型工厂的准备顺序与 dtype |
| E08 | [torch.utils.checkpoint](https://docs.pytorch.org/docs/2.9/checkpoint.html)，PyTorch 2.9 | **来源结论**：建议显式 `use_reentrant=False`；non-reentrant 可提前结束重算；两种实现对梯度条件和 RNG 等有差异。因此 GC 不保证恰好重算一整次 forward。 | GC 成本与梯度连接 |
| E09 | [Qwen2 模型源码](https://raw.githubusercontent.com/huggingface/transformers/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py)，Transformers v4.37.0 | **来源结论**：独立 eager/SDPA/FA2 类；因果 LM 在模型内部 shift 后用 `CrossEntropyLoss()`。该 forward 未提供现代 `num_items_in_batch` 路径。 | 最低架构支持≠现代 Trainer 语义 |
| E10 | [Qwen2 模型源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py)，v4.57.1 | **来源结论**：通过 attention registry 选择后端；LM 将 `**kwargs` 传到 `loss_function`。RMSNorm 内部的 fp32 累积与权重存储 dtype 应分别理解。 | 版本化集成参考 |
| E11 | [Qwen2 模型源码](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen2/modeling_qwen2.py)，v5.0.0 | **来源结论**：同样保留 `loss_function(..., **kwargs)` 路径。只证明这个 tag 的实现；不证明全部 5.x 行为一致。 | 4.x / 5.x 对照 |
| E12 | [Transformers FA2 适配源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/flash_attention.py)，v4.57.1 | **来源结论**：对 fp32 Q/K/V 可根据 autocast / quantization dtype 转回支持精度；源码提醒 fp32 norm 可能引入转换开销。是否发生转换仍需观察实际张量。 | PEFT + FA2 dtype 衔接 |
| E13 | [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)，2.9 | **来源结论**：SDPA 有 FlashAttention-2、memory-efficient、math 实现并自动选择；可用 `sdpa_kernel()` 控制。不同融合实现可能有数值差异，限制随输入变化。 | E2 / E3 合理对照与数值容差 |
| E14 | [torch.amp](https://docs.pytorch.org/docs/2.9/amp.html)，2.9 | **来源结论**：autocast 按算子选择 dtype；未列出的运算依其输入类型执行。参数存储 dtype 不等于每个算子执行精度。 | 解释 bf16 + fp32 混合 |
| E15 | [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)，HF，2024-10-16 | **来源结论**：变长 token 任务应将整个累积窗口的 loss 总和除以有效 token 总数；对每个 microbatch 的均值再平均会改变权重。发布文章不保证所有版本/模型均已修复。 | E5 变长 SFT loss |
| E16 | [Trainer 源码](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/trainer.py)，v4.57.1 | **来源结论**：会解包 PEFT 检查 loss kwargs；`_get_num_items_in_batch()` 以未屏蔽 label 计数。正确归一化仍依赖模型 forward/loss 的传递链。 | 运行前记录的训练语义 |
| E17 | [Trainer 源码](https://raw.githubusercontent.com/huggingface/transformers/v5.0.0/src/transformers/trainer.py)，v5.0.0 | **来源结论**：具有类似的 token 计数与 kwargs 检查，但分布式条件已有变化。单卡项目不应声称多卡行为已覆盖。 | 跨大版本边界 |
| E18 | [loss_utils 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/loss/loss_utils.py)，v4.57.1 | **来源结论**：causal loss 将 logits 转 fp32；有 `num_items_in_batch` 时用 sum 后除该值，无此参数时用 mean。末位预测通过 ignore label 排除。 | loss 正确性与大词表显存 |
| E19 | [loss_utils 源码](https://raw.githubusercontent.com/huggingface/transformers/v5.0.0/src/transformers/loss/loss_utils.py)，v5.0.0 | **来源结论**：该 tag 仍包含同类 fp32 logits 与 sum/mean 分支。不能据此推断采用了 fused/chunked CE。 | 全词表 logits 的分配边界 |
| E20 | [Trainer 文档](https://huggingface.co/docs/transformers/en/main_classes/trainer)，无版本文档 | **来源结论**：`resume_from_checkpoint` 可恢复模型、优化器、scheduler；`save_only_model=True` 不保留完整恢复状态。具体文件格式和调用顺序须核对安装版本。 | checkpoint 与最终 adapter 的区别 |
| E21 | [PEFT checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)，官方格式说明 | **来源结论**：adapter 权重文件只保存适配参数，依赖配置和基座；合并导出又有额外限制。adapter 的可加载性不等于完整训练恢复。 | artifact 清单、身份校验 |
| E22 | [CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)，PyTorch 2.9 | **来源结论**：CUDA 异步，精确墙钟计时需同步，或用 events；allocated 和 reserved 含义不同。清 allocator cache 并不能增加 PyTorch 可用的张量内存上限。 | step time 与显存定义 |
| E23 | [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/2.9/torch_cuda_memory.html)，2.9 | **来源结论**：memory snapshot 只看到 PyTorch allocator 管理的分配；直接 CUDA 分配不可见。snapshot 是诊断材料，不自动补齐所有显存来源。 | paged optimizer 的 allocator 盲区 |
| E24 | [NVML Device Queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)，NVIDIA API | **来源结论**：Linux/TCC 下整卡 used memory 涉及设备活跃通道，MIG 句柄与权限影响范围，v2 reserved 计入 used。没有 UVM 迁移事件含义。 | telemetry 数据解释 |
| E25 | [bitsandbytes optimizer.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/optimizer.py)，main | **来源结论**：`get_state_buffer()` 对 paged 且元素数至少 100000 的参数使用 paged buffer；8-bit 状态另有较小张量阈值和尺度元数据。阈值是源码行为，须随实际版本固定。 | E4 是否具有 paged state |
| E26 | [bitsandbytes functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)，main | **来源结论**：`get_paged()` 调用 managed pointer 接口并包装张量；存在 prefetch 逻辑。`is_paged=True` 表示分配能力，不表示已发生压力驱动迁移。 | UVM 分配与迁移证据分离 |
| E27 | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)，Unified Memory 部分 | **来源结论**：可观察 UVM transfer、eviction、CPU/GPU page fault；fault 采集可能产生很大开销。带 profiler 的诊断应单独运行，不能直接作为正式吞吐。 | E4 迁移证据 |
| E28 | [PaLM 原论文](https://arxiv.org/html/2204.02311v5)，§4.1 Training Efficiency、Appendix B | **来源结论**：MFU 排除 rematerialization；HFU 的分析估计可包含重算。附录给出全参 dense 模型 `6N + 12LHQT` 的计数。本文下一节另推导冻结训练近似，不能将两者等同。 | 指标定义修正 |
| E29 | [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)，A100 性能表 | **来源结论**：A100 BF16 Tensor Core 峰值 312 TFLOPS，结构化稀疏可报 624；峰值基于 boost clock。应匹配实际型号及精度，且不是持续可达吞吐承诺。 | FLOPs 指标分母 |
| E30 | [bitsandbytes 8-bit optimizers 解释](https://huggingface.co/docs/bitsandbytes/en/explanations/optimizers)，Paged optimizers | **来源结论**：基于 CUDA unified memory，以迁移换取显存余量；代价受数据量、总线和访问模式影响。文档示例的 PCIe 数值不是本机带宽。 | E4 预期与限制 |
| E31 | [Transformers modeling_utils](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/modeling_utils.py)，v4.57.1 | **来源结论**：该版本支持 `dtype`，同时将 `torch_dtype` 视为兼容弃用参数。不能只根据模型卡 4.37 下限推断项目 `dtype=` 调用可用。 | 模型加载 API 边界 |
| E32 | [PEFT v0.18.0 release](https://github.com/huggingface/peft/releases/tag/v0.18.0)，Changes / Transformers compatibility | **来源结论**：官方明确 Transformers v5 与 PEFT <0.18.0 不兼容。升级一个库不构成完整可运行组合；FA2、bnb、PyTorch、CUDA 仍有各自约束。 | 版本组合边界 |
| E33 | [PEFT dtype troubleshooting](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)，Selecting the dtype of the adapter | **来源结论**：v0.12.0 起适用情况下自动将 adapter fp16/bf16 提升到 fp32；`autocast_adapter_dtype=False` 可改变行为。模型工厂的显式策略仍应记录。 | adapter 精度与统计 |

## MFU 的待修正口径

**来源结论（E28）**：PaLM §4.1 第 3 段至 Table 3 前明确区分 MFU 与包含重算的 HFU；Appendix B “Compute Usage and Environmental Impact” 开头给出全参训练基线。用“章节 + 段落”定位避免不同 PDF 版本页码漂移。A100 分母的规格依据见 E29。

**项目推导**：令 `P` 为稠密 bf16 峰值，`R` 为同口径 tokens/s，合理的命名应区分：

```text
MFU_task_est = R × C_required / P
HFU_analytic_est = R × (C_required + C_recompute + 其他计入的实现运算) / P
```

`C_required` 描述给定任务与冻结策略所必需的 forward/backward，GC on/off 应保持一致。后一指标也只是分析估算，不是硬件计数器实测。GC 若增加了 50% 运算却只降低 20% 吞吐，现代码可能让名为 MFU 的值升高 `1.5 × 0.8 = 1.2` 倍，掩盖其与任务效率的区别。

本项目的 `4N` 本身也不是标准“全参 6N”定义：对冻结线性层 `Y=XW`，forward 与输入梯度 `dX` 大体各需一次矩阵乘法，省去 `dW`；这是 `4N` 近似的来源。可训练 LoRA 分支仍有参数梯度，而 embedding lookup、第一层输入是否需要梯度、冻结输出头、非矩阵运算都要分别计数。因此可称为 **冻结训练任务 FLOPs 估计**，不能与 PaLM 全参 MFU、FA2 论文的完整模型训练 MFU 不加说明地比较。

根据 E01 的结构，本项目 `13.39e9` 可解释为以下矩阵参数近似，均非实测：

```text
Transformer block matrices = 40 × (4 × 5120² + 3 × 5120 × 13696)
                           = 12,609,126,400
lm_head                    = 5120 × 152064 = 778,567,680
N_eff                      = 13,387,694,080 ≈ 13.39e9
```

该值排除 embedding lookup，也忽略 norm、bias。它不是完整模型参数量。若 `4N` 假设成立，基础矩阵计算约为 53.55 GFLOPs/token；adapter 与 attention 需另加。GC 的 non-reentrant early-stop、FA2 自身 backward 的重算、基座是否量化，都使“一律额外 2N”只能是待核实的成本近似。

E28 附录的 attention 计数展开为 `12 × layers × hidden × sequence_length` FLOPs/token（全矩形 dense attention 计数）。代入本项目，S=2048 约 5.03 GFLOPs/token，S=8192 约 20.13 GFLOPs/token，分别约占 `4N_eff` 的 9.4% 与 37.6%。因果三角实现会改变工作量，不能把这里当 kernel 实际执行 FLOPs；但已足以否定长序列“attention 忽略项始终小于 5%”的默认假设。

## 训练 loss 与依赖版本的阅读边界

| 源码样本 | 已核实的差异 | 对项目的含义 |
| --- | --- | --- |
| Transformers v4.37.0（E09） | LM 内部 shift + 默认 mean CE | 是模型支持起点；变长 GAccum 不能借用新版 loss 假设 |
| v4.57.1（E10/E16/E18） | Trainer 识别 loss kwargs；Qwen2 传递 kwargs；CE 可用 token 总数归一化 | 三段必须连通；只见 `num_items_in_batch` 参数名不足以验收 |
| v5.0.0（E11/E17/E19） | 保留上述路径，但 Trainer 与集成实现继续变化 | 必须选配满足版本要求的 PEFT；不能将 E07 的 0.17.1 与 v5 拼成兼容组合 |
| PEFT v0.18.0 发布说明（E32） | 明确 TF v5 要求 PEFT >=0.18.0 | 是必要约束示例，不能作为所有后续 5.x 组合的充分条件 |

**项目推导**：目前 task 数据把非空 prompt 全部标为 `-100`，因此在已审计数据上，`labels.ne(-100)` 和 shift 后的有效目标计数一致；不要把 Trainer 历史源码的计数法推广到首 token 也有 label 的任意 causal 数据。benchmark 的 `labels=ids` 使用满标签，但直接调用模型、没有 Trainer GAccum 归一化，属于另一条链路。

**待验证**：锁定运行版本时，应保存实际 Qwen2 forward、Trainer `model_accepts_loss_kwargs`、loss 函数/归一化模式、有效目标计数及 PEFT helper 的实现身份。变长样本的“整 batch 与 microbatch 累积”等价诊断有助于核对归一化语义。

## 显存与性能的三类边界

**项目推导：全词表 loss 可能成为 FA2 之后的主要开销。** 词表 152064，`B=2, S=8192` 的 `[B,S,V]` 共 2,491,416,576 元素；bf16 约 4.98 GB，fp32 约 9.97 GB。E18/E19 中的 logits fp32 转换使这个量级值得单独审计。两者不一定全程同时存活，也不等于可直接相加的实际峰值。`prediction_loss_only=True` 只影响 Trainer 输出保留，不能证明模型内部没有生成全词表 logits。FA2 优化 attention，不负责消除这部分分配。

**项目推导：PEFT helper 的瞬时显存也要记录。** embedding 与 lm_head 各有 778,567,680 个权重，二者由 bf16 变 fp32 比原状态多约 3.11 GB（2.90 GiB）。工厂调用 helper 后再转回 bf16，可以降低最终常驻量，但中间的转换仍可能触发 OOM 或影响 reserved；“最终 dtype 正确”并不涵盖加载路径峰值。

**项目推导：E4 的可迁移状态规模有限。** 当前 q/k/v/o、r=64 的 LoRA 参数共 `40×4×64×(5120+5120)=104,857,600`。Adam 两组 8-bit 状态主体约 209.7 MB，另有尺度等元数据；每个 A/B 张量 327680 元素，超过 E25 的 paged 分配阈值。此配置适合观察 paged 路径，但即使分配成功，也不能预期出现很大的整卡显存差距，更不能直接宣称防止某个激活 OOM。

正式 step 墙钟应包含 forward、backward、clip、optimizer 以及所需同步；数据预载版本可以明确排除 H2D/数据整理。CUDA events 更接近设备区间时间，和含 Python dispatch 的墙钟不是同一指标（E22）。100ms NVML 采样有机会漏过短峰值；其 memory used 可以帮助观察外部分配，但不能反推出某个张量的驻留状态（E24/E26）。

## 实现审查与改进项

以下定位对应资料核对时的代码；“待验证”表示需要运行证据确认的技术问题。

| 优先级 | 当前位置 | 问题与改进建议 | 依据 |
| --- | --- | --- | --- |
| 高 | `src/llm_ft_repro/bench.py:14`；`docs/benchmark_protocol.md:17` | **确认的口径问题**：GC 改变名为 MFU 的运算系数。后续应明确任务 FLOPs 与包含重算的分析 HFU；同步修改汇总复算和引用门槛。`>0.6` 只是项目审查阈值，不是理论物理界限。 | E28/E29 |
| 高 | `src/llm_ft_repro/bench.py:65` | **待验证的峰值来源**：全词表 logits/CE 可能在 4K/8K 显著占用内存。使用 allocation trace 或阶段快照解释 OOM，再决定是否引入 chunked/fused loss；变更后实验协议也要重定义。 | E18/E19/E23 + 本文计算 |
| 高 | `src/llm_ft_repro/model_factory.py:127` | **现有证据不足**：只检查 `_attn_implementation` 不能证明实际 SDPA kernel。增加独立 profiler 证据并记录输入 dtype/shape、padding/causal 设置。 | E10/E12/E13 |
| 中 | `src/llm_ft_repro/model_factory.py:87` | **待验证的加载峰值**：helper 暂时上转两张大表，转回 bf16 不能消除之前的峰值。保留当前准备前后快照，后续另补过程峰值/受控准备策略研究。 | E07/E22 |
| 中 | `src/llm_ft_repro/model_factory.py:100` | **待验证的 dtype 开销**：RMSNorm 本身内部 fp32 计算，不必然要求权重也存 fp32；显式转换可能改变 hidden-state 类型与转换次数。不要在无数值/梯度证据下调整。 | E10/E12/E14 |
| 中 | `src/llm_ft_repro/train.py:74`；`pyproject.toml` | **版本相关的兼容语义**：processing_class、eval_strategy、dtype 与 loss kwargs 依赖版本。依赖锁定时应记录实际 API/源码行为，而不是以模型卡版本下限替代。 | E09–E19/E31/E32 |
| 中 | `src/llm_ft_repro/model_factory.py:15`；`src/llm_ft_repro/telemetry.py` | **allocator 盲区**：E4 外部 managed allocation 可能让 paged 方案的 allocated 更低，不能因此断言总显存等量降低。并列解释 PyTorch 与 NVML。 | E23–E26 |
| 中 | `src/llm_ft_repro/bench.py:47` | **范围声明**：全部 token batches 预载到 GPU 并常驻，峰值包含该基线，但计时排除数据传输；应把它命名为预载数据的计算步骤吞吐。 | 代码审查 + E22 |
| 中 | `src/llm_ft_repro/bench.py:92` | **统计口径**：`tokens/median(step)` 是典型步骤吞吐，`total_tokens/sum(step)` 才是整个计时区间吞吐，两者不能混称；已保存原始步时可支持两者。 | 本项目统计推导 |
| 中 | `src/llm_ft_repro/train.py:79`；`src/llm_ft_repro/artifacts.py` | **恢复边界**：文件哈希、identity 与 optimizer/RNG 文件清单是好的完整性证据；仍需运行证据证明 adapter、优化器、scheduler、RNG 和数据位置实际恢复一致。 | E20/E21 |
| 低 | `src/llm_ft_repro/bench.py:100` | 当前 `paging_observed=None` 是合适的未知状态。UVM profiler 采集须独立于正式吞吐，并区分分配、prefetch、fault、eviction。 | E25–E27/E30 |

## 接下来值得阅读的问题

1. 用 E07 → E08 → E12 连起来理解：冻结 embedding 后 LoRA 梯度为何仍存在，reentrant 与 non-reentrant 各需要什么条件，fp32 norm 是否引入多余转换。
2. 用 E09/E10/E11 比较版本演进：固定本项目安装版本后，重读相同位置；不要按最新版文档“自动升级并假定兼容”。
3. 用 E13 → E22 → E23 理解：同样的 attention 字符串如何进入不同 kernel，为什么 dtype、mask、padding 会改变性能与峰值。
4. 用 E25/E26 → E24 → E27 区分：paged state 存在、GPU 驻留量变化、实际页迁移各自需要哪些证据。
5. 用 E28 → E29 重写指标设计：先确定冻结训练任务所需 FLOPs，再单列重算与量化实现成本，最后与原始 tokens/s、步时、显存共同解释。

本文共收录 **33 个可直接定位的一手来源**。历史源码用于说明版本行为差异；环境兼容性、加速比、显存节省与恢复一致性结论均需相应的运行证据支持。
