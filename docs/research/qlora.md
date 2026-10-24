# QLoRA：论文、实现与本项目的对应关系

检索日期：2026-10-24（Asia/Shanghai）。本页实际检索并打开了 25 个一手资料页面，包含论文正文、官方文档、作者仓库和实现源码。只打开摘要的论文在目录中明确标为“摘要”；阅读源码表示静态阅读，未安装这些版本、未执行训练或 GPU 验证。`main` 链接是可变的，不能替代实验依赖锁和已安装源码哈希。下文的“项目推断”均不是实测。

## 先掌握的几个关系

QLoRA 通过冻结的量化基座传播梯度，只更新 LoRA 参数。冻结基座并不等于跳过整个基座的反向传播：前层 adapter 的梯度仍依赖后续层对输入的梯度。4-bit 主要描述权重存储，不能把激活、梯度、优化器、注意力计算全部记为 4-bit。[作者论文第 2–3 节](https://arxiv.org/html/2305.14314v1#S3)

理解代码时，将五种量分开：基座的量化编码、非量化参数 dtype、4-bit 层 compute dtype、adapter 参数 dtype、优化器状态 dtype。项目配置是 NF4 + double quantization + bf16 compute；本地 `model_factory.py` 又显式把可训练参数转成 fp32，因此本项目不能直接套用论文中 bf16 adapter 的显存账本。[PEFT dtype 文档](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)

Paged 和 8-bit 也是两个维度：前者管理优化器状态的内存位置，后者压缩状态表示。`PagedAdamW8bit` 同时采用两者；“使用 paged 优化器”“分配了 managed memory”“实际发生 CPU/GPU 页迁移”是三条需要分别取证的事实。[bitsandbytes 优化器说明](https://huggingface.co/docs/bitsandbytes/en/explanations/optimizers)

## 一手资料目录

每项同时记录可信结论、使用边界和项目用途。编号用于本页内部定位；链接均指向实际阅读的页面。

| 编号 | 来源与阅读深度 | 可据此得到的结论 | 使用边界与项目用途 |
| --- | --- | --- | --- |
| Q01 | [QLoRA 原论文，2023](https://arxiv.org/html/2305.14314v1)；正文第 2–4 节、附录 E/G | NF4、双重量化和分页状态组成方法；论文的一级块为 64、二级块为 256；量化基座冻结。 | 65B/48GB 是论文特定设置；不能换成 Qwen 中文任务的质量或吞吐保证。优先读方法和显存分解。 |
| Q02 | [LoRA 原论文，2021](https://arxiv.org/abs/2106.09685)；摘要 | 冻结预训练权重，用低秩分解表示可训练更新。 | 这里只核对摘要；不把论文模型上的收益倍数直接移植到本项目。用于辨认 LoRA 与全参数训练。 |
| Q03 | [8-bit Optimizers 原论文，2021/2022](https://arxiv.org/abs/2110.02861)；摘要 | 分块动态量化压缩优化器统计量；论文包含稳定 embedding 的设计。 | 压缩对象是优化器状态；本项目 embedding 冻结，不应把 StableEmbedding 收益写入项目结论。 |
| Q04 | [NF4 Isn't Information Theoretically Optimal (and that's Good)，2023](https://arxiv.org/html/2306.06965v1)；正文第 1–3 节、摘要 | absmax 归一化后的分布随块大小变化，固定 NF4 码本不具有无条件最优性；作者明确不否定 QLoRA 的经验结果。 | 这是理论边界和初步实验，不能反推“NF4 无效”。文档宜写“面向近似正态权重设计”。 |
| Q05 | [作者 QLoRA 仓库](https://github.com/artidoro/qlora)；README 的量化、优化器、复现说明 | 提供 Guanaco 训练入口和原始配置；README 展示 `paged_adamw_32bit`。 | 历史示例有当年的依赖与 API；当前项目的 paged 8-bit 是另一个明确的配置选择。 |
| Q06 | [作者 `qlora.py`](https://raw.githubusercontent.com/artidoro/qlora/main/qlora.py)；`find_all_linear_names`、模型准备、dtype 分支 | 作者扫描线性模块作为 LoRA 目标并排除 `lm_head`；代码显式处理 norm、embedding/head 精度。 | 项目只注入 q/k/v/o，应披露目标差异；不能照抄作者打印统计时的旧 4-bit 计数修正。 |
| Q07 | [Hugging Face 联合发布文章](https://huggingface.co/blog/4bit-transformers-bitsandbytes)；方法与资源段 | 给出库集成关系和官方 notebook 入口，解释按需解压的存储/计算分工。 | 文章含 2023 年安装命令与 API。用于入门和图解，不用于锁定现在的环境。 |
| Q08 | [Transformers bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)；4-bit、dtype、nested quantization 与训练段 | `bnb_4bit_compute_dtype` 控制计算；NF4 与 double quant 各有显式配置；文档将 `device_map="auto"` 留给推理。 | 项目单卡显式 `device_map={"":0}` 更便于保持测量边界。加载 dtype 不等于 adapter dtype。 |
| Q09 | [PEFT 量化指南](https://huggingface.co/docs/peft/en/developer_guides/quantization)；量化、LoRA、LoftQ 段 | 给出 `BitsAndBytesConfig → prepare_model_for_kbit_training → get_peft_model` 顺序；支持 NF4/bf16/DQ。 | 教程 q/k/v/o 示例与论文式 all-linear 需要区别。作为模型工厂的 API 对照。 |
| Q10 | [PEFT LoRA 参考文档 v0.21.0](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)；初始化和 QLoRA-style 段 | 默认初始化使 adapter 初始贡献为零；论文式 QLoRA 可用 `target_modules="all-linear"`。 | `all-linear` 会改变参数量和训练容量。当前协议固定 104,857,600 参数，扩展实验应另立组。 |
| Q11 | [PEFT `utils/other.py`](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/utils/other.py)；`prepare_model_for_kbit_training` 函数 | 先冻结参数；所读分支把非 `Params4bit` 的 fp16/bf16 参数上转 fp32；GC 默认开启，可显式关闭。 | 传 `use_gradient_checkpointing=False` 只控制 GC，不关闭 dtype 转换。项目已恢复大表为 bf16，应记录转换前后与加载峰值。 |
| Q12 | [PEFT troubleshooting](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)；dtype 小节 | 文档说明 v0.12.0 起 adapter 可自动上转 fp32；`autocast_adapter_dtype=False` 能控制相关默认行为。 | 参数存储精度与 autocast 下计算精度仍需分别记录。项目已主动固定训练参数 fp32。 |
| Q13 | [bitsandbytes Linear4bit API](https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit)；完整模块说明 | `Linear4bit` 支持量化类型、compute dtype、压缩统计与 quant storage；示例量化发生在设备转换时。 | 裸模块默认值包含 `quant_type='fp4'`，不能以“4-bit”代替“已确认 NF4”。应审计实例字段与 quant state。 |
| Q14 | [bitsandbytes `functional.py`](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/functional.py)；`quantize_4bit`、`get_paged` | DQ 分支中心化 absmax 后再次量化；保存 offset/state2；默认一级块为 64。托管内存可经自有 C 接口分配。 | `numel × element_size` 只覆盖 packed 参数，不含这些旁路张量；模型显存清单应递归盘点 quant state。 |
| Q15 | [bitsandbytes 0.43.3 `MatMul4Bit`](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.43.3/bitsandbytes/autograd/_functions.py)；前反向源码 | 该历史版本前向显式反量化后执行 `F.linear`；反向支持输入梯度而不计算量化权重梯度。 | 用于解释旧实现；不是本项目推荐版本，也不能代表全部后续版本。 |
| Q16 | [bitsandbytes main `MatMul4Bit`](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/autograd/_functions.py)；前反向源码 | 本次读到的前向改用 `gemm_4bit` 分派，并处理 nested quant；输入梯度仍有反量化矩阵乘。 | 单看入口不能判断实际 CUDA kernel。需联读 Q25，避免把算子名称当作融合证据。 |
| Q17 | [bitsandbytes AdamW API](https://huggingface.co/docs/bitsandbytes/en/reference/optim/adamw)；类与参数说明 | `optim_bits`、`is_paged`、`min_8bit_size` 是不同配置维度；小张量阈值影响实际状态精度。 | 不能把全部状态简单等同于每元素 1 byte；项目应按实际状态张量核算。 |
| Q18 | [bitsandbytes 优化器原理](https://huggingface.co/docs/bitsandbytes/en/explanations/optimizers)；8-bit 与 paged 段 | 状态可在更新时解量化；分页建立在 CUDA Unified Memory 之上，迁移性能与总线和访问方式有关。 | 文档的带宽估算是示例，不是本项目实测。E4 应报告观察证据、时间与显存，保留未知状态。 |
| Q19 | [bitsandbytes `optim/optimizer.py`](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/optim/optimizer.py)；状态初始化、预取、step | 本次源码仅在 GPU 参数不少于 100,000 元素且 paged 时分配分页状态；step 中存在预取和同步。 | `is_paged=True` 不保证所有状态分页，也不证明迁移；库版本变化可能影响步时。 |
| Q20 | [PEFT LoRA/bnb 实现](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/bnb.py)；4-bit forward、merge | 前向分别计算基座和 adapter 分支；merge 路径会反量化、加入增量并重新量化，源码提示舍入可能改变生成。 | QLoRA adapter 合并不是自动无损；评测应固定部署形态，合并产物另作评测。 |
| Q21 | [Transformers 4-bit quantizer](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/quantizers/quantizer_bnb_4bit.py)；模块替换与序列化字段 | 本次实现确实把 `llm_int8_skip_modules` 送入 4-bit 排除列表，并列出 nested quant 元数据。 | 参数名有 int8 不代表 4-bit 分支不使用；仍需核查项目所锁版本和最终模块类型。 |
| Q22 | [PEFT checkpoint 格式](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)；文件格式与 base model 信息 | adapter 权重文件只含 adapter 参数，配置记录基座、revision 和 adapter 结构。 | adapter 文件不等于完整训练恢复包；项目另存优化器、调度器、RNG、数据游标及身份信息是合理补充。 |
| Q23 | [bitsandbytes FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/en/fsdp_qlora)；quant storage 段 | `quant_storage` 可用浮点容器承载打包量化数据，以配合 FSDP；compute dtype 是另一概念。 | 单卡项目暂不需要 FSDP；学习该页用于防止把浮点 storage 容器误认为每权重恢复成浮点精度。 |
| Q24 | [LoftQ 原论文](https://arxiv.org/abs/2310.08659)；摘要 | 同时设计量化与低秩初始化，以缓和量化模型和全精度模型之间的差距。 | 仅摘要阅读；不推导实现细节。适合质量方向后续扩展，应独立于当前量化/注意力速度矩阵。 |
| Q25 | [bitsandbytes CUDA backend](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/backends/cuda/ops.py)；`gemm_4bit` 分派、fallback 与阈值 | 本次源码按架构、dtype、M/N/K 选择 custom kernel 或反量化后 `F.linear`；CUDA 的 custom 路径上限 M 为 1536。 | 这是分派逻辑观察，不是测速。本项目默认 B=2、S=2048，展平 M=4096，按此源码推断会回退。 |

## 按执行过程理解关键技术

### NF4：码本、块与旁路元数据

NF4 可理解为 16 个非均匀重建值的索引；块内权重先按 absmax 归一化，再映射到码本，重建时乘回尺度。归一化分布会受到块大小影响，因此“4-bit 浮点”不足以完整描述误差来源。[NF4 理论边界论文](https://arxiv.org/html/2306.06965v1)

double quantization 的第二次量化针对尺度统计量。当前 `functional.py` 对一级 absmax 减均值、执行块大小 256 的二级量化，并保留 offset 与 state2。它没有把每个 NF4 索引再压缩成 2 bit。[bitsandbytes 量化源码](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/functional.py)

根据一级块 64、尺度 fp32 和二级块 256 的理想账本，单参数元数据从 `32/64 = 0.5 bit` 降到 `8/64 + 32/(64×256) = 0.126953125 bit`，节约 `0.373046875 bit`。量化部分合计约 `4.126953125 bit/参数`。这些值只计主体编码和块尺度，不含张量对象、码本、对齐及 allocator 开销。[QLoRA 方法第 3 节](https://arxiv.org/html/2305.14314v1#S3)

**项目建议：** E0 保存每类量化线性层的 `quant_type`、`quant_state.blocksize`、`nested`、packed shape 和原始 shape，显存账本单列 `absmax`、`state2.absmax`、code 与 offset。不要修改量化块大小来凑显存估算。

### 梯度：冻结的权重仍参与输入梯度

一个量化线性层加 LoRA 可写成：

```text
Y = X × dequant(Wq) + scale × dropout(X) × A × B
```

这是帮助读代码的表达式，转置取决于实现约定。优化器只接收 A/B 等可训练参数，但前一层 adapter 的更新仍需要本层的输入梯度。bitsandbytes `MatMul4Bit.backward` 的 `grad_A` 指该算子的第一个输入的梯度，**不是** LoRA 的 A 矩阵；其 `grad_B=None` 对应量化基座权重。[当前 autograd 源码](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/autograd/_functions.py)

**项目建议：** 后续 GPU 验收分别检查基座 `requires_grad=False`、adapter 有限且非空的梯度、优化器参数集合只覆盖 adapter。不能用全模型 `torch.no_grad()` 包住量化基座训练前向。首次 step 中零初始化的 LoRA B 可能使 A 的初始梯度为零，应检查多步更新，避免把初始化行为误判成断梯度。[LoRA 初始化说明](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)

### PEFT 准备函数：显式处理 GC 与 dtype

准备函数的源码行为比函数名更重要：本次 `main` 在关闭 GC 时仍然冻结并上转非 `Params4bit` 的半精度参数；默认 GC 分支还可能启用输入梯度 hook。[PEFT helper 源码](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/utils/other.py)

本项目已经显式关闭 helper 内部 GC，再由项目统一开启或关闭；也已经把 embedding/head 恢复为 bf16、norm 保持 fp32、adapter 固定 fp32。这些策略应继续保留在 E0 记录中。**项目推断：** 大表先上转再恢复，可能造成加载峰值高于最终静态显存；最终“能放下”不能单独保证加载过程无 OOM。

### 分页优化器：两种阈值与三种证据

不要混淆 8-bit 小张量阈值与分页分配阈值。API 中 `min_8bit_size` 默认 4096；本次 `get_state_buffer` 另按参数 `numel < 1e5` 决定是否走分页分配。[AdamW API](https://huggingface.co/docs/bitsandbytes/en/reference/optim/adamw)、[状态分配源码](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/optim/optimizer.py)

本项目每个注意力 LoRA 矩阵有 `5120×64 = 327,680` 个元素，**项目推断**：在所读源码、GPU 参数和现有 rank 下，A/B 状态具备进入分页分配的大小条件。仍不能据此断言实际向 CPU 迁移；还需要 Unified Memory trace 等证据。E4 比较两个 8-bit 优化器能集中观察 paging 维度，但不是 8-bit 与 fp32 状态精度比较。

### 速度：训练形状决定路径

旧版 0.43.3 与本次 main 的实现入口已经不同；本次 CUDA 后端依然可能回退到反量化再调用 `F.linear`。这说明“QLoRA 只有反量化实现”和“当前 QLoRA 一定使用融合 kernel”都过于笼统。[历史实现](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.43.3/bitsandbytes/autograd/_functions.py)、[当前分派](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/backends/cuda/ops.py)

**项目推断：** 默认 packed 批次给线性层的 M 约为 `micro_batch×seq_len=4096`，超过本次 CUDA custom 路径的 1536 上限。后续应在固定版本中记录 kernel trace，再解释吞吐。推理单 token 的 kernel 收益不能直接用于预测本项目的长序列训练。

QLoRA、FA2 和 GC 的收益也不能直接相乘：它们分别影响权重与状态、注意力中间量与调度、激活保存与重计算，最终步时由总工作负载决定。此处为**工程推断**，具体结论仍由 E1–E4 的同硬件、同 token 工作量测量给出。

## 本项目的显存账本：理论推算示例

以下仅按仓库当前固定结构及 dtype 策略计算；不是模型下载后的参数审计，不是 GPU 实测。`GiB=2^30 bytes`。假定每层 q/k/v/o 均为 `5120×5120`，三个 MLP 矩阵为 `5120×13696`，40 层，embedding 与 head 不共享且均为 `152064×5120`。

| 项目 | 推导 | 理论占用 |
| --- | --- | --- |
| 可量化线性权重 | `40×(4×5120² + 3×5120×13696) = 12,609,126,400` 参数，按上文 4.126953125 bit | 约 6.058 GiB |
| bf16 embedding + lm_head | `2×152064×5120 = 1,557,135,360` 参数，每参数 2 bytes | 约 2.900 GiB |
| 基座以上两项小计 | 尚不含 norm、bias、碎片、上下文与中间量 | 约 8.958 GiB |
| DQ 相对未压缩 fp32 尺度节约 | `12,609,126,400×0.373046875/8` bytes | 约 0.548 GiB |
| 四投影 LoRA 参数 | `40×4×64×(5120+5120)` | 104,857,600 参数 |
| fp32 adapter + fp32 梯度 + 两个 8-bit 状态 | `104,857,600×(4+4+1+1)` bytes，忽略状态尺度 | 约 0.977 GiB |
| fp32 logits 的一个完整张量 | `2×2048×152064×4` bytes | 约 2.320 GiB |

这些公式由当前配置直接算出，不能把小计相加后当作精确峰值：激活、logits、临时反量化、优化器状态的生命周期并不同步，实际 loss 实现还可能额外复制或切片。表格的作用是标出“14B×0.5 byte”之外的重要项，而不是承诺某档 GPU 一定可跑。

**项目建议：** 按加载后、helper 后、精度恢复后、adapter 注入后、首个 forward/backward/step 后记录实际占用；优化器懒初始化使第一个 step 后的状态量尤其重要。对 paged 状态，同时看 PyTorch allocator 与设备总占用；仅用 `memory_allocated` 无法完整描述库外分配。[状态初始化与托管分配源码](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/optim/optimizer.py)、[get_paged 源码](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/functional.py)

## 对现有协议的具体建议

1. **方法命名要披露目标层。** 当前固定 q/k/v/o、rank 64，适合作为“NF4 QLoRA + 四投影 LoRA”实验；all-linear 会加入 MLP，改变训练容量和参数量，应独立扩展并与同目标的 bf16 LoRA 配对。[官方 QLoRA-style 说明](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)
2. **锁定五类精度。** 将 NF4/DQ、base 非量化 dtype、compute dtype、adapter dtype、state dtype 写成独立字段，并保存准备函数和关键 kernel 源码身份。当前代码已有 helper 哈希与前后清单，可扩展到 quant state 与 optimizer state。
3. **保留分页证据的未知值。** E4 的 `paging_observed=null` 合理；只有 trace 能支持“发生迁移”。单凭优化器类名、NVML 波动或不 OOM 都不能完成归因。
4. **让性能解释绑定版本与张量形状。** 至少记录 torch/transformers/peft/bitsandbytes/flash-attn 版本、CUDA 与 GPU、M/N/K、序列长度、有效 token、GC、attention 后端。对比中任何一项变化都要单独披露。
5. **让质量结论绑定评测形态。** 当前 E6 使用相同 NF4 基座前后比较，能回答 adapter 的效果；若要回答“量化损失多少”，另需同 prompt 的 bf16 原始基座对照。合并重新量化的模型作为另一个产物评测。[PEFT merge 源码](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/bnb.py)
6. **将扩展留给独立实验。** all-linear、LoftQ、fp32 paged、不同 rank 都是有依据的扩展，但不宜混入当前 attention×GC 矩阵。先让每个实验只回答一个清楚的问题。[LoftQ 摘要](https://arxiv.org/abs/2310.08659)

## 建议阅读顺序

先读 Q01 第 2–3 节建立存储/梯度心智模型，再读 Q08/Q09 对应 API；然后用 Q11/Q12 核对 dtype 与 GC，沿 Q13/Q14/Q16/Q25 追踪量化层实际路径，最后用 Q17–Q19 理解分页证据。需要论文忠实程度时对照 Q06/Q10；需要理论边界时读 Q04；质量扩展看 Q24；恢复与部署看 Q20/Q22。Q23 作为未来多卡知识，当前单卡项目无需执行其部署流程。
