# QLoRA 与 FlashAttention-2 资料库

资料核对日期：**2026-10-24**。资料范围包括论文、作者仓库、Hugging Face / PyTorch / NVIDIA 官方文档及版本化源码，并说明其与 Qwen1.5-14B 单卡微调的对应关系。

## 从这里进入

| 文档 | 内容 | 资料目录条目数 |
| --- | --- | ---: |
| [QLoRA 专题](qlora.md) | NF4、双重量化、冻结梯度、dtype、Paged Optimizer、显存账本、4-bit kernel 路径 | 25 |
| [FlashAttention-2 专题](flashattention-2.md) | IO-aware attention、online softmax、FA2 并行分工、SDPA、mask、varlen、GC、数值与 profiling | 18 |
| [工程集成与测量](engineering-and-measurement.md) | Qwen 身份、版本源码、Trainer/loss、检查点、CUDA/NVML/UVM、MFU/HFU | 33 |
| [扩展论文与官方课程](related-work-and-learning.md) | LoRA/量化变体、Triton、CS336、FA3/FA4、CCE/Liger | 30 |
| [数据与评测资料](data-and-evaluation.md) | DuReader、CSDS、LCSTS、中文 ROUGE 与 EM/F1 口径 | 10 |

上述 **116 个目录条目存在交叉引用**，不是 116 篇不同论文。目录内实际引用 111 个不同 URL，规范化后为 **107 个来源条目**，包括不同版本的官方源码/API 页面；作者站与 arXiv 的不同站点入口仍可能指向同一论文。去重目录见 [source-catalog.json](source-catalog.json)，计数和归并规则见 [检索方法](search-method.md)。条目区分论文正文、仅摘要、文档和源码阅读；“已阅读”不代表逐行精读全部代码或验证全部实验。

- [阅读路线与知识地图](reading-path.md)：从原理到实现的顺序，以及 12 道自检题。
- [对项目的影响与待修订项](project-implications.md)：将研究结论转为后续代码、指标和实验设计任务。

## 最值得优先读的原始资料

1. [QLoRA 原论文](https://arxiv.org/html/2305.14314v1) 与 [作者仓库](https://github.com/artidoro/qlora)：先理解方法，再对照原训练配置。
2. [FlashAttention 原论文](https://arxiv.org/pdf/2205.14135)、[FlashAttention-2 原论文](https://tridao.me/publications/flash2/flash2.pdf) 与 [作者讲解](https://crfm.stanford.edu/2023/07/17/flash2.html)：先建立 IO 与分块直觉，再读并行分工。
3. [PEFT 量化指南](https://huggingface.co/docs/peft/en/developer_guides/quantization)、[bitsandbytes Linear4bit](https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit)、[优化器说明](https://huggingface.co/docs/bitsandbytes/en/explanations/optimizers)：把存储、计算、可训练参数和优化器分别定位。
4. [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) 与 [后端控制](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html)：理解接口、自动分派和实际 kernel 的区别。
5. [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：在代码中寻找逐块统计、累加器与前反向调度。

## 核心技术结论

| 结论 | 对项目意味着什么 |
| --- | --- |
| 当前 `mfu()` 随 GC 加入重算 FLOPs，与排除重算的 MFU 定义不同。 | 指标命名和公式需修订，分别描述任务 FLOPs 与实现重算成本。[PaLM](https://arxiv.org/pdf/2204.02311) |
| q/k/v/o 四投影与论文式 all-linear QLoRA 训练范围不同。 | 披露配置差异，扩展 all-linear 时重算参数量并独立比较。[PEFT](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora) |
| PyTorch SDPA 已有 FA2 支持。 | E2 比较的是集成路径；算法归因还需要后端控制与 trace。[PyTorch 发布说明](https://docs.pytorch.org/blog/pytorch2-2/) |
| FA2 避免完整 attention 中间矩阵，dense 计算仍为二次。 | 不把线性中间存储误写成线性训练耗时或线性总显存。[FA2 正文](https://tridao.me/publications/flash2/flash2.pdf) |
| dtype、4-bit matmul 分派、loss 与恢复行为随版本变化。 | 阅读版本只是样本，最终结论绑定实际安装源码与模型 revision。[版本化工程资料](engineering-and-measurement.md) |
| paged 配置、托管分配与真实迁移需要不同证据。 | E4 保留未知值，并区分 allocator、NVML 与 UVM trace。[bitsandbytes](https://huggingface.co/docs/bitsandbytes/en/explanations/optimizers) |

