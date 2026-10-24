# 检索范围、深度与目录规则

资料核对日期：2026-10-24。资料范围涵盖 QLoRA、FA2、工程测量，以及前置数学、邻近方法、课程与数据评测。技术结论以论文、作者仓库及官方文档等一手资料为依据。

## 检索问题与关键词

| 方向 | 核心问题 | 英文检索词 / 源码定位词 |
| --- | --- | --- |
| QLoRA | 权重量化、误差与训练路径 | QLoRA NF4 double quantization block size frozen weights |
| 量化实现 | 编码与前反向调用 | Linear4bit quant_state compress_statistics MatMul4Bit gemm_4bit |
| Paged Optimizer | 分配、预取与真实迁移 | PagedAdamW8bit get_state_buffer get_paged Unified Memory page fault |
| LoRA | 目标层、缩放与初始化 | target_modules all-linear alpha rank rsLoRA LoftQ PiSSA DoRA |
| FA2 原理 | IO、重算、并行划分 | FlashAttention-2 online softmax sliced-Q sequence parallelism |
| attention 集成 | 真实后端与支持条件 | scaled_dot_product_attention sdpa_kernel causal varlen cu_seqlens |
| 训练数值 | dtype、dropout、GC 与 loss | prepare_model_for_kbit_training use_reentrant num_items_in_batch loss_function |
| 测量 | 时间、显存与运算量 | CUDA synchronize memory allocated reserved NVML MFU HFU rematerialization |
| 大词表 | FA2 后的输出头峰值 | Cut Your Losses Cut Cross Entropy Liger FusedLinearCrossEntropy |
| 任务评测 | 中文问答与摘要协议 | DuReader CSDS LCSTS ROUGE jieba EM F1 |

## 阅读深度的含义

- **论文正文：** 已查看相关方法、算法或实验段落；标明章节，不代表全部附录均精读。
- **仅摘要：** 已核对原作者摘要与元信息，主要用于扩展阅读，不据此推导 kernel 细节或环境兼容性。
- **官方文档 / README：** 覆盖相关 API 或操作说明；文档说明不构成运行兼容性证据。
- **源码：** 静态阅读指定 tag 或资料核对日期对应 main 中的相关函数；不构成运行验证。

工程专题中的 E04/E05 为原作者论文摘要与模型背景，E28 为正文相关段落；其他条目按其明确标注的 README、文档、许可证或函数范围阅读。论文、文档和运行事实之间保留边界：来源说明“支持”不等于本项目已成功运行，代码中存在分支不等于某个 workload 实际命中它。

## 收录与计数

5 个资料目录共 116 个编号条目，实际引用 111 个不同 URL，按下述规则得到 107 个规范来源条目。相同资料可同时服务不同专题，版本化源码又是独立页面，所以这些数量都不能称为“独立论文数”。

机器目录 `source-catalog.json` 从这些专题中的实际引用链接生成，并保存所有引用位置。规范化时去掉 URL fragment 与显示用 query；同一 arXiv 编号的 abs / html / pdf 及版本表现合并为一项；GitHub raw 与相同 ref 的 blob 合并；版本不同的 API/源码页面保留为不同来源。目录也保留个别资料中链接到的另一种正文表示，因此目录条目数与手工编号数不必相等。

目录记录的是**来源定位与覆盖范围**，不是论文全文镜像或逐页阅读日志。部分 `main` 页面可能晚于历史实验方案，读取时间不能替代 commit。无版本官方文档重定向后的内容不构成已验证的环境锁。

## 已排除或保留为未知的情况

- 搜索结果中的转载、讨论帖和无出处性能数字，不作为工程断言的依据。
- 标题与实际 arXiv 内容不一致的候选链接，核实后排除，没有进入资料目录。
- 对单次访问超时的页面，不能凭 URL 猜测全文；采用可访问的官方资料，并标明阅读范围。
- 数据仓库可公开访问，不等于已完成数据包许可、下载可用性和哈希核验；这些仍留待选定来源后处理。
- 资料中的兼容说明、理论计算与性能结果均受各自版本和实验条件约束；项目建议与运行证据分别表述。
