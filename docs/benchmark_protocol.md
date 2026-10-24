# GPU 基准测量规程

此文档承接原计划的 gpu-bench-protocol 操作要求，不依赖特定 Agent 平台的技能目录。

**研究后的待修订项（2026-10-24）：** 下文沿用初版规格中的 MFU 近似。新核对的文献指出，MFU 应排除 GC 重算，现有代码的 4N/6N 随 GC 变化混入了重算成本；60% 也不是通用物理上限。正式实验和报告前应先按 [研究结论](research/project-implications.md) 修订指标与 schema。[PaLM §4.1 / 附录 B](https://arxiv.org/pdf/2204.02311)

## 固定项

每个正式配置独立进程，3 次重复，每次 10 步预热后测 30 步。基准 grad_accum=1；学习率保持固定，无调度器，全部配置相同。相同长度和 batch 的配置复用同一 token 流前缀，数据哈希写入日志。

计时边界是同步后的 wall time：zero_grad → forward → backward → clip → optimizer.step → synchronize。报告步时中位数、IQR；吞吐 = 输入 token 数 / 步时。另记录实际 causal 预测位置数 `batch × (seq - 1)`，两者不要混用。第一次梯度连通性检查发生在预热中；数值诊断的第一步额外开销不用于性能结论。

预热后重置 allocated / reserved 峰值，报告主指标为 PyTorch `max_memory_allocated`。保留 load 后、LoRA 后、第一次 forward/backward 后的静态快照。NVML 每 100ms 采样整卡占用、SM 时钟、功耗、温度和 throttle reasons；采样不是精确峰值捕获，可能漏过短脉冲。整卡指标会包含其他进程和 CUDA 上下文。

标准单位：字节为原始存储，GB = bytes / 1e9。若展示 GiB，必须另写 bytes / 2^30。不得把 nvidia-smi MiB 原数当成 GB。

## MFU 与理论边界

估算 `N_eff = 13.39e9`：无 GC 每 token 4N FLOPs，有 GC 6N。MFU = tokens/s × FLOPs/token ÷ 标称稠密 bf16 峰值。A100 默认分母 312e12，MFU > 0.60 的行自动阻止引用。

该口径是运算量估算，忽略 attention、LoRA 与量化开销；冻结层的输入梯度计算、头部计算、实际重算范围可能不同。尤其 seq=8192 时不能直接沿用“忽略项小于 5%”的说法。MFU 不能代表硬件计数器测得的真实利用率。

全参训练显存只做理论讨论：按每参数总计约 16 字节的某种混合精度 Adam 状态布局，14.17B × 16 ≈ 226.7GB，尚未计激活。原计划中“梯度与优化器 198GB”对应总计扣掉权重后的部分，不能再次按每参数 16 字节加到权重上。具体优化器精度会改变该估算。

## 矩阵

| 实验 | 配置数 | 重复 | 注意事项 |
| --- | ---: | ---: | --- |
| E1 | 2 | 3 | bf16 LoRA、QLoRA，FA2+GC；全参不执行 |
| E2 | 6 | 3 | eager / sdpa / FA2 × GC；另保留 OOM 降级点 |
| E3 | 15 | 3 | 长度 512 / 1024 / 2048 / 4096 / 8192，batch 8 / 4 / 2 / 1 / 1 |
| E4 | 2 | 3 | 两种 8-bit AdamW；换页未知时不推断 |

E3 前四档每步 4096 输入 token，最后一档 8192 token；横向解释需注明有效 batch 和每步 token 变化。只能声称“已扫描范围内最大成功长度”，没有更长失败点时不能声称找到了最大可训练长度。

## 一致性审查

- 参数实际计数必须等于结构设计值；GC 和 attention 实际状态必须一致。
- 静态显存估算偏差 >15% 时，解释非量化层精度、allocator、量化元数据，不能改估算贴结果。
- 3 次独立进程吞吐极差 / 中位数 ≤10%；重复丢失或额外重复不自动挑选最好结果。
- 数值诊断使用相同初始化种子、数据、优化器、0 dropout；保存容差与损失差。有限步 loss 比较不等于数学证明。
- 对 GC 步时增幅与估算 +33%～+50% 的差异写解释；没有解释不能生成百分比宣传。
- 检查 GPU UUID、驱动、功耗墙、温度、时钟、其他进程占用；单个 NVML throttle bit 不自动归因为热降频。
- `sdpa` 可能自动选择不同内核。本代码记录请求的实现和版本，不声称已经证明其具体底层 kernel；要确认需 profiler。

## OOM 与换页

异常保存 traceback 和可获取的 allocator 统计，结束当前进程。只有 E2 eager/no-GC 的独立降级配置会自动追加，所有失败保留。

NVML 占用量和 optimizer.is_paged 不能证明发生了 UVM 页迁移。E4 的 `paging_observed=null` 表示未知。是否换页需要 Nsight/CUPTI 等另行采集证据，在未采集前不写“已换页”“未换页”或“防止 OOM”。
