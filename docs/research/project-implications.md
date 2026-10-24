# 文献对本项目的影响与待修订项

检索日期：2026-10-24。本页将论文、官方文档和源码与当前项目对照。它记录研究判断，尚未进行 GPU 验证，也未在本轮改变训练代码或实验配置。原始计划中的估算继续保留为历史规格，不能据此声称指标定义已核验。

## 优先修订的口径

| 优先级 | 问题 | 研究依据 | 建议落点 |
| --- | --- | --- | --- |
| P0 | 当前 `bench.mfu()` 对 GC 开/关分别使用 6N/4N，混合了任务运算与实现重算成本。 | [PaLM §4.1 / 附录 B](https://arxiv.org/pdf/2204.02311) 区分 MFU 与计入 rematerialization 的 analytical HFU。 | 分开记录不含重算的冻结训练运算估算、含重算的运算估算，以及各自假设；使用新的字段和 schema，重新生成派生结果。 |
| P0 | 把 60% 当成物理上限，会误判高效实现。 | [FA2 原论文](https://tridao.me/publications/flash2/flash2.pdf) 的内核与端到端训练数值来自不同对象，部分利用率超过该门槛。 | 把超线定义为需核对工作量、分母与计时的事件；确认合理后允许附解释，保留原始值。 |
| P0 | `sdpa` 不是“非 Flash”的同义词。 | [PyTorch 2.2 发布说明](https://docs.pytorch.org/blog/pytorch2-2/)、[SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。 | E2 名称写明集成路径；另设受控 SDPA backend 诊断，保存实际 kernel 证据。 |
| P1 | 当前仅 q/k/v/o，不能代表论文全部默认训练配置。 | [PEFT QLoRA-style](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)、[作者训练代码](https://raw.githubusercontent.com/artidoro/qlora/main/qlora.py)。 | 说明是四投影 NF4 QLoRA 配置；若增加 all-linear，另立容量与质量实验，并重算参数量。 |
| P1 | Qwen1.5-14B 的结构不能从其他 Qwen 型号泛化。 | [本模型 config](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json)。 | 保留 40 KV heads 的 MHA 断言；不要使用 GQA 的 KV 显存估算替代。 |
| P1 | 不同版本的模型 loss 与 Trainer 交互可能改变梯度累积的归一化。 | [v4.37.0 Qwen2](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py)、[较新 loss_utils](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/loss/loss_utils.py)。 | 在目标版本核对有效监督 token 计数与 loss kwargs；把变长样本的单大 batch / 多 microbatch 等价性作为验收，不能只看“支持 Qwen2”。 |

PaLM 讨论的是全参数 dense Transformer 的运算模型。本项目冻结大量基座参数，即使排除了 GC 重算，4N 也只是自己的近似口径，不能不加说明地与标准全参 6N 的 MFU 横比。长序列、attention 内部重算、量化算子与实际 autograd 路径需要进一步核算。这是根据论文定义和本地工作负载作出的**项目推断**。[PaLM](https://arxiv.org/pdf/2204.02311)

## 后续增加的观测证据

| 观测对象 | 应保存的信息 | 为何需要 |
| --- | --- | --- |
| 量化层 | quant_type、blocksize、nested、原始/packed shape、quant state 张量 | 参数 tensor_bytes 不包含全部量化元数据。[bnb functional](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/functional.py) |
| 模型准备 | helper 后、精度恢复后、LoRA 后、首步后占用；加载阶段峰值 | helper 的上转与恢复会形成瞬时峰值，稳态占用不等于加载需求。[PEFT helper](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/utils/other.py) |
| 注意力执行 | attention 类、Q/K/V dtype、attention dropout、实际 kernel trace | 请求字符串和 dtype 配置不能单独证明真实内核路径。[Qwen2 示例源码](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py) |
| 4-bit matmul | 版本、M/N/K、分派条件与 kernel | 入口 `gemm_4bit` 仍可能回退；短 token 推理与长序列训练形状不同。[CUDA 分派源码](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/backends/cuda/ops.py) |
| 优化器 | state dtype、managed 分配与实际 UVM 迁移分别记录 | paged 类名和内存波动不足以证明页迁移。[bnb optimizer](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/optim/optimizer.py) |
| 全词表 loss | logits 与 loss 的临时张量、算子和生命周期 | attention 省下显存后，词表头仍可能形成峰值。[CCE 论文](https://arxiv.org/abs/2411.09009) |

Profiler 采样会影响时序、张量引用及内存。用于确认后端的诊断 trace 应与正式吞吐/峰值测量分别运行和解释。[PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html)

## 保留与补充的实验问题

- **E1：** 同目标层、同 adapter dtype 的 bf16 LoRA / NF4 对照有意义；主要回答权重压缩的成本和收益，不预设 QLoRA 更快。
- **E2：** 当前矩阵比较三条集成路径与 GC；强制 SDPA backend 的组属于新增诊断，不能暗中替换原组。
- **E3：** 保留每步 token、batch、长度；在扫描边界内讨论成功长度，不把所有开销简化为 attention 的 O(N²) 存储。FA2 的 dense 运算仍为二次。[FA2](https://tridao.me/publications/flash2/flash2.pdf)
- **E4：** “换页未知”比无证据地写“没有发生换页”准确；若 profiler 未采到迁移，仍应说明采样范围和工具能力。
- **E5：** 报告实际输入 token、受监督 token、截断比例与 loss 归一化；loss mask 不等于 attention mask。
- **E6：** 同 NF4 基座前后比较测 adapter 效果；量化质量损失需另加 bf16 基座。合并/requantize 的产物应独立评测。[PEFT bnb merge 实现](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/bnb.py)

以上是研究后的项目设计建议；没有把文献中的显存、吞吐或质量数字录入 results/raw。
