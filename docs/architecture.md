# 实现设计

## 模型入口

`model_factory.build_model` 被 E0、基准、训练和评测共同使用。结构断言来自官方 [config.json](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json)：40 层、hidden 5120、40 个注意力与 KV 头、intermediate 13696、词表 152064、embedding 与 lm_head 不共享。进一步检查 160 个 q/k/v/o 投影的输入输出都是 5120。

LoRA 设计值为 `40 × 4 × 64 × (5120 + 5120) = 104,857,600`。该值为结构推导值，运行时通过实际参数计数校验。训练构建时必须用 `requires_grad` 参数实际求和断言；不启动全参训练，E1 全参仅保留理论讨论。

所有 GPU 代码延迟导入。配置校验、计划查看、CSV 与 Markdown 汇总可脱离 GPU 环境运行。`device_map={"": 0}` 固定一个设备，拒绝多个可见 GPU 和 `WORLD_SIZE != 1`，不使用自动 offload。

## 精度与 GC

量化采用 NF4、Double Quantization、bf16 compute dtype，lm_head 显式排除，embedding 保持非量化，并检查实际模块类型。PEFT 准备函数显式传 `use_gradient_checkpointing=False`，记录其签名、源文件内容哈希及前后参数精度。E0 额外保存该函数源码。

当前 [PEFT 实现](https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py) 会把非量化的低精度参数转为 fp32。项目在调用后把两个大型冻结表恢复到 bf16，norm 和可训练 LoRA 参数保留 fp32；bf16 LoRA 路径也采用 fp32 norm 与 adapter。此策略需要在目标依赖版本下核实，加载峰值可能高于准备后的稳态值。

随后统一设置实际 GC，默认 `use_reentrant=False`；reentrant 配置显式启用输入梯度。训练时 `use_cache=False`。基准在第一次 backward 检查全部 adapter 梯度连接及有限性。训练使用 autocast，Trainer 不重新配置 GC。

评测加载相同基座配置和已训练 adapter；基座与微调后默认都以 NF4 推理，使对比集中在微调效果。E6 结论不能称为“bf16 原始基座对比”或把 NF4 推理结果混入不同精度的表。

## 数据与损失

标准输入只接受公开、经过来源登记的数据。按规范化后的 context 分组，训练/验证/测试互斥；同 context、同 question、同任务视为重复，即使答案文字有差异也保留第一条。近重复或语义相似文本尚不自动识别。

固定测试集每任务 500 条、验证集每任务 100 条，其余最多 20,000 条用于训练。不能满足配额时失败，不用重复样本凑数。分组多余样本丢弃而不移到另一个集合，清楚记录数量。

使用模型自带 `apply_chat_template`，校验生成提示词与完整对话的 token 前缀一致。prompt labels 为 -100，回答才参与 loss，batch padding 同样为 -100。默认给回答预留 512 token，prompt 从左侧截断保留末尾问题，答案右截断；会损失开头 system/context，必须在报告披露。原始完整答案仍用于 E6 评分。

E5 的验证集只用于验证 loss；E6 才使用 test。测试 prompt_ids 在数据准备时固定，基座和 adapter 完全复用。

## 测量与预算

正式基准使用梯度累积 1，计时包含 zero_grad、forward、backward、clip 和 optimizer.step，排除数据加载与 CPU→GPU 拷贝。所有固定 batch 预装载到 GPU；这部分小额 buffer 会包含在峰值显存，日志明确保存三处静态检查点。

每个配置与重复启动新进程。矩阵执行器记录每个子进程的实际墙钟时间（包含加载，保守估计 GPU 占用时间），默认累计 15 小时触发预算确认阈值，20 小时为硬上限。它不等价于云平台账单；单独调用 bench/train/eval 不走矩阵预算器。机器掉电造成未闭合账本时拒绝继续，需追加修正记录。

## 证据与审查

记录配置全文及哈希、源码指纹、git 提交与 dirty 状态、依赖版本、实际 GPU、UTC 时间、repeat 和 attempt_id。每次尝试独立日志，写入 flush + fsync；OOM 是终止结果，不会被成功重跑覆盖。汇总按配置、源码、版本、GPU、数据指纹分组，避免把不同环境的重复合并。

`summary.json` 和 `audit.csv` 可以含待审核数字，`accepted.csv` 只含自动校验与绑定证据哈希的审查都通过的结果。报告正文、性能图表与结果摘要只从 accepted 集合生成。
