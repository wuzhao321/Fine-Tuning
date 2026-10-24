# 日志、审查与数字溯源

## 原始日志

每个尝试为 `results/raw/<run_id>.r<repeat>.<attempt_id>.jsonl`。每行是完整 JSON 对象，UTF-8，只追加。budget.jsonl 是独立的资源账本，不算测量结果。

| 字段 | 说明 |
| --- | --- |
| schema_version / event | 结构版本与事件类型 |
| run_id / experiment / attempt_id / repeat | 配置标识、实验号、唯一尝试、重复序号 |
| job_id | 矩阵子进程身份；直接调用运行器时为 null |
| timestamp | UTC 时间 |
| config / config_sha256 | 完整合并配置和规范化 SHA256 |
| provenance | git commit、dirty、源码指纹、Python/依赖版本 |
| gpu | 名称、实际显存、UUID、CUDA runtime、所声明峰值 |
| model_ready | 实际可训练参数、GC、attention、显存、精度分组 |
| data / training_inputs / evaluation_inputs | 数据哈希、来源、评测解码条件或检查点身份 |
| telemetry | 整卡采样、采样失败、驱动、无法确定的换页状态 |
| result | ok / oom / failed；步时、吞吐、MFU、loss 或任务指标 |
| prediction | 每条测试样本的 ID、输出、参考答案和分数 |

早于 GPU 初始化的 started/failed 行允许 gpu=null。没有 result 的尝试汇总为 interrupted。损坏 JSON 行会使汇总显式失败，不能静默跳过。开始前不要在 description、路径或配置里放密钥；常见 HF token 和异常文本会做脱敏，但它不是通用 DLP 系统。

参数 inventory 中的 tensor_bytes 是实际存储张量字节，并不包含全部量化 metadata 或 CUDA 工作区。显存最终以 allocator 实测为准。

## 派生结果

- `summary.json`：所有分组和检查状态，保留 config、原始引用与审查记录。
- `audit.csv`：包括失败和待审查；不允许直接用于简历。
- `accepted.csv`：通过校验与审查的子集。
- `traceability.csv`：每个数值键 → 聚合值 → 原始文件与行号 → 文件集合摘要。
- `summary.xlsx`：可选审计工作簿；依赖 openpyxl，未安装时不导出。
- `report.md`、`resume_snippet.md`：只引用 accepted；无结果时写【未测】。

benchmark 汇总自动检查成功状态、三个不同重复、实际参数量、NVML 测量样本、MFU、重复吞吐差和提交号。环境异质性用分组指纹隔离。为避免误用，其他角色也需要审查记录后才可进入报告。

现在还会从完整逐步耗时复算中位数、IQR、吞吐与 MFU，核对 warmup/测量损失数量、输入/预测 token 数以及 allocator 数值关系。存在统计不一致的成功行仍然被阻止引用。E6 汇总检查逐样本 ID 唯一、两任务实际条数与完整数据包一致、分数范围与宏平均是否匹配；ROUGE 的文本到分数计算仍需在有依赖的环境中验证。

读取 JSONL 时保留物理行号（包括空行），检查同文件内的配置/尝试身份、配置哈希和终止事件唯一性；文件在汇总读取期间发生变化会报错。静态显存的溯源行指向 model_ready，而非没有该字段的 result 行。不同完整 provenance、训练身份或 adapter 完整性记录不会合并为同一分组。E3 图表也按比较环境分开连线。

## 审查文件

先生成 summary.json，再用 `review-template --summary results/summary/summary.json --output 新文件路径` 生成独立审查草稿，自动填写 group_key 和 evidence_digest。`data/review-template.json` 保留为 benchmark 格式示例，默认不能通过。每项需填 `status: passed` 和非空 evidence，说明具体来源、比较与判断。

| 角色 | 必需审查项 |
| --- | --- |
| benchmark | 静态显存、数值等价、GC 开销、硬件、来源 |
| inspect | 模型结构、硬件、来源 |
| train | 数据完整性、检查点完整性、收敛、硬件、来源 |
| eval | 数据完整性、adapter 身份、指标、硬件、来源 |
| equivalence | 数值比较、硬件、来源；始终不能作为性能结果发表 |

不适用的项需写明理由，例如基座评测没有 adapter；不要为 E6 机械填写 GC 性能审查。即使审查文件写 approve，自动检查失败的结果仍然不会通过。

静态显存偏差、数值比较、GC 开销、GPU 时钟/功耗和 provenance 尚不能完全由代码自动判断；记录由审查人完成。本实现校验审查字段与证据摘要是否匹配，但不自动理解文字证据是否充分。

文件新增或变更使 evidence_digest 改变，旧审查立即失效。不要编辑原始日志纠错，保留失败并写新的说明或新命名配置。重跑超过三个重复时不会自动择优；使用明确的实验轮次和独立目录，报告中解释所有历史尝试。

## 简历规则

只允许引用 accepted 的字段；每个性能数字旁有 `<group_key>.<field>`。自动模板仅在唯一的 E1 QLoRA 合格分组存在时回填峰值、吞吐和 MFU，避免在多套硬件或多次轮次中偷偷选择。其余比较结论需要结合 E1/E2/E3/E5/E6 单独撰写。

必须说明复现实验、硬件、日期、公开替代数据。无实测或未审查均输出【未测】。不把新复现结果写成过去实习的同期数字。
