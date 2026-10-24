# 数据协议与来源登记

**目前没有选定或下载任何候选数据集。** 原计划的 DuReader / BELLE / Firefly / CSDS / LCSTS 仅为候选，许可证与当前可用性未在本轮核验。后续选定前需要核实来源、版本、许可及用途，再填写 sources 清单；不要把新闻摘要称作客服工单摘要。

## 标准输入

训练准备入口接受 UTF-8 JSONL，每行：

```json
{"task":"qa","source_id":"approved_source","context":"资料正文","question":"问题","answers":["参考答案","可接受的另一答案"]}
{"task":"summary","source_id":"approved_source","context":"对话或新闻正文","answers":["参考摘要"]}
```

`task` 只能是 qa 或 summary，context 必须非空，QA 需要 question，answers 必须是非空字符串列表。训练使用第一条答案，评测取多参考答案中的最佳匹配。此例仅说明格式，禁止把示例作为实验样本或用重复样本凑数量。

来源模板见 data/sources.template.json。每项都需真实 `id / url / license / license_url / revision / approved_by / approved_at / approved`；模板默认 approved=false，因此不会误触发数据处理。代码检查记录是否完整，不能代替阅读许可证与判断权限。下载与特定数据集字段映射在选定来源后再补，当前支持已经正规化的本地文件。

## 清洗与划分

先做 NFKC 和空白归一化用于去重，不改变实际训练文本。按 task + context + question 识别重复；同样输入不同答案仅保留第一条，去除数量写入 manifest。所有相同 context 共用 group_id，即使来源或问题不同也不会跨集合。

随机种子 42，测试每任务 500 条、验证每任务 100 条，训练上限 20,000 条。实际条数不足不会虚构补齐；训练量可能少于上限，manifest 中真实 token 数决定最终结论。对分组多余样本进行丢弃以满足固定配额，不移入训练集。

默认使用 tokenizer 的 ChatML 模板，以中文 system/user 提示词表达“根据资料回答”或“概括文本”。提示词定义在 data.prompt_messages 与 SYSTEM；内容和预处理策略改变后必须重新生成数据并保留新的 manifest。

默认最长 2048 token，回答预算 512，prompt 预算 1536。保存原始长度、prompt/target 截断标记；按任务、split 输出 count、min、p50、p95、max、截断比例、输入 token 数、监督 token 数。截断统计不是 padding 统计。

输出 train.jsonl / validation.jsonl / test.jsonl 与 manifest.json。manifest 保存来源、输入文件 SHA256、模型配置、模板 SHA256、输出文件 SHA256 和统计。训练及评测前校验对应文件哈希。

第二轮起，任务 manifest 使用 schema_version=2，并保存 split_policy 中的测试/验证配额及训练上限。训练、评测和 preflight 会读取三个划分，重新计算样本 ID 与 context 分组 ID，拒绝跨集合重叠、重复样本、标签污染、错误 token 范围或长度、与 prompt_ids 不一致的输入，以及声明配额与实际条数不符。仅更新文件哈希不能绕过内容校验。缺少 split_policy 的早期数据需重新准备。

微调后评测还核对 adapter 的完整性记录、训练配置中的模型/量化/LoRA 设置、三个数据文件和 manifest 身份。训练时同样校验实际 tokenizer 模板哈希；不能只让评测阶段检查模板。

## 基准数据

pack-data 从本地 UTF-8 中文文本逐行 tokenize，用 EOS 连接，保存 little-endian uint32 token 流。所有基准从同一流的开头取固定数量 token，无 padding；数据不够时报错，不重复循环语料。这里的 synthetic_packed 指固定形状的基准输入，并非随机生成 token。

每个正式配置消耗 40 × batch × seq token。默认最低 327,680 token，可覆盖 E3 的 8192 长度；较小配置使用同一流的前缀。benchmark 只算全序列语言模型 loss，与 E5 的回答区间 SFT loss 不可直接比较绝对值。

## 指标

QA：NFKC、去标点与空白、转小写；EM 是规范化字符串相等，F1 按字符 multiset 计算，多参考取最大值。不声称与某个官方数据集评测脚本等价。

摘要：jieba 分词，去除纯空白 token，使用 rouge-score 自定义 tokenizer，报告 ROUGE-1/2/L 的 F-measure，关闭英文 stemming。多参考逐指标取最大，按样本宏平均，范围 0–1。jieba 与 rouge-score 版本写入日志。

E6 贪心解码、num_beams=1、最多 512 新 token；EOS 包含 tokenizer.eos_token_id 与可识别的 `<|im_end|>`。生成截断可能影响长摘要，应在报告披露。不加入未登记的 LLM 评审。
