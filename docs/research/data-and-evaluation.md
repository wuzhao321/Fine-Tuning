# 数据与评测补充资料

资料核对日期：2026-10-24。本页提供 E5/E6 的候选数据集与评测协议资料。选用数据时，需分别记录论文、代码仓库和数据包的版本、来源与使用条件。

| 编号 | 一手来源 | 资料覆盖范围 | 对项目的影响 |
| --- | --- | --- | --- |
| D01 | [DuReader 原论文](https://arxiv.org/abs/1711.05073) | 原作者摘要：中文真实问题/文档、人工答案、多种问题类型。 | 生成式问答不一定适合只用严格 EM；须区分问答子集与数据版本。 |
| D02 | [DuReader 官方仓库](https://github.com/baidu/DuReader) | 系列任务目录与官方说明。 | 选定哪个系列后再写字段映射，不能把多个版本混为同一语料。 |
| D03 | [DuReader 2.0 官方说明](https://github.com/baidu/DuReader/blob/master/DuReader-2.0/README.md?plain=1) | 官方采用 BLEU/ROUGE 的评测说明。 | 本项目字符 F1/EM 是自定义简化口径，不能冒充该数据集的官方分数。 |
| D04 | [CSDS 论文](https://aclanthology.org/2021.emnlp-main.365/) | 原作者摘要：整体、角色视角与主题结构的客服摘要。 | 需先确定使用 overall、user 或 agent summary；它们并非可随意混用的参考答案。 |
| D05 | [CSDS 官方仓库](https://github.com/xiaolinAndy/CSDS) | README 说明其字符到索引的 ROUGE 处理，以及句子切分对 ROUGE-L 的影响。 | jieba 分词的项目指标与作者脚本不可直接横比；若要复现论文成绩，须保留官方协议。 |
| D06 | [LCSTS 论文](https://arxiv.org/abs/1506.05865) | 原作者摘要：来自微博的中文短文本及摘要。 | 仅能作为另一种摘要任务的替代，不能写成已复现客服工单数据效果。 |
| D07 | [ROUGE 原论文](https://aclanthology.org/W04-1013/) | ACL 元信息与原论文入口。 | 研究 n-gram/LCS 指标原始定义；阅读范围为元信息与原论文入口。 |
| D08 | [Google rouge-score 实现](https://github.com/google-research/google-research/tree/master/rouge) | 官方包目录和实现入口。 | 核实 tokenizer、rougeL/rougeLsum、P/R/F 与多参考汇总策略，避免只写“ROUGE”而不写口径。 |
| D09 | [jieba 官方仓库](https://github.com/fxsjy/jieba) | README 中的分词模式、词典与用户词典选项。 | 冻结版本、分词模式与词典；同一字符串采用不同分词会改变 n-gram。 |
| D10 | [SQuAD 官方站点](https://rajpurkar.github.io/SQuAD-explorer/) | QA 任务、无答案问题与官方评测脚本入口。 | 用于理解 EM/F1 与答案规范化；不是为本中文项目选择英文训练集。 |

建议 E6 报告额外列出“预测到达生成上限比例”“prompt 截断比例”“空答案比例”和少量原文/预测/参考对照。它们是本项目的分析建议，不是上述数据集论文的统一规定。字符 F1、中文分词 ROUGE 与任务事实正确性分别反映不同方面，不能用单个重叠指标代表全部质量。
