# Qwen1.5-14B 单卡 QLoRA + FlashAttention-2 微调复现项目

项目计划书与 AI Agent 执行 Prompt · v1.0 · 2026-10-23

## 0. 文档用法

- 第一部分是项目计划书，供你阅读和决策：目标、假设、技术方案、实验设计、风险与排期。
- 第二部分是 Agent Prompt，自包含，可整段复制给 AI Agent 执行，不依赖第一部分。
- 文中所有显存、吞吐数字都是按模型结构推算的估算区间，只用来检验实测值是否合理，不能当作实测值写进简历。

## 第一部分　项目计划书

### 1. 项目目标与成功标准

**背景。** 原简历描述中的模型结构、可训练参数量、吞吐和显存数字互相矛盾，缺少可追溯的原始数据。本项目在单张 A100 80GB 上重新做一遍 Qwen1.5-14B 的 QLoRA 与 FlashAttention-2 实验，目的是得到一条可核查的证据链：配置、原始日志、汇总表、报告、简历措辞。

**目标。** 在单卡约束下完成三件事：（1）用实测数据比较全参、bf16 LoRA、QLoRA 三条路径；（2）对 attention 实现（eager、sdpa、FA2）与 gradient checkpointing（GC）做完整消融，量化各自对显存和吞吐的贡献；（3）用小规模的问答与摘要任务验证微调链路确实有效。

**成功标准**（全部可检验）：

1. 可训练参数量打印值精确等于 104,857,600（r=64，q/k/v/o，40 层）。
2. attention × GC 的矩阵（含 sdpa）全部有实测显存和吞吐；eager 无 GC 的 OOM 边界被记录，而不是省略。
3. bf16 LoRA 与 QLoRA 有同配置对照：静态显存、峰值显存、吞吐、收敛曲线。
4. 每个吞吐结果都折算成 MFU 并检查；MFU 超过 60% 的结果必须先排查测量方法，不得直接采用。
5. 简历里出现的每个数字，都能在 results/raw 下找到对应的原始日志行。
6. 两个下游任务各有评测指标，包含基座与微调后的对比。

### 2. 范围

**做：** 单卡训练与基准测试；公开数据集替代业务数据；显存与吞吐的消融；小规模下游评测；报告与简历措辞。

**不做：** 多卡与分布式；DPO 等对齐训练；模型部署与推理优化；任何内部业务数据的使用。

### 3. 前置条件与假设

| 项目 | 要求 | 不满足时的处理 |
| --- | --- | --- |
| GPU | 1×A100 80GB（SXM 或 PCIe 均可，需记录型号） | 其他 80GB 卡可跑，但 MFU 峰值要换成该卡的标称值，结论只能写成「在该卡上」；40GB 卡无法跑 bf16 LoRA 对照，需删除 E1 中该项 |
| 系统 | Linux，驱动支持所选 CUDA 镜像 | Agent 在 Phase 0 检查驱动版本并选择兼容的镜像 |
| 磁盘 | 至少 150GB 可用（模型约 28GB、镜像约 20GB、缓存与检查点） | 不足则停止并报告 |
| 内存 | 至少 64GB | 低于此值时 Paged Optimizer 与数据加载可能受影响，需记录 |
| 网络 | 可访问 Hugging Face 与 PyPI（可用镜像站） | 无法联网则停止并报告 |
| 预算 | 约 20 个 A100 小时，含调试余量 | 累计超过 15 小时时 Agent 必须先汇报 |
| 数据 | 公开数据集，许可证允许研究使用 | Agent 联网核实后再选定，不使用来源不明的数据 |

### 4. 理论基线（推导过程）

这一节的作用是：先用结构算出「应该是多少」，实测值偏离太多就说明配置或测量有问题。

#### 4.1 模型结构

模型是 Qwen1.5-14B，结构取自其 config.json：40 层，hidden=5120，40 个注意力头，KV 头也是 40 个（标准 MHA，不是 GQA），head\_dim=128，MLP 中间维度 13696，词表 152064，embedding 与 lm\_head 不共享。按此推算总参数约 14.17B，其中 embedding 和 lm\_head 各约 0.78B，参与矩阵乘的参数 N\_eff 约 13.39B。

#### 4.2 LoRA 参数量

r=64，作用在 q/k/v/o 四个投影，每个投影都是 5120×5120，所以每个投影的 LoRA 参数为 64×(5120+5120)=655,360，每层 4 个共 2,621,440，40 层共 104,857,600，约占总参数的 0.74%。alpha=128 对应缩放系数 2。

#### 4.3 显存拆解（估算）

| 组成 | bf16 全参 | bf16 LoRA | QLoRA |
| --- | --- | --- | --- |
| 基座权重 | 28.3GB | 28.3GB | 约 9.6GB（线性层 4-bit 约 6.5GB，embedding 与 lm\_head 保持 bf16 约 3.1GB） |
| 梯度与优化器 | 约 198GB（按每参数共 16 字节算） | 约 1GB | 约 1GB |
| 合计（不含激活） | 约 227GB | 约 29–30GB | 约 10–11GB |

补充说明：

- LoRA 参数按 fp32 存放约 0.42GB，梯度约 0.42GB，8-bit Adam 的两个状态约 0.21GB，所以优化器状态本身很小，Paged Optimizer 在这个规模下基本不会触发换页。
- 已知陷阱：peft 的 prepare\_model\_for\_kbit\_training 默认会开启 GC，并把 bf16 参数上转为 fp32，这会让 embedding 与 lm\_head 多占约 3GB。Agent 必须在所装版本里核实这一行为，并以加载后的实测 memory\_allocated 为准。
- eager 注意力的分数矩阵：batch=2、seq=2048、40 头时，单层单份 bf16 约 0.67GB。HF 的 eager 实现用 fp32 做 softmax，反向要保存多份，所以 40 层合计下限约 54GB，实际更高。这就是 eager 不开 GC 会 OOM 的原因。
- logits 的 fp32 副本约 2.5GB（4096 token × 152064 词表 × 4 字节），算上梯度，损失计算处会形成约 5GB 的峰值。这是 FA2+GC 下激活显存的主要部分。

#### 4.4 FLOPs 与 MFU 公式

冻结权重时无需计算权重梯度，所以每 token 约 4×N\_eff FLOPs（约 53.6 GFLOPs）；开 GC 多一次前向重算，约 6×N\_eff（约 80.3 GFLOPs）。注意力和 LoRA 的 FLOPs 合计不到 5%，忽略，并在报告中注明。

MFU = 每秒 token 数 × 每 token FLOPs ÷ 312 TFLOPS（A100 的 bf16 稠密峰值）。100% 利用率下的理论上限：无 GC 约 5,800 tokens/s，开 GC 约 3,900 tokens/s。QLoRA 的实际利用率通常落在 30%–50%。

#### 4.5 预期区间（仅用于校验实测）

| 配置（bs=2，seq=2048） | 激活与 logits | 总峰值 | 吞吐 |
| --- | --- | --- | --- |
| eager，无 GC | 85GB 以上 | OOM | 无 |
| eager + GC | 约 8–10GB | 约 20–23GB | 约 0.9–1.7k tokens/s |
| FA2，无 GC | 约 30–34GB | 约 42–46GB | 约 1.7–2.9k tokens/s |
| FA2 + GC | 约 6–9GB | 约 17–21GB | 约 1.2–1.9k tokens/s |

这张表的含义：在 seq=2048 时，GC 是显存节省的主力，eager+GC 与 FA2+GC 只差 2–3GB；FA2 的显存优势在不开 GC 或序列变长时才明显。因此 E3 的序列长度扫描，是这个项目里最能支撑「FA2 价值」的实验。

### 5. 技术方案

#### 5.1 加载与量化

- 计算精度用 bf16，bnb\_4bit\_compute\_dtype 设为 bf16。
- 量化类型 NF4，开启 Double Quantization，量化块大小取默认值。
- lm\_head 与 embedding 不量化。
- 加载后立即记录 memory\_allocated，作为静态显存基线。

#### 5.2 LoRA

target\_modules 为 q\_proj、k\_proj、v\_proj、o\_proj；r=64；alpha=128；dropout=0.05；bias 不训练；任务类型 CAUSAL\_LM。MLP 不加 adapter（与原方案一致）。

#### 5.3 优化与超参（正式训练用）

| 项 | 取值 |
| --- | --- |
| 学习率 | 2e-4，cosine 衰减，3% warmup |
| 有效 batch | 16 条序列（micro-batch 2 × 梯度累积 8） |
| 训练量 | 问答与摘要混合约 2 万条样本，1 个 epoch，约 1,000 万 token |
| 最大长度 | 2048，超长样本截断并记录比例 |
| 优化器 | paged\_adamw\_8bit（对照：adamw\_8bit，仅比较步耗时与峰值） |
| 权重衰减 / 梯度裁剪 | 0.0 / 1.0 |
| 随机种子 | 固定为 42，并在报告中说明 |

#### 5.4 注意力实现与 GC

- 三种 attn\_implementation：eager、sdpa、flash\_attention\_2。sdpa 是新版 transformers 的默认实现，底层也能走 FlashAttention 内核，所以必须作为对照，否则「以 eager 作基线」会被追问。
- GC 使用 use\_reentrant=False；若使用 reentrant 版本，则需要 enable\_input\_require\_grads，否则 LoRA 梯度会断。
- 做「无 GC」配置时，必须显式关闭 prepare\_model\_for\_kbit\_training 的默认 GC，并在日志里打印 GC 的实际状态。

#### 5.5 其他固定项

训练时 use\_cache=False；同一张实验表里所有配置共用相同的环境变量（尤其是 PYTORCH\_CUDA\_ALLOC\_CONF），不得在配置间改动；每个配置都用独立进程运行，避免上一个配置的显存碎片或 OOM 污染下一个。

### 6. 数据方案

**基准数据（用于 E1–E4 的显存和吞吐测量）。** 不使用业务语义，使用固定形状的批次：从任一中文语料中取文本，拼接并切成恰好 2048（或扫描所需长度）token 的序列，无 padding，所有配置喂同一批数据。这样吞吐只反映算力，不被 padding 比例干扰。

**训练与评测数据（E5–E6）。** 内部业务数据无法获取，使用公开数据集替代，并在所有产出里写明「替代数据」。

| 任务 | 候选（Agent 须联网核实可用性与许可证后选定） | 评测指标 |
| --- | --- | --- |
| 知识问答 | 中文文档问答或中文指令数据中的问答子集（如 DuReader 系列，或从 BELLE、Firefly 类指令集中筛选） | 对留出集做 F1 或 EM；必要时加 LLM 评审，评审模型与提示词需写入报告 |
| 工单摘要 | 客服对话摘要数据（如 CSDS）；LCSTS 是新闻摘要，只作备选 | ROUGE-1/2/L（中文需先分词，并记录分词器） |

数据处理要求：统一转为 Qwen1.5 的对话模板；训练集与评测集按样本去重且互不重叠；每个任务留出 500 条评测；记录各数据集的 token 长度分布与截断比例；保存数据集版本号或哈希。

### 7. 实验设计

每个实验都有编号，Agent 的所有产出必须引用这些编号。

| 编号 | 内容 | 变量 | 固定项 | 产出 |
| --- | --- | --- | --- | --- |
| E0 | 环境与结构自检 | 无 | 无 | GPU 型号、驱动、版本清单、参数量断言 |
| E1 | 三条路径对照 | 全参（仅理论）、bf16 LoRA、QLoRA | bs=2、seq=2048、FA2+GC | 静态显存、峰值显存、吞吐；收敛曲线放在 E5 |
| E2 | attention × GC 消融 | attn ∈ {eager, sdpa, FA2}，GC ∈ {开, 关} | QLoRA、bs=2、seq=2048 | 6 个配置的静态与峰值显存、吞吐、MFU；eager 无 GC 若 OOM，则降到 bs=1、seq=1024 找到可运行点，并作为独立配置记录 |
| E3 | 序列长度扫描 | seq ∈ {512, 1024, 2048, 4096, 8192} | QLoRA、GC 开、每步 token 数尽量恒定（bs 随 seq 反比调整，下限为 1） | eager、sdpa、FA2 的显存与吞吐曲线，以及各自的最大可训练 seq |
| E4 | 优化器对照 | paged\_adamw\_8bit、adamw\_8bit | QLoRA、FA2+GC | 步耗时与峰值，并检查是否发生换页 |
| E5 | 收敛对照 | bf16 LoRA、QLoRA | 相同数据、超参、种子 | 训练损失曲线、留出集损失、总耗时 |
| E6 | 下游评测 | 基座、QLoRA 微调后 | 两个任务的留出集 | 评测指标与若干样例输出 |

E1 的 bf16 LoRA 峰值约 38GB，80GB 卡上可行，这条对照回答「为什么选 QLoRA」。如果数据显示 bf16 LoRA 更快且显存够用，就如实写：QLoRA 的价值是把峰值降到约 20GB 量级，使 24GB 级显卡也能复现，而不是在 80GB 卡上更快。E1 的 QLoRA 峰值要用 FA2+GC 配置测得。

### 8. 测量协议与一致性校验

**测量协议（所有基准实验共用）：**

1. 每个配置独立进程，先跑 10 步 warmup，再测 30 步。
2. 步耗时用 CUDA 同步后的 wall time；报告中位数与四分位距，不用平均值。
3. 吞吐 = micro-batch × seq ÷ 步耗时（基准实验梯度累积为 1）。
4. 显存同时记录 max\_memory\_allocated 与 max\_memory\_reserved，并在运行中用 NVML 采样整卡占用；报告以 allocated 为主，reserved 与整卡占用为辅，并写明差异原因。
5. 另记录三个时间点的静态显存：模型加载后、套 LoRA 后、第一次前反向后。
6. 同时采样 GPU 时钟、功耗与温度，检测降频。
7. 每个配置重复 3 次（3 个新进程），报告中位数与范围。
8. 所有结果以一行 JSON 追加到原始日志，含配置、版本、时间戳、git 提交号。

**一致性校验（任一失败都不得进入报告，须先排查）：**

| 校验项 | 通过条件 |
| --- | --- |
| 参数量 | 可训练参数等于 104,857,600 |
| 静态显存 | 实测与 4.3 的估算相差不超过 15%；超出则要解释（如 fp32 上转） |
| MFU | 不超过 60%；超过说明测量方法有问题 |
| 数值等价 | eager 与 FA2 从同一初始化出发，前 20 步损失差在 bf16 噪声范围内 |
| GC 开销 | 步耗时增幅与理论的 +33% 到 +50% 比较；偏小要给出解释，解释不了就不写百分比 |
| 重复性 | 3 次重复的吞吐极差不超过中位数的 10% |

### 9. 风险与对策

| 风险 | 对策 |
| --- | --- |
| flash-attn 没有匹配的预编译包，源码编译很慢或失败 | 优先选用与 torch、CUDA、Python 版本都匹配的预编译包；否则在带 CUDA 开发工具的镜像里编译，并限制并行编译数以免内存耗尽 |
| bitsandbytes 与 CUDA 版本不匹配，4-bit 内核加载失败 | Phase 1 用最小脚本验证 4-bit 线性层可以前反向，再进入正式实验 |
| 模型下载慢或中断 | 使用断点续传与镜像站，下载后校验文件大小与哈希 |
| OOM 污染后续实验 | 每个配置独立进程；OOM 记为结果并保存报错栈，不重试同一配置 |
| 租用机中断 | 训练使用定期检查点和可恢复脚本；基准实验天然幂等，可按编号重跑 |
| 硬件差异（SXM 与 PCIe 的功耗墙不同） | 记录型号与功耗，MFU 的分母只用标称峰值 |
| 数据许可证或可用性问题 | Phase 2 先核实，不明确的数据集不用 |
| 实测与估算差距大 | 先查 prepare\_model\_for\_kbit\_training 的上转、PYTORCH\_CUDA\_ALLOC\_CONF，以及是否把 padding 计入 token |

### 10. 里程碑与资源预算

| 阶段 | 内容 | 预计时间 | 验收门 |
| --- | --- | --- | --- |
| Phase 0 | 硬件与软件自检、安装 | 第 1 天 | E0 通过 |
| Phase 1 | 模型下载、4-bit 加载、静态显存、参数量断言 | 第 2 天 | 参数量等于 104,857,600 |
| Phase 2 | 数据准备与基准数据构造 | 第 3 天 | 数据统计表完成 |
| Phase 3 | 基准测试框架与 E1、E2 | 第 4–5 天 | 一致性校验全过 |
| Phase 4 | E3、E4 | 第 6 天 | 曲线齐全 |
| Phase 5 | E5 收敛对照 | 第 7–8 天 | 两条训练完成 |
| Phase 6 | E6 评测 | 第 9 天 | 指标与样例齐全 |
| Phase 7 | 汇总、报告、简历措辞 | 第 10 天 | 所有简历数字可追溯 |

GPU 时间粗估：基准类实验合计约 3–4 小时；两次训练（bf16 LoRA 与 QLoRA）各约 1,000 万 token，在 1.2–2k tokens/s 下合计约 3–5 小时；评测约 1 小时；其余为调试余量，总预算约 20 小时。费用按所租平台价格自行折算。

### 11. 交付物与成果使用声明

交付物：代码仓库与锁定的环境清单；results/raw 原始日志；results/summary 汇总表（CSV 与 Excel）；figures 图表；report.md 实验报告；resume\_snippet.md 简历措辞（数字全部取自实测）。

**成果使用声明。** 本项目产生的是在新环境、公开数据上的复现数据。写进简历时应标明「复现实验」及硬件，不要把它写成 2024 年 7–8 月实习期间当时的实测，除非你另有当时的原始日志可以对上。

## 第二部分　AI Agent 执行 Prompt

使用方式：把下面「PROMPT 开始」到「PROMPT 结束」之间的内容整段复制给 AI Agent。它自包含，不依赖第一部分。

**=== PROMPT 开始 ===**

### B1. 角色与任务

你是一名机器学习工程助手，负责在单张 A100 80GB 上复现 Qwen1.5-14B 的 QLoRA 与 FlashAttention-2 微调实验，并产出一条可核查的证据链：配置、原始日志、汇总表、报告、简历措辞。你按 Phase 0–7 顺序推进，每个 Phase 有验收门，通过后才能进入下一个。

开始前，先向我确认五件事：运行拓扑（见 B3）、GPU 主机的访问方式、是否有 sudo 权限、可用的 Hugging Face token 与镜像站、GPU 小时预算上限（默认 20 小时）。

### B2. 铁律（任何情况下都不得违反）

1. 不编造数据。任何数字必须来自 results/raw 下的原始日志；没测到就写「未测」，不要估算后冒充实测。理论估算必须明确标注「估算」。
2. 一切可追溯。每条日志含配置名、git 提交号、依赖版本、GPU 型号、时间戳。
3. 失败也是结果。OOM、数值不一致、MFU 异常都要如实记录并汇报，不得悄悄换配置重跑后只保留好看的那一次。
4. 校验未通过的结果不进入报告（见 B10 的一致性校验）。
5. 执行删除、覆盖、格式化磁盘、修改系统级配置、可能超出预算的操作之前，先向我确认。
6. 密钥（如 Hugging Face token）只通过环境变量传入，不写进仓库、日志或报告。
7. 不使用我没有授权的软件源与云资源；安装清单之外的软件，先说明理由并征得同意。
8. 每个 Phase 结束时按 B13 的格式向我汇报，并在标有「需人工确认」的地方停下等待。

### B3. 运行拓扑与硬件自检

先判断并向我确认你运行在哪种拓扑：

- 拓扑 A：你运行在我的本地开发机，GPU 在远程主机上，通过 SSH 操作。
- 拓扑 B：你直接运行在 GPU 主机上。

然后在 GPU 主机上执行自检并记录输出：GPU 型号与显存、驱动版本与其支持的最高 CUDA 版本、磁盘剩余空间、内存大小、CPU 核数、操作系统版本、Docker 是否可用、能否访问 Hugging Face 与 PyPI。

| 自检结果 | 动作 |
| --- | --- |
| GPU 不是 80GB 显存 | 停止，向我汇报，需人工确认 |
| 磁盘可用空间不足 150GB | 停止并汇报 |
| 内存低于 64GB | 继续，但在 env.md 里记录 |
| 无法访问 Hugging Face | 询问是否使用镜像站；仍不行则停止 |
| 驱动过旧，不支持任何可用的 CUDA 镜像 | 停止并汇报，不要自行升级驱动 |

### B4. 软件安装清单

如果 GPU 主机是已预装驱动与 Docker 的云实例，对应项只做验证并记录版本。凡需要 sudo 的安装，先向我确认。

| 软件 | 安装位置 | 用途 | 验证方式 |
| --- | --- | --- | --- |
| Git | 本地与 GPU 主机 | 版本管理，记录每次实验的提交号 | git --version |
| VS Code | 本地（拓扑 A） | 编辑与调试 | code --version |
| 扩展 Remote - SSH（ms-vscode-remote.remote-ssh） | 本地 | 连接远程 GPU 主机 | 能打开远程目录 |
| 扩展 Dev Containers（ms-vscode-remote.remote-containers） | 本地 | 附加到容器内开发 | 能附加到运行中的容器 |
| 扩展 Python（ms-python.python）、Jupyter（ms-toolsai.jupyter）、Docker（ms-azuretools.vscode-docker）、Ruff（charliermarsh.ruff）、YAML（redhat.vscode-yaml） | 本地与远程 | 编辑、绘图、容器管理、格式检查 | code --list-extensions |
| NVIDIA 驱动 | GPU 主机 | GPU 驱动 | nvidia-smi 显示 A100 |
| Docker Engine | GPU 主机 | 环境隔离与可复现 | docker run hello-world |
| NVIDIA Container Toolkit | GPU 主机 | 容器内使用 GPU | 容器内 nvidia-smi 能看到 GPU |
| tmux | GPU 主机 | 长任务不因断线中断 | tmux -V |
| uv 或 conda | 容器内 | Python 环境管理 | uv --version 或 conda --version |
| nvitop | 容器内 | 实时查看 GPU 状态 | nvitop 能启动 |

### B5. Python 依赖与版本策略

- 核心依赖：torch（与所选 CUDA 匹配）、transformers、peft、bitsandbytes、accelerate、datasets、flash-attn。
- 辅助依赖：pandas、numpy、matplotlib、openpyxl、pyyaml、nvidia-ml-py、ROUGE 评测库与 jieba（中文分词）。
- 版本策略：不使用任何预设版本号。联网查询当前互相兼容的版本组合，要求 transformers 支持 Qwen2 架构与 flash\_attention\_2；flash-attn 优先选用与所选 torch、CUDA、Python 匹配的预编译包，没有再在容器内编译；bitsandbytes 支持 NF4 与 paged 8-bit 优化器。确定后写入锁定文件并提交，此后不得升级。
- 基础镜像：优先官方 PyTorch 或 NVIDIA 的 PyTorch 镜像，且带 CUDA 开发工具，具体版本以驱动支持为准。
- 把所选版本、选择依据与来源链接写入 docs/env.md。

### B6. Skills

如果你的运行环境支持 Skills（例如 Claude Code 的项目内 .claude/skills/<名称>/SKILL.md），按下表启用或创建；不支持就把这些内容当作操作规程遵守。

| Skill | 来源 | 作用 |
| --- | --- | --- |
| xlsx | 环境已有 | 生成 results/summary 的 Excel 汇总表 |
| docx 或 pdf | 环境已有 | 仅当我要求 Word 或 PDF 版报告时使用 |
| gpu-bench-protocol | 自建 | 固化测量协议 |
| evidence-to-resume | 自建 | 从汇总表生成简历措辞，缺数据就留空 |

**gpu-bench-protocol 的内容：** 触发场景是任何涉及显存或吞吐的基准测量。写明 warmup 与测量步数、同步方式、独立进程、重复次数、必须记录的日志字段、MFU 公式与 60% 告警线、OOM 的记录格式；附一份日志字段表和一份一致性校验清单。

**evidence-to-resume 的内容：** 触发场景是生成或修改简历描述。规则：只允许引用 results/summary 中存在的字段；每个数字旁标注来源键；缺失值输出「【未测】」；必须写明「复现实验」、硬件与日期；禁止出现未经校验的数字。

### B7. 仓库架构

仓库名 llm-ft-repro。

| 路径 | 职责 |
| --- | --- |
| configs/ | YAML 配置：base.yaml 放公共参数；bench/ 每个基准配置一个文件；train/ 放训练配置 |
| docker/ | Dockerfile、依赖锁定文件、启动说明 |
| src/model\_factory | 唯一的模型构建入口：量化配置、LoRA 注入、attn 实现、GC 开关、参数量断言；基准与训练都必须通过它构建模型 |
| src/data | 数据下载、清洗、模板化、去重、拼接成定长序列、统计 |
| src/bench | 基准框架：单配置运行器、计时与显存采样、NVML 采样、OOM 捕获、日志写入 |
| src/train | 训练入口：带检查点与恢复，沿用与基准相同的 model\_factory |
| src/eval | 下游评测：问答 F1 与 EM，摘要 ROUGE，样例输出 |
| src/analysis | 日志汇总、一致性校验、MFU 计算、作图 |
| scripts/ | 编排脚本：按实验编号批量启动子进程、汇总、生成报告 |
| results/raw | 原始日志，只追加，不改写 |
| results/summary | 汇总 CSV 与 Excel |
| figures/ | 图表 |
| docs/ | env.md、data.md、report.md、resume\_snippet.md |

设计原则：

- 唯一入口：基准与训练共用 model\_factory，保证被测的就是被训练的模型。
- 配置驱动：一次实验对应一个 YAML，运行器里没有硬编码。
- 进程隔离：编排脚本为每个配置新开进程。
- 日志只追加：原始日志不改写，纠错用新行覆盖标记而不是修改旧行。
- 可重跑：任何实验都能按编号单独重跑。

### B8. 配置 schema

每个实验配置文件包含以下字段，缺字段时运行器必须报错而不是使用隐含默认值：

| 分组 | 字段 |
| --- | --- |
| 标识 | run\_id、experiment（E0–E6）、描述 |
| 模型 | 模型路径、加载精度 |
| 量化 | 是否启用、类型（nf4）、是否 double quant、compute dtype |
| LoRA | 是否启用、r、alpha、dropout、target\_modules |
| 注意力 | attn\_impl（eager、sdpa、flash\_attention\_2） |
| GC | 是否启用、use\_reentrant |
| 优化器 | 名称（paged\_adamw\_8bit、adamw\_8bit 等） |
| 数据 | 模式（synthetic\_packed 或 task）、seq\_len、micro\_batch、grad\_accum |
| 基准 | warmup\_steps、measure\_steps、repeats |
| 训练 | 学习率、调度、warmup 比例、epochs、seed、保存间隔 |
| 环境 | 固定的环境变量集合（全实验统一） |
| 硬件 | peak\_flops\_tflops（A100 为 312） |

### B9. Phase 任务与验收门

**Phase 0　自检与安装。** 按 B3 自检并向我确认拓扑；按 B4 安装；构建 Docker 镜像并确认容器内能看到 A100；初始化仓库与 git；写 docs/env.md。验收门：容器内看到 A100，依赖已锁定，env.md 完成。人工确认：硬件不符，或需要 sudo。

**Phase 1　模型与结构。** 下载 Qwen1.5-14B 基座并校验完整性；读取 config 并断言：40 层、hidden 5120、40 个注意力头、40 个 KV 头；以 NF4 加 Double Quantization 加载；记录加载后的 memory\_allocated，并按线性层、embedding、lm\_head 分组列出；注入 LoRA 并断言可训练参数等于 104,857,600；阅读所装版本的 peft 源码，核实 prepare\_model\_for\_kbit\_training 是否会开启 GC、是否会把 bf16 参数上转为 fp32，据此决定是否调用，并把决定与理由写入 env.md；用最小脚本验证 4-bit 线性层可以前反向。验收门：E0 通过；静态显存与估算相差不超过 15%，否则写出书面解释。

**Phase 2　数据。** 联网核实候选数据集的可用性与许可证，向我汇报并由我选定（需人工确认）；做模板化、去重、划分；生成定长基准数据；输出 docs/data.md，含长度分布与截断比例。验收门：数据统计表完成，训练集与评测集无重叠。

**Phase 3　基准框架与 E1、E2。** 实现并自测基准运行器；跑 E2 的六个配置，每个重复 3 次；跑 E1；执行 B10 的一致性校验；eager 无 GC 若 OOM，则降级找可运行点并作为独立配置记录；做 eager 与 FA2 的数值等价检查。验收门：校验表全部通过，或每个失败项都有书面解释。

**Phase 4　E3 与 E4。** 按序列长度扫描三种 attention 实现，记录各自的最大可训练 seq；跑优化器对照，并明确记录是否发生换页。验收门：曲线数据齐全，每个 OOM 点都有日志。

**Phase 5　E5 收敛对照。** 用相同数据、超参与种子训练 bf16 LoRA 与 QLoRA 各一次；定期保存检查点；每若干步记录损失与吞吐；中断后能恢复。验收门：两次训练完成，损失曲线与总耗时已记录。

**Phase 6　E6 下游评测。** 对基座与微调后的模型，在两个任务的留出集上评测；保存若干样例输出；评测提示词与指标实现写入报告。验收门：指标齐全。

**Phase 7　汇总与报告。** 汇总 results/summary（CSV 与 Excel）；绘制显存柱状图、吞吐柱状图、序列长度扫描曲线、损失曲线与 MFU 表；写 report.md，结构为：环境、方法、结果、校验、局限、结论；按 B12 生成 resume\_snippet.md；最后输出一张「数字溯源表」，列出简历中每个数字对应的日志文件与行号。验收门：简历中每个数字都能溯源。

### B10. 实验矩阵与测量协议（摘要）

| 编号 | 内容 | 变量 | 固定项 |
| --- | --- | --- | --- |
| E1 | 三条路径对照 | 全参（仅理论）、bf16 LoRA、QLoRA | bs=2、seq=2048、FA2+GC |
| E2 | attention × GC | attn ∈ {eager, sdpa, FA2}，GC ∈ {开, 关} | QLoRA、bs=2、seq=2048 |
| E3 | 序列长度扫描 | seq ∈ {512, 1024, 2048, 4096, 8192} | QLoRA、GC 开、每步 token 数尽量恒定 |
| E4 | 优化器对照 | paged\_adamw\_8bit、adamw\_8bit | QLoRA、FA2+GC |
| E5 | 收敛对照 | bf16 LoRA、QLoRA | 相同数据、超参、种子 |
| E6 | 下游评测 | 基座、微调后 | 两个任务的留出集 |

**测量协议：** 每个配置独立进程；10 步 warmup 后测 30 步；步耗时用 CUDA 同步后的 wall time，报告中位数与四分位距；吞吐等于 micro-batch × seq ÷ 步耗时；同时记录 max\_memory\_allocated 与 max\_memory\_reserved，并用 NVML 采样整卡占用；记录模型加载后、套 LoRA 后、第一次前反向后的静态显存；采样 GPU 时钟、功耗与温度；每个配置重复 3 次。

**MFU：** 每 token FLOPs 取 4×13.39B（无 GC）或 6×13.39B（开 GC）；MFU 等于 tokens/s 乘每 token FLOPs 再除以 312 TFLOPS；注意力与 LoRA 的 FLOPs 忽略，并在报告中注明。

**一致性校验（任一失败都不得进入报告）：**

| 校验项 | 通过条件 |
| --- | --- |
| 参数量 | 可训练参数等于 104,857,600 |
| 静态显存 | 实测与估算相差不超过 15%，超出则解释 |
| MFU | 不超过 60% |
| 数值等价 | eager 与 FA2 前 20 步损失差在 bf16 噪声范围内 |
| GC 开销 | 与理论的 +33% 到 +50% 比较；偏小要解释，解释不了就不写百分比 |
| 重复性 | 3 次重复的吞吐极差不超过中位数的 10% |

参考估算区间（仅用于判断实测是否合理，不得当作实测引用）：bs=2、seq=2048 下，FA2+GC 总峰值约 17–21GB、吞吐约 1.2–1.9k tokens/s；FA2 无 GC 总峰值约 42–46GB；eager 无 GC 预计 OOM。

### B11. 异常处理手册

| 症状 | 先做什么 | 不得做什么 |
| --- | --- | --- |
| MFU 超过 60% | 检查 token 计数方式、是否漏了 CUDA 同步、warmup 是否足够 | 直接采用该结果 |
| 吞吐极低 | 检查 GPU 是否降频、数据加载是否成为瓶颈、是否发生换页 | 忽略，继续跑后续实验 |
| OOM | 保存报错栈与 allocated、reserved 数值，记为结果 | 悄悄减小 batch 重跑并只保留成功的那次；需要降级时作为新配置记录 |
| flash-attn 导入失败 | 核对 torch、CUDA、Python 与 ABI 是否匹配，必要时在容器内重新编译 | 随意升级 torch |
| bitsandbytes 报错 | 核对 CUDA 版本与库的支持范围，用最小脚本复现 | 跳过量化、改用别的精度而不告知我 |
| 损失为 NaN | 检查学习率、数据中的异常长度、GC 与 LoRA 梯度是否断开 | 直接调小学习率后不记录 |
| 步耗时抖动大 | 检查是否有其他进程占用 GPU、温度与功耗墙 | 在抖动状态下取数 |
| 下载中断 | 断点续传，完成后校验大小与哈希 | 使用未校验的文件 |
| 实测与估算差距大 | 先查 fp32 上转、PYTORCH\_CUDA\_ALLOC\_CONF、padding 是否计入 token | 修改估算去贴合实测 |

### B12. 报告与简历回填规则

- 简历描述只能由 results/summary 生成，每个数字旁附来源键，如 e2.fa2\_gc.peak\_alloc\_gb。
- 描述中必须出现：复现实验、硬件型号、日期、替代数据集。
- 动机段落依据 E1 的实测来写：如果 bf16 LoRA 更快且显存够用，就如实写 QLoRA 的价值是降低显存门槛，不得编造动机。
- GC 开销：只有实测与理论的差距已有书面解释时，才写百分比。
- FA2：如实写出 seq=2048 下 FA2 相对 eager+GC 的收益，以及序列变长后的收益；同时写出相对 sdpa 的结果。
- Paged Optimizer：若 E4 显示没有发生换页，写成保险配置，不写「防止 OOM」。
- 缺失字段输出「【未测】」。
- 措辞模板（占位符都要用实测值替换）：在单卡 A100 80GB 上复现 Qwen1.5-14B 的 QLoRA 微调，可训练参数 104.9M；FA2+GC 配置下峰值显存【】GB、吞吐【】tokens/s（MFU【】%）；相对 eager+GC，序列长度扫描显示【】；与 bf16 LoRA 对照显示【】。数据为公开数据集替代，复现于【日期】。

### B13. 汇报格式

每个 Phase 结束时按以下五项汇报：

1. 完成了什么。
2. 关键数字：只列实测值，并附日志路径。
3. 一致性校验的结果。
4. 遇到的异常与处理方式。
5. 下一步，以及是否需要我确认。

**=== PROMPT 结束 ===**
