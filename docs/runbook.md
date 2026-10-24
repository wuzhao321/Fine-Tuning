# 运行手册

本手册覆盖实验身份固定、数据准备、E0–E6 执行和报告生成。所有命令从仓库根目录运行，GPU 命令需要与配置匹配的 CUDA 环境和已登记数据。

## 1. 先冻结实验身份

- 按 env.md 选定 Python / torch / CUDA / transformers / peft / bitsandbytes / flash-attn 组合，完成最小验证后保存锁文件和镜像 digest。
- 把 configs/base.yaml 的 `model.revision` 从 `main` 改成已核实的 Hugging Face commit SHA；数据准备和全部实验使用同一 revision。当前 `main` 只方便阅读配置，实际远程模型/tokenizer 加载会拒绝未固定的 revision。
- 配置 `CUDA_VISIBLE_DEVICES` 与 `hardware.nvml_device` 为同一块物理卡；后者可填 GPU UUID。A100 默认峰值 312 TFLOPS，换卡要核实稠密 bf16 峰值。
- 提交代码、配置和已锁定依赖，记录提交号。汇总会阻止缺少提交号的结果被引用。
- 在目标机器检查 80GB 显存、150GB 空闲磁盘、64GB 内存；记录 GPU 型号、驱动、CPU、功耗限制。资源要求应与具体实验配置一并核实。

## 2. E0

```bash
python scripts/repro.py validate
python scripts/repro.py matrix E0 --execute
```

这会构建完整模型并执行一个极短前反向。检查 raw 日志中的结构、可训练参数、GC、精度和显存分组。模型下载由 Hugging Face 按所配置 revision 进行，token 仅通过环境传入。若需要离线固定快照，可在另行下载完成后验证：

```bash
python scripts/hash_model.py /path/to/model-snapshot --output /path/to/model-manifest.json
```

此命令检查索引引用的 shard 存在并计算本地 SHA256，不会把“有哈希”误称为“已与上游权威哈希逐一比对”。本地模型目录的版本身份还需记录到 env.md。

## 3. 准备数据

按 docs/data.md 填写真实来源清单和 normalized JSONL，不能使用仓库里的演示句子作为实验数据。数据下载和字段映射应依据已选来源的版本与许可进行。

```bash
python scripts/repro.py prepare-data --config configs/train/e5_qlora.yaml \
  --input /path/to/normalized.jsonl --sources /path/to/approved-sources.json \
  --output data/processed/tasks

python scripts/repro.py pack-data --config configs/bench/e1_qlora.yaml \
  --input /path/to/chinese-corpus.txt --sources /path/to/approved-sources.json \
  --output data/processed/bench/tokens.u32
```

默认至少 327,680 token，覆盖 8192 长度 × batch 1 × 40 步。任务数据与 benchmark 文本必须分别准确登记来源；多来源合并文本的归属由来源清单负责。数据输出目标必须尚不存在，避免不小心覆盖已参与实验的数据。

## 4. E1–E4

```bash
python scripts/repro.py plan E2
python scripts/repro.py preflight --config configs/bench/e1_qlora.yaml
python scripts/repro.py matrix E1 --execute
python scripts/repro.py matrix E2 --execute
python scripts/repro.py matrix E3 --execute
python scripts/repro.py matrix E4 --execute
```

E1 / E2 / E3 / E4 分别计划 6 / 18 / 45 / 6 个独立子进程。矩阵每次执行都会新增尝试，不自动跳过旧日志。已完成矩阵请勿无意重复执行；有计划的重跑应使用独立 raw 目录及审查说明。

矩阵执行前会先预检所有正式配置的本地输入；每次启动前再次核对配置哈希，配置在运行中被修改会停止。预算账本固定在项目 results/raw；测量日志遵循配置的 output.raw_dir。

E2 eager 无 GC 第一次若 OOM，会追加 `e2_eager_fallback`（batch 1 / seq 1024）的三个独立尝试。原配置 OOM 保留，剩余相同配置的重复会跳过并写入账本，降级点不代替原表。其他 OOM 不自动改变配置，同样跳过剩余重复。出现 OOM 后矩阵可继续后续不同配置，退出码 2 表示包含失败结果。非 OOM 失败或子进程缺少终止证据时立即停止，避免继续消耗预算。

每个子进程使用唯一 job_id 与结果日志关联，不以最新修改时间推断结果。修改 raw_dir 或目录里保留旧 OOM 日志不会把历史失败误认为当前任务结果。

数值等价检查独立于性能矩阵，不开 dropout。提前在审查记录中写明容差理由，再运行：

```bash
python scripts/repro.py bench --config configs/diagnostics/e2_equivalence_eager.yaml
python scripts/repro.py bench --config configs/diagnostics/e2_equivalence_fa2.yaml
# LEFT / RIGHT 替换成上述两个 raw JSONL；容差替换成事前确定的值。
python scripts/repro.py compare-numerics LEFT RIGHT --atol ATOL --rtol RTOL
```

诊断是 batch 1 / seq 1024 / GC 开的局部比较，不能声称覆盖全部长度和无 GC 配置；必要时增加明确命名的配对诊断。不得看到结果后随意放宽容差。

## 5. E5

```bash
python scripts/repro.py matrix E5 --execute
# 中断恢复使用完整 checkpoint 路径，而非 final_adapter：
python scripts/repro.py preflight --config configs/train/e5_qlora.yaml \
  --resume checkpoints/e5_qlora/ATTEMPT/checkpoint-100
python scripts/repro.py train --config configs/train/e5_qlora.yaml \
  --resume checkpoints/e5_qlora/ATTEMPT/checkpoint-100
```

恢复校验完整训练身份：配置、三个数据划分、manifest、源码快照、Python、平台和记录的包版本。保存结束时写入 artifact-manifest.json，包含所有文件的大小与 SHA256；恢复前逐一核对 adapter、优化器、调度器、RNG 和 Trainer 状态，检查 global_step 与目录名一致。不反序列化状态文件来完成预检。

源码指纹包含 src、scripts、configs；因此修改其他实验配置也会使严格恢复身份变化。需要继续原实验时使用原始快照，不要绕过校验。历史未带完整性记录的检查点会被拒绝；不能凭文件存在就宣称可精确恢复。

新尝试写入新目录并记录父检查点及其完整性记录哈希，旧证据保留。自动只保留最近三个中间检查点（TrainingArguments 的 save_total_limit）；最终 adapter 单独保存并生成完整性记录。恢复后应核对优化器状态、RNG 状态和 global_step，以确认训练连续性。

## 6. E6

把 configs/eval/e6_finetuned.yaml 的 adapter_path 改成 E5 QLoRA 生成的 final_adapter 目录，再运行：

```bash
python scripts/repro.py matrix E6 --execute
```

基座和微调后均采用同一 NF4 加载方式、同一测试 prompt_ids、贪心解码和相同生成上限。每个样本的生成文本、参考答案、指标都写入 raw，不使用外部 LLM 评审。

## 7. 汇总与报告

```bash
python scripts/repro.py summarize
python scripts/repro.py review-template --output /path/to/reviews.json
# 按 evidence.md 完成审查记录之后：
python scripts/repro.py summarize --reviews /path/to/reviews.json --excel --figures
```

默认输出 results/summary；图表在 results/summary/figures。没有原始数据时生成空表和“未测”报告；有数据但未审查时生成审计表，不自动发表数字。docs/report.md 是说明模板，实际派生报告是 results/summary/report.md。

review-template 不自动批准任何结果，且拒绝覆盖已有审查文件。新增实测后按新分组与摘要生成新草稿；不要把数据、配置或文件已改变后的旧审批复制为有效状态。

## 预算与中断

矩阵累计墙钟写入 results/raw/budget.jsonl。累计超过 15 小时时触发确认阈值；核对资源账本后可提高 `--acknowledged-hours`，其值仍不能超过 `--budget-hours`。不要删除账本重置预算。

如果 launcher 被强制杀死，检查该 job 的子进程是否仍运行并核实耗时，然后向账本追加同 job_id 的 `job_finished` 记录（elapsed_s、returncode、outcome），保留原行。这是人工核对步骤，不自动猜测时间。
