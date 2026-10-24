# 环境状态与版本冻结

更新日期：2026-10-24（Asia/Shanghai）。

**状态：目标 GPU 环境未搭建、未核验；没有声明任何已验证的依赖组合。** 本次仅使用本地已有 Python 做代码与配置的轻量检查。没有安装软件、使用 sudo、启动 Docker 或连接云主机。

## 后续填写

| 项目 | 当前记录 |
| --- | --- |
| 运行拓扑与访问方式 | 未确定 |
| GPU、UUID、SXM/PCIe、显存 | 未测 |
| NVIDIA 驱动与 CUDA runtime | 未测 |
| OS、CPU、RAM、磁盘余量 | 未测 |
| 镜像名称与 digest | 未选定 |
| Python / torch / CUDA / flash-attn | 未锁定 |
| transformers / peft / bitsandbytes / accelerate | 未锁定 |
| 模型与 tokenizer revision | 配置暂为 main；正式准备数据前必须换成 commit SHA |
| 数据集版本与许可证 | 待用户选定并核实 |
| 实验代码提交 | 尚无初始提交 |

pyproject.toml 中是**功能依赖声明，不是已验证的可重现环境**。`docker/requirements.in` 是选择清单，不能当作 lock。flash-attn 与 torch/CUDA/Python/ABI 的组合必须在实际机器上确定后才能锁定；不承诺当前最新版本彼此兼容。

## 已参考的官方接口

- [Qwen1.5-14B 模型配置](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json)：结构断言依据。
- [PEFT 模型与 adapter 加载](https://huggingface.co/docs/peft/en/package_reference/peft_model)：adapter 注入与恢复接口。
- [PEFT k-bit 准备函数源码](https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py)：GC 默认行为与精度转换；运行时还会保存实际安装版本的函数哈希。
- [Transformers bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)：NF4、Double Quantization 与 compute dtype。
- [Transformers Trainer 文档](https://huggingface.co/docs/transformers/en/main_classes/trainer)：processing_class、eval_strategy、训练恢复接口。

代码使用当前接口名称（例如 `dtype`、`processing_class`、`eval_strategy`）。并未用兼容分支悄悄改变行为；目标环境若采用旧版本，须确认接口后明确修改并重新冻结。

## 运行前最小验收

先确认 CUDA/bf16/单卡，再运行 E0。核实 lm_head 未量化、实际可训练参数量、GC 状态、k-bit 准备前后的 dtype、第一步梯度连通性。FA2/bitsandbytes 加载失败时按组合诊断，不能自动升级某一个包或改用另一种 attention。

成功后保存实际包版本和镜像 digest，计算模型 shard 哈希，提交代码和配置。原始日志的 source_sha256 覆盖源码、脚本和配置，但不代替完整依赖锁定。
