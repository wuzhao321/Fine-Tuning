# 环境配置与版本冻结

## 运行要求

GPU 训练使用 Linux、Python 3.10+ 和支持 bf16 的单张 CUDA GPU。性能对照配置以 A100 80GB 为参考硬件；使用其他型号时，需调整 `hardware.peak_flops_tflops` 并记录硬件规格。配置校验、执行计划与基础汇总支持独立于 GPU 的运行方式。

`pyproject.toml` 声明功能依赖；`docker/requirements.in` 列出容器依赖。环境复现需固定全部传递依赖、wheel 哈希和基础镜像 digest，构建步骤见 [容器说明](../docker/README.md)。

## 实验环境记录

| 类别 | 应记录的字段 |
| --- | --- |
| GPU | 型号、UUID、显存、SXM/PCIe、功耗限制、稠密 bf16 峰值 |
| CUDA | 驱动版本、CUDA runtime、编译工具链 |
| 主机 | OS、CPU、RAM、磁盘空间 |
| 容器 | 镜像名称、digest、运行参数 |
| Python 依赖 | Python、torch、transformers、peft、bitsandbytes、accelerate、flash-attn 的版本 |
| 模型 | 基座、tokenizer revision 与模型文件哈希 |
| 数据 | 来源、版本、许可证与处理后文件哈希 |
| 代码 | Git commit、源码指纹、完整实验配置 |

## 兼容性与模型身份

torch、CUDA、Python、flash-attn 的 ABI 与 bitsandbytes 支持范围需一起核对。架构支持的最低版本不等于项目所有训练接口的兼容下限。

模型加载使用 `dtype`，训练入口使用 `processing_class` 和 `eval_strategy`。选用其他版本时，应核查相应 API、loss 归一化和恢复行为，并将修改后的组合重新冻结。版本差异与源码对照见 [工程资料](research/engineering-and-measurement.md)。

远程模型 `model.revision` 必须固定为完整 commit SHA；数据准备和所有对照配置使用同一 revision。本地快照可通过 `scripts/hash_model.py` 生成文件清单。

## 运行检查

E0 检查单卡 CUDA/bf16、模型结构、lm_head 非量化、实际可训练参数量、GC 状态、k-bit 准备前后 dtype 与梯度连通性。检查失败时保存原始错误和配置，依据已固定的依赖组合诊断。

原始日志的 `source_sha256` 覆盖源码、脚本与配置；包版本和镜像 digest 记录依赖身份。两者共同用于实验溯源。

## 官方参考

- [Qwen1.5-14B 配置](https://huggingface.co/Qwen/Qwen1.5-14B/blob/main/config.json)
- [PEFT 模型接口](https://huggingface.co/docs/peft/en/package_reference/peft_model)
- [PEFT k-bit 准备函数](https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py)
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
