# 容器构建

Dockerfile 需要显式传入与目标 GPU 兼容的 PyTorch/NVIDIA CUDA development 基础镜像，以及包含精确版本和哈希的 requirements.lock。构建会拒绝缺失的基础镜像参数或锁文件，以保证依赖可追溯。

构建流程：

1. 根据 GPU 驱动、Python、torch、CUDA 与 ABI 选择镜像并记录 digest。
2. 选择 transformers、peft、bitsandbytes、accelerate 和 flash-attn 的兼容组合。优先使用合适的 FA2 wheel；如需编译，应单独构建、记录编译参数并计算 wheel SHA256。
3. 在目标环境完成 E0 后，生成包含传递依赖、精确版本与哈希的 requirements.lock，提交并记录来源。基础镜像预装 torch 与锁文件必须一致，不任意替换。
4. 再使用 `docker build --build-arg BASE_IMAGE=已选镜像@sha256:实际摘要 -f docker/Dockerfile .`。

Dockerfile 的 `--require-hashes` 要求完整锁定全部依赖。仅 `pip freeze` 的版本快照不足以作为该文件。pyproject 构建所需 setuptools 也须由锁文件或镜像满足。

建议运行时将模型缓存、数据、checkpoints 和 results 挂载到持久目录；仅通过环境或安全 secret 传入 HF_TOKEN，不把 token 作为 build-arg。
