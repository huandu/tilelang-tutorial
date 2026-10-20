# 第 17 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）、CPU 环境；**没有 NVIDIA
GPU，也没有昇腾 NPU**。因此涉及加速器的代码示例最高验证级别是「仅静态核对」；凡是
标注「CPU 逻辑验证」的，都在本机真实运行过，输出已写进正文与各示例 README。

---

## A. 锁定源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| V1 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:96`、`:111-113`、`:144-148` | `tilelang.jit(out_idx=..., execution_backend=...)` 的签名、`out_idx` 语义、`T.empty` 与 `out_idx` 冲突检查 | 已核验（2026-10-01）：阅读完整函数签名与 docstring |
| V2 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:206-222` | `JITKernel.__call__` 转发到 `self.torch_function`——kernel 调用返回的就是 torch 张量 | 已核验（2026-10-01） |
| V3 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:340-405` | `_create_adapter` 的分派：`tvm_ffi`/`cython`/`nvrtc`/`torch`/`cutedsl`；其中 `torch` 后端 `assert is_metal_target(target)` | 已核验（2026-10-01）：逐行阅读，确认 `torch` 后端只服务 Metal |
| V4 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/torch/__init__.py:1-3` | torch adapter 目录里**只导出** `MetalKernelAdapter` | 已核验（2026-10-01）：`ls tilelang/jit/adapter/torch/` 只有 `__init__.py`、`kernel_cache.py`、`metal.py` |
| V5 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/adapter.py:403-420` | `_process_buffer_device`：`is_ascend_target` → `torch.device("npu")`，`is_cuda_target`/`is_hip_target` → `torch.device("cuda")` | 已核验（2026-10-01） |
| V6 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/adapter.py:440-460` | `_convert_torch_func`：把 torch 张量指针转成 C `void*` 调 `lib.call` | 已核验（2026-10-01） |
| V7 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/base.py:27-53` | `BaseKernelAdapter.result_idx`：`out_idx` 的正/负索引如何合法化为参数下标 | 已核验（2026-10-01） |
| V8 | `_vendor/tilelang-v0.1.15/tilelang/backend/execution_backend.py:1-27` | `ExecutionBackendSpec`：`supports_callee_allocated_outputs` 决定 out_idx 输出由 host 函数分配 | 已核验（2026-10-01） |
| V9 | `_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:24-33` | CUDA 默认后端顺序：`tvm_ffi` → `nvrtc` → `cython`（`auto` 取第一个可用的） | 已核验（2026-10-01） |
| V10 | `_vendor/tilelang-v0.1.15/tilelang/ascend/execution_backend.py:1-21` | 昇腾默认后端：`tvm_ffi` → `cython`，注释说明 out_idx 的 ABI 由后端声明 | 已核验（2026-10-01） |
| V11 | `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:39-66` | 最小 RMSNorm（`T.copy`/`T.Parallel`/`T.reduce_sum`/`T.rsqrt`）与 `ref_program` | 已核验（2026-10-01）；**未运行**（无 GPU） |
| V12 | `_vendor/tilelang-v0.1.15/examples/norm/layernorm.py:6`、`:61` | `@tilelang.jit(out_idx=[-3,-2,-1])` 的多输出写法 | 已核验（2026-10-01） |
| V13 | `_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py:1-45` | attention 示例：`@tilelang.jit(out_idx=[3], pass_configs={...})`、`T.Kernel(..., threads=...)` 的网格 | 已核验（2026-10-01）；**未运行** |
| V14 | `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py` | MoE 的 fused 算子示例（用于说明「MoE 只在模型里有 MoE 时才值得换」） | 已核验（2026-10-01）：文件存在，未逐行精读 |
| V15 | `_vendor/tilelang-v0.1.15/examples/ascend/README.md:1-35` | NV 仓库内昇腾示例清单（GEMM/Split-K/VecAdd/RMSNorm/FP8）与运行方式 | 已核验（2026-10-01） |
| V16 | `_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:44-70` | 昇腾 RMSNorm：`T.alloc_reducer`/`T.reducer_init`/`T.finalize_reducer`/`T.SimtVF`/`T.annotate_buffer_versions` | 已核验（2026-10-01） |
| V17 | `_vendor/tilelang-v0.1.15/examples/ascend/test_rmsnorm.py:15-16` | 昇腾调用形态：`tilelang.compile(..., target="ascend", out_idx=[1,3])`，张量用 `.npu()` | 已核验（2026-10-01） |

## B. 锁定源码引用（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:52` | 「已测试并验证过的 NPU：A2 和 A3」 | 已核验（2026-10-01） |
| A2 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:79` | 安装前提：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1 | 已核验（2026-10-01） |
| A3 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:27` | `torch_tl_ascend` PyTorch 集成示例（PR#292） | 已核验（2026-10-01） |
| A4 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:32` | 自动核内同步插入（`InsertSync` 思路，PR#74） | 已核验（2026-10-01） |
| A5 | `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/README.md` | 路径 B：`setup.py install` → `torch.ops.tl_ascend.flash_attention` | 已核验（2026-10-01） |
| A6 | `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/test_torch.py`、`test_source.py` | 两条调用路径的完整可运行形态（含 `torch_npu`、`torch.testing.assert_close`） | 已核验（2026-10-01）；**未运行**（无 NPU） |
| A7 | `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/src/_inner.cpp:1-60` | C++ 侧如何取当前 NPU stream（`c10_npu::getCurrentNPUStream`）并注册 `TORCH_LIBRARY` | 已核验（2026-10-01） |
| A8 | `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/compile_tl_op/flash_attention.py:1-60` | 编译产物 `.so` 与算子源码如何打进 Python 包（`CythonKernelAdapter` + `libpath`） | 已核验（2026-10-01） |

## C. corpus 资料引用（`../tilelang-corpus/13-e2e-training-and-serving/raw/`）

下表中的 `❌` 表示原始抓取页在 `_index.md` 里被标记为「官网直连不可达、改用
`hf-mirror.com` 镜像」，内容与官网一致，引用时以官网地址为准；`✅` 表示从源仓库
抓取成功。

| id | corpus 路径 | 原始 URL | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-datasets-docs.md` | `https://huggingface.co/docs/datasets/index` | 流式读取、`streaming=True`、Arrow 零拷贝；数据管线权威出处 | 已核验（2026-10-01）：本地文件存在；原文为索引页，深度细节需查官方 how-to |
| R2 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-qwen2-5-7b-card.md` | `https://huggingface.co/Qwen/Qwen2.5-7B` | 7B 结构参数：7.61B（非嵌入 6.53B）、28 层、GQA 28Q/4KV、RoPE/SwiGLU/RMSNorm/QKV bias、context 131072 | 已核验（2026-10-01）：逐项对照原文 |
| R3 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-fineweb-edu-card.md` | `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu` | 语料规模：1.3T token 保底版；10B/100B/350B 子集；35 亿行、5.84 TB | 已核验（2026-10-01）：卡片中数值直接抄录 |
| R4 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-tinystories-card.md` | `https://huggingface.co/datasets/roneneldan/TinyStories` | 小规模语料：合成短故事，用来「先跑通再放大」 | 已核验（2026-10-01）：卡片中数值直接抄录 |
| R5 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-transformers-docs.md` | `https://huggingface.co/docs/transformers/index` | `Trainer`、混合精度、`torch.compile`、FlashAttention | 已核验（2026-10-01）：本地文件存在；原文为索引页 |
| R6 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-accelerate-docs.md` | `https://huggingface.co/docs/accelerate/index` | 「四行代码」把训练代码搬到多卡；用于说明分布式不归 TileLang 管 | 已核验（2026-10-01） |
| R7 | `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-peft-docs.md` | `https://huggingface.co/docs/peft/index` | LoRA/QLoRA 等参数高效微调；「不做全量训练时优化哪几个算子」 | 已核验（2026-10-01） |
| R8 | `../tilelang-corpus/13-e2e-training-and-serving/raw/github-pytorch-torchtitan.md` | `https://github.com/pytorch/torchtitan` | 官方训练参考：FSDP2/TP/PP/CP、gradient accumulation、async checkpointing、bf16 optimizer states、MFU 日志 | 已核验（2026-10-01）：README 全文抽取成功 |
| R9 | `../tilelang-corpus/13-e2e-training-and-serving/raw/github-litgpt.md` | `https://github.com/Lightning-AI/litgpt` | 可读性最强的训练/微调参考；Qwen2.5 全系列规模表 | 已核验（2026-10-01） |
| R10 | `../tilelang-corpus/13-e2e-training-and-serving/raw/github-nanogpt.md` | `https://github.com/karpathy/nanoGPT` | 最小可读训练循环；GPT-2 124M 在 8×A100 40GB 上约 4 天、val loss ≈ 2.85 | 已核验（2026-10-01）：README 全文抽取成功 |

## D. 上游链接

* `https://huggingface.co/docs/datasets/stream` —— datasets 流式读取（已核验其
  官网可解析，2026-10-01；本机直连 huggingface.co 不可达，改用镜像核对，见 R1 的抓取说明）。
* `https://huggingface.co/Qwen/Qwen2.5-7B` —— 模型卡（同上，镜像核对）。
* `https://github.com/pytorch/torchtitan` —— 训练参考实现（已核验可达，2026-10-01）。
* `https://github.com/hiascend` / CANN 8.3.RC1 下载页 —— 昇腾工具链（引用 A2 的链接，
  **未核验**：本机无昇腾环境，未打开下载页）。

## E. 待补资料

写本章时在 corpus 内没有找到、但对正文有直接帮助的条目，交由后续版本决定是否补采：

1. **（已关闭 2026-10-01）Qwen2.5-7B 的 `config.json` 原文** —— 已补入
   `../tilelang-corpus/13-e2e-training-and-serving/raw/hf-qwen2-5-7b-config.md`（含原始
   JSON），`hidden_size=3584` / `intermediate_size=18944` / `vocab_size=152064` /
   `rope_theta=1e6` 已可直接引用；`tokenizer_config.json` 仍未存档。
   （`hidden_size`/`intermediate_size`/`vocab_size`/`rope_theta` 的精确值）。目前 §5 的
   结构参数里，只有层数（28）与 GQA 头数（28/4）直接来自模型卡散文；`hidden=3584`、
   `intermediate=18944`、`vocab` 是依据公开配置的常见值填写，**需要以官方配置文件为准**。
   建议来源：`https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json`。
2. **torchtitan 的 `train_configs` 与 `loss_compare.py` 原文**。R8 只抽到了 README，
   要讲清「多种精度下的 loss 偏差门限」需要源码。
3. **`datasets` 的 `streaming` + `IterableDataset.shuffle(buffer_size=...)` 官方 how-to**。
   R1 只抽到索引页，正文 §4 的 shuffle buffer 语义需要该页佐证。
4. **CANN / torch-npu 在训练态下的已知限制清单**（哪些算子回退 CPU、ACLGraph 是否
   支持反向）。这决定 §8「昇腾训练现实约束」能写多细。

## F. 引用与版本规则

* 源码引用一律写成 `_vendor/<repo>/<相对路径>:<行号区间>`；行号以 2026-10-01 的锁定工作树为准。
* 资料引用一律写成 `../tilelang-corpus/...`，上游 URL 用完整地址。
* 「已核验」＝在锁定版本工作树上用 `grep -n`/`sed` 定位并阅读过；**不等于运行验证**。
