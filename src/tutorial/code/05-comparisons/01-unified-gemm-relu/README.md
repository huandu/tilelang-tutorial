# 01-unified-gemm-relu — 统一例子：`C = relu(A @ B)`

## 目的

把**同一个**算子“带融合 epilogue 的 GEMM”在六个系统里的写法并排放，用来观察：

* 用户必须提供的硬件相关决策有哪些（搬数据 / 切 tile / 同步）；
* 哪些决策被搬进了编译器；
* 语言的层次差异，而不是“谁的行数少”。

统一参数（与 TileLang 官方 quickstart 一致）：`M = N = K = 1024`，fp16 输入、fp32
累加，tile `128×128×32`，融合 ReLU。

## 文件

| 文件 | 系统 | 性质 | 来源 |
| --- | --- | --- | --- |
| `tilelang_gemm_relu.py` | TileLang | 真实 API（未运行） | `_vendor/tilelang-v0.1.15/examples/quickstart.py:1-56` |
| `triton_gemm_relu_sketch.py` | Triton（今天） | **教学伪代码**，未在 corpus 内校验 API | 语义按 R2/R4 重写 |
| `cuda_gemm_relu_illustrative.cu` | CUDA | **教学伪代码** | 结构依据 R8:1745-1830 |
| `cute_dsl_gemv_excerpt.py` | CUTLASS-CuTe（CuTe DSL） | **真实片段**（GEMV，非 GEMM） | R27（TileFoundry） |
| `ascendc_leakyrelu_excerpt.cpp` | Ascend C | **真实片段**（逐元素，非 GEMM） | R14:350-440 |
| `pto_dsl_matmul_excerpt.py` | PTO-DSL | **真实片段**（matmul 基线） | R19 |

> 说明：CUTLASS-CuTe 与 Ascend C 两侧，本 corpus **没有**“GEMM + 融合 epilogue”的完整
> 源文件，因此这里用各自仓库里能拿到的真实片段来展示“同层的写法长什么样”。缺口登记
> 见本章 `references.md` 的“待补资料”第 2、3 条。

## 运行前提与命令

* `tilelang_gemm_relu.py`：需要 NVIDIA GPU（或昇腾 950 环境，需把 `target` 改成
  `"ascend"` 并改写为昇腾方言）+ 与锁定版本一致的 TileLang。命令：
  `python tilelang_gemm_relu.py`（原示例会 `assert_close` 对齐 `torch.relu(a @ b)`）。
* `triton_gemm_relu_sketch.py`：需要 NVIDIA GPU + `pip install triton`。**未在本机验证**。
* `cuda_gemm_relu_illustrative.cu`、`ascendc_leakyrelu_excerpt.cpp`：是节选/示意，
  不能单独编译；只用于阅读。
* `cute_dsl_gemv_excerpt.py`：需要 `nvidia-cutlass-dsl` 与该示例的其余文件；**仅节选**。
* `pto_dsl_matmul_excerpt.py`：需要 PTO-DSL + CANN + 昇腾 NPU；**仅节选**。

## 预期输出（正确性判据 + 性能口径）

* 正确性判据：与 `torch.relu(A.float() @ B.float())` 比较，`rtol=1e-2, atol=1e-2`
  （TileLang 原示例用的就是这组容差）。
* 性能口径：**本章不给性能结论**。任何数字都必须带来源与测量条件；本目录不含任何
  实测数字。

## 作者实际做到的验证级别

* `tilelang_gemm_relu.py`：`仅静态核对`（API 名与签名逐个在 `_vendor/tilelang-v0.1.15/`
  比对；**未运行**）。
* `triton_gemm_relu_sketch.py`：`仅静态核对`（仅保证语法可解析；API 名未在 corpus 内校验）。
* 其余：`仅静态核对`（片段/示意，未运行）。

## 在有卡机器上怎么验证

1. NVIDIA 机器：按 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` 之外的
   `README.md` quickstart 流程安装 TileLang，跑 `tilelang_gemm_relu.py`；
   用 `kernel.get_kernel_source()` 打印生成的 CUDA，确认 ReLU 与 GEMM 在同一 kernel 内。
2. 昇腾机器：按 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:14-45` 安装
   （CANN + `torch_npu`，`USE_ASCEND=ON USE_CUDA=OFF`），跑同一节里的昇腾版
   `gemm_relu.py`。
3. 记录时务必写清：GPU/NPU 型号、软件版本、形状、tile、重复次数与统计口径。
