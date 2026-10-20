# 第 03 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，`VERSION` 文件 = `0.1.15`）、
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`
（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。** 正文与示例中不再重复版本号，
也不对「上游现在是什么状态」作任何未核验的断言。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）；**没有 NVIDIA GPU，也没有
昇腾 NPU**。因此本章代码示例的最高验证级别是「仅静态核对」，另一个纯标准库脚本是
「CPU 逻辑验证」（见 `code/README.md` 的验证级别总表）。**本章不包含任何性能数字。**
文中所有版本结论都锚定在上表给出的两个 tag 上。

核验日期均为 2026-10-01。核验方式：在锁定版本的工作树上用 `grep -n` 定位定义行，
阅读完整签名与 docstring，并逐行核对正文引用的代码片段。**这不是运行验证。**

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S1 | `VERSION` | 版本事实 = `0.1.15` | 已核验（2026-10-01） |
| S2 | `tilelang/language/__init__.py:1-12`、`:32` | 门面：`tilelang.language` 再导出 CUDA 方言，`__tilelang_dialect__ = "cuda"` | 已核验（2026-10-01）：全文 32 行逐行阅读 |
| S3 | `tilelang/language/common.py:1-8`、`:17`、`:79`、`:144-145`、`:186-260` | 后端无关语言面：模块 docstring、`from .eager import *`、`gemm`/`dynamic`/注解的再导出、`__all__` 拼装 | 已核验（2026-10-01） |
| S4 | `tilelang/language/kernel.py:431`、`:458`、`:587` | `T.Kernel` / `T.ClusterKernel` / `T.get_thread_binding` | 已核验（2026-10-01） |
| S5 | `tilelang/cuda/language/kernel.py:12-54` | CUDA 方言的 `T.Kernel`：docstring `:21-25`（「exposes `threadIdx`」）、`threads` 缺省 128（`:32-35`）、`cluster_dims` | 已核验（2026-10-01） |
| S6 | `tilelang/language/allocate.py:49`、`:67`、`:81`、`:400` | `alloc_shared`（`shared.dyn`）/ `alloc_local`（`local`）/ `alloc_fragment`（`local.fragment`）/ `empty`（eager 输出张量） | 已核验（2026-10-01） |
| S7 | `tilelang/language/copy_op.py:59`、`:130` | `T.copy` 签名、docstring，以及它只产出一个 `tl.tileop.copy` intrinsic | 已核验（2026-10-01） |
| S8 | `tilelang/language/gemm_op.py:160`、`:166`、`:199-208` | `T.gemm` 签名（含 `policy`）、产出的 `tl.tileop.gemm` | 已核验（2026-10-01） |
| S9 | `tilelang/language/loop.py:13`、`:43-60`、`:82`、`:112`、`:123-133`、`:168-170`、`:189` | `T.Parallel`（含 `loop_layout` 与硬性约束）、`T.Persistent`、`T.Pipelined`（`num_stages` 语义、`stage`/`order` 使用建议）、`T.serial` | 已核验（2026-10-01） |
| S10 | `tilelang/language/annotations.py:30`、`:45` | `T.use_swizzle`、`T.annotate_layout`（含 `PartialFragment` / `Fragment` 断言） | 已核验（2026-10-01） |
| S11 | `tilelang/layout/__init__.py:4-5`、`tilelang/layout/swizzle.py:66` | `T.Layout` / `T.Fragment` 的再导出、`make_swizzled_layout` 签名 | 已核验（2026-10-01） |
| S12 | `tilelang/tileop/base.py:5-12` | `GemmWarpPolicy` 三个取值（`Square` / `FullRow` / `FullCol`）与缺省 `Square` | 已核验（2026-10-01） |
| S13 | `tilelang/language/eager/builder.py:1177-1196`；`tilelang/language/symbolics.py:13` | `T.const`（docstring 明确 eager-only；非 eager 上下文抛 `JITNoBuilderError`）、`T.dynamic` | 已核验（2026-10-01） |
| S14 | `tilelang/jit/__init__.py:277-312` | `@tilelang.jit` 的 lazy / eager 两种执行模式及其推断规则 | 已核验（2026-10-01） |
| S15 | `tilelang/transform/__init__.py:19-38` | `PipelinePlanning()`、`LayoutInference()` 两个 pass 的 Python 入口 | 已核验（2026-10-01） |
| S16 | `tilelang/cuda/pipeline.py:70`、`:127-142`、`:258-259` | `MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)`；`PipelinePlanning → InjectSoftwarePipeline → LayoutInference → LowerTileOp` 的顺序与注释；`ThreadSync("shared")` / `ThreadSync("shared.dyn")` | 已核验（2026-10-01） |
| S17 | `src/transform/layout_inference/layout_inference.cc:1-4` | LayoutInference 的职责说明（「infer the fragment/shared memory layout」） | 已核验（2026-10-01） |
| S18 | `tilelang/language/tir/ir.py:506`；`tilelang/language/ast/ir.py:2020`、`:2030`、`:2039`、`:2126` | `tvm_thread_allreduce`、`attr`、`evaluate`、`comm_reducer` 出现在语言层导出清单里 | 已核验（2026-10-01）：`tir/ir.py` 的 `__all__` 中可见 `"tvm_thread_allreduce"`；`ast/ir.py` 的 `__all__` 中可见 `"attr"` / `"evaluate"` / `"comm_reducer"`。**未导入运行**（本机装不上 tilelang，缺编译产物） |
| S19 | `tilelang/cuda/language/__init__.py:5`、`:59-65`、`:88`；`tilelang/cuda/language/gemm_op.py:19-29` | CUDA 方言的再导出顺序与「shadows a handful of common constructs」注释——`T.gemm` 等被 CUDA 版本遮蔽；`wgmma_gemm` / `tcgen05_gemm` / `tcgen05_gemm_blockscaled` / `mma_gemm_blockscaled` 的完整清单 | 已核验（2026-10-01） |
| S34 | `tilelang/language/tir/exports.py:7-23`、`tilelang/cuda/language/tir.py:555`、`tilelang/language/builtin.py:321` | `ptx_ldmatrix` 被登记为 **CUDA-only**（因此不在 `tilelang.language.common` 的 `__all__` 里），由 `tilelang/cuda/language/tir.py` 绑定并随 CUDA 方言导出；`tma_load` 的定义位置 | 已核验（2026-10-01）：三处均以 `grep -n` 定位并阅读。**未导入运行** |
| S20 | `tilelang/ascend/language/__init__.py:1-40` | 后端方言的存在性举例（昇腾方言在 common 之上加 `alloc_l1` / `alloc_l0a` / `alloc_l0c` / `dual_copy` 等） | 已核验（2026-10-01）；**未在 NPU 上运行** |

### 官方文档（锁定版本内）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S21 | `docs/get_started/overview.md:5`、`:17-30`、`:20`、`:74`、`:78-84` | 三层接口的定义与「mix these interfaces within the same kernel」；**beginner 层「not yet fully implemented」**；tile 的官方定义；显式存储分配与 Layout Inference Pass | 已核验（2026-10-01）：逐行阅读，行号与内容一致 |
| S22 | `docs/deeplearning_operators/matmul.md:17-21`、`:19`、`:41`、`:203`、`:214-218` | Level 1/2/3 与 TVM / Triton / CUDA 的类比；Level 1 的代价（auto-tuning 更久、复杂融合更吃力）；`T.gemm` 的底层实现；**混用不同层次**的官方表述 | 已核验（2026-10-01） |
| S23 | `docs/programming_guides/instructions.md:40-47` | `T.copy` 的同步语义与「compiler is free to lower it to different mechanisms」 | 已核验（2026-10-01） |
| S24 | `docs/programming_guides/software_pipeline.md:44-60` | `T.Pipelined` 的手工 `stage` / `order` 语义 | 已核验（2026-10-01） |
| S25 | `docs/programming_guides/language_basics.md:100-122` | `T.Kernel` 的启动注解在各方言下的差异；「`T.get_thread_binding()` … a kernel that uses it is rejected when compiled for a target without SIMT threads」 | 已核验（2026-10-01） |
| S26 | `docs/tools/layout_visualization.md:1-30` | 布局可视化两种用法的入口（`tilelang.tools.plot_layout` 与 layout inference 可视化） | 已核验（2026-10-01）；本章只在正文与示例 README 里作为「如何自行确认 swizzle 是否生效」的指引，未运行 |

### 官方示例（锁定版本内）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S27 | `examples/quickstart.py:18-46` | 正文 2.2 节引用的 tile 风格 GEMM + ReLU（developer 层） | 已核验（2026-10-01）：逐行比对；**未运行**（无 GPU） |
| S28 | `examples/gemv/example_gemv.py:14-42`、`:191-207` | 正文 3.2 节引用的 expert 层写法（`T.get_thread_binding` + `T.alloc_local` + `T.serial`）与落到 TIR 内建的线程归约 | 已核验（2026-10-01）；未运行 |
| S29 | `examples/gemm/example_gemm_intrinsics.py:15-17`、`:122-142` | 正文 3.3 节例子 B：`T.Parallel` / `T.Pipelined` 与 `mma_emitter.ldmatrix_a/b`、`mma`、`stmatrix` 同处一个 kernel；`:15-17` 的 `can_swizzle` 判据 | 已核验（2026-10-01）；未运行 |
| S30 | `examples/flash_attention/example_mha_fwd_bshd.py:29-92` | 正文 3.1 节引用的 developer 层 FlashAttention（`T.copy` / `T.gemm(policy=...)` / `T.reduce_*` / `T.Parallel` / `T.Pipelined`） | 已核验（2026-10-01）；未运行 |
| S31 | `examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py:196-263`（expert macro）、`:279-315`（developer macro）、`:336`、`:344-348`、`:366-373` | 正文 3.3 节例子 A 与图 3：同一个流水循环里两种层次的实现并存，并且四个调度维度都出现 | 已核验（2026-10-01）；未运行 |
| S32 | `examples/eager_jit/eagerjit.en.ipynb` | 正文 2.5 / 3.4 节：eager 前端的官方形态（`@tilelang.jit` + `T.const` / `T.dynamic` + `T.Tensor[[...]]`），以及「The function signature looks similar to Triton」的自述 | 已核验（2026-10-01）：用 `json` 读取 notebook 源码单元；未执行 notebook |
| S33 | `examples/visual_layout_inference/visual_layout_inference.py:1-25` | 布局可视化在真实 kernel 上的用法（`TL_LAYOUT_VISUALIZATION_ENABLE` / `..._FORMATS`） | 已核验（2026-10-01）；未运行 |

---

## corpus 资料引用（`../tilelang-corpus/`）

corpus 抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md` | https://arxiv.org/abs/2504.17577 | 本章的立论来源：论文摘要中「decouples scheduling space (thread binding, layout, tensorize and pipeline) from dataflow」与「customization annotations and primitives」两句，正文 2.1 节逐字引用 | 已核验（2026-10-01）：本地文件存在，内容为**摘要 + 元数据**（含作者、arXiv 号、v2 更新日期）；链接可达。**正文不在本地**，见「待补资料」 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md` | https://tilelang.com/get_started/overview.html | 三层接口的站点源文件版本；与 S21 内容一致，用于交叉确认「beginner 层未完成」不是本地仓库的偶发修改 | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md` | https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf ；DOI 10.1145/3315508.3329973 | 第 5 节与图 4：Triton 的 tile 定义（`:36-37`）、自动推导 schedule 的取舍（`:248-256`）、「hiding important performance details」三类细节（`:325-329`）、单线程模型与「CUDA-like concurrency primitives … are inexistent」（`:361-370`） | 已核验（2026-10-01）：本地为逐页文本提取，全文 1150 行 |
| R4 | `../tilelang-corpus/11-related-compilers/raw/triton-docs.md` | https://triton-lang.org/main/index.html | 第 5.1 节末尾：Triton 文档中确实存在更低层的 **Gluon** 编程模型（作为「这条路线也有边界」的佐证） | 已核验（2026-10-01） |
| R5 | `../tilelang-corpus/sources/annotated-bibliography.md:641-663` | （corpus 内注释书目） | 两条对照判断：「Triton 让编译器选调度，TileLang 把显式 tile / layout / 存储位置留在语言里」；以及「2019 版 Triton 比今天的语言小得多」的时间线提醒 | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/README.md:22-30` | （corpus 说明） | 对 TileLang 的概括引用（三个层次的接口、可混用、既能写 tile 原语也能下钻到 thread/硬件存储），用于确认本章的框架与 corpus 一致 | 已核验（2026-10-01） |

---

## 上游链接核验说明

* 上表标注「已核验」的链接，核验方式是确认对应 corpus 文件在本地存在、且其头部记录了
  抓取日期与原始 URL。**本章写作时没有重新发起网络请求**（corpus 抓取日期 2026-09-30，
  早于本章写作日期 2026-10-01，相差 1 天）。
* `R1` 的 arXiv 链接只用来确认论文标识（arXiv:2504.17577v2，2025-04-27 更新）与作者列表；
  本章引用它时只使用**摘要原文**，没有引用任何评测数字或正文图。
* `R3`（Triton MAPL 2019）在 corpus 里是本地 PDF 的逐页文本提取；引用时按本地文件的行号
  给出位置，不依赖网络。
* 本章**没有**引用任何需要登录才能访问的来源，也没有引用社区文章中的性能说法。

---

## 关于数字与「性能」的说明

本章**没有引入任何性能数字**，包括：

* 没有引用厂商或论文的加速比、延迟、吞吐（`docs/deeplearning_operators/matmul.md:228-236`
  里确实有一组内部 benchmark 数字，本章明确不采用——它属于第 14、28 章，且必须连同
  「internal benchmarks / 不同硬件 / 未公开脚本」这些条件一起呈现）；
* 没有比较四种调度变体的快慢（`code/03-schedule-as-annotation/` 只比较**正确性**）；
* 唯一出现的「计数类」数字是 `code/01-layer-inventory/` 的输出（47 个 API 名字、
  62 个源文件），它们由脚本本身产生，可用同一条命令复算。

---

## 待补资料

1. **TileLang 论文正文缺失。** corpus 里的 `arxiv-2504-17577.md` 只有摘要与元数据
   （742 字节级别），正文（尤其是第 4–6 节关于 layout inference、pipeline 与评测的部分）
   不在本地。本章因此只引用了摘要中可直接核对的两句话。**如果第 21、22 章需要论文级别的
   设计说明，建议后续版本补抓 arXiv HTML（`https://arxiv.org/abs/2504.17577v2` 的
   `ar5iv`/HTML 版本）或 PDF 全文。**
2. **`docs/compiler_internals/` 下没有 layout inference 的专页。**
   锁定版本的 `docs/compiler_internals/` 只有 4 篇（`letstmt_inline`、`inject_fence_proxy`、
   `tensor_checks`、`metal_tilelang_development`）。本章对 LayoutInference 的描述来自
   `docs/get_started/overview.md:82`、`tilelang/language/loop.py:43-60` 与 C++ 源文件头部注释；
   更细的推导规则需要读 `src/transform/layout_inference/`，那属于第 21 章。
3. **eager 前端的正式文档缺失。** 目前能核对到的只有 `tilelang/jit/__init__.py` 的
   docstring、`tilelang/language/eager/` 的源码与 `examples/eager_jit/*.ipynb`；
   没有 `docs/` 下的对应页面。本章因此在 2.5 / 3.4 节都写明「这是官方文档标注为未完成的
   方向」，并把话说在这个范围内。若上游补了文档，请以文档为准重新核对本节。
4. **Triton 的对照材料偏旧。** 第 5 节的主要依据是 2019 年的 MAPL 论文，它描述的语言
   比今天的 Triton 小得多（corpus 的注释书目也提醒了这一点）。corpus 里有
   `triton-docs.md`（站点首页）与 `github-triton-lang-triton.md`（仓库快照），但缺一份
   「今天的 Triton 语言/调度分工」的一手文档。本章因此**只做路线取向的对照**，
   不声称今天的 Triton 具体行为。第 5 章做横向对比时若需要更细的对照，
   建议补抓 `https://triton-lang.org/main/programming-guide/chapter-2/related-work.html`
   与 Gluon 的编程指南页。
