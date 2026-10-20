# 第 10 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有代码示例的最高验证级别是「仅静态核对」，唯一的例外是
`code/01-layout-algebra/layout_algebra.py`（纯标准库，已在本机实跑，属「CPU 逻辑验证」）。
**没有任何一条性能数字来自本机实测。**

---

## 一、锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

### 1.1 语言层（Python）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/layout/layout.py:12-43` | `T.Layout` 的定义与 `forward_fn` 语义（逻辑下标 → 物理下标） | 已核验（2026-10-01）：`grep -n "class Layout" …` → 12；完整阅读 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/layout/layout.py:93-226` | `get_linearized_forward_index`(93) / `map_forward_index`(97) / `repeat`(127) / `expand`(172) / `inverse`(217) | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:12-99` | `T.Fragment` 的构造签名（`forward_fn` / `forward_thread_fn` / `replicate` / `forward_index_fn`）与线程、槽语义 | 已核验（2026-10-01）：`class Fragment` 在 12、`__init__` 在 23 |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:117-186` | `repeat` / `replicate` / `map_forward_thread` | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/layout/swizzle.py:66-68` | `make_swizzled_layout(buffer, k_major=True, allow_pad=True)` 签名 | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/layout/swizzle.py:64-65` | 源码注释：「Swizzling should be enabled or disabled based on whether TMA is applied」 | 已核验（2026-10-01）：逐行阅读 |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/layout/swizzle.py:94-126` | `make_{quarter,half,full}_bank_swizzled_layout(buffer)` 三个签名 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/layout/swizzle.py:168-189` | `make_fully_replicated_layout_fragment`（replicate 的用途） | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/layout/ascend.py:20-92` | `make_ascend_nz_layout` / `major_k` / `major_mn` / `l0c` 的物理形状与正映射 | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:45-66` | `T.annotate_layout` 全文：dict 入参、回调包装、fragment/reducer 断言 | 已核验（2026-10-01）：完整阅读 |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:30-42` | `T.use_swizzle`：写入 `threadblock_swizzle_pattern`，`rasterization2DRow/Column/MLX` | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49-103,171` | `alloc_shared`(49, `shared.dyn`) / `alloc_local`(67, `local`) / `alloc_fragment`(81, `local.fragment`) / `alloc_var`(96-103, `local.var`) / `alloc_global`(171) | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:252,255` | `TL_LAYOUT_VISUALIZATION_ENABLE` / `TL_LAYOUT_VISUALIZATION_FORMATS` | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/tools/plot_layout.py:6-61` | `plot_layout(layout, save_directory, name, colormap, verbose, formats, view, grid_shape)` 签名与分发 | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/analysis/layout_visual.py:15-111` | 编译器侧 layout visualization 的实现（`print_fragment_format` 15、`_LayoutVisualVisitor` 47、`visit_sblock_` 88、`LayoutVisual` 107） | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/layout/mma_layout.py:336` | `make_mma_swizzle_layout(shared_buf, is_smooth=False)` | 已核验（2026-10-01）：另一条同源引用见 `tilelang/cuda/intrinsics/__init__.py:14` |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-70` | 昇腾的 `alloc_shared`（UB）/ `alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c`，以及 L0C 自动挂布局 | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532`、`tilelang/jit/__init__.py:507` | `get_kernel_source()` | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py:4-26` | 本章最小 GEMM 骨架的来源 | 已核验（2026-10-01）；**未运行**（无 GPU） |
| S20 | `_vendor/tilelang-v0.1.15/examples/plot_layout/layout_transform.py`、`fragment_mma_load_a.py` | 官方 `plot_layout` 示例 | 已核验（2026-10-01）：文件存在、逐行阅读；**未运行** |
| S21 | `_vendor/tilelang-v0.1.15/testing/python/layout/test_tilelang_bank_swizzle_expand.py:23-26` | `make_*_bank_swizzled_layout` 的**正确**调用形态（先 `decl_buffer`，再传 buffer） | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-v0.1.15/docs/tools/layout_visualization.md`（全文） | 两种观察手段、`plot_layout` 参数表、pass 配置键、限制 | 已核验（2026-10-01） |
| S23 | `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/matmul.md:60-105` | 官方 GEMM 教程里 `T.annotate_layout(...)` / `T.use_swizzle(...)` 被注释掉的默认位置 | 已核验（2026-10-01） |
| S24 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:120-160` | 昇腾侧 `T.alloc_shared`→UB、`alloc_l1`→L1、`alloc_l0a/b/c`→Cube 的对照，以及「与 Ascend C 的分工」表 | 已核验（2026-10-01）；**未在 NPU 上运行** |

### 1.2 编译期（C++）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S25 | `_vendor/tilelang-v0.1.15/src/layout/layout.h:47-155` | `LayoutNode`：`InputShape` / `OutputShape` / `Forward` / `Repeat` / `Expand` / `Inverse` / `MapRegionBounds`；顶点注释给出 NZ 的例子 | 已核验（2026-10-01） |
| S26 | `_vendor/tilelang-v0.1.15/src/layout/layout.h:171-269` | `FragmentNode`：`forward_thread`、`replicate_size`、`ThreadExtent` | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/src/layout/layout.h:25-41` | `LayoutConflictException` / `LoopLayoutInjectiveException` 的定义 | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:47-53` | `MakeGemmFragment8x8`（图 4 的公式来源） | 已核验（2026-10-01） |
| S29 | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:376-393` | `xor2x2` / `xor4x4` / `xor8x8` | 已核验（2026-10-01）；`code/01-layout-algebra` 逐行复刻 |
| S30 | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:442-530` | `Make{Quarter,Half,Full}BankSwizzleLayout2D`（含 `stride==4` 特例与 `ICHECK`） | 已核验（2026-10-01） |
| S31 | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:921-947` | `MakeSwizzledLayout`（按 `stride`/`continuous`/`element_size` 选粒度） | 已核验（2026-10-01） |
| S32 | `_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:966-1026` | `DetectSwizzleMode` / `MergeSwizzleLayouts`（冲突时取更小粒度） | 已核验（2026-10-01） |
| S33 | `_vendor/tilelang-v0.1.15/src/transform/layout_inference/layout_inference.cc:378-393` | reducer 的 `LayoutConflictException`（可被捕获） | 已核验（2026-10-01） |
| S34 | `_vendor/tilelang-v0.1.15/src/transform/layout_inference/layout_inference.cc:440-468` | `ProveFragmentContains` 路径、`MergeSwizzleLayouts` 调用、`LOG(FATAL) "Get different layout for …"` | 已核验（2026-10-01） |
| S35 | `_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1-6` | 「每个 kernel 只能有一份动态/静态 shared 分配」→ 合并与对齐 | 已核验（2026-10-01） |
| S36 | `_vendor/tilelang-v0.1.15/src/ascend/layout/ascend_layouts.h:1-70`（`C0` 的定义在 `:19,34-37`） | 昇腾 fractal 的 `C0`、四种 kind、物理形状与正映射 | 已核验（2026-10-01） |
| S37 | `_vendor/tilelang-v0.1.15/src/ascend/transform/{insert_nd2nz,insert_oob_padding,normalize_fractal_storage}.cc` | ND→NZ、OOB padding、fractal 归一化三个 pass 的实现文件 | 已核验（2026-10-01）：三个文件均存在；`AscendInsertOOBPadding` 的注册在 `insert_oob_padding.cc:515-532` |
| S38 | `_vendor/tilelang-v0.1.15/docs/programming_guides/instructions.md` | 「作用域/向量宽度不支持时编译失败，而不是悄悄降级」 | 已核验（2026-10-01） |
| S39 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:83-101` | 昇腾 pipeline 里的顺序：`AscendLayoutInference`(83) → `NormalizeAscendFractalStorage`(86) → `InsertNd2Nz`(91) → `AscendInsertOOBPadding`(101) | 已核验（2026-10-01）：逐行阅读；**未在 NPU 上运行** |

「已核验」的含义：在锁定版本的工作树上用 `grep -n` 定位到定义行，并阅读了完整的签名 /
代码段；行号区间与文件内容一致。**这不是运行验证。**

---

## 二、corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:1545-1554` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | bank 的定义：等宽模块、连续字到连续 bank、同 bank 串行化、广播例外；cc 5.x 起每 bank 每拍 32 bit、warp=32 | 已核验（2026-10-01）：本地为全文，逐行阅读 |
| R2 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:2622-2641` | 同 R1 | 经典例：32×32 float tile 按列写共享内存 → 每线程撞同一 bank；padding 到 33 后 `stride = w+1` 消除冲突；原文给出 V100 上的带宽前后值 | 已核验（2026-10-01）。**数字（12.8 / 140.2 / 199.4 GB/s）是原文在 Tesla V100 上的测量，本章只用于说明机制，未作为 TileLang 的性能引用** |
| R3 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md:50-51,226-228` | https://docs.nvidia.com/cuda/hopper-tuning-guide/ | H100（cc 9.0）：每 SM shared 228 KB、每 block 最多 227 KB、静态上限 48 KB、carveout 可选值 | 已核验（2026-10-01）：厂商文档口径，非实测 |
| R4 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md:50-51,93` | https://docs.nvidia.com/cuda/blackwell-tuning-guide/ | B200（cc 10.0）：每 SM 228 KB、每 block 最多 227 KB、CUDA 每 block 预留 1 KB | 已核验（2026-10-01）：厂商文档口径，非实测 |
| R5 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md` | https://tilelang.com/get_started/overview.html | `T.alloc_fragment` 对应寄存器文件、`T.alloc_*` 描述的是**整个线程块**的分配、`T.Fragment` 由 Layout Inference Pass 推导 | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/matmul.md` 与 `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/matmul.md:60-105` | https://tilelang.com/deeplearning_operators/matmul.html | Level 2 教程：`T.annotate_layout` / `T.use_swizzle` 的默认位置（被注释掉）、`T.alloc_fragment` 的说明 | 已核验（2026-10-01） |
| R7 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/layout_visualization.md` 与 `_vendor/tilelang-v0.1.15/docs/tools/layout_visualization.md` | https://tilelang.com/tools/layout_visualization.html | 两种观察手段、`plot_layout` 参数、编译器可视化的两个 pass config 键、限制 | 已核验（2026-10-01）：corpus 与 vendor 两份内容一致 |
| R8 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/deepseek_mla.md` | https://tilelang.com/deeplearning_operators/deepseek_mla.html | 「shared memory swizzling 把原本落在同一 bank 的地址重映射到不同 bank，可用 XOR 或其它位运算实现」；AMD 侧用不同 swizzle 策略 | 已核验（2026-10-01） |
| R9 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cutlass-media-docs-cute.md` | https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/… | CuTe 的 layout algebra（shape/stride、组合、线程划分）：TileLang 的 `repeat`/`expand` 解的是同一类问题 | 已核验（2026-10-01）：corpus 为文档快照 |
| R10 | `../tilelang-corpus/05-nvidia-backend/raw/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md` | 同 R9 | 「优化这类 kernel 常常就是找一个好的 shared memory layout，同时满足写入与读取的向量化并对齐 bank」 | 已核验（2026-10-01） |
| R11 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md` | https://arxiv.org/abs/2504.17577 | TileLang 论文（本地为摘要）：调度空间与数据流解耦、layout inference 的定位 | 已核验（2026-10-01）：本地仅摘要，未引用其中任何数字 |
| R12 | `../tilelang-corpus/90-synthesis/01-knowledge-system.md`（「五个核心概念」表） | （本书 corpus 内综合文档，无独立上游链接） | Layout / Storage scope 的定位，以及「layout 在编译器里的落点」 | 已核验（2026-10-01） |
| R13 | `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md`（§3.3、§4） | 同上 | `LayoutInference` 的职责；昇腾的 fractal 归一化 / ND 变换 / OOB padding 顺序（该文档写作 `InsertNd`，v0.1.15 里的实际 pass 名是 `InsertNd2Nz`，已按 S39 校正） | 已核验（2026-10-01） |
| R14 | `../tilelang-corpus/90-synthesis/03-ascend-route.md` | 同上 | 昇腾侧 fractal / ND→NZ / 边界填充作为该章主线的定位 | 已核验（2026-10-01） |

---

## 三、上游链接核验说明

* 上表标注「已核验」的链接，核验方式是：确认对应 corpus 文件在本地存在，且其头部记录了
  原始 URL 与抓取日期；本章写作时**没有重新发起网络请求**（corpus 抓取日期 2026-09-30，
  本章写作日期 2026-10-01）。
* R3 / R4 是 NVIDIA 厂商文档的公开规格，属**未实测口径**。本章只把它们当作「容量上限
  量级」引用，写代码时不应把具体数值当常数。
* R2 原文中的带宽数字（12.8 / 140.2 / 199.4 GB/s）来自 CUDA Best Practices Guide 在
  Tesla V100 上的测量。本章**只引用其机制结论**（同一列访问撞同一 bank、padding 消除），
  没有把它当成 TileLang 的性能数据，也没有在任何地方复用这些数字。

---

## 四、本章涉及的数字汇总（含口径）

| 数字 | 来源 | 测量 / 口径条件 |
| --- | --- | --- |
| 32 banks、每 bank 每拍 32 bit、warp=32 | R1 | cc 5.x 及以后，官方文档口径 |
| H100 每 SM shared 228 KB；每 block 最多 227 KB；静态 48 KB | R3 | 厂商文档，configurable |
| B200 每 SM shared 228 KB；每 block 最多 227 KB；CUDA 预留 1 KB | R4 | 厂商文档，configurable |
| `C0` = 32B / element_size（bf16/fp16 = 16、fp8 = 32、fp32 = 8） | `tilelang/layout/ascend.py:20-37`、`src/ascend/layout/ascend_layouts.h:19,34-37` | 源码口径 |
| 三种 swizzle 的粒度：32B/64B/128B、1/2/3-bit XOR、8 行周期 | `src/layout/gemm_layouts.cc:442-530` | 源码口径；`code/01-layout-algebra` 复算 |
| stride=32 → 32 路冲突；stride=33 → 无冲突 | **本书用 `code/01-layout-algebra/layout_algebra.py` 复算的排列结果** | 纯模型，非实测 |

**没有来源的、或来自厂商宣传口径的「性能提升百分比」一律未采用。** 所有需要真实加速器
才能得到的数字（实测吞吐、延迟、bank conflict 实测拍数）在本章中均未出现。

---

## 五、待补资料 / 待确认

1. **`examples/plot_layout/layout_swizzle.py` 与 v0.1.15 的库签名不一致（需上游确认）。**
   该示例调用 `make_full_bank_swizzled_layout(8, 64, element_size)` 等三个位置参数
   （`_vendor/tilelang-v0.1.15/examples/plot_layout/layout_swizzle.py:15-30`），而 v0.1.15
   的库签名是 `make_full_bank_swizzled_layout(buffer)`（只接一个 buffer，见
   `tilelang/layout/swizzle.py:94-102`），测试也用后者
   （`testing/python/layout/test_tilelang_bank_swizzle_expand.py:23-26`）。
   **本章没有运行该示例**（本机无 TileLang），因此只能报告「示例与库签名不一致」这一
   **静态观察**，不能断言示例跑不通。本机无法定论，建议后续版本补一条：在装好 TileLang 的
   机器上跑一次 `examples/plot_layout/layout_swizzle.py`，确认它是过期示例还是 API 迁移。
   本机 `code/02-plot-layout/plot_layout_demo.py` 按**库签名**书写。
2. **「layout inference 原理」的官方专页缺失。** `docs/compiler_internals/` 下只有 4 篇
   （`letstmt_inline`、`inject_fence_proxy`、`tensor_checks`、`metal_tilelang_development`），
   没有 layout inference 的独立原理页。本章的 4.1 节依据的是源码
   `src/transform/layout_inference/layout_inference.cc` 与
   `layout_cost_model.{h,cc}`。第 21 章需要更细的算法描述时，可能需要在 corpus 里补抓
   TileLang 论文正文（目前只有摘要）。
3. **昇腾侧的对齐 / 容量表未采用。** `_vendor/tilelang-ascend-v0.1.2.000/.agents/skills/.../
   api-kernel-memory.md` 里给了一张按 **Ascend 910B3** 平台列出的对齐与容量上限表
   （如 L0A/L0B 上限 65536 Byte、L0C 131072 Byte、L1 524288 Byte、UB 196608 Byte）。
   这是**适配器仓库自带的开发指引**，不是官方硬件规格书，且平台被限定为 910B3；本章因此
   **没有引用这些数值**，只引用了 v0.1.15 主干源码里可核对的 `scope` 与 fractal 定义。
   若第 19 / 21 章需要昇腾容量表，建议补一条华为官方规格来源。
