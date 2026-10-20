# 第 21 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述；凡涉及只在 `main` 上的东西都会显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有
昇腾 NPU**。因此本章所有 kernel / 工具示例的最高验证级别是「仅静态核对」
（API 存在性与签名逐个在 `_vendor/` 里比对过，语法用 `py_compile` 过了），
**没有任何一条性能数字或布局图来自本机实测**。

核验日期统一为 **2026-10-01**；核验方式见每行的「核验方式」列（`ls` / `rg -n` /
逐行阅读）。

---

## A. 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

### A.1 layout 代数与定义

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/layout/layout.py:12-43` | `T.Layout(shape, forward_fn)` 的定义：逻辑坐标 → 物理坐标 | 已核验（2026-10-01）：`rg -n "^class Layout"` → 12；逐行读构造函数 |
| S2 | `.../tilelang/layout/layout.py:93-95,97-125,127-170,172-215,217-226,228-237,254-266` | `get_linearized_forward_index` / `map_forward_index` / `repeat` / `expand` / `inverse` / `reshape` / `map_region_bounds` | 已核验（2026-10-01）：`rg -n "def "` 逐条比对行号 |
| S3 | `.../tilelang/layout/fragment.py:12-99,117-186` | `T.Fragment` = Layout + `forward_thread` + `forward_index` + `replicate`；`repeat` / `replicate` / `condense_rep_var` | 已核验（2026-10-01） |
| S4 | `.../src/layout/layout.h:41-190` | C++ 侧 `LayoutNode` 的方法集（`InputShape` / `OutputShape` / `Forward` / `GetLinearizedForwardIndex` / `Repeat` / `Expand` / `Inverse` / `Reshape`） | 已核验（2026-10-01） |
| S5 | `.../src/layout/layout.cc:1101-1160,1282-1285` | `InverseWithLevel`：静态形状走 `IterMapLevel::Bijective`，往返不一致抛 `NormalizeIterException` | 已核验（2026-10-01）：逐行读 |
| S5b | `.../src/layout/layout.cc:1118-1125` | 含符号维时退回 `IterMapLevel::NoCheck` 并发 `DLOG(WARNING)` | 已核验（2026-10-01） |
| S6 | `.../src/layout/layout.h:20-40` | `LayoutConflictException` / `LoopLayoutInjectiveException` 两个异常类型 | 已核验（2026-10-01） |
| S7 | `.../tilelang/layout/partial_fragment.py:12-95` | `PartialFragment`：reducer 的「加数」式复制（与「等值副本」不同） | 已核验（2026-10-01） |

### A.2 swizzle 与 GEMM 布局构造器

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S8 | `.../src/layout/gemm_layouts.cc:442-530` | 32B（quarter）/ 64B（half）/ 128B（full）三档 swizzle 的构造 | 已核验（2026-10-01） |
| S9 | `.../src/layout/gemm_layouts.cc:999-1023` | `MergeSwizzleLayouts`：两个 swizzle 冲突时取**更小粒度** | 已核验（2026-10-01） |
| S10 | `.../src/layout/utils.cc:443-480` | `ProveFragmentContains`：包含检查（含「全复制永远兼容」捷径） | 已核验（2026-10-01）；作者注释标 `@huanqicao` |
| S11 | `.../tilelang/layout/swizzle.py:66-126` | `make_swizzled_layout(buffer)` 与 `make_full/half/quarter_bank_swizzled_layout(buffer)` 的**buffer 版签名** | 已核验（2026-10-01）：签名为单参数 buffer（见本章「待核」） |
| S12 | `.../tilelang/cuda/intrinsics/macro/mma_macro_generator.py:684-810,814-886` | `make_mma_load_layout` / `make_mma_store_layout`：MMA 装载与累加器的 fragment 布局规则 | 已核验（2026-10-01） |
| S13 | `.../tilelang/cuda/intrinsics/layout/mma_layout.py:74-120`（含 `:117` 的别名定义） | `shared_16x8/16x16 → mma 32x4/32x8` 底座映射函数与 `..._sr_a` 别名 | 已核验（2026-10-01） |

### A.3 每个 tile op 的 InferLayout（约束生产者）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S14 | `.../src/op/operator.h:188` | `virtual LayoutMap InferLayout(const LayoutInferArgs&, InferLevel) const` 基类接口 | 已核验（2026-10-01） |
| S15 | `.../src/op/gemm.cc:235-268` | `GemmNode::InferLayout`：转交 `tl.gemm.infer_layout`；`reuse_existing_shared_layout` 让位逻辑 | 已核验（2026-10-01） |
| S16 | `.../tilelang/tileop/gemm/__init__.py:12-15,90-101` | Python 侧 `tl.gemm.infer_layout` 注册与 `Gemm.infer_layout` 分派 | 已核验（2026-10-01） |
| S17 | `.../tilelang/tileop/gemm/gemm_base.py:1-62` | `GemmBase` 与 `ss/sr/rs/rr/ts` 的分类 | 已核验（2026-10-01） |
| S18 | `.../tilelang/cuda/op/gemm/gemm_mma.py:42-66` | `GemmMMA.infer_layout`：swizzle + mma fragment 两种约束的源头 | 已核验（2026-10-01） |
| S19 | `.../src/op/parallel.cc:311-329,331-...` | `ParallelOpNode::InferLayout`：三级语义注释 + loop layout 推断 | 已核验（2026-10-01） |
| S20 | `.../src/op/copy.cc:553-564`、`fill.cc:209`、`reduce.cc:205`、`atomic_add.cc:126`、`transpose.cc:216`、`scan.cc:147,169` | 其余 tile op 的 `InferLayout` | 已核验（2026-10-01）：逐个 `rg -n "InferLayout"` |
| S20b | `.../tilelang/language/annotations.py:45-62` | `T.annotate_layout` 的契约（reducer 必须传 `PartialFragment`，fragment 必须传 `Fragment`） | 已核验（2026-10-01） |

### A.4 推断引擎与代价模型

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S21 | `.../src/transform/layout_inference/layout_inference.cc:137-142` | `LayoutInferenceResult`（layout_map / for_map / predicate_map / padding_guard_map） | 已核验（2026-10-01） |
| S22 | `.../src/transform/layout_inference/layout_inference.cc:243-518` | `RunInferStep`：调用 `InferLayout` 并落地/检查冲突 | 已核验（2026-10-01） |
| S23 | `.../src/transform/layout_inference/layout_inference.cc:520-536,538-690` | `FinishInferQueue` + `Run`：step 0–4 的完整流程 | 已核验（2026-10-01） |
| S24 | `.../src/transform/layout_inference/layout_inference.cc:762-777` | `EnqueueWithPriority`：工作列表优先级 | 已核验（2026-10-01） |
| S25 | `.../src/transform/layout_inference/layout_inference.cc:437-468` | 冲突处理：包含升级 / swizzle 合并 / `LOG(FATAL)`（fatal 在 `:463`） | 已核验（2026-10-01） |
| S26 | `.../src/transform/layout_inference/layout_inference.cc:1396-1550` | `InferInFreeMode`：并查集分量 + 逐根重推 + 代价择优 | 已核验（2026-10-01） |
| S27 | `.../src/transform/layout_inference/layout_inference.cc:1551-1658` | `LayoutInferencer` 与 pass 入口（结果写成注解） | 已核验（2026-10-01） |
| S27b | `.../src/transform/layout_inference/layout_inference.cc:312-360,621-661,1272-1315` | `propagate_alias`（alias 位宽换算与冲突 `ICHECK`）、step 4 的 alias 收尾、引擎成员声明 | 已核验（2026-10-01） |
| S28 | `.../src/transform/layout_inference/layout_cost_model.h:1-90` | `LayoutCostModel` 接口、`AttemptCost{mem,regs}` 与字典序 `BetterThan` | 已核验（2026-10-01） |
| S29 | `.../src/transform/layout_inference/layout_cost_model.cc:978-992,1001-1035,1180-1195` | `RegisterCountCostModel` / `IOAwareCostModel` / `Create()` 按名选择 | 已核验（2026-10-01） |
| S30 | `.../tilelang/transform/pass_config.py:122,252-256`、`.../src/config.h:36-40` | `tl.layout_cost_model` 与 `tl.layout_visualization_*` 的配置键与读取 | 已核验（2026-10-01） |
| S31 | `.../src/transform/layout_inference/parallel_loop_layout_validator.h` | pass 末尾的并行循环布局校验 | 已核验（2026-10-01）：文件存在，被 pass 入口调用 |

### A.5 管线位置

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S32 | `.../tilelang/cuda/pipeline.py:118-140` | CUDA 管线：`PipelinePlanning → InjectSoftwarePipeline → LayoutInference → ReducerPlanAndMaterialize → LayoutVisual → LowerTileOp` | 已核验（2026-10-01）：逐行读注释与调用 |
| S33 | `.../tilelang/ascend/pipeline.py:83-101` | 昇腾管线：`AscendLayoutInference → NormalizeAscendFractalStorage → InsertNd2Nz → VFChecker → AscendInsertOOBPadding` | 已核验（2026-10-01） |

### A.6 昇腾侧

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `.../src/ascend/layout/ascend_layouts.h:1-120` | 分形布局公式、C0 定义、`AscendFractalKind` 四种 kind、L0C 固定 16×16 | 已核验（2026-10-01） |
| A2 | `.../src/ascend/layout/ascend_layouts.cc:111-115,117-145,254-274` | `AscendC0`、MajorK 2D 映射、L0C 2D 映射 | 已核验（2026-10-01） |
| A3 | `.../tilelang/layout/ascend.py:20-115,186-206` | `make_ascend_nz_layout` / `l0c` / `major_k` / `major_mn` / `sf` 与 `try_extract_fractal_layout` | 已核验（2026-10-01） |
| A4 | `.../src/ascend/transform/layout_inference.cc:990-995`、`.../tilelang/ascend/transform/__init__.py:142-145` | `AscendLayoutInference`：与主干同构的昇腾分支；VF 区域对 worklist 不透明 | 已核验（2026-10-01）：读源码注释与 Python docstring |
| A5 | `.../tilelang/ascend/op/gemm/gemm_mad.py:119-140` | `GemmMAD.infer_layout`：按 `trans_A/trans_B` 在 MajorK/MajorMN 间选 | 已核验（2026-10-01） |
| A6 | `.../src/ascend/transform/normalize_fractal_storage.cc:1-6,68-80,485-492` | `NormalizeAscendFractalStorage`：统一 alias、吸收 reshape、加 padding（**不置零**） | 已核验（2026-10-01） |
| A7 | `.../src/ascend/transform/insert_nd2nz.cc:1-3,35-46,101-115,480-486` | `InsertNd2Nz`：UB→L1 copy 重写为 scatter+post_copy；仅支持 16/32 位浮点 | 已核验（2026-10-01） |
| A8 | `.../src/ascend/transform/insert_oob_padding.cc:1-30,328-...，515-527` | `AscendInsertOOBPadding`：clamp OOB 尾部 + 调度前补 fill | 已核验（2026-10-01） |
| A9 | `.../tilelang/ascend/transform/__init__.py:9-11,127-165` | 四个 pass 的 Python 入口 | 已核验（2026-10-01） |

### A.7 工具与示例

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S34 | `.../tilelang/tools/plot_layout.py:6-58,80-110,112-...` | `plot_layout` 的签名、格式保存、Fragment 画法 | 已核验（2026-10-01） |
| S35 | `.../tilelang/analysis/layout_visual.py:15-45,47-112` | 编译期 `LayoutVisual` 旁路与 `print_fragment_format` | 已核验（2026-10-01） |
| S36 | `.../docs/tools/layout_visualization.md` | 官方文档：两条工作流、`plot_layout` API、`TL_LAYOUT_VISUALIZATION_*`、限制 | 已核验（2026-10-01） |
| S37 | `.../examples/plot_layout/fragment_mma_load_a.py` | MMA 装载布局的三级放大示例 | 已核验（2026-10-01）；**未运行** |
| S38 | `.../examples/visual_layout_inference/visual_layout_inference.py` | 32×32×32 GEMM 的 layout visualization 示例（含期望输出） | 已核验（2026-10-01）；**未运行** |
| S39 | `.../examples/plot_layout/layout_swizzle.py` | swizzle 可视化示例——**其调用形式与 v0.1.15 的 buffer 版签名不一致**，见「待核」 | 已核验（2026-10-01）：比对后确认不一致；**未运行** |

---

## B. 资料库引用（`../tilelang-corpus/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| P1 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577-fulltext.md:70-76` | 论文：写高性能 kernel 的三类手工优化（Thread Binding / Memory Layout / Intrinsic Tensorization） | 已核验（2026-10-01）：`rg -n -i "layout"` + 逐行读 |
| P2 | `.../arxiv-2504-17577-fulltext.md:129,134,422-443,448-457` | 论文：Layout Inference 是自动并行化的核心；每个 Tile Operator 实现 `Lower` + `InferLayout`；`T.gemm` 的 `MakeSwizzleLayout` / `MakeMMASTMatrixLayout` | 已核验（2026-10-01） |
| P3 | `.../arxiv-2504-17577-fulltext.md:533-586` | 论文 §4.1：Layout 是 $\mathbb{K}^n\to\mathbb{K}^m$ 的映射、线性化地址 $\sum y_i s_i$、可复合、支持非双射 padding | 已核验（2026-10-01） |
| P4 | `.../arxiv-2504-17577-fulltext.md:587-608` | 论文：Fragment 值域固定为 $\mathbb{K}^2$；四个扩展原语；`m16k16 → m128k16` 组合示例（图 6） | 已核验（2026-10-01） |
| C1 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20250107-1.md:158-160` | 厂商文档：baseM/baseN 按 16 元素对齐、baseK 按 C0_size 对齐；half/bf16 C0=16、float C0=8、int8 C0=32、int4 C0=64 | 已核验（2026-10-01）：`rg -n "分形"` 定位；vendor 口径 |
| C2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/layout_visualization.md` | 官方文档在 corpus 中的镜像（与 S36 内容一致，用于交叉核对「官方文档确实这么写」） | 已核验（2026-10-01） |
| C3 | `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md:77-81,130,135,143` | 综合笔记：`LayoutInference` 的职责概览与「用 plot_layout / layout_visualization 看布局」的定位 | 已核验（2026-10-01）；综合笔记，非一手来源 |
| C4 | `../tilelang-corpus/06-huawei-ascend/SUMMARY.md` | 昇腾侧资料导览（Cube/AIC 与 Vector/AIV、UB/L1/L0A-L0C 的定位） | 已核验（2026-10-01） |

---

## C. 上游链接

| id | URL | 说明 | 核验状态 |
| --- | --- | --- | --- |
| U1 | https://github.com/tile-ai/tilelang | 上游仓库主页 | 未核验（本章不依赖其内容；仅作定位用） |
| U2 | https://arxiv.org/abs/2504.17577 | TileLang 论文 | 未核验（本章引用的是 corpus 内的 `arxiv-2504-17577-fulltext.md` 快照，非实时页面） |
| U3 | https://www.hiascend.com/developer/techArticles/20250107-1 | 昇腾 C0 对齐约束的技术文章 | 未核验（本章引用 corpus 镜像 C1） |

> 说明：本项目所在环境无法保证外部链接实时可达；凡未在本机核验的 URL 一律标
> 「未核验」，并以 corpus 快照或 `_vendor/` 源码为准。

---

## D. 待核事项（留给后续版本处理）

1. **swizzle 示例与签名不一致**（S39）。`_vendor/tilelang-v0.1.15/examples/plot_layout/layout_swizzle.py`
   以三参数形式调用 `make_quarter_bank_swizzled_layout(8, 16, element_size)` 等，
   而 v0.1.15 的包装函数只接受一个 buffer（S11）。二者矛盾，疑似示例未随 API 迁移。
   本章示例一律使用 buffer 版 API；建议后续版本向上游确认该示例的适用版本。
2. **`compose` 算子**（§2.4）。论文称布局变换「可复合」，但 v0.1.15 的
   `Layout`/`Fragment` 没有名为 `compose` 的公开方法。目前复合通过 `forward_fn`
   函数复合与 `IndexMap.map_indices` 实现。若后续版本新增显式 `compose`，本章需
   更新。
3. **「插入布局转换的代价」没有独立建模**（§3.4）。v0.1.15 的代价模型只在候选整解
   之间比 `{mem, regs}`，没有把「在某个 op 旁插入一次转换」建成独立选择边。这一点
   属于实现现状描述，需在跨版本复核时重新确认。
4. **昇腾 pass 顺序中的 `ReducerPlanAndMaterialize`**。它插在
   `AscendLayoutInference` 与 `NormalizeAscendFractalStorage` 之间
   （`tilelang/ascend/pipeline.py:83-86`），其具体作用超出本章范围，留待第 22/23 章
   说明；本章仅按位置引用。

## 资料缺口（本章无）

本章写作期间没有发现必须补充的 corpus 材料。若后续需要，请按
本书的约定在此登记「缺什么、为什么需要、
建议来源」。
