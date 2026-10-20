# 第 30 章 参考资料

本章的每条结论都必须能落到两类证据之一：**锁定版本的源码**（`_vendor/`）
或 **corpus 里的资料快照**（`../tilelang-corpus/`）。凡是两者都覆盖不到的，
正文一律写「无法确认」。

核对日期：**2026-10-01**（本章全部源码引用与命令输出都在这一天复核）。

## 0. 版本基线

| 组件 | 版本 / commit | 用途 |
| --- | --- | --- |
| TileLang（基线） | `v0.1.15` / `a35f8ddf45eba16c21211ec8822d56ce5363036f`（2026-09-30） | 一切「已具备 / 未具备」判断的基准 |
| TileLang（对照） | `v0.1.14` / `7e3bbf3703e769479757eb7ecec2f2626f3cd0a8`（2026-09-02） | 回答「是不是 v0.1.15 新增」 |
| TileLang main 快照（corpus 只读） | commit `994b44eca1a83a00d19d926e5264a6c608700656`，提交信息 `[Ascend] Add PTO backend for Ascend950 (#3310)` | 回答「仅 main」 |
| TileLang-Ascend（适配器） | `TileLang-ascend-v0.1.2.000-release` / `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`（2026-09-08） | A2/A3 路线，对照用 |

基线定义见 [VERSIONS.md](../../VERSIONS.md)。main 快照的路径是
`../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/`（`git log -1` 输出如上）。

---

## 1. 源码引用（`_vendor/tilelang-v0.1.15/`）

以下均已在本地逐条打开核对（核验状态：**已核验 2026-10-01**）。

### 1.1 多后端解耦（§3.1）

| 位置 | 内容 |
| --- | --- |
| `tilelang/backend/__init__.py:1-18` | 导出 `BackendModule` / `register_backend` / `get_backend` / `list_backends`、target 探测器与归一化注册 |
| `tilelang/backend/module.py:238,275,285` | `register_backend` / `get_backend` / `list_backends` 定义 |
| `tilelang/backend/target.py`、`tilelang/backend/device_codegen.py`、`tilelang/backend/host_codegen.py`、`tilelang/backend/pass_pipeline/` | 后端契约的四个面 |
| `tilelang/{cuda,ascend,rocm,metal,cpu,webgpu}/pipeline.py` | 每个后端自己的 PassPipeline |
| `tilelang/ascend/language/kernel.py:139-171` | 昇腾 `T.Kernel`：1-D 网格、无 `threads=` 参数、文档字符串明写 |
| `tilelang/ascend/language/frame.py:107,179` | `T.SimtVF` / `T.SimdVF` 定义 |
| `.agents/skills/tilelang-backend/SKILL.md` | 加/审后端的集成契约（对应 issue #2115 里的 PR #2900） |

### 1.2 TVM 依赖与 Z3（§3.2）

| 位置 | 内容 |
| --- | --- |
| `.gitmodules:4-6` | `3rdparty/tvm -> https://github.com/TileLang/tvm`（仍是分叉） |
| `CMakeLists.txt:401-402` | `set(USE_Z3 ON CACHE STRING ...)`、`set(USE_PYPI_Z3 ON ...)` |
| `src/transform/verify_parallel_loop.cc:106` | `analyzer.z3_prover.GetModel(race_free)`（prover 挂在 TVM 的 `arith::Analyzer` 上） |
| `tilelang/engine/lower.py:9` | `from tvm import tirx`（分叉的 `tirx` 命名空间） |
| `src/transform/vendored/z3_prover.cc` | **不存在**（RFC #2196 下游方案，v0.1.14 里也不存在） |

### 1.3 CuteDSL 后端（§3.3）

| 位置 | 内容 |
| --- | --- |
| `tilelang/cuda/cutedsl_backend.py:1-22` | 独立 `BackendModule`，`pipelines={"cuda": pipeline.CUDA_PIPELINE}` |
| `tilelang/cuda/codegen.py:8-13` | `is_cutedsl_target` / `is_plain_cuda_target` |
| `src/cuda/codegen/codegen_cutedsl.cc`、`src/cuda/codegen/rt_mod_cutedsl.cc` | device codegen 与 module 注册 |
| `tilelang/jit/adapter/cutedsl/{adapter,libgen,wrapper,checks}.py` | JIT 通道与依赖门禁 |
| `tilelang/contrib/cutedsl/{gemm_v1,gemm_v2,gemm_tcgen05,cpasync,ldsm,ptx_mma,cluster,atomic,reduce,...}.py` | 组件库 |
| `docs/get_started/targets.md` | `cutedsl` 被列为常见 target，注明需 `nvidia-cutlass-dsl` |
| `examples/conftest.py:39-46` | `CUTEDSL_KNOWN_FAILURES`（2 条，`xfail(strict=False)`） |

### 1.4 昇腾 950 后端与相关管线（§4.1–§4.3）

| 位置 | 内容 |
| --- | --- |
| `tilelang/ascend/README.md` | 标题为 `# TileLang Ascend 950 Backend` |
| `tilelang/ascend/language/` | 方言：`allocate.py`、`copy_op.py`、`frame.py`、`gemm_op.py`、`kernel.py`、`loop.py`、`simd.py`、`sync.py`、`tile_schedule.py`… |
| `src/ascend/codegen/{codegen_ascend.cc,codegen_ascend.h,intrin_rule_ascend.cc,rt_mod_ascend.cc}` | AscendC 路线的 codegen |
| `src/ascend/transform/` | `auto_schedule/`、`insert_sync.cc`、`insert_nd2nz.cc`、`insert_oob_padding.cc`、`materialize_multi_buffer.cc`、`rewrite_dual_copy.cc`、`unroll_loop_skip_vf.cc`… |
| `tilelang/ascend/pipeline.py:74-75` | `CanonicalizeLegacyReducer` / `VerifyReducerEpoch` |
| `tilelang/ascend/pipeline.py:76,80` | `VerifyBufferInit` / `UnrollLoopSkipVF`（PR #3320 要插在这两者之间） |
| `tilelang/ascend/pipeline.py:106-114` | AutoSchedule 一段：`MaterializeScheduleUnits` → `AutoSchedule` → `InsertSync` → `MaterializeMultiBuffer` |
| `tilelang/contrib/bisheng.py:93` | 默认 `npu_arch = "dav-3510"` |
| `tilelang/ascend/transform/__init__.py:77,112,132` | `AutoSchedule` / `RewriteDualCopy` / `AscendSimdVFLowerParallel` 的 Python 包装 |
| `tilelang/ascend/language/tile_schedule.py` | 现有的 persistent GEMM 调度器（对应手写 `T.Persistent`） |
| `tilelang/language/loop.py:82` | `T.Persistent`（v0.1.15 里 persistent 的唯一入口） |
| `src/backend/common/` | `rg -n 'TargetIsAscend' src/backend/common/` **无输出**（后端特例已收进后端目录） |

### 1.5 reducer（§5.1）

| 位置 | 内容 |
| --- | --- |
| `tilelang/language/allocate.py:296` | `alloc_reducer` 定义；`replication` 标注为 deprecated |
| `tilelang/language/allocate.py:336-346` | legacy 分支与 `DeprecationWarning`（含迁移提示文案） |
| `tilelang/language/reduce_op.py:358,398,434` | `reducer_init` / `reducer_update` / `finalize_reducer` |
| `src/transform/verify_reducer_epoch.cc:1-30,326` | epoch 状态机校验的规则注释与 pass 注册 |
| `src/transform/canonicalize_legacy_reducer.cc:1-45,458` | legacy → v2 的白名单规范化与 pass 注册 |
| `src/transform/reducer_plan_materialize.cc` | 物理计划物化 |
| `tilelang/cuda/pipeline.py:91-92` | CUDA pipeline 里的两步挂载 |
| `examples/gemv/example_gemv.py:239,250` | v0.1.15 自带的 GEMV 例子**仍在使用 legacy 写法** |
| `_vendor/tilelang-v0.1.14/tilelang/language/reduce_op.py:364,404,440` | 同样的 v2 API 在 v0.1.14 已存在（回答「是不是本次新增」） |

### 1.6 atomic 内存序与 cache policy（§5.2–§5.3）

| 位置 | 内容 |
| --- | --- |
| `src/tl_templates/cuda/atomic.h:472,508,517` | `AtomicAdd` / `AtomicAddRet`：接受并使用 `memory_order` |
| `src/tl_templates/cuda/atomic.h:308-336` | `AtomicAddx2Scalar`(:310) / `AtomicAddx4Scalar`(:316) / `AtomicAddx2ScalarRet`(:323) / `AtomicAddx4ScalarRet`(:331)：**不带** `memory_order`，逐个调 `atomicAdd` |
| `src/tl_templates/cuda/atomic.h:85-95` | `IsAcquireLikeMemoryOrder` 把 `consume` 与 `acquire` 等价处理 |
| `tilelang/language/atomic.py:39,125,211,390,436` | `atomic_max` / `atomic_min` / `atomic_add` / `atomic_load` / `atomic_store`；文档字符串说明 HIP 与 CuteDSL 忽略 `memory_order` |
| `tilelang/language/copy_op.py:140,178-179` | 既有的 `eviction_policy`（TMA L2 淘汰优先级），不是 PTX cache operator |
| `tilelang/language/copy_op.py`（全文） | `load_cache_policy` / `store_cache_policy` **不存在** |

### 1.7 profiling 与其他（§4.5）

| 位置 | 内容 |
| --- | --- |
| `tilelang/profiler/msprof.py:14,40` | `get_msprof_cache_flush_kernel` 与 `bench_with_msprof`；无 host/device 关联结构 |
| `tilelang/profiler/torch_bench.py:136,138` | NVIDIA 侧用 CUDA event 的 `self_device_time_total` 统计单事件设备时间；这是**每事件**的耗时统计，不是 PR #3321 说的 host/device 关联结构（因此判定该 PR 未落地时，只以昇腾侧的 `msprof.py` 为准） |
| `tilelang/ascend/language/copy_op.py:139-151` | L0C→UB 的 dtype 不同即 `raise ValueError`（对应 PR #3328 的现状） |
| `examples/ascend/` | 共 13 项，无 causal sparse attention 例子 |
| `.agents/skills/` | v0.1.15 有 9 个 skill，含 `tilelang-ascend`；**没有** `tilelang-ascend-op` |
| `agent/core/` | **不存在**（PR #3329 的工作流目录） |

---

## 2. main 快照与 tag 的差异（§4.1.2）

`../tilelang-corpus/02-tilelang-core-repo/raw/tilelang`（`994b44e`）相对
`_vendor/tilelang-v0.1.15`（`a35f8ddf`）的差异，经文件级比较为
**新增 20 个文件、搬迁 2 个文件**，全部来自 PR #3310：

```text
新增（20）
  src/ascend/codegen/codegen_pto.cc
  src/ascend/codegen/codegen_pto.h
  src/ascend/codegen/rt_mod_pto.cc
  src/ascend/codegen/sfu_precision.h
  src/backend/common/codegen/codegen_py.cc        (自 src/cuda/codegen/ 搬入)
  src/backend/common/codegen/codegen_py.h         (自 src/cuda/codegen/ 搬入)
  tilelang/contrib/ptodsl/{__init__,common,dcache_bypass,gemm,mixed_kernel,rng,simd_inst,simt,sync}.py
  tilelang/jit/adapter/pto/{__init__.py,adapter.py,cython_wrapper.pyx,libgen.py,wrapper.py}

只在 tag 侧（2，属于搬迁的另一半）
  src/cuda/codegen/codegen_py.cc
  src/cuda/codegen/codegen_py.h
```

核验命令见 §4 的 `tree_diff --selftest`。

---

## 3. corpus 资料引用

以下路径均相对 `../tilelang-corpus/`（核验状态：**已核验 2026-10-01**）。

### 3.1 前沿与未合入话题（`08-frontier-unmerged/raw/`）

| 文件 | 用在本章哪里 |
| --- | --- |
| `github-issue-2115.md` | §3.1 多后端解耦的动机原文与 PR 清单 |
| `github-issue-2196.md` | §3.2 TVM 迁移 RFC：仓库考古、Z3 迁移、PR 序列 |
| `github-issue-1454.md` | §3.3 CuteDSL roadmap 条目 |
| `github-issue-56.md`、`github-issue-79.md` | §2.4「roadmap 不等于待办」的旁证（非 NV 后端路线、0.2.0 发布计划） |
| `github-issue-2708.md` | §5.2 atomic 内存序 RFC 的问题陈述与验收标准 |
| `github-issue-2897.md` | §5.1 reducer RFC：动机、最小错码例子、v2 API、四个阶段与迁移计划 |
| `github-issue-3178.md` | §5.3 cache policy RFC 的提议 API 与非目标 |
| `github-pr-3308.md` | §4.1.1 950 后端的能力清单、硬件边界、验证条件与改动历史 |
| `github-pr-3310.md` | §4.1.2 PTO 后端的摘要；同时是「页面状态与代码快照矛盾」的证据 |
| `github-pr-3325.md` | §4.2 `T.PersistentKernel` / `AutoPersistent`：背景、映射公式、性能表 |
| `github-pr-3320.md` | §4.3 `AutoSimtVF`：Stage 1 的四个模块与边界 |
| `github-pr-3329.md` | §4.4 算子生成 agent：工作流、8 个 skill、安装方式 |
| `github-issue-3317.md` | §4.1.3「两条 950 路线怎么选」：提问存在、**维护者回答缺失** |

### 3.2 合成与索引

| 文件 | 用在本章哪里 |
| --- | --- |
| `90-synthesis/03-ascend-route.md` §7 | §4.5 四条 PR（#3328/#3322/#3324/#3321）的**唯一**线索来源；本章据此做反向验证 |
| `08-frontier-unmerged/raw/_index.md` | 该目录 14 条资料的清单与收集动机（说明目录名不等于落地状态） |
| `02-tilelang-core-repo/raw/tilelang/` | main 快照本体（`994b44e`），§4.1.2 与 §9 的「仅 main」判定 |
| `12-operator-libraries/raw/_index.md`、`PROVENANCE.md` | §6 各生态仓库快照的抓取日期与 commit |

### 3.3 生态（§6）

| 文件 | 关键内容（原文口径） |
| --- | --- |
| `12-operator-libraries/raw/github-tile-ai-tileops.md` | 「Spec-driven LLM operators across backends — built by agents」；191 stars；最后 push 2026-09-30 |
| `12-operator-libraries/raw/github-tile-ai-tilert.md` | 「Tile-Based Runtime for Ultra-Low-Latency LLM Inference」；1830 stars；最后 push 2026-09-24 |
| `12-operator-libraries/raw/github-tile-ai-tilesight.md` | 解析性能模型，可在 CPU 上跑；10 stars；2026-08-18 创建 |
| `12-operator-libraries/raw/github-tile-ai-tilelang-benchmark.md` | README 首行即「Archived, please checkout TileOPs」 |
| `07-tilelang-ascend-adapter/raw/github-tile-ai-tilefoundry.md` | 「Hand the compiler to the agent」；121 stars；最后 push 2026-09-29；PyPI 0.0.2；News 里的四个 agent 实例与吞吐数字 |
| `07-tilelang-ascend-adapter/raw/github-tile-ai-tilelang-mlir-ascend.md` | §6 地图里的 AscendNPU IR 路线仓库 |

### 3.4 NVIDIA 侧对照（§7）

| 文件 | 关键内容 |
| --- | --- |
| `11-related-compilers/raw/github-nvidia-cutile-python.md` | cuTile 的示例 kernel、Tile IR、驱动 r580+、`tileiras` 13.2 只支持 Blackwell 与 Ampere/Ada |
| `11-related-compilers/raw/nvidia-cutedsl-docs.md` | CuTe DSL 文档（来自 CUTLASS 仓库的 `media/docs/pythonDSL/` 源文件） |

> 注意：`nvidia-cutedsl-docs.md` 是该文档站的**壳**（corpus 说明里写明「本机只能抓到
> 导航壳（1.3 KB）」这一类情况），本章关于 CuTe DSL 的判断均来自 TileLang 侧代码
> （§1.3 的表），不来自这份文档正文。该缺口登记在 §5「待补资料」。

---

## 4. 验证命令（逐条判定，可直接复制执行）

以下命令都在 `.workspace/research/tilelang-book/` 下执行；`<root>` 指本书根目录。

### 4.1 总体自检

```bash
python3 chapters/30-frontier/code/01-state-audit/state_audit.py            # self-check: ok
python3 chapters/30-frontier/code/02-tree-diff/tree_diff.py --selftest     # selftest: ok
python3 chapters/30-frontier/code/03-upstream-watch/upstream_watch.py --run
```

### 4.2 逐条验证（与 §9 的总表一一对应）

```bash
# —— 多后端解耦 #2115（A 档）——
rg -n 'def register_backend' _vendor/tilelang-v0.1.15/tilelang/backend/module.py
ls _vendor/tilelang-v0.1.15/tilelang/{cuda,ascend,rocm,metal,cpu,webgpu}/pipeline.py

# —— TVM 迁移 #2196（未落地）——
cat _vendor/tilelang-v0.1.15/.gitmodules
rg -n 'set\(USE_Z3' _vendor/tilelang-v0.1.15/CMakeLists.txt
rg -n 'z3_prover' _vendor/tilelang-v0.1.15/src/transform/verify_parallel_loop.cc
ls _vendor/tilelang-v0.1.15/src/transform/vendored            # No such file or directory

# —— CuteDSL 后端 #1454（A 档，实验通道）——
sed -n '1,22p' _vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py
sed -n '39,45p' _vendor/tilelang-v0.1.15/examples/conftest.py

# —— 昇腾 950 AscendC 后端 #3308（A 档）——
ls _vendor/tilelang-v0.1.15/tilelang/ascend/language
ls _vendor/tilelang-v0.1.15/src/ascend/codegen
rg -n 'dav-3510' _vendor/tilelang-v0.1.15/tilelang/contrib/bisheng.py
rg -n 'TargetIsAscend' _vendor/tilelang-v0.1.15/src/backend/common      # 无输出
ls _vendor/tilelang-v0.1.14/tilelang/ascend 2>/dev/null                 # v0.1.14 里不存在

# —— 昇腾 950 PTO 后端 #3310（仅 main）——
rg -n 'tilelang_pto|"pto"' _vendor/tilelang-v0.1.15/tilelang/ _vendor/tilelang-v0.1.15/src/   # 无输出
ls _vendor/tilelang-v0.1.15/src/ascend/codegen/codegen_pto.cc 2>/dev/null                    # 不存在
git -C ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang log -1 --oneline
python3 chapters/30-frontier/code/02-tree-diff/tree_diff.py --selftest

# —— T.PersistentKernel #3325 / AutoSimtVF #3320 / 算子 agent #3329（均未合入）——
rg -n 'PersistentKernel|AutoPersistent' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
rg -n 'AutoSimtVF' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
ls _vendor/tilelang-v0.1.15/agent/core 2>/dev/null
ls _vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend-op 2>/dev/null
rg -n 'def Persistent' _vendor/tilelang-v0.1.15/tilelang/language/loop.py       # v0.1.15 只有手写版

# —— #3328 / #3322 / #3324 / #3321（反向验证）——
sed -n '139,151p' _vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py
ls _vendor/tilelang-v0.1.15/examples/ascend/
rg -n 'task_queue' _vendor/tilelang-v0.1.15/tilelang/ascend/
rg -n -i 'correlat|host_time' _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py   # 无输出

# —— reducer RFC #2897（主体已发布）——
rg -n 'def reducer_init|def reducer_update|def finalize_reducer' _vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py
rg -n 'def reducer_update' _vendor/tilelang-v0.1.14/tilelang/language/reduce_op.py   # v0.1.14 已有
ls _vendor/tilelang-v0.1.15/src/transform/{verify_reducer_epoch,canonicalize_legacy_reducer}.cc
rg -n 'CanonicalizeLegacyReducer|VerifyReducerEpoch' _vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py
sed -n '336,346p' _vendor/tilelang-v0.1.15/tilelang/language/allocate.py     # legacy 的弃用提示

# —— atomic 内存序 #2708（未落地）——
sed -n '308,337p' _vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h
sed -n '472,520p' _vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h
diff -q _vendor/tilelang-v0.1.14/src/tl_templates/cuda/atomic.h \
        _vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h              # 无差异

# —— cache policy #3178（未落地）——
rg -n 'load_cache_policy|store_cache_policy' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src  # 无输出
rg -n 'eviction_policy' _vendor/tilelang-v0.1.15/tilelang/language/copy_op.py                               # 既有能力
```

### 4.3 主快照与 tag 的差异（文件级）

```bash
diff <(cd _vendor/tilelang-v0.1.15 && find . -path ./3rdparty -prune -o -path ./.git -prune -o -type f -print | sort) \
     <(cd ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang && find . -path ./3rdparty -prune -o -path ./.git -prune -o -type f -print | sort) \
     | grep '^<' | head -30
```

---

## 5. 待补资料

以下缺口影响本章的**结论强度**（不影响已有的结论正确性）：

1. **PR #3321 / #3322 / #3324 / #3328 的页面快照**。corpus 只有
   `90-synthesis/03-ascend-route.md` 的一句转述。没有快照，本章只能做反向验证，
   结论只能写到「未在 v0.1.15 / main 快照中找到；PR 细节无法确认」。
   建议来源：GitHub 上这四个 PR 的页面（标题、状态、改动文件）。
2. **PR #3310 的状态与时间线**。页面快照显示 draft，main 源码快照里却有合并 commit；
   两者矛盾。若需要精确时间线，应补一份带时间戳的 PR 事件记录（opened / converted
   to draft / ready for review / merged）。
3. **`nvidia-cutedsl-docs.md` 的正文**。当前抓到的只是文档站导航壳，
   本章关于 CuTe DSL 的判断全部来自 TileLang 侧代码。若要在第 24 章之外
   深入对比 CuTe DSL 的官方语义，需要重新抓取渲染后的文档正文。
4. **TileFoundry 的仓库快照**。本章只用到了它的 README（位于
   `07-tilelang-ascend-adapter/raw/github-tile-ai-tilefoundry.md`），没有克隆源码，
   因此不评价其实现方式，只引用其自述定位。
5. **`tile-ai/tilelang-ascend` 与主干的关系时间线**。本章只核对了适配器仓库的 tag
   commit 与 README 定位，没有逐条比对它与主干 `tilelang/ascend/` 的 API 差异
   （第 25 章 §6.1 已给出「两棵树的 API 不通用」的核对结论）。

---

## 6. 与相邻章节的一致性说明

本章与第 24、25 章在以下三处有交叉，写法上的差异是**有意的**，不是冲突：

| 主题 | 第 24 / 25 章的写法 | 本章的写法 | 说明 |
| --- | --- | --- | --- |
| PTO 后端 | 第 25 章 §4.10 依据 PR **页面快照**把 #3310 记为「draft（未合入）」 | 本章依据 PR 页面快照 **与 corpus 的 main 源码快照**，判定为「仅 main」 | 本章拆开了两个证据：PR 评审状态无法确认、代码状态可确认。两章结论在「v0.1.15 里没有 `target="pto"`」这一点上完全一致 |
| CuTe DSL | 第 24 章 §4.8 讲机制（codegen、adapter、依赖门禁） | 本章只讲状态与走向（属 A 档、仍是实验通道） | 机制不复述，避免两章重复 |
| 多后端结构 | 第 4 章讲注册表结构本身 | 本章讲它在 issue #2115 里的来龙去脉与当前状态 | 依赖第 4 章的心智模型 |

`references.md` 中所有 `_vendor/` 引用与 corpus 引用的核验状态均为
**已核验（2026-10-01）**；上游 URL 的可达性未在本机联网核验，
因此本章只使用 corpus 快照里已经存在的 URL，不使用其它外部链接。
