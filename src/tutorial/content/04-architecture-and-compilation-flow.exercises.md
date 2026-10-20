# 第 04 章 习题

说明：

* 难度：`★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链较长或需要动手设计）。
* “需要硬件”：`否` 表示只需要一台能跑 Python 的机器（本章绝大多数题目都是这样）；
  `是（GPU）` / `是（NPU）` 表示必须用真实加速器才能完成实验，这类题都标为**可选**。
* 所有题目只依赖第 1–4 章的知识。凡涉及后续章节的地方会显式提示。
* 题中出现的所有行号都以 `_vendor/tilelang-v0.1.15/` 为准；用
  `sed -n 'A,Bp' <file>` 可以现场核对。

---

## E1 概念题 ★ 不需要硬件

本章 2.1 节给出了一次编译的五个阶段（前端 → 编译期 → 代码生成 → 宿主封装 → 运行时）
以及每个阶段的入口函数。

1. 写出这五个阶段的名称，以及第 ② 阶段的入口函数全名。
2. 为什么说“后端差异被压缩进了第 ② 阶段的一个函数调用里”？请指出那个调用的
   文件与行号，并说明其余四个阶段为什么是共享的。
3. “前端共享”是否意味着“IR 也共享”？请举出本章给出的一个反例。

---

## E2 概念题 ★ 不需要硬件

1. 用自己的话区分 **target backend** 与 **execution backend**，各写出它的职责清单。
2. 从 v0.1.15 中各举两个例子，并给出源码位置。
3. 判断下列说法正误，并说明理由：
   * “选择 `nvrtc` 执行后端，意味着走的是另一条 CUDA pass pipeline。”
   * “CuTeDSL 与 CUDA 是两个不同的 target backend，因此它们的 IR lowering 也不同。”

---

## E3 概念题 ★★ 不需要硬件

阅读 `tilelang/backend/module.py:42-92`（`BackendModule.__post_init__` 的校验）
与 `:238-272`（`register_backend`）。

1. 列出 `__post_init__` 至少四条硬校验，并说明每条想防住什么错误。
2. 为什么“同一个 target kind 上允许多个 manifest”时，必须要求这些 manifest 都提供
   `supports_target` 谓词？如果不加这条约束，`_resolve_backend_module`（`:300-307`）
   会发生什么？
3. `register_backend` 在注册 FFI callback 失败时会做回滚（`:257-270`）。
   如果去掉回滚，会发生什么后果？

---

## E4 代码阅读题 ★★ 不需要硬件

下面是 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:120-144` 的节选（注释已删）：

```python
mod = tilelang.transform.IfStmtBinding()(mod)
mod = tilelang.transform.UnrollLoop()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.LayoutInference()(mod)
mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
LayoutVisual(mod)
mod = tilelang.transform.LowerTileOp()(mod)
```

1. 结合源码注释回答：为什么 `PipelinePlanning` / `InjectSoftwarePipeline` 必须排在
   `LayoutInference` **之前**？如果交换顺序，layout 推导会看到什么样的 IR？
2. `LayoutVisual(mod)` 为什么不需要重新赋值 `mod`？（提示：看
   `tilelang/backend/pass_pipeline/pipeline_utils.py:105-109`。）
3. `LowerTileOp` 之后，`T.gemm` / `T.copy` 还存在吗？这一步对调试有什么意义？

---

## E5 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:103-118` 与 `:169-170`：

```python
if auto_schedule_enabled:
    mod = ascend_transform.NormalizeControlFlowForSchedule()(mod)
    ...
    mod = ascend_transform.AutoSchedule()(mod)
    ...
    mod = ascend_transform.RestoreWhileLoops()(mod)

mod = tilelang.transform.Simplify()(mod)
...
if not auto_schedule_enabled:
    mod = ascend_transform.InferBufferAliases()(mod)
```

1. 这 13 步自动调度分别在解决什么问题？请至少说出“任务化”“资源建模”“求解”“落地”
   四个词各对应哪几步。
2. `AutoSchedule` 之前的 `EstimateLatency` 和 `AnnotateMultiBufferEligible` 为什么必须
   排在求解之前？
3. 第 72 步（`:169-170`）为什么写成 `if not auto_schedule_enabled:`？这说明
   “关掉自动调度”对编译器意味着什么？

---

## E6 代码阅读题 ★★ 不需要硬件

`_vendor/tilelang-v0.1.15/tilelang/language/__init__.py` 的开头是：

```python
from tilelang.cuda.language import *  # noqa: F401,F403
from tilelang.cuda.language import __all__ as __all__
from tilelang.cuda.language import (
    Kernel, Parallel, Unroll, atomic_add, copy, gemm, ...
)
__tilelang_dialect__ = "cuda"
```

1. 这三段代码各自解决什么问题？（提示：`__all__` 与静态类型检查器。）
2. 如果一份昇腾 kernel 文件里写了 `import tilelang.language as T`，
   会发生什么？请给出正确写法。
3. 对比 `tilelang/language/kernel.py:431` 与 `tilelang/cuda/language/kernel.py:13`
   两个 `Kernel` 的签名，说出它们的关系。`ClusterKernel` 为什么放在 CUDA 那一侧？

---

## E7 动手题 ★ 不需要硬件

运行本章的离线脚本，回答下面的问题（全部答案都能从输出里直接读出来）：

```bash
cd chapters/04-architecture-and-compilation-flow/code
python3 pass-order/dump_pass_order.py
```

1. CUDA pipeline 一共多少步？其中 `CUDAPassPipelineBodyPrologue` 贡献多少步？
2. Ascend pipeline 一共多少步？
3. 两侧同名的 pass 有多少个？至少列出 10 个你认识的。
4. 带条件守卫的 pass，CUDA 有几个、Ascend 有几个？把 CUDA 的 7 个全部列出来，
   并写出各自的条件表达式。

---

## E8 动手题 ★★ 不需要硬件

1. 运行 `python3 repo-map/inventory.py`，把
   `tilelang_top_level_dirs` / `tilelang_python_files` / `src_transform_cc_files` /
   `examples_subdirs` / `examples_top_level_files` / `docs_markdown_files` 六个数字抄下来。
2. `VERSIONS.md` 写“`examples/` 有 57 个目录”，脚本给出的是 54 个目录 + 3 个文件。
   请解释这两个口径的差异，并说明章为什么采用脚本口径。
3. 运行 `python3 compile-path-check/check_compile_path.py`。如果有跳失败，
   用 `sed -n '<行号>p' <文件>` 现场核对并给出你的判断（是章节写错了，还是
   vendor 目录版本不对）。把脚本改成 1 行，让它打印 `Hop` 的数量（提示：`len(HOPS)`）。

---

## E9 动手题 ★★ 需要硬件（GPU，可选）

在一台有 NVIDIA GPU、且已安装 `tilelang==0.1.15` 的机器上：

1. 运行 `python3 code/quickstart-gemm/quickstart_gemm.py --out /tmp/tl-debug`。
2. 打开 `/tmp/tl-debug/tilelang_jit_kernel_*.c`，找出
   `__global__`、`__shared__`、`cp.async` 或 TMA 相关的行各一处，并说明它们分别对应
   本章表格里的哪个 pass 产物。
3. 打开 `/tmp/tl-debug/tilelang_jit_program_*.py`，与原始 kernel 对比：
   `T.gemm` / `T.copy` / `T.Pipelined` 还在吗？这说明了什么？
4. 把 `--block-k` 从 32 改成 64 再跑一次，观察生成的 `__shared__` 大小与
   pipeline 步数的变化。

**没有 GPU 时**：请只完成第 3 题的静态部分，在 `_vendor/tilelang-v0.1.15/` 里找出
`LowerTileOp` 前后 IR 里 tile op 是否还存在（提示：`grep -rn "tile_op" src/op/`），
并明确写出你没有验证的部分。

---

## E10 概念题 ★★ 不需要硬件

对比两侧的 `MaterializeKernelLaunch` 调用：

```python
# tilelang/cuda/pipeline.py:70
MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)

# tilelang/ascend/pipeline.py:45-51
MaterializeKernelLaunch(lower_grid_binding=True,
                        lower_thread_binding=False,
                        default_threads=None,
                        unsupported_annotations=["cluster_dims"],
                        launch_dim_tags=["cthread"])
```

1. 这两组参数分别会产生什么 IR 结构（`blockIdx` / `threadIdx` / `cthread` 各在哪里出现）？
2. 这个差异对**用户能写什么代码**有什么直接影响？请用 `T.SimtVF` 解释昇腾侧
   “线程域只存在于向量区域内”这句话。
3. `unsupported_annotations=["cluster_dims"]` 的作用是什么？如果用户对昇腾目标写
   `T.Kernel(..., cluster_dims=2)` 会发生什么？

---

## E11 概念题 ★★ 不需要硬件

有人告诉你：“TileLang 现在已经支持 `T.PersistentKernel`，也有 PTO 执行后端了。”
你手头有 `_vendor/tilelang-v0.1.15/` 与 corpus 的 main 快照
`../tilelang-corpus/02-tilelang-core-repo/raw/tilelang`。

1. 写出至少三条核验命令，用来判断这两个说法在**哪个 checkout** 里成立。
2. 分别给出结论：`T.PersistentKernel` 与 PTO 到底在哪个版本里？
3. 为什么本书要求“讨论 main 上的能力时必须显式标注，并且放在第 30 章”？
4. 判断这条说法属于本章 5.3 节表格的哪一行，并说明你自己的核验路径。

---

## E12 动手题 ★★★ 不需要硬件（设计题）

你要给 TileLang 增加一个虚构的新后端 `foo`（假设它的 target kind 就叫 `foo`，
代码生成走一个已有的 native 函数 `target.build.tilelang_foo`）。

1. 按 `tilelang/backend/README.md:325-345` 的“Backend Package Layout”，
   写出你要新建的文件清单，并说明每个文件的职责。
2. 在 `BackendModule` 里你必须提供哪些字段？哪些字段是可选的？给出一个最小 manifest
   草稿（Python 代码，不用真的可运行，但要字段名正确）。
3. 如果你的 `foo` 想复用 CUDA 的 pipeline，应该怎么写？这个做法在 v0.1.15 里有没有
   先例？给出先例的文件与行号。
4. 按 `:374-392` 的 Ownership Rules 逐条检查你的方案，指出最容易违规的一条并说明
   怎么避免。
