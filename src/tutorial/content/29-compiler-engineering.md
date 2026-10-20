# 第 29 章 编译器工程：写 pass、测试与回归

## 学习目标

读完本章，你应当能够：

1. 拿起任意一个 TileLang pass，**说出它在哪一侧**（C++ 还是 Python）、依据是什么，
   并复述“定义 → 实现 → 注册 → 挂载 → 测试”这条完整链路。
2. 解释 `tilelang/transform/_ffi_api.py` 与 C++ 侧 `TVM_FFI_STATIC_INIT_BLOCK()` /
   `refl::GlobalDef().def(...)` 的对应关系，说清**改名时哪三处必须同时改**，
   以及为什么改错的情形只在运行时暴露。
3. 照着本章的两个最小示例写出一个能跑的新 pass：Python 版（不重编 TileLang）与
   C++ 版（含 `CreatePrimFuncPass` 注册与挂载），并给出可复现的验证命令。
4. 为一个 pass 设计**四层测试**（IR diff 基线 / 单元测试 / 边界条件 / 端到端对拍），
   说出每层能证明什么、不能证明什么。
5. 用 benchmark + `pass_timing` + `lower_trace` 判定一次改动是否造成性能回归，
   并把回归定位到具体 pass。
6. 说出昇腾侧的额外约束：两棵源码树各自改什么、适配器仓库 `AGENTS.md` 里哪些是
   硬约束、昇腾侧的测试与基准入口在哪。

## 前置知识

* 第 20 章“IR 与编译期管线精读”：本章是它的续集。第 20 章教你**读出**每个 pass
  对 IR 做了什么，本章教你**改出来**。如果还分不清 `IRModule` / `PrimFunc` /
  `SBlock` / `ForKind`，先回去读第 20 章第 2 节。
* 第 4 章“架构总览与一次编译的全流程”：知道五个阶段与 `context.lower(mod)` 这个
  唯一分岔点。
* 第 12 章“调试与性能分析”：`lower_trace` / `pass_diff` / `compile_only` 在那里
  首次出现；本章只讲**怎么用它们调试你自己写的 pass**。
* 第 21–23 章：布局推导、流水与同步、昇腾自动调度。那三章的 pass 是“读懂一个
  pass 再改它”的最佳练习对象，但读本章不需要先读完它们。
* 会读 Python 与 **C++17**；知道“pass 就是一个 `IRModule → IRModule` 的函数”。
  不需要写过 CUDA，也不需要读过 TVM 源码。

本章回答一个问题：**从“我能读懂这个 pass”到“我改了它，并且知道它没坏”，
中间缺的是哪几步。**

---

## 1. 问题的由来

### 1.1 读 pass 和写 pass 差的不是知识，是流程

第 20 章结束时，你应该能看着 `tilelang/ascend/pipeline.py` 说出 80 条 pass 调用
各自在解决什么问题。但真正坐到键盘前要改一个 pass 时，冒出来的是另一批问题：

* 我要加的这条检查，**该写在 C++ 还是 Python**？
* 我写好了，**怎么让它被 Python 看到**？为什么 `tilelang.transform.X()` 找不到
  我的函数？
* 我把它挂进 pipeline 了，编译还是过的，**怎么知道它到底跑了、跑对了**？
* 我把性能跑慢了 4%，**是我这个 pass 干的吗**？
* 昇腾那边有一棵**另外的仓库**，我要改哪一个？

这五个问题都不是“API 怎么用”，而是**工程流程**问题。它们各自的答案都很短，
但缺一条就卡住。

### 1.2 一个真实的痛点：新检查有三种“静默失效”

考虑一个具体场景。你想加一条检查：**“local 作用域的数组不能太大”**。照着现有
verify pass 抄完之后，有三种失败方式是编译期完全不报错的：

1. `VisitStmt_(const BufferStoreNode *op)` 少写一个下划线，成了
   `VisitStmt(const BufferStoreNode *op)`，编译通过、pass 也跑，只是永远不会被
   调用（C++ 的重载规则，见 3.3 小节）。
2. C++ 注册名写成 `"tl.transform.VerifyLocalAlloc"`，Python 侧调的是
   `_ffi_api.VerifyLocalAllocLimit()`，`AttributeError`，但只在首次调用时炸。
3. 检查逻辑全对，但它被挂在 `LowerOpaqueBlock` 之后，此时 `alloc_buffers` 已经
   被改写、`SBlock` 也没了。它每次都“没发现问题”。

本章的每一小节，本质上都是在拆掉这三种静默失效中的一种。

### 1.3 本章的四条纪律

1. **所有源码引用来自锁定版本。** 行号一律写成
   `_vendor/tilelang-v0.1.15/src/transform/verify_buffer_init.cc:419-422` 这种形式，
   并且在 2026-10-01 现场核对过。涉及昇腾适配器的写
   `_vendor/tilelang-ascend-v0.1.2.000/`。版本以
   [../../VERSIONS.md](../../VERSIONS.md) 为准。
2. **不写没有跑过的运行结果。** 本机是 macOS（Apple Silicon），**没有 NVIDIA
   GPU，没有昇腾 NPU，也没有安装 tilelang/tvm**（`import tilelang` 直接
   `ModuleNotFoundError`，`command -v nvcc` 无输出）。因此凡是“跑起来是什么样”
   的结论，一律写成“应当是什么样”并给出验证命令；**“能编译”这类结论一律标为
   待验证**。
3. **测试不是附属品。** 一个 pass 没有 IR 基线就不算完成：`它没坏` 与
   `它没改坏别的` 是两件事。
4. **一句话说不清的设计取舍，不要放进 pipeline。** 加 pass 的成本不只是代码，
   还有编译时间，以及每一个后续 pass 看到的 IR。

---

## 2. 概念与模型

### 2.1 一个 pass 的五个要素

不管写在哪一侧，一个 pass 都由五件事组成。本章后面所有内容都挂在这五个词上：

| 要素 | 问题 | C++ 侧 | Python 侧 |
| --- | --- | --- | --- |
| **定义** | 解决什么问题、精度边界在哪 | 文件头注释（`verify_buffer_init.cc:1-16`） | 模块 docstring |
| **实现** | 遍历哪些节点、怎么判定 | `StmtExprVisitor` / `StmtExprMutator` 子类 | `PyStmtExprVisitor` / `PyStmtExprMutator` + `@tirx.functor.*` |
| **注册** | Python 怎么找到它 | `TVM_FFI_STATIC_INIT_BLOCK()` + `refl::GlobalDef().def(...)` | 不需要，`from .mod import X` |
| **挂载** | 什么时候跑 | backend `pipeline.py` 里的 `mod = X()(mod)` | 同一行，写在 `tilelang/transform/__init__.py` 导出后 |
| **测试** | 怎么证明它对了 | `testing/python/transform/test_*.py` | 同上 |

### 2.2 双语结构：哪些在 C++，哪些在 Python

图 1（`figures/fig-01-dual-tree-and-ffi.md`）画了完整通路，这里只给判断规则。

**由 C++ 实现（主体）**

* `src/transform/*.cc`：42 个顶层文件，另有 `layout_inference/`、`common/`
  两个子目录；通用 pass 都在这里（`verify_buffer_init.cc`、
  `pipeline_planning.cc`、`layout_inference/`、`thread_storage_sync.cc` …）。
* `src/ascend/transform/*.cc`：31 个顶层文件，另有 `auto_schedule/` 子目录；
  昇腾特有 pass 都在这里（`insert_nd2nz.cc`、`insert_sync.cc`、
  `materialize_multi_buffer.cc` …）。

**由 Python 实现**

* `tilelang/transform/` 下 3 个：`add_bufstore_wrapper.py`、
  `decouple_type_cast.py`、`hoist_broadcast_values.py`。它们的共同点是
  **改写形状清晰、不需要新 IR 节点、且仍在演进**。
* `tilelang/analysis/` 下 5 个检查器：`ast_printer.py`、`fragment_loop_checker.py`、
  `layout_visual.py`、`nested_loop_checker.py`、`parallel_local_index_checker.py`。
* `tilelang/ascend/analysis/vf_checker.py`：`VFChecker`，被
  `tilelang/ascend/pipeline.py:95` 直接调用。
* `tilelang/metal/transform/` 下 2 个。
* 另有一类**不是 pass** 的 Python 函数：`tilelang/ascend/transform/z3_scheduler.py`
  用 `@tvm_ffi.register_global_func("tl.transform.z3_schedule_python")`（`:228`）与
  `"tl.transform.z3_schedule_loop_python"`（`:784`）注册了两个函数，由 C++ 侧
  `AutoSchedule` 回调。**它们不是 pass，是把 Python 当算法库用**。这是昇腾路线
  的一个特点，第 23 章详细讲。

**怎么选？** 按顺序问自己：

1. 需要遍历/改写上游已有的 TIR 节点吗？需要 → 两边都能做，接着看第 2 问。
2. 需要访问 C++ 侧独有的语义吗（`TileOperator` 接口、layout 对象、`BufferRegion`
   上的 access 信息）？需要 → **写 C++**。
3. 改动会频繁迭代，或需要在没有编译环境的地方调试？→ **写 Python**。
4. 会被多个后端复用、且对编译器自身耗时敏感？→ **写 C++**。

一个可检验的经验法则：`verify_buffer_init` 这类检查如果现在重写，很可能写成
Python。它只需要 `BufferStore` / `BufferLoad` 与 tile op 的读写集合；而
`LayoutInference` 必须留在 C++，因为它要访问 layout 对象与 op 的 `InferLayout`
接口。事实上 `verify_parallel_loop.cc:3` 就 `#include "common/constr_visitor.h"`，
复用了 C++ 侧的约束集合基础设施（见第 23 章），这正是“留在 C++”的理由。

### 2.3 FFI：注册名怎样变成 Python 属性

这是最容易出错、也最容易讲清的一环。整条链路只有两行关键代码：

```python
# tilelang/transform/_ffi_api.py:3,6（全文件 6 行，没有别的逻辑）
import tvm_ffi
tvm_ffi.init_ffi_api("tl.transform", __name__)
```

```cpp
// src/transform/verify_buffer_init.cc:419-422
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyBufferInit", VerifyBufferInit);
}
```

`init_ffi_api("tl.transform", __name__)` 做的是：把注册表中所有以
`"tl.transform."` 开头的全局函数，按去掉前缀后的名字，挂成本模块（`__name__`
即 `tilelang.transform._ffi_api`）的属性。于是：

```text
注册名 "tl.transform.VerifyBufferInit"
        └─────────┬──────────────┘
           前缀 tl.transform        ← 被 init_ffi_api 的第一个参数吃掉
                      余下 VerifyBufferInit → 变成属性 _ffi_api.VerifyBufferInit
```

**改名时要同时改三处**（code review 最常抓的一类错）：

1. C++ 里的注册字符串 `"tl.transform.X"`；
2. C++ 里 `CreatePrimFuncPass(pass_func, 0, "tl.X", {})` 的第三个参数。它出现在
   `TL_LOWER_TRACE` / `TL_PASS_PROFILE` 报告里，**可以和注册名不同**；
3. Python 里的薄包装 `def X(): return _ffi_api.X()`。

第 1、3 处改错 → `AttributeError`；第 2 处改错 → 不报错，只是报告里出现一个你
认不出来的名字。**三种错误的暴露时机不同**，这就是 3.2 节说“必须有一条会命中
的测试”是唯一可靠防线的原因。

### 2.4 挂载：命令式 pipeline，没有注册表

熟悉 LLVM 或 TVM 官方 `PassManager` 的人会本能地去找“注册到某个 pass manager”。
TileLang 不是这样。挂载**就是一行函数调用**：

```python
# tilelang/cuda/pipeline.py:97-98
mod = tilelang.transform.VerifyBufferInit()(mod)
```

每条 pipeline 就是一个普通 Python 函数（`CUDAPassPipelineBodyPrologue` /
`CUDAPassPipelineBody` / `AscendPassPipelineBody`），由 `PassPipeline` 包一层
instrumentation 作用域（`tilelang/backend/pass_pipeline/pipeline.py:11,22-31`）：

```python
CUDA_PIPELINE = PassPipeline("cuda", CUDAPassPipelineBody)         # pipeline.py:284
ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)   # ascend/pipeline.py:190
```

这个设计有两个直接后果，写 pass 时都要记住：

* **顺序是硬编码的**，没有“优先级”可以声明。你的 pass 放在哪一行，就在哪个阶段
  的 IR 形态上工作。
* **想被所有后端复用，就得在每个后端各加一行。** 一个通用 pass
  （如 `VerifyBufferInit`）在 v0.1.15 里被 **6 条** pipeline 各自调用一次：
  `cuda/pipeline.py:98`、`ascend/pipeline.py:76`、`cpu/pipeline.py:35`、
  `metal/pipeline.py:40`、`rocm/pipeline.py:39`、`webgpu/pipeline.py:41`。
  `DecoupleTypeCast` 同样是 6 处。**“只挂了一个后端”是 PR 评审里最常见的遗漏。**

### 2.5 pass 之间的顺序约束

把 pass 加进 pipeline 时，真正要判断的是**它需要的 IR 形态在什么时候还在**。
锁定版本里现成的例子：

* `VerifyBufferInit` 必须在 `PipelinePlanning` **之前**。CUDA 侧的注释写得很直白
  （`tilelang/cuda/pipeline.py:93-98`）：“Runs after the reducer passes above …
  and before PipelinePlanning and LowerTileOp, while loop bodies are still in
  source order and tile ops still declare their access regions.”。它依赖
  “源码顺序”和“tile op 的 `GetAccessRegions()` 还在”两个前提。
* `DecoupleTypeCast` 必须在 `VectorizeLoop` **之前**，理由写在
  `tilelang/transform/decouple_type_cast.py:650-652`：“must be applied before
  VectorizeLoop and StorageRewrite passes, while the IR still uses
  BufferLoad/BufferStore (not `tvm_access_ptr`)”。
* `AddWrapperForSingleBufStore` 挂在几乎所有 pipeline 的**最前面**
  （`cuda/pipeline.py:79`、`ascend/pipeline.py:58`），因为它要把“裸 store”包成
  `For(PARALLEL)`，而之后所有 pass 都假设 store 已经在并行循环里。

一张可复用的判断表：

| 你的 pass 需要什么 | 它必须放在谁之前 |
| --- | --- |
| `T.copy` / `T.gemm` 这类 tile op 还在，且带 access region | `LowerTileOp` 之前 |
| 循环体还是源码顺序、`for` 还没被打散 | `PipelinePlanning` / `InjectSoftwarePipeline` 之前 |
| 还是 `BufferLoad` / `BufferStore`，没被降成 `tvm_access_ptr` | `VectorizeLoop` / `FlattenBuffer` 之前 |
| `SBlock` 还在，`alloc_buffers` / `annotations` 还能读 | `LowerOpaqueBlock` 之前 |
| `thread_extent` 已经是真实线程域（只对 SIMT 后端） | `MaterializeKernelLaunch` 之后 |
| 已有 layout 信息 | `LayoutInference` 之后 |

---

## 3. 用法：从零写一个 pass

### 3.1 五步法

```text
① 定义     写清「报什么、不报什么、精度边界在哪」。这段会进文件头注释。
② 选语言   按 2.2 节的四问决定写 C++ 还是 Python。
③ 实现     写出 visitor / mutator；pass_func 保持「要么原样返回，要么返回新 func」。
④ 注册      C++：注册宏 + Python 薄包装；Python：写进 tilelang/transform/__init__.py。
⑤ 挂载+测试 对着 2.5 节的表选位置挂进 pipeline，再补四层测试。
```

下面给两个最小完整示例。**Python 版可以立刻验证语法与 API；
C++ 版需要编译环境，本机没有验证过。**

### 3.2 示例 A：Python pass（不重编 TileLang）

完整代码在 `code/01-python-pass/`：`hoist_constant_broadcast.py` 是 pass 本体，
`test_hoist_constant_broadcast.py` 是三个 IR 前后对照用例。

它做的事是：把 buffer store 里的**立即数广播** `T.Broadcast(<IntImm|FloatImm>,
lanes)` 提取成一个 `Bind` 定义的变量。原型是真实存在的
`tilelang/transform/hoist_broadcast_values.py`（pass 主体在 `:75-99`，mutator 在
`:16-72`）；我们把它简化、改名，保留两个关键设计点：

```python
@tirx.functor.mutator
class HoistConstantBroadcastMutator(PyStmtExprMutator):
    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[Var, tirx.PrimExpr]] = []

    def visit_broadcast_(self, op: Broadcast):
        if isinstance(op.value, (tirx.IntImm, tirx.FloatImm)):
            value = self.visit_expr(op.value)
            new_var = Var("broadcast_value", dtype=value.dtype)
            self.pending.append((new_var, value))
            return Broadcast(new_var, op.lanes)
        return Broadcast(self.visit_expr(op.value), self.visit_expr(op.lanes))

    def visit_buffer_store_(self, op: BufferStore):
        saved_pending = self.pending
        self.pending = []                      # ① 进入子语句：清空
        new_indices = [self.visit_expr(index) for index in op.indices]
        new_stmt = BufferStore(op.buffer, self.visit_expr(op.value), new_indices)
        if self.pending:                       # ② 离开子语句：把定义挂到语句之前
            stmts: list[tirx.Stmt] = [Bind(var, value) for var, value in self.pending]
            stmts.append(new_stmt)
            new_stmt = SeqStmt(stmts)
        self.pending = saved_pending           # ③ 恢复
        stamp_stmt_spans(new_stmt, get_stmt_span(op))
        return new_stmt


def HoistConstantBroadcast():
    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = HoistConstantBroadcastMutator()
        return func.with_body(mutator.visit_stmt(func.body), span=func.span)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.HoistConstantBroadcast")
```

四个逐行要点：

* **`PyStmtExprMutator` + `@tirx.functor.mutator`**：装饰器把 Python 方法
  `visit_broadcast_` 接到 C++ visitor 的 `VisitExpr_(const BroadcastNode*)` 上，
  命名规则是“`visit_` + 小写节点名 + `_`”。写错名字不报错，只是不会被调用。
* **`pending` 的清空—恢复**：这是 mutator 的局部状态管理。少了这三行，第二条
  store 会引用第一条 store 定义的变量，生成的 IR 直接非法。
* **`stamp_stmt_spans`**：重建节点会丢 span，而 span 是编译报错指回 Python 行的
  唯一依据（`TL_EMIT_LINE_DIRECTIVES` 也依赖它）。`tilelang/ir.py` 里的
  `get_stmt_span`（`:105`）与 `stamp_stmt_spans`（`:123`）就是为此存在。
* **`prim_func_pass(pass_fn, opt_level=0, name=...)`**：`name` 决定
  `TL_LOWER_TRACE` / `TL_PASS_PROFILE` 报告里的名字。不传 name 的 Python pass
  （如上游 `HoistBroadcastValues`，`hoist_broadcast_values.py:99`）在报告里会退化
  成自动推断的名字，能跑，但不好认。

挂载只有两处改动：

```python
# tilelang/transform/__init__.py，紧挨其他 Python pass 的 import
from .hoist_constant_broadcast import HoistConstantBroadcast  # noqa: F401

# 某个 backend 的 pipeline 里（例如 tilelang/cuda/pipeline.py）
mod = tilelang.transform.HoistConstantBroadcast()(mod)
```

**验证级别**：本机只做到 `python3 -m py_compile` 通过，以及文件里出现的每个 API
名逐个在 `_vendor/tilelang-v0.1.15/` 里核对。真正跑测试需要装好 tilelang
（不需要 GPU，因为用例停在 `IRModule` 层）。

### 3.3 示例 B：C++ pass（模板 + 构建步骤）

完整代码在 `code/02-cpp-pass-template/`：`count_parallel_loops.cc` 是 pass 本体，
`python_wrapper_snippet.py` 是三处改动的粘贴片段，
`test_count_parallel_loops.py` 是 pytest 骨架。它同样“只报不改”，数出所有
`For(kind=kParallel)` 并打成一条警告。骨架逐块对照上游：

| 模板里的部分 | 对照的上游代码（`_vendor/tilelang-v0.1.15/`） |
| --- | --- |
| `struct ... : public StmtExprVisitor` + `VisitStmt_(const ForNode*)` | `src/transform/verify_parallel_loop.cc:28-56` |
| `SpanHintSuffix({span})` + `#include "span_utils.h"` | `src/transform/verify_buffer_init.cc:22,383` |
| `CreatePrimFuncPass(pass_func, 0, "tl.X", {})` | `src/transform/verify_buffer_init.cc:416` |
| `TVM_FFI_STATIC_INIT_BLOCK` + `refl::GlobalDef().def(...)` | `src/transform/verify_buffer_init.cc:419-422`、`src/ascend/transform/insert_nd2nz.cc:488-490` |
| Python 薄包装 | `tilelang/transform/__init__.py:86-94` |

核心三块（完整版见文件）：

```cpp
namespace tvm::tl {
using namespace tirx;          // StmtExprVisitor / ForKind / PrimFunc
using namespace ffi;           // 让下面的 refl::GlobalDef 能解析

namespace {
struct ParallelLoopReporter : public StmtExprVisitor {
  std::vector<LoopInfo> loops_;
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      loops_.push_back({op->span, StaticExtent(op->extent)});
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void EmitReport() const { /* 合成一条 LOG(WARNING) */ }
};
} // namespace

using namespace tirx::transform;

tvm::transform::Pass CountParallelLoops() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ParallelLoopReporter reporter;
    reporter(f->body);
    reporter.EmitReport();
    return f;                          // 只报不改
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CountParallelLoops", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.CountParallelLoops", CountParallelLoops);
}
} // namespace tvm::tl
```

**必须知道的三个 C++ 陷阱**（都写进了文件注释）：

* 重载名不能写错。`VisitStmt_(const ForNode *op) final` 写成 `VisitStmt(op)` 会
  编译通过、静默不生效。这是 C++ pass 最常见的一类“什么都没发生”。
* `pass_func` 里**先记录还是先递归**必须和你的语义一致：对报告型 pass 只是排序
  问题，对改 IR 的 pass 就是语义错误。
* **`CreatePrimFuncPass(pass_func, 0, ...)` 的 `opt_level=0` 意味着“必须执行”**，
  不会因优化开关被跳过。语义 pass（含检查）都应该用 0。

**构建与验证步骤**（本机没有编译环境，以下命令是**待验证项**，完整版见
`code/02-cpp-pass-template/README.md`）：

```bash
# 准备一棵可写的 checkout（不是 _vendor/）
git clone --recurse-submodules git@github.com:<your-name>/tilelang.git
cd tilelang
uv venv --seed .venv && source .venv/bin/activate
uv pip install --requirements requirements-dev.txt
pre-commit install --install-hooks

# 落地：新 .cc + Python 薄包装 + pipeline 一行 + 测试
cp <THIS_DIR>/count_parallel_loops.cc src/transform/count_parallel_loops.cc
cp <THIS_DIR>/test_count_parallel_loops.py testing/python/transform/

# 编译扩展
python3 -m pip install --no-build-isolation --verbose --editable .

# 冒烟：确认 Python 能拿到这个 pass（不需要 GPU）
python3 -c "import tilelang; print(tilelang.transform.CountParallelLoops())"

# 单元测试（前两条不需要 GPU）
python3 -m pytest testing/python/transform/test_count_parallel_loops.py -q

# 风格
bash format.sh --files src/transform/count_parallel_loops.cc
```

为什么第 6 步要**重新 configure**：CMake 用 `file(GLOB ...)` 收集源文件
（`CMakeLists.txt:374-389`），glob 在 **configure 期**求值。新加 `.cc` 后不重新
configure，文件不会进构建，于是冒烟那一步报的仍然是 `AttributeError`，很容易
误判成“注册名写错了”。

### 3.4 给 pass 加开关

如果检查可能误报，就给它一个开关。仓库里现成的模式是
`tl.disable_buffer_init_check`：

```cpp
// src/op/builtin.h:118-124（注释 + 常量名）与 src/op/builtin.cc:37（注册）
static constexpr const char *kDisableBufferInitCheck =
    "tl.disable_buffer_init_check";
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableBufferInitCheck, Bool);
```

```cpp
// src/transform/verify_buffer_init.cc:406-408
if (ctx->GetConfig(kDisableBufferInitCheck, Bool(false)).value()->value) {
  return f;
}
```

Python 侧的枚举在 `tilelang/transform/pass_config.py:52-58`
（`TL_DISABLE_BUFFER_INIT_CHECK`）；pipeline 里的**源码级守卫**用
`tilelang/backend/pass_pipeline/pipeline_utils.py` 里的一组 `should_*` /
`allow_*` 函数（`:25-76`），例如 `should_enable_race_check()` 决定
`VerifyParallelLoop` 挂不挂（`cuda/pipeline.py:83-84`）。

**加一个 config key 要改四处**：`src/op/builtin.h` 的常量、`src/op/builtin.cc` 的
`TVM_REGISTER_PASS_CONFIG_OPTION`、C++ 里的 `ctx->GetConfig(...)`、
`tilelang/transform/pass_config.py` 的枚举。少一处，用户就会拿到“配置不生效但
也不报错”。

---

## 4. 实现机制

### 4.1 三条职责链

| 职责 | 位置 | 关键点 |
| --- | --- | --- |
| 序列 | `tilelang/{cuda,ascend,cpu,metal,rocm,webgpu}/pipeline.py` | 纯 Python，一行一个 pass；`PassPipeline`（`tilelang/backend/pass_pipeline/pipeline.py:11`）只包 instrumentation 作用域与错误富化 |
| 逻辑 | `src/transform/*.cc`、`src/ascend/transform/*.cc`、`tilelang/transform/*.py`、`tilelang/analysis/*.py` | 编译器算法本身 |
| 门面 | `tilelang/transform/__init__.py` + `_ffi_api.py` | 对外的唯一入口；`_ffi_api.py` 只有 6 行 |

`tilelang/transform/__init__.py` 是**唯一门面**，读者从 Python 侧看到的 pass 名字
都在这里出现。它同时 import 两类东西：C++ 包装（
`def PipelinePlanning(): return _ffi_api.PipelinePlanning()`，`:19-26`）与纯 Python
pass（`from .hoist_broadcast_values import HoistBroadcastValues`，`:10`）。

### 4.2 pass 名是怎么被观察工具看到的

`TL_LOWER_TRACE` / `TL_PASS_PROFILE` / `TILELANG_PASS_DIFF` 是**同一个机制**的
不同消费者，机制在 `tilelang/instrumentation/`：

* `tilelang/instrumentation/session.py:163` 的 `compile_pass_instrumentation(...)`
  开一个“一次编译”的作用域；
* `:213` 的 `instrument_current_pass_context()` 把已注册的工具装进 TVM 的
  `PassContext`；
* 工具通过 `:130` 的 `register_pass_instrumentation_tool(...)` 注册自己；
* `PassPipeline.lower()` 用 `pass_pipeline(self.name)` 给每个 pass 事件打阶段标签
  （`pipeline.py:22-31`）。

于是两个工具都只是“注册一个工具”：

* `tilelang/tools/pass_timing.py:334` 的 `create_pass_timing_tool(pass_configs)`，
  由 `tilelang/jit/kernel.py:243` 在每次 JIT 编译时装配；
* `tilelang/tools/lower_trace/__init__.py` 的 `lower_trace(func, passes, ...)`
  （`:49`）与 `tilelang/tools/lower_trace/core.py:663` 的 `enable()`。

**对你写 pass 的直接意义**：只要给 `CreatePrimFuncPass` / `prim_func_pass` 传一个
可读的 `name`，你的 pass 就会带着这个名字出现在这三份报告里。

### 4.3 instrumentation 与“pass 有没有跑”的关系

新手最容易困惑的一点：**`TL_LOWER_TRACE=1` 看不到你的 pass，不代表它没跑。**
`lower_trace` 挂在完整编译入口（`tilelang/engine/lower.py:117,162,234` 三处
`compile_pass_instrumentation`）上；如果你只是单独调
`tilelang.transform.X()(mod)`，那就没有 session，自然也没有记录。
想观察单个 pass，用程序 API：

```python
from tilelang.tools.lower_trace import lower_trace
import tilelang

lower_trace(func, [("Before", tilelang.transform.Simplify()),
                   ("Mine",   tilelang.transform.HoistConstantBroadcast())],
            mode="terminal")
```

### 4.4 新增 C++ 文件要动的构建点

结论已经说过一次，这里给出完整清单，因为它是“编译通过但不生效”的最常见根因：

| 新文件类型 | 需要动的地方 | 为什么 |
| --- | --- | --- |
| `src/transform/*.cc` | 无需改 CMake，但要**重新 configure** | `CMakeLists.txt:374-389` 的 `file(GLOB ...)` 在 configure 期求值 |
| `src/ascend/transform/*.cc` | 同上；且要 `USE_ASCEND=ON` | `src/ascend/CMakeLists.txt:15-20` 只无条件收 3 个 always 文件，`:22` 处 `if(NOT USE_ASCEND) return()`，`transform/*.cc` 在 `:38` 才进列表 |
| 新的 Python 模块 | `tilelang/transform/__init__.py` 加 import | Python 侧没有自动发现机制 |

昇腾这一点尤其容易踩：`src/ascend/CMakeLists.txt:1-20` 的注释解释了为什么
`src/ascend/ir.cc`、`src/ascend/layout/ascend_layouts.cc` 等**必须无条件编译**，
`tilelang/__init__.py` 会无条件 import `tilelang.ascend`，缺了对象类型注册就会
`Cannot find object type index`，直接 `import tilelang` 失败。而 `transform/` 下的
pass 只在 `USE_ASCEND=ON` 时进构建。

---

## 5. pass 的正确性测试

### 5.1 四种证据，缺一不可

见图 3（`figures/fig-03-test-layers.md`）。下面给每层一个最小可用的做法。

### 5.2 IR diff 基线

最严格的一层：把 pass 作用前后的 IR 都写下来，断言**结构相等**。

```python
def _check(before, after) -> None:
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(AUTO_TARGET)(mod)
    mod = HoistConstantBroadcast()(mod)

    expected = tvm.IRModule.from_expr(after.with_attr("global_symbol", "main"))
    expected = tvm.tirx.transform.BindTarget(AUTO_TARGET)(expected)
    tvm.ir.assert_structural_equal(mod["main"], expected["main"], map_free_vars=True)
```

三个细节：

* **`BindTarget` 不能省。** 没 bind target 的 IR 与 bind 过的 IR 结构不等，
  不 bind 会让“本来一样”的两次比较挂掉。
* **`map_free_vars=True`。** 用例里的 `Var` 是自由变量，允许按位置映射比较。
* **基线要能一键生成。** 手写 `after` 只在用例极小时可行；一旦超过 30 行，
  就该用脚本把当前输出写成快照并在 CI 里 diff。

现成范式：`testing/python/transform/test_tilelang_transform_hoist_broadcast_values.py:56-64`
（`_check`）与 `:67-133`（四个 before/after 用例，其中一个专门守一个历史 bug）。

### 5.3 单元测试：`testing/` 的写法

目录约定：

```text
testing/python/transform/test_tilelang_transform_<pass 名小写>.py   # 通用 pass
testing/ascend/transform/test_ascend_<pass 名小写>.py               # 昇腾 pass
testing/ascend/_ir.py                                               # 昇腾侧共用 IR 构造助手
```

昇腾侧特别注意 `testing/ascend/_ir.py`：它提供 `seq()` / `region()` / `copy()` /
`kernel()` / `calls()` / `nodes()` 这组助手（`:15-58`），让你能手搓一个带
`tl.tileop.ascend_copy` 的 IR，而不用起一个前端 kernel。因为昇腾 pass 的输入常常
是“已经排过序的 IR”。用法见
`testing/ascend/transform/test_ascend_insert_nd2nz.py:11-46`。

通用 pass 侧更朴素：**手搓 `IRModule` → 调 pass → 断言**。两个范式：

```python
# 断言 stderr 文本（verify / report 型 pass）
def _verify(mod, capfd) -> str:
    tilelang.transform.VerifyBufferInit()(mod)
    return capfd.readouterr().err

assert MESSAGE in _verify(kernel, capfd)
```

```python
# 断言 IR 里出现了某个 op（重写型 pass）
from testing.ascend._ir import calls
(scatter,) = calls(after, "tl.ascend_nd2nz_scatter")
assert not calls(after, "tl.tileop.ascend_copy")
```

### 5.4 端到端对拍

前三层都在 IR 层。真正会漏掉的是：**新 pass 让真实编译挂掉**。所以至少要有一条：

```python
@tilelang.testing.requires_cuda
def test_pipeline_warns_by_default(capfd, no_kernel_cache):
    tilelang.compile(_gemm_kernel("none"))
    assert MESSAGE in capfd.readouterr().err
```

两个来自上游的细节值得照抄：

* **`no_kernel_cache` fixture**（`test_tilelang_transform_verify_buffer_init.py:377-388`）：
  命中 kernel cache 时根本不跑 pass，警告会“第二次编译时消失”，用这个 fixture
  强制每次重编。
* **数值对拍要和 `torch.allclose` 一起写**：编译过只是活着，算对才是正确。
  昇腾侧对应的是 `torch_npu` 张量与 `T.dump_tensor`（见第 19 章）。

### 5.5 边界条件清单

verify pass 的价值几乎全在边界上。以 `verify_buffer_init.cc` 的用例集为模板，
一个检查至少要有这几类反向用例（每一条都对应上游一个真实测试）：

| 边界 | 为什么必须有 | 上游用例 |
| --- | --- | --- |
| 恰好不在管辖范围（global scope） | 防止过度报告 | `test_tilelang_transform_verify_buffer_init.py:111-120` |
| 由外部手段建立（`call_extern` 填的 buffer） | 保守策略“当作已写” | 同文件 `:106-108` |
| 读自己的目的地址（`x[i] = x[i] + 1`） | store 先求值再写入 | 同文件 `:183-201` |
| 跨线程作用域的顺序无关性（warp specialization） | shared 靠 barrier 而不是源码顺序 | 同文件 `:218-237` |
| 参数 buffer（由调用方填） | 不该报 | 同文件 `:123-141` |
| 动态形状 / 折不动 | **折不动就不报** | 见 `code/03-verify-pass-template/` |
| 一个 buffer 读多次只报一次 | 报告可读性 | 同文件 `:358-371` |

---

## 6. 性能回归

### 6.1 benchmark 方法

三个层次，按需选：

| 层次 | 入口 | 说明 |
| --- | --- | --- |
| 单个算子的端到端对比 | `_vendor/tilelang-v0.1.15/benchmark/`（`matmul/`、`matmul_fp8/`、`mamba2/`、`blocksparse_attention/`、`compile_speed/`、`matmul_metal/`） | 回答“这个 kernel 快不快” |
| 公共算子集 | corpus `12-operator-libraries/raw/tilelang-benchmark/`：`README.md` 给出 GEMM / FlashAttention / Linear Attention 三张形状表，`hopper_benchmark/`、`ampere_benchmark/`、`ada_benchmark/`、`cdna_benchmark/` 分区存放各代硬件的结果 | 回答“同一批形状跨硬件横向对比”；注意该仓库标题已写明 Archived、作者指向 TileOPs |
| 改动前后的回归判定 | `maint/scripts/run_perf_regression.sh` | 回答“我的 PR 有没有把谁改慢”。**这才是改 pass 时要跑的** |

`run_perf_regression.sh` 的做法值得记住：它**不在你的工作树里切分支**，而是
`git worktree add --detach` 出一份基线 checkout（`:150-168`），两边分别建 venv、
建 wheel（带缓存，`:240-369`），各跑一遍 `run_current_regression.py` 产出 JSON，
最后交给 `compare_perf_regression.py` 出 md / png / json。常用环境变量写在脚本头
`:15-32`：

```bash
# 只跑你改动相关的那几组
./maint/scripts/run_perf_regression.sh -k gemm/regression_example_gemm.py

# 判为回归就退出码非零，供 CI 用
FAIL_ON_REGRESSION=1 REGRESSION_THRESHOLD=0.95 ./maint/scripts/run_perf_regression.sh
```

### 6.2 怎么判断“是不是回归”

**不要看单条数字。** 仓库里的判定规则在
`maint/scripts/test_perf_regression.py:24-32,243-257`：

```text
speedup = old_latency / new_latency
regressed ⟺ 至少有 MIN_COUNT 条（默认 2）speedup < THRESHOLD（默认 0.90）
```

那段注释给出了标定依据：到 PR #3176 为止的 33 份机器人报告、1901 个样本中，
真回归都表现为 14–23 条同时塌陷；而 7 份假警报恰好只有一条落在极端值
（最低 0.044x）上，其余全是 1.000。于是“至少两条同时掉”这个门槛正好把两类分开。

`code/04-regression-judge/` 把这条规则抽成可离线运行的判定器，`--self-test` 输出
（本机真实运行过）：

```text
场景 A：40 条里 1 条 0.85x
**不判为回归**：1 条低于 0.9x，未达到 2 条的门槛。

场景 B：40 条里 2 条 0.88x
**回归：2/40 条低于 0.9x**
```

也就是说：**“有一条变慢了”和“发生了回归”是两句不同的话**，后者需要样本量。
同一个脚本还演示了第三种情况，只在一边出现的 case 不参与比较，
“少了结果”与“结果变差”在 `compare_perf_regression.py` 里对应
`--fail-on-missing` 与 `--fail-on-regression` 两个不同开关。

**注意两个不同的默认值，别把它们混成一个。** 上面这套“0.90 + 至少 2 条”是
PR 机器人报告的判定口径（`maint/scripts/test_perf_regression.py:31-32`）。而
`run_perf_regression.sh:28` 给 `FAIL_ON_REGRESSION` 用的 `REGRESSION_THRESHOLD`
默认是 **1.0**，也就是“只要有一条比基线慢就退出码非零”。这是 CI 门禁的保守
选择，不是回归判定本身。要做批量判定，用前者；要做门禁，用后者并显式设定阈值。

### 6.3 定位回归来源

判定为回归之后，第一步是分清**编译变慢**还是**产物变慢**，两件独立的事，
两套工具：

```bash
# ① 编译期：每个 pass 的 inclusive / self 用时
TILELANG_PASS_PROFILE=1 python3 my_case.py
#    入口：tilelang/tools/pass_timing.py:334 create_pass_timing_tool

# ② 产物：哪些 pass 改了 IR
TL_LOWER_TRACE=terminal python3 my_case.py
#    入口：tilelang/tools/lower_trace/__init__.py:49 / core.py:663
```

读 `pass_timing` 的关键是分清两列：**self** 是这个 pass 自己花的时间，**inclusive**
包含它调用的子 pass（`tilelang/tools/pass_timing.py:64-73` 的
`PassTimingRecord.self_duration_s` / `duration_s`）。看到“外层 pass 很慢”先看
self，否则会把子 pass 的成本算到外层头上。

读 `lower_trace` 的关键是**只盯你自己那个 pass 的 diff**：报告按阶段与顺序编号，
你要确认三件事。它确实被调用了、它的 before/after 差异是你预期的那几行、
它之后的下游 pass 没有因此产生额外变化。

图 4（`figures/fig-04-regression-triage.md`）把这一步画成了流程。

### 6.4 昇腾侧的回归入口

昇腾不共用 CUDA 的脚本：

* `maint/scripts/run_perf_regression_ascend.sh`：基线默认取
  `https://github.com/tile-ai/tilelang.git` 的 `main`，用 `NINJA_JOBS`
  控制并行重编（默认 64），可用 `SKIP_BUILD=1` 跳过两次 checkout 之间的重编。
* `maint/scripts/ascend_perf_regression.py`：对应的 Python 侧驱动（由
  `run_perf_regression_ascend.sh:55` 的 `DRIVER_PATH` 调用）。
* `maint/scripts/run_local_ci_test_ascend.sh`：单测入口，跑 `examples/ascend/` 与
  `testing/ascend/`，**不加载 CUDA 调度插件、不探测 GPU**，默认 4 个 example
  worker + 32 个单测 worker。

---

## 7. 调试一个 pass 的手段

### 7.1 IR 打印

按成本从低到高：

1. **`mod.script()`**：把 `IRModule` 打印成 TIR 文本。写单测时最常用，
   加减一行就能看。
2. **`TL_LOWER_TRACE=1`**：跑一次真实编译，拿逐 pass 的 HTML 报告 + 磁盘上的
   `.tir` 快照（官方说明见 `docs/tools/lower_trace.md`）。这是**观察别人的
   pass** 最省事的方式。
3. **`TL_ENABLE_DUMP_IR` / `TL_DUMP_IR_PATH`**：常量定义在
   `src/op/builtin.h:129-130`，适合在 CI 或崩溃现场留档。
4. **设备侧打印（`T.printf` / `T.dump_tensor`）**：只能回答“运行时值对不对”，
   回答不了“IR 变成什么样”。第 12 章讲过。

### 7.2 `pass_diff` 与 `lower_trace`

两者都是“逐 pass diff”，差别在工程成熟度：

* `tilelang/utils/pass_diff.py:1030` 的 `pass_diff(func, passes, mode=...)` 是**程序
  API**：给定 pass 链，输出终端彩色 diff 或 HTML。适合“我就想看我这一小段”。
* `TL_LOWER_TRACE` / `tilelang/tools/lower_trace/` 是**整条 pipeline 的观察器**：
  带阶段标签、codegen 捕获、可编辑重编、崩溃安全的增量 HTML。

官方文档已经把 `pass_diff` 标成被取代：`docs/tools/pass_diff.md` 顶部有一条
admonition“Superseded — use IR Lower Trace”，
`TILELANG_PASS_DIFF` 仅为向后兼容保留。**新写 pass 时用 `lower_trace`。**

### 7.3 断言：三种强度

| 手段 | 什么时候用 | 崩溃形态 |
| --- | --- | --- |
| `ICHECK(cond) << msg` | 内部不变量（“这里必须有 `tilelang_root`”） | 编译期立即失败，带上下文 |
| `LOG(FATAL) << msg` | 用户代码导致的不可能状态（如管线规划冲突） | 编译期立即失败 |
| `LOG(WARNING) << msg` | 可绕过的问题（未初始化 buffer、数据竞争） | 编译继续，警告进 stderr |

经验法则：**用户能修的问题用 WARNING，编译器自己的 bug 用 ICHECK**。
把“用户写错了”做成 FATAL，会让 `tilelang.compile` 直接抛异常；反过来把内部
不变量做成 WARNING，会让 bug 潜伏到很后面才以别的方式爆出来。

顺便注意 WARNING 的可见性：`tilelang/transform/pass_config.py:60-68` 说明
`tl.disable_data_race_check` 默认是**关**的（因为易误报），需要
`TILELANG_ENABLE_DATA_RACE_CHECK` 环境变量或 pass config 打开。写新检查时，
“默认开还是默认关”本身就是一个要写进 docstring 的决定。

### 7.4 用 verify pass 当模板

`verify_buffer_init` 与 `verify_parallel_loop` 是最好的入门模板，因为它们
**不改 IR**，把“写 pass”的风险压到只剩遍历一种。
`code/03-verify-pass-template/extract_verify_skeleton.py` 可以把这两份骨架连行号
打印出来（本机已运行）。它的输出长这样：

```text
=== verify_buffer_init ===
  header comment       : lines 1-16 (16 lines)
  total lines          : 426
  visitor overrides:
    L 196  void VisitStmt_(const BufferStoreNode *op) final {
    L 201  void VisitExpr_(const CallNode *op) final {
    L 296  void VisitStmt_(const EvaluateNode *op) final {
    L 315  void VisitStmt_(const BufferStoreNode *op) final {
    L 323  void VisitExpr_(const BufferLoadNode *op) final {
    L 330  void VisitExpr_(const CallNode *op) final {
  EmitReport           : L372
  pass factory         : L400  Pass VerifyBufferInit()
  CreatePrimFuncPass   : L416  CreatePrimFuncPass(pass_func, 0, "tl.VerifyBufferInit", {})
  FFI block            : lines 419-422
    registered name    : tl.transform.VerifyBufferInit
```

（看到两个同名的 `VisitStmt_(BufferStoreNode*)` 不要去重：文件里有两个 visitor，
`PotentialWriteCollector` 与 `BufferInitVerifier`，一个收集“谁可能写”，
一个做判定。这是这类 pass 的标准分工。）

抄这三块就够了：

1. **pass_func 永远原样返回 `f`**（`verify_buffer_init.cc:400-417`）；
2. **`reports_` + `EmitReport()`** 把发现合成一条警告
   （`verify_parallel_loop.cc:125-148`）；
3. **每条发现都带补救建议**（`verify_buffer_init.cc:384-389` 对 gemm 累加器
   专门给 `T.clear(C)` / `clear_accum=True`），而不是所有发现共用一句话。

`code/03-verify-pass-template/verify_local_alloc_limit.cc` 是照着这三块写出的第三个
verify pass，附四条边界用例。**验证级别：仅静态核对，未编译。**

---

## 8. 代码风格与评审要求

### 8.1 C++：`docs/developer_guide/cpp_style.md`

`CONTRIBUTING.md:40-62` 把风格要求指向 `docs/developer_guide/cpp_style.md`。
写 pass 时最常撞上的六条（都能在那个文件里找到对应小节）：

| 规则 | 例子 |
| --- | --- |
| 文件名 `lower_snake_case` | `lower_tile_op.cc`、`layout_reducer.h` |
| 类型 `PascalCase`，函数 `PascalCase` | `LayoutNode`、`InferLayout`、`MakeLinearLayout` |
| 布尔函数 `Is`/`Has`/`Can` + PascalCase | `IsSharedBuffer`、`HasTmaBarrier` |
| 参数与局部变量 `lower_snake_case` | `layout_map`、`thread_bounds` |
| 私有/保护成员 `lower_snake_case_` | `layout_remap_`、`analyzer_` |
| 常量与枚举值 `kPascalCase` | `kWarpSize`、`kAccessRead` |

三条容易违反、且**会被评审直接打回**的：

* **handle 类型跨函数边界用 `Stmt` / `Buffer` / `Var`，`*Node` 指针只在 visitor
  回调和局部检查里出现。** 别把 `const ForNode *` 存进成员变量，节点会被
  `CopyOnWrite()` 替换，存下来的指针会失效。
* **比较身份用 `.same_as(other)`，不要用 `.get() ==`。**
* **不要用 `T` 当参数名。** 在 C++ 侧 `T` 通常指 tile op 的模板参数，在 Python 侧
  `T` 是 DSL 命名空间；同名会让读者错乱。

另外注意一条与 FFI 有关的兼容性提醒：**被反射的字段名是 Python/调试接口的一部分**
（cpp_style 的“ObjectNode Fields”一节）。改 `def_ro("src", ...)` 里的名字属于
破坏性改动，即使编译能过。

### 8.2 Python 与格式化

* 格式化：`bash format.sh --files <changed-file>...`（`CONTRIBUTING.md:42-46`）。
  Python 走 Ruff/pre-commit，C/C++ 走 clang-format。**风格改动要和行为改动分开**，
  否则 diff 无法评审。
* 提交前跑 `pre-commit run --all-files`（`CONTRIBUTING.md:111-117`）。
* 有条件时跑一次建议性的 C++ API 审计：
  `python3 maint/scripts/audit_cpp_api_style.py`（`CONTRIBUTING.md:53-62`）。
  它不是批量改名工具，而是评审辅助，每条命中都要先判断 FFI 可见性、生成代码、
  后端 shim 与外部 API 约束，再决定改不改。

### 8.3 `maint/` 里与 pass 相关的自检

`maint/` 不是“杂物间”，它装着几套**金标准（golden）测试**，改 pass 时非常有用：

| 目录 | 作用 |
| --- | --- |
| `maint/layout_inference/` | 构造好的 IR 用例 + 已评审的**期望 layout**。README 说得很清楚：任何改 cost model 的改动都可能静默翻转胜出的 layout，金标准让这种翻转变成一次可评审的 diff，而不是运行时性能谜案 |
| `maint/gemm/` | GEMM 正确性评估脚本（含 sm70 / tcgen05 / tcgen05-2cta / fp4 unpacked 变体） |
| `maint/precision/` | 精度对照（TileLang / Triton / PyTorch / CUDA，Precise vs Fast） |
| `maint/host_checks/` | 10 个**故意触发** host 侧校验错误的最小复现脚本，配套 `docs/compiler_internals/tensor_checks.md` |
| `maint/scripts/` | CI、性能回归、风格审计、wheel 冒烟等入口脚本 |

### 8.4 一个 pass PR 应该带什么

评审人会按这个顺序看：

1. **文件头注释**：报什么、不报什么、精度边界在哪（对照
   `verify_buffer_init.cc:1-16` 的写法）。
2. **它在 pipeline 里的位置**，以及为什么必须在那里（对照 2.5 节的表）。
3. **挂了几个后端**。只挂 CUDA 而在昇腾上静默失效，是高频问题。
4. **测试**：IR 基线有没有、边界用例有没有、端到端那条有没有标
   `requires_cuda` / 昇腾前提。
5. **开关**：误报可能性 + 是否提供了 pass config（对照 3.4 节）。
6. **风格**：命名、handle/Node 用法、`.same_as`、反射字段是否被动过。
7. **性能影响**：编译器自身耗时与产物性能，各给一句话结论与测量方式。

---

## 9. 昇腾侧的额外要求

### 9.1 两棵树的关系

本教材里同时存在两套源码：

| 树 | 路径 | 内容 |
| --- | --- | --- |
| TileLang 主干（含昇腾后端） | `_vendor/tilelang-v0.1.15/` | `src/ascend/transform/` 31 个 `.cc` + `auto_schedule/`；Python 侧 `tilelang/ascend/`（`pipeline.py`、`language/`、`analysis/`、`transform/`） |
| TileLang-Ascend 适配器仓库 | `_vendor/tilelang-ascend-v0.1.2.000/` | 一棵**自带 `src/transform/` 与 `tilelang/transform/`** 的分支树（`ascend_sync_insert.cc`、`ascend_storage_rewrite.cc`、`ascend_memory_planning.cc`、`cross_core_pipeline.cc` …），接口仍是旧的 `tvm._ffi._init_api` |

版本按 [../../VERSIONS.md](../../VERSIONS.md)：适配器仓库的 tag 是
`TileLang-ascend-v0.1.2.000-release`，但仓库内 `VERSION` 文件写的是 `0.1.4`，
**以 tag 为准**。第 20 章已经用差集说明了“两棵树的 pass 词汇不同”。

### 9.2 改哪一棵？

判断顺序：

1. 你要改的是**通用 pass**（`src/transform/` 下的）？→ 改主干，并且在**适配器仓库**
   里同步对应文件，否则两棵树继续漂移。
2. 你要改的是**昇腾专属 pass**？先看主干 `src/ascend/transform/` 里有没有同名 pass。
   有 → 改主干；这是 v0.1.15 之后的主线（`tilelang/ascend/transform/__init__.py`
   的 `__all__` 已经列出 30 个包装函数）。没有 → 说明这个 pass 还在适配器路线里。
3. 你要改的是**调度/同步算法**？主干侧在 `src/ascend/transform/auto_schedule/` 与
   `insert_sync.cc`；适配器侧在 `cross_core_pipeline.cc`、`ascend_sync_insert.cc`。
   两条路线的方法论见第 23、25 章。

一个可现场执行的对照（不用编译）：同一个功能名，在两棵树里 grep 一下，看它落在
哪一侧、用哪个 FFI 接口。

```bash
# 主干
grep -rn "InsertNd2Nz" _vendor/tilelang-v0.1.15/src/ascend/transform/ \
                        _vendor/tilelang-v0.1.15/tilelang/ascend/transform/
# 适配器
grep -rn "SyncInsert\|sync_insert" _vendor/tilelang-ascend-v0.1.2.000/src/transform/
```

### 9.3 适配器仓库 `AGENTS.md` 的硬约束

`_vendor/tilelang-ascend-v0.1.2.000/AGENTS.md:59-75` 的四条通用原则，字面上就是给
AI 协作者写的，但同样约束人：

1. **如实报告，禁止伪完成**，未验证的结果不得表述为“已完成”或“已通过”。
   这与本章 1.3 节的第二条纪律是同一条。
2. **先验证，再下结论**，能通过代码/文件/日志/测试确认的优先基于证据判断。
3. **区分事实、推断与建议**，禁止编造文件、输出、报错、性能收益。
4. **遵循最小必要改动**，优先复用既有实现与模式，不无依据地重写或扩面。

以及 `:79-113` 的核心算子开发原则里与“改编译器”直接相关的三条：**不要凭记忆猜
API**（先查 API 速查、再查 `examples/`、最后查源码，且 `tilelang/language/pto.py`
已废弃禁止使用）；**遇到错误先分析原因、不绕过门禁**；**代码仓探索必须使用
subagent**（这条约束的是多代理工作流，不是人的工作方式）。

命名规范见 `AGENTS.md:117-122`：Python 用 PEP 8 + 类型注解、行宽 100；C++ 用
Google Style + clang-format、行宽 100；文件名 `snake_case.py` /
`snake_case.cc` / `test_<模块>_<功能>.py`。这与主干的 `cpp_style.md` 一致，
但**行宽 100 与测试命名是适配器仓库额外明确的**。

还有一处对写 pass 的人最直接：`AGENTS.md:24-28` 的 Skills 索引里，**Pass 分析与
设计**被单独列成一类，包含 `tilelang-pass-analyzer`（功能分析 / 分类查询）、
`tilelang-pass-workflow-analyzer`（工作流、执行顺序、依赖关系、**新 Pass 定位**）、
`tilelang-pass-design`（设计方案与实现模式）。对应实现在
`.agents/skills/` 下（另有 `tilelang-pass-generate`）。
也就是说，适配器仓库把“我要加一个 pass，它该放哪一段”这件事**显式写进了它的
工作约定**。这与本章 2.5 节的顺序判断表是同一个问题，只是它给了工具化入口。

### 9.4 昇腾侧的测试与基准入口

```bash
# 单测（不加载 CUDA 插件、不探测 GPU）
bash maint/scripts/run_local_ci_test_ascend.sh

# 性能回归（基线默认取 tile-ai/tilelang 的 main）
./maint/scripts/run_perf_regression_ascend.sh
```

昇腾侧单测的目录是 `testing/ascend/`，里面按主题分区：
`transform/`（15 个 pass 测试）、`language/`、`layout/`、`analysis/`、
`auto_schedule/`、`runtime/`、`cache/`、`target/`。新加昇腾 pass 的测试放
`testing/ascend/transform/test_ascend_<名字>.py`，用 `testing/ascend/_ir.py`
的助手构造 IR。

**适配器仓库的入口与主干不同**，别混用：它没有 `maint/scripts/run_*` 这一套，
而是 `maint/scripts/ci_performance.py` / `performance.py`（性能）与
`ci/operator_test_manifest.yaml`（算子测试清单）；`testing/` 下只有 `cpp/` 与
`python/`。改适配器之前先把这两套入口对上号。

---

## 10. 改动类型 → 需要跑的测试 / 基准 → 评审关注点

这张表是本章的落脚点。它把前面九节压成一张可以贴在显示器旁边的对照表。

| 改动类型 | 必须跑的测试 | 必须跑的基准 | 评审关注点 |
| --- | --- | --- | --- |
| 新增 Python pass（改 IR） | IR 基线（`assert_structural_equal`）+ 单元测试 + 至少一条端到端对拍 | 至少一个受影响算子的 benchmark；若影响所有 kernel，跑 `run_perf_regression.sh` | 挂在哪几个后端？顺序对不对（2.5 表）？`name` 传了吗？span 保住了吗？ |
| 新增 C++ pass（改 IR） | 同上，另加：重载名拼写（必须有一条会命中的用例）、`CreatePrimFuncPass` 的 opt_level | 同上，另加 `TILELANG_PASS_PROFILE=1` 看编译器自身耗时 | 注册名三处一致；`src/op/builtin.h` 是否需要新 config；handle/Node 用法；是否该用 `ICHECK` |
| 新增 verify / check pass | IR 层验证：必须命中 + 至少 5 条边界反向用例（5.5 表）+ 一条“不打断真实编译”的端到端用例 | 通常不需要（它不改 IR）；但要在 CI 里跑一次全量编译确认不引入噪声 | 默认开还是关？误报怎么办（开关）？报告是否聚合？有没有给补救建议？折不动时是否保持沉默？ |
| 修改现有 pass 的算法 | 该 pass 已有的全部测试 + 新增一条覆盖本次行为变化的用例 | 该 pass 影响的所有算子；用 `lower_trace` 确认下游 diff 可控 | 改的是“判定”还是“重写”？旧行为有没有被依赖（金标准）？`maint/layout_inference/` 这类 golden 是否翻转？ |
| 新增 pass config key | 开关打开/关闭两条用例（对照 `test_..._verify_buffer_init.py:458-483`） | 不需要 | 四处是否都改了（3.4 节）？默认值是否写进 docstring？`PassConfigKey` 枚举名与 C++ 常量字符串是否一致？ |
| 调整 pass 在 pipeline 中的顺序 | 全量 `testing/python/transform/` + 端到端对拍 | **必跑** `run_perf_regression.sh`（顺序变化最容易既改性能又改正确性） | 有没有违反 2.5 表里的前置依赖？是否为某个后端单独调序（会导致后端行为分叉）？ |
| 新增昇腾专属 pass | `testing/ascend/transform/test_ascend_<名字>.py`；可用 `testing/ascend/_ir.py` 手搓 IR，不依赖前端 | 昇腾 benchmark；必要时 `run_perf_regression_ascend.sh` | 放在 `src/ascend/transform/` 而不是通用目录；`USE_ASCEND=ON` 才能编译；自动调度段之前还是之后（第 23 章）？ |
| 改适配器仓库的 pass | 适配器自己的 `testing/` + 主干侧确认没有同名漂移 | 适配器的 benchmark 脚本 | 用的是旧 FFI（`tvm._ffi._init_api`）；是否需要同步回主干；两棵树的版本号会继续分叉 |
| 只改注释 / 文档 | 不需要 | 不需要 | 注释是否与代码一致；有没有引入“最新版”这类模糊说法 |
| 只加测试 | 新测试必须先在**没有改动**的树上跑一遍并确认它是绿的 | 不需要 | 测试是否真的在测行为，而不是在测实现细节（例如断言了会变的日志前缀） |

一个必须重复的判断：**这张表的每一行都要求“先在没有改动的树上跑一遍”**。
一个从来没绿过的测试，比没有测试更危险。

---

## 11. 本章小结

1. **一个 pass = 定义 + 实现 + 注册 + 挂载 + 测试**，五件事缺一不可。缺注册 →
   `AttributeError`；缺挂载 → 静默不生效；缺测试 → 上面两条都发现不了。
2. **双语结构有明确的判断规则**：需要 C++ 独有语义（tile op、layout 对象、
   `BufferRegion`）就写 C++；改动频繁、需要快速迭代就写 Python。v0.1.15 里
   `src/transform/` 有 42 个顶层 `.cc`、`src/ascend/transform/` 有 31 个，
   Python 侧真正的 pass 是 `tilelang/transform/` 下 3 个加
   `tilelang/analysis/`、`tilelang/ascend/analysis/`、`tilelang/metal/transform/`
   下的一批检查器。
3. **FFI 的通路是“注册名 = 前缀 + 属性名”**：C++ 的
   `refl::GlobalDef().def("tl.transform.X", X)` ↔ Python 的
   `tvm_ffi.init_ffi_api("tl.transform", __name__)` ↔ `_ffi_api.X` ↔
   薄包装 `tilelang.transform.X`。改名要同时改三处，其中一处改错不报错。
4. **挂载是一行函数调用**，顺序硬编码在每个后端的 `pipeline.py` 里；通用 pass
   要在 6 条 pipeline 里各加一行。
5. **测试有四层**：IR diff 基线（最严格）、单元测试（行为）、边界条件（防误报与
   漏报）、端到端对拍（防打断真实编译）。每一层能证明的东西不同，缺一层就漏一类错。
6. **性能回归有明确的判定规则**：不是看单条数字，而是
   “≥ `MIN_COUNT`(2) 条 speedup < `THRESHOLD`(0.90)”。定位顺序是
   benchmark → 重跑排除噪声 → `pass_timing` 分编译期/产物 → `lower_trace` 找 diff。
7. **昇腾要额外处理两棵树**：主干 `src/ascend/transform/` 是主线，适配器仓库是
   另一套词汇 + 旧 FFI；`AGENTS.md` 的四条通用原则（如实报告、先验证、区分事实与
   推断、最小必要改动）与本章的纪律是同一条。

## 延伸阅读

* 本章代码与图：`code/README.md`、`figures/fig-01-*` ~ `fig-04-*`。
* 上游一手材料与逐条核验状态：见 `references.md`，源码引用 S1–S50、昇腾双树 A1–A5、
  corpus 资料 R1–R6、本章自产材料 C1–C5、上游链接 U1–U9，以及第十二节的“待补资料”。
* 想继续往下读：第 30 章讲只有 `main` 上才有的方向（多后端解耦、CuteDSL 后端、
  `T.PersistentKernel` 等），第 31 章把“从 issue 到 PR”的完整流程与三种贡献
  （改编译器 / 加后端 / 加算子）的检查清单串起来。
* 相邻章节回顾：第 20 章（读 pass 与 IR）、第 12 章（调试与性能分析）、
  第 21–23 章（layout / 流水同步 / 昇腾自动调度）、第 25 章（昇腾后端）。
