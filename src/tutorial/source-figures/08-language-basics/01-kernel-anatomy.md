# 图 1 · 一个 TileLang kernel 的构件地图

正文位置：第 2 节（概念与模型）。

这张图回答一个问题：**一个 TileLang kernel 里，哪些东西是必须显式写出来的，
哪些是编译器的活儿。** 左边一列是语言表面的六个构件，右边是后面各节的位置。

```mermaid
flowchart TB
    subgraph SRC["你在 Python 文件里写的"]
        direction TB
        A["1. kernel 定义与网格<br/>T.Kernel(gx, gy, threads=...)<br/>→ 第 3 节"]
        B["2. 数据类型<br/>'float16' / T.bfloat16 / torch.float16<br/>→ 第 4 节"]
        C["3. buffer 与内存作用域<br/>T.alloc_shared / fragment / local / var<br/>→ 第 5 节"]
        D["4. 迭代<br/>T.serial / T.unroll / T.vectorized / T.Parallel / T.Pipelined<br/>→ 第 6 节"]
        E["5. 控制流<br/>if/else、T.if_then_else、while<br/>→ 第 7 节"]
        F["6. Python 语法可用边界<br/>能用什么、不能用什么<br/>→ 第 8 节"]
    end

    subgraph IR["编译器看到的东西"]
        direction TB
        G["thread-binding frame<br/>(blockIdx.x/y/z + threadIdx.x/y/z)"]
        H["dtype 规范化<br/>字符串 / 对象 / 框架 dtype → 同一个 DataType"]
        I["buffer + storage scope<br/>shared.dyn / local.fragment / ..."]
        J["ForKind<br/>Serial / Unrolled / Vectorized / Parallel / Pipelined"]
        K["Stmt: IfThenElse / Evaluate(if_then_else) / While"]
        L["只留 trace 期能确定的部分<br/>纯 Python 的分支与循环在 trace 期展开"]
    end

    A --> G
    B --> H
    C --> I
    D --> J
    E --> K
    F --> L

    G --> OUT["一次编译的产物<br/>IR → pass → codegen → 可执行模块"]
    H --> OUT
    I --> OUT
    J --> OUT
    K --> OUT
    L --> OUT
```

## 关键分工

* **你写的是块（tile）级的意图**，不是线程级的指令。`T.Kernel` 之内默认是
  「一个块要做的全部事」，`T.Parallel` 的循环边界是**逻辑迭代空间**，
  线程映射由布局推导决定（第 3、6 节）。
* **类型、内存作用域、迭代、控制流**这四件事必须显式写出来；它们直接决定
  IR 里的 `DataType`、`storage_scope`、`ForKind` 和语句形状。
* **trace 期的 Python 与运行期的 TIR 是两种东西**：纯 Python 的 `if`、`for`、
  列表推导在构造 IR 时就求值完了；TIR 表达式写成的 `if`、`while` 才是设备上的
  控制流（第 7、8 节）。

## 事实来源

* `_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-55`（`T.Kernel`）
* `_vendor/tilelang-v0.1.15/src/ir.cc:236-262`（网格绑定与 `blockIdx` 标签）
* `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49-192`（分配函数与 scope）
* `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13-357`（循环构造子）
* `_vendor/tilelang-v0.1.15/docs/programming_guides/python_compatibility.md`

自绘，随本书正文。
