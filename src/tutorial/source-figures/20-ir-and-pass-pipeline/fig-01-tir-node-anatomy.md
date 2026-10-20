# 图 1：TIR 节点解剖（自绘）

一个 TileLang kernel 在编译期就是一棵 TIR 树。下面这张图把「前端 dump」
里能直接看到的节点类型与它们的从属关系画出来。节点名与层级来自
`_vendor/tilelang-v0.1.15/examples/eager_jit/eagerjit.en.ipynb` 里
`print(prim_func)` 的真实输出（见 `code/02-ir-snippet-extract/`）。

```text
IRModule                        一个「模块」= {GlobalVar -> BaseFunc} 的字典
│                               一个 kernel 通常只有 "main" 这一个函数
└── PrimFunc  (global_symbol="main", target=…)
    │                          一个函数 = 参数列表 + 一个 body
    │
    ├── params / buffer_map    T.handle 参数 -> Buffer（T.match_buffer 建立）
    │       x_handle : T.handle
    │       x        = T.match_buffer(x_handle, (2,), strides=(1,))
    │
    └── body  (Stmt)
        │
        ├── AttrStmt("thread_extent")      launch 维度（物化之后是 threadIdx.*）
        │   └── For(kind=THREAD_BINDING)   grid 维度（blockIdx.x）
        │
        └── SBlock("tilelang_root")        kernel 主体的作用域块
            ├── reads  / writes            声明式读写集合（给调度/同步用）
            ├── alloc_buffers              块内分配（scope 决定内存层级）
            ├── annotations                编译期元数据（layout、launch、pipeline…）
            └── body  (SeqStmt)
                │
                ├── BufferStore   x[1] = 1.0        「写」
                ├── BufferLoad    A[i]              「读」
                ├── LetStmt       _tmp: int32 = i   「模板变量绑定」
                ├── For           serial / parallel / vectorized / unrolled
                └── Call          tl.tileop.* / tirx.* / tl.*   内建与 tile op

叶子（PrimExpr）：
    Var        变量（i、tx、N…）
    IntImm / FloatImm        常量
    Add / Mul / FloorDiv / Mod / …
    Ramp(base, stride, lanes)  向量索引（向量化的载体）
    Cast / Select / Call
```

三个容易混的点：

1. **`T.reads()` / `T.writes()` 是声明，不是指令。** 它们是给 layout 推导与
   同步分析用的读写集合；拿到它们不等于发生了内存访问。
2. **`thread_extent` 是 `AttrStmt`，不是 `For`。** 网格（`blockIdx`）才是
   `For(kind=THREAD_BINDING)`；线程域（`threadIdx`）在物化之后是
   `AttrStmt("thread_extent")` 把 `Var` 绑到线程标签上。
3. **`scope` 决定一切。** `alloc_buffer(..., scope="shared")` 与
   `scope="local.fragment"`、`scope="local.var"`、`scope="shared.dyn"` 在后继
   pass 里走完全不同的布局、同步与内存规划路径。
