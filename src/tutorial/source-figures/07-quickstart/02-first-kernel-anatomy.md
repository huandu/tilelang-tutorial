# 图 2 · 一个 kernel 的四个部件

正文位置：第 4、5 节。

## 2.1 四个部件与它们的落点

```text
  Python（你写的）                          编译器兑现成                      生成的 CUDA（节选）
  ───────────────────────────────────────   ──────────────────────────────    ────────────────────────────
  ① 网格   T.Kernel(ceil(N/256),            网格 → 轮次/核编号                  extern "C" __global__ void
           threads=256) as bx                threads → 每块线程数                __launch_bounds__(256) main_kernel(...)
                                              bx → blockIdx.x                    ← gridDim.x = ceil(N/256)

  ② buffer T.alloc_fragment((256,), dtype)  LayoutInference 决定每线程              bfloat16_t A_local[8];
            或 T.alloc_shared(...)           持有哪几个元素（shared 落成             __shared__ ...（若 shared）
                                              片上数组）

  ③ copy   T.copy(A[bx*256], A_local)       LowerTileOp → 向量化后的 LDG/STG    *(uint4*)(A_local+0) =
           T.copy(A_local, C[bx*256])         或 TMA / cp.async                    *(uint4*)(A + ...);

  ④ par    for i in T.Parallel(256):        把块级并行域映射到线程 + 向量化      if (threadIdx.x < 32) {
             C_local[i] = A_local[i]+...                                          ... nv_bfloat162 加法 ...
                                                                                   }
```

## 2.2 一次 1-D 向量加法的「块级」视角

```text
  全局向量 A / B / C（长度 N）
  ├──────────────┼──────────────┼──────────────┼──────────────┤
   block 0        block 1        block 2          ...
   (bx = 0)       (bx = 1)       (bx = 2)

  每个 block 负责 256 个连续元素（block_N = threads = 256）：
  ┌────────────────────────────────────────────────────────────┐
  │  T.Parallel(256)  →  编译器决定怎么铺到 256 个线程上         │
  │  例：每线程 8 个元素 + 128-bit 向量化 → 只用 32 个线程       │
  │  例：每线程 1 个元素                   → 用 256 个线程       │
  └────────────────────────────────────────────────────────────┘
```

## 2.3 为什么是「block 级」而不是「thread 级」

```text
  CUDA（thread 级）                             TileLang（block 级）
  ─────────────────────────────────────────    ────────────────────────────────────────
  int i = threadIdx.x + blockIdx.x*blockDim.x;  for i in T.Parallel(256):
  if (i < N) c[i] = a[i] + b[i];                    c_local[i] = a_local[i] + b_local[i]

  你要自己算线程号、边界、                     你只描述「这块的 256 个元素各做一次」；
  要不要向量化、怎么合并访存。                 线程映射 / 向量化 / 边界由编译器补。
  （想要线程级也可以，用 T.get_thread_binding()，属 expert 层。）
```

## 事实来源

* `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455`（`T.Kernel` 的
  "grid of tile programs" 语义）。
* `_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-52`（CUDA 方言的
  `threads=`、`T.get_thread_binding()`）。
* `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13-80`（`T.Parallel` 是块级
  并行域；布局由 `LayoutInference` 推导）。
* `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-132`（`T.copy` 是 tile op，
  低级循环在 lowering 时构造）。
* `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:39-41,120-171`
  （"block level, not thread level" 的官方表述与两份生成结果）。
