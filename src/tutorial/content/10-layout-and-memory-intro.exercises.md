# 第 10 章 习题

说明：

* 难度：`★`（基础）/ `★★`（要综合本章内容）/ `★★★`（推理链较长或要动手）。
* 「需要硬件」一栏：`否` = 一台能跑 Python 的机器即可；`是（GPU）` = 需要 NVIDIA GPU
  才能完成实验。
* 所有题目只依赖第 10 章及更早的知识；涉及后续章节的地方会显式提示。
* 本章写作机没有 GPU / NPU，因此所有「需要硬件」的题都只要求你设计实验并写出**判据**，
  不要求你现在就报告某个数字。

---

## E1 概念题 ★ 不需要硬件

1. 「逻辑形状」和「物理布局」分别是什么？各自由谁决定？
2. 本章说布局是「接口」而不是「实现细节」，给出三条理由，每条用一句话。
3. 下面这句话为什么是错的：「`T.alloc_fragment((128, 128), "float32")` 表示每个线程
   持有 128×128 个 float。」

---

## E2 概念题 ★ 不需要硬件

1. `T.Layout` 的 `forward_fn` 的输入和输出分别是什么？
2. `T.Fragment` 比 `T.Layout` 多描述了哪两样东西？各自解决什么问题？
3. 一个 `Fragment` 的 `replicate` 为 4 意味着什么？举一个本章提到的、需要 replicate
   的场景。

---

## E3 概念题 ★★ 不需要硬件

已知 `MakeGemmFragment8x8` 的定义是
`forward_thread = j // 2 + 4 * i`、`index = j % 2`（`i, j ∈ [0,8)`）。

1. 这个 8×8 fragment 一共用到多少个线程？每个线程持有几个元素？
2. 线程 5 持有哪几个逻辑元素 `(i, j)`？请逐个写出来。
3. 如果把同一个 8×8 tile 放到一个 **64 线程**的 block 里，这个 fragment 布局还能成立
   吗？为什么？

---

## E4 概念题 ★ 不需要硬件

1. 列出 TileLang 语言层 `T.alloc_*` 对应的四个内存层级（global / shared / fragment /
   local），并各说一句它们的典型用途。
2. 在 NVIDIA GPU 上，shared memory 的容量上限是「每 SM」和「每 block」两个不同口径。
   本章给出的 H100 数字分别是多少？静态 shared 的默认上限又是多少？
3. 为什么说「padding 会占容量」？把 32×32 fp32 tile 的 padding 从 0 改成 1（行长
   32→33），多占多少字节？

---

## E5 概念题 ★★ 不需要硬件

一个 warp（32 线程）访问一个 32×32 的 `float` 共享 tile。

1. 比较两种简化访问模式，哪个会冲突？各自给出 bank 的推导式：
   * **模式 A**：线程 `t` 读 `[0][t]`（第 0 行、第 `t` 列，即沿行扫）；
   * **模式 B**：线程 `t` 读 `[t][0]`（第 `t` 行、第 0 列，即沿列扫）。
2. 把行长从 32 改成 33，为什么冲突就消失了？用 `mod 32` 说明。
3. XOR swizzle 在不改行长、不加容量的前提下是怎么做到的？请写出「第 s 行第 c 个单元」
   被搬到哪里的公式。

---

## E6 代码阅读题 ★★ 不需要硬件

阅读下面这段来自 `src/layout/gemm_layouts.cc` 的代码（已略去无关行）：

```cpp
PrimExpr xor8x8(const PrimExpr &i, const PrimExpr j) {
  PrimExpr i0 = FloorMod(i, 2);  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor4x4(i1, j1) + xor2x2(i0, j0);
}

static Layout MakeFullBankSwizzleLayout2D(int stride, int continuous, int element_size) {
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride == 4 || stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 8) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);   PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c  = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}
```

1. `xor8x8(c, s)` 在数学上等价于什么运算？（提示：看它怎么拆位）
2. 对 fp16，`vector_size` 等于多少？为什么是「16B 单元」？
3. `stride == 4` 这个特例是给谁用的？（提示：注释里提到 gather4/scatter4）
4. 假设 `continuous = 8`（fp16），这个函数能构造成功吗？请解释 `ICHECK` 会不会触发。

---

## E7 代码阅读题 ★★ 不需要硬件

阅读 `tilelang/language/annotations.py` 里的 `annotate_layout`（本章 3.2 节已给出全文
摘录）：

1. `layout_map` 的键是什么类型的对象？值可以是哪两类东西？
2. 如果 `buffer` 是一个 fragment，而你传进去的 `layout` 是一个普通 `T.Layout`（没有线程
   映射），会发生什么？
3. 为什么 reducer（`T.alloc_reducer`）要求 `PartialFragment` 而不是普通 `Fragment`？
4. 函数最后返回的 `sblock_attr({"layout_map": ...})` 去了哪里？（提示：本章 4.1 节）

---

## E8 代码阅读题 ★ 不需要硬件

阅读 `use_swizzle`：

```python
def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    if order == "row":
        device_func = "rasterization2DRow"
    elif order == "column":
        device_func = "rasterization2DColumn"
    elif order == "mlx":
        device_func = "rasterization2DMLX"
    else:
        raise ValueError(f"Unsupported swizzle order: {order}")
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", tvm_tuple(device_func, panel_size))
```

1. 它写入的注解键名是什么？从这个键名能看出它影响的是什么？
2. 它能不能消除 shared memory 的 bank conflict？为什么？
3. 想消除 bank conflict，应该用哪个 API？给出调用示例。

---

## E9 动手题 ★★ 不需要硬件

运行 `code/01-layout-algebra/layout_algebra.py`（纯标准库）。

1. 报告第 1 节里 stride=32 与 stride=33 两种情况的 `distinct banks` 与
   `worst per-bank`。
2. 把第 1 节的模型改成「warp 读第 0 行的一整行，线程 `t` 读第 `t` 列」，重新计算
   `distinct banks`，并解释结果。
3. 修改脚本里的 `SWIZZLE_BITS`，加入一个假的「256B，4-bit XOR」粒度，观察
   `verify_swizzle_bijection` 是否仍然成立。（提示：`xor2x2/xor4x4/xor8x8` 是从
   C++ 抄来的；自己写 `xor16x16` 时要保持「按位异或」的语义。）

---

## E10 动手题 ★★ 需要硬件（GPU）

在一台有 NVIDIA GPU 的机器上，编译 `code/03-annotate-layout/annotate_swizzle_gemm.py`
的 kernel（先只编译、不运行）。

1. 打开 `TL_LAYOUT_VISUALIZATION_ENABLE`，把 `C_local inferred layout` 的文本抄下来。
2. 把 `T.annotate_layout({A_shared: ..., B_shared: ...})` 那两行删掉，重新编译，对比
   打印出来的布局有没有变化。
3. 用 `kernel.get_kernel_source()` 找出地址表达式里是否出现了异或（或等价的取模组合），
   并说明它对应哪一种 swizzle 粒度。
4. 只报告你**实际看到**的东西；不要用别人的截图或网上的数字。

---

## E11 动手题 ★★★ 需要硬件（GPU）

设计一个实验，验证「同一个 kernel，只改一个布局注解，正确性不变而耗时变化」。

要求：

1. 选定一个 kernel 与一个可切换的注解（例如 `make_swizzled_layout` 的
   `k_major`，或加/不加 `make_mma_swizzle_layout`）。
2. 固定所有其它变量（tile 大小、stage 数、线程数、输入数据）。
3. 给出**正确性判据**（例如与 `torch.matmul` 在给定容差下比较）与**性能口径**（计时
   方式、重复次数、是否包含 warmup）。
4. 预先写下你的**假设**（哪个更快、大概为什么），再跑，再解释结果与假设是否一致。
5. 明确说明：结果只对你这台机器、这个版本成立。

（提示：本章只提供观察工具，不提供性能结论；第 11、12、28 章会给更系统的调优与
性能模型方法。）

---

## E12 概念题 ★★ 不需要硬件

1. 昇腾的 fractal 存储把逻辑 `[rows, cols]` 映射到物理的哪四个维度？`C0` 由什么决定？
2. ND 和 NZ 分别指什么？为什么需要 ND→NZ 这一步？
3. 为什么需要 OOB padding？补零区会不会影响正确结果？
4. 用一句话说明「NVIDIA 侧的布局」和「昇腾侧的布局」在**由谁决定**这件事上的根本区别。
