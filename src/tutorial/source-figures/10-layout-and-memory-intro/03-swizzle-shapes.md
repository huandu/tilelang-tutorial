# 图 3：三种 bank swizzle 的粒度（前 / 后对照）

对应正文第 2.4 节。TileLang 在 `tilelang/layout/swizzle.py:94-126` 暴露了三个固定
粒度的 bank swizzle 构造函数，它们都落在 `src/layout/gemm_layouts.cc` 的
`Make{Quarter,Half,Full}BankSwizzleLayout2D` 上（`src/layout/gemm_layouts.cc:442-530`）。

```text
                    一个 16B 单元的大小        行周期      异或位宽    典型用途
  ─────────────────────────────────────────────────────────────────────────────
  make_quarter_bank  32B  = 2 × 16B 单元      8 行        1 bit      TMA 的 32B 模式
  make_half_bank     64B  = 4 × 16B 单元      8 行        2 bit      TMA 的 64B 模式
  make_full_bank     128B = 8 × 16B 单元      8 行        3 bit      TMA 的 128B 模式
```

粒度由**共享内存里一个 swizzle 原子有多大**决定；测试与示例里最常用的三种是
（`tilelang/layout/swizzle.py:94-126`）：

* `make_full_bank_swizzled_layout(buffer)` ，128B；
* `make_half_bank_swizzled_layout(buffer)` ，64B；
* `make_quarter_bank_swizzled_layout(buffer)` ，32B。

> 名称里的数字是**字节宽度**，不是 bank 数。一个 bank 是 4B，所以 128B = 32 个 bank 的
> 宽度，“full bank”指这个 swizzle 原子横跨整组 32 个 bank。

## swizzle 前 / 后：同一个 8×64 fp16 tile 的物理摆放

```text
逻辑：A[8, 64] fp16

swizzle 前（纯行主序）                swizzle 后（full-bank，128B，3-bit XOR）
  row\col  0    1    2   ...  7         row\col  0    1    2   ...  7
   0      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      0      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]
   1      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      1      [ 1][ 0][ 3][ 2][ 5][ 4][ 7][ 6]
   2      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      2      [ 2][ 3][ 0][ 1][ 6][ 7][ 4][ 5]
   3      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      3      [ 3][ 2][ 1][ 0][ 7][ 6][ 5][ 4]
   4      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      4      [ 4][ 5][ 6][ 7][ 0][ 1][ 2][ 3]
   5      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      5      [ 5][ 4][ 7][ 6][ 1][ 0][ 3][ 2]
   6      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      6      [ 6][ 7][ 4][ 5][ 2][ 3][ 0][ 1]
   7      [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]      7      [ 7][ 6][ 5][ 4][ 3][ 2][ 1][ 0]
          ↑ 每行都一样：第 c 个单元             ↑ 第 s 行第 c 个单元现在住着逻辑单元 c xor s
            固定在偏移 c
```

表里每格是一个 **16B 单元**（fp16 下 8 个元素）；`…` 表示 8 个单元只是这一行 64 个
元素的前 8 个。规则只有一条：

```text
物理单元号 = 逻辑单元号  XOR  行号(在本行周期内的位置)
```

## 为什么这能消冲突，又不涨容量

```text
竖着访问一列（每个线程读自己那行的同一个逻辑列）：

  swizzle 前：所有线程都读固定的单元 c   → 同一个 bank，撞在一起
  swizzle 后：第 s 行读到单元 c xor s    → s 取遍 0..7 时落在 8 个不同单元

  映射是双射（每个 8×8 块仍是 0..7 的置换），所以：
    * 数据一个不少，只是换了位置；
    * 存储总量不变（不像 padding 要加一整列）。
```

来源公式：`_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:376-393`（
`xor2x2/xor4x4/xor8x8` 的定义）与 `:442-530`（三种 2D swizzle）。上表由
`../code/01-layout-algebra/layout_algebra.py` 复算生成。
