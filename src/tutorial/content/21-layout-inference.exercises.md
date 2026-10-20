# 第 21 章 习题

说明：

* 难度：`★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链较长或要动手）。
* 「需要硬件」一栏中，`否` 表示只需一台能跑 Python 的机器；`是（GPU）` /
  `是（NPU）` 表示必须用真实加速器才能得到**运行结果**（题目本身仍可用纸笔或纯
  Python 完成）。本机环境（macOS、无 GPU / NPU）做不了硬件实验。
* 所有题目只依赖第 21 章及更早的知识；涉及后续章节处会显式提示。

---

## E1 概念题 ★ 不需要硬件

用「约束满足问题」的语言描述下面这段 kernel 里 `A_shared` 与 `C_local` 的布局
推导：

```python
with T.Kernel(...) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), T.float16)
    C_local  = T.alloc_fragment((block_M, block_N), T.float32)
    T.copy(A[by * block_M, 0], A_shared)
    T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C[by * block_M, bx * block_N])
```

分别写出：每个 buffer 的**未知量**、每条语句给出的**约束**、以及冲突可能发生在
哪里。

---

## E2 概念题 ★ 不需要硬件

判断下列说法对错，并给出理由（要求引用源码位置或论文行号）：

1. `T.Layout` 的输出一定是 1 维。
2. `T.Fragment` 的 output shape 是「每个线程的寄存器槽数」。
3. 只要 `T.Layout` 是静态形状，`inverse()` 就一定成功。
4. `T.Fragment` 可以有 replicate。

---

## E3 概念题 ★★ 不需要硬件

v0.1.15 把推断分成 `kStrict` / `kCommon` / `kFree` 三级。回答：

1. 三级各自的约束松紧如何？哪一级「不自己发明布局」？
2. 用户用 `T.annotate_layout` 给的布局进入哪一级？它为什么不能被后续推断否决？
3. 为什么要分三级，而不是只跑最自由的一级？

---

## E4 概念题 ★★ 不需要硬件

`InferInFreeMode` 用「并查集分连通分量 + 逐根重推 + 代价择优」来解冲突。回答：

1. 两个 op 在什么条件下会被放进同一个连通分量？
2. 为什么要「把每个 op 都当一次根」重推，而不是只从一个固定的 op 出发？
3. 如果分量里所有尝试都失败，v0.1.15 会怎么做？在什么情况下会变成编译期硬失败？

---

## E5 概念题 ★★ 不需要硬件

比较 `RegisterCountCostModel` 与 `IOAwareCostModel`：

1. 两者的 `AttemptCost` 各字段分别统计什么？
2. 打分如何比较（给出比较顺序）？
3. 举一个「默认模型会选错、切到 io-aware 才会改变结果」的场景。

---

## E6 代码阅读题 ★ 不需要硬件

读下面这段 v0.1.15 的代码
（`_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_mma.py:42-66`）：

```python
def infer_layout(self, target, thread_nums):
    mma_emitter = self._make_mma_emitter(target, thread_nums)
    if self.is_gemm_ss():
        return {
            self.A: make_swizzled_layout(self.A),
            self.B: make_swizzled_layout(self.B),
            self.C: mma_emitter.make_mma_store_layout(self.C),
        }
    elif self.is_gemm_sr():
        return {
            self.A: make_swizzled_layout(self.A),
            self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
            self.C: mma_emitter.make_mma_store_layout(self.C),
        }
    ...
```

回答：

1. 为什么 `A`、`B` 在 shared 时用 `make_swizzled_layout`，在 fragment 时却用
   `make_mma_load_layout`？
2. `ss` / `sr` / `rs` / `rr` 描述的是什么？
3. 这段函数返回的字典，在引擎里会被当成「约束」还是「最终结果」？说明理由。

---

## E7 代码阅读题 ★★ 不需要硬件

读 `src/transform/layout_inference/layout_inference.cc:437-468` 里处理
「同一 buffer 收到两个不同布局」的流程，回答：

1. 三条出路分别叫什么？各自在什么条件下触发？
2. `ProveFragmentContains` 对「全复制」的 fragment 有什么捷径？为什么这个捷径
   是安全的？
3. 若三条出路都不成立，代码会做什么？这个行为对用户意味着什么？

---

## E8 代码阅读题 ★★ 不需要硬件

读 `AscendC0` 与分形布局的定义
（`_vendor/tilelang-v0.1.15/src/ascend/layout/ascend_layouts.cc:111-115`、
`src/ascend/layout/ascend_layouts.h:20-27`）：

1. 写出 fp16、fp32、int8、int4 四种 dtype 的 C0 值，并写出算式。
2. 一个逻辑 `[64, 48]` 的 bf16 tile，它的物理形状（含各维数值）是什么？
3. `ICHECK(256 % element_bits == 0)` 会把哪种位宽拒掉？举一个具体 dtype。

---

## E9 代码阅读题 ★★ 不需要硬件

读昇腾四个 pass 的入口与文件头注释：

1. `NormalizeAscendFractalStorage` 为什么特意声明「不初始化 padding」？下游消费方
   因此必须遵守什么纪律？
2. `InsertNd2Nz` 为什么必须跑在 `AutoSchedule` 之前？
3. `AscendInsertOOBPadding` 把 padding fill 插成独立语句、而不是塞进拷贝 lowering，
   好处是什么？

---

## E10 动手题 ★★ 不需要硬件（纯 Python）

在 `code/01-layout-algebra/layout_algebra.py` 的模式下（不依赖 TileLang），用纯
Python 实现并验证：

1. 一个 `[8, 8]` 的转置映射 $f(i,j) = (j,i)$，证明它是双射（枚举全部 64 个点，
   检查线程/槽位不碰撞）。
2. 一个 `[8, 8]` 的 padding 映射 $f(i,j) = i \cdot 10 + j$，说明它把逻辑空间映射
   到 `[0, 80)` 且**不是**满射（列出未被占用的物理下标）。
3. 给出一条「往返（round-trip）不成立」的映射，并说明为什么 v0.1.15 的
   `inverse()` 会对它抛 `NormalizeIterException`。

---

## E11 动手题 ★★★ 需要 GPU（可选）

按本章 §6.3 的步骤，在一台有 NVIDIA GPU 的机器上：

1. 把 `examples/visual_layout_inference/visual_layout_inference.py` 的
   `block_M/block_N/block_K` 从 `32/32/32` 改成 `64/64/32`，`threads` 从 128 改成
   256，重跑，记录 `C_local` 的 `Shape / Thread / Index / Replicate` 四行。
2. 与你对 `32/32/32, threads=128` 的记录比较：`Shape` 变化了吗？`Thread` 表达式
   里的常量变化了吗？
3. 解释你观察到变化与「warp 划分」之间的关系。

（本机没有 GPU，此题的运行结果**在本章内不存在**；不要把它当作已验证结论。）

---

## E12 动手题 ★★★ 需要 GPU（可选）

用 `T.annotate_layout` 故意制造一次布局冲突：对一个 `T.gemm` 的 `A_shared`
手工指定一个与 swizzle 不兼容的布局（例如一个纯粹的转置布局），观察编译器：

1. 是编译失败还是被自动合并？如果失败，失败信息是哪一条（对应源码哪一行）？
2. 如果把 annotation 换成「粒度更大的 swizzle」，行为会不会不同？为什么？
3. 记录你实际看到的报错文本（**只写你真正看到的**）。

---

## E13 动手题 ★★★ 需要 NPU（可选）

在昇腾机器上：

1. 写一个最小 `T.gemm`（L1→L0A/L0B→L0C），编译时打开 `LayoutVisual`，
   记录 **L0A / L0B / L0C** 三个 buffer 被推断出的布局表达式。
2. 把 B 的 `transpose_B` 从 `True` 改成 `False`，观察 `GemmMAD.infer_layout`
   选的 major 布局是否变（对照 `tilelang/ascend/op/gemm/gemm_mad.py:119-140`）。
3. 说明这个变化为什么是「让 reduce-K 轴落到 C0 轴」。

（本机没有 NPU，此题的运行结果**在本章内不存在**。）
