# 第 21 章 布局推导（Layout Inference）原理与实现

## 学习目标

读完本章，你应当能够：

1. 说清为什么每个 buffer 的**物理布局（layout）**是一个必须在编译期求解的
   **约束满足问题**，而不是一个可以随手选定的实现细节：张量核指令要求特定的
   fragment 布局，数据搬运要求特定的 swizzle 或分形排布，存储又要求对齐。
2. 用 **layout 代数**（apply / inverse / repeat / expand / reshape /
   map-region-bounds）写出一个映射，并判断它是否双射（bijective）；能说清
   `T.Layout` 与 `T.Fragment` 在代数上的差别。
3. 复述 v0.1.15 的推导流程：约束从哪里来（每个 tile op 的 `InferLayout`）、
   怎么传播（工作列表 BFS）、冲突怎么处理（包含检查 / swizzle 合并 / free-mode
   代价重排），以及「三级推断」（strict → common → free）分别允许什么。
4. 说出两个代价模型（`register-count` 默认、`io-aware` 可选）分别在优化什么，
   以及为什么「插一次布局转换」是一个要用代价衡量的取舍，而不是免费操作。
5. 解释昇腾侧 `AscendLayoutInference`、`NormalizeAscendFractalStorage`、
   `InsertNd2Nz`、`AscendInsertOOBPadding` 四个 pass 的职责，并说明 Cube 的
   分形（fractal）物理约束与 NVIDIA 张量核的布局约束差在哪里。
6. 用官方 `tilelang.tools.plot_layout` 与编译期 layout 可视化，观察一次 GEMM 的
   推导结果，并据此判断布局是否合理。

## 前置知识

* 第 10 章：`T.Layout` / `T.Fragment` 的直觉、swizzle 与 bank conflict、
  用 `plot_layout` 读一张布局图。本章把那些直觉升级为完整的推导算法。
* 第 9 章：`T.copy` / `T.gemm` / `T.Parallel` 的语义。
* 第 20 章：TIR 与两条编译管线（CUDA / Ascend）的 pass 位置；本章的
  `LayoutInference` 就是那条管线里的一个承重 pass。
* 会读 Python 的 lambda 与 C++ 的类；能读懂 `(i, j) -> (j, i)` 这类下标变换。

本章**不要求**任何 CUDA 或昇腾编程经验，也不要求读过编译器教材。所有形式化都
只用「集合、映射、约束」这三样东西。

---

## 1. 问题的由来：为什么 layout 必须被「推导」

### 1.1 三类硬约束

第 10 章讲过一句结论：layout 是接口。本章要回答的是它的下一个问题——
**接口的双方对不上时，编译期怎么把它凑上？**

先把「对不上」讲具体。TileLang 程序里，一个片上 buffer 会被多方使用，每一方对
它的物理摆法都有要求：

* **张量核指令要求特定的 fragment 布局。** 在 NVIDIA 上，`mma.sync` 的 A/B 操作数
  由 `ldmatrix` 装载，一个 16×16 的 fp16 底座矩阵必须落在 32 个 lane 的特定位置，
  每个 lane 拿到哪几个元素是固定的。TileLang 里这份要求在
  `TensorCoreIntrinEmitter.make_mma_load_layout` 里被写成一张映射表
  （`_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/mma_macro_generator.py:684-810`），
  累加器 C 的写回布局写在 `make_mma_store_layout`
  （同文件 `:814-886`）。
* **搬运要求特定的 swizzle。** 从 shared memory 喂给张量核时，如果一行 128 字节
  在同一个 bank 上反复命中，装载会被 bank conflict 拖慢。消它的办法是把地址做
  XOR 置换，也就是 swizzle。于是 shared buffer 的布局必须是「某个具体的 swizzle」，
  而不是自然行优先。TileLang 提供 32B / 64B / 128B 三档
  （`src/layout/gemm_layouts.cc:442-530`）。
* **存储要求对齐与分形。** 在昇腾上，Cube 单元的输入必须按「分形（fractal）」
  排布：底座是 16 行 × C0 列的小块，其中 `C0 = 32 字节 / 元素大小`
  （`src/ascend/layout/ascend_layouts.h:20-27`）。合入前数据必须先从 ND 转成 NZ。

这三类要求互相独立，且都作用在**同一块 buffer** 上。`A_shared` 既被 `T.copy`
写入（它有自己的自然顺序），又被 `T.gemm` 读出（它有张量核要求的顺序）；`C_local`
既被 mma 写（固定 fragment），又被 `T.copy` 读回显存（希望访存合并）。任何一处
改变，都会牵动其他地方的合法性。

### 1.2 物理布局是未知量

把上面的观察翻译成一句代数陈述：

> 对程序里每一个 fragment / shared buffer $B$，它的物理布局 $L_B$ 是**未知量**；
> 每条涉及 $B$ 的 tile op 给出一个关于 $L_B$ 的**约束**；求解器要找到一组满足
> 所有约束的 $\{L_B\}$，并让某个代价函数尽量小。

这就是「布局推导（layout inference）」要解的问题（图 1）。

```text
     tile op 1 (T.copy)          tile op 2 (T.gemm)        tile op 3 (T.Parallel)
          │                            │                            │
          │ 约束: L_B 必须能          │ 约束: L_B 必须是           │ 约束: L_B 必须让
          │ 让搬运高效 (swizzle?)     │ 某条 mma 的 fragment       │ 循环能寻址到元素
          ▼                            ▼                            ▼
     ┌──────────────────────────────────────────────────────────────────────┐
     │                     求解：给 B 选一个 L_B                              │
     │                 冲突时：包含关系？合并？插入转换？重排？                 │
     └──────────────────────────────────────────────────────────────────────┘
```

### 1.3 为什么不能「每条语句各自决定」

如果每条 op 都只按自己的口味定布局，那么同一个 buffer 在不同 op 眼里会是两张不同
的表，程序无法生成。所以物理布局是**跨语句共享的状态**：一旦某个 op 先落地了
$L_B$，其余 op 就必须在 $L_B$ 的框架下工作。于是推导天然是一个**迭代到不动点**的
过程——这也解释了为什么 v0.1.15 的实现是一个工作列表（worklist）而不是一次性的
自顶向下改写。

代价从哪来？当两个 op 的要求无法调和时，编译器有两个出路：要么选一个双方都能
接受的折中（例如把两个 swizzle 合并成粒度更小的那一个），要么在其中一个 op 旁边
**插入一次布局转换**（多读写一轮片上存储）。前者可能降低访存效率，后者直接增加
搬运量。无论哪条路，都不是免费的，需要在多个候选方案里挑小的——这就引出了
§3.4 的代价模型。

### 1.4 先手算一遍：一个最小 GEMM 的推导

为了不让上面的抽象悬在空中，先不看代码，用第 9 章那个最小 GEMM 走一遍完整的推导
（以下推理是**本书推导**，用来解释算法；具体的表达式请以 §6 的工具输出为准）：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_K, block_N), dtype)
    C_local  = T.alloc_fragment((block_M, block_N), accum)
    T.clear(C_local)
    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[by * block_M, k * block_K], A_shared)
        T.copy(B[k * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C[by * block_M, bx * block_N])
```

1. **约束从 `T.gemm` 先落地。** 它是整条链上要求最硬的 op。它的 `InferLayout`
   一次性给出三条约束：`A_shared`、`B_shared` 要 swizzle，`C_local` 要是该 MMA
   指令的 store fragment（§3.1）。
2. **约束沿 `C_local` 反向传回 `T.copy(C_local, C[...])`。** 回写语句原本可以有
   自己的向量化偏好，但它现在只能**接受** `C_local` 已有的 fragment 布局，并把
   「循环怎么分线程」这件事推给 loop layout。
3. **约束沿 `A_shared` 反向传回 `T.copy(A[...], A_shared)`。** 搬运语句必须在
   swizzle 之后写入——也就是说，它的写入顺序要为 swizzle 让路，而不是反过来。
4. **`T.copy` 自己又要能寻址。** 它内部会走 `ParallelOp` 的推断，从「已经有布局的
   buffer」里挑来源（§3.1）。这里两个来源都已经被 GEMM 定死了，所以它没有选择权，
   只能按既定的 swizzle / fragment 生成线程绑定与向量化。
5. **没有任何一步在「自由选择」。** 整个链路由一个硬约束（GEMM）定调，其余 op
   依次接受。只有当链路里**没有**这样的硬约束锚点时（例如纯 elementwise 或纯
   copy 的 kernel），第 3 步的 free-mode 才会真的去做代价决策。

这条走查解释了一件常被误解的事：**布局推导并不总在「优化」，多数时候是在传递一个
不可协商的要求。** 优化的空间只存在于那些没有硬约束的环节，以及冲突发生之后。

---

## 2. 形式化：layout 代数

### 2.1 `T.Layout`：逻辑坐标 → 物理坐标

TileLang 用一个对象描述布局（`_vendor/tilelang-v0.1.15/tilelang/layout/layout.py:12-43`）：

```python
class Layout(Node):
    def __init__(self, shape, forward_fn):
        # 为 shape 的每一维建一个 IterVar（名字 i0, i1, ...）
        # forward_fn(*vars) 返回「物理坐标」表达式（可能是一维或多维）
        ...
        self.__init_handle_by_constructor__(_ffi_api.Layout, forward_vars, forward_index)
```

也就是说，布局就是一张**下表映射**：

$$
\text{forward}: (i_0, i_1, \dots, i_{n-1}) \mapsto [f_0, f_1, \dots, f_{m-1}],\qquad
f_j = \text{forward\_index}[j].
$$

论文把它写成 $f: \mathbb{K}^n \to \mathbb{K}^m$，并强调「一个物理地址就是
$\sum_i y_i s_i$」这种线性化形式
（`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577-fulltext.md:537-545`）。
TileLang 的 `GetLinearizedForwardIndex()`
（`tilelang/layout/layout.py:93-95`，C++ 侧 `src/layout/layout.h:63-71`）就是把这个
$\sum y_i s_i$ 显式算出来的工具。

两个常用例子：

* **行优先**：`Layout([M, N], lambda i, j: i * N + j)`。
* **转置**：`Layout([4, 4], lambda i, j: (j, i))`。

输入形状由 `shape` 给出，输出形状由 `forward_fn` 的值域推出（`get_output_shape()`）。
论文里的两个示例——2D 展平成 1D、以及给缓冲区加 padding——都是同一套机制
（论文 `:555-586`）。

### 2.2 `T.Fragment`：再加「线程」和「槽位」

Fragment 是 Layout 的子类，多两样东西（`_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:12-99`）：

$$
\text{fragment}: (i_0, \dots, i_{n-1}) \mapsto
\underbrace{\text{forward\_thread}}_{\text{哪个线程}},\;
\underbrace{(\text{forward\_index}[0], \dots)}_{\text{该线程的第几个寄存器槽}},
$$

并且可以有一个 `replicate`（复制）维度：当 `replicate > 1` 时，同一逻辑元素会被
多个线程各持有一份副本（`fragment.py:60-70`）。累积器的容量、共享内存的容量都
直接由这个复制因子决定——这正是第 10 章强调「布局决定片上容量」的形式化版本。

论文对 Fragment 的定义与此一致：它的值域固定为 $\mathbb{K}^2$，两个输出分别是
「线程内的位置」和「线程本地寄存器文件的下标」，并说明为什么寄存器文件必须按
线程切分（论文 `:587-597`）。

一个容易混淆的点值得在这里点明：**`Fragment` 的 output shape 是所有线程的输出
坐标的并集形状，不是单个线程的槽位数。** 单线程槽位数由 `local_size_*` 之类的量
（见 `mma_macro_generator.py`）给出。

### 2.3 布局算子

v0.1.15 的 `Layout` 提供以下算子（Python 包装在
`tilelang/layout/layout.py:93-266`，C++ 实现在 `src/layout/layout.h:76-190`）：

| 算子 | 语义 | 代码位置 |
| --- | --- | --- |
| `map_forward_index` / `__call__` | 应用：给定逻辑坐标，算出物理坐标 | `layout.py:97-125` |
| `inverse` | 求逆：物理坐标 → 逻辑坐标 | `layout.py:217-226`，`src/layout/layout.cc:1101-1160`、`:1282-1285` |
| `repeat(dim, factor)` | 沿某个输入维重复，并新增一个「重复组」输出维 | `layout.py:127-170` |
| `expand(leading_shape)` | 在前面插入新的输入维，原样透传到输出 | `layout.py:172-215` |
| `reshape(shape, …)` | 改变输入形状，可带元素位宽重解释 | `layout.py:228-237` |
| `map_region_bounds(region)` | 把一个逻辑子区域映射到物理包围盒 | `layout.py:254-266` |

Fragment 另有 `repeat(repeats, repeat_on_thread, lower_dim_first)`、`replicate`、
`condense_rep_var`（`fragment.py:117-186`）。这些就是论文说的「扩展 Fragment 的四个
原语」的实现；论文用它把单 warp 的 `m16k16` 底座一步步扩到 4 个 warp 的
`m128k16` 块布局（论文 `:598-608`，图 6）。

### 2.4 双射、非双射与复合

**双射（bijective）** $\Leftrightarrow$ 逻辑空间与物理空间一一对应、没有碰撞也没有
空洞。求逆只在双射（或加了 padding guard 的静态形状）时有意义：
`InverseWithLevel` 对静态形状用 `arith::IterMapLevel::Bijective` 去证明，并在
「往返（round-trip）不成立」时抛 `NormalizeIterException`
（`src/layout/layout.cc:1101-1160`）。这条规则很关键：**推导器只有在能证明逆映射不
改变坐标时，才使用它**——否则宁可报错，也不悄悄产出一个错的坐标。

**非双射**的例子是 padding：给一个 `[128, 32]` 的缓冲区分配 `[128, 40]` 的物理
存储，映射 `(i, j) -> i*40 + j`（论文图 5(c) `:583-586`）。多个逻辑坐标不会撞到
同一地址，反而留下空洞。这对 bank conflict 有用，但也意味着反向查找要靠
`map_region_bounds` 做区域级而非点级的推断。

**复合（compose）**：论文明确说这些变换「是可复合的」
（论文 `:584`）。在 v0.1.15 的代码里，复合并不是一个名为 `compose` 的算子，而是
两种形式：其一是构造 `forward_fn` 时直接做函数复合，其二是在
`map_forward_index` / `map_forward_thread` 里借 `IndexMap` 链接多段映射
（`layout.py:97-125`、`fragment.py:171-186`）。MMA 装载布局正是用后者拼出来的：
先用 `IndexMap.from_func` 把一套「shared 坐标 → (lane, local)」的规则编成一个
index map，再把它包成 `forward_thread` / `forward_index`
（`mma_macro_generator.py:768-790`）。

> **待核**：v0.1.15 的 `Layout` / `Fragment` 公开 API 里没有名为 `compose` 的方法
> （`rg -n "def compose" tilelang/layout/` 无命中）。论文的「composable」是性质
> 描述，落地时是上面这两种形式。若后续版本新增了显式的 `compose` 算子，本章需
> 相应更新。

### 2.5 约束怎么表示

求解器的状态是一张表：`Map<Buffer, Layout>`，外加每个 `T.Parallel` 循环自己的
loop layout（见 `LayoutInferenceResult`，`src/transform/layout_inference/layout_inference.cc:137-142`）：

```cpp
struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;   // 每个 buffer 的物理布局
  Map<For, Fragment>  for_map;      // 每个并行循环的 loop layout
  Map<For, PrimExpr>  predicate_map; // 循环的谓词（尾部处理）
  Map<For, Bool>      padding_guard_map;
};
```

每个 tile op 的「约束」就写成 `InferLayout` 的返回值：一个 `Map<Buffer, Layout>`
形式的**建议布局**。用户也可以直接给一张初始表：`T.annotate_layout(...)`，它在
推导开始前就被读进 `annotated_layout_map_`
（`layout_inference.cc:986-1059`、`:1315`，`layout_map` 的初值在 `:562`）。

---

## 3. 算法：约束产生、传播与求解

### 3.1 约束从哪来：每个 tile op 的 `InferLayout`

约束的生产者是**每个 tile op 自己**。基类声明了这个接口
（`src/op/operator.h:188`）：

```cpp
virtual LayoutMap InferLayout(const LayoutInferArgs &layout_args, InferLevel level) const;
```

`LayoutInferArgs` 带上 target、线程范围、当前已知的 layout 表和分析器
（见 `layout_inference.cc:280-287` 的构造）。不同 op 的实现不一样：

* **GEMM**（`src/op/gemm.cc:235-268`）自己不算布局，它把活转交给一个在 Python 侧
  注册的全局函数 `tl.gemm.infer_layout`
  （`tilelang/tileop/gemm/__init__.py:12-15`），后者按选出的指令（MMA / WGMMA /
  TCGEN05 / MFMA / Ascend MAD…）调到对应的实现类。以 `GemmMMA` 为例
  （`tilelang/cuda/op/gemm/gemm_mma.py:42-66`）：

  ```python
  def infer_layout(self, target, thread_nums):
      mma_emitter = self._make_mma_emitter(target, thread_nums)
      if self.is_gemm_ss():   # A、B 都在 shared
          return {
              self.A: make_swizzled_layout(self.A),      # 约束一：shared 要 swizzle
              self.B: make_swizzled_layout(self.B),
              self.C: mma_emitter.make_mma_store_layout(self.C),  # 约束二：累加器 fragment
          }
      elif self.is_gemm_sr(): ...
  ```

  这段代码就是「张量核要求 fragment 布局」与「搬运要求 swizzle」两条约束的源头。
  注意 `reuse_existing_shared_layout` 这个开关（`gemm.cc:254-266`）：对某些指令，
  如果 shared buffer 已经由前一个 op 定下了布局，GEMM 会**让位**而不是制造冲突——
  这是内建的冲突规避。

* **`T.Parallel`**（`src/op/parallel.cc:331-...`）推断的是 **loop layout**：把循环的
  迭代空间分给线程、并决定向量化宽度。它会从「已经有布局的 buffer」里挑一个最可信
  的来源（写者优先、读者按索引数排序），让循环的划分去迁就那个 fragment 布局。
  如果循环没有任何 buffer 可依靠，它会在 `kFree` 级自己造一个布局。

* 其余 op：`T.copy` 复用并行循环逻辑（`src/op/copy.cc:553-564`）；`T.fill`
  （`fill.cc:209`）、`T.reduce_*`（`reduce.cc:205`）、`atomic_add`
  （`atomic_add.cc:126`）、`transpose`（`transpose.cc:216`）、`scan`
  （`scan.cc:147`）各自给出自己的建议。

### 3.2 传播：三级推断 + 工作列表 BFS

引擎在 `BufferUseDefCollector::Run()` 里（`layout_inference.cc:538-690`）按四步走：

```text
step 0   浮动 fragment（在 TileOp 之外被读写的，比如出现在 if 条件里）
         → 直接置为「全复制」布局 (§3.2.1)
step 1   kStrict  级：只接受显式定义的布局（用户 annotation / 已定死的要求）
step 2   kCommon  级：工作列表 BFS，传播「常见」布局
step 3   kFree    级：放宽约束，按代价模型在多个候选里择优（§3.4）
step 4   收尾：按底层存储 Var 把布局补齐到所有 alias buffer
```

其中 step 2 的传播是一个**广度优先的工作列表**：每次某个 op 产出新布局，就把它
写入 `layout_map`，再通过 `use_list_`（buffer → 使用它的 op 列表）把下游 op 重新
入队（`layout_inference.cc:455-480`）。入队带优先级：如果该 op 已经接触到的 buffer
里有一个已知布局，就 `push_front`，否则 `push_back`
（`EnqueueWithPriority`，`:762-777`）。这是一个很典型的「约束图上的不动点迭代」：
循环直到队列空。

三级的意义在于**约束的松紧**。`InferLevel` 在文档里写得很直白
（`src/op/parallel.cc:311-329`）：

* `kStrict`：最保守，只用已定义的布局，不自己发明；
* `kCommon`：允许传播常见模式；
* `kFree`：最自由，即使没有来源 buffer 也要造一个布局，用来追求性能。

用三级而不是一级，是为了让「用户显式写下的布局」永远压过「编译器猜出来的布局」。
用户 annotation 会被同时镜像进 `strict_layout_map`
（`layout_inference.cc:597-599`），在后续每一步里都作为不可否决的下界。

### 3.3 冲突：三条出路

当同一个 buffer 收到两个不同的布局建议时，引擎依次尝试（`layout_inference.cc:437-468`）：

**（一）包含检查（containment）。** 若新布局「包含」旧布局——也就是旧布局访问
的元素集合是新布局的子集——就无冲突地升级到更宽的布局。判定由
`ProveFragmentContains` 完成（`src/layout/utils.cc:443-480`）：它把小 fragment 的
每个元素用大 fragment 的逆映射反查回逻辑坐标，再验证线程集合一致。注释里点明
作者是 `@huanqicao`。注意这里对「全复制」的 fragment 有一个捷径：全复制的
fragment 永远兼容（`utils.cc:468-471`）。

**（二）swizzle 合并。** 若两个建议都是 swizzle 布局且缓冲区的最后两维是常量形状，
就取**粒度更小**的那一档（`src/layout/gemm_layouts.cc:999-1023`，
`MergeSwizzleLayouts`）。直觉上：粒度小的 swizzle 能表达粒度大的访问模式，反过来
不行。合并只会降级、不会升级，所以是一个安全的收敛方向。

**（三）自由模式重排。** step 3 里，`InferInFreeMode`
（`layout_inference.cc:1396-1550`）先把 op 按「共享同一 buffer」用并查集分组成连通
分量，然后**依次把分量里的每个 op 当作根**，从它出发重新推导一遍整条链路，用代价
模型给每次尝试打分，最后整体采用分数最好的那份方案。这就是「冲突不是靠局部改写
解决，而是靠换个起点重推」。

如果三条路都走不通，就会打印 `LOG(FATAL) << "Get different layout for ..."`
（`layout_inference.cc:463`）——这是一个**编译期硬失败**，不是静默降级。

### 3.4 代价：把「转换代价」量化

自由模式的择优需要一个分数。v0.1.15 把这件事抽象成可插拔的 `LayoutCostModel`
（`src/transform/layout_inference/layout_cost_model.h:1-90`），分数是一个二元组，
按字典序比较（`layout_cost_model.h:60-70`）：

```cpp
struct AttemptCost {
  int64_t mem{0};   // 第一关键字：字节计的访存/溢出代价
  int64_t regs{0};  // 第二关键字：寄存器槽位总数
  bool BetterThan(const AttemptCost &other) const {
    if (mem != other.mem) return mem < other.mem;
    return regs < other.regs;
  }
};
```

两个具体模型：

* **`RegisterCountCostModel`（默认）**：`mem` 只统计「寄存器数组被降级为 local
  memory」的溢出字节数，`regs` 统计全部 fragment 的寄存器槽位
  （`layout_cost_model.cc:978-992`）。它优化的是**寄存器压力**。
* **`IOAwareCostModel`（可选，`tl.layout_cost_model="io-aware"`）**：额外遍历分量里
  所有触及 global memory 的语句（fragment↔global 的 copy、以及直接访存的并行
  循环），在候选布局下估算 `max(带宽字节, 发射等价字节)`，计入 `mem`；寄存器仍是
  第二关键字（`layout_cost_model.cc:1001-1035`、`:1180-1192`）。

配置名来自 `tl.layout_cost_model`（`tilelang/transform/pass_config.py:122`，
读取在 `src/config.h:36-40`，创建在 `layout_cost_model.cc:1180-1195`）。默认
`register-count`。选择 `io-aware` 会改变同一次编译的推导结果——这本身就是
「转换代价被忽略会怎样」的现成实验。

> **注意**：`mem` 里没有单独的「插入一次布局转换」的显式项；转换的代价通过
> 「候选布局会让某条 global 访存多搬多少字节」以及「寄存器溢出多少字节」间接
> 体现。说得更准确：**v0.1.15 没有把布局转换建模成一个可加的选择边，而是在候选
> 整解之间比总代价。**

### 3.5 终止性

工作列表里的每个 op 在 `RunInferStep` 中要么产出空更新、要么把 `layout_map` 推向
一个更「宽」的状态（包含关系升级 / swizzle 粒度下降 / reducer 的 widen），因此
`kCommon` 阶段的循环是单调的，会终止。`kFree` 阶段是显式枚举（分量 × 根 × 少数
候选），次数有限。这解释了为什么 v0.1.15 用「有界枚举打分」而不是通用求解器
（对比第 23 章昇腾自动调度里用 Z3 的那条路线）。

### 3.6 为什么不是「一次性求解器」

一个自然的疑问是：既然可以把它写成一个约束满足问题，为什么不用 SMT / ILP 一次性
解出来？v0.1.15 的答案是**分层 + 局部枚举**，理由有三条：

* **约束是可扩展的、不是固定集合。** 每加一个后端（`cg` / `wgmma` / `tcgen05` /
  Ascend MAD…）就多一套 `InferLayout` 规则（§3.1 那张表）。把规则写成「全局约束
  系统」意味着每加一条指令都要回头改求解器；写成「op 自己产出建议布局」则只需要
  加一个类。
* **大部分实例根本不需要搜索。** §1.4 的走查说明：有硬约束锚点时，传播本身就给出
  唯一解，搜索是浪费。真正需要搜索的只是「没有任何 op 能定调」的那些连通分量。
* **代价函数还在演化。** `register-count` 与 `io-aware` 两个模型并存（§3.4），
  说明「什么算好」在本项目里尚无定论。把搜索限制在「有界候选 × 可插拔打分」上，
  改代价函数不会牵动求解器。

这是一条**工程取舍**：牺牲「全局最优」的保证，换来「规则可扩展、失败可解释」。
代价是：当分量很大、候选很多时，`kFree` 的枚举是 `O(分量大小 × 根数)` 量级——这也是
为什么第 23 章在昇腾侧引入了另一条基于 Z3 的调度路线来处理资源与延迟建模。

---

## 4. 实现：v0.1.15 的代码地图

### 4.1 `src/transform/layout_inference/`

```text
src/transform/layout_inference/
├─ layout_inference.cc          主引擎（1664 行）
├─ layout_cost_model.cc/.h      代价模型（1201 / 90 行）
└─ parallel_loop_layout_validator.h   并行循环布局的最终校验
```

`layout_inference.cc` 的结构：

| 部分 | 行号 | 作用 |
| --- | --- | --- |
| `LayoutInferenceResult` | `:137-142` | 推导的输出（layout_map / for_map / predicate_map） |
| `ReducerDstSteering` | `:162-236` | reducer 目的地缓冲区的「预留」机制 |
| `BufferUseDefCollector` | `:237-1550` | 收集 op、跑三级推断、负责冲突处理 |
| `RunInferStep` | `:243-518` | 调用一个 op 的 `InferLayout` 并落地结果 |
| `FinishInferQueue` | `:520-536` | 工作列表主循环 |
| `Run` | `:538-690` | 四级流程（step 0–4） |
| `VisitExpr_(CallNode)` | `:779-812` | 把 tile op 收进 `infer_list_`，记线程范围 |
| `VisitStmt_(ForNode)` | `:876-...` | 并行 `For` 转成 loop layout 的推断 |
| `InferInFreeMode` | `:1396-1550` | 连通用并查集 + 逐根重推 + 代价择优 |
| `LayoutInferencer` | `:1551-1645` | 把结果写回 IR（Block annotation / For annotation） |
| `LayoutInference()` | `:1647-1658` | pass 入口，末尾调用 `ParallelLoopLayoutValidator::Validate` |

结果不是把 IR 改写成显式线程绑定，而是**挂成注解**：`layout_map` 挂在 Block 的
`kLayoutMap` 注解上（`:1581-1586`），loop layout 挂在 `For` 的
`kParallelLoopLayout` 注解上（`:1605-1633`）。真正的线程绑定发生在更后面的
`LowerTileOp` 里。这一点很重要：**布局推导是「标注」，不是「改写」**，所以
`LayoutVisual` 这类旁路工具才能不影响语义地读出结果。

### 4.2 它在两条管线里的位置

**CUDA 管线**（`tilelang/cuda/pipeline.py:118-140`）：

```text
… → PipelinePlanning → InjectSoftwarePipeline → Simplify
  → LayoutInference        ← 本 pass
  → ReducerPlanAndMaterialize
  → LayoutVisual           ← 可视化旁路
  → LowerTileOp
```

`LayoutInference` 被刻意放在软件流水改写**之后**：这样推断看到的是最终的流水结构，
不会因为后续再拆循环而失效（见该处注释）。

**昇腾管线**（`tilelang/ascend/pipeline.py:83-101`）：

```text
… → UnrollLoopSkipVF → Simplify
  → AscendLayoutInference          ← 昇腾分支
  → ReducerPlanAndMaterialize
  → LayoutVisual
  → NormalizeAscendFractalStorage  ← §5.3
  → InsertNd2Nz                    ← §5.4
  → VFChecker
  → AscendInsertOOBPadding         ← §5.5
  → AutoSchedule（可选）…
```

### 4.3 引擎的关键数据结构

读 `BufferUseDefCollector` 时，下面几个成员是理解全局的钥匙
（`src/transform/layout_inference/layout_inference.cc:1268-1335` 一带）：

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `infer_list_` | `vector<TileOperator>` | 所有**参与推断**的 tile op，按程序顺序编号；`RunInferStep` 的下标就是它 |
| `infer_list_stmt_` | `vector<ObjectRef>` | 与 `infer_list_` 对应的原始 IR，用于报错时打印 `stmt`（见 `:289-295`） |
| `thread_index_vec_` / `thread_bounds_vec_` | `vector<PrimExpr>` / `vector<Range>` | 每个 op **所在作用域**的线程号与线程范围；昇腾的核间/核内作用域不同，靠它区分 |
| `use_list_` | `map<Buffer, vector<int>>` | buffer → 使用它的 op 下标；传播就是沿这些边把 op 重新入队 |
| `op_touched_buffers_` | `map<int, vector<Buffer>>` | op → 它接触的 buffer；用于 `EnqueueWithPriority` 判断优先顺序 |
| `buffer_data_to_buffers_` | `map<Var, Array<Buffer>>` | 同一个存储 Var 的所有 Buffer 视图（alias）；step 4 与 `propagate_alias` 靠它 |
| `annotated_layout_map_` | `LayoutMap` | 用户 `T.annotate_layout` 的初始表，同时镜像进 `strict_layout_map` |
| `floating_fragment_buffers_` | `map<Buffer, Range>` | 在 TileOp 之外被读写的 fragment，step 0 强制置为全复制 |
| `Analyzer`（每 op 一份） | `arith::Analyzer` | 该 op 作用域下的算术分析器快照，用于证明包含关系与求逆 |

有两个设计细节值得单独点出：

* **每个 op 带自己的 analyzer 快照。** 入队时 `analyzer_vec_.push_back(analyzer_.Clone())`
  （`:805-807`），因为同一个 op 在不同作用域（例如在 `k` 循环的不同迭代里）看到的
  常量范围可能不同。用一份全局 analyzer 会给出错误的证明。
* **线程范围是显式的入参。** `InferLayout` 拿到的不是「本 kernel 的线程数」，而是
  「这个 op 所在作用域的线程范围」（`LayoutInferArgs.thread_bounds`）。这是昇腾
  混合核（AIC / AIV）能在同一个 kernel 里各自推导的前提。

---

## 5. 昇腾侧：fractal / NZ / OOB

### 5.1 Cube 的物理布局约束

昇腾的矩阵单元叫 Cube，它读的是 **L0A / L0B**，写的是 **L0C**；数据从 GM 经
**L1** 再进 L0。与 NVIDIA 最大的不同是：**Cube 的输入不是「寄存器里的 fragment」，
而是「L1/L0 里按分形排好的矩阵」**。分形的定义写在源码注释里
（`_vendor/tilelang-v0.1.15/src/ascend/layout/ascend_layouts.h:20-27`）：

$$
\text{逻辑 } [\dots, \text{rows}, \text{cols}]
\;\longrightarrow\;
\text{物理 } [\dots, \text{cols}/C_0, \text{rows}/16, 16, C_0],
\qquad C_0 = \frac{32\ \text{bytes}}{\text{element size}} = \frac{256}{\text{element bits}}.
$$

`AscendC0()` 的实现就是这个公式（`src/ascend/layout/ascend_layouts.cc:111-115`），
它顺便用 `ICHECK(256 % element_bits == 0)` 把不支持的位宽直接拒掉。系数与厂商文档
一致：fp16/bf16 的 C0 是 16，fp32 是 8，int8 是 32，int4 是 64；
文档同时要求 baseM/baseN 按 16 元素对齐、baseK 按 C0_size 对齐
（`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20250107-1.md:158-160`）。

L0C 是固定累加器，分形固定为 16×16
（`src/ascend/layout/ascend_layouts.cc:254-274`，物理形状
`[.., N/16, M/16, M%16, N%16]`）。Python 侧把这些映射封成
`make_ascend_nz_layout` / `make_ascend_l0c_layout` / `make_ascend_major_k_layout` /
`make_ascend_major_mn_layout`（`tilelang/layout/ascend.py:20-115`）。

昇腾的 GEMM 约束同样由 op 自己给出：`GemmMAD.infer_layout` 按转置标志在两个
「major」布局之间选（`tilelang/ascend/op/gemm/gemm_mad.py:119-140`）——

```python
def _a_layout(buf):
    return (make_ascend_major_mn_layout(buf, k_align=k_align) if self.trans_A
            else make_ascend_major_k_layout(buf, k_align=k_align))
```

注释解释了这个选择的理由：「让每个操作数的 reduce-K 轴落到 C0 轴」——也就是把
「归约轴必须连续成 C0」这条硬件约束编码进了推断规则。

**与 NV 的差异，一句话：** NV 的张量核布局是**寄存器（fragment）层面**的约束，
shared memory 只要能喂饱 `ldmatrix`（swizzle 即可）就行；昇腾的 Cube 布局是
**片上存储（L1/L0）层面**的约束，数据必须真的以分形格式落盘——这直接导致了
`NormalizeAscendFractalStorage` 与 `InsertNd2Nz` 这两个 NV 侧没有的 pass。

### 5.2 `AscendLayoutInference`：为什么要有分支

昇腾没有直接用主干的 `LayoutInference`，而是有一个同构的分支
（`src/ascend/transform/layout_inference.cc`，1802 行；Python 入口
`tilelang/ascend/transform/__init__.py:142-145`）。它的源码注释说明了原因：**VF 区域
（`T.SimtVF` / `T.SimdVF` 这些向量块）对工作列表本身是不透明的**——遇到 VF 区域
只先访问其内部、让嵌套的 tile op 在区域自己的 lane 作用域里推断，而不是让区域内部
的访存参与外层的布局传播（`src/ascend/transform/layout_inference.cc:990-995`）。
除此之外，它与主干共享
`transform/layout_inference/layout_cost_model.h` 与
`parallel_loop_layout_validator.h`（见该文件 `:38-46` 的 include 列表），说明两级
引擎的**算法是一致的**，差别只在作用域与昇腾专属布局构造器上。

### 5.3 `NormalizeAscendFractalStorage`

（`src/ascend/transform/normalize_fractal_storage.cc`，入口 `:485-492`）

职责：在 OOB 处理与调度**之前**，把 Cube 的存储与别名「规范化」。核心是把
fragment 式的 alias（同一个 storage Var 的多个 Buffer 视图）统一成**同一个带
padding 的矩阵形状 + 同一个规范布局**，并把旧的 reshape 吸收进索引/区域里，而不是
留成一个隐藏 pitch（`:60-66` 的注释）。这样后续的容量核算与多缓冲版本展开看到的
就是**完整的存储**，不会漏算。

它只处理带了 canonical fractal 标签的布局，且缓冲区必须是 L1 / L0A / L0B / L0C
（`:75-80`）；GM/UB 与紧凑 NZ 不动。文件头明确写：**这个 pass 不初始化新增的
padding**，消费方不能假设填充元素是零（`:2-6`）——这是个容易踩的坑，见 §7。

### 5.4 `InsertNd2Nz`

（`src/ascend/transform/insert_nd2nz.cc`，入口 `:480-486`）

职责：利用刚刚推断出的 NZ 布局，把 **UB → L1 的 copy** 重写成
「scatter + post_copy」两步（文件头 `:1-3`）。也就是说，Cube 需要的 NZ 排布不是
由一次通用拷贝完成的，而是先把数据在 UB 里按 NZ 顺序写出来（scatter），再整体搬
到 L1。必须在 `AutoSchedule` 之前跑完，这样调度器才能给新产生的 op 排流水与屏障。

一个实现细节值得注意：它只支持 16/32 位浮点（`Nd2NzDTypeName`，`:35-46`），
其它 dtype 会抛 `ValueError` 并提示去 `src/tl_templates/ascend/nd2nz_copy.h` 添特化
——这是昇腾后端当前的能力边界，不是通用限制。

### 5.5 `AscendInsertOOBPadding`

（`src/ascend/transform/insert_oob_padding.cc`，入口 `:515-527`）

职责（文件头 `:1-30`）：处理会越过张量边界的 DMA 拷贝。

* 对 GM↔UB、GM→L1、L0C→GM 这类可能出界的拷贝，**把区域参数夹紧（clamp）到语义上
  合法的范围内**；真正的越界控制流由后续 `LowerDMACopy` 根据夹紧后的范围推导出来。
* 对被 padding 的 GM→L1 拷贝，**追加独立的 `tl.fill` 语句**补足 padding 区。
* 对 FP8 + MX scale 的 L1 数据，在配对的数据装载之后**自动补 K 尾部的 fill**。

为什么要在 `AutoSchedule` **之前**插入这些 fill，注释给了理由：只有这样，fill 才
能被当作一个独立的 MTE2 任务参与流水与屏障分析，从而正确排在被它填充的缓冲区被
读取之前。如果放到 `copy.cc` 的 lowering（在调度之后）里做，时序就不受分析控制了。

### 5.6 昇腾四 pass 小结

| pass | 输入状态 | 做什么 | 为什么在这个位置 |
| --- | --- | --- | --- |
| `AscendLayoutInference` | tile op 图 | 推断（VF 内部不透明） | 与 NV 同构，但作用域不同 |
| `NormalizeAscendFractalStorage` | 已推断的布局 | 统一 alias / 吸收 reshape | 让容量核算看到完整存储 |
| `InsertNd2Nz` | NZ 布局已定 | UB→L1 copy → scatter+post_copy | 必须在调度前，让新 op 被排 |
| `AscendInsertOOBPadding` | 拷贝区域可能出界 | clamp + 补 fill | 必须在调度前，让 fill 参与依赖 |

---

## 6. 实战：观察一次 GEMM 的 layout 推导

本节给出**可复现步骤**。本机（macOS / Apple Silicon，无 GPU / NPU）只做到
「仅静态核对」；下面每一步都标了在哪种机器上才能真的跑出结果。

### 6.1 直接画一张布局

先用 `plot_layout` 认识工具本身。它接受 `T.Layout` 或 `T.Fragment`
（`tilelang/tools/plot_layout.py:6-58`），输出 PDF / PNG / SVG
（`:80-110` 的 `_save_plot`）：

```python
import tilelang.language as T
from tilelang.tools import plot_layout

transpose = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(transpose, save_directory="./tmp", name="transpose_4x4", formats="png")
```

Fragment 的图会按线程着色、把每个格子标成 `T<thread> L<slot>`
（`plot_layout.py:112-...` 的 `_plot_fragment_layout`）。官方文档在
`_vendor/tilelang-v0.1.15/docs/tools/layout_visualization.md`，示例在
`_vendor/tilelang-v0.1.15/examples/plot_layout/`。

> **待核**：该示例目录里的 `layout_swizzle.py` 用
> `make_quarter_bank_swizzled_layout(8, 16, element_size)` 这种三参数形式调用，
> 但 v0.1.15 的包装函数只接受一个 buffer
> （`tilelang/layout/swizzle.py:118-126`，签名 `make_quarter_bank_swizzled_layout(buffer)`）。
> 两者对不上，疑为随版本漂移的旧写法。本章的示例代码一律用 **buffer 版 API**，
> 并已在 `_vendor/` 逐个核对签名。

### 6.2 用 MMA 装载布局看「约束长什么样」

`examples/plot_layout/fragment_mma_load_a.py` 用纯 Python 复刻了
`make_mma_load_layout` 的底座映射，并把「底座 → warp → block」三级放大画出来。
这正是论文图 6 的内容。读这张图时关注两点：**线程号（颜色）随 warp 维怎么变**，
以及**每线程的寄存器槽在放大后怎么增加**。

### 6.3 让编译器自己画：layout visualization

官方提供了编译期旁路（`tilelang/analysis/layout_visual.py:47-112`，导出为
`LayoutVisual`）。开启方式是在 JIT 上加两个 pass config
（`tilelang/transform/pass_config.py:252-256`）：

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
})
def matmul(...): ...
```

编译任意一个 GEMM kernel，每个被推断出的二维 fragment 都会被打印成
`Shape / Thread / Index / Replicate` 四行，并（在装了 matplotlib 时）存成
`./tmp/<buffer>_layout.svg`。官方示例
`_vendor/tilelang-v0.1.15/examples/visual_layout_inference/visual_layout_inference.py`
是一份 32×32×32 的 GEMM，其文件里已经记录了期望输出：

```text
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
```

**可复现步骤（需要 NVIDIA GPU + TileLang）**：

```bash
pip install "tilelang[vis]"            # tilelang + matplotlib
cd _vendor/tilelang-v0.1.15/examples/visual_layout_inference
python visual_layout_inference.py      # 需要 CUDA
# 结果：控制台打印上述四行，./tmp/ 下生成 *layout.svg
```

**在没有 GPU 的机器上能做什么**：把 `TL_LAYOUT_VISUALIZATION_ENABLE` 打开、只做
`tilelang.compile(...)` 而**不启动 kernel**——只要你的 tilelang 安装能完成
lowering。本机未安装 TileLang，所以本章的示例代码只做到 `py_compile` 与 API
核对，**没有运行过，也没有产出任何真实布局图**。

### 6.4 怎么读那四行输出

拿到一份 `Shape / Thread / Index / Replicate` 输出后，按下面的顺序读：

1. **先看 `Replicate`。** 它等于 1，说明每个逻辑元素只被一个线程持有；大于 1 就
   意味着**存储和搬运都成倍增加**。绝大多数「为什么这个 kernel 比预期慢」的问题，
   第一步都应该在这里排除。
2. **再看 `Shape`。** 左边是逻辑形状，右边是物理输出形状（可能有多个维度）。
   物理形状的维度数就是这张映射的「值域维数」——比如 `[32, 32] -> [8]` 表示 8 个
   物理下标，配合线程表达式就能算出一个线程持有多少个元素。
3. **然后看 `Thread` 与 `Index` 的常量。** 表达式里的每个常数都对应一个具体的
   warp 划分或 micro tile 尺寸。把 `threads` 或 tile 大小改一下再看这两行，变化
   最大的那个常数就是被改动的硬件参数（§6.3 的 E11）。
4. **最后与预期对照。** 你想让 C 的哪个维度连续？`Thread` 表达式里，线程号是否
   沿你希望的维度单调递增？如果连续维度和线程号对不上，回写 global 就会退化成
   散点访存——这正是 §7.2 说的「代价被忽略」。

一个提醒：这四行是**推断结果**，不是校验。它不会告诉你「这个布局是错的」，只会
告诉你「编译器最终选了它」。判断对错要靠第 10 章的直觉（bank conflict、合并访存）
与本章 §7 的陷阱清单。

---

## 7. 进阶与陷阱

### 7.1 布局冲突导致性能断崖

推导器在冲突时只会做三件事：包含升级、swizzle 降级合并、换根重推。若这三次都
落在一个「能编译但访存变差」的方案上，编译会成功，性能却会断崖——因为 swizzle
降级会重新引入 bank conflict，而包含升级会让某些线程持有本不必持有的数据。
识别信号：同一份 kernel 只改了 tile 大小或 `threads`，编译通过但变慢；打开
`TL_LAYOUT_VISUALIZATION_ENABLE` 对比两次的 `Thread` / `Index` 表达式是否变了。

### 7.2 忽略转换代价

默认代价模型只数**寄存器与溢出**（`RegisterCountCostModel`）。这意味着一个
「让 fragment 复制得更多、但让 global 访存更差」的方案不会被默认模型惩罚。
如果你怀疑是这种情况，用 `tl.layout_cost_model="io-aware"` 再编一遍，对比布局
是否改变（§3.4）。这是 v0.1.15 里最直接的 A/B 手段。

### 7.3 fragment 大小超寄存器预算

`AttemptCost.regs` 就是为此存在的。一个 `[128, 256]` 的 fp32 累加器 = 32768 个
32 位值；分给 128 个线程是 256 个寄存器/线程，已经逼近硬件上限。replicate 因子
一旦大于 1，容量按倍数上涨。**推导器只看「不溢出」，不代表能跑得快。**

### 7.4 shared 容量与对齐超限

swizzle 布局与 NZ 分形都会**改变物理占用**：C0 分组可能把最后一维向上取整
（`cols` 不是 C0 的倍数时），`NormalizeAscendFractalStorage` 还会显式加 padding 行
（§5.3），而它**不初始化**这些填充。若下游误把填充当有效数据，就是正确性 bug，
不是性能问题。昇腾侧因此把「有效区域」明确传给消费方，而不是让它们假设整块干净。

### 7.5 用用户 annotation 表达「我知道我要什么」

当推导结果不符合预期，最稳的干预是 `T.annotate_layout({buf: layout})`
（`layout_inference.cc:986-1059`）。它以 `kStrict` 级别进入，成为不可否决的下界。
代价是：如果你给错了，冲突会直接编译失败，而不会被编译器「救回来」。

### 7.6 动态形状与符号维

`InverseWithLevel` 对**含符号维**的布局会退回 `IterMapLevel::NoCheck` 并打一条
`DLOG(WARNING)`（`src/layout/layout.cc:1118-1125`）。也就是说：动态形状下逆映射
不再被证明，只由运行期 guard 兜底。写动态形状 kernel 时，凡是依赖「布局逆映射
精确」的推理，都要额外小心——第 10 章讲过 `T.copy` 在不支持的形状下宁可编译失败，
这里是同一原则的另一面。

### 7.7 reducer 与 alias 的坑

两个容易忽略的机制：

* **reducer 的 `PartialFragment` 不是普通 Fragment。** 它的副本是「加数」，不是
  「等值副本」，所以 `T.annotate_layout` 对 reducer buffer 强制要求传
  `PartialFragment`（`tilelang/language/annotations.py:49-56`）。传普通 Fragment
  会被断言拦下。
* **alias 自动补齐。** 同一个存储 Var 的多个 Buffer 视图（例如一个 `[M,K]` 的
  `[M/16, K/C0, 16, C0]` 视图）在 step 4 会被自动传播布局
  （`layout_inference.cc:621-661`）；元素位宽不同时用 `Reshape` 按**存储位宽比**
  换算（`:339-343`）。如果两个视图推导出的布局对不上，会在 `propagate_alias` 里
  直接 `ICHECK` 失败（`:344-348`）。

---

## 8. 与论文的关系

TileLang 论文里 layout 相关内容分两處，正好对应本章的两条线：

**（一）设计动机与总体机制。** 论文把 layout 列为写高性能 kernel 必须手工处理的
三类优化之一，指出它「通常需要把自然表示改造成与架构内存子系统对齐的 tiled /
blocked 格式」（论文 `:70-73`）；并在贡献里把 Layout Inference 说成
「自动并行化」的核心机制之一（论文 `:129`、`:134`）。论文明确说每个 tile operator
都要实现 `Lower` 和 `InferLayout` 两个接口，并用 `T.gemm` 作为例子——它对 shared
输入施加 swizzle 布局、对累加器用矩阵专属布局（论文 `:432-443`，图 4 中
`AShared: MakeSwizzleLayout, BShared: MakeSwizzleLayout, CLocal: MakeMMASTMatrixLayout`）。

**（二）形式化。** §4.1「Memory Layout Composition」给出本章 §2 使用的两个定义：

* Layout 是 $\mathbb{K}^n \to \mathbb{K}^m$ 的可复合、可堆叠的映射，物理地址是
  $\sum y_i s_i$（论文 `:537-545`）；
* Fragment 的输出固定为 $\mathbb{K}^2$，两个分量是「线程位置」和「本地寄存器
  下标」；并用 `repeat` / `repeat_on_thread` / `lower_first_dim` / `replicate`
  四个原语从 `m16k16` 底座拼出 `m128k16` 块布局（论文 `:587-608`，图 6）。

论文给出了**接口与代数**，但把「约束怎么传播、冲突怎么解决、代价怎么选」留给实现。
本章 §3 与 §5 补上的正是这一层：工作列表 BFS、三级约束、包含检查、swizzle 合并、
free-mode 重排、两个代价模型，以及昇腾的三条额外 pass。**这也是读源码才有价值
的地方**——论文里的 `InferLayout` 是一行签名，v0.1.15 里是 1664 行引擎加一整套
昇腾专科医生。

一个需要如实说明的边界：论文面向 NVIDIA / AMD，**没有**覆盖昇腾的分形与
ND→NZ 重排；本章 §5 完全来自 v0.1.15 与 v0.1.2.000 的源码与厂商文档，不可反推到
论文。

---

## 9. 本章小结

* **问题**：每个 buffer 的物理布局是跨语句共享的未知量；张量核指令、搬运 swizzle、
  存储对齐三方各自提出约束，构成一个约束满足问题。
* **形式化**：`T.Layout` 是逻辑坐标→物理坐标的映射；`T.Fragment` 再加线程与寄存器
  槽（并可有 replicate）；算子是 apply / inverse / repeat / expand / reshape /
  map-region-bounds；静态形状下求逆要走双射证明，往返不一致就报错。
* **算法**：约束由每个 tile op 的 `InferLayout` 给出；引擎用 strict → common →
  free 三级加工作列表 BFS 传播到不动点；冲突先试包含升级，再试 swizzle 降级合并，
  最后用 free-mode 换根重推 + 代价择优；都不行就编译期硬失败。
* **代价**：默认模型数寄存器与溢出，`io-aware` 模型再加上 global 访存字节；两者
  都用 `{mem, regs}` 字典序打分。v0.1.15 没有把「插一次布局转换」建成显式选择边。
* **昇腾**：Cube 的输入必须是分形（16×C0，C0=32B/元素）排布，L0C 固定 16×16；
  因此多了 `AscendLayoutInference`（VF 不透明）、`NormalizeAscendFractalStorage`
  （统一 alias、吸收 reshape、加 padding 且不置零）、`InsertNd2Nz`（UB→L1 改写为
  scatter+post_copy）、`AscendInsertOOBPadding`（clamp + 调度前补 fill）四个 pass。
* **工具**：`plot_layout` 画已有布局，`TL_LAYOUT_VISUALIZATION_ENABLE` 让编译器
  打印/导出它推断出的 fragment 布局。无硬件时只能做静态核对。
* **论文**：给出了 Layout / Fragment 的定义与复合性，`InferLayout` 接口与
  `T.gemm` 的布局约束；传播、冲突与代价这三层是论文没写、只能读源码的部分。

---

## 延伸阅读

* 本章的引用清单与核验状态见 `references.md`（S1–S28、A1–A4、P1–P2、C1–C4）。
* 上手练习见 `exercises.md` / `solutions.md`；可离线静态检查的代码见 `code/`。
* 相关章节：第 10 章（布局直觉）、第 20 章（IR 与 pass 位置）、第 22 章（流水线与
  同步插入，它消费本章的 loop layout）、第 23 章（昇腾自动调度：Z3 路线）、
  第 25 章（昇腾代码生成与硬件特性）。
