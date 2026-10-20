# 第 10 章 习题答案

与 `exercises.md` 编号一一对应。每题给出**解题思路**、**参考答案**与**常见错误**。
凡涉及硬件容量的数字都标注了来源；本章写作环境没有 GPU / NPU，因此 E10、E11 只给
实验设计与判据，**不提供任何实测数字**。

---

## E1 概念题 ★

**解题思路**：先把“你写的”和“硬件上真实摆法”分开，再从“跨语句一致性 / 容量 /
访存效率”三个方面解释为什么它必须显式。

**参考答案**：

1. 逻辑形状是你写在 `T.alloc_shared((M, N))` / `T.Tensor((M, N))` 里的尺寸；物理布局是
   元素实际落在哪个线程、哪个寄存器槽、哪个字节偏移上的映射。逻辑形状由你写；物理布局
   默认由编译器（`LayoutInference`）推导，也可以被 `T.annotate_layout` 覆盖。
2. 三条理由：
   * **跨语句一致性**：生产者和消费者（`T.copy` 与 `T.gemm`）对同一 buffer 可能有不同
     要求，不一致就得插入转换或报错；
   * **容量**：replicate / padding / swizzle 都直接改变片上占用；
   * **访存效率**：bank conflict、合并访存、张量核 operand 排布都由它决定。
3. 因为 `T.alloc_fragment((128, 128))` 描述的是**整个线程块**的寄存器切片；编译器用
   layout inference 把它切成每线程若干元素。128×128 = 16384 个元素摊到 128 个线程，
   是每线程 128 个槽（具体数字取决于推导出的 `Fragment`），不是每线程 128×128。

**常见错误**：把 fragment 当成“每线程私有数组”（那是 `T.alloc_local`）；或者以为
逻辑形状就是物理布局。

---

## E2 概念题 ★

**解题思路**：直接照 `layout.py` / `fragment.py` 的签名回答。

**参考答案**：

1. 输入是逻辑下标（一组 `i0, i1, ...`），输出是物理下标（可以是一个表达式或一个列表）。
2. 多两样：
   * `forward_thread(i0, i1, ...) -> 线程号`（线程映射）；
   * `replicate`（同一份数据在多少个线程上各存一份）。
   前者解决“谁持有”，后者解决“要不要在多个线程上放副本”。
3. `replicate = 4` 表示同一份逻辑数据在 4 个线程上各有一份相同副本。需要它的场景是：
   **所有线程都必须读到同一份小数据**（如索引表、mask、小权重），此时复制比广播/重算
   更省事；`make_fully_replicated_layout_fragment` 就是干这个的
   （`tilelang/layout/swizzle.py:168-189`）。

**常见错误**：把 `replicate` 与 `repeat` 混为一谈。`repeat` 是**铺开**（输入变大、
输出多一维），`replicate` 是**复制**（同一份数据存多份）。

---

## E3 概念题 ★★

**解题思路**：把 `forward_thread` 当函数算一遍。

**参考答案**：

1. `thread = j//2 + 4*i`，`i, j ∈ [0,8)`：`i=7, j=7` 时 `thread = 3 + 28 = 31`，
   所以用到 **32 个线程**（0..31）。64 个元素 ÷ 32 线程 = **每线程 2 个元素**。
2. 线程 5 满足 `j//2 + 4*i = 5`：
   * `i=0`：`j//2 = 5` → `j ∈ {10, 11}`，超出 `[0,8)` → 无；
   * `i=1`：`j//2 = 1` → `j ∈ {2, 3}` → `(1,2)`、`(1,3)`；
   所以线程 5 持有 `(1,2)` 与 `(1,3)`（槽号分别是 `j%2 = 0` 和 `1`）。
3. 布局本身描述的是“元素 → 线程”的固定映射，`thread` 最大到 31，所以它只用到 32 个
   线程；放进 64 线程的 block 里，剩下 32 个线程在这份 fragment 上没有元素。这不代表
   编译错误，但意味着**这份 fragment 的并行度只有 32**，另一半线程会闲置，除非它们
   参与别的计算。真正要扩大并行度，得换一个 `Fragment`（比如换 MMA 原子或改用
   `replicate`）。

**常见错误**：答“不能用，因为线程数是 32”。布局不要求用满全部线程；是否用满是
性能与设计问题，不是合法性判定（合法性看的是索引是否越界、映射是否自洽）。

---

## E4 概念题 ★

**解题思路**：回忆 `allocate.py` 的签名与 H100 规格。

**参考答案**：

1. |
   | 层级 | 语言原语 | 典型用途 |
   | --- | --- | --- |
   | global | 传给 kernel 的 `T.Tensor` / `T.alloc_global` | 输入输出、workspace |
   | shared | `T.alloc_shared` | 线程块共享的 tile，给张量核喂数据 |
   | fragment | `T.alloc_fragment` | 累加器、tile 级中间结果（寄存器） |
   | local | `T.alloc_local`、`T.alloc_var` | 每线程私有数组 / 标量 |
2. H100（cc 9.0）：共享内存容量 **每 SM 228 KB**，**每 block 最多 227 KB**
   （CUDA 每 block 预留 1 KB）；静态 shared 默认上限 **48 KB**，超过要用动态分配并显式
   申请。（来源：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md:50-51,226-228`，
   厂商文档口径。）
3. padding 是在行与行之间多插入若干元素，它们同样占存储。32×32 fp32 行长 32→33：
   每行多 1 个 4B 字，共 32 行 → **多 128 B**（约 3%）。

**常见错误**：把“每 SM”和“每 block”混为一谈；或把 48 KB 当成“最大就是 48 KB”
（其实动态分配可以更高，只是需要显式 opt-in）。

---

## E5 概念题 ★★

**解题思路**：把“地址 → bank”写成一个取模式，冲突就用“有多少个线程落到同一个余数”
来判断。

**参考答案**：

1. 行长 32 个字（4B）时：
   * **模式 A**（线程 `t` 读 `[0][t]`）：`addr = t`，`bank = t mod 32 = t`
     → 32 个线程恰好铺满 32 个 bank → **无冲突**。
   * **模式 B**（线程 `t` 读 `[t][0]`）：`addr = t * 32`，`bank = (32t) mod 32 = 0`
     → 全部落在 bank 0 → **32 路冲突**。
2. 行长改成 33：模式 B 的 `addr = 33t`，`bank = (33t) mod 32 = t`（因为
   `33 ≡ 1 (mod 32)`）→ 铺满 32 个 bank → 冲突消失。
3. 异或 swizzle 不改行长，只把“第 `s` 行第 `c` 个 16B 单元”搬到第 `c xor s` 个单元。
   `s → c xor s` 是双射，所以每行的单元仍是 0..width-1 的一个排列（数据不丢）；
   而固定一个逻辑列 `c` 时，8 行给出 `{c xor 0, ..., c xor 7}`，在分组宽度内互不相同
   （128B 时是 8 个不同值）→ 列方向的冲突被摊开。

**常见错误**：只回答“padding 解决了”而不写 `mod` 推导；或忘了 swizzle 的行周期是
**8 行**（不是任意行数），把 8 行模型套到 16 行的 tile 上而忘记周期性。

---

## E6 代码阅读题 ★★

**解题思路**：逐位拆开 `xor8x8`；再代入 fp16 的 `element_size`；`ICHECK` 是硬断言。

**参考答案**：

1. `xor8x8(i, j)` 把两个数按二进制逐位拆分后做模 2 加（即按位异或），所以等价于
   **`i` 与 `j` 在低 3 位上的按位异或**（`xor4x4` 处理次低位、`xor2x2` 处理最低位）。
2. fp16 的 `element_size = 16`（bit）→ `vector_size = 128 / 16 = 8` 个元素 = 8×2B =
   **16B**。这正是 shared memory“一个 swizzle 原子”的传输粒度。
3. `stride == 4` 这个特例是给 `tile::gather4/scatter4` 用的截断 4 行周期（注释
   `// stride==4 is a truncated 4-row period used by tile::gather4/scatter4`，
   `src/layout/gemm_layouts.cc:508-509`）：`s = i % 8 ∈ [0,4)` 是 8 行 XOR 模式的合法子集，
   与 TMA 在硬件里逐行施加的模式一致。
4. fp16 且 `continuous = 8` 时，`vector_size = 8`，`vector_size * 8 = 64`，
   `8 % 64 = 8 ≠ 0` → 第二个 `ICHECK` 触发，构造失败。也就是说：**128B swizzle 要求
   `continuous` 至少是 64 个 fp16 元素（= 128B）**，否则只能用更小粒度的 swizzle。

**常见错误**：把 `stride` / `continuous` 理解成“二维数组的两个长度”以外的意思。
在这份代码里它们确实就是 tile 的行数（stride）与列数（continuous）。

---

## E7 代码阅读题 ★★

**解题思路**：把 `annotate_layout` 的 5 条分支逐条对应到语义。

**参考答案**：

1. 键是 **buffer 对象**（源码取 `buffer.data` 作为存储键）；值的类型可以是
   `Layout`/`Fragment` 对象，**也可以是一个回调函数**（会被包装成
   `Layout(buffer.shape, fn)`）。
2. fragment buffer 必须配 `Fragment`，否则断言失败：
   `assert isinstance(layout, Fragment), "for Fragment ..., layout must be a Fragment"`。
   也就是说“只给下标映射、不给线程映射”在 fragment 上是非法的。
3. 因为 reducer 的多个 replica 是**待求和的加数**（需要在 finalize collective 里合并），
   不是“数值相等的副本”。如果按普通 `Fragment` 语义理解，会把加数当成同值副本，
   造成错误的优化。所以强制用 `PartialFragment`，它编码了 `combine_size` 这条边界。
4. 它成为 block 的 attribute `layout_map`，被 `LayoutInference` pass 读走，作为
   **约束的起点**（严格快照），随后与 op 推导出来的布局合并或冲突（4.1 节）。

**常见错误**：以为注解“只是提示”。对 fragment/reducer，它是**硬约束**，写错会断言
失败或与推导结果冲突而报错。

---

## E8 代码阅读题 ★

**解题思路**：看键名，不看函数名。

**参考答案**：

1. 键名是 **`threadblock_swizzle_pattern`**。“threadblock”直接说明它影响的是**线程块**
   （block）的调度/栅格化顺序，而不是 buffer 的排布。
2. **不能**。它只重排 block 的启动顺序（目的是 L2 复用），既不改 shared memory 的
   物理排布，也不改任何地址表达式，因此与 bank conflict 无关。
3. 用 `T.annotate_layout` 配 swizzle 构造器，例如：

   ```python
   from tilelang.layout import make_swizzled_layout
   T.annotate_layout({A_shared: make_swizzled_layout(A_shared)})
   ```

   或 CUDA 侧：`from tilelang.cuda.intrinsics import make_mma_swizzle_layout`。

**常见错误**：把两个“swizzle”当成同一件事。这是本章点名的头号混淆点。

---

## E9 动手题 ★★

**解题思路**：跑脚本看第一节两行的 `distinct banks` / `worst per-bank`；第二问自己改
访问模式（把 `t*stride` 换成 `t`）；第三问照抄 C++ 的递归写法加一位。

**参考答案**：

1. 脚本输出为：
   * `stride=32`：`distinct banks=1`，`worst per-bank=32` → **32 路冲突**；
   * `stride=33`：`distinct banks=32`，`worst per-bank=1` → **无冲突**。
2. 模式 A（读第 0 行、线程 `t` 读第 `t` 列）的地址是 `addr = 0 * stride + t = t`，
   与 stride 无关 → `bank = t mod 32` 恰好铺满 32 个 bank → **两种 stride 都无冲突**
   （`distinct banks=32, worst per-bank=1`）。这也说明：**冲突来自“访问方向”和
   “行长”的组合，而不是行长本身**。
3. 加一个“256B，4-bit XOR”粒度时，只要 `xor16x16` 仍然实现**真正的按位异或**
   （`c xor s`）且行周期仍是 8，那么 `verify_swizzle_bijection` **仍然成立**：
   对每个固定的 `s`，`c → c xor s` 是 0..15 的排列；整体 8×16 个 `(s, c)` 映射到
   8×16 个不同 `(s, c')`。注意“一列在 8 行内的散布”仍然只有 8 个值（受行周期限制），
   所以 `column_spread` 类的不变量应按“min(8, 分组宽度)”来写，而不是硬编码 8。

**常见错误**：把“无 padding 就一定有冲突”当定律；或写了 `xor16x16` 但实现成
加法/取模而不是按位异或，导致映射不再是排列。

---

## E10 动手题 ★★ 需要硬件（GPU）

**解题思路**：这是“读，不改算法”的题。关键是**只报告实际看到的东西**。

**参考答案（步骤，不给结果）**：

1. 在带 GPU 的机器上装上 TileLang v0.1.15 与 PyTorch(CUDA)。编译
   `annotate_swizzle_gemm.py` 里的 kernel（只编译、不跑，先拿文本）。
2. 抄下每个 `C_local inferred layout:` 块里的 `Shape` / `Thread` / `Index` / `Replicate`。
3. 注释掉那两行 `T.annotate_layout(...)`，重新编译，再抄一遍，逐字段比对。
4. 在 `kernel.get_kernel_source()` 的字符串里搜索是否出现 XOR 风格表达式（形如
   `(a ^ b)` 或等价的 `(a + b) % 2` 组合），并对照图 3 判断粒度。

**判据**：只要“两次打印的 Thread/Index 表达式”有差异，就说明注解确实改变了布局；
差异为空则说明默认推导本来就是这个布局（也很正常）。

**常见错误**：把“编译通过”当成“布局被改变”；或用别处的截图当自己的证据。

---

## E11 动手题 ★★★ 需要硬件（GPU）

**解题思路**：这是一道**实验设计**题，评分点是“单变量、有假设、有判据”。

**参考答案（框架）**：

1. 变量：只切换 `make_swizzled_layout(A_shared, k_major=...)` 的取值（或换粒度），
   其余全部固定。
2. 固定项：`block_M/block_N/block_K`、`num_stages`、线程数、输入张量的形状与数值。
3. 判据：
   * 正确性：与 `torch.matmul` 在给定 `rtol/atol` 下 `assert_close` 通过；
   * 性能：固定 warmup 次数 + 重复次数，取中位数/最小值的口径要写清楚（不要只报一次
     的最好成绩）。
4. 假设要先写：比如“这个 GEMM 的 shared 写入本来就是冲突的，所以加 swizzle 应该更快”，
   或者“这个 shape 下 shared 访问不是瓶颈，加 swizzle 只会更慢”。
5. 结论要限定条件：“在 GPU 型号 X / CUDA Y / TileLang v0.1.15 / 形状 Z 下观察到……”。

**常见错误**：同时改多个变量；只跑一次就下结论；把一次噪声当成趋势；忘了在结论里写
限定条件。

---

## E12 概念题 ★★

**解题思路**：照 2.5 节与 `ascend.py` / `ascend_layouts.h` 回答。

**参考答案**：

1. `[rows, cols] → [cols/C0, rows/16, 16, C0]`；`C0 = 32 byte / element_size`
   （bf16/fp16 是 16，fp8 是 32，fp32 是 8）。
2. ND = 普通行主序（row-major）二维数组；NZ = 分形排布，即 `[cols/C0, rows/16, 16, C0]`。
   需要 ND→NZ 是因为 Cube 单元要求操作数在 L1 里已经是分形，而 GM/UB 上的数据是
   行主序，必须重排搬运（pass `InsertNd2Nz`，见 `tilelang/ascend/pipeline.py:91`）。
3. 因为分形要求 `rows` 是 16 的倍数、`cols` 是 C0 的倍数，真实 shape 往往不满足，
   于是要在逻辑 region 之外补零把物理块填满、并保证越界读得到 0
   （pass `AscendInsertOOBPadding`）。补零区在对应 region 里会被裁掉，**不影响正确结果**。
4. NVIDIA 侧的布局由编译器**推导**（layout inference），用户可覆盖；昇腾侧的 fractal
   是**硬件规定的**，编译器做的是“归一化 + 重排 + 补零”去迎合它。

**常见错误**：把 `C0` 当成固定的 16（fp8 时是 32、fp32 时是 8）；或以为 padding 的
零会参与最终输出。
