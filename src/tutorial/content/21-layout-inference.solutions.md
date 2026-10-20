# 第 21 章 习题答案

与 `exercises.md` 一一对应（E1–E13）。每题给「思路 → 参考答案 → 常见错误」。
凡引用源码处都给出 `_vendor/` 路径与行号；凡下结论处都标明它是「源码事实」还是
「本书推导」。

---

## E1 概念题 ★

**思路**：套 §1.2 的三段式（未知量 / 约束 / 冲突）。

**参考答案**：

* **未知量**：$L_{A\_shared}$（shared 布局）与 $L_{C\_local}$（fragment 布局）。
* **约束**：
  * `T.copy(A[...], A_shared)`：写出侧希望自然的行优先/向量化顺序（本书推导：
    见 §3.1 对 `Copy`/`Parallel` 的说明）；
  * `T.gemm(A_shared, B_shared, C_local)`：**硬约束**——$L_{A\_shared}$ 必须是
    swizzle 布局，$L_{C\_local}$ 必须是该 MMA 指令的 store fragment 布局
    （源码事实：`gemm_mma.py:42-50`）；
  * `T.copy(C_local, C[...])`：$L_{C\_local}$ 决定回写 global 的寻址方式
    （$L_{C\_local}$ 不能自由选，因为已被 mma 定死）。
* **冲突点**：如果 `A_shared` 已经被别的 `T.gemm` 或用户 annotation 定了另一种
  swizzle，就可能冲突；此时走 §3.3 的三条出路。

**常见错误**：把 `B_shared` 漏掉（它同样有约束）；把 `C_local` 说成自由的
（它由 mma 指令决定）。

---

## E2 概念题 ★

**思路**：回到源码定义，逐条核对。

**参考答案**：

1. **错。** `forward_fn` 可以返回多输出；`Layout` 的输出维度由
   `get_output_shape()` 推出（源码事实：`tilelang/layout/layout.py:14-53`、
   `:76-84`）。论文也把 layout 写成 $\mathbb{K}^n \to \mathbb{K}^m$（论文 `:545`）。
2. **错。** Output shape 是**所有线程输出坐标的并集形状**；单线程槽位数由别处
   （如 emitter 的 `local_size_*`）给出。见 §2.2。
3. **错。** 静态形状下 `inverse()` 走 `IterMapLevel::Bijective`，往返（round-trip）
   不一致会抛 `NormalizeIterException`（源码事实：`src/layout/layout.cc:1150-1160`）。
4. **对。** `Fragment.__init__` 有 `replicate` 参数，`>1` 时引入 rep IterVar
   （源码事实：`tilelang/layout/fragment.py:24-70`，`replicate()` 方法 `:148-160`）。

**常见错误**：第 3 问误以为「静态就一定可逆」；非满射/非单射的映射即使静态也
不可逆。

---

## E3 概念题 ★★

**思路**：读 `src/op/parallel.cc:311-329` 的三级定义与 `layout_inference.cc:538-690`
的调用顺序。

**参考答案**：

1. `kStrict` 最保守（只用已定义布局，**不自己发明**）；`kCommon` 居中（允许传播
   常见模式）；`kFree` 最自由（没来源也要造布局）。
2. 用户 annotation 在推导开始前读入 `annotated_layout_map_`
   （`layout_inference.cc:986-1059`），并镜像进 `strict_layout_map`
   （`:597-599`），此后每一步都把它当不可否决的下界——因为**用户显式声明的意图
   必须压过编译器猜测**。
3. 因为需要分级让步：先让「必须满足」的硬约束落地，再传播「常见」模式，最后才
   在剩下的自由空间里优化性能。只跑 `kFree` 会让用户的 annotation 被性能选择
   覆盖掉。

**常见错误**：把 `kStrict` 说成「报错最严格」——它的重点是**不发明布局**。

---

## E4 概念题 ★★

**思路**：读 `InferInFreeMode`（`layout_inference.cc:1396-1550`）。

**参考答案**：

1. 两个 op 只要有**共享的 buffer**（`use_list_`），或共享同一个底层 storage Var
   （alias），就被并进同一分量（`:1414-1445`）。
2. 因为不同起点会导向不同的整解。以某个 op 的布局为根重推，可能得到一份全局代价
   更小的方案；逐个根试等价于在「用哪个 op 定调」的离散空间里做穷举（次数有界）。
3. 先尝试「wide fallback dst 预置」（`:1507-1530`，为 reducer 目的地预留机制
   `ReducerDstSteering` 提供的种子）；若某分量连这个都失败，则
   `ICHECK(has_best) << "no available layout found"`（`:1532`）——这是编译期硬
   失败。此外，非自由模式下的硬冲突会走 `LOG(FATAL) << "Get different layout for"`
   （`:463`）。两者对用户都意味着「改 annotation 或改写法」。

**常见错误**：以为冲突总有自动解——实际上「都不行」是一个明确的编译失败路径。

---

## E5 概念题 ★★

**思路**：读 `layout_cost_model.h:60-70` 与 `.cc:978-1035`、`:1180-1195`。

**参考答案**：

1. `RegisterCountCostModel`：`mem` = 寄存器数组溢出到 local memory 的字节数；
   `regs` = 全部 fragment 的寄存器槽位数（`.cc:978-992`）。`IOAwareCostModel`：
   在同样 `mem`（含溢出）基础上，额外把每条触及 global 的语句按
   `max(带宽字节, 发射等价字节)` 计入 `mem`（`.cc:1001-1035`）。
2. `BetterThan` 先比 `mem`，相等再比 `regs`（字典序，`h:65-70`）。
3. 场景：一份 kernel 有两个候选布局，A 让累加器复制更少（`regs` 小）但让回写
   global 的访存更碎；B 反之。默认模型只看 `regs` 会选 A，切到 `io-aware` 后 A
   的 global 字节被计入 `mem`，B 可能反超。**是否真的会改变结果需要实机验证**
   （本机无 GPU，本章不断言）。

**常见错误**：以为 `io-aware` 会惩罚 swizzle——它是**可选**模型，且默认仍是
`register-count`（`config.h:36-40`）。

---

## E6 代码阅读题 ★

**思路**：对照 §1.1 的三类约束。

**参考答案**：

1. shared 里的数据要喂给 `ldmatrix`，消 bank conflict 靠 swizzle，所以用
   `make_swizzled_layout`；而 fragment 里的数据**已经在线程寄存器里**，必须符合
   mma 指令对「哪个 lane 拿哪几个元素」的要求，所以用
   `make_mma_load_layout`（`mma_macro_generator.py:684-810`）。
2. 描述 A、B 各自落在哪种内存：`s` = shared，`r` = 寄存器（fragment）；所以
   `ss/sr/rs/rr` 是 A/B 的两个 scope 组合（`gemm_base.py:53-72`）。
3. 是**约束（建议布局）**，不是最终结果。证据：它由 `InferLayout` 返回，经过
   `RunInferStep` 的冲突检查与传播（`layout_inference.cc:243-518`），可能被
   `reuse_existing_shared_layout` 让位（`gemm.cc:254-266`），也可能触发合并或
   失败。

**常见错误**：把返回值当成「直接写进 IR」——它只是写进待解的 `layout_map`。

---

## E7 代码阅读题 ★★

**思路**：定位 `layout_inference.cc:437-468`。

**参考答案**：

1. 三条出路：**(a) 包含升级**——用 `ProveFragmentContains` 证明旧布局是新布局的
   子集（`:440-446`）；**(b) swizzle 合并**——两者都是 swizzle 时取粒度更小者
   （`:451-461`，实现在 `src/layout/gemm_layouts.cc:999-1023`）；**(c) 硬失败**。
   （自由模式下的「换根重推」是引擎级处理，发生在 step 3。）
2. 捷径：全复制的 fragment 直接被判为兼容（`src/layout/utils.cc:468-471`）。
   安全，因为每个线程都持有全部元素，任何访问模式都能满足。
3. `LOG(FATAL) << "Get different layout for " << buffer ...`。对用户意味着
   **必须**改代码或 annotation；编译器不会静默降级。

**常见错误**：把「包含」理解成「维度更多」——它是**元素集合的包含**，不是形状的
包含。

---

## E8 代码阅读题 ★★

**思路**：$C_0 = 256 / \text{bits}$（`ascend_layouts.cc:111-115`）。

**参考答案**：

1. fp16/bf16（16 bit）→ `256/16 = 16`；fp32（32 bit）→ `256/32 = 8`；
   int8（8 bit）→ `256/8 = 32`；int4（4 bit）→ `256/4 = 64`。与厂商文档一致
   （corpus `.../zh_developer_techArticles_20250107-1.md:158-160`）。
2. 逻辑 `[64, 48]`，`rows=64`，`cols=48`，C0=16（bf16）。物理形状
   `[cols/C0, rows/16, 16, C0] = [48/16, 64/16, 16, 16] = [3, 4, 16, 16]`。
3. 会拒掉所有 `256 % bits != 0` 的位宽，例如 3-bit 或 64-bit；`float64`
   （64 bit）在 `256 % 64 = 0` 时**不会**被这一条拒掉，但 C0=4，属另一码事。
   更典型被拒的是 24-bit 之类。**注意**：`ICHECK` 失败是硬报错，不是降级。

**常见错误**：把 `cols/C0` 写成 `cols/16`（只在 fp16 时恰好相等）。

---

## E9 代码阅读题 ★★

**思路**：读三个文件头注释与 §5。

**参考答案**：

1. 因为该 pass 只做**形状/别名的规范化**，不写数据（源码注释
   `normalize_fractal_storage.cc:2-6`）。下游因此必须**按有效区域读写**，不能
   假设 padding 是零——否则是正确性 bug。
2. 因为它会新产生 scatter 与 post_copy 两类 op，需要被 `AutoSchedule` 拿去排
   流水与屏障（注释 `insert_nd2nz.cc:1-3`，pipeline `ascend/pipeline.py:88-93`）。
3. 好处是让 padding fill 成为一个**独立可调度的 MTE2 任务**，能被流水/屏障分析
   看见，从而保证它排在被填充缓冲区被读取之前（注释
   `insert_oob_padding.cc:19-30`）。塞进 lowering 就失去这个时序保证。

**常见错误**：以为 `NormalizeAscendFractalStorage` 会清零 padding。

---

## E10 动手题 ★★

**思路**：直接枚举。参考实现风格见 `code/01-layout-algebra/layout_algebra.py`。

**参考答案**（关键片段）：

```python
def transpose(i, j):
    return (j, i)

seen = {}
for i in range(8):
    for j in range(8):
        out = transpose(i, j)
        seen.setdefault(out, []).append((i, j))
assert all(len(v) == 1 for v in seen.values())   # 双射：无碰撞
assert len(seen) == 64                            # 且满射

def padded(i, j):
    return i * 10 + j

used = {padded(i, j) for i in range(8) for j in range(8)}
missing = sorted(set(range(80)) - used)           # 每行后 2 个槽空着
print(len(used), len(missing))                    # 64 16

def broken(i, j):        # 故意丢掉低位：非单射
    return (i // 2) * 4 + j // 2
```

第 3 问：`broken` 把 4 个逻辑点映到同一物理点，往返后坐标改变，因此 v0.1.15 的
`inverse()` 在静态形状下会因 round-trip 检查失败抛 `NormalizeIterException`
（`src/layout/layout.cc:1150-1160`）。

**常见错误**：只检查「没有碰撞」就下结论是双射——还要检查**值域覆盖完整**
（满射）。

---

## E11 动手题 ★★★（需要 GPU）

**思路**：需要真机；本机跑不了运行结果。

**参考答案（思路，不含实测数字）**：

1. 把两个配置各编译一次，抄下四行输出。
2. `Shape` 与 `Thread` 表达式**预期会变**：`threads` 从 128→256 通常改变 warp
   划分，`Thread` 里会出现不同的常系数；`Shape` 反映 block 级 fragment 形状。
3. 关系：`Thread` 表达式是 warp 划分（`block_row_warps` / `block_col_warps`）与
   micro tile 的复合结果，见 `mma_macro_generator.py` 的
   `make_mma_store_layout`（`:814-886`）。

**常见错误**：把论文/示例里记录的 `32/32/32` 输出当成 `64/64/32` 的答案。**本机
没有 GPU，任何实测数字都必须自己跑出来后如实填写。**

---

## E12 动手题 ★★★（需要 GPU）

**思路**：故意触发 §3.3 的第三条出路。

**参考答案（思路）**：

1. **可能**失败（硬冲突 → `LOG(FATAL) << "Get different layout for "`，
   `layout_inference.cc:463`），也可能被合并（若两者都是 swizzle →
   `MergeSwizzleLayouts`，`gemm_layouts.cc:999-1023`）。取决于你给的是不是
   swizzle 布局；一个纯转置布局不是 swizzle，通常走失败路径。
2. 换成长粒度的 swizzle 时行为可能不同：合并取**更小粒度**，所以最终多半落到
   你给的更小档位而编译通过。
3. 报错文本必须来自实机；本章不提供杜撰的报错。

**常见错误**：把「推测可能合并」写成「一定合并」。

---

## E13 动手题 ★★★（需要 NPU）

**思路**：对照 `GemmMAD.infer_layout`。

**参考答案（思路）**：

1. L0A / L0B 应是 MajorK 或 MajorMN 的分形布局（物理形状形如
   `[cols/C0, rows/16, 16, C0]`）；L0C 应是固定 16×16 分形
   （`[N/16, M/16, M%16, N%16]`）。
2. `trans_B=True → MajorK`，`trans_B=False → MajorMN`
   （`gemm_mad.py:119-140`）。
3. 因为把 reduce-K 轴放在 C0 轴上，使 K 方向的连续数据满足 Cube 的
   「K 按 C0 对齐」约束。

**常见错误**：把 L0C 也当成可选的 major 布局——它是**固定**的 16×16 累加器。
