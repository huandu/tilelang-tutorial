# 第 25 章 习题答案

每题给出**解题思路 → 参考答案 → 常见错误**。所有源码行号可在
`_vendor/tilelang-v0.1.15/`、`_vendor/tilelang-ascend-v0.1.2.000/` 与 corpus 里现场核对。

---

## E1

**思路**：回看正文 §1.1 的表格，但不要背表；每条都说出「源码里哪一行决定了这一点」。

**参考答案**：

| 路线 | 目标 | 生成物 | 编译器 |
| --- | --- | --- | --- |
| M1 | `target="ascend"`（别名 `"asc"`） | CCE/Ascend C，`asc_*` 内置函数风格 | `bisheng --npu-arch=dav-3510` + `ld.lld` → 可执行 CCE ELF |
| A1 | `target="ascendc"` | AscendC 类库 C++（`AscendC::TPipe` / `TBuf`） | `bisheng --npu-arch=dav-2201 -xasc ... --shared` → `.so` |
| A2 | `target="pto"` | PTO-DSL C++（`pto::Tile` 模板） | `bisheng --cce-aicore-arch=dav-c220\|dav-c310 -xcce` → `.so` |
| L1 | TVM target kind `npuir` | AscendNPU IR（MLIR 文本） | `bishengir-compile --enable-hivm-compile=true ...` → `.o` |

依据：`tilelang/ascend/target.py:40-58`、`tilelang/ascend/backend.py:52-62`、
`tilelang/contrib/bisheng.py:88-228`、`_vendor/tilelang-ascend-v0.1.2.000/tilelang/engine/lower.py:156-170`、
`tilelang/jit/adapter/libgen.py:150-230`、corpus `tilelang-mlir-ascend/tilelang/jit/jit_npu.py:1862-1916`。

**常见错误**：把 M1 写成 `target="ascendc"`（那是 A 树）；或者把 L1 说成「也是
Ascend C」——它的中间表示是 MLIR。

---

## E2

**思路**：`PrintFuncPrefix` 的三个分支一一对应 `kMix` / `kVector` / `kCube`，
再看函数体预扫描怎么得到它。

**参考答案**：

1. `kMix` → `extern "C" __global__ __mix__(1, 2)`（同一 kernel 里既有 Cube 又有
   Vector 工作）；`kVector` → `__vector__`（只有 Vector）；`kCube` → `__cube__`
   （只有 Cube）。
2. 因为 950 的物理比例是 1 Cube : 2 Vector。源码依据：`codegen_ascend.cc:339-342`
   的注释「dav-3510's direct AIC-to-AIV data paths require the physical 1:2 group」。
   corpus 依据：硬件页写 Ascend 950PR 是 Cube Core 32/28、Vector Core 64/56
   （`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md:57-58`）。
3. `AddFunction` 预扫描函数体，看有没有 Cube 工作、有没有 Vector 工作
   （`codegen_ascend.cc:538-560` 一带的 `has_cube` / `has_vector`），据此置
   `kernel_mode_`。

**常见错误**：以为 `(1, 2)` 是「1 个 AIC + 2 个 AIV 的逻辑编号」而不是物理组；
或以为用户要手动指定——用户只写 `T.Kernel(N)` / `T.MixedKernel(N, sids=…)`。

---

## E3

**思路**：先对着 `allocate.py` 抄默认 scope，再想 `alloc_l0c` 为什么不靠这段函数
定布局。

**参考答案**：

1. `shared` / `shared.dyn` → UB（`__ubuf__`）；`shared.l1` → L1（`__cbuf__`）；
   `shared.l0a` → L0A（`__ca__`）；`shared.l0b` → L0B（`__cb__`）；
   `shared.l0c` → L0C（`__cc__`）。
2. `alloc_shared` → `shared.dyn`；`alloc_l1` → `shared.l1`；`alloc_l0a` →
   `shared.l0a`；`alloc_l0b` → `shared.l0b`；`alloc_l0c` → `shared.l0c`。
   五个都被覆盖。
3. 因为「在哪种存储」和「这块存储的元素怎么排」是两件事。`alloc_l0c` 在分配时
   通过 `make_ascend_l0c_layout(buf)` 把 L0C 累加器布局写进 `layout_map`
   （`allocate.py:64-70`）；而 `alloc_l0a`/`alloc_l0b` 故意把布局留给消费它的
   `T.gemm` 按 transpose 推断（`allocate.py:37-52` 的 docstring）。限定符只回答
   第一个问题，所以不需要处理 `local` 系列。

**常见错误**：把 `T.alloc_shared` 当成 CUDA 的 shared memory 并去查 swizzle——
在昇腾方言里它是 UB，布局由 fractal/OOB 那套 pass 处理。

---

## E4

**思路**：先看 scope 组合，再查 `GetDMAPath` 的分支，最后查 codegen 里对应的
`Emit*_` 函数。

**参考答案**：

| | src→dst | DMAPath | pipe | intrinsic |
| --- | --- | --- | --- | --- |
| (a) | GM → L1 | `kGMToL1` | MTE2 | `asc_copy_gm2l1_nd2nz`（`transpose=True` 时 `dn2nz`） |
| (b) | L1 → L0A | `kL1ToL0A` | MTE1 | `asc_copy_l12l0a`（transpose 有 `_transpose` 变体） |
| (c) | L0C → UB | `kL0CToUB` | FIX | `asc_copy_l0c2ub`，前置 `asc_set_l0c_copy_nz_para(1,0,0)` |
| (d) | UB → GM | `kUBToGM` | MTE3 | `asc_copy_ub2gm_align` |

依据：`src/ascend/op/oob_padding.cc:46-68`、`src/ascend/transform/ascend_pipe.h:52-63`、
`src/ascend/codegen/codegen_ascend.cc:1383-1609`。

**常见错误**：以为 (c) 走 MTE3——L0C 的搬出由 Fixpipe 负责，这是昇腾特有的
收尾单元；把它当普通 GM→UB 会得到错误的同步推理。

---

## E5

**思路**：这三个函数是昇腾后端的「去选择化」证据：没有可选项，就没有选择逻辑。

**参考答案**：

1. `SelectInst` 永远返回常量 `kAscendMAD`，字面值是 `"ascend.mad"`。
2. `m_warp = 1`、`n_warp = 1`。昇腾没有 SIMT warp：一个核内的计算不需要按
   warp 划分归属，划分由 `AssignCore` 在 Cube/Vector 之间做。
3. 体现在参数上：
   * 累加/清零：`clear_accum`（映射到 `unit_flag_ctrl`）；
   * 精度模式：`T.set_hf32_mode("nearest_zero" | "nearest_even" | None)`；
   * 量化：`T.gemm_blockscaled` + `T.alloc_l0a_sf` / `alloc_l0b_sf` 的 MX 尺度槽位，
     最终走 `asc_mmad_mx`。
   另外 M/N/K 必须是编译期静态值，否则 `T.gemm` 直接抛 `ValueError`
   （`gemm_op.py:55-60`）。

**常见错误**：答「因为昇腾只有一个矩阵指令」但不指出还有 `asc_mmad_mx` 这条
块缩放变体。

---

## E6

**思路**：`dual_copy` 的 docstring 把契约写全了，逐条抄比猜更快。

**参考答案**：

1. rank ≥ 2 时，恰好一个尾部维度在源/目的之间是 2:1。
   切 M 维（`[M,N] → [M/2,N]`）对应 `dual_dst_ctl=0b01`；
   切 N 维（`[M,N] → [M,N/2]`）对应 `dual_dst_ctl=0b10`。
   静态已知的整体 extent 必须是偶数，奇数直接报错。
2. `L0C→UB` 走硬件双目的（`asc_copy_l0c2ub` 的 `asc_dual_dst_mode`）；
   `GM→UB`、`UB→GM`、`UB→L1` 走按 `cthread`（子核 ID）切片的逐 AIV 软件拷贝
   （`RewriteDualCopy` 在布局/OOB/调度之前完成改写）。
3. 因为 `dav-950` **不支持**在 `cc→ub` 双目的拷贝里做 `quant_pre` 类型转换；
   源码注释写明「If src and dst dtypes differ, the hardware may silently produce
   incorrect results」。所以 Python 层主动 raise，宁可编译失败也不静默出错。

**常见错误**：以为 `dual_copy` 是「自动把 Vector kernel 变成双 AIV 启动」——
它必须在混合 kernel 里、且 Cube 侧工作要用户自己写。

---

## E7

**思路**：一个看线程域，一个看寄存器宽度约束。

**参考答案**：

1. `T.SimtVF` 有真线程域：块体第一个 `AttrStmt(thread_extent)` 绑定
   `threadIdx.x`（`vf_regions.h:38-60`）。`T.SimdVF` 没有线程，是寄存器级。
2. `SimdVF` 里的向量类型必须恰好占一个 **2048-bit** 寄存器
   （`element_bits × lanes == 2048`），predicate 必须是 `boolx256`。
   不满足时 `ValidateSimdVectorType` 直接 `LOG(FATAL)`（编译期报错），
   不做任何自动降级。
3. float32 时 2048/32 = 64，所以一次处理 64 个元素，循环步长是 `tile // 64`。

**常见错误**：把 `latency=` 当成实测周期；它只是调度模型的输入。

---

## E8

**思路**：脚本已经抽好三张表，你只需要加一条断言。

**参考答案**：

1. 运行 `python3 .../02-scope-dma-pipes/scope_dma_pipes.py`，会看到
   `kGMToUB/kGMToL1 → kMTE2`、`kUBToGM/kUBToL1 → kMTE3`、
   `kL1ToL0A/kL1ToL0B/kL1ToL0ASF/kL1ToL0BSF → kMTE1`、
   `kL0CToUB/kL0CToGM → kFixpipe`。
2. 在 `main()` 的检查区加：

   ```python
   for path in ("kL0CToUB", "kL0CToGM"):
       if EXPECTED_PIPE.get(path) != "kFixpipe":
           failures.append(f"{path} must be owned by FIX")
   ```

3. 因为 `kGMToUB` 与 `kGMToL1` 都是「从 GM 读进片上」，同属入向搬运，
   共用 MTE2；而 `kL0CToUB` 的源是 Cube 的累加器，它的搬出附带量化/ReLU 等
   后处理，由 Fixpipe（`PIPE_FIX`）承担，不能交给 MTE3（那是 UB 的出向搬运）。

**常见错误**：把 `kFixpipe` 当成「另一个 MTE」——它是矩阵收尾单元，
`ResourcePipe` 里是独立的第 6 位。

---

## E9

**思路**：矩阵脚本直接告诉你答案；第三问考的是「两棵树」这条主线。

**参考答案**：

1. `alloc_ub` 只出现在 **A 树**列（`tilelang/language/allocate.py:156`）；
   M 树列是 `MISSING`。M 树用 `T.alloc_shared`（scope `shared.dyn`）。
2. `T.tile` 命名空间出现在 **A 树**列，注册在
   `tilelang/language/__init__.py:86`（`from . import ascend_tile as tile`）。
   M 树的 `tilelang/ascend/language/__init__.py` 里没有这个别名。
3. **能**。`T.alloc_l1`（小写）只存在于 M 树，`T.alloc_L1`（大写）只存在于 A 树；
   同一段代码同时出现两者，说明作者把两棵树的 API 混在了一起，必然有一侧是
   `AttributeError`。（除非他手动给自己起了一个同名变量，那是另一回事。）

**常见错误**：以为大小写只是风格差异。`alloc_l1` 与 `alloc_L1` 是两个不同文件里
的两个不同函数。

---

## E10

**思路**：把「谁做这件事」按两条路线各列一次；关键是**责任边界在 TIR 内还是在
MLIR 内**。

**参考答案**：

| 职责 | M 树（TIR） | L1（AscendNPU IR） |
| --- | --- | --- |
| CV 融合 | `AssignCore` 选 `kCoreCube`/`kCoreVector`，再生成 `__mix__(1,2)` 函数前缀 | HIVM 的 CV 核映射（Mix Kernel 融合、CVPipeline、AutoSubTiling） |
| 多缓冲 | `AutoSchedule`（Z3）给出版本数 → `PrepareMultiBuffer` → `MaterializeMultiBuffer` | `bishengir-compile --enable-auto-multi-buffer=true` |
| 同步 | `InsertSync` 生成 `hard_event`，`RewriteFlagToBuf` 压进 8 槽 flag | HIVM 的 AutoSync / CVPipeline（`--disable-hivm-auto-inject-sync` 可关） |
| 内存规划 | `MergeUBAllocations(align_bytes=32)` | PlanMemory |

依据：`tilelang/ascend/pipeline.py:83-182`、`src/ascend/transform/assign_core.cc:64-75`；
corpus `tilelang-mlir-ascend/tilelang/jit/jit_npu.py:1862-1916` 与
`huawei-techarticle-tilelang-npuir.md:136-146`。

**常见错误**：把两边的 pass 名混着写（例如说 M 树有 PlanMemory）。
判断标准很简单：计划/求解/物化这套词汇在 `src/ascend/transform/` 里，
PlanMemory/HIVM 这套在 MLIR 侧。

---

## E11

**思路**：看编译器参数就能反推路线。

**参考答案**：

* 第一行是 **A1**（`target="ascendc"`）：`--npu-arch=dav-2201` 是 A2 的 arch，
  `-xasc` 告诉 bisheng 这是 Ascend C 源；`--shared` 说明产物是 `.so`。
* 第二行是 **A2**（`target="pto"`）：`--cce-aicore-arch` 选的 `dav-c220` 是
  A2/A3 的 CCE arch（A5 会是 `dav-c310`），`-D MEMORY_BASE` 与 A5 的
  `REGISTER_BASE` 对应，`-xcce` 指定 CCE 源，`-I.../3rdparty/pto-isa/include`
  拉进 PTO-ISA 头。
* 第三行是 **L1**（`npuir`）：`bishengir-compile` 吃的是 MLIR 文本，
  `--enable-hivm-compile=true` 走 HIVM 流程，
  `--enable-triton-kernel-compile=true` 走 Triton 风格内核编译。

为什么参数不同：**输入语言不同**。`-xasc` 告诉编译器源文件是 Ascend C；
`-xcce` 告诉它是 CCE 源；`bishengir-compile` 的输入是 MLIR，所以不需要 `-x*`，
目标 arch 用它的 `--target=` 传递（A5 分支里就是
`f"--target={target_arch}"`，`jit_npu.py:1878`）。

**常见错误**：把 `--npu-arch` 与 `--cce-aicore-arch` 当成同一个开关的两种写法
——它们是两个编译器前端（`-xasc` vs `-xcce`）的参数。

---

## E12

**思路**：把「硬件是 A2」「现有库是 Ascend C」「要求生成物可读」「想在无 NPU
机器上先验证」四条约束分别对到四条路线上。

**参考答案**：

推荐 **A 树的 `target="ascendc"`（A1 路线）**，理由：

1. **设备匹配**：PR3308 明确说主干 M1 的目标硬件是 Ascend 950（`dav-3510`），
   A2/A3 请用社区维护的 TileLang-Ascend（`github-pr-3308.md`）。所以 M1 不可选。
2. **生态匹配**：A1 生成的是 AscendC 类库 C++（`AscendC::TPipe` / `TBuf`），
   与既有手写 Ascend C 算子的阅读习惯一致，也便于和现有库对照。
3. **生成物可读**：用 `func.get_kernel_source()` 直接打印
   （编程指南 §5.2），并且本目录的 `code/05-emit-ascend-c/print_ascend_c.py`
   给了核对清单。

不推荐的地方与边界：

* **A2（PTO）**虽然也能在 A2/A3 上跑，但它的生成物是 PTO-DSL 模板，团队要额外
  熟悉一套指令集；如果只是「先把算子写出来」，A1 的学习成本更低。
* **L1（AscendNPU IR）**的中间表示是 MLIR，不是团队现有技能栈；而且它是另一条
  分支，与「现有 Ascend C 库零拷贝集成」的目标距离更远。
* **无 NPU 开发机上的功能验证**：这是最容易被高估的一条。camodel 仿真在锁定
  版本里**只支持 PTO 后端的 A5 平台**（第 19 章 §3.6），而团队目标是 A2——所以在
  无卡机器上能做的只是「生成代码 + 人工核对」，**不能替他们把功能跑通**。

建议的验证顺序（依次执行下面四条命令）：

```bash
python3 code/01-route-map/route_map.py
python3 code/03-api-presence-matrix/api_matrix.py
python3 code/05-emit-ascend-c/print_ascend_c.py --audit
python3 code/05-emit-ascend-c/print_ascend_c.py --emit --target ascendc
```

最后一条只在装了 CANN 工具链的机器上有意义；有卡机器上再加一次数值比对，
才算真正验证。

**必须上板才能确认**的部分：数值正确性、并发/流水下的同步正确性、
`num_stages`/buffer 版本取值是否合适、以及任何性能结论。本章（以及任何
「仅静态核对」的材料）都不能替代这些。

**常见错误**：回答「用 M 树 + camodel」——M 树的路线里没有 camodel 入口，
camodel 在锁定版本里挂在 A 树的 PTO 后端、且只对 A5。
