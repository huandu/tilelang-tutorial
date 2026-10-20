# 第 12 章 调试与性能分析

## 学习目标

读完本章，你应当能够：

1. 拿到一个“跑不起来 / 结果不对 / 太慢 / 偶尔出错”的 kernel，**先判断它属于哪一类**，
   再按决策图选工具，而不是凭直觉改代码；
2. 打开并读懂 TileLang 的编译期日志（`TVM_LOG_LEVEL` / `TVM_LOG_DEBUG`），
   知道一条 `Check failed:` 报错背后是哪个环节在断言；
3. 在**没有 GPU/NPU 的机器上**，用 compile-only、IR Lower Trace、Pass Visualizer、
   布局可视化把“编译器产出了什么”看清楚；
4. 用 `T.print` / `T.printf` / `T.dump_tensor` 与 AICore 异常 dump 在**运行期**取到中间值，
   并且清楚它们的代价与上限（为什么打印完必须删掉）；
5. 用 PyTorch / fp64 参考实现对拍，**说得出容差是从哪来的**，并能定位“偶发结果错误”
   （大多数是同步问题）；
6. 在测性能之前先算出 roofline 预期，再用 Nsight Compute（NVIDIA）或 msProf（昇腾）
   回答“差在哪一项”，而不是先测一堆数字再说。

## 前置知识

* 第 04 章（一次编译的全流程）：知道 lowering、pass、codegen 的顺序，本章才有落点。
* 第 07–11 章：你已经写过至少一个 kernel、用过 `T.copy` / `T.gemm` / `T.Pipelined`，
  并且知道 `alloc_shared` 与 `alloc_fragment` 的区别。
* 第 10 章：知道 layout 与 fragment 的基本概念，本章“把布局画出来”那一节才不会陌生。

本章是第 20–29 章的**工具体检章**：后面讲 IR 管线、布局推导、同步插入、codegen 时，
都要用到这里介绍的工具去观察编译器。

---

## 1. 问题的由来

### 1.1 写 kernel 只有四件事会出错

写 TileLang kernel 时会遇到的麻烦，按**症状**可以干净地分成四类：

| 类别 | 症状 | 你手上有的线索 |
| --- | --- | --- |
| 生成问题（generation） | 跑不起来：lowering 报错、codegen 断言、编译失败 | 一段报错文本 |
| 正确性问题（correctness） | 能跑，结果和参考对不上 | 一个（或一批）对不上的数值 |
| 性能问题（performance） | 结果对，但比理论预期慢很多 | 一个耗时数字 |
| 偶发 / 非确定性 | 同一份输入，有时对有时错，或每次错得不一样 | 复现率 |

前两类是官方调试文档明确划分的（见
`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md`
的 Overview 与 Conclusion），第三类由该文档指向 profiler，第四类本章单独拎出来讲，
它是“正确性”的一个特例，但排查方法完全不同。

分类的重要性在于：**不同的类别对应不同的工具**。拿 profiler 去查一个编译错误是浪费，
拿 `print` 去查一个“差 0.1% 的数值误差”也是浪费。那是容差口径问题，不是数值错误。

### 1.2 为什么要在“实用部分”就讲工具

第 20–29 章会讲 TileLang 怎么推导布局、怎么插入同步、怎么把 `mma` 降级成 `wgmma`。
如果没有本章的工具，那些章节只能靠读源码理解；有了工具，你可以**亲手让一个 pass
跑一遍，把它前后的 IR 摆在一起看**。

所以本章的结构不是“按工具罗列”，而是分三层可见性：

* **编译期（编译器里）**：日志、compile-only、IR Lower Trace / Pass Diff / Pass Visualizer、
  布局可视化。回答“编译器做了什么”。
* **运行期（kernel 里）**：`T.print` / `T.printf` / `T.dump_tensor` / 异常 dump。
  回答“运行时数值是什么”。
* **硬件层（芯片里）**：Nsight Compute / msProf。回答“时间花在哪个单元上”。

越往下越贵、越需要真实的卡；越往上越便宜、越能在本机完成。

### 1.3 本机能做什么，必须在有卡机器上做什么

本书的写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [../../VERSIONS.md](../../VERSIONS.md) 的“环境记录”）。这条限制恰好帮你划出一条线：

* **本机就能做**：读编译期日志、compile-only 拿源码、IR Lower Trace 看 pass 前后、
  Pass Visualizer 看结构树、布局可视化、写对拍脚本、算 roofline 预期、读别人的 profiler 报告。
  这些都不需要加速器，其中大部分连 driver 都不需要。
* **必须在有卡机器上做**：真正的 kernel 执行、`T.print` 的实际输出、对拍的实测误差、
  Nsight Compute / msProf 的采样数据。

本章所有“需要卡”的结论都会标明验证级别；本机**没有**运行过任何 kernel，
因此正文不出现任何自测性能数字。

---

## 2. 概念与模型

### 2.1 四类问题的可观测信号

把 1.1 的表再压一层，其实只有四个可观测信号：

1. **有没有报错**（有文本）；
2. **结果对不对**（有参考）；
3. **快不快**（有预期）；
4. **稳不稳**（有复现率）。

排查的第一步永远是把信号拿到手：出错就把完整报错抄下来（不要只抄最后一行），
结果不对就构造一个最小可复现输入，太慢就先算预期，偶发就先统计复现率。
没有信号就开始改代码，是最常见的浪费。

### 2.2 三层可见性与“先观察、后假设”

一个有用的纪律：**先用工具把事实摆出来，再提假设**。

* 怀疑 layout 错，不要先去读 `layout_inference.cc`，先 `plot_layout` 或打开布局可视化，
  看编译器实际推出来的映射长什么样；
* 怀疑某个 pass 引入了 bug，不要先读那个 pass 的 C++，先 `TL_LOWER_TRACE=1` 跑一遍，
  看它前后的 IR 到底哪一行变了；
* 怀疑 tensor core 没吃满，不要先改 tile 大小，先 `Analyzer` 算一下这个形状是
  memory-bound 还是 compute-bound，再看 profiler。

本章的工具都是为“先观察”服务的。

### 2.3 决策图：问题分类 → 工具 → 定位路径

**图 1**（`figures/01-decision-map.md`）是本章的总入口。它的主干是：

```text
生成问题 → 日志 → compile_only → IR Lower Trace → Pass Visualizer → 布局可视化
正确性问题 → 对拍 + 容差 → T.print / dump_tensor → device_assert → AutoDD
偶发问题 → 先下并发（单 block / num_stages=1）→ 二分加回 → 回到对拍
性能问题 → 先算 roofline 预期 → Nsight Compute / msProf → 对照预期 → 改一个变量重测
```

后面的 3.1–3.10 节就是这张图的展开。如果你现在就有一个问题，可以直接跳到对应分支。

---

## 3. 用法

### 3.1 编译期（一）：打开日志

TileLang 复用 TVM 的日志系统。它的实现就在锁定版本里：
`_vendor/tilelang-v0.1.15/src/runtime/logging.cc`。三个宏的关系是（见该文件
`:17-23`、`:51-70`）：

| 宏 | 用途 | 什么时候能看到 |
| --- | --- | --- |
| `LOG(INFO/WARNING/ERROR/FATAL)` | 常驻日志；**TileLang 大量的 C++ 报错就是 `LOG(FATAL)`** | 任何时候 |
| `DLOG(...)` | 开发者调试输出 | 需要 Debug 构建 + 运行时开关 |
| `VLOG(n)` | 带详细级别的 verbose 日志（n = 1…6） | 同 `DLOG`（TVM 的 VLOG 建立在 DLOG 之上） |

还有一组断言：`CHECK` / `DCHECK` / `ICHECK`，其中 `ICHECK` 在 Release 构建里也保留，
你在报错里看到的 `Check failed: ...` 就是这一类。

**打开运行期日志**：设置 `TVM_LOG_LEVEL`。**打开 Debug 日志**：需要在构建时定义
`TVM_LOG_DEBUG`（CMake 的 Debug 模式会自动加，见
`_vendor/tilelang-v0.1.15/CMakeLists.txt:487-490`），运行时再用同名环境变量指定每个文件
的级别。

这里有一个**官方文档自己都没绕清楚的坑**：同一个名字 `TVM_LOG_DEBUG` 被用在两个地方
（编译期宏、运行时环境变量），它们语义不同却同名（`logging.cc:73-135` 是运行时说明符
的解析逻辑，格式是逗号分隔的 `<文件名>=<级别>`，`DEFAULT` 表示全部文件，级别 `-1`
表示关闭）。官方文档（`.../tutorials/logging.md`）在“Enabling Log Output in Tilelang”
一节里说要设 `TVM_LOG_LEVEL`，示例却写成 `TVM_LOG_DEBUG=1 python3 code.py`，
这不是笔误，而是那两个同名机制的历史遗留。实践建议：

* 只想看 INFO 以上：设 `TVM_LOG_LEVEL`；
* 想看 `DLOG`/`VLOG`：确认你的构建是 Debug（或 wheel 里编进了 `TVM_LOG_DEBUG`），
  再用 `TVM_LOG_DEBUG=<说明符>` 打开具体文件。Release wheel 里 `DLOG` 是被死代码
  消除掉的，设了也不会出现。

**什么时候用**：一条 `Check failed:` 报错里通常带着 `文件:行号`。先读它，再决定是去读
那个 C++ 文件，还是在 Python 层把自己的 kernel 改对。绝大多数“TileLang 的报错”
其实是你的形状/类型不满足某个约束，`LOG(FATAL)` 的消息已经把约束写清楚了。

### 3.2 编译期（二）：不跑加速器，先看产物

编译期调试最便宜的一步是：**跳过 device 编译，只把 lowering 的结果打出来**。
这就是 Compile-Only：

```bash
python -I -m tilelang.tools.compile_only --output_file out.c example.py
python -I -m tilelang.tools.compile_only --target cuda --output_file out.cu example.py
```

它停在 `tilelang.lower(..., enable_device_compile=False)`，不调用 `nvcc`，也不碰设备。
官方文档 `.../tools/compile_only.md` 明确写了三点：默认的 `c` 目标在任何机器上都能跑；
`--target cuda` 只需要 wheel 里编进了 CUDA codegen FFI（Linux wheel 有，macOS Metal
wheel 没有，会报 `CUDA codegen FFI missing`）；`--target auto` 会被拒绝，因为 `auto`
需要探测设备。

程序里等价的做法是 `compile_kernel_source(func, target)`；已经拿到 JITKernel 时，
`get_kernel_source()` 也能直接取源码（`_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532`）。

如果编译产物已经生成，但你只想在最后一步做点手脚（比如插一行 `printf`），
可以注册 codegen 后处理回调：CUDA 用
`_vendor/tilelang-v0.1.15/tilelang/engine/callback.py:55` 的
`register_cuda_postproc_callback`，昇腾用
`_vendor/tilelang-v0.1.15/tilelang/ascend/callback.py:32` 的
`register_ascend_postproc_callback`。这是一个全局回调，实验要限制在独立进程里做。

复现步骤见 `code/03-compile-only/`。

### 3.3 编译期（三）：看一个 pass 前后的变化

这是本章最重要的工具。TileLang 的 lowering 是**渐进式**的：一个 `T.copy` 会先展开成
`T.Parallel`，再展开成更低层的语句，最后才变成 CUDA C。中间任何一步出问题，IR 都会
“看起来对，但下一步就错”。

**IR Lower Trace（推荐）**会捕获**每一个** pass 的前后 IR，再加上最终的 codegen 步骤：

```bash
TL_LOWER_TRACE=1 python3 my_script.py        # 写 HTML 报告（默认）
TL_LOWER_TRACE=terminal python3 my_script.py # 只在终端打彩色 diff
TL_LOWER_TRACE=both python3 my_script.py     # 两者都要
```

输出目录默认在 `./tmp/lower_trace_dir/<脚本名>/`，里面既有稳定的 `report.html`
符号链接，也有按 phase 分目录的 `.tir` 快照（例如 `pipeline_c/00_BindTarget_before.tir`
与 `_after.tir`），还有 codegen 的 `.cpp`。目录结构、三个 codegen 文件的
edit-and-recompile 工作流、以及 HTML 报告的键盘操作，都在
`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/lower_trace.md`
里。

只想固定比较几个 pass 时，用程序 API：

```python
from tilelang.tools import lower_trace as lt
results = lt.lower_trace(func, tilelang.transform.ThreadSync("shared"), mode="terminal")
```

返回值是一串字典，每个含 `name` / `before_script` / `after_script` / `diff_lines` /
`insertions` / `deletions` / `changed`。要追踪整个 pipeline 就用 `lt.enable(mode="both")`，
不再需要时 `lt.disable()`
（`_vendor/tilelang-v0.1.15/tilelang/tools/lower_trace/core.py:663-705`）。

**Pass Diff 是它的前身**，保留只是为了向后兼容（环境变量 `TILELANG_PASS_DIFF`，
程序入口 `_vendor/tilelang-v0.1.15/tilelang/utils/pass_diff.py:1030`）。官方文档已经把
IR Lower Trace 标为它的严格超集，新代码直接用后者。

**Pass Visualizer** 换了一个视角：Pass Diff 比的是 TVMScript 的**文本行**，
Pass Visualizer 渲染的是 `SBlock` 的**结构树**（`reads` / `writes` / `alloc_buffers` /
`annotations` 字段），并且把每个 tile op 按**字段名**展开（`M=64`、`K=32`、`policy=0`
这种），tile op / 同步原语 / 已降级的硬件 intrinsic 用不同颜色区分。调试
**结构性 pass**（布局推导、warp specialization、流水线）时它比文本 diff 好用：

```bash
python -m tilelang.tools.pass_visualizer.viewer \
    tilelang/tools/pass_visualizer/examples/gemm_relu.py \
    --set M=1024 --set N=1024 --set K=1024 \
    --set block_M=128 --set block_N=128 --set block_K=32 \
    --out gemm_relu_passes.html
```

它的 CLI 目前只支持 CUDA 目标（见
`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md`
的 Pass Visualizer 一节）。

最后一个是**编译期耗时**：`tilelang.tools.pass_timing`
（`_vendor/tilelang-v0.1.15/tilelang/tools/pass_timing.py:148`）记录每个 pass 花了多少时间。
它回答的是“编译为什么这么慢”，**不是**“kernel 为什么这么慢”，两个“性能”不要混用。

复现步骤见 `code/04-lower-trace/`。

### 3.4 编译期（四）：把布局画出来

布局（layout）是 TileLang 把“逻辑 tile”映射到“物理线程 / 寄存器”的方式。当结果
整块错位、或者只有某些位置对时，先看布局往往比读代码快。

两条路径：

* `tilelang.tools.plot_layout` ，给你手上的 `T.Layout` / `T.Fragment` 画图；
* **布局推导可视化** ，打开 `TL_LAYOUT_VISUALIZATION_ENABLE` 之后，编译器会把每个
  推导出来的 fragment 的 thread / local-index 映射打出来，并可选写出 png/pdf/svg。

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
    }
)
```

文本输出形如：

```text
C_local inferred layout:
  Shape: [32, 32] -> [8]
  Thread: <thread-index expression>
  Index:  [<local-index expression>]
```

要注意的是：出图需要 Matplotlib（`pip install "tilelang[vis]"`），而且绘图是逐元素
枚举的，大 shape 会产生很大的图并消耗大量内存。完整参数表、支持的格式与限制见
`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/layout_visualization.md`。

### 3.5 运行期（一）：CUDA 侧的 `T.print` 与 `T.device_assert`

`T.print` 是 TileLang 带的门控打印原语，实现在
`_vendor/tilelang-v0.1.15/tilelang/cuda/language/print.py:128-207`。它的行为**取决于
参数类型和 buffer 的 storage scope**：

| 参数 | 行为 | 门控 |
| --- | --- | --- |
| `T.print(msg="...")` | 只打印一条消息 | 无（每个线程都会执行！） |
| `T.print(expr)` | 打印一个 `PrimExpr` 的值 | 无（要自己加条件） |
| `T.print(buffer)` | 打印整个 buffer 的元素 | 见下表 |

| buffer scope | 门控条件 |
| --- | --- |
| `local` | 恒真（所有线程都打印） |
| `local.fragment` | `tx == main_lane and ty == 0 and tz == 0`（先 `copy` 到 shared 再打印） |
| `shared` / `shared.dyn` | 同上 |
| `global` | 恒真（所有线程都打印，输出会被重复 `threads` 倍） |

`main_lane = warp_group_id * 128 + warp_id * 32`（`print.py:149-151`）。也就是说，
**打印标量表达式时 TileLang 不帮你门控**，必须自己写
`if tx == 0 and bx == 0: T.print(...)`。输出格式由 C++ 模板给出，例如
`msg='...' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): dtype=half value=...`（模板见
`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/debug.h`）。

`T.device_assert(condition, msg)` 会在条件不成立时让 kernel 报错，消息里默认带调用栈
（`_vendor/tilelang-v0.1.15/tilelang/cuda/debug.py:23`）。

**代价与限制**：device 侧 `printf` 走的是受限的片上缓冲，输出会串行化并且**显著改变
耗时**；官方调试文档也提醒要注意 GPU 代码里的并发与线程同步。`T.print(global_buf)`
这类无门控用法很容易把缓冲打满。**调完必须删掉**。

复现步骤见 `code/05-print-and-assert/`。

### 3.6 运行期（二）：昇腾侧的 `T.printf` / `T.dump_tensor` 与异常 dump

昇腾侧没有 CUDA 那套 device `printf` 语义可直接用。适配器提供了两个设备端调试接口
（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534` 与 `:582`）：

| 接口 | 作用 |
| --- | --- |
| `T.printf(format_str, *args)` | C 风格格式化打印；支持 `%d` / `%i` / `%f` / `%x` / `%s` / `%p`（`%p` 会被改写成 `0x%x`，官方建议直接用 `%x` 打地址） |
| `T.dump_tensor(tensor, desc, dump_size, shape_info=())` | 把指定张量转储到 host；支持 UB / L1 / L0C / global；`desc` 是自定义标记（只支持 `uint32_t`），`dump_size` 是元素数，`shape_info` 用来格式化输出 |

打开方式有讲究。旧式的 `TL_PTO_DEBUG=1` **只对 `target="pto"` 起作用**，
不会打开 AscendC 的调试选项；AscendC 请在 kernel 作用域用编译选项：

```python
@tilelang.jit(target="ascendc", compile_flags=["-D_DEBUG", "--cce-enable-print"])
```

**代价与限制**：设备端打印/转储会明显影响算子实际性能；**每个核上所有 dump 接口使用的
总空间上限是 1 MB**，超出会被丢弃（见
`../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/tutorials/print.md`）。

第二条路径是**事后取现场**：AI Core 硬件异常（非法访问、MTE 错误、Cube/Vector 溢出等）
发生时，CANN 会异步回调一个 TileLang 注册的异常处理器，把 kernel 的**输入张量**
落盘。开关是 `TL_ASCEND_EXCEPTION_DUMP`（在
`_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:62`，默认关闭），
需要 CANN >= 9.3.0。两个环境变量必须在 `import torch` / ACL 初始化**之前**设置：

```python
import os
os.environ["ASCEND_DUMP_PATH"] = "/tmp/exc_dump"
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"
```

然后在 `except` 块里用 `parse_exception_dump(dump_path, kernel_name)` 一站式取回
`np.ndarray`（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/tools/ascend_exception_dump_bin.py:67`），
或者分步用 CANN 的 `msaicerr.py` 解析再用 `read_msaicerr_bin()` 读。
落盘路径规则、`ParamSizeInfo` 与 MAGIC 搜索的机制见正文 4.5 与
`.../docs/tutorials/ascend_aicore_exception_dump_zh.md`。

复现步骤见 `code/06-ascend-dump/`。

### 3.7 正确性：与 PyTorch 参考实现对拍

#### 3.7.1 先定容差，再谈对错

对拍的第一件事不是写代码，是**说清楚容差从哪来**。低精度算子的误差主要来自“把输入
舍入到低精度”这一步：

* fp16 有 11 位有效位（含隐含前导 1），机器精度 `u = 2^-11 ≈ 4.88e-4`；
* bf16 有 8 位有效位，`u = 2^-8 ≈ 3.91e-3`；
* fp32 累加自身的误差（`u ≈ 6e-8`）比它们小四个数量级，是二阶项。

对一个长度 K 的归约（dot、GEMM 的 K 维），输入舍入误差相对 `S = Σ|a_i·b_i|` 的
最坏情形是 `u`，均方根是 `u/√K`。这就是“长归约反而更准”的来源，也是为什么
**累加类型必须写 fp32**：`T.alloc_fragment((...), accum_dtype)` 里的 `accum_dtype`
一旦写成 fp16，误差就从 `u_fp16` 级别跳到“每次加法都舍入”级别。

`code/02-tolerance-model/tolerance_model.py` 把这套推导做成了可运行的脚本，并用
`fp16/bf16 输入 + fp32 累加 → 与 fp64 精确值比较` 做了实测：实测中位数与 `u/√K`
预算的比值稳定在 0.5 左右，说明这个量级模型可用。

**换算到常用的“相对结果”口径**：零均值数据下 `S / |Σa_i·b_i| ≈ √K`，所以相对结果
的误差可以放大到 `u·√K`（fp16、K=4096 约 3e-2）。**这就是长归约不能只用 `rtol`、
必须配 `atol` 兜住接近零的输出的原因。**

锁定版本里有两个现成的对拍工具，它们的缺省值就是这个量级：

* `tilelang.utils.tensor.torch_assert_close`
  （`_vendor/tilelang-v0.1.15/tilelang/utils/tensor.py:205`）：
  默认 `rtol=1e-2, atol=1e-3, max_mismatched_ratio=0.001`，
  注意它允许 0.1% 的元素不满足逐元素判据；
* `Profiler.assert_allclose`
  （`_vendor/tilelang-v0.1.15/tilelang/profiler/__init__.py:104`）：
  默认 `atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.01`。

**工程起点**：fp16 用 `rtol=1e-3, atol=1e-3`；bf16 用 `rtol=1e-2, atol=1e-2`；
再配一个 `max_mismatched_ratio`。然后把参考实现换成 **fp64** 再算一遍。如果 fp64
参考与 fp32 参考差得比你的 kernel 还大，那问题在参考实现，不在 kernel。

#### 3.7.2 定位“偶发结果错误”：先怀疑同步

同一份输入，有时对有时错，或者每次错的地方不一样。这类问题里**绝大多数是同步问题**，
而不是“随机的数值误差”。常见的三种：

1. **缺 barrier**：同一个 block 里，某个线程读到了另一个线程还没写完的 shared buffer。
   TileLang 会在 lowering 里插入同步（例如
   `_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:108` 的
   `ThreadSync(storage_scope)`），昇腾侧对应
   `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/__init__.py` 里的 InsertSync 逻辑。
   但如果你的写法让依赖关系无法被推导出来（比如通过不确定的别名访问同一块内存），
   插入的同步就可能不够。
2. **pipeline 缓冲版本错**：`T.Pipelined(..., num_stages=n)` 会用多份缓冲重叠搬运与
   计算，`stage` / `order` 与实际依赖不符时，某一轮会读到下一轮的数据。
3. **跨 block 的无序写**：多个 block 对同一地址做 `T.atomic_add` 或者自己实现归约时，
   浮点加法的顺序不确定，结果会有微小抖动；如果偏差很大，那就不是“正常的非确定性”，
   而是真的竞争。

**排查顺序永远是“先下并发，再加回并发”**：

1. 固定 seed 与输入，重复跑 N 次，先量出复现率；
2. 把 `num_stages` 降到 1、关掉流水、把 grid 降到单个 block。如果这时候变对了，
   就是同步/并发问题，继续第 3 步；如果还是错，那就是普通的正确性问题，回到 3.7.1；
3. 每次只加回一个维度（先加 block、再加 stages），找到第一次出错的位置；
4. 在那个位置用 `T.print` / `T.dump_tensor` 打出中间张量与索引。

还有一招：如果怀疑某个中间结果，把它 dump 出来和 **PyTorch 的中间结果**逐元素比较，
而不是只比最终输出。第 17 章训练 7B 模型时，这套“分段对拍”会反复用到。

#### 3.7.3 AutoDD：把失败的复现缩小

当你能稳定复现一个失败，但复现代码有一两百行时，用 AutoDD 自动缩小：

```bash
python -m tilelang.autodd examples/autodd/tilelang_buggy.py \
  --err-msg "T.gemm K shape check failed" -o minimized.py
```

它逐个候选地删代码，只保留“stdout/stderr 里仍然出现这段子串”的版本
（`_vendor/tilelang-v0.1.15/tilelang/autodd.py:1399`）。**前提是失败必须已经确定化**，
偶发问题先按 3.7.2 变确定，再交给 AutoDD。

### 3.8 性能（一）：先建立 roofline 预期

性能分析最常见的浪费是“先测一堆数，再想它意味什么”。正确顺序是先算出理论上下界：

$$
\begin{aligned}
T_{\text{mem}} &= \frac{\text{bytes\_moved}}{\text{HBM带宽}},\\
T_{\text{compute}} &= \frac{\text{total\_flops}}{\text{峰值算力}},\\
T_{\text{sol}} &= \max(T_{\text{mem}},T_{\text{compute}}),\\
\eta &= \frac{T_{\text{sol}}}{T_{\text{actual}}}.
\end{aligned}
$$

$T_{\text{sol}}$ 是 roofline 给出的时间下界，$\eta$ 是实际效率。

这段定义与“`bytes_moved` 用算法最小搬运量、不是 profiler 测到的 DRAM 流量”这些约定，
在 `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md` 里有
完整说明（TileOPs 是另一个算子库，这里只借它的方法学，不是 TileLang 的 API）。

TileLang 自带一个静态估算器，把这套流程做成了代码：
`tilelang.tools.Analyzer`（`_vendor/tilelang-v0.1.15/tilelang/tools/Analyzer.py:196-216`）。

```python
from tilelang.carver.arch import CUDA
from tilelang.tools import Analyzer

tir = matmul.get_tir(block_M=128, block_N=128, block_K=32)
result = Analyzer.analysis(tir, CUDA("cuda"))
print(result.total_flops, result.total_global_bytes, result.estimated_time)
```

它的输出字段是 `total_flops`（只统计 `T.gemm`）、`total_global_bytes`（只统计跨 global
边界的 `T.copy`）、`estimated_time`（取 compute 与 memory 的较大者）、
`expected_tflops`、`expected_bandwidth_GBps`。

**必须清楚它的局限**（官方 `.../tools/analyzer.md` 的 Limitations 一节列得很全）：

* 它是**静态估算**，不是 GPU 实测；
* FLOPs 只认 `T.gemm`，逐元素运算不计；
* 字节只认触及函数参数的 `T.copy`，直接 load/store 不计；
* grid 缩放只跟踪 `blockIdx.x` / `blockIdx.y`，`blockIdx.z` 不计；
* 内置峰值表只有 NVIDIA compute capability 8.0 / 8.6 / 8.9。

另外，`CUDA("cuda")` 会去查询设备 0，所以**构造 device model 这一步需要可见的 GPU**；
没有 GPU 的机器上，只能读到别人报告里的 `expected_tflops` 与带宽，或者自己查
`_vendor/tilelang-v0.1.15/tilelang/carver/arch/cuda.py` 里的表。

图 4（`figures/04-roofline-before-measure.md`）画的就是这个流程：先有预期，再用
profiler 回答“差在哪一项”。

### 3.9 性能（二）：NVIDIA 侧的 Nsight Compute

Roofline 告诉你“差多少”，Nsight Compute（`ncu`）告诉你“差在哪”。corpus 里有完整
的官方文档镜像：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-nsight-compute-docs.md`。

**最小可用流程**（CLI 与 UI 等价）：

1. 用 kernel 过滤把范围缩小到一个 kernel，跳过 warmup 的若干次 launch；
2. 采集；报告生成后默认落在 **Summary** 页，Details 页按 section 组织；
3. 先看 **GPU Speed Of Light Roofline Chart**，把实测吞吐与硬件峰值放在同一张图上，
   一眼看出是被内存墙还是被计算墙卡住（`.../nvidia-nsight-compute-docs.md:1119-1129`）；
4. 再看 **Memory Workload Analysis**：它的 Memory chart 画出数据搬运、各级 cache 命中率、
   指令数与 memory request（同文件 `:1107-1109`）；
5. 然后看 **Occupancy** 与 **Launch Statistics**：占用率低不一定是坏事，但如果
   响应延迟型停顿很多，就要看是寄存器/共享内存限制，还是 block 太小；
6. 最后看 **Source Counters** 的 hot spot 表与 **warp stall** 采样：stall 类型里
   long/short scoreboard 是最常见的两类（同文件 `:1249-1258`），完整的原因表见文档里
   指向的 Warp Stall Reasons table；
7. 报告里的 **rules / Recommendations** 会直接给出问题与“预计能提速多少”
   （`Est. Speedup` / `Est. Local Speedup`），可以当作下一步的方向。

两个使用要点：

* **一次只改一个变量**，改完用同一 section 集合复采，并用报告的
  Compare / Baseline 功能做差异对比（文档 `:1090-1100`）；
* profiler 会**重放** kernel 来采集不同 pass 的计数器（corpus 文档里提到 Range Replay
  的限制），所以被 profile 的 kernel 必须可重复执行、输入不能被破坏。

### 3.10 性能（三）：昇腾侧的 msProf

昇腾侧的入口是 `msprof`。最常见的采集方式是一键式：

```bash
msprof --output=<path> <app>
```

它会在 `--output` 下生成 `PROF_XXX` 目录，里面默认包含 `msprof_*.db`、
`msprof_*.json`（timeline）、`op_summary_*.csv`、`op_statistic_*.csv`、
`task_time_*.csv` 等（见
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/CANNCommunityEdition/CANNCommunityEdition_920beta1_devaids_Profiling_atlasprofiling_16_0005_html.md`）。
json 用 `chrome://tracing` 打开看时序，csv 直接看每个算子在硬件上的耗时。
在 Ascend RC（以 PCIe 从模式跑）的场景下，采集和解析是两步：先把 `PROF_XXX` 目录拿回
开发环境，再执行 `msprof --export=on --output=<dir>` 导出。

算子级调优用 `msprof op`，分**上板（device）**与**仿真（simulator）**两种模式；
仿真模式会给出指令级流水图（`visualize_data.bin` 或 `trace.json`），例如：

```bash
msprof op simulator --soc-version=Ascend910B --output=./output_data ./execute_sqrt_op
```

注意 `--soc-version` 要写你实际用的型号。

**在 TileLang 里更方便的一条路**是直接用 benchmark 的 profiler 后端：

```python
from tilelang.profiler import do_bench
samples_ms = [do_bench(run, backend="msprof", warmup=25, rep=100) for _ in range(5)]
profile = do_bench(run, backend="msprof_detail")
print(profile.dur_ns / 1e6, profile.aic_mad, profile.aiv_vec)
```

`tilelang/profiler/bench.py` 的 `do_bench(backend=...)` 支持 `msprof` 与 `msprof_detail`
两个后端（`_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:39`）：

* `backend="msprof"` 返回**毫秒**数；
* `backend="msprof_detail"` 返回 `KernelProfile`
  （`_vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py:59`），字段含 `dur_ns` 与
  AIC 侧 `mad / scalar / mte1 / mte2 / mte3 / fixpipe`、AIV 侧 `vec / scalar / mte2 / mte3`
  的**利用率比例**。

三个必须记住的口径：

1. 这些比例的分母是各自的 `aic_total_cycles` / `aiv_total_cycles`，**不能相加**成
   一个“总利用率”（官方 profiling 参考里专门强调过这一点）；
2. 它会先跑一个 cache-flush kernel 再计时，并在解析结果时过滤掉这个 kernel；
   如果你的 callable 启动了多个 kernel，结果会把它们**求和**，需要自己拆开看；
3. 解析依赖 `ffts_profile` 等原始文件；如果记录为空或缺少目标 kernel，即使脚本退出码
   是 0，这个 profiler 结论也不成立。要检查启动与解析日志。

做指令级、周期级的测量时，还可以用 `npusim record` 抓取指令流水（例如
`npusim record -s Ascend950 -o ./cycle-profile -g -n 0 'python launcher.py'`），
再用同一套 CANN 的 `instr_decoder` 解码；细节与“怎么把区间换算成 cycles”的注意事项见
`../tilelang-corpus/02-tilelang-core-repo/raw/.agents/skills/tilelang-ascend/references/profiling.md`。

---

## 4. 实现机制

### 4.1 日志来自 TVM

TileLang 没有自己的日志系统，它用的是 TVM 的：实现是
`_vendor/tilelang-v0.1.15/src/runtime/logging.cc`。

* `level_strings[]`（`:17-23`）定义五级前缀：Debug / Info / Warning / Error / Fatal；
* `LogMessageImpl`（`:51-64`）负责格式化输出，Release 构建里会省掉文件路径；
* `LogFatalImpl`（`:65-70`）在打印后抛 `tvm::ffi::Error`。所以一条 `Check failed:`
  最终会变成一个 Python 异常；
* `TvmLogDebugSettings::ParseSpec`（`:73-135`）解析 `TVM_LOG_DEBUG` 的说明符。

CMake 的 Debug 模式通过 `target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")`
把 `DLOG` 编进去（`_vendor/tilelang-v0.1.15/CMakeLists.txt:487-490`）。

### 4.2 IR Lower Trace 的三层实现

IR Lower Trace 由三层协作（`.../tools/lower_trace.md` 的 How It Works 一节）：

1. **TVM `PassInstrument`**：在每个 pass 的前后回调里抓 `str(mod)`、算增删行数，
   追加一条 `LowerRecord`；不在任何 pipeline 窗口里的 pass 被标成 `unscoped`。
2. **`PassPipeline.lower` 作用域**：给调用方拥有的 session 贡献一个 phase 标签
   （如 `pipeline_c`），使报告能按阶段分组。
3. **后端 codegen hook**：device/host codegen 注册表把每个 `target.build.*` 调用
   路由进当前 session，捕获 TIR→源码这一步，并驱动三文件
   （`codegen.cpp` / `.original` / `.latest`）的 edit-and-recompile 工作流。

它**不是**通过替换全局 FFI 实现的：`enable()` 只注册配置，真正被影响的是 TileLang 自己
管理的编译会话。这一点很重要。它不会污染你进程里别的 TVM `PassContext`。

### 4.3 `T.print` 的实现与门控

CUDA 侧的 `print()`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/print.py:128-207`）
是一个 Python 函数而不是宏，它在**构造 IR 的时候**就决定了门控：

* `tirx.Buffer`：按 `buffer.scope()` 分流。`local` / `global` 走 `condition = True`；
  `local.fragment` 先把 buffer `copy` 到一块 shared（`print_fragment_buffer_with_condition`），
  再用 `tx == main_lane` 门控；
* `tirx.PrimExpr`：直接生成 `debug_print_var` 调用，**没有门控**；
* `None`：生成 `debug_print_msg`。

最终落到 C++ 模板（`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/debug.h`）里的
`printf`。昇腾版（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/print.py:31`）
复用同一批 `debug_print_*` extern，但去掉了 CUDA 式的单线程门控。因为 NPU 上一个
kernel body 本来就按 AI core 执行，UB/L1 是核内私有的。

### 4.4 Analyzer 与 profiler

`Analyzer`（`_vendor/tilelang-v0.1.15/tilelang/tools/Analyzer.py:16-33` 定义结果结构，
`:196-216` 汇总结算）走一遍 TIR：识别 `T.gemm` 累加 FLOPs、识别跨 global 边界的
`T.copy` 累加字节，再乘以所在循环与 grid 的 extent，最后
`estimated_time = max(mem_time, compute_time)`。

运行期的计时在 `_vendor/tilelang-v0.1.15/tilelang/profiler/` 下：
`bench.py` 是统一入口，`torch_bench.py` 覆盖 CUDA event / CUPTI / CUDA graph，
`msprof.py` 覆盖昇腾，`wall.py` 是纯主机计时。`msprof.py` 里
`_parse_ffts_profile` 逐个读 `ffts_profile*` 二进制记录，按 `total_cycles` 归一化出
AIC/AIV 各 pipe 的比例（`_vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py:184-266`）。

### 4.5 昇腾异常 dump 的 MAGIC 机制

异常回调拿到的是一段**原始参数缓冲区**，回调本身不知道哪几个位置是 tensor。
TileLang 的做法是在编译时往 kernel 参数末尾附加一个 `ParamSizeInfo` 结构体，里面放：

* 一个 8 字节 MAGIC 值 `0x474e414c454c4954`，正是 ASCII 的 `"TILELANG"`；
* kernel 名（用作 dump 文件前缀）；
* 每个 tensor 的字节数、设备地址、ACL 数据类型编码。

回调按 8 字节步长在参数缓冲区里搜 MAGIC，找到 `ParamSizeInfo` 后就能读出所有 tensor 的
描述，构造 CANN 的 `acldumpTensorInfo` 数组并调用 `acldumpSaveExceptionInfo` 落盘。
这个设计使回调代码与 kernel 的具体参数布局**完全解耦**，无论你的 kernel 有几个输入，
回调都不需要改。实现见
`_vendor/tilelang-ascend-v0.1.2.000/src/tl_templates/ascend/exception_dump.h`，完整原理
见 `.../ascend/docs/tutorials/ascend_aicore_exception_dump_zh.md`。

---

## 5. 进阶与陷阱

### 5.1 常见错误模式清单

下表把本章内容压成一张排查卡。每一条都是“症状 → 可能原因 → 定位手段”，
定位手段里的工具都在前面出现过。

| # | 症状 | 可能原因 | 定位手段 |
| --- | --- | --- | --- |
| 1 | lowering 中途 `Check failed: ...` | 某个 tile op 的形状/类型不满足约束（如 `T.gemm` 的 K 维、dtype 组合） | 读报错原文的 `文件:行号` → compile-only 看产物 → `TL_LOWER_TRACE` 找到是哪个 pass 引入的 |
| 2 | codegen 阶段报 `lanes <= 4` 之类的断言 | 引入了不被支持的向量化宽度（ramp 太长） | 看 Lower Trace 的 `codegen_before.tir`，检查 `T.Parallel` / 显式 vectorize 的写法 |
| 3 | Python 层抛异常，traceback 指向你的 kernel 行 | 前端语言错误（参数、类型、作用域），不是编译器 bug | 看 traceback 最后一帧，对照第 08 章的语言规则 |
| 4 | `CUDA codegen FFI missing` | 用了 macOS Metal wheel，没编进 CUDA codegen | 换 Linux wheel，或只用 `--target c` |
| 5 | `target must be explicit; do not use auto` | compile-only 不接受 `auto` | 显式写 `--target c` 或 `--target cuda` |
| 6 | kernel 一启动就 illegal memory access / 越界 | 索引计算错，或没处理 tail / OOB padding | 降为单 block，用 `T.print` / `T.dump_tensor` 打索引与 tile 内容 |
| 7 | kernel 卡死、超时 | 同步点数量不匹配（barrier 数不一致、wait 了永远不会来的 flag） | 关流水、单 block、二分定位 |
| 8 | 昇腾上 AI Core 报硬件异常 | 非法访问 / MTE 错误 / Cube、Vector 溢出 | 开 `TL_ASCEND_EXCEPTION_DUMP` 取输入现场，再用 `parse_exception_dump` 读回 |
| 9 | 输出整块是零或明显未初始化 | 忘了 `T.clear`，或 `T.copy` 方向反了 | 打印 accumulator 与输出 tile（`T.print` / `T.dump_tensor`） |
| 10 | 少量元素错，位置恰好落在 tile 边界 | 边界条件 / OOB padding 处理错 | 打印边界索引与对应元素；用非整齐 shape（如 M=1023）复现 |
| 11 | 结果系统性偏小或偏大 | 累加类型写成 fp16，或 epilogue 的 scale / cast 错 | 检查 `accum_dtype` 与 epilogue；与 fp64 参考对比误差量级 |
| 12 | 每次结果略有不同（小抖动） | 浮点归约顺序不确定（atomic、split-K） | 若抖动在容差内，接受并用统计容差；若偏差大，转 3.7.2 |
| 13 | 只有 `num_stages>1` 时才错 | pipeline 缓冲版本或依赖不对 | `num_stages=1` 对比，二分加回 stage 数 |
| 14 | 远低于 roofline 预期，且是 memory-bound | 多搬了字节：未融合的中间张量、非合并访问、重复 load | Nsight Compute 的 Memory Workload / SOL；昇腾看 msProf 的 `mte2` 比例 |
| 15 | compute-bound 但 tensor core 利用率低 | tile 形状太小、occupancy 低、K 维切分不合适 | Nsight Compute 的 SOL + Occupancy + stall；昇腾看 `mad` 比例 |
| 16 | 数字很慢，但其实不是 kernel 慢 | 没 warmup、没清 cache，或者用 wall clock 测了端到端 | 用 `do_bench` 的 event / msprof 后端，区分 kernel 时间与端到端时间 |
| 17 | 加了打印后明显变慢 | device printf 串行化；昇腾每核 1 MB dump 上限被打满 | 移除打印后重测；把打印限制在一个线程 |

### 5.2 性能数字的口径

本章涉及的“数字”有四种口径，混用会得出错误结论：

| 口径 | 来源 | 能回答什么 | 不能回答什么 |
| --- | --- | --- | --- |
| 静态估算 | `Analyzer` | 这个形状理论上限在哪 | 你的实现实际有多快 |
| kernel 耗时 | `do_bench` / profiler | 单个 kernel 多久 | 端到端为何慢 |
| 硬件计数器 | Nsight Compute / msProf | 时间花在哪个单元 | 算法是否有冗余工作 |
| 编译耗时 | `pass_timing` | 编译器为什么慢 | kernel 为什么慢 |

另外两条纪律：

* **文本变化 ≠ 语义变化 ≠ 性能变化。** IR Lower Trace 比较的是 `IRModule.script()`
  的文本；某个 pass 让文本变了，不代表这个变化影响了结果或性能。
* **昇腾的 pipe 利用率不能相加。** AIC 的 `mad / mte1 / mte2 / fixpipe` 与 AIV 的
  `vec / mte2 / mte3` 分母各自不同，把它们加成一个“总利用率”是错的。

### 5.3 两侧工具对照与本机 / 有卡分工

| 环节 | NVIDIA | 昇腾 |
| --- | --- | --- |
| 编译期看产物 | `python -m tilelang.tools.compile_only --target cuda` | `tilelang.lower(..., target="ascend")` 取 `kernel_source` |
| pass 前后 | `TL_LOWER_TRACE` / `pass_visualizer` | `TL_LOWER_TRACE`（backend 相关 pass 会打在对应 phase 下） |
| 设备端打印 | `T.print`（CUDA 方言，带单线程门控）+ `T.device_assert` | `T.print`（昇腾方言，无 CUDA 式门控）、`T.printf`、`T.dump_tensor` |
| 异常现场 | 依赖 CUDA 工具链与 `device_assert` | `TL_ASCEND_EXCEPTION_DUMP` + `parse_exception_dump` |
| 硬件 profiler | Nsight Compute（SOL / Occupancy / Memory / Stall） | msProf（`msprof op`、`do_bench(backend="msprof")`）、`npusim record` |

**本机（无卡）能做的**：3.1–3.4（除 `--target cuda` 的 FFI 限制外全部可做）、3.7 的
对拍脚本与容差推导、3.8 的 roofline 预期（除 `CUDA("cuda")` 那一步）。
**必须有卡才能做的**：3.5、3.6 的实际输出，3.9、3.10 的采样数据，以及任何实测误差与耗时。

### 5.4 版本边界

本章所有结论都锚定在锁定版本上：

* `TL_LOWER_TRACE`、`lower_trace()`、`pass_visualizer`、`compile_only`、`Analyzer`、
  `plot_layout`、`pass_timing`、`T.print`（CUDA 与昇腾两种方言）、`T.device_assert`、
  `do_bench(backend="msprof")`：都在
  `_vendor/tilelang-v0.1.15/` 里（可用 `code/01-tool-inventory/` 复核）。
* `T.printf`、`T.dump_tensor`、`TL_ASCEND_EXCEPTION_DUMP`、`parse_exception_dump`：
  在 `_vendor/tilelang-ascend-v0.1.2.000/`（昇腾适配器仓库）里，**不在主干 v0.1.15 里**。
* `Pass Visualizer` 的 CLI 目前只支持 CUDA 目标；昇腾侧要走 backend 相关 pipeline 时，
  以 IR Lower Trace 的 phase 分组为准。

如果上游发布了新 tag，先重跑 `code/01-tool-inventory/tool_inventory.py`；它报“缺失 N 个”
的地方，就是本章需要重新核对的地方。

---

## 6. 本章小结

* 问题分四类：生成、正确性、性能、偶发。**先分类，再选工具**（图 1）。
* 编译期工具（日志、compile-only、IR Lower Trace / Pass Diff / Pass Visualizer、
  布局可视化、pass_timing）**都不需要加速器**；这是“没卡也能做一半调试”的底气。
* 运行期工具（`T.print` / `T.printf` / `T.dump_tensor` / `device_assert` / 异常 dump）
  都有代价与上限（打印串行化、昇腾每核 1 MB），只在排障时开，调完就删。
* 正确性对拍要先定容差：fp16 `u ≈ 4.9e-4`、bf16 `u ≈ 3.9e-3`；长归约必须配 `atol`。
* 偶发错误先怀疑同步：**先下并发，再加回并发**。
* 性能分析先算 roofline 预期，再用 Nsight Compute / msProf 回答“差在哪一项”，
  并且始终说清楚数字是哪种口径。
* 一切结论都要能追溯：报错原文、IR 快照、profiler 报告、`.tir` 文件，都可以留在
  你的工作目录里，成为下一次回归的基线。

## 延伸阅读

* 编译期调试全流程：`R1`（debug_tools_for_tilelang）、`R2`（logging）。
* 看 IR 变化：`R3`（lower_trace）、`R4`（pass_diff）、`R5`（debug_tools 的 Pass Visualizer 一节）。
* 不跑 GPU 看产物：`R6`（compile_only）。
* 布局：`R7`（layout_visualization）。
* 性能估算与 profiler：`R8`（analyzer）、`R9`（Nsight Compute）、`R10`（msProf 采集）。
* 昇腾运行期调试：`R11`（print/dump_tensor）、`R12`（AICore 异常 dump）。
* 代码示例：`code/01-tool-inventory/`、`code/02-tolerance-model/`、
  `code/03-compile-only/`、`code/04-lower-trace/`、`code/05-print-and-assert/`、
  `code/06-ascend-dump/`。

各条目的完整路径与核验状态见 [references.md](references.md)。
