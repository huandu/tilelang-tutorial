# 第 12 章 习题

说明：

* 难度：`★`（读完正文即可作答）/ `★★`（需要综合本章多处内容）/ `★★★`（需要动手、
  多步推理，或需要对照源码）。
* “需要硬件”一栏中，`否` 表示只要一台能跑 Python 的机器；`是（GPU）` 表示必须用
  真实的 NVIDIA GPU；`是（NPU）` 表示必须用真实的昇腾 NPU；`否（但要读源码）` 表示
  不需要硬件，但必须在 `_vendor/` 里核对。
* 所有题目只依赖第 01–12 章的知识。涉及后续章节的地方会显式标注。
* 本章的**必做部分不要求任何硬件**；标“是（GPU）/ 是（NPU）”的题都是选做的动手题。
* 写作环境没有 GPU/NPU（见 `../../VERSIONS.md`），所以本章习题的参考答案里凡是涉及
  实测的，都只给“怎么测”与“判据”，不给“测出来是多少”。

---

## E1 概念题 ★ 不需要硬件

正文 1.1 节把问题分成四类：生成、正确性、性能、偶发。

1. 写出这四类各自的**可观测信号**（即你靠什么判断属于这一类）。
2. “偶发”为什么没有被并进“正确性”？举一个例子说明：同样表现为“结果错”，
   这两类问题的排查入口为什么不同。
3. 正文说“拿 profiler 去查一个编译错误是浪费”。请再补一个“用错了工具”的例子，
   并说明正确工具应该是什么。

---

## E2 概念题 ★ 不需要硬件

正文 1.3 节把工作分成“本机可做”与“必须在有卡机器上做”。

1. 判断下面每一项属于哪一边，并说明理由：
   （a）用 `TL_LOWER_TRACE=1` 看某个 pass 前后的 IR；
   （b）拿到 `T.print` 在 kernel 里的实际输出；
   （c）用 `compile_kernel_source(func, "cuda")` 打印 CUDA 源码；
   （d）跑 `Nsight Compute` 得到一张 SOL 图；
   （e）算一个 GEMM 的 `memory_time` 与 `compute_time`。
2. （c）里为什么要特意注明“不需要 GPU”？它需要什么？

---

## E3 概念题 ★★ 不需要硬件

正文 3.1 节讲了 `TVM_LOG_LEVEL` 与 `TVM_LOG_DEBUG`。

1. 这两个名字分别控制什么？为什么说“同一个名字 `TVM_LOG_DEBUG` 被用在两个地方”
   是官方文档容易让人困惑的地方？
2. 为什么 Release wheel 里设 `TVM_LOG_DEBUG` 也看不到 `DLOG` 输出？请结合
   `_vendor/tilelang-v0.1.15/CMakeLists.txt` 里的做法回答。
3. 一条 `Check failed: ...` 最终以什么形式出现在 Python 层？请指出
   `_vendor/tilelang-v0.1.15/src/runtime/logging.cc` 里负责这一步的函数。

---

## E4 概念题 ★★ 不需要硬件

正文 3.3 节比较了 IR Lower Trace、Pass Diff、Pass Visualizer 三个工具。

1. 三者各自回答什么问题？请按“整条 pipeline / 指定的一串 pass / 结构树”来区分。
2. 什么情况下你应该选 Pass Visualizer 而不是 IR Lower Trace？
3. IR Lower Trace 的报告里，`codegen` 那一条和普通 pass 那一条有什么不同？
   （提示：看它的输出文件扩展名。）

---

## E5 概念题 ★★ 不需要硬件

正文 4.2 节说 IR Lower Trace 用三层实现，并且“不是通过替换全局 FFI 实现的”。

1. 写出这三层的名字与各自职责。
2. 为什么说“不替换全局 FFI”是一件重要的事？如果一个工具替换了全局 FFI，
   在什么场景下会出问题？
3. 文档说 pass 记录是“运行时追加”而不是“预先注册”的。请解释这为什么让报告更可信
   （提示：想一个会被条件跳过、根本没执行的 pass）。

---

## E6 代码阅读题 ★★ 不需要硬件（但要读源码）

阅读 `_vendor/tilelang-v0.1.15/tilelang/cuda/language/print.py` 的 `print()` 函数
（`:128-207`），回答：

1. 当传入的是一个 `local.fragment` 的 buffer 时，它做了哪两步？为什么要多一步？
2. 当传入的是一个 `global` 的 buffer 时，门控条件是什么？直接在 GPU 上打印一个大
   的 global tensor 会发生什么？
3. 当传入的是一个 `PrimExpr`（比如 `bx + by`）时，`print()` 有没有做门控？
   这给使用者留下了什么责任？
4. `main_lane` 是怎么算出来的？写 `if tx == main_lane:` 和写 `if tx == 0:` 有什么差别？

---

## E7 代码阅读题 ★★ 不需要硬件（但要读源码）

阅读 `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py` 的 `printf` 与
`dump_tensor`（`:534`、`:582`），回答：

1. `printf` 对 `%p` 做了什么处理？为什么官方建议直接用 `%x` 打地址？
2. `dump_tensor` 的 `desc` 参数有类型限制吗？如果传一个负数会怎样？
3. `dump_tensor` 把 `shape_info` 变成了什么传给底层 intrinsic？
4. 这两个接口各自生成哪个 intrinsic？为什么说它们是“设备端”工具
   （提示：正文 3.6 节说主机端该用什么）？

---

## E8 代码阅读题 ★★ 不需要硬件（但要读源码）

阅读 `_vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py` 的 `KernelProfile`
（`:59-135`）与 `bench_with_msprof`（`:40-306`）。

1. `KernelProfile` 里 AIC 侧与 AIV 侧各有哪些 pipe 字段？
2. 为什么这些比例**不能相加**成一个“总利用率”？（提示：看 `__add__` 是怎么按
   cycle 加权平均的，以及每个比例各自的分母。）
3. 计时前为什么要先跑一个 cache-flush kernel？解析结果时又为什么要把它过滤掉？
4. 如果一个 callable 里启动了三个 kernel，`bench_with_msprof` 返回的 `dur_ns`
   代表什么？

---

## E9 动手题 ★★ 不需要硬件

按下面的步骤，用本章的示例把一个 kernel 的编译产物看清楚（**全部不需要加速器**）：

1. 运行 `python3 chapters/12-debugging-and-profiling/code/01-tool-inventory/tool_inventory.py`，
   记录它报出的入口总数与缺失数；
2. 打开 `code/03-compile-only/add_kernel_compile_only.py`，回答：脚本为什么把
   `import tilelang` 放进函数内部，而不是放在文件顶部？
3. 用一句话说明：为什么“compile-only 能生成 CUDA 源码”不等于“这台机器一定能跑
   CUDA kernel”？
4. 如果你要把 `TL_LOWER_TRACE` 打开，为什么必须在 `import tilelang` **之前**设好
   环境变量？

---

## E10 动手题 ★★ 不需要硬件

运行 `python3 chapters/12-debugging-and-profiling/code/02-tolerance-model/tolerance_model.py`
（默认参数），然后回答：

1. fp16 与 bf16 的机器精度分别是多少？它们的比值大概是多少？
2. “中位数 / 预算”这一列在 fp16 与 bf16 下分别是多少？这说明
   `u/√K` 这个模型是不是可用？
3. 把 `--lengths` 改成只有 `4096`，再跑一次。相对误差的**中位数**是变大还是变小？
   为什么？这和“长归约必须配 atol”矛盾吗？
4. 把这个脚本的输出与正文里的“工程起点”（fp16 `rtol=1e-3`）对照：为什么示例给的
   建议 `rtol` 比 `u/√K` 大很多？

---

## E11 动手题 ★★★ 需要硬件（GPU）

**只有在你有一块 NVIDIA GPU 时做。** 目标：把“偶发结果错误 = 同步问题”这条经验
走一遍。

1. 写一个用 `T.Pipelined(..., num_stages=3)` + `T.copy` + `T.gemm` 的小 GEMM kernel，
   先确认它在 `num_stages=3` 下结果正确；
2. 人为引入一个错误：把某个中间 shared buffer 的复用改成“同一份缓冲被两个不同
   阶段的写入共用”（例如把两个不同 `T.copy` 的目标写成同一块 `alloc_shared`）；
3. 观察：(a) `num_stages=1` 时是否还错？(b) `num_stages=3` 时是否错？
   记录你实际看到的差异；
4. 用 `TL_LOWER_TRACE=terminal python your_script.py`，在报告里找出
   `ThreadSync` 相关的 pass，说明它对这段 IR 做了什么；
5. 写下你的结论与判据。**不要凭想象填数字**，记录你实际观察到的现象。

---

## E12 动手题 ★★★ 需要硬件（NPU，选做）

**只有在你有一台装好 CANN (>= 9.3.0) 的昇腾机器时做。**

1. 按 `code/06-ascend-dump/ascend_dump_demo.py` 的骨架，写一个会**故意**触发越界
   访问的 kernel；
2. 在设置 `ASCEND_DUMP_PATH` 与 `ASCEND_DUMP_SCENE` 之后再 `import torch`，
   并用 `pass_configs={tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True}` 编译；
3. 捕获异常后调用 `parse_exception_dump()`，检查返回的 tensor 列表；
4. 回答：如果忘了在 `import torch` 之前设置环境变量，会发生什么？为什么？
5. 再回答：`T.dump_tensor` 与异常 dump 这两条路径，各自适合什么场景？
