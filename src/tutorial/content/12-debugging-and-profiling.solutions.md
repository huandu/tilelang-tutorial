# 第 12 章 习题答案

每题按“解题思路 → 参考答案 → 常见错误”来写。凡涉及实测的题，只给**怎么测**与**判据**，
不给具体数字，写作机没有 GPU/NPU，任何数字都必须在读者自己的机器上取。

---

## E1 解题思路与答案

**思路**：这道题考的是“分类”这个动作本身。分类的依据必须是**可观测的**，不能是猜的。

**参考答案**

1. 四类的可观测信号：

   | 类别 | 信号 |
   | --- | --- |
   | 生成 | 有报错文本（lowering 抛异常、codegen 断言、编译失败） |
   | 正确性 | 有参考（同一个输入，输出与参考不符） |
   | 性能 | 有预期（实测耗时明显高于 roofline 预期） |
   | 偶发 / 非确定性 | 有复现率（同一输入多次运行结果不一致） |

2. 因为**排查入口完全不同**。举例：一个 GEMM 每次都错 3% 的数值，那是容差/累加类型
   问题，入口是“对齐参考实现与容差口径”（正文 3.7.1）；而同样表现为“结果错”，
   如果它只在少数几次运行里错、且错的位置每次不同，那入口是“先去掉并发”
   （`num_stages=1`、单 block，正文 3.7.2）。前者要读数值，后者要读时序。

3. 例如：用 `T.print` 去查“编译期报错”是浪费，报错在编译期就发生了，kernel 根本没跑，
   打印不会有任何输出。正确工具是 compile-only（看编译产物）与 IR Lower Trace
   （看是哪个 pass 引入的变化）。

**常见错误**：把“偶发”当成“随机数值误差”直接放宽容差，掩盖掉真实的同步 bug。

---

## E2 解题思路与答案

**思路**：判断依据是“这一步是否必须执行真实的 kernel”。

**参考答案**

1. （a）本机可做，IR Lower Trace 只做 lowering，不执行 kernel；
   （b）必须在有卡机器上做，device 打印要真的把 kernel 跑起来；
   （c）本机可做，CUDA 目标的 codegen 只把架构钉死成 sm_80，不探测设备、不调用 nvcc；
   （d）必须在有卡机器上做，Nsight Compute 要真的采样硬件计数器；
   （e）本机可做，`memory_time` / `compute_time` 是纸上计算（`Analyzer` 里唯有构造
   `CUDA("cuda")` 那一步需要设备）。

2. 因为“能生成 CUDA 源码”和“能不能在本机跑 CUDA”是两件事。（c）需要的是 wheel 里
   编进了 CUDA codegen FFI（Linux wheel 有，macOS Metal wheel 没有）；它**不需要**
   GPU、不需要 CUDA driver、不需要 nvcc。

**常见错误**：以为“本机没有 GPU”就等于“本机什么都做不了”。本章一半以上的工具都不
需要卡。

---

## E3 解题思路与答案

**思路**：回到源码 `_vendor/tilelang-v0.1.15/src/runtime/logging.cc` 与
`_vendor/tilelang-v0.1.15/CMakeLists.txt` 核对，而不是凭记忆。

**参考答案**

1. `TVM_LOG_LEVEL` 是运行时日志等级；`TVM_LOG_DEBUG` **同时**是编译期宏（决定 DLOG
   是否被编进 .so）与运行时环境变量（决定每个文件的日志级别）。官方文档在
   “Enabling Log Output in Tilelang”里说要设 `TVM_LOG_LEVEL`，示例却写成
   `TVM_LOG_DEBUG=1 python3 code.py`，两个语义不同却同名的机制叠在一起，读者容易混淆。

2. 因为 `DLOG` 是**编译期**被死代码消除的：CMake 只在 Debug 模式（或显式打开）时执行
   `target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")`
   （`CMakeLists.txt:487-490`）。Release wheel 里 DLOG 语句根本不存在，运行时设环境变量
   也无效。

3. 以 Python 异常的形式。`LogFatalImpl`（`logging.cc:65-70`）在打印后抛出
   `tvm::ffi::Error`，穿越 FFI 边界后变成 Python 异常。

**常见错误**：在 Release 环境里反复调 `TVM_LOG_DEBUG` 期望看到 DLOG 输出。

---

## E4 解题思路与答案

**思路**：三个工具各有一条“比较对象”的定义，抓住这个定义就能分开。

**参考答案**

1. IR Lower Trace 回答“**整条 pipeline** 上每一步发生了什么”（环境变量一开就抓
   `tilelang.compile()` / `lower()` 里的每个 pass，外加最终 codegen）；
   `lower_trace()` 程序 API 与 `pass_diff` 回答“**我指定的这一串 pass** 前后变了吗”；
   Pass Visualizer 回答“**结构**变了吗”。它渲染的是 `SBlock` 的结构树，按字段展开
   tile op。

2. 当问题出在**结构性 pass**（布局推导、warp specialization、流水线）时。这类变化的
   文本 diff 会很碎（行序、缩进都变），而结构树能直接告诉你 `alloc_buffers` 多了一块、
   `T.copy` 的 `reads`/`writes` 变了。

3. 普通 pass 的前后产物都是 `.tir`；codegen 那一条的 after 产物是 `.cpp`（生成源码），
   因为它做的是 TIR → 源码的降低，不是 TIR → TIR。

**常见错误**：把 Pass Diff 当成首选。它在锁定版本里已被显式标注为“superseded”，
保留只是为了向后兼容。

---

## E5 解题思路与答案

**思路**：三层对应“pass 事件 / 阶段归属 / 源码产出”。

**参考答案**

1. 三层：
   * TVM `PassInstrument`：在 pass 前后回调里抓 `str(mod)`、算增删行数、记一条
     `LowerRecord`；
   * `PassPipeline.lower` 作用域：给当前 session 贡献 phase 标签（如 `pipeline_c`），
     让报告能按阶段分组；
   * 后端 codegen hook：把每个 `target.build.*` 调用路由进当前 session，捕获
     TIR→源码，并驱动三文件 edit-and-recompile。

2. 因为替换全局 FFI 是**进程级副作用**：同一进程里其他代码（例如另一个库、另一次编译）
   也会被你的替换影响，行为变得不可预期。IR Lower Trace 选择只影响 TileLang 自己管理的
   编译会话（`enable()` 只注册配置，会话在编译开始时快照配置），因此不会污染别人的
   `PassContext`。

3. 因为“被条件跳过的 pass”根本不会出现在报告里。如果实现成预先注册全部 pass，报告里
   就会出现一堆“跳过”的空槽位，读者无法区分“这个 pass 没执行”和“这个 pass 执行了
   但没改东西”。运行时追加避免了这种幻影行。

**常见错误**：以为 `enable()` 会改当前的 TVM `PassContext`。文档明确说不会。

---

## E6 解题思路与答案

**思路**：直接读 `print()` 的分支结构（正文给出了精确行号）。

**参考答案**

1. 两步：先把这块 fragment `copy` 到一块临时 `alloc_shared`，再在门控下用
   `debug_print_buffer_value` 逐个打印。多一步是因为 fragment 物理上分散在各个线程的
   寄存器里，单个线程无法直接读别人的寄存器；必须先汇到 shared。

2. 门控条件是 `condition = True`，也就是**不门控**，每个线程都会打印。所以打印一个大
   的 global tensor 会产生 `threads` 倍重复输出，很容易把 device 侧的打印缓冲打满，
   并显著拖慢 kernel。

3. 没有门控。`PrimExpr` 分支直接生成 `debug_print_var` 调用。所以责任在使用者：必须
   自己写 `if tx == 0:` 之类的条件。

4. `main_lane = warp_group_id * warp_group_size + warp_id * warp_size`，其中
   `warp_group_size = 128`、`warp_size = 32`。写 `tx == 0` 只会让 block 的第一个线程
   打印；写 `tx == main_lane` 在默认 `warp_group_id=0, warp_id=0` 时等价于 `tx == 0`，
   但在多 warp group（例如 warp specialization）的 kernel 里允许你通过参数选择由哪个
   warp group 打印。

**常见错误**：以为 `T.print(buffer)` 一定是单线程的。`local` 与 `global` 两种 scope 不
门控。

---

## E7 解题思路与答案

**思路**：读 `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py` 的两个函数体。

**参考答案**

1. `format_str = format_str.replace("%p", "0x%x")`，把 `%p` 直接改写成 `0x%x`。所以
   官方建议直接用 `%x` 打地址：`%p` 反正会被换掉，写 `%x` 更直观，也避免不同实现对
   `%p` 的解释差异。

2. 有类型限制：`desc` 必须是 `int` 且在 `[0, 0xFFFFFFFF]`（`uint32`）之内；否则直接
   抛 `ValueError`（`if not isinstance(desc, int) or desc < 0 or desc > 0xFFFFFFFF`）。
   传负数会抛异常。

3. 把一个 `len(shape_info)` 和 `shape_info` 的各维依次作为参数传给底层 intrinsic
   （`tl.ascend_dump_tensor`），由 codegen 负责格式化输出。

4. `printf` 生成 `tl.ascend_printf`，`dump_tensor` 生成 `tl.ascend_dump_tensor`。
   它们都是**设备端**工具，生成的调用要在 AI Core 上执行，所以要打开 Bisheng 的
   `-D_DEBUG --cce-enable-print`；主机端要打印就直接用 Python 的 `print`。

**常见错误**：误以为 `TL_PTO_DEBUG=1` 能打开 AscendC 的打印。它只对 `target="pto"` 起
作用。

---

## E8 解题思路与答案

**思路**：读 `KernelProfile` 的字段定义与 `bench_with_msprof` 的解析流程。

**参考答案**

1. AIC 侧：`aic_mad`、`aic_scalar`、`aic_mte1`、`aic_mte2`、`aic_mte3`、`aic_fixpipe`
   （外加 `aic_total_cycles`）。AIV 侧：`aiv_vec`、`aiv_scalar`、`aiv_mte2`、`aiv_mte3`
   （外加 `aiv_total_cycles`）。

2. 因为每个比例的分母不同：AIC 的比例除以 `aic_total_cycles`，AIV 的比例除以
   `aiv_total_cycles`。它们是**两组各自归一化**的占比，把两组数字加起来（或者在同一
   组内把“各 pipe 占比”当互斥份额相加）都没有物理意义。文档里也专门强调：详细比例是
   相对于对应的 AIC / AIV cycles，不要把它们当成一个公共时间区间上的互斥分数。

3. 计时前先跑 cache-flush kernel，是为了让每次计时的 cache 状态可比（否则第一次跑和
   后面几次的 cache 命中情况不同）。解析时过滤掉它，是因为它只是测量工具的一部分，
   不该计入被测 kernel 的耗时。

4. 代表这三个 kernel 耗时的**总和**（按 `n_repeat` 平均后相加）。所以要解释多个 kernel
   的结果时，需要自己按名字拆开看，而不是直接拿总和下结论。

**常见错误**：把 `aic_mad` 与 `aiv_vec` 相加，报出一个“总利用率 130%”。

---

## E9 解题思路与答案

**思路**：动手跑，然后把“为什么”答成机制层面的原因。

**参考答案**

1. 写作机上的实际输出是“合计 36 个入口，找到 36 个，缺失 0 个”，退出码 0。
   （你自己跑出来的数字可能因为版本/统计口径略有差异，以实际输出为准。）

2. 因为 `--static-only` 要在**没装 tilelang** 的机器上也能跑。如果 `import tilelang`
   写在文件顶部，`--static-only` 会先因为 ImportError 崩掉，这个“离线自查”入口就
   没用了。把导入放进 `build_kernel()` / `show_source()` 内部，才能让 `--static-only`
   只依赖标准库。

3. 因为 codegen 只是把 TIR 翻译成 CUDA C 文本；它**不调用 nvcc、不连接 CUDA driver、
   也不装载模块**。能不能真的跑 CUDA kernel，取决于机器上有没有可用的 GPU 与驱动，
   这和“wheel 里有没有编进 CUDA codegen FFI”是两件独立的事。

4. 因为 `TL_LOWER_TRACE` 是在 `import tilelang` 时被**读取并注册**的；import 之后再设
   环境变量，对已经注册好的会话不生效（要改就得调用 `lower_trace.enable()`）。这也是
   示例脚本把“设环境变量”放在“导入 tilelang”之前的原因。

**常见错误**：在 import 之后用 `os.environ["TL_LOWER_TRACE"] = "1"`，然后疑惑为什么
没有报告。

---

## E10 解题思路与答案

**思路**：先跑脚本拿到实际输出，再用“输入舍入是主项”这一模型解释。

**参考答案**

1. fp16 的 `u = 2^-11 ≈ 4.88e-4`，bf16 的 `u = 2^-8 ≈ 3.906e-3`；两者相差约 8 倍
   （正好是有效位从 8 位到 11 位的差距）。

2. “中位数 / 预算”在 fp16 与 bf16 下都稳定在 **0.5 左右**（写作机上的实测是
   0.46–0.51）。它随 K 几乎不变，说明 `u/√K` 这个量级模型可用：误差中位数确实按
   `1/√K` 下降。

3. 相对误差的**中位数变小**：`u/√K` 随 K 增大而减小。它和“长归约必须配 atol”不矛盾，
   因为两者用的基准不同：这里的相对误差以 `S = Σ|a_i·b_i|` 为基准（随 K 线性增长）；
   而日常断言用的“相对结果”以 `|Σa_i·b_i|` 为基准，在零均值数据下 `S ≈ √K·|Σa|`，
   所以换算过去能放大到 `u·√K`，随 K **增大**。输出的最后一列 `≈ u·√K` 就是那个换算。

4. 因为示例讲的 `u/√K` 是以 `S` 为基准的理论**中位数**量级，而 `rtol` 要覆盖的是
   “相对结果”口径下的**离群**情况（数据可能出现接近零的输出，相对误差会爆掉）。
   所以工程上给的 `rtol` 要比 `u/√K` 大得多，而且必须配 `atol` 兜住近零输出，
   这正是正文说“长归约不能只用 rtol”的原因。

**常见错误**：直接把 `u` 当成 `rtol`，忽略近零输出与统计分布，导致对拍时好时坏。

---

## E11 解题思路与答案

**思路**：这道题的目的是**亲手体验**“偶发错误 → 先下并发”这条流程，而不是背结论。

**参考答案（思路与判据）**

1. 先写一个正确的基线：`T.Pipelined(num_stages=3)` + 两个 `T.copy` 到各自独立的
   `alloc_shared` + `T.gemm`，用 `Profiler.assert_allclose` 对一个 fp64/参考实现确认
   通过。

2. 故意把错误引入：让两个不同的 `T.copy` 写到**同一块** shared buffer。这样在流水线里
   后一轮的写会覆盖前一轮还没读完的数据。

3. 预期与判据（**必须用你自己的实测替换**）：

   | 配置 | 预期现象 | 判据 |
   | --- | --- | --- |
   | `num_stages=1` | 往往**正确**或错误大幅减少 | 关掉重叠后，写-读依赖变简单 |
   | `num_stages=3` | 出错，且错的位置/次数**不稳定** | 多跑几次，结果不一致即为偶发 |

   如果 `num_stages=1` 也错，说明这更像普通正确性问题，不是同步问题。这本身就是
   有价值的结论。

4. `TL_LOWER_TRACE=terminal` 的报告里，找 `ThreadSync` 相关的 pass。你会看到它根据
   buffer 的读写依赖插入 `tvm_storage_sync`（或对应 intrinsic）类的语句；但当你把两块
   不同逻辑写进同一块缓冲时，编译器能推导出的依赖关系与实际意图不符，同步就不够，
   于是出现偶发错误。

5. **结论写法**：写清你的 kernel、两个配置下的现象、以及“关掉流水后是否恢复”。
   不要写“一定是缺少 barrier”，要写“在 num_stages=3 下偶发、在 num_stages=1 下
   消失，说明是流水线重叠暴露出的依赖问题”。

**常见错误**：直接读 InsertSync 的 C++ 源码找 bug，而不是先做 `num_stages` 对比实验。

---

## E12 解题思路与答案

**思路**：这道题考的是“把异常 dump 正确打开”的时序与前提，而不是真的要看异常。

**参考答案（思路与判据）**

1. 用 `code/06-ascend-dump/ascend_dump_demo.py` 的骨架；故意越界可以用一个明显超过
   张量边界的索引表达式写一次读/写。

2. 顺序必须是：先设 `ASCEND_DUMP_PATH` / `ASCEND_DUMP_SCENE`，**再** `import torch`，
   最后 `import tilelang` 并编译。编译时通过
   `pass_configs={tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True}` 打开开关；
   还要确认 CANN 版本 >= 9.3.0。

3. 在 `except` 块里调 `parse_exception_dump(dump_path, kernel_name=...)`，判据是返回
   非空的 `list[dict]`，每个 dict 含 `data`（`np.ndarray`，1-D）、`type`、`index`、
   `dtype`、`file`。因为回调是异步写盘的，函数内部会等一小段时间（`wait_seconds`）。

4. 如果忘了在 `import torch` / ACL 初始化之前设置环境变量，CANN 不会读取到这两个配置，
   于是**即使真的发生异常也不会落盘**（`ASCEND_DUMP_SCENE` 就是那个使能开关），
   `parse_exception_dump()` 会找不到 dump 文件而报错。原因是这两个变量在 ACL 初始化时
   被 CANN 读取一次，之后改就晚了。

5. `T.dump_tensor` 适合**主动**观察：你怀疑某个 tile 有问题，就直接把它的内容打到
   host（注意每个核 1 MB 上限、会影响性能）。异常 dump 适合**事后**取证：硬件已经出错、
   kernel 已经中断，此时需要把输入张量恢复出来在本地复现。两者的取舍是
   “计划性观察”对“被动取现场”。

**常见错误**：在 `import torch` 之后才设置 `ASCEND_DUMP_PATH`，然后怀疑是 TileLang
的 dump 功能坏了。
