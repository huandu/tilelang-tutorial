# 第 23 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不使用任何含糊的版本说法；凡涉及版本处一律写具体 commit 或 tag。

写作环境：macOS (Apple Silicon)、Python 3.11.7；**没有 NVIDIA GPU，没有昇腾 NPU，
没有安装 `z3-solver`**。因此本章没有任何本机实测的性能数字。

---

## 锁定的源码引用（`_vendor/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:22-33` | 默认开启 AutoSchedule（`TL_ENABLE_AUTO_SCHEDULE`，默认 `True`），关掉只是“不调度” | 已核验（2026-10-01）：逐行阅读 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:103-116` | 自动调度 13 个 pass 的顺序 | 已核验（2026-10-01）：逐行阅读，并由 `code/pass_order/` 脚本复核 |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/__init__.py:31,47,52,77,82,87,92,97,102,210` | 各 pass 的 Python 门面与 docstring；`AnnotateMultiBufferEligible` 对前序 pass 的依赖；`RewriteFlagToBuf` 的 8 槽 flag / 32 槽池 | 已核验（2026-10-01）：逐行阅读 |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py:67-226, 268-783` | Z3 调度的两个入口：直线调度（min makespan）与循环调度（min II + buffer 份数） | 已核验（2026-10-01）：逐行阅读；**未运行** |
| S5 | 同上 `:17, 132, 391-420, 431-436, 442-455, 456-468, 472-499, 503-514, 518-550, 565-575, 849` | 全局锁、私有 `z3.Context`、变量、容量约束、数据/资源/owner/stage/offset 约束、二分 II、手动不可行报错、静默退化分支、seed | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/src/ascend/transform/ascend_pipe.h:52-62` | `ResourcePipe` 七位掩码与 MTE1/2/3、Cube、Vector、Fixpipe、Scalar 的语义 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/task_annotations.h:41-42, 89-153` | `latency` / `ii` 字段与 `latency >= ii` 校验 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/dependency_analysis.h:38-76` | `kData` / `kOwnerExclusion` 两类依赖与 `distance` 三档语义 | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/ir_structure.h:324-333, 494, 519, 533-536` | `TaskNode`：stage / latency / II / pipe mask / 读写区域 | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.h:28-160` | 延迟与 II 估算的默认硬件参数（含 `cube_throughput_fp16{8192}`、`mte2_gm_to_l1_base_latency{190}`、`valu_bandwidth{256}` 等） | 已核验（2026-10-01）：**这些是模型参数，不是实测值** |
| S11 | `_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.cc:330-345, 941-991` | 延迟/II 算法（访存 + 计算、按流水取 max） | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule.cc:245-258, 452-676, 680-745, 846-857, 1110-1144, 1272-1295, 1318-1347, 1371-1400` | `MergeSelectedBufferVersion` 与 accepted limitation、直线/循环调度入口、FFI 取函数、容量扣减、stage 推导、overall latency、`ScheduleSingleKernel` 的 `SetMemoryLimit` 与容量预检、pass 创建 | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/src/ascend/transform/materialize_schedule_units.cc:18-22, 578` | 任务边界规范化与“物化平凡调度单元” | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/src/ascend/transform/annotate_multi_buffer_eligible.cc:19-38, 835` | 多缓冲资格推断（owner 推断、显式 claim 优先级） | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:751-770, 2270-2297, 2363-2422, 2568` | `GetHardEvent` 归一化、Set/Wait 与跨核 Set/Wait 的 IR 构造、同步插入 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/src/ascend/transform/rewrite_flag_to_buf.cc:3-14, 201-210, 293-294` | 8 槽 flag 命名空间、0/1 背包（容量 8）、溢出到 32 槽 `get_buf`/`rls_buf` 池 | 已核验（2026-10-01）：源码描述；**未在硬件上验证槽位数量** |
| S17 | `_vendor/tilelang-v0.1.15/src/ascend/transform/prepare_multi_buffer.cc:19-22, 1249`、`assign_core.cc:401`、`resolve_core.cc:19-22, 559`、`materialize_multi_buffer.cc:19-22, 561`、`lower_scheduled_tir.cc:744`、`restore_while_loops.cc:117` | 解落地的各步（核分配、多缓冲时钟、版本物化、body 生成） | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/src/ascend/transform/attr.h:18-39` | `tl.ascend_per_core_task` / `tl.ascend_task` / `tl.ascend_stage` / `tl.schedule_unit` 四个属性 | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/schedule_hint.py:13, 65, 118, 196, 246` | `T.PerCoreTask` / `T.Task` / `T.Stage` / `T.assume_no_conflict` / `T.assume_conflict` | 已核验（2026-10-01） |
| S20 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19, 41-46, 107, 165` | `T.annotate_buffer_versions`（含“启发式漏读时用 `{buf: 1}`”）/ `annotate_manual_multi_buffer` / `annotate_unlimit_memory` | 已核验（2026-10-01） |
| S21 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/sync.py`（`ascend_set_flag` / `ascend_wait_flag` 的 docstring） | 生成 `AscendC::SetFlag<AscendC::HardEvent::XXX>(id)` 的语义 | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:141-152` | 与手写 Ascend C 的分工对照表，含 `Synchronization: None (Just write operations in order!)` | 已核验（2026-10-01） |
| S23 | `_vendor/tilelang-v0.1.15/examples/ascend/example_gemm.py:60-80, 98` | `T.Persistent` + `T.Pipelined(num_stages=NUM_STAGES)` 的 GEMM；`do_bench(..., backend="msprof")` | 已核验（2026-10-01）：文件存在，API 逐个比对；**未运行** |
| S24 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:129-130, 251, 258-259` | CUDA 侧对应位置是 `PipelinePlanning` / `InjectSoftwarePipeline` / `MergeSharedMemoryAllocations` / `ThreadSync` | 已核验（2026-10-01） |
| S25 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1495-1550` | CUDA 多缓冲版本数 = `use - def + 1`（结构化上界），含 `num_versions == 2` 特例 | 已核验（2026-10-01） |
| S26 | `_vendor/tilelang-v0.1.15/src/cuda/transform/auto_schedule/role_based_scheduler.h:17-22` | CUDA 的“自动调度”是 Fixed-role heuristic，不是求解器 | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:76, 338-342` | `TL_ENABLE_AUTO_SCHEDULE = "tl.enable_auto_schedule"`、`TL_ENABLE_DUMP_IR` / `TL_DUMP_IR_DIR` | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:194-232, 1094-1205, 2527-2560` | A5 Camodel 仿真（含指令/UB dump）、`T.Pipelined` 的 intra-core / inter-core 用法、msProf 调优 | 已核验（2026-10-01）：本地文档快照；**未运行** |

---

## corpus 资料引用

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| C1 | `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md` | 第 1 节引用的“脑内时序图”原话、模型五要素的自述、`latency` vs `issue interval`、“Z3 不懂 Cube” | **community 级**（来源 id `zhihu-dazhiruoyu-deepseek-ascend`，`trust: community`）；本地全文已核验（2026-10-01），性能类说法一律视为未确认 |
| C2 | `../tilelang-corpus/90-synthesis/03-ascend-route.md` | 昇腾路线综述（任务化/资源建模/Z3 求解/落地四段）、机器模型对照表 | corpus 内合成文档，已核验（2026-10-01）；其版本提示要求以 `_vendor/` 为准 |
| C3 | `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md` 第 3.2 节 | `PipelinePlanning` + `InjectSoftwarePipeline` 的作用与位置 | 已核验（2026-10-01）；同样以 `_vendor/` 为准 |

---

## 上游链接

| id | URL | 用途 | 核验状态 |
| --- | --- | --- | --- |
| U1 | <https://pypi.org/project/z3-solver/> | 本章玩具模型可选依赖 `z3-solver` 的安装来源 | 已核验可达（2026-10-01，HTTP 200） |
| U2 | <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html> | msProf 算子调优官方文档（第 5.5 节实验引用） | 已核验可达（2026-10-01，HTTP 200） |
| U3 | <https://github.com/Z3Prover/z3> | Z3 上游仓库 | **未核验**：本环境访问 GitHub 超时（curl 返回 `000`） |
| U4 | <https://github.com/tile-ai/tilelang> | TileLang 上游仓库 | **未核验**：同上 |
| U5 | <https://www.zhihu.com/question/2088578683623814216/answer/2088624284100999121> | C1 的原始出处 | **未核验**：curl 返回 HTTP 403（反爬）；本章以 corpus 内全文快照为准 |

---

## 环境核验记录（本章实际执行的命令）

| id | 命令 | 结果 | 含义 |
| --- | --- | --- | --- |
| V1 | `ls _vendor/tilelang-v0.1.15/tilelang/ascend/transform/` | 有 `__init__.py` 与 `z3_scheduler.py` | 求解器在主干仓库（不是适配器仓库） |
| V2 | `ls _vendor/tilelang-v0.1.15/src/ascend/transform/` | 34 个 `.cc` + `auto_schedule/` 子目录 | 13 个 pass 的实现都在主干 |
| V3 | `python3 code/pass_order/ascend_pass_order.py` | 13 个 pass，顺序一致，全部找到 FFI 注册点，退出码 0 | 第 4.1 节表格的来源 |
| V4 | `python3 code/toy_scheduler/toy_schedule.py demo` | `II = 28`、`buffer versions = [2]` | 第 4.3 节的玩具实例结果 |
| V5 | `python3 code/toy_scheduler/toy_schedule.py sensitivity` | 低估 MTE2 延迟（40→16）后调度器给出 `II = 16`，在真值模型下报出 24 cycle 违约 | 第 5.1 节的证据（玩具规模） |
| V6 | 把玩具实例 `memory_groups` 改成 `[[1500, 0]]` 后求解 | `II = 55`、`buffer versions = [1]` | 习题 E10 的答案 |
| V7 | `grep -rl "AutoSchedule" _vendor/tilelang-ascend-v0.1.2.000/src _vendor/tilelang-ascend-v0.1.2.000/tilelang` | 无输出 | 适配器仓库 v0.1.2.000 里没有 AutoSchedule |
| V8 | `python3 -c "import z3"` | `ModuleNotFoundError: No module named 'z3'` | 本机没有 z3-solver；`--z3` 分支未验证 |
| V9 | `grep -rn "ascend_auto_cv_combine" _vendor/tilelang-v0.1.15/` | 无输出；该键只出现在适配器仓库文档与 skill 说明里 | 第 5.4 节标“待核”的依据 |

---

## 待核事项（需要后续确认，本章不自行补全）

1. **`tl.ascend_auto_cv_combine` / `tl.ascend_auto_cross_core_sync` 的状态。** 适配器文档
   要求使用核间流水线时必须开启，但 v0.1.15 的 `pass_config.py` 里没有这两个键。需要确认：
   它们是**被主干的新机制取代了**，还是**尚未合入**。
2. **`RewriteFlagToBuf` 的 8 槽 / 32 槽是否对应真实硬件限制。** 源码如此描述，但没有
   厂商文档佐证，也未在硬件上验证。
3. **`EstimateLatency` 默认参数的目标芯片。** `estimate_latency.h` 的注释只说明
   “convention”与单位，没有写这些数字对应哪一代昇腾（`cube_throughput_fp16{8192}`
   是否只对 950 成立）。需要确认标定来源与适用范围。
4. **知乎原文的性能对比结论**（“与主线 Ascend C 打得有来有回”）属 community 级，
   需要独立复测；本章只作为社区说法引用，未当作结论。

## 待补资料（建议后续版本处理）

* corpus 内**没有** Z3 自身的资料（论文或手册）。本章只在概念层面提到 SMT，
  未引用 Z3 的算法细节；如果后续要讲“为什么 Z3 比启发式更合适”，需要补一条
  Z3 的一手来源。
* corpus 内**没有**昇腾官方关于 `SetFlag/WaitFlag` 槽位数量的硬件手册条目。
