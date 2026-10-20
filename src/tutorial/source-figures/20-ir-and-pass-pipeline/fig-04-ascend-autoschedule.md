# 图 4：昇腾自动调度段（`tilelang/ascend/pipeline.py:104-116`）（自绘）

昇腾侧没有 CUDA 那样的通用软件流水，而是把「任务化 → 资源建模 → 求解 → 物化 →
同步」做成一整段，全部受 `allow_autoschedule()` 守卫（默认开启，
配置键 `tl.enable_auto_schedule`）。

```text
                      ┌───────────────────────────────────────────────┐
  NormalizeControlFlow │ 把 while 变成有界的 for(synthetic_while)，     │
  ForSchedule   L104   │ 把复杂 if 条件与 buffer 相关边界提成 Bind     │
                      └───────────────────────────┬───────────────────┘
                                                  ▼
  NormalizeConflictHints L105   T.assume_no_conflict / T.assume_conflict 归一化
                                                  ▼
  MaterializeScheduleUnits L106  把计算/搬运拆成带资源标签的任务单元
                                                  ▼
  AnnotateMultiBufferEligible L107   标注哪些 buffer 可以多份
                                                  ▼
  EstimateLatency L108   给每个任务标注 latency 与 II（启动间隔）
                                                  ▼
  AutoSchedule L109   Z3/SMT 求解：在资源约束下重排任务、决定流水
        │             （Python 侧求解器经 FFI 注册为 tl.transform.z3_schedule_*）
        ▼
  AssignCore L110   给每个任务打上最宽的合法 AIV/AIC 掩码
  PrepareMultiBuffer L111   选多缓冲时钟，插入计数器 init/advance 任务
  ResolveCore L112   按实际的标量/计数器消费者收窄核候选
        ▼
  InsertSync L113   依据读写足迹插入同步；同时产出 buffer 复用契约
        ▼
  MaterializeMultiBuffer L114   把逻辑多缓冲时钟落成物理 buffer 版本
        ▼
  LowerScheduledTIR L115   把调度结果落成最终的 AIV/AIC core 体
  RestoreWhileLoops L116   把 synthetic_while 还原成 while(true)
```

每个 pass 的 C++ 本体位置（锁定版本）：

| pass | 文件:行 |
| --- | --- |
| `AutoSchedule` | `src/ascend/transform/auto_schedule.cc:1371` |
| `InsertSync` | `src/ascend/transform/insert_sync.cc:2551` |
| `MaterializeMultiBuffer` | `src/ascend/transform/materialize_multi_buffer.cc:544` |
| `EstimateLatency` | `src/ascend/transform/estimate_latency.cc` |
| `AssignCore` / `ResolveCore` | `src/ascend/transform/assign_core.cc` / `resolve_core.cc` |
| `PrepareMultiBuffer` | `src/ascend/transform/prepare_multi_buffer.cc` |
| `LowerScheduledTIR` / `RestoreWhileLoops` | `.../lower_scheduled_tir.cc` / `restore_while_loops.cc` |

**为什么这段只能属于昇腾**：CUDA 的软件流水是围绕 `T.Pipelined` 这一个循环构造
展开的（`PipelinePlanning` + `InjectSoftwarePipeline`）；而昇腾的一个 kernel 里
同时有 Cube、Vector、MTE1/2/3、Fixpipe 多条流水，任务之间的依赖与资源占用必须
一起解，才能回答「这一轮 MTE2 搬到哪了 / Cube 什么时候能吃 / 何时能覆盖 UB」。
第 23 章会把 Z3 求解与资源建模展开。
