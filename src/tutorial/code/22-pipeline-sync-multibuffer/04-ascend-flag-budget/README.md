# 04 · 昇腾 flag 命名空间预算

## 目的

昇腾侧的多缓冲同步最后会落到 `SetFlag/WaitFlag` 上，而每一对
`hard_event`（producer PIPE → consumer PIPE）**只有 8 个 event-id 槽**
（`_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:2135`，
`kIntraCoreFlagLimit = 8`；跨核是 16，`:2136`）。一个 sync point 的 event id 是
`iteration % num_versions + base_id`（`MakeFlagEventId`，`:2299`），所以一个
`num_versions = v` 的同步点要吃 **v 个连续槽**，运行时无法跨越 8 的边界。

槽不够时，`RewriteFlagToBuf`
（`_vendor/tilelang-v0.1.15/src/ascend/transform/rewrite_flag_to_buf.cc:201-336`）
用 0/1 背包（容量 8）挑一组最能填满的保留为 `set_flag/wait_flag`，其余的
**溢出**到共享的 `asc_lock/asc_unlock` 互斥池——溢出不是消失，而是变成阻塞的
锁操作。

这个模型把上面这套算术摊开，回答「`num_stages` 加到多大时，瓶颈从 shared memory
变成 flag 槽位」。

## 运行前提

Python 3.9+，只用标准库。不需要 NPU。

## 运行命令

```bash
python3 flag_budget.py                # 逐 depth 扫，intra-core 8 槽
python3 flag_budget.py --sharing      # 允许可复用的同步点合并 base
python3 flag_budget.py --cross-core   # 跨核 16 槽命名空间
python3 flag_budget.py --depth 3      # 只看一档
```

## 期望输出（本机实际运行结果，节选）

```text
depth=2  num_versions per sync point = 2
  blocks: acquire_A=2, acquire_B=2, release_A=2, release_B=2
  allocator sum = 8 ids
  -> fits in 8 slots; no spill  (0 slots spare)
     acquire_A base  0  id = i % 2 + 0
     acquire_B base  2  id = i % 2 + 2
     release_A base  4  id = i % 2 + 4
     release_B base  6  id = i % 2 + 6

depth=3  num_versions per sync point = 3
  blocks: acquire_A=3, acquire_B=3, release_A=3, release_B=3
  allocator sum = 12 ids
  -> exceeds 8 slots: 0/1 knapsack keeps 6/8 slots as set/wait flags
     kept    : acquire_A, acquire_B
     spilled : release_A, release_B  -> asc_lock/asc_unlock mutex pool (blocking)
```

## 要读出什么

1. **`num_versions` 乘在每个同步点上。** 同一个逻辑依赖，深度 1 时只占 1 个
   flag id，深度 4 时占 4 个。
2. **8 个槽是很小的预算。** 两条 copy、两条 release，深度 3 就已经溢出；
   真实 kernel 的 sync point 远不止四个。
3. **溢出会改变性能口径。** 被背包丢掉的同步点从「事件标志」变成「互斥锁」，
   这是阻塞语义——正是 §6.2 里「多一个同步掉性能」的另一条路径。
4. **`--sharing` 是编译器真实在做的事。** `FlagAllocator` 的 `reusable_pairs_`
   （`insert_sync.cc:1729`、`:1742`）会把不可能同时在飞的同步点合并到同一个
   base，:depth 2 时 8 个 id 降到 6 个。
5. **跨核有 16 个槽**（`--cross-core`），所以「什么时候需要跨核同步」也在
   预算之内。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过。**没有**在任何
昇腾 NPU 上对照过。

## 这个模型不做什么

* 「可复用」的判据（`reusable_pairs_`）在真 pass 里由依赖分析给出，这里用一个
  `--sharing` 开关代表，是**上界**而不是等价实现；
* 它不建模跨核的 16 槽与 8 槽之间的关系，也不建模 `CrossCoreFlagReservation`
  对用户显式 flag 的预留；
* 它不给出任何性能数字，只说「保持在事件标志里」还是「溢出到锁」。
