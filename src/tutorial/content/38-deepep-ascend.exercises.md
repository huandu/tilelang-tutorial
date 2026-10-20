# 第 38 章 习题

难度用 ★（概念/复述）、★★（要读代码或算一遍）、★★★（要综合本章多处结论）
标注。“硬件”一列写明是否需要真正的多卡昇腾环境，**本机没有 NPU**，
标“需要”的题只能在真机上完成，或按题意做纸面推演。

---

### E1（★，无需硬件）通信量从哪来

设一层 MoE：`T = 8192` tokens/rank、`hidden = 7168`、`top-6`、`num_experts = 256`、
`EP = 8`。token 数据用 BF16。

1. 按第 16 章的账本（跨卡时每 token 每方向 `k·d·b` 字节），写出 dispatch
   与 combine 各自的**上界**字节数（假设 k 个专家都落在不同 rank）。
2. 若改用 FP8 dispatch、combine 仍为 BF16，dispatch 的上界变成多少？
3. 用一句话说明：为什么这个数字“不随 GEMM 调优而变小”。

### E2（★，无需硬件）三个参数各自在防什么坑

分别解释 `EPBuffer` 的这三个参数**各自防的是哪一类错误**：

1. `num_max_tokens_per_rank` 为什么要求所有 rank 一致？
2. `expert_alignment` 为什么必须和后面 grouped GEMM 的 M 对齐一致？
3. `do_zero_padding` 在 `do_cpu_sync=False` 时**不能**保证什么？

### E3（★★，无需硬件）读 `get_buffer_size_hint`

读 `csrc/buffers/ep.hpp:581-605` 与 `deep_ep/include/deep_ep/layout/ep/token.hpp:9-83`，
回答：

1. `get_buffer_size_hint` 在 dispatch 布局与 combine 布局之间取什么值？
2. 它用什么常量做最终对齐？这个常量是多少字节？
3. SF 的 pack 数在 hint 里是 `ceil_div(hidden, 32)`，而实际运行时是
   `ceil_div(hidden, 64)`。这会造成什么偏差？会给出偏小还是偏大的容量？

### E4（★★，无需硬件）FP8 dispatch 的 payload

测试里造 FP8 输入的方式是：

```python
x_data, sf = per_token_cast_to_fp8(x)      # sf 形状 (num_tokens, hidden/128)，fp32
sf = sf.view(torch.int16)                  # 形状 (num_tokens, hidden/64)，int16
```

（`DeepEP-Ascend/tests/ep/test_ep.py:89-99`）

1. `hidden = 7168` 时，`num_sf_packs` 是多少？
2. 一个 token 的 SF 占多少字节？
3. `.view(torch.int16)` 在这里是“数值转换”还是“字节重解释”？请给出一条
   能证明你答案的判据（提示：对比字节总数）。

### E5（★★★，无需硬件）为什么 EP 越大越慢

README 的表格里 dispatch 从 EP8 的 373–375 GB/s 降到 EP128 的 313–320 GB/s。
给出**至少三条**理由，并说明每一条能在源码或 README 里指到哪里。

### E6（★★★，无需硬件）为什么 combine 比 dispatch 慢

同一个 EP8：dispatch 373–375 GB/s，combine 345–347 GB/s。给出三条原因，
并说明其中哪一条是**本地**开销、哪一条与 HBM 有关。

### E7（★★，无需硬件）读 dispatch 的三段

读 `deep_ep/include/deep_ep/impls/ep/dispatch.hpp:86-549`、`:327-429`、`:806-865`，
回答：

1. 第一段（SIMT）里，“去重”是针对什么去重？用什么原语做的？
2. 第二段写 SQE 时，一个 token 的**数据**和它的**元数据**分别落在哪里？
   这带来什么好处？
3. 第三段里，落在本 rank 的 token 走什么路径？它占 URMA 带宽吗？

### E8（★★，无需硬件）defer_epilogue 的两个硬约束

1. `defer_epilogue=True` 时必须同时打开哪个参数？在源码哪一行断言？
2. 如果通信之前已经排了别的 kernel，为什么还要用 `buffer.capture()`？
3. 在什么情况下“延迟 epilogue”反而比不延迟更慢？

### E9（★★，无需硬件）核心数与 handle 的关系

1. `num_sms` 在昇腾上到底选的是什么？
2. cached dispatch 与 combine 时，传 `num_sms=8` 但 handle 里存的是 32，
   实际会用几个核？源码依据在哪？
3. `num_sms=0` 在昇腾上会返回什么？

### E10（★★★，无需硬件）画通信栈

不看 §5 的图，自己画一张 HCCL / HCOMM / UBMEM / URMA 的分层图，并回答：

1. “连接建立”走哪一层？“数据搬运”走哪一层？
2. 为什么 `get_physical_domain_size()` 返回 `{1, num_ranks}` 而不是
   `{num_ranks, 1}` 或别的组合？
3. 这个库对“每对 peer 之间必须有 UBMEM 直连”的检查写在哪？断言什么？

### E11（★★★，无需硬件）DeepJIT 与 TileLang JIT 的职责

把第 26 章 TileLang 的 JIT 与本章的 DeepJIT 逐项对照，至少填写下面四格：

| 维度 | TileLang | DeepJIT |
| --- | --- | --- |
| 编译的输入是什么 | ? | ? |
| 缓存键里包含什么 | ? | ? |
| host 侧 ABI 谁生成 | ? | ? |
| 使用者要自己管什么 | ? | ? |

然后用一句话总结：为什么 DeepEP 这类库**不需要** TileLang 那套 host ABI？

### E12（★★，需要真实硬件，或按 §8 做纸面判断）选型题

下面四个需求，各自在 DeepEP-Ascend 上**现在**能不能做？不能的话给出替代方案。

1. 用 Bucket 做 CP 的 all-reduce。
2. 在线检测到热点专家后，动态把权重搬到冗余副本上，并在反向把梯度收回。
3. 把一次次 decode step 的 EP 通信 capture 进图，减少下发开销。
4. 用 7 张卡跑 EP（`num_ranks = 7`）。

### E13（★★★，无需硬件）复算“90–95%”的含义

1. 按测试代码的口径，性能表里的 GB/s 是**哪些字节**除以时间？
   （它包含 padding 吗？包含 barrier 吗？）
2. 把 EP8、EP16、EP32 的 dispatch 数字分别除以 0.90 与 0.95，
   得到三个区间的并集，写出这个反推出来的“物理 payload 带宽上限”量级。
3. 为什么这个反推**不能**当成硬件规格？请说出两条理由。

### E14（★★，无需硬件）TileLang 主线

1. 进 `EPBuffer.dispatch` 的三个张量（`topk_idx`、`topk_weights`、`x`）
   分别由哪个 TileLang 算子产出？给出文件名。
2. 为什么 DeepEP-Ascend 自己不用 TileLang？（提示：看 §4.3 与 §5.2 里它
   到底在操作什么。）
3. MegaMoE 走了另一条路：它把什么融进了一个 kernel？两条路各自的
   适用场景是什么？
