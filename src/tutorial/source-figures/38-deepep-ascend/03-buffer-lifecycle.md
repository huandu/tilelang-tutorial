# 图 3：EPBuffer 的生命周期

**用途**：正文 §2.6。

**事实来源**（静态阅读）：

* `DeepEP-Ascend/README.md:88-116`（每个 EP group 一个长生命周期 buffer、容量一致、用完 destroy）。
* `DeepEP-Ascend/deep_ep/buffers/ep.py:175-249`（构造与 `destroy`）。
* `DeepEP-Ascend/csrc/buffers/base.hpp:17-35`（`explicitly_destroy` 与泄漏警告）。
* `DeepEP-Ascend/csrc/kernels/comm/hccl.hpp:48-98`（对称内存分配与注册）。

```text
 ① 规划
    get_buffer_size_hint(group, num_max_tokens_per_rank, hidden, topk, use_fp8)
      → 返回 2 MB 对齐的字节数；所有 rank 用同一组参数 → 同一容量
          │
          ▼
 ② 构造
    EPBuffer(group, num_max_tokens_per_rank=..., hidden=..., num_topk=...,
             use_fp8_dispatch=..., explicitly_destroy=True)
      → 申请并注册对称内存 [ workspace | buffer ]
      → 建立 HCCL 信道与 Jetty（每 AIV 一条）
      → synchronize + dist.barrier(group)      ← 所有 rank 就绪
          │
          ▼
 ③ 使用（可循环很多次）
    dispatch → 专家 GEMM → combine
      · handle 复用：下次 dispatch 传 handle=... 可跳过布局计算
      · 允许缓存并跨 micro-batch 复用，但**不能改容量/对齐/topk**
      · 所有在途操作（含 .wait()）必须结束，才能改 buffer
          │
          ▼
 ④ 销毁
    buffer.destroy()
      → barrier → 注销 host workspace → HCCL finalize
      → 然后才允许 dist.destroy_process_group()
      · explicitly_destroy=True 且忘记 destroy() → 打印泄漏警告并真的泄漏
```

**要点**

* 一个 EP group **只建一个** buffer，跨层、跨 micro-batch 复用；不要每层建一个。
* 容量由 `num_max_tokens_per_rank` 决定，**所有 rank 必须一致**。
* handle 可以缓存，但前提是 `(num_experts, expert_alignment, num_max_tokens_per_rank)` 与当初一致。
* `destroy()` 必须在拆进程组**之前**调用。
