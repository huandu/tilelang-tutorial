# 图 6：能力状态清单

**用途**：正文 §8。

**事实来源**：`DeepEP-Ascend/README.md:60-84`（Features 表、Ongoing 清单、
“Hybrid communication, CPU-backed Engram storage and graph capture remain
unsupported”），以及源码里的断言（`csrc/buffers/ep.hpp:79,85,116`、
`csrc/kernels/comm/hccl.hpp:85`）。

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| EP dispatch（BF16 / FP8，展开布局） | 可用 | `README.md:66,72` |
| EP combine（BF16，FP32 累加） | 可用 | `README.md:66` |
| cached handle / deterministic layout / expert padding | 可用 | `README.md:66` |
| 延迟 epilogue（defer_epilogue） | 可用（需 async） | `csrc/buffers/ep.hpp:398` |
| Bucket batched all-gather | 可用（experimental） | `README.md:69,80` |
| PPBuffer（相邻 rank send/recv） | 实验性 | `README.md:67,82` |
| EngramBuffer（NPU 侧表 + fetch hook） | 实验性 | `README.md:68,82` |
| Bucket reduce-scatter / all-reduce | 未实现 | `README.md:80` |
| lb_prefetch_weights / lb_reduce_grads | 未实现（接口已对齐） | `README.md:81` |
| 混合通信（hybrid） | 不支持 | `README.md:84` |
| CPU-backed Engram 存储 | 不支持 | `README.md:84` |
| graph capture（EP 入图） | 不支持 | `README.md:84` |
| 显式 RDMA 服务等级 / QP 数 | 不支持 | `README.md:74` |
| 奇数 rank 数 | 不支持（须偶数） | `README.md:29` |

**要点**

* 只有 EP 主线（dispatch/combine + 三个可选特性）是“可用”。
* Bucket 只做到 all-gather；reduce-scatter / all-reduce 仍在开发。
* 动态冗余专家（EPLB）目前只有规划，没有搬运。
* hybrid、CPU Engram、graph capture 三条在 NVIDIA 侧有、昇腾侧没有。
