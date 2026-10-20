# 第 38 章 代码

本章的四个示例都只依赖 Python 标准库（**没有任何一个需要 GPU / NPU / TileLang**）。
它们验证的是**账本、容量公式和口径**，不是 kernel 性能——本机也没有任何加速器
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。

```
code/
├─ README.md                              本文件
├─ 01-ep-minimal/                         README 里那段 EP 用法的最小复刻（静态阅读 + 不变式检查）
├─ 02-comm-ledger/                        一层 MoE 的 dispatch / combine 字节账
├─ 03-buffer-size/                        复算 TokenLayout / BufferLayout 与 get_buffer_size_hint
└─ 04-bandwidth-normalize/                把 README 性能表换算成「相对物理上限」
```

每个子目录都能独立复制出去运行，互不依赖，也不依赖其它章节的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-ep-minimal/ep_minimal.py` | Python 3.9+；`torch` / `torch_npu` 可选 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过；无 NPU 时走静态讲解分支 |
| `02-comm-ledger/comm_ledger.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过 |
| `03-buffer-size/buffer_size.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过 |
| `04-bandwidth-normalize/bandwidth_normalize.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过 |

**没有任何一个示例在 GPU / NPU 上运行过**，也没有任何一条来自本机的性能数字。
性能口径与来源见 [`../references.md`](../references.md)。

## 这些脚本在证明什么

| 脚本 | 回答的问题 | 结论 |
| --- | --- | --- |
| `01` | README 那段用法在参数层面是否自洽？哪些参数必须保持默认？ | 展开布局、偶数 rank、`topk ≤ 32`、`do_handle_copy=False` 等约束都能在源码里指到 |
| `02` | 一层 MoE 到底要搬多少字节？EP 越大越多吗？ | 是的；EP 越大，token 被拆到越多 rank，跨设备字节越多 |
| `03` | `get_buffer_size_hint()` 到底怎么算？和实际运行时的 SF 数量差多少？ | 与 `calculate_buffer_size` 公式一致；hint 为 SF 预留的量是实际的 2 倍 |
| `04` | “物理 payload 带宽上限的 90–95%”反推出来是多少？ | 量级 350–420 GB/s/rank，且明确标注为区间估计 |
