# 图 5：带宽随 EP 规模变化

**用途**：正文 §7。

**数据来源**：`DeepEP-Ascend/README.md:13-19` 一张表，**原文数字照抄**。

**测量条件**：Ascend 950DT NPU + CANN 9.2.0 + 手工配置的 PoC HDK；
netlayer 1（超节点 external Clos 网络）；16,384 tokens/rank、hidden 7168、
top-6 / 256 experts、32 AI cores / 64 AIVs、expert alignment 128、无 bias；
dispatch 为 FP8 + row-major scale，combine 为 BF16；10 warmup + 50 samples，
刷 cache；带宽范围覆盖所有 rank。

```text
| EP size | Dispatch（FP8, GB/s） | Combine（BF16, GB/s） |
| ------- | --------------------- | --------------------- |
| EP8     | 373–375               | 345–347               |
| EP16    | 348–352               | 338–341               |
| EP32    | 335–340               | 320–324               |
| EP64    | 323–327               | 294–298               |
| EP128   | 313–320               | 272–278               |

趋势（取区间中点）：
  dispatch ：375 → 320 GB/s，EP8 → EP128 下降约 15%
  combine  ：346 → 275 GB/s，每个 EP 档位都低于同档 dispatch
  combine / dispatch 的比值：约 0.87 – 0.97
```

**要点**

* 两条曲线都随 EP 增大而下降：EP 越大，一个 token 越可能被拆到更多 rank，
  跨设备字节量与固定开销占比都上升。
* combine 在所有 EP 规模上都低于 dispatch：payload 是 BF16（dispatch 是 FP8）、
  多一次本地归约、且与 HBM 抢带宽。
* README 的结论：EP≤32 时可持续 dispatch 达到物理 payload 带宽上限的 90–95%；
  更大的 EP 与 combine 仍在优化中。
* EP128 对应官方文章里的 128 卡超节点方向（`deepseek-article-20260930.md:13`）。
