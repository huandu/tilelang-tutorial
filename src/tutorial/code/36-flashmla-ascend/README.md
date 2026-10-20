# 第 36 章 示例代码

四个示例，全部**不需要 GPU / NPU**，可以在任意装了 Python 3 的机器上运行。
它们覆盖本章的三条线：**算代价**（01、02）、**验证算法**（03）、**划 TileLang 的边界**（04）。

| 目录 | 内容 | 对应正文 | 验证级别 |
| --- | --- | --- | --- |
| `01-head-roofline/` | 复现报告 `:85-111` 的算存比表（`cube_cycles` / `mem_cycles`），推出「head ≥ 64 才是 compute bound」 | §3.1 | **本机运行过**；输出与报告表逐行一致（`--assert-report` 通过） |
| `02-fixpipe-bound/` | 复现 1C1V 的绑定算术：解出 `B_TOPK > 73.14`，复算 MTE2 队列 32 个 token 与 HF32 关闭后的 8192 周期 | §3.2–§3.3 | **本机运行过**；关键数字与报告一致 |
| `03-online-softmax-skip-scale/` | NumPy 版「带 skip-scale 的 online softmax」，与「一次算完的参考实现」对拍；演示阈值 0 / 6 / 极大三档 | §2.2–§2.3 | **本机运行过**；只验证算法等价性 |
| `04-tilelang-expressibility/` | 「TileLang 能表达吗」的 A/B/C 三分类清单，并在锁定源码树里做存在性/缺失性检查 | §6 | **本机运行过**（静态核对工具） |

## 运行环境

* Python 3.8+；
* `01`、`02`、`04` 只用标准库；`03` 需要 **NumPy**（Python 3.11 + NumPy 1.24 上验证过），
  可选 **PyTorch**（只在 `--check-torch` 时使用）。
* **没有昇腾 NPU、没有 NVIDIA GPU**，本章不提供任何需要硬件才能跑的例子；
  与真实 kernel 相关的结论一律是**静态阅读**（见 `../references.md` 与正文 §7）。

## 一句话总览

```bash
python3 01-head-roofline/head_roofline.py --assert-report
python3 02-fixpipe-bound/fixpipe_bound.py --assert-report
python3 03-online-softmax-skip-scale/online_softmax_skip_scale.py --check-torch
python3 04-tilelang-expressibility/expressibility_map.py
```

前三个在任何机器上都能跑；第四个的源码树检查只在本书仓库内（有 `_vendor/`）
才有意义，否则它会打印完整清单并提示跳过检查。
