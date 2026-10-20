# 第 23 章 代码

本章的两个示例都是**纯 Python、标准库、可在无 NPU 的机器上运行**的分析工具。
它们不编译 TileLang、不连 NPU，只做两件事：把调度模型缩到能看懂、把 pass 顺序
在锁定源码里核实一遍。

| 目录 | 目的 | 是否需要硬件 | 依赖 | 作者验证级别 |
| --- | --- | --- | --- | --- |
| `toy_scheduler/` | 昇腾软件流水调度的最小模型（II、模序、多缓冲、片上容量） | 否 | 无（装了 `z3-solver` 可加验） | `在 CPU 上验证过逻辑` |
| `pass_order/` | 从锁定版本源码静态抽取并核对 AutoSchedule 的 13 个 pass | 否 | 无 | `在 CPU 上验证过逻辑` |

## 运行方式

```bash
cd toy_scheduler
python3 toy_schedule.py demo          # 最小实例：求 II 与 buffer 份数
python3 toy_schedule.py sensitivity   # 延迟估计不准 → 调度出错
python3 toy_schedule.py --print-smt   # 打印与 z3_scheduler.py 等价的约束
python3 toy_schedule.py --z3          # 本机装了 z3-solver 时复算并比对

cd ../pass_order
python3 ascend_pass_order.py          # 自动找 _vendor/tilelang-v0.1.15
python3 ascend_pass_order.py --repo /path/to/tilelang-v0.1.15
```

## 验证边界（必读）

* 写作环境是 macOS（Apple Silicon），**没有昇腾 NPU**，也**没有安装 `z3-solver`**
  （`python3 -c "import z3"` → `ModuleNotFoundError`，见 `references.md` V8）。
* 因此本目录**没有**任何在 Z3 上求解或在 NPU 上运行的结果。`toy_schedule.py` 用
  自己写的回溯搜索复现了 `z3_scheduler.py` 的**约束语义**，这不等于验证了 Z3 的行为。
* `pass_order/ascend_pass_order.py` 只做静态检查：读源码、比对顺序、确认
  `refl::GlobalDef().def("tl.transform.<Pass>")` 存在。**不代表这些 pass 能跑通**。
* 两个脚本都不写文件、不访问网络、不读 `_vendor/` 和自身目录以外的路径。
* `examples/ascend/example_gemm.py` 里的真实流程（`T.Persistent` + `T.Pipelined` +
  `do_bench(backend="msprof")`）**没有被本目录执行过**；它出现在正文里只作为
  “在有 NPU 的机器上怎么验证”的入口，验证步骤见 `chapter.md` 第 5.5 节。
