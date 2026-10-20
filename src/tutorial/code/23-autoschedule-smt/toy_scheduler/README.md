# toy_scheduler：昇腾软件流水的最小调度模型

## 目的

把 `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py` 里
`z3_schedule_loop_python`（第 268–783 行）的模型缩到 4 个任务、2 个 buffer 变量，
用来手推“为什么多开一份 buffer 能把 II 压下来”。

模型要素与源码一一对应：

| 玩具模型 | 源码对应 |
| --- | --- |
| `Task(name, latency, ii, pipe)` | `latency` / `ii` / `resource_flags` 三个数组（`z3_scheduler.py:67-107`） |
| `MTE1/MTE2/.../SCALAR` 位掩码 | `ResourcePipe` 枚举（`src/ascend/transform/ascend_pipe.h:52-62`） |
| `data_deps (u, v, distance, latency)` | `start_v - start_u >= latency - II * distance`（`z3_scheduler.py:442-455`） |
| `distance < 0` = 自动多缓冲 | `buffer_var_id = -distance - 1`（`z3_scheduler.py:449-455`） |
| `resource_deps (i, j)` | 模序二选一 + 绕回保护（`z3_scheduler.py:456-470`） |
| `memory_groups` 容量约束 | `Sum(buf[i] * size[i]) <= cap`（`z3_scheduler.py:431-436`） |
| 目标 = 最小 II | 对 II 二分（`z3_scheduler.py:398-540`） |

**唯一被故意去掉的是 owner-exclusion**（`z3_scheduler.py:471-491`）：两个不相关的
owner 共用同一段物理 ring，需要同时保护正向与绕回两个交接。玩具模型不含这个约束，
所以在正文里只用来解释概念，不能当作生产行为。

## 运行前提

* Python 3.9+，**无第三方依赖**。
* 不需要 GPU / NPU / TileLang 安装。
* 可选：`pip install z3-solver` 后可跑 `--z3` 分支（本机未装，未验证）。

## 运行命令与预期输出

```bash
python3 toy_schedule.py demo
```

预期（本机实测输出）：

```text
  first feasible II = 28, buffer versions = [2]
  II = 28（相邻两轮迭代的最小启动间隔）
  buffer versions = [2]  (容量占用 2000 / 4096)
  start(k, r) 按开始时间排序：
    [  0] loadA(MTE2 GM->L1)         k=0 r= 0 latency=40 ii=10 pipe=MTE2
    [ 40] l1->l0(MTE1)               k=1 r=12 latency=15 ii= 3 pipe=MTE1
    [ 55] cube(Cube)                 k=1 r=27 latency=30 ii= 8 pipe=Cube
    [ 85] l0c->gm(Fixpipe)           k=3 r= 1 latency=25 ii= 6 pipe=Fixpipe
  makespan（单轮）= 110 cycles
```

判据：`II=28` 且 `buffer versions=[2]`。手算校验：跨迭代 WAR 要求
`start_MTE1 <= II * versions - 15`，而真依赖要求 `start_MTE1 >= 40`；
`versions=1` 时需 `II >= 55`，`versions=2` 时 `II >= 28`，与输出一致。

```bash
python3 toy_schedule.py sensitivity
```

预期：真值模型给出 `II=28`；把 MTE2 的 latency 从 40 低报成 16 后，调度器给出
`II=16`，并在真值模型下报出“l1->l0(MTE1) 比 loadA(MTE2 GM->L1) 早了 24 cycles”。
这就是正文第 5.1 节说的“模型不准会导致错误”的可复现证据（在小模型上）。

## 验证级别

`在 CPU 上验证过逻辑`——脚本在本机（macOS / Python 3.11.7）实跑通过，
`python3 -m py_compile toy_schedule.py` 通过。**未在 NPU 上验证，也未与 Z3 比对。**
