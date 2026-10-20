# 01-python-pass — 最小 Python pass 与它的单元测试

## 目的

演示本章 3.2 节「写一个 pass 的六步」中最短的一条路径：**不改 C++、不重编
TileLang**，只用 ``prim_func_pass`` 在 Python 里写一个会真正改 IR 的 pass，
并用 IR 前后对照来测它。

这个 pass 的原型是锁定版本里真实存在的
`tilelang/transform/hoist_broadcast_values.py`，我们把它简化、改名成
`HoistConstantBroadcast`，并保留了两个关键设计点：

1. **mutator 的局部状态**（`pending`）必须在进入子语句时清空、离开时恢复；
2. 重写过的节点要重新盖 span，否则报错定位会丢。

## 文件

| 文件 | 作用 |
| --- | --- |
| `hoist_constant_broadcast.py` | pass 本体（mutator + `prim_func_pass` 工厂） |
| `test_hoist_constant_broadcast.py` | 三个 IR 前后对照用例 + 一个反向用例 |

## 运行前提

* **静态检查**：只需要 Python 3.10+，不需要 tilelang/tvm。
* **真正运行测试**：需要一棵装好的 TileLang 源码树（`import tilelang` 可用），
  以及 `pytest`。**不需要 GPU/NPU**——用例止步于 `IRModule`，
  不触发 codegen 与设备编译。

## 运行命令

```bash
# 1) 静态检查：只用标准库（不需要 tilelang/tvm）
python3 -m py_compile hoist_constant_broadcast.py test_hoist_constant_broadcast.py

# 2) 打印本文件依赖的 API 清单（仍然需要 tvm 可 import，因为顶层就 import 了它）
python3 hoist_constant_broadcast.py

# 3) 真正运行测试（需要装好的 TileLang；不需要 GPU）
python3 -m pytest test_hoist_constant_broadcast.py -q
```

## 预期输出

* 第 1 步：`py_compile` 无输出即通过——**这是本机唯一真正验证过的一步**。
* 第 2 步：打印 5 行 API 清单。在没装 tilelang/tvm 的机器上，这一步会以
  `ModuleNotFoundError: No module named 'tvm'` 失败：本文件顶层就
  `from tvm import tirx`，这正是「静态检查」与「运行」的分界线。
* 第 3 步：`3 passed`。若失败，最有价值的两个断点是
  `test_two_stores_do_not_share_hoisted_bindings`（局部状态串味）与
  `test_leaves_a_non_immediate_broadcast_alone`（把变量广播也提了）。

## 在这台机器上达到的验证级别

**仅静态核对。** 作者只有 macOS / Apple Silicon，没有 tilelang/tvm，也没有
GPU/NPU：`py_compile` 通过，且文件里出现的每个 API 名都在
`_vendor/tilelang-v0.1.15/` 里逐个核对过（`tvm.tirx.Broadcast` /
`PyStmtExprMutator` / `tvm.tirx.stmt.Bind` / `tvm.tirx.transform.prim_func_pass` /
`tilelang.ir.stamp_stmt_spans` / `tirx.functor.mutator`）。
**没有运行过测试，也没有 import 过它。** 在有环境的机器上，请把第 2、3 步的真实
输出记下来再引用。

## 挂进 pipeline 的方法

在 `tilelang/transform/__init__.py` 里 import 并转出，然后像其他 pass 一样在
backend pipeline 里调用（两条都是**一行**）：

```python
# tilelang/transform/__init__.py
from .hoist_constant_broadcast import HoistConstantBroadcast  # noqa: F401

# tilelang/cuda/pipeline.py 里，放在 VectorizeLoop 之前
mod = tilelang.transform.HoistConstantBroadcast()(mod)
```

顺序原则见本章 4.3 节：这个 pass 依赖「store 还是 store、Broadcast 还没被拆」，
所以必须早于 `VectorizeLoop` / `FlattenBuffer` / `LowerOpaqueBlock`。
