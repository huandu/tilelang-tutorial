# 示例 5：运行期打印与断言（NVIDIA / CUDA 侧）

## 目的

编译期工具回答“IR 对不对”，运行期工具回答“数值对不对”。本示例把 `T.print` 的四种
用法与 `T.device_assert` 放在一起，并重点标注**单线程门控语义**——这是最容易踩的坑。

门控规则（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/print.py:155-194`）：

| buffer scope | 门控条件 | 含义 |
| --- | --- | --- |
| `local` | `True` | 所有线程都打印 |
| `local.fragment` | `tx == main_lane and ty == 0 and tz == 0` | 先 `copy` 到 shared，再只由一个线程打印 |
| `shared` / `shared.dyn` | 同上 | 只由一个线程打印 |
| `global` | `True` | 所有线程都打印，输出会被重复 N 次 |

`main_lane` 由 `warp_group_id * 128 + warp_id * 32` 算出（`print.py:149-151`）。
打印标量表达式时，`T.print` 不做任何门控，必须自己加 `if tx == 0` 之类的条件。

## 运行前提

* `--static-only`：任意机器。
* 实际运行：NVIDIA GPU + CUDA toolkit + torch(CUDA) + TileLang v0.1.15。

## 运行命令

```bash
python3 debug_print_demo.py --static-only    # 任何机器
python3 debug_print_demo.py                  # 需要 NVIDIA GPU
```

## 期望输出

`--static-only`（写作机实际输出，退出码 0）：

```
py_compile: ok
需要与锁定版本核对的入口（逐个在 _vendor/tilelang-v0.1.15/ 下确认）：
  T.print（CUDA 方言，含门控）                   tilelang/cuda/language/print.py:128-207
  T.device_assert（CUDA 方言）               tilelang/cuda/debug.py:23
  device 侧 printf 模板                     src/tl_templates/cuda/debug.h:11-55
  单线程门控常量（warp_group_size=128）           tilelang/cuda/language/print.py:149-151
（本机没有 GPU：这里只核对语法与入口存在性，未运行任何 kernel。）
```

在有 GPU 的机器上运行时，device 端输出形如（格式来自官方 debug 文档与
`src/tl_templates/cuda/debug.h` 的模板，**具体数值取决于实际调度**）：

```
msg='shared_buf' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): buffer=shared_buf, index=0, dtype=half value=...
msg='block_sum' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): dtype=int value=0
msg='hello from debug_print_demo' BlockIdx=(...), ThreadIdx=(...): dtype=unknown value=...
```

**判据**：kernel 正常结束（没有 device assert 失败），stdout 里能看到上面的消息。

## 代价与限制

* device 侧 `printf` 走受限的片上缓冲，会串行化并且**显著改变耗时**；它只应出现在调试
  版本里，调完必须移除。
* `T.print(global_buf)` 默认每个线程各打印一遍，输出会被放大 `threads` 倍，很容易把
  缓冲区打满。
* `T.device_assert` 的消息会带调用栈（除非 `no_stack_info=True`），非空消息会轻微拖慢
  kernel（源码里有对应告警）。

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过，4 个入口逐个核对；**本机没有 GPU，未运行任何 kernel**。
