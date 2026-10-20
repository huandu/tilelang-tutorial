# 示例 03：手写双缓冲的逐元素 add（A 树）

## 目的

本章 §4.4。展示“多缓冲到底要管什么”：**每加一个缓冲版本，就多一组 set/wait 的配对负担**。
它是对照 M 树 `T.annotate_buffer_versions`（编译器代管）的“手工版基线”。

## 三对 flag 分别保护什么

| flag（src → dst） | 语义 |
| --- | --- |
| `mte2 -> v` | 槽位数据搬完了，Vector 可以开始读 |
| `v -> mte3` | 槽位算完了，MTE3 可以开始写回 |
| `mte3 -> mte2` | 槽位被消费完了，下一轮预取可以复用这块物理缓冲 |

`init_flag()` 预先给两个槽位各发一个 `mte3 -> mte2` token（初始状态“可写”）；
`clear_flag()` 把退出前最后一次归还的 token 消费掉。**少掉任何一边都会造成偶发错误。**

## 运行前提

* 昇腾 NPU + CANN + `torch_npu` + TileLang-Ascend（A 树）；
* `block_M % sub_M == 0`，且 `sub_M % VEC_NUM == 0`（VEC_NUM = 2）；
* M、N 能被 `block-m`、`block-n` 整除。

## 运行命令

```bash
python3 vec_add_pipeline.py --m 1024 --n 1024 --block-m 128 --block-n 128 --sub-m 32
```

## 期望输出

```
init successful!
Kernel Output Match!
```

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`T.alloc_ub`、`T.set_flag`、`T.wait_flag`、
`T.macro`、`T.Parallel`、`T.Scope` 逐个在 A 树中比对过；flag 的配对逻辑对照
A 树 `examples/elementwise/elementwise_add_pipeline.py` 与
`docs/TileLang-Ascend Programming Guide.md:2292-2387`（同步原语）逐条核过。

## 已知局限

* 手工 flag 的**核间**配对（`set_cross_flag`/`wait_cross_flag`）本示例没有涉及，
  它只演示同一核内的 PIPE 同步。
* 手工多缓冲“复杂控制流下可能被分析错”是已知风险（见 M 树
  `references/synchronization.md` 的“Manual multibuffering: limited support”）。
  能用编译器代管就用编译器代管。
