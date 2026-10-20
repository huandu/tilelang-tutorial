# 示例 6：昇腾 attention 的静态地图

## 目的

正文第 4.5、4.7、5.6 节讨论昇腾侧的 attention 时，依赖四类事实。这个示例把四类事实
全部从**锁定版本的文件里读出来打印**，不靠记忆：

* **A. 搬运关系表**：直接从锁定版《TileLang-Ascend Programming Guide》里解析
  `GM/L1/L0A/L0B/L0C/UB` 的合法搬运对，用来证明“**没有 `L0C → UB` 这一行**”，
  因此 Cube 算出的分数必须绕一趟全局内存；
* **B. API 行号**：适配器仓库（`v0.1.2.000`）与主干仓库（`v0.1.15`）两棵树里的昇腾
  专属 API（`alloc_L1/alloc_L0C/alloc_ub`、`set_cross_flag/wait_cross_flag`、
  `gemm_v0`、`T.tile.topk/gather_mask/cast/broadcast`、`T.Scope`、`dual_copy`、
  `SimdVF`、`Stage`、`annotate_buffer_versions`）；
* **C. 五个昇腾 attention 例子的结构**：kernel 启动、Cube/Vector 两个 scope、
  跨核 flag、softmax 递推、paged 的 `block_table`、sink、varlen、lightning indexer；
* **D. 瓶颈结论**：从调优文档里定位并打印那几行（Double Buffer、最佳 `num_stages`、
  减少核间同步、`T.pipelined` 不支持嵌套）。

## 运行前提

* Python 3.9+（只用标准库）；
* 书稿工作树上有 `_vendor/tilelang-ascend-v0.1.2.000/` 与 `_vendor/tilelang-v0.1.15/`。

**不需要 CANN、不需要 NPU、不 import tilelang。**

## 运行命令

```bash
python3 ascend_attention_map.py
python3 ascend_attention_map.py --strict   # 有漂移则退出码 1
```

## 期望输出（写作机实际输出，退出码 0）

节选：

```text
A. the copy matrix from the locked programming guide
  L  629  | GM   | L1   | 数据从Global Memory搬运到L1 Buffer        |
  L  630  | L1   | L0A  | 数据从L1 Buffer搬运到L0A Buffer，Cube计算的左矩阵 |
  L  631  | L1   | L0B  | 数据从L1 Buffer搬运到L0B Buffer，Cube计算的右矩阵 |
  L  632  | L0C  | GM   | 数据从L0C Buffer搬运到Global Memory       |
  L  633  | GM   | UB   | 数据从Global Memory搬运到Unified Buffer   |
  L  634  | UB   | GM   | 数据从Unified Buffer搬运到Global Memory   |
  L  635  | UB   | UB   | 数据从Unified Buffer搬运到Unified Buffer  |
  L  636  | UB   | L1   | 数据从Unified Buffer搬运到L1 Buffer       |

  key point: there is no L0C -> UB row. A Cube-produced score tile
  reaches the Vector core only by going through GM (the workspace).

B1. adapter repo (TileLang-ascend v0.1.2.000)
  alloc_L1                 tilelang/language/allocate.py:140
  set_cross_flag           tilelang/language/ascend.py:191
  gemm_v0                  tilelang/language/ascend.py:418
  topk                     tilelang/language/ascend_tile.py:550

B2. mainline repo Ascend dialect (TileLang v0.1.15)
  alloc_l1                 tilelang/ascend/language/allocate.py:26
  dual_copy                tilelang/ascend/language/copy_op.py:53
  SimdVF                   tilelang/ascend/language/frame.py:179
  Stage                    tilelang/ascend/language/schedule_hint.py:118

C. Ascend attention examples in the locked adapter tree
examples/flash_attention/flash_attn_bhsd.py  (260 lines)
    cube scope               :86
    vector scope             :116
    cube->vector flag        :98
    QK^T on cube             :93
    running max              :142

D. the bottleneck notes, quoted from the locked tuning document
  L  52  ## 二、性能优化总体思路
  L  84  ### 3.1 Double Buffer 原理
  L 191  ### 4.1 找到最佳 num_stages
  L 212  ### 4.2 核间同步优化
  L 255  **重要**：`T.pipelined` 不支持嵌套使用。

drift/miss count: 0
```

## 判据

* A 段必须打印出 8 行搬运对，且其中**没有** `L0C → UB`；
* B 段所有 API 都能找到，且行号与正文一致；
* C 段的每一个探针都命中；
* D 段五个标题/句子都能定位；
* `--strict` 退出码 0。

## 注意事项

* 与示例 5 一样，这里只做**文本匹配**。`examples/flash_attention/flash_attn_bhsd.py`
  这个文件的 workspace 写法与主干仓库的自动 workspace 消除是**两代做法**，正文
  第 4.5 节把两者都提到了，脚本分别指向它们。
* A 段里“没有 `L0C → UB`”是**这张表**的结论。主干仓库的编程指南 4.1.5.2 节说明了
  编译器会把这种跨核搬运自动改写为 `L0C → GM → UB` 两步，两者并不矛盾：
  **一条 `T.copy` 在前端可以写，在硬件上要走两跳。**
* 脚本打印的“结构性结论”（分数必须绕 workspace）来自这张表与例子的代码，属于
  **静态阅读**；真实带宽/延迟表现需要 NPU 上的 msProf 数据，本章没有。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 `--strict`，
输出 `drift/miss count: 0`，退出码 0。
**没有 CANN、没有 NPU，没有编译或运行过任何昇腾 kernel。**
