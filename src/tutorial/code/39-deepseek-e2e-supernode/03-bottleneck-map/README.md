# 03-bottleneck-map：「章节 → 组件 → 典型瓶颈」映射

## 目的

把第 39 章 §5 的检查清单变成一张可执行、可核验的映射表：
**每一行都指名一章**，说明这个组件在第五部分的哪一章讲过、它的典型瓶颈是什么、
以及该用什么去查。

`--verify` 会检查表里引用的章节目录是否真的存在——这样这张表不会悄悄过期。

## 运行前提

* Python 3.9 及以上；
* **只使用 Python 标准库**；
* `--verify` 需要本书的 `chapters/` 目录（默认相对路径
  `../../../..` = `.workspace/research/tilelang-book/`）。

## 运行命令

```bash
python3 bottleneck_map.py                                  # 打印映射表
python3 bottleneck_map.py --verify                         # 再核验章节目录
python3 bottleneck_map.py --verify --book-dir /path/to/tilelang-book
```

## 预期输出

（2026-10-01 在写作机上运行，节选）

```markdown
| 章 | 组件 | 典型瓶颈 | 怎么查 |
| --- | --- | --- | --- |
| 第 13 章 | 逐元素 / 归一化 / RoPE | 启动开销 + 重复读小张量 | 核数、kernel 数量；第 28 章 M6 |
| 第 15 章 | Attention 家族 | online softmax 的中间量带宽 | L0C / UB 容量与 FixPipe；第 36 章 §3.2 |
| 第 16 章 | MoE 与 MLA | 两段零 FLOP 的重排与通信 | 第 16 章 §7 的 regime 表 |
| 第 34 章 | Engram / mHC（不是矩阵乘） | kernel 数量 + 迭代的编译期常量 | 数 launch 次数；第 34 章 §7.1 |
| 第 35 章 | DeepGEMM-Ascend（dense / grouped / MQA / MegaMoE） | compute / memory / FixPipe 三种都要判 | 三个比值；第 35 章 §5.1 |
| 第 37 章 | DeepSelect TopK | 有效内存带宽（不是 FLOPs） | 输入+输出字节 ÷ 时间；第 37 章 §6.1 |
| 第 38 章 | DeepEP-Ascend（EP dispatch / combine） | 卡间带宽 + 固定 launch 开销 | 通算流水图的 Wait Time Ratio；第 38 章 §7 |
```

`--verify` 的输出：

```text
核验章节目录是否存在（chapters/ 的上一级 = .../tilelang-book）
  ok   ../../../../chapters/13-operators-elementwise-norm-rope
  ...
  ok   ../../../../chapters/38-deepep-ascend

全部命中：这张映射表指向的章节都存在。
```

## 验证级别

**CPU 逻辑验证**：在写作环境（macOS Apple Silicon、Python 3.11）上实际运行过，
`--verify` 对本书 `chapters/` 下 10 个章节全部命中。

## 用途与边界

* 这张表回答「**该去读哪一章**」，不回答「你的 kernel 为什么慢」——
  后者要用正文 §5 的六步顺序，配合真机 profiler。
* 表里没有性能数字，因此不存在口径问题；唯一的机械核验就是 `--verify`。
* 如果你在写自己的项目文档，可以把这个脚本当模板：**把每条结论指回它的出处**，
  再用一个开关去核验出处还在不在。
