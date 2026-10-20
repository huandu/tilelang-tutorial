# 示例 04：“TileLang 能表达吗”的三分类清单

## 目的

把正文 §6 的核心结论做成一台**可查询、可自检**的工具：给本章每个技术点标一个类别

* **A 直接可表达**：tile + 循环 + `T.gemm`/`T.copy`/逐元素运算；
* **B 可表达但要下钻**：`T.SimdVF` + `T.simd.*` + 注解/同步原语；
* **C 必须内建**：结论绑在 CCE/Ascend C 指令的参数或协议上。

并且做两类**静态**检查（不需要硬件）：

1. B 类提到的 `T.*` 原语，逐个在锁定源码树
   `_vendor/tilelang-v0.1.15/` 里 `grep` 确认存在；
2. C 类提到的私有名词（`loadalign_brc`、`burst_count`、`ssbuf`），
   **确认主干 TileLang 里没有对应原语**。

## 运行前提

* Python 3.8+，**无第三方依赖**；
* 想跑存在性检查，需要在**本仓库的 `.workspace`** 里运行
  （脚本默认指向 `../../../../_vendor/tilelang-v0.1.15`）；
* 站点上的读者没有 vendor 源码，脚本仍会打印完整清单，只是跳过检查（会明确提示）；
* **不需要** GPU / NPU。

## 运行命令

```bash
python3 expressibility_map.py
python3 expressibility_map.py --stats-only
python3 expressibility_map.py --vendor-root /path/to/tilelang-v0.1.15
```

## 期望输出（节选）

```
== FlashMLA 昇腾稀疏 attention 的 TileLang 表达力映射 ==
vendor root: .../_vendor/tilelang-v0.1.15

--- A 类 ---
  [A] P = Q @ K^T 与 O += S @ V
        证据: kernel_body.h:1403-1491, 1493-1638
        TileLang: T.gemm  [T.* 存在 ✓]
...
--- B 类 ---
  [B] ND2NZ 的写（vsstb）
        证据: kernel_body.h:315-330
        TileLang: T.simd.vsstb(update=True) + T.simd.make_ubuf_ptr  [T.* 存在 ✓]
...
--- C 类 ---
  [C] gather2 的参数（burst_count / src_stride）
        证据: kernel_body.h:1043-1067
        TileLang: —  [主干无此原语 ✓]

== 小结 ==
  A 6 项 · B 8 项 · C 7 项
  判据一句话：换一条语义相近的指令，结论还成立吗？不成立的就是 C 类。

OK: B 类原语在锁定源码树中全部找到；C 类私有名词在主干中均未出现。
```

## 作者实际做到的验证级别

**本机运行过**（macOS Apple Silicon + Python 3.11.7），vendor 检查通过：
B 类原语全部在 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/` 中找到，
C 类私有名词（`loadalign_brc` / `burst_count` / `ssbuf`）在主干相应文件里均**未**出现。

## 已知局限

* 这是**清单 + 静态检查**，不是“证明 TileLang 一定写得出来”。
  B 类里每一项都还需要真正的实现与验证（本机没有 NPU，无法做）。
* 分类会随版本移动：今天在 C 类的，明天可能因为 `T.*` 补了原语变成 B 类
  （见正文 §6.5 与第 30 章）。
* `T.*` 的存在性检查用的是正则匹配 `def <name>(`，只能证明“函数名存在”，
  不能证明语义一致；语义一致性靠正文 §6.3 的逐条对照。
