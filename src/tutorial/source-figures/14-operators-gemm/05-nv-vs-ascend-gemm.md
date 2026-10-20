# 图 5：NV 与昇腾的 GEMM 数据通路对照

**用途**：正文第 6 节。

```text
NVIDIA（TileLang CUDA 后端）

  GM(global) ──T.copy / TMA──▶ shared memory ──ldmatrix──▶ fragment(reg)
       ▲                              │                        │
       │                              │                        ▼
       └──T.copy(C_local, C)──────────┴──stmatrix──◀──── T.gemm(MMA)

  · 片上复用单元 = shared memory（每个 SM 可配到 228 KB）
  · 累加器       = fragment（寄存器）
  · 同步         = 隐式（cp.async / mbarrier 由 pass 处理）


昇腾（TileLang Ascend 后端）

  GM ──DataCopy──▶ L1(512KB) ──LoadData──▶ L0A/L0B(各 64KB)
                                                  │
                                       T.gemm_v0 / T.mma (Cube)
                                                  │
                                                  ▼
                                            L0C(128KB, fp32 累加)
                                                  │
                            FixPipe / dual_copy ──┴──▶ UB(192KB, Vector) ──▶ GM

  · 片上复用单元 = L1
  · 累加器       = L0C（不是寄存器文件）
  · 同步         = MTE1/MTE2/Cube/Vector 之间的 Set/Wait flag
                   （主干后端由 InsertSync 自动插入；适配器示例里手写）
```

## 术语对照

| 概念 | NVIDIA | 昇腾 |
| --- | --- | --- |
| 片上共享缓冲 | shared memory | L1 |
| 矩阵乘的输入缓冲 | （fragment） | L0A / L0B |
| 累加器 | fragment（寄存器） | L0C |
| 向量/后处理缓冲 | （fragment / shared） | UB |
| tile 级矩阵乘 | `T.gemm` | `T.gemm_v0` / `T.gemm`（带 `unit_flag_ctrl`） |
| 同步 | 隐式 | 显式 flag（主干自动、适配器手工） |

来源：S16、S17（`tilelang/ascend/README.md`）、R5（Atlas A2 存储层级：
L1 512KB、L0A/L0B 各 64KB、L0C 128KB、UB 192KB）。
