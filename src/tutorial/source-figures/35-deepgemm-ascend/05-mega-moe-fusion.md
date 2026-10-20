# 图 5：MegaMoE 的五段融合

**用途**：正文 §5.6。

**事实来源**（静态阅读）：

* `DeepGEMM-Ascend/README.md:146-158`（官方定义与性能表）。
* `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/layout/mega_moe.hpp:26-33`（四个进度阶段）。
* `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/mega_moe.hpp:20-107`
  （AIC/AIV0/AIV1 分工与 pipe event 分配）。
* `DeepGEMM-Ascend/csrc/apis/mega_moe.hpp:128-204`（缓冲区与约束）。

```mermaid
graph LR
    X["x（fp8 + sf）<br/>topk_idx / topk_weights"] --> D["① EP dispatch<br/>把 token 送到专家所在 rank"]
    D --> G1["② Linear1<br/>grouped GEMM (fp8×fp4)"]
    G1 --> S["③ SwiGLU + 量化到 fp8<br/>（AIV0 解码 fp4 权重 → fp8 NZ）"]
    S --> G2["④ Linear2<br/>grouped GEMM → bf16"]
    G2 --> C["⑤ EP combine<br/>按 top-k 权重归约回原 token"]
    C --> Y["y（bf16）"]

    subgraph K["同一个 kernel（AIC + 2×AIV 分工）"]
        D
        G1
        S
        G2
        C
    end

    subgraph P["进度阶段（layout/mega_moe.hpp:26-33）"]
        P0["kDispatched<br/>Linear1 输入就绪"] --> P1["kLinear1Ready<br/>Linear2 输入就绪"]
        P1 --> P2["kLinear2Ready<br/>BF16 专家输出就绪"]
        P2 --> P3["kCombineSent<br/>发送完成，输出环可复用"]
    end
```

**要点**

* 五段被压进**一次 launch**：中间结果不再写回 HBM 再读出来，通信与计算重叠。
* 代价是约束变紧：多进程 + HCCL 对称缓冲区、形状必须提前注册、
  核数固定 32（`csrc/apis/mega_moe.hpp:159`）、每次 launch 要清 progress 区
  （`csrc/jit_kernels/mega_moe.hpp:43-46`）。
* 性能表里有**单独一列 Communication bandwidth**（33–135 GB/s），
  说明通信在这个融合 kernel 里是一等公民。

