# 图 1：一次训练迭代的算子地图

对应正文第 1.1 节。一次 `loss.backward(); optimizer.step()` 里，算子出现在四个位置：
前向、反向、优化器、通信。每个位置下面标出这一阶段在 roofline 上的典型判据。

```mermaid
flowchart TB
    DATA["一个 batch<br/>input_ids / labels"] --> FWD

    subgraph FWD["前向（forward）：每一层重复"]
        direction TB
        F1["norm<br/>RMSNorm / LayerNorm"]
        F2["投影 GEMM<br/>QKV / O / gate / up / down"]
        F3["attention<br/>QK^T → softmax → PV"]
        F4["逐元素与融合<br/>RoPE、残差加、SiLU、mask"]
        F5["loss<br/>cross entropy / logits"]
        F1 --> F2 --> F3 --> F4 --> F2 --> F5
    end

    FWD --> LOSS["标量 loss"]
    LOSS --> BWD

    subgraph BWD["反向（backward）：与前向逐层镜像"]
        direction TB
        B1["dY = dOut · W^T（每个前向 GEMM 的 2 倍 FLOPs）"]
        B2["dW = X^T · dOut（权重梯度）"]
        B3["norm / 激活 / mask 的反向"]
        B4["attention 反向<br/>需要重算或保存 softmax 中间量"]
        B1 --> B2 --> B3 --> B4
    end

    BWD --> OPT

    subgraph OPT["优化器（optimizer）：参数更新"]
        direction TB
        O1["AdamW：读 grad / param / m / v"]
        O2["写回 param / m / v"]
        O1 --> O2
    end

    BWD --> COMM
    subgraph COMM["通信（数据并行时）"]
        direction TB
        C1["梯度 all-reduce / reduce-scatter<br/>无算术，纯搬运"]
    end

    OPT --> NEXT["下一个 iteration"]
    COMM --> NEXT
```

各阶段的判据（用本书 `code/02-step-budget/step_budget.py` 的粗模型算出来，
8B 级模型、seq_len=4096、batch=1、8 卡、H200 标定参数）：

| 阶段 | 下界由谁决定 | 为什么 |
| --- | --- | --- |
| 前向 | 算力 | 长序列下 attention 与 GEMM 的 FLOPs 远大于权重与激活的字节数 |
| 反向 | 算力 | 每个 GEMM 的反向是它的 2 倍 FLOPs |
| 优化器 | 带宽 | 每参数要搬 ±26 字节（grad、param、m、v 的读写），算术只有十几 FLOP |
| 通信 | 网络带宽 | 没有算术，只有 all-reduce 的搬运 |

## 来源

* 优化器的状态张量与公式：`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_aolapi_context_ops-nn_aclnnApplyAdamW_md.md:25-60`
  （CANN 官方算子文档，`aclnnApplyAdamW` 的 `varRef/mRef/vRef` 与公式）。
* 「反向 ≈ 2 倍前向」的推导见正文 1.1 节（对单个 GEMM 逐项做加法）。
* 各阶段占比的具体数字由 `code/02-step-budget/step_budget.py` 复算，属本书的粗模型推导，
  **不是实测**。
