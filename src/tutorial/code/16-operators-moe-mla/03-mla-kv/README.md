# 03-mla-kv · KV 预算与「对 kernel 友好/不友好」量化

**目的**：算清三件事——(A) MHA/GQA/MLA 的 KV 预算；(B) decode 的算术强度，
以及 MLA 跨过机器平衡点需要多少 query head；(C) MLA 的 head 维度 576/512
带来的寄存器与片上容量压力，以及官方文档给出的解法（`policy=FullCol` 切 dim）。

**运行前提**：Python 3.9+，纯标准库。

**运行命令**

```bash
python3 kv_budget.py
python3 kv_budget.py --seq 65536
```

**预期输出（判据）**

* A：`MLA / MHA` 的每 token 字节比（结构化常数，与序列长度无关）。
* B：MHA/GQA 的 `AI = 2*H_q/(H_kv*b)`；MLA 的 `AI = 2*H_q*(2*d_c+d_r)/(b*(d_c+d_r))`；
  以及跨过 H100 公开平衡点所需的 `H_q`。
* C：`acc_o`、`acc_s` 的寄存器数，和「生命周期简单相加」的 shared memory 上界
  （明确标注为**上界**，真实占用因编译器缓冲复用而更小）。

**作者实际做到的验证级别**：`CPU 逻辑验证`（实际运行过）。
