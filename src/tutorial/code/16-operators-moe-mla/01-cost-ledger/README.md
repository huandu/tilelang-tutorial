# 01-cost-ledger · MoE / MLA 的成本账本

**目的**：把第 02 章的 roofline 记账口径套到「一层 MoE」和「一步 MLA decode」
上，得到可复算的 FLOP / 字节 / 算术强度，并据此回答「瓶颈在矩阵乘还是在通信与
重排」。其中的 DeepSeek-V3.2 形状直接读锁定版本里的第一方配置。

**运行前提**：Python 3.9+，纯标准库，无需 GPU/NPU。读配置时若找不到
`_vendor/`（把目录单独复制出去），会自动退回内置常量并在输出里标 `builtin-fallback`。

**运行命令**

```bash
python3 decode_step_ledger.py
python3 decode_step_ledger.py --json
```

**预期输出（判据，不是性能数字）**

* MoE 一节：默认形状下 `assignments = num_tokens * top_k`；打印 router / permute /
  grouped GEMM / combine 的 FLOP 与字节，并给出 grouped GEMM 的算术强度与判类。
* 单专家一节：`tokens_for_expert` 从 1 增到 4096，算术强度单调上升并在
  ~333 FLOP/Byte 处翻成 compute-bound —— 说明「每个专家 token 太少」会把
  GEMM 压成 memory-bound。
* MLA 一节：打印 MHA / GQA / MLA 每 token 每层 KV 字节、decode 的 FLOP / 字节 /
  算术强度，以及三者在同为 128 个 query head 时的算术强度对比。
* DeepSeek-V3.2 一节：从 `config_671B_v3.2.json` 读出的结构参数与
  「每 token 路由专家 FLOP」「KV cache 字节」。
* 总结表：一个 decode step 里各算子受什么限制（本节末）。

**作者实际做到的验证级别**：`CPU 逻辑验证`。在 macOS (Apple Silicon)、Python 3.11
上运行过；输出见 `chapter.md` 引用的数字。**没有任何本机实测性能数字**。
