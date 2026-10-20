# 06-ascend-map · 昇腾侧对照与 EP 通信账

**目的**：(A) 把昇腾侧四个 example（fusedmoe / moe_token_permute /
dispatch_combine / deepseek_v4）对齐到本章的 MoE 数据流，并核对每条引用的行号；
(B) 算专家并行（EP）下 dispatch/combine 的字节量，并与激活字节对比。

**运行前提**：Python 3.9+，纯标准库；核对需要 `_vendor/`。

**运行命令**

```bash
python3 ascend_map.py
python3 ascend_map.py --tokens 8192 --top-k 8 --d-model 7168
```

**预期输出（判据）**

* 引用核对：`命中 12 处，未命中 0 处`。
* 通信账：每 token 每方向 `k*d*b` 字节；dispatch/combine 总量；与激活字节的比值；
  以及一组**明确标注为占位**的带宽-时间换算表（时间 = 字节 / 带宽）。

**作者实际做到的验证级别**：`CPU 逻辑验证`（实际运行过）。
**没有**任何设备带宽或实测性能数字；带宽只是占位示例。
