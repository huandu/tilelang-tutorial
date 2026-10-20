# 图 2：topk_gate 双后端并排对照

**用途**：正文 §4.4。

事实来源：`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_cuda.py`
（全 68 行）与 `.../moe/topk_gate_asc.py`（全 157 行），逐行对照。
API 定义核对自 `_vendor/tilelang-v0.1.15/`（`language/allocate.py:296`、
`language/reduce_op.py:358,398,434`、`ascend/language/annotations.py:19`、
`ascend/language/kernel.py:139`、`ascend/language/simd.py`）。

```text
                    CUDA 版 (topk_gate_cuda.py)          Ascend 版 (topk_gate_asc.py)
                    ─────────────────────────────         ────────────────────────────
  网格               T.Kernel(num_tokens, threads=32)      T.Kernel(num_vec_cores)
                     :22                                   :30
  并行单位           1 block = 1 token，1 warp = 32 线程    1 Vector 核 = 1 token，64 lane/指令

  一行的存储         T.alloc_fragment（寄存器）             T.alloc_shared（UB）
                     :25-27                                :31

  「归约」            T.alloc_reducer(op='min'/'sum')       S.vcmax / S.vcmin / S.vmax / S.vmin
                     + reducer_init/update/finalize        :89,98-99
                     :28,41-45,54-58

  越界元素           补齐成 -inf                           用掩码屏蔽
                     :37-38                                :81,120

  非法值检测         reducer 累加 + 循环外一次 assert      x-x != 0 打标 + vmax 累积
                     :41-46                                + 循环外一次 assert :66-71,155

  剩余行            网格 = token 数，天然整除             if row < num_tokens 守卫
                                                           :47

  多迭代划分        无（一 block 一行）                    按核交错 row = w*V + core_id
                                                           :46

  流水               无                                    T.Pipelined(..., num_stages=6)
                                                           :41-45

  缓冲版本           编译器决定                            T.annotate_buffer_versions（显式）
                                                           :35

  显式同步           无                                    一句 S.mem_bar('VST_VLD')
                                                           :151

  向量宽度           编译器选（T.Parallel + 向量化）        写死 vector_length = 64
                                                           :11

  int64 输出         T.copy 做 int32→int64 转换             T.view 成 int32 视图，逐 lane 写
                     :64                                   :34,101,153

  启动/结束提示      T.pdl_sync() / T.pdl_trigger()        无（PDL 是 CUDA-only）
                     :23-24,65-66

  pass_configs       关掉 warp specialization :8-10        无（AutoSchedule 默认开）
```

## 读图提示

* 上表**没有一行改变算法语义**。每一行都在回答“这块芯片上并行单位是什么、
  数据放哪、什么时候同步、缓冲区有几份”。
* 左边多出来的两行（PDL、pass_configs）都是“CUDA 已有的能力，昇腾没有对应物”；
  右边多出来的四行（流水、缓冲版本、`mem_bar`、显式向量宽度）都是
  “昇腾必须显式表达、CUDA 交给编译器”。
* tie-break 契约（并列取最小下标）在两边都用“最小归约”实现：
  左边是 `op='min'` 的 reducer，右边是 `S.vmin` + `S.vcmin`。
