# 第 18 章 习题答案

每题给出**解题思路 → 参考答案 → 常见错误**。涉及计算的题目都标出可复算的脚本
（在 `code/` 下），因为本章的所有数字都应该是可复算的。

---

## E1 概念题 ★

**思路**：回到 `AI = FLOP / Byte` 的定义，看哪一项在 `M` 很小时占主导。

**参考答案**：

1. `FLOP = 2MNK`，`Byte = b·(M·K + K·N + M·N)`。当 `M ≪ N, K` 时，
   `K·N`（权重）远大于 `M·K` 与 `M·N`（激活），可以省略后两项：

   ```text
   AI ≈ 2MNK / (bKN) = 2M/b
   ```

   省略的依据是“权重一次读进来服务整个 batch，激活的字节数与 M 成正比而
   权重的字节数与 M 无关”。

2. 令 `2M/b = 333`，`b = 2`，得 `M* ≈ 333`。二分求解（考虑被省略的两项）
   得到 375（`N=18944, K=3584`）。可用
   `python3 code/01-prefill-decode-roofline/prefill_decode_roofline.py` 复核。

3. 方形 GEMM 里 M、N、K 一起涨，`Byte ≈ 3bM²`、`FLOP = 2M³`，于是
   `AI = 2M/(3b)`，`M* ≈ 3b·balance/2 ≈ 10³`。本章固定 N、K 只让 M 涨，
   所以是 375。**报 `M*` 必须一起报形状**：只报数字等于没报。

**常见错误**：把 `2M/b` 里的 `b` 当成“字节数 2”又乘一次；或者直接照抄
第 14 章的 `10³` 而不说明形状前提。

---

## E2 概念题 ★

**思路**：先写公式，再代数；最后一问要把“省显存”和“KV 复用率”联系起来。

**参考答案**：

1. 每 token 每层 `2 · H_kv · head_dim · b = 2 · 4 · 128 · 2 = 2048 B = 2.0 KiB`。
2. 每 token 全模型 `2048 · 28 = 57344 B = 56.0 KiB`；context = 131072 时
   `57344 · 131072 ≈ 7.52e9 B = 7.00 GiB`。
3. MHA 是 `2 · 28 · 128 · 2 = 14336 B`/层/token，全模型 392 KiB/token，
   同 context 下 49.00 GiB（是 GQA 的 7 倍）。
4. `AI = 2·H_q / (H_kv·b)`。GQA：`2·28/(4·2) = 7.0`；MHA：`2·28/(28·2) = 1.0`。
   两者都远低于平衡点 333，都是 memory-bound。
5. GQA 之所以省 7 倍显存，正是因为它让一份 KV 被 7 个 query head 复用；
   复用率越高，每个 KV 字节被用到的次数就越多（算术强度越高），省显存与
   “每字节多算几次”是同一件事的两种说法，代价是表达能力受限。

可用 `python3 code/02-kv-ledger/kv_ledger.py` 复核全部数字，它会同时打印
“GQA 每 token 比 MHA 省 7 倍”和“算术强度只有 7.0”两句话。

**常见错误**：忘记 KV 是两份（K 与 V）；或者把 `H_q` 也算进 KV 的字节数里。

---

## E3 概念题 ★★

**思路**：从“长度事先不可知”这个事实出发。

**参考答案**：

1. 分页解决的是“KV 长度事先不知道、连续分配只能按上界预留”的问题。
   浪费从“整条序列跑不满预留长度（可能浪费几十 MB 到几 GB）”转移到
   “每条序列最后一块内的碎片（最多 `block_size−1` 个 token 槽位）+
   `block_table` 的元数据”。
2. 新要求是 **KV 的地址要查表**：从 `k · block_N` 这类算术地址变成
   “读 `block_table[seq][logical_block]` 再加偏移”。TMA（以及任何基于
   静态描述符的搬运）要求地址在描述符构造时确定，而查表结果是运行期的，
   所以 paged KV 的路径通常要走 SIMT/普通 `T.copy` 的 fallback。
3. `block_size = 16`：每层每块 32.00 KiB，全模型每块 896.00 KiB，
   最后一块最多浪费 15 个槽位（块内 93.8%）。
   `block_size = 128`：每层每块 256.00 KiB，全模型每块 7.00 MiB，
   最后一块最多浪费 127 个槽位（块内 99.2%）。
   更大的块 = 更少的分配元数据与更少的查表次数，但单条序列的碎片更大。

**常见错误**：以为分页能“完全消除碎片”；实际上它只是把碎片限制在块内。

---

## E4 概念题 ★★

**思路**：抓住“一个 decode step 的耗时基本不随 batch 变化”这个事实。

**参考答案**：

1. 因为一个 decode step 的时间主要由“读一遍权重”决定（与 batch 无关），
   所以让空槽位立刻被新请求填上，等于用同一份权重流量产出更多 token：
   吞吐上去，同时新请求不用等整批结束、已完成请求也不用陪跑，所以平均延迟
   也下降（前提是槽位没被占满）。
2. 三件新要求：**varlen**（同批长度不齐，要用 `cu_seqlens`）、**paged KV**
   （KV 在显存里不连续，要查表）、**prefill/decode 混步**
   （chunked prefill 让长 prompt 与 decode 同一步跑）。
3. 因为 grid 大小与中间张量形状都由 `max_seqlen` 决定，它必须在编译期确定。
   运维现象：换一个更长的 prompt 会触发重新编译（JIT 缓存未命中），
   表现为“第一个长请求特别慢”；框架的应对是按桶编译或固定形状后 padding。
4. 槽位占满后，再加并发只会让请求在队列里等：吞吐趋于平台的饱和值（可能
   因为 batch 变大导致单步变慢而略降），延迟则随并发数上升。
   依据：第 3.1 节的敏感度表，batch 从 1 到 32、context 8192 时，单步下界
   从 4.87 ms 涨到 9.79 ms，说明“加 batch 摊薄权重”的收益是递减的。

**常见错误**：以为“吞吐”与“延迟”在任何情况下都此消彼长；在槽位未占满时
它们可以同时变好。

---

## E5 代码阅读题 ★★

**思路**：先算 group 数，再看 grid 的组织方式。

**参考答案**：

1. `kv_group_num = 28 // 4 = 7`；`valid_block_H = min(64, 7) = 7`。
   （这就是为什么官方用 `min`：不能一次处理超过一个 group 的 KV head 数，
   否则一个 block 内不同 head 对应的 KV head 会不同。）
2. `cur_kv_head` 算的是本 block 第一个 query head 对应的 KV head。
   它算错时通常**不会崩溃**，索引仍然在合法范围内，只是读了别的 head 的
   KV，结果是“数值偏了但形状对”。这类错误靠相似度指标
   （官方用 `assert_similar(..., eps=1e-2)`）才能发现。
3. `(batch, heads // valid_block_H, num_split)`：batch 方向、head 分组方向、
   KV 分段方向。第二维写成 `heads // valid_block_H` 是因为一个 block 一次
   处理 `valid_block_H` 个 query head，head 方向被这个块宽度整除了。

**常见错误**：以为 `cur_kv_head = hid`（忽略 group）；或者忽略
`valid_block_H = min(block_H, kv_group_num)` 这个夹取。

---

## E6 代码阅读题 ★★

**思路**：从“每段输出已经各自归一化过”出发。

**参考答案**：

1. 第 s 段输出已经除过自己的 `ℓ_s`：
   `O_s = (1/ℓ_s)·Σ_{j∈s} exp(s·x_j − m_s)·v_j`。真实输出是
   `O = Σ_j P_j v_j = (Σ_s ℓ_s·O_s) / (Σ_s ℓ_s)`，所以权重必须是
   `ℓ_s / Σ_{s'} ℓ_{s'}`，而 `ℓ_s = exp(lse_s)`，于是
   `scale = exp2(lse_s − logsumexp(lse))`。**不是简单相加。**
2. 是同一个技巧。先减最大值保证指数不为正、不会上溢；`exp2` 与 `log2`
   配对着实现“比值”这一运算。这与第 15 章 online softmax 的
   `c = exp(m_prev − m_new)` 是同一个数值手法的迁移。
3. 风险在低精度下的归约顺序与舍入。本章引用过的量级：昇腾 backward 里
   dQ 用 bf16 跨 64 个 KV tile 原子累加，相对 L2 误差约 1.024%，
   且不保证 bitwise 确定性（**厂商自报，作者未复现**）。

**常见错误**：把 `ℓ_s` 当成常数（各段长度相等时不成立）；或者用
`exp` 而不是 `exp2` 却仍然按 `log2` 域的 lse 去减。

---

## E7 代码阅读题 ★★

**思路**：注意对齐公式里的 `+1`，以及 kernel 侧对越界行的夹取。

**参考答案**：

1. `+1` 让每个专家的对齐区间多出一行。作用是给“行数正好是 `padding_M`
   整数倍”的专家留出余量，避免边界上出现“零宽度块”或越界读；
   也让 `group_idx_for_bx` 在扫描时更容易为每个 block 找到归属。
   读别人的实现时，这类 `+1` 一定要单独确认语义，不能想当然。
2. `T.max(0, ...)` 防御的是“本 block 落在对齐区间里、但已经完全超出该专家
   有效行数”的情况（对齐 padding 的行）。此时 `actual_rows = 0`，
   循环里所有写回都被 `if i < actual_rows` 挡掉，等于空转一个 block。
3. 如果 host 侧的 `group_sizes` / 偏移与 kernel 拿到的 `X` 行顺序不一致，
   kernel 会读到别的专家的 token：不崩溃、形状对，但输出完全错。
   这也是为什么 MoE 的 dispatch 契约值得单独写一个可复算的检查
   （见 `code/07-tilelang-moe-and-quant/moe_dispatch_contract.py` 的三项断言）。

**常见错误**：以为 `actual_rows = 0` 是异常；它是对齐设计的正常结果。

---

## E8 代码阅读题 ★★★

**思路**：这题的关键是版本归属，不是语法。

**参考答案**：

1. **不能。** 主干 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py`
   里没有 `is_npu` 这个参数；`is_npu` 是适配器
   `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/kernel.py:220,247`
   的参数。验证方式：用 `rg -n "is_npu" <目录>` 在两条工作树上各检索一次，
   主干无命中。主干的昇腾写法是 `with T.Kernel(N_CORES) as core_id`
   （见 `_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py`），
   并把 auto-schedule 交给 pass context 控制
   （`tilelang/ascend/pipeline.py` 里的 `allow_autoschedule`）。
   同理，`TL_ASCEND_AUTO_SYNC` 这类 pass 配置键只在适配器里存在。
2. `cid` 是核（block）的索引，`vid` 是核内向量单元 / 子任务的索引。
   按 `vec_num` 把 `block_m` 行切成两半，是为了让一个核内的多个向量单元
   同时处理不同的行，提高并行度；`row_x = cid * block_m + vid * rows_per_vec`
   就是把“核 + 核内子任务”映射到全局行号。
3. 需要保证 capture 阶段的所有输入形状是静态的（`m`、`head_dim`、
   `block_m` 都是编译期常量），并且没有 host 侧同步或依赖数据值的
   动态控制流；否则 capture 会失败或 replay 结果错。另一个容易漏的点是
   所有算子必须落在同一条 NPU stream 上。这正是
   `tilelang/ascend/torch_npu_exchange.pyx:1-6` 那段 DLPack 补丁的作用。

**常见错误**：把两套方言的 api 混着写（`T.tile.mul` 是适配器的，
主干的昇腾例子用 `T.SimtVF` + `T.Parallel`）。

---

## E9 动手题 ★★

**思路**：把 `b` 从 2 改成 1，两处公式各代一次。

**参考答案**（可用 `code/01` 的 `--json` 取形状后手算）：

1. 简化式 `M* ≈ balance · b / 2 = 333 · 1 / 2 ≈ 167`；把被省略的 `M·K`、
   `M·N` 两项放回去再二分求解得到 **177**（可用
   `crossover_m(18944, 3584, 1)` 复算）。翻转点相对 bf16 的 375 提前了一半多。
2. 权重字节 15.23 GB / 2 = 7.615 GB；`7.615e9 / 3.0e12 = 2.54 ms`；
   单请求上界 `1000/2.54 ≈ 394 tok/s`。
3. 因为 decode 的耗时几乎完全由“读字节数”决定（`AI ≪ balance`），
   而量化直接把字节数减半；prefill 处在 compute-bound 一侧，时间由 FLOP
   决定，量化只能间接帮上忙（把张量核喂饱、减少搬运），收益小得多。

**常见错误**：把 fp8 的 `b` 写成 1 之后忘了 KV cache 是否也量化，
KV 若保持 bf16，attention 那部分的账本不变。

---

## E10 动手题 ★★

**思路**：改 `--batch`，观察哪一列在涨、哪一列不变。

**参考答案**：

1. 参考输出（可用脚本复算）：

   | batch | attention 占比 | 权重类算子占比 | 单步下界 |
   | --- | --- | --- | --- |
   | 1 | 3.2% | 96.8% | 4.87 ms |
   | 8 | 20.9% | 79.0% | 5.98 ms |
   | 32 | 51.2% | 48.7% | 9.79 ms |
   | 128 | 80.2% | 19.7% | 25.00 ms |

   （context = 8192。）

2. 权重类算子的**字节数不随 batch 变**（权重只读一遍），所以它们的绝对时间
   不变；但 attention 的字节数正比于“所有在跑序列的 KV 总长”，也就是
   `batch × context`，所以随 batch 线性增长。当 attention 涨到超过权重流量
   之后，单步时间就被它拖着走。
3. 至少还缺：真实硬件的实际带宽与算力（模型用的是 H100 公开规格）、
   每个 kernel 的实际效率（模型假设完美重叠）、算子启动与同步开销、
   KV cache 的 dtype 与量化方案、并行策略（TP/EP）与通信、
   以及真实请求的长度分布（模型只用了同一个 context）。

**常见错误**：把“下界时间变长”解读成“加 batch 有害”，
吞吐是“token 数 / 时间”，要一起看。

---

## E11 动手题 ★★★（可选硬件）

**思路**：这题的价值在于“口径”而不是“数字”。**没跑过就不要写数字。**

**参考答案**：

1. 官方示例的正确性判据是 `assert_similar(o, o_ref, name="o_ref")` 与
   `assert_similar(o, o_ref_split, name="o_ref_split")`（相似度 `eps = 1e-2`），
   通过时打印 `All checks pass.`；延迟用
   `profiler.do_bench(warmup=500)` 打印 `Ref: ... ms` / `Tile-lang: ... ms`
   以及对应的 TFlops（`2·batch·heads·kv_seqlen·dim`）。
2. `num_split ∈ {1, 4, 8}` 时你会看到：split 太少并行度不足、太多则 combine
   段与 `Output_partial` 的流量开始占主导。曲线的形状取决于卡型与
   `seqlen_kv`；**具体极值必须在你的机器上测**，不能照抄别人的配置。
3. 对比表应包含：卡型、`batch`、`heads`、`groups`、`seqlen_kv`、`dim`、
   `block_M`、`block_N`、`num_split`、`num_stages`、`threads`、预热方式、
   取数方式（mean/median/min）、以及是否 flush L2。
4. 与 4.1 节一致的方向是：并行度随 `num_split` 上升、边际收益递减；
   官方把 `num_split` 放进 autotune 空间（`[1, 2, 4, 8]`）本身就说明
   它没有解析最优解。

**常见错误**：不 flush L2 就报带宽；或者把 `do_bench` 的默认
`return_mode="mean"` 改成 `min` 之后与别人的 `mean` 比。

---

## E12 动手题 ★★★（可选硬件）

**思路**：三条命令分别回答“对不对”“入没入图”“快在哪”。

**参考答案**：

1. `python3 aclgraph_decode_block.py` 应打印 `Kernel Output Match!`
   （`torch.testing.assert_close(..., rtol=1e-2, atol=1e-2)` 通过）。
   注意这个判据只说明“数值对”，不说明“入图成功”，两者要分开验证。
2. 入图确认：设 `ASCEND_HOST_LOG_FILE_NUM=1000`、
   `ASCEND_PROCESS_LOG_PATH=logs`、`ASCEND_GLOBAL_LOG_LEVEL=1`，
   重跑后在 host 日志里搜 `capture model`（1 = 入图生效，0 = 未入图）。
   这与适配器 `examples/aclgraph/README.md` 的“Verifying AclGraph is
   Enabled”一节一致。
3. `msprof op --kernel-name="main_rms" python3 aclgraph_decode_block.py`
   会给出计算内存热力图、Roofline 瓶颈分析图、Cache 热力图、通算流水图、
   算子代码热点图。它们分别回答：数据搬了多少、离 roofline 多远、
   cache 是否命中、搬运与计算有没有重叠、代码热点在哪个循环。
4. eager 与 graph 的**端到端**时间差主要来自 host 侧下发：
   capture 之后一次 replay 就把整段算子序列提交下去，绕过了 Python 与框架
   调度；而 kernel 本身的时间两者相同。差异在算子多、kernel 小的时候最明显
   （host-bound 场景），在单个大 kernel 时可以忽略。

**常见错误**：只做静态核对却在报告里写“已入图”；或者用 kernel-only 的
`do_bench` 数字去论证图模式的收益（图模式的收益在端到端口径里）。
