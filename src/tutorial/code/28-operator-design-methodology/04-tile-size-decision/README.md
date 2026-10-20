# 示例 04：tile 尺寸决策流程

## 目的

把第 28 章第 4 节「先粗算再微调」写成一台检查机。它按固定顺序检查四条约束，
并报出**每条候选被哪一条卡住**：

```
C0  DRAM 流量    b*M*N*K*(1/block_N + 1/block_M) + b*M*N   （越小越好，没有上限）
C1  片上容量     stages * (block_M*block_K + block_K*block_N) * elem_bytes
                 <= 每 block 可用 shared memory
C2  寄存器压力   累加器/线程，按「每 warp 256 个寄存器」向上取整，<= 255 /线程
C3  指令形状     block_M/block_N/block_K/threads 必须是所选 MMA 形状的整数倍
C4  并行度       grid / (blocks_per_SM * SM 数) -> 波数与最后一波的填充率
```

常量与判据的出处：

| 事实 | 出处 |
| --- | --- |
| shared memory 每 SM 228 KB、每 block 上限 227 KB | `tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md` |
| 每 SM 65536 个 32 位寄存器、最多 2048 线程、100% occupancy 时每线程 32 个、寄存器按每 warp 256 个取整 | `tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:3126` |
| `mma` 密集 f16 形状 `m16n8k8` / `m16n8k16` | `tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:23476-23477` |
| `wgmma` 密集 f16 形状 `m64nNk16` | 同上 `:28132` |
| warpgroup = 4 个连续 warp = 128 线程 | 同上 `:28113-28114` |

## 运行前提

* Python 3.9+，只用标准库。
* **不需要 GPU / NPU / TileLang / PyTorch。**
* 通过 `--sm-count` / `--smem-kb` 描述目标设备，默认值对应 H100/H200（132 SM、228 KB）。

## 运行命令

```bash
# 扫一遍，按 DRAM 流量排序给出候选
python3 tile_decision.py --topk 8

# 给一份配置做逐条体检
python3 tile_decision.py --diagnose --block-m 128 --block-n 256 --block-k 64 \
                         --stages 3 --threads 256

# 故意超预算，看报错长什么样
python3 tile_decision.py --diagnose --block-m 256 --block-n 256 --block-k 64 \
                         --stages 4 --threads 256

# 换成 mma 路径 / 换设备
python3 tile_decision.py --instr mma --topk 8
python3 tile_decision.py --sm-count 108 --smem-kb 164 --topk 8
```

## 期望输出

正确性判据：退出码为 0；`--diagnose` 对合法配置打印 `OK all four constraints satisfied`，
对超预算配置打印 `FAIL` 行并指出是哪一条。

`--topk 8` 的实际输出（节选）：

```text
feasible configurations found: 459
 block_M block_N block_K st  thr trafficMB  smemKB regs/thr blk/SM waves   tail  binding
     256     256      16  2  512    1107.3    32.0      128      1     2    94%  regs
     256     256      16  3  512    1107.3    48.0      128      1     2    94%  regs
     ...
configurations tied at the minimum traffic: 8
best last-wave fill among feasible configurations: 94%
```

`--diagnose --block-m 256 --block-n 256 --block-k 64 --stages 4 --threads 256` 的实际输出：

```text
  C1 shared/stage     64.0 KB -> total    256.0 KB
  C2 accumulators   256/thread ->  256 regs/thread (warp-rounded),  65536 regs/block
     occupancy: regs -> 1 blk/SM, smem -> 0, threads -> 8  => 0 blk/SM ...
  FAIL  C1 shared 262144 B > 232448 B per block at stages=4
  FAIL  C2 accumulators alone need 256 registers/thread (> 255); expect spilling to local memory
  FAIL  C1/C2 no resident block fits: blocks_per_SM = 0
```

## 性能口径

**不测量任何性能，也不预测时间。** 输出里没有 µs 这一列，因为一个「每波多久」的
估计需要每 SM 的 MMA 吞吐，而本机的素材里没有一条可直接引用的公开数字。
这份脚本只回答「能不能放下、够不够并行、形状合不合法」——这三件事是可以纯算术
判定的事实，剩下的必须去测。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认、`--diagnose`
（含故意超预算的两组）、`--instr mma` 与多组设备参数；输出与本文件一致。
没有在 GPU 上运行过任何 kernel。

## 已知局限

* 寄存器估算只算累加器，**不含**地址、索引、流水所需的寄存器；真实占用一定更高。
  所以 `regs/thr` 是一个下界，`blocks/SM` 是一个上界。
* `C1` 只有 A/B 两个 tile，不含 epilogue、barrier、padding；
  真实的 TileLang 会做 shared memory 合并与 live-range 复用
  （见第 11 章 4.5 节），所以这一列也是上界。
* `C4` 只算「波数」和「最后一波的填充率」，不含 wave quantization 的耗时模型。
* `--instr` 只区分 `mma` / `wgmma` / `simt` 三类形状约束，不区分布局与 swizzle 约束。
