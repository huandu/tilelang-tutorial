"""DeepGEMM-Ascend 最小调用样例（照官方 tests/ 的姿势写）。

验证级别：**仅静态核对**。本机没有昇腾 NPU，作者未运行过本文件。
运行前提见同目录 README.md。

函数名与参数顺序全部对照 corpus 快照：
  DeepGEMM-Ascend/csrc/apis/gemm.hpp:537-662 （register_apis）
  DeepGEMM-Ascend/deep_gemm/__init__.py:25-79   （Python 侧导出）
"""

import argparse

import torch
import torch_npu  # noqa: F401  # 必须在 import deep_gemm 之前，npu 后端才可用

import deep_gemm


def make_ue8m0_sf(rows: int, k: int, seed: int) -> torch.Tensor:
    """造一个昇腾格式的 scaling factor。

    关键点：SF 是「每 32 个 K 元素一个 ue8m0 字节」，再**每两个字节重新解释成
    一个 int16**（见 DeepGEMM-Ascend/README.md:48-50）。
    所以先生成 uint8 的 [rows, ceil(K/32)]，再用 .view(torch.int16) 重新解释——
    **不要**写 .to(torch.int16)（那是数值转换，不是打包）。
    对照：tests/test_mega_moe.py:22-36。
    """
    torch.npu.manual_seed(seed)
    num_elems = -(-k // 32)  # ceil(K / 32)
    if num_elems % 2 != 0:
        num_elems += 1  # 打包成 int16 需要偶数个字节
    exponent = torch.randint(123, 131, (rows, num_elems), dtype=torch.uint8, device='npu')
    return exponent.view(torch.int16)


def case_bf16() -> None:
    """D  = A @ B^T，bf16，无 scale factor。"""
    m, n, k = 4096, 7168, 16384
    a = torch.randn(m, k, dtype=torch.bfloat16, device='npu')
    b = torch.randn(n, k, dtype=torch.bfloat16, device='npu')   # NT: B 是 [N, K]
    d = torch.empty(m, n, dtype=torch.bfloat16, device='npu')
    deep_gemm.bf16_gemm_nt(a, b, d)
    torch.npu.synchronize()
    assert d.shape == (m, n)
    print(f'bf16_gemm_nt        : m={m} n={n} k={k} -> {tuple(d.shape)} OK')


def case_fp8() -> None:
    """D = A @ B^T，fp8xfp8，per-32-channel（granularity 1x32）scale。"""
    m, n, k = 4096, 7168, 16384
    a = torch.randn(m, k, dtype=torch.bfloat16, device='npu').to(torch.float8_e4m3fn)
    b = torch.randn(n, k, dtype=torch.bfloat16, device='npu').to(torch.float8_e4m3fn)
    sf_a = make_ue8m0_sf(m, k, seed=1)
    sf_b = make_ue8m0_sf(n, k, seed=2)
    d = torch.empty(m, n, dtype=torch.bfloat16, device='npu')
    deep_gemm.fp8_gemm_nt((a, sf_a), (b, sf_b), d,
                          recipe_a=(1, 32), recipe_b=(1, 32))
    torch.npu.synchronize()
    assert d.shape == (m, n)
    print(f'fp8_gemm_nt         : m={m} n={n} k={k} -> {tuple(d.shape)} OK')


def case_mgrouped() -> None:
    """MoE 的 contiguous grouped GEMM：D = A @ B^T，B 是 [G, N, K]。

    A 的所有专家行首尾相接；grouped_layout[g] 是该专家的行数。
    每个专家的行数必须对齐到 get_theoretical_mk_alignment_for_contiguous_layout()（=256）。
    """
    num_groups, m_per_group, n, k = 4, 8192, 6144, 7168
    align = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    assert align == 256 and m_per_group % align == 0

    m = num_groups * m_per_group
    a = torch.randn(m, k, dtype=torch.bfloat16, device='npu').to(torch.float8_e4m3fn)
    b = torch.randn(num_groups, n, k, dtype=torch.int8, device='npu')  # fp4 打包成 int8
    sf_a = make_ue8m0_sf(m, k, seed=3)
    sf_b = make_ue8m0_sf(num_groups * n, k, seed=4)
    d = torch.empty(m, n, dtype=torch.bfloat16, device='npu')
    grouped_layout = torch.full((num_groups,), m_per_group, dtype=torch.int32, device='npu')

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a, sf_a), (b, sf_b), d, grouped_layout)
    torch.npu.synchronize()
    assert d.shape == (m, n)
    print(f'm_grouped_*_nt_contiguous: groups={num_groups} m/group={m_per_group} '
          f'n={n} k={k} -> {tuple(d.shape)} OK')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['bf16', 'fp8', 'mgrouped'], required=True)
    args = parser.parse_args()

    # 按需限制核数；0 表示让库去问设备（见 csrc/jit/jit.hpp:34-41）。
    print('num_sms =', deep_gemm.get_num_sms(), ' npu_arch =', deep_gemm.get_npu_arch())
    {'bf16': case_bf16, 'fp8': case_fp8, 'mgrouped': case_mgrouped}[args.case]()


if __name__ == '__main__':
    main()

