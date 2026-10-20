"""The mHC mix data flow on the CPU, kernel by kernel (no accelerator needed).

Everything here mirrors the pure-PyTorch baselines shipped with TileKernels
(``tile_kernels/torch/mhc.py``) and the shapes the kernels assert on
(``tile_kernels/mhc/*_kernel.py``):

  expand            x[n, h]                       -> [n, mhc, h]
  prenorm GEMM      (DeepGEMM, not in this file)  -> mixes / sqrsum partials
  reduce+rmsnorm    mixes[n, 24]                  (here: skipped, fed directly)
  split_mixes       mixes[n, 24]                  -> pre(4) post(4) comb(4x4)
  sinkhorn          comb 4x4                      -> doubly stochastic comb
  pre_apply_mix     residual, pre                 -> layer_input[n, h]
  <attention + MoE>
  post              layer_output, post, comb, res -> new residual[n, 4, h]
  head_compute_mix  head mixes                    -> sigmoid(scale * x + base) + eps

The script prints shapes, the doubly-stochastic check on ``comb``, and the
residual-stream recurrence ``res_{l+1} = post * out_l + comb^T @ res_l`` that
``mhc_post`` and ``mhc_multilayer_recompute`` both implement.

Run:  python3 mhc_mix_flow.py
"""

from __future__ import annotations

import torch

MHC = 4
MHC3 = 2 * MHC + MHC * MHC  # 24 == 2 * 4 + 16


def split_mixes(mixes, scale, base, post_mult, pre_eps):
    """``mhc_pre_split_mixes_ref`` (tile_kernels/torch/mhc.py:27)."""
    scale = torch.cat(
        [
            scale[0].expand(MHC),
            scale[1].expand(MHC),
            scale[2].expand(MHC * MHC),
        ]
    )
    mixes = mixes * scale + base
    pre = mixes[..., :MHC].sigmoid().unsqueeze(-1) + pre_eps
    post = (mixes[..., MHC : 2 * MHC].sigmoid() * post_mult).unsqueeze(-1)
    comb = mixes[..., 2 * MHC :].view(*mixes.shape[:-1], MHC, MHC)
    return pre, post, comb


def sinkhorn(x, repeat=10, eps=1e-6):
    """Verbatim ``sinkhorn_normalize_ref`` (tile_kernels/torch/mhc.py:8)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def apply_mix(x, mix):
    """``mhc_pre_apply_mix_ref``: o[n, h] = sum_m mix[n, m] * x[n, m, h]."""
    return (x * mix).sum(-2).bfloat16()


def post(x, residual, post_layer_mix, comb_res_mix):
    """``mhc_post_ref``: out[n, o] = x[n] * post[n, o] + sum_m comb[m, o] * res[n, m].

    ``mhc_post_ref`` uses the 4D layout ``(a, b, m, m)`` and
    ``(a, b, m, c)``; here the sequence dims are already flattened, so the
    equivalent contraction is ``einsum('nmo,nmc->noc')``.
    """
    term2 = torch.einsum("nmo,nmc->noc", comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def main() -> None:
    torch.manual_seed(0)
    num_seqs, num_tokens, hidden = 1, 3, 64
    n = num_seqs * num_tokens

    pre_mix = torch.randn(num_seqs, num_tokens, MHC, 1, dtype=torch.float32)
    post_mix = torch.randn(num_seqs, num_tokens, MHC, 1, dtype=torch.float32)
    comb_mix = torch.randn(num_seqs, num_tokens, MHC, MHC, dtype=torch.float32)
    mixes = torch.cat([pre_mix.squeeze(-1), post_mix.squeeze(-1), comb_mix.view(num_seqs, num_tokens, MHC * MHC)], dim=-1)
    residual = torch.randn(n, MHC, hidden, dtype=torch.bfloat16)
    layer_output = torch.randn(n, hidden, dtype=torch.bfloat16)

    scale = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    base = torch.zeros(MHC3, dtype=torch.float32)

    print("== shapes ==")
    for name, tensor in [
        ("residual (M branches)", residual),
        ("mixes (per token)", mixes.view(n, MHC3)),
        ("layer output", layer_output),
    ]:
        print(f"  {name:24s} {tuple(tensor.shape)} {tensor.dtype}")
    print(f"  MHC3 = 2*MHC + MHC^2 = 2*{MHC} + {MHC * MHC} = {MHC3}")

    pre, post_m, comb = split_mixes(mixes.view(n, MHC3), scale, base, post_mult=1.0, pre_eps=1e-6)
    comb_norm = sinkhorn(comb, repeat=10, eps=1e-6)

    print("\n== split + sinkhorn ==")
    print(f"  pre  {tuple(pre.shape)}  post {tuple(post_m.shape)}  comb {tuple(comb.shape)} -> normalized")
    print(f"  row sums of comb: {[round(v, 6) for v in comb_norm.sum(-1)[0].tolist()]}")
    print(f"  col sums of comb: {[round(v, 6) for v in comb_norm.sum(-2)[0].tolist()]}")

    layer_input = apply_mix(residual, pre)
    print(f"\n  layer_input = sum_m pre_m * residual_m : {tuple(layer_input.shape)} {layer_input.dtype}")

    new_residual = post(layer_output, residual, post_m, comb_norm)
    print(f"  new residual = post * layer_output + comb^T @ residual : {tuple(new_residual.shape)}")

    # The recurrence that both `mhc_post` and `mhc_multilayer_recompute` implement:
    print("\n== residual-stream recurrence, layer by layer ==")
    res = residual
    for layer in range(3):
        li = apply_mix(res, pre)
        if layer < 2:
            res = post(layer_output, res, post_m, comb_norm)
        print(f"  layer {layer}: |layer_input| mean = {li.float().abs().mean().item():.4f}, |residual| mean = {res.float().abs().mean().item():.4f}")

    # The backward of `expand` is a sum over branches; the head version is a sigmoid.
    print("\n== the two boundary ops ==")
    plain = residual.mean(dim=1).contiguous()  # a single stream [n, h]
    expanded = plain.unsqueeze(1).expand(n, MHC, hidden).contiguous()  # fwd: copy into M branches
    collapsed = expanded.float().sum(1)  # bwd: sum over branches
    print(f"  expand   fwd: x[n,h] -> [{MHC}] copies;  bwd: sum over the {MHC} branches")
    print(f"  expanded {tuple(expanded.shape)} -> collapsed {tuple(collapsed.shape)} (same as the input)")
    print(f"  head_compute_mix: sigmoid(x * scale + base) + eps  (eps keeps the gate > 0)")


if __name__ == "__main__":
    main()
