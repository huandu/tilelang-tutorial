"""Shape walkthrough and algebra check for the Engram gate (CPU only).

This script reproduces the arithmetic of ``engram_gate_fwd`` on the CPU so that
the shapes and the *fusion identity* can be checked without any accelerator.

Two code paths are evaluated on the same random data:

  * ``gate_naive``  mirrors the "one kernel per op" reading:
    RMSNorm(x, w_h) and RMSNorm(k, w_e) are materialised as two separate
    tensors, and their dot product is taken afterwards.
  * ``gate_fused``  mirrors what the shipped kernel actually does: the two
    RMSNorm weights are pre-multiplied into a single ``weight_fused`` and one
    accumulator pass computes ``sum(x * w_h * w_e * k)``.

Reference: the pure-PyTorch baseline ``engram_gate_ref`` in
``TileKernels/tile_kernels/torch/engram.py`` and the kernel docstrings in
``TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py``.

Run:  python3 engram_gate_shapes.py
"""

from __future__ import annotations

import torch


def gate_naive(x, k, w_hidden, w_embed, eps: float, clamp_value: float):
    """Two explicit RMSNorms, then their dot product (the "unfused" reading)."""
    x = x.float()
    k = k.float()

    rstd_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    rstd_k = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
    normed_x = x * rstd_x * w_hidden.float()
    normed_k = k * rstd_k * w_embed.float()

    dot = (normed_x * normed_k).sum(-1) * x.shape[-1] ** -0.5
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    return dot, signed_sqrt.sigmoid().unsqueeze(-1)


def gate_fused(x, k, weight_fused, eps: float, clamp_value: float):
    """One accumulator pass with a pre-multiplied weight (what the kernel does)."""
    x = x.float()
    k = k.float()

    rstd_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    rstd_k = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
    raw_dot = (x * weight_fused * k).sum(-1) * rstd_x.squeeze(-1) * rstd_k.squeeze(-1)
    dot = raw_dot * x.shape[-1] ** -0.5
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    return raw_dot, dot, signed_sqrt.sigmoid().unsqueeze(-1)


def main() -> None:
    torch.manual_seed(0)
    num_tokens, hc_mult, hidden_size = 5, 4, 256
    eps, clamp_value = 1e-20, 1e-6

    x = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, hc_mult + 1, hidden_size, dtype=torch.bfloat16)
    w_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
    w_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)

    # `fused_weight` is exactly this product, in fp32 (see engram_fused_weight_kernel.py).
    weight_fused = (w_hidden.float() * w_embed.float()).to(torch.float32)

    k = kv[:, :hc_mult, :]  # the first hc_mult rows of `kv` are the keys
    v = kv[:, hc_mult, :]  # the last row is the shared value

    print("== shapes ==")
    for name, tensor in [
        ("hidden_states", x),
        ("kv", kv),
        ("key k  = kv[:, :hc_mult]", k),
        ("value v = kv[:, hc_mult]", v),
        ("weight_hidden", w_hidden),
        ("weight_embed", w_embed),
        ("weight_fused (fp32)", weight_fused),
    ]:
        print(f"  {name:28s} {tuple(tensor.shape)}  {tensor.dtype}")

    dot_naive, gate_naive_v = gate_naive(x, k, w_hidden, w_embed, eps, clamp_value)
    raw_dot, dot_fused, gate_fused_v = gate_fused(x, k, weight_fused, eps, clamp_value)

    print("\n== fusion identity: unfused vs fused ==")
    print(f"  max|gate_naive - gate_fused| = {(gate_naive_v - gate_fused_v).abs().max().item():.3e}")
    print(f"  max|dot_naive  - dot_fused | = {(dot_naive - dot_fused).abs().max().item():.3e}")
    assert torch.allclose(gate_naive_v, gate_fused_v, atol=1e-5, rtol=1e-4)
    assert torch.allclose(dot_naive, dot_fused, atol=1e-5, rtol=1e-4)

    out = (x.float() + gate_fused_v * v.float().unsqueeze(1)).bfloat16()
    print(f"\n  output               shape {tuple(out.shape)}  {out.dtype}")
    print(f"  gate_score (0, :)    {gate_fused_v[0].flatten().tolist()}")
    print(f"  raw_dot   (0, :)     {raw_dot[0].tolist()}")

    # What is written out for the backward pass, and only when it is needed:
    saved = {
        "dot": raw_dot,
        "gate_score": gate_fused_v.squeeze(-1),
        "rstd_x": torch.rsqrt(x.float().pow(2).mean(-1) + eps),
        "rstd_k": torch.rsqrt(k.float().pow(2).mean(-1) + eps),
    }
    print("\n== tensors saved for backward (shape, bytes) ==")
    total = 0
    for name, tensor in saved.items():
        size = tensor.numel() * tensor.element_size()
        total += size
        print(f"  {name:11s} {tuple(tensor.shape)} {tensor.dtype}  {size:6d} B")
    print(f"  {'total':11s} {total} B for {num_tokens} tokens x {hc_mult} heads")

    print("\n== naive path: what the fusion removes ==")
    per_tensor = num_tokens * hc_mult * hidden_size * 4  # fp32, one per RMSNorm output
    print(f"  normed_x + normed_k intermediates: {2 * per_tensor} B (fp32), written then read")
    print(f"  fused path keeps only 3 per-thread accumulators (x^2, k^2, x*w*k)")

    print("\n== how many bytes does the fused kernel touch (bf16 in, bf16 out)? ==")
    read_bytes = num_tokens * hc_mult * hidden_size * 2 + num_tokens * (hc_mult + 1) * hidden_size * 2
    write_bytes = num_tokens * hc_mult * hidden_size * 2
    print(f"  read  x + kv : {read_bytes:7d} B")
    print(f"  write output : {write_bytes:7d} B")
    print(f"  weight_fused : {hc_mult * hidden_size * 4} B (fp32, resident)")


if __name__ == "__main__":
    main()
