"""Engram addressing: multiplicative-XOR hashing and an O(1) table lookup.

The Engram module retrieves static memory rows with *deterministic* indices:
the index depends only on the input token ids, never on the hidden state. That
is what lets the paper claim the big embedding table can be offloaded to host
memory and prefetched ahead of time.

This script reproduces ``engram_hash_ref`` (``tile_kernels/torch/engram.py:118``)
and the kernel's index arithmetic (``tile_kernels/engram/engram_hash_cuda.py:54``)
on the CPU, and then shows that:

  * the same token ids always produce the same indices (deterministic addressing);
  * the number of gathered rows per token is fixed by the *architecture*
    (``(max_ngram_size - 1) * num_embed_table_per_ngram`` per layer), not by the
    table size, so growth in table size is free at lookup time;
  * image tokens get index ``-1``, the kernel's "do not retrieve" marker.

Run:  python3 engram_hash_lookup.py
"""

from __future__ import annotations

from math import isqrt

import torch


def next_prime_after(value: int, seen: set[int]) -> int:
    candidate = value + 1
    while True:
        if candidate > 1 and all(candidate % d for d in range(2, isqrt(candidate) + 1)) and candidate not in seen:
            seen.add(candidate)
            return candidate
        candidate += 1


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sums per layer (``tile_kernels/torch/engram.py:99``)."""
    layers = []
    for layer in range(vocab_sizes.shape[0]):
        flat = vocab_sizes[layer].reshape(-1)
        prefix = torch.cat([torch.zeros(1, dtype=torch.int32), flat[:-1].cumsum(0, dtype=torch.int32)])
        layers.append(prefix)
    return torch.stack(layers, dim=0)


def engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets):
    """Mirrors ``engram_hash_ref`` and the kernel's unrolled hash (CPU, int64)."""
    num_layers, max_ngram = multipliers.shape
    heads = vocab_sizes.shape[-1]
    out_cols = (max_ngram - 1) * heads

    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)
    hashes = prod[:, :, 0].clone()
    per_layer = [[] for _ in range(num_layers)]
    for ngram in range(1, max_ngram):
        hashes.bitwise_xor_(prod[:, :, ngram])
        for layer in range(num_layers):
            mod = vocab_sizes[layer, ngram - 1].to(torch.int64).unsqueeze(0)
            per_layer[layer].append((hashes[layer].unsqueeze(-1) % mod).to(torch.int32))
    out = torch.stack([torch.cat(cols, dim=-1) for cols in per_layer], dim=0)
    return out + offsets.unsqueeze(1)


def main() -> None:
    torch.manual_seed(0)
    num_tokens, max_ngram, num_layers, heads = 6, 3, 2, 8

    vocab_size = 2**17  # the demo uses `engram_vocab_size = 129280 * 5`-ish magnitudes
    seen: set[int] = set()
    vocab_sizes = torch.empty(num_layers, max_ngram - 1, heads, dtype=torch.int32)
    for layer in range(num_layers):
        for ngram in range(max_ngram - 1):
            for head in range(heads):
                vocab_sizes[layer, ngram, head] = next_prime_after(vocab_size + head, seen)

    offsets = make_offsets(vocab_sizes)
    multipliers = torch.randint(1, vocab_size, (num_layers, max_ngram), dtype=torch.int64) * 2 + 1
    token_ids = torch.randint(0, vocab_size, (num_tokens, max_ngram), dtype=torch.int32)

    indices = engram_hash(token_ids, multipliers, vocab_sizes, offsets)

    print("== shapes ==")
    print(f"  ngram_token_ids     {tuple(token_ids.shape)} int32")
    print(f"  multipliers         {tuple(multipliers.shape)} int64")
    print(f"  vocab_sizes         {tuple(vocab_sizes.shape)} int32 (prime, per head)")
    print(f"  offsets             {tuple(offsets.shape)} int32 (exclusive prefix sum)")
    print(f"  indices             {tuple(indices.shape)} int32")
    print(f"    = num_layers x num_tokens x ((max_ngram-1) * heads) = {num_layers} x {num_tokens} x {(max_ngram - 1) * heads}")
    print(f"  index range: min={int(indices.min())} max={int(indices.max())} (table spans 0 .. {int(vocab_sizes.sum()) - 1})")

    print("\n== deterministic addressing ==")
    again = engram_hash(token_ids, multipliers, vocab_sizes, offsets)
    print(f"  rerun on the same ids: identical = {bool(torch.equal(indices, again))}")
    print(f"  layer 0, token 0 indices: {indices[0, 0].tolist()}")

    print("\n== the number of gathered rows is set by the architecture ==")
    per_token = (max_ngram - 1) * heads
    for slots in (int(2.58e5), int(1.0e7)):
        table_mib = slots * 4 * 4 / 2**20  # fp32 rows of 4 dims, for scale only
        print(f"  table with {slots:>9,d} slots (~{table_mib:8.1f} MiB at 4 fp32 dims): rows gathered per token = {per_token}")
    print("  => a 40x larger table costs the same number of lookups; the paper's")
    print("     'O(1) lookup' claim is about exactly this (arXiv text: 14-.../raw/paper-deepseek-engram.md:27,71).")

    print("\n== image tokens: the kernel writes -1 ==")
    mask = torch.tensor([True, False, False, True, False, False])
    marked = indices.clone()
    marked[:, mask, :] = -1
    print(f"  mask                 {mask.tolist()}")
    print(f"  layer 0, token 0     {marked[0, 0].tolist()}  (all -1, no lookup)")
    print(f"  layer 0, token 1     {marked[0, 1].tolist()}  (real indices)")

    print("\n== the actual lookup is a plain gather ==")
    table = torch.randn(int(vocab_sizes.sum()), 4, dtype=torch.float32) * 0.02
    rows = table[indices[0, 0].long()]
    print(f"  table {tuple(table.shape)}; gathered {tuple(rows.shape)} -> concat into the memory vector e_t")
    print(f"  e_t per token = {(max_ngram - 1) * heads} rows x 4 dims = {(max_ngram - 1) * heads * 4} values")


if __name__ == "__main__":
    main()
