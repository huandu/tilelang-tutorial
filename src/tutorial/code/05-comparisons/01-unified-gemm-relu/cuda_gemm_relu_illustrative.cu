// Illustrative CUDA skeleton for C = relu(A @ B).  NOT compilable as-is.
//
// This is a teaching sketch, not a real kernel. Its structure follows the
// shared-memory tiled matmul example in the CUDA C++ Best Practices Guide:
//   __shared__ float aTile[TILE_DIM][TILE_DIM];
// mirrored at
//   ../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:1745-1830
//
// The point (chapter section 3.1) is what belongs to the *user* here:
//   * the block/thread mapping (blockIdx.y, threadIdx.y, ...);
//   * placing tiles in __shared__ memory;
//   * barrier placement (__syncthreads()).
// In TileLang, the last two are the compiler's job.
//
// Verification level: static only; never compiled or run.

#include <cuda_fp16.h>

#define TILE_DIM 32          // must match the block shape below
#define BM 128               // chapter's tile M
#define BN 128               // chapter's tile N
#define BK 32                // chapter's tile K

__global__ void gemm_relu_kernel(const __half* A, const __half* B, __half* C,
                                 int M, int N, int K) {
    // User-chosen mapping: one thread per output element of the tile.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    // User-managed on-chip storage.
    __shared__ __half aTile[TILE_DIM][TILE_DIM];
    __shared__ __half bTile[TILE_DIM][TILE_DIM];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE_DIM) {
        // Cooperative load into shared memory (coalescing is the user's concern).
        aTile[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
        bTile[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];

        // User-managed synchronization.
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            acc += __half2float(aTile[threadIdx.y][k]) * __half2float(bTile[k][threadIdx.x]);
        }

        __syncthreads();
    }

    // Fused epilogue: the user puts relu here, before the store.
    acc = acc > 0.0f ? acc : 0.0f;
    C[row * N + col] = __float2half(acc);
}

// Note: this sketch deliberately uses a naive inner loop (no tensor cores).
// A production version would issue mma / wgmma / tcgen05 instructions, which is
// exactly the level at which CUTLASS-CuTe and TileLang's T.gemm do the work.
