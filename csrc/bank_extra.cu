// from https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/8_kernel_bank_extra_col.cuh

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankExtraCol(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    const int extraCols = 5;
    __shared__ float Bs[BK * (BN + extraCols)];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate the SMEM caches
        // transpose A while loading it
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
        Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // block into registers
            for (uint i = 0; i < TM; ++i)
            {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i)
            {
                regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            // perform GEMM update in reg
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
            // write back
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
    }
}

torch::Tensor bank_extra(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "fp32 only");

    A = A.contiguous();
    B = B.contiguous();

    c10::cuda::CUDAGuard guard(A.device());

    const auto M = A.size(0);
    const auto K = A.size(1);
    auto N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "K mismatch");

    auto C = torch::empty({M, N}, A.options());

    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    dim3 blockDim((BM * BN) / (TM * TN)); // 256 threads in 1D

    // Grid dimensions: one block per output tile
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim, 0, stream>>>(M, N, K, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}