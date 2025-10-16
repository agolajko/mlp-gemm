#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace
{

    __global__ void dummy_sgemm_kernel(const float *A, const float *B, float *C,
                                       int M, int N, int K)
    {
        // Naive (slow) reference to get plumbing working; replace with your fast kernel
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
            {
                acc += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }
    }

    __global__ void dummy_sgemm_bias_relu_kernel(const float *A, const float *B, const float *bias, float *C,
                                                 int M, int N, int K)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
            {
                acc += A[row * K + k] * B[k * N + col];
            }
            acc += bias[col];            // fused bias
            acc = acc > 0.f ? acc : 0.f; // fused ReLU
            C[row * N + col] = acc;      // single global store
        }
    }

} // namespace

torch::Tensor sgemm_forward(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "fp32 only");

    TORCH_CHECK(A.device() == B.device(), "A and B must be on same device");

    A = A.contiguous();
    B = B.contiguous();
    c10::cuda::CUDAGuard guard(A.device());

    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "K mismatch");

    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    dummy_sgemm_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}

torch::Tensor sgemm_bias_relu_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(A.device() == B.device() && A.device() == bias.device(), "All tensors must be on same device");

    A = A.contiguous();
    B = B.contiguous();
    bias = bias.contiguous();

    c10::cuda::CUDAGuard guard(A.device());

    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "K mismatch");

    TORCH_CHECK(bias.numel() == N, "bias must be length N");

    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    dummy_sgemm_bias_relu_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}
