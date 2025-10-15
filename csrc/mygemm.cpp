#include <torch/extension.h>

// Declarations implemented in .cu
torch::Tensor sgemm_forward(torch::Tensor A, torch::Tensor B);
torch::Tensor sgemm_bias_relu_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias);
torch::Tensor BankExtra(torch::Tensor A, torch::Tensor B);

TORCH_LIBRARY(mygemm, m)
{
    m.def("sgemm(Tensor A, Tensor B) -> Tensor");
    m.def("sgemm_bias_relu(Tensor A, Tensor B, Tensor bias) -> Tensor");
    m.def("bank_extra(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(mygemm, CUDA, m)
{
    m.impl("sgemm", &sgemm_forward);
    m.impl("sgemm_bias_relu", &sgemm_bias_relu_forward);
    m.impl("bank_extra", &BankExtra);
}
