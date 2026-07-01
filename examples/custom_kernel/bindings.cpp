#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declarations matching kernel.cu launch wrappers
void scale_launch(const __half*, const __half*, __half*, int, cudaStream_t);
void bias_launch(const __half*, const __half*, __half*, int, cudaStream_t);
void scale_bias_fused_launch(const __half*, const __half*, const __half*, __half*, int, cudaStream_t);

static __half* ptr(torch::Tensor t) {
    return reinterpret_cast<__half*>(t.data_ptr<at::Half>());
}

torch::Tensor scale(torch::Tensor x, torch::Tensor w) {
    auto out = torch::empty_like(x);
    scale_launch(ptr(x), ptr(w), ptr(out), x.numel(), at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor bias(torch::Tensor x, torch::Tensor b) {
    auto out = torch::empty_like(x);
    bias_launch(ptr(x), ptr(b), ptr(out), x.numel(), at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor scale_bias_fused(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    auto out = torch::empty_like(x);
    scale_bias_fused_launch(ptr(x), ptr(w), ptr(b), ptr(out), x.numel(),
                             at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale",            &scale,            "elementwise scale (fp16)");
    m.def("bias",             &bias,             "elementwise bias add (fp16)");
    m.def("scale_bias_fused", &scale_bias_fused, "fused scale+bias (fp16)");
}
