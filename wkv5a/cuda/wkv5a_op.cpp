#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef float DTYPE;

void cuda_forward(int B, int T, int C, int H, DTYPE *r, DTYPE *k, DTYPE *v, float *w1, DTYPE *u1, float *w2, DTYPE *u2, DTYPE *y);
void cuda_backward(int B, int T, int C, int H, DTYPE *r, DTYPE *k, DTYPE *v, float *w1, float *ww1, DTYPE *u1, float *w2, float *ww2, DTYPE *u2, DTYPE *gy, DTYPE *gr, DTYPE *gk, DTYPE *gv, DTYPE *gw1, DTYPE *gu1, DTYPE *gw2, DTYPE *gu2);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w1, torch::Tensor &u1, torch::Tensor &w2, torch::Tensor &u2, torch::Tensor &y) {
    cuda_forward(B, T, C, H, r.data_ptr<DTYPE>(), k.data_ptr<DTYPE>(), v.data_ptr<DTYPE>(), w1.data_ptr<float>(), u1.data_ptr<DTYPE>(), w2.data_ptr<float>(), u2.data_ptr<DTYPE>(), y.data_ptr<DTYPE>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w1, torch::Tensor &ww1, torch::Tensor &u1, torch::Tensor &w2, torch::Tensor &ww2, torch::Tensor &u2,
        torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw1, torch::Tensor &gu1, torch::Tensor &gw2, torch::Tensor &gu2) {
    cuda_backward(B, T, C, H, r.data_ptr<DTYPE>(), k.data_ptr<DTYPE>(), v.data_ptr<DTYPE>(), w1.data_ptr<float>(), ww1.data_ptr<float>(), u1.data_ptr<DTYPE>(), w2.data_ptr<float>(), ww2.data_ptr<float>(), u2.data_ptr<DTYPE>(),
        gy.data_ptr<DTYPE>(), gr.data_ptr<DTYPE>(), gk.data_ptr<DTYPE>(), gv.data_ptr<DTYPE>(), gw1.data_ptr<DTYPE>(), gu1.data_ptr<DTYPE>(), gw2.data_ptr<DTYPE>(), gu2.data_ptr<DTYPE>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv5a forward");
    m.def("backward", &backward, "wkv5a backward");
}

TORCH_LIBRARY(wkv5a, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
