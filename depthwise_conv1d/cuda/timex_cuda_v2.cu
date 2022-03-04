#include <stdio.h>

// require T % 4 == 0 and T <= 1024

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const w, const F *__restrict__ const k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int tt = threadIdx.x;
    const int t = tt << 2;

    __shared__ F wk[2048];
    ((float4 *)wk)[tt] = ((float4 *)w)[(i % C) * (T >> 2) + tt];
    ((float4 *)wk)[256 + tt] = ((float4 *)k)[i * (T >> 2) + tt];
    __syncthreads();

    float4 s = {eps, eps, eps, eps};

    const F *__restrict__ const ww = wk + T - t - 4;
    const F *__restrict__ const kk = wk + 1024;
    for (int u = 0; u <= t; u++) {
        F x = kk[u];
        s.x += ww[u + 3] * x;
        s.y += ww[u + 2] * x;
        s.z += ww[u + 1] * x;
        s.w += ww[u + 0] * x;
    }
    s.y += ww[t + 3] * kk[t + 1];
    s.z += ww[t + 2] * kk[t + 1];
    s.z += ww[t + 3] * kk[t + 2];
    s.w += ww[t + 1] * kk[t + 1];
    s.w += ww[t + 2] * kk[t + 2];
    s.w += ww[t + 3] * kk[t + 3];

    ((float4 *)x)[i * (T >> 2) + tt] = s;
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const w, const F *__restrict__ const k, const F *__restrict__ const gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int tt = threadIdx.x;
    const int t = tt << 2;

    __shared__ F ww[1024];
    __shared__ F kk[1024];
    __shared__ F gg[1024];
    ((float4 *)ww)[tt] = ((float4 *)w)[(i % C) * (T >> 2) + tt];
    ((float4 *)kk)[tt] = ((float4 *)k)[i * (T >> 2) + tt];
    ((float4 *)gg)[tt] = ((float4 *)gwk)[i * (T >> 2) + tt];
    __syncthreads();

    float4 s = {0, 0, 0, 0};
    const F *__restrict__ const ggk = gg + T - t - 4;

    for (int u = 0; u <= t; u++) {
        F x = kk[u];
        s.x += ggk[u + 3] * x;
        s.y += ggk[u + 2] * x;
        s.z += ggk[u + 1] * x;
        s.w += ggk[u] * x;
    }
    s.y += ggk[t + 3] * kk[t + 1];
    s.z += ggk[t + 2] * kk[t + 1];
    s.z += ggk[t + 3] * kk[t + 2];
    s.w += ggk[t + 1] * kk[t + 1];
    s.w += ggk[t + 2] * kk[t + 2];
    s.w += ggk[t + 3] * kk[t + 3];

    ((float4 *)gw)[i * (T >> 2) + tt] = s;

    s.x = 0;
    s.y = 0;
    s.z = 0;
    s.w = 0;
    const F *__restrict__ const ggw = gg + T + t - 3;

    for (int u = t + 3; u < T; u++) {
        F x = ww[u];
        s.x += ggw[2 - u] * x;
        s.y += ggw[3 - u] * x;
        s.z += ggw[4 - u] * x;
        s.w += ggw[5 - u] * x;
    }
    s.x += ggw[2 - t] * ww[t + 0];
    s.x += ggw[1 - t] * ww[t + 1];
    s.x += ggw[0 - t] * ww[t + 2];
    s.y += ggw[2 - t] * ww[t + 1];
    s.y += ggw[1 - t] * ww[t + 2];
    s.z += ggw[2 - t] * ww[t + 2];

    ((float4 *)gk)[i * (T >> 2) + tt] = s;
}

void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T) {
    dim3 gridDim(1, B * C);
    dim3 blockDim(T >> 2);
    kernel_forward<<<gridDim, blockDim>>>(w, k, x, eps, B, C, T);
}
void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T) {
    dim3 gridDim(1, B * C);
    dim3 blockDim(T >> 2);
    kernel_backward<<<gridDim, blockDim>>>(w, k, gwk, gw, gk, B, C, T);
}
