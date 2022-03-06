#include <stdio.h>

// require T <= Tmax

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const w, const F *__restrict__ const k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T)
{
    const int i = blockIdx.y;
    const int t = threadIdx.x;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax];
    ww[t] = w[(i % C) * T + t];
    kk[t] = k[i * T + t];

    __syncthreads();

    F s = eps;
    const F *__restrict__ const www = ww + (T - 1) - t;
    for (int u = 0; u <= t; u++)
    {
        s += www[u] * kk[u];
    }
    x[i * T + t] = s;
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const w, const F *__restrict__ const k, const F *__restrict__ const gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T)
{
    const int i = blockIdx.y;
    const int t = threadIdx.x;

    __shared__ F gg[Tmax];
    __shared__ F kk[Tmax];
    __shared__ F ww[Tmax];
    gg[t] = gwk[i * T + t];
    kk[t] = k[i * T + t];
    ww[t] = w[(i % C) * T + t];

    __syncthreads();

    F s = 0;
    const F *__restrict__ const ggk = gg + (T - 1) - t;
    for (int u = 0; u <= t; u++)
    {
        s += ggk[u] * kk[u];
    }
    gw[i * T + t] = s;

    s = 0;
    const F *__restrict__ const ggw = gg + (T - 1) + t;
    for (int u = t; u < T; u++)
    {
        s += ggw[-u] * ww[u];
    }
    gk[i * T + t] = s;
}

void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T)
{
    dim3 gridDim(1, B * C);
    dim3 blockDim(T);
    kernel_forward<<<gridDim, blockDim>>>(w, k, x, eps, B, C, T);
}
void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T)
{
    dim3 gridDim(1, B * C);
    dim3 blockDim(T);
    kernel_backward<<<gridDim, blockDim>>>(w, k, gwk, gw, gk, B, C, T);
}
