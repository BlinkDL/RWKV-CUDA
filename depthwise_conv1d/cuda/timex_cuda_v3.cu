#include <stdio.h>

// require T <= Tmax, T % 4 == 0, B % BF == 0, B % BB === 0 (Tmax and BF and BB are passed by compiler)

#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const __w, const F *__restrict__ const __k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;
    const int ti = t + T * i;
    const int tj = T * (B * C) / BF;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax * BF];
    F4(ww, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BF; j++) {
        F4(kk, t + Tmax * j) = F4(__k, ti + tj * j);
    }
    __syncthreads();

    float4 ss[BF];
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        ss[j] = {eps, eps, eps, eps};
    }
    for (int u = 0; u <= t; u++) {
        const F *__restrict__ const w = ww + T - t + u - 4;
        #pragma unroll
        for (int j = 0; j < BF; j++) {
            float4 *__restrict__ const s = ss + j;
            const F k = kk[u + Tmax * j];
            s->x += w[3] * k;
            s->y += w[2] * k;
            s->z += w[1] * k;
            s->w += w[0] * k;
        }
    }
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const w = ww + T - 3;
        const F *__restrict__ const k = kk + Tmax * j + t + 1;
        s->y += w[2] * k[0];
        s->z += w[1] * k[0];
        s->z += w[2] * k[1];
        s->w += w[0] * k[0];
        s->w += w[1] * k[1];
        s->w += w[2] * k[2];
        F4(x, ti + tj * j) = *s;
    }
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const __w, const F *__restrict__ const __k, const F *__restrict__ const __gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;
    const int ti = t + T * i;
    const int tj = T * (B * C) / BB;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax * BB];
    __shared__ F gg[Tmax * BB];
    F4(ww, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        F4(kk, t + Tmax * j) = F4(__k, ti + tj * j);
        F4(gg, t + Tmax * j) = F4(__gwk, ti + tj * j);
    }
    __syncthreads();

    float4 ss[BB];
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        ss[j] = {0, 0, 0, 0};
    }
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            float4 *__restrict__ const s = ss + j;
            const F *__restrict__ const g = gg + Tmax * j + T - t + u - 4;
            const F k = kk[u + Tmax * j];
            s->x += g[3] * k;
            s->y += g[2] * k;
            s->z += g[1] * k;
            s->w += g[0] * k;
        }
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const k = kk + Tmax * j + t + 1;
        const F *__restrict__ const g = gg + Tmax * j + T - 3;
        s->y += g[2] * k[0];
        s->z += g[1] * k[0];
        s->z += g[2] * k[1];
        s->w += g[0] * k[0];
        s->w += g[1] * k[1];
        s->w += g[2] * k[2];
        F4(gw, ti + tj * j) = *s;
    }

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        ss[j] = {0, 0, 0, 0};
    }
    for (int u = t + 3; u < T; u++) {
        const F w = ww[u];
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            float4 *__restrict__ const s = ss + j;
            const F *__restrict__ const g = gg + Tmax * j + T + t - u - 1;
            s->x += g[0] * w;
            s->y += g[1] * w;
            s->z += g[2] * w;
            s->w += g[3] * w;
        }        
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const g = gg + Tmax * j + T - 3;
        const F *__restrict__ const w = ww + t;
        s->x += g[2] * w[0];
        s->x += g[1] * w[1];
        s->x += g[0] * w[2];
        s->y += g[2] * w[1];
        s->y += g[1] * w[2];
        s->z += g[2] * w[2];
        F4(gk, ti + tj * j) = *s;
    }
}

void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T) {
    dim3 gridDim(1, B * C / BF);
    dim3 blockDim(T >> 2);
    kernel_forward<<<gridDim, blockDim>>>(w, k, x, eps, B, C, T);
}

void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T) {
    dim3 gridDim(1, B * C / BB);
    dim3 blockDim(T >> 2);
    kernel_backward<<<gridDim, blockDim>>>(w, k, gwk, gw, gk, B, C, T);
}
