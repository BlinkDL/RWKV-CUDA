#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                                      const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u,
                                      F *__restrict__ const _y)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _h = (idx / N) - ((idx / N) /  H) * H;
    const int _i = idx - (idx / N) * N;

    const int _o0 = _b * T * C + _h * N;
    const int _o1 = _h * N;

    const float4 *__restrict__ const k = (float4 *)(_k + _o0);
    const float4 *__restrict__ const r = (float4 *)(_r + _o0);
    const float4 *__restrict__ const w = (float4 *)(_w + _o1);
    const float4 *__restrict__ const u = (float4 *)(_u + _o1);
    const F *__restrict__ const v = _v + _o0 + _i;
    F *__restrict__ const y = _y + _o0 + _i;

    __align__(16) float4 state[N >> 2] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f) };

    for (int __t = 0; __t < T; __t++)
    {
        const int _t = __t * (C >> 2);
        const int tt = __t * C;
        const F vv = v[tt];
        float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int _j = 0; _j < N >> 2; _j++)
        {
            const int j = _t + _j;

            const float4 k_val = k[j];
            const float4 r_val = r[j];
            float4 x;
            x.x = k_val.x * vv;
            x.y = k_val.y * vv;
            x.z = k_val.z * vv;
            x.w = k_val.w * vv;

            float4 s = state[_j];

            result.x += r_val.x * (u[_j].x * x.x + s.x);
            result.y += r_val.y * (u[_j].y * x.y + s.y);
            result.z += r_val.z * (u[_j].z * x.z + s.z);
            result.w += r_val.w * (u[_j].w * x.w + s.w);

            state[_j].x = s.x * w[_j].x + x.x;
            state[_j].y = s.y * w[_j].y + x.y;
            state[_j].z = s.z * w[_j].z + x.z;
            state[_j].w = s.w * w[_j].w + x.w;
        }

        y[tt] += result.x + result.y + result.z + result.w;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
                                const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _gy,
                                F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw, F *__restrict__ const _gu)
{
    
}

void cuda_forward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *y)
{
    assert(H*N == C);
    const int SIZE = B*C;
    dim3 threadsPerBlock(min(SIZE, 32));
    assert(SIZE % threadsPerBlock.x == 0);
    dim3 numBlocks(SIZE / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    assert(H*N == C);
    const int SIZE = B*C;
    dim3 threadsPerBlock(min(SIZE, 32));
    assert(SIZE % threadsPerBlock.x == 0);
    dim3 numBlocks(SIZE / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}
