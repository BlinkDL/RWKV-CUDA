#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u,
                               F *__restrict__ const _y)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _h = (idx / N) % H;
    const int i = idx % N;

    const int _o0 = _b*T*C + _h*N;
    const int _o1 = _h*N;
    const float4 *__restrict__ const r = (float4 *)(_r + _o0);
    const float4 *__restrict__ const k = (float4 *)(_k + _o0);
    const float4 *__restrict__ const w = (float4 *)(_w + _o1);
    const float4 *__restrict__ const u = (float4 *)(_u + _o1);

    const F *__restrict__ const v = _v + _o0 + i;
    F *__restrict__ const y = _y + _o0 + i;

    __align__(16) float4 state[N/4] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f) };

    for (int _t = 0; _t < T; _t++)
    {
        const int tt = _t*C;
        const int ttt = _t*(C >> 2);
        const F vv = v[tt];
        F yy = 0;

        #pragma unroll
        for (int j = 0; j < N/4; j++)
        {
            const float4 rr = r[ttt + j];
            const float4 kk = k[ttt + j];
            const float4 ww = w[j];
            const float4 uu = u[j];

            float4 x;
            x.x = kk.x * vv;
            x.y = kk.y * vv;
            x.z = kk.z * vv;
            x.w = kk.w * vv;

            float4 s = state[j];
            yy += rr.x * (uu.x * x.x + s.x) + rr.y * (uu.y * x.y + s.y) + rr.z * (uu.z * x.z + s.z) + rr.w * (uu.w * x.w + s.w);

            float4* ss = state + j;
            ss->x = s.x * ww.x + x.x;
            ss->y = s.y * ww.y + x.y;
            ss->z = s.z * ww.z + x.z;
            ss->w = s.w * ww.w + x.w;
        }
        y[tt] = yy;
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
