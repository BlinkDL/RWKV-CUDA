#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    const int _o0 = b*T*C + h*N;
    const int _o1 = h*N;
    const float *__restrict__ const r = (float *)(_r + _o0);
    const float *__restrict__ const k = (float *)(_k + _o0);
    const float *__restrict__ const w = (float *)(_w + _o1);
    const float *__restrict__ const u = (float *)(_u + _o1);

    const F *__restrict__ const v = _v + _o0 + i;
    F *__restrict__ const y = _y + _o0 + i;

    __shared__ float state[N * N], rr[N], kk[N];

    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;

    for (int _t = 0; _t < T; _t++)
    {
        const int tt = _t*C;
        const F vv = v[tt];
        F yy = 0;

        rr[i] = r[tt + i];
        kk[i] = k[tt + i];

        for (int j = 0; j < N; j++)
        {
            const float ww = w[j];
            const float uu = u[j];

            float x = kk[j] * vv;

            float s = state[j];
            yy += rr[j] * (uu * x + s);
            state[j] = s * ww + x;
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
    kernel_forward<<<dim3(B * H), dim3(N)>>>(B, T, C, H, r, k, v, w, u, y);
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
