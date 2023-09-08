#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x - (blockIdx.x / H) * H;
    const int i = threadIdx.x;
    _w += h*N;
    _u += h*N;

    const float4 *__restrict__ const k = (float4 *)(_k);
    const float4 *__restrict__ const r = (float4 *)(_r);
    const float4 *__restrict__ const w = (float4 *)(_w);
    const float4 *__restrict__ const u = (float4 *)(_u);
    const F *__restrict__ const v = _v;
    F *__restrict__ const y = _y;

    __shared__ float state[N * N];
    __shared__ float4 rr[N >> 2], kk[N >> 2];

    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0.0f;

    for (int _tt = b*T*C + h*N + i, _tend = (b+1)*T*C + h*N + i; _tt < _tend; _tt += C)
    {
        const int _t = _tt >> 2;
        const F vv = v[_tt];
        F yy = 0.0;
        
        rr[i >> 2] = r[_t];
        kk[i >> 2] = k[_t];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < N >> 2; j++)
        {
            const int j4n = (j << 2) * N;

            const float4 ww = w[j];
            const float4 uu = u[j];
            const float4 rrr = rr[j];
            const float4 kkk = kk[j];

            float4 x;
            x.x = kkk.x * vv;
            x.y = kkk.y * vv;
            x.z = kkk.z * vv;
            x.w = kkk.w * vv;

            F &s0 = state[j4n + i];
            F &s1 = state[j4n + N + i];
            F &s2 = state[j4n + 2*N + i];
            F &s3 = state[j4n + 3*N + i];

            yy += rrr.x * (uu.x * x.x + s0) + rrr.y * (uu.y * x.y + s1) + rrr.z * (uu.z * x.z + s2) + rrr.w * (uu.w * x.w + s3);
            s0 = s0 * ww.x + x.x;
            s1 = s1 * ww.y + x.y;
            s2 = s2 * ww.z + x.z;
            s3 = s3 * ww.w + x.w;
        }
        y[_tt] = yy;
        // __syncthreads();
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
