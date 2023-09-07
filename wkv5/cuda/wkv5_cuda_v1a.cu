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
    const int _i = idx % N;

    const int _o0 = _b*T*C + _h*N;
    const int _o1 = _h*N;
    const F *__restrict__ const k = _k + _o0;
    const F *__restrict__ const v = _v + _o0 + _i;
    const F *__restrict__ const r = _r + _o0;
    F *__restrict__ const y = _y + _o0 + _i;

    float state[N] = {0};

    for (int __t = 0; __t < T; __t++)
    {
        const int _t = __t*C;
        const F vv = v[_t];
        F yy = 0;

        #pragma unroll
        for (int _j = 0; _j < N; _j++) 
        {
            const int j = _t + _j;
            const int m = _o1 + _j;

            const float x = k[j] * vv;
            const float s = state[_j];
            
            yy += r[j] * (_u[m] * x + s);
            state[_j] = s * _w[m] + x;
        }
        y[_t] = yy;
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
