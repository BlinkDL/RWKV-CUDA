#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u,
                               F *__restrict__ const _y)
{
    const int N = C / H;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / H;
    const int _h = idx % H;
    const int _o0 = _b*T*H*N + _h*N;
    const int _o1 = _h*N;

    const F *__restrict__ const k = _k + _o0;
    const F *__restrict__ const v = _v + _o0;
    const F *__restrict__ const r = _r + _o0;
    F *__restrict__ const y = _y + _o0;

    float state[NN] = {0};
    for (int _t = 0; _t < T; _t++) {
        for (int _i = 0; _i < N; _i++) 
        {    
            const int i = _t*H*N + _i;
            const F vv = v[i];

            for (int _j = 0; _j < N; _j++) 
            {
                const int j = _t*H*N + _j;
                const int m = _o1 + _j;
                const int ij = _i*N + _j;

                const float x = k[j] * vv;
                const float s = state[ij];
                
                y[i] += r[j] * (_u[m] * x + s);
                state[ij] = s * _w[m] + x;
            }
        }
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
    dim3 threadsPerBlock( min(B*H, 32) );
    assert(B * H % threadsPerBlock.x == 0);
    dim3 numBlocks(B * H / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    dim3 threadsPerBlock( min(B*H, 32) );
    assert(B * H % threadsPerBlock.x == 0);
    dim3 numBlocks(B * H / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}
