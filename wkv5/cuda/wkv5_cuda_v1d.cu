#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*N;
    _u += h*N;

    __shared__ float state[N*N], rr[N], kk[N];

    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j*N + i] = 0;

    for (int _t = b*T*C + h*N + i; _t < (b+1)*T*C + h*N + i; _t += C)
    {        
        __syncthreads();
        rr[i] = _r[_t];
        kk[i] = _k[_t];
        __syncthreads();
        
        const float vv = _v[_t];
        float yy = 0;

        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            float x = kk[j] * vv;
            float s = state[j*N + i];

            yy += rr[j] * (_u[j] * x + s);
            state[j*N + i] = s * _w[j] + x;
        }
        _y[_t] = yy;
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

}
