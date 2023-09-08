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
        state[j*N + i] = 0; // will __syncthreads soon

    for (int ti = b*T*C + 0*C + h*N+i; ti < b*T*C + T*C + h*N+i; ti += C)
    {
        __syncthreads();
        rr[i] = _r[ti]; // fill rr[0..N]
        kk[i] = _k[ti]; // fill kk[0..N]
        __syncthreads();
        
        const float vv = _v[ti];
        float yy = 0;

        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            float x = kk[j] * vv;
            float s = state[j*N + i]; // much faster than i*N+j

            yy += rr[j] * (_u[j] * x + s);
            state[j*N + i] = s * _w[j] + x;
        }
        _y[ti] = yy;
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
    kernel_forward<<<dim3(B*H), dim3(N)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{

}