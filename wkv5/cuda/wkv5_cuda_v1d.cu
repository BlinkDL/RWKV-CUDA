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
    const float4 *__restrict__ const ww = (float4 *)(_w + h*N);
    const float4 *__restrict__ const uu = (float4 *)(_u + h*N);

    __shared__ float state[N*N], rr[N], kk[N];

    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j*N + i] = 0; // will __syncthreads soon

    for (int bthi = b*T*H*N + 0*H*N + h*N + i; bthi < b*T*H*N + T*H*N + h*N + i; bthi += C)
    {
        __syncthreads();
        rr[i] = _r[bthi]; // rr[0:N] = _r[b,t,h,0:N]
        kk[i] = _k[bthi]; // kk[0:N] = _r[b,t,h,0:N]
        __syncthreads();
        
        const float v = _v[bthi];
        float y = 0;

        const float4 *__restrict__ const rrr = (float4 *)(rr);
        const float4 *__restrict__ const kkk = (float4 *)(kk);
        float4 x, s;

        #pragma unroll
        for (int j = 0; j < N/4; ++j)
        {
            const float4 r = rrr[j];
            const float4 k = kkk[j];
            const float4 w = ww[j];
            const float4 u = uu[j];
            
            x.x = k.x * v;
            x.y = k.y * v;
            x.z = k.z * v;
            x.w = k.w * v;

            const int jj = (j<<2)*N + i;
            s.x = state[jj + 0*N];
            s.y = state[jj + 1*N];
            s.z = state[jj + 2*N];
            s.w = state[jj + 3*N];

            y += r.x * (u.x * x.x + s.x);
            y += r.y * (u.y * x.y + s.y);
            y += r.z * (u.z * x.z + s.z);
            y += r.w * (u.w * x.w + s.w);

            state[jj + 0*N] = s.x * w.x + x.x;
            state[jj + 1*N] = s.y * w.y + x.y;
            state[jj + 2*N] = s.z * w.z + x.z;
            state[jj + 3*N] = s.w * w.w + x.w;
        }
        _y[bthi] = y;
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
    assert(N%4 == 0);
    kernel_forward<<<dim3(B*H), dim3(N)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{

}
