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

        for (int _j = 0; _j < N; _j++) 
        {
            const int j = _t + _j;
            const int m = _o1 + _j;

            const float x = k[j] * vv;
            const float s = state[_j];
            
            atomicAdd(y + _t, r[j] * (_u[m] * x + s));
            state[_j] = s * _w[m] + x;
        }
    }
}

template <typename F>
__global__ void kernel_backward (const int B, const int T, const int C, const int H,
    const F *__restrict__ const r, const F *__restrict__ const k, const F *__restrict__ const v, const F *__restrict__ w, const F *__restrict__ const wwww, const F *__restrict__ u, const F *__restrict__ const gy,
    F *__restrict__ const gr, F *__restrict__ const gk, F *__restrict__ const gv, F *__restrict__ const gw, F *__restrict__ gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    w += h*N;
    u += h*N;
    gu += h*N;

    __shared__ float state[N * N], vv[N], gyy[N];

    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;
    
    const float ww = w[i];
    const float uu = u[i];

    for (int _t = b*T*C + h*N + i, _tend = (b+1)*T*C + h*N + i; _t < _tend; _t += C)
    {
        const F kk = k[_t];
        const F rr = r[_t];
        F grr = 0;
        F guu = 0;

        vv[i] = v[_t];
        gyy[i] = gy[_t];

        __syncthreads();

        for (int j = 0; j < N; j++)
        {

            float x = vv[j] * kk;
            float s = state[j * N + i];

            grr += gyy[j] * (uu * x + s);
            state[j * N + i] = s * ww + x;
            guu += rr * x * gyy[j];
        }
        gr[_t] = grr;
        atomicAdd(gu + i, guu);

        __syncthreads();
    }
}

void cuda_forward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *y)
{
    dim3 threadsPerBlock( min(B*C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *ww, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    assert(H*N == C);
    kernel_backward<<<dim3(B * H), dim3(N)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
