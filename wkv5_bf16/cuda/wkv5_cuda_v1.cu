#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*N;
    _u += h*N;

    __shared__ float state[N * N];
    __shared__ F rr[N], kk[N];

    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;

    for (int _t = b*T*C + h*N + i, _tend = (b+1)*T*C + h*N + i; _t < _tend; _t += C)
    {
        __syncthreads();
        rr[i] = _r[_t];
        kk[i] = _k[_t];
        __syncthreads();

        const F vv = _v[_t];
        F yy = 0;

        for (int j = 0; j < N; j++)
        {
            F x = kk[j] * vv;

            float s = state[j * N + i];
            state[j * N + i] = s * _w[j] + float(x);

            yy += rr[j] * (_u[j] * x + F(s));
        }
        _y[_t] = yy;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const r, const F *__restrict__ const k, const F *__restrict__ const v, const float *__restrict__ w, const float *__restrict__ wwww, const F *__restrict__ u, const F *__restrict__ const gy,
    F *__restrict__ const gr, F *__restrict__ const gk, F *__restrict__ const gv, F *__restrict__ gw, F *__restrict__ gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    w += h*N;
    u += h*N;
    gu += h*N;
    gw += h*N;
    wwww += h*N;

    __shared__ float state[N * N];
    __shared__ F vv[N], rr[N], kk[N], gyy[N];

    #pragma unroll
    for (int j = 0; j < N; ++j){
        state[j * N + i] = 0;
    }
    
    const float ww = w[i];
    const F uu = u[i];
    const float wwwww = wwww[i];
    float saaaa[N] = {0.0f}, sbbbb[N] = {0.0f};

    for (int _t = b*T*C + h*N + i, _tend = (b+1)*T*C + h*N + i; _t < _tend; _t += C)
    {
        __syncthreads();
        vv[i] = v[_t];
        gyy[i] = gy[_t];
        __syncthreads();

        const F kk = k[_t];
        const F rr = r[_t];
        F grr = 0;
        F guu = 0;

        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            F x = vv[j] * kk;
            float s = state[j * N + i];
            state[j * N + i] = s * ww + float(x);

            grr += gyy[j] * (uu * x + F(s));
            guu += rr * x * gyy[j];
        }

        gr[_t] = grr;
        atomicAdd(gu + i, guu);

        if (_t < _tend - 2 * C){

            __syncthreads();
            gyy[i] = gy[_t+2*C];
            __syncthreads();

            const F rr_value = r[_t+2*C];

            #pragma unroll
            for (int j = 0; j < N; j++){
                F x = vv[j] * kk;
                saaaa[j] = ww * (saaaa[j] + sbbbb[j] + float(x));
                sbbbb[j] = ww * (sbbbb[j] + float(x));
                
                atomicAdd(gw+i, rr_value * wwwww * F(saaaa[j]) * gyy[j]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;
    
    for (int _t = (b+1)*T*C + h*N + i - C, _tend = b*T*C + h*N + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        vv[i] = v[_t];
        gyy[i] = gy[_t];
        __syncthreads();

        const F rr = r[_t];
        F gkk = 0;

        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            F x = gyy[j] * rr;
            float s = state[j * N + i];
            state[j * N + i] = s * ww + float(x);

            gkk += vv[j] * (uu * x + F(s));
        }
        gk[_t] = gkk;
    }

    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;

    for (int _t = (b+1)*T*C + h*N + i - C, _tend = b*T*C + h*N + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        kk[i] = k[_t];
        rr[i] = r[_t];
        __syncthreads();

        const F gy_value = gy[_t];
        F gvv = 0;

        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            F x = gy_value * rr[j];
            float s = state[j * N + i];
            state[j * N + i] = s * w[j] + float(x);

            gvv += kk[j] * (u[j] * x + F(s));
        }
        gv[_t] = gvv;
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*N == C);
    kernel_forward<<<dim3(B * H), dim3(N)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*N == C);
    kernel_backward<<<dim3(B * H), dim3(N)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
