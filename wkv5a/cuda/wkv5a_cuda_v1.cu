#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef float DTYPE;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w1, const F *__restrict__ _u1, const float *__restrict__ _w2, const F *__restrict__ _u2,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w1 += h*_N_;
    _u1 += h*_N_;

    __shared__ float r[_N_], k[_N_], u__[_N_], w__[_N_];

    __syncthreads();
    w__[i] = _w1[i];
    u__[i] = float(_u1[i]);
    __syncthreads();

    float state[_N_] = {0};
    float u[_N_], w[_N_];

    #pragma unroll
    for (int j = 0; j < _N_; j++) {
        w[j] = w__[j] * _w2[h*_N_+i];
        u[j] = u__[j] + _u2[h*_N_+i];
    }

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = k[j] * v;

            float s = state[j];
            state[j] = s * w[j] + x;

            y += r[j] * (u[j] * x + s);
        }
        _y[t] = F(y);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w1, const float *__restrict__ __w1, const F *__restrict__ _u1, const float *__restrict__ _w2, const float *__restrict__ __w2, const F *__restrict__ _u2, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw1, F *__restrict__ const _gu1, F *__restrict__ const _gw2, F *__restrict__ const _gu2)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    _w1 += h*_N_;
    _u1 += h*_N_;
    __w1 += h*_N_;
    _w2 += h*_N_;
    _u2 += h*_N_;
    __w2 += h*_N_;
    const float w1 = _w1[i];
    const float u1 = float(_u1[i]);
    const float ww1 = __w1[i];
    const float w2 = _w2[i];
    const float u2 = float(_u2[i]);
    const float ww2 = __w2[i];

    __shared__ float v[_N_], r[_N_], k[_N_], gy[_N_];
    
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};
    float gu1 = 0, gu2 = 0, gw1 = 0, gw2 = 0;

    for (int t = b*T*C + h*_N_ + i, tend = (b+1)*T*C + h*_N_ + i; t < tend; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        k[i] = float(_k[t]);
        r[i] = float(_r[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        float ki = k[i];
        float vi = v[i];
        float ri = r[i];
        float gyi = gy[i];

        float gr = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = v[j] * ki;
            float x2 = vi * k[j];
            float s = state[j];
            state[j] = s * (w1*_w2[j]) + x;

            gr += gy[j] * ((u1+_u2[j]) * x + s);
            gu1 += ri * x * gy[j];
            gu2 += r[j] * x2 * gyi;
        }

        _gr[t] = F(gr);

        if (t < tend - 2*C)
        {
            __syncthreads();
            gy[i] = float(_gy[t + 2*C]);
            r[i] = float(_r[t + 2*C]);
            __syncthreads();

            float ri = r[i];
            float gyi = gy[i];
    
            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                float x = v[j] * ki;
                float x2 = vi * k[j];
                saaaa[j] = (w1*_w2[j]) * (saaaa[j] + sbbbb[j] + x);
                sbbbb[j] = (w1*_w2[j]) * (sbbbb[j] + x);
                scccc[j] = (_w1[j]*w2) * (scccc[j] + sdddd[j] + x2);
                sdddd[j] = (_w1[j]*w2) * (sdddd[j] + x2);
                
                gw1 += ri * ww1 * saaaa[j] * gy[j];
                gw2 += r[j] * ww2 * scccc[j] * gyi;
            }
        }
    }
    _gu1[b*C + h*_N_ + i] = F(gu1);
    _gu2[b*C + h*_N_ + i] = F(gu2);
    _gw1[b*C + h*_N_ + i] = F(gw1);
    _gw2[b*C + h*_N_ + i] = F(gw2);

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j] = 0;
    
    for (int t = (b+1)*T*C + h*_N_ + i - C, tend = b*T*C + h*_N_ + i; t >= tend; t -= C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = float(_r[t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gy[j] * rr;
            float s = state[j];
            state[j] = s * (w1*_w2[j]) + x;

            gk += v[j] * ((u1+_u2[j]) * x + s);
        }
        _gk[t] = F(gk);
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j] = 0;

    for (int t = (b+1)*T*C + h*_N_ + i - C, tend = b*T*C + h*_N_ + i; t >= tend; t -= C)
    {
        __syncthreads();
        k[i] = float(_k[t]);
        r[i] = float(_r[t]);
        __syncthreads();

        const float gy = float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gy * r[j];
            float s = state[j];
            state[j] = s * float(_w1[j]*w2) + x;

            gv += k[j] * (float(_u1[j]+u2) * x + s);
        }
        _gv[t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, DTYPE *r, DTYPE *k, DTYPE *v, float *w1, DTYPE *u1, float *w2, DTYPE *u2, DTYPE *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w1, u1, w2, u2, y);
}

void cuda_backward(int B, int T, int C, int H, DTYPE *r, DTYPE *k, DTYPE *v, float *w1, float *ww1, DTYPE *u1, float *w2, float *ww2, DTYPE *u2, DTYPE *gy, DTYPE *gr, DTYPE *gk, DTYPE *gv, DTYPE *gw1, DTYPE *gu1, DTYPE *gw2, DTYPE *gu2)
{
    assert(H*_N_ == C);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w1, ww1, u1, w2, ww2, u2, gy, gr, gk, gv, gw1, gu1, gw2, gu2);
}
