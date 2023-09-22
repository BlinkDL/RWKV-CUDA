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
    const float w2 = _w2[h*_N_+i];
    const float u2 = float(_u2[h*_N_+i]);

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];

    __syncthreads();
    w[i] = _w1[i];
    u[i] = float(_u1[i]);
    __syncthreads();

    float state[_N_] = {0};

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;
    
            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;
    
            y += r_.x * ((u2+u_.x) * x.x + s.x);
            y += r_.y * ((u2+u_.y) * x.y + s.y);
            y += r_.z * ((u2+u_.z) * x.z + s.z);
            y += r_.w * ((u2+u_.w) * x.w + s.w);
    
            s.x = s.x * (w2*w_.x) + x.x;
            s.y = s.y * (w2*w_.y) + x.y;
            s.z = s.z * (w2*w_.z) + x.z;
            s.w = s.w * (w2*w_.w) + x.w;
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

    __shared__ float w1_[_N_], u1_[_N_], w2_[_N_], u2_[_N_];
    __syncthreads();
    w1_[i] = _w1[i];
    u1_[i] = float(_u1[i]);
    w2_[i] = _w2[i];
    u2_[i] = float(_u2[i]);
    __syncthreads();
    
    const float w1 = w1_[i];
    const float u1 = u1_[i];
    const float ww1 = __w1[i];
    const float w2 = w2_[i];
    const float u2 = u2_[i];
    const float ww2 = __w2[i];

    __shared__ float v[_N_], r[_N_], k[_N_], gy[_N_];
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    float gu1 = 0, gu2 = 0, gw1 = 0, gw2 = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;
    const int t222 = t111 - 2*C;

    for (int t = t000; t < t111; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float ki = k[i];
        const float vi = v[i];
        float gr = 0, gu1_ = 0, gu2_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = ki * v[j];
            float x2 = vi * k[j];

            gr += ((u1+u2_[j]) * x + s) * gy[j];
            gu1_ += x * gy[j];
            gu2_ += x2 * r[j];

            s = s * (w1*w2_[j]) + x;
        }
        _gr[t] = F(gr);
        gu1 += gu1_ * r[i];
        gu2 += gu2_ * gy[i];
    }
    _gu1[b*C + h*_N_ + i] = F(gu1);
    _gu2[b*C + h*_N_ + i] = F(gu2);

    for (int t = t000; t < t222; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t + 2*C]);
        k[i] = float(_k[t]);
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t + 2*C]);
        __syncthreads();

        const float ki = k[i];
        const float vi = v[i];
        const float ri = r[i];
        const float gyi = gy[i];

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& sa = saaaa[j];
            float& sb = sbbbb[j];
            float& sc = scccc[j];
            float& sd = sdddd[j];
            float x = ki * v[j];
            float x2 = vi * k[j];

            float tmp = (w1*w2_[j]) * (sa + x);
            sa = tmp;
            sb = tmp + (w1*w2_[j]) * sb;
            tmp = (w1_[j]*w2) * (sc + x2);
            sc = tmp;
            sd = tmp + (w1_[j]*w2) * sd;
            
            gw1 += ri * sb * gy[j];
            gw2 += gyi * sd * r[j];
        }
    }
    _gw1[b*C + h*_N_ + i] = F(ww1 * gw1);
    _gw2[b*C + h*_N_ + i] = F(ww2 * gw2);

    #pragma unroll
    for (int j = 0; j < _N_; ++j) {
        saaaa[j] = 0;
        sbbbb[j] = 0;
    }

    for (int t = t111 - C; t >= t000; t -= C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = r[i];
        const float gyy = gy[i];
        float gk = 0, gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float& s2 = sbbbb[j];
            float x = rr * gy[j];
            float x2 = gyy * r[j];

            gk += ((u1+u2_[j]) * x + s) * v[j];
            gv += ((u2+u1_[j]) * x2 + s2) * k[j];
            s = x + s * (w1*w2_[j]);
            s2 = x2 + s2 * (w2*w1_[j]);
        }
        _gk[t] = F(gk);
        _gv[t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, DTYPE *r, DTYPE *k, DTYPE *v, float *w1, DTYPE *u1, float *w2, DTYPE *u2, DTYPE *y)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w1, u1, w2, u2, y);
}

void cuda_backward(int B, int T, int C, int H, DTYPE *r, DTYPE *k, DTYPE *v, float *w1, float *ww1, DTYPE *u1, float *w2, float *ww2, DTYPE *u2, DTYPE *gy, DTYPE *gr, DTYPE *gk, DTYPE *gv, DTYPE *gw1, DTYPE *gu1, DTYPE *gw2, DTYPE *gu2)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w1, ww1, u1, w2, ww2, u2, gy, gr, gk, gv, gw1, gu1, gw2, gu2);
}
