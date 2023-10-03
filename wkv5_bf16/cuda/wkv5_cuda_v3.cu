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
    _w += h*_N_;
    _u += h*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_] = {0};

    __syncthreads();
    u[i] = float(_u[i]);
    w[i] = float(_w[i]);
    __syncthreads();

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

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const float *__restrict__ __w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw, F *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;
    __w += h*_N_;
    const float w = _w[i];
    const float u = float(_u[i]);
    const float ww = __w[i];

    __shared__ float v[_N_], r[_N_], k[_N_], gy[_N_], gy2[_N_], w_[_N_], u_[_N_];    
    float state[_N_*2] = {0};

    float gw = 0, gu = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;
    const int t222 = t111 - 2*C;

    for (int _t = t000; _t < t111; _t += C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy[i] = float(_gy[_t]);
        __syncthreads();

        const float k = float(_k[_t]);
        const float r = float(_r[_t]);
        
        float gr = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = v[j] * k;
            float s = state[j];
            state[j] = s * w + x;

            gr += gy[j] * (u * x + s);
            gu += r * x * gy[j];
        }
        _gr[_t] = F(gr);
    }
    _gu[b*C + h*_N_ + i] = F(gu);

    #pragma unroll
    for (int j = 0; j < _N_*2; ++j) {
        state[j] = 0;
    }
    
    for (int _t = t000; _t < t222; _t += C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy2[i] = float(_gy[_t + 2*C]);
        __syncthreads();
        const float r2 = float(_r[_t + 2*C]);
        const float k = float(_k[_t]);

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = v[j] * k;
            // accum[j] = w[h,i] * (accum[j] + accum[j+N] + x)
            // accum[j+N] = w[h,i] * (accum[j+N] + x)
            // gw[h,i] += r[b,t+2,h,i] * _w[h,i] * accum[j] * gy[b,t+2,h,j]
            state[j] = w * (state[j] + state[j+_N_] + x);
            state[j+_N_] = w * (state[j+_N_] + x);
            gw += r2 * ww * state[j] * gy2[j];
        }
    }
    
    _gw[b*C + h*_N_ + i] = F(gw);

    #pragma unroll
    for (int j = 0; j < _N_; ++j) {
        state[j] = 0;
    }

    __syncthreads();
    w_[i] = float(_w[i]);
    u_[i] = float(_u[i]);
    __syncthreads();
    
    for (int _t = t111 - C; _t >= t000; _t -= C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy[i] = float(_gy[_t]);
        k[i] = float(_k[_t]);
        r[i] = float(_r[_t]);
        __syncthreads();

        float gk = 0, x, s;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            x = gy[j] * r[i];
            s = state[j];
            state[j] = s * w + x;
            gk += v[j] * (u * x + s);
        }
        _gk[_t] = F(gk);
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j) {
        state[j] = 0;
    }

    for (int _t = t111 - C; _t >= t000; _t -= C)
    {
        __syncthreads();
        gy[i] = float(_gy[_t]);
        r[i] = float(_r[_t]);
        k[i] = float(_k[_t]);
        __syncthreads();

        float gv = 0, x, s;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            x = gy[i] * r[j];
            s = state[j];
            state[j] = s * w_[j] + x;
            gv += k[j] * (u_[j] * x + s);
        }
        _gv[_t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*_N_ == C);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
