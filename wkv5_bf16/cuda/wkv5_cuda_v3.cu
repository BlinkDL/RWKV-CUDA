#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template<typename T>
struct alignas(8 * sizeof(T)) Pack8X {
  T x, y, z, w, a, b, c, d;
};

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               F *__restrict__ _r, F *__restrict__ _k, const F *__restrict__ const _v, const float *__restrict__ _w, F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x - (blockIdx.x / H) * H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;

    Pack8X<F>* k = reinterpret_cast<Pack8X<F>*>(_k);
    Pack8X<F>* r = reinterpret_cast<Pack8X<F>*>(_r);
    const float4 *__restrict__ const w = (float4 *)(_w);
    Pack8X<F>* u = reinterpret_cast<Pack8X<F>*>(_u);
    const F *__restrict__ const v = _v;
    F *__restrict__ const y = _y;

    __align__(16) float4 state[_N_ >> 2] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f) };
    for ( int  i =0 ;i < _N_ >> 2; i++){
        state[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __shared__ Pack8X<F> rr[_N_ >> 3], kk[_N_ >> 3];

    for (int _tt = b*T*C + h*_N_ + i, _tend = (b+1)*T*C + h*_N_ + i; _tt < _tend; _tt += C)
    {
        const int _t = _tt >> 3;
        const float vv = float(v[_tt]);
        float yy = 0.0;
        
        __syncthreads();
        rr[i >> 3] = r[_t];
        kk[i >> 3] = k[_t];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < _N_ >> 3; j++)
        {
            const Pack8X<F> uu = u[j];
            const Pack8X<F> rrr = rr[j];
            const Pack8X<F> kkk = kk[j];

            float4 x1;
            x1.x = float(kkk.x) * vv;
            x1.y = float(kkk.y) * vv;
            x1.z = float(kkk.z) * vv;
            x1.w = float(kkk.w) * vv;
            float4 &s1 = state[j<<1];
            const float4 ww1 = w[j<<1];

            yy += float(rrr.x) * (float(uu.x) * x1.x + s1.x);
            yy += float(rrr.y) * (float(uu.y) * x1.y + s1.y);
            yy += float(rrr.z) * (float(uu.z) * x1.z + s1.z);
            yy += float(rrr.w) * (float(uu.w) * x1.w + s1.w);
            s1.x = s1.x * ww1.x + x1.x;
            s1.y = s1.y * ww1.y + x1.y;
            s1.z = s1.z * ww1.z + x1.z;
            s1.w = s1.w * ww1.w + x1.w;

            float4 x2;
            x2.x = float(kkk.a) * vv;
            x2.y = float(kkk.b) * vv;
            x2.z = float(kkk.c) * vv;
            x2.w = float(kkk.d) * vv;
            float4 &s2 = state[j<<1|1];
            const float4 ww2 = w[j<<1|1];
            
            yy += float(rrr.a) * (float(uu.a) * x2.x + s2.x);
            yy += float(rrr.b) * (float(uu.b) * x2.y + s2.y);
            yy += float(rrr.c) * (float(uu.c) * x2.z + s2.z);
            yy += float(rrr.d) * (float(uu.d) * x2.w + s2.w);
            s2.x = s2.x * ww2.x + x2.x;
            s2.y = s2.y * ww2.y + x2.y;
            s2.z = s2.z * ww2.z + x2.z;
            s2.w = s2.w * ww2.w + x2.w;
        }
        y[_tt] = F(yy);
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
