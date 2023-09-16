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
            const float4 ww = w[j*2];
            const Pack8X<F> uu = u[j];
            const Pack8X<F> rrr = rr[j];
            const Pack8X<F> kkk = kk[j];

            Pack8X<F> x;
            x.x = float(kkk.x) * vv;
            x.y = float(kkk.y) * vv;
            x.z = float(kkk.z) * vv;
            x.w = float(kkk.w) * vv;
            x.a = float(kkk.a) * vv;
            x.b = float(kkk.b) * vv;
            x.c = float(kkk.c) * vv;
            x.d = float(kkk.d) * vv;

            float4 &s = state[j*2];

            yy += float(rrr.x) * (float(uu.x) * float(x.x) + s.x) 
                    + float(rrr.y) * (float(uu.y) * float(x.y) + s.y) 
                    + float(rrr.z) * (float(uu.z) * float(x.z) + s.z) 
                    + float(rrr.w) * (float(uu.w) * float(x.w) + s.w);
            s.x = s.x * ww.x + float(x.x);
            s.y = s.y * ww.y + float(x.y);
            s.z = s.z * ww.z + float(x.z);
            s.w = s.w * ww.w + float(x.w);

            const float4 ww2 = w[j*2+1];
            float4 &s2 = state[j*2+1];
            yy += float(rrr.a) * (float(uu.a) * float(x.a) + s2.x) 
                    + float(rrr.b) * (float(uu.b) * float(x.b) + s2.y) 
                    + float(rrr.c) * (float(uu.c) * float(x.c) + s2.z) 
                    + float(rrr.d) * (float(uu.d) * float(x.d) + s2.w);
            s2.x = s2.x * ww2.x + float(x.a);
            s2.y = s2.y * ww2.y + float(x.b);
            s2.z = s2.z * ww2.z + float(x.c);
            s2.w = s2.w * ww2.w + float(x.d);
        }
        y[_tt] = F(yy);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const float *__restrict__ __w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, float *__restrict__ _gw, float *__restrict__ _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _w += h*_N_;
    _u += h*_N_;
    __w += h*_N_;

    __shared__ float v[_N_], r[_N_], k[_N_], gy[_N_];
    
    const float w = _w[i];
    const float u = float(_u[i]);
    const float ww = __w[i];
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0};

    for (int _t = b*T*C + h*_N_ + i, _tend = (b+1)*T*C + h*_N_ + i; _t < _tend; _t += C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy[i] = float(_gy[_t]);
        __syncthreads();

        const float k = float(_k[_t]);
        const float r = float(_r[_t]);
        float gr = 0;
        float gu = 0;

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
        _gu[_t] = F(gu);
        
        float gw = 0;
        if (_t < _tend - 2*C)
        {
            __syncthreads();
            gy[i] = float(_gy[_t + 2*C]);
            __syncthreads();

            const float r = float(_r[_t + 2*C]);

            #pragma unroll
            for (int j = 0; j < _N_; j++)
            {
                float x = v[j] * k;
                saaaa[j] = w * (saaaa[j] + sbbbb[j] + x);
                sbbbb[j] = w * (sbbbb[j] + x);
                
                gw += r * ww * saaaa[j] * gy[j];
            }
        }
        _gw[_t] = gw;
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j] = 0;
    
    for (int _t = (b+1)*T*C + h*_N_ + i - C, _tend = b*T*C + h*_N_ + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        v[i] = float(_v[_t]);
        gy[i] = float(_gy[_t]);
        __syncthreads();

        const float r = float(_r[_t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gy[j] * r;
            float s = state[j];
            state[j] = s * w + x;

            gk += v[j] * (u * x + s);
        }
        _gk[_t] = F(gk);
    }

    #pragma unroll
    for (int j = 0; j < _N_; ++j)
        state[j] = 0;

    for (int _t = (b+1)*T*C + h*_N_ + i - C, _tend = b*T*C + h*_N_ + i; _t >= _tend; _t -= C)
    {
        __syncthreads();
        k[i] = float(_k[_t]);
        r[i] = float(_r[_t]);
        __syncthreads();

        const float gy = float(_gy[_t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float x = gy * r[j];
            float s = state[j];
            state[j] = s * float(_w[j]) + x;

            gv += k[j] * (float(_u[j]) * x + s);
        }
        _gv[_t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, float *gw, float *gu)
{
    assert(H*_N_ == C);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
