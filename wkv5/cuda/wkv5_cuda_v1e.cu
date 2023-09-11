#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x - (blockIdx.x / H) * H;
    const int i = threadIdx.x;
    _w += h*N;
    _u += h*N;

    const float4 *__restrict__ const k = (float4 *)(_k);
    const float4 *__restrict__ const r = (float4 *)(_r);
    const float4 *__restrict__ const w = (float4 *)(_w);
    const float4 *__restrict__ const u = (float4 *)(_u);
    const F *__restrict__ const v = _v;
    F *__restrict__ const y = _y;

    __shared__ float state[N * N];
    __shared__ float4 rr[N >> 2], kk[N >> 2];

    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0.0f;

    for (int _tt = b*T*C + h*N + i, _tend = (b+1)*T*C + h*N + i; _tt < _tend; _tt += C)
    {
        const int _t = _tt >> 2;
        const F vv = v[_tt];
        F yy = 0.0;
        
        rr[i >> 2] = r[_t];
        kk[i >> 2] = k[_t];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < N >> 2; j++)
        {
            const int j4n = (j << 2) * N;

            const float4 ww = w[j];
            const float4 uu = u[j];
            const float4 rrr = rr[j];
            const float4 kkk = kk[j];

            float4 x;
            x.x = kkk.x * vv;
            x.y = kkk.y * vv;
            x.z = kkk.z * vv;
            x.w = kkk.w * vv;

            F &s0 = state[j4n + i];
            F &s1 = state[j4n + N + i];
            F &s2 = state[j4n + 2*N + i];
            F &s3 = state[j4n + 3*N + i];

            yy += rrr.x * (uu.x * x.x + s0) + rrr.y * (uu.y * x.y + s1) + rrr.z * (uu.z * x.z + s2) + rrr.w * (uu.w * x.w + s3);
            s0 = s0 * ww.x + x.x;
            s1 = s1 * ww.y + x.y;
            s2 = s2 * ww.z + x.z;
            s3 = s3 * ww.w + x.w;
        }
        y[_tt] = yy;
        __syncthreads();
    }
}

template <typename F>
__global__ void kernel_backward (const int B, const int T, const int C, const int H,
    const F *__restrict__ const r, const F *__restrict__ const k, const F *__restrict__ const v, const F *__restrict__ w, const F *__restrict__ wwww, const F *__restrict__ u, const F *__restrict__ const gy,
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

    __shared__ float state[N * N], vv[N], rr[N], kk[N], gyy[N];

    #pragma unroll
    for (int j = 0; j < N; ++j){
        state[j * N + i] = 0;
    }
    
    const float ww = w[i];
    const float uu = u[i];
    const float wwwww = wwww[i];
    float saaaa[N] = {0.0f}, sbbbb[N] = {0.0f};

    for (int _t = b*T*C + h*N + i, _tend = (b+1)*T*C + h*N + i; _t < _tend; _t += C)
    {
        const F kk = k[_t];
        const F rr = r[_t];
        F grr = 0;
        F guu = 0;

        vv[i] = v[_t];
        gyy[i] = gy[_t];

        __syncthreads();

        #pragma unroll
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
        if (_t < _tend - 2 * C){
            const F rr_value = r[_t+2*C];
            gyy[i] = gy[_t+2*C];
            __syncthreads();

            #pragma unroll
            for (int j = 0; j < N; j++){
                float x = vv[j] * kk;
                saaaa[j] = ww * (saaaa[j] + sbbbb[j] + x);
                sbbbb[j] = ww * (sbbbb[j] + x);
                atomicAdd(gw+i, rr_value * wwwww * saaaa[j] * gyy[j]);
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;
    
    for (int _t = (b+1)*T*C + h*N + i - C, _tend = b*T*C + h*N + i; _t >= _tend; _t -= C)
    {
        const F rr = r[_t];
        F gkk = 0;

        vv[i] = v[_t];
        gyy[i] = gy[_t];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < N; j++)
        {

            float x = gyy[j] * rr;
            float s = state[j * N + i];

            gkk += vv[j] * (uu * x + s);
            state[j * N + i] = s * ww + x;
        }
        gk[_t] = gkk;
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j * N + i] = 0;

    for (int _t = (b+1)*T*C + h*N + i - C, _tend = b*T*C + h*N + i; _t >= _tend; _t -= C)
    {
        const F gy_value = gy[_t];
        F gvv = 0;

        kk[i] = k[_t];
        rr[i] = r[_t];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < N; j++)
        {

            float x = gy_value * rr[j];
            float s = state[j * N + i];

            gvv += kk[j] * (u[j] * x + s);
            state[j * N + i] = s * w[j] + x;
        }
        gv[_t] = gvv;
        __syncthreads();
    }
}

void cuda_forward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *y)
{
    assert(H*N == C);
    kernel_forward<<<dim3(B * H), dim3(N)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *ww, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    assert(H*N == C);
    kernel_backward<<<dim3(B * H), dim3(N)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
