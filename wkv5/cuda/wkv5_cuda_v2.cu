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

    const int _o0 = _b * T * C + _h * N;
    const int _o1 = _h * N;

    const float4 *__restrict__ const k = (float4 *)(_k + _o0);
    const float4 *__restrict__ const r = (float4 *)(_r + _o0);
    const float4 *__restrict__ const w = (float4 *)(_w + _o1);
    const float4 *__restrict__ const u = (float4 *)(_u + _o1);
    const F *__restrict__ const v = _v + _o0 + _i;
    F *__restrict__ const y = _y + _o0 + _i;

    __align__(16) float4 state[N / 4] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f) };

    for (int __t = 0; __t < T; __t++)
    {
        const int _t = __t * (C >> 2);
        const int tt = __t * C;
        const F vv = v[tt];

        for (int _j = 0; _j < N / 4; _j++)
        {
            const int j = _t + _j;

            const float4 k_val = k[j];
            const float4 r_val = r[j];
            float4 x;
            x.x = k_val.x * vv;
            x.y = k_val.y * vv;
            x.z = k_val.z * vv;
            x.w = k_val.w * vv;

            float4 s = state[_j];

            float4 result;
            result.x = r_val.x * (u[_j].x * x.x + s.x);
            result.y = r_val.y * (u[_j].y * x.y + s.y);
            result.z = r_val.z * (u[_j].z * x.z + s.z);
            result.w = r_val.w * (u[_j].w * x.w + s.w);

            atomicAdd(&(y[tt]), result.x);
            atomicAdd(&(y[tt]), result.y);
            atomicAdd(&(y[tt]), result.z);
            atomicAdd(&(y[tt]), result.w);

            state[_j].x = s.x * w[_j].x + x.x;
            state[_j].y = s.y * w[_j].y + x.y;
            state[_j].z = s.z * w[_j].z + x.z;
            state[_j].w = s.w * w[_j].w + x.w;
        }
    }
}

template <typename F>
__global__ void kernel_backward (const int B, const int T, const int C, const int H,
    const F *__restrict__ const r, const F *__restrict__ const k, const F *__restrict__ const v, const F *__restrict__ const w, const F *__restrict__ const wwww, const F *__restrict__ const _u, const F *__restrict__ const gy,
    F *__restrict__ const gr, F *__restrict__ const gk, F *__restrict__ const gv, F *__restrict__ const gw, F *__restrict__ const gu)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // B * H * T * N
    const int b = idx / H / T / N;
    const int h = (idx / T / N) % H;
    const int t = (idx / N) % T;
    const int n = idx % N;
    
    for(int nn = 0; nn < N; nn++){
        for (int tt = 0; tt <= t; tt++) {
            F ww = (tt == t) ? _u[h*N + n] : pow(w[h*N + n], t-tt-1);
            
            gr[b*T*H*N + t*H*N + h*N + n] += ww * k[b*T*H*N + tt*H*N + h*N + n] *
                v[b*T*H*N + tt*H*N + h*N + nn] * gy[b*T*H*N + t*H*N + h*N + nn];
        }

        for (int tt = t; tt < T; tt++) {
            F ww = (tt == t) ? _u[h*N + n] : pow(w[h*N + n], tt-t-1);
            
            gk[b*T*H*N + t*H*N + h*N + n] += r[b*T*H*N + tt*H*N + h*N + n] * ww *
                v[b*T*H*N + t*H*N + h*N + nn] * gy[b*T*H*N + tt*H*N + h*N + nn];

            ww = (tt == t) ? _u[h*N + nn] : pow(w[h*N + nn], tt-t-1);
            
            gv[b*T*H*N + t*H*N + h*N + n] += r[b*T*H*N + tt*H*N + h*N + nn] * ww *
                k[b*T*H*N + t*H*N + h*N + nn] * gy[b*T*H*N + tt*H*N + h*N + n];
        }

        atomicAdd(gu + h*N + n, r[b*T*H*N + t*H*N + h*N + n] * k[b*T*H*N + t*H*N + h*N + n] *
                v[b*T*H*N + t*H*N + h*N + nn] * gy[b*T*H*N + t*H*N + h*N + nn]);

        for (int tt = 0; tt < t-1; tt++) {
            F ww = (t-tt-1) * wwww[h*N + n] * pow(w[h*N + n], t-tt-1);

            atomicAdd(gw + h*N + n, r[b*T*H*N + t*H*N + h*N + n] * ww * k[b*T*H*N + tt*H*N + h*N + n] *
                v[b*T*H*N + tt*H*N + h*N + nn] * gy[b*T*H*N + t*H*N + h*N + nn]);
        }
    }
}

void cuda_forward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *y)
{
    assert(H*N == C);
    const int SIZE = B*C;
    dim3 threadsPerBlock(min(SIZE, 32));
    assert(SIZE % threadsPerBlock.x == 0);
    dim3 numBlocks(SIZE / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *ww, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    dim3 threadsPerBlock( min(B*H*T*N, 256) );
    assert(B * H * T * N % threadsPerBlock.x == 0);
    dim3 numBlocks(B * H * T * N / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}
