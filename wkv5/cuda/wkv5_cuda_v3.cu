#include <stdio.h>
#include <assert.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                                      const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u,
                                      F *__restrict__ const _y)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_div_n_4 = idx / (N >> 2);
    const int _b = idx / (C >> 2);
    const int _h = idx_div_n_4 - idx_div_n_4 / H * H ;
    const int _i = (idx -  idx_div_n_4 * (N >> 2)) << 2;

    const int _o0 = _b * T * C + _h * N;
    const int _o1 = _h * N;

    const float4 *__restrict__ const k = (float4 *)(_k + _o0);
    const float4 *__restrict__ const v =  (float4 *)(_v + _o0 + _i); 
    const float4 *__restrict__ const r = (float4 *)(_r + _o0);
    float4 *__restrict__ const y = (float4 *)(_y + _o0 + _i); 

    __align__(16) float4 state[N / 4] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f) };

    for (int __t = 0; __t < T; __t++)
    {
        const int _t = __t * (C / 4); 
        const float4 vv = v[_t];

        float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int _j = 0; _j < N / 4; _j++)
        {
            const int j = _t + _j;
            const int m = _o1 + _j * 4; 

            const float4 k_val = k[j];
            const float4 r_val = r[j];
            float4 x;
            x.x = k_val.x * vv.x;
            x.y = k_val.y * vv.y;
            x.z = k_val.z * vv.z;
            x.w = k_val.w * vv.w;

            float4 s = state[_j];

            result.x += r_val.x * (_u[m] * x.x + s.x);
            result.y += r_val.y * (_u[m + 1] * x.y + s.y);
            result.z += r_val.z * (_u[m + 2] * x.z + s.z);
            result.w += r_val.w * (_u[m + 3] * x.w + s.w);

            state[_j].x = s.x * _w[m] + x.x;
            state[_j].y = s.y * _w[m + 1] + x.y;
            state[_j].z = s.z * _w[m + 2] + x.z;
            state[_j].w = s.w * _w[m + 3] + x.w;
        }

        atomicAdd(&(y[_t].x), result.x);
        atomicAdd(&(y[_t].y), result.y);
        atomicAdd(&(y[_t].z), result.z);
        atomicAdd(&(y[_t].w), result.w);
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
    if (N % 4 == 0 && C % 4 == 0){
        dim3 threadsPerBlock( min(B * C / 4, 32) );
        assert((B * C / 4) % threadsPerBlock.x == 0);
        dim3 numBlocks(B * C / 4 / threadsPerBlock.x);
        kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
    }
    else{
        dim3 threadsPerBlock( min(B*C, 32) );
        assert(B * C % threadsPerBlock.x == 0);
        dim3 numBlocks(B * C / threadsPerBlock.x);
        kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
    }
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    dim3 threadsPerBlock( min(B*C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}
