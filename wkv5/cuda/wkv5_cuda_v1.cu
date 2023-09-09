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
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
                                const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _gy,
                                F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gw, F *__restrict__ const _gu)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / C;
    const int h = (idx / N) % H;
    const int n = idx % N;

    for(int t=0; t<T; t++){
        for (int nn = 0; nn < N; nn++){
            for(int tt=0; tt <= t; tt++){
                F ww = (tt == t) ? _u[h*N+n] : pow(_w[h*N+n], t - tt - 1);
                _gr[b*T*H*N + t*H*N + h*N + n] += ww * _k[b*T*H*N + tt*H*N + h*N + n] *
                                                _v[b*T*H*N + tt*H*N + h*N + nn] *
                                                _gy[b*T*H*N + t*H*N + h*N + nn];
            }

            for(int tt=t; tt < T; tt++){
                F ww = (tt == t) ? _u[h*N+n] : pow(_w[h*N+n], tt - t - 1);
                _gk[b*T*H*N + t*H*N + h*N + n] += _r[b*T*H*N + tt*H*N + h*N + n] * ww *
                                                _v[b*T*H*N + t*H*N + h*N + nn] * 
                                                _gy[b*T*H*N + tt*H*N + h*N + nn];

                ww = (tt == t) ? _u[h*N+nn] : pow(_w[h*N+nn], tt - t - 1);
                _gv[b*T*H*N + t*H*N + h*N + n] += _r[b*T*H*N + tt*H*N + h*N + nn] * ww *
                                                _k[b*T*H*N + t*H*N + h*N + nn] *
                                                _gy[b*T*H*N + tt*H*N + h*N + n];  
            }

            atomicAdd(_gu+h*N+n, _r[b*T*H*N + t*H*N + h*N + n] *
                            _k[b*T*H*N + t*H*N + h*N + n] *
                            _v[b*T*H*N + t*H*N + h*N + nn] *
                            _gy[b*T*H*N + t*H*N + h*N + nn]);

            for(int tt=0; tt<t-1; tt++){
                F ww = (t-tt-1) * log(_w[h*N+n]) * pow(_w[h*N+n], t - tt - 1);
                atomicAdd(_gw + h*N+n, _r[b*T*H*N + t*H*N + h*N + n] * ww *
                            _k[b*T*H*N + tt*H*N + h*N + n] *
                            _v[b*T*H*N + tt*H*N + h*N + nn] *  
                            _gy[b*T*H*N + t*H*N + h*N + nn]);
            }
        }
    }
}

void cuda_forward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *y)
{
    dim3 threadsPerBlock( min(B*C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu)
{
    dim3 threadsPerBlock( min(B*C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}
