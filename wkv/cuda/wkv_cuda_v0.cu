#include <stdio.h>

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y) {
    const int _b = blockIdx.x;
    const int _c = threadIdx.x;
    const int _offset = _b*T*C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    y[0] = v[0];
    F a = v[0];
    F b = 1;
    F p = k[0];
    for (int i = 1; i < T; i++) 
    {
        const int ii = i*C;
        F kk = k[ii];
        F vv = v[ii];

        F q = max(p, u+kk);
        F e1 = exp(p - q);
        F e2 = exp(u+kk - q);
        y[ii] = (e1 * a + e2 * vv) / (e1 * b + e2);

        q = max(p+w, kk);
        e1 = exp(p+w - q);
        e2 = exp(kk - q);
        a = e1 * a + e2 * vv;
        b = e1 * b + e2;
        p = q;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _gy,
                               F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv) {
    const int _b = blockIdx.x;
    const int _c = threadIdx.x;

    F u = _u[_c];
    F w = _w[_c];
    const int _offset = _b*T*C + _c;
    const F *__restrict__ const gy = _gy + _offset;
    F k[1024];
    F v[1024];
    for (int i = 0; i < T; i++) 
    {
        const int ii = _offset + i*C; 
        k[i] = _k[ii];
        v[i] = _v[ii];
    }

    F gw = 0;
    F gu = 0;
    F gk[1024] = {0};
    F gv[1024] = {0};

    F a = 0;
    F b = 0;
    F p = -65500;
    F qq = 0;
    F r = 0;
    F rr = 0;
    F s = 0;
    F ss = 0;
    F ee = 0;
    for (int i = 0; i < T; i++) 
    {
        F kk = k[i];
        F vv = v[i];
        F gg = gy[i*C];

        F q = max(p, u+kk);
        F e1 = exp(p - q);
        F e2 = exp(u+kk - q);
        
        F c = e1 * a + e2 * vv;
        F d = e1 * b + e2;
        
        for (int j = 0; j < i; j++) 
        {
            ee = exp((i-j-1)*w + k[j] - q) * gg / d;
            gv[j] += ee;
            gk[j] += ee * (v[j] - c / d);
        }
        ee = e2 * gg / d;
        gv[i] += ee;
        ee *= (vv - c / d);
        gk[i] += ee;
        gu += ee;

        if (i > 2) 
        {
            e1 = exp(w + qq - q);
            e2 = exp(w + k[i-2] - q);
            ss = e1 * ss + e2;
            s = e1 * s + ss;
            rr = e1 * rr + e2 * v[i-2];
            r = e1 * r + rr;
        }
        if (i == 2) 
        {
            ss = exp(w + k[0] - q);
            s = ss;
            rr = ss * v[0];
            r = rr;
        }
        gw += (r / d - c * s / (d * d)) * gg * w;
        qq = q;

        q = max(p+w, kk);
        e1 = exp(p+w - q);
        e2 = exp(kk - q);
        a = e1 * a + e2 * vv;
        b = e1 * b + e2;
        p = q;
    }

    const int _offsetBC = _b*C + _c;
    _gw[_offsetBC] += gw;
    _gu[_offsetBC] += gu;
    F *__restrict__ const __gk = _gk + _offset;
    F *__restrict__ const __gv = _gv + _offset;
    for (int i = 0; i < T; i++) 
    {
        const int ii = i*C;
        __gk[ii] = gk[i];
        __gv[ii] = gv[i];
    }
}

// note: test B,C & 1,BC & BC,1 combinations

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y) {
    dim3 numBlocks(B);
    dim3 threadsPerBlock(C);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv)
{
    dim3 numBlocks(B);
    dim3 threadsPerBlock(C);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}
