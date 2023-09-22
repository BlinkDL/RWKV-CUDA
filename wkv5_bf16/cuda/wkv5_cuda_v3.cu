// Forward Origin Author: Bleatan
#include <assert.h>
#include <stdio.h>
#include <torch/extension.h>

typedef at::BFloat16 bf16;

constexpr int N = _N_;
constexpr int PARALLEL_SCAN_BLOCK = 512;
constexpr int TILE_T = 16;

__global__ void kernel_forward_accumulate_state(const int nblocks_per_sample,
                                                float *__restrict__ state,
                                                const float *__restrict__ w) {
  const int b = blockIdx.x, B = gridDim.x;
  const int h = blockIdx.y / N, H = gridDim.y / N, HNN = H * N * N;
  const int i = blockIdx.y % N;
  const int j = threadIdx.x;

  float wp = powf(w[h * N + i], PARALLEL_SCAN_BLOCK);
  state += (((b * nblocks_per_sample * H + h) * N) + j) * N + i;
  float s = 0;

  for (int block = 0; block < nblocks_per_sample; ++block) {
    const float next_s = s * wp + *state;
    *state = s;
    s = next_s;
    state += HNN;
  }
}

template <bool DoY, typename F>
__global__ void
kernel_forward(const int T, const F *__restrict__ const _r,
               const F *__restrict__ const _k, const F *__restrict__ const _v,
               const float *__restrict__ _w, const F *__restrict__ _u,
               F *__restrict__ const _y, float *__restrict__ state) {
  const int b = blockIdx.x, B = gridDim.x;
  if (!DoY && b == B - 1)
    return;

  const int h = blockIdx.y, H = gridDim.y;
  const int i = threadIdx.x;
  _w += h * N;
  _u += h * N;

  __shared__ float rr[TILE_T][N], kk[TILE_T][N];

  state += (b * H + h) * N * N;

  if constexpr (!DoY)
    for (int j = 0; j < N; ++j)
      state[j * N + i] = 0;

  for (int _t = b * T * H * N + h * N + i,
           _tend = (b + 1) * T * H * N + h * N + i;
       _t < _tend; _t += TILE_T * H * N) {
    float yy[TILE_T], vv[TILE_T];
#pragma unroll(TILE_T)
    for (int tt = 0, t = _t; tt < TILE_T; (++tt), (t += H * N)) {
      vv[tt] = (float)_v[t];
      if constexpr (DoY) {
        yy[tt] = 0.f;
        rr[tt][i] = (float)_r[t];
      }
      kk[tt][i] = (float)_k[t];
    }

    __syncthreads();

    for (int j = 0; j < N; j += 1) {
      float s = state[j * N + i];
      float w = _w[j];
      float u = (float)_u[j];

#pragma unroll(TILE_T)
      for (int tt = 0, t = _t; tt < TILE_T; (++tt), (t += H * N)) {
        float x = kk[tt][j] * vv[tt];
        if constexpr (DoY) {
            float yyy = rr[tt][j] * (u * x + s);
            yy[tt] += yyy;
        }
        s = s * w + x;
      }
      state[j * N + i] = s;
    }

    if constexpr (DoY)
#pragma unroll(TILE_T)
      for (int tt = 0, t = _t; tt < TILE_T; (++tt), (t += H * N))
        _y[t] = (F)yy[tt];

    __syncthreads();
  }
}

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r,
             torch::Tensor &k, torch::Tensor &v, torch::Tensor &w,
             torch::Tensor &u, torch::Tensor &y) {
  assert(H * N == C);
  assert(T % PARALLEL_SCAN_BLOCK == 0);
  const int blocksz = PARALLEL_SCAN_BLOCK;
  const int nblocks_per_sample = T / blocksz;
  const int nblocks = B * nblocks_per_sample;
  cudaEvent_t events[5];
  for (int i = 0; i < 5; ++i)
    cudaEventCreate(&events[i]);

  cudaEventRecord(events[0]);
  torch::Tensor states =
      torch::empty({B, nblocks_per_sample, H, N, N}, w.options());
  cudaEventRecord(events[1]);
  kernel_forward<false><<<dim3(nblocks, H), dim3(N)>>>(
      blocksz, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(),
      w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>(),
      states.data_ptr<float>());
  cudaEventRecord(events[2]);
  kernel_forward_accumulate_state<<<dim3(B, H * N), dim3(N)>>>(
      nblocks_per_sample, states.data_ptr<float>(), w.data_ptr<float>());
  cudaEventRecord(events[3]);
  kernel_forward<true><<<dim3(nblocks, H), dim3(N)>>>(
      blocksz, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(),
      w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>(),
      states.data_ptr<float>());
  cudaEventRecord(events[4]);

  cudaEventSynchronize(events[4]);
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
    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0};

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
            saaaa[j] = w * (saaaa[j] + sbbbb[j] + x);
            sbbbb[j] = w * (sbbbb[j] + x);
            
            gw += r2 * ww * saaaa[j] * gy2[j];
        }
    }
    
    _gw[b*C + h*_N_ + i] = F(gw);

    #pragma unroll
    for (int j = 0; j < _N_; ++j) {
        saaaa[j] = 0;
        sbbbb[j] = 0;
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

        float gk = 0, gv = 0, x, s;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            x = gy[j] * r[i];
            s = saaaa[j];
            saaaa[j] = s * w + x;
            gk += v[j] * (u * x + s);

            x = gy[i] * r[j];
            s = sbbbb[j];
            sbbbb[j] = s * w_[j] + x;
            gv += k[j] * (u_[j] * x + s);
        }
        _gk[_t] = F(gk);
        _gv[_t] = F(gv);
    }
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*_N_ == C);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, ww, u, gy, gr, gk, gv, gw, gu);
}

void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &ww, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), ww.data_ptr<float>(), u.data_ptr<bf16>(), gy.data_ptr<bf16>(), gr.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "wkv5 forward");
  m.def("backward", &backward, "wkv5 backward");
}

TORCH_LIBRARY(wkv5, m) {
  m.def("forward", forward);
  m.def("backward", backward);
}
