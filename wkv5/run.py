import torch, os, sys
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import numpy as np
from math import exp
np.set_printoptions(precision=4, suppress=True, linewidth=200)
# turn off TF32 for higher accuracy
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda'
CUDA_KERNEL_VERSION = '1'

'''
cd /fsx/BlinkDL/CODE/_PUBLIC_/RWKV-CUDA/wkv5
python run.py correctness
python run.py correctness && python run.py correctness_more && python run.py benchmark
python run.py backward

python run.py correctness_more && python run.py benchmark
python run.py benchmark
python run.py torch
'''
JOB = sys.argv[1].strip()

######################################################################################################
# Python version
######################################################################################################

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def val(x):
    return x.detach().cpu().numpy()

def RUN_FORMULA_1(B, T, C, H, r, k, v, w, u):
    N = C // H
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    w = w.view(H, N)
    u = u.view(H, N)
    out = torch.zeros((B, T, H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for t in range(T):
                for n in range(N):
                    for nn in range(N):
                        for tt in range(t+1):
                            ww = u[h,nn] if (tt == t) else w[h,nn] ** (t - tt - 1)
                            out[b,t,h,n] += r[b,t,h,nn] * ww * k[b,tt,h,nn] * v[b,tt,h,n]

    return out.view(B, T, C)

def RUN_FORMULA_2(B, T, C, H, r, k, v, w, u):
    N = C // H
    r = r.flatten().contiguous() # BTHN
    k = k.flatten().contiguous() # BTHN
    v = v.flatten().contiguous() # BTHN
    w = w.flatten().contiguous() # HN
    u = u.flatten().contiguous() # HN
    out = torch.zeros(B*T*C, device=DEVICE).contiguous()

    # kernel for v1/v1a/v1b
    for b in range(B):
        for h in range(H):
            state = torch.zeros(N*N, device=DEVICE).contiguous()
            for t in range(T):

                _o0 = b*H*T*N + t*H*N + h*N
                _o1 = h*N

                for _i in range(N):

                    i = _o0 + _i
                    
                    for _j in range(N):
                        
                        j = _o0 + _j
                        m = _o1 + _j
                        ij = _i * N + _j

                        x = k[j] * v[i]
                        s = state[ij]
                        
                        out[i] += r[j] * (u[m] * x + s)
                        state[ij] = s * w[m] + x

    # kernel for v1c
    # for blockIdx in range(B*H):
    #     b = blockIdx // H
    #     h = blockIdx % H
    #     for i in range(N):
    #         state = torch.empty(N*N, device=DEVICE).contiguous().uniform_(-1, 1)
    #         rr = torch.empty(N, device=DEVICE).contiguous().uniform_(-1, 1)
    #         kk = torch.empty(N, device=DEVICE).contiguous().uniform_(-1, 1)

    #         for j in range(N):
    #             state[j*N + i] = 0
            
    #         for _t in range(b*T*C + h*N + i, (b+1)*T*C + h*N + i, C):

    #             for ii in range(N): # emulate __syncthreads()
    #                 rr[ii] = r[_t - i + ii]
    #                 kk[ii] = k[_t - i + ii]
                
    #             vv = v[_t]
    #             yy = 0
    #             for j in range(N):
    #                 x = kk[j] * vv
    #                 s = state[j*N + i]

    #                 yy += rr[j] * (u[h*N+j] * x + s)
    #                 state[j*N + i] = s * w[h*N+j] + x
    #             out[_t] = yy

    return out.view(B, T, C)

def RUN_BACKWARD_1(B, T, C, H, gy, r, k, v, __w, u):
    N = C // H
    gy = gy.view(B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    _w = -torch.exp(__w).view(H, N)
    u = u.view(H, N)
    w = torch.exp(_w)

    gr = torch.zeros((B, T, H, N), device=DEVICE)    
    gk = torch.zeros((B, T, H, N), device=DEVICE)
    gv = torch.zeros((B, T, H, N), device=DEVICE)
    gw = torch.zeros((H, N), device=DEVICE)
    gu = torch.zeros((H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for n in range(N):
                for t in range(T):
                    for nn in range(N):

                        for tt in range(t+1):
                            ww = u[h,n] if (tt == t) else w[h,n] ** (t - tt - 1)
                            gr[b,t,h,n] += ww * k[b,tt,h,n] * v[b,tt,h,nn] * gy[b,t,h,nn]

                        for tt in range(t,T):
                            ww = u[h,n] if (tt == t) else w[h,n] ** (tt - t - 1)
                            gk[b,t,h,n] += r[b,tt,h,n] * ww * v[b,t,h,nn] * gy[b,tt,h,nn]

                            ww = u[h,nn] if (tt == t) else w[h,nn] ** (tt - t - 1)
                            gv[b,t,h,n] += r[b,tt,h,nn] * ww * k[b,t,h,nn] * gy[b,tt,h,n]

                        gu[h,n] += r[b,t,h,n] * k[b,t,h,n] * v[b,t,h,nn] * gy[b,t,h,nn]

                        for tt in range(t-1):
                            ww = (t-tt-1) * _w[h,n] * (w[h,n] ** (t - tt - 1))
                            gw[h,n] += r[b,t,h,n] * ww * k[b,tt,h,n] * v[b,tt,h,nn] * gy[b,t,h,nn]

    return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw.view(C), gu.view(C)

######################################################################################################
# Original pytorch version (requires w & u to be constant within each head)
######################################################################################################

class RUN_TORCH(torch.jit.ScriptModule):
    def __init__(self, chunk_len):
        super().__init__()
        self.chunk_len = chunk_len

    @torch.jit.script_method
    def jit_func(self, r, k, v, w, wk, wb, ws):
        B, T, C = r.size()
        H = w.size()[1]
        Z = self.chunk_len
        N = C // H
        r = r.view(B, T, H, N).transpose(1, 2) # BTC -> BHTN
        k = k.view(B, T, H, N).transpose(1, 2).transpose(-2, -1) # BTC -> BHTN -> BHNT
        v = v.view(B, T, H, N).transpose(1, 2) # BTC -> BHTN

        s = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype) # state
        x = torch.zeros(B, H, T, N, device=r.device, dtype=r.dtype) # output

        for i in range(T // Z):
            rr = r[:, :, i*Z:i*Z+Z, :]
            kk = k[:, :, :, i*Z:i*Z+Z]
            vv = v[:, :, i*Z:i*Z+Z, :]
            x[:, :, i*Z:i*Z+Z, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb
            s = ws * s + (kk * wk) @ vv

        return x.transpose(1, 2).contiguous().view(B, T, C) # BHTN -> BTHN -> BTC

    def forward(self, B, T, C, H, r, k, v, w, u):
        w = w.view(H, 1)
        u = u.view(H, 1)
        Z = self.chunk_len

        ws = w.pow(Z).reshape(1, H, 1, 1)

        ind = torch.arange(Z-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Z).pow(ind)

        wk = w.reshape(1, H, 1, Z)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Z))
        w = torch.tile(w, [Z])
        w = w[:, :-Z].reshape(-1, Z, 2 * Z - 1)
        w = w[:, :, Z-1:].reshape(1, H, Z, Z)

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        return self.jit_func(r, k, v, w, wk, wb, ws)

######################################################################################################
# CUDA kernel
######################################################################################################

if JOB == 'correctness' or JOB == 'backward':
    # B = 16
    # T = 5
    # C = 16
    # HEAD_SIZE = 4

    # B = 2
    # T = 5
    # C = 16
    # HEAD_SIZE = 4

    B = 2
    T = 5
    C = 4
    HEAD_SIZE = 2

    # B = 1
    # T = 5
    # C = 1
    # HEAD_SIZE = 1

elif JOB == 'correctness_more':
    # B = 13
    # T = 4097
    # C = 4160
    # HEAD_SIZE = 64
    B = 8
    T = 4096
    C = 4096
    HEAD_SIZE = 64
    
elif JOB == 'benchmark' or JOB == 'torch':
    B = 8
    T = 4096
    C = 4096
    HEAD_SIZE = 64
    
    # B = 1
    # T = 5
    # C = 1
    # HEAD_SIZE = 1

H = C // HEAD_SIZE

######################################################################################################
# CUDA reference
######################################################################################################

from torch.utils.cpp_extension import load
wkv_cuda_ref = load(name="wkv5_ref", sources=["cuda/wkv5_ref.cpp", f"cuda/wkv5_cuda_ref.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DN={HEAD_SIZE}"])

class WKV_5_REF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        assert HEAD_SIZE == C // H
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.H = H
        r = r.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        w = w.contiguous()
        u = u.contiguous()
        ew = -torch.exp(w)
        eew = torch.exp(ew)
        ctx.save_for_backward(r, k, v, eew, ew, u)
        y = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda_ref.forward(B, T, C, H, r, k, v, eew, u, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        H = ctx.H
        r, k, v, eew, ew, u = ctx.saved_tensors
        gr = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gk = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gv = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gw = torch.zeros((B, H, C//H), device='cuda', requires_grad=False)
        gu = torch.zeros((B, H, C//H), device='cuda', requires_grad=False)
        wkv_cuda_ref.backward(B, T, C, H, r, k, v, eew, ew, u, gy.contiguous(), gr, gk, gv, gw, gu)
        gw = torch.sum(gw, dim=0).flatten()
        gu = torch.sum(gu, dim=0).flatten()
        return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_REF(B, T, C, H, r, k, v, w, u):
    return WKV_5_REF.apply(B, T, C, H, r.cuda(), k.cuda(), v.cuda(), w.cuda(), u.cuda())

######################################################################################################
# CUDA kernel
######################################################################################################

wkv_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda_v{CUDA_KERNEL_VERSION}.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DN={HEAD_SIZE}", f"-DCC={C}"])
    
class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        assert HEAD_SIZE == C // H
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.H = H
        r = r.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        w = w.contiguous()
        u = u.contiguous()
        ew = -torch.exp(w)
        eew = torch.exp(ew)
        ctx.save_for_backward(r, k, v, eew, ew, u)
        y = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        H = ctx.H
        r, k, v, eew, ew, u = ctx.saved_tensors
        gr = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gk = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gv = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gw = torch.zeros((B, H, C//H), device='cuda', requires_grad=False)
        gu = torch.zeros((B, H, C//H), device='cuda', requires_grad=False)
        wkv_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy.contiguous(), gr, gk, gv, gw, gu)
        gw = torch.sum(gw, dim=0).flatten()
        gu = torch.sum(gu, dim=0).flatten()
        return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r.cuda(), k.cuda(), v.cuda(), w.cuda(), u.cuda())

######################################################################################################
# Check correctness
######################################################################################################

def CHECK_CORRECT():
    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        # print(f'r\n{val(r)}\n')
        # print(f'k\n{val(k)}\n')
        # print(f'v\n{val(v)}\n')
        # print(f'w\n{val(w)}\n')
        # print(f'u\n{val(u)}\n')

    if JOB == 'correctness_more':
        y0 = RUN_CUDA_REF(B, T, C, H, r, k, v, w, u)
        y1 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
        print('--> correct =', torch.allclose(y0, y1), ', err ratio =', get_err_ratio(y0, y1))
    
    else:

        y0 = RUN_FORMULA_1(B, T, C, H, r, k, v, torch.exp(-torch.exp(w)), u)
        print(f'result\n{val(y0)}\n')

        y1 = RUN_FORMULA_2(B, T, C, H, r, k, v, torch.exp(-torch.exp(w)), u)
        print(f'result\n{val(y1)}\n')

        y2 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
        print(f'result\n{val(y2)}\n')

        print('--> correct =', torch.allclose(y0, y1), torch.allclose(y0, y2),
            ', err ratio =', get_err_ratio(y0, y1), get_err_ratio(y0, y2))

        
        if JOB == 'backward':

            def LOSS(y): # a strange loss for better verification
                return ((y * y) - torch.tanh(y)).sum()

            y0 = RUN_FORMULA_1(B, T, C, H, r, k, v, torch.exp(-torch.exp(w)), u)
            yy = y0.clone().detach().requires_grad_(True)
            LOSS(yy).backward()
            gy = yy.grad.data.clone()
            # print(y0)
            # print(f'grad_y\n{val(gy)}\n')

            LOSS(y0).backward()
            gr0 = r.grad.data.clone()
            gk0 = k.grad.data.clone()
            gv0 = v.grad.data.clone()
            gw0 = w.grad.data.clone()
            gu0 = u.grad.data.clone()
            # print(f'g_r0\n{val(gr0)}\n')
            # print(f'g_k0\n{val(gk0)}\n')
            # print(f'g_v0\n{val(gv0)}\n')
            # print(f'g_w0\n{val(gw0)}\n')
            # print(f'g_u0\n{val(gu0)}\n')

            r.grad.data.zero_()
            k.grad.data.zero_()
            v.grad.data.zero_()
            w.grad.data.zero_()
            u.grad.data.zero_()
            
            print('# Check torch ref')
            gr2, gk2, gv2, gw2, gu2 = RUN_BACKWARD_1(B, T, C, H, gy, r, k, v, w, u)
            print('--> g_r correct =', torch.allclose(gr0, gr2), ', err ratio =', get_err_ratio(gr0, gr2))
            print('--> g_k correct =', torch.allclose(gk0, gk2), ', err ratio =', get_err_ratio(gk0, gk2))
            print('--> g_v correct =', torch.allclose(gv0, gv2), ', err ratio =', get_err_ratio(gv0, gv2))
            print('--> g_w correct =', torch.allclose(gw0, gw2), ', err ratio =', get_err_ratio(gw0, gw2))
            print('--> g_u correct =', torch.allclose(gu0, gu2), ', err ratio =', get_err_ratio(gu0, gu2))

            print('# Check CUDA Ref')
            y1 = RUN_CUDA_REF(B, T, C, H, r, k, v, w, u)
            LOSS(y1).backward()

            gr1 = r.grad.data.clone()
            gk1 = k.grad.data.clone()
            gv1 = v.grad.data.clone()
            gw1 = w.grad.data.clone()
            gu1 = u.grad.data.clone()            
            # print(f'g_r1\n{val(gr1)}\n')
            # print(f'g_k1\n{val(gk1)}\n')
            # print(f'g_v1\n{val(gv1)}\n')
            # print(f'g_w1\n{val(gw1)}\n')
            # print(f'g_u1\n{val(gu1)}\n')

            print('--> g_r correct =', torch.allclose(gr0, gr1), ', err ratio =', get_err_ratio(gr0, gr1))
            print('--> g_k correct =', torch.allclose(gk0, gk1), ', err ratio =', get_err_ratio(gk0, gk1))
            print('--> g_v correct =', torch.allclose(gv0, gv1), ', err ratio =', get_err_ratio(gv0, gv1))
            print('--> g_w correct =', torch.allclose(gw0, gw1), ', err ratio =', get_err_ratio(gw0, gw1))
            print('--> g_u correct =', torch.allclose(gu0, gu1), ', err ratio =', get_err_ratio(gu0, gu1))

######################################################################################################
# Check vs pytorch
######################################################################################################

def CHECK_TORCH():

    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(H, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(H, requires_grad=True, device=DEVICE).uniform_(-1, 1)
    # print(f'r\n{val(r)}\n')
    # print(f'k\n{val(k)}\n')
    # print(f'v\n{val(v)}\n')
    # print(f'w\n{val(w)}\n')
    # print(f'u\n{val(u)}\n')

    assert T == 4096
    print(f'B={B} T={T} C={C} HEAD_SIZE={HEAD_SIZE}')
    
    rwkv5_torch = RUN_TORCH(chunk_len = 512)

    y0 = rwkv5_torch.forward(B, T, C, H, r, k, v, w, u)
    y0 = rwkv5_torch.forward(B, T, C, H, r, k, v, w, u)
    # print(f'result\n{val(y0)}\n')
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y0 = rwkv5_torch.forward(B, T, C, H, r, k, v, w, u)
    print('Torch forward\n', prof.key_averages(group_by_stack_n=5).table(
        sort_by='self_cuda_time_total', row_limit=5))
    
    ww = w.repeat_interleave(HEAD_SIZE)
    uu = u.repeat_interleave(HEAD_SIZE)
    y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, uu)
    y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, uu)
    # print(f'result\n{val(y1)}\n')f
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, uu)
    print(f'CUDA kernel v{CUDA_KERNEL_VERSION} forward\n', prof.key_averages(group_by_stack_n=5).table(
        sort_by='self_cuda_time_total', row_limit=5))
    
    print('--> correct =', torch.allclose(y0, y1),
        ', err ratio =', get_err_ratio(y0, y1))

######################################################################################################
# Check speed
######################################################################################################

def CHECK_SPEED(silent=False):
    print('B', B, 'T', T, 'C', C, 'H', H)

    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r = RUN_CUDA(B, T, C, H, r, k, v, w, u)
    if not silent:
        print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

if __name__ == "__main__":

    if JOB == 'correctness' or JOB == 'backward':
        print(f'\n\nCheck CUDA kernel v{CUDA_KERNEL_VERSION} correctness...')
        CHECK_CORRECT()
    
    elif JOB == 'correctness_more':
        print(f'\n\nCheck CUDA kernel v{CUDA_KERNEL_VERSION} correctness (more)...')
        CHECK_CORRECT()
    
    elif JOB == 'benchmark':
        print(f'\n\nCUDA kernel v{CUDA_KERNEL_VERSION} warmup...')
        CHECK_SPEED(silent=True)  # warmup
        CHECK_SPEED()
    
    elif JOB == 'torch':
        CHECK_TORCH()
