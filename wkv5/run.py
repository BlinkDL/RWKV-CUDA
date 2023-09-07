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

DEVICE = 'cuda'
CUDA_KERNEL_VERSION = 'coll'

'''
python run.py correctness && python run.py benchmark
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

    return out.view(B, T, C)

######################################################################################################
# CUDA kernel
######################################################################################################

if JOB == 'correctness':
    HEAD_SIZE = 3
else:
    HEAD_SIZE = 64

from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda_v{CUDA_KERNEL_VERSION}.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 999", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DN={HEAD_SIZE}"])

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
        ctx.save_for_backward(r, k, v, w, u)
        y = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.forward(B, T, C, H, r, k, v, w, u, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        H = ctx.H
        r, k, v, w, u = ctx.saved_tensors
        gr = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gk = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gv = torch.zeros((B, T, C), device='cuda', requires_grad=False)
        gw = torch.zeros((B, H, C//H), device='cuda', requires_grad=False)
        gu = torch.zeros((B, H, C//H), device='cuda', requires_grad=False)
        wkv_cuda.backward(B, T, C, H, r, k, v, w, u, gy.contiguous(), gr, gk, gv, gw, gu)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r.cuda(), k.cuda(), v.cuda(), w.cuda(), u.cuda())

######################################################################################################
# Check correctness & speed benchmark
######################################################################################################

def CHECK_CORRECT():

    # B = 16
    # T = 4
    # C = 12
    # H = 4

    B = 2
    T = 4
    C = 12
    H = 4

    # B = 1
    # T = 2
    # C = 2
    # H = 2

    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)

    y0 = RUN_FORMULA_1(B, T, C, H, r, k, v, w, u)
    print('result', val(y0), '\n\n')

    y1 = RUN_FORMULA_2(B, T, C, H, r, k, v, w, u)
    print('result', val(y1), '\n\n')

    y2 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
    print('result', val(y2), '\n\n')

    print('--> correct =', torch.allclose(y0, y1), torch.allclose(y0, y2),
          ', err ratio =', get_err_ratio(y0, y1), get_err_ratio(y0, y2))

def CHECK_SPEED(silent=False):

    B = 8
    T = 4096
    C = 4096
    H = C // HEAD_SIZE
    print('B', B, 'T', T, 'C', C, 'H', H)

    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(8):
            r = RUN_CUDA(B, T, C, H, r, k, v, w, u)
    if not silent:
        print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

if __name__ == "__main__":

    if JOB == 'correctness':
        print('\n\nCheck correctness...')
        CHECK_CORRECT()
    else:
        print('\n\nCUDA warmup...')
        CHECK_SPEED(silent=True)  # warmup
        CHECK_SPEED()
