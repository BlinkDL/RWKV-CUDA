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
CUDA_KERNEL_VERSION = '1c'

'''
python run.py correctness && python run.py correctness_more && python run.py benchmark
python run.py backward
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

def RUN_BACKWARD_1(B, T, C, H, gy, r, k, v, w, u):
    N = C // H
    gy = gy.view(B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    w = w.view(H, N)
    u = u.view(H, N)

    gr = torch.zeros((B, T, H, N), device=DEVICE)    
    gk = torch.zeros((B, T, H, N), device=DEVICE)
    gv = torch.zeros((B, T, H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for t in range(T):
                for n in range(N):
                    for nn in range(N):

                        for tt in range(t+1):
                            ww = u[h,n] if (tt == t) else w[h,n] ** (t - tt - 1)
                            gr[b,t,h,n] += ww * k[b,tt,h,n] * v[b,tt,h,nn] * gy[b,t,h,nn]

                        for tt in range(t,T):
                            ww = u[h,n] if (tt == t) else w[h,n] ** (tt - t - 1)
                            gk[b,t,h,n] += r[b,tt,h,n] * ww * v[b,t,h,nn] * gy[b,tt,h,nn]

                            ww = u[h,nn] if (tt == t) else w[h,nn] ** (tt - t - 1)
                            gv[b,t,h,n] += r[b,tt,h,nn] * ww * k[b,t,h,nn] * gy[b,tt,h,n]

    return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C)

######################################################################################################
# CUDA kernel
######################################################################################################

if JOB == 'correctness' or JOB == 'backward':
    # B = 16
    # T = 4
    # C = 16
    # HEAD_SIZE = 4

    # B = 2
    # T = 4
    # C = 16
    # HEAD_SIZE = 4

    # B = 2
    # T = 3
    # C = 4
    # HEAD_SIZE = 2

    B = 1
    T = 3
    C = 2
    HEAD_SIZE = 1
else:
    B = 8
    T = 4096
    C = 4096
    HEAD_SIZE = 64

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
        ctx.save_for_backward(r, k, v, w, u)
        y = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda_ref.forward(B, T, C, H, r, k, v, w, u, y)
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
        wkv_cuda_ref.backward(B, T, C, H, r, k, v, w, u, gy.contiguous(), gr, gk, gv, gw, gu)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
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
    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)

    if JOB == 'correctness_more':
        y0 = RUN_CUDA_REF(B, T, C, H, r, k, v, w, u)
        y1 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
        print('--> correct =', torch.allclose(y0, y1), ', err ratio =', get_err_ratio(y0, y1))
    
    else:

        y0 = RUN_FORMULA_1(B, T, C, H, r, k, v, w, u)
        print(f'result\n{val(y0)}\n')

        y1 = RUN_FORMULA_2(B, T, C, H, r, k, v, w, u)
        print(f'result\n{val(y1)}\n')

        y2 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
        print(f'result\n{val(y2)}\n')

        print('--> correct =', torch.allclose(y0, y1), torch.allclose(y0, y2),
            ', err ratio =', get_err_ratio(y0, y1), get_err_ratio(y0, y2))

        
        if JOB == 'backward':

            def LOSS(y): # a strange loss for better verification
                return ((y * y) - torch.tanh(y)).sum()

            yy = y0.clone().detach().requires_grad_(True)
            LOSS(yy).backward()
            gy = yy.grad.data.clone()
            print(f'grad_y\n{val(gy)}\n')

            LOSS(y0).backward()
            gr0 = r.grad.data.clone()
            gk0 = k.grad.data.clone()
            gv0 = v.grad.data.clone()
            gw0 = w.grad.data.clone()
            gu0 = u.grad.data.clone()
            print(f'g_r0\n{val(gr0)}\n')
            print(f'g_k0\n{val(gk0)}\n')
            print(f'g_v0\n{val(gv0)}\n')
            print(f'g_w0\n{val(gw0)}\n')
            print(f'g_u0\n{val(gu0)}\n')

            gr1, gk1, gv1 = RUN_BACKWARD_1(B, T, C, H, gy, r, k, v, w, u)
            print(f'g_r1\n{val(gr1)}\n')
            print(f'g_k1\n{val(gk1)}\n')
            print(f'g_v1\n{val(gv1)}\n')

            print('--> g_r correct =', torch.allclose(gr0, gr1), ', err ratio =', get_err_ratio(gr0, gr1))
            print('--> g_k correct =', torch.allclose(gk0, gk1), ', err ratio =', get_err_ratio(gk0, gk1))
            print('--> g_v correct =', torch.allclose(gv0, gv1), ', err ratio =', get_err_ratio(gv0, gv1))


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
    else:
        print(f'\n\nCUDA kernel v{CUDA_KERNEL_VERSION} warmup...')
        CHECK_SPEED(silent=True)  # warmup
        CHECK_SPEED()
