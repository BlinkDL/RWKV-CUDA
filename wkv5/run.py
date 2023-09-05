import torch
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
CUDA_KERNEL_VERSION = 1

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
    S = C // H
    r = r.view(B, T, H, S)
    k = k.view(B, T, H, S)
    v = v.view(B, T, H, S)
    w = w.view(H, S)
    u = u.view(H, S)
    out = torch.zeros((B, T, H, S), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for t in range(T):
                for i in range(S):
                    for j in range(S):
                        for tt in range(0, t+1):
                            ww = u[h,j] if (tt == t) else w[h,j] ** (t - tt - 1)
                            out[b,t,h,i] += r[b,t,h,j] * ww * k[b,tt,h,j] * v[b,tt,h,i]

    return out.view(B, T, C)

def RUN_FORMULA_2(B, T, C, H, r, k, v, w, u):
    S = C // H
    r = r.flatten().contiguous() # BTHS
    k = k.flatten().contiguous() # BTHS
    v = v.flatten().contiguous() # BTHS
    w = w.flatten().contiguous() # HS
    u = u.flatten().contiguous() # HS
    out = torch.zeros(B*T*C, device=DEVICE).contiguous()

    for b in range(B):
        for h in range(H):
            ss = torch.zeros(S*S, device=DEVICE).contiguous()
            for t in range(T):
                ooo = b*H*T*S + t*H*S + h*S
                oo = h*S
                for n in range(S*S):
                    i = ooo + n // S
                    j = ooo + n % S
                    m = oo + n % S
                    x = k[j] * v[i]
                    s = ss[n]
                    out[i] += r[j] * (u[m] * x + s)
                    ss[n] = s * w[m] + x

    return out.view(B, T, C)

######################################################################################################
# Load the CUDA kernel
######################################################################################################

# B = 4
# T = 7
# C = 32
# H = 16

B = 2
T = 4
C = 9
H = 3

# B = 1
# T = 2
# C = 2
# H = 2

SSS = C*C//(H*H)
from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda_v{CUDA_KERNEL_VERSION}.cu"],
                  verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DSSS={SSS}"])

class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
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

def CHECK_PYTORCH():

    set_seed(42)
    with torch.no_grad():
        r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        w = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device=DEVICE).uniform_(-1, 1)

    # y0 = RUN_FORMULA_1(B, T, C, H, r, k, v, w, u)
    y0 = RUN_FORMULA_2(B, T, C, H, r, k, v, w, u)
    print('result', val(y0), '\n\n')

    # y1 = RUN_FORMULA_2(B, T, C, H, r, k, v, w, u)
    y1 = RUN_CUDA(B, T, C, H, r, k, v, w, u)
    print('result', val(y1), '\n\n')

    print('--> correct =', torch.allclose(y0, y1),
          ', err ratio =', get_err_ratio(y0, y1))

if __name__ == "__main__":
    print('\n\nVerify pytorch...')
    CHECK_PYTORCH()
