import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import numpy as np
from math import exp
np.set_printoptions(precision=4, suppress=True, linewidth=200)
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

DTYPE = torch.bfloat16

DEVICE = 'cuda'
CUDA_KERNEL_VERSION = 'v1'

'''
Self CUDA

cd /mnt/program/_RWKV_/_REF_/RWKV-CUDA/wkv6_state
python run.py

'''

CHECK_REF = True
CHECK_BWD = True

if CHECK_REF:
    # B = 10
    # T = 7
    # C = 8
    # HEAD_SIZE = 4
    # H = C // HEAD_SIZE

    B = 3
    T = 7
    C = 8
    HEAD_SIZE = 4
    H = C // HEAD_SIZE

    # B = 1
    # T = 3
    # C = 4
    # HEAD_SIZE = 4
    # H = C // HEAD_SIZE
else:
    # B = 8
    # T = 64
    # C = 64
    # HEAD_SIZE = 8
    # H = C // HEAD_SIZE

    B = 8
    T = 4096 # 64 works, but 128 does not work
    C = 4096
    HEAD_SIZE = 64
    H = C // HEAD_SIZE

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def val(x):
    return x.detach().float().cpu().numpy()

########################################################################################################
# CUDA Kernel
########################################################################################################

run_rocm = True

if run_rocm:
    wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda_{CUDA_KERNEL_VERSION}.cu"],
        verbose=True, extra_cuda_cflags=["-O3", "--hipstdpar", "-xhip", "--hip-link", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
else:
    wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda_{CUDA_KERNEL_VERSION}.cu"],
        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
    

class WKV_6STATE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u, s):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert s.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert s.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            # ew = torch.sigmoid(-w.float()).contiguous()
            ctx.save_for_backward(r, k, v, ew, u, s)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6state_cuda.forward(B, T, C, H, r, k, v, ew, u, s, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u, s = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6state_cuda.backward(B, T, C, H, r, k, v, ew, u, s, gy, gr, gk, gv, gw, gu, gs)
            gu = torch.sum(gu, 0).view(H, C//H)
            gs = torch.sum(gs, 0).view(H, C//H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu, gs)

def RUN_CUDA(B, T, C, H, r, k, v, w, u, s):
    return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)

######################################################################################################
# reference pytorch version (very very slow - otherwise pytorch can't autograd)
######################################################################################################

class RUN_TORCH(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    def forward(self, B, T, C, H, r, k, v, w, u, _s):
        N = C // H
        r = r.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        w = torch.exp(-torch.exp(w))
        # w = torch.sigmoid(-w)
        w = w.view(B, T, H, N)
        u = u.view(H, N)
        out = torch.zeros((B, T, H, N), device=DEVICE)

        for b in range(B):
            for h in range(H):
                for t in range(T):
                    for i in range(N):
                        for j in range(N):
                            for tt in range(-1, t+1):
                                ww = 1
                                if tt == t:
                                    ww = u[h,j]
                                elif tt+1 < t:
                                    for zz in range(tt+1, t):
                                        ww = ww * w[b,zz,h,j]
                                if tt == -1:
                                    out[b,t,h,i] += r[b,t,h,j] * ww * _s[h,i,j]
                                else:
                                    out[b,t,h,i] += r[b,t,h,j] * ww * k[b,tt,h,j] * v[b,tt,h,i]

        return out.view(B, T, C)

######################################################################################################
# Check correctness
######################################################################################################

def CHECK_BACKWARD():
    def LOSS(y): # a strange loss for better verification
        return ((y * y) - torch.tanh(y)).sum()

    set_seed(42)
    with torch.no_grad():
        r = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE)
        k = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE)
        v = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE)
        u = torch.empty(H, HEAD_SIZE, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE)
        s = torch.empty(H, HEAD_SIZE, HEAD_SIZE, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE)
    
        if CHECK_REF:
            x = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            w = torch.empty(1, 1, C, device=DEVICE).uniform_(-3, 1).to(dtype=DTYPE).float()
            ww1 = torch.empty(C, 3, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            ww2 = torch.empty(3, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            r = r.float()
            k = k.float()
            v = v.float()
            u = u.float()
            s = s.float()
        else:
            w = torch.empty(H, HEAD_SIZE, device=DEVICE).uniform_(-8, 1).to(dtype=DTYPE)

    r.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    w.requires_grad_()
    if CHECK_REF:
        x.requires_grad_()
        ww1.requires_grad_()
        ww2.requires_grad_()
    u.requires_grad_()
    s.requires_grad_()

    print(f'B={B} T={T} C={C} HEAD_SIZE={HEAD_SIZE}')
    print('[original torch] vs [current cuda]')

    # collect fp32 reference values

    if CHECK_REF:
        rwkv_torch = RUN_TORCH()
        ww = w + ((x @ ww1) @ ww2)
        ww.retain_grad()
        y = rwkv_torch.forward(B, T, C, H, r, k, v, ww, u, s)
    # else:
    #     y = RUN_CUDA_5(B, T, C, H, r, k, v, w, u)
    
    if CHECK_BWD:
        yy = y.clone().detach().requires_grad_(True)
        LOSS(yy).backward()
        gy = yy.grad.data.clone()

        print('r', val(r))
        print('k', val(k))
        print('v', val(v))
        print('w', val(ww))
        print('u', val(u))
        print('s', val(s))

        LOSS(y).backward()
        gr = r.grad.data.clone()
        gk = k.grad.data.clone()
        gv = v.grad.data.clone()
        gw = w.grad.data.clone()
        gu = u.grad.data.clone()
        gs = s.grad.data.clone()

        if CHECK_REF:
            gx = x.grad.data.clone()
            gww = ww.grad.data.clone()
            gww1 = ww1.grad.data.clone()
            gww2 = ww2.grad.data.clone()

    with torch.no_grad():
        if CHECK_BWD:
            r.grad.data.zero_()
            k.grad.data.zero_()
            v.grad.data.zero_()
            w.grad.data.zero_()
            u.grad.data.zero_()
            s.grad.data.zero_()

        r = r.to(dtype=DTYPE)
        k = k.to(dtype=DTYPE)
        v = v.to(dtype=DTYPE)
        w = w.to(dtype=DTYPE)
        u = u.to(dtype=DTYPE)
        s = s.to(dtype=DTYPE)
        if CHECK_REF:
            if CHECK_BWD:
                x.grad.data.zero_()
                ww1.grad.data.zero_()
                ww2.grad.data.zero_()

            x = x.to(dtype=DTYPE)
            ww1 = ww1.to(dtype=DTYPE)
            ww2 = ww2.to(dtype=DTYPE)
    if CHECK_BWD:
        r.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        w.requires_grad_()
        u.requires_grad_()
        s.requires_grad_()
        if CHECK_REF:
            x.requires_grad_()
            ww1.requires_grad_()
            ww2.requires_grad_()

    if CHECK_REF:
        ww = w + ((x @ ww1) @ ww2)
    else:
        ww = w.reshape(1,1,C).repeat(B,T,1)
    y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, u, s)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, u, s)
    print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=2))
    print('!!! CUDA correct =', torch.allclose(y.float(), y1.float()), ', err ratio =', get_err_ratio(y.float(), y1.float()))

    # if CHECK_REF:
    #     print('y', val(y))
    #     print('y_cuda', val(y1))
    #     print('PYTHON fwd', val(y))
    #     print('CUDA fwd', val(y1))

    # exit(0)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        LOSS(y1).backward()
    print('CUDA backward\n', prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=2))
    
    gr1 = r.grad.data.clone()
    gk1 = k.grad.data.clone()
    gv1 = v.grad.data.clone()
    gw1 = w.grad.data.clone()
    gu1 = u.grad.data.clone()
    gs1 = s.grad.data.clone()
    print('!!! CUDA g_r correct =', torch.allclose(gr.float(), gr1.float()), ', err ratio =', get_err_ratio(gr, gr1.float()))
    print('!!! CUDA g_k correct =', torch.allclose(gk.float(), gk1.float()), ', err ratio =', get_err_ratio(gk, gk1.float()))
    print('!!! CUDA g_v correct =', torch.allclose(gv.float(), gv1.float()), ', err ratio =', get_err_ratio(gv, gv1.float()))
    print('!!! CUDA g_w correct =', torch.allclose(gw.float(), gw1.float()), ', err ratio =', get_err_ratio(gw, gw1.float()))
    print('!!! CUDA g_u correct =', torch.allclose(gu.float(), gu1.float()), ', err ratio =', get_err_ratio(gu, gu1.float()))
    print('!!! CUDA g_s correct =', torch.allclose(gs.float(), gs1.float()), ', err ratio =', get_err_ratio(gs, gs1.float()))

    # print('PYTHON bwd', val(gs))
    # print('CUDA bwd', val(gs1))

    if CHECK_REF:
        gx1 = x.grad.data.clone()
        gww11 = ww1.grad.data.clone()
        gww21 = ww2.grad.data.clone()
        print('!!! CUDA g_x correct =', torch.allclose(gx.float(), gx1.float()), ', err ratio =', get_err_ratio(gx, gx1.float()))
        print('!!! CUDA g_ww1 correct =', torch.allclose(gww1.float(), gww11.float()), ', err ratio =', get_err_ratio(gww1, gww11.float()))
        print('!!! CUDA g_ww2 correct =', torch.allclose(gww2.float(), gww21.float()), ', err ratio =', get_err_ratio(gww2, gww21.float()))

    # print('PYTHON bwd', val(gw))
    # print('CUDA bwd', val(gw1))


CHECK_BACKWARD()
