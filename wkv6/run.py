import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

'''

CHECK_REF = False

if CHECK_REF:
    # B = 10
    # T = 4
    # C = 4
    # HEAD_SIZE = 4
    # H = C // HEAD_SIZE

    B = 2
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
    T = 64 # 64 works, but 128 does not work
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

run_rocm = True

if run_rocm:
    wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda_v1b2.cu"],
        verbose=True, extra_cuda_cflags=["-O3", "--hipstdpar", "-xhip", f"-D_N_={HEAD_SIZE}"])
else:
    wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda_v1b2.cu"],
        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
        
class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
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
            ew = (-torch.exp(w.float())).contiguous()
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
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
            r, k, v, eew, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            gw = torch.sum(gw, 0).view(H, C//H)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################
# CUDA Kernel
########################################################################################################

if run_rocm:
    wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda_{CUDA_KERNEL_VERSION}.cu"],
                verbose=True, extra_cuda_cflags=["-O3", "--hipstdpar", "-xhip", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
    
else:
    wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda_{CUDA_KERNEL_VERSION}.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}"])
    
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
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
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
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
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw = torch.zeros((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16).contiguous() # .uniform_(-1, 1)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

######################################################################################################
# Original pytorch version (requires w & u to be constant within each head)
######################################################################################################

class RUN_TORCH(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    def forward(self, B, T, C, H, r, k, v, w, u):
        N = C // H
        r = r.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        w = torch.exp(-torch.exp(w))
        w = w.view(B, T, H, N)
        u = u.view(H, N)
        out = torch.zeros((B, T, H, N), device=DEVICE)

        for b in range(B):
            for h in range(H):
                for t in range(T):
                    for i in range(N):
                        for j in range(N):
                            for tt in range(t+1):
                                ww = 1
                                if tt == t:
                                    ww = u[h,j]
                                elif tt+1 < t:
                                    for zz in range(tt+1, t):
                                        ww = ww * w[b,zz,h,j]
                                out[b,t,h,i] += r[b,t,h,j] * ww * k[b,tt,h,j] * v[b,tt,h,i]

        # for b in range(B):
        #     for h in range(H):
        #         state = torch.zeros((N,N), device=DEVICE).contiguous()
        #         for t in range(T):
        #             for i in range(N):
        #                 for j in range(N):
        #                     x = k[b,t,h,j] * v[b,t,h,i]
        #                     s = state[i,j]
        #                     out[b,t,h,i] += r[b,t,h,j] * (u[h,j] * x + s)
        #                     state[i,j] = s * w[b,t,h,j] + x

        return out.view(B, T, C)

def TORCH_BACKWARD(B, T, C, H, gy, r, k, v, __w, u):
    N = C // H
    gy = gy.view(B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    _w = -torch.exp(__w).view(B, T, H, N)
    u = u.view(H, N)
    w = torch.exp(_w)

    gr = torch.zeros((B, T, H, N), device=DEVICE).contiguous()
    gk = torch.zeros((B, T, H, N), device=DEVICE).contiguous()
    gv = torch.zeros((B, T, H, N), device=DEVICE).contiguous()
    gw = torch.zeros((B, T, H, N), device=DEVICE).contiguous()
    gu = torch.zeros((H, N), device=DEVICE).contiguous()
    buf = torch.zeros((T-2, N), device=DEVICE).contiguous()

    for b in range(B):
        for h in range(H):
            for i in range(N):

                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T):
                    for j in range(N):
                        s = state[j]
                        x = k[b,t,h,i] * v[b,t,h,j]

                        gr[b,t,h,i] += (u[h,i] * x + s) * gy[b,t,h,j]
                        gu[h,i] += r[b,t,h,i] * x * gy[b,t,h,j]

                        state[j] = s * w[b,t,h,i] + x

                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T-2):
                    for j in range(N):
                        x = k[b,t,h,i] * v[b,t,h,j]
                        s = state[j] * w[b,t,h,i] + x
                        buf[t,j] = s
                        state[j] = s
                        # print(f"TORCH i {i} t {t} j {j} buf {s}")
                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T-1,1,-1):
                    sum = 0
                    for j in range(N):
                        x = r[b,t,h,i] * gy[b,t,h,j]
                        s = state[j] * w[b,t,h,i] + x
                        sum += s * buf[t-2,j]
                        state[j] = s
                        # print(f"TORCH i {i} t {t} j {j} buf {buf[t-2,j]} tmp {s}")
                    gw[b,t-1,h,i] = sum * w[b,t-1,h,i] * _w[b,t-1,h,i]

                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T-1,-1,-1):
                    for j in range(N):
                        x = r[b,t,h,i] * gy[b,t,h,j]
                        s = state[j]
                        gk[b,t,h,i] += v[b,t,h,j] * (u[h,i] * x + s)
                        state[j] = s * w[b,t,h,i] + x

                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T-1,-1,-1):
                    for j in range(N):
                        x = gy[b,t,h,i] * r[b,t,h,j]
                        s = state[j]
                        gv[b,t,h,i] += k[b,t,h,j] * (u[h,j] * x + s)
                        state[j] = s * w[b,t,h,j] + x

    return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw.view(B, T, C), gu.view(H, N)

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
    
        if CHECK_REF:
            x = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            w = torch.empty(1, 1, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            ww1 = torch.empty(C, 3, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            ww2 = torch.empty(3, C, device=DEVICE).uniform_(-1, 1).to(dtype=DTYPE).float()
            r = r.float()
            k = k.float()
            v = v.float()
            u = u.float()
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

    print(f'B={B} T={T} C={C} HEAD_SIZE={HEAD_SIZE}')
    print('[original torch (const w & u within a head)] vs [current cuda]')

    # collect fp32 reference values

    if CHECK_REF:
        rwkv5_torch = RUN_TORCH()
        ww = w + ((x @ ww1) @ ww2)
        ww.retain_grad()
        y = rwkv5_torch.forward(B, T, C, H, r, k, v, ww, u)
    else:
        y = RUN_CUDA_5(B, T, C, H, r, k, v, w, u)
    
    yy = y.clone().detach().requires_grad_(True)
    LOSS(yy).backward()
    gy = yy.grad.data.clone()

    # print('r', val(r))
    # print('k', val(k))
    # print('v', val(v))
    # print('w', val(ww))
    # print('u', val(u))

    LOSS(y).backward()
    gr = r.grad.data.clone()
    gk = k.grad.data.clone()
    gv = v.grad.data.clone()
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()

    if CHECK_REF:
        gx = x.grad.data.clone()
        gww = ww.grad.data.clone()
        gww1 = ww1.grad.data.clone()
        gww2 = ww2.grad.data.clone()

    if CHECK_REF:
        gr2, gk2, gv2, gw2, gu2 = TORCH_BACKWARD(B, T, C, H, gy, r, k, v, ww, u)
        print('--> ref g_r correct =', torch.allclose(gr, gr2), ', err ratio =', get_err_ratio(gr, gr2))
        print('--> ref g_k correct =', torch.allclose(gk, gk2), ', err ratio =', get_err_ratio(gk, gk2))
        print('--> ref g_v correct =', torch.allclose(gv, gv2), ', err ratio =', get_err_ratio(gv, gv2))
        print('--> ref g_w correct =', torch.allclose(gww, gw2), ', err ratio =', get_err_ratio(gww, gw2))
        print('--> ref g_u correct =', torch.allclose(gu, gu2), ', err ratio =', get_err_ratio(gu, gu2))

        # print('gy', val(gy))
        # print('gww', val(gww))
        # print('gw2', val(gw2))

    # exit(0)

    with torch.no_grad():
        r.grad.data.zero_()
        k.grad.data.zero_()
        v.grad.data.zero_()
        w.grad.data.zero_()
        u.grad.data.zero_()

        r = r.to(dtype=DTYPE)
        k = k.to(dtype=DTYPE)
        v = v.to(dtype=DTYPE)
        w = w.to(dtype=DTYPE)
        u = u.to(dtype=DTYPE)
        if CHECK_REF:
            x.grad.data.zero_()
            ww1.grad.data.zero_()
            ww2.grad.data.zero_()

            x = x.to(dtype=DTYPE)
            ww1 = ww1.to(dtype=DTYPE)
            ww2 = ww2.to(dtype=DTYPE)
    r.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    w.requires_grad_()
    u.requires_grad_()
    if CHECK_REF:
        x.requires_grad_()
        ww1.requires_grad_()
        ww2.requires_grad_()    

    if CHECK_REF:
        ww = w + ((x @ ww1) @ ww2)
    else:
        ww = w.reshape(1,1,C).repeat(B,T,1)
    y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, u)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1 = RUN_CUDA(B, T, C, H, r, k, v, ww, u)
    print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=2))
    print('!!! CUDA correct =', torch.allclose(y.float(), y1.float()), ', err ratio =', get_err_ratio(y.float(), y1.float()))

    if CHECK_REF:
        print('y', val(y))
        print('y_cuda', val(y1))
        print('PYTHON fwd', val(y))
        print('CUDA fwd', val(y1))

    # exit(0)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        LOSS(y1).backward()
    print('CUDA backward\n', prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=2))
    
    gr1 = r.grad.data.clone()
    gk1 = k.grad.data.clone()
    gv1 = v.grad.data.clone()
    gw1 = w.grad.data.clone()
    gu1 = u.grad.data.clone()
    print('!!! CUDA g_r correct =', torch.allclose(gr.float(), gr1.float()), ', err ratio =', get_err_ratio(gr, gr1.float()))
    print('!!! CUDA g_k correct =', torch.allclose(gk.float(), gk1.float()), ', err ratio =', get_err_ratio(gk, gk1.float()))
    print('!!! CUDA g_v correct =', torch.allclose(gv.float(), gv1.float()), ', err ratio =', get_err_ratio(gv, gv1.float()))
    print('!!! CUDA g_w correct =', torch.allclose(gw.float(), gw1.float()), ', err ratio =', get_err_ratio(gw, gw1.float()))
    print('!!! CUDA g_u correct =', torch.allclose(gu.float(), gu1.float()), ', err ratio =', get_err_ratio(gu, gu1.float()))
    if CHECK_REF:
        gx1 = x.grad.data.clone()
        gww11 = ww1.grad.data.clone()
        gww21 = ww2.grad.data.clone()
        print('!!! CUDA g_x correct =', torch.allclose(gx.float(), gx1.float()), ', err ratio =', get_err_ratio(gx, gx1.float()))
        print('!!! CUDA g_ww1 correct =', torch.allclose(gww1.float(), gww11.float()), ', err ratio =', get_err_ratio(gww1, gww11.float()))
        print('!!! CUDA g_ww2 correct =', torch.allclose(gww2.float(), gww21.float()), ', err ratio =', get_err_ratio(gww2, gww21.float()))
    print('PYTHON bwd', val(gw))
    print('CUDA bwd', val(gw1))

CHECK_BACKWARD()
