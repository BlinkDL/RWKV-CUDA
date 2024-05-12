import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import numpy as np
from math import exp
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

######################################################################################################
# From https://github.com/BlinkDL/RWKV-CUDA
######################################################################################################

CUDA_KERNEL_VERSION = 2  # CUDA kernel version = 0,1,2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

######################################################################################################
# u.shape = (C)
# w.shape = (C)
# k.shape = (B, T, C)
# v.shape = (B, T, C)
# out.shape = (B, T, C)
#
# for a giving b and c:
# out[0] = v0
# out[1] = (e^{k0} v0 + e^{u+k1} v1) / (e^{k0} + e^{u+k1})
# out[2] = (e^{w+k0} v0 + e^{k1} v1 + e^{u+k2} v2) / (e^{w+k0} + e^{k1} + e^{u+k2})
# ...
######################################################################################################


def RUN_FORMULA_VERY_SLOW(B, T, C, w, u, k, v):
    w = -torch.exp(w)
    out = torch.empty((B, T, C), device='cuda')
    for b in range(B):
        for c in range(C):
            out[b][0][c] = v[b][0][c]
            for t in range(1, T):
                p = 0
                q = 0
                for s in range(t+1):
                    if s == t:
                        ek = exp(k[b][s][c] + u[c])
                    else:
                        ek = exp(k[b][s][c] + w[c]*(t-s-1))
                    p += ek * v[b][s][c]
                    q += ek
                out[b][t][c] = p / q
    return out


def RUN_PYTORCH(B, T, C, w, u, k, v, time_curve):
    # this shall equal the formula

    ek = torch.exp(k.transpose(1,2))
    ekv = ek * v.transpose(1,2)

    time_w = torch.cat([torch.exp(w).unsqueeze(1) * time_curve, u.unsqueeze(1)], dim=-1)
    w = torch.exp(time_w).unsqueeze(1)

    wkv = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(ekv), w, groups=C)
    wk = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(ek), w, groups=C)

    return (wkv / wk).transpose(1,2)


######################################################################################################
# Load the CUDA kernel
######################################################################################################

from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", f"cuda/wkv_cuda_v{CUDA_KERNEL_VERSION}.cu"],
                  verbose=True, extra_cuda_cflags=["--O3", "-xhip", "--hipstdpar"], extra_cflags=['/wd4624'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        w = -torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda')
        gu = torch.zeros((B, C), device='cuda')
        gk = torch.zeros((B, T, C), device='cuda')
        gv = torch.zeros((B, T, C), device='cuda')
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


######################################################################################################
# Check correctness & speed benchmark
######################################################################################################

def CHECK_PYTORCH():
    B = 3
    T = 11
    C = 5

    set_seed(42)
    with torch.no_grad():
        w = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)

    r0 = RUN_FORMULA_VERY_SLOW(B, T, C, w, u, k, v)

    time_curve = torch.tensor([-(T - 2 - i) for i in range(T-1)]).unsqueeze(0).cuda()
    r1 = RUN_PYTORCH(B, T, C, w, u, k, v, time_curve)

    print('--> pytorch correct =', torch.allclose(r0, r1),
          ', err ratio =', get_err_ratio(r0, r1))

def CHECK_CUDA(silent=False):
    # B = 16
    # T = 1024 # T <= 1024 for current kernel
    # C = 2048 # only CUDA_KERNEL_VERSION >=2 can support C > 1024 

    B = 32
    T = 768 # T <= 1024 for current kernel
    C = 768

    set_seed(42)
    with torch.no_grad():

        ###### extreme values: pytorch will overflow, but CUDA is fine ######
        # w = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-10, 10)
        # u = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-10, 10)
        # k = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1000, 1000)
        # v = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-10, 10)

        # large values
        # w = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-10, 10)
        # u = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-10, 10)
        # k = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-40, 40)
        # v = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)

        # usual values
        w = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-1, 1)
        u = torch.zeros(C, requires_grad=True, device='cuda').uniform_(-1, 1)
        k = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)
        v = torch.zeros(B, T, C, requires_grad=True, device='cuda').uniform_(-1, 1)

    time_curve = torch.tensor([-(T - 2 - i) for i in range(T-1)]).unsqueeze(0).cuda()

    # check forward

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r1 = RUN_PYTORCH(B, T, C, w, u, k, v, time_curve)
    if not silent:
        print('pytorch forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r2 = RUN_CUDA(B, T, C, w, u, k, v)
    if not silent:
        print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    if not silent:
        print('--> fwd correct =', torch.allclose(r1, r2),
            ', err ratio =', get_err_ratio(r1, r2))

    # check backward

    # a strange loss for better verification
    loss1 = ((r1 * r1) - torch.tanh(r1)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss1.backward()
    if not silent:
        print('pytorch backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw1 = w.grad.data.clone()
    gu1 = u.grad.data.clone()
    gk1 = k.grad.data.clone()
    gv1 = v.grad.data.clone()

    w.grad.data.zero_()
    u.grad.data.zero_()
    k.grad.data.zero_()
    v.grad.data.zero_()

    loss2 = ((r2 * r2) - torch.tanh(r2)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss2.backward()
    if not silent:
        print('CUDA backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw2 = w.grad.data.clone()
    gu2 = u.grad.data.clone()
    gk2 = k.grad.data.clone()
    gv2 = v.grad.data.clone()
    if not silent:
        print('\ngw\n',gw2.cpu().squeeze().numpy(), '\ngu\n',gu2.cpu().squeeze().numpy(), '\ngk\n',gk2.cpu().squeeze().numpy(), '\ngv\n',gv2.cpu().squeeze().numpy())
        print('--> bwd gradW correct =', torch.allclose(gw1, gw2),
            ', err ratio =', get_err_ratio(gw1, gw2))
        print('--> bwd gradU correct =', torch.allclose(gu1, gu2),
            ', err ratio =', get_err_ratio(gu1, gu2))
        print('--> bwd gradK correct =', torch.allclose(gk1, gk2),
            ', err ratio =', get_err_ratio(gk1, gk2))
        print('--> bwd gradV correct =', torch.allclose(gv1, gv2),
            ', err ratio =', get_err_ratio(gv1, gv2))

if __name__ == "__main__":
    print('\n\nVerify pytorch...')
    CHECK_PYTORCH()
    print('\n\nCUDA warmup...')
    CHECK_CUDA(silent=True)  # warmup
    CHECK_CUDA(silent=True)  # warmup
    print('\n\nCUDA benchmark...')
    CHECK_CUDA(silent=False)  # benchmark
