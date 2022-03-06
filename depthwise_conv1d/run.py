import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

######################################################################################################
# From https://github.com/BlinkDL/RWKV-CUDA
# On GTX1070 mobile:
# pytorch = fwd 94ms bwd 529ms
# CUDA kernel v0 = fwd 45ms bwd 84ms (simple)
# CUDA kernel v1 = fwd 17ms bwd 43ms (shared memory)
# CUDA kernel v2 = fwd 13ms bwd 31ms (float4)
# CUDA kernel v3 = fwd 3.4ms bwd 23ms (B-group)
######################################################################################################

CUDA_KERNEL_VERSION = 3  # CUDA kernel version = 0,1,2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

######################################################################################################
# The formula:
# w.shape = (C, T)
# k.shape = (B, C, T)
# out.shape = (B, C, T)
# out[b][c][t] = sum_u{ w[c][(T-1)-(t-u)] * k[b][c][u] }
######################################################################################################


def RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps):
    # this is the formula (very slow)
    out = torch.empty((B, C, T), device='cuda')
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = eps
                for u in range(0, t+1):
                    s += w[c][(T-1)-(t-u)] * k[b][c][u]
                out[b][c][t] = s
    return out


def RUN_PYTORCH(w, k, B, C, T, eps):
    # this shall equal the formula
    return F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w.unsqueeze(1), groups=C) + eps


######################################################################################################
# Load the CUDA kernel
######################################################################################################

T_MAX = 768
B_GROUP_FORWARD = 8
B_GROUP_BACKWARD = 2

timex_cuda = load(name="timex", sources=["cuda/timex_op.cpp", "cuda/timex_cuda_v" + str(CUDA_KERNEL_VERSION) + ".cu"],
                  verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}'], extra_cflags=['/wd4624'])


# we call it the "TimeX" operator because it's used for time-mixing in my RWKV language model
class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.forward(w, k, wk, eps, B, C, T)
        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        # actually pytorch will do gw.sum(dim=0) but we will do it anyway just to be safe
        return (gw.sum(dim=0), gk, None, None, None, None)


def RUN_CUDA(w, k, B, C, T, eps):
    return TimeX.apply(w.cuda(), k.cuda(), B, C, T, eps)


######################################################################################################
# Check correctness & speed benchmark
######################################################################################################

def CHECK_PYTORCH():
    B = 3
    C = 5
    T = 11
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    r0 = RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps)
    r1 = RUN_PYTORCH(w, k, B, C, T, eps)

    print('--> pytorch correct =', torch.allclose(r0, r1),
          ', err ratio =', get_err_ratio(r0, r1))


def CHECK_CUDA(silent=False):
    B = 32
    C = 768
    T = 768
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    # check forward

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r1 = RUN_PYTORCH(w, k, B, C, T, eps)
    if not silent:
        print('pytorch forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r2 = RUN_CUDA(w, k, B, C, T, eps)
    if not silent:
        print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

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
    gk1 = k.grad.data.clone()

    w.grad.data.zero_()
    k.grad.data.zero_()

    loss2 = ((r2 * r2) - torch.tanh(r2)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss2.backward()
    if not silent:
        print('CUDA backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw2 = w.grad.data.clone()
    gk2 = k.grad.data.clone()

    print('--> bwd gradW correct =', torch.allclose(gw1, gw2),
          ', err ratio =', get_err_ratio(gw1, gw2))
    print('--> bwd gradK correct =', torch.allclose(gk1, gk2),
          ', err ratio =', get_err_ratio(gk1, gk2))


if __name__ == "__main__":
    print('\n\nVerify pytorch...')
    CHECK_PYTORCH()
    print('\n\nCUDA warmup...')
    CHECK_CUDA(silent=True)  # warmup
    CHECK_CUDA(silent=True)  # warmup
    print('\n\nCUDA benchmark...')
    CHECK_CUDA(silent=False)  # benchmark
