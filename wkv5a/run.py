import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from math import exp
import torch, os, sys, argparse
from torch.utils.cpp_extension import load
# turn off TF32 for higher accuracy
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
np.set_printoptions(precision=4, suppress=True, linewidth=200)
parser = argparse.ArgumentParser()
parser.add_argument("--job", type=int)
parser.add_argument("--algo", type=int)
args = parser.parse_args()

DEVICE = 'cuda'
DTYPE = torch.float32
CUDA_KERNEL_VERSION = 'v1a'

JOB = args.job
ALGO = args.algo
print('job', JOB, 'algo', ALGO, 'CUDA_KERNEL_VERSION', CUDA_KERNEL_VERSION)
'''
cd /fsx/BlinkDL/CODE/_PUBLIC_/RWKV-CUDA/wkv5_v2
python run.py --job 0 --algo 1
python run.py --job 0 --algo 2

python run.py --job 1 --algo 1
python run.py --job 1 --algo 2

python run.py --job 3 --algo 2
python run.py --job 4 --algo 2

python run.py --job 1 --algo 2
python run.py --job 4 --algo 2

'''
if JOB == 0:
    B = 1
    T = 4
    C = 4
    HEAD_SIZE = 4

    # B = 1
    # T = 1
    # C = 1
    # HEAD_SIZE = 1

elif JOB == 1:
    B = 3
    T = 5
    C = 8
    HEAD_SIZE = 4

elif JOB == 2:
    B = 13
    T = 513
    C = 17*31
    HEAD_SIZE = 17

elif JOB == 3:
    B = 8
    T = 4096
    C = 4096
    HEAD_SIZE = 64

elif JOB == 4:
    B = 8
    T = 4096
    C = 4096
    HEAD_SIZE = 128

N = HEAD_SIZE
H = C // N

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

def compare(x, y, name=""):
    print('B',B,'T',T,'C',C,'N',N, 'algo', ALGO, f'{name} correct =', torch.allclose(x,y), ', err ratio =', get_err_ratio(x,y))

########################################################################################################
# CUDA Kernel
########################################################################################################

wkv5a_cuda = torch.utils.cpp_extension.load(name="wkv5a", sources=["cuda/wkv5a_op.cpp", f"cuda/wkv5a_cuda_{CUDA_KERNEL_VERSION}.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    
class WKV_5A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w1, u1, w2, u2):
        with torch.no_grad():
            # assert r.dtype == torch.bfloat16
            # assert k.dtype == torch.bfloat16
            # assert v.dtype == torch.bfloat16
            # assert w1.dtype == torch.bfloat16
            # assert u1.dtype == torch.bfloat16
            # assert w2.dtype == torch.bfloat16
            # assert u2.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w1.is_contiguous()
            assert u1.is_contiguous()
            assert w2.is_contiguous()
            assert u2.is_contiguous()
            ew1 = (-torch.exp(w1.float())).contiguous()
            eew1 = (torch.exp(ew1)).contiguous()
            ew2 = (-torch.exp(w2.float())).contiguous()
            eew2 = (torch.exp(ew2)).contiguous()
            ctx.save_for_backward(r, k, v, eew1, ew1, u1, eew2, ew2, u2)
            y = torch.empty((B, T, C), device=r.device, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5a_cuda.forward(B, T, C, H, r, k, v, eew1, u1, eew2, u2, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == DTYPE
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, eew1, ew1, u1, eew2, ew2, u2 = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw1 = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gu1 = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw2 = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gu2 = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5a_cuda.backward(B, T, C, H, r, k, v, eew1, ew1, u1, eew2, ew2, u2, gy, gr, gk, gv, gw1, gu1, gw2, gu2)
            gw1 = torch.sum(gw1, 0).view(H, C//H)
            gu1 = torch.sum(gu1, 0).view(H, C//H)
            gw2 = torch.sum(gw2, 0).view(H, C//H)
            gu2 = torch.sum(gu2, 0).view(H, C//H)
            return (gr, gk, gv, gw1, gu1, gw2, gu2)

def CUDA_1(r, k, v, w1, u1, w2, u2):
    return WKV_5A.apply(r, k, v, w1, u1, w2, u2)


######################################################################################################
# Python version
######################################################################################################

def PYTHON_1(r, k, v, w1, u1, w2=None, u2=None):
    w1 = torch.exp(-torch.exp(w1))
    if w2 != None:
        w2 = torch.exp(-torch.exp(w2))
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    w1 = w1.view(H, N)
    u1 = u1.view(H, N)
    if w2 != None:
        w2 = w2.view(H, N)
        u2 = u2.view(H, N)
    out = torch.zeros((B, T, H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for t in range(T):
                for i in range(N):
                    for j in range(N):
                        for tt in range(t+1):
                            if w2 != None:
                                ww = (u1[h,j]+u2[h,i]) if (tt == t) else (w1[h,j]*w2[h,i]) ** (t - tt - 1)
                            else:
                                ww = u1[h,j] if (tt == t) else w1[h,j] ** (t - tt - 1)
                            out[b,t,h,i] += r[b,t,h,j] * ww * k[b,tt,h,j] * v[b,tt,h,i]

    return out.view(B, T, C)

def PYTHON_1_BWD(gy, r, k, v, __w1, u1, __w2=None, u2=None):
    gy = gy.view(B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)

    _w1 = -torch.exp(__w1).view(H, N)
    w1 = torch.exp(_w1)
    u1 = u1.view(H, N)
    if __w2 != None:
        _w2 = -torch.exp(__w2).view(H, N)
        w2 = torch.exp(_w2)
        u2 = u2.view(H, N)

    gr = torch.zeros((B, T, H, N), device=DEVICE)
    gk = torch.zeros((B, T, H, N), device=DEVICE)
    gv = torch.zeros((B, T, H, N), device=DEVICE)
    gw1 = torch.zeros((H, N), device=DEVICE)
    gu1 = torch.zeros((H, N), device=DEVICE)
    if __w2 != None:
        gw2 = torch.zeros((H, N), device=DEVICE)
        gu2 = torch.zeros((H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for i in range(N):
                for t in range(T):
                    for j in range(N):

                        for tt in range(t+1):
                            if __w2 != None:
                                ww = (u1[h,i]+u2[h,j]) if (tt == t) else (w1[h,i]*w2[h,j]) ** (t - tt - 1)
                            else:
                                ww = u1[h,i] if (tt == t) else w1[h,i] ** (t - tt - 1)
                            gr[b,t,h,i] += ww * k[b,tt,h,i] * v[b,tt,h,j] * gy[b,t,h,j]

                        for tt in range(t,T):
                            if __w2 != None:
                                ww = (u1[h,i]+u2[h,j]) if (tt == t) else (w1[h,i]*w2[h,j]) ** (tt - t - 1)
                            else:
                                ww = u1[h,i] if (tt == t) else w1[h,i] ** (tt - t - 1)
                            gk[b,t,h,i] += r[b,tt,h,i] * ww * v[b,t,h,j] * gy[b,tt,h,j]

                            if __w2 != None:
                                ww = (u1[h,j]+u2[h,i]) if (tt == t) else (w1[h,j]*w2[h,i]) ** (tt - t - 1)
                            else:
                                ww = u1[h,j] if (tt == t) else w1[h,j] ** (tt - t - 1)
                            gv[b,t,h,i] += r[b,tt,h,j] * ww * k[b,t,h,j] * gy[b,tt,h,i]

                        gu1[h,i] += r[b,t,h,i] * k[b,t,h,i] * v[b,t,h,j] * gy[b,t,h,j]
                        if __w2 != None:
                            gu2[h,i] += r[b,t,h,j] * k[b,t,h,j] * v[b,t,h,i] * gy[b,t,h,i]

                        for tt in range(t-1):
                            if __w2 != None:
                                ww = (t-tt-1) * _w1[h,i] * ((w1[h,i]*w2[h,j]) ** (t - tt - 1))
                            else:
                                ww = (t-tt-1) * _w1[h,i] * (w1[h,i] ** (t - tt - 1))
                            gw1[h,i] += r[b,t,h,i] * ww * k[b,tt,h,i] * v[b,tt,h,j] * gy[b,t,h,j]

                            if __w2 != None:
                                ww = (t-tt-1) * _w2[h,i] * ((w1[h,j]*w2[h,i]) ** (t - tt - 1))
                                gw2[h,i] += r[b,t,h,j] * ww * k[b,tt,h,j] * v[b,tt,h,i] * gy[b,t,h,i]

    if __w2 != None:
        return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw1, gu1, gw2, gu2
    return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw1, gu1

def PYTHON_2(r, k, v, w1, u1, w2=None, u2=None):
    w1 = torch.exp(-torch.exp(w1))
    if w2 != None:
        w2 = torch.exp(-torch.exp(w2))
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    w1 = w1.view(H, N)
    u1 = u1.view(H, N)
    if w2 != None:
        w2 = w2.view(H, N)
        u2 = u2.view(H, N)
    out = torch.zeros((B, T, H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for i in range(N):
                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T):
                    for j in range(N):
                        x = k[b,t,h,j] * v[b,t,h,i]
                        s = state[j]
                        if w2 != None:
                            out[b,t,h,i] += r[b,t,h,j] * ((u1[h,j]+u2[h,i]) * x + s)
                            state[j] = s * w1[h,j]*w2[h,i] + x # must do this last in python
                        else:
                            out[b,t,h,i] += r[b,t,h,j] * (u1[h,j] * x + s)
                            state[j] = s * w1[h,j] + x # must do this last in python

    return out.view(B, T, C)

def PYTHON_2_BWD(gy, r, k, v, __w1, u1, __w2=None, u2=None):
    gy = gy.view(B, T, H, N)
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)

    _w1 = -torch.exp(__w1).view(H, N)
    w1 = torch.exp(_w1)
    u1 = u1.view(H, N)
    if __w2 != None:
        _w2 = -torch.exp(__w2).view(H, N)
        w2 = torch.exp(_w2)
        u2 = u2.view(H, N)

    gr = torch.zeros((B, T, H, N), device=DEVICE)
    gk = torch.zeros((B, T, H, N), device=DEVICE)
    gv = torch.zeros((B, T, H, N), device=DEVICE)
    gw1 = torch.zeros((H, N), device=DEVICE)
    gu1 = torch.zeros((H, N), device=DEVICE)
    if __w2 != None:
        gw2 = torch.zeros((H, N), device=DEVICE)
        gu2 = torch.zeros((H, N), device=DEVICE)

    for b in range(B):
        for h in range(H):
            for i in range(N):

                state = torch.zeros(N, device=DEVICE).contiguous()
                saaaa = torch.zeros(N, device=DEVICE).contiguous()
                sbbbb = torch.zeros(N, device=DEVICE).contiguous()
                if __w2 != None:
                    scccc = torch.zeros(N, device=DEVICE).contiguous()
                    sdddd = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T):
                    for j in range(N):
                        x = k[b,t,h,i] * v[b,t,h,j]
                        if __w2 != None:
                            x2 = k[b,t,h,j] * v[b,t,h,i]
                        s = state[j]
                        if __w2 != None:
                            gr[b,t,h,i] += gy[b,t,h,j] * ((u1[h,i]+u2[h,j]) * x + s)
                            state[j] = s * (w1[h,i]*w2[h,j]) + x
                        else:
                            gr[b,t,h,i] += gy[b,t,h,j] * (u1[h,i] * x + s)
                            state[j] = s * w1[h,i] + x

                        gu1[h,i] += r[b,t,h,i] * x * gy[b,t,h,j]
                        if __w2 != None:
                            gu2[h,i] += r[b,t,h,j] * x2 * gy[b,t,h,i]

                        if t < T-2:
                            if __w2 != None:
                                saaaa[j] = (w1[h,i]*w2[h,j]) * (saaaa[j] + sbbbb[j] + x)
                                sbbbb[j] = (w1[h,i]*w2[h,j]) * (sbbbb[j] + x)
                            else:
                                saaaa[j] = w1[h,i] * (saaaa[j] + sbbbb[j] + x)
                                sbbbb[j] = w1[h,i] * (sbbbb[j] + x)
                            gw1[h,i] += r[b,t+2,h,i] * _w1[h,i] * saaaa[j] * gy[b,t+2,h,j]
                            if __w2 != None:
                                scccc[j] = (w1[h,j]*w2[h,i]) * (scccc[j] + sdddd[j] + x2)
                                sdddd[j] = (w1[h,j]*w2[h,i]) * (sdddd[j] + x2)
                                gw2[h,i] += r[b,t+2,h,j] * _w2[h,i] * scccc[j] * gy[b,t+2,h,i]


                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T-1,-1,-1):
                    for j in range(N):
                        x = r[b,t,h,i] * gy[b,t,h,j]
                        s = state[j]
                        if __w2 != None:
                            gk[b,t,h,i] += v[b,t,h,j] * ((u1[h,i]+u2[h,j]) * x + s)
                            state[j] = s * (w1[h,i]*w2[h,j]) + x
                        else:
                            gk[b,t,h,i] += v[b,t,h,j] * (u1[h,i] * x + s)
                            state[j] = s * w1[h,i] + x
 
                state = torch.zeros(N, device=DEVICE).contiguous()
                for t in range(T-1,-1,-1):
                    for j in range(N):
                        x = gy[b,t,h,i] * r[b,t,h,j]
                        s = state[j]
                        if __w2 != None:
                            gv[b,t,h,i] += k[b,t,h,j] * ((u1[h,j]+u2[h,i]) * x + s)
                            state[j] = s * (w1[h,j]*w2[h,i]) + x
                        else:
                            gv[b,t,h,i] += k[b,t,h,j] * (u1[h,j] * x + s)
                            state[j] = s * w1[h,j] + x

    if __w2 != None:
        return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw1, gu1, gw2, gu2
    return gr.view(B, T, C), gk.view(B, T, C), gv.view(B, T, C), gw1, gu1

######################################################################################################
# Forward
######################################################################################################

set_seed(42)
with torch.no_grad():
    ll = -8
    r = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
    k = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
    v = torch.zeros(B, T, C, requires_grad=True, device=DEVICE).uniform_(-1, 1)
    w = torch.zeros(H, N, requires_grad=True, device=DEVICE).uniform_(ll, 1)
    u = torch.zeros(H, N, requires_grad=True, device=DEVICE).uniform_(-1, 1)
    w1 = torch.zeros(H, N, requires_grad=True, device=DEVICE).uniform_(ll, 1)
    u1 = torch.zeros(H, N, requires_grad=True, device=DEVICE).uniform_(-1, 1)
    w2 = torch.zeros(H, N, requires_grad=True, device=DEVICE).uniform_(ll, 1)
    u2 = torch.zeros(H, N, requires_grad=True, device=DEVICE).uniform_(-1, 1)

    # print(f'r\n{val(r)}\n')
    # print(f'k\n{val(k)}\n')
    # print(f'v\n{val(v)}\n')
    # print(f'w1\n{val(w1)}\n')
    # print(f'u1\n{val(u1)}\n')
    # print(f'w2\n{val(w2)}\n')
    # print(f'u2\n{val(u2)}\n')

def LOSS(y): # a strange loss for better verification
    return ((y * y) - torch.tanh(y)).sum()

if JOB >= 3:
    y1 = CUDA_1(r, k, v, w1, u1, w2, u2)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1 = CUDA_1(r, k, v, w1, u1, w2, u2)
    print('CUDA forward\n', prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=2))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        LOSS(y1).backward()
    print('CUDA backward\n', prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=2))

    exit(0)

# f_list = [PYTHON_1, PYTHON_2]
f_list = [PYTHON_1, CUDA_1]

y_list = []
for f in f_list:
    if ALGO == 1:
        y = f(r, k, v, w, u)
    elif ALGO == 2:
        y = f(r, k, v, w1, u1, w2, u2)
    if JOB == 0:
        print(f'result\n{val(y)}\n')
    y_list.append(y)

for i in range(1, len(y_list)):
    compare(y_list[0], y_list[i], 'fwd')

# exit(0)

######################################################################################################
# Backward
######################################################################################################

yy = y_list[0].clone().detach().requires_grad_(True)
LOSS(yy).backward()
gy = yy.grad.data.clone()

LOSS(y_list[0]).backward()
gr = r.grad.data.clone()
gk = k.grad.data.clone()
gv = v.grad.data.clone()
if ALGO == 1:
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()
elif ALGO == 2:
    gw1 = w1.grad.data.clone()
    gw2 = w2.grad.data.clone()
    gu1 = u1.grad.data.clone()
    gu2 = u2.grad.data.clone()

# print(f'g_r\n{val(gr)}\n')
# print(f'g_k\n{val(gk)}\n')
# print(f'g_v\n{val(gv)}\n')
# print(f'g_w1\n{val(gw1)}\n')
# print(f'g_u1\n{val(gu1)}\n')
# print(f'g_w2\n{val(gw2)}\n')
# print(f'g_u2\n{val(gu2)}\n')

r.grad.data.zero_()
k.grad.data.zero_()
v.grad.data.zero_()
if ALGO == 1:
    w.grad.data.zero_()
    u.grad.data.zero_()
elif ALGO == 2:
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    u1.grad.data.zero_()
    u2.grad.data.zero_()

if ALGO == 1:
    gr1, gk1, gv1, gw1, gu1 = PYTHON_1_BWD(gy, r, k, v, w, u)
    compare(gr, gr1, 'g_r')
    compare(gk, gk1, 'g_k')
    compare(gv, gv1, 'g_v')
    compare(gw, gw1, 'g_w')
    compare(gu, gu1, 'g_u')

    gr1, gk1, gv1, gw1, gu1 = PYTHON_2_BWD(gy, r, k, v, w, u)
    compare(gr, gr1, 'g_r')
    compare(gk, gk1, 'g_k')
    compare(gv, gv1, 'g_v')
    compare(gw, gw1, 'g_w')
    compare(gu, gu1, 'g_u')    
else:
    LOSS(y_list[1]).backward()
    gr1 = r.grad.data.clone()
    gk1 = k.grad.data.clone()
    gv1 = v.grad.data.clone()
    gw11 = w1.grad.data.clone()
    gw21 = w2.grad.data.clone()
    gu11 = u1.grad.data.clone()
    gu21 = u2.grad.data.clone()

    # gr1, gk1, gv1, gw11, gu11, gw21, gu21 = PYTHON_1_BWD(gy, r, k, v, w1, u1, w2, u2)

    compare(gr, gr1, 'g_r')
    compare(gk, gk1, 'g_k')
    compare(gv, gv1, 'g_v')
    compare(gw1, gw11, 'g_w1')
    compare(gu1, gu11, 'g_u1')
    compare(gw2, gw21, 'g_w2')
    compare(gu2, gu21, 'g_u2')
    exit(0)

    gr1, gk1, gv1, gw11, gu11, gw21, gu21 = PYTHON_2_BWD(gy, r, k, v, w1, u1, w2, u2)
    compare(gr, gr1, 'g_r')
    compare(gk, gk1, 'g_k')
    compare(gv, gv1, 'g_v')
    compare(gw1, gw11, 'g_w1')
    compare(gu1, gu11, 'g_u1')
    compare(gw2, gw21, 'g_w2')
    compare(gu2, gu21, 'g_u2')