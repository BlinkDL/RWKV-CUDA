import time, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

'''
cd /mnt/program/_RWKV_/_ref_/RWKV-CUDA/rwkv7_fast_fused; python rwkv7_cuda_benchmark_state.py fp32 0; python rwkv7_cuda_benchmark_state.py fp32 1
cd /mnt/program/_RWKV_/_ref_/RWKV-CUDA/rwkv7_fast_fused; python rwkv7_cuda_benchmark_state.py bf16 0; python rwkv7_cuda_benchmark_state.py bf16 1
'''

DTYPE = torch.float if sys.argv[1].strip()=='fp32' else torch.bfloat16
BENCHMARK_SPEED = True if int(sys.argv[2].strip()) == 1 else 0

######################################################################################################

DEVICE = "cuda"

B, T, CHUNK_LEN, C, N = 2, 64, 16, 32, 16
if BENCHMARK_SPEED:
    B, T, CHUNK_LEN, C, N = 8, 4096, 16, 4096, 64

H, HEAD_SIZE = C // N, N
print(f"\n\nB={B} T={T} C={C} HEAD_SIZE={HEAD_SIZE} DTYPE={str(DTYPE).replace('torch.','')}\n\n")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def val(x):
    return x.detach().float().cpu().numpy()

def err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

######################################################################################################

def RWKV7_STATE_CLAMPW_REF(state, r, w, k, v, a, b):
    r = r.view(B, T, H, N)
    k = k.view(B, T, H, N)
    v = v.view(B, T, H, N)
    a = a.view(B, T, H, N)
    b = b.view(B, T, H, N)

    w = -F.softplus(-w) - 0.5 # soft-clamp, after exp becomes sigmoid in CUDA kernel
    w = torch.exp(-torch.exp(w.view(B, T, H, N)))

    out = torch.zeros((B, T, H, N), device=DEVICE)

    for t in range(T):
        rr = r[:, t, :]
        kk = k[:, t, :]
        vv = v[:, t, :]
        aa = a[:, t, :]
        bb = b[:, t, :]
        sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
        state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
        out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)

    return out.view((B, T, C))

######################################################################################################

if DTYPE == torch.bfloat16:
    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="rwkv7_state_clampw", sources=[f'cuda/rwkv7_state_clampw.cu', 'cuda/rwkv7_state_clampw.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
        @staticmethod
        def forward(ctx,s0,r,w,k,v,a,b):
            B,T,H,N = r.shape
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [r,w,k,v,a,b])
            assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
            assert s0.dtype==torch.float
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,N,N, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,N,dtype=torch.float32,device=w.device)
            torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
            return y
        @staticmethod
        def backward(ctx,dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            r,w,k,v,a,b,s,sa = ctx.saved_tensors
            B,T,H,N = r.shape
            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
            ds0 = torch.empty(B,H,N,N,dtype=torch.bfloat16,device=r.device)
            torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
            return ds0,dr,dw,dk,dv,da,db
    def RWKV7_STATE_CLAMPW_CUDA(s0,r,w,k,v,a,b):
        B,T,HC = r.shape
        r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
        return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC)

elif DTYPE == torch.float:
    flags = ['-res-usage', f'-D_N_={HEAD_SIZE}', "-D_FP32_", f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="rwkv7_state_clampw", sources=[f'cuda/rwkv7_state_clampw.cu', 'cuda/rwkv7_state_clampw.cpp'], is_python_module=False, verbose=True, extra_cflags=["-D_FP32_"], extra_cuda_cflags=flags)
    class RWKV7_STATE_CLAMPW_CUDA_OP(torch.autograd.Function):
        @staticmethod
        def forward(ctx,s0,r,w,k,v,a,b):
            B,T,H,C = r.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.float32 for i in [s0,r,w,k,v,a,b])
            assert all(i.is_contiguous() for i in [s0,r,w,k,v,a,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.rwkv7_state_clampw.forward(s0,r,w,k,v,a,b,y,s,sa)
            ctx.save_for_backward(r,w,k,v,a,b,s,sa)
            return y
        @staticmethod
        def backward(ctx,dy):
            assert all(i.dtype==torch.float32 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            r,w,k,v,a,b,s,sa = ctx.saved_tensors
            dr,dw,dk,dv,da,db = [torch.empty_like(x) for x in [r,w,k,v,a,b]]
            ds0 = torch.empty(B,H,N,N,dtype=torch.float32,device=r.device)
            torch.ops.rwkv7_state_clampw.backward(r,w,k,v,a,b,dy,s,sa,ds0,dr,dw,dk,dv,da,db)
            return ds0,dr,dw,dk,dv,da,db
    def RWKV7_STATE_CLAMPW_CUDA(s0,r,w,k,v,a,b):
        B,T,HC = r.shape
        r,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [r,w,k,v,a,b]]
        return RWKV7_STATE_CLAMPW_CUDA_OP.apply(s0,r,w,k,v,a,b).view(B,T,HC)

######################################################################################################

with torch.no_grad():
    r = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1) * 3
    w = torch.empty(B, T, C, device=DEVICE).uniform_(-6, 0)
    k = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1) * 3
    v = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1) * 3
    a = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1) * 2
    b = torch.empty(B, T, C, device=DEVICE).uniform_(-1, 1) * 2
    a = F.normalize(a, dim=-1, p=2.0)
    b = F.normalize(b, dim=-1, p=2.0)
    s = torch.empty(B, H, N, N, device=DEVICE).uniform_(-1, 1) * 10

params = (s,r,w,k,v,a,b)

def LOSS(y):
    return ((y * y) - torch.tanh(y)).sum()

def clear_grad():   
    for t in params:
        t.requires_grad_(True)
        if t.grad is not None:
            t.grad.zero_()

######################################################################################################

if not BENCHMARK_SPEED:
    clear_grad()
    y = RWKV7_STATE_CLAMPW_REF(*params)
    LOSS(y).backward()
    grad_ref = [t.grad.detach().clone() for t in params]

    clear_grad()
    if DTYPE == torch.float:
        y_cuda = RWKV7_STATE_CLAMPW_CUDA(*params)
    else:
        ss,rr,ww,kk,vv,aa,bb = s,r.bfloat16(),w.bfloat16(),k.bfloat16(),v.bfloat16(),a.bfloat16(),b.bfloat16()
        y_cuda = RWKV7_STATE_CLAMPW_CUDA(ss,rr,ww,kk,vv,aa,bb).float()
    LOSS(y_cuda).backward()
    grad_cuda = [t.grad.detach().clone() for t in params]

    print('!!! y err !!!', err_ratio(y, y_cuda))
    for name, g_ref, g_cuda in zip('srwkvab', grad_ref, grad_cuda):
        print(f'!!! g_{name} err !!!', err_ratio(g_ref, g_cuda))

else:
    print('benchmark speed...')
    repeats = 10
    fwd_times = []
    bwd_times = []
    for _ in range(repeats):
        clear_grad()

        if DTYPE == torch.float:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            y_cuda = RWKV7_STATE_CLAMPW_CUDA(*params)
            torch.cuda.synchronize(); fwd_times.append(time.perf_counter() - t0)

            torch.cuda.synchronize(); t0 = time.perf_counter()
            LOSS(y_cuda).backward()
            torch.cuda.synchronize(); bwd_times.append(time.perf_counter() - t0)
        else:
            ss,rr,ww,kk,vv,aa,bb = s,r.bfloat16(),w.bfloat16(),k.bfloat16(),v.bfloat16(),a.bfloat16(),b.bfloat16()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            y_cuda = RWKV7_STATE_CLAMPW_CUDA(ss,rr,ww,kk,vv,aa,bb)
            torch.cuda.synchronize(); fwd_times.append(time.perf_counter() - t0)

            torch.cuda.synchronize(); t0 = time.perf_counter()
            LOSS(y_cuda).backward()
            torch.cuda.synchronize(); bwd_times.append(time.perf_counter() - t0)

    print('fwd time =', min(fwd_times))
    print('bwd time =', min(bwd_times))
