# RWKV-CUDA
The CUDA version of the RWKV language model ( https://github.com/BlinkDL/RWKV-LM )

## Towards RWKV-4 (see the wkv folder)

I have a basic RWKV-4 kernel in the wkv folder. Let's optimize it.

<img src="https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v3-plan.png">

## Experiment 1 - depthwise_conv1d - 20x faster than pytorch

The formula:
```
w.shape = (C, T)
k.shape = (B, C, T)
out.shape = (B, C, T)
out[b][c][t] = sum_u{ w[c][(T-1)-(t-u)] * k[b][c][u] }
```

pytorch = fwd 94ms bwd 529ms

CUDA kernel v0 = fwd 45ms bwd 84ms (simple)

CUDA kernel v1 = fwd 17ms bwd 43ms (shared memory)

CUDA kernel v2 = fwd 13ms bwd 31ms (float4)

CUDA kernel v3 = fwd 3.4ms bwd 23ms (B-group)

More test on RTX3090:

pytorch = fwd 14ms bwd 65ms

CUDA kernel v3 = fwd 0.8ms bwd 5.5ms

How to use: ```python run.py``` and it will compile everything for you (```pip install Ninja``` if you don't have it).
