# RWKV-CUDA
The CUDA version of the RWKV language model ( https://github.com/BlinkDL/RWKV-LM )

## Experiment 1 - depthwise_conv1d

The formula:
```
w.shape = (C, T)
k.shape = (B, C, T)
out.shape = (B, C, T)
out[b][c][t] = sum_u{ w[c][(T-1)-(t-u)] * k[b][c][u] }
```

pytorch = fwd 94ms bwd 534ms

CUDA kernel v0 = fwd 45ms bwd 84ms (simple)

CUDA kernel v1 = fwd 17ms bwd 45ms (shared memory)

CUDA kernel v2 = fwd 14ms bwd 31ms (float4)

How to use: ```python run.py``` and it will compile everything for you (```pip install Ninja``` if you don't have it).
