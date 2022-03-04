# RWKV-CUDA
The CUDA version of the RWKV language model ( https://github.com/BlinkDL/RWKV-LM )

## Experiment 1 - depthwise_conv1d

pytorch = fwd 94ms bwd 534ms

CUDA kernel v0 = fwd 45ms bwd 84ms (simple)

CUDA kernel v1 = fwd 17ms bwd 45ms (shared memory)

CUDA kernel v2 = fwd 14ms bwd 31ms (float4)

How to use: ```python run.py``` and it will compile everything for you (```pip install Ninja``` if you don't have it).
