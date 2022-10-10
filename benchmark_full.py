"""
    This code is modified to fit the CVT kernel calls from the benchmark code provided at
    https://github.com/pytorch/pytorch/pull/68238

    TORCH_NORM == True:      torch.nn.LayerNorm implementation
        - GammaBetaBackwardCUDAKernel<float, float>(long, long, float const*, float const*, float const*, float const*, float*, float*)", "grid dim": [24, 1, 1], "block dim": [16, 32, 1], "shared size": 4096, "stream": 0x0
        - GammaBetaBackwardCUDAKernel<float, float>(long, long, float const*, float const*, float const*, float const*, float*, float*)", "grid dim": [12, 1, 1], "block dim": [16, 32, 1], "shared size": 4096, "stream": 0x0
        - GammaBetaBackwardCUDAKernel<float, float>(long, long, float const*, float const*, float const*, float const*, float*, float*)", "grid dim": [4, 1, 1], "block dim": [16, 32, 1], "shared size": 4096, "stream": 0x0

    TORCH_NORM == False:    apex.normalization.FusedLayerNorm
        - cuComputePartGradGammaBeta<float, float, float>(float const*, float const*, int, int, float const*, float const*, float, float*, float*)", "grid dim": [6, 64, 1], "block dim": [64, 4, 1], "shared size": 8320,
        - cuComputeGradGammaBeta<float, float>(float const*, float const*, int, int, int, float*, float*)", "grid dim": [6, 1, 1], "block dim": [64, 8, 1], "shared size": 2048,

        - cuComputePartGradGammaBeta<float, float, float>(float const*, float const*, int, int, float const*, float const*, float, float*, float*)", "grid dim": [3, 64, 1], "block dim": [64, 4, 1], "shared size": 8320,
        - cuComputeGradGammaBeta<float, float>(float const*, float const*, int, int, int, float*, float*)", "grid dim": [3, 1, 1], "block dim": [64, 8, 1], "shared size": 2048,

        - cuComputePartGradGammaBeta<float, float, float>(float const*, float const*, int, int, float const*, float const*, float, float*, float*)", "grid dim": [1, 64, 1], "block dim": [64, 4, 1], "shared size": 8320,
        - cuComputeGradGammaBeta<float, float>(float const*, float const*, int, int, int, float*, float*)", "grid dim": [1, 1, 1], "block dim": [64, 8, 1], "shared size": 2048, "stream": "0x0"
"""

import torch
from torch.utils.benchmark import Timer, Compare
from apex.normalization import FusedLayerNorm
# CVT
config_m_cvt = [50432, 50176, 200704, 802816]
config_n_cvt = [384, 384, 192, 64]

# https://github.com/pytorch/pytorch/pull/68238#issue-1051621716 
config_m_68238 = [200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272]
config_n_68238 = [256,256,256,256,512,512,512,512,1024,1024,1024,1024,1536,1536,1536,1536,2048,2048,2048,2048,3072,3072,3072,3072]

# https://github.com/pytorch/pytorch/pull/27634
config_m_27634 = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
config_n_27634 = [2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192]

config_m = config_m_cvt + config_m_68238 + config_m_27634
config_n = config_n_cvt + config_n_68238 + config_n_27634
"""
Shape = (128, 2097152)
Shape = (256, 1048576)
Shape = (512, 524288)
Shape = (1024, 262144)
Shape = (2048, 131072)
Shape = (4096, 65536)
Shape = (8192, 32768)
Shape = (16384, 16384)
Shape = (32768, 8192)
"""
results = []

TORCH_NORM = True
#TORCH_NORM = False

dtype = torch.float
for dtype in (torch.float, torch.half):
    for ci in range(len(config_m)):
        if (TORCH_NORM):
            if dtype == torch.half:
                ln = torch.nn.LayerNorm((config_n[ci],)).half().to("cuda")
            else:
                ln = torch.nn.LayerNorm((config_n[ci],)).to("cuda")
        else:
            if dtype == torch.half:
                ln = FusedLayerNorm((config_n[ci],)).half().to("cuda")
            else:
                ln = FusedLayerNorm((config_n[ci],)).to("cuda")

        X = torch.randn(config_m[ci], config_n[ci], device="cuda", dtype=dtype, requires_grad=True)
        gO = torch.rand_like(X)
        stmtfwd = "ln(X)"
        stmtfwdbwd = "X.grad=None; ln.zero_grad(set_to_none=True); out = ln(X); out.backward(gO)"
        tfwd = Timer(stmt=stmtfwd, label="ln", sub_label=f"{config_m[ci]:5}, {config_n[ci]:5}", description=f"fwd, {dtype}", globals=globals())
        tfwdbwd = Timer(stmt=stmtfwdbwd, label="ln", sub_label=f"{config_m[ci]:5}, {config_n[ci]:5}", description=f"fwdbwd, {dtype}", globals=globals())
        for t in (tfwd, tfwdbwd):
            results.append(t.blocked_autorange())
        print(config_n[ci], end='\r')

c = Compare(results)
c.print()

