# Ref:
#       1. https://github.com/pytorch/pytorch/pull/26201
#       2. https://github.com/pytorch/pytorch/pull/68238

from distutils.command.config import config
import torch
from torch.nn import LayerNorm
import timeit

number_runs = 1000  # TODO: Modify this to save time!
def test_forward(layer_norm_cuda, input_cuda):
    layer_norm_cuda(input_cuda); torch.cuda.synchronize()

def test_backward(out_cuda, layer_norm_grad_cuda, create_graph):
    out_cuda.backward(layer_norm_grad_cuda, retain_graph=True, create_graph=create_graph); torch.cuda.synchronize()

def test_fwdbwd(input_cuda, layer_norm_cuda, gO):
    input_cuda.grad = None
    layer_norm_cuda.zero_grad(set_to_none=True)
    out = layer_norm_cuda(input_cuda)
    out.backward(gO)
    torch.cuda.synchronize()


def benchmark(config_m, config_n):

    print("M | N | fwd (half) | fwdbwd (half) | fwd (float) | fwdbwd (float)")
    if len(config_m) != len(config_n):
        print("Please make sure the lengths of config_m and config_m are the same.")

    for i in range(len(config_m)):
        normalized_shape = config_n[i]
        results = [config_m[i], config_n[i]]
        for dtype in (torch.half, torch.float):
            if dtype == torch.half:
                layer_norm_cuda = LayerNorm(normalized_shape).half().cuda()
            else:
                layer_norm_cuda = LayerNorm(normalized_shape).cuda()

            input_cuda = torch.randn(config_m[i], config_n[i], device='cuda', dtype=dtype, requires_grad=True)

            # print("cuda forward:")
            result_fwd = timeit.timeit(lambda: test_forward(layer_norm_cuda, input_cuda), number=number_runs)
            results.append(result_fwd / number_runs * 1000)

            gO = torch.rand_like(input_cuda)

            result_fwdbwd = timeit.timeit(lambda: test_fwdbwd(input_cuda, layer_norm_cuda, gO), number=number_runs)
            results.append(result_fwdbwd / number_runs * 1000)

        print('{:09d}|{:09d}|{:9.5f}|{:9.5f}|{:9.5f}|{:9.5f}'.format(results[0], results[1], results[2], results[3], results[4], results[5]))

    print("Times are in microseconds (us).")

# CVT
config_m_cvt = [50432, 50176, 200704, 802816]
config_n_cvt = [384, 384, 192, 64]

# https://github.com/pytorch/pytorch/pull/68238#issue-1051621716
#config_m_68238 = [200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272, 200, 1000, 6000, 6272]
#config_n_68238 = [256,256,256,256,512,512,512,512,1024,1024,1024,1024,1536,1536,1536,1536,2048,2048,2048,2048,3072,3072,3072,3072]

# https://github.com/pytorch/pytorch/pull/27634
#config_m_27634 = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
#config_n_27634 = [2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192]

#config_m = config_m_cvt + config_m_68238 + config_m_27634
#config_n = config_n_cvt + config_n_68238 + config_n_27634

config_m = [1,1,1,1,32,32,32,32,64,64,64,64,256,256,256,256] # M = bs
config_n = [32,64,128,256,32,64,128,256,32,64,128,256,32,64,128,256] # N = seq_len

benchmark(config_m, config_n)

