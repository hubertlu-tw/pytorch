import torch
def run_model_on_device(fs, X, gO, device_string, numeric_type):
    ln = torch.nn.LayerNorm((fs,), device=device_string, dtype=numeric_type)
    ln.reset_parameters()
    X.grad = None
    ln.zero_grad(set_to_none=True)
    out = ln(X)
    out.backward(gO)
    return (ln.weight.grad, ln.bias.grad)

def run_correctness_test(eps_weight, eps_bias):
    dtype = torch.float
    #for fs in (512, 1024, 2048, 4096, 8192, 10000, 500, 1000, 2001, 4005, 8117):
    for fs in (1, 32, 64, 256):
        #for bs in (512, 1024, 2048, 4096, 525, 1033, 2064, 3000):
        for bs in (1, 32, 64, 256):
            mean_adjustment = torch.randn(fs, device="cpu", dtype=torch.float)
            X = mean_adjustment * torch.randn(
                bs, fs, device="cpu", dtype=torch.float, requires_grad=True
            )

            X = X.detach().requires_grad_()
            gO = torch.rand_like(X)
            X_gpu = X.to("cuda")
            X_gpu = X_gpu.detach().requires_grad_()
            gO_gpu = gO.to("cuda")
            gO_gpu = gO_gpu.detach().requires_grad_()

            grad_cpu_ref = run_model_on_device(fs, X, gO, "cpu", dtype)
            grad_gpu = run_model_on_device(fs, X_gpu, gO_gpu, "cuda", dtype)
            weight_grad_gpu_target = grad_gpu[0].detach().to("cpu")
            bias_grad_gpu_target = grad_gpu[1].detach().to("cpu")

            weight_delta = torch.abs(grad_cpu_ref[0] - weight_grad_gpu_target)
            weight_mismatches = (weight_delta >= eps_weight).nonzero()
            weight_mismatch_pct = len(weight_mismatches) / len(weight_delta) * 100

            bias_delta = torch.abs(grad_cpu_ref[1] - bias_grad_gpu_target)
            bias_mismatches = (bias_delta >= eps_bias).nonzero()
            bias_mismatch_pct = len(bias_mismatches) / len(bias_delta) * 100

            print(
                "Size ({} x {}) mismatch percentage: weight {:3.2f} bias {:3.2f}".format(
                    fs, bs, weight_mismatch_pct, bias_mismatch_pct
                )
            )

run_correctness_test(eps_weight=1e-3, eps_bias=1e-3)
