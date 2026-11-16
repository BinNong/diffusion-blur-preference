import sys
import torch

sys.path.insert(0, "/data/experiment/de_conv_1/code/blind_deblur_diffusion")
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

B = int(sys.argv[1])
dev = torch.device("cuda")
m = load_local_model("edm-imagenet-64x64-cond-adm.pkl").to(dev)
CI = 207


def den(x, t, *a, **k):
    lab = torch.full((x.shape[0],), CI, device=x.device, dtype=torch.long)
    oh = torch.nn.functional.one_hot(lab, 1000).to(torch.float32)
    return m(x.to(torch.float32), t, class_labels=oh, *a, **k)


prior = ScorePrior(denoiser=den)
flow, ts = get_ode_flow(prior)
x = torch.randn(B, 3, 64, 64, device=dev)
try:
    with torch.no_grad():
        v = likelihood_fn(x, flow, t_spans=[80.0, 0.002])
    print(f"B={B} OK peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB "
          f"reserved={torch.cuda.max_memory_reserved()/2**20:.0f} MiB "
          f"val={float(v.reshape(-1)[0]):.1f}")
except torch.OutOfMemoryError:
    print(f"B={B} OOM peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
