import numpy as np
import torch

for f in ["/tmp/pool_ffhq64_256_s0.pt",
          "/tmp/pool_afhqv2-64x64-vp_256_s0.pt"]:
    p = torch.load(f, map_location="cpu").to(torch.float32)
    with torch.no_grad():
        P = (torch.fft.rfft2(p.to(torch.float64), dim=(-2, -1)).abs() ** 2) / (64 * 64)
        S = P.mean(1).mean(0)          # (33, 33)
    H = W = 64
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
    edges = np.linspace(0, R.max() + 1e-6, 9)
    idx = np.clip(np.digitize(R.ravel(), edges) - 1, 0, 8).reshape(R.shape)
    Sc = S.numpy()
    rad = [float(Sc[idx == b].mean()) for b in range(9)]
    print(f)
    print("  shape", tuple(p.shape),
          "range [%.3f, %.3f]" % (float(p.min()), float(p.max())))
    print("  径向功率(内->外):", " ".join("%.4g" % v for v in rad))
    print("  外/内 = %.4g" % (rad[-1] / rad[0]))
