"""诊断：五个观察核族的 |h|^2 径向剖面有什么结构差异？

目的：defocus 与 gaussian 过渡带宽度几乎相同（9.27 vs 8.86）但指数差 1.5
（5.62 vs 4.21）。要找出可控制的结构差异。

对每族每个 theta，打印径向分桶（48 桶）后的 |h|^2 剖面摘要：
  * 单调性：剖面是否单调下降
  * 旁瓣 revived：|h|^2 在首次掉到 0.01 之后是否回升，回升到多高
  * 零点比例、有效支撑

用法： python diag_kernel_profiles.py
"""

import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

DEV = torch.device("cuda")
H = W = 64
KS = 15
NB = 48

FAMILIES = {
    "gaussian": (GaussianBlur, "sigma", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "defocus": (DefocusBlur, "radius", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "airy": (AiryDisc, "radius", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "motion": (MotionBlur, "length", [1, 3, 5, 7, 9, 13]),
}


def khat(kernel):
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    kh, kw = kernel.shape[-2:]
    pad = torch.zeros((1, 1, H, W), device=DEV, dtype=torch.float64)
    t, l = (H - kh) // 2, (W - kw) // 2
    pad[:, :, t:t + kh, l:l + kw] = kernel.to(torch.float64)
    return torch.fft.rfft2(pad, dim=(-2, -1)).abs() ** 2


def main():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, NB + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, NB - 1).reshape(Rr.shape)
    rho_b = np.array([Rr.ravel()[idx.ravel() == b].mean() if (idx == b).any() else 0
                      for b in range(NB)])

    print("=" * 108)
    print(" 各核族 |h|^2 径向剖面（48 桶）摘要")
    print("=" * 108)
    print(f"{'族':>9} {'theta':>6} | {'单调?':>5} | {'首陷 rho':>8} | {'旁瓣回升':>8} | "
          f"{'回升处 rho':>9} | {'零点比例':>8} | {'剖面(每 4 桶取样)':>44}")
    print("-" * 108)
    for fam, (cls, pname, levels) in FAMILIES.items():
        kf = cls(kernel_size=KS, device=DEV, dtype=torch.float64)
        for th in levels:
            h2 = khat(kf.kernel(th))[0, 0].cpu().numpy()
            prof = np.array([h2[idx == b].mean() if (idx == b).any() else 0.0
                             for b in range(NB)])
            zeros = float((h2 < 1e-6).mean())
            # 首次陷到 0.01 以下
            below = np.where(prof < 0.01)[0]
            first = int(below[0]) if len(below) else NB - 1
            # 陷后是否回升
            revive = 0.0
            rv_rho = 0.0
            if first < NB - 2:
                after = prof[first:]
                j = int(np.argmax(after))
                if after[j] > 3 * max(prof[first], 1e-9) and after[j] > 0.02:
                    revive = float(after[j])
                    rv_rho = float(rho_b[first + j])
            mono = "是" if np.all(np.diff(prof) <= 1e-9) else "否"
            samp = " ".join(f"{v:.2f}" for v in prof[::4])
            print(f"{fam:>9} {th:>6g} | {mono:>5} | {rho_b[first]:>8.2f} | "
                  f"{revive:>8.3f} | {rv_rho:>9.2f} | {zeros:>8.3f} | {samp:>44}")
        print("-" * 108)

    print("\n[判据]")
    print("  * 若 defocus 的剖面在陷后大幅回升（旁瓣），而 gaussian 单调，")
    print("    则待控制的量是'旁瓣回升高度'")
    print("  * 若 defocus 与 airy 剖面相近但指数差很多，则差异来自参数范围/拟合区间")


if __name__ == "__main__":
    main()
