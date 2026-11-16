"""判决性诊断：defocus 与 gaussian 在相同过渡带宽度下，|h(rho)|^2 差多少？

已知：
  * defocus 指数 5.83 +/- 0.07，gaussian 4.17 +/- 0.05（FFHQ），差 20sigma
  * 两者过渡带宽度几乎相同（9.27 vs 8.86）
  * 两者都各向同性（|h|^2 只依赖 rho）
  * 径向剖面形状看起来相似

问题：逐频率比较，|h_defocus(rho)|^2 与 |h_gaussian(rho)|^2 到底差多少？
若差异很小（几个百分点）=> 指数不是径向谱的函数，存在非谱依赖（惊人）
若差异很大 => 可以指认差异在哪个频段

做法：对每个 defocus 的 theta，找一个 gaussian 的 sigma 使过渡带宽度匹配，
然后逐径向桶打印两者的 |h|^2 之比。

用法： python diag_defocus_vs_gaussian.py
"""

import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from putils.convolution import GaussianBlur, DefocusBlur

DEV = torch.device("cuda")
H = W = 64
KS = 15
NB = 48


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
    rho_b = np.array([Rr.ravel()[idx.ravel() == b].mean() if (idx == b).any() else 0.0
                      for b in range(NB)])

    def profile(kernel):
        h2 = khat(kernel)[0, 0].cpu().numpy()
        return np.array([h2[idx == b].mean() if (idx == b).any() else 0.0
                         for b in range(NB)])

    def cross(prof, level):
        """prof 随 rho 递减，返回首次穿越 level 的半径（线性插值）。"""
        for i in range(len(prof) - 1):
            a, b = prof[i] - level, prof[i + 1] - level
            if a * b <= 0 and a != b:
                return rho_b[i] + (level - prof[i]) * (rho_b[i + 1] - rho_b[i]) / \
                    (prof[i + 1] - prof[i])
        return float("nan")

    def width(prof):
        r9 = cross(prof, 0.1)      # prof = |h|^2，从 1 往下；d=0.1 <=> prof=0.9
        r1 = cross(prof, 0.9)      # d = 0.9 <=> prof = 0.1
        if not (np.isfinite(r9) and np.isfinite(r1)):
            return float("nan")
        return float(r9 - r1)

    gf = GaussianBlur(kernel_size=KS, device=DEV, dtype=torch.float64)
    df = DefocusBlur(kernel_size=KS, device=DEV, dtype=torch.float64)
    gsig = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
    gprof = {s: profile(gf.kernel(s)) for s in gsig}
    gwidth = {s: width(p) for s, p in gprof.items()}

    print("=" * 112)
    print(" defocus vs gaussian：匹配过渡带宽度后的逐频率比较")
    print("=" * 112)
    print(f"{'defocus':>8} | {'匹配 sigma':>9} | {'宽度_d':>7} | {'宽度_g':>7} | "
          f"{'max|比值-1|':>11} | {'比值>1.1 的桶':>12} | {'逐桶比值(每4桶)':>40}")
    print("-" * 112)
    for th in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        pd = profile(df.kernel(th))
        wd = width(pd)
        if not np.isfinite(wd):
            print(f"{th:>8g} | {'宽度无定义':>9}")
            continue
        # 找宽度最接近的 gaussian
        cand = [s for s in gsig if np.isfinite(gwidth[s])]
        if not cand:
            print(f"{th:>8g} | {'无可用匹配':>9}")
            continue
        best = min(cand, key=lambda s: abs(gwidth[s] - wd))
        pg = gprof[best]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pg > 1e-6, pd / np.maximum(pg, 1e-12), np.nan)
        dev = np.nanmax(np.abs(ratio - 1.0))
        nfar = int(np.nansum(np.abs(ratio - 1.0) > 0.1))
        samp = " ".join(f"{v:.2f}" if np.isfinite(v) else "  - " for v in ratio[::4])
        print(f"{th:>8g} | {best:>9g} | {wd:>7.2f} | {gwidth[best]:>7.2f} | "
              f"{dev:>11.3f} | {nfar:>12d} | {samp:>40}")
    print("-" * 112)
    print("\n[判据]")
    print("  * max|比值-1| 若 < 0.1  => 两者径向谱几乎相同，指数差 1.4 无法用径向谱解释")
    print("  * 若差异集中在高端频率 => 差异在尾部衰减速率")
    print("  * 若差异集中在低端 => 差异在通过带")


if __name__ == "__main__":
    main()
