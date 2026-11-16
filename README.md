# Diffusion Blur Preference

Reproduction materials for **Quantifying Blur Preference in Diffusion Priors: Spectral Scaling and Implications for Blind Image Deconvolution**.

## Contents

- `repro/`: measurement, optimisation, analysis, plotting, and validation scripts.
- `results/`: raw JSON results, logs, saved score-profile endpoints, verbatim Hutchinson probes, and integrity records.
- `paper/`: manuscript source and figures, compiled PDF, and point-by-point review responses.
- `code/`: upstream EDM, blind-deblurring, FFHQ, improved-diffusion, and guided-diffusion source code. Upstream licenses are retained in their respective directories.
- `tools/`: environment and asset setup scripts.

Start with `results/review2_repro_manifest.json` for file hashes and the exact review2 command sequence. The scientific scope and point-by-point modifications are summarised in `paper/response_review2.md`.

## Numerical checks

```bash
python repro/analyze_review2.py --nboot 2000
python repro/analyze_law_stability.py
python repro/analyze_score_profile.py
python repro/validate_review2_physics.py
```

The tested Linux environment is recorded in `results/review2_environment.txt`. Local syntax checks do not replace GPU execution of the measurement scripts.

## Data and checkpoints

Model checkpoints are not included. EDM and ADM SHA-256 values are in `results/review2_edm_checkpoint_sha256.txt` and `results/external_assets.json`. The generated EDM pools can be rebuilt with `repro/make_pool.py`. Real BSDS image provenance and per-image hashes are in `results/bsds300_real_manifest.json`; the raw BSDS images are not redistributed here.

The archive excludes credentials, operational server recovery scripts, and model weights. GitHub source archives and the manifest provide complementary integrity checks.

## Scope

The evidence is empirical and limited to the stated checkpoints, image pools, noise levels, operators, and numerical protocols. Plug-and-play grid selections are not certified profile-MAP solutions. Direct score-gradient results remain approximate and non-global. No universal cross-architecture law or improved restoration algorithm is claimed.

## Licensing

Third-party projects retain their original license files under `code/`. No new top-level license has been selected for the combined repository; contact the authors before reusing original materials outside the terms of the upstream projects.
