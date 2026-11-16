# Cover letter — IEEE Transactions on Image Processing

**Manuscript type:** Regular Paper  
**Title:** Quantifying Blur Preference in Diffusion Priors: Spectral Scaling and Implications for Blind Image Deconvolution  
**Authors:** Bin Nong (Tianfu College of Southwestern University of Finance and Economics; corresponding author, nongbin@tfswufe.edu.cn), Weihong Fu (Xidian University)

---

Dear Editors,

We submit the above manuscript for consideration as a Regular Paper in *IEEE Transactions on Image Processing*.

## What the paper is about

Prior work has established that diffusion models can assign high density to blurry images and that learned-prior MAP objectives can exhibit a blurry trap. We do not claim to rediscover those phenomena. Our narrower question is quantitative: how large is the paired log-density difference between an image and its blurred version, how does it vary with spectral attenuation and diffusion noise, and what can—and cannot—be inferred for blind deconvolution?

For an EDM prior trained on FFHQ-64 and evaluated on its generated samples,

> Λ(t, u) = log p_t(h ⋆ x) − log p_t(x)

is positive and large in the tested low-noise regime. The main image pools are model-generated, not real FFHQ or AFHQv2 dataset splits. On the first 32 sorted real BSDS300 test images, all 256 low-noise image–kernel pairs are positive. The corresponding image-level count is 32/32, with a two-sided 95% Clopper–Pearson interval [0.8911, 1], rather than a degenerate bootstrap interval [1,1]. At t = 0.20, 120/128 raw pairs are positive; all four tested kernels are simultaneously positive on 27/32 images (image-level interval [0.6721, 0.9472]). These are selected real natural images, not a real-FFHQ validation or a universal population probability.

## Main contributions

**1. Paired measurement with explicit numerical limits.** We distinguish the implemented unweighted rFFT coordinate from the full-FFT average. The terminal Gaussian contribution is derived using the actual image spectrum through Parseval's identity. Evaluation on 120 generated images and eight kernels gives terminal terms of 0.0066–0.2436 nats, below the rigorous 0.96-nat bound. Independent 32-repeat measurements give mean ± SE values of 31851.9 ± 54.2 nats at t = 0.002 and 195.3 ± 10.3 nats at t = 0.20; these cover one image/operator, not every configuration. Adaptive-solver checks are reported with the same scope. A full-RGB, conjugate-complete spectral reference gives a 12.57× low-noise shortfall in the representative spectrum-matched Gaussian case, rather than the retired 71× value.

**2. An empirical scaling law with a defined estimator and scope.** Seven smooth profiles admit Λ ≈ 𝓜(shape)·u^{b_∞}·exp((A/q)u^q), with b(u) = d log Λ/d log u ≈ b_∞ + A·u^q (b_∞ ≈ −0.73, A ≈ −7.2, q ≈ 3.4 at t = 0.002). The manuscript derives the integrated form and distinguishes its shape factor 𝓜(shape) from the separate frozen-exponent diagnostic C(u). It specifies the arithmetic image mean, unweighted degree-five derivative estimator, all-common-u fitting grid, closed-form subinterval, and absence of per-point variance weights. Four unseen shape classes plus an internal control provide a limited out-of-sample test. The parameters drift with t; the form fits through t = 0.05 in the tested range and degrades at t = 0.16. We do not claim a universal shape descriptor or a noise-independent law.

A 2000-replicate image bootstrap gives three-family-mean 95% intervals [−0.7367, −0.7221], [−7.6602, −6.8310], and [3.3247, 3.5231] for b∞, A, and q. Refitting the same stored measurements with full-FFT coordinates gives (−0.7191, −7.1706, 3.3881), with less than 0.001 change in per-family fit rms. We also report first-20/first-60/last-60 image subsets. Polynomial-degree and fitting-window sensitivity is explicitly disclosed: the precise parameter vector is estimator-dependent, although the exponent remains negative and steepens with u. Image-bootstrap intervals are not presented as total uncertainty from Hutchinson estimation, integration, or smoothing.

A new full-curve audit uses four fixed generated images, three profile families, 20 operators, and four independent Rademacher probes shared across each curve. The three-family-mean across-probe SDs for b∞, A, and q are 0.00240, 0.0954, and 0.0317. Holding the probe fixed, RK45 1e-5 versus 1e-6 changes Λ by at most 5.27 nats and b(u) by at most 0.0342. These are scoped numerical diagnostics, separate from image-bootstrap and smoothing uncertainty.

Across 851 operator rows, the full-RGB/full-FFT spectral reference has median absolute ratio |Λ_G/Λ| = 0.789 at t = 0.002 and 4.868 at t = 0.16. It is not uniformly inaccurate or universally underestimating. A directional ODE-potential Taylor test includes two finite-difference spacings, but only one direction per image/noise setting; it is not a full-Hessian evaluation or a theorem about all local methods.

**3. A mathematically delimited blind-deconvolution analysis.** We define L = D + q_t and derive E_PnP = Δq_t − Λ_clean(t) + (D_PnP − D_*) by direct substitution. The identity holds for any feasible PnP output. A negative value certifies only that the generating pair is not a global minimiser of the declared smoothed-prior surrogate; it does not certify a profile minimum.

The manuscript now explicitly distinguishes this surrogate from the exact posterior for the original clean-image process. It also gives the complete implemented Douglas–Rachford-type update and explains why Tweedie's identity supplies a score gradient but does not make the denoiser the proximal map of q_t. Consequently, the 79/80 Gaussian result is reported as a **PnP-grid minimum count**, not a 79/80 exact profile-MAP result. The multi-kernel and real-BSDS pilots frequently under-blur, so neither direction is universal. Fixed-grid and selected-point decompositions remain descriptive rather than causal.

A direct score-gradient pilot uses the actual gradient of the declared L, 600 Adam steps, two starts, the same 21-point grid, and shared-probe initial/final ODE evaluation. All 336/336 runs decrease L. The score-grid selections under-blur in 4/4 generated and 4/4 real configurations; on those generated configurations, PnP instead over-blurs in 4/4. This directly demonstrates why PnP-grid directions cannot be interpreted as profile-MAP directions. The pilot remains small, approximate, unconstrained, and without a global-optimality certificate; it is not an improved restoration method.

All 336 stored endpoints are then re-evaluated with four probes per profile. All 32/32 probe-specific selections and 8/8 four-probe-mean profiles under-blur; maximum endpoint q SD is 64.8 nats. A B1 check reconstructs the original stored q to within 0.0234 nats. Exact low-noise rankings remain confined to near-zero blur, whereas all high-noise probes select 0.375. These replications support direction stability, not an exact global argmin.

A pre-fixed full-FFT spectral correction changes 3/80 PnP-grid minima on each generated-image pool. Mean absolute relative kernel error changes from 0.5323 to 0.5365 on the main pool and from 0.5615 to 0.5531 on the second, so no consistent algorithmic benefit or downstream restoration improvement is claimed.

## Model and resolution pilots

We evaluate two official OpenAI ADM checkpoints on real BSDS images using an explicitly defined continuous VP-ODE surrogate, not a discrete DDPM variational bound. The unconditional ImageNet-64 prior is positive in 96/96 tested pairs. The fixed-label conditional native-128 prior is positive in 16/16 pairs at σ = 0.05 but 0/16 at σ = 0.20. The table includes both Λ and Λ/N per scalar intensity. DOP853 checks preserve the reported signs at the tested points. Architecture, training data, conditioning, and resolution differ jointly, and the Gaussian adapter check does not prove equality with the original discrete models' normalised densities. These are exploratory surrogate-prior results, not a general architecture or resolution theorem.

## Relation to prior work and scope

Nguyen, Pauwels and Weiss (*A Morse-Bott Framework for Blind Inverse Problems*) already report the qualitative preference and analyse MAP failure. Karczewski et al. (*Diffusion Models as Cartoonists*, ICLR 2025) find blurry or cartoon-like high-density samples. Our contribution is paired quantitative measurement, empirical spectral scaling, controlled reference comparisons, and precise delimitation of the optimisation interpretation—not discovery of blur preference, a universal law, or an improved restoration algorithm.

The study does not establish real-FFHQ validity, cross-architecture or cross-resolution universality, prior independence, a general failure of second-order methods, or a decided validity boundary for Lemma 12. Clean-pair preference is not noisy-posterior preference. PnP-grid selections are not certified profile-MAP solutions. We disclose a half-pixel kernel-construction error that affected thirteen scripts and retain corrected measurements and the audit trail.

## Confirmations

- The manuscript is original and has not been published or submitted elsewhere.
- All authors have approved the manuscript and agree with its submission.
- The reproducibility package contains scripts, criteria, judges, raw results, figure sources, image indices, random seeds, and checkpoint hashes. Pre-registration records and post-hoc analyses are labelled separately. A local archive `results/review2_reproducibility.zip` and SHA-256 manifest have passed integrity and exclusion checks; executable guards cover all eight reference kernels and 69 raw synthetic-operator rows. No model weights or credentials are included. An anonymised public repository URL and immutable archive identifier must still be inserted before submission.

We would be glad to address any questions.

Sincerely,  
Bin Nong  
Tianfu College of Southwestern University of Finance and Economics  
Mianyang, China  
nongbin@tfswufe.edu.cn
