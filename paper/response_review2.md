# Response to review2（逐条核查工作稿）

以下对应 `paper/review2.md`。所有正式数值均可在 `results/` 的原始 JSON 中复核；“限定范围内解决”不表示已经证明跨模型、跨数据集或全局最优的普遍结论。

## Major 1：创新性与新增科学认识

**回应：已重新定位贡献，保留明确的经验性边界。**

- 不再声称发现 blur preference 或 MAP failure；明确承认 Nguyen et al. 与 Karczewski et al. 的优先工作。
- 新内容是配对 nats 量纲测量、共同频谱坐标、噪声依赖、局部指数及积分形式、完整 FFT/RGB 参考误差和数值不确定性的分离。
- 新形状实验验证的是已测曲线的指数/归一化带迁移，不宣称无需测量即可预测任意新算子的 Λ 或形状因子。
- 形状描述量的失败、径向位置敏感性和 `t=0.16` 的拟合退化均保留；没有统一机制或普遍算法收益。
- 位置：主稿 Introduction、Related work、Scaling law、Limitations。

## Major 2：MAP 目标与 PnP-HQS 的数学一致性

**回应：已实质修改并完成直接梯度复核；不宣称全局 MAP。**

1. 实际实现是 Douglas–Rachford 型 PnP，不是原先泛称的 HQS；正文给出完整更新。
2. 明确写出 `grad q_t=(x-D_t)/t^2`，以及真正 prox 所需的隐式条件，denoiser 一般不对应 `prox(q_t/rho)`。
3. 区分平滑先验代理目标与原始干净图像后验；`t=sigma_n` 不自动给出正确的 clean-image prior。
4. 79/80 改称 PnP-grid 最低值计数；负 no-blur 差仅作为可行点证据。
5. 新直接 score-gradient pilot：21 点、双初值、600 Adam、无裁剪、初末 ODE q、方向中心差分。336/336 次目标下降。
6. 生成图直接 score-grid 4/4 欠模糊，旧 PnP 同配置 4/4 过模糊；真实图直接 score-grid 4/4 欠模糊。两者不能混作同一 MAP 结论。
7. 全部336个终点完成四探针复评：32/32 单探针和8/8四探针均值 profile 均欠模糊；最大端点 q SD 64.8 nats。B1 原探针复现误差不超过0.0234 nats。
8. 仍保留非零终点梯度、无全局证书、无约束像素越界和小样本限制；不称为改进算法。

位置：`paper/main.tex` 的 Objective/optimisation、PnP profiles、Direct score-gradient、Endpoint trace replication。
文件：`score_profile_generated.json`、`score_profile_real.json`、对应 `_recheck.json`、`score_profile_probe_b1.json`。

## Major 3：概率流 ODE 与 Hutchinson 数值可靠性

**回应：已在限定范围内完成全曲线数值传播。**

- 32 次重复仍明确限定为单图/算子、两个 t。
- 新增4图×3族×20算子×4独立 Rademacher；同一重复内跨全部 u 共享探针，重复间独立，并保存原始向量。
- 三族均值 b∞/A/q 的跨探针 SD 为0.00240/0.0954/0.0317；各族 b(u) 单点最大 SD 为0.0287/0.0394/0.0467。
- 同探针 RK45 1e-5 与1e-6：最大 |ΔΛ|=4.33/4.11/5.27 nats；最大 |Δb(u)|=0.0036/0.0117/0.0342。
- 图像 bootstrap、探针、容差、平滑敏感性分别报告，没有合成虚假的总置信区间。
- 严格全曲线臂是 RK45，不冒充所有求解器/模型/噪声级别的普遍收敛证明。

文件：`law_stability.json`、`law_stability_analysis.json`。

## Major 4：缩放律参数、形状因子和噪声范围

**回应：已补齐估计细节、区间、敏感性并收缩结论。**

- 明确先取120图算术均值再取 log；20公共点拟合无权重五次多项式；区间内19点拟合 b∞+Au^q。
- 2000次共享图像索引 bootstrap：b∞ 95% CI [-0.7367,-0.7221]，A [-7.6602,-6.8310]，q [3.3247,3.5231]。
- 报告 first-20/first-60/last-60 子集。
- 次数3–6与三个拟合窗口导致明显参数变化，因此不宣称唯一参数；测量区间内 b(u) 相对稳定，仍保持负值和随 u 变陡。
- 形状因子改称 M(shape)，与局部冻结指数诊断 C(u) 分开。
- 有效范围、参数随 t 漂移和 t=0.16 退化均明确。

文件：`review2_statistics.json`。

## Major 5：真实图像统计说服力

**回应：统计错误已修正；更广自然图像覆盖仍是限制。**

- 以图像为独立单位，低噪声32/32的精确95% Clopper–Pearson区间为 [0.8911,1]，不再报告退化 [1,1] 作为总体概率证据。
- 高噪声四核同时为正为27/32，区间 [0.6721,0.9472]；保留120/128原始 pair 计数但明确其图内依赖。
- 仍仅使用前32张排序 BSDS 测试图；没有冒充完整 BSDS300、其他自然数据集或真实 FFHQ 验证。

## Major 6：模型架构、分辨率和 ADM surrogate

**回应：已收缩表述并增加归一化；未声称受控架构定理。**

- ADM128 始终标为固定 label=0 条件模型，无 classifier guidance。
- 连续 VP-ODE 明确为 surrogate，不是离散 DDPM VLB，也不声称与原离散模型归一化密度完全相同。
- 架构、训练集、条件和分辨率共同变化，不能归因于纯分辨率。
- 表中新增每标量强度 Λ/N：ADM64三档2.437/2.061/2.013，ADM128两档0.689/-0.261。
- 未增加真正连续时间非 EDM score model；该项仍列在 Limitations。

## Major 7：代码与实验数据可复现性

**回应：本地复现包已完成；公开 URL/DOI 仍需作者发布。**

- `results/review2_reproducibility.zip` 与 `review2_repro_manifest.json` 包含源码、原始 JSON、日志、环境版本、图像索引、随机探针和检查点哈希；不含模型权重或凭据。
- 实际环境：`review2_environment.txt`；EDM/ADM 权重 SHA-256 已固定。
- `validate_review2_physics.py` 已实跑通过8个参考核及69条原始合成记录的归一化、中心对称、共轭权重、滤波和 Parseval 守卫。
- score 端点图像和四探针原始结果、原始探针向量均已保存。
- **匿名公开仓库 URL 与不可变 DOI 不能由本地归档代替，必须由作者账号发布。**

## Minor 1：full-FFT 坐标下的主要缩放律

已在相同 Λ 上重拟合：三族均值变为 (-0.7191,-7.1706,3.3881)，图像 bootstrap 区间为 [-0.7259,-0.7116]、[-7.6008,-6.7827]、[3.2930,3.4877]；每族 rms 变化小于0.001。形式保留，但参数不视为不变。

## Minor 2：跨分辨率归一化

ADM 表同时报告 Λ 和 Λ/N，N=3HW，单位为每标量强度 nats。

## Minor 3：clean-image 与 noisy-posterior preference

摘要、测量章节、Scope 和 Conclusion 均明确两者不同；clean-pair 正偏好不能决定 noisy posterior 符号或算子估计方向。

## Minor 4：C(shape) 与 C(u) 记号冲突

积分形状因子改为 M(shape)；图中 C(u) 只保留为局部冻结指数归一化量，图题明确二者不同。

## Minor 5：局部二阶检验的方向和幅度

补入 ±0.25 的具体误差，并与 ±0.5 对照。较小拟合间距没有拯救到 s=1 的外推；共享探针只去除该线上的独立迹抖动。仍只有一个实际模糊方向/图/噪声，不宣称多方向或 full-Hessian 失败定理。

## 尚未由本轮实验消除的外部限制

真实 FFHQ、更大自然图像集合、受控跨架构/分辨率比较、统一形状机制、更大直接优化与全局收敛、有效复原收益仍未建立。公开匿名仓库和永久 DOI 需要作者执行发布。
