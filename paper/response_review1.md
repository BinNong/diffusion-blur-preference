# Response to review1（逐条回复工作稿）

以下对应 `paper/review1.md`，以当前最终稿 `paper/main.tex` 和公开复现材料为准。后续 review2 引出的直接 score 优化、四探针复评和公开归档也一并列出。凡标注“限定范围内解决”的条目，均不表示已经证明普遍理论或全局最优。

公开材料：
- 仓库：https://github.com/BinNong/diffusion-blur-preference
- 不可变 Release：https://github.com/BinNong/diffusion-blur-preference/releases/tag/v1.0.1
- Zenodo DOI：https://doi.org/10.5281/zenodo.24279314

## Major 1：积分形式的符号不一致

**已解决。**

正确关系为

\[
\Lambda\approx \mathcal M(\mathrm{shape})u^{b_\infty}\exp\left(\frac{A}{q}u^q\right),
\qquad b(u)=\frac{d\log\Lambda}{d\log u}\approx b_\infty+Au^q.
\]

正文已从参考点 \(u_0\) 开始给出完整积分推导，并通过反向微分验证；指数项为 \(+A/q\)，不再是 \(-A/q\)。摘要、贡献、正文和图例已统一。当前参数约为 \(b_\infty=-0.73,A=-7.2,q=3.4\)，故 \(A/q\) 本身为负，但公式前不额外添加负号。

同时撤回错误的“冻结局部指数”写法 \(\Lambda=Cu^{b(u)}\)；图中 \(C(u)\) 仅作为固定 \(u\) 下的归一化诊断，与积分形状因子 \(\mathcal M(\mathrm{shape})\) 明确区分。

## Major 2：终端高斯修正与实际图像频谱

**已解决。**

- 不再使用 \(\|h*x\|^2=u\|x\|^2\) 的一般性错误假设。
- 终端项改为由 Parseval 恒等式得到的实际图像频谱表达：
  \[
  T(x,h)=\frac{\|x\|^2-\|h*x\|^2}{2\sigma_{\max}^2}
  =\frac{\sum_f(1-|\hat h(f)|^2)|\hat x(f)|^2}{2\sigma_{\max}^2}.
  \]
- 对收缩算子给出与图像频谱无关的保守上界 \(0\le T\le N/(2\sigma_{\max}^2)=0.96\) nats。
- 在120张生成图和8个原始参考核的960个组合上直接计算，实际终端项为0.0066–0.2436 nats；退役的 \(u\) 近似最大误差为0.5898 nats。
- ODE 测量本身已经包含真实终端贡献，不额外施加基于 \(u\) 的二次修正。rFFT 坐标与完整 FFT 共轭权重也已区分。

位置：Measurement / terminal-term audit。
文件：`review_numeric.json`、`fullfft_reference_audit.json`。

## Major 3：MAP degradation identity 的定义和理论意义

**已在正确定义和理论边界内解决；不再声称新 MAP 定理。**

- 数据项统一记为 \(D_\theta(x;y)=\|h_\theta*x-y\|^2/(2\sigma_n^2)\)，先验势能统一为 \(q_t=-\log p_t\)，避免同一 \(q\) 同时表示 data fidelity 和 prior。
- 完整定义 \(L=D+q_t\)、\(L_\star\)、\(D_\star\)、\(\Delta q_t\)、\(\Lambda_{\rm clean}\) 和 PnP 输出点的目标差。
- 原式现为
  \[
  E_{\rm PnP}=\Delta q_t-\Lambda_{\rm clean}(t)+(D_{\rm PnP}-D_\star),
  \]
  由直接代入完整推导，明确是代数恒等式。
- \(10^{-14}\) 残差仅验证浮点记账，不再作为新理论贡献。
- 负值只证明所构造的无模糊可行点优于生成对，因而不必然断言 PnP 已求得 profile minimum。
- 不再声称 prior independence；正文明确两项附加项不能丢弃，单一 \(\Lambda_{\rm clean}\) 预测器已被拒绝。
- 后续又区分了 PnP 固定点与真实目标梯度，并完成直接 score-gradient 小样本复核及四探针端点复评；仍不宣称全局 MAP。

位置：Objective, optimisation map, and an algebraic no-blur gap。

## Major 4：对 Lemma 12 的逻辑过度延伸

**已解决，结论降为样本级符号一致性。**

- 明确原始条件具有 \(\forall x,\forall h\) 的普遍量词，15个有限配置不能证明该条件。
- 15/15 仅报告所测配置的符号一致性，不再称为“精确判定已发表引理的适用边界”。
- 对固定图像/核，样本符号等价只是定义的直接结果，不用于证明或反驳普遍假设。
- “clean”明确指无观测噪声的输入图像对，不是 \(t=0\)；实验仍满足 \(t\ge0.002\)，在 MAP 实验中显式取 \(t=\sigma_n\)。
- 平滑先验代理目标与原始 \(p_0\) clean-image posterior 也已区分。

## Major 5：Second-order Methods 的概念混淆

**已通过改题、重构和限定范围内的直接方向检验解决。**

- 标题已改为 *Quantifying Blur Preference in Diffusion Priors: Spectral Scaling and Implications for Blind Image Deconvolution*，删除“Why Second-Order Methods Fail”。
- 明确区分局部 Hessian 的 Taylor 近似、全局/平稳高斯协方差和算子频谱泛函。
- 三种参考模型的结果只用于说明特定 Gaussian/spectral reference 的误差，不能推广为全部二阶方法失效。
- 新增直接沿实际模糊方向评估 ODE log-density potential 的方向二次检验，而非用 denoiser Jacobian 冒充 log-density Hessian。
- 报告 \(\pm0.5\) 与 \(\pm0.25\) 两种拟合间距的具体误差；较小间距没有拯救到 \(s=1\) 的外推。
- 仍仅覆盖每个图像/噪声设置的一个实际模糊方向，不宣称 full-Hessian 或所有局部方法的失败定理。

位置：Limits of stationary Gaussian and spectral approximations。

## Major 6：过模糊实验的独立性、核范围和优化细节

**已做实质性补充，并据负面结果进一步收缩结论。**

### 6.1 图像内依赖

20张图的四个噪声配置按整图 cluster bootstrap 重采样。79/80 对应比例的95%区间为 [0.9625,1]，平均因子1.5323的区间为 [1.4938,1.5656]。这些不是80个独立样本，也不是独立观测噪声重复。

### 6.2 核形状、强度和真实图像

- 新增8张生成图上的 Gaussian 强度1/2、Defocus 1.5、Motion 3 与两个噪声级别。
- 多核结果只有19/64过模糊、37/64欠模糊；\(\sigma_n=0.5\) 时0/32过模糊，推翻普遍过模糊读法。
- 真实 BSDS 8图 pilot 为4/16过模糊、12/16欠模糊，且 profile 很平坦。
- 两个20图池均明确为同一 EDM 生成池的不同索引，不再称为真实 held-out FFHQ 或独立真实队列。

### 6.3 优化协议与 MAP 一致性

- 完整公开21点网格、240次 PnP–Douglas–Rachford、双初值、裁剪和固定预算；旧称 HQS 已纠正。
- 明确 denoiser 一般不是 \(q_t\) 的 prox，因此79/80只称为 **PnP-grid 最低值计数**，不再当作真正 profile-MAP argmin。
- 新增直接 score-gradient pilot：实际目标梯度、600 Adam、双初值、相同21点、初末 ODE q 和方向梯度检查。
- 336/336 次直接优化目标下降；32/32单探针和8/8四探针均值 profile 均欠模糊，进一步说明 PnP 方向不能自动解释为 MAP 方向。
- 仍保留非零终点梯度、无全局证书和小样本限制。

### 6.4 因果解释

除固定网格点外，已在经验选择的 PnP-grid 点分解 \(D\) 和 \(q\)，但明确这些描述性分解不证明因果。直接 score pilot 也只支持算法解释边界，不证明普遍算子估计规则。

文件：`overblur_cluster.json`、`overblur_generalization_analysis.json`、`real_profile_analysis.json`、`score_profile_analysis.json` 及对应原始文件。

## Major 7：跨模型、数据分布和分辨率泛化

**限定范围内补充，普遍性问题仍未完全解决。**

- 主实验仍限定 EDM/FFHQ 64×64 和生成图池；EDM/AFHQv2 仅为有限的第二 checkpoint 比较。
- 新增32张真实 BSDS 测试图、384个 clean-pair 测量；低噪声256/256为正，高噪声120/128原始 pair 为正。图像独立单位的精确区间已报告，不把 BSDS 冒充真实 FFHQ。
- 新增官方 OpenAI ADM ImageNet-64 无条件模型，96/96为正。
- 新增原生 ADM ImageNet-128 固定 label=0 条件模型；\(\sigma=0.05\) 为16/16正，\(\sigma=0.20\) 为0/16正。
- ADM 使用明确构造的连续 VP-ODE surrogate，不是离散 DDPM VLB；解析高斯检查不证明连续 surrogate 与原离散模型密度完全相同。
- ADM128 同时改变架构、训练集、条件和分辨率，不能归因为纯分辨率效应。表中同时报告 \(\Lambda/N\)。
- 真实 FFHQ、真正连续时间非 EDM 模型、受控跨架构/分辨率规律和普遍自然图像结论仍未建立。

## Additional 1：Hutchinson、ODE 和图像方差

**已在分层的限定范围内解决。**

- 原4次388 nats仅保留为试点尺度。
- 新32次独立重复：低噪声均值31851.9、SD 306.5、SE 54.2 nats；\(t=0.20\) 均值195.3、SD 58.0、SE 10.3 nats，均值95%区间约[175.2,215.4]。
- 因此162 nats的单次点估计不再被当成无不确定性的显著结果。
- 同点共享探针的 RK45/DOP853 与容差检查分别报告，不与图像方差混合。
- 后续完整曲线复核又加入4图×3族×20算子×4探针，以及同探针严格容差拟合；图像 bootstrap、探针、ODE 和平滑敏感性分开报告。

文件：`review_numeric.json`、`law_stability_analysis.json`、`review2_statistics.json`。

## Additional 2：Table 1 的统计代表性

**已解决。**

- 表题和正文明确 Table 1 是一个 EDM/FFHQ 生成图的单图噪声扫描，不是120图均值，也不再误称真实 held-out image。
- 主图和缩放律使用120个生成图/算子；真实图像实验单独报告32张 BSDS。
- 每个数值实验的图像数、来源和重复次数均在对应章节、表题或 Limitations 中说明。

## Additional 3：代码、预注册和原始数据链接

**已解决公开发布问题。**

- 公开仓库、不可变 Release 和 Zenodo DOI 见本回复开头。
- 仓库包含脚本、原始 JSON、日志、图像索引、环境版本、检查点哈希、原始探针向量和 manifest。
- 预注册判据和自述日期直接保留在公开脚本的 docstring 中；审稿后分析明确标为 post-hoc。脚本日期与 GitHub 提交时间均不冒充独立时间戳认证。
- 自动物理守卫已实跑通过8个参考核及69条原始合成记录。
- 模型权重和凭据不包含在归档中。

## Additional 4：参考文献和已有工作

**已解决。**

- Buskulic–Calatroni 已更新为 *2026 European Signal Processing Conference (EUSIPCO)*，并保留 Paper 1505/session 信息。
- Karczewski et al. 已按 ICLR 2025 正式论文引用，并在 Related work 中讨论高密度区域与模糊/卡通图像。
- Nguyen–Pauwels–Weiss 的 Morse–Bott 框架、blurry trap、MAP failure 和 MNIST 反例均被明确讨论。
- 本文不再声称首次发现 blur preference；贡献定位为配对定量测量、经验频谱标度、参考近似误差和优化解释边界。

## Additional 5：实际算法价值与复原指标

**未声称已解决算法改进；已用完整负结果回应并删除收益主张。**

- 预先固定的 full-FFT 频谱高斯修正只改变每个池3/80个 PnP-grid 最低点。
- 平均绝对相对核误差在主池0.5323→0.5365，在第二池0.5615→0.5531，方向不一致。
- 因未建立稳定核估计收益，没有选择性报告 PSNR/SSIM，也没有把未进行的 LADiBI 对比伪装成结果。
- 当前论文明确是测量与诊断研究，不提出改进后的盲反卷积算法。实际复原收益仍是开放的科学问题。

## 最终定位

当前稿件保留的核心是严格限定范围的定量测量、经验频谱行为、参考模型误差和盲反卷积优化解释。统一机制、真实 FFHQ、普遍跨架构规律、全局 MAP 证明和有效复原算法均未宣称完成。
