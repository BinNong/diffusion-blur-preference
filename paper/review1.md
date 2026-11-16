# IEEE Transactions on Image Processing（IEEE TIP）审稿意见

稿件题目： Blur Preference under Diffusion Priors: A Measurement, a Scaling Law, and Why Second-Order Methods Fail in Blind Deconvolution



建议：Reject（当前版本不建议接收）

需要实质性修改



我对这篇稿件的11页正文、数学公式、实验图表及参考文献进行了检查。

我的总体判断是：论文选题有一定新意，作者也做了不少实验，但目前存在若干影响核心结论可信度的数学问题、实验设计问题和理论论证问题。对于 IEEE TIP 的发表标准，我认为当前版本尚不成熟。

尤其值得注意的是，文章并非完全没有价值。作者试图回答一个值得研究的问题：

> 扩散模型作为图像先验时，是否天然倾向于给模糊图像分配更高的概率密度？这种偏好是否会影响盲反卷积中的模糊核估计？

这是一个有意义的研究方向。但是，从提出一个可测量的现象，到建立普遍性的标度规律，再到解释盲反卷积的 MAP 失败，文章中间存在一些尚未解决的逻辑跳跃。

# 一、论文总体评价

## 1. 论文主要贡献

作者提出了一个核心测量量：

\\[ \Lambda(t,u) = \log p_t(h\*x)-\log p_t(x) \\]

其中：

- \\(x\\)：原始清晰图像；
- \\(h\*x\\)：模糊后的图像；
- \\(p_t\\)：扩散模型在噪声水平 \\(t\\) 下的概率密度；
- \\(u=\langle |\hat h|^2\rangle\\)：模糊算子的平均频谱保留功率。

当 \\(\Lambda>0\\) 时，表示扩散先验赋予模糊图像更高的概率密度。

论文围绕这个量展开了四项主要研究：

| 研究内容                 | 作者的主要结论          | 我的评价        |
| -------------------- | ---------------- | ----------- |
| Blur Preference 测量   | 模糊图像具有更高的概率密度    | 有研究价值       |
| Scaling Law          | 建立了统一的标度关系       | 需要进一步核查数学表达 |
| Second-order Failure | 高斯及频谱二阶近似无法解释该现象 | 结论表述过于宽泛    |
| MAP Degradation      | 模糊偏好会导致盲反卷积过模糊   | 实验支撑还不充分    |

## 2. 值得肯定的地方

首先，这篇论文有几个优点。

第一，研究问题具有一定新颖性。

作者不是简单地提出另一种扩散模型反卷积算法，而是从概率密度的角度研究扩散先验本身的行为。

第二，实验组织相对完整。

包括不同模糊核、不同扩散噪声水平、不同频谱形状，以及额外的验证实验。

第三，作者主动披露了实验代码中的核构造错误。

第9节详细解释了半像素核偏移问题，并重新计算相关实验结果。这种处理值得肯定。

第四，作者尝试把先验概率偏好与 MAP 估计失败联系起来。

这使得文章不只是描述一个概率密度现象，而是试图讨论其对实际逆问题求解的影响。

但是，以下问题比较严重。

# 二、主要审稿意见（Major Comments）

## Major Comment 1：公式（4）的数学表达存在明显的符号不一致

严重程度：高

这是我在检查论文时发现的一个非常明确的数学问题。

论文第4页公式（4）给出：

\\[ \Lambda = C(\mathrm{shape})u^{b\_\infty} \exp\left(-\frac{A}{q}u^q\right) \\]

同时定义局部对数指数：

\\[ b(u) = \frac{d\log\Lambda}{d\log u} \approx b\_\infty+Au^q \\]

作者给出的实验参数为：

\\[ b\_\infty\approx-0.73 \\]

\\[ A\approx-7.2,\qquad q\approx3.4 \\]

但是，这两个表达式在数学上并不一致。

对论文公式（4）取对数：

\\[ \log\Lambda = \log C+b\_\infty\log u -\frac{A}{q}u^q \\]

两边对 \\(\log u\\) 求导：

\\[ \frac{d\log\Lambda}{d\log u} = b\_\infty-Au^q \\]

而不是作者给出的：

\\[ b\_\infty+Au^q \\]

如果作者希望保留：

\\[ b(u)=b\_\infty+Au^q \\]

那么积分后应该得到：

\\[ \boxed{ \Lambda = C u^{b\_\infty} \exp\left(\frac{A}{q}u^q\right) } \\]

注意，这里指数前面不应该再有负号。

这一点不仅出现在正文中，也影响摘要和贡献总结。

尤其是第6页 Figure 2 的图例实际上写的是：

\\[ b(u)\approx-0.73-7.2u^{3.4} \\]

这与正确积分后的表达式一致，却与正文公式（4）不一致。

审稿要求：

作者必须全面核查公式（4）、摘要、贡献列表、图例以及所有依赖该模型的数值拟合结果。

同时建议给出从局部指数到积分表达式的完整推导，而不只是声称公式是积分形式。

这属于必须解决的数学正确性问题。

## Major Comment 2：归一化概率密度的计算与终端高斯修正之间存在数学问题

严重程度：高

论文第3页是整篇文章的测量基础。

作者使用概率流 ODE 计算扩散模型的 log-density，并指出由于两个图像共享相同的求解器、积分步数和 Hutchinson 随机迹估计，因此配对对数密度差能够抵消积分中的常数项。

这个思路本身是合理的。

但是，作者在解释最大噪声水平 \\(\sigma\_{\max}=80\\) 处的终端高斯修正时，使用了：

\\[ \\|h\*x\\|^2=u\\|x\\|^2 \\]

进而认为终端高斯项的差异为：

\\[ \frac{(1-u)\\|x\\|^2}{2\sigma\_{\max}^2} \\]

这里存在一个问题。

根据 Parseval 定理：

\\[ \\|h\*x\\|^2 = \sum_f |\hat h(f)|^2|\hat x(f)|^2 \\]

而论文定义：

\\[ u=\frac{1}{N_f}\sum_f|\hat h(f)|^2 \\]

两者并不存在一般性的等价关系。

只有当图像频谱功率满足特定条件，例如频谱近似平坦时，才能将上述加权求和简化为：

\\[ \\|h\*x\\|^2\approx u\\|x\\|^2 \\]

但自然图像通常具有明显的非均匀频谱分布。

因此，作者使用平均频谱保留功率 \\(u\\) 来直接推导图像能量变化，并不严格成立。

这会影响作者关于终端高斯修正项小于0.9 nats的论证。

需要强调，这不一定意味着最终的概率密度差数值错误。对于频谱收缩的模糊算子，可以考虑使用更保守的能量上界。但是，当前推导不能直接支持作者给出的精确表达式。

审稿要求：

1. 给出终端高斯修正项的严格推导。
2. 使用实际图像频谱计算该修正，而不是直接使用平均保留功率 \\(u\\)。
3. 报告不同图像、不同模糊核情况下的终端修正误差。
4. 说明该误差如何影响最终的 \\(\Lambda\\) 测量值。

由于整篇文章的核心就是精确测量概率密度差，这个问题不能忽略。

## Major Comment 3：公式（8）的 MAP degradation identity 缺乏充分定义与理论解释

严重程度：高

这是我认为论文第7节最值得质疑的地方。

作者提出：

\\[ E\_{\mathrm{hqs}} = \Delta q-\Lambda\_{\mathrm{clean}} + (\mathrm{data}\_{\mathrm{term}}-\mathrm{data}) \tag{8} \\]

并将其称为：

> An exact MAP-degradation identity

作者强调，该恒等式在75个配置下的数值误差只有：

\\[ 1.6\times10^{-14} \\]

同时认为该关系具有 prior-independent 特性。

但是，目前的正文没有充分解释公式中各个量的定义。

例如：

- \\(\Delta q\\) 究竟表示什么？
- \\(\Lambda\_{\mathrm{clean}}\\) 是怎样计算的？
- \\(\mathrm{data}\_{\mathrm{term}}\\) 与 \\(\mathrm{data}\\) 分别是什么？
- \\(L\_\star\\) 如何定义？
- 为什么该式可以直接解释 global MAP degradation？

还有一个特别值得关注的问题：

在第3.1节中，作者将 \\(q\\) 定义为 data-fidelity term；但是第8节又写：

\\[ L=\mathrm{data}+q \\]

此处的 \\(q\\) 看起来又成为 prior term。

不同章节对同一个符号的使用不够清晰，容易导致公式的物理含义发生混淆。

此外，作者将恒等式计算误差达到 \\(10^{-14}\\) 作为一个重要贡献，但数值恒等并不能自动说明它具有新的理论意义。

如果公式只是通过定义重新排列目标函数得到，那么它的数值精度主要验证的是代数实现，而不是发现了新的 MAP 理论性质。

我希望作者回答一个关键问题：

公式（8）究竟提供了什么此前无法直接获得的理论信息？

作者需要：

1. 明确给出完整的目标函数。
2. 定义清楚每一项的数学含义。
3. 从目标函数出发给出完整推导。
4. 解释该恒等式为什么能够预测 MAP degradation。
5. 证明所谓 prior-independent 性质，而不是仅报告两个先验之间的数值接近程度。

目前第7节的理论贡献没有得到充分建立。

## Major Comment 4：关于已发表 Lemma 12 的结论存在逻辑上的过度延伸

严重程度：高

作者在第8页声称：

> A published sufficient condition, decided exactly.

并指出，只需要观察：

\\[ \operatorname{sign}(\Lambda\_{\mathrm{clean}}) \\]

就可以决定某个已发表引理中的充分条件是否成立。

作者在15个配置上获得了15/15的预测结果。

但这里存在一个重要的逻辑问题。

论文所讨论的充分条件涉及：

\\[ q(h\*x)\leq q(x) \\]

并且其原始条件涉及对所有图像 \\(x\\) 和所有模糊核 \\(h\\) 的约束。

这实际上是一个具有普遍量词的条件：

\\[ \forall x,\forall h \\]

作者现在使用15组有限实验配置来验证这个条件。

有限实验中的15/15吻合，并不能证明一个对所有图像和模糊核成立的充分条件已经被精确判定。

对于单独一个固定的图像和模糊核，概率密度比与势函数差之间的符号关系当然可以建立。

但这并不自动解决原始定理中对整个图像空间和核集合的普遍量化问题。

作者自己在第8页实际上也承认：

> For a single (x,h) pair the equivalence is close to a restatement of the hypothesis.

我认为这个承认非常重要。

建议作者：

- 明确区分单个样本的经验验证和原始引理的全局条件。
- 如果要声称决定了 Lemma 12 的 validity boundary，需要提供严格的数学推导。
- 否则，应当将贡献限定为在所测试配置上的经验性符号一致性，而不能称为精确判定了该定理的适用边界。

此外，\\(\Lambda\_{\mathrm{clean}}\\) 的计算方法也需要明确。第3节规定扩散噪声水平 \\(t\\) 不低于0.002，但第7节并未清楚解释 clean 对应的密度是否在 \\(t=0\\) 处计算，以及如何处理模型在该位置的数值限制。

## Major Comment 5：论文对 Second-order Methods 的定义不够严谨，存在概念混淆

严重程度：高

这是论文标题中的关键贡献之一。

作者声称：

> Why Second-Order Methods Fail in Blind Deconvolution.

第6节使用了三种参考模型：

1. Isotropic Gaussian reference。
2. Spectral functional。
3. Spectrum-matched Gaussian。

作者证明这些参考模型无法准确解释实际测得的 \\(\Lambda\\)。

但是，问题在于作者对 second-order method 的定义存在混淆。

论文第6页写道：

> Expand log \\(p_t\\) around its mode, keep the quadratic term, and compare the resulting Gaussians.

随后又将 second-order 近似解释为仅由图像功率谱决定的高斯先验。

实际上，这两者并不等价。

对概率对数密度进行局部二阶 Taylor 展开，得到的是由局部 Hessian 决定的近似：

\\[ \log p_t(x) \approx \log p_t(x_0) + \nabla\log p_t(x_0)^T(x-x_0) + \frac12(x-x_0)^T H_t(x_0)(x-x_0) \\]

这里的：

\\[ H_t(x_0) = \nabla^2\log p_t(x_0) \\]

是一个依赖于局部图像 \\(x_0\\) 的 Hessian 矩阵。

它不一定等于由全局图像协方差构造的逆矩阵，也不一定具有频域对角化结构。

因此：

作者证明了若干基于全局高斯假设和频谱统计量的近似方法失效，但并没有充分证明一般意义上的二阶近似方法失效。

这是两种不同的结论。

建议作者采取以下两种方案之一：

- 将论文中的结论严格限定为 stationary Gaussian/spectral approximations；
- 或者增加真正基于局部 Hessian 的二阶近似实验，验证该类方法是否同样失效。

另外，作者目前对三种参考模型的实验比较，并不足以证明：

> A second-order quantity is exactly wrong about which structural feature of the operator drives the effect.

这个结论需要更加谨慎。

## Major Comment 6：盲反卷积过模糊实验的说服力不足

严重程度：高

第8节是文章尝试连接理论和实际盲反卷积的关键部分。

作者报告：

- 20张真实图像；
- 4个较高的观测噪声水平；
- 共80个配置；
- 79个配置出现过模糊；
- 平均过模糊系数为1.5323。

这个结果看起来比较显著。

但是，进一步检查后，我认为目前的实验不足以支撑较为普遍的结论。

### 6.1 样本独立性问题

80个配置实际上来自：

\\[ 20\text{张图像}\times4\text{个噪声水平} \\]

同一张图像在不同噪声水平下被重复使用。

因此，这80个配置不能简单地看成80个相互独立的样本。

建议作者按图像进行 cluster bootstrap 或其他适当的分层统计分析，报告过模糊比例及其置信区间。

### 6.2 模糊核参数范围过于有限

作者主要使用：

\\[ \theta^\star=1.5 \\]

作为真实模糊强度，然后在21个离散点上寻找 profile optimum。

但实际盲反卷积中，真实模糊核可能具有不同的：

- 核尺寸；
- 空间分布；
- 方向性；
- 频谱衰减特性；
- 噪声水平。

仅使用一个真实模糊强度，无法证明该现象对不同模糊条件都具有普遍性。

### 6.3 Profile optimization 细节不足

论文定义：

\\[ \ell_y^{\mathrm{opt}}(\theta) = \min_x L(x,\theta;y) \\]

但没有充分说明：

- 图像优化具体使用什么算法？
- 使用多少次迭代？
- 是否使用多次随机初始化？
- 如何判断达到最优？
- 如何处理非凸优化中的局部极小值？
- 如何保证不同模糊强度下优化结果具有可比性？

这非常重要，因为整篇文章关于过模糊的主要结论都依赖这个优化过程。

另外，包含先验项的目标函数更准确地说是 profile MAP objective，而不是通常意义上的 profile likelihood。

### 6.4 因果关系尚未充分建立

作者使用固定网格点：

\\[ \theta=1.5,\qquad\theta=2.5 \\]

分析 data term 和 prior term 的变化。

但是，作者也明确表示，这不是在 profile optimum 处进行的分解。

因此，固定点上的先验项变化，并不能直接证明实际优化过程中最优模糊核偏移完全由这个先验项引起。

建议：

增加不同真实核强度、不同核形状、不同图像数据集的实验，并在 profile optimum 处分析两个目标项的变化。

否则，建议把结论限定为特定条件下观察到的经验性过模糊现象。

## Major Comment 7：实验的泛化能力不足，难以支撑面向 Diffusion Priors 的广泛结论

严重程度：高

这是影响论文发表价值的另一个重要问题。

目前论文主要依赖：

- EDM；
- FFHQ-64；
- 单一分辨率64×64 RGB；
- AFHQv2作为辅助先验替换实验。

作者在第10节也承认，目前没有真正不同的非 EDM prior 测试。

但是，论文标题和摘要多次使用：

- Diffusion Priors；
- Blind Deconvolution；
- Second-order Methods Fail。

这些表达容易让读者认为结论具有较强的普遍性。

实际上，不同扩散模型可能采用不同的：

- 网络架构；
- 训练数据分布；
- 噪声参数化；
- 训练目标；
- 预条件方式；
- 概率密度估计误差。

目前仅仅在FFHQ和AFHQv2两个数据先验之间进行比较，不足以支撑关于不同先验模型的普遍结论。

我建议至少增加以下实验：

| 实验方向   | 建议                              |
| ------ | ------------------------------- |
| 不同训练数据 | FFHQ、ImageNet、自然场景数据            |
| 不同模型架构 | EDM、其他 score-based diffusion 模型 |
| 不同分辨率  | 64×64 与 128×128                 |
| 不同模糊类型 | Gaussian、Motion、Defocus、非均匀模糊   |
| 不同图像内容 | 人脸、动物、自然场景                      |

尤其是图像内容方面，FFHQ具有较强的人脸分布特性。

目前尚不能确定论文发现的是一般自然图像分布中的规律，还是特定人脸生成模型中的规律。

# 三、其他值得关注的问题（Additional Comments）

除了上述主要问题，还有一些需要修改的地方。

### 1. 概率密度估计的数值精度

第3.3节称：

\\[ \operatorname{sd}(\Lambda)=388\text{ nats} \\]

但该估计仅通过4次独立 Hutchinson 随机迹估计获得。

4次重复不足以稳定估计复杂高维问题中的数值误差分布。

建议增加独立重复次数，并分别报告：

- Hutchinson 估计方差；
- ODE 数值积分误差；
- 图像样本之间的统计方差。

还要注意，第4页 Table 1 中 \\(t=0.20\\) 时：

\\[ \Lambda=162\text{ nats} \\]

小于报告的388 nats噪声标准差。

因此，摘要中强调严格单调衰减至162 nats，需要增加不确定性说明。

### 2. Table 1 的统计代表性

Table 1的表注明确说明，该表使用的是 single held-out image。

但是，摘要和整体论述容易让读者认为这些数据代表120张图像的平均结果。

建议明确说明每个噪声水平实验具体使用多少图像，并增加跨图像统计结果。

### 3. 缺少代码仓库的实际链接

第10节 Data and Code Availability 声称：

> All sweep scripts, pre-registered criteria, offline judges, raw JSON results and figure scripts are in the project repository.

但是论文没有给出实际仓库 URL。

作者还多次使用 pre-registered criteria 作为实验可信度的依据，却没有提供可核验的注册记录、时间戳和实验数据。

这不利于复现，也无法核实所谓预注册实验是否确实早于实验结果。

建议提供匿名 GitHub 仓库或完整补充材料。

### 4. 参考文献需要更新

参考文献[47]目前仅列出arXiv预印本，但该工作已经出现在EUSIPCO 2026会议论文集中，应更新正式发表信息。

[image](https://www.google.com/s2/favicons?domain=https://eurasip.org\&sz=32)

EUSIPCO 2026+1



此外，作者应该更加充分地讨论已有的高密度区域研究。

例如，ICLR 2025论文 Diffusion Models as Cartoonists 已经报告扩散模型的高密度区域可能包含模糊图像。

[image](https://www.google.com/s2/favicons?domain=https://proceedings.iclr.cc\&sz=32)

国际学习表征会议论文集



Nguyen等人的Morse–Bott研究也已经讨论了学习型先验下的 blurry trap 和 MAP 失败问题。

[image](https://www.google.com/s2/favicons?domain=https://www.alphaxiv.org\&sz=32)

alphaXiv



因此，本文的创新点应该更加明确地落在：

- 配对概率密度差的定量测量；
- 不同模糊核之间的标度关系；
- 对现有高斯频谱近似的定量分析。

而不能让读者误以为此前完全没有研究过扩散先验的模糊偏好。

### 5. 实际算法价值还需要进一步体现

论文主要提供的是现象分析和经验规律，目前没有提出能够利用这些规律改善盲反卷积的算法。

特别是论文已经引用了2026年IEEE TIP发表的 Blind Inversion Using Latent Diffusion Priors，这类工作可以作为讨论和实验比较的重要对象。

[image](https://www.google.com/s2/favicons?domain=https://pubmed.ncbi.nlm.nih.gov\&sz=32)

PubMed



建议增加一个实际应用实验：

利用测得的 Blur Preference 对盲反卷积算法进行偏差修正，然后比较修正前后的：

- Kernel estimation error；
- PSNR；
- SSIM；
- Reconstruction quality。

这样才能更加有力地说明该研究对图像复原领域的实际价值。

# 四、论文各部分的审稿评价

为了便于作者修改，我给出一个相对明确的评价。

| 论文部分                  | 评价     | 核心问题               |
| --------------------- | ------ | ------------------ |
| Abstract              | 需要修改   | 公式符号及结论范围          |
| Introduction          | 基本合理   | 创新贡献需进一步聚焦         |
| Related Work          | 需要补充   | 与已有研究的实质差异         |
| Measurement           | 需要大幅加强 | log-density计算和终端修正 |
| Scaling Law           | 存在错误   | 公式（4）符号不一致         |
| Second-order Analysis | 需要重构   | 二阶方法概念定义不严谨        |
| MAP Identity          | 存在明显缺陷 | 定义不完整，理论意义不清       |
| Over-blur Experiments | 证据不足   | 优化细节与样本独立性         |
| Limitations           | 比较诚实   | 但目前局限影响核心结论        |
| Conclusion            | 过于宽泛   | 需要与实际证据保持一致        |

# 五、审稿决定

# Reject

不建议当前版本直接接收 IEEE TIP。

主要原因：

1. 标度规律存在明确的公式符号错误。
2. 概率密度归一化及终端修正的推导不够严谨。
3. MAP degradation identity 缺少完整定义和理论支撑。
4. 对已发表定理适用边界的结论存在逻辑上的过度延伸。
5. Second-order methods 的概念定义与实际实验不完全匹配。
6. 盲反卷积实验不足以支撑较为普遍的结论。
7. 模型泛化能力和实验可复现性不足。

## 给作者最重要的修改建议

我认为，这篇论文真正值得保留的核心是：

扩散模型对模糊图像的概率密度偏好，以及这种偏好如何影响盲反卷积中的算子估计。

建议不要过度强调 MAP 理论新定理，也不要把三个特定高斯近似模型的失效扩大为所有二阶方法的失效。

更合理的论文定位是：

Quantifying Blur Preference in Diffusion Priors and Its Implications for Blind Image Deconvolution.

然后围绕三个核心问题重新组织论文：

核心贡献 1

Quantitative Measurement

建立严格、可复现的扩散先验模糊偏好测量方法。

核心贡献 2

Spectral Scaling Behavior

建立不同模糊核形状、不同噪声水平下的频谱标度规律，并提供更充分的跨模型验证。

核心贡献 3

Impact on Blind Deconvolution

通过理论分析及实际算法实验，解释并验证这种偏好如何影响模糊核估计。

## 六、我的最终判断

这篇论文有一个值得研究的科学问题，但目前存在一个比较突出的问题：论文的结论力度超过了实验和理论论证所能支持的范围。

如果我是 IEEE TIP 的正式审稿人，我不会仅仅因为公式（4）的符号错误就否定整篇文章。真正影响决定的是：

- 第7节的理论贡献没有讲清楚；
- 第8节的实验还不能充分证明作者提出的因果机制；
- 第3节的概率密度测量需要更严谨的数值论证；
- 整体研究的普遍性尚未得到充分验证。

我认为作者应当进行实质性重构，而不是简单修改文字、补充几张实验图就重新投稿。

建议下一步优先处理第3节、第5节、第7节和第8节。 这四部分直接决定整篇论文的科学可信度和发表价值。