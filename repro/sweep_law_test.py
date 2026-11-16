"""定律检验：用四个没测过的剖面族，检验 Lambda(shape,u) = C(shape) * u^{b(u)}

背景（RESULTS §3.9(l)）
-----------------------
受控实验（零相位谱门 g2=(1-beta)+beta*ref2，a(f)=beta*a_ref(f)）在 8 个参考剖面上得到：
  * 局部指数 b(u)=d log Lambda/d log u 跨 8 种形状普适（离散 9-29%）：
        u in [0.65,0.40]  b = -1.54 (范围 -1.68..-1.31)
        u in [0.40,0.30]  b = -0.87 (范围 -0.91..-0.82)
        u in [0.30,0.15]  b = -0.78 (范围 -0.90..-0.68)
  * 匹配 u 处，7 个真实平滑模糊族的 Lambda 只差 1.08-1.15x（0.11 dex）
  * 理想低通（硬截止）是离群点，只有平滑族均值的 0.65x

本实验加四个**从未测过**的剖面族，每个都是光滑单调但形状参数化不同：

  W1  Weibull p=1   g2 = exp(-(r/rc)^1)      指数衰减，重尾
  W2  Weibull p=2   g2 = exp(-(r/rc)^2)      == 高斯剖面（**内部对照**，应复现 gaussian 族）
  W4  Weibull p=4   g2 = exp(-(r/rc)^4)      超高斯，逼近理想低通
  B1  Butterworth n=1  g2 = 1/(1+(r/rc)^2)   Lorentzian，最重尾
  M2  two-scale    g2 = .5 exp(-(r/rc1)^2)+.5 exp(-(r/rc2)^2)  双尺度，非单峰曲率

**预言（在看到数据之前写定，防事后合理化）**：

  P1 指数普适：每个新族在 u in [0.65,0.40] 的 b 应落在 [-1.68,-1.31]；
    在 u in [0.30,0.15] 应落在 [-0.90,-0.68]。
  P2 前缀有界：匹配 u 处，C = Lambda * u^{-b(u)} 对新族应落在 7 个旧族的
    [min,max] 之内，或最多超出 1.3x。
  P3 趋近硬截止：W4（最接近理想低通）的 C 应是四个新族里最低的，
    并向理想低通的 0.65x 移动。
  P4 内部对照：W2 的 C 应复现 gaussian 族到几个百分点。

若 P1/P2 通过 => Lambda = C(shape) u^{b(u)} 是一条有预测力的定律（不只是拟合）。
若 P3 通过 => 同时给出了"平滑近似在哪里失效"的定量边界。

====================  第二组：陡峭度扫描（--families W2 W8 W16 W32 B8）====================
第一组已确定：12 个平滑剖面的 C 在 0.95-1.04，理想低通（硬截止）C=0.69，
而 Weibull p=4 只走完了"平滑均值→硬截止"这段路的 13%。
本组把这段路走完，并回答"哪个剖面描述量能预测 C"。

新族（参考强度仍定到 kunif_ref=0.920）：
  W2  Weibull p=2   （重复第一组的 W2，作为跨轮次重复性对照）
  W8  Weibull p=8
  W16 Weibull p=16
  W32 Weibull p=32  （逼近理想低通）
  B8  Butterworth n=8

**预言（在看数据之前写定）**：

  S1 单调：C 随陡峭度单调下降。Weibull 轴 C(2) > C(8) > C(16) > C(32)；
     Butterworth 轴 C(1)=1.022 > C(8)。
  S2 极限：把 C 对 1/p（或 1/n）拟合 C = C_inf + A/p^alpha，得到的 C_inf
     应与理想低通的 0.69 一致（落入其不确定度）。
  S3 指数不受影响：陡峭族的 b(u) 仍落在旧 8 族的范围内
     （[.65,.40]: [-1.68,-1.31]；[.4,.3]: [-.91,-.82]；[.3,.15]: [-.90,-.68]）。
     即退化只发生在前缀 C，不发生在指数 b——定律的结构比前缀更稳健。
  S4 重复性：W2 复现第一组的 C=1.030（两轮独立的 Hutchinson 噪声）。

====================  第三组：B2 预官的样本外检验 ====================================
第二组已在 17 个已测剖面上得到 C = 4.642 - 5.887*B2（knif=0.60，R2=0.825，LOO 0.051 dex），
其中 B2 = <a^2>/<a> 是"衰减剖面的二值化程度"。但那只是留一验证。
本组用五个**全新的形状类**（从未测过），参考强度统一调到 knif_ref=0.60，
**先用 B2 算出预官值，再测量**。

  已测 17 族在 knif=0.60 上的 B2 范围 = [0.607, 0.668]；实测 C 范围 = 0.687-1.038

  RP35  ramp q=0.35     幂律斜坡        B2=0.607  C_pred=1.070  （内插，在下边界）
  RP75  ramp q=0.75     幂律斜坡        B2=0.624  C_pred=0.968  （内插）
  LG35  logistic w=0.35 logistic 边     B2=0.628  C_pred=0.944  （内插）
  TS2   twoscale R=2    双尺度高斯混合   B2=0.685  C_pred=0.608  （近外推）
  EF35  erf w=0.35      erf 边          B2=0.704  C_pred=0.496  （近外推）

**预官（在看数据之前写定；C_pred = 4.642 - 5.887*B2，knif=0.60）**：

  P1 内插检验：RP75→0.968, LG35→0.944（B2 落在已测范围内）。
     若 B2 是 C 的充分描述量，实测 C 应落在预官的 ±0.05 dex 内
     （= 留一验证给出的样本外精度 0.051 dex）。
  P2 边界检验：RP35→1.070（B2 在已测范围下边界）。允许偏差，
     但不应高于 1.10。
  P3 外推检验：TS2→0.608, EF35→0.496。允许系统性偏差，但顺序必须
     TS2 > EF35，且都低于内插族。
  P4 若内插两个整体失败，说明 B2 只是必要非充分——剖面形状还有 B2 抓不到的维度。

====================  第四组：tw_lev 定律的预官检验（2026-10-18，跑前写定）====================
第三组推翻了 B2。hunt_descriptor.py 在 22 个已测剖面上换了一批**尺度不变**的描述量，
胜出者是**相对过渡带宽度**

    tw_lev = (r at a=0.9*range - r at a=0.1*range) / (r at a=0.5*range)

（range = 径向均值剖面 max-min；a 缩到 knif=0.60。尺度不变，所以不受 beta 压缩与
knif_ref 差异影响——这正是 B2 只齐次一次、必须靠标度协变技巧才能跨组比的根本弱点。）

在 22 个已测剖面上（C 取共同 u 网格 {0.4, 0.65} 的中位数）：
    C = 0.696 + 0.377 / (1 + (tw/0.47)^(-1.09))        R2=0.909
    LOO RMSE 0.0147 dex，LOO 最大残差 0.0358 dex
对照 B2（同 C 惯例）：C = 2.333 - 2.185*B2，R2=0.310；匹配 B2 处 C 的 max/min = 1.475
    tw_lev 的同一指标 = 1.030
且 tw_lev 通过了**语义检查**（validate_tw.py）：7 个族扫形状参数、32 项单调性检查全过
（"边更宽 => tw 更大"），而 B2 与 tw 反向运动。所以 tw_lev 不是又一个能拟合的标量。

物理读法：衰减在频域上"滚过去"的带宽与它自身的径向尺度可相比时（tw >~ 1），C ≈ 1
（线性/高斯 regime）；只有滚降带远窄于尺度（tw << 1，算子像尖锐谱投影）时 C 才下降。
这把 §3.9(o) 里"铺开 vs 二值化"的定性说法变成了一个可测量的刻度。

本组加 6 个从未测过的算子（参考强度仍定到 knif_ref=0.92，targets 用默认 0.35/0.60/0.85，
使新族落在与已测 22 个共同的 u 网格 {0.4, 0.65} 上）。其中 TK30/TK50 是**全新形状类**
（升余弦/Tukey 边：平通带 + cos 过渡 + 平阻带，真实滤波器的 canonical 形状）。

  族      kind        par   tw_lev   C_pred(tw)   C_pred(B2)   C_pred(supp20)
  TK30    tukey       0.30   0.209      0.806        0.918          0.821
  TK50    tukey       0.50   0.395      0.866        0.927          0.859
  EF60    erf         0.60   0.686      0.922        1.016          1.032
  W3      weibull     3.00   0.959      0.954        0.952          0.936
  LG60    logistic    0.60   1.133      0.968        1.021          1.032
  TS3     twoscale    3.00   2.549      1.021        0.988          1.010

（C_pred(B2) 用第三组否掉的那个式子在新 C 惯例下重拟合；supp20 是 hunt 里 R2 次高的候选。
三列分歧最大的 TK30（13.9%）和 EF60（11.8%）是**对抗性**探针。）

**预言（在看数据之前写定）**：

  T1 排序（无需容差）：6 个新族的实测 C 必须随 tw_lev 单调递增。
     tw 跨度 0.209→2.549（12x），随机排序全对的概率 1/720。

  ----------------------------------------------------------------------------------------
  **修正案（2026-10-18 13:05，在看到任何 C 之前声明；原稿 md5=3738455f...，跑已启动）**
  上面 T1 的严格形式**统计上不公平**，必须改。理由（跑前可算）：
    * 相邻点的 C 预官间距（dex）：TK30→TK50 0.031、TK50→EF60 0.027、
      EF60→W3 0.015、W3→LG60 0.006、LG60→TS3 0.024。
    * 单点 C 的噪声约 0.02 dex（Λ 的标准误 140/1924=7.3%、216/3947=5.5%，
      来自本任务自己的打印）。
    * 于是 EF60→W3 只有 0.75σ、W3→LG60 只有 0.3σ。要求 5 个符号全对，
      即使定律**完全正确**也只有约 35% 概率通过——那是抛硬币，不是检验。
  改为：
    T1a 秩相关：6 个新族的 Spearman(tw_lev, C) >= 0.94（至多一次相邻逆序）。
    T1b 可分辨的三条：C(TK30) < C(EF60)（间距 0.058 dex = 2.9σ）、
        C(TK30) < C(TS3)（0.103 dex = 5σ）、C(EF60) < C(TS3)（0.044 dex = 2.2σ）。
    T1c 仅记录不作判据：EF60/W3、W3/LG60 两个符号（间距低于噪声）。
  披露：任务启动后 TK30 的两个算子已打印，TK30 的 C 技术上已可见；但排序检验
    需要全部 6 个族，我尚未计算任何 C。此修正只关于**判据的统计公平性**，
    不涉及任何阈值的事后放宽（容差仍 0.045 dex，来源不变）。
  ----------------------------------------------------------------------------------------
     容差不是硬编码的，取模型在**自己训练数据**上的 LOO 最大残差
     （共同 u 网格 0.0358 dex / 旧惯例 0.0452 dex，取大者 0.045）。
     两种 C 取值惯例给出的 C_pred 相差 <0.02，远小于容差，所以结论不依赖惯例。
  T3 对抗性判别：TK30 与 EF60 的实测 C 必须分别比 B2 的预言低 >=5%
     （B2 在这两个点上再次失败，而 tw_lev 给对）。
  T4 膝部：tw in [0.4, 0.7] 的 TK50 与 EF60，其实测 C 必须严格落在
     已测"陡峭簇"最高值 B8 的 0.834 与已测"饱和簇"最低值 W2 的 0.983 之间。
     （这一段正是已测 tw 覆盖的缺口 [0.37, 0.73]。）
  T5 形状无关性：两个 tukey 族（全新形状类）在匹配 tw 处必须与
     Weibull/Butterworth 轴给同样的 C——即 T2 对它们同样成立。
  T6 指数不受影响（与 S3 一致的复查）：6 个新族的 b(u) 仍落在旧范围
     [.65,.40]: [-1.68,-1.31]；[.4,.3]: [-.91,-.82]；[.3,.15]: [-.90,-.68]。

若 T1-T4 通过 => "C(shape) 由衰减剖面的相对过渡带宽度决定"是一条有预言力的定律，
  且它比 B2 更强（R2 0.909 vs 0.825、匹配处散度 1.030 vs 1.475、有语义检查）。
若 T3/T4 失败 => tw_lev 也只是必要非充分，记录失败并继续找判据。

====================  第五组：twod 定律的预官检验（2026-10-18，跑前写定）====================
第四组找到 $tw_{\rm lev}$ 并通过 5/6，但两个**底座剖面**（EF60/LG60）决定性地失败
（+0.064 dex = 16%，6-10 sigma）。机制：它们的过渡中心被推出频域，衰减无通带，
$tw_{\rm lev}$ 的 levels 取剖面自己的 range，于是量的是平坦剖面的斜率而非截止锐度。

hunt2.py / validate_twod.py 在 28 个已测剖面（22 基线 + 6 新族）上搜第二批描述量，
胜出者是把过渡宽度**除以相对动态范围**：

    twod = tw_lev / dyn,   dyn = (amax-amin)/<a>

物理读法：1/twod = r50 * dyn/w_abs = 特征半径 x 剖面在自身尺度上的总变化量（以 <a> 为单位）。
硬截止 => 变化量巨大 => twod->0 => C->0.70；缓坡 => twod->inf => C 饱和。
尺度不变（tw_lev 与 dyn 都尺度不变）。

28 个已测剖面上（饱和形式；定律本来就是饱和的，线性 R2 会冤枉它）：

    描述量                  R2sat   Spear   LOO dex   最紧bin   逐点最大残差
    twod = tw_lev/dyn      0.911   0.85    0.0144    1.008    0.035
    tw_lev                 0.827   0.69    0.0199    1.008    0.064(EF60)
    supp20                 0.816   0.75    0.0206    1.138    --
    B2                     0.457   -0.79   0.0389    1.475    --

V1 语义单调性 32/32 通过（7 个族扫形状参数）。V2 **弱样本外**：只用 23 个有通带剖面拟合，
再去预言 5 个底座剖面，残差全部 <=0.038 dex（EF60 从 tw_lev 的 +0.064 降到 +0.018）。
但 twod 是**看着全部 28 个剖面设计出来的**，所以 V2 只是弱样本外，必须再跑预官。

**冻结的定律（28 个已测剖面，共同 u 网格 {0.4, 0.65}）**：

    twod  :  C = 0.704 + 0.361/(1+(twod/0.421)^(-1.23))     R2=0.911
    tw_lev:  C = 0.718 + 0.315/(1+(tw/0.381)^(-1.68))       R2=0.827

本组加 8 个从未测过的族，参考强度定到 knif_ref=0.92，targets 用默认 0.35/0.60/0.85
（使新族落在共同 u 网格上）。4 个有通带 + 4 个底座剖面，twod 跨度 0.45 -> 26.6：

  族      kind        par   twod     dyn    phi    C(twod)预官  C(tw_lev)预官  分歧dex
  W6      weibull     6.00   0.453  1.087  0.000      0.893        0.909       0.008
  TK80    tukey       0.80   0.724  1.087  0.000      0.943        0.961       0.009
  RQ400   ramp        4.00   0.894  0.547  0.540      0.963        0.908       0.025  <- 底座
  TK95    tukey       0.95   0.983  1.087  0.000      0.971        0.986       0.007
  W15     weibull     1.50   1.786  1.087  0.000      1.013        1.014       0.000
  RQ200   ramp        2.00   3.456  0.259  0.828      1.040        0.972       0.029  <- 底座
  EF100   erf         1.00   4.528  0.212  0.861      1.047        0.978       0.029  <- 底座
  LG150   logistic    1.50  26.603  0.052  0.973      1.063        1.001       0.026  <- 底座(外推)

TK80/TK95 是升余弦（Tukey）边——第四组引入的全新形状类。LG150 的 twod 超出已测范围
上界（8.80），是**外推**探针（定律预言 C 饱和在 1.065）。

**预言（在看数据之前写定）**：

  U1 前缀有界：8 个新族的实测 C 落在 twod 定律预官的 +/-0.038 dex 内。
     容差不是硬编码的，取 twod 定律在**自己训练数据（28 个已测剖面）**上的
     LOO 最大残差 0.0383 dex。
  U2 结构判别（区分 twod 与 tw_lev）：4 个底座剖面（RQ400/RQ200/EF100/LG150）的
     平均 |残差| 在 twod 定律下必须**小于**在 tw_lev 定律下。
     （无阈值，纯比较式；twod 的预官比 tw_lev 高 0.025-0.029 dex，而单点 C 的
     噪声 ~0.01 dex，所以这个差别可测。）
  U3 排序（统计公平性先声明）：相邻点的 C 预官间距是 0.023 / 0.009 / 0.003 / 0.018 /
     0.011 / 0.003 / 0.006 dex，而单点 C 的噪声 ~0.01 dex——后四项低于噪声，
     要求 8 个全对即使定律正确也通不过。所以：
       U3a Spearman(twod, C) >= 0.90（至多一次相邻逆序）；
       U3b 只要求间距 > 0.02 dex 的两条：C(W6) < C(TK80)、C(W6) < C(LG150)。
  U4 形状无关性：两个 tukey 族（TK80/TK95，有通带）同样满足 U1。
  U5 指数不受影响（复查）：8 个新族的 b(u) 仍落在旧范围
     [.65,.40]: [-1.68,-1.31]；[.4,.3]: [-.91,-.82]；[.3,.15]: [-.90,-.68]。

若 U1-U4 通过 => 「C 由 twod = tw_lev/dyn 决定」是一条同时覆盖有通带与底座两个
  regime 的定律，它把第四组对 tw_lev 的部分撤回升级为"有通带只是 twod 的特例"。
若 U1/U2 失败 => 记录失败。按项目规矩，不为失败的预言改阈值。

====================  第六组：均匀传输算子——底座平台是不是「均匀缩放」？（2026-10-18，跑前写定）======
hunt3.py / diag_offset.py 发现：kref=0.92 的 6 个底座剖面，$tw_{\rm od}$ 跨 0.894->26.603
（**30 倍**）而 $C$ 只在 1.049->1.102（**1.05 倍**）之间——这是一个**平台**，
$tw_{\rm od}$ 曲线根本不适用于它。8 个候选描述量 + 5 种模型形式都没能吃掉这个偏移。

而这 6 个剖面有一个干净的共同点：**传输 $g_2=1-a$ 全被限制在 [0.35, 0.68] 这个窄带内，
既不到 0 也不到 1**（因为 knif_ref=0.92 的 beta 压缩给 g2 一个 0.348 的地板）。
对照 kref=0.60 的底座剖面（beta=1，g2 能到 0）：EF35/LG35/RP35/RP75，C 散在 0.900-1.028。

**物理读数**：当算子的传输被限制在窄带内，$h\star x$ 在每个频率上都被同等地衰减，
模糊偏好应当**退化为均匀振幅缩放的偏好**。这个读数有一个直接的判决性预言：
  均匀算子（$g_2\equiv c$，$a\equiv1-c$）的 $C$ 应该落在底座平台的 1.070 附近，
  而不是落在 $tw_{\rm od}$ 曲线给外推值的地方。

注意：均匀算子的 kunif = 1-c 由 c 决定，不能独立选 target，所以用 --uniform-c 单独构造。
取 c = 0.40 / 0.60 / 0.65 => kunif = 0.60 / 0.40 / 0.35 => u = 0.40 / 0.60 / 0.65，
其中 u=0.40 与 u=0.65 正是与全部已测剖面共同的网格点。

**预言（在看数据之前写定；平台均值 1.070 来自那 6 个已测底座剖面）**：

  V1 平台读数：均匀算子的 C（共同网格 u in {0.4, 0.65} 上取中位）落在
     1.070 的 +/-0.038 dex 内。容差不是硬编码的，取 twod 定律在 28 个训练剖面上的
     LOO 最大残差 0.0383 dex（与第五组同一个来源）。
  V2 指数普适：均匀算子在 u in [0.65, 0.40] 的 b(u) 仍落在旧范围
     [-1.68, -1.31]——即指数普适性对"最极端的均匀算子"也成立。
  V3 判别（区分「平台」与「twod 曲线外推」）：均匀算子的 C 必须 >= 1.05。
     平台 6 个值的范围是 [1.049, 1.102]；而 twod 定律在已测最大 twod=26.6 处给 1.063、
     在 twod=0.894 处只给 0.963。若 C 掉到 1.0 以下，说明平台的读数错了。

若 V1-V3 通过 => 「传输被限制在窄带内的算子，其模糊偏好退化为均匀振幅缩放的偏好，
  C ~ 1.070」是一条新的、可判决的陈述，也解释了为什么 twod 曲线在那里失效。
若失败 => 记录失败，平台的物理读数需要换解释。

====================  第七组：b(u) 的密网格与解析形式（2026-10-18，跑前写定）====================
定律 $\Lambda=C(\text{shape})\,u^{b(u)}$ 里 $b(u)$ 只有分段测量：
    u in [0.65,0.40]  b = -1.54 （8 族范围 -1.68..-1.31）
    u in [0.40,0.30]  b = -0.87 （范围 -0.91..-0.82）
    u in [0.30,0.15]  b = -0.78 （范围 -0.90..-0.68）
而**全部合成族的 u 只有 6 个点**：0.08 / 0.15 / 0.356 / 0.40 / 0.65 / 0.724。
所以 [0.40,0.30] 那一档实际是靠 0.356 与 0.15 两点跨 0.2 插值算出来的——
开放问题 8（5/8 个新族的 b 在该段贴边落在旧范围外 0.01-0.04）很可能就是这个插值噪声。

本组把 u in [0.15, 0.70] 加密到 13 个点，用 3 个形状迥异的族（W2 高斯=内部对照、
W8 陡峭、TK50 升余弦=不同形状类），另外给两个当时贴边违规的族（RQ400/EF100）补
u=0.30/0.35 两点，让它们的 [0.40,0.30] 档不再依赖插值。

密网格 u（13 点，kunif = 1-u 全部 <= 0.85 < kunif_ref=0.92）：
  0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70

**预言（在看数据之前写定）**：

  B1 结构：b(u) 对 u **单调不增**（u 越大 b 越负），且 b(u) > -2 处处成立
     （物理动机：先验沿模糊方向局部二次 => 指数应为 2，见 RESULTS §3.9 结论 15）。
  B2 低 u 极限：u < 0.26 时 b(u) 落在 [-0.95, -0.65]。
     （来源：旧 [0.30,0.15] 档实测 -0.78、8 族范围 -0.90..-0.68，各放宽 0.05。）
  B3 普适性（密网格上复查）：3 个族在**每一个**密网格 u 上，b 的极差 <= 0.50。
     （来源：旧 [0.65,0.40] 档 8 族的 b 极差是 0.37；放宽到 0.50。）
  B4 插值噪声假说：RQ400 与 EF100 在 [0.40,0.30] 档的 b（用**直接测得**的
     L(0.40) 与 L(0.30) 算，不插值）落在旧范围 [-0.91, -0.82] 内。
     若通过 => 开放问题 8 的"贴边违规"确实是插值噪声，b(u) 的普适性恢复。
  B5 无新结构：3 个族的 log Λ vs log u 曲线光滑单调，无新的拐点/振荡。

  b(u) 的提取方式（跑前定死，防事后挑方法）：
    对每个族，把 log Λ 作为 log u 的函数用**5 次多项式**拟合（13 个点、6 个参数，
    不会过拟合），b(u) = d logΛ / d log u 取该多项式的解析导数；
    标准误由对 20 张图的有放回重采样给出（2000 次）。
    候选闭式（按预官顺序）：(i) b = b_inf + A*u^q；(ii) b = b_inf + A/(1+u/u0)^p；
    (iii) b = -2 + A/u^q。哪个在 3 个族上都最好且残差在噪声内，就报哪个。

若 B1-B4 通过 => 得到 b(u) 的密网格测量 + 一个闭式候选，且 b(u) 的普适性在密网格上确认。
若 B4 失败 => 开放问题 8 是真效应（b(u) 并非完全普适），记录失败。

====================  第八组：b(u) 是否随噪声级 t 变？（2026-10-18 21:44，跑前写定）======
第七组拿到第一个闭式候选 $b(u)=b_\infty+A\,u^q\approx-0.70-6.4\,u^{3.1}$，但它只在
**3 个族、一个 $t=0.002$、一个模型**上拟过。开放问题 11 的后半：$b_\infty$ 与 $q$ 是
普适常数，还是随 $t$ 变？本组只改 $t$，其余逐项与第七组对齐。

物理动机（为什么"应该"变）
--------------------------
  * $\Lambda(t)$ 本身随 $t$ 掉两个数量级：$t=0.002$ 处 $+3.4\times10^4$ nats，
    $t=0.20$ 处 $+50$ nats；而高斯对照 $\Lambda_{\rm Gauss}$ 几乎不动
    （$+453\to+129$），放大倍数 $74.9\to0.39$（RESULTS §3.2 表）。
    即 $\Lambda$ 的 $t$ 依赖性**全部来自非高斯部分**。
  * 高斯先验可以闭式算：算子 $a(f)=\beta\,a_{\rm ref}(f)$，$u=1-\beta\langle a_{\rm ref}\rangle$，
    $\Lambda_{\rm Gauss}=\beta\sum_f a_{{\rm ref},f}|X_f|^2/(2(S_f+t^2))$ 对 $\beta$
    **严格线性** $\Rightarrow b_{\rm Gauss}(u)=-u/(1-u)$，与 $t$ 无关。
    所以 $t\to\infty$ 时 $b(u)\to-u/(1-u)$：$b(0.14)$: $-0.16$（实测 $-0.68$）；
    $b(0.85)$: $-5.67$（实测 $-4.28$）；$b_\infty\to0$（实测 $-0.70$）。
  * 若 $\Lambda=\Lambda_{\rm Gauss}+\Lambda_{\rm ng}$ 且 $b_{\rm ng}$ 与 $t$ 无关，则
    $b(u,t)=[\Lambda_{\rm G}b_{\rm G}(u)+\Lambda_{\rm ng}b_{\rm ng}(u)]/(\Lambda_{\rm G}+\Lambda_{\rm ng})$，
    权重随 $t$ 从 $b_{\rm ng}$ 移向 $b_{\rm G}$。这给出**方向性预言**
    $\Delta b_\infty>0$（$-0.70$ 朝 $0$ 走）、$\Delta q<0$（$3.1$ 朝 $1$ 走）。
    但量级估计很小：$t=0.05$ 处 $\Lambda_{\rm ng}/\Lambda_{\rm G}\approx9.7$，算得
    $\Delta b(0.14)\approx+0.05$、$\Delta b(0.85)\approx-0.13$ dex——与 $b$ 自身的
    bootstrap SE（$0.02$–$0.31$）同量级。**所以"符号对"不是判决性的，"测不到"更不等于
    普适**；必须同时报可探测下限（见 T6）。

实验设计（只改 $t$）
--------------------
  同样的 3 个族（W2 高斯=内部对照 / W8 陡峭 / TK50 升余弦）、同样 20 张图、
  同样的 $u$ 网格（17 个 target + $\beta=0.3/0.7$ + raw = 20 个 $u$ 点，
  拟合用其中 $u\in[0.13,0.86]$ 的 19 个）、同样的提取方式（$\log\Lambda$ 对 $\log u$
  的 5 次多项式 + 2000 次对 20 张图的有放回重采样）。kunif_ref 仍为 0.92。
  $t$ 取 0.005 / 0.01 / 0.02 / 0.05，全部落在该模型的训练噪声范围 $[0.002,80]$ 内。
  **$t=0.001$ 被主动排除**：实测 `EDMPrecond.sigma_min = 0`，所以 $t=0.001$ 不会报错，
  但它低于训练范围下界 0.002，属轻度外推——宁可不要这个点，也不要一个看着像物理的外推伪影。

**预言（在看到任何新数据之前写定）**：

  T1 守门：定律在新 $t$ 仍成立。每个 $t$、每个族，闭式 $b=b_\infty+A u^q$ 的拟合 rms
     <= 2 倍 $t=0.002$ 的 rms（0.016 / 0.047 / 0.037，即上限 0.032 / 0.094 / 0.074）。
     T1 不过 => 测到的是垃圾，不许外推。
  T2 $b_\infty$ 的 $t$ 不变性：$|b_\infty(t')-b_\infty(0.002)|\le 2\,{\rm SE}_\Delta$。
     ${\rm SE}_\Delta$ 由**配对** bootstrap 给出：同一批 20 张图的重采样同时用于两个 $t$
     的整套拟合（多项式 $\to$ 导数 $\to$ 闭式），两次拟合的相关性因此被正确处理。
  T3 $q$ 的 $t$ 不变性：$|q(t')-q(0.002)|\le 2\,{\rm SE}_\Delta$。
  T4 无模型因式分解检验（最直接）：定义逐图配对差
     $D_{t'}(u)=\log\Lambda_{t'}(u)-\log\Lambda_{0.002}(u)$，对 20 张图取均值。
     若 $b(u)$ 与 $t$ 无关，则 $\Lambda=C(t)\,u^{b(u)}$，$D(u)$ 必须是**与 $u$ 无关的
     常数**。判据：$D$ 对 $\log u$ 的回归斜率 $\le 2\,{\rm SE}$（同样配对 bootstrap）。
  T5 方向性（附带，**非判决**）：若 T2/T3 失败，检查符号是否符合上面的高斯收敛预言
     （$\Delta b_\infty>0$、$\Delta q<0$）。跑前记录：该预言的量级（$\sim0.05$ dex）
     与噪声同量级，**符号不符不能作为反证**，只作描述。
  T6 必须同时报告可探测下限：每个 $t$ 上报 $2{\rm SE}_\Delta(b_\infty)$ 与 $2{\rm SE}_\Delta(q)$。
     "通过"若不附这两个数就不可报告——否则等于把"测不到"当成"没有"。

若 T1–T4 全通过 => $b(u)$ 是概率流 ODE 的普定量（至少跨 25 倍的 $\Lambda_{\rm ng}$ 变化），
这比"在 3 个族上拟合得好"强得多，可以进论文。
若 T2/T3/T4 失败 => 拿到 $b(t)$ 的第一条曲线；若同时 T5 符号符合高斯收敛预言，
就能反推先验谱的形状（这是原本想要的 payoff）。
若 T1 失败 => 定律在新 $t$ 不成立，记录失败，不外推。

用法（第八组；注意 --targets 传的是 kunif = 1-u，不是 u）：
      venv2/bin/python sweep_law_test.py --nimg 20 --t 0.05 \
              --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86 \
              --families W2 W8 TK50 --out sweep_bt005.json

====================  第八组修正案：T5b 定量高斯收敛预言（2026-10-18 22:02，跑前写定）======
**披露**：t=0.05 那一轮已跑到 10/60，进度日志里能看到每个算子的原始 Λ 与 Λ_Gauss
（但看不到任何 b(u)、b_inf、q、D(u)——判据 evaluated 的量一个都还没算）。本条修正
在**分析任何新 t 的斜率之前**写定。

T5 只查符号，太弱。这里给出同一机理的**定量**形式，全部输入在跑前就已知：
  * 高斯分量闭式：Λ_Gauss(t,u) = Σ_f a_f |X_f|² / (2(S_f+t²))，每个算子、每个 t
    都由 sweep 直接存下（`lam_gauss`）。
  * 高斯的指数闭式且与 t 无关：b_Gauss(u) = -u/(1-u)（推导见第八组正文）。
  * 两分量分解：Λ(t,u) = Λ_Gauss(t,u) + Λ_ng(t,u)，且**非高斯分量的指数 b_ng(u)
    与 t 无关**（这是待检验的假设，不是已知事实）。
  * b_ng(u) 由 t=0.002 基线自校准（因为该处 Λ_ng/Λ_G ≈ 74，b_G 的权重只有 1.3%）：
        b_ng(u) = [Λ_0(u) b_0(u) - Λ_G,0(u) b_Gauss(u)] / [Λ_0(u) - Λ_G,0(u)]
  * 于是新 t 处的斜率被**完全预言**（只用已测的 Λ 电平，不用任何新拟合）：
        b_pred(t,u) = [Λ_Gauss(t,u) b_Gauss(u)
                       + (Λ(t,u) - Λ_Gauss(t,u)) b_ng(u)] / Λ(t,u)

**T5b 判据**：每个 t、每个族，19 个 u 上 |b_测(t,u) - b_pred(t,u)| 的 rms <= 0.10，
且均值偏差 <= 2*SE_bootstrap。
  （容差 0.10 的来源：t=0.002 基线上 b 的 bootstrap SE 是 0.017–0.311，取其中端；
   判据本身是"两分量分解能否解释全部 t 依赖性"，比 T2/T3/T4 的零假设强。）

若 T5b 通过 => "Λ = 高斯分量 + 非高斯分量，后者指数与 t 无关"解释了全部 t 依赖性，
  且 b(u) 的 t 漂移可纯由两个分量的权重变化算出——这是可写进论文的机理陈述。
若 T5b 失败 => 非高斯分量自身的指数随 t 变，b(t) 不能由两分量混合解释，记录失败。

====================  第九组：b(u) 是否随**先验**（模型）变？（2026-10-19，跑前写定）======
第八组证明 $b(u)$ 随**噪声级** $t$ 变（$b_\infty$ 从 $-0.705$ 升到 $-0.402$）。本组问另一半：
$b(u)$ 是先验谱的形状读数，还是概率流 ODE 框架的普适量？

设计：2×2 因子表，但只缺一格（ffhq+ffhq 即第八组基线，已有）。

  |            | ffhq 图池 | afhqv2 图池 |
  |------------|-----------|-------------|
  | ffhq 先验  | **已有**（基线） | **格子 C：新** |
  | afhqv2 先验| （省略）   | **格子 B：新** |

**格子 B − 格子 C = 固定图像下的先验效应**（同一批 20 张 afhqv2 图，只换先验），这是判决量。
格子 C − 基线 = 固定先验下的图像效应，用作对照：如果先验效应并不比"随便换个扰动"更大，
那"$b(u)$ 依赖先验"就是句空话。

每格：3 族（W2/W8/TK50）× 19 个 $u$ 点（17 个 target + $\beta$=0.3/0.7 + raw）× 20 图 = 60 算子，
$t$=0.002，kunif_ref=0.92，提取方式与第七/八组逐字对齐。

**为什么预期"能测出差异"（这是物理预期，不是推导）**：$b(u)=\mathrm{d}\log\Lambda/\mathrm{d}\log u$
是 $p_t$ 的性质，而 $p_t$ 由先验和 $t$ 共同决定。第八组已经证明它对 $t$ 敏感；若它对先验完全不敏感，
那需要一个特殊的相消机制。所以主判据的预期是"测得出"——但这个预期的**方向**我给不出来，
因为没有一个闭式能告我 afhqv2 的谱形状会把 $b_\infty$ 往哪推。跑前明确记录：**不预官方向**。

**M6 是一条能推导的定量判据（附带，且是有条件的）**：固定图像时 $\Lambda_{\rm Gauss}$ 完全相同
（它只用图池谱 $S_f+t^2$，与先验无关），所以若 $b_{\rm ng}$ 与先验无关，则
$$\Delta b(u)=(b_{\rm Gauss}(u)-b_{\rm ng}(u))\cdot\Lambda_{\rm Gauss}\cdot\Big(\frac{1}{\Lambda_{\rm afq}}-\frac{1}{\Lambda_{\rm ffq}}\Big)$$
而 $b_{\rm Gauss}-b_{\rm ng}$ 在 $u$=0.14 处为 **+0.52**、在 $u$=0.85 处为 **−1.39**（符号相反）——
所以该模型预言 **$\Delta b(u)$ 的符号在低 $u$ 与高 $u$ 相反**。这是一个可判决的特征。
跑前记录：这条预言的**条件本身正是待检验的假设**（$b_{\rm ng}$ 与先验无关），
第八组已否掉它在 $t$ 上的版本；它在这里的价值是把"$b_{\rm ng}$ 与先验无关"变成一句可判决的话。

**预言（在看到任何新数据之前写定）**：

  M1 守门：两格的闭式 rms <= 2 倍基线（0.032/0.094/0.074）。不过 => 定律在换先验后不成立，
     记录失败，不外推。（afhqv2 先验判 ffhq 图、或反之，都可能是"离流形"输入。）
  M2 主判据（二元）：固定图像下 $|b_\infty^{\rm afq\,prior}-b_\infty^{\rm ffq\,prior}|$ 与
     配对 bootstrap 的 $2\,$SE 比。预期：测得出（理由见上，物理预期非推导）。
  M3 同上，对 $q$。
  M4 无模型：$D_{\rm prior}(u)=\log\Lambda_{\rm afq\,prior}(u)-\log\Lambda_{\rm ffq\,prior}(u)$
     对 $\log u$ 的斜率 <= $2\,$SE（零假设：先验无关）。
  M5 图像对照：同时报格子 C − 基线的 $b_\infty$ 变化。若先验效应与图像效应同量级或更小，
     必须在结论里写明"$b(u)$ 对一切扰动都敏感"，不能只说"依赖先验"。
  M6 符号翻转（附带）：检查 $\Delta b(u)$ 是否低 $u$ 与高 $u$ 反号，以及与上式给的量级比。
  M7 可探测下限：每格报 $2\,$SE$(b_\infty)$、$2\,$SE$(q)$。通过而不附此数不可报告。

若 M1–M4 全过且 M2/M4 测得出 => $b(u,t)$ 读的是先验谱，不是 ODE 框架——这是第八组想要的 payoff。
若 M2/M4 测不出 => 在所测精度内 $b(u)$ 与先验无关（附 M7 的下限），则 $b(u)$ 的 $t$ 依赖性
  来自 ODE 积分本身而非先验谱，这同样是有信息量的结论。
若 M1 不过 => 换先验后定律不成立，记录失败。

用法（第九组；注意 afhqv2 池已由 2026-10-11 生成在 /tmp/pool_afhqv2-64x64-vp_256_s0.pt，
      256 张、与 ffhq 池同规格；径向谱确实不同：afhqv2 低频更弱、最外环高 29%）：
      venv2/bin/python make_pool.py --ckpt edm-afhqv2-64x64-uncond-vp.pkl \
              --out /tmp/pool_afhqv2_256_s0.pt
      venv2/bin/python sweep_law_test.py --nimg 20 --t 0.002 \
              --ckpt edm-afhqv2-64x64-uncond-vp.pkl \
              --pool /tmp/pool_afhqv2-64x64-vp_256_s0.pt \
              --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86 \
              --families W2 W8 TK50 --out sweep_m_afq_afq.json
      venv2/bin/python sweep_law_test.py --nimg 20 --t 0.002 \
              --ckpt edm-ffhq-64x64-uncond-vp.pkl \
              --pool /tmp/pool_afhqv2-64x64-vp_256_s0.pt \
              --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86 \
              --families W2 W8 TK50 --out sweep_m_ffq_afq.json

====================  第十组：ρ(t) 的饱和区（2026-10-20，跑前写定）==========================

背景（§3.9(x)）：ρ(t,u)=Λ(t,u)/Λ_0(u)≈1-14t 在 t≤0.02 近似线性；(1-ρ̄)/t =
11.2/14.7/16.6/13.9（t=0.005/0.01/0.02/0.05）在 t=0.05 已回落；饱和型 A·t/(B+t)
拟出 A=4.23>1（无物理意义，因 1-ρ≤1）。**转折点在 t∈[0.02,0.2] 但没数据。**

本组补 t=0.08/0.12/0.16 三点。族/u 网格/图数/提取方式与第八组逐字对齐（W2 W8 TK50、
17 个 kunif 目标、20 图、logΛ 对 logu 的 5 次多项式取解析导数）。

**预官（写预官前已从已有数据算出的参照值，全部列在这里）**：
  公共网格 20 点 u∈[0.08,0.85]；Λ_0(u) 三族几何均值：Λ_0(0.85)=859、Λ_0(0.14)=10406、
  Λ_0(0.08)=21856。Λ_Gauss = cap·(1-u) 已验证到 0.3%（u=0.85: 161.7 vs 162.3；
  u=0.14: 927.4 vs 930.3）。b_0(u→0) = -0.705（§3.9(s) 闭式）。

S1 减速（**可能失败**）：1-ρ̄ 随 t 单调增，且 (1-ρ̄)/t 在 t≥0.02 严格下降
   （实测 16.6→13.9）。**预言三个新点的 (1-ρ̄)/t 全部 < 13.9。**
   另报 7 点整体能否被 A·t/(B+t) 且 A≤1 拟合——4 个旧点线性化后截距为负（A<0，
   无物理意义），所以这条**正是新数据要回答的**：不预设通过。

S2 u 散度有可计算的上界（**最尖的一条**）：高斯极限下
       ρ_G(0.85)/ρ_G(0.14) = [cap·0.15/Λ_0(0.85)] / [cap·0.86/Λ_0(0.14)] = 2.113
   （cap 约掉，与 t 无关）。实测该比从 1.058（t=0.005）张到 1.895（t=0.05）。
   **预言：它继续单调增长，且在 t=0.16 处落入 [1.895, 2.113]**——即逼近但不超过
   高斯极限值。若超过 2.113，说明 Λ 还没走向高斯极限，"上界"论断失败。
   **这条若通过就是可写进论文的强陈述**：b(u) 的 t 依赖性有上限。

S3 Δb_∞ 的上界（**可能失败**）：高斯极限下 b_Gauss(u)=-u/(1-u)→0（u→0），
   而 b_0(u→0)=-0.705，故 Δb_∞→+0.705。实测 0.047/0.113/0.210/0.316。
   **预言：Δb_∞ 单调增、在 t=0.16 处落入 [0.316, 0.705]，且增量减速**
   （Δb_∞/t 从 t=0.01 的 11.3 降到 t=0.05 的 6.32，预言 t=0.16 处 <6.32）。

S4 放大倍数穿 1（**可能失败**）：§3.2 的放大倍数 74.9/10.7/4.6/0.39
   （t=0.002/0.05/0.10/0.20）在 t∈(0.10,0.20) 穿过 1。本组用 W2/W8/TK50，
   **预言三个新 t 上放大倍数继续下降，并在 t=0.16 处 < 5**（不给穿 1 的硬预言，
   因族不同）。

S5 信噪比守门（**必须做**）：逐格报 SNR=Λ̄·√n/s。§3.9(x) 已见 t=0.05 时 Λ 降到
   317-2573 而单图噪声 ~200。**预言 t=0.16 处部分格 SNR<3**；这些格从 S1/S2/S3 中
   剔除并**明确计数**，不静默丢弃。

S6 拟合机器守门：闭式 b=b_∞+A·u^q 的拟合 rms 在第八组四个新 t 上 ≤0.044。
   **预言三个新 t 上 rms ≤ 0.10**（b 本身跨 ~4）。不过则先查拟合，不报物理。

用法（第十组；--targets 传 kunif=1-u）：
      venv2/bin/python sweep_law_test.py --nimg 20 --t 0.08 \
              --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86 \
              --families W2 W8 TK50 --out sweep_bs008.json
      （0.12 / 0.16 同理，out 分别 sweep_bs012.json / sweep_bs016.json）

用法： venv2/bin/python sweep_law_test.py --smoke
      venv2/bin/python sweep_law_test.py --nimg 20
      venv2/bin/python sweep_law_test.py --nimg 20 --families W2 W8 W16 W32 B8 \
              --out sweep_steepness.json
      venv2/bin/python sweep_law_test.py --nimg 20 --kunif-ref 0.60 \
              --targets 0.35 0.50 --families RP35 RP75 LG35 TS2 EF35 \
              --out sweep_b2test.json
      venv2/bin/python sweep_law_test.py --nimg 20 --families TK30 TK50 EF60 W3 LG60 TS3 \
              --out sweep_twlaw.json
      venv2/bin/python sweep_law_test.py --nimg 20 \
              --families W6 TK80 RQ400 TK95 W15 RQ200 EF100 LG150 --out sweep_twod.json
      venv2/bin/python sweep_law_test.py --nimg 20 \
              --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 \
              --families W2 W8 TK50 --out sweep_bdense.json
      venv2/bin/python sweep_law_test.py --nimg 20 --targets 0.30 0.35 \
              --families RQ400 EF100 --out sweep_bgap.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
POOL = "/tmp/pool_ffhq64_256_s0.pt"
TARGETS = [0.35, 0.60, 0.85]
BETAS = [0.3, 0.7]
KUNIF_REF = 0.92          # 每个新族的参考强度（第一/二组用 0.92，第三组用 0.60）


def halfplane():
    """(H, W//2+1) 的半径网格，归一到 [0,1]（用半平面自己的最大值）。"""
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
    return R / R.max()


def make_profile(kind, par, sc, r):
    """kind=剖面族；par=形状参数；sc=待定参数（尺度或位置，由 tune 二分）。"""
    if kind == "weibull":
        return np.exp(-(r / sc) ** par)
    if kind == "butterworth":
        return 1.0 / (1.0 + (r / sc) ** (2 * par))
    if kind == "twoscale":
        # 两个高斯的等权混合，第二尺度 = sc*par（par = 尺度比）
        return 0.5 * np.exp(-(r / sc) ** 2) + 0.5 * np.exp(-(r / (sc * par)) ** 2)
    if kind == "ramp":
        # 幂律斜坡：衰减 a = (r/sc)^par，sc 调到命中 knif
        return np.clip((r / sc) ** par, 0.0, 1.0)
    if kind == "logistic":
        # logistic 边：sc 是翻转点位置 r0，par 是宽度
        return 1.0 / (1.0 + np.exp(-(r - sc) / par))
    if kind == "erf":
        from math import erf
        return 0.5 * (1.0 + np.vectorize(erf)((r - sc) / par))
    if kind == "tukey":
        # 升余弦（Tukey）边：平通带 + cos 过渡 + 平阻带。真实滤波器的 canonical 形状。
        # par = 过渡带占 [0, sc] 的比例；sc = g2 归零的半径。
        r1 = sc * (1.0 - par)
        g = np.zeros_like(r)
        g[r <= r1] = 1.0
        m = (r > r1) & (r < sc)
        t = (r[m] - r1) / max(sc - r1, 1e-12)
        g[m] = 0.5 * (1.0 + np.cos(np.pi * t))
        return g
    raise ValueError(kind)


def kunif_of(kind, par, sc, r):
    return float(1.0 - make_profile(kind, par, sc, r).mean())


# kunif 随可调参数 sc 的单调方向：
#   weibull / butterworth / twoscale：sc 是尺度，越大越不衰减 => kunif 下降
#   ramp / logistic / erf：sc 越大衰减越弱 => kunif 上升
INC = {"ramp", "logistic", "erf"}


def tune_rc(kind, par, r, target):
    """二分可调参数 sc 使 knif_ref = 1-<g2> 命中 target。"""
    def kunif(sc):
        return float(1.0 - make_profile(kind, par, sc, r).mean())
    inc = kind in INC
    lo, hi = 1e-4, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        k = kunif(mid)
        too_big = (k > target) if inc else (k < target)
        if too_big:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# 新族定义：(id, kind, par)
NEW_FAMILIES = [
    ("W1", "weibull", 1.0),
    ("W2", "weibull", 2.0),
    ("W4", "weibull", 4.0),
    ("B1", "butterworth", 1.0),
    ("M2", "twoscale", 5.0),      # par = 尺度比 c2/c1
    # ---- 第二组：陡峭度扫描 ----
    ("W8", "weibull", 8.0),
    ("W16", "weibull", 16.0),
    ("W32", "weibull", 32.0),
    ("B8", "butterworth", 8.0),
    # ---- 第三组：B2 预言的样本外检验（四个全新形状类）----
    ("RP35", "ramp", 0.35),
    ("RP75", "ramp", 0.75),
    ("LG35", "logistic", 0.35),
    ("TS2", "twoscale", 2.0),
    ("EF35", "erf", 0.35),
    # ---- 第四组：tw_lev 定律的预官检验（见 docstring 第四组，2026-10-18 跑前写定）----
    ("TK30", "tukey", 0.30),
    ("TK50", "tukey", 0.50),
    ("EF60", "erf", 0.60),
    ("W3", "weibull", 3.0),
    ("LG60", "logistic", 0.60),
    ("TS3", "twoscale", 3.0),
    # ---- 第五组：twod 定律的预官检验（见 docstring 第五组，2026-10-18 跑前写定）----
    ("W6", "weibull", 6.0),
    ("TK80", "tukey", 0.80),
    ("RQ400", "ramp", 4.0),
    ("TK95", "tukey", 0.95),
    ("W15", "weibull", 1.5),
    ("RQ200", "ramp", 2.0),
    ("EF100", "erf", 1.0),
    ("LG150", "logistic", 1.5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--kunif-ref", type=float, default=None,
                    help="参考强度；默认 KUNIF_REF=0.92。B2 检验组用 0.60")
    ap.add_argument("--targets", nargs="*", type=float, default=None,
                    help="A 组目标 kunif；默认 TARGETS。必须 <= kunif-ref")
    ap.add_argument("--out", default="sweep_law_test.json")
    ap.add_argument("--uniform-c", type=float, nargs="*", default=[],
                    help="额外加均匀传输算子 g2=c（kunif=1-c）；用于测底座平台的极限")
    ap.add_argument("--ckpt", default=CKPT,
                    help="换先验模型（第九组用；默认 ffhq）")
    ap.add_argument("--pool", default=POOL,
                    help="图池 .pt 文件（第九组用；默认 ffhq 池）。"
                         "换先验但想固定图像时，只改 --ckpt 不改 --pool")
    args = ap.parse_args()
    # --smoke 只跑 2 张图，绝不能覆盖正式数据（这个坑踩过一次：smoke 把
    # sweep_law_test.json 写成了 2 行，只好从本地备份恢复）。
    if args.smoke and args.out == "sweep_law_test.json":
        args.out = "smoke_law_test.json"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    nimg = 2 if args.smoke else args.nimg
    fams = NEW_FAMILIES
    if args.families:
        fams = [f for f in NEW_FAMILIES if f[0] in args.families]
    elif args.uniform_c:
        fams = []          # 只跑均匀算子，不跑剖面族
    if args.smoke:
        fams = fams[:1]
    kref = args.kunif_ref if args.kunif_ref else KUNIF_REF
    targets = args.targets if args.targets else TARGETS
    bad = [t for t in targets if t > kref + 1e-9]
    if bad:
        print(f"[警告] 目标 {bad} 超过 knif_ref={kref}，需要 beta>1，已剔除")
        targets = [t for t in targets if t <= kref + 1e-9]

    r = halfplane()
    ops_u = []
    # 定标：每个族的 rc 调到 kunif_ref = KUNIF_REF
    tuned = []
    for fid, kind, par in fams:
        sc = tune_rc(kind, par, r, kref)
        g2 = make_profile(kind, par, sc, r)
        tuned.append((fid, kind, par, sc, g2, float(1.0 - g2.mean())))
        print(f"[tune] {fid}: kind={kind} par={par} scale={sc:.4f} "
              f"kunif_ref={1.0-g2.mean():.4f}", flush=True)

    # ---- 可选的「均匀传输」算子：g2 = c 常数，a = 1-c 常数 ----
    # 用途：底座剖面平台（C ~ 1.070，twod 跨 30x 而 C 只变 1.05x）的物理读法是
    # 「传输被限制在窄带内 => 模糊偏好退化为均匀振幅缩放的偏好」。这里直接测它。
    # 均匀算子的 kunif = 1-c 由 c 决定，不能独立选 target，所以单独构造。
    for c in args.uniform_c:
        g2c = np.full_like(r, float(c))
        ops_u.append((f"UNI{c:g}", g2c,
                      dict(fid=f"UNI{c:g}", kind="uniform", par=float(c),
                           scale=float("nan"), K=float(1.0 - c), beta=1.0,
                           kunif_ref=float(1.0 - c))))
        print(f"[uniform] g2 = {c:g}  =>  kunif = {1.0-c:.4f}  (u = {c:.4f})",
              flush=True)

    # 算子表：每个族 3 个 A 目标 + 2 个 C beta + 1 个 V(beta=1)
    ops = []
    for fid, kind, par, sc, g2, kref in tuned:
        for K in targets:
            beta = K / kref
            ops.append((f"{fid}@K{K:g}", (1.0 - beta) + beta * g2,
                        dict(fid=fid, kind=kind, par=float(par),
                             scale=float(sc), K=float(K), beta=float(beta),
                             kunif_ref=kref)))
        for beta in BETAS:
            ops.append((f"{fid}@b{beta:g}", (1.0 - beta) + beta * g2,
                        dict(fid=fid, kind=kind, par=float(par),
                             scale=float(sc), beta=float(beta),
                             kunif_ref=kref)))
        ops.append((f"{fid}@raw", g2.copy(),
                    dict(fid=fid, kind=kind, par=float(par),
                         scale=float(sc), beta=1.0, kunif_ref=kref)))
    ops = ops + ops_u
    print(f"[cfg] nimg={nimg} t={args.t:g} knif_ref={kref} targets={targets} "
          f"{len(fams)} 族 => {len(ops)} 个算子", flush=True)

    model = load_local_model(args.ckpt).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(args.pool, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    with torch.no_grad():
        P = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            / (H * W)
        Sfull = P.mean(1).mean(0)
        Zx = (torch.fft.rfft2(xs.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            .mean(1, keepdim=True).mean(0)[0] / (H * W)
    Sf = Sfull + args.t ** 2
    wg = Zx / (2.0 * Sf)
    cap = float((Zx / (2.0 * Sf)).sum())

    B = 4

    @torch.no_grad()
    def logp(z):
        out = []
        for i in range(0, z.shape[0], B):
            nll = likelihood_fn(z[i:i + B].to(DT), ode_flow,
                                t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    t0 = time.time()
    lp_sharp = logp(xs)
    print(f"[sharp] t={args.t:g} mean log p = {lp_sharp.mean():.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for i, (oid, g2, meta) in enumerate(ops):
        g = torch.from_numpy(g2).to(DEV)[None, None].to(torch.float64)
        with torch.no_grad():
            y = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                 dim=(-2, -1)) * g.sqrt(),
                                 s=(H, W), dim=(-2, -1)).to(DT)
        tt = time.time()
        lp = logp(y)
        lam = lp - lp_sharp
        lam_g = float(((1.0 - g2) * wg.cpu().numpy()).sum())
        kunif = float(1.0 - g2.mean())
        rows.append(dict(id=oid, meta=meta, kunif=kunif,
                         lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                         lam_per_img=[float(v) for v in lam],
                         lam_gauss=lam_g,
                         logp_blur_mean=float(lp.mean()),
                         logp_sharp_mean=float(lp_sharp.mean())))
        snr = lam.mean() * math.sqrt(nimg) / max(lam.std(), 1e-9)
        print(f"  [{i+1}/{len(ops)}] {oid:<14} knif={kunif:.4f} "
              f"Lambda={lam.mean():>9.0f}+-{lam.std()/math.sqrt(nimg):>6.0f} "
              f"SNR={snr:>5.1f}  L_G={lam_g:>7.1f}  A={lam.mean()/lam_g:>6.2f} "
              f"({time.time()-tt:.0f}s)", flush=True)
        del y, g
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg=nimg, t=args.t, ckpt=args.ckpt, pool=args.pool,
                            kunif_ref_target=kref, targets=targets, cap=cap,
                            families=[dict(id=f[0], kind=f[1], par=float(f[2]))
                                      for f in fams]),
                    rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
