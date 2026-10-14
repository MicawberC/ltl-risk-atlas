# 联合风险价格图谱：可行性与创新性实验报告

日期：2026-10-12。状态：可重复的研究原型与合成模型压力测试；尚非完整的 CARLA 投稿实验。

## 问题与主张

给定固定的 LTL 产品 MDP 转移核 P、奖励 r、折扣因子 γ 和起始产品状态 α，风险严重度随参数 θ 改变：

    cθ = c0 + Σj θj hj.

优化目标保留原作者的软风险罚与硬风险预算：

    maxπ Rπ − max(0, Cθ,π − S)   subject to Cθ,π ≤ H.

提出的研究主张是：对一系列风险图更新，不必每次重建完整 LP 或重新搜索风险价格；可以缓存确定策略、占用度量和联合“风险价格—风险图参数”区域，通过原始可行策略和对偶上界决定何时复用、何时扩充。当前实验中的 H=5、S=0.8、γ=0.9，与作者公开的最小示例代码一致；论文现行版本的 CARLA 实验默认值是硬阈值 2、软阈值 1、折扣因子 0.8。两套数值结果不能直接当作同一实验比较。这是期望折扣风险约束，不是每条驾驶轨迹的安全保证。

**适用范围必须明确：**当前主实验固定 P、r、产品自动机，允许局部风险严重度与起始状态变化。原作者公开的 reactive_LTL_main.py 会在抽象状态变化时重建产品 MDP；本方法尚未证明能复用这种改变 P 或标签的模型。大地图都是我们的合成产品模型。4×4 的 P、c、r 和起点已由 check.py 与作者代码逐项核对。

## 数学核心

对风险价格 t≥0，标量化奖励为 r − t cθ。引入联合坐标

    z = (t, tθ1, …, tθd).

对固定确定策略 π，

    vπ(z) = (I−γPπ)⁻¹ (rπ − z0 c0,π − Σj zj hj,π)

是 z 的仿射函数。因此每个 Bellman 动作不等式也是 z 的线性不等式，π 保持标量化最优的区域是一个多面体。在线参数 θ 对应射线 z=t(1,θ1,…,θd)。缓存策略区域与射线的交给出无需再解 MDP 的有效价格区间。

对任意有效 t，构造原问题对偶上界

    U(t,θ)=Vt(θ)+S min(1,t)+H max(0,t−1).

推导是把软罚写成 `max(0,C−S)=max_{0≤u≤1}u(C−S)`，再给硬约束引入 λ≥0。令 t=u+λ；由于本实验 H>S，固定 t 时最小常数项 `uS+λH` 取 `u=min(1,t)`，得上述上界。若某策略在价格 t 的 Bellman 不等式最小余量不低于 −δ，则其标量化价值与最优值差至多 δ/(1−γ)，故对偶上界仍有效。这一推导同时解释为什么只缓存策略目标值而不核查 Bellman 区域不够。

实际实现允许 Bellman 残差 δ≤10⁻⁸，并把 δ/(1−γ) 加入上界。缓存策略的折扣占用度量在 P、α 固定时不变；其任意凸组合仍满足流守恒。枚举单策略以及两策略在 C=S 或 C=H 处的混合，得到原始可行下界 L。只有 C≤H 且 U−L≤max(10⁻⁴,0.01|L|)+数值容差时才输出“认证”。否则先在邻近缓存风险价格处调用**一次无约束标量化 MDP 求解**，加入新策略与区域，重新测试证书；若仍失败才完整求解当前 CMDP。对起点变化，使用缓存的稀疏 LU 因子重新计算各策略占用度量，再执行同样检查。区域检索先测试参数最近的 8 个策略；证书失败时继续扫描其余策略。

**候选主方法**是联合风险价格图谱、软硬活跃集的原始—对偶认证，以及证书指导的单次策略求解扩充。单独的多面体、SCC 分解、CMDP 敏感性、占用度量混合、证书均不是新发现；只替换原 LP 求解器也不是本工作的创新主张。

## 文献对比与风险

| 工作 | 与本研究的关系 |
| --- | --- |
| Qi 等，IROS 2025，Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications | 给出风险 LTL 产品模型和占用度量 LP；我们的模型和问题起点 |
| Altman 与 Shwartz，1991，Sensitivity of constrained Markov decision processes | 已研究 CMDP 对参数变化的敏感性；不能声称首创灵敏度分析 |
| Shin 与 Yang，IEEE L-CSS 2024，Multiparametric Analysis of Multi-Task Markov Decision Processes | 已给出奖励参数变化时保持同一最优 MDP 策略的多面体区域；不能声称首创策略区域 |
| Stradi 等，ICML 2024，Online Learning in CMDPs | 研究未知转移与随机/对抗约束下的长期在线学习；我们的已知固定 P、重复求解风险图更新问题与其目标不同 |
| Junges 等，2024，Parameter synthesis for Markov models | 已系统研究参数 MDP 区域覆盖 |
| Baier 等，2025，Certificates and Witnesses for Multi-objective ω-regular Queries in MDPs | 已研究时序/多目标 MDP 可检验证书 |
| Li 与 Fu，2019，Topological Approximate Dynamic Programming under Temporal Logic Constraints | 已用任务自动机拓扑求解 |

相较 Shin 与 Yang 的*参数化奖励无约束 MDP*，这里风险代价同时改变标量化奖励和 CMDP 约束，最优平稳随机策略可能在软阈值 S、硬阈值 H 之间切换。相较静态证书论文，这里将可检验的误差界用于多次查询的复用与扩充。**目前尚未对所有相关文献证明不存在同构算法，也不能声称“首次”。**

文献链接：

- https://research.tue.nl/en/publications/risk-aware-autonomous-driving-with-linear-temporal-logic-specific/
- https://link.springer.com/article/10.1007/BF02204825
- https://doi.org/10.1109/LCSYS.2024.3405958
- https://proceedings.mlr.press/v235/stradi24a.html
- https://link.springer.com/article/10.1007/s10703-023-00442-x
- https://arxiv.org/abs/2508.17859
- https://arxiv.org/abs/1907.10510

## 主实验：25 次风险更新

两处风险区域各取 −5%、−2%、0、+2%、+5%，组成 5×5 参数网格；每组随机打乱查询顺序三次。所有方法共享 P、r；计时均不含产品模型建造。图谱计入名义点准备、失败查询、完整求解扩充及新策略区域建造。所有图谱返回策略均用新模型独立 evaluate_policy 验证风险/目标，并与每个点的从零求解界交叉检查。另有每组 3 个点用独立稀疏 LP 核对。

原论文报告的 720–2,880 状态、单次 LP 约 15–60 ms，来自其 CARLA 场景和实验平台；下面是我们的合成产品模型上多次风险更新的累计时间。模型、参数、硬件、计时范围不同，不能把两个数字直接相除称为相对原论文的速度提升。

| 合成产品模型 | 状态数 | 25 次图谱命中 | 图谱总时间（三序列范围） | 25 次从零分层求解 | HiGHS 保基热重优化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24×24 硬预算活跃 | 4,609 | 22/25 | 3.00–3.17 s | 13.78 s | 9.81 s |
| 32×32 硬预算活跃 | 8,193 | 21/25 | 8.60–10.84 s | 35.47 s | 47.77 s |
| 24×24 中部目标 | 4,609 | 24/25 | 0.74–0.80 s | 6.99 s | 20.54 s |
| 32×32 中部目标 | 8,193 | 25/25 | 1.04–1.11 s | 17.98 s | 38.24 s |

“保基热重优化”使用 highspy 1.15.1，同一 LP 保留流守恒矩阵及 simplex basis，用 3 个风险总量辅助变量使每次更新仅改 4 个约束系数。各点 LP 目标值落在从零求解的上下界之间。它是比每次重建 LP 更强的基线。32×32 硬预算的一次 HiGHS IPM 独立核查失败，改用 dual simplex 成功，详情在数据中；不把失败耗时计入图谱提速。

硬预算模型的 25 点中，15 点的策略风险为 H=5，10 点为 S=0.8，确实发生软/硬活跃集切换。四组全部返回策略保持硬预算、通过约 1% 原始—对偶误差证书。24×24/32×32 硬预算地图均为人为设计压力测试，两个尺寸存在相似的近起点决策结构。

## 反例与增强实验

**小模型反例。**原作者 4×4 产品模型为 129 状态。在 25 次更新上图谱约 0.050–0.053 s，而 HiGHS 热重优化约 0.027 s；小模型用图谱并不更快。一个非名义风险点初始证书失败，扩充后目标与独立 LP 同为 1.39708301937。

**同一公开示例的直接比较。**另外直接调用作者公开的 `Risk_LTL_LP.solve`，使用其原始 128 状态产品转移、初始状态、接受集和风险成本；我们的等价稀疏模型额外有一个零回报吸收状态，因此为 129 状态。对两处风险区域人为施加同样的 ±5% 更新，25 点随机顺序重复三次；作者原始 Gurobi LP 函数每次新建模型，累计 5.76–6.37 s；图谱包含准备和扩充为 0.062–0.072 s；保持 LP 矩阵与 simplex basis 的 HiGHS 为 0.029–0.037 s。75 个点原始 Gurobi 与保基 LP 目标一致，图谱目标处于独立 LP 目标与自身证书上界之间，硬预算均满足。**原始函数直接调用的巨大时间差主要反映 LP 构建/实现方式，不能视为图谱的独有算法收益；在这个小模型上强保基 LP 比图谱更快。**该风险更新序列由我们人为生成，不是论文 CARLA 场景数据。

**简单缓存消融。**只保存名义点的两条策略并混合，完全不计算新的对偶区域或扩充，在主实验 ±5% 的四组模型均有 25/25 点达到 1% 目标差。这说明小扰动的**策略质量**不需要新图谱；这里图谱的额外作用主要是证明。不能把小扰动下的目标优势归功于本方法。

**困难扰动。**24×24 中部目标两处风险分别取 −80%、−32%、0、+32%、+80%（所有风险成本仍非负）。精确 LP 下简单名义策略缓存只有 **12/25** 点达到 1% 差距，最大目标损失约 0.11054。旧版“证书失败即完整求解”在三种随机顺序中直接命中 18/25、17/25、17/25，累计提速 1.65–1.95 倍。加入单次策略求解扩充后，直接命中 16/25、16/25、17/25；额外分别有 7、6、6 点只需一次 MDP 求解，完整 CMDP 回退仅 2、3、2 次。计入准备、查询、扩充后分别耗时 2.10、3.96、2.77 s；从零分层求解 10.07 s、HiGHS 保基 LP 22.34 s，分别提速 4.80、2.55、3.64 倍。所有 75 次返回均逐点与精确 LP 核对目标与硬约束。

**五维风险图。**同一 24×24 中部目标模型有 5 个物理风险位置，各自独立变化 ±80%；固定随机样本 50 点、三种查询顺序。只用名义策略缓存有 34/50 点进入 1% 误差，最大损失约 0.09288。新版图谱分别直接认证 16/50、19/50、18/50 点；剩余点中一次 MDP 求解解决 34、30、32 点，仅一次顺序需要 1 次完整 CMDP 回退。图谱总时间 5.04、5.92、5.08 s；50 次从零分层求解为 25.51 s，累计提速 5.06、4.31、5.02 倍。所有返回均在新风险图上独立评估，并与从零求解的界核对；另对 5 个抽样点进行独立 LP 检查。

**强基线消融。**即使不先测试图谱，每次都用附近价格求解一个普通 MDP、再以同一证书与必要的完整回退保证质量，也能获得收益。二维大扰动三种顺序耗时 3.35、5.07、4.11 s；五维耗时 5.75、6.73、5.81 s。图谱的对应时间分别为 2.10、3.96、2.77 s 与 5.04、5.92、5.08 s；因此图谱相对这个强基线的增益小于相对从零 LP 的倍数，五维约 1.14 倍。该消融明确限制了“几倍提速完全来自新方法”的说法。

**五维保基 LP。**将同一 HiGHS 保基重优化基线扩展到 5 个风险系数，三种顺序的 50 次总时间为 31.20、31.28、33.56 s；每点保持有效 simplex basis，目标均位于从零求解的上下界之间。该基线在此案例比 25.51 s 的从零分层求解还慢，因此主要比较同时保留两者。

**更大局部风险图。**32×32 中部目标产品模型有 8,193 个状态，5 个局部风险点各自独立变化 ±80%，共 25 个随机更新。名义策略缓存只有 17/25 点达到 1% 目标差；图谱 23/25 点直接认证，2 点需完整 CMDP 回退，总时间 5.13 s；从零分层求解 20.83 s，HiGHS 保基 LP 88.52 s。每点风险和目标均经过独立策略评估、从零求解上下界交叉检查；另有 3 点由独立稀疏 LP 抽查。这是规模趋势的正面证据，但只有一种随机序列。

**硬预算大范围风险图反例。**另一张 24×24 地图含 48 个风险位置，分成 5 个空间风险区，严重度各自变化 ±80%，共 25 点。名义策略缓存的目标质量已有 23/25 点达到 1%，但直接可认证仅 3/25。一次 MDP 后立即回退的版本有 18 次完整 CMDP 回退，35.78 s，略慢于从零求解的 35.48 s。沿缓存 Bellman 临界价格继续试探，并用邻近策略热启动策略迭代后，完整回退降为 3 次，累计 88 次普通 MDP 求解，总时间 19.68 s，约为同期从零求解 34.52 s 的 1.75 倍提速；同一 25 点的 HiGHS 保基 LP 为 18.88 s，**仍比图谱快**。该结果否定“所有复杂风险图上稳定胜过保基 LP”的主张，也表明风险更新覆盖范围影响图谱收益。

临界价格连续追踪在局部五维地图上也不总划算：三种顺序的速度是从零求解的 2.09、4.40、4.80 倍，而单次策略扩充为 5.06、4.31、5.02 倍。应将扩充策略视为工作负载选择，而非统一改进。

**跨地图。**24×24 硬预算模型的另外 4 个布局种子均为 22/25 直接命中，累计提速 4.81–5.42 倍。硬预算风险变化扩大到 ±20% 时为 21/25、4.72 倍；中部目标同幅度变化为 20/25、3.24 倍。

**起始状态改变，仍固定 P。**5 个安全起点与 5 组风险更新组合为 25 次查询。四组模型的命中数依次为 23、23、24、24；最新重复的图谱总时间依次约 2.97、6.54、1.43、3.19 s；从零分层求解约 17.61、42.37、11.68、27.19 s。HiGHS 保基 LP 也改变流约束起点，四组约 9.59、43.96、20.75、75.86 s。各点热启动 LP 的目标值均落在图谱/从零求解的上下界内，风险约束满足。此测试不包含 P 或标签变化。

**顺序任务自动机。**手工构造的三状态目标进度自动机产生 6,913 状态产品模型；两处风险各 ±5% 时 25/25 直接认证，图谱 2.28 s、从零分层求解 39.62 s、保基 LP 21.40 s。该公式结构由我们手工构造，并非作者 CARLA 场景。

## 结论与投稿门槛

实验支持**固定转移且风险更新局部化的多次查询**这一研究问题：联合价格区域、原始—对偶证书和选择性普通 MDP 扩充在这些模型上满足原约束，并在二维、五维、大模型局部更新中获得累计收益。±80% 局部更新下简单缓存质量不再足够，而图谱仍可认证并逐步覆盖。风险大范围变化且硬预算活跃时，保基 LP 可胜过图谱；相对“每次一次普通 MDP”的增益也较小。该方法目前是**有明确适用域和反例的可行研究原型**，不能表述为通用求解器或普遍提速。

现阶段可以把它称为“可辩护的主创新原型”，但还不能声称已有完整的 IROS 级投稿证据。剩余门槛是：（1）大模型是合成地图，尚无 CARLA/真实风险更新序列；（2）原作者反应式代码重建 P 和产品状态，当前只处理固定 P 与起点变化；（3）数学结果须写成正式定理，并与 Shin–Yang 的 mp-LP 工作逐条区分；（4）需要更多独立地图、目标位置、高维参数和成对重复计时，特别是确认相对“每次一次普通 MDP”的增益。若未完成这些，不应宣称“解决原论文全部在线瓶颈”。

**局部更新后续实验（2026-10-12）。**新增两种中部布局家族、靠近交互区的起点、连续与独立风险序列、逐点保基 LP 对照及简单缓存消融。目标旁风险簇有两种布局太容易，预设的多布局门槛没有通过；随后不同拓扑的五格风险带在 24×24 和 32×32 均产生非平凡更新，示例参数下图谱累计 1.35–3.01 s、保基 LP 14.92–69.86 s。论文 CARLA 默认参数下，图谱 3.15–8.18 s、保基 LP 11.92–59.62 s；但 32×32 的图谱第 95 百分位仍达 1.1–1.4 s，尚非实时性证据。32×32 某独立序列的从零稀疏 LP 抽查超时，逐点保基 LP 与从零 CMDP 上下界均通过。完整门槛、反例与数据见 [LOCAL_UPDATE_GATE.md](LOCAL_UPDATE_GATE.md)。这些仍是合成产品模型结果；下一步是构建 CARLA 场景并采集真实风险更新流。

## 文件与复现

- 算法：joint_price_atlas.py。
- 新扩充方案：wide_middle_fast.py、wide_middle_fast.json；five_zone_near.py --fast、five_zone_fast.json。
- 主实验：joint_atlas_benchmark.py、joint_atlas_benchmark.json。
- 跨图/较大扰动：joint_atlas_stress.py、joint_atlas_stress.json、wide_middle_stress.json、wide_middle_orders.json。
- 起点变化：joint_atlas_initial.py、joint_atlas_initial.json。
- LP 热重优化：highs_reopt.py、highs_reopt.json、highs_reopt_initial.py、highs_reopt_initial.json、wide_middle_warm.py、wide_middle_warm.json。
- 五维验证：five_zone_stress.py、five_zone_stress.json、five_zone_reference.json、five_zone_lp_spotcheck.py、five_zone_lp_spotcheck.json、highs_five_zone.py、highs_five_zone.json、five_zone_adaptive.json。
- 大模型五维：five_zone_scale.py、five_zone_scale.json。
- 硬预算反例和增强：five_zone_hard.py、five_zone_hard_fast.json、five_zone_hard.json、highs_hard_five.py、highs_hard_five.json、hard_lp_spotcheck.py、hard_lp_spotcheck.json；其中 `five_zone_hard_fast.json` 是单次 MDP 扩充结果，`five_zone_hard.json` 是临界价格跟踪和策略迭代热启动结果。
- 强制单次 MDP 基线：forced_oracle_baseline.py、forced_oracle_baseline.json。
- 简单缓存消融：primal_cache_ablation.py、primal_cache_ablation.json。
- 顺序任务：sequence_stress.json、highs_sequence.json。
- 小模型反例：small_case.json。
- 作者公开模型直接对比：same_demo_benchmark.py、same_demo_benchmark.json。

环境：Windows，Python 3.11.14，NumPy 2.4.6，SciPy 1.17.1，highspy 1.15.1；硬件为 Intel Core i5-14450HX。所有计时为单机探索结果，不能外推为真实车辆反应时间。

在仓库根目录执行下列命令可重做核心实验；脚本会生成或覆盖同名 JSON，计时会随机器与运行负载改变：

```powershell
python -m pip install numpy scipy highspy==1.15.1
python reproduction/adaptive_risk_ltl/check.py
python reproduction/adaptive_risk_ltl/same_demo_benchmark.py
python reproduction/adaptive_risk_ltl/joint_atlas_benchmark.py
python reproduction/adaptive_risk_ltl/wide_middle_warm.py
python reproduction/adaptive_risk_ltl/wide_middle_fast.py
python reproduction/adaptive_risk_ltl/five_zone_stress.py
python reproduction/adaptive_risk_ltl/five_zone_near.py --fast
python reproduction/adaptive_risk_ltl/five_zone_lp_spotcheck.py
python reproduction/adaptive_risk_ltl/highs_five_zone.py
python reproduction/adaptive_risk_ltl/forced_oracle_baseline.py
python reproduction/adaptive_risk_ltl/five_zone_scale.py
python reproduction/adaptive_risk_ltl/five_zone_hard.py --fast
python reproduction/adaptive_risk_ltl/five_zone_hard.py
python reproduction/adaptive_risk_ltl/highs_hard_five.py
python reproduction/adaptive_risk_ltl/hard_lp_spotcheck.py
python reproduction/adaptive_risk_ltl/check_joint_atlas.py
```

其中 `wide_middle_fast.py` 依赖已有的 `wide_middle_stress.json` 与 `wide_middle_warm.json`；仓库已经保存这两份基准数据。`five_zone_near.py --fast` 可复用已保存的 50 点从零求解结果；如删除 `five_zone_reference.json`，会先重新计算。主实验和五维脚本运行可能需要数分钟。
