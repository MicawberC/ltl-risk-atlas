# LTL 风险规划的证书引导自适应状态抽象：复现实验

**新实验：**针对同一个完整乘积模型的分层标量化求解、价格断点、跨布局证书与重复计时，见 [STRATIFIED_REPORT.md](STRATIFIED_REPORT.md)。下文记录此前的自适应抽象实验，两组结果的计时与结论应分别阅读。

局部风险反复更新与进入 CARLA 场景构建前的配对实验，见 [LOCAL_UPDATE_GATE.md](LOCAL_UPDATE_GATE.md)；其算法背景见 [JOINT_ATLAS_REPORT.md](JOINT_ATLAS_REPORT.md)。

## 范围与出处

本目录是对 Qi 等人的 IROS 2025 论文《Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications》[公开简化代码](https://github.com/Miracle-qi/Risk_LTL_Planning)所做的**独立实验扩展**，不是作者提供的算法。原仓库公开的是不依赖 CARLA 的最小示例，未公开完整 CARLA 实验流水线。本实验中的 8×8 至 32×32 地图是合成扩展，不是论文的真实自动驾驶结果。

`check.py` 检查 4×4 情况与仓库内保存的原作者乘积 MDP：转移、成本、奖励、初始状态逐项一致，稀疏 LP 目标值为 `1.3620227941214877`。为将原示例的边界欠随机行表示为标准随机 MDP，增加了零奖励、零成本的吸收状态；原状态上的转移和目标保持一致。

## 研究问题与方法

目标是在**同一个完整乘积 MDP** 上，输出满足硬风险约束的具体策略及其目标值上下界，并尽量用较小的聚合 MDP 求解。记折扣占据分布为 `d_pi`，奖励为 `R_pi`，风险为 `C_pi`，软阈值 `b_s=0.8`、硬阈值 `b_h=5`。原始目标为

`max_pi J_pi = R_pi - max(0, C_pi-b_s)`，约束 `C_pi <= b_h`。

初始块仅合并位于同一空间网格、交通灯状态、目标自动机状态、安全自动机状态和命题标签的状态。聚合转移为块内均匀平均的 `C P B`。解聚合占据 LP 后，将其随机策略提升至完整模型，并在**完整模型**上解稀疏线性方程，得到确切的 `J_pi` 和 `C_pi`，其中 `J_pi` 是可行策略给出的下界。

聚合 LP 给出软、硬风险的非负对偶价格 `lambda_s, lambda_h`。在完整模型上解奖励 `r-(lambda_s+lambda_h)c` 的普通折扣 MDP，其最优值函数记为 `V_lambda`。Bellman 最大残差 `delta` 给出保守上界

`U = V_lambda(s0) + delta/(1-gamma) + b_s*lambda_s + b_h*lambda_h`。

这里 `lambda_s <= 1` 由软风险松弛变量的 LP 对偶条件保证，`gamma=0.9`。每轮检查 `C_pi <= 5+1e-7` 和 `U-J_pi <= max(1e-4, 0.01*abs(J_pi))+1e-7`；达到两项才标记为“已认证”。这是**目标值相对误差至多约 1% 的证书**，不是所有状态转移都被精确保留的证书，也不是路径级碰撞安全保证。

核心细化分数来自标量化策略遗憾的精确分解：

`V_lambda(s0) - (R_pi-lambda*C_pi) = sum_s d_pi(s) [max_a Q_lambda(s,a) - sum_a pi(a|s)Q_lambda(s,a)]`。

因此先细化占据多且存在较大行动价值差异的块。代码逐轮验证这个等式，允许数值误差 `1e-5`。如果硬风险超限，增加风险占据分数。这个分数不同于只看模型转移误差，也不同于只看状态访问量。其理论作用是针对**当前任务的策略目标**定位导致上下界差距的块；严格认证仍由完整模型评估和上界计算完成。

对照方法：`occupancy` 只按访问量细化；`model_error` 只按一步转移、奖励和成本聚合误差细化，是本实验的通用、任务无关抽象基线；`uniform` 同时细化所有可细化块；`sparse_lp` 在同一完整乘积模型上用 SciPy HiGHS 解占据 LP。所有策略的目标与风险都在完整模型上评估。`racing` 是一次失败的候选策略试验，不能作为最终改进。

## 结果

测试环境：Windows，Intel Core i5-14450HX，Python 3.11.14，NumPy 2.4.6，SciPy 1.17.1。时间只计求解，不计公共模型构建；重复计时在同一进程中预热 HiGHS，然后交错运行方法。下面的时间是重复实验中位数，4 个较小规模各重复 5 次，32×32 重复 3 次。不同进程的一次性时间不能直接与这些中位数混用。

| 合成场景 | 完整状态数 | 稀疏 LP 中位数 | 引导抽象中位数 | LP/引导 |
|---|---:|---:|---:|---:|
| 8×8 近目标 | 513 | 0.069 s | 0.461 s | 0.15× |
| 16×16 近目标 | 2049 | 1.332 s | 1.299 s | 1.03× |
| 20×20 近目标 | 3201 | 3.525 s | 2.048 s | 1.72× |
| 24×24 近目标 | 4609 | 7.227 s | 3.213 s | 2.25× |
| 20×20 硬风险压力 | 3201 | 2.318 s | 1.144 s | 2.03× |
| 24×24 硬风险压力 | 4609 | 4.902 s | 1.713 s | 2.86× |

32×32 重复计时见 `timing_large32.json`。完整模型有 8193 个状态：近目标场景为稀疏 LP 13.082 s、引导抽象 4.005 s，中位数提速 3.27×；硬风险压力场景为 11.447 s、2.912 s，提速 3.93×。

按相同的 1% 证书标准，24×24 近目标中，引导方法用 653 状态，任务无关模型误差基线用 1603 状态；32×32 近目标中，采用 4% 分裂比例时，引导方法用 811 状态，模型误差基线用 2532 状态。32×32 硬风险压力中，引导方法用 693 状态得到证书；模型误差基线在 20 轮、2725 状态时仍未认证。这个压力场景刻意把目标奖励乘以 10，并引入安全成本带，使硬阈值 5 达到边界；它**不是论文原始参数**。

必须保留反例：只按访问量的基线在 24×24 近目标、4% 分裂比例下用 **425** 状态获得证书，少于引导方法的 **467** 状态；32×32 近目标用 **736** 状态，也少于引导方法的 **811** 状态。因而实验支持“相比这个任务无关模型误差抽象，在若干大模型上更省状态且快于稀疏 LP”，**不支持**“对所有抽象基线都更省状态或更快”。参数扫描中，引导方法在 15/15 组场景与分裂比例组合上认证，占据率基线在 8/15 组认证；优势更偏向参数鲁棒性，而不是逐例最佳状态数。

近目标扩展把目标留在起点附近，新增的大部分区域距离目标较远；这是有利于抽象的规模增长方式，尚不足以证明在复杂、必须远距离规划的交通网络也会提速。还需要多种拓扑、目标位置、自动机大小、风险预算、种子和真实场景，才能支撑论文中的普遍性主张。

对不利场景的补充扫描见 `middle_limit.json`：16×16 地图把目标移至中部，4%、8% 分裂比例的引导方法在 20 轮内未达到 1% 证书；增加到 16% 后以 573 状态认证，但这次用时 1.109 s，稀疏 LP 为约 0.8 s。相同 16% 设置下，占据率基线 1387 状态、7.247 s 后仍未认证。这说明参数需要随场景调整，且证书成功不等于一定提速。

后续诊断发现一个应先于细化执行的标准 CMDP 修复：32×32 硬风险案例的证书差距约 0.00122，几乎全部来自硬风险的对偶互补项，策略决策遗憾约为零。`mixing_probe.py` 在完整模型上求得风险 14.034 和风险 0 的两个标量化策略，按 0.35627695 与 0.64372305 混合占据分布，再归一化为平稳随机策略，得到期望风险恰为 5、目标值 0.8310559006211404，与完整稀疏 LP 一致。详细数值在 `mixing_probe.json`。策略混合是已有的 CMDP 原理，**不是本文新定理**；这个例子仅说明该证书差距无需继续拆状态。约束是折扣期望风险，不保证每条轨迹的风险都不超过 5。

## 运行

从仓库根目录，以装有 NumPy/SciPy 的 Python 运行：

```powershell
python reproduction/adaptive_risk_ltl/check.py
python reproduction/adaptive_risk_ltl/experiment.py --sizes 16 20 24 --scenarios near barrier_hard --strategies guided occupancy model_error uniform --fraction 0.08 --rounds 20 --output reproduction/adaptive_risk_ltl/final_selected.json
python reproduction/adaptive_risk_ltl/sensitivity.py
python reproduction/adaptive_risk_ltl/benchmark.py --repeats 5 --cases 8:near 16:near 20:near 24:near 20:barrier_hard 24:barrier_hard --output reproduction/adaptive_risk_ltl/timing_strict.json
python reproduction/adaptive_risk_ltl/benchmark.py --repeats 3 --cases 32:near 32:barrier_hard --output reproduction/adaptive_risk_ltl/timing_large32.json
```

原始逐轮结果在 `final_selected.json`、`large32.json`、`sensitivity.json` 和两份 `timing_*.json`，中部目标的参数摘要在 `middle_limit.json`。`final_selected.json` 中 `model_error` 或 `occupancy` 的未认证项表示 20 轮内未达到证书，不能把这些项的目标值作为同等精度结果比较。

`plot_results.py` 只读取上述 JSON 生成 `scaling.png`，不重新运行求解器。
