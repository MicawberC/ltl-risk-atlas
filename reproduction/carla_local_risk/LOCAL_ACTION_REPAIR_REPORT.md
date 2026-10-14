# CARLA 风险更新：局部动作修复实验（2026-10-13）

## 问题与方法

每个风险观测到来时，先对缓存策略做原有的风险可行性和最优性证书检查。证书成立时直接返回策略，不运行修复。证书失败时，实验性方法选取少量 Bellman 优势为正、且旧折扣占用量较高的产品状态，只修改这些状态的动作。设旧策略的系统矩阵为 $M_\pi=I-\gamma P_\pi$；改动 $m$ 行后，用缓存的稀疏 LU 分解和 Woodbury 公式更新策略价值与折扣占用量。无需重新分解整个策略矩阵。

修复后重新计算完整产品模型中的目标值和风险。新的可行占用量给出目标下界；旧证书的单调上界，或对新策略计算的 Bellman 残差上界，给出最优目标上界。仅当硬风险不超过 2、且上下界差不超过目标值的 1% 时接受修复；否则调用完整的保留基稀疏 LP。修复失败的候选占用量不会进入长期缓存。`--repair-max-candidates 64` 是可选的搜索范围门控：Bellman 候选状态过多时直接走后续路径。它只影响耗时，不影响正确性。

还实现了一个低维占用量 LP：固定其他状态的旧动作，只优化少数候选状态的动作占用量，并用旧 LU 消去完整流守恒等式。它是一个消融方案，并非最终推荐方法；在本次数据上没有比直接低秩修复更好。

## 场景与验证

- 原有四条 Town01 单行人 CARLA 轨迹，每条约 160 个 20 Hz 观测，在同一个 20×20 网格、3201 产品状态、9 个动作的模型上运行。测试三个轨迹顺序，每个顺序 654 次更新。
- 新采集一条真实 CARLA 0.9.16 Town01 双行人轨迹：静止车辆，一名行人先横穿，第二名行人在第 100 个 0.05 秒仿真步开始从另一侧横穿；连续 260 帧记录两人的位置。风险系数由到各冲突单元格最近的行人距离经预先设定的高斯函数计算。CARLA 只提供道路和行人位置；运动概率、风险映射和 LTL 产品模型仍是研究假设。此试验是离线逐帧规划回放，不是车辆闭环安全试验。
- 每个已输出的策略都由独立 HiGHS 保留基稀疏 LP 核验：重新计算策略目标和风险、检查风险硬阈值、检查最优值在上下界内，以及 1% 证书间隙。所有完成的运行均通过断言。双行人结果的最大目标绝对差为 0.001950，最大证书间隙为 0.007548；最大规划风险为 1，低于硬阈值 2。这里的硬约束是松弛的，因此不能据此声称验证了硬约束活跃情形。

## 单行人：三种顺序的消融

下表是同一 654 次更新序列上的**完整 LP 回退次数**。时间有系统负载与 LP 基路径波动，不把少量毫秒差称为稳定提速。

| 轨迹顺序 | 单调证书＋LP 锚点 | 加已有的前瞻锚点 | 再加局部动作修复 |
|---|---:|---:|---:|
| 1 | 12 | 8 | 7 |
| 2 | 12 | 10 | 8 |
| 3 | 13 | 10 | 10 |

局部修复最多改变 4 个产品状态动作；三个顺序分别成功 3、3、2 次。它有时能再省 1–2 次 LP，却没有在三个顺序中稳定降低累计规划时间。更宽的 8 动作修复也没有稳定改善。

## 双行人：交错顺序、服务端关闭后的重复计时

采集结束后关闭 CARLA 服务端，避免其 CPU/GPU 负载污染规划计时。下列两轮按“基线→前瞻→前瞻＋修复”和反向顺序串行运行；各方法都在同一 260 帧上逐帧核验。

| 方法 | 完整 LP 回退 | 第 1 轮总时间，含准备 | 第 2 轮总时间，含准备 | 中位数 |
|---|---:|---:|---:|---:|
| 单调证书＋LP 锚点 | 89 | 2.905 s | 2.910 s | 2.908 s |
| 加前瞻锚点 | 35 | 2.417 s | 2.485 s | **2.451 s** |
| 再加局部修复，候选范围≤64 | 35 | 2.447 s | 2.467 s | 2.457 s |

因此，相对已有前瞻方法，局部修复没有减少 LP 次数，累计时间也没有可辨别优势。单独启用局部修复时仍回退 89 次（虽有 1 次局部修复成功）；无限制地在每次证书未命中时尝试局部修复，或扩大到 16 个修改动作，均未降低回退次数，并增加耗时。第二名行人加入后，局部 Bellman 候选状态通常增至约 99–101 个；只改 4 个动作时新策略的 Bellman 残差仍大，难以形成紧上界。前瞻锚点直接处理持续下降的风险序列，才是本轨迹减少回退的主要来源。

仅“前瞻锚点”相对强单调证书基线的中位总时间降低约 15.7%，且比每次完整 LP 的独立基线更快。但当前仍只是一条双行人道路、静止车辆的离线回放，不能由此推断闭环驾驶延迟或安全收益。

## 更细网格压力测试

将四条已采集的 CARLA 风险序列用最近邻道路标签投影到 32×32 网格，取每条前 80 帧，共 320 次更新。**这是离散化加密，不是新的 CARLA 采集。**基线完整 LP 回退 23 次，总时间 12.840 s；4 动作局部修复尝试 22 次、成功 0 次，回退仍 23 次，总时间 13.266 s；8 动作版本回退 21 次，总时间 12.979 s。HiGHS 在跨轨迹切换时曾出现热启动基数值失败，实验代码重置基并把恢复耗时计入结果，故本组不适合作为精确加速比证据。它至少说明当前局部修复不会随模型放大自动变强。

## 研究判断与下一步

局部动作修复的数学步骤和证书链是可行的，能作为**负面消融或受限场景的辅助模块**写入实验部分；现有结果不足以把它写成论文主创新或声称稳定提速。当前较可靠的正面结果来自已有的单调证书＋前瞻锚点。若继续追求局部修复作为主创新，需要从“只改 4 个动作”升级为能够处理大影响区的结构化区域分解，同时在多道路、多行人、移动自车和硬风险约束活跃的闭环场景中验证，并与保留基稀疏 LP 和前瞻方法分别比较。不能仅增加候选动作数或只换一个小 LP 求解器。

## 关键文件

- `local_policy_repair.py`：低秩动作修复和证书；位于 `reproduction/adaptive_risk_ltl/`。
- `local_reduced_lp.py`：低维占用量 LP 消融；位于同一目录。
- `multi_episode.py`：实验开关、逐帧验证、计时及 JSON 输出。
- `capture_crossing.py` 与 `carla_trace_20hz_two_walkers.json`：双行人 CARLA 采集器及真实观测。
- `rep_1_*.json`、`rep_2_*.json`：双行人交错顺序重复实验的逐帧原始结果。
- `refine_trace_grid.py` 与 `grid32_first80/`：可复现的网格加密压力测试。

在仓库根目录、安装了项目依赖后可重跑双行人主消融；此回放只读取已保存的 CARLA JSON，不需要启动 CARLA 服务端：

```powershell
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\multi_episode.py --lp-anchor-fallback --traces reproduction\carla_local_risk\carla_trace_20hz_two_walkers.json --output reproduction\carla_local_risk\recheck_control.json
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\multi_episode.py --lp-anchor-fallback --lookahead-anchor 2 --traces reproduction\carla_local_risk\carla_trace_20hz_two_walkers.json --output reproduction\carla_local_risk\recheck_lookahead.json
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\multi_episode.py --lp-anchor-fallback --lookahead-anchor 2 --low-rank-repair --repair-max-changes 4 --repair-max-bases 1 --repair-gap-factor 1.5 --repair-max-candidates 64 --traces reproduction\carla_local_risk\carla_trace_20hz_two_walkers.json --output reproduction\carla_local_risk\recheck_local_repair.json
```
