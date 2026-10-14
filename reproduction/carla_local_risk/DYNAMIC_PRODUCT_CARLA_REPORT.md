# 动态产品模型与 CARLA 闭环实验（2026-10-13）

## 研究问题与实现

本实验从原有固定产品模型扩展到：自车起点随驾驶移动；行人位置引发局部两态环境转移概率变化；冲突区的安全命题标签在距离阈值 2.5 m 处切换。路网和 LTL 自动机的**状态集合及公式固定**。环境转移概率、标签阈值和距离到风险严重度的映射是研究者设定的 CARLA 观测解释，未经数据估计。CARLA 提供 Town01 道路与实际演员位置。

产品状态仍有 3201 个。局部更新只重建：环境转移改变的物理格的产品出边，以及标签改变格及其一步前驱的产品出边。典型一步为 360/28,809 个状态动作转移行；标签切换时可增至 792–1,512 行。起点变动只改初始状态索引。模型的风险代价和任务奖励随后按新模型更新。26 个保存的 CARLA 观测点中，局部更新与每次完整构造模型的转移、奖励、代价逐项一致。

旧 LP 策略的占用量不能跨转移变化直接沿用。我们在新产品模型中重新计算该策略占用量，给出可行目标下界。旧 LP 对偶值函数 $v$ 和软/硬风险价格 $\lambda_s,\lambda_h$ 在新模型中的正 Bellman 残差为

$$e(s)=\max_a[r'(s,a)-(\lambda_s+\lambda_h)c'(s)+\gamma P'_{s,a}v-v(s)]_+.$$

从 $\max_s e(s)/(1-\gamma)$ 的全局超解开始迭代 $w\leftarrow e+\gamma\max_aP'w$，每次迭代仍是有效上界。于是 $v+w$ 给出新模型的标量化对偶可行上界，结合软/硬阈值常数得到原问题目标上界。仅当新策略风险 $\leq2$ 且上界与下界差 $\leq1\%$ 时复用；否则调用 LP。该证书保持**抽象模型中的期望折扣风险与目标界**，不是连续车辆零碰撞证明。

实现文件：`dynamic_product.py`、`dynamic_certificate.py`、`dynamic_warm_lp.py`。只有转移和标签**局部改变**；当前策略重评仍解一个全局稀疏线性系统，残差上界也扫描全部产品状态，因此“局部更新”不等于全部计算都局部化。

## CARLA 闭环试验设置

CARLA 0.9.16 Town01、0.05 s 同步仿真步长；每 10 步（0.5 s 仿真时间）重规划一次。低层控制器使车辆以约 3 m/s 前进或制动，执行高层策略中“前进”动作的概率是否至少 0.5。为了与直线道路上的车辆动作相容，**这个闭环实验使用前进/等待的车道受限产品模型**：代码仍保留 9 个动作索引，除前进动作外的索引在该实验都代表等待。这与此前允许横向移动的 9 动作网格实验不是同一模型，不能把两者耗时合并。

三组在同一场景交替运行两轮：①完整重建产品＋从零稀疏 LP；②局部更新产品＋从零稀疏 LP；③局部更新产品＋旧策略/对偶证书，未命中时从零 LP。规划时间包括各自产品更新和求解；不包括公共 CARLA 仿真步、风险观测与独立核验。`wall_time_s` 包括仿真与规划，但同步 CARLA 在规划时暂停，**不等于真实车辆行驶时间**。

| 场景（两轮中位数） | 完整重建＋LP | 局部产品＋LP | 局部产品＋证书/LP | 仿真到达时间 |
| --- | ---: | ---: | ---: | ---: |
| 单行人，24 次重规划 | 5.999 s | 4.153 s | **2.142 s**；10 次 LP | 三组均 11.55 s |
| 双行人，31 次重规划 | 7.597 s | 5.056 s | **3.682 s**；18 次 LP | 三组约 15.4–15.5 s |

上述两种场景共 12 次试跑均到达目标、未记录碰撞。**加速体现为规划计算减少，未体现为仿真中的任务更快完成。**同步仿真暂停了规划期间的物理时间；而且三组高层动作大多相同。单行人方法的两轮规划时间为 2.120/2.164 s，双行人为 3.792/3.572 s。双行人每次完整更新的第 95 百分位耗时约 259 ms，远大于 50 ms 仿真步，不能声称 20 Hz 实时规划。

若另看从演员生成到终止的**计算机墙钟时间**，单行人三组中位数依次为 7.423/5.444/3.089 s，双行人为 9.460/6.649/4.972 s。同步 CARLA 可以比真实时间更快地推进仿真，同时在求解器运行时暂停，所以这些墙钟数字反映当前实验脚本的吞吐量，不能解释为真实车辆在路上节省了这么多秒。

原始闭环数据：`closed_loop_single_threeway.json`、`closed_loop_two_threeway.json`。另一次双行人方法试跑将 31/31 个策略逐点与独立 LP 比较，并在新产品上重评策略，通过所有断言：`closed_loop_two_verified.json`。

## 更强的保留基 LP 对照与反例

我们还实现了保留 HiGHS simplex basis 的动态 LP：只改受影响流守恒系数、风险系数、奖励和起点，不重建 LP。为消除不同控制策略导致的轨迹差异，把**同一条实测 CARLA 自车及行人轨迹**同时送给保留基 LP、证书＋从零 LP、证书＋保留基 LP，以及完整重建 LP，并逐点用独立 LP 核查。

| 同轨迹离线回放 | 保留基 LP | 证书＋保留基 LP | 证书＋从零 LP | 完整重建＋从零 LP |
| --- | ---: | ---: | ---: | ---: |
| 单行人，24 点 | **1.134 s** | 1.186 s | 1.456 s | 3.831 s |
| 双行人，31 点 | **1.697 s** | 1.821 s | 2.278 s | 4.745 s |

这两条流中，证书分别回退 10/24 和 18/31 次。**当前方法没有快于强保留基 LP 基线**；预检查及策略重评的开销超过被省下的 LP 热重优化时间。回放最大目标差在机器数值精度量级，所有策略风险不超过硬阈值；最大风险约 1，硬阈值 2 **未激活**。数据见 `dynamic_live_replay_single_hybrid.json` 与 `dynamic_live_replay_two_hybrid.json`。

单独让保留基 LP 驱动双行人 CARLA 场景时，两轮都在 9.2 s 仿真时间与行人发生碰撞，未完成任务；事件的 `other_actor` 为 `walker.pedestrian.0001`。逐点重新评价保留基策略并与从零 LP 对比均通过目标与风险检查。于第 160 步，它选择前进，而其他方法在各自试跑中选择等待；抽象目标值相同。**这说明抽象期望风险/目标最优性不能保证连续车辆安全，也不能从这两次事件断言某算法普遍更安全。**原始记录：`closed_loop_two_warm_repeated.json`、`closed_loop_warm_diagnostic.json`、`closed_loop_warm_collision_event.json`。

## 研究判断

1. 已得到可运行的局部产品更新和跨起点、局部转移、局部标签变化的风险/目标证书；独立 LP 与完整产品重建抽查通过。
2. 与从零重建求解相比可减少规划计算；与**最强保留基 LP**相比当前没有速度优势。因此不能把现有结果写成“动态变化下稳定提速”的论文主创新。
3. 两种到达时间比较没有任务效率改善。同步 2 Hz 高层规划也没有实时安全证明。更严谨的下一步是缩短证书预检查、只在真正受影响的自动机可达域传播对偶残差，并在多地图、约束活跃、异步闭环与独立种子上比较。还需独立设计物理安全屏障和策略等价时的安全择优规则；不能用优化目标证书替代碰撞保护。

## 复现

在仓库根目录使用 `.venv-repro`。完整 CARLA 对照需要启动 0.9.16 服务端；已保存的离线配对不需要服务端：

```powershell
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\closed_loop_dynamic.py --methods full_lp patched_lp incremental --repeats 2 --second-walker --max-ticks 440 --plan-every 10 --output reproduction\carla_local_risk\closed_loop_two_threeway.json
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\benchmark_dynamic_live_replay.py --live reproduction\carla_local_risk\closed_loop_two_threeway.json --trial 0 --output reproduction\carla_local_risk\dynamic_live_replay_two_hybrid.json
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\benchmark_dynamic_product.py --stride 10 --frames 26 --ego-from reproduction\carla_local_risk\closed_loop_two_complete.json --output reproduction\carla_local_risk\dynamic_product_benchmark.json
```
