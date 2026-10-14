# 证书修正后的 CARLA Town01 闭环复测

本次使用提交 `8792ab4` 的代码、CARLA 0.9.16，在 Town01 重跑原来的单行人和双行人实验。每种场景重复两轮；仿真步长 0.05 秒（20 Hz），每 tick 查询规划器，每 10 tick 执行一次采样动作。单行人最多 900 tick，双行人最多 1200 tick。比较对象是保留上次求解基础的稀疏 HiGHS LP。轮次 1 将两种方法的执行顺序倒过来。

`planning_time_s` 是各方法自身产品构造及累计规划查询的墙钟时间，不含 CARLA 仿真推进。闭环控制可能让两种方法走过不同轨迹，因此各自的查询数和到达时刻不必相同。旧版原始结果仍保留在 `../observed_hybrid_probe/`；本表全部来自修正代码的新 JSON。

| 场景与轮次 | 图谱累计规划 | 保基 LP 累计规划 | 倍数（LP/图谱） | 图谱直接命中 | 图谱/LP 查询数 | 图谱/LP 到达仿真时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单行人 0 | **3.300 s** | 5.205 s | 1.58× | 417/421 | 421/420 | 21.05/21.00 s |
| 单行人 1 | **3.826 s** | 7.151 s | 1.87× | 597/601 | 601/601 | 30.05/30.05 s |
| 双行人 0 | **7.816 s** | 10.467 s | 1.34× | 711/721 | 721/694 | 36.05/34.70 s |
| 双行人 1 | **6.489 s** | 8.450 s | 1.30× | 704/714 | 714/580 | 35.70/29.00 s |

八段驾驶均到达终点，均未记录碰撞或规划异常。图谱不使用 LP 兜底；四轮分别有 4、4、10、10 次图谱证书未直接命中，先尝试补充策略，必要时再做完整 CMDP 搜索，最终均获得证书。基线 LP 每次查询均进行保基重优化。

对图谱真实经历的四条观测轨迹，另用 `benchmark_fixed_moving_replay.py` 逐点运行独立保基 LP 并重新评估图谱返回的策略。单行人 421/601 个点、双行人 721/714 个点全部通过：策略风险不超过硬阈值 $H=2$，图谱目标值不高于 LP 最优值，LP 最优值不高于图谱证书上界，且上下界间隙满足实验设定的 1% 或绝对容差。四条轨迹的最大风险分别约为 1.928、1.928、1.978、1.978；没有严格达到硬风险边界。此处的“通过”是对高层产品模型的期望折扣指标进行数值核验，不是逐条驾驶轨迹的数学安全证明。

图谱方法的在线单次规划 p95 分别为 6.5、6.8、11.0、8.1 ms，但最慢单次为 **531、486、745、713 ms**，均超过 20 Hz 的 50 ms 步长。因此累计规划更快不能写成严格实时保证。两种方法在双行人第 1 轮的到达时刻相差 6.70 秒；规划计算提速也不能写成车辆行驶更快。

与修正前记录相比，图谱累计耗时上升，加速倍数下降。这次修正需要在选定价格处实际计算 Bellman 残差，增加了查询工作量；但旧、新结果不是受控微基准，机器负载、CARLA 轨迹和 LP 本身耗时也有变化，不能仅凭这两次运行把全部差值归因于代码修正。当前证据只支持在这四轮里图谱的**累计规划时间**低于保基 LP；还不足以声称跨场景稳定提速。

## 原始数据与复现

- `single_20hz_900_observed_vs_lp.json`、`double_20hz_1200_observed_vs_lp.json`：CARLA 闭环逐 tick 记录和计时。
- `single_replay_rep0.json`、`single_replay_rep1.json`、`double_replay_rep0.json`、`double_replay_rep1.json`：图谱轨迹的独立 LP 核验。回放时间包含独立核验，与上表闭环规划时间不可混用。

从仓库根目录运行；先启动本地 CARLA 0.9.16 RPC 服务（端口 2000），再使用含 `carla`、NumPy、SciPy、HiGHS 等依赖的 Python 环境：

```bash
python reproduction/carla_local_risk/closed_loop_dynamic.py --town Town01 --methods fixed_observed_hybrid fixed_warm_lp --repeats 2 --max-ticks 900 --plan-every 1 --action-every 10 --sample-policy --output reproduction/carla_local_risk/revised_certificate_20261014/single_20hz_900_observed_vs_lp.json
python reproduction/carla_local_risk/closed_loop_dynamic.py --town Town01 --methods fixed_observed_hybrid fixed_warm_lp --repeats 2 --second-walker --max-ticks 1200 --plan-every 1 --action-every 10 --sample-policy --output reproduction/carla_local_risk/revised_certificate_20261014/double_20hz_1200_observed_vs_lp.json
```

对每个图谱轮次再调用 `benchmark_fixed_moving_replay.py --hybrid --observed-prepare --quiet`；单行人使用 `carla_trace_20hz.json` 作为参考地图和标签，双行人使用 `carla_trace_20hz_two_walkers.json`，`--trial` 分别选 0 和 3。
