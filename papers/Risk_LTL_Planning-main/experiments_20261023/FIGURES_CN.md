# 实验图片及建议图注

全部图片由本目录 Python 脚本从原始记录生成，提供 PNG 和 SVG。
误差条或阴影均为本批观测到的最小—最大范围，不是置信区间。

| 图片 | 建议图注与口径 | 生成代码 |
|---|---|---|
| [主消融](analysis/ablation_totals.png) | 三种固定产品模型下各方法自身初始化与连续 20 次查询的累计时间。每模型三种风险序列种子、每种三次计时；展示中位数和观测范围。 | analyze_results.py |
| [策略组合和完整求解](analysis/mixture_and_expansion.png) | 不同消融方法中缓存混合策略的返回比例，以及完整求解或 LP 救援的比例。这两类事件可能在同一查询发生，不是互斥类别。 | plot_supplement.py |
| [长序列](analysis/long_sequence.png) | 160 次连续局部风险更新的累计时间，包含各方法初始化；三次相同输入的计时范围。 | analyze_results.py |
| [原十二条输入](analysis/historical_twelve_ratios.png) | 原十二条输入在修正证书的统一驱动下，保基 LP 与完整方法的配对时间比。大于 1 表示完整方法较快；不是独立道路分布上的概率。 | plot_supplement.py |
| [固定 CARLA 观测回放](analysis/paired_replay_times.png) | 相同历史观测与起点序列上三种方法的重新计时，各重复三次；不代表新闭环驾驶。 | plot_supplement.py |
| [新驾驶的道路进度](analysis/new_carla_progress.png) | 新驾驶中车辆纵向位置随仿真时间变化，虚线为目标阈值；叉号为规划异常前最后记录位置。各方法均保留未到达结果。 | plot_supplement.py |
| [成功驾驶规划时间](analysis/new_carla_totals.png) | 新驾驶中成功且无碰撞、无规划失败的自身累计规划时间。失败次数单列，失败的较短或不完整计时不用于提速结论。不同轨迹不是同输入求解器配对。 | plot_supplement.py |
| [新驾驶延迟](analysis/new_carla_latency.png) | 已保存查询耗时的经验分布和 50 ms 参考线。未记录耗时的异常调用不视为零耗时；同步 20 Hz 不等于墙钟实时。 | plot_supplement.py |

SVG 与表内 PNG 同名。详细配置、重复层次、任务结果和证书核验见 [报告](README_CN.md)。
