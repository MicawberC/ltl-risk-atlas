# CARLA Town01 行人横穿：局部风险更新实验

后续的多轨迹单调证书与 LP 锚点实验见 [MONOTONE_LP_ANCHOR_REPORT.md](MONOTONE_LP_ANCHOR_REPORT.md)。下表保留最初 20 点图谱方法的结果，不能代表后续改进。

下降风险、局部上界和策略生成的三阶段改进实验见 [THREE_STAGE_UPDATE_REPORT.md](THREE_STAGE_UPDATE_REPORT.md)。

## 场景和与原论文的关系

本场景参考 Qi 等的 [Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications](https://arxiv.org/pdf/2409.09769) 中的行人横穿实验：目标是最终到达目标区，同时避免“行人在冲突区时车辆进入冲突区”。论文使用行人环境马尔可夫链、LTL 公式 `□(p → ¬c) ∧ ◇t`，碰撞代价 8。这里的独立实现沿用两状态环境链和两个自动机，并额外用 CARLA 车道掩码标记离路区域。不是作者未公开的 CARLA 项目文件或场景复现。

仿真器为 CARLA 0.9.16、Town01、固定 0.05 秒同步步长。车辆在横穿点前约 8 米处保持制动，一名行人以 1.3 m/s 横穿车道。每 8 帧记录一次行人位置，共 20 次、约 8 秒。每次把行人与五个固定冲突单元格的距离转成局部风险严重度，形成同一个产品模型上的 20 次风险代价更新。五个单元格按车道纵向排列，因横穿轨迹接近对称，其五维风险向量并非五个独立扰动。

20×20、每格 2 米的网格根据 Town01 车道 waypoint 标记可行驶区域，54/400 个网格可行驶。两个环境状态、两态目标自动机、两态安全自动机和吸收态形成 **3201 个产品状态、9 个规划动作**。车辆运动成功率 0.75、环境链转移概率 0.8/0.2、距离到风险严重度的高斯形状均是当前研究模型的人工设定，**不是由 CARLA 数据估计**。固定环境转移和固定标签是策略图谱证书有效的前提。本次 CARLA 只提供地图与行人位置；没有让图谱策略控制车辆，不是闭环自动驾驶试验。车辆静止时无碰撞只是数据采集状态，不可当作策略安全成功率。

## 结果（2026-10-13，一次完整配对运行）

使用论文 CARLA 实验的 `γ=0.8`、软风险阈值 `1`、硬风险阈值 `2`。强基线是同一产品模型上的 HiGHS 稀疏占用度量 LP，保存模型和 simplex basis，每次仅更新风险系数。计时含各自准备阶段，逐次规划时不含 CARLA 启动、地图构造和仿真运行时间。

| 指标 | 策略图谱（本方法） | 保留基 LP |
|---|---:|---:|
| 准备时间 | 0.794 s | 0.712 s |
| 20 次规划总时间，含准备 | **5.373 s** | **1.732 s** |
| 逐次延迟中位数 | 61.0 ms | 41.4 ms |
| 逐次延迟第 95 百分位 | 935.1 ms | 68.5 ms |
| 最大逐次延迟 | 1064.3 ms | 218.7 ms |
| 硬风险阈值满足次数 | 20/20 | 20/20 |

图谱 10/20 次直接复用已有证书，4 次触发完整 CMDP 扩充；直接命中的多数点为 11–19 ms，但扩充导致很长的尾部延迟。图谱目标值与最优 LP 的最大绝对差为 0.000512，最大证书间隙为 0.006602，均在图谱预设的 1% 相对容差内。20 次图谱风险值均为 1.0，低于硬阈值 2。三个时点另用从零稀疏 LP 验证了热启动 LP 的最优目标值。只缓存名义策略的简单方法在 16/20 点也达到 1% 目标容差。

**结论：这一条 CARLA 观测序列中，本方法没有实现总体提速；含准备时间约为强 LP 的 3.1 倍。** 因此不能把此前合成网格中局部更新提速的结论直接外推到真实仿真轨迹。场景主要揭示了新风险位置使证书失效、策略扩充造成延迟尖峰。当前 0.05 秒仿真步长与逐次求解时间也不能支持实时驾驶主张。原论文表 I 的单次 LP 时间使用不同的场景状态数和平台，不能直接和这里的 3201 状态实验相除。

## 文件与运行

- `capture_crossing.py`：在 CARLA Town01 中采集行人轨迹与车道标签。
- `carla_trace.json`：实际采集的 20 次 CARLA 观测。
- `benchmark.py`：图谱、保留基 LP 与三个从零 LP 抽查。
- `benchmark.json`：逐次结果及汇总。
- `plot_results.py` / `carla_benchmark.png`：风险变化和逐次规划延迟。

启动 CARLA 0.9.16 服务端后，在仓库根目录执行（Windows PowerShell）：

```powershell
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\capture_crossing.py
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\benchmark.py
.\.venv-repro\Scripts\python.exe reproduction\carla_local_risk\plot_results.py
```

所需 Python 包：`carla==0.9.16`、`numpy`、`scipy`、`highspy`、`matplotlib`。CARLA 服务端安装包位于项目仓库外，不提交到 Git；输出 JSON 和图片已提交，可直接复核数据。`benchmark.py` 使用 `reproduction/adaptive_risk_ltl/` 下的数学模型与基线代码。

下一阶段若要检验论文级主张，应先解决扩充尖峰，再让策略驱动车辆并评估多次闭环过街、碰撞与交通违例；同时估计或校准环境转移模型，并在不同路段及行人速度下重复实验。
