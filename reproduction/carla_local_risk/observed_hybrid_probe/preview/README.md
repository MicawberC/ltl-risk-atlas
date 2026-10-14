# CARLA 场景可视回放

`single_walker.png` 和 `two_walkers.png` 是 Town01 路线 0 的 CARLA 0.9.16 画面。黄色方块标出高层规划模型中的五处风险单元格；单行人从一侧横穿，双行人中第二人在第 100 个仿真 tick 从另一侧加入。

`reproduction/carla_local_risk/show_crossing_scenarios.py` 读取已保存的闭环试验位置记录，在可见的 CARLA 窗口中依次回放两段场景。**这是可视化回放，不重新运行规划器，也不产生新的任务成功率或计时结果。**

启动 CARLA 0.9.16 服务端后，在仓库根目录运行：

```powershell
.\.venv-repro\Scripts\python.exe reproduction/carla_local_risk/show_crossing_scenarios.py --cycles 1
```
