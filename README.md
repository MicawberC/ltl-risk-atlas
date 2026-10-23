# LTL 风险规划：策略图谱与 CARLA 实验

本项目研究固定 LTL 产品模型中局部风险代价连续变化时的重复规划：用策略图谱、风险与目标值证书及按需扩充减少累计规划时间。主要对照是保留 simplex basis 的稀疏占据度量 LP。研究代码基于 IROS 2025 风险感知 LTL 规划公开示例独立扩展。

## 从哪里开始

- [2026-10-23 补充实验报告](papers/Risk_LTL_Planning-main/experiments_20261023/README_CN.md)：统一条件消融、重复计时、风险变化敏感性、固定轨迹重跑和新增 CARLA 场景；与下列历史批次分别统计。
- [阶段性中文结果总报告](docs/RESULTS_SUMMARY_CN.md)：统一汇总修正代码后的时间、证书、CARLA 任务结果、反例与论文边界。
- [原论文代码快照](papers/Risk_LTL_Planning-main/)：IROS 2025 作者公开的简化示例。
- [原示例复现说明](reproduction/README.md)：来源、兼容处理及复现边界。
- [合成模型实验](reproduction/adaptive_risk_ltl/)：产品模型、图谱算法、保基 LP 基线和 24×24、32×32 结果。
- [历史关键计时报告](reproduction/adaptive_risk_ltl/key_timing_20261014/KEY_TIMING_REPORT_CN.md)：残差修正前的固定产品模型配对规划时间。
- [CARLA 实验](reproduction/carla_local_risk/)：场景、闭环控制、轨迹与基线。
- [四组规划计时和任务到达报告](reproduction/carla_local_risk/observed_hybrid_probe/FOUR_CASE_AND_ARRIVAL_CN.md)：合成网格、单行人和双行人结果及其限制。
- [修正后 CARLA 闭环复测](reproduction/carla_local_risk/revised_certificate_20261014/REPORT_CN.md)：Town01 单、双行人各两轮，保基 LP 对比和逐点核验。
- [方法定理、证明与修订版验证](docs/METHOD_THEORY.md)：策略区域、占据度量组合、上下界证书、失败时的返回条件及修订代码复测。

合成网格的加速结论不能直接等同于 CARLA 的实时驾驶性能。CARLA 实验使用研究者构建的高层网格和风险映射，不是原论文未公开的完整仿真代码。所有数值和适用边界以对应报告及原始 JSON 为准。
