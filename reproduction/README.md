# IROS 2025 公开示例复现

此处保留 Qi 等人《Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications》的[公开代码](https://github.com/Miracle-qi/Risk_LTL_Planning)及本机复现。源代码快照位于 `papers/Risk_LTL_Planning-main/`，对应上游提交 `fcd7424be5bbd92a2bd47611dc3aad208194f678`。

公开仓库仅提供简化示例，没有完整 CARLA 场景和评测流水线；`adaptive_risk_ltl/` 和 `carla_local_risk/` 是本项目后续独立研究代码。

## 环境与快速检查

Windows、Python 3.11。仓库根目录运行：

```powershell
uv venv .venv-repro --python 3.11
uv pip install --python .venv-repro/Scripts/python.exe -r reproduction/requirements-frozen.txt
./reproduction/run_all.ps1
```

`run_all.ps1` 只执行 IROS 公开示例的无界面复现及原论文策略提取的诊断，不启动 CARLA，也不运行耗时的合成或闭环基准。结果保存在 `iros_core_results.json` 和 `policy_extraction_*_results.json`；运行日志在被 Git 忽略的 `reproduction/logs/`。

原代码的 `ltlf2dfa==2.0.0` 在 Windows 调用 POSIX 专用的 `os.setsid`。`iros_core.py` 与 `iros_demo_smoke.py` 为作者示例中固定的 `F(t)` 和 `G(~g -> ~c) & G(~v)` 提供等价的两态 DFA，仍使用作者的抽象、产品构造、风险 LP 和示例 MPC。这不代表任意 LTL 公式都已在 Windows 复现。

原示例的产品模型有 128 个状态、9 个动作；其 LP 目标值约为 `1.3620227941`，期望风险约为 `0.8075040145`，硬上限为 `5`。示例网格边界处存在行和小于 1 的转移。后续实验在需要标准随机 MDP 时明确加入吸收态。原代码从占据分布提取单个确定性动作，也未必保留随机 CMDP 解的风险保证；相关诊断文件留在本目录。

当前方法、基线与结果请读[项目入口](../README.md)、[合成模型说明](adaptive_risk_ltl/README.md)及[CARLA 结果](carla_local_risk/observed_hybrid_probe/FOUR_CASE_AND_ARRIVAL_CN.md)。
