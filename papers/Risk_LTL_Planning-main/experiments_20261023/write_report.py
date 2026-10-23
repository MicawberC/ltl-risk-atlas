"""Summarize measured data only. Never fabricates missing experiment rows."""
from pathlib import Path
import collections,csv,json,statistics as st
HERE=Path(__file__).resolve().parent
METHODS=['warm_lp','full','no_mix','no_regions','no_stream','forced_mdp','frozen_lp']
def load(folder):
    out=[]
    for p in sorted((HERE/folder).glob('*.json')):
        try:
            d=json.loads(p.read_text('utf8'))
            if d.get('run_complete'):out.append(d)
        except (ValueError,OSError):pass
    return out
def med(ds,key):return st.median(d[key] for d in ds)
def main():
    data=load('results');replays=load('replay_results')
    audit=json.loads((HERE/'analysis/audit.json').read_text('utf8'))
    replay_audit=json.loads((HERE/'replay_analysis/audit.json').read_text('utf8')) if (HERE/'replay_analysis/audit.json').exists() else {}
    lines=['# 修订证书后的受控消融、重复统计和 CARLA 扩展实验','',
    '这是本团队在原作者代码快照目录下新增的实验产出，不属于 IROS 原作者的原始实现。基线快照为 `1dced97`；本目录包含新运行的原始记录、核验、绘图代码及环境信息。旧版计时不与本批数据合并。','',
    '## 1 实验口径与有效范围','',
    '- 固定产品转移、标签与 LTL 任务，仅更新局部风险代价；回放另外更新起点。γ=0.8，软预算 S=1，硬预算 H=2。',
    '- 主实验 3 个输入序列种子，每个种子重复计时 3 次。三次计时不是三个独立道路场景；表中中位数及最小—最大范围仅作描述统计，不作为置信区间。',
    '- 各方法在独立新进程中顺序执行，顺序由固定随机种子打乱；包含自身初始化和返回策略的查询成本，排除公共模型构建、Python 导入和独立策略评价。',
    '- 控制 OpenBLAS/OMP/MKL 与 HiGHS 为单线程。LP 保留模型和 simplex basis；原始/对偶可行性容差为 1e-9。',
    '- `full` 是统一受控驱动中的完整组合，复用生产实现的策略生成和残差上界；它并非历史闭环程序查询路径的逐行副本，其时间不得替换历史原始计时。',
    '- 硬预算压力例人为放大奖励，属于辅助压力测试；不能解释为 CARLA 风险已校准。',
    '- 风险和目标差证书针对高层模型的期望折扣指标；浮点残差检查不是区间算术的机器严格证明。','',
    '## 2 消融定义','',
    '| 版本 | 唯一主要变化 |','|---|---|',
    '| full | 缓存组合、区域价格检索、流式上界复用和按需扩充全部开启 |',
    '| no_mix | 禁止缓存确定策略之间的占据分布混合；完整 CMDP 回退仍可返回随机策略，保持原优化问题不变 |',
    '| no_regions | 不求策略区域端点，改用 0、1、源价格与上次价格计算有效残差上界；扩充规则相同 |',
    '| no_stream | 不使用单调上界档案或上次区域快速路径，每次扫描策略库 |',
    '| forced_mdp | 每次先求一个普通标量化 MDP，再认证及必要扩充 |',
    '| frozen_lp | 只保留初始化策略库；未认证时用持久化 LP 救援，不再扩充策略库；属于诊断性替代机制 |',
    '| warm_lp | 每次使用保基 LP 重优化，并恢复非负概率策略 |','',
    '`forced_mdp` 的 direct_hit 是强制求 MDP 后认证成功，不能解释为免求解的缓存命中。策略组合字段为 primal_i、primal_j、primal_weight；仅不同索引且权重严格介于 0 和 1 时计入实际组合。','',
    '## 3 主消融结果','',
    '| 模型 | 方法 | 运行数 | 累计时间中位数 [最小, 最大] / 秒 | 缓存组合返回数 | 完整求解数 | 查询失败数 |',
    '|---|---|---:|---:|---:|---:|---:|']
    grouped=collections.defaultdict(list)
    for d in data:
        if d['group']=='core':grouped[(d['case'],d['method'])].append(d)
    for case in dict.fromkeys(d['case'] for d in data if d['group']=='core'):
        for method in METHODS:
            ds=grouped.get((case,method),[])
            if not ds:continue
            times=[d['total_seconds'] for d in ds]
            lines.append(f"| {case} | {method} | {len(ds)} | {st.median(times):.3f} [{min(times):.3f}, {max(times):.3f}] | {sum(d['mixed_cache_returns'] for d in ds)} | {sum(d['full_solves'] for d in ds)} | {sum(d['failures'] for d in ds)} |")
    lines+=['','![受控消融累计时间](analysis/ablation_totals.png)','','速度差异须结合各序列的配对结果、缓存组合次数和完整求解次数解释；不将某一模块在全部场景中都有效作为预设结论。','',
    '## 4 支持范围、变化幅度、拓扑及长序列','',
    '| 模型 / 更新方式 | 支持维数 | 幅度 | 查询数 | 方法 | 时间中位数 / 秒 | 最大缓存策略数 | 最大缓存数组与 LU / MiB |',
    '|---|---:|---:|---:|---|---:|---:|---:|']
    sg=collections.defaultdict(list)
    for d in data:
        if d['group']=='sensitivity':sg[(d['case'],d['stream'],d['dimension'],d['magnitude'],d['count'],d['method'])].append(d)
    for (case,stream,dim,mag,count,method),ds in sg.items():
        rows=[r for d in ds for r in d['rows'] if r['status']=='ok']
        lines.append(f"| {case} / {stream} | {dim} | {mag} | {count} | {method} | {med(ds,'total_seconds'):.3f} | {max(r.get('cache_size',0) for r in rows)} | {max(r.get('cache_array_bytes',0) for r in rows)/2**20:.2f} |")
    lines+=['','![长序列累计时间](analysis/long_sequence.png)','','缓存数组与 LU 是可计数的数据量，不包含全部 Python/SuperLU 对象开销。RSS 为采样进程内存，并包含为独立核验保存的返回策略，不应称为纯规划器内存或操作系统严格峰值。长序列中的有限增长情况不能代替长期容量界或均摊复杂度证明。','',
    '## 5 同观测序列的重新计时','',
    '以下输入来自既有四条 CARLA 每 tick 观测序列；此次重新执行求解器计时和独立策略评价，不将它称为新驾驶或把旧回放脚本总耗时当作规划耗时。各方法的 theta 与初始状态序列以 SHA-256 对齐。','',
    '| 序列 | 方法 | 计时重复数 | 累计时间中位数 / 秒 | 查询失败数 |',
    '|---|---|---:|---:|---:|']
    rg=collections.defaultdict(list)
    for d in replays:rg[(d['case'],d['method'])].append(d)
    for (case,method),ds in rg.items():lines.append(f"| {case} | {method} | {len(ds)} | {med(ds,'total_seconds'):.3f} | {sum(d['failures'] for d in ds)} |")
    lines+=['','## 6 新增 CARLA 闭环实验','',
    '闭环沿用原 lane-only 高层模型和前进/等待控制，0.05 秒仿真步长，每 tick 查询，每 10 tick 执行动作。各方法经历的轨迹、查询数和到达时刻可能不同，累计计算时间不能直接解释为同输入纯求解器加速或车辆更快到达。','',
    '| 配置 | 方法 / 重复 | 到达 | 碰撞 | 规划失败 | 查询数 | 累计规划 / 秒 | P95 / 毫秒 | 最慢 / 毫秒 |',
    '|---|---|---|---|---|---:|---:|---:|---:|']
    live_rows=[];executions=[]
    for p in sorted((HERE/'carla_results').glob('*.json')):
        d=json.loads(p.read_text('utf8'))
        if p.name.endswith('_execution.json'):executions.append({'file':p.name,**d});continue
        if 'trials' not in d:continue
        import numpy as np
        for t in d['trials']:
            times=[(r.get('product_s',0)+r.get('plan_s',0))*1000 for r in t['rows']]
            row={'case':p.stem,'method':t['method'],'repeat':t['repetition'],'goal':t['goal'],'collision':t['collision'],'planner_failure':t['planner_failure'],'queries':t['planning_calls'],'planning_s':t['planning_time_s'],'p95_ms':float(np.percentile(times,95)) if times else None,'max_ms':max(times) if times else None,'arrival_sim_s':t['goal_sim_time_s']};live_rows.append(row)
            lines.append(f"| {p.stem} | {t['method']} / {t['repetition']} | {t['goal']} | {t['collision']} | {t['planner_failure']} | {t['planning_calls']} | {t['planning_time_s']:.3f} | {row['p95_ms'] if row['p95_ms'] is not None else 'NA'} | {row['max_ms'] if row['max_ms'] is not None else 'NA'} |")
    (HERE/'analysis/carla_summary.json').write_text(json.dumps({'trials':live_rows,'executions':executions},indent=2),encoding='utf8')
    lines+=['','仿真同步 20 Hz 不等于墙钟严格实时；应同时查看尾延迟和超过 50 ms 的次数。未到达、初始化失败、碰撞及规划失败均保留，不因不利结果从统计中删除。高层证书不构成连续轨迹碰撞概率或完整滚动 LTL 满足性的证明。','',
    '## 7 数值核验与修复','',
    f"正式合成/敏感性批次已完成 {audit['completed_runs']} 组、{audit['query_records']} 条查询记录；审计异常 {len(audit['violations'])} 条，缺少配对 LP 的运行 {len(audit['unpaired_runs'])} 组。详见 analysis/audit.json。",
    f"固定序列回放完成 {replay_audit.get('completed_runs',0)} 组、{replay_audit.get('query_records',0)} 条记录；审计异常 {len(replay_audit.get('violations',[]))} 条。",
    '每次返回检查风险、目标差、动作概率合法性和独立策略评价；与相同参数 LP 的目标进行上下界交叉核查。预设的验证阈值未因失败记录而放宽。',
    '诊断发现 LP 可行性容差允许微小负占据量。直接归一化可将它们在近零访问状态放大为负动作概率。非负截断后再归一化解决了该问题，并增加回归测试。默认容差试验与修复前计时全部排除于正式表格；numerical_pilot.json 保留问题证据。',
    '已完成 48 个所有消融分支的独立 LP/策略核验，以及原实现 30 个标量化上界、50 个缓存凸包 LP 和 40 个方向迁移上界检查。方向迁移功能不被归因为本批未启用的主实验收益。','',
    '## 8 复现','',
    '建议 Python 3.11，依赖版本见 requirements-run.txt；CARLA 服务端与客户端为 0.9.16。','',
    '```bash','python check_probability_recovery.py','python check_controlled.py','python run_suite.py','python analyze_results.py','python run_replay_suite.py','python analyze_results.py --results replay_results --out replay_analysis --expected 36',
    '# 先单独启动 CARLA 服务端，关闭其他计时进程，再执行：','python run_carla_suite.py','python write_report.py','```','',
    '代码从仓库相对目录导入实现；原始记录的 command 字段含本机路径，可按当前仓库位置调整。所有 CSV、JSON、图与报告均应从同一批完成的结果生成。','',
    '## 9 仍需区分的研究问题','',
    '这些补实验增加了成本归因、重复统计和适用边界证据，不能自动证明方法新颖性，也不能替代与参数 LP、策略列生成和 CMDP 复用工作的系统比较。风险映射仍为研究者设定，闭环低层精化和长期缓存容量控制仍是限制。','']
    (HERE/'README_CN.md').write_text('\n'.join(lines),encoding='utf8')
    print('Report generated from',len(data),'synthetic/sensitivity runs,',len(replays),'replays,',len(live_rows),'live trials')
if __name__=='__main__':main()
