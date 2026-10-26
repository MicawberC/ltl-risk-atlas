from pathlib import Path
import json,collections,statistics as st
HERE=Path(__file__).resolve().parent
def main():
    audit=json.loads((HERE/'analysis/audit.json').read_text('utf8'))
    wins=json.loads((HERE/'analysis/wins.json').read_text('utf8'))
    summaries=json.loads((HERE/'analysis/summary.json').read_text('utf8'))
    data=[d for p in (HERE/'results').glob('*.json') if (d:=json.loads(p.read_text('utf8'))).get('run_complete')]
    lines=['# 缓存对偶定价与策略迭代热启动：优化复测报告','',
    '本批改进保留 2026-10-23 代码及全部旧结果。旧批次的 85/93 更快属于旧计时记录；本报告使用当天重新运行的相同保基 LP，分别报告既有输入复测和开发前固定的新输入。图片入口及图注见 [FIGURES_CN.md](FIGURES_CN.md)。','',
    '不同日期的机器负载、温度和计时波动可能改变旧版胜负，不能直接把两批胜率之差全归因于代码修改。同日另运行旧版 full 对照，覆盖主实验、旧套件的不占优输入，以及全部新风险序列；优化效果应同时查看这些同日比较。','',
    '## 改了什么','',
    '压力例剖析显示主要时间花在完整 CMDP 的风险价格搜索和重复普通 MDP 求解。新版从缓存最优占据分布组合的奖励—风险斜率推算候选价格，在当前风险下求解一次普通 MDP，形成有效全局上界并补充策略；仍不够好时继续定价或进入原有验证后的完整扩充。',
    '候选价格只是求解建议，返回仍需通过原来的硬风险与目标差证书。旧策略的动作还可用作策略迭代起点；该收益通过 priced（关闭这个热启动）对照单独观察。',
    '这些修改采用已有的对偶定价、列生成及策略热启动思想，不能仅因本次更快就宣称全新理论。优化实现目前位于统一受控驱动；未替换历史 CARLA 闭环控制器。','',
    '## 比较口径','',
    '- LP 保留模型与 simplex basis，HiGHS 单线程、原始/对偶可行性容差 1e-9；其他线性代数库也单线程。',
    '- 各方法顺序执行、各次独立进程、顺序固定种子打乱；包含自身初始化和策略恢复，排除公共模型构建、导入与独立策略评价。',
    '- 目标差门槛仍为 max(1e-4, 1%×|返回目标值|)，并保留数值容差；并非近零目标值下的纯相对 1%。',
    '- 开发只用两个已知压力序列。新输入为五个预先指定种子 × 三种配置，每个重复三次；不用于调参。各项计时重复不等于独立道路。','',
    '## 配对时间结果','',
    '| 批次 | 配对数 | 新版更快 | 新版更慢 | 无效配对 | LP/新版配对比中位数 [最小, 最大] |','|---|---:|---:|---:|---:|---:|']
    for x in wins:
        ratio='NA' if x['median_ratio'] is None else f"{x['median_ratio']:.2f} [{x['min_ratio']:.2f}, {x['max_ratio']:.2f}]"
        name={'core':'主实验','historical':'旧十二条输入','sensitivity':'敏感性','replay':'固定观测回放','heldout':'留出新风险序列'}[x['group']]
        lines.append(f"| {name} | {x['pairs']} | {x['faster']} | {x['slower']} | {x['invalid']} | {ratio} |")
    lines+=['','配对时间比先对每次相同输入求 LP 时间/新版时间，再取中位数；它不一定等于两种方法各自时间中位数之商。范围为观测最小—最大值，不是置信区间。']
    existing=[x for x in wins if x['group']!='heldout'];held=[x for x in wins if x['group']=='heldout'][0]
    lines+=['',f"既有输入合计 {sum(x['pairs'] for x in existing)} 次配对，其中新版更快 {sum(x['faster'] for x in existing)} 次、较慢 {sum(x['slower'] for x in existing)} 次；新输入 {held['pairs']} 次配对，更快 {held['faster']} 次、较慢 {held['slower']} 次。它们是固定输入上的计时观察，不能作为一般场景成功概率。",'',
    '## 分配置时间和求解次数','',
    '| 批次 / 配置 | 方法 | 运行数 | 累计时间中位数 [最小, 最大] / 秒 | 在线 MDP 调用总数 | 完整扩充/LP 调用总数 |','|---|---|---:|---:|---:|---:|']
    for x in sorted(summaries,key=lambda x:(str(x['key']),x['method'])):
        label=' / '.join(str(v) for v in x['key'])
        lines.append(f"| {label} | {x['method']} | {x['runs']} | {x['median_s']:.3f} [{x['min_s']:.3f}, {x['max_s']:.3f}] | {x['scalarized_calls']} | {x['full_solves']} |")
    lines+=['','MDP 调用列只计在线阶段；累计时间包含各方法自身初始化。完整扩充计数指进入旧完整价格搜索程序，并非宣称新版不求解约束问题；新版定价轮次中的普通 MDP 均计入调用数。LP 行的完整调用计数指 LP 查询。尾延迟、采样 RSS 和每次运行的完整指标见 analysis/run_summary.csv，缓存规模及数据量保存在原始结果。','',
    '![压力例定价与热启动消融](analysis/pressure_ablation.png)','','上图只包含两个开发输入的三次重复，不能当作新序列上的独立消融结论。','',
    '![全部配对时间比](analysis/paired_speedups.png)','','![预先固定的新输入](analysis/heldout_configs.png)','','![在线求解工作量](analysis/oracle_work.png)','','![长序列累计时间](analysis/long_sequence.png)','','## 如何解释','',
    '主要收益候选是未直接复用成功时减少完整价格搜索；不能将普通 MDP 调用减少等同于零求解复用。比较 full、priced 和 optimized 可以区分原完整实现、缓存定价，以及额外策略热启动的影响。若热启动的计时改进不稳定，不能声称其总是有益。',
    '胜负以所有完成且独立核验有效的同输入累计时间比较。失败或未完成配对不计为胜利。所有数据、P95、最大延迟、缓存和内存记录均保留，不能只看胜率。',
    '即使本批全部获胜，也只说明在这些输入、机器和软件版本上获胜。局部更新次数不足、模型改变、缓存维护变贵、风险问题不可行或基线实现更高效，都可能改变结果。没有任意输入下必定快于 LP 的定理。',
    '本批合成配置规模为 16×16 至 32×32，旧作者公开的 4×4 小例子不在这 93 次比较中，未据本批结果宣称该小例子也已提速。新输入是新的风险更新序列，使用的三种模型配置仍与已说明的模型族相同。',
    '本次不运行新 CARLA 驾驶；固定观测回放的计时不能改变旧批次 24 次驾驶中 12 次未到达的结论，也不提供物理碰撞概率或严格实时保证。','',
    '## 独立核验','',f"完成运行 {audit['runs']}/{audit['expected_runs']}，查询记录 {audit['query_records']}；审计异常 {len(audit['violations'])} 条，未完成文件 {len(audit['incomplete'])} 个。详见 analysis/audit.json。",
    '逐查询核验动作概率、独立策略目标与风险、硬预算、目标差及相同输入下的 LP 下上界关系；开发先导另外通过 60 次独立 LP/策略检查。',
    '新版有时会比旧完整搜索更早在允许的 1%/绝对目标差门槛内停止，两者承诺的门槛相同，但不能因此宣称返回值精度逐点相等。实际相对 LP 的误差与预算使用比例另见 analysis/quality.json。', '',
    '## 复现','',
    '依赖与原批次相同，Python 3.11，见 ../experiments_20261023/requirements-run.txt。',
    '```bash','python check_pilots.py','python run_suite.py','python analyze.py','python plot_results.py','python write_report.py','```','',
    '模型与风险序列从旧脚本导入，旧脚本本身不修改。run_manifest.json 和每份结果保存优化源文件 SHA-256。新输入列表与比较规则见 PROTOCOL_CN.md。','']
    optimized={d['pair_id']:d for d in data if d['method']=='optimized'}
    lp_reference={d['pair_id']:d for d in data if d['method']=='warm_lp'}
    same_day=[d for d in data if d['method']=='full']
    ratios=[d['total_seconds']/optimized[d['pair_id']]['total_seconds'] for d in same_day]
    pressure=[d for d in data if d['group']=='core' and d['case']=='16:barrier_hard_active:0']
    calls=lambda m:sum(r.get('scalarized_calls',0) for d in pressure if d['method']==m for r in d['rows'])
    failures=lambda m:sum(d['full_solves'] for d in pressure if d['method']==m)
    development=[d for d in data if d['group']=='historical' and d.get('seed')==73047]
    times={m:st.median(d['total_seconds'] for d in development if d['method']==m) for m in ('full','priced','optimized','warm_lp')}
    quality=json.loads((HERE/'analysis/quality.json').read_text('utf8'))
    budget=max(q['max_budget_used'] for q in quality if q['method']=='optimized')
    interpretation=[
        f"同日旧版对照共 {len(same_day)} 次，新版较快 {sum(r>1 for r in ratios)} 次；旧版/新版配对时间比中位数 {st.median(ratios):.2f}，范围 [{min(ratios):.2f}, {max(ratios):.2f}]。这比直接比较不同日期的 85/93 与本批胜率更能支持代码改进效果。",'',
        f"在这同一批 {len(same_day)} 次输入与重复计时上，旧版相对 LP 为 {sum(d['total_seconds']<lp_reference[d['pair_id']]['total_seconds'] for d in same_day)} 次更快、{sum(d['total_seconds']>lp_reference[d['pair_id']]['total_seconds'] for d in same_day)} 次更慢；新版相对 LP 为 {sum(optimized[d['pair_id']]['total_seconds']<lp_reference[d['pair_id']]['total_seconds'] for d in same_day)} 次更快。这是已说明的子集，不代表旧版全部 93 次输入的同日胜率。",'',
        f"主实验压力例的在线普通 MDP 调用累计由 {calls('full')} 次降至 {calls('optimized')} 次，进入旧完整搜索程序由 {failures('full')} 次降至 {failures('optimized')} 次。收益主要表现为未命中后更有针对性地补算，不是将所有查询变成零求解缓存命中。",'',
        f"旧十二条中原不占优的开发输入（种子 73047），同日三次计时中位数：旧完整实现 {times['full']:.3f} 秒，保基 LP {times['warm_lp']:.3f} 秒，仅缓存定价 {times['priced']:.3f} 秒，加动作热启动 {times['optimized']:.3f} 秒。该例首先由定价减少搜索成本，动作热启动再提供额外收益；这不证明每种输入中热启动都同样有效。",'',
        f"新版实际目标差最大占用了约 {100*budget:.1f}% 的允许误差预算，未突破原门槛；它不是声称所有目标值的纯相对误差都等于该百分比。",'']
    position=lines.index('## 如何解释')+2;lines[position:position]=interpretation
    if (HERE/'diagnostics/interim_audit_overlap/EXCLUDED.json').exists():
        position=lines.index('## 独立核验')+2
        lines[position:position]=['一对原始回放计时与中途 JSON 汇总审计重叠，因此两个方法均在所有其他计时与绘图结束后整对重跑，保持相同输入、代码与门槛。原两份记录保留在 diagnostics/interim_audit_overlap，正式表使用重跑结果而非挑选最好时间；最终保留 357 组，另有这两份被替代的尝试记录。','']
    (HERE/'README_CN.md').write_text('\n'.join(lines),encoding='utf8');print('Report written')
if __name__=='__main__':main()
