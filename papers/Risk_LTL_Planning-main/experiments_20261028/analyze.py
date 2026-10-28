"""Rebuild tables, figures and a Chinese report from retained experiment files."""
from pathlib import Path
import csv,json,math,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
METHODS=['warm_lp','full','optimized']
NAMES={'warm_lp':'Warm LP','full':'Basic atlas','optimized':'Optimized atlas'}
OUTCOMES=['goal','timeout','planner_failure','collision']

def load(path):return json.loads(path.read_text('utf8'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def median(values):return float(np.median(values)) if values else None
def fmt(x):return '—' if x is None else f'{x:.3f}'
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        columns=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
def savefig(fig,name):
    for suffix in ['png','pdf','eps']:fig.savefig(HERE/'figures'/f'{name}.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    (HERE/'figures').mkdir(exist_ok=True)
    status=load(HERE/'suite_status.json');manifest=load(HERE/'run_manifest.json')
    recovery=load(HERE/'recovery_status.json') if (HERE/'recovery_status.json').exists() else {'recovered':[]}
    trials=[load(p) for p in sorted((HERE/'results').glob('*.json'))]
    if len(trials)!=36 or not all(t.get('trial_complete') for t in trials):
        raise RuntimeError('Report requires all 36 completed drives, including unsuccessful driving outcomes')
    audit=load(HERE/'validation/audit.json')
    if audit['trial_files']!=36 or audit['limited_run']:raise RuntimeError('Incomplete audit scope')
    replay_status=load(HERE/'replays/status.json')
    replay_runs=[load(p) for p in (HERE/'replays').glob('*_replay*.json')]
    if len(replay_runs)!=72:raise RuntimeError('Missing prespecified replay runs')
    summaries=[];individual=[]
    for t in trials:
        ms=[r['plan_s']*1000 for r in t['rows']]
        individual.append({'case':t['case'],'method':t['method'],'seed':t['seed'],'outcome':t['outcome'],
                           'queries':len(ms),'returned_queries':t['returned_queries'],'planning_time_s':t['planning_time_s'],
                           'p50_ms':float(np.percentile(ms,50)),'p95_ms':float(np.percentile(ms,95)),
                           'maximum_ms':max(ms),'over_50ms':sum(x>50 for x in ms),
                           'goal_sim_time_s':t.get('goal_sim_time_s'),'peak_rss_mb':t['peak_process_rss_mb']})
    for method in METHODS:
        ts=[t for t in trials if t['method']==method];rows=[r for t in ts for r in t['rows']]
        times=[r['plan_s']*1000 for r in rows]
        summaries.append({'method':method,**{o:sum(t['outcome']==o for t in ts) for o in OUTCOMES},
                          'queries':len(rows),'returned_queries':sum('policy_key' in r for r in rows),
                          'cumulative_plan_s':sum(t['planning_time_s'] for t in ts),
                          'drive_time_median_s':median([t['planning_time_s'] for t in ts]),
                          'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),
                          'max_ms':max(times),'over_50ms':sum(t>50 for t in times),
                          'scalarized_calls':sum(r.get('scalarized_calls',0) for r in rows),
                          'full_solves':sum(r.get('full_solve',False) for r in rows),
                          'max_cache':max(r.get('cached_policies',0) for r in rows)})
    groups={};pairs=[];replay_checks=[]
    for r in replay_runs:groups.setdefault((r['trace'],r['repetition']),{})[r['method']]=r
    for (trace,repetition),g in sorted(groups.items()):
        a,b=g.get('optimized'),g.get('warm_lp')
        if not a or not b:raise RuntimeError(f'Missing paired replay {trace} {repetition}')
        if not a['complete'] or not b['complete']:
            pairs.append({'trace':trace,'repetition':repetition,'valid':False,'speedup':None});continue
        if a['trace_sha256']!=b['trace_sha256']:raise RuntimeError('Unpaired inputs')
        errors=[]
        if len(a['rows'])!=len(b['rows']):errors.append('Different query counts')
        for x,y in zip(a['rows'],b['rows']):
            if x['tick']!=y['tick']:errors.append('Different ticks')
            if not (x['objective']-1e-5<=y['objective']<=x['objective']+x['gap']+1e-5):errors.append(f"LP bracket at tick {x['tick']}")
            if max(x['risk'],y['risk'])>2+1e-7:errors.append(f"Risk gate at tick {x['tick']}")
        replay_checks.append({'trace':trace,'repetition':repetition,'queries':len(a['rows']),'errors':errors})
        pairs.append({'trace':trace,'case':a['case'],'source_method':a['source_method'],'source_outcome':a['source_outcome'],
                      'repetition':repetition,'queries':a['expected_queries'],'valid':not errors,
                      'optimized_s':a['planning_time_s'],'warm_lp_s':b['planning_time_s'],
                      'speedup':b['planning_time_s']/a['planning_time_s']})
    ratios=[p['speedup'] for p in pairs if p['valid']]
    extra_runs=[load(p) for p in (HERE/'failure_inclusive').glob('*_replay*.json')]
    if len(extra_runs)!=18 or not all(r['complete'] for r in extra_runs):raise RuntimeError('Incomplete failure-inclusive supplement')
    extra_groups={};extra_pairs=[]
    for r in extra_runs:extra_groups.setdefault((r['trace'],r['repetition']),{})[r['method']]=r
    for (trace,rep),g in sorted(extra_groups.items()):
        a,b=g['optimized'],g['warm_lp']
        if a['trace_sha256']!=b['trace_sha256'] or len(a['rows'])!=len(b['rows']):raise RuntimeError('Supplement inputs differ')
        for x,y in zip(a['rows'],b['rows']):
            if x['tick']!=y['tick'] or x['returned']!=y['returned']:raise RuntimeError('Supplement return status differs')
            if x['returned'] and not x['objective']-1e-5<=y['objective']<=x['objective']+x['gap']+1e-5:raise RuntimeError('Supplement LP bracket failed')
        extra_pairs.append({'trace':trace,'repetition':rep,'optimized_s':a['planning_time_s'],'warm_lp_s':b['planning_time_s'],
                            'optimized_failed_query_s':a['failed_query_time_s'],'warm_lp_failed_query_s':b['failed_query_time_s'],
                            'speedup':b['planning_time_s']/a['planning_time_s']})
    extra_ratios=[p['speedup'] for p in extra_pairs]
    failures=[]
    for p in sorted((HERE/'validation').glob('*.json')):
        if p.name=='audit.json':continue
        v=load(p)
        for row in v.get('rows',[]):
            if not row['returned']:failures.append({'trial':p.name,'tick':row['tick'],'diagnosis':row.get('fresh_lp')})
    aggregate={'live':summaries,'replay_pairs':len(pairs),'replay_valid_pairs':len(ratios),
               'replay_faster':sum(r>1 for r in ratios),'replay_slower_or_equal':sum(r<=1 for r in ratios),
               'replay_median_speedup':median(ratios),'replay_min_speedup':min(ratios) if ratios else None,
               'replay_max_speedup':max(ratios) if ratios else None,
               'drive_trial_count':len(trials),'policy_audit':{k:v for k,v in audit.items() if k!='trials'},
               'failure_diagnostics':failures,'infrastructure_failure_count':len(status['infrastructure_failures']),
               'fresh_lp_infeasible_count':sum(f.get('diagnosis',{}).get('status')=='Infeasible' for f in failures),
               'explicitly_recovered_drives':len(recovery['recovered'])}
    aggregate['failure_inclusive_supplement']={'pairs':len(extra_pairs),'faster':sum(r>1 for r in extra_ratios),
        'median_speedup':median(extra_ratios),'minimum_speedup':min(extra_ratios),'maximum_speedup':max(extra_ratios),
        'scope':'Post-hoc, all three failed-driving r0 traces, including final failure costs'}
    dump(HERE/'summary.json',aggregate);dump(HERE/'replay_checks.json',replay_checks)
    write_csv(HERE/'driving_summary.csv',individual);write_csv(HERE/'replay_pairs.csv',pairs)
    write_csv(HERE/'failure_inclusive_pairs.csv',extra_pairs)
    fig,ax=plt.subplots(figsize=(5.6,3.1));bottom=np.zeros(3)
    for outcome,color,label in zip(OUTCOMES,['#4682b4','#c5a04d','#c76c65','#833c69'],['Reached goal','Time limit','Planner failure','Collision']):
        values=np.array([s[outcome] for s in summaries]);ax.bar([NAMES[m] for m in METHODS],values,bottom=bottom,label=label,color=color);bottom+=values
    ax.set_ylabel('Drives (12 per method)');ax.set_ylim(0,15);ax.legend(ncol=2,fontsize=8,loc='upper center',frameon=False)
    fig.tight_layout();savefig(fig,'closed_loop_outcomes')
    fig,ax=plt.subplots(figsize=(5.6,3.2));cases=list(dict.fromkeys(j['case'] for j in manifest['jobs']))
    for i,case in enumerate(cases):
        values=[p['speedup'] for p in pairs if p.get('case')==case and p['valid']]
        ax.scatter(i+np.linspace(-.18,.18,len(values)),values,s=21,color='#356b9e')
        if values:ax.plot([i-.22,i+.22],[median(values)]*2,color='black',lw=2)
    ax.axhline(1,color='gray',ls='--');ax.set_yscale('log');ax.set_ylabel('Warm LP / optimized time')
    ax.set_xticks(range(len(cases)),['T01 route0\n1 walker','T01 route0\n2 walkers','T01 route1\n1 walker','T02 route0\n1 walker']);ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout();savefig(fig,'closed_loop_replay_speedups')
    fig,axes=plt.subplots(2,2,figsize=(7,4.4),sharey=True)
    for ax,case in zip(axes.flat,cases):
        for method in METHODS:
            t=next(t for t in trials if t['case']==case and t['method']==method and t['repetition']==0)
            ax.plot([r['sim_time_s'] for r in t['rows']],[r['longitudinal_m'] for r in t['rows']],label=NAMES[method],lw=1.2)
        ax.axhline(7.5,color='gray',ls='--',lw=.7);ax.set_title(case,fontsize=8);ax.set_xlabel('Simulation time (s)',fontsize=8)
    axes[0,0].set_ylabel('Position along road (m)');axes[1,0].set_ylabel('Position along road (m)')
    axes[0,0].legend(fontsize=6,frameon=False);fig.tight_layout();savefig(fig,'closed_loop_paths_r0')
    lines=['# 优化算法接入 CARLA 闭环驾驶：结果与边界','',
           '本批实际启动CARLA 0.9.16，实时读取行人与车辆位置，更新风险和起点，规划后向车辆发送油门/制动，再用车辆的新位置继续规划。它是仿真闭环驾驶，而不是只对旧观测文件做回放。低层仍为直路前进/等待控制。','',
           '## 实验设计','',
           'Town01的路线0单行人/双行人、路线1单行人，以及Town02路线0单行人，共4个场景。每场景使用3个固定动作种子，对比优化图谱、基础图谱和保基LP，共36次驾驶。算法沿用2026-10-26版本，风险预算和目标差门槛未放宽。每次驾驶启动全新服务器和客户进程。方法执行顺序预先打乱，完整协议见[PROTOCOL_CN.md](PROTOCOL_CN.md)。','',
           '所有规划调用（含失败调用）均计时，初始化计入首次查询；公共建模、仿真推进、策略存盘、独立核验不计入规划耗时。以下驾驶耗时来自不同闭环轨迹，不能直接解释为同输入的求解器加速比。','',
           '## 驾驶结果','',
           '| 方法 | 到达终点 | 到时未到达 | 规划失败 | 碰撞 | 查询数（含失败） | 单次驾驶累计规划耗时中位数/s |',
           '|---|---:|---:|---:|---:|---:|---:|']
    labels={'warm_lp':'保基LP','full':'基础图谱','optimized':'优化图谱'}
    for s in summaries:lines.append(f"| {labels[s['method']]} | {s['goal']} | {s['timeout']} | {s['planner_failure']} | {s['collision']} | {s['queries']} | {s['drive_time_median_s']:.3f} |")
    lines+=['','各场景分别看：','','| 场景 | 保基LP | 基础图谱 | 优化图谱 |','|---|---|---|---|']
    outcome_labels={'goal':'到达','timeout':'超时','planner_failure':'规划失败','collision':'碰撞'}
    for case in cases:
        cells=[]
        for method in METHODS:
            group=[t for t in trials if t['case']==case and t['method']==method]
            cells.append('，'.join(f'{count}次{outcome_labels[o]}' for o in OUTCOMES if (count:=sum(t['outcome']==o for t in group))))
        lines.append('| '+case+' | '+' | '.join(cells)+' |')
    if len({tuple(s[o] for o in OUTCOMES) for s in summaries})==1:
        s=summaries[0]
        lines+=['',f"三种方法的驾驶结局计数相同：各{s['goal']}/12次到达、{s['timeout']}次超时、{s['planner_failure']}次规划失败、{s['collision']}次记录到碰撞。因此，本批没有观察到优化方法提高驾驶成功率。它的作用应定位为减少规划计算，而非解决既有控制器的全部驾驶问题。"]
    lines+=['','![驾驶结局](figures/closed_loop_outcomes.png)','',
            '到达终点、数值证书通过和计算更快是三个不同指标。规划失败表示该时刻未返回合格策略；实验立即停止正常驾驶，尚未实现经过验证的紧急接管。到时未到达也保留在分母内。三次种子重复不足以估计实际道路事故概率。','',
            '![预先指定的第0次重复轨迹](figures/closed_loop_paths_r0.png)','',
            '上图固定展示每场景第0次重复，包含所有三种方法，未按是否到达选择轨迹。横轴是仿真时间；曲线提前结束可能是到达、规划失败或碰撞，具体结局见[逐次驾驶表](driving_summary.csv)。','',
            '## 相同输入下是否更快','',
            '为排除驾驶路径不同带来的影响，预先指定每场景、每种驾驶方法第0次重复的输入，共12条序列。所有已返回查询的前缀都保留，原驾驶失败也不删除。关闭模拟器后，优化方法与保基LP依次处理相同风险和起点，各计时3次，共72次运行、36次配对。失败前缀上的计算优势不能替代完整驾驶成功。','',
            f"有效配对{len(ratios)}/{len(pairs)}次，其中优化方法更快{sum(r>1 for r in ratios)}次，不快于LP {sum(r<=1 for r in ratios)}次。LP/优化耗时比中位数为{fmt(median(ratios))}，范围为{fmt(min(ratios) if ratios else None)}–{fmt(max(ratios) if ratios else None)}。这是本批固定输入上的测量结果，重复计时不是新增独立场景。",'',
            '![同输入规划时间比](figures/closed_loop_replay_speedups.png)','',
            '图中每点为一次配对，比值大于1表示优化方法更快，黑线为中位数。详见[配对数据](replay_pairs.csv)。','',
            f"补充分析：发现最大延迟出现在未返回查询后，对上述12条输入中全部3条失败驾驶序列，再计入最终失败查询的耗时，各重复3次，共9次配对。优化方法更快{sum(r>1 for r in extra_ratios)}/9次，耗时比中位数为{fmt(median(extra_ratios))}，范围{fmt(min(extra_ratios))}–{fmt(max(extra_ratios))}。这项分析在查看闭环延迟后追加，属于事后补充，不与预定36次配对混合。详见[含失败查询的配对表](failure_inclusive_pairs.csv)。",'',
            '## 延迟与数值核验','',
            '| 方法 | 查询P50/ms | P95/ms | 最大/ms | 超过50ms次数 |',
            '|---|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {labels[s['method']]} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['max_ms']:.3f} | {s['over_50ms']} |")
    opt=next(s for s in summaries if s['method']=='optimized')
    lines+=['',f"优化方法仍有{opt['over_50ms']}/{opt['queries']}次查询超过50ms，最长约{opt['max_ms']/1000:.3f}秒，发生在未返回策略的失败查询。因此，累计计算更快不代表每次查询都更快，也不代表已具备严格实时保证。"]
    lines+=['',f"已独立读取并核验{audit['checked_returned_policies']}个实际返回的完整策略，另诊断{audit['nonreturning_failed_queries']}个未返回查询；核验异常数为{audit['violation_count']}。检查包括策略文件哈希、动作概率、实际采样动作、占据分布流平衡、风险预算、目标值重算和独立LP上下界。完整结果见[核验汇总](validation/audit.json)，失败点的新建LP诊断见[summary.json](summary.json)。",'',
            '这些是浮点数值证据，不是精确算术证明。风险上限约束的是离散模型中的期望折扣风险，并不直接等于真实碰撞概率。固定0.05秒仿真步长也不代表墙钟下已达到严格20Hz实时控制。','',
            f"{len(failures)}个未返回查询中，全新建立的LP有{aggregate['fresh_lp_infeasible_count']}个得到Infeasible状态。这表明这些失败不能简单归因于策略复用太慢；起点随车辆移动后，原风险预算可能已无法满足。当前执行层尚未保证下一时刻仍有可行方案。失败后停止仿真并不等于实现了安全停车，因此零记录碰撞也不能解释为完整驾驶安全保证。",'',
            '## 可以写入论文的结论与限制','',
            '本批将优化算法接入了实际运行的CARLA闭环，并通过保存完整策略及实际采样动作，使模型证书与执行接口可以事后核对。计算效率应依据同输入回放的配对结果；驾驶能力应依据全部36次的到达、超时、失败和碰撞统计，不能互相替代。','',
            '当前仍是固定车道、直路前进/等待和预设行人条件，风险输入使用CARLA对象的真值坐标，不是摄像头感知系统。沿用的起点映射每次将任务历史位归零，因此不能宣称已完成完整LTL历史监控；风险映射与连续控制之间也尚无校准的安全保证。下一步应完善任务状态跟踪、规划不可行时的安全接管、更多独立道路与行人行为、实时延迟控制，以及长时间策略库增长测试。','',
            f"基础设施异常共记录{len(status['infrastructure_failures'])}次尝试，保存在diagnostics；不会将其当作策略成功或隐去。开发期一次地图重载后发生CARLA访问冲突，正式协议因而采用每次新进程。正式运行未因驾驶结局重跑。",'',
            f"另有{len(recovery['recovered'])}次驾驶通过公开修订的基础设施恢复阶段补齐。原失败尝试和补测清单都保留，详见[协议修订说明](PROTOCOL_AMENDMENTS_CN.md)。原矩阵的重试上限曾增加，不能表述为全程没有协议调整；冻结的规划算法、风险门槛和场景种子未变。",'',
            '## 复现与文件','',
            '运行环境与冻结源文件哈希见[run_manifest.json](run_manifest.json)，逐次驾驶在results，完整策略NPY压缩包在policies，服务器与客户日志在server_logs和logs。核验与绘图都在计时之后运行。','',
            '核验本批存档并重新生成图表：','','```bash','python verify_artifacts.py','python analyze.py','```','',
            '重新驾驶时请指定新的输出目录，避免覆盖本批记录；示例：','','```bash',
            'python run_suite.py --server /path/to/CarlaUE4.exe --out ./new_run --infra-retries 1',
            'python validate_results.py --results-dir ./new_run/results --out ./new_run/validation',
            'python replay.py --results-dir ./new_run/results --out ./new_run/replays','```','',
            '本批事后补充计时入口是replay_failure_inclusive.py，默认读取本批results并重写failure_inclusive，请在独立仓库副本中重跑。analyze.py默认汇总本批保留的数据。若新驾驶出现基础设施缺测，先按协议修订说明运行recover_missing.py并指定相同的--out及--server；完整驾驶失败不可重试。','',
            '请在完整仓库中、本实验目录下执行，安装requirements.txt所列依赖；Linux服务器路径可替换。服务器启动和地图加载不计入规划耗时。EPS图片可用于现有CCC的LaTeX编译链。文件校验清单为SHA256SUMS.txt。']
    (HERE/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
