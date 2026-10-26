"""Figures of all observed runs, without confidence-interval claims."""
from pathlib import Path
import json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'analysis'
COLORS={'warm_lp':'#888888','full':'#b17a43','priced':'#527fa6','optimized':'#18764b'}
LABELS={'warm_lp':'Warm LP','full':'Previous full','priced':'Cached pricing','optimized':'Pricing + warm policy'}
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):
        path=OUT/f'{name}.{ext}';fig.savefig(path,dpi=220,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text('utf8').splitlines())+'\n',encoding='utf8')
    plt.close(fig)
def bars(ax,series,methods):
    for j,m in enumerate(methods):
        values=series.get(m,[])
        if not values:continue
        median=np.median(values);ax.bar(j,median,color=COLORS[m]);ax.errorbar(j,median,yerr=[[median-min(values)],[max(values)-median]],fmt='none',color='black',capsize=3)
    ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=25,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
def main():
    OUT.mkdir(exist_ok=True)
    ds=[d for p in (HERE/'results').glob('*.json') if (d:=json.loads(p.read_text('utf8'))).get('run_complete')]
    refs={d['pair_id']:d for d in ds if d['method']=='warm_lp'}
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,group,seed in zip(axes,['core','historical'],[73044,73047]):
        subset=[d for d in ds if d['group']==group and d.get('seed')==seed and d['case']=='16:barrier_hard_active:0']
        series={m:[d['total_seconds'] for d in subset if d['method']==m] for m in LABELS}
        bars(ax,series,list(LABELS));ax.set_ylabel('Setup + 20 queries (s)');ax.set_title(f'Development input seed {seed}',fontsize=10)
    save(fig,'pressure_ablation')
    groups=['core','historical','sensitivity','replay','heldout'];fig,ax=plt.subplots(figsize=(8,4))
    for j,group in enumerate(groups):
        ratios=[refs[d['pair_id']]['total_seconds']/d['total_seconds'] for d in ds if d['method']=='optimized' and d['group']==group and d['pair_id'] in refs]
        if not ratios:continue
        x=j+np.linspace(-.17,.17,len(ratios));ax.scatter(x,ratios,s=12,alpha=.55,color=['#18764b' if r>1 else '#b84444' for r in ratios])
        ax.plot([j-.22,j+.22],[np.median(ratios)]*2,color='black',lw=2)
    ax.axhline(1,ls='--',color='#888888');ax.set_yscale('log');ax.set_xticks(range(5),['Core','Twelve inputs','Sensitivity','Trace replay','New inputs']);ax.set_ylabel('Warm LP / optimized planning time');ax.set_title('Every paired timing; black marks are medians');ax.grid(axis='y',alpha=.2);save(fig,'paired_speedups')
    cases=list(dict.fromkeys(d['case'] for d in ds if d['group']=='heldout'));fig,axes=plt.subplots(1,len(cases),figsize=(4.5*len(cases),4),squeeze=False)
    for ax,case in zip(axes[0],cases):
        series={m:[d['total_seconds'] for d in ds if d['group']=='heldout' and d['case']==case and d['method']==m] for m in ('warm_lp','full','optimized')}
        bars(ax,series,list(series));ax.set_ylabel('Setup + 20 queries (s)');ax.set_title(case,fontsize=9)
    fig.suptitle('New inputs: 5 seeds x 3 timing repeats; median and min-max',fontsize=11);save(fig,'heldout_configs')
    core_cases=list(dict.fromkeys(d['case'] for d in ds if d['group']=='core'));fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,method in enumerate(('full','optimized')):
        calls=[];complete=[]
        for case in core_cases:
            ss=[d for d in ds if d['group']=='core' and d['case']==case and d['method']==method];count=sum(d['count'] for d in ss)
            calls.append(sum(r.get('scalarized_calls',0) for d in ss for r in d['rows'])/count)
            complete.append(100*sum(d['full_solves'] for d in ss)/count)
        x=np.arange(len(core_cases))+(j-.5)*.32
        axes[0].bar(x,calls,width=.32,color=COLORS[method],label=LABELS[method]);axes[1].bar(x,complete,width=.32,color=COLORS[method],label=LABELS[method])
    for ax in axes:
        ax.set_xticks(range(len(core_cases)),core_cases,rotation=20,ha='right',fontsize=8);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Online scalarized MDP calls / query');axes[1].set_ylabel('Queries with complete expansion (%)');save(fig,'oracle_work')
    long=[d for d in ds if d['group']=='sensitivity' and d['count']==160];fig,ax=plt.subplots(figsize=(6,4))
    for m in ('warm_lp','optimized'):
        values=np.asarray([d['setup_seconds']+np.cumsum([r['seconds'] for r in d['rows']]) for d in long if d['method']==m])
        if not len(values):continue
        x=np.arange(1,values.shape[1]+1);ax.plot(x,np.median(values,axis=0),label=LABELS[m],color=COLORS[m]);ax.fill_between(x,values.min(axis=0),values.max(axis=0),alpha=.15,color=COLORS[m])
    ax.set(xlabel='Updates',ylabel='Cumulative planning time (s)',title='160-query sequence, own initialization included');ax.legend();ax.grid(alpha=.2);save(fig,'long_sequence')
    print('Generated five PNG/SVG figure pairs')
if __name__=='__main__':main()
