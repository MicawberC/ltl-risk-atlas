"""Audit paired numerical results and regenerate publication figures."""
from pathlib import Path
import argparse,collections,csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ORDER=['warm_lp','full','no_mix','no_regions','no_stream','forced_mdp','frozen_lp']
LABELS={'warm_lp':'Warm LP','full':'Full controlled','no_mix':'No cached mixtures','no_regions':'No region search','no_stream':'No streaming reuse','forced_mdp':'Forced MDP','frozen_lp':'Frozen + LP rescue'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=HERE/'results');p.add_argument('--out',type=Path,default=HERE/'analysis');p.add_argument('--expected',type=int,default=225);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=[]
    for path in sorted(a.results.glob('*.json')):
        d=json.loads(path.read_text('utf8'))
        if d.get('run_complete'):d['_file']=path.name;data.append(d)
    reference={(d['case'],d.get('stream'),d.get('seed'),d['repeat'],d.get('magnitude'),d.get('dimension'),d['count'],d.get('stride')):d for d in data if d['method']=='warm_lp'}
    violations=[];unpaired=[];rows=[];query_total=0
    for d in data:
        key=(d['case'],d.get('stream'),d.get('seed'),d['repeat'],d.get('magnitude'),d.get('dimension'),d['count'],d.get('stride'))
        ref=reference.get(key);d['lp_checked']=ref is not None
        if ref is None:unpaired.append(d['_file'])
        if ref is not None and d['theta_sha256']!=ref['theta_sha256']:violations.append([d['_file'],'trace mismatch'])
        for i,r in enumerate(d['rows']):
            query_total+=1
            if r['status']!='ok' or not r.get('evaluation_ok',False):violations.append([d['_file'],i,'failed query/evaluation']);continue
            gap=r['upper']-r['objective'];tol=max(1e-4,.01*abs(r['objective']))+1e-6
            if r['evaluated_risk']>2+1e-7 or gap < -1e-6 or gap>tol:violations.append([d['_file'],i,'certificate'])
            if ref is not None and i<len(ref['rows']) and ref['rows'][i].get('status')=='ok':
                exact=ref['rows'][i]['objective']
                if r['objective']>exact+1e-5 or exact>r['upper']+1e-5:violations.append([d['_file'],i,'LP bracket'])
        last=next((r for r in reversed(d['rows']) if r['status']=='ok'),{})
        rows.append({k:d.get(k) for k in ['case','group','method','seed','repeat','count','total_seconds','setup_seconds','online_seconds','p95_ms','max_ms','direct_hits','full_solves','mixed_cache_returns','failures','validation_failures','peak_sampled_rss_mb']}|{'file':d['_file'],'cache_final':last.get('cache_size'),'cache_arrays_mb':last.get('cache_array_bytes',0)/2**20,'lp_checked':d['lp_checked']})
    if rows:
        with (a.out/'run_summary.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=collections.defaultdict(list)
    for d in data:
        if d['group']=='core':groups[(d['case'],d['method'])].append(d)
    summary=[]
    cases=list(dict.fromkeys(d['case'] for d in data if d['group']=='core'))
    for case in cases:
        for method in ORDER:
            ds=groups.get((case,method),[])
            if not ds:continue
            summary.append({'case':case,'method':method,'runs':len(ds),'median_total_s':float(np.median([d['total_seconds'] for d in ds])),'min_total_s':min(d['total_seconds'] for d in ds),'max_total_s':max(d['total_seconds'] for d in ds),'median_p95_ms':float(np.median([d['p95_ms'] for d in ds])),'direct_hits':sum(d['direct_hits'] for d in ds),'queries':sum(d['count'] for d in ds),'mixed_returns':sum(d['mixed_cache_returns'] for d in ds),'full_solves':sum(d['full_solves'] for d in ds),'failures':sum(d['failures'] for d in ds)})
    (a.out/'aggregate.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    audit={'completed_runs':len(data),'query_records':query_total,'violations':violations,'unpaired_runs':unpaired,'aggregation':'medians and min-max across runs, descriptive only; 3 timing repeats nested within 3 input seeds','all_complete':len(data)==a.expected}
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    if cases:
        fig,axes=plt.subplots(1,len(cases),figsize=(5*len(cases),4.4),squeeze=False)
        colors=['#888888','#176b43','#de9a32','#507da5','#9a6ab0','#b85d59','#697c37']
        for ax,case in zip(axes[0],cases):
            items=[x for x in summary if x['case']==case];xs=np.arange(len(items));med=[x['median_total_s'] for x in items]
            lo=[x['median_total_s']-x['min_total_s'] for x in items];hi=[x['max_total_s']-x['median_total_s'] for x in items]
            ax.bar(xs,med,color=[colors[ORDER.index(x['method'])] for x in items],yerr=[lo,hi],capsize=3)
            ax.set_xticks(xs,[LABELS[x['method']] for x in items],rotation=55,ha='right',fontsize=8);ax.set_ylabel('Setup + query time (s)');ax.set_title(case,fontsize=10);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Matched ablations: median and observed min-max (not confidence intervals)',fontsize=11);fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(a.out/f'ablation_totals.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    # Long-sequence prefix amortization: include each method's own initialization.
    long=[d for d in data if d['group']=='sensitivity' and d['count']>=100]
    if long:
        fig,ax=plt.subplots(figsize=(6,4))
        for method in ['full','warm_lp']:
            ds=[d for d in long if d['method']==method]
            if not ds:continue
            values=np.asarray([d['setup_seconds']+np.cumsum([r['seconds'] for r in d['rows']]) for d in ds])
            x=np.arange(1,values.shape[1]+1);ax.plot(x,np.median(values,axis=0),label=LABELS[method]);ax.fill_between(x,values.min(axis=0),values.max(axis=0),alpha=.15)
        ax.set(xlabel='Number of updates',ylabel='Cumulative planning time (s)',title='Initialization and reuse amortization');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(a.out/f'long_sequence.{ext}',dpi=220)
        plt.close(fig)
    print(json.dumps(audit))
if __name__=='__main__':main()
