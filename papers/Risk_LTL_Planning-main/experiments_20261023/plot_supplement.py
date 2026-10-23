"""Figures from measured JSON only; error bars are observed min-max."""
from pathlib import Path
import collections,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'analysis'
def load(folder):
    out=[]
    for p in (HERE/folder).glob('*.json'):
        try:
            d=json.loads(p.read_text('utf8'))
            if d.get('run_complete'):out.append(d)
        except Exception:pass
    return out
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
def main():
    OUT.mkdir(exist_ok=True)
    ds=load('results');rp=load('replay_results');historical=load('historical_results')
    if historical:
        fig,ax=plt.subplots(figsize=(8,4))
        for number in sorted(set(d['source_suite_index'] for d in historical)):
            full={d['repeat']:d for d in historical if d['source_suite_index']==number and d['method']=='full'}
            lp={d['repeat']:d for d in historical if d['source_suite_index']==number and d['method']=='warm_lp'}
            ratios=[lp[k]['total_seconds']/full[k]['total_seconds'] for k in set(full)&set(lp)]
            if not ratios:continue
            y=np.median(ratios)
            ax.errorbar(number,y,yerr=[[y-min(ratios)],[max(ratios)-y]],fmt='o',capsize=4,color='#176b43' if y>1 else '#b04444')
        ax.axhline(1,color='#777777',ls='--');ax.set_yscale('log');ax.set_xticks(range(12))
        ax.set(xlabel='Original input trace index',ylabel='Warm LP / Full planning time',title='All twelve historical inputs: paired median and observed min-max')
        ax.grid(axis='y',alpha=.2);save(fig,'historical_twelve_ratios')
    methods=['full','no_mix','no_regions','no_stream','forced_mdp','frozen_lp','warm_lp']
    cases=list(dict.fromkeys(d['case'] for d in ds if d['group']=='core'))
    if cases:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for ax,key,title in zip(axes,['mixed_cache_returns','full_solves'],['Cached-mixture returns (%)','Complete solves / LP rescues (%)']):
            matrix=np.full((len(cases),len(methods)),np.nan)
            for i,case in enumerate(cases):
                for j,m in enumerate(methods):
                    ss=[d for d in ds if d['group']=='core' and d['case']==case and d['method']==m]
                    if ss:matrix[i,j]=100*sum(d[key] for d in ss)/sum(d['count'] for d in ss)
            im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
            for i in range(len(cases)):
                for j in range(len(methods)):
                    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',color='white' if matrix[i,j]>55 else 'black',fontsize=8)
            ax.set_xticks(range(len(methods)),methods,rotation=55,ha='right',fontsize=8);ax.set_yticks(range(len(cases)),cases,fontsize=8);ax.set_title(title,fontsize=10)
        axes[1].set_yticks([])
        fig.subplots_adjust(left=.21,bottom=.28,right=.9,wspace=.18)
        colorbar_axis=fig.add_axes([.925,.28,.015,.58])
        fig.colorbar(im,cax=colorbar_axis)
        for ext in ['png','svg']:fig.savefig(OUT/f'mixture_and_expansion.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    if rp:
        cases=list(dict.fromkeys(d['case'] for d in rp));fig,axes=plt.subplots(2,2,figsize=(8,6.5))
        for ax,case in zip(axes.flat,cases):
            for j,m in enumerate(['full','no_stream','warm_lp']):
                vals=[d['total_seconds'] for d in rp if d['case']==case and d['method']==m]
                if not vals:continue
                y=np.median(vals);ax.bar(j,y,color=['#176b43','#9a6ab0','#888888'][j]);ax.errorbar(j,y,yerr=[[y-min(vals)],[max(vals)-y]],fmt='none',color='black',capsize=3)
            ax.set_xticks(range(3),['Full','No stream','Warm LP']);ax.set_ylabel('Setup + queries (s)');ax.set_title(case.replace('replay:','').replace('_20hz_','\n'),fontsize=8);ax.grid(axis='y',alpha=.2)
        save(fig,'paired_replay_times')
    live=[]
    for p in (HERE/'carla_results').glob('*.json'):
        d=json.loads(p.read_text('utf8'))
        if 'trials' in d:
            for t in d['trials']:live.append((p.stem,t))
    if live:
        cases=list(dict.fromkeys(c for c,t in live));fig,axes=plt.subplots(2,2,figsize=(9,6.5))
        colors={'fixed_observed_hybrid':'#176b43','fixed_warm_lp':'#777777'}
        for ax,case in zip(axes.flat,cases):
            labelled=set()
            for c,t in live:
                if c!=case:continue
                rows=[r for r in t['rows'] if 'longitudinal_m' in r]
                if not rows:continue
                method=t['method'];label=('Atlas' if 'hybrid' in method else 'Warm LP') if method not in labelled else None
                ax.plot([r['sim_time_s'] for r in rows],[r['longitudinal_m'] for r in rows],color=colors[method],alpha=.65,label=label)
                if t['planner_failure']:ax.plot(rows[-1]['sim_time_s'],rows[-1]['longitudinal_m'],'x',color=colors[method],ms=7)
                labelled.add(method)
            ax.axhline(7.5,color='#b04444',ls='--',lw=1,label='Goal threshold')
            ax.set(title=case,xlabel='Simulation time (s)',ylabel='Road longitudinal position (m)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
        save(fig,'new_carla_progress')
        cases=list(dict.fromkeys(c for c,t in live));fig,axes=plt.subplots(2,2,figsize=(8,6))
        for ax,case in zip(axes.flat,cases):
            for j,m in enumerate(['fixed_observed_hybrid','fixed_warm_lp']):
                trials=[t for c,t in live if c==case and t['method']==m]
                vals=[t['planning_time_s'] for t in trials if t['goal'] and not t['collision'] and not t['planner_failure']]
                incomplete=len(trials)-len(vals)
                if incomplete:
                    ax.text(j,.95,f'{incomplete}/{len(trials)}\nunsuccessful',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8,color='#b04444')
                if not vals:continue
                y=np.median(vals);ax.bar(j,y,color=['#176b43','#888888'][j]);ax.errorbar(j,y,yerr=[[y-min(vals)],[max(vals)-y]],fmt='none',color='black',capsize=3)
            ax.set_xlim(-.6,1.6)
            if not any(t['goal'] and not t['collision'] and not t['planner_failure'] for c,t in live if c==case):
                ax.set_yticks([]);ax.text(.5,.45,'No successful drives',transform=ax.transAxes,ha='center',color='#555555')
            ax.set_xticks([0,1],['Atlas','Warm LP']);ax.set_ylabel('Planning in successful drives (s)');ax.set_title(case,fontsize=9);ax.grid(axis='y',alpha=.2)
        save(fig,'new_carla_totals')
        fig,ax=plt.subplots(figsize=(7,4))
        for m in ['fixed_observed_hybrid','fixed_warm_lp']:
            values=sorted((r.get('product_s',0)+r['plan_s'])*1000 for c,t in live if t['method']==m for r in t['rows'] if 'plan_s' in r)
            if values:ax.plot(values,np.arange(1,len(values)+1)/len(values),label='Atlas' if 'hybrid' in m else 'Warm LP')
        ax.axvline(50,color='#b04444',ls='--',label='50 ms reference');ax.set_xscale('log');ax.set(xlabel='Recorded query time (ms)',ylabel='Empirical cumulative fraction',title='New closed-loop latency; untimed exception calls excluded');ax.legend();ax.grid(alpha=.2);save(fig,'new_carla_latency')
    print('Supplement figures generated')
if __name__=='__main__':main()
