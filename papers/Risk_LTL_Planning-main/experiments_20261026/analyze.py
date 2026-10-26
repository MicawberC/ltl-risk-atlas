"""Audit same-input primal/LP/dual brackets, then summarize all outcomes."""
from pathlib import Path
import json,csv,collections,statistics as st
HERE=Path(__file__).resolve().parent

def summary_key(d):
    keys=['group','case']
    if d['group'] in ('historical','sensitivity'):keys+=['stream','seed','magnitude','dimension','count']
    return tuple(d.get(k) for k in keys)

def input_key(d):
    return tuple(d.get(k) for k in ('group','case','stream','seed','repeat','magnitude','dimension','count','stride'))

def main():
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    data=[];incomplete=[]
    for p in sorted((HERE/'results').glob('*.json')):
        d=json.loads(p.read_text('utf8'));d['file']=p.name
        (data if d.get('run_complete') else incomplete).append(d)
    refs={d['pair_id']:d for d in data if d['method']=='warm_lp'}
    errors=[];queries=0;invalid=set();runs=[];quality=collections.defaultdict(list)
    historical_inputs={}
    for folder in ('results','historical_results','replay_results'):
        for p in (HERE.parent/'experiments_20261023'/folder).glob('*.json'):
            previous=json.loads(p.read_text('utf8'))
            if previous.get('run_complete') and previous.get('method')=='warm_lp':historical_inputs[input_key(previous)]=previous['theta_sha256']
    input_audit=[]
    for d in data:
        r=refs.get(d['pair_id'])
        if r is None:errors.append([d['file'],'missing LP']);invalid.add(d['pair_id']);continue
        if d['theta_sha256']!=r['theta_sha256'] or d['count']!=r['count']:errors.append([d['file'],'input mismatch']);invalid.add(d['pair_id']);continue
        if d['method']=='optimized' and d['group']!='heldout':
            identical=historical_inputs.get(input_key(d))==d['theta_sha256']
            input_audit.append({'pair_id':d['pair_id'],'identical_to_previous_batch':identical})
            if not identical:errors.append([d['file'],'previous batch input mismatch']);invalid.add(d['pair_id'])
        for i,(q,lp) in enumerate(zip(d['rows'],r['rows'])):
            queries+=1;problem=[]
            if q['status']!='ok' or not q.get('evaluation_ok'):problem.append('failed query/evaluation')
            elif lp['status']!='ok' or not lp.get('evaluation_ok'):problem.append('invalid LP reference')
            else:
                gap=q['upper']-q['evaluated_objective'];tol=max(1e-4,.01*abs(q['evaluated_objective']))+1e-6
                if q['evaluated_risk']>2+1e-7 or gap < -1e-5 or gap>tol:problem.append('risk/gap')
                if q['objective']>lp['objective']+1e-5 or lp['objective']>q['upper']+1e-5:problem.append('LP bracket')
                quality[(d['group'],d['method'])].append({'risk':q['evaluated_risk'],'error':max(0.,lp['objective']-q['evaluated_objective']),'budget':max(1e-4,.01*abs(q['evaluated_objective'])),'gap':gap})
            if problem:errors.append([d['file'],i,*problem]);invalid.add(d['pair_id'])
        runs.append({k:d.get(k) for k in ['file','pair_id','group','case','seed','repeat','method','count','setup_seconds','online_seconds','total_seconds','p95_ms','max_ms','failures','validation_failures','direct_hits','full_solves','mixed_cache_returns','peak_sampled_rss_mb']}|{'scalarized_calls':sum(q.get('scalarized_calls',0) for q in d['rows'])})
    pairs=[]
    for d in data:
        if d['method']!='optimized' or d['pair_id'] not in refs:continue
        lp=refs[d['pair_id']];valid=d['pair_id'] not in invalid
        pairs.append({'pair_id':d['pair_id'],'group':d['group'],'case':d['case'],'seed':d.get('seed'),'repeat':d['repeat'],'optimized_s':d['total_seconds'],'lp_s':lp['total_seconds'],'lp_over_optimized':lp['total_seconds']/d['total_seconds'],'valid':valid,'optimized_faster':valid and d['total_seconds']<lp['total_seconds']})
    grouped=collections.defaultdict(list)
    for d in data:grouped[(summary_key(d),d['method'])].append(d)
    summary=[]
    for (key,method),ds in grouped.items():
        times=[d['total_seconds'] for d in ds]
        summary.append({'key':list(key),'group':key[0],'case':key[1],'method':method,'runs':len(ds),'median_s':st.median(times),'min_s':min(times),'max_s':max(times),'median_p95_ms':st.median(d['p95_ms'] for d in ds),'max_ms':max(d['max_ms'] for d in ds),'scalarized_calls':sum(q.get('scalarized_calls',0) for d in ds for q in d['rows']),'full_solves':sum(d['full_solves'] for d in ds),'failures':sum(d['failures'] for d in ds)})
    wins=[]
    for group in ['core','historical','sensitivity','replay','heldout']:
        ps=[p for p in pairs if p['group']==group];ratios=[p['lp_over_optimized'] for p in ps if p['valid']]
        wins.append({'group':group,'pairs':len(ps),'faster':sum(p['optimized_faster'] for p in ps),'slower':sum(p['valid'] and p['lp_over_optimized']<1 for p in ps),'invalid':sum(not p['valid'] for p in ps),'median_ratio':st.median(ratios) if ratios else None,'min_ratio':min(ratios) if ratios else None,'max_ratio':max(ratios) if ratios else None})
    for name,items in [('run_summary',runs),('paired_times',pairs)]:
        if items:
            with (out/(name+'.csv')).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (out/'wins.json').write_text(json.dumps(wins,indent=2),encoding='utf8')
    qs=[]
    for (group,method),values in quality.items():
        qs.append({'group':group,'method':method,'queries':len(values),'max_risk':max(v['risk'] for v in values),'max_objective_error':max(v['error'] for v in values),'median_budget_used':st.median(v['error']/v['budget'] for v in values),'max_budget_used':max(v['error']/v['budget'] for v in values),'max_certificate_budget_used':max(v['gap']/v['budget'] for v in values)})
    (out/'quality.json').write_text(json.dumps(qs,indent=2),encoding='utf8')
    (out/'previous_input_audit.json').write_text(json.dumps(input_audit,indent=2),encoding='utf8')
    audit={'runs':len(data),'expected_runs':357,'query_records':queries,'incomplete':[d['file'] for d in incomplete],'violations':errors,'pairs':len(pairs),'expected_pairs':138,'previous_inputs_checked':len(input_audit),'all_complete':len(data)==357 and len(pairs)==138 and not incomplete,'all_valid':not errors}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit));print(json.dumps(wins,indent=2))

if __name__=='__main__':main()
