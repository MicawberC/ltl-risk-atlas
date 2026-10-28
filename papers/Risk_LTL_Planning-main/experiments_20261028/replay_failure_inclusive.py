"""Post-hoc supplement: include terminal failed-query cost, without deleting it.

Select all failed-driving traces among the original twelve r0 sources.
This is reported separately from the prespecified returned-prefix benchmark.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,json,random,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from replay import save

def worker(a):
    import numpy as np
    from planner import Planner,build_model
    from validate_results import resolve_recorded_path
    trial=json.loads(a.trace.read_text('utf8'))
    reference=resolve_recorded_path(trial['reference_path'],a.trace)
    base,directions=build_model(json.loads(reference.read_text('utf8')))
    result={'trace':a.trace.name,'trace_sha256':hashlib.sha256(a.trace.read_bytes()).hexdigest(),
            'method':a.method,'repetition':a.repetition,'case':trial['case'],
            'expected_queries':len(trial['rows']),'rows':[],'complete':False}
    planner=None
    for row in trial['rows']:
        theta=np.asarray(row['theta']);initial=int(row['initial'])
        expected_failure=bool(row.get('planner_failure'));began=time.perf_counter()
        try:
            if planner is None:planner=Planner(base,directions,theta,initial,a.method)
            answer=planner.query(theta,initial)
            measured={'tick':row['tick'],'plan_s':time.perf_counter()-began,'returned':True,
                      'objective':float(answer['objective']),'risk':float(answer['risk']),'gap':float(answer['gap'])}
            if expected_failure:result['error']='Unexpected certified return on the source failed input'
        except Exception as exc:
            measured={'tick':row['tick'],'plan_s':time.perf_counter()-began,'returned':False,
                      'expected_failure':expected_failure,'error':repr(exc)}
            if not expected_failure:result['error']=traceback.format_exc()
        result['rows'].append(measured)
        if result.get('error') or not measured['returned']:break
    result['complete']=len(result['rows'])==len(trial['rows']) and not result.get('error')
    result['planning_time_s']=sum(r['plan_s'] for r in result['rows'])
    result['failed_query_time_s']=sum(r['plan_s'] for r in result['rows'] if not r['returned'])
    save(a.output,result)
    print(a.trace.stem,a.method,a.repetition,result['complete'],round(result['planning_time_s'],3),
          'failed_s',round(result['failed_query_time_s'],3),flush=True)
    return 0 if result['complete'] else 2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trace',type=Path)
    ap.add_argument('--method',choices=['optimized','warm_lp']);ap.add_argument('--repetition',type=int)
    ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.trace:return worker(a)
    out=HERE/'failure_inclusive';traces=[]
    for p in sorted((HERE/'results').glob('*.json')):
        t=json.loads(p.read_text('utf8'))
        if t['repetition']==0 and t['outcome']=='planner_failure':traces.append(p)
    rng=random.Random(20261028);jobs=[]
    for p in traces:
        for repeat in range(3):
            methods=['optimized','warm_lp'];rng.shuffle(methods)
            jobs.extend((p,repeat,m) for m in methods)
    save(out/'manifest.json',{'scope':'Post-hoc complete-prefix supplement including the final failed call',
         'selection':'All failed-driving r0 traces among the original twelve sources',
         'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in traces},
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'jobs':[{'trace':p.name,'repetition':r,'method':m} for p,r,m in jobs]})
    failures=[]
    for p,r,m in jobs:
        target=out/f'{p.stem}_replay{r}_{m}.json'
        cmd=[sys.executable,str(Path(__file__)),'--trace',str(p),'--method',m,'--repetition',str(r),'--output',str(target)]
        proc=subprocess.run(cmd,timeout=600)
        if proc.returncode:failures.append(target.name)
    save(out/'status.json',{'runs':len(jobs),'failed':failures,'complete':not failures})
    return 0 if not failures else 2
if __name__=='__main__':raise SystemExit(main())
