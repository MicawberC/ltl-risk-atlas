"""Paired timing of all prespecified repetition-zero driving prefixes.

Run after CARLA has stopped and independently of verification workloads.
Every individual replay is a fresh process. No failure-dependent trace selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,random,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')

def worker(args):
    import numpy as np
    from planner import Planner,build_model
    from validate_results import resolve_recorded_path
    trial=json.loads(args.trace.read_text('utf8'))
    reference=resolve_recorded_path(trial['reference_path'],args.trace)
    base,directions=build_model(json.loads(reference.read_text('utf8')))
    inputs=[r for r in trial['rows'] if 'policy_key' in r]
    result={'trace':args.trace.name,'trace_sha256':hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            'source_method':trial['method'],'source_outcome':trial.get('outcome'),
            'case':trial['case'],'method':args.method,'repetition':args.repetition,
            'expected_queries':len(inputs),'rows':[],'complete':False}
    controller=None
    for i,row in enumerate(inputs):
        theta=np.asarray(row['theta']);initial=int(row['initial']);began=time.perf_counter()
        try:
            if controller is None:controller=Planner(base,directions,theta,initial,args.method)
            answer=controller.query(theta,initial)
            elapsed=time.perf_counter()-began
            result['rows'].append({'tick':row['tick'],'plan_s':elapsed,'objective':float(answer['objective']),
                                   'risk':float(answer['risk']),'gap':float(answer['gap']),
                                   'scalarized_calls':int(answer.get('scalarized_calls',0)),
                                   'full_solve':bool(answer.get('full_solve',False))})
        except Exception as exc:
            result['rows'].append({'tick':row['tick'],'plan_s':time.perf_counter()-began,'error':repr(exc)})
            result['error']=traceback.format_exc();break
    result['complete']=len(result['rows'])==len(inputs) and not result.get('error') and len(inputs)>0
    result['planning_time_s']=sum(r['plan_s'] for r in result['rows'])
    save(args.output,result)
    print(args.trace.stem,args.method,args.repetition,result['complete'],round(result['planning_time_s'],3),flush=True)
    return 0 if result['complete'] else 2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results-dir',type=Path,default=HERE/'results')
    ap.add_argument('--out',type=Path,default=HERE/'replays');ap.add_argument('--trace',type=Path)
    ap.add_argument('--method',choices=['optimized','warm_lp']);ap.add_argument('--repetition',type=int)
    ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.trace:return worker(args)
    traces=[]
    for path in sorted(args.results_dir.glob('*.json')):
        trial=json.loads(path.read_text('utf8'))
        if trial.get('repetition')==0:
            if not trial.get('trial_complete'):raise RuntimeError(f'Incomplete prespecified trace: {path}')
            traces.append(path)
    if len(traces)!=12:raise RuntimeError(f'Expected all 12 prespecified traces, found {len(traces)}')
    rng=random.Random(20261028);jobs=[]
    for trace in traces:
        for repetition in range(3):
            methods=['optimized','warm_lp'];rng.shuffle(methods)
            for method in methods:jobs.append((trace,repetition,method))
    manifest={'selection':'Every scene and every driving method, repetition 0; all returned prefixes',
              'traces':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in traces},
              'order':[{'trace':p.name,'repetition':r,'method':m} for p,r,m in jobs],
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/'planner.py')}}
    save(args.out/'manifest.json',manifest);failed=[]
    for trace,repetition,method in jobs:
        target=args.out/f'{trace.stem}_replay{repetition}_{method}.json'
        cmd=[sys.executable,str(Path(__file__)),'--trace',str(trace),'--method',method,
             '--repetition',str(repetition),'--output',str(target)]
        run=subprocess.run(cmd,timeout=600)
        if run.returncode:failed.append(target.name)
    save(args.out/'status.json',{'runs':len(jobs),'failed':failed,'complete':not failed})
    return 0 if not failed else 2
if __name__=='__main__':raise SystemExit(main())
