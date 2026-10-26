"""Fresh, sequential, matched timing. Failed jobs are retained, not discarded."""
from pathlib import Path
import sys,json,subprocess,random,time,hashlib,argparse
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'experiments_20261023';REPO=HERE.parents[2]
sys.path.insert(0,str(OLD));sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_suite',OLD/'run_suite.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
from run_paper_atlas_suite import CASES

def workloads():
    result=[]
    for name,params,method,repeat,group in frozen.jobs():
        if method!='full':continue
        result.append((name.removesuffix('_full'),params,repeat,group,group=='core'))
    for index,(case,stream,seed) in enumerate(CASES):
        for repeat in range(3):
            result.append((f'historical_c{index}_r{repeat}',dict(case=case,stream=stream,seed=seed,count=20,magnitude=.8,dimension=5),repeat,'historical',index==6))
    carla=REPO/'reproduction/carla_local_risk'
    for kind,trial in [('single',0),('single',3),('double',0),('double',3)]:
        for repeat in range(3):
            live=carla/'revised_certificate_20261014'/(f'{kind}_20hz_'+('900' if kind=='single' else '1200')+'_observed_vs_lp.json')
            reference=carla/('carla_trace_20hz.json' if kind=='single' else 'carla_trace_20hz_two_walkers.json')
            result.append((f'replay_{kind}_t{trial}_r{repeat}',dict(replay=str(live),reference=str(reference),trial=trial),repeat,'replay',False))
    for seed in range(2026102601,2026102606):
        for index,(case,stream) in enumerate([('16:barrier_hard_active:0','iid'),('16:barrier_hard_active:1','walk'),('24:middle_fast_crossing:0','walk')]):
            for repeat in range(3):
                result.append((f'heldout_c{index}_s{seed}_r{repeat}',dict(case=case,stream=stream,seed=seed,count=20,magnitude=.8,dimension=5),repeat,'heldout',True))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',default='all');a=p.parse_args()
    out=HERE/'results';out.mkdir(exist_ok=True)
    code=hashlib.sha256((HERE/'optimized_controller.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    tasks=[]
    # Run the pressure inputs first to resolve the original slow cases early.
    work=workloads();work.sort(key=lambda w:(0 if w[3]=='core' and w[1]['case'].startswith('16:') else 1))
    for index,(name,params,repeat,group,include_old) in enumerate(work):
        if a.group not in ('all',group):continue
        methods=['warm_lp','optimized']+(['full'] if include_old else [])
        if (group=='core' and params.get('seed')==73044 and params['case'].startswith('16:')) or (group=='historical' and params.get('seed')==73047):methods.append('priced')
        random.Random(26000+index).shuffle(methods)
        for method in methods:tasks.append((name,params,repeat,group,method))
    (HERE/'run_manifest.json').write_text(json.dumps({'optimized_source_sha256':code,'workload_pairs':len([w for w in work if a.group in ('all',w[3])]),'jobs':len(tasks),'group':a.group,'prespecified_heldout_seeds':list(range(2026102601,2026102606))},indent=2),encoding='utf8')
    for index,(name,params,repeat,group,method) in enumerate(tasks):
        path=out/f'{name}_{method}.json'
        if path.exists():
            previous=json.loads(path.read_text('utf8'))
            if previous.get('run_complete') and previous.get('optimized_source_sha256')==code:continue
            raise RuntimeError('Existing result has another source version or was incomplete; preserve before rerunning '+str(path))
        command=[sys.executable,str(HERE/'run_one.py'),'--method',method,'--output',str(path)]
        for key,value in params.items():command.extend(['--'+key,str(value)])
        print(f'[{index+1}/{len(tasks)}] {name} {method}',flush=True);started=time.time()
        try:
            with path.with_suffix('.log').open('wb') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            if r.returncode:raise RuntimeError('subprocess exit '+str(r.returncode))
            data=json.loads(path.read_text('utf8'));data['run_complete']=True
        except Exception as exc:data={'run_complete':False,'error':repr(exc),'method':method}
        data.update(group=group,repeat=repeat,pair_id=name,command=command,optimized_source_sha256=code,wall_seconds=time.time()-started)
        path.write_text(json.dumps(data,indent=2),encoding='utf8')
        print('seconds',round(data.get('total_seconds',0),3),'failures',data.get('failures',data.get('error')),flush=True)
    print('SUITE FINISHED',flush=True)

if __name__=='__main__':main()
