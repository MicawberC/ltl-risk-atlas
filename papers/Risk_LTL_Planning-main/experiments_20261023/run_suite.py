"""Sequential subprocess runs; deterministic workload/order; resumable output."""
from pathlib import Path
import argparse,itertools,json,os,random,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def jobs():
    cases=[('24:middle_fast_crossing:0','walk'),('32:middle_fast_crossing:0','walk'),('16:barrier_hard_active:0','iid')]
    methods=['warm_lp','full','no_mix','no_regions','no_stream','forced_mdp','frozen_lp']
    tasks=[]
    for ci,(case,stream) in enumerate(cases):
        for seed in [73044,73045,73046]:
            for repeat in range(3):
                order=methods.copy();random.Random(91000+ci*100+seed+repeat).shuffle(order)
                for method in order:
                    tasks.append((f'core_c{ci}_s{seed}_r{repeat}_{method}',dict(case=case,stream=stream,seed=seed,count=20,magnitude=.8,dimension=5),method,repeat,'core'))
    # Sensitivity: topology, support size, magnitude, and long-cache workload.
    sensitivity=[('24:middle_fast_varied:1','walk',20,.8,5),('24:middle_fast_crossing:0','walk',20,.2,5),('24:middle_fast_crossing:0','iid',20,.8,5),('24:middle_fast_crossing:0','walk',20,.8,1),('24:middle_fast_crossing:0','walk',20,.8,3),('24:middle_fast_crossing:0','walk',160,.8,5)]
    for ci,(case,stream,count,mag,dim) in enumerate(sensitivity):
        for repeat in range(3):
            order=['warm_lp','full'];random.Random(8200+ci+repeat).shuffle(order)
            for method in order:
                tasks.append((f'sensitivity_c{ci}_r{repeat}_{method}',dict(case=case,stream=stream,seed=83001,count=count,magnitude=mag,dimension=dim),method,repeat,'sensitivity'))
    return tasks

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--group',default='all');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    tasks=[t for t in jobs() if a.group=='all' or t[-1]==a.group]
    manifest=[]
    for i,(name,params,method,repeat,group) in enumerate(tasks):
        path=a.out/(name+'.json');cmd=[sys.executable,str(HERE/'controlled_experiment.py'),'--method',method,'--output',str(path)]
        for key,value in params.items():cmd += ['--'+key,str(value)]
        if path.exists():
            try:
                data=json.loads(path.read_text('utf8'))
                if data.get('run_complete'):continue
            except (ValueError,OSError):pass
        started=time.time()
        print(f'[{i+1}/{len(tasks)}] {name}',flush=True)
        try:
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
            (a.out/(name+'.log')).write_bytes(result.stdout)
            if result.returncode:raise RuntimeError('subprocess returncode '+str(result.returncode))
            data=json.loads(path.read_text('utf8'));data.update(repeat=repeat,group=group,run_complete=True,command=cmd,wall_seconds=time.time()-started)
            path.write_text(json.dumps(data,indent=2),encoding='utf8')
            print(' total',round(data['total_seconds'],3),'failures',data['failures'],flush=True)
        except Exception as exc:
            error={'error':repr(exc),'command':cmd,'run_complete':False,'group':group,'repeat':repeat}
            path.write_text(json.dumps(error,indent=2),encoding='utf8');print(error['error'],flush=True)
        manifest.append(name)
    print('SUITE FINISHED',flush=True)
if __name__=='__main__':main()
