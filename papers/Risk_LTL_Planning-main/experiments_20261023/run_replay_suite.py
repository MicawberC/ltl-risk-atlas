"""Paired fresh timing on four archived 20 Hz CARLA observation sequences.

No simulator is needed. These are newly executed solver measurements on old
traces, not new closed-loop driving. Methods run sequentially and each receives
the exact same every-tick observations and changing initial states.
"""
from pathlib import Path
import argparse,json,random,subprocess,sys,time
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'replay_results');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    carla=REPO/'reproduction/carla_local_risk';old=carla/'revised_certificate_20261014'
    tasks=[('single',trial) for trial in (0,3)]+[('double',trial) for trial in (0,3)]
    for kind,trial in tasks:
        for repeat in range(3):
            methods=['full','warm_lp','no_stream'];random.Random(1200+trial+repeat).shuffle(methods)
            for method in methods:
                name=f'{kind}_trial{trial}_r{repeat}_{method}';path=a.out/(name+'.json')
                if path.exists():
                    try:
                        if json.loads(path.read_text('utf8')).get('run_complete'):continue
                    except Exception:pass
                live=old/(f'{kind}_20hz_'+('900' if kind=='single' else '1200')+'_observed_vs_lp.json')
                reference=carla/('carla_trace_20hz.json' if kind=='single' else 'carla_trace_20hz_two_walkers.json')
                cmd=[sys.executable,str(HERE/'controlled_experiment.py'),'--method',method,'--replay',str(live),'--reference',str(reference),'--trial',str(trial),'--output',str(path)]
                print('REPLAY',name,flush=True);started=time.time()
                try:
                    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1200)
                    (a.out/(name+'.log')).write_bytes(result.stdout)
                    if result.returncode:raise RuntimeError('exit '+str(result.returncode))
                    d=json.loads(path.read_text('utf8'));d.update(group='replay',repeat=repeat,run_complete=True,command=cmd,wall_seconds=time.time()-started);path.write_text(json.dumps(d,indent=2),encoding='utf8')
                    print(' seconds',round(d['total_seconds'],3),'failures',d['failures'],flush=True)
                except Exception as exc:
                    path.write_text(json.dumps({'error':repr(exc),'command':cmd,'group':'replay','repeat':repeat,'run_complete':False},indent=2),encoding='utf8')
                    print('FAILED',repr(exc),flush=True)
    print('REPLAY SUITE FINISHED',flush=True)
if __name__=='__main__':main()
