"""Explicit post-batch recovery of infrastructure-only missing drives.

Never rerun a completed driving outcome, including planner failure/time limit.
This is an amendment, not an unreported extra repeat of the primary protocol.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_suite import OwnedServer,archive_attempt,code_hashes,load_json,save,utc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--server',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=HERE);a=ap.parse_args();out=a.out.resolve()
    manifest=load_json(out/'run_manifest.json');hashes=code_hashes()
    if manifest['source_hashes_lf']!=hashes:raise RuntimeError('Frozen driver or algorithm changed')
    jobs=[]
    for job in manifest['jobs']:
        stem=f"{job['case']}_r{job['repetition']}_{job['method']}"
        if not load_json(out/'results'/f'{stem}.json').get('trial_complete'):jobs.append((stem,job))
    amendment={'created_utc':utc(),'reason':'Recover infrastructure-only missing data after the primary matrix',
               'maximum_additional_attempts':2,'never_retry_complete_outcomes':True,
               'primary_protocol_sha256':manifest['protocol_sha256'],'jobs':[j for _,j in jobs],
               'source_hashes_lf':hashes,'recovery_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out/'recovery_manifest.json',amendment)
    status=load_json(out/'suite_status.json');recovered=[]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cases=list(dict.fromkeys(j['case'] for j in manifest['jobs']))
    for stem,job in jobs:
        rp=out/'results'/f'{stem}.json';pp=out/'policies'/f'{stem}.zip'
        lp=out/'logs'/f'{stem}.log';ep=out/'logs'/f'{stem}_execution.json'
        previous=load_json(ep);first=int(previous.get('attempt',-1))+1
        for attempt in range(first,first+2):
            if load_json(rp).get('trial_complete'):raise RuntimeError('Refusing to rerun a completed outcome')
            archive_attempt(out,stem,attempt-1,[rp,pp,lp,ep])
            port=manifest['base_port']+10*cases.index(job['case'])
            server=OwnedServer(a.server,job['town'],port,out/'server_logs'/f'recover_{time.time_ns()}',180)
            cmd=[sys.executable,str(HERE/'run_trial.py'),'--town',job['town'],'--route-index',str(job['route_index']),
                 '--max-ticks',str(job['max_ticks']),'--port',str(port),'--method',job['method'],
                 '--seed',str(job['seed']),'--repetition',str(job['repetition']),'--output',str(rp),
                 '--policies',str(pp),'--reference',str(out/'references'/f"{job['case']}.json")]
            if job['second_walker']:cmd.append('--second-walker')
            execution={'started_utc':utc(),'command':cmd,'attempt':attempt,'protocol_sha256':manifest['protocol_sha256'],
                       'source_hashes_lf':hashes,'explicit_infrastructure_recovery':True}
            began=time.perf_counter();success=False
            print('RECOVERY',stem,'attempt',attempt,flush=True)
            try:
                server.start()
                with lp.open('wb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
                execution['returncode']=r.returncode;trial=load_json(rp)
                success=r.returncode==0 and trial.get('trial_complete') and pp.exists()
                execution.update(trial_complete=bool(success),outcome=trial.get('outcome'))
                if not success:execution['error']=trial.get('infrastructure_failure','Incomplete drive output')
            except Exception as exc:execution.update(returncode=None,error=repr(exc),traceback=traceback.format_exc())
            finally:server.stop()
            execution.update(ended_utc=utc(),wall_seconds=time.perf_counter()-began);save(ep,execution)
            if success:
                item={'trial':stem,'outcome':execution['outcome'],'attempt':attempt,'explicit_infrastructure_recovery':True}
                status['completed'].append(item);recovered.append(item);save(out/'suite_status.json',status);break
            location=archive_attempt(out,stem,attempt,[rp,pp,lp,ep])
            status['infrastructure_failures'].append({'trial':stem,'attempt':attempt,'diagnostics':location,'error':execution.get('error')})
            save(out/'suite_status.json',status)
    status.update(complete=len(status['completed'])==len(manifest['jobs']),ended_utc=utc())
    save(out/'suite_status.json',status);save(out/'recovery_status.json',{'recovered':recovered,'complete':status['complete']})
    return 0 if status['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
