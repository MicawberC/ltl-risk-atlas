"""Rerun all twelve historical input traces with the corrected common driver.

These times replace no historical implementation times. All twelve inputs are
included, including the original reward-scaled pressure cases and counterexample.
"""
from pathlib import Path
import sys,json,random,subprocess,time
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
from run_paper_atlas_suite import CASES

def main():
    out=HERE/'historical_results';out.mkdir(exist_ok=True)
    for number,(case,stream,seed) in enumerate(CASES):
        for repeat in range(3):
            methods=['full','warm_lp'];random.Random(15300+number*10+repeat).shuffle(methods)
            for method in methods:
                path=out/f'historical_c{number}_r{repeat}_{method}.json'
                if path.exists():
                    old=json.loads(path.read_text('utf8'))
                    if old.get('run_complete'):continue
                command=[sys.executable,str(HERE/'controlled_experiment.py'),'--case',case,'--stream',stream,'--seed',str(seed),'--count','20','--magnitude','0.8','--dimension','5','--method',method,'--output',str(path)]
                print('HISTORICAL',number,repeat,method,flush=True);started=time.time()
                with path.with_suffix('.log').open('wb') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                if result.returncode:
                    data={'case':case,'stream':stream,'seed':seed,'method':method,'run_complete':False,'error':'subprocess exit '+str(result.returncode)}
                else:
                    data=json.loads(path.read_text('utf8'));data['run_complete']=True
                data.update(group='historical',repeat=repeat,source_suite_index=number,command=command,wall_seconds=time.time()-started)
                path.write_text(json.dumps(data,indent=2),encoding='utf8')
                print('seconds',round(data.get('total_seconds',0),3),'failures',data.get('failures',data.get('error')),flush=True)
    print('HISTORICAL SUITE FINISHED',flush=True)

if __name__=='__main__':main()
