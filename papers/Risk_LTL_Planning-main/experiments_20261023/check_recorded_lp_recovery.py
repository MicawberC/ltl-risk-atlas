"""Recheck the actual 32x32 LP occupation that exposed the recovery bug."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from controlled_experiment import *

def main():
    z=np.load(Path(__file__).with_name('lp_recovery_fixture.npz'),allow_pickle=False)
    model,dirs,_=make_case('32:middle_fast_crossing:0')
    expected=2.9609965677110606
    values=[]
    for clip in [False,True]:
        occ=np.maximum(z['occupation'],0) if clip else z['occupation'];total=occ.sum(axis=1)
        pi=np.divide(occ,total[:,None],out=np.zeros_like(occ),where=total[:,None]>1e-12);pi[total<=1e-12,0]=1
        e=evaluate_policy(replace(model,cost=model.cost+z['theta']@dirs),pi)
        values.append({'clip':clip,'probability_min':float(pi.min()),'probability_max':float(pi.max()),'objective':e['objective'],'risk':e['risk'],'objective_error':abs(e['objective']-expected)})
    assert values[0]['probability_min']<0
    assert values[1]['probability_min']>=0 and values[1]['probability_max']<=1
    assert values[1]['objective_error']<1e-5
    Path(__file__).with_name('recorded_lp_recovery_check.json').write_text(json.dumps({'passed':True,'rows':values},indent=2),encoding='utf8')
    print('PASS: recorded LP recovery fixture; repaired error',values[1]['objective_error'])
if __name__=='__main__':main()
