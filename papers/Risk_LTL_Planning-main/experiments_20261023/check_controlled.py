"""Independent LP + policy-evaluation checks for every ablation branch."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from controlled_experiment import *

def main():
    model,dirs,thetas,initials,meta=make_synthetic('16:barrier_hard_active:0','iid',73047,8)
    lp=lp_prepare(model,dirs,feasibility_tolerance=1e-9,solver_threads=1)
    exact=[lp_query(lp,t)['objective'] for t in thetas]
    checks=[]
    for method in METHODS[1:]:
        c=Controller(model,dirs,thetas[0],method)
        for i,t in enumerate(thetas):
            a=c.query(t)
            e=evaluate_policy(replace(model,cost=model.cost+t@dirs),a['policy'])
            assert abs(e['objective']-a['objective'])<1e-5
            assert abs(e['risk']-a['risk'])<1e-5 and e['risk']<=HARD+1e-7
            assert a['objective']<=exact[i]+1e-5<=a['upper']+2e-5
            assert acceptable((a['objective'],a['risk']),a['upper'])
            if method=='no_mix':assert not a['mixed_cache']
            if method=='frozen_lp':assert len(c.atlas['policies'])==2
            if method=='forced_mdp':assert a['scalarized_calls']>=1
            checks.append({'method':method,'index':i,'objective':a['objective'],'risk':a['risk'],'upper':a['upper'],'lp':exact[i],'certificate_type':a['certificate_type']})
    out=Path(__file__).with_name('controlled_check_results.json')
    out.write_text(json.dumps({'passed':True,'independent_query_checks':len(checks),'rows':checks},indent=2),encoding='utf8')
    print('PASS:',len(checks),'independent LP/policy checks over all ablations')
if __name__=='__main__':main()
