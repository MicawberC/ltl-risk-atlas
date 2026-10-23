"""Check primal activity AND positive hard-risk dual multipliers, untimed."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from controlled_experiment import *

def main():
    rows=[]
    for seed in [73044,73045,73046]:
        m,d,theta,_,_=make_synthetic('16:barrier_hard_active:0','iid',seed,20)
        lp=lp_prepare(m,d,feasibility_tolerance=1e-9,solver_threads=1)
        for i,t in enumerate(theta):
            answer=lp_query(lp,t);dual=np.asarray(lp['solver'].getSolution().row_dual)
            hard=max(0.,-float(dual[m.states+len(d)+2]));soft=max(0.,-float(dual[m.states+len(d)+1]))
            rows.append({'seed':seed,'index':i,'risk':answer['risk'],'lambda_hard':hard,'lambda_soft':soft,'hard_at_boundary':abs(answer['risk']-HARD)<1e-5,'positive_hard_multiplier':hard>1e-8})
    result={'scope':'Untimed additional LP diagnostic; not another independent workload or timing repeat','rows':rows,'active_and_positive':sum(r['hard_at_boundary'] and r['positive_hard_multiplier'] for r in rows),'count':len(rows)}
    Path(__file__).with_name('hard_activity.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('Hard boundary with positive multiplier:',result['active_and_positive'],'/',len(rows))
if __name__=='__main__':main()
