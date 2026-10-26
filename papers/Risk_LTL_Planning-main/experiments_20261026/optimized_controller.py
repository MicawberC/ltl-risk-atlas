"""Certified pricing from the cached primal hull, with warm policy iteration.

The old experimental driver is imported unchanged. Prices are proposals only;
every returned policy passes the same primal/global-dual gap gate.
"""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments_20261023'))
import controlled_experiment as old
import numpy as np
from dataclasses import replace
from stratified import solve_scalarized,dual_bound

class OptimizedController(old.Controller):
    def __init__(self,model,directions,theta,method='optimized',warm=True,rounds=6):
        super().__init__(model,directions,theta,'full')
        self.warm=warm;self.pricing_rounds=rounds

    def proposed_price(self,theta,p):
        if p is None:return max(1.,2*float(self.last_price or 1.))
        _,risk,i,j,_=p
        if i==j:
            return 0. if risk<old.SOFT-1e-8 else (1. if risk<old.HARD-1e-8 else max(1.,float(self.last_price or 1.)))
        a,b=self.atlas['policies'][i],self.atlas['policies'][j]
        ca=float(a['risk_coeff'][0]+a['risk_coeff'][1:]@theta)
        cb=float(b['risk_coeff'][0]+b['risk_coeff'][1:]@theta)
        price=max(0.,float((a['reward']-b['reward'])/(ca-cb))) if abs(ca-cb)>1e-12 else 1.
        return min(1.,price) if abs(risk-old.SOFT)<1e-7 else max(1.,price)

    def query(self,theta,initial=None):
        if initial is not None and initial!=self.atlas['model'].initial:
            old.ja.rebase_initial(self.atlas,initial);self.archive.clear()
            self.model=replace(self.model,initial=initial)
        answer=self.certify(theta)
        if answer is not None:return {**answer,'direct_hit':True,'full_solve':False,'scalarized_calls':0,'pricing_rounds':0}
        changed=replace(self.atlas['model'],cost=self.model.cost+theta@self.directions)
        best_upper=np.inf;calls=0
        for step in range(self.pricing_rounds):
            p=old.primal(self.atlas,theta)
            price=self.proposed_price(theta,p)
            if not np.isfinite(price):break
            initial_action=None
            if self.warm and self.atlas['policies']:
                # Maximize this query's scalarized cached value; no input IDs.
                pol=max(self.atlas['policies'],key=lambda pol:pol['reward']-price*float(pol['risk_coeff'][0]+pol['risk_coeff'][1:]@theta))
                initial_action=pol['action']
            result=solve_scalarized(changed,price,initial_action=initial_action)
            calls+=1;self.last_price=price
            best_upper=min(best_upper,dual_bound(result)+old.ja.BELLMAN_TOLERANCE/(1-old.GAMMA))
            added=old.ja.add_policy(self.atlas,result,theta)
            p=old.primal(self.atlas,theta)
            if old.acceptable(p,best_upper):
                self.archive.append((theta.copy(),best_upper))
                action=np.argmax(result['policy'],axis=1)
                self.last_dual=next(i for i,q in enumerate(self.atlas['policies']) if np.array_equal(q['action'],action))
                return {'objective':p[0],'risk':p[1],'upper':float(best_upper),'gap':float(best_upper-p[0]),
                        'policy':old.policy_for(self.atlas,p),'certificate_type':'cached_primal_pricing',
                        'primal_i':p[2],'primal_j':p[3],'primal_weight':p[4],
                        'mixed_cache':p[2]!=p[3] and 1e-9<p[4]<1-1e-9,
                        'direct_hit':False,'full_solve':False,'scalarized_calls':calls,'pricing_rounds':step+1}
            if not added and p is not None:break
        # Same verified fallback as the frozen implementation. Include all
        # attempted pricing work in the caller's timer and oracle count.
        answer=super().query(theta,initial)
        answer['scalarized_calls']+=calls
        answer['pricing_rounds']=calls
        answer['direct_hit']=False
        return answer
