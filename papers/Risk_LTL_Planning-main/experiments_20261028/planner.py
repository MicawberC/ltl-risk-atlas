"""Common live-driving adapter; the frozen optimization algorithm is unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(LTL_RISK_GAMMA='0.8', LTL_RISK_SOFT='1.0', LTL_RISK_HARD='2.0')
from pathlib import Path
import sys
import numpy as np
from dataclasses import replace
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
sys.path.insert(0,str(HERE.parent/'experiments_20261023'))
sys.path.insert(0,str(HERE.parent/'experiments_20261026'))
import controlled_experiment as old
from optimized_controller import OptimizedController
from highs_reopt import prepare as prepare_lp, query as query_lp
from experiment import GAMMA,SOFT,HARD,build_grid
assert (GAMMA,SOFT,HARD)==(.8,1.,2.)

def build_model(reference):
    base=build_grid(int(reference['grid_n']),'carla_crossing',
                    labels_override=np.asarray(reference['labels'],dtype=np.int8),lane_only=True)
    directions=np.stack([np.where(np.all(base.xy==cell,axis=1),base.cost,0.)
                         for cell in reference['risk_cells']])
    return base,directions

class Planner:
    def __init__(self,base,directions,theta,initial,method):
        self.base=base;self.directions=directions;self.method=method
        model=replace(base,initial=int(initial))
        if method=='optimized':self.controller=OptimizedController(model,directions,theta)
        elif method=='full':self.controller=old.Controller(model,directions,theta,'full')
        elif method=='warm_lp':self.lp=prepare_lp(model,directions,feasibility_tolerance=1e-9,solver_threads=1)
        else:raise ValueError(method)
    def query(self,theta,initial):
        if self.method=='warm_lp':
            solved=query_lp(self.lp,theta,initial=int(initial))
            if not solved['basis_valid']:raise RuntimeError('LP returned invalid basis')
            occ=np.asarray(self.lp['solver'].getSolution().col_value[:self.lp['nx']]).reshape(self.base.states,self.base.actions)
            occ=np.maximum(occ,0.);totals=occ.sum(axis=1)
            policy=np.divide(occ,totals[:,None],out=np.zeros_like(occ),where=totals[:,None]>1e-12)
            policy[totals<=1e-12,0]=1.
            answer={**solved,'policy':policy,'upper':float(solved['objective'])+1e-5,
                    'gap':1e-5,'certificate_type':'LP_optimal_status',
                    'direct_hit':False,'full_solve':True,'scalarized_calls':0,'pricing_rounds':0}
        else:answer=self.controller.query(np.asarray(theta),initial=int(initial))
        policy=np.asarray(answer['policy'],dtype=np.float64)
        if not np.all(np.isfinite(policy)) or policy.min()<-1e-12 or np.max(abs(policy.sum(axis=1)-1))>1e-10:
            raise RuntimeError('Invalid action probability distribution')
        if not all(np.isfinite(float(answer[k])) for k in ('objective','risk','upper','gap')):
            raise RuntimeError('Non-finite certificate')
        if answer['risk']>HARD+1e-7:raise RuntimeError('Hard risk gate failed')
        if self.method!='warm_lp' and not (-1e-6<=answer['upper']-answer['objective']<=max(1e-4,.01*abs(answer['objective']))+1e-7):
            raise RuntimeError('Objective gap gate failed')
        return answer
