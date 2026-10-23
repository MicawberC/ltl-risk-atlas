"""Matched post-residual-correction ablations. No production solver is patched.

All variants share the driver below, the same scalarized solver, occupancy
evaluation, Bellman-residual upper bound and fallback. no_mix disables cached
pair interpolation only; a complete CMDP solve can still return a randomized
policy. This distinction avoids changing the underlying optimization problem.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ.update(LTL_RISK_GAMMA='0.8',LTL_RISK_SOFT='1.0',LTL_RISK_HARD='2.0')
import argparse,hashlib,json,platform,sys,time,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
import psutil

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
import joint_price_atlas as ja
from experiment import HARD,SOFT,GAMMA,evaluate_policy,build_grid
from highs_reopt import prepare as lp_prepare,query as lp_query
from local_update_gate import risk_trace
from paper_atlas_comparison import make_case
from stratified import solve_scalarized

METHODS=('warm_lp','full','no_mix','no_regions','no_stream','forced_mdp','frozen_lp')

def primal(atlas,theta,mix=True):
    policies=atlas['policies'];risks=[float(p['risk_coeff'][0]+p['risk_coeff'][1:]@theta) for p in policies]
    best=None
    for i,p in enumerate(policies):
        if risks[i]<=HARD+1e-9:
            x=(p['reward']-max(0.,risks[i]-SOFT),risks[i],i,i,1.)
            if best is None or x[0]>best[0]:best=x
        if not mix:continue
        for j in range(i):
            span=risks[i]-risks[j]
            if abs(span)<1e-12:continue
            for boundary in (SOFT,HARD):
                w=(boundary-risks[j])/span
                if -1e-9<=w<=1+1e-9:
                    w=float(np.clip(w,0,1));risk=w*risks[i]+(1-w)*risks[j]
                    reward=w*p['reward']+(1-w)*policies[j]['reward']
                    x=(reward-max(0.,risk-SOFT),risk,i,j,w)
                    if risk<=HARD+1e-9 and (best is None or x[0]>best[0]):best=x
    return best

def policy_for(atlas,p):
    _,_,i,j,w=p
    occ=w*atlas['policies'][i]['occupation']+(1-w)*atlas['policies'][j]['occupation']
    occ=np.maximum(occ,0.)
    totals=occ.sum(axis=1)
    pi=np.divide(occ,totals[:,None],out=np.zeros_like(occ),where=totals[:,None]>1e-12)
    pi[totals<=1e-12,0]=1.
    return pi

def acceptable(p,upper):
    return p is not None and p[1]<=HARD+1e-8 and -1e-6<=upper-p[0]<=max(1e-4,.01*abs(p[0]))+1e-7

def cache_bytes(atlas):
    total=0
    for p in atlas['policies']:
        for v in p.values():
            if isinstance(v,np.ndarray):total+=v.nbytes
        # Include retained sparse LU factors, exclude Python/SuperLU overhead.
        for factor in (p['lu'].L,p['lu'].U):
            total+=factor.data.nbytes+factor.indices.nbytes+factor.indptr.nbytes
    return total

class Controller:
    def __init__(self,model,directions,theta,method):
        self.method=method;self.model=model;self.directions=directions
        self.lp=None;self.archive=[];self.last_dual=None;self.last_price=None
        self.atlas=ja.prepare(model,directions,observed_theta=theta)
    def certify(self,theta):
        a=self.atlas;p=primal(a,theta,self.method!='no_mix')
        if p is None:return None
        stream=self.method!='no_stream'
        regions=self.method!='no_regions'
        def answer(upper,kind,idx=None,price=None):
            if not acceptable(p,upper):return None
            if stream and kind!='monotone':self.archive.append((theta.copy(),upper))
            if idx is not None:self.last_dual=idx;self.last_price=price
            return {'objective':p[0],'risk':p[1],'upper':float(upper),'gap':float(upper-p[0]),
                    'policy':policy_for(a,p),'certificate_type':kind,'primal_i':p[2],
                    'primal_j':p[3],'primal_weight':p[4],'mixed_cache':p[2]!=p[3] and 1e-9<p[4]<1-1e-9}
        if stream:
            bounds=[ub for anchor,ub in self.archive if np.all(anchor<=theta+1e-12)]
            if bounds:
                result=answer(min(bounds),'monotone')
                if result:return result
        def prices_for(pol):
            if not regions:
                return sorted(set([0.,1.,float(pol['source_price'])]+([] if self.last_price is None else [self.last_price])))
            interval=ja.price_interval(pol,theta)
            if interval is None:return []
            lo,hi=interval
            return sorted(set(t for t in [lo,hi,1.,self.last_price] if t is not None and np.isfinite(t) and lo-1e-8<=t<=hi+1e-8))
        if stream and self.last_dual is not None:
            idx=self.last_dual;prices=prices_for(a['policies'][idx])
            if prices:
                upper,price=min((ja.dual_bound(a['policies'][idx],theta,t),t) for t in prices)
                result=answer(upper,'previous_region' if regions else 'previous_price',idx,price)
                if result:return result
        bounds=[]
        for idx,pol in enumerate(a['policies']):
            for price in prices_for(pol):bounds.append((ja.dual_bound(pol,theta,price),idx,price))
        if bounds:
            upper,idx,price=min(bounds)
            return answer(upper,'region_scan' if regions else 'fixed_price_scan',idx,price)
        return None
    def forced_oracle(self,theta):
        pols=sorted(self.atlas['policies'],key=lambda p:float(np.linalg.norm(p['source_theta']-theta)))
        price=next((p['source_price'] for p in pols if p['source_price']>1e-10),1.)
        current=replace(self.atlas['model'],cost=self.model.cost+theta@self.directions)
        return int(ja.add_policy(self.atlas,solve_scalarized(current,price),theta))
    def query(self,theta,initial=None):
        if initial is not None and initial!=self.atlas['model'].initial:
            ja.rebase_initial(self.atlas,initial);self.archive.clear()
            self.model=replace(self.model,initial=initial)
        forced=self.method=='forced_mdp'
        if forced:self.forced_oracle(theta)
        answer=self.certify(theta)
        direct=answer is not None;oracles=int(forced);full=False;fallback_randomized=False
        if answer is None and self.method=='frozen_lp':
            if self.lp is None:self.lp=lp_prepare(self.model,self.directions,feasibility_tolerance=1e-9,solver_threads=1)
            exact=lp_query(self.lp,theta,initial=self.model.initial)
            occ=np.asarray(self.lp['solver'].getSolution().col_value[:self.lp['nx']]).reshape(self.model.states,self.model.actions)
            occ=np.maximum(occ,0.)
            total=occ.sum(axis=1);pi=np.divide(occ,total[:,None],out=np.zeros_like(occ),where=total[:,None]>1e-12);pi[total<=1e-12,0]=1
            answer={**exact,'upper':exact['objective']+1e-6,'policy':pi,'certificate_type':'frozen_lp_rescue','mixed_cache':False}
            full=True
        elif answer is None:
            if not forced:self.forced_oracle(theta);oracles+=1
            answer=self.certify(theta)
            if answer is None:
                full=True;ext=ja.expand(self.atlas,theta);answer=self.certify(theta)
                oracles+=int(ext['fresh']['evaluations'])
                # Cached mixing ablation preserves full CMDP feasibility: a
                # freshly solved randomized policy is allowed, but not reused.
                if answer is None:
                    fresh=ext['fresh'];policies=[]
                    for price in fresh['prices']:
                        matches=[p for p in self.atlas['policies'] if abs(p['source_price']-price)<1e-10 and np.allclose(p['source_theta'],theta,atol=1e-12)]
                        if not matches:
                            changed=replace(self.model,cost=self.model.cost+theta@self.directions)
                            oracle=solve_scalarized(changed,price);oracles+=1
                            # Duplicate policy may have an earlier source price.
                            occ=oracle['occupation'];policies.append({'occupation':occ,'risk':oracle['risk'],'reward':oracle['reward']})
                        else:
                            p=matches[-1];policies.append({'occupation':p['occupation'],'risk':float(p['risk_coeff'][0]+p['risk_coeff'][1:]@theta),'reward':p['reward']})
                    if fresh['mixed']:
                        lo,hi=policies;w=(fresh['target']-hi['risk'])/(lo['risk']-hi['risk']);occ=w*lo['occupation']+(1-w)*hi['occupation']
                    else:occ=policies[-1]['occupation']
                    occ=np.maximum(occ,0.)
                    total=occ.sum(axis=1);pi=np.divide(occ,total[:,None],out=np.zeros_like(occ),where=total[:,None]>1e-12);pi[total<=1e-12,0]=1.
                    answer={**fresh,'policy':pi,'certificate_type':'fresh_cmdp','mixed_cache':False}
                    fallback_randomized=bool(fresh['mixed'])
        if answer is None:raise RuntimeError('No certified answer after full expansion')
        if not acceptable((answer['objective'],answer['risk']),answer['upper']):raise AssertionError('Invalid final certificate')
        return {**answer,'direct_hit':direct,'full_solve':full,'scalarized_calls':oracles,'fresh_randomized':fallback_randomized}

def make_synthetic(case,stream,seed,count,magnitude=.8,dimension=5):
    model,dirs,positions=make_case(case)
    if dimension!=5:
        if dimension<5:dirs=dirs[:dimension];positions=positions[:dimension]
        else:
            # Explicitly extend support over existing positive-cost locations.
            cells=np.unique(model.xy[model.cost>0],axis=0)
            chosen=cells[:dimension]
            if len(chosen)<dimension:raise ValueError('Not enough positive risk cells')
            dirs=np.stack([np.where(np.all(model.xy==p,axis=1),model.cost,0) for p in chosen]);positions=chosen.tolist()
    theta=risk_trace(stream,count,seed,len(dirs),magnitude=magnitude)
    return model,dirs,theta,[model.initial]*count,{'case':case,'stream':stream,'seed':seed,'risk_cells':positions,'magnitude':magnitude,'dimension':len(dirs)}

def make_replay(path,reference_path,trial,stride=1):
    live=json.loads(Path(path).read_text('utf8'));ref=json.loads(Path(reference_path).read_text('utf8'))
    sys.path.insert(0,str(REPO/'reproduction/carla_local_risk'))
    from benchmark_dynamic_live_replay import risk_observation
    from dynamic_product import frame_parameters
    n=int(ref['grid_n']);base=build_grid(n,'carla_crossing',labels_override=np.asarray(ref['labels'],dtype=np.int8),lane_only=True)
    directions=np.stack([np.where(np.all(base.xy==cell,axis=1),base.cost,0) for cell in ref['risk_cells']])
    thetas=[];initials=[]
    for row in live['trials'][trial]['rows'][::stride]:
        obs=risk_observation(ref,row);_,_,severity=frame_parameters(ref,obs)
        thetas.append([severity[int(x+n*y)]-1 for x,y in ref['risk_cells']])
        x,y=map(int,row['ego_xy']);occupied=int(min(obs['risk_distance_m'])<=2.5);initials.append((x+n*y)*8+occupied*4)
    return replace(base,initial=initials[0]),directions,np.asarray(thetas),initials,{'case':'replay:'+Path(path).stem+':'+str(trial),'stride':stride,'trace_source':str(path),'reference_source':str(reference_path),'risk_cells':ref['risk_cells']}

def run(model,dirs,thetas,initials,meta,method,output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();ctrl=None;lp=None
    if method=='warm_lp':lp=lp_prepare(model,dirs,feasibility_tolerance=1e-9,solver_threads=1)
    else:ctrl=Controller(model,dirs,thetas[0],method)
    setup=time.perf_counter()-started;rows=[];policies=[];proc=psutil.Process();rss=proc.memory_info().rss
    for index,(theta,initial) in enumerate(zip(thetas,initials)):
        t=time.perf_counter()
        try:
            if method=='warm_lp':
                a=lp_query(lp,theta,initial=initial)
                occ=np.asarray(lp['solver'].getSolution().col_value[:lp['nx']]).reshape(model.states,model.actions)
                occ=np.maximum(occ,0.)
                total=occ.sum(axis=1);pi=np.divide(occ,total[:,None],out=np.zeros_like(occ),where=total[:,None]>1e-12);pi[total<=1e-12,0]=1
                a.update(policy=pi,upper=a['objective']+1e-6,direct_hit=False,full_solve=True,scalarized_calls=0,mixed_cache=False,certificate_type='warm_lp')
            else:a=ctrl.query(theta,initial)
            elapsed=time.perf_counter()-t;policy=a.pop('policy')
            if np.min(policy)<-1e-12 or np.max(np.abs(policy.sum(axis=1)-1))>1e-10:raise AssertionError('Returned action probabilities are invalid')
            policies.append(policy)
            row={k:(v.item() if isinstance(v,np.generic) else v) for k,v in a.items() if k not in ('prices','oracle_results')}
            row.update(index=index,theta=theta.tolist(),initial=int(initial),seconds=elapsed,cache_size=0 if ctrl is None else len(ctrl.atlas['policies']),archive_size=0 if ctrl is None else len(ctrl.archive),cache_array_bytes=0 if ctrl is None else cache_bytes(ctrl.atlas),rss_mb=proc.memory_info().rss/2**20,status='ok')
        except Exception as exc:
            row={'index':index,'seconds':time.perf_counter()-t,'status':'failed','error':repr(exc),'traceback':traceback.format_exc()};policies.append(None)
        rows.append(row);rss=max(rss,proc.memory_info().rss)
    # Independent evaluation excluded from setup/query timing, all methods.
    for row,pi,theta,initial in zip(rows,policies,thetas,initials):
        if pi is None:continue
        evaluated=evaluate_policy(replace(model,initial=initial,cost=model.cost+theta@dirs),pi)
        row['evaluated_objective']=float(evaluated['objective']);row['evaluated_risk']=float(evaluated['risk'])
        row['evaluation_ok']=bool(abs(evaluated['risk']-row['risk'])<1e-5 and abs(evaluated['objective']-row['objective'])<1e-5 and evaluated['risk']<=HARD+1e-7)
    times=np.asarray([r['seconds'] for r in rows]);result={**meta,'method':method,'states':model.states,'actions':model.actions,'count':len(rows),'theta_sha256':hashlib.sha256(np.asarray(thetas,dtype='<f8').tobytes()+np.asarray(initials,dtype='<i8').tobytes()).hexdigest(),'setup_seconds':setup,'online_seconds':float(times.sum()),'total_seconds':float(setup+times.sum()),'p95_ms':float(np.percentile(times,95)*1000),'max_ms':float(times.max()*1000),'peak_sampled_rss_mb':rss/2**20,'failures':sum(r['status']!='ok' for r in rows),'validation_failures':sum(not r.get('evaluation_ok',False) for r in rows),'direct_hits':sum(r.get('direct_hit',False) for r in rows),'full_solves':sum(r.get('full_solve',False) for r in rows),'mixed_cache_returns':sum(r.get('mixed_cache',False) for r in rows),'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'gamma':GAMMA,'soft':SOFT,'hard':HARD},'rows':rows}
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(meta['case'],method,'seconds',round(result['total_seconds'],3),'failures',result['failures'],'hits',result['direct_hits'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--case',default='24:middle_fast_crossing:0');p.add_argument('--stream',default='walk');p.add_argument('--seed',type=int,default=73044);p.add_argument('--count',type=int,default=20);p.add_argument('--magnitude',type=float,default=.8);p.add_argument('--dimension',type=int,default=5);p.add_argument('--output',required=True);p.add_argument('--replay');p.add_argument('--reference');p.add_argument('--trial',type=int,default=0);p.add_argument('--stride',type=int,default=1);a=p.parse_args()
    data=make_replay(a.replay,a.reference,a.trial,a.stride) if a.replay else make_synthetic(a.case,a.stream,a.seed,a.count,a.magnitude,a.dimension)
    run(*data,a.method,a.output)
if __name__=='__main__':main()
