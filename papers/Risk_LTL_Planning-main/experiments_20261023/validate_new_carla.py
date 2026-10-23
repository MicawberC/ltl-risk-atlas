"""Reconstruct production atlas policies and compare independent LPs on new traces.

This is a correctness pass, not another closed-loop planning-time benchmark.
"""
from pathlib import Path
import os,sys,json,traceback
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  LTL_RISK_GAMMA='0.8',LTL_RISK_SOFT='1.0',LTL_RISK_HARD='2.0')
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
sys.path.insert(0,str(REPO/'reproduction/carla_local_risk'))
import benchmark_fixed_moving_replay as replay
from experiment import GAMMA,SOFT,HARD
assert (GAMMA,SOFT,HARD)==(.8,1.,2.), 'Validation parameters differ from the driving experiment'
original_prepare=replay.prepare_lp
replay.prepare_lp=lambda m,d:original_prepare(m,d,feasibility_tolerance=1e-9,solver_threads=1)
original_evaluate=replay.evaluate_policy
PROBABILITY_STATS={}
def checked_evaluate(model,policy):
    np=replay.np
    minimum=float(np.min(policy));error=float(np.max(np.abs(policy.sum(axis=1)-1)))
    assert minimum>=-1e-12 and error<1e-10,'Invalid reconstructed action probabilities'
    PROBABILITY_STATS['checks']=PROBABILITY_STATS.get('checks',0)+1
    PROBABILITY_STATS['minimum']=min(PROBABILITY_STATS.get('minimum',1.),minimum)
    PROBABILITY_STATS['max_row_sum_error']=max(PROBABILITY_STATS.get('max_row_sum_error',0.),error)
    return original_evaluate(model,policy)
replay.evaluate_policy=checked_evaluate

def validate_warm(reference,live,trial):
    """Repeat the LP path and independently evaluate its recovered policies."""
    np=replay.np;replace=replay.replace
    grid=int(reference['grid_n'])
    base=replay.build_grid(grid,'carla_crossing',labels_override=np.asarray(reference['labels'],dtype=np.int8),lane_only=True)
    directions=np.stack([np.where(np.all(base.xy==cell,axis=1),base.cost,0.) for cell in reference['risk_cells']])
    lp=None;rows=[]
    for recorded in live['trials'][trial]['rows']:
        obs=replay.risk_observation(reference,recorded)
        _,_,severity=replay.frame_parameters(reference,obs)
        theta=np.asarray([severity[int(x+grid*y)]-1. for x,y in reference['risk_cells']])
        x,y=map(int,recorded['ego_xy']);occupied=int(min(obs['risk_distance_m'])<=2.5)
        start=(x+grid*y)*8+occupied*4
        if lp is None:lp=replay.prepare_lp(replace(base,initial=start),directions)
        answer=replay.query_lp(lp,theta,initial=start)
        occupation=np.asarray(lp['solver'].getSolution().col_value[:lp['nx']]).reshape(base.states,base.actions)
        policy=replay.policy_from_occupation(occupation)
        assert np.min(policy)>=0 and np.max(np.abs(policy.sum(axis=1)-1))<1e-10
        value=replay.evaluate_policy(replace(base,initial=start,cost=base.cost+theta@directions),policy)
        assert abs(value['objective']-answer['objective'])<1e-5
        assert abs(value['risk']-answer['risk'])<1e-5 and value['risk']<=replay.HARD+1e-7
        rows.append({'tick':recorded['tick'],'objective':answer['objective'],'risk':answer['risk'],'atlas_forward':float(policy[start,5]),'evaluated_objective':float(value['objective']),'evaluated_risk':float(value['risk'])})
    return {'count':len(rows),'rows':rows,'scope':'Untimed reconstructed warm LP, independently evaluated policies'}

def main():
    out=HERE/'new_carla_validation';out.mkdir(exist_ok=True);summary=[]
    for p in sorted((HERE/'carla_results').glob('*.json')):
        live=json.loads(p.read_text('utf8'))
        if 'trials' not in live:continue
        ref=json.loads(p.with_name(p.stem+'_reference.json').read_text('utf8'))
        for index,trial in enumerate(live['trials']):
            if trial['method'] not in ('fixed_observed_hybrid','fixed_warm_lp') or not trial['rows']:continue
            failed_queries=[{'tick':r['tick'],'error':r.get('planner_failure','No returned objective')} for r in trial['rows'] if 'objective' not in r]
            returned=[r for r in trial['rows'] if 'objective' in r]
            validation_live={**live,'trials':list(live['trials'])}
            validation_live['trials'][index]={**trial,'rows':returned}
            path=out/f'{p.stem}_trial{index}.json'
            if path.exists():
                old=json.loads(path.read_text('utf8'))
                if old.get('passed') and old['summary'].get('probability_checks')==old['summary'].get('count'):summary.append(old['summary']);continue
            print('VALIDATE NEW TRACE',p.stem,index,flush=True)
            try:
                PROBABILITY_STATS.clear()
                result=(replay.run(ref,validation_live,index,hybrid=True,observed_prepare=True,quiet=True)
                        if trial['method']=='fixed_observed_hybrid' else validate_warm(ref,validation_live,index))
                mismatches=[]
                if len(returned) != len(result['rows']):
                    raise AssertionError('Stored and reconstructed trace lengths differ')
                for old,new in zip(returned,result['rows']):
                    if abs(old['objective']-new['objective'])>1e-5 or abs(old['risk']-new['risk'])>1e-5 or abs(old['forward_probability']-new['atlas_forward'])>1e-5:mismatches.append(old['tick'])
                assert PROBABILITY_STATS.get('checks')==result['count']
                item={'case':p.stem,'trial':index,'method':trial['method'],'count':result['count'],'passed':not mismatches,'stored_policy_mismatches':mismatches,'nonreturning_failed_queries':failed_queries,'drive_goal':trial['goal'],'probability_checks':PROBABILITY_STATS['checks'],'minimum_probability':PROBABILITY_STATS['minimum'],'max_probability_row_sum_error':PROBABILITY_STATS['max_row_sum_error']}
                result.update(passed=not mismatches,summary=item)
            except Exception as exc:
                item={'case':p.stem,'trial':index,'passed':False,'error':repr(exc)};result={'passed':False,'summary':item,'traceback':traceback.format_exc()}
            path.write_text(json.dumps(result,indent=2),encoding='utf8');summary.append(item)
    (out/'audit.json').write_text(json.dumps({'scope':'Returned policies only; nonreturning planning failures retained separately, not counted as certified policies or successful drives','parameters':{'gamma':GAMMA,'soft':SOFT,'hard':HARD},'traces':summary,'checked_points':sum(x.get('count',0) for x in summary),'nonreturning_failed_queries':sum(len(x.get('nonreturning_failed_queries',[])) for x in summary),'all_passed':bool(summary) and all(x['passed'] for x in summary)},indent=2),encoding='utf8')
    print('NEW TRACE VALIDATION FINISHED',flush=True)
if __name__=='__main__':main()
