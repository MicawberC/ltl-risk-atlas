"""Classify recorded nonreturning planner failures with a fresh tight LP.

This diagnostic neither changes the driving data nor counts failures as passes.
"""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_new_carla import replay,HERE

def main():
    np=replay.np;replace=replay.replace;rows=[]
    for path in sorted((HERE/'carla_results').glob('*.json')):
        live=json.loads(path.read_text('utf8'))
        if 'trials' not in live:continue
        ref=json.loads(path.with_name(path.stem+'_reference.json').read_text('utf8'))
        grid=int(ref['grid_n'])
        base=replay.build_grid(grid,'carla_crossing',labels_override=np.asarray(ref['labels'],dtype=np.int8),lane_only=True)
        directions=np.stack([np.where(np.all(base.xy==cell,axis=1),base.cost,0.) for cell in ref['risk_cells']])
        for index,trial in enumerate(live['trials']):
            for measured in trial['rows']:
                if 'objective' in measured:continue
                # Failure rows retain measured distances but not walker poses.
                obs={'risk_distance_m':measured['risk_distance_m']}
                _,_,severity=replay.frame_parameters(ref,obs)
                theta=np.asarray([severity[int(x+grid*y)]-1 for x,y in ref['risk_cells']])
                x,y=map(int,measured['ego_xy']);occupied=int(min(obs['risk_distance_m'])<=2.5)
                start=(x+grid*y)*8+occupied*4
                item={'case':path.stem,'trial':index,'method':trial['method'],'tick':measured['tick'],'initial':start,'theta':theta.tolist(),'recorded_error':measured.get('planner_failure')}
                try:
                    lp=replay.prepare_lp(replace(base,initial=start),directions)
                    answer=replay.query_lp(lp,theta,initial=start)
                    item.update(fresh_lp_status='Optimal',objective=answer['objective'],risk=answer['risk'])
                except Exception as exc:item['fresh_lp_status']=str(exc)
                rows.append(item)
    result={'scope':'Untimed fresh LP diagnoses; Infeasible is a solver classification, not an exact-arithmetic infeasibility proof','rows':rows,'count':len(rows)}
    (HERE/'failed_query_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
