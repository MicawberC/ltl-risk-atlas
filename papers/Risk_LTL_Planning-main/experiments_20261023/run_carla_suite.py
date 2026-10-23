"""Additional closed-loop routes, preserved failures, three repeats per case.

Run only after the CPU timing suite ends. Start a CARLA 0.9.16 server first.
The original controller/solver code is used; model and actuation limitations
remain unchanged. A route-specific reference is recorded for later replays.
"""
import argparse,json,subprocess,sys,time,traceback,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
CARLA_CODE=REPO/'reproduction/carla_local_risk'
sys.path.insert(0,str(CARLA_CODE));sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
os.environ.update(LTL_RISK_GAMMA='0.8',LTL_RISK_SOFT='1.0',LTL_RISK_HARD='2.0')
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

def main():
    import carla
    from capture_crossing import choose_road
    from closed_loop_dynamic import road_labels
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'carla_results');p.add_argument('--port',type=int,default=2000);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    cases=[('Town01',0,False),('Town01',0,True),('Town01',1,False),('Town02',0,False)]
    for town,route,two in cases:
        name=f'{town}_route{route}_'+('double' if two else 'single');out=a.out/(name+'.json')
        if out.exists():
            try:
                old=json.loads(out.read_text('utf8'))
                if len(old.get('trials',[]))==6:continue
            except Exception:pass
        started=time.time()
        try:
            client=carla.Client('127.0.0.1',a.port);client.set_timeout(120.)
            world=client.get_world()
            if not world.get_map().name.endswith('/'+town):world=client.load_world(town)
            spawn,crossing,_=choose_road(world,route)
            labels,cells,drivable=road_labels(world,crossing.transform.location,crossing.transform.get_forward_vector(),crossing.transform.get_right_vector())
            center=crossing.transform.location;forward=crossing.transform.get_forward_vector();right=crossing.transform.get_right_vector()
            ref={'grid_n':20,'labels':labels,'risk_cells':cells,'town':town,'route_index':route,'drivable':drivable,
                 'crossing_xyz':[center.x,center.y,center.z],'forward_xy':[forward.x,forward.y],'right_xy':[right.x,right.y]}
            (a.out/(name+'_reference.json')).write_text(json.dumps(ref,indent=2),encoding='utf8')
            script=CARLA_CODE/'closed_loop_dynamic.py'
            code=('import sys,runpy,functools;sys.path.insert(0,'+repr(str(REPO/'reproduction/adaptive_risk_ltl'))+');'
                  'sys.path.insert(0,'+repr(str(CARLA_CODE))+');import highs_reopt;'
                  'highs_reopt.prepare=functools.partial(highs_reopt.prepare,feasibility_tolerance=1e-9,solver_threads=1);'
                  'runpy.run_path('+repr(str(script))+',run_name="__main__")')
            cmd=[sys.executable,'-c',code,'--town',town,'--route-index',str(route),'--port',str(a.port),'--methods','fixed_observed_hybrid','fixed_warm_lp','--repeats','3','--max-ticks',str(1200 if two else 900),'--plan-every','1','--action-every','10','--sample-policy','--output',str(out)]
            if two:cmd.append('--second-walker')
            print('CARLA CASE',name,flush=True)
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=2400)
            (a.out/(name+'.log')).write_bytes(result.stdout)
            (a.out/(name+'_execution.json')).write_text(json.dumps({'command':cmd,'returncode':result.returncode,'wall_seconds':time.time()-started},indent=2),encoding='utf8')
            print(name,'exit',result.returncode,flush=True)
        except Exception as exc:
            (a.out/(name+'_execution.json')).write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.time()-started},indent=2),encoding='utf8')
            print(name,'FAILED',repr(exc),flush=True)
if __name__=='__main__':main()
