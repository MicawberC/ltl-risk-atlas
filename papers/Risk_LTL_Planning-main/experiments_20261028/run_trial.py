"""One live CARLA drive with the common fixed-product planning adapter.

High-level model and forward/wait low-level control match the earlier study.
Returned full policies and unsuccessful-query timings are now preserved.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.update(LTL_RISK_GAMMA='0.8',LTL_RISK_SOFT='1.0',LTL_RISK_HARD='2.0')
from pathlib import Path
import argparse,hashlib,io,json,sys,time,traceback,zipfile
from dataclasses import replace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
sys.path.insert(0,str(REPO/'reproduction/carla_local_risk'))
sys.path.insert(0,str(HERE))
import carla
import numpy as np
import psutil
from planner import Planner,build_model,GAMMA,SOFT,HARD
from closed_loop_dynamic import road_labels,observation,start_state
from capture_crossing import choose_road,add
from dynamic_product import frame_parameters

def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.writing')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    temporary.replace(path)

def recorded_path(path):
    try:return path.resolve().relative_to(HERE).as_posix()
    except ValueError:return str(path.resolve())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--town',required=True);ap.add_argument('--route-index',type=int,required=True)
    ap.add_argument('--second-walker',action='store_true');ap.add_argument('--max-ticks',type=int,required=True)
    ap.add_argument('--port',type=int,default=2060);ap.add_argument('--method',choices=['optimized','full','warm_lp'],required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--repetition',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--policies',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    args=ap.parse_args()
    case=f'{args.town}_route{args.route_index}_'+('double' if args.second_walker else 'single')
    result={'case':case,'method':args.method,'seed':args.seed,'repetition':args.repetition,
            'trial_complete':False,'parameters':{'gamma':GAMMA,'soft':SOFT,'hard':HARD},
            'max_ticks':args.max_ticks,'fixed_delta_seconds':.05,'plan_every_ticks':1,'action_every_ticks':10,
            'reference_path':recorded_path(args.reference),'policy_archive':recorded_path(args.policies),
            'scope':'Fixed product, observed initialization, moving initial state; lane-only forward/wait execution',
            'automaton_state_scope':'Legacy start_state resets goal/bad to zero at each observation; not full execution-history monitoring',
            'rows':[],'goal':False,'collision':False,'planner_failure':None}
    save(args.output,result)
    client=carla.Client('127.0.0.1',args.port);client.set_timeout(120.)
    world=None;original=None;actors=[];events=[];planner=None
    started=time.perf_counter();rng=np.random.default_rng(args.seed);proc=psutil.Process();peak=proc.memory_info().rss/2**20
    try:
        world=client.get_world()
        if not world.get_map().name.endswith('/'+args.town):raise RuntimeError('Unexpected server map')
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
        world.apply_settings(settings)
        # The scheduler starts a fresh server for each drive. Avoid an in-process
        # reload: the Windows 0.9.16 pilot crashed after its first controlled tick.
        original=world.get_settings()
        world.set_weather(carla.WeatherParameters.ClearNoon)
        spawn,crossing,_=choose_road(world,args.route_index)
        center=crossing.transform.location;forward=crossing.transform.get_forward_vector();right=crossing.transform.get_right_vector()
        labels,cells,drivable=road_labels(world,center,forward,right)
        reference={'grid_n':20,'labels':labels,'risk_cells':cells,'drivable':drivable,
                   'town':args.town,'route_index':args.route_index,
                   'crossing_xyz':[center.x,center.y,center.z],
                   'forward_xy':[forward.x,forward.y],'right_xy':[right.x,right.y]}
        if args.reference.exists() and json.loads(args.reference.read_text('utf8'))!=reference:
            raise RuntimeError('Road reference differs between methods')
        save(args.reference,reference)
        began=time.perf_counter();base,directions=build_model(reference)
        result['fixed_product_build_s']=time.perf_counter()-began
        result['simulator_version']=client.get_server_version()
        result['reference_sha256']=hashlib.sha256(args.reference.read_bytes()).hexdigest()
        library=world.get_blueprint_library()
        vehicles=sorted(library.filter('vehicle.*'),key=lambda x:x.id)
        vehicle_bp=next((b for b in vehicles if b.id=='vehicle.tesla.model3'),vehicles[0])
        ego=world.try_spawn_actor(vehicle_bp,spawn)
        if ego is None:raise RuntimeError('Unable to spawn ego')
        actors.append(ego)
        sensor=world.spawn_actor(library.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        actors.append(sensor)
        sensor.listen(lambda event:events.append({'frame':event.frame,'other_actor':event.other_actor.type_id,'impulse':event.normal_impulse.length()}))
        walker_bp=sorted(library.filter('walker.pedestrian.*'),key=lambda x:x.id)[0]
        if walker_bp.has_attribute('is_invincible'):walker_bp.set_attribute('is_invincible','false')
        first_start=add(center,right,-5.);first_start.z+=1.
        first=world.try_spawn_actor(walker_bp,carla.Transform(first_start,carla.Rotation()))
        if first is None:raise RuntimeError('Unable to spawn first walker')
        actors.append(first);walkers=[first]
        first.apply_control(carla.WalkerControl(direction=carla.Vector3D(right.x,right.y,0.),speed=1.3))
        world.tick();go=False
        args.policies.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.policies,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for tick in range(args.max_ticks):
                if args.second_walker and tick==100:
                    position=add(add(center,forward,3.),right,5.);position.z+=1.
                    walker=world.try_spawn_actor(walker_bp,carla.Transform(position,carla.Rotation()))
                    if walker is None:raise RuntimeError('Unable to spawn second walker')
                    actors.append(walker);walkers.append(walker)
                    walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(-right.x,-right.y,0.),speed=1.2))
                obs=observation(walkers,center,forward,right,cells)
                initial,xy,longitudinal=start_state(ego,center,forward,right,obs)
                began=time.perf_counter()
                _,_,severity=frame_parameters(reference,obs)
                theta=np.asarray([severity[x+20*y]-1. for x,y in cells])
                product_s=time.perf_counter()-began
                row={'tick':tick,'sim_time_s':tick*.05,'ego_xy':xy,'initial':int(initial),
                     'longitudinal_m':longitudinal,'theta':theta.tolist(),
                     'risk_distance_m':obs['risk_distance_m'],'walkers_xyz':obs['walkers_xyz'],
                     'product_s':product_s,'world_frame':world.get_snapshot().frame}
                began=time.perf_counter();initializing=planner is None
                try:
                    if initializing:planner=Planner(base,directions,theta,initial,args.method)
                    answer=planner.query(theta,initial)
                    row['plan_s']=time.perf_counter()-began
                except Exception as exc:
                    row.update(plan_s=time.perf_counter()-began,initializing=initializing,
                               planner_failure=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                    result['rows'].append(row);result['planner_failure']=row['planner_failure']
                    print(case,args.method,'PLANNER FAILURE',tick,row['planner_failure'],flush=True)
                    break
                policy=np.asarray(answer['policy'],dtype=np.float64,order='C')
                key=f'{tick:06d}.npy';buffer=io.BytesIO();np.save(buffer,policy,allow_pickle=False)
                archive.writestr(key,buffer.getvalue())
                forward_probability=float(policy[initial,5]);action_due=tick%10==0
                action_random=float(rng.random()) if action_due else None
                if action_due:go=bool(action_random<forward_probability)
                row.update(initializing=initializing,objective=float(answer['objective']),risk=float(answer['risk']),
                           upper=float(answer['upper']),gap=float(answer['gap']),policy_key=key,
                           policy_sha256=hashlib.sha256(policy.tobytes()).hexdigest(),
                           forward_probability=forward_probability,action_due=action_due,action_random=action_random,go=go,
                           certificate_type=answer.get('certificate_type'),direct_hit=bool(answer.get('direct_hit',False)),
                           full_solve=bool(answer.get('full_solve',False)),scalarized_calls=int(answer.get('scalarized_calls',0)),
                           pricing_rounds=int(answer.get('pricing_rounds',0)),
                           mixed_cache=bool(answer.get('mixed_cache',False)),
                           cached_policies=(len(planner.controller.atlas['policies']) if args.method!='warm_lp' else 0))
                result['rows'].append(row)
                peak=max(peak,proc.memory_info().rss/2**20)
                if tick%50==0:
                    save(args.output,result)
                    print(case,args.method,'tick',tick,'x',round(longitudinal,3),'risk',round(row['risk'],6),'plan_ms',round(1000*row['plan_s'],3),flush=True)
                speed=ego.get_velocity().length()
                control=(carla.VehicleControl(throttle=float(np.clip(.30*(3.-speed),0.,.5)),brake=0.,steer=0.)
                         if go else carla.VehicleControl(throttle=0.,brake=1.,steer=0.))
                ego.apply_control(control);world.tick()
                _,_,after=start_state(ego,center,forward,right,observation(walkers,center,forward,right,cells))
                if after>=7.5:
                    result['goal']=True;result['goal_sim_time_s']=(tick+1)*.05;break
                if events:break
        time.sleep(.1)  # permit asynchronous collision callbacks from the final tick
        result.update(trial_complete=True,collision=bool(events),collision_events=events,
                      elapsed_sim_time_s=(tick+1)*.05,peak_process_rss_mb=peak,
                      planning_calls=len(result['rows']),returned_queries=sum('policy_key' in r for r in result['rows']),
                      planning_time_s=sum(r['plan_s'] for r in result['rows']),
                      input_update_time_s=sum(r['product_s'] for r in result['rows']),
                      wall_time_s=time.perf_counter()-started,
                      outcome=('collision' if events else 'goal' if result['goal'] else 'planner_failure' if result['planner_failure'] else 'timeout'))
    except Exception as exc:
        result.update(infrastructure_failure=repr(exc),traceback=traceback.format_exc(),wall_time_s=time.perf_counter()-started)
    finally:
        save(args.output,result)
        for actor in reversed(actors):
            try:
                if actor.is_alive:actor.destroy()
            except Exception:pass
        if world is not None:
            try:world.tick()
            except Exception:pass
    print('DRIVE END',case,args.method,result.get('outcome'),result.get('returned_queries'),flush=True)
    return 0 if result['trial_complete'] else 2
if __name__=='__main__':raise SystemExit(main())
