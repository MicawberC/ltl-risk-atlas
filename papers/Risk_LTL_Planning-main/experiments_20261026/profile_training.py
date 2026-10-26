"""Reproduce the training profile only, never concurrently with timing runs."""
from pathlib import Path
import sys,json,cProfile,pstats,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from optimized_controller import old

def main():
    out=Path(__file__).resolve().parent/'profile_repeat';out.mkdir(exist_ok=True)
    m,d,theta,_,_=old.make_synthetic('16:barrier_hard_active:0','iid',73044,20)
    profile=cProfile.Profile();profile.enable();c=old.Controller(m,d,theta[0],'full');rows=[]
    for index,t in enumerate(theta):
        started=time.perf_counter();answer=c.query(t)
        rows.append({'index':index,'seconds':time.perf_counter()-started,'scalarized_calls':answer['scalarized_calls'],'full_solve':answer['full_solve'],'cache_size':len(c.atlas['policies'])})
    profile.disable()
    with (out/'profile.txt').open('w',encoding='utf8') as stream:pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(45)
    text=out/'profile.txt';text.write_text(text.read_text('utf8').rstrip()+'\n',encoding='utf8')
    (out/'queries.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print('Training profile written; profiling overhead is not benchmark time.')
if __name__=='__main__':main()
