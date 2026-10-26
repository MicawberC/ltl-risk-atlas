"""Rerun both members of one pair that overlapped an interim JSON audit.

Run AFTER the formal suite and all initial analysis/plot processes end.
Original records are retained separately; the new outcome is used regardless
of whether it improves or worsens the speed ratio. No algorithm is changed.
"""
from pathlib import Path
import json,shutil,subprocess,time
HERE=Path(__file__).resolve().parent
def main():
    pair='replay_double_t0_r2';diagnostic=HERE/'diagnostics/interim_audit_overlap';diagnostic.mkdir(parents=True,exist_ok=True)
    for method in ('warm_lp','optimized'):
        path=HERE/'results'/f'{pair}_{method}.json';old=json.loads(path.read_text('utf8'))
        if old.get('isolated_pair_recheck'):continue
        assert old['run_complete'] and old['failures']==0
        shutil.copy2(path,diagnostic/path.name)
        logfile=path.with_suffix('.log')
        if logfile.exists():shutil.copy2(logfile,diagnostic/logfile.name)
        started=time.time()
        with logfile.open('wb') as stream:subprocess.run(old['command'],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
        data=json.loads(path.read_text('utf8'))
        data.update({k:old[k] for k in ('group','repeat','pair_id','command','optimized_source_sha256')})
        data.update(run_complete=True,isolated_pair_recheck=True,wall_seconds=time.time()-started)
        path.write_text(json.dumps(data,indent=2),encoding='utf8')
        print(method,data['total_seconds'],'failures',data['failures'],flush=True)
    (diagnostic/'EXCLUDED.json').write_text(json.dumps({'reason':'The original LP timing overlapped an interim JSON audit. Both methods rerun as a complete same-input pair after all other timed jobs and plotting ended; all source parameters and thresholds unchanged. Original records retained and excluded from formal summary.','pair_id':pair},indent=2),encoding='utf8')
if __name__=='__main__':main()
