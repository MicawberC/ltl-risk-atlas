"""Record final artifact hashes and check required audits before publication."""
from pathlib import Path
import ast,hashlib,json,datetime
HERE=Path(__file__).resolve().parent

def main():
    audits={}
    for folder,expected,queries in [('analysis',225,5340),('historical_analysis',72,1440),('replay_analysis',36,22113)]:
        a=json.loads((HERE/folder/'audit.json').read_text('utf8'))
        assert a['completed_runs']==expected and a['all_complete']
        assert a['query_records']==queries
        assert not a['violations'] and not a['unpaired_runs']
        audits[folder]={k:a[k] for k in ['completed_runs','query_records','all_complete']}
    historic=json.loads((HERE/'historical_input_audit.json').read_text('utf8'))
    assert historic['runs']==72 and historic['all_identical']
    live=json.loads((HERE/'new_carla_validation/audit.json').read_text('utf8'))
    assert len(live['traces'])==24 and live['all_passed']
    assert all(t.get('probability_checks')==t['count'] for t in live['traces'])
    assert live['parameters']=={'gamma':.8,'soft':1.,'hard':2.}
    drives=[]
    for p in (HERE/'carla_results').glob('*.json'):
        d=json.loads(p.read_text('utf8'))
        drives.extend(d.get('trials',[]))
    assert len(drives)==24
    assert live['checked_points']+live['nonreturning_failed_queries']==sum(t['planning_calls'] for t in drives)
    for p in HERE.glob('*.py'):ast.parse(p.read_text('utf8'),filename=str(p))
    manifest={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'base_commit':'1dced97','timing_audits':audits,
              'historical_inputs_identical':True,
              'new_carla':{'drives':len(drives),'goals':sum(bool(t['goal']) for t in drives),
                           'collisions':sum(bool(t['collision']) for t in drives),
                           'planner_failures':sum(bool(t['planner_failure']) for t in drives),
                           'returned_policy_checks':live['checked_points'],
                           'nonreturning_failed_queries':live['nonreturning_failed_queries'],
                           'all_returned_checks_passed':live['all_passed']},
              'hash_basis':'Text suffixes use CRLF-to-LF normalized bytes, independent of Git/Windows line endings; binary files use raw bytes. The bytes field measures hashed content.',
              'files':[]}
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='artifact_manifest.json':continue
        content=p.read_bytes()
        if p.suffix=='.json':json.loads(content.decode('utf8'))
        assert len(content)<100*1024**2,'Unexpected large artifact '+str(p)
        is_text=p.suffix in ('.py','.json','.log','.md','.txt','.csv','.svg')
        hashed=content.replace(b'\r\n',b'\n') if is_text else content
        manifest['files'].append({'path':p.relative_to(HERE).as_posix(),'bytes':len(hashed),'hash_mode':'LF-normalized' if is_text else 'raw','sha256':hashlib.sha256(hashed).hexdigest()})
    (HERE/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
    print('Hashed',len(manifest['files']),'artifacts')

if __name__=='__main__':main()
