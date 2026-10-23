"""Check that new twelve-trace inputs equal the original stored parameters."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]

def main():
    rows=[]
    for path in sorted((HERE/'historical_results').glob('*.json')):
        d=json.loads(path.read_text('utf8'))
        if not d.get('run_complete'):continue
        name=f"{d['source_suite_index']}_{d['case'].replace(':','_')}_{d['stream']}_{d['seed']}_warm_lp.json"
        source=REPO/'reproduction/adaptive_risk_ltl/paper_suite'/name
        previous=json.loads(source.read_text('utf8'))
        old=[r['theta'] for r in previous['rows']];new=[r['theta'] for r in d['rows']]
        rows.append({'run':path.name,'old_input':str(source.relative_to(REPO)),'identical_theta':old==new,'points':len(new),'old_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    result={'runs':len(rows),'all_identical':bool(rows) and all(r['identical_theta'] for r in rows),'rows':rows}
    (HERE/'historical_input_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Original input equality:',result['all_identical'],'for',len(rows),'runs')
    if len(rows)!=72 or not result['all_identical']:raise AssertionError('Historical inputs differ or fewer than 72 completed runs')

if __name__=='__main__':main()
