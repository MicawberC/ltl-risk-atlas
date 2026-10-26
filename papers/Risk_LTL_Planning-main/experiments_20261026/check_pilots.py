"""Independent LP brackets for development inputs, not held-out performance."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
def main():
    checks=[]
    for first,second in [('pilot_optimized.json','pilot_lp.json'),('pilot_73047_optimized.json','pilot_73047_lp.json'),('pilot_priced.json','pilot_lp.json')]:
        a=json.loads((HERE/first).read_text('utf8'));b=json.loads((HERE/second).read_text('utf8'))
        assert a['theta_sha256']==b['theta_sha256']
        for x,y in zip(a['rows'],b['rows']):
            assert x['status']=='ok' and x['evaluation_ok'] and y['evaluation_ok']
            assert x['evaluated_risk']<=2+1e-7
            assert x['objective']<=y['objective']+1e-5 and y['objective']<=x['upper']+1e-5
            assert -1e-6<=x['upper']-x['objective']<=max(1e-4,.01*abs(x['objective']))+1e-6
            checks.append({'run':first,'index':x['index']})
    (HERE/'pilot_check.json').write_text(json.dumps({'passed':True,'count':len(checks),'checks':checks},indent=2),encoding='utf8')
    print('PASS',len(checks),'independent LP/policy checks')
if __name__=='__main__':main()
