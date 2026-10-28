"""Check retained audit/source hashes without repeating timed computations."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=read(HERE/'validation/audit.json')
    assert audit['trial_files']==36 and audit['all_trials_fully_checked'] and audit['all_integrity_checks_passed']
    count=0
    for item in audit['trials']:
        source=HERE/'results'/Path(item['source']).name
        # Absolute source strings were recorded on Windows; keep exports portable.
        if not source.exists():source=HERE/'results'/item['source'].replace('\\','/').split('/')[-1]
        trial=read(source)
        assert sha(source)==item['source_sha256'],source
        assert sha(HERE/trial['policy_archive'])==item['policy_archive_sha256']
        assert sha(HERE/trial['reference_path'])==item['reference_sha256']==trial['reference_sha256']
        assert sha(HERE/'validate_results.py')==item['validator_sha256']
        count+=len(trial['rows'])
    assert count==audit['checked_queries']
    manifest=read(HERE/'run_manifest.json')
    for relative,digest in manifest['source_hashes_lf'].items():
        assert hashlib.sha256((REPO/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,relative
    for folder,key in [('replays','traces'),('failure_inclusive','source_hashes')]:
        data=read(HERE/folder/'manifest.json')
        for name,digest in data[key].items():assert sha(HERE/'results'/name)==digest,name
        assert read(HERE/folder/'status.json')['complete']
    print(f'PASS: 36 drives; {count} queries; archived data, policy, reference and frozen-source hashes match.')
if __name__=='__main__':main()
