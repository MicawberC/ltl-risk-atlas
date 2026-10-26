"""Validate finalized results and create portable artifact hashes."""
from pathlib import Path
import json,hashlib,datetime,ast
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def main():
    audit=json.loads((HERE/'analysis/audit.json').read_text('utf8'))
    assert audit['all_complete'] and audit['all_valid'] and not audit['violations']
    assert audit['runs']==357 and audit['pairs']==138 and audit['query_records']==22242
    assert audit['previous_inputs_checked']==93
    dependencies=json.loads((HERE/'source_dependencies.json').read_text('utf8'))
    for dependency in dependencies['files']:assert digest(REPO/dependency['path'])==dependency['sha256_lf'],dependency['path']
    run=json.loads((HERE/'run_manifest.json').read_text('utf8'))
    source=digest(HERE/'optimized_controller.py')
    assert source==run['optimized_source_sha256']
    for path in (HERE/'results').glob('*.json'):
        d=json.loads(path.read_text('utf8'));assert d['optimized_source_sha256']==source
    for path in HERE.glob('*.py'):ast.parse(path.read_text('utf8'),filename=str(path))
    wins=json.loads((HERE/'analysis/wins.json').read_text('utf8'))
    manifest={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_commit':'46bbc0a','optimized_source_sha256_lf':source,'audit':audit,'wins':wins,'hash_basis':'Text bytes normalize CRLF to LF; PNG/NPZ/ZIP binary bytes are unchanged. Size describes hashed bytes.','files':[]}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or 'profile_repeat' in path.parts or path.suffix=='.prof' or path.name=='artifact_manifest.json':continue
        content=path.read_bytes()
        if path.suffix=='.json':json.loads(content.decode('utf8'))
        binary=path.suffix in ('.png','.npz','.zip')
        if not binary:content=content.replace(b'\r\n',b'\n')
        manifest['files'].append({'path':path.relative_to(HERE).as_posix(),'bytes':len(content),'hash_mode':'raw' if binary else 'LF-normalized','sha256':hashlib.sha256(content).hexdigest()})
    (HERE/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Validated',audit['runs'],'runs and',audit['query_records'],'query records; hashed',len(manifest['files']),'files')
if __name__=='__main__':main()
