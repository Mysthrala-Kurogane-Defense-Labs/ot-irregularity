"""Freeze a development-selected thermal supplement before creating new holdouts."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import write_json
from scripts.validate_physical import hashes


def run(args):
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('Commit experiment code before freezing')
    result=json.loads((args.tail/'results.json').read_text());selected=result['selected']
    if not selected or not selected['eligible']:raise ValueError('No development-eligible candidate')
    if hashes(args.baseline)!=result['baseline_files'] or hashes(args.sample)!=result['source_hashes']:
        raise ValueError('Source changed after development selection')
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Duplicate holdout seeds')
    if any((args.raw_root/f'holdout-{seed}').exists() for seed in args.seeds):
        raise ValueError('A reserved holdout already exists; freeze must precede generation')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'frozen.json',{'format_version':1,'candidate':'baseline OR sample-thermal-tail',
        'frozen_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'selected':selected,'anchors':result['anchors'],'baseline_files':result['baseline_files'],
        'sample_files':result['source_hashes'],'tail_results_sha256':hashlib.sha256((args.tail/'results.json').read_bytes()).hexdigest(),
        'holdout_seeds':args.seeds,'baseline_threshold':json.loads((args.baseline/'contextual_models.json').read_text())['decision_threshold'],
        'controls':['unchanged v0.4 complete-tail','original shared-threshold v0.5 primary'],
        'relational_control_files':hashes(args.relational),
        'primary_gate':'No baseline event lost, cooling gain, precision >=.5, total events >=.5, false windows/asset-day <=10',
        'component_limit':'Thermal component validation does not replace v0.5. Compare physical and integrity families; joint relational+thermal integration still requires development selection and independent validation.',
        'evaluation':'All original intervals, identical complete minutes; paired run bootstrap; count unavailable windows and initial-context dependency',
        'test_used_for_selection':False})
    print(str(args.output/'frozen.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tail','sample','baseline','relational','raw-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    run(p.parse_args())
