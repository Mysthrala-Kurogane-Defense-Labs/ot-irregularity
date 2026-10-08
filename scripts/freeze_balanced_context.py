"""Freeze the selected ensemble, code and new source identities before generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import _sha256
from scripts.research_detection import write_json
from scripts.validate_physical import hashes
from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble

LAB_COMMIT = '718babb7772c3a21f0b87c403f628540cbce58db'
MIXED_SEEDS = [910571, 910572, 910573]
NORMAL_SEED = 920641


def verified_reference(study):
    """Reconstruct the selected reference from recorded development calibration."""
    reference = study/'median-0.9975.json'
    doc = json.loads(reference.read_text())
    frames = [pl.read_parquet(study/(kind+'-raw.parquet')).filter(pl.col('calibration'))
              for kind in ('historical','normal')]
    frame = pl.concat(frames,how='diagonal_relaxed')
    raw = frame.select(['raw_'+str(s) for s in (20261005,20261006,20261007)]).to_numpy()
    model = ClassMagnitudeEnsemble('median',.9975).fit(raw,frame['asset_class'].to_numpy(),np.zeros(len(frame),dtype=bool))
    if (doc.get('version') != 1 or doc.get('aggregation') != 'median' or doc.get('quantile') != .9975
        or doc.get('minimum_windows') != 400 or doc.get('references') != model.references_
        or set(model.references_) != {'CNC','COMPRESSOR','CONVEYOR','PUMP'}):
        raise ValueError('Class reference differs from development calibration')
    return reference


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def runtime_hashes(repo):
    return {p.relative_to(repo).as_posix():_sha256(p)
            for folder in ('src','scripts') for p in sorted((repo/folder).rglob('*.py'))}


def freeze(args):
    repo = Path(__file__).resolve().parents[1]
    if git(repo,'status','--porcelain') or git(args.lab,'status','--porcelain'):
        raise ValueError('Clean committed detector and Lab required before freeze')
    if git(args.lab,'rev-parse','HEAD') != LAB_COMMIT:
        raise ValueError('Pinned Lab commit required')
    if args.output.exists():
        raise FileExistsError('New freeze output required')
    source_names = [f'holdout-{s}' for s in MIXED_SEEDS] + [f'normal-exposure-{NORMAL_SEED}']
    if any((args.source_root/name).exists() for name in source_names):
        raise FileExistsError('Freeze must precede new source generation')
    selected = json.loads((args.study/'results.json').read_text())
    public = json.loads((repo/'docs/results/class-ensemble-20261005.json').read_text())
    if selected != public or selected['selected']['aggregation'] != 'median' or selected['selected']['quantile'] != .9975 or not selected['selected']['eligible'] or selected['test_used']:
        raise ValueError('Verified selected development class median required')
    relations = json.loads((args.relationships/'frozen.json').read_text())
    if hashes(args.baseline) != relations['baseline_files']:
        raise ValueError('Baseline differs from frozen control')
    if [m['seed'] for m in relations['models']] != [20261005,20261006,20261007]:
        raise ValueError('Fixed members required')
    for spec in relations['models']:
        if hashes(args.relationships/spec['directory']) != spec['files']:
            raise ValueError('Relational member changed')
    health_hash = json.loads((repo/'docs/results/telemetry-health-20261005-r2.json').read_text())['health_reference_sha256']
    reference = verified_reference(args.study)
    if _sha256(args.health_reference) != health_hash:
        raise ValueError('Health reference changed')
    context_results=json.loads((args.context_study/'results.json').read_text())
    if context_results!=json.loads((repo/'docs/results/balanced-context-20261008.json').read_text()) or not context_results['eligible'] or context_results['test_used']:
        raise ValueError('Selected expanded normal context study required')
    context_path=args.context_study/'context-reference.json'
    if json.loads(context_path.read_text())!=json.loads((repo/'docs/results/balanced-context-reference-20261008.json').read_text()):
        raise ValueError('Expanded context reference changed')
    previous = []; run_seeds = set(); masters = set()
    for path in sorted(set(args.prior_manifests)):
        doc = json.loads(path.read_text())
        if not isinstance(doc.get('master_seed'), int) or not doc.get('runs'):
            raise ValueError('Raw source manifests with seeds required')
        masters.add(doc['master_seed'])
        run_seeds.update(r['seed'] for r in doc['runs'])
        previous.append({'name':path.parent.name,'sha256':_sha256(path),'master_seed':doc['master_seed'],'runs':len(doc['runs'])})
    required = {20261011+7919*i for i in range(15)} | {920601,920611,920621,920631} | set(range(930101,930110)) | {910500+offset+i for offset in (0,10,20,30,40,50,60) for i in (1,2,3)}
    if not required.issubset(masters) or masters.intersection([*MIXED_SEEDS,NORMAL_SEED]):
        raise ValueError('Prior source inventory incomplete or proposed master seeds already used')
    frozen = {'version':1,'frozen_at':datetime.now(timezone.utc).isoformat(),'git_commit':git(repo,'rev-parse','HEAD'),
        'runtime_files':runtime_hashes(repo),'protocol_sha256':_sha256(repo/'docs/BALANCED_CONTEXT_CONFIRMATION_PROTOCOL.md'),
        'lab_commit':LAB_COMMIT,'lab_uv_lock_sha256':_sha256(args.lab/'uv.lock'),
        'mixed_seeds':MIXED_SEEDS,'mixed_runs_per_seed':240,'mixed_test_runs_per_seed':36,
        'mixed_suite_sha256':_sha256(args.lab/'suites/training-v0.2.yaml'),
        'normal_seed':NORMAL_SEED,'normal_runs':180,'minimum_asset_hours_per_class':168,
        'normal_suite_sha256':_sha256(repo/'configs/normal-exposure-v05.yaml'),
        'baseline_files':relations['baseline_files'],'members':relations['models'],
        'health_reference_sha256':health_hash,'health_threshold':.05,'repetition':False,
        'decision':'context-ensemble','context_reference_sha256':_sha256(context_path),'context_development_sha256':_sha256(args.context_study/'results.json'),'class_reference_sha256':_sha256(reference),'protocol_file':'docs/BALANCED_CONTEXT_CONFIRMATION_PROTOCOL.md','development_results_sha256':_sha256(args.study/'results.json'),
        'prior_sources':previous,'excluded_run_seeds':sorted(run_seeds),'source_names':source_names,
        'bootstrap_repetitions':2000,'bootstrap_seed':20261005,'selection_after_test':False}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'class-reference.json').write_bytes(reference.read_bytes())
    (args.output/'context-reference.json').write_bytes(context_path.read_bytes())
    write_json(args.output/'frozen.json',frozen)
    print(json.dumps({'git_commit':frozen['git_commit'],'excluded_runs':len(run_seeds),'mixed_seeds':MIXED_SEEDS,'normal_seed':NORMAL_SEED}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','lab','source-root','study','baseline','relationships','health-reference','context-study'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-manifests',type=Path,nargs='+',required=True);freeze(p.parse_args())
