"""Freeze existing controls and the normal-exposure protocol before generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import _sha256
from scripts.research_detection import write_json
from scripts.validate_physical import hashes


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def freeze(args):
    repo = Path(__file__).resolve().parents[1]
    if git(repo, 'status', '--porcelain') or git(args.lab, 'status', '--porcelain'):
        raise ValueError('Detector and simulator must be clean and committed')
    lab_commit = git(args.lab, 'rev-parse', 'HEAD')
    if lab_commit != '718babb7772c3a21f0b87c403f628540cbce58db':
        raise ValueError('Simulator differs from the declared protocol')
    if args.source.exists() or args.output.exists():
        raise FileExistsError('New source and freeze paths required before generation')
    suite = repo / 'configs/normal-exposure-v05.yaml'
    config = yaml.safe_load(suite.read_text())
    if config['generation']['anomaly_probability'] != 0 or config['partitions'] != {'train': 0., 'validation': 0., 'test': 1.}:
        raise ValueError('Expected normal-only test exposure')
    thermal = json.loads((args.thermal / 'frozen.json').read_text())
    joint = json.loads((args.joint / 'results.json').read_text())
    if joint['selected'] is not None or joint['test_used']:
        raise ValueError('Expected rejected development experiment')
    baseline_files = hashes(args.baseline)
    if baseline_files != thermal['baseline_files'] or baseline_files != joint['baseline_files']:
        raise ValueError('Baseline changed')
    if hashes(args.sample) != thermal['sample_files'] or hashes(args.relational) != thermal['relational_control_files']:
        raise ValueError('Thermal or relational control changed')
    model = next(m for m in joint['relational_models'] if m['seed'] == 20261005)
    if hashes(args.relationships / model['directory']) != model['files']:
        raise ValueError('Joint relational model changed')
    point = next(r for r in joint['rows'] if r['seed'] == 20261005 and r['thermal_enabled'] and r['quantile'] == .9975)
    if point['eligible']:
        raise ValueError('Expected rejected diagnostic point')
    frozen = {
        'format_version': 1, 'frozen_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': git(repo, 'rev-parse', 'HEAD'), 'lab_commit': lab_commit,
        'master_seed': 920601, 'runs': 180, 'minimum_asset_hours_per_class': 168,
        'suite_sha256': _sha256(suite),
        'protocol_sha256': _sha256(repo / 'docs/NORMAL_EXPOSURE_PROTOCOL.md'),
        'source_directory_name': args.source.name,
        'baseline_files': baseline_files, 'baseline_threshold': thermal['baseline_threshold'],
        'sample_files': thermal['sample_files'], 'thermal_selected': thermal['selected'],
        'thermal_anchors': thermal['anchors'],
        'relational_control_files': thermal['relational_control_files'],
        'relational_control_threshold': json.loads((args.relational / 'candidate.json').read_text())['threshold'],
        'joint_relational_model': model, 'joint_point': point,
        'joint_relational_anchors': joint['anchors']['20261005'],
        'joint_results_sha256': _sha256(args.joint / 'results.json'),
        'thermal_freeze_sha256': _sha256(args.thermal / 'frozen.json'),
        'bootstrap_seed': 20261005, 'bootstrap_repetitions': 2000,
        'episode_rule': 'Consecutive touching alert windows in one run/asset; any normal or missing window ends episode',
        'selection_or_retraining_allowed': False,
        'joint_status': 'Rejected development diagnostic, not a promoted candidate',
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'frozen.json', frozen)
    print(json.dumps({'freeze': str(args.output), 'git_commit': frozen['git_commit'], 'seed': 920601}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lab', 'source', 'output', 'baseline', 'sample', 'thermal', 'relational', 'relationships', 'joint'):
        parser.add_argument('--' + name, type=Path, required=True)
    freeze(parser.parse_args())
