"""Development-only screen of separate baseline and relational alert decisions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.relationship_operating_points import residual_scores
from scripts.research_detection import cache_windows, write_json
from scripts.research_physical import load_events, summarize
from scripts.research_relationships import closed
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket
from ot_irregularity.pipeline import _scores


def union_decisions(baseline, relational, baseline_threshold, relational_threshold):
    """Unavailable relational scores use zero upstream; both thresholds must be positive."""
    a, b = np.asarray(baseline, dtype=float), np.asarray(relational, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite([*a, *b]).all():
        raise ValueError('Scores must be finite vectors with identical shapes')
    if not 0 < baseline_threshold <= 1 or not 0 < relational_threshold <= 1:
        raise ValueError('Unresolvable or invalid threshold')
    return (a >= baseline_threshold) | (b >= relational_threshold)


def decision_metrics(frame, decisions, events):
    result = summarize(frame, np.asarray(decisions, dtype=float), .5, events)
    # Binary decisions do not define a continuous ranking; never publish their AP/AUC.
    for key in ('pr_auc', 'roc_auc', 'threshold'):
        result.pop(key, None)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    quantiles = [.995, .9975, .999]
    write_json(args.output/'protocol.json', {
        'stage': 'development only; no new fitting or test access',
        'quantiles': quantiles, 'primary_seed': 20261005,
        'gate': 'No baseline alarm lost; physical events increase; false windows <= baseline',
        'selection': 'Primary seed only: most physical events, then total events, then fewer false windows, then higher quantile; other seeds report variability',
        'source': 'docs/V0.5_MAINTENANCE.md; predeclared before this screen'})
    frozen = json.loads((args.validation/'frozen.json').read_text())
    if hashes(args.baseline) != frozen['baseline_files']:
        raise ValueError('Baseline changed')
    for model in frozen['models']:
        if hashes(args.validation/model['directory']) != model['files']:
            raise ValueError('Relational model changed')
    cfg = yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    schema = json.loads((args.baseline/'feature_schema.json').read_text())
    cached = cache_windows(args.dataset, schema, cfg, args.cache)
    va = closed(cached['validation'], pl.read_parquet(args.dataset/'validation/telemetry.parquet'))
    cal = np.array([run_bucket(r, 2) == 0 for r in va['run_id']]) & ~va['is_anomaly'].to_numpy()
    devmask = np.array([run_bucket(r, 2) == 1 for r in va['run_id']])
    dev = va.filter(pl.Series(devmask))
    events, truth_hashes = load_events(args.raw_root, dev)
    _, _, _, _, base, base_threshold = _scores(args.baseline, va)
    baseline_alerts = base[devmask] >= base_threshold
    baseline = decision_metrics(dev, baseline_alerts, events)
    rows = []
    for model in frozen['models']:
        scores = residual_scores(args.validation/model['directory'], va, args.device)['ae']
        current = decision_metrics(dev, np.maximum(base, scores)[devmask] >= model['threshold'], events)
        for q in quantiles:
            threshold = float(np.nextafter(np.quantile(scores[cal], q, method='higher'), np.inf))
            row = {'seed': model['seed'], 'quantile': q, 'baseline_threshold': base_threshold,
                   'relational_threshold': threshold, 'eligible': False}
            if threshold > 1:
                row['rejected_reason'] = 'Unresolvable calibration quantile'
            else:
                decisions = union_decisions(base[devmask], scores[devmask], base_threshold, threshold)
                metrics = decision_metrics(dev, decisions, events)
                lost = int(np.sum(baseline_alerts & ~decisions))
                row.update(metrics=metrics, lost_baseline_alert_windows=lost,
                           additional_false_windows=metrics['false_positive_windows']-baseline['false_positive_windows'],
                           eligible=bool(lost == 0 and metrics['physical_events']['detected'] > baseline['physical_events']['detected']
                                         and metrics['false_positive_windows'] <= baseline['false_positive_windows']))
            row['previous_shared_threshold_metrics'] = current
            rows.append(row)
            write_json(args.output/'progress.json', {'phase': 'development', 'completed': len(rows), 'total': 9, 'rows': rows})
            print(json.dumps(row), flush=True)
    eligible = [r for r in rows if r['seed'] == 20261005 and r['eligible']]
    selected = max(eligible, key=lambda r: (r['metrics']['physical_events']['detected'], r['metrics']['detected_events'],
                                          -r['metrics']['false_positive_windows'], r['quantile'])) if eligible else None
    write_json(args.output/'results.json', {'baseline': baseline, 'rows': rows, 'selected': selected,
        'windows': len(dev), 'events': len(events), 'test_used': False, 'new_training': False,
        'truth_hashes': truth_hashes, 'baseline_files': frozen['baseline_files'],
        'model_files': frozen['models'], 'cache_manifest': json.loads((args.cache/'cache_manifest.json').read_text())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'validation', 'dataset', 'cache', 'raw-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())
