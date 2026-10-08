"""Development-only integration of baseline integrity, relational AE and frozen thermal decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_alert_budgets import decision_metrics,union_decisions
from scripts.research_detection import cache_windows,write_json
from scripts.research_physical import load_events
from scripts.research_relationships import closed
from scripts.relationship_operating_points import residual_scores
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket,tail_errors
from ot_irregularity.models import ae_errors,load_ae
from ot_irregularity.pipeline import _scores
from ot_irregularity.sample_thermal import bounded_tail_score


def relational_magnitude_scores(root,frame,device):
    score=np.zeros(len(frame));anchors={}
    for group in sorted(frame['asset_class'].unique()):
        folder=root/group;model=joblib.load(folder/'detector.joblib')
        mask=frame['asset_class'].to_numpy()==group
        residual,available,_=model['relationship'].transform(frame.filter(pl.Series(mask)))
        raw,_=tail_errors(ae_errors(load_ae(folder/'autoencoder.pt',device),model['scaler'].transform(residual)),model['ae_scale'],2)
        anchor=max(float(np.quantile(model['references']['ae'],.99)),1e-9)
        s=bounded_tail_score(raw,anchor);s[~available]=0.;score[mask]=s;anchors[group]=anchor
    return score,anchors


def passes_gate(metrics, lost, baseline, relational_control, thermal_control):
    """Keep the declared detection floors and the unchanged false-window budget."""
    return bool(lost == 0
        and metrics['detected_events'] >= relational_control['detected_events']
        and metrics['physical_events']['detected'] >= relational_control['physical_events']['detected']
        and metrics['by_family']['cooling_degradation']['detected'] >= thermal_control['by_family']['cooling_degradation']['detected']
        and metrics['false_positive_windows'] <= baseline['false_positive_windows']
        and metrics['precision'] >= .5)


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    protocol={'stage':'Joint decisions selected only on development; no test data',
        'normalization':'Relational top-2 normalized AE error r/(r+normal calibration q99); floor 1e-9. Thermal selected development score/threshold unchanged.',
        'quantiles':[.99,.995,.9975,.999],'thermal_enabled':[False,True],
        'primary_seed':20261005,'seeds':[20261005,20261006,20261007],
        'decision':'baseline >= original threshold OR relational >= its own threshold OR optional thermal >= frozen development threshold',
        'gate':'No baseline alerts lost; physical/total events >= original primary v0.5; cooling >= selected thermal union; false windows <= v0.4 baseline; precision >= .5',
        'selection':'Primary seed only; most physical, cooling, all events, then fewer false windows. Other seeds confirmation only.',
        'test_used':False,'new_training':False}
    write_json(out/'protocol.json',protocol)
    relational=json.loads((args.relationships/'frozen.json').read_text());thermal=json.loads((args.thermal/'results.json').read_text())
    if hashes(args.baseline)!=relational['baseline_files'] or hashes(args.baseline)!=thermal['baseline_files']:raise ValueError('Baseline provenance mismatch')
    for model in relational['models']:
        if hashes(args.relationships/model['directory'])!=model['files']:raise ValueError('Relational model changed')
    selection=thermal['selected']
    if selection is None or not selection['eligible']:raise ValueError('Thermal selection not eligible')
    cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text());schema=json.loads((args.baseline/'feature_schema.json').read_text())
    cached=cache_windows(args.dataset,schema,cfg,args.cache)
    va=closed(cached['validation'],pl.read_parquet(args.dataset/'validation/telemetry.parquet'))
    cal=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    dm=np.array([run_bucket(r,2)==1 for r in va['run_id']]);dev=va.filter(pl.Series(dm));events,truth_hashes=load_events(args.raw_root,dev)
    keys=['run_id','asset_id','window_start','window_end'];saved=pl.read_parquet(args.thermal/'validation_scores.parquet')
    if not va.select(keys).equals(saved.select(keys)):raise ValueError('Thermal score identities/order differ')
    ts=saved[f"{selection['initial_context']}-{selection['score']}"].to_numpy();tth=selection['threshold']
    _,_,_,_,base,bth=_scores(args.baseline,va);base_alert=base[dm]>=bth
    baseline=decision_metrics(dev,base_alert,events)
    thermal_control=decision_metrics(dev,base_alert|(ts[dm]>=tth),events)
    primary=next(m for m in relational['models'] if m['seed']==protocol['primary_seed'])
    original=np.maximum(base,residual_scores(args.relationships/primary['directory'],va,args.device)['ae'])
    relational_control=decision_metrics(dev,original[dm]>=primary['threshold'],events)
    rows=[];anchors={};pred=va.select(keys)
    for model in relational['models']:
        score,anchor=relational_magnitude_scores(args.relationships/model['directory'],va,args.device)
        anchors[str(model['seed'])]=anchor;pred=pred.with_columns(pl.Series(str(model['seed']),score))
        for enabled in (False,True):
            for q in protocol['quantiles']:
                threshold=float(np.nextafter(np.quantile(score[cal],q,method='higher'),np.inf))
                row={'seed':model['seed'],'thermal_enabled':enabled,'quantile':q,'threshold':threshold,'eligible':False}
                if threshold>1:row['rejected_reason']='Unresolvable calibration'
                else:
                    alert=union_decisions(base[dm],score[dm],bth,threshold)
                    if enabled:alert|=ts[dm]>=tth
                    metrics=decision_metrics(dev,alert,events);lost=int(np.sum(base_alert&~alert))
                    eligible=passes_gate(metrics,lost,baseline,relational_control,thermal_control)
                    row.update(metrics=metrics,lost_baseline_alert_windows=lost,eligible=bool(eligible))
                rows.append(row);write_json(out/'progress.json',{'phase':'development','completed':len(rows),'total':24,'rows':rows});print(json.dumps(row),flush=True)
    eligible=[r for r in rows if r['seed']==protocol['primary_seed'] and r['eligible']]
    selected=max(eligible,key=lambda r:(r['metrics']['physical_events']['detected'],r['metrics']['by_family']['cooling_degradation']['detected'],r['metrics']['detected_events'],-r['metrics']['false_positive_windows'])) if eligible else None
    pred.write_parquet(out/'validation_relational_scores.parquet')
    write_json(out/'results.json',{'protocol':protocol,'baseline':baseline,'relational_control':relational_control,'thermal_control':thermal_control,
        'rows':rows,'selected':selected,'anchors':anchors,'thermal_selection':selection,'truth_hashes':truth_hashes,
        'baseline_files':relational['baseline_files'],'relational_models':relational['models'],
        'thermal_results_sha256':hashlib.sha256((args.thermal/'results.json').read_bytes()).hexdigest(),
        'thermal_scores_sha256':hashlib.sha256((args.thermal/'validation_scores.parquet').read_bytes()).hexdigest(),
        'test_used':False,'new_training':False})
    write_json(out/'progress.json',{'phase':'completed','completed':len(rows),'total':24,'selected':selected,'rows':rows})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','relationships','thermal','dataset','cache','raw-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
