# Feature definitions v1

One row represents an asset/run window. Windows are aligned to epoch at configured stride and use fixed duration. The compatibility policy `legacy` may emit an incomplete final window; `window.tail_policy: complete` emits only windows whose end has been reached by the last observed timestamp of that run/asset. Signal value statistics and slopes use all records except `quality=bad`; `uncertain` and null quality are retained because uncertainty does not mean the measurement is unusable. Quality and sampling counts include all records.

Window aggregation now keys channels by optional `measurement_role`; when absent, it falls back to `signal_class`. This preserves distinct measurement points that share a broad class. The OT Lab adapter maps its canonical `tag_id` values to `measurement_role` for this pseudo-synthetic study; generic datasets should supply stable semantic roles rather than product-specific identifiers. See the [semantic feature audit](docs/FEATURE_SEMANTICS_AUDIT_OTLAB.md).

Training stores which signal classes occur for each `asset_class`. Inference uses that map before describing `signal_loss` or `sampling_degradation`; legacy artifacts without it suppress those observations because structural absence and real loss cannot be distinguished. The underlying feature columns for structurally absent channels remain in the current fixed-width model input, so this does not yet correct model training semantics.

| Feature suffix | Formula | Inputs | Reason |
|---|---|---|---|
| `mean`, `median`, `min`, `max` | corresponding statistic | non-bad values in window | level and excursion |
| `range` | max - min | non-bad values | excursion size |
| `std` | population standard deviation | non-bad values | variability |
| `mad` | median(abs(x - median(x))) | non-bad values | robust variability |
| `last` | last non-bad value by timestamp | non-bad values | latest observed state |
| `delta` | last - first non-bad value | non-bad values | net within-window change |
| `slope` | least-squares slope against seconds relative to the window start | non-bad values | temporal trend |
| `sample_count` | count of records | all quality states | observed sampling |
| `expected_sample_count` | window duration / declared interval | sampling metadata | expected sampling |
| `coverage_ratio` | min(sample_count / expected, 1) | counts above | sampling degradation |
| `good_ratio`, `uncertain_ratio`, `bad_ratio` | quality count / total count | quality | data quality |
| `missing` | 1 if no record for signal in window, else 0 | signal presence | explicit signal loss |
| `context_asset_class_*` | train-vocabulary one-hot | asset_class | asset class context |
| `context_operating_regime_*` | train-vocabulary one-hot | operating_regime | regime context |
| `context_regime_unknown` | regime absent from train vocabulary | operating_regime | explicit regime mismatch |

Rows with `value_kind` other than `continuous` are currently excluded from this numeric feature pipeline. This prevents state codes from being treated as continuous measurements; category occupancy and transition features remain future work. Datasets without `value_kind` retain the legacy numeric behavior.

Signal prefixes come from normalized `measurement_role` values when supplied, otherwise `signal_class`. Cross-signal correlations, lag features and physical ratios are deferred; v0.1 does not generate them automatically.
# Experimental relationship residuals

`RelationshipResiduals` is an optional research transform, not enabled in the baseline feature pipeline. It emits one residual per configured continuous measurement role (six per class in this study).

| Name | Formula | Inputs and window | Reason |
|---|---|---|---|
| `{role}_relationship_residual` | `z_i - (b_i + sum(beta_ij * z_j, j != i))`, where `z` is the role's mean centered by normal-fit median and divided by normal-fit IQR | Current complete 1m means of the other roles of the same class; Ridge alpha=10 fitted only on normal fit runs | Detect departures from learned cross-signal relationships even when each marginal magnitude looks ordinary; no causal interpretation |
| `{role}_temporal_relationship_residual` | Same residual with additional predictors `z_(t-1)` for all roles | Current other-role means plus previous contiguous 1m means in the same run/asset | Test whether one-step history explains normal dynamics; target's current value is never an input to its own predictor |

Temporal history never crosses run, asset or gap boundaries and never reads a future window. At warmup, or if previous quality is insufficient, the temporal transform uses its static predictor. All current roles must have good ratio >=0.95, coverage >=0.9, finite means and missing=0; otherwise this residual detector is **unavailable**. A score sentinel of zero only disables its contribution to the supplemental maximum; it must not be reported as physical normality. The baseline detector still evaluates that window.

The direct research control scores the two largest absolute residuals divided by their normal early-stop q95. The residual AE uses RobustScaler on normal-fit residuals, latent 2, and the two largest squared reconstruction errors normalized by their normal early-stop q95. Both use normal-calibration empirical CDFs. The supplemental maximum with v0.4 receives its own normal-calibration threshold. Selection, availability and results are recorded separately; these features are not universally valid physical laws.
## Experimental thermal dynamics (development screen)

`thermal_delta_residual_degC = (temperature_last[t] - temperature_last[t-1]) - Ridge(previous_temperature_last, current_driver_means)`.
Fit per asset class and thermal role on normal fit runs only; predictor RobustScaler fits there too, with fixed Ridge alpha=1. Drivers are the applicable nonthermal, nonvibration operating roles, excluding photoeye counts. Current target temperature is never a predictor. The one-minute endpoint change represents thermal inertia under changing operation; residual magnitude is calibrated against normal validation residuals, separately for each target. Maximum percentile across available targets is the class thermal score. These are normal-reference percentiles, not fault probabilities.

Both consecutive windows must have >=95% good and >=90% coverage for required signals, no missing/nonfinite inputs, same run/asset and a contiguous boundary. Window duration must match fitting. Missing history yields unavailable status; internal zero disables the supplement and is not a normality assertion. The minute-level screen was rejected; see [thermal diagnostic](docs/workflows/model-training/runs/2026-10-05-thermal.md). No production defaults changed.

## Sample-first thermal features (research component)

| Name | Formula | Inputs / window | Reason |
|---|---|---|---|
| `thermal_rate_residual` | `(T[t]-T[t-1])/dt - Ridge(T[t-1], drivers[t], optional T_initial)` | Consecutive good finite samples, same run/asset; drivers exclude all current temperatures, vibration and photoeye counts | Model thermal change before averaging can dilute the deviation |
| `abs_time_weighted_mean_rate` | `abs(sum(residual * dt) / sum(dt))` | Available sample residuals ending within the complete minute | Persistent signed thermal departure, in degC/s |
| `q95_abs_rate` | `quantile(abs(residual), .95)` | Same minute and required samples | Large residual excursions without relying on a single maximum |
| `initial_temperature_context` | First fully usable target temperature observed in the run/asset, retained causally | Optional predictor, full-run batch context only | Investigate sensitivity to starting thermal conditions; this is not measured ambient temperature |
| `thermal_relative_magnitude` | `r/(r+a)`, `a=max(normal calibration q99 of r, 1e-9 degC/s)` | One summary/thermal role; maximum across available roles, then pooled normal threshold | Retain exceedance magnitude beyond empirical-CDF saturation; bounded magnitude, not percentile or failure probability |

Exact timestamp alignment; no interpolation/backfill. Maximum gap is 1.5 times the **first declared run/asset interval**, not a pooled interval across datasets. Each complete minute needs at least 20 valid pairs and 90% of expected pairs. Cadence changes and asynchronous signals need explicit handling before broader use. Normal-only fit runs train Ridge(alpha=1) and predictor RobustScaler; normal calibration runs supply score references. Development compares variants; independent test does not tune thresholds. See [protocol and limitations](docs/workflows/model-training/runs/2026-10-05-sample-thermal.md).
# Experimental telemetry health features

The standalone research channel in `telemetry_health.py` uses normalized role observations in complete one-minute windows. It does not change the existing model feature schema or generic inference CLI. See [protocol](docs/TELEMETRY_HEALTH_PROTOCOL.md).

| Name | Formula | Inputs | Window | Reason |
|---|---|---|---|---|
| bad/uncertain ratio | received grade count / received observations | role quality grades | 1 minute | Explain observed grade degradation independently of reconstruction |
| finite value count | count of finite values | role values | 1 minute | Measure usable numeric availability |
| expected sample count | window duration / previously declared interval | role cadence declaration | 1 minute | Establish an explicit sampling expectation |
| coverage ratio | finite count / expected count | preceding counts | 1 minute | Quantify sampling shortfall; retain values above one |
| repetition ratio | exactly equal adjacent good finite pairs / usable pairs | values, quality, timestamps, cadence | 1 minute | Describe repeated observations without diagnosing a sensor |
| mean value | mean of finite received values | role values in consistent units | 1 minute | Select normal repetition reference by fit-only mean quartiles |

Quality is unavailable for empty windows or unknown grades. Repetition requires at least 20 good adjacent pairs with positive gap at most 1.5 declared intervals. Cadence changes invalidate coverage/repetition until a new run segment; absent declarations never borrow a pooled default. Calibration counts, bounds, fallback and availability reasons are persisted. Observation labels describe strictly positive deviations; the experiment's separately configured health threshold controls alerts. These bounded deviations are not probabilities. Entirely unavailable windows remain null. Complete-file extraction cannot detect silence beyond the last observation without a streaming watermark.
