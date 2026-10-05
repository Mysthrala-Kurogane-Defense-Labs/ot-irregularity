# Architecture

Independent package under `src/ot_irregularity`; it has no Hub SDK, database, network or infrastructure dependencies. Input validation and semantic fields remain separate from numeric ML scaling. Windows are grouped by run and asset. Features combine signal statistics, data quality, sampling and train-derived one-hot asset/regime context. Missing trained signal classes become explicit missingness/coverage features; novel regimes emit `regime_mismatch`.

A small symmetric PyTorch MLP autoencoder minimizes reconstruction MSE. The Isolation Forest learns normal window feature distributions. Each raw detector score is mapped to its empirical CDF among normal validation windows: `rank / N`, clipped naturally to `[0,1]`; larger means farther into the anomalous tail. `score_samples` is negated first because sklearn's lower values are more anomalous. The ensemble is the configured nonnegative weighted mean; default weights and decision threshold are baselines, not optimized values.

Validation-normal AE and Isolation Forest score arrays, scalers, feature schema, parameters, thresholds and metadata are saved together. The scaler is fit on normal training windows only. The autoencoder checkpoint is selected by validation-normal reconstruction loss with early stopping. CPU seed control is used; deterministic PyTorch operations are requested.

ONNX export is deferred until exporter/runtime parity can be tested. Possible future model adapters include XGBoost, LSTM, temporal transformers and forecasting. They are not part of v0.1.
## Familia contextual opcional

`model_family: contextual` selecciona un AE y un Isolation Forest por `asset_class`. La normalización semántica y generación de ventanas son compartidas con la familia global; solo se ajustan features de roles aplicables. Cada clase persiste su scaler y modelos. `contextual_models.json` registra referencias normales y escala de residuos; los consumidores detectan ese manifiesto al cargar. Véase [el contrato y los límites](docs/MODEL_V0.4_CONTEXTUAL.md). `operating_regime` no condiciona todavía una dinámica específica en esta variante.
