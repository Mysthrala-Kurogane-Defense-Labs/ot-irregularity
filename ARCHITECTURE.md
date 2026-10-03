# Architecture

Independent package under `src/ot_irregularity`; it has no Hub SDK, database, network or infrastructure dependencies. Input validation and semantic fields remain separate from numeric ML scaling. Windows are grouped by run and asset. Features combine signal statistics, data quality, sampling and train-derived one-hot asset/regime context. Missing trained signal classes become explicit missingness/coverage features; novel regimes emit `regime_mismatch`.

A small symmetric PyTorch MLP autoencoder minimizes reconstruction MSE. The Isolation Forest learns normal window feature distributions. Each raw detector score is mapped to its empirical CDF among normal validation windows: `rank / N`, clipped naturally to `[0,1]`; larger means farther into the anomalous tail. `score_samples` is negated first because sklearn's lower values are more anomalous. The ensemble is the configured nonnegative weighted mean; default weights and decision threshold are baselines, not optimized values.

Validation-normal AE and Isolation Forest score arrays, scalers, feature schema, parameters, thresholds and metadata are saved together. The scaler is fit on normal training windows only. The autoencoder checkpoint is selected by validation-normal reconstruction loss with early stopping. CPU seed control is used; deterministic PyTorch operations are requested.

ONNX export is deferred until exporter/runtime parity can be tested. Possible future model adapters include XGBoost, LSTM, temporal transformers and forecasting. They are not part of v0.1.
