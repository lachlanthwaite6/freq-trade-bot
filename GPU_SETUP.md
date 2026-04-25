# FreqAI CPU to GPU Setup

This repo includes a working CPU baseline for FreqAI using:

- `FreqaiExampleStrategy`
- `SKLearnRandomForestRegressor`
- Binance spot data
- BTC/USDT, ETH/USDT, and SOL/USDT
- 5m base candles with 1h informative features

The CPU baseline is useful because it is simple and stable, but it does not use a graphics card.
To train on a GPU, switch from the sklearn model to a GPU-capable model such as XGBoost.

## 1. Clone the Repo

```bash
git clone https://github.com/Management6/freq-trade-bot.git
cd freq-trade-bot
```

If the repo contains the Freqtrade source at the root, stay in this folder. If you cloned into a
wrapper folder, move into the folder that contains `pyproject.toml`, `freqtrade/`, and `user_data/`.

## 2. Create a Local Config

Do not commit your real `user_data/config.json`.

```bash
cp user_data/config.example.json user_data/config.json
```

Then edit `user_data/config.json` and set any private values you need, such as exchange keys or API
server credentials.

For backtesting, empty Binance keys are fine.

## 3. Set Up Python

Use Python 3.12 for Freqtrade 2026.3.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e .
pip install -r requirements-freqai.txt
```

On macOS, LightGBM and XGBoost need OpenMP:

```bash
brew install libomp
```

Check imports:

```bash
.venv/bin/python -c "import sklearn, xgboost, lightgbm; print('ok')"
```

## 4. Download Historical Data

The tuned config uses 60 days of training history before the backtest starts, so download data from
at least `20210101` if you plan to backtest from `20210315`.

```bash
.venv/bin/freqtrade download-data \
  --config user_data/config.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT SOL/USDT \
  --timeframes 5m 1h \
  --timerange 20210101-20250101
```

Confirm the data is available:

```bash
.venv/bin/freqtrade list-data --config user_data/config.json --show-timerange
```

## 5. Run the CPU Baseline

This uses the custom sklearn random forest regressor. It is CPU-only.

```bash
.venv/bin/freqtrade backtesting \
  --config user_data/config.json \
  --strategy FreqaiExampleStrategy \
  --freqaimodel SKLearnRandomForestRegressor \
  --timeframe 5m \
  --timerange 20210315-20241231
```

## 6. Switch to GPU Training

`SKLearnRandomForestRegressor` cannot use a GPU. To use a GPU, switch to XGBoost.

In `user_data/config.json`, change:

```json
"identifier": "sklearn-rf-regressor-5m-v2"
```

to:

```json
"identifier": "xgboost-rf-regressor-5m-gpu-v1"
```

Then replace the `model_training_parameters` block with:

```json
"model_training_parameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42
}
```

Run the GPU-capable backtest:

```bash
.venv/bin/freqtrade backtesting \
  --config user_data/config.json \
  --strategy FreqaiExampleStrategy \
  --freqaimodel XGBoostRFRegressor \
  --timeframe 5m \
  --timerange 20210315-20241231
```

## 7. Verify GPU Usage

On NVIDIA systems, watch GPU activity in another terminal:

```bash
nvidia-smi -l 1
```

If GPU usage stays at zero, confirm:

- NVIDIA drivers are installed.
- CUDA is installed and visible.
- `nvidia-smi` works.
- XGBoost can see CUDA.

Quick Python check:

```bash
.venv/bin/python - <<'PY'
import xgboost
print(xgboost.__version__)
PY
```

## Notes

- NVIDIA GPUs are the most straightforward for XGBoost GPU training.
- Apple Silicon GPUs are not a drop-in replacement for XGBoost CUDA training.
- AMD GPU support is more complex and usually not the easiest path for this FreqAI setup.
- The first FreqAI run with a new `identifier` will say it cannot find cached model or prediction
  files. That is normal. It means FreqAI is training fresh models.
- Keep `user_data/config.json`, downloaded data, trained models, and backtest results out of git.
