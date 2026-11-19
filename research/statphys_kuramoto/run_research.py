from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence"
CACHE = ROOT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "TRXUSDT", "ETCUSDT", "SOLUSDT",
]
START = os.environ.get("KURAMOTO_START", "2021-01")
END = os.environ.get("KURAMOTO_END", "2026-10")
PERIODS = (24, 72, 168)
HORIZONS = (1, 4, 24, 120, 480)
EPS = 1.0e-12
DEV = {
    "devA": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:00", tz="UTC")),
    "devB": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:00", tz="UTC")),
}
POST = {
    "validation": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:00", tz="UTC")),
    "holdout": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:00", tz="UTC")),
    "confirmation": (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-10-31 23:00", tz="UTC")),
}
BASIC_PARENTS = [
    "ret1", "mom4", "mom24", "mom72", "mom168", "mom480",
    "vol24", "vol72", "vol168", "vol720", "ret_z168", "absret_z168",
    "activity_z168", "activity_z720", "range_z168", "range_z720",
    "close_location", "taker_imbalance", "taker_imbalance_z168",
    "ret_autocorr24", "ret_autocorr168", "activity_change24",
    "volume_momentum24", "skew168", "kurt168",
    "market_ret1", "market_mom24", "market_mom72", "market_mom168",
    "market_vol168", "beta168", "corr168", "relative_mom24", "relative_mom168",
]
LOWER_PER_PERIOD = [
    "amplitude", "phase_sin", "phase_cos", "relative_phase_sin", "relative_phase_cos",
    "order_parameter", "market_phase_sin", "market_phase_cos", "phase_force",
    "local_coherence", "phase_velocity", "market_phase_velocity", "velocity_mismatch",
    "order_change", "uncoupled_forecast1", "uncoupled_forecast24",
    "coupled_forecast1", "coupled_forecast24",
]
FORMULAS = {
    "coupling_correction1": "kuramoto_coupling",
    "coupling_correction4": "kuramoto_coupling",
    "coupling_correction24": "kuramoto_coupling",
    "coupling_correction120": "kuramoto_coupling",
    "coupling_correction480": "kuramoto_coupling",
    "phase_force_response": "kuramoto_coupling",
    "chimera_catchup": "chimera_state",
    "chimera_phase_slip": "chimera_state",
    "phase_slip_reversal": "phase_slip",
    "phase_velocity_catchup": "phase_slip",
    "synchronization_change_transport": "synchronization_dynamics",
    "desynchronization_momentum": "synchronization_dynamics",
    "entrainment_energy_transport": "synchronization_dynamics",
}
CROSS_FORMULAS = {
    "multiscale_coupling_consensus": "multiscale_phase_geometry",
    "multiscale_coupling_dispersion": "multiscale_phase_geometry",
    "phase_torsion_24_72": "multiscale_phase_geometry",
    "phase_torsion_72_168": "multiscale_phase_geometry",
    "phase_curvature_consensus": "multiscale_phase_geometry",
    "synchronization_front": "multiscale_phase_geometry",
}


def month_range(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")]


def download_one(symbol: str, month: str):
    target = CACHE / f"{symbol}-1h-{month}.zip"
    if target.exists() and target.stat().st_size > 100:
        return symbol, month, target.read_bytes(), None
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers={"User-Agent": "kuramoto-research/1.0"}, timeout=40)
            if response.status_code == 404:
                return symbol, month, None, "404"
            response.raise_for_status()
            target.write_bytes(response.content)
            return symbol, month, response.content, None
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1 + attempt)
    return symbol, month, None, error


def parse_zip(symbol: str, payload: bytes) -> pd.DataFrame:
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(x for x in archive.namelist() if x.endswith(".csv"))
        with archive.open(name) as stream:
            frame = pd.read_csv(stream, header=None, names=columns)
    stamp = pd.to_numeric(frame.open_time, errors="coerce")
    if stamp.dropna().median() > 1e14:
        stamp = stamp / 1000
    frame["timestamp"] = pd.to_datetime(stamp, unit="ms", utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_quote"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = symbol
    return frame[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_quote"]]


def load_data():
    jobs = [(symbol, month) for symbol in SYMBOLS for month in month_range(START, END)]
    parts = {symbol: [] for symbol in SYMBOLS}
    errors = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending = {pool.submit(download_one, symbol, month): (symbol, month) for symbol, month in jobs}
        for future in as_completed(pending):
            symbol, month = pending[future]
            try:
                _, _, payload, error = future.result()
            except Exception as exc:
                payload, error = None, f"future:{type(exc).__name__}:{exc}"
            if payload is None:
                if error != "404":
                    errors.append({"symbol": symbol, "month": month, "error": error})
                continue
            try:
                parts[symbol].append(parse_zip(symbol, payload))
            except Exception as exc:
                errors.append({"symbol": symbol, "month": month, "error": f"parse:{type(exc).__name__}:{exc}"})
    output = {}
    for symbol, frames in parts.items():
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp", "close"])
        frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").set_index("timestamp")
        if len(frame) >= 20000:
            output[symbol] = frame
    if len(output) < 8:
        raise RuntimeError(f"only {len(output)} symbols loaded: {list(output)}")
    audit = {
        "source": "Binance Data Vision spot monthly 1h klines",
        "start": START,
        "end": END,
        "symbols": sorted(output),
        "rows": {key: int(len(value)) for key, value in output.items()},
        "date_min": {key: str(value.index.min()) for key, value in output.items()},
        "date_max": {key: str(value.index.max()) for key, value in output.items()},
        "errors_non_404": errors[:200],
        "sha256": {
            key: hashlib.sha256(pd.util.hash_pandas_object(value.reset_index(), index=False).values.tobytes()).hexdigest()
            for key, value in output.items()
        },
    }
    return output, audit


def prior_z(series: pd.Series, window: int, minimum: int | None = None):
    minimum = minimum or max(24, window // 4)
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0)
    return ((series - mean) / (std + EPS)).clip(-8, 8)


def base_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    log_close = np.log(raw.close.where(raw.close > 0))
    log_open = np.log(raw.open.where(raw.open > 0))
    ret = log_close.diff()
    activity = np.log1p(raw.quote_volume.clip(lower=0))
    range_state = np.log(raw.high.where(raw.high > 0) / raw.low.where(raw.low > 0))
    imbalance = 2 * raw.taker_quote / (raw.quote_volume + EPS) - 1
    out["ret1"] = ret
    for horizon in [4, 24, 72, 168, 480]:
        out[f"mom{horizon}"] = ret.rolling(horizon, min_periods=max(2, horizon // 2)).sum()
    for window in [24, 72, 168, 720]:
        out[f"vol{window}"] = ret.rolling(window, min_periods=max(12, window // 3)).std(ddof=0)
    out["ret_z168"] = prior_z(ret, 168, 48)
    out["absret_z168"] = prior_z(ret.abs(), 168, 48)
    out["activity_z168"] = prior_z(activity, 168, 48)
    out["activity_z720"] = prior_z(activity, 720, 168)
    out["range_z168"] = prior_z(range_state, 168, 48)
    out["range_z720"] = prior_z(range_state, 720, 168)
    out["close_location"] = (raw.close - raw.low) / (raw.high - raw.low + EPS)
    out["taker_imbalance"] = imbalance
    out["taker_imbalance_z168"] = prior_z(imbalance, 168, 48)
    out["ret_autocorr24"] = ret.rolling(24, min_periods=12).corr(ret.shift(1))
    out["ret_autocorr168"] = ret.rolling(168, min_periods=56).corr(ret.shift(1))
    out["activity_change24"] = activity.diff(24)
    out["volume_momentum24"] = prior_z(activity.diff(24), 168, 48)
    out["skew168"] = ret.rolling(168, min_periods=72).skew()
    out["kurt168"] = ret.rolling(168, min_periods=72).kurt()
    out["next_ret1"] = log_close.shift(-1) - log_open.shift(-1)
    prior_vol = ret.shift(1).rolling(720, min_periods=168).std(ddof=0)
    for horizon in HORIZONS:
        target = log_close.shift(-horizon) - log_open.shift(-1)
        out[f"yraw{horizon}"] = target
        out[f"ystd{horizon}"] = target / (prior_vol * math.sqrt(horizon) + EPS)
    return out.astype(np.float32)


def wrap_phase(value):
    return np.angle(np.exp(1j * value))


def complex_state(history: np.ndarray, period: int):
    window = min(len(history), max(4 * period, 96))
    values = history[-window:]
    age = np.arange(window - 1, -1, -1, dtype=float)
    weights = np.exp(-age / (2.0 * period))
    kernel = weights * np.exp(1j * 2.0 * math.pi * age / period)
    valid = np.isfinite(values)
    numerator = np.nansum(np.where(valid, values, 0.0) * kernel[:, None], axis=0)
    denominator = np.sum(weights[:, None] * valid, axis=0) + EPS
    state = 2.0 * numerator / denominator
    coverage = valid.mean(axis=0)
    state[coverage < 0.75] = np.nan + 1j * np.nan
    return state


def integrated_forecast(amplitude, market_phase, delta, period, horizon, coupling):
    omega = 2.0 * math.pi / period
    uncoupled = np.zeros(len(amplitude))
    coupled = np.zeros(len(amplitude))
    relative = delta.copy()
    for step in range(1, horizon + 1):
        phase_market = market_phase + omega * step
        uncoupled += amplitude * np.cos(phase_market + delta)
        relative = wrap_phase(relative - coupling * np.sin(relative))
        coupled += amplitude * np.cos(phase_market + relative)
    return uncoupled, coupled


def phase_coordinates(ret_matrix, issue_indices):
    n_assets = ret_matrix.shape[1]
    lower = {}
    candidates = {}
    previous_theta = {period: None for period in PERIODS}
    previous_market_phase = {period: np.nan for period in PERIODS}
    previous_order = {period: np.nan for period in PERIODS}
    previous_delta = {period: None for period in PERIODS}
    state_cache = {}
    for period in PERIODS:
        for name in LOWER_PER_PERIOD:
            lower[f"p{period}__{name}"] = np.full((len(issue_indices), n_assets), np.nan, np.float32)
        for name in FORMULAS:
            candidates[f"p{period}__{name}"] = np.full((len(issue_indices), n_assets), np.nan, np.float32)
    for name in CROSS_FORMULAS:
        candidates[f"cross__{name}"] = np.full((len(issue_indices), n_assets), np.nan, np.float32)

    for output_index, time_index in enumerate(issue_indices):
        state_cache.clear()
        for period in PERIODS:
            if time_index < max(4 * period, 96):
                continue
            state = complex_state(ret_matrix[:time_index + 1], period)
            valid = np.isfinite(state.real) & np.isfinite(state.imag)
            if valid.sum() < 8:
                continue
            amplitude = np.abs(state)
            theta = np.angle(state)
            market_vector = np.nansum(state[valid])
            market_phase = float(np.angle(market_vector))
            order = float(abs(market_vector) / (np.nansum(amplitude[valid]) + EPS))
            delta = wrap_phase(theta - market_phase)
            correlation_window = min(4 * period, time_index)
            recent = ret_matrix[time_index - correlation_window + 1:time_index + 1]
            corr_matrix = np.corrcoef(np.nan_to_num(recent, nan=0.0), rowvar=False)
            np.fill_diagonal(corr_matrix, 0.0)
            neighbors = np.argsort(corr_matrix, axis=1)[:, -3:]
            local_coherence = np.zeros(n_assets)
            for asset in range(n_assets):
                local_coherence[asset] = np.nanmean(np.cos(theta[asset] - theta[neighbors[asset]]))
            if previous_theta[period] is None:
                phase_velocity = np.zeros(n_assets)
                market_velocity = 0.0
                order_change = 0.0
                delta_change = np.zeros(n_assets)
            else:
                phase_velocity = wrap_phase(theta - previous_theta[period]) / 24.0
                market_velocity = float(wrap_phase(market_phase - previous_market_phase[period]) / 24.0)
                order_change = order - previous_order[period]
                delta_change = wrap_phase(delta - previous_delta[period])
            velocity_mismatch = phase_velocity - market_velocity
            force = -order * np.sin(delta)
            coupling = max(order, 0.0) / period
            forecasts = {}
            for horizon in HORIZONS:
                forecasts[horizon] = integrated_forecast(amplitude, market_phase, delta, period, horizon, coupling)
            uncoupled1, coupled1 = forecasts[1]
            uncoupled24, coupled24 = forecasts[24]
            low_values = {
                "amplitude": amplitude,
                "phase_sin": np.sin(theta),
                "phase_cos": np.cos(theta),
                "relative_phase_sin": np.sin(delta),
                "relative_phase_cos": np.cos(delta),
                "order_parameter": np.full(n_assets, order),
                "market_phase_sin": np.full(n_assets, math.sin(market_phase)),
                "market_phase_cos": np.full(n_assets, math.cos(market_phase)),
                "phase_force": force,
                "local_coherence": local_coherence,
                "phase_velocity": phase_velocity,
                "market_phase_velocity": np.full(n_assets, market_velocity),
                "velocity_mismatch": velocity_mismatch,
                "order_change": np.full(n_assets, order_change),
                "uncoupled_forecast1": uncoupled1,
                "uncoupled_forecast24": uncoupled24,
                "coupled_forecast1": coupled1,
                "coupled_forecast24": coupled24,
            }
            for name, values in low_values.items():
                lower[f"p{period}__{name}"][output_index] = values.astype(np.float32)
            formula_values = {
                "coupling_correction1": forecasts[1][1] - forecasts[1][0],
                "coupling_correction4": forecasts[4][1] - forecasts[4][0],
                "coupling_correction24": forecasts[24][1] - forecasts[24][0],
                "coupling_correction120": forecasts[120][1] - forecasts[120][0],
                "coupling_correction480": forecasts[480][1] - forecasts[480][0],
                "phase_force_response": -amplitude * np.sin(theta) * force,
                "chimera_catchup": (forecasts[24][1] - forecasts[24][0]) * np.maximum(order - local_coherence, 0.0),
                "chimera_phase_slip": -amplitude * delta_change * np.maximum(order - local_coherence, 0.0),
                "phase_slip_reversal": -amplitude * np.sign(delta_change) * np.maximum(np.abs(delta_change) - math.pi / 2.0, 0.0),
                "phase_velocity_catchup": -amplitude * velocity_mismatch * order,
                "synchronization_change_transport": (forecasts[24][1] - forecasts[24][0]) * order_change,
                "desynchronization_momentum": amplitude * np.cos(theta) * (1.0 - order),
                "entrainment_energy_transport": amplitude * np.cos(delta) * force,
            }
            for name, values in formula_values.items():
                candidates[f"p{period}__{name}"][output_index] = values.astype(np.float32)
            state_cache[period] = {
                "amplitude": amplitude,
                "delta": delta,
                "order": order,
                "force": force,
                "correction24": forecasts[24][1] - forecasts[24][0],
            }
            previous_theta[period] = theta.copy()
            previous_market_phase[period] = market_phase
            previous_order[period] = order
            previous_delta[period] = delta.copy()
        if all(period in state_cache for period in PERIODS):
            corrections = np.stack([state_cache[period]["correction24"] for period in PERIODS])
            deltas = np.stack([state_cache[period]["delta"] for period in PERIODS])
            orders = np.array([state_cache[period]["order"] for period in PERIODS])
            forces = np.stack([state_cache[period]["force"] for period in PERIODS])
            candidates["cross__multiscale_coupling_consensus"][output_index] = np.nanmean(corrections, axis=0)
            candidates["cross__multiscale_coupling_dispersion"][output_index] = -np.nanstd(corrections, axis=0) * np.nanmean(corrections, axis=0)
            candidates["cross__phase_torsion_24_72"][output_index] = wrap_phase(deltas[0] - deltas[1]) * np.nanmean(corrections[:2], axis=0)
            candidates["cross__phase_torsion_72_168"][output_index] = wrap_phase(deltas[1] - deltas[2]) * np.nanmean(corrections[1:], axis=0)
            candidates["cross__phase_curvature_consensus"][output_index] = wrap_phase(deltas[0] - 2.0 * deltas[1] + deltas[2]) * np.nanmean(forces, axis=0)
            candidates["cross__synchronization_front"][output_index] = (orders[0] - orders[2]) * np.nanmean(corrections, axis=0)
    return lower, candidates


def market_parents(base_by_symbol, timeline, symbols):
    returns = np.column_stack([base_by_symbol[symbol].ret1.reindex(timeline).to_numpy(float) for symbol in symbols])
    market = pd.Series(np.nanmean(returns, axis=1), index=timeline)
    output = {}
    output["market_ret1"] = market
    for horizon in [24, 72, 168]:
        output[f"market_mom{horizon}"] = market.rolling(horizon, min_periods=max(12, horizon // 2)).sum()
    output["market_vol168"] = market.rolling(168, min_periods=56).std(ddof=0)
    for asset, symbol in enumerate(symbols):
        series = pd.Series(returns[:, asset], index=timeline)
        covariance = series.rolling(168, min_periods=56).cov(market)
        variance = market.rolling(168, min_periods=56).var(ddof=0)
        beta = covariance / (variance + EPS)
        correlation = series.rolling(168, min_periods=56).corr(market)
        residual = series - beta * market
        output[f"{symbol}__beta168"] = beta
        output[f"{symbol}__corr168"] = correlation
        output[f"{symbol}__relative_mom24"] = residual.rolling(24, min_periods=12).sum()
        output[f"{symbol}__relative_mom168"] = residual.rolling(168, min_periods=56).sum()
    return output, returns


def corr(x, y, minimum=100):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < minimum:
        return np.nan
    a = x[valid].astype(float)
    b = y[valid].astype(float)
    a -= a.mean()
    b -= b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denominator) if denominator > 1e-15 else np.nan


def mask_period(panel, bounds):
    return ((panel.timestamp >= bounds[0]) & (panel.timestamp <= bounds[1])).to_numpy()


def direct(panel, signal, horizon, mask, direction=1.0):
    raw, std, observations = [], [], []
    for _, group in panel.loc[mask].groupby("symbol", sort=False):
        x = direction * group[signal].to_numpy(float)
        raw_value = corr(x, group[f"yraw{horizon}"].to_numpy(float), 100)
        std_value = corr(x, group[f"ystd{horizon}"].to_numpy(float), 100)
        if np.isfinite(raw_value) and np.isfinite(std_value):
            raw.append(raw_value)
            std.append(std_value)
            observations.append(len(group))
    rank_raw, rank_std = [], []
    subset = panel.loc[mask, ["timestamp", "symbol", signal, f"yraw{horizon}", f"ystd{horizon}"]].dropna()
    for _, group in subset.groupby("timestamp", sort=False):
        if len(group) < 8:
            continue
        x = rankdata(direction * group[signal].to_numpy(float))
        rank_raw.append(corr(x, rankdata(group[f"yraw{horizon}"]), 8))
        rank_std.append(corr(x, rankdata(group[f"ystd{horizon}"]), 8))
    return {
        "tsic_raw": float(np.nanmean(raw)) if raw else np.nan,
        "tsic_std": float(np.nanmean(std)) if std else np.nan,
        "rankic_raw": float(np.nanmean(rank_raw)) if rank_raw else np.nan,
        "rankic_std": float(np.nanmean(rank_std)) if rank_std else np.nan,
        "n_assets": len(raw),
        "mean_obs": float(np.mean(observations)) if observations else 0.0,
        "rankic_days": len(rank_raw),
    }


def family(name):
    return name.split("__")[1]


def transform(X0, a0, X1, a1, nassets):
    means = np.full((nassets, X0.shape[1]), np.nan, np.float32)
    for asset in range(nassets):
        mask = a0 == asset
        if mask.any():
            means[asset] = np.nanmean(X0[mask], axis=0)
    global_mean = np.nanmean(X0, axis=0).astype(np.float32)
    missing = ~np.isfinite(means)
    means[missing] = np.broadcast_to(global_mean, means.shape)[missing]
    A = np.where(np.isfinite(X0), X0 - means[a0], 0).astype(np.float32)
    B = np.where(np.isfinite(X1), X1 - means[a1], 0).astype(np.float32)
    scale = np.sqrt(np.mean(A.astype(float) ** 2, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1).astype(np.float32)
    return np.clip(A / scale, -10, 10), np.clip(B / scale, -10, 10)


def center_y(y, assets, nassets):
    means = np.full(nassets, np.nan)
    for asset in range(nassets):
        mask = assets == asset
        if mask.any():
            means[asset] = np.mean(y[mask])
    means[~np.isfinite(means)] = np.mean(y)
    return y - means[assets], means


def ridge(X0, c0, y0, X1, c1, y1, a0, a1, nassets, alpha, dates):
    X0, X1 = transform(X0, a0, X1, a1, nassets)
    c0, c1 = transform(c0[:, None], a0, c1[:, None], a1, nassets)
    c0 = c0[:, 0].astype(float)
    c1 = c1[:, 0].astype(float)
    centered_y, intercept = center_y(y0.astype(float), a0, nassets)
    X = X0.astype(float)
    gram = X.T @ X
    gram.flat[::gram.shape[0] + 1] += 10
    beta = np.linalg.solve(gram, X.T @ centered_y)
    cross = X.T @ c0
    gram_cross = np.linalg.solve(gram, cross)
    schur = float(c0 @ c0 - cross @ gram_cross)
    coefficient = float(c0 @ centered_y - cross @ beta) / (schur + alpha)
    base = X1.astype(float) @ beta + intercept[a1]
    augmented = base + (c1 - X1.astype(float) @ gram_cross) * coefficient
    loss0 = (y1 - base) ** 2
    loss1 = (y1 - augmented) ** 2
    daily = pd.DataFrame({"date": pd.to_datetime(dates).floor("D"), "base": loss0, "aug": loss1}).groupby("date").mean()
    daily["gain"] = daily.base - daily.aug
    result = {
        "base_mse": float(daily.base.mean()),
        "aug_mse": float(daily.aug.mean()),
        "relative_improvement": float((daily.base.mean() - daily.aug.mean()) / daily.base.mean()),
        "candidate_coef": coefficient,
        "schur": schur,
        "train_n": len(y0),
        "test_n": len(y1),
    }
    return result, daily.reset_index()


def bootstrap(gain, block=30, draws=1999):
    gain = np.asarray(gain, float)
    gain = gain[np.isfinite(gain)]
    if len(gain) < 60:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_le_zero": np.nan, "n": len(gain)}
    starts = np.arange(len(gain) - block + 1)
    count = int(math.ceil(len(gain) / block))
    rng = np.random.default_rng(20261120)
    values = np.empty(draws)
    for index in range(draws):
        chosen = rng.choice(starts, size=count, replace=True)
        sample = np.concatenate([np.arange(start, start + block) for start in chosen])[:len(gain)]
        values[index] = gain[sample].mean()
    return {
        "ci_low": float(np.quantile(values, .025)),
        "ci_high": float(np.quantile(values, .975)),
        "p_le_zero": float((1 + (values <= 0).sum()) / (draws + 1)),
        "n": len(gain),
    }


def account(panel, signal, horizon, direction):
    subset = panel[["timestamp", "symbol", signal, "next_ret1"]].dropna().copy()
    issue = subset.loc[subset.timestamp.dt.hour.eq(0)]
    rows = []
    for timestamp, group in issue.groupby("timestamp", sort=False):
        if len(group) < 8:
            continue
        ranks = rankdata(direction * group[signal].to_numpy(float)) / (len(group) + 1)
        position = np.zeros(len(group))
        long = ranks >= .75
        short = ranks <= .25
        position[long] = .5 / max(long.sum(), 1)
        position[short] = -.5 / max(short.sum(), 1)
        rows.append(pd.DataFrame({"timestamp": timestamp, "symbol": group.symbol.to_numpy(), "position": position}))
    if not rows:
        return pd.DataFrame()
    issue_positions = pd.concat(rows).pivot(index="timestamp", columns="symbol", values="position").fillna(0).sort_index()
    returns = subset.pivot(index="timestamp", columns="symbol", values="next_ret1").fillna(0).sort_index()
    columns = sorted(set(issue_positions.columns) | set(returns.columns))
    returns = returns.reindex(columns=columns, fill_value=0)
    issued = issue_positions.reindex(index=returns.index, columns=columns, fill_value=0)
    sleeves = max(1, int(math.ceil(horizon / 24)))
    effective = issued.rolling(horizon, min_periods=1).sum() / sleeves
    turnover = effective.diff().abs().sum(axis=1).fillna(effective.abs().sum(axis=1))
    gross = (effective * returns).sum(axis=1)
    output = []
    for bps in [0, 2, 5, 10]:
        net = gross - turnover * bps / 10000
        equity = net.cumsum()
        drawdown = equity - equity.cummax()
        annual = float(net.mean() * 24 * 365)
        volatility = float(net.std(ddof=0) * math.sqrt(24 * 365))
        output.append({
            "cost_bps": bps,
            "annual_log_return": annual,
            "sharpe": annual / volatility if volatility > 1e-12 else np.nan,
            "max_drawdown": float(drawdown.min()),
            "mean_hourly_turnover": float(turnover.mean()),
            "hours": len(net),
        })
    return pd.DataFrame(output)


def main():
    started = time.time()
    raw_data, audit = load_data()
    base_by_symbol = {symbol: base_features(frame) for symbol, frame in raw_data.items()}
    symbols = sorted(base_by_symbol)
    timeline = pd.date_range(min(frame.index.min() for frame in base_by_symbol.values()),
                             max(frame.index.max() for frame in base_by_symbol.values()), freq="h", tz="UTC")
    market_states, return_matrix = market_parents(base_by_symbol, timeline, symbols)
    issue_indices = np.flatnonzero(timeline.hour == 0)
    issue_time = timeline[issue_indices]
    lower_parts, candidate_parts = phase_coordinates(return_matrix, issue_indices)
    issue_rows, full_rows = [], []
    for asset, symbol in enumerate(symbols):
        base = base_by_symbol[symbol]
        issue = base.reindex(issue_time).copy()
        issue["market_ret1"] = market_states["market_ret1"].reindex(issue_time).to_numpy(np.float32)
        for horizon in [24, 72, 168]:
            issue[f"market_mom{horizon}"] = market_states[f"market_mom{horizon}"].reindex(issue_time).to_numpy(np.float32)
        issue["market_vol168"] = market_states["market_vol168"].reindex(issue_time).to_numpy(np.float32)
        for name in ["beta168", "corr168", "relative_mom24", "relative_mom168"]:
            issue[name] = market_states[f"{symbol}__{name}"].reindex(issue_time).to_numpy(np.float32)
        for name, values in lower_parts.items():
            issue[f"lower__{name}"] = prior_z(pd.Series(values[:, asset], index=issue_time), 252, 60).to_numpy(np.float32)
        for name, values in candidate_parts.items():
            issue[f"candidate__{FORMULAS.get(name.split('__')[-1], CROSS_FORMULAS.get(name.split('__')[-1]))}__{name}"] = np.tanh(
                prior_z(pd.Series(values[:, asset], index=issue_time), 252, 60) / 3
            ).to_numpy(np.float32)
        issue["timestamp"] = issue_time
        issue["symbol"] = symbol
        issue["asset"] = asset
        issue_rows.append(issue.reset_index(drop=True))
        full = base.copy()
        full["timestamp"] = full.index
        full["symbol"] = symbol
        full["asset"] = asset
        signal_columns = [column for column in issue.columns if column.startswith("candidate__")]
        signals = issue.set_index("timestamp")[signal_columns]
        full = full.join(signals, how="left")
        full[signal_columns] = full[signal_columns].ffill(limit=23)
        full_rows.append(full.reset_index(drop=True))
    panel = pd.concat(issue_rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    panel_full = pd.concat(full_rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    candidates = [column for column in panel if column.startswith("candidate__")]
    lower = [column for column in panel if column.startswith("lower__")]
    parents = BASIC_PARENTS + lower

    development_rows = []
    for candidate in candidates:
        for horizon in HORIZONS:
            metrics = {name: direct(panel, candidate, horizon, mask_period(panel, bounds)) for name, bounds in DEV.items()}
            values = [metrics[period][key] for period in DEV for key in ["tsic_raw", "tsic_std"]]
            direction = float(np.sign(np.nanmean(values))) if np.isfinite(values).any() else 0
            oriented = [direction * value for value in values]
            development_rows.append({
                "candidate": candidate,
                "family": family(candidate),
                "horizon": horizon,
                "direction": direction,
                "min_oriented_tsic": float(np.nanmin(oriented)),
                "mean_oriented_tsic": float(np.nanmean(oriented)),
                **{f"{period}_{key}": value for period, result in metrics.items() for key, value in result.items()},
            })
    development = pd.DataFrame(development_rows)
    development.to_csv(OUT / "development_direct.csv", index=False)
    gate = development[(development.direction != 0) & (development.min_oriented_tsic > .004) &
                       (development.devA_n_assets >= 8) & (development.devB_n_assets >= 8)]
    shortlist_rows = []
    for family_name, group in gate.sort_values(["family", "min_oriented_tsic"], ascending=[True, False]).groupby("family"):
        shortlist_rows.extend(group.head(2).to_dict("records"))
    shortlist = pd.DataFrame(shortlist_rows)
    shortlist.to_csv(OUT / "development_shortlist.csv", index=False)
    if shortlist.empty:
        (OUT / "DECISION.json").write_text(json.dumps({"status": "no_direct_candidate", "admitted": []}, indent=2))
        return

    model_rows = []
    for _, claim in shortlist.iterrows():
        candidate = claim.candidate
        horizon = int(claim.horizon)
        direction = float(claim.direction)
        for year in [2022, 2023]:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")
            train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) &
                     (panel.timestamp < start - pd.Timedelta(hours=horizon))).to_numpy()
            test = ((panel.timestamp >= start) & (panel.timestamp <= end)).to_numpy()
            valid = panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy()
            train &= valid
            test &= valid
            if train.sum() < 2000 or test.sum() < 900:
                continue
            X0 = panel.loc[train, parents].to_numpy(np.float32)
            X1 = panel.loc[test, parents].to_numpy(np.float32)
            c0 = direction * panel.loc[train, candidate].to_numpy(np.float32)
            c1 = direction * panel.loc[test, candidate].to_numpy(np.float32)
            a0 = panel.loc[train, "asset"].to_numpy()
            a1 = panel.loc[test, "asset"].to_numpy()
            dates = panel.loc[test, "timestamp"].to_numpy()
            for target in ["raw", "std"]:
                y0 = panel.loc[train, f"y{target}{horizon}"].to_numpy(float)
                y1 = panel.loc[test, f"y{target}{horizon}"].to_numpy(float)
                for alpha in [10., 1000., 100000.]:
                    result, _ = ridge(X0, c0, y0, X1, c1, y1, a0, a1, len(symbols), alpha, dates)
                    model_rows.append({"candidate": candidate, "family": claim.family, "horizon": horizon,
                                       "direction": direction, "year": year, "target": target, "alpha": alpha, **result})
    model = pd.DataFrame(model_rows)
    model.to_csv(OUT / "development_model_grid.csv", index=False)
    frozen_rows = []
    for keys, group in model.groupby(["candidate", "family", "horizon", "direction", "alpha"]):
        lookup = {(int(row.year), row.target): row.relative_improvement for row in group.itertuples()}
        values = [lookup.get((year, target), np.nan) for year in [2022, 2023] for target in ["raw", "std"]]
        if np.isfinite(values).all() and min(values) > 0:
            frozen_rows.append(dict(zip(["candidate", "family", "horizon", "direction", "alpha"], keys)) |
                               {"min_dev_increment": min(values), "mean_dev_increment": float(np.mean(values))})
    frozen_grid = pd.DataFrame(frozen_rows)
    frozen_rows = []
    if not frozen_grid.empty:
        robust = frozen_grid.sort_values(["candidate", "horizon", "alpha"], ascending=[True, True, False]).groupby(
            ["candidate", "family", "horizon", "direction"], as_index=False).head(1)
        for family_name, group in robust.groupby("family"):
            frozen_rows.extend(group.sort_values(["min_dev_increment", "mean_dev_increment"], ascending=False).head(2).to_dict("records"))
    frozen = pd.DataFrame(frozen_rows)
    frozen.to_csv(OUT / "frozen_claims.csv", index=False)
    if frozen.empty:
        (OUT / "DECISION.json").write_text(json.dumps({"status": "no_strong_parent_candidate", "admitted": []}, indent=2))
        return

    heldout = {symbol for symbol in symbols if int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % 3 == 0}
    train_symbols = set(symbols) - heldout
    annual_rows, daily_rows, direct_rows = [], [], []
    for _, claim in frozen.iterrows():
        candidate = claim.candidate
        family_name = claim.family
        horizon = int(claim.horizon)
        direction = float(claim.direction)
        alpha = float(claim.alpha)
        siblings = [column for column in candidates if family(column) == family_name and column != candidate]
        for period, bounds in POST.items():
            if panel.timestamp.max() < bounds[0]:
                continue
            direct_rows.append({"candidate": candidate, "family": family_name, "horizon": horizon,
                                "period": period, **direct(panel, candidate, horizon, mask_period(panel, bounds), direction)})
            for mode, extra in [("ordinary", []), ("same_family", siblings)]:
                controls = parents + extra
                train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) &
                         (panel.timestamp < bounds[0] - pd.Timedelta(hours=horizon))).to_numpy()
                test = mask_period(panel, bounds)
                valid = panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy()
                train &= valid
                test &= valid
                if train.sum() < 2000 or test.sum() < 800:
                    continue
                X0 = panel.loc[train, controls].to_numpy(np.float32)
                X1 = panel.loc[test, controls].to_numpy(np.float32)
                c0 = direction * panel.loc[train, candidate].to_numpy(np.float32)
                c1 = direction * panel.loc[test, candidate].to_numpy(np.float32)
                a0 = panel.loc[train, "asset"].to_numpy()
                a1 = panel.loc[test, "asset"].to_numpy()
                dates = panel.loc[test, "timestamp"].to_numpy()
                for target in ["raw", "std"]:
                    result, daily = ridge(X0, c0, panel.loc[train, f"y{target}{horizon}"].to_numpy(float),
                                          X1, c1, panel.loc[test, f"y{target}{horizon}"].to_numpy(float),
                                          a0, a1, len(symbols), alpha, dates)
                    annual_rows.append({"candidate": candidate, "family": family_name, "horizon": horizon,
                                        "period": period, "parent_mode": mode, "target": target, **result})
                    daily_rows.append(daily.assign(candidate=candidate, horizon=horizon, period=period,
                                                   parent_mode=mode, target=target))
            train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) &
                     (panel.timestamp < bounds[0] - pd.Timedelta(hours=horizon)) & panel.symbol.isin(train_symbols)).to_numpy()
            test = mask_period(panel, bounds) & panel.symbol.isin(heldout).to_numpy()
            valid = panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy()
            train &= valid
            test &= valid
            if train.sum() >= 2000 and test.sum() >= 500:
                X0 = panel.loc[train, parents].to_numpy(np.float32)
                X1 = panel.loc[test, parents].to_numpy(np.float32)
                c0 = direction * panel.loc[train, candidate].to_numpy(np.float32)
                c1 = direction * panel.loc[test, candidate].to_numpy(np.float32)
                mapping = {symbol: index for index, symbol in enumerate(sorted(train_symbols))}
                unseen = len(mapping)
                a0 = panel.loc[train, "symbol"].map(mapping).to_numpy(np.int16)
                a1 = np.full(test.sum(), unseen, np.int16)
                dates = panel.loc[test, "timestamp"].to_numpy()
                for target in ["raw", "std"]:
                    result, daily = ridge(X0, c0, panel.loc[train, f"y{target}{horizon}"].to_numpy(float),
                                          X1, c1, panel.loc[test, f"y{target}{horizon}"].to_numpy(float),
                                          a0, a1, unseen + 1, alpha, dates)
                    annual_rows.append({"candidate": candidate, "family": family_name, "horizon": horizon,
                                        "period": period, "parent_mode": "heldout", "target": target, **result})
                    daily_rows.append(daily.assign(candidate=candidate, horizon=horizon, period=period,
                                                   parent_mode="heldout", target=target))
    annual = pd.DataFrame(annual_rows)
    direct_post = pd.DataFrame(direct_rows)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    annual.to_csv(OUT / "post_annual.csv", index=False)
    direct_post.to_csv(OUT / "post_direct.csv", index=False)
    daily.to_csv(OUT / "post_daily.csv", index=False)
    bootstrap_rows = []
    if not daily.empty:
        for keys, group in daily.groupby(["candidate", "horizon", "period", "parent_mode", "target"]):
            bootstrap_rows.append(dict(zip(["candidate", "horizon", "period", "parent_mode", "target"], keys)) |
                                  bootstrap(group.sort_values("date").gain.to_numpy()))
    boot = pd.DataFrame(bootstrap_rows)
    boot.to_csv(OUT / "bootstrap.csv", index=False)

    decisions, cost_frames = [], []
    required = [period for period in ["validation", "holdout", "confirmation"] if period in set(annual.period)]
    for _, claim in frozen.iterrows():
        candidate = claim.candidate
        family_name = claim.family
        horizon = int(claim.horizon)
        subset_annual = annual[annual.candidate == candidate]
        subset_boot = boot[boot.candidate == candidate]
        subset_direct = direct_post[direct_post.candidate == candidate]
        model_pass = True
        for mode in ["ordinary", "same_family", "heldout"]:
            a = subset_annual[(subset_annual.parent_mode == mode) & subset_annual.period.isin(required)]
            b = subset_boot[(subset_boot.parent_mode == mode) & subset_boot.period.isin(required)]
            if len(a) < 2 * len(required) or not (a.relative_improvement > 0).all():
                model_pass = False
            if len(b) < 2 * len(required) or not (b.ci_low > 0).all():
                model_pass = False
        direct_pass = all(
            not subset_direct[subset_direct.period == period].empty and
            subset_direct[subset_direct.period == period].iloc[0].tsic_raw > 0 and
            subset_direct[subset_direct.period == period].iloc[0].tsic_std > 0
            for period in required
        )
        account_pass = False
        if model_pass and direct_pass:
            for period in required:
                frame = account(panel_full.loc[mask_period(panel_full, POST[period])], candidate, horizon, float(claim.direction))
                if not frame.empty:
                    cost_frames.append(frame.assign(candidate=candidate, period=period, horizon=horizon))
            costs = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
            five = costs[(costs.candidate == candidate) & (costs.cost_bps == 5) & costs.period.isin(required)] if not costs.empty else pd.DataFrame()
            account_pass = len(five) == len(required) and (five.annual_log_return > 0).all() and (five.sharpe > 0).all()
        decisions.append({
            "candidate": candidate,
            "family": family_name,
            "horizon": horizon,
            "direction": float(claim.direction),
            "alpha": float(claim.alpha),
            "direct_pass": bool(direct_pass),
            "model_input_pass": bool(model_pass and direct_pass),
            "cost_pass": bool(account_pass),
            "admit_model_input": bool(model_pass and direct_pass),
            "independent_alpha": bool(model_pass and direct_pass and account_pass),
        })
    decision = pd.DataFrame(decisions)
    decision.to_csv(OUT / "admission.csv", index=False)
    costs = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
    costs.to_csv(OUT / "costs.csv", index=False)
    admitted = decision[decision.admit_model_input].to_dict("records")
    independent = decision[decision.independent_alpha].to_dict("records")
    final = {
        "status": "complete",
        "data_audit": audit,
        "symbols": symbols,
        "heldout_symbols": sorted(heldout),
        "candidate_count": len(candidates),
        "parent_count": len(parents),
        "shortlist_count": len(shortlist),
        "frozen_count": len(frozen),
        "admitted_model_inputs": admitted,
        "independent_alpha": independent,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "DECISION.json").write_text(json.dumps(final, indent=2, default=str))
    (OUT / "metadata.json").write_text(json.dumps({
        "candidates": candidates,
        "families": {column: family(column) for column in candidates},
        "parents": parents,
        "horizons": HORIZONS,
        "periods": PERIODS,
        "clock": "causal complex filters through completed issue hour; one daily issue; next-hour-open execution",
        "source": "Binance Data Vision spot hourly klines",
        "adaptive_history": True,
    }, indent=2))
    print(json.dumps({"admitted": admitted, "independent": independent,
                      "elapsed_seconds": final["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
