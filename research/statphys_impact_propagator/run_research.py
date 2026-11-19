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
START = os.environ.get("IMPACT_START", "2021-01")
END = os.environ.get("IMPACT_END", "2026-10")
WINDOWS = (720, 2160)
LAGS = 24
HORIZONS = (1, 4, 24, 120, 480)
EPS = 1e-12
DEV = {
    "devA": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:00", tz="UTC")),
    "devB": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:00", tz="UTC")),
}
POST = {
    "validation": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:00", tz="UTC")),
    "holdout": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:00", tz="UTC")),
    "confirmation": (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-10-31 23:00", tz="UTC")),
}
LOWER_NAMES = [
    "impact_backlog", "current_impact_prediction", "impact_residual",
    "kernel_immediate", "kernel_permanent", "kernel_transient",
    "kernel_center", "kernel_entropy", "kernel_half_life",
    "kernel_l1", "kernel_curvature", "flow_memory_exponent",
    "impact_decay_exponent", "critical_compensation_gap",
    "fdt_shape_mismatch", "buy_sell_asymmetry", "impact_convexity",
    "flow_autocorr1", "flow_autocorr24", "kernel_rotation",
    "backlog_change", "residual_change",
]
FORMULAS = {
    "impact_backlog": "propagator_backlog",
    "activity_scaled_backlog": "propagator_backlog",
    "permanent_impact_continuation": "propagator_decomposition",
    "transient_impact_release": "propagator_decomposition",
    "impact_residual_reversal": "impact_surprise",
    "impact_residual_flow_reversal": "impact_surprise",
    "shock_absorption_reversal": "impact_surprise",
    "shock_underreaction_continuation": "impact_surprise",
    "critical_compensation_transport": "critical_diffusivity",
    "flow_memory_transport": "critical_diffusivity",
    "impact_decay_transport": "critical_diffusivity",
    "fdt_mismatch_transport": "fluctuation_response",
    "fdt_backlog_response": "fluctuation_response",
    "buy_sell_asymmetry_transport": "nonlinear_impact",
    "convexity_flow_transport": "nonlinear_impact",
    "impact_entropy_transport": "kernel_geometry",
    "kernel_center_transport": "kernel_geometry",
    "kernel_curvature_transport": "kernel_geometry",
    "kernel_rotation_response": "kernel_dynamics",
    "backlog_acceleration": "kernel_dynamics",
    "residual_acceleration_reversal": "kernel_dynamics",
}
BASIC_PARENTS = [
    "ret1", "mom4", "mom24", "mom72", "mom168", "mom480",
    "vol24", "vol72", "vol168", "vol720", "ret_z168", "absret_z168",
    "activity_z168", "activity_z720", "range_z168", "range_z720",
    "close_location", "taker_imbalance", "taker_imbalance_z168",
    "ret_autocorr24", "ret_autocorr168", "activity_change24",
    "volume_momentum24", "skew168", "kurt168", "flow_state",
]


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
            response = requests.get(url, headers={"User-Agent": "impact-propagator-research/1.0"}, timeout=40)
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
        pending = {pool.submit(download_one, s, m): (s, m) for s, m in jobs}
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
        "start": START, "end": END, "symbols": sorted(output),
        "rows": {k: int(len(v)) for k, v in output.items()},
        "date_min": {k: str(v.index.min()) for k, v in output.items()},
        "date_max": {k: str(v.index.max()) for k, v in output.items()},
        "errors_non_404": errors[:200],
        "sha256": {k: hashlib.sha256(pd.util.hash_pandas_object(v.reset_index(), index=False).values.tobytes()).hexdigest() for k, v in output.items()},
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
    imbalance = 2 * raw.taker_quote / (raw.quote_volume + EPS) - 1
    range_state = np.log(raw.high.where(raw.high > 0) / raw.low.where(raw.low > 0))
    flow_raw = imbalance * np.sqrt(raw.quote_volume.clip(lower=0) / (raw.quote_volume.rolling(168, min_periods=48).median() + EPS))
    flow = prior_z(flow_raw, 168, 48)
    out["ret1"] = ret
    for h in [4, 24, 72, 168, 480]: out[f"mom{h}"] = ret.rolling(h, min_periods=max(2, h // 2)).sum()
    for w in [24, 72, 168, 720]: out[f"vol{w}"] = ret.rolling(w, min_periods=max(12, w // 3)).std(ddof=0)
    out["ret_z168"] = prior_z(ret, 168, 48); out["absret_z168"] = prior_z(ret.abs(), 168, 48)
    out["activity_z168"] = prior_z(activity, 168, 48); out["activity_z720"] = prior_z(activity, 720, 168)
    out["range_z168"] = prior_z(range_state, 168, 48); out["range_z720"] = prior_z(range_state, 720, 168)
    out["close_location"] = (raw.close - raw.low) / (raw.high - raw.low + EPS)
    out["taker_imbalance"] = imbalance; out["taker_imbalance_z168"] = prior_z(imbalance, 168, 48)
    out["ret_autocorr24"] = ret.rolling(24, min_periods=12).corr(ret.shift(1)); out["ret_autocorr168"] = ret.rolling(168, min_periods=56).corr(ret.shift(1))
    out["activity_change24"] = activity.diff(24); out["volume_momentum24"] = prior_z(activity.diff(24), 168, 48)
    out["skew168"] = ret.rolling(168, min_periods=72).skew(); out["kurt168"] = ret.rolling(168, min_periods=72).kurt()
    out["flow_state"] = flow
    out["next_ret1"] = log_close.shift(-1) - log_open.shift(-1)
    prior_vol = ret.shift(1).rolling(720, min_periods=168).std(ddof=0)
    for horizon in HORIZONS:
        target = log_close.shift(-horizon) - log_open.shift(-1)
        out[f"yraw{horizon}"] = target; out[f"ystd{horizon}"] = target / (prior_vol * math.sqrt(horizon) + EPS)
    return out.astype(np.float32)


def lag_matrix(flow: np.ndarray, start: int, end: int, lags: int):
    rows = np.arange(start + lags, end)
    X = np.column_stack([flow[rows - lag] for lag in range(1, lags + 1)])
    return rows, X


def fit_kernel(ret: np.ndarray, flow: np.ndarray, start: int, end: int):
    rows, X = lag_matrix(flow, start, end, LAGS)
    y = ret[rows]
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if valid.sum() < max(240, (end - start) // 3):
        return None
    Xv, yv = X[valid], y[valid]
    Xv = Xv - Xv.mean(axis=0); yv = yv - yv.mean()
    gram = Xv.T @ Xv; ridge = 0.1 * np.trace(gram) / LAGS + 1e-6
    kernel = np.linalg.solve(gram + ridge * np.eye(LAGS), Xv.T @ yv)
    return kernel, Xv, yv


def power_slope(values: np.ndarray):
    values = np.asarray(values, float)
    lags = np.arange(1, len(values) + 1, dtype=float)
    valid = np.isfinite(values) & (values > 1e-6)
    if valid.sum() < 5:
        return 0.0
    slope = np.polyfit(np.log(lags[valid]), np.log(values[valid]), 1)[0]
    return float(max(-slope, 0.0))


def kernel_coordinates(ret: np.ndarray, flow: np.ndarray, activity: np.ndarray, issue_indices: np.ndarray, window: int):
    lower = {name: np.full(len(issue_indices), np.nan, np.float32) for name in LOWER_NAMES}
    candidates = {name: np.full(len(issue_indices), np.nan, np.float32) for name in FORMULAS}
    previous_kernel = None; previous_backlog = np.nan; previous_residual = np.nan
    for output_index, t in enumerate(issue_indices):
        if t < window + LAGS or t >= len(ret):
            continue
        fitted = fit_kernel(ret, flow, t - window, t)
        if fitted is None:
            continue
        kernel, X, y = fitted
        q_current = flow[t] if np.isfinite(flow[t]) else 0.0
        current_lags = np.array([flow[t + 1 - lag] for lag in range(1, LAGS + 1)], float)
        prior_lags = np.array([flow[t - lag] for lag in range(1, LAGS + 1)], float)
        if not np.isfinite(current_lags).all() or not np.isfinite(prior_lags).all() or not np.isfinite(ret[t]):
            continue
        backlog = float(kernel @ current_lags)
        current_prediction = float(kernel @ prior_lags)
        residual = float(ret[t] - current_prediction)
        immediate = float(kernel[0]); permanent = float(np.mean(kernel[-6:])); transient = immediate - permanent
        weights = np.abs(kernel); total = float(weights.sum()) + EPS
        center = float(np.sum((np.arange(LAGS) + 1) * weights) / total)
        probabilities = weights / total; entropy = float(-np.sum(probabilities * np.log(probabilities + EPS)) / math.log(LAGS))
        half = abs(immediate) / 2; hits = np.flatnonzero(np.abs(kernel) <= half); half_life = float(hits[0] + 1 if len(hits) else LAGS)
        curvature = float(np.sum(np.abs(np.diff(kernel, n=2))))
        qhist = flow[t-window:t]; valid_q = np.isfinite(qhist)
        qh = qhist[valid_q]; autocorr = []
        for lag in range(1, LAGS + 1):
            if len(qh) <= lag + 20: autocorr.append(np.nan); continue
            a, b = qh[:-lag], qh[lag:]
            den = np.std(a) * np.std(b)
            autocorr.append(float(np.cov(a, b, ddof=0)[0, 1] / den) if den > 1e-8 else np.nan)
        autocorr = np.asarray(autocorr); memory_exponent = power_slope(np.maximum(autocorr, 0))
        decay_exponent = power_slope(np.abs(kernel - permanent)); compensation = memory_exponent + 2 * decay_exponent - 1
        response = np.array([np.mean(y * X[:, lag]) / (np.mean(X[:, lag] ** 2) + EPS) for lag in range(LAGS)])
        response_norm = response / (np.linalg.norm(response) + EPS); kernel_norm = kernel / (np.linalg.norm(kernel) + EPS)
        fdt_mismatch = float(np.linalg.norm(response_norm - kernel_norm))
        flow_lag = X[:, 0]
        q75, q25 = np.quantile(flow_lag, [.75, .25])
        buy = y[flow_lag >= q75]; sell = y[flow_lag <= q25]
        buy_impact = float(np.mean(buy)) if len(buy) else 0.0; sell_impact = float(-np.mean(sell)) if len(sell) else 0.0
        asymmetry = buy_impact - sell_impact
        median_abs = np.median(np.abs(flow_lag)); high = np.abs(flow_lag) >= median_abs; low = ~high
        high_response = float(np.mean(np.abs(y[high])) / (np.mean(np.abs(flow_lag[high])) + EPS)) if high.any() else 0.0
        low_response = float(np.mean(np.abs(y[low])) / (np.mean(np.abs(flow_lag[low])) + EPS)) if low.any() else 0.0
        convexity = high_response - low_response
        rotation = float(np.linalg.norm(kernel - previous_kernel)) if previous_kernel is not None else 0.0
        backlog_change = backlog - previous_backlog if np.isfinite(previous_backlog) else 0.0
        residual_change = residual - previous_residual if np.isfinite(previous_residual) else 0.0
        ac1 = float(autocorr[0]) if np.isfinite(autocorr[0]) else 0.0; ac24 = float(autocorr[-1]) if np.isfinite(autocorr[-1]) else 0.0
        lower_values = {
            "impact_backlog": backlog, "current_impact_prediction": current_prediction, "impact_residual": residual,
            "kernel_immediate": immediate, "kernel_permanent": permanent, "kernel_transient": transient,
            "kernel_center": center, "kernel_entropy": entropy, "kernel_half_life": half_life,
            "kernel_l1": total, "kernel_curvature": curvature, "flow_memory_exponent": memory_exponent,
            "impact_decay_exponent": decay_exponent, "critical_compensation_gap": compensation,
            "fdt_shape_mismatch": fdt_mismatch, "buy_sell_asymmetry": asymmetry, "impact_convexity": convexity,
            "flow_autocorr1": ac1, "flow_autocorr24": ac24, "kernel_rotation": rotation,
            "backlog_change": backlog_change, "residual_change": residual_change,
        }
        for name, value in lower_values.items(): lower[name][output_index] = np.float32(value)
        activity_now = float(activity[t]) if np.isfinite(activity[t]) else 0.0
        sign_flow = float(np.sign(q_current)); over = max(abs(residual) - abs(current_prediction), 0.0); under = max(abs(current_prediction) - abs(residual), 0.0)
        formulas = {
            "impact_backlog": backlog,
            "activity_scaled_backlog": backlog * activity_now,
            "permanent_impact_continuation": q_current * permanent,
            "transient_impact_release": -q_current * transient,
            "impact_residual_reversal": -residual,
            "impact_residual_flow_reversal": -residual * abs(q_current),
            "shock_absorption_reversal": -sign_flow * over,
            "shock_underreaction_continuation": sign_flow * under,
            "critical_compensation_transport": q_current * compensation,
            "flow_memory_transport": q_current * memory_exponent,
            "impact_decay_transport": q_current * decay_exponent,
            "fdt_mismatch_transport": q_current * fdt_mismatch,
            "fdt_backlog_response": backlog * fdt_mismatch,
            "buy_sell_asymmetry_transport": q_current * asymmetry,
            "convexity_flow_transport": q_current * convexity,
            "impact_entropy_transport": backlog * entropy,
            "kernel_center_transport": q_current * center,
            "kernel_curvature_transport": q_current * curvature,
            "kernel_rotation_response": backlog * rotation,
            "backlog_acceleration": backlog_change,
            "residual_acceleration_reversal": -residual_change,
        }
        for name, value in formulas.items(): candidates[name][output_index] = np.float32(value)
        previous_kernel = kernel.copy(); previous_backlog = backlog; previous_residual = residual
    return lower, candidates


def corr(x, y, minimum=100):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < minimum: return np.nan
    a, b = x[valid].astype(float), y[valid].astype(float); a -= a.mean(); b -= b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b)); return float(a @ b / den) if den > 1e-15 else np.nan


def mask_period(panel, bounds): return ((panel.timestamp >= bounds[0]) & (panel.timestamp <= bounds[1])).to_numpy()


def direct(panel, signal, horizon, mask, direction=1.0):
    raw, std, obs = [], [], []
    for _, group in panel.loc[mask].groupby("symbol", sort=False):
        x = direction * group[signal].to_numpy(float); r = corr(x, group[f"yraw{horizon}"].to_numpy(float), 100); s = corr(x, group[f"ystd{horizon}"].to_numpy(float), 100)
        if np.isfinite(r) and np.isfinite(s): raw.append(r); std.append(s); obs.append(len(group))
    rank_raw, rank_std = [], []; subset = panel.loc[mask, ["timestamp", "symbol", signal, f"yraw{horizon}", f"ystd{horizon}"]].dropna()
    for _, group in subset.groupby("timestamp", sort=False):
        if len(group) < 8: continue
        x = rankdata(direction * group[signal].to_numpy(float)); rank_raw.append(corr(x, rankdata(group[f"yraw{horizon}"]), 8)); rank_std.append(corr(x, rankdata(group[f"ystd{horizon}"]), 8))
    return {"tsic_raw": float(np.nanmean(raw)) if raw else np.nan, "tsic_std": float(np.nanmean(std)) if std else np.nan,
            "rankic_raw": float(np.nanmean(rank_raw)) if rank_raw else np.nan, "rankic_std": float(np.nanmean(rank_std)) if rank_std else np.nan,
            "n_assets": len(raw), "mean_obs": float(np.mean(obs)) if obs else 0, "rankic_days": len(rank_raw)}


def family(name): return name.split("__")[1]


def transform(X0, a0, X1, a1, nassets):
    means = np.full((nassets, X0.shape[1]), np.nan, np.float32)
    for asset in range(nassets):
        m = a0 == asset
        if m.any(): means[asset] = np.nanmean(X0[m], axis=0)
    gm = np.nanmean(X0, axis=0).astype(np.float32); missing = ~np.isfinite(means); means[missing] = np.broadcast_to(gm, means.shape)[missing]
    A = np.where(np.isfinite(X0), X0 - means[a0], 0).astype(np.float32); B = np.where(np.isfinite(X1), X1 - means[a1], 0).astype(np.float32)
    scale = np.sqrt(np.mean(A.astype(float) ** 2, axis=0)); scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1).astype(np.float32)
    return np.clip(A / scale, -10, 10), np.clip(B / scale, -10, 10)


def center_y(y, assets, nassets):
    means = np.full(nassets, np.nan)
    for asset in range(nassets):
        m = assets == asset
        if m.any(): means[asset] = np.mean(y[m])
    means[~np.isfinite(means)] = np.mean(y); return y - means[assets], means


def ridge(X0, c0, y0, X1, c1, y1, a0, a1, nassets, alpha, dates):
    X0, X1 = transform(X0, a0, X1, a1, nassets); c0, c1 = transform(c0[:, None], a0, c1[:, None], a1, nassets); c0, c1 = c0[:, 0].astype(float), c1[:, 0].astype(float)
    yc, intercept = center_y(y0.astype(float), a0, nassets); X = X0.astype(float); gram = X.T @ X; gram.flat[::gram.shape[0] + 1] += 10
    beta = np.linalg.solve(gram, X.T @ yc); cross = X.T @ c0; gcross = np.linalg.solve(gram, cross); schur = float(c0 @ c0 - cross @ gcross)
    coefficient = float(c0 @ yc - cross @ beta) / (schur + alpha); base = X1.astype(float) @ beta + intercept[a1]; augmented = base + (c1 - X1.astype(float) @ gcross) * coefficient
    loss0, loss1 = (y1 - base) ** 2, (y1 - augmented) ** 2; daily = pd.DataFrame({"date": pd.to_datetime(dates).floor("D"), "base": loss0, "aug": loss1}).groupby("date").mean(); daily["gain"] = daily.base - daily.aug
    return {"base_mse": float(daily.base.mean()), "aug_mse": float(daily.aug.mean()), "relative_improvement": float((daily.base.mean() - daily.aug.mean()) / daily.base.mean()),
            "candidate_coef": coefficient, "schur": schur, "train_n": len(y0), "test_n": len(y1)}, daily.reset_index()


def bootstrap(gain, block=30, draws=1999):
    gain = np.asarray(gain, float); gain = gain[np.isfinite(gain)]
    if len(gain) < 60: return {"ci_low": np.nan, "ci_high": np.nan, "p_le_zero": np.nan, "n": len(gain)}
    starts = np.arange(len(gain) - block + 1); count = int(math.ceil(len(gain) / block)); rng = np.random.default_rng(20261120); values = np.empty(draws)
    for i in range(draws):
        chosen = rng.choice(starts, size=count, replace=True); index = np.concatenate([np.arange(x, x + block) for x in chosen])[:len(gain)]; values[i] = gain[index].mean()
    return {"ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)), "p_le_zero": float((1 + (values <= 0).sum()) / (draws + 1)), "n": len(gain)}


def account(panel, signal, horizon, direction):
    subset = panel[["timestamp", "symbol", signal, "next_ret1"]].dropna().copy(); issue = subset.loc[subset.timestamp.dt.hour.eq(0)]; rows = []
    for timestamp, group in issue.groupby("timestamp", sort=False):
        if len(group) < 8: continue
        ranks = rankdata(direction * group[signal].to_numpy(float)) / (len(group) + 1); position = np.zeros(len(group)); long = ranks >= .75; short = ranks <= .25
        position[long] = .5 / max(long.sum(), 1); position[short] = -.5 / max(short.sum(), 1); rows.append(pd.DataFrame({"timestamp": timestamp, "symbol": group.symbol.to_numpy(), "position": position}))
    if not rows: return pd.DataFrame()
    issue_positions = pd.concat(rows).pivot(index="timestamp", columns="symbol", values="position").fillna(0).sort_index(); returns = subset.pivot(index="timestamp", columns="symbol", values="next_ret1").fillna(0).sort_index()
    columns = sorted(set(issue_positions.columns) | set(returns.columns)); returns = returns.reindex(columns=columns, fill_value=0); issued = issue_positions.reindex(index=returns.index, columns=columns, fill_value=0)
    sleeves = max(1, int(math.ceil(horizon / 24))); effective = issued.rolling(horizon, min_periods=1).sum() / sleeves; turnover = effective.diff().abs().sum(axis=1).fillna(effective.abs().sum(axis=1)); gross = (effective * returns).sum(axis=1)
    rows = []
    for bps in [0, 2, 5, 10]:
        net = gross - turnover * bps / 10000; equity = net.cumsum(); dd = equity - equity.cummax(); annual = float(net.mean() * 24 * 365); vol = float(net.std(ddof=0) * math.sqrt(24 * 365))
        rows.append({"cost_bps": bps, "annual_log_return": annual, "sharpe": annual / vol if vol > 1e-12 else np.nan, "max_drawdown": float(dd.min()), "mean_hourly_turnover": float(turnover.mean()), "hours": len(net)})
    return pd.DataFrame(rows)


def main():
    started = time.time(); raw_data, audit = load_data(); issue_rows, full_rows = [], []
    for asset, (symbol, raw) in enumerate(sorted(raw_data.items())):
        base = base_features(raw); issue_time = base.index[base.index.hour == 0]; issue = base.reindex(issue_time).copy(); ret = base.ret1.to_numpy(float); flow = base.flow_state.to_numpy(float); activity = base.activity_z168.to_numpy(float); issue_indices = np.flatnonzero(base.index.hour == 0)
        lower_parts, candidate_parts = {}, {}
        for window in WINDOWS:
            lower, candidates = kernel_coordinates(ret, flow, activity, issue_indices, window)
            for name, values in lower.items(): lower_parts[f"lower__w{window}__{name}"] = values
            for name, values in candidates.items(): candidate_parts[f"candidate__{FORMULAS[name]}__w{window}__{name}"] = values
        for name, values in lower_parts.items(): issue[name] = prior_z(pd.Series(values, index=issue_time), 252, 60).to_numpy(np.float32)
        for name, values in candidate_parts.items(): issue[name] = np.tanh(prior_z(pd.Series(values, index=issue_time), 252, 60) / 3).to_numpy(np.float32)
        issue["timestamp"] = issue_time; issue["symbol"] = symbol; issue["asset"] = asset; issue_rows.append(issue.reset_index(drop=True))
        full = base.copy(); full["timestamp"] = full.index; full["symbol"] = symbol; full["asset"] = asset; signals = issue.set_index("timestamp")[list(candidate_parts)]; full = full.join(signals, how="left"); full[list(candidate_parts)] = full[list(candidate_parts)].ffill(limit=23); full_rows.append(full.reset_index(drop=True))
    panel = pd.concat(issue_rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True); panel_full = pd.concat(full_rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    candidates = [x for x in panel if x.startswith("candidate__")]; lower = [x for x in panel if x.startswith("lower__")]; parents = BASIC_PARENTS + lower; symbols = sorted(panel.symbol.unique())
    dev_rows = []
    for candidate in candidates:
        for horizon in HORIZONS:
            metrics = {name: direct(panel, candidate, horizon, mask_period(panel, bounds)) for name, bounds in DEV.items()}; values = [metrics[p][k] for p in DEV for k in ["tsic_raw", "tsic_std"]]; direction = float(np.sign(np.nanmean(values))) if np.isfinite(values).any() else 0; oriented = [direction * x for x in values]
            dev_rows.append({"candidate": candidate, "family": family(candidate), "horizon": horizon, "direction": direction, "min_oriented_tsic": float(np.nanmin(oriented)), "mean_oriented_tsic": float(np.nanmean(oriented)), **{f"{p}_{k}": v for p, m in metrics.items() for k, v in m.items()}})
    dev = pd.DataFrame(dev_rows); dev.to_csv(OUT / "development_direct.csv", index=False); gate = dev[(dev.direction != 0) & (dev.min_oriented_tsic > .004) & (dev.devA_n_assets >= 8) & (dev.devB_n_assets >= 8)]; shortlist = []
    for fam, group in gate.sort_values(["family", "min_oriented_tsic"], ascending=[True, False]).groupby("family"): shortlist.extend(group.head(2).to_dict("records"))
    shortlist = pd.DataFrame(shortlist); shortlist.to_csv(OUT / "development_shortlist.csv", index=False)
    if shortlist.empty: (OUT / "DECISION.json").write_text(json.dumps({"status": "no_direct_candidate", "admitted": []}, indent=2)); return
    model_rows = []
    for _, claim in shortlist.iterrows():
        candidate, horizon, direction = claim.candidate, int(claim.horizon), float(claim.direction)
        for year in [2022, 2023]:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC"); end = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"); train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) & (panel.timestamp < start - pd.Timedelta(hours=horizon))).to_numpy(); test = ((panel.timestamp >= start) & (panel.timestamp <= end)).to_numpy(); valid = panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy(); tr, te = train & valid, test & valid
            if tr.sum() < 2000 or te.sum() < 900: continue
            X0, X1 = panel.loc[tr, parents].to_numpy(np.float32), panel.loc[te, parents].to_numpy(np.float32); c0 = direction * panel.loc[tr, candidate].to_numpy(np.float32); c1 = direction * panel.loc[te, candidate].to_numpy(np.float32); a0, a1 = panel.loc[tr, "asset"].to_numpy(), panel.loc[te, "asset"].to_numpy(); dates = panel.loc[te, "timestamp"].to_numpy()
            for target in ["raw", "std"]:
                y0 = panel.loc[tr, f"y{target}{horizon}"].to_numpy(float); y1 = panel.loc[te, f"y{target}{horizon}"].to_numpy(float)
                for alpha in [10., 1000., 100000.]:
                    result, _ = ridge(X0, c0, y0, X1, c1, y1, a0, a1, len(symbols), alpha, dates); model_rows.append({"candidate": candidate, "family": claim.family, "horizon": horizon, "direction": direction, "year": year, "target": target, "alpha": alpha, **result})
    model = pd.DataFrame(model_rows); model.to_csv(OUT / "development_model_grid.csv", index=False); frozen_rows = []
    for keys, group in model.groupby(["candidate", "family", "horizon", "direction", "alpha"]):
        lookup = {(int(x.year), x.target): x.relative_improvement for x in group.itertuples()}; values = [lookup.get((year, target), np.nan) for year in [2022, 2023] for target in ["raw", "std"]]
        if np.isfinite(values).all() and min(values) > 0: frozen_rows.append(dict(zip(["candidate", "family", "horizon", "direction", "alpha"], keys)) | {"min_dev_increment": min(values), "mean_dev_increment": float(np.mean(values))})
    grid = pd.DataFrame(frozen_rows); frozen_rows = []
    if not grid.empty:
        robust = grid.sort_values(["candidate", "horizon", "alpha"], ascending=[True, True, False]).groupby(["candidate", "family", "horizon", "direction"], as_index=False).head(1)
        for fam, group in robust.groupby("family"): frozen_rows.extend(group.sort_values(["min_dev_increment", "mean_dev_increment"], ascending=False).head(2).to_dict("records"))
    frozen = pd.DataFrame(frozen_rows); frozen.to_csv(OUT / "frozen_claims.csv", index=False)
    if frozen.empty: (OUT / "DECISION.json").write_text(json.dumps({"status": "no_strong_parent_candidate", "admitted": []}, indent=2)); return
    heldout = {s for s in symbols if int(hashlib.sha256(s.encode()).hexdigest(), 16) % 3 == 0}; train_symbols = set(symbols) - heldout; annual, daily_rows, direct_rows = [], [], []
    for _, claim in frozen.iterrows():
        candidate, fam, horizon = claim.candidate, claim.family, int(claim.horizon); direction, alpha = float(claim.direction), float(claim.alpha); siblings = [x for x in candidates if family(x) == fam and x != candidate]
        for period, bounds in POST.items():
            if panel.timestamp.max() < bounds[0]: continue
            direct_rows.append({"candidate": candidate, "family": fam, "horizon": horizon, "period": period, **direct(panel, candidate, horizon, mask_period(panel, bounds), direction)})
            for mode, extra in [("ordinary", []), ("same_family", siblings)]:
                controls = parents + extra; train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) & (panel.timestamp < bounds[0] - pd.Timedelta(hours=horizon))).to_numpy(); test = mask_period(panel, bounds); valid = panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy(); tr, te = train & valid, test & valid
                if tr.sum() < 2000 or te.sum() < 800: continue
                X0, X1 = panel.loc[tr, controls].to_numpy(np.float32), panel.loc[te, controls].to_numpy(np.float32); c0 = direction * panel.loc[tr, candidate].to_numpy(np.float32); c1 = direction * panel.loc[te, candidate].to_numpy(np.float32); a0, a1 = panel.loc[tr, "asset"].to_numpy(), panel.loc[te, "asset"].to_numpy(); dates = panel.loc[te, "timestamp"].to_numpy()
                for target in ["raw", "std"]:
                    result, daily = ridge(X0, c0, panel.loc[tr, f"y{target}{horizon}"].to_numpy(float), X1, c1, panel.loc[te, f"y{target}{horizon}"].to_numpy(float), a0, a1, len(symbols), alpha, dates); annual.append({"candidate": candidate, "family": fam, "horizon": horizon, "period": period, "parent_mode": mode, "target": target, **result}); daily_rows.append(daily.assign(candidate=candidate, horizon=horizon, period=period, parent_mode=mode, target=target))
            train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) & (panel.timestamp < bounds[0] - pd.Timedelta(hours=horizon)) & panel.symbol.isin(train_symbols)).to_numpy(); test = mask_period(panel, bounds) & panel.symbol.isin(heldout).to_numpy(); valid = panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy(); tr, te = train & valid, test & valid
            if tr.sum() >= 2000 and te.sum() >= 500:
                X0, X1 = panel.loc[tr, parents].to_numpy(np.float32), panel.loc[te, parents].to_numpy(np.float32); c0 = direction * panel.loc[tr, candidate].to_numpy(np.float32); c1 = direction * panel.loc[te, candidate].to_numpy(np.float32); mapping = {s: i for i, s in enumerate(sorted(train_symbols))}; unseen = len(mapping); a0 = panel.loc[tr, "symbol"].map(mapping).to_numpy(np.int16); a1 = np.full(te.sum(), unseen, np.int16); dates = panel.loc[te, "timestamp"].to_numpy()
                for target in ["raw", "std"]:
                    result, daily = ridge(X0, c0, panel.loc[tr, f"y{target}{horizon}"].to_numpy(float), X1, c1, panel.loc[te, f"y{target}{horizon}"].to_numpy(float), a0, a1, unseen + 1, alpha, dates); annual.append({"candidate": candidate, "family": fam, "horizon": horizon, "period": period, "parent_mode": "heldout", "target": target, **result}); daily_rows.append(daily.assign(candidate=candidate, horizon=horizon, period=period, parent_mode="heldout", target=target))
    annual = pd.DataFrame(annual); direct_post = pd.DataFrame(direct_rows); daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame(); annual.to_csv(OUT / "post_annual.csv", index=False); direct_post.to_csv(OUT / "post_direct.csv", index=False); daily.to_csv(OUT / "post_daily.csv", index=False)
    boot_rows = []
    if not daily.empty:
        for keys, group in daily.groupby(["candidate", "horizon", "period", "parent_mode", "target"]): boot_rows.append(dict(zip(["candidate", "horizon", "period", "parent_mode", "target"], keys)) | bootstrap(group.sort_values("date").gain.to_numpy()))
    boot = pd.DataFrame(boot_rows); boot.to_csv(OUT / "bootstrap.csv", index=False); decisions, cost_frames = [], []; required = [p for p in ["validation", "holdout"] if p in set(annual.period)]
    for _, claim in frozen.iterrows():
        candidate, fam, horizon = claim.candidate, claim.family, int(claim.horizon); suba, subb = annual[annual.candidate == candidate], boot[boot.candidate == candidate]; subd = direct_post[direct_post.candidate == candidate]; model_pass = True
        for mode in ["ordinary", "same_family", "heldout"]:
            a = suba[(suba.parent_mode == mode) & suba.period.isin(required)]; b = subb[(subb.parent_mode == mode) & subb.period.isin(required)]
            if len(a) < 2 * len(required) or not (a.relative_improvement > 0).all(): model_pass = False
            if len(b) < 2 * len(required) or not (b.ci_low > 0).all(): model_pass = False
        direct_pass = all(not subd[subd.period == p].empty and subd[subd.period == p].iloc[0].tsic_raw > 0 and subd[subd.period == p].iloc[0].tsic_std > 0 for p in required); account_pass = False
        if model_pass and direct_pass:
            for period in required:
                frame = account(panel_full.loc[mask_period(panel_full, POST[period])], candidate, horizon, float(claim.direction))
                if not frame.empty: cost_frames.append(frame.assign(candidate=candidate, period=period, horizon=horizon))
            costs = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame(); five = costs[(costs.candidate == candidate) & (costs.cost_bps == 5) & costs.period.isin(required)] if not costs.empty else pd.DataFrame(); account_pass = len(five) == len(required) and (five.annual_log_return > 0).all() and (five.sharpe > 0).all()
        decisions.append({"candidate": candidate, "family": fam, "horizon": horizon, "direction": float(claim.direction), "alpha": float(claim.alpha), "direct_pass": bool(direct_pass), "model_input_pass": bool(model_pass and direct_pass), "cost_pass": bool(account_pass), "admit_model_input": bool(model_pass and direct_pass), "independent_alpha": bool(model_pass and direct_pass and account_pass)})
    decision = pd.DataFrame(decisions); decision.to_csv(OUT / "admission.csv", index=False); costs = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame(); costs.to_csv(OUT / "costs.csv", index=False); admitted = decision[decision.admit_model_input].to_dict("records"); independent = decision[decision.independent_alpha].to_dict("records")
    final = {"status": "complete", "data_audit": audit, "symbols": symbols, "heldout_symbols": sorted(heldout), "candidate_count": len(candidates), "parent_count": len(parents), "shortlist_count": len(shortlist), "frozen_count": len(frozen), "admitted_model_inputs": admitted, "independent_alpha": independent, "elapsed_seconds": time.time() - started}; (OUT / "DECISION.json").write_text(json.dumps(final, indent=2, default=str)); (OUT / "metadata.json").write_text(json.dumps({"candidates": candidates, "families": {x: family(x) for x in candidates}, "parents": parents, "horizons": HORIZONS, "windows": WINDOWS, "lags": LAGS, "clock": "strict-prior impact kernel; one daily issue; next-hour-open execution", "source": "Binance Data Vision spot hourly klines", "adaptive_history": True}, indent=2)); print(json.dumps({"admitted": admitted, "independent": independent, "elapsed_seconds": final["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__": main()
