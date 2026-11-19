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
START = os.environ.get("SENSEMBLE_START", "2021-01")
END = os.environ.get("SENSEMBLE_END", "2026-10")
WINDOWS = (720, 2160)
HORIZONS = (1, 4, 24, 120, 480)
REFRESH = 24
S_GRID = np.array([-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5])
EPS = 1e-12
ALPHA = 0.5
DEV = {
    "devA": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:00", tz="UTC")),
    "devB": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:00", tz="UTC")),
}
POST = {
    "validation": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:00", tz="UTC")),
    "holdout": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:00", tz="UTC")),
    "confirmation": (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-10-31 23:00", tz="UTC")),
}
STATE_COORDS = {
    "ra4": np.array([-1.0, -1.0, 1.0, 1.0]),
    "r3a2": np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]),
    "r2v2a2": np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
}
LOWER = [
    "physical_drift", "physical_activity", "transition_entropy", "spectral_gap",
    "physical_committor", "physical_escape", "chi0", "chi_peak", "s_star",
    "tilted_gap", "active_drift", "inactive_drift", "active_committor",
    "inactive_committor", "active_kl", "inactive_kl", "active_activity",
    "inactive_activity", "psi_skew",
]
FORMULAS = {
    "doob_drive": "doob_transport",
    "critical_doob_drive": "doob_transport",
    "gap_doob_drive": "doob_transport",
    "committor_shift": "doob_committor",
    "critical_committor": "doob_committor",
    "kl_asymmetric_drive": "doob_entropy",
    "activity_gap_drive": "dynamic_activity",
    "instanton_shift": "first_passage_instanton",
    "critical_instanton": "first_passage_instanton",
    "phase_boundary_drive": "dynamic_phase_boundary",
    "scgf_skew_transport": "dynamic_phase_boundary",
    "doob_entropy_transport": "doob_entropy",
}
BASIC_PARENTS = [
    "ret1", "mom4", "mom24", "mom72", "mom168", "mom480",
    "vol24", "vol72", "vol168", "vol720", "ret_z168", "absret_z168",
    "activity_z168", "activity_z720", "range_z168", "range_z720",
    "close_location", "taker_imbalance", "taker_imbalance_z168",
    "ret_autocorr24", "ret_autocorr168", "activity_change24",
    "volume_momentum24", "skew168", "kurt168",
]


def months(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")]


def download(symbol: str, month: str):
    target = CACHE / f"{symbol}-1h-{month}.zip"
    if target.exists() and target.stat().st_size > 100:
        return symbol, month, target.read_bytes(), None
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers={"User-Agent": "statphys-research/1.0"}, timeout=40)
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
    jobs = [(s, m) for s in SYMBOLS for m in months(START, END)]
    parts = {s: [] for s in SYMBOLS}
    errors = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        pending = {pool.submit(download, s, m): (s, m) for s, m in jobs}
        for future in as_completed(pending):
            s, m = pending[future]
            try:
                _, _, payload, error = future.result()
            except Exception as exc:
                payload, error = None, f"future:{type(exc).__name__}:{exc}"
            if payload is None:
                if error != "404":
                    errors.append({"symbol": s, "month": m, "error": error})
                continue
            try:
                parts[s].append(parse_zip(s, payload))
            except Exception as exc:
                errors.append({"symbol": s, "month": m, "error": f"parse:{type(exc).__name__}:{exc}"})
    data = {}
    for symbol, frames in parts.items():
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp", "close"])
        frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").set_index("timestamp")
        if len(frame) >= 20000:
            data[symbol] = frame
    if len(data) < 8:
        raise RuntimeError(f"only {len(data)} symbols loaded: {list(data)}")
    audit = {
        "source": "Binance Data Vision spot monthly 1h klines",
        "start": START, "end": END, "symbols": sorted(data),
        "rows": {k: int(len(v)) for k, v in data.items()},
        "date_min": {k: str(v.index.min()) for k, v in data.items()},
        "date_max": {k: str(v.index.max()) for k, v in data.items()},
        "errors_non_404": errors[:200],
        "sha256": {k: hashlib.sha256(pd.util.hash_pandas_object(v.reset_index(), index=False).values.tobytes()).hexdigest() for k, v in data.items()},
    }
    return data, audit


def prior_z(series: pd.Series, window: int, minimum: int | None = None):
    minimum = minimum or max(24, window // 4)
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0)
    return ((series - mean) / (std + EPS)).clip(-8, 8)


def features(raw: pd.DataFrame):
    out = pd.DataFrame(index=raw.index)
    log_close = np.log(raw.close.where(raw.close > 0))
    log_open = np.log(raw.open.where(raw.open > 0))
    ret = log_close.diff()
    activity = np.log1p(raw.quote_volume.clip(lower=0))
    range_state = np.log(raw.high.where(raw.high > 0) / raw.low.where(raw.low > 0))
    taker = 2 * raw.taker_quote / (raw.quote_volume + EPS) - 1
    out["ret1"] = ret
    for h in [4, 24, 72, 168, 480]:
        out[f"mom{h}"] = ret.rolling(h, min_periods=max(2, h // 2)).sum()
    for w in [24, 72, 168, 720]:
        out[f"vol{w}"] = ret.rolling(w, min_periods=max(12, w // 3)).std(ddof=0)
    out["ret_z168"] = prior_z(ret, 168, 48)
    out["absret_z168"] = prior_z(ret.abs(), 168, 48)
    out["activity_z168"] = prior_z(activity, 168, 48)
    out["activity_z720"] = prior_z(activity, 720, 168)
    out["range_z168"] = prior_z(range_state, 168, 48)
    out["range_z720"] = prior_z(range_state, 720, 168)
    out["close_location"] = (raw.close - raw.low) / (raw.high - raw.low + EPS)
    out["taker_imbalance"] = taker
    out["taker_imbalance_z168"] = prior_z(taker, 168, 48)
    out["ret_autocorr24"] = ret.rolling(24, min_periods=12).corr(ret.shift(1))
    out["ret_autocorr168"] = ret.rolling(168, min_periods=56).corr(ret.shift(1))
    out["activity_change24"] = activity.diff(24)
    out["volume_momentum24"] = prior_z(activity.diff(24), 168, 48)
    out["skew168"] = ret.rolling(168, min_periods=72).skew()
    out["kurt168"] = ret.rolling(168, min_periods=72).kurt()
    out["next_ret1"] = log_close.shift(-1) - log_open.shift(-1)
    prior_vol = ret.shift(1).rolling(720, min_periods=168).std(ddof=0)
    for h in HORIZONS:
        target = log_close.shift(-h) - log_open.shift(-1)
        out[f"yraw{h}"] = target
        out[f"ystd{h}"] = target / (prior_vol * math.sqrt(h) + EPS)
    return out.astype(np.float32)


def states(feat: pd.DataFrame):
    r = feat.ret_z168.to_numpy(float)
    a = feat.activity_z168.to_numpy(float)
    v = feat.absret_z168.to_numpy(float)
    result = {}
    valid = np.isfinite(r) & np.isfinite(a)
    x = np.full(len(feat), -1, np.int16)
    x[valid] = (r[valid] >= 0).astype(np.int16) * 2 + (a[valid] >= 0).astype(np.int16)
    result["ra4"] = x
    valid = np.isfinite(r) & np.isfinite(a)
    rb = np.digitize(r, [-0.75, 0.75]).astype(np.int16)
    x = np.full(len(feat), -1, np.int16)
    x[valid] = rb[valid] * 2 + (a[valid] >= 0).astype(np.int16)
    result["r3a2"] = x
    valid = np.isfinite(r) & np.isfinite(a) & np.isfinite(v)
    x = np.full(len(feat), -1, np.int16)
    x[valid] = (r[valid] >= 0).astype(np.int16) * 4 + (v[valid] >= 0).astype(np.int16) * 2 + (a[valid] >= 0).astype(np.int16)
    result["r2v2a2"] = x
    return result


def perron(matrix):
    values, vectors = np.linalg.eig(matrix)
    k = int(np.argmax(values.real))
    lam = max(float(values[k].real), EPS)
    right = np.maximum(np.abs(np.real(vectors[:, k])), 1e-10)
    other = np.delete(np.abs(values), k)
    gap = max(0.0, 1 - (float(np.max(other)) / lam if len(other) else 0.0))
    return lam, right, gap


def driven_matrix(P, activity, s):
    tilted = P * np.exp(-s * activity)
    lam, right, gap = perron(tilted)
    driven = tilted * (right[None, :] / right[:, None]) / lam
    driven = np.maximum(driven, 0)
    driven /= driven.sum(axis=1, keepdims=True) + EPS
    return driven, math.log(lam), gap


def committor(P, coordinate):
    pos = coordinate > 0.25
    neg = coordinate < -0.25
    transient = ~(pos | neg)
    q = np.zeros(len(coordinate))
    q[pos] = 1
    if transient.any():
        Q = P[np.ix_(transient, transient)]
        rhs = P[np.ix_(transient, pos)].sum(axis=1)
        A = np.eye(Q.shape[0]) - Q + 1e-8 * np.eye(Q.shape[0])
        try:
            q[transient] = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            q[transient] = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.clip(q, 0, 1)


def mfpt(P, target):
    transient = ~target
    out = np.zeros(P.shape[0])
    if transient.any():
        Q = P[np.ix_(transient, transient)]
        A = np.eye(Q.shape[0]) - Q + 1e-8 * np.eye(Q.shape[0])
        try:
            out[transient] = np.linalg.solve(A, np.ones(Q.shape[0]))
        except np.linalg.LinAlgError:
            out[transient] = np.linalg.lstsq(A, np.ones(Q.shape[0]), rcond=None)[0]
    return np.maximum(out, 0)


def escape(P, coordinate):
    up = mfpt(P, coordinate > 0.25)
    down = mfpt(P, coordinate < -0.25)
    return 1 / (1 + up) - 1 / (1 + down)


def trajectory_coordinates(state, coordinate, window):
    n = len(state)
    lower = {x: np.full(n, np.nan, np.float32) for x in LOWER}
    candidates = {x: np.full(n, np.nan, np.float32) for x in FORMULAS}
    nstates = len(coordinate)
    activity = (np.arange(nstates)[:, None] != np.arange(nstates)[None, :]).astype(float)
    for t in range(window, n, REFRESH):
        history = state[t-window:t]
        valid = (history[:-1] >= 0) & (history[1:] >= 0)
        if valid.sum() < max(200, window // 3):
            continue
        counts = np.full((nstates, nstates), ALPHA)
        np.add.at(counts, (history[:-1][valid], history[1:][valid]), 1)
        P = counts / counts.sum(axis=1, keepdims=True)
        pgap = perron(P)[2]
        entropy = -(P * np.log(P + EPS)).sum(axis=1)
        pdrift = P @ coordinate - coordinate
        pact = (P * activity).sum(axis=1)
        pq = committor(P, coordinate)
        pesc = escape(P, coordinate)
        driven = {}
        psi, gaps = [], []
        for s in S_GRID:
            M, value, gap = driven_matrix(P, activity, float(s))
            driven[float(s)] = M
            psi.append(value)
            gaps.append(gap)
        psi = np.array(psi)
        second = np.gradient(np.gradient(psi, S_GRID), S_GRID)
        zero = int(np.where(S_GRID == 0)[0][0])
        peak = int(np.nanargmax(second))
        chi0 = max(float(second[zero]), 0)
        chipeak = max(float(second[peak]), 0)
        sstar = float(S_GRID[peak])
        tgap = float(np.min(gaps))
        pskew = float(psi[np.where(S_GRID == -1)[0][0]] - psi[np.where(S_GRID == 1)[0][0]])
        active = driven[-1.0]
        inactive = driven[1.0]
        adrift = active @ coordinate - coordinate
        idrift = inactive @ coordinate - coordinate
        aq, iq = committor(active, coordinate), committor(inactive, coordinate)
        aesc, iesc = escape(active, coordinate), escape(inactive, coordinate)
        aact, iact = (active * activity).sum(axis=1), (inactive * activity).sum(axis=1)
        akl = (active * np.log((active + EPS) / (P + EPS))).sum(axis=1)
        ikl = (inactive * np.log((inactive + EPS) / (P + EPS))).sum(axis=1)
        for row in range(t, min(t + REFRESH, n)):
            st = int(state[row])
            if st < 0:
                continue
            drive = float(adrift[st] - idrift[st])
            qshift = float(aq[st] - iq[st])
            inst = float(aesc[st] - iesc[st])
            values = {
                "physical_drift": pdrift[st], "physical_activity": pact[st],
                "transition_entropy": entropy[st], "spectral_gap": pgap,
                "physical_committor": pq[st], "physical_escape": pesc[st],
                "chi0": chi0, "chi_peak": chipeak, "s_star": sstar,
                "tilted_gap": tgap, "active_drift": adrift[st],
                "inactive_drift": idrift[st], "active_committor": aq[st],
                "inactive_committor": iq[st], "active_kl": akl[st],
                "inactive_kl": ikl[st], "active_activity": aact[st],
                "inactive_activity": iact[st], "psi_skew": pskew,
            }
            for name, value in values.items():
                lower[name][row] = np.float32(value)
            cv = {
                "doob_drive": drive,
                "critical_doob_drive": drive * math.sqrt(chipeak + EPS),
                "gap_doob_drive": drive / math.sqrt(tgap + 0.05),
                "committor_shift": qshift,
                "critical_committor": qshift * math.sqrt(chipeak + EPS),
                "kl_asymmetric_drive": math.copysign(abs(akl[st] - ikl[st]), drive) if drive else 0,
                "activity_gap_drive": (aact[st] - iact[st]) * drive,
                "instanton_shift": inst,
                "critical_instanton": inst * math.sqrt(chipeak + EPS),
                "phase_boundary_drive": -sstar * drive,
                "scgf_skew_transport": pskew * coordinate[st],
                "doob_entropy_transport": (akl[st] + ikl[st]) * drive,
            }
            for name, value in cv.items():
                candidates[name][row] = np.float32(value)
    return pd.DataFrame(lower), pd.DataFrame(candidates)


def symbol_panel(symbol, raw):
    feat = features(raw)
    pieces = [feat]
    for state_name, array in states(feat).items():
        for window in WINDOWS:
            lower, candidate = trajectory_coordinates(array, STATE_COORDS[state_name], window)
            lower.index = feat.index
            candidate.index = feat.index
            lower.columns = [f"lower__{state_name}__w{window}__{x}" for x in lower]
            candidate.columns = [f"candidate__{FORMULAS[x]}__{state_name}__w{window}__{x}" for x in candidate]
            for column in lower:
                lower[column] = prior_z(lower[column], 2160, 360).astype(np.float32)
            for column in candidate:
                candidate[column] = np.tanh(prior_z(candidate[column], 2160, 360) / 3).astype(np.float32)
            pieces.extend([lower, candidate])
    result = pd.concat(pieces, axis=1)
    result["symbol"] = symbol
    result["timestamp"] = result.index
    return result.reset_index(drop=True)


def corr(x, y, minimum=100):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < minimum:
        return np.nan
    a, b = x[valid].astype(float), y[valid].astype(float)
    a -= a.mean(); b -= b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 1e-15 else np.nan


def mask_period(panel, bounds):
    return ((panel.timestamp >= bounds[0]) & (panel.timestamp <= bounds[1])).to_numpy()


def direct(panel, signal, horizon, mask, direction=1.0):
    raw, std, obs = [], [], []
    for _, group in panel.loc[mask].groupby("symbol", sort=False):
        x = direction * group[signal].to_numpy(float)
        r = corr(x, group[f"yraw{horizon}"].to_numpy(float), 200)
        s = corr(x, group[f"ystd{horizon}"].to_numpy(float), 200)
        if np.isfinite(r) and np.isfinite(s):
            raw.append(r); std.append(s); obs.append(len(group))
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
        "n_assets": len(raw), "mean_obs": float(np.mean(obs)) if obs else 0,
        "rankic_hours": len(rank_raw),
    }


def family(name):
    return name.split("__")[1]


def transform(X0, a0, X1, a1, nassets):
    means = np.full((nassets, X0.shape[1]), np.nan, np.float32)
    for j in range(nassets):
        m = a0 == j
        if m.any():
            means[j] = np.nanmean(X0[m], axis=0)
    global_mean = np.nanmean(X0, axis=0).astype(np.float32)
    bad = ~np.isfinite(means)
    means[bad] = np.broadcast_to(global_mean, means.shape)[bad]
    A = np.where(np.isfinite(X0), X0 - means[a0], 0).astype(np.float32)
    B = np.where(np.isfinite(X1), X1 - means[a1], 0).astype(np.float32)
    scale = np.sqrt(np.mean(A.astype(float) ** 2, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1).astype(np.float32)
    return np.clip(A / scale, -10, 10), np.clip(B / scale, -10, 10)


def center_y(y, assets, nassets):
    means = np.full(nassets, np.nan)
    for j in range(nassets):
        m = assets == j
        if m.any():
            means[j] = np.mean(y[m])
    means[~np.isfinite(means)] = np.mean(y)
    return y - means[assets], means


def ridge(X0, c0, y0, X1, c1, y1, a0, a1, nassets, alpha, dates):
    X0, X1 = transform(X0, a0, X1, a1, nassets)
    c0, c1 = transform(c0[:, None], a0, c1[:, None], a1, nassets)
    c0, c1 = c0[:, 0].astype(float), c1[:, 0].astype(float)
    yc, intercept = center_y(y0.astype(float), a0, nassets)
    X = X0.astype(float)
    G = X.T @ X
    G.flat[::G.shape[0] + 1] += 10
    beta = np.linalg.solve(G, X.T @ yc)
    cross = X.T @ c0
    gcross = np.linalg.solve(G, cross)
    schur = float(c0 @ c0 - cross @ gcross)
    coef = float(c0 @ yc - cross @ beta) / (schur + alpha)
    base = X1.astype(float) @ beta + intercept[a1]
    augmented = base + (c1 - X1.astype(float) @ gcross) * coef
    l0, l1 = (y1 - base) ** 2, (y1 - augmented) ** 2
    daily = pd.DataFrame({"date": pd.to_datetime(dates).floor("D"), "base": l0, "aug": l1}).groupby("date").mean()
    daily["gain"] = daily.base - daily.aug
    result = {
        "base_mse": float(daily.base.mean()), "aug_mse": float(daily.aug.mean()),
        "relative_improvement": float((daily.base.mean() - daily.aug.mean()) / daily.base.mean()),
        "candidate_coef": coef, "schur": schur, "train_n": len(y0), "test_n": len(y1),
    }
    return result, daily.reset_index()


def bootstrap(gain, block=30, draws=1999):
    gain = np.asarray(gain, float)
    gain = gain[np.isfinite(gain)]
    if len(gain) < 60:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_le_zero": np.nan, "n": len(gain)}
    starts = np.arange(len(gain) - block + 1)
    blocks = int(math.ceil(len(gain) / block))
    rng = np.random.default_rng(20261119)
    values = np.empty(draws)
    for i in range(draws):
        chosen = rng.choice(starts, size=blocks, replace=True)
        index = np.concatenate([np.arange(x, x + block) for x in chosen])[:len(gain)]
        values[i] = gain[index].mean()
    return {"ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
            "p_le_zero": float((1 + (values <= 0).sum()) / (draws + 1)), "n": len(gain)}


def account(panel, signal, horizon, direction):
    subset = panel[["timestamp", "symbol", signal, "next_ret1"]].dropna().copy()
    issue = subset.loc[subset["timestamp"].dt.hour.eq(0)].copy()
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
        rows.append(pd.DataFrame({
            "timestamp": timestamp,
            "symbol": group.symbol.to_numpy(),
            "position": position,
        }))
    if not rows:
        return pd.DataFrame()
    issue_positions = (
        pd.concat(rows)
        .pivot(index="timestamp", columns="symbol", values="position")
        .fillna(0)
        .sort_index()
    )
    returns = (
        subset.pivot(index="timestamp", columns="symbol", values="next_ret1")
        .fillna(0)
        .sort_index()
    )
    columns = sorted(set(issue_positions.columns) | set(returns.columns))
    returns = returns.reindex(columns=columns, fill_value=0)
    issued = issue_positions.reindex(index=returns.index, columns=columns, fill_value=0)
    sleeve_count = max(1, int(math.ceil(horizon / 24)))
    effective = issued.rolling(horizon, min_periods=1).sum() / sleeve_count
    turnover = effective.diff().abs().sum(axis=1).fillna(effective.abs().sum(axis=1))
    gross = (effective * returns).sum(axis=1)
    output = []
    for bps in [0, 2, 5, 10]:
        net = gross - turnover * bps / 10000
        equity = net.cumsum()
        drawdown = equity - equity.cummax()
        annual = float(net.mean() * 24 * 365)
        vol = float(net.std(ddof=0) * math.sqrt(24 * 365))
        output.append({
            "cost_bps": bps,
            "annual_log_return": annual,
            "sharpe": annual / vol if vol > 1e-12 else np.nan,
            "max_drawdown": float(drawdown.min()),
            "mean_hourly_turnover": float(turnover.mean()),
            "hours": len(net),
        })
    return pd.DataFrame(output)


def main():
    started = time.time()
    market, audit = load_data()
    panels = []
    for i, (symbol, raw) in enumerate(sorted(market.items()), 1):
        print(f"build {symbol} {i}/{len(market)}", flush=True)
        panels.append(symbol_panel(symbol, raw))
    panel = pd.concat(panels, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    panel.timestamp = pd.to_datetime(panel.timestamp, utc=True)
    symbols = sorted(panel.symbol.unique())
    amap = {s: i for i, s in enumerate(symbols)}
    panel["asset"] = panel.symbol.map(amap).astype(np.int16)
    panel_full = panel.copy()
    panel = panel.loc[panel["timestamp"].dt.hour.eq(0)].copy().reset_index(drop=True)
    candidates = [x for x in panel if x.startswith("candidate__")]
    lower = [x for x in panel if x.startswith("lower__")]
    parents = BASIC_PARENTS + lower

    dev_rows = []
    for candidate in candidates:
        for horizon in HORIZONS:
            metrics = {name: direct(panel, candidate, horizon, mask_period(panel, bounds)) for name, bounds in DEV.items()}
            values = [metrics[p][k] for p in DEV for k in ["tsic_raw", "tsic_std"]]
            direction = float(np.sign(np.nanmean(values))) if np.isfinite(values).any() else 0
            oriented = [direction * x for x in values]
            dev_rows.append({"candidate": candidate, "family": family(candidate), "horizon": horizon,
                             "direction": direction, "min_oriented_tsic": float(np.nanmin(oriented)),
                             "mean_oriented_tsic": float(np.nanmean(oriented)),
                             **{f"{p}_{k}": v for p, m in metrics.items() for k, v in m.items()}})
    dev_direct = pd.DataFrame(dev_rows)
    dev_direct.to_csv(OUT / "development_direct.csv", index=False)
    gate = dev_direct[(dev_direct.direction != 0) & (dev_direct.min_oriented_tsic > .004) &
                      (dev_direct.devA_n_assets >= 8) & (dev_direct.devB_n_assets >= 8)]
    shortlist = []
    for fam, group in gate.sort_values(["family", "min_oriented_tsic"], ascending=[True, False]).groupby("family"):
        shortlist.extend(group.head(2).to_dict("records"))
    shortlist = pd.DataFrame(shortlist)
    shortlist.to_csv(OUT / "development_shortlist.csv", index=False)
    if shortlist.empty:
        (OUT / "DECISION.json").write_text(json.dumps({"status": "no_direct_candidate", "admitted": []}, indent=2))
        return

    model_rows = []
    for _, claim in shortlist.iterrows():
        candidate, horizon, direction = claim.candidate, int(claim.horizon), float(claim.direction)
        for year in [2022, 2023]:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC"); end = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")
            train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) &
                     (panel.timestamp < start - pd.Timedelta(hours=horizon))).to_numpy()
            test = ((panel.timestamp >= start) & (panel.timestamp <= end)).to_numpy()
            cols = parents + [candidate, f"yraw{horizon}", f"ystd{horizon}"]
            tr = train & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy(); te = test & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy()
            if tr.sum() < 2500 or te.sum() < 1200:
                continue
            X0, X1 = panel.loc[tr, parents].to_numpy(np.float32), panel.loc[te, parents].to_numpy(np.float32)
            c0 = direction * panel.loc[tr, candidate].to_numpy(np.float32)
            c1 = direction * panel.loc[te, candidate].to_numpy(np.float32)
            a0, a1 = panel.loc[tr, "asset"].to_numpy(), panel.loc[te, "asset"].to_numpy()
            d1 = panel.loc[te, "timestamp"].to_numpy()
            for target in ["raw", "std"]:
                y0 = panel.loc[tr, f"y{target}{horizon}"].to_numpy(float)
                y1 = panel.loc[te, f"y{target}{horizon}"].to_numpy(float)
                for alpha in [10., 1000., 100000.]:
                    result, _ = ridge(X0, c0, y0, X1, c1, y1, a0, a1, len(symbols), alpha, d1)
                    model_rows.append({"candidate": candidate, "family": claim.family, "horizon": horizon,
                                       "direction": direction, "year": year, "target": target,
                                       "alpha": alpha, **result})
    model = pd.DataFrame(model_rows)
    model.to_csv(OUT / "development_model_grid.csv", index=False)
    frozen_rows = []
    for keys, group in model.groupby(["candidate", "family", "horizon", "direction", "alpha"]):
        lookup = {(int(x.year), x.target): x.relative_improvement for x in group.itertuples()}
        values = [lookup.get((y, t), np.nan) for y in [2022, 2023] for t in ["raw", "std"]]
        if np.isfinite(values).all() and min(values) > 0:
            frozen_rows.append(dict(zip(["candidate", "family", "horizon", "direction", "alpha"], keys)) |
                               {"min_dev_increment": min(values), "mean_dev_increment": float(np.mean(values))})
    fg = pd.DataFrame(frozen_rows)
    frozen = []
    if not fg.empty:
        robust = fg.sort_values(["candidate", "horizon", "alpha"], ascending=[True, True, False]).groupby(
            ["candidate", "family", "horizon", "direction"], as_index=False).head(1)
        for fam, group in robust.groupby("family"):
            frozen.extend(group.sort_values(["min_dev_increment", "mean_dev_increment"], ascending=False).head(2).to_dict("records"))
    frozen = pd.DataFrame(frozen)
    frozen.to_csv(OUT / "frozen_claims.csv", index=False)
    if frozen.empty:
        (OUT / "DECISION.json").write_text(json.dumps({"status": "no_strong_parent_candidate", "admitted": []}, indent=2))
        return

    heldout = {s for s in symbols if int(hashlib.sha256(s.encode()).hexdigest(), 16) % 3 == 0}
    train_symbols = set(symbols) - heldout
    annual, daily_rows, direct_rows = [], [], []
    for _, claim in frozen.iterrows():
        candidate, fam, horizon = claim.candidate, claim.family, int(claim.horizon)
        direction, alpha = float(claim.direction), float(claim.alpha)
        siblings = [x for x in candidates if family(x) == fam and x != candidate]
        for period, bounds in POST.items():
            if panel.timestamp.max() < bounds[0]:
                continue
            direct_rows.append({"candidate": candidate, "family": fam, "horizon": horizon,
                                "period": period, **direct(panel, candidate, horizon, mask_period(panel, bounds), direction)})
            for mode, extra in [("ordinary", []), ("same_family", siblings)]:
                controls = parents + extra
                train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) &
                         (panel.timestamp < bounds[0] - pd.Timedelta(hours=horizon))).to_numpy()
                test = mask_period(panel, bounds)
                cols = controls + [candidate, f"yraw{horizon}", f"ystd{horizon}"]
                tr = train & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy(); te = test & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy()
                if tr.sum() < 2500 or te.sum() < 1000:
                    continue
                X0, X1 = panel.loc[tr, controls].to_numpy(np.float32), panel.loc[te, controls].to_numpy(np.float32)
                c0 = direction * panel.loc[tr, candidate].to_numpy(np.float32)
                c1 = direction * panel.loc[te, candidate].to_numpy(np.float32)
                a0, a1 = panel.loc[tr, "asset"].to_numpy(), panel.loc[te, "asset"].to_numpy()
                dates = panel.loc[te, "timestamp"].to_numpy()
                for target in ["raw", "std"]:
                    y0 = panel.loc[tr, f"y{target}{horizon}"].to_numpy(float)
                    y1 = panel.loc[te, f"y{target}{horizon}"].to_numpy(float)
                    result, daily = ridge(X0, c0, y0, X1, c1, y1, a0, a1, len(symbols), alpha, dates)
                    annual.append({"candidate": candidate, "family": fam, "horizon": horizon,
                                   "period": period, "parent_mode": mode, "target": target, **result})
                    daily_rows.append(daily.assign(candidate=candidate, horizon=horizon, period=period,
                                                   parent_mode=mode, target=target))
            train = ((panel.timestamp >= pd.Timestamp("2021-01-01", tz="UTC")) &
                     (panel.timestamp < bounds[0] - pd.Timedelta(hours=horizon)) & panel.symbol.isin(train_symbols)).to_numpy()
            test = mask_period(panel, bounds) & panel.symbol.isin(heldout).to_numpy()
            cols = parents + [candidate, f"yraw{horizon}", f"ystd{horizon}"]
            tr = train & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy(); te = test & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]].notna().all(axis=1).to_numpy()
            if tr.sum() >= 2500 and te.sum() >= 700:
                X0, X1 = panel.loc[tr, parents].to_numpy(np.float32), panel.loc[te, parents].to_numpy(np.float32)
                c0 = direction * panel.loc[tr, candidate].to_numpy(np.float32)
                c1 = direction * panel.loc[te, candidate].to_numpy(np.float32)
                mapping = {s: i for i, s in enumerate(sorted(train_symbols))}; unseen = len(mapping)
                a0 = panel.loc[tr, "symbol"].map(mapping).to_numpy(np.int16)
                a1 = np.full(te.sum(), unseen, np.int16); dates = panel.loc[te, "timestamp"].to_numpy()
                for target in ["raw", "std"]:
                    y0 = panel.loc[tr, f"y{target}{horizon}"].to_numpy(float)
                    y1 = panel.loc[te, f"y{target}{horizon}"].to_numpy(float)
                    result, daily = ridge(X0, c0, y0, X1, c1, y1, a0, a1, unseen + 1, alpha, dates)
                    annual.append({"candidate": candidate, "family": fam, "horizon": horizon,
                                   "period": period, "parent_mode": "heldout", "target": target, **result})
                    daily_rows.append(daily.assign(candidate=candidate, horizon=horizon, period=period,
                                                   parent_mode="heldout", target=target))
    annual = pd.DataFrame(annual); direct_post = pd.DataFrame(direct_rows)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    annual.to_csv(OUT / "post_annual.csv", index=False); direct_post.to_csv(OUT / "post_direct.csv", index=False)
    daily.to_csv(OUT / "post_daily.csv", index=False)
    boot_rows = []
    if not daily.empty:
        for keys, group in daily.groupby(["candidate", "horizon", "period", "parent_mode", "target"]):
            boot_rows.append(dict(zip(["candidate", "horizon", "period", "parent_mode", "target"], keys)) |
                             bootstrap(group.sort_values("date").gain.to_numpy()))
    boot = pd.DataFrame(boot_rows); boot.to_csv(OUT / "bootstrap.csv", index=False)

    decisions, cost_frames = [], []
    required = [p for p in ["validation", "holdout"] if p in set(annual.period)]
    for _, claim in frozen.iterrows():
        candidate, fam, horizon = claim.candidate, claim.family, int(claim.horizon)
        suba, subb = annual[annual.candidate == candidate], boot[boot.candidate == candidate]
        subd = direct_post[direct_post.candidate == candidate]
        model_pass = True
        for mode in ["ordinary", "same_family", "heldout"]:
            a = suba[(suba.parent_mode == mode) & suba.period.isin(required)]
            b = subb[(subb.parent_mode == mode) & subb.period.isin(required)]
            if len(a) < 2 * len(required) or not (a.relative_improvement > 0).all(): model_pass = False
            if len(b) < 2 * len(required) or not (b.ci_low > 0).all(): model_pass = False
        direct_pass = all(
            not subd[subd.period == p].empty and subd[subd.period == p].iloc[0].tsic_raw > 0 and subd[subd.period == p].iloc[0].tsic_std > 0
            for p in required
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
        decisions.append({"candidate": candidate, "family": fam, "horizon": horizon,
                          "direction": float(claim.direction), "alpha": float(claim.alpha),
                          "direct_pass": bool(direct_pass), "model_input_pass": bool(model_pass and direct_pass),
                          "cost_pass": bool(account_pass), "admit_model_input": bool(model_pass and direct_pass),
                          "independent_alpha": bool(model_pass and direct_pass and account_pass)})
    decision_frame = pd.DataFrame(decisions); decision_frame.to_csv(OUT / "admission.csv", index=False)
    costs = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame(); costs.to_csv(OUT / "costs.csv", index=False)
    admitted = decision_frame[decision_frame.admit_model_input].to_dict("records")
    independent = decision_frame[decision_frame.independent_alpha].to_dict("records")
    final = {"status": "complete", "data_audit": audit, "symbols": symbols,
             "heldout_symbols": sorted(heldout), "candidate_count": len(candidates),
             "parent_count": len(parents), "shortlist_count": len(shortlist),
             "frozen_count": len(frozen), "admitted_model_inputs": admitted,
             "independent_alpha": independent, "elapsed_seconds": time.time() - started}
    (OUT / "DECISION.json").write_text(json.dumps(final, indent=2, default=str))
    (OUT / "metadata.json").write_text(json.dumps({"candidates": candidates,
        "families": {x: family(x) for x in candidates}, "parents": parents,
        "horizons": HORIZONS, "windows": WINDOWS, "s_grid": S_GRID.tolist(),
        "clock": "transition law strictly prior to each 24-hour refresh; current completed hour selects state readout",
        "source": "Binance Data Vision spot hourly klines", "adaptive_history": True}, indent=2))
    print(json.dumps({"admitted": admitted, "independent": independent,
                      "elapsed_seconds": final["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
