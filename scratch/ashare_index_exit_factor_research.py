#!/usr/bin/env python3
"""Frozen cross-index experiment for an A-share index-exit event Alpha.

The independent observation is a point-in-time index-membership transition,
not a transformation of price or volume.  Candidate conditions are selected
only on 2016-2021 CSI300/CSI500-union exits.  The selected condition and
orientation are then tested on 2022-2026 source exits and on CSI1000 exits that
do not immediately promote into CSI300/CSI500.

Issue time is the close of the first completed session outside the index.  The
target is next-session open to the fifth subsequent-session close, relative to
the same-date median among active CSI300/CSI500/CSI1000 members.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from sklearn.linear_model import Ridge

LOGGER = logging.getLogger("index-exit-alpha")
RANDOM_SEED = 22731
MATCH_K = 5
MAX_RIDGE_TRAIN = 400_000
SOURCE_MARKETS = ("csi300", "csi500")
TARGET_MARKET = "csi1000"
RELEASE_SHA256 = "7c796f66efb54053ec84d747cb8cb7612188d4f9c7a8efde6fcf6d3c55260d1c"

MODEL_FEATURES = [
    "pre_ret20",
    "pre_ret60",
    "pre_vol20",
    "pre_path_eff20",
    "pre_log_amount20",
    "ret1",
    "overnight",
    "intraday",
    "volume_shock",
    "log_range",
    "close_location",
]
PRE_MATCH_FEATURES = MODEL_FEATURES[:5]
STRICT_MATCH_FEATURES = MODEL_FEATURES
CANDIDATES = (
    "all_exit",
    "negative_exit_pressure",
    "positive_exit_pressure",
    "negative_pre_exit_runup",
    "positive_pre_exit_runup",
    "negative_runup_and_pressure",
    "positive_runup_and_pressure",
)
FACTOR_IDS = {
    "all_exit": "INDEX_EXIT_POST_EFFECT_RESPONSE5",
    "negative_exit_pressure": "NEGATIVE_INDEX_EXIT_PRESSURE_RESPONSE5",
    "positive_exit_pressure": "POSITIVE_INDEX_EXIT_PRESSURE_RESPONSE5",
    "negative_pre_exit_runup": "PRE_EXIT_WEAKNESS_RESPONSE5",
    "positive_pre_exit_runup": "PRE_EXIT_RUNUP_RESPONSE5",
    "negative_runup_and_pressure": "INDEX_EXIT_WEAKNESS_CONTINUATION5",
    "positive_runup_and_pressure": "INDEX_EXIT_RUNUP_EXHAUSTION5",
}


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str


PERIODS = (
    Period("development", "2016-01-01", "2018-12-31"),
    Period("selection", "2019-01-01", "2021-12-31"),
    Period("confirmation", "2022-01-01", "2024-12-31"),
    Period("final", "2025-01-01", "2026-10-31"),
)


@dataclass(frozen=True)
class EventMetric:
    universe: str
    candidate: str
    period: str
    n_events: int
    n_dates: int
    mean_target_bp: float
    median_target_bp: float
    negative_share: float
    mean_pre_match_diff_bp: float
    mean_strict_match_diff_bp: float
    mean_parent_residual_bp: float


def period_mask(frame: pd.DataFrame, period: Period) -> pd.Series:
    return frame["date"].between(
        pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both"
    )


def parse_instrument_file(provider_uri: str, market: str, start: str, end: str) -> pd.DataFrame:
    path = Path(provider_uri) / "instruments" / f"{market}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) < 3:
            continue
        begin = max(pd.Timestamp(parts[1]), lower)
        finish = min(pd.Timestamp(parts[2]), upper)
        if begin <= finish:
            rows.append(
                {
                    "market": market,
                    "code": parts[0].upper(),
                    "start": begin,
                    "end": finish,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no membership intervals for {market}")
    return frame.sort_values(["code", "start", "end"]).reset_index(drop=True)


def load_prices(codes: list[str], start: str, end: str) -> pd.DataFrame:
    fields = ["$open", "$high", "$low", "$close", "$volume", "$change", "$factor"]
    frame = D.features(codes, fields, start_time=start, end_time=end, freq="day")
    if frame.empty:
        raise RuntimeError("Qlib returned no price data")
    frame = frame.reset_index().rename(
        columns={
            "instrument": "code",
            "datetime": "date",
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$change": "change",
            "$factor": "factor",
        }
    )
    frame["code"] = frame["code"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["code", "date"])
    for column in ["open", "high", "low", "close", "volume", "change", "factor"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"])


def span_map(intervals: pd.DataFrame, market: str) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    subset = intervals.loc[intervals["market"].eq(market)]
    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for code, group in subset.groupby("code", sort=False):
        result[str(code)] = [
            (pd.Timestamp(row.start), pd.Timestamp(row.end))
            for row in group.itertuples()
        ]
    return result


def calendar_mask(
    calendar: pd.DatetimeIndex,
    spans: Iterable[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    values = calendar.to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(len(calendar), dtype=bool)
    for start, end in spans:
        left = int(np.searchsorted(values, np.datetime64(start), side="left"))
        right = int(np.searchsorted(values, np.datetime64(end), side="right"))
        if left < right:
            mask[left:right] = True
    return mask


def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.log(numerator / denominator).where(numerator.gt(0) & denominator.gt(0))


def prepare_code(
    code: str,
    raw: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    spans: dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
) -> pd.DataFrame:
    g = raw.set_index("date").reindex(calendar).copy()
    g.index.name = "date"
    g["code"] = code
    for column in ["open", "high", "low", "close", "volume", "change", "factor"]:
        if column not in g:
            g[column] = np.nan
    g["complete"] = g[["open", "high", "low", "close"]].gt(0).all(axis=1) & g["volume"].gt(0)
    log_close = np.log(g["close"].where(g["close"].gt(0)))
    g["ret1"] = log_close.diff()
    g["pre_ret20"] = log_close.shift(1) - log_close.shift(21)
    g["pre_ret60"] = log_close.shift(1) - log_close.shift(61)
    g["pre_vol20"] = g["ret1"].shift(1).rolling(20, min_periods=15).std(ddof=0)
    travel = g["ret1"].abs().shift(1).rolling(20, min_periods=15).sum()
    g["pre_path_eff20"] = g["pre_ret20"].abs() / (travel + 1e-12)
    amount = g["volume"] * g["close"]
    g["pre_log_amount20"] = np.log1p(amount.shift(1).rolling(20, min_periods=15).mean())
    volume_median = g["volume"].shift(1).rolling(60, min_periods=30).median()
    g["volume_shock"] = g["volume"] / volume_median.replace(0.0, np.nan)
    g["overnight"] = safe_log_ratio(g["open"], g["close"].shift(1))
    g["intraday"] = safe_log_ratio(g["close"], g["open"])
    g["log_range"] = safe_log_ratio(g["high"], g["low"]).where(g["high"].ge(g["low"]))
    width = g["high"] - g["low"]
    g["close_location"] = ((g["close"] - g["low"]) / width.replace(0.0, np.nan)).clip(0.0, 1.0)
    g["target_gross"] = safe_log_ratio(g["close"].shift(-5), g["open"].shift(-1))
    g["pre_observations"] = g["complete"].shift(1).fillna(False).cumsum()

    for market in (*SOURCE_MARKETS, TARGET_MARKET):
        g[f"member_{market}"] = calendar_mask(calendar, spans[market].get(code, []))
    g["member_source"] = g[[f"member_{market}" for market in SOURCE_MARKETS]].any(axis=1)
    g["member_target_only"] = g[f"member_{TARGET_MARKET}"] & ~g["member_source"]
    g["member_any"] = g["member_source"] | g[f"member_{TARGET_MARKET}"]
    g["exit_source"] = g["member_source"].shift(1, fill_value=False) & ~g["member_source"]
    g["exit_target"] = (
        g["member_target_only"].shift(1, fill_value=False)
        & ~g[f"member_{TARGET_MARKET}"]
        & ~g["member_source"]
    )
    g["valid_event"] = g["complete"] & g["target_gross"].notna() & g["pre_observations"].ge(120)
    return g.reset_index()


def prepare_panel(raw: pd.DataFrame, calendar: pd.DatetimeIndex, intervals: pd.DataFrame) -> pd.DataFrame:
    spans = {
        market: span_map(intervals, market)
        for market in (*SOURCE_MARKETS, TARGET_MARKET)
    }
    chunks: list[pd.DataFrame] = []
    for sequence, (code, group) in enumerate(raw.groupby("code", sort=False), start=1):
        chunks.append(prepare_code(str(code), group, calendar, spans))
        if sequence % 250 == 0:
            LOGGER.info("prepared securities=%s", sequence)
    panel = pd.concat(chunks, ignore_index=True)
    keep = panel["member_any"] | panel["exit_source"] | panel["exit_target"]
    panel = panel.loc[keep].copy()
    panel["control_eligible"] = panel["member_any"] & panel["complete"] & panel["target_gross"].notna()
    benchmark = panel["target_gross"].where(panel["control_eligible"]).groupby(panel["date"]).median()
    ret_benchmark = panel["ret1"].where(panel["control_eligible"]).groupby(panel["date"]).median()
    pre_benchmark = panel["pre_ret20"].where(panel["control_eligible"]).groupby(panel["date"]).median()
    panel["target"] = panel["target_gross"] - panel["date"].map(benchmark)
    panel["event_rel_ret1"] = panel["ret1"] - panel["date"].map(ret_benchmark)
    panel["pre_rel_ret20"] = panel["pre_ret20"] - panel["date"].map(pre_benchmark)
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def candidate_mask(frame: pd.DataFrame, event_column: str, candidate: str) -> pd.Series:
    event = frame[event_column] & frame["valid_event"]
    if candidate == "all_exit":
        return event
    if candidate == "negative_exit_pressure":
        return event & frame["event_rel_ret1"].lt(0)
    if candidate == "positive_exit_pressure":
        return event & frame["event_rel_ret1"].gt(0)
    if candidate == "negative_pre_exit_runup":
        return event & frame["pre_rel_ret20"].lt(0)
    if candidate == "positive_pre_exit_runup":
        return event & frame["pre_rel_ret20"].gt(0)
    if candidate == "negative_runup_and_pressure":
        return event & frame["pre_rel_ret20"].lt(0) & frame["event_rel_ret1"].lt(0)
    if candidate == "positive_runup_and_pressure":
        return event & frame["pre_rel_ret20"].gt(0) & frame["event_rel_ret1"].gt(0)
    raise KeyError(candidate)


def rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in MODEL_FEATURES:
        out[f"rank__{column}"] = out.groupby("date", observed=True)[column].rank(pct=True, method="average")
    return out


def sample_train(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    counts = frame.groupby("date", observed=True)["code"].transform("size").astype(float)
    weights = (1.0 / counts)
    weights /= weights.sum()
    return frame.sample(n=maximum, weights=weights, random_state=seed, replace=False)


def add_parent_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = rank_features(frame)
    out["parent_prediction"] = np.nan
    features = [f"rank__{column}" for column in MODEL_FEATURES]
    unique_dates = np.asarray(sorted(out["date"].dropna().unique()))
    for sequence, period in enumerate(PERIODS, start=1):
        before = unique_dates[unique_dates < np.datetime64(period.start)]
        if len(before) <= 10:
            continue
        cutoff = pd.Timestamp(before[-6])
        train = out.loc[
            out["control_eligible"]
            & out["date"].lt(cutoff)
            & out["target"].notna(),
            ["date", "target"] + features,
        ].copy()
        if len(train) < 20_000:
            continue
        train = sample_train(train, MAX_RIDGE_TRAIN, RANDOM_SEED + sequence)
        medians = train[features].median()
        x_train = train[features].fillna(medians).to_numpy(dtype=float)
        y_train = train["target"].to_numpy(dtype=float)
        counts = train.groupby("date", observed=True)["date"].transform("size").to_numpy(dtype=float)
        sample_weight = 1.0 / counts
        sample_weight *= len(sample_weight) / sample_weight.sum()
        model = Ridge(alpha=25.0, fit_intercept=True)
        model.fit(x_train, y_train, sample_weight=sample_weight)
        test = period_mask(out, period) & out["target"].notna() & (out["control_eligible"] | out["exit_source"] | out["exit_target"])
        if test.any():
            out.loc[test, "parent_prediction"] = model.predict(
                out.loc[test, features].fillna(medians).to_numpy(dtype=float)
            )
        LOGGER.info("parent period=%s train=%s", period.name, len(train))
    out["parent_residual"] = out["target"] - out["parent_prediction"]
    return out


def robust_scale(pool: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    median = pool[features].median()
    mad = (pool[features] - median).abs().median() * 1.4826
    scale = mad.where(mad.gt(1e-8), pool[features].std(ddof=0)).replace(0.0, np.nan)
    return median, scale


def match_one(event: pd.Series, pool: pd.DataFrame, features: list[str]) -> tuple[float, str]:
    candidates = pool.loc[pool[features + ["target"]].notna().all(axis=1)].copy()
    if len(candidates) < 30 or event[features].isna().any():
        return np.nan, ""
    median, scale = robust_scale(candidates, features)
    valid = scale.notna()
    if int(valid.sum()) < max(3, len(features) // 2):
        return np.nan, ""
    used = list(scale.index[valid])
    x_pool = (candidates[used] - median[used]) / scale[used]
    x_event = (event[used] - median[used]) / scale[used]
    distance = np.sqrt(np.square(x_pool.to_numpy(dtype=float) - x_event.to_numpy(dtype=float)).mean(axis=1))
    order = np.argsort(distance)[: min(MATCH_K, len(distance))]
    selected = candidates.iloc[order]
    return float(event["target"] - selected["target"].mean()), ";".join(selected["code"].astype(str))


def event_records(frame: pd.DataFrame, universe: str, candidate: str) -> pd.DataFrame:
    event_column = "exit_source" if universe == "source" else "exit_target"
    selected = frame.loc[candidate_mask(frame, event_column, candidate)].copy()
    rows: list[dict[str, object]] = []
    for event in selected.itertuples(index=False):
        row = pd.Series(event._asdict())
        pool = frame.loc[
            frame["date"].eq(event.date)
            & frame["control_eligible"]
            & frame["code"].ne(event.code)
        ].copy()
        pre_diff, pre_codes = match_one(row, pool, PRE_MATCH_FEATURES)
        strict_diff, strict_codes = match_one(row, pool, STRICT_MATCH_FEATURES)
        rows.append(
            {
                "universe": universe,
                "candidate": candidate,
                "date": pd.Timestamp(event.date),
                "code": str(event.code),
                "target": float(row["target"]),
                "parent_prediction": float(row["parent_prediction"]) if pd.notna(row["parent_prediction"]) else np.nan,
                "parent_residual": float(row["parent_residual"]) if pd.notna(row["parent_residual"]) else np.nan,
                "pre_match_diff": pre_diff,
                "strict_match_diff": strict_diff,
                "event_rel_ret1": float(row["event_rel_ret1"]),
                "pre_rel_ret20": float(row["pre_rel_ret20"]),
                "pre_control_codes": pre_codes,
                "strict_control_codes": strict_codes,
            }
        )
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame, universe: str, candidate: str, period: Period) -> EventMetric:
    subset = records.loc[period_mask(records, period)] if not records.empty else records
    target = subset["target"].dropna() if not subset.empty else pd.Series(dtype=float)
    return EventMetric(
        universe=universe,
        candidate=candidate,
        period=period.name,
        n_events=int(len(subset)),
        n_dates=int(subset["date"].nunique()) if not subset.empty else 0,
        mean_target_bp=float(target.mean() * 1e4) if len(target) else np.nan,
        median_target_bp=float(target.median() * 1e4) if len(target) else np.nan,
        negative_share=float((target < 0).mean()) if len(target) else np.nan,
        mean_pre_match_diff_bp=float(subset["pre_match_diff"].mean() * 1e4) if not subset.empty else np.nan,
        mean_strict_match_diff_bp=float(subset["strict_match_diff"].mean() * 1e4) if not subset.empty else np.nan,
        mean_parent_residual_bp=float(subset["parent_residual"].mean() * 1e4) if not subset.empty else np.nan,
    )


def daily_series(records: pd.DataFrame, start: str, end: str, column: str) -> pd.Series:
    subset = records.loc[records["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")]
    return subset.groupby("date", observed=True)[column].mean().sort_index()


def block_interval(values: pd.Series, block: int, draws: int = 5000) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < max(8, block // 2):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED + block + len(x))
    starts = np.arange(max(1, len(x) - block + 1))
    means = np.empty(draws, dtype=float)
    n_blocks = int(math.ceil(len(x) / block))
    for draw in range(draws):
        sample = np.concatenate([x[start : start + block] for start in rng.choice(starts, size=n_blocks, replace=True)])[: len(x)]
        means[draw] = sample.mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def queue_means(records: pd.DataFrame, start: str, end: str, column: str) -> list[float]:
    daily = daily_series(records, start, end, column)
    return [float(daily.iloc[offset::6].mean() * 1e4) for offset in range(6) if len(daily.iloc[offset::6])]


def overlay_metrics(records: pd.DataFrame, adjustment: float, period: Period) -> dict[str, float]:
    subset = records.loc[period_mask(records, period)].dropna(subset=["target", "parent_prediction"])
    if subset.empty:
        return {"n_events": 0, "mse_improvement_bp2": np.nan, "adjusted_residual_bp": np.nan}
    baseline = subset["target"] - subset["parent_prediction"]
    adjusted = subset["target"] - (subset["parent_prediction"] + adjustment)
    return {
        "n_events": int(len(subset)),
        "mse_improvement_bp2": float((np.square(baseline) - np.square(adjusted)).mean() * 1e8),
        "adjusted_residual_bp": float(adjusted.mean() * 1e4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)
    interval_frames = [
        parse_instrument_file(args.provider_uri, market, args.start, args.end)
        for market in (*SOURCE_MARKETS, TARGET_MARKET)
    ]
    intervals = pd.concat(interval_frames, ignore_index=True)
    codes = sorted(intervals["code"].unique())
    calendar = pd.DatetimeIndex(D.calendar(start_time=args.start, end_time=args.end, freq="day"))
    raw = load_prices(codes, args.start, args.end)
    LOGGER.info("raw rows=%s codes=%s dates=%s", len(raw), raw["code"].nunique(), raw["date"].nunique())
    panel = add_parent_predictions(prepare_panel(raw, calendar, intervals))
    del raw

    records_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    all_records: list[pd.DataFrame] = []
    for universe in ("source", "target"):
        for candidate in CANDIDATES:
            records = event_records(panel, universe, candidate)
            records_by_key[(universe, candidate)] = records
            all_records.append(records)
            for period in PERIODS:
                metric_rows.append(asdict(summarize(records, universe, candidate, period)))
    metrics = pd.DataFrame(metric_rows)

    development = metrics.loc[
        metrics["universe"].eq("source") & metrics["period"].eq("development")
    ].set_index("candidate")
    selection = metrics.loc[
        metrics["universe"].eq("source") & metrics["period"].eq("selection")
    ].set_index("candidate")
    qualified: list[tuple[str, int, float]] = []
    for candidate in CANDIDATES:
        dev = development.loc[candidate]
        sel = selection.loc[candidate]
        effects = [dev.mean_target_bp, dev.mean_strict_match_diff_bp, sel.mean_target_bp, sel.mean_strict_match_diff_bp]
        if not all(np.isfinite(value) for value in effects):
            continue
        signs = [int(np.sign(value)) for value in effects]
        if (
            dev.n_events >= 12
            and sel.n_events >= 15
            and len(set(signs)) == 1
            and signs[0] != 0
            and np.isfinite(sel.mean_parent_residual_bp)
            and int(np.sign(sel.mean_parent_residual_bp)) == signs[0]
        ):
            worst = min(abs(dev.mean_strict_match_diff_bp), abs(sel.mean_strict_match_diff_bp))
            qualified.append((candidate, signs[0], worst))
    qualified.sort(key=lambda item: (-item[2], item[0]))
    selected = qualified[0][0] if qualified else None
    orientation = qualified[0][1] if qualified else 0

    gates: dict[str, bool] = {"selection_candidate_exists": selected is not None}
    intervals_out: dict[str, object] = {}
    overlays: dict[str, object] = {}
    adjustment = np.nan

    if selected is not None:
        source_records = records_by_key[("source", selected)]
        target_records = records_by_key[("target", selected)]
        selection_period = next(period for period in PERIODS if period.name == "selection")
        selection_records = source_records.loc[period_mask(source_records, selection_period)].dropna(subset=["parent_residual"])
        adjustment = float(selection_records["parent_residual"].mean())

        metric_lookup = metrics.set_index(["universe", "candidate", "period"])
        for universe, records in (("source", source_records), ("target", target_records)):
            for period_name in ("confirmation", "final"):
                period = next(period for period in PERIODS if period.name == period_name)
                for column in ("strict_match_diff", "parent_residual"):
                    daily = daily_series(records, period.start, period.end, column)
                    for block in (20, 40):
                        raw_ci = block_interval(daily, block)
                        oriented = [orientation * value * 1e4 for value in raw_ci]
                        if orientation < 0:
                            oriented = [oriented[0], oriented[2], oriented[1]]
                        intervals_out[f"{universe}_{period_name}_{column}_{block}d_oriented_bp"] = oriented
                overlays[f"{universe}_{period_name}"] = overlay_metrics(records, adjustment, period)

        combined_start = "2022-01-01"
        combined_daily = daily_series(target_records, combined_start, args.end, "strict_match_diff")
        for block in (20, 40):
            raw_ci = block_interval(combined_daily, block)
            oriented = [orientation * value * 1e4 for value in raw_ci]
            if orientation < 0:
                oriented = [oriented[0], oriented[2], oriented[1]]
            intervals_out[f"target_combined_strict_{block}d_oriented_bp"] = oriented
        queues = [orientation * value for value in queue_means(target_records, combined_start, args.end, "strict_match_diff")]
        intervals_out["target_combined_six_queue_oriented_bp"] = queues

        source_confirmation = metric_lookup.loc[("source", selected, "confirmation")]
        source_final = metric_lookup.loc[("source", selected, "final")]
        target_confirmation = metric_lookup.loc[("target", selected, "confirmation")]
        target_final = metric_lookup.loc[("target", selected, "final")]
        ci20 = intervals_out["target_combined_strict_20d_oriented_bp"]
        ci40 = intervals_out["target_combined_strict_40d_oriented_bp"]
        gates.update(
            {
                "source_confirmation_support": bool(source_confirmation.n_events >= 15),
                "source_final_support": bool(source_final.n_events >= 8),
                "target_confirmation_support": bool(target_confirmation.n_events >= 12),
                "target_final_support": bool(target_final.n_events >= 8),
                "source_confirmation_strict_consistent": bool(orientation * source_confirmation.mean_strict_match_diff_bp > 0),
                "source_final_strict_consistent": bool(orientation * source_final.mean_strict_match_diff_bp > 0),
                "target_confirmation_strict_consistent": bool(orientation * target_confirmation.mean_strict_match_diff_bp > 0),
                "target_final_strict_consistent": bool(orientation * target_final.mean_strict_match_diff_bp > 0),
                "source_confirmation_residual_consistent": bool(orientation * source_confirmation.mean_parent_residual_bp > 0),
                "source_final_residual_consistent": bool(orientation * source_final.mean_parent_residual_bp > 0),
                "target_confirmation_residual_consistent": bool(orientation * target_confirmation.mean_parent_residual_bp > 0),
                "target_final_residual_consistent": bool(orientation * target_final.mean_parent_residual_bp > 0),
                "target_20d_interval_positive": bool(np.isfinite(ci20[1]) and ci20[1] > 0),
                "target_40d_interval_positive": bool(np.isfinite(ci40[1]) and ci40[1] > 0),
                "five_of_six_queues_consistent": bool(len(queues) == 6 and sum(value > 0 for value in queues) >= 5),
                "target_final_overlay_mse_positive": bool(overlays["target_final"]["mse_improvement_bp2"] > 0),
            }
        )

    passed = bool(selected is not None and all(gates.values()))
    direction = "long" if orientation > 0 else "relative_short_long_avoidance" if orientation < 0 else None
    decision = {
        "status": "admit_external_independent_source_candidate" if passed else "retain_research_only",
        "selected_candidate": selected,
        "factor_id": FACTOR_IDS.get(selected) if selected else None,
        "orientation": int(orientation),
        "direction": direction,
        "independent_alpha": passed,
        "default_enabled": False,
        "issue_time": "first completed session outside the relevant index, at close",
        "prediction_horizon": "next-session open to fifth subsequent-session close",
        "selection_adjustment_bp": float(adjustment * 1e4) if np.isfinite(adjustment) else None,
        "gates": gates,
        "limitations": [
            "Qlib community membership intervals are historical metadata, not announcement timestamps",
            "signal is deliberately delayed until the first completed outside-index session",
            "no cost, T+1, borrow, queue, capacity, continuous-account or production certification",
            "historical cross-index evidence is not fresh future-market confirmation",
        ],
    }
    metadata = {
        "source": "immutable Qlib CN community release",
        "release_sha256": RELEASE_SHA256,
        "provider_uri": args.provider_uri,
        "start": args.start,
        "end": args.end,
        "calendar_dates": int(len(calendar)),
        "codes": int(len(codes)),
        "source_markets": list(SOURCE_MARKETS),
        "target_market": TARGET_MARKET,
        "candidate_family": list(CANDIDATES),
        "selection_rule": "same oriented target and strict-match effect in source development/selection, same selection parent-residual orientation, maximize worst development/selection strict effect",
        "information_identity": "point-in-time dynamic index membership exit; external to completed-session price-volume features",
    }

    metrics.to_csv(output / "INDEX_EXIT_METRICS.csv", index=False)
    if all_records:
        pd.concat(all_records, ignore_index=True).to_csv(output / "INDEX_EXIT_EVENT_RECORDS.csv", index=False)
    (output / "INDEX_EXIT_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "INDEX_EXIT_INTERVALS.json").write_text(json.dumps(intervals_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "INDEX_EXIT_OVERLAY.json").write_text(json.dumps(overlays, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "INDEX_EXIT_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_metrics = metrics.loc[metrics["candidate"].eq(selected)] if selected else metrics.iloc[0:0]
    lines = [
        "# A-share index-exit independent-source experiment",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Candidate: `{selected}`",
        f"- Factor ID: `{decision['factor_id']}`",
        f"- Direction: `{direction}`",
        f"- Independent alpha: `{passed}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gates.items())
    lines.extend([
        "",
        "## Selected metrics",
        "",
        selected_metrics.to_markdown(index=False) if not selected_metrics.empty else "No candidate qualified in development/selection.",
        "",
        "## Intervals",
        "",
        "```json",
        json.dumps(intervals_out, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evidence boundary",
        "",
        "Passing would admit only a default-off external-source historical candidate. It would not certify execution, costs, T+1, borrow, capacity, an account, production use, or future-market confirmation.",
        "",
    ])
    report = "\n".join(lines)
    (output / "INDEX_EXIT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print("INDEX_EXIT_DECISION=" + json.dumps(decision, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
