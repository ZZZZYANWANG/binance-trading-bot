#!/usr/bin/env python3
"""Frozen public-data experiment for an A-share index-entry event Alpha.

This temporary computation script uses the public, event-level CSI300 inclusion
panel from Leonard-Don/index-inclusion-research.  It selects a small,
predeclared family only on 2020-2023, freezes one candidate, and evaluates it on
2024 and 2025-2026.  The target is the source panel's market-adjusted CAR from
effective date +2 through +20 trading sessions.

The script never writes a factor into Trade.  It only emits an inspectable
decision package.  A Trade integration is permitted only when every gate in
DECISION.json passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

SOURCE_COMMIT = "d5bef04b4af42ad04ca3ee64890e89e4656b3889"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Leonard-Don/index-inclusion-research/"
    f"{SOURCE_COMMIT}/results/real_tables/cma_gap_event_level.csv"
)
RANDOM_SEED = 22391
TARGET = "post_effective_reversal"
NUMERIC_FEATURES = [
    "pre_announce_runup",
    "announce_jump",
    "gap_drift",
    "effective_jump",
    "gap_length_days",
]


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str


PERIODS = (
    Period("development", "2020-01-01", "2021-12-31"),
    Period("selection", "2022-01-01", "2023-12-31"),
    Period("confirmation", "2024-01-01", "2024-12-31"),
    Period("final", "2025-01-01", "2026-10-31"),
)

CANDIDATES = (
    "all_entry",
    "positive_pre_runup",
    "positive_announce_jump",
    "positive_gap_drift",
    "positive_effective_jump",
    "positive_pre_effect_total",
    "pre_runup_and_gap_drift",
    "announce_and_gap_positive",
)

FACTOR_IDS = {
    "all_entry": "CSI300_ENTRY_POST_EFFECT_REVERSAL20",
    "positive_pre_runup": "CSI300_ENTRY_RUNUP_EXHAUSTION_REVERSAL20",
    "positive_announce_jump": "CSI300_ENTRY_ANNOUNCEMENT_PRESSURE_REVERSAL20",
    "positive_gap_drift": "CSI300_ENTRY_GAP_DRIFT_REVERSAL20",
    "positive_effective_jump": "CSI300_ENTRY_EFFECTIVE_JUMP_REVERSAL20",
    "positive_pre_effect_total": "CSI300_ENTRY_PRE_EFFECT_PRESSURE_REVERSAL20",
    "pre_runup_and_gap_drift": "CSI300_ENTRY_RUNUP_GAP_EXHAUSTION20",
    "announce_and_gap_positive": "CSI300_ENTRY_ANNOUNCE_GAP_EXHAUSTION20",
}


@dataclass(frozen=True)
class Metric:
    candidate: str
    period: str
    n_events: int
    n_batches: int
    event_mean: float
    event_median: float
    negative_share: float
    batch_equal_mean: float
    batch_t: float
    batch_p_one_sided: float
    parent_residual_mean: float
    candidate_minus_complement: float


def _period_mask(frame: pd.DataFrame, period: Period) -> pd.Series:
    return frame["effective_date"].between(
        pd.Timestamp(period.start), pd.Timestamp(period.end), inclusive="both"
    )


def _candidate_mask(frame: pd.DataFrame, candidate: str) -> pd.Series:
    if candidate == "all_entry":
        return pd.Series(True, index=frame.index)
    if candidate == "positive_pre_runup":
        return frame["pre_announce_runup"].gt(0.0)
    if candidate == "positive_announce_jump":
        return frame["announce_jump"].gt(0.0)
    if candidate == "positive_gap_drift":
        return frame["gap_drift"].gt(0.0)
    if candidate == "positive_effective_jump":
        return frame["effective_jump"].gt(0.0)
    if candidate == "positive_pre_effect_total":
        return frame["pre_effect_total"].gt(0.0)
    if candidate == "pre_runup_and_gap_drift":
        return frame["pre_announce_runup"].gt(0.0) & frame["gap_drift"].gt(0.0)
    if candidate == "announce_and_gap_positive":
        return frame["announce_jump"].gt(0.0) & frame["gap_drift"].gt(0.0)
    raise KeyError(candidate)


def _download(output_dir: Path) -> tuple[Path, str]:
    path = output_dir / "source_cma_gap_event_level.csv"
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Trade-independent-alpha-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ticker": str, "event_id": str})
    required = {
        "event_id",
        "market",
        "ticker",
        "announce_date",
        "effective_date",
        "sector",
        "batch_id",
        *NUMERIC_FEATURES,
        TARGET,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"source data missing columns: {missing}")
    frame = frame.loc[
        frame["market"].eq("CN")
        & frame["event_id"].astype(str).str.contains("-add-", regex=False)
    ].copy()
    frame["announce_date"] = pd.to_datetime(frame["announce_date"], errors="coerce")
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce")
    for column in [*NUMERIC_FEATURES, TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["event_id", "ticker", "batch_id", "effective_date", *NUMERIC_FEATURES, TARGET]
    ).copy()
    frame = frame.loc[
        frame["effective_date"].between(
            pd.Timestamp(PERIODS[0].start), pd.Timestamp(PERIODS[-1].end), inclusive="both"
        )
    ].copy()
    frame["pre_effect_total"] = frame[
        ["pre_announce_runup", "announce_jump", "gap_drift", "effective_jump"]
    ].sum(axis=1)
    frame["batch_date"] = frame.groupby("batch_id", observed=True)["effective_date"].transform("min")
    return frame.sort_values(["effective_date", "batch_id", "ticker"]).reset_index(drop=True)


def _expanding_parent(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit a price-path parent strictly on earlier inclusion batches."""
    out = frame.copy()
    out["parent_prediction"] = np.nan
    batches = (
        out[["batch_id", "batch_date"]]
        .drop_duplicates()
        .sort_values(["batch_date", "batch_id"])
        .reset_index(drop=True)
    )
    for row in batches.itertuples(index=False):
        train = out.loc[out["batch_date"].lt(row.batch_date)].copy()
        test = out["batch_id"].eq(row.batch_id)
        if len(train) < 35 or int(train["batch_id"].nunique()) < 3 or not test.any():
            continue
        medians = train[NUMERIC_FEATURES].median()
        x_train = train[NUMERIC_FEATURES].fillna(medians).to_numpy(dtype=float)
        x_test = out.loc[test, NUMERIC_FEATURES].fillna(medians).to_numpy(dtype=float)
        center = np.mean(x_train, axis=0)
        scale = np.std(x_train, axis=0)
        scale = np.where(scale > 1e-10, scale, 1.0)
        x_train = (x_train - center) / scale
        x_test = (x_test - center) / scale
        y_train = train[TARGET].to_numpy(dtype=float)
        batch_size = train.groupby("batch_id", observed=True)["batch_id"].transform("size")
        sample_weight = 1.0 / batch_size.to_numpy(dtype=float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        model = Ridge(alpha=10.0, fit_intercept=True)
        model.fit(x_train, y_train, sample_weight=sample_weight)
        out.loc[test, "parent_prediction"] = model.predict(x_test)
    out["parent_residual"] = out[TARGET] - out["parent_prediction"]
    return out


def _batch_means(frame: pd.DataFrame, value: str) -> pd.Series:
    return frame.groupby("batch_id", observed=True)[value].mean().sort_index()


def _candidate_complement(frame: pd.DataFrame, candidate: str) -> pd.Series:
    selected = _candidate_mask(frame, candidate)
    rows: list[float] = []
    for _, group in frame.groupby("batch_id", observed=True):
        flag = selected.loc[group.index]
        if flag.any() and (~flag).any():
            rows.append(float(group.loc[flag, TARGET].mean() - group.loc[~flag, TARGET].mean()))
    return pd.Series(rows, dtype=float)


def _metric(frame: pd.DataFrame, candidate: str, period: Period) -> Metric:
    period_frame = frame.loc[_period_mask(frame, period)].copy()
    selected = period_frame.loc[_candidate_mask(period_frame, candidate)].copy()
    values = selected[TARGET].dropna()
    batches = _batch_means(selected, TARGET)
    if len(batches) >= 2 and float(batches.std(ddof=1)) > 0:
        t_value = float(batches.mean() / (batches.std(ddof=1) / math.sqrt(len(batches))))
        p_value = float(stats.t.cdf(t_value, df=len(batches) - 1))
    else:
        t_value = np.nan
        p_value = np.nan
    residual = selected["parent_residual"].dropna()
    complement = _candidate_complement(period_frame, candidate)
    return Metric(
        candidate=candidate,
        period=period.name,
        n_events=int(len(selected)),
        n_batches=int(selected["batch_id"].nunique()),
        event_mean=float(values.mean()) if len(values) else np.nan,
        event_median=float(values.median()) if len(values) else np.nan,
        negative_share=float((values < 0).mean()) if len(values) else np.nan,
        batch_equal_mean=float(batches.mean()) if len(batches) else np.nan,
        batch_t=t_value,
        batch_p_one_sided=p_value,
        parent_residual_mean=float(residual.mean()) if len(residual) else np.nan,
        candidate_minus_complement=float(complement.mean()) if len(complement) else np.nan,
    )


def _moving_batch_interval(
    batch_values: pd.Series,
    block: int,
    draws: int = 10_000,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    x = pd.to_numeric(batch_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < max(3, block):
        return np.nan, np.nan, np.nan
    starts = np.arange(len(x) - block + 1)
    rng = np.random.default_rng(seed + block + len(x))
    samples = np.empty(draws, dtype=float)
    n_blocks = int(math.ceil(len(x) / block))
    for draw in range(draws):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([x[start : start + block] for start in chosen])[: len(x)]
        samples[draw] = float(sample.mean())
    return float(x.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _combined_batch_series(frame: pd.DataFrame, candidate: str, column: str) -> pd.Series:
    subset = frame.loc[
        frame["effective_date"].between(
            pd.Timestamp("2024-01-01"), pd.Timestamp(PERIODS[-1].end), inclusive="both"
        )
    ].copy()
    subset = subset.loc[_candidate_mask(subset, candidate)]
    return _batch_means(subset, column)


def _combined_complement_series(frame: pd.DataFrame, candidate: str) -> pd.Series:
    subset = frame.loc[
        frame["effective_date"].between(
            pd.Timestamp("2024-01-01"), pd.Timestamp(PERIODS[-1].end), inclusive="both"
        )
    ].copy()
    selected = _candidate_mask(subset, candidate)
    values: dict[str, float] = {}
    for batch, group in subset.groupby("batch_id", observed=True):
        flag = selected.loc[group.index]
        if flag.any() and (~flag).any():
            values[str(batch)] = float(
                group.loc[flag, TARGET].mean() - group.loc[~flag, TARGET].mean()
            )
    return pd.Series(values, dtype=float).sort_index()


def _overlay_metrics(frame: pd.DataFrame, candidate: str, adjustment: float) -> dict[str, float]:
    final = frame.loc[_period_mask(frame, PERIODS[-1])].copy()
    final = final.loc[_candidate_mask(final, candidate)].dropna(
        subset=[TARGET, "parent_prediction"]
    )
    if final.empty:
        return {
            "n_events": 0,
            "baseline_mse": np.nan,
            "overlay_mse": np.nan,
            "mse_improvement": np.nan,
            "baseline_bias": np.nan,
            "overlay_bias": np.nan,
        }
    baseline_error = final[TARGET] - final["parent_prediction"]
    overlay_error = final[TARGET] - (final["parent_prediction"] + adjustment)
    return {
        "n_events": int(len(final)),
        "baseline_mse": float(np.mean(np.square(baseline_error))),
        "overlay_mse": float(np.mean(np.square(overlay_error))),
        "mse_improvement": float(
            np.mean(np.square(baseline_error) - np.square(overlay_error))
        ),
        "baseline_bias": float(baseline_error.mean()),
        "overlay_bias": float(overlay_error.mean()),
    }


def main() -> int:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "ashare_index_entry_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path, source_sha256 = _download(output_dir)
    frame = _expanding_parent(_load(source_path))

    metrics = pd.DataFrame(
        [
            asdict(_metric(frame, candidate, period))
            for candidate in CANDIDATES
            for period in PERIODS
        ]
    )
    metrics.to_csv(output_dir / "CANDIDATE_METRICS.csv", index=False)

    development = metrics.loc[metrics["period"].eq("development")].set_index("candidate")
    selection = metrics.loc[metrics["period"].eq("selection")].set_index("candidate")
    qualified: list[str] = []
    for candidate in CANDIDATES:
        dev = development.loc[candidate]
        sel = selection.loc[candidate]
        if (
            dev.n_events >= 12
            and dev.n_batches >= 2
            and sel.n_events >= 12
            and sel.n_batches >= 2
            and dev.event_mean < 0
            and dev.batch_equal_mean < 0
            and sel.event_mean < 0
            and sel.batch_equal_mean < 0
            and np.isfinite(sel.parent_residual_mean)
            and sel.parent_residual_mean < 0
        ):
            qualified.append(candidate)

    selected = None
    if qualified:
        selected = min(
            qualified,
            key=lambda candidate: (
                selection.loc[candidate, "parent_residual_mean"],
                selection.loc[candidate, "batch_equal_mean"],
            ),
        )

    gates: dict[str, bool] = {"selection_candidate_exists": selected is not None}
    intervals: dict[str, list[float]] = {}
    overlay: dict[str, float] = {}
    adjustment = np.nan

    if selected is not None:
        confirmation = metrics.loc[
            metrics["candidate"].eq(selected) & metrics["period"].eq("confirmation")
        ].iloc[0]
        final = metrics.loc[
            metrics["candidate"].eq(selected) & metrics["period"].eq("final")
        ].iloc[0]
        target_batches = _combined_batch_series(frame, selected, TARGET)
        residual_batches = _combined_batch_series(frame, selected, "parent_residual")
        complement_batches = _combined_complement_series(frame, selected)
        for label, series in (
            ("target", target_batches),
            ("parent_residual", residual_batches),
            ("candidate_minus_complement", complement_batches),
        ):
            for block in (1, 2):
                intervals[f"{label}_block{block}_batches"] = list(
                    _moving_batch_interval(series, block)
                )

        selection_rows = frame.loc[_period_mask(frame, PERIODS[1])].copy()
        selection_rows = selection_rows.loc[_candidate_mask(selection_rows, selected)]
        adjustment = float(selection_rows["parent_residual"].dropna().mean())
        overlay = _overlay_metrics(frame, selected, adjustment)

        year_means = (
            frame.loc[
                frame["effective_date"].between(
                    pd.Timestamp("2024-01-01"), pd.Timestamp(PERIODS[-1].end), inclusive="both"
                )
                & _candidate_mask(frame, selected)
            ]
            .assign(year=lambda x: x["effective_date"].dt.year)
            .groupby("year", observed=True)[TARGET]
            .mean()
        )
        intervals["calendar_year_means"] = [float(value) for value in year_means]

        gates.update(
            {
                "confirmation_support": bool(confirmation.n_events >= 8 and confirmation.n_batches >= 1),
                "final_support": bool(final.n_events >= 8 and final.n_batches >= 1),
                "confirmation_target_negative": bool(confirmation.event_mean < 0 and confirmation.batch_equal_mean < 0),
                "final_target_negative": bool(final.event_mean < 0 and final.batch_equal_mean < 0),
                "confirmation_parent_residual_negative": bool(np.isfinite(confirmation.parent_residual_mean) and confirmation.parent_residual_mean < 0),
                "final_parent_residual_negative": bool(np.isfinite(final.parent_residual_mean) and final.parent_residual_mean < 0),
                "confirmation_complement_difference_negative": bool(np.isfinite(confirmation.candidate_minus_complement) and confirmation.candidate_minus_complement < 0),
                "final_complement_difference_negative": bool(np.isfinite(final.candidate_minus_complement) and final.candidate_minus_complement < 0),
                "target_block1_upper_negative": bool(intervals["target_block1_batches"][2] < 0),
                "target_block2_upper_negative": bool(intervals["target_block2_batches"][2] < 0),
                "residual_block1_upper_negative": bool(intervals["parent_residual_block1_batches"][2] < 0),
                "residual_block2_upper_negative": bool(intervals["parent_residual_block2_batches"][2] < 0),
                "complement_block1_upper_negative": bool(intervals["candidate_minus_complement_block1_batches"][2] < 0),
                "complement_block2_upper_negative": bool(intervals["candidate_minus_complement_block2_batches"][2] < 0),
                "all_post_selection_years_negative": bool(len(year_means) >= 2 and (year_means < 0).all()),
                "final_overlay_mse_positive": bool(overlay.get("mse_improvement", np.nan) > 0),
            }
        )

    passed = bool(selected is not None and all(gates.values()))
    decision = {
        "status": "admit_external_independent_source_candidate" if passed else "retain_research_only",
        "selected_candidate": selected,
        "factor_id": FACTOR_IDS.get(selected) if selected else None,
        "independent_alpha": passed,
        "default_enabled": False,
        "direction": "relative_short_long_avoidance" if selected else None,
        "issue_time": "completed CSI300 effective-session close",
        "prediction_horizon": "effective session +2 through +20 market-adjusted CAR",
        "selection_adjustment": float(adjustment) if np.isfinite(adjustment) else None,
        "gates": gates,
        "limitations": [
            "public historical event study, not fresh future market-time confirmation",
            "CSI300 inclusion additions only",
            "event target is market-adjusted CAR from +2 to +20, not an execution simulation",
            "no A-share T+1, transaction-cost, borrow, queue, capacity or account certification",
            "sector and latent selection-rule confounding remain possible",
        ],
    }

    metadata = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": source_sha256,
        "source_blob_sha": "9734e06be24050235806d08c36dcd5db7a248c93",
        "rows_after_filter": int(len(frame)),
        "batches_after_filter": int(frame["batch_id"].nunique()),
        "date_range": [str(frame["effective_date"].min().date()), str(frame["effective_date"].max().date())],
        "target": TARGET,
        "candidate_count": len(CANDIDATES),
        "candidate_family": list(CANDIDATES),
        "selection_rule": "negative development and selection event/batch means plus negative selection expanding-parent residual; choose most negative selection residual",
        "parent": "expanding Ridge on prior inclusion batches using pre-announcement, announcement, gap, effective-jump and gap-length coordinates",
        "bootstrap": "moving block bootstrap over chronological inclusion batches, 10000 draws",
    }

    (output_dir / "INTERVALS.json").write_text(
        json.dumps(intervals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "OVERLAY.json").write_text(
        json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_records = frame.iloc[0:0].copy()
    if selected is not None:
        selected_records = frame.loc[_candidate_mask(frame, selected)].copy()
    selected_records.to_csv(output_dir / "SELECTED_EVENT_RECORDS.csv", index=False)

    selected_metrics = metrics.loc[metrics["candidate"].eq(selected)] if selected else metrics.iloc[0:0]
    lines = [
        "# CSI300 index-entry independent-source Alpha experiment",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Selected candidate: `{selected}`",
        f"- Factor ID: `{decision['factor_id']}`",
        f"- Independent alpha: `{passed}`",
        f"- Source SHA-256: `{source_sha256}`",
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in gates.items())
    lines.extend(
        [
            "",
            "## Selected candidate metrics",
            "",
            selected_metrics.to_markdown(index=False) if not selected_metrics.empty else "No candidate qualified in development/selection.",
            "",
            "## Intervals",
            "",
            "```json",
            json.dumps(intervals, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Parent overlay",
            "",
            "```json",
            json.dumps(overlay, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Evidence boundary",
            "",
            "The index-membership event is an external information source. Passing these gates would admit only a default-off historical candidate. It would not certify execution, costs, borrow, capacity, a continuous account, production use, or fresh future-market confirmation.",
            "",
        ]
    )
    report = "\n".join(lines)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print("DECISION=" + json.dumps(decision, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
