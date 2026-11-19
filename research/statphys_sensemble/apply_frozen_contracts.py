from __future__ import annotations

from pathlib import Path


PATH = Path(__file__).with_name("run_research.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    old_availability = (
        'tr = train & panel[cols].notna().all(axis=1).to_numpy(); '
        'te = test & panel[cols].notna().all(axis=1).to_numpy()'
    )
    new_availability = (
        'tr = train & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]]'
        '.notna().all(axis=1).to_numpy(); '
        'te = test & panel[[candidate, f"yraw{horizon}", f"ystd{horizon}"]]'
        '.notna().all(axis=1).to_numpy()'
    )
    if text.count(old_availability) != 3:
        raise RuntimeError(
            f"availability: expected three occurrences, found {text.count(old_availability)}"
        )
    text = text.replace(old_availability, new_availability)

    text = replace_once(
        text,
        '    log_close = np.log(raw.close.where(raw.close > 0))\n    ret = log_close.diff()',
        '    log_close = np.log(raw.close.where(raw.close > 0))\n'
        '    log_open = np.log(raw.open.where(raw.open > 0))\n'
        '    ret = log_close.diff()',
        "log-open insertion",
    )
    text = replace_once(
        text,
        '    out["next_ret1"] = ret.shift(-1)',
        '    out["next_ret1"] = log_close.shift(-1) - log_open.shift(-1)',
        "next-hour payment",
    )
    text = replace_once(
        text,
        '        target = log_close.shift(-h) - log_close',
        '        target = log_close.shift(-h) - log_open.shift(-1)',
        "forward target",
    )
    text = replace_once(
        text,
        '    panel["asset"] = panel.symbol.map(amap).astype(np.int16)\n'
        '    candidates = [x for x in panel if x.startswith("candidate__")]',
        '    panel["asset"] = panel.symbol.map(amap).astype(np.int16)\n'
        '    panel_full = panel.copy()\n'
        '    panel = panel.loc[panel["timestamp"].dt.hour.eq(0)].copy().reset_index(drop=True)\n'
        '    candidates = [x for x in panel if x.startswith("candidate__")]',
        "daily issuance",
    )

    replacements = {
        'group.head(4).to_dict("records")': 'group.head(2).to_dict("records")',
        'for alpha in [10., 100., 1000., 10000., 100000.]':
            'for alpha in [10., 1000., 100000.]',
        'if tr.sum() < 8000 or te.sum() < 2000:':
            'if tr.sum() < 2500 or te.sum() < 1200:',
        'if tr.sum() < 8000 or te.sum() < 1500:':
            'if tr.sum() < 2500 or te.sum() < 1000:',
        'if tr.sum() >= 8000 and te.sum() >= 1000:':
            'if tr.sum() >= 2500 and te.sum() >= 700:',
        'frame = account(panel.loc[mask_period(panel, POST[period])], candidate, horizon, float(claim.direction))':
            'frame = account(panel_full.loc[mask_period(panel_full, POST[period])], candidate, horizon, float(claim.direction))',
    }
    for old, new in replacements.items():
        text = replace_once(text, old, new, old)

    account_start = text.index("def account(panel, signal, horizon, direction):")
    account_end = text.index("\n\ndef main():", account_start)
    account_function = '''def account(panel, signal, horizon, direction):
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
'''
    text = text[:account_start] + account_function + text[account_end:]
    PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
