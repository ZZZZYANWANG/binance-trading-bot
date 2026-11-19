from pathlib import Path


PATH = Path(__file__).with_name("run_research.py")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    old = (
        'market_pos = np.nanmean(pos_matrix, axis=1); '
        'market_neg = np.nanmean(neg_matrix, axis=1); '
        'issue_indices = np.flatnonzero(timeline.hour == 0)'
    )
    new = (
        'market_pos = (np.nanmean(pos_matrix, axis=1) >= 0.20).astype(float); '
        'market_neg = (np.nanmean(neg_matrix, axis=1) >= 0.20).astype(float); '
        'issue_indices = np.flatnonzero(timeline.hour == 0)'
    )
    if text.count(old) != 1:
        raise RuntimeError(f"market-event contract: expected one occurrence, found {text.count(old)}")
    PATH.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main()
