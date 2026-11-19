from pathlib import Path


PATH = Path(__file__).with_name("run_research.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '    "critical_intensity_acceleration": "intensity_dynamics",\n]\nBASIC_PARENTS = [',
        '    "critical_intensity_acceleration": "intensity_dynamics",\n}\nBASIC_PARENTS = [',
        "formula-dictionary syntax",
    )

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
    text = replace_once(text, old, new, "market-event contract")
    PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
