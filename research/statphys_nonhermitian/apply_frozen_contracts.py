from pathlib import Path


PATH = Path(__file__).with_name("run_research.py")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    old = (
        "    pairwise = np.abs(values[:, None] - values[None, :])\n"
        "    pairwise += np.eye(n) * np.inf"
    )
    new = (
        "    pairwise = np.abs(values[:, None] - values[None, :])\n"
        "    np.fill_diagonal(pairwise, np.inf)"
    )
    if text.count(old) != 1:
        raise RuntimeError(f"exceptional-gap contract: expected one occurrence, found {text.count(old)}")
    PATH.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main()
