from pathlib import Path


PATH = Path(__file__).with_name("run_research.py")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    old = "        current = event[-1]"
    new = (
        "        current = ((np.abs(ret_z[t]) > 1.0) & (act_z[t] > 0)).astype(float)\n"
        "        current[~np.isfinite(ret_z[t]) | ~np.isfinite(act_z[t])] = np.nan"
    )
    if text.count(old) != 1:
        raise RuntimeError(f"current-state clock contract: expected one occurrence, found {text.count(old)}")
    PATH.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main()
