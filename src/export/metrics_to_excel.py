"""Export per-epoch training histories from outputs/Training*/history_*.json into an Excel workbook.

Usage:
    python -m src.export.metrics_to_excel              # writes outputs/training_metrics.xlsx
    python -m src.export.metrics_to_excel out.xlsx     # explicit output path
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path("outputs")
DEFAULT_OUT = OUTPUTS_DIR / "training_metrics.xlsx"

STAGE_LABEL = {
    "history_train_head": "Stage1_Head",
    "history_head": "Stage1_Head_v2",
    "history_head_full": "Head_Full",
    "history_finetune": "Stage2_FineTune",
    "history_fine-tuning": "Stage2_FineTune",
}


def _stage_from_filename(name: str) -> str:
    stem = Path(name).stem
    return STAGE_LABEL.get(stem, stem)


def _load_history(path: Path) -> pd.DataFrame | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or "epoch" not in data:
        return None
    n = len(data["epoch"])
    cols = {k: v for k, v in data.items() if isinstance(v, list) and len(v) == n}
    df = pd.DataFrame(cols)
    df.insert(0, "epoch", df.pop("epoch"))
    return df


def _merge_resources(history_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """If a sibling resources_<stage>.json exists, left-join its per-epoch columns."""
    stem = history_path.stem  # e.g. history_head
    if not stem.startswith("history_"):
        return df
    res_path = history_path.parent / f"resources_{stem[len('history_'):]}.json"
    if not res_path.exists():
        return df
    try:
        data = json.loads(res_path.read_text())
    except Exception:
        return df
    if not isinstance(data, dict) or "epoch" not in data:
        return df
    n = len(data["epoch"])
    cols = {k: v for k, v in data.items() if isinstance(v, list) and len(v) == n}
    res_df = pd.DataFrame(cols)
    return df.merge(res_df, on="epoch", how="left")


def collect() -> list[tuple[str, str, Path, pd.DataFrame]]:
    """Return list of (run_name, stage_label, source_path, df) for every history file."""
    rows: list[tuple[str, str, Path, pd.DataFrame]] = []
    for path in sorted(OUTPUTS_DIR.glob("Training*/history_*.json")):
        df = _load_history(path)
        if df is None or df.empty:
            continue
        df = _merge_resources(path, df)
        rows.append((path.parent.name, _stage_from_filename(path.name), path, df))
    return rows


def summary_row(run: str, stage: str, df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    best = df.loc[df["val_accuracy"].idxmax()] if "val_accuracy" in df else last
    return {
        "Run": run,
        "Stage": stage,
        "Epochs": int(len(df)),
        "Best Val Acc": float(best.get("val_accuracy", float("nan"))),
        "Best Epoch": int(best.get("epoch", 0)),
        "Final Val Acc": float(last.get("val_accuracy", float("nan"))),
        "Final Val Loss": float(last.get("val_loss", float("nan"))),
        "Final Train Acc": float(last.get("accuracy", float("nan"))),
        "Final Train Loss": float(last.get("loss", float("nan"))),
        "Train-Val Gap": float(last.get("accuracy", 0)) - float(last.get("val_accuracy", 0)),
    }


def _agg(df: pd.DataFrame, col: str, op: str):
    if col not in df:
        return float("nan")
    s = df[col].dropna()
    if s.empty:
        return float("nan")
    return float(getattr(s, op)())


def resource_row(run: str, stage: str, df: pd.DataFrame) -> dict | None:
    """Return a resources summary row, or None if no resource columns present."""
    if "cpu_percent_mean" not in df.columns:
        return None
    return {
        "Run": run,
        "Stage": stage,
        "Epochs": int(len(df)),
        "Total Seconds": _agg(df, "epoch_seconds", "sum"),
        "Mean Epoch Sec": _agg(df, "epoch_seconds", "mean"),
        "CPU% Mean": _agg(df, "cpu_percent_mean", "mean"),
        "CPU% Peak": _agg(df, "cpu_percent_max", "max"),
        "RAM% Mean": _agg(df, "ram_percent_mean", "mean"),
        "RAM% Peak": _agg(df, "ram_percent_peak", "max"),
        "RAM GB Mean": _agg(df, "ram_used_gb_mean", "mean"),
        "RAM GB Peak": _agg(df, "ram_used_gb_peak", "max"),
        "Temp C Mean": _agg(df, "temp_c_mean", "mean"),
        "Temp C Peak": _agg(df, "temp_c_max", "max"),
    }


def write_workbook(out_path: Path) -> dict:
    runs = collect()
    if not runs:
        raise SystemExit("No history_*.json files found under outputs/Training*/")

    summary = pd.DataFrame([summary_row(r, s, df) for r, s, _, df in runs])
    summary = summary.sort_values(["Run", "Stage"]).reset_index(drop=True)

    res_rows = [r for r in (resource_row(rn, s, df) for rn, s, _, df in runs) if r is not None]
    resources = pd.DataFrame(res_rows).sort_values(["Run", "Stage"]).reset_index(drop=True) if res_rows else pd.DataFrame()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        if not resources.empty:
            resources.to_excel(xl, sheet_name="Resources", index=False)
        used: set[str] = {"Summary", "Resources"}
        for run, stage, _, df in runs:
            sheet = f"{run}_{stage}"[:31]
            n = 2
            while sheet in used:
                sheet = f"{run}_{stage}"[: 31 - len(str(n)) - 1] + f"_{n}"
                n += 1
            used.add(sheet)
            df.to_excel(xl, sheet_name=sheet, index=False)

    extra_sheets = 1 + (1 if not resources.empty else 0)  # Summary + maybe Resources
    return {"path": str(out_path), "runs": len(runs), "sheets": extra_sheets + len(runs)}


def main(argv: list[str]) -> None:
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    info = write_workbook(out)
    print(f"Wrote {info['path']}  ({info['runs']} histories, {info['sheets']} sheets)")


if __name__ == "__main__":
    main(sys.argv)
