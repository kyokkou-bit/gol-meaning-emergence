from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\AI-workspace\test")
DENSITY = 0.70
WARMUPS = tuple(range(8, 16))
SIZE = 40
N_TRIALS = 500
MAX_STEPS = 1000
EPSILON = 0.005
OSC_MIN_PERIOD = 2
OSC_MAX_PERIOD = 10
SEED = 42
G_MAX = 0.005


def gol_step(grid: np.ndarray) -> np.ndarray:
    neighbors = (
        np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
        + np.roll(grid, 1, axis=0)
        + np.roll(np.roll(grid, 1, axis=0), -1, axis=1)
        + np.roll(grid, 1, axis=1)
        + np.roll(grid, -1, axis=1)
        + np.roll(np.roll(grid, -1, axis=0), 1, axis=1)
        + np.roll(grid, -1, axis=0)
        + np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    )
    birth = (neighbors == 3) & (grid == 0)
    survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    return (birth | survive).astype(np.uint8)


def local_mean(grid: np.ndarray) -> np.ndarray:
    total = (
        np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
        + np.roll(grid, 1, axis=0)
        + np.roll(np.roll(grid, 1, axis=0), -1, axis=1)
        + np.roll(grid, 1, axis=1)
        + grid
        + np.roll(grid, -1, axis=1)
        + np.roll(np.roll(grid, -1, axis=0), 1, axis=1)
        + np.roll(grid, -1, axis=0)
        + np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    )
    return total / 9.0


def weak_average_coupling(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    coupling: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    prob_a = np.clip((1.0 - coupling) * grid_a.astype(float) + coupling * local_mean(grid_b), 0.0, 1.0)
    prob_b = np.clip((1.0 - coupling) * grid_b.astype(float) + coupling * local_mean(grid_a), 0.0, 1.0)
    out_a = (rng.random(grid_a.shape) < prob_a).astype(np.uint8)
    out_b = (rng.random(grid_b.shape) < prob_b).astype(np.uint8)
    return out_a, out_b


def classify_label(
    current_grid: np.ndarray,
    new_grid: np.ndarray,
    prev_grid: np.ndarray,
    history: deque[bytes],
) -> tuple[str | None, bool]:
    if np.array_equal(new_grid, current_grid):
        return "fixed", True
    new_state = new_grid.tobytes()
    history.append(current_grid.tobytes())
    max_period = min(OSC_MAX_PERIOD, len(history))
    for period in range(OSC_MIN_PERIOD, max_period + 1):
        if new_state == history[-period]:
            return "osc", True
    delta = np.count_nonzero(new_grid != prev_grid) / (SIZE * SIZE)
    if delta < EPSILON:
        return "delta", True
    return None, False


def strategy_name(warmup: int) -> str:
    return f"warmup_{warmup}"


def choose_g(warmup: int, step: int) -> float:
    return 0.0 if step <= warmup else G_MAX


def simulate_condition(warmup: int, rng: np.random.Generator) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []
    strategy = strategy_name(warmup)

    for trial in range(1, N_TRIALS + 1):
        grids = {
            "A": (rng.random((SIZE, SIZE)) < DENSITY).astype(np.uint8),
            "B": (rng.random((SIZE, SIZE)) < DENSITY).astype(np.uint8),
        }
        prev_grids = {name: grid.copy() for name, grid in grids.items()}
        recent_states = {name: deque(maxlen=OSC_MAX_PERIOD) for name in grids}
        labels = {"A": "max", "B": "max"}
        stop_steps = {"A": MAX_STEPS, "B": MAX_STEPS}
        active = {"A": True, "B": True}

        for step in range(1, MAX_STEPS + 1):
            current_g = choose_g(warmup, step)
            new_a = gol_step(grids["A"]) if active["A"] else grids["A"].copy()
            new_b = gol_step(grids["B"]) if active["B"] else grids["B"].copy()
            if current_g > 0.0:
                coupled_a, coupled_b = weak_average_coupling(new_a, new_b, current_g, rng)
            else:
                coupled_a, coupled_b = new_a, new_b
            new_grids = {"A": coupled_a, "B": coupled_b}

            for name in ("A", "B"):
                if not active[name]:
                    continue
                label, stopped = classify_label(
                    current_grid=grids[name],
                    new_grid=new_grids[name],
                    prev_grid=prev_grids[name],
                    history=recent_states[name],
                )
                if stopped:
                    labels[name] = label or "max"
                    stop_steps[name] = step
                    active[name] = False

            grids = new_grids
            for name in ("A", "B"):
                if active[name]:
                    prev_grids[name] = grids[name].copy()

            if not any(active.values()):
                break

        for name in ("A", "B"):
            grid_records.append(
                {
                    "strategy": strategy,
                    "warmup": warmup,
                    "trial": trial,
                    "grid_id": name,
                    "label": labels[name],
                    "stop_step": stop_steps[name],
                }
            )

        both_osc = labels["A"] == "osc" and labels["B"] == "osc"
        one_side_osc = (labels["A"] == "osc") ^ (labels["B"] == "osc")
        pair_records.append(
            {
                "strategy": strategy,
                "warmup": warmup,
                "trial": trial,
                "label_A": labels["A"],
                "label_B": labels["B"],
                "stop_A": stop_steps["A"],
                "stop_B": stop_steps["B"],
                "stop_diff_abs": abs(stop_steps["A"] - stop_steps["B"]),
                "both_osc": int(both_osc),
                "one_side_osc": int(one_side_osc),
            }
        )

    return grid_records, pair_records


def summarize(grid_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for warmup in WARMUPS:
        strategy = strategy_name(warmup)
        grid_group = grid_df[grid_df["warmup"] == warmup]
        pair_group = pair_df[pair_df["warmup"] == warmup]
        ratios = grid_group["label"].value_counts(normalize=True)
        rows.append(
            {
                "warmup": warmup,
                "strategy": strategy,
                "osc_rate": float(ratios.get("osc", 0.0)),
                "osc_pair_rate": float(pair_group["both_osc"].mean()),
                "one_side_osc_rate": float(pair_group["one_side_osc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("warmup").reset_index(drop=True)


def stop_diff_distribution(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for warmup in WARMUPS:
        strategy = strategy_name(warmup)
        group = pair_df[pair_df["warmup"] == warmup]
        dist = group["stop_diff_abs"].value_counts(normalize=True).sort_index()
        counts = group["stop_diff_abs"].value_counts().sort_index()
        for stop_diff, prob in dist.items():
            rows.append(
                {
                    "warmup": warmup,
                    "strategy": strategy,
                    "stop_diff_abs": int(stop_diff),
                    "probability": float(prob),
                    "count": int(counts.loc[stop_diff]),
                }
            )
    return pd.DataFrame(rows)


def render_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_outputs(
    grid_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stop_diff_df: pd.DataFrame,
) -> None:
    grid_csv = ROOT / "gol_density070_warmup_branch_trials.csv"
    pair_csv = ROOT / "gol_density070_warmup_branch_pairs.csv"
    summary_csv = ROOT / "gol_density070_warmup_branch_summary.csv"
    stop_diff_csv = ROOT / "gol_density070_warmup_branch_stopdiff.csv"
    report_md = ROOT / "gol_density070_warmup_branch_report.md"

    grid_df.to_csv(grid_csv, index=False, encoding="utf-8")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    stop_diff_df.to_csv(stop_diff_csv, index=False, encoding="utf-8")

    lines = [
        "# density=0.70 warmup branch scan",
        "",
        "- Rule: Conway's Game of Life (B3/S23), toroidal boundary, size=40.",
        "- Coupling: symmetric weak-average.",
        f"- density={DENSITY}, warmups={WARMUPS}, trials={N_TRIALS}, max_steps={MAX_STEPS}, seed={SEED}",
        "- `osc_rate` is the per-grid oscillatory stop ratio.",
        "- `osc_pair_rate` is the trial ratio where both grids stop via `osc`.",
        "- `one_side_osc_rate` is the trial ratio where exactly one grid stops via `osc`.",
        "- Full |stop_A-stop_B| distributions are saved in `gol_density070_warmup_branch_stopdiff.csv`.",
        "",
        "## Summary",
        "",
        "| warmup | strategy | osc rate | osc pair rate | one-side osc rate |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "| {warmup} | {strategy} | {osc_rate} | {osc_pair_rate} | {one_side_osc_rate} |".format(
                warmup=int(row["warmup"]),
                strategy=row["strategy"],
                osc_rate=render_metric(row["osc_rate"]),
                osc_pair_rate=render_metric(row["osc_pair_rate"]),
                one_side_osc_rate=render_metric(row["one_side_osc_rate"]),
            )
        )

    for warmup in WARMUPS:
        strategy = strategy_name(warmup)
        lines.append("")
        lines.append(f"## |stop_A-stop_B| distribution: {strategy}")
        lines.append("")
        lines.append("| stop diff | probability | count |")
        lines.append("| ---: | ---: | ---: |")
        subset = stop_diff_df[stop_diff_df["warmup"] == warmup]
        for _, row in subset.iterrows():
            lines.append(
                "| {diff} | {prob} | {count} |".format(
                    diff=int(row["stop_diff_abs"]),
                    prob=render_metric(row["probability"]),
                    count=int(row["count"]),
                )
            )

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved grid trial CSV: {grid_csv}")
    print(f"Saved pair CSV: {pair_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved stop-diff distribution CSV: {stop_diff_csv}")
    print(f"Saved report: {report_md}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []

    for warmup in WARMUPS:
        cond_grid_records, cond_pair_records = simulate_condition(warmup, rng)
        grid_records.extend(cond_grid_records)
        pair_records.extend(cond_pair_records)

        cond_df = pd.DataFrame(cond_grid_records)
        cond_pair_df = pd.DataFrame(cond_pair_records)
        ratios = cond_df["label"].value_counts(normalize=True)
        print(
            "warmup={warmup} osc={osc:.3f} osc_pair={osc_pair:.3f} one_side_osc={one_side:.3f}".format(
                warmup=warmup,
                osc=ratios.get("osc", 0.0),
                osc_pair=cond_pair_df["both_osc"].mean(),
                one_side=cond_pair_df["one_side_osc"].mean(),
            ),
            flush=True,
        )

    grid_df = pd.DataFrame(grid_records)
    pair_df = pd.DataFrame(pair_records)
    summary_df = summarize(grid_df, pair_df)
    stop_diff_df = stop_diff_distribution(pair_df)
    save_outputs(grid_df, pair_df, summary_df, stop_diff_df)


if __name__ == "__main__":
    main()
