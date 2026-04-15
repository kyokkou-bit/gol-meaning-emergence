from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\AI-workspace\test")
DENSITIES = (0.73, 0.74, 0.75)
COUPLINGS = (0.00, 0.01, 0.02, 0.05, 0.10)
SIZE = 40
N_TRIALS = 300
MAX_STEPS = 1000
EPSILON = 0.005
OSC_MIN_PERIOD = 2
OSC_MAX_PERIOD = 10
SEED = 42
TYPE_ORDER = ("fixed", "osc", "delta", "max")


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


def apply_coupling(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    coupling: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if coupling <= 0.0:
        return grid_a, grid_b

    mask_a = rng.random(grid_a.shape) < coupling
    mask_b = rng.random(grid_b.shape) < coupling
    coupled_a = np.where(mask_a, grid_b, grid_a).astype(np.uint8)
    coupled_b = np.where(mask_b, grid_a, grid_b).astype(np.uint8)
    return coupled_a, coupled_b


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    sorted_events = events[order]
    unique_times = np.unique(sorted_times)

    at_risk = len(sorted_times)
    survival = 1.0
    km_times: list[float] = []
    km_survival: list[float] = []

    for current_time in unique_times:
        mask = sorted_times == current_time
        n_total = int(np.sum(mask))
        n_events = int(np.sum(sorted_events[mask]))
        if at_risk > 0 and n_events > 0:
            survival *= 1.0 - (n_events / at_risk)
            km_times.append(float(current_time))
            km_survival.append(float(survival))
        at_risk -= n_total

    return np.asarray(km_times, dtype=float), np.asarray(km_survival, dtype=float)


def fit_alpha_from_km(times: np.ndarray, events: np.ndarray) -> float | None:
    km_times, km_survival = kaplan_meier(times, events)
    mask = (km_times > 0) & (km_survival > 0)
    if np.count_nonzero(mask) < 4:
        return None

    log_x = np.log(km_times[mask])
    log_y = np.log(km_survival[mask])
    slope = np.polyfit(log_x, log_y, 1)[0]
    return float(1.0 - slope)


def classify_two_grids(
    density: float,
    coupling: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []

    for trial in range(1, N_TRIALS + 1):
        grids = [
            (rng.random((SIZE, SIZE)) < density).astype(np.uint8),
            (rng.random((SIZE, SIZE)) < density).astype(np.uint8),
        ]
        prev_grids = [grid.copy() for grid in grids]
        recent_states = [deque(maxlen=OSC_MAX_PERIOD), deque(maxlen=OSC_MAX_PERIOD)]
        labels = ["max", "max"]
        stop_steps = [MAX_STEPS, MAX_STEPS]
        active = [True, True]

        for step in range(1, MAX_STEPS + 1):
            new_grids = [grid.copy() for grid in grids]
            for idx in range(2):
                if active[idx]:
                    new_grids[idx] = gol_step(grids[idx])

            new_grids[0], new_grids[1] = apply_coupling(new_grids[0], new_grids[1], coupling, rng)

            for idx in range(2):
                if not active[idx]:
                    continue

                if np.array_equal(new_grids[idx], grids[idx]):
                    labels[idx] = "fixed"
                    stop_steps[idx] = step
                    active[idx] = False
                    continue

                new_state = new_grids[idx].tobytes()
                recent_states[idx].append(grids[idx].tobytes())
                max_period = min(OSC_MAX_PERIOD, len(recent_states[idx]))
                oscillated = False
                for period in range(OSC_MIN_PERIOD, max_period + 1):
                    if new_state == recent_states[idx][-period]:
                        labels[idx] = "osc"
                        stop_steps[idx] = step
                        active[idx] = False
                        oscillated = True
                        break
                if oscillated:
                    continue

                delta = np.count_nonzero(new_grids[idx] != prev_grids[idx]) / (SIZE * SIZE)
                if delta < EPSILON:
                    labels[idx] = "delta"
                    stop_steps[idx] = step
                    active[idx] = False
                    continue

            grids = new_grids
            for idx in range(2):
                if active[idx]:
                    prev_grids[idx] = grids[idx].copy()

            if not any(active):
                break

        for idx in range(2):
            grid_records.append(
                {
                    "density": density,
                    "coupling": coupling,
                    "trial": trial,
                    "grid_id": idx + 1,
                    "label": labels[idx],
                    "stop_step": stop_steps[idx],
                    "stopped_by_1000": 0 if labels[idx] == "max" else 1,
                }
            )

        pair_records.append(
            {
                "density": density,
                "coupling": coupling,
                "trial": trial,
                "grid1_label": labels[0],
                "grid2_label": labels[1],
                "mode_match": int(labels[0] == labels[1]),
                "grid1_stop_step": stop_steps[0],
                "grid2_stop_step": stop_steps[1],
            }
        )

    return grid_records, pair_records


def summarize(grid_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (density, coupling), group in grid_df.groupby(["density", "coupling"], sort=True):
        pair_group = pair_df[(np.isclose(pair_df["density"], density)) & (np.isclose(pair_df["coupling"], coupling))]
        row: dict[str, object] = {
            "density": float(density),
            "coupling": float(coupling),
            "n_grid_runs": int(len(group)),
            "n_trials": int(len(pair_group)),
        }

        ratios = group["label"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0.0)
        for label in TYPE_ORDER:
            row[f"{label}_ratio"] = float(ratios[label])

        times = group["stop_step"].to_numpy(dtype=float)
        events = group["stopped_by_1000"].to_numpy(dtype=int)
        _, km_survival = kaplan_meier(times, events)
        row["survival_at_1000"] = float(km_survival[-1]) if len(km_survival) else 1.0

        for label in ("osc", "delta"):
            label_events = (group["label"] == label).to_numpy(dtype=int)
            row[f"{label}_alpha_km"] = fit_alpha_from_km(times, label_events)

        row["mode_match_rate"] = float(pair_group["mode_match"].mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["density", "coupling"]).reset_index(drop=True)


def render_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_outputs(grid_df: pd.DataFrame, pair_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    grid_csv = ROOT / "gol_two_grid_coupling_trials.csv"
    pair_csv = ROOT / "gol_two_grid_coupling_pairs.csv"
    summary_csv = ROOT / "gol_two_grid_coupling_summary.csv"
    report_md = ROOT / "gol_two_grid_coupling_report.md"

    grid_df.to_csv(grid_csv, index=False, encoding="utf-8")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    lines = [
        "# Two-grid weak-coupling scan",
        "",
        "- Rule: Conway's Game of Life (B3/S23), toroidal boundary, size=40.",
        "- Each trial simulates two grids with the same initial density.",
        "- Coupling rule: after each independent GoL update, each cell copies the corresponding state from the other grid with probability `g`, independently for grid A and grid B.",
        f"- Densities: {DENSITIES}",
        f"- Couplings: {COUPLINGS}",
        f"- Trials per condition: {N_TRIALS}",
        f"- Max steps: {MAX_STEPS}",
        "- `mode_match_rate` is the fraction of trials where both grids end in the same stop mode.",
        "",
    ]

    for density in DENSITIES:
        lines.append(f"## density={density:.2f}")
        lines.append("")
        lines.append(
            "| coupling | fixed | osc | delta | max | survival@1000 | osc alpha KM | delta alpha KM | mode match |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        subset = summary_df[np.isclose(summary_df["density"], density)]
        for _, row in subset.iterrows():
            lines.append(
                "| {coupling:.2f} | {fixed} | {osc} | {delta} | {maxv} | {survival} | {osc_alpha} | {delta_alpha} | {match} |".format(
                    coupling=row["coupling"],
                    fixed=render_metric(row["fixed_ratio"]),
                    osc=render_metric(row["osc_ratio"]),
                    delta=render_metric(row["delta_ratio"]),
                    maxv=render_metric(row["max_ratio"]),
                    survival=render_metric(row["survival_at_1000"]),
                    osc_alpha=render_metric(row["osc_alpha_km"]),
                    delta_alpha=render_metric(row["delta_alpha_km"]),
                    match=render_metric(row["mode_match_rate"]),
                )
            )
        lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved grid trial CSV: {grid_csv}")
    print(f"Saved pair CSV: {pair_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved report: {report_md}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []

    for density in DENSITIES:
        for coupling in COUPLINGS:
            cond_grid_records, cond_pair_records = classify_two_grids(density, coupling, rng)
            grid_records.extend(cond_grid_records)
            pair_records.extend(cond_pair_records)

            cond_df = pd.DataFrame(cond_grid_records)
            ratios = cond_df["label"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0.0)
            match_rate = pd.DataFrame(cond_pair_records)["mode_match"].mean()
            print(
                "density={density:.2f} coupling={coupling:.2f} fixed={fixed:.3f} osc={osc:.3f} delta={delta:.3f} max={maxv:.3f} match={match:.3f}".format(
                    density=density,
                    coupling=coupling,
                    fixed=ratios["fixed"],
                    osc=ratios["osc"],
                    delta=ratios["delta"],
                    maxv=ratios["max"],
                    match=match_rate,
                ),
                flush=True,
            )

    grid_df = pd.DataFrame(grid_records)
    pair_df = pd.DataFrame(pair_records)
    summary_df = summarize(grid_df, pair_df)
    save_outputs(grid_df, pair_df, summary_df)


if __name__ == "__main__":
    main()
