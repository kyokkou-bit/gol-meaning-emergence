from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\AI-workspace\test")
DENSITIES = (0.73, 0.74)
SIZE = 40
N_TRIALS = 300
MAX_STEPS = 1000
EPSILON = 0.005
OSC_MIN_PERIOD = 2
OSC_MAX_PERIOD = 10
SEED = 42
TYPE_ORDER = ("fixed", "osc", "delta", "max")
G_MAX = 0.005
STRATEGIES = ("fixed_0.005", "warmup_5", "warmup_10", "warmup_20", "warmup_50", "warmup_100")


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
    active_a: bool,
    active_b: bool,
    coupling: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    out_a = grid_a.copy()
    out_b = grid_b.copy()
    if active_a:
        prob_a = np.clip((1.0 - coupling) * grid_a.astype(float) + coupling * local_mean(grid_b), 0.0, 1.0)
        out_a = (rng.random(grid_a.shape) < prob_a).astype(np.uint8)
    if active_b:
        prob_b = np.clip((1.0 - coupling) * grid_b.astype(float) + coupling * local_mean(grid_a), 0.0, 1.0)
        out_b = (rng.random(grid_b.shape) < prob_b).astype(np.uint8)
    return out_a, out_b


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


def strategy_to_warmup(strategy: str) -> int | None:
    if strategy == "fixed_0.005":
        return None
    return int(strategy.split("_")[1])


def choose_g(strategy: str, step: int) -> float:
    warmup = strategy_to_warmup(strategy)
    if warmup is None:
        return G_MAX
    return 0.0 if step <= warmup else G_MAX


def simulate_condition(
    density: float,
    strategy: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grid_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []
    g_records: list[dict[str, object]] = []

    for trial in range(1, N_TRIALS + 1):
        grids = {
            "A": (rng.random((SIZE, SIZE)) < density).astype(np.uint8),
            "B": (rng.random((SIZE, SIZE)) < density).astype(np.uint8),
        }
        prev_grids = {name: grid.copy() for name, grid in grids.items()}
        recent_states = {name: deque(maxlen=OSC_MAX_PERIOD) for name in grids}
        labels = {"A": "max", "B": "max"}
        stop_steps = {"A": MAX_STEPS, "B": MAX_STEPS}
        active = {"A": True, "B": True}
        trial_g_used: list[float] = []

        for step in range(1, MAX_STEPS + 1):
            current_g = choose_g(strategy, step)
            trial_g_used.append(current_g)

            new_a = gol_step(grids["A"]) if active["A"] else grids["A"].copy()
            new_b = gol_step(grids["B"]) if active["B"] else grids["B"].copy()
            coupled_a, coupled_b = weak_average_coupling(new_a, new_b, active["A"], active["B"], current_g, rng)
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
                    "density": density,
                    "strategy": strategy,
                    "trial": trial,
                    "grid_id": name,
                    "label": labels[name],
                    "stop_step": stop_steps[name],
                    "stopped_by_1000": 0 if labels[name] == "max" else 1,
                }
            )

        pair_records.append(
            {
                "density": density,
                "strategy": strategy,
                "trial": trial,
                "label_A": labels["A"],
                "label_B": labels["B"],
                "stop_A": stop_steps["A"],
                "stop_B": stop_steps["B"],
                "stop_diff_abs": abs(stop_steps["A"] - stop_steps["B"]),
                "mode_match": int(labels["A"] == labels["B"]),
            }
        )

        g_records.append(
            {
                "density": density,
                "strategy": strategy,
                "trial": trial,
                "g_mean_used": float(np.mean(trial_g_used)),
            }
        )

    return grid_records, pair_records, g_records


def summarize(grid_df: pd.DataFrame, pair_df: pd.DataFrame, g_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (density, strategy), group in grid_df.groupby(["density", "strategy"], sort=True):
        pair_group = pair_df[np.isclose(pair_df["density"], density) & (pair_df["strategy"] == strategy)]
        g_group = g_df[np.isclose(g_df["density"], density) & (g_df["strategy"] == strategy)]
        row: dict[str, object] = {
            "density": float(density),
            "strategy": str(strategy),
            "warmup_length": strategy_to_warmup(strategy),
        }
        ratios = group["label"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0.0)
        for label in TYPE_ORDER:
            row[f"{label}_ratio"] = float(ratios[label])
        times = group["stop_step"].to_numpy(dtype=float)
        events = group["stopped_by_1000"].to_numpy(dtype=int)
        _, km_survival = kaplan_meier(times, events)
        row["survival_at_1000"] = float(km_survival[-1]) if len(km_survival) else 1.0
        row["mode_match_rate"] = float(pair_group["mode_match"].mean())
        row["mean_abs_stop_diff"] = float(pair_group["stop_diff_abs"].mean())
        row["g_mean_used"] = float(g_group["g_mean_used"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["density", "warmup_length"], na_position="first").reset_index(drop=True)


def stop_diff_distribution(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (density, strategy), group in pair_df.groupby(["density", "strategy"], sort=True):
        dist = group["stop_diff_abs"].value_counts(normalize=True).sort_index()
        for stop_diff, prob in dist.items():
            rows.append(
                {
                    "density": float(density),
                    "strategy": str(strategy),
                    "stop_diff_abs": int(stop_diff),
                    "probability": float(prob),
                    "count": int(round(prob * len(group))),
                }
            )
    return pd.DataFrame(rows)


def warmup_focus_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = summary_df[summary_df["strategy"] != "fixed_0.005"]
    for _, row in subset.iterrows():
        rows.append(
            {
                "density": row["density"],
                "warmup_length": int(row["warmup_length"]),
                "osc_ratio": row["osc_ratio"],
                "mode_match_rate": row["mode_match_rate"],
                "mean_abs_stop_diff": row["mean_abs_stop_diff"],
            }
        )
    return pd.DataFrame(rows).sort_values(["density", "warmup_length"]).reset_index(drop=True)


def render_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_outputs(
    grid_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    g_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stopdiff_df: pd.DataFrame,
    warmup_df: pd.DataFrame,
) -> None:
    grid_csv = ROOT / "gol_weak_average_warmup_trials.csv"
    pair_csv = ROOT / "gol_weak_average_warmup_pairs.csv"
    g_csv = ROOT / "gol_weak_average_warmup_gmean.csv"
    summary_csv = ROOT / "gol_weak_average_warmup_summary.csv"
    stopdiff_csv = ROOT / "gol_weak_average_warmup_stopdiff.csv"
    warmup_csv = ROOT / "gol_weak_average_warmup_focus.csv"
    report_md = ROOT / "gol_weak_average_warmup_report.md"

    grid_df.to_csv(grid_csv, index=False, encoding="utf-8")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8")
    g_df.to_csv(g_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    stopdiff_df.to_csv(stopdiff_csv, index=False, encoding="utf-8")
    warmup_df.to_csv(warmup_csv, index=False, encoding="utf-8")

    lines = [
        "# Weak-average warmup sweep",
        "",
        "- Rule: Conway's Game of Life (B3/S23), toroidal boundary, size=40.",
        "- Coupling: symmetric weak-average.",
        "- `fixed_0.005`: constant baseline.",
        "- `warmup_k`: use `g = 0` for the first `k` steps, then `g = 0.005` thereafter.",
        f"- Densities: {DENSITIES}, trials={N_TRIALS}, max_steps={MAX_STEPS}",
        "- Exact `|stop_A-stop_B|` distributions are saved in `gol_weak_average_warmup_stopdiff.csv`.",
        "",
    ]

    for density in DENSITIES:
        lines.append(f"## density={density:.2f}")
        lines.append("")
        lines.append("| strategy | fixed | osc | delta | max | mode match | mean |stop_A-stop_B| | survival@1000 | mean g used |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        subset = summary_df[np.isclose(summary_df["density"], density)]
        for _, row in subset.iterrows():
            lines.append(
                "| {strategy} | {fixed} | {osc} | {delta} | {maxv} | {match} | {mean_diff} | {survival} | {gmean} |".format(
                    strategy=row["strategy"],
                    fixed=render_metric(row["fixed_ratio"]),
                    osc=render_metric(row["osc_ratio"]),
                    delta=render_metric(row["delta_ratio"]),
                    maxv=render_metric(row["max_ratio"]),
                    match=render_metric(row["mode_match_rate"]),
                    mean_diff=render_metric(row["mean_abs_stop_diff"]),
                    survival=render_metric(row["survival_at_1000"]),
                    gmean=render_metric(row["g_mean_used"]),
                )
            )
        lines.append("")
        lines.append("### Warmup Length vs osc / mode match")
        lines.append("")
        lines.append("| warmup | osc | mode match | mean |stop_A-stop_B| |")
        lines.append("| ---: | ---: | ---: | ---: |")
        warm_subset = warmup_df[np.isclose(warmup_df["density"], density)]
        for _, row in warm_subset.iterrows():
            lines.append(
                "| {warmup} | {osc} | {match} | {mean_diff} |".format(
                    warmup=int(row["warmup_length"]),
                    osc=render_metric(row["osc_ratio"]),
                    match=render_metric(row["mode_match_rate"]),
                    mean_diff=render_metric(row["mean_abs_stop_diff"]),
                )
            )
        lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved grid trial CSV: {grid_csv}")
    print(f"Saved pair CSV: {pair_csv}")
    print(f"Saved g-mean CSV: {g_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved stopdiff CSV: {stopdiff_csv}")
    print(f"Saved warmup focus CSV: {warmup_csv}")
    print(f"Saved report: {report_md}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid_records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []
    g_records: list[dict[str, object]] = []

    for density in DENSITIES:
        for strategy in STRATEGIES:
            cond_grid_records, cond_pair_records, cond_g_records = simulate_condition(density, strategy, rng)
            grid_records.extend(cond_grid_records)
            pair_records.extend(cond_pair_records)
            g_records.extend(cond_g_records)

            cond_df = pd.DataFrame(cond_grid_records)
            pair_df = pd.DataFrame(cond_pair_records)
            ratios = cond_df["label"].value_counts(normalize=True).reindex(TYPE_ORDER, fill_value=0.0)
            print(
                "density={density:.2f} strategy={strategy} fixed={fixed:.3f} osc={osc:.3f} delta={delta:.3f} max={maxv:.3f} match={match:.3f}".format(
                    density=density,
                    strategy=strategy,
                    fixed=ratios["fixed"],
                    osc=ratios["osc"],
                    delta=ratios["delta"],
                    maxv=ratios["max"],
                    match=pair_df["mode_match"].mean(),
                ),
                flush=True,
            )

    grid_df = pd.DataFrame(grid_records)
    pair_df = pd.DataFrame(pair_records)
    g_df = pd.DataFrame(g_records)
    summary_df = summarize(grid_df, pair_df, g_df)
    stopdiff_df = stop_diff_distribution(pair_df)
    warmup_df = warmup_focus_table(summary_df)
    save_outputs(grid_df, pair_df, g_df, summary_df, stopdiff_df, warmup_df)


if __name__ == "__main__":
    main()
