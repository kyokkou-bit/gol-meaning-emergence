from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\AI-workspace\test")
DENSITIES = np.round(np.arange(0.05, 1.00, 0.05), 2)
SIZES = (40, 80)
N_TRIALS = 200
MAX_STEPS = 1000
EPSILON = 0.005
OSC_MIN_PERIOD = 2
OSC_MAX_PERIOD = 10
SEED = 42


@dataclass
class TrialOutcome:
    label: str
    step: int


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


def fit_ccdf_power_law(steps: list[int]) -> tuple[float | None, float | None]:
    if len(steps) < 4:
        return None, None

    sorted_steps = np.sort(np.asarray(steps, dtype=float))
    unique, counts = np.unique(sorted_steps, return_counts=True)
    ccdf = np.cumsum(counts[::-1])[::-1] / len(sorted_steps)

    mask = (unique > 0) & (ccdf > 0)
    if np.count_nonzero(mask) < 4:
        return None, None

    log_x = np.log(unique[mask])
    log_y = np.log(ccdf[mask])
    coeffs = np.polyfit(log_x, log_y, 1)
    slope = coeffs[0]
    pred = slope * log_x + coeffs[1]
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    alpha = 1.0 - slope
    return alpha, r2


def simulate_density(
    size: int,
    density: float,
    n_trials: int,
    max_steps: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    counts = {"fixed": 0, "osc": 0, "delta": 0, "max": 0}
    steps_by_label = {"fixed": [], "osc": [], "delta": [], "max": []}

    for _ in range(n_trials):
        grid = (rng.random((size, size)) < density).astype(np.uint8)
        recent_states: deque[bytes] = deque(maxlen=OSC_MAX_PERIOD)
        prev_grid = grid.copy()
        outcome = TrialOutcome("max", max_steps)

        for step in range(1, max_steps + 1):
            new_grid = gol_step(grid)

            if np.array_equal(new_grid, grid):
                outcome = TrialOutcome("fixed", step)
                break

            new_state = new_grid.tobytes()
            recent_states.append(grid.tobytes())
            max_period = min(OSC_MAX_PERIOD, len(recent_states))
            for period in range(OSC_MIN_PERIOD, max_period + 1):
                if new_state == recent_states[-period]:
                    outcome = TrialOutcome("osc", step)
                    break
            if outcome.label == "osc":
                break

            delta = np.count_nonzero(new_grid != prev_grid) / (size * size)
            if delta < epsilon:
                outcome = TrialOutcome("delta", step)
                break

            prev_grid = new_grid
            grid = new_grid

        counts[outcome.label] += 1
        steps_by_label[outcome.label].append(outcome.step)

    row: dict[str, object] = {
        "size": size,
        "density": density,
        "n_trials": n_trials,
        "max_steps": max_steps,
        "epsilon": epsilon,
    }

    for label in ("fixed", "osc", "delta", "max"):
        row[f"{label}_count"] = counts[label]
        row[f"{label}_ratio"] = counts[label] / n_trials

    for label in ("osc", "delta"):
        alpha, r2 = fit_ccdf_power_law(steps_by_label[label])
        row[f"{label}_alpha"] = alpha
        row[f"{label}_r2"] = r2

    return row


def format_metric(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "size",
        "density",
        "n_trials",
        "max_steps",
        "epsilon",
        "fixed_count",
        "fixed_ratio",
        "osc_count",
        "osc_ratio",
        "osc_alpha",
        "osc_r2",
        "delta_count",
        "delta_ratio",
        "delta_alpha",
        "delta_r2",
        "max_count",
        "max_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# GoL density scan",
        "",
        "- Rule: Conway's Game of Life (B3/S23) with toroidal boundary (`np.roll` wrap).",
        f"- Densities: {DENSITIES[0]:.2f} to {DENSITIES[-1]:.2f} in 0.05 steps.",
        f"- Trials per density: {N_TRIALS}",
        f"- Max steps: {MAX_STEPS}",
        f"- Delta threshold: {EPSILON}",
        "- `fixed`: next state equals current state.",
        "- `osc`: next state matches a prior state with period 2-10.",
        "- `delta`: fraction of changed cells between consecutive states is below epsilon before fixed/osc triggers.",
        "- `max`: no stop condition met by `max_steps`.",
        "",
    ]

    for size in SIZES:
        lines.append(f"## size={size}")
        lines.append("")
        lines.append(
            "| density | fixed | osc | osc alpha | osc R^2 | delta | delta alpha | delta R^2 | max |"
        )
        lines.append(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in (r for r in rows if r["size"] == size):
            lines.append(
                "| {density:.2f} | {fixed_ratio} | {osc_ratio} | {osc_alpha} | {osc_r2} | {delta_ratio} | {delta_alpha} | {delta_r2} | {max_ratio} |".format(
                    density=row["density"],
                    fixed_ratio=format_metric(row["fixed_ratio"]),
                    osc_ratio=format_metric(row["osc_ratio"]),
                    osc_alpha=format_metric(row["osc_alpha"]),
                    osc_r2=format_metric(row["osc_r2"]),
                    delta_ratio=format_metric(row["delta_ratio"]),
                    delta_alpha=format_metric(row["delta_alpha"]),
                    delta_r2=format_metric(row["delta_r2"]),
                    max_ratio=format_metric(row["max_ratio"]),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for size in SIZES:
        for density in DENSITIES:
            row = simulate_density(
                size=size,
                density=float(density),
                n_trials=N_TRIALS,
                max_steps=MAX_STEPS,
                epsilon=EPSILON,
                rng=rng,
            )
            rows.append(row)
            print(
                "size={size} density={density:.2f} fixed={fixed:.3f} osc={osc:.3f} delta={delta:.3f} max={maxv:.3f}".format(
                    size=size,
                    density=row["density"],
                    fixed=row["fixed_ratio"],
                    osc=row["osc_ratio"],
                    delta=row["delta_ratio"],
                    maxv=row["max_ratio"],
                ),
                flush=True,
            )

    csv_path = ROOT / "gol_density_scan_results.csv"
    md_path = ROOT / "gol_density_scan_results.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
