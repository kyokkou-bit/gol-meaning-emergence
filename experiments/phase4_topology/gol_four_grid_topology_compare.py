from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\AI-workspace\test")
DENSITY = 0.70
SIZE = 40
N_TRIALS = 500
MAX_STEPS = 1000
EPSILON = 0.005
OSC_MIN_PERIOD = 2
OSC_MAX_PERIOD = 10
SEED = 42
G = 0.002
WARMUP = 10
GRID_IDS = ("A", "B", "C", "D")
PAIR_IDS = tuple(combinations(GRID_IDS, 2))
TOPOLOGIES = ("all_to_all", "ring_1d")
NEIGHBORS = {
    "all_to_all": {
        "A": ("B", "C", "D"),
        "B": ("A", "C", "D"),
        "C": ("A", "B", "D"),
        "D": ("A", "B", "C"),
    },
    "ring_1d": {
        "A": ("B", "D"),
        "B": ("A", "C"),
        "C": ("B", "D"),
        "D": ("C", "A"),
    },
}


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


def choose_g(step: int) -> float:
    return 0.0 if step <= WARMUP else G


def weak_average_coupling(
    grids: dict[str, np.ndarray],
    active: dict[str, bool],
    topology: str,
    coupling: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if coupling <= 0.0:
        return {name: grid.copy() for name, grid in grids.items()}

    local_means = {name: local_mean(grid) for name, grid in grids.items()}
    out: dict[str, np.ndarray] = {}
    for name in GRID_IDS:
        if not active[name]:
            out[name] = grids[name].copy()
            continue
        neighbor_names = NEIGHBORS[topology][name]
        neighbor_signal = np.mean([local_means[other] for other in neighbor_names], axis=0)
        prob = np.clip((1.0 - coupling) * grids[name].astype(float) + coupling * neighbor_signal, 0.0, 1.0)
        out[name] = (rng.random(grids[name].shape) < prob).astype(np.uint8)
    return out


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


def cluster_type(labels: list[str]) -> str:
    counts = sorted(Counter(labels).values(), reverse=True)
    if counts == [4]:
        return "4+0"
    if counts == [3, 1]:
        return "3+1"
    if counts == [2, 2]:
        return "2+2"
    if counts == [2, 1, 1]:
        return "2+1+1"
    return "all_diff"


def cluster_signature(labels: dict[str, str]) -> str:
    counts = Counter(labels.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return " / ".join(f"{label}x{count}" for label, count in ordered)


def within_between_stop_diff(labels: dict[str, str], stops: dict[str, int]) -> tuple[float | None, float | None]:
    within: list[int] = []
    between: list[int] = []
    for left, right in PAIR_IDS:
        diff = abs(stops[left] - stops[right])
        if labels[left] == labels[right]:
            within.append(diff)
        else:
            between.append(diff)
    return (
        float(np.mean(within)) if within else None,
        float(np.mean(between)) if between else None,
    )


def simulate_topology(
    topology: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_records: list[dict[str, object]] = []
    trial_records: list[dict[str, object]] = []

    for trial in range(1, N_TRIALS + 1):
        grids = {name: (rng.random((SIZE, SIZE)) < DENSITY).astype(np.uint8) for name in GRID_IDS}
        prev_grids = {name: grid.copy() for name, grid in grids.items()}
        recent_states = {name: deque(maxlen=OSC_MAX_PERIOD) for name in GRID_IDS}
        labels = {name: "max" for name in GRID_IDS}
        stop_steps = {name: MAX_STEPS for name in GRID_IDS}
        active = {name: True for name in GRID_IDS}

        for step in range(1, MAX_STEPS + 1):
            current_g = choose_g(step)
            evolved = {name: gol_step(grids[name]) if active[name] else grids[name].copy() for name in GRID_IDS}
            new_grids = weak_average_coupling(evolved, active, topology, current_g, rng)

            for name in GRID_IDS:
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
            for name in GRID_IDS:
                if active[name]:
                    prev_grids[name] = grids[name].copy()

            if not any(active.values()):
                break

        for name in GRID_IDS:
            grid_records.append(
                {
                    "topology": topology,
                    "trial": trial,
                    "grid_id": name,
                    "label": labels[name],
                    "stop_step": stop_steps[name],
                }
            )

        within_mean, between_mean = within_between_stop_diff(labels, stop_steps)
        trial_record = {
            "topology": topology,
            "trial": trial,
            "cluster_type": cluster_type(list(labels.values())),
            "cluster_signature": cluster_signature(labels),
            "within_cluster_stop_diff_mean": within_mean,
            "between_cluster_stop_diff_mean": between_mean,
        }
        for name in GRID_IDS:
            trial_record[f"label_{name}"] = labels[name]
            trial_record[f"stop_{name}"] = stop_steps[name]
        trial_records.append(trial_record)

    return grid_records, trial_records


def summarize(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    structure_rows: list[dict[str, object]] = []
    content_rows: list[dict[str, object]] = []

    for topology in TOPOLOGIES:
        subset = trial_df[trial_df["topology"] == topology]
        for cluster_name in ("4+0", "3+1", "2+2", "2+1+1", "all_diff"):
            cluster_subset = subset[subset["cluster_type"] == cluster_name]
            structure_rows.append(
                {
                    "topology": topology,
                    "cluster_type": cluster_name,
                    "rate": float(len(cluster_subset) / len(subset)),
                    "within_cluster_stop_diff_mean": float(cluster_subset["within_cluster_stop_diff_mean"].dropna().mean())
                    if cluster_subset["within_cluster_stop_diff_mean"].notna().any()
                    else None,
                    "between_cluster_stop_diff_mean": float(cluster_subset["between_cluster_stop_diff_mean"].dropna().mean())
                    if cluster_subset["between_cluster_stop_diff_mean"].notna().any()
                    else None,
                }
            )

        signature_counts = subset["cluster_signature"].value_counts(normalize=True)
        signature_cluster_type = subset.drop_duplicates("cluster_signature").set_index("cluster_signature")["cluster_type"]
        signature_within = subset.groupby("cluster_signature")["within_cluster_stop_diff_mean"].mean()
        signature_between = subset.groupby("cluster_signature")["between_cluster_stop_diff_mean"].mean()
        for signature, rate in signature_counts.items():
            content_rows.append(
                {
                    "topology": topology,
                    "cluster_type": signature_cluster_type.loc[signature],
                    "cluster_signature": signature,
                    "rate": float(rate),
                    "within_cluster_stop_diff_mean": float(signature_within.loc[signature])
                    if pd.notna(signature_within.loc[signature])
                    else None,
                    "between_cluster_stop_diff_mean": float(signature_between.loc[signature])
                    if pd.notna(signature_between.loc[signature])
                    else None,
                }
            )

    return (
        pd.DataFrame(structure_rows).sort_values(["topology", "cluster_type"]).reset_index(drop=True),
        pd.DataFrame(content_rows).sort_values(["topology", "cluster_type", "rate"], ascending=[True, True, False]).reset_index(drop=True),
    )


def render_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_outputs(
    grid_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    content_df: pd.DataFrame,
) -> None:
    grid_csv = ROOT / "gol_four_grid_topology_trials.csv"
    trial_csv = ROOT / "gol_four_grid_topology_clusters.csv"
    structure_csv = ROOT / "gol_four_grid_topology_structure_summary.csv"
    content_csv = ROOT / "gol_four_grid_topology_content_summary.csv"
    report_md = ROOT / "gol_four_grid_topology_report.md"

    grid_df.to_csv(grid_csv, index=False, encoding="utf-8")
    trial_df.to_csv(trial_csv, index=False, encoding="utf-8")
    structure_df.to_csv(structure_csv, index=False, encoding="utf-8")
    content_df.to_csv(content_csv, index=False, encoding="utf-8")

    lines = [
        "# 4-grid topology comparison",
        "",
        "- Rule: Conway's Game of Life (B3/S23), toroidal boundary, size=40.",
        "- weak-average, density=0.70, warmup=10, g=0.002, trials=500, seed=42.",
        "- `all_to_all`: each grid couples to the mean local_mean of the other three grids.",
        "- `ring_1d`: A-B-C-D-A, each grid couples to the mean local_mean of its two ring neighbors.",
        "",
    ]

    for topology in TOPOLOGIES:
        lines.append(f"## {topology}")
        lines.append("")
        lines.append("| cluster | rate | within stop diff | between stop diff |")
        lines.append("| --- | ---: | ---: | ---: |")
        structure_subset = structure_df[structure_df["topology"] == topology]
        for _, row in structure_subset.iterrows():
            lines.append(
                "| {cluster} | {rate} | {within} | {between} |".format(
                    cluster=row["cluster_type"],
                    rate=render_metric(row["rate"]),
                    within=render_metric(row["within_cluster_stop_diff_mean"]),
                    between=render_metric(row["between_cluster_stop_diff_mean"]),
                )
            )
        lines.append("")
        lines.append("| contents | cluster | rate | within stop diff | between stop diff |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        content_subset = content_df[content_df["topology"] == topology].head(10)
        for _, row in content_subset.iterrows():
            lines.append(
                "| {signature} | {cluster} | {rate} | {within} | {between} |".format(
                    signature=row["cluster_signature"],
                    cluster=row["cluster_type"],
                    rate=render_metric(row["rate"]),
                    within=render_metric(row["within_cluster_stop_diff_mean"]),
                    between=render_metric(row["between_cluster_stop_diff_mean"]),
                )
            )
        lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved grid trial CSV: {grid_csv}")
    print(f"Saved trial cluster CSV: {trial_csv}")
    print(f"Saved structure summary CSV: {structure_csv}")
    print(f"Saved content summary CSV: {content_csv}")
    print(f"Saved report: {report_md}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid_records: list[dict[str, object]] = []
    trial_records: list[dict[str, object]] = []

    for topology in TOPOLOGIES:
        cond_grid_records, cond_trial_records = simulate_topology(topology, rng)
        grid_records.extend(cond_grid_records)
        trial_records.extend(cond_trial_records)

        cond_df = pd.DataFrame(cond_trial_records)
        structure_rate = cond_df["cluster_type"].value_counts(normalize=True)
        print(
            "topology={topology} 4+0={all_same:.3f} 3+1={three_one:.3f} 2+2={two_two:.3f} 2+1+1={two_one_one:.3f} all_diff={all_diff:.3f}".format(
                topology=topology,
                all_same=structure_rate.get("4+0", 0.0),
                three_one=structure_rate.get("3+1", 0.0),
                two_two=structure_rate.get("2+2", 0.0),
                two_one_one=structure_rate.get("2+1+1", 0.0),
                all_diff=structure_rate.get("all_diff", 0.0),
            ),
            flush=True,
        )

    grid_df = pd.DataFrame(grid_records)
    trial_df = pd.DataFrame(trial_records)
    structure_df, content_df = summarize(trial_df)
    save_outputs(grid_df, trial_df, structure_df, content_df)


if __name__ == "__main__":
    main()
