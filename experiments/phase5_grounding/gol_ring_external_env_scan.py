from __future__ import annotations

from collections import Counter, deque
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
G_INT = 0.002
WARMUP = 10
G_EXT_VALUES = (0.0, 0.001, 0.002, 0.005, 0.010)
GRID_IDS = ("A", "B", "C", "D")
NEIGHBORS = {
    "A": ("B", "D"),
    "B": ("A", "C"),
    "C": ("B", "D"),
    "D": ("C", "A"),
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
    return 0.0 if step <= WARMUP else G_INT


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


def build_delta_environment(rng: np.random.Generator) -> np.ndarray:
    while True:
        grid = (rng.random((SIZE, SIZE)) < DENSITY).astype(np.uint8)
        prev_grid = grid.copy()
        history: deque[bytes] = deque(maxlen=OSC_MAX_PERIOD)
        current = grid.copy()
        for _ in range(1, MAX_STEPS + 1):
            new_grid = gol_step(current)
            label, stopped = classify_label(current, new_grid, prev_grid, history)
            if stopped:
                if label == "delta":
                    return new_grid.copy()
                break
            prev_grid = current.copy()
            current = new_grid


def coupled_step(
    grids: dict[str, np.ndarray],
    active: dict[str, bool],
    g_int: float,
    g_ext: float,
    env_mean: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    evolved = {name: gol_step(grids[name]) if active[name] else grids[name].copy() for name in GRID_IDS}
    local_means = {name: local_mean(grid) for name, grid in evolved.items()}
    new_grids: dict[str, np.ndarray] = {}
    for name in GRID_IDS:
        if not active[name]:
            new_grids[name] = grids[name].copy()
            continue
        neighbor_mean = np.mean([local_means[other] for other in NEIGHBORS[name]], axis=0)
        prob = (1.0 - g_int - (g_ext if name == "A" else 0.0)) * evolved[name].astype(float) + g_int * neighbor_mean
        if name == "A" and g_ext > 0.0:
            prob += g_ext * env_mean
        prob = np.clip(prob, 0.0, 1.0)
        new_grids[name] = (rng.random(evolved[name].shape) < prob).astype(np.uint8)
    return new_grids


def simulate_condition(g_ext: float, rng: np.random.Generator) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_records: list[dict[str, object]] = []
    trial_records: list[dict[str, object]] = []

    for trial in range(1, N_TRIALS + 1):
        env_grid = build_delta_environment(rng)
        env_mean = local_mean(env_grid)
        grids = {name: (rng.random((SIZE, SIZE)) < DENSITY).astype(np.uint8) for name in GRID_IDS}
        prev_grids = {name: grid.copy() for name, grid in grids.items()}
        recent_states = {name: deque(maxlen=OSC_MAX_PERIOD) for name in GRID_IDS}
        labels = {name: "max" for name in GRID_IDS}
        stop_steps = {name: MAX_STEPS for name in GRID_IDS}
        active = {name: True for name in GRID_IDS}

        for step in range(1, MAX_STEPS + 1):
            current_g_int = choose_g(step)
            current_g_ext = 0.0 if step <= WARMUP else g_ext
            new_grids = coupled_step(grids, active, current_g_int, current_g_ext, env_mean, rng)

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
                    "g_ext": g_ext,
                    "trial": trial,
                    "grid_id": name,
                    "label": labels[name],
                    "stop_step": stop_steps[name],
                }
            )

        osc_nodes = [name for name in GRID_IDS if labels[name] == "osc"]
        isolated_osc_node = osc_nodes[0] if len(osc_nodes) == 1 else None
        trial_record = {
            "g_ext": g_ext,
            "trial": trial,
            "cluster_type": cluster_type(list(labels.values())),
            "cluster_signature": cluster_signature(labels),
            "A_label": labels["A"],
            "isolated_osc_node": isolated_osc_node,
        }
        for name in GRID_IDS:
            trial_record[f"label_{name}"] = labels[name]
            trial_record[f"stop_{name}"] = stop_steps[name]
        trial_records.append(trial_record)

    return grid_records, trial_records


def summarize(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structure_rows: list[dict[str, object]] = []
    a_mode_rows: list[dict[str, object]] = []
    osc_position_rows: list[dict[str, object]] = []

    for g_ext in G_EXT_VALUES:
        subset = trial_df[np.isclose(trial_df["g_ext"], g_ext)]
        total = len(subset)
        for cluster_name in ("4+0", "3+1", "2+2", "2+1+1", "all_diff"):
            count = int((subset["cluster_type"] == cluster_name).sum())
            structure_rows.append(
                {
                    "g_ext": g_ext,
                    "cluster_type": cluster_name,
                    "count": count,
                    "ratio": float(count / total) if total else 0.0,
                }
            )

        a_counts = subset["A_label"].value_counts()
        for label in ("osc", "delta", "fixed", "max"):
            count = int(a_counts.get(label, 0))
            a_mode_rows.append(
                {
                    "g_ext": g_ext,
                    "A_label": label,
                    "count": count,
                    "ratio": float(count / total) if total else 0.0,
                }
            )

        subset_3p1 = subset[subset["cluster_type"] == "3+1"]
        total_3p1 = len(subset_3p1)
        pos_counts = subset_3p1["isolated_osc_node"].value_counts()
        for node in GRID_IDS:
            count = int(pos_counts.get(node, 0))
            osc_position_rows.append(
                {
                    "g_ext": g_ext,
                    "isolated_osc_node": node,
                    "count": count,
                    "ratio_within_3plus1": float(count / total_3p1) if total_3p1 else 0.0,
                }
            )

    return (
        pd.DataFrame(structure_rows).sort_values(["g_ext", "cluster_type"]).reset_index(drop=True),
        pd.DataFrame(a_mode_rows).sort_values(["g_ext", "A_label"]).reset_index(drop=True),
        pd.DataFrame(osc_position_rows).sort_values(["g_ext", "isolated_osc_node"]).reset_index(drop=True),
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
    a_mode_df: pd.DataFrame,
    osc_position_df: pd.DataFrame,
) -> None:
    grid_csv = ROOT / "gol_ring_external_env_trials.csv"
    trial_csv = ROOT / "gol_ring_external_env_clusters.csv"
    structure_csv = ROOT / "gol_ring_external_env_structure_summary.csv"
    a_mode_csv = ROOT / "gol_ring_external_env_A_mode_summary.csv"
    osc_pos_csv = ROOT / "gol_ring_external_env_osc_position_summary.csv"
    report_md = ROOT / "gol_ring_external_env_report.md"

    grid_df.to_csv(grid_csv, index=False, encoding="utf-8")
    trial_df.to_csv(trial_csv, index=False, encoding="utf-8")
    structure_df.to_csv(structure_csv, index=False, encoding="utf-8")
    a_mode_df.to_csv(a_mode_csv, index=False, encoding="utf-8")
    osc_position_df.to_csv(osc_pos_csv, index=False, encoding="utf-8")

    lines = [
        "# ring_1d + external delta environment scan",
        "",
        "- Internal topology: A-B-C-D-A ring with weak-average coupling.",
        "- External node E is a fixed delta-environment template regenerated per trial and coupled one-way to A only.",
        f"- density={DENSITY}, warmup={WARMUP}, trials={N_TRIALS}, g_int={G_INT}, g_ext sweep={G_EXT_VALUES}, seed={SEED}",
        "",
    ]

    for g_ext in G_EXT_VALUES:
        lines.append(f"## g_ext={g_ext:.3f}")
        lines.append("")
        lines.append("| cluster | count | ratio |")
        lines.append("| --- | ---: | ---: |")
        for _, row in structure_df[np.isclose(structure_df["g_ext"], g_ext)].iterrows():
            lines.append(
                "| {cluster} | {count} | {ratio} |".format(
                    cluster=row["cluster_type"],
                    count=int(row["count"]),
                    ratio=render_metric(row["ratio"]),
                )
            )
        lines.append("")
        lines.append("| A mode | count | ratio |")
        lines.append("| --- | ---: | ---: |")
        for _, row in a_mode_df[np.isclose(a_mode_df["g_ext"], g_ext)].iterrows():
            lines.append(
                "| {mode} | {count} | {ratio} |".format(
                    mode=row["A_label"],
                    count=int(row["count"]),
                    ratio=render_metric(row["ratio"]),
                )
            )
        lines.append("")
        lines.append("| isolated osc node in 3+1 | count | ratio within 3+1 |")
        lines.append("| --- | ---: | ---: |")
        for _, row in osc_position_df[np.isclose(osc_position_df["g_ext"], g_ext)].iterrows():
            lines.append(
                "| {node} | {count} | {ratio} |".format(
                    node=row["isolated_osc_node"],
                    count=int(row["count"]),
                    ratio=render_metric(row["ratio_within_3plus1"]),
                )
            )
        lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved grid trial CSV: {grid_csv}")
    print(f"Saved trial cluster CSV: {trial_csv}")
    print(f"Saved structure summary CSV: {structure_csv}")
    print(f"Saved A-mode summary CSV: {a_mode_csv}")
    print(f"Saved osc-position summary CSV: {osc_pos_csv}")
    print(f"Saved report: {report_md}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid_records: list[dict[str, object]] = []
    trial_records: list[dict[str, object]] = []

    for g_ext in G_EXT_VALUES:
        cond_grid_records, cond_trial_records = simulate_condition(g_ext, rng)
        grid_records.extend(cond_grid_records)
        trial_records.extend(cond_trial_records)

        cond_df = pd.DataFrame(cond_trial_records)
        structure_rate = cond_df["cluster_type"].value_counts(normalize=True)
        print(
            "g_ext={g_ext:.3f} 4+0={all_same:.3f} 3+1={three_one:.3f} 2+2={two_two:.3f} 2+1+1={two_one_one:.3f}".format(
                g_ext=g_ext,
                all_same=structure_rate.get("4+0", 0.0),
                three_one=structure_rate.get("3+1", 0.0),
                two_two=structure_rate.get("2+2", 0.0),
                two_one_one=structure_rate.get("2+1+1", 0.0),
            ),
            flush=True,
        )

    grid_df = pd.DataFrame(grid_records)
    trial_df = pd.DataFrame(trial_records)
    structure_df, a_mode_df, osc_position_df = summarize(trial_df)
    save_outputs(grid_df, trial_df, structure_df, a_mode_df, osc_position_df)


if __name__ == "__main__":
    main()
