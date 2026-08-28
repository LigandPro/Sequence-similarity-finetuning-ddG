"""The data-partition schematic (Fig. 1, panel c) — synthetic, no inputs.

Draws the pretrain / control / target / target-train partition as annotated Gaussian point
clouds. Every coordinate below is hand-placed to illustrate the split; none of it is measured.
Deterministic at seed 42.

    python figures/point_cloud.py

Writes ``figures/data.svg``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 — registers the "science" style used below
from matplotlib.patches import Ellipse

FIGURES = Path(__file__).resolve().parent
SEED = 42

# Fixes matplotlib's generated SVG element ids, so re-rendering gives a byte-identical file.
HASHSALT = "thermality"

SINGLE_CLUSTER_VARIANCE = (0.04, 0.02)

# (x, y, x_var, y_var, n_wt, wt_variance_range, label) for multi-WT clusters,
# (x, y, x_var, y_var, label) for a single WT at that exact position.
CLUSTERS = [
    (2.0, 4.2, 0.1, 0.5, 3, SINGLE_CLUSTER_VARIANCE, "PRETRAIN"),
    (2.3, 3.0, 0.5, 0.1, 2, SINGLE_CLUSTER_VARIANCE, "CONTROL"),
    (4.2, 2.8, *SINGLE_CLUSTER_VARIANCE, "TEST"),
    (4.5, 3.8, *SINGLE_CLUSTER_VARIANCE, "TEST"),
    (5.3, 2.8, *SINGLE_CLUSTER_VARIANCE, "FT"),
]

# (cx, cy, x_var, y_var, label, angle_deg) — the faint ellipses behind the point clouds.
CLUSTER_BACKINGS = [
    (2.0, 4.6, 0.25, 0.32, "PRETRAIN", -10),
    (2.3, 2.7, 0.3, 0.1, "CONTROL", -20),
    (4.7, 3.2, 0.4, 0.3, "TEST", 0),
]

LABEL_COLORS = {
    "PRETRAIN": "xkcd:blue grey",
    "TEST": "xkcd:bright blue",
    "FT": "xkcd:goldenrod",
    "CONTROL": "xkcd:red",
}


def generate_cluster_point_clouds(
    clusters, n_points_per_wt=100, variance_perturbation=0.1, inner_radius_factor=0.7, random_seed=None
):
    """Sample each wild type's points in an annulus, so every WT centre stays visible."""
    if random_seed is not None:
        np.random.seed(random_seed)

    all_points, cluster_labels, wt_info = [], [], []

    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) == 5:
            wt_x, wt_y, x_var, y_var, label = cluster
            if variance_perturbation > 0:
                x_var = x_var * (1 + np.random.uniform(-variance_perturbation, variance_perturbation))
                y_var = y_var * (1 + np.random.uniform(-variance_perturbation, variance_perturbation))
            subclusters = [(wt_x, wt_y, x_var, y_var)]
        else:
            cluster_x, cluster_y, cluster_x_var, cluster_y_var, n_wt, wt_variance_range, label = cluster
            cluster_x_radius = np.sqrt(cluster_x_var) * 2
            cluster_y_radius = np.sqrt(cluster_y_var) * 2
            subclusters = []
            for _ in range(n_wt):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 1)
                wt_x = cluster_x + radius * cluster_x_radius * np.cos(angle)
                wt_y = cluster_y + radius * cluster_y_radius * np.sin(angle)
                wt_x_variance = np.random.uniform(*wt_variance_range)
                wt_y_variance = np.random.uniform(*wt_variance_range)
                if variance_perturbation > 0:
                    x_var = wt_x_variance * (1 + np.random.uniform(-variance_perturbation, variance_perturbation))
                    y_var = wt_y_variance * (1 + np.random.uniform(-variance_perturbation, variance_perturbation))
                else:
                    x_var, y_var = wt_x_variance, wt_y_variance
                subclusters.append((wt_x, wt_y, x_var, y_var))

        for wt_x, wt_y, x_var, y_var in subclusters:
            outer_radius_factor = 2.5
            r_min_sq = inner_radius_factor**2
            r_max_sq = outer_radius_factor**2
            r = np.sqrt(r_min_sq + np.random.uniform(0, 1, n_points_per_wt) * (r_max_sq - r_min_sq))
            theta = np.random.uniform(0, 2 * np.pi, n_points_per_wt)
            x_points = wt_x + r * np.cos(theta) * np.sqrt(x_var)
            y_points = wt_y + r * np.sin(theta) * np.sqrt(y_var)

            all_points.append(np.column_stack([x_points, y_points]))
            cluster_labels.extend([cluster_id] * n_points_per_wt)
            wt_info.append({"cluster_id": cluster_id, "center": (wt_x, wt_y), "label": label})

    return np.vstack(all_points), np.array(cluster_labels), wt_info


def main() -> None:
    plt.style.use(["science", "no-latex"])
    plt.rcParams["svg.hashsalt"] = HASHSALT
    np.random.seed(SEED)

    points, cluster_labels, wt_info = generate_cluster_point_clouds(
        clusters=CLUSTERS, n_points_per_wt=100, variance_perturbation=0.15, random_seed=SEED
    )

    _, ax = plt.subplots(figsize=(12, 10))

    for cx, cy, x_var, y_var, label, angle in CLUSTER_BACKINGS:
        color = LABEL_COLORS[label]
        ax.add_patch(
            Ellipse(
                (cx, cy), width=4 * np.sqrt(x_var), height=4 * np.sqrt(y_var), angle=angle,
                facecolor=color, alpha=0.09, edgecolor=color, linestyle="--", zorder=0,
            )
        )

    in_legend: set[str] = set()
    for cluster_id, cluster in enumerate(CLUSTERS):
        label = cluster[-1]
        color = LABEL_COLORS[label]
        mask = cluster_labels == cluster_id
        plt.scatter(
            points[mask, 0], points[mask, 1], alpha=0.5, s=3, c=color, zorder=2,
            label=label if label not in in_legend else None,
        )
        in_legend.add(label)
        for wt in wt_info:
            if wt["cluster_id"] == cluster_id:
                plt.scatter(*wt["center"], s=40, c=color, edgecolors="black", linewidths=1, zorder=5)

    plt.legend(frameon=True, fancybox=True, edgecolor="grey", shadow=True, fontsize=20, markerscale=3)
    ax.axis("off")
    plt.ylim(1.9, 5.9)
    plt.tight_layout()
    plt.gcf().set_size_inches(5, 10)
    plt.savefig(FIGURES / "data.svg", format="svg", metadata={"Date": None})
    plt.close()
    print(f"wrote data.svg to {FIGURES}")


if __name__ == "__main__":
    main()
