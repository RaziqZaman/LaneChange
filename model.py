#!/usr/bin/env python3
"""Generate a simple schematic depicting a two-lane road and surrounding vehicles."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

OUTPUT_PATH = Path("outputs/model.png")


def draw_vehicle(ax, center, heading_deg, label, color):
    length = 4.5
    width = 2.0
    heading = math.radians(heading_deg)

    dx = length / 2 * math.cos(heading)
    dy = length / 2 * math.sin(heading)

    corners = [
        (-length / 2, -width / 2),
        (-length / 2, width / 2),
        (length / 2, width / 2),
        (length / 2, -width / 2),
    ]

    rot_corners = []
    for x, y in corners:
        rx = x * math.cos(heading) - y * math.sin(heading)
        ry = x * math.sin(heading) + y * math.cos(heading)
        rot_corners.append((center[0] + rx, center[1] + ry))

    xs = [pt[0] for pt in rot_corners] + [rot_corners[0][0]]
    ys = [pt[1] for pt in rot_corners] + [rot_corners[0][1]]

    ax.fill(xs, ys, color=color, alpha=0.85, linewidth=1.5, edgecolor="black")

    arrow = FancyArrow(
        center[0],
        center[1],
        dx,
        dy,
        width=0.4,
        length_includes_head=True,
        head_width=1.2,
        head_length=1.2,
        color="white",
        alpha=0.8,
    )
    ax.add_patch(arrow)

    ax.text(
        center[0],
        center[1],
        label,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 10))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_facecolor("#eeeeee")

    lane_width = 3.5
    road_left = -lane_width
    road_right = lane_width

    ax.fill_between([-6, 6], -12, 12, color="#444444", alpha=0.9)
    ax.hlines([road_left, 0, road_right], -6, 6, color="#f4f6f7", linewidth=3, linestyles="solid", alpha=0.4)
    ax.hlines([road_left + lane_width / 2, road_right - lane_width / 2], -6, 6, color="#fefefe", linestyle="--", linewidth=2)

    ax.text(0, 11.5, "Downstream", ha="center", va="center", fontsize=14, color="#222222")
    ax.text(0, -11.5, "Upstream", ha="center", va="center", fontsize=14, color="#222222")

    vehicles = [
        ((-1.75, 0.5), 15, "SV", "#2c3e50"),
        ((-1.75, 4.5), 0, "LC", "#c0392b"),
        ((-1.75, -4.5), 0, "RC", "#8e44ad"),
        ((1.75, 6.5), 0, "LT", "#27ae60"),
        ((1.75, -6.5), 0, "RT", "#d35400"),
    ]

    for center, heading, label, color in vehicles:
        draw_vehicle(ax, center, heading, label, color)

    ax.axis("off")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved schematic to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
