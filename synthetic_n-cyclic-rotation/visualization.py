"""Combine saved n-element GAN trajectories into one publication figure.

Run after script.sh (or after all 20 independent main.py jobs):
    python3 visualization.py --dir_output ./result

Only NumPy and Matplotlib are required; PyTorch and a GPU are unnecessary.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


ROWS = (2, 3, 4, 5)
HEADERS = ("P=0", "p=1/3", "p=2/3", "p=1", "(ours) P=1")
COLORS = ("#333333", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def ordinal(value):
    if 11 <= value % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def probabilities(n):
    # Match the values and filename rounding used by script.sh.
    return (0.0, (n - 1) / (3 * n), 2 * (n - 1) / (3 * n),
            (n - 1) / n, 1.0)


def load_means(data_dir, n, probability):
    stem = f"fake_process_n,{n},P,{probability:.02f}"
    candidates = (data_dir / f"{stem}.npz", data_dir / f"{stem}.png.npz")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing result for n={n}, P={probability:.6g}; expected "
            + " or ".join(str(item) for item in candidates)
        )
    with np.load(path, allow_pickle=False) as data:
        # Old runs used positional arrays; new runs use descriptive keys.
        means = np.asarray(data["means" if "means" in data else "arr_0"])
    if means.ndim != 2 or means.shape[0] != n or means.shape[1] < 2:
        raise ValueError(f"Unexpected trajectory shape {means.shape} in {path}")
    return means


def make_figure(dir_output, output, font_size, legend_size, width, height, dpi):
    data_dir = dir_output / "npz"
    # Check every run before creating a partial figure.
    trajectories = {(n, col): load_means(data_dir, n, p)
                    for n in ROWS
                    for col, p in enumerate(probabilities(n))}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": font_size,
                         "axes.labelsize": font_size, "xtick.labelsize": font_size - 2,
                         "ytick.labelsize": font_size - 2})
    fig, axes = plt.subplots(len(ROWS), len(HEADERS), figsize=(width, height),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.08,
                        wspace=0.17, hspace=0.23)

    for row, n in enumerate(ROWS):
        for col in range(len(HEADERS)):
            ax = axes[row, col]
            means = trajectories[(n, col)]
            x = np.arange(1, means.shape[1] + 1)
            for dim, y in enumerate(means):
                ax.plot(x, y, color=COLORS[dim], linewidth=2.0 if dim == 0 else 1.5,
                        linestyle="-" if dim == 0 else "--",
                        label=f"{ordinal(dim + 1)}-dim.",
                        zorder=n - dim)
            ax.set_xscale("log")
            ax.set_xlim(1, max(10_000, means.shape[1]))
            # Draw vertical gridlines at 10, 100, and 1000 without crowding the labels.
            ax.xaxis.set_major_locator(FixedLocator((1, 10, 100, 1000, 10_000)))
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _: {1: r"$10^0$", 100: r"$10^2$",
                                  10_000: r"$10^4$"}.get(value, "")))
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
            ax.set_axisbelow(True)
            ax.grid(True, which="major", color="#858b92", linewidth=0.75, alpha=0.65)
            ax.tick_params(axis="both", which="both", length=2.5,
                           labelbottom=row == len(ROWS) - 1, labelleft=col == 0)
            if row == len(ROWS) - 1:
                ax.set_xlabel("iteration")
            if col == 0:
                ax.set_ylabel("intensity")
                ax.legend(loc="center right", fontsize=legend_size, framealpha=0.94,
                          edgecolor="#cccccc", borderpad=0.3,
                          labelspacing=0.25, handlelength=1.8)

    # Position headers relative to the actual subplot bounds.
    for col, header in enumerate(HEADERS):
        bounds = axes[0, col].get_position()
        fig.text((bounds.x0 + bounds.x1) / 2, 0.962, header,
                 ha="center", va="center", fontsize=font_size + 4)
    for row, n in enumerate(ROWS):
        bounds = axes[row, 0].get_position()
        fig.text(0.025, (bounds.y0 + bounds.y1) / 2, f"n={n}",
                 ha="center", va="center", rotation=90, fontsize=font_size + 4)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir_output", type=Path, default=Path("./result"),
                        help="Training output directory containing npz/")
    parser.add_argument("--output", type=Path, default=Path("n-element-GAN.png"),
                        help="Combined figure path; .pdf and .svg also work")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--legend-size", type=int, default=14)
    parser.add_argument("--width", type=float, default=16.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=10.5, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    make_figure(args.dir_output, args.output, args.font_size, args.legend_size,
                args.width, args.height, args.dpi)


if __name__ == "__main__":
    main()
