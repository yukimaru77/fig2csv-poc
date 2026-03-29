from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("data/samples")
    out.mkdir(parents=True, exist_ok=True)

    x = np.arange(0, 6)
    y = np.array([10, 14, 13, 18, 21, 25])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_title("Sample Line Chart for fig2csv PoC")
    ax.grid(True, alpha=0.3)

    path = out / "sample_chart.png"
    fig.savefig(path, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
